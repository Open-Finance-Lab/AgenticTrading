"""Execution and ledger coverage for the environment-backed Platform lane."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import httpx
import pytest
from openai import _base_client as openai_base_client

from dashboard.backend.domain.credits.repository import CreditsStore
from dashboard.backend.domain.credits.service import CreditsService
from dashboard.backend.domain.model_providers.repository import ModelProviderStore
from dashboard.backend.domain.model_providers.service import ModelProviderService
from dashboard.backend.infrastructure.llm.execution.adapters.base import (
    AdapterResponse,
    ProviderExecutionError,
)
from dashboard.backend.infrastructure.llm.execution.adapters.openai import (
    OpenAICompatibleAdapter,
)
from dashboard.backend.infrastructure.llm.execution.errors import (
    ExecutionErrorCategory,
    LLMExecutionError,
    RetryHint,
)
from dashboard.backend.infrastructure.llm.execution.models import (
    BillingEvidence,
    BillingMode,
    LLMExecutionRequest,
    LLMExecutionResult,
    LLMMessage,
    LLMUsage,
    PricingSnapshot,
    UsagePolicy,
)
from dashboard.backend.infrastructure.llm.execution.service import LLMExecutionService
from dashboard.backend.infrastructure.llm.execution import service as execution_service_module


USER_ID = 1
ADMIN_ID = 2
MODEL_ID = "openai/gpt-5.5"


class FakeExecutionAdapter:
    def __init__(self, usage: LLMUsage | None):
        self.usage = usage
        self.secrets: list[str] = []

    def complete(self, request, credential, provider):
        self.secrets.append(credential.secret)
        return AdapterResponse(
            text="BUY",
            model_id=request.model_id,
            usage=self.usage,
        )


class ScriptedExecutionAdapter:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def complete(self, request, credential, provider):
        self.calls.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _seed_users(path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO users (
                id, email, display_name, password_hash, role, created_at
            ) VALUES (?, ?, ?, 'unused', ?, '2026-08-25T00:00:00+00:00')
            """,
            [
                (USER_ID, "env-platform-user@example.test", "User", "user"),
                (ADMIN_ID, "env-platform-admin@example.test", "Admin", "admin"),
            ],
        )


def _enable_openrouter(store: ModelProviderStore) -> None:
    provider = store.get_provider("openrouter")
    assert provider is not None
    store.upsert_provider(
        provider_id="openrouter",
        display_name=provider["display_name"],
        adapter_type=provider["adapter_type"],
        approved_base_url=provider["approved_base_url"],
        capabilities=provider["capabilities"],
        byok_enabled=provider["byok_enabled"],
        platform_enabled=True,
        status=provider["status"],
    )


def _seed_balances(store: CreditsStore) -> None:
    store.fund_grant_pool(
        pool_id="default",
        amount_micro=100_000,
        operation_id="env_test_fund",
        idempotency_key="env_test_fund_request",
        request_digest="env_test_fund_digest",
        actor_user_id=ADMIN_ID,
        source="test",
        reason="Seed the test Grant balance.",
    )
    store.assign_grant(
        user_id=USER_ID,
        pool_id="default",
        amount_micro=100_000,
        operation_id="env_test_assign",
        idempotency_key="env_test_assign_request",
        request_digest="env_test_assign_digest",
        actor_user_id=ADMIN_ID,
        source="test",
        reason="Assign the test Grant balance.",
    )
    order = store.create_or_get_order(
        order_id="env_test_purchase",
        user_id=USER_ID,
        client_request_id="env_test_purchase_request",
        amount_usd_cents=100,
        credits_micro=1_000_000,
    )
    store.attach_checkout_session(
        order["id"], checkout_session_id="env_test_checkout"
    )
    settled = store.settle_paid_checkout(
        event_id="env_test_event",
        event_type="checkout.session.completed",
        livemode=False,
        object_id="env_test_checkout",
        payload_sha256="e" * 64,
        order_id=order["id"],
        checkout_session_id="env_test_checkout",
        payment_intent_id="env_test_payment",
        currency="usd",
        amount_usd_cents=100,
    )
    assert settled["outcome"] == "processed"


def _request(run_id: str) -> LLMExecutionRequest:
    return LLMExecutionRequest(
        user_id=USER_ID,
        run_id=run_id,
        call_index=0,
        billing_mode=BillingMode.PLATFORM_CREDITS,
        provider_id="openrouter",
        model_id=MODEL_ID,
        system_message="Return one trading decision.",
        messages=(LLMMessage(role="user", content="Analyze the market."),),
        usage_policy=UsagePolicy(max_output_tokens=100),
    )


def _failover_request(run_id: str) -> LLMExecutionRequest:
    """A platform request carrying the route's OpenRouter -> CommonStack tuple.

    The worker never widens a lone candidate, so failover tests hand over the
    ordered tuple exactly as the route would.
    """
    return _request(run_id).model_copy(
        update={"provider_ids": ("openrouter", "commonstack")}
    )


def _execution_service(
    tmp_path, monkeypatch, adapter, *, adapters=None, sleep=None, clock=None
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-execution-test-abcd")
    provider_store = ModelProviderStore(tmp_path / "providers.db")
    _enable_openrouter(provider_store)
    credits_path = tmp_path / "credits.db"
    _seed_users(credits_path)
    credits_store = CreditsStore(credits_path)
    _seed_balances(credits_store)
    credits_service = CreditsService(store=credits_store)
    def snapshot_for(model_id: str, provider_id: str) -> PricingSnapshot:
        return PricingSnapshot(
            provider_id=provider_id,
            model_id=model_id,
            input_usd_per_million_tokens=1000.0,
            output_usd_per_million_tokens=1000.0,
            source_version="test-pricing",
        )

    def resolve_adapter(provider):
        return adapters[provider.provider_id] if adapters is not None else adapter

    timing = {}
    if sleep is not None:
        timing["sleep"] = sleep
    if clock is not None:
        timing["clock"] = clock
    service = LLMExecutionService(
        providers=ModelProviderService(store=provider_store),
        credits=credits_service,
        adapter_resolver=resolve_adapter,
        pricing_snapshot_factory=snapshot_for,
        **timing,
    )
    return service, credits_store


def test_platform_execution_uses_env_key_and_debits_grant_before_purchased(
    tmp_path, monkeypatch
):
    adapter = FakeExecutionAdapter(LLMUsage(input_tokens=100, output_tokens=100))
    service, credits_store = _execution_service(tmp_path, monkeypatch, adapter)

    result = service.execute(_request("env-platform-run"))

    assert adapter.secrets == ["sk-or-execution-test-abcd"]
    assert result.credential_id is None
    assert result.credential_key_last_four == "abcd"
    assert result.billing.billing_source == BillingMode.PLATFORM_CREDITS
    assert result.billing.debited_credits_micro == 200_000
    balance = credits_store.get_balance_projection(USER_ID)
    assert balance["grant_available_micro"] == 0
    assert balance["purchased_available_micro"] == 900_000


def test_platform_execution_settles_covered_reservation_overage(
    tmp_path, monkeypatch
):
    adapter = FakeExecutionAdapter(LLMUsage(input_tokens=100, output_tokens=600))
    service, credits_store = _execution_service(tmp_path, monkeypatch, adapter)

    result = service.execute(_request("env-platform-covered-overage"))

    assert result.billing.provider_cost_credits_micro == 700_000
    assert result.billing.debited_credits_micro == 700_000
    assert result.billing.outstanding_credits_micro == 0
    assert credits_store.get_account_billing_state(USER_ID) == {
        "account_status": "active",
        "restriction_reason": None,
        "outstanding_credits_micro": 0,
    }


def test_platform_execution_emits_bucketed_resource_evidence(
    tmp_path,
    monkeypatch,
):
    adapter = FakeExecutionAdapter(LLMUsage(input_tokens=100, output_tokens=100))
    service, _credits_store = _execution_service(tmp_path, monkeypatch, adapter)
    events = []
    monkeypatch.setattr(
        execution_service_module.analytics_instrumentation,
        "emit_resource_event",
        lambda **kwargs: events.append(kwargs),
    )

    service.execute(_request("analytics-platform-run"))

    names = [event["event_name"] for event in events]
    assert names == [
        "credits_reserved",
        "credits_reserved",
        "credits_settled",
        "credits_settled",
        "credits_refunded",
        "model_usage_recorded",
    ]
    usage = events[-1]
    assert usage["billing_mode"] == "platform_credits"
    assert usage["properties"] == {
        "input_tokens": 100,
        "output_tokens": 100,
        "cost_micro_usd": 200_000,
    }
    assert "sk-or-execution-test-abcd" not in repr(events)


def test_platform_quota_error_retries_once_through_commonstack(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("COMMONSTACK_API_KEY", "cs-fake-failover-abcd")
    primary_adapter = ScriptedExecutionAdapter(
        [
            ProviderExecutionError(
                ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED
            )
        ]
    )
    fallback_adapter = ScriptedExecutionAdapter(
        [
            AdapterResponse(
                text="BUY",
                model_id="qwen/qwen3.7-plus",
                usage=LLMUsage(input_tokens=40, output_tokens=20),
                finish_reason="stop",
            )
        ]
    )
    service, credits_store = _execution_service(
        tmp_path,
        monkeypatch,
        primary_adapter,
        adapters={
            "openrouter": primary_adapter,
            "commonstack": fallback_adapter,
        },
    )
    assert service.providers.store.get_provider("commonstack")["platform_enabled"] is True
    request = _failover_request("failover-run").model_copy(
        update={
            "model_id": "qwen/qwen3.7-plus",
            "reasoning_effort": "high",
            "temperature": 0.2,
        }
    )

    result = service.execute(request)

    assert result.provider_id == "commonstack"
    assert result.requested_provider_id == "openrouter"
    assert [call.provider_id for call in primary_adapter.calls] == ["openrouter"]
    assert [call.provider_id for call in fallback_adapter.calls] == ["commonstack"]
    primary_request = primary_adapter.calls[0]
    fallback_request = fallback_adapter.calls[0]
    assert fallback_request.model_id == primary_request.model_id
    assert fallback_request.system_message == primary_request.system_message
    assert fallback_request.messages == primary_request.messages
    assert fallback_request.usage_policy == primary_request.usage_policy
    assert fallback_request.temperature == primary_request.temperature
    assert fallback_request.reasoning_effort == primary_request.reasoning_effort
    with credits_store._get_connection() as connection:
        rows = connection.execute(
            "SELECT attempt_index, provider_id, status "
            "FROM credit_llm_reservations WHERE run_id = ? "
            "ORDER BY attempt_index",
            ("failover-run",),
        ).fetchall()
    assert [tuple(row) for row in rows] == [
        (0, "openrouter", "released"),
        (1, "commonstack", "settled"),
    ]


@pytest.mark.parametrize(
    "category",
    [
        ExecutionErrorCategory.RESPONSE_INVALID,
        ExecutionErrorCategory.USAGE_UNAVAILABLE,
        ExecutionErrorCategory.BILLING_FAILED,
        ExecutionErrorCategory.ACCOUNT_RESTRICTED,
    ],
)
def test_non_quota_platform_failures_do_not_fail_over(
    tmp_path, monkeypatch, category
):
    monkeypatch.setenv("COMMONSTACK_API_KEY", "cs-fake-unused-abcd")
    primary = ScriptedExecutionAdapter([ProviderExecutionError(category)])
    fallback = ScriptedExecutionAdapter([])
    service, _store = _execution_service(
        tmp_path,
        monkeypatch,
        primary,
        adapters={"openrouter": primary, "commonstack": fallback},
    )

    with pytest.raises(LLMExecutionError) as exc_info:
        service.execute(_failover_request(f"no-failover-{category.value}"))

    assert exc_info.value.category is category
    assert fallback.calls == []


@pytest.mark.parametrize(
    "category",
    [
        ExecutionErrorCategory.PROVIDER_TIMEOUT,
        ExecutionErrorCategory.PROVIDER_UNAVAILABLE,
        ExecutionErrorCategory.CREDENTIAL_INVALID,
    ],
)
def test_provider_selection_failures_fail_over_to_commonstack(
    tmp_path, monkeypatch, category
):
    monkeypatch.setenv("COMMONSTACK_API_KEY", "cs-fake-failover-abcd")
    primary = ScriptedExecutionAdapter([ProviderExecutionError(category)])
    fallback = ScriptedExecutionAdapter(
        [
            AdapterResponse(
                text="BUY",
                model_id=MODEL_ID,
                usage=LLMUsage(input_tokens=10, output_tokens=5),
            )
        ]
    )
    service, _store = _execution_service(
        tmp_path,
        monkeypatch,
        primary,
        adapters={"openrouter": primary, "commonstack": fallback},
    )

    result = service.execute(_failover_request(f"failover-{category.value}"))

    assert result.provider_id == "commonstack"
    assert result.requested_provider_id == "openrouter"
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1


def test_byok_quota_error_never_uses_platform_fallback(tmp_path, monkeypatch):
    primary = ScriptedExecutionAdapter(
        [
            ProviderExecutionError(
                ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED
            )
        ]
    )
    fallback = ScriptedExecutionAdapter([])
    service, _store = _execution_service(
        tmp_path,
        monkeypatch,
        primary,
        adapters={"openrouter": primary, "commonstack": fallback},
    )
    calls = []

    def fail_once(request, **_kwargs):
        calls.append(request.provider_id)
        raise LLMExecutionError(ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED)

    monkeypatch.setattr(service, "_execute_once", fail_once)
    with pytest.raises(LLMExecutionError) as exc_info:
        service.execute(
            _request("byok-no-fallback").model_copy(
                update={"billing_mode": BillingMode.BYOK}
            )
        )
    assert exc_info.value.category is ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED
    assert calls == ["openrouter"]
    assert fallback.calls == []


def test_fallback_failure_returns_commonstack_safe_category(tmp_path, monkeypatch):
    monkeypatch.setenv("COMMONSTACK_API_KEY", "cs-fake-timeout-abcd")
    primary = ScriptedExecutionAdapter(
        [
            ProviderExecutionError(
                ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED
            )
        ]
    )
    fallback = ScriptedExecutionAdapter(
        [ProviderExecutionError(ExecutionErrorCategory.PROVIDER_TIMEOUT)]
    )
    service, _store = _execution_service(
        tmp_path,
        monkeypatch,
        primary,
        adapters={"openrouter": primary, "commonstack": fallback},
    )
    with pytest.raises(LLMExecutionError) as exc_info:
        service.execute(_failover_request("dual-failure"))
    assert exc_info.value.category is ExecutionErrorCategory.PROVIDER_TIMEOUT
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1


def test_missing_commonstack_route_preserves_primary_quota_error(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("COMMONSTACK_API_KEY", raising=False)
    primary = ScriptedExecutionAdapter(
        [
            ProviderExecutionError(
                ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED
            )
        ]
    )
    fallback = ScriptedExecutionAdapter([])
    service, _store = _execution_service(
        tmp_path,
        monkeypatch,
        primary,
        adapters={"openrouter": primary, "commonstack": fallback},
    )
    with pytest.raises(LLMExecutionError) as exc_info:
        service.execute(_request("missing-fallback"))
    assert exc_info.value.category is ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED
    assert fallback.calls == []


def test_primary_release_failure_aborts_before_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("COMMONSTACK_API_KEY", "cs-fake-unused-abcd")
    primary = ScriptedExecutionAdapter(
        [
            ProviderExecutionError(
                ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED
            )
        ]
    )
    fallback = ScriptedExecutionAdapter([])
    service, _store = _execution_service(
        tmp_path,
        monkeypatch,
        primary,
        adapters={"openrouter": primary, "commonstack": fallback},
    )

    def fail_release(*_args, **_kwargs):
        raise RuntimeError("synthetic release failure")

    monkeypatch.setattr(service.credits, "release_llm_credits", fail_release)
    with pytest.raises(LLMExecutionError) as exc_info:
        service.execute(_failover_request("release-failure"))
    assert exc_info.value.category is ExecutionErrorCategory.BILLING_FAILED
    assert fallback.calls == []


def test_two_quota_failures_stop_after_commonstack(tmp_path, monkeypatch):
    monkeypatch.setenv("COMMONSTACK_API_KEY", "cs-fake-double-quota-abcd")
    primary = ScriptedExecutionAdapter(
        [
            ProviderExecutionError(
                ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED
            )
        ]
    )
    fallback = ScriptedExecutionAdapter(
        [
            ProviderExecutionError(
                ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED
            )
        ]
    )
    service, _store = _execution_service(
        tmp_path,
        monkeypatch,
        primary,
        adapters={"openrouter": primary, "commonstack": fallback},
    )
    with pytest.raises(LLMExecutionError) as exc_info:
        service.execute(_failover_request("double-quota"))
    assert exc_info.value.category is ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1


def test_byok_usage_reports_tokens_with_zero_atl_cost(monkeypatch):
    events = []
    monkeypatch.setattr(
        execution_service_module.analytics_instrumentation,
        "emit_resource_event",
        lambda **kwargs: events.append(kwargs),
    )
    request = _request("analytics-byok-run").model_copy(
        update={"billing_mode": BillingMode.BYOK}
    )
    result = LLMExecutionResult(
        text="BUY",
        provider_id="openrouter",
        model_id=MODEL_ID,
        usage=LLMUsage(input_tokens=50, output_tokens=25),
        billing=BillingEvidence(
            billing_source=BillingMode.BYOK,
            usage_authority="not_billable_by_atl",
            provider_cost_usd=99.0,
        ),
    )

    LLMExecutionService._emit_model_usage(request, result)

    assert events[0]["billing_mode"] == "byok"
    assert events[0]["properties"] == {
        "input_tokens": 50,
        "output_tokens": 25,
        "cost_micro_usd": 0,
    }


def test_byok_usage_records_the_platform_price_estimate(monkeypatch):
    """BYOK debits nothing, but the lane must still be expressible in Credits:
    the event carries the platform list-price estimate of the same tokens,
    never the provider cost the user paid on their own key."""
    events = []
    monkeypatch.setattr(
        execution_service_module.analytics_instrumentation,
        "emit_resource_event",
        lambda **kwargs: events.append(kwargs),
    )
    request = _request("analytics-byok-estimate-run").model_copy(
        update={"billing_mode": BillingMode.BYOK}
    )
    result = LLMExecutionResult(
        text="BUY",
        provider_id="openrouter",
        model_id=MODEL_ID,
        usage=LLMUsage(input_tokens=50, output_tokens=25),
        billing=BillingEvidence(
            billing_source=BillingMode.BYOK,
            usage_authority="provider_usage_pricing_snapshot",
            provider_cost_usd=99.0,
            estimated_cost_usd=0.42,
        ),
    )

    LLMExecutionService._emit_model_usage(request, result)

    assert events[0]["billing_mode"] == "byok"
    assert events[0]["properties"] == {
        "input_tokens": 50,
        "output_tokens": 25,
        "cost_micro_usd": 420_000,
    }


def test_execution_failure_emits_only_safe_error_category(
    tmp_path,
    monkeypatch,
):
    adapter = FakeExecutionAdapter(None)
    service, _credits_store = _execution_service(tmp_path, monkeypatch, adapter)
    errors = []
    monkeypatch.setattr(
        execution_service_module.analytics_instrumentation,
        "emit_safe_error_event",
        lambda **kwargs: errors.append(kwargs),
    )

    with pytest.raises(LLMExecutionError):
        service.execute(_request("analytics-failed-run"))

    assert errors[0]["error_category"] == "internal_error"
    assert "Analyze the market" not in repr(errors[0])
    assert "sk-or-execution-test-abcd" not in repr(errors[0])


def test_platform_execution_releases_reservation_when_usage_is_missing(
    tmp_path, monkeypatch
):
    adapter = FakeExecutionAdapter(None)
    service, credits_store = _execution_service(tmp_path, monkeypatch, adapter)

    with pytest.raises(LLMExecutionError) as exc_info:
        service.execute(_request("env-platform-failed-run"))

    assert exc_info.value.category is ExecutionErrorCategory.USAGE_UNAVAILABLE
    assert credits_store.get_balance_projection(USER_ID) == {
        "grant_committed_micro": 100_000,
        "purchased_committed_micro": 1_000_000,
        "grant_available_micro": 100_000,
        "purchased_available_micro": 1_000_000,
        "total_available_micro": 1_100_000,
    }
    with sqlite3.connect(credits_store.db_path) as conn:
        status = conn.execute(
            "SELECT status FROM credit_llm_reservations WHERE run_id = ?",
            ("env-platform-failed-run",),
        ).fetchone()[0]
    assert status == "released"


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("llm_overage", "Add at least 0.150000 Credits"),
        ("refund_reconciliation", "payment refund review"),
    ],
)
def test_restricted_account_execution_error_is_actionable(
    tmp_path, monkeypatch, reason, expected
):
    adapter = FakeExecutionAdapter(LLMUsage(input_tokens=100, output_tokens=100))
    service, credits_store = _execution_service(tmp_path, monkeypatch, adapter)
    if reason == "llm_overage":
        reservation = credits_store.reserve_llm_credits(
            reservation_id="restricted-error-reservation",
            user_id=USER_ID,
            run_id="restricted-error-seed",
            call_index=0,
            provider_id="openrouter",
            attempt_index=0,
            reserved_micro=1_000_000,
            operation_key="restricted-error-seed",
            request_digest="r" * 64,
        )
        credits_store.settle_llm_credits(
            reservation["reservation_id"],
            actual_micro=1_250_000,
            evidence={"provider_id": "openrouter", "model_id": MODEL_ID},
        )
    else:
        credits_store.restrict_account(USER_ID, reason=reason)

    with pytest.raises(LLMExecutionError) as exc_info:
        service.execute(_request(f"restricted-{reason}"))

    assert exc_info.value.category is ExecutionErrorCategory.ACCOUNT_RESTRICTED
    assert expected in exc_info.value.safe_message
    assert "CreditAccountRestrictedStoreError" not in exc_info.value.safe_message


def _platform_request(run_id: str, provider_ids: tuple[str, ...]) -> LLMExecutionRequest:
    return LLMExecutionRequest(
        user_id=USER_ID,
        run_id=run_id,
        call_index=0,
        billing_mode=BillingMode.PLATFORM_CREDITS,
        provider_id=provider_ids[0],
        provider_ids=provider_ids,
        model_id=MODEL_ID,
        system_message="Return one trading decision.",
        messages=(LLMMessage(role="user", content="Analyze the market."),),
        usage_policy=UsagePolicy(max_output_tokens=100),
    )


def _quota_error() -> ProviderExecutionError:
    return ProviderExecutionError(ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED)


def _ok_response() -> AdapterResponse:
    return AdapterResponse(
        text="BUY",
        model_id=MODEL_ID,
        usage=LLMUsage(input_tokens=40, output_tokens=20),
        finish_reason="stop",
    )


@pytest.fixture
def fresh_quota_reports(monkeypatch):
    monkeypatch.setattr(execution_service_module, "_quota_exhausted_reported", set())


def test_drained_commonstack_prints_one_operator_error_per_process(
    tmp_path, monkeypatch, capsys, fresh_quota_reports
):
    monkeypatch.setenv("COMMONSTACK_API_KEY", "cs-fake-drained-abcd")
    commonstack = ScriptedExecutionAdapter([_quota_error(), _quota_error()])
    openrouter = ScriptedExecutionAdapter([_ok_response(), _ok_response()])
    service, _store = _execution_service(
        tmp_path,
        monkeypatch,
        openrouter,
        adapters={"commonstack": commonstack, "openrouter": openrouter},
    )

    for run_id in ("drained-1", "drained-2"):
        result = service.execute(
            _platform_request(run_id, ("commonstack", "openrouter"))
        )
        assert result.provider_id == "openrouter"

    lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if "llm.platform_quota_exhausted" in line
    ]
    assert lines == [
        "ERROR: llm.platform_quota_exhausted provider=commonstack fallback=openrouter"
    ]


def test_quota_exhaustion_with_no_next_candidate_names_none(
    tmp_path, monkeypatch, capsys, fresh_quota_reports
):
    monkeypatch.setenv("COMMONSTACK_API_KEY", "cs-fake-drained-only-abcd")
    commonstack = ScriptedExecutionAdapter([_quota_error()])
    service, _store = _execution_service(
        tmp_path,
        monkeypatch,
        commonstack,
        adapters={"commonstack": commonstack},
    )

    with pytest.raises(LLMExecutionError) as exc_info:
        service.execute(_platform_request("drained-only", ("commonstack",)))

    assert exc_info.value.category is ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED
    assert (
        "ERROR: llm.platform_quota_exhausted provider=commonstack fallback=none"
        in capsys.readouterr().out
    )


def test_byok_quota_exhaustion_prints_no_operator_error(
    tmp_path, monkeypatch, capsys, fresh_quota_reports
):
    service, _store = _execution_service(
        tmp_path,
        monkeypatch,
        ScriptedExecutionAdapter([]),
    )

    def fail_byok_once(*_args, **_kwargs):
        raise LLMExecutionError(ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED)

    monkeypatch.setattr(service, "_execute_once", fail_byok_once)
    request = _request("byok-drained").model_copy(
        update={"billing_mode": BillingMode.BYOK}
    )

    with pytest.raises(LLMExecutionError):
        service.execute(request)

    assert "llm.platform_quota_exhausted" not in capsys.readouterr().out


def test_quota_report_is_once_under_concurrency_and_flushed(
    capsys, fresh_quota_reports
):
    import inspect
    import threading

    barrier = threading.Barrier(8)

    def report():
        barrier.wait()
        execution_service_module._report_platform_quota_exhausted(
            "commonstack", "openrouter"
        )

    threads = [threading.Thread(target=report) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert capsys.readouterr().out.count("llm.platform_quota_exhausted") == 1
    # A killed child's block-buffered stdout dies with it, and the 3600s
    # timeout kill is exactly when this line matters.
    assert "flush=True" in inspect.getsource(
        execution_service_module._report_platform_quota_exhausted
    )


def test_worker_never_widens_the_route_s_lone_candidate(
    tmp_path, monkeypatch, capsys, fresh_quota_reports
):
    """A lone ("openrouter",) is what the route decided -- CommonStack pulled
    from ATL_PLATFORM_PROVIDER_ORDER (including by a typo the route rejected),
    or ineligible for the model. The worker used to re-derive routing and add
    CommonStack back; with the default order and CommonStack fully eligible
    it must still bill only the lane it was handed."""
    monkeypatch.delenv("ATL_PLATFORM_PROVIDER_ORDER", raising=False)
    monkeypatch.setenv("COMMONSTACK_API_KEY", "cs-fake-pulled-abcd")
    openrouter = ScriptedExecutionAdapter([_quota_error()])
    commonstack = ScriptedExecutionAdapter([_ok_response()])
    service, _store = _execution_service(
        tmp_path,
        monkeypatch,
        openrouter,
        adapters={"openrouter": openrouter, "commonstack": commonstack},
    )

    with pytest.raises(LLMExecutionError) as exc_info:
        service.execute(_platform_request("pulled-lane", ("openrouter",)))

    assert exc_info.value.category is ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED
    assert commonstack.calls == []


# --- Same-provider retries: LLMExecutionService is the only retry owner -----
# The SDKs run with max_retries=0. The service repeats an attempt at the same
# provider only for pre-send failures and fast rejections, each on its own
# reservation, and never repeats a read timeout.


class _FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class TimedScriptedAdapter:
    """Each outcome is ``(seconds the provider took, response or exception)``."""

    def __init__(self, clock: _FakeClock, outcomes):
        self.clock = clock
        self.outcomes = list(outcomes)
        self.calls = []

    def complete(self, request, credential, provider):
        self.calls.append(request)
        seconds, outcome = self.outcomes.pop(0)
        self.clock.now += seconds
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _rejected(status: int = 503, retry_after: float | None = None):
    return ProviderExecutionError(
        ExecutionErrorCategory.PROVIDER_UNAVAILABLE,
        retry_hint=RetryHint.REJECTED,
        provider_status_code=status,
        retry_after_seconds=retry_after,
    )


def _pre_send():
    return ProviderExecutionError(
        ExecutionErrorCategory.PROVIDER_UNAVAILABLE, retry_hint=RetryHint.PRE_SEND
    )


def _read_timeout():
    return ProviderExecutionError(
        ExecutionErrorCategory.PROVIDER_TIMEOUT,
        retry_hint=RetryHint.NONE,
        timeout_phase="read",
    )


def _reservation_rows(credits_store, run_id: str):
    with credits_store._get_connection() as connection:
        rows = connection.execute(
            "SELECT attempt_index, provider_id, status, failure_reason "
            "FROM credit_llm_reservations WHERE run_id = ? ORDER BY attempt_index",
            (run_id,),
        ).fetchall()
    return [tuple(row) for row in rows]


def _usage_entry_count(credits_store, run_id: str) -> int:
    with credits_store._get_connection() as connection:
        return connection.execute(
            "SELECT COUNT(*) FROM credit_llm_usage_entries WHERE run_id = ?",
            (run_id,),
        ).fetchone()[0]


def _attempt_lines(out: str) -> list[str]:
    return [
        line
        for line in out.splitlines()
        if line.startswith("ERROR: llm.provider_attempt_failed ")
    ]


@pytest.fixture
def retry_harness(tmp_path, monkeypatch):
    monkeypatch.setenv("COMMONSTACK_API_KEY", "cs-fake-retry-abcd")

    def build(commonstack_outcomes, openrouter_outcomes=()):
        clock = _FakeClock()
        sleeps: list[float] = []
        commonstack = TimedScriptedAdapter(clock, commonstack_outcomes)
        openrouter = TimedScriptedAdapter(clock, openrouter_outcomes)
        service, store = _execution_service(
            tmp_path,
            monkeypatch,
            commonstack,
            adapters={"commonstack": commonstack, "openrouter": openrouter},
            sleep=sleeps.append,
            clock=clock,
        )
        return SimpleNamespace(
            service=service,
            store=store,
            commonstack=commonstack,
            openrouter=openrouter,
            sleeps=sleeps,
        )

    return build


_PROD_ORDER = ("commonstack", "openrouter")


def test_fast_rejections_retry_same_provider_with_fresh_reservations(
    retry_harness, capsys
):
    h = retry_harness(
        [(1.0, _rejected()), (1.0, _rejected()), (2.0, _ok_response())]
    )

    result = h.service.execute(_platform_request("fast-rejections", _PROD_ORDER))

    assert result.provider_id == "commonstack"
    assert len(h.commonstack.calls) == 3
    assert h.openrouter.calls == []
    assert h.sleeps == [4.0, 12.0]
    assert _reservation_rows(h.store, "fast-rejections") == [
        (0, "commonstack", "released", "provider_unavailable"),
        (1, "commonstack", "released", "provider_unavailable"),
        (2, "commonstack", "settled", None),
    ]
    assert _usage_entry_count(h.store, "fast-rejections") == 1
    lines = _attempt_lines(capsys.readouterr().out)
    assert [line.split(" attempt=")[1].split()[0] for line in lines] == ["0", "1"]
    assert all(line.endswith(" next=retry") for line in lines)
    assert all(" status=503 hint=rejected elapsed_s=1.0 " in line for line in lines)


def test_three_fast_rejections_fail_over_with_next_attempt_index(retry_harness):
    h = retry_harness(
        [(1.0, _rejected()), (1.0, _rejected()), (1.0, _rejected())],
        [(2.0, _ok_response())],
    )

    result = h.service.execute(_platform_request("rejections-failover", _PROD_ORDER))

    assert result.provider_id == "openrouter"
    assert result.requested_provider_id == "commonstack"
    assert h.sleeps == [4.0, 12.0]
    assert _reservation_rows(h.store, "rejections-failover") == [
        (0, "commonstack", "released", "provider_unavailable"),
        (1, "commonstack", "released", "provider_unavailable"),
        (2, "commonstack", "released", "provider_unavailable"),
        (3, "openrouter", "settled", None),
    ]


@pytest.mark.parametrize(("elapsed", "retried"), [(15.0, True), (15.5, False), (24.0, False)])
def test_rejection_is_retried_only_when_it_failed_fast(retry_harness, elapsed, retried):
    # 24s is CommonStack's 2026-09-27 HTTP 500: it arrived after the
    # generation, and three SDK replays of it bought nothing.
    h = retry_harness(
        [(elapsed, _rejected(500)), (2.0, _ok_response())],
        [(2.0, _ok_response())],
    )

    result = h.service.execute(_platform_request(f"gate-{elapsed}", _PROD_ORDER))

    if retried:
        assert result.provider_id == "commonstack"
        assert len(h.commonstack.calls) == 2
        assert h.sleeps == [4.0]
    else:
        assert result.provider_id == "openrouter"
        assert len(h.commonstack.calls) == 1
        assert h.sleeps == []


def test_pre_send_failure_retries_regardless_of_elapsed(retry_harness):
    h = retry_harness([(20.0, _pre_send()), (2.0, _ok_response())])

    result = h.service.execute(_platform_request("pre-send", _PROD_ORDER))

    assert result.provider_id == "commonstack"
    assert len(h.commonstack.calls) == 2
    assert h.sleeps == [4.0]


@pytest.mark.parametrize("elapsed", [1.0, 180.0])
def test_read_timeout_is_never_retried_on_same_provider(retry_harness, elapsed, capsys):
    h = retry_harness([(elapsed, _read_timeout())], [(2.0, _ok_response())])

    result = h.service.execute(_platform_request(f"read-{elapsed}", _PROD_ORDER))

    assert result.provider_id == "openrouter"
    assert len(h.commonstack.calls) == 1
    assert h.sleeps == []
    assert _reservation_rows(h.store, f"read-{elapsed}") == [
        (0, "commonstack", "released", "provider_timeout"),
        (1, "openrouter", "settled", None),
    ]
    (line,) = _attempt_lines(capsys.readouterr().out)
    assert " provider=commonstack " in line
    assert " category=provider_timeout phase=read status=- hint=none " in line
    assert line.endswith(" next=openrouter")


@pytest.mark.parametrize(
    ("retry_after", "expected_sleeps", "provider"),
    [
        (1.0, [4.0], "commonstack"),
        (7.0, [7.0], "commonstack"),
        # Up to the SDKs' own 60s ceiling: what the SDK used to wait out, we do.
        (45.0, [45.0], "commonstack"),
        (60.0, [60.0], "commonstack"),
        (61.0, [], "openrouter"),
    ],
)
def test_retry_after_is_honoured_up_to_cap(
    retry_harness, retry_after, expected_sleeps, provider
):
    h = retry_harness(
        [(0.5, _rejected(429, retry_after)), (2.0, _ok_response())],
        [(2.0, _ok_response())],
    )

    result = h.service.execute(_platform_request(f"ra-{retry_after}", _PROD_ORDER))

    assert result.provider_id == provider
    assert h.sleeps == expected_sleeps


@pytest.mark.parametrize(
    "category",
    [
        ExecutionErrorCategory.RESPONSE_INVALID,
        ExecutionErrorCategory.USAGE_UNAVAILABLE,
        ExecutionErrorCategory.BILLING_FAILED,
        ExecutionErrorCategory.ACCOUNT_RESTRICTED,
    ],
)
def test_non_failover_categories_never_retry_even_with_a_stray_hint(
    retry_harness, category
):
    h = retry_harness(
        [(1.0, ProviderExecutionError(category, retry_hint=RetryHint.PRE_SEND))]
    )

    with pytest.raises(LLMExecutionError) as exc_info:
        h.service.execute(_platform_request(f"stray-{category.value}", _PROD_ORDER))

    assert exc_info.value.category is category
    assert len(h.commonstack.calls) == 1
    assert h.openrouter.calls == []
    assert h.sleeps == []


@pytest.mark.parametrize(
    "error",
    [
        ProviderExecutionError(
            ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED,
            retry_hint=RetryHint.PRE_SEND,
        ),
        ProviderExecutionError(
            ExecutionErrorCategory.CREDENTIAL_INVALID, retry_hint=RetryHint.PRE_SEND
        ),
        RuntimeError("adapter bug"),
    ],
    ids=["quota", "credential", "raw-exception"],
)
def test_lane_state_and_unclassified_failures_fail_over_without_retry(
    retry_harness, error, fresh_quota_reports
):
    h = retry_harness([(1.0, error)], [(2.0, _ok_response())])

    result = h.service.execute(_platform_request("lane-state", _PROD_ORDER))

    assert result.provider_id == "openrouter"
    assert len(h.commonstack.calls) == 1
    assert h.sleeps == []


def test_byok_fast_rejection_retries_without_reservations(
    retry_harness, monkeypatch, capsys
):
    h = retry_harness([])
    monkeypatch.setattr(
        execution_service_module.analytics_instrumentation,
        "emit_resource_event",
        lambda **_kwargs: None,
    )
    first, second = _rejected(), _rejected()
    first.provider_elapsed_seconds = second.provider_elapsed_seconds = 1.0
    success = LLMExecutionResult(
        text="BUY",
        provider_id="openrouter",
        model_id=MODEL_ID,
        usage=LLMUsage(input_tokens=5, output_tokens=2),
        billing=BillingEvidence(
            billing_source=BillingMode.BYOK, usage_authority="not_billable_by_atl"
        ),
    )
    outcomes = [first, second, success]
    attempts = []

    def execute_once(request, *, attempt_index, requested_provider_id):
        attempts.append(attempt_index)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(h.service, "_execute_once", execute_once)
    request = _request("byok-retry").model_copy(update={"billing_mode": BillingMode.BYOK})

    assert h.service.execute(request) is success
    assert attempts == [0, 1, 2]
    assert h.sleeps == [4.0, 12.0]
    assert _reservation_rows(h.store, "byok-retry") == []
    lines = _attempt_lines(capsys.readouterr().out)
    assert len(lines) == 2
    assert all(" billing=byok " in line and line.endswith(" next=retry") for line in lines)


def test_byok_final_failure_line_names_no_fallback(retry_harness, monkeypatch, capsys):
    h = retry_harness([])
    error = _read_timeout()
    error.provider_elapsed_seconds = 180.0

    def execute_once(request, **_kwargs):
        raise error

    monkeypatch.setattr(h.service, "_execute_once", execute_once)
    request = _request("byok-timeout").model_copy(update={"billing_mode": BillingMode.BYOK})

    with pytest.raises(LLMExecutionError) as exc_info:
        h.service.execute(request)

    assert exc_info.value is error
    assert h.sleeps == []
    (line,) = _attempt_lines(capsys.readouterr().out)
    assert " billing=byok " in line and line.endswith(" next=none")


def test_reused_attempt_index_fails_billing(retry_harness):
    """Why retries take a fresh attempt_index: the reservation key would collide."""

    h = retry_harness([(1.0, _rejected()), (1.0, _ok_response())])
    request = _platform_request("reused-index", ("commonstack",))

    with pytest.raises(LLMExecutionError):
        h.service._execute_once(
            request, attempt_index=0, requested_provider_id="commonstack"
        )
    with pytest.raises(LLMExecutionError) as exc_info:
        h.service._execute_once(
            request, attempt_index=0, requested_provider_id="commonstack"
        )

    assert exc_info.value.category is ExecutionErrorCategory.BILLING_FAILED
    assert len(h.commonstack.calls) == 1


_C = ExecutionErrorCategory


@pytest.mark.parametrize(
    ("commonstack_category", "openrouter_category", "expected"),
    [
        (_C.PROVIDER_TIMEOUT, _C.PROVIDER_QUOTA_EXHAUSTED, _C.PROVIDER_TIMEOUT),
        (_C.PROVIDER_UNAVAILABLE, _C.PROVIDER_QUOTA_EXHAUSTED, _C.PROVIDER_UNAVAILABLE),
        (_C.PROVIDER_QUOTA_EXHAUSTED, _C.PROVIDER_TIMEOUT, _C.PROVIDER_TIMEOUT),
        (_C.PROVIDER_QUOTA_EXHAUSTED, _C.PROVIDER_QUOTA_EXHAUSTED, _C.PROVIDER_QUOTA_EXHAUSTED),
        (_C.PROVIDER_UNAVAILABLE, _C.CREDENTIAL_INVALID, _C.PROVIDER_UNAVAILABLE),
        (_C.PROVIDER_TIMEOUT, _C.PROVIDER_UNAVAILABLE, _C.PROVIDER_TIMEOUT),
    ],
)
def test_representative_error_prefers_call_specific_failure(
    retry_harness,
    monkeypatch,
    fresh_quota_reports,
    commonstack_category,
    openrouter_category,
    expected,
):
    """09-28: CommonStack timed out, OpenRouter answered 402, and the run
    reported "insufficient balance". The first timeout/unavailable wins."""

    errors = []
    monkeypatch.setattr(
        execution_service_module.analytics_instrumentation,
        "emit_safe_error_event",
        lambda **kwargs: errors.append(kwargs),
    )
    h = retry_harness(
        [(1.0, ProviderExecutionError(commonstack_category))],
        [(1.0, ProviderExecutionError(openrouter_category))],
    )

    with pytest.raises(LLMExecutionError) as exc_info:
        h.service.execute(_platform_request("representative", _PROD_ORDER))

    assert exc_info.value.category is expected
    assert errors[-1]["error_category"] == expected.value
    assert h.sleeps == []


def test_quota_line_names_next_candidate_after_same_provider_retries(
    retry_harness, capsys, fresh_quota_reports
):
    h = retry_harness(
        [(1.0, _pre_send()), (1.0, _quota_error())],
        [(1.0, _ok_response())],
    )

    result = h.service.execute(_platform_request("quota-after-retry", _PROD_ORDER))

    assert result.provider_id == "openrouter"
    assert (
        "ERROR: llm.platform_quota_exhausted provider=commonstack fallback=openrouter"
        in capsys.readouterr().out
    )
    assert [row[:2] for row in _reservation_rows(h.store, "quota-after-retry")] == [
        (0, "commonstack"),
        (1, "commonstack"),
        (2, "openrouter"),
    ]


def test_no_attempt_line_for_response_invalid_or_quota(
    retry_harness, capsys, fresh_quota_reports
):
    h = retry_harness(
        [
            (1.0, ProviderExecutionError(ExecutionErrorCategory.RESPONSE_INVALID)),
            (1.0, _quota_error()),
        ],
        [(1.0, _ok_response())],
    )
    with pytest.raises(LLMExecutionError):
        h.service.execute(_platform_request("invalid-reply", _PROD_ORDER))
    h.service.execute(_platform_request("quota-reply", _PROD_ORDER))

    assert _attempt_lines(capsys.readouterr().out) == []


def test_incident_shape_end_to_end(tmp_path, monkeypatch, capsys, fresh_quota_reports):
    """Run agent_20260928_024706_cbda3555, replayed through the real SDK.

    CommonStack stalls past the read deadline, OpenRouter is quota-dead. On
    main that was three CommonStack generations inside one reservation and a
    run reported as quota exhaustion.
    """

    monkeypatch.setenv("COMMONSTACK_API_KEY", "cs-fake-incident-abcd")
    monkeypatch.delenv("BROKER_CREDENTIAL_VERIFICATION_PROXY", raising=False)
    sends = []

    def stall(request: httpx.Request) -> httpx.Response:
        sends.append(request.url.host)
        raise httpx.ReadTimeout("generation still running", request=request)

    monkeypatch.setattr(
        "dashboard.backend.infrastructure.llm.execution.adapters.base."
        "build_pinned_transport",
        lambda *_args, **_kwargs: httpx.MockTransport(stall),
    )
    sdk_sleeps: list[float] = []
    monkeypatch.setattr(openai_base_client.time, "sleep", sdk_sleeps.append)
    service_sleeps: list[float] = []
    openrouter = ScriptedExecutionAdapter([_quota_error()])
    service, store = _execution_service(
        tmp_path,
        monkeypatch,
        openrouter,
        adapters={
            "commonstack": OpenAICompatibleAdapter(),
            "openrouter": openrouter,
        },
        sleep=service_sleeps.append,
    )
    request = _platform_request("incident-shape", _PROD_ORDER).model_copy(
        update={"model_id": "deepseek/deepseek-v4-pro"}
    )

    with pytest.raises(LLMExecutionError) as exc_info:
        service.execute(request)

    assert exc_info.value.category is ExecutionErrorCategory.PROVIDER_TIMEOUT
    assert sends == ["api.commonstack.ai"]
    assert sdk_sleeps == [] and service_sleeps == []
    assert _reservation_rows(store, "incident-shape") == [
        (0, "commonstack", "released", "provider_timeout"),
        (1, "openrouter", "released", "provider_quota_exhausted"),
    ]
    out = capsys.readouterr().out
    (line,) = _attempt_lines(out)
    assert " provider=commonstack model=deepseek/deepseek-v4-pro " in line
    assert " category=provider_timeout phase=read status=- hint=none " in line
    assert " read_timeout_s=180 next=openrouter" in line
    assert "ERROR: llm.platform_quota_exhausted provider=openrouter fallback=none" in out
    assert "cs-fake-incident-abcd" not in out


def test_byok_waits_out_a_long_retry_after_rather_than_aborting(
    retry_harness, monkeypatch
):
    """BYOK has no fallback: "fail over instead of waiting" would abort the run."""

    h = retry_harness([])
    refused = _rejected(429, retry_after=45.0)
    refused.provider_elapsed_seconds = 0.5
    outcomes = [refused, "ok"]

    def execute_once(request, **_kwargs):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(h.service, "_execute_once", execute_once)
    monkeypatch.setattr(h.service, "_emit_model_usage", lambda *_args: None)
    request = _request("byok-429").model_copy(update={"billing_mode": BillingMode.BYOK})

    assert h.service.execute(request) == "ok"
    assert h.sleeps == [45.0]
