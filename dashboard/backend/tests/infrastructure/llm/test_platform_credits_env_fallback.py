"""Execution and ledger coverage for the environment-backed Platform lane."""

from __future__ import annotations

import sqlite3

import pytest

from dashboard.backend.domain.credits.repository import CreditsStore
from dashboard.backend.domain.credits.service import CreditsService
from dashboard.backend.domain.model_providers.repository import ModelProviderStore
from dashboard.backend.domain.model_providers.service import ModelProviderService
from dashboard.backend.infrastructure.llm.execution.adapters.base import (
    AdapterResponse,
    ProviderExecutionError,
)
from dashboard.backend.infrastructure.llm.execution.errors import (
    ExecutionErrorCategory,
    LLMExecutionError,
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


def _execution_service(tmp_path, monkeypatch, adapter, *, adapters=None):
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

    service = LLMExecutionService(
        providers=ModelProviderService(store=provider_store),
        credits=credits_service,
        adapter_resolver=resolve_adapter,
        pricing_snapshot_factory=snapshot_for,
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
    request = _request("failover-run").model_copy(
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
        service.execute(_request(f"no-failover-{category.value}"))

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

    result = service.execute(_request(f"failover-{category.value}"))

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
        service.execute(_request("dual-failure"))
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
        service.execute(_request("release-failure"))
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
        service.execute(_request("double-quota"))
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
