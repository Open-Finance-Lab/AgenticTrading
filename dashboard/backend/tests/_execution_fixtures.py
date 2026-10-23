"""A real LLM execution service over SQLite stores, for tests that bill through it.

Shared by the Platform-lane ledger tests and the backtest retry tests, which
both need the service end to end -- reservation, adapter call, settlement or
release -- rather than a stand-in.
"""

from __future__ import annotations

import sqlite3

from dashboard.backend.domain.credits.repository import CreditsStore
from dashboard.backend.domain.credits.service import CreditsService
from dashboard.backend.domain.model_providers.repository import ModelProviderStore
from dashboard.backend.domain.model_providers.service import ModelProviderService
from dashboard.backend.infrastructure.llm.execution.models import PricingSnapshot
from dashboard.backend.infrastructure.llm.execution.service import LLMExecutionService


USER_ID = 1
ADMIN_ID = 2
MODEL_ID = "openai/gpt-5.5"


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


def seed_users(path) -> None:
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


def enable_openrouter(store: ModelProviderStore) -> None:
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


def seed_balances(store: CreditsStore) -> None:
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


def build_execution_service(
    tmp_path,
    monkeypatch,
    adapter,
    *,
    adapters=None,
    sleep=None,
    clock=None,
    usd_per_million_tokens: float = 1000.0,
):
    """A real ``LLMExecutionService`` over SQLite stores, with ``adapter`` as every provider.

    The user holds a 0.1-Credit Grant and 1 purchased Credit. The default
    $1000/Mtok price keeps ledger arithmetic legible; a caller sending a
    real-sized prompt lowers it, or the reservation outgrows that balance.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-execution-test-abcd")
    provider_store = ModelProviderStore(tmp_path / "providers.db")
    enable_openrouter(provider_store)
    credits_path = tmp_path / "credits.db"
    seed_users(credits_path)
    credits_store = CreditsStore(credits_path)
    seed_balances(credits_store)
    credits_service = CreditsService(store=credits_store)

    def snapshot_for(model_id: str, provider_id: str) -> PricingSnapshot:
        return PricingSnapshot(
            provider_id=provider_id,
            model_id=model_id,
            input_usd_per_million_tokens=usd_per_million_tokens,
            output_usd_per_million_tokens=usd_per_million_tokens,
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
