"""SQLite Credits ledger, payment operations, and refund reservations."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from dashboard.backend.domain.credits.repository import (
    CreditsStore,
    OrderConflictError,
    RefundNotAllowedError,
)
from dashboard.backend.domain.credits.repository_common import (
    CreditAccountRestrictedStoreError,
    LLMReservationConflictError,
)
from dashboard.backend.domain.model_providers.repository_common import (
    ModelProviderStoreError,
)


def _store(tmp_path) -> CreditsStore:
    path = tmp_path / "credits.db"
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
            INSERT INTO users (id, email, display_name, password_hash, role, created_at)
            VALUES (?, ?, ?, 'unused', ?, '2026-08-13T00:00:00+00:00')
            """,
            [
                (1, "buyer@example.com", "Buyer", "user"),
                (2, "admin@example.com", "Admin", "admin"),
                (3, "other@example.com", "Other", "user"),
            ],
        )
    return CreditsStore(db_path=path)


def _pending_order(
    store: CreditsStore,
    *,
    order_id: str = "ord_10",
    user_id: int = 1,
    client_request_id: str = "11111111-1111-4111-8111-111111111111",
    amount_usd_cents: int = 1000,
    credits_micro: int = 10_000_000,
):
    order = store.create_or_get_order(
        order_id=order_id,
        user_id=user_id,
        client_request_id=client_request_id,
        amount_usd_cents=amount_usd_cents,
        credits_micro=credits_micro,
    )
    return store.attach_checkout_session(
        order["id"], checkout_session_id=f"cs_test_{order_id}"
    )


def _pay_order(
    store: CreditsStore,
    *,
    order_id: str = "ord_10",
    event_id: str = "evt_paid_10",
    amount_usd_cents: int = 1000,
):
    result = store.settle_paid_checkout(
        event_id=event_id,
        event_type="checkout.session.completed",
        livemode=False,
        object_id=f"cs_test_{order_id}",
        payload_sha256="a" * 64,
        order_id=order_id,
        checkout_session_id=f"cs_test_{order_id}",
        payment_intent_id=f"pi_test_{order_id}",
        currency="usd",
        amount_usd_cents=amount_usd_cents,
    )
    assert result["outcome"] == "processed"
    return result


def _settle_activity_call(
    store: CreditsStore,
    *,
    user_id: int = 1,
    run_id: str,
    call_index: int,
    actual_micro: int,
    provider_id: str = "openrouter",
    model_id: str = "anthropic/claude-haiku-4-5",
):
    reservation_id = f"activity:{user_id}:{run_id}:{call_index}"
    reservation = store.reserve_llm_credits(
        reservation_id=reservation_id,
        user_id=user_id,
        run_id=run_id,
        call_index=call_index,
        provider_id=provider_id,
        attempt_index=0,
        reserved_micro=actual_micro,
        operation_key=f"reserve:{reservation_id}",
        request_digest=f"{user_id}{call_index}".ljust(64, "a")[:64],
    )
    return store.settle_llm_credits(
        reservation["reservation_id"],
        actual_micro=actual_micro,
        evidence={
            "billing_source": "platform_credits",
            "pricing_snapshot": {
                "provider_id": provider_id,
                "model_id": model_id,
            },
        },
    )


def test_schema_is_created_and_new_account_balance_is_zero(tmp_path):
    store = _store(tmp_path)

    account = store.ensure_account(1)

    assert account["user_id"] == 1
    assert account["status"] == "active"
    assert store.get_balance_micro(1) == 0
    with store._get_connection() as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert {
        "credit_accounts",
        "credit_payment_orders",
        "credit_refund_requests",
        "stripe_webhook_events",
        "credit_ledger_entries",
        "credit_grant_pools",
        "credit_grant_pool_ledger_entries",
    }.issubset(tables)


def test_sqlite_migrates_legacy_reservation_settlement_constraint(tmp_path):
    path = tmp_path / "legacy-credits.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE credit_llm_reservations (
                reservation_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                run_id TEXT NOT NULL,
                call_index INTEGER NOT NULL,
                reserved_micro INTEGER NOT NULL,
                reserved_grant_micro INTEGER NOT NULL,
                reserved_purchased_micro INTEGER NOT NULL,
                settled_micro INTEGER NOT NULL DEFAULT 0,
                actual_micro INTEGER NOT NULL DEFAULT 0,
                outstanding_micro INTEGER NOT NULL DEFAULT 0,
                outstanding_recovered_micro INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'open',
                operation_key TEXT NOT NULL UNIQUE,
                request_digest TEXT NOT NULL,
                evidence_json TEXT,
                failure_reason TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                CHECK (reserved_micro = reserved_grant_micro + reserved_purchased_micro),
                CHECK (settled_micro <= reserved_micro),
                UNIQUE (user_id, run_id, call_index),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE credit_llm_usage_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                reservation_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                call_index INTEGER NOT NULL,
                bucket TEXT NOT NULL,
                amount_micro INTEGER NOT NULL,
                operation_key TEXT NOT NULL UNIQUE,
                evidence_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (reservation_id) REFERENCES credit_llm_reservations(reservation_id)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO credit_llm_reservations (
                reservation_id, user_id, run_id, call_index, reserved_micro,
                reserved_grant_micro, reserved_purchased_micro, operation_key,
                request_digest, created_at, updated_at
            ) VALUES ('legacy-reservation', 1, 'legacy-run', 0, 1000, 1000, 0,
                      'legacy-operation', 'a', '2026-08-01', '2026-08-01')
            """
        )
        conn.execute(
            """
            INSERT INTO credit_llm_usage_entries (
                user_id, reservation_id, run_id, call_index, bucket,
                amount_micro, operation_key, evidence_json, created_at
            ) VALUES (1, 'legacy-reservation', 'legacy-run', 0, 'grant',
                      -100, 'legacy-usage', '{}', '2026-08-01')
            """
        )

    # The Credits store can initialize before UserStore creates its parent
    # table. The first pass must not fail or mark the migration complete.
    store = CreditsStore(path)

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
        conn.execute(
            "INSERT INTO users VALUES (1, 'legacy@example.com', 'Legacy', 'unused', 'user', '2026-08-01')"
        )
    settled = store.settle_llm_credits(
        "legacy-reservation",
        actual_micro=1_200,
        evidence={"provider_id": "openrouter", "model_id": "qwen/qwen3"},
    )

    with store._get_connection() as conn:
        schema = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'credit_llm_reservations'"
        ).fetchone()[0]
        row = conn.execute(
            "SELECT reserved_micro, operation_key, attempt_index, provider_id "
            "FROM credit_llm_reservations"
        ).fetchone()
        usage = conn.execute(
            "SELECT operation_key, amount_micro FROM credit_llm_usage_entries"
        ).fetchone()

    assert "settled_micro <= reserved_micro" not in "".join(schema.lower().split())
    assert tuple(row) == (1000, "legacy-operation", 0, None)
    assert tuple(usage) == ("legacy-usage", -100)
    assert settled["settled_micro"] == 1000
    assert settled["outstanding_micro"] == 200


def test_provider_attempts_have_independent_reservations_for_one_logical_call(
    tmp_path,
):
    store = _store(tmp_path)
    _pending_order(store, amount_usd_cents=100, credits_micro=1_000_000)
    _pay_order(store, amount_usd_cents=100)

    primary = store.reserve_llm_credits(
        reservation_id="attempt-primary",
        user_id=1,
        run_id="attempt-run",
        call_index=3,
        attempt_index=0,
        provider_id="openrouter",
        reserved_micro=100_000,
        operation_key="attempt-primary-operation",
        request_digest="a" * 64,
    )
    store.release_llm_credits(
        primary["reservation_id"], reason="provider_quota_exhausted"
    )
    fallback = store.reserve_llm_credits(
        reservation_id="attempt-fallback",
        user_id=1,
        run_id="attempt-run",
        call_index=3,
        attempt_index=1,
        provider_id="commonstack",
        reserved_micro=100_000,
        operation_key="attempt-fallback-operation",
        request_digest="b" * 64,
    )

    assert primary["attempt_index"] == 0
    assert primary["provider_id"] == "openrouter"
    assert fallback["attempt_index"] == 1
    assert fallback["provider_id"] == "commonstack"


def test_reservation_attempt_identity_is_validated_and_idempotent(tmp_path):
    store = _store(tmp_path)
    _pending_order(store, amount_usd_cents=100, credits_micro=1_000_000)
    _pay_order(store, amount_usd_cents=100)
    payload = {
        "reservation_id": "attempt-validation",
        "user_id": 1,
        "run_id": "attempt-validation-run",
        "call_index": 0,
        "attempt_index": 0,
        "provider_id": "openrouter",
        "reserved_micro": 100_000,
        "operation_key": "attempt-validation-operation",
        "request_digest": "v" * 64,
    }
    store.reserve_llm_credits(**payload)
    assert store.reserve_llm_credits(**payload)["attempt_index"] == 0

    with pytest.raises(ValueError, match="attempt_index"):
        store.reserve_llm_credits(**{**payload, "attempt_index": -1})
    with pytest.raises(ModelProviderStoreError, match="provider id"):
        store.reserve_llm_credits(**{**payload, "provider_id": " "})
    with pytest.raises(LLMReservationConflictError, match="different input"):
        store.reserve_llm_credits(**{**payload, "provider_id": "commonstack"})
    with pytest.raises(LLMReservationConflictError, match="different input"):
        store.reserve_llm_credits(**{**payload, "attempt_index": 1})


def test_llm_overage_debits_the_hold_and_restricts_the_account(tmp_path):
    store = _store(tmp_path)
    _pending_order(store, amount_usd_cents=110, credits_micro=1_100_000)
    _pay_order(store, amount_usd_cents=110)
    reservation = store.reserve_llm_credits(
        reservation_id="llm-res-overage",
        user_id=1,
        run_id="run-overage",
        call_index=0,
        provider_id="openrouter",
        attempt_index=0,
        reserved_micro=1_000_000,
        operation_key="llm-reserve-overage",
        request_digest="a" * 64,
    )
    evidence = {
        "provider_id": "openrouter",
        "model_id": "openai/gpt-5.5",
        "provider_cost_credits_micro": 1_250_000,
        "debited_credits_micro": 1_000_000,
        "outstanding_credits_micro": 250_000,
    }

    settled = store.settle_llm_credits(
        reservation["reservation_id"],
        actual_micro=1_250_000,
        evidence=evidence,
    )

    assert settled["status"] == "settled"
    assert settled["actual_micro"] == 1_250_000
    assert settled["settled_micro"] == 1_100_000
    assert settled["outstanding_micro"] == 150_000
    assert settled["outstanding_recovered_micro"] == 0
    assert settled["released_micro"] == 0
    assert settled["grant_debited_micro"] == 0
    assert settled["purchased_debited_micro"] == 1_100_000
    assert store.get_balance_micro(1) == 0
    assert store.ensure_account(1)["status"] == "restricted"
    assert store.get_account_billing_state(1) == {
        "account_status": "restricted",
        "restriction_reason": "llm_overage",
        "outstanding_credits_micro": 150_000,
    }
    assert store.settle_llm_credits(
        reservation["reservation_id"],
        actual_micro=1_250_000,
        evidence=evidence,
    ) == settled

    with pytest.raises(LLMReservationConflictError, match="different input"):
        store.settle_llm_credits(
            reservation["reservation_id"],
            actual_micro=1_250_001,
            evidence=evidence,
        )
    with pytest.raises(CreditAccountRestrictedStoreError, match="restricted"):
        store.reserve_llm_credits(
            reservation_id="llm-res-blocked",
            user_id=1,
            run_id="run-blocked",
            call_index=0,
            provider_id="openrouter",
            attempt_index=0,
            reserved_micro=1,
            operation_key="llm-reserve-blocked",
            request_digest="b" * 64,
        )


def test_llm_overage_uses_unreserved_grant_before_restricting(tmp_path):
    store = _store(tmp_path)
    store.fund_grant_pool(
        pool_id="default",
        amount_micro=2_000_000,
        operation_id="fund-covered-overage",
        idempotency_key="fund-covered-overage",
        request_digest="f" * 64,
        actor_user_id=2,
        source="test",
        reason="Fund covered overage.",
    )
    store.assign_grant(
        user_id=1,
        pool_id="default",
        amount_micro=2_000_000,
        operation_id="assign-covered-overage",
        idempotency_key="assign-covered-overage",
        request_digest="g" * 64,
        actor_user_id=2,
        source="test",
        reason="Assign covered overage.",
    )
    reservation = store.reserve_llm_credits(
        reservation_id="llm-res-covered-overage",
        user_id=1,
        run_id="run-covered-overage",
        call_index=0,
        provider_id="openrouter",
        attempt_index=0,
        reserved_micro=1_000_000,
        operation_key="llm-reserve-covered-overage",
        request_digest="h" * 64,
    )

    settled = store.settle_llm_credits(
        reservation["reservation_id"],
        actual_micro=1_250_000,
        evidence={
            "provider_id": "openrouter",
            "model_id": "qwen/qwen3",
            "debited_credits_micro": 1_000_000,
            "outstanding_credits_micro": 250_000,
        },
    )

    assert settled["settled_micro"] == 1_250_000
    assert settled["outstanding_micro"] == 0
    assert settled["released_micro"] == 0
    assert settled["grant_debited_micro"] == 1_250_000
    assert settled["purchased_debited_micro"] == 0
    assert store.get_balance_micro(1) == 750_000
    assert store.get_account_billing_state(1) == {
        "account_status": "active",
        "restriction_reason": None,
        "outstanding_credits_micro": 0,
    }


def test_llm_overage_does_not_spend_another_open_reservation(tmp_path):
    store = _store(tmp_path)
    _pending_order(store, amount_usd_cents=200, credits_micro=2_000_000)
    _pay_order(store, amount_usd_cents=200)
    current = store.reserve_llm_credits(
        reservation_id="llm-res-current",
        user_id=1,
        run_id="run-current",
        call_index=0,
        provider_id="openrouter",
        attempt_index=0,
        reserved_micro=1_000_000,
        operation_key="llm-reserve-current",
        request_digest="i" * 64,
    )
    protected = store.reserve_llm_credits(
        reservation_id="llm-res-protected",
        user_id=1,
        run_id="run-protected",
        call_index=0,
        provider_id="openrouter",
        attempt_index=0,
        reserved_micro=500_000,
        operation_key="llm-reserve-protected",
        request_digest="j" * 64,
    )

    settled = store.settle_llm_credits(
        current["reservation_id"],
        actual_micro=1_750_000,
        evidence={"provider_id": "openrouter", "model_id": "qwen/qwen3"},
    )

    assert settled["settled_micro"] == 1_500_000
    assert settled["outstanding_micro"] == 250_000
    assert store.get_balance_projection(1)["total_available_micro"] == 0
    with store._get_connection() as conn:
        row = conn.execute(
            "SELECT status, reserved_micro FROM credit_llm_reservations "
            "WHERE reservation_id = ?",
            (protected["reservation_id"],),
        ).fetchone()
    assert tuple(row) == ("open", 500_000)


def test_grant_recovers_llm_overage_and_reinstates_account(tmp_path):
    store = _store(tmp_path)
    _pending_order(store, amount_usd_cents=110, credits_micro=1_100_000)
    _pay_order(store, amount_usd_cents=110)
    reservation = store.reserve_llm_credits(
        reservation_id="llm-res-recover",
        user_id=1,
        run_id="run-recover",
        call_index=0,
        provider_id="openrouter",
        attempt_index=0,
        reserved_micro=1_000_000,
        operation_key="llm-reserve-recover",
        request_digest="c" * 64,
    )
    store.settle_llm_credits(
        reservation["reservation_id"],
        actual_micro=1_250_000,
        evidence={"provider_id": "openrouter", "model_id": "qwen/qwen3"},
    )
    store.fund_grant_pool(
        pool_id="default",
        amount_micro=250_000,
        operation_id="fund-recovery",
        idempotency_key="fund-recovery",
        request_digest="d" * 64,
        actor_user_id=2,
        source="test",
        reason="Fund recovery.",
    )
    assigned = store.assign_grant(
        pool_id="default",
        user_id=1,
        amount_micro=250_000,
        operation_id="assign-recovery",
        idempotency_key="assign-recovery",
        request_digest="e" * 64,
        actor_user_id=2,
        source="test",
        reason="Recover overage.",
    )

    assert assigned["recovery"] == {
        "recovered_micro": 150_000,
        "outstanding_micro": 0,
        "account_status": "active",
        "restriction_reason": None,
    }
    replayed = store.assign_grant(
        user_id=1,
        pool_id="default",
        amount_micro=250_000,
        operation_id="assign-recovery",
        idempotency_key="assign-recovery",
        request_digest="e" * 64,
        actor_user_id=2,
        source="test",
        reason="Recover overage.",
    )
    assert replayed == assigned
    assert store.get_account_billing_state(1) == {
        "account_status": "active",
        "restriction_reason": None,
        "outstanding_credits_micro": 0,
    }
    with store._get_connection() as conn:
        row = conn.execute(
            "SELECT outstanding_recovered_micro FROM credit_llm_reservations WHERE reservation_id = ?",
            (reservation["reservation_id"],),
        ).fetchone()
        assert row["outstanding_recovered_micro"] == 150_000


@pytest.mark.parametrize(
    ("cents", "micro"),
    [(500, 5_000_000), (1000, 10_000_000), (20_000, 200_000_000)],
)
def test_order_stores_exact_integer_amounts(tmp_path, cents, micro):
    store = _store(tmp_path)

    order = store.create_or_get_order(
        order_id=f"ord_{cents}",
        user_id=1,
        client_request_id=f"11111111-1111-4111-8111-{cents:012d}",
        amount_usd_cents=cents,
        credits_micro=micro,
    )

    assert order["amount_usd_cents"] == cents
    assert order["credits_micro"] == micro
    assert order["currency"] == "usd"
    assert order["stripe_mode"] == "test"
    assert order["status"] == "pending"


@pytest.mark.parametrize(
    ("cents", "micro"),
    [(10.0, 100_000), (10, 100_000.0), (0, 0), (-1, -10_000)],
)
def test_order_rejects_float_zero_and_negative_amounts(tmp_path, cents, micro):
    store = _store(tmp_path)

    with pytest.raises(ValueError, match="positive integer"):
        store.create_or_get_order(
            order_id="ord_bad",
            user_id=1,
            client_request_id="22222222-2222-4222-8222-222222222222",
            amount_usd_cents=cents,
            credits_micro=micro,
        )


def test_client_request_retry_returns_same_order_but_changed_amount_conflicts(tmp_path):
    store = _store(tmp_path)
    original = _pending_order(store)

    retried = store.create_or_get_order(
        order_id="ord_different_ignored",
        user_id=1,
        client_request_id="11111111-1111-4111-8111-111111111111",
        amount_usd_cents=1000,
        credits_micro=10_000_000,
    )

    assert retried["id"] == original["id"]
    with pytest.raises(OrderConflictError, match="different purchase"):
        store.create_or_get_order(
            order_id="ord_conflict",
            user_id=1,
            client_request_id="11111111-1111-4111-8111-111111111111",
            amount_usd_cents=500,
            credits_micro=5_000_000,
        )


def test_checkout_session_attachment_is_compare_and_set(tmp_path):
    store = _store(tmp_path)
    _pending_order(store)

    same = store.attach_checkout_session(
        "ord_10", checkout_session_id="cs_test_ord_10"
    )
    assert same["stripe_checkout_session_id"] == "cs_test_ord_10"

    with pytest.raises(OrderConflictError, match="Checkout Session"):
        store.attach_checkout_session(
            "ord_10", checkout_session_id="cs_test_other"
        )


def test_paid_checkout_posts_one_purchase_even_when_events_repeat(tmp_path):
    store = _store(tmp_path)
    _pending_order(store)

    first = _pay_order(store)
    same_event = store.settle_paid_checkout(
        event_id="evt_paid_10",
        event_type="checkout.session.completed",
        livemode=False,
        object_id="cs_test_ord_10",
        payload_sha256="a" * 64,
        order_id="ord_10",
        checkout_session_id="cs_test_ord_10",
        payment_intent_id="pi_test_ord_10",
        currency="usd",
        amount_usd_cents=1000,
    )
    second_event = store.settle_paid_checkout(
        event_id="evt_paid_10_retry",
        event_type="checkout.session.completed",
        livemode=False,
        object_id="cs_test_ord_10",
        payload_sha256="b" * 64,
        order_id="ord_10",
        checkout_session_id="cs_test_ord_10",
        payment_intent_id="pi_test_ord_10",
        currency="usd",
        amount_usd_cents=1000,
    )

    assert first["balance_micro"] == 10_000_000
    assert same_event["outcome"] == "duplicate"
    assert second_event["outcome"] == "duplicate"
    assert store.get_balance_micro(1) == 10_000_000
    ledger = store.list_ledger_entries(1)
    assert len(ledger["items"]) == 1
    assert ledger["items"][0]["entry_type"] == "purchase"
    assert ledger["items"][0]["amount_micro"] == 10_000_000


@pytest.mark.parametrize("terminal_status", ["expired", "failed"])
def test_unpaid_checkout_terminal_event_updates_pending_order_without_credit(
    tmp_path, terminal_status
):
    store = _store(tmp_path)
    _pending_order(store)

    result = store.settle_unpaid_checkout(
        event_id=f"evt_{terminal_status}",
        event_type=(
            "checkout.session.expired"
            if terminal_status == "expired"
            else "checkout.session.async_payment_failed"
        ),
        livemode=False,
        object_id="cs_test_ord_10",
        payload_sha256=terminal_status.ljust(64, "a"),
        order_id="ord_10",
        checkout_session_id="cs_test_ord_10",
        terminal_status=terminal_status,
    )

    assert result == {"outcome": "processed", "status": terminal_status}
    assert store.get_order_for_user("ord_10", 1)["status"] == terminal_status
    assert store.get_balance_micro(1) == 0
    assert store.list_ledger_entries(1)["items"] == []


def test_unpaid_checkout_event_cannot_downgrade_a_paid_order(tmp_path):
    store = _store(tmp_path)
    _pending_order(store)
    _pay_order(store)

    result = store.settle_unpaid_checkout(
        event_id="evt_late_expiry",
        event_type="checkout.session.expired",
        livemode=False,
        object_id="cs_test_ord_10",
        payload_sha256="z" * 64,
        order_id="ord_10",
        checkout_session_id="cs_test_ord_10",
        terminal_status="expired",
    )

    assert result["outcome"] == "ignored"
    assert store.get_order_for_user("ord_10", 1)["status"] == "paid"
    assert store.get_balance_micro(1) == 10_000_000


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("livemode", True, "Live Mode"),
        ("currency", "eur", "currency"),
        ("amount_usd_cents", 999, "amount"),
        ("checkout_session_id", "cs_test_wrong", "Checkout Session"),
    ],
)
def test_mismatched_paid_event_is_recorded_but_never_credits(
    tmp_path, field, value, reason
):
    store = _store(tmp_path)
    _pending_order(store)
    payload = {
        "event_id": f"evt_bad_{field}",
        "event_type": "checkout.session.completed",
        "livemode": False,
        "object_id": "cs_test_ord_10",
        "payload_sha256": "c" * 64,
        "order_id": "ord_10",
        "checkout_session_id": "cs_test_ord_10",
        "payment_intent_id": "pi_test_ord_10",
        "currency": "usd",
        "amount_usd_cents": 1000,
    }
    payload[field] = value

    result = store.settle_paid_checkout(**payload)

    assert result["outcome"] == "rejected"
    assert reason.lower() in result["reason"].lower()
    assert store.get_balance_micro(1) == 0
    assert store.list_ledger_entries(1)["items"] == []


def test_ledger_is_cursor_paginated_and_exposes_no_mutation_method(tmp_path):
    store = _store(tmp_path)
    _pending_order(store)
    _pay_order(store)
    store.reserve_refund(
        refund_id="rfnd_2",
        payment_order_id="ord_10",
        user_id=1,
        requested_by_user_id=2,
        amount_usd_cents=200,
        credits_micro=2_000_000,
    )
    store.attach_stripe_refund("rfnd_2", stripe_refund_id="re_test_2")
    store.settle_succeeded_refund(
        event_id="evt_refund_2",
        event_type="refund.created",
        livemode=False,
        object_id="re_test_2",
        payload_sha256="d" * 64,
        refund_id="rfnd_2",
        stripe_refund_id="re_test_2",
        payment_intent_id="pi_test_ord_10",
        currency="usd",
        amount_usd_cents=200,
    )

    first = store.list_ledger_entries(1, limit=1)
    second = store.list_ledger_entries(1, limit=1, cursor=first["next_cursor"])

    assert len(first["items"]) == len(second["items"]) == 1
    assert first["items"][0]["id"] != second["items"][0]["id"]
    assert not hasattr(store, "update_ledger_entry")
    assert not hasattr(store, "delete_ledger_entry")


def test_activity_aggregates_settled_calls_and_buckets_by_run(tmp_path):
    store = _store(tmp_path)
    grant_common = {
        "pool_id": "default",
        "actor_user_id": 2,
        "source": "test",
    }
    store.fund_grant_pool(
        amount_micro=500,
        operation_id="activity-fund",
        idempotency_key="activity-fund-key",
        request_digest="activity-fund-digest",
        reason="Fund activity test pool.",
        **grant_common,
    )
    store.assign_grant(
        user_id=1,
        amount_micro=500,
        operation_id="activity-assign",
        idempotency_key="activity-assign-key",
        request_digest="activity-assign-digest",
        reason="Assign activity test Credits.",
        **grant_common,
    )
    _pending_order(store)
    _pay_order(store)
    _settle_activity_call(
        store, run_id="run-aggregate", call_index=0, actual_micro=600
    )
    _settle_activity_call(
        store, run_id="run-aggregate", call_index=1, actual_micro=684
    )

    items = store.list_ledger_entries(1, limit=50)["items"]
    usage = [item for item in items if item["entry_type"] == "backtest_usage"]

    assert len(usage) == 1
    assert usage[0]["amount_micro"] == -1_284
    assert usage[0]["model_call_count"] == 2
    assert usage[0]["run_id"] == "run-aggregate"
    assert usage[0]["provider_id"] == "openrouter"
    assert usage[0]["model_id"] == "anthropic/claude-haiku-4-5"
    assert usage[0]["provider_mixed"] is False
    assert usage[0]["model_mixed"] is False
    assert usage[0]["billing_source"] == "platform_credits"
    assert "reservation_id" not in usage[0]
    assert "call_index" not in usage[0]
    assert "evidence_json" not in usage[0]
    with store._get_connection() as conn:
        row_count = conn.execute(
            "SELECT COUNT(*) FROM credit_llm_usage_entries WHERE run_id = ?",
            ("run-aggregate",),
        ).fetchone()[0]
    assert row_count == 3


def test_activity_summarizes_mixed_and_malformed_evidence(tmp_path):
    store = _store(tmp_path)
    _pending_order(store)
    _pay_order(store)
    _settle_activity_call(
        store,
        run_id="run-mixed",
        call_index=0,
        actual_micro=100,
        provider_id="openrouter",
        model_id="model-a",
    )
    _settle_activity_call(
        store,
        run_id="run-mixed",
        call_index=1,
        actual_micro=200,
        provider_id="commonstack",
        model_id="model-a",
    )
    _settle_activity_call(
        store,
        run_id="run-unknown",
        call_index=0,
        actual_micro=300,
    )
    with store._get_connection() as conn:
        conn.execute(
            "UPDATE credit_llm_usage_entries SET evidence_json = ? WHERE run_id = ?",
            ("not-json", "run-unknown"),
        )

    items = store.list_ledger_entries(1, limit=50)["items"]
    mixed = next(item for item in items if item.get("run_id") == "run-mixed")
    unknown = next(item for item in items if item.get("run_id") == "run-unknown")

    assert mixed["provider_id"] is None
    assert mixed["model_id"] == "model-a"
    assert mixed["provider_mixed"] is True
    assert mixed["model_mixed"] is False
    assert unknown["amount_micro"] == -300
    assert unknown["provider_id"] is None
    assert unknown["model_id"] is None
    assert unknown["provider_mixed"] is False
    assert unknown["model_mixed"] is False


def test_activity_paginates_whole_runs_and_isolates_equal_run_ids(tmp_path):
    store = _store(tmp_path)
    _pending_order(store)
    _pay_order(store)
    _pending_order(
        store,
        order_id="ord_other",
        user_id=3,
        client_request_id="33333333-3333-4333-8333-333333333333",
    )
    _pay_order(store, order_id="ord_other", event_id="evt_paid_other")
    for call_index, amount in enumerate((101, 102, 103)):
        _settle_activity_call(
            store,
            user_id=1,
            run_id="run-shared",
            call_index=call_index,
            actual_micro=amount,
        )
    _settle_activity_call(
        store,
        user_id=3,
        run_id="run-shared",
        call_index=0,
        actual_micro=900,
    )

    pages = []
    cursor = None
    while True:
        page = store.list_ledger_entries(1, limit=1, cursor=cursor)
        pages.extend(page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            break

    user_one = [item for item in pages if item.get("run_id") == "run-shared"]
    user_three = store.list_ledger_entries(3, limit=50)["items"]
    user_three_usage = next(
        item for item in user_three if item.get("run_id") == "run-shared"
    )
    assert len(user_one) == 1
    assert user_one[0]["amount_micro"] == -306
    assert user_one[0]["model_call_count"] == 3
    assert user_three_usage["amount_micro"] == -900
    assert user_three_usage["model_call_count"] == 1


def test_activity_omits_released_run_without_settled_usage(tmp_path):
    store = _store(tmp_path)
    _pending_order(store)
    _pay_order(store)
    reservation = store.reserve_llm_credits(
        reservation_id="activity-released",
        user_id=1,
        run_id="run-released-without-charge",
        call_index=0,
        provider_id="openrouter",
        attempt_index=0,
        reserved_micro=1_000,
        operation_key="activity-released-reserve",
        request_digest="r" * 64,
    )
    store.release_llm_credits(
        reservation["reservation_id"],
        reason="synthetic provider failure",
    )

    items = store.list_ledger_entries(1, limit=50)["items"]
    assert all(
        item.get("run_id") != "run-released-without-charge" for item in items
    )


def test_partial_refund_posts_negative_entry_and_updates_refundable_amount(tmp_path):
    store = _store(tmp_path)
    _pending_order(store)
    _pay_order(store)
    reservation = store.reserve_refund(
        refund_id="rfnd_4",
        payment_order_id="ord_10",
        user_id=1,
        requested_by_user_id=2,
        amount_usd_cents=400,
        credits_micro=4_000_000,
    )
    assert reservation["status"] == "pending"
    store.attach_stripe_refund("rfnd_4", stripe_refund_id="re_test_4")

    result = store.settle_succeeded_refund(
        event_id="evt_refund_4",
        event_type="refund.created",
        livemode=False,
        object_id="re_test_4",
        payload_sha256="e" * 64,
        refund_id="rfnd_4",
        stripe_refund_id="re_test_4",
        payment_intent_id="pi_test_ord_10",
        currency="usd",
        amount_usd_cents=400,
    )

    assert result["outcome"] == "processed"
    assert result["balance_micro"] == 6_000_000
    assert store.get_order_for_user("ord_10", 1)["status"] == "partially_refunded"
    entries = store.list_ledger_entries(1)["items"]
    assert sorted(entry["amount_micro"] for entry in entries) == [
        -4_000_000,
        10_000_000,
    ]
    assert {entry["bucket"] for entry in entries} == {"purchased"}
    assert {entry["source"] for entry in entries} == {"stripe"}
    admin_order = store.list_orders_for_admin()["items"][0]
    assert admin_order["refundable_credits_micro"] == 6_000_000
    assert admin_order["refundable_usd_cents"] == 600


def test_pending_refund_reserves_amount_and_failure_releases_it(tmp_path):
    store = _store(tmp_path)
    _pending_order(store)
    _pay_order(store)
    store.reserve_refund(
        refund_id="rfnd_7",
        payment_order_id="ord_10",
        user_id=1,
        requested_by_user_id=2,
        amount_usd_cents=700,
        credits_micro=7_000_000,
    )

    with pytest.raises(RefundNotAllowedError, match="unused"):
        store.reserve_refund(
            refund_id="rfnd_4",
            payment_order_id="ord_10",
            user_id=1,
            requested_by_user_id=2,
            amount_usd_cents=400,
            credits_micro=4_000_000,
        )

    store.attach_stripe_refund("rfnd_7", stripe_refund_id="re_test_7")
    failed = store.fail_refund(
        event_id="evt_refund_failed_7",
        event_type="refund.failed",
        livemode=False,
        object_id="re_test_7",
        payload_sha256="f" * 64,
        refund_id="rfnd_7",
        stripe_refund_id="re_test_7",
    )
    assert failed["outcome"] == "processed"
    assert store.get_balance_micro(1) == 10_000_000

    replacement = store.reserve_refund(
        refund_id="rfnd_10",
        payment_order_id="ord_10",
        user_id=1,
        requested_by_user_id=2,
        amount_usd_cents=1000,
        credits_micro=10_000_000,
    )
    assert replacement["status"] == "pending"


def test_concurrent_refund_reservations_cannot_over_refund(tmp_path):
    store = _store(tmp_path)
    _pending_order(store)
    _pay_order(store)

    def reserve(refund_id):
        separate = CreditsStore(db_path=store.db_path)
        try:
            separate.reserve_refund(
                refund_id=refund_id,
                payment_order_id="ord_10",
                user_id=1,
                requested_by_user_id=2,
                amount_usd_cents=700,
                credits_micro=7_000_000,
            )
            return "reserved"
        except RefundNotAllowedError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(reserve, ["rfnd_a", "rfnd_b"]))

    assert sorted(outcomes) == ["rejected", "reserved"]


def test_cross_user_order_read_is_hidden_and_admin_list_is_paginated(tmp_path):
    store = _store(tmp_path)
    _pending_order(store)
    _pay_order(store)
    _pending_order(
        store,
        order_id="ord_other",
        user_id=3,
        client_request_id="33333333-3333-4333-8333-333333333333",
        amount_usd_cents=500,
        credits_micro=5_000_000,
    )
    _pay_order(
        store,
        order_id="ord_other",
        event_id="evt_paid_other",
        amount_usd_cents=500,
    )

    assert store.get_order_for_user("ord_10", 3) is None
    first = store.list_orders_for_admin(limit=1)
    second = store.list_orders_for_admin(limit=1, cursor=first["next_cursor"])
    assert len(first["items"]) == len(second["items"]) == 1
    assert first["items"][0]["id"] != second["items"][0]["id"]
