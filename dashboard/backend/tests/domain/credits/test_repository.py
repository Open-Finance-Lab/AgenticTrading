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
    }.issubset(tables)


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
