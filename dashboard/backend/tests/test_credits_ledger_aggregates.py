"""The credits domain answers the analytics domain's ledger questions itself.

Design SS6.14: a domain reads another domain's *service*, never its tables.
Until PR A, ``value_repository.py`` opened ``credits_store._get_connection()``
and ran its own SQL against the ledger; these four methods are that SQL, moved
onto the store that owns the tables.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import date, datetime, timedelta, timezone

import pytest

from dashboard.backend.domain.credits.repository import CreditsStore
from dashboard.backend.tests._postgres_testing import require_local_postgres_url
from dashboard.backend.users import UserStore


NOW = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)
DAY = date(2026, 9, 11)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _ledger_row(user_id, entry_type, amount_micro, created_at, *, key):
    """One row satisfying credit_ledger_entries' CHECK for its entry_type."""
    base = {
        "user_id": user_id,
        "entry_type": entry_type,
        "amount_micro": amount_micro,
        "operation_key": f"op:{key}",
        "operation_id": f"operation-{key}",
        "idempotency_key": f"idem:{key}",
        "source": "test",
        "reason": "ledger aggregate test",
        "created_at": _iso(created_at),
        "payment_order_id": None,
        "refund_request_id": None,
        "stripe_event_id": None,
        "request_digest": None,
        "actor_user_id": None,
        "reference_type": None,
        "reference_id": None,
    }
    if entry_type == "purchase":
        base.update(
            bucket="purchased", payment_order_id=f"order-{key}",
            stripe_event_id=f"evt-{key}",
        )
    elif entry_type == "refund":
        base.update(
            bucket="purchased", payment_order_id=f"order-{key}",
            refund_request_id=f"refund-{key}", stripe_event_id=f"evt-{key}",
        )
    else:  # admin_grant_assign / admin_grant_reclaim
        base.update(
            bucket="grant", request_digest="digest", actor_user_id=1,
            reference_type="grant_pool", reference_id="default",
        )
    return base


def _insert(path, table, rows):
    # A bare connection: FKs are off by default in sqlite3, so the payment
    # order / refund / stripe event FKs need no parent rows here. The CHECK
    # constraints still apply, which is why _ledger_row fills every column the
    # constraint for its entry_type inspects.
    with sqlite3.connect(path) as conn:
        for row in rows:
            columns = ", ".join(row)
            marks = ", ".join("?" for _ in row)
            conn.execute(
                f"INSERT INTO {table} ({columns}) VALUES ({marks})",
                tuple(row.values()),
            )


def _usage_row(user_id, amount_micro, created_at, *, key, bucket="grant"):
    return {
        "user_id": user_id,
        "reservation_id": f"res-{key}",
        "run_id": f"run-{key}",
        "call_index": 0,
        "bucket": bucket,
        "amount_micro": amount_micro,
        "operation_key": f"settle:{key}",
        "evidence_json": "{}",
        "created_at": _iso(created_at),
    }


def _store(tmp_path):
    path = tmp_path / "credits.db"
    users = UserStore(db_path=path)
    alice = int(users.create_user("alice@example.test", "Alice", "SecurePass1!")["id"])
    bob = int(users.create_user("bob@example.test", "Bob", "SecurePass1!")["id"])
    store = CreditsStore(path)
    day_start = datetime.combine(DAY, datetime.min.time(), tzinfo=timezone.utc)
    _insert(
        path,
        "credit_ledger_entries",
        [
            _ledger_row(alice, "purchase", 10_000_000, NOW - timedelta(days=40), key="a1"),
            _ledger_row(alice, "purchase", 3_000_000, day_start + timedelta(hours=1), key="a2"),
            _ledger_row(alice, "refund", -6_000_000, day_start + timedelta(hours=2), key="a3"),
            _ledger_row(alice, "admin_grant_assign", 1_500_000, day_start + timedelta(hours=3), key="a4"),
            _ledger_row(alice, "admin_grant_reclaim", -500_000, day_start + timedelta(hours=4), key="a5"),
            _ledger_row(bob, "purchase", 2_000_000, NOW - timedelta(days=3), key="b1"),
        ],
    )
    _insert(
        path,
        "credit_llm_usage_entries",
        [
            _usage_row(alice, -400_000, day_start + timedelta(hours=5), key="a6"),
            _usage_row(alice, -100_000, day_start + timedelta(hours=6), key="a7", bucket="purchased"),
            _usage_row(alice, -50_000, day_start + timedelta(days=1, hours=1), key="a8"),
            _usage_row(bob, -25_000, day_start + timedelta(hours=9), key="b2"),
        ],
    )
    return store, alice, bob, day_start


def test_aggregate_commercial_ledger_matches_the_reader_it_replaces(tmp_path):
    store, alice, bob, day_start = _store(tmp_path)

    totals = store.aggregate_commercial_ledger(
        [alice, bob], start=day_start, end=day_start + timedelta(days=1)
    )

    assert totals[alice] == {
        "lifetime_purchased_micro": 13_000_000,
        "lifetime_refunded_micro": 6_000_000,
        "purchased_micro": 3_000_000,
        "refunded_micro": 6_000_000,
        "grant_activity_micro": 2_000_000,
        "consumed_micro": 500_000,
    }
    # Bob bought outside the window and consumed inside it.
    assert totals[bob]["lifetime_purchased_micro"] == 2_000_000
    assert totals[bob]["purchased_micro"] == 0
    assert totals[bob]["consumed_micro"] == 25_000
    assert store.aggregate_commercial_ledger([], start=day_start, end=NOW) == {}


def test_list_credit_activity_timestamps_returns_purchases_and_consumption_only(tmp_path):
    store, alice, _bob, day_start = _store(tmp_path)

    stamps = store.list_credit_activity_timestamps(
        [alice], start=day_start, end=day_start + timedelta(days=1)
    )

    assert sorted(stamps[alice]) == [
        _iso(day_start + timedelta(hours=1)),  # the purchase
        _iso(day_start + timedelta(hours=5)),  # consumption
        _iso(day_start + timedelta(hours=6)),  # consumption
    ]


def test_aggregate_ledger_for_day_takes_no_user_id(tmp_path):
    store, alice, bob, day_start = _store(tmp_path)

    totals = store.aggregate_ledger_for_day(DAY)

    assert totals[alice]["own_spend_micro"] == 500_000
    assert totals[alice]["lifetime_net_purchased_micro"] == 7_000_000
    assert totals[alice]["last_activity_at"] == _iso(day_start + timedelta(hours=6))
    assert totals[bob]["own_spend_micro"] == 25_000
    assert totals[bob]["lifetime_net_purchased_micro"] == 2_000_000
    assert totals[bob]["last_activity_at"] == _iso(day_start + timedelta(hours=9))
    # The day after has only Alice's 50_000 consumption and no purchases.
    next_day = store.aggregate_ledger_for_day(DAY + timedelta(days=1))
    assert next_day[alice]["own_spend_micro"] == 50_000
    assert next_day[alice]["lifetime_net_purchased_micro"] == 7_000_000


def test_list_account_billing_states_agrees_with_the_single_user_reader(tmp_path):
    store, alice, bob, _day_start = _store(tmp_path)
    store.restrict_account(alice, reason="llm_overage")
    store.ensure_account(bob)

    batched = store.list_account_billing_states([alice, bob])
    everyone = store.list_account_billing_states()

    for user_id in (alice, bob):
        assert batched[user_id] == store.get_account_billing_state(user_id)
    assert everyone == batched
    assert batched[alice]["account_status"] == "restricted"
    assert batched[alice]["restriction_reason"] == "llm_overage"
    assert store.list_account_billing_states([]) == {}


def test_backfill_source_rows_come_from_the_credits_store(tmp_path):
    store, alice, _bob, day_start = _store(tmp_path)
    _insert(
        tmp_path / "credits.db",
        "credit_llm_reservations",
        [
            {
                "reservation_id": "res-a6",
                "user_id": alice,
                "run_id": "run-a6",
                "call_index": 0,
                "reserved_micro": 500_000,
                "reserved_grant_micro": 500_000,
                "reserved_purchased_micro": 0,
                "status": "settled",
                "operation_key": "reserve:a6",
                "request_digest": "digest",
                "created_at": _iso(day_start + timedelta(hours=4)),
                "updated_at": _iso(day_start + timedelta(hours=5)),
            }
        ],
    )

    reservations = store.list_llm_reservation_rows()
    usage = store.list_llm_usage_rows()

    assert [row["reservation_id"] for row in reservations] == ["res-a6"]
    assert set(reservations[0]) == {
        "reservation_id", "user_id", "run_id", "call_index",
        "reserved_grant_micro", "reserved_purchased_micro", "status",
        "created_at", "updated_at",
    }
    assert [row["reservation_id"] for row in usage] == ["res-a6", "res-a7", "res-b2", "res-a8"]
    assert set(usage[0]) == {
        "id", "user_id", "reservation_id", "run_id", "call_index", "bucket",
        "amount_micro", "created_at",
    }


# Alice's window day nets a 3.0 purchase against a 6.0 refund (refund rows
# are stored negative), ignores both grant rows, and consumes 0.5; the next
# day she consumes 0.05. Bob consumes 0.025 on the window day.
DAILY_EXPECTED = [
    {"day_string": "2026-09-11", "purchases_micro": -3_000_000, "consumption_micro": 525_000},
    {"day_string": "2026-09-12", "purchases_micro": 0, "consumption_micro": 50_000},
]


def test_list_daily_purchase_and_consumption_nets_refunds_and_skips_grants(tmp_path):
    store, alice, bob, day_start = _store(tmp_path)

    rows = store.list_daily_purchase_and_consumption(
        [alice, bob], start=day_start, end=day_start + timedelta(days=2)
    )

    assert rows == DAILY_EXPECTED
    assert store.list_daily_purchase_and_consumption([], start=day_start, end=NOW) == []


# --- the Postgres twin, on the live tier -------------------------------------

TEST_POSTGRES_URL = os.getenv("TEST_POSTGRES_URL")
pg_only = pytest.mark.skipif(
    not TEST_POSTGRES_URL,
    reason="TEST_POSTGRES_URL not set; skipping live-Postgres tests",
)
_PG_ALICE, _PG_BOB = 910_001, 910_002


def _pg_delete(conn):
    with conn.cursor() as cur:
        ids = [_PG_ALICE, _PG_BOB]
        cur.execute("DELETE FROM credit_llm_usage_entries WHERE user_id = ANY(%s)", (ids,))
        cur.execute("DELETE FROM credit_ledger_entries WHERE user_id = ANY(%s)", (ids,))


@pg_only
def test_postgres_daily_purchase_and_consumption_matches_the_sqlite_twin():
    require_local_postgres_url(TEST_POSTGRES_URL)
    from dashboard.backend.domain.credits.repository_postgres import PostgresCreditsStore
    from dashboard.backend.users_postgres import PostgresUserStore

    PostgresUserStore(TEST_POSTGRES_URL)  # the credits DDL references users
    store = PostgresCreditsStore(TEST_POSTGRES_URL)
    day_start = datetime.combine(DAY, datetime.min.time(), tzinfo=timezone.utc)
    ledger = [
        _ledger_row(_PG_ALICE, "purchase", 10_000_000, NOW - timedelta(days=40), key="pa1"),
        _ledger_row(_PG_ALICE, "purchase", 3_000_000, day_start + timedelta(hours=1), key="pa2"),
        _ledger_row(_PG_ALICE, "refund", -6_000_000, day_start + timedelta(hours=2), key="pa3"),
        _ledger_row(_PG_ALICE, "admin_grant_assign", 1_500_000, day_start + timedelta(hours=3), key="pa4"),
        _ledger_row(_PG_ALICE, "admin_grant_reclaim", -500_000, day_start + timedelta(hours=4), key="pa5"),
        _ledger_row(_PG_BOB, "purchase", 2_000_000, NOW - timedelta(days=3), key="pb1"),
    ]
    usage = [
        _usage_row(_PG_ALICE, -400_000, day_start + timedelta(hours=5), key="pa6"),
        _usage_row(_PG_ALICE, -100_000, day_start + timedelta(hours=6), key="pa7", bucket="purchased"),
        _usage_row(_PG_ALICE, -50_000, day_start + timedelta(days=1, hours=1), key="pa8"),
        _usage_row(_PG_BOB, -25_000, day_start + timedelta(hours=9), key="pb2"),
    ]
    with store._get_connection() as conn:
        _pg_delete(conn)
        with conn.cursor() as cur:
            # Same shortcut as the SQLite fixture's FK-less connection: the
            # CHECK constraints still apply, the parent rows (users, orders,
            # refunds, stripe events, reservations) are not needed.
            cur.execute("SET LOCAL session_replication_role = replica")
            for table, rows in (("credit_ledger_entries", ledger), ("credit_llm_usage_entries", usage)):
                for row in rows:
                    cur.execute(
                        f"INSERT INTO {table} ({', '.join(row)}) VALUES ({', '.join('%s' for _ in row)})",
                        tuple(row.values()),
                    )
    try:
        rows = store.list_daily_purchase_and_consumption(
            [_PG_ALICE, _PG_BOB], start=day_start, end=day_start + timedelta(days=2)
        )
        assert rows == DAILY_EXPECTED
    finally:
        with store._get_connection() as conn:
            _pg_delete(conn)
