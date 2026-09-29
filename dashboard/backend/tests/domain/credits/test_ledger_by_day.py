"""CreditsStore.sum_ledger_by_day: per-UTC-day ledger totals for the /admin charts."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone

import pytest

from dashboard.backend.domain.credits.repository import CreditsStore
from dashboard.backend.domain.credits.repository_common import LedgerDayTotal


UTC = timezone.utc


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
            "INSERT INTO users VALUES (?, ?, ?, 'unused', 'user', '2026-08-01T00:00:00+00:00')",
            [(1, "one@example.com", "One"), (2, "two@example.com", "Two"), (3, "three@example.com", "Three")],
        )
    return CreditsStore(db_path=path)


def _ledger(store, *, user_id, entry_type, amount_micro, created_at, key):
    """A minimal row satisfying the per-type shape CHECK on credit_ledger_entries.

    Amounts are passed with the sign the ledger stores: refunds and reclaims
    negative. That sign is exactly what the per-day query has to get right.
    """
    is_grant = entry_type.startswith("admin_grant")
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            """
            INSERT INTO credit_ledger_entries (
                user_id, bucket, entry_type, amount_micro, payment_order_id,
                refund_request_id, stripe_event_id, operation_key, operation_id,
                idempotency_key, request_digest, actor_user_id, source, reason,
                reference_type, reference_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'test', 'test', ?, ?, ?)
            """,
            (
                user_id,
                "grant" if is_grant else "purchased",
                entry_type,
                amount_micro,
                None if is_grant else f"order-{key}",
                f"refund-{key}" if entry_type == "refund" else None,
                None if is_grant else f"evt-{key}",
                f"op-{key}",
                f"opid-{key}",
                f"idem-{key}",
                f"digest-{key}" if is_grant else None,
                1 if is_grant else None,
                "grant_pool" if is_grant else None,
                "pool-1" if is_grant else None,
                created_at.isoformat(),
            ),
        )


def _usage(store, *, user_id, amount_micro, created_at, key):
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            """
            INSERT INTO credit_llm_usage_entries (
                user_id, reservation_id, run_id, call_index, bucket,
                amount_micro, operation_key, evidence_json, created_at
            ) VALUES (?, ?, 'run-1', 0, 'grant', ?, ?, '{}', ?)
            """,
            (user_id, f"res-{key}", amount_micro, f"usage-{key}", created_at.isoformat()),
        )


DAY_ONE = datetime(2026, 9, 10, 8, 0, tzinfo=UTC)
DAY_TWO = datetime(2026, 9, 11, 23, 59, tzinfo=UTC)
WINDOW = {"start": datetime(2026, 9, 10, tzinfo=UTC), "end": datetime(2026, 9, 12, tzinfo=UTC)}


def test_refunds_are_positive_magnitudes_and_grants_are_excluded(tmp_path):
    store = _store(tmp_path)
    _ledger(store, user_id=1, entry_type="purchase", amount_micro=50_000_000, created_at=DAY_ONE, key="p1")
    _ledger(store, user_id=1, entry_type="refund", amount_micro=-20_000_000, created_at=DAY_ONE, key="r1")
    _ledger(store, user_id=2, entry_type="purchase", amount_micro=1_000_000, created_at=DAY_TWO, key="p2")
    _ledger(store, user_id=1, entry_type="admin_grant_assign", amount_micro=100_000_000, created_at=DAY_TWO, key="g1")
    _ledger(store, user_id=1, entry_type="admin_grant_reclaim", amount_micro=-40_000_000, created_at=DAY_TWO, key="g2")
    _usage(store, user_id=1, amount_micro=-250_000, created_at=DAY_ONE, key="u1")
    _usage(store, user_id=2, amount_micro=-750_000, created_at=DAY_TWO, key="u2")

    rows = store.sum_ledger_by_day([1, 2], **WINDOW)

    assert rows == [
        # The PR's first cut negated the already-negative refund and reported 70.
        LedgerDayTotal(day=date(2026, 9, 10), purchased_micro=50_000_000, refunded_micro=20_000_000, consumed_micro=250_000),
        # 100 Credits granted and 40 reclaimed move neither revenue nor consumption.
        LedgerDayTotal(day=date(2026, 9, 11), purchased_micro=1_000_000, refunded_micro=0, consumed_micro=750_000),
    ]


def test_series_matches_the_commercial_headline_it_is_charted_under(tmp_path):
    store = _store(tmp_path)
    _ledger(store, user_id=1, entry_type="purchase", amount_micro=5_000_000, created_at=DAY_ONE, key="p1")
    _ledger(store, user_id=1, entry_type="refund", amount_micro=-2_000_000, created_at=DAY_TWO, key="r1")
    _ledger(store, user_id=2, entry_type="admin_grant_assign", amount_micro=3_000_000, created_at=DAY_TWO, key="g1")
    _usage(store, user_id=2, amount_micro=-600_000, created_at=DAY_TWO, key="u1")

    rows = store.sum_ledger_by_day([1, 2], **WINDOW)
    headline = store.aggregate_commercial_ledger([1, 2], **WINDOW)

    for field in ("purchased_micro", "refunded_micro", "consumed_micro"):
        assert sum(getattr(row, field) for row in rows) == sum(
            totals[field] for totals in headline.values()
        ), field


def test_scoped_to_user_ids_and_the_half_open_window(tmp_path):
    store = _store(tmp_path)
    _ledger(store, user_id=1, entry_type="purchase", amount_micro=1_000_000, created_at=DAY_ONE, key="in")
    _ledger(store, user_id=3, entry_type="purchase", amount_micro=9_000_000, created_at=DAY_ONE, key="other-user")
    _ledger(store, user_id=1, entry_type="purchase", amount_micro=7_000_000, created_at=WINDOW["end"], key="at-end")
    _ledger(store, user_id=1, entry_type="purchase", amount_micro=8_000_000, created_at=WINDOW["start"] - timedelta(microseconds=1), key="before")

    rows = store.sum_ledger_by_day([1], **WINDOW)

    assert rows == [LedgerDayTotal(day=date(2026, 9, 10), purchased_micro=1_000_000)]
    assert store.sum_ledger_by_day([], **WINDOW) == []


def test_rejects_a_reversed_or_naive_window(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        store.sum_ledger_by_day([1], start=WINDOW["end"], end=WINDOW["start"])
    with pytest.raises(ValueError):
        store.sum_ledger_by_day([1], start=datetime(2026, 9, 10), end=WINDOW["end"])
