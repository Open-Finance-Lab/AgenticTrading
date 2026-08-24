"""Shared accounting contract for the SQLite Grant Credits repository."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from dashboard.backend.domain.credits import repository as repository_module
from dashboard.backend.domain.credits.repository import CreditsStore
from dashboard.backend.domain.credits.repository_common import (
    CreditAccountRestrictedStoreError,
    GrantPoolInsufficientError,
    GrantReclaimExceedsAvailableError,
    IdempotencyConflictError,
)


USER_ID = 1
OTHER_USER_ID = 2
ADMIN_ID = 3
MONTH_START_ISO = "2026-08-01T00:00:00+00:00"


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
            )
            VALUES (?, ?, ?, 'unused', ?, '2026-08-01T00:00:00+00:00')
            """,
            [
                (USER_ID, "user@example.com", "User", "user"),
                (OTHER_USER_ID, "other@example.com", "Other", "user"),
                (ADMIN_ID, "admin@example.com", "Admin", "admin"),
            ],
        )


@pytest.fixture
def store(tmp_path) -> CreditsStore:
    path = tmp_path / "credits.db"
    _seed_users(path)
    return CreditsStore(path)


def _mutation_args(
    operation: str,
    *,
    amount_micro: int,
    digest: str | None = None,
) -> dict[str, object]:
    return {
        "pool_id": "default",
        "amount_micro": amount_micro,
        "operation_id": f"grant_{operation}",
        "idempotency_key": f"request_{operation}",
        "request_digest": digest or f"digest_{operation}",
        "actor_user_id": ADMIN_ID,
        "source": "research_budget",
        "reason": f"Audit reason for {operation}.",
    }


def _fund(store: CreditsStore, amount_micro: int = 10_000_000) -> dict:
    return store.fund_grant_pool(**_mutation_args("fund", amount_micro=amount_micro))


def _purchase(store: CreditsStore, amount_micro: int = 2_000_000) -> None:
    amount_usd_cents = amount_micro // 10_000
    order = store.create_or_get_order(
        order_id="ord_purchase",
        user_id=USER_ID,
        client_request_id="11111111-1111-4111-8111-111111111111",
        amount_usd_cents=amount_usd_cents,
        credits_micro=amount_micro,
    )
    store.attach_checkout_session(order["id"], checkout_session_id="cs_purchase")
    result = store.settle_paid_checkout(
        event_id="evt_purchase",
        event_type="checkout.session.completed",
        livemode=False,
        object_id="cs_purchase",
        payload_sha256="a" * 64,
        order_id=order["id"],
        checkout_session_id="cs_purchase",
        payment_intent_id="pi_purchase",
        currency="usd",
        amount_usd_cents=amount_usd_cents,
    )
    assert result["outcome"] == "processed"


def _seed_legacy_credits_schema(path) -> None:
    """Create the exact pre-Grant Credits journal without the new store helper."""

    _seed_users(path)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(
            """
            CREATE TABLE credit_accounts (
                user_id INTEGER PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'restricted')),
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE credit_payment_orders (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                user_id INTEGER NOT NULL,
                client_request_id TEXT NOT NULL,
                stripe_mode TEXT NOT NULL DEFAULT 'test'
                    CHECK (stripe_mode IN ('test', 'live')),
                currency TEXT NOT NULL DEFAULT 'usd',
                amount_usd_cents INTEGER NOT NULL CHECK (amount_usd_cents > 0),
                credits_micro INTEGER NOT NULL CHECK (credits_micro > 0),
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN (
                        'pending', 'paid', 'expired', 'failed',
                        'partially_refunded', 'refunded'
                    )),
                stripe_checkout_session_id TEXT UNIQUE,
                stripe_payment_intent_id TEXT UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                paid_at TEXT,
                UNIQUE (user_id, client_request_id),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES credit_accounts(user_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE credit_refund_requests (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                payment_order_id TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                requested_by_user_id INTEGER,
                amount_usd_cents INTEGER NOT NULL CHECK (amount_usd_cents > 0),
                credits_micro INTEGER NOT NULL CHECK (credits_micro > 0),
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN (
                        'pending', 'submitted', 'succeeded', 'failed', 'cancelled'
                    )),
                stripe_refund_id TEXT UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                succeeded_at TEXT,
                FOREIGN KEY (payment_order_id)
                    REFERENCES credit_payment_orders(id) ON DELETE RESTRICT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (requested_by_user_id)
                    REFERENCES users(id) ON DELETE RESTRICT
            );

            CREATE TABLE stripe_webhook_events (
                stripe_event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                livemode INTEGER NOT NULL CHECK (livemode IN (0, 1)),
                object_id TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                outcome TEXT NOT NULL
                    CHECK (outcome IN ('processed', 'ignored', 'rejected')),
                reason TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE credit_ledger_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                entry_type TEXT NOT NULL CHECK (entry_type IN ('purchase', 'refund')),
                amount_micro INTEGER NOT NULL CHECK (amount_micro <> 0),
                payment_order_id TEXT NOT NULL,
                refund_request_id TEXT,
                stripe_event_id TEXT NOT NULL,
                operation_key TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (payment_order_id)
                    REFERENCES credit_payment_orders(id) ON DELETE RESTRICT,
                FOREIGN KEY (refund_request_id)
                    REFERENCES credit_refund_requests(id) ON DELETE RESTRICT,
                FOREIGN KEY (stripe_event_id)
                    REFERENCES stripe_webhook_events(stripe_event_id)
                    ON DELETE RESTRICT
            );

            CREATE INDEX idx_credit_payment_orders_user_sequence
            ON credit_payment_orders(user_id, sequence DESC);

            CREATE INDEX idx_credit_refunds_order_status
            ON credit_refund_requests(payment_order_id, status);

            CREATE INDEX idx_credit_ledger_user_id
            ON credit_ledger_entries(user_id, id DESC);

            CREATE INDEX idx_credit_ledger_payment_order
            ON credit_ledger_entries(payment_order_id, id DESC);
            """
        )
        conn.execute(
            """
            INSERT INTO credit_accounts (user_id, status, created_at)
            VALUES (?, 'active', '2026-07-01T00:00:00+00:00')
            """,
            (USER_ID,),
        )
        conn.execute(
            """
            INSERT INTO credit_payment_orders (
                id, user_id, client_request_id, amount_usd_cents, credits_micro,
                status, stripe_checkout_session_id, stripe_payment_intent_id,
                created_at, updated_at, paid_at
            )
            VALUES (
                'ord_legacy', ?, 'legacy-request', 1000, 10000000,
                'partially_refunded', 'cs_legacy', 'pi_legacy',
                '2026-07-01T00:00:00+00:00',
                '2026-07-02T00:00:00+00:00',
                '2026-07-01T00:00:00+00:00'
            )
            """,
            (USER_ID,),
        )
        conn.execute(
            """
            INSERT INTO credit_refund_requests (
                id, payment_order_id, user_id, requested_by_user_id,
                amount_usd_cents, credits_micro, status, stripe_refund_id,
                created_at, updated_at, succeeded_at
            )
            VALUES (
                'rfnd_legacy', 'ord_legacy', ?, ?, 200, 2000000,
                'succeeded', 're_legacy',
                '2026-07-02T00:00:00+00:00',
                '2026-07-02T00:00:00+00:00',
                '2026-07-02T00:00:00+00:00'
            )
            """,
            (USER_ID, ADMIN_ID),
        )
        conn.executemany(
            """
            INSERT INTO stripe_webhook_events (
                stripe_event_id, event_type, livemode, object_id,
                payload_sha256, outcome, created_at
            )
            VALUES (?, ?, 0, ?, ?, 'processed', ?)
            """,
            [
                (
                    "evt_legacy_purchase",
                    "checkout.session.completed",
                    "cs_legacy",
                    "a" * 64,
                    "2026-07-01T00:00:00+00:00",
                ),
                (
                    "evt_legacy_refund",
                    "refund.created",
                    "re_legacy",
                    "b" * 64,
                    "2026-07-02T00:00:00+00:00",
                ),
            ],
        )
        conn.executemany(
            """
            INSERT INTO credit_ledger_entries (
                id, user_id, entry_type, amount_micro, payment_order_id,
                refund_request_id, stripe_event_id, operation_key, created_at
            )
            VALUES (?, ?, ?, ?, 'ord_legacy', ?, ?, ?, ?)
            """,
            [
                (
                    1,
                    USER_ID,
                    "purchase",
                    10_000_000,
                    None,
                    "evt_legacy_purchase",
                    "purchase:ord_legacy",
                    "2026-07-01T00:00:00+00:00",
                ),
                (
                    2,
                    USER_ID,
                    "refund",
                    -2_000_000,
                    "rfnd_legacy",
                    "evt_legacy_refund",
                    "refund:rfnd_legacy",
                    "2026-07-02T00:00:00+00:00",
                ),
            ],
        )


def test_sqlite_migration_preserves_existing_stripe_journal(tmp_path):
    path = tmp_path / "legacy.db"
    _seed_legacy_credits_schema(path)

    store = CreditsStore(path)
    page = store.list_ledger_entries(USER_ID, limit=50)

    assert [
        (
            row["id"],
            row["bucket"],
            row["entry_type"],
            row["amount_micro"],
            row["payment_order_id"],
            row["refund_request_id"],
            row["stripe_event_id"],
            row["operation_key"],
            row["created_at"],
        )
        for row in page["items"]
    ] == [
        (
            2,
            "purchased",
            "refund",
            -2_000_000,
            "ord_legacy",
            "rfnd_legacy",
            "evt_legacy_refund",
            "refund:rfnd_legacy",
            "2026-07-02T00:00:00+00:00",
        ),
        (
            1,
            "purchased",
            "purchase",
            10_000_000,
            "ord_legacy",
            None,
            "evt_legacy_purchase",
            "purchase:ord_legacy",
            "2026-07-01T00:00:00+00:00",
        ),
    ]
    assert all(row["request_digest"] is None for row in page["items"])
    assert all(row["actor_user_id"] is None for row in page["items"])
    assert [row["operation_id"] for row in page["items"]] == [
        "refund:rfnd_legacy",
        "purchase:ord_legacy",
    ]
    assert [row["idempotency_key"] for row in page["items"]] == [
        "refund:rfnd_legacy",
        "purchase:ord_legacy",
    ]
    assert [row["source"] for row in page["items"]] == ["stripe", "stripe"]
    assert [row["reason"] for row in page["items"]] == [
        "Historical Stripe refund.",
        "Historical Stripe purchase.",
    ]
    assert store.get_balance_micro(USER_ID) == 8_000_000


def _remove_sqlite_grant_pool_snapshots(path) -> None:
    """Recreate the exact pre-snapshot pool ledger while preserving its rows."""

    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.executescript(
            """
            ALTER TABLE credit_grant_pool_ledger_entries
            RENAME TO credit_grant_pool_ledger_entries_pre_snapshot;

            CREATE TABLE credit_grant_pool_ledger_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pool_id TEXT NOT NULL,
                entry_type TEXT NOT NULL
                    CHECK (entry_type IN ('fund', 'reduce', 'assign', 'reclaim')),
                amount_micro INTEGER NOT NULL CHECK (amount_micro <> 0),
                operation_id TEXT NOT NULL UNIQUE
                    CHECK (length(trim(operation_id)) > 0),
                idempotency_key TEXT NOT NULL UNIQUE
                    CHECK (length(trim(idempotency_key)) > 0),
                request_digest TEXT NOT NULL
                    CHECK (length(trim(request_digest)) > 0),
                actor_user_id INTEGER NOT NULL,
                source TEXT NOT NULL CHECK (length(trim(source)) > 0),
                reason TEXT NOT NULL CHECK (length(trim(reason)) > 0),
                user_id INTEGER,
                user_ledger_entry_id INTEGER UNIQUE,
                created_at TEXT NOT NULL,
                FOREIGN KEY (pool_id)
                    REFERENCES credit_grant_pools(pool_id) ON DELETE RESTRICT,
                FOREIGN KEY (actor_user_id)
                    REFERENCES users(id) ON DELETE RESTRICT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE RESTRICT,
                FOREIGN KEY (user_ledger_entry_id)
                    REFERENCES credit_ledger_entries(id) ON DELETE RESTRICT,
                CHECK (
                    (entry_type = 'fund' AND amount_micro > 0
                        AND user_id IS NULL AND user_ledger_entry_id IS NULL)
                    OR (entry_type = 'reduce' AND amount_micro < 0
                        AND user_id IS NULL AND user_ledger_entry_id IS NULL)
                    OR (entry_type = 'assign' AND amount_micro < 0
                        AND user_id IS NOT NULL AND user_ledger_entry_id IS NOT NULL)
                    OR (entry_type = 'reclaim' AND amount_micro > 0
                        AND user_id IS NOT NULL AND user_ledger_entry_id IS NOT NULL)
                )
            );

            INSERT INTO credit_grant_pool_ledger_entries (
                id, pool_id, entry_type, amount_micro, operation_id,
                idempotency_key, request_digest, actor_user_id, source,
                reason, user_id, user_ledger_entry_id, created_at
            )
            SELECT
                id, pool_id, entry_type, amount_micro, operation_id,
                idempotency_key, request_digest, actor_user_id, source,
                reason, user_id, user_ledger_entry_id, created_at
            FROM credit_grant_pool_ledger_entries_pre_snapshot
            ORDER BY id;

            DROP TABLE credit_grant_pool_ledger_entries_pre_snapshot;

            CREATE INDEX idx_credit_grant_pool_ledger_pool_id
            ON credit_grant_pool_ledger_entries(pool_id, id DESC);
            """
        )


def test_sqlite_migrates_pre_snapshot_grant_pool_ledger(tmp_path):
    path = tmp_path / "pre-snapshot-grants.db"
    _seed_users(path)
    store = CreditsStore(path)
    fund_command = _mutation_args("upgrade_fund", amount_micro=10_000_000)
    assign_command = _mutation_args("upgrade_assign", amount_micro=3_000_000)
    funded = store.fund_grant_pool(**fund_command)
    assigned = store.assign_grant(user_id=USER_ID, **assign_command)

    with store._get_connection() as conn:
        conn.execute(
            """
            UPDATE credit_grant_pools
            SET name = 'Legacy Research Pool', status = 'disabled'
            WHERE pool_id = 'default'
            """
        )
    _remove_sqlite_grant_pool_snapshots(path)

    upgraded = CreditsStore(path)
    with upgraded._get_connection() as conn:
        columns = {
            row["name"]: row
            for row in conn.execute(
                "PRAGMA table_info(credit_grant_pool_ledger_entries)"
            )
        }
        rows = conn.execute(
            """
            SELECT * FROM credit_grant_pool_ledger_entries ORDER BY id
            """
        ).fetchall()

    assert columns["pool_name_snapshot"]["notnull"] == 1
    assert columns["pool_status_snapshot"]["notnull"] == 1
    assert [row["id"] for row in rows] == [
        funded["entry"]["id"],
        assigned["entry"]["id"],
    ]
    assert [row["amount_micro"] for row in rows] == [10_000_000, -3_000_000]
    assert rows[1]["user_ledger_entry_id"] == assigned["user_entry"]["id"]
    assert all(row["pool_name_snapshot"] == "Legacy Research Pool" for row in rows)
    assert all(row["pool_status_snapshot"] == "disabled" for row in rows)

    with upgraded._get_connection() as conn:
        conn.execute(
            """
            UPDATE credit_grant_pools
            SET name = 'Current Pool', status = 'active'
            WHERE pool_id = 'default'
            """
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                UPDATE credit_grant_pool_ledger_entries
                SET pool_name_snapshot = ' '
                WHERE id = ?
                """,
                (funded["entry"]["id"],),
            )

    replayed = upgraded.assign_grant(user_id=USER_ID, **assign_command)
    assert replayed["entry"]["id"] == assigned["entry"]["id"]
    assert replayed["pool"]["name"] == "Legacy Research Pool"
    assert replayed["pool"]["status"] == "disabled"
    assert replayed["pool"]["balance_micro"] == 7_000_000
    assert replayed["user_balance"]["grant_available_micro"] == 3_000_000


def test_sqlite_ledger_schema_enforces_entry_shapes(store):
    _purchase(store)
    with store._get_connection() as conn:
        base = (
            USER_ID,
            "grant",
            "admin_grant_assign",
            1_000_000,
            None,
            None,
            None,
            "invalid:grant",
            "invalid_grant",
            "invalid_grant",
            None,
            ADMIN_ID,
            "admin",
            "Missing digest.",
            "2026-08-01T00:00:00+00:00",
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO credit_ledger_entries (
                    user_id, bucket, entry_type, amount_micro, payment_order_id,
                    refund_request_id, stripe_event_id, operation_key,
                    operation_id, idempotency_key, request_digest, actor_user_id,
                    source, reason, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                base,
            )

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO credit_ledger_entries (
                    user_id, bucket, entry_type, amount_micro, payment_order_id,
                    refund_request_id, stripe_event_id, operation_key,
                    operation_id, idempotency_key, request_digest, actor_user_id,
                    source, reason, created_at
                )
                VALUES (
                    ?, 'purchased', 'purchase', 1000000, NULL, NULL, NULL,
                    'invalid:purchase', 'invalid_purchase', 'invalid_purchase',
                    NULL, NULL, 'stripe', 'Missing Stripe evidence.',
                    '2026-08-01T00:00:00+00:00'
                )
                """,
                (USER_ID,),
            )


def assert_four_operations_are_signed_and_paired(store) -> None:
    funded = _fund(store)
    reduced = store.reduce_grant_pool(
        **_mutation_args("reduce", amount_micro=1_000_000)
    )
    assigned = store.assign_grant(
        user_id=USER_ID,
        **_mutation_args("assign", amount_micro=4_000_000),
    )
    reclaimed = store.reclaim_grant(
        user_id=USER_ID,
        **_mutation_args("reclaim", amount_micro=1_500_000),
    )

    assert funded["entry"]["entry_type"] == "fund"
    assert funded["entry"]["amount_micro"] == 10_000_000
    assert reduced["entry"]["entry_type"] == "reduce"
    assert reduced["entry"]["amount_micro"] == -1_000_000
    assert assigned["entry"]["amount_micro"] == -4_000_000
    assert assigned["user_entry"]["amount_micro"] == 4_000_000
    assert reclaimed["entry"]["amount_micro"] == 1_500_000
    assert reclaimed["user_entry"]["amount_micro"] == -1_500_000
    assert assigned["entry"]["operation_id"] == assigned["user_entry"]["operation_id"]
    assert reclaimed["entry"]["operation_id"] == reclaimed["user_entry"]["operation_id"]
    assert reclaimed["pool"]["balance_micro"] == 6_500_000
    assert reclaimed["user_balance"]["grant_available_micro"] == 2_500_000


def test_four_operations_are_signed_and_paired(store):
    assert_four_operations_are_signed_and_paired(store)


def assert_grant_mutations_leave_purchased_balance_unchanged(store) -> None:
    _purchase(store, 2_000_000)
    purchased_before = store.get_balance_projection(USER_ID)[
        "purchased_committed_micro"
    ]
    _fund(store)

    store.assign_grant(
        user_id=USER_ID,
        **_mutation_args("isolation_assign", amount_micro=3_000_000),
    )
    store.reclaim_grant(
        user_id=USER_ID,
        **_mutation_args("isolation_reclaim", amount_micro=1_000_000),
    )

    projection = store.get_balance_projection(USER_ID)
    assert projection["purchased_committed_micro"] == purchased_before
    assert projection["grant_committed_micro"] == 2_000_000
    assert projection["total_available_micro"] == 4_000_000
    assert store.get_balance_micro(USER_ID) == 4_000_000


def test_grant_mutations_leave_purchased_balance_unchanged(store):
    assert_grant_mutations_leave_purchased_balance_unchanged(store)


def test_overdrafts_are_rejected_without_changing_ledgers(store):
    _fund(store, 2_000_000)

    with pytest.raises(GrantPoolInsufficientError):
        store.reduce_grant_pool(
            **_mutation_args("reduce_too_much", amount_micro=2_000_001)
        )
    with pytest.raises(GrantPoolInsufficientError):
        store.assign_grant(
            user_id=USER_ID,
            **_mutation_args("assign_too_much", amount_micro=2_000_001),
        )

    store.assign_grant(
        user_id=USER_ID,
        **_mutation_args("assign_one", amount_micro=1_000_000),
    )
    with pytest.raises(GrantReclaimExceedsAvailableError):
        store.reclaim_grant(
            user_id=USER_ID,
            **_mutation_args("reclaim_too_much", amount_micro=1_000_001),
        )

    summary = store.get_grant_pool_summary("default", MONTH_START_ISO)
    assert summary["pool_available_micro"] == 1_000_000
    assert summary["allocated_to_users_micro"] == 1_000_000


def test_restricted_account_rejects_assignment_but_allows_reclaim(store):
    _fund(store)
    store.assign_grant(
        user_id=USER_ID,
        **_mutation_args("before_restrict", amount_micro=2_000_000),
    )
    store.restrict_account(USER_ID)

    with pytest.raises(CreditAccountRestrictedStoreError):
        store.assign_grant(
            user_id=USER_ID,
            **_mutation_args("restricted_assign", amount_micro=1_000_000),
        )
    reclaimed = store.reclaim_grant(
        user_id=USER_ID,
        **_mutation_args("restricted_reclaim", amount_micro=1_000_000),
    )

    assert reclaimed["user_balance"]["grant_available_micro"] == 1_000_000


def test_idempotent_replay_returns_original_entries_and_conflicts_on_digest(store):
    _fund(store)
    command = _mutation_args("idempotent_assign", amount_micro=2_000_000)

    first = store.assign_grant(user_id=USER_ID, **command)
    store.assign_grant(
        user_id=USER_ID,
        **_mutation_args("after_idempotent_assign", amount_micro=1_000_000),
    )
    with store._get_connection() as conn:
        conn.execute(
            """
            UPDATE credit_grant_pools
            SET name = 'Renamed Research Pool', status = 'disabled'
            WHERE pool_id = 'default'
            """
        )
    replayed = store.assign_grant(user_id=USER_ID, **command)

    assert replayed == first
    current = store.get_grant_pool_summary("default", MONTH_START_ISO)
    assert current["pool_name"] == "Renamed Research Pool"
    assert current["pool_status"] == "disabled"
    assert store.get_balance_projection(USER_ID)["grant_committed_micro"] == 3_000_000
    assert len(store.list_grant_pool_activity("default")["items"]) == 3

    with pytest.raises(IdempotencyConflictError):
        store.assign_grant(
            user_id=USER_ID,
            **{
                **command,
                "request_digest": "different_digest",
            },
        )
    with pytest.raises(IdempotencyConflictError):
        store.assign_grant(
            user_id=USER_ID,
            **{**command, "amount_micro": 2_000_001},
        )
    with pytest.raises(IdempotencyConflictError):
        store.assign_grant(
            user_id=OTHER_USER_ID,
            **command,
        )
    with pytest.raises(IdempotencyConflictError):
        store.assign_grant(
            user_id=USER_ID,
            **{**command, "source": "different_budget"},
        )


def test_failure_after_user_insert_rolls_back_both_ledgers(store, monkeypatch):
    _fund(store, 3_000_000)

    def fail_pool_insert(*args, **kwargs):
        raise RuntimeError("injected pool insert failure")

    monkeypatch.setattr(
        store,
        "_insert_grant_pool_entry_in_transaction",
        fail_pool_insert,
    )

    with pytest.raises(RuntimeError, match="injected"):
        store.assign_grant(
            user_id=USER_ID,
            **_mutation_args("rollback_assign", amount_micro=1_000_000),
        )

    assert store.get_balance_projection(USER_ID)["grant_committed_micro"] == 0
    summary = store.get_grant_pool_summary("default", MONTH_START_ISO)
    assert summary["pool_available_micro"] == 3_000_000
    assert len(store.list_grant_pool_activity("default")["items"]) == 1


def test_concurrent_assignments_cannot_spend_the_last_pool_credit(store):
    _fund(store, 1_000_000)

    def assign(user_id: int) -> str:
        separate = CreditsStore(store.db_path)
        try:
            separate.assign_grant(
                user_id=user_id,
                **_mutation_args(f"race_{user_id}", amount_micro=1_000_000),
            )
            return "assigned"
        except GrantPoolInsufficientError:
            return "insufficient"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(assign, (USER_ID, OTHER_USER_ID)))

    assert sorted(outcomes) == ["assigned", "insufficient"]
    summary = store.get_grant_pool_summary("default", MONTH_START_ISO)
    assert summary["pool_available_micro"] == 0
    assert summary["allocated_to_users_micro"] == 1_000_000


def test_concurrent_reclaims_cannot_spend_the_last_user_grant(store):
    _fund(store, 2_000_000)
    store.assign_grant(
        user_id=USER_ID,
        **_mutation_args("reclaim_race_seed", amount_micro=1_000_000),
    )

    def reclaim(number: int) -> str:
        separate = CreditsStore(store.db_path)
        try:
            separate.reclaim_grant(
                user_id=USER_ID,
                **_mutation_args(f"reclaim_race_{number}", amount_micro=1_000_000),
            )
            return "reclaimed"
        except GrantReclaimExceedsAvailableError:
            return "insufficient"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(reclaim, (1, 2)))

    assert sorted(outcomes) == ["insufficient", "reclaimed"]
    assert store.get_balance_projection(USER_ID)["grant_available_micro"] == 0
    assert (
        store.get_grant_pool_summary("default", MONTH_START_ISO)["pool_available_micro"]
        == 2_000_000
    )


def test_grant_pool_summary_is_one_snapshot_during_concurrent_assign(
    store, monkeypatch
):
    _fund(store)
    with store._get_connection() as conn:
        assert conn.execute("PRAGMA journal_mode = WAL").fetchone()[0] == "wal"

    balance_read = Event()
    assignment_committed = Event()
    original_balance = store._pool_balance_in_transaction

    def pause_after_pool_balance(conn, pool_id):
        balance = original_balance(conn, pool_id)
        balance_read.set()
        assert assignment_committed.wait(timeout=5)
        return balance

    monkeypatch.setattr(
        store,
        "_pool_balance_in_transaction",
        pause_after_pool_balance,
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        summary_future = executor.submit(
            store.get_grant_pool_summary, "default", MONTH_START_ISO
        )
        assert balance_read.wait(timeout=5)
        try:
            CreditsStore(store.db_path).assign_grant(
                user_id=USER_ID,
                **_mutation_args("summary_race", amount_micro=3_000_000),
            )
        finally:
            assignment_committed.set()
        concurrent_summary = summary_future.result(timeout=5)

    assert concurrent_summary["pool_available_micro"] == 10_000_000
    assert concurrent_summary["allocated_to_users_micro"] == 0
    assert concurrent_summary["assigned_this_month_micro"] == 0

    fresh = store.get_grant_pool_summary("default", MONTH_START_ISO)
    assert fresh["pool_available_micro"] == 7_000_000
    assert fresh["allocated_to_users_micro"] == 3_000_000
    assert fresh["assigned_this_month_micro"] == 3_000_000


def test_batch_projection_includes_zero_accounts_and_preserves_input_ids(store):
    _purchase(store)
    _fund(store)
    store.assign_grant(
        user_id=USER_ID,
        **_mutation_args("batch_assign", amount_micro=3_000_000),
    )

    assert store.get_balance_projections([]) == {}
    projections = store.get_balance_projections([USER_ID, OTHER_USER_ID])

    assert list(projections) == [USER_ID, OTHER_USER_ID]
    assert projections[USER_ID] == store.get_balance_projection(USER_ID)
    assert projections[OTHER_USER_ID] == {
        "grant_committed_micro": 0,
        "purchased_committed_micro": 0,
        "grant_available_micro": 0,
        "purchased_available_micro": 0,
        "total_available_micro": 0,
    }


def test_monthly_summary_uses_inclusive_utc_boundary(store, monkeypatch):
    monkeypatch.setattr(
        repository_module,
        "_utcnow_iso",
        lambda: "2026-07-10T12:00:00+00:00",
    )
    _fund(store)
    store.assign_grant(
        user_id=USER_ID,
        **_mutation_args("july_assign", amount_micro=3_000_000),
    )

    monkeypatch.setattr(
        repository_module,
        "_utcnow_iso",
        lambda: MONTH_START_ISO,
    )
    store.assign_grant(
        user_id=USER_ID,
        **_mutation_args("august_assign", amount_micro=4_000_000),
    )
    monkeypatch.setattr(
        repository_module,
        "_utcnow_iso",
        lambda: "2026-08-02T00:00:00+00:00",
    )
    store.reclaim_grant(
        user_id=USER_ID,
        **_mutation_args("august_reclaim", amount_micro=1_000_000),
    )

    summary = store.get_grant_pool_summary("default", MONTH_START_ISO)
    assert summary == {
        "pool_id": "default",
        "pool_name": "Platform Research Grants",
        "pool_status": "active",
        "pool_available_micro": 4_000_000,
        "allocated_to_users_micro": 6_000_000,
        "assigned_this_month_micro": 4_000_000,
        "reclaimed_this_month_micro": 1_000_000,
        "month_start_iso": MONTH_START_ISO,
    }
    assert (
        store.get_grant_pool_summary("default", "2026-08-01T00:00:00Z")[
            "month_start_iso"
        ]
        == MONTH_START_ISO
    )


def test_default_pool_seed_adds_no_money_and_preserves_admin_state(store):
    initial = store.get_grant_pool_summary("default", MONTH_START_ISO)
    assert initial["pool_name"] == "Platform Research Grants"
    assert initial["pool_available_micro"] == 0
    assert store.list_grant_pool_activity("default")["items"] == []

    _fund(store, 2_000_000)
    with store._get_connection() as conn:
        conn.execute(
            """
            UPDATE credit_grant_pools
            SET name = 'Renamed Research Pool', status = 'disabled'
            WHERE pool_id = 'default'
            """
        )

    reopened = CreditsStore(store.db_path)
    preserved = reopened.get_grant_pool_summary("default", MONTH_START_ISO)
    assert preserved["pool_name"] == "Renamed Research Pool"
    assert preserved["pool_status"] == "disabled"
    assert preserved["pool_available_micro"] == 2_000_000
    assert len(reopened.list_grant_pool_activity("default")["items"]) == 1


def test_grant_pool_activity_is_cursor_paginated(store):
    _fund(store)
    store.reduce_grant_pool(**_mutation_args("page_reduce", amount_micro=1_000_000))
    store.assign_grant(
        user_id=USER_ID,
        **_mutation_args("page_assign", amount_micro=2_000_000),
    )
    store.reclaim_grant(
        user_id=USER_ID,
        **_mutation_args("page_reclaim", amount_micro=1_000_000),
    )

    first = store.list_grant_pool_activity("default", limit=2)
    second = store.list_grant_pool_activity(
        "default", limit=2, cursor=first["next_cursor"]
    )

    assert len(first["items"]) == len(second["items"]) == 2
    assert first["next_cursor"] == first["items"][-1]["id"]
    assert second["next_cursor"] is None
    assert {row["id"] for row in first["items"]}.isdisjoint(
        row["id"] for row in second["items"]
    )
