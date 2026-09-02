"""SQLite persistence for Credits, payment operations, and webhook receipts."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dashboard.backend.database import DB_PATH
from dashboard.backend.db_url import describe_database_url
from dashboard.backend.domain.credits.repository_common import (
    CreditAccountRestrictedStoreError,
    GrantPoolInsufficientError,
    GrantReclaimExceedsAvailableError,
    InsufficientCreditsError,
    IdempotencyConflictError,
    LLMReservationConflictError,
    OrderConflictError,
    RefundNotAllowedError,
    decode_activity_cursor,
    encode_activity_cursor,
    normalize_activity_item,
    _positive_integer,
    _positive_limit,
    _required_text,
    _utcnow_iso,
    _validate_amount_pair,
)
from dashboard.backend.domain.model_providers.repository_common import (
    validate_provider_id,
)


_SETTLEMENT_OUTCOME_EVIDENCE_KEYS = {
    "debited_credits_micro",
    "outstanding_credits_micro",
}


def _evidence_json(evidence: dict[str, Any]) -> str:
    return json.dumps(
        evidence,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _evidence_identity_json(evidence_json: str) -> str:
    try:
        payload = json.loads(evidence_json)
    except (TypeError, ValueError):
        return evidence_json
    if not isinstance(payload, dict):
        return evidence_json
    payload = {
        key: value
        for key, value in payload.items()
        if key not in _SETTLEMENT_OUTCOME_EVIDENCE_KEYS
    }
    return _evidence_json(payload)


_CREDIT_LEDGER_DDL = """
CREATE TABLE credit_ledger_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    bucket TEXT NOT NULL CHECK (bucket IN ('grant', 'purchased')),
    entry_type TEXT NOT NULL CHECK (entry_type IN (
        'purchase', 'refund', 'admin_grant_assign', 'admin_grant_reclaim'
    )),
    amount_micro INTEGER NOT NULL CHECK (amount_micro <> 0),
    payment_order_id TEXT,
    refund_request_id TEXT,
    stripe_event_id TEXT,
    operation_key TEXT NOT NULL UNIQUE
        CHECK (length(trim(operation_key)) > 0),
    operation_id TEXT NOT NULL CHECK (length(trim(operation_id)) > 0),
    idempotency_key TEXT NOT NULL UNIQUE
        CHECK (length(trim(idempotency_key)) > 0),
    request_digest TEXT CHECK (
        request_digest IS NULL OR length(trim(request_digest)) > 0
    ),
    actor_user_id INTEGER,
    source TEXT NOT NULL CHECK (length(trim(source)) > 0),
    reason TEXT NOT NULL CHECK (length(trim(reason)) > 0),
    reference_type TEXT,
    reference_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (payment_order_id)
        REFERENCES credit_payment_orders(id) ON DELETE RESTRICT,
    FOREIGN KEY (refund_request_id)
        REFERENCES credit_refund_requests(id) ON DELETE RESTRICT,
    FOREIGN KEY (stripe_event_id)
        REFERENCES stripe_webhook_events(stripe_event_id) ON DELETE RESTRICT,
    FOREIGN KEY (actor_user_id) REFERENCES users(id) ON DELETE RESTRICT,
    CHECK (
        (
            entry_type = 'purchase'
            AND bucket = 'purchased'
            AND amount_micro > 0
            AND payment_order_id IS NOT NULL
            AND refund_request_id IS NULL
            AND stripe_event_id IS NOT NULL
            AND request_digest IS NULL
            AND actor_user_id IS NULL
            AND reference_type IS NULL
            AND reference_id IS NULL
        )
        OR (
            entry_type = 'refund'
            AND bucket = 'purchased'
            AND amount_micro < 0
            AND payment_order_id IS NOT NULL
            AND refund_request_id IS NOT NULL
            AND stripe_event_id IS NOT NULL
            AND request_digest IS NULL
            AND actor_user_id IS NULL
            AND reference_type IS NULL
            AND reference_id IS NULL
        )
        OR (
            entry_type IN ('admin_grant_assign', 'admin_grant_reclaim')
            AND bucket = 'grant'
            AND payment_order_id IS NULL
            AND refund_request_id IS NULL
            AND stripe_event_id IS NULL
            AND request_digest IS NOT NULL
            AND actor_user_id IS NOT NULL
            AND reference_type = 'grant_pool'
            AND reference_id IS NOT NULL
            AND (
                (entry_type = 'admin_grant_assign' AND amount_micro > 0)
                OR (entry_type = 'admin_grant_reclaim' AND amount_micro < 0)
            )
        )
    )
)
"""


_LLM_RESERVATION_DDL = """
CREATE TABLE IF NOT EXISTS credit_llm_reservations (
    reservation_id TEXT PRIMARY KEY CHECK (length(trim(reservation_id)) > 0),
    user_id INTEGER NOT NULL,
    run_id TEXT NOT NULL CHECK (length(trim(run_id)) > 0),
    call_index INTEGER NOT NULL CHECK (call_index >= 0),
    provider_id TEXT,
    attempt_index INTEGER NOT NULL DEFAULT 0 CHECK (attempt_index >= 0),
    reserved_micro INTEGER NOT NULL CHECK (reserved_micro > 0),
    reserved_grant_micro INTEGER NOT NULL CHECK (reserved_grant_micro >= 0),
    reserved_purchased_micro INTEGER NOT NULL CHECK (reserved_purchased_micro >= 0),
    settled_micro INTEGER NOT NULL DEFAULT 0 CHECK (settled_micro >= 0),
    actual_micro INTEGER NOT NULL DEFAULT 0 CHECK (actual_micro >= 0),
    outstanding_micro INTEGER NOT NULL DEFAULT 0 CHECK (outstanding_micro >= 0),
    outstanding_recovered_micro INTEGER NOT NULL DEFAULT 0
        CHECK (outstanding_recovered_micro >= 0),
    status TEXT NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'settled', 'released')),
    operation_key TEXT NOT NULL UNIQUE CHECK (length(trim(operation_key)) > 0),
    request_digest TEXT NOT NULL CHECK (length(trim(request_digest)) > 0),
    evidence_json TEXT,
    failure_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CHECK (reserved_micro = reserved_grant_micro + reserved_purchased_micro),
    UNIQUE (user_id, run_id, call_index, attempt_index)
)
"""


_LLM_USAGE_LEDGER_DDL = """
CREATE TABLE IF NOT EXISTS credit_llm_usage_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    reservation_id TEXT NOT NULL,
    run_id TEXT NOT NULL CHECK (length(trim(run_id)) > 0),
    call_index INTEGER NOT NULL CHECK (call_index >= 0),
    bucket TEXT NOT NULL CHECK (bucket IN ('grant', 'purchased')),
    amount_micro INTEGER NOT NULL CHECK (amount_micro < 0),
    operation_key TEXT NOT NULL UNIQUE CHECK (length(trim(operation_key)) > 0),
    evidence_json TEXT NOT NULL CHECK (length(trim(evidence_json)) > 0),
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (reservation_id)
        REFERENCES credit_llm_reservations(reservation_id) ON DELETE RESTRICT
)
"""


_GRANT_POOL_DDL = """
CREATE TABLE IF NOT EXISTS credit_grant_pools (
    pool_id TEXT PRIMARY KEY CHECK (length(trim(pool_id)) > 0),
    name TEXT NOT NULL CHECK (length(trim(name)) > 0),
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'disabled')),
    created_at TEXT NOT NULL
)
"""


_GRANT_POOL_LEDGER_DDL = """
CREATE TABLE IF NOT EXISTS credit_grant_pool_ledger_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pool_id TEXT NOT NULL,
    pool_name_snapshot TEXT NOT NULL
        CHECK (length(trim(pool_name_snapshot)) > 0),
    pool_status_snapshot TEXT NOT NULL
        CHECK (pool_status_snapshot IN ('active', 'disabled')),
    entry_type TEXT NOT NULL
        CHECK (entry_type IN ('fund', 'reduce', 'assign', 'reclaim')),
    amount_micro INTEGER NOT NULL CHECK (amount_micro <> 0),
    operation_id TEXT NOT NULL UNIQUE
        CHECK (length(trim(operation_id)) > 0),
    idempotency_key TEXT NOT NULL UNIQUE
        CHECK (length(trim(idempotency_key)) > 0),
    request_digest TEXT NOT NULL CHECK (length(trim(request_digest)) > 0),
    actor_user_id INTEGER NOT NULL,
    source TEXT NOT NULL CHECK (length(trim(source)) > 0),
    reason TEXT NOT NULL CHECK (length(trim(reason)) > 0),
    user_id INTEGER,
    user_ledger_entry_id INTEGER UNIQUE,
    created_at TEXT NOT NULL,
    FOREIGN KEY (pool_id)
        REFERENCES credit_grant_pools(pool_id) ON DELETE RESTRICT,
    FOREIGN KEY (actor_user_id) REFERENCES users(id) ON DELETE RESTRICT,
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
)
"""


_PROMOTION_GRANT_DDL = """
CREATE TABLE IF NOT EXISTS credit_promotion_grants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    campaign_key TEXT NOT NULL CHECK (length(trim(campaign_key)) > 0),
    amount_micro INTEGER NOT NULL CHECK (amount_micro > 0),
    operation_id TEXT NOT NULL UNIQUE CHECK (length(trim(operation_id)) > 0),
    idempotency_key TEXT NOT NULL UNIQUE
        CHECK (length(trim(idempotency_key)) > 0),
    request_digest TEXT NOT NULL CHECK (length(trim(request_digest)) > 0),
    source TEXT NOT NULL CHECK (length(trim(source)) > 0),
    reason TEXT NOT NULL CHECK (length(trim(reason)) > 0),
    created_at TEXT NOT NULL,
    UNIQUE (campaign_key, user_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
)
"""


def _dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


class CreditsStore:
    """Account-scoped append-only Credits ledger backed by SQLite."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path or DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection that is committed/rolled back **and closed**.

        ``with sqlite3.connect(...) as conn`` is sqlite3's *transaction*
        context manager: it commits on success and rolls back on an exception,
        but it never closes the connection. Every one of this store's methods
        used it that way, so each call leaked an OS file handle and — in WAL
        mode — a read mark that holds off checkpointing until CPython happens
        to collect it. Under the Credits page's order poll (120 requests per
        minute per user, one connection each) that accumulates against a
        512MB instance and an ever-growing ``-wal`` sidecar.

        Every other store in this repo (``users.py``, ``domain/agents``,
        ``domain/strategies``) closes explicitly instead. Wrapping the helper
        keeps that guarantee in one place rather than repeating a ``finally``
        in all ~20 callers, which is also why no call site changed.

        ``PRAGMA foreign_keys = ON`` stays: the ledger's ``ON DELETE RESTRICT``
        foreign keys are what stop a purchase row from being deleted out from
        under the money it accounts for. It is scoped to this connection, so
        with the close in place it no longer outlives the call.
        """
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS credit_accounts (
                    user_id INTEGER PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'restricted')),
                    restriction_reason TEXT
                        CHECK (restriction_reason IN (
                            'llm_overage', 'refund_reconciliation'
                        ) OR restriction_reason IS NULL),
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS credit_payment_orders (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    user_id INTEGER NOT NULL,
                    client_request_id TEXT NOT NULL,
                    stripe_mode TEXT NOT NULL DEFAULT 'test'
                        CHECK (stripe_mode IN ('test', 'live')),
                    currency TEXT NOT NULL DEFAULT 'usd',
                    amount_usd_cents INTEGER NOT NULL
                        CHECK (amount_usd_cents > 0),
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

                CREATE INDEX IF NOT EXISTS idx_credit_payment_orders_user_sequence
                ON credit_payment_orders(user_id, sequence DESC);

                CREATE TABLE IF NOT EXISTS credit_refund_requests (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    payment_order_id TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    requested_by_user_id INTEGER,
                    amount_usd_cents INTEGER NOT NULL
                        CHECK (amount_usd_cents > 0),
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
                    FOREIGN KEY (requested_by_user_id) REFERENCES users(id)
                        ON DELETE RESTRICT
                );

                CREATE INDEX IF NOT EXISTS idx_credit_refunds_order_status
                ON credit_refund_requests(payment_order_id, status);

                CREATE TABLE IF NOT EXISTS stripe_webhook_events (
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
                """
            )
            self._begin(conn)
            self._migrate_credit_ledger_in_transaction(conn)
            self._create_llm_billing_schema_in_transaction(conn)
            self._create_grant_pool_schema_in_transaction(conn)
            self._create_promotion_grant_schema_in_transaction(conn)

    @staticmethod
    def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
        return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})")}

    @classmethod
    def _migrate_credit_ledger_in_transaction(
        cls, conn: sqlite3.Connection
    ) -> None:
        columns = cls._table_columns(conn, "credit_ledger_entries")
        if not columns:
            conn.execute(_CREDIT_LEDGER_DDL)
        elif "bucket" not in columns:
            legacy_evidence = conn.execute(
                """
                SELECT COUNT(*) AS row_count,
                       COALESCE(SUM(amount_micro), 0) AS amount_sum
                FROM credit_ledger_entries
                """
            ).fetchone()
            conn.execute(
                "ALTER TABLE credit_ledger_entries "
                "RENAME TO credit_ledger_entries_legacy"
            )
            conn.execute(_CREDIT_LEDGER_DDL)
            conn.execute(
                """
                INSERT INTO credit_ledger_entries (
                    id, user_id, bucket, entry_type, amount_micro,
                    payment_order_id, refund_request_id, stripe_event_id,
                    operation_key, operation_id, idempotency_key,
                    request_digest, actor_user_id, source, reason,
                    reference_type, reference_id, created_at
                )
                SELECT
                    id,
                    user_id,
                    'purchased',
                    entry_type,
                    amount_micro,
                    payment_order_id,
                    refund_request_id,
                    stripe_event_id,
                    operation_key,
                    operation_key,
                    operation_key,
                    NULL,
                    NULL,
                    'stripe',
                    CASE entry_type
                        WHEN 'purchase' THEN 'Historical Stripe purchase.'
                        ELSE 'Historical Stripe refund.'
                    END,
                    NULL,
                    NULL,
                    created_at
                FROM credit_ledger_entries_legacy
                ORDER BY id
                """
            )
            migrated_evidence = conn.execute(
                """
                SELECT COUNT(*) AS row_count,
                       COALESCE(SUM(amount_micro), 0) AS amount_sum
                FROM credit_ledger_entries
                """
            ).fetchone()
            if (
                int(migrated_evidence["row_count"])
                != int(legacy_evidence["row_count"])
                or int(migrated_evidence["amount_sum"])
                != int(legacy_evidence["amount_sum"])
            ):
                raise RuntimeError("Credits ledger migration evidence mismatch")
            conn.execute("DROP TABLE credit_ledger_entries_legacy")

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_credit_ledger_user_id
            ON credit_ledger_entries(user_id, id DESC)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_credit_ledger_payment_order
            ON credit_ledger_entries(payment_order_id, id DESC)
            """
        )

    @classmethod
    def _create_llm_billing_schema_in_transaction(
        cls,
        conn: sqlite3.Connection,
    ) -> None:
        conn.execute(_LLM_RESERVATION_DDL)
        reservation_columns = cls._table_columns(conn, "credit_llm_reservations")
        if "actual_micro" not in reservation_columns:
            conn.execute(
                "ALTER TABLE credit_llm_reservations "
                "ADD COLUMN actual_micro INTEGER NOT NULL DEFAULT 0 "
                "CHECK (actual_micro >= 0)"
            )
        if "outstanding_micro" not in reservation_columns:
            conn.execute(
                "ALTER TABLE credit_llm_reservations "
                "ADD COLUMN outstanding_micro INTEGER NOT NULL DEFAULT 0 "
                "CHECK (outstanding_micro >= 0)"
            )
        if "outstanding_recovered_micro" not in reservation_columns:
            conn.execute(
                "ALTER TABLE credit_llm_reservations "
                "ADD COLUMN outstanding_recovered_micro INTEGER NOT NULL DEFAULT 0 "
                "CHECK (outstanding_recovered_micro >= 0)"
            )
        if "provider_id" not in reservation_columns:
            conn.execute(
                "ALTER TABLE credit_llm_reservations ADD COLUMN provider_id TEXT"
            )
        if "attempt_index" not in reservation_columns:
            conn.execute(
                "ALTER TABLE credit_llm_reservations "
                "ADD COLUMN attempt_index INTEGER NOT NULL DEFAULT 0 "
                "CHECK (attempt_index >= 0)"
            )
        account_columns = cls._table_columns(conn, "credit_accounts")
        if "restriction_reason" not in account_columns:
            conn.execute(
                "ALTER TABLE credit_accounts ADD COLUMN restriction_reason TEXT"
            )
        conn.execute(
            """
            UPDATE credit_llm_reservations
            SET actual_micro = settled_micro
            WHERE status = 'settled' AND actual_micro = 0
            """
        )
        cls._migrate_llm_reservation_constraint_in_transaction(conn)
        conn.execute(_LLM_USAGE_LEDGER_DDL)
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_credit_llm_reservations_user_status
            ON credit_llm_reservations(user_id, status, created_at)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_credit_llm_reservations_run_status
            ON credit_llm_reservations(run_id, status, call_index, attempt_index)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_credit_llm_usage_user_id
            ON credit_llm_usage_entries(user_id, id DESC)
            """
        )

    @classmethod
    def _migrate_llm_reservation_constraint_in_transaction(
        cls, conn: sqlite3.Connection
    ) -> None:
        """Rebuild legacy SQLite reservations that capped settled debits."""

        # Some maintenance/CLI entry points instantiate the Credits store
        # before the account database has created its users table. The legacy
        # table can remain in place safely; the migration will run on the next
        # normal application initialization once its parent table exists.
        if not cls._table_columns(conn, "users"):
            return

        reservation_columns = cls._table_columns(conn, "credit_llm_reservations")
        row = conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name = 'credit_llm_reservations'"
        ).fetchone()
        table_sql = "".join(str(row["sql"] or "").lower().split()) if row else ""
        needs_rebuild = (
            "provider_id" not in reservation_columns
            or "attempt_index" not in reservation_columns
            or "unique(user_id,run_id,call_index)" in table_sql
            or "settled_micro<=reserved_micro" in table_sql
        )
        if not needs_rebuild:
            return

        usage_exists = bool(cls._table_columns(conn, "credit_llm_usage_entries"))
        if usage_exists:
            conn.execute(
                "CREATE TABLE credit_llm_usage_entries_migration_backup AS "
                "SELECT * FROM credit_llm_usage_entries"
            )
            conn.execute("DROP TABLE credit_llm_usage_entries")

        conn.execute(
            "ALTER TABLE credit_llm_reservations "
            "RENAME TO credit_llm_reservations_migration_legacy"
        )
        conn.execute(_LLM_RESERVATION_DDL)
        columns = (
            "reservation_id, user_id, run_id, call_index, provider_id, "
            "attempt_index, reserved_micro, reserved_grant_micro, "
            "reserved_purchased_micro, settled_micro, actual_micro, "
            "outstanding_micro, outstanding_recovered_micro, status, operation_key, "
            "request_digest, evidence_json, failure_reason, created_at, updated_at"
        )
        provider_expression = "provider_id" if "provider_id" in reservation_columns else "NULL"
        attempt_expression = "attempt_index" if "attempt_index" in reservation_columns else "0"
        select_columns = (
            "reservation_id, user_id, run_id, call_index, "
            f"{provider_expression}, {attempt_expression}, reserved_micro, "
            "reserved_grant_micro, reserved_purchased_micro, settled_micro, "
            "actual_micro, outstanding_micro, outstanding_recovered_micro, status, "
            "operation_key, request_digest, evidence_json, failure_reason, "
            "created_at, updated_at"
        )
        conn.execute(
            f"INSERT INTO credit_llm_reservations ({columns}) "
            f"SELECT {select_columns} FROM credit_llm_reservations_migration_legacy"
        )
        conn.execute("DROP TABLE credit_llm_reservations_migration_legacy")

        if usage_exists:
            conn.execute(_LLM_USAGE_LEDGER_DDL)
            usage_columns = (
                "id, user_id, reservation_id, run_id, call_index, bucket, "
                "amount_micro, operation_key, evidence_json, created_at"
            )
            conn.execute(
                f"INSERT INTO credit_llm_usage_entries ({usage_columns}) "
                f"SELECT {usage_columns} "
                "FROM credit_llm_usage_entries_migration_backup"
            )
            conn.execute("DROP TABLE credit_llm_usage_entries_migration_backup")

    @classmethod
    def _create_grant_pool_schema_in_transaction(
        cls,
        conn: sqlite3.Connection,
    ) -> None:
        conn.execute(_GRANT_POOL_DDL)
        columns = cls._table_columns(conn, "credit_grant_pool_ledger_entries")
        if not columns:
            conn.execute(_GRANT_POOL_LEDGER_DDL)
        elif (
            not {
                "pool_name_snapshot",
                "pool_status_snapshot",
            }
            <= columns
        ):
            legacy_evidence = conn.execute(
                """
                SELECT COUNT(*) AS row_count,
                       COALESCE(SUM(amount_micro), 0) AS amount_sum,
                       COALESCE(MAX(id), 0) AS max_id
                FROM credit_grant_pool_ledger_entries
                """
            ).fetchone()
            pool_name = (
                "legacy.pool_name_snapshot"
                if "pool_name_snapshot" in columns
                else "pool.name"
            )
            pool_status = (
                "legacy.pool_status_snapshot"
                if "pool_status_snapshot" in columns
                else "pool.status"
            )
            conn.execute(
                "ALTER TABLE credit_grant_pool_ledger_entries "
                "RENAME TO credit_grant_pool_ledger_entries_legacy"
            )
            conn.execute(_GRANT_POOL_LEDGER_DDL)
            conn.execute(
                f"""
                INSERT INTO credit_grant_pool_ledger_entries (
                    id, pool_id, pool_name_snapshot, pool_status_snapshot,
                    entry_type, amount_micro, operation_id, idempotency_key,
                    request_digest, actor_user_id, source, reason, user_id,
                    user_ledger_entry_id, created_at
                )
                SELECT
                    legacy.id,
                    legacy.pool_id,
                    {pool_name},
                    {pool_status},
                    legacy.entry_type,
                    legacy.amount_micro,
                    legacy.operation_id,
                    legacy.idempotency_key,
                    legacy.request_digest,
                    legacy.actor_user_id,
                    legacy.source,
                    legacy.reason,
                    legacy.user_id,
                    legacy.user_ledger_entry_id,
                    legacy.created_at
                FROM credit_grant_pool_ledger_entries_legacy AS legacy
                JOIN credit_grant_pools AS pool ON pool.pool_id = legacy.pool_id
                ORDER BY legacy.id
                """
            )
            migrated_evidence = conn.execute(
                """
                SELECT COUNT(*) AS row_count,
                       COALESCE(SUM(amount_micro), 0) AS amount_sum,
                       COALESCE(MAX(id), 0) AS max_id
                FROM credit_grant_pool_ledger_entries
                """
            ).fetchone()
            if tuple(migrated_evidence) != tuple(legacy_evidence):
                raise RuntimeError("Grant Pool ledger migration evidence mismatch")
            conn.execute("DROP TABLE credit_grant_pool_ledger_entries_legacy")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_credit_grant_pool_ledger_pool_id
            ON credit_grant_pool_ledger_entries(pool_id, id DESC)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_credit_grant_user_reference
            ON credit_ledger_entries(reference_type, reference_id, user_id, id DESC)
            """
        )
        conn.execute(
            """
            INSERT INTO credit_grant_pools (pool_id, name, status, created_at)
            VALUES ('default', 'Platform Research Grants', 'active', ?)
            ON CONFLICT(pool_id) DO NOTHING
            """,
            (_utcnow_iso(),),
        )

    @classmethod
    def _create_promotion_grant_schema_in_transaction(
        cls, conn: sqlite3.Connection
    ) -> None:
        conn.execute(_PROMOTION_GRANT_DDL)
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_credit_promotion_grants_user_id
            ON credit_promotion_grants(user_id, id DESC)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_credit_promotion_grants_campaign_user
            ON credit_promotion_grants(campaign_key, user_id)
            """
        )

    @staticmethod
    def _begin(conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN IMMEDIATE")

    @staticmethod
    def _ensure_account_in_transaction(
        conn: sqlite3.Connection, user_id: int
    ) -> None:
        conn.execute(
            """
            INSERT INTO credit_accounts (user_id, status, created_at)
            VALUES (?, 'active', ?)
            ON CONFLICT(user_id) DO NOTHING
            """,
            (user_id, _utcnow_iso()),
        )

    def ensure_account(self, user_id: int) -> dict[str, Any]:
        _positive_integer(user_id, "user_id")
        with self._get_connection() as conn:
            self._begin(conn)
            self._ensure_account_in_transaction(conn, user_id)
            row = conn.execute(
                "SELECT * FROM credit_accounts WHERE user_id = ?", (user_id,)
            ).fetchone()
            return dict(row)

    def get_account_billing_state(self, user_id: int) -> dict[str, Any]:
        _positive_integer(user_id, "user_id")
        with self._get_connection() as conn:
            self._begin(conn)
            self._ensure_account_in_transaction(conn, user_id)
            account = conn.execute(
                "SELECT status, restriction_reason FROM credit_accounts WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            outstanding = conn.execute(
                """
                SELECT COALESCE(SUM(
                    MAX(outstanding_micro - outstanding_recovered_micro, 0)
                ), 0) AS outstanding_micro
                FROM credit_llm_reservations
                WHERE user_id = ? AND status = 'settled'
                """,
                (user_id,),
            ).fetchone()
            reason = account["restriction_reason"]
            if account["status"] == "restricted" and reason not in {
                "llm_overage",
                "refund_reconciliation",
            }:
                reason = "refund_reconciliation"
            return {
                "account_status": account["status"],
                "restriction_reason": reason,
                "outstanding_credits_micro": int(outstanding["outstanding_micro"]),
            }

    @staticmethod
    def _balance_projection_in_transaction(
        conn: sqlite3.Connection,
        user_id: int,
        through_entry_id: int | None = None,
    ) -> dict[str, int]:
        cutoff_sql = ""
        params: list[Any] = [user_id]
        if through_entry_id is not None:
            cutoff_sql = "AND id <= ?"
            params.append(through_entry_id)
        row = conn.execute(
            f"""
            SELECT
                COALESCE(SUM(
                    CASE WHEN bucket = 'grant' THEN amount_micro ELSE 0 END
                ), 0) AS grant_committed_micro,
                COALESCE(SUM(
                    CASE WHEN bucket = 'purchased' THEN amount_micro ELSE 0 END
                ), 0) AS purchased_committed_micro
            FROM credit_ledger_entries
            WHERE user_id = ?
              {cutoff_sql}
            """,
            params,
        ).fetchone()
        grant_micro = int(row["grant_committed_micro"])
        purchased_micro = int(row["purchased_committed_micro"])
        if through_entry_id is None:
            promotion = conn.execute(
                """
                SELECT COALESCE(SUM(amount_micro), 0) AS grant_micro
                FROM credit_promotion_grants
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
            grant_micro += int(promotion["grant_micro"])
        reserved_grant_micro = 0
        reserved_purchased_micro = 0
        if through_entry_id is None:
            usage = conn.execute(
                """
                SELECT
                    COALESCE(SUM(CASE WHEN bucket = 'grant' THEN amount_micro ELSE 0 END), 0)
                        AS grant_usage_micro,
                    COALESCE(SUM(CASE WHEN bucket = 'purchased' THEN amount_micro ELSE 0 END), 0)
                        AS purchased_usage_micro
                FROM credit_llm_usage_entries
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
            grant_micro += int(usage["grant_usage_micro"])
            purchased_micro += int(usage["purchased_usage_micro"])
            reserved = conn.execute(
                """
                SELECT
                    COALESCE(SUM(reserved_grant_micro), 0) AS grant_micro,
                    COALESCE(SUM(reserved_purchased_micro), 0) AS purchased_micro
                FROM credit_llm_reservations
                WHERE user_id = ? AND status = 'open'
                """,
                (user_id,),
            ).fetchone()
            reserved_grant_micro = int(reserved["grant_micro"])
            reserved_purchased_micro = int(reserved["purchased_micro"])
        grant_available = grant_micro - reserved_grant_micro
        purchased_available = purchased_micro - reserved_purchased_micro
        return {
            "grant_committed_micro": grant_micro,
            "purchased_committed_micro": purchased_micro,
            "grant_available_micro": grant_available,
            "purchased_available_micro": purchased_available,
            "total_available_micro": grant_available + purchased_available,
        }

    def get_balance_projection(self, user_id: int) -> dict[str, int]:
        _positive_integer(user_id, "user_id")
        with self._get_connection() as conn:
            return self._balance_projection_in_transaction(conn, user_id)

    def get_balance_projections(
        self, user_ids: list[int] | tuple[int, ...]
    ) -> dict[int, dict[str, int]]:
        if not isinstance(user_ids, (list, tuple)):
            raise ValueError("user_ids must be a list or tuple")
        validated = [_positive_integer(user_id, "user_id") for user_id in user_ids]
        if not validated:
            return {}

        unique_ids = list(dict.fromkeys(validated))
        placeholders = ", ".join("?" for _ in unique_ids)
        with self._get_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    user_id,
                    COALESCE(SUM(
                        CASE WHEN bucket = 'grant' THEN amount_micro ELSE 0 END
                    ), 0) AS grant_committed_micro,
                    COALESCE(SUM(
                        CASE WHEN bucket = 'purchased' THEN amount_micro ELSE 0 END
                    ), 0) AS purchased_committed_micro
                FROM credit_ledger_entries
                WHERE user_id IN ({placeholders})
                GROUP BY user_id
                """,
                unique_ids,
            ).fetchall()

            promotion_rows = conn.execute(
                f"""
                SELECT user_id, COALESCE(SUM(amount_micro), 0) AS grant_micro
                FROM credit_promotion_grants
                WHERE user_id IN ({placeholders})
                GROUP BY user_id
                """,
                unique_ids,
            ).fetchall()

            usage_rows = conn.execute(
                f"""
                SELECT
                    user_id,
                    COALESCE(SUM(CASE WHEN bucket = 'grant' THEN amount_micro ELSE 0 END), 0)
                        AS grant_usage_micro,
                    COALESCE(SUM(CASE WHEN bucket = 'purchased' THEN amount_micro ELSE 0 END), 0)
                        AS purchased_usage_micro
                FROM credit_llm_usage_entries
                WHERE user_id IN ({placeholders})
                GROUP BY user_id
                """,
                unique_ids,
            ).fetchall()
            reservation_rows = conn.execute(
                f"""
                SELECT
                    user_id,
                    COALESCE(SUM(reserved_grant_micro), 0) AS reserved_grant_micro,
                    COALESCE(SUM(reserved_purchased_micro), 0) AS reserved_purchased_micro
                FROM credit_llm_reservations
                WHERE user_id IN ({placeholders}) AND status = 'open'
                GROUP BY user_id
                """,
                unique_ids,
            ).fetchall()

        amounts = {
            int(row["user_id"]): (
                int(row["grant_committed_micro"]),
                int(row["purchased_committed_micro"]),
            )
            for row in rows
        }
        for row in promotion_rows:
            user_id = int(row["user_id"])
            grant, purchased = amounts.get(user_id, (0, 0))
            amounts[user_id] = (grant + int(row["grant_micro"]), purchased)
        usage_amounts = {
            int(row["user_id"]): (
                int(row["grant_usage_micro"]),
                int(row["purchased_usage_micro"]),
            )
            for row in usage_rows
        }
        reserved_amounts = {
            int(row["user_id"]): (
                int(row["reserved_grant_micro"]),
                int(row["reserved_purchased_micro"]),
            )
            for row in reservation_rows
        }
        projections: dict[int, dict[str, int]] = {}
        for user_id in unique_ids:
            grant_micro, purchased_micro = amounts.get(user_id, (0, 0))
            grant_usage, purchased_usage = usage_amounts.get(user_id, (0, 0))
            grant_micro += grant_usage
            purchased_micro += purchased_usage
            reserved_grant, reserved_purchased = reserved_amounts.get(
                user_id, (0, 0)
            )
            grant_available = grant_micro - reserved_grant
            purchased_available = purchased_micro - reserved_purchased
            projections[user_id] = {
                "grant_committed_micro": grant_micro,
                "purchased_committed_micro": purchased_micro,
                "grant_available_micro": grant_available,
                "purchased_available_micro": purchased_available,
                "total_available_micro": grant_available + purchased_available,
            }
        return projections

    def get_balance_micro(self, user_id: int) -> int:
        return self.get_balance_projection(user_id)["total_available_micro"]

    def list_user_ids(self) -> list[int]:
        """Return account IDs for an idempotent promotion backfill."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT id FROM users ORDER BY id").fetchall()
        return [int(row["id"]) for row in rows]

    def grant_promotion_credits(
        self,
        *,
        user_id: int,
        campaign_key: str,
        amount_micro: int,
        operation_id: str,
        idempotency_key: str,
        request_digest: str,
        source: str,
        reason: str,
    ) -> dict[str, Any]:
        """Post one system promotion grant, safely replayable by its key."""
        _positive_integer(user_id, "user_id")
        _required_text(campaign_key, "campaign_key", max_length=120)
        _positive_integer(amount_micro, "amount_micro")
        _required_text(operation_id, "operation_id")
        _required_text(idempotency_key, "idempotency_key")
        _required_text(request_digest, "request_digest")
        _required_text(source, "source", max_length=120)
        _required_text(reason, "reason", max_length=500)
        with self._get_connection() as conn:
            self._begin(conn)
            self._ensure_account_in_transaction(conn, user_id)
            existing = conn.execute(
                """
                SELECT * FROM credit_promotion_grants
                WHERE idempotency_key = ? OR operation_id = ?
                """,
                (idempotency_key, operation_id),
            ).fetchone()
            if existing is not None:
                if (
                    existing["user_id"] != user_id
                    or existing["campaign_key"] != campaign_key
                    or int(existing["amount_micro"]) != amount_micro
                    or existing["operation_id"] != operation_id
                    or existing["idempotency_key"] != idempotency_key
                    or existing["request_digest"] != request_digest
                    or existing["source"] != source
                    or existing["reason"] != reason
                ):
                    raise IdempotencyConflictError(
                        "Promotion idempotency key conflicts with an existing grant"
                    )
                return {"created": False, "grant": dict(existing)}
            conn.execute(
                """
                INSERT INTO credit_promotion_grants (
                    user_id, campaign_key, amount_micro, operation_id,
                    idempotency_key, request_digest, source, reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    campaign_key,
                    amount_micro,
                    operation_id,
                    idempotency_key,
                    request_digest,
                    source,
                    reason,
                    _utcnow_iso(),
                ),
            )
            row = conn.execute(
                """
                SELECT * FROM credit_promotion_grants WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()
            return {"created": True, "grant": dict(row)}

    @staticmethod
    def _llm_reservation_result_in_transaction(
        conn: sqlite3.Connection,
        reservation: sqlite3.Row,
    ) -> dict[str, Any]:
        usage_entries = conn.execute(
            """
            SELECT id, bucket, amount_micro
            FROM credit_llm_usage_entries
            WHERE reservation_id = ? AND operation_key NOT LIKE '%:recovery:%'
            ORDER BY id
            """,
            (reservation["reservation_id"],),
        ).fetchall()
        grant_debited = sum(
            -int(entry["amount_micro"])
            for entry in usage_entries
            if entry["bucket"] == "grant"
        )
        purchased_debited = sum(
            -int(entry["amount_micro"])
            for entry in usage_entries
            if entry["bucket"] == "purchased"
        )
        row = dict(reservation)
        row.update(
            released_micro=max(
                int(row["reserved_micro"]) - int(row["settled_micro"]), 0
            ),
            outstanding_recovered_micro=int(
                row["outstanding_recovered_micro"] or 0
            ),
            grant_debited_micro=grant_debited,
            purchased_debited_micro=purchased_debited,
            ledger_entry_ids=tuple(int(entry["id"]) for entry in usage_entries),
        )
        return row

    def reserve_llm_credits(
        self,
        *,
        reservation_id: str,
        user_id: int,
        run_id: str,
        call_index: int,
        attempt_index: int,
        provider_id: str,
        reserved_micro: int,
        operation_key: str,
        request_digest: str,
    ) -> dict[str, Any]:
        reservation_id = _required_text(
            reservation_id, "reservation_id", max_length=160
        )
        run_id = _required_text(run_id, "run_id", max_length=128)
        operation_key = _required_text(
            operation_key, "operation_key", max_length=200
        )
        request_digest = _required_text(
            request_digest, "request_digest", max_length=128
        )
        _positive_integer(user_id, "user_id")
        _positive_integer(reserved_micro, "reserved_micro")
        if isinstance(call_index, bool) or not isinstance(call_index, int) or call_index < 0:
            raise ValueError("call_index must be a non-negative integer")
        if isinstance(attempt_index, bool) or not isinstance(attempt_index, int) or attempt_index < 0:
            raise ValueError("attempt_index must be a non-negative integer")
        provider_id = validate_provider_id(provider_id)

        with self._get_connection() as conn:
            self._begin(conn)
            self._migrate_llm_reservation_constraint_in_transaction(conn)
            existing = conn.execute(
                """
                SELECT * FROM credit_llm_reservations
                WHERE reservation_id = ? OR operation_key = ?
                   OR (user_id = ? AND run_id = ? AND call_index = ? AND attempt_index = ?)
                """,
                (reservation_id, operation_key, user_id, run_id, call_index, attempt_index),
            ).fetchone()
            if existing:
                if (
                    existing["reservation_id"] != reservation_id
                    or int(existing["user_id"]) != user_id
                    or existing["run_id"] != run_id
                    or int(existing["call_index"]) != call_index
                    or int(existing["attempt_index"]) != attempt_index
                    or existing["provider_id"] != provider_id
                    or int(existing["reserved_micro"]) != reserved_micro
                    or existing["operation_key"] != operation_key
                    or existing["request_digest"] != request_digest
                ):
                    raise LLMReservationConflictError(
                        "reservation key already represents different input"
                    )
                return dict(existing)

            self._ensure_account_in_transaction(conn, user_id)
            account = conn.execute(
                "SELECT status FROM credit_accounts WHERE user_id = ?", (user_id,)
            ).fetchone()
            if account["status"] == "restricted":
                raise CreditAccountRestrictedStoreError(
                    "restricted credit account cannot reserve model usage"
                )
            projection = self._balance_projection_in_transaction(conn, user_id)
            if projection["total_available_micro"] < reserved_micro:
                raise InsufficientCreditsError("insufficient available Credits")
            reserved_grant = min(
                max(projection["grant_available_micro"], 0), reserved_micro
            )
            reserved_purchased = reserved_micro - reserved_grant
            now = _utcnow_iso()
            conn.execute(
                """
                INSERT INTO credit_llm_reservations (
                    reservation_id, user_id, run_id, call_index, provider_id,
                    attempt_index,
                    reserved_micro, reserved_grant_micro,
                    reserved_purchased_micro, settled_micro, status,
                    operation_key, request_digest, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'open', ?, ?, ?, ?)
                """,
                (
                    reservation_id,
                    user_id,
                    run_id,
                    call_index,
                    provider_id,
                    attempt_index,
                    reserved_micro,
                    reserved_grant,
                    reserved_purchased,
                    operation_key,
                    request_digest,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM credit_llm_reservations WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
            return dict(row)

    def settle_llm_credits(
        self,
        reservation_id: str,
        *,
        actual_micro: int,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        reservation_id = _required_text(
            reservation_id, "reservation_id", max_length=160
        )
        if (
            isinstance(actual_micro, bool)
            or not isinstance(actual_micro, int)
            or actual_micro < 0
        ):
            raise ValueError("actual_micro must be a non-negative integer")
        evidence_json = _evidence_json(evidence)
        with self._get_connection() as conn:
            self._begin(conn)
            self._migrate_llm_reservation_constraint_in_transaction(conn)
            reservation = conn.execute(
                "SELECT * FROM credit_llm_reservations WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
            if not reservation:
                raise LLMReservationConflictError("reservation was not found")
            if reservation["status"] == "settled":
                if (
                    int(reservation["actual_micro"]) != actual_micro
                    or _evidence_identity_json(
                        str(reservation["evidence_json"] or "")
                    )
                    != _evidence_identity_json(evidence_json)
                ):
                    raise LLMReservationConflictError(
                        "settlement replay has different input"
                    )
                return self._llm_reservation_result_in_transaction(
                    conn, reservation
                )
            if reservation["status"] != "open":
                raise LLMReservationConflictError(
                    "released reservation cannot be settled"
                )
            reserved_micro = int(reservation["reserved_micro"])
            excess_micro = max(actual_micro - reserved_micro, 0)
            supplementary_grant = 0
            supplementary_purchased = 0
            if excess_micro > 0:
                projection = self._balance_projection_in_transaction(
                    conn, int(reservation["user_id"])
                )
                supplementary_grant = min(
                    excess_micro,
                    max(int(projection["grant_available_micro"]), 0),
                )
                supplementary_purchased = min(
                    excess_micro - supplementary_grant,
                    max(int(projection["purchased_available_micro"]), 0),
                )
            supplementary_micro = supplementary_grant + supplementary_purchased
            debit_micro = min(actual_micro, reserved_micro) + supplementary_micro
            outstanding_micro = actual_micro - debit_micro
            grant_debit = min(
                min(actual_micro, reserved_micro),
                int(reservation["reserved_grant_micro"]),
            )
            purchased_debit = min(actual_micro, reserved_micro) - grant_debit
            now = _utcnow_iso()
            evidence_payload = dict(evidence)
            evidence_payload["debited_credits_micro"] = debit_micro
            evidence_payload["outstanding_credits_micro"] = outstanding_micro
            settled_evidence_json = _evidence_json(evidence_payload)
            for bucket, amount, suffix in (
                ("grant", grant_debit, "grant"),
                ("purchased", purchased_debit, "purchased"),
                ("grant", supplementary_grant, "overage:grant"),
                ("purchased", supplementary_purchased, "overage:purchased"),
            ):
                if amount <= 0:
                    continue
                conn.execute(
                    """
                    INSERT INTO credit_llm_usage_entries (
                        user_id, reservation_id, run_id, call_index,
                        bucket, amount_micro, operation_key,
                        evidence_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        reservation["user_id"],
                        reservation_id,
                        reservation["run_id"],
                        reservation["call_index"],
                        bucket,
                        -amount,
                        f"{reservation['operation_key']}:{suffix}",
                        settled_evidence_json,
                        now,
                    ),
                )
            conn.execute(
                """
                UPDATE credit_llm_reservations
                SET settled_micro = ?, actual_micro = ?, outstanding_micro = ?,
                    status = 'settled',
                    evidence_json = ?, failure_reason = NULL, updated_at = ?
                WHERE reservation_id = ?
                """,
                (
                    debit_micro,
                    actual_micro,
                    outstanding_micro,
                    settled_evidence_json,
                    now,
                    reservation_id,
                ),
            )
            if outstanding_micro > 0:
                conn.execute(
                    """
                    UPDATE credit_accounts
                    SET status = 'restricted', restriction_reason = 'llm_overage'
                    WHERE user_id = ? AND status <> 'restricted'
                    """,
                    (reservation["user_id"],),
                )
            settled = conn.execute(
                "SELECT * FROM credit_llm_reservations WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
            return self._llm_reservation_result_in_transaction(conn, settled)

    def recover_llm_overage(
        self, user_id: int, *, source_operation_key: str
    ) -> dict[str, Any]:
        """Apply newly available Credits to an LLM overage atomically."""
        _positive_integer(user_id, "user_id")
        source_operation_key = _required_text(
            source_operation_key, "source_operation_key", max_length=200
        )
        with self._get_connection() as conn:
            self._begin(conn)
            return self._recover_llm_overage_in_transaction(
                conn, user_id, source_operation_key=source_operation_key
            )

    @staticmethod
    def _recovery_result_for_source_operation_in_transaction(
        conn: sqlite3.Connection,
        user_id: int,
        *,
        source_operation_key: str,
    ) -> dict[str, Any] | None:
        rows = conn.execute(
            "SELECT amount_micro, evidence_json FROM credit_llm_usage_entries "
            "WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        recovered = 0
        for row in rows:
            try:
                evidence = json.loads(row["evidence_json"])
            except (TypeError, ValueError):
                continue
            if not isinstance(evidence, dict):
                continue
            if evidence.get("recovery_source") == source_operation_key:
                recovered += max(-int(row["amount_micro"]), 0)
        if recovered <= 0:
            return None

        account = conn.execute(
            "SELECT status, restriction_reason FROM credit_accounts WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        remaining = conn.execute(
            """
            SELECT COALESCE(SUM(
                MAX(outstanding_micro - outstanding_recovered_micro, 0)
            ), 0) AS outstanding_micro
            FROM credit_llm_reservations
            WHERE user_id = ? AND status = 'settled'
            """,
            (user_id,),
        ).fetchone()
        reason = account["restriction_reason"]
        if account["status"] == "restricted" and reason not in {
            "llm_overage",
            "refund_reconciliation",
        }:
            reason = "refund_reconciliation"
        return {
            "recovered_micro": recovered,
            "outstanding_micro": int(remaining["outstanding_micro"]),
            "account_status": account["status"],
            "restriction_reason": reason,
        }

    @staticmethod
    def _recover_llm_overage_in_transaction(
        conn: sqlite3.Connection,
        user_id: int,
        *,
        source_operation_key: str,
    ) -> dict[str, Any]:
        CreditsStore._ensure_account_in_transaction(conn, user_id)
        previous = CreditsStore._recovery_result_for_source_operation_in_transaction(
            conn, user_id, source_operation_key=source_operation_key
        )
        if previous is not None:
            return previous
        account = conn.execute(
            "SELECT status, restriction_reason FROM credit_accounts WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        reason = account["restriction_reason"]
        if account["status"] != "restricted" or reason != "llm_overage":
            return {
                "recovered_micro": 0,
                "outstanding_micro": 0,
                "account_status": account["status"],
                "restriction_reason": reason,
            }

        projection = CreditsStore._balance_projection_in_transaction(conn, user_id)
        grant_available = max(int(projection["grant_available_micro"]), 0)
        purchased_available = max(int(projection["purchased_available_micro"]), 0)
        remaining_funds = grant_available + purchased_available
        recovered_total = 0
        reservations = conn.execute(
            """
            SELECT * FROM credit_llm_reservations
            WHERE user_id = ? AND status = 'settled'
              AND outstanding_micro > outstanding_recovered_micro
            ORDER BY created_at, reservation_id
            """,
            (user_id,),
        ).fetchall()
        for reservation in reservations:
            debt = max(
                int(reservation["outstanding_micro"])
                - int(reservation["outstanding_recovered_micro"]),
                0,
            )
            amount = min(debt, remaining_funds)
            if amount <= 0:
                break
            grant_debit = min(amount, grant_available)
            purchased_debit = amount - grant_debit
            evidence_json = json.dumps(
                {
                    "recovery_source": source_operation_key,
                    "reservation_id": reservation["reservation_id"],
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            for bucket, debit in (
                ("grant", grant_debit),
                ("purchased", purchased_debit),
            ):
                if debit <= 0:
                    continue
                operation_key = (
                    f"{reservation['operation_key']}:recovery:"
                    f"{source_operation_key}:{bucket}"
                )
                existing = conn.execute(
                    "SELECT amount_micro FROM credit_llm_usage_entries WHERE operation_key = ?",
                    (operation_key,),
                ).fetchone()
                if existing is None:
                    conn.execute(
                        """
                        INSERT INTO credit_llm_usage_entries (
                            user_id, reservation_id, run_id, call_index,
                            bucket, amount_micro, operation_key,
                            evidence_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            user_id,
                            reservation["reservation_id"],
                            reservation["run_id"],
                            reservation["call_index"],
                            bucket,
                            -debit,
                            operation_key,
                            evidence_json,
                            _utcnow_iso(),
                        ),
                    )
            conn.execute(
                """
                UPDATE credit_llm_reservations
                SET outstanding_recovered_micro =
                    outstanding_recovered_micro + ?, updated_at = ?
                WHERE reservation_id = ?
                """,
                (amount, _utcnow_iso(), reservation["reservation_id"]),
            )
            grant_available -= grant_debit
            purchased_available -= purchased_debit
            remaining_funds -= amount
            recovered_total += amount

        remaining = conn.execute(
            """
            SELECT COALESCE(SUM(
                MAX(outstanding_micro - outstanding_recovered_micro, 0)
            ), 0) AS outstanding_micro
            FROM credit_llm_reservations
            WHERE user_id = ? AND status = 'settled'
            """,
            (user_id,),
        ).fetchone()
        outstanding = int(remaining["outstanding_micro"])
        if outstanding == 0:
            conn.execute(
                """
                UPDATE credit_accounts
                SET status = 'active', restriction_reason = NULL
                WHERE user_id = ? AND status = 'restricted'
                  AND restriction_reason = 'llm_overage'
                """,
                (user_id,),
            )
            account_status = "active"
            reason = None
        else:
            account_status = "restricted"
        return {
            "recovered_micro": recovered_total,
            "outstanding_micro": outstanding,
            "account_status": account_status,
            "restriction_reason": reason,
        }

    def release_llm_credits(
        self,
        reservation_id: str,
        *,
        reason: str,
    ) -> dict[str, Any]:
        reservation_id = _required_text(
            reservation_id, "reservation_id", max_length=160
        )
        reason = _required_text(reason, "reason", max_length=120)
        with self._get_connection() as conn:
            self._begin(conn)
            reservation = conn.execute(
                "SELECT * FROM credit_llm_reservations WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
            if not reservation:
                raise LLMReservationConflictError("reservation was not found")
            if reservation["status"] == "settled":
                return self._llm_reservation_result_in_transaction(
                    conn, reservation
                )
            if reservation["status"] == "released":
                return self._llm_reservation_result_in_transaction(
                    conn, reservation
                )
            now = _utcnow_iso()
            conn.execute(
                """
                UPDATE credit_llm_reservations
                SET status = 'released', failure_reason = ?, updated_at = ?
                WHERE reservation_id = ?
                """,
                (reason, now, reservation_id),
            )
            released = conn.execute(
                "SELECT * FROM credit_llm_reservations WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
            return self._llm_reservation_result_in_transaction(conn, released)

    def release_run_llm_reservations(
        self,
        run_id: str,
        *,
        reason: str,
    ) -> list[dict[str, Any]]:
        run_id = _required_text(run_id, "run_id", max_length=128)
        reason = _required_text(reason, "reason", max_length=120)
        with self._get_connection() as conn:
            self._begin(conn)
            now = _utcnow_iso()
            conn.execute(
                """
                UPDATE credit_llm_reservations
                SET status = 'released', failure_reason = ?, updated_at = ?
                WHERE run_id = ? AND status = 'open'
                """,
                (reason, now, run_id),
            )
            rows = conn.execute(
                """
                SELECT * FROM credit_llm_reservations
                WHERE run_id = ? ORDER BY call_index, attempt_index, reservation_id
                """,
                (run_id,),
            ).fetchall()
            return [
                self._llm_reservation_result_in_transaction(conn, row)
                for row in rows
            ]

    def create_or_get_order(
        self,
        *,
        order_id: str,
        user_id: int,
        client_request_id: str,
        amount_usd_cents: int,
        credits_micro: int,
    ) -> dict[str, Any]:
        _validate_amount_pair(amount_usd_cents, credits_micro)
        _positive_integer(user_id, "user_id")
        if not str(order_id).strip() or not str(client_request_id).strip():
            raise ValueError("order_id and client_request_id are required")
        with self._get_connection() as conn:
            self._begin(conn)
            existing = conn.execute(
                """
                SELECT * FROM credit_payment_orders
                WHERE user_id = ? AND client_request_id = ?
                """,
                (user_id, client_request_id),
            ).fetchone()
            if existing:
                if (
                    existing["amount_usd_cents"] != amount_usd_cents
                    or existing["credits_micro"] != credits_micro
                    or existing["currency"] != "usd"
                    or existing["stripe_mode"] != "test"
                ):
                    raise OrderConflictError(
                        "client request already represents a different purchase"
                    )
                return dict(existing)

            self._ensure_account_in_transaction(conn, user_id)
            now = _utcnow_iso()
            try:
                conn.execute(
                    """
                    INSERT INTO credit_payment_orders (
                        id, user_id, client_request_id, stripe_mode, currency,
                        amount_usd_cents, credits_micro, status, created_at, updated_at
                    )
                    VALUES (?, ?, ?, 'test', 'usd', ?, ?, 'pending', ?, ?)
                    """,
                    (
                        order_id,
                        user_id,
                        client_request_id,
                        amount_usd_cents,
                        credits_micro,
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise OrderConflictError("order ID already exists") from exc
            row = conn.execute(
                "SELECT * FROM credit_payment_orders WHERE id = ?", (order_id,)
            ).fetchone()
            return dict(row)

    def attach_checkout_session(
        self, order_id: str, *, checkout_session_id: str
    ) -> dict[str, Any]:
        if not str(checkout_session_id).strip():
            raise ValueError("checkout_session_id is required")
        with self._get_connection() as conn:
            self._begin(conn)
            row = conn.execute(
                "SELECT * FROM credit_payment_orders WHERE id = ?", (order_id,)
            ).fetchone()
            if not row:
                raise KeyError("payment order not found")
            current = row["stripe_checkout_session_id"]
            if current and current != checkout_session_id:
                raise OrderConflictError(
                    "payment order already has a different Checkout Session"
                )
            if not current:
                try:
                    conn.execute(
                        """
                        UPDATE credit_payment_orders
                        SET stripe_checkout_session_id = ?, updated_at = ?
                        WHERE id = ? AND stripe_checkout_session_id IS NULL
                        """,
                        (checkout_session_id, _utcnow_iso(), order_id),
                    )
                except sqlite3.IntegrityError as exc:
                    raise OrderConflictError(
                        "Checkout Session is already attached to another order"
                    ) from exc
            updated = conn.execute(
                "SELECT * FROM credit_payment_orders WHERE id = ?", (order_id,)
            ).fetchone()
            return dict(updated)

    @staticmethod
    def _existing_event(
        conn: sqlite3.Connection,
        *,
        event_id: str,
        event_type: str,
        livemode: bool,
        object_id: str,
        payload_sha256: str,
    ) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT * FROM stripe_webhook_events WHERE stripe_event_id = ?",
            (event_id,),
        ).fetchone()
        if not row:
            return None
        if (
            row["event_type"] != event_type
            or bool(row["livemode"]) != bool(livemode)
            or row["object_id"] != object_id
            or row["payload_sha256"] != payload_sha256
        ):
            raise OrderConflictError("Stripe event ID was reused with different data")
        return dict(row)

    @staticmethod
    def _insert_event(
        conn: sqlite3.Connection,
        *,
        event_id: str,
        event_type: str,
        livemode: bool,
        object_id: str,
        payload_sha256: str,
        outcome: str,
        reason: str | None = None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO stripe_webhook_events (
                stripe_event_id, event_type, livemode, object_id,
                payload_sha256, outcome, reason, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                event_type,
                int(bool(livemode)),
                object_id,
                payload_sha256,
                outcome,
                reason,
                _utcnow_iso(),
            ),
        )

    def record_webhook_event(
        self,
        *,
        event_id: str,
        event_type: str,
        livemode: bool,
        object_id: str,
        payload_sha256: str,
        outcome: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        if outcome not in {"processed", "ignored", "rejected"}:
            raise ValueError("invalid webhook outcome")
        with self._get_connection() as conn:
            self._begin(conn)
            existing = self._existing_event(
                conn,
                event_id=event_id,
                event_type=event_type,
                livemode=livemode,
                object_id=object_id,
                payload_sha256=payload_sha256,
            )
            if existing:
                return {"outcome": "duplicate", "reason": existing["reason"]}
            self._insert_event(
                conn,
                event_id=event_id,
                event_type=event_type,
                livemode=livemode,
                object_id=object_id,
                payload_sha256=payload_sha256,
                outcome=outcome,
                reason=reason,
            )
            return {"outcome": outcome, "reason": reason}

    def settle_unpaid_checkout(
        self,
        *,
        event_id: str,
        event_type: str,
        livemode: bool,
        object_id: str,
        payload_sha256: str,
        order_id: str,
        checkout_session_id: str,
        terminal_status: str,
    ) -> dict[str, Any]:
        if terminal_status not in {"expired", "failed"}:
            raise ValueError("invalid unpaid Checkout terminal status")
        with self._get_connection() as conn:
            self._begin(conn)
            existing = self._existing_event(
                conn,
                event_id=event_id,
                event_type=event_type,
                livemode=livemode,
                object_id=object_id,
                payload_sha256=payload_sha256,
            )
            if existing:
                return {"outcome": "duplicate", "status": terminal_status}

            order = conn.execute(
                "SELECT * FROM credit_payment_orders WHERE id = ?", (order_id,)
            ).fetchone()
            reason = None
            if not order:
                reason = "payment order not found"
            elif livemode or order["stripe_mode"] != "test":
                reason = "Live Mode payment is not accepted"
            # NULL means "not recorded yet" — see settle_paid_checkout.
            elif order["stripe_checkout_session_id"] not in (
                None,
                checkout_session_id,
            ):
                reason = "Checkout Session does not match the order"
            elif object_id != checkout_session_id:
                reason = "event object does not match the Checkout Session"
            if reason:
                self._insert_event(
                    conn,
                    event_id=event_id,
                    event_type=event_type,
                    livemode=livemode,
                    object_id=object_id,
                    payload_sha256=payload_sha256,
                    outcome="rejected",
                    reason=reason,
                )
                return {"outcome": "rejected", "reason": reason}

            if order["status"] != "pending":
                reason = f"payment order is already {order['status']}"
                self._insert_event(
                    conn,
                    event_id=event_id,
                    event_type=event_type,
                    livemode=livemode,
                    object_id=object_id,
                    payload_sha256=payload_sha256,
                    outcome="ignored",
                    reason=reason,
                )
                return {"outcome": "ignored", "reason": reason, "status": order["status"]}

            now = _utcnow_iso()
            self._insert_event(
                conn,
                event_id=event_id,
                event_type=event_type,
                livemode=livemode,
                object_id=object_id,
                payload_sha256=payload_sha256,
                outcome="processed",
            )
            conn.execute(
                "UPDATE credit_payment_orders SET status = ?, updated_at = ? WHERE id = ?",
                (terminal_status, now, order_id),
            )
            return {"outcome": "processed", "status": terminal_status}

    def settle_paid_checkout(
        self,
        *,
        event_id: str,
        event_type: str,
        livemode: bool,
        object_id: str,
        payload_sha256: str,
        order_id: str,
        checkout_session_id: str,
        payment_intent_id: str,
        currency: str,
        amount_usd_cents: int,
    ) -> dict[str, Any]:
        with self._get_connection() as conn:
            self._begin(conn)
            existing_event = self._existing_event(
                conn,
                event_id=event_id,
                event_type=event_type,
                livemode=livemode,
                object_id=object_id,
                payload_sha256=payload_sha256,
            )
            if existing_event:
                order = conn.execute(
                    "SELECT user_id FROM credit_payment_orders WHERE id = ?",
                    (order_id,),
                ).fetchone()
                balance = self._balance_in_transaction(conn, order["user_id"]) if order else 0
                return {"outcome": "duplicate", "balance_micro": balance}

            order = conn.execute(
                "SELECT * FROM credit_payment_orders WHERE id = ?", (order_id,)
            ).fetchone()
            reason = None
            if not order:
                reason = "payment order not found"
            elif livemode or order["stripe_mode"] != "test":
                reason = "Live Mode payment is not accepted"
            elif currency.lower() != order["currency"]:
                reason = "payment currency does not match the order"
            elif amount_usd_cents != order["amount_usd_cents"]:
                reason = "payment amount does not match the order"
            # NULL means "not recorded yet", not "mismatch" — the same idiom the
            # PaymentIntent check below already uses. attach_checkout_session
            # runs *after* Stripe has created a payable session, so a crash or
            # restart in that window leaves this column NULL; treating that as a
            # mismatch permanently rejects the payment webhook for an order the
            # customer has already been charged for, and writes an event row
            # that makes the rejection unreplayable. Provenance does not rest on
            # this column: the caller has already matched the signed event's
            # atl_order_id / atl_user_reference / atl_credits_micro metadata
            # against the order, and currency and amount are checked above.
            elif order["stripe_checkout_session_id"] not in (
                None,
                checkout_session_id,
            ):
                reason = "Checkout Session does not match the order"
            elif object_id != checkout_session_id:
                reason = "event object does not match the Checkout Session"
            elif order["stripe_payment_intent_id"] not in (None, payment_intent_id):
                reason = "PaymentIntent does not match the order"

            if reason:
                self._insert_event(
                    conn,
                    event_id=event_id,
                    event_type=event_type,
                    livemode=livemode,
                    object_id=object_id,
                    payload_sha256=payload_sha256,
                    outcome="rejected",
                    reason=reason,
                )
                return {"outcome": "rejected", "reason": reason}

            operation_key = f"purchase:{order_id}"
            existing_entry = conn.execute(
                """
                SELECT id FROM credit_ledger_entries WHERE operation_key = ?
                """,
                (operation_key,),
            ).fetchone()
            if existing_entry:
                self._insert_event(
                    conn,
                    event_id=event_id,
                    event_type=event_type,
                    livemode=livemode,
                    object_id=object_id,
                    payload_sha256=payload_sha256,
                    outcome="ignored",
                    reason="purchase already posted",
                )
                return {
                    "outcome": "duplicate",
                    "balance_micro": self._balance_in_transaction(conn, order["user_id"]),
                }

            now = _utcnow_iso()
            self._insert_event(
                conn,
                event_id=event_id,
                event_type=event_type,
                livemode=livemode,
                object_id=object_id,
                payload_sha256=payload_sha256,
                outcome="processed",
            )
            conn.execute(
                """
                INSERT INTO credit_ledger_entries (
                    user_id, bucket, entry_type, amount_micro, payment_order_id,
                    refund_request_id, stripe_event_id, operation_key,
                    operation_id, idempotency_key, source, reason, created_at
                )
                VALUES (
                    ?, 'purchased', 'purchase', ?, ?, NULL, ?, ?, ?, ?,
                    'stripe', 'Stripe checkout purchase.', ?
                )
                """,
                (
                    order["user_id"],
                    order["credits_micro"],
                    order_id,
                    event_id,
                    operation_key,
                    operation_key,
                    operation_key,
                    now,
                ),
            )
            conn.execute(
                """
                UPDATE credit_payment_orders
                SET status = 'paid', stripe_payment_intent_id = ?,
                    stripe_checkout_session_id =
                        COALESCE(stripe_checkout_session_id, ?),
                    updated_at = ?, paid_at = COALESCE(paid_at, ?)
                WHERE id = ?
                """,
                (payment_intent_id, checkout_session_id, now, now, order_id),
            )
            recovery = self._recover_llm_overage_in_transaction(
                conn, order["user_id"], source_operation_key=operation_key
            )
            return {
                "outcome": "processed",
                "balance_micro": self._balance_in_transaction(conn, order["user_id"]),
                **recovery,
            }

    @staticmethod
    def _balance_in_transaction(conn: sqlite3.Connection, user_id: int) -> int:
        projection = CreditsStore._balance_projection_in_transaction(conn, user_id)
        return projection["total_available_micro"]

    def get_order_for_user(
        self, order_id: str, user_id: int
    ) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM credit_payment_orders
                WHERE id = ? AND user_id = ?
                """,
                (order_id, user_id),
            ).fetchone()
            return _dict(row)

    def get_order_for_admin(self, order_id: str) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM credit_payment_orders WHERE id = ?", (order_id,)
            ).fetchone()
            return _dict(row)

    def get_order_by_payment_intent(
        self, payment_intent_id: str
    ) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM credit_payment_orders
                WHERE stripe_payment_intent_id = ?
                """,
                (payment_intent_id,),
            ).fetchone()
            return _dict(row)

    def get_refund_by_stripe_id(
        self, stripe_refund_id: str
    ) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM credit_refund_requests WHERE stripe_refund_id = ?
                """,
                (stripe_refund_id,),
            ).fetchone()
            return _dict(row)

    def get_refund_by_id(self, refund_id: str) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM credit_refund_requests WHERE id = ?", (refund_id,)
            ).fetchone()
            return _dict(row)

    def restrict_account(
        self,
        user_id: int,
        *,
        reason: str = "refund_reconciliation",
    ) -> dict[str, Any]:
        _positive_integer(user_id, "user_id")
        if reason not in {"llm_overage", "refund_reconciliation"}:
            raise ValueError("invalid account restriction reason")
        with self._get_connection() as conn:
            self._begin(conn)
            self._ensure_account_in_transaction(conn, user_id)
            conn.execute(
                """
                UPDATE credit_accounts
                SET status = 'restricted', restriction_reason = ?
                WHERE user_id = ?
                """,
                (reason, user_id),
            )
            row = conn.execute(
                "SELECT * FROM credit_accounts WHERE user_id = ?", (user_id,)
            ).fetchone()
            return dict(row)

    def reinstate_account(self, user_id: int) -> dict[str, Any]:
        """Clear a restriction after an admin has reconciled the refund.

        ``restrict_account`` is written automatically when an out-of-band Stripe
        refund exceeds an order's refundable amount. With the purchase gate now
        enforced server-side rather than only in the browser, that flag is a
        hard stop, so leaving it clearable only by hand-written SQL would make
        an automatic action permanently un-undoable by the operator it is
        pointed at.
        """
        _positive_integer(user_id, "user_id")
        with self._get_connection() as conn:
            self._begin(conn)
            self._ensure_account_in_transaction(conn, user_id)
            conn.execute(
                "UPDATE credit_accounts SET status = 'active', restriction_reason = NULL WHERE user_id = ?",
                (user_id,),
            )
            row = conn.execute(
                "SELECT * FROM credit_accounts WHERE user_id = ?", (user_id,)
            ).fetchone()
            return dict(row)

    def list_ledger_entries(
        self,
        user_id: int,
        *,
        limit: int = 50,
        cursor: str | int | None = None,
    ) -> dict[str, Any]:
        page_size = _positive_limit(limit)
        boundary: tuple[str, str, int] | None = None
        with self._get_connection() as conn:
            if cursor is not None:
                decoded = decode_activity_cursor(cursor)
                if isinstance(decoded, int):
                    legacy = conn.execute(
                        "SELECT created_at FROM credit_ledger_entries "
                        "WHERE user_id = ? AND id = ?",
                        (user_id, decoded),
                    ).fetchone()
                    if not legacy:
                        raise ValueError("invalid activity cursor")
                    boundary = (str(legacy["created_at"]), "ledger", decoded)
                else:
                    boundary = decoded
            params: list[Any] = [user_id, user_id, user_id]
            cursor_sql = ""
            if boundary is not None:
                created_at, source_kind, source_id = boundary
                cursor_sql = """
                    WHERE created_at < ?
                       OR (created_at = ? AND source_kind < ?)
                       OR (created_at = ? AND source_kind = ? AND source_id < ?)
                """
                params.extend([
                    created_at,
                    created_at,
                    source_kind,
                    created_at,
                    source_kind,
                    source_id,
                ])
            params.append(page_size + 1)
            rows = conn.execute(
                f"""
                WITH historical_activity AS (
                    SELECT id AS source_id, 'ledger' AS source_kind,
                           user_id, bucket, entry_type, amount_micro,
                           payment_order_id, refund_request_id, stripe_event_id,
                           operation_key, operation_id, idempotency_key,
                           request_digest, actor_user_id, source, reason,
                           reference_type, reference_id, created_at,
                           NULL AS reservation_id, NULL AS run_id,
                           NULL AS call_index, NULL AS model_call_count,
                           NULL AS evidence_json
                    FROM credit_ledger_entries
                    WHERE user_id = ?
                ),
                llm_activity AS (
                    SELECT MAX(id) AS source_id, 'llm_usage' AS source_kind,
                           user_id, NULL AS bucket,
                           'backtest_usage' AS entry_type,
                           SUM(amount_micro) AS amount_micro,
                           NULL AS payment_order_id,
                           NULL AS refund_request_id, NULL AS stripe_event_id,
                           NULL AS operation_key, NULL AS operation_id,
                           NULL AS idempotency_key, NULL AS request_digest,
                           NULL AS actor_user_id, 'llm_execution' AS source,
                           'Backtest usage.' AS reason,
                           NULL AS reference_type, NULL AS reference_id,
                           MAX(created_at) AS created_at,
                           NULL AS reservation_id, run_id, NULL AS call_index,
                           COUNT(DISTINCT call_index) AS model_call_count,
                           NULL AS evidence_json
                    FROM credit_llm_usage_entries
                    WHERE user_id = ? AND operation_key NOT LIKE '%:recovery:%'
                    GROUP BY user_id, run_id
                ),
                promotion_activity AS (
                    SELECT id AS source_id, 'promotion' AS source_kind,
                           user_id, 'grant' AS bucket,
                           'system_promotion_grant' AS entry_type,
                           amount_micro,
                           NULL AS payment_order_id,
                           NULL AS refund_request_id,
                           NULL AS stripe_event_id,
                           'promotion:' || operation_id AS operation_key,
                           operation_id, idempotency_key,
                           request_digest, NULL AS actor_user_id,
                           source, reason, 'promotion' AS reference_type,
                           campaign_key AS reference_id, created_at,
                           NULL AS reservation_id, NULL AS run_id,
                           NULL AS call_index, NULL AS model_call_count,
                           NULL AS evidence_json
                    FROM credit_promotion_grants
                    WHERE user_id = ?
                ),
                activity AS (
                    SELECT * FROM historical_activity
                    UNION ALL
                    SELECT * FROM llm_activity
                    UNION ALL
                    SELECT * FROM promotion_activity
                )
                SELECT * FROM activity
                {cursor_sql}
                ORDER BY created_at DESC, source_kind DESC, source_id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
            selected_rows = rows[:page_size]
            run_ids = list(
                dict.fromkeys(
                    str(row["run_id"])
                    for row in selected_rows
                    if row["source_kind"] == "llm_usage"
                    and row["run_id"] is not None
                )
            )
            evidence_by_run: dict[str, list[object]] = {
                run_id: [] for run_id in run_ids
            }
            if run_ids:
                placeholders = ", ".join("?" for _ in run_ids)
                evidence_rows = conn.execute(
                    f"""
                    SELECT DISTINCT run_id, evidence_json
                    FROM credit_llm_usage_entries
                    WHERE user_id = ? AND run_id IN ({placeholders})
                      AND operation_key NOT LIKE '%:recovery:%'
                    """,
                    [user_id, *run_ids],
                ).fetchall()
                for evidence_row in evidence_rows:
                    evidence_by_run[str(evidence_row["run_id"])].append(
                        evidence_row["evidence_json"]
                    )
        has_more = len(rows) > page_size
        items = [
            normalize_activity_item(
                dict(row),
                evidence_json_values=evidence_by_run.get(str(row["run_id"]), ()),
            )
            for row in selected_rows
        ]
        return {
            "items": items,
            "next_cursor": (
                encode_activity_cursor(
                    str(items[-1]["created_at"]),
                    str(items[-1]["source_kind"]),
                    int(items[-1]["id"]),
                )
                if has_more and items
                else None
            ),
        }

    @staticmethod
    def _refundable_in_transaction(
        conn: sqlite3.Connection, order: sqlite3.Row
    ) -> tuple[int, int]:
        row = conn.execute(
            """
            SELECT
                COALESCE(SUM(amount_usd_cents), 0) AS reserved_cents,
                COALESCE(SUM(credits_micro), 0) AS reserved_micro
            FROM credit_refund_requests
            WHERE payment_order_id = ?
              AND status IN ('pending', 'submitted', 'succeeded')
            """,
            (order["id"],),
        ).fetchone()
        return (
            int(order["amount_usd_cents"]) - int(row["reserved_cents"]),
            int(order["credits_micro"]) - int(row["reserved_micro"]),
        )

    def reserve_refund(
        self,
        *,
        refund_id: str,
        payment_order_id: str,
        user_id: int,
        requested_by_user_id: int,
        amount_usd_cents: int,
        credits_micro: int,
    ) -> dict[str, Any]:
        _validate_amount_pair(amount_usd_cents, credits_micro)
        with self._get_connection() as conn:
            self._begin(conn)
            existing = conn.execute(
                "SELECT * FROM credit_refund_requests WHERE id = ?", (refund_id,)
            ).fetchone()
            if existing:
                if (
                    existing["payment_order_id"] != payment_order_id
                    or existing["user_id"] != user_id
                    or existing["requested_by_user_id"] != requested_by_user_id
                    or existing["amount_usd_cents"] != amount_usd_cents
                    or existing["credits_micro"] != credits_micro
                ):
                    raise OrderConflictError(
                        "refund ID already represents a different request"
                    )
                return dict(existing)

            order = conn.execute(
                "SELECT * FROM credit_payment_orders WHERE id = ?",
                (payment_order_id,),
            ).fetchone()
            if not order or order["user_id"] != user_id:
                raise RefundNotAllowedError("paid purchase was not found")
            if order["status"] not in {"paid", "partially_refunded"}:
                raise RefundNotAllowedError("purchase is not refundable")
            refundable_cents, refundable_micro = self._refundable_in_transaction(
                conn, order
            )
            if (
                amount_usd_cents > refundable_cents
                or credits_micro > refundable_micro
            ):
                raise RefundNotAllowedError(
                    "refund exceeds the unused purchased Credits"
                )
            now = _utcnow_iso()
            conn.execute(
                """
                INSERT INTO credit_refund_requests (
                    id, payment_order_id, user_id, requested_by_user_id,
                    amount_usd_cents, credits_micro, status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    refund_id,
                    payment_order_id,
                    user_id,
                    requested_by_user_id,
                    amount_usd_cents,
                    credits_micro,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM credit_refund_requests WHERE id = ?", (refund_id,)
            ).fetchone()
            return dict(row)

    def reserve_reconciliation_refund(
        self,
        *,
        refund_id: str,
        payment_order_id: str,
        user_id: int,
        amount_usd_cents: int,
        credits_micro: int,
        stripe_refund_id: str,
    ) -> dict[str, Any]:
        _validate_amount_pair(amount_usd_cents, credits_micro)
        with self._get_connection() as conn:
            self._begin(conn)
            existing = conn.execute(
                """
                SELECT * FROM credit_refund_requests
                WHERE id = ? OR stripe_refund_id = ?
                """,
                (refund_id, stripe_refund_id),
            ).fetchone()
            if existing:
                if (
                    existing["payment_order_id"] != payment_order_id
                    or existing["user_id"] != user_id
                    or existing["amount_usd_cents"] != amount_usd_cents
                    or existing["credits_micro"] != credits_micro
                    or existing["stripe_refund_id"] != stripe_refund_id
                ):
                    raise OrderConflictError(
                        "Stripe Refund already represents a different request"
                    )
                return dict(existing)

            order = conn.execute(
                "SELECT * FROM credit_payment_orders WHERE id = ?",
                (payment_order_id,),
            ).fetchone()
            if not order or order["user_id"] != user_id:
                raise RefundNotAllowedError("paid purchase was not found")
            if order["status"] not in {"paid", "partially_refunded"}:
                raise RefundNotAllowedError("purchase is not refundable")
            refundable_cents, refundable_micro = self._refundable_in_transaction(
                conn, order
            )
            if (
                amount_usd_cents > refundable_cents
                or credits_micro > refundable_micro
            ):
                raise RefundNotAllowedError(
                    "refund exceeds the unused purchased Credits"
                )
            now = _utcnow_iso()
            try:
                conn.execute(
                    """
                    INSERT INTO credit_refund_requests (
                        id, payment_order_id, user_id, requested_by_user_id,
                        amount_usd_cents, credits_micro, status,
                        stripe_refund_id, created_at, updated_at
                    )
                    VALUES (?, ?, ?, NULL, ?, ?, 'submitted', ?, ?, ?)
                    """,
                    (
                        refund_id,
                        payment_order_id,
                        user_id,
                        amount_usd_cents,
                        credits_micro,
                        stripe_refund_id,
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise OrderConflictError(
                    "Stripe Refund is already attached to another request"
                ) from exc
            row = conn.execute(
                "SELECT * FROM credit_refund_requests WHERE id = ?", (refund_id,)
            ).fetchone()
            return dict(row)

    def attach_stripe_refund(
        self, refund_id: str, *, stripe_refund_id: str
    ) -> dict[str, Any]:
        with self._get_connection() as conn:
            self._begin(conn)
            row = conn.execute(
                "SELECT * FROM credit_refund_requests WHERE id = ?", (refund_id,)
            ).fetchone()
            if not row:
                raise KeyError("refund request not found")
            current = row["stripe_refund_id"]
            if current and current != stripe_refund_id:
                raise OrderConflictError(
                    "refund request already has a different Stripe Refund"
                )
            if row["status"] not in {"pending", "submitted"}:
                if current == stripe_refund_id:
                    return dict(row)
                raise OrderConflictError("refund request is already terminal")
            if not current:
                try:
                    conn.execute(
                        """
                        UPDATE credit_refund_requests
                        SET stripe_refund_id = ?, status = 'submitted', updated_at = ?
                        WHERE id = ?
                        """,
                        (stripe_refund_id, _utcnow_iso(), refund_id),
                    )
                except sqlite3.IntegrityError as exc:
                    raise OrderConflictError(
                        "Stripe Refund is already attached to another request"
                    ) from exc
            updated = conn.execute(
                "SELECT * FROM credit_refund_requests WHERE id = ?", (refund_id,)
            ).fetchone()
            return dict(updated)

    def cancel_refund_reservation(self, refund_id: str) -> dict[str, Any] | None:
        """Release a reservation whose Stripe call never landed.

        ``_refundable_in_transaction`` counts ``pending``/``submitted``/
        ``succeeded`` against the order's refundable lot, so a reservation left
        ``pending`` by a failed Stripe call subtracts from that lot forever: one
        transient 5xx on a full refund makes the order permanently
        un-refundable. ``cancelled`` is declared in both twins' CHECK
        constraints for exactly this and was previously never written.

        Only a still-``pending`` row with no Stripe Refund attached is
        cancellable. Once ``attach_stripe_refund`` has moved it to
        ``submitted`` the money is genuinely in flight, and releasing the lot
        then would let an admin over-refund the order.
        """
        with self._get_connection() as conn:
            self._begin(conn)
            row = conn.execute(
                "SELECT * FROM credit_refund_requests WHERE id = ?", (refund_id,)
            ).fetchone()
            if not row:
                return None
            if row["status"] != "pending" or row["stripe_refund_id"]:
                return dict(row)
            conn.execute(
                """
                UPDATE credit_refund_requests
                SET status = 'cancelled', updated_at = ?
                WHERE id = ? AND status = 'pending'
                  AND stripe_refund_id IS NULL
                """,
                (_utcnow_iso(), refund_id),
            )
            updated = conn.execute(
                "SELECT * FROM credit_refund_requests WHERE id = ?", (refund_id,)
            ).fetchone()
            return dict(updated)

    def settle_succeeded_refund(
        self,
        *,
        event_id: str,
        event_type: str,
        livemode: bool,
        object_id: str,
        payload_sha256: str,
        refund_id: str,
        stripe_refund_id: str,
        payment_intent_id: str,
        currency: str,
        amount_usd_cents: int,
    ) -> dict[str, Any]:
        with self._get_connection() as conn:
            self._begin(conn)
            existing_event = self._existing_event(
                conn,
                event_id=event_id,
                event_type=event_type,
                livemode=livemode,
                object_id=object_id,
                payload_sha256=payload_sha256,
            )
            if existing_event:
                refund = conn.execute(
                    "SELECT user_id FROM credit_refund_requests WHERE id = ?",
                    (refund_id,),
                ).fetchone()
                balance = self._balance_in_transaction(conn, refund["user_id"]) if refund else 0
                return {"outcome": "duplicate", "balance_micro": balance}

            refund = conn.execute(
                "SELECT * FROM credit_refund_requests WHERE id = ?", (refund_id,)
            ).fetchone()
            order = (
                conn.execute(
                    "SELECT * FROM credit_payment_orders WHERE id = ?",
                    (refund["payment_order_id"],),
                ).fetchone()
                if refund
                else None
            )
            reason = None
            if not refund or not order:
                reason = "refund request was not found"
            elif livemode:
                reason = "Live Mode refund is not accepted"
            elif object_id != stripe_refund_id or refund["stripe_refund_id"] != stripe_refund_id:
                reason = "Stripe Refund does not match the request"
            elif order["stripe_payment_intent_id"] != payment_intent_id:
                reason = "PaymentIntent does not match the purchase"
            elif currency.lower() != order["currency"]:
                reason = "refund currency does not match the purchase"
            elif amount_usd_cents != refund["amount_usd_cents"]:
                reason = "refund amount does not match the request"

            if reason:
                self._insert_event(
                    conn,
                    event_id=event_id,
                    event_type=event_type,
                    livemode=livemode,
                    object_id=object_id,
                    payload_sha256=payload_sha256,
                    outcome="rejected",
                    reason=reason,
                )
                return {"outcome": "rejected", "reason": reason}

            operation_key = f"refund:{refund_id}"
            existing_entry = conn.execute(
                "SELECT id FROM credit_ledger_entries WHERE operation_key = ?",
                (operation_key,),
            ).fetchone()
            if existing_entry or refund["status"] == "succeeded":
                self._insert_event(
                    conn,
                    event_id=event_id,
                    event_type=event_type,
                    livemode=livemode,
                    object_id=object_id,
                    payload_sha256=payload_sha256,
                    outcome="ignored",
                    reason="refund already posted",
                )
                return {
                    "outcome": "duplicate",
                    "balance_micro": self._balance_in_transaction(conn, refund["user_id"]),
                }
            if refund["status"] not in {"pending", "submitted"}:
                reason = "refund request is not awaiting settlement"
                self._insert_event(
                    conn,
                    event_id=event_id,
                    event_type=event_type,
                    livemode=livemode,
                    object_id=object_id,
                    payload_sha256=payload_sha256,
                    outcome="rejected",
                    reason=reason,
                )
                return {"outcome": "rejected", "reason": reason}

            now = _utcnow_iso()
            self._insert_event(
                conn,
                event_id=event_id,
                event_type=event_type,
                livemode=livemode,
                object_id=object_id,
                payload_sha256=payload_sha256,
                outcome="processed",
            )
            conn.execute(
                """
                INSERT INTO credit_ledger_entries (
                    user_id, bucket, entry_type, amount_micro, payment_order_id,
                    refund_request_id, stripe_event_id, operation_key,
                    operation_id, idempotency_key, source, reason, created_at
                )
                VALUES (
                    ?, 'purchased', 'refund', ?, ?, ?, ?, ?, ?, ?,
                    'stripe', 'Stripe refund.', ?
                )
                """,
                (
                    refund["user_id"],
                    -int(refund["credits_micro"]),
                    refund["payment_order_id"],
                    refund_id,
                    event_id,
                    operation_key,
                    operation_key,
                    operation_key,
                    now,
                ),
            )
            conn.execute(
                """
                UPDATE credit_refund_requests
                SET status = 'succeeded', updated_at = ?, succeeded_at = ?
                WHERE id = ?
                """,
                (now, now, refund_id),
            )
            successful = conn.execute(
                """
                SELECT COALESCE(SUM(amount_usd_cents), 0) AS cents
                FROM credit_refund_requests
                WHERE payment_order_id = ? AND status = 'succeeded'
                """,
                (refund["payment_order_id"],),
            ).fetchone()
            order_status = (
                "refunded"
                if int(successful["cents"]) >= int(order["amount_usd_cents"])
                else "partially_refunded"
            )
            conn.execute(
                """
                UPDATE credit_payment_orders SET status = ?, updated_at = ?
                WHERE id = ?
                """,
                (order_status, now, refund["payment_order_id"]),
            )
            return {
                "outcome": "processed",
                "balance_micro": self._balance_in_transaction(conn, refund["user_id"]),
            }

    def fail_refund(
        self,
        *,
        event_id: str,
        event_type: str,
        livemode: bool,
        object_id: str,
        payload_sha256: str,
        refund_id: str,
        stripe_refund_id: str,
    ) -> dict[str, Any]:
        with self._get_connection() as conn:
            self._begin(conn)
            existing = self._existing_event(
                conn,
                event_id=event_id,
                event_type=event_type,
                livemode=livemode,
                object_id=object_id,
                payload_sha256=payload_sha256,
            )
            if existing:
                return {"outcome": "duplicate"}
            refund = conn.execute(
                "SELECT * FROM credit_refund_requests WHERE id = ?", (refund_id,)
            ).fetchone()
            if (
                not refund
                or livemode
                or object_id != stripe_refund_id
                or refund["stripe_refund_id"] != stripe_refund_id
                or refund["status"] not in {"pending", "submitted"}
            ):
                reason = "refund failure event does not match an active request"
                self._insert_event(
                    conn,
                    event_id=event_id,
                    event_type=event_type,
                    livemode=livemode,
                    object_id=object_id,
                    payload_sha256=payload_sha256,
                    outcome="rejected",
                    reason=reason,
                )
                return {"outcome": "rejected", "reason": reason}
            now = _utcnow_iso()
            self._insert_event(
                conn,
                event_id=event_id,
                event_type=event_type,
                livemode=livemode,
                object_id=object_id,
                payload_sha256=payload_sha256,
                outcome="processed",
            )
            conn.execute(
                """
                UPDATE credit_refund_requests
                SET status = 'failed', updated_at = ? WHERE id = ?
                """,
                (now, refund_id),
            )
            return {"outcome": "processed"}

    def list_orders_for_admin(
        self, *, limit: int = 50, cursor: int | None = None
    ) -> dict[str, Any]:
        page_size = _positive_limit(limit)
        params: list[Any] = []
        cursor_sql = ""
        if cursor is not None:
            _positive_integer(cursor, "cursor")
            cursor_sql = "AND o.sequence < ?"
            params.append(cursor)
        params.append(page_size + 1)
        with self._get_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    o.*,
                    a.status AS account_status,
                    o.amount_usd_cents - COALESCE((
                        SELECT SUM(r.amount_usd_cents)
                        FROM credit_refund_requests r
                        WHERE r.payment_order_id = o.id
                          AND r.status IN ('pending', 'submitted', 'succeeded')
                    ), 0) AS refundable_usd_cents,
                    o.credits_micro - COALESCE((
                        SELECT SUM(r.credits_micro)
                        FROM credit_refund_requests r
                        WHERE r.payment_order_id = o.id
                          AND r.status IN ('pending', 'submitted', 'succeeded')
                    ), 0) AS refundable_credits_micro
                FROM credit_payment_orders o
                JOIN credit_accounts a ON a.user_id = o.user_id
                WHERE o.status IN ('paid', 'partially_refunded', 'refunded')
                  {cursor_sql}
                ORDER BY o.sequence DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        has_more = len(rows) > page_size
        items = [dict(row) for row in rows[:page_size]]
        return {
            "items": items,
            "next_cursor": items[-1]["sequence"] if has_more and items else None,
        }

    @staticmethod
    def _ensure_pool_in_transaction(
        conn: sqlite3.Connection, pool_id: str
    ) -> sqlite3.Row:
        if pool_id == "default":
            conn.execute(
                """
                INSERT INTO credit_grant_pools (pool_id, name, status, created_at)
                VALUES ('default', 'Platform Research Grants', 'active', ?)
                ON CONFLICT(pool_id) DO NOTHING
                """,
                (_utcnow_iso(),),
            )
        pool = conn.execute(
            "SELECT * FROM credit_grant_pools WHERE pool_id = ?", (pool_id,)
        ).fetchone()
        if pool is None:
            raise ValueError("grant pool does not exist")
        return pool

    @staticmethod
    def _pool_balance_in_transaction(
        conn: sqlite3.Connection,
        pool_id: str,
        through_entry_id: int | None = None,
    ) -> int:
        cutoff_sql = ""
        params: list[Any] = [pool_id]
        if through_entry_id is not None:
            cutoff_sql = "AND id <= ?"
            params.append(through_entry_id)
        row = conn.execute(
            f"""
            SELECT COALESCE(SUM(amount_micro), 0) AS balance_micro
            FROM credit_grant_pool_ledger_entries
            WHERE pool_id = ?
              {cutoff_sql}
            """,
            params,
        ).fetchone()
        return int(row["balance_micro"])

    @staticmethod
    def _insert_user_grant_entry_in_transaction(
        conn: sqlite3.Connection,
        *,
        pool_id: str,
        user_id: int,
        entry_type: str,
        amount_micro: int,
        operation_id: str,
        idempotency_key: str,
        request_digest: str,
        actor_user_id: int,
        source: str,
        reason: str,
        created_at: str,
    ) -> int:
        cursor = conn.execute(
            """
            INSERT INTO credit_ledger_entries (
                user_id, bucket, entry_type, amount_micro,
                payment_order_id, refund_request_id, stripe_event_id,
                operation_key, operation_id, idempotency_key,
                request_digest, actor_user_id, source, reason,
                reference_type, reference_id, created_at
            )
            VALUES (
                ?, 'grant', ?, ?, NULL, NULL, NULL,
                ?, ?, ?, ?, ?, ?, ?, 'grant_pool', ?, ?
            )
            """,
            (
                user_id,
                entry_type,
                amount_micro,
                f"{operation_id}:user",
                operation_id,
                f"{idempotency_key}:user",
                request_digest,
                actor_user_id,
                source,
                reason,
                pool_id,
                created_at,
            ),
        )
        return int(cursor.lastrowid)

    @staticmethod
    def _insert_grant_pool_entry_in_transaction(
        conn: sqlite3.Connection,
        *,
        pool_id: str,
        pool_name_snapshot: str,
        pool_status_snapshot: str,
        entry_type: str,
        amount_micro: int,
        operation_id: str,
        idempotency_key: str,
        request_digest: str,
        actor_user_id: int,
        source: str,
        reason: str,
        user_id: int | None,
        user_ledger_entry_id: int | None,
        created_at: str,
    ) -> int:
        cursor = conn.execute(
            """
            INSERT INTO credit_grant_pool_ledger_entries (
                pool_id, pool_name_snapshot, pool_status_snapshot,
                entry_type, amount_micro, operation_id,
                idempotency_key, request_digest, actor_user_id,
                source, reason, user_id, user_ledger_entry_id, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pool_id,
                pool_name_snapshot,
                pool_status_snapshot,
                entry_type,
                amount_micro,
                operation_id,
                idempotency_key,
                request_digest,
                actor_user_id,
                source,
                reason,
                user_id,
                user_ledger_entry_id,
                created_at,
            ),
        )
        return int(cursor.lastrowid)

    @classmethod
    def _grant_mutation_result_in_transaction(
        cls, conn: sqlite3.Connection, pool_entry: sqlite3.Row
    ) -> dict[str, Any]:
        user_id = pool_entry["user_id"]
        user_entry = None
        user_balance = None
        if pool_entry["user_ledger_entry_id"] is not None:
            user_entry = conn.execute(
                "SELECT * FROM credit_ledger_entries WHERE id = ?",
                (pool_entry["user_ledger_entry_id"],),
            ).fetchone()
        if user_id is not None:
            user_balance = cls._balance_projection_in_transaction(
                conn,
                int(user_id),
                through_entry_id=int(pool_entry["user_ledger_entry_id"]),
            )
        return {
            "entry": dict(pool_entry),
            "user_entry": _dict(user_entry),
            "pool": {
                "pool_id": pool_entry["pool_id"],
                "name": pool_entry["pool_name_snapshot"],
                "status": pool_entry["pool_status_snapshot"],
                "balance_micro": cls._pool_balance_in_transaction(
                    conn,
                    pool_entry["pool_id"],
                    through_entry_id=int(pool_entry["id"]),
                ),
            },
            "user_balance": user_balance,
        }

    def _grant_mutation(
        self,
        *,
        operation_type: str,
        pool_id: str,
        amount_micro: int,
        operation_id: str,
        idempotency_key: str,
        request_digest: str,
        actor_user_id: int,
        source: str,
        reason: str,
        user_id: int | None = None,
    ) -> dict[str, Any]:
        _required_text(pool_id, "pool_id", max_length=120)
        _positive_integer(amount_micro, "amount_micro")
        _required_text(operation_id, "operation_id")
        _required_text(idempotency_key, "idempotency_key")
        _required_text(request_digest, "request_digest")
        _positive_integer(actor_user_id, "actor_user_id")
        _required_text(source, "source", max_length=120)
        _required_text(reason, "reason", max_length=500)
        if operation_type not in {"fund", "reduce", "assign", "reclaim"}:
            raise ValueError("unsupported Grant operation")
        if operation_type in {"assign", "reclaim"}:
            _positive_integer(user_id, "user_id")
        elif user_id is not None:
            raise ValueError("user_id is only valid for assign and reclaim")
        expected_pool_amount = (
            amount_micro
            if operation_type in {"fund", "reclaim"}
            else -amount_micro
        )

        with self._get_connection() as conn:
            self._begin(conn)
            pool = self._ensure_pool_in_transaction(conn, pool_id)
            existing = conn.execute(
                """
                SELECT * FROM credit_grant_pool_ledger_entries
                WHERE idempotency_key = ? OR operation_id = ?
                """,
                (idempotency_key, operation_id),
            ).fetchone()
            if existing is not None:
                if (
                    existing["idempotency_key"] != idempotency_key
                    or existing["operation_id"] != operation_id
                    or existing["request_digest"] != request_digest
                    or existing["pool_id"] != pool_id
                    or existing["entry_type"] != operation_type
                    or int(existing["amount_micro"]) != expected_pool_amount
                    or existing["actor_user_id"] != actor_user_id
                    or existing["source"] != source
                    or existing["reason"] != reason
                    or existing["user_id"] != user_id
                ):
                    raise IdempotencyConflictError(
                        "Grant idempotency key conflicts with an existing operation"
                    )
                result = self._grant_mutation_result_in_transaction(conn, existing)
                if operation_type == "assign":
                    recovery = self._recovery_result_for_source_operation_in_transaction(
                        conn, int(user_id), source_operation_key=operation_id
                    )
                    if recovery is not None:
                        result["recovery"] = recovery
                return result

            pool_balance = self._pool_balance_in_transaction(conn, pool_id)
            if operation_type in {"reduce", "assign"} and pool_balance < amount_micro:
                raise GrantPoolInsufficientError(
                    "Grant Pool does not have enough available Credits"
                )

            user_entry_id = None
            created_at = _utcnow_iso()
            if operation_type == "assign":
                self._ensure_account_in_transaction(conn, int(user_id))
                account = conn.execute(
                    "SELECT status, restriction_reason FROM credit_accounts WHERE user_id = ?",
                    (user_id,),
                ).fetchone()
                if (
                    account["status"] == "restricted"
                    and account["restriction_reason"] != "llm_overage"
                ):
                    raise CreditAccountRestrictedStoreError(
                        "refund-review credit account requires administrator review"
                    )
                user_entry_id = self._insert_user_grant_entry_in_transaction(
                    conn,
                    pool_id=pool_id,
                    user_id=int(user_id),
                    entry_type="admin_grant_assign",
                    amount_micro=amount_micro,
                    operation_id=operation_id,
                    idempotency_key=idempotency_key,
                    request_digest=request_digest,
                    actor_user_id=actor_user_id,
                    source=source,
                    reason=reason,
                    created_at=created_at,
                )
            elif operation_type == "reclaim":
                projection = self._balance_projection_in_transaction(
                    conn, int(user_id)
                )
                if projection["grant_available_micro"] < amount_micro:
                    raise GrantReclaimExceedsAvailableError(
                        "reclaim exceeds available Grant Credits"
                    )
                user_entry_id = self._insert_user_grant_entry_in_transaction(
                    conn,
                    pool_id=pool_id,
                    user_id=int(user_id),
                    entry_type="admin_grant_reclaim",
                    amount_micro=-amount_micro,
                    operation_id=operation_id,
                    idempotency_key=idempotency_key,
                    request_digest=request_digest,
                    actor_user_id=actor_user_id,
                    source=source,
                    reason=reason,
                    created_at=created_at,
                )

            pool_entry_id = self._insert_grant_pool_entry_in_transaction(
                conn,
                pool_id=pool_id,
                pool_name_snapshot=pool["name"],
                pool_status_snapshot=pool["status"],
                entry_type=operation_type,
                amount_micro=expected_pool_amount,
                operation_id=operation_id,
                idempotency_key=idempotency_key,
                request_digest=request_digest,
                actor_user_id=actor_user_id,
                source=source,
                reason=reason,
                user_id=user_id,
                user_ledger_entry_id=user_entry_id,
                created_at=created_at,
            )
            pool_entry = conn.execute(
                "SELECT * FROM credit_grant_pool_ledger_entries WHERE id = ?",
                (pool_entry_id,),
            ).fetchone()
            if operation_type == "assign":
                recovery = self._recover_llm_overage_in_transaction(
                    conn, int(user_id), source_operation_key=operation_id
                )
            else:
                recovery = {}
            result = self._grant_mutation_result_in_transaction(conn, pool_entry)
            if recovery.get("recovered_micro", 0) > 0:
                result["recovery"] = recovery
            return result

    def fund_grant_pool(self, **kwargs: Any) -> dict[str, Any]:
        return self._grant_mutation(operation_type="fund", **kwargs)

    def reduce_grant_pool(self, **kwargs: Any) -> dict[str, Any]:
        return self._grant_mutation(operation_type="reduce", **kwargs)

    def assign_grant(self, *, user_id: int, **kwargs: Any) -> dict[str, Any]:
        return self._grant_mutation(
            operation_type="assign", user_id=user_id, **kwargs
        )

    def reclaim_grant(self, *, user_id: int, **kwargs: Any) -> dict[str, Any]:
        return self._grant_mutation(
            operation_type="reclaim", user_id=user_id, **kwargs
        )

    @staticmethod
    def _validate_utc_boundary(month_start_iso: str) -> str:
        value = _required_text(month_start_iso, "month_start_iso")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("month_start_iso must be an ISO timestamp") from exc
        if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
            raise ValueError("month_start_iso must use UTC")
        return parsed.isoformat()

    def get_grant_pool_summary(
        self, pool_id: str, month_start_iso: str
    ) -> dict[str, Any]:
        _required_text(pool_id, "pool_id", max_length=120)
        boundary = self._validate_utc_boundary(month_start_iso)
        with self._get_connection() as conn:
            # Pin every metric to the same SQLite snapshot under concurrent writes.
            conn.execute("BEGIN")
            pool = conn.execute(
                "SELECT * FROM credit_grant_pools WHERE pool_id = ?", (pool_id,)
            ).fetchone()
            if pool is None:
                raise ValueError("grant pool does not exist")
            pool_available_micro = self._pool_balance_in_transaction(conn, pool_id)
            allocated = conn.execute(
                """
                SELECT
                    COALESCE((
                        SELECT SUM(amount_micro)
                        FROM credit_ledger_entries
                        WHERE bucket = 'grant'
                          AND reference_type = 'grant_pool'
                          AND reference_id = ?
                    ), 0)
                    + CASE WHEN ? = 'default' THEN COALESCE((
                        SELECT SUM(amount_micro)
                        FROM credit_llm_usage_entries
                        WHERE bucket = 'grant'
                    ), 0) ELSE 0 END AS amount_micro
                """,
                (pool_id, pool_id),
            ).fetchone()
            monthly = conn.execute(
                """
                SELECT
                    COALESCE(SUM(
                        CASE WHEN entry_type = 'assign' THEN -amount_micro ELSE 0 END
                    ), 0) AS assigned_micro,
                    COALESCE(SUM(
                        CASE WHEN entry_type = 'reclaim' THEN amount_micro ELSE 0 END
                    ), 0) AS reclaimed_micro
                FROM credit_grant_pool_ledger_entries
                WHERE pool_id = ? AND created_at >= ?
                """,
                (pool_id, boundary),
            ).fetchone()
            return {
                "pool_id": pool["pool_id"],
                "pool_name": pool["name"],
                "pool_status": pool["status"],
                "pool_available_micro": pool_available_micro,
                "allocated_to_users_micro": int(allocated["amount_micro"]),
                "assigned_this_month_micro": int(monthly["assigned_micro"]),
                "reclaimed_this_month_micro": int(monthly["reclaimed_micro"]),
                "month_start_iso": boundary,
            }

    def list_grant_pool_activity(
        self,
        pool_id: str,
        *,
        limit: int = 50,
        cursor: int | None = None,
    ) -> dict[str, Any]:
        _required_text(pool_id, "pool_id", max_length=120)
        page_size = _positive_limit(limit)
        params: list[Any] = [pool_id]
        cursor_sql = ""
        if cursor is not None:
            cursor_sql = "AND id < ?"
            params.append(_positive_integer(cursor, "cursor"))
        params.append(page_size + 1)
        with self._get_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM credit_grant_pool_ledger_entries
                WHERE pool_id = ? {cursor_sql}
                ORDER BY id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        has_more = len(rows) > page_size
        items = [dict(row) for row in rows[:page_size]]
        return {
            "items": items,
            "next_cursor": items[-1]["id"] if has_more and items else None,
        }


def _build_credits_store():
    # Credits belong to the account database. Do not fall back to either the
    # content or run-history database: those can have different retention and
    # ownership boundaries.
    database_url = os.getenv("USERS_DATABASE_URL")
    if database_url:
        from dashboard.backend.domain.credits.repository_postgres import (
            PostgresCreditsStore,
        )

        print(
            f"credits_store backend: postgres ({describe_database_url(database_url)})"
        )
        return PostgresCreditsStore(database_url)
    print("credits_store backend: sqlite (ephemeral on Render)")
    return CreditsStore()


credits_store = _build_credits_store()
