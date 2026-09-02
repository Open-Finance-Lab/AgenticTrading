"""Postgres persistence for Credits, payment operations, and webhook receipts."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

import psycopg

from dashboard.backend.db_url import require_postgres_url
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


# Kept as literal DDL so the SQLite/Postgres parity guard can compare the
# authoritative table and column contracts without requiring a live database.
CREDITS_POSTGRES_DDL = """
CREATE TABLE IF NOT EXISTS credit_accounts (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'restricted')),
    restriction_reason TEXT
        CHECK (restriction_reason IN ('llm_overage', 'refund_reconciliation')
               OR restriction_reason IS NULL),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS credit_payment_orders (
    sequence BIGSERIAL PRIMARY KEY,
    id TEXT NOT NULL UNIQUE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    client_request_id TEXT NOT NULL,
    stripe_mode TEXT NOT NULL DEFAULT 'test'
        CHECK (stripe_mode IN ('test', 'live')),
    currency TEXT NOT NULL DEFAULT 'usd',
    amount_usd_cents BIGINT NOT NULL CHECK (amount_usd_cents > 0),
    credits_micro BIGINT NOT NULL CHECK (credits_micro > 0),
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
    FOREIGN KEY (user_id) REFERENCES credit_accounts(user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_credit_payment_orders_user_sequence
ON credit_payment_orders(user_id, sequence DESC);

CREATE TABLE IF NOT EXISTS credit_refund_requests (
    sequence BIGSERIAL PRIMARY KEY,
    id TEXT NOT NULL UNIQUE,
    payment_order_id TEXT NOT NULL
        REFERENCES credit_payment_orders(id) ON DELETE RESTRICT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    requested_by_user_id INTEGER
        REFERENCES users(id) ON DELETE RESTRICT,
    amount_usd_cents BIGINT NOT NULL CHECK (amount_usd_cents > 0),
    credits_micro BIGINT NOT NULL CHECK (credits_micro > 0),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN (
            'pending', 'submitted', 'succeeded', 'failed', 'cancelled'
        )),
    stripe_refund_id TEXT UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    succeeded_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_credit_refunds_order_status
ON credit_refund_requests(payment_order_id, status);

CREATE TABLE IF NOT EXISTS stripe_webhook_events (
    stripe_event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    livemode BOOLEAN NOT NULL,
    object_id TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    outcome TEXT NOT NULL
        CHECK (outcome IN ('processed', 'ignored', 'rejected')),
    reason TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS credit_ledger_entries (
    id BIGSERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    bucket TEXT NOT NULL CHECK (bucket IN ('grant', 'purchased')),
    entry_type TEXT NOT NULL CHECK (entry_type IN (
        'purchase', 'refund', 'admin_grant_assign', 'admin_grant_reclaim'
    )),
    amount_micro BIGINT NOT NULL CHECK (amount_micro <> 0),
    payment_order_id TEXT
        REFERENCES credit_payment_orders(id) ON DELETE RESTRICT,
    refund_request_id TEXT
        REFERENCES credit_refund_requests(id) ON DELETE RESTRICT,
    stripe_event_id TEXT
        REFERENCES stripe_webhook_events(stripe_event_id) ON DELETE RESTRICT,
    operation_key TEXT NOT NULL UNIQUE
        CHECK (length(trim(operation_key)) > 0),
    operation_id TEXT NOT NULL CHECK (length(trim(operation_id)) > 0),
    idempotency_key TEXT NOT NULL
        CHECK (length(trim(idempotency_key)) > 0),
    request_digest TEXT CHECK (
        request_digest IS NULL OR length(trim(request_digest)) > 0
    ),
    actor_user_id INTEGER REFERENCES users(id) ON DELETE RESTRICT,
    source TEXT NOT NULL CHECK (length(trim(source)) > 0),
    reason TEXT NOT NULL CHECK (length(trim(reason)) > 0),
    reference_type TEXT,
    reference_id TEXT,
    created_at TEXT NOT NULL,
    CONSTRAINT credit_ledger_entries_shape CHECK (
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
);

CREATE INDEX IF NOT EXISTS idx_credit_ledger_user_id
ON credit_ledger_entries(user_id, id DESC);

CREATE INDEX IF NOT EXISTS idx_credit_ledger_payment_order
ON credit_ledger_entries(payment_order_id, id DESC);

CREATE TABLE IF NOT EXISTS credit_grant_pools (
    pool_id TEXT PRIMARY KEY CHECK (length(trim(pool_id)) > 0),
    name TEXT NOT NULL CHECK (length(trim(name)) > 0),
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'disabled')),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS credit_grant_pool_ledger_entries (
    id BIGSERIAL PRIMARY KEY,
    pool_id TEXT NOT NULL
        REFERENCES credit_grant_pools(pool_id) ON DELETE RESTRICT,
    pool_name_snapshot TEXT NOT NULL
        CHECK (length(trim(pool_name_snapshot)) > 0),
    pool_status_snapshot TEXT NOT NULL
        CHECK (pool_status_snapshot IN ('active', 'disabled')),
    entry_type TEXT NOT NULL
        CHECK (entry_type IN ('fund', 'reduce', 'assign', 'reclaim')),
    amount_micro BIGINT NOT NULL CHECK (amount_micro <> 0),
    operation_id TEXT NOT NULL UNIQUE CHECK (length(trim(operation_id)) > 0),
    idempotency_key TEXT NOT NULL UNIQUE
        CHECK (length(trim(idempotency_key)) > 0),
    request_digest TEXT NOT NULL CHECK (length(trim(request_digest)) > 0),
    actor_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    source TEXT NOT NULL CHECK (length(trim(source)) > 0),
    reason TEXT NOT NULL CHECK (length(trim(reason)) > 0),
    user_id INTEGER REFERENCES users(id) ON DELETE RESTRICT,
    user_ledger_entry_id BIGINT UNIQUE
        REFERENCES credit_ledger_entries(id) ON DELETE RESTRICT,
    created_at TEXT NOT NULL,
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

CREATE INDEX IF NOT EXISTS idx_credit_grant_pool_ledger_pool_id
ON credit_grant_pool_ledger_entries(pool_id, id DESC);

CREATE TABLE IF NOT EXISTS credit_promotion_grants (
    id BIGSERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    campaign_key TEXT NOT NULL CHECK (length(trim(campaign_key)) > 0),
    amount_micro BIGINT NOT NULL CHECK (amount_micro > 0),
    operation_id TEXT NOT NULL UNIQUE CHECK (length(trim(operation_id)) > 0),
    idempotency_key TEXT NOT NULL UNIQUE
        CHECK (length(trim(idempotency_key)) > 0),
    request_digest TEXT NOT NULL CHECK (length(trim(request_digest)) > 0),
    source TEXT NOT NULL CHECK (length(trim(source)) > 0),
    reason TEXT NOT NULL CHECK (length(trim(reason)) > 0),
    created_at TEXT NOT NULL,
    UNIQUE (campaign_key, user_id)
);

CREATE INDEX IF NOT EXISTS idx_credit_promotion_grants_user_id
ON credit_promotion_grants(user_id, id DESC);

CREATE INDEX IF NOT EXISTS idx_credit_promotion_grants_campaign_user
ON credit_promotion_grants(campaign_key, user_id);

CREATE TABLE IF NOT EXISTS credit_llm_reservations (
    reservation_id TEXT PRIMARY KEY CHECK (length(trim(reservation_id)) > 0),
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    run_id TEXT NOT NULL CHECK (length(trim(run_id)) > 0),
    call_index INTEGER NOT NULL CHECK (call_index >= 0),
    provider_id TEXT,
    attempt_index INTEGER NOT NULL DEFAULT 0 CHECK (attempt_index >= 0),
    reserved_micro BIGINT NOT NULL CHECK (reserved_micro > 0),
    reserved_grant_micro BIGINT NOT NULL CHECK (reserved_grant_micro >= 0),
    reserved_purchased_micro BIGINT NOT NULL CHECK (reserved_purchased_micro >= 0),
    settled_micro BIGINT NOT NULL DEFAULT 0 CHECK (settled_micro >= 0),
    actual_micro BIGINT NOT NULL DEFAULT 0 CHECK (actual_micro >= 0),
    outstanding_micro BIGINT NOT NULL DEFAULT 0 CHECK (outstanding_micro >= 0),
    outstanding_recovered_micro BIGINT NOT NULL DEFAULT 0
        CHECK (outstanding_recovered_micro >= 0),
    status TEXT NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'settled', 'released')),
    operation_key TEXT NOT NULL UNIQUE CHECK (length(trim(operation_key)) > 0),
    request_digest TEXT NOT NULL CHECK (length(trim(request_digest)) > 0),
    evidence_json TEXT,
    failure_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (reserved_micro = reserved_grant_micro + reserved_purchased_micro),
    CONSTRAINT credit_llm_reservations_logical_attempt_key
        UNIQUE (user_id, run_id, call_index, attempt_index)
);

CREATE INDEX IF NOT EXISTS idx_credit_llm_reservations_user_status
ON credit_llm_reservations(user_id, status, created_at);

CREATE TABLE IF NOT EXISTS credit_llm_usage_entries (
    id BIGSERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    reservation_id TEXT NOT NULL
        REFERENCES credit_llm_reservations(reservation_id) ON DELETE RESTRICT,
    run_id TEXT NOT NULL CHECK (length(trim(run_id)) > 0),
    call_index INTEGER NOT NULL CHECK (call_index >= 0),
    bucket TEXT NOT NULL CHECK (bucket IN ('grant', 'purchased')),
    amount_micro BIGINT NOT NULL CHECK (amount_micro < 0),
    operation_key TEXT NOT NULL UNIQUE CHECK (length(trim(operation_key)) > 0),
    evidence_json TEXT NOT NULL CHECK (length(trim(evidence_json)) > 0),
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_credit_llm_usage_user_id
ON credit_llm_usage_entries(user_id, id DESC);
"""


CREDITS_POSTGRES_GRANT_MIGRATION_DDL = """
ALTER TABLE credit_ledger_entries
ADD COLUMN IF NOT EXISTS bucket TEXT;
ALTER TABLE credit_ledger_entries
ADD COLUMN IF NOT EXISTS operation_id TEXT;
ALTER TABLE credit_ledger_entries
ADD COLUMN IF NOT EXISTS idempotency_key TEXT;
ALTER TABLE credit_ledger_entries
ADD COLUMN IF NOT EXISTS request_digest TEXT;
ALTER TABLE credit_ledger_entries
ADD COLUMN IF NOT EXISTS actor_user_id INTEGER;
ALTER TABLE credit_ledger_entries
ADD COLUMN IF NOT EXISTS source TEXT;
ALTER TABLE credit_ledger_entries
ADD COLUMN IF NOT EXISTS reason TEXT;
ALTER TABLE credit_ledger_entries
ADD COLUMN IF NOT EXISTS reference_type TEXT;
ALTER TABLE credit_ledger_entries
ADD COLUMN IF NOT EXISTS reference_id TEXT;

ALTER TABLE credit_llm_reservations
ADD COLUMN IF NOT EXISTS actual_micro BIGINT NOT NULL DEFAULT 0;
ALTER TABLE credit_llm_reservations
ADD COLUMN IF NOT EXISTS outstanding_micro BIGINT NOT NULL DEFAULT 0;
ALTER TABLE credit_llm_reservations
ADD COLUMN IF NOT EXISTS outstanding_recovered_micro BIGINT NOT NULL DEFAULT 0;
ALTER TABLE credit_llm_reservations
ADD COLUMN IF NOT EXISTS provider_id TEXT;
ALTER TABLE credit_llm_reservations
ADD COLUMN IF NOT EXISTS attempt_index INTEGER NOT NULL DEFAULT 0;
ALTER TABLE credit_llm_reservations
DROP CONSTRAINT IF EXISTS credit_llm_reservations_attempt_index_check;
ALTER TABLE credit_llm_reservations
ADD CONSTRAINT credit_llm_reservations_attempt_index_check
CHECK (attempt_index >= 0);
CREATE INDEX IF NOT EXISTS idx_credit_llm_reservations_run_status
ON credit_llm_reservations(run_id, status, call_index, attempt_index);
ALTER TABLE credit_accounts
ADD COLUMN IF NOT EXISTS restriction_reason TEXT;
UPDATE credit_llm_reservations
SET actual_micro = settled_micro
WHERE status = 'settled' AND actual_micro = 0;
DO $$
DECLARE
    legacy_constraint TEXT;
BEGIN
    FOR legacy_constraint IN
        SELECT conname
        FROM pg_constraint
        WHERE conrelid = 'credit_llm_reservations'::regclass
          AND contype = 'c'
          AND pg_get_constraintdef(oid) ~* 'settled_micro\\s*<=\\s*reserved_micro'
    LOOP
        EXECUTE format(
            'ALTER TABLE credit_llm_reservations DROP CONSTRAINT %I',
            legacy_constraint
        );
    END LOOP;
END
$$;
ALTER TABLE credit_llm_reservations
DROP CONSTRAINT IF EXISTS credit_llm_reservations_settled_micro_check;
ALTER TABLE credit_llm_reservations
DROP CONSTRAINT IF EXISTS credit_llm_reservations_settled_micro_nonnegative_check;
ALTER TABLE credit_llm_reservations
ADD CONSTRAINT credit_llm_reservations_settled_micro_nonnegative_check
CHECK (settled_micro >= 0);
DO $$
DECLARE
    legacy_constraint TEXT;
BEGIN
    FOR legacy_constraint IN
        SELECT con.conname
        FROM pg_constraint AS con
        WHERE con.conrelid = 'credit_llm_reservations'::regclass
          AND con.contype = 'u'
          AND (
              SELECT array_agg(att.attname ORDER BY key.ord)
              FROM unnest(con.conkey) WITH ORDINALITY AS key(attnum, ord)
              JOIN pg_attribute AS att
                ON att.attrelid = con.conrelid
               AND att.attnum = key.attnum
          ) = ARRAY['user_id', 'run_id', 'call_index']::name[]
    LOOP
        EXECUTE format(
            'ALTER TABLE credit_llm_reservations DROP CONSTRAINT %I',
            legacy_constraint
        );
    END LOOP;
END
$$;
ALTER TABLE credit_llm_reservations
DROP CONSTRAINT IF EXISTS credit_llm_reservations_logical_attempt_key;
ALTER TABLE credit_llm_reservations
ADD CONSTRAINT credit_llm_reservations_logical_attempt_key
UNIQUE (user_id, run_id, call_index, attempt_index);

ALTER TABLE credit_grant_pool_ledger_entries
ADD COLUMN IF NOT EXISTS pool_name_snapshot TEXT;
ALTER TABLE credit_grant_pool_ledger_entries
ADD COLUMN IF NOT EXISTS pool_status_snapshot TEXT;

UPDATE credit_grant_pool_ledger_entries AS ledger
SET pool_name_snapshot = COALESCE(ledger.pool_name_snapshot, pool.name),
    pool_status_snapshot = COALESCE(ledger.pool_status_snapshot, pool.status)
FROM credit_grant_pools AS pool
WHERE pool.pool_id = ledger.pool_id
  AND (
      ledger.pool_name_snapshot IS NULL
      OR ledger.pool_status_snapshot IS NULL
  );

ALTER TABLE credit_grant_pool_ledger_entries
ALTER COLUMN pool_name_snapshot SET NOT NULL;
ALTER TABLE credit_grant_pool_ledger_entries
ALTER COLUMN pool_status_snapshot SET NOT NULL;
ALTER TABLE credit_grant_pool_ledger_entries
DROP CONSTRAINT IF EXISTS
credit_grant_pool_ledger_entries_pool_name_snapshot_check;
ALTER TABLE credit_grant_pool_ledger_entries
DROP CONSTRAINT IF EXISTS
credit_grant_pool_ledger_entries_pool_status_snapshot_check;
ALTER TABLE credit_grant_pool_ledger_entries
ADD CONSTRAINT credit_grant_pool_ledger_entries_pool_name_snapshot_check
CHECK (length(trim(pool_name_snapshot)) > 0);
ALTER TABLE credit_grant_pool_ledger_entries
ADD CONSTRAINT credit_grant_pool_ledger_entries_pool_status_snapshot_check
CHECK (pool_status_snapshot IN ('active', 'disabled'));

UPDATE credit_ledger_entries
SET bucket = COALESCE(bucket, 'purchased'),
    operation_id = COALESCE(operation_id, operation_key),
    idempotency_key = COALESCE(idempotency_key, operation_key),
    source = COALESCE(source, 'stripe'),
    reason = COALESCE(
        reason,
        CASE entry_type
            WHEN 'purchase' THEN 'Historical Stripe purchase.'
            ELSE 'Historical Stripe refund.'
        END
    );

ALTER TABLE credit_ledger_entries ALTER COLUMN bucket SET NOT NULL;
ALTER TABLE credit_ledger_entries ALTER COLUMN operation_id SET NOT NULL;
ALTER TABLE credit_ledger_entries ALTER COLUMN idempotency_key SET NOT NULL;
ALTER TABLE credit_ledger_entries ALTER COLUMN source SET NOT NULL;
ALTER TABLE credit_ledger_entries ALTER COLUMN reason SET NOT NULL;
ALTER TABLE credit_ledger_entries ALTER COLUMN payment_order_id DROP NOT NULL;
ALTER TABLE credit_ledger_entries ALTER COLUMN stripe_event_id DROP NOT NULL;

ALTER TABLE credit_ledger_entries
DROP CONSTRAINT IF EXISTS credit_ledger_entries_entry_type_check;
ALTER TABLE credit_ledger_entries
DROP CONSTRAINT IF EXISTS credit_ledger_entries_shape;
ALTER TABLE credit_ledger_entries
DROP CONSTRAINT IF EXISTS credit_ledger_entries_bucket_check;
ALTER TABLE credit_ledger_entries
DROP CONSTRAINT IF EXISTS credit_ledger_entries_operation_key_check;
ALTER TABLE credit_ledger_entries
DROP CONSTRAINT IF EXISTS credit_ledger_entries_operation_id_check;
ALTER TABLE credit_ledger_entries
DROP CONSTRAINT IF EXISTS credit_ledger_entries_idempotency_key_check;
ALTER TABLE credit_ledger_entries
DROP CONSTRAINT IF EXISTS credit_ledger_entries_request_digest_check;
ALTER TABLE credit_ledger_entries
DROP CONSTRAINT IF EXISTS credit_ledger_entries_source_check;
ALTER TABLE credit_ledger_entries
DROP CONSTRAINT IF EXISTS credit_ledger_entries_reason_check;

ALTER TABLE credit_ledger_entries
ADD CONSTRAINT credit_ledger_entries_bucket_check
CHECK (bucket IN ('grant', 'purchased'));
ALTER TABLE credit_ledger_entries
ADD CONSTRAINT credit_ledger_entries_operation_key_check
CHECK (length(trim(operation_key)) > 0);
ALTER TABLE credit_ledger_entries
ADD CONSTRAINT credit_ledger_entries_operation_id_check
CHECK (length(trim(operation_id)) > 0);
ALTER TABLE credit_ledger_entries
ADD CONSTRAINT credit_ledger_entries_idempotency_key_check
CHECK (length(trim(idempotency_key)) > 0);
ALTER TABLE credit_ledger_entries
ADD CONSTRAINT credit_ledger_entries_request_digest_check
CHECK (request_digest IS NULL OR length(trim(request_digest)) > 0);
ALTER TABLE credit_ledger_entries
ADD CONSTRAINT credit_ledger_entries_source_check
CHECK (length(trim(source)) > 0);
ALTER TABLE credit_ledger_entries
ADD CONSTRAINT credit_ledger_entries_reason_check
CHECK (length(trim(reason)) > 0);

ALTER TABLE credit_ledger_entries
ADD CONSTRAINT credit_ledger_entries_shape CHECK (
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
);

ALTER TABLE credit_ledger_entries
DROP CONSTRAINT IF EXISTS credit_ledger_entries_actor_user_id_fkey;
ALTER TABLE credit_ledger_entries
ADD CONSTRAINT credit_ledger_entries_actor_user_id_fkey
FOREIGN KEY (actor_user_id) REFERENCES users(id) ON DELETE RESTRICT;

CREATE UNIQUE INDEX IF NOT EXISTS uq_credit_ledger_operation_id
ON credit_ledger_entries(operation_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_credit_ledger_idempotency_key
ON credit_ledger_entries(idempotency_key);
CREATE INDEX IF NOT EXISTS idx_credit_grant_user_reference
ON credit_ledger_entries(reference_type, reference_id, user_id, id DESC);
"""


class PostgresCreditsStore:
    """Account-scoped append-only Credits ledger backed by Postgres."""

    def __init__(self, database_url: str):
        self.database_url = require_postgres_url(database_url)
        self._init_schema()

    def _get_connection(self):
        from dashboard.backend.db_pool import get_pool

        return get_pool(self.database_url).connection()

    def _init_schema(self) -> None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(CREDITS_POSTGRES_DDL)
                cur.execute(CREDITS_POSTGRES_GRANT_MIGRATION_DDL)
                cur.execute(
                    """
                    INSERT INTO credit_grant_pools (
                        pool_id, name, status, created_at
                    ) VALUES (
                        'default', 'Platform Research Grants', 'active', %s
                    )
                    ON CONFLICT (pool_id) DO NOTHING
                    """,
                    (_utcnow_iso(),),
                )

    @staticmethod
    def _lock_event(cur, event_id: str) -> None:
        # Serializes duplicate deliveries before the unique event row exists.
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (event_id,))

    @staticmethod
    def _ensure_account_in_transaction(cur, user_id: int) -> None:
        cur.execute(
            """
            INSERT INTO credit_accounts (user_id, status, created_at)
            VALUES (%s, 'active', %s)
            ON CONFLICT(user_id) DO NOTHING
            """,
            (user_id, _utcnow_iso()),
        )

    def ensure_account(self, user_id: int) -> dict[str, Any]:
        _positive_integer(user_id, "user_id")
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                self._ensure_account_in_transaction(cur, user_id)
                cur.execute(
                    "SELECT * FROM credit_accounts WHERE user_id = %s", (user_id,)
                )
                return dict(cur.fetchone())

    def get_account_billing_state(self, user_id: int) -> dict[str, Any]:
        _positive_integer(user_id, "user_id")
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                self._ensure_account_in_transaction(cur, user_id)
                cur.execute(
                    "SELECT status, restriction_reason FROM credit_accounts WHERE user_id = %s",
                    (user_id,),
                )
                account = cur.fetchone()
                cur.execute(
                    """
                    SELECT COALESCE(SUM(
                        GREATEST(outstanding_micro - outstanding_recovered_micro, 0)
                    ), 0) AS outstanding_micro
                    FROM credit_llm_reservations
                    WHERE user_id = %s AND status = 'settled'
                    """,
                    (user_id,),
                )
                outstanding = cur.fetchone()
                reason = account["restriction_reason"]
                if account["status"] == "restricted" and reason not in {
                    "llm_overage",
                    "refund_reconciliation",
                }:
                    reason = "refund_reconciliation"
                return {
                    "account_status": account["status"],
                    "restriction_reason": reason,
                    "outstanding_credits_micro": int(
                        outstanding["outstanding_micro"]
                    ),
                }

    def get_balance_projection(self, user_id: int) -> dict[str, int]:
        _positive_integer(user_id, "user_id")
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                return self._balance_projection_in_transaction(cur, user_id)

    def get_balance_projections(
        self, user_ids: list[int] | tuple[int, ...]
    ) -> dict[int, dict[str, int]]:
        if not isinstance(user_ids, (list, tuple)):
            raise ValueError("user_ids must be a list or tuple")
        validated = [_positive_integer(user_id, "user_id") for user_id in user_ids]
        if not validated:
            return {}

        unique_ids = list(dict.fromkeys(validated))
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        user_id,
                        COALESCE(SUM(
                            CASE WHEN bucket = 'grant' THEN amount_micro ELSE 0 END
                        ), 0) AS grant_committed_micro,
                        COALESCE(SUM(
                            CASE WHEN bucket = 'purchased' THEN amount_micro ELSE 0 END
                        ), 0) AS purchased_committed_micro
                    FROM credit_ledger_entries
                    WHERE user_id = ANY(%s)
                    GROUP BY user_id
                    """,
                    (unique_ids,),
                )
                rows = cur.fetchall()
                cur.execute(
                    """
                    SELECT user_id, COALESCE(SUM(amount_micro), 0) AS grant_micro
                    FROM credit_promotion_grants
                    WHERE user_id = ANY(%s)
                    GROUP BY user_id
                    """,
                    (unique_ids,),
                )
                promotion_rows = cur.fetchall()
                cur.execute(
                    """
                    SELECT
                        user_id,
                        COALESCE(SUM(CASE WHEN bucket = 'grant' THEN amount_micro ELSE 0 END), 0)
                            AS grant_usage_micro,
                        COALESCE(SUM(CASE WHEN bucket = 'purchased' THEN amount_micro ELSE 0 END), 0)
                            AS purchased_usage_micro
                    FROM credit_llm_usage_entries
                    WHERE user_id = ANY(%s)
                    GROUP BY user_id
                    """,
                    (unique_ids,),
                )
                usage_rows = cur.fetchall()
                cur.execute(
                    """
                    SELECT
                        user_id,
                        COALESCE(SUM(reserved_grant_micro), 0) AS reserved_grant_micro,
                        COALESCE(SUM(reserved_purchased_micro), 0) AS reserved_purchased_micro
                    FROM credit_llm_reservations
                    WHERE user_id = ANY(%s) AND status = 'open'
                    GROUP BY user_id
                    """,
                    (unique_ids,),
                )
                reservation_rows = cur.fetchall()

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
            reserved_grant, reserved_purchased = reserved_amounts.get(user_id, (0, 0))
            grant_micro += grant_usage
            purchased_micro += purchased_usage
            projections[user_id] = {
                "grant_committed_micro": grant_micro,
                "purchased_committed_micro": purchased_micro,
                "grant_available_micro": grant_micro - reserved_grant,
                "purchased_available_micro": purchased_micro - reserved_purchased,
                "total_available_micro": grant_micro + purchased_micro - reserved_grant - reserved_purchased,
            }
        return projections

    def get_balance_micro(self, user_id: int) -> int:
        return self.get_balance_projection(user_id)["total_available_micro"]

    def list_user_ids(self) -> list[int]:
        """Return account IDs for an idempotent promotion backfill."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM users ORDER BY id")
                rows = cur.fetchall()
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
            with conn.cursor() as cur:
                self._ensure_account_in_transaction(cur, user_id)
                for lock_key in sorted(
                    {
                        f"promotion-idempotency:{idempotency_key}",
                        f"promotion-operation:{operation_id}",
                    }
                ):
                    cur.execute(
                        "SELECT pg_advisory_xact_lock(hashtext(%s))", (lock_key,)
                    )
                cur.execute(
                    """
                    SELECT * FROM credit_promotion_grants
                    WHERE idempotency_key = %s OR operation_id = %s
                    FOR UPDATE
                    """,
                    (idempotency_key, operation_id),
                )
                existing = cur.fetchone()
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
                cur.execute(
                    """
                    INSERT INTO credit_promotion_grants (
                        user_id, campaign_key, amount_micro, operation_id,
                        idempotency_key, request_digest, source, reason, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING *
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
                return {"created": True, "grant": dict(cur.fetchone())}

    @staticmethod
    def _llm_reservation_result(cur, reservation) -> dict[str, Any]:
        cur.execute(
            """
            SELECT id, bucket, amount_micro
            FROM credit_llm_usage_entries
            WHERE reservation_id = %s AND operation_key NOT LIKE '%%:recovery:%%'
            ORDER BY id
            """,
            (reservation["reservation_id"],),
        )
        entries = cur.fetchall()
        row = dict(reservation)
        row.update(
            released_micro=max(
                int(row["reserved_micro"]) - int(row["settled_micro"]), 0
            ),
            outstanding_recovered_micro=int(
                row.get("outstanding_recovered_micro") or 0
            ),
            grant_debited_micro=sum(-int(e["amount_micro"]) for e in entries if e["bucket"] == "grant"),
            purchased_debited_micro=sum(-int(e["amount_micro"]) for e in entries if e["bucket"] == "purchased"),
            ledger_entry_ids=tuple(int(e["id"]) for e in entries),
        )
        return row

    def reserve_llm_credits(self, *, reservation_id: str, user_id: int, run_id: str,
                            call_index: int, attempt_index: int, provider_id: str,
                            reserved_micro: int,
                            operation_key: str, request_digest: str) -> dict[str, Any]:
        reservation_id = _required_text(reservation_id, "reservation_id", max_length=160)
        run_id = _required_text(run_id, "run_id", max_length=128)
        operation_key = _required_text(operation_key, "operation_key", max_length=200)
        request_digest = _required_text(request_digest, "request_digest", max_length=128)
        _positive_integer(user_id, "user_id")
        _positive_integer(reserved_micro, "reserved_micro")
        provider_id = validate_provider_id(provider_id)
        if not isinstance(call_index, int) or isinstance(call_index, bool) or call_index < 0:
            raise ValueError("call_index must be a non-negative integer")
        if not isinstance(attempt_index, int) or isinstance(attempt_index, bool) or attempt_index < 0:
            raise ValueError("attempt_index must be a non-negative integer")
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                self._ensure_account_in_transaction(cur, user_id)
                cur.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(%s))",
                    (operation_key,),
                )
                cur.execute(
                    "SELECT * FROM credit_llm_reservations WHERE reservation_id = %s OR operation_key = %s OR (user_id = %s AND run_id = %s AND call_index = %s AND attempt_index = %s) FOR UPDATE",
                    (reservation_id, operation_key, user_id, run_id, call_index, attempt_index),
                )
                existing = cur.fetchone()
                if existing:
                    if (existing["reservation_id"] != reservation_id
                            or int(existing["user_id"]) != user_id
                            or existing["run_id"] != run_id
                            or int(existing["call_index"]) != call_index
                            or int(existing["attempt_index"]) != attempt_index
                            or existing["provider_id"] != provider_id
                            or int(existing["reserved_micro"]) != reserved_micro
                            or existing["operation_key"] != operation_key or existing["request_digest"] != request_digest):
                        raise LLMReservationConflictError("reservation key already represents different input")
                    return dict(existing)
                cur.execute("SELECT status, restriction_reason FROM credit_accounts WHERE user_id = %s FOR UPDATE", (user_id,))
                account = cur.fetchone()
                if account["status"] == "restricted":
                    detail = (
                        "refund-review credit account requires administrator review"
                        if account.get("restriction_reason") != "llm_overage"
                        else "credit account has an unpaid model-usage overage"
                    )
                    raise CreditAccountRestrictedStoreError(detail)
                projection = self._balance_projection_in_transaction(cur, user_id)
                if projection["total_available_micro"] < reserved_micro:
                    raise InsufficientCreditsError("insufficient available Credits")
                reserved_grant = min(max(projection["grant_available_micro"], 0), reserved_micro)
                reserved_purchased = reserved_micro - reserved_grant
                now = _utcnow_iso()
                cur.execute(
                    """
                    INSERT INTO credit_llm_reservations (
                        reservation_id, user_id, run_id, call_index, provider_id,
                        attempt_index,
                        reserved_micro, reserved_grant_micro, reserved_purchased_micro,
                        operation_key, request_digest, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING *
                    """,
                    (reservation_id, user_id, run_id, call_index, provider_id,
                     attempt_index, reserved_micro, reserved_grant, reserved_purchased,
                     operation_key, request_digest, now, now),
                )
                return dict(cur.fetchone())

    def settle_llm_credits(self, reservation_id: str, *, actual_micro: int,
                           evidence: dict[str, Any]) -> dict[str, Any]:
        reservation_id = _required_text(reservation_id, "reservation_id", max_length=160)
        if not isinstance(actual_micro, int) or isinstance(actual_micro, bool) or actual_micro < 0:
            raise ValueError("actual_micro must be a non-negative integer")
        evidence_json = _evidence_json(evidence)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT user_id FROM credit_llm_reservations "
                    "WHERE reservation_id = %s",
                    (reservation_id,),
                )
                owner = cur.fetchone()
                if not owner:
                    raise LLMReservationConflictError("reservation was not found")
                self._ensure_account_in_transaction(cur, int(owner["user_id"]))
                cur.execute(
                    "SELECT status FROM credit_accounts WHERE user_id = %s FOR UPDATE",
                    (owner["user_id"],),
                )
                cur.execute(
                    "SELECT * FROM credit_llm_reservations "
                    "WHERE reservation_id = %s FOR UPDATE",
                    (reservation_id,),
                )
                reservation = cur.fetchone()
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
                        raise LLMReservationConflictError("settlement replay has different input")
                    return self._llm_reservation_result(cur, reservation)
                if reservation["status"] != "open":
                    raise LLMReservationConflictError("released reservation cannot be settled")
                reserved_micro = int(reservation["reserved_micro"])
                excess_micro = max(actual_micro - reserved_micro, 0)
                supplementary_grant = 0
                supplementary_purchased = 0
                if excess_micro > 0:
                    projection = self._balance_projection_in_transaction(
                        cur, int(reservation["user_id"])
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
                hold_debit = min(actual_micro, reserved_micro)
                grant_debit = min(hold_debit, int(reservation["reserved_grant_micro"]))
                purchased_debit = hold_debit - grant_debit
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
                    cur.execute(
                        """
                        INSERT INTO credit_llm_usage_entries (
                            user_id, reservation_id, run_id, call_index, bucket,
                            amount_micro, operation_key, evidence_json, created_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (reservation["user_id"], reservation_id, reservation["run_id"], reservation["call_index"], bucket,
                         -amount, f"{reservation['operation_key']}:{suffix}", settled_evidence_json, now),
                    )
                cur.execute(
                    "UPDATE credit_llm_reservations SET settled_micro = %s, actual_micro = %s, outstanding_micro = %s, status = 'settled', evidence_json = %s, failure_reason = NULL, updated_at = %s WHERE reservation_id = %s RETURNING *",
                    (debit_micro, actual_micro, outstanding_micro, settled_evidence_json, now, reservation_id),
                )
                settled = cur.fetchone()
                if outstanding_micro > 0:
                    cur.execute(
                        """
                        UPDATE credit_accounts
                        SET status = 'restricted', restriction_reason = 'llm_overage'
                        WHERE user_id = %s AND status <> 'restricted'
                        """,
                        (reservation["user_id"],),
                    )
                return self._llm_reservation_result(cur, settled)

    def recover_llm_overage(
        self, user_id: int, *, source_operation_key: str
    ) -> dict[str, Any]:
        _positive_integer(user_id, "user_id")
        source_operation_key = _required_text(
            source_operation_key, "source_operation_key", max_length=200
        )
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                return self._recover_llm_overage_in_transaction(
                    cur, user_id, source_operation_key=source_operation_key
                )

    @staticmethod
    def _recovery_result_for_source_operation_in_transaction(
        cur, user_id: int, *, source_operation_key: str
    ) -> dict[str, Any] | None:
        cur.execute(
            "SELECT amount_micro, evidence_json FROM credit_llm_usage_entries "
            "WHERE user_id = %s",
            (user_id,),
        )
        recovered = 0
        for row in cur.fetchall():
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

        cur.execute(
            "SELECT status, restriction_reason FROM credit_accounts WHERE user_id = %s",
            (user_id,),
        )
        account = cur.fetchone()
        cur.execute(
            """
            SELECT COALESCE(SUM(
                GREATEST(outstanding_micro - outstanding_recovered_micro, 0)
            ), 0) AS outstanding_micro
            FROM credit_llm_reservations
            WHERE user_id = %s AND status = 'settled'
            """,
            (user_id,),
        )
        remaining = cur.fetchone()
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
        cur, user_id: int, *, source_operation_key: str
    ) -> dict[str, Any]:
        PostgresCreditsStore._ensure_account_in_transaction(cur, user_id)
        cur.execute(
            "SELECT status, restriction_reason FROM credit_accounts WHERE user_id = %s FOR UPDATE",
            (user_id,),
        )
        account = cur.fetchone()
        previous = PostgresCreditsStore._recovery_result_for_source_operation_in_transaction(
            cur, user_id, source_operation_key=source_operation_key
        )
        if previous is not None:
            return previous
        reason = account["restriction_reason"]
        if account["status"] != "restricted" or reason != "llm_overage":
            return {
                "recovered_micro": 0,
                "outstanding_micro": 0,
                "account_status": account["status"],
                "restriction_reason": reason,
            }
        projection = PostgresCreditsStore._balance_projection_in_transaction(
            cur, user_id
        )
        grant_available = max(int(projection["grant_available_micro"]), 0)
        purchased_available = max(int(projection["purchased_available_micro"]), 0)
        remaining_funds = grant_available + purchased_available
        recovered_total = 0
        cur.execute(
            """
            SELECT * FROM credit_llm_reservations
            WHERE user_id = %s AND status = 'settled'
              AND outstanding_micro > outstanding_recovered_micro
            ORDER BY created_at, reservation_id
            FOR UPDATE
            """,
            (user_id,),
        )
        reservations = cur.fetchall()
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
                cur.execute(
                    "SELECT amount_micro FROM credit_llm_usage_entries WHERE operation_key = %s",
                    (operation_key,),
                )
                if cur.fetchone() is None:
                    cur.execute(
                        """
                        INSERT INTO credit_llm_usage_entries (
                            user_id, reservation_id, run_id, call_index,
                            bucket, amount_micro, operation_key,
                            evidence_json, created_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            cur.execute(
                """
                UPDATE credit_llm_reservations
                SET outstanding_recovered_micro =
                    outstanding_recovered_micro + %s, updated_at = %s
                WHERE reservation_id = %s
                """,
                (amount, _utcnow_iso(), reservation["reservation_id"]),
            )
            grant_available -= grant_debit
            purchased_available -= purchased_debit
            remaining_funds -= amount
            recovered_total += amount

        cur.execute(
            """
            SELECT COALESCE(SUM(
                GREATEST(outstanding_micro - outstanding_recovered_micro, 0)
            ), 0) AS outstanding_micro
            FROM credit_llm_reservations
            WHERE user_id = %s AND status = 'settled'
            """,
            (user_id,),
        )
        outstanding = int(cur.fetchone()["outstanding_micro"])
        if outstanding == 0:
            cur.execute(
                """
                UPDATE credit_accounts
                SET status = 'active', restriction_reason = NULL
                WHERE user_id = %s AND status = 'restricted'
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

    def release_llm_credits(self, reservation_id: str, *, reason: str) -> dict[str, Any]:
        reservation_id = _required_text(reservation_id, "reservation_id", max_length=160)
        reason = _required_text(reason, "reason", max_length=120)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM credit_llm_reservations WHERE reservation_id = %s FOR UPDATE", (reservation_id,))
                reservation = cur.fetchone()
                if not reservation:
                    raise LLMReservationConflictError("reservation was not found")
                if reservation["status"] == "open":
                    cur.execute("UPDATE credit_llm_reservations SET status = 'released', failure_reason = %s, updated_at = %s WHERE reservation_id = %s RETURNING *", (reason, _utcnow_iso(), reservation_id))
                    reservation = cur.fetchone()
                return self._llm_reservation_result(cur, reservation)

    def release_run_llm_reservations(self, run_id: str, *, reason: str) -> list[dict[str, Any]]:
        run_id = _required_text(run_id, "run_id", max_length=128)
        reason = _required_text(reason, "reason", max_length=120)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE credit_llm_reservations SET status = 'released', failure_reason = %s, updated_at = %s WHERE run_id = %s AND status = 'open'", (reason, _utcnow_iso(), run_id))
                cur.execute("SELECT * FROM credit_llm_reservations WHERE run_id = %s ORDER BY call_index, attempt_index, reservation_id", (run_id,))
                return [self._llm_reservation_result(cur, row) for row in cur.fetchall()]

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

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT * FROM credit_payment_orders
                        WHERE user_id = %s AND client_request_id = %s
                        FOR UPDATE
                        """,
                        (user_id, client_request_id),
                    )
                    existing = cur.fetchone()
                    if existing:
                        return self._matching_order(
                            existing, amount_usd_cents, credits_micro
                        )

                    self._ensure_account_in_transaction(cur, user_id)
                    now = _utcnow_iso()
                    cur.execute(
                        """
                        INSERT INTO credit_payment_orders (
                            id, user_id, client_request_id, stripe_mode, currency,
                            amount_usd_cents, credits_micro, status,
                            created_at, updated_at
                        )
                        VALUES (%s, %s, %s, 'test', 'usd', %s, %s,
                                'pending', %s, %s)
                        RETURNING *
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
                    return dict(cur.fetchone())
        except psycopg.errors.UniqueViolation as exc:
            # Two identical first requests can race before either row exists.
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT * FROM credit_payment_orders
                        WHERE user_id = %s AND client_request_id = %s
                        """,
                        (user_id, client_request_id),
                    )
                    existing = cur.fetchone()
                    if existing:
                        return self._matching_order(
                            existing, amount_usd_cents, credits_micro
                        )
            raise OrderConflictError("order ID already exists") from exc

    @staticmethod
    def _matching_order(
        existing: dict[str, Any], amount_usd_cents: int, credits_micro: int
    ) -> dict[str, Any]:
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

    def attach_checkout_session(
        self, order_id: str, *, checkout_session_id: str
    ) -> dict[str, Any]:
        if not str(checkout_session_id).strip():
            raise ValueError("checkout_session_id is required")
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT * FROM credit_payment_orders WHERE id = %s
                        FOR UPDATE
                        """,
                        (order_id,),
                    )
                    row = cur.fetchone()
                    if not row:
                        raise KeyError("payment order not found")
                    current = row["stripe_checkout_session_id"]
                    if current and current != checkout_session_id:
                        raise OrderConflictError(
                            "payment order already has a different Checkout Session"
                        )
                    if not current:
                        cur.execute(
                            """
                            UPDATE credit_payment_orders
                            SET stripe_checkout_session_id = %s, updated_at = %s
                            WHERE id = %s AND stripe_checkout_session_id IS NULL
                            RETURNING *
                            """,
                            (checkout_session_id, _utcnow_iso(), order_id),
                        )
                        return dict(cur.fetchone())
                    return dict(row)
        except psycopg.errors.UniqueViolation as exc:
            raise OrderConflictError(
                "Checkout Session is already attached to another order"
            ) from exc

    @staticmethod
    def _existing_event(
        cur,
        *,
        event_id: str,
        event_type: str,
        livemode: bool,
        object_id: str,
        payload_sha256: str,
    ) -> dict[str, Any] | None:
        cur.execute(
            "SELECT * FROM stripe_webhook_events WHERE stripe_event_id = %s",
            (event_id,),
        )
        row = cur.fetchone()
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
        cur,
        *,
        event_id: str,
        event_type: str,
        livemode: bool,
        object_id: str,
        payload_sha256: str,
        outcome: str,
        reason: str | None = None,
    ) -> None:
        cur.execute(
            """
            INSERT INTO stripe_webhook_events (
                stripe_event_id, event_type, livemode, object_id,
                payload_sha256, outcome, reason, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                event_id,
                event_type,
                bool(livemode),
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
            with conn.cursor() as cur:
                self._lock_event(cur, event_id)
                existing = self._existing_event(
                    cur,
                    event_id=event_id,
                    event_type=event_type,
                    livemode=livemode,
                    object_id=object_id,
                    payload_sha256=payload_sha256,
                )
                if existing:
                    return {"outcome": "duplicate", "reason": existing["reason"]}
                self._insert_event(
                    cur,
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
            with conn.cursor() as cur:
                self._lock_event(cur, event_id)
                existing = self._existing_event(
                    cur,
                    event_id=event_id,
                    event_type=event_type,
                    livemode=livemode,
                    object_id=object_id,
                    payload_sha256=payload_sha256,
                )
                if existing:
                    return {"outcome": "duplicate", "status": terminal_status}

                cur.execute(
                    "SELECT * FROM credit_payment_orders WHERE id = %s FOR UPDATE",
                    (order_id,),
                )
                order = cur.fetchone()
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
                        cur,
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
                        cur,
                        event_id=event_id,
                        event_type=event_type,
                        livemode=livemode,
                        object_id=object_id,
                        payload_sha256=payload_sha256,
                        outcome="ignored",
                        reason=reason,
                    )
                    return {
                        "outcome": "ignored",
                        "reason": reason,
                        "status": order["status"],
                    }

                now = _utcnow_iso()
                self._insert_event(
                    cur,
                    event_id=event_id,
                    event_type=event_type,
                    livemode=livemode,
                    object_id=object_id,
                    payload_sha256=payload_sha256,
                    outcome="processed",
                )
                cur.execute(
                    "UPDATE credit_payment_orders SET status = %s, updated_at = %s WHERE id = %s",
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
            with conn.cursor() as cur:
                self._lock_event(cur, event_id)
                existing_event = self._existing_event(
                    cur,
                    event_id=event_id,
                    event_type=event_type,
                    livemode=livemode,
                    object_id=object_id,
                    payload_sha256=payload_sha256,
                )
                if existing_event:
                    cur.execute(
                        "SELECT user_id FROM credit_payment_orders WHERE id = %s",
                        (order_id,),
                    )
                    order = cur.fetchone()
                    balance = (
                        self._balance_in_transaction(cur, order["user_id"])
                        if order
                        else 0
                    )
                    return {"outcome": "duplicate", "balance_micro": balance}

                cur.execute(
                    """
                    SELECT * FROM credit_payment_orders WHERE id = %s FOR UPDATE
                    """,
                    (order_id,),
                )
                order = cur.fetchone()
                reason = None
                if not order:
                    reason = "payment order not found"
                elif livemode or order["stripe_mode"] != "test":
                    reason = "Live Mode payment is not accepted"
                elif currency.lower() != order["currency"]:
                    reason = "payment currency does not match the order"
                elif amount_usd_cents != order["amount_usd_cents"]:
                    reason = "payment amount does not match the order"
                # NULL means "not recorded yet", not "mismatch" — the same idiom
                # the PaymentIntent check below already uses.
                # attach_checkout_session runs *after* Stripe has created a
                # payable session, so a crash or restart in that window leaves
                # this column NULL; treating that as a mismatch permanently
                # rejects the payment webhook for an order the customer has
                # already been charged for, and writes an event row that makes
                # the rejection unreplayable. Provenance does not rest on this
                # column: the caller has already matched the signed event's
                # atl_order_id / atl_user_reference / atl_credits_micro metadata
                # against the order, and currency and amount are checked above.
                elif order["stripe_checkout_session_id"] not in (
                    None,
                    checkout_session_id,
                ):
                    reason = "Checkout Session does not match the order"
                elif object_id != checkout_session_id:
                    reason = "event object does not match the Checkout Session"
                elif order["stripe_payment_intent_id"] not in (
                    None,
                    payment_intent_id,
                ):
                    reason = "PaymentIntent does not match the order"

                if reason:
                    self._insert_event(
                        cur,
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
                cur.execute(
                    "SELECT id FROM credit_ledger_entries WHERE operation_key = %s",
                    (operation_key,),
                )
                if cur.fetchone():
                    self._insert_event(
                        cur,
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
                        "balance_micro": self._balance_in_transaction(
                            cur, order["user_id"]
                        ),
                    }

                now = _utcnow_iso()
                self._insert_event(
                    cur,
                    event_id=event_id,
                    event_type=event_type,
                    livemode=livemode,
                    object_id=object_id,
                    payload_sha256=payload_sha256,
                    outcome="processed",
                )
                cur.execute(
                    """
                    INSERT INTO credit_ledger_entries (
                        user_id, bucket, entry_type, amount_micro,
                        payment_order_id, refund_request_id, stripe_event_id,
                        operation_key, operation_id, idempotency_key,
                        source, reason, created_at
                    )
                    VALUES (
                        %s, 'purchased', 'purchase', %s, %s, NULL, %s,
                        %s, %s, %s, 'stripe', 'Stripe checkout purchase.', %s
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
                cur.execute(
                    """
                    UPDATE credit_payment_orders
                    SET status = 'paid', stripe_payment_intent_id = %s,
                        stripe_checkout_session_id =
                            COALESCE(stripe_checkout_session_id, %s),
                        updated_at = %s, paid_at = COALESCE(paid_at, %s)
                    WHERE id = %s
                    """,
                    (payment_intent_id, checkout_session_id, now, now, order_id),
                )
                recovery = self._recover_llm_overage_in_transaction(
                    cur, order["user_id"], source_operation_key=operation_key
                )
                return {
                    "outcome": "processed",
                    "balance_micro": self._balance_in_transaction(
                        cur, order["user_id"]
                    ),
                    **recovery,
                }

    @staticmethod
    def _balance_projection_in_transaction(
        cur,
        user_id: int,
        through_entry_id: int | None = None,
    ) -> dict[str, int]:
        cutoff_sql = ""
        params: list[Any] = [user_id]
        if through_entry_id is not None:
            cutoff_sql = "AND id <= %s"
            params.append(through_entry_id)
        cur.execute(
            f"""
            SELECT
                COALESCE(SUM(
                    CASE WHEN bucket = 'grant' THEN amount_micro ELSE 0 END
                ), 0) AS grant_committed_micro,
                COALESCE(SUM(
                    CASE WHEN bucket = 'purchased' THEN amount_micro ELSE 0 END
                ), 0) AS purchased_committed_micro
            FROM credit_ledger_entries
            WHERE user_id = %s
              {cutoff_sql}
            """,
            params,
        )
        row = cur.fetchone()
        grant_micro = int(row["grant_committed_micro"])
        purchased_micro = int(row["purchased_committed_micro"])
        if through_entry_id is None:
            cur.execute(
                """
                SELECT COALESCE(SUM(amount_micro), 0) AS grant_micro
                FROM credit_promotion_grants
                WHERE user_id = %s
                """,
                (user_id,),
            )
            promotion = cur.fetchone()
            grant_micro += int(promotion["grant_micro"])
            cur.execute(
                """
                SELECT
                    COALESCE(SUM(CASE WHEN bucket = 'grant' THEN amount_micro ELSE 0 END), 0)
                        AS grant_usage_micro,
                    COALESCE(SUM(CASE WHEN bucket = 'purchased' THEN amount_micro ELSE 0 END), 0)
                        AS purchased_usage_micro
                    FROM credit_llm_usage_entries
                    WHERE user_id = %s
                """,
                (user_id,),
            )
            usage = cur.fetchone()
            grant_micro += int(usage["grant_usage_micro"])
            purchased_micro += int(usage["purchased_usage_micro"])
            cur.execute(
                """
                SELECT
                    COALESCE(SUM(reserved_grant_micro), 0) AS reserved_grant_micro,
                    COALESCE(SUM(reserved_purchased_micro), 0) AS reserved_purchased_micro
                FROM credit_llm_reservations
                WHERE user_id = %s AND status = 'open'
                """,
                (user_id,),
            )
            reserved = cur.fetchone()
            reserved_grant = int(reserved["reserved_grant_micro"])
            reserved_purchased = int(reserved["reserved_purchased_micro"])
        else:
            reserved_grant = reserved_purchased = 0
        return {
            "grant_committed_micro": grant_micro,
            "purchased_committed_micro": purchased_micro,
            "grant_available_micro": grant_micro - reserved_grant,
            "purchased_available_micro": purchased_micro - reserved_purchased,
            "total_available_micro": grant_micro + purchased_micro - reserved_grant - reserved_purchased,
        }

    @staticmethod
    def _balance_in_transaction(cur, user_id: int) -> int:
        projection = PostgresCreditsStore._balance_projection_in_transaction(
            cur, user_id
        )
        return projection["total_available_micro"]

    def get_order_for_user(self, order_id: str, user_id: int) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM credit_payment_orders
                    WHERE id = %s AND user_id = %s
                    """,
                    (order_id, user_id),
                )
                row = cur.fetchone()
                return dict(row) if row else None

    def get_order_for_admin(self, order_id: str) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM credit_payment_orders WHERE id = %s", (order_id,)
                )
                row = cur.fetchone()
                return dict(row) if row else None

    def get_order_by_payment_intent(
        self, payment_intent_id: str
    ) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM credit_payment_orders
                    WHERE stripe_payment_intent_id = %s
                    """,
                    (payment_intent_id,),
                )
                row = cur.fetchone()
                return dict(row) if row else None

    def get_refund_by_stripe_id(self, stripe_refund_id: str) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM credit_refund_requests
                    WHERE stripe_refund_id = %s
                    """,
                    (stripe_refund_id,),
                )
                row = cur.fetchone()
                return dict(row) if row else None

    def get_refund_by_id(self, refund_id: str) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM credit_refund_requests WHERE id = %s", (refund_id,)
                )
                row = cur.fetchone()
                return dict(row) if row else None

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
            with conn.cursor() as cur:
                self._ensure_account_in_transaction(cur, user_id)
                cur.execute(
                    """
                    UPDATE credit_accounts
                    SET status = 'restricted', restriction_reason = %s
                    WHERE user_id = %s
                    RETURNING *
                    """,
                    (reason, user_id),
                )
                return dict(cur.fetchone())

    def reinstate_account(self, user_id: int) -> dict[str, Any]:
        """Twin of ``CreditsStore.reinstate_account``."""
        _positive_integer(user_id, "user_id")
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                self._ensure_account_in_transaction(cur, user_id)
                cur.execute(
                    """
                    UPDATE credit_accounts SET status = 'active', restriction_reason = NULL
                    WHERE user_id = %s
                    RETURNING *
                    """,
                    (user_id,),
                )
                return dict(cur.fetchone())

    def list_ledger_entries(
        self,
        user_id: int,
        *,
        limit: int = 50,
        cursor: str | int | None = None,
    ) -> dict[str, Any]:
        page_size = _positive_limit(limit)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                boundary: tuple[str, str, int] | None = None
                if cursor is not None:
                    decoded = decode_activity_cursor(cursor)
                    if isinstance(decoded, int):
                        cur.execute(
                            "SELECT created_at FROM credit_ledger_entries "
                            "WHERE user_id = %s AND id = %s",
                            (user_id, decoded),
                        )
                        legacy = cur.fetchone()
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
                        WHERE created_at < %s
                           OR (created_at = %s AND source_kind < %s)
                           OR (created_at = %s AND source_kind = %s AND source_id < %s)
                    """
                    params.extend([
                        created_at, created_at, source_kind,
                        created_at, source_kind, source_id,
                    ])
                params.append(page_size + 1)
                cur.execute(
                    f"""
                    WITH historical_activity AS (
                        SELECT id AS source_id, 'ledger' AS source_kind,
                               user_id, bucket, entry_type, amount_micro,
                               payment_order_id, refund_request_id,
                               stripe_event_id, operation_key, operation_id,
                               idempotency_key, request_digest, actor_user_id,
                               source, reason, reference_type, reference_id,
                               created_at,
                               NULL::TEXT AS reservation_id, NULL::TEXT AS run_id,
                               NULL::BIGINT AS call_index,
                               NULL::BIGINT AS model_call_count,
                               NULL::TEXT AS evidence_json
                        FROM credit_ledger_entries
                        WHERE user_id = %s
                    ),
                    llm_activity AS (
                        SELECT MAX(id) AS source_id, 'llm_usage' AS source_kind,
                               user_id, NULL::TEXT AS bucket,
                               'backtest_usage' AS entry_type,
                               SUM(amount_micro) AS amount_micro,
                               NULL::TEXT AS payment_order_id,
                               NULL::TEXT AS refund_request_id,
                               NULL::TEXT AS stripe_event_id,
                               NULL::TEXT AS operation_key,
                               NULL::TEXT AS operation_id,
                               NULL::TEXT AS idempotency_key,
                               NULL::TEXT AS request_digest,
                               NULL::INTEGER AS actor_user_id,
                               'llm_execution' AS source,
                               'Backtest usage.' AS reason,
                               NULL::TEXT AS reference_type,
                               NULL::TEXT AS reference_id,
                               MAX(created_at) AS created_at,
                               NULL::TEXT AS reservation_id, run_id,
                               NULL::BIGINT AS call_index,
                               COUNT(DISTINCT call_index) AS model_call_count,
                               NULL::TEXT AS evidence_json
                        FROM credit_llm_usage_entries
                        WHERE user_id = %s
                          AND operation_key NOT LIKE '%%:recovery:%%'
                        GROUP BY user_id, run_id
                    ),
                    promotion_activity AS (
                        SELECT id AS source_id, 'promotion' AS source_kind,
                               user_id, 'grant'::TEXT AS bucket,
                               'system_promotion_grant' AS entry_type,
                               amount_micro,
                               NULL::TEXT AS payment_order_id,
                               NULL::TEXT AS refund_request_id,
                               NULL::TEXT AS stripe_event_id,
                               'promotion:' || operation_id AS operation_key,
                               operation_id, idempotency_key, request_digest,
                               NULL::INTEGER AS actor_user_id,
                               source, reason, 'promotion'::TEXT AS reference_type,
                               campaign_key AS reference_id, created_at,
                               NULL::TEXT AS reservation_id,
                               NULL::TEXT AS run_id,
                               NULL::BIGINT AS call_index,
                               NULL::BIGINT AS model_call_count,
                               NULL::TEXT AS evidence_json
                        FROM credit_promotion_grants
                        WHERE user_id = %s
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
                    LIMIT %s
                    """,
                    params,
                )
                rows = cur.fetchall()
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
                    cur.execute(
                        """
                        SELECT DISTINCT run_id, evidence_json
                        FROM credit_llm_usage_entries
                        WHERE user_id = %s AND run_id = ANY(%s)
                          AND operation_key NOT LIKE '%%:recovery:%%'
                        """,
                        (user_id, run_ids),
                    )
                    for evidence_row in cur.fetchall():
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
    def _refundable_in_transaction(cur, order: dict[str, Any]) -> tuple[int, int]:
        cur.execute(
            """
            SELECT
                COALESCE(SUM(amount_usd_cents), 0) AS reserved_cents,
                COALESCE(SUM(credits_micro), 0) AS reserved_micro
            FROM credit_refund_requests
            WHERE payment_order_id = %s
              AND status IN ('pending', 'submitted', 'succeeded')
            """,
            (order["id"],),
        )
        row = cur.fetchone()
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
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT * FROM credit_payment_orders WHERE id = %s
                        FOR UPDATE
                        """,
                        (payment_order_id,),
                    )
                    order = cur.fetchone()
                    cur.execute(
                        """
                        SELECT * FROM credit_refund_requests WHERE id = %s
                        FOR UPDATE
                        """,
                        (refund_id,),
                    )
                    existing = cur.fetchone()
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

                    if not order or order["user_id"] != user_id:
                        raise RefundNotAllowedError("paid purchase was not found")
                    if order["status"] not in {"paid", "partially_refunded"}:
                        raise RefundNotAllowedError("purchase is not refundable")
                    refundable_cents, refundable_micro = (
                        self._refundable_in_transaction(cur, order)
                    )
                    if (
                        amount_usd_cents > refundable_cents
                        or credits_micro > refundable_micro
                    ):
                        raise RefundNotAllowedError(
                            "refund exceeds the unused purchased Credits"
                        )
                    now = _utcnow_iso()
                    cur.execute(
                        """
                        INSERT INTO credit_refund_requests (
                            id, payment_order_id, user_id, requested_by_user_id,
                            amount_usd_cents, credits_micro, status,
                            created_at, updated_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, 'pending', %s, %s)
                        RETURNING *
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
                    return dict(cur.fetchone())
        except psycopg.errors.UniqueViolation as exc:
            raise OrderConflictError("refund ID already exists") from exc

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
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT * FROM credit_payment_orders WHERE id = %s
                        FOR UPDATE
                        """,
                        (payment_order_id,),
                    )
                    order = cur.fetchone()
                    cur.execute(
                        """
                        SELECT * FROM credit_refund_requests
                        WHERE id = %s OR stripe_refund_id = %s
                        FOR UPDATE
                        """,
                        (refund_id, stripe_refund_id),
                    )
                    existing = cur.fetchone()
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
                    if not order or order["user_id"] != user_id:
                        raise RefundNotAllowedError("paid purchase was not found")
                    if order["status"] not in {"paid", "partially_refunded"}:
                        raise RefundNotAllowedError("purchase is not refundable")
                    refundable_cents, refundable_micro = (
                        self._refundable_in_transaction(cur, order)
                    )
                    if (
                        amount_usd_cents > refundable_cents
                        or credits_micro > refundable_micro
                    ):
                        raise RefundNotAllowedError(
                            "refund exceeds the unused purchased Credits"
                        )
                    now = _utcnow_iso()
                    cur.execute(
                        """
                        INSERT INTO credit_refund_requests (
                            id, payment_order_id, user_id, requested_by_user_id,
                            amount_usd_cents, credits_micro, status,
                            stripe_refund_id, created_at, updated_at
                        )
                        VALUES (%s, %s, %s, NULL, %s, %s, 'submitted', %s, %s, %s)
                        RETURNING *
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
                    return dict(cur.fetchone())
        except psycopg.errors.UniqueViolation as exc:
            raise OrderConflictError(
                "Stripe Refund is already attached to another request"
            ) from exc

    def attach_stripe_refund(
        self, refund_id: str, *, stripe_refund_id: str
    ) -> dict[str, Any]:
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT * FROM credit_refund_requests WHERE id = %s
                        FOR UPDATE
                        """,
                        (refund_id,),
                    )
                    row = cur.fetchone()
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
                        cur.execute(
                            """
                            UPDATE credit_refund_requests
                            SET stripe_refund_id = %s, status = 'submitted',
                                updated_at = %s
                            WHERE id = %s
                            RETURNING *
                            """,
                            (stripe_refund_id, _utcnow_iso(), refund_id),
                        )
                        return dict(cur.fetchone())
                    return dict(row)
        except psycopg.errors.UniqueViolation as exc:
            raise OrderConflictError(
                "Stripe Refund is already attached to another request"
            ) from exc

    def cancel_refund_reservation(self, refund_id: str) -> dict[str, Any] | None:
        """Release a reservation whose Stripe call never landed.

        Twin of ``CreditsStore.cancel_refund_reservation``; see that docstring
        for why ``cancelled`` exists and why only a still-``pending`` row with
        no Stripe Refund attached may be released.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM credit_refund_requests WHERE id = %s
                    FOR UPDATE
                    """,
                    (refund_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                if row["status"] != "pending" or row["stripe_refund_id"]:
                    return dict(row)
                cur.execute(
                    """
                    UPDATE credit_refund_requests
                    SET status = 'cancelled', updated_at = %s
                    WHERE id = %s AND status = 'pending'
                      AND stripe_refund_id IS NULL
                    RETURNING *
                    """,
                    (_utcnow_iso(), refund_id),
                )
                updated = cur.fetchone()
                return dict(updated) if updated else dict(row)

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
            with conn.cursor() as cur:
                self._lock_event(cur, event_id)
                existing_event = self._existing_event(
                    cur,
                    event_id=event_id,
                    event_type=event_type,
                    livemode=livemode,
                    object_id=object_id,
                    payload_sha256=payload_sha256,
                )
                if existing_event:
                    cur.execute(
                        "SELECT user_id FROM credit_refund_requests WHERE id = %s",
                        (refund_id,),
                    )
                    refund = cur.fetchone()
                    balance = (
                        self._balance_in_transaction(cur, refund["user_id"])
                        if refund
                        else 0
                    )
                    return {"outcome": "duplicate", "balance_micro": balance}

                # Lock ORDER first, then REFUND. That ordering is load-bearing,
                # not stylistic: reserve_refund and reserve_reconciliation_refund
                # both take the two row locks in that order, and taking them the
                # other way round here deadlocks an admin's second partial
                # refund against a webhook settling the first one on the same
                # order. Postgres resolves that with 40P01, which no caller
                # catches — it escapes as a bare 500 to the admin and as an
                # un-retried 500 to Stripe. The SQLite twin cannot reproduce it
                # (one whole-DB BEGIN IMMEDIATE serializes both paths), so the
                # entire SQLite suite stays green either way.
                #
                # The unlocked probe only resolves payment_order_id, which is
                # write-once on a refund row; every value the decision below
                # rests on is re-read under FOR UPDATE afterwards.
                cur.execute(
                    """
                    SELECT payment_order_id FROM credit_refund_requests
                    WHERE id = %s
                    """,
                    (refund_id,),
                )
                probe = cur.fetchone()
                refund = None
                order = None
                if probe:
                    cur.execute(
                        """
                        SELECT * FROM credit_payment_orders WHERE id = %s FOR UPDATE
                        """,
                        (probe["payment_order_id"],),
                    )
                    order = cur.fetchone()
                    cur.execute(
                        """
                        SELECT * FROM credit_refund_requests WHERE id = %s
                        FOR UPDATE
                        """,
                        (refund_id,),
                    )
                    refund = cur.fetchone()

                reason = None
                if not refund or not order:
                    reason = "refund request was not found"
                elif livemode:
                    reason = "Live Mode refund is not accepted"
                elif (
                    object_id != stripe_refund_id
                    or refund["stripe_refund_id"] != stripe_refund_id
                ):
                    reason = "Stripe Refund does not match the request"
                elif order["stripe_payment_intent_id"] != payment_intent_id:
                    reason = "PaymentIntent does not match the purchase"
                elif currency.lower() != order["currency"]:
                    reason = "refund currency does not match the purchase"
                elif amount_usd_cents != refund["amount_usd_cents"]:
                    reason = "refund amount does not match the request"

                if reason:
                    self._insert_event(
                        cur,
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
                cur.execute(
                    "SELECT id FROM credit_ledger_entries WHERE operation_key = %s",
                    (operation_key,),
                )
                if cur.fetchone() or refund["status"] == "succeeded":
                    self._insert_event(
                        cur,
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
                        "balance_micro": self._balance_in_transaction(
                            cur, refund["user_id"]
                        ),
                    }
                if refund["status"] not in {"pending", "submitted"}:
                    reason = "refund request is not awaiting settlement"
                    self._insert_event(
                        cur,
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
                    cur,
                    event_id=event_id,
                    event_type=event_type,
                    livemode=livemode,
                    object_id=object_id,
                    payload_sha256=payload_sha256,
                    outcome="processed",
                )
                cur.execute(
                    """
                    INSERT INTO credit_ledger_entries (
                        user_id, bucket, entry_type, amount_micro,
                        payment_order_id, refund_request_id, stripe_event_id,
                        operation_key, operation_id, idempotency_key,
                        source, reason, created_at
                    )
                    VALUES (
                        %s, 'purchased', 'refund', %s, %s, %s, %s,
                        %s, %s, %s, 'stripe', 'Stripe refund.', %s
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
                cur.execute(
                    """
                    UPDATE credit_refund_requests
                    SET status = 'succeeded', updated_at = %s, succeeded_at = %s
                    WHERE id = %s
                    """,
                    (now, now, refund_id),
                )
                cur.execute(
                    """
                    SELECT COALESCE(SUM(amount_usd_cents), 0) AS cents
                    FROM credit_refund_requests
                    WHERE payment_order_id = %s AND status = 'succeeded'
                    """,
                    (refund["payment_order_id"],),
                )
                successful = cur.fetchone()
                order_status = (
                    "refunded"
                    if int(successful["cents"]) >= int(order["amount_usd_cents"])
                    else "partially_refunded"
                )
                cur.execute(
                    """
                    UPDATE credit_payment_orders SET status = %s, updated_at = %s
                    WHERE id = %s
                    """,
                    (order_status, now, refund["payment_order_id"]),
                )
                return {
                    "outcome": "processed",
                    "balance_micro": self._balance_in_transaction(
                        cur, refund["user_id"]
                    ),
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
            with conn.cursor() as cur:
                self._lock_event(cur, event_id)
                existing = self._existing_event(
                    cur,
                    event_id=event_id,
                    event_type=event_type,
                    livemode=livemode,
                    object_id=object_id,
                    payload_sha256=payload_sha256,
                )
                if existing:
                    return {"outcome": "duplicate"}
                cur.execute(
                    """
                    SELECT * FROM credit_refund_requests WHERE id = %s FOR UPDATE
                    """,
                    (refund_id,),
                )
                refund = cur.fetchone()
                if (
                    not refund
                    or livemode
                    or object_id != stripe_refund_id
                    or refund["stripe_refund_id"] != stripe_refund_id
                    or refund["status"] not in {"pending", "submitted"}
                ):
                    reason = "refund failure event does not match an active request"
                    self._insert_event(
                        cur,
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
                    cur,
                    event_id=event_id,
                    event_type=event_type,
                    livemode=livemode,
                    object_id=object_id,
                    payload_sha256=payload_sha256,
                    outcome="processed",
                )
                cur.execute(
                    """
                    UPDATE credit_refund_requests
                    SET status = 'failed', updated_at = %s WHERE id = %s
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
            cursor_sql = "AND o.sequence < %s"
            params.append(cursor)
        params.append(page_size + 1)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
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
                    LIMIT %s
                    """,
                    params,
                )
                rows = cur.fetchall()
        has_more = len(rows) > page_size
        items = [dict(row) for row in rows[:page_size]]
        return {
            "items": items,
            "next_cursor": items[-1]["sequence"] if has_more and items else None,
        }

    @staticmethod
    def _ensure_pool_in_transaction(cur, pool_id: str) -> dict[str, Any]:
        if pool_id == "default":
            cur.execute(
                """
                INSERT INTO credit_grant_pools (
                    pool_id, name, status, created_at
                ) VALUES (
                    'default', 'Platform Research Grants', 'active', %s
                )
                ON CONFLICT (pool_id) DO NOTHING
                """,
                (_utcnow_iso(),),
            )
        cur.execute(
            "SELECT * FROM credit_grant_pools WHERE pool_id = %s FOR UPDATE",
            (pool_id,),
        )
        pool = cur.fetchone()
        if pool is None:
            raise ValueError("grant pool does not exist")
        return dict(pool)

    @staticmethod
    def _pool_balance_in_transaction(
        cur,
        pool_id: str,
        through_entry_id: int | None = None,
    ) -> int:
        cutoff_sql = ""
        params: list[Any] = [pool_id]
        if through_entry_id is not None:
            cutoff_sql = "AND id <= %s"
            params.append(through_entry_id)
        cur.execute(
            f"""
            SELECT COALESCE(SUM(amount_micro), 0) AS balance_micro
            FROM credit_grant_pool_ledger_entries
            WHERE pool_id = %s
              {cutoff_sql}
            """,
            params,
        )
        return int(cur.fetchone()["balance_micro"])

    @staticmethod
    def _insert_user_grant_entry_in_transaction(
        cur,
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
        cur.execute(
            """
            INSERT INTO credit_ledger_entries (
                user_id, bucket, entry_type, amount_micro,
                payment_order_id, refund_request_id, stripe_event_id,
                operation_key, operation_id, idempotency_key,
                request_digest, actor_user_id, source, reason,
                reference_type, reference_id, created_at
            ) VALUES (
                %s, 'grant', %s, %s, NULL, NULL, NULL,
                %s, %s, %s, %s, %s, %s, %s, 'grant_pool', %s, %s
            )
            RETURNING id
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
        return int(cur.fetchone()["id"])

    @staticmethod
    def _insert_grant_pool_entry_in_transaction(
        cur,
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
        cur.execute(
            """
            INSERT INTO credit_grant_pool_ledger_entries (
                pool_id, pool_name_snapshot, pool_status_snapshot,
                entry_type, amount_micro, operation_id,
                idempotency_key, request_digest, actor_user_id,
                source, reason, user_id, user_ledger_entry_id, created_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s
            )
            RETURNING id
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
        return int(cur.fetchone()["id"])

    @classmethod
    def _grant_mutation_result_in_transaction(
        cls, cur, pool_entry: dict[str, Any]
    ) -> dict[str, Any]:
        user_id = pool_entry["user_id"]
        user_entry = None
        user_balance = None
        if pool_entry["user_ledger_entry_id"] is not None:
            cur.execute(
                "SELECT * FROM credit_ledger_entries WHERE id = %s",
                (pool_entry["user_ledger_entry_id"],),
            )
            user_entry = cur.fetchone()
        if user_id is not None:
            user_balance = cls._balance_projection_in_transaction(
                cur,
                int(user_id),
                through_entry_id=int(pool_entry["user_ledger_entry_id"]),
            )
        return {
            "entry": dict(pool_entry),
            "user_entry": dict(user_entry) if user_entry else None,
            "pool": {
                "pool_id": pool_entry["pool_id"],
                "name": pool_entry["pool_name_snapshot"],
                "status": pool_entry["pool_status_snapshot"],
                "balance_micro": cls._pool_balance_in_transaction(
                    cur,
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
            amount_micro if operation_type in {"fund", "reclaim"} else -amount_micro
        )

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                pool = self._ensure_pool_in_transaction(cur, pool_id)

                account = None
                if operation_type in {"assign", "reclaim"}:
                    self._ensure_account_in_transaction(cur, int(user_id))
                    cur.execute(
                        """
                        SELECT status, restriction_reason FROM credit_accounts
                        WHERE user_id = %s FOR UPDATE
                        """,
                        (user_id,),
                    )
                    account = cur.fetchone()

                for lock_key in sorted(
                    {
                        f"grant-idempotency:{idempotency_key}",
                        f"grant-operation:{operation_id}",
                    }
                ):
                    cur.execute(
                        "SELECT pg_advisory_xact_lock(hashtext(%s))", (lock_key,)
                    )

                cur.execute(
                    """
                    SELECT * FROM credit_grant_pool_ledger_entries
                    WHERE idempotency_key = %s OR operation_id = %s
                    FOR UPDATE
                    """,
                    (idempotency_key, operation_id),
                )
                existing = cur.fetchone()
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
                    result = self._grant_mutation_result_in_transaction(cur, existing)
                    if operation_type == "assign":
                        recovery = self._recovery_result_for_source_operation_in_transaction(
                            cur, int(user_id), source_operation_key=operation_id
                        )
                        if recovery is not None:
                            result["recovery"] = recovery
                    return result

                pool_balance = self._pool_balance_in_transaction(cur, pool_id)
                if (
                    operation_type in {"reduce", "assign"}
                    and pool_balance < amount_micro
                ):
                    raise GrantPoolInsufficientError(
                        "Grant Pool does not have enough available Credits"
                    )

                user_entry_id = None
                created_at = _utcnow_iso()
                if operation_type == "assign":
                    if (
                        account["status"] == "restricted"
                        and account["restriction_reason"] != "llm_overage"
                    ):
                        raise CreditAccountRestrictedStoreError(
                            "refund-review credit account requires administrator review"
                        )
                    user_entry_id = self._insert_user_grant_entry_in_transaction(
                        cur,
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
                        cur, int(user_id)
                    )
                    if projection["grant_available_micro"] < amount_micro:
                        raise GrantReclaimExceedsAvailableError(
                            "reclaim exceeds available Grant Credits"
                        )
                    user_entry_id = self._insert_user_grant_entry_in_transaction(
                        cur,
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
                    cur,
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
                cur.execute(
                    """
                    SELECT * FROM credit_grant_pool_ledger_entries WHERE id = %s
                    """,
                    (pool_entry_id,),
                )
                pool_entry = cur.fetchone()
                if operation_type == "assign":
                    recovery = self._recover_llm_overage_in_transaction(
                        cur, int(user_id), source_operation_key=operation_id
                    )
                else:
                    recovery = {}
                result = self._grant_mutation_result_in_transaction(cur, pool_entry)
                if recovery.get("recovered_micro", 0) > 0:
                    result["recovery"] = recovery
                return result

    def fund_grant_pool(self, **kwargs: Any) -> dict[str, Any]:
        return self._grant_mutation(operation_type="fund", **kwargs)

    def reduce_grant_pool(self, **kwargs: Any) -> dict[str, Any]:
        return self._grant_mutation(operation_type="reduce", **kwargs)

    def assign_grant(self, *, user_id: int, **kwargs: Any) -> dict[str, Any]:
        return self._grant_mutation(operation_type="assign", user_id=user_id, **kwargs)

    def reclaim_grant(self, *, user_id: int, **kwargs: Any) -> dict[str, Any]:
        return self._grant_mutation(operation_type="reclaim", user_id=user_id, **kwargs)

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
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        p.pool_id,
                        p.name AS pool_name,
                        p.status AS pool_status,
                        COALESCE((
                            SELECT SUM(amount_micro)
                            FROM credit_grant_pool_ledger_entries
                            WHERE pool_id = p.pool_id
                        ), 0) AS pool_available_micro,
                        COALESCE((
                            SELECT SUM(amount_micro)
                            FROM credit_ledger_entries
                            WHERE bucket = 'grant'
                              AND reference_type = 'grant_pool'
                              AND reference_id = p.pool_id
                        ), 0) + CASE WHEN p.pool_id = 'default' THEN COALESCE((
                            SELECT SUM(amount_micro)
                            FROM credit_llm_usage_entries
                            WHERE bucket = 'grant'
                        ), 0) ELSE 0 END AS allocated_to_users_micro,
                        COALESCE((
                            SELECT SUM(
                                CASE WHEN entry_type = 'assign'
                                    THEN -amount_micro ELSE 0 END
                            )
                            FROM credit_grant_pool_ledger_entries
                            WHERE pool_id = p.pool_id AND created_at >= %s
                        ), 0) AS assigned_this_month_micro,
                        COALESCE((
                            SELECT SUM(
                                CASE WHEN entry_type = 'reclaim'
                                    THEN amount_micro ELSE 0 END
                            )
                            FROM credit_grant_pool_ledger_entries
                            WHERE pool_id = p.pool_id AND created_at >= %s
                        ), 0) AS reclaimed_this_month_micro
                    FROM credit_grant_pools p
                    WHERE p.pool_id = %s
                    """,
                    (boundary, boundary, pool_id),
                )
                row = cur.fetchone()
        if row is None:
            raise ValueError("grant pool does not exist")
        return {
            "pool_id": row["pool_id"],
            "pool_name": row["pool_name"],
            "pool_status": row["pool_status"],
            "pool_available_micro": int(row["pool_available_micro"]),
            "allocated_to_users_micro": int(row["allocated_to_users_micro"]),
            "assigned_this_month_micro": int(row["assigned_this_month_micro"]),
            "reclaimed_this_month_micro": int(row["reclaimed_this_month_micro"]),
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
            cursor_sql = "AND id < %s"
            params.append(_positive_integer(cursor, "cursor"))
        params.append(page_size + 1)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT * FROM credit_grant_pool_ledger_entries
                    WHERE pool_id = %s {cursor_sql}
                    ORDER BY id DESC
                    LIMIT %s
                    """,
                    params,
                )
                rows = cur.fetchall()
        has_more = len(rows) > page_size
        items = [dict(row) for row in rows[:page_size]]
        return {
            "items": items,
            "next_cursor": items[-1]["id"] if has_more and items else None,
        }
