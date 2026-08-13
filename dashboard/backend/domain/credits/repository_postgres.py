"""Postgres persistence for Credits, payment operations, and webhook receipts."""

from __future__ import annotations

from typing import Any

import psycopg

from dashboard.backend.db_url import require_postgres_url
from dashboard.backend.domain.credits.repository_common import (
    OrderConflictError,
    RefundNotAllowedError,
    _positive_integer,
    _positive_limit,
    _utcnow_iso,
    _validate_amount_pair,
)


# Kept as literal DDL so the SQLite/Postgres parity guard can compare the
# authoritative table and column contracts without requiring a live database.
CREDITS_POSTGRES_DDL = """
CREATE TABLE IF NOT EXISTS credit_accounts (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'restricted')),
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
    entry_type TEXT NOT NULL CHECK (entry_type IN ('purchase', 'refund')),
    amount_micro BIGINT NOT NULL CHECK (amount_micro <> 0),
    payment_order_id TEXT NOT NULL
        REFERENCES credit_payment_orders(id) ON DELETE RESTRICT,
    refund_request_id TEXT
        REFERENCES credit_refund_requests(id) ON DELETE RESTRICT,
    stripe_event_id TEXT NOT NULL
        REFERENCES stripe_webhook_events(stripe_event_id) ON DELETE RESTRICT,
    operation_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_credit_ledger_user_id
ON credit_ledger_entries(user_id, id DESC);

CREATE INDEX IF NOT EXISTS idx_credit_ledger_payment_order
ON credit_ledger_entries(payment_order_id, id DESC);
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

    def get_balance_micro(self, user_id: int) -> int:
        _positive_integer(user_id, "user_id")
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                return self._balance_in_transaction(cur, user_id)

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
                elif checkout_session_id != order["stripe_checkout_session_id"]:
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
                    return {"outcome": "ignored", "reason": reason, "status": order["status"]}

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
                elif checkout_session_id != order["stripe_checkout_session_id"]:
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
                        user_id, entry_type, amount_micro, payment_order_id,
                        refund_request_id, stripe_event_id, operation_key, created_at
                    )
                    VALUES (%s, 'purchase', %s, %s, NULL, %s, %s, %s)
                    """,
                    (
                        order["user_id"],
                        order["credits_micro"],
                        order_id,
                        event_id,
                        operation_key,
                        now,
                    ),
                )
                cur.execute(
                    """
                    UPDATE credit_payment_orders
                    SET status = 'paid', stripe_payment_intent_id = %s,
                        updated_at = %s, paid_at = COALESCE(paid_at, %s)
                    WHERE id = %s
                    """,
                    (payment_intent_id, now, now, order_id),
                )
                return {
                    "outcome": "processed",
                    "balance_micro": self._balance_in_transaction(
                        cur, order["user_id"]
                    ),
                }

    @staticmethod
    def _balance_in_transaction(cur, user_id: int) -> int:
        cur.execute(
            """
            SELECT COALESCE(SUM(amount_micro), 0) AS balance_micro
            FROM credit_ledger_entries WHERE user_id = %s
            """,
            (user_id,),
        )
        return int(cur.fetchone()["balance_micro"])

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

    def restrict_account(self, user_id: int) -> dict[str, Any]:
        _positive_integer(user_id, "user_id")
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                self._ensure_account_in_transaction(cur, user_id)
                cur.execute(
                    """
                    UPDATE credit_accounts SET status = 'restricted'
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
        cursor: int | None = None,
    ) -> dict[str, Any]:
        page_size = _positive_limit(limit)
        params: list[Any] = [user_id]
        cursor_sql = ""
        if cursor is not None:
            _positive_integer(cursor, "cursor")
            cursor_sql = "AND id < %s"
            params.append(cursor)
        params.append(page_size + 1)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT * FROM credit_ledger_entries
                    WHERE user_id = %s {cursor_sql}
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

                cur.execute(
                    """
                    SELECT * FROM credit_refund_requests WHERE id = %s FOR UPDATE
                    """,
                    (refund_id,),
                )
                refund = cur.fetchone()
                order = None
                if refund:
                    cur.execute(
                        """
                        SELECT * FROM credit_payment_orders WHERE id = %s FOR UPDATE
                        """,
                        (refund["payment_order_id"],),
                    )
                    order = cur.fetchone()

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
                        user_id, entry_type, amount_micro, payment_order_id,
                        refund_request_id, stripe_event_id, operation_key, created_at
                    )
                    VALUES (%s, 'refund', %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        refund["user_id"],
                        -int(refund["credits_micro"]),
                        refund["payment_order_id"],
                        refund_id,
                        event_id,
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
