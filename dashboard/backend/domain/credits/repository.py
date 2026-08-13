"""SQLite persistence for Credits, payment operations, and webhook receipts."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dashboard.backend.database import DB_PATH
from dashboard.backend.db_url import describe_database_url


class CreditsStoreError(RuntimeError):
    """Base class for expected Credits-store failures."""


class OrderConflictError(CreditsStoreError):
    """An idempotent operation was retried with different data."""


class RefundNotAllowedError(CreditsStoreError):
    """A refund would exceed the unused, unrefunded purchase lot."""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_amount_pair(amount_usd_cents: int, credits_micro: int) -> None:
    cents = _positive_integer(amount_usd_cents, "amount_usd_cents")
    credits = _positive_integer(credits_micro, "credits_micro")
    if credits != cents * 10_000:
        raise ValueError("credits_micro must equal amount_usd_cents * 10,000")


def _positive_limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100:
        raise ValueError("limit must be an integer from 1 through 100")
    return value


def _dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


class CreditsStore:
    """Account-scoped append-only Credits ledger backed by SQLite."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path or DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS credit_accounts (
                    user_id INTEGER PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'restricted')),
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

                CREATE TABLE IF NOT EXISTS credit_ledger_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    entry_type TEXT NOT NULL
                        CHECK (entry_type IN ('purchase', 'refund')),
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

                CREATE INDEX IF NOT EXISTS idx_credit_ledger_user_id
                ON credit_ledger_entries(user_id, id DESC);

                CREATE INDEX IF NOT EXISTS idx_credit_ledger_payment_order
                ON credit_ledger_entries(payment_order_id, id DESC);
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

    def get_balance_micro(self, user_id: int) -> int:
        _positive_integer(user_id, "user_id")
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(amount_micro), 0) AS balance_micro
                FROM credit_ledger_entries
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
            return int(row["balance_micro"])

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
            elif checkout_session_id != order["stripe_checkout_session_id"]:
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
                    user_id, entry_type, amount_micro, payment_order_id,
                    refund_request_id, stripe_event_id, operation_key, created_at
                )
                VALUES (?, 'purchase', ?, ?, NULL, ?, ?, ?)
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
            conn.execute(
                """
                UPDATE credit_payment_orders
                SET status = 'paid', stripe_payment_intent_id = ?,
                    updated_at = ?, paid_at = COALESCE(paid_at, ?)
                WHERE id = ?
                """,
                (payment_intent_id, now, now, order_id),
            )
            return {
                "outcome": "processed",
                "balance_micro": self._balance_in_transaction(conn, order["user_id"]),
            }

    @staticmethod
    def _balance_in_transaction(conn: sqlite3.Connection, user_id: int) -> int:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(amount_micro), 0) AS balance_micro
            FROM credit_ledger_entries WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
        return int(row["balance_micro"])

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

    def restrict_account(self, user_id: int) -> dict[str, Any]:
        _positive_integer(user_id, "user_id")
        with self._get_connection() as conn:
            self._begin(conn)
            self._ensure_account_in_transaction(conn, user_id)
            conn.execute(
                "UPDATE credit_accounts SET status = 'restricted' WHERE user_id = ?",
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
        cursor: int | None = None,
    ) -> dict[str, Any]:
        page_size = _positive_limit(limit)
        params: list[Any] = [user_id]
        cursor_sql = ""
        if cursor is not None:
            _positive_integer(cursor, "cursor")
            cursor_sql = "AND id < ?"
            params.append(cursor)
        params.append(page_size + 1)
        with self._get_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM credit_ledger_entries
                WHERE user_id = ? {cursor_sql}
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
                    user_id, entry_type, amount_micro, payment_order_id,
                    refund_request_id, stripe_event_id, operation_key, created_at
                )
                VALUES (?, 'refund', ?, ?, ?, ?, ?, ?)
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
