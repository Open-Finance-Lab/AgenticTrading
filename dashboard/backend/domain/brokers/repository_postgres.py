"""Postgres-backed BrokerConnectionStore implementation.

Selected instead of the default SQLite BrokerConnectionStore when
``USERS_DATABASE_URL`` or ``CONTENT_DATABASE_URL`` is set (see repository.py's
``_build_user_store``). Exists because the SQLite store lives in ``DB_PATH``,
which resets to the committed seed database on every deploy of the disk-less
Render free-tier host -- silently discarding every linked brokerage account and
its refresh token, with no error surfaced anywhere. Method surface, keyword
arguments, and return schemas are identical to BrokerConnectionStore; only the
SQL dialect and the timestamp column types differ.

Encryption is *not* reimplemented here: ``_encrypt``/``_decrypt`` are imported
from the SQLite twin so both backends share one key derivation and one
ciphertext format, and a connection written by either can be read by the other.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from dashboard.backend.db_url import require_postgres_url
from dashboard.backend.domain.brokers.repository import (
    _BROKER,
    _decrypt,
    _encrypt,
    _public_row,
    _utcnow_iso,
)

# Columns declared TIMESTAMPTZ below; psycopg returns them as datetime, while
# the SQLite twin stores and returns the ISO text _utcnow_iso() produced.
_TIMESTAMP_COLUMNS = ("connected_at", "updated_at")


def _utcnow_ts() -> datetime:
    """Second-truncated, timezone-aware ``now`` for the TIMESTAMPTZ columns.

    Derived from the shared ``_utcnow_iso`` rather than calling
    ``datetime.now`` again, so both twins stamp rows from one clock helper with
    identical truncation. Passed to psycopg as a real ``datetime`` so the
    server never has to infer a type for an untyped string literal.
    """
    return datetime.fromisoformat(_utcnow_iso())


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Render the TIMESTAMPTZ columns as ISO strings, matching the SQLite twin.

    ``_public_row`` passes ``connected_at`` straight through to the API
    response, so without this the two backends would emit different JSON for
    the same connection.
    """
    data = dict(row)
    for column in _TIMESTAMP_COLUMNS:
        value = data.get(column)
        if isinstance(value, datetime):
            data[column] = (
                value.astimezone(timezone.utc).replace(microsecond=0).isoformat()
            )
    return data


class BrokerConnectionStorePostgres:
    """Persist encrypted broker OAuth tokens, backed by Postgres."""

    def __init__(self, database_url: str):
        self.database_url = require_postgres_url(database_url)
        self._init_schema()

    def _get_connection(self):
        # Pooled checkout: same context-manager transaction semantics as
        # psycopg.connect (commit on clean exit), returned to the pool on close.
        from dashboard.backend.db_pool import get_pool
        return get_pool(self.database_url).connection()

    def _init_schema(self) -> None:
        # Runs once per process, from __init__ -- not per query.
        #
        # ADDING A COLUMN LATER? It must go in an `ALTER TABLE ... ADD COLUMN IF
        # NOT EXISTS` below, *not* only in the CREATE. CREATE TABLE IF NOT
        # EXISTS silently no-ops once the table exists, so an existing
        # deployment would never gain the column and every query naming it would
        # raise UndefinedColumn -- 500ing this whole surface while /health stays
        # green. SQLite is the default tier in tests and CI's Postgres container
        # is empty on every run, so the @pg_only tier only exercises the CREATE
        # path. See agents/repository_postgres.py for the worked example.
        #
        # user_id is a plain BIGINT with no FK to users(id): the SQLite twin's
        # declared FK was never enforced (nothing sets PRAGMA foreign_keys), and
        # a real FK would break the supported split config where
        # USERS_DATABASE_URL points at a different database than
        # CONTENT_DATABASE_URL -- Postgres has no cross-database foreign keys.
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS broker_connections (
                        user_id BIGINT NOT NULL,
                        broker TEXT NOT NULL DEFAULT 'robinhood',
                        access_token_enc TEXT NOT NULL,
                        refresh_token_enc TEXT,
                        client_id TEXT,
                        agentic_account_id TEXT,
                        token_expires_at TEXT,
                        metadata_json TEXT,
                        connected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (user_id, broker)
                    )
                    """
                )

    def get_public(self, user_id: int, broker: str = _BROKER) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM broker_connections WHERE user_id = %s AND broker = %s",
                    (int(user_id), broker),
                )
                row = cur.fetchone()
        if not row:
            return None
        return _public_row(_normalize_row(row))

    def get_tokens(self, user_id: int, broker: str = _BROKER) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM broker_connections WHERE user_id = %s AND broker = %s",
                    (int(user_id), broker),
                )
                row = cur.fetchone()
        if not row:
            return None
        data = dict(row)
        return {
            "access_token": _decrypt(data["access_token_enc"]),
            "refresh_token": (
                _decrypt(data["refresh_token_enc"]) if data.get("refresh_token_enc") else None
            ),
            "client_id": data.get("client_id"),
            "agentic_account_id": data.get("agentic_account_id"),
            "token_expires_at": data.get("token_expires_at"),
        }

    def upsert_tokens(
        self,
        user_id: int,
        *,
        broker: str = _BROKER,
        access_token: str,
        refresh_token: Optional[str] = None,
        client_id: Optional[str] = None,
        agentic_account_id: Optional[str] = None,
        token_expires_at: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = _utcnow_ts()
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # COALESCE on client_id / agentic_account_id / metadata_json
                # mirrors the SQLite twin: a re-link that omits them must not
                # blank out what the original link recorded.
                cur.execute(
                    """
                    INSERT INTO broker_connections (
                        user_id, broker, access_token_enc, refresh_token_enc, client_id,
                        agentic_account_id, token_expires_at, metadata_json, connected_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, broker) DO UPDATE SET
                        access_token_enc = EXCLUDED.access_token_enc,
                        refresh_token_enc = EXCLUDED.refresh_token_enc,
                        client_id = COALESCE(EXCLUDED.client_id, broker_connections.client_id),
                        agentic_account_id = COALESCE(
                            EXCLUDED.agentic_account_id, broker_connections.agentic_account_id
                        ),
                        token_expires_at = EXCLUDED.token_expires_at,
                        metadata_json = COALESCE(
                            EXCLUDED.metadata_json, broker_connections.metadata_json
                        ),
                        updated_at = EXCLUDED.updated_at
                    RETURNING *
                    """,
                    (
                        int(user_id),
                        broker,
                        _encrypt(access_token),
                        _encrypt(refresh_token) if refresh_token else None,
                        client_id,
                        agentic_account_id,
                        token_expires_at,
                        json.dumps(metadata or {}),
                        now,
                        now,
                    ),
                )
                row = cur.fetchone()
        if row is None:  # pragma: no cover - the upsert above always returns a row
            raise RuntimeError(
                f"broker connection for user {user_id} vanished immediately after upsert"
            )
        return _public_row(_normalize_row(row))

    def update_tokens(
        self,
        user_id: int,
        *,
        broker: str = _BROKER,
        access_token: str,
        refresh_token: Optional[str] = None,
        token_expires_at: Optional[str] = None,
    ) -> None:
        now = _utcnow_ts()
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # The ::text cast keeps a NULL refresh_token from reaching
                # COALESCE as an untyped parameter, which Postgres cannot
                # resolve on its own.
                cur.execute(
                    """
                    UPDATE broker_connections
                    SET access_token_enc = %s,
                        refresh_token_enc = COALESCE(%s::text, refresh_token_enc),
                        token_expires_at = %s,
                        updated_at = %s
                    WHERE user_id = %s AND broker = %s
                    """,
                    (
                        _encrypt(access_token),
                        _encrypt(refresh_token) if refresh_token else None,
                        token_expires_at,
                        now,
                        int(user_id),
                        broker,
                    ),
                )

    def delete(self, user_id: int, broker: str = _BROKER) -> bool:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM broker_connections WHERE user_id = %s AND broker = %s",
                    (int(user_id), broker),
                )
                deleted = cur.rowcount > 0
        return deleted
