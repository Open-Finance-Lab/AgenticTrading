"""Persist encrypted OAuth tokens for user broker connections."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from dashboard.backend.database import DB_PATH
from dashboard.backend.db_url import describe_database_url

_BROKER = "robinhood"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


_fernet_instance: Optional[Any] = None

_KEY_ENV_VAR = "BROKER_TOKEN_ENCRYPTION_KEY"

_GENERATE_KEY_HINT = (
    'Generate one with: python -c "from cryptography.fernet import Fernet; '
    'print(Fernet.generate_key().decode())"'
)


def _get_fernet():
    """Return the process-wide Fernet, or raise if no usable key is configured.

    Fails closed. The previous fallback derived a key from the literal
    ``"dev-broker-token-encryption"`` when the env var was unset -- a constant
    published in this public repository, so anyone holding a copy of the
    database could decrypt every live-brokerage refresh token in it. That is
    plaintext storage wearing a ciphertext costume, and it applied by default.

    Raises at *call* time, never at import time: this module is imported
    transitively by the composition root, so an import-time raise would take
    the entire app down at boot instead of failing only the requests that
    actually touch broker credentials.
    """
    global _fernet_instance
    if _fernet_instance is None:
        from cryptography.fernet import Fernet

        raw = (os.getenv(_KEY_ENV_VAR) or "").strip()
        if not raw:
            raise RuntimeError(
                f"{_KEY_ENV_VAR} is not set - refusing to store broker credentials. "
                f"{_GENERATE_KEY_HINT}"
            )
        try:
            _fernet_instance = Fernet(raw.encode())
        except (ValueError, TypeError) as exc:
            # Name the variable, never echo its value: this message reaches
            # logs and, via a 500, potentially a response body.
            raise RuntimeError(
                f"{_KEY_ENV_VAR} is set but is not a valid Fernet key "
                f"(expected 32 url-safe base64-encoded bytes). {_GENERATE_KEY_HINT}"
            ) from exc
    return _fernet_instance


def _encrypt(value: str) -> str:
    return _get_fernet().encrypt(value.encode("utf-8")).decode("utf-8")


def _decrypt(value: str) -> str:
    return _get_fernet().decrypt(value.encode("utf-8")).decode("utf-8")


def _public_row(row: sqlite3.Row | Dict[str, Any]) -> Dict[str, Any]:
    data = dict(row)
    meta = {}
    if data.get("metadata_json"):
        try:
            meta = json.loads(data["metadata_json"])
        except (TypeError, ValueError):
            meta = {}
    return {
        "broker": data.get("broker") or _BROKER,
        "connected": True,
        "client_id": data.get("client_id"),
        "agentic_account_id": data.get("agentic_account_id"),
        "connected_at": data.get("connected_at"),
        "token_expires_at": data.get("token_expires_at"),
        "metadata": meta,
    }


class BrokerConnectionStore:
    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path or DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        # timeout=30 matches database.py: this file is shared with the backtest
        # DB layer, whose finalize writes can hold the write lock for seconds,
        # and the 5s sqlite3 default surfaces that as "database is locked" on a
        # token read.
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.Error:
            # journal_mode is persisted in the file, so this is a no-op after
            # the first successful switch. Filesystems without shared-memory
            # support (some network mounts) and read-only files refuse it --
            # keep the default rollback journal there rather than raising from
            # a method _init_schema calls at import time.
            pass
        return conn

    def _init_schema(self) -> None:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS broker_connections (
                user_id INTEGER NOT NULL,
                broker TEXT NOT NULL DEFAULT 'robinhood',
                access_token_enc TEXT NOT NULL,
                refresh_token_enc TEXT,
                client_id TEXT,
                agentic_account_id TEXT,
                token_expires_at TEXT,
                metadata_json TEXT,
                connected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, broker),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        conn.commit()
        conn.close()

    def get_public(self, user_id: int, broker: str = _BROKER) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM broker_connections WHERE user_id = ? AND broker = ?",
            (int(user_id), broker),
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        return _public_row(row)

    def get_tokens(self, user_id: int, broker: str = _BROKER) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM broker_connections WHERE user_id = ? AND broker = ?",
            (int(user_id), broker),
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        data = dict(row)
        return {
            "access_token": _decrypt(data["access_token_enc"]),
            "refresh_token": _decrypt(data["refresh_token_enc"]) if data.get("refresh_token_enc") else None,
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
        now = _utcnow_iso()
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO broker_connections (
                user_id, broker, access_token_enc, refresh_token_enc, client_id,
                agentic_account_id, token_expires_at, metadata_json, connected_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, broker) DO UPDATE SET
                access_token_enc = excluded.access_token_enc,
                refresh_token_enc = excluded.refresh_token_enc,
                client_id = COALESCE(excluded.client_id, broker_connections.client_id),
                agentic_account_id = COALESCE(excluded.agentic_account_id, broker_connections.agentic_account_id),
                token_expires_at = excluded.token_expires_at,
                metadata_json = COALESCE(excluded.metadata_json, broker_connections.metadata_json),
                updated_at = excluded.updated_at
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
        conn.commit()
        cursor.execute(
            "SELECT * FROM broker_connections WHERE user_id = ? AND broker = ?",
            (int(user_id), broker),
        )
        row = cursor.fetchone()
        conn.close()
        return _public_row(row)

    def update_tokens(
        self,
        user_id: int,
        *,
        broker: str = _BROKER,
        access_token: str,
        refresh_token: Optional[str] = None,
        token_expires_at: Optional[str] = None,
    ) -> None:
        now = _utcnow_iso()
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE broker_connections
            SET access_token_enc = ?, refresh_token_enc = COALESCE(?, refresh_token_enc),
                token_expires_at = ?, updated_at = ?
            WHERE user_id = ? AND broker = ?
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
        conn.commit()
        conn.close()

    def delete(self, user_id: int, broker: str = _BROKER) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM broker_connections WHERE user_id = ? AND broker = ?",
            (int(user_id), broker),
        )
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted


def _build_user_store():
    """Select the broker-connection backend from the environment.

    print(), not logger.info(): dashboard.backend.* loggers sit at WARNING in
    every real deployment, so an info() line would be invisible exactly where
    it matters. Name the target too -- "postgres" alone reads the same whether
    this is the intended Neon database or a typo'd/staging URL.
    """
    url = (os.getenv("USERS_DATABASE_URL") or os.getenv("CONTENT_DATABASE_URL") or "").strip()
    if url:
        # Imported inside the branch so environments without psycopg installed
        # still import this module and boot on SQLite.
        from dashboard.backend.domain.brokers.repository_postgres import (
            BrokerConnectionStorePostgres,
        )

        print(f"broker_connections backend: postgres ({describe_database_url(url)})")
        # Deliberately no try/except: a construction failure must abort boot.
        # Silently falling back to SQLite would park live-brokerage refresh
        # tokens on the disk-less Render free tier, where the file resets to the
        # committed seed database on every deploy -- exactly the failure this
        # twin exists to prevent, and one that reports itself as a healthy app.
        return BrokerConnectionStorePostgres(url)
    print(f"broker_connections backend: sqlite ({DB_PATH})")
    return BrokerConnectionStore()


broker_store = _build_user_store()
