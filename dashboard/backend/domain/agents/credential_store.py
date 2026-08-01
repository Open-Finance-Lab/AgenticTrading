"""Encrypted, per-agent credentials for hosted runtimes.

Secrets live outside ``runtime_config`` so agent JSON, marketplace templates,
run metadata, and temporary strategy files can never contain them. Routes only
expose configured status; plaintext is returned solely to the trusted backtest
launcher immediately before it builds the subprocess environment.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from dashboard.backend.database import DB_PATH
from dashboard.backend.db_url import describe_database_url
from dashboard.backend.domain.agents.repository import _utcnow_iso
from dashboard.backend.domain.brokers.repository import _decrypt, _encrypt

FINANCIAL_DATASETS_CREDENTIAL = "financial_datasets_api_key"


def _decrypt_value(value: str) -> str:
    from cryptography.fernet import InvalidToken

    try:
        return _decrypt(value)
    except InvalidToken as exc:
        raise RuntimeError(
            "Stored agent credential could not be decrypted with the configured key"
        ) from exc


def _public_row(row: sqlite3.Row | Dict[str, Any]) -> Dict[str, Any]:
    data = dict(row)
    return {
        "credential": data["credential_name"],
        "configured": True,
        "updated_at": data.get("updated_at"),
    }


class AgentCredentialStore:
    """Persist encrypted hosted-runtime credentials in SQLite."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path or DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        conn = self._get_connection()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_credentials (
                agent_id TEXT NOT NULL,
                credential_name TEXT NOT NULL,
                value_enc TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (agent_id, credential_name)
            )
            """
        )
        conn.commit()
        conn.close()

    def get_public(
        self, agent_id: str, credential_name: str
    ) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM agent_credentials WHERE agent_id = ? AND credential_name = ?",
            (agent_id, credential_name),
        ).fetchone()
        conn.close()
        return _public_row(row) if row else None

    def get_secret(self, agent_id: str, credential_name: str) -> Optional[str]:
        conn = self._get_connection()
        row = conn.execute(
            "SELECT value_enc FROM agent_credentials "
            "WHERE agent_id = ? AND credential_name = ?",
            (agent_id, credential_name),
        ).fetchone()
        conn.close()
        return _decrypt_value(row["value_enc"]) if row else None

    def upsert(
        self, agent_id: str, credential_name: str, value: str
    ) -> Dict[str, Any]:
        now = _utcnow_iso()
        encrypted = _encrypt(value)
        conn = self._get_connection()
        conn.execute(
            """
            INSERT INTO agent_credentials (
                agent_id, credential_name, value_enc, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(agent_id, credential_name) DO UPDATE SET
                value_enc = excluded.value_enc,
                updated_at = excluded.updated_at
            """,
            (agent_id, credential_name, encrypted, now, now),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM agent_credentials WHERE agent_id = ? AND credential_name = ?",
            (agent_id, credential_name),
        ).fetchone()
        conn.close()
        return _public_row(row)

    def delete(self, agent_id: str, credential_name: str) -> bool:
        conn = self._get_connection()
        cursor = conn.execute(
            "DELETE FROM agent_credentials WHERE agent_id = ? AND credential_name = ?",
            (agent_id, credential_name),
        )
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted

    def delete_all(self, agent_id: str) -> bool:
        conn = self._get_connection()
        cursor = conn.execute(
            "DELETE FROM agent_credentials WHERE agent_id = ?", (agent_id,)
        )
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted


def _build_agent_credential_store():
    database_url = (os.getenv("CONTENT_DATABASE_URL") or "").strip()
    if database_url:
        from dashboard.backend.domain.agents.credential_store_postgres import (
            PostgresAgentCredentialStore,
        )

        print(
            "agent_credential_store backend: postgres "
            f"({describe_database_url(database_url)})"
        )
        return PostgresAgentCredentialStore(database_url)
    print("agent_credential_store backend: sqlite (ephemeral on Render)")
    return AgentCredentialStore()


agent_credential_store = _build_agent_credential_store()
