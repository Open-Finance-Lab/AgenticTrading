"""Postgres twin for encrypted hosted-runtime agent credentials."""

from __future__ import annotations

from typing import Any, Dict, Optional

from dashboard.backend.db_url import require_postgres_url
from dashboard.backend.domain.agents.credential_store import _decrypt_value, _public_row
from dashboard.backend.domain.agents.repository import _utcnow_iso
from dashboard.backend.domain.brokers.repository import _encrypt


class PostgresAgentCredentialStore:
    """Persist encrypted hosted-runtime credentials in Postgres."""

    def __init__(self, database_url: str):
        self.database_url = require_postgres_url(database_url)
        self._init_schema()

    def _get_connection(self):
        from dashboard.backend.db_pool import get_pool

        return get_pool(self.database_url).connection()

    def _init_schema(self) -> None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
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

    def get_public(
        self, agent_id: str, credential_name: str
    ) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM agent_credentials "
                    "WHERE agent_id = %s AND credential_name = %s",
                    (agent_id, credential_name),
                )
                row = cur.fetchone()
        return _public_row(row) if row else None

    def get_secret(self, agent_id: str, credential_name: str) -> Optional[str]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT value_enc FROM agent_credentials "
                    "WHERE agent_id = %s AND credential_name = %s",
                    (agent_id, credential_name),
                )
                row = cur.fetchone()
        return _decrypt_value(row["value_enc"]) if row else None

    def upsert(
        self, agent_id: str, credential_name: str, value: str
    ) -> Dict[str, Any]:
        now = _utcnow_iso()
        encrypted = _encrypt(value)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO agent_credentials (
                        agent_id, credential_name, value_enc, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (agent_id, credential_name) DO UPDATE SET
                        value_enc = EXCLUDED.value_enc,
                        updated_at = EXCLUDED.updated_at
                    RETURNING *
                    """,
                    (agent_id, credential_name, encrypted, now, now),
                )
                row = cur.fetchone()
        return _public_row(row)

    def delete(self, agent_id: str, credential_name: str) -> bool:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM agent_credentials "
                    "WHERE agent_id = %s AND credential_name = %s",
                    (agent_id, credential_name),
                )
                deleted = cur.rowcount > 0
        return deleted

    def delete_all(self, agent_id: str) -> bool:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM agent_credentials WHERE agent_id = %s",
                    (agent_id,),
                )
                deleted = cur.rowcount > 0
        return deleted
