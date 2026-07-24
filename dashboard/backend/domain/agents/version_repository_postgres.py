"""Postgres-backed AgentVersionStore implementation.

Selected instead of the default SQLite AgentVersionStore when CONTENT_DATABASE_URL is
set (see version_repository.py's _build_agent_version_store). Versions are
immutable reproducibility snapshots and were previously lost with their agents
on every deploy of the disk-less Render free-tier host. Method surface, return
schemas, and behavior are identical to AgentVersionStore; only the SQL dialect
differs.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from dashboard.backend.db_url import require_postgres_url
from dashboard.backend.domain.agents.version_repository import (
    _new_version_id,
    _public_version,
    _short_hash,
    _utcnow_iso,
)


class PostgresAgentVersionStore:
    """Persist immutable agent version snapshots, backed by Postgres."""

    def __init__(self, database_url: str):
        self.database_url = require_postgres_url(database_url)
        self._init_schema()

    def _get_connection(self):
        # Pooled checkout: same context-manager transaction semantics as
        # psycopg.connect (commit on clean exit), returned to the pool on close.
        from dashboard.backend.db_pool import get_pool
        return get_pool(self.database_url).connection()

    def _init_schema(self) -> None:
        # Adding a column later requires an `ALTER TABLE ... ADD COLUMN IF NOT
        # EXISTS` below, not just an edit to the CREATE -- see the same note in
        # repository_postgres.py for why nothing would catch the omission.
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_versions (
                        agent_version_id TEXT PRIMARY KEY,
                        agent_id TEXT NOT NULL,
                        version TEXT NOT NULL,
                        execution_mode TEXT NOT NULL DEFAULT 'external',
                        architecture TEXT,
                        model_backbones TEXT NOT NULL DEFAULT '[]',
                        decision_frequency TEXT NOT NULL DEFAULT '1h',
                        code_commit TEXT,
                        prompt_hash TEXT,
                        config_hash TEXT,
                        verification_level TEXT NOT NULL DEFAULT 'self_reported',
                        created_at TEXT
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_agent_versions_agent
                    ON agent_versions(agent_id, created_at)
                    """
                )

    def create_version(
        self,
        *,
        agent_id: str,
        version: str,
        execution_mode: str = "external",
        architecture: Optional[str] = None,
        model_backbones: Optional[List[str]] = None,
        decision_frequency: str = "1h",
        code_commit: Optional[str] = None,
        prompt_hash: Optional[str] = None,
        config_hash: Optional[str] = None,
        prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        verification_level: str = "self_reported",
    ) -> Dict[str, Any]:
        """Create a new immutable version. Hashes can be derived from raw
        prompt/config if explicit hashes are not provided."""
        agent_version_id = _new_version_id()
        now = _utcnow_iso()
        backbones = json.dumps(list(model_backbones or []))
        prompt_hash = prompt_hash or _short_hash(prompt)
        config_hash = config_hash or _short_hash(config)

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO agent_versions (
                        agent_version_id, agent_id, version, execution_mode, architecture,
                        model_backbones, decision_frequency, code_commit, prompt_hash,
                        config_hash, verification_level, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING *
                    """,
                    (
                        agent_version_id,
                        agent_id,
                        version,
                        execution_mode,
                        architecture,
                        backbones,
                        decision_frequency,
                        code_commit,
                        prompt_hash,
                        config_hash,
                        verification_level,
                        now,
                    ),
                )
                row = cur.fetchone()
        return _public_version(row)

    def get_version(self, agent_version_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM agent_versions WHERE agent_version_id = %s",
                    (agent_version_id,),
                )
                row = cur.fetchone()
        return _public_version(row) if row else None

    def list_versions(self, agent_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM agent_versions
                    WHERE agent_id = %s
                    ORDER BY created_at DESC, agent_version_id DESC
                    """,
                    (agent_id,),
                )
                rows = cur.fetchall()
        return [_public_version(row) for row in rows]
