"""Postgres-backed StrategyStore implementation.

Selected instead of the default SQLite StrategyStore when CONTENT_DATABASE_URL is set
(see repository.py's _build_strategy_store). Method surface and behavior are
identical to StrategyStore, with one structural difference: the share-code
retry loop uses ``INSERT ... ON CONFLICT (code) DO NOTHING`` and checks
``cursor.rowcount`` instead of catching IntegrityError. In Postgres a
UniqueViolation aborts the whole transaction (every later statement on the
connection raises InFailedSqlTransaction), so SQLite's catch-and-retry on one
connection cannot be ported literally -- it would 500 on the first real
collision. Retry count, code-space widening, and the RuntimeError fallback
are preserved exactly.
"""

from __future__ import annotations

import secrets
from typing import Any, Optional

from dashboard.backend.db_url import require_postgres_url
from dashboard.backend.domain.strategies.repository import (
    _CODE_LENGTH,
    _now_iso,
    _public,
)


class PostgresStrategyStore:
    """Persist free-form strategy prompts, backed by Postgres."""

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
        # domain/agents/repository_postgres.py for why nothing would catch the
        # omission.
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS strategies (
                        code TEXT PRIMARY KEY,
                        prompt TEXT NOT NULL,
                        description TEXT,
                        source TEXT,
                        owner TEXT,
                        created_at TEXT,
                        last_run_id TEXT,
                        last_run_at TEXT
                    )
                    """
                )

    def create(
        self,
        *,
        prompt: str,
        description: Optional[str] = None,
        source: Optional[str] = None,
        owner: Optional[str] = None,
    ) -> dict[str, Any]:
        cleaned = (prompt or "").strip()
        if not cleaned:
            raise ValueError("Strategy prompt cannot be empty.")
        desc = (description or "").strip() or None
        now = _now_iso()
        with self._get_connection() as conn:
            with conn.cursor() as cur:

                def _insert(candidate: str) -> bool:
                    # ON CONFLICT DO NOTHING + rowcount instead of catching
                    # UniqueViolation: an aborted transaction would poison the
                    # connection for every retry (see module docstring).
                    cur.execute(
                        "INSERT INTO strategies "
                        "(code, prompt, description, source, owner, created_at, last_run_id, last_run_at) "
                        "VALUES (%s, %s, %s, %s, %s, %s, NULL, NULL) "
                        "ON CONFLICT (code) DO NOTHING",
                        (candidate, cleaned, desc, source or None, owner or None, now),
                    )
                    return cur.rowcount == 1

                code = None
                for _ in range(20):
                    candidate = secrets.token_hex(_CODE_LENGTH // 2)
                    if _insert(candidate):
                        code = candidate
                        break
                if code is None:
                    # Astronomically unlikely: widen the code space (64 bits) and
                    # try once more, matching the SQLite store's graceful fallback
                    # rather than surfacing a 500.
                    candidate = secrets.token_hex(_CODE_LENGTH)
                    if not _insert(candidate):
                        raise RuntimeError("Could not allocate a unique strategy code")
                    code = candidate
                cur.execute("SELECT * FROM strategies WHERE code = %s", (code,))
                row = cur.fetchone()
                return _public(row)

    def get(self, code: str) -> Optional[dict[str, Any]]:
        if not code:
            return None
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM strategies WHERE code = %s", (code,))
                row = cur.fetchone()
        return _public(row) if row else None

    def set_last_run(self, code: str, run_id: str) -> Optional[dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE strategies SET last_run_id = %s, last_run_at = %s "
                    "WHERE code = %s RETURNING *",
                    (run_id, _now_iso(), code),
                )
                row = cur.fetchone()
        return _public(row) if row else None
