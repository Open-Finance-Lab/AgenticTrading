"""Postgres-backed AgentStore implementation.

Selected instead of the default SQLite AgentStore when CONTENT_DATABASE_URL is set
(see repository.py's _build_agent_store). Exists because the SQLite store
lives in DATABASE_PATH, which resets to the committed seed database on every
deploy of the disk-less Render free-tier host -- silently deleting every
registered agent and invalidating every issued API key (resolve_api_key is
the sole auth path for /api/v1 and /api/v2). Method surface, return schemas,
and behavior are identical to AgentStore; only the SQL dialect differs.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from dashboard.backend.db_url import require_postgres_url
from dashboard.backend.domain.agents.repository import (
    DEFAULT_SCOPES,
    _UNSET,
    _hash_api_key,
    _new_api_key,
    _public_agent,
    _utcnow_iso,
)


class PostgresAgentStore:
    """Persist external agents and their trading session IDs, backed by Postgres."""

    def __init__(self, database_url: str):
        self.database_url = require_postgres_url(database_url)
        self._init_schema()

    def _get_connection(self):
        # Pooled checkout: same context-manager transaction semantics as
        # psycopg.connect (commit on clean exit), returned to the pool on close.
        from dashboard.backend.db_pool import get_pool
        return get_pool(self.database_url).connection()

    def _init_schema(self) -> None:
        # ADDING A COLUMN LATER? It must go in an `ALTER TABLE ... ADD COLUMN IF
        # NOT EXISTS` below, *not* only in the CREATE above. CREATE TABLE IF NOT
        # EXISTS silently no-ops once the table exists, so an existing
        # deployment would never gain the column, and every query naming it
        # would raise UndefinedColumn -- 500ing this whole surface while /health
        # stays green. Nothing catches it first: the SQLite tier is the default
        # in tests, and CI's Postgres service container is empty on every run,
        # so the @pg_only tier only ever exercises the CREATE path -- except
        # test_agent_schema_lazily_migrates_an_old_table_postgres, which recreates
        # the pre-migration table to force the migrate path. The six ALTER ...
        # ADD COLUMN IF NOT EXISTS statements below re-add the columns the SQLite
        # store accreted as lazy migrations, so a deployment whose table predates
        # them is brought up to shape on the next boot; on a fresh or current
        # table they no-op. Add the next column the same way (Postgres supports
        # ADD COLUMN IF NOT EXISTS natively; users_postgres.py has the pattern).
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # owner_user_id is deliberately a plain INTEGER with no FK to
                # users(id): SQLite never enforced the declared FK (no PRAGMA
                # foreign_keys anywhere), the app owns this integrity
                # (owns_agent / claim_browser_agents_to_user), and a FK would
                # break the split config where USERS_DATABASE_URL points at a
                # different database (Postgres has no cross-database FKs).
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS external_agents (
                        agent_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        session_id TEXT NOT NULL UNIQUE,
                        api_key_hash TEXT NOT NULL UNIQUE,
                        api_key_prefix TEXT NOT NULL,
                        model_name TEXT NOT NULL DEFAULT 'local-model',
                        scopes TEXT NOT NULL DEFAULT '{DEFAULT_SCOPES}',
                        owner_user_id INTEGER,
                        owner_browser_session TEXT,
                        created_at TEXT,
                        last_used_at TEXT,
                        agent_type TEXT NOT NULL DEFAULT 'external',
                        description TEXT,
                        pipeline_config TEXT,
                        cash_allocation DOUBLE PRECISION,
                        backtest_allocation DOUBLE PRECISION,
                        live_trading_enabled BOOLEAN NOT NULL DEFAULT FALSE
                    )
                    """
                )
                # Lazy migrations for a table created before these columns
                # existed (mirrors the SQLite store's five post-ship ALTERs).
                # Each no-ops on a fresh or already-current table. Must run
                # before idx_external_agents_type, which indexes agent_type.
                cur.execute(
                    "ALTER TABLE external_agents "
                    "ADD COLUMN IF NOT EXISTS agent_type TEXT NOT NULL DEFAULT 'external'"
                )
                cur.execute(
                    "ALTER TABLE external_agents ADD COLUMN IF NOT EXISTS description TEXT"
                )
                cur.execute(
                    "ALTER TABLE external_agents "
                    "ADD COLUMN IF NOT EXISTS pipeline_config TEXT"
                )
                cur.execute(
                    "ALTER TABLE external_agents "
                    "ADD COLUMN IF NOT EXISTS cash_allocation DOUBLE PRECISION"
                )
                cur.execute(
                    "ALTER TABLE external_agents "
                    "ADD COLUMN IF NOT EXISTS backtest_allocation DOUBLE PRECISION"
                )
                cur.execute(
                    "ALTER TABLE external_agents "
                    f"ADD COLUMN IF NOT EXISTS scopes TEXT NOT NULL DEFAULT '{DEFAULT_SCOPES}'"
                )
                cur.execute(
                    "ALTER TABLE external_agents "
                    "ADD COLUMN IF NOT EXISTS live_trading_enabled BOOLEAN NOT NULL DEFAULT FALSE"
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_external_agents_owner_user
                    ON external_agents(owner_user_id)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_external_agents_owner_browser
                    ON external_agents(owner_browser_session)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_external_agents_type
                    ON external_agents(agent_type)
                    """
                )

    def create_agent(
        self,
        *,
        name: str,
        model_name: str = "local-model",
        owner_user_id: Optional[int] = None,
        owner_browser_session: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_type: str = "external",
        description: Optional[str] = None,
        cash_allocation: Optional[float] = None,
        backtest_allocation: Optional[float] = None,
    ) -> Dict[str, Any]:
        agent_id = f"agent_{uuid.uuid4().hex[:12]}"
        session_id = session_id or str(uuid.uuid4())
        api_key = _new_api_key()
        api_key_hash = _hash_api_key(api_key)
        api_key_prefix = api_key[:12]
        now = _utcnow_iso()

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO external_agents (
                        agent_id, name, session_id, api_key_hash, api_key_prefix,
                        model_name, agent_type, description, cash_allocation,
                        backtest_allocation,
                        owner_user_id, owner_browser_session, created_at, last_used_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING *
                    """,
                    (
                        agent_id,
                        name.strip(),
                        session_id,
                        api_key_hash,
                        api_key_prefix,
                        model_name.strip() or "local-model",
                        (agent_type or "external").strip() or "external",
                        (description or None),
                        cash_allocation,
                        backtest_allocation,
                        owner_user_id,
                        owner_browser_session,
                        now,
                        now,
                    ),
                )
                row = cur.fetchone()

        agent = _public_agent(row)
        agent["api_key"] = api_key
        return agent

    def register_or_get_agent(
        self,
        *,
        session_id: str,
        name: str,
        model_name: str = "local-model",
        owner_user_id: Optional[int] = None,
        owner_browser_session: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Link an existing trading session to an agent (idempotent)."""
        existing = self.get_agent_by_session(session_id)
        if existing:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE external_agents
                        SET name = %s, model_name = %s, last_used_at = %s,
                            owner_user_id = COALESCE(%s, owner_user_id),
                            owner_browser_session = COALESCE(%s, owner_browser_session)
                        WHERE session_id = %s
                        """,
                        (
                            name.strip(),
                            model_name.strip() or "local-model",
                            _utcnow_iso(),
                            owner_user_id,
                            owner_browser_session,
                            session_id,
                        ),
                    )
            return self.get_agent_by_session(session_id) or existing

        return self.create_agent(
            name=name,
            model_name=model_name,
            owner_user_id=owner_user_id,
            owner_browser_session=owner_browser_session,
            session_id=session_id,
        )

    def list_agents(
        self,
        *,
        owner_user_id: Optional[int] = None,
        owner_browser_session: Optional[str] = None,
        trading_session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        seen: set = set()

        with self._get_connection() as conn:
            with conn.cursor() as cur:

                def _add_rows(query: str, params: tuple) -> None:
                    cur.execute(query, params)
                    for row in cur.fetchall():
                        if row["agent_id"] not in seen:
                            seen.add(row["agent_id"])
                            rows.append(row)

                if owner_user_id is not None:
                    _add_rows(
                        """
                        SELECT * FROM external_agents
                        WHERE owner_user_id = %s
                        ORDER BY created_at DESC
                        """,
                        (owner_user_id,),
                    )
                    # Also surface unclaimed agents created in this browser
                    # before login/claim (same rationale as the SQLite twin).
                    if owner_browser_session:
                        _add_rows(
                            """
                            SELECT * FROM external_agents
                            WHERE owner_browser_session = %s
                              AND owner_user_id IS NULL
                            ORDER BY created_at DESC
                            """,
                            (owner_browser_session,),
                        )
                elif owner_browser_session:
                    _add_rows(
                        """
                        SELECT * FROM external_agents
                        WHERE owner_browser_session = %s
                        ORDER BY created_at DESC
                        """,
                        (owner_browser_session,),
                    )

                if trading_session_id:
                    _add_rows(
                        """
                        SELECT * FROM external_agents
                        WHERE session_id = %s
                        ORDER BY created_at DESC
                        """,
                        (trading_session_id,),
                    )

        # Each _add_rows call is independently sorted, but the groups are
        # appended query-by-query, so a recent unclaimed browser agent could
        # otherwise land after an older owned agent. Re-sort the union once.
        rows.sort(key=lambda row: row["created_at"], reverse=True)
        return [_public_agent(row) for row in rows]

    def list_builtin_agents(self) -> List[Dict[str, Any]]:
        """List every built-in (platform-hosted) agent, newest first.

        Built-in agents are globally discoverable (e.g. from the Discord
        ``/agent`` command) regardless of which account created them.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM external_agents
                    WHERE agent_type = 'builtin'
                    ORDER BY created_at DESC
                    """
                )
                rows = cur.fetchall()
        return [_public_agent(row) for row in rows]

    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM external_agents WHERE agent_id = %s", (agent_id,)
                )
                row = cur.fetchone()
        return _public_agent(row) if row else None

    def get_agent_by_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM external_agents WHERE session_id = %s", (session_id,)
                )
                row = cur.fetchone()
        return _public_agent(row) if row else None

    def resolve_api_key(self, api_key: str, touch: bool = True) -> Optional[Dict[str, Any]]:
        if not api_key or not api_key.strip():
            return None
        key_hash = _hash_api_key(api_key.strip())
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM external_agents WHERE api_key_hash = %s",
                    (key_hash,),
                )
                row = cur.fetchone()
                if row and touch:
                    cur.execute(
                        "UPDATE external_agents SET last_used_at = %s WHERE agent_id = %s",
                        (_utcnow_iso(), row["agent_id"]),
                    )
        return _public_agent(row) if row else None

    def claim_browser_agents_to_user(
        self,
        browser_session: str,
        user_id: int,
    ) -> int:
        """Attach all browser-owned agents to a logged-in user account."""
        if not browser_session or user_id is None:
            return 0
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE external_agents
                    SET owner_user_id = %s,
                        owner_browser_session = COALESCE(owner_browser_session, %s),
                        last_used_at = %s
                    WHERE owner_browser_session = %s
                      AND (owner_user_id IS NULL OR owner_user_id = %s)
                    """,
                    (user_id, browser_session, _utcnow_iso(), browser_session, user_id),
                )
                updated = cur.rowcount
        return updated

    def claim_agent(
        self,
        agent_id: str,
        *,
        owner_user_id: Optional[int] = None,
        owner_browser_session: Optional[str] = None,
    ) -> None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE external_agents
                    SET owner_user_id = COALESCE(%s, owner_user_id),
                        owner_browser_session = COALESCE(%s, owner_browser_session),
                        last_used_at = %s
                    WHERE agent_id = %s
                    """,
                    (owner_user_id, owner_browser_session, _utcnow_iso(), agent_id),
                )

    def reclaim_agent(
        self,
        agent_id: str,
        *,
        owner_user_id: Optional[int] = None,
        owner_browser_session: Optional[str] = None,
    ) -> None:
        """Re-bind an agent to the current browser/user (dashboard session proof)."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE external_agents
                    SET owner_user_id = COALESCE(%s, owner_user_id),
                        owner_browser_session = %s,
                        last_used_at = %s
                    WHERE agent_id = %s
                    """,
                    (owner_user_id, owner_browser_session, _utcnow_iso(), agent_id),
                )

    def rotate_api_key(self, agent_id: str) -> Optional[str]:
        """Issue a new API key for an agent. Returns the raw key once."""
        api_key = _new_api_key()
        api_key_hash = _hash_api_key(api_key)
        api_key_prefix = api_key[:12]
        now = _utcnow_iso()

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE external_agents
                    SET api_key_hash = %s,
                        api_key_prefix = %s,
                        last_used_at = %s
                    WHERE agent_id = %s
                    """,
                    (api_key_hash, api_key_prefix, now, agent_id),
                )
                updated = cur.rowcount > 0
        return api_key if updated else None

    def update_agent(
        self,
        agent_id: str,
        *,
        name: Optional[str] = None,
        model_name: Optional[str] = None,
        description: Optional[str] = None,
        pipeline: Any = _UNSET,
        cash_allocation: Any = _UNSET,
        backtest_allocation: Any = _UNSET,
        live_trading_enabled: Any = _UNSET,
    ) -> Optional[Dict[str, Any]]:
        """Update display fields for an agent. Returns the updated record or None.

        ``pipeline`` uses a sentinel default so callers can distinguish "leave
        the stored pipeline untouched" (omit the arg) from "clear it" (pass
        ``None``). A list is serialized to JSON.
        """
        sets: list[str] = []
        params: list[Any] = []
        if name is not None:
            sets.append("name = %s")
            params.append(name.strip())
        if model_name is not None:
            sets.append("model_name = %s")
            params.append(model_name.strip())
        if description is not None:
            sets.append("description = %s")
            params.append(description.strip() if description else None)
        if pipeline is not _UNSET:
            sets.append("pipeline_config = %s")
            params.append(json.dumps(pipeline) if pipeline else None)
        if cash_allocation is not _UNSET:
            sets.append("cash_allocation = %s")
            params.append(cash_allocation)
        if backtest_allocation is not _UNSET:
            sets.append("backtest_allocation = %s")
            params.append(backtest_allocation)
        if live_trading_enabled is not _UNSET:
            sets.append("live_trading_enabled = %s")
            params.append(bool(live_trading_enabled))
        if not sets:
            return self.get_agent(agent_id)
        sets.append("last_used_at = %s")
        params.append(_utcnow_iso())
        params.append(agent_id)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE external_agents SET {', '.join(sets)} "
                    "WHERE agent_id = %s RETURNING *",
                    params,
                )
                row = cur.fetchone()
        return _public_agent(row) if row else None

    def delete_agent(self, agent_id: str) -> bool:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM external_agents WHERE agent_id = %s", (agent_id,)
                )
                deleted = cur.rowcount > 0
        return deleted

    def owns_agent(
        self,
        agent: Dict[str, Any],
        *,
        owner_user_id: Optional[int] = None,
        owner_browser_session: Optional[str] = None,
    ) -> bool:
        agent_id = agent.get("agent_id") if isinstance(agent, dict) else agent
        if not agent_id:
            return False
        # Read the ownership columns straight from the row. owner_browser_session
        # is deliberately omitted from the public agent dict (it is a private
        # credential), so we must NOT rely on the passed-in dict for it — doing so
        # is why owner_browser_session ownership silently never matched.
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT owner_user_id, owner_browser_session "
                    "FROM external_agents WHERE agent_id = %s",
                    (agent_id,),
                )
                row = cur.fetchone()
        if not row:
            return False
        if owner_user_id is not None and row["owner_user_id"] == owner_user_id:
            return True
        if owner_browser_session and row["owner_browser_session"] == owner_browser_session:
            return True
        # NOTE: session_id is NOT an ownership credential. It is an internal
        # trading-session identifier that is discoverable (it used to be returned
        # by the public /builtin listing), so matching it against a caller-supplied
        # session would let anyone who learned it take over the agent. Ownership
        # requires owner_user_id or owner_browser_session (a real, private
        # credential), or the agent API key (checked at the route layer).
        return False
