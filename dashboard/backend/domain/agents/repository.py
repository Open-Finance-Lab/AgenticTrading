"""Registered external agents with persistent trading sessions and API keys.

Moved verbatim (Phase 3A1) from ``dashboard/backend/agent_store.py``; the
original module was removed in Phase 4A. Public classes, the ``agent_store``
singleton, SQL, return schemas, and behavior are unchanged; only the module
location moved.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dashboard.backend.database import DB_PATH
from dashboard.backend.db_url import describe_database_url
from dashboard.backend.domain.agents.runtime import DEFAULT_RUNTIME_TYPE

DEFAULT_SCOPES = "agents:register,runs:write,context:read,decisions:write,runs:read"

# Sentinel to distinguish "argument omitted" from an explicit ``None`` in
# update_agent (so a caller can clear the pipeline vs. leave it untouched).
_UNSET = object()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _new_api_key() -> str:
    return f"ag_{secrets.token_urlsafe(24)}"


def _parse_pipeline(raw: Any) -> Optional[List[Dict[str, Any]]]:
    """Decode the stored sub-agent pipeline JSON. Returns None if unset/invalid."""
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if isinstance(parsed, list):
        return parsed
    return None


def _parse_runtime_config(raw: Any) -> Dict[str, Any]:
    """Decode runtime JSON, defaulting legacy/invalid rows to an empty config."""
    if not raw:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _public_agent(row: sqlite3.Row | Dict[str, Any]) -> Dict[str, Any]:
    data = dict(row)
    raw_scopes = data.get("scopes") or DEFAULT_SCOPES
    return {
        "agent_id": data["agent_id"],
        "name": data["name"],
        "session_id": data["session_id"],
        "model_name": data.get("model_name") or "local-model",
        "agent_type": data.get("agent_type") or "external",
        "runtime_type": data.get("runtime_type") or DEFAULT_RUNTIME_TYPE,
        "runtime_config": _parse_runtime_config(data.get("runtime_config")),
        "description": data.get("description"),
        "pipeline": _parse_pipeline(data.get("pipeline_config")),
        "cash_allocation": data.get("cash_allocation"),
        "backtest_allocation": data.get("backtest_allocation"),
        "live_trading_enabled": bool(data.get("live_trading_enabled")),
        "category": data.get("category"),
        "api_key_prefix": data.get("api_key_prefix") or "",
        "owner_user_id": data.get("owner_user_id"),
        "scopes": [s for s in str(raw_scopes).split(",") if s],
        "created_at": data.get("created_at"),
        "last_used_at": data.get("last_used_at"),
    }


class AgentStore:
    """Persist external agents and their trading session IDs."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path or DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS external_agents (
                agent_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                session_id TEXT NOT NULL UNIQUE,
                api_key_hash TEXT NOT NULL UNIQUE,
                api_key_prefix TEXT NOT NULL,
                model_name TEXT NOT NULL DEFAULT 'local-model',
                scopes TEXT NOT NULL DEFAULT 'agents:register,runs:write,context:read,decisions:write,runs:read',
                owner_user_id INTEGER,
                owner_browser_session TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used_at TIMESTAMP,
                runtime_type TEXT NOT NULL DEFAULT 'pipeline',
                runtime_config TEXT NOT NULL DEFAULT '{}',
                category TEXT,
                FOREIGN KEY (owner_user_id) REFERENCES users(id) ON DELETE SET NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_external_agents_owner_user
            ON external_agents(owner_user_id)
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_external_agents_owner_browser
            ON external_agents(owner_browser_session)
            """
        )

        # Lightweight migrations: add columns introduced after the original
        # schema shipped. SQLite cannot ADD COLUMN IF NOT EXISTS, so probe first.
        cursor.execute("PRAGMA table_info(external_agents)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        if "agent_type" not in existing_columns:
            cursor.execute(
                "ALTER TABLE external_agents "
                "ADD COLUMN agent_type TEXT NOT NULL DEFAULT 'external'"
            )
        if "description" not in existing_columns:
            cursor.execute(
                "ALTER TABLE external_agents ADD COLUMN description TEXT"
            )
        if "pipeline_config" not in existing_columns:
            cursor.execute(
                "ALTER TABLE external_agents ADD COLUMN pipeline_config TEXT"
            )
        if "cash_allocation" not in existing_columns:
            cursor.execute(
                "ALTER TABLE external_agents ADD COLUMN cash_allocation REAL"
            )
        if "backtest_allocation" not in existing_columns:
            cursor.execute(
                "ALTER TABLE external_agents ADD COLUMN backtest_allocation REAL"
            )
        if "live_trading_enabled" not in existing_columns:
            cursor.execute(
                "ALTER TABLE external_agents ADD COLUMN live_trading_enabled INTEGER NOT NULL DEFAULT 0"
            )
        if "scopes" not in existing_columns:
            cursor.execute(
                "ALTER TABLE external_agents ADD COLUMN scopes TEXT "
                f"NOT NULL DEFAULT '{DEFAULT_SCOPES}'"
            )
        if "runtime_type" not in existing_columns:
            cursor.execute(
                "ALTER TABLE external_agents "
                "ADD COLUMN runtime_type TEXT NOT NULL DEFAULT 'pipeline'"
            )
        if "runtime_config" not in existing_columns:
            cursor.execute(
                "ALTER TABLE external_agents "
                "ADD COLUMN runtime_config TEXT NOT NULL DEFAULT '{}'"
            )
        if "category" not in existing_columns:
            cursor.execute(
                "ALTER TABLE external_agents ADD COLUMN category TEXT"
            )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_external_agents_type
            ON external_agents(agent_type)
            """
        )

        conn.commit()
        conn.close()

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
        runtime_type: str = DEFAULT_RUNTIME_TYPE,
        runtime_config: Optional[Dict[str, Any]] = None,
        cash_allocation: Optional[float] = None,
        backtest_allocation: Optional[float] = None,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        agent_id = f"agent_{uuid.uuid4().hex[:12]}"
        session_id = session_id or str(uuid.uuid4())
        api_key = _new_api_key()
        api_key_hash = _hash_api_key(api_key)
        api_key_prefix = api_key[:12]
        now = _utcnow_iso()

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO external_agents (
                agent_id, name, session_id, api_key_hash, api_key_prefix,
                model_name, agent_type, description, cash_allocation,
                backtest_allocation, runtime_type, runtime_config, category,
                owner_user_id, owner_browser_session, created_at, last_used_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                (runtime_type or DEFAULT_RUNTIME_TYPE).strip() or DEFAULT_RUNTIME_TYPE,
                json.dumps(runtime_config or {}),
                category,
                owner_user_id,
                owner_browser_session,
                now,
                now,
            ),
        )
        conn.commit()
        cursor.execute("SELECT * FROM external_agents WHERE agent_id = ?", (agent_id,))
        row = cursor.fetchone()
        conn.close()

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
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                # Same no-steal guard as claim_agent: this is the fourth writer
                # of owner_user_id and is reachable from
                # POST /agents/import-session, whose session_id comes straight
                # off a caller-supplied header. A bare COALESCE would let any
                # non-null caller id replace the existing owner.
                """
                UPDATE external_agents
                SET name = ?, model_name = ?, last_used_at = ?,
                    owner_user_id = CASE
                        WHEN owner_user_id IS NULL AND ? IS NOT NULL THEN ?
                        ELSE owner_user_id
                    END,
                    owner_browser_session = COALESCE(?, owner_browser_session)
                WHERE session_id = ?
                  AND (
                        owner_user_id IS NULL
                        OR (? IS NOT NULL AND owner_user_id = ?)
                      )
                """,
                (
                    name.strip(),
                    model_name.strip() or "local-model",
                    _utcnow_iso(),
                    owner_user_id,
                    owner_user_id,
                    owner_browser_session,
                    session_id,
                    owner_user_id,
                    owner_user_id,
                ),
            )
            conn.commit()
            conn.close()
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
        conn = self._get_connection()
        cursor = conn.cursor()
        rows: List[sqlite3.Row] = []
        seen: set = set()

        def _add_rows(query: str, params: tuple) -> None:
            cursor.execute(query, params)
            for row in cursor.fetchall():
                if row["agent_id"] not in seen:
                    seen.add(row["agent_id"])
                    rows.append(row)

        if owner_user_id is not None:
            _add_rows(
                """
                SELECT * FROM external_agents
                WHERE owner_user_id = ?
                ORDER BY created_at DESC
                """,
                (owner_user_id,),
            )
            # Also surface unclaimed agents created in this browser before
            # login/claim. Without this, a signed-in list drops the guest
            # "My Foundation Agent" whenever claim races or is skipped, while
            # logout (browser-only list) still shows it.
            if owner_browser_session:
                _add_rows(
                    """
                    SELECT * FROM external_agents
                    WHERE owner_browser_session = ?
                      AND owner_user_id IS NULL
                    ORDER BY created_at DESC
                    """,
                    (owner_browser_session,),
                )
        elif owner_browser_session:
            # Logged-out list: only *guest* agents for this browser. Account-
            # bound rows keep owner_browser_session stamped at create time; if
            # we matched on browser alone, logout would re-surface every agent
            # the previous signed-in user created on this machine.
            _add_rows(
                """
                SELECT * FROM external_agents
                WHERE owner_browser_session = ?
                  AND owner_user_id IS NULL
                ORDER BY created_at DESC
                """,
                (owner_browser_session,),
            )

        if trading_session_id:
            # session_id is not an ownership credential. Only fold in the
            # active trading session when it already belongs to this caller
            # (same user, or an unclaimed browser agent).
            if owner_user_id is not None:
                _add_rows(
                    """
                    SELECT * FROM external_agents
                    WHERE session_id = ?
                      AND owner_user_id = ?
                    ORDER BY created_at DESC
                    """,
                    (trading_session_id, owner_user_id),
                )
            elif owner_browser_session:
                # ``owner_browser_session = session_id`` is the import-session
                # shape (see api/dependencies.py::_owner_context): those rows
                # have no separate browser credential, so requiring the caller's
                # browser id to match would hide an agent the caller does own.
                _add_rows(
                    """
                    SELECT * FROM external_agents
                    WHERE session_id = ?
                      AND owner_user_id IS NULL
                      AND (
                            owner_browser_session = ?
                            OR owner_browser_session = session_id
                          )
                    ORDER BY created_at DESC
                    """,
                    (trading_session_id, owner_browser_session),
                )

        conn.close()
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
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM external_agents
            WHERE agent_type = 'builtin'
            ORDER BY created_at DESC
            """
        )
        rows = cursor.fetchall()
        conn.close()
        return [_public_agent(row) for row in rows]

    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM external_agents WHERE agent_id = ?", (agent_id,))
        row = cursor.fetchone()
        conn.close()
        return _public_agent(row) if row else None

    def get_agent_by_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM external_agents WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        conn.close()
        return _public_agent(row) if row else None

    def resolve_api_key(self, api_key: str, touch: bool = True) -> Optional[Dict[str, Any]]:
        if not api_key or not api_key.strip():
            return None
        key_hash = _hash_api_key(api_key.strip())
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM external_agents WHERE api_key_hash = ?",
            (key_hash,),
        )
        row = cursor.fetchone()
        if row and touch:
            cursor.execute(
                "UPDATE external_agents SET last_used_at = ? WHERE agent_id = ?",
                (_utcnow_iso(), row["agent_id"]),
            )
            conn.commit()
        conn.close()
        return _public_agent(row) if row else None

    def claim_browser_agents_to_user(
        self,
        browser_session: str,
        user_id: int,
    ) -> int:
        """Attach all browser-owned agents to a logged-in user account."""
        if not browser_session or user_id is None:
            return 0
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE external_agents
            SET owner_user_id = ?,
                owner_browser_session = COALESCE(owner_browser_session, ?),
                last_used_at = ?
            WHERE owner_browser_session = ?
              AND (owner_user_id IS NULL OR owner_user_id = ?)
            """,
            (user_id, browser_session, _utcnow_iso(), browser_session, user_id),
        )
        updated = cursor.rowcount
        conn.commit()
        conn.close()
        return updated

    def claim_agent(
        self,
        agent_id: str,
        *,
        owner_user_id: Optional[int] = None,
        owner_browser_session: Optional[str] = None,
    ) -> None:
        """Bind ownership without stealing another account's agent.

        ``owner_user_id`` is only written when the row is still unclaimed.
        ``owner_browser_session`` may refresh on rows this user already owns
        (or unclaimed rows). A different account's agent is a no-op.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE external_agents
            SET owner_user_id = CASE
                    WHEN owner_user_id IS NULL AND ? IS NOT NULL THEN ?
                    ELSE owner_user_id
                END,
                owner_browser_session = COALESCE(?, owner_browser_session),
                last_used_at = ?
            WHERE agent_id = ?
              AND (
                    owner_user_id IS NULL
                    OR (? IS NOT NULL AND owner_user_id = ?)
                  )
            """,
            (
                owner_user_id,
                owner_user_id,
                owner_browser_session,
                _utcnow_iso(),
                agent_id,
                owner_user_id,
                owner_user_id,
            ),
        )
        conn.commit()
        conn.close()

    def reclaim_agent(
        self,
        agent_id: str,
        *,
        owner_user_id: Optional[int] = None,
        owner_browser_session: Optional[str] = None,
    ) -> None:
        """Re-bind browser ownership for a guest or already-owned agent.

        Refuses to touch a row owned by a different user — that path used to
        let activate/restore steal agents across accounts on a shared browser.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE external_agents
            SET owner_user_id = CASE
                    WHEN owner_user_id IS NULL AND ? IS NOT NULL THEN ?
                    ELSE owner_user_id
                END,
                owner_browser_session = ?,
                last_used_at = ?
            WHERE agent_id = ?
              AND (
                    owner_user_id IS NULL
                    OR (? IS NOT NULL AND owner_user_id = ?)
                  )
            """,
            (
                owner_user_id,
                owner_user_id,
                owner_browser_session,
                _utcnow_iso(),
                agent_id,
                owner_user_id,
                owner_user_id,
            ),
        )
        conn.commit()
        conn.close()

    def rotate_api_key(self, agent_id: str) -> Optional[str]:
        """Issue a new API key for an agent. Returns the raw key once."""
        api_key = _new_api_key()
        api_key_hash = _hash_api_key(api_key)
        api_key_prefix = api_key[:12]
        now = _utcnow_iso()

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE external_agents
            SET api_key_hash = ?,
                api_key_prefix = ?,
                last_used_at = ?
            WHERE agent_id = ?
            """,
            (api_key_hash, api_key_prefix, now, agent_id),
        )
        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return api_key if updated else None

    def update_agent(
        self,
        agent_id: str,
        *,
        name: Optional[str] = None,
        model_name: Optional[str] = None,
        description: Optional[str] = None,
        pipeline: Any = _UNSET,
        runtime_type: Any = _UNSET,
        runtime_config: Any = _UNSET,
        cash_allocation: Any = _UNSET,
        backtest_allocation: Any = _UNSET,
        live_trading_enabled: Any = _UNSET,
        category: Any = _UNSET,
    ) -> Optional[Dict[str, Any]]:
        """Update display fields for an agent. Returns the updated record or None.

        ``pipeline`` uses a sentinel default so callers can distinguish "leave
        the stored pipeline untouched" (omit the arg) from "clear it" (pass
        ``None``). A list is serialized to JSON.
        """
        sets: list[str] = []
        params: list[Any] = []
        if name is not None:
            sets.append("name = ?")
            params.append(name.strip())
        if model_name is not None:
            sets.append("model_name = ?")
            params.append(model_name.strip())
        if description is not None:
            sets.append("description = ?")
            params.append(description.strip() if description else None)
        if pipeline is not _UNSET:
            sets.append("pipeline_config = ?")
            params.append(json.dumps(pipeline) if pipeline else None)
        if runtime_type is not _UNSET:
            sets.append("runtime_type = ?")
            params.append(
                (runtime_type or DEFAULT_RUNTIME_TYPE).strip() or DEFAULT_RUNTIME_TYPE
            )
        if runtime_config is not _UNSET:
            sets.append("runtime_config = ?")
            params.append(json.dumps(runtime_config or {}))
        if cash_allocation is not _UNSET:
            sets.append("cash_allocation = ?")
            params.append(cash_allocation)
        if backtest_allocation is not _UNSET:
            sets.append("backtest_allocation = ?")
            params.append(backtest_allocation)
        if live_trading_enabled is not _UNSET:
            sets.append("live_trading_enabled = ?")
            params.append(1 if live_trading_enabled else 0)
        if category is not _UNSET:
            sets.append("category = ?")
            params.append(category)
        if not sets:
            return self.get_agent(agent_id)
        sets.append("last_used_at = ?")
        params.append(_utcnow_iso())
        params.append(agent_id)
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE external_agents SET {', '.join(sets)} WHERE agent_id = ?",
            params,
        )
        updated = cursor.rowcount > 0
        conn.commit()
        if not updated:
            conn.close()
            return None
        cursor.execute("SELECT * FROM external_agents WHERE agent_id = ?", (agent_id,))
        row = cursor.fetchone()
        conn.close()
        return _public_agent(row) if row else None

    def delete_agent(self, agent_id: str) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM external_agents WHERE agent_id = ?", (agent_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted


    def count_agents(self) -> int:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS n FROM external_agents")
        row = cursor.fetchone()
        conn.close()
        return int(row["n"] if row else 0)

    def list_owner_scope_agent_ids(self, agent_id: str) -> List[str]:
        """Agent ids that share an owner with ``agent_id``.

        "Owner" is the account when the agent is claimed, and otherwise the
        browser session it was created under — the two identities the concurrent
        -backtest entitlement has to be able to bill, so that signing in cannot
        leave a user with a *smaller* budget than signing out (see
        ``domain/runs/service.resolve_owner_cap_context``).

        Resolved inside the store on purpose. ``_public_agent`` withholds
        ``owner_browser_session``, and that is not an oversight to route around:
        ``api/dependencies._owner_context`` accepts that value *as* an ownership
        credential, so a projection that returned it would hand one caller
        another's. Only ids come back out.

        An agent with neither owner is its own scope, so the caller still gets a
        budget rather than the ``[]`` that would read as "no agents".
        """
        if not agent_id:
            return []
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT owner_user_id, owner_browser_session FROM external_agents "
                "WHERE agent_id = ?",
                (agent_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return []
            if row["owner_user_id"] is not None:
                cursor.execute(
                    "SELECT agent_id FROM external_agents WHERE owner_user_id = ?",
                    (row["owner_user_id"],),
                )
            elif row["owner_browser_session"]:
                # owner_user_id IS NULL: a claimed agent is billed to its
                # account, never to the browser that happened to create it.
                cursor.execute(
                    "SELECT agent_id FROM external_agents "
                    "WHERE owner_browser_session = ? AND owner_user_id IS NULL",
                    (row["owner_browser_session"],),
                )
            else:
                return [agent_id]
            ids = [r["agent_id"] for r in cursor.fetchall() if r["agent_id"]]
        finally:
            conn.close()
        if agent_id not in ids:
            ids.append(agent_id)
        return ids

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
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT owner_user_id, owner_browser_session FROM external_agents WHERE agent_id = ?",
            (agent_id,),
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            return False
        bound_user = row["owner_user_id"]
        # Account-bound agents are only accessible to that account. Matching
        # owner_browser_session alone used to let any later login (or logout)
        # on the same browser GET/activate/claim another user's agents.
        if bound_user is not None:
            return owner_user_id is not None and bound_user == owner_user_id
        if owner_browser_session and row["owner_browser_session"] == owner_browser_session:
            return True
        # NOTE: session_id is NOT an ownership credential. It is an internal
        # trading-session identifier that is discoverable (it used to be returned
        # by the public /builtin listing), so matching it against a caller-supplied
        # session would let anyone who learned it take over the agent. Ownership
        # requires owner_user_id (for bound agents) or owner_browser_session (for
        # still-unclaimed guest agents), or the agent API key (route layer).
        return False


def _build_agent_store():
    database_url = os.getenv("CONTENT_DATABASE_URL")
    if database_url:
        from dashboard.backend.domain.agents.repository_postgres import PostgresAgentStore

        # print(), not logger.info() -- info is invisible under the prod logging
        # config. See users.py's _build_user_store for the full rationale.
        print(f"agent_store backend: postgres ({describe_database_url(database_url)})")
        return PostgresAgentStore(database_url)
    print("agent_store backend: sqlite (ephemeral on Render)")
    return AgentStore()


agent_store = _build_agent_store()
