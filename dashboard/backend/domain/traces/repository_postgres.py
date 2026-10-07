"""Postgres twin for the ordered Agent trace store."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from dashboard.backend.db_url import init_schema_unless_worker, require_postgres_url
from dashboard.backend.domain.traces.repository import (
    _json_text,
    _new_event_id,
    _public_event,
    _public_trace,
    _reject_sensitive,
    _utcnow_iso,
)


class PostgresTraceStore:
    def __init__(self, database_url: str):
        self.database_url = require_postgres_url(database_url)
        init_schema_unless_worker("trace_store", self._init_schema)

    def _get_connection(self):
        from dashboard.backend.db_pool import get_pool

        return get_pool(self.database_url).connection()

    def _init_schema(self) -> None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT status FROM agent_traces WHERE trace_id = %s", (trace_id,))
                current = cur.fetchone()
                if current and current["status"] in {"completed", "failed"}:
                    if status and status != current["status"]:
                        status = None
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_traces (
                        trace_id TEXT PRIMARY KEY,
                        agent_id TEXT,
                        agent_version_id TEXT,
                        run_id TEXT NOT NULL UNIQUE,
                        user_id INTEGER,
                        trace_kind TEXT NOT NULL DEFAULT 'trading_run',
                        status TEXT NOT NULL DEFAULT 'running',
                        initial_input_json TEXT NOT NULL DEFAULT '{}',
                        final_output_summary TEXT,
                        started_at TEXT NOT NULL,
                        ended_at TEXT,
                        created_at TEXT NOT NULL,
                        parent_trace_id TEXT
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_trace_events (
                        event_id TEXT PRIMARY KEY,
                        trace_id TEXT NOT NULL REFERENCES agent_traces(trace_id),
                        sequence_no INTEGER NOT NULL,
                        event_type TEXT NOT NULL,
                        actor_type TEXT NOT NULL,
                        actor_id TEXT,
                        step_id TEXT,
                        decision_id TEXT,
                        artifact_id TEXT,
                        parent_event_id TEXT,
                        payload_json TEXT NOT NULL DEFAULT '{}',
                        occurred_at TEXT NOT NULL,
                        ingested_at TEXT NOT NULL,
                        idempotency_key TEXT,
                        schema_version INTEGER NOT NULL DEFAULT 1,
                        UNIQUE (trace_id, sequence_no),
                        UNIQUE (trace_id, idempotency_key)
                    )
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_trace_events_trace_sequence "
                    "ON agent_trace_events(trace_id, sequence_no)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_traces_agent_created "
                    "ON agent_traces(agent_id, created_at DESC)"
                )
                cur.execute("ALTER TABLE agent_traces ADD COLUMN IF NOT EXISTS parent_trace_id TEXT")
                cur.execute("ALTER TABLE agent_trace_events ADD COLUMN IF NOT EXISTS parent_event_id TEXT")

    def create_trace(
        self,
        *,
        trace_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_version_id: Optional[str] = None,
        run_id: str,
        user_id: Optional[int] = None,
        trace_kind: str = "trading_run",
        initial_input: Optional[Dict[str, Any]] = None,
        started_at: Optional[str] = None,
        parent_trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        import uuid

        trace_id = trace_id or f"trace_{uuid.uuid4().hex[:16]}"
        started_at = started_at or _utcnow_iso()
        initial_input = initial_input or {}
        _reject_sensitive(initial_input, "initial_input")
        input_json = _json_text(initial_input, field="initial_input")
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO agent_traces (
                        trace_id, agent_id, agent_version_id, run_id, user_id,
                        trace_kind, status, initial_input_json, started_at, created_at, parent_trace_id
                    ) VALUES (%s, %s, %s, %s, %s, %s, 'running', %s, %s, %s, %s)
                    RETURNING *
                    """,
                    (trace_id, agent_id, agent_version_id, run_id, user_id,
                     trace_kind, input_json, started_at, _utcnow_iso(), parent_trace_id),
                )
                row = cur.fetchone()
        return _public_trace(row)

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM agent_traces WHERE trace_id = %s", (trace_id,))
                row = cur.fetchone()
        return _public_trace(row) if row else None

    def get_trace_for_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM agent_traces WHERE run_id = %s", (run_id,))
                row = cur.fetchone()
        return _public_trace(row) if row else None

    def list_traces(
        self,
        *,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        limit = max(1, min(int(limit), 100))
        offset = max(0, int(offset))
        clauses = []
        params: list[Any] = []
        for column, value in (("agent_id", agent_id), ("run_id", run_id), ("status", status)):
            if value is not None:
                clauses.append(f"{column} = %s")
                params.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT * FROM agent_traces {where} "
                    "ORDER BY created_at DESC, trace_id DESC LIMIT %s OFFSET %s",
                    (*params, limit + 1, offset),
                )
                rows = cur.fetchall()
        has_more = len(rows) > limit
        items = [_public_trace(row) for row in rows[:limit]]
        return {
            "items": items,
            "has_more": has_more,
            "next_cursor": str(offset + limit) if has_more else None,
        }

    def update_trace(
        self,
        trace_id: str,
        *,
        status: Optional[str] = None,
        final_output_summary: Optional[Dict[str, Any]] = None,
        ended_at: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        summary_json = (
            _json_text(final_output_summary, field="final_output_summary")
            if final_output_summary is not None
            else None
        )
        if final_output_summary is not None:
            _reject_sensitive(final_output_summary, "final_output_summary")
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE agent_traces
                    SET status = COALESCE(%s, status),
                        final_output_summary = COALESCE(%s, final_output_summary),
                        ended_at = COALESCE(%s, ended_at)
                    WHERE trace_id = %s
                    """,
                    (status, summary_json, ended_at, trace_id),
                )
                cur.execute("SELECT * FROM agent_traces WHERE trace_id = %s", (trace_id,))
                row = cur.fetchone()
        return _public_trace(row) if row else None

    def append_event(
        self,
        *,
        trace_id: str,
        event_type: str,
        actor_type: str,
        payload: Optional[Dict[str, Any]] = None,
        actor_id: Optional[str] = None,
        step_id: Optional[str] = None,
        decision_id: Optional[str] = None,
        artifact_id: Optional[str] = None,
        occurred_at: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        schema_version: int = 1,
        parent_event_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = payload or {}
        _reject_sensitive(payload)
        payload_json = _json_text(payload, field="payload")
        occurred_at = occurred_at or _utcnow_iso()
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT trace_id FROM agent_traces WHERE trace_id = %s FOR UPDATE", (trace_id,))
                if cur.fetchone() is None:
                    raise KeyError(f"unknown trace: {trace_id}")
                if idempotency_key:
                    cur.execute(
                        "SELECT * FROM agent_trace_events WHERE trace_id = %s AND idempotency_key = %s",
                        (trace_id, idempotency_key),
                    )
                    existing = cur.fetchone()
                    if existing:
                        return _public_event(existing)
                cur.execute(
                    "SELECT COALESCE(MAX(sequence_no), 0) + 1 AS next_sequence "
                    "FROM agent_trace_events WHERE trace_id = %s",
                    (trace_id,),
                )
                sequence_no = int(cur.fetchone()["next_sequence"])
                event_id = _new_event_id()
                cur.execute(
                    """
                    INSERT INTO agent_trace_events (
                        event_id, trace_id, sequence_no, event_type, actor_type,
                        actor_id, step_id, decision_id, artifact_id, parent_event_id, payload_json,
                        occurred_at, ingested_at, idempotency_key, schema_version
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING *
                    """,
                    (event_id, trace_id, sequence_no, event_type, actor_type,
                     actor_id, step_id, decision_id, artifact_id, parent_event_id, payload_json,
                     occurred_at, _utcnow_iso(), idempotency_key, int(schema_version)),
                )
                saved = cur.fetchone()
        return _public_event(saved)

    def list_events(
        self, trace_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> Dict[str, Any]:
        limit = max(1, min(int(limit), 100))
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM agent_trace_events "
                    "WHERE trace_id = %s AND sequence_no > %s "
                    "ORDER BY sequence_no ASC LIMIT %s",
                    (trace_id, int(after_sequence), limit + 1),
                )
                rows = cur.fetchall()
        has_more = len(rows) > limit
        items = [_public_event(row) for row in rows[:limit]]
        next_sequence = items[-1]["sequence_no"] if items else int(after_sequence)
        return {
            "items": items,
            "next_sequence_no": next_sequence + 1,
            "has_more": has_more,
        }
