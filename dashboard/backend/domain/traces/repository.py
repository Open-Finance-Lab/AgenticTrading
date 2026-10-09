"""SQLite persistence for ordered Agent execution traces.

The trace store is deliberately small: business records remain in their
existing stores and trace events reference them by id.  The Postgres twin uses
the same public methods and is selected by the production composition root in
the next loop.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from dashboard.backend.database import DB_PATH, enable_wal
from dashboard.backend.db_url import describe_database_url
from dashboard.backend.domain.traces.common import (
    _json_text,
    _new_event_id,
    _prepare_batch_event,
    _public_event,
    _public_trace,
    _reject_sensitive,
    _utcnow_iso,
)


class TraceStore:
    """Persist one run envelope and its append-only event timeline."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path or DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        enable_wal(self.db_path)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def _init_schema(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(
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
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS agent_trace_events (
                    event_id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    sequence_no INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    actor_type TEXT NOT NULL,
                    actor_id TEXT,
                    step_id TEXT,
                    decision_id TEXT,
                    artifact_id TEXT,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    occurred_at TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    idempotency_key TEXT,
                    schema_version INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY (trace_id) REFERENCES agent_traces(trace_id),
                    UNIQUE (trace_id, sequence_no),
                    UNIQUE (trace_id, idempotency_key)
                );
                CREATE INDEX IF NOT EXISTS idx_agent_trace_events_trace_sequence
                    ON agent_trace_events(trace_id, sequence_no);
                CREATE INDEX IF NOT EXISTS idx_agent_traces_agent_created
                    ON agent_traces(agent_id, created_at DESC);
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(agent_traces)")}
            if "parent_trace_id" not in columns:
                conn.execute("ALTER TABLE agent_traces ADD COLUMN parent_trace_id TEXT")
            event_columns = {row[1] for row in conn.execute("PRAGMA table_info(agent_trace_events)")}
            if "parent_event_id" not in event_columns:
                conn.execute("ALTER TABLE agent_trace_events ADD COLUMN parent_event_id TEXT")

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
        trace_id = trace_id or f"trace_{uuid.uuid4().hex[:16]}"
        started_at = started_at or _utcnow_iso()
        input_json = _json_text(initial_input or {}, field="initial_input")
        _reject_sensitive(initial_input or {}, "initial_input")
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO agent_traces (
                    trace_id, agent_id, agent_version_id, run_id, user_id,
                    trace_kind, status, initial_input_json, started_at, created_at, parent_trace_id
                ) VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?, ?, ?)
                """,
                (trace_id, agent_id, agent_version_id, run_id, user_id,
                 trace_kind, input_json, started_at, _utcnow_iso(), parent_trace_id),
            )
            row = conn.execute(
                "SELECT * FROM agent_traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
        return _public_trace(row)

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM agent_traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
        return _public_trace(row) if row else None

    def get_trace_for_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM agent_traces WHERE run_id = ?", (run_id,)
            ).fetchone()
        return _public_trace(row) if row else None

    def list_traces(
        self,
        *,
        agent_id: Optional[str] = None,
        agent_ids: Optional[Sequence[str]] = None,
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
                clauses.append(f"{column} = ?")
                params.append(value)
        if agent_ids is not None:
            ids = [str(value) for value in agent_ids]
            if not ids:
                return {"items": [], "has_more": False, "next_cursor": None}
            placeholders = ", ".join("?" for _ in ids)
            clauses.append(f"agent_id IN ({placeholders})")
            params.extend(ids)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._get_connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM agent_traces {where} "
                "ORDER BY created_at DESC, trace_id DESC LIMIT ? OFFSET ?",
                (*params, limit + 1, offset),
            ).fetchall()
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
            current = conn.execute(
                "SELECT status FROM agent_traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
            if current and current["status"] in {"completed", "failed"}:
                if status and status != current["status"]:
                    status = None
            conn.execute(
                """
                UPDATE agent_traces
                SET status = COALESCE(?, status),
                    final_output_summary = COALESCE(?, final_output_summary),
                    ended_at = COALESCE(?, ended_at)
                WHERE trace_id = ?
                """,
                (status, summary_json, ended_at, trace_id),
            )
            row = conn.execute(
                "SELECT * FROM agent_traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
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
        event_id = _new_event_id()
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            trace = conn.execute(
                "SELECT trace_id FROM agent_traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
            if trace is None:
                raise KeyError(f"unknown trace: {trace_id}")
            if idempotency_key:
                existing = conn.execute(
                    "SELECT * FROM agent_trace_events "
                    "WHERE trace_id = ? AND idempotency_key = ?",
                    (trace_id, idempotency_key),
                ).fetchone()
                if existing:
                    return _public_event(existing)
            row = conn.execute(
                "SELECT COALESCE(MAX(sequence_no), 0) + 1 AS next_sequence "
                "FROM agent_trace_events WHERE trace_id = ?",
                (trace_id,),
            ).fetchone()
            sequence_no = int(row["next_sequence"])
            conn.execute(
                """
                INSERT INTO agent_trace_events (
                    event_id, trace_id, sequence_no, event_type, actor_type,
                    actor_id, step_id, decision_id, artifact_id, parent_event_id, payload_json,
                    occurred_at, ingested_at, idempotency_key, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (event_id, trace_id, sequence_no, event_type, actor_type,
                 actor_id, step_id, decision_id, artifact_id, parent_event_id, payload_json,
                 occurred_at, _utcnow_iso(), idempotency_key, int(schema_version)),
            )
            saved = conn.execute(
                "SELECT * FROM agent_trace_events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return _public_event(saved)

    def append_events(self, *, trace_id: str, events: Sequence[Dict[str, Any]]) -> int:
        """Append several events in one transaction; returns how many were new.

        ``append_event`` costs a lock, an idempotency probe, a sequence read and
        an insert per event. A caller writing a burst of events (the decision
        tape flushes many bars at once) pays that once per batch instead. Every
        event is validated before the transaction opens, so one bad payload
        rejects the batch without writing a partial one. Events whose
        ``idempotency_key`` already exists are skipped, as ``append_event``
        would return the stored row for them.
        """
        prepared = [_prepare_batch_event(event) for event in events]
        if not prepared:
            return 0
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute(
                "SELECT trace_id FROM agent_traces WHERE trace_id = ?", (trace_id,)
            ).fetchone() is None:
                raise KeyError(f"unknown trace: {trace_id}")
            keys = [event["idempotency_key"] for event in prepared if event["idempotency_key"]]
            existing = set()
            if keys:
                placeholders = ",".join("?" for _ in keys)
                existing = {
                    row["idempotency_key"]
                    for row in conn.execute(
                        "SELECT idempotency_key FROM agent_trace_events "
                        f"WHERE trace_id = ? AND idempotency_key IN ({placeholders})",
                        (trace_id, *keys),
                    )
                }
            fresh = []
            for event in prepared:
                key = event["idempotency_key"]
                if key and key in existing:
                    continue
                if key:
                    existing.add(key)  # a key repeated inside the batch lands once
                fresh.append(event)
            if not fresh:
                return 0
            next_sequence = int(conn.execute(
                "SELECT COALESCE(MAX(sequence_no), 0) + 1 AS next_sequence "
                "FROM agent_trace_events WHERE trace_id = ?",
                (trace_id,),
            ).fetchone()["next_sequence"])
            ingested_at = _utcnow_iso()
            conn.executemany(
                """
                INSERT INTO agent_trace_events (
                    event_id, trace_id, sequence_no, event_type, actor_type,
                    actor_id, step_id, decision_id, artifact_id, parent_event_id, payload_json,
                    occurred_at, ingested_at, idempotency_key, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (_new_event_id(), trace_id, next_sequence + offset, event["event_type"],
                     event["actor_type"], event["actor_id"], event["step_id"],
                     event["decision_id"], event["artifact_id"], event["parent_event_id"],
                     event["payload_json"], event["occurred_at"], ingested_at,
                     event["idempotency_key"], event["schema_version"])
                    for offset, event in enumerate(fresh)
                ],
            )
        return len(fresh)

    def event_high_watermark(self, trace_id: str) -> int:
        """Freeze the export boundary before walking append-only event pages."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT COALESCE(MAX(sequence_no), 0) AS sequence_no FROM agent_trace_events WHERE trace_id = ?", (trace_id,)).fetchone()
            return int(row["sequence_no"])

    def list_events(
        self, trace_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> Dict[str, Any]:
        limit = max(1, min(int(limit), 100))
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_trace_events "
                "WHERE trace_id = ? AND sequence_no > ? "
                "ORDER BY sequence_no ASC LIMIT ?",
                (trace_id, int(after_sequence), limit + 1),
            ).fetchall()
        has_more = len(rows) > limit
        items = [_public_event(row) for row in rows[:limit]]
        next_sequence = items[-1]["sequence_no"] if items else int(after_sequence)
        return {
            "items": items,
            "next_sequence_no": next_sequence + 1,
            "has_more": has_more,
        }


def _build_trace_store():
    # AGENT_RUNS_DATABASE_URL, not CONTENT_DATABASE_URL: a trace is run data
    # (keyed by run_id, one event per step), and since the decision tape writes
    # a decision/execution pair per bar of every dashboard backtest it is the
    # fastest-growing run data there is. That growth belongs in the dedicated
    # run-history project, isolated from the auth-critical users/content
    # database -- the reason AGENT_RUNS_DATABASE_URL exists. No fallback to
    # CONTENT_DATABASE_URL, by the same rule database.py's _build_backtest_db
    # follows.
    database_url = os.getenv("AGENT_RUNS_DATABASE_URL")
    if database_url:
        from dashboard.backend.domain.traces.repository_postgres import PostgresTraceStore

        print(f"trace_store backend: postgres ({describe_database_url(database_url)})")
        return PostgresTraceStore(database_url)
    print("trace_store backend: sqlite (ephemeral on Render)")
    return TraceStore()


trace_store = _build_trace_store()
