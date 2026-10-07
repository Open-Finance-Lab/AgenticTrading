"""SQLite persistence for ordered Agent execution traces.

The trace store is deliberately small: business records remain in their
existing stores and trace events reference them by id.  The Postgres twin uses
the same public methods and is selected by the production composition root in
the next loop.
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from dashboard.backend.database import DB_PATH, enable_wal

_MAX_JSON_BYTES = 64 * 1024
_SENSITIVE_KEY_RE = re.compile(
    r"(?:api[_-]?key|authorization|cookie|password|secret|token)", re.IGNORECASE
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_event_id() -> str:
    return f"tev_{uuid.uuid4().hex[:16]}"


def _json_text(value: Any, *, field: str) -> str:
    try:
        encoded = json.dumps(value if value is not None else {}, ensure_ascii=False,
                             separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be JSON serializable") from exc
    if len(encoded.encode("utf-8")) > _MAX_JSON_BYTES:
        raise ValueError(f"{field} exceeds {_MAX_JSON_BYTES} bytes")
    return encoded


def _reject_sensitive(value: Any, path: str = "payload") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if _SENSITIVE_KEY_RE.search(str(key)):
                raise ValueError(f"sensitive field is not allowed in {path}.{key}")
            _reject_sensitive(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_sensitive(child, f"{path}[{index}]")


def _public_trace(row: sqlite3.Row | Dict[str, Any]) -> Dict[str, Any]:
    data = dict(row)
    return data


def _public_event(row: sqlite3.Row | Dict[str, Any]) -> Dict[str, Any]:
    data = dict(row)
    try:
        data["payload"] = json.loads(data.pop("payload_json") or "{}")
    except (TypeError, ValueError):
        data["payload"] = {}
    return data


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
                    trace_kind, status, initial_input_json, started_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?, ?)
                """,
                (trace_id, agent_id, agent_version_id, run_id, user_id,
                 trace_kind, input_json, started_at, _utcnow_iso()),
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
    ) -> Dict[str, Any]:
        payload = payload or {}
        _reject_sensitive(payload)
        payload_json = _json_text(payload, field="payload")
        occurred_at = occurred_at or _utcnow_iso()
        event_id = _new_event_id()
        with self._get_connection() as conn:
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
                    actor_id, step_id, decision_id, artifact_id, payload_json,
                    occurred_at, ingested_at, idempotency_key, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (event_id, trace_id, sequence_no, event_type, actor_type,
                 actor_id, step_id, decision_id, artifact_id, payload_json,
                 occurred_at, _utcnow_iso(), idempotency_key, int(schema_version)),
            )
            saved = conn.execute(
                "SELECT * FROM agent_trace_events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return _public_event(saved)

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


trace_store = TraceStore()
