"""Protocol Run records.

A protocol Run (run_xxx) is the generalized successor to a Backtest. It links
an AgentVersion + environment + config to the underlying execution engine
(an external backtest session, backtest_id) and, once finalized, to the stored
result run (agent_runs.run_id, e.g. ext_...). Persisting this mapping lets the
Run API answer GET /runs/{run_id} and result/metric queries even after the
in-memory engine session is gone.

Moved verbatim (Phase 3B1) from ``dashboard/backend/run_store.py``; the original
module was removed in Phase 4A. Public class, the ``run_store`` singleton,
SQL, return schemas, and behavior are unchanged; only the module location moved.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dashboard.backend.database import DB_PATH, enable_wal

# Identifies this process in protocol_runs.owner_instance. Fresh per process on
# purpose: after a crash/restart the old instance's rows stop being heartbeated
# and become recoverable by ANY instance via fail_stale_runs (multi-worker-safe,
# unlike the blanket startup recovery).
INSTANCE_ID = uuid.uuid4().hex[:12]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:12]}"


def _public_run(row: sqlite3.Row | Dict[str, Any]) -> Dict[str, Any]:
    data = dict(row)
    try:
        config = json.loads(data.get("config") or "{}")
    except (json.JSONDecodeError, TypeError):
        config = {}
    return {
        "run_id": data["run_id"],
        "agent_id": data.get("agent_id"),
        "agent_version_id": data.get("agent_version_id"),
        "session_id": data.get("session_id"),
        "environment_id": data.get("environment_id"),
        "environment_type": data.get("environment_type"),
        "config": config,
        "backtest_id": data.get("backtest_id"),
        "result_run_id": data.get("result_run_id"),
        "status": data.get("status"),
        "step_index": data.get("step_index"),
        "total_steps": data.get("total_steps"),
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
    }


class RunStore:
    """Persist protocol run metadata and engine/result linkage."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path or DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        enable_wal(self.db_path)  # shared helper — one definition for both layers
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        # Step writes share this file with the engine's heavy finalize writes;
        # wait for the lock instead of failing fast with "database is locked".
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def _init_schema(self) -> None:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS protocol_runs (
                run_id TEXT PRIMARY KEY,
                agent_id TEXT,
                agent_version_id TEXT,
                session_id TEXT NOT NULL,
                environment_id TEXT,
                environment_type TEXT,
                config TEXT NOT NULL DEFAULT '{}',
                backtest_id TEXT,
                result_run_id TEXT,
                status TEXT NOT NULL DEFAULT 'created',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                owner_instance TEXT,
                heartbeat_at TEXT,
                step_index INTEGER,
                total_steps INTEGER
            )
            """
        )
        # Existing DBs predate the recovery/step columns; CREATE TABLE IF NOT
        # EXISTS won't add them, so probe and ALTER (same pattern as the
        # main DB's _migrate_schema).
        cursor.execute("PRAGMA table_info(protocol_runs)")
        columns = {row[1] for row in cursor.fetchall()}
        self._add_recovery_columns(cursor, columns)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_protocol_runs_agent ON protocol_runs(agent_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_protocol_runs_backtest ON protocol_runs(backtest_id)"
        )
        # protocol_steps mirrors ProtocolRun's in-memory step bookkeeping so a
        # process restart keeps historical step-id queries and idempotent
        # decision replays working (H4 follow-up). One row per step: a step has
        # at most one accepted decision, so the accepted idempotency_key and
        # its result live on the step row.
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS protocol_steps (
                run_id TEXT NOT NULL,
                step_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                timestamp TEXT,
                deadline_at TEXT,
                status TEXT NOT NULL DEFAULT 'awaiting_decision',
                idempotency_key TEXT,
                result_json TEXT,
                PRIMARY KEY (run_id, step_id),
                UNIQUE (run_id, sequence)
            )
            """
        )
        conn.commit()
        conn.close()

    @staticmethod
    def _add_recovery_columns(cursor, existing_columns) -> None:
        """ALTER in the recovery + step-count columns missing from
        ``existing_columns`` (owner_instance/heartbeat_at for multi-worker
        recovery; step_index/total_steps so a rehydrated terminal run reports
        real progress instead of 0/0).

        Tolerates losing the probe→ALTER race to a concurrently-starting
        sibling process (multi-worker startup): a duplicate-column error
        means the column is there, which is the goal — anything else is
        real and re-raised."""
        for col_name, col_def in (
            ("owner_instance", "TEXT"),
            ("heartbeat_at", "TEXT"),
            ("step_index", "INTEGER"),
            ("total_steps", "INTEGER"),
        ):
            if col_name in existing_columns:
                continue
            try:
                cursor.execute(
                    f"ALTER TABLE protocol_runs ADD COLUMN {col_name} {col_def}"
                )
            except sqlite3.OperationalError as exc:
                if "duplicate column" not in str(exc).lower():
                    raise

    def create_run(
        self,
        *,
        agent_id: Optional[str],
        agent_version_id: Optional[str],
        session_id: str,
        environment_id: Optional[str],
        environment_type: str,
        config: Dict[str, Any],
        backtest_id: Optional[str] = None,
        status: str = "created",
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        # run_id: v2 mints its own canonical id before creating the backend;
        # v1 leaves it None and gets a run_<hex> id minted here.
        run_id = run_id or _new_run_id()
        now = _utcnow_iso()
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO protocol_runs (
                run_id, agent_id, agent_version_id, session_id, environment_id,
                environment_type, config, backtest_id, result_run_id, status,
                created_at, updated_at, owner_instance, heartbeat_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                agent_id,
                agent_version_id,
                session_id,
                environment_id,
                environment_type,
                json.dumps(config or {}),
                backtest_id,
                None,
                status,
                now,
                now,
                INSTANCE_ID,
                now,
            ),
        )
        conn.commit()
        cursor.execute("SELECT * FROM protocol_runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        conn.close()
        return _public_run(row)

    def update_run(
        self,
        run_id: str,
        *,
        backtest_id: Optional[str] = None,
        result_run_id: Optional[str] = None,
        status: Optional[str] = None,
        step_index: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> None:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE protocol_runs
            SET backtest_id = COALESCE(?, backtest_id),
                result_run_id = COALESCE(?, result_run_id),
                status = COALESCE(?, status),
                step_index = COALESCE(?, step_index),
                total_steps = COALESCE(?, total_steps),
                updated_at = ?
            WHERE run_id = ?
            """,
            (backtest_id, result_run_id, status, step_index, total_steps,
             _utcnow_iso(), run_id),
        )
        conn.commit()
        conn.close()

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM protocol_runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        conn.close()
        return _public_run(row) if row else None

    def list_runs(self, agent_id: str) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM protocol_runs WHERE agent_id = ? ORDER BY created_at DESC",
            (agent_id,),
        )
        rows = cursor.fetchall()
        conn.close()
        return [_public_run(row) for row in rows]

    # Runs that are still consuming resources (not yet terminal).
    _ACTIVE_STATUSES = ("created", "loading", "running")

    def count_active_runs(self, agent_id: str) -> int:
        """Number of an agent's runs that are not yet completed/failed."""
        placeholders = ",".join("?" for _ in self._ACTIVE_STATUSES)
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT COUNT(*) FROM protocol_runs "
            f"WHERE agent_id = ? AND status IN ({placeholders})",
            (agent_id, *self._ACTIVE_STATUSES),
        )
        (count,) = cursor.fetchone()
        conn.close()
        return int(count)

    def count_active_runs_for_agents(self, agent_ids: List[str]) -> int:
        """Active runs across a set of agents — the per-account cap's count.

        The account→agent mapping lives in the agents store (possibly a
        different database under ``CONTENT_DATABASE_URL``), so the caller
        resolves the id list and this counts locally; a JOIN is impossible
        across that seam. Chunked because nothing bounds how many agents one
        account may own, and SQLite refuses a statement whose bind-parameter
        count exceeds SQLITE_MAX_VARIABLE_NUMBER (999 on older builds) — one
        oversized IN list would turn this account's every create into a 500.
        """
        ids = [str(a) for a in agent_ids if a]
        if not ids:
            return 0
        status_ph = ",".join("?" for _ in self._ACTIVE_STATUSES)
        total = 0
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            for start in range(0, len(ids), 500):
                chunk = ids[start:start + 500]
                id_ph = ",".join("?" for _ in chunk)
                cursor.execute(
                    f"SELECT COUNT(*) FROM protocol_runs "
                    f"WHERE agent_id IN ({id_ph}) AND status IN ({status_ph})",
                    (*chunk, *self._ACTIVE_STATUSES),
                )
                (count,) = cursor.fetchone()
                total += int(count)
        finally:
            conn.close()
        return total

    def count_active_runs_total(self) -> int:
        """Active runs across ALL agents — the global backstop cap's count.

        Deliberately reconciliation-free: both surfaces flip their row to a
        terminal status synchronously inside the finalizing request, so a raw
        COUNT tracks reality; the residual (terminal run, no subsequent poll)
        is bounded by the reaper interval and errs toward a spurious 429 —
        the safe direction, and Retry-After tells the client when to retry."""
        placeholders = ",".join("?" for _ in self._ACTIVE_STATUSES)
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT COUNT(*) FROM protocol_runs WHERE status IN ({placeholders})",
            self._ACTIVE_STATUSES,
        )
        (count,) = cursor.fetchone()
        conn.close()
        return int(count)

    def fail_unfinished_runs(self) -> int:
        """Mark runs left non-terminal by a crash/restart as failed.

        The in-memory engine session does not survive a process restart, so a
        run still marked running/loading/created can never resume — fail it so
        it stops counting against the per-agent active-run cap and reads report
        an honest terminal state. Returns the number of rows updated.
        """
        placeholders = ",".join("?" for _ in self._ACTIVE_STATUSES)
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE protocol_runs SET status = 'failed', updated_at = ? "
            f"WHERE status IN ({placeholders})",
            (_utcnow_iso(), *self._ACTIVE_STATUSES),
        )
        updated = cursor.rowcount
        conn.commit()
        conn.close()
        return int(updated)

    def heartbeat_runs(self, run_ids: List[str]) -> None:
        """Refresh heartbeat_at (and claim owner_instance) for live runs.

        Called by the reaper each pass for every run whose engine session is
        alive in this process — a run that keeps heartbeating is never
        eligible for fail_stale_runs, no matter which instance created it."""
        if not run_ids:
            return
        placeholders = ",".join("?" for _ in run_ids)
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE protocol_runs SET heartbeat_at = ?, owner_instance = ? "
            f"WHERE run_id IN ({placeholders})",
            (_utcnow_iso(), INSTANCE_ID, *run_ids),
        )
        conn.commit()
        conn.close()

    def fail_stale_runs(self, stale_seconds: float) -> int:
        """Fail non-terminal runs whose heartbeat stopped (multi-worker-safe
        recovery). Unlike fail_unfinished_runs' blanket UPDATE, this only
        touches rows no live process is heartbeating, so one worker's startup
        can't kill a sibling's in-flight runs. Rows from before the heartbeat
        column fall back to updated_at. Returns rows updated."""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=stale_seconds)
        ).replace(microsecond=0).isoformat()
        placeholders = ",".join("?" for _ in self._ACTIVE_STATUSES)
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE protocol_runs SET status = 'failed', updated_at = ? "
            f"WHERE status IN ({placeholders}) "
            f"AND COALESCE(heartbeat_at, updated_at, created_at) < ?",
            (_utcnow_iso(), *self._ACTIVE_STATUSES, cutoff),
        )
        updated = cursor.rowcount
        conn.commit()
        conn.close()
        return int(updated)

    # -- protocol_steps: persisted step bookkeeping (H4 follow-up) ---------

    def save_step(self, run_id: str, step_id: str, sequence: int,
                  timestamp: Any = None, deadline_at: Any = None) -> None:
        """Upsert a step row when it is (re-)awaited — mirrors ensure_step_id."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO protocol_steps (run_id, step_id, sequence, timestamp,
                                        deadline_at, status)
            VALUES (?, ?, ?, ?, ?, 'awaiting_decision')
            ON CONFLICT(run_id, step_id) DO UPDATE SET
                timestamp = excluded.timestamp,
                deadline_at = excluded.deadline_at,
                status = 'awaiting_decision'
            """,
            (
                run_id,
                step_id,
                int(sequence),
                str(timestamp) if timestamp is not None else None,
                str(deadline_at) if deadline_at is not None else None,
            ),
        )
        conn.commit()
        conn.close()

    def finalize_step(self, run_id: str, step_id: str, idempotency_key: str,
                      result: Dict[str, Any]) -> None:
        """Record the step's one accepted decision — mirrors submit_decision."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE protocol_steps
            SET status = 'completed', idempotency_key = ?, result_json = ?
            WHERE run_id = ? AND step_id = ?
            """,
            (idempotency_key, json.dumps(result, default=str), run_id, step_id),
        )
        conn.commit()
        conn.close()

    def get_steps(self, run_id: str) -> List[Dict[str, Any]]:
        """All persisted steps for a run, in sequence order (result parsed).

        Unbounded on purpose: a run's step count is capped by its backtest
        window (total_steps), so the full history is small by construction."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM protocol_steps WHERE run_id = ? ORDER BY sequence",
            (run_id,),
        )
        rows = cursor.fetchall()
        conn.close()
        steps = []
        for row in rows:
            data = dict(row)
            raw = data.pop("result_json", None)
            try:
                data["result"] = json.loads(raw) if raw else None
            except (TypeError, ValueError):
                data["result"] = None
            steps.append(data)
        return steps


run_store = RunStore()
