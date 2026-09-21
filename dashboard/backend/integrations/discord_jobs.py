"""Persist Discord-triggered backtest notification jobs.

The Discord bot starts a backtest via HTTP, then watches in the background and
posts results when the API job finishes. Jobs are stored in a small SQLite file
so a bot restart can resume in-flight watches (delivery does not rely on the
15-minute Discord interaction token).
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dashboard.backend.paths import DATA_DIR

DEFAULT_JOBS_DB = DATA_DIR / "discord_backtest_jobs.db"

STATUS_PENDING = "pending"
STATUS_WATCHING = "watching"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
# A run the SERVER stopped -- cancelled by its owner, or halted at its own
# wall-clock budget. Deliberately not `STATUS_FAILED`: the rest of this feature
# (`_finalize_slot`'s docstring, the status route's branch order, app.js's
# four-state panel, `.is-timed-out` in styles.css) draws the same line four
# times, that neither outcome is the user's error, and a job store that files
# them under "failed" makes Discord the one surface that disagrees. Like
# `STATUS_FAILED` it is terminal and absent from `_OPEN_STATUSES`, so a restart
# does not resume it.
STATUS_STOPPED = "stopped"
STATUS_NOTIFIED = "notified"
STATUS_NOTIFY_FAILED = "notify_failed"

_OPEN_STATUSES = (STATUS_PENDING, STATUS_WATCHING)


@dataclass(frozen=True)
class DiscordBacktestJob:
    job_id: str
    discord_user_id: str
    channel_id: int
    session_id: str
    label: str
    status: str
    created_at: float
    live_run_id: Optional[str] = None
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    share_url: Optional[str] = None
    run_id: Optional[str] = None
    error: Optional[str] = None
    notified_at: Optional[float] = None
    guild_id: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "discord_user_id": self.discord_user_id,
            "channel_id": self.channel_id,
            "guild_id": self.guild_id,
            "session_id": self.session_id,
            "label": self.label,
            "status": self.status,
            "created_at": self.created_at,
            "live_run_id": self.live_run_id,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "share_url": self.share_url,
            "run_id": self.run_id,
            "error": self.error,
            "notified_at": self.notified_at,
        }


class DiscordJobStore:
    """Thread-safe SQLite store for Discord backtest notify jobs."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_JOBS_DB
        self._lock = threading.Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS discord_backtest_jobs (
                        job_id TEXT PRIMARY KEY,
                        discord_user_id TEXT NOT NULL,
                        channel_id INTEGER NOT NULL,
                        guild_id INTEGER,
                        session_id TEXT NOT NULL,
                        label TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        live_run_id TEXT,
                        agent_id TEXT,
                        agent_name TEXT,
                        share_url TEXT,
                        run_id TEXT,
                        error TEXT,
                        notified_at REAL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_discord_jobs_status
                    ON discord_backtest_jobs(status)
                    """
                )
                conn.commit()

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> DiscordBacktestJob:
        return DiscordBacktestJob(
            job_id=row["job_id"],
            discord_user_id=row["discord_user_id"],
            channel_id=int(row["channel_id"]),
            session_id=row["session_id"],
            label=row["label"],
            status=row["status"],
            created_at=float(row["created_at"]),
            live_run_id=row["live_run_id"],
            agent_id=row["agent_id"],
            agent_name=row["agent_name"],
            share_url=row["share_url"],
            run_id=row["run_id"],
            error=row["error"],
            notified_at=row["notified_at"],
            guild_id=int(row["guild_id"]) if row["guild_id"] is not None else None,
        )

    def create_job(
        self,
        *,
        discord_user_id: str,
        channel_id: int,
        session_id: str,
        label: str,
        live_run_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        share_url: Optional[str] = None,
        guild_id: Optional[int] = None,
        job_id: Optional[str] = None,
    ) -> DiscordBacktestJob:
        job = DiscordBacktestJob(
            job_id=job_id or uuid.uuid4().hex[:12],
            discord_user_id=str(discord_user_id),
            channel_id=int(channel_id),
            session_id=session_id,
            label=label,
            status=STATUS_PENDING,
            created_at=time.time(),
            live_run_id=live_run_id,
            agent_id=agent_id,
            agent_name=agent_name,
            share_url=share_url,
            guild_id=guild_id,
        )
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO discord_backtest_jobs (
                        job_id, discord_user_id, channel_id, guild_id, session_id,
                        label, status, created_at, live_run_id, agent_id, agent_name,
                        share_url
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job.job_id,
                        job.discord_user_id,
                        job.channel_id,
                        job.guild_id,
                        job.session_id,
                        job.label,
                        job.status,
                        job.created_at,
                        job.live_run_id,
                        job.agent_id,
                        job.agent_name,
                        job.share_url,
                    ),
                )
                conn.commit()
        return job

    def get(self, job_id: str) -> Optional[DiscordBacktestJob]:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM discord_backtest_jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
        return self._row_to_job(row) if row else None

    def list_open(self) -> list[DiscordBacktestJob]:
        placeholders = ",".join("?" for _ in _OPEN_STATUSES)
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT * FROM discord_backtest_jobs
                    WHERE status IN ({placeholders})
                    ORDER BY created_at ASC
                    """,
                    _OPEN_STATUSES,
                ).fetchall()
        return [self._row_to_job(r) for r in rows]

    def update(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        run_id: Optional[str] = None,
        error: Optional[str] = None,
        notified_at: Optional[float] = None,
        live_run_id: Optional[str] = None,
    ) -> Optional[DiscordBacktestJob]:
        fields: list[str] = []
        values: list[Any] = []
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if run_id is not None:
            fields.append("run_id = ?")
            values.append(run_id)
        if error is not None:
            fields.append("error = ?")
            values.append(error)
        if notified_at is not None:
            fields.append("notified_at = ?")
            values.append(notified_at)
        if live_run_id is not None:
            fields.append("live_run_id = ?")
            values.append(live_run_id)
        if not fields:
            return self.get(job_id)
        values.append(job_id)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    f"UPDATE discord_backtest_jobs SET {', '.join(fields)} WHERE job_id = ?",
                    values,
                )
                conn.commit()
        return self.get(job_id)


_store: Optional[DiscordJobStore] = None
_store_lock = threading.Lock()


def get_job_store(db_path: Optional[Path] = None) -> DiscordJobStore:
    """Process-wide job store (override path mainly for tests)."""
    global _store
    with _store_lock:
        if db_path is not None:
            return DiscordJobStore(db_path)
        if _store is None:
            import os

            raw = (os.getenv("DISCORD_JOBS_DB") or "").strip()
            path = Path(raw) if raw else DEFAULT_JOBS_DB
            _store = DiscordJobStore(path)
        return _store


def reset_job_store_for_tests() -> None:
    """Drop the cached store so the next get_job_store() rebuilds (tests only)."""
    global _store
    with _store_lock:
        _store = None
