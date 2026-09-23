"""Storage for the research-agent module (design N2/PR2).

Three concerns, one small sqlite module (same DATABASE_PATH as the rest of the
dashboard — the deploy runs sqlite stores, consistent with the agent store):

- ``research_agent_adds``   : which user cloned which research template
                              (the research analogue of the marketplace clone)
- ``research_runs``         : one row per submitted research run
- ``research_artifacts``    : the completed run's deliverables, stored as
                              base64 text (Render's filesystem is ephemeral;
                              the database is the only durable home)

Write-shape notes:
- Every helper opens its own connection (short-lived, like users_store).
- ``_init_schema`` runs on module import — cheap CREATE IF NOT EXISTS, and the
  research module is the only caller.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Optional

from dashboard.backend.database import DB_PATH


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _init_schema() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS research_agent_adds (
                user_id INTEGER NOT NULL,
                template_id TEXT NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, template_id)
            );

            CREATE TABLE IF NOT EXISTS research_runs (
                run_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                template_id TEXT NOT NULL,
                service_run_id TEXT,
                status TEXT NOT NULL DEFAULT 'queued',
                settings_json TEXT NOT NULL,
                email_me INTEGER NOT NULL DEFAULT 0,
                emailed INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS research_artifacts (
                run_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                filename TEXT,
                content_base64 TEXT,
                PRIMARY KEY (run_id, kind)
            );

            CREATE INDEX IF NOT EXISTS idx_research_runs_user
                ON research_runs(user_id, created_at DESC);
            """
        )


_init_schema()


# --- adds (the research "clone") -------------------------------------------

def add_research_agent(user_id: int, template_id: str) -> bool:
    """Idempotent add. Returns True when a new row was created."""
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO research_agent_adds (user_id, template_id) VALUES (?, ?)",
            (user_id, template_id),
        )
        return cursor.rowcount > 0


def remove_research_agent(user_id: int, template_id: str) -> bool:
    with _connect() as conn:
        cursor = conn.execute(
            "DELETE FROM research_agent_adds WHERE user_id = ? AND template_id = ?",
            (user_id, template_id),
        )
        return cursor.rowcount > 0


def list_added_template_ids(user_id: int) -> List[str]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT template_id FROM research_agent_adds WHERE user_id = ? ORDER BY added_at DESC",
            (user_id,),
        ).fetchall()
    return [row["template_id"] for row in rows]


# --- runs -------------------------------------------------------------------

def create_run(
    *,
    run_id: str,
    user_id: int,
    template_id: str,
    service_run_id: str,
    status: str,
    settings: Dict[str, Any],
    email_me: bool,
) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO research_runs (run_id, user_id, template_id, service_run_id,"
            " status, settings_json, email_me) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, user_id, template_id, service_run_id, status,
             json.dumps(settings, ensure_ascii=False), int(email_me)),
        )


def get_run(run_id: str, user_id: int) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM research_runs WHERE run_id = ? AND user_id = ?",
            (run_id, user_id),
        ).fetchone()
    return dict(row) if row else None


def list_runs_for_user(user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT run_id, template_id, status, settings_json, error,"
            " created_at, completed_at FROM research_runs"
            " WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def update_run_status(run_id: str, status: str, error: Optional[str] = None,
                      completed: bool = False) -> None:
    with _connect() as conn:
        if completed:
            conn.execute(
                "UPDATE research_runs SET status = ?, error = ?,"
                " completed_at = CURRENT_TIMESTAMP WHERE run_id = ?",
                (status, error, run_id),
            )
        else:
            conn.execute(
                "UPDATE research_runs SET status = ?, error = ? WHERE run_id = ?",
                (status, error, run_id),
            )


def mark_emailed(run_id: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE research_runs SET emailed = 1 WHERE run_id = ?", (run_id,))


# --- artifacts --------------------------------------------------------------

def store_artifacts(run_id: str, artifacts: Dict[str, Dict[str, str]],
                    evidence: Dict[str, Any], report_markdown: str) -> None:
    """Persist everything a completed run delivered (replace-on-complete)."""
    rows = [
        (run_id, "markdown", f"{run_id}.md", None),
    ]
    with _connect() as conn:
        conn.execute("DELETE FROM research_artifacts WHERE run_id = ?", (run_id,))
        conn.execute(
            "INSERT INTO research_artifacts (run_id, kind, filename, content_base64)"
            " VALUES (?, 'markdown_report', ?, ?)",
            (run_id, f"{run_id}.md", report_markdown),
        )
        for kind, item in (artifacts or {}).items():
            safe_kind = str(kind).replace("/", "_")
            conn.execute(
                "INSERT INTO research_artifacts (run_id, kind, filename, content_base64)"
                " VALUES (?, ?, ?, ?)",
                (run_id, safe_kind,
                 item.get("filename") or f"{run_id}.{safe_kind}",
                 item.get("content_base64")),
            )
        if evidence is not None:
            conn.execute(
                "INSERT INTO research_artifacts (run_id, kind, filename, content_base64)"
                " VALUES (?, 'evidence_json', ?, ?)",
                (run_id, f"{run_id}_evidence.json",
                 json.dumps(evidence, ensure_ascii=False)),
            )


def get_artifact(run_id: str, kind: str) -> Optional[Dict[str, Any]]:
    kind_map = {"markdown": "markdown_report", "evidence_json": "evidence_json"}
    lookup = kind_map.get(kind, kind)
    with _connect() as conn:
        row = conn.execute(
            "SELECT kind, filename, content_base64 FROM research_artifacts"
            " WHERE run_id = ? AND kind = ?",
            (run_id, lookup),
        ).fetchone()
    return dict(row) if row else None
