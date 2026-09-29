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
                reservation_id TEXT,
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
        # Existing installs created the table before billing (route 0) added
        # the reservation column; CREATE IF NOT EXISTS won't add it there.
        migrations = (
            "ALTER TABLE research_runs ADD COLUMN reservation_id TEXT",
            "ALTER TABLE research_runs ADD COLUMN estimate_micro INTEGER",
        )
        for statement in migrations:
            try:
                with _connect() as conn:
                    conn.execute(statement)
            except sqlite3.OperationalError:
                pass  # column already exists


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
    reservation_id: str,
    estimate_micro: int,
    status: str,
    settings: Dict[str, Any],
    email_me: bool,
) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO research_runs (run_id, user_id, template_id, service_run_id,"
            " reservation_id, estimate_micro, status, settings_json, email_me)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, user_id, template_id, service_run_id, reservation_id,
             int(estimate_micro), status,
             json.dumps(settings, ensure_ascii=False), int(email_me)),
        )


def list_nonterminal_runs() -> List[Dict[str, Any]]:
    """All queued/running runs across users — the sweeper's work queue."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM research_runs WHERE status IN ('queued', 'running')"
        ).fetchall()
    return [dict(row) for row in rows]



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


# ---------------------------------------------------------------------------
# Postgres twin (production durability). The SQLite module above is the dev
# and test backend; on Render the filesystem is ephemeral and every restart
# wiped the user's added agents and run history — the same class of loss the
# users table had before USERS_DATABASE_URL. Selected when
# CONTENT_DATABASE_URL is set (research adds/runs are user content, same
# scope as agents/strategies per the spec's Decision 2).
# ---------------------------------------------------------------------------

import os as _os
import json as _json


def _research_postgres_url() -> str:
    return (_os.getenv("CONTENT_DATABASE_URL") or "").strip()


def _build_research_store():
    url = _research_postgres_url()
    if not url:
        return "sqlite"
    # Late import: psycopg is only installed in deployments that need it.
    from dashboard.backend.db_pool import get_pool
    from dashboard.backend.db_url import init_schema_unless_worker

    class _PostgresResearchStore:
        def __init__(self, url):
            self.url = url
            init_schema_unless_worker("research_store", self._init_schema)

        def _conn(self):
            return get_pool(self.url).connection()

        def _init_schema(self):
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS research_agent_adds (
                            user_id INTEGER NOT NULL,
                            template_id TEXT NOT NULL,
                            added_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                            PRIMARY KEY (user_id, template_id)
                        )
                    """)
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS research_runs (
                            run_id TEXT PRIMARY KEY,
                            user_id INTEGER NOT NULL,
                            template_id TEXT NOT NULL,
                            service_run_id TEXT,
                            reservation_id TEXT,
                            estimate_micro INTEGER,
                            status TEXT NOT NULL DEFAULT 'queued',
                            settings_json TEXT NOT NULL,
                            email_me BOOLEAN NOT NULL DEFAULT FALSE,
                            emailed BOOLEAN NOT NULL DEFAULT FALSE,
                            error TEXT,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                            completed_at TIMESTAMPTZ
                        )
                    """)
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS research_artifacts (
                            run_id TEXT NOT NULL,
                            kind TEXT NOT NULL,
                            filename TEXT,
                            content_base64 TEXT,
                            PRIMARY KEY (run_id, kind)
                        )
                    """)
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_research_runs_user
                            ON research_runs(user_id, created_at DESC)
                    """)

        def add_research_agent(self, user_id, template_id):
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO research_agent_adds (user_id, template_id)"
                        " VALUES (%s, %s) ON CONFLICT DO NOTHING",
                        (user_id, template_id),
                    )
                    return cur.rowcount > 0

        def remove_research_agent(self, user_id, template_id):
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM research_agent_adds"
                        " WHERE user_id = %s AND template_id = %s",
                        (user_id, template_id),
                    )
                    return cur.rowcount > 0

        def list_added_template_ids(self, user_id):
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT template_id FROM research_agent_adds"
                        " WHERE user_id = %s ORDER BY added_at DESC",
                        (user_id,),
                    )
                    return [r["template_id"] for r in cur.fetchall()]

        def create_run(self, *, run_id, user_id, template_id,
                       service_run_id, reservation_id, estimate_micro,
                       status, settings, email_me):
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO research_runs"
                        " (run_id, user_id, template_id, service_run_id,"
                        "  reservation_id, estimate_micro, status,"
                        "  settings_json, email_me)"
                        " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        (run_id, user_id, template_id, service_run_id,
                         reservation_id, estimate_micro, status,
                         _json.dumps(settings, ensure_ascii=False), email_me),
                    )

        def get_run(self, run_id, user_id):
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT * FROM research_runs"
                        " WHERE run_id = %s AND user_id = %s",
                        (run_id, user_id),
                    )
                    row = cur.fetchone()
            if not row:
                return None
            # dict_row: the pool already returns column-name → value dicts;
            # no re-zip needed (zipping a dict with its own keys yields
            # {name: name}, a row of column names instead of data).
            return dict(row)

        def list_runs_for_user(self, user_id, limit=50):
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT run_id, template_id, status, settings_json,"
                        " error, created_at, completed_at FROM research_runs"
                        " WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
                        (user_id, limit),
                    )
                    rows = cur.fetchall()
                # dict_row returns dicts; keep settings_json verbatim because the
            # /runs route owns the parsing (it pops settings_json itself) —
            # pre-parsing here made the router's KeyError fallback wipe every
            # Postgres-backed run's settings to {} on the wire.
            return [dict(row) for row in rows]

        def update_run_status(self, run_id, status, error=None, completed=False):
            with self._conn() as conn:
                with conn.cursor() as cur:
                    if completed:
                        cur.execute(
                            "UPDATE research_runs SET status=%s, error=%s,"
                            " completed_at=now() WHERE run_id=%s",
                            (status, error, run_id),
                        )
                    else:
                        cur.execute(
                            "UPDATE research_runs SET status=%s, error=%s"
                            " WHERE run_id=%s",
                            (status, error, run_id),
                        )

        def mark_emailed(self, run_id):
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE research_runs SET emailed=TRUE WHERE run_id=%s",
                        (run_id,),
                    )

        def store_artifacts(self, run_id, artifacts, evidence, report_markdown):
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM research_artifacts WHERE run_id=%s", (run_id,))
                    cur.execute(
                        "INSERT INTO research_artifacts (run_id, kind, filename, content_base64)"
                        " VALUES (%s, 'markdown_report', %s, %s)",
                        (run_id, f"{run_id}.md", report_markdown),
                    )
                    for kind, item in (artifacts or {}).items():
                        safe = str(kind).replace("/", "_")
                        cur.execute(
                            "INSERT INTO research_artifacts (run_id, kind, filename, content_base64)"
                            " VALUES (%s, %s, %s, %s)",
                            (run_id, safe, item.get("filename") or f"{run_id}.{safe}",
                             item.get("content_base64")),
                        )
                    if evidence is not None:
                        cur.execute(
                            "INSERT INTO research_artifacts (run_id, kind, filename, content_base64)"
                            " VALUES (%s, 'evidence_json', %s, %s)",
                            (run_id, f"{run_id}_evidence.json",
                             _json.dumps(evidence, ensure_ascii=False)),
                        )

        def get_artifact(self, run_id, kind):
            kind_map = {"markdown": "markdown_report", "evidence_json": "evidence_json"}
            lookup = kind_map.get(kind, kind)
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT kind, filename, content_base64 FROM research_artifacts"
                        " WHERE run_id=%s AND kind=%s",
                        (run_id, lookup),
                    )
                    row = cur.fetchone()
            return dict(row) if row else None

        def list_nonterminal_runs(self):
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT run_id, user_id, template_id, service_run_id,"
                        " reservation_id, estimate_micro, status, settings_json,"
                        " email_me, emailed, error, created_at FROM research_runs"
                        " WHERE status IN ('queued','running') ORDER BY created_at",
                    )
                    rows = cur.fetchall()
                # dict_row; settings_json kept raw — the sweeper only reads
            # service_run_id/reservation_id/status, not settings.
            return [dict(row) for row in rows]

    return _PostgresResearchStore(url)


_backend = _build_research_store()

if _backend != "sqlite":
    # Re-export the Postgres methods at module level so callers see the
    # same names as the SQLite section above (zero router changes).
    _pg = _backend
    def add_research_agent(user_id, template_id): return _pg.add_research_agent(user_id, template_id)
    def remove_research_agent(user_id, template_id): return _pg.remove_research_agent(user_id, template_id)
    def list_added_template_ids(user_id): return _pg.list_added_template_ids(user_id)
    def create_run(**kw): return _pg.create_run(**kw)
    def get_run(run_id, user_id): return _pg.get_run(run_id, user_id)
    def list_runs_for_user(user_id, limit=50): return _pg.list_runs_for_user(user_id, limit)
    def update_run_status(run_id, status, error=None, completed=False): return _pg.update_run_status(run_id, status, error, completed)
    def mark_emailed(run_id): return _pg.mark_emailed(run_id)
    def store_artifacts(run_id, artifacts, evidence, report_markdown): return _pg.store_artifacts(run_id, artifacts, evidence, report_markdown)
    def get_artifact(run_id, kind): return _pg.get_artifact(run_id, kind)
    def list_nonterminal_runs(): return _pg.list_nonterminal_runs()
