# Backtest Run Visibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make failed/interrupted backtests visible and durable (attempts journal + card/history/selector surfaces + LLM coverage disclosure), and fix the two launch-time visibility bugs (editor overlay never closes; `/app` HTML cached for an hour).

**Architecture:** Two independent PRs per the approved spec (`docs/superpowers/specs/2026-08-04-backtest-run-visibility-design.md`). PR-1 is frontend-only (vercel.json cache headers + editor close-on-run). PR-2 adds an additive `backtest_attempts` journal table to the run-history store (both twins), rides all new data on existing responses (no new routes), and renders failure + coverage states from the payload alone.

**Tech Stack:** FastAPI + SQLite/Postgres twins, vanilla JS frontend (no build step), pytest static-text guards for frontend contracts.

## Global Constraints

- **No new API routes.** All data rides existing responses. The three route-contract freeze guards must not need updating; if one reddens, the change is wrong.
- **Additive, skew-tolerant fields only** (issue #304): a new frontend against an old backend must render exactly today's UI. Frontend checks use `typeof` / explicit null checks, never truthiness on numerics (`Number(null) === 0` trap).
- **Literal DDL only** in both DB twins — `test_store_twin_parity.py` parses source text; an f-string collapses to a placeholder it cannot see through. `backtest_attempts` is a **new table**, so `CREATE TABLE IF NOT EXISTS` alone reaches deployed databases (the `ALTER ... IF NOT EXISTS` rule applies only to columns added to existing tables).
- **`print()`, never `logger.*`** for backend diagnostics — logger output is invisible under deployed uvicorn. Tests assert on `capsys`, not `caplog`.
- **Journal writes are best-effort**: wrapped in try/except, failures printed, never abort a launch or finalize.
- **Error text cap: 500 chars** at write; sanitization is reused from `_sanitize_backtest_error` output (never re-derive).
- **Coverage threshold is 0.95** — the H6 `MIN_LLM_DECISION_COVERAGE` constant. Frontend hardcodes `0.95` with a comment naming the constant.
- **Interrupted classification is read-side only** (D3): no startup sweep, no write that could clobber a genuine prod run from a dev process sharing `AGENT_RUNS_DATABASE_URL`.
- **Routes stay sync `def`** (the #292 threadpool convention) — this plan adds no routes, but do not convert any touched route to `async`.
- **Never `git add -A`** in this repo (committed seed DB mutates on import). Stage files by name.
- **Shared checkout hazard:** all work happens in dedicated worktrees, never by switching branches in `/mnt/d/github/agent-trading-lab`.
- Tests run from the worktree root: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/<file> -v`.
- Cache-buster floors at time of writing: `app.js?v=62`, `styles.css?v=79`, `js/agent-editor.js?v=22`. Each PR bumps only the files it touches, by +1 over whatever is current **at implementation time**, and updates the exact-string pins in `test_frontend_fast_boot.py` (lines ~174-177) plus any `agent-editor.js?v=` pin found by grepping the tests.

## PR structure

| PR | Branch (new worktree off latest `origin/main`) | Tasks | Content |
|----|---|---|---|
| PR-1 | `fix/backtest-launch-visibility` | 1–2 | vercel.json `/app` cache headers; editor close-on-run |
| PR-2 | `feat/backtest-attempts-journal` | 3–13 | journal + payload merges + frontend failure states + coverage chip |

PR-2 does not depend on PR-1 at the code level, but cut its branch **after PR-1 merges** so the `app.js` buster bump doesn't collide (the #295/#296 v=57 collision). PR bodies: short, per the repo convention; neither PR needs a merge gate — both are complete when opened.

---

### Task 1: `/app` cache headers (PR-1)

The HTML that carries the cache-busters is the one thing cached for an hour: `vercel.json`'s catch-all `/(.*)` sets `max-age=3600` and overrides exist for `/`, `/app.js`, `/styles.css` — but not `/app` or `/app.html` (spec Finding 5). Vercel applies every matching headers rule; on a duplicate key the **later** rule wins, so an `/app` rule overrides Cache-Control while the catch-all's CSP still applies (same mechanism `/app.js` uses today).

**Files:**
- Modify: `dashboard/frontend/vercel.json` (headers array, after the `/` entry)
- Test: `dashboard/backend/tests/test_vercel_cache_headers.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing code-visible; deploy behavior only.

- [ ] **Step 1: Write the failing test**

```python
"""Cache-header contract for the deployed frontend (vercel.json).

/app serves the buster-carrying HTML; a cached copy pins users to old JS for
up to an hour after every deploy (Finding 5, 2026-08-04 backtest-visibility
spec). Vercel applies all matching header rules with the last match winning
per key, so the catch-all CSP survives these overrides — assert both halves
so neither regresses silently.
"""

import json
from pathlib import Path

VERCEL = json.loads(
    (Path(__file__).resolve().parents[2] / "frontend" / "vercel.json")
    .read_text(encoding="utf-8")
)


def _cache_control(source: str):
    values = [
        h["value"]
        for entry in VERCEL["headers"]
        if entry["source"] == source
        for h in entry["headers"]
        if h["key"] == "Cache-Control"
    ]
    return values[-1] if values else None


def test_app_html_routes_must_revalidate():
    for source in ("/app", "/app.html"):
        assert _cache_control(source) == "public, max-age=0, must-revalidate", source


def test_existing_overrides_unchanged():
    assert _cache_control("/") == "public, max-age=0, must-revalidate"
    assert _cache_control("/app.js") == "public, max-age=0, must-revalidate"
    assert _cache_control("/styles.css") == "public, max-age=0, must-revalidate"
    assert _cache_control("/assets/(.*)") == "public, max-age=31536000, immutable"


def test_catch_all_keeps_csp():
    catch_all = next(e for e in VERCEL["headers"] if e["source"] == "/(.*)")
    assert any(h["key"] == "Content-Security-Policy" for h in catch_all["headers"])
```

- [ ] **Step 2: Run it — expect the first test to FAIL** (`_cache_control("/app") is None`)

Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_vercel_cache_headers.py -v`

- [ ] **Step 3: Add the two header rules** to `dashboard/frontend/vercel.json`, inserted into the `headers` array directly after the existing `"source": "/"` entry:

```json
    {
      "source": "/app",
      "headers": [
        {
          "key": "Cache-Control",
          "value": "public, max-age=0, must-revalidate"
        }
      ]
    },
    {
      "source": "/app.html",
      "headers": [
        {
          "key": "Cache-Control",
          "value": "public, max-age=0, must-revalidate"
        }
      ]
    },
```

- [ ] **Step 4: Run the test file again — all PASS.** Also validate the JSON parses: `~/atl-venv/bin/python -c "import json; json.load(open('dashboard/frontend/vercel.json'))"`

- [ ] **Step 5: Commit**

```bash
git add dashboard/frontend/vercel.json dashboard/backend/tests/test_vercel_cache_headers.py
git commit -m "fix(deploy): stop caching /app HTML for an hour"
```

Post-merge note for the PR body (one line): verification is **external-probe only** — after the Vercel deploy, `curl -sI https://<prod-host>/app | grep -i cache-control` must show `max-age=0`; the local file proves nothing about prod.

---

### Task 2: Editor close-on-run (PR-1)

The agent editor is a fullscreen `z-index:1200` overlay; `runBacktest()` navigates to My Agents underneath it and the user keeps staring at the settings page (spec Finding 4). Close it at the same optimistic point the modal closes, using the `agent-editor-open-run` pattern (`window.AgentEditor.close(true)` — `force=true` skips the unsaved-changes confirm, matching the existing handler at the `agent-editor-open-run` listener).

**Files:**
- Modify: `dashboard/frontend/app.js` (`async function runBacktest()`, the launch block near `closeRunBacktestModal()` / `prepareLiveBacktestView(...)` — currently ~line 6034)
- Modify: `dashboard/frontend/app.html` (bump `app.js?v=` by +1)
- Modify: `dashboard/backend/tests/test_frontend_fast_boot.py` (`test_cache_busters_bumped` exact-string pin)
- Test: `dashboard/backend/tests/test_backtest_launch_visibility.py` (create)

**Interfaces:**
- Consumes: `window.AgentEditor.close(force)` from `js/agent-editor.js` (existing).
- Produces: nothing later tasks rely on.

- [ ] **Step 1: Write the failing guard test**

```python
"""Launch-time visibility guards (PR-1 of the 2026-08-04 backtest spec)."""

from dashboard.backend.tests._frontend_source import fn_body


def test_run_backtest_closes_editor_overlay_before_navigating():
    """A run launched from inside the agent editor must close the overlay.

    The editor is position:fixed inset:0 z-index:1200; navigateToPage()
    repaints My Agents underneath it, invisibly (spec Finding 4).
    """
    body = fn_body("async function runBacktest")
    close_at = body.index("window.AgentEditor.close(true)")
    navigate_at = body.index("navigateToPage('playground'")
    assert close_at < navigate_at
```

- [ ] **Step 2: Run it — expect FAIL** (`ValueError: substring not found`)

Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_backtest_launch_visibility.py -v`

- [ ] **Step 3: Add the close call** in `app.js`. Find this existing block inside `runBacktest()`:

```javascript
    // Pin live view BEFORE navigateToPage → showPlaygroundPanel → loadData(),
    // otherwise the async history load paints the previous run over the chart.
    closeRunBacktestModal();
```

and change it to:

```javascript
    // Pin live view BEFORE navigateToPage → showPlaygroundPanel → loadData(),
    // otherwise the async history load paints the previous run over the chart.
    closeRunBacktestModal();
    // The agent editor is a fullscreen overlay (z-index 1200) and the run modal
    // sits above it — without this, a run launched from inside the editor
    // repaints My Agents invisibly underneath the settings page.
    if (window.AgentEditor?.close) window.AgentEditor.close(true);
```

- [ ] **Step 4: Bump the buster.** In `app.html` change `app.js?v=62` to `app.js?v=63` (or current+1 if main has moved). Update the pin in `test_frontend_fast_boot.py::test_cache_busters_bumped` to the same value, and extend its comment with one line: `# v=63: editor close-on-run (backtest launch visibility PR-1)`.

- [ ] **Step 5: Run the new test file + fast-boot + full suite**

Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_backtest_launch_visibility.py dashboard/backend/tests/test_frontend_fast_boot.py -v` → PASS, then the full suite `~/atl-venv/bin/python -m pytest dashboard/backend/tests/ -q` → green.

- [ ] **Step 6: Commit, push, open PR-1**

```bash
git add dashboard/frontend/app.js dashboard/frontend/app.html \
        dashboard/backend/tests/test_backtest_launch_visibility.py \
        dashboard/backend/tests/test_frontend_fast_boot.py
git commit -m "fix(ux): close agent editor when a backtest launches"
```

Open the PR titled `fix(ux): backtest launch visibility (editor overlay + /app cache)` with a 3-line body: the two findings, one line each, plus the external-probe verification note from Task 1.

---

### Task 3: `backtest_attempts` store — SQLite twin (PR-2)

The journal lives in the **run-history store** (`database.py` / `database_postgres.py`, selected by `AGENT_RUNS_DATABASE_URL`) because its rows join `agent_runs.run_id` and must survive Render's SQLite reset exactly when run history does. The twins are already registered in `test_store_twin_parity.py` (`BacktestDatabase` / `PostgresBacktestDatabase`), so parity coverage is automatic once both carry the methods.

**Files:**
- Modify: `dashboard/backend/database.py` (`_init_database` — add DDL before the final `conn.commit()`; new methods after `insert_run`)
- Test: `dashboard/backend/tests/test_backtest_attempts_db.py` (create)

**Interfaces:**
- Consumes: existing `BacktestDatabase._get_connection()` (row_factory dict-able rows).
- Produces (identical signatures required on the Postgres twin in Task 4):
  - `insert_attempt(run_id: str, session_id: str, *, agent_id=None, agent_name=None, start_date=None, end_date=None, params=None, timeout_seconds=None) -> None`
  - `finalize_attempt(run_id: str, status: str, *, error=None, session_id=None) -> None`
  - `get_attempts_for_session(session_id: str, limit: int = 50) -> List[Dict]`
  - `get_latest_attempt_for_agents(agent_ids: List[str]) -> Dict[str, Dict]`
  - Row dict keys: `run_id, agent_id, session_id, agent_name, start_date, end_date, params_json, status, error, timeout_seconds, created_at, finished_at`. `status` values written by the store: `'running' | 'completed' | 'failed'` (`'interrupted'` exists only read-side, Task 5).

- [ ] **Step 1: Write the failing tests**

```python
"""backtest_attempts journal — store-level lifecycle (SQLite twin).

The journal is the only durable record of a failed backtest: agent_runs rows
are written only after a run completes (spec Finding 1), so these rows must
exist from launch and survive an error path that never reaches insert_run.
"""

import uuid

from dashboard.backend.database import BacktestDatabase


def _db(tmp_path):
    return BacktestDatabase(db_path=tmp_path / "attempts.db")


def _insert(db, run_id, session_id="sess-1", agent_id="agent-1", **kw):
    db.insert_attempt(
        run_id,
        session_id,
        agent_id=agent_id,
        agent_name=kw.get("agent_name", "My Agent"),
        start_date=kw.get("start_date", "2026-05-01"),
        end_date=kw.get("end_date", "2026-05-07"),
        params=kw.get("params", {"data_source": "alpaca", "initial_capital": 10000}),
        timeout_seconds=kw.get("timeout_seconds", 1800),
    )


def test_insert_then_finalize_failed_round_trips(tmp_path):
    db = _db(tmp_path)
    run_id = f"agent_test_{uuid.uuid4().hex[:8]}"
    _insert(db, run_id)

    rows = db.get_attempts_for_session("sess-1")
    assert [r["run_id"] for r in rows] == [run_id]
    row = rows[0]
    assert row["status"] == "running"
    assert row["error"] is None
    assert row["timeout_seconds"] == 1800
    assert row["created_at"]
    assert row["finished_at"] is None

    db.finalize_attempt(run_id, "failed", error="Backtest failed with return code 1. quota exceeded")
    row = db.get_attempts_for_session("sess-1")[0]
    assert row["status"] == "failed"
    assert "quota exceeded" in row["error"]
    assert row["finished_at"]


def test_finalize_completed_clears_error(tmp_path):
    db = _db(tmp_path)
    _insert(db, "run-ok")
    db.finalize_attempt("run-ok", "completed")
    row = db.get_attempts_for_session("sess-1")[0]
    assert row["status"] == "completed"
    assert row["error"] is None


def test_finalize_without_prior_insert_upserts_terminal_row(tmp_path):
    """The failure record must survive even when the launch-time insert failed."""
    db = _db(tmp_path)
    db.finalize_attempt("run-orphan", "failed", error="boom", session_id="sess-2")
    rows = db.get_attempts_for_session("sess-2")
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["error"] == "boom"
    assert rows[0]["created_at"] and rows[0]["finished_at"]


def test_finalize_without_insert_and_without_session_is_a_noop(tmp_path):
    db = _db(tmp_path)
    db.finalize_attempt("run-unknown", "failed", error="boom")  # must not raise
    assert db.get_attempts_for_session("sess-1") == []


def test_error_capped_at_500_chars(tmp_path):
    db = _db(tmp_path)
    _insert(db, "run-long")
    db.finalize_attempt("run-long", "failed", error="x" * 2000)
    assert len(db.get_attempts_for_session("sess-1")[0]["error"]) == 500


def test_get_attempts_for_session_orders_newest_first_and_limits(tmp_path):
    db = _db(tmp_path)
    for i in range(5):
        _insert(db, f"run-{i}")
    rows = db.get_attempts_for_session("sess-1", limit=3)
    assert len(rows) == 3
    created = [r["created_at"] for r in rows]
    assert created == sorted(created, reverse=True)


def test_get_latest_attempt_for_agents_is_batched_by_agent(tmp_path):
    db = _db(tmp_path)
    _insert(db, "run-a1-old", agent_id="agent-a")
    _insert(db, "run-a1-new", agent_id="agent-a")
    _insert(db, "run-b1", agent_id="agent-b", session_id="sess-b")
    db.finalize_attempt("run-a1-new", "failed", error="boom")

    latest = db.get_latest_attempt_for_agents(["agent-a", "agent-b", "agent-missing"])
    assert set(latest) == {"agent-a", "agent-b"}
    # CURRENT_TIMESTAMP has 1-second resolution, so the two agent-a rows tie on
    # created_at; the store must break ties by insertion order (rowid DESC) for
    # this to hold deterministically.
    assert latest["agent-a"]["run_id"] == "run-a1-new"
    assert latest["agent-b"]["run_id"] == "run-b1"
    assert db.get_latest_attempt_for_agents([]) == {}
```

- [ ] **Step 2: Run — expect FAIL** (`AttributeError: insert_attempt`)

Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_backtest_attempts_db.py -v`

- [ ] **Step 3: Implement.** In `_init_database`, before the final `conn.commit()` (after the `run_manifest` block), add — literal DDL, house SQLite style:

```python
        # backtest_attempts: launch-time journal for frontend backtests.
        # One row per POST /backtest/run; the only durable record of a run
        # that failed before insert_run (2026-08-04 backtest-visibility spec).
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_attempts (
                run_id TEXT PRIMARY KEY,
                agent_id TEXT,
                session_id TEXT NOT NULL,
                agent_name TEXT,
                start_date TEXT,
                end_date TEXT,
                params_json TEXT,
                status TEXT NOT NULL,
                error TEXT,
                timeout_seconds INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                finished_at TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_backtest_attempts_agent
            ON backtest_attempts(agent_id, created_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_backtest_attempts_session
            ON backtest_attempts(session_id, created_at)
        """)
```

Methods, placed after `insert_run` (module constant near the top of the class file, next to other constants):

```python
ATTEMPT_ERROR_MAX_CHARS = 500
```

```python
    # ------------------------------------------------------------------
    # backtest_attempts journal (2026-08-04 backtest-visibility spec)
    # ------------------------------------------------------------------

    def insert_attempt(self, run_id: str, session_id: str, *,
                       agent_id: Optional[str] = None,
                       agent_name: Optional[str] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       params: Optional[Dict[str, Any]] = None,
                       timeout_seconds: Optional[int] = None) -> None:
        """Record a launched frontend backtest as 'running'."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO backtest_attempts
            (run_id, agent_id, session_id, agent_name, start_date, end_date,
             params_json, status, timeout_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?)
        """, (run_id, agent_id, session_id, agent_name, start_date, end_date,
              json.dumps(params) if params is not None else None,
              timeout_seconds))
        conn.commit()
        conn.close()

    def finalize_attempt(self, run_id: str, status: str, *,
                         error: Optional[str] = None,
                         session_id: Optional[str] = None) -> None:
        """Mark an attempt terminal ('completed'/'failed').

        If the launch-time insert never landed, upsert a minimal terminal row
        (needs ``session_id`` — NOT NULL) so the failure record survives; a
        missing row with no session is a no-op, never an error.
        """
        error_text = str(error)[:ATTEMPT_ERROR_MAX_CHARS] if error else None
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE backtest_attempts
            SET status = ?, error = ?, finished_at = CURRENT_TIMESTAMP
            WHERE run_id = ?
        """, (status, error_text, run_id))
        if cursor.rowcount == 0 and session_id:
            cursor.execute("""
                INSERT OR REPLACE INTO backtest_attempts
                (run_id, session_id, status, error, finished_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (run_id, session_id, status, error_text))
        conn.commit()
        conn.close()

    def get_attempts_for_session(self, session_id: str,
                                 limit: int = 50) -> List[Dict]:
        """Newest-first attempts for a session (the history-merge read)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM backtest_attempts
            WHERE session_id = ?
            ORDER BY created_at DESC, rowid DESC
            LIMIT ?
        """, (session_id, limit))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_latest_attempt_for_agents(self, agent_ids: List[str]) -> Dict[str, Dict]:
        """Latest attempt per agent, one query (My Agents list is the hot path)."""
        wanted = [a for a in agent_ids if a]
        if not wanted:
            return {}
        conn = self._get_connection()
        cursor = conn.cursor()
        placeholders = ",".join("?" * len(wanted))
        cursor.execute(f"""
            SELECT * FROM backtest_attempts
            WHERE agent_id IN ({placeholders})
            ORDER BY created_at DESC, rowid DESC
        """, wanted)
        rows = cursor.fetchall()
        conn.close()
        latest: Dict[str, Dict] = {}
        for row in rows:
            record = dict(row)
            latest.setdefault(record["agent_id"], record)
        return latest
```

- [ ] **Step 4: Run the file — all PASS.** Also run the parity guard (it must still pass — the Postgres twin lacks the methods until Task 4, so **expect it to FAIL now**; that failure is Task 4's failing test): `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_store_twin_parity.py -v -k Backtest`

- [ ] **Step 5: Commit.** The twin-parity guard is red between this commit and Task 4's — acceptable on a feature branch (CI runs only on the PR, by which time Task 4 has landed); proceed to Task 4 immediately:

```bash
git add dashboard/backend/database.py dashboard/backend/tests/test_backtest_attempts_db.py
git commit -m "feat(backtest): backtest_attempts journal store (sqlite)"
```

---

### Task 4: `backtest_attempts` store — Postgres twin (PR-2)

**Files:**
- Modify: `dashboard/backend/database_postgres.py` (`_init_schema` DDL; methods after `insert_run`)
- Test: `dashboard/backend/tests/test_store_twin_parity.py` (already covers the pair — no edit needed) and `dashboard/backend/tests/test_backtest_db_postgres.py` (append one `@pg_only` behavioral test)

**Interfaces:**
- Consumes: the exact signatures produced by Task 3.
- Produces: the same four methods on `PostgresBacktestDatabase`.

- [ ] **Step 1: The failing test already exists** — `test_store_twin_parity.py` red on the missing methods. Confirm:

Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_store_twin_parity.py -v -k Backtest` → FAIL

- [ ] **Step 2: Add DDL** in `_init_schema`, after the `run_manifest` CREATE and before the index block (new table → CREATE alone reaches the deployed Neon DB; the "ADDING A COLUMN LATER?" comment's ALTER rule applies only to existing tables):

```python
                # backtest_attempts: launch-time journal for frontend
                # backtests (2026-08-04 backtest-visibility spec). New table,
                # so CREATE IF NOT EXISTS alone reaches deployed databases.
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS backtest_attempts (
                        run_id TEXT PRIMARY KEY,
                        agent_id TEXT,
                        session_id TEXT NOT NULL,
                        agent_name TEXT,
                        start_date TEXT,
                        end_date TEXT,
                        params_json TEXT,
                        status TEXT NOT NULL,
                        error TEXT,
                        timeout_seconds INTEGER,
                        created_at TEXT NOT NULL {created_at_default},
                        finished_at TEXT
                    )
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_backtest_attempts_agent "
                    "ON backtest_attempts(agent_id, created_at)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_backtest_attempts_session "
                    "ON backtest_attempts(session_id, created_at)"
                )
```

(`created_at_default` is the existing local variable producing SQLite's exact `CURRENT_TIMESTAMP` string format — same pattern as every other table in this file. The parity guard compares column *names*, which the f-string interpolation of the default does not hide.)

- [ ] **Step 3: Add the methods** after `insert_run` (Postgres has no rowid; `ctid` is not ordering-stable, so tie-break on `run_id DESC` — ties within one second are cosmetic, both twins just need a *deterministic* order):

```python
    # ------------------------------------------------------------------
    # backtest_attempts journal (2026-08-04 backtest-visibility spec)
    # ------------------------------------------------------------------

    def insert_attempt(self, run_id: str, session_id: str, *,
                       agent_id: Optional[str] = None,
                       agent_name: Optional[str] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       params: Optional[Dict[str, Any]] = None,
                       timeout_seconds: Optional[int] = None) -> None:
        """Record a launched frontend backtest as 'running'."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO backtest_attempts
                    (run_id, agent_id, session_id, agent_name, start_date,
                     end_date, params_json, status, timeout_seconds)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 'running', %s)
                    ON CONFLICT (run_id) DO NOTHING
                    """,
                    (run_id, agent_id, session_id, agent_name, start_date,
                     end_date,
                     json.dumps(params) if params is not None else None,
                     timeout_seconds),
                )

    def finalize_attempt(self, run_id: str, status: str, *,
                         error: Optional[str] = None,
                         session_id: Optional[str] = None) -> None:
        """Mark an attempt terminal; upsert a minimal row if insert never landed."""
        from dashboard.backend.database import ATTEMPT_ERROR_MAX_CHARS

        error_text = str(error)[:ATTEMPT_ERROR_MAX_CHARS] if error else None
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE backtest_attempts
                    SET status = %s, error = %s,
                        finished_at = to_char(now() AT TIME ZONE 'utc',
                                              'YYYY-MM-DD HH24:MI:SS')
                    WHERE run_id = %s
                    """,
                    (status, error_text, run_id),
                )
                if cur.rowcount == 0 and session_id:
                    cur.execute(
                        """
                        INSERT INTO backtest_attempts
                        (run_id, session_id, status, error, finished_at)
                        VALUES (%s, %s, %s, %s,
                                to_char(now() AT TIME ZONE 'utc',
                                        'YYYY-MM-DD HH24:MI:SS'))
                        ON CONFLICT (run_id) DO NOTHING
                        """,
                        (run_id, session_id, status, error_text),
                    )

    def get_attempts_for_session(self, session_id: str,
                                 limit: int = 50) -> List[Dict]:
        """Newest-first attempts for a session (the history-merge read)."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM backtest_attempts
                    WHERE session_id = %s
                    ORDER BY created_at DESC, run_id DESC
                    LIMIT %s
                    """,
                    (session_id, limit),
                )
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def get_latest_attempt_for_agents(self, agent_ids: List[str]) -> Dict[str, Dict]:
        """Latest attempt per agent, one query (My Agents list is the hot path)."""
        wanted = [a for a in agent_ids if a]
        if not wanted:
            return {}
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM backtest_attempts
                    WHERE agent_id = ANY(%s)
                    ORDER BY created_at DESC, run_id DESC
                    """,
                    (wanted,),
                )
                rows = cur.fetchall()
        latest: Dict[str, Dict] = {}
        for row in rows:
            record = dict(row)
            latest.setdefault(record["agent_id"], record)
        return latest
```

- [ ] **Step 4: Append one `@pg_only` round-trip test** to `test_backtest_db_postgres.py`, using its existing `pg_backtest_db` fixture and `pg_only` marker:

```python
@pg_only
def test_attempt_lifecycle_round_trips_postgres(pg_backtest_db):
    """insert → running; finalize → failed; orphan finalize upserts."""
    pg_backtest_db.insert_attempt(
        "att-pg-1", "sess-pg",
        agent_id="agent-pg", agent_name="PG Agent",
        start_date="2026-05-01", end_date="2026-05-07",
        params={"initial_capital": 10000}, timeout_seconds=1800,
    )
    rows = pg_backtest_db.get_attempts_for_session("sess-pg")
    assert rows[0]["status"] == "running" and rows[0]["finished_at"] is None

    pg_backtest_db.finalize_attempt("att-pg-1", "failed", error="e" * 900)
    row = pg_backtest_db.get_attempts_for_session("sess-pg")[0]
    assert row["status"] == "failed"
    assert len(row["error"]) == 500 and row["finished_at"]

    pg_backtest_db.finalize_attempt("att-pg-2", "failed", error="boom",
                                    session_id="sess-pg")
    run_ids = {r["run_id"] for r in pg_backtest_db.get_attempts_for_session("sess-pg")}
    assert "att-pg-2" in run_ids

    latest = pg_backtest_db.get_latest_attempt_for_agents(["agent-pg"])
    assert latest["agent-pg"]["run_id"] == "att-pg-1"
```

Check the fixture's cleanup convention in that file first (whether it truncates tables between tests) and add `backtest_attempts` to any table-cleanup list it maintains — otherwise reruns against a persistent test DB collide on the PK.

- [ ] **Step 5: Run parity + local suite** (the `@pg_only` test skips locally — that is the documented fail-open tier; it runs when the draft PR's CI executes, and the executor must later confirm it in the CI log by the skip count):

Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_store_twin_parity.py dashboard/backend/tests/test_backtest_db_postgres.py dashboard/backend/tests/test_backtest_attempts_db.py -v` → parity PASS, pg tests SKIP locally.

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/database_postgres.py dashboard/backend/tests/test_backtest_db_postgres.py
git commit -m "feat(backtest): backtest_attempts journal store (postgres twin)"
```

---

### Task 5: Read-side presentation helpers (PR-2)

D3: a `running` row older than its own `timeout_seconds` + a 10-minute grace is *presented* as `interrupted` — no write-side sweep. One module owns the rule so every reader agrees.

**Files:**
- Create: `dashboard/backend/domain/backtesting/attempts.py`
- Test: `dashboard/backend/tests/test_backtest_attempts_presentation.py` (create)

**Interfaces:**
- Consumes: raw journal row dicts (Task 3 shape).
- Produces (used by Tasks 6–8):
  - `present_attempt(row: Dict, *, now: Optional[datetime] = None) -> Dict` — copy of the row; stale `running` becomes `status='interrupted'`, `error=INTERRUPTED_MESSAGE`.
  - `summarize_attempt(row: Dict) -> Dict` — the 5-key card payload `{run_id, status, error, created_at, finished_at}` (presented).
  - `attempt_as_run_entry(presented: Dict) -> Dict` — an `agent_runs`-shaped entry for history merges (keys: `run_id, agent_name, mode, start_date, end_date, initial_equity=0.0, num_trades=0, created_at, status, error`).
  - Constants: `INTERRUPTED_GRACE_SECONDS = 600`, `DEFAULT_ATTEMPT_TIMEOUT_SECONDS = 1800`, `INTERRUPTED_MESSAGE = "Backtest interrupted (server restarted)."`

- [ ] **Step 1: Write the failing tests**

```python
"""Read-side classification of backtest_attempts rows (spec decision D3)."""

from datetime import datetime, timedelta, timezone

from dashboard.backend.domain.backtesting.attempts import (
    DEFAULT_ATTEMPT_TIMEOUT_SECONDS,
    INTERRUPTED_GRACE_SECONDS,
    INTERRUPTED_MESSAGE,
    attempt_as_run_entry,
    present_attempt,
    summarize_attempt,
)


def _row(status="running", created_at="2026-08-04 10:00:00", timeout_seconds=1800):
    return {
        "run_id": "att-1", "agent_id": "agent-1", "session_id": "sess-1",
        "agent_name": "My Agent", "start_date": "2026-05-01",
        "end_date": "2026-05-07", "params_json": None, "status": status,
        "error": None, "timeout_seconds": timeout_seconds,
        "created_at": created_at, "finished_at": None,
    }


def _now(created="2026-08-04 10:00:00", plus_seconds=0):
    base = datetime.strptime(created, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return base + timedelta(seconds=plus_seconds)


def test_fresh_running_row_stays_running():
    presented = present_attempt(_row(), now=_now(plus_seconds=60))
    assert presented["status"] == "running"
    assert presented["error"] is None


def test_running_past_own_timeout_plus_grace_presents_interrupted():
    stale = 1800 + INTERRUPTED_GRACE_SECONDS + 1
    presented = present_attempt(_row(), now=_now(plus_seconds=stale))
    assert presented["status"] == "interrupted"
    assert presented["error"] == INTERRUPTED_MESSAGE


def test_running_within_timeout_plus_grace_stays_running():
    inside = 1800 + INTERRUPTED_GRACE_SECONDS - 1
    assert present_attempt(_row(), now=_now(plus_seconds=inside))["status"] == "running"


def test_missing_timeout_falls_back_to_default_budget():
    stale = DEFAULT_ATTEMPT_TIMEOUT_SECONDS + INTERRUPTED_GRACE_SECONDS + 1
    row = _row(timeout_seconds=None)
    assert present_attempt(row, now=_now(plus_seconds=stale))["status"] == "interrupted"


def test_terminal_rows_and_unparseable_timestamps_pass_through():
    assert present_attempt(_row(status="failed"))["status"] == "failed"
    assert present_attempt(_row(status="completed"))["status"] == "completed"
    weird = _row(created_at="not-a-time")
    assert present_attempt(weird, now=_now(plus_seconds=999999))["status"] == "running"


def test_present_attempt_does_not_mutate_its_input():
    row = _row()
    present_attempt(row, now=_now(plus_seconds=10**6))
    assert row["status"] == "running"


def test_summarize_attempt_is_the_five_key_card_payload():
    stale = 1800 + INTERRUPTED_GRACE_SECONDS + 1
    summary = summarize_attempt(_row(), now=_now(plus_seconds=stale))
    assert summary == {
        "run_id": "att-1", "status": "interrupted",
        "error": INTERRUPTED_MESSAGE,
        "created_at": "2026-08-04 10:00:00", "finished_at": None,
    }


def test_attempt_as_run_entry_shapes_a_history_row():
    presented = dict(_row(status="failed"), error="boom")
    entry = attempt_as_run_entry(presented)
    assert entry == {
        "run_id": "att-1", "agent_name": "My Agent", "mode": "backtest",
        "start_date": "2026-05-01", "end_date": "2026-05-07",
        "initial_equity": 0.0, "num_trades": 0,
        "created_at": "2026-08-04 10:00:00",
        "status": "failed", "error": "boom",
    }
    anonymous = attempt_as_run_entry(dict(presented, agent_name=None))
    assert anonymous["agent_name"] == "Agent"
```

- [ ] **Step 2: Run — expect FAIL** (module missing)

Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_backtest_attempts_presentation.py -v`

- [ ] **Step 3: Implement** `dashboard/backend/domain/backtesting/attempts.py`:

```python
"""Read-side presentation of backtest_attempts journal rows.

Interrupted classification lives here and nowhere else (spec D3): a 'running'
row older than its own subprocess budget plus a grace margin is *presented*
as interrupted — no writer ever stamps that state, so a dev process pointed
at the shared AGENT_RUNS_DATABASE_URL cannot clobber a genuine prod run.
"""
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Margin past the run's own subprocess timeout before a still-'running' row is
# presented as interrupted (covers finalize lag between subprocess exit and
# the journal write, plus a restart landing mid-write).
INTERRUPTED_GRACE_SECONDS = 600
# Rows that predate timeout_seconds, or whose insert dropped it.
DEFAULT_ATTEMPT_TIMEOUT_SECONDS = 1800

INTERRUPTED_MESSAGE = "Backtest interrupted (server restarted)."

_JOURNAL_TS_FORMAT = "%Y-%m-%d %H:%M:%S"


def _parse_journal_timestamp(value: Any) -> Optional[datetime]:
    """Journal timestamps are UTC CURRENT_TIMESTAMP strings on both twins."""
    if not value:
        return None
    text = str(value).replace("T", " ")[:19]
    try:
        return datetime.strptime(text, _JOURNAL_TS_FORMAT).replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def present_attempt(row: Dict[str, Any], *,
                    now: Optional[datetime] = None) -> Dict[str, Any]:
    """Return a copy of ``row`` with stale 'running' presented as interrupted."""
    presented = dict(row)
    if presented.get("status") != "running":
        return presented
    created = _parse_journal_timestamp(presented.get("created_at"))
    if created is None:
        return presented
    budget = presented.get("timeout_seconds") or DEFAULT_ATTEMPT_TIMEOUT_SECONDS
    current = now or datetime.now(timezone.utc)
    if (current - created).total_seconds() > budget + INTERRUPTED_GRACE_SECONDS:
        presented["status"] = "interrupted"
        presented["error"] = INTERRUPTED_MESSAGE
    return presented


def summarize_attempt(row: Dict[str, Any], *,
                      now: Optional[datetime] = None) -> Dict[str, Any]:
    """The agent-card payload: latest_backtest_attempt's five keys."""
    presented = present_attempt(row, now=now)
    return {
        key: presented.get(key)
        for key in ("run_id", "status", "error", "created_at", "finished_at")
    }


def attempt_as_run_entry(presented: Dict[str, Any]) -> Dict[str, Any]:
    """A journal row shaped like an agent_runs entry for history-list merges.

    initial_equity is 0.0 on purpose: the frontend's metrics guard
    (``!metrics.initial_equity``) then routes failed entries to the no-metrics
    path without a second status check.
    """
    return {
        "run_id": presented.get("run_id"),
        "agent_name": presented.get("agent_name") or "Agent",
        "mode": "backtest",
        "start_date": presented.get("start_date") or "",
        "end_date": presented.get("end_date") or "",
        "initial_equity": 0.0,
        "num_trades": 0,
        "created_at": presented.get("created_at") or "",
        "status": presented.get("status"),
        "error": presented.get("error"),
    }
```

- [ ] **Step 4: Run — all PASS.** Also run `test_architecture_boundaries.py` (new domain module must not trip it).

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/backtesting/attempts.py \
        dashboard/backend/tests/test_backtest_attempts_presentation.py
git commit -m "feat(backtest): read-side attempt presentation (interrupted rule)"
```

---

### Task 6: Journal lifecycle wiring in the backtests router (PR-2)

Insert `running` at launch (after minting `live_run_id`, before `thread.start()`); finalize in `run_backtest_background` on all three exits (returncode 0 → completed, non-zero → failed with the already-sanitized string, exception incl. timeout → failed). All best-effort.

**Files:**
- Modify: `dashboard/backend/api/routers/backtests.py` (endpoint block near the `live_run_id` mint ~line 1229; `run_backtest_background` terminal branches ~lines 550–575; two small helpers near `run_backtest_background`)
- Test: `dashboard/backend/tests/test_backtest_attempts_journal.py` (create — direct-call finalize tests; **not** in `test_backtests_router.py`, whose autouse fixture stubs `run_backtest_background`) and `dashboard/backend/tests/test_backtests_router.py` (append the endpoint-insert test — its autouse stub is exactly what keeps the thread inert)

**Interfaces:**
- Consumes: Task 3's `db.insert_attempt` / `db.finalize_attempt`; existing `_backtest_subprocess_timeout`, `_sanitize_backtest_error`, `agent_service`.
- Produces: journal rows for Tasks 7–8. No signature changes to `run_backtest_background`.

- [ ] **Step 1: Write the failing endpoint test** (append to `test_backtests_router.py`; the autouse `_reset_backtest_guards` fixture already stubs the background worker):

```python
def test_backtest_run_inserts_running_attempt_row():
    session = str(uuid.uuid4())
    resp = TestClient(app).post(
        "/backtest/run",
        json={"start_date": "2026-05-01", "end_date": "2026-05-07"},
        headers={"X-Session-Id": session},
    )
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]

    rows = bt.db.get_attempts_for_session(resp.json()["session_id"])
    assert [r["run_id"] for r in rows] == [run_id]
    row = rows[0]
    assert row["status"] == "running"
    assert row["start_date"] == "2026-05-01"
    assert row["end_date"] == "2026-05-07"
    assert row["timeout_seconds"] and row["timeout_seconds"] >= 1800


def test_backtest_run_journal_insert_failure_does_not_block_launch(monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("journal down")

    monkeypatch.setattr(bt.db, "insert_attempt", boom)
    resp = TestClient(app).post(
        "/backtest/run",
        json={"start_date": "2026-05-01", "end_date": "2026-05-07"},
        headers=_sess(),
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert "attempt journal insert failed" in capsys.readouterr().out
```

- [ ] **Step 2: Write the failing finalize tests** in the new `test_backtest_attempts_journal.py` (direct call, monkeypatched subprocess):

```python
"""run_backtest_background finalizes the attempt journal on every exit path."""

import subprocess
import types
import uuid

from dashboard.backend.api.routers import backtests as bt


def _launch(monkeypatch, fake_run):
    """Call the real worker with subprocess.run stubbed; return (run_id, session)."""
    monkeypatch.setattr(subprocess, "run", fake_run)
    run_id = f"agent_journal_{uuid.uuid4().hex[:8]}"
    session = f"sess-{uuid.uuid4().hex[:8]}"
    bt.db.insert_attempt(run_id, session, start_date="2026-05-01",
                         end_date="2026-05-02", timeout_seconds=1800)
    bt.run_backtest_background(
        start_date="2026-05-01",
        end_date="2026-05-02",
        session_id=session,
        live_run_id=run_id,
    )
    return run_id, session


def test_nonzero_returncode_finalizes_failed_with_sanitized_error(monkeypatch):
    fake = lambda *a, **k: types.SimpleNamespace(
        returncode=1, stdout="", stderr="provider quota exceeded"
    )
    run_id, session = _launch(monkeypatch, fake)
    row = next(r for r in bt.db.get_attempts_for_session(session)
               if r["run_id"] == run_id)
    assert row["status"] == "failed"
    assert "return code 1" in row["error"]
    assert row["finished_at"]


def test_zero_returncode_finalizes_completed(monkeypatch):
    fake = lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="ok", stderr="")
    run_id, session = _launch(monkeypatch, fake)
    row = next(r for r in bt.db.get_attempts_for_session(session)
               if r["run_id"] == run_id)
    assert row["status"] == "completed"
    assert row["error"] is None


def test_timeout_finalizes_failed(monkeypatch):
    def fake(*a, **k):
        raise subprocess.TimeoutExpired(cmd="x", timeout=1800)

    run_id, session = _launch(monkeypatch, fake)
    row = next(r for r in bt.db.get_attempts_for_session(session)
               if r["run_id"] == run_id)
    assert row["status"] == "failed"


def test_finalize_without_prior_insert_still_records_failure(monkeypatch):
    """The upsert path: launch insert never landed, failure must survive."""
    def fake(*a, **k):
        raise RuntimeError("engine blew up")

    monkeypatch.setattr(subprocess, "run", fake)
    run_id = f"agent_orphan_{uuid.uuid4().hex[:8]}"
    session = f"sess-{uuid.uuid4().hex[:8]}"
    bt.run_backtest_background(
        start_date="2026-05-01", end_date="2026-05-02",
        session_id=session, live_run_id=run_id,
    )
    rows = bt.db.get_attempts_for_session(session)
    assert rows and rows[0]["status"] == "failed"


def test_finalize_failure_never_raises(monkeypatch, capsys):
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    def boom(*a, **k):
        raise RuntimeError("journal down")

    monkeypatch.setattr(bt.db, "finalize_attempt", boom)
    bt.run_backtest_background(
        start_date="2026-05-01", end_date="2026-05-02",
        session_id="sess-x", live_run_id="run-x",
    )  # must not raise
    assert "attempt journal finalize failed" in capsys.readouterr().out
```

- [ ] **Step 3: Run both — expect FAIL** (no journal writes happen yet).

- [ ] **Step 4: Implement.** Two helpers next to `run_backtest_background`:

```python
def _journal_attempt_start(*, run_id, session_id, agent_id, start_date,
                           end_date, data_source, model, runtime_type,
                           initial_capital, timeout_seconds):
    """Best-effort launch-time journal write — must never block a launch."""
    try:
        agent_name = None
        if agent_id:
            try:
                agent_name = (agent_service.get_agent(agent_id) or {}).get("name")
            except Exception:
                agent_name = None
        db.insert_attempt(
            run_id,
            session_id,
            agent_id=agent_id,
            agent_name=agent_name,
            start_date=start_date,
            end_date=end_date,
            params={
                "data_source": data_source,
                "model": model,
                "runtime_type": runtime_type,
                "initial_capital": initial_capital,
            },
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        print(f"⚠️ attempt journal insert failed: {exc}", flush=True)


def _journal_attempt_final(run_id, status, *, error=None, session_id=None):
    """Best-effort terminal journal write — must never mask the run's outcome."""
    if not run_id:
        return
    try:
        db.finalize_attempt(run_id, status, error=error, session_id=session_id)
    except Exception as exc:
        print(f"⚠️ attempt journal finalize failed: {exc}", flush=True)
```

(If `agent_service` is not already imported at module scope in `backtests.py`, it is — `_resolve_backtest_session` uses it; do not add a second import.)

**Endpoint wiring** — in `run_backtest_endpoint`, directly after the `live_run_id` mint (before the `backtest_status` publication block):

```python
    _journal_attempt_start(
        run_id=live_run_id,
        session_id=session_id,
        agent_id=agent_id,
        start_date=start_date,
        end_date=end_date,
        data_source=data_source,
        model=model,
        runtime_type=runtime_type,
        initial_capital=initial_capital,
        timeout_seconds=_backtest_subprocess_timeout(
            runtime_type, start_date, end_date
        ),
    )
```

Use the **same variable names the thread kwargs use** (the `kwargs={...}` dict a few lines below is the authority — `agent_id` there is the one that reaches the worker; do not substitute a query-only variant).

**Worker wiring** in `run_backtest_background` — three edits, reusing the already-computed sanitized strings verbatim:

In the `result.returncode != 0` branch, after `backtest_status["error"] = ...`:

```python
            _journal_attempt_final(
                live_run_id,
                "failed",
                error=f"Backtest failed with return code {result.returncode}. {summary}",
                session_id=session_id,
            )
```

In the `else` (success) branch, after the `runs_count` bookkeeping:

```python
            _journal_attempt_final(live_run_id, "completed", session_id=session_id)
```

In the `except Exception as e:` branch, after `backtest_status["error"] = summary`:

```python
        _journal_attempt_final(
            live_run_id, "failed", error=summary, session_id=session_id
        )
```

- [ ] **Step 5: Run both test files — PASS**, then the whole router file: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_backtests_router.py dashboard/backend/tests/test_backtest_attempts_journal.py -v`

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/api/routers/backtests.py \
        dashboard/backend/tests/test_backtest_attempts_journal.py \
        dashboard/backend/tests/test_backtests_router.py
git commit -m "feat(backtest): journal attempt lifecycle at launch/finalize"
```

---

### Task 7: `/api/backtest/runs` merge + `RunMetadata.status`/`error` (PR-2)

Every row reaching the list route carries `status` so consumers switch on one field: `agent_runs` rows are `"completed"`; failed/interrupted attempts are merged in (dedup by `run_id`), newest first. Running attempts are **not** merged (the live run is the poll loop's job).

**Files:**
- Modify: `dashboard/backend/api/routers/backtests.py` (`RunMetadata` model, `_run_metadata_response`, `get_backtest_runs`; import from `attempts.py`)
- Test: `dashboard/backend/tests/test_backtests_router.py` (append)

**Interfaces:**
- Consumes: Task 3 store reads; Task 5 `present_attempt` / `attempt_as_run_entry`.
- Produces: `RunMetadata` gains `status: Optional[str] = None` and `error: Optional[str] = None`; `/api/backtest/runs` entries all carry `status`.

- [ ] **Step 1: Write the failing tests** (append to `test_backtests_router.py`):

```python
def _seed_completed_run(session_id, run_id="run-done"):
    bt.db.insert_run(
        run_id=run_id, session_id=session_id, agent_name="Agent",
        mode="backtest", start_date="2026-05-01", end_date="2026-05-07",
        initial_equity=10000, final_equity=10100, total_return=0.01,
    )


def test_backtest_runs_marks_completed_and_merges_failed_attempts():
    session = str(uuid.uuid4())
    _seed_completed_run(session)
    bt.db.insert_attempt("att-fail", session, agent_id="agent-1",
                         agent_name="Agent", start_date="2026-05-08",
                         end_date="2026-05-09", timeout_seconds=1800)
    bt.db.finalize_attempt("att-fail", "failed", error="quota exceeded")

    resp = TestClient(app).get(
        "/api/backtest/runs", headers={"X-Session-Id": session}
    )
    assert resp.status_code == 200
    by_id = {r["run_id"]: r for r in resp.json()}
    assert by_id["run-done"]["status"] == "completed"
    assert by_id["att-fail"]["status"] == "failed"
    assert by_id["att-fail"]["error"] == "quota exceeded"
    assert by_id["att-fail"]["initial_equity"] == 0.0


def test_backtest_runs_dedups_attempts_by_run_id_and_skips_running():
    session = str(uuid.uuid4())
    _seed_completed_run(session, run_id="run-shared")
    # completed attempt for the same run_id: its agent_runs row already appears
    bt.db.insert_attempt("run-shared", session, timeout_seconds=1800)
    bt.db.finalize_attempt("run-shared", "completed")
    # fresh running attempt: the live poll loop's job, not history's
    bt.db.insert_attempt("att-live", session, timeout_seconds=1800)

    body = TestClient(app).get(
        "/api/backtest/runs", headers={"X-Session-Id": session}
    ).json()
    run_ids = [r["run_id"] for r in body]
    assert run_ids.count("run-shared") == 1
    assert "att-live" not in run_ids


def test_backtest_runs_journal_read_failure_degrades_to_runs_only(monkeypatch, capsys):
    session = str(uuid.uuid4())
    _seed_completed_run(session)

    def boom(*a, **k):
        raise RuntimeError("journal down")

    monkeypatch.setattr(bt.db, "get_attempts_for_session", boom)
    resp = TestClient(app).get(
        "/api/backtest/runs", headers={"X-Session-Id": session}
    )
    assert resp.status_code == 200
    assert [r["run_id"] for r in resp.json()] == ["run-done"]
    assert "attempt journal read failed" in capsys.readouterr().out
```

- [ ] **Step 2: Run — expect FAIL** (`status` missing / KeyError).

- [ ] **Step 3: Implement.**

Model — append to `RunMetadata` after `t1_deferred_shares`:

```python
    # Attempt-journal presentation (2026-08-04 spec): every entry on the list
    # routes carries a status so consumers switch on one field. agent_runs rows
    # are always "completed"; merged journal rows are "failed"/"interrupted"
    # and carry the sanitized error. Optional so cached clients tolerate absence.
    status: Optional[str] = None
    error: Optional[str] = None
```

`_run_metadata_response` — after `payload["data_source"] = data_source or ALPACA` add:

```python
    payload["status"] = "completed"
```

Imports at the top of `backtests.py`:

```python
from dashboard.backend.domain.backtesting.attempts import (
    attempt_as_run_entry,
    present_attempt,
)
```

Route:

```python
@router.get("/api/backtest/runs", response_model=List[RunMetadata])
def get_backtest_runs(request: Request):
    """Completed runs plus failed/interrupted attempts for this session."""
    session_id = get_session_id_from_request(request)
    runs = db.get_runs_by_session(session_id)
    runs = [r for r in runs if r['mode'] == 'backtest']
    entries = [_run_metadata_response(run) for run in runs]
    known = {r['run_id'] for r in runs}
    try:
        attempts = db.get_attempts_for_session(session_id)
    except Exception as exc:
        print(f"⚠️ attempt journal read failed: {exc}", flush=True)
        attempts = []
    for attempt in attempts:
        presented = present_attempt(attempt)
        if presented.get("status") not in ("failed", "interrupted"):
            continue
        if presented.get("run_id") in known:
            continue
        entries.append(RunMetadata(**attempt_as_run_entry(presented)))
    entries.sort(key=lambda e: e.created_at or "", reverse=True)
    return entries
```

- [ ] **Step 4: Run the file — PASS.** Then the route-freeze guards must be untouched-green: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/ -q -k "route or freeze or contract"` (whatever matches; a red here means a route changed, which is a plan violation, not a test to update).

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/api/routers/backtests.py dashboard/backend/tests/test_backtests_router.py
git commit -m "feat(backtest): surface failed attempts on /api/backtest/runs"
```

---

### Task 8: Agents payload — `latest_backtest_attempt` + editor-history merge (PR-2)

Cards read `latest_backtest_attempt` (batched — the My Agents list is the hot path, no N+1). The editor's run history additionally merges failed/interrupted attempts into `agent.runs` — but **only on the single-agent read path**, so listings never pay a per-agent journal read.

**Files:**
- Modify: `dashboard/backend/domain/agents/service.py` (`agent_with_stats` signature + body; the two batched list methods `list_builtin_agents_with_stats` / `list_agents_with_stats`; the single-agent read path — the method serving `GET /api/v1/agents/{agent_id}` whose body is `return self.attach_equity_sparklines([self.agent_with_stats(agent)])[0]`; locate every call site with `command grep -n "agent_with_stats(" dashboard/backend/domain/agents/service.py`)
- Test: `dashboard/backend/tests/test_agents_api.py` (append; follow that file's existing client/session conventions)

**Interfaces:**
- Consumes: Task 3 `get_latest_attempt_for_agents` / `get_attempts_for_session`; Task 5 `summarize_attempt` / `present_attempt` / `attempt_as_run_entry`.
- Produces:
  - `agent_with_stats(agent, *, session_runs=None, latest_attempts=None, include_attempt_history=False)`
  - every enriched agent carries `latest_backtest_attempt: dict|None` (5 keys, presented);
  - on `include_attempt_history=True`, `result["runs"]` entries all carry `status` (existing → `"completed"`) and failed/interrupted attempts are merged (dedup by `run_id`, newest first).

- [ ] **Step 1: Write the failing tests** (append to `test_agents_api.py`, adapting to its fixtures — it already creates agents over the API; use the service directly where simpler):

```python
def test_agents_listing_carries_latest_backtest_attempt(client):
    owner = str(uuid.uuid4())
    created = client.post(
        "/api/v1/agents",
        json={"name": "Attempt Agent", "agent_type": "builtin"},
        headers={"X-Session-Id": owner},
    ).json()
    agent_id = created["agent"]["agent_id"]
    session_id = created["session_id"]

    from dashboard.backend.database import db as backtest_db
    backtest_db.insert_attempt("att-card-1", session_id, agent_id=agent_id,
                               agent_name="Attempt Agent", timeout_seconds=1800)
    backtest_db.finalize_attempt("att-card-1", "failed", error="quota exceeded")

    listing = client.get("/agents/builtin").json()
    agent = next(a for a in listing["agents"] if a["agent_id"] == agent_id)
    attempt = agent["latest_backtest_attempt"]
    assert attempt["run_id"] == "att-card-1"
    assert attempt["status"] == "failed"
    assert attempt["error"] == "quota exceeded"
    assert set(attempt) == {"run_id", "status", "error", "created_at", "finished_at"}


def test_agents_without_attempts_carry_null_attempt(client):
    owner = str(uuid.uuid4())
    created = client.post(
        "/api/v1/agents",
        json={"name": "Clean Agent", "agent_type": "builtin"},
        headers={"X-Session-Id": owner},
    ).json()
    listing = client.get("/agents/builtin").json()
    agent = next(a for a in listing["agents"]
                 if a["agent_id"] == created["agent"]["agent_id"])
    assert agent["latest_backtest_attempt"] is None


def test_single_agent_read_merges_failed_attempts_into_runs(client):
    owner = str(uuid.uuid4())
    created = client.post(
        "/api/v1/agents",
        json={"name": "History Agent", "agent_type": "builtin"},
        headers={"X-Session-Id": owner},
    ).json()
    agent_id = created["agent"]["agent_id"]
    session_id = created["session_id"]

    from dashboard.backend.database import db as backtest_db
    backtest_db.insert_run(
        run_id="hist-ok", session_id=session_id, agent_name="History Agent",
        mode="backtest", start_date="2026-05-01", end_date="2026-05-07",
        initial_equity=10000, final_equity=10100,
    )
    backtest_db.insert_attempt("hist-fail", session_id, agent_id=agent_id,
                               timeout_seconds=1800)
    backtest_db.finalize_attempt("hist-fail", "failed", error="boom")

    agent = client.get(
        f"/api/v1/agents/{agent_id}", headers={"X-Session-Id": owner}
    ).json()["agent"]
    runs = {r["run_id"]: r for r in agent["runs"]}
    assert runs["hist-ok"]["status"] == "completed"
    assert runs["hist-fail"]["status"] == "failed"
    assert runs["hist-fail"]["error"] == "boom"
```

Adjust route paths/response shapes to what `test_agents_api.py` already asserts (e.g. whether `/agents/builtin` wraps in `{"agents": [...]}`) — copy an adjacent test's access pattern rather than inventing one.

- [ ] **Step 2: Run — expect FAIL** (KeyError `latest_backtest_attempt`).

- [ ] **Step 3: Implement** in `service.py`. Import at top:

```python
from dashboard.backend.domain.backtesting.attempts import (
    attempt_as_run_entry,
    present_attempt,
    summarize_attempt,
)
```

`agent_with_stats` — new keyword params and body additions before `return result`:

```python
    def agent_with_stats(
        self,
        agent: Dict[str, Any],
        *,
        session_runs: Optional[List[Dict[str, Any]]] = None,
        latest_attempts: Optional[Dict[str, Dict[str, Any]]] = None,
        include_attempt_history: bool = False,
    ) -> Dict[str, Any]:
```

```python
        # Attempt journal (2026-08-04 spec): the card's failure signal. Batched
        # callers prefetch latest_attempts; the single-agent path fetches its own.
        agent_id = agent.get("agent_id")
        if latest_attempts is None:
            latest_attempts = (
                self.db.get_latest_attempt_for_agents([agent_id]) if agent_id else {}
            )
        raw_attempt = latest_attempts.get(agent_id) if agent_id else None
        result["latest_backtest_attempt"] = (
            summarize_attempt(raw_attempt) if raw_attempt else None
        )

        if include_attempt_history and agent_id:
            merged = [
                {**run, "status": run.get("status") or "completed"}
                for run in result["runs"]
            ]
            known = {run.get("run_id") for run in merged}
            for attempt in self.db.get_attempts_for_session(agent["session_id"]):
                if attempt.get("agent_id") != agent_id:
                    continue
                presented = present_attempt(attempt)
                if presented.get("status") not in ("failed", "interrupted"):
                    continue
                if presented.get("run_id") in known:
                    continue
                merged.append(attempt_as_run_entry(presented))
            merged.sort(key=lambda r: r.get("created_at") or "", reverse=True)
            result["runs"] = merged
```

Wrap the journal reads in this block with try/except printing `⚠️ attempt journal read failed: {exc}` and degrading to `latest_backtest_attempt = None` / unmerged runs — same fail-open contract as Task 7.

Call sites:
- `list_builtin_agents_with_stats` and `list_agents_with_stats`: before the enrich loop add `latest_attempts = self.db.get_latest_attempt_for_agents([a.get("agent_id") for a in agents])` (guarded by the same try/except → `{}`), and pass `latest_attempts=latest_attempts` to each `agent_with_stats` call.
- The single-agent read method serving `GET /api/v1/agents/{agent_id}`: pass `include_attempt_history=True`.
- Every other `agent_with_stats(` call site: leave untouched (defaults are additive).

- [ ] **Step 4: Run — PASS**, plus `test_agents_api.py` end to end and `test_architecture_boundaries.py`.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/agents/service.py dashboard/backend/tests/test_agents_api.py
git commit -m "feat(agents): expose latest backtest attempt + failed run history"
```

---

### Task 9: Engine LLM coverage metadata (PR-2)

Disclose how much of a "successful" run the model actually drove (spec Finding 3 / D4). Written whenever the run was configured with a model — including a quota-dead key that never completed one call (coverage 0). Rule-based configs get **no keys** (absence ≠ 0%).

**Files:**
- Modify: `dashboard/backend/domain/backtesting/engine.py` (module-level helper; loop counter; the `insert_run` call site)
- Test: `dashboard/backend/tests/test_llm_coverage_metadata.py` (create)

**Interfaces:**
- Consumes: `manager.llm_decisions` (H6 counter — **do not add write sites**), `total_steps` (in scope at the call site), `runtime_invoked` per hosted step.
- Produces: `metadata` keys `llm_decisions: int`, `llm_total_steps: int`, `llm_decision_coverage: float 0-1` on `agent_runs.metadata`, consumed by Task 12.

- [ ] **Step 1: Write the failing tests**

```python
"""_llm_coverage_metadata: the H6-coverage disclosure written per run (D4)."""

from dashboard.backend.domain.backtesting.engine import _llm_coverage_metadata


def test_llm_config_with_zero_driven_steps_discloses_zero_coverage():
    """The quota-dead-key case: a 'successful' run the model never drove."""
    meta = _llm_coverage_metadata(model_configured=True, driven_steps=0, total_steps=40)
    assert meta == {
        "llm_decisions": 0,
        "llm_total_steps": 40,
        "llm_decision_coverage": 0.0,
    }


def test_partial_coverage_rounds_to_four_places():
    meta = _llm_coverage_metadata(model_configured=True, driven_steps=1, total_steps=3)
    assert meta["llm_decision_coverage"] == 0.3333


def test_full_coverage():
    meta = _llm_coverage_metadata(model_configured=True, driven_steps=40, total_steps=40)
    assert meta["llm_decision_coverage"] == 1.0


def test_rule_based_config_writes_no_keys():
    """Absence must not render as 0% coverage — no keys at all."""
    assert _llm_coverage_metadata(model_configured=False, driven_steps=0, total_steps=40) == {}


def test_zero_total_steps_writes_no_keys():
    assert _llm_coverage_metadata(model_configured=True, driven_steps=0, total_steps=0) == {}
```

- [ ] **Step 2: Run — expect FAIL** (import error).

- [ ] **Step 3: Implement.** Module-level function in `engine.py` (near the other module helpers):

```python
def _llm_coverage_metadata(*, model_configured: bool, driven_steps: int,
                           total_steps: int) -> Dict:
    """Per-run H6-coverage disclosure (spec D4).

    Keys are written whenever the run was configured with a model — a
    quota-dead key from step 1 must still yield coverage 0 — and omitted for
    rule-based configs, so absence never renders as 0%.
    """
    if not model_configured or not total_steps or total_steps <= 0:
        return {}
    driven = max(0, int(driven_steps))
    return {
        "llm_decisions": driven,
        "llm_total_steps": int(total_steps),
        "llm_decision_coverage": round(driven / total_steps, 4),
    }
```

Loop counter — where `llm_calls_count = 0` is initialized (before the hourly loop), add:

```python
        runtime_driven_steps = 0
```

and in the hosted-runtime branch, immediately after `runtime_invoked = self.runtime_dispatcher.calls > runtime_calls_before`:

```python
                if runtime_invoked:
                    runtime_driven_steps += 1
```

Call site — replace `metadata=self._agent_run_metadata(),` in the agent `insert_run` call with a prepared local (placed just above the `db.insert_run(` call):

```python
        run_metadata = self._agent_run_metadata()
        # Pipeline runs: the H6 counter (manager.llm_decisions, success-exit
        # only). Hosted runs: steps the runtime dispatcher actually drove —
        # held/fallback steps don't count.
        if self.runtime_type == PIPELINE_RUNTIME_TYPE:
            model_configured = bool(self.use_llm)
            driven_steps = manager.llm_decisions
        else:
            model_configured = True
            driven_steps = runtime_driven_steps
        run_metadata.update(_llm_coverage_metadata(
            model_configured=model_configured,
            driven_steps=driven_steps,
            total_steps=total_steps,
        ))
```

and pass `metadata=run_metadata`. Only the **agent** `insert_run` call changes — the baseline `insert_run` calls (buy-hold etc.) are untouched.

- [ ] **Step 4: Run** the new file + the engine's existing tests: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_llm_coverage_metadata.py -v` and the full backend suite stays green.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/backtesting/engine.py dashboard/backend/tests/test_llm_coverage_metadata.py
git commit -m "feat(backtest): persist per-run LLM decision coverage"
```

---

### Task 10: Frontend — failed-card notice + dismissal (PR-2)

The card derives its failed state from the agents payload alone (D5, payload-as-source-of-truth): any visit to My Agents from any device shows it, no live poll required. One derivation function; dismissal is localStorage by `run_id` (D7).

**Files:**
- Modify: `dashboard/frontend/app.js` (new helpers near `resolveLatestAgentRun`; `renderAgentCardBody` — the `backtested` branch and the final draft return; `renderAgentCards` — dismiss binding next to the other button bindings)
- Modify: `dashboard/frontend/styles.css` (append)
- Test: `dashboard/backend/tests/test_backtest_failure_visibility.py` (create)

**Interfaces:**
- Consumes: `agent.latest_backtest_attempt` (Task 8), existing `resolveLatestAgentRun`, `escapeHtml`, `formatRelativeTime`, `applyAgentFilters`.
- Produces (used by Task 11's tests as stable names): `resolveFailedBacktestNotice(agent)`, `renderFailedBacktestNotice(agent)`, `dismissBacktestFailure(runId)`, `journalTimeToIso(value)`, const `DISMISSED_BACKTEST_FAILURES_KEY`.

- [ ] **Step 1: Write the failing guard tests**

```python
"""Failure-visibility guards for the My Agents card (PR-2, spec WS3.1/3.2)."""

import re

from dashboard.backend.tests._frontend_source import APP_JS, css_blocks, fn_body

# Strip // and /* */ comments before negative/count assertions; the lookbehind
# spares protocol-relative and https:// URLs.
_COMMENT_RE = re.compile(r"/\*.*?\*/|(?<!:)//[^\n]*", re.DOTALL)


def _strip(source: str) -> str:
    return _COMMENT_RE.sub("", source)


def test_failed_state_derives_in_exactly_one_function():
    """Every latest_backtest_attempt consumer goes through the one resolver
    (the PR #277 two-renderers lesson)."""
    body = _strip(fn_body("function resolveFailedBacktestNotice"))
    assert body.count("latest_backtest_attempt") > 0
    assert _strip(APP_JS).count("latest_backtest_attempt") == body.count(
        "latest_backtest_attempt"
    )


def test_failed_state_is_payload_driven_not_poll_driven():
    """Must render from the agents payload alone — no poll-loop globals."""
    body = _strip(fn_body("function resolveFailedBacktestNotice"))
    for poll_global in ("liveBacktestRunId", "liveBacktestProgress",
                       "backtestPollTimer", "backtest_status"):
        assert poll_global not in body


def test_notice_renders_on_both_card_body_branches():
    """Backtested AND draft branches — a first-run failure lands on a card
    that still says 'No backtests yet'."""
    body = fn_body("function renderAgentCardBody")
    assert body.count("renderFailedBacktestNotice(agent)") >= 2


def test_notice_suppressed_by_newer_success_and_dismissal():
    body = _strip(fn_body("function resolveFailedBacktestNotice"))
    assert "resolveLatestAgentRun" in body
    assert "readDismissedBacktestFailures" in body


def test_dismissal_is_client_local_by_run_id():
    assert "dismissed-backtest-failures" in APP_JS
    grid_binding = fn_body("function renderAgentCards")
    assert "agent-failed-dismiss-btn" in grid_binding


def test_failed_notice_styles_exist():
    assert css_blocks(".agent-card-failed-notice")
```

- [ ] **Step 2: Run — expect FAIL.**

Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_backtest_failure_visibility.py -v`

- [ ] **Step 3: Implement in `app.js`.** Helpers, placed directly after `resolveLatestAgentRunId`:

```javascript
// ---------------------------------------------------------------------------
// Failed-backtest notice (attempts journal, 2026-08-04 spec WS3)
// ---------------------------------------------------------------------------

const DISMISSED_BACKTEST_FAILURES_KEY = 'dismissed-backtest-failures';

function readDismissedBacktestFailures() {
  try {
    const raw = localStorage.getItem(DISMISSED_BACKTEST_FAILURES_KEY);
    return new Set(raw ? JSON.parse(raw) : []);
  } catch (_) {
    return new Set();
  }
}

function dismissBacktestFailure(runId) {
  if (!runId) return;
  const dismissed = readDismissedBacktestFailures();
  dismissed.add(runId);
  try {
    localStorage.setItem(
      DISMISSED_BACKTEST_FAILURES_KEY,
      JSON.stringify([...dismissed].slice(-50)),
    );
  } catch (_) { /* storage full: the notice just reappears */ }
}

/** Journal timestamps are UTC "YYYY-MM-DD HH:MM:SS" — make Date() parse them as UTC. */
function journalTimeToIso(value) {
  if (!value) return null;
  const s = String(value);
  return s.includes('T') ? s : `${s.replace(' ', 'T')}Z`;
}

/**
 * The ONE derivation of an agent card's failed-backtest state (both card-body
 * branches consume this; nothing else may read latest_backtest_attempt).
 * Absent field (old backend, issue #304 skew) → null → today's card exactly.
 */
function resolveFailedBacktestNotice(agent) {
  const attempt = agent?.latest_backtest_attempt;
  if (!attempt || typeof attempt !== 'object' || !attempt.run_id) return null;
  if (attempt.status !== 'failed' && attempt.status !== 'interrupted') return null;
  if (readDismissedBacktestFailures().has(attempt.run_id)) return null;
  // A successful run newer than the failure supersedes it.
  const latest = resolveLatestAgentRun(agent);
  if (
    latest?.created_at && attempt.created_at &&
    String(latest.created_at) >= String(attempt.created_at)
  ) {
    return null;
  }
  const firstLine = attempt.error
    ? String(attempt.error).split('\n')[0].slice(0, 160)
    : null;
  return {
    runId: attempt.run_id,
    interrupted: attempt.status === 'interrupted',
    error: firstLine,
    when: journalTimeToIso(attempt.finished_at || attempt.created_at),
  };
}

function renderFailedBacktestNotice(agent) {
  const notice = resolveFailedBacktestNotice(agent);
  if (!notice) return '';
  const heading = notice.interrupted ? 'Backtest interrupted' : 'Backtest failed';
  const detail = notice.interrupted
    ? 'The server restarted before this run finished.'
    : (notice.error || 'The run ended before results were saved.');
  const when = notice.when ? formatRelativeTime(notice.when) : '';
  return `
    <div class="agent-card-failed-notice" role="status">
      <div class="agent-card-failed-head">
        <strong>${escapeHtml(heading)}</strong>
        ${when ? `<span class="agent-card-failed-when">${escapeHtml(when)}</span>` : ''}
        <button type="button" class="agent-failed-dismiss-btn"
          data-run-id="${escapeHtml(notice.runId)}" aria-label="Dismiss">&times;</button>
      </div>
      <p class="agent-card-failed-detail">${escapeHtml(detail)}</p>
    </div>`;
}
```

`renderAgentCardBody` — prefix the **backtested** branch's return and the final (draft) return with the notice; the `paper` branch is untouched:

```javascript
  if (statusKey === 'backtested') {
    ...
    return `
      ${renderFailedBacktestNotice(agent)}
      ${renderAgentAllocatedCapitalHero(agent)}
      ...`;
  }

  return `
    ${renderFailedBacktestNotice(agent)}
    ${renderAgentAllocatedCapitalHero(agent)}
    <div class="agent-card-empty">
    ...`;
```

`renderAgentCards` — add next to the other `grid.querySelectorAll` bindings:

```javascript
  grid.querySelectorAll('.agent-failed-dismiss-btn').forEach((btn) => {
    btn.addEventListener('click', (event) => {
      event.stopPropagation();
      dismissBacktestFailure(btn.dataset.runId);
      applyAgentFilters(false);
    });
  });
```

- [ ] **Step 4: CSS** — append to `styles.css`:

```css
/* --- Failed-backtest notice on My Agents cards (2026-08-04 spec WS3.1) --- */
.agent-card-failed-notice {
  margin: 0 0 12px;
  padding: 10px 12px;
  border-radius: 10px;
  border: 1px solid rgba(239, 68, 68, 0.35);
  border-left: 3px solid #ef4444;
  background: rgba(239, 68, 68, 0.08);
}
.agent-card-failed-head {
  display: flex;
  align-items: center;
  gap: 8px;
}
.agent-card-failed-head strong {
  color: #f87171;
  font-size: 0.85rem;
}
.agent-card-failed-when {
  color: #9ca3af;
  font-size: 0.75rem;
}
.agent-failed-dismiss-btn {
  margin-left: auto;
  background: none;
  border: none;
  cursor: pointer;
  color: #9ca3af;
  font-size: 1rem;
  line-height: 1;
  padding: 2px 4px;
}
.agent-failed-dismiss-btn:hover {
  color: #f87171;
}
.agent-card-failed-detail {
  margin: 6px 0 0;
  font-size: 0.78rem;
  color: #9ca3af;
  overflow-wrap: anywhere;
}
```

- [ ] **Step 5: Run the guard file — PASS**, plus the whole frontend-guard set: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/ -q -k frontend`.

- [ ] **Step 6: Commit**

```bash
git add dashboard/frontend/app.js dashboard/frontend/styles.css \
        dashboard/backend/tests/test_backtest_failure_visibility.py
git commit -m "feat(ux): failed-backtest notice on agent cards"
```

---

### Task 11: Frontend — history surfaces (selector, Backtest panel, editor) (PR-2)

Failed attempts are selectable history: the selector labels them, selection renders the config panel through the existing `statusLabel` path with the stored error and **skips chart fetches** (no 404s); the editor list badges them without a return figure.

**Files:**
- Modify: `dashboard/frontend/app.js` (`formatBacktestRunLabel`, `formatBacktestRunPrimary` callers stay untouched; `loadData` — insert the failed branch after `resolveSelectedRun`; `renderBacktestRunConfig` — one guarded capital line)
- Modify: `dashboard/frontend/js/agent-editor.js` (`renderRunHistory`)
- Modify: `dashboard/frontend/styles.css` (editor badge class)
- Test: `dashboard/backend/tests/test_backtest_failure_visibility.py` (append)

**Interfaces:**
- Consumes: `status`/`error` on `/api/backtest/runs` entries (Task 7) and on `agent.runs` (Task 8); existing `renderBacktestRunConfig` `statusLabel` option; existing `formatBacktestError`, `displayNoMetrics`, `clearTradingLog`, `showBacktestRunProgress`, `updateBacktestRunProgress`, `renderBacktestDataSourceBadge`.
- Produces: `isFailedRunEntry(run)` helper in app.js.

- [ ] **Step 1: Append failing guard tests**

```python
def test_selector_labels_failed_and_interrupted_runs():
    body = fn_body("function formatBacktestRunLabel")
    assert "'failed'" in body and "Failed" in body
    assert "'interrupted'" in body and "Interrupted" in body


def test_load_data_skips_chart_fetch_for_failed_selection():
    """The failed branch must return before the chart-data fetch (no 404s)."""
    body = fn_body("async function loadData")
    assert body.index("isFailedRunEntry(selectedRun)") < body.index("chart-data")


def test_editor_history_badges_failed_runs_without_return_figure():
    from pathlib import Path
    editor = (Path(__file__).resolve().parents[2] /
              "frontend" / "js" / "agent-editor.js").read_text(encoding="utf-8")
    assert "agent-editor-run-status" in editor
    assert "Interrupted" in editor
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement in `app.js`.**

Helper next to `isBaselineRun`:

```javascript
function isFailedRunEntry(run) {
    return run?.status === 'failed' || run?.status === 'interrupted';
}
```

`formatBacktestRunLabel`:

```javascript
function formatBacktestRunLabel(run) {
    const base = [formatBacktestRunPrimary(run), formatBacktestRunSecondary(run)]
        .filter(Boolean).join(' · ');
    if (run.status === 'failed') return `Failed · ${base}`;
    if (run.status === 'interrupted') return `Interrupted · ${base}`;
    return base;
}
```

`loadData` — insert immediately after `const selectedRun = resolveSelectedRun(sessionRuns);` and before the `window.SELECTED_RUN = selectedRun;` line:

```javascript
            // Failed/interrupted attempts are selectable history but have no
            // chart, metrics or trades — render the config panel with the
            // stored error and stop before any per-run fetch (no 404s).
            if (selectedRun && isFailedRunEntry(selectedRun)) {
                localStorage.setItem(SELECTED_BACKTEST_RUN_KEY, selectedRun.run_id);
                window.SELECTED_RUN = null;
                window.MY_ALGO_RUN_ID = null;
                window.EXTERNAL_AGENT_RUN_ID = null;
                renderBacktestDataSourceBadge(selectedRun);
                renderBacktestRunConfig(selectedRun, {
                    launchConfig: getBacktestLaunchConfig(selectedRun.run_id),
                    statusLabel: selectedRun.status === 'interrupted'
                        ? 'Interrupted'
                        : 'Failed',
                });
                comparisonData = null;
                backtestChartData = null;
                // Blank the chart area (there is no chart for a failed run);
                // NOT initLiveBacktestChart(), which labels the dataset
                // "Agent (live)" and re-arms the live-attach check.
                if (chartInstance) {
                    chartInstance.destroy();
                    chartInstance = null;
                }
                displayNoMetrics();
                clearTradingLog('This backtest did not complete — no trades were recorded.');
                showBacktestRunProgress(true, { isError: true });
                updateBacktestRunProgress({
                    elapsedSeconds: 0,
                    message: selectedRun.status === 'interrupted'
                        ? 'Backtest interrupted (server restarted).'
                        : formatBacktestError(
                            selectedRun.error || 'Backtest failed.',
                            selectedRun.data_source,
                        ),
                });
                return;
            }
```

`renderBacktestRunConfig` — the capital line only (avoids "$0" from the attempt's synthetic `initial_equity: 0.0`; the `Number(null) === 0` trap applies):

```javascript
    const capital = cfg?.initialCapital
        ?? (isFailedRunEntry(run) ? null : run?.initial_equity);
```

and guard its display line:

```javascript
    setBacktestConfigText(
        'backtestConfigCapital',
        capital != null && Number.isFinite(Number(capital))
            ? `$${Number(capital).toLocaleString()}`
            : '—',
    );
```

- [ ] **Step 4: Implement in `js/agent-editor.js`** — `renderRunHistory`'s map callback becomes:

```javascript
    container.innerHTML = sorted
      .map((run) => {
        const failed = run.status === 'failed' || run.status === 'interrupted';
        const badge = failed
          ? `<span class="agent-editor-run-status is-failed">${run.status === 'interrupted' ? 'Interrupted' : 'Failed'}</span>`
          : '';
        const dates = [run.start_date, run.end_date].filter(Boolean).join(' → ');
        const primary = failed ? (dates || run.run_id || 'Backtest run') : formatRunPrimary(run);
        const meta = failed ? '' : formatRunMeta(run);
        return `
          <button type="button" class="agent-editor-run-item" data-run-id="${escapeHtml(run.run_id)}" role="listitem">
            <span class="agent-editor-run-primary">${escapeHtml(primary)}${badge}</span>
            <span class="agent-editor-run-secondary">${escapeHtml(formatRunSecondary(run))}</span>
            ${meta ? `<span class="agent-editor-run-meta">${escapeHtml(meta)}</span>` : ''}
          </button>`;
      })
      .join('');
```

(The click handler is unchanged — clicking a failed entry dispatches `agent-editor-open-run`, lands on the Backtest tab with that `run_id` selected, and `loadData`'s new branch renders it.)

- [ ] **Step 5: CSS** — append to the Task 10 block in `styles.css`:

```css
.agent-editor-run-status {
  margin-left: 8px;
  font-size: 0.7rem;
  font-weight: 600;
  padding: 1px 6px;
  border-radius: 6px;
  vertical-align: middle;
}
.agent-editor-run-status.is-failed {
  color: #f87171;
  background: rgba(239, 68, 68, 0.12);
}
```

- [ ] **Step 6: Run the guard file + full frontend-guard set — PASS.** #277's invariants must stay green (`deriveRunningProgress` gating, `progress_age_seconds`, ETA anchor — their existing tests are in the suite).

- [ ] **Step 7: Commit**

```bash
git add dashboard/frontend/app.js dashboard/frontend/js/agent-editor.js \
        dashboard/frontend/styles.css \
        dashboard/backend/tests/test_backtest_failure_visibility.py
git commit -m "feat(ux): failed runs selectable in history surfaces"
```

---

### Task 12: Frontend — LLM coverage chip (PR-2)

On a completed run's config panel and its editor-history entry, when coverage metadata exists and is `< 0.95`: disclose the rule-based fallback. Old runs without the keys show nothing.

**Files:**
- Modify: `dashboard/frontend/app.html` (one new row in the backtest-config `<dl>`, after the Status row — copy the Status row's exact wrapper markup/classes, new ids)
- Modify: `dashboard/frontend/app.js` (`renderBacktestRunConfig`)
- Modify: `dashboard/frontend/js/agent-editor.js` (`formatRunMeta`)
- Modify: `dashboard/frontend/styles.css` (append)
- Test: `dashboard/backend/tests/test_backtest_failure_visibility.py` (append)

**Interfaces:**
- Consumes: `run.metadata.llm_decision_coverage` / `.llm_decisions` (Task 9; `metadata` is already a parsed dict on run payloads).
- Produces: DOM ids `backtestConfigCoverageRow`, `backtestConfigCoverage`.

- [ ] **Step 1: Append failing guard tests**

```python
def test_coverage_chip_keys_on_h6_threshold_with_typeof_guard():
    from dashboard.backend.tests._frontend_source import APP_HTML
    body = fn_body("function renderBacktestRunConfig")
    assert "llm_decision_coverage" in body
    # typeof, not truthiness: Number(null) === 0 would fake 0% coverage.
    assert "typeof coverage === 'number'" in body
    assert "0.95" in body
    assert 'id="backtestConfigCoverageRow"' in APP_HTML


def test_editor_history_meta_discloses_low_coverage():
    from pathlib import Path
    editor = (Path(__file__).resolve().parents[2] /
              "frontend" / "js" / "agent-editor.js").read_text(encoding="utf-8")
    assert "llm_decision_coverage" in editor
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement.**

`app.html` — duplicate the Status row's exact wrapper structure (same surrounding div/dt/dd classes as its siblings in the config `<dl>`), inserted after it, with `hidden` on the wrapper:

```html
                    <div id="backtestConfigCoverageRow" hidden>
                        <dt>Model coverage</dt>
                        <dd id="backtestConfigCoverage" class="backtest-config-coverage-warning">—</dd>
                    </div>
```

(Match the sibling rows' wrapper markup byte-for-byte apart from ids/content — if siblings carry a row class, carry it too.)

`renderBacktestRunConfig` — after the `backtestConfigStatus` write:

```javascript
    // H6's MIN_LLM_DECISION_COVERAGE (0.95): below it the model did not
    // really drive this run — say so instead of letting the per-step
    // rule-based fallback pass silently (the quota-dead-key case produces a
    // "successful" run the model never touched). Old runs lack the keys and
    // show nothing; typeof guard because Number(null) === 0.
    const coverage = metadata.llm_decision_coverage;
    const coverageRow = document.getElementById('backtestConfigCoverageRow');
    const showCoverage = !running
        && typeof coverage === 'number'
        && coverage < 0.95;
    if (coverageRow) coverageRow.hidden = !showCoverage;
    if (showCoverage) {
        const coverageEl = document.getElementById('backtestConfigCoverage');
        if (coverageEl) {
            coverageEl.textContent = coverage === 0
                ? 'Model drove 0 steps — rule-based fallback filled the rest; check your API key/quota.'
                : `Model drove ${Math.round(coverage * 100)}% of steps — rule-based fallback filled the rest; check your API key/quota.`;
        }
    }
```

`js/agent-editor.js` — in `formatRunMeta`, before `return parts.join(' · ');`:

```javascript
    const meta = run.metadata;
    const coverage = meta && typeof meta.llm_decision_coverage === 'number'
      ? meta.llm_decision_coverage
      : null;
    if (coverage != null && coverage < 0.95) {
      parts.push(coverage === 0
        ? '⚠ model drove 0 steps'
        : `⚠ model drove ${Math.round(coverage * 100)}% of steps`);
    }
```

`styles.css` — append:

```css
.backtest-config-coverage-warning {
  color: #fbbf24;
  font-size: 0.82rem;
}
```

- [ ] **Step 4: Run the guard file — PASS.**

- [ ] **Step 5: Commit**

```bash
git add dashboard/frontend/app.html dashboard/frontend/app.js \
        dashboard/frontend/js/agent-editor.js dashboard/frontend/styles.css \
        dashboard/backend/tests/test_backtest_failure_visibility.py
git commit -m "feat(ux): disclose LLM decision coverage below the H6 bar"
```

---

### Task 13: Busters, pins, full suite, PR-2 (PR-2)

- [ ] **Step 1: Bump busters** in `app.html` for every file PR-2 touched: `app.js` → current+1 (nominally v=64 if PR-1 landed v=63), `js/agent-editor.js` → v=23, `styles.css` → v=80. Use the values current in the rebased tree, +1 each.

- [ ] **Step 2: Update every exact-string pin**: `test_frontend_fast_boot.py::test_cache_busters_bumped` (app.js + styles.css lines) and any pin found by `command grep -rn "agent-editor.js?v=" dashboard/backend/tests/`. Extend the comment: `# v=NN: backtest failure visibility (attempts journal PR-2)`.

- [ ] **Step 3: Manual smoke** (optional but cheap): `DATABASE_PATH=/tmp/claude-smoke.db ~/atl-venv/bin/python -m uvicorn dashboard.backend.app:app --port 8010` — POST a backtest with an invalid model to a dead key, watch the card show the failed notice after the run dies, dismiss it, re-select the failed run in the Backtest tab. Never run against the committed seed DB.

- [ ] **Step 4: Full suite**

Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/ -q`
Expected: green (a red is a real regression — the suite has no tolerated failures).

- [ ] **Step 5: Commit + push + open PR-2 as a draft** until CI (including the `@pg_only` tier and CodeQL) is green, then mark ready:

```bash
git add dashboard/frontend/app.html dashboard/backend/tests/test_frontend_fast_boot.py
git commit -m "chore(frontend): bump cache busters for failure-visibility round"
```

PR title: `feat(backtest): failed-run journal + visibility`. Body (short): the three spec findings it closes, the no-new-routes note, and the deploy-skew note (additive fields; new frontend tolerates old backend). Verify in the CI log that the `@pg_only` tests **ran** (skip count drops) rather than silently skipping.

---

## Out of scope / follow-ups (do not implement)

- **Cancel** — issue #273 (process handle + `cancelled` status + PR #163 race).
- **Changing the silent per-step LLM fallback** — disclosure only this round.
- **RTD/user-facing docs**: `docs/source/lab/getting_started.rst` (launch flow) and `docs/source/lab/architecture.rst` (endpoints + subprocess) go stale when PR-2 lands — surfaced to the requester as an end-of-round follow-up; never edited mid-session.
- **Multi-run concurrency** — the one-run-per-process global stays.

## Self-review notes (already applied)

- Spec coverage: WS1→Tasks 3-6, WS2→Tasks 7-8, WS3→Tasks 10-11, WS4→Tasks 9+12, WS5→Task 2, WS6→Task 1, D1-D7 all encoded as constraints or task content.
- `attempt_as_run_entry.initial_equity = 0.0` is load-bearing (routes failed entries into the frontend's existing `!metrics.initial_equity` no-metrics guard) — documented in its docstring.
- Postgres tie-break is `run_id DESC` vs SQLite `rowid DESC`: deterministic on both; divergence only within a same-second tie between two attempts, which cannot occur for the single-run-at-a-time frontend launcher.
- The two `TIMESTAMP` columns on SQLite vs `TEXT` on Postgres mirror the existing `agent_runs` convention; the parity guard compares names only.
