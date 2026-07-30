# Durable run history (`AGENT_RUNS_DATABASE_URL`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans`. Execute tasks in order; do not skip the
> "run to verify it fails" steps — several tasks here are guard-test-driven and a step that
> passes before the implementation exists means the guard is not wired.

**Goal:** Make backtest run history survive a Render deploy. Five "cold" tables — `agent_runs`,
`equity_timeseries`, `trades`, `backtest_decisions`, `run_manifest` — move to a dedicated Neon
Postgres project selected by `AGENT_RUNS_DATABASE_URL`, behind a new `PostgresBacktestDatabase`
twin that is signature-identical to `BacktestDatabase`. The hot per-step table
(`idempotency_keys`) stays on local SQLite via delegation, so no agent request gains a network
round-trip. Unset `AGENT_RUNS_DATABASE_URL` ⇒ today's behavior, byte for byte.

**Architecture:** The established twin pattern, sixth instance. `database.py` keeps
`BacktestDatabase` unchanged as the SQLite default; a `_build_backtest_db()` factory (cloned
from `users.py::_build_user_store`) replaces the bare `db = BacktestDatabase()` singleton at
`database.py:998`. `database_postgres.py` sits beside it (flat module, `*_postgres.py` suffix —
load-bearing for twin auto-discovery) and implements the cold half against Neon while
delegating `get_idempotency`/`put_idempotency` to an embedded plain `BacktestDatabase`. No base
class, no ORM, no SQL-translation layer. All 14 importers of `db` are untouched.

**Tech Stack:** Python 3.13, psycopg3 + psycopg_pool (via `dashboard/backend/db_pool.py`),
SQLite3 stdlib, FastAPI, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-20-run-history-persistence-design.md` — **read its
two Amendments sections first.** Pass 2 records two guard collisions this plan implements
around, and three "follow-up" defects that are already fixed on `main` (do not re-file them).

**Branch:** `feat/runs-database-url` (already cut; the amended spec is its first commit).

## Global Constraints

- **Task order 1 → 3 is non-negotiable.** Task 1 (conftest strip) must land before any factory
  exists, or a developer's exported `AGENT_RUNS_DATABASE_URL` points the whole suite at prod
  Neon.
  Task 2 (SQLite-side symmetric changes) must land before Task 3, because the twin's DDL is
  written to match the *post-Task-2* SQLite shape and the column-parity guard compares the two.
- **Test command:** `pytest dashboard/backend/tests/ -q` (full suite must stay green at every
  commit). CI runs `pytest dashboard/backend/tests/ --timeout=180 -p no:cacheprovider`.
- **The `@pg_only` tier cannot run in this environment** — no local docker/sudo, and
  `py-pglite` segfaults psycopg. It fails *open* (skips) when `TEST_POSTGRES_URL` is unset, so
  a local green run proves nothing about Tasks 3–5, 9 and 11. Verify those in CI:
  `gh run list --branch feat/runs-database-url --limit 1`, then
  `gh run view <id> --log | grep -E "backtest_db_postgres|passed|skipped" | tail -30`.
- **Never `git add -A` in this repo.** Before every commit run
  `git status --short dashboard/storage/data/backtest.db` — it must print **nothing**. The
  committed seed DB *is* prod's database, and importing any backend module runs lazy DDL
  against `DATABASE_PATH`, which defaults to that file.
- **`print()`, never `logger.info()`** for anything meant to be visible in prod — backend
  loggers emit nothing under the deployed uvicorn config. Assert on `capsys`, never `caplog`.
- **Credentials never enter the repo.** `AGENT_RUNS_DATABASE_URL` lives only in the Render
  dashboard. `describe_database_url()` (host/db only) is the *only* thing that may be printed.
- **Commit style:** `feat(runs): …` / `test(runs): …` / `docs: …`, one commit per task.
- **`AGENT_RUNS_DATABASE_URL` is already set in Render** (2026-07-29, web service only —
  see Task 12 Step 3), so the deploy-gate that would normally hold this PR is discharged
  *before* the code exists. The PR still opens as a **DRAFT** while in flight, but the
  first-line gate is now about the backfill, not the var. Re-check the var is present
  immediately before merging: `main` auto-deploys prod, an unset var silently selects
  ephemeral SQLite, and a comment is not a gate.
- **Local dev now inherits the var too** — it lives in `dashboard/.env` of the main checkout
  (renamed alongside this plan). Nothing reads it until Task 6, but from that commit onward a
  local `uvicorn` writes run history straight to **prod Neon**. Comment it out locally, or
  accept that local runs share prod's history. Tests are unaffected (Task 1 strips it).

## File Structure

| File | Action | Task |
|---|---|---|
| `dashboard/backend/tests/conftest.py` + `test_env_isolation.py` | Modify — strip `AGENT_RUNS_DATABASE_URL` | 1 |
| `dashboard/backend/database.py` | Modify — `actions_trace_ref`, `run_manifest` cleanup, factory | 2, 6 |
| `dashboard/backend/tests/test_database_cold_half.py` | **Create** — SQLite-side coverage | 2 |
| `dashboard/backend/database_postgres.py` | **Create** — the twin | 3, 4, 5 |
| `dashboard/backend/tests/test_store_twin_parity.py` | Modify — `_TWINS` entry | 7 |
| `dashboard/backend/tests/test_backtest_db_postgres.py` | **Create** — dispatch + `@pg_only` mirror | 8, 9 |
| `dashboard/backend/tests/test_postgres_url_guard.py` | Modify — fixture import + parametrize | 9 |
| `dashboard/backend/app.py` | Modify — delete raw-sqlite3 debug blocks | 10 |
| `dashboard/scripts/backfill_runs_to_postgres.py` | **Create** — one-time backfill | 11 |
| `dashboard/backend/tests/test_backfill_runs.py` | **Create** | 11 |
| `.env.example`, `render.yaml`, `CLAUDE.md` | Modify — document the var | 6, 12 |

**Import-cycle safety:** `database_postgres.py` imports `BacktestDatabase` from `database.py`
at module top; `database.py` imports `PostgresBacktestDatabase` **inside** `_build_backtest_db()`
only. Never at module scope in `database.py` — that is the cycle.

**Where the twin lives:** `dashboard/backend/database_postgres.py`, flat, beside `database.py`.
`test_every_postgres_twin_module_is_registered` globs `backend.rglob("*_postgres.py")` and
`_module_source_path` maps a dotted path straight to `<path>.py`, so a package directory or a
different suffix silently ships with **zero** parity coverage.

---

### Task 1: Strip `AGENT_RUNS_DATABASE_URL` in the test suite

**Files:**
- Modify: `dashboard/backend/tests/conftest.py`

**Interfaces:**
- Consumes: nothing. Produces: nothing importable. Every later task depends on this landing
  first.

This is the same guarantee `USERS_DATABASE_URL` and `CONTENT_DATABASE_URL` already have, and it
must exist *before* a factory reads the var, not after.

- [ ] **Step 1: Add the strip beside its two siblings**

In `conftest.py`, immediately after the `CONTENT_DATABASE_URL` pop (currently ~line 54) — in
that group, **not** in the later "scale knobs" block:

```python
# Same guarantee for AGENT_RUNS_DATABASE_URL: it selects the Postgres backend for
# backtest run history (agent_runs, equity_timeseries, trades,
# backtest_decisions, run_manifest). A value inherited from the developer's
# environment would point the whole suite at the live runs database -- whose
# @pg_only-style destructive helpers would then run against prod. Strip it
# before any backend module is imported.
os.environ.pop("AGENT_RUNS_DATABASE_URL", None)
```

- [ ] **Step 2: Prove it strips at import time, not fixture time**

The existing coverage lives in `dashboard/backend/tests/test_env_isolation.py` — add the third
case there, mirroring how `USERS_DATABASE_URL`/`CONTENT_DATABASE_URL` are asserted:

```python
def test_runs_database_url_is_stripped():
    assert "AGENT_RUNS_DATABASE_URL" not in os.environ
```

- [ ] **Step 3: Run** `pytest dashboard/backend/tests/ -q`. Expected: green, no change in count
      beyond the new test.

- [ ] **Step 4: Commit**

```bash
git status --short dashboard/storage/data/backtest.db   # must print nothing
git add dashboard/backend/tests/conftest.py dashboard/backend/tests/test_env_isolation.py
git commit -m "test(runs): strip AGENT_RUNS_DATABASE_URL before backend import"
```

---

### Task 2: SQLite-side symmetric changes (`actions_trace_ref`, `run_manifest` cleanup)

**Files:**
- Modify: `dashboard/backend/database.py`
- Create: `dashboard/backend/tests/test_database_cold_half.py` — there is **no** `test_database.py`
  and nothing currently covers `delete_run`/`clear_all` at all. Follow the sibling convention in
  `test_currency_audit_database.py`: no fixture, just `BacktestDatabase(tmp_path / "x.db")` and a
  small `_insert_run` helper.

**Interfaces:**
- Consumes: nothing. Produces: the post-migration SQLite shape that Task 3's DDL mirrors.

Two changes, both required by spec pass 2:

1. **`actions_trace_ref TEXT`** on `backtest_decisions` — spec Decision 4. It goes on the
   SQLite side *as well as* Postgres because `test_postgres_twin_schema_columns_match_sqlite`
   compares column sets **bidirectionally**; a Postgres-only column reddens CI.
2. **`run_manifest` in `delete_run` and `clear_all`** — today both touch only four tables
   (`database.py:970–994`), orphaning manifest rows, and `run_manifest` has no FK for a
   Postgres cascade to sweep. Fixing both sides keeps the `@pg_only` mirror suite honest.

- [ ] **Step 1: Write the failing tests**

```python
"""Cold-half schema and delete semantics for the SQLite BacktestDatabase."""

from dashboard.backend.database import BacktestDatabase


def _insert_run(db: BacktestDatabase, run_id: str) -> None:
    db.insert_run(
        run_id=run_id, session_id="cold-half", agent_name="Agent", mode="backtest",
        start_date="2026-01-01", end_date="2026-01-02", initial_equity=1_000,
    )


def test_backtest_decisions_has_actions_trace_ref(tmp_path):
    db = BacktestDatabase(tmp_path / "cold.db")
    conn = db._get_connection()
    cols = {row[1] for row in conn.execute("PRAGMA table_info(backtest_decisions)")}
    assert "actions_trace_ref" in cols


def test_delete_run_removes_the_manifest(tmp_path):
    db = BacktestDatabase(tmp_path / "cold.db")
    _insert_run(db, "r1")
    db.insert_run_manifest("r1", {"any": "thing"})
    db.delete_run("r1")
    assert db.get_run_manifest("r1") is None


def test_clear_all_removes_manifests(tmp_path):
    db = BacktestDatabase(tmp_path / "cold.db")
    _insert_run(db, "r1")
    db.insert_run_manifest("r1", {"any": "thing"})
    db.clear_all()
    assert db.get_run_manifest("r1") is None
```

Always construct against `tmp_path`; never touch the module-level `db` singleton, which points
at the committed seed.

- [ ] **Step 2: Run to verify they fail** —
      `pytest dashboard/backend/tests/test_database_cold_half.py -q`. Expected: 3 failed.

- [ ] **Step 3: Add the column in both places**

In the `backtest_decisions` `CREATE TABLE` (~line 158–176), after `context_ref TEXT`:

```sql
    actions_trace_ref TEXT,
```

And in `_migrate_schema`, beside the existing `context_ref` lazy ALTER (~line 289), guarded the
SQLite way (no `IF NOT EXISTS` in SQLite):

```python
if "actions_trace_ref" not in decision_columns:
    cursor.execute("ALTER TABLE backtest_decisions ADD COLUMN actions_trace_ref TEXT")
```

The column is nullable and never written by this change — it reserves the R2 offload seam while
that is free (spec Decision 4). **Both** the CREATE and the ALTER are required: `CREATE TABLE IF
NOT EXISTS` no-ops on an existing deployment, so the ALTER is the only path that reaches one.

- [ ] **Step 4: Add `run_manifest` to both delete paths**

`delete_run` — add before the `agent_runs` delete (order is cosmetic here; `run_manifest` has no
FK):

```python
cursor.execute("DELETE FROM run_manifest WHERE run_id = ?", (run_id,))
```

`clear_all` — add alongside the other four:

```python
cursor.execute("DELETE FROM run_manifest")
```

- [ ] **Step 5: Run to verify they pass** — `pytest dashboard/backend/tests/ -q`. Expected:
      full suite green. **Then check the seed DB was not mutated:**
      `git status --short dashboard/storage/data/backtest.db` must print nothing. If it does,
      a test or import ran against the real path — restore with
      `git checkout -- dashboard/storage/data/backtest.db` and find the leak before continuing.

- [ ] **Step 6: Commit**

```bash
git status --short dashboard/storage/data/backtest.db   # must print nothing
git add dashboard/backend/database.py dashboard/backend/tests/test_database.py
git commit -m "feat(runs): reserve actions_trace_ref and clean up run_manifest on delete"
```

---

### Task 3: `database_postgres.py` — schema, connection, delegation

**Files:**
- Create: `dashboard/backend/database_postgres.py`

**Interfaces:**
- Consumes: `dashboard.backend.db_url.require_postgres_url`,
  `dashboard.backend.db_pool.get_pool`, `dashboard.backend.database.BacktestDatabase`.
- Produces: `PostgresBacktestDatabase(database_url)` with `_get_connection()`, `_init_schema()`,
  and the two delegated idempotency methods. Tasks 4–5 fill in the rest of the surface.

- [ ] **Step 1: Module docstring, imports, `__init__`, connection**

Follow the canonical twin shape (`domain/agents/repository_postgres.py`) exactly:

```python
"""Postgres-backed BacktestDatabase implementation (run history, the "cold half").

Selected instead of the default SQLite BacktestDatabase when AGENT_RUNS_DATABASE_URL is
set (see database.py's _build_backtest_db). Exists because the SQLite store lives
in DATABASE_PATH, which resets to the committed seed database on every deploy of
the disk-less Render free-tier host -- silently deleting every backtest run,
equity curve, trade log and decision log, which is why issue #145 (the leaderboard
refresh cron) is blocked. Method surface, return schemas and behavior are
identical to BacktestDatabase; only the SQL dialect differs, plus two named
timestamp divergences on upsert (see insert_run) and a batched get_equity_curves.

The hot half stays local: get_idempotency/put_idempotency are delegated to an
embedded plain BacktestDatabase so the per-step agent request never gains a
network round-trip. protocol_runs/protocol_steps are not ours -- they belong to
domain/runs/repository.py and are untouched.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from dashboard.backend.database import BacktestDatabase
from dashboard.backend.db_url import require_postgres_url


class PostgresBacktestDatabase:
    def __init__(self, database_url: str):
        self.database_url = require_postgres_url(database_url)
        self._sqlite = BacktestDatabase()   # hot half: idempotency_keys stays local
        self._init_schema()

    def _get_connection(self):
        # Pooled checkout: same context-manager transaction semantics as
        # psycopg.connect (commit on clean exit), returned to the pool on close.
        from dashboard.backend.db_pool import get_pool

        return get_pool(self.database_url).connection()
```

Note the ordering in `__init__`: `require_postgres_url` first (it raises `ValueError` before
psycopg can echo a DSN containing a password into a traceback), then the embedded SQLite store,
then `_init_schema()` — so an unreachable Neon aborts boot rather than degrading. Do **not**
wrap any of it in try/except.

- [ ] **Step 2: `_init_schema()` — CREATE for the five tables**

Mirror the post-Task-2 SQLite shape. Type mapping: `REAL` → `DOUBLE PRECISION`, `INTEGER
PRIMARY KEY AUTOINCREMENT` → `BIGINT GENERATED BY DEFAULT AS IDENTITY`, `TEXT` → `TEXT`.

For `created_at`/`updated_at`, declare **TEXT with a Postgres DEFAULT that produces SQLite's
exact `CURRENT_TIMESTAMP` string**, so read shapes are identical and the *database* stamps the
row just as SQLite does:

```sql
created_at TEXT NOT NULL DEFAULT to_char(now() AT TIME ZONE 'utc', 'YYYY-MM-DD HH24:MI:SS')
```

(The spec says "populated app-side"; a DB default is the better instrument for the same stated
goal — same string format, and it keeps the created_at/updated_at divergences in Task 4
well-defined rather than clock-dependent.)

Required elements, none optional:

- `agent_runs` — the full post-migration column set: the base CREATE columns **plus**
  `session_id`, `llm_model`, `baseline_djia_run_id`, `baseline_buyhold_run_id`, `llm_calls`,
  `input_tokens`, `output_tokens`, `est_cost_usd`, `metadata`. `run_id TEXT PRIMARY KEY`.
- `equity_timeseries`, `trades`, `backtest_decisions` — with **real, enforced** FKs:
  `REFERENCES agent_runs(run_id) ON DELETE CASCADE`. SQLite declares these but never enforces
  them (no `PRAGMA foreign_keys` anywhere); Postgres enforces by default. Every writer in this
  codebase already writes parent-before-child, and `delete_run` already deletes children first,
  so nothing changes — but a *new* caller that inserts an orphan child will now fail loudly.
- `backtest_decisions.actions_trace_ref TEXT` (nullable), matching Task 2.
- `run_manifest` — `run_id TEXT PRIMARY KEY`, `manifest_json TEXT NOT NULL`, `created_at`.
  **No FK** (deliberate: the SQLite side has none, and its rows are deleted explicitly).
- The five indexes: `idx_agent_runs_session`, `idx_agent_runs_session_mode`,
  `idx_run_timestamp`, `idx_trades_run`, `idx_decisions_run`.
- **`UNIQUE (run_id, timestamp)` on `equity_timeseries`** — a table *constraint*, not one of the
  five indexes, and load-bearing twice: `ON CONFLICT (run_id, timestamp)` raises without a
  matching unique constraint, and it is the natural key that makes a rerun replace rather than
  duplicate.

- [ ] **Step 3: `_init_schema()` — the ALTER block**

Every column the SQLite side adds via a lazy `ALTER TABLE … ADD COLUMN` must be re-added here
via `ADD COLUMN IF NOT EXISTS`, or `test_postgres_twin_repeats_every_sqlite_lazy_migration`
fails. The complete list, read out of `database.py`'s migration helpers:

| SQLite helper | Table | Columns |
|---|---|---|
| `_migrate_schema` | `agent_runs` | `session_id`, `llm_model`, `baseline_djia_run_id`, `baseline_buyhold_run_id`, `llm_calls`, `input_tokens`, `output_tokens`, `est_cost_usd`, `metadata` |
| `_migrate_schema` | `backtest_decisions` | `context_ref`, `actions_trace_ref` (Task 2) |
| `_migrate_trades_schema` | `trades` | `quantity`, `side`, `value`, `reason` |
| `_migrate_currency_audit_schema` | `equity_timeseries` | `native_equity`, `native_cash`, `native_positions_value`, `fx_rate` |
| `_migrate_currency_audit_schema` | `trades` | `native_price`, `native_value`, `fx_rate` |

Re-derive this table from the code before writing it — if a migration was added after this plan,
the guard will tell you, but only for `ADD COLUMN` migrations (see Step 4).

Carry the canonical institutional-memory comment, cross-referencing the worked example:

```python
        # ADDING A COLUMN LATER? It must go in an `ALTER TABLE ... ADD COLUMN IF
        # NOT EXISTS` below, *not* only in the CREATE above. CREATE TABLE IF NOT
        # EXISTS silently no-ops once the table exists, so an existing deployment
        # would never gain the column, and every query naming it would raise
        # UndefinedColumn -- 500ing this whole surface while /health stays green.
        # See domain/agents/repository_postgres.py for the full worked example.
```

- [ ] **Step 4: The uniqueness migration — and why no guard covers it**

Add an idempotent counterpart to SQLite's `_ensure_equity_timeseries_uniqueness`:

```python
cur.execute(
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_equity_timeseries_run_timestamp "
    "ON equity_timeseries (run_id, timestamp)"
)
```

**Nothing static enforces this.** The lazy-migration guard regex-matches `ALTER TABLE … ADD
COLUMN` strings only (`test_store_twin_parity.py:232–236`), and the SQLite original adds a
unique *index*, so it is invisible to the check. The only thing that would catch its omission is
the `@pg_only` upsert test in Task 9 — which is why that test is mandatory, not optional.
Stakes are low in practice (the Neon project is empty, so the inline `CREATE TABLE` constraint
covers every row that will ever exist there), but write it anyway and label it.

- [ ] **Step 5: Delegate the hot half**

```python
    def get_idempotency(self, run_id: str, step_index: int, idem_key: str) -> Optional[Dict[str, Any]]:
        return self._sqlite.get_idempotency(run_id, step_index, idem_key)

    def put_idempotency(self, run_id: str, step_index: int, idem_key: str, ack: Dict[str, Any]) -> None:
        self._sqlite.put_idempotency(run_id, step_index, idem_key, ack)
```

Copy both signatures out of the live class — `test_postgres_twin_signatures_match_sqlite`
compares parameter names, order, kind and defaults, and both of these take `step_index`.

- [ ] **Step 6: Commit** (the module will not be import-clean as a *store* until Tasks 4–5;
      that is fine, it is a class definition)

```bash
git status --short dashboard/storage/data/backtest.db   # must print nothing
git add dashboard/backend/database_postgres.py
git commit -m "feat(runs): add PostgresBacktestDatabase schema and delegation"
```

---

### Task 4: Port the writers

**Files:**
- Modify: `dashboard/backend/database_postgres.py`

**Interfaces:**
- Produces: `insert_run`, `update_run_baselines`, `insert_equity_point`, `insert_equity_points`,
  `insert_trades`, `insert_decisions`, `insert_run_manifest`, `delete_run`, `clear_all` —
  every signature copied verbatim from `BacktestDatabase`.

Mechanical rules for all of them: `?` → `%s`; `with self._get_connection() as conn: with
conn.cursor() as cur:` (never call `conn.commit()` — the pool's context manager commits on clean
exit); never pass `row_factory` (the pool sets `dict_row` at construction); import pure helpers
from `database.py` rather than reimplementing them.

- [ ] **Step 1: `insert_run` — the upsert, with both timestamp divergences**

`INSERT OR REPLACE` → `INSERT … ON CONFLICT (run_id) DO UPDATE SET …`. This is load-bearing for
the daily leaderboard, which re-runs a deterministic `run_id` (`lb_<strategy>_<start>_<end>`) and
relies on overwrite idempotency.

`insert_run`'s SQLite column list is 18 columns and omits **both** timestamps
(`database.py:515–521`), so SQLite's REPLACE resets `created_at` *and* `updated_at`. The upsert
must therefore be explicit about each, in opposite directions:

- **`created_at` — do NOT touch it in `DO UPDATE SET`.** Preserved on purpose; SQLite's reset is
  accidental and nothing reads the column as "last refreshed".
- **`updated_at` — explicitly set it.** Omitting it (the natural copy-paste outcome of the line
  above) would freeze it at the original insert while SQLite refreshes it, diverging in the
  opposite direction exactly on `force_refresh`:

```sql
    ON CONFLICT (run_id) DO UPDATE SET
        session_id = EXCLUDED.session_id,
        ...,
        updated_at = to_char(now() AT TIME ZONE 'utc', 'YYYY-MM-DD HH24:MI:SS')
```

`metadata` is `json.dumps`'d to TEXT exactly as the SQLite side does — not a native `json`/`jsonb`
column, so `_parse_run_row` keeps working unchanged for both backends.

- [ ] **Step 2: `insert_equity_points` — keep the delete**

Do **not** "simplify" this into a pure upsert. The live method (`b988f24`, 2026-07-23) does
`DELETE FROM equity_timeseries WHERE run_id = ?` **then** an `executemany` insert, both in one
transaction, gated by `replace: bool = True`, with an empty-`points` list short-circuiting to a
no-op rather than a wipe. A rerun can legitimately produce a *different* set of timestamps
(fewer bars, a partial run, a changed symbol list) and the `(run_id, timestamp)` key only
collapses timestamps that *repeat* — without the delete, leftovers of the previous, longer curve
stay spliced into the new one, silently, precisely in the force-refresh case.

Use `executemany` (psycopg3 pipelines it), **not** a Python loop over `_get_connection()`: a
checkout per row would serialize 2,102 rows behind `max_size=5` and can trip the 10 s pool
timeout, which would look like Neon being slow rather than like a loop written wrong.

`insert_equity_point` (singular) keeps its own `ON CONFLICT (run_id, timestamp) DO UPDATE`.

- [ ] **Step 3: `insert_trades` / `insert_decisions` — plain appends**

Both are append-only, written once per run at finalize; the `id` columns are
`GENERATED BY DEFAULT AS IDENTITY`. Note the SQLite `insert_trades` inspects the live table to
cope with legacy `shares`/`action`/`total_value` columns — the Postgres table never had them, so
write the modern column set directly and drop the introspection.

- [ ] **Step 4: `insert_run_manifest`, `update_run_baselines`**

`insert_run_manifest`: `ON CONFLICT (run_id) DO UPDATE`. `update_run_baselines`: a plain UPDATE,
keyword-only `djia_run_id` / `buyhold_run_id`, both optional — port the "only update what was
passed" logic exactly.

- [ ] **Step 5: `delete_run` / `clear_all`**

- `delete_run(run_id)` — delete the `agent_runs` row (children go via `ON DELETE CASCADE`) plus
  an explicit `DELETE FROM run_manifest WHERE run_id = %s`, since no FK covers it.
- `clear_all()` — truncate all five Postgres tables, **and** delegate to `self._sqlite.clear_all()`
  so stale cold rows in the local file (the seed data, still present) don't outlive a wipe.
- **Neither touches `idempotency_keys`.** The spec's line about "clears any local idempotency
  rows for that run" would introduce a divergence: SQLite's `delete_run` leaves them, and the
  Task 9 mirror suite asserts the two backends behave identically. Parity wins; note it in the
  method's docstring so the omission reads as deliberate.

- [ ] **Step 6: Commit**

```bash
git status --short dashboard/storage/data/backtest.db   # must print nothing
git add dashboard/backend/database_postgres.py
git commit -m "feat(runs): port BacktestDatabase writers to Postgres"
```

---

### Task 5: Port the readers

**Files:**
- Modify: `dashboard/backend/database_postgres.py`

**Interfaces:**
- Produces: `get_run`, `get_run_with_session`, `get_all_runs`, `get_runs_by_session`,
  `get_runs_by_sessions`, `get_runs_by_mode`, `get_equity_curve`, `get_equity_curves`,
  `get_trades`, `get_decisions`, `get_run_manifest`.

- [ ] **Step 1: Re-enumerate the reader set from the live class first**

`git grep -n "    def get_" dashboard/backend/database.py`. The list above is what exists today;
anything added since belongs here too. `test_postgres_twin_exposes_every_sqlite_method` will
fail loudly for a missed one — but only *after* Task 7 registers the twin, so check by hand now.

- [ ] **Step 2: Port the straightforward ones**

`?` → `%s`, `IN (…)` → `= ANY(%s)` with a list parameter. Rows already arrive as dicts
(`dict_row` at the pool), so the shared shaping helpers work unchanged. Reuse
`BacktestDatabase._parse_run_row` (a `@staticmethod`) for every method that returns `agent_runs`
rows rather than reimplementing the `metadata` JSON decode.

Replicate the None-stripping of currency-audit fields (`native_equity`, `native_cash`,
`native_positions_value`, `fx_rate`, `native_price`, `native_value`) that `get_equity_curve` and
`get_trades` do — a caller that gets an extra `"fx_rate": None` key back from Postgres but not
from SQLite is a parity break the static guards cannot see.

- [ ] **Step 3: `get_equity_curves` — the one method that is NOT a line-for-line port**

The SQLite version (`database.py:753–758`) is a Python `for` loop calling `get_equity_curve`
once per id, and `domain/agents/service.py:185` hands it one run id **per agent**. Free against
a local file; over Neon it is one network round-trip per agent on every My Agents page load.
Implement it as a **single** query and group in Python:

```python
    def get_equity_curves(self, run_ids: List[str]) -> Dict[str, List[Dict]]:
        """Batched: one query for all runs (the SQLite twin loops; over the
        network that would be one round-trip per agent on the My Agents page)."""
        result: Dict[str, List[Dict]] = {run_id: [] for run_id in run_ids}
        if not run_ids:
            return result
        # ... SELECT ... WHERE run_id = ANY(%s) ORDER BY run_id, timestamp ASC
```

Same signature, same return shape (every requested id present as a key, even with no rows) —
the loop is an implementation detail of the SQLite side, not a promise. Confirm the empty-list
and unknown-id cases match SQLite's before moving on.

- [ ] **Step 4: Commit**

```bash
git status --short dashboard/storage/data/backtest.db   # must print nothing
git add dashboard/backend/database_postgres.py
git commit -m "feat(runs): port BacktestDatabase readers to Postgres"
```

---

### Task 6: The factory, `.env.example`, `render.yaml`

**Files:**
- Modify: `dashboard/backend/database.py` (bottom of file), `.env.example`, `render.yaml`

**Interfaces:**
- Consumes: `db_url.describe_database_url`. Produces: `_build_backtest_db()` and the `db`
  singleton — same name, same module, so all 14 importers are untouched.

- [ ] **Step 1: Replace the bare singleton at `database.py:998`**

```python
def _build_backtest_db():
    # AGENT_RUNS_DATABASE_URL only, deliberately: CONTENT_DATABASE_URL is scoped to
    # agents/versions/strategies and USERS_DATABASE_URL to accounts; neither may
    # select the run-history database (spec, Decision 3). Do not "simplify" this
    # into a fallback chain.
    database_url = os.getenv("AGENT_RUNS_DATABASE_URL")
    if database_url:
        from dashboard.backend.database_postgres import PostgresBacktestDatabase

        # print(), not logger.info() -- info is invisible under the prod logging
        # config. See users.py's _build_user_store for the full rationale.
        print(f"run history backend: postgres ({describe_database_url(database_url)})")
        return PostgresBacktestDatabase(database_url)
    print("run history backend: sqlite (ephemeral on Render)")
    return BacktestDatabase()


db = _build_backtest_db()
```

Import `describe_database_url` from `dashboard.backend.db_url` at module top (it has no psycopg
dependency); import `PostgresBacktestDatabase` **inside** the function only.

- [ ] **Step 2: `.env.example`** — add an `AGENT_RUNS_DATABASE_URL` block after
      `CONTENT_DATABASE_URL`, in the same commented style. Say what it covers (run history:
      runs, equity, trades, decisions, manifests), that it does not fall back to or from the
      other two, that a fully
      durable deploy sets all three, that it is deliberately not `DATABASE_URL`, and that it
      should be the pooled (`-pooler`) host.

- [ ] **Step 3: `render.yaml`** — add beside its two siblings, with the same "documentation, not
      the mechanism" comment:

```yaml
    # Durable Postgres for backtest run history (agent_runs, equity_timeseries,
    # trades, backtest_decisions, run_manifest). Accounts and content are
    # separate: USERS_DATABASE_URL / CONTENT_DATABASE_URL. Set in the Render
    # dashboard BEFORE merging -- this yaml is documentation, not the mechanism
    # (prod does not sync from it; see CLAUDE.md "Prod deploy reality").
    - key: AGENT_RUNS_DATABASE_URL
      sync: false
```

- [ ] **Step 4: Run** `pytest dashboard/backend/tests/ -q`. Expected: green. The factory takes
      the SQLite branch (Task 1 stripped the var), so nothing changes.

- [ ] **Step 5: Commit**

```bash
git status --short dashboard/storage/data/backtest.db   # must print nothing
git add dashboard/backend/database.py .env.example render.yaml
git commit -m "feat(runs): select the run-history backend from AGENT_RUNS_DATABASE_URL"
```

---

### Task 7: Register the twin and make the five parity guards green

**Files:**
- Modify: `dashboard/backend/tests/test_store_twin_parity.py`

**Interfaces:**
- Consumes: Tasks 3–5. Produces: static parity coverage for the new twin.

- [ ] **Step 1: Add the seventh `_TWINS` entry**

```python
    (
        "dashboard.backend.database",
        "BacktestDatabase",
        "dashboard.backend.database_postgres",
        "PostgresBacktestDatabase",
    ),
```

Nothing else in that file changes — `_TWIN_IDS` and all five parametrized tests derive from
`_TWINS`. Note that `test_every_postgres_twin_module_is_registered` would have failed the moment
Task 3 created the file, so the suite has been red since then by design; this is the step that
closes it.

- [ ] **Step 2: Run and fix until green** —
      `pytest dashboard/backend/tests/test_store_twin_parity.py -v`. Expect real findings here;
      each one is a genuine port defect:
      - *method missing* → a reader/writer skipped in Tasks 4–5.
      - *signature mismatch* → a parameter copied from the spec instead of the live class
        (`insert_equity_points(self, run_id, points, replace=True)` is the classic).
      - *`sqlite-only=[…]` / `postgres-only=[…]`* → a column in one DDL and not the other. The
        check is **bidirectional**; `actions_trace_ref` is on both sides thanks to Task 2.
      - *lazy migration missing* → an `ADD COLUMN IF NOT EXISTS` absent from Task 3 Step 3.

- [ ] **Step 3: Run the full suite** — `pytest dashboard/backend/tests/ -q`. Expected: green.

- [ ] **Step 4: Commit**

```bash
git status --short dashboard/storage/data/backtest.db   # must print nothing
git add dashboard/backend/tests/test_store_twin_parity.py
git commit -m "test(runs): register the run-history twin in the parity registry"
```

---

### Task 8: Dispatch tests (no live Postgres)

**Files:**
- Create: `dashboard/backend/tests/test_backtest_db_postgres.py`

**Interfaces:**
- Consumes: `_build_backtest_db`, `PostgresBacktestDatabase`. Produces: the ordinary-CI half of
  the mirror suite. Task 9 appends the `@pg_only` half to the same file.

These all run in ordinary CI with no Postgres server. Model them on
`test_agent_store_postgres.py`'s dispatch section:

- [ ] **Step 1: The file header** — docstring naming both tiers, plus the local `@pg_only` recipe
      (`docker run --rm -e POSTGRES_PASSWORD=test -e POSTGRES_DB=atl_test -p 5433:5432
      postgres:18-alpine` then `export TEST_POSTGRES_URL=…`), and the standard `pg_only` marker:

```python
TEST_POSTGRES_URL = os.getenv("TEST_POSTGRES_URL")

pg_only = pytest.mark.skipif(
    not TEST_POSTGRES_URL,
    reason="TEST_POSTGRES_URL not set; skipping live-Postgres tests",
)
```

`TEST_POSTGRES_URL` must be a **module-level global** — `test_postgres_url_guard.py`
monkeypatches it on the module object in Task 9.

- [ ] **Step 2: The five dispatch tests**

- `test_build_backtest_db_defaults_to_sqlite` — var unset ⇒ `BacktestDatabase`.
- `test_build_backtest_db_picks_postgres_when_url_set` — monkeypatch the class with a
  `FakePostgresBacktestDatabase` capturing `database_url`.
- `test_build_backtest_db_ignores_content_and_users_database_url` — set both siblings, assert
  SQLite. This is the no-fallback-chain guarantee, and it is the test that would catch someone
  "simplifying" the factory later.
- `test_build_backtest_db_never_prints_the_credentials` — `capsys`; assert the secret is absent
  **and** that the exact line `run history backend: postgres (host/db)` is present.
- `test_unreachable_postgres_raises_instead_of_falling_back` — real
  `PostgresBacktestDatabase("postgresql://u:p@127.0.0.1:1/nope?connect_timeout=2")` ⇒
  `psycopg.OperationalError`. A closed port refuses instantly; no server needed.
- `test_malformed_url_is_rejected_before_psycopg_can_echo_it` — a URL containing a fake secret ⇒
  `ValueError`, and the secret must not appear in the exception text.

- [ ] **Step 3: Run** `pytest dashboard/backend/tests/test_backtest_db_postgres.py -v`.
      Expected: all pass, none skipped (these are not `@pg_only`).

- [ ] **Step 4: Commit**

```bash
git status --short dashboard/storage/data/backtest.db   # must print nothing
git add dashboard/backend/tests/test_backtest_db_postgres.py
git commit -m "test(runs): dispatch tests for the run-history backend factory"
```

---

### Task 9: `@pg_only` behavioral mirror suite + destructive-fixture guard

**Files:**
- Modify: `dashboard/backend/tests/test_backtest_db_postgres.py`
- Modify: `dashboard/backend/tests/test_postgres_url_guard.py`

**Interfaces:**
- Produces: `pg_backtest_db` fixture; live-Postgres behavioral parity coverage.

- [ ] **Step 1: The destructive fixture — guard call FIRST**

```python
@pytest.fixture
def pg_backtest_db():
    require_local_postgres_url(TEST_POSTGRES_URL)
    from dashboard.backend.database_postgres import PostgresBacktestDatabase

    store = PostgresBacktestDatabase(TEST_POSTGRES_URL)
    with store._get_connection() as conn:
        with conn.cursor() as cur:
            # children first, then parents -- the FKs are enforced here
            cur.execute("DELETE FROM equity_timeseries")
            cur.execute("DELETE FROM trades")
            cur.execute("DELETE FROM backtest_decisions")
            cur.execute("DELETE FROM run_manifest")
            cur.execute("DELETE FROM agent_runs")
    yield store
```

`require_local_postgres_url` must be the **first statement** — it raises unless the host is
`localhost`/`127.0.0.1`/`::1`, which is what keeps these unconditional `DELETE`s away from the
prod Neon project. (Verified: it rejects the `ATL-runs-main` endpoint host.)

- [ ] **Step 2: Wire it into `test_postgres_url_guard.py`** — two edits, both required:

1. Add to the module-level fixture re-import block near the top:
   `from dashboard.backend.tests.test_backtest_db_postgres import pg_backtest_db  # noqa: F401`
   (pytest needs the fixture visible in *this* module's namespace).
2. Add `("pg_backtest_db", "dashboard.backend.tests.test_backtest_db_postgres")` to the
   `test_destructive_fixture_refuses_remote_url` parametrize list.

That list is hand-maintained and nothing enforces membership — it already omits
`pg_portfolio_store`. Treat this step as a checklist item, not something CI will remind you of.

- [ ] **Step 3: The behavioral tests** (all `@pg_only`, names ending `_postgres`)

Round-trip through public methods only — never raw SQL with `?` placeholders copied from
SQLite-era test files. Cover, at minimum:

- Every ported writer/reader round-trips (insert, then fetch through a *different* accessor to
  prove persistence rather than echo).
- **The three upsert paths**, including the leaderboard case: re-insert an existing `run_id`
  that already has equity rows — the FK-sensitive one.
- **Both timestamp divergences**: after a re-insert, `created_at` unchanged **and** `updated_at`
  advanced. This is the test that catches the copy-paste failure described in Task 4 Step 1.
- **`insert_equity_points` semantics**: a rerun with a *shorter* timestamp set leaves no
  leftovers from the longer curve; `replace=False` appends; an empty list is a no-op, not a wipe.
- **The uniqueness constraint** — the only thing standing in for the missing static guard
  (Task 3 Step 4): inserting a duplicate `(run_id, timestamp)` updates rather than duplicating.
- **FK cascade** on `delete_run`, plus the explicit `run_manifest` deletion no FK covers.
- **`clear_all`** empties all five tables.
- **Delegation**: `put_idempotency`/`get_idempotency` round-trip, and the row lands in SQLite —
  assert `agent_runs`-style Postgres tables are untouched and the embedded store has it.
- **`get_equity_curves`** batched behavior: multiple ids in one call, unknown id ⇒ empty list
  key present, empty input ⇒ empty dict.

- [ ] **Step 4: Verify in CI, not locally.** The tier skips here. Push, then:

```bash
gh run list --branch feat/runs-database-url --limit 1
gh run view <run-id> --log | grep -E "test_backtest_db_postgres|passed|skipped" | tail -30
```

Expected: the `@pg_only` tests **ran** (not skipped) and passed. A skipped tier in CI means
`TEST_POSTGRES_URL` did not reach the step — check `test_ci_postgres_wired`.

- [ ] **Step 5: Commit**

```bash
git status --short dashboard/storage/data/backtest.db   # must print nothing
git add dashboard/backend/tests/test_backtest_db_postgres.py dashboard/backend/tests/test_postgres_url_guard.py
git commit -m "test(runs): live-Postgres mirror suite for the run-history twin"
```

---

### Task 10: Delete the raw-sqlite3 startup debug blocks

**Files:**
- Modify: `dashboard/backend/app.py`

**Interfaces:**
- Consumes: nothing. Produces: a startup log where the only run-history line is the truthful one.

`app.py:109–135` and `app.py:137–153` are two near-duplicate blocks that open
`sqlite3.connect(str(DB_PATH))` directly and print `agent_runs` counts, bypassing the `db`
singleton — the one seam leak in the codebase. After this change they would print counts from
the *ephemeral SQLite file* on every boot, immediately beside `run history backend: postgres
(…)`, which is this design's only misconfiguration tripwire. Two contradictory numbers is worse
than none: it teaches the reader to distrust the tripwire.

- [ ] **Step 1: Delete both blocks** (lines 109–153 inclusive, from `# DEBUG: Database location`
      through the second block's trailing `print()`), leaving `print("🚀 Starting API server...")`
      and the `📊 Backtesting: …` line that follows. Drop the now-unused `import sqlite3` at
      `app.py:105` if nothing else in `startup_event` uses it — check first.

- [ ] **Step 2: Optional replacement.** If a boot-time row count is still wanted, one line
      through the singleton, which reports whichever backend is live:
      `print(f"run history: {len(db.get_all_runs())} runs")`. Prefer deleting outright; the
      count was debug scaffolding, and `get_all_runs()` is a full-table scan at every boot.

- [ ] **Step 3: Run** `pytest dashboard/backend/tests/ -q` and check the app still starts:
      `python -c "import dashboard.backend.app"` is **not** acceptable here (it runs lazy DDL
      against the seed DB). Use `DATABASE_PATH=/tmp/atl-smoke.db uvicorn dashboard.backend.app:app
      --port 8123` and confirm the startup lines, then stop it.

- [ ] **Step 4: Commit**

```bash
git status --short dashboard/storage/data/backtest.db   # must print nothing
git add dashboard/backend/app.py
git commit -m "refactor: drop raw-sqlite3 startup debug blocks"
```

---

### Task 11: Backfill script

**Files:**
- Create: `dashboard/scripts/backfill_runs_to_postgres.py`
- Create: `dashboard/backend/tests/test_backfill_runs.py`

**Interfaces:**
- Consumes: a source SQLite file + `AGENT_RUNS_DATABASE_URL`. Produces: an idempotent one-time
  migration, re-runnable safely.

Until this runs, prod's `/runs` listing is **empty** — run it immediately after the first green
deploy.

- [ ] **Step 1: Two traps to design around, before writing any code**

1. **Do not let the import mutate the seed DB.** Importing `dashboard.backend.database` builds
   the singleton, and the Postgres twin constructs an embedded `BacktestDatabase()` against
   `DATABASE_PATH` — which defaults to the committed seed. Set `DATABASE_PATH` to a throwaway
   path **before** the first backend import, exactly as `conftest.py` does.
2. **Read the source read-only.** `sqlite3.connect(f"file:{src}?mode=ro", uri=True)` — the
   seed-integrity test uses `immutable=1` for the same reason: to avoid creating `-wal`/`-shm`
   sidecars beside a committed file.

- [ ] **Step 2: The script**

`argparse` with `--source` (default: the committed seed path), `--dry-run`, matching
`refresh_daily_leaderboard.py`'s conventions — `print()` for info, `print(…, file=sys.stderr)`
for errors, `return`ing an exit code from `main()` with `raise SystemExit(main())`.

Order is fixed by the enforced FKs: **`agent_runs` first**, then `equity_timeseries`, `trades`,
`backtest_decisions`, `run_manifest`. Write through the twin's public methods (`insert_run`,
`insert_equity_points`, …) so idempotency comes free from the same upserts — never raw SQL.

Report what it actually moved, per table, and compare against the source counts. Expect roughly
17 `agent_runs` and 2,102 `equity_timeseries` rows with the other three empty — treat those as
order-of-magnitude, not assertions; re-count at run time.

- [ ] **Step 3: The test** (`@pg_only`): build a small source SQLite via `BacktestDatabase`
      against a `tmp_path` file, run the backfill into the test Postgres, assert row counts and
      spot-check one full run (row + curve + trades), then **run it again** and assert the counts
      are unchanged. The re-run assertion is the point of the test.

- [ ] **Step 4: Commit**

```bash
git status --short dashboard/storage/data/backtest.db   # must print nothing
git add dashboard/scripts/backfill_runs_to_postgres.py dashboard/backend/tests/test_backfill_runs.py
git commit -m "feat(runs): idempotent backfill script for run history"
```

---

### Task 12: Docs, then the rollout sequence

**Files:**
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: everything above. Produces: a merge-ready PR and a correct issue trail.

- [ ] **Step 1: `CLAUDE.md`** — three edits:
      (a) add a `AGENT_RUNS_DATABASE_URL` bullet beside its two siblings, same level of detail;
      (b) fix the now-ambiguous "Persisted stores (protocol runs, strategies) live in this DB"
      line — protocol runs still do, backtest run history no longer does;
      (c) update the "Prod deploy reality" gotcha: run history no longer evaporates once this
      ships (accounts, content, and now runs are all durable; the ephemeral seed still backs
      protocol/idempotency state).

- [ ] **Step 2: Open the PR as a DRAFT** while the branch is in flight. First line of the body:
      `DO NOT MERGE until the backfill script (Task 11) is in the diff and has been dry-run.`
      Keep the title short (`feat: durable run history`); details go in the body, briefly.

- [x] **Step 3: Set `AGENT_RUNS_DATABASE_URL` in the Render dashboard** — **done 2026-07-29,
      ahead of the code**, so the usual "unset silently selects ephemeral SQLite" merge gate is
      already discharged. Set via the Render API on the web service `srv-d7lbmpjbc2fs73bcr6t0`
      (*AgenticTrading*) only — the Discord-bot worker carries neither `CONTENT_DATABASE_URL`
      nor `USERS_DATABASE_URL` and reaches the backend over HTTP, so it does not get this one
      either. Value is the pooled (`-pooler`) `ATL-runs-main` string, verified byte-identical by
      SHA-256 round-trip. **Setting it did not trigger a deploy** (native autoDeploy is off; the
      CI hook drives every deploy), so the running instance has not yet seen it — it lands with
      the merge deploy, which is the ordering we want. Re-verify it is still present before
      merging (`GET /v1/services/<id>/env-vars`); a rotation or a teammate's edit re-opens the
      gate silently.

- [ ] **Step 4: Merge → auto-deploy → verify the log line** reads
      `run history backend: postgres (ep-…-pooler…/neondb)`. A typo'd or staging URL shows up
      here because the line names host/db (credentials never).

- [ ] **Step 5: Run the backfill** against prod immediately after the first green deploy.
      Re-verify `/runs` is no longer empty.

- [ ] **Step 6: Follow-ups — file exactly two issues, and correct one.** Per the spec's pass-2
      audit, three of the five originally-listed follow-ups are already fixed on `main`; filing
      them would create dead issues.
      - File: **hot-half durability** (`idempotency_keys`, `protocol_runs`, `protocol_steps`) —
        #140 stays open as the tracker.
      - File: **`dashboard/scripts/backtest.py:397`** — `insert_run` omits the required
        `session_id`, an unhandled `TypeError` on every CLI save
        (`backtest_custom_algo.py:323` is the correct sibling to copy).
      - Correct **#140's own caveat** ("`agent_runs` is the hottest/largest table — every
        backtest step writes decision rows"): it is wrong, and it is the estimate that made this
        work look expensive enough to defer twice. No per-step write touches any of the five
        cold tables.
      - Ask before filing — issues on a shared repo implicitly assign work to others.

- [ ] **Step 7: Commit**

```bash
git status --short dashboard/storage/data/backtest.db   # must print nothing
git add CLAUDE.md
git commit -m "docs: document AGENT_RUNS_DATABASE_URL"
```

---

## Deliberate divergences from the spec (decided while planning)

Three places where this plan does something other than the spec's literal text, each for a
stated reason. Flip any of them if you disagree — they are noted here so the difference is
visible rather than silently absorbed.

1. **`created_at`/`updated_at` are stamped by a Postgres `DEFAULT` expression**, not app-side as
   the spec says. Same TEXT format (`YYYY-MM-DD HH:MM:SS`, UTC), but it preserves SQLite's
   semantics that the *database* stamps the row, which is what makes the two named divergences
   in Task 4 well-defined instead of clock-dependent.
2. **`delete_run` does not clear local idempotency rows**, though the spec says it does. SQLite's
   `delete_run` leaves them, and the mirror suite asserts the two backends behave identically —
   the spec's version would introduce the divergence it is trying to avoid.
3. **`get_equity_curves` is batched, not ported line-for-line** (Task 5 Step 3) — the SQLite
   loop is one network round-trip per agent over Neon. Signature and return shape are identical.
   This is now recorded in the spec's Performance section too.

## Changes made after the plan, from code review

Work this plan did not anticipate. Recorded here because two of them change things outside the
plan's declared file list.

1. **`DELETE /admin/clear` was removed** (`api/routers/admin.py`). Neither this plan nor the spec
   mentions the endpoint, but making run history durable is exactly what changes its blast
   radius: it calls `db.clear_all()` behind no authentication — the session middleware only
   checks that `X-Session-Id` parses as a UUID, which any caller can mint — and pre-change it
   wiped an ephemeral file that the next redeploy restored. Post-change a single anonymous
   request destroys the only durable copy, with nothing to recover from. Nothing called it, and
   there is no admin tier in this codebase to gate it with, so it is gone rather than
   authenticated. **This is a route-contract change**: both golden sets in
   `test_app_composition.py` (`EXPECTED_ADMIN_ROUTES` and `EXPECTED_FULL_CONTRACT`) were updated.
   `db.clear_all()` itself stays — tests and `backtest_hourly_agent.py --clear` use it.
2. **The twin's `clear_all` no longer delegates to the embedded SQLite store.** That delegation
   could only do harm: the embedded `BacktestDatabase` exists solely for the `idempotency_keys`
   hot half, which `clear_all` deliberately skips, so its DELETEs only ever reached cold tables
   the twin never reads — in a file defaulting to `DATABASE_PATH`, i.e. the committed seed and
   its `lb_*` leaderboard rows.
3. **Timestamp normalisation was added to `BacktestDatabase` as well as the twin**
   (`as_timestamp_text`). Task 4 ported `insert_trades`'s `isoformat()` but not the three other
   timestamp writers, which passed a `datetime` straight through: SQLite absorbed it via
   `sqlite3`'s *deprecated* default adapter (storing a space separator), Postgres rejected it
   outright. Fixing only the twin would have replaced a 500 with a silent cross-store text
   divergence that then inverts when Python removes the adapter, so both sides now convert.
4. **The backfill gained per-run failure isolation and an advisory lock.** "Idempotent" was true;
   "re-runnable to completion" was not — one bad legacy row aborted `main()`, and every rerun
   re-failed on it, so runs ordered after it never migrated. `_restore_created_at` needed more
   than a `try`/`except` because it shares one connection: on Postgres a failed statement aborts
   the whole transaction, so each UPDATE now runs in its own `conn.transaction()` block (which,
   on a pooled connection with nothing yet executed, is an independent `BEGIN`/`COMMIT` rather
   than a `SAVEPOINT` — verified against psycopg 3.3.4; either gives the needed isolation).
   Separately, the trades/decisions skip-logic is a read-then-write across two pooled checkouts,
   so two concurrent invocations would double every row in the two tables that have no unique
   key to collapse them.
5. **A pre-existing schema divergence surfaced and is now pinned, not fixed.**
   `trades.quantity`/`side`/`value` are `NOT NULL` under `CREATE TABLE` but nullable on the
   lazy-ALTER path, because SQLite rejects `ADD COLUMN ... NOT NULL` without a default and these
   are backfilled from the legacy columns afterwards. Latent (every writer coerces before the
   value reaches SQLite) and older than this work; only a full table rebuild would close it.
   `test_database_lazy_migrations.py` now lists it explicitly and fails if a *fourth* divergence
   appears — or if this one is ever resolved and the list goes stale.
