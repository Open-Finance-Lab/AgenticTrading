# Admin Layer PR A: The Rewrite — Create Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the analytics data model the design doc's §6 describes — `user_activity`, `user_daily_facts`, `lifecycle_transitions`, `agent_runs.owner_user_id`, the read-time lifecycle calculator, the compare-and-set day claim, the population-wide operational signals, and a daily job on its own worker thread that fills the fact table once per UTC day at a query count independent of user count — while leaving every existing read path, every existing table and every response shape exactly as it is on `main`, so that a reviewer can accept this PR on the strength of its tests and PR B can rewire the nine endpoints onto data that already has eight days of history behind it.

**Architecture:** The event log stays the write-side source of truth; ingestion (`AnalyticsService.record_server_event`) additionally maintains one current-state row per user (`user_activity`) with a single MIN/MAX upsert, so no recompute path exists. A new `domain/analytics/daily_facts.py` job, driven by a dedicated thread in `domain/analytics/daily_job.py` (not the run-reaper tick), claims the previous UTC day on the existing `analytics_projection_jobs` row, runs eight set-based steps over the whole population and upserts `user_daily_facts` and `lifecycle_transitions`. Every cross-domain read the job needs — ledger aggregates, run-history cost, billing states, credential facts, agent ownership, terminal protocol runs — is a public method on the store that owns the table (both twins), and the analytics package opens no connection but its own; the same rule is applied retroactively to `value_repository.py`'s two credit readers and `backfill.py`'s three hand-rolled reads. Read-time lifecycle (`lifecycle_reads.build_lifecycle_inputs`) and the D24 hygiene items land here without being wired to any route.

**Tech Stack:** Python 3.11+, FastAPI + Pydantic v2 (backend), SQLite (local/tests) and PostgreSQL via psycopg 3 (prod twins), pytest run from the repo root, a `threading.Thread` worker inside the single web process.

**Spec:** `docs/superpowers/specs/2026-09-15-admin-layer-redesign-design.md` — implements **§6 in full** (6.1 principle, 6.2 axes — `user_group` in place of the struck `cohort`, 6.3 freshness tiers as the daily job's contract, 6.4 sources incl. `agent_runs.owner_user_id`, 6.5 `user_activity`, 6.6 lifecycle at read time, 6.7 `user_daily_facts` with `user_group` and `operational_reason_code`, 6.8 `lifecycle_transitions`, 6.9 the daily job on its own thread with the two-field day claim, 6.11 event-log discipline incl. rule 7, 6.12 read budget incl. the wall-clock bound, 6.13's struct is deferred to PR B, 6.14 ownership boundaries rows "PR A", 6.15 statements of fact), **§11** (the `user_activity` sweep exception and the `activated_at` seed), **§12 items 1, 2, 4, 5, 6 and 7**, **§13 row A** including its "Must not" column, and decisions **D21–D24**. §4.3–§4.5 describe the state this plan starts from; §10.1 says PR A changes no response shape.

## Global Constraints

- **Change no response shape** (§13 row A, §10.1, D18): the nine `/api/admin/analytics/*` routes return byte-identical bodies for identical inputs after this PR. `tests/test_admin_analytics_api.py`, `test_admin_analytics_frontend.py` and `test_admin_analytics_value_frontend.py` are the conformance oracle and are edited here only to delete the dead users-list stack's own test (Task 13).
- **Drop no table or column** (§13 row A, §12 item 7): `user_analytics_snapshots`, `user_lifecycle_daily_snapshots`, `states.py`, `lifecycle_backfill.py`, the repair pass and the five-state vocabulary all stay. Every existing read path keeps working on the old tables; PR B moves the reads and then drops.
- **Touch no frontend file** (§13 row A): nothing under `dashboard/frontend/` changes, so no `?v=` bump happens in this PR. If any later edit to this plan does bump one, `test_admin_analytics_frontend.py::test_app_lifecycle_and_cache_versions_are_wired` must be updated in the same step.
- **Add no `cohort`** (D9, §13 row A): the axis is `users.user_group` (PR #465). No column, validator, filter or suggestion list named `cohort` is created; the only legitimate occurrence stays `RetentionCohort.cohort_week`.
- Never commit `dashboard/storage/data/backtest.db` — a bare backend import runs `CREATE TABLE IF NOT EXISTS` against `DATABASE_PATH` and rewrites that seed file; stage files by name (`git add <path> <path>`), never `git add -A`, `git add .` or a bare `git add -u`.
- Run every `pytest` invocation from the repo root.
- Node-driven frontend tests are `pytest.mark.skipif(shutil.which("node") is None, ...)`; this PR adds none and touches none.
- **Both store twins change together** in every task that touches a store, and CI's Postgres tier (`.github/workflows/ci.yml`, `TEST_POSTGRES_URL: postgresql://postgres:test@localhost:5432/atl_test`, the `postgres:18-alpine` service) must be green before merge. Locally: `docker run --rm -d --name atl-pg -e POSTGRES_PASSWORD=atl -p 55432:5432 postgres:18` and `TEST_POSTGRES_URL=postgresql://postgres:atl@localhost:55432/postgres python -m pytest dashboard/backend/tests/ -q`; the `@pg_only` marker fails open, so a skip count that does not drop when the variable is set means the tier never ran.
- **PR 0 and PR T are merged before this branch is cut** (§13 ordering: 0 first, T before A). This plan therefore edits the post-PR-0 `service.py`/`instrumentation.py`/`app.py` (the singleton builds with `project_snapshots=False`; `disable_synchronous_projection()` is called at startup) and the post-PR-T `value_repository.py`/`value_repository_postgres.py` (`ValueAnalyticsStore` is SQLite-only, `PostgresValueAnalyticsStore` exists, `build_value_analytics_store()` is the factory, `_fetchall`/`_user_clause`/`_projection_job_name`/`_current_snapshot_from_row` are module functions, and `test_store_twin_parity.py` carries `_DIALECT_BRANCH_ALLOWLIST`). Where a task quotes "current" text from those files it quotes the text those two plans leave behind; confirm with `git log --oneline -5 -- <file>` before editing.
- Every new `ValueAnalyticsStore` method goes on **both** `ValueAnalyticsStore` and `PostgresValueAnalyticsStore`; `test_store_twin_parity.py::test_postgres_twin_exposes_every_sqlite_method` and `::test_postgres_twin_signatures_match_sqlite` fail the build otherwise (§12 item 2 — this is why PR T went first). **As merged (PR #498, 2026-09-19) that file carries a fourth axis this plan predates:** `_DUPLICATED_BODIES` + `test_duplicated_bodies_match_their_declaration` compare each pair's method bodies by `ast.unparse` and fail on a method that is byte-identical across the twins without being named in the `"PostgresValueAnalyticsStore"` frozenset. A method carrying its own placeholder (`?` vs `%s`, inline or via `_SQL.format(ph=...)`) differs and needs no entry; one that holds no placeholder at all — a pure-Python helper, or a body that only delegates to a shared module-level function — will be identical, and the build goes red as `undeclared`. Either share the implementation between the twins, or add the name with a reason **in the same commit**. The axis is two-way, so a name already declared that later stops matching fails as `diverged`: that is the fix-applied-to-one-copy-only case, and the answer is to fix both, never to delete the name.
- **Both twin constructors refuse a base of the wrong dialect** (PR #498): `ValueAnalyticsStore` raises `TypeError` when the resolved analytics base exposes `database_url`, and `PostgresValueAnalyticsStore` when it does not. Neither parity axis inspects `__init__` (both build their name list from `dir(cls)` and skip `_`-prefixed names), which is why the guard is in the constructor rather than in a test. Production code builds through `build_value_analytics_store()`; a test that instantiates a twin directly must hand it a matching base — a `@pg_only` test the real Postgres base, a SQLite test a plain object. Neither message quotes any part of the connection string, and `test_value_repository_postgres.py::test_dialect_guards_quote_no_part_of_the_connection_string` pins that.
- The daily job's query count is a **constant fixed by this plan** (Task 14 pins it) and no query inside it is parameterised by a single user id (§6.12).
- The docs PR that adds the design document and deletes `docs/superpowers/plans/2026-09-12-user-analytics-architecture.md` lands before this one, so nothing here points at that file as live; where a task carries text forward it cites "the superseded 2026-09-12 plan (`git show c3bbf2ed:docs/superpowers/plans/2026-09-12-user-analytics-architecture.md`)".

## File Map

- `dashboard/backend/domain/analytics/repository.py`, `repository_postgres.py`: three new tables in both DDL constants; `list_daily_subjects`, `list_existing_source_event_ids` on both twins.
- `dashboard/backend/database.py`, `database_postgres.py`: `agent_runs.owner_user_id`; `insert_run(owner_user_id=)`; `aggregate_operator_cost_for_day`.
- `dashboard/backend/domain/analytics/value_repository.py`, `value_repository_postgres.py`: models `UserActivity`, `RecentFactTotals`, `DayEventTotals`, `UserDailyFact`, `LifecycleTransitionRow`; activity upsert/seed, fact upserts, day claim, population operational signals, recompute detection, history copy; the two credit readers rewired onto `CreditsStore`.
- `dashboard/backend/domain/analytics/service.py`: `maintain_activity` flag and `_try_record_activity`.
- `dashboard/backend/domain/analytics/lifecycle.py`: `consecutive_failed_terminal_runs`, `_FAILED_RUN_STATUSES`.
- `dashboard/backend/domain/analytics/lifecycle_reads.py` (new): `build_lifecycle_inputs`.
- `dashboard/backend/domain/analytics/daily_facts.py` (new): `run_daily_facts`, `DailyFactsReport`, `DAILY_FACTS_JOB`.
- `dashboard/backend/domain/analytics/daily_job.py` (new): `start_daily_facts_worker`, `daily_job_interval_seconds`.
- `dashboard/backend/domain/analytics/facts_migration.py` (new): `run_startup_migrations`, `migrate_lifecycle_history`.
- `dashboard/backend/domain/analytics/maintenance.py`: reduced to the two throttled snapshot repairs.
- `dashboard/backend/domain/analytics/backfill.py`: the three `_query_all` reads and `_existing_source_event_ids` moved onto owning-store methods.
- `dashboard/backend/domain/credits/repository.py`, `repository_postgres.py`: `aggregate_commercial_ledger`, `list_credit_activity_timestamps`, `aggregate_ledger_for_day`, `list_account_billing_states`, `list_llm_reservation_rows`, `list_llm_usage_rows`.
- `dashboard/backend/domain/model_providers/repository.py`, `repository_postgres.py`, `repository_common.py`: `DefaultCredentialFacts`, `list_default_credential_facts`, `list_platform_credential_statuses`.
- `dashboard/backend/domain/agents/repository.py`, `repository_postgres.py`: `list_agent_owners`, `list_agent_source_rows`.
- `dashboard/backend/domain/runs/repository.py`: `list_terminal_runs_since`.
- `dashboard/backend/domain/backtesting/engine.py`, `dashboard/scripts/backtest_hourly_agent.py`, `dashboard/backend/api/routers/backtests.py`: the four owner hops.
- `dashboard/backend/api/routers/admin_analytics.py`, `dashboard/backend/domain/analytics/query_service.py`: D24 hygiene.
- `dashboard/backend/app.py`: retention registration moves into the job; the worker starts beside the reaper.
- `dashboard/backend/tests/conftest.py`, `CLAUDE.md`: the new env var.
- Tests: `tests/domain/analytics/{test_repository_contract,test_repository_postgres,test_user_activity,test_lifecycle_reads,test_operational_signals,test_value_repository,test_daily_facts,test_daily_job,test_facts_migration,test_read_budget,test_retention,test_backfill}.py`, `tests/domain/analytics/_store_spies.py`, `tests/{test_backtest_owner_attribution,test_credits_ledger_aggregates,test_run_history_aggregates,test_analytics_maintenance,test_run_lifecycle_unification,test_store_twin_parity,test_architecture_boundaries,test_event_loop_threadpool,test_admin_analytics_api,test_admin_analytics_hygiene}.py`.

---

### Task 1: The three new tables, on both twins

**Files:**
- Modify: `dashboard/backend/domain/analytics/repository.py` — `ANALYTICS_SQLITE_DDL`, insert after the `analytics_projection_jobs` block (lines 161-168) and before `CREATE TABLE IF NOT EXISTS analytics_subject_settings` (line 170)
- Modify: `dashboard/backend/domain/analytics/repository_postgres.py` — `ANALYTICS_POSTGRES_DDL`, insert after the `analytics_projection_jobs` block (lines 155-162) and before `CREATE TABLE IF NOT EXISTS analytics_subject_settings` (line 164)
- Test: `dashboard/backend/tests/domain/analytics/test_repository_contract.py` (extend)
- Test: `dashboard/backend/tests/domain/analytics/test_repository_postgres.py` (extend)

**Interfaces:**
- Consumes: nothing.
- Produces: tables `user_activity`, `user_daily_facts`, `lifecycle_transitions` in both DDL constants. Tasks 3, 9 and 10 write to them.

Carried from the superseded 2026-09-12 plan's Task B1 (`git show c3bbf2ed:docs/superpowers/plans/2026-09-12-user-analytics-architecture.md`, lines 1137-1407) with two amendments from design §6.7 and D9: `user_daily_facts` carries `user_group TEXT NOT NULL DEFAULT 'unknown'` (constrained to the six PR #465 values) **instead of** a nullable `cohort`, and gains a nullable `operational_reason_code TEXT` so "why are users blocked" is a grouped read. The `cohort` index becomes a `user_group` index and a `(snapshot_date, operational_state, operational_reason_code)` index is added for the same grouped read.

Both DDL constants are plain triple-quoted string literals, not f-strings. Keep them that way: `tests/test_store_twin_parity.py::_parse_ddl` reads the source text, and an interpolated block collapses to nothing it can compare.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/backend/tests/domain/analytics/test_repository_contract.py`:

```python
def test_sqlite_declares_the_daily_fact_tables(sqlite_contract):
    store, _admin_id, _user_id = sqlite_contract

    with store._get_connection() as conn:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        fact_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(user_daily_facts)").fetchall()
        }
        activity_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(user_activity)").fetchall()
        }
        transition_columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(lifecycle_transitions)"
            ).fetchall()
        }

    assert {
        "user_activity",
        "user_daily_facts",
        "lifecycle_transitions",
    } <= names
    assert fact_columns == {
        "snapshot_date",
        "user_id",
        "lifecycle_segment",
        "lifecycle_reason_code",
        "operational_state",
        "operational_reason_code",
        "tier",
        "user_group",
        "active",
        "runs_requested",
        "runs_completed",
        "runs_failed",
        "runs_cancelled",
        "operator_cost_micro",
        "own_spend_micro",
        "data_quality",
        "calculated_at",
    }
    # D9: the struck axis must not come back under its old name.
    assert "cohort" not in fact_columns
    assert activity_columns == {
        "user_id",
        "activated_at",
        "last_meaningful_activity_at",
        "updated_at",
    }
    assert transition_columns == {
        "transition_id",
        "user_id",
        "snapshot_date",
        "from_segment",
        "to_segment",
        "inactive_days",
        "data_quality",
        "created_at",
    }


def test_user_group_on_facts_defaults_to_unknown_and_is_constrained(sqlite_contract):
    store, _admin_id, user_id = sqlite_contract

    with store._get_connection() as conn:
        conn.execute(
            """
            INSERT INTO user_daily_facts (
                snapshot_date, user_id, lifecycle_segment, lifecycle_reason_code,
                operational_state, tier, data_quality, calculated_at
            ) VALUES (
                '2026-09-11', ?, 'growing', 'growing_activated_below_core_threshold',
                'healthy', 'unpaid', 'complete', '2026-09-12T00:05:00+00:00'
            )
            """,
            (user_id,),
        )
        row = conn.execute(
            "SELECT user_group, operational_reason_code FROM user_daily_facts"
        ).fetchone()
        assert row["user_group"] == "unknown"
        assert row["operational_reason_code"] is None
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO user_daily_facts (
                    snapshot_date, user_id, lifecycle_segment, lifecycle_reason_code,
                    operational_state, tier, user_group, data_quality, calculated_at
                ) VALUES (
                    '2026-09-10', ?, 'growing', 'growing_activated_below_core_threshold',
                    'healthy', 'unpaid', 'lab', 'complete', '2026-09-11T00:05:00+00:00'
                )
                """,
                (user_id,),
            )


def test_lifecycle_transitions_are_unique_per_user_per_day(sqlite_contract):
    """A retried daily job must not append the same transition twice."""
    store, _admin_id, user_id = sqlite_contract

    with store._get_connection() as conn:
        for _ in range(2):
            conn.execute(
                """
                INSERT OR IGNORE INTO lifecycle_transitions (
                    user_id, snapshot_date, from_segment, to_segment,
                    inactive_days, created_at
                ) VALUES (?, '2026-09-11', 'growing', 'at_risk', 8,
                          '2026-09-12T00:05:00+00:00')
                """,
                (user_id,),
            )
        count = conn.execute(
            "SELECT COUNT(*) FROM lifecycle_transitions"
        ).fetchone()[0]

    assert count == 1
```

`sqlite3` and `pytest` are already imported at the top of that file (lines 5 and 9).

Append to `dashboard/backend/tests/domain/analytics/test_repository_postgres.py` (after `test_postgres_ddl_declares_user_value_projection_storage`):

```python
def test_postgres_ddl_declares_the_daily_fact_tables():
    ddl = pg_module.ANALYTICS_POSTGRES_DDL
    assert "CREATE TABLE IF NOT EXISTS user_activity" in ddl
    assert "CREATE TABLE IF NOT EXISTS user_daily_facts" in ddl
    assert "CREATE TABLE IF NOT EXISTS lifecycle_transitions" in ddl
    for column in (
        "operator_cost_micro",
        "own_spend_micro",
        "runs_completed",
        "data_quality",
        "operational_reason_code",
        "user_group TEXT NOT NULL DEFAULT 'unknown'",
        "tier",
    ):
        assert column in ddl
    assert "cohort" not in ddl
    # Wide counters are BIGINT on Postgres, matching analytics_daily_rollups.
    assert "operator_cost_micro BIGINT" in ddl
    assert "own_spend_micro BIGINT" in ddl
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_repository_contract.py dashboard/backend/tests/domain/analytics/test_repository_postgres.py -v -k "daily_fact or transitions_are_unique or user_group_on_facts"
```

Expected: FAIL. `PRAGMA table_info` returns an empty set for a table that does not exist, so `fact_columns == {...}` fails first in the SQLite test; the `INSERT` in the second test raises `sqlite3.OperationalError: no such table`; the Postgres DDL assertion `"CREATE TABLE IF NOT EXISTS user_activity" in ddl` fails.

- [ ] **Step 3: Add the SQLite DDL**

In `dashboard/backend/domain/analytics/repository.py`, inside `ANALYTICS_SQLITE_DDL`, immediately after this existing block:

```sql
CREATE TABLE IF NOT EXISTS analytics_projection_jobs (
    job_name TEXT PRIMARY KEY,
    window_start TEXT NOT NULL,
    window_end TEXT NOT NULL,
    cursor TEXT,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'complete')),
    updated_at TEXT NOT NULL
);
```

insert:

```sql
CREATE TABLE IF NOT EXISTS user_activity (
    user_id INTEGER PRIMARY KEY,
    activated_at TEXT,
    last_meaningful_activity_at TEXT,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS user_daily_facts (
    snapshot_date TEXT NOT NULL CHECK (length(snapshot_date) = 10),
    user_id INTEGER NOT NULL,
    lifecycle_segment TEXT NOT NULL,
    lifecycle_reason_code TEXT NOT NULL,
    operational_state TEXT NOT NULL
        CHECK (operational_state IN ('blocked', 'needs_attention', 'healthy')),
    operational_reason_code TEXT,
    tier TEXT NOT NULL
        CHECK (tier IN ('unpaid', 'starter', 'invested', 'high_value')),
    user_group TEXT NOT NULL DEFAULT 'unknown'
        CHECK (user_group IN (
            'internal', 'invited', 'organic', 'competition', 'partner', 'unknown'
        )),
    active INTEGER NOT NULL DEFAULT 0 CHECK (active IN (0, 1)),
    runs_requested INTEGER NOT NULL DEFAULT 0 CHECK (runs_requested >= 0),
    runs_completed INTEGER NOT NULL DEFAULT 0 CHECK (runs_completed >= 0),
    runs_failed INTEGER NOT NULL DEFAULT 0 CHECK (runs_failed >= 0),
    runs_cancelled INTEGER NOT NULL DEFAULT 0 CHECK (runs_cancelled >= 0),
    operator_cost_micro INTEGER NOT NULL DEFAULT 0
        CHECK (operator_cost_micro >= 0),
    own_spend_micro INTEGER NOT NULL DEFAULT 0 CHECK (own_spend_micro >= 0),
    data_quality TEXT NOT NULL CHECK (data_quality IN ('complete', 'partial')),
    calculated_at TEXT NOT NULL,
    PRIMARY KEY (snapshot_date, user_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_daily_facts_segment
    ON user_daily_facts(snapshot_date, lifecycle_segment);
CREATE INDEX IF NOT EXISTS idx_daily_facts_user
    ON user_daily_facts(user_id, snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_facts_user_group
    ON user_daily_facts(snapshot_date, user_group);
CREATE INDEX IF NOT EXISTS idx_daily_facts_tier
    ON user_daily_facts(snapshot_date, tier);
CREATE INDEX IF NOT EXISTS idx_daily_facts_operational
    ON user_daily_facts(snapshot_date, operational_state, operational_reason_code);

CREATE TABLE IF NOT EXISTS lifecycle_transitions (
    transition_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    snapshot_date TEXT NOT NULL CHECK (length(snapshot_date) = 10),
    from_segment TEXT NOT NULL,
    to_segment TEXT NOT NULL,
    inactive_days INTEGER NOT NULL DEFAULT 0 CHECK (inactive_days >= 0),
    data_quality TEXT NOT NULL DEFAULT 'complete'
        CHECK (data_quality IN ('complete', 'partial')),
    created_at TEXT NOT NULL,
    UNIQUE (user_id, snapshot_date),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_lifecycle_transitions_day
    ON lifecycle_transitions(snapshot_date, to_segment);
```

The `UNIQUE (user_id, snapshot_date)` on `lifecycle_transitions` is load-bearing, not decoration: the daily job retries a failed day on the next tick, and without it a retry appends a second copy of every transition it already wrote. It is a conflict *target*, though, not a write barrier — Task 9's `append_lifecycle_transitions` resolves it with `DO UPDATE`, never `DO NOTHING`, so a retry can **correct** a transition a partial attempt got wrong (design §6.8). `data_quality` on the transition exists for the same reason: a transition derived from a day whose ledger or run step failed is a weaker claim than one derived from a complete day, and "activated user reached fourteen inactive days" firing on incomplete evidence is exactly the mistake the column prevents.

The six-value `CHECK` on `user_group` is the same list `dashboard/backend/domain/user_groups.py::USER_GROUPS` declares; it is spelled out rather than interpolated because the DDL is a plain literal (see the note above Step 1).

- [ ] **Step 4: Add the Postgres DDL**

In `dashboard/backend/domain/analytics/repository_postgres.py`, inside `ANALYTICS_POSTGRES_DDL`, in the same relative position (immediately after the `analytics_projection_jobs` block that ends `updated_at TEXT NOT NULL\n);`):

```sql
CREATE TABLE IF NOT EXISTS user_activity (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    activated_at TEXT,
    last_meaningful_activity_at TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_daily_facts (
    snapshot_date TEXT NOT NULL CHECK (length(snapshot_date) = 10),
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    lifecycle_segment TEXT NOT NULL,
    lifecycle_reason_code TEXT NOT NULL,
    operational_state TEXT NOT NULL
        CHECK (operational_state IN ('blocked', 'needs_attention', 'healthy')),
    operational_reason_code TEXT,
    tier TEXT NOT NULL
        CHECK (tier IN ('unpaid', 'starter', 'invested', 'high_value')),
    user_group TEXT NOT NULL DEFAULT 'unknown'
        CHECK (user_group IN (
            'internal', 'invited', 'organic', 'competition', 'partner', 'unknown'
        )),
    active BOOLEAN NOT NULL DEFAULT FALSE,
    runs_requested INTEGER NOT NULL DEFAULT 0 CHECK (runs_requested >= 0),
    runs_completed INTEGER NOT NULL DEFAULT 0 CHECK (runs_completed >= 0),
    runs_failed INTEGER NOT NULL DEFAULT 0 CHECK (runs_failed >= 0),
    runs_cancelled INTEGER NOT NULL DEFAULT 0 CHECK (runs_cancelled >= 0),
    operator_cost_micro BIGINT NOT NULL DEFAULT 0
        CHECK (operator_cost_micro >= 0),
    own_spend_micro BIGINT NOT NULL DEFAULT 0 CHECK (own_spend_micro >= 0),
    data_quality TEXT NOT NULL CHECK (data_quality IN ('complete', 'partial')),
    calculated_at TEXT NOT NULL,
    PRIMARY KEY (snapshot_date, user_id)
);
CREATE INDEX IF NOT EXISTS idx_daily_facts_segment
    ON user_daily_facts(snapshot_date, lifecycle_segment);
CREATE INDEX IF NOT EXISTS idx_daily_facts_user
    ON user_daily_facts(user_id, snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_facts_user_group
    ON user_daily_facts(snapshot_date, user_group);
CREATE INDEX IF NOT EXISTS idx_daily_facts_tier
    ON user_daily_facts(snapshot_date, tier);
CREATE INDEX IF NOT EXISTS idx_daily_facts_operational
    ON user_daily_facts(snapshot_date, operational_state, operational_reason_code);

CREATE TABLE IF NOT EXISTS lifecycle_transitions (
    transition_id BIGSERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    snapshot_date TEXT NOT NULL CHECK (length(snapshot_date) = 10),
    from_segment TEXT NOT NULL,
    to_segment TEXT NOT NULL,
    inactive_days INTEGER NOT NULL DEFAULT 0 CHECK (inactive_days >= 0),
    data_quality TEXT NOT NULL DEFAULT 'complete'
        CHECK (data_quality IN ('complete', 'partial')),
    created_at TEXT NOT NULL,
    UNIQUE (user_id, snapshot_date)
);
CREATE INDEX IF NOT EXISTS idx_lifecycle_transitions_day
    ON lifecycle_transitions(snapshot_date, to_segment);
```

`CREATE INDEX IF NOT EXISTS` matches by **name only** on Postgres, so it will not notice a definition change on an existing index. These names are new, so that is fine here, but do not rename an index later and expect the statement to rebuild it.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_repository_contract.py dashboard/backend/tests/domain/analytics/test_repository_postgres.py dashboard/backend/tests/test_store_twin_parity.py -v
TEST_POSTGRES_URL=postgresql://postgres:atl@localhost:55432/postgres \
  python -m pytest dashboard/backend/tests/domain/analytics/ -q
```

Expected: PASS on all three files (the parity test's `test_postgres_twin_schema_columns_match_sqlite[PostgresAnalyticsStore]` now compares three more tables' column sets and they match), and the second run reports fewer skips than the first.

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/domain/analytics/repository.py \
        dashboard/backend/domain/analytics/repository_postgres.py \
        dashboard/backend/tests/domain/analytics/test_repository_contract.py \
        dashboard/backend/tests/domain/analytics/test_repository_postgres.py
git status --short
git commit -m "feat: add the user_activity, user_daily_facts and lifecycle_transitions tables"
```

---

### Task 2: Attribute dashboard runs to their owner (`agent_runs.owner_user_id`)

**Files:**
- Modify: `dashboard/backend/database.py` — `agent_runs` CREATE TABLE (lines 129-151), the `token_columns` migration list (lines 356-371), `insert_run` (lines 630-675)
- Modify: `dashboard/backend/database_postgres.py` — `agent_runs` CREATE TABLE (lines 107-134), the `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` block (lines 236-278), `insert_run` (lines 474-583)
- Modify: `dashboard/backend/domain/backtesting/engine.py` — `HourlyBacktester.__init__` (signature lines 180-205, attribute assignment at line 237), the three `db.insert_run(` calls (lines 1748, 1859, 1941)
- Modify: `dashboard/scripts/backtest_hourly_agent.py` — argparse beside `--run-id` (line 223), the `HourlyBacktester(` construction (lines 443-461)
- Modify: `dashboard/backend/api/routers/backtests.py` — `run_backtest_background` signature (lines 1381-1401), the `cmd += ["--run-id", ...]` line (1534), the `_BackgroundThread(` construction (line 3177; kwargs on lines 3178-3197)
- Test: `dashboard/backend/tests/test_backtest_owner_attribution.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `agent_runs.owner_user_id` (nullable INTEGER, both twins) and `insert_run(..., owner_user_id: Optional[int] = None)` on both twins. Task 5's `aggregate_operator_cost_for_day` groups on it.

Unchanged from the superseded 2026-09-12 plan's Task B3 (`git show c3bbf2ed:docs/superpowers/plans/2026-09-12-user-analytics-architecture.md`, lines 1589-1775), with line numbers re-read against the current worktree. Per-user model cost cannot come from the event log: the live path emits `credits_reserved` / `credits_settled` / `credits_refunded`, which measure the user's own credit spend; `model_usage_recorded` carries `cost_micro_usd` only on the historical backfill path. Operator-funded cost lives in `agent_runs.est_cost_usd` and nowhere else, so the run row needs the owner (design §6.4).

Nullable and not backfilled. Rows written before this column, and runs with no authenticated caller such as the leaderboard's scheduled deploys, stay unattributed and are reported as unattributed rather than assigned to anyone.

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/test_backtest_owner_attribution.py`:

```python
"""agent_runs rows carry the authenticated caller who started them.

Operator-funded model cost is in ``est_cost_usd`` on this row and nowhere
else, so without an owner column there is no per-user cost at all.
"""

from __future__ import annotations

import inspect

from dashboard.backend.database import BacktestDatabase


def test_insert_run_accepts_an_owner(tmp_path):
    db = BacktestDatabase(tmp_path / "runs.db")
    db.insert_run(
        run_id="run-owned",
        session_id="session-1",
        agent_name="Agent",
        mode="backtest",
        start_date="2026-09-01",
        end_date="2026-09-02",
        initial_equity=100000.0,
        est_cost_usd=1.25,
        owner_user_id=7,
    )

    conn = db._get_connection()
    try:
        row = conn.execute(
            "SELECT owner_user_id, est_cost_usd FROM agent_runs WHERE run_id = ?",
            ("run-owned",),
        ).fetchone()
    finally:
        conn.close()

    assert row["owner_user_id"] == 7
    assert row["est_cost_usd"] == 1.25


def test_owner_is_optional_and_defaults_to_null(tmp_path):
    """A scheduled leaderboard deploy has no caller and must still insert."""
    db = BacktestDatabase(tmp_path / "runs.db")
    db.insert_run(
        run_id="run-unowned",
        session_id="session-1",
        agent_name="Agent",
        mode="backtest",
        start_date="2026-09-01",
        end_date="2026-09-02",
        initial_equity=100000.0,
    )

    conn = db._get_connection()
    try:
        row = conn.execute(
            "SELECT owner_user_id FROM agent_runs WHERE run_id = ?",
            ("run-unowned",),
        ).fetchone()
    finally:
        conn.close()

    assert row["owner_user_id"] is None


def test_the_owner_reaches_the_subprocess_and_the_engine():
    """The four hops from the route to the row, pinned by source shape."""
    from dashboard.backend.api.routers import backtests
    from dashboard.backend.domain.backtesting import engine

    assert "owner_user_id" in inspect.signature(
        backtests.run_backtest_background
    ).parameters
    assert "--owner-user-id" in inspect.getsource(backtests.run_backtest_background)
    assert "owner_user_id" in inspect.signature(
        engine.HourlyBacktester.__init__
    ).parameters
```

`BacktestDatabase.__init__(self, db_path: Path = None)` takes the path positionally (database.py:107), and `_get_connection()` returns a plain `sqlite3.Connection` that the caller closes (database.py:117-121) — hence the `try/finally` rather than a `with`.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest dashboard/backend/tests/test_backtest_owner_attribution.py -v
```

Expected: FAIL. The first case raises `TypeError: insert_run() got an unexpected keyword argument 'owner_user_id'`; the second raises `sqlite3.OperationalError: no such column: owner_user_id`; the third fails at the first `assert` (`owner_user_id` is not a parameter of `run_backtest_background`).

- [ ] **Step 3: Add the column on both twins**

`dashboard/backend/database.py`, in `_init_schema`'s `CREATE TABLE IF NOT EXISTS agent_runs` body, change:

```sql
                est_cost_usd REAL DEFAULT 0,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
```

to:

```sql
                est_cost_usd REAL DEFAULT 0,
                metadata TEXT,
                owner_user_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
```

and in `_migrate_schema`'s `token_columns` list, change the last entry:

```python
                ("metadata",
                 "ALTER TABLE agent_runs ADD COLUMN metadata TEXT"),
            ]
```

to:

```python
                ("metadata",
                 "ALTER TABLE agent_runs ADD COLUMN metadata TEXT"),
                # Analytics attribution (design §6.4): the authenticated caller
                # who started a dashboard backtest. Nullable and never
                # backfilled -- scheduled leaderboard deploys have no caller.
                ("owner_user_id",
                 "ALTER TABLE agent_runs ADD COLUMN owner_user_id INTEGER"),
            ]
```

The loop that follows (`for col_name, add_column_sql in token_columns: if col_name not in columns: ...`) already applies every entry, and the literal-SQL form is what the twin-parity guard's lazy-migration check reads.

`dashboard/backend/database_postgres.py`, in `_init_schema`'s `CREATE TABLE IF NOT EXISTS agent_runs`, change:

```sql
                        est_cost_usd DOUBLE PRECISION DEFAULT 0,
                        metadata TEXT,
                        created_at TEXT NOT NULL {created_at_default},
```

to:

```sql
                        est_cost_usd DOUBLE PRECISION DEFAULT 0,
                        metadata TEXT,
                        owner_user_id INTEGER,
                        created_at TEXT NOT NULL {created_at_default},
```

and after:

```python
                cur.execute(
                    "ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS metadata TEXT"
                )
```

insert:

```python
                cur.execute(
                    "ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS owner_user_id INTEGER"
                )
```

There is deliberately **no** foreign key to `users(id)`: run history lives in its own Neon project (`AGENT_RUNS_DATABASE_URL`, `ATL-runs-main`), and the users table is in a different database entirely.

- [ ] **Step 4: Extend `insert_run` on both twins**

Add `owner_user_id: Optional[int] = None` as the **last** keyword parameter on both signatures, identically — `test_store_twin_parity.py::test_postgres_twin_signatures_match_sqlite` compares parameter order and defaults, so a different position fails it.

`dashboard/backend/database.py`, change the signature's last line and the statement:

```python
                   est_cost_usd: float = 0.0,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
```

to:

```python
                   est_cost_usd: float = 0.0,
                   metadata: Optional[Dict[str, Any]] = None,
                   owner_user_id: Optional[int] = None) -> None:
```

and:

```python
        cursor.execute("""
            INSERT OR REPLACE INTO agent_runs
            (run_id, session_id, agent_name, mode, start_date, end_date,
             initial_equity, final_equity, total_return, sharpe_ratio,
             max_drawdown, num_trades, llm_model,
             llm_calls, llm_decisions, input_tokens, output_tokens,
             est_cost_usd, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (run_id, session_id, agent_name, mode, start_date, end_date,
              initial_equity, final_equity, total_return, sharpe_ratio,
              max_drawdown, num_trades, llm_model,
              llm_calls, llm_decisions, input_tokens, output_tokens,
              est_cost_usd,
              json.dumps(metadata) if metadata is not None else None))
```

to:

```python
        cursor.execute("""
            INSERT OR REPLACE INTO agent_runs
            (run_id, session_id, agent_name, mode, start_date, end_date,
             initial_equity, final_equity, total_return, sharpe_ratio,
             max_drawdown, num_trades, llm_model,
             llm_calls, llm_decisions, input_tokens, output_tokens,
             est_cost_usd, metadata, owner_user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (run_id, session_id, agent_name, mode, start_date, end_date,
              initial_equity, final_equity, total_return, sharpe_ratio,
              max_drawdown, num_trades, llm_model,
              llm_calls, llm_decisions, input_tokens, output_tokens,
              est_cost_usd,
              json.dumps(metadata) if metadata is not None else None,
              owner_user_id))
```

`dashboard/backend/database_postgres.py`, change the signature the same way, and the statement:

```python
                    INSERT INTO agent_runs
                    (run_id, session_id, agent_name, mode, start_date, end_date,
                     initial_equity, final_equity, total_return, sharpe_ratio,
                     max_drawdown, num_trades, llm_model,
                     llm_calls, llm_decisions, input_tokens, output_tokens,
                     est_cost_usd, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id) DO UPDATE SET
```

to:

```python
                    INSERT INTO agent_runs
                    (run_id, session_id, agent_name, mode, start_date, end_date,
                     initial_equity, final_equity, total_return, sharpe_ratio,
                     max_drawdown, num_trades, llm_model,
                     llm_calls, llm_decisions, input_tokens, output_tokens,
                     est_cost_usd, metadata, owner_user_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::integer)
                    ON CONFLICT (run_id) DO UPDATE SET
```

then in the `DO UPDATE SET` list, change:

```python
                        metadata = EXCLUDED.metadata,
                        updated_at = to_char(now() AT TIME ZONE 'utc', 'YYYY-MM-DD HH24:MI:SS')
```

to:

```python
                        metadata = EXCLUDED.metadata,
                        owner_user_id = EXCLUDED.owner_user_id,
                        updated_at = to_char(now() AT TIME ZONE 'utc', 'YYYY-MM-DD HH24:MI:SS')
```

and the parameter tuple's tail:

```python
                        est_cost_usd,
                        json.dumps(metadata) if metadata is not None else None,
                    ),
```

to:

```python
                        est_cost_usd,
                        json.dumps(metadata) if metadata is not None else None,
                        owner_user_id,
                    ),
```

`%s::integer` rather than a bare `%s`: the value is frequently `None`, and psycopg sends a bare `None` as an untyped NULL (OID 0); the cast is cheap insurance and no SQLite test can catch its absence.

- [ ] **Step 5: Thread the owner from the route to the row**

Four hops, each mirroring how `--run-id` and `live_run_id` already travel:

1. `dashboard/backend/api/routers/backtests.py`, `run_backtest_background`: change the signature's tail

```python
    execution_handoff_payload: Optional[str] = None,
    universe_selection: Optional[Dict[str, Any]] = None,
):
```

to:

```python
    execution_handoff_payload: Optional[str] = None,
    universe_selection: Optional[Dict[str, Any]] = None,
    owner_user_id: Optional[int] = None,
):
```

and change:

```python
        cmd += ["--run-id", resolved_live_run_id, "--progress-file", progress_file]
```

to:

```python
        cmd += ["--run-id", resolved_live_run_id, "--progress-file", progress_file]
        if owner_user_id is not None:
            cmd += ["--owner-user-id", str(int(owner_user_id))]
```

2. Same file, the `_BackgroundThread(` call: change

```python
            "execution_handoff_payload": execution_handoff_payload,
            **({"universe_selection": universe_selection} if universe_selection is not None else {}),
        },
```

to:

```python
            "execution_handoff_payload": execution_handoff_payload,
            # The caller's OWN account, the same user_id _backtest_owner_key
            # bills the slot to -- never the session the results file under,
            # which for a built-in agent is the agent's, not the caller's.
            "owner_user_id": user_id,
            **({"universe_selection": universe_selection} if universe_selection is not None else {}),
        },
```

`user_id` is the variable already passed to `_try_acquire_backtest_slot(... user_id=user_id)` about 26 lines above (line 3151; the `_BackgroundThread(` construction starts at line 3177).

3. `dashboard/scripts/backtest_hourly_agent.py`, after

```python
    parser.add_argument("--run-id", default=None, help="Preset run id (used for live progress + DB row)")
```

insert:

```python
    parser.add_argument(
        "--owner-user-id",
        type=int,
        default=None,
        help="Authenticated caller who started this run (analytics attribution)",
    )
```

and in the `HourlyBacktester(` construction, change:

```python
        live_run_id=args.run_id,
        progress_file=args.progress_file,
```

to:

```python
        live_run_id=args.run_id,
        owner_user_id=args.owner_user_id,
        progress_file=args.progress_file,
```

4. `dashboard/backend/domain/backtesting/engine.py`, `HourlyBacktester.__init__`: change

```python
        live_run_id: str = None,
        progress_file: str = None,
```

to:

```python
        live_run_id: str = None,
        owner_user_id: Optional[int] = None,
        progress_file: str = None,
```

change:

```python
        self.live_run_id = (live_run_id or "").strip() or None
        self.progress_file = (progress_file or "").strip() or None
```

to:

```python
        self.live_run_id = (live_run_id or "").strip() or None
        self.owner_user_id = int(owner_user_id) if owner_user_id is not None else None
        self.progress_file = (progress_file or "").strip() or None
```

and in **all three** `db.insert_run(` calls (the agent run at line 1748 whose last argument is `metadata=self._agent_run_metadata(),`, the buy-and-hold baseline at line 1859 and the DJIA baseline at line 1941 whose last arguments are `metadata=self._run_metadata(...)`), add `owner_user_id=self.owner_user_id,` as the final keyword argument, e.g.:

```python
            est_cost_usd=est_cost,
            metadata=self._agent_run_metadata(),
            owner_user_id=self.owner_user_id,
        )
```

Baseline rows carry the owner too so that a per-owner listing of a run's companions is complete; their `est_cost_usd` is 0, so they contribute nothing to operator cost.

Leave the leaderboard's `insert_run` calls (`domain/leaderboard/service.py`) and the paper-trading call (`api/routers/paper_trading.py`) alone. The leaderboard runs on a schedule with no caller. Paper trading is a different mode and not operator-funded LLM spend.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
python -m pytest dashboard/backend/tests/test_backtest_owner_attribution.py -v
python -m pytest dashboard/backend/tests/test_store_twin_parity.py -q
python -m pytest dashboard/backend/tests/ -q -k "backtest"
TEST_POSTGRES_URL=postgresql://postgres:atl@localhost:55432/postgres \
  python -m pytest dashboard/backend/tests/ -q -k "postgres and (backtest or database)"
```

Expected: PASS on all four. `test_store_twin_parity.py::test_postgres_twin_repeats_every_sqlite_lazy_migration[PostgresBacktestDatabase]` is the case that would catch a missing `ADD COLUMN IF NOT EXISTS` on the Postgres side.

- [ ] **Step 7: Commit**

```bash
git add dashboard/backend/database.py dashboard/backend/database_postgres.py \
        dashboard/backend/domain/backtesting/engine.py \
        dashboard/scripts/backtest_hourly_agent.py \
        dashboard/backend/api/routers/backtests.py \
        dashboard/backend/tests/test_backtest_owner_attribution.py
git status --short
git commit -m "feat: record the owner on dashboard backtest rows"
```

---

### Task 3: Ingestion maintains `user_activity`, seeded from `user_analytics_snapshots`

**Files:**
- Modify: `dashboard/backend/domain/analytics/value_repository.py` — add `UserActivity` and `_activity_from_row` after `UserLifecycleDailySnapshot` (post-PR-T lines 392-405); add `record_activity`, `get_activity`, `list_activity`, `seed_activity_from_snapshots` to `ValueAnalyticsStore` after `has_daily_before`; export `UserActivity` in `__all__`
- Modify: `dashboard/backend/domain/analytics/value_repository_postgres.py` — the same four methods on `PostgresValueAnalyticsStore`, after its `has_daily_before`; extend the `from .value_repository import (...)` block with `_activity_from_row` and `UserActivity`
- Modify: `dashboard/backend/domain/analytics/service.py` — `AnalyticsService.__init__` (lines 38-53), new `_try_record_activity` after `_try_recalculate_snapshots` (line 77), `record_server_event` (lines 169-182), `_build_analytics_service` (lines 235-245 as left by PR 0 Task 2 and PR T Task 2 step 6)
- Test: `dashboard/backend/tests/domain/analytics/test_user_activity.py` (create)
- Test: `dashboard/backend/tests/domain/analytics/test_service.py` (extend)

**Interfaces:**
- Consumes: `user_activity` (Task 1); `build_value_analytics_store` (PR T).
- Produces:
  - `UserActivity(user_id: int, activated_at: datetime | None, last_meaningful_activity_at: datetime | None, updated_at: datetime)` in `value_repository.py`
  - `record_activity(user_id: int, *, occurred_at: datetime, activating: bool, now: datetime) -> None`, `get_activity(user_id: int) -> UserActivity | None`, `list_activity(user_ids: Sequence[int] | None = None) -> dict[int, UserActivity]`, `seed_activity_from_snapshots(*, now: datetime) -> int` on both value twins
  - `AnalyticsService(store, *, state_store=None, value_store=None, project_snapshots: bool = False, maintain_activity: bool = False)` — a **new** flag beside the existing one; `project_snapshots` and `_try_recalculate_snapshots` stay in place, switched off, until PR B deletes them (design §6.15 item 1).

  Task 4 reads `UserActivity`; Task 9's `events` and `ledger` steps correct the same row; Task 10's startup migration calls `seed_activity_from_snapshots`.

Carried from the superseded 2026-09-12 plan's Task B4 (`git show c3bbf2ed:...`, lines 1779-2096) with three amendments: (1) the constructor flag is **added**, not renamed — renaming `project_snapshots` would break `states.py`, `test_states.py` and PR 0's `test_built_singleton_does_not_synchronously_project_snapshots` on a code path PR B, not PR A, deletes (§13 row B); (2) `instrumentation.py` is **not** edited — PR 0 disarmed its fallback and PR B deletes it; (3) the seed from `user_analytics_snapshots` is added per design §11 and §12 item 6: without it every existing user's `activated_at` would be lost on the night PR B drops the old table, while `test_activated_at_holds_the_earliest_success` stayed green.

This is the change that makes the 2026-09-11 outage structurally impossible: one upsert of two timestamps replaces a 180-day read of the event log, and it happens on the write the event log already performs.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/domain/analytics/test_user_activity.py`:

```python
"""Ingestion maintains one current-state row per user and never reads history."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from dashboard.backend.domain.analytics import rollups as rollups_module
from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.service import AnalyticsService
from dashboard.backend.domain.analytics.value_repository import (
    build_value_analytics_store,
)
from dashboard.backend.tests.domain.analytics.test_value_repository import (
    _value_snapshot,
)


NOW = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)


class CountingAnalyticsStore(AnalyticsStore):
    """The real SQLite store, counting every per-user history read."""

    def __init__(self, path):
        super().__init__(path)
        self.event_reads = 0

    def list_user_events(self, *args, **kwargs):
        self.event_reads += 1
        return super().list_user_events(*args, **kwargs)


def _fixture(tmp_path, monkeypatch):
    path = tmp_path / "activity.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                email TEXT NOT NULL,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                user_group TEXT NOT NULL DEFAULT 'unknown',
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO users VALUES (1, 'user@example.test', 'User', 'x', 'user', "
            "'unknown', ?)",
            ((NOW - timedelta(days=60)).isoformat(),),
        )
    store = CountingAnalyticsStore(path)

    def _no_history_scan(*_args, **_kwargs):
        raise AssertionError("ingestion must not scan analytics_events")

    monkeypatch.setattr(
        rollups_module.AnalyticsRollupStore, "list_events", _no_history_scan
    )
    value_store = build_value_analytics_store(
        store,
        credits_base=object(),
        provider_base=object(),
        agent_base=object(),
        run_base=object(),
    )
    service = AnalyticsService(
        store,
        value_store=value_store,
        maintain_activity=True,
    )
    return service, store, value_store


def _emit(service, index, occurred_at, *, name="backtest_requested"):
    return service.record_server_event(
        event_name=name,
        user_id=1,
        source_event_id=f"run:{name}:run-{index}",
        source_record_type="run",
        source_record_id=f"run-{index}",
        occurred_at=occurred_at,
        received_at=NOW,
    )


def test_ingestion_never_reads_a_users_history(tmp_path, monkeypatch):
    """Accepting an event costs one upsert, not a scan.

    The 2026-09-11 outage was this path: every accepted lifecycle event
    recomputed a label from a 180-day read of the log it had just appended to.
    """
    service, store, _value_store = _fixture(tmp_path, monkeypatch)
    for index in range(20):
        _emit(service, index, NOW - timedelta(seconds=index))

    assert store.event_reads == 0


def test_activated_at_holds_the_earliest_success(tmp_path, monkeypatch):
    """Activation is the *first* success, in occurred_at order."""
    service, _store, value_store = _fixture(tmp_path, monkeypatch)

    _emit(service, "a", NOW - timedelta(days=3), name="backtest_completed")
    first = value_store.get_activity(1).activated_at

    _emit(service, "b", NOW, name="backtest_completed")
    activity = value_store.get_activity(1)

    assert first == NOW - timedelta(days=3)
    assert activity.activated_at == first
    assert activity.last_meaningful_activity_at == NOW


def test_an_out_of_order_success_moves_activated_at_earlier(tmp_path, monkeypatch):
    """The newest write does not own the value; the earliest instant does.

    Ingestion sees events in arrival order, not occurred_at order: a
    `backtest_completed` can be appended late, replayed from the queue, or
    backdated by the 24-hour acceptance window in `service.py:96-97`. If
    activation were pinned by whichever row landed first, this user's
    activation cohort week -- and therefore their whole column of the
    retention grid -- would be permanently wrong, with nothing able to
    correct it.
    """
    service, _store, value_store = _fixture(tmp_path, monkeypatch)

    for suffix, occurred in (("late", NOW), ("early", NOW - timedelta(days=5))):
        _emit(service, suffix, occurred, name="backtest_completed")

    activity = value_store.get_activity(1)

    assert activity.activated_at == NOW - timedelta(days=5)
    # The activity clock still only advances.
    assert activity.last_meaningful_activity_at == NOW


def test_page_views_and_signups_do_not_advance_the_activity_clock(
    tmp_path, monkeypatch
):
    """Only the meaningful-activity set counts. A visit is not activity."""
    service, _store, value_store = _fixture(tmp_path, monkeypatch)

    service.record_server_event(
        event_name="account_signed_up",
        user_id=1,
        source_event_id="account:account_signed_up:1",
        source_record_type="user",
        source_record_id="1",
        occurred_at=NOW,
        received_at=NOW,
    )

    assert value_store.get_activity(1) is None


def test_a_replayed_event_does_not_touch_the_row(tmp_path, monkeypatch):
    """`created=False` means the log already had it; the clock must not move."""
    service, _store, value_store = _fixture(tmp_path, monkeypatch)

    _emit(service, "once", NOW - timedelta(days=1))
    before = value_store.get_activity(1)
    replay = _emit(service, "once", NOW - timedelta(days=1))

    assert replay.created is False
    assert value_store.get_activity(1) == before


def test_list_activity_returns_every_row_when_no_ids_are_given(tmp_path, monkeypatch):
    service, _store, value_store = _fixture(tmp_path, monkeypatch)
    _emit(service, "x", NOW - timedelta(hours=1))

    everyone = value_store.list_activity()
    some = value_store.list_activity([1])
    none = value_store.list_activity([])

    assert set(everyone) == {1}
    assert some == everyone
    assert none == {}


def test_seed_copies_the_legacy_timestamps_and_is_idempotent(tmp_path, monkeypatch):
    """Design doc SS11: the migration must carry ``activated_at`` across.

    Ingestion only writes ``user_activity`` for events that arrive after the
    deploy, and the daily job's one-day scan cannot see an activation from
    last year. Without this seed every existing user's activation date would
    be lost on the night PR B drops ``user_analytics_snapshots``.
    """
    _service, _store, value_store = _fixture(tmp_path, monkeypatch)
    legacy = _value_snapshot(1)  # activated NOW-20d, last activity NOW-2d
    value_store.upsert_current_snapshot(legacy)

    first = value_store.seed_activity_from_snapshots(now=NOW)
    seeded = value_store.get_activity(1)
    second = value_store.seed_activity_from_snapshots(now=NOW + timedelta(hours=1))
    reseeded = value_store.get_activity(1)

    assert first == 1
    assert seeded.activated_at == legacy.activated_at
    assert seeded.last_meaningful_activity_at == legacy.last_meaningful_activity_at
    assert second == 1  # the upsert touches the row again ...
    assert reseeded.activated_at == seeded.activated_at  # ... and changes nothing
    assert reseeded.last_meaningful_activity_at == seeded.last_meaningful_activity_at


def test_seed_never_regresses_a_row_ingestion_already_advanced(tmp_path, monkeypatch):
    """MIN on activation, MAX on activity: the seed can only fill gaps."""
    service, _store, value_store = _fixture(tmp_path, monkeypatch)
    value_store.upsert_current_snapshot(_value_snapshot(1))  # NOW-20d / NOW-2d
    _emit(service, "earlier", NOW - timedelta(days=30), name="backtest_completed")
    _emit(service, "today", NOW)

    value_store.seed_activity_from_snapshots(now=NOW)
    activity = value_store.get_activity(1)

    assert activity.activated_at == NOW - timedelta(days=30)
    assert activity.last_meaningful_activity_at == NOW
```

`_value_snapshot(user_id)` is the helper `test_value_repository.py` already defines at line 126 (`activated_at=NOW - timedelta(days=20)`, `last_meaningful_activity_at=NOW - timedelta(days=2)` relative to that module's own `NOW`, `2026-09-03T12:00Z`); its absolute instants are what the seed assertions compare against, so the differing `NOW` constants are harmless.

Append to `dashboard/backend/tests/domain/analytics/test_service.py` (it already defines `RecordingStore` and `NOW`; PR 0 added `test_built_singleton_does_not_synchronously_project_snapshots` alongside):

```python
def test_built_singleton_maintains_user_activity(monkeypatch):
    from dashboard.backend.domain.analytics import service as service_module

    monkeypatch.setattr(service_module, "analytics_store", RecordingStore())

    built = service_module._build_analytics_service()

    assert built.maintain_activity is True
    assert built.project_snapshots is False


def test_maintaining_activity_requires_a_value_store():
    with pytest.raises(ValueError):
        AnalyticsService(RecordingStore(), maintain_activity=True)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_user_activity.py dashboard/backend/tests/domain/analytics/test_service.py -v
```

Expected: FAIL. Every `test_user_activity.py` case fails in `_fixture` with `TypeError: AnalyticsService.__init__() got an unexpected keyword argument 'maintain_activity'`; `test_built_singleton_maintains_user_activity` fails with `AttributeError: 'AnalyticsService' object has no attribute 'maintain_activity'`; `test_maintaining_activity_requires_a_value_store` fails because the `TypeError` is not a `ValueError`.

- [ ] **Step 3: Add the model and the upserts to the SQLite twin**

In `dashboard/backend/domain/analytics/value_repository.py`, immediately after the `UserLifecycleDailySnapshot` class (which ends with its `require_timezone` validator), insert:

```python
class UserActivity(BaseModel):
    """One user's current activity clock. One row, overwritten in place.

    Design SS6.5: ``activated_at`` is the first accepted ``backtest_completed``
    and is set once; ``last_meaningful_activity_at`` is the greatest
    ``occurred_at`` of any accepted event in the lifecycle activity set. It is
    current state, not history, and is never swept (SS11).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: int = Field(gt=0)
    activated_at: datetime | None = None
    last_meaningful_activity_at: datetime | None = None
    updated_at: datetime

    @field_validator("activated_at", "last_meaningful_activity_at", "updated_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        return _utc(value) if value is not None else None


def _activity_from_row(row: Any) -> UserActivity:
    """Shared by both twins."""
    return UserActivity(
        user_id=int(_row_value(row, "user_id")),
        activated_at=_optional_timestamp(_row_value(row, "activated_at")),
        last_meaningful_activity_at=_optional_timestamp(
            _row_value(row, "last_meaningful_activity_at")
        ),
        updated_at=_timestamp(_row_value(row, "updated_at")),
    )
```

Then on `ValueAnalyticsStore`, immediately after `has_daily_before` (which ends `return row is not None`), insert:

```python
    def record_activity(
        self,
        user_id: int,
        *,
        occurred_at: datetime,
        activating: bool,
        now: datetime,
    ) -> None:
        """Advance one user's activity timestamps. One statement, no read.

        The two columns move in opposite directions, on purpose.

        ``activated_at`` keeps the **earliest** success, because activation
        is defined as the first server-authoritative ``backtest_completed``
        by ``occurred_at`` -- not by arrival order. Ingestion does not see
        events in occurred_at order: a completion can be appended late,
        replayed, or backdated up to the 24 hours ``service.py:96-97``
        accepts. A plain ``COALESCE(stored, incoming)`` would freeze
        whichever row happened to land first and make the value
        uncorrectable afterwards, which would also silently turn the daily
        job's ``events`` step repair into a no-op. A non-activating event
        passes NULL and the COALESCE pair leaves the stored value alone.

        ``last_meaningful_activity_at`` only ever advances, so a
        late-arriving old event cannot make a user look more dormant than
        they are.

        Timestamps are ISO-8601 UTC text throughout this schema, which orders
        lexicographically, so MIN and MAX over the text are the same
        comparisons as over the instants.

        SQLite's scalar ``max(X, Y)`` and ``min(X, Y)`` both return NULL if
        **either** argument is NULL, which is why each stored value is
        COALESCEd against the incoming one before the comparison rather than
        passed raw.
        """
        subject_id = positive_user_id(user_id)
        occurred = utc_iso(_utc(occurred_at, "occurred_at"))
        values = (
            subject_id,
            occurred if activating else None,
            occurred,
            utc_iso(_utc(now, "now")),
        )
        sql = """
            INSERT INTO user_activity (
                user_id, activated_at, last_meaningful_activity_at, updated_at
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                activated_at = MIN(
                    COALESCE(user_activity.activated_at, excluded.activated_at),
                    COALESCE(excluded.activated_at, user_activity.activated_at)
                ),
                last_meaningful_activity_at = MAX(
                    COALESCE(
                        user_activity.last_meaningful_activity_at,
                        excluded.last_meaningful_activity_at
                    ),
                    excluded.last_meaningful_activity_at
                ),
                updated_at = excluded.updated_at
        """
        with self._analytics_connection() as conn:
            conn.execute(sql, values)

    def get_activity(self, user_id: int) -> UserActivity | None:
        """One user's stored activity row, or None if they have none yet."""
        subject_id = positive_user_id(user_id)
        with self._analytics_connection() as conn:
            row = conn.execute(
                "SELECT * FROM user_activity WHERE user_id = ?", (subject_id,)
            ).fetchone()
        return _activity_from_row(row) if row is not None else None

    def list_activity(
        self,
        user_ids: Sequence[int] | None = None,
    ) -> dict[int, UserActivity]:
        """Activity rows for many users, or for everyone when ``user_ids`` is None.

        ``None`` is the daily job's shape: one statement over the whole
        population, parameterised by nothing. A sequence is batched by
        ``MAX_USER_BATCH`` the way ``list_current_snapshots`` already is.
        """
        result: dict[int, UserActivity] = {}
        if user_ids is None:
            with self._analytics_connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM user_activity ORDER BY user_id"
                ).fetchall()
            for row in rows:
                activity = _activity_from_row(row)
                result[activity.user_id] = activity
            return result
        ids = _ids(user_ids)
        if not ids:
            return {}
        for offset in range(0, len(ids), MAX_USER_BATCH):
            chunk = ids[offset : offset + MAX_USER_BATCH]
            clause = f"user_id IN ({', '.join('?' for _ in chunk)})"
            with self._analytics_connection() as conn:
                rows = conn.execute(
                    f"SELECT * FROM user_activity WHERE {clause} ORDER BY user_id",
                    chunk,
                ).fetchall()
            for row in rows:
                activity = _activity_from_row(row)
                result[activity.user_id] = activity
        return result

    def seed_activity_from_snapshots(self, *, now: datetime) -> int:
        """Copy ``activated_at`` / ``last_meaningful_activity_at`` from the legacy row.

        Design SS11: ``user_analytics_snapshots`` is the only table that knows
        when an existing user first activated, and it is dropped in PR B.
        Ingestion only maintains ``user_activity`` for events arriving after
        this deploy, so without this copy every pre-existing activation date
        would vanish on the night of the drop.

        Idempotent by construction -- the same MIN/MAX upsert
        ``record_activity`` uses -- so PR B can re-run it immediately before
        the drop (the legacy row keeps being maintained by the throttled
        repair between the two PRs). Returns the number of rows touched.
        """
        stamp = utc_iso(_utc(now, "now"))
        sql = """
            INSERT INTO user_activity (
                user_id, activated_at, last_meaningful_activity_at, updated_at
            )
            SELECT user_id, activated_at, last_meaningful_activity_at, ?
            FROM user_analytics_snapshots
            WHERE activated_at IS NOT NULL
               OR last_meaningful_activity_at IS NOT NULL
            ON CONFLICT(user_id) DO UPDATE SET
                activated_at = MIN(
                    COALESCE(user_activity.activated_at, excluded.activated_at),
                    COALESCE(excluded.activated_at, user_activity.activated_at)
                ),
                last_meaningful_activity_at = MAX(
                    COALESCE(
                        user_activity.last_meaningful_activity_at,
                        excluded.last_meaningful_activity_at
                    ),
                    COALESCE(
                        excluded.last_meaningful_activity_at,
                        user_activity.last_meaningful_activity_at
                    )
                ),
                updated_at = excluded.updated_at
        """
        with self._analytics_connection() as conn:
            cursor = conn.execute(sql, (stamp,))
            return max(0, int(cursor.rowcount))
```

The seed's `SELECT` deliberately carries a `WHERE` clause: SQLite requires one on an `INSERT ... SELECT ... ON CONFLICT` to disambiguate the upsert from a join, and the filter is also the right one — a legacy row with neither timestamp has nothing to carry across.

Add `"UserActivity",` to `__all__` (alphabetically, after `"ProjectionJob",`).

- [ ] **Step 4: Add the same four methods to the Postgres twin**

In `dashboard/backend/domain/analytics/value_repository_postgres.py`, extend the `from .value_repository import (...)` block with `UserActivity,` (after `ProjectionJob,`) and `_activity_from_row,` (after `_TERMINAL_RUN_STATUSES,`, keeping the block's existing order of private names). Then on `PostgresValueAnalyticsStore`, immediately after its `has_daily_before`, insert:

```python
    def record_activity(
        self,
        user_id: int,
        *,
        occurred_at: datetime,
        activating: bool,
        now: datetime,
    ) -> None:
        """See the SQLite twin. Postgres ``LEAST``/``GREATEST`` skip NULLs, so
        the COALESCE pair is redundant here but harmless; keeping both dialects
        spelled the same way is worth more than saving two lines. ``%s::text``
        on the nullable ``activated_at`` gives psycopg a type for the ``None``
        it would otherwise send as OID 0.
        """
        subject_id = positive_user_id(user_id)
        occurred = utc_iso(_utc(occurred_at, "occurred_at"))
        values = (
            subject_id,
            occurred if activating else None,
            occurred,
            utc_iso(_utc(now, "now")),
        )
        sql = """
            INSERT INTO user_activity (
                user_id, activated_at, last_meaningful_activity_at, updated_at
            ) VALUES (%s, %s::text, %s, %s)
            ON CONFLICT(user_id) DO UPDATE SET
                activated_at = LEAST(
                    COALESCE(user_activity.activated_at, EXCLUDED.activated_at),
                    COALESCE(EXCLUDED.activated_at, user_activity.activated_at)
                ),
                last_meaningful_activity_at = GREATEST(
                    COALESCE(
                        user_activity.last_meaningful_activity_at,
                        EXCLUDED.last_meaningful_activity_at
                    ),
                    EXCLUDED.last_meaningful_activity_at
                ),
                updated_at = EXCLUDED.updated_at
        """
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, values)

    def get_activity(self, user_id: int) -> UserActivity | None:
        """One user's stored activity row, or None if they have none yet."""
        subject_id = positive_user_id(user_id)
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM user_activity WHERE user_id = %s", (subject_id,)
                )
                row = cur.fetchone()
        return _activity_from_row(row) if row is not None else None

    def list_activity(
        self,
        user_ids: Sequence[int] | None = None,
    ) -> dict[int, UserActivity]:
        """See the SQLite twin."""
        result: dict[int, UserActivity] = {}
        if user_ids is None:
            with self._analytics_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM user_activity ORDER BY user_id")
                    rows = cur.fetchall()
            for row in rows:
                activity = _activity_from_row(row)
                result[activity.user_id] = activity
            return result
        ids = _ids(user_ids)
        if not ids:
            return {}
        for offset in range(0, len(ids), MAX_USER_BATCH):
            chunk = ids[offset : offset + MAX_USER_BATCH]
            with self._analytics_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT * FROM user_activity WHERE user_id = ANY(%s) "
                        "ORDER BY user_id",
                        (chunk,),
                    )
                    rows = cur.fetchall()
            for row in rows:
                activity = _activity_from_row(row)
                result[activity.user_id] = activity
        return result

    def seed_activity_from_snapshots(self, *, now: datetime) -> int:
        """See the SQLite twin."""
        stamp = utc_iso(_utc(now, "now"))
        sql = """
            INSERT INTO user_activity (
                user_id, activated_at, last_meaningful_activity_at, updated_at
            )
            SELECT user_id, activated_at, last_meaningful_activity_at, %s
            FROM user_analytics_snapshots
            WHERE activated_at IS NOT NULL
               OR last_meaningful_activity_at IS NOT NULL
            ON CONFLICT(user_id) DO UPDATE SET
                activated_at = LEAST(
                    COALESCE(user_activity.activated_at, EXCLUDED.activated_at),
                    COALESCE(EXCLUDED.activated_at, user_activity.activated_at)
                ),
                last_meaningful_activity_at = GREATEST(
                    COALESCE(
                        user_activity.last_meaningful_activity_at,
                        EXCLUDED.last_meaningful_activity_at
                    ),
                    COALESCE(
                        EXCLUDED.last_meaningful_activity_at,
                        user_activity.last_meaningful_activity_at
                    )
                ),
                updated_at = EXCLUDED.updated_at
        """
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (stamp,))
                return max(0, int(cur.rowcount))
```

- [ ] **Step 5: Add the flag and the write to the ingestion service**

In `dashboard/backend/domain/analytics/service.py`, change `AnalyticsService.__init__`:

```python
    def __init__(
        self,
        store,
        *,
        state_store=None,
        value_store=None,
        project_snapshots: bool = False,
    ):
        if not isinstance(project_snapshots, bool):
            raise TypeError("project_snapshots must be a boolean")
        if project_snapshots and (state_store is None or value_store is None):
            raise ValueError("snapshot stores are required when projection is enabled")
        self.store = store
        self.state_store = state_store
        self.value_store = value_store
        self.project_snapshots = project_snapshots
```

to:

```python
    def __init__(
        self,
        store,
        *,
        state_store=None,
        value_store=None,
        project_snapshots: bool = False,
        maintain_activity: bool = False,
    ):
        if not isinstance(project_snapshots, bool):
            raise TypeError("project_snapshots must be a boolean")
        if not isinstance(maintain_activity, bool):
            raise TypeError("maintain_activity must be a boolean")
        if project_snapshots and (state_store is None or value_store is None):
            raise ValueError("snapshot stores are required when projection is enabled")
        if maintain_activity and value_store is None:
            raise ValueError("a value store is required to maintain user activity")
        self.store = store
        self.state_store = state_store
        self.value_store = value_store
        self.project_snapshots = project_snapshots
        self.maintain_activity = maintain_activity
```

Immediately after `_try_recalculate_snapshots` (which ends with the `print("WARNING: analytics.value_projection_failed ...")` block), insert:

```python
    def _try_record_activity(
        self,
        *,
        user_id: int,
        event: AnalyticsEventRecord,
        now: datetime,
    ) -> None:
        """Advance the stored activity clock for one accepted event.

        Best-effort, like every other analytics write: a failure here must
        never change the outcome of the operation that emitted the event.
        This is the write that replaces the per-event history recompute
        (design SS6.5): one upsert of two timestamps, no read.
        """
        if not self.maintain_activity:
            return
        try:
            self.value_store.record_activity(
                user_id,
                occurred_at=event.occurred_at,
                activating=event.event_name == "backtest_completed",
                now=now,
            )
        except Exception as exc:
            print(
                "WARNING: analytics.activity_update_failed "
                f"event={event.event_name} category={type(exc).__name__[:80]}"
            )
```

In `record_server_event`, change:

```python
        result = self.store.append_event(event)
        from .lifecycle import is_lifecycle_activity

        projection_relevant = is_lifecycle_activity(event) or event_name in {
            "account_signed_up",
            "safe_error_recorded",
        }
        if result.created and projection_relevant:
```

to:

```python
        result = self.store.append_event(event)
        from .lifecycle import is_lifecycle_activity

        lifecycle_activity = is_lifecycle_activity(event)
        if result.created and lifecycle_activity:
            self._try_record_activity(user_id=subject_id, event=event, now=received)
        projection_relevant = lifecycle_activity or event_name in {
            "account_signed_up",
            "safe_error_recorded",
        }
        if result.created and projection_relevant:
```

`account_signed_up` and `safe_error_recorded` are deliberately outside the activity branch: signup time is `users.created_at`, which the lifecycle rules already read, and a safe error is not activity. The `projection_relevant` branch that follows is PR 0's disarmed legacy path and is left exactly as it is for PR B to delete.

Change `_build_analytics_service` from its post-PR-0/PR-T text:

```python
def _build_analytics_service() -> AnalyticsService:
    from .states import AnalyticsStateStore
    from .value_repository import build_value_analytics_store

    state_store = AnalyticsStateStore(analytics_store)
    return AnalyticsService(
        store=analytics_store,
        state_store=state_store,
        value_store=build_value_analytics_store(analytics_store),
        project_snapshots=False,
    )
```

to:

```python
def _build_analytics_service() -> AnalyticsService:
    from .states import AnalyticsStateStore
    from .value_repository import build_value_analytics_store

    state_store = AnalyticsStateStore(analytics_store)
    return AnalyticsService(
        store=analytics_store,
        state_store=state_store,
        value_store=build_value_analytics_store(analytics_store),
        # PR 0 switched the per-event snapshot recompute off; PR A replaces it
        # with the activity upsert. PR B deletes the projection flag entirely.
        project_snapshots=False,
        maintain_activity=True,
    )
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_user_activity.py dashboard/backend/tests/domain/analytics/test_service.py dashboard/backend/tests/domain/analytics/test_instrumentation.py dashboard/backend/tests/domain/analytics/test_states.py dashboard/backend/tests/test_store_twin_parity.py -v
TEST_POSTGRES_URL=postgresql://postgres:atl@localhost:55432/postgres \
  python -m pytest dashboard/backend/tests/domain/analytics/ -q
```

Expected: PASS on all. `test_states.py` and `test_instrumentation.py` are unchanged and must stay green — this task adds a flag beside `project_snapshots` rather than renaming it. Run the Postgres tier here specifically: `GREATEST` versus `MAX` is exactly the kind of difference SQLite alone cannot catch.

- [ ] **Step 7: Commit**

```bash
git add dashboard/backend/domain/analytics/value_repository.py \
        dashboard/backend/domain/analytics/value_repository_postgres.py \
        dashboard/backend/domain/analytics/service.py \
        dashboard/backend/tests/domain/analytics/test_user_activity.py \
        dashboard/backend/tests/domain/analytics/test_service.py
git status --short
git commit -m "feat: maintain user_activity from ingestion and seed it from the legacy snapshots"
```

---

### Task 4: Lifecycle inputs from stored facts (read-time calculator, not yet routed)

**Files:**
- Create: `dashboard/backend/domain/analytics/lifecycle_reads.py`
- Modify: `dashboard/backend/domain/analytics/value_repository.py` — add `RecentFactTotals`, `_RECENT_FACTS_SQL`, `_recent_totals_from_row` after `_activity_from_row`; add `sum_recent_facts` to `ValueAnalyticsStore` after `seed_activity_from_snapshots`; export `RecentFactTotals`
- Modify: `dashboard/backend/domain/analytics/value_repository_postgres.py` — `sum_recent_facts` on `PostgresValueAnalyticsStore`; import the three new names
- Test: `dashboard/backend/tests/domain/analytics/test_lifecycle_reads.py` (create)

**Interfaces:**
- Consumes: `UserActivity` (Task 3); `user_daily_facts` (Task 1); `LifecycleInputs` and `calculate_lifecycle` (`lifecycle.py`, unchanged).
- Produces:
  - `RecentFactTotals(active_days, successful_backtests, runs_requested, runs_completed, runs_failed, runs_cancelled, operator_cost_micro, own_spend_micro, days_present, last_active_date: date | None)` in `value_repository.py`
  - `sum_recent_facts(user_ids: Sequence[int] | None, *, start: date, end: date) -> dict[int, RecentFactTotals]` on both value twins (`None` = whole population, the daily job's shape)
  - `build_lifecycle_inputs(user_id: int, *, created_at: datetime, activity: UserActivity | None, totals: RecentFactTotals | None, as_of: datetime, day_activity_at: datetime | None = None) -> LifecycleInputs` in `lifecycle_reads.py`

  Task 9 calls all three for the whole population. **No route calls `build_lifecycle_inputs` in this PR** (§13 row A: "read-time lifecycle calculator (not yet wired to routes)"); PR B wires `/users/{id}`, and PR B — not this task — adds `resolve_group_badge` (D11).

Carried from the superseded 2026-09-12 plan's Task B5 (`git show c3bbf2ed:...`, lines 2098-2538) minus `resolve_group_badge` (moved to PR B by D11/§13) and minus every `cohort` reference (D9). `calculate_lifecycle` already takes `LifecycleInputs`, a struct of stored timestamps and two counts — it never read an event list. The event reading was in `states.py`, which built those inputs by scanning. This task replaces the scan with two indexed reads and leaves the rules untouched.

`as_of` is required, not defaulted: the live read path (PR B) passes `now` and the daily job passes the end of the day it is computing, and a default would silently give the job the live answer. `day_activity_at` *is* defaulted, because it only exists for the daily job — it is the user's last activity inside the day being computed, which is the one thing no window over past fact rows can supply.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/domain/analytics/test_lifecycle_reads.py`:

```python
"""Lifecycle from stored facts, with the rules unchanged."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from dashboard.backend.domain.analytics.lifecycle import calculate_lifecycle
from dashboard.backend.domain.analytics.lifecycle_reads import build_lifecycle_inputs
from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.value_repository import (
    RecentFactTotals,
    UserActivity,
    build_value_analytics_store,
)
from dashboard.backend.users import UserStore


NOW = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)


def _totals(**overrides):
    base = dict(
        active_days=0,
        successful_backtests=0,
        runs_requested=0,
        runs_completed=0,
        runs_failed=0,
        runs_cancelled=0,
        operator_cost_micro=0,
        own_spend_micro=0,
        days_present=30,
    )
    base.update(overrides)
    return RecentFactTotals(**base)


def test_core_needs_three_active_days_and_three_successes():
    activity = UserActivity(
        user_id=1,
        activated_at=NOW - timedelta(days=20),
        last_meaningful_activity_at=NOW - timedelta(days=1),
        updated_at=NOW,
    )
    inputs = build_lifecycle_inputs(
        1,
        created_at=NOW - timedelta(days=60),
        activity=activity,
        totals=_totals(active_days=3, successful_backtests=3),
        as_of=NOW,
    )

    assert calculate_lifecycle(inputs, NOW).segment == "core"

    below = build_lifecycle_inputs(
        1,
        created_at=NOW - timedelta(days=60),
        activity=activity,
        totals=_totals(active_days=3, successful_backtests=2),
        as_of=NOW,
    )
    assert calculate_lifecycle(below, NOW).segment == "growing"


def test_a_user_with_no_activity_row_falls_back_to_signup():
    inputs = build_lifecycle_inputs(
        1,
        created_at=NOW - timedelta(days=2),
        activity=None,
        totals=None,
        as_of=NOW,
    )
    result = calculate_lifecycle(inputs, NOW)

    assert result.segment == "new"
    assert result.active_days_30d == 0


def test_evidence_newer_than_as_of_is_clamped_not_raised():
    """Serving a past day from a table that only knows the present.

    `user_activity` is one row per user, overwritten in place. Asked for
    the end of yesterday while holding a timestamp from this morning,
    `calculate_lifecycle` raises rather than guessing -- so the clamp has to
    happen here, and the fact table is where the answer for that past day
    actually lives.
    """
    end_of_yesterday = datetime(2026, 9, 11, 23, 59, 59, tzinfo=timezone.utc)
    activity = UserActivity(
        user_id=1,
        activated_at=NOW,  # activated today
        last_meaningful_activity_at=NOW,  # active today
        updated_at=NOW,
    )

    inputs = build_lifecycle_inputs(
        1,
        created_at=NOW - timedelta(days=60),
        activity=activity,
        totals=_totals(active_days=1, last_active_date=date(2026, 9, 4)),
        as_of=end_of_yesterday,
    )

    # Not activated as of yesterday, and last active on the 4th.
    assert inputs.first_successful_backtest_at is None
    assert inputs.last_meaningful_activity_at == datetime(
        2026, 9, 4, tzinfo=timezone.utc
    )
    result = calculate_lifecycle(inputs, end_of_yesterday)
    assert result.segment == "at_risk"


def test_activity_inside_the_computed_day_beats_the_window_fallback():
    """A user active only on D is not dormant, and the window cannot say so.

    The daily job narrows the fact window to D-29..D-1, so a user whose first
    activity in a month lands on D has no row in it. Falling straight through
    to `last_active_date` reads None and classifies them Dormant on the very
    day they came back.
    """
    end_of_yesterday = datetime(2026, 9, 11, 23, 59, 59, tzinfo=timezone.utc)
    activity = UserActivity(
        user_id=1,
        activated_at=NOW - timedelta(days=200),
        last_meaningful_activity_at=NOW,  # newer than as_of, so clamped
        updated_at=NOW,
    )

    inputs = build_lifecycle_inputs(
        1,
        created_at=NOW - timedelta(days=300),
        activity=activity,
        totals=_totals(active_days=1, last_active_date=None),
        as_of=end_of_yesterday,
        day_activity_at=datetime(2026, 9, 11, 18, 0, tzinfo=timezone.utc),
    )

    assert inputs.last_meaningful_activity_at == datetime(
        2026, 9, 11, 18, 0, tzinfo=timezone.utc
    )
    assert calculate_lifecycle(inputs, end_of_yesterday).segment == "growing"


def test_the_live_path_clamps_nothing():
    """as_of=now is the identity case, and must stay cost-free."""
    activity = UserActivity(
        user_id=1,
        activated_at=NOW - timedelta(days=3),
        last_meaningful_activity_at=NOW,
        updated_at=NOW,
    )

    inputs = build_lifecycle_inputs(
        1,
        created_at=NOW - timedelta(days=60),
        activity=activity,
        totals=_totals(active_days=5, successful_backtests=5),
        as_of=NOW,
    )

    assert inputs.first_successful_backtest_at == activity.activated_at
    assert inputs.last_meaningful_activity_at == activity.last_meaningful_activity_at


def test_inactivity_crosses_at_risk_without_any_new_evidence():
    """Time alone moves the segment. That is why nothing needs recomputing."""
    activity = UserActivity(
        user_id=1,
        activated_at=NOW - timedelta(days=40),
        last_meaningful_activity_at=NOW - timedelta(days=7),
        updated_at=NOW,
    )
    inputs = build_lifecycle_inputs(
        1,
        created_at=NOW - timedelta(days=90),
        activity=activity,
        totals=_totals(),
        as_of=NOW,
    )

    assert calculate_lifecycle(inputs, NOW).segment == "growing"
    assert calculate_lifecycle(inputs, NOW + timedelta(days=1)).segment == "at_risk"
    assert calculate_lifecycle(inputs, NOW + timedelta(days=23)).segment == "dormant"


def _fact_store(tmp_path):
    path = tmp_path / "facts.db"
    users = UserStore(db_path=path)
    first = int(users.create_user("a@example.test", "A", "SecurePass1!")["id"])
    second = int(users.create_user("b@example.test", "B", "SecurePass1!")["id"])
    analytics = AnalyticsStore(db_path=path)
    store = build_value_analytics_store(
        analytics,
        credits_base=object(),
        provider_base=object(),
        agent_base=object(),
        run_base=object(),
    )
    with analytics._get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO user_daily_facts (
                snapshot_date, user_id, lifecycle_segment, lifecycle_reason_code,
                operational_state, tier, user_group, active, runs_requested,
                runs_completed, runs_failed, runs_cancelled, operator_cost_micro,
                own_spend_micro, data_quality, calculated_at
            ) VALUES (?, ?, 'growing', 'growing_activated_below_core_threshold',
                      'healthy', 'unpaid', 'unknown', ?, ?, ?, ?, ?, ?, ?,
                      'complete', '2026-09-12T00:05:00+00:00')
            """,
            [
                ("2026-09-01", first, 1, 2, 1, 1, 0, 1_000_000, 0),
                ("2026-09-05", first, 1, 1, 1, 0, 0, 0, 250_000),
                ("2026-09-09", first, 0, 0, 0, 0, 0, 0, 0),
                ("2026-09-11", first, 1, 3, 2, 0, 1, 500_000, 0),
                ("2026-09-11", second, 1, 1, 1, 0, 0, 0, 0),
                ("2026-08-01", first, 1, 9, 9, 0, 0, 0, 0),  # outside the window
            ],
        )
    return store, first, second


def test_sum_recent_facts_groups_the_window_in_one_statement(tmp_path):
    store, first, second = _fact_store(tmp_path)

    totals = store.sum_recent_facts(
        [first, second], start=date(2026, 8, 13), end=date(2026, 9, 11)
    )

    assert totals[first] == RecentFactTotals(
        active_days=3,
        successful_backtests=4,
        runs_requested=6,
        runs_completed=4,
        runs_failed=1,
        runs_cancelled=1,
        operator_cost_micro=1_500_000,
        own_spend_micro=250_000,
        days_present=4,
        last_active_date=date(2026, 9, 11),
    )
    assert totals[second].active_days == 1
    assert totals[second].days_present == 1
    # None means the whole population -- the daily job's shape.
    assert store.sum_recent_facts(
        None, start=date(2026, 8, 13), end=date(2026, 9, 11)
    ) == totals
    assert store.sum_recent_facts([], start=date(2026, 8, 13), end=date(2026, 9, 11)) == {}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_lifecycle_reads.py -v
```

Expected: FAIL at import — `ImportError: cannot import name 'RecentFactTotals' from 'dashboard.backend.domain.analytics.value_repository'` (and `lifecycle_reads` does not exist).

- [ ] **Step 3: Add the totals model and the batched sum to both twins**

In `dashboard/backend/domain/analytics/value_repository.py`, immediately after `_activity_from_row`, insert:

```python
class RecentFactTotals(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    active_days: int = Field(default=0, ge=0)
    successful_backtests: int = Field(default=0, ge=0)
    runs_requested: int = Field(default=0, ge=0)
    runs_completed: int = Field(default=0, ge=0)
    runs_failed: int = Field(default=0, ge=0)
    runs_cancelled: int = Field(default=0, ge=0)
    operator_cost_micro: int = Field(default=0, ge=0)
    own_spend_micro: int = Field(default=0, ge=0)
    # How many of the requested dates actually have a row. Fewer than
    # requested means the window is incomplete, which the UI labels rather
    # than rendering as zero.
    days_present: int = Field(default=0, ge=0)
    # The latest date inside the window on which this user was active, or
    # None. The daily job needs it to answer "what did this user's activity
    # look like as of the end of day D" -- `user_activity` holds one row
    # overwritten in place and cannot answer a question about the past.
    last_active_date: date | None = None


_RECENT_FACTS_SQL = """
    SELECT user_id,
           SUM(CASE WHEN active THEN 1 ELSE 0 END) AS active_days,
           SUM(runs_completed) AS successful_backtests,
           SUM(runs_requested) AS runs_requested,
           SUM(runs_failed) AS runs_failed,
           SUM(runs_cancelled) AS runs_cancelled,
           SUM(operator_cost_micro) AS operator_cost_micro,
           SUM(own_spend_micro) AS own_spend_micro,
           COUNT(*) AS days_present,
           MAX(CASE WHEN active THEN snapshot_date END) AS last_active_date
    FROM user_daily_facts
    WHERE snapshot_date >= {p} AND snapshot_date <= {p}{user_clause}
    GROUP BY user_id
"""


def _recent_totals_from_row(row: Any) -> RecentFactTotals:
    """Shared by both twins."""
    last_active = _row_value(row, "last_active_date")
    completed = int(_row_value(row, "successful_backtests", 0) or 0)
    return RecentFactTotals(
        active_days=int(_row_value(row, "active_days", 0) or 0),
        successful_backtests=completed,
        runs_requested=int(_row_value(row, "runs_requested", 0) or 0),
        runs_completed=completed,
        runs_failed=int(_row_value(row, "runs_failed", 0) or 0),
        runs_cancelled=int(_row_value(row, "runs_cancelled", 0) or 0),
        operator_cost_micro=int(_row_value(row, "operator_cost_micro", 0) or 0),
        own_spend_micro=int(_row_value(row, "own_spend_micro", 0) or 0),
        days_present=int(_row_value(row, "days_present", 0) or 0),
        last_active_date=(
            date.fromisoformat(str(last_active)) if last_active else None
        ),
    )
```

`SUM(CASE WHEN active THEN 1 ELSE 0 END)` rather than `SUM(active)`: `active` is an integer on SQLite and a boolean on Postgres, and Postgres rejects `SUM(boolean)` outright; the `CASE` works on both. `successful_backtests` is `runs_completed` by definition — a completed backtest is a successful one, and activation is the first `backtest_completed` event — which is why both model fields are filled from the same column. `snapshot_date` is ISO-8601 text, which orders lexicographically, so `MAX` over it is `MAX` over the dates.

On `ValueAnalyticsStore`, immediately after `seed_activity_from_snapshots`, insert:

```python
    def sum_recent_facts(
        self,
        user_ids: Sequence[int] | None,
        *,
        start: date,
        end: date,
    ) -> dict[int, RecentFactTotals]:
        """Trailing-window totals for many users in one query.

        ``start`` and ``end`` are inclusive UTC dates. ``None`` for
        ``user_ids`` means the whole population -- the daily job's shape,
        parameterised by two dates and nothing else. One statement for the
        whole batch: a per-user loop here is the shape that caused the
        outage, and the read-budget test fails on it.
        """
        if end < start:
            raise ValueError("end must not precede start")
        params: list[Any] = [start.isoformat(), end.isoformat()]
        user_clause = ""
        if user_ids is not None:
            ids = _ids(user_ids)
            if not ids:
                return {}
            user_clause = f" AND user_id IN ({', '.join('?' for _ in ids)})"
            params.extend(ids)
        sql = _RECENT_FACTS_SQL.format(p="?", user_clause=user_clause)
        with self._analytics_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return {
            int(_row_value(row, "user_id")): _recent_totals_from_row(row)
            for row in rows
        }
```

In `dashboard/backend/domain/analytics/value_repository_postgres.py`, extend the import block with `RecentFactTotals,` (after `ProjectionJob,`), `_RECENT_FACTS_SQL,` and `_recent_totals_from_row,` (with the other private names), and on `PostgresValueAnalyticsStore`, immediately after `seed_activity_from_snapshots`, insert:

```python
    def sum_recent_facts(
        self,
        user_ids: Sequence[int] | None,
        *,
        start: date,
        end: date,
    ) -> dict[int, RecentFactTotals]:
        """See the SQLite twin."""
        if end < start:
            raise ValueError("end must not precede start")
        params: list[Any] = [start.isoformat(), end.isoformat()]
        user_clause = ""
        if user_ids is not None:
            ids = _ids(user_ids)
            if not ids:
                return {}
            user_clause = " AND user_id = ANY(%s)"
            params.append(ids)
        sql = _RECENT_FACTS_SQL.format(p="%s", user_clause=user_clause)
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return {
            int(_row_value(row, "user_id")): _recent_totals_from_row(row)
            for row in rows
        }
```

Add `"RecentFactTotals",` to `value_repository.py`'s `__all__` (after `"ProjectionJob",`).

- [ ] **Step 4: Add the input builder**

Create `dashboard/backend/domain/analytics/lifecycle_reads.py`:

```python
"""Build lifecycle inputs from stored facts instead of an event scan.

``calculate_lifecycle`` always took a struct of timestamps and two counts.
What changed is where the struct comes from: two indexed reads of
``user_activity`` and ``user_daily_facts`` rather than a 180-day scan of
``analytics_events`` per user (design SS6.6).

Nothing in ``api/`` calls this module yet. PR B wires it into the one-user
profile route; the daily job (``daily_facts.py``) is its only caller in PR A.
"""

from __future__ import annotations

from datetime import datetime, time, timezone

from .lifecycle import LifecycleInputs
from .repository_common import positive_user_id
from .value_repository import RecentFactTotals, UserActivity


def build_lifecycle_inputs(
    user_id: int,
    *,
    created_at: datetime,
    activity: UserActivity | None,
    totals: RecentFactTotals | None,
    as_of: datetime,
    day_activity_at: datetime | None = None,
) -> LifecycleInputs:
    """Assemble one user's lifecycle inputs from stored rows, as of ``as_of``.

    A user with no ``user_activity`` row has never done anything meaningful;
    ``calculate_lifecycle`` anchors on ``created_at`` in that case, so a
    brand-new account reads as New rather than as instantly Dormant.

    ``active_days_30d`` is capped at 30 because the model bounds it there and
    a migrated or double-counted window must not raise a validation error on
    a read path.

    **Every timestamp is clamped to ``as_of``, and that is not defensive
    programming -- it is the only thing that lets this function serve a past
    day at all.** ``calculate_lifecycle`` raises
    ``ValueError("lifecycle evidence cannot occur after as_of")`` when its
    anchor is newer than ``as_of`` (lifecycle.py:242-243), and
    ``user_activity`` is a single row per user overwritten in place: it says
    what is true *now*, never what was true at the end of some earlier day.
    The live read path passes ``as_of=now`` and nothing clamps; the daily job
    passes the end of the day it is computing, and without this every user
    who acted after midnight would abort the whole population's fact write.

    ``activated_at`` newer than ``as_of`` becomes None rather than being
    clamped to ``as_of``: "activated after this day" means "not activated as
    of this day", and pretending they activated at midnight would put them in
    the wrong retention cohort.

    ``last_meaningful_activity_at`` is only replaced when the stored value is
    newer than ``as_of``, because when it is not, it is authoritative and more
    precise than any fallback. The replacement walks newest-first:

    1. ``day_activity_at`` -- the user's own last activity **inside the day
       being computed**, from the daily job's one-day event aggregate. This
       has to come first and it is the case a date-window fallback alone
       silently gets wrong: a user whose only recent activity is on ``D``
       itself has no row in the ``D-29..D-1`` window at all, so skipping
       straight to ``totals`` would read them as inactive for thirty days and
       classify an active user as Dormant.
    2. ``totals.last_active_date`` -- the fact table *does* keep history,
       which is the whole reason it exists. Midnight-start of that date is the
       conservative choice; ``calculate_lifecycle`` counts ``inactive_days`` in
       whole UTC dates, so the time of day is never read.
    3. None -- no activity anywhere in the window, so ``created_at`` anchors
       the calculation, which is what a dormant classification needs.

    A user whose ``created_at`` is after ``as_of`` is **not** this function's
    problem to clamp -- ``calculate_lifecycle`` would raise on
    ``account_age_days < 0``, and rightly so, because an account that did not
    exist on day D has no day D. The daily job excludes those users from the
    eligible set instead; the live read path can never hit it.
    """
    subject_id = positive_user_id(user_id)
    counts = totals or RecentFactTotals()
    boundary = as_of

    activated_at = activity.activated_at if activity is not None else None
    if activated_at is not None and activated_at > boundary:
        activated_at = None

    last_activity = (
        activity.last_meaningful_activity_at if activity is not None else None
    )
    if last_activity is not None and last_activity > boundary:
        if day_activity_at is not None and day_activity_at <= boundary:
            last_activity = day_activity_at
        elif counts.last_active_date is not None:
            last_activity = datetime.combine(
                counts.last_active_date, time.min, tzinfo=timezone.utc
            )
        else:
            last_activity = None

    return LifecycleInputs(
        user_id=subject_id,
        created_at=created_at,
        first_successful_backtest_at=activated_at,
        last_meaningful_activity_at=last_activity,
        active_days_30d=min(30, counts.active_days),
        successful_backtests_30d=counts.successful_backtests,
    )


__all__ = ["build_lifecycle_inputs"]
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_lifecycle_reads.py dashboard/backend/tests/domain/analytics/test_lifecycle.py dashboard/backend/tests/test_store_twin_parity.py -v
TEST_POSTGRES_URL=postgresql://postgres:atl@localhost:55432/postgres \
  python -m pytest dashboard/backend/tests/domain/analytics/ -q
```

Expected: PASS. `test_lifecycle.py` must stay green untouched — the rules did not change, only their inputs' provenance.

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/domain/analytics/lifecycle_reads.py \
        dashboard/backend/domain/analytics/value_repository.py \
        dashboard/backend/domain/analytics/value_repository_postgres.py \
        dashboard/backend/tests/domain/analytics/test_lifecycle_reads.py
git status --short
git commit -m "feat: compute lifecycle inputs from stored facts"
```

---

### Task 5: Ledger and run aggregates live on the stores that own the tables

**Files:**
- Modify: `dashboard/backend/domain/credits/repository.py` — imports (lines 8, 10); add `aggregate_commercial_ledger`, `list_credit_activity_timestamps`, `aggregate_ledger_for_day`, `list_account_billing_states` to `CreditsStore` after `list_user_ids` (lines 1012-1017)
- Modify: `dashboard/backend/domain/credits/repository_postgres.py` — imports (lines 6-7); the same four methods on `PostgresCreditsStore` after `list_user_ids` (lines 865-872)
- Modify: `dashboard/backend/database.py` — imports (lines 9-13); add `aggregate_operator_cost_for_day` to `BacktestDatabase` after `get_runs_by_mode` (lines 907-921)
- Modify: `dashboard/backend/database_postgres.py` — imports (lines 44-45); add `aggregate_operator_cost_for_day` to `PostgresBacktestDatabase` after `get_runs_by_mode` (lines 1007-1020)
- Modify: `dashboard/backend/domain/analytics/value_repository.py` — rewrite `list_commercial_values` and `list_credit_activity` (post-PR-T lines 946-1127) to call the store methods; delete the now-unused module functions `_fetchall` and `_user_clause` (post-PR-T lines 553-572)
- Modify: `dashboard/backend/domain/analytics/value_repository_postgres.py` — the same two rewrites (verbatim copies); drop `_fetchall` and `_user_clause` from the import block
- Modify: `dashboard/backend/tests/test_store_twin_parity.py` — the `value_repository.py` entry of `_DIALECT_BRANCH_ALLOWLIST` (added by PR T Task 4) gets a new reason
- Modify: `dashboard/backend/tests/domain/analytics/test_value_repository.py` — `SyntheticCreditsStore` (lines 28-93) gains the two batched readers
- Test: `dashboard/backend/tests/test_credits_ledger_aggregates.py` (create)
- Test: `dashboard/backend/tests/test_run_history_aggregates.py` (create)

**Interfaces:**
- Consumes: `agent_runs.owner_user_id` (Task 2).
- Produces, on `CreditsStore` **and** `PostgresCreditsStore`:
  - `aggregate_commercial_ledger(user_ids: Sequence[int], *, start: datetime, end: datetime) -> dict[int, dict[str, int]]` — keys `lifetime_purchased_micro`, `lifetime_refunded_micro`, `purchased_micro`, `refunded_micro`, `grant_activity_micro`, `consumed_micro`; three statements, the exact SQL `list_commercial_values` runs today
  - `list_credit_activity_timestamps(user_ids: Sequence[int], *, start: datetime, end: datetime) -> dict[int, list[str]]` — `created_at` text of purchases and consumption inside the window; two statements, the exact SQL `list_credit_activity` runs today
  - `aggregate_ledger_for_day(day: date) -> dict[int, dict[str, Any]]` — keys `own_spend_micro`, `lifetime_net_purchased_micro`, `last_activity_at` (ISO text or None); three statements, none parameterised by a user id (design §6.9 step 4, §12 item 1)
  - `list_account_billing_states(user_ids: Sequence[int] | None = None) -> dict[int, dict[str, Any]]` — the batched twin of `get_account_billing_state`; two statements (design §6.9 step 5)
- Produces, on `BacktestDatabase` **and** `PostgresBacktestDatabase`: `aggregate_operator_cost_for_day(day: date) -> Dict[int, int]` — micro-USD of `est_cost_usd` per `owner_user_id` for runs whose `updated_at` falls on `day`; one statement (design §6.9 step 3, §12 item 1).
- Changes on both value twins: `list_commercial_values` and `list_credit_activity` keep their signatures and return values byte-for-byte and stop opening `self.credits_base._get_connection()`.

  Task 9 calls `aggregate_ledger_for_day` and `aggregate_operator_cost_for_day` once per day; Task 7 calls `list_account_billing_states`.

This is design §12 item 1 and D23. The superseded 2026-09-12 plan's Task B8 step 3 put `aggregate_operator_cost_for_day` and `aggregate_ledger_for_day` on `ValueAnalyticsStore` with hand-written SQL against `agent_runs`, `credit_ledger_entries` and `credit_llm_usage_entries` — which would have taken the cross-domain-SQL idiom from two readers to four. The correction moves both onto the owning stores, and — because "fixing two of three regrows the third" (D23) — also moves the two readers `value_repository.py` already has. After this task and Task 6, `rg "_get_connection" dashboard/backend/domain/analytics` returns only the analytics stores' own connections (`self._get_connection`, `self.analytics_base._get_connection`, `self.base_store._get_connection`), which Task 14's rule-7 test then pins.

Every SQL string below is transcribed from the reader it replaces, so the response shapes of `/commercial`, `/lifecycle` and `/retention` (which reach these two readers through `value_queries.py:536`, `:883` and `:1365`) do not move.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/test_credits_ledger_aggregates.py`:

```python
"""The credits domain answers the analytics domain's ledger questions itself.

Design SS6.14: a domain reads another domain's *service*, never its tables.
Until PR A, ``value_repository.py`` opened ``credits_store._get_connection()``
and ran its own SQL against the ledger; these four methods are that SQL, moved
onto the store that owns the tables.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone

from dashboard.backend.domain.credits.repository import CreditsStore
from dashboard.backend.users import UserStore


NOW = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)
DAY = date(2026, 9, 11)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _ledger_row(user_id, entry_type, amount_micro, created_at, *, key):
    """One row satisfying credit_ledger_entries' CHECK for its entry_type."""
    base = {
        "user_id": user_id,
        "entry_type": entry_type,
        "amount_micro": amount_micro,
        "operation_key": f"op:{key}",
        "operation_id": f"operation-{key}",
        "idempotency_key": f"idem:{key}",
        "source": "test",
        "reason": "ledger aggregate test",
        "created_at": _iso(created_at),
        "payment_order_id": None,
        "refund_request_id": None,
        "stripe_event_id": None,
        "request_digest": None,
        "actor_user_id": None,
        "reference_type": None,
        "reference_id": None,
    }
    if entry_type == "purchase":
        base.update(
            bucket="purchased", payment_order_id=f"order-{key}",
            stripe_event_id=f"evt-{key}",
        )
    elif entry_type == "refund":
        base.update(
            bucket="purchased", payment_order_id=f"order-{key}",
            refund_request_id=f"refund-{key}", stripe_event_id=f"evt-{key}",
        )
    else:  # admin_grant_assign / admin_grant_reclaim
        base.update(
            bucket="grant", request_digest="digest", actor_user_id=1,
            reference_type="grant_pool", reference_id="default",
        )
    return base


def _insert(path, table, rows):
    # A bare connection: FKs are off by default in sqlite3, so the payment
    # order / refund / stripe event FKs need no parent rows here. The CHECK
    # constraints still apply, which is why _ledger_row fills every column the
    # constraint for its entry_type inspects.
    with sqlite3.connect(path) as conn:
        for row in rows:
            columns = ", ".join(row)
            marks = ", ".join("?" for _ in row)
            conn.execute(
                f"INSERT INTO {table} ({columns}) VALUES ({marks})",
                tuple(row.values()),
            )


def _usage_row(user_id, amount_micro, created_at, *, key, bucket="grant"):
    return {
        "user_id": user_id,
        "reservation_id": f"res-{key}",
        "run_id": f"run-{key}",
        "call_index": 0,
        "bucket": bucket,
        "amount_micro": amount_micro,
        "operation_key": f"settle:{key}",
        "evidence_json": "{}",
        "created_at": _iso(created_at),
    }


def _store(tmp_path):
    path = tmp_path / "credits.db"
    users = UserStore(db_path=path)
    alice = int(users.create_user("alice@example.test", "Alice", "SecurePass1!")["id"])
    bob = int(users.create_user("bob@example.test", "Bob", "SecurePass1!")["id"])
    store = CreditsStore(path)
    day_start = datetime.combine(DAY, datetime.min.time(), tzinfo=timezone.utc)
    _insert(
        path,
        "credit_ledger_entries",
        [
            _ledger_row(alice, "purchase", 10_000_000, NOW - timedelta(days=40), key="a1"),
            _ledger_row(alice, "purchase", 3_000_000, day_start + timedelta(hours=1), key="a2"),
            _ledger_row(alice, "refund", -6_000_000, day_start + timedelta(hours=2), key="a3"),
            _ledger_row(alice, "admin_grant_assign", 1_500_000, day_start + timedelta(hours=3), key="a4"),
            _ledger_row(alice, "admin_grant_reclaim", -500_000, day_start + timedelta(hours=4), key="a5"),
            _ledger_row(bob, "purchase", 2_000_000, NOW - timedelta(days=3), key="b1"),
        ],
    )
    _insert(
        path,
        "credit_llm_usage_entries",
        [
            _usage_row(alice, -400_000, day_start + timedelta(hours=5), key="a6"),
            _usage_row(alice, -100_000, day_start + timedelta(hours=6), key="a7", bucket="purchased"),
            _usage_row(alice, -50_000, day_start + timedelta(days=1, hours=1), key="a8"),
            _usage_row(bob, -25_000, day_start + timedelta(hours=9), key="b2"),
        ],
    )
    return store, alice, bob, day_start


def test_aggregate_commercial_ledger_matches_the_reader_it_replaces(tmp_path):
    store, alice, bob, day_start = _store(tmp_path)

    totals = store.aggregate_commercial_ledger(
        [alice, bob], start=day_start, end=day_start + timedelta(days=1)
    )

    assert totals[alice] == {
        "lifetime_purchased_micro": 13_000_000,
        "lifetime_refunded_micro": 6_000_000,
        "purchased_micro": 3_000_000,
        "refunded_micro": 6_000_000,
        "grant_activity_micro": 2_000_000,
        "consumed_micro": 500_000,
    }
    # Bob bought outside the window and consumed inside it.
    assert totals[bob]["lifetime_purchased_micro"] == 2_000_000
    assert totals[bob]["purchased_micro"] == 0
    assert totals[bob]["consumed_micro"] == 25_000
    assert store.aggregate_commercial_ledger([], start=day_start, end=NOW) == {}


def test_list_credit_activity_timestamps_returns_purchases_and_consumption_only(tmp_path):
    store, alice, _bob, day_start = _store(tmp_path)

    stamps = store.list_credit_activity_timestamps(
        [alice], start=day_start, end=day_start + timedelta(days=1)
    )

    assert sorted(stamps[alice]) == [
        _iso(day_start + timedelta(hours=1)),  # the purchase
        _iso(day_start + timedelta(hours=5)),  # consumption
        _iso(day_start + timedelta(hours=6)),  # consumption
    ]


def test_aggregate_ledger_for_day_takes_no_user_id(tmp_path):
    store, alice, bob, day_start = _store(tmp_path)

    totals = store.aggregate_ledger_for_day(DAY)

    assert totals[alice]["own_spend_micro"] == 500_000
    assert totals[alice]["lifetime_net_purchased_micro"] == 7_000_000
    assert totals[alice]["last_activity_at"] == _iso(day_start + timedelta(hours=6))
    assert totals[bob]["own_spend_micro"] == 25_000
    assert totals[bob]["lifetime_net_purchased_micro"] == 2_000_000
    assert totals[bob]["last_activity_at"] == _iso(day_start + timedelta(hours=9))
    # The day after has only Alice's 50_000 consumption and no purchases.
    next_day = store.aggregate_ledger_for_day(DAY + timedelta(days=1))
    assert next_day[alice]["own_spend_micro"] == 50_000
    assert next_day[alice]["lifetime_net_purchased_micro"] == 7_000_000


def test_list_account_billing_states_agrees_with_the_single_user_reader(tmp_path):
    store, alice, bob, _day_start = _store(tmp_path)
    store.restrict_account(alice, reason="llm_overage")
    store.ensure_account(bob)

    batched = store.list_account_billing_states([alice, bob])
    everyone = store.list_account_billing_states()

    for user_id in (alice, bob):
        assert batched[user_id] == store.get_account_billing_state(user_id)
    assert everyone == batched
    assert batched[alice]["account_status"] == "restricted"
    assert batched[alice]["restriction_reason"] == "llm_overage"
    assert store.list_account_billing_states([]) == {}
```

Create `dashboard/backend/tests/test_run_history_aggregates.py`:

```python
"""Operator-funded model cost per owner comes from the run-history store."""

from __future__ import annotations

from datetime import date

from dashboard.backend.database import BacktestDatabase


def _seed(db, run_id, *, owner, cost, day):
    db.insert_run(
        run_id=run_id,
        session_id="session-1",
        agent_name="Agent",
        mode="backtest",
        start_date="2026-09-01",
        end_date="2026-09-02",
        initial_equity=100000.0,
        est_cost_usd=cost,
        owner_user_id=owner,
    )
    conn = db._get_connection()
    try:
        # created_at/updated_at are CURRENT_TIMESTAMP text ("YYYY-MM-DD HH:MM:SS");
        # pin the row onto the day under test.
        conn.execute(
            "UPDATE agent_runs SET updated_at = ? WHERE run_id = ?",
            (f"{day} 15:30:00", run_id),
        )
        conn.commit()
    finally:
        conn.close()


def test_operator_cost_is_grouped_by_owner_for_one_day(tmp_path):
    db = BacktestDatabase(tmp_path / "runs.db")
    _seed(db, "a", owner=7, cost=1.25, day="2026-09-11")
    _seed(db, "b", owner=7, cost=0.5, day="2026-09-11")
    _seed(db, "c", owner=9, cost=0.1, day="2026-09-11")
    _seed(db, "d", owner=7, cost=3.0, day="2026-09-10")  # another day
    _seed(db, "e", owner=None, cost=2.0, day="2026-09-11")  # unattributed

    totals = db.aggregate_operator_cost_for_day(date(2026, 9, 11))

    assert totals == {7: 1_750_000, 9: 100_000}
    assert db.aggregate_operator_cost_for_day(date(2026, 9, 12)) == {}


def test_a_negative_stored_cost_cannot_violate_the_fact_check(tmp_path):
    db = BacktestDatabase(tmp_path / "runs.db")
    _seed(db, "neg", owner=3, cost=-0.75, day="2026-09-11")

    assert db.aggregate_operator_cost_for_day(date(2026, 9, 11)) == {3: 0}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest dashboard/backend/tests/test_credits_ledger_aggregates.py dashboard/backend/tests/test_run_history_aggregates.py -v
```

Expected: FAIL with `AttributeError: 'CreditsStore' object has no attribute 'aggregate_commercial_ledger'` (and the three siblings) and `AttributeError: 'BacktestDatabase' object has no attribute 'aggregate_operator_cost_for_day'`.

- [ ] **Step 3: Add the four ledger methods to `CreditsStore`**

In `dashboard/backend/domain/credits/repository.py`, change the imports:

```python
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
```

to:

```python
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
```

Immediately before `class CreditsStore:` add the module helpers:

```python
def _utc_text(value: datetime, name: str) -> str:
    """ISO-8601 UTC text, the format every ``created_at`` in this ledger uses."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return value.astimezone(timezone.utc).isoformat()


def _day_bounds(day: date) -> tuple[str, str]:
    start = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
    return _utc_text(start, "day"), _utc_text(start + timedelta(days=1), "day")


def _unique_user_ids(user_ids: Sequence[int]) -> list[int]:
    if not isinstance(user_ids, (list, tuple)):
        raise ValueError("user_ids must be a list or tuple")
    return list(
        dict.fromkeys(_positive_integer(user_id, "user_id") for user_id in user_ids)
    )
```

Then on `CreditsStore`, immediately after `list_user_ids` (which ends `return [int(row["id"]) for row in rows]` — read it to confirm the exact last line), insert:

```python
    def aggregate_commercial_ledger(
        self,
        user_ids: Sequence[int],
        *,
        start: datetime,
        end: datetime,
    ) -> dict[int, dict[str, int]]:
        """Lifetime and windowed ledger totals for many users, three statements.

        The exact SQL ``domain/analytics/value_repository.py::list_commercial_values``
        ran against this store's connection until PR A; moved here so the
        analytics package reads the ledger through the credits domain instead
        of opening its connection (design SS6.14). The window is
        ``[start, end)`` in ISO-8601 UTC text, the format ``created_at`` holds.
        """
        ids = _unique_user_ids(user_ids)
        if not ids:
            return {}
        placeholders = ", ".join("?" for _ in ids)
        window = [_utc_text(start, "start"), _utc_text(end, "end")]
        with self._get_connection() as conn:
            lifetime_rows = conn.execute(
                f"""
                SELECT user_id,
                       COALESCE(SUM(CASE WHEN entry_type = 'purchase'
                           THEN amount_micro ELSE 0 END), 0) AS purchased_micro,
                       COALESCE(SUM(CASE WHEN entry_type = 'refund'
                           THEN -amount_micro ELSE 0 END), 0) AS refunded_micro
                FROM credit_ledger_entries
                WHERE user_id IN ({placeholders})
                  AND entry_type IN ('purchase', 'refund')
                GROUP BY user_id
                """,
                ids,
            ).fetchall()
            period_rows = conn.execute(
                f"""
                SELECT user_id,
                       COALESCE(SUM(CASE WHEN entry_type = 'purchase'
                           THEN amount_micro ELSE 0 END), 0) AS purchased_micro,
                       COALESCE(SUM(CASE WHEN entry_type = 'refund'
                           THEN -amount_micro ELSE 0 END), 0) AS refunded_micro,
                       COALESCE(SUM(CASE
                           WHEN entry_type = 'admin_grant_assign' THEN amount_micro
                           WHEN entry_type = 'admin_grant_reclaim' THEN -amount_micro
                           ELSE 0 END), 0) AS grant_activity_micro
                FROM credit_ledger_entries
                WHERE user_id IN ({placeholders})
                  AND created_at >= ?
                  AND created_at < ?
                GROUP BY user_id
                """,
                [*ids, *window],
            ).fetchall()
            usage_rows = conn.execute(
                f"""
                SELECT user_id,
                       COALESCE(SUM(-amount_micro), 0) AS consumed_micro
                FROM credit_llm_usage_entries
                WHERE user_id IN ({placeholders})
                  AND created_at >= ?
                  AND created_at < ?
                GROUP BY user_id
                """,
                [*ids, *window],
            ).fetchall()
        return _assemble_commercial_ledger(ids, lifetime_rows, period_rows, usage_rows)

    def list_credit_activity_timestamps(
        self,
        user_ids: Sequence[int],
        *,
        start: datetime,
        end: datetime,
    ) -> dict[int, list[str]]:
        """``created_at`` of every purchase and consumption inside ``[start, end)``.

        Purchases and model consumption are the two ledger movements that count
        as meaningful activity (design SS15.1); refunds and admin grants are
        not the user's own action. Text, not datetimes: the caller parses.
        """
        ids = _unique_user_ids(user_ids)
        result: dict[int, list[str]] = {user_id: [] for user_id in ids}
        if not ids:
            return result
        placeholders = ", ".join("?" for _ in ids)
        params = [*ids, _utc_text(start, "start"), _utc_text(end, "end")]
        with self._get_connection() as conn:
            purchase_rows = conn.execute(
                f"""
                SELECT user_id, created_at
                FROM credit_ledger_entries
                WHERE user_id IN ({placeholders})
                  AND entry_type = 'purchase'
                  AND created_at >= ?
                  AND created_at < ?
                """,
                params,
            ).fetchall()
            usage_rows = conn.execute(
                f"""
                SELECT user_id, created_at
                FROM credit_llm_usage_entries
                WHERE user_id IN ({placeholders})
                  AND created_at >= ?
                  AND created_at < ?
                """,
                params,
            ).fetchall()
        for row in (*purchase_rows, *usage_rows):
            result[int(row["user_id"])].append(str(row["created_at"]))
        return result

    def aggregate_ledger_for_day(self, day: date) -> dict[int, dict[str, Any]]:
        """Own spend on ``day`` plus lifetime net purchases, per user.

        The daily job's ledger step (design SS6.9 step 4). Three statements,
        none parameterised by a user id: the day's consumption grouped by
        user, the lifetime purchase-minus-refund total grouped by user (what
        ``commercial_tier()`` consumes, so the stored tier is correct as of the
        end of the day), and the latest purchase inside the day. The
        ``last_activity_at`` it yields is how a dropped ``credits_settled``
        event is prevented from leaving a paying user looking inactive.
        """
        day_start, day_end = _day_bounds(day)
        with self._get_connection() as conn:
            usage_rows = conn.execute(
                """
                SELECT user_id,
                       COALESCE(SUM(-amount_micro), 0) AS consumed_micro,
                       MAX(created_at) AS last_usage_at
                FROM credit_llm_usage_entries
                WHERE created_at >= ? AND created_at < ?
                GROUP BY user_id
                """,
                (day_start, day_end),
            ).fetchall()
            lifetime_rows = conn.execute(
                """
                SELECT user_id,
                       COALESCE(SUM(CASE WHEN entry_type = 'purchase'
                           THEN amount_micro ELSE 0 END), 0) AS purchased_micro,
                       COALESCE(SUM(CASE WHEN entry_type = 'refund'
                           THEN -amount_micro ELSE 0 END), 0) AS refunded_micro
                FROM credit_ledger_entries
                WHERE entry_type IN ('purchase', 'refund')
                GROUP BY user_id
                """
            ).fetchall()
            purchase_rows = conn.execute(
                """
                SELECT user_id, MAX(created_at) AS last_purchase_at
                FROM credit_ledger_entries
                WHERE entry_type = 'purchase'
                  AND created_at >= ? AND created_at < ?
                GROUP BY user_id
                """,
                (day_start, day_end),
            ).fetchall()
        return _assemble_ledger_day(usage_rows, lifetime_rows, purchase_rows)

    def list_account_billing_states(
        self,
        user_ids: Sequence[int] | None = None,
    ) -> dict[int, dict[str, Any]]:
        """Billing state for many accounts (``None`` = every account) in two queries.

        The batched twin of ``get_account_billing_state``: same columns, same
        restriction-reason normalisation, no account row created as a side
        effect. A user with no ``credit_accounts`` row is simply absent, which
        the caller reads as an unrestricted account -- the same answer the
        single-user reader gives after it lazily creates the row.
        """
        clause = ""
        params: list[Any] = []
        if user_ids is not None:
            ids = _unique_user_ids(user_ids)
            if not ids:
                return {}
            clause = f" WHERE user_id IN ({', '.join('?' for _ in ids)})"
            params = list(ids)
        with self._get_connection() as conn:
            account_rows = conn.execute(
                f"SELECT user_id, status, restriction_reason FROM credit_accounts{clause}",
                params,
            ).fetchall()
            outstanding_rows = conn.execute(
                f"""
                SELECT user_id,
                       COALESCE(SUM(
                           MAX(outstanding_micro - outstanding_recovered_micro, 0)
                       ), 0) AS outstanding_micro
                FROM credit_llm_reservations
                WHERE status = 'settled'{clause.replace(' WHERE ', ' AND ')}
                GROUP BY user_id
                """,
                params,
            ).fetchall()
        return _assemble_billing_states(account_rows, outstanding_rows)
```

and, at module scope immediately after `_unique_user_ids`, the three assembly helpers both twins share (the Postgres twin imports them):

```python
def _assemble_commercial_ledger(
    ids: list[int], lifetime_rows, period_rows, usage_rows
) -> dict[int, dict[str, int]]:
    lifetime = {
        int(row["user_id"]): (
            int(row["purchased_micro"] or 0),
            int(row["refunded_micro"] or 0),
        )
        for row in lifetime_rows
    }
    period = {
        int(row["user_id"]): (
            int(row["purchased_micro"] or 0),
            int(row["refunded_micro"] or 0),
            int(row["grant_activity_micro"] or 0),
        )
        for row in period_rows
    }
    usage = {int(row["user_id"]): int(row["consumed_micro"] or 0) for row in usage_rows}
    result: dict[int, dict[str, int]] = {}
    for user_id in ids:
        lifetime_purchased, lifetime_refunded = lifetime.get(user_id, (0, 0))
        purchased, refunded, grant_activity = period.get(user_id, (0, 0, 0))
        result[user_id] = {
            "lifetime_purchased_micro": lifetime_purchased,
            "lifetime_refunded_micro": lifetime_refunded,
            "purchased_micro": purchased,
            "refunded_micro": refunded,
            "grant_activity_micro": grant_activity,
            "consumed_micro": usage.get(user_id, 0),
        }
    return result


def _assemble_ledger_day(usage_rows, lifetime_rows, purchase_rows) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}

    def entry(user_id: int) -> dict[str, Any]:
        return result.setdefault(
            user_id,
            {
                "own_spend_micro": 0,
                "lifetime_net_purchased_micro": 0,
                "last_activity_at": None,
            },
        )

    def later(current: str | None, candidate: Any) -> str | None:
        if candidate is None:
            return current
        text = str(candidate)
        return text if current is None or text > current else current

    for row in usage_rows:
        record = entry(int(row["user_id"]))
        record["own_spend_micro"] = max(0, int(row["consumed_micro"] or 0))
        record["last_activity_at"] = later(record["last_activity_at"], row["last_usage_at"])
    for row in lifetime_rows:
        record = entry(int(row["user_id"]))
        record["lifetime_net_purchased_micro"] = max(
            0, int(row["purchased_micro"] or 0) - int(row["refunded_micro"] or 0)
        )
    for row in purchase_rows:
        record = entry(int(row["user_id"]))
        record["last_activity_at"] = later(
            record["last_activity_at"], row["last_purchase_at"]
        )
    return result


def _assemble_billing_states(account_rows, outstanding_rows) -> dict[int, dict[str, Any]]:
    outstanding = {
        int(row["user_id"]): int(row["outstanding_micro"] or 0) for row in outstanding_rows
    }
    result: dict[int, dict[str, Any]] = {}
    for row in account_rows:
        user_id = int(row["user_id"])
        reason = row["restriction_reason"]
        if row["status"] == "restricted" and reason not in {
            "llm_overage",
            "refund_reconciliation",
        }:
            reason = "refund_reconciliation"
        result[user_id] = {
            "account_status": row["status"],
            "restriction_reason": reason,
            "outstanding_credits_micro": outstanding.get(user_id, 0),
        }
    return result
```

`_assemble_ledger_day` returns every user that appears in **any** of the three results, so a user who bought credits last year and did nothing today still gets a `lifetime_net_purchased_micro` — that is what makes the stored `tier` correct for a dormant paying user.

- [ ] **Step 4: Add the same four methods to `PostgresCreditsStore`**

In `dashboard/backend/domain/credits/repository_postgres.py`, change:

```python
from datetime import datetime, timedelta
from typing import Any
```

to:

```python
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from typing import Any
```

and extend the existing `from dashboard.backend.domain.credits.repository_common import (...)` — no: the helpers live in `repository.py`, so add a **new** import line after the `repository_common` block:

```python
from dashboard.backend.domain.credits.repository import (
    _assemble_billing_states,
    _assemble_commercial_ledger,
    _assemble_ledger_day,
    _day_bounds,
    _unique_user_ids,
    _utc_text,
)
```

(`repository_postgres.py` does not currently import `repository.py`; check with `python -c "import dashboard.backend.domain.credits.repository_postgres"` after the edit that no import cycle appears — `repository.py` imports the Postgres twin only lazily inside `_build_credits_store`, so none does.)

Then on `PostgresCreditsStore`, immediately after `list_user_ids`, insert:

```python
    def aggregate_commercial_ledger(
        self,
        user_ids: Sequence[int],
        *,
        start: datetime,
        end: datetime,
    ) -> dict[int, dict[str, int]]:
        """See the SQLite twin."""
        ids = _unique_user_ids(user_ids)
        if not ids:
            return {}
        window = (_utc_text(start, "start"), _utc_text(end, "end"))
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_id,
                           COALESCE(SUM(CASE WHEN entry_type = 'purchase'
                               THEN amount_micro ELSE 0 END), 0) AS purchased_micro,
                           COALESCE(SUM(CASE WHEN entry_type = 'refund'
                               THEN -amount_micro ELSE 0 END), 0) AS refunded_micro
                    FROM credit_ledger_entries
                    WHERE user_id = ANY(%s)
                      AND entry_type IN ('purchase', 'refund')
                    GROUP BY user_id
                    """,
                    (ids,),
                )
                lifetime_rows = cur.fetchall()
                cur.execute(
                    """
                    SELECT user_id,
                           COALESCE(SUM(CASE WHEN entry_type = 'purchase'
                               THEN amount_micro ELSE 0 END), 0) AS purchased_micro,
                           COALESCE(SUM(CASE WHEN entry_type = 'refund'
                               THEN -amount_micro ELSE 0 END), 0) AS refunded_micro,
                           COALESCE(SUM(CASE
                               WHEN entry_type = 'admin_grant_assign' THEN amount_micro
                               WHEN entry_type = 'admin_grant_reclaim' THEN -amount_micro
                               ELSE 0 END), 0) AS grant_activity_micro
                    FROM credit_ledger_entries
                    WHERE user_id = ANY(%s)
                      AND created_at >= %s
                      AND created_at < %s
                    GROUP BY user_id
                    """,
                    (ids, *window),
                )
                period_rows = cur.fetchall()
                cur.execute(
                    """
                    SELECT user_id,
                           COALESCE(SUM(-amount_micro), 0) AS consumed_micro
                    FROM credit_llm_usage_entries
                    WHERE user_id = ANY(%s)
                      AND created_at >= %s
                      AND created_at < %s
                    GROUP BY user_id
                    """,
                    (ids, *window),
                )
                usage_rows = cur.fetchall()
        return _assemble_commercial_ledger(ids, lifetime_rows, period_rows, usage_rows)

    def list_credit_activity_timestamps(
        self,
        user_ids: Sequence[int],
        *,
        start: datetime,
        end: datetime,
    ) -> dict[int, list[str]]:
        """See the SQLite twin."""
        ids = _unique_user_ids(user_ids)
        result: dict[int, list[str]] = {user_id: [] for user_id in ids}
        if not ids:
            return result
        params = (ids, _utc_text(start, "start"), _utc_text(end, "end"))
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_id, created_at
                    FROM credit_ledger_entries
                    WHERE user_id = ANY(%s)
                      AND entry_type = 'purchase'
                      AND created_at >= %s
                      AND created_at < %s
                    """,
                    params,
                )
                purchase_rows = cur.fetchall()
                cur.execute(
                    """
                    SELECT user_id, created_at
                    FROM credit_llm_usage_entries
                    WHERE user_id = ANY(%s)
                      AND created_at >= %s
                      AND created_at < %s
                    """,
                    params,
                )
                usage_rows = cur.fetchall()
        for row in (*purchase_rows, *usage_rows):
            result[int(row["user_id"])].append(str(row["created_at"]))
        return result

    def aggregate_ledger_for_day(self, day: date) -> dict[int, dict[str, Any]]:
        """See the SQLite twin."""
        day_start, day_end = _day_bounds(day)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_id,
                           COALESCE(SUM(-amount_micro), 0) AS consumed_micro,
                           MAX(created_at) AS last_usage_at
                    FROM credit_llm_usage_entries
                    WHERE created_at >= %s AND created_at < %s
                    GROUP BY user_id
                    """,
                    (day_start, day_end),
                )
                usage_rows = cur.fetchall()
                cur.execute(
                    """
                    SELECT user_id,
                           COALESCE(SUM(CASE WHEN entry_type = 'purchase'
                               THEN amount_micro ELSE 0 END), 0) AS purchased_micro,
                           COALESCE(SUM(CASE WHEN entry_type = 'refund'
                               THEN -amount_micro ELSE 0 END), 0) AS refunded_micro
                    FROM credit_ledger_entries
                    WHERE entry_type IN ('purchase', 'refund')
                    GROUP BY user_id
                    """
                )
                lifetime_rows = cur.fetchall()
                cur.execute(
                    """
                    SELECT user_id, MAX(created_at) AS last_purchase_at
                    FROM credit_ledger_entries
                    WHERE entry_type = 'purchase'
                      AND created_at >= %s AND created_at < %s
                    GROUP BY user_id
                    """,
                    (day_start, day_end),
                )
                purchase_rows = cur.fetchall()
        return _assemble_ledger_day(usage_rows, lifetime_rows, purchase_rows)

    def list_account_billing_states(
        self,
        user_ids: Sequence[int] | None = None,
    ) -> dict[int, dict[str, Any]]:
        """See the SQLite twin."""
        account_sql = "SELECT user_id, status, restriction_reason FROM credit_accounts"
        outstanding_sql = """
            SELECT user_id,
                   COALESCE(SUM(
                       GREATEST(outstanding_micro - outstanding_recovered_micro, 0)
                   ), 0) AS outstanding_micro
            FROM credit_llm_reservations
            WHERE status = 'settled'
        """
        params: tuple[Any, ...] = ()
        if user_ids is not None:
            ids = _unique_user_ids(user_ids)
            if not ids:
                return {}
            account_sql += " WHERE user_id = ANY(%s)"
            outstanding_sql += " AND user_id = ANY(%s)"
            params = (ids,)
        outstanding_sql += " GROUP BY user_id"
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(account_sql, params)
                account_rows = cur.fetchall()
                cur.execute(outstanding_sql, params)
                outstanding_rows = cur.fetchall()
        return _assemble_billing_states(account_rows, outstanding_rows)
```

- [ ] **Step 5: Add `aggregate_operator_cost_for_day` to both run-history twins**

In `dashboard/backend/database.py`, change `from typing import List, Dict, Optional, Any` to add a datetime import directly above it:

```python
from datetime import date, timedelta
from typing import List, Dict, Optional, Any
```

and on `BacktestDatabase`, immediately after `get_runs_by_mode` (which ends `return [self._parse_run_row(dict(row)) for row in rows]`), insert:

```python
    def aggregate_operator_cost_for_day(self, day: date) -> Dict[int, int]:
        """Operator-funded model cost per owner for runs updated on ``day``, in micro-USD.

        The daily job's run step (design SS6.9 step 3): one statement grouped
        by ``owner_user_id``, parameterised by two day bounds and nothing else.
        Rows with a NULL owner -- everything before Task 2's column, and every
        scheduled leaderboard deploy -- are skipped rather than attributed to
        anyone. ``updated_at`` is CURRENT_TIMESTAMP text
        ("YYYY-MM-DD HH:MM:SS", UTC) on both twins, so the bounds are text of
        the same shape and compare correctly.

        ``est_cost_usd`` is a float; converted with ``round(value * 1_000_000)``
        and clamped at zero so a negative stored value cannot violate
        ``user_daily_facts.operator_cost_micro``'s CHECK.
        """
        start = f"{day.isoformat()} 00:00:00"
        end = f"{(day + timedelta(days=1)).isoformat()} 00:00:00"
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT owner_user_id, COALESCE(SUM(est_cost_usd), 0) AS cost_usd
            FROM agent_runs
            WHERE owner_user_id IS NOT NULL
              AND updated_at >= ? AND updated_at < ?
            GROUP BY owner_user_id
            """,
            (start, end),
        )
        rows = cursor.fetchall()
        conn.close()
        return {
            int(row["owner_user_id"]): max(0, round(float(row["cost_usd"] or 0) * 1_000_000))
            for row in rows
        }
```

In `dashboard/backend/database_postgres.py`, change `from typing import Any, Dict, List, Optional` to add above it:

```python
from datetime import date, timedelta
from typing import Any, Dict, List, Optional
```

and on `PostgresBacktestDatabase`, immediately after `get_runs_by_mode` (which ends `return [BacktestDatabase._parse_run_row(row) for row in rows]`), insert:

```python
    def aggregate_operator_cost_for_day(self, day: date) -> Dict[int, int]:
        """See the SQLite twin."""
        start = f"{day.isoformat()} 00:00:00"
        end = f"{(day + timedelta(days=1)).isoformat()} 00:00:00"
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT owner_user_id, COALESCE(SUM(est_cost_usd), 0) AS cost_usd
                    FROM agent_runs
                    WHERE owner_user_id IS NOT NULL
                      AND updated_at >= %s AND updated_at < %s
                    GROUP BY owner_user_id
                    """,
                    (start, end),
                )
                rows = cur.fetchall()
        return {
            int(row["owner_user_id"]): max(0, round(float(row["cost_usd"] or 0) * 1_000_000))
            for row in rows
        }
```

- [ ] **Step 6: Rewire the two value-store readers onto the credits store, on both twins**

In `dashboard/backend/domain/analytics/value_repository.py`, replace the whole body of `list_commercial_values` (from `ids = _ids(user_ids)` through `return result`) with:

```python
        ids = _ids(user_ids)
        window_start, window_end = _validate_window(start, end)
        if not ids:
            return {}

        # The ledger is read through the credits domain (design SS6.14). The
        # guard mirrors the old ``hasattr(self.credits_base, "_get_connection")``
        # one: retention.py constructs this store with ``credits_base=object()``.
        ledger = (
            self.credits_base.aggregate_commercial_ledger(
                ids, start=window_start, end=window_end
            )
            if hasattr(self.credits_base, "aggregate_commercial_ledger")
            else {}
        )
        balances = (
            self.credits_base.get_balance_projections(ids)
            if hasattr(self.credits_base, "get_balance_projections")
            else {}
        )
        result: dict[int, CommercialValueFact] = {}
        for user_id in ids:
            totals = ledger.get(user_id, {})
            lifetime_purchased = max(int(totals.get("lifetime_purchased_micro", 0)), 0)
            lifetime_refunded = max(int(totals.get("lifetime_refunded_micro", 0)), 0)
            purchased = max(int(totals.get("purchased_micro", 0)), 0)
            refunded = max(int(totals.get("refunded_micro", 0)), 0)
            grant_activity = max(int(totals.get("grant_activity_micro", 0)), 0)
            consumed = max(int(totals.get("consumed_micro", 0)), 0)
            net_purchased = max(lifetime_purchased - lifetime_refunded, 0)
            balance = balances.get(user_id, {})
            result[user_id] = CommercialValueFact(
                user_id=user_id,
                lifetime_net_purchased_micro=net_purchased,
                commercial_tier=commercial_tier(net_purchased),
                purchased_micro=purchased,
                refunded_micro=refunded,
                consumed_micro=consumed,
                admin_grant_activity_micro=grant_activity,
                grant_available_micro=max(
                    int(_object_value(balance, "grant_available_micro")), 0
                ),
                purchased_available_micro=max(
                    int(_object_value(balance, "purchased_available_micro")), 0
                ),
                total_available_micro=max(
                    int(_object_value(balance, "total_available_micro")), 0
                ),
            )
        return result
```

The `max(..., 0)` clamps are the ones the old assembly applied to each row (`max(int(_row_value(row, "purchased_micro", 0)), 0)` and so on), kept so every `CommercialValueFact` field is computed from the same clamped inputs as before.

Replace the whole body of `list_credit_activity` with:

```python
        ids = _ids(user_ids)
        window_start, window_end = _validate_window(start, end)
        if not ids or not hasattr(self.credits_base, "list_credit_activity_timestamps"):
            return {user_id: () for user_id in ids}
        stamps = self.credits_base.list_credit_activity_timestamps(
            ids, start=window_start, end=window_end
        )
        return {
            user_id: tuple(sorted(_timestamp(value) for value in stamps.get(user_id, ())))
            for user_id in ids
        }
```

Delete the module functions `_fetchall` and `_user_clause` (nothing calls them once both readers are rewritten; confirm with `rg -n "_fetchall|_user_clause" dashboard/backend/`). Keep `_object_value`, `_timestamp`, `_ids`, `_validate_window` and `commercial_tier` — all still used.

Apply the identical two body replacements to `PostgresValueAnalyticsStore.list_commercial_values` and `.list_credit_activity` in `value_repository_postgres.py` (they are verbatim copies, and stay verbatim copies), and remove `_fetchall,` and `_user_clause,` from its `from .value_repository import (...)` block.

In `dashboard/backend/tests/test_store_twin_parity.py`, the `_DIALECT_BRANCH_ALLOWLIST` entry for `"dashboard/backend/domain/analytics/value_repository.py"` (its reason currently begins `"list_commercial_values and list_credit_activity branch on the injected credits_base's own dialect ..."`) is now stale in reason but not in fact — `build_value_analytics_store` still says `hasattr(resolved_analytics_base, "database_url")`. Replace its value with:

```python
    "dashboard/backend/domain/analytics/value_repository.py": (
        "build_value_analytics_store() dispatches on hasattr(resolved_analytics_base, "
        "'database_url') to pick the twin, mirroring repository.py's "
        "_build_analytics_store(). That is the factory's job, not an inline "
        "dialect branch: every method on both twins has exactly one code path. "
        "The two credit readers that used to branch here moved onto "
        "CreditsStore / PostgresCreditsStore in admin layer redesign PR A."
    ),
```

- [ ] **Step 7: Teach the synthetic credits store the two batched readers**

In `dashboard/backend/tests/domain/analytics/test_value_repository.py`, `SyntheticCreditsStore` currently exposes `_get_connection`, `get_balance_projections` and `get_account_billing_state`. Append two methods to the class (after `get_account_billing_state`), keeping its trace callback so `test_commercial_value_keeps_revenue_grants_usage_and_balance_separate`'s `len(credits.select_statements) == 3` still holds:

```python
    def aggregate_commercial_ledger(self, user_ids, *, start, end):
        ids = list(dict.fromkeys(int(user_id) for user_id in user_ids))
        placeholders = ", ".join("?" for _ in ids)
        window = [start.isoformat(), end.isoformat()]
        with self._get_connection() as conn:
            lifetime = conn.execute(
                f"""
                SELECT user_id,
                       COALESCE(SUM(CASE WHEN entry_type = 'purchase'
                           THEN amount_micro ELSE 0 END), 0) AS purchased_micro,
                       COALESCE(SUM(CASE WHEN entry_type = 'refund'
                           THEN -amount_micro ELSE 0 END), 0) AS refunded_micro
                FROM credit_ledger_entries
                WHERE user_id IN ({placeholders}) AND entry_type IN ('purchase', 'refund')
                GROUP BY user_id
                """,
                ids,
            ).fetchall()
            period = conn.execute(
                f"""
                SELECT user_id,
                       COALESCE(SUM(CASE WHEN entry_type = 'purchase'
                           THEN amount_micro ELSE 0 END), 0) AS purchased_micro,
                       COALESCE(SUM(CASE WHEN entry_type = 'refund'
                           THEN -amount_micro ELSE 0 END), 0) AS refunded_micro,
                       COALESCE(SUM(CASE
                           WHEN entry_type = 'admin_grant_assign' THEN amount_micro
                           WHEN entry_type = 'admin_grant_reclaim' THEN -amount_micro
                           ELSE 0 END), 0) AS grant_activity_micro
                FROM credit_ledger_entries
                WHERE user_id IN ({placeholders}) AND created_at >= ? AND created_at < ?
                GROUP BY user_id
                """,
                [*ids, *window],
            ).fetchall()
            usage = conn.execute(
                f"""
                SELECT user_id, COALESCE(SUM(-amount_micro), 0) AS consumed_micro
                FROM credit_llm_usage_entries
                WHERE user_id IN ({placeholders}) AND created_at >= ? AND created_at < ?
                GROUP BY user_id
                """,
                [*ids, *window],
            ).fetchall()
        by_user = {
            user_id: {
                "lifetime_purchased_micro": 0,
                "lifetime_refunded_micro": 0,
                "purchased_micro": 0,
                "refunded_micro": 0,
                "grant_activity_micro": 0,
                "consumed_micro": 0,
            }
            for user_id in ids
        }
        for row in lifetime:
            by_user[int(row["user_id"])].update(
                lifetime_purchased_micro=int(row["purchased_micro"]),
                lifetime_refunded_micro=int(row["refunded_micro"]),
            )
        for row in period:
            by_user[int(row["user_id"])].update(
                purchased_micro=int(row["purchased_micro"]),
                refunded_micro=int(row["refunded_micro"]),
                grant_activity_micro=int(row["grant_activity_micro"]),
            )
        for row in usage:
            by_user[int(row["user_id"])]["consumed_micro"] = int(row["consumed_micro"])
        return by_user

    def list_credit_activity_timestamps(self, user_ids, *, start, end):
        ids = list(dict.fromkeys(int(user_id) for user_id in user_ids))
        placeholders = ", ".join("?" for _ in ids)
        params = [*ids, start.isoformat(), end.isoformat()]
        result = {user_id: [] for user_id in ids}
        with self._get_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT user_id, created_at FROM credit_ledger_entries
                WHERE user_id IN ({placeholders}) AND entry_type = 'purchase'
                  AND created_at >= ? AND created_at < ?
                """,
                params,
            ).fetchall()
            rows += conn.execute(
                f"""
                SELECT user_id, created_at FROM credit_llm_usage_entries
                WHERE user_id IN ({placeholders}) AND created_at >= ? AND created_at < ?
                """,
                params,
            ).fetchall()
        for row in rows:
            result[int(row["user_id"])].append(str(row["created_at"]))
        return result
```

The synthetic store's tables are keyed exactly as the test's raw `INSERT ... VALUES (?, ?, ?, ?)` rows expect (`user_id, entry_type, amount_micro, created_at` and `user_id, amount_micro, operation_key, created_at`), so neither existing test's seeding changes.

- [ ] **Step 8: Run the tests to verify they pass**

```bash
python -m pytest dashboard/backend/tests/test_credits_ledger_aggregates.py dashboard/backend/tests/test_run_history_aggregates.py dashboard/backend/tests/domain/analytics/test_value_repository.py dashboard/backend/tests/domain/analytics/test_value_queries.py dashboard/backend/tests/domain/analytics/test_states.py dashboard/backend/tests/domain/analytics/test_lifecycle_backfill.py dashboard/backend/tests/test_store_twin_parity.py dashboard/backend/tests/test_admin_analytics_api.py -v
TEST_POSTGRES_URL=postgresql://postgres:atl@localhost:55432/postgres \
  python -m pytest dashboard/backend/tests/ -q -k "postgres or credits or analytics"
rg -n "_get_connection" dashboard/backend/domain/analytics/value_repository.py dashboard/backend/domain/analytics/value_repository_postgres.py
```

Expected: PASS on all; the `rg` prints exactly two lines, both `return self.analytics_base._get_connection()` (one per twin). `test_admin_analytics_api.py` is the conformance oracle for `/commercial`, `/lifecycle` and `/retention` and must not change.

- [ ] **Step 9: Commit**

```bash
git add dashboard/backend/domain/credits/repository.py \
        dashboard/backend/domain/credits/repository_postgres.py \
        dashboard/backend/database.py dashboard/backend/database_postgres.py \
        dashboard/backend/domain/analytics/value_repository.py \
        dashboard/backend/domain/analytics/value_repository_postgres.py \
        dashboard/backend/tests/test_store_twin_parity.py \
        dashboard/backend/tests/domain/analytics/test_value_repository.py \
        dashboard/backend/tests/test_credits_ledger_aggregates.py \
        dashboard/backend/tests/test_run_history_aggregates.py
git status --short
git commit -m "refactor: read the credit ledger and run cost through their owning stores"
```

---

### Task 6: `backfill.py` reads through owning stores; the allowlist entry goes

**Files:**
- Modify: `dashboard/backend/domain/analytics/backfill.py` — delete `_query_all` (lines 119-131); `_credit_candidates` (lines 145-166); `AuthoritativeBackfillSource.collect` (lines 378-387); `_existing_source_event_ids` (lines 506-534)
- Modify: `dashboard/backend/domain/credits/repository.py`, `repository_postgres.py` — add `list_llm_reservation_rows`, `list_llm_usage_rows` after `list_account_billing_states` (Task 5)
- Modify: `dashboard/backend/domain/agents/repository.py`, `repository_postgres.py` — add `list_agent_source_rows` after `count_agents` (lines 697-703 / 625-630)
- Modify: `dashboard/backend/domain/analytics/repository.py`, `repository_postgres.py` — add `list_existing_source_event_ids` after `list_excluded_user_ids` (lines 529-546 / 429-445)
- Modify: `dashboard/backend/tests/test_store_twin_parity.py` — delete the `backfill.py` entry from `_DIALECT_BRANCH_ALLOWLIST`
- Modify: `dashboard/backend/tests/domain/analytics/test_backfill.py` — `SqliteRows` (lines 326-333) and its two uses (lines 475, 478)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `CreditsStore.list_llm_reservation_rows() -> list[dict[str, Any]]`, `CreditsStore.list_llm_usage_rows() -> list[dict[str, Any]]` (both twins)
  - `AgentStore.list_agent_source_rows() -> List[Dict[str, Any]]` (both twins)
  - `AnalyticsStore.list_existing_source_event_ids(source_ids: Sequence[str]) -> set[str]` (both twins)
  - `backfill.py` no longer contains `_get_connection` or `hasattr(..., "database_url")`.

D23 names `backfill.py`'s `_query_all` as the original instance of the hand-rolled cross-domain read (`hasattr(store, "database_url")` → pick a connection shape → run SQL against another domain's table), and §12 item 1 requires it fixed "in the same pass". PR T's `_DIALECT_BRANCH_ALLOWLIST` entry for this file says as much ("admin layer redesign PR A moves those two ledger and run reads onto CreditsStore/PostgresCreditsStore ... which removes this branch"); its stale-entry assertion is what forces the entry out once the branch is gone.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/backend/tests/test_credits_ledger_aggregates.py`:

```python
def test_backfill_source_rows_come_from_the_credits_store(tmp_path):
    store, alice, _bob, day_start = _store(tmp_path)
    _insert(
        tmp_path / "credits.db",
        "credit_llm_reservations",
        [
            {
                "reservation_id": "res-a6",
                "user_id": alice,
                "run_id": "run-a6",
                "call_index": 0,
                "reserved_micro": 500_000,
                "reserved_grant_micro": 500_000,
                "reserved_purchased_micro": 0,
                "status": "settled",
                "operation_key": "reserve:a6",
                "request_digest": "digest",
                "created_at": _iso(day_start + timedelta(hours=4)),
                "updated_at": _iso(day_start + timedelta(hours=5)),
            }
        ],
    )

    reservations = store.list_llm_reservation_rows()
    usage = store.list_llm_usage_rows()

    assert [row["reservation_id"] for row in reservations] == ["res-a6"]
    assert set(reservations[0]) == {
        "reservation_id", "user_id", "run_id", "call_index",
        "reserved_grant_micro", "reserved_purchased_micro", "status",
        "created_at", "updated_at",
    }
    assert [row["reservation_id"] for row in usage] == ["res-a6", "res-a7", "res-b2", "res-a8"]
    assert set(usage[0]) == {
        "id", "user_id", "reservation_id", "run_id", "call_index", "bucket",
        "amount_micro", "created_at",
    }
```

(`_insert`'s bare connection does not enforce the reservation FK from usage rows, which is why the Task 5 fixture could insert usage rows without reservations; this row satisfies the reservation table's own CHECKs — `reserved_micro = grant + purchased`, non-empty keys.)

Create the agents case in a new file `dashboard/backend/tests/test_agent_source_rows.py`:

```python
"""Agent ownership rows for the analytics backfill come from the agent store."""

from __future__ import annotations

from dashboard.backend.domain.agents.repository import AgentStore


def test_list_agent_source_rows_returns_ownership_in_creation_order(tmp_path):
    store = AgentStore(tmp_path / "agents.db")
    owned = store.create_agent(name="Owned", owner_user_id=1, session_id="session-1")
    guest = store.create_agent(name="Guest", owner_browser_session="browser-1")

    rows = store.list_agent_source_rows()

    assert [row["agent_id"] for row in rows] == [owned["agent_id"], guest["agent_id"]]
    assert set(rows[0]) == {"agent_id", "session_id", "owner_user_id", "created_at"}
    assert rows[0]["owner_user_id"] == 1
    assert rows[1]["owner_user_id"] is None
```

Append to `dashboard/backend/tests/domain/analytics/test_repository_contract.py`:

```python
def test_existing_source_event_ids_are_looked_up_in_batches(sqlite_contract):
    store, _admin_id, user_id = sqlite_contract
    for suffix in ("a", "b"):
        store.append_event(
            event_record(
                user_id,
                event_name="backtest_completed",
                event_group="run",
                event_source="server",
                source_event_id=f"run:backtest_completed:run-{suffix}",
                page_view=None,
                device_category=None,
                browser_family=None,
            )
        )

    found = store.list_existing_source_event_ids(
        ["run:backtest_completed:run-a", "run:backtest_completed:run-zzz"]
        + [f"run:backtest_completed:filler-{i}" for i in range(600)]
    )

    assert found == {"run:backtest_completed:run-a"}
    assert store.list_existing_source_event_ids([]) == set()
```

Then in `dashboard/backend/tests/domain/analytics/test_backfill.py`, replace the `SqliteRows` class:

```python
class SqliteRows:
    def __init__(self, path):
        self.path = path

    def _get_connection(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn
```

with two fakes that expose the owning-store methods instead of a connection:

```python
class FakeAgentRows:
    """What AgentStore.list_agent_source_rows() answers, without a store."""

    def __init__(self, path):
        self.path = path

    def list_agent_source_rows(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT agent_id, session_id, owner_user_id, created_at "
                    "FROM external_agents ORDER BY created_at, agent_id"
                ).fetchall()
            ]
        finally:
            conn.close()


class FakeCreditRows:
    """What CreditsStore.list_llm_*_rows() answer, without a store."""

    def __init__(self, path):
        self.path = path

    def _rows(self, sql):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            return [dict(row) for row in conn.execute(sql).fetchall()]
        finally:
            conn.close()

    def list_llm_reservation_rows(self):
        return self._rows(
            "SELECT reservation_id, user_id, run_id, call_index, "
            "reserved_grant_micro, reserved_purchased_micro, status, created_at, "
            "updated_at FROM credit_llm_reservations ORDER BY created_at, reservation_id"
        )

    def list_llm_usage_rows(self):
        return self._rows(
            "SELECT id, user_id, reservation_id, run_id, call_index, bucket, "
            "amount_micro, created_at FROM credit_llm_usage_entries ORDER BY created_at, id"
        )
```

and change the two construction lines in `test_authoritative_source_combines_safe_independent_store_evidence` from `agent_store=SqliteRows(content_path),` / `credits_store=SqliteRows(credits_path),` to `agent_store=FakeAgentRows(content_path),` / `credits_store=FakeCreditRows(credits_path),`.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest dashboard/backend/tests/test_credits_ledger_aggregates.py::test_backfill_source_rows_come_from_the_credits_store dashboard/backend/tests/test_agent_source_rows.py dashboard/backend/tests/domain/analytics/test_repository_contract.py::test_existing_source_event_ids_are_looked_up_in_batches dashboard/backend/tests/domain/analytics/test_backfill.py -v
```

Expected: FAIL with `AttributeError` on each missing store method; `test_authoritative_source_combines_safe_independent_store_evidence` fails because `backfill.py`'s `_query_all` calls `_get_connection` on the new fakes, which have none.

- [ ] **Step 3: Add the owning-store methods (both twins each)**

`dashboard/backend/domain/credits/repository.py`, on `CreditsStore` after `list_account_billing_states`:

```python
    def list_llm_reservation_rows(self) -> list[dict[str, Any]]:
        """Every LLM reservation, oldest first, for the analytics backfill.

        Safe columns only -- no evidence, no digests. ``backfill.py`` read this
        table through this store's connection until PR A (design SS6.14).
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT reservation_id, user_id, run_id, call_index,
                       reserved_grant_micro, reserved_purchased_micro,
                       status, created_at, updated_at
                FROM credit_llm_reservations
                ORDER BY created_at, reservation_id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def list_llm_usage_rows(self) -> list[dict[str, Any]]:
        """Every LLM usage entry, oldest first, for the analytics backfill."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, user_id, reservation_id, run_id, call_index,
                       bucket, amount_micro, created_at
                FROM credit_llm_usage_entries
                ORDER BY created_at, id
                """
            ).fetchall()
        return [dict(row) for row in rows]
```

`dashboard/backend/domain/credits/repository_postgres.py`, on `PostgresCreditsStore` after `list_account_billing_states`:

```python
    def list_llm_reservation_rows(self) -> list[dict[str, Any]]:
        """See the SQLite twin."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT reservation_id, user_id, run_id, call_index,
                           reserved_grant_micro, reserved_purchased_micro,
                           status, created_at, updated_at
                    FROM credit_llm_reservations
                    ORDER BY created_at, reservation_id
                    """
                )
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def list_llm_usage_rows(self) -> list[dict[str, Any]]:
        """See the SQLite twin."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, user_id, reservation_id, run_id, call_index,
                           bucket, amount_micro, created_at
                    FROM credit_llm_usage_entries
                    ORDER BY created_at, id
                    """
                )
                rows = cur.fetchall()
        return [dict(row) for row in rows]
```

`dashboard/backend/domain/agents/repository.py`, on `AgentStore` after `count_agents`:

```python
    def list_agent_source_rows(self) -> List[Dict[str, Any]]:
        """agent_id, session_id, owner_user_id, created_at for every agent.

        The analytics backfill's ownership source (design SS6.14): it used to
        read these four columns through this store's connection. Guest agents
        (NULL owner) are returned and skipped by the caller, so the caller can
        count them as unmapped rather than silently losing them.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT agent_id, session_id, owner_user_id, created_at
            FROM external_agents
            ORDER BY created_at, agent_id
            """
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
```

`dashboard/backend/domain/agents/repository_postgres.py`, on `PostgresAgentStore` after `count_agents`:

```python
    def list_agent_source_rows(self) -> List[Dict[str, Any]]:
        """See the SQLite twin."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT agent_id, session_id, owner_user_id, created_at
                    FROM external_agents
                    ORDER BY created_at, agent_id
                    """
                )
                rows = cur.fetchall()
        return [dict(row) for row in rows]
```

`dashboard/backend/domain/analytics/repository.py`, on `AnalyticsStore` after `list_excluded_user_ids` (add `Sequence` to the `from typing import Any, Iterator` line):

```python
    def list_existing_source_event_ids(self, source_ids: Sequence[str]) -> set[str]:
        """Which of ``source_ids`` already have a row. Batched by 500."""
        values = sorted({str(value) for value in source_ids})
        existing: set[str] = set()
        if not values:
            return existing
        with self._get_connection() as conn:
            for offset in range(0, len(values), 500):
                chunk = values[offset : offset + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    "SELECT source_event_id FROM analytics_events "
                    f"WHERE source_event_id IN ({placeholders})",
                    chunk,
                ).fetchall()
                existing.update(str(row["source_event_id"]) for row in rows)
        return existing
```

`dashboard/backend/domain/analytics/repository_postgres.py`, on `PostgresAnalyticsStore` after `list_excluded_user_ids` (add `from typing import Any, Sequence`):

```python
    def list_existing_source_event_ids(self, source_ids: Sequence[str]) -> set[str]:
        """See the SQLite twin."""
        values = sorted({str(value) for value in source_ids})
        existing: set[str] = set()
        if not values:
            return existing
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for offset in range(0, len(values), 500):
                    chunk = values[offset : offset + 500]
                    cur.execute(
                        "SELECT source_event_id FROM analytics_events "
                        "WHERE source_event_id = ANY(%s)",
                        (chunk,),
                    )
                    existing.update(str(row["source_event_id"]) for row in cur.fetchall())
        return existing
```

- [ ] **Step 4: Rewire `backfill.py`**

Delete `_query_all` entirely (the function at lines 119-131, including its docstring).

In `_credit_candidates`, replace:

```python
    try:
        reservations = _query_all(
            credits_store,
            """
            SELECT reservation_id, user_id, run_id, call_index,
                   reserved_grant_micro, reserved_purchased_micro,
                   status, created_at, updated_at
            FROM credit_llm_reservations
            ORDER BY created_at, reservation_id
            """,
        )
        usage_entries = _query_all(
            credits_store,
            """
            SELECT id, user_id, reservation_id, run_id, call_index,
                   bucket, amount_micro, created_at
            FROM credit_llm_usage_entries
            ORDER BY created_at, id
            """,
        )
    except Exception:
        return [], 1
```

with:

```python
    try:
        reservations = credits_store.list_llm_reservation_rows()
        usage_entries = credits_store.list_llm_usage_rows()
    except Exception:
        return [], 1
```

In `AuthoritativeBackfillSource.collect`, replace:

```python
        try:
            agents = _query_all(
                self.agent_store,
                """
                SELECT agent_id, session_id, owner_user_id, created_at
                FROM external_agents
                ORDER BY created_at, agent_id
                """,
            )
        except Exception:
            agents = []
            invalid += 1
```

with:

```python
        try:
            agents = self.agent_store.list_agent_source_rows()
        except Exception:
            agents = []
            invalid += 1
```

Replace `_existing_source_event_ids` in full with:

```python
def _existing_source_event_ids(store: Any, source_ids: Iterable[str]) -> set[str]:
    values = sorted(set(source_ids))
    if not values or not hasattr(store, "list_existing_source_event_ids"):
        return set()
    return store.list_existing_source_event_ids(values)
```

Update the module docstring's last two sentences from "The source adapter deliberately reads each database independently; cross-database ownership is resolved in Python." to "The source adapter reads each database through the store that owns it and never opens another domain's connection; cross-database ownership is resolved in Python."

In `dashboard/backend/tests/test_store_twin_parity.py`, delete the whole `"dashboard/backend/domain/analytics/backfill.py": (...)` entry from `_DIALECT_BRANCH_ALLOWLIST` — `test_dialect_branches_outside_a_registered_twin_are_allowlisted`'s `stale` assertion fails if it stays.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
python -m pytest dashboard/backend/tests/test_credits_ledger_aggregates.py dashboard/backend/tests/test_agent_source_rows.py dashboard/backend/tests/domain/analytics/test_repository_contract.py dashboard/backend/tests/domain/analytics/test_backfill.py dashboard/backend/tests/test_store_twin_parity.py -v
rg -n "_get_connection|database_url" dashboard/backend/domain/analytics/backfill.py
TEST_POSTGRES_URL=postgresql://postgres:atl@localhost:55432/postgres \
  python -m pytest dashboard/backend/tests/ -q -k "postgres"
```

Expected: PASS on all; the `rg` prints nothing.

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/domain/analytics/backfill.py \
        dashboard/backend/domain/credits/repository.py \
        dashboard/backend/domain/credits/repository_postgres.py \
        dashboard/backend/domain/agents/repository.py \
        dashboard/backend/domain/agents/repository_postgres.py \
        dashboard/backend/domain/analytics/repository.py \
        dashboard/backend/domain/analytics/repository_postgres.py \
        dashboard/backend/tests/test_store_twin_parity.py \
        dashboard/backend/tests/test_credits_ledger_aggregates.py \
        dashboard/backend/tests/test_agent_source_rows.py \
        dashboard/backend/tests/domain/analytics/test_repository_contract.py \
        dashboard/backend/tests/domain/analytics/test_backfill.py
git status --short
git commit -m "refactor: backfill reads source rows through the owning stores"
```

---

### Task 7: Operational signals for the whole population at a fixed query count

**Files:**
- Modify: `dashboard/backend/domain/analytics/lifecycle.py` — add `_FAILED_RUN_STATUSES` and `consecutive_failed_terminal_runs` after `calculate_operational_state` (line 346); export
- Modify: `dashboard/backend/domain/model_providers/repository_common.py` — add `DefaultCredentialFacts`
- Modify: `dashboard/backend/domain/model_providers/repository.py`, `repository_postgres.py` — add `list_default_credential_facts`, `list_platform_credential_statuses` after `list_all_providers` (lines 411-417 / 294-299)
- Modify: `dashboard/backend/domain/credits/repository.py`, `repository_postgres.py` — `get_balance_projections` accepts `None` (lines 898-1007 / 753-860)
- Modify: `dashboard/backend/domain/agents/repository.py`, `repository_postgres.py` — add `list_agent_owners` after `list_agent_source_rows` (Task 6)
- Modify: `dashboard/backend/domain/runs/repository.py` — add `list_terminal_runs_since` after `list_runs` (lines 257-266)
- Modify: `dashboard/backend/domain/analytics/value_repository.py` — add `_population_operational_signals` at module scope; `list_operational_signals` on `ValueAnalyticsStore`; `_run_health`'s counting loop
- Modify: `dashboard/backend/domain/analytics/value_repository_postgres.py` — `list_operational_signals`; `_run_health`'s counting loop; imports
- Create: `dashboard/backend/tests/domain/analytics/_store_spies.py`
- Test: `dashboard/backend/tests/domain/analytics/test_operational_signals.py` (create)

**Interfaces:**
- Consumes: `list_account_billing_states` (Task 5).
- Produces:
  - `consecutive_failed_terminal_runs(statuses: Sequence[str]) -> int` and `_FAILED_RUN_STATUSES` in `lifecycle.py`
  - `DefaultCredentialFacts(status: str, default_provider_ids: frozenset[str], verified_default_provider_counts: Mapping[str, int])` in `model_providers/repository_common.py`
  - `ModelProviderStore.list_default_credential_facts(user_ids: Sequence[int] | None = None) -> dict[int, DefaultCredentialFacts]` and `.list_platform_credential_statuses() -> dict[str, str]` (both twins)
  - `CreditsStore.get_balance_projections(user_ids: list[int] | tuple[int, ...] | None)` — `None` means every account (both twins; the existing list/tuple contract is unchanged)
  - `AgentStore.list_agent_owners(user_ids: Sequence[int] | None = None) -> Dict[str, int]` (both twins)
  - `RunStore.list_terminal_runs_since(*, since: datetime) -> Dict[str, Tuple[Tuple[datetime, str], ...]]`
  - `list_operational_signals(user_ids: Sequence[int], *, now: datetime, population_wide: bool = False) -> dict[int, OperationalSignals]` on both value twins — always answers for exactly `user_ids`; with `population_wide=True` the seven source reads carry no id list at all
  - `CountingSpy` and `SpyBundle` in `tests/domain/analytics/_store_spies.py`, the shared wrapping helpers Task 14 reuses

  Task 9's step 5 calls `list_operational_signals(eligible_ids, now=<end of D>, population_wide=True)` once per day.

Carried from the superseded 2026-09-12 plan's Task B7 (`git show c3bbf2ed:...`, lines 2711-3059) with one amendment from design §6.12: on the job's path every source read takes **no user-id list** (`population_wide=True`), because an `IN` list that grows with the population is a query parameterised by the population, and `_ids` caps such lists at `MAX_USER_BATCH` (500) — which would have made the job fail on the 501st user. The method still answers for exactly the ids it is given, so a user with no row in any source (no ledger entry, no credential, no agent) is computed from the model's defaults and a zero balance — the same answer `get_operational_facts` gives — rather than being silently absent. The equivalence test passes explicit ids in both modes.

Three facts about the real schema shape this task, exactly as the 09-12 plan recorded them: (1) `protocol_runs` has no owner column and lives in `DATABASE_PATH` SQLite while `external_agents.owner_user_id` lives in `CONTENT_DATABASE_URL`, so the run count is **two** statements plus a fold in Python — both set-based, so user-count independence holds; (2) the rule is consecutive-from-most-recent, not a total — failed, failed, succeeded, failed, failed inside 24 hours scores **2**, and `calculate_operational_state` fires `needs_attention` at `>= 3` with evidence that says "consecutive", so the rule is extracted once and called from both paths; (3) `usable_billing_lane` and `selected_provider_enabled` are batchable: `platform_credits_available` in `ModelProviderService.list_execution_options` depends only on the provider row, the platform credential and the environment secret, never on the user (`model_providers/service.py:162-213`).

`get_operational_facts` keeps its signature and call site exactly as they are: it is the live path for one user's profile, where one user's worth of fan-out is correct. `_run_health`'s only change is calling the shared counter.

- [ ] **Step 1: Write the shared spy and the failing tests**

Create `dashboard/backend/tests/domain/analytics/_store_spies.py`:

```python
"""Call-counting wrappers for the read-budget tests.

Not a test module (no ``test_`` prefix): imported by
test_operational_signals.py and test_read_budget.py. Wraps any store so every
public method call is recorded as ``(name, args, kwargs)`` and delegates to the
real object, so the SQL still runs against real SQLite -- the budget is
measured on real behaviour, not on a stub that agrees with everything.
"""

from __future__ import annotations

import inspect
from typing import Any


class CountingSpy:
    def __init__(self, target: Any, name: str) -> None:
        self._target = target
        self.name = name
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, attr: str) -> Any:
        value = getattr(self._target, attr)
        if attr.startswith("_") or not callable(value):
            # Private helpers (``_get_connection``) and plain attributes
            # (``analytics_base``, ``credits_base``) pass through unchanged.
            return value

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((attr, args, kwargs))
            return value(*args, **kwargs)

        return wrapped

    def reset(self) -> None:
        self.calls.clear()

    @property
    def total_calls(self) -> int:
        return len(self.calls)

    @property
    def calls_with_scalar_user_id(self) -> list[str]:
        """Calls that bound an ``int`` to a parameter named ``user_id``.

        A batched call passes a sequence (or None) and does not match. Binding
        through the real signature rather than eyeballing ``args[0]`` means a
        positional ``user_id`` is caught as surely as a keyword one.
        """
        offenders: list[str] = []
        for name, args, kwargs in self.calls:
            try:
                signature = inspect.signature(getattr(self._target, name))
                bound = signature.bind_partial(*args, **kwargs)
            except (TypeError, ValueError):
                continue
            if isinstance(bound.arguments.get("user_id"), int):
                offenders.append(f"{self.name}.{name}")
        return offenders


class SpyBundle:
    """Every spy the daily job touches, addressable by store name."""

    def __init__(self, **spies: CountingSpy) -> None:
        self.spies = spies

    def __getattr__(self, name: str) -> CountingSpy:
        try:
            return self.spies[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def reset(self) -> None:
        for spy in self.spies.values():
            spy.reset()

    @property
    def total_calls(self) -> int:
        return sum(spy.total_calls for spy in self.spies.values())

    @property
    def calls_with_scalar_user_id(self) -> list[str]:
        return [
            offender
            for spy in self.spies.values()
            for offender in spy.calls_with_scalar_user_id
        ]

    def calls_by_store(self) -> dict[str, int]:
        return {name: spy.total_calls for name, spy in self.spies.items()}
```

Create `dashboard/backend/tests/domain/analytics/test_operational_signals.py`:

```python
"""The batched operational read agrees with the per-user one.

Two implementations of one rule set drift silently. This test is the only
thing that keeps the daily job's answer equal to the profile's answer.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from cryptography.fernet import Fernet

from dashboard.backend.domain.agents.repository import AgentStore
from dashboard.backend.domain.analytics.lifecycle import (
    calculate_operational_state,
    consecutive_failed_terminal_runs,
)
from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.value_repository import (
    build_value_analytics_store,
)
from dashboard.backend.domain.brokers import repository as broker_repository
from dashboard.backend.domain.credits.repository import CreditsStore
from dashboard.backend.domain.model_providers.repository import ModelProviderStore
from dashboard.backend.domain.runs.repository import RunStore
from dashboard.backend.tests.domain.analytics._store_spies import CountingSpy, SpyBundle
from dashboard.backend.tests.test_credits_ledger_aggregates import _insert, _ledger_row
from dashboard.backend.users import UserStore


NOW = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _encryption_key(monkeypatch):
    monkeypatch.setenv("BROKER_TOKEN_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setattr(broker_repository, "_fernet_instance", None)
    # The platform lane must be decided by the stored credential alone.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("COMMONSTACK_API_KEY", raising=False)


class OperationalFixture:
    def __init__(self, path, *, users):
        self.path = path
        UserStore(db_path=path)  # creates the users table
        with sqlite3.connect(path) as conn:
            conn.executemany(
                "INSERT INTO users (id, email, display_name, password_hash, role, "
                "user_group, created_at) VALUES (?, ?, ?, 'x', 'user', 'unknown', ?)",
                [
                    (index, f"user{index}@example.test", f"User {index}",
                     (NOW - timedelta(days=30)).isoformat())
                    for index in range(1, users + 1)
                ],
            )
        self.user_ids = list(range(1, users + 1))
        self.credits = CreditsStore(path)
        self.providers = ModelProviderStore(path)
        self.agents = AgentStore(path)
        self.runs = RunStore(path)
        self.analytics = AnalyticsStore(path)
        self.spies = SpyBundle(
            credits=CountingSpy(self.credits, "credits"),
            providers=CountingSpy(self.providers, "providers"),
            agents=CountingSpy(self.agents, "agents"),
            runs=CountingSpy(self.runs, "runs"),
        )
        self.store = build_value_analytics_store(
            self.analytics,
            credits_base=self.spies.credits,
            provider_base=self.spies.providers,
            agent_base=self.spies.agents,
            run_base=self.spies.runs,
        )
        self.interleaved_failures_id = 4
        self._seed()

    # -- seeding -----------------------------------------------------------

    def _provider(self, provider_id, *, status="enabled", platform_enabled=None):
        conn = self.providers._get_connection()
        try:
            conn.execute(
                "UPDATE provider_registry SET status = ? WHERE provider_id = ?",
                (status, provider_id),
            )
            if platform_enabled is not None:
                conn.execute(
                    "UPDATE provider_registry SET platform_enabled = ? WHERE provider_id = ?",
                    (int(platform_enabled), provider_id),
                )
            conn.commit()
        finally:
            conn.close()

    def _default_credential(self, user_id, provider_id, *, status="verified"):
        created = self.providers.create_user_credential(
            user_id=user_id,
            provider_id=provider_id,
            label=f"{provider_id}-{user_id}",
            secret="sk-synthetic-secret-1234",
            status="verified",
            set_default=True,
        )
        if status != "verified":
            # The store clears is_default when a credential stops being
            # verified, so an *invalid default* -- the exact state
            # ``default_credential_status == "invalid"`` describes -- can only
            # be reached by a row that went invalid underneath its flag. Write
            # that row directly; both readers see the same table.
            conn = self.providers._get_connection()
            try:
                conn.execute(
                    "UPDATE user_model_credentials SET status = ? WHERE credential_id = ?",
                    (status, created["credential_id"]),
                )
                conn.commit()
            finally:
                conn.close()

    def _run(self, agent, status, *, hours_ago):
        run = self.runs.create_run(
            agent_id=agent["agent_id"],
            agent_version_id=None,
            session_id=agent["session_id"],
            environment_id=None,
            environment_type="backtest",
            config={},
            status=status,
        )
        stamp = (NOW - timedelta(hours=hours_ago)).isoformat()
        conn = self.runs._get_connection()
        try:
            conn.execute(
                "UPDATE protocol_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
                (stamp, stamp, run["run_id"]),
            )
            conn.commit()
        finally:
            conn.close()

    def _grant(self, user_id, amount_micro, *, key):
        _insert(
            self.path,
            "credit_ledger_entries",
            [_ledger_row(user_id, "admin_grant_assign", amount_micro, NOW - timedelta(days=1), key=key)],
        )

    def _seed(self):
        # One platform lane is open for everyone with a balance.
        self._provider("openrouter", status="enabled", platform_enabled=True)
        self.providers.upsert_platform_credential(
            provider_id="openrouter", secret="sk-platform-secret-9999", status="verified"
        )
        for user_id in self.user_ids:
            self._grant(user_id, 1_000_000, key=f"grant-{user_id}")
        # 1: restricted account (blocked, highest precedence).
        self.credits.restrict_account(1, reason="llm_overage")
        # 2: invalid default credential (needs_attention).
        self._default_credential(2, "openai", status="invalid")
        # 3: three consecutive failed terminal runs inside 24h (needs_attention).
        agent_three = self.agents.create_agent(name="Three", owner_user_id=3)
        for hours in (1, 2, 3):
            self._run(agent_three, "failed", hours_ago=hours)
        # 4: failed, failed, succeeded, failed, failed -> consecutive count 2.
        agent_four = self.agents.create_agent(name="Four", owner_user_id=4)
        for status, hours in (("failed", 1), ("failed", 2), ("completed", 3), ("failed", 4), ("failed", 5)):
            self._run(agent_four, status, hours_ago=hours)
        # 5: failures spread across two agents; pooled newest-first is what counts.
        agent_five_a = self.agents.create_agent(name="Five A", owner_user_id=5)
        agent_five_b = self.agents.create_agent(name="Five B", owner_user_id=5)
        self._run(agent_five_a, "failed", hours_ago=1)
        self._run(agent_five_b, "failed", hours_ago=2)
        self._run(agent_five_a, "completed", hours_ago=3)
        self._run(agent_five_b, "failed", hours_ago=30)  # outside 24h
        # 6: default credential on a provider that is then disabled (blocked).
        # Order matters: create_user_credential refuses a disabled provider,
        # so the credential is created first and the provider disabled after.
        # "anthropic" is one of the four SEEDED_PROVIDERS (repository_common.py)
        # and is BYOK-enabled; nobody else in this fixture uses it.
        self._default_credential(6, "anthropic")
        self._provider("anthropic", status="disabled")
        # 7: no usable lane -- remove the grant so the balance is zero (blocked).
        with sqlite3.connect(self.path) as conn:
            conn.execute("DELETE FROM credit_ledger_entries WHERE user_id = 7")
        # 8+: clean accounts (healthy).

    def reset(self):
        self.spies.reset()

    @property
    def total_calls(self):
        return self.spies.total_calls

    @property
    def calls_with_scalar_user_id(self):
        return self.spies.calls_with_scalar_user_id


def _operational_fixture(path, *, users):
    return OperationalFixture(path / "operational.db", users=users)


@pytest.fixture
def operational_fixture(tmp_path):
    return _operational_fixture(tmp_path, users=8)


def test_batched_signals_match_the_per_user_facts(operational_fixture):
    """Same users, same instant, same answer -- and every seeded state is hit."""
    store, user_ids = operational_fixture.store, operational_fixture.user_ids

    batched = store.list_operational_signals(user_ids, now=NOW)

    states = {}
    for user_id in user_ids:
        facts = store.get_operational_facts(user_id, now=NOW)
        signals = batched[user_id]
        assert signals.account_restricted == facts.account_restricted, user_id
        assert signals.usable_billing_lane == facts.usable_billing_lane, user_id
        assert signals.selected_provider_enabled == facts.selected_provider_enabled, user_id
        assert signals.default_credential_status == facts.default_credential_status, user_id
        assert signals.failed_terminal_runs_24h == facts.failed_terminal_runs_24h, user_id
        states[user_id] = calculate_operational_state(signals, NOW)
        assert states[user_id].state == calculate_operational_state(
            type(signals)(
                user_id=user_id,
                account_restricted=facts.account_restricted,
                usable_billing_lane=facts.usable_billing_lane,
                selected_provider_enabled=facts.selected_provider_enabled,
                default_credential_status=facts.default_credential_status,
                failed_terminal_runs_24h=facts.failed_terminal_runs_24h,
                run_beyond_safe_deadline=False,
            ),
            NOW,
        ).state
    assert states[1].reason_code == "account_restricted"
    assert states[2].reason_code == "invalid_default_credential"
    assert states[3].reason_code == "three_consecutive_failed_runs"
    assert states[4].state == "healthy"
    assert states[6].reason_code == "provider_disabled"
    assert states[7].reason_code == "billing_lane_unavailable"
    assert states[8].state == "healthy"


def test_the_population_path_answers_for_the_given_ids_without_an_in_list(operational_fixture):
    store = operational_fixture.store
    ids = operational_fixture.user_ids

    wide = store.list_operational_signals(ids, now=NOW, population_wide=True)
    listed = store.list_operational_signals(ids, now=NOW)

    assert set(wide) == set(ids)
    assert wide == listed
    # User 7 has no row in any source; both modes still answer for them.
    assert wide[7].usable_billing_lane is False
    assert store.list_operational_signals([], now=NOW, population_wide=True) == {}


def test_batched_signals_do_not_query_per_user(tmp_path):
    """Query count is fixed, not proportional to the population.

    Asserting a literal ceiling pins an implementation detail this task
    cannot honour: the run count alone needs two statements, because
    protocol_runs and external_agents are in different databases. The
    property that matters is that neither number moves when the user count
    does.
    """
    small = _operational_fixture(tmp_path / "small", users=10)
    large = _operational_fixture(tmp_path / "large", users=40)

    small.reset()
    small.store.list_operational_signals(small.user_ids, now=NOW, population_wide=True)
    small_calls = small.total_calls

    large.reset()
    large.store.list_operational_signals(large.user_ids, now=NOW, population_wide=True)

    assert large.calls_with_scalar_user_id == []
    assert large.total_calls == small_calls
    # A loose absolute bound as well, so "zero queries because it silently
    # returned defaults" cannot pass the equality above.
    assert 4 <= small_calls <= 8


def test_the_batched_count_is_consecutive_not_total(operational_fixture):
    """failed, failed, succeeded, failed, failed inside 24h scores 2.

    This is the one place the two implementations could disagree while
    every other assertion stayed green, because both numbers are plausible
    and only one of them matches the reason code the UI renders.
    """
    user_id = operational_fixture.interleaved_failures_id

    signals = operational_fixture.store.list_operational_signals(
        [user_id], now=NOW
    )[user_id]

    assert signals.failed_terminal_runs_24h == 2
    assert calculate_operational_state(signals, NOW).state == "healthy"


def test_runs_are_pooled_across_an_owners_agents_before_counting(operational_fixture):
    signals = operational_fixture.store.list_operational_signals([5], now=NOW)[5]

    # Newest-first across both agents: failed(A,-1h), failed(B,-2h), completed(A,-3h).
    assert signals.failed_terminal_runs_24h == 2


def test_a_user_with_no_rows_agrees_with_the_per_user_reader(operational_fixture):
    """Absence is computed, not defaulted.

    A user with no ledger row, no credential and no agent has a zero balance
    and no BYOK lane, so both readers say `billing_lane_unavailable`. The
    superseded plan asserted `healthy` here against synthetic stores whose
    defaults were permissive; against real stores the honest answer is
    "blocked", and it must be the *same* honest answer on both paths.
    """
    store = operational_fixture.store
    absent = max(operational_fixture.user_ids) + 1

    signals = store.list_operational_signals([absent], now=NOW)[absent]
    facts = store.get_operational_facts(absent, now=NOW)

    assert signals.default_credential_status == "missing"
    assert signals.failed_terminal_runs_24h == 0
    assert signals.usable_billing_lane is facts.usable_billing_lane is False
    assert (
        calculate_operational_state(signals, NOW).reason_code
        == "billing_lane_unavailable"
    )


@pytest.mark.parametrize(
    "statuses,expected",
    [
        ((), 0),
        (("failed", "failed", "failed"), 3),
        (("failed", "timed_out", "completed", "failed"), 2),
        (("completed", "failed", "failed"), 0),
        (("cancelled", "failed"), 0),
    ],
)
def test_consecutive_failed_terminal_runs_counts_leading_failures(statuses, expected):
    assert consecutive_failed_terminal_runs(statuses) == expected
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_operational_signals.py -v
```

Expected: FAIL at import — `ImportError: cannot import name 'consecutive_failed_terminal_runs' from 'dashboard.backend.domain.analytics.lifecycle'`.

- [ ] **Step 3: Extract the consecutive-failure rule**

In `dashboard/backend/domain/analytics/lifecycle.py`, immediately after `calculate_operational_state` (which ends `return _operational_result("healthy", "no_supported_issue", calculation_time,)`), insert:

```python
_FAILED_RUN_STATUSES = frozenset({"failed", "timed_out"})


def consecutive_failed_terminal_runs(statuses: Sequence[str]) -> int:
    """Count leading failures in a newest-first sequence of terminal statuses.

    The rule `calculate_operational_state` reports as
    "three_consecutive_failed_runs", lifted out of `_run_health` so the live
    profile and the daily job cannot answer it differently. A total count is
    a different number -- failed, failed, succeeded, failed, failed is 2 here
    and 4 to a `COUNT(*)` -- and the reason code the UI renders says
    "consecutive", so the total would be wrong on screen as well as
    inconsistent between the two paths.

    Pure and sequence-shaped rather than query-shaped on purpose: one caller
    has rows from a store's list, the other from a cross-database fold, and
    neither can express this in SQL over its own data alone.
    """
    count = 0
    for status in statuses:
        if status not in _FAILED_RUN_STATUSES:
            break
        count += 1
    return count
```

Add `"consecutive_failed_terminal_runs",` to `__all__` (after `"commercial_tier",`).

- [ ] **Step 4: Add the batched store reads to the owning domains (both twins each)**

`dashboard/backend/domain/model_providers/repository_common.py`: add `from dataclasses import dataclass` to the imports and, after the exception classes, insert:

```python
@dataclass(frozen=True)
class DefaultCredentialFacts:
    """One user's default credentials, folded to what operational state needs.

    ``status`` follows get_operational_facts' worst-first precedence: invalid
    beats verification_unavailable beats verified. ``default_provider_ids``
    carries every provider a default credential points at, and
    ``verified_default_provider_counts`` the per-provider count of *verified*
    defaults -- the ``== 1`` test in ModelProviderService.list_execution_options
    (service.py:194) is a count, not a boolean, and collapsing it to one loses
    the two-defaults case.
    """

    status: str
    default_provider_ids: frozenset[str]
    verified_default_provider_counts: Mapping[str, int]


def fold_default_credential_rows(rows) -> dict[int, DefaultCredentialFacts]:
    """Shared by both twins: rows of (user_id, provider_id, status) -> facts."""
    statuses: dict[int, set[str]] = {}
    providers: dict[int, set[str]] = {}
    verified: dict[int, dict[str, int]] = {}
    for row in rows:
        user_id = int(row["user_id"])
        provider_id = str(row["provider_id"])
        status = str(row["status"])
        statuses.setdefault(user_id, set()).add(status)
        providers.setdefault(user_id, set()).add(provider_id)
        if status == "verified":
            counts = verified.setdefault(user_id, {})
            counts[provider_id] = counts.get(provider_id, 0) + 1
    result: dict[int, DefaultCredentialFacts] = {}
    for user_id, seen in statuses.items():
        if "invalid" in seen:
            status = "invalid"
        elif "verification_unavailable" in seen:
            status = "verification_unavailable"
        elif "verified" in seen:
            status = "verified"
        else:
            status = "missing"
        result[user_id] = DefaultCredentialFacts(
            status=status,
            default_provider_ids=frozenset(providers[user_id]),
            verified_default_provider_counts=dict(verified.get(user_id, {})),
        )
    return result
```

`dashboard/backend/domain/model_providers/repository.py`: add `DefaultCredentialFacts, fold_default_credential_rows,` to the `from .repository_common import (...)` block, `from collections.abc import Sequence` to the imports, and on `ModelProviderStore` immediately after `list_all_providers`:

```python
    def list_default_credential_facts(
        self, user_ids: Sequence[int] | None = None
    ) -> dict[int, DefaultCredentialFacts]:
        """Each user's default credentials as one row per user (``None`` = everyone).

        Same predicate ``list_user_credentials`` applies (``status <> 'revoked'``)
        narrowed to ``is_default = 1``, so the per-user reader and this batched
        one see identical rows. A user with no default credential is absent;
        the caller reads absence as ``missing``.
        """
        clause = ""
        params: list[Any] = []
        if user_ids is not None:
            ids = list(dict.fromkeys(int(user_id) for user_id in user_ids))
            if not ids:
                return {}
            clause = f" AND user_id IN ({', '.join('?' for _ in ids)})"
            params = ids
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT user_id, provider_id, status FROM user_model_credentials "
            f"WHERE is_default = 1 AND status <> 'revoked'{clause}",
            params,
        ).fetchall()
        conn.close()
        return fold_default_credential_rows(rows)

    def list_platform_credential_statuses(self) -> dict[str, str]:
        """Every provider's platform-credential status. Takes no user id.

        The population-wide half of ``platform_credits_available``: it depends
        only on the provider row, its platform credential and the environment
        secret, never on who is asking. One call for the whole job.
        """
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT provider_id, status FROM platform_model_credentials"
        ).fetchall()
        conn.close()
        return {str(row["provider_id"]): str(row["status"]) for row in rows}
```

`dashboard/backend/domain/model_providers/repository_postgres.py`: the same two imports, and on `PostgresModelProviderStore` after `list_all_providers`:

```python
    def list_default_credential_facts(
        self, user_ids: Sequence[int] | None = None
    ) -> dict[int, DefaultCredentialFacts]:
        """See the SQLite twin."""
        sql = (
            "SELECT user_id, provider_id, status FROM user_model_credentials "
            "WHERE is_default = TRUE AND status <> 'revoked'"
        )
        params: tuple[Any, ...] = ()
        if user_ids is not None:
            ids = list(dict.fromkeys(int(user_id) for user_id in user_ids))
            if not ids:
                return {}
            sql += " AND user_id = ANY(%s)"
            params = (ids,)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return fold_default_credential_rows(rows)

    def list_platform_credential_statuses(self) -> dict[str, str]:
        """See the SQLite twin."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT provider_id, status FROM platform_model_credentials")
                rows = cur.fetchall()
        return {str(row["provider_id"]): str(row["status"]) for row in rows}
```

`dashboard/backend/domain/credits/repository.py`, `get_balance_projections`: change the signature and the guard

```python
    def get_balance_projections(
        self, user_ids: list[int] | tuple[int, ...]
    ) -> dict[int, dict[str, int]]:
        if not isinstance(user_ids, (list, tuple)):
            raise ValueError("user_ids must be a list or tuple")
        validated = [_positive_integer(user_id, "user_id") for user_id in user_ids]
        if not validated:
            return {}

        unique_ids = list(dict.fromkeys(validated))
        placeholders = ", ".join("?" for _ in unique_ids)
```

to:

```python
    def get_balance_projections(
        self, user_ids: list[int] | tuple[int, ...] | None
    ) -> dict[int, dict[str, int]]:
        """Balances for many accounts; ``None`` means every account with a row.

        ``None`` is the daily job's shape (design SS6.12): the four statements
        below then carry no ``WHERE user_id IN`` clause at all, so the query
        count and shape are the same at 200 and 20,000 users.
        """
        if user_ids is None:
            unique_ids = None
            where = ""
            params: list[Any] = []
        else:
            if not isinstance(user_ids, (list, tuple)):
                raise ValueError("user_ids must be a list or tuple")
            validated = [_positive_integer(user_id, "user_id") for user_id in user_ids]
            if not validated:
                return {}
            unique_ids = list(dict.fromkeys(validated))
            where = f"WHERE user_id IN ({', '.join('?' for _ in unique_ids)})"
            params = list(unique_ids)
```

then replace each of the four `WHERE user_id IN ({placeholders})` in that method's SQL with `{where}` (for the reservations statement the existing text is `WHERE user_id IN ({placeholders}) AND status = 'open'`; it becomes `{where} {'AND' if where else 'WHERE'} status = 'open'` — write it as two literal branches to keep the SQL greppable:

```python
            reservation_filter = (
                f"{where} AND status = 'open'" if where else "WHERE status = 'open'"
            )
```

and use `{reservation_filter}`), pass `params` instead of `unique_ids` to each `execute`, and change the assembly loop's header from `for user_id in unique_ids:` to:

```python
        if unique_ids is None:
            unique_ids = sorted(set(amounts) | set(usage_amounts) | set(reserved_amounts))
        for user_id in unique_ids:
```

Mirror the same change on `PostgresCreditsStore.get_balance_projections` (`WHERE user_id = ANY(%s)` becomes `{where}` with `where = "WHERE user_id = ANY(%s)"` and `params = (unique_ids,)`, the reservations filter as above, and the same `if unique_ids is None:` line before the assembly loop). `test_store_twin_parity.py::test_postgres_twin_signatures_match_sqlite` checks the two new signatures agree.

`dashboard/backend/domain/agents/repository.py`, on `AgentStore` after `list_agent_source_rows` (add `Sequence` to `from typing import Any, Dict, List, Optional`):

```python
    def list_agent_owners(self, user_ids: Sequence[int] | None = None) -> Dict[str, int]:
        """agent_id -> owner_user_id for these owners (``None`` = every owned agent).

        `external_agents.owner_user_id` is indexed
        (`idx_external_agents_owner_user`), and this table is in
        CONTENT_DATABASE_URL while protocol_runs is in DATABASE_PATH, so this
        mapping has to come back to Python before the runs can be counted.
        Rows with a NULL owner are omitted rather than grouped.
        """
        clause = ""
        params: List[Any] = []
        if user_ids is not None:
            ids = list(dict.fromkeys(int(user_id) for user_id in user_ids))
            if not ids:
                return {}
            clause = f" AND owner_user_id IN ({', '.join('?' for _ in ids)})"
            params = ids
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT agent_id, owner_user_id FROM external_agents "
            f"WHERE owner_user_id IS NOT NULL{clause}",
            params,
        )
        rows = cursor.fetchall()
        conn.close()
        return {str(row["agent_id"]): int(row["owner_user_id"]) for row in rows}
```

`dashboard/backend/domain/agents/repository_postgres.py`, on `PostgresAgentStore` after `list_agent_source_rows` (add `Sequence` to its typing import):

```python
    def list_agent_owners(self, user_ids: Sequence[int] | None = None) -> Dict[str, int]:
        """See the SQLite twin."""
        sql = "SELECT agent_id, owner_user_id FROM external_agents WHERE owner_user_id IS NOT NULL"
        params: tuple[Any, ...] = ()
        if user_ids is not None:
            ids = list(dict.fromkeys(int(user_id) for user_id in user_ids))
            if not ids:
                return {}
            sql += " AND owner_user_id = ANY(%s)"
            params = (ids,)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return {str(row["agent_id"]): int(row["owner_user_id"]) for row in rows}
```

`dashboard/backend/domain/runs/repository.py`, on `RunStore` after `list_runs` (add `Tuple` to `from typing import Any, Dict, List, Optional`):

```python
    _TERMINAL_STATUSES = ("completed", "failed", "cancelled", "closed", "timed_out")

    def list_terminal_runs_since(
        self,
        *,
        since: datetime,
    ) -> Dict[str, Tuple[Tuple[datetime, str], ...]]:
        """Terminal runs for every agent since ``since``, newest first per agent.

        Returns `(effective_time, status)` pairs per agent, where the effective
        time is `COALESCE(updated_at, created_at)` -- the same ordering key
        `_run_health` uses. Bounded by construction: a 24-hour window of
        terminal runs across the whole population is a small result, and the
        caller pools and re-sorts them per owner because the consecutive rule
        runs over an owner's agents together, not per agent.

        The two timestamp columns hold mixed shapes (``CURRENT_TIMESTAMP``
        text on old rows, ISO-8601 with an offset from ``update_run``), so the
        SQL bounds the scan to calendar days by the 10-character date prefix
        and the exact comparison happens in Python.
        """
        if since.tzinfo is None or since.utcoffset() is None:
            raise ValueError("since must include a timezone")
        floor = since.astimezone(timezone.utc).date().isoformat()
        placeholders = ", ".join("?" for _ in self._TERMINAL_STATUSES)
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT agent_id, status, created_at, updated_at
            FROM protocol_runs
            WHERE agent_id IS NOT NULL
              AND status IN ({placeholders})
              AND substr(COALESCE(updated_at, created_at), 1, 10) >= ?
            """,
            (*self._TERMINAL_STATUSES, floor),
        )
        rows = cursor.fetchall()
        conn.close()
        grouped: Dict[str, List[Tuple[datetime, str]]] = {}
        for row in rows:
            stamp = _parse_run_timestamp(row["updated_at"]) or _parse_run_timestamp(
                row["created_at"]
            )
            if stamp is None or stamp < since:
                continue
            grouped.setdefault(str(row["agent_id"]), []).append((stamp, str(row["status"])))
        return {
            agent_id: tuple(sorted(pairs, key=lambda pair: pair[0], reverse=True))
            for agent_id, pairs in grouped.items()
        }
```

and at module scope, after `_public_run`:

```python
def _parse_run_timestamp(value: object) -> Optional[datetime]:
    """Both shapes protocol_runs holds: CURRENT_TIMESTAMP text and ISO-8601."""
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
```

`RunStore` has no Postgres twin (it is hardcoded to `DATABASE_PATH`; see `CLAUDE.md`, "Persistence"), so this is the one method in this task with a single implementation.

- [ ] **Step 5: Compose them on both value twins and share the counter with `_run_health`**

In `dashboard/backend/domain/analytics/value_repository.py`, add `OperationalSignals, consecutive_failed_terminal_runs,` to the `from .lifecycle import (...)` block, and at module scope (after `_recent_totals_from_row`) insert:

```python
PLATFORM_CREDENTIAL_ENVIRONMENT_SECRET = "dashboard.backend.domain.model_providers.service"


def _platform_lane_open(providers: Sequence[Mapping[str, Any]], platform_statuses: Mapping[str, str]) -> bool:
    """The population-wide half of ``platform_credits_available``.

    Mirrors ModelProviderService.list_execution_options (service.py:175-186):
    an enabled, platform-enabled provider whose platform credential is
    verified or whose deployment secret is set. No user appears in this
    expression, which is the whole reason the lane is batchable at all.
    """
    from dashboard.backend.domain.model_providers.service import (
        _environment_platform_secret,
    )

    for provider in providers:
        provider_id = str(provider.get("provider_id"))
        if provider.get("status") != "enabled" or not provider.get("platform_enabled"):
            continue
        if platform_statuses.get(provider_id) == "verified":
            return True
        if _environment_platform_secret(provider_id):
            return True
    return False


def _population_operational_signals(
    store: Any,
    user_ids: Sequence[int],
    *,
    now: datetime,
    population_wide: bool,
) -> dict[int, OperationalSignals]:
    """Shared by both twins: seven owning-store calls, one fold, no per-user query.

    ``user_ids`` is always the set answered for. ``population_wide`` decides
    how the sources are *read*: False passes the ids as an IN list (the
    live/batched shape, capped by ``_ids``); True passes ``None`` so every
    statement is population-wide and the same at any user count -- the daily
    job's shape. Either way a user with no row in any source is computed from
    the model's defaults and a zero balance, exactly as ``get_operational_facts``
    computes them, rather than being dropped.
    """
    current = _utc(now, "now")
    if population_wide:
        if not isinstance(user_ids, (list, tuple)):
            raise ValueError("user_ids must be a list or tuple")
        ids = list(dict.fromkeys(positive_user_id(item) for item in user_ids))
        query_ids = None
    else:
        ids = _ids(user_ids)
        query_ids = ids
    if not ids:
        return {}

    def batched(base: Any, method: str, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(base, method):
            return {}
        result = getattr(base, method)(*args, **kwargs)
        if not result:
            # Fail-visible: every OperationalSignals default is permissive, so
            # a source that silently answers nothing reads as "everyone
            # healthy". A fresh deploy may legitimately have no rows, which is
            # why this is a line and not an exception.
            print(f"WARNING: analytics.operational_signals_empty source={method}")
        return result

    balances = batched(store.credits_base, "get_balance_projections", query_ids)          # 1
    billing = batched(store.credits_base, "list_account_billing_states", query_ids)       # 2
    credentials = batched(store.provider_base, "list_default_credential_facts", query_ids)  # 3
    providers = (
        list(store.provider_base.list_all_providers())                                    # 4
        if hasattr(store.provider_base, "list_all_providers")
        else []
    )
    platform_statuses = batched(store.provider_base, "list_platform_credential_statuses")  # 5
    owners = batched(store.agent_base, "list_agent_owners", query_ids)                    # 6
    runs = (
        store.run_base.list_terminal_runs_since(since=current - timedelta(hours=24))      # 7
        if hasattr(store.run_base, "list_terminal_runs_since")
        else {}
    )

    providers_by_id = {str(row.get("provider_id")): row for row in providers}
    platform_open = _platform_lane_open(providers, platform_statuses)
    agents_by_owner: dict[int, list[str]] = {}
    for agent_id, owner in owners.items():
        agents_by_owner.setdefault(int(owner), []).append(agent_id)

    signals: dict[int, OperationalSignals] = {}
    for user_id in ids:
        facts = credentials.get(user_id)
        default_ids = facts.default_provider_ids if facts is not None else frozenset()
        verified_counts = facts.verified_default_provider_counts if facts is not None else {}
        selected_enabled = all(
            providers_by_id.get(provider_id, {}).get("status") == "enabled"
            for provider_id in default_ids
        )
        verified_byok = any(
            row.get("status") == "enabled"
            and bool(row.get("byok_enabled"))
            and verified_counts.get(str(row.get("provider_id")), 0) == 1
            for row in providers
        )
        total_available = int(
            _object_value(balances.get(user_id, {}), "total_available_micro")
        )
        platform_lane = total_available > 0 and platform_open
        pooled = sorted(
            (
                pair
                for agent_id in agents_by_owner.get(user_id, ())
                for pair in runs.get(agent_id, ())
                if pair[0] <= current
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        signals[user_id] = OperationalSignals(
            user_id=user_id,
            account_restricted=(
                billing.get(user_id, {}).get("account_status") == "restricted"
            ),
            usable_billing_lane=platform_lane or verified_byok,
            selected_provider_enabled=selected_enabled,
            default_credential_status=facts.status if facts is not None else "missing",
            failed_terminal_runs_24h=consecutive_failed_terminal_runs(
                [status for _stamp, status in pooled]
            ),
            # A live condition about a run happening right now; a fact row for
            # a completed day cannot meaningfully carry it.
            run_beyond_safe_deadline=False,
        )
    return signals
```

Then on `ValueAnalyticsStore`, immediately after `get_operational_facts`, insert:

```python
    def list_operational_signals(
        self,
        user_ids: Sequence[int],
        *,
        now: datetime,
        population_wide: bool = False,
    ) -> dict[int, OperationalSignals]:
        """Operational signals for many users, at a fixed query count.

        The per-user twin, ``get_operational_facts``, fans out into roughly
        five store calls plus one per agent the user owns. That is right for
        one profile and catastrophic for a population, so the daily job uses
        this instead, with ``population_wide=True`` so no statement carries an
        IN list that grows with the population. Both must produce the same
        answer; the equivalence is pinned by
        tests/domain/analytics/test_operational_signals.py.
        """
        return _population_operational_signals(
            self, user_ids, now=now, population_wide=population_wide
        )
```

and in `_run_health` (both twins), replace:

```python
        consecutive_failures = 0
        for run in terminal_24h:
            if str(run.get("status")) not in {"failed", "timed_out"}:
                break
            consecutive_failures += 1
```

with:

```python
        consecutive_failures = consecutive_failed_terminal_runs(
            [str(run.get("status")) for run in terminal_24h]
        )
```

In `value_repository_postgres.py`, add `_population_operational_signals,` to the import block, `consecutive_failed_terminal_runs` and `OperationalSignals` via `from .lifecycle import OperationalSignals, consecutive_failed_terminal_runs`, and the identical `list_operational_signals` method on `PostgresValueAnalyticsStore` after its `get_operational_facts`.

Seven calls, not four. The number is incidental; what matters is that none of them takes a scalar user id and none of them grows with the population — which is what this task's budget case and Task 14's guard assert. `positive_user_id` is already imported in `value_repository.py` from `.repository_common`; the Postgres twin does not need it because the shared function lives in the SQLite module. `selected_provider_enabled` for a user with no default is `all(())` — `True` — which is the existing behaviour and must stay: "no default credential" is not "a disabled provider".

- [ ] **Step 6: Run the tests to verify they pass**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_operational_signals.py dashboard/backend/tests/domain/analytics/test_value_repository.py dashboard/backend/tests/test_store_twin_parity.py dashboard/backend/tests/domain/model_providers/ dashboard/backend/tests/test_credits_ledger_aggregates.py -v
python -m pytest dashboard/backend/tests/ -q -k "credits or agent or run_store or protocol"
TEST_POSTGRES_URL=postgresql://postgres:atl@localhost:55432/postgres \
  python -m pytest dashboard/backend/tests/ -q -k "postgres"
```

Expected: PASS on all. `test_value_repository.py::test_operational_facts_use_public_sources_without_reading_credentials` still asserts `failed_terminal_runs_24h == 3` — the proof that extracting the counter changed no behaviour.

- [ ] **Step 7: Commit**

```bash
git add dashboard/backend/domain/analytics/lifecycle.py \
        dashboard/backend/domain/analytics/value_repository.py \
        dashboard/backend/domain/analytics/value_repository_postgres.py \
        dashboard/backend/domain/model_providers/repository_common.py \
        dashboard/backend/domain/model_providers/repository.py \
        dashboard/backend/domain/model_providers/repository_postgres.py \
        dashboard/backend/domain/credits/repository.py \
        dashboard/backend/domain/credits/repository_postgres.py \
        dashboard/backend/domain/agents/repository.py \
        dashboard/backend/domain/agents/repository_postgres.py \
        dashboard/backend/domain/runs/repository.py \
        dashboard/backend/tests/domain/analytics/_store_spies.py \
        dashboard/backend/tests/domain/analytics/test_operational_signals.py
git status --short
git commit -m "perf: batch operational signals across the whole population"
```

---

### Task 8: Claim one UTC day exactly once, with a reclaimable lease

**Files:**
- Modify: `dashboard/backend/domain/analytics/value_repository.py` — add `claim_projection_day`, `complete_projection_day`, `release_projection_day` to `ValueAnalyticsStore` after `save_projection_job`
- Modify: `dashboard/backend/domain/analytics/value_repository_postgres.py` — the same three methods on `PostgresValueAnalyticsStore`
- Test: `dashboard/backend/tests/domain/analytics/test_value_repository.py` (extend)

**Interfaces:**
- Consumes: `analytics_projection_jobs`, which already exists on both twins with `cursor`, `status ∈ {pending, running, complete}` and `updated_at`.
- Produces, on both value twins:
  - `claim_projection_day(job_name: str, *, day: date, now: datetime, stale_after: timedelta = timedelta(hours=2)) -> bool`
  - `complete_projection_day(job_name: str, *, day: date, now: datetime) -> None`
  - `release_projection_day(job_name: str, *, now: datetime) -> None`

  Task 9's job calls all three once per tick at most.

Carried from the superseded 2026-09-12 plan's Task B6 (`git show c3bbf2ed:...`, lines 2540-2707) with the design §6.9 amendment: the claim and the cursor are **two fields**. The 09-12 version moved `cursor` to `D` in the same statement that claimed it, so a job that crashed after claiming could never retry — "never twice" and "retry on failure" were mutually exclusive. Here the compare-and-set moves only `status` (`pending`/`complete` → `running`) and stamps `updated_at`; the `cursor` advances to `D` and `status` returns to `complete` only in `complete_projection_day`, which the job calls after its fact step succeeds. A `running` claim whose `updated_at` is older than `stale_after` is treated as abandoned and may be re-claimed, so a crash mid-job retries on the next tick rather than never. This replaces the process-local `_last_rollup_day` guard in `maintenance.py`, which only ever protected a single process and did nothing about a restart.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/backend/tests/domain/analytics/test_value_repository.py` (it already imports `date`, `datetime`, `timedelta`, `timezone` and defines `_stores`/`_value_store`):

```python
JOB = "analytics_daily_facts"
DAY = date(2026, 9, 11)
CLAIM_AT = datetime(2026, 9, 12, 0, 5, tzinfo=timezone.utc)


def test_only_one_claim_of_a_day_succeeds(tmp_path):
    _user_id, analytics, credits = _stores(tmp_path)
    store = _value_store(analytics, credits)

    first = store.claim_projection_day(JOB, day=DAY, now=CLAIM_AT)
    second = store.claim_projection_day(JOB, day=DAY, now=CLAIM_AT + timedelta(minutes=1))
    job = store.get_projection_job(JOB)

    assert first is True
    assert second is False
    assert job.status == "running"
    assert job.cursor is None  # the cursor moves only when the day completes


def test_a_crashed_claim_is_reclaimable_after_two_hours(tmp_path):
    _user_id, analytics, credits = _stores(tmp_path)
    store = _value_store(analytics, credits)
    store.claim_projection_day(JOB, day=DAY, now=CLAIM_AT)

    too_soon = store.claim_projection_day(JOB, day=DAY, now=CLAIM_AT + timedelta(hours=1))
    reclaimed = store.claim_projection_day(JOB, day=DAY, now=CLAIM_AT + timedelta(hours=2, minutes=1))

    assert too_soon is False
    assert reclaimed is True


def test_a_completed_day_is_never_run_again(tmp_path):
    _user_id, analytics, credits = _stores(tmp_path)
    store = _value_store(analytics, credits)
    store.claim_projection_day(JOB, day=DAY, now=CLAIM_AT)
    store.complete_projection_day(JOB, day=DAY, now=CLAIM_AT + timedelta(minutes=2))

    again = store.claim_projection_day(JOB, day=DAY, now=CLAIM_AT + timedelta(hours=5))
    earlier = store.claim_projection_day(JOB, day=DAY - timedelta(days=1), now=CLAIM_AT + timedelta(hours=5))
    next_day = store.claim_projection_day(
        JOB, day=DAY + timedelta(days=1), now=CLAIM_AT + timedelta(days=1)
    )
    job = store.get_projection_job(JOB)

    assert again is False
    assert earlier is False
    assert next_day is True
    assert job.cursor == DAY.isoformat()
    assert job.status == "running"
    assert job.window_end == DAY + timedelta(days=1)


def test_a_released_claim_can_be_retried_at_once(tmp_path):
    _user_id, analytics, credits = _stores(tmp_path)
    store = _value_store(analytics, credits)
    store.claim_projection_day(JOB, day=DAY, now=CLAIM_AT)

    store.release_projection_day(JOB, now=CLAIM_AT + timedelta(minutes=1))
    retried = store.claim_projection_day(JOB, day=DAY, now=CLAIM_AT + timedelta(minutes=2))
    job = store.get_projection_job(JOB)

    assert retried is True
    assert job.cursor is None
    assert job.status == "running"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_value_repository.py -v -k "claim or completed_day or released_claim"
```

Expected: FAIL with `AttributeError: 'ValueAnalyticsStore' object has no attribute 'claim_projection_day'`.

- [ ] **Step 3: Implement the claim on the SQLite twin**

On `ValueAnalyticsStore`, immediately after `save_projection_job` (which ends `return job`), insert:

```python
    def claim_projection_day(
        self,
        job_name: str,
        *,
        day: date,
        now: datetime,
        stale_after: timedelta = timedelta(hours=2),
    ) -> bool:
        """Take the lease on ``day`` for ``job_name``; True for the caller that won.

        Two fields, two questions (design SS6.9). ``cursor`` answers "which
        day is done" and only ``complete_projection_day`` moves it, so a day
        is never run twice once it succeeded. ``status`` answers "is someone
        running it now": the compare-and-set below moves it from
        ``pending``/``complete`` to ``running`` and stamps ``updated_at``; a
        second process whose UPDATE matches nothing gets False. A ``running``
        lease older than ``stale_after`` is a crash, not a worker, and may be
        taken over -- which is how a job killed mid-day retries on the next
        tick instead of never.

        The cursor comparison is lexicographic on ISO dates. ``cursor >= day``
        means the day (or a later one) already completed and the claim is
        refused without a write.
        """
        name = _projection_job_name(job_name)
        target = day.isoformat()
        stamp = utc_iso(_utc(now, "now"))
        stale_before = utc_iso(_utc(now, "now") - stale_after)
        with self._analytics_connection() as conn:
            conn.execute(
                """
                INSERT INTO analytics_projection_jobs (
                    job_name, window_start, window_end, cursor, status, updated_at
                ) VALUES (?, ?, ?, NULL, 'pending', ?)
                ON CONFLICT(job_name) DO NOTHING
                """,
                (name, target, target, stamp),
            )
            cursor = conn.execute(
                """
                UPDATE analytics_projection_jobs
                   SET status = 'running', window_end = ?, updated_at = ?
                 WHERE job_name = ?
                   AND (cursor IS NULL OR cursor < ?)
                   AND (
                        status IN ('pending', 'complete')
                        OR (status = 'running' AND updated_at < ?)
                   )
                """,
                (target, stamp, name, target, stale_before),
            )
            return cursor.rowcount == 1

    def complete_projection_day(self, job_name: str, *, day: date, now: datetime) -> None:
        """Record ``day`` as done: cursor -> day, status -> complete."""
        name = _projection_job_name(job_name)
        with self._analytics_connection() as conn:
            conn.execute(
                """
                UPDATE analytics_projection_jobs
                   SET cursor = ?, status = 'complete', updated_at = ?
                 WHERE job_name = ?
                """,
                (day.isoformat(), utc_iso(_utc(now, "now")), name),
            )

    def release_projection_day(self, job_name: str, *, now: datetime) -> None:
        """Give a failed day back: status -> pending, cursor untouched.

        Called when a step failed, so the next tick's claim retries the same
        day at once instead of waiting out the stale window.
        """
        name = _projection_job_name(job_name)
        with self._analytics_connection() as conn:
            conn.execute(
                """
                UPDATE analytics_projection_jobs
                   SET status = 'pending', updated_at = ?
                 WHERE job_name = ? AND status = 'running'
                """,
                (utc_iso(_utc(now, "now")), name),
            )
```

The `INSERT ... ON CONFLICT DO NOTHING` makes the first-ever call safe under a race: exactly one process creates the row, both then compete on the `UPDATE`, and exactly one `UPDATE` matches.

- [ ] **Step 4: Implement the same three on the Postgres twin**

On `PostgresValueAnalyticsStore`, immediately after `save_projection_job`:

```python
    def claim_projection_day(
        self,
        job_name: str,
        *,
        day: date,
        now: datetime,
        stale_after: timedelta = timedelta(hours=2),
    ) -> bool:
        """See the SQLite twin."""
        name = _projection_job_name(job_name)
        target = day.isoformat()
        stamp = utc_iso(_utc(now, "now"))
        stale_before = utc_iso(_utc(now, "now") - stale_after)
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO analytics_projection_jobs (
                        job_name, window_start, window_end, cursor, status, updated_at
                    ) VALUES (%s, %s, %s, NULL, 'pending', %s)
                    ON CONFLICT(job_name) DO NOTHING
                    """,
                    (name, target, target, stamp),
                )
                cur.execute(
                    """
                    UPDATE analytics_projection_jobs
                       SET status = 'running', window_end = %s, updated_at = %s
                     WHERE job_name = %s
                       AND (cursor IS NULL OR cursor < %s)
                       AND (
                            status IN ('pending', 'complete')
                            OR (status = 'running' AND updated_at < %s)
                       )
                    """,
                    (target, stamp, name, target, stale_before),
                )
                return cur.rowcount == 1

    def complete_projection_day(self, job_name: str, *, day: date, now: datetime) -> None:
        """See the SQLite twin."""
        name = _projection_job_name(job_name)
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE analytics_projection_jobs
                       SET cursor = %s, status = 'complete', updated_at = %s
                     WHERE job_name = %s
                    """,
                    (day.isoformat(), utc_iso(_utc(now, "now")), name),
                )

    def release_projection_day(self, job_name: str, *, now: datetime) -> None:
        """See the SQLite twin."""
        name = _projection_job_name(job_name)
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE analytics_projection_jobs
                       SET status = 'pending', updated_at = %s
                     WHERE job_name = %s AND status = 'running'
                    """,
                    (utc_iso(_utc(now, "now")), name),
                )
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_value_repository.py dashboard/backend/tests/test_store_twin_parity.py -v
TEST_POSTGRES_URL=postgresql://postgres:atl@localhost:55432/postgres \
  python -m pytest dashboard/backend/tests/domain/analytics/test_repository_postgres.py -q
```

Expected: PASS on both tiers.

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/domain/analytics/value_repository.py \
        dashboard/backend/domain/analytics/value_repository_postgres.py \
        dashboard/backend/tests/domain/analytics/test_value_repository.py
git status --short
git commit -m "feat: claim a projection day with a two-field compare-and-set lease"
```

---

### Task 9: The daily job

**Files:**
- Create: `dashboard/backend/domain/analytics/daily_facts.py`
- Modify: `dashboard/backend/domain/analytics/repository.py`, `repository_postgres.py` — add `list_daily_subjects` after `list_existing_source_event_ids` (Task 6)
- Modify: `dashboard/backend/domain/analytics/value_repository.py` — add models `DayEventTotals`, `ActivityUpdate`, `UserDailyFact`, `LifecycleTransitionRow`, `_fact_from_row`, `_EVENTS_FOR_DAY_SQL`, `_DAILY_FACT_COLUMNS`; add `aggregate_events_for_day`, `record_activity_batch`, `upsert_daily_facts`, `append_lifecycle_transitions`, `list_facts_for_date`, `list_days_needing_recompute` to `ValueAnalyticsStore` after `release_projection_day`; export the four models
- Modify: `dashboard/backend/domain/analytics/value_repository_postgres.py` — the same six methods on `PostgresValueAnalyticsStore`; imports
- Test: `dashboard/backend/tests/domain/analytics/test_daily_facts.py` (create)

**Interfaces:**
- Consumes: `claim_projection_day`/`complete_projection_day`/`release_projection_day` (Task 8), `list_operational_signals` (Task 7), `aggregate_ledger_for_day` and `aggregate_operator_cost_for_day` (Task 5), `build_lifecycle_inputs` and `sum_recent_facts` (Task 4), `record_activity`/`list_activity` (Task 3), `owner_user_id` (Task 2), the three tables (Task 1), `users.user_group` (PR #465), `rollup_day` (`rollups.py`, unchanged) and `analytics_retention_coordinator` (`retention.py`, unchanged).
- Produces:
  - `run_daily_facts(*, now: datetime | None = None, value_store=None, run_history_store=None, rollup=None, retention=None) -> DailyFactsReport` and `DailyFactsReport(snapshot_date: date | None, claimed: bool, users_written: int, transitions_written: int, partial: bool, failed_steps: tuple[str, ...], recomputed_dates: tuple[date, ...])`, `DAILY_FACTS_JOB = "analytics_daily_facts"` in `daily_facts.py`
  - `AnalyticsStore.list_daily_subjects() -> list[dict[str, Any]]` (both twins): `id`, `user_group`, `created_at` of every non-admin, non-excluded user
  - on both value twins: `aggregate_events_for_day(day) -> dict[int, DayEventTotals]`, `record_activity_batch(updates: Sequence[ActivityUpdate], *, now) -> int`, `upsert_daily_facts(rows: Sequence[UserDailyFact]) -> int`, `append_lifecycle_transitions(rows: Sequence[LifecycleTransitionRow]) -> int`, `list_facts_for_date(day) -> list[UserDailyFact]`, `list_days_needing_recompute(*, since: date, until: date) -> list[date]`

  Task 11's worker thread calls `run_daily_facts` every tick; Task 14 measures it.

Carried from the superseded 2026-09-12 plan's Task B8 (`git show c3bbf2ed:...`, lines 3061-3500) with these amendments: the two cross-domain aggregates are called on the owning stores (Task 5; §12 item 1); the day claim is Task 8's two-field lease, so a failed day is **released** rather than having its cursor rolled back; the fact row carries `user_group` (read from `users` at the write) and `operational_reason_code` (§6.7, §6.9 step 6); the population reads take no id list (Task 7); `record_activity` corrections are one `executemany` batch per source rather than a call per user, because a call per user is a query count that grows with the population (§6.12); the job is **not** registered on the reaper — Task 11 gives it its own thread (D23). `rollup_day` and the retention coordinator move here from `maintenance.py`/`app.py` so the two never double-write (§6.9 steps 1 and 8).

**Two properties decide whether this job is correct, and neither is obvious from the step list.** The day's evidence must be clamped to the end of D — `user_activity` stores no history, so feeding it to `calculate_lifecycle(as_of=<end of D>)` raises on the first user active after midnight (Task 4 clamps, this task threads `day_activity_at` through). And the day must stay recomputable after it is written, because events legitimately arrive after the aggregate is taken (`list_days_needing_recompute`, step 7). Read the two paragraphs marked **9a** and **9b** below before writing any of this.

The **query constant** this job commits to, per claimed tick, counted as public store-method calls (the unit Task 14's spies measure):

| Store | Calls | Methods |
|---|---|---|
| value store | 12 | `claim_projection_day`, `aggregate_events_for_day`, `record_activity_batch` (events), `record_activity_batch` (ledger), `list_operational_signals`, `list_activity`, `sum_recent_facts`, `upsert_daily_facts`, `list_facts_for_date`, `append_lifecycle_transitions`, `list_days_needing_recompute`, `complete_projection_day` |
| analytics base | 2 | `list_excluded_user_ids` (inside `rollup_day` → `AnalyticsRollupStore.list_events`), `list_daily_subjects` |
| credits | 3 | `aggregate_ledger_for_day`, `get_balance_projections`, `list_account_billing_states` |
| providers | 3 | `list_default_credential_facts`, `list_all_providers`, `list_platform_credential_statuses` |
| agents | 1 | `list_agent_owners` |
| protocol runs | 1 | `list_terminal_runs_since` |
| run history | 1 | `aggregate_operator_cost_for_day` |
| **total** | **23** | plus one full `compute_day` (22, no claim/complete/recompute-list) per late-arrival day, at most two per tick |

Every call is issued unconditionally — `record_activity_batch([])` is still a call that returns 0 — so the count is the same for an empty population and a full one. An idle tick (the day already complete) costs exactly one call: the refused `claim_projection_day`.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/domain/analytics/test_daily_facts.py`:

```python
"""One set-based pass per UTC day: correctness properties of the daily job."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, time, timedelta, timezone

import pytest
from cryptography.fernet import Fernet

from dashboard.backend.database import BacktestDatabase
from dashboard.backend.domain.agents.repository import AgentStore
from dashboard.backend.domain.analytics.daily_facts import (
    DAILY_FACTS_JOB,
    DailyFactsReport,
    run_daily_facts,
)
from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.service import AnalyticsService
from dashboard.backend.domain.analytics.value_repository import (
    UserDailyFact,
    build_value_analytics_store,
)
from dashboard.backend.domain.brokers import repository as broker_repository
from dashboard.backend.domain.credits.repository import CreditsStore
from dashboard.backend.domain.model_providers.repository import ModelProviderStore
from dashboard.backend.domain.runs.repository import RunStore
from dashboard.backend.users import UserStore


D = date(2026, 9, 11)
NOW = datetime(2026, 9, 12, 0, 5, tzinfo=timezone.utc)  # the tick after midnight
END_OF_D = datetime.combine(D, time(23, 59, 59, 999999), tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _encryption_key(monkeypatch):
    monkeypatch.setenv("BROKER_TOKEN_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setattr(broker_repository, "_fernet_instance", None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("COMMONSTACK_API_KEY", raising=False)


class _NoRetention:
    def __init__(self):
        self.calls = 0

    def run_if_due(self):
        self.calls += 1
        return None


class DailyFixture:
    def __init__(self, path):
        self.path = path / "daily.db"
        self.runs_path = path / "runs.db"
        UserStore(db_path=self.path)
        self.analytics = AnalyticsStore(self.path)
        self.credits = CreditsStore(self.path)
        self.providers = ModelProviderStore(self.path)
        self.agents = AgentStore(self.path)
        self.protocol_runs = RunStore(self.path)
        self.run_history = BacktestDatabase(self.runs_path)
        self.store = build_value_analytics_store(
            self.analytics,
            credits_base=self.credits,
            provider_base=self.providers,
            agent_base=self.agents,
            run_base=self.protocol_runs,
        )
        self.service = AnalyticsService(
            self.analytics, value_store=self.store, maintain_activity=True
        )
        self.retention = _NoRetention()
        self._next_id = 1
        self.admin_id = self.create_user_at(NOW - timedelta(days=90), role="admin")
        self.excluded_id = self.create_user_at(NOW - timedelta(days=90))
        self.analytics.set_subject_exclusion(
            self.excluded_id, excluded=True, actor_user_id=self.admin_id, reason="test"
        )
        self.active_id = self.create_user_at(NOW - timedelta(days=60), user_group="organic")
        self.at_risk_id = self.create_user_at(NOW - timedelta(days=60))
        self.created_today_id = None
        self._seed()

    # -- helpers -------------------------------------------------------------

    def create_user_at(self, when, *, role="user", user_group="unknown"):
        user_id = self._next_id
        self._next_id += 1
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT INTO users (id, email, display_name, password_hash, role, "
                "user_group, created_at) VALUES (?, ?, ?, 'x', ?, ?, ?)",
                (user_id, f"user{user_id}@example.test", f"User {user_id}", role,
                 user_group, when.isoformat()),
            )
        if when > END_OF_D:
            self.created_today_id = user_id
        return user_id

    def append_event(self, user_id, *, event_name, occurred_at, received_at, index=None):
        suffix = index if index is not None else occurred_at.isoformat()
        return self.service.record_server_event(
            event_name=event_name,
            user_id=user_id,
            source_event_id=f"run:{event_name}:{user_id}:{suffix}",
            source_record_type="run",
            source_record_id=f"run-{user_id}-{suffix}",
            occurred_at=occurred_at,
            received_at=received_at,
        )

    def touch_activity(self, user_id, *, at):
        self.store.record_activity(user_id, occurred_at=at, activating=False, now=at)

    def seed_previous_segment(self, segment, *, user_id=None):
        self.store.upsert_daily_facts(
            [
                UserDailyFact(
                    snapshot_date=D - timedelta(days=1),
                    user_id=user_id or self.at_risk_id,
                    lifecycle_segment=segment,
                    lifecycle_reason_code="growing_activated_below_core_threshold",
                    operational_state="healthy",
                    operational_reason_code=None,
                    tier="unpaid",
                    user_group="unknown",
                    active=False,
                    data_quality="complete",
                    calculated_at=NOW - timedelta(days=1),
                )
            ]
        )

    def seed_fact(self, day, user_id, *, active, runs_completed):
        self.store.upsert_daily_facts(
            [
                UserDailyFact(
                    snapshot_date=day,
                    user_id=user_id,
                    lifecycle_segment="growing",
                    lifecycle_reason_code="growing_activated_below_core_threshold",
                    operational_state="healthy",
                    operational_reason_code=None,
                    tier="unpaid",
                    user_group="unknown",
                    active=active,
                    runs_requested=runs_completed,
                    runs_completed=runs_completed,
                    data_quality="complete",
                    calculated_at=NOW - timedelta(days=1),
                )
            ]
        )

    def seed_two_of_three_before_d(self):
        user_id = self.create_user_at(NOW - timedelta(days=60))
        self.touch_activity(user_id, at=NOW - timedelta(days=20))
        self.store.record_activity(
            user_id,
            occurred_at=NOW - timedelta(days=20),
            activating=True,
            now=NOW - timedelta(days=20),
        )
        self.seed_fact(D - timedelta(days=5), user_id, active=True, runs_completed=1)
        self.seed_fact(D - timedelta(days=3), user_id, active=True, runs_completed=1)
        return user_id

    def seed_success_on(self, day, user_id):
        at = datetime.combine(day, time(15, 0), tzinfo=timezone.utc)
        self.append_event(
            user_id, event_name="backtest_completed", occurred_at=at,
            received_at=at + timedelta(seconds=1),
        )

    def break_step(self, name):
        assert name == "ledger", "only the ledger step is breakable in this fixture"

        def _broken(*_args, **_kwargs):
            raise RuntimeError("private ledger detail")

        self.credits.aggregate_ledger_for_day = _broken

    def repair_step(self, name):
        assert name == "ledger"
        del self.credits.aggregate_ledger_for_day

    def fact_for(self, user_id, day=D):
        rows = {row.user_id: row for row in self.store.list_facts_for_date(day)}
        return rows[user_id]

    def transitions_for(self, user_id):
        with self.analytics._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM lifecycle_transitions WHERE user_id = ? ORDER BY snapshot_date",
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def run(self, *, now=NOW):
        return run_daily_facts(
            now=now,
            value_store=self.store,
            run_history_store=self.run_history,
            retention=self.retention,
        )

    # -- seed --------------------------------------------------------------

    def _seed(self):
        active_day = datetime.combine(D, time(10, 0), tzinfo=timezone.utc)
        for index, name in enumerate(
            ("backtest_requested", "backtest_completed", "backtest_requested", "backtest_failed")
        ):
            self.append_event(
                self.active_id,
                event_name=name,
                occurred_at=active_day + timedelta(minutes=index),
                received_at=active_day + timedelta(minutes=index, seconds=1),
                index=index,
            )
        self.run_history.insert_run(
            run_id="run-active",
            session_id="session-active",
            agent_name="Agent",
            mode="backtest",
            start_date="2026-09-01",
            end_date="2026-09-02",
            initial_equity=100000.0,
            est_cost_usd=1.25,
            owner_user_id=self.active_id,
        )
        conn = self.run_history._get_connection()
        try:
            conn.execute(
                "UPDATE agent_runs SET updated_at = ? WHERE run_id = ?",
                (f"{D.isoformat()} 10:30:00", "run-active"),
            )
            conn.commit()
        finally:
            conn.close()
        # at_risk: activated long ago, last meaningful activity ten days before D.
        self.store.record_activity(
            self.at_risk_id,
            occurred_at=NOW - timedelta(days=40),
            activating=True,
            now=NOW - timedelta(days=40),
        )
        self.touch_activity(self.at_risk_id, at=datetime.combine(D - timedelta(days=10), time(9), tzinfo=timezone.utc))


@pytest.fixture
def daily_fixture(tmp_path):
    return DailyFixture(tmp_path)


def test_a_day_is_written_once_and_only_once(daily_fixture):
    """The second call in the same UTC day does nothing."""
    first = daily_fixture.run()
    second = daily_fixture.run()

    assert isinstance(first, DailyFactsReport)
    assert first.claimed is True
    assert first.snapshot_date == D
    assert second.claimed is False
    assert second.users_written == 0
    job = daily_fixture.store.get_projection_job(DAILY_FACTS_JOB)
    assert job.cursor == D.isoformat()
    assert job.status == "complete"


def test_admins_and_excluded_users_get_no_rows(daily_fixture):
    daily_fixture.run()
    written = {row.user_id for row in daily_fixture.store.list_facts_for_date(D)}

    assert daily_fixture.admin_id not in written
    assert daily_fixture.excluded_id not in written
    assert daily_fixture.active_id in written
    assert daily_fixture.at_risk_id in written


def test_run_outcomes_cost_group_and_states_land_on_the_row(daily_fixture):
    daily_fixture.run()
    row = daily_fixture.fact_for(daily_fixture.active_id)

    assert row.runs_requested == 2
    assert row.runs_completed == 1
    assert row.runs_failed == 1
    assert row.runs_cancelled == 0
    assert row.operator_cost_micro == 1_250_000
    assert row.own_spend_micro == 0
    assert row.active is True
    assert row.user_group == "organic"
    assert row.tier == "unpaid"
    assert row.lifecycle_segment == "growing"
    assert row.operational_state == "blocked"  # no balance, no BYOK credential
    assert row.operational_reason_code == "billing_lane_unavailable"
    assert row.data_quality == "complete"


def test_the_at_risk_user_reads_from_the_clock_not_the_events(daily_fixture):
    daily_fixture.run()
    row = daily_fixture.fact_for(daily_fixture.at_risk_id)

    assert row.active is False
    assert row.lifecycle_segment == "at_risk"
    assert row.lifecycle_reason_code == "at_risk_previously_activated"


def test_a_segment_change_appends_exactly_one_transition(daily_fixture):
    daily_fixture.seed_previous_segment("growing")
    daily_fixture.run()
    daily_fixture.run()

    transitions = daily_fixture.transitions_for(daily_fixture.at_risk_id)
    assert len(transitions) == 1
    assert transitions[0]["from_segment"] == "growing"
    assert transitions[0]["to_segment"] == "at_risk"
    assert transitions[0]["inactive_days"] == 10
    assert transitions[0]["data_quality"] == "complete"


def test_no_previous_row_means_no_transition(daily_fixture):
    daily_fixture.run()

    assert daily_fixture.transitions_for(daily_fixture.active_id) == []


def test_a_failed_step_marks_the_day_partial_and_retries_it(daily_fixture, capsys):
    daily_fixture.break_step("ledger")
    first = daily_fixture.run()
    printed = capsys.readouterr().out

    assert first.claimed is True
    assert first.partial is True
    assert "ledger" in first.failed_steps
    assert "WARNING: analytics.daily_facts.ledger_failed category=RuntimeError" in printed
    assert "private ledger detail" not in printed
    assert daily_fixture.fact_for(daily_fixture.active_id).data_quality == "partial"
    job = daily_fixture.store.get_projection_job(DAILY_FACTS_JOB)
    assert job.status == "pending"  # released, not completed
    assert job.cursor is None

    daily_fixture.repair_step("ledger")
    retry = daily_fixture.run(now=NOW + timedelta(minutes=5))
    assert retry.claimed is True
    assert retry.partial is False
    assert daily_fixture.fact_for(daily_fixture.active_id).data_quality == "complete"


def test_a_retry_corrects_the_transition_the_partial_day_wrote(daily_fixture):
    """DO UPDATE, not DO NOTHING.

    The day a transition is rewritten is the day the first attempt was
    wrong. Freezing it leaves `lifecycle_transitions` permanently
    disagreeing with the `user_daily_facts` row the retry did correct.
    """
    daily_fixture.seed_previous_segment("growing")
    daily_fixture.break_step("ledger")
    daily_fixture.run()
    first = daily_fixture.transitions_for(daily_fixture.at_risk_id)[0]

    daily_fixture.repair_step("ledger")
    daily_fixture.run(now=NOW + timedelta(minutes=5))
    corrected = daily_fixture.transitions_for(daily_fixture.at_risk_id)

    assert len(corrected) == 1
    assert first["data_quality"] == "partial"
    assert corrected[0]["data_quality"] == "complete"


def test_a_user_active_after_midnight_does_not_abort_the_day(daily_fixture):
    """The regression that would have fired on the first night in prod.

    `user_activity` is overwritten in place, so a user who acts at 00:02
    has a `last_meaningful_activity_at` newer than the end of D. Fed to
    `calculate_lifecycle(as_of=<end of D>)` unclamped that raises, and the
    single try/except around the step turns one such user into zero fact
    rows for the entire population.
    """
    daily_fixture.touch_activity(daily_fixture.active_id, at=NOW)
    daily_fixture.create_user_at(NOW)

    report = daily_fixture.run()

    assert report.partial is False
    assert report.failed_steps == ()
    assert report.users_written >= 2
    written = {row.user_id for row in daily_fixture.store.list_facts_for_date(D)}
    assert daily_fixture.active_id in written
    # Created after D ended: no day D to describe.
    assert daily_fixture.created_today_id not in written


def test_the_stored_segment_counts_its_own_day(daily_fixture):
    """The trailing window is 30 dates ending at D, D included.

    Sitting at two active days and two successes before D, plus an active
    day with a success on D, is the `core` threshold exactly. Summing only
    D-29..D-1 writes `growing` and the read path answers `core` the next
    morning.
    """
    user_id = daily_fixture.seed_two_of_three_before_d()
    daily_fixture.seed_success_on(D, user_id)

    daily_fixture.run()

    assert daily_fixture.fact_for(user_id).lifecycle_segment == "core"


def test_an_event_arriving_after_the_day_was_written_is_picked_up(daily_fixture):
    """Late arrivals are recomputed, not lost.

    A run finishing at 23:59 can be appended minutes later, and the
    frontend route accepts `occurred_at` up to 24 hours old. Both land
    inside D after D's aggregate was taken, and the cursor has already
    moved past D. The next day's job finds them by `received_at`.
    """
    daily_fixture.run()
    before = daily_fixture.fact_for(daily_fixture.active_id).runs_completed

    daily_fixture.append_event(
        daily_fixture.active_id,
        event_name="backtest_completed",
        occurred_at=datetime(2026, 9, 11, 23, 59, tzinfo=timezone.utc),
        received_at=NOW + timedelta(minutes=5),
        index="late",
    )
    report = daily_fixture.run(now=NOW + timedelta(days=1))

    assert report.snapshot_date == D + timedelta(days=1)
    assert D in report.recomputed_dates
    assert daily_fixture.fact_for(daily_fixture.active_id).runs_completed == before + 1


def test_a_settled_day_is_not_recomputed_forever(daily_fixture):
    """The sweep is driven by evidence, not by a timer."""
    daily_fixture.run()
    report = daily_fixture.run(now=NOW + timedelta(days=1))

    assert report.recomputed_dates == ()


def test_the_retention_coordinator_runs_on_every_tick(daily_fixture):
    daily_fixture.run()
    daily_fixture.run(now=NOW + timedelta(minutes=5))  # idle tick

    assert daily_fixture.retention.calls == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_daily_facts.py -v
```

Expected: FAIL at import — `ModuleNotFoundError: No module named 'dashboard.backend.domain.analytics.daily_facts'`.

- [ ] **Step 3: Add the eligible-subjects read to both analytics twins**

`dashboard/backend/domain/analytics/repository.py`, on `AnalyticsStore` after `list_existing_source_event_ids`:

```python
    def list_daily_subjects(self) -> list[dict[str, Any]]:
        """Every non-admin, non-excluded account: id, user_group, created_at.

        The one place the daily job materialises the whole user list -- a
        few hundred rows of three columns. The predicate is the one
        ``list_stale_user_ids`` and the lifecycle backfill already use for
        aggregate exclusion (design SS6.7). ``created_at`` comes back as the
        raw text the users table holds (CURRENT_TIMESTAMP text on SQLite,
        ISO-8601 on Postgres); the caller parses.
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT users.id, users.user_group, users.created_at
                FROM users
                LEFT JOIN analytics_subject_settings AS settings
                  ON settings.user_id = users.id
                WHERE users.role <> 'admin'
                  AND COALESCE(settings.excluded, 0) = 0
                ORDER BY users.id
                """
            ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "user_group": row["user_group"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
```

`dashboard/backend/domain/analytics/repository_postgres.py`, on `PostgresAnalyticsStore` after `list_existing_source_event_ids`:

```python
    def list_daily_subjects(self) -> list[dict[str, Any]]:
        """See the SQLite twin."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT users.id, users.user_group, users.created_at
                    FROM users
                    LEFT JOIN analytics_subject_settings AS settings
                      ON settings.user_id = users.id
                    WHERE users.role <> 'admin'
                      AND COALESCE(settings.excluded, FALSE) = FALSE
                    ORDER BY users.id
                    """
                )
                rows = cur.fetchall()
        return [
            {
                "id": int(row["id"]),
                "user_group": row["user_group"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
```

- [ ] **Step 4: Add the fact models and the six set-based methods to both value twins**

In `dashboard/backend/domain/analytics/value_repository.py`, add `_LIFECYCLE_ACTIVITY_EVENTS,` and `OperationalState,` (already imported) / `LifecycleSegment` (already imported) to the `from .lifecycle import (...)` block — `_LIFECYCLE_ACTIVITY_EVENTS` is the new name — then immediately after `_recent_totals_from_row` insert:

```python
class DayEventTotals(BaseModel):
    """One user's run outcomes and activity marks inside one UTC day."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: int = Field(gt=0)
    runs_requested: int = Field(default=0, ge=0)
    runs_completed: int = Field(default=0, ge=0)
    runs_failed: int = Field(default=0, ge=0)
    runs_cancelled: int = Field(default=0, ge=0)
    first_success_at: datetime | None = None
    last_activity_at: datetime | None = None

    @property
    def active(self) -> bool:
        return self.last_activity_at is not None


class ActivityUpdate(BaseModel):
    """One row of a batched ``record_activity``; either timestamp may be absent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: int = Field(gt=0)
    activated_at: datetime | None = None
    last_activity_at: datetime | None = None


class UserDailyFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot_date: date
    user_id: int = Field(gt=0)
    lifecycle_segment: LifecycleSegment
    lifecycle_reason_code: str = Field(min_length=1, max_length=100)
    operational_state: OperationalState
    operational_reason_code: str | None = Field(default=None, max_length=100)
    tier: CommercialTier
    user_group: str = Field(min_length=1, max_length=32)
    active: bool = False
    runs_requested: int = Field(default=0, ge=0)
    runs_completed: int = Field(default=0, ge=0)
    runs_failed: int = Field(default=0, ge=0)
    runs_cancelled: int = Field(default=0, ge=0)
    operator_cost_micro: int = Field(default=0, ge=0)
    own_spend_micro: int = Field(default=0, ge=0)
    data_quality: Literal["complete", "partial"]
    calculated_at: datetime

    @field_validator("calculated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return _utc(value)


class LifecycleTransitionRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: int = Field(gt=0)
    snapshot_date: date
    from_segment: LifecycleSegment
    to_segment: LifecycleSegment
    inactive_days: int = Field(default=0, ge=0)
    data_quality: Literal["complete", "partial"] = "complete"
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return _utc(value)


_DAILY_FACT_COLUMNS = (
    "snapshot_date", "user_id", "lifecycle_segment", "lifecycle_reason_code",
    "operational_state", "operational_reason_code", "tier", "user_group", "active",
    "runs_requested", "runs_completed", "runs_failed", "runs_cancelled",
    "operator_cost_micro", "own_spend_micro", "data_quality", "calculated_at",
)
_DAILY_FACT_UPDATES = ", ".join(
    f"{column} = excluded.{column}" for column in _DAILY_FACT_COLUMNS[2:]
)
_TRANSITION_COLUMNS = (
    "user_id", "snapshot_date", "from_segment", "to_segment", "inactive_days",
    "data_quality", "created_at",
)
_TRANSITION_UPDATES = ", ".join(
    f"{column} = excluded.{column}" for column in _TRANSITION_COLUMNS[2:]
)
_ACTIVITY_EVENT_NAMES = tuple(sorted(_LIFECYCLE_ACTIVITY_EVENTS))
_EVENTS_FOR_DAY_SQL = """
    SELECT user_id,
           SUM(CASE WHEN event_name = 'backtest_requested' THEN 1 ELSE 0 END)
               AS runs_requested,
           SUM(CASE WHEN event_name = 'backtest_completed' THEN 1 ELSE 0 END)
               AS runs_completed,
           SUM(CASE WHEN event_name = 'backtest_failed' THEN 1 ELSE 0 END)
               AS runs_failed,
           SUM(CASE WHEN event_name = 'backtest_cancelled' THEN 1 ELSE 0 END)
               AS runs_cancelled,
           MIN(CASE WHEN event_name = 'backtest_completed' THEN occurred_at END)
               AS first_success_at,
           MAX(CASE WHEN event_name IN ({activity}) THEN occurred_at END)
               AS last_activity_at
    FROM analytics_events
    WHERE occurred_at >= {p} AND occurred_at < {p}
    GROUP BY user_id
"""
_RECOMPUTE_EVENTS_SQL = """
    SELECT substr(occurred_at, 1, 10) AS day, MAX(received_at) AS last_received
    FROM analytics_events
    WHERE occurred_at >= {p} AND occurred_at < {p}
    GROUP BY substr(occurred_at, 1, 10)
"""
_RECOMPUTE_FACTS_SQL = """
    SELECT snapshot_date, MIN(calculated_at) AS calculated_at
    FROM user_daily_facts
    WHERE snapshot_date >= {p} AND snapshot_date <= {p}
    GROUP BY snapshot_date
"""


def _day_bounds_iso(day: date) -> tuple[str, str]:
    start = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
    return utc_iso(start), utc_iso(start + timedelta(days=1))


def _events_totals_from_row(row: Any) -> DayEventTotals:
    """Shared by both twins."""
    return DayEventTotals(
        user_id=int(_row_value(row, "user_id")),
        runs_requested=int(_row_value(row, "runs_requested", 0) or 0),
        runs_completed=int(_row_value(row, "runs_completed", 0) or 0),
        runs_failed=int(_row_value(row, "runs_failed", 0) or 0),
        runs_cancelled=int(_row_value(row, "runs_cancelled", 0) or 0),
        first_success_at=_optional_timestamp(_row_value(row, "first_success_at")),
        last_activity_at=_optional_timestamp(_row_value(row, "last_activity_at")),
    )


def _fact_from_row(row: Any) -> UserDailyFact:
    """Shared by both twins."""
    return UserDailyFact(
        snapshot_date=date.fromisoformat(str(_row_value(row, "snapshot_date"))),
        user_id=int(_row_value(row, "user_id")),
        lifecycle_segment=_row_value(row, "lifecycle_segment"),
        lifecycle_reason_code=_row_value(row, "lifecycle_reason_code"),
        operational_state=_row_value(row, "operational_state"),
        operational_reason_code=_row_value(row, "operational_reason_code"),
        tier=_row_value(row, "tier"),
        user_group=_row_value(row, "user_group"),
        active=bool(_row_value(row, "active", 0)),
        runs_requested=int(_row_value(row, "runs_requested", 0) or 0),
        runs_completed=int(_row_value(row, "runs_completed", 0) or 0),
        runs_failed=int(_row_value(row, "runs_failed", 0) or 0),
        runs_cancelled=int(_row_value(row, "runs_cancelled", 0) or 0),
        operator_cost_micro=int(_row_value(row, "operator_cost_micro", 0) or 0),
        own_spend_micro=int(_row_value(row, "own_spend_micro", 0) or 0),
        data_quality=_row_value(row, "data_quality"),
        calculated_at=_timestamp(_row_value(row, "calculated_at")),
    )


def _fact_values(row: UserDailyFact, *, active_value: Any) -> tuple[Any, ...]:
    """Shared by both twins; ``active_value`` is int on SQLite, bool on Postgres."""
    return (
        row.snapshot_date.isoformat(),
        row.user_id,
        row.lifecycle_segment,
        row.lifecycle_reason_code,
        row.operational_state,
        row.operational_reason_code,
        row.tier,
        row.user_group,
        active_value,
        row.runs_requested,
        row.runs_completed,
        row.runs_failed,
        row.runs_cancelled,
        row.operator_cost_micro,
        row.own_spend_micro,
        row.data_quality,
        utc_iso(row.calculated_at),
    )


def _transition_values(row: LifecycleTransitionRow) -> tuple[Any, ...]:
    return (
        row.user_id,
        row.snapshot_date.isoformat(),
        row.from_segment,
        row.to_segment,
        row.inactive_days,
        row.data_quality,
        utc_iso(row.created_at),
    )


def _activity_update_values(update: ActivityUpdate, stamp: str) -> tuple[Any, ...]:
    return (
        update.user_id,
        utc_iso(update.activated_at) if update.activated_at is not None else None,
        utc_iso(update.last_activity_at) if update.last_activity_at is not None else None,
        stamp,
    )


def _recompute_days(event_rows: Sequence[Any], fact_rows: Sequence[Any]) -> list[date]:
    """Shared by both twins: days whose newest event landed after their facts."""
    calculated = {
        str(_row_value(row, "snapshot_date")): str(_row_value(row, "calculated_at"))
        for row in fact_rows
    }
    stale: list[date] = []
    for row in event_rows:
        day = str(_row_value(row, "day"))
        last_received = _row_value(row, "last_received")
        if day in calculated and last_received is not None and str(last_received) > calculated[day]:
            stale.append(date.fromisoformat(day))
    return sorted(stale, reverse=True)
```

The `MAX(received_at) > MIN(calculated_at)` comparison is text on ISO-8601 UTC, which orders like the instants. `received_at` is the right witness because it is the one timestamp the writer cannot backdate.

On `ValueAnalyticsStore`, immediately after `release_projection_day`, insert:

```python
    def aggregate_events_for_day(self, day: date) -> dict[int, DayEventTotals]:
        """Per-user outcome counts and activity marks for one UTC day.

        The one-day scan the event-log discipline permits (design SS6.11 rule
        4); it takes no user id. The fifteen activity names are bound as
        parameters rather than interpolated.
        """
        start, end = _day_bounds_iso(day)
        activity = ", ".join("?" for _ in _ACTIVITY_EVENT_NAMES)
        sql = _EVENTS_FOR_DAY_SQL.format(activity=activity, p="?")
        with self._analytics_connection() as conn:
            rows = conn.execute(sql, [*_ACTIVITY_EVENT_NAMES, start, end]).fetchall()
        return {
            int(_row_value(row, "user_id")): _events_totals_from_row(row) for row in rows
        }

    def record_activity_batch(
        self,
        updates: Sequence[ActivityUpdate],
        *,
        now: datetime,
    ) -> int:
        """Many ``record_activity`` corrections in one ``executemany``.

        Same MIN/MAX upsert as ``record_activity`` and ``seed_activity_from_snapshots``,
        with both incoming columns nullable: the daily job's ledger step knows
        a user's last credit activity but nothing about activation, and its
        events step knows both. A per-user ``record_activity`` loop would be a
        query count that grows with the population (design SS6.12). Returns
        the number of updates submitted; an empty batch touches nothing.
        """
        if not updates:
            return 0
        stamp = utc_iso(_utc(now, "now"))
        sql = """
            INSERT INTO user_activity (
                user_id, activated_at, last_meaningful_activity_at, updated_at
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                activated_at = MIN(
                    COALESCE(user_activity.activated_at, excluded.activated_at),
                    COALESCE(excluded.activated_at, user_activity.activated_at)
                ),
                last_meaningful_activity_at = MAX(
                    COALESCE(
                        user_activity.last_meaningful_activity_at,
                        excluded.last_meaningful_activity_at
                    ),
                    COALESCE(
                        excluded.last_meaningful_activity_at,
                        user_activity.last_meaningful_activity_at
                    )
                ),
                updated_at = excluded.updated_at
        """
        with self._analytics_connection() as conn:
            conn.executemany(
                sql, [_activity_update_values(update, stamp) for update in updates]
            )
        return len(updates)

    def upsert_daily_facts(self, rows: Sequence[UserDailyFact]) -> int:
        """Write one batch of fact rows. One executemany, not a loop."""
        if not rows:
            return 0
        columns = ", ".join(_DAILY_FACT_COLUMNS)
        placeholders = ", ".join("?" for _ in _DAILY_FACT_COLUMNS)
        sql = f"""
            INSERT INTO user_daily_facts ({columns})
            VALUES ({placeholders})
            ON CONFLICT(snapshot_date, user_id) DO UPDATE SET {_DAILY_FACT_UPDATES}
        """
        with self._analytics_connection() as conn:
            conn.executemany(
                sql, [_fact_values(row, active_value=int(row.active)) for row in rows]
            )
        return len(rows)

    def append_lifecycle_transitions(
        self, rows: Sequence[LifecycleTransitionRow]
    ) -> int:
        """Write segment changes for one day, correcting any already there.

        ``ON CONFLICT (user_id, snapshot_date) DO UPDATE`` -- **not** DO
        NOTHING. The day this table is rewritten is exactly the day something
        went wrong: a step failed and the row was derived from a missing
        source, or a late event changed the day's totals. DO NOTHING would
        freeze that first, weakest answer forever while the retry corrected
        ``user_daily_facts`` (which upserts), leaving the two tables in
        permanent disagreement (design SS6.8). The UNIQUE constraint's job is
        to stop a *duplicate*, which DO UPDATE does equally well.
        """
        if not rows:
            return 0
        columns = ", ".join(_TRANSITION_COLUMNS)
        placeholders = ", ".join("?" for _ in _TRANSITION_COLUMNS)
        sql = f"""
            INSERT INTO lifecycle_transitions ({columns})
            VALUES ({placeholders})
            ON CONFLICT(user_id, snapshot_date) DO UPDATE SET {_TRANSITION_UPDATES}
        """
        with self._analytics_connection() as conn:
            conn.executemany(sql, [_transition_values(row) for row in rows])
        return len(rows)

    def list_facts_for_date(self, day: date) -> list[UserDailyFact]:
        """Every fact row for one date. Used for the previous day's segments."""
        with self._analytics_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM user_daily_facts WHERE snapshot_date = ? ORDER BY user_id",
                (day.isoformat(),),
            ).fetchall()
        return [_fact_from_row(row) for row in rows]

    def list_days_needing_recompute(self, *, since: date, until: date) -> list[date]:
        """Past days whose events arrived after their facts were computed.

        Two statements, neither parameterised by a user: events grouped by
        UTC day with ``MAX(received_at)``, and facts grouped by date with
        ``MIN(calculated_at)``; a day is returned, newest first, when the
        former exceeds the latter. This exists because
        ``aggregate_events_for_day`` filters on ``occurred_at`` while the job
        runs minutes after midnight: the frontend route accepts ``occurred_at``
        up to 24 hours old (``service.py:96-97``), and a server event for a run
        that finished at 23:59 can be appended after the aggregate was taken.
        Without this sweep every such row would be silently dropped from
        ``active``, DAU and the run counts, permanently and with no signal.
        """
        if until < since:
            raise ValueError("until must not precede since")
        start, _unused = _day_bounds_iso(since)
        _unused, end = _day_bounds_iso(until)
        with self._analytics_connection() as conn:
            event_rows = conn.execute(
                _RECOMPUTE_EVENTS_SQL.format(p="?"), (start, end)
            ).fetchall()
            fact_rows = conn.execute(
                _RECOMPUTE_FACTS_SQL.format(p="?"),
                (since.isoformat(), until.isoformat()),
            ).fetchall()
        return _recompute_days(event_rows, fact_rows)
```

Add `"ActivityUpdate", "DayEventTotals", "LifecycleTransitionRow", "UserDailyFact",` to `__all__` in alphabetical position.

In `value_repository_postgres.py`, extend the import block with `ActivityUpdate, DayEventTotals, LifecycleTransitionRow, UserDailyFact, _ACTIVITY_EVENT_NAMES, _DAILY_FACT_COLUMNS, _DAILY_FACT_UPDATES, _EVENTS_FOR_DAY_SQL, _RECOMPUTE_EVENTS_SQL, _RECOMPUTE_FACTS_SQL, _TRANSITION_COLUMNS, _TRANSITION_UPDATES, _activity_update_values, _day_bounds_iso, _events_totals_from_row, _fact_from_row, _fact_values, _recompute_days, _transition_values` and on `PostgresValueAnalyticsStore`, immediately after `release_projection_day`, insert:

```python
    def aggregate_events_for_day(self, day: date) -> dict[int, DayEventTotals]:
        """See the SQLite twin."""
        start, end = _day_bounds_iso(day)
        activity = ", ".join("%s" for _ in _ACTIVITY_EVENT_NAMES)
        sql = _EVENTS_FOR_DAY_SQL.format(activity=activity, p="%s")
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, [*_ACTIVITY_EVENT_NAMES, start, end])
                rows = cur.fetchall()
        return {
            int(_row_value(row, "user_id")): _events_totals_from_row(row) for row in rows
        }

    def record_activity_batch(
        self,
        updates: Sequence[ActivityUpdate],
        *,
        now: datetime,
    ) -> int:
        """See the SQLite twin."""
        if not updates:
            return 0
        stamp = utc_iso(_utc(now, "now"))
        sql = """
            INSERT INTO user_activity (
                user_id, activated_at, last_meaningful_activity_at, updated_at
            ) VALUES (%s, %s::text, %s::text, %s)
            ON CONFLICT(user_id) DO UPDATE SET
                activated_at = LEAST(
                    COALESCE(user_activity.activated_at, EXCLUDED.activated_at),
                    COALESCE(EXCLUDED.activated_at, user_activity.activated_at)
                ),
                last_meaningful_activity_at = GREATEST(
                    COALESCE(
                        user_activity.last_meaningful_activity_at,
                        EXCLUDED.last_meaningful_activity_at
                    ),
                    COALESCE(
                        EXCLUDED.last_meaningful_activity_at,
                        user_activity.last_meaningful_activity_at
                    )
                ),
                updated_at = EXCLUDED.updated_at
        """
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    sql, [_activity_update_values(update, stamp) for update in updates]
                )
        return len(updates)

    def upsert_daily_facts(self, rows: Sequence[UserDailyFact]) -> int:
        """See the SQLite twin. ``active`` is BOOLEAN here."""
        if not rows:
            return 0
        columns = ", ".join(_DAILY_FACT_COLUMNS)
        placeholders = ", ".join("%s" for _ in _DAILY_FACT_COLUMNS)
        sql = f"""
            INSERT INTO user_daily_facts ({columns})
            VALUES ({placeholders})
            ON CONFLICT(snapshot_date, user_id) DO UPDATE SET {_DAILY_FACT_UPDATES}
        """
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    sql, [_fact_values(row, active_value=bool(row.active)) for row in rows]
                )
        return len(rows)

    def append_lifecycle_transitions(
        self, rows: Sequence[LifecycleTransitionRow]
    ) -> int:
        """See the SQLite twin."""
        if not rows:
            return 0
        columns = ", ".join(_TRANSITION_COLUMNS)
        placeholders = ", ".join("%s" for _ in _TRANSITION_COLUMNS)
        sql = f"""
            INSERT INTO lifecycle_transitions ({columns})
            VALUES ({placeholders})
            ON CONFLICT(user_id, snapshot_date) DO UPDATE SET {_TRANSITION_UPDATES}
        """
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, [_transition_values(row) for row in rows])
        return len(rows)

    def list_facts_for_date(self, day: date) -> list[UserDailyFact]:
        """See the SQLite twin."""
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM user_daily_facts WHERE snapshot_date = %s ORDER BY user_id",
                    (day.isoformat(),),
                )
                rows = cur.fetchall()
        return [_fact_from_row(row) for row in rows]

    def list_days_needing_recompute(self, *, since: date, until: date) -> list[date]:
        """See the SQLite twin."""
        if until < since:
            raise ValueError("until must not precede since")
        start, _unused = _day_bounds_iso(since)
        _unused, end = _day_bounds_iso(until)
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(_RECOMPUTE_EVENTS_SQL.format(p="%s"), (start, end))
                event_rows = cur.fetchall()
                cur.execute(
                    _RECOMPUTE_FACTS_SQL.format(p="%s"),
                    (since.isoformat(), until.isoformat()),
                )
                fact_rows = cur.fetchall()
        return _recompute_days(event_rows, fact_rows)
```

`excluded.<column>` in the `DO UPDATE SET` lists is lower-case on purpose: Postgres folds the unquoted identifier and SQLite accepts either, so one `_DAILY_FACT_UPDATES` string serves both twins — exactly as `save_projection_job` already does.

- [ ] **Step 5: Write the job**

Create `dashboard/backend/domain/analytics/daily_facts.py`:

```python
"""One set-based pass per UTC day over the whole user population.

Every query here is parameterised by a date, never by a user id. That is the
property the read-budget test enforces, and it is the difference between this
module and the snapshot sweep it replaces: cost grows with days, not with
users times minutes (design SS6.9, SS6.12).

Scheduling lives in ``daily_job.py`` (its own worker thread, D23). This module
is the tick: claim yesterday, compute it, release or complete it, recompute any
recent day whose evidence changed, and give the retention coordinator its turn.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from dashboard.backend.domain.user_groups import coerce_user_group

from .lifecycle import (
    OperationalSignals,
    calculate_lifecycle,
    calculate_operational_state,
    commercial_tier,
)
from .lifecycle_reads import build_lifecycle_inputs
from .value_repository import (
    ActivityUpdate,
    LifecycleTransitionRow,
    RecentFactTotals,
    UserDailyFact,
    _timestamp,
)


DAILY_FACTS_JOB = "analytics_daily_facts"
RECOMPUTE_LOOKBACK_DAYS = 6
MAX_RECOMPUTES_PER_TICK = 2
TRAILING_WINDOW_DAYS = 30


class DailyFactsReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot_date: date | None = None
    claimed: bool = False
    users_written: int = Field(default=0, ge=0)
    transitions_written: int = Field(default=0, ge=0)
    partial: bool = False
    failed_steps: tuple[str, ...] = ()
    # Earlier days this tick recomputed because late events landed in them.
    # Reported rather than silent: a day that keeps reappearing here is a
    # clock-skew or a replay problem, and the only place it is visible.
    recomputed_dates: tuple[date, ...] = ()


class _DayOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    users_written: int = 0
    transitions_written: int = 0
    failed_steps: tuple[str, ...] = ()


def _utc_now(value: datetime | None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("now must include a timezone")
    return current.astimezone(timezone.utc)


def _default_value_store():
    from .repository import analytics_store
    from .value_repository import build_value_analytics_store

    return build_value_analytics_store(analytics_store)


def _default_run_history_store():
    from dashboard.backend.database import db

    return db


def _default_rollup() -> Callable[..., Any]:
    from .rollups import rollup_day

    return rollup_day


def _default_retention():
    from .retention import analytics_retention_coordinator

    return analytics_retention_coordinator


def _end_of_day(day: date) -> datetime:
    return datetime.combine(day, time(23, 59, 59, 999999), tzinfo=timezone.utc)


def _midnight(day: date) -> datetime:
    return datetime.combine(day, time.min, tzinfo=timezone.utc)


def _count(totals: Any, field: str) -> int:
    return int(getattr(totals, field)) if totals is not None else 0


def _compute_day(
    day: date,
    *,
    now: datetime,
    store: Any,
    run_history_store: Any,
    rollup: Callable[..., Any],
) -> _DayOutcome:
    """Steps 1-6 of design SS6.9 for one day, each wrapped individually.

    Step order differs from the design's numbering in one place: the eligible
    population is read first, because the operational step (5) answers for
    exactly those ids. Query count is unchanged.
    """
    from .rollups import AnalyticsRollupStore

    analytics = store.analytics_base
    end_of_day = _end_of_day(day)
    failed: list[str] = []

    def step(name: str, action: Callable[[], Any], default: Any) -> Any:
        try:
            return action()
        except Exception as exc:
            failed.append(name)
            print(
                f"WARNING: analytics.daily_facts.{name}_failed "
                f"category={type(exc).__name__[:80]}"
            )
            return default

    # 0. Who has a day D at all: every non-admin, non-excluded account created
    #    on or before the end of D (9a: an account created after D has no D).
    def subjects_step() -> list[dict[str, Any]]:
        eligible = []
        for subject in analytics.list_daily_subjects():
            created_at = _timestamp(subject["created_at"])
            if created_at <= end_of_day:
                eligible.append({**subject, "created_at": created_at})
        return eligible

    subjects = step("subjects", subjects_step, None)
    if subjects is None:
        # Nothing can be written without the population; every later step
        # would only mark a day partial that has no rows to be partial.
        return _DayOutcome(failed_steps=tuple(failed))
    eligible_ids = [int(subject["id"]) for subject in subjects]

    # 1. Anonymous rollups for D (moved here from run_analytics_maintenance).
    step(
        "rollup",
        lambda: rollup(day, store=AnalyticsRollupStore(analytics), now=_midnight(now.date())),
        None,
    )

    # 2. One one-day event scan, then the idempotent activity correction.
    def events_step() -> dict[int, Any]:
        totals = store.aggregate_events_for_day(day)
        store.record_activity_batch(
            [
                ActivityUpdate(
                    user_id=item.user_id,
                    activated_at=item.first_success_at,
                    last_activity_at=item.last_activity_at,
                )
                for item in totals.values()
            ],
            now=now,
        )
        return totals

    events = step("events", events_step, {})

    # 3. Operator-funded cost from the run-history store.
    operator_cost = step(
        "runs", lambda: run_history_store.aggregate_operator_cost_for_day(day), {}
    )

    # 4. Own spend and lifetime net purchases from the credits store, then the
    #    same activity correction from the ledger's own timestamps.
    def ledger_step() -> dict[int, dict[str, Any]]:
        totals = store.credits_base.aggregate_ledger_for_day(day)
        store.record_activity_batch(
            [
                ActivityUpdate(
                    user_id=user_id,
                    last_activity_at=_timestamp(row["last_activity_at"]),
                )
                for user_id, row in totals.items()
                if row.get("last_activity_at")
            ],
            now=now,
        )
        return totals

    ledger = step("ledger", ledger_step, {})

    # 5. Operational signals for the eligible population, as of the end of D,
    #    read population-wide so no statement carries an IN list.
    signals = step(
        "operational",
        lambda: store.list_operational_signals(
            eligible_ids, now=end_of_day, population_wide=True
        ),
        {},
    )

    quality = "partial" if failed else "complete"

    # 6. The fact rows, then the transitions.
    def facts_step() -> tuple[int, list[tuple[UserDailyFact, int]]]:
        activity = store.list_activity(None)
        window = store.sum_recent_facts(
            None,
            start=day - timedelta(days=TRAILING_WINDOW_DAYS - 1),
            end=day - timedelta(days=1),
        )
        rows_with_days: list[tuple[UserDailyFact, int]] = []
        for subject in subjects:
            user_id = int(subject["id"])
            day_totals = events.get(user_id)
            active = day_totals is not None and day_totals.active
            prior = window.get(user_id, RecentFactTotals())
            # 9b: the window is D-29..D-1 in the table; D's own contribution
            # comes from the events aggregate already in hand, so the stored
            # segment equals what the read path computes tomorrow.
            combined = RecentFactTotals(
                active_days=prior.active_days + (1 if active else 0),
                successful_backtests=prior.successful_backtests
                + _count(day_totals, "runs_completed"),
                runs_requested=prior.runs_requested + _count(day_totals, "runs_requested"),
                runs_completed=prior.runs_completed + _count(day_totals, "runs_completed"),
                runs_failed=prior.runs_failed + _count(day_totals, "runs_failed"),
                runs_cancelled=prior.runs_cancelled + _count(day_totals, "runs_cancelled"),
                operator_cost_micro=prior.operator_cost_micro,
                own_spend_micro=prior.own_spend_micro,
                days_present=prior.days_present + 1,
                last_active_date=day if active else prior.last_active_date,
            )
            # 9a: evidence is clamped to the end of D inside build_lifecycle_inputs;
            # the day's own last activity is threaded in so a user whose only
            # recent activity is on D is not read as dormant.
            inputs = build_lifecycle_inputs(
                user_id,
                created_at=subject["created_at"],
                activity=activity.get(user_id),
                totals=combined,
                as_of=end_of_day,
                day_activity_at=day_totals.last_activity_at if day_totals is not None else None,
            )
            lifecycle = calculate_lifecycle(inputs, end_of_day)
            operational = calculate_operational_state(
                signals.get(user_id) or OperationalSignals(user_id=user_id), end_of_day
            )
            ledger_row = ledger.get(user_id, {})
            rows_with_days.append(
                (
                    UserDailyFact(
                        snapshot_date=day,
                        user_id=user_id,
                        lifecycle_segment=lifecycle.segment,
                        lifecycle_reason_code=lifecycle.reason_code,
                        operational_state=operational.state,
                        operational_reason_code=(
                            None if operational.state == "healthy" else operational.reason_code
                        ),
                        tier=commercial_tier(
                            int(ledger_row.get("lifetime_net_purchased_micro", 0))
                        ),
                        user_group=coerce_user_group(subject.get("user_group")),
                        active=bool(active),
                        runs_requested=_count(day_totals, "runs_requested"),
                        runs_completed=_count(day_totals, "runs_completed"),
                        runs_failed=_count(day_totals, "runs_failed"),
                        runs_cancelled=_count(day_totals, "runs_cancelled"),
                        operator_cost_micro=int(operator_cost.get(user_id, 0)),
                        own_spend_micro=int(ledger_row.get("own_spend_micro", 0)),
                        data_quality=quality,
                        calculated_at=now,
                    ),
                    lifecycle.inactive_days,
                )
            )
        written = store.upsert_daily_facts([row for row, _days in rows_with_days])
        return written, rows_with_days

    written, rows_with_days = step("facts", facts_step, (0, []))
    if "facts" in failed:
        return _DayOutcome(failed_steps=tuple(failed))

    def transitions_step() -> int:
        previous = {
            row.user_id: row for row in store.list_facts_for_date(day - timedelta(days=1))
        }
        changed: list[LifecycleTransitionRow] = []
        for row, inactive_days in rows_with_days:
            before = previous.get(row.user_id)
            if before is None or before.lifecycle_segment == row.lifecycle_segment:
                continue
            changed.append(
                LifecycleTransitionRow(
                    user_id=row.user_id,
                    snapshot_date=day,
                    from_segment=before.lifecycle_segment,
                    to_segment=row.lifecycle_segment,
                    inactive_days=inactive_days,
                    data_quality=quality,
                    created_at=now,
                )
            )
        return store.append_lifecycle_transitions(changed)

    transitions = step("transitions", transitions_step, 0)
    return _DayOutcome(
        users_written=written,
        transitions_written=transitions,
        failed_steps=tuple(failed),
    )


def run_daily_facts(
    *,
    now: datetime | None = None,
    value_store: Any = None,
    run_history_store: Any = None,
    rollup: Callable[..., Any] | None = None,
    retention: Any = None,
) -> DailyFactsReport:
    """One worker tick: yesterday if due, late arrivals, retention.

    The due check is the claim itself (Task 8): a tick on a day that is
    already complete costs exactly one store call, the refused
    compare-and-set. A claimed day whose steps all succeed is completed; a
    claimed day with any failed step is written ``partial`` (the UI labels
    such periods "Incomplete data") and **released**, so the next tick
    retries the whole day and corrects the partial rows once the source is
    back. Every retry is the same fixed set of set-based queries, so a source
    that stays down costs one bounded batch per tick, never more.

    The late-arrival sweep (design SS6.9 step 7) runs only on a claimed tick,
    for the six days before D; it does not touch the claim. The retention
    coordinator runs on every tick -- its own clock makes that free when it
    is not due, and calling it here rather than from the reaper is what keeps
    the two from double-writing (design SS6.9 step 8).
    """
    current = _utc_now(now)
    store = value_store if value_store is not None else _default_value_store()
    runs = run_history_store if run_history_store is not None else _default_run_history_store()
    rollup_fn = rollup if rollup is not None else _default_rollup()
    coordinator = retention if retention is not None else _default_retention()
    target = current.date() - timedelta(days=1)

    claimed = store.claim_projection_day(DAILY_FACTS_JOB, day=target, now=current)
    report = DailyFactsReport(snapshot_date=target, claimed=claimed)
    if claimed:
        outcome = _compute_day(
            target, now=current, store=store, run_history_store=runs, rollup=rollup_fn
        )
        if outcome.failed_steps:
            store.release_projection_day(DAILY_FACTS_JOB, now=current)
        else:
            store.complete_projection_day(DAILY_FACTS_JOB, day=target, now=current)

        recomputed: list[date] = []
        try:
            stale_days = store.list_days_needing_recompute(
                since=target - timedelta(days=RECOMPUTE_LOOKBACK_DAYS),
                until=target - timedelta(days=1),
            )
        except Exception as exc:
            stale_days = []
            print(
                "WARNING: analytics.daily_facts.recompute_scan_failed "
                f"category={type(exc).__name__[:80]}"
            )
        for stale_day in stale_days[:MAX_RECOMPUTES_PER_TICK]:
            _compute_day(
                stale_day, now=current, store=store, run_history_store=runs, rollup=rollup_fn
            )
            recomputed.append(stale_day)

        report = DailyFactsReport(
            snapshot_date=target,
            claimed=True,
            users_written=outcome.users_written,
            transitions_written=outcome.transitions_written,
            partial=bool(outcome.failed_steps),
            failed_steps=outcome.failed_steps,
            recomputed_dates=tuple(recomputed),
        )

    try:
        coordinator.run_if_due()
    except Exception as exc:
        print(
            "WARNING: analytics.daily_facts.retention_failed "
            f"category={type(exc).__name__[:80]}"
        )
    return report


__all__ = ["DAILY_FACTS_JOB", "DailyFactsReport", "run_daily_facts"]
```

The query-constant table above counts `list_daily_subjects` under the analytics base and `list_operational_signals` under the value store; moving the subjects read to step 0 changes neither count. A `subjects` failure (the users table unreadable) is the one failure that writes nothing and is reported alone in `failed_steps`, because a partial row for nobody is not a partial row.

**Job step 9a — the evidence must be clamped to D, or the job crashes every night.** `calculate_lifecycle` raises `ValueError("lifecycle evidence cannot occur after as_of")` when its anchor is newer than `as_of` (`lifecycle.py:242-243`). The job asks for `as_of = <end of D>` but `user_activity` holds one row per user overwritten in place — it is *as of now*, and it has no history at all. So any user who emitted a lifecycle event since midnight has `last_meaningful_activity_at` on today's date, `inactive_days` computes to `-1`, and the exception fires. Since the step wraps the whole population in one try/except, **one such user marks the entire day partial and writes zero fact rows** — every night, forever, for a site with any overnight traffic. `build_lifecycle_inputs` (Task 4) clamps: `activated_at` after D becomes None; `last_meaningful_activity_at` after D falls back to the day's own last activity (`day_activity_at`, from the events aggregate — the only source that knows about D, which is why it is threaded in explicitly), then to `last_active_date` from the fact window, then to None. `created_at` after D is not clamped — that account had no day D and is dropped from the eligible set here.

**Job step 9b — the trailing window must include D, or the stored segment is computed from 29 days.** `sum_recent_facts(start=D-29, end=D-1)` is issued by the very step that is about to write D's row, so it covers 29 dates. A user whose third active day and third successful backtest both land on D would be written `growing`; the next day the read path sums 30 dates *including* D and answers `core` — the users list, segment distribution and `lifecycle_transitions` would carry the stored value while the profile showed the live one, and a spurious growing→core transition would land a day late in the outreach hook. The fix is in memory: D's own `active` and `runs_completed` are added from the events aggregate already in hand. `test_the_stored_segment_counts_its_own_day` pins it.

**Recomputing a past day uses current-state sources where no history exists.** `list_operational_signals(now=<end of E>)` re-runs the terminal-run window against `E`'s 24 hours (the run store filters by timestamp), but billing state, credentials and balances are current-state tables with no history; a recompute of `E` reads them as they are today. That is the same limitation the live profile has and is accepted: the recompute exists to catch late *events*, and the run-outcome and activity columns are what it corrects.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_daily_facts.py -v
python -m pytest dashboard/backend/tests/domain/analytics/ dashboard/backend/tests/test_store_twin_parity.py -q
TEST_POSTGRES_URL=postgresql://postgres:atl@localhost:55432/postgres \
  python -m pytest dashboard/backend/tests/domain/analytics/ -q
```

Expected: PASS on all.

- [ ] **Step 7: Commit**

```bash
git add dashboard/backend/domain/analytics/daily_facts.py \
        dashboard/backend/domain/analytics/repository.py \
        dashboard/backend/domain/analytics/repository_postgres.py \
        dashboard/backend/domain/analytics/value_repository.py \
        dashboard/backend/domain/analytics/value_repository_postgres.py \
        dashboard/backend/tests/domain/analytics/test_daily_facts.py
git status --short
git commit -m "feat: write user daily facts once per UTC day"
```

---

### Task 10: Copy eight weeks of lifecycle history into `user_daily_facts`

**Files:**
- Create: `dashboard/backend/domain/analytics/facts_migration.py`
- Modify: `dashboard/backend/domain/analytics/value_repository.py` — add `copy_daily_snapshot_history` to `ValueAnalyticsStore` after `list_days_needing_recompute`
- Modify: `dashboard/backend/domain/analytics/value_repository_postgres.py` — the same on `PostgresValueAnalyticsStore`
- Test: `dashboard/backend/tests/domain/analytics/test_facts_migration.py` (create)

**Interfaces:**
- Consumes: `user_daily_facts` (Task 1), `seed_activity_from_snapshots` (Task 3), `upsert_daily_facts`/`list_facts_for_date` (Task 9).
- Produces:
  - `copy_daily_snapshot_history(*, since: date, now: datetime) -> int` on both value twins
  - `migrate_lifecycle_history(*, value_store, now: datetime) -> int`, `run_startup_migrations(*, value_store=None, now: datetime | None = None) -> StartupMigrationReport`, `StartupMigrationReport(activity_seeded: int, history_copied: int)`, `HISTORY_COPY_DAYS = 56` in `facts_migration.py`

  Task 11's worker calls `run_startup_migrations` once, before its first tick.

This is the **copy** half of the superseded 2026-09-12 plan's Task B9 (`git show c3bbf2ed:...`, lines 3502-3566). Its **drop** half — deleting `states.py`, `lifecycle_backfill.py`, the value columns and `user_lifecycle_daily_snapshots` — is **not** in this PR (design §12 item 7, §13 row B): PR A only creates; PR B rewires the reads and then drops. Design §6.9 "Migration of existing history": eight weeks of `user_lifecycle_daily_snapshots` are copied into `user_daily_facts` with run, cost and tier columns at their neutral values and `data_quality='partial'`, so the movement charts keep their history once PR B moves them; `user_group` is read from `users` at the copy (D9 — the 09-12 version wrote `cohort=NULL`).

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/domain/analytics/test_facts_migration.py`:

```python
"""Existing lifecycle history is carried into the fact table as labelled partial rows."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from dashboard.backend.domain.analytics.facts_migration import (
    HISTORY_COPY_DAYS,
    StartupMigrationReport,
    migrate_lifecycle_history,
    run_startup_migrations,
)
from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.value_repository import (
    UserDailyFact,
    UserLifecycleDailySnapshot,
    build_value_analytics_store,
)
from dashboard.backend.tests.domain.analytics.test_value_repository import (
    _value_snapshot,
)
from dashboard.backend.users import UserStore


NOW = datetime(2026, 9, 12, 0, 5, tzinfo=timezone.utc)


class MigrationFixture:
    def __init__(self, tmp_path):
        path = tmp_path / "migration.db"
        users = UserStore(db_path=path)
        self.user_id = int(users.create_user("m@example.test", "M", "SecurePass1!")["id"])
        users.apply_admin_patch(self.user_id, user_group="partner")
        self.analytics = AnalyticsStore(db_path=path)
        self.store = build_value_analytics_store(
            self.analytics,
            credits_base=object(),
            provider_base=object(),
            agent_base=object(),
            run_base=object(),
        )
        self.legacy_dates = [
            date(2026, 8, 20),
            date(2026, 9, 1),
            date(2026, 9, 10),
        ]
        for day in self.legacy_dates + [date(2026, 6, 1)]:  # June is older than 56 days
            self.store.upsert_daily_snapshot(
                UserLifecycleDailySnapshot(
                    snapshot_date=day,
                    user_id=self.user_id,
                    lifecycle_segment="growing",
                    lifecycle_reason_code="growing_activated_below_core_threshold",
                    data_quality="complete",
                    calculated_at=datetime.combine(
                        day + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
                    ),
                )
            )
        self.legacy_row_count = len(self.legacy_dates)

    def seed_complete_fact(self, day):
        self.store.upsert_daily_facts(
            [
                UserDailyFact(
                    snapshot_date=day,
                    user_id=self.user_id,
                    lifecycle_segment="core",
                    lifecycle_reason_code="core_repeated_value",
                    operational_state="healthy",
                    operational_reason_code=None,
                    tier="starter",
                    user_group="partner",
                    active=True,
                    runs_completed=3,
                    data_quality="complete",
                    calculated_at=NOW,
                )
            ]
        )

    def fact_for(self, day):
        return {row.user_id: row for row in self.store.list_facts_for_date(day)}[self.user_id]


def test_history_is_copied_as_partial(tmp_path):
    """Movement charts keep their history; the missing columns stay honest."""
    fixture = MigrationFixture(tmp_path)

    copied = migrate_lifecycle_history(value_store=fixture.store, now=NOW)

    assert copied == fixture.legacy_row_count
    for day in fixture.legacy_dates:
        row = fixture.fact_for(day)
        assert row.data_quality == "partial"
        assert row.lifecycle_segment == "growing"
        assert row.runs_completed == 0
        assert row.operator_cost_micro == 0
        assert row.tier == "unpaid"
        assert row.operational_state == "healthy"
        assert row.operational_reason_code is None
        assert row.user_group == "partner"  # read from users at the copy (D9)
        assert row.active is False
    assert fixture.store.list_facts_for_date(date(2026, 6, 1)) == []
    assert (NOW.date() - timedelta(days=HISTORY_COPY_DAYS)) > date(2026, 6, 1)


def test_migration_is_idempotent(tmp_path):
    fixture = MigrationFixture(tmp_path)
    migrate_lifecycle_history(value_store=fixture.store, now=NOW)

    second = migrate_lifecycle_history(value_store=fixture.store, now=NOW)

    assert second == 0


def test_migration_never_overwrites_a_complete_row(tmp_path):
    """A day the new job already computed wins over a migrated stub."""
    fixture = MigrationFixture(tmp_path)
    fixture.seed_complete_fact(date(2026, 9, 10))

    migrate_lifecycle_history(value_store=fixture.store, now=NOW)
    row = fixture.fact_for(date(2026, 9, 10))

    assert row.data_quality == "complete"
    assert row.lifecycle_segment == "core"
    assert row.tier == "starter"


def test_startup_migrations_seed_activity_and_copy_history(tmp_path):
    fixture = MigrationFixture(tmp_path)
    fixture.store.upsert_current_snapshot(_value_snapshot(fixture.user_id))

    report = run_startup_migrations(value_store=fixture.store, now=NOW)
    again = run_startup_migrations(value_store=fixture.store, now=NOW)

    assert report == StartupMigrationReport(activity_seeded=1, history_copied=3)
    assert again == StartupMigrationReport(activity_seeded=1, history_copied=0)
    assert fixture.store.get_activity(fixture.user_id).activated_at is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_facts_migration.py -v
```

Expected: FAIL at import — `ModuleNotFoundError: No module named 'dashboard.backend.domain.analytics.facts_migration'`.

- [ ] **Step 3: Add the copy to both value twins**

On `ValueAnalyticsStore`, immediately after `list_days_needing_recompute`:

```python
    def copy_daily_snapshot_history(self, *, since: date, now: datetime) -> int:
        """Copy legacy ``user_lifecycle_daily_snapshots`` rows into the fact table.

        Run, cost and tier columns are filled with their neutral values and
        every copied row is ``partial``, so the UI labels the period
        "Incomplete data" instead of charting zeros as fact (design SS6.7).
        ``tier='unpaid'`` is a placeholder, not a claim, for the same reason.
        ``user_group`` is read from ``users`` at the copy (D9). ``ON CONFLICT
        DO NOTHING`` makes it both idempotent and unable to clobber a row the
        daily job already computed. Returns the number of rows inserted.
        """
        stamp = utc_iso(_utc(now, "now"))
        sql = """
            INSERT INTO user_daily_facts (
                snapshot_date, user_id, lifecycle_segment, lifecycle_reason_code,
                operational_state, operational_reason_code, tier, user_group, active,
                runs_requested, runs_completed, runs_failed, runs_cancelled,
                operator_cost_micro, own_spend_micro, data_quality, calculated_at
            )
            SELECT s.snapshot_date, s.user_id, s.lifecycle_segment,
                   s.lifecycle_reason_code, 'healthy', NULL, 'unpaid',
                   users.user_group, 0, 0, 0, 0, 0, 0, 0, 'partial', ?
            FROM user_lifecycle_daily_snapshots AS s
            JOIN users ON users.id = s.user_id
            WHERE s.snapshot_date >= ?
            ON CONFLICT(snapshot_date, user_id) DO NOTHING
        """
        with self._analytics_connection() as conn:
            cursor = conn.execute(sql, (stamp, since.isoformat()))
            return max(0, int(cursor.rowcount))
```

On `PostgresValueAnalyticsStore`, immediately after `list_days_needing_recompute`:

```python
    def copy_daily_snapshot_history(self, *, since: date, now: datetime) -> int:
        """See the SQLite twin. ``active`` is BOOLEAN here."""
        stamp = utc_iso(_utc(now, "now"))
        sql = """
            INSERT INTO user_daily_facts (
                snapshot_date, user_id, lifecycle_segment, lifecycle_reason_code,
                operational_state, operational_reason_code, tier, user_group, active,
                runs_requested, runs_completed, runs_failed, runs_cancelled,
                operator_cost_micro, own_spend_micro, data_quality, calculated_at
            )
            SELECT s.snapshot_date, s.user_id, s.lifecycle_segment,
                   s.lifecycle_reason_code, 'healthy', NULL, 'unpaid',
                   users.user_group, FALSE, 0, 0, 0, 0, 0, 0, 'partial', %s
            FROM user_lifecycle_daily_snapshots AS s
            JOIN users ON users.id = s.user_id
            WHERE s.snapshot_date >= %s
            ON CONFLICT(snapshot_date, user_id) DO NOTHING
        """
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (stamp, since.isoformat()))
                return max(0, int(cur.rowcount))
```

- [ ] **Step 4: Write the migration module**

Create `dashboard/backend/domain/analytics/facts_migration.py`:

```python
"""Startup migrations for the fact tables PR A creates.

Both are idempotent and both run again on every boot; the worker thread in
``daily_job.py`` calls ``run_startup_migrations`` once before its first tick.
PR B re-runs the same seed immediately before dropping the legacy tables,
because the legacy snapshot row keeps being maintained between the two PRs
(design SS6.9 "Migration of existing history", SS11).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


HISTORY_COPY_DAYS = 56


class StartupMigrationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    activity_seeded: int = Field(default=0, ge=0)
    history_copied: int = Field(default=0, ge=0)


def _utc_now(value: datetime | None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("now must include a timezone")
    return current.astimezone(timezone.utc)


def migrate_lifecycle_history(*, value_store: Any, now: datetime) -> int:
    """Copy the last eight weeks of legacy daily snapshots; returns rows inserted."""
    current = _utc_now(now)
    return value_store.copy_daily_snapshot_history(
        since=current.date() - timedelta(days=HISTORY_COPY_DAYS), now=current
    )


def run_startup_migrations(
    *,
    value_store: Any = None,
    now: datetime | None = None,
) -> StartupMigrationReport:
    """Seed ``user_activity`` from the legacy row, then copy the history."""
    current = _utc_now(now)
    if value_store is None:
        from .repository import analytics_store
        from .value_repository import build_value_analytics_store

        value_store = build_value_analytics_store(analytics_store)
    seeded = value_store.seed_activity_from_snapshots(now=current)
    copied = migrate_lifecycle_history(value_store=value_store, now=current)
    return StartupMigrationReport(activity_seeded=seeded, history_copied=copied)


__all__ = [
    "HISTORY_COPY_DAYS",
    "StartupMigrationReport",
    "migrate_lifecycle_history",
    "run_startup_migrations",
]
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_facts_migration.py dashboard/backend/tests/test_store_twin_parity.py -v
TEST_POSTGRES_URL=postgresql://postgres:atl@localhost:55432/postgres \
  python -m pytest dashboard/backend/tests/domain/analytics/ -q
```

Expected: PASS on both tiers.

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/domain/analytics/facts_migration.py \
        dashboard/backend/domain/analytics/value_repository.py \
        dashboard/backend/domain/analytics/value_repository_postgres.py \
        dashboard/backend/tests/domain/analytics/test_facts_migration.py
git status --short
git commit -m "feat: copy eight weeks of lifecycle history into user_daily_facts"
```

---

### Task 11: The job on its own worker thread; maintenance keeps only the repairs

**Files:**
- Create: `dashboard/backend/domain/analytics/daily_job.py`
- Modify: `dashboard/backend/app.py` — `startup_event`: delete the retention registration block (lines 259-270); insert the worker start block after PR 0's `disable_synchronous_projection()` block and before the `start_reaper` block (line 290 on `main`, shifted by PR 0's insertion)
- Modify: `dashboard/backend/domain/analytics/maintenance.py` — full-file rewrite (176 lines → the two repairs)
- Modify: `dashboard/backend/tests/test_analytics_maintenance.py` — rewrite the two behaviour tests (lines 15-117); keep `test_app_registers_analytics_maintenance_through_reaper` and PR 0's `test_startup_disables_synchronous_snapshot_projection`
- Modify: `dashboard/backend/tests/test_run_lifecycle_unification.py` — `test_startup_registers_analytics_retention_sweep_once` (lines 186-189)
- Modify: `dashboard/backend/tests/conftest.py` — one `os.environ.pop` after line 93
- Modify: `CLAUDE.md` — one new env-var bullet after the `MAX_LEGACY_ACTIVE_PER_SESSION` bullet (line 76); one clause in the reaper-cadence gotcha PR 0 appended
- Test: `dashboard/backend/tests/domain/analytics/test_daily_job.py` (create)

**Interfaces:**
- Consumes: `run_daily_facts` (Task 9), `run_startup_migrations` (Task 10), `analytics_retention_coordinator` (unchanged).
- Produces:
  - `daily_job_interval_seconds() -> float` (env `ANALYTICS_DAILY_JOB_INTERVAL_SECONDS`, default **300**, range 5–3600, junk-tolerant), `start_daily_facts_worker(interval_seconds: float | None = None, stop_event: threading.Event | None = None, *, tick=None, prepare=None) -> threading.Thread`, `stop_daily_facts_worker(timeout: float = 5.0) -> None` in `daily_job.py`
  - `run_analytics_maintenance(*, now=None, snapshot_limit=100, repair_snapshots=None, repair_value_snapshots=None) -> AnalyticsMaintenanceReport(repaired_snapshots, repaired_value_snapshots, failures)` — the rollup, the lifecycle backfill, `_guard_lock`, `_last_rollup_day` and `reset_maintenance_guard_for_tests` are gone
  - `app.py` starts the worker beside the reaper and no longer registers the retention coordinator as a reaper sweep.

This is D23 and design §6.9 "Amended — scheduling": the job runs on **its own** thread with its own interval and stop event, not as a `register_reaper_sweep` step, because the reaper's job is to keep run heartbeats fresh against `RUN_HEARTBEAT_STALE_SECONDS` (300 s) and a whole-population, eight-step, three-database batch on that thread would let a slow analytics night mark live runs as orphaned. The superseded 2026-09-12 plan's Task B8 step 5 ("Replace the body of `run_analytics_maintenance` ... with a call to `run_daily_facts`") is exactly what this task does **not** do. `rollup_day` and the retention coordinator leave the reaper for the job (§6.9 steps 1 and 8) so the two can never double-write; the lifecycle backfill batch leaves with them — Task 10's copy is what carries the history now — and `maintenance.py` keeps only the two 24-hour-throttled snapshot repairs PR 0 left, until PR B deletes the module (§13 row B).

The env var is parsed with the same junk-tolerant shape as `MAX_ACTIVE_DASHBOARD_BACKTESTS` (`api/routers/backtests.py:554-592`): a bad value logs and falls back rather than raising at import, because an unparseable value read with a bare `int()` at module scope has killed app boot in this repo before (`CLAUDE.md`, `PIPELINE_SECONDS_PER_LLM_CALL`).

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/domain/analytics/test_daily_job.py`:

```python
"""The daily-facts worker owns its schedule (design D23, SS6.9)."""

from __future__ import annotations

import inspect
import threading

import pytest

import dashboard.backend.app as app_module
from dashboard.backend.domain.analytics import daily_facts as daily_facts_module
from dashboard.backend.domain.analytics import daily_job


@pytest.fixture(autouse=True)
def _stop_worker():
    yield
    daily_job.stop_daily_facts_worker()


def test_interval_defaults_to_five_minutes(monkeypatch):
    monkeypatch.delenv("ANALYTICS_DAILY_JOB_INTERVAL_SECONDS", raising=False)
    assert daily_job.daily_job_interval_seconds() == 300.0


@pytest.mark.parametrize("raw", ["junk", "", "  ", "4", "3601", "-1"])
def test_bad_intervals_fall_back_with_a_line_instead_of_raising(monkeypatch, capsys, raw):
    monkeypatch.setenv("ANALYTICS_DAILY_JOB_INTERVAL_SECONDS", raw)
    assert daily_job.daily_job_interval_seconds() == 300.0
    if raw.strip():
        assert "ANALYTICS_DAILY_JOB_INTERVAL_SECONDS" in capsys.readouterr().out


def test_a_valid_interval_is_honoured(monkeypatch):
    monkeypatch.setenv("ANALYTICS_DAILY_JOB_INTERVAL_SECONDS", "60")
    assert daily_job.daily_job_interval_seconds() == 60.0


def test_the_worker_prepares_once_then_ticks_until_stopped():
    stop = threading.Event()
    calls: list[str] = []

    def prepare():
        calls.append("prepare")

    def tick():
        calls.append("tick")
        if calls.count("tick") >= 3:
            stop.set()

    thread = daily_job.start_daily_facts_worker(
        0.01, stop, tick=tick, prepare=prepare
    )
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert calls[0] == "prepare"
    assert calls.count("prepare") == 1
    assert calls.count("tick") >= 3
    assert thread.daemon is True
    assert thread.name == "analytics-daily-facts"


def test_a_failing_tick_is_logged_and_the_loop_continues(capsys):
    stop = threading.Event()
    ticks: list[int] = []

    def tick():
        ticks.append(len(ticks))
        if len(ticks) == 1:
            raise RuntimeError("private tick detail")
        if len(ticks) >= 2:
            stop.set()

    daily_job.start_daily_facts_worker(0.01, stop, tick=tick, prepare=lambda: None).join(5)
    printed = capsys.readouterr().out

    assert len(ticks) >= 2
    assert "WARNING: analytics.daily_facts.tick_failed category=RuntimeError" in printed
    assert "private tick detail" not in printed


def test_a_failing_prepare_is_logged_and_the_worker_still_ticks(capsys):
    stop = threading.Event()

    def prepare():
        raise RuntimeError("private migration detail")

    daily_job.start_daily_facts_worker(0.01, stop, tick=stop.set, prepare=prepare).join(5)
    printed = capsys.readouterr().out

    assert "WARNING: analytics.facts_migration_failed category=RuntimeError" in printed
    assert "private migration detail" not in printed


def test_starting_twice_returns_the_live_thread():
    stop = threading.Event()
    first = daily_job.start_daily_facts_worker(60, stop, tick=lambda: None, prepare=lambda: None)
    second = daily_job.start_daily_facts_worker(60, stop, tick=lambda: None, prepare=lambda: None)

    assert first is second
    daily_job.stop_daily_facts_worker()
    assert not first.is_alive()


def test_startup_starts_the_worker_and_leaves_the_reaper_to_heartbeats():
    source = inspect.getsource(app_module.startup_event)

    assert source.count("start_daily_facts_worker()") == 1
    assert "register_reaper_sweep(analytics_retention_coordinator.run_if_due)" not in source
    assert "register_reaper_sweep(run_daily_facts" not in source
    # The throttled snapshot repairs stay on the reaper until PR B deletes them.
    assert "register_reaper_sweep(run_analytics_maintenance)" in source


def test_the_job_owns_the_retention_coordinator_now():
    source = inspect.getsource(daily_facts_module)
    assert source.count("analytics_retention_coordinator") == 1
```

Rewrite `dashboard/backend/tests/test_analytics_maintenance.py`'s first two tests (keep the module docstring, the imports PR 0 added — `inspect`, `dashboard.backend.app as app_module` — `NOW`, and the two source-pin tests at the bottom):

```python
def test_maintenance_repairs_bounded_batches_and_nothing_else():
    repair_limits = []
    value_repair_limits = []
    repair_order = []

    def repair(**kwargs):
        repair_order.append("legacy")
        repair_limits.append(kwargs["limit"])
        return 25

    def repair_values(**kwargs):
        repair_order.append("value")
        value_repair_limits.append(kwargs["limit"])
        return 12

    report = maintenance.run_analytics_maintenance(
        now=NOW,
        snapshot_limit=250,
        repair_snapshots=repair,
        repair_value_snapshots=repair_values,
    )

    assert report.repaired_snapshots == 25
    assert report.repaired_value_snapshots == 12
    assert report.failures == 0
    assert repair_limits == [100]
    assert value_repair_limits == [100]
    assert repair_order == ["value", "legacy"]


def test_maintenance_isolates_repair_failures():
    def fail_repair(**_kwargs):
        raise RuntimeError("private user detail")

    def fail_value_repair(**_kwargs):
        raise RuntimeError("private value projection detail")

    report = maintenance.run_analytics_maintenance(
        now=NOW,
        repair_snapshots=fail_repair,
        repair_value_snapshots=fail_value_repair,
    )

    assert report.repaired_snapshots == 0
    assert report.repaired_value_snapshots == 0
    assert report.failures == 2


def test_maintenance_no_longer_owns_rollups_or_the_lifecycle_backfill():
    """Design SS6.9 step 1 / D23: the daily job owns rollup_day now, so the
    reaper tick must not be able to write the same rollup rows."""
    parameters = inspect.signature(maintenance.run_analytics_maintenance).parameters
    source = inspect.getsource(maintenance)

    assert "rebuild_rollup" not in parameters
    assert "backfill_lifecycle" not in parameters
    assert "rollup_day" not in source
    assert "lifecycle_backfill" not in source
    assert not hasattr(maintenance, "reset_maintenance_guard_for_tests")
    assert not hasattr(maintenance, "_last_rollup_day")
```

Delete the old `test_maintenance_rebuilds_one_day_and_bounds_snapshot_repairs` and `test_maintenance_isolates_rollup_and_snapshot_failures` and the `from types import SimpleNamespace` import they used.

In `dashboard/backend/tests/test_run_lifecycle_unification.py`, replace:

```python
def test_startup_registers_analytics_retention_sweep_once():
    source = inspect.getsource(app_module.startup_event)
    call = "register_reaper_sweep(analytics_retention_coordinator.run_if_due)"
    assert source.count(call) == 1
```

with:

```python
def test_startup_leaves_analytics_retention_to_the_daily_job():
    """Admin layer redesign PR A: the retention coordinator runs from the
    daily-facts worker (domain/analytics/daily_facts.py), not the reaper."""
    source = inspect.getsource(app_module.startup_event)
    call = "register_reaper_sweep(analytics_retention_coordinator.run_if_due)"
    assert source.count(call) == 0
    assert source.count("start_daily_facts_worker()") == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_daily_job.py dashboard/backend/tests/test_analytics_maintenance.py dashboard/backend/tests/test_run_lifecycle_unification.py -v
```

Expected: FAIL. `test_daily_job.py` fails at import (`ModuleNotFoundError: ... daily_job`); the rewritten maintenance tests fail with `TypeError` (`run_analytics_maintenance` still requires the rollup/backfill collaborators to be absent from the signature it does not yet have) and `assert "rollup_day" not in source`; `test_startup_leaves_analytics_retention_to_the_daily_job` fails at `source.count(call) == 0` (it is 1).

- [ ] **Step 3: Write the worker**

Create `dashboard/backend/domain/analytics/daily_job.py`:

```python
"""The daily-facts worker: its own thread, its own interval, its own stop event.

Design D23 / SS6.9: the run reaper exists to keep ``heartbeat_at`` fresh against
``RUN_HEARTBEAT_STALE_SECONDS`` (300 s). A whole-population batch across three
databases on that thread would let a slow analytics night mark live runs as
orphaned, so the job is *not* a ``register_reaper_sweep`` step. It is still
inside the single web process (design SS2.3): no external scheduler, no second
service.

Every tick is ``daily_facts.run_daily_facts``; the first thing the thread does,
once, is ``facts_migration.run_startup_migrations`` (seed ``user_activity``,
copy the legacy history). Both are idempotent, so a restart costs nothing.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Callable


_DEFAULT_INTERVAL_SECONDS = 300
_MIN_INTERVAL_SECONDS = 5
_MAX_INTERVAL_SECONDS = 3600
_ENV_NAME = "ANALYTICS_DAILY_JOB_INTERVAL_SECONDS"


def daily_job_interval_seconds() -> float:
    """Seconds between worker ticks. Junk falls back to the default with a line.

    Same shape as ``MAX_ACTIVE_DASHBOARD_BACKTESTS`` in
    ``api/routers/backtests.py``: an unparseable value read with a bare
    ``int()`` at module scope once killed app boot, so a bad value here logs
    and uses 300 rather than raising.
    """
    raw = os.getenv(_ENV_NAME)
    if raw is None or not str(raw).strip():
        return float(_DEFAULT_INTERVAL_SECONDS)
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        print(
            f"{_ENV_NAME} is not an integer ({raw!r}); using {_DEFAULT_INTERVAL_SECONDS}",
            flush=True,
        )
        return float(_DEFAULT_INTERVAL_SECONDS)
    if value < _MIN_INTERVAL_SECONDS or value > _MAX_INTERVAL_SECONDS:
        print(
            f"{_ENV_NAME} is out of range ({value}; allowed "
            f"{_MIN_INTERVAL_SECONDS}-{_MAX_INTERVAL_SECONDS}); "
            f"using {_DEFAULT_INTERVAL_SECONDS}",
            flush=True,
        )
        return float(_DEFAULT_INTERVAL_SECONDS)
    return float(value)


_worker_lock = threading.Lock()
_worker_thread: threading.Thread | None = None
_worker_stop: threading.Event | None = None


def _default_tick() -> Any:
    from .daily_facts import run_daily_facts

    return run_daily_facts()


def _default_prepare() -> Any:
    from .facts_migration import run_startup_migrations

    return run_startup_migrations()


def start_daily_facts_worker(
    interval_seconds: float | None = None,
    stop_event: threading.Event | None = None,
    *,
    tick: Callable[[], Any] | None = None,
    prepare: Callable[[], Any] | None = None,
) -> threading.Thread:
    """Start the worker (idempotent -- a second call while it is alive no-ops).

    ``tick`` and ``prepare`` are injectable for tests; production leaves both
    at their defaults. The loop waits ``interval_seconds`` *before* the first
    tick, so a boot does not run the day's job on top of everything else the
    startup hook is doing; the claim makes the first tick cheap when the day
    is already done.
    """
    global _worker_thread, _worker_stop
    interval = (
        float(interval_seconds) if interval_seconds is not None else daily_job_interval_seconds()
    )
    with _worker_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return _worker_thread
        stop = stop_event if stop_event is not None else threading.Event()
        tick_fn = tick if tick is not None else _default_tick
        prepare_fn = prepare if prepare is not None else _default_prepare

        def _loop() -> None:
            try:
                prepare_fn()
            except Exception as exc:
                print(
                    "WARNING: analytics.facts_migration_failed "
                    f"category={type(exc).__name__[:80]}"
                )
            while not stop.wait(interval):
                try:
                    tick_fn()
                except Exception as exc:
                    print(
                        "WARNING: analytics.daily_facts.tick_failed "
                        f"category={type(exc).__name__[:80]}"
                    )

        thread = threading.Thread(target=_loop, daemon=True, name="analytics-daily-facts")
        thread.start()
        _worker_thread = thread
        _worker_stop = stop
        return thread


def stop_daily_facts_worker(timeout: float = 5.0) -> None:
    """Stop the worker and wait for it; a no-op when none is running (tests)."""
    global _worker_thread, _worker_stop
    with _worker_lock:
        thread, stop = _worker_thread, _worker_stop
        _worker_thread = None
        _worker_stop = None
    if stop is not None:
        stop.set()
    if thread is not None and thread.is_alive():
        thread.join(timeout=timeout)


__all__ = [
    "daily_job_interval_seconds",
    "start_daily_facts_worker",
    "stop_daily_facts_worker",
]
```

- [ ] **Step 4: Rewrite `maintenance.py` to the two repairs**

Replace the whole of `dashboard/backend/domain/analytics/maintenance.py` with:

```python
"""Throttled repair of the legacy Analytics snapshot rows.

Admin layer redesign PR A moved ``rollup_day`` and the lifecycle backfill off
this reaper tick: the daily-facts job (``daily_facts.py``, on its own worker
thread, ``daily_job.py``) owns the day rollup now, so the two can never write
the same rows (design SS6.9 step 1, D23), and the eight-week history copy in
``facts_migration.py`` replaces the backfill. What remains is the two
24-hour-throttled snapshot repairs PR 0 left, which keep
``user_analytics_snapshots`` maintained for the read paths that still use it.
PR B moves those read paths and deletes this module.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AnalyticsMaintenanceReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repaired_snapshots: int = Field(default=0, ge=0)
    repaired_value_snapshots: int = Field(default=0, ge=0)
    failures: int = Field(default=0, ge=0)


def _current_utc(value: datetime | None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("now must include a timezone")
    return current.astimezone(timezone.utc)


def run_analytics_maintenance(
    *,
    now: datetime | None = None,
    snapshot_limit: int = 100,
    repair_snapshots: Any | None = None,
    repair_value_snapshots: Any | None = None,
) -> AnalyticsMaintenanceReport:
    """Repair bounded batches of stale dual-axis and legacy snapshots."""

    current = _current_utc(now)
    if isinstance(snapshot_limit, bool) or not isinstance(snapshot_limit, int):
        raise ValueError("snapshot_limit must be an integer")
    page_size = max(1, min(snapshot_limit, 100))
    failures = 0

    if repair_snapshots is None:
        from .states import repair_stale_snapshots

        repair_snapshots = repair_stale_snapshots
    if repair_value_snapshots is None:
        from .states import repair_stale_value_snapshots

        repair_value_snapshots = repair_stale_value_snapshots

    repaired_values = 0
    try:
        repaired_values = int(
            repair_value_snapshots(
                now=current,
                limit=page_size,
            )
        )
    except Exception as exc:
        failures += 1
        print(
            "WARNING: analytics.value_snapshot_maintenance_failed "
            f"category={type(exc).__name__[:80]}"
        )

    repaired = 0
    try:
        repaired = int(
            repair_snapshots(
                now=current,
                limit=page_size,
            )
        )
    except Exception as exc:
        failures += 1
        print(
            "WARNING: analytics.snapshot_maintenance_failed "
            f"category={type(exc).__name__[:80]}"
        )

    return AnalyticsMaintenanceReport(
        repaired_snapshots=max(0, repaired),
        repaired_value_snapshots=max(0, repaired_values),
        failures=failures,
    )


__all__ = [
    "AnalyticsMaintenanceReport",
    "run_analytics_maintenance",
]
```

Confirm nothing else imports the deleted names: `rg -n "reset_maintenance_guard_for_tests|rollup_days|rollup_rebuilt|backfilled_lifecycle|lifecycle_backfill_complete" dashboard/backend/` must return only the rewritten test file's absence (i.e. no hits).

- [ ] **Step 5: Rewire `app.py`**

In `dashboard/backend/app.py::startup_event`, **delete** this block:

```python
    try:
        from dashboard.backend.domain.analytics.retention import (
            analytics_retention_coordinator,
        )
        from dashboard.backend.domain.runs.service import register_reaper_sweep
        register_reaper_sweep(analytics_retention_coordinator.run_if_due)
        print("🧹 Analytics retention sweep registered with the reaper")
    except Exception as e:
        print(
            "WARNING: analytics.retention_registration_failed "
            f"category={type(e).__name__}"
        )
```

Leave the `register_reaper_sweep(run_analytics_maintenance)` block and PR 0's `disable_synchronous_projection()` block exactly as they are. Immediately before:

```python
    try:
        from dashboard.backend.domain.runs.service import start_reaper
        start_reaper()
        print("🧹 Run reaper started")
    except Exception as e:
        print(f"⚠️ Run reaper start error: {e}")
```

insert:

```python
    try:
        # Admin layer redesign PR A (design D23, SS6.9): the daily-facts job
        # runs on its own thread, not as a reaper sweep -- a whole-population
        # batch across three databases on the heartbeat thread would let a
        # slow analytics night mark live runs as orphaned. The worker also
        # owns rollup_day and the retention coordinator now, and runs the
        # idempotent user_activity seed + history copy once before ticking.
        from dashboard.backend.domain.analytics.daily_job import (
            start_daily_facts_worker,
        )

        start_daily_facts_worker()
        print("🧹 Analytics daily-facts worker started")
    except Exception as e:
        print(
            "WARNING: analytics.daily_job_start_failed "
            f"category={type(e).__name__}"
        )
```

- [ ] **Step 6: Strip the new env var in `conftest.py` and document it**

In `dashboard/backend/tests/conftest.py`, after `os.environ.pop("MAX_LEGACY_ACTIVE_GLOBAL", None)` insert:

```python
os.environ.pop("ANALYTICS_DAILY_JOB_INTERVAL_SECONDS", None)
```

In `CLAUDE.md`, after the `MAX_LEGACY_ACTIVE_PER_SESSION` / `MAX_LEGACY_ACTIVE_GLOBAL` bullet (line 76), insert:

```markdown
- `ANALYTICS_DAILY_JOB_INTERVAL_SECONDS` (optional, default **300**, range 5–3600): how often the analytics daily-facts worker (`domain/analytics/daily_job.py`, started from `app.py` beside the run reaper) wakes to ask whether yesterday's `user_daily_facts` row set is due. It runs on **its own thread**, not the 60-second reaper tick — the reaper exists to keep run heartbeats fresh against `RUN_HEARTBEAT_STALE_SECONDS`, and a whole-population batch across three databases on that thread would let a slow analytics night mark live runs as orphaned (design D23). An idle wake costs exactly one store call (the refused day claim on `analytics_projection_jobs`), so the interval buys freshness after midnight, not cost. Junk or out-of-range values log and fall back to 300 rather than raising at import — a bare `int()` at module scope has killed app boot in this repo before. `tests/conftest.py` strips it. The worker also runs the idempotent `user_activity` seed and the eight-week history copy (`domain/analytics/facts_migration.py`) once before its first tick, and owns `rollup_day` and the analytics retention coordinator; the reaper keeps only the two throttled snapshot repairs in `maintenance.py` until PR B deletes them. Design: `docs/superpowers/specs/2026-09-15-admin-layer-redesign-design.md` §6.9, §6.12.
```

In the reaper-cadence gotcha PR 0 appended at the end of `## Gotchas`, change the clause

```markdown
it is throttled to a 24-hour staleness window by PR 0 and deleted outright by PR A once `user_daily_facts` replaces `user_analytics_snapshots`.
```

to:

```markdown
it is throttled to a 24-hour staleness window by PR 0, left on the reaper by PR A (which gives the daily-facts job its own worker thread, `domain/analytics/daily_job.py`, precisely so that job is *not* another reaper sweep, and moves `rollup_day` and the retention coordinator onto it), and deleted outright by PR B once every read path has moved to `user_daily_facts`.
```

- [ ] **Step 7: Run the tests to verify they pass**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_daily_job.py dashboard/backend/tests/test_analytics_maintenance.py dashboard/backend/tests/test_run_lifecycle_unification.py dashboard/backend/tests/test_app_composition.py -v
python -m pytest dashboard/backend/tests/ -q
```

Expected: PASS. `test_app_composition.py` freezes the route set only, which this task does not change.

- [ ] **Step 8: Commit**

```bash
git add dashboard/backend/domain/analytics/daily_job.py \
        dashboard/backend/domain/analytics/maintenance.py \
        dashboard/backend/app.py \
        dashboard/backend/tests/conftest.py \
        dashboard/backend/tests/domain/analytics/test_daily_job.py \
        dashboard/backend/tests/test_analytics_maintenance.py \
        dashboard/backend/tests/test_run_lifecycle_unification.py \
        CLAUDE.md
git status --short
git commit -m "feat: run the daily-facts job on its own worker thread"
```

---

### Task 12: The retention sweep never touches `user_activity`

**Files:**
- Test: `dashboard/backend/tests/domain/analytics/test_retention.py` (extend)

**Interfaces:**
- Consumes: `record_activity`/`get_activity` (Task 3), `AnalyticsRetentionService` (unchanged).
- Produces: nothing. This is the guard design §11 asks PR A to add.

Design §11, "The `user_activity` exception, stated in words": lifetime metrics survive solely because `activated_at` is set once on a one-row-per-user table that is never swept. Every other user-keyed analytics table is retained 180 days, so a future tidy-up that "completes" the 180-day rule over every user-keyed table would delete the only lifetime evidence in the system with every test still green. This test is what makes that tidy-up red. `retention.py` itself does not change in this PR — its sweep still expires `analytics_events`, `admin_analytics_access_log` and `user_lifecycle_daily_snapshots` (PR B re-points the last at `user_daily_facts` and `lifecycle_transitions`).

- [ ] **Step 1: Write the failing test (it passes against current code, and that is the point — see Step 2)**

Append to `dashboard/backend/tests/domain/analytics/test_retention.py` (it already imports `AnalyticsStore`, `AnalyticsRetentionService`, `RAW_EVENT_RETENTION_DAYS`, `UserStore`, `NOW`, `timedelta` and defines `_event`):

```python
def test_the_retention_sweep_never_touches_user_activity(tmp_path):
    """Design doc SS11: user_activity is current state, not history.

    Every other user-keyed analytics table is retained 180 days. This one
    holds the only record of when a user first activated, on a row that is
    overwritten in place; sweeping it would delete every lifetime metric with
    nothing else failing. A future "complete the 180-day rule" tidy-up must
    turn this red.
    """
    import inspect
    import sqlite3

    from dashboard.backend.domain.analytics import retention as retention_module
    from dashboard.backend.domain.analytics.value_repository import (
        build_value_analytics_store,
    )

    path = tmp_path / "retention.db"
    UserStore(db_path=path)
    with sqlite3.connect(path) as conn:
        conn.executemany(
            "INSERT INTO users (id, email, display_name, password_hash, role, user_group, created_at) "
            "VALUES (?, ?, ?, 'x', 'user', 'unknown', ?)",
            [
                (1, "one@example.test", "One", (NOW - timedelta(days=500)).isoformat()),
                (2, "two@example.test", "Two", (NOW - timedelta(days=500)).isoformat()),
            ],
        )
    analytics = AnalyticsStore(path)
    value_store = build_value_analytics_store(
        analytics,
        credits_base=object(),
        provider_base=object(),
        agent_base=object(),
        run_base=object(),
    )
    ancient = NOW - timedelta(days=RAW_EVENT_RETENTION_DAYS + 200)
    for user_id in (1, 2):
        value_store.record_activity(user_id, occurred_at=ancient, activating=True, now=ancient)
        analytics.append_event(_event(user_id, ancient))
    before = {user_id: value_store.get_activity(user_id) for user_id in (1, 2)}

    result = AnalyticsRetentionService(store=analytics, value_store=value_store).run_once(NOW)

    assert result.raw_events_deleted == 2
    assert {user_id: value_store.get_activity(user_id) for user_id in (1, 2)} == before
    assert before[1].activated_at == ancient
    assert "user_activity" not in inspect.getsource(retention_module)
```

- [ ] **Step 2: Run it and prove it has teeth**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_retention.py::test_the_retention_sweep_never_touches_user_activity -v
```

Expected: **PASS** — `retention.py` does not touch the table today, and this PR does not change `retention.py`. A guard that only ever passes is worthless unless it is shown to fail on the shape it forbids, so temporarily add, inside `AnalyticsRetentionService.run_once` just before the `return RetentionResult(...)`:

```python
        with self.store._get_connection() as conn:
            conn.execute("DELETE FROM user_activity WHERE updated_at < ?", (raw_before.isoformat(),))
```

re-run the single test and confirm it FAILS at `assert {...} == before` (both rows are gone, `get_activity` returns None), then remove the two lines and confirm it passes again. Do not commit the probe.

- [ ] **Step 3: Run the file and commit**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_retention.py -v
git add dashboard/backend/tests/domain/analytics/test_retention.py
git status --short
git commit -m "test: pin that the retention sweep never touches user_activity"
```

---

### Task 13: Router hygiene (D24)

**Files:**
- Modify: `dashboard/backend/api/routers/admin_analytics.py` — import block (lines 14-21), constants (lines 53-55), delete `_user_filters` (lines 298-362), `_raise_service_error` (lines 378-383)
- Modify: `dashboard/backend/domain/analytics/query_service.py` — delete `_USER_STATES` (line 35), `AnalyticsUserFilters` (lines 123-161), `PaginatedUsers` (lines 164-170), `AnalyticsQueryService.list_users` (lines 961-1068); `__all__` (lines 1330, 1334)
- Modify: `dashboard/backend/tests/test_event_loop_threadpool.py` — `BLOCKING_IO_ROUTER_MODULES` (lines 27-42)
- Modify: `dashboard/backend/tests/test_admin_analytics_api.py` — import (line 20), the `service.list_users(...)` block (lines 328-333) and its two assertions
- Modify: `dashboard/backend/tests/domain/analytics/test_repository_contract.py` — import (lines 13-16), the `service.list_users(...)` block (lines 294-299) and its two assertions
- Test: `dashboard/backend/tests/test_admin_analytics_hygiene.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `_raise_service_error` prints `ERROR: admin_analytics.unhandled category=<Class>` and raises the 503 `from exc`; `dashboard.backend.api.routers.admin_analytics` is in `BLOCKING_IO_ROUTER_MODULES`; `_user_filters`, `AnalyticsUserFilters`, `PaginatedUsers` and `AnalyticsQueryService.list_users` no longer exist. **No route, path, response model or response body changes** (§10.1): the dead stack was reachable from no route (design §4.4).

D24 folds these into PR A because "a hygiene-only PR is the one that sits until it conflicts". The fact-base for §4.4 confirmed each deletion: `rg -n "_user_filters\(request\)"` in `api/` returns nothing, the only `.list_users(` call in `api/` (`admin_analytics.py:534`) is on `ValueAnalyticsQueryService`, and `AnalyticsUserFilters` is imported into the router solely for `_user_filters`. Two tests construct the dead stack directly and are edited here; that is the whole extent of the conformance-oracle edit this PR is allowed (Global Constraints).

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/test_admin_analytics_hygiene.py`:

```python
"""D24 hygiene on the admin analytics router: loud 503s, no dead users-list stack."""

from __future__ import annotations

import inspect

import pytest
from fastapi import HTTPException

from dashboard.backend.api.routers import admin_analytics
from dashboard.backend.domain.analytics import query_service


def test_an_unrecognised_exception_is_logged_before_it_becomes_a_503(capsys):
    """A bad SQL statement and an exhausted pool used to be indistinguishable
    in prod logs: the router had no print and raised `from None`."""
    with pytest.raises(HTTPException) as info:
        admin_analytics._raise_service_error(RuntimeError("secret-canary"))

    printed = capsys.readouterr().out
    assert info.value.status_code == 503
    assert isinstance(info.value.__cause__, RuntimeError)
    assert "ERROR: admin_analytics.unhandled category=RuntimeError" in printed
    assert "secret-canary" not in printed
    # The display-safe detail is unchanged.
    assert info.value.detail == "Analytics is temporarily unavailable."


@pytest.mark.parametrize(
    "exc,status",
    [(LookupError("missing"), 404), (ValueError("bad"), 422)],
)
def test_recognised_exceptions_keep_their_quiet_mapping(capsys, exc, status):
    with pytest.raises(HTTPException) as info:
        admin_analytics._raise_service_error(exc)

    assert info.value.status_code == status
    assert info.value.__cause__ is None
    assert capsys.readouterr().out == ""


def test_the_dead_users_list_stack_is_gone():
    assert not hasattr(admin_analytics, "_user_filters")
    assert not hasattr(admin_analytics, "_USER_SORTS")
    assert not hasattr(query_service, "AnalyticsUserFilters")
    assert not hasattr(query_service, "PaginatedUsers")
    assert not hasattr(query_service.AnalyticsQueryService, "list_users")
    assert "AnalyticsUserFilters" not in query_service.__all__
    assert "PaginatedUsers" not in query_service.__all__
    # The live users route is untouched: it still answers from the value stack.
    assert "_value_user_filters" in inspect.getsource(admin_analytics)
```

In `dashboard/backend/tests/test_event_loop_threadpool.py`, add to `BLOCKING_IO_ROUTER_MODULES` after `"dashboard.backend.api.routers.admin_users",`:

```python
    "dashboard.backend.api.routers.admin_analytics",
```

(`test_blocking_io_routers_have_no_async_handlers` then checks every handler in the module is a plain `def` — all nine are today — and `test_covered_modules_actually_serve_routes` checks the module still serves routes.)

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest dashboard/backend/tests/test_admin_analytics_hygiene.py dashboard/backend/tests/test_event_loop_threadpool.py -v
```

Expected: `test_an_unrecognised_exception_is_logged_before_it_becomes_a_503` FAILS at `isinstance(info.value.__cause__, RuntimeError)` (the cause is `None` today) and at the printed line; `test_the_dead_users_list_stack_is_gone` FAILS at `not hasattr(admin_analytics, "_user_filters")`; the threadpool tests PASS already (the module's handlers are sync) — the registration is a pin, not a fix.

- [ ] **Step 3: Log the unhandled arm**

In `dashboard/backend/api/routers/admin_analytics.py`, replace:

```python
def _raise_service_error(exc: Exception) -> Never:
    if isinstance(exc, LookupError):
        raise HTTPException(status_code=404, detail=_NOT_FOUND_DETAIL) from None
    if isinstance(exc, (ValidationError, ValueError)):
        raise HTTPException(status_code=422, detail=_INVALID_QUERY_DETAIL) from None
    raise HTTPException(status_code=503, detail=_UNAVAILABLE_DETAIL) from None
```

with:

```python
def _raise_service_error(exc: Exception) -> Never:
    if isinstance(exc, LookupError):
        raise HTTPException(status_code=404, detail=_NOT_FOUND_DETAIL) from None
    if isinstance(exc, (ValidationError, ValueError)):
        raise HTTPException(status_code=422, detail=_INVALID_QUERY_DETAIL) from None
    # Category only, never the message: a psycopg OperationalError carries the
    # DSN and a ValidationError carries field values. The class name is what
    # tells "the pool is exhausted" from "a bad SQL statement" in prod logs,
    # which the bare `from None` 503 never could (design D24, SS4.4).
    print(f"ERROR: admin_analytics.unhandled category={type(exc).__name__[:80]}")
    raise HTTPException(status_code=503, detail=_UNAVAILABLE_DETAIL) from exc
```

The two recognised arms keep `from None`: a 404 or 422 is the caller's fault and needs no server-side trace.

- [ ] **Step 4: Delete the dead users-list stack**

`dashboard/backend/api/routers/admin_analytics.py`:
- In the `from dashboard.backend.domain.analytics.query_service import (...)` block, delete the line `AnalyticsUserFilters,`.
- Delete the constants `_USER_SORTS = {...}` and `_SORT_ORDERS = {...}` (lines 54-55). Keep `_USER_STATES`: `_value_user_filters` still validates `legacy_status` against it (line 248) — that is the live `/users` route's `status` parameter, which D20 removes in **PR B**, not here.
- Delete the whole `_user_filters` function (from `def _user_filters(request: Request) -> tuple[AnalyticsUserFilters, int, int]:` through its `return filters, limit, offset`).

`dashboard/backend/domain/analytics/query_service.py`:
- Delete `_USER_STATES = {"blocked", "needs_attention", "dormant", "onboarding", "active"}` (line 35; its only remaining reader was `AnalyticsUserFilters.validate_status`). Keep `_ATTENTION_STATES`.
- Delete `class AnalyticsUserFilters(BaseModel): ...` (through the end of `validate_activity_range`) and `class PaginatedUsers(BaseModel): ...`.
- Delete `AnalyticsQueryService.list_users` (from `def list_users(` through `return PaginatedUsers(... offset=offset,)`).
- Remove `"AnalyticsUserFilters",` and `"PaginatedUsers",` from `__all__`.
- Then `python -c "import dashboard.backend.domain.analytics.query_service"` and `rg -n "AnalyticsUserFilters|PaginatedUsers|_user_filters\b|_USER_SORTS|_SORT_ORDERS" dashboard/backend/` — the second must return only the two test files edited next.

`dashboard/backend/tests/test_admin_analytics_api.py`: remove `AnalyticsUserFilters,` from the import block (line 20); in the test around line 328, delete:

```python
    listing = service.list_users(
        filters=AnalyticsUserFilters(),
        limit=25,
        offset=0,
        now=NOW,
    )
```

and its two assertions `assert listing.total == 1` and `assert listing.items[0].user_id == 1`.

`dashboard/backend/tests/domain/analytics/test_repository_contract.py`: change the import to `from dashboard.backend.domain.analytics.query_service import AnalyticsQueryService`; in `assert_pr2_query_contract` delete:

```python
    users = service.list_users(
        filters=AnalyticsUserFilters(),
        limit=10,
        offset=0,
        now=NOW,
    )
```

and its two assertions `assert users.total == 1` and `assert users.items[0].status == "active"`.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
python -m pytest dashboard/backend/tests/test_admin_analytics_hygiene.py dashboard/backend/tests/test_event_loop_threadpool.py dashboard/backend/tests/test_admin_analytics_api.py dashboard/backend/tests/domain/analytics/test_repository_contract.py dashboard/backend/tests/test_app_composition.py dashboard/backend/tests/test_admin_analytics_frontend.py dashboard/backend/tests/test_admin_analytics_value_frontend.py -v
TEST_POSTGRES_URL=postgresql://postgres:atl@localhost:55432/postgres \
  python -m pytest dashboard/backend/tests/domain/analytics/test_repository_postgres.py -q
```

Expected: PASS. `test_app_composition.py`'s frozen `(method, path, name)` triples for `admin_analytics` are unchanged — proof that nothing routable moved.

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/api/routers/admin_analytics.py \
        dashboard/backend/domain/analytics/query_service.py \
        dashboard/backend/tests/test_event_loop_threadpool.py \
        dashboard/backend/tests/test_admin_analytics_api.py \
        dashboard/backend/tests/domain/analytics/test_repository_contract.py \
        dashboard/backend/tests/test_admin_analytics_hygiene.py
git status --short
git commit -m "fix: log unhandled analytics errors and delete the dead users-list stack"
```

---

### Task 14: Pin the read budget and the event-log discipline

**Files:**
- Create: `dashboard/backend/tests/domain/analytics/test_read_budget.py`
- Modify: `dashboard/backend/tests/test_architecture_boundaries.py` — append after `test_portfolio_manager_construction_always_declares_settlement` (line 552)

**Interfaces:**
- Consumes: everything above; `CountingSpy`/`SpyBundle` (Task 7).
- Produces: nothing. This is the guard that keeps the architecture true (design §6.11, §6.12).

Carried from the superseded 2026-09-12 plan's Task B10 (`git show c3bbf2ed:...`, lines 3610-3745) with two amendments from design §12: the budget spies **every** store the job touches — analytics base, value store, credits, run history, agents, providers, protocol runs — not the analytics store alone, because §6.9 moved the reads onto the owning domains and a single spy could not see them (§6.12); and it bounds **wall-clock** as well as query count (§12 item 5), so a step that is one query but scans an unindexed table or hides a quadratic fold is caught. The discipline test gains rule 7 (the analytics package opens no connection but its own).

- [ ] **Step 1: Write the budget tests**

Create `dashboard/backend/tests/domain/analytics/test_read_budget.py`:

```python
"""The architectural claim of the design, stated as tests (design SS6.12).

Every store the daily job touches is wrapped in a counting spy and the job is
driven against 200 and then 400 synthetic users. If the query count moves,
someone has put a user id into a query inside the job; if the wall-clock ratio
moves, a constant-count step hides a scan that grows with the population.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import date, datetime, time as clock_time, timedelta, timezone

import pytest
from cryptography.fernet import Fernet

from dashboard.backend.database import BacktestDatabase
from dashboard.backend.domain.agents.repository import AgentStore
from dashboard.backend.domain.analytics.daily_facts import run_daily_facts
from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.service import AnalyticsService
from dashboard.backend.domain.analytics.value_repository import (
    build_value_analytics_store,
)
from dashboard.backend.domain.brokers import repository as broker_repository
from dashboard.backend.domain.credits.repository import CreditsStore
from dashboard.backend.domain.model_providers.repository import ModelProviderStore
from dashboard.backend.domain.runs.repository import RunStore
from dashboard.backend.tests.domain.analytics._store_spies import CountingSpy, SpyBundle
from dashboard.backend.tests.test_credits_ledger_aggregates import _insert, _ledger_row
from dashboard.backend.users import UserStore


D = date(2026, 9, 11)
NOW = datetime(2026, 9, 12, 0, 5, tzinfo=timezone.utc)

# The constant the daily job commits to, per claimed tick, in public store-method
# calls (the unit CountingSpy measures). Task 9's table in the PR A plan is the
# source of this number; if it moves, either the job or the table drifted, and
# the table is authoritative until the design changes.
#   value store   12  claim, aggregate_events, record_activity_batch x2,
#                     list_operational_signals, list_activity, sum_recent_facts,
#                     upsert_daily_facts, list_facts_for_date,
#                     append_lifecycle_transitions, list_days_needing_recompute,
#                     complete_projection_day
#   analytics      2  list_excluded_user_ids (inside rollup_day), list_daily_subjects
#   credits        3  aggregate_ledger_for_day, get_balance_projections,
#                     list_account_billing_states
#   providers      3  list_default_credential_facts, list_all_providers,
#                     list_platform_credential_statuses
#   agents         1  list_agent_owners
#   protocol runs  1  list_terminal_runs_since
#   run history    1  aggregate_operator_cost_for_day
DAILY_JOB_CLAIMED_TICK_CALLS = 23
WALL_CLOCK_BUDGET_SECONDS_AT_200 = 20.0
WALL_CLOCK_RATIO_LIMIT = 2.5
# A ratio over tiny absolute times is noise, not a scan; floor the denominator.
WALL_CLOCK_FLOOR_SECONDS = 0.25


@pytest.fixture(autouse=True)
def _encryption_key(monkeypatch):
    monkeypatch.setenv("BROKER_TOKEN_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setattr(broker_repository, "_fernet_instance", None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("COMMONSTACK_API_KEY", raising=False)


class _NoRetention:
    def run_if_due(self):
        return None


class _DailyFixture:
    def __init__(self, path, *, users):
        path.mkdir(parents=True, exist_ok=True)
        self.path = path / "budget.db"
        UserStore(db_path=self.path)
        with sqlite3.connect(self.path) as conn:
            conn.executemany(
                "INSERT INTO users (id, email, display_name, password_hash, role, "
                "user_group, created_at) VALUES (?, ?, ?, 'x', 'user', 'unknown', ?)",
                [
                    (index, f"user{index}@example.test", f"User {index}",
                     (NOW - timedelta(days=60)).isoformat())
                    for index in range(1, users + 1)
                ],
            )
        self.user_ids = list(range(1, users + 1))
        analytics = AnalyticsStore(self.path)
        credits = CreditsStore(self.path)
        providers = ModelProviderStore(self.path)
        agents = AgentStore(self.path)
        protocol_runs = RunStore(self.path)
        run_history = BacktestDatabase(path / "runs.db")
        self.spies = SpyBundle(
            analytics=CountingSpy(analytics, "analytics"),
            credits=CountingSpy(credits, "credits"),
            providers=CountingSpy(providers, "providers"),
            agents=CountingSpy(agents, "agents"),
            runs=CountingSpy(protocol_runs, "runs"),
            run_history=CountingSpy(run_history, "run_history"),
        )
        real_value_store = build_value_analytics_store(
            self.spies.analytics,
            credits_base=self.spies.credits,
            provider_base=self.spies.providers,
            agent_base=self.spies.agents,
            run_base=self.spies.runs,
        )
        self.value_store = CountingSpy(real_value_store, "value_store")
        self.spies.spies["value_store"] = self.value_store
        # Seed through the real (unspied) stores so setup is not counted.
        service = AnalyticsService(analytics, value_store=real_value_store, maintain_activity=True)
        day_start = datetime.combine(D, clock_time(9, 0), tzinfo=timezone.utc)
        for user_id in self.user_ids:
            service.record_server_event(
                event_name="backtest_requested",
                user_id=user_id,
                source_event_id=f"run:backtest_requested:{user_id}",
                source_record_type="run",
                source_record_id=f"run-{user_id}",
                occurred_at=day_start + timedelta(seconds=user_id),
                received_at=day_start + timedelta(seconds=user_id + 1),
            )
            if user_id % 5 == 0:
                service.record_server_event(
                    event_name="backtest_completed",
                    user_id=user_id,
                    source_event_id=f"run:backtest_completed:{user_id}",
                    source_record_type="run",
                    source_record_id=f"run-{user_id}",
                    occurred_at=day_start + timedelta(minutes=1, seconds=user_id),
                    received_at=day_start + timedelta(minutes=1, seconds=user_id + 1),
                )
            if user_id % 2 == 0:
                agents.create_agent(name=f"Agent {user_id}", owner_user_id=user_id)
        _insert(
            self.path,
            "credit_ledger_entries",
            [
                _ledger_row(user_id, "admin_grant_assign", 1_000_000, NOW - timedelta(days=2), key=f"g{user_id}")
                for user_id in self.user_ids
                if user_id % 3 == 0
            ],
        )
        self.spies.reset()

    def run(self, *, now=NOW):
        return run_daily_facts(
            now=now,
            value_store=self.value_store,
            run_history_store=self.spies.run_history,
            retention=_NoRetention(),
        )


def _timed(action):
    started = time.perf_counter()
    result = action()
    return result, time.perf_counter() - started


def test_the_daily_job_costs_the_same_at_any_user_count(tmp_path):
    """Doubling the population changes nothing about the query count.

    This is the architectural claim of the whole design, stated as a test.
    If it ever fails, someone has put a user id into a query inside the job.
    """
    small = _DailyFixture(tmp_path / "small", users=200)
    large = _DailyFixture(tmp_path / "large", users=400)

    small_report = small.run()
    large_report = large.run()

    assert small_report.claimed and large_report.claimed
    assert small_report.partial is False and large_report.partial is False
    assert small_report.users_written == 200
    assert large_report.users_written == 400
    assert small.spies.total_calls == large.spies.total_calls, (
        small.spies.calls_by_store(),
        large.spies.calls_by_store(),
    )
    assert small.spies.total_calls == DAILY_JOB_CLAIMED_TICK_CALLS, small.spies.calls_by_store()
    assert small.spies.calls_with_scalar_user_id == []
    assert large.spies.calls_with_scalar_user_id == []


def test_the_daily_job_stays_inside_the_wall_clock_budget(tmp_path):
    """A constant query count can still hide a scan (design SS12 item 5)."""
    small = _DailyFixture(tmp_path / "small", users=200)
    large = _DailyFixture(tmp_path / "large", users=400)

    _report, small_elapsed = _timed(small.run)
    _report, large_elapsed = _timed(large.run)

    assert small_elapsed < WALL_CLOCK_BUDGET_SECONDS_AT_200, small_elapsed
    assert large_elapsed <= WALL_CLOCK_RATIO_LIMIT * max(small_elapsed, WALL_CLOCK_FLOOR_SECONDS), (
        small_elapsed,
        large_elapsed,
    )


def test_an_idle_tick_costs_one_query(tmp_path):
    """Most ticks in a day do nothing. They must cost (almost) nothing."""
    fixture = _DailyFixture(tmp_path, users=50)
    fixture.run()
    fixture.spies.reset()

    for tick in range(60):
        report = fixture.run(now=NOW + timedelta(minutes=5 * tick))
        assert report.claimed is False

    assert fixture.spies.total_calls == 60
    assert fixture.spies.calls_by_store()["value_store"] == 60
    assert all(
        count == 0
        for name, count in fixture.spies.calls_by_store().items()
        if name != "value_store"
    )
```

- [ ] **Step 2: Write the discipline guard**

Append to `dashboard/backend/tests/test_architecture_boundaries.py` (it already imports `ast`, `pytest` and defines `_BACKEND`, `_REPO_ROOT`):

```python
# ---------------------------------------------------------------------------
# Event-log discipline (admin layer redesign design doc SS6.11, rules 1-7)
# ---------------------------------------------------------------------------

_ANALYTICS = _BACKEND / "domain" / "analytics"
_EVENT_READ_NAMES = {"list_events", "list_user_events", "list_metric_events"}
# Files that may call a raw-event reader, each with the reason. Rules 3-5: the
# paginated timeline and the overview's one-day scan are the only permitted raw
# reads; everything else reads rollups or user_daily_facts. Every entry must
# still contain a hit (the stale check below), so PR B removes the last two as
# it deletes the files and narrows the others as it moves the reads.
_EVENT_READ_ALLOWLIST = {
    "dashboard/backend/domain/analytics/rollups.py": (
        "defines AnalyticsRollupStore.list_events; rollup_day / rollup_current_day "
        "are the day rollups and the overview's current-day scan (rule 5), "
        "narrowed to one UTC day in PR B"
    ),
    "dashboard/backend/domain/analytics/query_service.py": (
        "the paginated timeline (rule 3) and the overview's current-day read "
        "(rule 5); PR B narrows the overview read and gives sessions a 30-day window"
    ),
    "dashboard/backend/domain/analytics/value_queries.py": (
        "the retention grid's per-activation-week scan and the one-user profile "
        "scan; PR B moves both onto user_daily_facts / user_activity"
    ),
    "dashboard/backend/domain/analytics/states.py": (
        "legacy five-state calculator: full-history per-user reads. DELETED IN "
        "PR B (design SS13 row B); allowlisted, not fixed, because PR A creates "
        "and PR B drops"
    ),
    "dashboard/backend/domain/analytics/lifecycle_backfill.py": (
        "historical eight-week reconstruction, per-user history reads. DELETED "
        "IN PR B; the copy in facts_migration.py carries the history now"
    ),
}
_OWN_CONNECTION_RECEIVERS = {"self", "self.analytics_base", "self.base_store"}
_OWN_DIALECT_RECEIVERS = {
    "base_store",
    "self.base_store",
    "analytics_base",
    "self.analytics_base",
    "resolved_analytics_base",
}
# Rule 7 tolerates nothing after PR A. A future entry needs a file path and a
# reason, exactly like _EVENT_READ_ALLOWLIST -- and a design-doc amendment,
# because the rule it relaxes is SS6.14's first row.
_CROSS_DOMAIN_READ_ALLOWLIST: dict[str, str] = {}
_PER_USER_QUERY_PREFIXES = (
    "aggregate_", "list_", "upsert_", "record_", "sum_", "claim_",
    "complete_", "release_", "get_", "append_", "copy_", "seed_",
)


def _analytics_sources():
    for path in sorted(_ANALYTICS.glob("*.py")):
        yield path.relative_to(_REPO_ROOT).as_posix(), ast.parse(
            path.read_text(encoding="utf-8")
        )


def _dotted(node) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def test_event_log_rule_1_and_2_constants_hold():
    from dashboard.backend.domain.analytics import models, retention

    assert models.MAX_PROPERTIES_BYTES == 1024
    assert isinstance(models.ALLOWED_SERVER_EVENT_NAMES, (set, frozenset))
    assert retention.RAW_EVENT_RETENTION_DAYS == 180
    mutators = []
    for relative, tree in _analytics_sources():
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"add", "update", "discard", "remove"}
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id.startswith("ALLOWED_")
            ):
                mutators.append((relative, node.lineno))
    assert mutators == [], f"the event allowlist is mutated at runtime: {mutators}"


def test_raw_event_reads_stay_inside_the_allowlist():
    """No new caller may read the event log (rules 3, 4, 5).

    On 2026-09-11 a maintenance sweep read every user's 180-day history
    every fifteen minutes, exhausted a 5 GB monthly egress allowance, and
    500'd login for five hours. Nothing in the suite noticed, because every
    behavioural assertion still passed. This is the assertion that would have.
    """
    hits: dict[str, list[tuple[str, int]]] = {}
    for relative, tree in _analytics_sources():
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _EVENT_READ_NAMES
            ):
                hits.setdefault(relative, []).append((node.func.attr, node.lineno))
    unapproved = {path: calls for path, calls in hits.items() if path not in _EVENT_READ_ALLOWLIST}
    assert unapproved == {}, f"unapproved raw-event reads: {unapproved}"
    stale = sorted(set(_EVENT_READ_ALLOWLIST) - set(hits))
    assert stale == [], (
        "allowlist entries with no raw-event read left in them -- delete the entry "
        f"so the exemption cannot be inherited by the next reader: {stale}"
    )


def test_the_daily_job_never_loops_over_users():
    """A for-loop issuing a query per user is the shape that caused the outage (rule 6)."""
    source = (_ANALYTICS / "daily_facts.py").read_text(encoding="utf-8")
    offenders = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.For, ast.AsyncFor)):
            continue
        # The loop's own iterable may be one set-based read; its body may not
        # issue any.
        for statement in node.body:
            for inner in ast.walk(statement):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr.startswith(_PER_USER_QUERY_PREFIXES)
                ):
                    offenders.append((inner.func.attr, inner.lineno))
    assert offenders == [], f"store calls inside a loop in daily_facts.py: {offenders}"


def test_the_analytics_package_opens_only_its_own_connection():
    """Rule 7 (design SS6.11, SS6.14 row 1): `_get_connection()` is called only
    on the analytics domain's own store, and no analytics module sniffs another
    store's dialect. Other domains' tables are read through public methods on
    the store that owns them -- CreditsStore.aggregate_ledger_for_day, not SQL
    against credit_ledger_entries from inside this package."""
    offenders = []
    for relative, tree in _analytics_sources():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute) and node.func.attr == "_get_connection":
                receiver = _dotted(node.func.value)
                if receiver not in _OWN_CONNECTION_RECEIVERS:
                    offenders.append((relative, node.lineno, f"{receiver}._get_connection()"))
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "hasattr"
                and len(node.args) == 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == "database_url"
            ):
                receiver = _dotted(node.args[0])
                if receiver not in _OWN_DIALECT_RECEIVERS:
                    offenders.append((relative, node.lineno, f'hasattr({receiver}, "database_url")'))
    unlisted = [entry for entry in offenders if entry[0] not in _CROSS_DOMAIN_READ_ALLOWLIST]
    assert unlisted == [], f"cross-domain connection or dialect sniff in domain/analytics: {unlisted}"
    stale = sorted(set(_CROSS_DOMAIN_READ_ALLOWLIST) - {entry[0] for entry in offenders})
    assert stale == [], f"stale rule-7 allowlist entries: {stale}"
```

- [ ] **Step 3: Run both files and prove the guards have teeth**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_read_budget.py dashboard/backend/tests/test_architecture_boundaries.py -v
```

Expected: PASS. Then, one at a time, make each guard fail and restore it:

1. In `daily_facts.py`'s `facts_step`, add `store.get_activity(user_id)` as the first line of the `for subject in subjects:` body. Re-run: `test_the_daily_job_costs_the_same_at_any_user_count` fails (`total_calls` differs by the population and `calls_with_scalar_user_id` lists `value_store.get_activity`), and `test_the_daily_job_never_loops_over_users` fails naming `get_activity`. Remove the line.
2. In `value_repository.py`'s `list_credit_activity`, temporarily replace the body with the pre-Task-5 version that opens `self.credits_base._get_connection()`. Re-run: `test_the_analytics_package_opens_only_its_own_connection` fails naming `self.credits_base._get_connection()`. Restore.
3. In `daily_facts.py`, add `store.analytics_base.list_user_events(1)` anywhere. Re-run: `test_raw_event_reads_stay_inside_the_allowlist` fails naming `daily_facts.py`. Remove.

- [ ] **Step 4: Run the whole suite on both tiers, then commit**

```bash
python -m pytest dashboard/backend/tests/ -q
TEST_POSTGRES_URL=postgresql://postgres:atl@localhost:55432/postgres \
  python -m pytest dashboard/backend/tests/ -q
git add dashboard/backend/tests/domain/analytics/test_read_budget.py \
        dashboard/backend/tests/test_architecture_boundaries.py
git status --short
git commit -m "test: pin the user-count-independent read budget and the event-log discipline"
```

Expected: PASS on both, with a lower skip count on the second. The `git status --short` before the commit is there so the seed `backtest.db` (rewritten by the full-suite run) is seen and **not** staged.

---

## Not in this PR

Per design §13 row A's "Must not" column, D9, D18/§10.1, §12 item 7 and §13 row B:

- **Any response-shape change.** The nine `/api/admin/analytics/*` routes answer byte-identically to `main`; `test_admin_analytics_api.py`, `test_admin_analytics_frontend.py` and `test_admin_analytics_value_frontend.py` are edited only to delete the dead stack's own construction (Task 13). *Reason (D18):* the existing wired frontend and its tests are a free conformance oracle while the data model underneath is replaced; a moved number has exactly one suspect.
- **Dropping any table or column**, including `user_analytics_snapshots`' value columns, `user_lifecycle_daily_snapshots`, `states.py`, `lifecycle_backfill.py`, the snapshot repair pass, `run_analytics_maintenance` and the five-state vocabulary. *Reason (§12 item 7):* the 09-12 plan dropped the snapshot tables in its data-model PR while its read-paths PR was the one that stopped reading them, so prod would have read dropped tables in between. PR B rewires every read path and then drops, in the same PR, after PR A's job has written at least eight days of facts.
- **Any frontend file**, and therefore any `?v=` bump. *Reason (§13 row A; D8/D19):* the page was rebuilt in PR C, which ships before this PR (§13, re-ordered 2026-09-16), against today's contract; the one re-cut is PR D.
- **A `cohort` column, validator, filter or suggestion list** (the 09-12 plan's Task B2 is struck in full). *Reason (D9):* two categorical labels on one user is two owners of one fact; `user_group` shipped in PR #465 and is the sole third axis.
- **`resolve_group_badge`** (the 09-12 plan's Task B5 step 3). *Reason (D11, §13 row B):* nothing computes it today and PR B adds it beside the read paths that consume it; PR D puts `group_badge` on the payloads.
- **Wiring `build_lifecycle_inputs` into any route.** *Reason (§13 row A):* "read-time lifecycle calculator (not yet wired to routes)"; the route move is PR B's.
- **Re-pointing the retention sweep at `user_daily_facts` / `lifecycle_transitions`, and the long-term rollups by tier and `user_group`** (the 09-12 plan's B9 step 5 and C4). *Reason (§13 row B):* they land with the read-path move; no fact row reaches 180 days before PR B.
- **Registering the daily job with `register_reaper_sweep`** (the 09-12 plan's B8 step 5). *Reason (D23):* the heartbeat thread must not carry a whole-population batch.
- **The 09-12 plan's `aggregate_operator_cost_for_day` / `aggregate_ledger_for_day` on `ValueAnalyticsStore`** (B8 step 3). *Reason (§12 item 1, D23):* they are methods on `BacktestDatabase` / `PostgresBacktestDatabase` and `CreditsStore` / `PostgresCreditsStore` (Task 5).
- **`max_active_dashboard_backtests` on `GET /api/admin/stats`**, `billing_lane_mix`, `top_operational_reasons`, `purchased_by_day`, `get_user_metrics`. *Reason (§9, §13 rows B/D):* PR B computes, PR D exposes (`max_active_dashboard_backtests` is PR C's, already merged).

## Acceptance

- [ ] `user_activity`, `user_daily_facts` (with `user_group NOT NULL DEFAULT 'unknown'` and nullable `operational_reason_code`, no `cohort`) and `lifecycle_transitions` (`UNIQUE (user_id, snapshot_date)`, `data_quality`) exist in both DDL constants and pass the twin-parity column check (Task 1).
- [ ] `agent_runs.owner_user_id` exists on both run-history twins, is written from the authenticated caller through the four hops, and stays NULL for callerless runs (Task 2).
- [ ] Ingestion maintains `user_activity` with one MIN/MAX upsert per accepted lifecycle event and reads no history; the live singleton has `maintain_activity=True` and `project_snapshots=False`; `seed_activity_from_snapshots` copies `activated_at` / `last_meaningful_activity_at` idempotently and never regresses a row (Task 3; design §6.5, §11, §12 item 6).
- [ ] `build_lifecycle_inputs` clamps evidence to `as_of`, prefers the day's own activity, and is called by no route (Task 4; §6.6).
- [ ] `aggregate_commercial_ledger`, `list_credit_activity_timestamps`, `aggregate_ledger_for_day`, `list_account_billing_states` exist on both credits twins; `aggregate_operator_cost_for_day` on both run-history twins; `list_commercial_values` / `list_credit_activity` return byte-identical results through them (Task 5; §12 item 1, D23).
- [ ] `backfill.py` contains no `_get_connection` and no `hasattr(..., "database_url")`; its entry in `_DIALECT_BRANCH_ALLOWLIST` is gone; `list_existing_source_event_ids`, `list_agent_source_rows`, `list_llm_reservation_rows`, `list_llm_usage_rows` exist on both twins of their owning stores (Task 6).
- [ ] `list_operational_signals(ids, now=, population_wide=True)` agrees with `get_operational_facts` for every seeded state, issues the same seven owning-store calls at 10 and 40 users, and none takes a scalar user id; `consecutive_failed_terminal_runs` has one owner (Task 7; §6.9 step 5).
- [ ] The day claim is a two-field lease: double claims refuse, a two-hour-old `running` lease is reclaimable, a completed cursor is never re-run, a released day retries at once (Task 8; §6.9 "Due check").
- [ ] `run_daily_facts` writes one row per eligible user per day with every §6.7 column, excludes admins, excluded users and accounts created after D, survives a user active after midnight, counts D in the trailing window, appends and corrects transitions with `DO UPDATE`, marks failed-step days `partial` and retries them, recomputes late-arrival days, and runs the retention coordinator every tick (Task 9; §6.9).
- [ ] Eight weeks of `user_lifecycle_daily_snapshots` are copied into `user_daily_facts` as `partial` rows with `user_group` from `users`, idempotently and without overwriting a complete row (Task 10; §6.9 "Migration").
- [ ] The job runs on `analytics-daily-facts`, its own daemon thread with `ANALYTICS_DAILY_JOB_INTERVAL_SECONDS` (default 300, junk-tolerant) and a stop event; `app.py` starts it beside the reaper; the retention coordinator is no longer a reaper sweep; `maintenance.py` keeps only the two throttled repairs and stays registered on the reaper (Task 11; D23, §6.9 steps 1 and 8).
- [ ] A retention sweep past the horizon leaves `user_activity` row count and timestamps unchanged, and `retention.py` never names the table (Task 12; §11).
- [ ] `_raise_service_error` prints `ERROR: admin_analytics.unhandled category=<Class>` and raises `from exc`; `admin_analytics` is in `BLOCKING_IO_ROUTER_MODULES`; `_user_filters`, `AnalyticsUserFilters`, `PaginatedUsers`, `AnalyticsQueryService.list_users` are deleted; the route set is unchanged (Task 13; D24).
- [ ] The read budget holds: 23 store calls per claimed tick at 200 and at 400 users, zero user-id-parameterised calls, 200-user wall-clock under 20 s, 400-user wall-clock at most 2.5× the 200-user run (floored at 0.25 s), one call per idle tick; rules 1–7 of §6.11 are pinned in `test_architecture_boundaries.py` with explicit, non-stale allowlists naming `states.py` and `lifecycle_backfill.py` as PR B deletions (Task 14; §6.11, §6.12).
- [ ] No response shape changed; no table or column dropped; no frontend file touched; no `cohort` added; every existing read path still serves from the old tables.
- [ ] CI's Postgres tier is green on this branch before merge; every new store method exists on both twins.

## Self-review notes

- **Spec coverage.** §6.1/6.2 → Tasks 3, 4, 9 (no recompute path; `user_group` as the third axis on the fact row). §6.3 → Task 9's "as of end of D" contract and Task 4's `as_of`. §6.4 → Task 2 (`owner_user_id`), Task 9 (`list_daily_subjects` joins `users`). §6.5 → Task 3. §6.6 → Task 4. §6.7 → Tasks 1, 9 (every column incl. `operational_reason_code`, `user_group`; `runs_failed` a bare count). §6.8 → Tasks 1, 9 (`DO UPDATE`, `data_quality`). §6.9 → Tasks 8, 9, 10, 11 (own thread, two-field claim, eight steps, migration timing). §6.11 → Task 14 (rules 1–7). §6.12 → Tasks 7, 14 (every store spied, wall-clock). §6.14 rows "PR A" → Tasks 5, 6, 11, 13. §6.15 → Tasks 3 and 11 leave PR 0's switched-off paths for PR B. §11 → Tasks 3, 10, 12. §12 items 1 (Tasks 5, 6), 2 (every twin task), 4 (Task 11), 5 (Task 14), 6 (Tasks 3, 10), 7 (nothing dropped; "Not in this PR"). §13 row A → the task list is exactly its "Content" cell; its "Must not" cell is the first four Global Constraints and is re-checked in Acceptance. D21 (this is "the rewrite"), D22 (PR T assumed merged; both twins in every store task), D23 (Tasks 5, 6, 11), D24 (Task 13). §4.3–4.5 were used as the starting inventory (14 `ValueAnalyticsStore` methods → PR T's twin; `analytics_projection_jobs` already exists → Task 8 extends it). §10.1 → Global Constraints and Task 13's route-set check.
- **Placeholder scan.** Every code block is the literal file content before or after the change, transcribed from the worktree at `c3bbf2ed` (or, for the four files PR 0 / PR T rewrite, from those plans' final text, which is stated where it applies). No `TBD`, `TODO`, "similar to Task N", "as in the old plan" or elided body; the one place a first draft of Task 9 referenced a helper that did not exist was rewritten so the module shown is the module to write. Every 09-12 task carried forward is cited by `git show c3bbf2ed:...` and line range, never as a live file. Every referenced function either exists in the current source (`rollup_day`, `analytics_retention_coordinator`, `calculate_lifecycle`, `calculate_operational_state`, `commercial_tier`, `coerce_user_group`, `get_balance_projections`, `list_all_providers`, `restrict_account`, `create_user_credential`, `upsert_platform_credential`, `create_agent`, `create_run`, `insert_run`, `_positive_integer`, `positive_user_id`, `utc_iso`, `_row_value`, `_timestamp`, `_optional_timestamp`, `_object_value`, `_ids`, `_validate_window`, `_projection_job_name`, `disable_synchronous_projection` (PR 0), `build_value_analytics_store` (PR T)) or is defined in an earlier task of this plan and named there.
- **Name consistency.** Checked across tasks: `record_activity` / `record_activity_batch` / `get_activity` / `list_activity` / `seed_activity_from_snapshots` (Tasks 3, 9, 10, 12); `RecentFactTotals` / `sum_recent_facts` / `build_lifecycle_inputs` (Tasks 4, 9); `aggregate_commercial_ledger` / `list_credit_activity_timestamps` / `aggregate_ledger_for_day` / `list_account_billing_states` / `list_llm_reservation_rows` / `list_llm_usage_rows` (Tasks 5, 6, 7, 9); `aggregate_operator_cost_for_day` (Tasks 5, 9, 14); `list_existing_source_event_ids` / `list_agent_source_rows` (Task 6); `DefaultCredentialFacts` / `list_default_credential_facts` / `list_platform_credential_statuses` / `list_agent_owners` / `list_terminal_runs_since` / `consecutive_failed_terminal_runs` / `list_operational_signals(..., population_wide=)` (Tasks 7, 9); `claim_projection_day` / `complete_projection_day` / `release_projection_day` (Tasks 8, 9); `DayEventTotals` / `ActivityUpdate` / `UserDailyFact` / `LifecycleTransitionRow` / `aggregate_events_for_day` / `upsert_daily_facts` / `append_lifecycle_transitions` / `list_facts_for_date` / `list_days_needing_recompute` / `list_daily_subjects` / `run_daily_facts` / `DailyFactsReport` / `DAILY_FACTS_JOB` (Tasks 9, 10, 11, 14); `copy_daily_snapshot_history` / `migrate_lifecycle_history` / `run_startup_migrations` / `StartupMigrationReport` / `HISTORY_COPY_DAYS` (Tasks 10, 11); `daily_job_interval_seconds` / `start_daily_facts_worker` / `stop_daily_facts_worker` (Task 11); `CountingSpy` / `SpyBundle` (Tasks 7, 14); `_DIALECT_BRANCH_ALLOWLIST` (PR T; Tasks 5, 6); the printed line names `analytics.daily_facts.<step>_failed`, `analytics.daily_facts.tick_failed`, `analytics.facts_migration_failed`, `analytics.operational_signals_empty`, `analytics.activity_update_failed`, `admin_analytics.unhandled` (Tasks 3, 7, 9, 11, 13).
- **Constraint check.** Response shapes: only Tasks 5 and 13 touch code on a request path, and both are behaviour-preserving (verbatim SQL moved; dead code deleted); Task 14's rule-7 test and `test_app_composition.py` are the mechanical checks. Tables/columns: every DDL edit in this plan is `CREATE TABLE IF NOT EXISTS` / `ADD COLUMN`; no `DROP` appears anywhere. Frontend: no path under `dashboard/frontend/` is named in any Files block. Cohort: the word appears in this plan only in "Not in this PR", the D9 citations and `RetentionCohort.cohort_week`.
- **Verified against source, not assumed.** `credit_ledger_entries`'s CHECK constraint (Task 5's raw-insert fixture fills every column each `entry_type` inspects); `create_user_credential` refusing a disabled provider and `set_user_credential_status` clearing `is_default` off a non-verified row (Task 7's fixture order and raw UPDATE); `SEEDED_PROVIDERS` = openrouter, commonstack, openai, anthropic (Task 7 uses `anthropic`, not the 09-12 plan's `gemini`); `protocol_runs` timestamps in two shapes (Task 7's date-prefix bound + Python filter); `agent_runs.updated_at` as `CURRENT_TIMESTAMP` text on both twins (Task 5's text bounds); `get_balance_projections` returning rows only for users with ledger activity (why `list_operational_signals` answers for the given ids rather than for "users seen"); `AnalyticsStore.list_excluded_user_ids` already reading `users` (why `list_daily_subjects` on the analytics store is inside rule 7); `BROKER_TOKEN_ENCRYPTION_KEY` being the credential-vault key every model-provider test sets; `_environment_platform_secret` reading `OPENROUTER_API_KEY` / `COMMONSTACK_API_KEY`.
- **Open item.** The design doc names the retention coordinator as job step 8 ("if due"); this plan calls `run_if_due()` on every tick rather than only on a claimed one, so the coordinator's 60-second backlog retry keeps working. That is a reading of "if due" the design supports but does not spell out; recorded here rather than silently chosen.
