# PostgreSQL Attempt Index Migration Order Hotfix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the merged Credits provider-attempt release start against legacy PostgreSQL databases by creating `attempt_index` before any index references it.

**Architecture:** Keep `CREDITS_POSTGRES_DDL` responsible for creating the fresh schema and only indexes whose columns exist in that schema phase. Keep the existing idempotent migration block responsible for legacy columns, constraints, and the provider-attempt index. `PostgresCreditsStore._init_schema()` remains transactional and continues executing both blocks in order.

**Tech Stack:** Python 3.13, PostgreSQL, psycopg, pytest, GitHub Actions, Render.

**Spec:** `docs/superpowers/specs/2026-09-01-postgres-attempt-index-migration-order-design.md`

## Global Constraints

- Add `attempt_index` with `NOT NULL DEFAULT 0` to legacy `credit_llm_reservations` tables.
- Create `idx_credit_llm_reservations_run_status` only after `attempt_index` exists.
- Preserve all existing reservation and usage rows, constraint names, and index names.
- Keep fresh-schema and legacy-upgrade behavior idempotent and PostgreSQL-only.
- Do not execute production SQL, change Render settings, or include secrets, `backtest.db`, `.superpowers/`, or `work/` in the commit.

---

### Task 1: Add the failing migration-order regression

**Files:**
- Modify: `dashboard/backend/tests/domain/credits/test_repository_postgres.py:52-90`
- Modify: `dashboard/backend/tests/domain/credits/test_repository_postgres.py:161-229`
- Modify: `dashboard/backend/tests/domain/credits/test_repository_postgres.py:280-430`

**Interfaces:**
- Consumes: `pg_legacy_credits_url`, `LEGACY_CREDITS_POSTGRES_DDL`, `pg_module.PostgresCreditsStore`, and the existing `pg_only` marker.
- Produces: a static DDL-order assertion and a live PostgreSQL legacy-upgrade test that later implementation must satisfy.

- [ ] **Step 1: Add a static ordering assertion** after the existing provider-attempt schema assertions.

```python
def test_postgres_provider_attempt_index_is_created_after_its_column():
    base_ddl = pg_module.CREDITS_POSTGRES_DDL
    migration_ddl = pg_module.CREDITS_POSTGRES_GRANT_MIGRATION_DDL

    assert "idx_credit_llm_reservations_run_status" not in base_ddl
    add_column = migration_ddl.index(
        "ADD COLUMN IF NOT EXISTS attempt_index INTEGER NOT NULL DEFAULT 0"
    )
    create_index = migration_ddl.index("idx_credit_llm_reservations_run_status")
    assert add_column < create_index
```

- [ ] **Step 2: Extend the legacy fixture with the old reservation table** before the fixture creates the store.

```sql
CREATE TABLE credit_llm_reservations (
    reservation_id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    run_id TEXT NOT NULL,
    call_index INTEGER NOT NULL,
    reserved_micro BIGINT NOT NULL,
    reserved_grant_micro BIGINT NOT NULL,
    reserved_purchased_micro BIGINT NOT NULL,
    settled_micro BIGINT NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'open',
    operation_key TEXT NOT NULL UNIQUE,
    request_digest TEXT NOT NULL,
    evidence_json TEXT,
    failure_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (reserved_micro = reserved_grant_micro + reserved_purchased_micro),
    UNIQUE (user_id, run_id, call_index)
);
```

- [ ] **Step 3: Add the live upgrade test** immediately after `test_postgres_migration_preserves_legacy_stripe_ledger`.

```python
@pg_only
def test_postgres_migration_adds_attempt_index_before_dependent_index(
    pg_legacy_credits_url,
):
    with psycopg.connect(pg_legacy_credits_url) as conn:
        conn.execute(
            """
            INSERT INTO credit_llm_reservations (
                reservation_id, user_id, run_id, call_index,
                reserved_micro, reserved_grant_micro, reserved_purchased_micro,
                operation_key, request_digest, created_at, updated_at
            ) VALUES (
                'legacy-reservation', 1, 'legacy-run', 0,
                100, 100, 0, 'legacy-operation', 'legacy-digest',
                '2026-08-01T00:00:00+00:00', '2026-08-01T00:00:00+00:00'
            )
            """
        )

    db_pool._reset_for_tests()
    pg_module.PostgresCreditsStore(pg_legacy_credits_url)

    with psycopg.connect(pg_legacy_credits_url, row_factory=dict_row) as conn:
        columns = {
            row["column_name"]
            for row in conn.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = 'credit_llm_reservations'
                """
            )
        }
        reservation = conn.execute(
            """
            SELECT attempt_index, provider_id
            FROM credit_llm_reservations
            WHERE reservation_id = 'legacy-reservation'
            """
        ).fetchone()
        index_names = {
            row["indexname"]
            for row in conn.execute(
                """
                SELECT indexname
                FROM pg_indexes
                WHERE schemaname = current_schema()
                  AND tablename = 'credit_llm_reservations'
                """
            )
        }

    assert {"attempt_index", "provider_id"} <= columns
    assert reservation == {"attempt_index": 0, "provider_id": None}
    assert "idx_credit_llm_reservations_run_status" in index_names
```

- [ ] **Step 4: Run the new tests before implementation**.

Run: `pytest -q dashboard/backend/tests/domain/credits/test_repository_postgres.py -k 'provider_attempt_index_is_created_after_its_column or migration_adds_attempt_index_before_dependent_index'`

Expected: the static test fails because the index is still in the base DDL; the live test may be skipped when `TEST_POSTGRES_URL` is unset.

### Task 2: Move the dependent index into the migration DDL

**Files:**
- Modify: `dashboard/backend/domain/credits/repository_postgres.py:312-316`
- Modify: `dashboard/backend/domain/credits/repository_postgres.py:357-371`

**Interfaces:**
- Consumes: the existing base and migration DDL constants.
- Produces: fresh databases and legacy databases with the same `idx_credit_llm_reservations_run_status` index after `attempt_index` is present.

- [ ] **Step 1: Remove the dependent index from `CREDITS_POSTGRES_DDL`** while leaving `idx_credit_llm_reservations_user_status` unchanged.

Delete only:

```sql
CREATE INDEX IF NOT EXISTS idx_credit_llm_reservations_run_status
ON credit_llm_reservations(run_id, status, call_index, attempt_index);
```

- [ ] **Step 2: Add the same idempotent index creation** directly after the migration adds and checks `attempt_index`.

```sql
CREATE INDEX IF NOT EXISTS idx_credit_llm_reservations_run_status
ON credit_llm_reservations(run_id, status, call_index, attempt_index);
```

- [ ] **Step 3: Run the focused migration tests**.

Run: `pytest -q dashboard/backend/tests/domain/credits/test_repository_postgres.py -k 'postgres_schema_tracks_provider_attempt_identity or provider_attempt_index_is_created_after_its_column or migration_adds_attempt_index_before_dependent_index'`

Expected: PASS, or only the live PostgreSQL test skipped when `TEST_POSTGRES_URL` is unavailable.

### Task 3: Verify fresh schema, legacy upgrade, and rollback behavior

**Files:**
- Verify: `dashboard/backend/domain/credits/repository_postgres.py`
- Verify: `dashboard/backend/tests/domain/credits/test_repository_postgres.py`

**Interfaces:**
- Consumes: the migrated DDL and the existing transaction/rollback tests.
- Produces: evidence that the hotfix does not alter unrelated Credits schema behavior.

- [ ] **Step 1: Run all PostgreSQL Credits repository tests**.

Run: `pytest -q dashboard/backend/tests/domain/credits/test_repository_postgres.py`

Expected: PASS; live tests are skipped only if `TEST_POSTGRES_URL` is not set.

- [ ] **Step 2: Run the backend Credits and execution regression suites**.

Run: `pytest -q dashboard/backend/tests/domain/credits dashboard/backend/tests/infrastructure/llm/test_platform_credits_env_fallback.py`

Expected: PASS.

- [ ] **Step 3: Run static verification**.

Run: `python -m py_compile dashboard/backend/domain/credits/repository_postgres.py dashboard/backend/tests/domain/credits/test_repository_postgres.py && git diff --check`

Expected: no output and exit code `0`.

- [ ] **Step 4: Inspect the staged file list**.

Run: `git status --short && git diff --cached --name-only`

Expected: only the migration source, its PostgreSQL regression test, and the committed design/plan docs are staged; no database, secret, `.superpowers/`, or `work/` path is present.

### Task 4: Commit, push, and open the hotfix PR

**Files:**
- Commit: the files changed in Tasks 1-3.

**Interfaces:**
- Produces: a GitHub PR targeting `main` with the migration-order correction and regression coverage.

- [ ] **Step 1: Commit the implementation**.

```bash
git add dashboard/backend/domain/credits/repository_postgres.py dashboard/backend/tests/domain/credits/test_repository_postgres.py docs/superpowers/specs/2026-09-01-postgres-attempt-index-migration-order-design.md docs/superpowers/plans/2026-09-01-postgres-attempt-index-migration-order.md
git commit -m "fix(credits): order postgres attempt index migration"
```

- [ ] **Step 2: Push the branch**.

Run: `git push -u origin fix/postgres-attempt-index-migration-order`

- [ ] **Step 3: Open the pull request**.

```bash
gh pr create --base main --head fix/postgres-attempt-index-migration-order \
  --title "fix(credits): order postgres attempt index migration" \
  --body "Fix legacy PostgreSQL startup by creating attempt_index before idx_credit_llm_reservations_run_status. Adds a static DDL-order regression and a legacy-schema upgrade test. Focused Credits tests and py_compile pass; live PostgreSQL tests are reported according to TEST_POSTGRES_URL availability."
```

- [ ] **Step 4: Report Render deployment requirements**.

After merge, manually deploy the service because `autoDeploy` is disabled. Verify the new deployment is `live`, startup logs contain no `UndefinedColumn` for `attempt_index`, and only then run a single platform-model smoke test.
