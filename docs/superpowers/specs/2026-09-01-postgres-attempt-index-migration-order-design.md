# PostgreSQL Attempt Index Migration Order Hotfix Design

## Goal

Allow an existing PostgreSQL Credits database created before provider-attempt tracking to start on the current application version without manual production SQL or data loss.

## Current Failure

`PostgresCreditsStore._init_schema()` executes `CREDITS_POSTGRES_DDL` before `CREDITS_POSTGRES_GRANT_MIGRATION_DDL`. The base DDL contains `idx_credit_llm_reservations_run_status`, which references `attempt_index`. On a legacy `credit_llm_reservations` table, that column does not exist yet, so PostgreSQL raises `UndefinedColumn` before the migration can add it.

Fresh CI schemas do not expose the defect because the table is created with `attempt_index` already present. Production exposes it because `CREATE TABLE IF NOT EXISTS` preserves the older table shape.

## Chosen Design

Move creation of `idx_credit_llm_reservations_run_status` out of the base DDL and into the idempotent migration DDL immediately after the migration adds `provider_id` and `attempt_index`.

Initialization remains one transaction and keeps the existing order:

1. Run base DDL to create any missing tables and indexes that only reference baseline columns.
2. Run migration DDL to add missing reservation columns.
3. Create the provider-attempt index only after `attempt_index` exists.
4. Continue the existing constraint and Grant-ledger migrations.

Fresh databases still receive the same index because every `PostgresCreditsStore` initialization runs both DDL blocks. Existing databases upgrade without requiring an operator to alter the production database manually.

## Alternatives Considered

### Pre-DDL Compatibility Migration

Add a third DDL block that runs before the base schema and adds `attempt_index` to legacy tables. This fixes the immediate failure but introduces another migration phase and duplicates column ownership between two migration blocks.

### Direct Production SQL

Add the column manually in Render PostgreSQL before redeploying. This is faster for one environment but leaves the repository defect intact, does not protect other deployments, and creates configuration drift.

### Recommended Choice

Relocate the dependent index into the existing migration DDL. It is the smallest repository-level correction and directly enforces the dependency: the column is added before the index is created.

## Data and Compatibility

- Do not rebuild or replace `credit_llm_reservations`.
- Preserve all reservation and usage rows.
- Add `attempt_index` with `NOT NULL DEFAULT 0`, preserving the existing migration contract.
- Keep `provider_id` nullable for historical attempts.
- Keep the logical-attempt uniqueness constraint and all index names unchanged.
- Keep initialization idempotent for both fresh and already-upgraded databases.

## Error Handling

The migration continues to run in the existing PostgreSQL transaction. Any later migration failure rolls back the column, index, and constraint changes together. No fallback to SQLite and no silent error suppression are introduced.

## Verification

1. Add a static ordering regression that asserts the dependent index is absent from base DDL and appears after `ADD COLUMN IF NOT EXISTS attempt_index` in migration DDL.
2. Extend the live PostgreSQL legacy fixture with a pre-provider-attempt `credit_llm_reservations` table that omits `provider_id` and `attempt_index`.
3. Initialize `PostgresCreditsStore` against that legacy schema and assert both columns, the logical-attempt constraint, and `idx_credit_llm_reservations_run_status` exist afterward.
4. Reopen the upgraded store to prove the migration is idempotent.
5. Run focused Credits PostgreSQL tests, the backend test suite used by CI, and `git diff --check` before opening the PR.

## Deployment

After the hotfix PR merges, manually trigger a Render deployment because the service has `autoDeploy` disabled. Verify the deployed commit reaches `live`, confirm the startup migration no longer raises `UndefinedColumn`, then run one platform-model smoke test that can exercise OpenRouter-to-CommonStack failover.

## Amendment (PR #433, 2026-09-02): the index may already exist with the wrong columns

The design above assumes a legacy table has *no* `idx_credit_llm_reservations_run_status`. Production had one: commit `c0bcd863` (2026-08-24) created it over `(run_id, status, call_index)`, and `CREATE INDEX IF NOT EXISTS` matches by name alone, so the bare statement in the migration DDL no-ops against that table and the four-column definition never lands. The migration now drops the index only when `pg_get_indexdef` reports a column list other than `(run_id, status, call_index, attempt_index)`, then recreates it. The drop is conditional on purpose: this DDL runs at import on every deploy, and an unconditional DROP+CREATE would rebuild the index under ACCESS EXCLUSIVE each time instead of converging.

Pinned by `test_postgres_boot_migrates_pre_failover_reservation_table` (starts from the exact pre-#432 table, stale index included) and `test_postgres_boot_leaves_a_repaired_run_status_index_alone` (a second boot keeps the same index object). `test_store_twin_parity.py` now also checks that every Postgres twin creates an index only below the `ADD COLUMN` of any column it names, which is the general form of this defect.

## Amendment (PR #433 review follow-up, 2026-09-04): what converges, and what does not

Two corrections to the amendment above.

**The conditional drop needs `IF EXISTS` anyway.** The `IF EXISTS (...)` predicate that
decides whether to drop is evaluated before the `DROP INDEX` takes its lock, so two
processes booting against the same database can both pass it. The loser then raises
`index "..." does not exist`, which aborts `_init_schema`; because `credits_store` is
built at import, that takes down the whole app rather than one surface. Conditionality
comes from the drop's *position inside the guard*, not from the absence of `IF EXISTS`,
and the source guard no longer asserts otherwise.

**Converging was true of the index and false of everything beside it.** The migration
drops and re-adds thirteen constraints unconditionally on every boot. One of them —
`credit_llm_reservations_logical_attempt_key` — is a `UNIQUE`, i.e. exactly the full
index build under ACCESS EXCLUSIVE that the index repair was written to avoid, on the
same table. It is now guarded too: skipped when `pg_constraint.conkey` already names
`(user_id, run_id, call_index, attempt_index)`. Column identity is read from `conkey`
rather than matched against `pg_get_constraintdef` text, which is a rendering and drifts
between Postgres versions; a mismatch falls back to the previous drop+add.

The remaining twelve — the `CHECK`s on `credit_llm_reservations` and
`credit_ledger_entries`, and the `actor_user_id` foreign key — are **deliberately left
unconditional**, each costing a validating full-table scan per boot. Recognising an
existing `CHECK` has no `conkey` equivalent; it means comparing deparsed SQL text, and a
guard that silently stops matching converges to nothing while still looking correct.
Both tables sit behind a disabled billing flag today. Revisit if either grows — the cost
is real, it is just not yet worth buying with a fragile predicate.

`test_postgres_boot_leaves_a_repaired_run_status_index_alone` now pins the constraint's
`oid` and `conindid` across a second boot as well. `conindid` is the load-bearing half:
a drop and re-add keeps the constraint's *name*, so only the OID of the backing index
distinguishes a converged boot from a rebuilt one.
