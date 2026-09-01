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
