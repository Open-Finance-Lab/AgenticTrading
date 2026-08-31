# Credit Restriction Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make LLM-overage accounts recover automatically after new Credits arrive, keep refund-review accounts protected, and expose actionable restriction errors and admin controls.

**Architecture:** Extend the existing account and LLM reservation ledgers in both SQLite and PostgreSQL. Store a safe restriction reason and a per-reservation recovered-overage total; process recovery atomically inside purchase webhook and Grant transactions using idempotent usage entries. Add a dedicated execution error category and expose status/recovery data through the existing Credits and Admin Users routes and vanilla frontend modules.

**Tech Stack:** Python 3, FastAPI, Pydantic, SQLite, PostgreSQL/psycopg, vanilla JavaScript, pytest.

**Spec:** `docs/superpowers/specs/2026-08-31-credit-restriction-recovery-design.md`

## Global Constraints

- Keep SQLite and PostgreSQL behavior equivalent.
- Never return provider credentials, raw Stripe payloads, or stack traces.
- Preserve existing balance aliases, numeric ledger cursors, idempotency keys, and historical rows.
- Recovery must be atomic with the source purchase or Grant and safe under retries.
- Only `llm_overage` accounts may accept new Credits while restricted; `refund_reconciliation` accounts require explicit administrator review.
- Do not modify provider credentials, Render configuration, or unrelated backtest timeout behavior.

---

### Task 1: Add typed restriction and recovery fields

**Files:**
- Modify: `dashboard/backend/domain/credits/models.py`
- Modify: `dashboard/backend/domain/credits/repository_common.py`
- Test: `dashboard/backend/tests/domain/credits/test_grant_models.py`

**Interfaces:**
- Produces `RestrictionReason`, `BalanceResult.restriction_reason`, `BalanceResult.outstanding_credits_micro`, and `LLMSettlementResult.outstanding_recovered_micro`.

- [ ] **Step 1: Write failing model tests**

Add assertions that a balance exposes `restriction_reason`, non-negative `outstanding_credits_micro`, and that a settlement accepts `outstanding_recovered_micro` defaulting to zero while rejecting negative values.

- [ ] **Step 2: Implement the fields and safe reason constants**

Add a `Literal["llm_overage", "refund_reconciliation"]` alias, default legacy reasons to `None` on active accounts, and keep Pydantic `extra="forbid"` validation.

- [ ] **Step 3: Run the focused model tests**

Run `pytest dashboard/backend/tests/domain/credits/test_grant_models.py -q` and confirm the new and existing assertions pass.

- [ ] **Step 4: Commit**

Run `git add dashboard/backend/domain/credits/models.py dashboard/backend/domain/credits/repository_common.py dashboard/backend/tests/domain/credits/test_grant_models.py && git commit -m "feat(billing): model restriction recovery state"`.

### Task 2: Migrate stores and expose restriction metadata

**Files:**
- Modify: `dashboard/backend/domain/credits/repository.py`
- Modify: `dashboard/backend/domain/credits/repository_postgres.py`
- Modify: `dashboard/backend/domain/credits/service.py`
- Test: `dashboard/backend/tests/domain/credits/test_repository.py`
- Test: `dashboard/backend/tests/domain/credits/test_repository_postgres.py`

**Interfaces:**
- Produces `get_account_billing_state(user_id)`, account fields `restriction_reason`, and reservation field `outstanding_recovered_micro` for both stores.

- [ ] **Step 1: Add migration and metadata tests**

Create a restricted account in each store, assert the default reason is `refund_reconciliation`, create an overage reservation, and assert the result reports zero recovered amount and the aggregate outstanding amount.

- [ ] **Step 2: Add backward-compatible columns**

Add `credit_accounts.restriction_reason TEXT` and `credit_llm_reservations.outstanding_recovered_micro INTEGER/BIGINT NOT NULL DEFAULT 0`. Inspect existing SQLite columns before issuing `ALTER TABLE`; use `ADD COLUMN IF NOT EXISTS` in PostgreSQL migration SQL. Add constraints for the two reason values and non-negative recovery.

- [ ] **Step 3: Include metadata in balance projections**

Lock/read the account status and reason alongside the existing projection. Sum `max(outstanding_micro - outstanding_recovered_micro, 0)` over settled reservations for the user. Treat a restricted account with a null reason as `refund_reconciliation`.

- [ ] **Step 4: Run both store suites**

Run `pytest dashboard/backend/tests/domain/credits/test_repository.py dashboard/backend/tests/domain/credits/test_repository_postgres.py -k 'account or reservation or migration' -q`. PostgreSQL tests may skip only through the repository's existing environment convention.

- [ ] **Step 5: Commit**

Run `git add dashboard/backend/domain/credits/repository.py dashboard/backend/domain/credits/repository_postgres.py dashboard/backend/domain/credits/service.py dashboard/backend/tests/domain/credits/test_repository.py dashboard/backend/tests/domain/credits/test_repository_postgres.py && git commit -m "feat(billing): persist restriction reasons and recovery state"`.

### Task 3: Implement atomic overage recovery

**Files:**
- Modify: `dashboard/backend/domain/credits/repository.py`
- Modify: `dashboard/backend/domain/credits/repository_postgres.py`
- Modify: `dashboard/backend/domain/credits/service.py`
- Test: `dashboard/backend/tests/domain/credits/test_repository.py`
- Test: `dashboard/backend/tests/domain/credits/test_repository_postgres.py`

**Interfaces:**
- Produces `recover_llm_overage_in_transaction(user_id, source_operation_key)` returning recovered and remaining micro-Credits.

- [ ] **Step 1: Add failing partial/complete recovery tests**

Reserve 1.00 Credit, settle 1.25 Credits to create a 0.25 overage, add a 0.10 purchased or Grant entry, assert the account remains restricted with 0.15 outstanding, then add 0.15 and assert the account becomes active. Assert balance decreases by each recovery and identical source operation retries do not add another usage row.

- [ ] **Step 2: Implement the recovery transaction helper**

Lock the account, select oldest settled reservations with unrecovered overage, calculate available Grant-first and purchased balances, insert negative usage entries with a source-derived idempotency key, increment `outstanding_recovered_micro`, and clear `restriction_reason`/set `active` only when total unrecovered overage reaches zero. Return aggregate recovered and remaining amounts.

- [ ] **Step 3: Set overage reason during settlement**

When `outstanding_micro > 0`, update the account to `restricted` with `restriction_reason='llm_overage'`. Preserve refund-review restrictions and never overwrite them from model settlement.

- [ ] **Step 4: Verify both stores**

Run `pytest dashboard/backend/tests/domain/credits/test_repository.py dashboard/backend/tests/domain/credits/test_repository_postgres.py -k 'overage or recovery or restriction' -q`.

- [ ] **Step 5: Commit**

Run `git add dashboard/backend/domain/credits/repository.py dashboard/backend/domain/credits/repository_postgres.py dashboard/backend/domain/credits/service.py dashboard/backend/tests/domain/credits/test_repository.py dashboard/backend/tests/domain/credits/test_repository_postgres.py && git commit -m "feat(billing): recover model overage from added Credits"`.

### Task 4: Allow safe funding paths and recover atomically

**Files:**
- Modify: `dashboard/backend/domain/credits/repository.py`
- Modify: `dashboard/backend/domain/credits/repository_postgres.py`
- Modify: `dashboard/backend/domain/credits/service.py`
- Modify: `dashboard/backend/api/routers/credits.py`
- Modify: `dashboard/backend/api/routers/admin_credits.py`
- Test: `dashboard/backend/tests/domain/credits/test_service.py`
- Test: `dashboard/backend/tests/test_credits_api.py`

**Interfaces:**
- Checkout creation accepts restricted `llm_overage` accounts and rejects `refund_reconciliation` with a safe 403.
- Paid checkout settlement and Grant assignment call recovery using their stable operation IDs.

- [ ] **Step 1: Add failing service/API tests**

Assert checkout is allowed for an LLM-overage account, remains blocked for refund review, and that paid webhook settlement/Grant assignment automatically reduce outstanding overage and return the new account state.

- [ ] **Step 2: Implement reason-aware gates**

Change `create_checkout` and Grant assignment to inspect `restriction_reason`. Keep the existing refund-review refusal and map it to a message explaining administrator review.

- [ ] **Step 3: Wire recovery after successful funding**

In each store's paid checkout and Grant mutation transaction, call the recovery helper only after the positive ledger entry is written. Pass the order/event or Grant operation key so the recovery writes are replay-safe.

- [ ] **Step 4: Return recovery metadata**

Include `account_status`, `restriction_reason`, `outstanding_credits_micro`, and recovered amount in the relevant service result/API payload without changing existing fields.

- [ ] **Step 5: Run focused API/service tests and commit**

Run `pytest dashboard/backend/tests/domain/credits/test_service.py dashboard/backend/tests/test_credits_api.py -k 'checkout or grant or restricted or recovery' -q`, then commit with `feat(billing): unblock and recover overage funding`.

### Task 5: Add dedicated restricted execution errors

**Files:**
- Modify: `dashboard/backend/infrastructure/llm/execution/errors.py`
- Modify: `dashboard/backend/infrastructure/llm/execution/service.py`
- Modify: `dashboard/backend/api/routers/backtests.py`
- Test: `dashboard/backend/tests/infrastructure/llm/test_execution_client.py`
- Test: `dashboard/backend/tests/test_backtests_router.py`

**Interfaces:**
- Produces `ExecutionErrorCategory.ACCOUNT_RESTRICTED` and a safe message that directs the user to add Credits or contact an administrator.

- [ ] **Step 1: Add failing restricted-error tests**

Make a reservation mock raise `CreditAccountRestrictedStoreError` and assert the execution layer returns `account_restricted`, while provider/credential details remain absent from the public message.

- [ ] **Step 2: Implement category mapping and message**

Catch the restriction store error before the generic billing arm, map it to `ACCOUNT_RESTRICTED`, and use a fixed message distinguishing “add Credits to settle model usage” from “administrator review” based on the balance state available to the service.

- [ ] **Step 3: Verify router serialization**

Assert failed backtest responses contain the safe human-readable message, never Python class names, SQL text, or stack traces.

- [ ] **Step 4: Run focused execution/router tests and commit**

Run `pytest dashboard/backend/tests/infrastructure/llm/test_execution_client.py dashboard/backend/tests/test_backtests_router.py -k 'billing or restricted or error' -q`, then commit with `fix(billing): explain restricted Credit execution failures`.

### Task 6: Update Credits and Admin Users UI

**Files:**
- Modify: `dashboard/frontend/js/credits.js`
- Modify: `dashboard/frontend/js/admin-credits.js`
- Modify: `dashboard/frontend/app.html`
- Modify: `dashboard/frontend/styles.css`
- Test: `dashboard/backend/tests/test_credits_frontend.py`
- Test: `dashboard/backend/tests/test_admin_credits_frontend.py`

**Interfaces:**
- Credits UI shows reason/outstanding amount and enables purchase controls only for recoverable LLM overage.
- Admin Users displays status/reason/outstanding amount and exposes Reinstate only for refund-review accounts.

- [ ] **Step 1: Add failing frontend contract assertions**

Assert the scripts contain reason-aware copy, render the outstanding amount, call the existing Reinstate endpoint, and do not render a Reinstate action for `llm_overage` accounts.

- [ ] **Step 2: Extend admin user payload and table**

Return each user's account metadata from `/api/admin/credits/users`, add status/reason/outstanding columns, and implement an accessible button that posts to `/api/credits/admin/credits/accounts/{user_id}/reinstate` with confirmation and refresh.

- [ ] **Step 3: Update Credits state rendering**

Replace the generic “under review” message with reason-specific copy. Keep checkout enabled for `llm_overage` and disabled for refund review or unavailable billing. Display the exact remaining amount using the existing six-decimal formatter.

- [ ] **Step 4: Run frontend contract tests and commit**

Run `pytest dashboard/backend/tests/test_credits_frontend.py dashboard/backend/tests/test_admin_credits_frontend.py -q`, then commit with `feat(ui): surface Credit restriction recovery state`.

### Task 7: Full verification and PR

**Files:**
- No additional source files; review all task diffs and generated metadata.

- [ ] **Step 1: Run focused regression suite**

Run `pytest dashboard/backend/tests/domain/credits dashboard/backend/tests/test_credits_api.py dashboard/backend/tests/test_backtests_router.py dashboard/backend/tests/infrastructure/llm/test_execution_client.py -q`.

- [ ] **Step 2: Run complete backend tests**

Run `pytest -q`; record the pass/skip counts and investigate any regression before publishing.

- [ ] **Step 3: Audit the diff**

Run `git diff origin/main...HEAD --check`, inspect `git diff origin/main...HEAD`, and confirm no API keys, database URLs, `.superpowers/`, or `work/` files are staged.

- [ ] **Step 4: Push and open the PR**

Push the current branch with `git push -u origin fix/backtest-timeout-consistency` and create a PR targeting `main` titled `fix(billing): recover restricted Credit accounts safely`, including the test commands and the distinction between automatic overage recovery and manual refund review.
