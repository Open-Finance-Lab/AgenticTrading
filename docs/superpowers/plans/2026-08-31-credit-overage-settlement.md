# Credit Overage Settlement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Charge model-cost reservation overruns from a user's unreserved Credits before restricting the account.

**Architecture:** Keep settlement atomic inside each Credits store transaction. The reservation's held Grant/Purchased split is consumed first; supplementary available Grant/Purchased balances are computed from the same transaction snapshot and consumed only for the excess. SQLite and PostgreSQL retain parity, while the execution service continues to expose sanitized billing results.

**Tech Stack:** Python 3.13, SQLite, PostgreSQL/psycopg, Pydantic, pytest.

**Spec:** `docs/superpowers/specs/2026-08-31-credit-overage-settlement-design.md`

## Global Constraints

- Never spend Credits reserved by another open LLM reservation.
- Preserve idempotent settlement replay behavior and existing Grant-first accounting.
- Restrict an account only when an actual unpaid remainder remains.
- Do not expose provider response data or credentials in errors.
- Keep SQLite/PostgreSQL behavior and response fields equivalent.

### Task 1: Update Reservation Schema and Migrations

**Files:**
- Modify: `dashboard/backend/domain/credits/repository.py:116-145, 460-515`
- Modify: `dashboard/backend/domain/credits/repository_postgres.py:250-330`
- Test: `dashboard/backend/tests/domain/credits/test_repository.py`
- Test: `dashboard/backend/tests/domain/credits/test_repository_postgres.py`

**Interfaces:**
- Produces reservation tables that allow `settled_micro` to include supplementary debits.
- Preserves all existing columns, foreign keys, indexes, and historical rows.

- [x] **Step 1: Write failing migration tests** for a legacy SQLite reservation table and a PostgreSQL migration SQL assertion; assert the obsolete `settled_micro <= reserved_micro` constraint is absent and existing rows survive.
- [x] **Step 2: Run the focused tests** with `pytest -q dashboard/backend/tests/domain/credits/test_repository.py dashboard/backend/tests/domain/credits/test_repository_postgres.py`; confirm the new migration assertions fail before implementation.
- [x] **Step 3: Implement the schema migration**: use a SQLite table rebuild when the legacy reservation SQL contains the upper-bound check, and add PostgreSQL `DROP CONSTRAINT IF EXISTS` plus a non-negative replacement constraint.
- [x] **Step 4: Re-run the focused migration tests** and confirm they pass.
- [x] **Step 5: Commit** the migration changes in the final PR commit.

### Task 2: Implement Atomic Supplementary Settlement

**Files:**
- Modify: `dashboard/backend/domain/credits/repository.py:1100-1220`
- Modify: `dashboard/backend/domain/credits/repository_postgres.py:865-925`
- Test: `dashboard/backend/tests/domain/credits/test_repository.py`
- Test: `dashboard/backend/tests/domain/credits/test_repository_postgres.py`

**Interfaces:**
- `CreditsStore.settle_llm_credits(...)` and `PostgresCreditsStore.settle_llm_credits(...)` continue returning reservation dictionaries with `settled_micro`, `actual_micro`, `outstanding_micro`, and per-bucket ledger IDs.

- [x] **Step 1: Add failing SQLite tests** for fully covered and partially covered overruns, asserting supplementary Grant-first debits, active versus restricted account status, and remaining outstanding amount.
- [x] **Step 2: Add matching PostgreSQL contract tests** using the existing `@pg_only` fixture.
- [x] **Step 3: Implement settlement allocation**: calculate `excess_micro`, read unreserved Grant/Purchased projection inside the transaction, insert uniquely keyed supplementary usage entries, set `settled_micro` to reservation debit plus supplementary debit, and restrict only for a positive remainder.
- [x] **Step 4: Preserve replay behavior** by returning the settled row without creating duplicate usage entries when the same evidence is submitted again.
- [x] **Step 5: Run SQLite and PostgreSQL-focused tests** and confirm all pass (PostgreSQL may skip locally when unavailable).
- [x] **Step 6: Commit** the settlement changes in the final PR commit.

### Task 3: Wire Execution Billing Evidence and Regression Coverage

**Files:**
- Modify: `dashboard/backend/infrastructure/llm/execution/service.py:270-305`
- Test: `dashboard/backend/tests/infrastructure/llm/test_platform_credits_env_fallback.py`
- Test: `dashboard/backend/tests/infrastructure/llm/test_execution_client.py`

**Interfaces:**
- `LLMExecutionService._execute_platform` passes the actual provider cost to settlement and reports the store's full debit and outstanding values without changing safe error categories.

- [x] **Step 1: Add a failing execution-service regression** where actual provider cost exceeds the reservation but available Credits cover the difference; assert successful result and zero outstanding.
- [x] **Step 2: Update evidence mapping** so `debited_credits_micro` reflects the full amount charged after supplementary settlement while `outstanding_credits_micro` reflects only unpaid remainder.
- [x] **Step 3: Run the execution-focused tests** and confirm they pass.
- [x] **Step 4: Commit** the execution changes in the final PR commit.

### Task 4: Full Verification and Pull Request

**Files:**
- Verify all files changed in Tasks 1-3.

- [x] **Step 1: Run targeted regression suite**: `pytest -q dashboard/backend/tests/domain/credits dashboard/backend/tests/infrastructure/llm/test_platform_credits_env_fallback.py dashboard/backend/tests/infrastructure/llm/test_execution_client.py`.
- [x] **Step 2: Run static checks**: `python -m py_compile` on changed Python files, `git diff --check`, and repository JavaScript checks if frontend files are touched.
- [x] **Step 3: Inspect staged names and diff** for database files, secrets, `.superpowers/`, and `work/` before committing.
- [x] **Step 4: Push branch** with `git push -u origin fix/credit-overage-settlement`.
- [x] **Step 5: Open PR** targeting `main` with title `fix(credits): charge available balance for reservation overage`, documenting test results and any local PostgreSQL skips.
