# Credit Overage Settlement Design

## Goal

Prevent a Credits account from becoming restricted when a model call costs more than its reservation but the account still has enough unreserved Credits to pay the difference.

## Current Failure

`settle_llm_credits` debits at most `reserved_micro` and records `actual_micro - reserved_micro` as outstanding. It then restricts the account whenever that difference is positive. The code does not try the account's currently available Grant or Purchased balance, so a small estimator error can freeze an account with a large positive balance.

## Behavior

1. Settlement consumes the reservation's Grant allocation first, then its Purchased allocation, exactly as today.
2. When `actual_micro > reserved_micro`, settlement computes the user's unreserved balances in the same transaction. It debits the excess from available Grant Credits first and Purchased Credits second.
3. `settled_micro` records the full amount actually debited, including the excess. `outstanding_micro` records only the unpaid remainder.
4. The account is restricted with reason `llm_overage` only when the unreserved balance cannot cover the excess. A fully covered excess leaves the account active and reports zero outstanding.
5. The current reservation is excluded from the supplementary-balance calculation; other open reservations remain protected. Concurrent settlements serialize on the account/reservation transaction locks and cannot spend the same Credits twice.
6. Existing overage recovery remains unchanged and can settle historical debt through a later purchase or admin Grant.

## Storage Changes

- Replace the `settled_micro <= reserved_micro` reservation check with a non-negative settled amount check in fresh SQLite/PostgreSQL schemas.
- Add an idempotent migration for existing SQLite tables, preserving reservation and usage rows while removing the obsolete upper-bound check.
- Drop and recreate the equivalent PostgreSQL check constraint in the existing migration DDL.
- Keep `outstanding_recovered_micro` semantics unchanged: only rows with unpaid remainder participate in recovery.

## API and UI

No endpoint shape changes are required. Existing settlement and balance responses expose the corrected `settled_micro`, `outstanding_micro`, `account_status`, and `restriction_reason` values. User-facing restricted copy remains reserved for genuinely unpaid overage.

## Verification

- SQLite: a fully covered excess settles without restriction; a partially covered excess leaves only the remainder restricted; existing overage and recovery tests remain green.
- PostgreSQL contract tests cover the same two paths and verify the migration constraint update.
- Execution-service tests verify the billing evidence reports full debit and zero outstanding when supplementary balance covers the excess.
- Run targeted Credits/execution tests, syntax checks, and the repository's standard CI command before opening the PR.
