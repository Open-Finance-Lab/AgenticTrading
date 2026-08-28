# PR 411 Main Sync Design

## Goal

Make PR #411 mergeable against the latest `main` while preserving both the merged default-Credits behavior and PR #411's exact run-level Credit activity behavior.

## Context

PR #411 was created before PR #413 fixed the Marketplace provider-label test and before PR #410 added automatic welcome Credits and smaller checkout amounts. Re-running the old workflow reused the old pull-request merge snapshot, so it continued to report the already-fixed Marketplace failure. After PR #410 merged, GitHub also reported PR #411 as conflicting.

The textual conflicts are limited to:

- `dashboard/frontend/app.html`
- `dashboard/frontend/js/credits.js`
- `dashboard/backend/tests/test_frontend_fast_boot.py`

The local branch named `fix/backtest-credit-activity` contains unrelated, unpushed benchmark commits. The repair must therefore run from an isolated worktree based on `origin/fix/backtest-credit-activity` and push explicitly to the remote PR branch.

## Merge Strategy

Merge `origin/main` into an isolated repair branch. Use a merge commit instead of rebasing so the existing PR history is preserved and no force push is required.

Resolve the conflicts by composing the two behaviors rather than choosing one side wholesale:

- Keep `main`'s checkout packages (`$0.50`, `$1`, `$2`, `$5`), `$1` default selection, and `$0.50` through `$5.00` custom range.
- Render all package Credit quantities with the shared fixed-six-decimal formatter contract introduced by PR #411.
- Keep `main`'s `system_promotion_grant` Activity title, `Welcome Credits`.
- Keep PR #411's `backtest_usage` Activity title, run-level provider/model context, model-call count, and mixed-provider/model labels.
- Keep the shared `credit-format.js` helper before every consumer.
- Bump the combined `credits.js` asset version beyond both parents and retain the PR #411 asset versions for `admin-credits.js` and `admin-analytics.js`.
- Update the frontend fast-boot assertions to match the final asset graph.

## Data And Behavior Contracts

The merge must not change backend Credit arithmetic or introduce floating-point calculations. Public amounts remain display strings backed by integer micro-Credits.

Activity rendering must distinguish:

- `backtest_usage`: one aggregated row per backtest run, negative amount, safe provider/model/call-count/run context.
- `system_promotion_grant`: one positive `Welcome Credits` row.
- `refund`: negative `Refund` row.
- Other positive entries: `Credit purchase`.

No reservation IDs, call indexes, raw evidence JSON, secrets, or database artifacts may enter the frontend or commit.

## Verification

Run focused tests for the three resolved frontend files, exact Credit formatting, Credits API behavior, SQLite and PostgreSQL repository contracts, and the Marketplace regression fixed by PR #413. Then run the complete backend suite without a local PostgreSQL URL and let GitHub Actions execute the PostgreSQL tier with its service container.

The final PR must be conflict-free, contain no unrelated benchmark commits, and receive a fresh GitHub Actions run based on the updated head commit and current `main`.

## Out Of Scope

- Changing the Welcome Credit amount or promotion-ledger semantics from PR #410.
- Changing checkout package limits from PR #410.
- Changing PR #411's run-level aggregation or precision contract.
- Modifying benchmark comparison work that exists only on the separate local branch.
