# PR #289 Latest-Main Synchronization Design

## 1. Background

PR #289 (`feat(backtest): enforce A-share lot execution and order outcomes`)
implements a uniform 100-share lot rule for iFinD A-share backtests, bounded
order-outcome auditing, and Trading Log presentation. Its original CI passed,
but `main` continued to change during development and GitHub now marks the PR
as `CONFLICTING`.

As of 2026-08-05:

- iFinD historical A-share backtesting and A-share LLM backtesting are in `main`;
- A-share T+1 enforcement is in `main` through PR #272 and PR #288;
- PR #289 has not been merged;
- the latest `main` commit is `8317f42`;
- conflicts are limited to four files: one backtest API route and three
  Backtest-page assets; and
- the product cap for one backtest has changed from `$10,000` to `$3,000`.

This iteration will merge the latest `main` into the existing PR branch,
resolve conflicts without expanding the feature, perform a real iFinD browser
regression, and continue using PR #289.

## 2. Goals

1. Merge the latest `origin/main` into `feat/ifind-ashare-lot-size`.
2. Preserve PR #289's uniform 100-share lot rule and order auditing.
3. Preserve the latest API, security, authentication, Agent-page, and Backtest
   changes from `main`.
4. Replace stale `$10,000` capital documentation and assertions with the latest
   `$3,000` cap.
5. Complete automated, real-iFinD browser, and GitHub CI verification.
6. Restore PR #289 to a mergeable state without exposing credentials or
   committing local database changes.

## 3. Non-Goals

This iteration will not implement:

- the STAR Market 200-share minimum and incremental quantity rules;
- odd-lot sell handling or other board- or broker-specific exceptions;
- commission, stamp duty, transfer fees, slippage, spread, or volume limits;
- price limits, suspensions, or circuit-breaker simulation;
- a fix for same-bar close observation and same-bar close execution;
- live A-share paper trading, broker connectivity, or real orders;
- changes to Alpaca or vn.py order-quantity behavior; or
- merging the PR on behalf of the repository maintainers.

Both registered iFinD universes will continue to use `lot_size=100`. STAR
Market differences will be handled in a separate iteration so conflict
resolution does not expand the implementation and review surface.

## 4. Branch and Synchronization Strategy

Check out `feat/ifind-ashare-lot-size` in a separate clean worktree, then create
a normal merge from the latest `origin/main`:

```text
origin/main
    \
     +-- merge commit --> feat/ifind-ashare-lot-size --> PR #289
    /
existing PR #289 commits
```

A merge is preferred over a rebase or reconstructed cherry-pick series because
the PR is already public. A normal merge preserves the existing eight commits
and does not require a force push. The synchronization must not use an older
development worktree that contains a modified database.

## 5. Conflict Resolution

### 5.1 `dashboard/backend/api/routers/backtests.py`

- Preserve the latest synchronous `get_run_trades` route, session ownership
  check, and security boundary.
- Add PR #289's bounded `order_events` sample and count fields to the current
  response.
- Return an empty order-event sample for historical runs without affecting
  existing `trades` data.
- Do not turn order events into an unbounded list endpoint.

### 5.2 `dashboard/frontend/app.html`

- Preserve the latest page structure, Agent sections, capital limits, and asset
  version changes.
- Add Quantity, Status, and Reason presentation to the latest Trading Log.
- Keep all user-visible strings in English.
- Advance cache-buster versions when static assets change; never restore an
  older version.

### 5.3 `dashboard/frontend/app.js`

- Preserve the latest authentication, CSRF, Agent shelves, backtest creation,
  and capital-limit behavior.
- Port PR #289's order-event parsing, legacy-run fallback, status/reason mapping,
  and filtering logic.
- Prefer `order_events` for iFinD runs and fall back to `trades` for historical
  runs that do not contain the new field.
- Do not restore old functions or page state that the latest `main` replaced.

### 5.4 `dashboard/frontend/styles.css`

- Preserve all latest global and page styles.
- Add only the Trading Log status, quantity, and responsive styles required by
  PR #289.
- Scope new styles to the existing Trading Log selector hierarchy so they do
  not affect other tables or pages.

## 6. Runtime Semantics

```text
Agent decision
  -> MarketProfile (iFinD: lot_size=100, T+1=true)
  -> shared executor
  -> quantity validation
  -> BUY cash validation or SELL position/T+1 validation
  -> filled / partial / rejected
  -> trades + rejected_orders + order_events
  -> API
  -> Trading Log
```

- A rule-based A-share buy signal requests 100 shares.
- An LLM may request one or more full lots; 50, 150, and fractional quantities
  are rejected in full.
- Invalid quantities are not rounded, and insufficient-cash orders are not
  automatically reduced.
- A fully T+1-frozen sell is `rejected`; a partially sellable order is `partial`.
- `trades` records actual fills only, while `order_events` records exactly one
  outcome per attempted order.
- Rejections do not change cash, positions, trade count, return, or equity.
- A successful backtest may legitimately contain zero fills.

## 7. Capital and Currency Semantics

The synchronized branch will follow the latest product limits:

- minimum backtest capital: `$1`;
- default backtest capital: `$1,000`; and
- maximum backtest capital: `$3,000`.

iFinD backtests continue to convert the selected USD reporting capital into a
native CNY ledger using the applicable historical USD/CNY rate. Lot
affordability is evaluated in the CNY ledger. The API and frontend retain USD
reporting values alongside native-CNY and FX audit fields. Stale `$10,000`
limits in PR documentation or tests must be updated to `$3,000`; conflict
resolution must not restore the old product rule.

## 8. Error Handling, Compatibility, and Security

1. Invalid lots, insufficient cash, and T+1 freezes are order outcomes, not
   backtest failures or HTTP 500 errors.
2. A fully rejected order displays `--` for executed value rather than its
   requested value.
3. Historical runs without `order_events` continue to display their original
   trade records.
4. Alpaca, vn.py, and other `lot_size=1` paths retain their existing behavior.
5. Current login, session, CSRF, and asset-loading behavior must not regress.
6. iFinD tokens, LLM keys, Authorization headers, and other credentials must
   not appear in code, commits, logs, pages, test output, or the PR body.
7. `dashboard/storage/data/backtest.db` and real-backtest artifacts must not be
   committed.

## 9. Verification Plan

### 9.1 Merge Integrity

- Confirm that all conflict markers are removed.
- Run `git diff --check`.
- Review all four conflicted files to verify that current `main` behavior is
  preserved.
- Confirm that the commit diff contains no databases, credentials, or unrelated
  generated artifacts.

### 9.2 Focused Automated Coverage

Cover:

- the uniform 100-share rule for both iFinD profiles;
- accepted 100- and 200-share orders and rejected 50-, 150-, and fractional
  orders;
- full rejection for insufficient cash with no ledger mutation;
- full T+1 rejection, partial fills, and next-trading-day release;
- the shared rule for rule-based and LLM decisions;
- consistent `order_events` in final results, API responses, live progress, and
  the Trading Log;
- legacy-run fallback;
- the `$1,000` default and `$3,000` maximum; and
- unchanged Alpaca and vn.py behavior.

### 9.3 Full Regression

- Run the complete `dashboard/backend/tests` suite.
- Run the complete `packaging/agentictrading/tests` suite.
- Run the relevant frontend Node/static-contract tests.
- Confirm that no new failures are introduced.

### 9.4 Real iFinD Browser Acceptance

Start the local service on an unused port with a temporary database and the
existing local environment configuration. Do not print credentials. Run one
backtest through the browser with:

```text
Data source: iFinD China A-Shares (60 min)
Universe: A-Share Demo 6
Decision source: Rule-based
Initial capital: $1,000
```

Verify the data-source badge, initial and final capital, a nonblank equity
curve, either actual trades or a valid zero-fill result, order quantities,
statuses, and reasons. Check desktop and narrow viewports, browser-console
errors, and network failures. If the real data does not trigger a particular
rejection state, rely on deterministic automated coverage rather than fabricating
a real-backtest result.

### 9.5 GitHub Acceptance

- Push the normal merge commit and required compatibility fixes to the existing
  branch.
- Preserve the existing PR #289 URL and history.
- Require GitHub Backend tests, Packaging tests, CodeQL, and other relevant
  checks to pass.
- Confirm that GitHub changes PR #289 from `CONFLICTING` to mergeable.

## 10. Definition of Done

This iteration is complete only when all of the following are true:

1. PR #289 contains the latest `main`.
2. The four conflicts are resolved without regressing current `main` behavior.
3. Both iFinD universes retain the uniform 100-share lot rule.
4. T+1, insufficient-cash handling, order auditing, and historical-run
   compatibility pass verification.
5. The capital limit follows the latest `$3,000` product rule.
6. Automated and real-iFinD browser acceptance pass.
7. PR #289 has green GitHub CI and is mergeable.
8. No credentials, databases, or unrelated files are committed.
