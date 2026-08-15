# A-Share Daily Market Rules Implementation Plan

**Date:** 2026-08-12  
**Branch:** `feat/ashare-market-rules`  
**Base:** latest `origin/main` at implementation start  
**Design:** `docs/superpowers/specs/2026-08-12-ashare-market-rules-design.md`

## Objective

Enforce official iFinD full-day suspension and closing price-limit observations in ATL's
A-share historical backtests. Apply the same rule contract to Agent orders and the
buy-and-hold baseline without changing Alpaca or vn.py behavior. Keep the platform
simulation-only and keep every credential and unsanitized production response out of
Git.

## Guardrails

- Do not infer suspension or theoretical limit prices from prior close, board rules, or
  OHLCV.
- Do not apply an official end-of-day limit state to an earlier intraday bar.
- Do not hard-code an unverified iFinD indicator or response field.
- Fail an A-share run before execution when rule data is unavailable or incomplete.
- Keep all repository and GitHub text in English.
- Do not log request headers, tokens, account details, or raw production payloads.
- Do not commit `.env`, database, response capture, or temporary fixture files.
- Preserve current A-share lot-size, T+1, FX, and transaction-cost behavior.
- Preserve all Alpaca and vn.py execution behavior and response shapes.

## Task 1: Verify the Official iFinD Rule Command

**Read/inspect:**

- iFinD SuperCommand indicator generator or an authorized generated HTTP command
- `dashboard/backend/infrastructure/market_data/ifind_client.py`
- `dashboard/backend/tests/infrastructure/market_data/test_ifind_client.py`

**Steps:**

1. Generate or inspect the official commands for daily trading status, daily closing
   limit status, official daily close, and real-time upper/lower limit prices.
2. Run the narrowest authorized request for one registered A-share and a short historical
   date window.
3. Record only a sanitized structural summary: endpoint category, request field names,
   top-level response keys, table keys, indicator keys, value types, and row counts.
4. Confirm that historical quotation exposes daily trading/limit status and close, while
   real-time quotation alone exposes `upperLimit` and `downLimit`.
5. Confirm that requesting daily status fields from high-frequency history returns no
   point-in-time values, so closing state must never be backfilled onto earlier bars.
6. Confirm that blank history rows can be resolved only through the authorized same-date
   basic-data trading and limit-status indicators.
7. Keep the raw response outside the repository and delete temporary captures after the
   fixture shape is encoded.

**Stop condition:** No implementation proceeds until all vendor-specific names and value
encodings are verified. If the account lacks a required field permission, report the
permission failure and do not substitute inferred rules. Do not claim historical
intraday price-limit enforcement from daily closing data.

## Task 2: Add the Daily Rule Domain Contract and Adapter

**Create:**

- `dashboard/backend/domain/backtesting/market_rules.py`
- `dashboard/backend/infrastructure/market_data/ifind_market_rules.py`
- `dashboard/backend/tests/infrastructure/market_data/test_ifind_market_rules.py`
- `dashboard/backend/tests/domain/backtesting/test_market_rules.py`

**Modify:**

- `dashboard/backend/infrastructure/market_data/ifind_client.py`
- `dashboard/backend/infrastructure/market_data/ifind_ashare.py`
- corresponding client/provider tests

**Steps:**

1. Write failing tests for immutable `DailyMarketRule` and `MarketRuleCalendar` lookup.
2. Write sanitized fixture tests for normal, suspended, malformed, duplicate,
   missing-symbol, and missing-date responses.
3. Implement the verified iFinD client request without exposing vendor fields outside the
   adapter.
4. Normalize official values into an explicit suspension boolean, closing limit enum,
   and native-CNY official close.
5. Use iFinD's documented unadjusted settings for both high-frequency bars and daily
   closes so the two price paths share one executable-price basis.
6. Batch blank history rows by date and fetch only their official basic-data status
   supplement; never infer a blank row.
7. Validate coverage against every registered symbol and every combined-clock market
   date.
8. Derive the unique final hourly timestamp per active symbol-date and validate its close
   against the official daily close to the profile price tick.
9. Raise a dedicated sanitized `MarketRuleDataError` for transport, permission, schema,
   coverage, or close-alignment failures.

**Verification:** Run the new client, adapter, provider, and domain test modules plus
`git diff --check`.

**Commit:** `feat(market-data): load official A-share daily rules`

## Task 3: Enforce Rules in Shared Order Execution

**Modify:**

- `dashboard/backend/domain/trading/execution.py`
- `dashboard/backend/domain/backtesting/portfolio_manager.py`
- `dashboard/backend/domain/backtesting/engine.py`
- `dashboard/backend/tests/domain/trading/test_execution.py`
- `dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py`

**Steps:**

1. Add failing tests for suspended BUY/SELL, closing upper-limit BUY, closing lower-limit
   SELL, permitted opposite directions, and earlier-bar non-blocking.
2. Pass the daily rule from engine to portfolio manager and shared executor only for the
   iFinD A-share profile.
3. Run market-rule gates before lot-size, T+1, cash, and cost checks.
4. Use tick-safe native-CNY decimal comparison against the official daily close and
   require the validated final hourly timestamp for closing gates.
5. Emit `suspended`, `limit_up_buy_blocked`, or `limit_down_sell_blocked` with zero
   execution, zero fees, and complete rule audit fields.
6. Aggregate the three Agent rejection counts in run metadata.
7. Add Alpaca/vn.py characterization tests proving no market-rule gate leaks into their
   paths.

**Verification:** Run execution, portfolio, engine, and A-share integration tests.

**Commit:** `feat(backtest): enforce A-share suspension and price limits`

## Task 4: Apply the Same Rules to Buy-and-Hold

**Modify:**

- `dashboard/backend/baseline_generator.py`
- `dashboard/backend/domain/backtesting/engine.py`
- `dashboard/backend/tests/test_baseline_generator_offline.py`
- `dashboard/backend/tests/integration/test_ifind_ashare_backtest.py`

**Steps:**

1. Add failing tests where a baseline symbol is suspended or closing-upper-limit blocked
   at its first eligible bar and becomes eligible later.
2. Track pending initial allocations per symbol.
3. Retry a pending allocation only on a later symbol bar with a valid rule.
4. Recalculate price, affordable lot quantity, fees, and available cash at retry time.
5. Keep baseline rejection audit separate from Agent events.
6. Record delayed and end-of-run unfilled allocation counts in baseline metadata.
7. Confirm ordinary US and unrestricted baselines retain byte-for-byte-equivalent
   behavior where practical.

**Verification:** Run baseline and A-share integration tests.

**Commit:** `feat(backtest): delay blocked A-share baseline buys`

## Task 5: Persist and Expose Rule Audits

**Modify:**

- `dashboard/backend/database.py`
- `dashboard/backend/database_postgres.py`
- `dashboard/backend/api/routers/backtests.py`
- database, migration, router, and metadata tests

**Steps:**

1. Add nullable order-event/trade audit columns for rule date, suspended state, closing
   limit state, official native-CNY close, and closing-gate effective state using
   existing lazy-migration patterns.
2. Keep SQLite and PostgreSQL schemas and queries in parity.
3. Serialize the market-rule profile/version, enabled state, rejection totals, and
   baseline delay summary through the run APIs.
4. Map `MarketRuleDataError` to a sanitized English iFinD failure containing
   `Market rule data unavailable`.
5. Ensure old rows and non-A-share runs omit optional fields rather than fabricating
   values.

**Verification:** Run database parity, lazy migration, router, and metadata tests.

**Commit:** `feat(backtest): persist A-share market-rule audits`

## Task 6: Display English Rule Outcomes

**Modify:**

- `dashboard/frontend/app.html`
- `dashboard/frontend/app.js`
- `dashboard/frontend/styles.css` only if existing styles cannot support the audit line
- frontend contract tests, especially Trading Log and iFinD A-share tests

**Steps:**

1. Display `A-share market rules: Enabled` in Run config only when the run carries the
   rule profile.
2. Map stable rejection codes to `Suspended`, `Buy blocked at upper limit`, and
   `Sell blocked at lower limit`.
3. Display the official native-CNY close and closing limit state on rejected rows without
   showing fake fees.
4. Display non-zero rule rejection totals and baseline delay/unfilled counts compactly.
5. Preserve layout at desktop and mobile widths and avoid a new decorative panel.
6. Bump the frontend cache key only if the repository's delivery pattern requires it.

**Verification:** Run frontend contract tests, `node --check dashboard/frontend/app.js`,
and browser verification on desktop and mobile.

**Commit:** `feat(frontend): show A-share market-rule outcomes`

## Task 7: End-to-End and Regression Validation

**Steps:**

1. Run all focused iFinD, execution, baseline, database, API, and frontend tests.
2. Run the complete backend test suite.
3. Run frontend syntax checks and `git diff --check`.
4. Start ATL with a temporary database and fixed offline fixtures.
5. Manually verify one normal fill, each of the three market-rule rejection states, and
   one earlier bar on a closing-limit day that remains eligible.
6. Verify no fees/cash/position mutation on rejected rows and correct retry behavior in
   baseline metadata.
7. Run one controlled live iFinD rule-data fetch and, if a suitable historical event is
   available, one local A-share backtest. Keep credentials and raw response data local.
8. Inspect `git status`, changed-file list, and sensitive-file patterns before commit or
   push.

**Stop condition:** Do not open a PR while any focused/full test fails, UI audit output is
ambiguous, or the production iFinD command remains unverified.

## Task 8: Sync, Push, and Open the PR

**Steps:**

1. Fetch `origin/main` and inspect intervening commits.
2. Rebase the feature branch when overlap is safe; resolve with full focused regression
   tests if overlap exists.
3. Push `feat/ashare-market-rules`.
4. Open a new English PR; do not amend merged PR #317.
5. Describe official data dependency, strict-failure behavior, Agent/baseline parity,
   simulation-only scope, and regression evidence.
6. Wait for GitHub checks and address failures before handoff.

**PR title:** `feat(backtest): enforce A-share daily market rules`
