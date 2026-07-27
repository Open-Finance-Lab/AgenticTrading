# iFinD backtest defaults and diagnostics implementation plan

**Goal:** Keep the shared `$1,000` default while giving iFinD a valid one-month data window, aligning its default decision source with Alpaca LLM runs, and making short-data and zero-trade outcomes explicit.

**Design:** `docs/superpowers/specs/2026-07-25-ifind-backtest-demo-defaults-design.zh-CN.md`

**Constraints:** Do not change the deterministic 2% sizing algorithm, do not add FX conversion or A-share lot rules, do not commit credentials or `dashboard/storage/data/backtest.db`, and keep all UI copy in English.

## Task 1: Lock the behavior with failing tests

Files:

- `dashboard/backend/tests/test_ifind_ashare_frontend.py`
- `dashboard/backend/tests/infrastructure/market_data/test_market_profiles.py`

Add assertions for the one-month iFinD constants, date preservation/restoration, unchanged capital, LLM defaults for both iFinD profiles, `minimum=50`/`valid bars` error classification, and the completed zero-trade empty state. Run the focused tests and confirm they fail for the expected missing behavior.

## Task 2: Implement market and UI defaults

Files:

- `dashboard/backend/infrastructure/market_data/profiles.py`
- `dashboard/frontend/app.js`
- `dashboard/frontend/app.html`

Set both iFinD profiles to default to LLM while retaining Rule-based in their allow-lists. When entering iFinD, save the existing dates and apply `2026-04-01` through `2026-05-01`; do not touch the capital input. Restore dates on exit. Preserve the active Agent model and update the hint text accordingly.

## Task 3: Implement actionable diagnostics

File:

- `dashboard/frontend/app.js`

Classify the adapter's actual `valid bars; minimum=50` error before generic response-format handling. Use a specific successful-run empty message stating that the selected strategy produced no executable orders. Do not modify calculated metrics or synthesize trades.

## Task 4: Automated verification

Run focused frontend/profile/reference-agent tests, then the broader iFinD and backtesting suites. Run formatting/diff checks and confirm the database and credentials are not staged.

## Task 5: Real iFinD and browser verification

Start the PR worktree server on a free port using the existing local environment without printing secrets. Run the recommended iFinD window with the configured Agent model and inspect the completed page. Also run Rule-based with `$1,000` to verify that zero trades are reported as a valid, explained result. Check desktop and narrow viewport layout for non-overlap.

## Task 6: Deliver through PR #214

Commit the implementation and tests, push `feat/ifind-ashare-integration`, inspect PR #214, and wait for required CI checks. Report exact automated and browser verification results, including any external service limitation.
