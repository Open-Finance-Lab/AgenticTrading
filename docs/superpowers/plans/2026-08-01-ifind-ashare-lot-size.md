# iFinD A-Share Lot Execution Implementation Plan

**Goal:** Build on PR #272's T+1 behavior by adding a uniform 100-share lot
gate, auditable insufficient-cash rejections, canonical order outcomes, and a
Trading Log that explains fills and rejections for both iFinD universes. Keep
the `$1,000` default, the current `$3,000` maximum, and existing Alpaca and
vn.py behavior.

**Design:**
`docs/superpowers/specs/2026-08-01-ifind-ashare-lot-size-design.md`

**Method:** For each task, write a failing focused test, make the smallest
implementation change, rerun focused coverage, and commit only named files.
This feature remains independent from PR #272.

## Global Constraints

- ATL provides historical backtests and simulated execution only. Do not
  connect an A-share broker or real funds.
- Target `A-Share Demo 6` and `CSI 300 Sample 20 (2026 H2)`.
- Require positive 100-share multiples for both BUY and SELL; never round an
  invalid request.
- Reject an unaffordable BUY in full; never partially buy or increase capital.
- Preserve PR #272's T+1 and partial-SELL semantics.
- Keep the reporting-capital range at `$1` to `$3,000`, with `$1,000` as the
  default.
- Keep native-CNY accounting through historical USD/CNY rates; never treat
  `$1,000` as `CNY 1,000`.
- Store actual fills in `trades`; rejected quantities must not affect cash,
  positions, returns, equity, or trade count.
- Keep all user-visible copy, comments, documentation, commits, and PR content
  in English.
- Automated tests must not call real iFinD or LLM services.
- Never print, commit, or expose iFinD, Alpaca, OpenRouter, or other LLM
  credentials.
- Do not modify or commit `dashboard/storage/data/backtest.db`; do not use
  `git add -A`.
- Run focused coverage after each stage and complete regression at the end.

---

## Task 0: Confirm T+1 and Create the Independent Branch

No product-code changes.

1. Confirm PR #272 is merged.
2. Fetch the latest healthy `origin/main` in a clean worktree.
3. Create `feat/ifind-ashare-lot-size` from the latest `main`.
4. Bring this design and implementation plan onto the new branch.
5. Confirm the branch contains PR #272 and no database changes.

Checks:

```bash
gh pr view 272 --repo Open-Finance-Lab/AgenticTrading \
  --json state,mergedAt,mergeCommit,statusCheckRollup
git log --oneline --decorate -8
git status --short --branch
git diff --stat origin/main -- dashboard/storage/data/backtest.db
```

The database diff must be empty.

---

## Task 1: Declare and Propagate the Market Lot Rule

**Modify:**

- `dashboard/backend/infrastructure/market_data/profiles.py`
- `dashboard/backend/domain/backtesting/engine.py`
- `dashboard/backend/domain/backtesting/portfolio_manager.py`

**Test:**

- `dashboard/backend/tests/infrastructure/market_data/test_market_profiles.py`
- `dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py`
- `dashboard/backend/tests/backtesting/test_portfolio_manager_move.py`

Steps:

1. Assert `lot_size == 100` for both iFinD profiles and `lot_size == 1` for
   Alpaca and vn.py.
2. Assert `HourlyBacktester` injects the profile lot size into
   `PortfolioManager`.
3. Assert `_llm_market_context()` exposes structured `lot_size` provenance
   **only when `lot_size > 1`**, and assert a DJIA/Alpaca context carries
   neither `lot_size` nor `lot_size_note`. This dict is serialized straight
   into the LLM prompt, so an unconditional key changes every single-share
   market's prompt and makes new runs non-comparable with the historical ones
   already on the leaderboard — the same reason `settlement` is conditional.
4. Assert legacy `PortfolioManager()` construction defaults to `lot_size=1`.
5. Add `lot_size: int = 1` at the end of `MarketProfile` and set 100 only on
   the two iFinD profiles.
6. Store the value in `PortfolioManager`; do not scatter iFinD checks outside
   the profile and executor.
7. Publish optional `lot_size` run metadata for later audits.

Focused tests:

```bash
python -m pytest -q \
  dashboard/backend/tests/infrastructure/market_data/test_market_profiles.py \
  dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py \
  dashboard/backend/tests/backtesting/test_portfolio_manager_move.py
```

Commit:

```bash
git commit -m "feat(backtest): configure A-share lot sizes"
```

---

## Task 2: Produce Auditable Rule-Based and LLM Requests

**Modify:**

- `dashboard/backend/domain/backtesting/reference_agent.py`
- `dashboard/backend/domain/backtesting/portfolio_manager.py`
- `dashboard/backend/infrastructure/llm/backtest_harness.py`
- `dashboard/backend/domain/backtesting/engine.py`

**Test:**

- `dashboard/backend/tests/backtesting/test_reference_agent.py`
- `dashboard/backend/tests/backtesting/test_portfolio_manager_move.py`
- `dashboard/backend/tests/llm/test_backtest_harness.py`
- `dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py`

Steps:

1. Make an A-share rule-based BUY signal request 100 shares instead of a
   2%-of-equity fragment.
2. Submit the raw 100-share request even when cash is insufficient so the
   executor records the rejection.
3. Preserve the exact legacy sizing and cash pre-check for `lot_size=1`.
4. Tell the A-share LLM prompt that `position_size` must be a positive multiple
   of 100, while keeping the executor authoritative.
5. Preserve an A-share LLM quantity such as 100.5 until executor validation;
   do not coerce it with `int()`.
6. Let a valid but unaffordable A-share LLM request reach the executor.
7. **Do** floor a *calculated* missing `position_size` to a whole lot in the
   Agent layer, while still passing a quantity the model actually requested
   through untouched (step 5). The distinction is whose number it is: rounding
   the model's request would be the silent correction the executor refuses to
   make, but the fallback size is the harness's own risk-budget arithmetic, and
   a raw risk budget is essentially never a 100-multiple — submitting it raw
   mints a guaranteed `invalid_lot_size` rejection on every fallback and
   punishes the agent for our arithmetic. Below one lot the correct outcome is
   no order at all.
8. Preserve symbol allow-lists, action limits, confidence thresholds, and the
   existing maximum-share safety cap.

Focused tests:

```bash
python -m pytest -q \
  dashboard/backend/tests/backtesting/test_reference_agent.py \
  dashboard/backend/tests/backtesting/test_portfolio_manager_move.py \
  dashboard/backend/tests/llm/test_backtest_harness.py \
  dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py
```

Commits:

```bash
git commit -m "feat(backtest): preserve A-share LLM lot orders"
```

---

## Task 3: Enforce Lots, Cash, and Outcomes in the Shared Executor

**Modify:**

- `dashboard/backend/domain/trading/execution.py`
- `dashboard/backend/domain/backtesting/portfolio_manager.py`

**Test:**

- `dashboard/backend/tests/domain/trading/test_execution.py`
- `dashboard/backend/tests/domain/trading/test_portfolio_compatibility.py`
- `dashboard/backend/tests/backtesting/test_portfolio_manager_move.py`

Add backward-compatible optional arguments:

```python
lot_size: int = 1
order_events: Optional[List[Dict]] = None
```

Steps:

1. Accept 100 and 200; reject 50, 150, 250.5, malformed, non-finite, and
   boolean quantities with `invalid_lot_size`.
2. Make lot failure precede cash, T+1, and position checks.
3. Fill an affordable 100-share BUY.
4. Reject an unaffordable BUY with `insufficient_cash_for_lot`.
5. Reject a 200-share request in full when cash covers only 100 shares.
6. For a 200-share SELL with 100 sellable, fill 100 and emit
   `partial / t1_frozen`.
7. Preserve detailed rejection components if one SELL contains frozen and
   nonexistent shares; use `t1_frozen` as the single event's primary reason.
8. Fill the released lot on the next trading day.
9. Append exactly one `order_event` per BUY or SELL attempt and none for HOLD.
10. Include timestamp, symbol, side, requested/executed/unfilled shares, price,
    executed value, status, executor reason, and `strategy_reason`.
11. Never store zero-share fills in `trades`.
12. Keep all `lot_size=1` compatibility tests unchanged.

Focused tests:

```bash
python -m pytest -q \
  dashboard/backend/tests/domain/trading/test_execution.py \
  dashboard/backend/tests/domain/trading/test_portfolio_compatibility.py \
  dashboard/backend/tests/backtesting/test_portfolio_manager_move.py
```

Commit:

```bash
git commit -m "feat(backtest): enforce A-share lot execution"
```

---

## Task 4: Convert, Persist, and Publish Order Events

**Modify:**

- `dashboard/backend/domain/backtesting/currency.py`
- `dashboard/backend/domain/backtesting/engine.py`
- `dashboard/backend/api/routers/backtests.py`

**Test:**

- `dashboard/backend/tests/backtesting/test_currency_context.py`
- `dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py`
- `dashboard/backend/tests/test_backtests_router.py`

Steps:

1. Convert native-CNY `price` and `executed_value` to reporting USD while
   retaining `native_price`, `native_value`, and `fx_rate`.
2. Keep both executed values at zero for a fully rejected order.
3. Add `CurrencyContext.reporting_order_event()` rather than recalculating FX
   in the frontend.
4. Add engine serialization for JSON-safe timestamps and numeric values.
5. Publish bounded `trades`, `rejected_orders`, and `order_events` in live
   progress.
6. Persist `lot_size`, event counts, and a bounded sample of the **non-filled**
   events in run metadata (`engine._unfilled_order_events`). Fills are already
   uncapped rows in `trades`; copying them here duplicates the run's largest
   table into one JSON cell and lets a busy run's fills evict its own
   rejections from the sample.
7. Keep `num_trades == len(manager.trades)`.
8. Add separate order-event fields to the trades endpoint without changing
   the existing `count == len(trades)` contract.
9. Perform the existing session-ownership check before returning any audit
   data.

Focused tests:

```bash
python -m pytest -q \
  dashboard/backend/tests/backtesting/test_currency_context.py \
  dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py \
  dashboard/backend/tests/test_backtests_router.py \
  dashboard/backend/tests/integration/test_ifind_ashare_backtest.py
```

Commits:

```bash
git commit -m "feat(backtest): publish bounded order events"
```

---

## Task 5: Show Execution Outcomes in the Trading Log

**Modify:**

- `dashboard/frontend/app.html`
- `dashboard/frontend/app.js`
- `dashboard/frontend/styles.css`

**Test:**

- `dashboard/backend/tests/test_ifind_ashare_frontend.py`
- `dashboard/backend/tests/test_trading_log_order_events_ui.py`

Steps:

1. Use Node-backed tests against the shipped `app.js` normalization and
   rendering functions.
2. Keep eight columns: Time, Action, Company / Asset, Quantity, Price, Total
   Value, Status, and Reason.
3. Rename `All Trades` to `All Orders`; retain BUY and SELL filters.
4. **Merge** `trades` with `order_events` rather than preferring either:
   `trades` is every fill and uncapped, `order_events` is only what did not
   fill and is capped, so preferring one wholesale hides real rows either way.
   Match a `partial` event to its trade on `(timestamp, symbol, side)` and let
   the event replace it — one order must not render as two rows. Historical
   runs carry no events and fall through to `trades` unchanged.
5. Derive `FILLED` for legacy trades.
5b. Render an explicit "N more unfilled orders are not shown" row whenever the
   server reports truncation, and keep that notice across filter changes.
5c. Re-render from the normalized cache via `paintTradingLog`, never by feeding
   normalized records back through `normalizeOrderRecord` — the normalized
   shape uses `requestedShares`, not the wire's `requested_shares`, so a second
   pass zeroes every quantity.
5d. Show `repeat_count` when a rejection was collapsed, so a reader can tell
   "blocked one order" from "blocked the strategy all day".
6. Display Quantity as `executed / requested shares`.
7. Display `--` for a fully rejected executed value and the actual value for a
   partial fill.
8. Map stable reasons to safe English labels and use a fixed fallback for an
   unknown API value.
9. Use compact, accessible status labels and a horizontally scrollable narrow
   layout.
10. Update all empty-state column spans and static-asset cache-busters.

Focused tests:

```bash
python -m pytest -q \
  dashboard/backend/tests/test_ifind_ashare_frontend.py \
  dashboard/backend/tests/test_trading_log_order_events_ui.py \
  dashboard/backend/tests/test_frontend_fast_boot.py
```

Commit:

```bash
git commit -m "feat(frontend): show backtest order outcomes"
```

---

## Task 6: Lock Capital, FX, and Non-A-Share Compatibility

Use deterministic iFinD and FX fixtures to verify:

1. `$1,000` can buy a low-price lot after historical FX conversion.
2. `$1,000` yields a successful zero-fill run when no lot is affordable.
3. A selected value up to `$3,000` is honored, while larger values follow the
   current product cap.
4. Rule-based and LLM iFinD paths both submit full-lot orders.
5. Both iFinD universes share the rule.
6. Alpaca, vn.py, and packaging integration behavior is unchanged.
7. Test output and Git diffs contain no credentials or database changes.

Focused regression:

```bash
python -m pytest -q \
  dashboard/backend/tests/integration/test_ifind_ashare_backtest.py \
  dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py \
  dashboard/backend/tests/backtesting/test_reference_agent.py \
  dashboard/backend/tests/backtesting/test_vnpy_simulation_engine.py \
  dashboard/backend/tests/infrastructure/market_data/test_vnpy_adapter.py \
  dashboard/backend/tests/infrastructure/market_data/test_vnpy_simulation.py \
  dashboard/backend/tests/test_vnpy_simulation_frontend.py \
  packaging/agentictrading/tests/test_vnpy_cta_integration.py
```

---

## Task 7: Full Verification, Real iFinD Acceptance, and PR

1. Run the full backend suite:

```bash
python -m pytest -q dashboard/backend/tests
```

2. Run the packaging suite:

```bash
python -m pytest -q packaging/agentictrading/tests
```

3. Run integrity checks:

```bash
git diff --check
git status --short
git diff --stat origin/main -- dashboard/storage/data/backtest.db
```

4. Start an unused local port with existing environment variables and a
   temporary database. Do not print tokens.
5. Run a real iFinD historical backtest with `$1,000` and Rule-based decisions.
6. Inspect initial/final equity, curve, CNY/USD audit data, Quantity, Status,
   Reason, and All Orders filtering.
7. Inspect desktop and narrow screenshots for overlap and clipping, and inspect
   the browser console for new errors.
8. Use deterministic fixtures for statuses the chosen real history does not
   trigger; never fabricate a real result.
9. Push the existing feature branch and update PR #289 in English.
10. Wait for Backend, Packaging, and CodeQL checks to pass; fix any failure in
    another test-feedback loop.

## Definition of Done

This implementation is complete only when:

1. PR #272 is merged and the feature contains the latest `main`.
2. Both iFinD universes execute only valid positive 100-share multiples.
3. Invalid lots, insufficient cash, and T+1 outcomes are auditable in the
   Trading Log.
4. Default and selected capital use historical FX correctly within the current
   `$3,000` maximum.
5. Rejections do not affect trades, returns, cash, or equity.
6. Alpaca, vn.py, APIs, and historical runs remain compatible.
7. Automated, real-iFinD browser, and GitHub CI verification pass.
8. The PR contains no credentials, local databases, or unrelated artifacts.
