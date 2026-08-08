# A-Share Transaction Cost Model Implementation Plan

## Implementation Base

Start from `origin/main` at the PR #289 merge (`0560658`), not the stale local
`main` branch. Create a new feature branch from that commit. Carry the approved design
commit (`544b05e`) and this plan onto the feature branch. Preserve the pre-existing
`dashboard/storage/data/backtest.db` worktree change; it is unrelated to this feature.

## Scope

Implement the approved deterministic transaction-cost profile for iFinD A-share paper
backtests. Keep Alpaca and vn.py US behavior unchanged, and keep the system paper-only.
Do not add price limits, suspension detection, order-book matching, or user-editable cost
inputs in this plan.

## Work Sequence

### 1. Add the market transaction-cost value object

**Files**

- Modify `dashboard/backend/infrastructure/market_data/profiles.py`.
- Add focused tests in `dashboard/backend/tests/infrastructure/market_data/test_market_profiles.py`.

**Changes**

- Add an immutable `TransactionCostProfile` with explicit rates, minimum commission,
  slippage directions, price tick, currency, and a stable profile version.
- Attach a cost profile to A-share `MarketProfile` entries with these defaults:
  commission `0.00025`, minimum `CNY 5`, sell stamp duty `0.0005`, two-sided transfer
  fee `0.00001`, buy/sell slippage `0.0005`, and `CNY 0.01` price tick.
- Represent US profiles as legacy/no-cost profiles so the execution path can branch
  without changing US numeric behavior.
- Expose a JSON-safe metadata representation for the selected profile.

**Acceptance**

- Both A-share universes resolve the same cost profile.
- Alpaca and vn.py profiles resolve no-cost/legacy behavior.
- Existing market-profile and provider-gating tests remain green.

### 2. Implement deterministic cost calculation in the shared executor

**Files**

- Modify `dashboard/backend/domain/trading/execution.py`.
- Extend `dashboard/backend/tests/domain/trading/test_execution.py`.

**Changes**

- Add a pure calculation helper that receives side, reference price, filled quantity,
  selected profile, and order identity/state, and returns adjusted price, gross value,
  each fee, and net cash impact.
- Apply adverse-direction slippage before rounding to the A-share `CNY 0.01` tick:
  buy prices round upward and sell prices round downward.
- Use `Decimal`-based cent rounding for fees and cash deltas. Apply the `CNY 5` minimum
  commission once for one submitted order, including an order that fills in pieces.
- Keep existing `cost` and `proceeds` fields as gross traded value for compatibility;
  add explicit `gross_value`, `net_cash_impact`, `reference_price`, `slippage_amount`,
  `commission`, `stamp_duty`, and `transfer_fee` fields to A-share fills and order events.
- Check buy affordability using gross value plus all buy-side fees. Charge fees only for
  the actually filled quantity. Rejected and cancelled orders receive zero fees.
- Preserve the existing T+1, lot-size, rejection, and order-event semantics. A T+1
  partial sell remains one submitted action and is charged once for its actual fill.
- Ensure malformed prices and quantities produce explicit A-share rejection/error records,
  without changing the legacy US branch's behavior.

**Acceptance**

- The helper is deterministic and side-effect free.
- A-share cash never becomes negative because fees were omitted from the affordability
  check.
- A rejected order does not mutate positions, cash, or cost totals.
- Existing legacy execution tests still pass when the profile is disabled.

### 3. Wire the cost profile through portfolio and backtest orchestration

**Files**

- Modify `dashboard/backend/domain/backtesting/portfolio_manager.py`.
- Modify `dashboard/backend/domain/backtesting/engine.py`.
- Modify `dashboard/backend/baseline_generator.py` for the A-share initial buy allocation.
- Add or update tests under `dashboard/backend/tests/backtesting/`.

**Changes**

- Pass the resolved `MarketProfile` (or its cost profile) when constructing
  `PortfolioManager` in `HourlyBacktester.run_agent_backtest`.
- Pass the profile into `_execute_actions` while retaining the current A-share T+1 and
  lot-size state arguments.
- Maintain per-run native-CNY cost totals, including gross traded value, commission, stamp
  duty, transfer fee, slippage, and total transaction costs.
- Add the versioned cost configuration and aggregated totals to A-share run metadata.
- Keep `native_initial_capital`, equity valuation, and the existing CNY-to-USD reporting
  conversion unchanged.
- Apply the same calculator to each explicit initial buy in the A-share buy-and-hold
  baseline, treating one symbol allocation as one order. Reduce native cash by the
  resulting buy-side costs and persist aggregate baseline costs in that run's metadata;
  the baseline does not need to manufacture Trading Log rows. Do not change the US
  baseline path.

**Acceptance**

- A-share agent runs carry an auditable cost profile and totals.
- T+1 and lot-size records still appear exactly as before, with costs attached only to fills.
- US and vn.py runs have the old cash/equity behavior and no A-share metadata.

### 4. Preserve native values and expose reporting-currency cost fields

**Files**

- Modify `dashboard/backend/domain/backtesting/currency.py`.
- Modify `dashboard/backend/domain/backtesting/engine.py` serialization helpers.
- Extend `dashboard/backend/tests/backtesting/test_currency_context.py`.

**Changes**

- Extend trade and order-event reporting conversion to convert every monetary cost field
  from native CNY to reporting USD while preserving `native_*` audit fields and `fx_rate`.
- Keep rejected orders' executed value and all monetary costs at zero; never derive fees from
  requested quantity.
- Keep identity USD contexts byte-compatible for Alpaca and vn.py.
- Round displayed monetary values consistently after conversion without changing native
  ledger arithmetic.

**Acceptance**

- A-share logs can show both CNY audit values and USD reporting values.
- Currency tests prove no future FX rate is used and no rejected order looks paid.
- Existing identity-context equality tests remain green.

### 5. Surface cost details in the API and Trading Log

**Files**

- Modify `dashboard/backend/api/routers/backtests.py` only if response models or detail
  endpoints need new cost fields.
- Modify `dashboard/frontend/app.js` and, if required, `dashboard/frontend/styles.css`.
- Extend `dashboard/backend/tests/test_backtests_router.py` and
  `dashboard/backend/tests/test_trading_log_order_events_ui.py`.

**Changes**

- Keep the existing complementary `trades`/`order_events` response contract and add the
  cost fields without duplicating fills into bounded metadata samples.
- Normalize cost fields in the frontend order-record adapter.
- Show a compact English cost breakdown in A-share Trading Log rows or a row detail affordance:
  commission, stamp duty, transfer fee, slippage, and net cash impact. Do not add user cost
  inputs to the launch form.
- Show the selected cost profile/version and aggregate costs in the run result metadata area.
- Keep USD formatting for the main reporting values and CNY formatting for native audit lines.
- Preserve existing empty, partial, rejected, truncation, and filter states.

**Acceptance**

- A completed A-share run visibly explains its transaction costs.
- A rejected order shows no misleading fee or cash impact.
- US/vn.py Trading Log rendering remains unchanged apart from harmless zero-cost omission.

### 6. Add complete regression and integration coverage

**Files**

- Add a focused cost-profile test module if the existing execution test becomes too large.
- Extend `dashboard/backend/tests/integration/test_ifind_ashare_backtest.py`.
- Extend `dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py`.
- Extend existing Alpaca/vn.py execution and engine regression tests as needed.

**Test cases**

- Exact buy and sell calculations, including minimum commission and sell-only stamp duty.
- Buy and sell price-tick rounding after slippage.
- Insufficient cash after fees, invalid prices, invalid quantities, and malformed bars.
- T+1 partial sells, unavailable holdings, repeated rejection deduplication, and zero-fee
  rejected orders.
- Deterministic repeated fixture runs with identical cost breakdown and metadata.
- A-share integration with a fixed normalized-bar fixture; never call a live trading endpoint.
- Alpaca and vn.py regression runs proving no A-share rules or costs leak into US markets.

**Suggested commands**

```bash
pytest -q dashboard/backend/tests/domain/trading/test_execution.py
pytest -q dashboard/backend/tests/backtesting/test_currency_context.py
pytest -q dashboard/backend/tests/backtesting/test_ifind_ashare_engine.py
pytest -q dashboard/backend/tests/integration/test_ifind_ashare_backtest.py
pytest -q dashboard/backend/tests/test_backtests_router.py dashboard/backend/tests/test_trading_log_order_events_ui.py
```

## Commit and Review Boundaries

- Keep commits small and in English, for example:
  - `feat(backtest): add A-share transaction cost profile`
  - `feat(backtest): apply costs in shared execution`
  - `feat(frontend): display A-share transaction costs`
  - `test(backtest): cover A-share cost accounting`
- Do not include credentials, Render configuration, generated databases, or unrelated
  synchronization changes.
- Before opening a PR, run the focused tests plus the relevant full backend suite, inspect
  `git diff --check`, and manually run one fixed-fixture A-share paper backtest to verify the
  cost breakdown and curve.
