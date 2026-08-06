# iFinD A-Share Lot Execution and Capital Design

## 1. Background

ATL can load historical A-share bars from iFinD and run a native-CNY ledger
reported in USD through historical USD/CNY rates. PR #272 added T+1 execution
semantics to both registered iFinD universes:

- `A-Share Demo 6`
- `CSI 300 Sample 20 (2026 H2)`

The shared executor does not yet model this project's uniform A-share rule that
buy and sell quantities must be positive multiples of 100 shares. The
rule-based Agent can therefore request one or two shares, and the Trading Log
shows fills without explaining attempts rejected by lot, cash, position, or
T+1 constraints.

This design adds market-configured lot validation, auditable insufficient-cash
rejections, and one canonical outcome per attempted order. ATL remains a
historical backtest and simulation platform; it does not connect an A-share
broker or submit real orders.

## 2. Prerequisite and Branch Strategy

Implementation starts from `main` after PR #272 because it reuses the T+1
position state and `rejected_orders` audit contract.

- Do not append this feature to PR #272.
- Do not open a stacked code PR against an unmerged T+1 branch.
- After PR #272 merges, create a separate feature branch from the latest
  `main`.
- Keep this design and its implementation plan with that independent feature.

## 3. Goals and Non-Goals

### 3.1 Goals

1. Require buy and sell quantities in both iFinD universes to be positive
   multiples of 100 shares.
2. Reject invalid lots in full without rounding or rewriting the Agent's
   original request.
3. Make a rule-based A-share buy signal request exactly one 100-share lot.
4. Evaluate lot affordability in the native-CNY ledger using the selected
   backtest capital and historical FX rate.
5. Let a backtest complete successfully with zero fills when no lot is
   affordable.
6. Show filled, partially filled, and rejected attempts in the Trading Log.
7. Apply the same executor rules to rule-based and LLM backtests.
8. Preserve Alpaca, vn.py, and existing trade/return API behavior.

### 3.2 Non-Goals

This iteration does not implement:

- automatic rounding to the nearest lot;
- automatic capital increases;
- partial BUY execution when cash cannot cover the full request;
- odd-lot sell exceptions;
- board-specific STAR Market quantity rules;
- price limits, suspensions, commissions, stamp duty, or slippage;
- live A-share paper trading, broker orders, or real-money trading;
- changes to Alpaca or vn.py quantity rules; or
- a higher backtest-capital limit.

## 4. Architecture

Use a market-rule configuration plus one shared executor.

`MarketProfile` gains a lot-size setting:

```text
lot_size = 100  # both registered iFinD A-share universes
lot_size = 1    # Alpaca, vn.py, and existing markets
```

The market profile owns quantity rules. Rule-based, LLM, and future external
Agents must not copy their own final validation. The executor activates the new
lot gate only when `lot_size > 1`. A value of `1` preserves the market's
existing quantity semantics, including existing fractional behavior.

This boundary:

- keeps the rule with its market;
- routes every decision source through one validator;
- supports later market-specific configuration without scattered
  `data_source == "ifind"` branches; and
- allows profile, execution, ledger, API, and UI tests to remain independent.

## 5. Capital and Currency Semantics

The synchronized product limits are:

- minimum: `$1`;
- default: `$1,000`;
- maximum: `$3,000`.

iFinD backtests continue to use `CurrencyContext`:

1. The request supplies reporting capital in USD.
2. The backtest converts it to native CNY using the historical rate at the
   start of the run.
3. Buy cost, available cash, and lot affordability are evaluated in CNY.
4. APIs and the UI retain USD reporting values plus native-CNY and FX audit
   fields.

For example, `$1,000` at 7.2 becomes approximately `CNY 7,200`. A 100-share lot
at `CNY 50` costs `CNY 5,000` and is affordable. A lot at `CNY 180` costs
`CNY 18,000` and is rejected. The system must not treat `$1,000` as
`CNY 1,000` or raise the capital automatically.

## 6. Order Execution Flow

Each recognized BUY or SELL passes through:

```text
Agent order
  -> MarketProfile
  -> base field validation
  -> lot validation
  -> BUY cash validation or SELL position/T+1 validation
  -> filled, partial, or rejected outcome
  -> ledger mutation for fills and one order event for the attempt
```

### 6.1 Lot Validation

An iFinD quantity must be:

- a positive finite integer; and
- divisible by 100.

Quantities such as 50, 150, and 250.5 are rejected in full with
`invalid_lot_size`. Validation does not convert them to 0, 100, or 200.

Lot validation has precedence over cash and T+1 checks. For example, a 50-share
BUY with insufficient cash records `invalid_lot_size`, avoiding unstable
multiple primary reasons for one order.

### 6.2 BUY

When the rule-based Agent sees an A-share BUY signal, it requests 100 shares
instead of deriving a fractional-market position from
`total_equity * 0.02 / price`. The shared executor owns the cash gate so an
unaffordable signal remains auditable.

An LLM may request one or more full lots. The executor does not alter the
quantity. BUY remains all-or-none: cash must cover the complete request or the
order is rejected with `insufficient_cash_for_lot`. A 200-share request is not
partially filled when cash covers only 100 shares.

Insufficient cash is not a backtest exception. It changes no cash, position,
return, or equity state, and the full run may finish with zero trades.

### 6.3 SELL

A SELL passes lot validation before PR #272's available-position and T+1
checks:

- fully sellable quantity: `filled`;
- only part of the request sellable: `partial`;
- all requested shares frozen by same-day buys: `rejected / t1_frozen`; and
- request beyond total position: remaining quantity uses
  `insufficient_position`.

Normal buys, releases, and sells are multiples of 100, so a T+1 partial fill
also remains a full lot. Unfilled quantities do not enter `trades` or increase
the trade count.

One SELL can include both frozen and nonexistent shares. `rejected_orders`
retains the detailed reason components. The single `order_event` uses
`t1_frozen` as its primary reason when any requested shares are frozen, and
uses `insufficient_position` otherwise.

## 7. Status and Reason Contract

### 7.1 Status Values

The domain and API use:

- `filled`
- `partial`
- `rejected`

The frontend displays:

- `FILLED`
- `PARTIAL`
- `REJECTED`

### 7.2 Reason Values

| Machine value | English UI label |
|---|---|
| `invalid_lot_size` | `Invalid lot size` |
| `insufficient_cash_for_lot` | `Insufficient cash for one lot` |
| `t1_frozen` | `T+1 frozen` |
| `insufficient_position` | `Insufficient position` |

A filled order has no rejection reason. The Agent's explanation remains in
`strategy_reason`; it must not share the executor's reason field.

## 8. Trades, Rejections, and Order Events

Existing semantics remain:

- `trades` stores actual fills used by cash, positions, returns, trade count,
  and the equity curve;
- `rejected_orders` stores detailed unfilled audit records compatible with
  PR #272; and
- `order_events` records order outcomes, with exactly one event per attempted
  order in memory.

**Only the non-filled events are persisted or published.** The executor's
in-memory ledger is complete, but `engine._unfilled_order_events` drops
`status == "filled"` at the persistence boundary, because a fill is already a
row in the uncapped `trades` table and copying it into the bounded
`agent_runs.metadata` sample would (a) duplicate the run's largest table into
one JSON cell on free-tier Postgres, carrying the `[LLM] <reasoning>` prose a
second time, and (b) make the cap *lossy* — a busy run's fills would push its
own rejections out of the persisted window, and the Trading Log would silently
show the oldest 200 orders of a run that placed thousands.

A repeated rejection is collapsed per `(symbol, side, reason, trading_date)`
and carries `repeat_count`. An unaffordable signal re-fires on every bar for as
long as the indicator holds; without collapsing, those duplicates fill the head
sample end to end, so the audit is least informative exactly when the
constraint bound hardest. This is the same bound `t1_deferrals` uses. Fills and
partial fills never collapse — each moved the ledger.

An order event contains at least:

```json
{
  "timestamp": "2026-04-01T10:00:00+08:00",
  "symbol": "600519.SH",
  "side": "BUY",
  "requested_shares": 100,
  "executed_shares": 0,
  "unfilled_shares": 100,
  "price": 250.0,
  "executed_value": 0.0,
  "status": "rejected",
  "reason": "insufficient_cash_for_lot",
  "strategy_reason": "Buy signal",
  "native_price": 1800.0,
  "native_value": 0.0,
  "fx_rate": 7.2
}
```

`price` and `executed_value` use the reporting currency. Cross-currency events
also carry `native_price`, `native_value`, and `fx_rate`. A fully rejected
order has zero executed value in both currencies.

A partial SELL creates one real trade, one detailed unfilled audit, and one
`partial` order event. The Trading Log **merges** `trades` with `order_events`
rather than consuming either alone: preferring one wholesale hides real rows
either way — take `order_events` alone and every fill disappears, take `trades`
alone and every rejection does. The partial is the one order present in both,
so its event (a strict superset of the trade row) replaces the trade rather
than adding a second row, matched on `(timestamp, symbol, side)`. A rejection
executed nothing and therefore has no trade row to collide with.

Final results, run metadata, and live progress expose bounded order-event data.
Old clients may ignore new fields. The existing trades endpoint retains
`trades` and its original `count` meaning while adding separate order-event
fields.

## 9. Trading Log

Keep the existing Trading Log section and use these eight columns:

| Time | Action | Company / Asset | Quantity | Price | Total Value | Status | Reason |
|---|---|---|---|---|---|---|---|

Quantity displays actual fill over request:

- `100 / 100 shares`: filled;
- `100 / 200 shares`: partial; and
- `0 / 50 shares`: rejected.

`Total Value` represents the actual fill only. A full rejection displays `--`
because no cash moved. Cross-currency fills continue to show USD reporting and
native-CNY audit values.

Rename `All Trades` to `All Orders`; keep `Buys Only` and `Sells Only`. All new
visible copy is English.

## 10. Error Handling and Compatibility

1. Invalid lots and insufficient cash are normal simulated outcomes, not HTTP
   500 errors or failed runs.
2. Rejections do not mutate cash, positions, returns, trade count, or equity.
3. Zero fills is a valid successful backtest with a flat equity curve and
   auditable order outcomes.
4. Historical runs without `order_events` fall back to existing `trades`; no
   database migration is required.
5. Existing `lot_size=1` markets retain their current *execution* behavior and
   do not emit A-share-specific rejection reasons (`insufficient_cash_for_lot`
   stays A-share-only; single-share markets use `insufficient_cash`). They do
   now emit order events for unfilled orders: an "All Orders" log that silently
   drops DJIA's unaffordable buys is the same fail-closed-but-invisible gap the
   A-share path exists to close. `rejected_orders` remains A-share-only.
6. When the persisted sample is capped, the Trading Log renders an explicit
   "N more unfilled orders are not shown" row. A truncated log that reads as a
   complete one is the failure mode; the cap itself is not.
7. APIs, logs, fixtures, and pages must never expose iFinD, Alpaca, or LLM
   credentials.

## 11. Test Plan

### 11.1 Profile and Executor Unit Tests

Cover:

1. Both iFinD profiles use `lot_size=100`.
2. Alpaca and vn.py do not enable the A-share lot gate.
3. Quantities 100 and 200 pass.
4. Quantities 50, 150, fractional, malformed, and non-finite fail with
   `invalid_lot_size`.
5. Lot failure takes precedence over cash failure.
6. An affordable 100-share BUY fills.
7. An unaffordable lot records `insufficient_cash_for_lot` without ledger
   mutation.
8. A 200-share request affordable only at 100 shares is rejected in full.
9. A rule-based A-share BUY requests 100 shares.
10. An invalid LLM quantity reaches and is rejected by the same executor.

### 11.2 T+1 Composition Tests

Cover:

1. Buy 100 and sell 100 on the same day: `REJECTED / T+1 frozen`.
2. Request 200 with only 100 sellable: `PARTIAL / T+1 frozen`.
3. Sell 100 after the next actual trading-day release: `FILLED`.
4. Filled shares enter `trades`; frozen shares enter only audit data.
5. A rejection does not alter cash, return, or equity.

### 11.3 API and Frontend Tests

Cover:

1. Final results, metadata, and live progress expose order events.
2. Each attempted order appears exactly once in the Trading Log.
3. Status and English reason mappings are stable.
4. Quantity shows executed and requested shares.
5. A fully rejected Total Value displays `--`.
6. All Orders, BUY, and SELL filters work.
7. Historical runs without events still display fills.

### 11.4 Capital, FX, and Real-Data Acceptance

Deterministic fixtures verify:

- `$1,000` converts at the historical rate and can buy a low-price lot;
- an unaffordable high-price lot yields a successful zero-fill run;
- a user-selected amount above `$1,000`, up to `$3,000`, is honored; and
- native-CNY cash checks match USD reporting results.

Automated tests do not use real credentials. A local developer may run one real
iFinD historical backtest to inspect the curve, Trading Log, capital, FX, and
rejection display. Credentials must never enter the repository, test output,
screenshots, or PR.

### 11.5 Regression

Run the full backend and packaging suites plus relevant vn.py and frontend
tests. Confirm:

- Alpaca and vn.py fills do not change;
- existing `trades` and `rejected_orders` contracts stay compatible;
- rule-based and LLM A-share paths share one market rule; and
- rejected attempts are never counted as trades.

## 12. Acceptance Criteria

The feature is complete only when:

1. Both iFinD universes execute only positive 100-share multiples.
2. A rule-based A-share BUY requests one lot.
3. Invalid lots are not rounded and display `Invalid lot size`.
4. Insufficient cash does not fail the run and displays
   `Insufficient cash for one lot`.
5. T+1 full rejection and partial execution remain correct.
6. Each attempted order appears once with one stable status.
7. Default `$1,000` and selected higher capital up to `$3,000` use historical
   FX correctly.
8. A zero-fill run returns a valid result and equity curve.
9. Rejections do not change trade count, return, cash, or equity.
10. Alpaca, vn.py, and historical runs remain compatible.
11. Automated tests and one local real-iFinD browser acceptance pass.
12. No repository content or output contains credentials.
