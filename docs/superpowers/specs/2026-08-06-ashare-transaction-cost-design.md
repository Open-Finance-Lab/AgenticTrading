# A-Share Transaction Cost Model Design

## Status

Design approved during brainstorming; implementation has not started. This document is
intended for user review before an implementation plan is written.

## Goals

- Make iFinD-backed A-share paper backtests account for the main deterministic costs of
  the Chinese A-share market.
- Keep all cost calculations in one shared execution path so strategies, providers, and
  the frontend cannot silently apply different rules.
- Preserve the existing behavior of Alpaca US equities and vn.py simulated US equities.
- Make every cost and the exact cost configuration visible and reproducible in run data.
- Keep ATL strictly in backtest and paper-simulation scope; no real broker order is sent.

## Non-goals

This iteration does not implement price-limit enforcement, suspension detection, order-book
matching, queue priority, real broker fills, or new frontend cost controls. These can be
separate A-share realism iterations after this model is stable.

## Existing Context

`MarketProfile` already identifies the market behavior used by a run. The shared execution
layer currently derives a simulated fill from the bar close, while `PortfolioManager`
maintains cash, positions, order events, and A-share lot-size/T+1 state. The new model extends
these boundaries without moving market-specific policy into a strategy or data adapter.

## Architecture

### Market profile

The A-share profile owns a versioned `TransactionCostProfile` with these defaults:

| Cost | Rule |
| --- | --- |
| Commission | `0.025%` of traded value, minimum `CNY 5` per order |
| Stamp duty | `0.05%` of traded value on sells only |
| Transfer fee | `0.001%` of traded value on buys and sells |
| Slippage | Buy price `+0.05%`; sell price `-0.05%` |
| Currency | CNY |

The profile is selected by market configuration, not entered by each user. Alpaca and vn.py
US profiles retain their current behavior and do not inherit A-share costs.

### Shared execution layer

`execution.py` becomes the single place that:

1. validates the order against existing A-share lot-size, T+1, holdings, and cash rules;
2. derives a reference fill price from the current bar close;
3. applies adverse-direction slippage and A-share price-tick rounding;
4. computes gross traded value and each configured cost; and
5. emits either a rejected order or a fill with a complete cost breakdown.

The cash effects are:

```text
buy cash delta  = -(gross value + commission + transfer fee)
sell cash delta = +(gross value - commission - stamp duty - transfer fee)
```

The cash sufficiency check uses the complete buy cash delta, including fees. Rejected and
cancelled orders do not incur costs.

### Ledger, events, and metadata

`PortfolioManager` applies the net CNY cash delta and updates positions only for the filled
quantity. Each fill/order event exposes reference price, execution price, gross value,
slippage amount, commission, stamp duty, transfer fee, and net cash impact. For partial fills,
the minimum commission is applied once per order after aggregating its filled value; each fill
still remains observable. Run metadata stores the complete, versioned cost profile used for
the run.

`calculate_transaction_costs` carries no order identity, so its contract is **one call is one
submitted order** and the `CNY 5` floor is charged once per call. Every caller today executes
an action in a single call, which is what makes that correct; a future piecewise fill must
cost the whole order and take differences between quotes rather than calling per fragment.

Run metadata separates the market's rule from the run's ledger. `transaction_cost_profile` is
provenance and rides every row of that market — including the index reference curve, which is
a price series that places no orders. `transaction_costs_applied` says whether the run
actually paid, so a reference curve cannot be read as a costed book.

### Baseline sleeve under board lots

Whole lots and equal weight do not compose. An equal slice of the accounts this app allows
(`$1,000` default, `$3,000` max) across a 20-name universe is worth less than one 100-share
A-share lot, so flooring each slice independently strands the entire buy & hold sleeve in cash
— a flat line that every agent beats for free and that is indistinguishable from a legitimate
result. The generator therefore allocates in two passes: equal weight capped by each symbol's
own slice, then a poorest-first top-up sweep of the remainder into further whole lots. The
board lot comes from `MarketProfile.lot_size`, never inferred from the presence of a cost
profile — they are separate market rules. `baseline_allocation` in run metadata records
symbols requested/priced/bought and the invested ratio so a partly-placed sleeve is visible
rather than silent.

## Data Flow

```text
iFinD bars -> normalized ATL OHLCV bars -> chronological backtest loop
           -> Agent signal -> shared execution validation
           -> slippage-adjusted fill -> cost calculation
           -> CNY ledger/positions -> order log, equity curve, run metadata
```

iFinD remains a market-data provider only. It supplies prices and timestamps; ATL's execution
layer decides whether and how a simulated order fills. A missing iFinD credential or provider
error does not fall back silently to US or vn.py data.

## Error Handling and Boundaries

- Invalid or missing bar fields fail the affected run with symbol and timestamp context.
- Orders with non-positive values, non-100-share multiples, invalid prices, insufficient
  sellable holdings, T+1 violations, or insufficient post-cost cash are rejected explicitly.
- Costs are charged only against the actually filled quantity.
- Price values use the A-share `CNY 0.01` tick, rounded in the adverse direction after
  slippage. Monetary values use deterministic CNY-cent rounding.
- A-share short selling and fractional shares remain unsupported.
- Price limits, suspensions, and order-book effects remain out of scope for this iteration.

## Testing and Acceptance

### Unit tests

- Verify buy/sell slippage, commission rate, `CNY 5` minimum commission, sell-only stamp
  duty, two-sided transfer fee, price-tick rounding, and cent rounding.
- Verify exact buy and sell net cash deltas and the complete cost breakdown.

### Rule and failure tests

- Reject 50-share orders, same-day sells, sells exceeding available holdings, invalid prices,
  malformed bars, and orders that become unaffordable after costs.
- Verify rejected/cancelled orders produce no cost events.
- Verify partial fills aggregate minimum commission once while preserving per-fill records.

### Integration and regression tests

- Run an iFinD A-share paper backtest and verify CNY cash, positions, trading logs, equity
  curve, and cost metadata.
- Run existing Alpaca and vn.py US tests and confirm no A-share lot-size, T+1, or cost rules
  leak into them.
- Confirm the integration never calls a live trading endpoint.

The iteration is accepted only when all tests pass, a deterministic fixture reproduces the
same cost breakdown across runs, and the UI/API can display the recorded costs without
requiring new user-entered parameters.

## Future Extensions

Future loops may add price-limit and suspension rules, configurable broker fee schedules,
market-specific rounding policies, and richer execution simulation. Those changes should
extend `TransactionCostProfile` rather than duplicate calculations in providers or strategies.
