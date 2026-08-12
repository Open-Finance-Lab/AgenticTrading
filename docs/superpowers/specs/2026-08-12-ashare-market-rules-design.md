# A-Share Daily Market Rules Design

**Date:** 2026-08-12  
**Status:** Product design approved; written-spec review pending  
**Scope:** iFinD-backed A-share historical backtests and paper simulation

## Goal

Make ATL's iFinD A-share backtests enforce official daily suspension and price-limit
rules before applying the existing board-lot, T+1, cash, position, slippage, and fee
rules. The result must remain deterministic, auditable, and simulation-only.

This iteration covers:

- full-day suspension status;
- official daily upper-limit and lower-limit prices;
- direction-aware order blocking;
- the same rules for Agent portfolios and the buy-and-hold baseline;
- API, persistence, and English UI audit output; and
- strict failure when official market-rule data is unavailable or incomplete.

This iteration does not cover intraday temporary suspensions, auction queues,
order-book liquidity, partial queue fills, live A-share paper trading, broker
connectivity, or real-money orders.

## Confirmed Product Decisions

1. Official iFinD rule data is authoritative. ATL must not infer suspension or price
   limits from OHLCV alone.
2. A missing, malformed, unauthorized, or incomplete rule response fails the A-share
   backtest before execution. ATL must never silently fall back to unrestricted fills.
3. A full-day suspension blocks both buys and sells.
4. At the official upper-limit price, buys are blocked and sells may continue through
   the remaining execution checks.
5. At the official lower-limit price, sells are blocked and buys may continue through
   the remaining execution checks.
6. The buy-and-hold baseline follows the same market rules. If its initial buy is
   blocked, it retries at the next eligible bar instead of bypassing the rule or
   abandoning the position permanently.
7. All repository documentation, code-facing text, API reasons, and UI copy are in
   English. No credential or unsanitized iFinD response is persisted or logged.

## Data Architecture

### Separate price data from rule data

The existing iFinD provider continues to return normalized 60-minute OHLCV frames.
A second daily data path fetches official market-rule observations for the same
registered universe and date range.

The adapter normalizes the daily response into an immutable lookup keyed by symbol and
market date:

```text
(symbol, trading_date) -> DailyMarketRule
```

`DailyMarketRule` contains:

- `symbol`;
- `trading_date` in `Asia/Shanghai`;
- `suspended`;
- `upper_limit_price` in native CNY;
- `lower_limit_price` in native CNY; and
- source/version metadata sufficient to audit which rule contract was used.

The rule calendar remains separate from each hourly DataFrame. Daily values are not
duplicated across four bars, and a suspension day remains representable even when
iFinD omits all OHLCV rows for that date.

### Official iFinD command verification

iFinD's public documentation states that supported indicator names and generated
commands are obtained through SuperCommand. Before production code fixes any indicator
name, the implementation loop must generate or inspect the authorized command and
capture a sanitized response shape. Tests use fixtures derived from that shape; no
access token, refresh token, account identifier, or full raw production response is
committed.

The client may use one or more official iFinD endpoints, but the normalized domain
contract above must not expose vendor-specific field names to the execution engine.

## Validation and Strict Failure

Before a run starts executing orders, ATL validates the entire requested A-share rule
calendar. Required market dates are the `Asia/Shanghai` dates present in the combined
registered-universe backtest clock, not the dates present in each symbol's own OHLCV
frame. Therefore, a suspended symbol still requires an explicit rule on a date when
other symbols supply the hourly clock.

1. every registered symbol is represented;
2. every registered symbol has a rule for every required market date;
3. suspension values are unambiguous booleans;
4. active dates have finite, positive upper- and lower-limit prices;
5. `lower_limit_price < upper_limit_price`; and
6. prices conform to the A-share CNY price tick after normalization.

If authentication, permission, transport, response-shape, coverage, or value validation
fails, the provider raises a sanitized market-rule-data error. The API maps it to the
existing iFinD backtest error boundary with a user-facing English message containing
`Market rule data unavailable`. The run does not enter order execution and does not
produce a misleading partial result.

Logs may include the endpoint category, error class, symbol count, and date range. They
must not include credentials, request headers, full raw payloads, or sensitive account
details.

## Execution Semantics

The shared executor receives an optional daily rule. Non-iFinD profiles pass no rule and
retain their current behavior.

For iFinD A-share orders, checks run in this order:

1. required rule exists and is valid;
2. full-day suspension;
3. direction-aware price-limit gate;
4. existing board-lot validation;
5. existing T+1, holdings, and available-position validation;
6. existing cash and affordability validation; and
7. existing slippage, fees, cash movement, and fill recording.

The price-limit comparison uses the transaction's reference execution price and the
official native-CNY limit prices, with tick-safe decimal comparison. It does not infer a
limit event merely because a bar's high or low touched a boundary.

Rejection reasons are stable machine-readable English values:

- `suspended`;
- `limit_up_buy_blocked`; and
- `limit_down_sell_blocked`.

A market-rule rejection executes zero shares, charges zero costs, and leaves cash,
positions, T+1 balances, and cost totals unchanged. Its audit record includes the
applicable official limit prices and rule date.

## Agent and Baseline Flow

The Agent portfolio queries the rule calendar for the order symbol and the market date
of each hourly decision before calling the shared executor.

The buy-and-hold baseline uses the same calendar and execution semantics. Initial target
allocations remain pending per symbol when a buy is blocked by suspension or the upper
limit. The baseline retries the pending allocation on later bars until it fills or the
run ends. A retry uses the then-current price, market rule, available cash, lot size, and
transaction costs; it does not backdate a fill.

Baseline rule rejections are auditable but do not pollute the Agent's trading log or
Agent rejection totals. Baseline-specific summary metadata reports delayed or unfilled
initial allocations where needed for result interpretation.

## Persistence and API Contract

Order-event persistence in SQLite and PostgreSQL adds nullable market-rule audit fields
without rewriting historical runs:

- rule date;
- suspended flag;
- native upper-limit price; and
- native lower-limit price.

Existing reporting-currency conversion remains separate. Official price limits are
stored in native CNY and may additionally be converted for reporting when the current
currency audit layer supports that field.

Run metadata includes:

- market-rule profile/version;
- market-rule enforcement enabled state;
- counts for suspension, upper-limit buy, and lower-limit sell rejections; and
- baseline delayed/unfilled allocation counts when non-zero.

Older runs and Alpaca/vn.py runs keep their current response shape unless nullable or
optional fields are already part of the shared API model.

## User Interface

The existing backtest Run config displays a compact line:

```text
A-share market rules: Enabled
```

The Trading Log renders market-rule rejections with concise English labels:

- `Suspended`;
- `Buy blocked at upper limit`; or
- `Sell blocked at lower limit`.

The row exposes the official native-CNY upper and lower prices when applicable. Rejected
orders continue to show zero executed shares and no transaction costs.

The run summary displays non-zero counts for the three rejection categories. It must not
introduce a new decorative dashboard or expose vendor payload details.

## Testing Strategy

All automated iFinD tests use fixed, sanitized offline fixtures. No CI test calls a live
iFinD endpoint.

Coverage includes:

1. adapter normalization for active, suspended, malformed, missing-symbol, and
   missing-date responses;
2. strict provider failure for transport, authentication, permission, schema, and
   incomplete-coverage errors;
3. suspended BUY and SELL rejection;
4. upper-limit BUY rejection and SELL continuation;
5. lower-limit SELL rejection and BUY continuation;
6. normal-price orders retaining current lot-size, T+1, cash, and cost behavior;
7. zero fees and zero portfolio mutation for rule rejections;
8. tick-safe price comparisons;
9. buy-and-hold retry on the next eligible bar and clear end-of-run unfilled metadata;
10. SQLite/PostgreSQL migration and parity tests;
11. API serialization and sanitized error mapping;
12. English Trading Log and Run config rendering; and
13. Alpaca and vn.py regression tests proving no A-share rule leakage.

After offline coverage passes, a controlled local validation may call the authorized
iFinD account for a narrow universe/date range. The operator verifies the sanitized
shape, one normal day, and any available historical suspension or limit case. Local
credentials and captured raw responses remain outside Git.

## Acceptance Criteria

The iteration is complete when:

- official iFinD daily rule data gates every iFinD A-share Agent order;
- official rules also gate buy-and-hold initial allocations with later retries;
- incomplete official rule data prevents the run from starting;
- all three rejection types are persisted and visible in English;
- rejected orders do not mutate the portfolio or transaction-cost totals;
- existing A-share lot-size, T+1, FX, and cost behavior remains intact;
- Alpaca and vn.py behavior remains unchanged;
- focused and full backend tests pass;
- frontend syntax and rendering checks pass; and
- no credential, raw sensitive response, or local database is committed.
