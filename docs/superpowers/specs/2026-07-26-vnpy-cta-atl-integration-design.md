# vn.py CTA Strategy Integration Design

Date: 2026-07-26
Status: Approved for implementation
Target branch: `feat/vnpy-cta-integration`
Original base: `origin/main@3a7781a`

## 1. Context

ATL already validates a data-format path that converts simulated prices through
vn.py `BarData` and back into ATL OHLCV. That path does not run a user's vn.py
strategy. This integration adds the opposite direction:

```text
ATL market data
  -> vn.py BarData
  -> local CtaTemplate strategy
  -> buy/sell signal
  -> ATL typed Order
  -> ATL simulated execution
  -> equity curve, trades, metrics, and an audit artifact
```

## 2. Goals

The first release lets a user run an existing bar-driven vn.py `CtaTemplate`
strategy locally while ATL remains the source of truth for market data,
positions, execution, and performance.

The supported scope is deliberately narrow:

- one US equity, `AAPL`;
- hourly bars;
- long-only `buy` and `sell` actions;
- positive whole-share quantities;
- ATL market-order simulation;
- trusted local strategy execution;
- ATL-hosted curves, trades, and metrics;
- local vn.py-specific audit artifacts.

## 3. Non-goals

The first release does not provide:

- A-shares, futures, multiple symbols, or portfolio strategies;
- short selling, margin, `short`, or `cover` execution;
- TickData or live market feeds;
- vn.py Gateway, Interactive Brokers, TWS, IB Gateway, or broker execution;
- persistent limit orders, stop orders, cancellation, or partial fills;
- full `TargetPosTemplate` semantics;
- vn.py database preload through `load_bar`;
- untrusted server-side Python execution;
- a vn.py-specific leaderboard page;
- identical results to vn.py's native backtesting engine.

## 4. User Flow

The user installs the optional vn.py dependencies in a trusted Python 3.10 or
newer environment and runs:

```bash
python dashboard/examples/vnpy_cta_atl_backtest.py \
  --strategy my_strategy:DoubleMaStrategy \
  --symbol AAPL \
  --start 2026-04-01 \
  --end 2026-04-23
```

ATL credentials are supplied through environment variables. No Alpaca,
Interactive Brokers, or other data-provider credential is read by this client.

The command reports the ATL run and result URLs, common performance metrics,
diagnostic counts, and the path and SHA-256 digest of the local audit artifact.

## 5. Architecture

```text
ATL Run API
  | Observation: current AAPL OHLCV, portfolio, and constraints
  v
VnpyCtaAdapter on the user's machine
  | synchronize the ATL position
  | convert ATL OHLCV to vn.py BarData
  | apply a one-bar delay
  | call CtaTemplate.on_bar()
  | capture buy()/sell()
  v
ATL Decision and typed Order
  |
  v
ATL validation and simulated execution
  | cash and 25% position-weight checks
  | long-only enforcement
  | fills, equity curve, and metrics
  +-> ATL persists common results
  +-> local artifact persists vn.py diagnostics
```

The adapter never maintains a second cash or position ledger. ATL execution
results and each subsequent portfolio observation are authoritative.

## 6. Components

### 6.1 Protocol bars

The existing `observation.market.bars` field is populated with the complete
current OHLCV bar for each allowed symbol that has data at the current step:

```json
{
  "market": {
    "bars": {
      "AAPL": {
        "timestamp": "2026-04-15T10:00:00-04:00",
        "open": 198.1,
        "high": 201.3,
        "low": 197.8,
        "close": 200.75,
        "volume": 1280000.0
      }
    },
    "features": {},
    "events": []
  }
}
```

Timestamps are timezone-aware and match the step timestamp. Prices are finite
and positive; volume is finite and non-negative. The backend returns ordinary
dictionaries and does not depend on vn.py. Existing clients can ignore the
newly populated field, preserving protocol 1.0 compatibility.

### 6.2 `AtlCtaEngine`

This minimal local engine satisfies the `CtaTemplate` boundary without copying
vn.py's full CTA engine. It captures `send_order` calls, assigns stable local
order IDs, records original CTA arguments for audit, reports backtesting mode,
and records unsupported operations explicitly.

It recognizes `Direction.LONG + Offset.OPEN` as a buy and
`Direction.SHORT + Offset.CLOSE` as a sale of an existing long position. It does
not own cash, positions, or execution state.

### 6.3 `VnpyCtaRuntime`

The runtime lazily loads the optional vn.py dependencies, validates compatible
versions, constructs `AAPL.SMART` `BarData`, drives the strategy lifecycle, and
converts ATL fills and rejections into vn.py callbacks.

Lifecycle order is:

```text
construct
  -> on_init()
  -> inited=True
  -> trading=True
  -> on_start()
  -> on_bar()/on_order()/on_trade()
  -> on_stop()
  -> trading=False
```

`load_bar` returns no database history and records
`history_preload_unavailable`. Compatible strategies must warm up from the
stream of `on_bar` calls.

### 6.4 `VnpyCtaAdapter`

The adapter implements the ATL SDK `decide(observation)` boundary. It validates
bar data, synchronizes `strategy.pos` from the ATL portfolio, applies the
one-bar delay, captures CTA orders, maps supported actions into typed ATL
orders, and records every normal HOLD, error HOLD, rejection, timeout, and
unsupported action.

### 6.5 Runner and CLI

`VnpyCtaATLRunner` composes the existing `ATLClient` and `AgentRunner`; it does
not duplicate authentication, retry, idempotency, deadline, or polling logic.
The example CLI imports a trusted local strategy, parses public settings, runs
the backtest, and writes an artifact even when strategy-level errors occur.

## 7. One-bar Execution Delay

Calling `on_bar` after a bar closes and filling at the same close would leak
future information. The adapter therefore uses this state machine:

1. Buffer the first bar and submit a normal HOLD.
2. At step N, synchronize the position from ATL.
3. Pass buffered bar N-1 to `on_bar`.
4. Map captured orders into the decision for step N.
5. Buffer bar N for the next step.
6. Apply ATL fills or rejections to vn.py callbacks.
7. Record `terminal_bar_skipped` for the final bar, which has no later step.

Every executable order has an execution timestamp later than its signal bar.

## 8. Order Mapping

```text
strategy.buy(price, volume)
  -> Order(symbol="AAPL", side="buy", quantity_type="shares",
           quantity=volume, order_type="market")

strategy.sell(price, volume)
  -> Order(symbol="AAPL", side="sell", quantity_type="shares",
           quantity=volume, order_type="market")
```

Volume must be a finite positive integer and is never silently rounded. A sell
cannot exceed the synchronized long position. Unsupported directions or flags
are recorded as `unsupported_action`. The original CTA price is audited but not
enforced as a limit; every such mapping records `limit_price_not_enforced`.
ATL independently applies cash and maximum-position-weight checks.

## 9. Audit Artifact

The local JSON artifact contains:

- schema version and ATL run metadata;
- ATL SDK, vn.py, and `vnpy_ctastrategy` versions;
- strategy module, class, public settings, and optional code commit;
- each observation timestamp and OHLCV bar;
- captured CTA calls and mapped ATL orders;
- ATL fills, warnings, rejections, and run status;
- categorized HOLDs, unsupported actions, rejections, and timeouts;
- summary counts and a SHA-256 digest.

API keys, full environment variables, credentials embedded in URLs, and
strategy source are excluded. Sensitive settings and error messages are
redacted and bounded.

## 10. Error Categories

| Condition | Behavior | Category |
|---|---|---|
| Strategy emits no order | Submit empty orders | `strategy_hold` |
| First bar is buffered | Submit empty orders | `warmup_hold` |
| `on_bar` raises | Submit empty orders and continue | `error_hold` |
| Unsupported CTA action | Ignore that call, keep valid calls | `unsupported_action` |
| Invalid volume or excess sale | Reject locally | `local_rejection` |
| ATL rejects an order | Preserve ATL validation | `atl_rejection` |
| Decision deadline expires | Accept ATL automatic HOLD | `timeout_hold` |
| Bar contract is invalid | Stop the local driver | `fatal_data_error` |
| Final bar cannot execute | Do not call the strategy | `terminal_bar_skipped` |

Error HOLDs are never counted as normal strategy HOLDs. Runs containing errors,
fatal data failures, or timeouts remain useful for debugging but are not marked
clean.

## 11. Verification

The test suite covers protocol serialization, optional dependency behavior,
formal vn.py object construction, lifecycle ordering, order mapping, position
reconciliation, T+1 execution, error classification, artifact validation and
redaction, CLI behavior, and a deterministic offline end-to-end loop.

The deterministic fixture must produce at least one buy and one sell, never
create a short position, execute every order after its signal bar, and produce
the same order sequence and artifact summary when replayed.

A real-data smoke test records the ATL run, result, curve points, diagnostics,
and artifact digest. Its return is evidence about the tested strategy and data,
not proof of future profitability.

## 12. Acceptance Criteria

1. A local bar-driven `CtaTemplate` strategy runs with one command.
2. ATL observations expose complete current OHLCV bars.
3. The adapter constructs formal vn.py `BarData` objects.
4. A deterministic AAPL fixture produces both buy and sell actions.
5. Every strategy signal is executed with a one-bar delay.
6. ATL persists an equity curve, trades, and standard metrics.
7. The local artifact distinguishes all expected diagnostic categories.
8. SDK users who do not install vn.py remain unaffected.
9. SDK, backend, protocol, and end-to-end tests pass.
10. Documentation states the execution differences and first-release limits.

## 13. Future Work

Later loops can evaluate multiple US equities, fuller order semantics, A-share
market rules, vn.py Gateway support, isolated server-side execution, and
vn.py-specific leaderboard diagnostics. None is part of this first release.
