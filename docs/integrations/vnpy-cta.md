# Run a Local vn.py CTA Strategy on ATL

This integration runs an existing vn.py `CtaTemplate` strategy against the
standard Agentic Trading Lab (ATL) US equity backtest environment. ATL produces
the equity curve, trades, metrics, and execution results, while the integration
writes a local audit artifact for vn.py-specific diagnostics.

## Responsibilities

- **The vn.py CTA strategy** contains the trading rules and calls `buy()` or
  `sell()` as bars arrive.
- **The local adapter** converts ATL OHLCV data into vn.py `BarData` and maps CTA
  orders into ATL typed orders.
- **ATL** provides historical market data, validates orders, simulates fills,
  maintains cash and positions, and calculates performance metrics.

The adapter works like an interpreter between a driver and a racetrack: the CTA
strategy decides when to accelerate or brake, while ATL enforces the track rules
and records the result.

The first release follows this path:

```text
ATL AAPL hourly OHLCV
  -> vn.py BarData
  -> local CtaTemplate.on_bar()
  -> buy()/sell()
  -> ATL typed Order
  -> ATL market-order simulation
  -> equity curve, trades, metrics, and local artifact
```

Strategy source code always runs on the user's machine. ATL does not receive,
store, or execute the Python strategy source.

## Install

Create an isolated Python 3.10 or newer environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e 'packaging/agentictrading[vnpy]'
```

The optional dependency versions are pinned to:

- `vnpy==4.4.0`
- `vnpy_ctastrategy==1.3.0`

## Configure ATL

Create an ATL Agent and AgentVersion, then set these environment variables:

```bash
export ATL_BASE_URL="https://agentictrading.onrender.com"
export ATL_API_KEY="ag_xxxxxxxx"
export ATL_AGENT_VERSION_ID="agv_xxxxxxxx"
```

This flow does not require an Alpaca key, Interactive Brokers account, TWS,
IB Gateway, or any live brokerage account. The configured ATL environment
provides the backtest data. The API key only identifies the ATL Agent.

Never place a key in strategy source, a settings file, a command-line argument,
or the Git repository.

## Run the Included Double-Moving-Average Strategy

Inspect the command without importing vn.py, reading credentials, or making a
network request:

```bash
python dashboard/examples/vnpy_cta_atl_backtest.py --help
```

Run the included example:

```bash
python dashboard/examples/vnpy_cta_atl_backtest.py \
  --start 2026-04-01 \
  --end 2026-04-23 \
  --symbol AAPL
```

The strategy warms up from the incoming bar stream. It buys when the fast
moving average crosses above the slow moving average and sells an existing
position on the opposite crossover. This deterministic example verifies the
integration; it is not a recommended profitable strategy.

Public strategy parameters can be supplied as JSON:

```json
{
  "fast_window": 5,
  "slow_window": 20,
  "fixed_size": 1
}
```

```bash
python dashboard/examples/vnpy_cta_atl_backtest.py \
  --settings-file ./vnpy-settings.json \
  --start 2026-04-01 \
  --end 2026-04-23
```

Use `--output` to select the local audit path. Without it, artifacts are written
under `~/.agentictrading/vnpy-cta/runs/`. The CLI accepts `--initial-cash` for
forward compatibility, but the current ATL environment fixes initial equity at
1000, so the option should normally be omitted.

## Run a Custom Strategy

The CLI imports a strategy using `module:Class` syntax:

```bash
python dashboard/examples/vnpy_cta_atl_backtest.py \
  --strategy my_strategy:DoubleMaStrategy \
  --settings-file ./my-settings.json \
  --symbol AAPL \
  --start 2026-04-01 \
  --end 2026-04-23 \
  --output ./artifacts/my-vnpy-run.json
```

`my_strategy.py` must be importable from the current Python environment, and the
class must inherit from `vnpy_ctastrategy.CtaTemplate`. Import only trusted local
strategies because importing a module grants the same permissions as running
that Python file directly.

Settings are passed to the local strategy. Fields with sensitive names such as
`api_key`, `token`, `secret`, and `password` are replaced with `<redacted>` in
the ATL run configuration and the local artifact.

## Execution Rules

### T+1 one-bar delay

A strategy only knows a bar's closing price after that bar has completed. Filling
an order at the same close after observing it would introduce look-ahead bias.
The adapter therefore delays strategy signals by one bar:

1. The first bar is buffered and produces a normal HOLD.
2. Step N processes the signal from bar N-1.
3. ATL simulates the resulting order at step N.
4. The final buffered bar has no later execution step and is recorded as
   `terminal_bar_skipped`.

### Orders and risk controls

- The first release supports one US equity, `AAPL`, with hourly bars.
- It is long-only: `buy` opens a long position, and `sell` can only reduce an
  existing long position.
- Quantities must be positive whole shares.
- `short`, `cover`, stop, lock, net, and cancel requests are not executed and
  are recorded in diagnostics.
- A CTA price is preserved in the local artifact but is not enforced as a limit
  price. The first release maps valid orders to ATL market orders.
- ATL independently checks cash and the 25% maximum position weight. ATL's
  response is authoritative for rejections.
- ATL fill prices can differ from the vn.py limit-order backtesting engine, so
  the two engines are not expected to produce identical results.

## Current Limitations

- `load_bar` does not query a vn.py database. It records
  `history_preload_unavailable`, and the strategy must warm up from consecutive
  `on_bar` calls.
- Full `TargetPosTemplate` behavior, multiple symbols, TickData, A-shares,
  futures, short selling, and margin trading are not supported.
- The integration does not connect a vn.py Gateway, Interactive Brokers, TWS,
  IB Gateway, Alpaca account, or any live broker.
- This is a backtest integration, not live trading.
- It does not place live orders or model persistent limit orders, stop orders,
  partial fills, or order cancellation.

## Read the Results

The command prints:

- the ATL `run_id` and result URL;
- total return, Sharpe ratio, maximum drawdown, and trade count;
- counts for normal HOLDs, error HOLDs, unsupported actions, local rejections,
  ATL rejections, and timeouts;
- the local artifact path and SHA-256 digest.

The ATL URL shows the persisted equity curve, trades, and common metrics. The
local JSON artifact records each input bar, signal bar, captured CTA request,
mapped order, ATL fill or rejection, and error classification. The ATL API key
and strategy source are never intentionally recorded, and settings fields with
sensitive names (`api_key`, `token`, `secret`, `password`, ...) are redacted
recursively; free-form text such as exception messages is best-effort
redacted against common credential shapes (`key=value`, `key: value`,
`Bearer <token>`, URL-embedded credentials), not guaranteed against every
possible format.

`clean: true` means the run had no error HOLDs, rejections, timeouts, or fatal
data errors. It does not mean the strategy is profitable. A profitable backtest
also does not prove future performance.

## Diagnostic Reference

| Output | Meaning |
|---|---|
| `warmup_hold` | The first bar was buffered as expected. |
| `strategy_hold` | The strategy ran normally and emitted no order. |
| `error_hold` | `on_bar` raised an exception and the step became HOLD. |
| `unsupported_actions` | The strategy requested an unsupported CTA action. |
| `local_rejections` | Quantity, direction, or position checks failed locally. |
| `atl_rejections` | ATL rejected an order for cash, risk, or protocol reasons. |
| `timeout_hold` | The ATL decision deadline expired and the step became HOLD. |
| `fatal_data_error` | OHLCV or timestamp data violated the adapter contract. |
| `run_error` | An ATL API, network, or other run-level failure occurred. |
