# vn.py CTA Strategy Integration Implementation Plan

> Execute this plan as a sequence of loop-engineering cycles. Each task starts
> with a failing test, adds the smallest implementation, verifies the affected
> surface, and ends in a focused commit.

**Goal:** Run a local, bar-driven vn.py `CtaTemplate` strategy against ATL,
convert supported `buy` and `sell` calls into typed ATL orders with a one-bar
delay, and produce both ATL performance results and a local audit artifact.

**Architecture:** ATL populates the existing `Observation.market.bars` field.
The local `VnpyCtaAdapter` buffers one bar, calls the strategy through a minimal
`AtlCtaEngine`, maps captured orders, and delegates execution to the existing
`ATLClient` and `AgentRunner`. ATL remains the only source of truth for market
data, positions, fills, and equity.

**Stack:** Python 3.10+, vn.py 4.4.0, `vnpy_ctastrategy` 1.3.0, the ATL
`agentictrading` SDK, and pytest.

**Design:**
`docs/superpowers/specs/2026-07-26-vnpy-cta-atl-integration-design.md`

## Global Constraints

- Branch: `feat/vnpy-cta-integration`.
- First release: `AAPL`, hourly bars, whole shares, long-only market orders.
- Strategy source runs only on the user's trusted local machine.
- ATL owns market data, positions, execution, and equity.
- Every CTA signal is delayed by one bar to prevent look-ahead bias.
- The CTA price is audited but is not enforced as an ATL limit price.
- Unsupported operations are explicit diagnostics, never silent HOLDs.
- Normal HOLDs, error HOLDs, local rejections, ATL rejections, and timeouts are
  counted separately.
- vn.py dependencies remain optional and are imported lazily.
- Core SDK tests run without vn.py; formal-object compatibility tests may skip
  when the optional packages are absent.
- Backend protocol tests never depend on vn.py.
- Automated tests do not call live ATL, Alpaca, or broker services.
- Existing simulated vn.py market-data behavior remains unchanged.

## Task 1: Populate Current OHLCV in ATL Observations

**Files:**

- Modify `dashboard/backend/domain/backtesting/external_run_service.py`.
- Modify `dashboard/backend/domain/runs/service.py`.
- Modify the directly affected backend tests.

### Step 1: Add failing protocol tests

Cover these cases:

- `protocol_bars()` returns current `open`, `high`, `low`, `close`, `volume`,
  and a timezone-aware `timestamp` for each allowed symbol with data.
- Values are plain JSON-serializable floats.
- A single-symbol run exposes only `AAPL`.
- Existing market features, portfolio, constraints, and step timestamps remain
  unchanged.
- Missing bars are omitted instead of being synthesized or forward-filled from
  the future.

```bash
python -m pytest \
  dashboard/backend/tests/domain/backtesting/test_external_run_service_move.py \
  dashboard/backend/tests/test_protocol_api.py -q \
  -k 'protocol_bars or observation_bars'
```

Expected initial result: failure because observations still contain empty bars.

### Step 2: Implement the protocol boundary

Reuse the session's public market-data adapter and return an ordinary mapping:

```python
{
    "AAPL": {
        "timestamp": "2026-04-15T10:00:00-04:00",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1_000_000.0,
    }
}
```

Do not import vn.py in the backend.

### Step 3: Verify

```bash
python -m pytest \
  dashboard/backend/tests/domain/backtesting/test_external_run_service_move.py \
  dashboard/backend/tests/test_protocol_api.py -q
git diff --check
```

Commit: `feat(protocol): expose current OHLCV bars`

## Task 2: Define Orders and the Audit Contract

**Files:**

- Add `packaging/agentictrading/src/agentictrading/integrations/_vnpy_cta_core.py`.
- Add `packaging/agentictrading/src/agentictrading/integrations/vnpy_cta.py`.
- Extend `packaging/agentictrading/tests/test_vnpy_cta_integration.py`.

### Step 1: Add failing core tests

Cover:

- `LONG + OPEN` maps to buy and `SHORT + CLOSE` maps to sell.
- Volume must be a finite positive integer and is not silently rounded.
- A sale larger than the synchronized position is a local rejection.
- Short, cover, stop, lock, and net requests are unsupported actions.
- Original CTA prices are audited but do not become ATL limit prices.
- Diagnostic categories cannot be conflated.
- Artifacts support deterministic JSON round trips and SHA-256 validation.
- Loading rejects malformed JSON, unknown schemas, duplicate steps, and invalid
  statuses.
- Credentials and sensitive settings are redacted.
- Importing the public integration does not import optional vn.py packages.

### Step 2: Implement a pure-Python core

Add explicit dataclasses, mapping logic, serialization, validation, redaction,
and hashing. This task does not import vn.py, create a strategy, or send HTTP.

### Step 3: Verify

```bash
python -m pytest packaging/agentictrading/tests/test_vnpy_cta_integration.py -q
python -m pytest packaging/agentictrading/tests -q
git diff --check
```

Commit: `feat(sdk): add vn.py CTA audit contract`

## Task 3: Bridge the vn.py Runtime

**Files:**

- Add `_vnpy_cta_runtime.py`.
- Update the public vn.py integration facade.
- Add the `vnpy` optional dependency group to the SDK package.
- Extend integration tests.

### Step 1: Add failing runtime tests

Use injectable fake bindings to verify:

- Missing or incompatible optional dependencies fail only when the runtime is
  created and provide a clear installation command.
- Lifecycle order is `on_init -> on_start -> on_bar... -> on_stop`.
- Strategy `inited` and `trading` flags change at the correct boundaries.
- ATL bars become formal `BarData` with gateway `ATL`, exchange `SMART`, interval
  `HOUR`, and timezone-aware datetimes.
- `load_bar` returns no history and records `history_preload_unavailable`.
- Supported orders receive stable local IDs.
- ATL fills and rejections generate final `OrderData`; fills also generate
  `TradeData`.
- Each ATL portfolio observation overwrites stale `strategy.pos` state.

Add optional formal-object tests with `pytest.importorskip` while keeping the
fake-runtime tests mandatory.

### Step 2: Implement lazy bindings and the minimal engine

Only `_vnpy_cta_runtime.py` may import vn.py at runtime. Pin the optional extra:

```toml
[project.optional-dependencies]
vnpy = ["vnpy==4.4.0", "vnpy_ctastrategy==1.3.0"]
```

Do not add these packages to mandatory SDK dependencies.

### Step 3: Verify

```bash
python -m pytest packaging/agentictrading/tests/test_vnpy_cta_integration.py -q
python -m pytest packaging/agentictrading/tests -q
git diff --check
```

Commit: `feat(sdk): bridge vn.py CTA runtime`

## Task 4: Implement the One-Bar Adapter State Machine

**Files:**

- Add `_vnpy_cta_adapter.py`.
- Update the public facade and integration tests.

### Step 1: Add failing state-machine tests

Cover:

- The first observation buffers a bar and produces `warmup_hold`.
- Step N synchronizes ATL state and processes only bar N-1.
- The current bar cannot influence the current execution step.
- Decisions contain only typed whole-share market orders.
- Duplicate bars are rejected.
- Missing or malformed OHLCV, naive timestamps, and symbol mismatches raise
  `VnpyCtaDataError`.
- Strategy inactivity is `strategy_hold`; exceptions are `error_hold`.
- Supported and unsupported calls in one step are recorded independently.
- Execution hooks preserve fills and rejections and drive runtime callbacks.
- Finalization calls `on_stop` and records `terminal_bar_skipped`.
- Every signal timestamp is earlier than its execution timestamp.

### Step 2: Implement the adapter

Keep the adapter focused on ordering, runtime calls, and audit state. Reuse
`AgentRunner` for polling, deadlines, transport errors, and decision submission.

### Step 3: Verify

```bash
python -m pytest packaging/agentictrading/tests/test_vnpy_cta_integration.py -q
python -m pytest packaging/agentictrading/tests -q
```

Commit: `feat(sdk): adapt vn.py CTA signals at T+1`

## Task 5: Compose the ATL Runner

**Files:**

- Add `_vnpy_cta_runner.py`.
- Update adapter, facade, and integration tests.

### Step 1: Add failing runner tests

Verify that the runner:

- creates `us-equity-hourly-v1` runs for `AAPL`;
- stores integration and version metadata but no sensitive settings;
- reuses bounded `AgentRunner` loading and execution polling;
- submits each awaiting step exactly once;
- classifies ATL deadline HOLDs without resubmitting;
- retrieves `RunResult`, stops the runtime, and finalizes the artifact;
- propagates API failures with the run ID instead of converting them to strategy
  HOLDs;
- reports all diagnostic and execution counts accurately;
- marks runs with error HOLDs, fatal errors, or timeouts as not clean.

### Step 2: Implement by composition

Compose `AgentRunner(client, adapter)`. Do not duplicate authentication,
idempotency, polling, or error handling.

### Step 3: Verify

```bash
python -m pytest packaging/agentictrading/tests -q
python -m pytest dashboard/backend/tests/test_protocol_api.py -q
git diff --check
```

Commit: `feat(sdk): run vn.py CTA strategies on ATL`

## Task 6: Add a Deterministic Example and Documentation

**Files:**

- Add `dashboard/examples/vnpy_cta_double_ma_strategy.py`.
- Add `dashboard/examples/vnpy_cta_atl_backtest.py`.
- Add `docs/integrations/vnpy-cta.md`.
- Update `docs/source/lab/external_agents.rst`.
- Extend integration tests.

### Step 1: Add failing CLI and example tests

Cover:

- `--strategy module:Class`, `--settings-file`, `--symbol`, `--start`, `--end`,
  `--initial-cash`, and `--output`;
- rejection of unsupported symbols and invalid date ranges;
- clear errors for missing ATL configuration;
- `--help` without vn.py imports, credentials, or network access;
- sensitive settings excluded from config, logs, and artifacts;
- a final summary with URLs, metrics, diagnostics, artifact path, and digest;
- a formal `CtaTemplate` example using only bars, buy, sell, and streaming warmup.

### Step 2: Implement the local CLI

Import and initialize the strategy before creating an ATL run. Write artifacts
under `~/.agentictrading/vnpy-cta/runs/` by default and use temporary directories
in tests.

### Step 3: Document the user contract

Document responsibilities, installation, configuration, custom strategy import,
the complete data and order path, the one-bar delay, ATL risk controls, market
order semantics, unsupported features, result inspection, artifact diagnostics,
and the fact that backtest profit does not prove future performance.

### Step 4: Verify

```bash
python dashboard/examples/vnpy_cta_atl_backtest.py --help
python -m pytest packaging/agentictrading/tests/test_vnpy_cta_integration.py -q
git diff --check
```

Commit: `docs: add vn.py CTA ATL quickstart`

## Task 7: Complete Regression and Smoke Verification

### Step 1: Run all affected tests

```bash
python -m pip install -e 'packaging/agentictrading[vnpy]'
python -m pytest packaging/agentictrading/tests -q
python -m pytest dashboard/backend/tests -q
git diff --check
```

### Step 2: Run a deterministic offline loop

Use fixed OHLCV and a fake ATL client. Require at least one buy and one sell,
strictly later execution timestamps, no short position, no error or timeout
HOLDs, no unexpected rejections, and an identical order sequence and artifact
summary on replay.

### Step 3: Audit the change set

Confirm that no `.env`, API key, absolute user path, generated artifact,
database, private strategy source, or browser output is committed.

### Step 4: Run an authorized real-data smoke

After explicit authorization, run the included strategy against ATL-provided
AAPL data without connecting a broker. Record the run and result IDs, data
range, curve points, execution and diagnostic counts, metrics, artifact path,
and SHA-256 digest. Do not alter parameters to fabricate trades.

### Step 5: Report the result

Report the branch, commits, automated tests, offline evidence, real-data
observations, and remaining scope. A real-data return is an observation, not a
profitability claim.
