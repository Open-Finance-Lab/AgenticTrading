# Python SDK QuickStart (`agentictrading`)

The `agentictrading` package gives Python users a clean, typed, function-call
interface over the Agentic Trading Lab **Agent–Environment Protocol (v1)**. It is
a remote HTTP client: it talks to the same REST endpoints documented in
[`agent-environment-protocol-v1.md`](./agent-environment-protocol-v1.md) and works
whether the backend runs on your machine or is deployed elsewhere.

The SDK is dependency-free on macOS and Linux (standard-library `urllib` only).
On Windows it also installs the `tzdata` data wheel, because `zoneinfo` has no
system IANA time-zone database there.

## The layers

```
user-defined agent logic        # your decide(observation) method
        ↓
AgentRunner                      # optional high-level run loop
        ↓
ATLClient                        # typed Python methods
        ↓
REST endpoint (/api/v1/...)      # the language-agnostic source of truth
        ↓
FastAPI backend                  # validation + backtest engine
```

- **REST API endpoints** — `/api/v1/runs`, `/api/v1/runs/{id}/steps/next`, etc.
  These are the contract; any language can use them.
- **`ATLClient` methods** — one Python method per endpoint, returning typed
  models (`Run`, `Step`, `ExecutionResult`, `RunResult`, …).
- **`AgentRunner`** — drives the full step loop for you (create → poll → decide →
  submit → repeat → result).
- **Your agent** — you implement only `decide(observation)`.

## Install

From the repo:

```bash
pip install -e packaging/agentictrading
```

## Authentication

The SDK authenticates with your **agent API key** via the `X-API-Key` header.
You do **not** need to resolve the key into a browser session.

```bash
export ATL_API_KEY="ag_xxxxxxxx"
export ATL_BASE_URL="http://127.0.0.1:8000"
```

The SDK never logs or prints the API key.

## QuickStart with `ATLClient`

```python
import os
from agentictrading import ATLClient, Decision

client = ATLClient(
    base_url=os.environ["ATL_BASE_URL"],
    api_key=os.environ["ATL_API_KEY"],
)

run = client.create_run(
    agent_version_id="agv_xxx",        # create this once, reuse it
    environment_id="us-equity-hourly-v1",
    start_date="2026-04-15",
    end_date="2026-04-16",
    symbols=["AAPL", "MSFT"],
)

while True:
    step = client.get_next_step(run.id)

    if step.status == "completed":
        break
    if step.status == "loading":
        client.wait(2)
        continue

    client.submit_decision(
        run_id=run.id,
        step_id=step.id,
        decision=Decision(orders=[], rationale="No valid signal."),
    )

result = client.get_run_result(run.id)
print(result.metrics)
```

### Typed decisions and orders

```python
from agentictrading import Decision, Order

decision = Decision(
    orders=[
        Order(symbol="AAPL", side="buy", quantity_type="shares",
              quantity=10, order_type="market"),
    ],
    confidence=0.8,
    rationale="Positive momentum signal.",
)
```

An empty order list is an explicit **HOLD**:

```python
Decision(orders=[], rationale="No valid signal.")
```

Plain dictionaries are also accepted by `submit_decision`:

```python
client.submit_decision(run.id, step.id, {"orders": [], "rationale": "Hold."})
```

### Idempotency

`submit_decision` automatically uses a deterministic idempotency key of the form
`"{run_id}:{step_id}"` when you do not pass one. Retrying the same step therefore
reuses the same key, and the backend returns the original result instead of
double-executing. Pass `idempotency_key=...` to override.

## High-level: `AgentRunner`

Implement only `decide`; the runner handles the loop, the loading waits, and the
final result.

```python
import os
from agentictrading import ATLClient, AgentRunner

class MyAgent:
    def decide(self, observation):
        # observation.market, observation.features, observation.portfolio
        return {"orders": [], "rationale": "Hold."}

    # optional hooks (detected automatically if present)
    def on_execution_result(self, result):
        ...

    def on_run_completed(self, result):
        ...

client = ATLClient(base_url=os.environ["ATL_BASE_URL"], api_key=os.environ["ATL_API_KEY"])
runner = AgentRunner(client=client, agent=MyAgent())

result = runner.run_backtest(
    agent_version_id="agv_xxx",
    environment_id="us-equity-hourly-v1",
    start_date="2026-04-15",
    end_date="2026-04-16",
    symbols=["AAPL", "MSFT"],
)
print(result.metrics)
```

The runner raises `ATLRunFailedError` if the run enters `failed` or `cancelled`.

## AgentVersion lifecycle

An `AgentVersion` is an immutable snapshot of your agent's configuration. Create
it **once** and reuse it across many runs:

```text
Create Agent  →  Create AgentVersion (once)  →  Run that AgentVersion many times
```

`create_run()` requires an existing `agent_version_id`; the SDK never creates a
new version implicitly. Create one explicitly:

```python
version = client.create_agent_version(
    agent_id="agt_xxx",
    version="0.1.0",
    model_backbones=["claude-sonnet"],
    architecture="single_agent_tool_augmented",
    decision_frequency="1h",
)
print(version.id)  # agv_xxx
```

## `ATLClient` method ↔ REST endpoint

| Python method | REST endpoint |
| --- | --- |
| `create_agent_version(agent_id, ...)` | `POST /api/v1/agents/{agent_id}/versions` |
| `create_run(agent_version_id, ...)` | `POST /api/v1/runs` |
| `get_run(run_id)` | `GET /api/v1/runs/{run_id}` |
| `get_run_status(run_id)` | `GET /api/v1/runs/{run_id}/status` |
| `get_next_step(run_id)` | `GET /api/v1/runs/{run_id}/steps/next` |
| `get_step(run_id, step_id)` | `GET /api/v1/runs/{run_id}/steps/{step_id}` |
| `submit_decision(run_id, step_id, decision)` | `POST /api/v1/runs/{run_id}/steps/{step_id}/decision` |
| `get_run_result(run_id)` | `GET /api/v1/runs/{run_id}/result` |
| `get_run_trades(run_id)` | `GET /api/v1/runs/{run_id}/trades` |
| `get_run_decisions(run_id)` | `GET /api/v1/runs/{run_id}/decisions` |
| `get_run_metrics(run_id)` | `GET /api/v1/runs/{run_id}/metrics` |
| `list_environments()` | `GET /api/v1/environments` |

## Errors

All SDK errors derive from `ATLAPIError` and preserve `status_code`, `message`,
`path`, and (when available) the backend `code`.

| Exception | When |
| --- | --- |
| `ATLAuthenticationError` | HTTP 401/403 (missing/invalid key, wrong owner) |
| `ATLValidationError` | HTTP 400/422 (invalid request) |
| `ATLConflictError` | HTTP 409 (finalized step, deadline exceeded) |
| `ATLTimeoutError` | request timed out at the transport layer |
| `ATLRunFailedError` | run entered `failed`/`cancelled` |
| `ATLAPIError` | any other non-2xx response |

```python
from agentictrading import ATLValidationError

try:
    client.submit_decision(run.id, step.id, bad_decision)
except ATLValidationError as exc:
    print(exc.status_code, exc.code, exc.message)
```

## Examples

- `dashboard/examples/sdk_hold_agent.py` — minimal HOLD-every-step run.
- `dashboard/examples/sdk_rule_based_agent.py` — reads the observation and
  submits real orders from a deterministic rule.
- `dashboard/examples/sdk_custom_agent_runner.py` — implements only `decide()`
  and lets `AgentRunner` manage the loop.
- `dashboard/examples/sdk_quickstart_selftest.py` — one command that signs in,
  registers an agent, and runs a short backtest end to end against a live
  backend.
- `dashboard/examples/tradingagents_atl_backtest.py` — bridges
  [TradingAgents](https://github.com/TauricResearch/TradingAgents): it runs the
  multi-agent analysis locally (your model and data credentials never leave the
  machine), writes a replayable decision artifact, and replays that artifact
  through ATL at T+1. Setup guide:
  [`docs/integrations/tradingagents.zh-CN.md`](../integrations/tradingagents.zh-CN.md)
  (Simplified Chinese).

All examples read credentials from environment variables; do not place real API
keys, passwords, or tokens in source code.
