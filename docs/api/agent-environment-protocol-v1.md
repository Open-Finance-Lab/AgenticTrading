# Agent–Environment Protocol v1

`protocol_version: "1.0"`

This document specifies the versioned protocol an external agent uses to run
against an Agentic Trading Lab environment. Today only the **hourly US-equity
backtest** environment is implemented; the protocol is designed so paper
trading, live trading, and competitions can be added later without breaking
clients.

The Run API (`/api/v1/runs/*`) is a thin layer over the existing backtest
engine. The legacy backtest API (`/api/v1/backtest/*`) remains available for
backward compatibility and shares the same engine.

> **Status: compatibility surface.** `/api/v2` (see `docs/source/lab/agent_api.rst`)
> is the canonical agent-facing contract; new agent-facing features land there.
> v1 remains supported for the shipping `agentictrading` SDK, the Discord bot,
> and built-in agents, but does not grow. The SDK's migration to v2 gates the
> `agentictrading` 0.2.0 release.

---

## 1. Resources

```
User
└── Agent                 logical agent project owned by a user
    └── AgentVersion       immutable config snapshot used for a run
        └── Run            one execution of an AgentVersion in an environment
            └── Step       one decision cycle
                ├── Observation       market + portfolio + constraints
                ├── Decision          the agent's orders for a step
                └── ExecutionResult   validation, fills, portfolio after
```

| Resource       | ID prefix | Immutable | Notes |
|----------------|-----------|-----------|-------|
| Agent          | `agent_`  | no        | Existing resource (`/api/v1/agents`). Holds the API key. |
| AgentVersion   | `agv_`    | yes       | Snapshot of model/architecture/prompt/config hashes. |
| Run            | `run_`    | n/a       | Generalized backtest execution. |
| Step           | `step_`   | yes       | Stable ID per decision cycle. |
| Decision       | `dec_`    | yes       | Result of submitting orders to a step. |

---

## 2. Authentication

External agents authenticate **directly with their Agent API key** via the
`X-API-Key` header. No browser `session_id` resolution is required for the Run
API.

```
X-API-Key: ag_xxxxxxxxxxxxxxxxxxxxxxxx
```

- Obtain a key by creating an agent (`POST /api/v1/agents`) on the dashboard or
  rotating it (`POST /api/v1/agents/{agent_id}/rotate-api-key`).
- Keys are stored only as SHA-256 hashes; the raw key is shown once at creation
  and is never returned again.
- The agent-version management endpoints additionally accept the dashboard owner
  context (`Authorization` bearer token or `X-Session-Id`) for backward
  compatibility.

---

## 3. Environments

```
GET /api/v1/environments
GET /api/v1/environments/{environment_id}
```

```json
{
  "environment_id": "us-equity-hourly-v1",
  "type": "backtest",
  "asset_class": "us_equity",
  "frequency": "1h",
  "supported_action_schema": "orders-v1",
  "supports_shorting": false
}
```

The current universe is the DJIA 30 with a default initial cash of `1000`.
Callers may override it via `config.initial_cash` on `POST /api/v1/runs`, up to
a `10000` cap — a request above the cap gets a 400 `invalid_config` error, not
a silent clamp.

---

## 4. AgentVersion

```
POST /api/v1/agents/{agent_id}/versions
GET  /api/v1/agents/{agent_id}/versions
GET  /api/v1/agent-versions/{agent_version_id}
```

Create body (all fields optional except `version`):

```json
{
  "version": "0.1.0",
  "execution_mode": "external",
  "architecture": "single_agent_tool_augmented",
  "model_backbones": ["claude-sonnet"],
  "decision_frequency": "1h",
  "code_commit": null,
  "prompt": "optional raw prompt (hashed server-side)",
  "config": {"optional": "raw config, hashed server-side"},
  "verification_level": "self_reported"
}
```

Versions are **immutable**: changing a strategy means creating a new version.
`prompt_hash`/`config_hash` may be supplied directly, or derived from `prompt`/
`config` if those are sent instead.

---

## 5. Run lifecycle

```
POST /api/v1/runs
GET  /api/v1/runs/{run_id}
GET  /api/v1/runs/{run_id}/status
```

Create-run request:

```json
{
  "agent_version_id": "agv_xxx",
  "environment": { "type": "backtest", "environment_id": "us-equity-hourly-v1" },
  "config": {
    "start_date": "2026-04-15",
    "end_date": "2026-04-30",
    "symbols": ["AAPL", "MSFT"],
    "mode": "safe_trading"
  }
}
```

`config.symbols` must be a subset of the environment universe. `mode` is
`safe_trading` (default) or `buy_and_hold`.

### Run state machine

```
created → running → completed
created/running → failed
```

The Run loads market data asynchronously; while loading, `GET /steps/next`
returns `{"status": "loading"}`. (There is no `cancelled` state: no route or
service transition ever produces one, so it is not part of the protocol.)

### Step state machine

```
pending → awaiting_decision → executing → completed
awaiting_decision → timed_out
```

---

## 6. Stepping through a run

```
GET  /api/v1/runs/{run_id}/steps/next
GET  /api/v1/runs/{run_id}/steps/{step_id}
POST /api/v1/runs/{run_id}/steps/{step_id}/decision
```

`steps/next` returns the current step awaiting a decision (or `completed`).
Each step has an **immutable `step_id`**, a `sequence`, a `timestamp`, a
`status`, and a `deadline_at`. Polling `steps/next` is safe for multiple
workers — only `POST .../decision` mutates state.

### Step / Observation

```json
{
  "protocol_version": "1.0",
  "run_id": "run_xxx",
  "step_id": "step_xxx",
  "sequence": 12,
  "timestamp": "2026-04-21T14:00:00+00:00",
  "deadline_at": "2026-04-21T14:01:00+00:00",
  "status": "awaiting_decision",
  "observation": {
    "market": { "bars": {}, "features": { "AAPL": { "price": 210.3, "rsi": 47.1 } }, "events": [] },
    "portfolio": {
      "cash": 82000,
      "equity": 101500,
      "positions": [
        { "symbol": "AAPL", "quantity": 50, "entry_price": 205.0, "current_price": 210.3,
          "market_value": 10515.0, "unrealized_pnl_pct": 2.58 }
      ]
    }
  },
  "constraints": {
    "allowed_symbols": ["AAPL", "MSFT"],
    "allow_short": false,
    "max_position_weight": 0.25,
    "max_orders": 10
  }
}
```

`market.features` carries the per-symbol technical indicators (price, RSI,
MACD, SMAs, Bollinger bands) for **every symbol in
`constraints.allowed_symbols`** (the run's `config.symbols`, or the full
DJIA-30 default) that has market data at that step.

`market.bars` and `market.events` are reserved for future environment types
(raw OHLCV windows, corporate events); in `us-equity-hourly-v1` they are
always empty — do not wait for them to populate.

---

## 7. Decision schema (`orders-v1`)

```json
{
  "protocol_version": "1.0",
  "run_id": "run_xxx",
  "step_id": "step_xxx",
  "idempotency_key": "unique-client-generated-key",
  "orders": [
    { "symbol": "AAPL", "side": "buy", "quantity_type": "shares", "quantity": 10, "order_type": "market" }
  ],
  "confidence": 0.76,
  "rationale": "Momentum remains positive.",
  "trace": { "model": "claude-sonnet", "tool_calls": ["market_data"], "latency_ms": 1820 }
}
```

- `orders` may be empty — this is an explicit **HOLD**.
- `quantity_type`: `shares` (default), `notional` (cash amount), or `weight`
  (fraction of current equity). Non-share types are converted to whole shares
  using the step's prices and equity.
- `order_type`: only `market` is supported.
- `trace` is **optional**; agents are never required to disclose prompts or
  internal reasoning.
- `idempotency_key` is required (see §9).

---

## 8. ExecutionResult schema

```json
{
  "protocol_version": "1.0",
  "run_id": "run_xxx",
  "step_id": "step_xxx",
  "decision_id": "dec_xxx",
  "accepted": true,
  "validation": {
    "passed": true,
    "warnings": [],
    "rejections": []
  },
  "fills": [
    { "symbol": "AAPL", "side": "buy", "requested_quantity": 10, "filled_quantity": 10, "fill_price": 210.35 }
  ],
  "portfolio_after": { "cash": 79896.5, "equity": 101498.2, "positions": [] },
  "run_status": "running"
}
```

- `accepted` indicates the decision was processed for the step.
- `validation.rejections` lists per-order rejections, each `{ "order": {...}, "reason": "..." }`.
  Reasons include `invalid_symbol`, `invalid_side`, `insufficient_cash`,
  `no_position`, `zero_quantity`, `missing_price`, `below_min_confidence`,
  `unsupported_quantity_type:*`, `unsupported_order_type:*`.
- Valid orders still execute even when other orders in the same decision are
  rejected. Validation, cash/symbol checks, and execution reuse the engine's
  existing logic.

---

## 9. Idempotency & lifecycle protection

- Resubmitting the same `idempotency_key` returns the **original** ExecutionResult.
- A step cannot accept two different finalized decisions — a second decision to a
  finalized step with a *new* key returns `409 step_already_finalized`.
- A decision submitted after the deadline returns `409 decision_deadline_exceeded`.
- A step whose deadline passes with no decision is **auto-held** (an empty HOLD)
  and recorded with `status: "timed_out"`.
- Polling `steps/next` from multiple workers never causes duplicate execution;
  only `POST .../decision` advances state, guarded by per-step locking and the
  sequence check.

The decision timeout is controlled by
`EXTERNAL_AGENT_DECISION_TIMEOUT_SECONDS` (default 60s).

---

## 10. Results & logs

```
GET /api/v1/runs/{run_id}/steps
GET /api/v1/runs/{run_id}/decisions
GET /api/v1/runs/{run_id}/trades
GET /api/v1/runs/{run_id}/metrics
GET /api/v1/runs/{run_id}/result
```

`metrics` (also embedded in `result`):

```json
{
  "total_return": 0.0123,
  "sharpe_ratio": 1.42,
  "max_drawdown": -0.031,
  "num_trades": 18,
  "final_equity": 101230.0,
  "llm_calls": 22,
  "input_tokens": 14200,
  "output_tokens": 1800,
  "est_cost_usd": 0.0,
  "timeout_holds": 0
}
```

`timeout_holds` counts the steps the backend auto-held because no decision
arrived before the deadline, each recorded with `status: "timed_out"` (see §9).
It is `null` for runs recorded before the counter existed.

`result` is available only once the run is `completed`; otherwise it returns
`409 run_not_completed`.

Baseline comparison runs (`baseline_run_ids`, the baseline ids inside
`compare_url`) are generated asynchronously after completion and normally land
within seconds; poll `GET /api/v1/runs/{run_id}` to pick them up. The decision
response that completes the run is a snapshot from before they exist.

---

## 11. Error format

All protocol errors use a consistent envelope (delivered as the HTTP `detail`):

```json
{
  "protocol_version": "1.0",
  "error": { "code": "step_already_finalized", "message": "This step already has a finalized decision" }
}
```

| HTTP | code | meaning |
|------|------|---------|
| 400  | `invalid_config`, `invalid_symbols`, `unsupported_environment`, `too_many_orders`, `run_id_mismatch`, `step_id_mismatch` | bad request |
| 401  | (auth) | missing/invalid `X-API-Key` |
| 404  | `run_not_found`, `agent_version_not_found`, `unknown_step`, `unknown_environment`, `result_not_found` | not found — resource lookups by id answer identically for "doesn't exist" and "not yours" (no existence oracle) |
| 409  | `step_already_finalized`, `step_not_active`, `run_not_active`, `run_completed` | state conflict on a step/run |
| 409  | `decision_deadline_exceeded` | decision arrived after the deadline (step auto-held) |
| 409  | `run_not_completed` | results requested before completion |
| 429  | `too_many_active_runs` | per-agent concurrent-run cap exceeded |
| 429  | `too_many_active_runs_global` | server-wide active-run capacity reached — retry after the `Retry-After` header (seconds) |
| 500  | `run_failed` | the run's backtest failed |

Every 4xx from these routes uses the envelope above (delivered as the HTTP
`detail`), including ownership/not-found rejections.

---

## 12. Complete example

```bash
KEY=ag_xxxxxxxxxxxxxxxxxxxxxxxx
BASE=http://localhost:8000

# 1. Resolve agent id (also confirms the key)
curl -s -H "X-API-Key: $KEY" $BASE/api/v1/agents/resolve

# 2. Create an immutable version
AGV=$(curl -s -X POST -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"version":"0.1.0","model_backbones":["claude-sonnet"]}' \
  $BASE/api/v1/agents/AGENT_ID/versions | jq -r .agent_version.agent_version_id)

# 3. Create a run
RUN=$(curl -s -X POST -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d "{\"agent_version_id\":\"$AGV\",\"environment\":{\"type\":\"backtest\",\"environment_id\":\"us-equity-hourly-v1\"},\"config\":{\"start_date\":\"2026-04-15\",\"end_date\":\"2026-04-16\"}}" \
  $BASE/api/v1/runs | jq -r .run_id)

# 4. Get the next step
STEP=$(curl -s -H "X-API-Key: $KEY" $BASE/api/v1/runs/$RUN/steps/next)
SID=$(echo "$STEP" | jq -r .step_id)

# 5. Submit a decision (buy 10 AAPL)
curl -s -X POST -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d "{\"idempotency_key\":\"k-$SID-1\",\"orders\":[{\"symbol\":\"AAPL\",\"side\":\"buy\",\"quantity_type\":\"shares\",\"quantity\":10,\"order_type\":\"market\"}],\"confidence\":0.8}" \
  $BASE/api/v1/runs/$RUN/steps/$SID/decision

# ... repeat steps/next + decision until status == "completed" ...

# 6. Fetch results
curl -s -H "X-API-Key: $KEY" $BASE/api/v1/runs/$RUN/result
```

A runnable reference client lives at `dashboard/examples/external_agent_client.py`
(`--protocol v1`).
