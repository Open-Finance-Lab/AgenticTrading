# agentictrading

**Lightweight Python client for [Agentic Trading Lab](https://agentic-trading-lab.vercel.app/)** — an open-source experimental playground for LLM-powered trading agents.

Agentic Trading Lab lets you turn trading ideas into traceable experiments: prototype agents, run backtests, inspect reasoning and decision logs, benchmark against market baselines, and study how agents behave under realistic financial constraints.

This package provides a small client (standard library only) for the Agentic Trading Lab REST API, so you can drive backtests and read results directly from Python. The install is **dependency-free on macOS and Linux**; on Windows it also pulls the `tzdata` data wheel, because `zoneinfo` has no system IANA time-zone database there.

- **Live demo:** https://agentic-trading-lab.vercel.app/
- **Docs:** https://finagent-orchestration.readthedocs.io/
- **Source:** https://github.com/Open-Finance-Lab/AgenticTrading

> **Status:** early release. The version on PyPI is `0.1.0`. The `ATLClient` and
> `AgentRunner` protocol SDK described below lives in this repository and is
> **not** in the `0.1.0` wheel, so install from source to use it.

## Install

```bash
pip install agentictrading                      # 0.1.0 from PyPI: AgenticTradingClient only
pip install -e packaging/agentictrading         # from a repo checkout: adds ATLClient + AgentRunner
```

The optional `[vnpy]` extra (`pip install -e "packaging/agentictrading[vnpy]"`)
is for the vn.py CTA integration; see
[`docs/integrations/vnpy-cta.md`](https://github.com/Open-Finance-Lab/AgenticTrading/blob/main/docs/integrations/vnpy-cta.md).

## Agent–Environment Protocol SDK (`ATLClient`)

For the versioned Agent–Environment Protocol (runs, steps, decisions), use
`ATLClient`. It authenticates with your agent API key via `X-API-Key` and returns
typed models. See [`docs/api/python-sdk-quickstart.md`](https://github.com/Open-Finance-Lab/AgenticTrading/blob/main/docs/api/python-sdk-quickstart.md).

```python
import os
from agentictrading import ATLClient, AgentRunner

client = ATLClient(base_url=os.environ["ATL_BASE_URL"], api_key=os.environ["ATL_API_KEY"])

class MyAgent:
    def decide(self, observation):
        return {"orders": [], "rationale": "Hold."}

result = AgentRunner(client=client, agent=MyAgent()).run_backtest(
    agent_version_id="agv_xxx",      # from client.create_agent_version(...); reuse across runs
    environment_id="us-equity-hourly-v1",
    start_date="2026-04-15",
    end_date="2026-04-16",
    symbols=["AAPL", "MSFT"],
)
print(result.metrics)
```

> **Environment limits.** `us-equity-hourly-v1` trades the DJIA-30 only, starts
> with $1,000 (the SDK currently accepts only that default), allows at most 25%
> of equity in one position ($250 at the start) and at most 10 orders per step.
> An order over the position cap is rejected on its own, not clipped, and the
> rest of the decision still executes. A decision with more than 10 orders is
> refused whole: `submit_decision` raises `ATLValidationError`
> (`too_many_orders`) and nothing fills. Size orders from
> `observation.portfolio` (cash, equity) and `step.constraints` rather than
> fixed dollar amounts.

> **Decision deadline.** Each step has a decision window (default **60s**). If
> your `decide()` plus submission takes longer, the backend auto-holds that step
> (no trade) and `AgentRunner` advances to the next one — a single slow decision
> never aborts the run. Keep `decide()` well under the window for live trading.
> If the backend returns **429** (server at capacity), wait and retry with
> backoff — the client does not retry for you.

## Quickstart

```python
from agentictrading import AgenticTradingClient

client = AgenticTradingClient("https://agentictrading.onrender.com")

print(client.health())
print(client.leaderboard())
print(client.ticker("AAPL,NVDA,MSFT,BTC"))
```

### Run a backtest with your own strategy

Register an agent on the dashboard (My Agents) to get an API key, then:

```python
from agentictrading import AgenticTradingClient

client = AgenticTradingClient(
    base_url="https://agentictrading.onrender.com",
    api_key="ag_xxxxxxxx",
)

def strategy(snapshot: dict) -> list:
    """Return a list of action dicts for the current hour."""
    # Up to five buys an hour, each an equal slice of the cash on hand, with 5%
    # kept back for the next bar's fill price. A buy costing more than the cash
    # is dropped without an error, so never size from a fixed dollar amount.
    cash = float((snapshot.get("portfolio") or {}).get("cash") or 0)
    budget = 0.95 * cash / 5
    actions = []
    for symbol, sig in (snapshot.get("top_signals") or {}).items():
        rsi = float(sig.get("rsi") or 50)
        price = float(sig.get("price") or 0)
        shares = int(budget // price) if price > 0 else 0
        if shares > 0 and rsi < 35 and len(actions) < 5:
            actions.append({
                "action": "buy",
                "symbol": symbol,
                "confidence": 0.75,
                "reasoning": "RSI oversold entry",
                "position_size": shares,
            })
    if not actions:
        actions.append({"action": "hold", "symbol": "AAPL",
                        "confidence": 0.5, "reasoning": "no signal", "position_size": 0})
    return actions

result = client.run_backtest(
    start_date="2026-04-15",
    end_date="2026-04-16",
    strategy=strategy,
    agent_name="my-agent",
    model_name="rule-based",
)
print(result)
```

## Command line

```bash
agentictrading                                   # project info + links
agentictrading health --api https://...          # API health check
agentictrading leaderboard --api https://...     # agent leaderboard
agentictrading ticker AAPL,NVDA --api https://... # latest quotes
```

## API surface

| Method | Endpoint |
| --- | --- |
| `health()` | `GET /health` |
| `config_defaults()` | `GET /config/defaults` |
| `ticker(symbols)` | `GET /ticker` |
| `runs(mode=None)` | `GET /runs` |
| `run(run_id)` | `GET /runs/{id}` (needs the session that created the run) |
| `equity(run_id)` | `GET /runs/{id}/equity` (needs the session that created the run) |
| `compare(run_ids)` | `GET /compare` |
| `leaderboard()` | `GET /api/v1/leaderboard` |
| `paper_account()` / `paper_positions()` / `paper_trades()` | `GET /paper/...` (the API routes exist; paper trading in the dashboard is coming in the future) |
| `resolve()` | `GET /api/v1/agents/resolve` |
| `backtest_schema()` | `GET /api/v1/backtest/schema` |
| `start_backtest(...)` | `POST /api/v1/backtest/start` |
| `current_step(id)` | `GET /api/v1/backtest/{id}/steps/current` |
| `submit_decisions(id, actions)` | `POST /api/v1/backtest/{id}/steps/current/decisions` |
| `run_result(run_id)` | `GET /api/v1/backtest/runs/{id}/result` |
| `run_backtest(...)` | full loop helper |

## License

OpenMDW-1.0 — see [LICENSE](LICENSE). Copyright (c) SecureFinAI Lab.
