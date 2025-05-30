# 📘 Interface Design: FinAgent System (Protocol-Driven)

This document defines the API interface specifications for the **Data Agent Pool**, **Orchestrator**, **Alpha Agent Pool**, and **Memory Agent** under the protocol constraints of **MCP**, **ACP**, **A2A**, and **ANP**.

---

## 🧩 Protocol Overview

| Protocol | Full Name | Purpose |
|----------|-----------|---------|
| **MCP**  | Model Context Protocol | Agent inference, feature-based model/service calling |
| **ACP**  | Agent Communication Protocol | Status updates, logging, registry operations |
| **A2A**  | Agent-to-Agent Protocol | Critique exchange, voting, negotiation |
| **ANP**  | Agent Network Protocol | Discovery, registration, capability broadcasting |

---

## 1. 🧠 Data Agent Pool (MCP + ACP)

### `POST /data_agent/get_features` (MCP)

Extract structured features from market data.

```json
Request:
{
  "market_data_id": "md_20250519",
  "timestamp": "2025-05-19T13:00:00Z",
  "config": {...}
}
```

```json
Response:
{
  "features": {
    "sma_5": 1.02,
    "rsi_14": 67.3,
    ...
  }
}
```

---

### `POST /data_agent/register` (ACP)

Register agent metadata to the system directory.

```json
{
  "agent_id": "data_agent_001",
  "type": "feature_extractor",
  "capabilities": ["ohlc", "volume", "rsi"],
  "frequency": "1min"
}
```

---

## 2. 🧭 Orchestrator (MCP + ACP + A2A + ANP)

### `POST /orchestrator/request_plan` (MCP)

Request plan from a strategy agent.

```json
{
  "strategy_type": "momentum",
  "features": {...}
}
```

---

### `POST /orchestrator/log_plan_result` (ACP)

Log result of a strategy execution.

```json
{
  "plan_id": "plan_001",
  "result": {
    "pnl": 0.012,
    "drawdown": 0.004,
    "executed": true
  }
}
```

---

### `POST /orchestrator/broadcast_plan` (A2A)

Broadcast plan to Critique Agents for evaluation.

```json
{
  "plan_id": "plan_001",
  "weights": {...},
  "origin": "MomentumAgent"
}
```

---

### `GET /orchestrator/discover_agents` (ANP)

Retrieve list of available agents and metadata.

```json
Response:
[
  {
    "agent_id": "meanrev_02",
    "type": "strategy",
    "status": "available"
  },
  ...
]
```

---

## 3. ⚙️ Alpha Agents Pool (MCP + A2A)

### `POST /alpha_agent/propose_plan` (MCP)

Propose a trading plan based on input features.

```json
{
  "features": {...}
}
```

```json
Response:
{
  "plan_id": "plan_001",
  "weights": {"AAPL": 0.6, "SPY": 0.4},
  "confidence": 0.87
}
```

---

### `POST /alpha_agent/receive_critique` (A2A)

Receive structured critique from another agent.

```json
{
  "from": "CritiqueAgent",
  "plan_id": "plan_001",
  "reason": "Overweight on AAPL during high volatility",
  "score": -0.3
}
```

---

### `POST /alpha_agent/vote_on_plan` (A2A)

Vote on a peer’s plan.

```json
{
  "plan_id": "plan_002",
  "vote": 0.72
}
```

---

## 4. 💾 Memory Agent (ACP + MCP)

### `POST /memory/record_execution` (ACP)

Log execution result and reward.

```json
{
  "plan_id": "plan_001",
  "reward": 0.09,
  "return": 0.11,
  "risk": 0.03
}
```

---

### `POST /memory/log_critique` (ACP)

Log critique statement and associated vector.

```json
{
  "plan_id": "plan_001",
  "agent": "CritiqueAgent",
  "comment": "Too reactive in mean-reverting conditions",
  "vector": [0.12, -0.05, ...]
}
```

---

### `POST /memory/query_similar_plan` (MCP)

Retrieve past plans similar to the current features.

```json
{
  "features": {...}
}
```

```json
Response:
[
  {
    "plan_id": "plan_033",
    "reward": 0.08,
    "features_similarity": 0.92
  }
]
```

---

## ✅ Suggested Technologies

| Layer | Tech |
|-------|------|
| API transport | `FastAPI`, `aiohttp`, `gRPC` |
| Pub/Sub for A2A | `Redis`, `ZeroMQ`, `asyncio.Queue` |
| Memory DB | `PostgreSQL`, `DuckDB`, `FAISS` |
| Protocol routing | Internal Dispatcher per protocol type |

---

## 📌 Next Steps

- Implement MVP with Data → Strategy → Orchestrator → Executor → Memory path (MCP+ACP)
- Add Critique + MetaPlanner loop (A2A)
- Enable ANP with Agent Registry auto-discovery
