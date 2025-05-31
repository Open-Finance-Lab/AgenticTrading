# 📊 Data Agent Pool – FinAgent-Orchestration

The **Data Agent Pool** is a modular, schema-driven, and protocol-compatible subcomponent of the broader FinAgent-Orchestration system. It provides a unified interface for interacting with diverse market data sources, including crypto, equity, and news APIs.

---

## 🚀 Features

- ✅ **Unified Agent Interface** with `execute()` dispatch
- ✅ **Schema-based Configuration** using Pydantic + YAML
- ✅ **Support for Multiple Data Domains**:
  - `binance_agent` – Binance OHLCV data
  - `coinbase_agent` – Coinbase spot price
  - `alpaca_agent` – Alpaca equity data
  - `iex_agent` – IEX Cloud quote data
  - `newsapi_agent` – News headlines from NewsAPI
  - `rss_agent` – Custom RSS feed support
- ✅ **MCP-Compatible HTTP Server** with tool/resource support
- ✅ **Unified Client Script** to validate all agents

---

## 🧱 Folder Structure

```
data_agent_pool/
│
├── agents/
│   ├── crypto/
│   ├── equity/
│   └── news/
│
├── schema/
│   ├── crypto_schema.py
│   ├── equity_schema.py
│   └── news_schema.py
│
├── config/
│   ├── binance.yaml
│   ├── coinbase.yaml
│   ├── alpaca.yaml
│   ├── iex.yaml
│   ├── newsapi.yaml
│   └── rss.yaml
│
├── mcp_server.py
├── registry.py
└── unified_test_client.py
```

---

## 🛠️ How to Use

### 1. Start the Server
```bash
uvicorn mcp_server:app --port 8001 --reload
```

### 2. Test All Agents
```bash
python unified_test_client.py
```

### 3. Add a New Agent
- Create a new agent class in `agents/<domain>/`
- Define a Pydantic schema in `schema/`
- Add YAML config in `config/`
- Register it in `registry.py`

---

## 📎 Notes

- All configurations are validated using strict `Pydantic` schemas.
- Agent loading uses `load_config()` to parse YAML into schema-bound objects.
- Errors in missing config fields will be caught at load-time.

---

## 📬 MCP Tools & Resources

- `agent.execute`: Dispatch a method call to any registered agent.
- `register://<agent_id>`: Register a new agent manually.
- `heartbeat://<agent_id>`: Check liveness of an agent.

---

## 📍 Next Steps

- Add logging and memory database integration
- Connect to DAG Orchestrator
- Integrate into Alpha & Execution Agent feedback loop

---

FinAgent-Orchestration © 2025 – Designed for Adaptive, Composable, and Explainable Trading Systems.