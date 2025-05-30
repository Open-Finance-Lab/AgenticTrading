
# Interface Specification Between Orchestration and Data Agent Pool (Under MCP)

## 1. Overview

This specification defines the standardized communication interfaces between the **Orchestration layer** and the **Data Agent Pool**, under the Model Context Protocol (MCP). It distinguishes clearly between control-level orchestration and capability-exposing data agents, ensuring modular, secure, and extensible system design.

---

## 2. Roles and Protocol Usage

| Component      | Role in MCP     | Protocol Used | Description                                         |
|----------------|------------------|----------------|-----------------------------------------------------|
| Orchestrator   | Host             | MCP            | Issues task DAGs and invokes agent functions        |
| MCP Client SDK | Client           | MCP (JSON-RPC) | Forwards calls from orchestrator to agent servers   |
| Data Agent     | Server           | MCP            | Implements tool functions and responds with results |

---

## 3. Core Methods (MCP JSON-RPC)

### 3.1 `agent.execute`

- **Purpose:** Trigger specific tool function on a named agent
- **Direction:** Orchestrator → Agent

**Request Format:**

```json
{
  "method": "agent.execute",
  "params": {
    "agent_id": "news_fetcher",
    "function": "fetch_recent_articles",
    "input": {
      "symbol": "AAPL"
    },
    "meta": {
      "task_id": "task_001",
      "timestamp": "2025-05-28T15:00:00Z"
    }
  }
}
```

**Response Format:**

```json
{
  "result": {
    "articles": ["Apple releases new product", "Analysts bullish on AAPL"]
  },
  "meta": {
    "status": "success",
    "execution_time_ms": 134
  }
}
```

---

### 3.2 `server.register`

- **Purpose:** Agent self-registers its tools and capabilities
- **Direction:** Agent → MCP server

**Request Format:**

```json
{
  "method": "server.register",
  "params": {
    "agent_id": "feature_extractor",
    "capabilities": {
      "tools": ["extract_features"],
      "resources": ["ohlcv_source"],
      "schema": {
        "extract_features": {
          "input": { "symbol": "string", "interval": "string" },
          "output": { "features": "list[float]" }
        }
      }
    },
    "heartbeat_interval": 60
  }
}
```

---

### 3.3 `agent.heartbeat`

- **Purpose:** Periodic check-in to indicate agent health
- **Direction:** Agent → MCP server

```json
{
  "method": "agent.heartbeat",
  "params": {
    "agent_id": "data_cleaner",
    "status": "active"
  }
}
```

---

## 4. Agent Capability Schema Requirements

Each agent **must define** the following:

- `agent_id` (string)
- `tools` (list of method names exposed)
- `resources` (optional: data APIs or services used)
- `schema` (dict of input/output structures per tool)
- `constraints` (optional: memory/time limits, usage policy)

**Example:**

```json
{
  "agent_id": "binance_agent",
  "tools": ["fetch_ohlcv"],
  "resources": ["binance_api"],
  "schema": {
    "fetch_ohlcv": {
      "input": { "symbol": "string", "interval": "string" },
      "output": { "ohlcv": "list[list[float]]" }
    }
  },
  "constraints": {
    "timeout": 5,
    "memory": "128MB"
  }
}
```

---

## 5. Execution Lifecycle (MCP-based)

1. **Startup:**  
   Agent sends `server.register` to MCP server.

2. **Invocation:**  
   Orchestrator uses MCP client to send `agent.execute` request.

3. **Execution:**  
   Agent runs logic and returns result to orchestrator.

4. **Monitoring (Optional):**  
   Agent emits `agent.heartbeat` every N seconds.

---

## 6. Error Handling Specification

### 6.1 Standard Error Format

```json
{
  "error": {
    "code": -32000,
    "message": "Function not supported",
    "data": {
      "agent_id": "unknown"
    }
  }
}
```

### 6.2 Common Error Codes

| Code   | Message                | Cause                         |
|--------|------------------------|-------------------------------|
| -32601 | Method not found       | Wrong `function` name         |
| -32602 | Invalid params         | Malformed or missing input    |
| -32000 | Agent function error   | Tool raised an exception      |
| -32001 | Timeout or unreachable | Agent did not respond         |

---

## 7. Security and Governance

- **Authentication:** All MCP requests should be signed or use secure token headers  
- **Authorization:** Orchestrator can only invoke authorized agents/tools  
- **Traceability:** All `agent.execute` calls must include `task_id` for tracking  
- **Prompt Governance:** Orchestrator maintains full visibility and control over tool payloads

---

## 8. Design Notes for System Architects

- MCP provides the **external interface boundary** between orchestration and agents  
- A2A may be used **internally between agents** for decentralized collaboration, not control  
- Agents should be stateless or externally state-backed (e.g., via memory DBs)  
- Consider use of a `tool registry service` for runtime introspection and tool selection
