# FinAgents Memory Clients

This module provides unified async memory clients for use by agent pools in the FinAgent-Orchestration system. These clients allow agents to store, retrieve, and filter structured memory events via the MCP memory server and Neo4j backend.

## Standard Memory Agent Interface

The memory agent clients (`AlphaMemoryClient` and `DataMemoryClient`) expose a standard async interface for interacting with the memory system. This interface is designed to be simple, consistent, and extensible for all agent pools.

### Core Methods

All memory clients provide the following async methods:

- `store_event(event_type: str, summary: str, keywords: List[str], details: Dict[str, Any], log_level: str = "INFO", session_id: Optional[str] = None, correlation_id: Optional[str] = None)`
  - Store a structured event (transaction, info, etc.) in the memory agent.
- `store_error(summary: str, details: Dict[str, Any], keywords: List[str] = None, session_id: Optional[str] = None, correlation_id: Optional[str] = None)`
  - Store an error event in the memory agent (convenience wrapper for `store_event`).
- `store_action(summary: str, details: Dict[str, Any], keywords: List[str] = None, session_id: Optional[str] = None, correlation_id: Optional[str] = None)`
  - Store a generic agent action in the memory agent (convenience wrapper for `store_event`).
- `retrieve_events(search_query: str, limit: int = 10)`
  - Retrieve events from the memory agent using a full-text search.
- `retrieve_expanded(search_query: str, limit: int = 10)`
  - Retrieve events and their related memories using expansion (includes contextually linked events).
- `filter_events(filters: Dict[str, Any], limit: int = 100, offset: int = 0)`
  - Filter events using structured criteria (time, event_type, log_level, etc.).

All methods are asynchronous and return the result as a dictionary (parsed from the MCP server response).

### Example Usage

```python
from FinAgents.memory.alpha_memory_client import AlphaMemoryClient
import asyncio

async def main():
    client = AlphaMemoryClient(agent_id="alpha_agent_1")
    await client.store_event(
        event_type="USER_QUERY",
        summary="User asked for AAPL forecast",
        keywords=["AAPL", "forecast"],
        details={"query": "What is the forecast for AAPL?"}
    )
    results = await client.retrieve_events(search_query="AAPL", limit=5)
    print(results)

asyncio.run(main())
```

## Supported Clients

- **AlphaMemoryClient**: For use by the alpha agent pool
- **DataMemoryClient**: For use by the data agent pool

Both clients provide the same async interface for storing events, errors, actions, and retrieving/filtering memories.

## MCP Memory Server Dependency

Both clients require the MCP memory server to be running and accessible (default: `http://127.0.0.1:8010/mcp`). The memory server uses a Neo4j database (default: `bolt://localhost:7687`).

Start the server with:

```sh
uvicorn FinAgents.memory.memory_server:app --reload
```

Ensure Neo4j is running and accessible with the correct credentials.

## Usage Example

### DataMemoryClient
```python
from FinAgents.memory.data_memory_client import DataMemoryClient
import asyncio

async def main():
    client = DataMemoryClient(agent_id="data_agent_1")
    await client.store_event(
        event_type="DATA_EVENT",
        summary="Fetched historical data for MSFT",
        keywords=["MSFT", "historical"],
        details={"symbol": "MSFT", "range": "2020-2023"}
    )
    results = await client.filter_events(filters={"agent_id": "data_agent_1"}, limit=10)
    print(results)

asyncio.run(main())
```

## Methods
- `store_event(...)`: Store a structured event
- `store_error(...)`: Store an error event
- `store_action(...)`: Store a generic agent action
- `retrieve_events(...)`: Retrieve events by full-text search
- `retrieve_expanded(...)`: Retrieve events and related memories
- `filter_events(...)`: Filter events by structured criteria

See the code for full method signatures and details. 