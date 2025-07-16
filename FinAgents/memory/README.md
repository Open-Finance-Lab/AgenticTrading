# FinAgents Memory Clients

This module provides unified async memory clients for use by agent pools in the FinAgent-Orchestration system. These clients allow agents to store, retrieve, and filter structured memory events via the MCP memory server and Neo4j backend.

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

### AlphaMemoryClient
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