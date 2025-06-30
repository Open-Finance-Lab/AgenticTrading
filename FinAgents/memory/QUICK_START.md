# 🧠 FinAgent Memory System - Quick Start Guide

## Overview

The FinAgent Memory System provides centralized logging and event storage for all agent pools in the FinAgent-Orchestration system. It consists of two main components:

1. **External Memory Agent** (SQLite-based) - Production-ready, high-performance
2. **Memory Server** (Neo4j-based) - Graph-based memory with advanced relationships

## ✅ External Memory Agent (Ready to Use)

### Installation

```bash
pip install aiosqlite aiofiles
```

### Basic Usage

```python
from memory.external_memory_agent import ExternalMemoryAgent, EventType, LogLevel

# Initialize
memory_agent = ExternalMemoryAgent()
await memory_agent.initialize()

# Log an event
event_id = await memory_agent.log_event(
    event_type=EventType.MARKET_DATA,
    log_level=LogLevel.INFO,
    source_agent_pool="data_agent_pool",
    source_agent_id="polygon_agent",
    title="Market data retrieved",
    content="Successfully fetched AAPL daily data",
    tags={"market_data", "AAPL"},
    metadata={"symbol": "AAPL", "records": 252}
)

# Query events
from memory.external_memory_agent import QueryFilter
query_filter = QueryFilter(content_search="AAPL", limit=10)
result = await memory_agent.query_events(query_filter)

# Cleanup
await memory_agent.cleanup()
```

### Integration with Agent Pools

```python
class YourAgentPool:
    def __init__(self):
        self.memory_agent = ExternalMemoryAgent()
    
    async def start(self):
        await self.memory_agent.initialize()
    
    async def some_action(self):
        # Log the action
        await self.memory_agent.log_event(
            event_type=EventType.OPTIMIZATION,
            log_level=LogLevel.INFO,
            source_agent_pool="your_pool",
            source_agent_id="your_agent",
            title="Action completed",
            content="Description of what happened"
        )
```

## ⚠️ Memory Server (Requires Neo4j)

### Installation

```bash
# Install Neo4j dependencies
pip install neo4j

# Install and start Neo4j database
# Download from: https://neo4j.com/deployment-center/
# Set password to: FinOrchestration
```

### Starting the Memory Server

```bash
cd FinAgents/memory
python memory_server.py
```

### Available Tools

- `store_graph_memory` - Store memories with relationships
- `retrieve_graph_memory` - Search memories with full-text
- `retrieve_memory_with_expansion` - Find related memories
- `prune_graph_memories` - Clean old memories
- `create_relationship` - Link memories manually

## 🚀 Quick Start

### Run Tests

```bash
cd FinAgents/memory
python test_memory_agent.py      # Test External Memory Agent
python test_memory_system.py     # Test full system
python demo_memory_system.py     # Interactive demo
```

### Performance

- **SQLite Backend**: 77k+ events/second
- **Query Performance**: ~2ms for typical queries
- **Storage**: Efficient with indexing on timestamp, type, and source

### Event Types

- `TRANSACTION` - Trading transactions
- `OPTIMIZATION` - Portfolio optimization
- `MARKET_DATA` - Market data operations
- `PORTFOLIO_UPDATE` - Portfolio changes
- `AGENT_COMMUNICATION` - Inter-agent messages
- `ERROR` - Error events
- `WARNING` - Warning events
- `INFO` - Informational events
- `SYSTEM` - System events
- `USER_ACTION` - User actions

### Query Capabilities

- Filter by time range
- Filter by event types
- Filter by log levels
- Filter by source agent pools
- Content search (title and content)
- Session-based queries
- Correlation-based queries
- Tag-based filtering

### Best Practices

1. **Use consistent session IDs** for workflow tracking
2. **Use correlation IDs** for related events across pools
3. **Include meaningful tags** for categorization
4. **Add relevant metadata** for context
5. **Use appropriate log levels** for filtering
6. **Clean up resources** with `await memory_agent.cleanup()`

### Real-time Hooks

```python
def error_monitor(event):
    if event.log_level == LogLevel.ERROR:
        print(f"🚨 ERROR: {event.title}")

memory_agent.add_event_hook(error_monitor)
```

## 📊 Monitoring

```python
# Get comprehensive statistics
stats = await memory_agent.get_statistics()
print(f"Total events: {stats['storage_stats']['total_events']}")
print(f"Events by type: {stats['storage_stats']['events_by_type']}")
print(f"Events by pool: {stats['storage_stats']['events_by_pool']}")
```

## 🔧 Configuration

### SQLite Backend Configuration

```python
from memory.external_memory_agent import SQLiteStorageBackend

custom_backend = SQLiteStorageBackend(db_path="custom_memory.db")
memory_agent = ExternalMemoryAgent(
    storage_backend=custom_backend,
    enable_real_time_hooks=True,
    max_batch_size=1000
)
```

### Batch Operations

```python
# For high-throughput scenarios
events_data = [
    {
        "event_type": "market_data",
        "log_level": "info",
        "source_agent_pool": "data_pool",
        "source_agent_id": "agent_1",
        "title": "Data event",
        "content": "Event content"
    }
    # ... more events
]

event_ids = await memory_agent.log_events_batch(events_data)
```

## 🚨 Error Handling

```python
try:
    await memory_agent.log_event(...)
except Exception as e:
    print(f"Failed to log event: {e}")
    # Handle gracefully - memory failures shouldn't break main logic
```

## 📁 File Structure

```
FinAgents/memory/
├── external_memory_agent.py    # Main memory agent (SQLite)
├── memory_server.py            # Neo4j-based server
├── database.py                 # Neo4j database interface
├── test_memory_agent.py        # Basic tests
├── test_memory_system.py       # Comprehensive tests
├── demo_memory_system.py       # Interactive demo
├── README.md                   # Detailed documentation
└── INTEGRATION_GUIDE.md        # This guide
```

## 🔄 Migration from Old System

If you have existing logging, you can migrate by:

1. Replace direct file logging with `memory_agent.log_event()`
2. Add session IDs for workflow tracking
3. Use structured metadata instead of free-form logs
4. Add appropriate tags for categorization

## 🚀 Next Steps

1. ✅ **Start using External Memory Agent** - It's production-ready
2. 🔧 **Integrate with your agent pools** - Add logging calls
3. 📊 **Set up monitoring dashboards** - Use statistics API
4. 🔗 **Install Neo4j** for advanced graph features (optional)
5. 🚀 **Build custom analytics** on stored events

For detailed documentation, see `README.md` in the memory directory.
