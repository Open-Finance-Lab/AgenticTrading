# DAG Planner Component

## Overview

The DAG (Directed Acyclic Graph) Planner is a critical component in the FinAgent-Orchestration system, responsible for converting high-level strategic queries into executable task flows. It implements a protocol-oriented architecture that enables dynamic composition of agent behaviors through task-specific execution graphs.

## Architecture

### Core Components

1. **DAGPlannerAgent (Abstract Base Class)**
   - Defines the interface for DAG planning operations
   - Manages task registry and DAG construction
   - Provides utilities for DAG validation and execution order determination
   - Implemented by concrete planners (e.g., MockDAGPlanner for testing)

2. **Task Management**
   - `TaskNode`: Represents individual tasks in the DAG
   - `TaskDefinition`: Defines task properties and requirements
   - `AgentType`: Enumerates supported agent types (DATA, ALPHA, RISK, etc.)

3. **Communication Protocol (MCP)**
   - Implements the Multi-agent Control Protocol for DAG planning
   - Handles message serialization/deserialization
   - Manages request-response lifecycle
   - Supports asynchronous communication

### Directory Structure

```
DAG_planner/
├── __init__.py
├── planner.py           # Core DAG planning logic
├── task.py             # Task definitions and types
├── server.py           # MCP server implementation
├── client.py           # MCP client implementation
├── protocols/
│   ├── __init__.py
│   └── planner_protocol.py  # Message definitions and serialization
└── tests/
    ├── __init__.py
    ├── conftest.py     # Test configuration and fixtures
    └── test_planner.py # Test suite
```

## Design Principles

### 1. Protocol-Oriented Architecture

The system employs a protocol-oriented design pattern, where communication between components is governed by well-defined protocols:

- **Message Types**: Each message type (QUERY, MEMORY_UPDATE, DAG_RESPONSE, etc.) is explicitly defined
- **Serialization**: JSON-based message serialization for platform independence
- **Error Handling**: Comprehensive error handling and status reporting
- **Correlation**: Request tracking through correlation IDs

### 2. Asynchronous Execution

The implementation leverages Python's asyncio for non-blocking operations:

- Asynchronous server-client communication
- Concurrent task planning and execution
- Efficient resource utilization
- Scalable request handling

### 3. State Management

The system maintains several state collections:

- `active_requests`: Tracks in-progress planning tasks
- `completed_requests`: Stores successfully planned DAGs
- `memory_context`: Maintains system-wide context for planning
- `task_registry`: Manages task definitions and dependencies

## Communication Flow

1. **Query Reception**
   - Client sends a planning query with context
   - Server validates and processes the request
   - Correlation ID is generated for request tracking

2. **DAG Planning**
   - Planner agent converts query into task DAG
   - Dependencies are analyzed and validated
   - Task registry is updated with new tasks

3. **Response Handling**
   - Planned DAG is serialized and stored
   - Response is sent back to client
   - Results are cached for future requests

## Testing Strategy

The component implements a comprehensive testing strategy:

1. **Unit Tests**
   - Protocol message serialization
   - Task management operations
   - DAG validation and construction

2. **Integration Tests**
   - Server-client communication
   - End-to-end workflow validation
   - Memory context integration

3. **Mock Implementations**
   - MockDAGPlanner for testing
   - Simulated agent behaviors
   - Controlled test environments

## Usage Example

```python
# Create and start server
planner = DAGPlannerAgent()
server = DAGPlannerServer(planner)
await server.start()

# Client usage
client = DAGPlannerClient()
await client.connect()

# Plan DAG
dag = await client.plan_dag(
    query="Trading strategy",
    context={"market": "US", "timeframe": "1d"}
)

# Update memory
await client.update_memory({
    "market_data": {"AAPL": 150.0},
    "risk_metrics": {"volatility": 0.2}
})
```

## Future Enhancements

1. **Performance Optimization**
   - Implement caching strategies
   - Optimize DAG validation
   - Add request batching

2. **Feature Extensions**
   - Support for dynamic DAG modification
   - Enhanced error recovery
   - Real-time planning updates

3. **Monitoring and Observability**
   - Add metrics collection
   - Implement tracing
   - Enhanced logging

## Dependencies

- `networkx`: DAG construction and validation
- `asyncio`: Asynchronous operations
- `pydantic`: Data validation
- `pytest`: Testing framework
- `pytest-asyncio`: Async test support

## Contributing

When contributing to this component, please ensure:

1. All new features include comprehensive tests
2. Protocol changes maintain backward compatibility
3. Documentation is updated accordingly
4. Code follows the established style guide 