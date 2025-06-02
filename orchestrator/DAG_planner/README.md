# DAG Planner Component

## Project Status
![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-green)
![License](https://img.shields.io/badge/license-OpenMDW-yellow)

## Abstract

The DAG (Directed Acyclic Graph) Planner constitutes a fundamental component within the FinAgent-Orchestration system, facilitating the transformation of high-level strategic queries into executable task workflows. This component implements a protocol-oriented architecture that enables dynamic composition of agent behaviors through task-specific execution graphs, thereby enhancing the system's adaptability and extensibility.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

1. Create a configuration file `config.yaml`:
```yaml
planner:
  host: localhost
  port: 8000
  max_workers: 4
  timeout: 30
```

2. Set environment variables:
```bash
export DAG_PLANNER_CONFIG_PATH=/path/to/config.yaml
```

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
├── config.py           # Configuration management
├── exceptions.py       # Custom exceptions
├── utils/             # Utility functions
│   ├── __init__.py
│   ├── validation.py
│   └── logging.py
├── protocols/
│   ├── __init__.py
│   └── planner_protocol.py  # Message definitions and serialization
└── tests/
    ├── __init__.py
    ├── conftest.py     # Test configuration and fixtures
    ├── test_planner.py # Test suite
    └── test_integration.py # Integration tests
```

## API Documentation

### DAGPlannerAgent

```python
class DAGPlannerAgent:
    async def plan_dag(self, query: str, context: Dict) -> DAG:
        """
        Plans a DAG based on the provided query and context.
        
        Parameters:
            query (str): The planning query
            context (Dict): Contextual information
            
        Returns:
            DAG: The constructed directed acyclic graph
        """
        pass

    async def update_memory(self, memory_update: Dict) -> None:
        """
        Updates the system's memory context.
        
        Parameters:
            memory_update (Dict): Memory update data
        """
        pass
```

### DAGPlannerClient

```python
class DAGPlannerClient:
    async def connect(self) -> None:
        """Establishes connection to the DAG Planner server"""
        pass

    async def plan_dag(self, query: str, context: Dict) -> DAG:
        """Submits a planning request"""
        pass

    async def update_memory(self, memory_update: Dict) -> None:
        """Updates system memory"""
        pass
```

## Implementation Details

### Design Principles

1. **Protocol-Oriented Architecture**
   - Well-defined message types (QUERY, MEMORY_UPDATE, DAG_RESPONSE)
   - JSON-based message serialization
   - Comprehensive error handling
   - Request correlation through unique identifiers

2. **Asynchronous Execution**
   - Non-blocking server-client communication
   - Concurrent task planning and execution
   - Efficient resource utilization
   - Scalable request handling

3. **State Management**
   - Active request tracking
   - Completed request storage
   - System-wide context maintenance
   - Task registry management

### Communication Flow

1. **Query Reception**
   - Client query submission with context
   - Server-side request validation
   - Correlation ID generation

2. **DAG Planning**
   - Query-to-DAG transformation
   - Dependency analysis and validation
   - Task registry updates

3. **Response Handling**
   - DAG serialization and storage
   - Client response transmission
   - Result caching

## Testing Methodology

### Unit Testing
- Protocol message serialization
- Task management operations
- DAG validation and construction

### Integration Testing
- Server-client communication
- End-to-end workflow validation
- Memory context integration

### Mock Implementations
- MockDAGPlanner for testing
- Simulated agent behaviors
- Controlled test environments

## Performance Optimization

### Caching Strategies
- LRU cache implementation
- Query result caching
- Memory usage optimization

### Concurrency Management
- Parallel task execution
- Resource pool management
- Load balancing

## Monitoring and Observability

### Metrics Collection
- Request latency
- Memory utilization
- Task completion rate
- Error rate

### Logging Configuration
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Contribution Guidelines

### Code Standards
- PEP 8 compliance
- Type annotations
- Comprehensive docstrings

### Testing Requirements
- Unit test coverage > 80%
- Integration test suite
- Performance benchmarks

### Submission Process
- Feature branch creation
- Test case implementation
- Pull request submission

### Documentation
- API documentation updates
- Usage examples
- Changelog maintenance

## Dependencies

- `networkx>=2.6.3`: DAG construction and validation
- `asyncio>=3.4.3`: Asynchronous operations
- `pydantic>=1.9.0`: Data validation
- `pytest>=7.0.0`: Testing framework
- `pytest-asyncio>=0.18.0`: Async test support
- `pyyaml>=6.0`: Configuration management
- `structlog>=22.1.0`: Structured logging

## License

This component is part of the FinAgent-Orchestration project and is licensed under the OpenMDW License. See the [LICENSE](../../LICENSE) file in the project root directory for details.

## Contact Information

- Issue Tracking: GitHub Issues
- Email: [Maintainer Email]
- Documentation: [Documentation Link] 