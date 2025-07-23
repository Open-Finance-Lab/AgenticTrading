# FinAgent Memory System - Modular Architecture

## 🚀 Overview

The FinAgent Memory System has been completely redesigned with a **modular architecture** that provides clean separation of concerns, unified database management, and protocol-agnostic interfaces. This new architecture supports multiple deployment scenarios and maintains backward compatibility while enabling advanced features.

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Core Components](#core-components)
- [Server Types](#server-types)
- [Configuration Management](#configuration-management)
- [Installation & Setup](#installation--setup)
- [Usage Examples](#usage-examples)
- [API Documentation](#api-documentation)
- [Development Guide](#development-guide)
- [Deployment Scenarios](#deployment-scenarios)
- [Troubleshooting](#troubleshooting)

## 🏗️ Architecture Overview

The modular architecture consists of **unified core components** and **specialized protocol servers**:

```
┌─────────────────────────────────────────────────────────────┐
│                    FINAGENT MEMORY SYSTEM                  │
├─────────────────────────────────────────────────────────────┤
│  🔧 MCP Server     🧠 Memory Server     📡 A2A Server      │
│  (Port 8001)       (Port 8000)         (Port 8002)        │
├─────────────────────────────────────────────────────────────┤
│              🔄 UNIFIED INTERFACE MANAGER                  │
│  Protocol-agnostic tool definitions and execution         │
├─────────────────────────────────────────────────────────────┤
│              🗄️ UNIFIED DATABASE MANAGER                   │
│  Centralized Neo4j operations with intelligent indexing   │
├─────────────────────────────────────────────────────────────┤
│                    📊 CONFIGURATION LAYER                 │
│  Environment-specific settings and deployment configs     │
└─────────────────────────────────────────────────────────────┘
```

### Key Benefits

- ✅ **Clean Separation**: Each server handles specific protocols
- ✅ **Unified Backend**: Shared database and interface management
- ✅ **Scalable**: Easy to add new protocols or server types
- ✅ **Configurable**: Environment-specific configurations
- ✅ **Backward Compatible**: Legacy components still work
- ✅ **Easy Deployment**: Single launcher for all components

## 🧩 Core Components

### 1. Unified Database Manager (`unified_database_manager.py`)

Centralized database operations with advanced features:

- **UnifiedDatabaseManager**: Core database abstraction
- **Intelligent Indexing**: Automatic index creation and optimization
- **Semantic Search**: Vector-based similarity search
- **Legacy Compatibility**: TradingGraphMemory wrapper
- **Health Monitoring**: Connection and performance monitoring

```python
# Create database manager
db_manager = create_database_manager({
    "uri": "bolt://localhost:7687",
    "username": "neo4j", 
    "password": "FinOrchestration"
})

# Store memory with intelligent indexing
await db_manager.store_memory(
    query="BUY AAPL signal from momentum agent",
    keywords=["signal", "buy", "AAPL"],
    summary="Strong buy signal detected",
    agent_id="momentum_agent"
)
```

### 2. Unified Interface Manager (`unified_interface_manager.py`)

Protocol-agnostic tool management:

- **Tool Registration**: Centralized tool definitions
- **Multi-Protocol**: Supports MCP, HTTP, A2A, WebSocket
- **Execution Engine**: Unified tool execution with error handling
- **Performance Monitoring**: Tool usage analytics
- **Dynamic Loading**: Runtime tool registration

```python
# Create interface manager
interface_manager = create_interface_manager(database_config)

# Execute tool across any protocol
result = await interface_manager.execute_tool("store_graph_memory", {
    "query": "Market analysis complete",
    "agent_id": "analysis_agent"
})
```

### 3. Configuration Manager (`configuration_manager.py`)

Environment-specific configuration management:

- **Multi-Environment**: Development, Testing, Staging, Production
- **Server-Specific**: Different configs for MCP, A2A, Memory servers
- **Auto-Detection**: Environment detection from env vars
- **Validation**: Configuration validation and defaults
- **Export/Import**: YAML and JSON configuration support

```python
# Auto-configure based on environment
config = auto_configure()

# Get database config for current environment
db_config = get_database_config()

# Get server config for specific server type
server_config = get_server_config_dict(ServerType.MCP)
```

## 🖥️ Server Types

### Memory Server (Port 8000)

Enhanced memory server with dual architecture support:

- **Primary Interface**: Main memory operations endpoint
- **Dual Architecture**: Unified + Legacy support
- **FastAPI Backend**: RESTful API with automatic documentation
- **Health Monitoring**: Comprehensive system health checks
- **Error Handling**: Graceful fallback mechanisms

**Key Endpoints:**
- `POST /store` - Store memory with intelligent indexing
- `POST /retrieve` - Retrieve memories with semantic search
- `GET /health` - System health and component status
- `GET /docs` - Interactive API documentation

### MCP Server (Port 8001)

Model Context Protocol implementation:

- **MCP Compliance**: Full MCP protocol implementation
- **Tool Definitions**: Comprehensive memory operation tools
- **Lifecycle Management**: Proper MCP server lifecycle
- **Error Handling**: MCP-compliant error responses
- **Performance Monitoring**: Tool execution metrics

**Available Tools:**
- `store_graph_memory` - Store graph-based memories
- `retrieve_graph_memory` - Retrieve with graph traversal
- `semantic_search` - Vector-based similarity search
- `health_check` - System health verification
- `list_memories` - Memory enumeration and filtering

### A2A Server (Port 8002)

Agent-to-Agent communication server:

- **Signal Transmission**: Trading signal distribution
- **Strategy Sharing**: Strategy exchange between agents
- **Memory Sharing**: Collaborative memory access
- **WebSocket Support**: Real-time communication
- **Analytics**: Communication monitoring and metrics

**Key Features:**
- Real-time signal broadcasting
- Strategy performance tracking
- Agent connection management
- Message history and analytics
- Priority-based message handling

## ⚙️ Configuration Management

### Environment Types

1. **Development** (`development`):
   - Debug mode enabled
   - Verbose logging
   - Small connection pools
   - Local database

2. **Testing** (`testing`):
   - Test database isolation
   - Limited resources
   - Comprehensive logging
   - Mock services

3. **Staging** (`staging`):
   - Production-like setup
   - API key authentication
   - File logging enabled
   - Performance monitoring

4. **Production** (`production`):
   - Optimized performance
   - Security hardened
   - Restricted CORS
   - Error-only logging

### Configuration Files

Configuration files are stored in the `config/` directory:

```
config/
├── development.yaml    # Development environment
├── testing.yaml       # Testing environment  
├── staging.yaml       # Staging environment
└── production.yaml    # Production environment
```

### Environment Variables

Set the environment using the `FINAGENT_ENV` variable:

```bash
export FINAGENT_ENV=production  # Use production config
export FINAGENT_ENV=development # Use development config (default)
```

## 🔧 Installation & Setup

### Prerequisites

1. **Python 3.8+**
2. **Neo4j Database** (local or remote)
3. **Required Python packages** (see requirements below)

### Core Dependencies

```bash
# Essential packages
pip install neo4j>=5.0.0
pip install numpy>=1.21.0

# Web server packages (for Memory and A2A servers)
pip install fastapi>=0.100.0
pip install uvicorn>=0.20.0
pip install pydantic>=2.0.0

# MCP server packages  
pip install fastmcp>=0.1.0

# Configuration packages
pip install pyyaml>=6.0
pip install python-dotenv>=1.0.0

# Optional packages
pip install psutil>=5.9.0  # Process monitoring
pip install websockets>=11.0  # WebSocket support
```

### Quick Setup

1. **Clone and navigate to the memory directory:**
   ```bash
   cd FinAgent-Orchestration/FinAgents/memory/
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Neo4j database:**
   ```bash
   # Using Docker
   docker run -p 7687:7687 -p 7474:7474 \
     -e NEO4J_AUTH=neo4j/FinOrchestration \
     neo4j:latest

   # Or start local Neo4j service
   neo4j start
   ```

4. **Configure environment (optional):**
   ```bash
   export FINAGENT_ENV=development
   ```

5. **Launch all servers:**
   ```bash
   python launcher.py
   ```

## 🚀 Usage Examples

### Using the Launcher

The launcher provides unified management of all servers:

```bash
# Start all servers in development mode
python launcher.py

# Start specific servers
python launcher.py --servers memory mcp

# Use production environment
python launcher.py --env production

# Show configuration summary
python launcher.py --config-summary

# Show server information
python launcher.py --server-info
```

### Individual Server Usage

#### Memory Server

```python
import asyncio
from memory_server import app, interface_manager

# Store a memory
async def store_example():
    result = await interface_manager.execute_tool("store_graph_memory", {
        "query": "AAPL showing strong momentum with RSI at 75",
        "keywords": ["AAPL", "momentum", "RSI", "overbought"],
        "summary": "Technical analysis indicates potential reversal",
        "agent_id": "technical_analysis_agent",
        "event_type": "ANALYSIS_COMPLETE"
    })
    print(f"Stored memory: {result}")

# Retrieve memories
async def retrieve_example():
    result = await interface_manager.execute_tool("retrieve_graph_memory", {
        "query": "AAPL technical analysis",
        "agent_id": "technical_analysis_agent",
        "limit": 10
    })
    print(f"Retrieved {len(result.get('memories', []))} memories")

# Run examples
asyncio.run(store_example())
asyncio.run(retrieve_example())
```

#### MCP Server

```python
# MCP server runs as stdio transport
# Use with MCP clients like Claude Desktop or custom integrations

# Example client usage:
from mcp_client import create_mcp_client

async def mcp_example():
    client = create_mcp_client()
    
    # Store memory via MCP
    result = await client.call_tool("store_graph_memory", {
        "query": "Portfolio rebalancing completed",
        "agent_id": "portfolio_agent"
    })
    
    # Perform semantic search
    search_result = await client.call_tool("semantic_search", {
        "query": "portfolio management strategies",
        "limit": 5
    })
```

#### A2A Server

```python
import aiohttp
import asyncio

# Send trading signal
async def send_signal():
    signal_data = {
        "signal_id": "signal_001",
        "source_agent": "momentum_agent",
        "target_agents": ["portfolio_agent", "risk_agent"],
        "signal_type": "buy",
        "symbol": "AAPL",
        "confidence": 0.85,
        "price": 150.25
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8002/api/v1/signals/send",
            json=signal_data
        ) as response:
            result = await response.json()
            print(f"Signal sent: {result}")

# Share strategy
async def share_strategy():
    strategy_data = {
        "strategy_id": "momentum_v2",
        "source_agent": "strategy_agent",
        "strategy_name": "Enhanced Momentum Strategy",
        "strategy_data": {
            "lookback_period": 20,
            "rsi_threshold": 70,
            "volume_multiplier": 1.5
        },
        "performance_metrics": {
            "sharpe_ratio": 1.45,
            "max_drawdown": -0.12,
            "win_rate": 0.68
        }
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8002/api/v1/strategies/share",
            json=strategy_data
        ) as response:
            result = await response.json()
            print(f"Strategy shared: {result}")

# Run examples
asyncio.run(send_signal())
asyncio.run(share_strategy())
```

### WebSocket Communication (A2A)

```python
import websockets
import json
import asyncio

async def websocket_agent_example():
    uri = "ws://localhost:8002/ws/my_agent"
    
    async with websockets.connect(uri) as websocket:
        # Send ping
        ping_message = {"type": "ping"}
        await websocket.send(json.dumps(ping_message))
        
        # Listen for messages
        async for message in websocket:
            data = json.loads(message)
            print(f"Received: {data}")
            
            if data.get("type") == "pong":
                print("Pong received!")
                break

asyncio.run(websocket_agent_example())
```

## 📖 API Documentation

### Memory Server API

The Memory Server provides comprehensive REST API documentation via FastAPI:

- **Interactive Docs**: `http://localhost:8000/docs`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

### Health Check Endpoints

All servers provide health check endpoints:

```bash
# Memory Server health
curl http://localhost:8000/health

# A2A Server health  
curl http://localhost:8002/health

# Comprehensive status
curl http://localhost:8000/status
curl http://localhost:8002/status
```

### Response Formats

All APIs use consistent JSON response formats:

```json
{
  "status": "success",
  "data": {
    "memory_id": "mem_12345",
    "message": "Memory stored successfully"
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "execution_time": 0.045
}
```

Error responses:

```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid query parameter",
    "details": "Query cannot be empty"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## 👨‍💻 Development Guide

### Project Structure

```
FinAgents/memory/
├── unified_database_manager.py      # Centralized database operations
├── unified_interface_manager.py     # Protocol-agnostic tool management
├── configuration_manager.py         # Environment configuration management
├── memory_server.py                 # Enhanced memory server (FastAPI)
├── mcp_server.py                    # MCP protocol server
├── a2a_server.py                    # Agent-to-Agent communication server
├── launcher.py                      # Unified server launcher
├── README_Modular_Architecture.md   # This documentation
├── config/                          # Configuration files
│   ├── development.yaml
│   ├── testing.yaml
│   ├── staging.yaml
│   └── production.yaml
└── logs/                           # Log files (when file logging enabled)
```

### Adding New Tools

1. **Define tool in Unified Interface Manager:**

```python
# In unified_interface_manager.py
@self.register_tool
async def my_new_tool(self, param1: str, param2: int) -> Dict[str, Any]:
    \"\"\"
    Description of the new tool.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
    
    Returns:
        Dictionary with tool results
    \"\"\"
    try:
        # Tool implementation
        result = await self.database_manager.some_operation(param1, param2)
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}
```

2. **Tool automatically becomes available in all servers** (MCP, Memory, A2A)

### Adding New Server Types

1. **Create new server file** following the pattern of existing servers
2. **Update Configuration Manager** with new server type
3. **Update Launcher** to support new server type
4. **Add to documentation**

### Testing

```bash
# Run individual component tests
python -m pytest tests/test_unified_database_manager.py
python -m pytest tests/test_unified_interface_manager.py
python -m pytest tests/test_configuration_manager.py

# Integration tests
python -m pytest tests/test_integration.py

# Test specific server
python -m pytest tests/test_memory_server.py
python -m pytest tests/test_mcp_server.py
python -m pytest tests/test_a2a_server.py
```

## 🚀 Deployment Scenarios

### Development Deployment

```bash
# Start in development mode with all debugging features
export FINAGENT_ENV=development
python launcher.py --servers all

# Or start individual servers for debugging
python memory_server.py
python mcp_server.py  
python a2a_server.py
```

### Production Deployment

```bash
# Set production environment
export FINAGENT_ENV=production

# Start with production configuration
python launcher.py --env production

# Or use process manager like systemd/supervisor
systemctl start finagent-memory-system
```

### Docker Deployment

```dockerfile
# Dockerfile example
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

ENV FINAGENT_ENV=production
EXPOSE 8000 8001 8002

CMD ["python", "launcher.py", "--env", "production"]
```

### Kubernetes Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: finagent-memory-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: finagent-memory
  template:
    metadata:
      labels:
        app: finagent-memory
    spec:
      containers:
      - name: finagent-memory
        image: finagent/memory-system:2.0.0
        env:
        - name: FINAGENT_ENV
          value: "production"
        ports:
        - containerPort: 8000
        - containerPort: 8001
        - containerPort: 8002
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: Missing dependency imports
```
❌ FastAPI not available. Install with: pip install fastapi uvicorn
❌ Unified components not available
```

**Solution**:
```bash
pip install fastapi uvicorn pydantic
pip install neo4j numpy
pip install fastmcp  # For MCP server
```

#### 2. Database Connection Issues

**Problem**: Cannot connect to Neo4j
```
❌ Database connection failed: ServiceUnavailable
```

**Solution**:
```bash
# Check Neo4j is running
neo4j status

# Verify connection parameters
neo4j-admin dbms set-default-admin neo4j

# Test connection
python -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'FinOrchestration')); driver.verify_connectivity()"
```

#### 3. Port Conflicts

**Problem**: Address already in use
```
❌ [Errno 48] Address already in use
```

**Solution**:
```bash
# Find process using port
lsof -i :8000

# Kill process or use different port
python launcher.py --servers memory --port 8010
```

#### 4. Configuration Issues

**Problem**: Invalid configuration
```
❌ Configuration validation failed
```

**Solution**:
```bash
# Check configuration
python launcher.py --config-summary

# Recreate default configs
python configuration_manager.py
```

### Performance Optimization

#### Database Performance

1. **Enable indexing**:
   ```python
   config.database.enable_intelligent_indexing = True
   config.database.auto_index_creation = True
   ```

2. **Optimize connection pool**:
   ```python
   config.database.max_connection_pool_size = 100
   config.database.connection_timeout = 30
   ```

3. **Enable caching**:
   ```python
   config.memory.cache_enabled = True
   config.memory.memory_cache_size = 1000
   ```

#### Server Performance

1. **Use multiple workers** (production):
   ```bash
   uvicorn memory_server:app --workers 4 --host 0.0.0.0 --port 8000
   ```

2. **Enable gzip compression**:
   ```python
   app.add_middleware(GZipMiddleware, minimum_size=1000)
   ```

3. **Configure timeouts**:
   ```python
   config.server.keepalive_timeout = 5
   config.server.max_connections = 1000
   ```

### Monitoring and Logging

#### Enable detailed logging

```python
# Set debug logging
config.logging.level = "DEBUG"
config.logging.file_enabled = True
config.logging.file_path = "logs/finagent.log"
```

#### Monitor health endpoints

```bash
# Automated health monitoring
watch -n 5 'curl -s http://localhost:8000/health | jq .'
```

#### Performance metrics

```python
# Get server analytics
curl http://localhost:8002/api/v1/analytics/summary
```

## 📞 Support & Contributing

### Getting Help

1. **Check the documentation** in this README
2. **Review configuration** with `python launcher.py --config-summary`
3. **Check server info** with `python launcher.py --server-info`
4. **Review logs** in the `logs/` directory

### Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/my-new-feature`
3. **Follow the modular architecture patterns**
4. **Add tests** for new functionality
5. **Update documentation**
6. **Submit pull request**

### Code Style

- Follow **PEP 8** Python style guide
- Use **English comments** throughout
- Add **prominent section dividers** with `═══`
- Include **comprehensive docstrings**
- Add **type hints** for all functions

---

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

**FinAgent Memory System v2.0.0** - Modular Architecture  
*Built with ❤️ for intelligent financial agent orchestration*
