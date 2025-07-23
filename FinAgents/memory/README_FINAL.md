# FinAgent Memory System - Complete Guide

## 🚀 Overview

The FinAgent Memory System is a comprehensive, modular architecture that provides intelligent memory management for financial agents. This system features clean separation of concerns, unified database management, and protocol-agnostic interfaces supporting multiple deployment scenarios.

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Service Configuration](#service-configuration)
- [Testing & Validation](#testing--validation)
- [Monitoring & Logs](#monitoring--logs)
- [API Documentation](#api-documentation)
- [Development Guide](#development-guide)
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

### Key Components

- **Memory Server** (Port 8000): Core memory management with FastAPI HTTP endpoints
- **MCP Server** (Port 8001): Model Context Protocol for LLM integration
- **A2A Server** (Port 8002): Agent-to-Agent communication protocol
- **LLM Research Service**: Optional AI-powered analysis and insights

### Key Benefits

- ✅ **Clean Separation**: Each server handles specific protocols
- ✅ **Unified Backend**: Shared database and interface management
- ✅ **Scalable**: Easy to add new protocols or server types
- ✅ **Configurable**: Environment-specific configurations
- ✅ **Easy Deployment**: Single launcher for all components
- ✅ **Comprehensive Logging**: Centralized log management

## 🔧 System Requirements

### Prerequisites

1. **Conda Environment**: Must have `agent` environment
   ```bash
   # Create the environment if it doesn't exist
   conda create -n agent python=3.12
   conda activate agent
   
   # Install required packages
   pip install -r /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/requirements.txt
   ```

2. **Neo4j Database**: Running on `bolt://localhost:7687`
   - Username: `neo4j`
   - Password: `finagent123`

3. **OpenAI API Key** (optional, for LLM services):
   ```bash
   # Add to .env file in project root
   echo "OPENAI_API_KEY=your_api_key_here" >> /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/.env
   ```

### Port Requirements

- Port 8000: Memory Server
- Port 8001: MCP Protocol Server
- Port 8002: A2A Protocol Server

## 🚀 Quick Start

**Always start from the memory directory:**
```bash
cd /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/memory
```

### Option 1: Core Services (Memory + MCP + A2A)
```bash
# Recommended for basic functionality
./start_memory_services.sh memory
# or
./start_memory_services.sh core
```

### Option 2: LLM Research Services Only
```bash
# AI-powered analysis only
./start_memory_services.sh llm
```

### Option 3: Full System
```bash
# All services (Memory + MCP + A2A + LLM)
./start_memory_services.sh all
```

### Option 4: Stop Services
```bash
# Use Ctrl+C in the terminal running services
# The script automatically cleans up all processes
```

## ⚙️ Service Configuration

### Service Ports

| Service | Port | Description | Health Check |
|---------|------|-------------|--------------|
| Memory Server | 8000 | Core memory management | `http://localhost:8000/health` |
| MCP Protocol | 8001 | Model Context Protocol | `http://localhost:8001/` |
| A2A Protocol | 8002 | Agent-to-Agent communication | `http://localhost:8002/health` |

### Startup Modes

| Mode | Services Started | Use Case |
|------|------------------|----------|
| `memory` / `core` | Memory + MCP + A2A | Basic agent functionality |
| `llm` | LLM Research Service only | AI analysis only |
| `all` | All services | Complete system |

### Environment Variables

The system automatically configures environments. Key variables:
- `PYTHONPATH`: Set to project root
- `OPENAI_API_KEY`: Required for LLM services
- `NEO4J_URI`: Database connection (default: bolt://localhost:7687)

## 🧪 Testing & Validation

### 1. System Validation
```bash
cd /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/memory

# Test environment configuration
./test_services.sh
```

### 2. Service Integration Tests
```bash
# Test MCP and A2A services
python tests/test_services_runner.py

# Test specific services
python tests/test_services_runner.py --mcp-only
python tests/test_services_runner.py --a2a-only

# Simple connectivity tests (no dependencies)
python tests/test_services_runner.py --simple
```

### 3. Quick Tests
```bash
# Quick connectivity test
./tests/quick_test.sh

# Manual service checks
curl http://localhost:8000/health
curl http://localhost:8002/health
```

### 4. LLM Service Validation
```bash
# Validate LLM service
./validate_fix.sh

# Manual LLM test
conda activate agent
PYTHONPATH=/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration:$PYTHONPATH python -c "from FinAgents.memory.llm_research_service import llm_research_service; import asyncio; asyncio.run(llm_research_service.analyze_memory_patterns([]))"
```

### 5. A2A Integration Test
```bash
conda activate agent
python final_a2a_integration_test.py
```

## 📝 Monitoring & Logs

### Log File Locations

All logs are stored in `/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/memory/logs/`:

| Service | Log File | Description |
|---------|----------|-------------|
| Memory Server | `memory_server.log` | Core memory operations |
| MCP Server | `mcp_server.log` | MCP protocol activities |
| A2A Server | `a2a_server.log` | Agent communication |
| LLM Service | `llm_research_service.log` | AI analysis logs |

### Real-time Log Monitoring

```bash
# Memory server logs
tail -f logs/memory_server.log

# MCP server logs
tail -f logs/mcp_server.log

# A2A server logs
tail -f logs/a2a_server.log

# LLM service logs
tail -f logs/llm_research_service.log

# All logs simultaneously
tail -f logs/*.log
```

### Health Monitoring

```bash
# Check all service status
curl http://localhost:8000/health
curl http://localhost:8002/health
curl http://localhost:8002/status

# Check running processes
ps aux | grep -E "(memory_server|mcp_server|a2a_server|llm_research)"

# Check port usage
lsof -i :8000,8001,8002
```

## 🔌 API Documentation

### Memory Server (Port 8000)

**Health Check**
```bash
GET http://localhost:8000/health
```

**Memory Operations** (via MCP protocol)
- Store memory
- Retrieve memory
- Search memories
- Update memory relationships

### A2A Server (Port 8002)

**Health Check**
```bash
GET http://localhost:8002/health
```

**Status Check**
```bash
GET http://localhost:8002/status
```

**Agent Communication**
- Signal transmission
- Strategy sharing
- Real-time coordination
- Performance monitoring

### MCP Server (Port 8001)

**Protocol Endpoints**
- Tool discovery
- Tool execution
- Resource management
- Session management

## 🛠️ Development Guide

### File Structure

```
FinAgents/memory/
├── start_memory_services.sh    # Main startup script
├── test_services.sh           # Environment validation
├── validate_fix.sh           # LLM service validation
├── logs/                     # Log files directory
├── tests/                    # Test suites
│   ├── test_services_runner.py  # Comprehensive tests
│   ├── test_mcp_server.py      # MCP server tests
│   ├── test_a2a_server.py      # A2A server tests
│   └── quick_test.sh           # Quick connectivity tests
├── config/                   # Configuration files
├── memory_server.py          # Core memory server
├── mcp_server.py            # MCP protocol server
├── a2a_server.py            # A2A communication server
├── llm_research_service.py  # LLM analysis service
└── *.py                     # Other core components
```

### Adding New Services

1. Create new server file following existing patterns
2. Add port configuration to startup script
3. Create corresponding test file
4. Update documentation

### Configuration Management

The system uses YAML configuration files in `config/` directory:
- `development.yaml` - Development settings
- `testing.yaml` - Test environment
- `production.yaml` - Production settings

## 🚨 Troubleshooting

### Common Issues

**1. Port Already in Use**
```bash
# Find process using port
lsof -i :8000

# Kill process
kill <PID>
```

**2. Conda Environment Issues**
```bash
# Ensure agent environment exists
conda info --envs | grep agent

# Recreate if needed
conda create -n agent python=3.12
conda activate agent
```

**3. Database Connection Failed**
```bash
# Check Neo4j status
systemctl status neo4j  # Linux
brew services list | grep neo4j  # macOS

# Verify connection
python -c "from FinAgents.memory.configuration_manager import ConfigurationManager; cm = ConfigurationManager(); print(cm.get_database_config())"
```

**4. LLM Service Failures**
```bash
# Check OpenAI API key
grep OPENAI_API_KEY /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/.env

# Check quota/permissions in OpenAI dashboard
```

**5. Import Errors**
```bash
# Verify PYTHONPATH
echo $PYTHONPATH

# Set correct path
export PYTHONPATH=/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration:$PYTHONPATH
```

### Debug Mode

For detailed debugging, check individual log files:
```bash
# Memory server debug
tail -f logs/memory_server.log | grep -E "(ERROR|WARNING)"

# Service startup issues
./start_memory_services.sh memory 2>&1 | tee debug.log
```

### Performance Issues

```bash
# Check system resources
top -p $(pgrep -f "memory_server\|mcp_server\|a2a_server")

# Monitor memory usage
ps aux --sort=-%mem | head -10

# Check disk space for logs
du -sh logs/
```

## 📞 Support

For additional support:

1. **Check logs first**: All errors are logged with timestamps
2. **Run validation tests**: Use test scripts to identify issues
3. **Verify environment**: Ensure all prerequisites are met
4. **Check service status**: Use health endpoints to verify operation

### Useful Commands Summary

```bash
# Start services
./start_memory_services.sh all

# Test everything
./test_services.sh && python tests/test_services_runner.py

# Monitor logs
tail -f logs/*.log

# Health check
curl http://localhost:8000/health && curl http://localhost:8002/health

# Stop services (Ctrl+C in service terminal)
```

---

**🎉 Success!** Your FinAgent Memory System is ready for production use.
