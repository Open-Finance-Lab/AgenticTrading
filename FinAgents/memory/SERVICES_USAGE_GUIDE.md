# FinAgent Memory Services Usage Guide

## 📁 文件结构

经过清理后，项目中只保留了以下Shell脚本：

- **`start_memory_services.sh`** - 主要的服务启动脚本
- **`test_services.sh`** - 服务测试和验证脚本
- **`setup_integration.sh`** - 系统集成和安装脚本（在scripts目录）
- **`start_agent_pools.sh` / `stop_agent_pools.sh`** - Agent pools测试脚本（在tests目录）

## 🚀 使用方法

### 基本启动命令

```bash
# 切换到memory目录
cd /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/memory

# 启动Memory + MCP + A2A服务（推荐用于基本功能）
./start_memory_services.sh memory
# 或者
./start_memory_services.sh core

# 仅启动LLM研究服务
./start_memory_services.sh llm

# 启动所有服务（Memory + MCP + A2A + LLM）
./start_memory_services.sh all
```

### 📊 服务端口分配

- **Memory Server**: `http://localhost:8000`
- **MCP Protocol**: `http://localhost:8001`
- **A2A Protocol**: `http://localhost:8002`
- **Health Check**: `http://localhost:8000/health`

### 📝 日志文件位置

所有服务的日志都保存在 `/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/memory/logs/` 目录：

- `memory_server.log` - Memory服务器日志
- `mcp_server.log` - MCP协议服务器日志
- `a2a_server.log` - A2A协议服务器日志
- `llm_research_service.log` - LLM研究服务日志

### 🔍 日志查看命令

```bash
# 实时查看Memory服务器日志
tail -f /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/memory/logs/memory_server.log

# 实时查看MCP服务器日志
tail -f /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/memory/logs/mcp_server.log

# 实时查看A2A服务器日志
tail -f /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/memory/logs/a2a_server.log

# 实时查看LLM研究服务日志
tail -f /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/memory/logs/llm_research_service.log
```

### 🧪 测试和验证

运行测试脚本验证环境配置：
```bash
cd /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/memory
./test_services.sh
```

测试LLM研究服务：
```bash
cd /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/memory
./validate_fix.sh
```

手动测试LLM命令：
```bash
cd /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/memory
conda activate agent
PYTHONPATH=/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration:$PYTHONPATH python -c "from FinAgents.memory.llm_research_service import llm_research_service; import asyncio; asyncio.run(llm_research_service.analyze_memory_patterns([]))"
```

### 🏥 健康检查

```bash
# 检查Memory服务器状态
curl http://localhost:8000/health

# 检查端口使用情况
lsof -i :8000
lsof -i :8001
lsof -i :8002
```

### ⏹️ 停止服务

使用 `Ctrl+C` 停止所有服务，脚本会自动清理所有进程。

## 🔧 系统要求

- Conda环境名为 'agent'
- Neo4j数据库运行在 `bolt://localhost:7687`，用户名: neo4j，密码: finagent123
- OpenAI API密钥在 `.env` 文件中配置（仅LLM服务需要）

## 📋 服务架构

### Memory模式（memory/core）
```
🏗️ Memory Layer: Database + Memory Server + MCP Protocol + A2A Protocol
```

### LLM模式
```
🧠 Research Layer: LLM-powered Analysis + Insights
```

### All模式
```
🏗️ Memory Layer: Database + Memory Server + MCP Protocol + A2A Protocol
🧠 Research Layer: LLM-powered Analysis + Insights
🔗 Integration: Memory services provide data, LLM services provide insights
```

## 🚨 常见问题排除

1. **端口被占用**: 使用 `lsof -i :端口号` 查找占用进程，使用 `kill PID` 停止
2. **Conda环境问题**: 确保agent环境存在且已安装所需依赖
3. **数据库连接失败**: 确保Neo4j服务正在运行
4. **LLM服务启动失败**: 检查OpenAI API密钥配置

## 📞 获取帮助

查看脚本帮助信息：
```bash
./start_memory_services.sh invalid
```
