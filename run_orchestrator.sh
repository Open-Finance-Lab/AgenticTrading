#!/bin/bash

# 获取脚本所在目录（项目根目录）
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
AGENTS_DIR="$PROJECT_ROOT/FinAgents"
POOLS_DIR="$AGENTS_DIR/agent_pools"

# 设置 PYTHONPATH
# 包含各级目录，确保 Python 能找到各个 agent 模块
# 这里的 POOLS_DIR/alpha_agent_pool 路径会导致导入本地的 qlib 目录而非系统安装的 qlib 包
# 所以我们将其放到最后，或者如果不需要本地 qlib 覆盖，可以直接移除
export PYTHONPATH="$PROJECT_ROOT:$AGENTS_DIR:$POOLS_DIR:$POOLS_DIR/alpha_agent_demo:$POOLS_DIR/risk_agent_demo:$POOLS_DIR/portfolio_agent_demo:$POOLS_DIR/execution_agent_demo/execution_agent_demo:$POOLS_DIR/backtest_agent:$PYTHONPATH"

# 设置 API Key (如果本地没有设置，可以在这里临时导出，或者依赖 .env 文件)
# export OPENAI_API_KEY="your_key_here"
# export ALPACA_API_KEY="your_key_here"
# export ALPACA_SECRET_KEY="your_key_here"

echo "🚀 Starting Orchestrator Demo..."
echo "📂 Project Root: $PROJECT_ROOT"
echo "🐍 PYTHONPATH configured."

# 运行 Orchestrator
/Users/lijifeng/miniforge3/envs/agent/bin/python3 "$AGENTS_DIR/orchestrator_demo/orchestrator.py"

