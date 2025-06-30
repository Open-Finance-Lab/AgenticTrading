#!/bin/bash
# 简化的Agent Pools启动脚本

echo "🚀 启动所有Agent Pools"
echo "====================="

# 设置Python路径
export PYTHONPATH=/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration:$PYTHONPATH

# 创建日志目录
mkdir -p logs

# 停止所有现有进程
echo "🛑 停止现有进程..."
pkill -f "agent_pool" 2>/dev/null || true
sleep 2

# 启动Data Agent Pool (端口 8001)
echo "📊 启动Data Agent Pool (端口 8001)..."
cd /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration
nohup python -m FinAgents.agent_pools.data_agent_pool.core > logs/data_agent.log 2>&1 &
DATA_PID=$!
echo "   PID: $DATA_PID"
sleep 3

# 启动Alpha Agent Pool (端口 5050)
echo "🧠 启动Alpha Agent Pool (端口 5050)..."
cd /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration
nohup python -m FinAgents.agent_pools.alpha_agent_pool.core > logs/alpha_agent.log 2>&1 &
ALPHA_PID=$!
echo "   PID: $ALPHA_PID"
sleep 3

# 启动Transaction Cost Agent Pool (端口 6000)
echo "💰 启动Transaction Cost Agent Pool (端口 6000)..."
cd /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration
nohup python -m FinAgents.agent_pools.transaction_cost_agent_pool.core > logs/transaction_cost_agent.log 2>&1 &
COST_PID=$!
echo "   PID: $COST_PID"
sleep 3

# 启动Portfolio Construction Agent Pool (端口 8002)
echo "📈 启动Portfolio Construction Agent Pool (端口 8002)..."
cd /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration
nohup python -m FinAgents.agent_pools.portfolio_construction_agent_pool.core > logs/portfolio_agent.log 2>&1 &
PORTFOLIO_PID=$!
echo "   PID: $PORTFOLIO_PID"
sleep 3

# 启动Risk Agent Pool (端口 7001)
echo "🛡️ 启动Risk Agent Pool (端口 7001)..."
cd /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration
nohup python -m FinAgents.agent_pools.risk_agent_pool.core > logs/risk_agent.log 2>&1 &
RISK_PID=$!
echo "   PID: $RISK_PID"
sleep 3

echo ""
echo "✅ 所有Agent Pools启动完成"
echo ""
echo "📋 运行状态："
echo "   📊 Data Agent Pool: PID $DATA_PID, Port 8001"
echo "   🧠 Alpha Agent Pool: PID $ALPHA_PID, Port 5050"
echo "   💰 Transaction Cost Agent Pool: PID $COST_PID, Port 6000"
echo "   📈 Portfolio Agent Pool: PID $PORTFOLIO_PID, Port 8002"
echo "   🛡️ Risk Agent Pool: PID $RISK_PID, Port 7001"
echo ""
echo "🔍 检查服务状态："

# 检查端口
for port in 8001 5050 6000 8002 7001; do
    if lsof -i:$port &> /dev/null; then
        echo "   ✅ 端口 $port: 活跃"
    else
        echo "   ❌ 端口 $port: 无响应"
    fi
done

echo ""
echo "📋 查看日志："
echo "   tail -f logs/data_agent.log"
echo "   tail -f logs/alpha_agent.log"
echo "   tail -f logs/transaction_cost_agent.log"
echo "   tail -f logs/portfolio_agent.log"
echo "   tail -f logs/risk_agent.log"
echo ""
echo "🚀 现在可以运行测试:"
echo "   python tests/test_simple_llm_backtest.py"
