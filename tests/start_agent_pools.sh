#!/bin/bash
# 启动所有FinAgent Pool服务器用于自然语言回测

echo "🚀 启动FinAgent生态系统 - 自然语言回测"
echo "================================================"

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ Python未找到，请安装Python 3.8+"
    exit 1
fi

# 创建PID文件目录
mkdir -p logs

# 函数：启动agent pool
start_agent_pool() {
    local name=$1
    local port=$2
    local script=$3
    local log_file="logs/${name}.log"
    local pid_file="logs/${name}.pid"
    
    echo "🔧 启动 ${name} (端口 ${port})..."
    
    # 检查端口是否被占用
    if lsof -i:${port} &> /dev/null; then
        echo "⚠️ 端口 ${port} 已被占用，跳过 ${name}"
        return
    fi
    
    # 启动服务
    nohup python ${script} > ${log_file} 2>&1 &
    local pid=$!
    echo ${pid} > ${pid_file}
    
    echo "✅ ${name} 已启动 (PID: ${pid})"
    sleep 2
    
    # 验证服务是否仍在运行
    if ! kill -0 ${pid} 2>/dev/null; then
        echo "❌ ${name} 启动失败，检查日志: ${log_file}"
        return 1
    fi
    
    echo "✅ ${name} 运行正常"
}

# 启动Data Agent Pool
echo "📊 启动数据代理池..."
start_agent_pool "data_agent_pool" "8001" "-m FinAgents.agent_pools.data_agent_pool.core"

# 启动Alpha Agent Pool
echo "🧠 启动Alpha代理池..."
start_agent_pool "alpha_agent_pool" "5050" "-m FinAgents.agent_pools.alpha_agent_pool.core"

# 启动Portfolio Construction Agent Pool
echo "📈 启动投资组合构建代理池..."
start_agent_pool "portfolio_agent_pool" "8002" "-m FinAgents.agent_pools.portfolio_construction_agent_pool.core"

# 启动Transaction Cost Agent Pool
echo "💰 启动交易成本代理池..."
start_agent_pool "transaction_cost_agent_pool" "6000" "-m FinAgents.agent_pools.transaction_cost_agent_pool.core"

# 启动Risk Agent Pool
echo "🛡️ 启动风险管理代理池..."
start_agent_pool "risk_agent_pool" "7001" "-m FinAgents.agent_pools.risk_agent_pool.core"

# 启动Memory Agent
echo "🧠 启动内存代理..."
start_memory_agent() {
    local name="memory_agent"
    local port="8010"
    local log_file="logs/${name}.log"
    local pid_file="logs/${name}.pid"
    
    echo "🔧 启动 ${name} (端口 ${port})..."
    
    # 检查端口是否被占用
    if lsof -i:${port} &> /dev/null; then
        echo "⚠️ 端口 ${port} 已被占用，跳过 ${name}"
        return
    fi
    
    # 启动内存服务
    cd FinAgents/memory
    nohup python -c "
import uvicorn
import sys
sys.path.append('/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration')
from FinAgents.memory.memory_server import app
uvicorn.run(app, host='0.0.0.0', port=${port})
" > ../../${log_file} 2>&1 &
    local pid=$!
    echo ${pid} > ../../${pid_file}
    cd ../..
    
    echo "✅ ${name} 已启动 (PID: ${pid})"
    sleep 2
    
    # 验证服务是否仍在运行
    if ! kill -0 ${pid} 2>/dev/null; then
        echo "❌ ${name} 启动失败，检查日志: ${log_file}"
        return 1
    fi
    
    echo "✅ ${name} 运行正常"
}

start_memory_agent

echo ""
echo "⏳ 等待所有服务启动完成..."
sleep 5

echo ""
echo "🔍 检查服务状态..."

# 检查所有服务状态
check_service() {
    local name=$1
    local port=$2
    local pid_file="logs/${name}.pid"
    
    if [ -f "${pid_file}" ]; then
        local pid=$(cat ${pid_file})
        if kill -0 ${pid} 2>/dev/null; then
            if curl -s http://localhost:${port}/health &> /dev/null; then
                echo "✅ ${name}: 运行正常 (PID: ${pid}, Port: ${port})"
            else
                echo "⚠️ ${name}: 进程运行但服务不可用 (PID: ${pid}, Port: ${port})"
            fi
        else
            echo "❌ ${name}: 进程已停止"
        fi
    else
        echo "❌ ${name}: 未启动"
    fi
}

check_service "data_agent_pool" "8001"
check_service "alpha_agent_pool" "5050"
check_service "portfolio_agent_pool" "8002"
check_service "transaction_cost_agent_pool" "6000"
check_service "risk_agent_pool" "7001"
check_service "memory_agent" "8010"

echo ""
echo "🎯 FinAgent生态系统状态总结:"
echo "   • 数据代理池: http://localhost:8001"
echo "   • Alpha代理池: http://localhost:5050"
echo "   • 投资组合代理池: http://localhost:8002"
echo "   • 交易成本代理池: http://localhost:6000"
echo "   • 风险管理代理池: http://localhost:7001"
echo "   • 内存代理: http://localhost:8010"

echo ""
echo "🚀 现在可以运行自然语言回测:"
echo "   python tests/test_natural_language_backtest.py"
echo "   python tests/test_chatbot_client.py"

echo ""
echo "🛑 停止所有服务:"
echo "   ./stop_agent_pools.sh"
