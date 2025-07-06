#!/bin/bash
# Initialize all FinAgent Pool servers for natural language backtesting

echo "🚀 Initializing FinAgent Ecosystem - Natural Language Backtesting"
echo "=================================================================="

# Retrieve absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

echo "📁 Project root directory: ${PROJECT_ROOT}"

# Verify Python environment availability
if ! command -v python &> /dev/null; then
    echo "❌ Python not found, please install Python 3.8+"
    exit 1
fi

# Configure PYTHONPATH environment variable
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
echo "🐍 PYTHONPATH configured as: ${PYTHONPATH}"

# Pre-startup validation function
pre_check() {
    echo "🔍 Executing pre-startup validation checks..."
    
    # Validate Python module importability
    cd "${PROJECT_ROOT}"
    
    echo "   Validating core modules..."
    if ! python -c "from FinAgents.agent_pools.data_agent_pool import core" 2>/dev/null; then
        echo "❌ Failed to import data_agent_pool.core"
        return 1
    fi
    
    if ! python -c "from FinAgents.agent_pools.alpha_agent_pool import core" 2>/dev/null; then
        echo "❌ Failed to import alpha_agent_pool.core"
        return 1
    fi
    
    # Temporarily skip memory server validation
    # if ! python -c "from FinAgents.memory.memory_server import app" 2>/dev/null; then
    #     echo "❌ Failed to import memory_server.app"
    #     return 1
    # fi
    
    echo "✅ All core module validation checks passed"
    
    # Validate required directory structure
    if [ ! -d "${PROJECT_ROOT}/FinAgents/agent_pools" ]; then
        echo "❌ agent_pools directory not found"
        return 1
    fi
    
    echo "✅ Directory structure validation completed"
    return 0
}

# Execute pre-startup validation
if ! pre_check; then
    echo "❌ Pre-startup validation failed, terminating initialization"
    exit 1
fi

echo "✅ Pre-startup validation completed, initiating service startup..."
echo ""

# Create PID file directory
mkdir -p "${PROJECT_ROOT}/logs"

# Function to initialize agent pool services
start_agent_pool() {
    local name=$1
    local port=$2
    local script=$3
    local log_file="${PROJECT_ROOT}/logs/${name}.log"
    local pid_file="${PROJECT_ROOT}/logs/${name}.pid"
    
    echo "🔧 Initializing ${name} (port ${port})..."
    
    # Check if port is already in use
    if lsof -i:${port} &> /dev/null; then
        echo "⚠️ Port ${port} is already occupied, skipping ${name}"
        return
    fi
    
    # Change to project root directory
    cd "${PROJECT_ROOT}"
    
    # Launch service
    nohup python ${script} > ${log_file} 2>&1 &
    local pid=$!
    echo ${pid} > ${pid_file}
    
    echo "✅ ${name} has been initialized (PID: ${pid})"
    sleep 3
    
    # Verify if service is still running
    if ! kill -0 ${pid} 2>/dev/null; then
        echo "❌ ${name} initialization failed, check log: ${log_file}"
        return 1
    fi
    
    echo "✅ ${name} is running normally"
}

# Initialize Data Agent Pool
echo "📊 Initializing Data Agent Pool..."
start_agent_pool "data_agent_pool" "8001" "-m FinAgents.agent_pools.data_agent_pool.core"

# Initialize Alpha Agent Pool  
echo "🧠 Initializing Alpha Agent Pool..."
start_agent_pool "alpha_agent_pool" "8081" "-m FinAgents.agent_pools.alpha_agent_pool.core"

# Initialize Portfolio Construction Agent Pool
echo "📈 Initializing Portfolio Construction Agent Pool..."
start_agent_pool "portfolio_agent_pool" "8083" "-m FinAgents.agent_pools.portfolio_construction_agent_pool.core"

# Initialize Transaction Cost Agent Pool
echo "💰 Initializing Transaction Cost Agent Pool..."
start_agent_pool "transaction_cost_agent_pool" "8085" "-m FinAgents.agent_pools.transaction_cost_agent_pool.core"

# Initialize Risk Management Agent Pool
echo "🛡️ Initializing Risk Management Agent Pool..."
start_agent_pool "risk_agent_pool" "8084" "-m FinAgents.agent_pools.risk_agent_pool.core"

# Temporarily skip Memory Agent initialization
echo "🧠 Skipping Memory Agent initialization (temporarily not required)..."
# start_memory_agent() {
#     local name="memory_agent"
#     local port="8010"
#     local log_file="${PROJECT_ROOT}/logs/${name}.log"
#     local pid_file="${PROJECT_ROOT}/logs/${name}.pid"
#     
#     echo "🔧 Initializing ${name} (port ${port})..."
#     
#     # Check if port is already in use
#     if lsof -i:${port} &> /dev/null; then
#         echo "⚠️ Port ${port} is already occupied, skipping ${name}"
#         return
#     fi
#     
#     # Change to project root directory
#     cd "${PROJECT_ROOT}"
#     
#     # Launch memory service
#     nohup python -c "
# import sys
# import os
# sys.path.insert(0, '${PROJECT_ROOT}')
# sys.path.insert(0, '${PROJECT_ROOT}/FinAgents/memory')
# os.chdir('${PROJECT_ROOT}/FinAgents/memory')
# 
# import uvicorn
# from memory_server import app
# uvicorn.run(app, host='0.0.0.0', port=${port})
# " > ${log_file} 2>&1 &
#     local pid=$!
#     echo ${pid} > ${pid_file}
#     
#     echo "✅ ${name} has been initialized (PID: ${pid})"
#     sleep 3
#     
#     # Verify if service is still running
#     if ! kill -0 ${pid} 2>/dev/null; then
#         echo "❌ ${name} initialization failed, check log: ${log_file}"
#         return 1
#     fi
#     
#     echo "✅ ${name} is running normally"
# }
# 
# start_memory_agent

echo ""
echo "⏳ Waiting for all services to complete initialization..."
sleep 5

echo ""
echo "🔍 Checking service status..."

# Function to check service status
check_service() {
    local name=$1
    local port=$2
    local pid_file="${PROJECT_ROOT}/logs/${name}.pid"
    
    if [ -f "${pid_file}" ]; then
        local pid=$(cat ${pid_file})
        if kill -0 ${pid} 2>/dev/null; then
            if curl -s --connect-timeout 5 http://localhost:${port}/health &> /dev/null || curl -s --connect-timeout 5 http://localhost:${port}/ &> /dev/null; then
                echo "✅ ${name}: Running normally (PID: ${pid}, Port: ${port})"
            else
                echo "⚠️ ${name}: Process running but service unavailable (PID: ${pid}, Port: ${port})"
            fi
        else
            echo "❌ ${name}: Process terminated"
        fi
    else
        echo "❌ ${name}: Not initialized"
    fi
}

check_service "data_agent_pool" "8001"
check_service "alpha_agent_pool" "8081"
check_service "portfolio_agent_pool" "8083"
check_service "transaction_cost_agent_pool" "8085"
check_service "risk_agent_pool" "8084"
# check_service "memory_agent" "8010"  # Temporarily skip memory agent verification

echo ""
echo "🎯 FinAgent Ecosystem Status Summary:"
echo "   • Data Agent Pool: http://localhost:8001"
echo "   • Alpha Agent Pool: http://localhost:8081"
echo "   • Portfolio Agent Pool: http://localhost:8083"
echo "   • Transaction Cost Agent Pool: http://localhost:8085"
echo "   • Risk Management Agent Pool: http://localhost:8084"
echo "   • Memory Agent: Temporarily skipped initialization"

echo ""
echo "🚀 Ready to execute natural language backtesting:"
echo "   python tests/test_natural_language_backtest.py"
echo "   python tests/test_chatbot_client.py"

echo ""
echo "🛑 To terminate all services:"
echo "   ./stop_agent_pools.sh"

echo ""
echo "📝 To view service logs:"
echo "   tail -f ${PROJECT_ROOT}/logs/<service_name>.log"

echo ""
echo "🔧 If service initialization fails, please verify:"
echo "   1. Port availability: lsof -i:<port>"
echo "   2. Python dependencies installation: pip install -r requirements.txt"
echo "   3. Detailed error logs: cat ${PROJECT_ROOT}/logs/<service_name>.log"
