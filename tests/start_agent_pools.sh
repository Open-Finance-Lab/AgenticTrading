#!/bin/bash
# Initialize all FinAgent Pool servers for natural language backtesting

echo "🚀 Initializing FinAgent Ecosystem - Natural Language Backtesting"
echo "=================================================================="

# Retrieve absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

echo "📁 Project root directory: ${PROJECT_ROOT}"
echo "📁 Script directory: ${SCRIPT_DIR}"

# Check if we're running from the correct directory
if [[ "$(basename "$(pwd)")" != "FinAgent-Orchestration" ]]; then
    echo "⚠️  Warning: Script should be run from project root directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected to be in: ${PROJECT_ROOT}"
    echo ""
    echo "💡 Please run from project root:"
    echo "   cd ${PROJECT_ROOT}"
    echo "   bash tests/start_agent_pools.sh"
    echo ""
    read -p "Continue anyway? (y/N): " continue_choice
    if [[ ! "$continue_choice" =~ ^[Yy]$ ]]; then
        echo "❌ Exiting. Please run from project root directory."
        exit 1
    fi
fi

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
    if ! conda run -n agent python -c "from FinAgents.agent_pools.data_agent_pool import core" 2>/dev/null; then
        echo "❌ Failed to import data_agent_pool.core"
        return 1
    fi
    
    if ! conda run -n agent python -c "from FinAgents.agent_pools.alpha_agent_pool import core" 2>/dev/null; then
        echo "❌ Failed to import alpha_agent_pool.core"
        return 1
    fi
    
    # Check if memory agent is already running instead of validating import
    echo "   Checking memory agent availability..."
    if curl -s --connect-timeout 5 http://localhost:8010/health &> /dev/null || curl -s --connect-timeout 5 http://localhost:8010/ &> /dev/null; then
        echo "✅ Memory agent is already running on port 8010"
    else
        echo "⚠️  Memory agent not detected on port 8010, but continuing anyway"
        echo "   Please ensure memory agent is running before starting agent pools"
    fi
    
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
    
    # Launch service using conda environment
    nohup conda run -n agent python ${script} > ${log_file} 2>&1 &
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

# Initialize Data Agent Pool (using different port since memory is on 8010)
echo "📊 Initializing Data Agent Pool..."
start_agent_pool "data_agent_pool" "8011" "-m FinAgents.agent_pools.data_agent_pool.core"

# Function to start MCP server  
start_mcp_server() {
    local name="mcp_server"
    local port="8001"
    local log_file="${PROJECT_ROOT}/logs/${name}.log"
    local pid_file="${PROJECT_ROOT}/logs/${name}.pid"
    
    echo "� Initializing ${name} (port ${port})..."
    
    # Check if port is already in use
    if lsof -i:${port} &> /dev/null; then
        echo "⚠️ Port ${port} is already occupied, skipping ${name}"
        return
    fi
    
    # Change to FinAgents/memory directory
    cd "${PROJECT_ROOT}/FinAgents/memory"
    
    # Launch MCP server using conda environment
    nohup conda run -n agent python mcp_server.py > ${log_file} 2>&1 &
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

# Function to start A2A server
start_a2a_server() {
    local name="a2a_server"
    local port="8002"
    local log_file="${PROJECT_ROOT}/logs/${name}.log"
    local pid_file="${PROJECT_ROOT}/logs/${name}.pid"
    
    echo "🔧 Initializing ${name} (port ${port})..."
    
    # Check if port is already in use
    if lsof -i:${port} &> /dev/null; then
        echo "⚠️ Port ${port} is already occupied, skipping ${name}"
        return
    fi
    
    # Change to FinAgents/memory directory
    cd "${PROJECT_ROOT}/FinAgents/memory"
    
    # Launch A2A server using conda environment
    nohup conda run -n agent python a2a_server.py > ${log_file} 2>&1 &
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

echo ""
echo "🧠 Memory Services Status Check..."

# Check if memory services are already running
echo "🔍 Checking if memory agent is already running..."
if curl -s --connect-timeout 5 http://localhost:8010/health &> /dev/null || curl -s --connect-timeout 5 http://localhost:8010/ &> /dev/null; then
    echo "✅ Memory agent is already running on port 8010 - skipping memory service initialization"
    MEMORY_RUNNING=true
else
    echo "⚠️  Memory agent not detected - starting memory services..."
    MEMORY_RUNNING=false
    
    # Use existing memory services launcher
    echo "🔧 Starting Memory Services using dedicated launcher..."
    cd "${PROJECT_ROOT}/FinAgents/memory"
    nohup bash start_memory_services.sh memory > "${PROJECT_ROOT}/logs/memory_services.log" 2>&1 &
    MEMORY_PID=$!
    echo ${MEMORY_PID} > "${PROJECT_ROOT}/logs/memory_services.pid"
    echo "✅ Memory Services launcher started (PID: ${MEMORY_PID})"
    
    # Wait for memory services to start
    echo "⏳ Waiting for memory services to initialize..."
    sleep 10
fi

echo ""
echo "📊 Initializing Agent Pool Services..."

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

# Remove old temporary memory initialization code
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

check_service "memory_services" "8010"
check_service "data_agent_pool" "8011"
check_service "alpha_agent_pool" "8081"
check_service "portfolio_agent_pool" "8083"
check_service "transaction_cost_agent_pool" "8085"
check_service "risk_agent_pool" "8084"

echo ""
echo "🎯 FinAgent Ecosystem Status Summary:"
echo "   • Memory Services: http://localhost:8010 (+ MCP on 8001, A2A on 8002)"
echo "   • Data Agent Pool: http://localhost:8011"
echo "   • Alpha Agent Pool: http://localhost:8081 (Enhanced with Strategy Research Framework)"
echo "   • Portfolio Agent Pool: http://localhost:8083"
echo "   • Transaction Cost Agent Pool: http://localhost:8085"
echo "   • Risk Management Agent Pool: http://localhost:8084"

echo ""
echo "🚀 Ready to execute natural language backtesting:"
echo "   python tests/test_simple_llm_backtest.py"

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
