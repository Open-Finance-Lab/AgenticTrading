#!/bin/bash
# Terminate all FinAgent Pool servers

echo "🛑 Terminating FinAgent Ecosystem"
echo "================================="

# Retrieve absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

echo "📁 Project root directory: ${PROJECT_ROOT}"

# Function to terminate agent pool services
stop_agent_pool() {
    local name=$1
    local pid_file="${PROJECT_ROOT}/logs/${name}.pid"
    
    if [ -f "${pid_file}" ]; then
        local pid=$(cat ${pid_file})
        if kill -0 ${pid} 2>/dev/null; then
            echo "🛑 Terminating ${name} (PID: ${pid})..."
            kill ${pid}
            
            # Wait for process termination
            for i in {1..10}; do
                if ! kill -0 ${pid} 2>/dev/null; then
                    echo "✅ ${name} has been terminated"
                    break
                fi
                sleep 1
            done
            
            # Force termination if process is still running
            if kill -0 ${pid} 2>/dev/null; then
                echo "⚠️ Force terminating ${name}..."
                kill -9 ${pid}
                echo "✅ ${name} has been force terminated"
            fi
        else
            echo "⚠️ ${name} process does not exist"
        fi
        
        # Remove PID file
        rm -f ${pid_file}
    else
        echo "⚠️ ${name} PID file does not exist"
    fi
}

# Terminate all agent pools
stop_agent_pool "data_agent_pool"
stop_agent_pool "alpha_agent_pool"
stop_agent_pool "portfolio_agent_pool"
stop_agent_pool "transaction_cost_agent_pool"
stop_agent_pool "risk_agent_pool"
# stop_agent_pool "memory_agent"  # Temporarily skip memory agent

echo ""
echo "🧹 Cleaning temporary files..."
rm -f logs/*.pid

echo ""
echo "✅ All FinAgent services have been terminated"
