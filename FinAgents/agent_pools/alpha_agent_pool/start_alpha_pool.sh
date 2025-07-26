#!/bin/bash

# Alpha Agent Pool Startup Script
# ================================
# Simplified script to start the Alpha Agent Pool MCP Server

set -e

# Color definitions
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Configuration
ALPHA_POOL_PORT=8081
PROJECT_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PROJECT_ROOT="/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration"
ALPHA_POOL_DIR="${PROJECT_ROOT}/FinAgents/agent_pools/alpha_agent_pool"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/alpha_agent_pool.log"
PID_FILE="${LOG_DIR}/alpha_agent_pool.pid"

# Arrays to track child processes and ports
declare -a CHILD_PIDS=()
declare -a AGENT_PORTS=()

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

# Function to check if a port is available
check_port() {
    local port=$1
    if nc -z 127.0.0.1 $port 2>/dev/null; then
        return 1  # Port is occupied
    else
        return 0  # Port is available
    fi
}

# Function to find an available port starting from a given port
find_available_port() {
    local start_port=$1
    local max_attempts=${2:-10}
    
    for ((i=0; i<max_attempts; i++)); do
        local port=$((start_port + i))
        if check_port $port; then
            echo $port
            return 0
        fi
    done
    return 1  # No available port found
}

# Function to add PID to tracking array
track_process() {
    local pid=$1
    local description=$2
    CHILD_PIDS+=("$pid")
    echo "PID:$pid:$description:$(date)" >> "$PID_FILE"
    print_info "Tracking process $pid ($description)"
}

# Function to add port to tracking array
track_port() {
    local port=$1
    AGENT_PORTS+=("$port")
    echo "PORT:$port:$(date)" >> "$PID_FILE"
    print_info "Monitoring port $port"
}

# Comprehensive cleanup function
cleanup() {
    print_status "🧹 Initiating comprehensive cleanup of all agent processes..."
    
    # Write cleanup start to PID file
    echo "CLEANUP_STARTED:$(date)" >> "$PID_FILE"
    
    # Kill tracked child processes first
    if [ ${#CHILD_PIDS[@]} -gt 0 ]; then
        print_status "Terminating tracked child processes..."
        for pid in "${CHILD_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                print_info "Stopping process $pid..."
                kill -TERM "$pid" 2>/dev/null || true
                # Wait up to 5 seconds for graceful shutdown
                for i in {1..5}; do
                    if ! kill -0 "$pid" 2>/dev/null; then
                        print_success "Process $pid terminated gracefully"
                        break
                    fi
                    sleep 1
                done
                # Force kill if still running
                if kill -0 "$pid" 2>/dev/null; then
                    print_warning "Force killing process $pid"
                    kill -KILL "$pid" 2>/dev/null || true
                fi
            fi
        done
    fi
    
    # Kill processes by port
    if [ ${#AGENT_PORTS[@]} -gt 0 ]; then
        print_status "Cleaning up processes on tracked ports..."
        for port in "${AGENT_PORTS[@]}"; do
            local port_pids
            port_pids=$(lsof -ti:$port 2>/dev/null || true)
            if [ -n "$port_pids" ]; then
                print_info "Stopping processes on port $port: $port_pids"
                echo "$port_pids" | xargs kill -TERM 2>/dev/null || true
                sleep 2
                # Force kill if still running
                port_pids=$(lsof -ti:$port 2>/dev/null || true)
                if [ -n "$port_pids" ]; then
                    print_warning "Force killing processes on port $port"
                    echo "$port_pids" | xargs kill -KILL 2>/dev/null || true
                fi
            fi
        done
    fi
    
    # Kill any remaining Python processes related to alpha agent pool
    print_status "Cleaning up remaining Alpha Agent Pool processes..."
    
    # More comprehensive process cleanup
    local patterns=("alpha_agent_pool" "momentum_agent" "autonomous_agent")
    for pattern in "${patterns[@]}"; do
        local pattern_pids
        pattern_pids=$(pgrep -f "$pattern" 2>/dev/null || true)
        if [ -n "$pattern_pids" ]; then
            print_info "Stopping $pattern processes: $pattern_pids"
            echo "$pattern_pids" | xargs kill -TERM 2>/dev/null || true
            sleep 2
            # Force kill if still running
            pattern_pids=$(pgrep -f "$pattern" 2>/dev/null || true)
            if [ -n "$pattern_pids" ]; then
                print_warning "Force killing $pattern processes"
                echo "$pattern_pids" | xargs kill -KILL 2>/dev/null || true
            fi
        fi
    done
    
    # Clean up PID file
    if [ -f "$PID_FILE" ]; then
        print_info "Removing PID file: $PID_FILE"
        rm -f "$PID_FILE"
    fi
    
    print_success "🧹 Comprehensive cleanup completed - all agent processes terminated"
    exit 0
}

# Set up comprehensive signal handlers for graceful shutdown
trap cleanup SIGINT SIGTERM SIGQUIT EXIT

# Header display
echo -e "${PURPLE}"
echo "══════════════════════════════════════════════════════════════════════"
echo "              Alpha Agent Pool MCP Server Startup"
echo "              Enhanced Process Management & Robust Cleanup"
echo "══════════════════════════════════════════════════════════════════════"
echo -e "${NC}"

print_status "🚀 Initializing Alpha Agent Pool with Enhanced Process Tracking..."

# Check Python environment
print_status "🐍 Checking Python environment..."
if ! command -v python &> /dev/null; then
    print_error "Python is not installed or not in PATH"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1)
print_success "Python environment: $PYTHON_VERSION"

# Check required ports and track them
print_status "🔍 Checking port availability..."

# Write startup info to PID file
echo "STARTUP:$(date):$$" > "$PID_FILE"

# Check core server port (8081)
CORE_PORT=8081
if ! check_port $CORE_PORT; then
    print_warning "Core server port $CORE_PORT is occupied"
    ALTERNATIVE_CORE_PORT=$(find_available_port $((CORE_PORT + 1)))
    if [ $? -eq 0 ]; then
        print_info "Using alternative core port: $ALTERNATIVE_CORE_PORT"
        CORE_PORT=$ALTERNATIVE_CORE_PORT
    else
        print_error "No available ports found for core server"
        exit 1
    fi
else
    print_success "Core server port $CORE_PORT is available"
fi

# Track the core port
track_port $CORE_PORT

# Check momentum agent port (5051)
MOMENTUM_PORT=5051
if ! check_port $MOMENTUM_PORT; then
    print_warning "Momentum agent port $MOMENTUM_PORT is occupied"
    ALTERNATIVE_MOMENTUM_PORT=$(find_available_port $((MOMENTUM_PORT + 1)))
    if [ $? -eq 0 ]; then
        print_info "System will auto-assign momentum agent to port: $ALTERNATIVE_MOMENTUM_PORT"
        track_port $ALTERNATIVE_MOMENTUM_PORT
    else
        print_warning "Port conflict will be handled automatically by the system"
    fi
else
    print_success "Momentum agent port $MOMENTUM_PORT is available"
    track_port $MOMENTUM_PORT
fi

# Check autonomous agent port (5052)
AUTONOMOUS_PORT=5052
if ! check_port $AUTONOMOUS_PORT; then
    print_warning "Autonomous agent port $AUTONOMOUS_PORT is occupied"
    ALTERNATIVE_AUTONOMOUS_PORT=$(find_available_port $((AUTONOMOUS_PORT + 1)))
    if [ $? -eq 0 ]; then
        print_info "System will auto-assign autonomous agent to port: $ALTERNATIVE_AUTONOMOUS_PORT"
        track_port $ALTERNATIVE_AUTONOMOUS_PORT
    else
        print_warning "Port conflict will be handled automatically by the system"
    fi
else
    print_success "Autonomous agent port $AUTONOMOUS_PORT is available"
    track_port $AUTONOMOUS_PORT
fi

# Change to the project directory
cd "$PROJECT_ROOT"

print_status "📁 Working directory: $(pwd)"
print_success "Python path configured"

# Status display
echo -e "${PURPLE}"
echo "════════════════════════════════════════════════════════════════════"
echo "🌐 Starting Alpha Agent Pool MCP Server on port $CORE_PORT"
echo "📊 Alpha Strategy Research Framework: Ready for factor discovery"
echo "🔧 Port Conflict Detection: ENABLED"
echo "🔗 Enhanced A2A Memory Bridge: INITIALIZED"
echo "🤖 LLM Integration: ENABLED"
echo "🛡️  Robust Process Management: ENABLED"
echo "🔥 System Status: OPERATIONAL"
echo "════════════════════════════════════════════════════════════════════"
echo "Press Ctrl+C to stop the server and cleanup all processes"
echo "════════════════════════════════════════════════════════════════════"
echo -e "${NC}"

# Start the Alpha Agent Pool MCP Server
print_status "🎯 Launching Alpha Agent Pool MCP Server..."

# Start the Python server in background and capture its PID
python -c "
import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path('$PROJECT_ROOT')
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler('$LOG_FILE'),
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    try:
        from FinAgents.agent_pools.alpha_agent_pool.core import AlphaAgentPoolMCPServer
        
        print('🔗 Creating Alpha Agent Pool MCP Server instance...')
        
        # Create server instance with configured port
        server = AlphaAgentPoolMCPServer(port=$CORE_PORT)
        
        print('✅ Alpha Agent Pool MCP Server created successfully')
        print('📡 Starting server (this will block until stopped)...')
        
        # Start the server using synchronous method
        server.start()
        
    except KeyboardInterrupt:
        print('\n🛑 Shutdown signal received. Stopping Alpha Agent Pool...')
    except Exception as e:
        print(f'❌ Failed to start Alpha Agent Pool: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        print('🏁 Alpha Agent Pool shutdown completed')

if __name__ == '__main__':
    main()
" &

# Capture the PID of the Python process
PYTHON_PID=$!
track_process $PYTHON_PID "Alpha Agent Pool MCP Server"

print_success "🚀 Alpha Agent Pool MCP Server started with PID: $PYTHON_PID"
print_info "📝 Process tracking file: $PID_FILE"
print_info "📋 Tracked processes: ${#CHILD_PIDS[@]}"
print_info "🔌 Monitored ports: ${#AGENT_PORTS[@]}"


# Monitor the Python process and handle its completion
print_status "🔍 Monitoring Alpha Agent Pool MCP Server process..."

# Wait for the background process to complete
wait $PYTHON_PID
PYTHON_EXIT_CODE=$?

# If we reach here, the server has stopped
if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    print_status "🏁 Alpha Agent Pool stopped gracefully"
else
    print_error "🏁 Alpha Agent Pool stopped with exit code: $PYTHON_EXIT_CODE"
fi

# Cleanup will be called automatically by the EXIT trap
