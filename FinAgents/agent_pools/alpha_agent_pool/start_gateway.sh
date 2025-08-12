#!/bin/bash

# Alpha Pool Gateway Startup Script
# This script starts the Alpha Pool Gateway as a MCP server for external orchestration
# while maintaining MCP client connections to internal alpha agents

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATEWAY_PORT=${1:-8082}  # Default port 8082, can be overridden
LOG_LEVEL=${2:-INFO}     # Default log level INFO
ENABLE_INTERNAL_POOL=${3:-true}  # Enable internal alpha pool by default

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Function to check if port is available
check_port_available() {
    local port=$1
    local host=${2:-127.0.0.1}
    
    if command -v nc >/dev/null 2>&1; then
        # Use netcat if available
        ! nc -z "$host" "$port" >/dev/null 2>&1
    else
        # Fallback: use Python to check port
        python3 -c "
import socket
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex(('$host', $port))
    sock.close()
    exit(0 if result != 0 else 1)
except:
    exit(1)
"
    fi
}

# Function to find available port
find_available_port() {
    local start_port=$1
    local max_attempts=${2:-10}
    
    for ((port = start_port; port < start_port + max_attempts; port++)); do
        if check_port_available "$port"; then
            echo "$port"
            return 0
        fi
    done
    
    return 1
}

# Function to check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Python 3.8+
    if ! command -v python3 >/dev/null 2>&1; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    local python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    log_info "Python version: $python_version"
    
    # Check required Python packages
    local required_packages=(
        "mcp"
        "asyncio"
        "fastapi"
        "pydantic"
    )
    
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" >/dev/null 2>&1; then
            log_warn "Python package '$package' not found, attempting to install..."
            pip3 install "$package" || {
                log_error "Failed to install $package"
                exit 1
            }
        fi
    done
    
    log_info "Dependencies check completed"
}

# Function to setup environment
setup_environment() {
    log_info "Setting up environment..."
    
    # Set Python path to include project root
    local project_root
    project_root=$(cd "$SCRIPT_DIR/../../.." && pwd)
    export PYTHONPATH="$project_root:$PYTHONPATH"
    
    # Set log level
    export LOG_LEVEL="$LOG_LEVEL"
    
    # Create logs directory
    mkdir -p "$SCRIPT_DIR/logs"
    
    log_info "Environment setup completed"
    log_debug "Project root: $project_root"
    log_debug "Python path: $PYTHONPATH"
}

# Function to start gateway
start_gateway() {
    log_info "Starting Alpha Pool Gateway..."
    
    # Check if gateway port is available
    if ! check_port_available "$GATEWAY_PORT"; then
        log_warn "Port $GATEWAY_PORT is occupied, finding alternative..."
        local available_port
        available_port=$(find_available_port $((GATEWAY_PORT + 1)))
        
        if [[ -n "$available_port" ]]; then
            GATEWAY_PORT="$available_port"
            log_info "Using alternative port: $GATEWAY_PORT"
        else
            log_error "No available ports found starting from $((GATEWAY_PORT + 1))"
            exit 1
        fi
    fi
    
    # Create log file with timestamp
    local log_file="$SCRIPT_DIR/logs/gateway_$(date '+%Y%m%d_%H%M%S').log"
    
    log_info "Gateway will run on port: $GATEWAY_PORT"
    log_info "Log file: $log_file"
    log_info "Enable internal pool: $ENABLE_INTERNAL_POOL"
    
    # Start the gateway
    cd "$SCRIPT_DIR"
    
    # Set environment variables for the Python script
    export GATEWAY_HOST="0.0.0.0"
    export GATEWAY_PORT="$GATEWAY_PORT"
    export ENABLE_INTERNAL_POOL="$ENABLE_INTERNAL_POOL"
    
    # Start gateway with proper error handling
    {
        log_info "🚀 Launching Alpha Pool Gateway..."
        python3 alpha_pool_gateway.py 2>&1 | tee "$log_file"
    } || {
        log_error "Gateway startup failed"
        log_error "Check log file for details: $log_file"
        exit 1
    }
}

# Function to cleanup on exit
cleanup() {
    log_info "Cleaning up..."
    
    # Kill any remaining processes (optional)
    # pkill -f "alpha_pool_gateway.py" >/dev/null 2>&1 || true
    
    log_info "Cleanup completed"
}

# Function to display help
show_help() {
    cat << EOF
Alpha Pool Gateway Startup Script

Usage: $0 [PORT] [LOG_LEVEL] [ENABLE_INTERNAL_POOL]

Arguments:
    PORT                  Gateway port (default: 8082)
    LOG_LEVEL            Log level: DEBUG, INFO, WARN, ERROR (default: INFO)
    ENABLE_INTERNAL_POOL Enable internal alpha pool server (default: true)

Examples:
    $0                    # Start with default settings
    $0 8083              # Start on port 8083
    $0 8083 DEBUG        # Start on port 8083 with DEBUG logging
    $0 8083 INFO false   # Start on port 8083 without internal pool

Environment Variables:
    GATEWAY_HOST         Gateway host (default: 0.0.0.0)
    GATEWAY_PORT         Gateway port (overrides argument)
    LOG_LEVEL           Log level (overrides argument)
    ENABLE_INTERNAL_POOL Enable internal pool (overrides argument)

The gateway serves as:
- MCP Server for external orchestration systems (port $GATEWAY_PORT)
- MCP Client for internal alpha agents coordination
- Unified interface for alpha strategy generation and management

Internal agents expected to be running:
- Momentum Agent (http://127.0.0.1:5051/mcp)
- Autonomous Agent (http://127.0.0.1:5052/mcp)
- Alpha Pool Server (http://127.0.0.1:8081/mcp)

EOF
}

# Function to validate arguments
validate_arguments() {
    # Validate port
    if ! [[ "$GATEWAY_PORT" =~ ^[0-9]+$ ]] || [ "$GATEWAY_PORT" -lt 1024 ] || [ "$GATEWAY_PORT" -gt 65535 ]; then
        log_error "Invalid port number: $GATEWAY_PORT (must be between 1024-65535)"
        exit 1
    fi
    
    # Validate log level
    case "$LOG_LEVEL" in
        DEBUG|INFO|WARN|WARNING|ERROR)
            ;;
        *)
            log_error "Invalid log level: $LOG_LEVEL (must be DEBUG, INFO, WARN, WARNING, or ERROR)"
            exit 1
            ;;
    esac
    
    # Validate enable internal pool
    case "$ENABLE_INTERNAL_POOL" in
        true|false|TRUE|FALSE|1|0|yes|no|YES|NO)
            ;;
        *)
            log_error "Invalid value for ENABLE_INTERNAL_POOL: $ENABLE_INTERNAL_POOL (must be true/false)"
            exit 1
            ;;
    esac
}

# Main execution
main() {
    # Handle help request
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        show_help
        exit 0
    fi
    
    # Set trap for cleanup
    trap cleanup EXIT INT TERM
    
    log_info "=== Alpha Pool Gateway Startup ==="
    log_info "Gateway Port: $GATEWAY_PORT"
    log_info "Log Level: $LOG_LEVEL" 
    log_info "Enable Internal Pool: $ENABLE_INTERNAL_POOL"
    
    # Validate arguments
    validate_arguments
    
    # Check dependencies
    check_dependencies
    
    # Setup environment
    setup_environment
    
    # Start gateway
    start_gateway
    
    log_info "Gateway startup completed successfully"
}

# Execute main function
main "$@"
