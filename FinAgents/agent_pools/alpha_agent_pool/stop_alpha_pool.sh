#!/bin/bash

# Alpha Agent Pool Shutdown Script
# =================================
# Comprehensive shutdown script for Alpha Agent Pool MCP Server

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
PROJECT_ROOT="/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration"
LOG_DIR="${PROJECT_ROOT}/logs"
PID_FILE="${LOG_DIR}/alpha_agent_pool.pid"

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

# Header display
echo -e "${PURPLE}"
echo "══════════════════════════════════════════════════════════════════════"
echo "              Alpha Agent Pool MCP Server Shutdown"
echo "              Comprehensive Process Cleanup & Port Release"
echo "══════════════════════════════════════════════════════════════════════"
echo -e "${NC}"

print_status "🛑 Initiating Alpha Agent Pool shutdown sequence..."

# Function to kill processes gracefully then forcefully if needed
kill_process_gracefully() {
    local pid=$1
    local description=${2:-"process"}
    
    if ! kill -0 "$pid" 2>/dev/null; then
        print_info "Process $pid ($description) is not running"
        return 0
    fi
    
    print_info "Stopping $description (PID: $pid)..."
    
    # Send TERM signal
    kill -TERM "$pid" 2>/dev/null || true
    
    # Wait up to 10 seconds for graceful shutdown
    for i in {1..10}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            print_success "$description (PID: $pid) terminated gracefully"
            return 0
        fi
        sleep 1
    done
    
    # Force kill if still running
    if kill -0 "$pid" 2>/dev/null; then
        print_warning "Force killing $description (PID: $pid)"
        kill -KILL "$pid" 2>/dev/null || true
        sleep 1
        if ! kill -0 "$pid" 2>/dev/null; then
            print_success "$description (PID: $pid) force killed"
        else
            print_error "Failed to kill $description (PID: $pid)"
            return 1
        fi
    fi
    
    return 0
}

# Function to cleanup processes on a specific port
cleanup_port() {
    local port=$1
    local port_pids
    
    port_pids=$(lsof -ti:$port 2>/dev/null || true)
    if [ -n "$port_pids" ]; then
        print_info "Found processes on port $port: $port_pids"
        for pid in $port_pids; do
            kill_process_gracefully "$pid" "port $port process"
        done
        
        # Verify port is free
        sleep 2
        port_pids=$(lsof -ti:$port 2>/dev/null || true)
        if [ -z "$port_pids" ]; then
            print_success "Port $port is now free"
        else
            print_warning "Port $port still has processes: $port_pids"
        fi
    else
        print_info "No processes found on port $port"
    fi
}

# Main cleanup function
main_cleanup() {
    local cleanup_success=true
    
    # Read and process PID file if it exists
    if [ -f "$PID_FILE" ]; then
        print_status "📁 Processing PID file: $PID_FILE"
        
        # Extract tracked processes
        while IFS= read -r line; do
            if [[ $line == PID:* ]]; then
                pid=$(echo "$line" | cut -d: -f2)
                description=$(echo "$line" | cut -d: -f3)
                if ! kill_process_gracefully "$pid" "$description"; then
                    cleanup_success=false
                fi
            elif [[ $line == PORT:* ]]; then
                port=$(echo "$line" | cut -d: -f2)
                cleanup_port "$port"
            fi
        done < "$PID_FILE"
        
        # Remove PID file
        rm -f "$PID_FILE"
        print_success "PID file removed"
    else
        print_warning "PID file not found: $PID_FILE"
    fi
    
    # Cleanup common ports used by Alpha Agent Pool
    print_status "🔌 Cleaning up common Alpha Agent Pool ports..."
    
    # Core server ports
    for port in 8081 8082 8083; do
        cleanup_port "$port"
    done
    
    # Agent ports
    for port in 5051 5052 5053; do
        cleanup_port "$port"
    done
    
    # Pattern-based process cleanup
    print_status "🔍 Searching for remaining Alpha Agent Pool processes..."
    
    local patterns=("alpha_agent_pool" "momentum_agent" "autonomous_agent" "AlphaAgentPoolMCPServer")
    
    for pattern in "${patterns[@]}"; do
        local pattern_pids
        pattern_pids=$(pgrep -f "$pattern" 2>/dev/null || true)
        if [ -n "$pattern_pids" ]; then
            print_info "Found $pattern processes: $pattern_pids"
            for pid in $pattern_pids; do
                if ! kill_process_gracefully "$pid" "$pattern process"; then
                    cleanup_success=false
                fi
            done
        else
            print_info "No $pattern processes found"
        fi
    done
    
    # Additional cleanup for any Python processes that might be related
    print_status "🐍 Checking for orphaned Python processes..."
    
    local python_pids
    python_pids=$(pgrep -f "FinAgents.agent_pools.alpha_agent_pool" 2>/dev/null || true)
    if [ -n "$python_pids" ]; then
        print_info "Found orphaned Python processes: $python_pids"
        for pid in $python_pids; do
            if ! kill_process_gracefully "$pid" "orphaned Python process"; then
                cleanup_success=false
            fi
        done
    else
        print_success "No orphaned Python processes found"
    fi
    
    # Final verification
    print_status "🔍 Final verification of cleanup..."
    
    # Check if any processes are still running
    local remaining_processes=0
    for pattern in "${patterns[@]}"; do
        local count
        count=$(pgrep -f "$pattern" 2>/dev/null | wc -l || echo "0")
        remaining_processes=$((remaining_processes + count))
    done
    
    if [ $remaining_processes -eq 0 ]; then
        print_success "✅ All Alpha Agent Pool processes have been terminated"
    else
        print_warning "⚠️  $remaining_processes processes may still be running"
        cleanup_success=false
    fi
    
    # Check if common ports are free
    local occupied_ports=()
    for port in 8081 8082 8083 5051 5052 5053; do
        if ! nc -z 127.0.0.1 $port 2>/dev/null; then
            : # Port is free, do nothing
        else
            occupied_ports+=("$port")
        fi
    done
    
    if [ ${#occupied_ports[@]} -eq 0 ]; then
        print_success "✅ All common ports are now available"
    else
        print_warning "⚠️  Ports still occupied: ${occupied_ports[*]}"
    fi
    
    return $([ "$cleanup_success" = true ] && echo 0 || echo 1)
}

# Execute main cleanup
if main_cleanup; then
    echo -e "${PURPLE}"
    echo "════════════════════════════════════════════════════════════════════"
    echo "🏁 Alpha Agent Pool shutdown completed successfully"
    echo "✅ All processes terminated and ports released"
    echo "🧹 System is clean and ready for restart"
    echo "════════════════════════════════════════════════════════════════════"
    echo -e "${NC}"
    exit 0
else
    echo -e "${PURPLE}"
    echo "════════════════════════════════════════════════════════════════════"
    echo "⚠️  Alpha Agent Pool shutdown completed with warnings"
    echo "🔍 Some processes or ports may still be occupied"
    echo "💡 You may need to manually check and cleanup remaining processes"
    echo "════════════════════════════════════════════════════════════════════"
    echo -e "${NC}"
    exit 1
fi
