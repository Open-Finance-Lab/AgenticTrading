#!/bin/bash

# FinAgent Memory System - One-Click Startup Script
# 
# This script provides easy startup and management of the FinAgent Memory System
# including A2A Memory Server, MCP Server, Memory Server and testing utilities.

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
A2A_PORT=8002
MCP_PORT=8001
MEMORY_PORT=8000
LOG_DIR="$SCRIPT_DIR/logs"
PID_FILE="$LOG_DIR/memory_servers.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Helper functions
print_header() {
    echo -e "${BLUE}================================================================================================${NC}"
    echo -e "${BLUE}🚀 FINAGENT MEMORY SYSTEM - $1${NC}"
    echo -e "${BLUE}================================================================================================${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
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

check_dependencies() {
    print_info "Checking dependencies..."
    
    # Check Python
    if ! command -v python &> /dev/null; then
        print_error "Python is not installed or not in PATH"
        exit 1
    fi
    
    # Check if we're in the right directory
    if [[ ! -f "a2a_server.py" ]]; then
        print_error "a2a_server.py not found. Please run this script from the memory directory."
        exit 1
    fi
    
    if [[ ! -f "mcp_server.py" ]]; then
        print_error "mcp_server.py not found. Please run this script from the memory directory."
        exit 1
    fi
    
    if [[ ! -f "memory_server.py" ]]; then
        print_error "memory_server.py not found. Please run this script from the memory directory."
        exit 1
    fi
    
    print_success "Dependencies check passed"
}

check_server_status() {
    local port=$1
    local name=$2
    
    # Try different endpoints based on server type
    if [[ "$name" == *"A2A"* ]]; then
        # For A2A server, check if port is listening
        if nc -z localhost $port 2>/dev/null; then
            print_success "$name is running on port $port"
            return 0
        else
            print_warning "$name is not running on port $port"
            return 1
        fi
    elif curl -s -f "http://localhost:$port/health" > /dev/null 2>&1; then
        print_success "$name is running on port $port"
        return 0
    elif nc -z localhost $port 2>/dev/null; then
        print_success "$name is running on port $port (no health endpoint)"
        return 0
    else
        print_warning "$name is not running on port $port"
        return 1
    fi
}

start_a2a_server() {
    print_header "STARTING A2A MEMORY SERVER"
    
    # Check if server is already running
    if check_server_status $A2A_PORT "A2A Memory Server"; then
        print_warning "A2A Memory Server is already running on port $A2A_PORT"
        return 0
    fi
    
    print_info "Starting A2A Memory Server on port $A2A_PORT..."
    
    # Start server in background
    nohup python a2a_server.py --host 0.0.0.0 --port $A2A_PORT > "$LOG_DIR/a2a_server.log" 2>&1 &
    local server_pid=$!
    echo $server_pid >> "$PID_FILE"
    
    print_info "Server started with PID: $server_pid"
    
    # Wait for server to start with timeout
    local timeout=15
    local count=0
    while [ $count -lt $timeout ]; do
        if check_server_status $A2A_PORT "A2A Memory Server" >/dev/null 2>&1; then
            print_success "A2A Memory Server started successfully!"
            echo -e "${CYAN}📡 Server URL: http://localhost:$A2A_PORT${NC}"
            echo -e "${CYAN}📋 Agent Card: http://localhost:$A2A_PORT/.well-known/agent-card${NC}"
            echo -e "${CYAN}📄 Logs: $LOG_DIR/a2a_server.log${NC}"
            return 0
        fi
        sleep 1
        count=$((count + 1))
        print_info "Waiting for server to start... ($count/$timeout)"
    done
    
    print_error "Failed to start A2A Memory Server (timeout after ${timeout}s)"
    print_info "Check logs: tail -f $LOG_DIR/a2a_server.log"
    return 1
}

start_mcp_server() {
    print_header "STARTING MCP MEMORY SERVER"
    
    # Check if server is already running
    if check_server_status $MCP_PORT "MCP Memory Server"; then
        print_warning "MCP Memory Server is already running on port $MCP_PORT"
        return 0
    fi
    
    print_info "Starting MCP Memory Server on port $MCP_PORT..."
    
    # Start server in background
    nohup uvicorn mcp_server:app --host 0.0.0.0 --port $MCP_PORT > "$LOG_DIR/mcp_server.log" 2>&1 &
    local server_pid=$!
    echo $server_pid >> "$PID_FILE"
    
    print_info "Server started with PID: $server_pid"
    
    # Wait for server to start with timeout
    local timeout=15
    local count=0
    while [ $count -lt $timeout ]; do
        if check_server_status $MCP_PORT "MCP Memory Server" >/dev/null 2>&1; then
            print_success "MCP Memory Server started successfully!"
            echo -e "${CYAN}📡 Server URL: http://localhost:$MCP_PORT${NC}"
            echo -e "${CYAN}📋 Health Check: http://localhost:$MCP_PORT/health${NC}"
            echo -e "${CYAN}📄 Logs: $LOG_DIR/mcp_server.log${NC}"
            return 0
        fi
        sleep 1
        count=$((count + 1))
        print_info "Waiting for server to start... ($count/$timeout)"
    done
    
    print_error "Failed to start MCP Memory Server (timeout after ${timeout}s)"
    print_info "Check logs: tail -f $LOG_DIR/mcp_server.log"
    return 1
}

start_memory_server() {
    print_header "STARTING LEGACY MEMORY SERVER"
    
    # Check if server is already running
    if check_server_status $MEMORY_PORT "Memory Server"; then
        print_warning "Memory Server is already running on port $MEMORY_PORT"
        return 0
    fi
    
    print_info "Starting Memory Server on port $MEMORY_PORT..."
    
    # Start server in background
    nohup uvicorn memory_server:app --host 0.0.0.0 --port $MEMORY_PORT > "$LOG_DIR/memory_server.log" 2>&1 &
    local server_pid=$!
    echo $server_pid >> "$PID_FILE"
    
    print_info "Server started with PID: $server_pid"
    
    # Wait for server to start with timeout
    local timeout=15
    local count=0
    while [ $count -lt $timeout ]; do
        if check_server_status $MEMORY_PORT "Memory Server" >/dev/null 2>&1; then
            print_success "Memory Server started successfully!"
            echo -e "${CYAN}📡 Server URL: http://localhost:$MEMORY_PORT${NC}"
            echo -e "${CYAN}📋 Health Check: http://localhost:$MEMORY_PORT/health${NC}"
            echo -e "${CYAN}📄 Logs: $LOG_DIR/memory_server.log${NC}"
            return 0
        fi
        sleep 1
        count=$((count + 1))
        print_info "Waiting for server to start... ($count/$timeout)"
    done
    
    print_error "Failed to start Memory Server (timeout after ${timeout}s)"
    print_info "Check logs: tail -f $LOG_DIR/memory_server.log"
    return 1
}

start_all_servers() {
    print_header "STARTING ALL MEMORY SERVERS"
    
    check_dependencies
    
    # Start all servers
    start_a2a_server
    echo ""
    start_mcp_server
    echo ""
    start_memory_server
    
    echo ""
    print_header "ALL SERVERS STARTED"
    show_status
}

run_tests() {
    print_header "RUNNING MEMORY SYSTEM TESTS"
    
    check_dependencies
    
    # Check if test file exists in tests directory
    if [[ ! -f "tests/memory_test.py" ]]; then
        print_error "tests/memory_test.py not found"
        print_info "Please ensure tests directory exists with memory_test.py"
        return 1
    fi
    
    # Check if test system script exists
    if [[ -f "tests/test_system.sh" ]]; then
        print_info "Running comprehensive memory system tests via test_system.sh..."
        cd tests && ./test_system.sh "$@"
        cd ..
    else
        print_info "Running unified memory system tests directly..."
        
        # Ensure all servers are running for comprehensive testing
        local servers_started=false
        
        if ! check_server_status $MEMORY_PORT "Memory Server" >/dev/null 2>&1; then
            print_info "Starting Memory Server for testing..."
            start_memory_server >/dev/null 2>&1 &
            servers_started=true
            sleep 2
        fi
        
        if ! check_server_status $MCP_PORT "MCP Server" >/dev/null 2>&1; then
            print_info "Starting MCP Server for testing..."
            start_mcp_server >/dev/null 2>&1 &
            servers_started=true
            sleep 2
        fi
        
        if ! check_server_status $A2A_PORT "A2A Memory Server" >/dev/null 2>&1; then
            print_info "Starting A2A Memory Server for testing..."
            start_a2a_server >/dev/null 2>&1 &
            servers_started=true
            sleep 3
        fi
        
        if [[ "$servers_started" == "true" ]]; then
            print_info "Waiting for servers to fully initialize..."
            sleep 5
        fi
        
        # Run the unified test suite
        python tests/memory_test.py --verbose "$@"
        
        local test_exit_code=$?
        
        if [[ $test_exit_code -eq 0 ]]; then
            print_success "🎉 All tests passed! Memory system is fully functional."
        else
            print_warning "⚠️  Some tests failed, but system may still be functional."
            print_info "💡 Check test output above for details."
        fi
        
        return $test_exit_code
    fi
}

stop_servers() {
    print_header "STOPPING MEMORY SERVERS"
    
    if [[ -f "$PID_FILE" ]]; then
        print_info "Stopping servers..."
        while read -r pid; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid"
                print_info "Stopped process $pid"
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
        print_success "All memory servers stopped"
    else
        print_warning "No running servers found"
    fi
    
    # Double-check by port
    for port in $A2A_PORT $MCP_PORT $MEMORY_PORT; do
        pid=$(lsof -ti:$port 2>/dev/null || echo "")
        if [[ -n "$pid" ]]; then
            print_info "Force stopping process on port $port (PID: $pid)"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
}

show_status() {
    print_header "MEMORY SYSTEM STATUS"
    
    echo -e "${PURPLE}🖥️  System Information:${NC}"
    echo -e "   📁 Working Directory: $SCRIPT_DIR"
    echo -e "   📄 Log Directory: $LOG_DIR"
    echo -e "   🐍 Python Version: $(python --version)"
    echo ""
    
    echo -e "${PURPLE}🚀 Server Status:${NC}"
    check_server_status $A2A_PORT "A2A Memory Server" || true
    check_server_status $MCP_PORT "MCP Memory Server" || true
    check_server_status $MEMORY_PORT "Legacy Memory Server" || true
    echo ""
}

show_logs() {
    print_header "MEMORY SYSTEM LOGS"
    
    if [[ -f "$LOG_DIR/a2a_server.log" ]]; then
        print_info "A2A Server Logs (last 20 lines):"
        tail -n 20 "$LOG_DIR/a2a_server.log"
        echo ""
    fi
    
    if [[ -f "$LOG_DIR/mcp_server.log" ]]; then
        print_info "MCP Server Logs (last 20 lines):"
        tail -n 20 "$LOG_DIR/mcp_server.log"
        echo ""
    fi
    
    if [[ -f "$LOG_DIR/memory_server.log" ]]; then
        print_info "Memory Server Logs (last 20 lines):"
        tail -n 20 "$LOG_DIR/memory_server.log"
        echo ""
    fi
    
    if [[ ! -f "$LOG_DIR/a2a_server.log" && ! -f "$LOG_DIR/mcp_server.log" && ! -f "$LOG_DIR/memory_server.log" ]]; then
        print_warning "No server logs found"
    fi
}

show_help() {
    echo ""
    echo "================================================================================================"
    echo "🚀 FINAGENT MEMORY SYSTEM - HELP & USAGE"
    echo "================================================================================================"
    
    cat << EOF
COMMANDS:
  start        Start all memory servers (A2A, MCP, Legacy)
  a2a          Start only A2A Memory Server
  mcp          Start only MCP Memory Server  
  memory       Start only Legacy Memory Server
  test         Run comprehensive memory system tests
  stop         Stop all memory servers
  status       Check server status and system information
  logs         View server logs
  help         Show this help message

EXAMPLES:
  ./start_memory_system.sh                    # Start all servers
  ./start_memory_system.sh start              # Start all servers
  ./start_memory_system.sh a2a                # Start only A2A server
  ./start_memory_system.sh mcp                # Start only MCP server
  ./start_memory_system.sh test               # Run tests
  ./start_memory_system.sh test --timeout 60  # Run tests with 60s timeout
  ./start_memory_system.sh logs               # View server logs
  ./start_memory_system.sh stop               # Stop all servers

SERVER CONFIGURATION:
  A2A Memory Server Port:     $A2A_PORT
  MCP Memory Server Port:     $MCP_PORT
  Legacy Memory Server Port:  $MEMORY_PORT
  Log Directory:              $LOG_DIR

REQUIREMENTS:
  - Python 3.8+
  - Required packages: a2a, uvicorn, click, requests, fastapi, mcp
  - Neo4j database (optional, for full functionality)

SERVER DESCRIPTIONS:
  A2A Server:    Agent-to-Agent protocol compliant memory server
  MCP Server:    Model Context Protocol compliant memory server  
  Memory Server: Legacy FastAPI memory server with comprehensive tools
EOF
}

# Main script logic
main() {
    local command=${1:-"start"}
    shift || true  # Remove first argument, ignore error if no arguments
    
    case $command in
        "start"|"all")
            start_all_servers "$@"
            ;;
        "a2a")
            check_dependencies
            start_a2a_server "$@"
            ;;
        "mcp")
            check_dependencies
            start_mcp_server "$@"
            ;;
        "memory"|"legacy")
            check_dependencies  
            start_memory_server "$@"
            ;;
        "test")
            run_tests "$@"
            ;;
        "stop")
            stop_servers "$@"
            ;;
        "status")
            show_status "$@"
            ;;
        "logs")
            show_logs "$@"
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            print_error "Unknown command: $command"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"
