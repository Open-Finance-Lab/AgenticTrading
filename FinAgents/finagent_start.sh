#!/bin/bash

# FinAgent Orchestration System - Quick Start Script
# This script helps you quickly start the entire FinAgent system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  FinAgent Orchestration System${NC}"
echo -e "${BLUE}  Quick Start Script${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to start a service
start_service() {
    local service_name=$1
    local service_path=$2
    local port=$3
    local log_file="$PROJECT_ROOT/logs/${service_name}.log"
    
    echo -e "${YELLOW}Starting $service_name...${NC}"
    
    if check_port $port; then
        echo -e "${GREEN}✅ $service_name already running on port $port${NC}"
        return
    fi
    
    # Create logs directory if it doesn't exist
    mkdir -p "$PROJECT_ROOT/logs"
    
    # Start the service in background
    cd "$PROJECT_ROOT/$service_path"
    nohup python core.py > "$log_file" 2>&1 &
    echo $! > "$PROJECT_ROOT/logs/${service_name}.pid"
    
    # Wait a moment for service to start
    sleep 3
    
    if check_port $port; then
        echo -e "${GREEN}✅ $service_name started successfully on port $port${NC}"
    else
        echo -e "${RED}❌ Failed to start $service_name${NC}"
        echo -e "${RED}   Check log file: $log_file${NC}"
        return 1
    fi
}

# Function to stop a service
stop_service() {
    local service_name=$1
    local pid_file="$PROJECT_ROOT/logs/${service_name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 $pid 2>/dev/null; then
            echo -e "${YELLOW}Stopping $service_name (PID: $pid)...${NC}"
            kill $pid
            rm -f "$pid_file"
            echo -e "${GREEN}✅ $service_name stopped${NC}"
        else
            echo -e "${YELLOW}$service_name was not running${NC}"
            rm -f "$pid_file"
        fi
    else
        echo -e "${YELLOW}No PID file found for $service_name${NC}"
    fi
}

# Function to check service status
check_service_status() {
    local service_name=$1
    local port=$2
    
    if check_port $port; then
        echo -e "${GREEN}✅ $service_name: Running on port $port${NC}"
    else
        echo -e "${RED}❌ $service_name: Not running${NC}"
    fi
}

# Function to show help
show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start       Start all services with enhanced monitoring"
    echo "  stop        Stop all services"
    echo "  restart     Restart all services"
    echo "  status      Show status of all services with MCP validation"
    echo "  demo        Run demonstration with multiple options"
    echo "  cli         Start interactive natural language interface"
    echo "  health      Run comprehensive health check"
    echo "  logs        Show logs for all services"
    echo "  help        Show this help message"
    echo ""
    echo "Individual service commands:"
    echo "  start-data         Start Data Agent Pool only"
    echo "  start-alpha        Start Alpha Agent Pool only"
    echo "  start-risk         Start Risk Agent Pool only"
    echo "  start-cost         Start Transaction Cost Agent Pool only"
    echo "  start-memory       Start Memory Agent only"
    echo "  start-nl           Start Natural Language Interface"
    echo "  start-orchestrator Start Enhanced Orchestrator only"
    echo ""
    echo "Enhanced Features:"
    echo "  • LLM-powered strategy planning from natural language"
    echo "  • Real-time agent pool health monitoring"
    echo "  • MCP protocol validation and connectivity testing"
    echo "  • Interactive CLI for natural language commands"
    echo "  • Advanced system diagnostics and reporting"
}

# Function to start all services
start_all() {
    echo -e "${BLUE}Starting all FinAgent services...${NC}"
    
    # Create logs directory
    mkdir -p "$PROJECT_ROOT/logs"
    
    # Start Memory Agent first
    echo -e "\n${BLUE}1. Memory Agent${NC}"
    cd "$PROJECT_ROOT/FinAgents/memory"
    if check_port 8010; then
        echo -e "${GREEN}✅ Memory Agent already running on port 8010${NC}"
    else
        if [ -f "external_memory_agent.py" ]; then
            nohup python external_memory_agent.py > "$PROJECT_ROOT/logs/memory_agent.log" 2>&1 &
            echo $! > "$PROJECT_ROOT/logs/memory_agent.pid"
            sleep 3
            if check_port 8010; then
                echo -e "${GREEN}✅ Memory Agent started on port 8010${NC}"
            else
                echo -e "${RED}❌ Failed to start Memory Agent${NC}"
            fi
        else
            echo -e "${YELLOW}⚠️  Memory Agent script not found, skipping...${NC}"
        fi
    fi
    
    # Start Agent Pools with enhanced monitoring
    echo -e "\n${BLUE}2. Data Agent Pool${NC}"
    start_agent_pool_with_validation "data_agent_pool" "FinAgents/agent_pools/data_agent_pool" 8001
    
    echo -e "\n${BLUE}3. Alpha Agent Pool${NC}"
    start_agent_pool_with_validation "alpha_agent_pool" "FinAgents/agent_pools/alpha_agent_pool" 5050
    
    echo -e "\n${BLUE}4. Risk Agent Pool${NC}"
    start_agent_pool_with_validation "risk_agent_pool" "FinAgents/agent_pools/risk_agent_pool" 7000
    
    echo -e "\n${BLUE}5. Transaction Cost Agent Pool${NC}"
    start_agent_pool_with_validation "transaction_cost_agent_pool" "FinAgents/agent_pools/transaction_cost_agent_pool" 6000
    
    # Wait for all services to be ready
    echo -e "\n${YELLOW}Waiting for all services to be ready...${NC}"
    sleep 5
    
    # Start Natural Language Interface
    echo -e "\n${BLUE}6. Natural Language Interface${NC}"
    cd "$PROJECT_ROOT/FinAgents/orchestrator"
    if check_port 8020; then
        echo -e "${GREEN}✅ NL Interface already running on port 8020${NC}"
    else
        nohup python -c "
import asyncio
from core.mcp_nl_interface import MCPNaturalLanguageInterface
import yaml

try:
    with open('config/orchestrator_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    interface = MCPNaturalLanguageInterface(config)
    asyncio.run(interface.start_server('localhost', 8020))
except Exception as e:
    print(f'Error starting NL Interface: {e}')
" > "$PROJECT_ROOT/logs/nl_interface.log" 2>&1 &
        echo $! > "$PROJECT_ROOT/logs/nl_interface.pid"
        sleep 3
        if check_port 8020; then
            echo -e "${GREEN}✅ NL Interface started on port 8020${NC}"
        else
            echo -e "${RED}❌ Failed to start NL Interface${NC}"
        fi
    fi
    
    # Start Orchestrator with LLM enhancements
    echo -e "\n${BLUE}7. Enhanced Orchestrator${NC}"
    if check_port 9000; then
        echo -e "${GREEN}✅ Orchestrator already running on port 9000${NC}"
    else
        nohup python main_orchestrator.py --mode production --enable-llm > "$PROJECT_ROOT/logs/orchestrator.log" 2>&1 &
        echo $! > "$PROJECT_ROOT/logs/orchestrator.pid"
        sleep 5
        if check_port 9000; then
            echo -e "${GREEN}✅ Enhanced Orchestrator started on port 9000${NC}"
        else
            echo -e "${RED}❌ Failed to start Orchestrator${NC}"
        fi
    fi
    
    # Validate system with enhanced monitoring
    echo -e "\n${BLUE}8. System Validation${NC}"
    validate_system_startup
    
    echo -e "\n${GREEN}🎉 All services started successfully!${NC}"
    echo -e "${BLUE}System is ready for natural language trading interactions.${NC}"
    echo -e "${BLUE}Try: 'Execute a momentum strategy for AAPL and GOOGL'${NC}"
}

# Enhanced function to start agent pool with validation
start_agent_pool_with_validation() {
    local service_name=$1
    local service_path=$2
    local port=$3
    local log_file="$PROJECT_ROOT/logs/${service_name}.log"
    
    echo -e "${YELLOW}Starting $service_name with validation...${NC}"
    
    if check_port $port; then
        echo -e "${GREEN}✅ $service_name already running on port $port${NC}"
        validate_mcp_connectivity $service_name $port
        return
    fi
    
    # Check if service directory exists
    if [ ! -d "$PROJECT_ROOT/$service_path" ]; then
        echo -e "${RED}❌ Service directory not found: $service_path${NC}"
        return 1
    fi
    
    # Start the service in background
    cd "$PROJECT_ROOT/$service_path"
    if [ -f "core.py" ]; then
        nohup python core.py > "$log_file" 2>&1 &
        echo $! > "$PROJECT_ROOT/logs/${service_name}.pid"
        
        # Enhanced startup validation
        local max_attempts=30
        local attempt=0
        
        echo -e "${YELLOW}   Waiting for startup (max ${max_attempts}s)...${NC}"
        
        while [ $attempt -lt $max_attempts ]; do
            sleep 1
            attempt=$((attempt + 1))
            
            if check_port $port; then
                echo -e "${GREEN}✅ $service_name started successfully on port $port (${attempt}s)${NC}"
                
                # Validate MCP connectivity
                validate_mcp_connectivity $service_name $port
                return 0
            fi
            
            # Show progress every 5 seconds
            if [ $((attempt % 5)) -eq 0 ]; then
                echo -e "${YELLOW}   Still starting... (${attempt}/${max_attempts}s)${NC}"
            fi
        done
        
        echo -e "${RED}❌ Failed to start $service_name (timeout after ${max_attempts}s)${NC}"
        echo -e "${RED}   Check log file: $log_file${NC}"
        return 1
    else
        echo -e "${RED}❌ core.py not found in $service_path${NC}"
        return 1
    fi
}

# Function to validate MCP connectivity
validate_mcp_connectivity() {
    local service_name=$1
    local port=$2
    local url="http://localhost:$port"
    
    echo -e "${YELLOW}   Validating MCP connectivity...${NC}"
    
    # Test health endpoint
    if command -v curl >/dev/null 2>&1; then
        if curl -s --max-time 5 "$url/health" >/dev/null 2>&1; then
            echo -e "${GREEN}   ✅ Health endpoint responsive${NC}"
        else
            echo -e "${YELLOW}   ⚠️  Health endpoint not available${NC}"
        fi
        
        # Test MCP capabilities
        if curl -s --max-time 5 "$url/mcp/capabilities" >/dev/null 2>&1; then
            echo -e "${GREEN}   ✅ MCP protocol active${NC}"
        else
            echo -e "${YELLOW}   ⚠️  MCP endpoints not ready yet${NC}"
        fi
    else
        echo -e "${YELLOW}   ⚠️  curl not available, skipping detailed validation${NC}"
    fi
}

# Function to validate entire system startup
validate_system_startup() {
    echo -e "${YELLOW}Running system validation...${NC}"
    
    # Check if orchestrator is available to run validation
    cd "$PROJECT_ROOT/FinAgents/orchestrator"
    if [ -f "core/agent_pool_monitor.py" ]; then
        python -c "
import asyncio
import sys
import os
sys.path.append(os.getcwd())

async def validate():
    try:
        from core.agent_pool_monitor import AgentPoolMonitor
        import yaml
        
        # Load config
        with open('config/orchestrator_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize monitor
        monitor = AgentPoolMonitor(config)
        
        # Check all pools
        results = await monitor.check_all_pools()
        
        print('📊 System Validation Results:')
        healthy_count = 0
        for pool_name, pool_info in results.items():
            if pool_info.status.value == 'healthy':
                print(f'✅ {pool_name}: {pool_info.status.value}')
                healthy_count += 1
            else:
                print(f'❌ {pool_name}: {pool_info.status.value}')
                if pool_info.error_message:
                    print(f'   Error: {pool_info.error_message}')
        
        total_pools = len(results)
        print(f'📈 Health Summary: {healthy_count}/{total_pools} pools healthy')
        
        if healthy_count == total_pools:
            print('🎉 All systems operational!')
        elif healthy_count > 0:
            print('⚠️  System partially operational')
        else:
            print('❌ System not operational')
            
    except Exception as e:
        print(f'❌ Validation failed: {e}')

asyncio.run(validate())
" 2>/dev/null || echo -e "${YELLOW}   System validation script unavailable${NC}"
    else
        echo -e "${YELLOW}   Enhanced validation not available${NC}"
    fi
}

# Function to stop all services
stop_all() {
    echo -e "${BLUE}Stopping all FinAgent services...${NC}"
    
    stop_service "orchestrator"
    stop_service "transaction_cost_agent_pool"
    stop_service "risk_agent_pool"
    stop_service "alpha_agent_pool"
    stop_service "data_agent_pool"
    stop_service "memory_agent"
    
    echo -e "\n${GREEN}🛑 All services stopped${NC}"
}

# Function to check status of all services
check_status() {
    echo -e "${BLUE}FinAgent System Status:${NC}"
    echo ""
    check_service_status "Memory Agent" 8010
    check_service_status "Data Agent Pool" 8001
    check_service_status "Alpha Agent Pool" 5050
    check_service_status "Risk Agent Pool" 7000
    check_service_status "Transaction Cost Agent Pool" 6000
    check_service_status "Natural Language Interface" 8020
    check_service_status "Orchestrator" 9000
    
    echo ""
    echo -e "${BLUE}Enhanced Features:${NC}"
    
    # Test natural language interface if available
    if check_port 8020; then
        echo -e "${GREEN}✅ Natural Language Processing: Available${NC}"
        echo -e "${BLUE}   Try: python -c \"import asyncio; from core.mcp_nl_interface import *\"${NC}"
    else
        echo -e "${RED}❌ Natural Language Processing: Not Available${NC}"
    fi
    
    # Test LLM integration
    if check_port 9000; then
        echo -e "${GREEN}✅ LLM-Enhanced DAG Planning: Available${NC}"
    else
        echo -e "${RED}❌ LLM-Enhanced DAG Planning: Not Available${NC}"
    fi
    
    # Test MCP protocol
    echo -e "${BLUE}Testing MCP Protocol Connectivity...${NC}"
    for port in 8001 5050 7000 6000; do
        if check_port $port; then
            if command -v curl >/dev/null 2>&1; then
                if curl -s --max-time 2 "http://localhost:$port/health" >/dev/null 2>&1; then
                    echo -e "${GREEN}✅ Port $port: MCP Health OK${NC}"
                else
                    echo -e "${YELLOW}⚠️  Port $port: Service up, MCP status unknown${NC}"
                fi
            else
                echo -e "${YELLOW}⚠️  Port $port: curl unavailable for MCP test${NC}"
            fi
        fi
    done
}

# Function to show logs
show_logs() {
    echo -e "${BLUE}Recent log entries:${NC}"
    echo ""
    
    local log_dir="$PROJECT_ROOT/logs"
    if [ -d "$log_dir" ]; then
        for log_file in "$log_dir"/*.log; do
            if [ -f "$log_file" ]; then
                local service_name=$(basename "$log_file" .log)
                echo -e "${YELLOW}=== $service_name ===${NC}"
                tail -n 10 "$log_file" 2>/dev/null || echo "No logs available"
                echo ""
            fi
        done
    else
        echo "No log directory found"
    fi
}

# Function to run demo
run_demo() {
    echo -e "${BLUE}Running FinAgent demonstration...${NC}"
    
    # Check if services are running
    local all_running=true
    local required_ports=(8001 5050 7000 6000 9000)
    
    for port in "${required_ports[@]}"; do
        if ! check_port $port; then
            all_running=false
            break
        fi
    done
    
    if [ "$all_running" = false ]; then
        echo -e "${RED}❌ Not all services are running. Please start services first:${NC}"
        echo -e "${YELLOW}   $0 start${NC}"
        return 1
    fi
    
    cd "$PROJECT_ROOT/FinAgents/orchestrator"
    
    # Offer different demo options
    echo -e "${BLUE}Available demo options:${NC}"
    echo -e "${YELLOW}  1. Quick Start Demo (basic features)${NC}"
    echo -e "${YELLOW}  2. Enhanced Orchestrator Demo (LLM + NL interface)${NC}"
    echo -e "${YELLOW}  3. Interactive CLI (natural language interface)${NC}"
    echo -e "${YELLOW}  4. Agent Pool Health Check${NC}"
    echo ""
    
    read -p "Select demo option (1-4) or press Enter for default [1]: " demo_choice
    demo_choice=${demo_choice:-1}
    
    case $demo_choice in
        1)
            echo -e "${BLUE}Running Quick Start Demo...${NC}"
            python quick_start_demo.py --demo-type all
            ;;
        2)
            echo -e "${BLUE}Running Enhanced Orchestrator Demo...${NC}"
            python enhanced_orchestrator_demo.py
            ;;
        3)
            echo -e "${BLUE}Starting Interactive CLI...${NC}"
            echo -e "${YELLOW}Type natural language commands or 'quit' to exit${NC}"
            python finagent_cli.py
            ;;
        4)
            echo -e "${BLUE}Running Agent Pool Health Check...${NC}"
            python -c "
import asyncio
import sys
import os
sys.path.append(os.getcwd())

async def health_check():
    try:
        from core.agent_pool_monitor import AgentPoolMonitor
        import yaml
        
        with open('config/orchestrator_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        monitor = AgentPoolMonitor(config)
        results = await monitor.check_all_pools()
        
        print('🔍 Agent Pool Health Check Results:')
        print('=' * 50)
        
        for pool_name, pool_info in results.items():
            status_icon = '✅' if pool_info.status.value == 'healthy' else '❌'
            print(f'{status_icon} {pool_name}: {pool_info.status.value}')
            
            if pool_info.status.value == 'healthy':
                if pool_info.response_time:
                    print(f'   🕒 Response time: {pool_info.response_time:.3f}s')
                if pool_info.capabilities:
                    print(f'   🛠️  Capabilities: {len(pool_info.capabilities)} available')
                
                # Test MCP connectivity
                mcp_result = await monitor.validate_mcp_connectivity(pool_name)
                if mcp_result['success']:
                    tools_count = len(mcp_result.get('available_tools', []))
                    print(f'   🔗 MCP: OK ({tools_count} tools)')
                else:
                    print(f'   🔗 MCP: {mcp_result[\"error\"]}')
            else:
                if pool_info.error_message:
                    print(f'   ❌ Error: {pool_info.error_message}')
        
        # System summary
        healthy_count = sum(1 for p in results.values() if p.status.value == 'healthy')
        total_count = len(results)
        health_percentage = (healthy_count / total_count * 100) if total_count > 0 else 0
        
        print(f'\\n📊 Summary: {healthy_count}/{total_count} pools healthy ({health_percentage:.1f}%)')
        
    except Exception as e:
        print(f'❌ Health check failed: {e}')

asyncio.run(health_check())
"
            ;;
        *)
            echo -e "${RED}Invalid option. Running default demo...${NC}"
            python quick_start_demo.py --demo-type all
            ;;
    esac
}

# Function to run natural language interface
run_cli() {
    echo -e "${BLUE}Starting FinAgent Natural Language Interface...${NC}"
    
    cd "$PROJECT_ROOT/FinAgents/orchestrator"
    
    # Check if enhanced features are available
    if [ -f "finagent_cli.py" ]; then
        echo -e "${GREEN}✅ Enhanced CLI available${NC}"
        python finagent_cli.py
    else
        echo -e "${YELLOW}⚠️  Enhanced CLI not found, using basic demo${NC}"
        python quick_start_demo.py --demo-type basic
    fi
}

# Main script logic
case "${1:-}" in
    start)
        start_all
        ;;
    stop)
        stop_all
        ;;
    restart)
        stop_all
        sleep 2
        start_all
        ;;
    status)
        check_status
        ;;
    demo)
        run_demo
        ;;
    cli)
        run_cli
        ;;
    health)
        cd "$PROJECT_ROOT/FinAgents/orchestrator"
        if [ -f "core/agent_pool_monitor.py" ]; then
            echo -e "${BLUE}Running comprehensive health check...${NC}"
            python core/agent_pool_monitor.py
        else
            echo -e "${YELLOW}Enhanced health check not available${NC}"
            check_status
        fi
        ;;
    logs)
        show_logs
        ;;
    start-data)
        start_agent_pool_with_validation "data_agent_pool" "FinAgents/agent_pools/data_agent_pool" 8001
        ;;
    start-alpha)
        start_agent_pool_with_validation "alpha_agent_pool" "FinAgents/agent_pools/alpha_agent_pool" 5050
        ;;
    start-risk)
        start_agent_pool_with_validation "risk_agent_pool" "FinAgents/agent_pools/risk_agent_pool" 7000
        ;;
    start-cost)
        start_agent_pool_with_validation "transaction_cost_agent_pool" "FinAgents/agent_pools/transaction_cost_agent_pool" 6000
        ;;
    start-memory)
        cd "$PROJECT_ROOT/FinAgents/memory"
        if [ -f "external_memory_agent.py" ]; then
            nohup python external_memory_agent.py > "$PROJECT_ROOT/logs/memory_agent.log" 2>&1 &
            echo $! > "$PROJECT_ROOT/logs/memory_agent.pid"
            echo -e "${GREEN}✅ Memory Agent started${NC}"
        else
            echo -e "${RED}❌ Memory Agent script not found${NC}"
        fi
        ;;
    start-nl)
        cd "$PROJECT_ROOT/FinAgents/orchestrator"
        if [ -f "core/mcp_nl_interface.py" ]; then
            echo -e "${BLUE}Starting Natural Language Interface...${NC}"
            nohup python -c "
import asyncio
from core.mcp_nl_interface import MCPNaturalLanguageInterface
import yaml

try:
    with open('config/orchestrator_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    interface = MCPNaturalLanguageInterface(config)
    asyncio.run(interface.start_server('localhost', 8020))
except Exception as e:
    print(f'Error: {e}')
" > "$PROJECT_ROOT/logs/nl_interface.log" 2>&1 &
            echo $! > "$PROJECT_ROOT/logs/nl_interface.pid"
            echo -e "${GREEN}✅ Natural Language Interface started on port 8020${NC}"
        else
            echo -e "${RED}❌ Natural Language Interface not available${NC}"
        fi
        ;;
    start-orchestrator)
        cd "$PROJECT_ROOT/FinAgents/orchestrator"
        nohup python main_orchestrator.py --mode production --enable-llm > "$PROJECT_ROOT/logs/orchestrator.log" 2>&1 &
        echo $! > "$PROJECT_ROOT/logs/orchestrator.pid"
        echo -e "${GREEN}✅ Enhanced Orchestrator started${NC}"
        ;;
    help|--help|-h)
        show_help
        ;;
    "")
        echo -e "${RED}No command specified. Use 'help' for usage information.${NC}"
        show_help
        exit 1
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        show_help
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}For more information, check the documentation:${NC}"
echo -e "${BLUE}  - README.md in FinAgents/orchestrator/${NC}"
echo -e "${BLUE}  - Log files in logs/ directory${NC}"
