#!/bin/bash
# FinAgent A2A Memory Integration Setup and Test Script
# This script handles the complete setup and testing of the A2A memory integration

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NEO4J_PASSWORD="${NEO4J_PASSWORD:-finagent123}"
MEMORY_AGENT_PORT="${MEMORY_AGENT_PORT:-8010}"
ALPHA_POOL_PORT="${ALPHA_POOL_PORT:-8081}"

echo -e "${BLUE}🚀 FinAgent A2A Memory Integration Setup${NC}"
echo "=================================="

# Function to check if a service is running
check_service() {
    local port=$1
    local name=$2
    
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${GREEN}✅ $name is running on port $port${NC}"
        return 0
    else
        echo -e "${RED}❌ $name is not running on port $port${NC}"
        return 1
    fi
}

# Function to wait for service
wait_for_service() {
    local port=$1
    local name=$2
    local timeout=${3:-30}
    
    echo -e "${YELLOW}⏳ Waiting for $name on port $port...${NC}"
    
    for i in $(seq 1 $timeout); do
        if check_service $port "$name"; then
            return 0
        fi
        sleep 1
    done
    
    echo -e "${RED}❌ Timeout waiting for $name${NC}"
    return 1
}

# Function to start Neo4j
setup_neo4j() {
    echo -e "${BLUE}📊 Setting up Neo4j Database${NC}"
    
    # Check if Neo4j is already running
    if check_service 7687 "Neo4j"; then
        echo -e "${YELLOW}Neo4j is already running, using existing instance${NC}"
    else
        echo -e "${YELLOW}Starting Neo4j with Docker...${NC}"
        
        # Stop existing container if it exists
        docker stop neo4j-finagent 2>/dev/null || true
        docker rm neo4j-finagent 2>/dev/null || true
        
        # Start Neo4j container
        docker run -d \
            --name neo4j-finagent \
            -p 7474:7474 -p 7687:7687 \
            -e NEO4J_AUTH=neo4j/$NEO4J_PASSWORD \
            -e NEO4J_dbms_memory_heap_initial__size=512m \
            -e NEO4J_dbms_memory_heap_max__size=2G \
            -v neo4j-finagent-data:/data \
            neo4j:5.15
        
        # Wait for Neo4j to be ready
        wait_for_service 7687 "Neo4j" 60
    fi
    
    # Initialize database schema
    echo -e "${YELLOW}Initializing database schema...${NC}"
    cd "$PROJECT_ROOT"
    python scripts/setup_neo4j.py init
    
    echo -e "${GREEN}✅ Neo4j setup completed${NC}"
}

# Function to start Memory Agent
start_memory_agent() {
    echo -e "${BLUE}🧠 Starting Memory Agent${NC}"
    
    if check_service $MEMORY_AGENT_PORT "Memory Agent"; then
        echo -e "${YELLOW}Memory Agent is already running${NC}"
        return 0
    fi
    
    # Start memory agent in background
    cd "$PROJECT_ROOT"
    
    # Create memory agent startup script if it doesn't exist
    if [ ! -f "start_memory_agent.py" ]; then
        cat > start_memory_agent.py << EOF
#!/usr/bin/env python3
"""
Temporary Memory Agent Server for A2A Integration Testing
"""
import asyncio
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime

app = FastAPI(title="FinAgent Memory Agent", version="1.0.0")

# In-memory storage for testing
memory_store = {
    "signals": [],
    "strategies": [],
    "performance_data": []
}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "finagent-memory-agent"
    }

@app.post("/api/v1/store/signal")
async def store_signal(signal_data: dict):
    signal_data["timestamp"] = datetime.utcnow().isoformat()
    memory_store["signals"].append(signal_data)
    return {"status": "success", "signal_id": len(memory_store["signals"])}

@app.get("/api/v1/retrieve/strategies")
async def retrieve_strategies(agent_id: str = None):
    if agent_id:
        strategies = [s for s in memory_store["strategies"] if s.get("agent_id") == agent_id]
    else:
        strategies = memory_store["strategies"]
    
    return {"strategies": strategies}

@app.post("/api/v1/store/performance")
async def store_performance(perf_data: dict):
    perf_data["timestamp"] = datetime.utcnow().isoformat()
    memory_store["performance_data"].append(perf_data)
    return {"status": "success", "record_id": len(memory_store["performance_data"])}

@app.get("/api/v1/status")
async def get_status():
    return {
        "signals_count": len(memory_store["signals"]),
        "strategies_count": len(memory_store["strategies"]),
        "performance_records": len(memory_store["performance_data"])
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=$MEMORY_AGENT_PORT)
EOF
    fi
    
    python start_memory_agent.py &
    MEMORY_AGENT_PID=$!
    echo $MEMORY_AGENT_PID > memory_agent.pid
    
    # Wait for memory agent to start
    wait_for_service $MEMORY_AGENT_PORT "Memory Agent" 30
    
    echo -e "${GREEN}✅ Memory Agent started (PID: $MEMORY_AGENT_PID)${NC}"
}

# Function to install dependencies
install_dependencies() {
    echo -e "${BLUE}📦 Installing Python Dependencies${NC}"
    
    cd "$PROJECT_ROOT"
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo -e "${YELLOW}Creating virtual environment...${NC}"
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    pip install -r requirements.txt
    
    # Install additional test dependencies
    pip install pytest pytest-asyncio httpx
    
    echo -e "${GREEN}✅ Dependencies installed${NC}"
}

# Function to run integration tests
run_integration_tests() {
    echo -e "${BLUE}🧪 Running Integration Tests${NC}"
    
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    
    # Create test configuration
    cat > test_config.yaml << EOF
neo4j:
  uri: "bolt://localhost:7687"
  username: "neo4j"
  password: "$NEO4J_PASSWORD"
  database: "finagent"

memory_agent:
  url: "http://127.0.0.1:$MEMORY_AGENT_PORT"

alpha_pool:
  url: "http://127.0.0.1:$ALPHA_POOL_PORT"
EOF
    
    # Run the integration test pipeline
    python scripts/test_integration_pipeline.py --config test_config.yaml --verbose
    
    echo -e "${GREEN}✅ Integration tests completed${NC}"
}

# Function to cleanup
cleanup() {
    echo -e "${BLUE}🧹 Cleaning up...${NC}"
    
    # Stop memory agent
    if [ -f "memory_agent.pid" ]; then
        PID=$(cat memory_agent.pid)
        kill $PID 2>/dev/null || true
        rm memory_agent.pid
        echo -e "${YELLOW}Memory Agent stopped${NC}"
    fi
    
    # Option to stop Neo4j container
    read -p "Stop Neo4j container? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker stop neo4j-finagent 2>/dev/null || true
        echo -e "${YELLOW}Neo4j container stopped${NC}"
    fi
}

# Function to show status
show_status() {
    echo -e "${BLUE}📊 Service Status${NC}"
    echo "==================="
    
    check_service 7687 "Neo4j" || true
    check_service 7474 "Neo4j Browser" || true
    check_service $MEMORY_AGENT_PORT "Memory Agent" || true
    check_service $ALPHA_POOL_PORT "Alpha Pool" || true
    
    echo
    echo -e "${BLUE}🔗 Service URLs:${NC}"
    echo "Neo4j Browser: http://localhost:7474"
    echo "Memory Agent: http://127.0.0.1:$MEMORY_AGENT_PORT"
    echo "Alpha Pool: http://127.0.0.1:$ALPHA_POOL_PORT"
}

# Main function
main() {
    case "${1:-setup}" in
        "setup")
            install_dependencies
            setup_neo4j
            start_memory_agent
            show_status
            echo -e "${GREEN}🎉 Setup completed! Run './setup_integration.sh test' to run tests.${NC}"
            ;;
        "test")
            run_integration_tests
            ;;
        "status")
            show_status
            ;;
        "cleanup")
            cleanup
            ;;
        "restart")
            cleanup
            sleep 2
            setup_neo4j
            start_memory_agent
            show_status
            ;;
        *)
            echo "Usage: $0 {setup|test|status|cleanup|restart}"
            echo ""
            echo "Commands:"
            echo "  setup    - Install dependencies and start services"
            echo "  test     - Run integration tests"
            echo "  status   - Show service status"
            echo "  cleanup  - Stop services and cleanup"
            echo "  restart  - Restart all services"
            exit 1
            ;;
    esac
}

# Handle Ctrl+C
trap cleanup INT

# Run main function
main "$@"
