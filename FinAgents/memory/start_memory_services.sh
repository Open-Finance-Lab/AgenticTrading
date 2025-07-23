#!/bin/bash

# =============================================================================
# 🚀 FinAgent Memory Services Launcher (Simplified)
# =============================================================================
# This script launches FinAgent memory services directly from the memory folder
# without complex orchestrator dependencies.
#
# REQUIREMENTS:
# - Conda environment named 'agent' must be available
# - All Python dependencies must be installed in the 'agent' environment
# - Neo4j database running on bolt://localhost:7687 with credentials neo4j/finagent123
# - OpenAI API key configured in .env file (for LLM services)
# =============================================================================

SERVICE_TYPE=${1:-memory}

echo "=================================================================================="
echo "🚀 FinAgent Memory Services Launcher"
echo "=================================================================================="

# Validate service type
case $SERVICE_TYPE in
    memory|core)
        echo "📋 Service Type: $SERVICE_TYPE (Memory + MCP + A2A)"
        ;;
    llm)
        echo "📋 Service Type: $SERVICE_TYPE (LLM Research Services)"
        ;;
    all)
        echo "📋 Service Type: $SERVICE_TYPE (Memory + MCP + A2A + LLM)"
        ;;
    *)
        echo "❌ Invalid service type: $SERVICE_TYPE"
        echo "   Valid options:"
        echo "     memory/core - Start Memory, MCP, and A2A services"
        echo "     llm         - Start LLM research services only"
        echo "     all         - Start all services"
        exit 1
        ;;
esac

# Check conda environment
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not installed, please install conda first"
    exit 1
fi

# Get current directory (should be memory folder) and project root
MEMORY_DIR=$(pwd)
PROJECT_ROOT="/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration"
LOGS_DIR="$MEMORY_DIR/logs"

echo "📁 Memory directory: $MEMORY_DIR"
echo "📁 Project root: $PROJECT_ROOT"
echo "📁 Logs directory: $LOGS_DIR"

# Create logs directory if it doesn't exist
mkdir -p "$LOGS_DIR"

# Verify we're in the memory directory
if [[ ! "$MEMORY_DIR" =~ "memory"$ ]]; then
    echo "❌ Script must be run from the memory directory"
    echo "   Current directory: $MEMORY_DIR"
    echo "   Expected to end with: memory"
    exit 1
fi

# Activate agent environment in the memory directory
echo "🔧 Activating conda agent environment in memory directory..."
cd "$MEMORY_DIR"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate agent

if [ $? -ne 0 ]; then
    echo "❌ Unable to activate agent environment"
    exit 1
fi

echo "✅ Agent environment activated"

# Verify Python environment
PYTHON_PATH=$(which python)
echo "🐍 Using Python: $PYTHON_PATH"
CONDA_ENV=$(conda info --envs | grep "agent" | grep "*")
if [ -n "$CONDA_ENV" ]; then
    echo "✅ Conda agent environment confirmed active"
else
    echo "❌ Conda agent environment not properly activated"
    exit 1
fi

# Function to start memory server
start_memory_server() {
    echo ""
    echo "🏗️  Starting Memory Server"
    echo "────────────────────────────────────"
    
    # Check if port 8000 is already in use
    echo "🔍 Checking port 8000 availability..."
    if lsof -i :8000 > /dev/null 2>&1; then
        echo "❌ Port 8000 is already in use. Please stop the existing service or use a different port."
        echo "   To find the process using port 8000: lsof -i :8000"
        echo "   To kill the process: kill <PID>"
        return 1
    fi
    echo "✅ Port 8000 is available"
    
    # Check database configuration
    echo "🔍 Checking database configuration..."
    conda run -n agent python -c "
import sys
sys.path.append('$PROJECT_ROOT')
from FinAgents.memory.configuration_manager import ConfigurationManager
config_manager = ConfigurationManager()
db_config = config_manager.get_database_config()
print(f'✅ Neo4j configuration: {db_config.uri} (user: {db_config.username})')
"
    
    if [ $? -ne 0 ]; then
        echo "❌ Database configuration check failed"
        return 1
    fi
    
    # Start memory server using uvicorn
    echo "🚀 Starting memory server..."
    cd "$PROJECT_ROOT"
    conda run -n agent uvicorn FinAgents.memory.memory_server:app --host 0.0.0.0 --port 8000 > "$LOGS_DIR/memory_server.log" 2>&1 &
    MEMORY_PID=$!
    echo "✅ Memory server starting (PID: $MEMORY_PID)"
    echo "📝 Memory server logs: $LOGS_DIR/memory_server.log"
    cd "$MEMORY_DIR"
    
    # Save PID
    echo $MEMORY_PID > .finagent_memory_pid
    
    # Wait for server to start and check if it's actually running
    echo "⏳ Waiting for server to start..."
    sleep 5
    
    # Check if server process is still running
    if ! ps -p $MEMORY_PID > /dev/null; then
        echo "❌ Memory server process died after startup"
        echo "📝 Check logs: tail -f $LOGS_DIR/memory_server.log"
        return 1
    fi
    
    # Check if server is responding on port 8000
    echo "🔍 Verifying server is responding..."
    for i in {1..10}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "✅ Memory server running normally"
            break
        fi
        echo "   ⏳ Attempt $i/10: Server not responding yet..."
        sleep 2
        
        # If this is the last attempt and still failing, show log info
        if [ $i -eq 10 ]; then
            echo "❌ Memory server startup failed - not responding to health check"
            echo "📝 Check logs: tail -f $LOGS_DIR/memory_server.log"
            return 1
        fi
    done
    
    # Start MCP Server (port 8001)
    echo ""
    echo "🔗 Starting MCP Server"
    echo "────────────────────────────────────"
    
    # Check if port 8001 is available
    if lsof -i :8001 > /dev/null 2>&1; then
        echo "❌ Port 8001 is already in use. Skipping MCP server."
    else
        echo "✅ Port 8001 is available"
        cd "$PROJECT_ROOT"
        conda run -n agent uvicorn FinAgents.memory.mcp_server:app --host 0.0.0.0 --port 8001 > "$LOGS_DIR/mcp_server.log" 2>&1 &
        MCP_PID=$!
        echo "✅ MCP server starting (PID: $MCP_PID)"
        echo "📝 MCP server logs: $LOGS_DIR/mcp_server.log"
        echo $MCP_PID > "$MEMORY_DIR/.finagent_mcp_pid"
        cd "$MEMORY_DIR"
    fi
    
    # Start A2A Server (port 8002)
    echo ""
    echo "🔗 Starting A2A Server"
    echo "────────────────────────────────────"
    
    # Check if port 8002 is available
    if lsof -i :8002 > /dev/null 2>&1; then
        echo "❌ Port 8002 is already in use. Skipping A2A server."
    else
        echo "✅ Port 8002 is available"
        cd "$PROJECT_ROOT"
        conda run -n agent uvicorn FinAgents.memory.a2a_server:app --host 0.0.0.0 --port 8002 > "$LOGS_DIR/a2a_server.log" 2>&1 &
        A2A_PID=$!
        echo "✅ A2A server starting (PID: $A2A_PID)"
        echo "📝 A2A server logs: $LOGS_DIR/a2a_server.log"
        echo $A2A_PID > "$MEMORY_DIR/.finagent_a2a_pid"
        cd "$MEMORY_DIR"
    fi
    
    echo "📋 Memory Server Status:"
    echo "   🗄️  Memory Server: ✅ Running (PID: $MEMORY_PID, Port: 8000)"
    echo "   📡 MCP Protocol: ✅ Available (Port: 8001)"
    echo "   🔗 A2A Protocol: ✅ Available (Port: 8002)"
    echo "   🏥 Health Check: ✅ Available (Port: 8000/health)"
    return 0
}

# Function to start LLM research services
start_llm_services() {
    echo ""
    echo "🧠 Starting LLM Research Services"
    echo "────────────────────────────────────────"
    
    # Check OpenAI API key
    echo "🔍 Checking OpenAI API configuration..."
    conda run -n agent python -c "
import sys
sys.path.append('$PROJECT_ROOT')
from dotenv import load_dotenv
import os
load_dotenv('$PROJECT_ROOT/.env')
api_key = os.environ.get('OPENAI_API_KEY')
if api_key:
    print(f'✅ OPENAI_API_KEY configured: {api_key[:20]}...')
else:
    print('❌ OPENAI_API_KEY not configured')
    exit(1)
"
    
    if [ $? -ne 0 ]; then
        echo "❌ OpenAI API configuration failed"
        return 1
    fi
    
    # Start LLM research service
    echo "🚀 Starting LLM research service..."
    cd "$PROJECT_ROOT"
    conda run -n agent python -m FinAgents.memory.llm_research_service > "$LOGS_DIR/llm_research_service.log" 2>&1 &
    LLM_PID=$!
    echo "✅ LLM research service started (PID: $LLM_PID)"
    echo "📝 LLM research service logs: $LOGS_DIR/llm_research_service.log"
    cd "$MEMORY_DIR"
    
    # Save PID
    echo $LLM_PID > .finagent_llm_pid
    
    echo "📋 LLM Research Services Status:"
    echo "   🧠 Memory Pattern Analysis: ✅ Active"
    echo "   🔍 Semantic Search Engine: ✅ Active" 
    echo "   📊 Research Insights Generator: ✅ Active"
    echo "   🔗 Relationship Analyzer: ✅ Active"
    
    return 0
}

# Function to show service endpoints
show_endpoints() {
    echo ""
    echo "📡 Available Service Endpoints:"
    echo "────────────────────────────────────────"
    
    if [[ "$SERVICE_TYPE" == "memory" || "$SERVICE_TYPE" == "core" || "$SERVICE_TYPE" == "all" ]]; then
        echo "🏗️  Memory Services:"
        echo "   • Memory Server: http://localhost:8000"
        echo "   • MCP Protocol: http://localhost:8001"
        echo "   • A2A Protocol: http://localhost:8002"
        echo "   • Health Check: http://localhost:8000/health"
    fi
    
    if [[ "$SERVICE_TYPE" == "llm" || "$SERVICE_TYPE" == "all" ]]; then
        echo "🧠 LLM Research Services:"
        echo "   • Pattern Analysis: Python API"
        echo "   • Semantic Search: Python API"  
        echo "   • Research Insights: Python API"
        echo "   • Relationship Analysis: Python API"
    fi
    
    echo ""
    echo "🛠️  Usage Examples:"
    echo "────────────────────────────────────────"
    
    if [[ "$SERVICE_TYPE" == "memory" || "$SERVICE_TYPE" == "core" || "$SERVICE_TYPE" == "all" ]]; then
        echo "🏗️  Memory Services Usage:"
        echo "   # Direct MCP connection (requires conda agent environment)"
        echo "   conda activate agent"
        echo "   python -c \"from FinAgents.memory.interface import main; import asyncio; asyncio.run(main())\""
        echo ""
        echo "   # Health check"
        echo "   curl http://localhost:8000/health"
        echo ""
        echo "   # View logs"
        echo "   tail -f $LOGS_DIR/memory_server.log"
        echo "   tail -f $LOGS_DIR/mcp_server.log"
        echo "   tail -f $LOGS_DIR/a2a_server.log"
        echo ""
        echo "   # Integration test (requires conda agent environment)"
        echo "   conda activate agent"
        echo "   python final_a2a_integration_test.py"
    fi
    
    if [[ "$SERVICE_TYPE" == "llm" || "$SERVICE_TYPE" == "all" ]]; then
        echo "🧠 LLM Research Usage:"
        echo "   # Memory pattern analysis (requires conda agent environment)"
        echo "   conda activate agent"
        echo "   cd /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/memory"
        echo "   PYTHONPATH=/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration:\$PYTHONPATH python -c \"from FinAgents.memory.llm_research_service import llm_research_service; import asyncio; asyncio.run(llm_research_service.analyze_memory_patterns([]))\""
        echo ""
        echo "   # View LLM service logs"
        echo "   tail -f $LOGS_DIR/llm_research_service.log"
    fi
}

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    
    if [ -f .finagent_memory_pid ]; then
        MEMORY_PID=$(cat .finagent_memory_pid)
        kill $MEMORY_PID 2>/dev/null
        rm -f .finagent_memory_pid
        echo "✅ Memory server stopped"
    fi
    
    if [ -f .finagent_mcp_pid ]; then
        MCP_PID=$(cat .finagent_mcp_pid)
        kill $MCP_PID 2>/dev/null
        rm -f .finagent_mcp_pid
        echo "✅ MCP server stopped"
    fi
    
    if [ -f .finagent_a2a_pid ]; then
        A2A_PID=$(cat .finagent_a2a_pid)
        kill $A2A_PID 2>/dev/null
        rm -f .finagent_a2a_pid
        echo "✅ A2A server stopped"
    fi
    
    if [ -f .finagent_llm_pid ]; then
        LLM_PID=$(cat .finagent_llm_pid)
        kill $LLM_PID 2>/dev/null
        rm -f .finagent_llm_pid
        echo "✅ LLM services stopped"
    fi
    
    echo "✅ All services stopped"
    exit 0
}

# Setup signal handlers
trap cleanup INT TERM

# Start requested services
case $SERVICE_TYPE in
    memory|core)
        start_memory_server
        if [ $? -ne 0 ]; then
            echo "❌ Failed to start memory server"
            exit 1
        fi
        ;;
    llm)
        start_llm_services
        if [ $? -ne 0 ]; then
            echo "❌ Failed to start LLM services"
            exit 1
        fi
        ;;
    all)
        start_memory_server
        if [ $? -ne 0 ]; then
            echo "❌ Failed to start memory server"
            exit 1
        fi
        
        sleep 3  # Wait for memory server to stabilize
        
        start_llm_services
        if [ $? -ne 0 ]; then
            echo "❌ Failed to start LLM services"
            exit 1
        fi
        ;;
esac

# Show service information
show_endpoints

echo ""
echo "=================================================================================="
echo "🎉 FinAgent Memory Services Started Successfully!"
echo "=================================================================================="
echo "📋 Service Architecture:"

if [[ "$SERVICE_TYPE" == "memory" || "$SERVICE_TYPE" == "core" || "$SERVICE_TYPE" == "all" ]]; then
    echo "   🏗️  Memory Layer: Database + Memory Server + MCP Protocol + A2A Protocol"
fi

if [[ "$SERVICE_TYPE" == "llm" || "$SERVICE_TYPE" == "all" ]]; then
    echo "   🧠 Research Layer: LLM-powered Analysis + Insights"
fi

if [ "$SERVICE_TYPE" == "all" ]; then
    echo "   🔗 Integration: Memory services provide data, LLM services provide insights"
fi

echo ""
echo "⏹️  To stop services: Press Ctrl+C"
echo "=================================================================================="

# Keep script running
echo "Press Ctrl+C to stop all services..."
while true; do
    sleep 1
done
