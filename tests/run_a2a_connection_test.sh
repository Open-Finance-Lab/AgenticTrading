#!/bin/bash
"""
Quick A2A Connection Test Runner

This script provides a convenient way to run the Alpha Agent Pool and Memory Agent
A2A protocol connection test. It handles service startup verification and test execution.

Usage:
    ./tests/run_a2a_connection_test.sh

Prerequisites:
- Conda environment 'agent' must be available
- All required Python packages installed
- Neo4j database running
- OpenAI API key configured (if testing LLM services)

Author: FinAgent Test Team
Created: 2025-07-22
"""

echo "🔗 FinAgent A2A Connection Test Runner"
echo "========================================="

# Get project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

echo "📁 Project root: $PROJECT_ROOT"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install conda first."
    exit 1
fi

# Activate agent environment
echo "🔧 Activating conda agent environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate agent

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate agent environment"
    exit 1
fi

echo "✅ Agent environment activated"

# Check if services are running
echo ""
echo "🔍 Checking service status..."
echo "================================"

# Function to check if a service is running
check_service() {
    local service_name="$1"
    local url="$2"
    
    if curl -s --connect-timeout 3 "$url" > /dev/null 2>&1; then
        echo "✅ $service_name is running"
        return 0
    else
        echo "❌ $service_name is not responding at $url"
        return 1
    fi
}

# Check required services
all_services_running=true

if ! check_service "Memory Server" "http://localhost:8000/health"; then
    all_services_running=false
fi

if ! check_service "Memory A2A Server" "http://localhost:8002/health"; then
    all_services_running=false
fi

if ! check_service "Alpha Agent Pool" "http://localhost:8081/health"; then
    all_services_running=false
fi

if [ "$all_services_running" = false ]; then
    echo ""
    echo "⚠️  Some services are not running. Please start them first:"
    echo "   1. Memory services: cd $PROJECT_ROOT/FinAgents/memory && ./start_memory_services.sh all"
    echo "   2. Alpha Agent Pool: cd $PROJECT_ROOT/tests && ./start_agent_pools.sh"
    echo ""
    read -p "Do you want to continue anyway? (y/N): " continue_anyway
    if [[ ! "$continue_anyway" =~ ^[Yy]$ ]]; then
        echo "❌ Test cancelled. Please start the required services first."
        exit 1
    fi
fi

echo ""
echo "🚀 Starting A2A connection test..."
echo "=================================="

# Change to project root and run the test
cd "$PROJECT_ROOT"

# Set PYTHONPATH to include project root
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Run the test with Python
python tests/test_alpha_memory_a2a_connection.py

test_exit_code=$?

echo ""
echo "📋 Test Summary"
echo "==============="

if [ $test_exit_code -eq 0 ]; then
    echo "✅ A2A connection test completed successfully!"
    echo "📁 Check logs directory for detailed results: $PROJECT_ROOT/logs/"
else
    echo "❌ A2A connection test failed with exit code: $test_exit_code"
    echo "📁 Check logs directory for detailed error information: $PROJECT_ROOT/logs/"
fi

echo ""
echo "🔗 Test completed. Exit code: $test_exit_code"

exit $test_exit_code
