#!/bin/bash

# Test script for memory services
echo "🧪 Testing FinAgent Memory Services"
echo "=================================="

# Change to memory directory
cd /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/memory

# Test 1: Check script syntax
echo "📋 Test 1: Script syntax check"
if bash -n start_memory_services.sh; then
    echo "✅ Script syntax is valid"
else
    echo "❌ Script syntax error"
    exit 1
fi

# Test 2: Check invalid argument handling
echo ""
echo "📋 Test 2: Invalid argument handling"
output=$(bash start_memory_services.sh invalid 2>&1)
if echo "$output" | grep -q "Invalid service type"; then
    echo "✅ Invalid argument handling works"
else
    echo "❌ Invalid argument handling failed"
    exit 1
fi

# Test 3: Check help information
echo ""
echo "📋 Test 3: Help information"
output=$(bash start_memory_services.sh invalid 2>&1)
if echo "$output" | grep -q "memory/core.*Start Memory, MCP, and A2A services"; then
    echo "✅ Help information is correct"
else
    echo "❌ Help information is incorrect"
    exit 1
fi

# Test 4: Check conda environment detection
echo ""
echo "📋 Test 4: Conda environment detection"
if which conda > /dev/null 2>&1; then
    echo "✅ Conda is available"
else
    echo "❌ Conda is not available"
    exit 1
fi

# Test 5: Check agent environment exists
echo ""
echo "📋 Test 5: Agent environment check"
if conda info --envs | grep -q "agent"; then
    echo "✅ Agent environment exists"
else
    echo "❌ Agent environment does not exist"
    exit 1
fi

# Test 6: Check logs directory
echo ""
echo "📋 Test 6: Logs directory"
LOGS_DIR="/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/memory/logs"
if [ -d "$LOGS_DIR" ]; then
    echo "✅ Logs directory exists: $LOGS_DIR"
else
    echo "❌ Logs directory does not exist: $LOGS_DIR"
    exit 1
fi

# Test 7: Check port availability functions
echo ""
echo "📋 Test 7: Port availability checks"
ports=(8000 8001 8002)
for port in "${ports[@]}"; do
    if lsof -i :$port > /dev/null 2>&1; then
        echo "⚠️  Port $port is in use"
    else
        echo "✅ Port $port is available"
    fi
done

echo ""
echo "🎉 All tests passed! Script is ready for use."
echo ""
echo "📖 Usage examples:"
echo "   ./start_memory_services.sh memory  # Start memory + MCP + A2A"
echo "   ./start_memory_services.sh llm     # Start LLM research services"
echo "   ./start_memory_services.sh all     # Start all services"
