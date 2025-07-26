#!/bin/bash

# Enhanced Alpha Agent Pool Restart Script with SSE Error Handling

echo "🔄 Restarting Alpha Agent Pool with SSE connection fixes..."

# Kill existing Alpha Agent Pool processes
echo "🛑 Stopping existing Alpha Agent Pool processes..."
pkill -f "alpha_agent_pool" || echo "No existing processes found"
pkill -f "AlphaAgentPoolMCPServer" || echo "No MCP server processes found"

# Wait for processes to fully terminate
sleep 3

# Change to project directory
cd /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration

# Start Alpha Agent Pool with enhanced error handling
echo "🚀 Starting Alpha Agent Pool with SSE error handling..."
nohup bash ./FinAgents/agent_pools/alpha_agent_pool/start_alpha_pool.sh > logs/alpha_agent_pool_restart.log 2>&1 &

# Wait for startup
sleep 5

# Check if the service is running
if pgrep -f "alpha_agent_pool" > /dev/null; then
    echo "✅ Alpha Agent Pool restarted successfully with SSE error handling"
    echo "📊 Service is running on port 8081"
    echo "📋 Logs available in logs/alpha_agent_pool_restart.log"
else
    echo "❌ Failed to restart Alpha Agent Pool"
    echo "📋 Check logs/alpha_agent_pool_restart.log for details"
    exit 1
fi

echo "🎯 Ready for momentum agent testing with improved SSE stability"
