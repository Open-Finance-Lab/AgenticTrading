#!/usr/bin/env python3
"""
Test script for DataAgentPool's integrated agent management functionality.

This script tests:
1. Listing agents and their status
2. Starting/stopping/restarting agents
3. Health monitoring
4. Data fetching after agent restart
"""

import asyncio
import json
from mcp.client.sse import sse_client
from mcp import ClientSession

DATA_AGENT_POOL_URL = "http://localhost:8001/sse"

async def call_pool_tool(tool_name: str, arguments: dict = None) -> dict:
    """Call a tool on the DataAgentPool."""
    if arguments is None:
        arguments = {}
        
    try:
        async with sse_client(DATA_AGENT_POOL_URL, timeout=30) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                
                if result.content and len(result.content) > 0:
                    content_item = result.content[0]
                    if hasattr(content_item, 'text'):
                        return json.loads(content_item.text)
                
                return {"status": "error", "error": "No content in response"}
                
    except Exception as e:
        return {"status": "error", "error": str(e)}

async def test_list_agents():
    """Test listing agents and their status."""
    print("=== Testing Agent Listing ===")
    result = await call_pool_tool("list_agents")
    
    if result.get("status") != "error":
        print(f"✅ Found {result.get('total_agents', 0)} agents:")
        for agent_id, info in result.get("agents", {}).items():
            print(f"  📊 {agent_id}:")
            print(f"     Endpoint: {info.get('endpoint')}")
            print(f"     Port: {info.get('port')}")
            if info.get("process"):
                proc_info = info["process"]
                status = "🟢 Running" if proc_info.get("running") else "🔴 Stopped"
                print(f"     Status: {status} (PID: {proc_info.get('pid', 'N/A')})")
            else:
                print(f"     Status: 🔴 Not managed")
    else:
        print(f"❌ Failed to list agents: {result.get('error')}")
    
    return result

async def test_health_check():
    """Test pool and agent health check."""
    print("\n=== Testing Health Check ===")
    result = await call_pool_tool("health_check")
    
    if result.get("status") != "error":
        print(f"✅ Pool status: {result.get('pool_status')}")
        print(f"   Timestamp: {result.get('timestamp')}")
        
        agents = result.get("agents", {})
        for agent_id, health in agents.items():
            status = health.get("status", "unknown")
            emoji = "🟢" if status == "ok" else "🔴" if status == "error" else "🟡"
            print(f"   {emoji} {agent_id}: {status}")
            if status == "error":
                print(f"      Error: {health.get('error', 'Unknown')}")
    else:
        print(f"❌ Health check failed: {result.get('error')}")
    
    return result

async def test_agent_restart():
    """Test stopping and starting an agent."""
    agent_id = "polygon_agent"
    
    print(f"\n=== Testing Agent Restart ({agent_id}) ===")
    
    # Stop the agent
    print(f"🛑 Stopping {agent_id}...")
    stop_result = await call_pool_tool("stop_agent", {"agent_id": agent_id})
    
    if stop_result.get("status") == "success":
        print(f"✅ {stop_result.get('message')}")
    else:
        print(f"❌ Stop failed: {stop_result.get('message', 'Unknown error')}")
    
    # Wait a moment
    await asyncio.sleep(2)
    
    # Start the agent
    print(f"🚀 Starting {agent_id}...")
    start_result = await call_pool_tool("start_agent", {"agent_id": agent_id})
    
    if start_result.get("status") == "success":
        print(f"✅ {start_result.get('message')}")
    else:
        print(f"❌ Start failed: {start_result.get('message', 'Unknown error')}")
    
    # Wait for startup
    await asyncio.sleep(3)
    
    return start_result

async def test_data_fetch_after_restart():
    """Test fetching data after agent restart."""
    print("\n=== Testing Data Fetch After Restart ===")
    
    query = "Get daily price data for AAPL for the last 5 days"
    result = await call_pool_tool("process_market_query", {"query": query})
    
    if result.get("status") == "success":
        print("✅ Data fetch successful after restart")
        if "result" in result:
            data = result["result"]
            if isinstance(data, dict) and "data" in data:
                data_points = len(data["data"])
                print(f"   Retrieved {data_points} data points")
            elif isinstance(data, list):
                print(f"   Retrieved {len(data)} data points")
        else:
            print("   Data structure varies")
    else:
        print(f"❌ Data fetch failed: {result.get('error', 'Unknown error')}")
    
    return result

async def main():
    """Run all agent management tests."""
    print("🧪 Testing DataAgentPool Agent Management Features")
    print("=" * 60)
    
    try:
        # Test 1: List agents
        await test_list_agents()
        
        # Test 2: Health check
        await test_health_check()
        
        # Test 3: Agent restart
        await test_agent_restart()
        
        # Test 4: Health check after restart
        await test_health_check()
        
        # Test 5: Data fetch after restart
        await test_data_fetch_after_restart()
        
        print("\n🎉 All agent management tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
