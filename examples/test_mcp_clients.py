#!/usr/bin/env python3
"""
MCP Client Test Script

Test the DataAgentPool and PolygonAgent MCP servers using the correct MCP client libraries.
"""

import asyncio
import json
from mcp import ClientSession
from mcp.client.sse import sse_client


async def test_data_agent_pool():
    """Test DataAgentPool MCP server at localhost:8001"""
    print("=== Testing DataAgentPool MCP Server ===")
    
    try:
        # Use SSE client for DataAgentPool
        async with sse_client("http://localhost:8001/sse", timeout=10) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                print("Initializing session...")
                await session.initialize()
                
                # List available tools
                print("Listing tools...")
                tools_result = await session.list_tools()
                print(f"Available tools: {[tool.name for tool in tools_result.tools]}")
                
                # Test health check
                print("Testing health check...")
                health_result = await session.call_tool("health_check", {})
                print(f"Health check result: {health_result.content}")
                
                # Test list agents
                print("Testing list agents...")
                agents_result = await session.call_tool("list_agents", {})
                print(f"Agents list: {agents_result.content}")
                
    except Exception as e:
        import traceback
        print(f"Error testing DataAgentPool: {e}")
        print(f"Full traceback: {traceback.format_exc()}")


async def test_polygon_agent():
    """Test PolygonAgent MCP server at localhost:8002"""
    print("\n=== Testing PolygonAgent MCP Server ===")
    
    try:
        # Use SSE client for PolygonAgent
        async with sse_client("http://localhost:8002/sse") as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()
                
                # List available tools
                tools_result = await session.list_tools()
                print(f"Available tools: {[tool.name for tool in tools_result.tools]}")
                
                # Test health check
                health_result = await session.call_tool("health_check", {})
                print(f"Health check result: {health_result.content}")
                
                # Test a simple market query
                query_result = await session.call_tool("process_market_query", {
                    "query": "Get daily data for AAPL from 2024-01-01 to 2024-01-10"
                })
                print(f"Market query result: {json.dumps(query_result.content, indent=2)}")
                
    except Exception as e:
        print(f"Error testing PolygonAgent: {e}")


async def test_coordinated_fetch():
    """Test fetching data via DataAgentPool coordinator"""
    print("\n=== Testing Coordinated Data Fetch ===")
    
    try:
        # Use SSE client for DataAgentPool
        async with sse_client("http://localhost:8001/sse") as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()
                
                # Test process_market_query via coordinator
                query_result = await session.call_tool("process_market_query", {
                    "query": "Get daily data for AAPL from 2024-01-01 to 2024-01-10"
                })
                print(f"Coordinated market query result: {json.dumps(query_result.content, indent=2)}")
                
                # Test fetch_market_data via coordinator
                fetch_result = await session.call_tool("fetch_market_data", {
                    "symbol": "AAPL",
                    "start": "2024-01-01",
                    "end": "2024-01-10",
                    "interval": "1d"
                })
                print(f"Coordinated fetch result: {json.dumps(fetch_result.content, indent=2)}")
                
    except Exception as e:
        print(f"Error testing coordinated fetch: {e}")


async def main():
    """Main test function"""
    print("Testing MCP Servers with proper MCP client libraries...")
    
    await test_data_agent_pool()
    await test_polygon_agent()
    await test_coordinated_fetch()


if __name__ == "__main__":
    asyncio.run(main())
