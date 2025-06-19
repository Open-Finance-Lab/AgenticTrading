import pytest
import asyncio
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

MCP_URL = "http://localhost:8000/mcp/"

@pytest.mark.asyncio
async def test_init_agent():
    async with streamablehttp_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("init_agent", {"agent_id": "polygon_agent"})
            assert result["status"] == "ok"
            assert "polygon_agent" in result["initialized"]

@pytest.mark.asyncio
async def test_list_agents():
    async with streamablehttp_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("list_agents")
            assert "crypto" in result
            assert "equity" in result

@pytest.mark.asyncio
async def test_agent_status():
    async with streamablehttp_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("agent_status", {"agent_id": "polygon_agent"})
            assert result["agent_id"] == "polygon_agent"
            assert result["status"] in ("initialized", "running")

@pytest.mark.asyncio
async def test_health_check():
    async with streamablehttp_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("health_check")
            assert result["status"] == "ok"
            assert "agents" in result