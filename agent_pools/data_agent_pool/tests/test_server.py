import pytest
import asyncio
import httpx

MCP_URL = "http://localhost:8000/mcp/"

@pytest.mark.asyncio
async def test_init_agent():
    """测试初始化 polygon_agent"""
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "init_agent",
            "arguments": {"agent_id": "polygon_agent"}
        },
        "id": 1
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(MCP_URL, json=payload, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["result"]["status"] == "ok"
        assert "polygon_agent" in data["result"]["initialized"]

@pytest.mark.asyncio
async def test_list_agents():
    """测试获取所有 agent 状态"""
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "list_agents",
            "arguments": {}
        },
        "id": 2
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(MCP_URL, json=payload, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "crypto" in data["result"]
        assert "equity" in data["result"]

@pytest.mark.asyncio
async def test_agent_status():
    """测试查询 polygon_agent 状态"""
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "agent_status",
            "arguments": {"agent_id": "polygon_agent"}
        },
        "id": 3
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(MCP_URL, json=payload, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["result"]["agent_id"] == "polygon_agent"
        assert data["result"]["status"] in ("initialized", "running")

@pytest.mark.asyncio
async def test_health_check():
    """测试健康检查"""
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "health_check",
            "arguments": {}
        },
        "id": 4
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(MCP_URL, json=payload, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["result"]["status"] == "ok"
        assert "agents" in data["result"]