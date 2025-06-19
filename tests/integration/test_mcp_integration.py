import pytest
import asyncio
from httpx import AsyncClient
from agent_pools.data_agent_pool.core import DataAgentPool

@pytest.fixture
async def mcp_client():
    """Create test client for MCP server"""
    pool = DataAgentPool("test-integration")
    async with AsyncClient(base_url="http://test") as client:
        yield client

@pytest.mark.asyncio
class TestMCPIntegration:
    """Integration tests for MCP protocol endpoints"""

    async def test_fetch_data_endpoint(self, mcp_client):
        """Test the fetch_data MCP tool endpoint"""
        response = await mcp_client.post(
            "/mcp/tools/fetch_data",
            json={
                "symbol": "BTC/USDT",
                "start": "2024-01-01",
                "end": "2024-01-02",
                "interval": "1h"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "data" in data
        assert "metadata" in data

    async def test_heartbeat_resource(self, mcp_client):
        """Test the heartbeat MCP resource endpoint"""
        response = await mcp_client.get("/mcp/resources/heartbeat/crypto")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "active"
        assert "agent_count" in data
        assert "timestamp" in data