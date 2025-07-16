import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from FinAgents.memory.data_memory_client import DataMemoryClient

@pytest.mark.asyncio
async def test_data_memory_client_methods():
    client = DataMemoryClient(agent_id="test_data_agent")
    dummy_details = {"field": "value"}
    dummy_keywords = ["test", "memory"]
    dummy_summary = "Test summary"

    # Patch call_mcp_tool to always return a dummy response
    with patch("FinAgents.memory.data_memory_client.call_mcp_tool", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = {"status": "success", "result": "ok"}

        # Test store_event
        result = await client.store_event(
            event_type="DATA_EVENT",
            summary=dummy_summary,
            keywords=dummy_keywords,
            details=dummy_details,
            log_level="INFO"
        )
        assert result["status"] == "success"

        # Test store_error
        result = await client.store_error(
            summary="Error summary",
            details=dummy_details
        )
        assert result["status"] == "success"

        # Test store_action
        result = await client.store_action(
            summary="Action summary",
            details=dummy_details
        )
        assert result["status"] == "success"

        # Test retrieve_events
        result = await client.retrieve_events(search_query="test", limit=5)
        assert result["status"] == "success"

        # Test retrieve_expanded
        result = await client.retrieve_expanded(search_query="test", limit=5)
        assert result["status"] == "success"

        # Test filter_events
        result = await client.filter_events(filters={"agent_id": "test_data_agent"}, limit=5)
        assert result["status"] == "success" 