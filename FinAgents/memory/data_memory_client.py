import asyncio
from typing import List, Dict, Any, Optional
from FinAgents.memory.interface import call_mcp_tool
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

MCP_SERVER_URL = "http://127.0.0.1:8010/mcp"

class DataMemoryClient:
    """
    Client for interacting with the memory agent via the MCP server, for data agent pool use.
    Provides methods to store and retrieve structured events, errors, and actions.
    """
    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    async def store_event(self, event_type: str, summary: str, keywords: List[str], details: Dict[str, Any], log_level: str = "INFO", session_id: Optional[str] = None, correlation_id: Optional[str] = None):
        tool_args = {
            "query": str(details),
            "keywords": keywords,
            "summary": summary,
            "agent_id": self.agent_id,
            "event_type": event_type,
            "log_level": log_level,
            "session_id": session_id,
            "correlation_id": correlation_id,
        }
        async with streamablehttp_client(MCP_SERVER_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                return await call_mcp_tool(session, "store_graph_memory", tool_args)

    async def store_error(self, summary: str, details: Dict[str, Any], keywords: List[str] = None, session_id: Optional[str] = None, correlation_id: Optional[str] = None):
        return await self.store_event(
            event_type="ERROR",
            summary=summary,
            keywords=keywords or ["error"],
            details=details,
            log_level="ERROR",
            session_id=session_id,
            correlation_id=correlation_id,
        )

    async def store_action(self, summary: str, details: Dict[str, Any], keywords: List[str] = None, session_id: Optional[str] = None, correlation_id: Optional[str] = None):
        return await self.store_event(
            event_type="AGENT_ACTION",
            summary=summary,
            keywords=keywords or ["action"],
            details=details,
            log_level="INFO",
            session_id=session_id,
            correlation_id=correlation_id,
        )

    async def retrieve_events(self, search_query: str, limit: int = 10):
        tool_args = {
            "search_query": search_query,
            "limit": limit
        }
        async with streamablehttp_client(MCP_SERVER_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                return await call_mcp_tool(session, "retrieve_graph_memory", tool_args)

    async def retrieve_expanded(self, search_query: str, limit: int = 10):
        tool_args = {
            "search_query": search_query,
            "limit": limit
        }
        async with streamablehttp_client(MCP_SERVER_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                return await call_mcp_tool(session, "retrieve_memory_with_expansion", tool_args)

    async def filter_events(self, filters: Dict[str, Any], limit: int = 100, offset: int = 0):
        tool_args = {
            "filters": filters,
            "limit": limit,
            "offset": offset
        }
        async with streamablehttp_client(MCP_SERVER_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                return await call_mcp_tool(session, "filter_graph_memories", tool_args) 