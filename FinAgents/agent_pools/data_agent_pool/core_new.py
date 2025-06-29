# core.py
"""
Unified entry point for starting the Data Agent Pool MCP service.
This module manages the lifecycle and orchestration of data agents following the alpha agent pool architecture.
"""
import logging
import contextvars
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List
import asyncio
import os

from mcp.server.fastmcp import FastMCP
from FinAgents.agent_pools.data_agent_pool.registry import AGENT_REGISTRY, preload_default_agents
from FinAgents.agent_pools.data_agent_pool.memory_bridge import record_event
from FinAgents.agent_pools.data_agent_pool.agents.crypto.binance_agent import BinanceAgent
from FinAgents.agent_pools.data_agent_pool.agents.equity.polygon_agent import PolygonAgent

# Configure global logging with standardized format
logger = logging.getLogger("DataAgentPool")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '[%(asctime)s] %(levelname)s - %(name)s: %(message)s'
)
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

class DataAgentPoolMCPServer:
    """
    DataAgentPoolMCPServer is the central orchestrator for managing the lifecycle and unified access
    of all data agents in the FinAgent ecosystem. It exposes a set of MCP tools for direct agent
    interaction and orchestration, following the alpha agent pool architecture.

    Key responsibilities:
    - Direct agent tool exposure via MCP
    - Agent lifecycle management  
    - Unified data fetching interface
    - Health monitoring and diagnostics
    """

    def __init__(self, host="0.0.0.0", port=8000):
        """
        Initialize a new DataAgentPoolMCPServer instance.

        Args:
            host (str): Host address to bind the MCP server.
            port (int): Port number to bind the MCP server.
        """
        self.host = host
        self.port = port
        self.pool_server = FastMCP("DataAgentPoolMCPServer")
        self.agent_registry = {}  # agent_id -> agent_instance
        self.agent_status = {}    # agent_id -> status
        
        # Preload polygon agent specifically for this demo
        preload_default_agents("polygon_agent")
        for aid, agent in AGENT_REGISTRY.items():
            self.agent_registry[aid] = agent
            self.agent_status[aid] = "initialized"
        
        logger.info(f"Initialized DataAgentPoolMCPServer on {host}:{port}")
        logger.info(f"AGENT_REGISTRY keys: {list(AGENT_REGISTRY.keys())}")
        logger.info(f"Loaded agents: {list(self.agent_status.keys())}")
        
        self._register_pool_tools()

    def _register_pool_tools(self):
        """
        Register direct polygon agent tools on the pool's MCP server,
        following the alpha agent pool pattern.
        """
        # Get polygon agent instance
        polygon_agent = self.agent_registry.get("polygon_agent")
        
        @self.pool_server.tool(name="polygon_fetch_market_data", description="Fetch historical market data via Polygon agent")
        def polygon_fetch_market_data(symbol: str, start: str, end: str, interval: str = "1d") -> dict:
            """
            Fetch market data using polygon agent.
            Args:
                symbol: Stock symbol (e.g., 'AAPL')
                start: Start date (YYYY-MM-DD) 
                end: End date (YYYY-MM-DD)
                interval: Time interval (1d, 1h, etc.)
            """
            if not polygon_agent:
                return {"status": "error", "error": "Polygon agent not available"}
            try:
                df = polygon_agent.fetch(symbol=symbol, start=start, end=end, interval=interval)
                return {
                    "symbol": symbol,
                    "data": df.to_dict(orient="records"),
                    "status": "success", 
                    "count": len(df)
                }
            except Exception as e:
                return {"symbol": symbol, "status": "error", "error": str(e)}

        @self.pool_server.tool(name="polygon_process_intent", description="Process natural language queries via Polygon agent")
        async def polygon_process_intent(query: str) -> dict:
            """
            Process natural language market data requests via polygon agent.
            Args:
                query: Natural language query (e.g., "Get daily data for AAPL from 2024-01-01 to 2024-12-31")
            """
            if not polygon_agent:
                return {"status": "error", "error": "Polygon agent not available"}
            try:
                result = await polygon_agent.process_intent(query)
                return result
            except Exception as e:
                return {"status": "error", "error": str(e), "query": query}

        @self.pool_server.tool(name="polygon_get_company_info", description="Get company information via Polygon agent")
        def polygon_get_company_info(symbol: str) -> dict:
            """
            Get company information using polygon agent.
            Args:
                symbol: Stock symbol (e.g., 'AAPL')
            """
            if not polygon_agent:
                return {"status": "error", "error": "Polygon agent not available"}
            try:
                info = polygon_agent.get_company_info(symbol)
                return {"symbol": symbol, "company_info": info, "status": "success"}
            except Exception as e:
                return {"symbol": symbol, "status": "error", "error": str(e)}

        @self.pool_server.tool(name="list_agents", description="List all registered data agents.")
        def list_agents() -> list:
            """
            List all currently registered data agents in the pool.
            Returns:
                list: List of agent IDs with their status.
            """
            return [{"agent_id": aid, "status": self.agent_status.get(aid, "unknown")} 
                    for aid in self.agent_registry.keys()]

        @self.pool_server.tool(name="health_check", description="Health check for DataAgentPool MCP server")
        def health_check() -> dict:
            """
            Return the health status of the DataAgentPool MCP server and all managed agents.
            """
            return {
                "status": "ok",
                "timestamp": datetime.now().isoformat(),
                "agents": self.agent_status
            }

    def run(self):
        """
        Start the MCP pool server and display registered tools.
        """
        print(f"[DataAgentPool] MCP pool server starting on {self.host}:{self.port} ...")
        self.pool_server.settings.host = self.host
        self.pool_server.settings.port = self.port
        print("=== Registered MCP Pool Tools ===")
        tools = asyncio.run(self.pool_server.list_tools())
        for tool in tools:
            print(f"- {tool.name}")
        self.pool_server.run(transport="sse")

if __name__ == "__main__":
    # Script entry point: start the DataAgentPoolMCPServer
    pool = DataAgentPoolMCPServer(host="0.0.0.0", port=8000)
    pool.run()
