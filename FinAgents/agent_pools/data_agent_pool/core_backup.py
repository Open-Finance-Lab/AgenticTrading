# core.py

import logging
import contextvars
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from mcp.server.fastmcp import FastMCP
from fastapi import FastAPI, Request
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
import asyncio
import threading
import traceback
import os
import socket
import time

from FinAgents.agent_pools.data_agent_pool.registry import AGENT_REGISTRY, preload_default_agents
from FinAgents.agent_pools.data_agent_pool.memory_bridge import record_event
from FinAgents.agent_pools.data_agent_pool.agents.crypto.binance_agent import BinanceAgent
from FinAgents.agent_pools.data_agent_pool.agents.equity.mcp_adapter import MCPAdapter
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

# Global context management for request tracking
request_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "request_context", 
    default={}
)

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
        """
        Register MCP protocol tools for external orchestration and agent management.
        """

        @self.mcp.tool(name="init_agent", description="Initialize one or more data agents by agent_id(s)")
        def init_agent(agent_id: Any = None) -> dict:
            """
            Dynamically initialize one or more data agents (without starting their MCP servers).
            Args:
                agent_id: str, list[str], or None. If None, initialize all agents.
            """
            initialized = []
            already = []
            errors = {}

            # Normalize input: support str, list, or None
            if agent_id is None:
                agent_ids = None  # Signal to preload_default_agents to load all
            elif isinstance(agent_id, str):
                agent_ids = [agent_id]
            elif isinstance(agent_id, list):
                agent_ids = agent_id
            else:
                return {"status": "error", "error": "agent_id must be str, list, or None"}

            try:
                if agent_ids is None:
                    # Initialize all agents
                    preload_default_agents()
                    # Register all to self.agents
                    for aid, agent in AGENT_REGISTRY.items():
                        if self._is_agent_initialized(aid):
                            already.append(aid)
                        else:
                            agent_type = self._determine_agent_type(agent)
                            self.agents[agent_type][aid] = agent
                            self.agent_status[aid] = "initialized"
                            initialized.append(aid)
                else:
                    for aid in agent_ids:
                        if self._is_agent_initialized(aid):
                            already.append(aid)
                            continue
                        try:
                            preload_default_agents(aid)
                            agent = AGENT_REGISTRY.get(aid)
                            if not agent:
                                errors[aid] = "not found in registry"
                                continue
                            agent_type = self._determine_agent_type(agent)
                            self.agents[agent_type][aid] = agent
                            self.agent_status[aid] = "initialized"
                            initialized.append(aid)
                        except Exception as e:
                            logger.error(f"Failed to initialize agent {aid}: {e}")
                            logger.error(traceback.format_exc())
                            errors[aid] = str(e)
                logger.info(f"Initialized agents: {initialized}, already: {already}, errors: {errors}")
                return {
                    "status": "ok",
                    "initialized": initialized,
                    "already_initialized": already,
                    "errors": errors
                }
            except Exception as e:
                logger.error(f"Failed to initialize agents: {e}")
                logger.error(traceback.format_exc())
                return {"status": "error", "error": str(e)}

        @self.mcp.tool(name="start_agent_mcp", description="Start the MCPAdapter server for a given agent")
        def start_agent_mcp(agent_id: str, port: int = None) -> dict:
            """
            Start the MCPAdapter server for the specified agent (on a dedicated port).
            """
            if self.agent_status.get(agent_id) == "running":
                return {"status": "already_running"}
            try:
                agent = self._get_agent_instance(agent_id)
                if not agent:
                    return {"status": "error", "error": f"Agent {agent_id} not initialized"}
                adapter = MCPAdapter(agent, name=f"{agent_id}-MCP")
                self.agent_adapters[agent_id] = adapter
                def run_adapter():
                    try:
                        if port:
                            os.environ["PORT"] = str(port)
                        adapter.run()
                    except Exception as e:
                        logger.error(f"Exception in MCPAdapter thread for {agent_id}: {e}")
                        logger.error(traceback.format_exc())
                t = threading.Thread(target=run_adapter, daemon=True)
                t.start()
                self.agent_threads[agent_id] = t

                # Health Check: Wait up to 2 seconds, check if the port is listening
                if port:
                    for _ in range(10):
                        time.sleep(0.2)
                        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        try:
                            s.settimeout(0.2)
                            s.connect(("127.0.0.1", port))
                            s.close()
                            self.agent_status[agent_id] = "running"
                            logger.info(f"Started MCP server for {agent_id} on port {port}")
                            return {"status": "started"}
                        except Exception:
                            s.close()
                    logger.error(f"Failed to start MCP server for {agent_id} on port {port}: port not open")
                    return {"status": "error", "error": f"Port {port} not open after start"}
                else:
                    self.agent_status[agent_id] = "running"
                    logger.info(f"Started MCP server for {agent_id} on default port")
                    return {"status": "started"}
            except Exception as e:
                logger.error(f"Failed to start MCP server for {agent_id}: {e}")
                logger.error(traceback.format_exc())
                return {"status": "error", "error": str(e)}

        @self.mcp.tool(name="stop_agent_mcp", description="Stop the MCPAdapter server for a given agent")
        def stop_agent_mcp(agent_id: str) -> dict:
            """
            Attempt to stop the MCPAdapter server for the specified agent.
            """
            # Note: Actual thread/process termination is non-trivial in Python.
            # Here we only update status for demonstration.
            if self.agent_status.get(agent_id) != "running":
                return {"status": "not_running"}
            self.agent_status[agent_id] = "stopped"
            logger.info(f"Marked MCP server for {agent_id} as stopped (manual intervention may be required)")
            return {"status": "stopped"}

        @self.mcp.tool(name="list_agents", description="List all initialized agents and their status")
        def list_agents() -> dict:
            """
            List all initialized agents and their current lifecycle status.
            """
            return {
                "crypto": {k: self.agent_status.get(k, "uninitialized") for k in self.agents["crypto"].keys()},
                "equity": {k: self.agent_status.get(k, "uninitialized") for k in self.agents["equity"].keys()},
                "news": {k: self.agent_status.get(k, "uninitialized") for k in self.agents["news"].keys()}
            }

        @self.mcp.tool(name="agent_status", description="Get the status of a specific agent")
        def agent_status(agent_id: str) -> dict:
            """
            Get the current lifecycle status of the specified agent.
            """
            status = self.agent_status.get(agent_id, "uninitialized")
            return {"agent_id": agent_id, "status": status}

        @self.mcp.tool(name="health_check", description="Health check for DataAgentPool MCP server")
        def health_check() -> dict:
            """
            Return the health status of the DataAgentPool MCP server and all managed agents.
            """
            return {
                "status": "ok",
                "timestamp": datetime.now().isoformat(),
                "agents": self.agent_status
            }

    def _register_agent_tools(self):
        """
        Register individual agent tools directly on the pool's MCP server.
        This allows direct access to agent functionality without separate MCP servers.
        """
        # Register polygon agent tools
        polygon_agent = self.agents.get("equity", {}).get("polygon_agent")
        if polygon_agent:
            self._register_polygon_agent_tools(polygon_agent)

    def _register_polygon_agent_tools(self, polygon_agent: PolygonAgent):
        """
        Register polygon agent's tools on the pool's MCP server.
        """
        @self.mcp.tool(name="polygon_fetch_market_data", description="Fetch historical market data via Polygon agent")
        def polygon_fetch_market_data(symbol: str, start: str, end: str, interval: str = "1d") -> dict:
            """
            Fetch market data using polygon agent.
            Args:
                symbol: Stock symbol (e.g., 'AAPL')
                start: Start date (YYYY-MM-DD)
                end: End date (YYYY-MM-DD)
                interval: Time interval (1d, 1h, etc.)
            """
            try:
                df = polygon_agent.fetch(symbol=symbol, start=start, end=end, interval=interval)
                # Convert DataFrame to dict for JSON serialization
                return {
                    "symbol": symbol,
                    "data": df.to_dict(orient="records"),
                    "status": "success",
                    "count": len(df)
                }
            except Exception as e:
                return {
                    "symbol": symbol,
                    "status": "error",
                    "error": str(e)
                }

        @self.mcp.tool(name="polygon_process_intent", description="Process natural language queries via Polygon agent")
        async def polygon_process_intent(query: str) -> dict:
            """
            Process natural language market data requests via polygon agent.
            Args:
                query: Natural language query (e.g., "Get daily data for AAPL from 2024-01-01 to 2024-12-31")
            """
            try:
                result = await polygon_agent.process_intent(query)
                return result
            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e),
                    "query": query
                }

        @self.mcp.tool(name="polygon_get_company_info", description="Get company information via Polygon agent")
        def polygon_get_company_info(symbol: str) -> dict:
            """
            Get company information using polygon agent.
            Args:
                symbol: Stock symbol (e.g., 'AAPL')
            """
            try:
                info = polygon_agent.get_company_info(symbol)
                return {
                    "symbol": symbol,
                    "company_info": info,
                    "status": "success"
                }
            except Exception as e:
                return {
                    "symbol": symbol,
                    "status": "error",
                    "error": str(e)
                }

    def _is_agent_initialized(self, agent_id: str) -> bool:
        """
        Check if an agent is already initialized.
        """
        for group in self.agents.values():
            if agent_id in group:
                return True
        return False

    def _get_agent_instance(self, agent_id: str):
        """
        Retrieve an agent instance by agent_id from all agent groups.
        """
        for group in self.agents.values():
            if agent_id in group:
                return group[agent_id]
        return None

    def _determine_agent_type(self, agent: Any) -> str:
        """
        Determine the appropriate category for a given agent instance.
        """
        if isinstance(agent, BinanceAgent):
            return "crypto"
        elif isinstance(agent, PolygonAgent):
            return "equity"
        # Add more agent type checks as needed
        return "equity"

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """
        Manage the lifecycle of the DataAgentPool MCP server for FastAPI.
        """
        logger.info(f"Starting DataAgentPool {self.pool_id}")
        try:
            yield
        finally:
            logger.info(f"Shutting down DataAgentPool {self.pool_id}")

    def run(self):
        """
        Start the DataAgentPool MCP server (standalone mode using FastMCP's built-in server).
        """
        print(f"[DataAgentPool] Starting MCP server on 0.0.0.0:8000...")
        print("=== Loaded Agents ===")
        for agent_id, status in self.agent_status.items():
            print(f"- {agent_id}: {status}")
        print("=== Registered MCP Tools ===")
        import asyncio
        tools = asyncio.run(self.mcp.list_tools())
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")
        self.mcp.run(
            transport="streamable-http",
            host="0.0.0.0",
            port=8000
        )

    def get_fastapi_app(self) -> FastAPI:
        """
        返回挂载了 MCP 服务的 FastAPI 应用。
        """
        app = FastAPI(lifespan=self.lifespan)
        app.mount("/mcp", self.mcp.streamable_http_app())
        return app

pool = DataAgentPool("debug-pool")
# ======= Start entry =======
if __name__ == "__main__":
    # Use FastMCP's built-in server directly instead of FastAPI
    pool.run()

