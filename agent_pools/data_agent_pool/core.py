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

from agent_pools.data_agent_pool.registry import AGENT_REGISTRY, preload_default_agents
from agent_pools.data_agent_pool.memory_bridge import record_event
from agent_pools.data_agent_pool.agents.crypto.binance_agent import BinanceAgent
from agent_pools.data_agent_pool.agents.equity.mcp_adapter import MCPAdapter

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

class DataAgentPool:
    """
    DataAgentPool is the central orchestrator for managing the lifecycle, health, and unified access
    of all data agents in the FinAgent ecosystem. It exposes a set of MCP tools for dynamic agent
    initialization, lifecycle management, and health monitoring, enabling external orchestrators to 
    control agent services on demand.

    Key responsibilities:
    - Agent lifecycle management (init, start, stop, status)
    - Unified MCP protocol interface for orchestration
    - Intelligent request routing and error handling
    - Health monitoring and diagnostics
    """

    def __init__(self, pool_id: str):
        """
        Initialize a new DataAgentPool instance with the specified identifier.

        Args:
            pool_id (str): Unique identifier for this pool instance.

        Note:
            - Does NOT initialize or start any data agents by default.
            - Only the MCP server for the pool itself is started.
        """
        self.pool_id = pool_id
        self.agents = {
            "crypto": {},
            "equity": {},
            "news": {}
        }
        self.agent_threads = {}    # Tracks running agent MCP server threads
        self.agent_adapters = {}   # Tracks MCPAdapter instances
        self.agent_status = {}     # Tracks agent lifecycle status
        self.mcp = FastMCP(f"DataAgentPool-{pool_id}", stateless_http=True)
        logger.info(f"Initialized DataAgentPool with ID: {pool_id}")
        logger.info(f"AGENT_REGISTRY keys: {list(AGENT_REGISTRY.keys())}")
        self._register_mcp_endpoints()

    def _register_mcp_endpoints(self):
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
        Start the DataAgentPool MCP server (for standalone mode).
        """
        self.mcp.run(
            transport="streamable-http",
        )

    def get_fastapi_app(self) -> FastAPI:
        """
        返回挂载了 MCP 服务的 FastAPI 应用。
        """
        app = FastAPI(lifespan=self.lifespan)
        app.mount("/mcp", self.mcp.streamable_http_app())
        return app

pool = DataAgentPool("debug-pool")
app = pool.get_fastapi_app()
for route in app.routes:
    print("ROUTE:", route.path)
# ======= Start entry =======
if __name__ == "__main__":

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

