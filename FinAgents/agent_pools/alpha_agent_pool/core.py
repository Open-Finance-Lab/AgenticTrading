# core.py
"""
Unified entry point for starting the Alpha Agent Pool MCP service.
This module manages the lifecycle and orchestration of multiple sub-agents within the AlphaAgentPool.
"""
import multiprocessing
import os
import yaml
import threading
import asyncio
from mcp.server.fastmcp import FastMCP
from agent_pools.alpha_agent_pool.schema.theory_driven_schema import MomentumAgentConfig
from agent_pools.alpha_agent_pool.agents.theory_driven.momentum_agent import MomentumAgent

class AlphaAgentPoolMCPServer:
    def __init__(self, host="0.0.0.0", port=5050):
        """
        Initialize the AlphaAgentPoolMCPServer instance.
        Args:
            host (str): Host address to bind the MCP server.
            port (int): Port number to bind the MCP server.
        """
        self.host = host
        self.port = port
        self.pool_server = FastMCP("AlphaAgentPoolMCPServer")
        self.agent_registry = {}  # agent_id -> (agent, process/thread)
        self.config_dir = os.path.join(os.path.dirname(__file__), "config")
        self._register_pool_tools()

    def _register_pool_tools(self):
        """
        Register management tools for the agent pool, including starting and listing sub-agents.
        """
        @self.pool_server.tool(name="start_agent", description="Start the specified sub-agent service.")
        def start_agent(agent_id: str) -> str:
            """
            Start a sub-agent by agent_id. If already started, returns a status message.
            Args:
                agent_id (str): Identifier of the agent to start.
            Returns:
                str: Status message indicating the result.
            """
            if agent_id in self.agent_registry:
                return f"Agent '{agent_id}' is already running."
            if agent_id == "momentum_agent":
                config, process = self._start_momentum_agent()
                self.agent_registry[agent_id] = process
                return f"Momentum agent started on port {config.execution.port}"
            return f"Unknown agent: {agent_id}"

        @self.pool_server.tool(name="list_agents", description="List all registered sub-agents.")
        def list_agents() -> list:
            """
            List all currently registered sub-agents in the pool.
            Returns:
                list: List of agent IDs.
            """
            return list(self.agent_registry.keys())

    def _start_momentum_agent(self):
        """
        Start the MomentumAgent as a separate process using its configuration file.
        Returns:
            Tuple[MomentumAgentConfig, multiprocessing.Process]: The agent configuration and process handle.
        """
        cfg_path = os.path.join(self.config_dir, "momentum.yaml")
        with open(cfg_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        config = MomentumAgentConfig(**cfg_dict)
        p = multiprocessing.Process(target=run_momentum_agent, args=(cfg_dict,), daemon=True)
        p.start()
        return config, p

    def run(self):
        """
        Start the MCP pool server and display registered tools.
        """
        print(f"[AlphaAgentPool] MCP pool server starting on {self.host}:{self.port} ...")
        self.pool_server.settings.host = self.host
        self.pool_server.settings.port = self.port
        print("=== Registered MCP Pool Tools ===")
        tools = asyncio.run(self.pool_server.list_tools())
        for tool in tools:
            print(f"- {tool.name}")
        self.pool_server.run(transport="sse")

def run_momentum_agent(config_dict):
    """
    Entrypoint for running a MomentumAgent in a separate process.
    Args:
        config_dict (dict): Configuration dictionary for the agent.
    """
    from agent_pools.alpha_agent_pool.schema.theory_driven_schema import MomentumAgentConfig
    from agent_pools.alpha_agent_pool.agents.theory_driven.momentum_agent import MomentumAgent
    config = MomentumAgentConfig(**config_dict)
    agent = MomentumAgent(config)
    agent.start_mcp_server(port=config.execution.port, host="0.0.0.0", transport="sse")

if __name__ == "__main__":
    # Script entry point: start the AlphaAgentPoolMCPServer
    pool = AlphaAgentPoolMCPServer(host="0.0.0.0", port=5050)
    pool.run()
