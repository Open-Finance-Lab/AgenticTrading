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
import json
import csv
from mcp.server.fastmcp import FastMCP
from agent_pools.alpha_agent_pool.schema.theory_driven_schema import MomentumAgentConfig
from agent_pools.alpha_agent_pool.agents.theory_driven.momentum_agent import MomentumAgent

class MemoryUnit:
    """
    MemoryUnit provides a simple key-value store with optional persistence to a local JSON file.
    On initialization, it can automatically load a static dataset from a CSV file if provided.
    If reset_on_init is True, the memory file will be cleared on each initialization.
    """
    def __init__(self, file_path, autoload_csv_path=None, reset_on_init=False):
        self.file_path = file_path
        if reset_on_init and os.path.exists(self.file_path):
            try:
                os.remove(self.file_path)
            except Exception:
                pass
        self._data = {}
        self._load()
        if autoload_csv_path:
            self._autoload_csv(autoload_csv_path)

    def _autoload_csv(self, csv_path):
        if not os.path.exists(csv_path):
            return
        try:
            with open(csv_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    date = row['timestamp']
                    close = row['close']
                    self.set(f"AAPL_close_{date}", close)
        except Exception as e:
            pass  # Optionally log error

    def _load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}
        else:
            self._data = {}

    def _save(self):
        with open(self.file_path, 'w') as f:
            json.dump(self._data, f, indent=2)

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value):
        self._data[key] = value
        self._save()

    def delete(self, key):
        if key in self._data:
            del self._data[key]
            self._save()

    def keys(self):
        return list(self._data.keys())

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
        # Automatically load static dataset into memory unit on startup, and reset memory file
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/cache/AAPL_2024-01-01_2024-01-31_1d.csv"))
        self.memory = MemoryUnit(os.path.join(os.path.dirname(__file__), "memory_unit.json"), autoload_csv_path=csv_path, reset_on_init=True)
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

        @self.pool_server.tool(name="get_memory", description="Get a value from the internal memory unit by key.")
        def get_memory(key: str):
            """
            Retrieve a value from the memory unit by key.
            Args:
                key (str): The key to retrieve.
            Returns:
                The value associated with the key, or None if not found.
            """
            return self.memory.get(key)

        @self.pool_server.tool(name="set_memory", description="Set a value in the internal memory unit by key.")
        def set_memory(key: str, value):
            """
            Set a value in the memory unit by key.
            Args:
                key (str): The key to set.
                value: The value to store.
            Returns:
                str: Status message.
            """
            self.memory.set(key, value)
            return "OK"

        @self.pool_server.tool(name="delete_memory", description="Delete a key from the internal memory unit.")
        def delete_memory(key: str):
            """
            Delete a key from the memory unit.
            Args:
                key (str): The key to delete.
            Returns:
                str: Status message.
            """
            self.memory.delete(key)
            return "OK"

        @self.pool_server.tool(name="list_memory_keys", description="List all keys in the internal memory unit.")
        def list_memory_keys():
            """
            List all keys currently stored in the memory unit.
            Returns:
                list: List of keys.
            """
            return self.memory.keys()

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
