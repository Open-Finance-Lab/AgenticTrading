# core.py
"""
Unified entry point for starting the Alpha Agent Pool MCP service.
This module manages the lifecycle and orchestration of multiple sub-agents within the AlphaAgentPool.
Enhanced with comprehensive memory integration for strategy tracking and performance analytics.
"""
import os
import sys
import json
import time
import yaml
import logging
import asyncio
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, TypedDict, TYPE_CHECKING
from enum import Enum

from mcp.server.fastmcp import FastMCP

# Add project root to sys.path to ensure correct module resolution
# This is crucial for making sure imports like `FinAgents.memory...` work correctly
# when the script is run directly or as a subprocess.
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Try to import langgraph for the planner, but don't fail if it's not there
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    StateGraph = None
    END = None
    LANGGRAPH_AVAILABLE = False

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from FinAgents.agent_pools.alpha_agent_pool.memory_bridge import AlphaAgentPoolMemoryBridge, create_alpha_memory_bridge
    from FinAgents.memory.external_memory_interface import ExternalMemoryAgent, EventType, LogLevel

# --- Definitions for DAG Planner and Agent Status ---
# These are defined here but only used within the server class.
# This avoids polluting the module's top-level namespace too much.

class AgentStatus(Enum):
    """Enumeration for agent status."""
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"


class PlannerState(TypedDict):
    """Represents the state of the planner."""
    command: str
    agent_status: Dict[str, AgentStatus]
    recovery_action: Optional[str]
    result: str


class CommandPlanner:
    """
    A DAG-based planner to decode and execute external commands,
    monitor agent status, and perform self-regulation.
    """
    def __init__(self, pool_server: "AlphaAgentPoolMCPServer"):
        self.pool_server = pool_server
        if LANGGRAPH_AVAILABLE:
            self.workflow = self._build_graph()
        else:
            self.workflow = None
            logger.warning("langgraph not found. CommandPlanner will be disabled.")

    def _build_graph(self):
        """Builds the langgraph StateGraph."""
        workflow = StateGraph(PlannerState)

        workflow.add_node("parse_command", self.parse_command)
        workflow.add_node("check_agent_status", self.check_agent_status)
        workflow.add_node("execute_action", self.execute_action)
        workflow.add_node("perform_recovery", self.perform_recovery)

        workflow.set_entry_point("parse_command")
        workflow.add_edge("parse_command", "check_agent_status")
        workflow.add_conditional_edges(
            "check_agent_status",
            self.decide_next_step,
            {
                "execute": "execute_action",
                "recover": "perform_recovery",
            }
        )
        workflow.add_edge("execute_action", END)
        workflow.add_edge("perform_recovery", "check_agent_status") # Re-check status after recovery

        return workflow.compile()

    def parse_command(self, state: PlannerState) -> PlannerState:
        """Parses the initial command."""
        logger.info(f"Parsing command: {state['command']}")
        # In a real scenario, this would involve more complex NLP parsing.
        # For now, we'll assume the command is simple, e.g., "start_agent:momentum_agent"
        return state

    def check_agent_status(self, state: PlannerState) -> PlannerState:
        """Checks the status of all managed agents."""
        logger.info("Checking agent status...")
        status = self.pool_server.check_all_agents_status()
        state["agent_status"] = status
        
        # Simple self-regulation logic: if any agent is stopped, plan a recovery.
        if any(s == AgentStatus.STOPPED for s in status.values()):
            state["recovery_action"] = "restart_stopped_agents"
        else:
            state["recovery_action"] = None
        return state

    def decide_next_step(self, state: PlannerState) -> str:
        """Decides the next step based on agent status."""
        if state.get("recovery_action"):
            logger.warning("Agent issue detected. Planning recovery.")
            return "recover"
        logger.info("All agents are healthy. Proceeding with execution.")
        return "execute"

    def execute_action(self, state: PlannerState) -> PlannerState:
        """Executes the parsed command."""
        command = state["command"].strip()
        logger.info(f"Executing action: {command}")
        
        # This is a simplified execution logic
        try:
            if command == "list agents":
                # Correctly call the sync method on the pool_server instance
                agents = self.pool_server.list_agents_sync()
                if agents:
                    state["result"] = f"Running agents: {', '.join(agents)}"
                else:
                    state["result"] = "No agents are currently running."
            elif command.startswith("start agent"):
                agent_id = command.split(" ", 2)[-1]
                state["result"] = self.pool_server.start_agent_sync(agent_id)
            elif command == "check status":
                status_dict = self.pool_server.check_all_agents_status()
                state["result"] = json.dumps({k: v.name for k, v in status_dict.items()})
            else:
                state["result"] = f"Unknown command: {command}"
        except Exception as e:
            logger.error(f"Error executing command '{command}': {e}")
            state["result"] = f"Error: {e}"
        return state

    def perform_recovery(self, state: PlannerState) -> PlannerState:
        """Performs a recovery action."""
        action = state.get("recovery_action")
        logger.info(f"Performing recovery action: {action}")
        if action == "restart_stopped_agents":
            result = self.pool_server.restart_stopped_agents()
            state["result"] = f"Recovery attempted: {result}"
        
        # Clear the recovery action to avoid loops
        state["recovery_action"] = None
        return state

    async def run(self, command: str) -> Dict:
        """Runs the planner with a given command."""
        if not self.workflow:
            logger.warning("Planner not available, cannot run command.")
            return {"error": "Planner is not available due to missing dependencies."}
        initial_state = PlannerState(command=command, agent_status={}, recovery_action=None, result="")
        final_state = await self.workflow.ainvoke(initial_state)
        return final_state


class MemoryUnit:
    """
    MemoryUnit provides a simple key-value store with optional persistence to a local JSON file.
    On initialization, it can automatically load a static dataset from a CSV file if provided.
    If reset_on_init is True, the memory file will be cleared on each initialization.
    Enhanced with event logging capabilities for strategy flow tracking.
    """
    def __init__(self, file_path, autoload_csv_path=None, reset_on_init=False, memory_bridge=None):
        self.file_path = file_path
        self.memory_bridge = memory_bridge
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
            logger.warning(f"CSV file not found for autoloading: {csv_path}")
            return
        try:
            import csv  # Import csv module here
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Assuming 'timestamp' and 'close' columns exist
                    timestamp = row.get('timestamp')
                    close_price = row.get('close')
                    if timestamp and close_price:
                        # Create a unique key for each price entry
                        key = f"AAPL_close_{timestamp.split(' ')[0]}" # Use date as part of key
                        self.set(key, close_price, log_event=False) # Avoid logging for bulk load
            logger.info(f"Successfully autoloaded data from {csv_path}")
            # Asynchronously log the data loading event
            if self.memory_bridge:
                asyncio.create_task(self._log_data_loading_event(csv_path, len(self._data)))
        except Exception as e:
            logger.error(f"Failed to autoload CSV {csv_path}: {e}")

    async def _log_data_loading_event(self, csv_path: str, records_loaded: int):
        """Log data loading events to memory bridge"""
        if self.memory_bridge and hasattr(self.memory_bridge, '_log_system_event'):
            try:
                await self.memory_bridge._log_system_event(
                    event_type=EventType.SYSTEM if MEMORY_AVAILABLE else "system",
                    log_level=LogLevel.INFO if MEMORY_AVAILABLE else "info",
                    title="Historical Data Loaded",
                    content=f"Loaded {records_loaded} historical data records from {os.path.basename(csv_path)}",
                    metadata={
                        "source_file": csv_path,
                        "records_loaded": records_loaded,
                        "data_type": "historical_market_data"
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to log data loading event: {e}")

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

    def set(self, key, value, log_event=True):
        self._data[key] = value
        self._save()
        
        # Log memory operation if enabled
        if log_event and self.memory_bridge:
            asyncio.create_task(self._log_memory_operation("SET", key, value))

    def delete(self, key, log_event=True):
        if key in self._data:
            del self._data[key]
            self._save()
            
            # Log memory operation if enabled
            if log_event and self.memory_bridge:
                asyncio.create_task(self._log_memory_operation("DELETE", key, None))

    def keys(self):
        return list(self._data.keys())

    async def _log_memory_operation(self, operation: str, key: str, value: Any):
        """Log memory unit operations to memory bridge"""
        if self.memory_bridge and hasattr(self.memory_bridge, '_log_system_event'):
            try:
                await self.memory_bridge._log_system_event(
                    event_type=EventType.SYSTEM if MEMORY_AVAILABLE else "system",
                    log_level=LogLevel.DEBUG if MEMORY_AVAILABLE else "debug",
                    title=f"Memory Unit Operation: {operation}",
                    content=f"Performed {operation} operation on key '{key}'",
                    metadata={
                        "operation": operation,
                        "key": key,
                        "value_type": type(value).__name__ if value is not None else None,
                        "component": "local_memory_unit"
                    }
                )
            except Exception as e:
                logger.debug(f"Failed to log memory operation: {e}")

class AlphaAgentPoolMCPServer:
    def __init__(self, host="0.0.0.0", port=8081):
        """
        Initialize the AlphaAgentPoolMCPServer instance with enhanced memory capabilities.
        Args:
            host (str): Host address to bind the MCP server.
            port (int): Port number to bind the MCP server.
        """
        self.host = host
        self.port = port
        self.pool_server = FastMCP("AlphaAgentPoolMCPServer")
        self.agent_registry = {}  # agent_id -> (agent, process/thread)
        self.config_dir = os.path.join(os.path.dirname(__file__), "config")
        self.logger = logging.getLogger(__name__) # Add logger attribute
        
        # Initialize the Command Planner for DAG-based execution
        self.planner = CommandPlanner(self)
        
        # Initialize memory bridge for comprehensive strategy tracking
        self.memory_bridge: Optional["AlphaAgentPoolMemoryBridge"] = None
        # Note: Memory bridge will be initialized asynchronously when needed
        
        # Automatically load static dataset into memory unit on startup, and reset memory file
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/cache/AAPL_2022-01-01_2024-12-31_1d.csv"))
        self.memory = MemoryUnit(
            os.path.join(os.path.dirname(__file__), "memory_unit.json"), 
            autoload_csv_path=csv_path, 
            reset_on_init=True,
            memory_bridge=self.memory_bridge
        )
        
        # Initialize legacy memory agent if available
        self.memory_agent: Optional["ExternalMemoryAgent"] = None
        self.session_id = None
        self._initialize_memory_agent()  # Initialize memory agent synchronously
        
        # Strategy performance tracking
        self.strategy_performance_cache = {}
        self.signal_generation_history = []
        
        self._register_pool_tools()

    def _start_momentum_agent(self):
        """
        Starts the momentum agent as a separate process by correctly importing and instantiating it.
        """
        # Import agent-specific components here to avoid top-level import errors
        from FinAgents.agent_pools.alpha_agent_pool.schema.theory_driven_schema import MomentumAgentConfig
        from multiprocessing import Process

        config_path = os.path.join(self.config_dir, "momentum.yaml")
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        config = MomentumAgentConfig(**config_data)
        
        # Use a helper function to be the target of the process
        process = Process(target=run_momentum_agent_process, args=(config,))
        process.start()
        return config, process

    def _start_autonomous_agent(self):
        """
        Starts the autonomous agent as a separate process.
        """
        # Import agent-specific components here
        from FinAgents.agent_pools.alpha_agent_pool.agents.autonomous.autonomous_agent import run_autonomous_agent
        from multiprocessing import Process

        process = Process(target=run_autonomous_agent)
        process.start()
        return process

    async def _log_agent_lifecycle_event(self, agent_id: str, status: str, details: str):
        """Log agent lifecycle events to the memory bridge."""
        if self.memory_bridge:
            from FinAgents.memory.external_memory_interface import EventType, LogLevel
            try:
                await self.memory_bridge._log_system_event(
                    event_type=EventType.SYSTEM,
                    log_level=LogLevel.INFO,
                    title=f"Agent {status}: {agent_id}",
                    content=details,
                    metadata={
                        "agent_id": agent_id,
                        "status": status,
                        "details": details
                    }
                )
            except Exception as e:
                self.logger.warning(f"Failed to log agent lifecycle event: {e}")

    def _start_agent_sync(self, agent_id: str) -> str:
        """
        Synchronous version of start_agent logic.
        """
        if agent_id in self.agent_registry:
            # Check if the process is alive, if not, remove it before restarting
            process = self.agent_registry[agent_id]
            if (hasattr(process, 'is_alive') and process.is_alive()) or \
               (hasattr(process, 'poll') and process.poll() is None):
                return f"Agent '{agent_id}' is already running."
            else:
                self.logger.warning(f"Found dead process for agent '{agent_id}'. Removing before restart.")
                del self.agent_registry[agent_id]

        if agent_id == "momentum_agent":
            config, process = self._start_momentum_agent()
            self.agent_registry[agent_id] = process
            
            details = f"Momentum agent started on port {config.execution.port}"
            # Log agent startup event
            if self.memory_bridge:
                # Run the async logging function in a new event loop as we are in a sync context
                try:
                    asyncio.run(self._log_agent_lifecycle_event(
                        agent_id, "STARTED", details
                    ))
                except RuntimeError: # If a loop is already running
                    loop = asyncio.get_event_loop()
                    loop.create_task(self._log_agent_lifecycle_event(agent_id, "STARTED", details))

            return details
        elif agent_id == "autonomous_agent":
            process = self._start_autonomous_agent()
            self.agent_registry[agent_id] = process
            
            details = "Autonomous agent started on port 5051"
            # Log agent startup event
            if self.memory_bridge:
                # Run the async logging function in a new event loop as we are in a sync context
                try:
                    asyncio.run(self._log_agent_lifecycle_event(
                        agent_id, "STARTED", details
                    ))
                except RuntimeError:
                    loop = asyncio.get_event_loop()
                    loop.create_task(self._log_agent_lifecycle_event(agent_id, "STARTED", details))
            
            return details
        return f"Unknown agent: {agent_id}"

    def check_agent_status(self, agent_id: str) -> AgentStatus:
        """Checks the status of a single agent process."""
        if agent_id not in self.agent_registry:
            return AgentStatus.UNKNOWN
        
        process = self.agent_registry[agent_id]
        if hasattr(process, 'is_alive') and process.is_alive():
            return AgentStatus.RUNNING
        elif hasattr(process, 'poll') and process.poll() is None:
            return AgentStatus.RUNNING
        else:
            return AgentStatus.STOPPED

    def check_all_agents_status(self) -> Dict[str, AgentStatus]:
        """Checks and returns the status of all registered agents."""
        return {agent_id: self.check_agent_status(agent_id) for agent_id in self.agent_registry.keys()}

    def restart_stopped_agents(self) -> str:
        """Scans for and attempts to restart stopped agents."""
        restarted = []
        for agent_id, status in self.check_all_agents_status().items():
            if status == AgentStatus.STOPPED:
                logger.info(f"Agent '{agent_id}' is stopped. Attempting to restart.")
                try:
                    # Remove the dead process from registry before restarting
                    if agent_id in self.agent_registry:
                        del self.agent_registry[agent_id]
                    self.start_agent_sync(agent_id)
                    restarted.append(agent_id)
                except Exception as e:
                    logger.error(f"Failed to restart agent '{agent_id}': {e}")
        if restarted:
            return f"Restarted agents: {', '.join(restarted)}"
        return "No stopped agents found to restart."

    def _initialize_memory_bridge(self):
        """Initialize the advanced memory bridge for alpha agent pool"""
        # This method is now synchronous and returns the bridge instance or None
        # Import here to avoid top-level circular dependencies
        from FinAgents.agent_pools.alpha_agent_pool.memory_bridge import create_alpha_memory_bridge, MEMORY_BRIDGE_AVAILABLE

        if not MEMORY_BRIDGE_AVAILABLE:
            self.logger.warning("Memory bridge not available. Continuing without it.")
            return None
        
        bridge = None
        try:
            # create_alpha_memory_bridge is an async function, so we need a loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # This is tricky in a sync method. A better approach is to ensure
                    # the entire startup is async, but for now, we'll try to work around it.
                    # This might not be robust.
                    future = asyncio.run_coroutine_threadsafe(create_alpha_memory_bridge(config={...}), loop)
                    bridge = future.result(timeout=10)
                else:
                    bridge = asyncio.run(create_alpha_memory_bridge(config={...}))
            except RuntimeError: # No event loop
                 bridge = asyncio.run(create_alpha_memory_bridge(config={
                    "enable_pattern_learning": True,
                    "performance_tracking_enabled": True,
                    "real_time_logging": True
                }))

            self.logger.info("Alpha Agent Pool Memory Bridge successfully initialized")
            self.memory.memory_bridge = bridge # Ensure memory unit has the bridge
        except Exception as e:
            self.logger.error(f"Failed to initialize memory bridge: {e}")
        
        return bridge

    def _initialize_memory_agent(self):
        """Initialize the external memory agent"""
        # Import here to avoid top-level circular dependencies
        try:
            from FinAgents.memory.external_memory_interface import ExternalMemoryAgent
            MEMORY_AVAILABLE = True
        except ImportError:
            MEMORY_AVAILABLE = False

        if not MEMORY_AVAILABLE:
            logger.warning("External memory agent not available. Continuing without it.")
            return
        
        try:
            self.memory_agent = ExternalMemoryAgent()
            self.session_id = f"alpha_pool_session_{int(time.time())}"
            logger.info("External memory agent initialized for Alpha Agent Pool")
        except Exception as e:
            logger.error(f"Failed to initialize memory agent: {e}")
            self.memory_agent = None

    async def _log_memory_event(self, event_type, log_level, title: str, content: str, 
                               tags: set = None, metadata: Optional[Dict[str, Any]] = None):
        """Log an event to the memory agent with proper enum types"""
        # Import here to ensure it's available
        from FinAgents.memory.external_memory_interface import EventType, LogLevel

        if self.memory_agent and self.session_id:
            try:
                await self.memory_agent.log_event(
                    event_type=event_type,
                    log_level=log_level,
                    source_agent_pool="alpha_agent_pool",
                    source_agent_id="alpha_agent_pool_server",
                    title=title,
                    content=content,
                    tags=tags or set(),
                    metadata={
                        "session_id": self.session_id,
                        "agent_pool": "alpha",
                        **(metadata or {})
                    },
                    session_id=self.session_id
                )
            except Exception as e:
                logger.warning(f"Failed to log memory event: {e}")

    def _register_pool_tools(self):
        """
        Register management tools for the agent pool, including starting and listing sub-agents.
        Enhanced with comprehensive memory bridge integration and strategy flow tracking.
        """
        @self.pool_server.tool(name="start_agent", description="Start the specified sub-agent service.")
        def start_agent(agent_id: str) -> str:
            return self.start_agent_sync(agent_id)

        @self.pool_server.tool(name="list_agents", description="List all registered sub-agents.")
        def list_agents() -> list:
            return self.list_agents_sync()
            
        # Add sync versions for internal calls from planner
        self.start_agent_sync = self._start_agent_sync
        self.list_agents_sync = lambda: list(self.agent_registry.keys())

        @self.pool_server.tool(name="get_agent_status", description="Get the status of all agents.")
        def get_agent_status() -> Dict[str, str]:
            """Returns the status of all registered agents as a dictionary."""
            return {k: v.name for k, v in self.check_all_agents_status().items()}

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

        @self.pool_server.tool(name="generate_alpha_signals", description="Generate alpha signals based on market data")
        async def generate_alpha_signals(symbol: str = None, symbols: list = None, date: str = None, lookback_period: int = 20, price: float = None, memory: dict = None) -> dict:
            """
            Call the momentum_agent's generate_signal tool via SSE, passing historical price data.
            Supports both single symbol (symbol) and multiple symbols (symbols) parameters for compatibility.
            """
            try:
                from mcp.client.sse import sse_client
                from mcp import ClientSession
                from datetime import datetime, timedelta
                
                # Handle both symbol and symbols parameters for backward compatibility
                target_symbols = []
                if symbol:
                    target_symbols = [symbol]
                elif symbols:
                    target_symbols = symbols if isinstance(symbols, list) else [symbols]
                else:
                    return {"status": "error", "message": "Either 'symbol' or 'symbols' parameter is required"}
                
                # If no date provided, use a default date from memory
                if not date:
                    date = "2024-01-30"  # Default date
                
                results = {}
                
                for target_symbol in target_symbols:
                    logger.info(f"Processing symbol: {target_symbol}")
                    # The rest of the implementation from the original function follows...
                    # Prepare price list for the lookback period
                    price_list = []
                    current_price = price

                    # If price is not provided, get it from memory
                    if current_price is None:
                        # Use the memory unit if memory dict is not provided
                        memory_source = memory if memory is not None else self.memory
                        price_key = f"{target_symbol}_close_{date}"
                        current_price_str = memory_source.get(price_key) if hasattr(memory_source, 'get') else memory_source.get(price_key) if memory_source else None
                        if not current_price_str:
                            logger.warning(f"No price data for {target_symbol} on {date}, using default price")
                            current_price = 100.0  # Default price if not found
                        else:
                            current_price = float(current_price_str)

                    # Collect historical prices for the lookback period
                    current_date_obj = datetime.strptime(date, '%Y-%m-%d')
                    for i in range(lookback_period):
                        past_date = current_date_obj - timedelta(days=i+1)
                        past_date_str = past_date.strftime('%Y-%m-%d')
                        price_key = f"{target_symbol}_close_{past_date_str}"
                        memory_source = memory if memory is not None else self.memory
                        if memory_source:
                            past_price_str = memory_source.get(price_key) if hasattr(memory_source, 'get') else memory_source.get(price_key) if isinstance(memory_source, dict) else None
                            if past_price_str:
                                try:
                                    price_list.append(float(past_price_str))
                                except (ValueError, TypeError):
                                    continue

                    # The agent expects prices in chronological order, so we reverse the list
                    price_list.reverse()
                    # Add the current price to the end of the list
                    price_list.append(current_price)

                    logger.info(f"Calling momentum_agent with {len(price_list)} prices for symbol {target_symbol}")

                    # SSE client to momentum_agent (correct port 5051 from logs)
                    base_url = "http://127.0.0.1:5051/sse"
                    
                    # Initialize with a default error response
                    response = {"signal": "HOLD", "confidence": 0.0, "error": "Unknown error"}
                    
                    # Try the MCP SSE client first, then fallback to simple HTTP client
                    try:
                        # Add better error handling and logging
                        logger.info(f"Attempting SSE connection to {base_url} for symbol {target_symbol}")
                        
                        # Use a more explicit approach to handle the SSE connection
                        try:
                            async with sse_client(base_url, timeout=10) as (read, write):
                                logger.info(f"SSE connection established for {target_symbol}")
                                try:
                                    async with ClientSession(read, write) as session:
                                        try:
                                            await session.initialize()
                                            logger.info(f"SSE session initialized for {target_symbol}")
                                            
                                            # Create proper request parameters for momentum agent
                                            request_params = {
                                                "symbol": target_symbol, 
                                                "price_list": price_list
                                            }
                                            
                                            logger.info(f"Calling generate_signal tool for {target_symbol} with {len(price_list)} prices")
                                            try:
                                                response_parts = await session.call_tool(
                                                    "generate_signal",
                                                    request_params
                                                )
                                            except Exception as tool_error:
                                                logger.error(f"Error during session.call_tool for {target_symbol}: {tool_error}", exc_info=True)
                                                raise tool_error
                                            
                                            if response_parts and hasattr(response_parts, 'content') and response_parts.content:
                                                content_list = response_parts.content
                                                if len(content_list) > 0 and hasattr(content_list[0], 'text'):
                                                    content_str = content_list[0].text
                                                    try:
                                                        response = json.loads(content_str)
                                                    except json.JSONDecodeError:
                                                        response = {"signal": "HOLD", "confidence": 0.0, "raw_response": content_str}
                                                else:
                                                    response = {"signal": "HOLD", "confidence": 0.0, "error": "No text in response"}
                                            else:
                                                response = {"signal": "HOLD", "confidence": 0.0, "error": "Empty response"}
                                        except Exception as session_error:
                                            logger.error(f"Error in session operations for {target_symbol}: {session_error}", exc_info=True)
                                            raise session_error
                                except Exception as client_session_error:
                                    logger.error(f"Error creating ClientSession for {target_symbol}: {client_session_error}", exc_info=True)
                                    raise client_session_error
                        except Exception as sse_client_error:
                            logger.error(f"Error in SSE client for {target_symbol}: {sse_client_error}", exc_info=True)
                            raise sse_client_error
                    
                    except Exception as mcp_error:
                        logger.warning(f"MCP SSE client failed for {target_symbol}: {mcp_error}")
                        logger.info(f"Attempting fallback to simple HTTP client for {target_symbol}")
                        
                        # Fallback to simple HTTP client
                        try:
                            from .simple_momentum_client import SimpleMomentumClient
                            simple_client = SimpleMomentumClient()
                            response = await simple_client.call_generate_signal(target_symbol, price_list)
                            logger.info(f"Fallback HTTP client succeeded for {target_symbol}")
                        except Exception as fallback_error:
                            logger.error(f"Fallback HTTP client also failed for {target_symbol}: {fallback_error}")
                            response = {"signal": "HOLD", "confidence": 0.0, "error": f"Both MCP and HTTP failed: {mcp_error}"}

                    logger.info(f"Received response from momentum_agent for {target_symbol}: {response}")
                    results[target_symbol] = response
                    
                return {
                    "status": "success",
                    "alpha_signals": {
                        "signals": results
                    }
                }
            except Exception as top_level_error:
                logger.error(f"Top-level error in generate_alpha_signals: {top_level_error}", exc_info=True)
                import traceback
                error_traceback = traceback.format_exc()
                logger.error(f"Full top-level traceback: {error_traceback}")
                
                return {
                    "status": "error",
                    "message": f"Top-level error: {top_level_error}",
                    "alpha_signals": {
                        "signals": {}
                    }
                }
                try:
                    # Prepare price list for the lookback period
                    price_list = []
                    current_price = price

                    # If price is not provided, get it from memory
                    if current_price is None:
                        # Use the memory unit if memory dict is not provided
                        memory_source = memory if memory is not None else self.memory
                        price_key = f"{target_symbol}_close_{date}"
                        current_price_str = memory_source.get(price_key) if hasattr(memory_source, 'get') else memory_source.get(price_key) if memory_source else None
                        if not current_price_str:
                            logger.warning(f"No price data for {target_symbol} on {date}, using default price")
                            current_price = 100.0  # Default price if not found
                        else:
                            current_price = float(current_price_str)

                    # Collect historical prices for the lookback period
                    current_date_obj = datetime.strptime(date, '%Y-%m-%d')
                    for i in range(lookback_period):
                        past_date = current_date_obj - timedelta(days=i+1)
                        past_date_str = past_date.strftime('%Y-%m-%d')
                        price_key = f"{target_symbol}_close_{past_date_str}"
                        memory_source = memory if memory is not None else self.memory
                        if memory_source:
                            past_price_str = memory_source.get(price_key) if hasattr(memory_source, 'get') else memory_source.get(price_key) if isinstance(memory_source, dict) else None
                            if past_price_str:
                                try:
                                    price_list.append(float(past_price_str))
                                except (ValueError, TypeError):
                                    continue

                    # The agent expects prices in chronological order, so we reverse the list
                    price_list.reverse()
                    # Add the current price to the end of the list
                    price_list.append(current_price)

                    logger.info(f"Calling momentum_agent with {len(price_list)} prices for symbol {target_symbol}")

                    # SSE client to momentum_agent (correct port 5051 from logs)
                    base_url = "http://127.0.0.1:5051/sse"
                    
                    try:
                        # Add better error handling and logging
                        logger.info(f"Attempting SSE connection to {base_url} for symbol {target_symbol}")
                        
                        # Use a more explicit approach to handle the SSE connection
                        try:
                            async with sse_client(base_url, timeout=10) as (read, write):
                                logger.info(f"SSE connection established for {target_symbol}")
                                try:
                                    async with ClientSession(read, write) as session:
                                        try:
                                            await session.initialize()
                                            logger.info(f"SSE session initialized for {target_symbol}")
                                            
                                            # Create proper request parameters for momentum agent
                                            request_params = {
                                                "symbol": target_symbol, 
                                                "price_list": price_list
                                            }
                                            
                                            logger.info(f"Calling generate_signal tool for {target_symbol} with {len(price_list)} prices")
                                            try:
                                                response_parts = await session.call_tool(
                                                    "generate_signal",
                                                    request_params
                                                )
                                            except Exception as tool_error:
                                                logger.error(f"Error during session.call_tool for {target_symbol}: {tool_error}", exc_info=True)
                                                raise tool_error
                                            
                                            if response_parts and hasattr(response_parts, 'content') and response_parts.content:
                                                content_list = response_parts.content
                                                if len(content_list) > 0 and hasattr(content_list[0], 'text'):
                                                    content_str = content_list[0].text
                                                    try:
                                                        response = json.loads(content_str)
                                                    except json.JSONDecodeError:
                                                        response = {"signal": "HOLD", "confidence": 0.0, "raw_response": content_str}
                                                else:
                                                    response = {"signal": "HOLD", "confidence": 0.0, "error": "No text in response"}
                                            else:
                                                response = {"signal": "HOLD", "confidence": 0.0, "error": "Empty response"}
                                        except Exception as session_error:
                                            logger.error(f"Error in session operations for {target_symbol}: {session_error}", exc_info=True)
                                            raise session_error
                                except Exception as client_session_error:
                                    logger.error(f"Error creating ClientSession for {target_symbol}: {client_session_error}", exc_info=True)
                                    raise client_session_error
                        except Exception as sse_client_error:
                            logger.error(f"Error in SSE client for {target_symbol}: {sse_client_error}", exc_info=True)
                            raise sse_client_error
                    
                    except asyncio.TimeoutError:
                        logger.error(f"Timeout connecting to momentum_agent for {target_symbol}")
                        response = {"signal": "HOLD", "confidence": 0.0, "error": "Timeout"}
                    except ConnectionError as e:
                        logger.error(f"Connection error to momentum_agent for {target_symbol}: {e}")
                        response = {"signal": "HOLD", "confidence": 0.0, "error": f"Connection error: {e}"}
                    except Exception as e:
                        logger.error(f"Unexpected error calling momentum_agent for {target_symbol}: {e}")
                        
                        # Detailed error investigation for TaskGroup errors
                        import traceback
                        error_traceback = traceback.format_exc()
                        logger.error(f"Full traceback for {target_symbol}: {error_traceback}")
                        
                        # Attempt to extract inner exception details for TaskGroup errors
                        if hasattr(e, '__cause__') and e.__cause__:
                            logger.error(f"Root cause for {target_symbol}: {e.__cause__}")
                        if hasattr(e, '__context__') and e.__context__:
                            logger.error(f"Context for {target_symbol}: {e.__context__}")
                        if hasattr(e, 'exceptions'):
                            logger.error(f"TaskGroup exceptions for {target_symbol}: {e.exceptions}")
                            
                        response = {"signal": "HOLD", "confidence": 0.0, "error": f"Unexpected error: {e}"}

                    logger.info(f"Received response from momentum_agent for {target_symbol}: {response}")
                    results[target_symbol] = response
                    
                except Exception as e:
                    logger.error(f"Error calling momentum_agent for {target_symbol}: {e}", exc_info=True)
                    results[target_symbol] = {"status": "error", "message": str(e), "signal": "HOLD", "confidence": 0.0}

            return {
                "status": "success",
                "alpha_signals": {
                    "signals": results
                }
            }
        
        @self.pool_server.tool(name="process_strategy_request", description="Process strategy requests and generate alpha signals")
        async def process_strategy_request(query: str) -> str:
            """
            Process natural language strategy requests and generate alpha signals.
            This now uses the DAG planner to execute commands.
            
            Args:
                query (str): Natural language query describing the strategy request
                
            Returns:
                str: JSON string containing strategy response and alpha signals
            """
            try:
                logger.info(f"Received strategy request, routing to planner: '{query}'")
                # The query is treated as a command for the planner
                planner_result = await self.planner.run(command=query)
                
                # Log the outcome
                if self.memory_bridge:
                    # Import enums here for logging
                    from FinAgents.memory.external_memory_interface import EventType, LogLevel
                    asyncio.create_task(self.memory_bridge._log_system_event(
                        event_type=EventType.SYSTEM,
                        log_level=LogLevel.INFO,
                        title="Planner Execution Completed",
                        content=f"Planner processed command: {query}",
                        metadata={"result": planner_result.get("result")}
                    ))
                
                # Ensure the output is always a JSON string
                output = planner_result.get("result", "No result")
                return json.dumps({
                    "status": "success",
                    "planner_output": output
                })
                
            except Exception as e:
                logger.error(f"Error processing strategy request with planner: {e}")
                # Log the error
                if self.memory_bridge:
                    from FinAgents.memory.external_memory_interface import EventType, LogLevel
                    asyncio.create_task(self.memory_bridge._log_system_event(
                        event_type=EventType.SYSTEM,
                        log_level=LogLevel.ERROR,
                        title="Planner Execution Failed",
                        content=f"Planner failed on command '{query}': {e}",
                        metadata={"error": str(e)}
                    ))
                return json.dumps({"status": "error", "message": str(e)})

        @self.pool_server.tool(name="submit_strategy_event", description="Submit strategy flow events to memory system for tracking and analysis.")
        async def submit_strategy_event(event_type: str, strategy_id: str, event_data: dict, 
                                      metadata: Optional[dict] = None) -> str:
            """
            Submit a strategy flow event to the memory bridge for tracking.
            
            Args:
                event_type: Type of strategy event (SIGNAL_GENERATED, STRATEGY_EXECUTED, PERFORMANCE_UPDATED, etc.)
                strategy_id: Unique identifier for the strategy
                event_data: Core event data including signals, performance metrics, etc.
                metadata: Additional contextual information
            
            Returns:
                str: Event submission confirmation with storage ID
            """
            if not self.memory_bridge:
                logger.warning("Memory bridge not available, cannot submit strategy event.")
                return "Memory bridge not available"
            
            try:
                # Import memory schemas here
                # Correcting the import path for schema
                from FinAgents.agent_pools.alpha_agent_pool.memory_bridge import StrategyEventData, SignalGeneratedEvent, PerformanceUpdatedEvent

                # Determine the event type and structure the data accordingly
                if event_type == "SIGNAL_GENERATED":
                    event_data_structured = SignalGeneratedEvent(**event_data)
                elif event_type == "STRATEGY_EXECUTED":
                    event_data_structured = StrategyEventData(**event_data)
                elif event_type == "PERFORMANCE_UPDATED":
                    event_data_structured = PerformanceUpdatedEvent(**event_data)
                else:
                    logger.warning(f"Unknown event type: {event_type}. Data: {event_data}")
                    return "Unknown event type"

                # Submit the event to the memory bridge
                storage_id = await self.memory_bridge.submit_event(
                    event_type=event_type,
                    strategy_id=strategy_id,
                    event_data=event_data_structured,
                    metadata=metadata
                )
                
                logger.info(f"Submitted {event_type} for strategy {strategy_id} with storage ID {storage_id}")
                return f"Event submitted with storage ID: {storage_id}"
            
            except Exception as e:
                logger.error(f"Error submitting strategy event: {e}")
                return f"Error: {e}"

    def start(self):
        """
        Start the MCP server, initialize the memory bridge, and pre-start required agents.
        """
        # Initialization is now synchronous
        self.memory_bridge = self._initialize_memory_bridge()

        self.logger.info("Pre-starting required agents...")
        self.start_agent_sync("momentum_agent")
        # Give the agent a moment to initialize before starting the main server
        time.sleep(5) 

        self.logger.info(f"Starting AlphaAgentPoolMCPServer on {self.host}:{self.port}")
        self.pool_server.settings.host = self.host
        self.pool_server.settings.port = self.port
        # The run method is blocking, so it will keep the server alive.
        self.pool_server.run(transport="sse")

    def stop(self):
        """
        Stop the MCP server and all running agents.
        """
        self.logger.info("Stopping AlphaAgentPoolMCPServer and all running agents...")
        # Stop all registered agents
        for agent_id in list(self.agent_registry.keys()):
            try:
                self.logger.info(f"Stopping agent '{agent_id}'...")
                # Forcibly terminate the process if it's still running
                process = self.agent_registry[agent_id]
                if hasattr(process, 'terminate'):
                    process.terminate()
                elif hasattr(process, 'kill'):
                    process.kill()
                
                # Wait for the process to terminate
                process.join(timeout=5)
                
                # Remove from registry
                del self.agent_registry[agent_id]
                self.logger.info(f"Agent '{agent_id}' stopped.")
            except Exception as e:
                self.logger.error(f"Error terminating agent '{agent_id}': {e}")
        self.agent_registry.clear()
        self.logger.info("All sub-agents stopped.")

# Helper function to run the momentum agent in a separate process
# This must be a top-level function to be pickleable by multiprocessing
def run_momentum_agent_process(config):
    """
    The target function to run the momentum agent process.
    It imports the agent module and calls the run_agent function with the given config.
    """
    from FinAgents.agent_pools.alpha_agent_pool.agents.theory_driven.momentum_agent import main as run_agent
    import sys

    # Ensure the config is serializable
    config_dict = {
        "execution": {
            "host": "127.0.0.1",
            "port": 5051,  # Use correct port from config file
            "protocol": "http",
            "timeout": 10
        },
        "logging": {
            "level": "info",
            "handlers": ["console", "file"],
            "file": {
                "filename": "momentum_agent.log",
                "maxBytes": 10485760,  # 10 MB
                "backupCount": 5
            }
        },
        "agent_id": "momentum_agent",
        "strategy": {
            "type": "momentum",
            "lookback_period": 14,
            "threshold": 0.05
        },
        "data": {
            "symbol": "AAPL",
            "frequency": "1d",
            "start_date": "2022-01-01",
            "end_date": "2024-12-31"
        },
        "performance": {
            "tracking_enabled": True,
            "reporting_interval": 60
        },
        "real_time_logging": True
    }

    # Override with provided config
    config_dict.update(config)

    # Run the agent with the given configuration
    run_agent(config_dict)


def run_autonomous_agent_process():
    """
    The target function to run the autonomous agent process.
    """
    from FinAgents.agent_pools.alpha_agent_pool.agents.autonomous.autonomous_agent import run_autonomous_agent
    run_autonomous_agent()


if __name__ == "__main__":
    # Setup logging
    # Correctly determine the project root and log directory
    project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    log_dir = os.path.join(project_root_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, "alpha_agent_pool.log")
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout) # Ensure logs go to stdout
        ]
    )
    
    logger = logging.getLogger(__name__)

    # Create and run the server
    server = None
    try:
        logger.info("Initializing Alpha Agent Pool MCP Server...")
        server = AlphaAgentPoolMCPServer()
        logger.info("Starting server...")
        server.start()
    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Stopping server...")
    finally:
        if server:
            server.stop()
        logger.info("Server has been stopped.")
