"""
Alpha Agent Pool Core Module: Unified Entry Point for Multi-Agent Financial Strategy Orchestration

This module serves as the central orchestration hub for the Alpha Agent Pool, implementing
a comprehensive Model Context Protocol (MCP) service architecture for quantitative trading
strategy management and execution. The system employs Agent-to-Agent (A2A) protocol integration
for seamless memory coordination and cross-agent learning facilitation.

Core Academic Framework:
- Multi-agent orchestration theory for financial signal generation
- Memory-augmented reinforcement learning for strategy optimization
- Real-time event streaming and performance analytics
- Cross-sectional alpha generation with risk-adjusted portfolio construction

System Architecture:
- Centralized MCP server for agent lifecycle management
- A2A protocol implementation for distributed memory coordination
- Strategy performance tracking with academic risk metrics
- Real-time signal generation and backtesting capabilities

Author: FinAgent Research Team
License: Open Source Research License
Created: 2025-07-25
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

# Configure module logger
logger = logging.getLogger(__name__)


def check_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """
    Check if a port is available for binding.
    
    Args:
        port: Port number to check
        host: Host address to check (default: 127.0.0.1)
        
    Returns:
        bool: True if port is available, False if occupied
    """
    import socket
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            return result != 0  # Port is available if connection fails
    except Exception:
        return False


def find_available_port(start_port: int, max_attempts: int = 10) -> Optional[int]:
    """
    Find an available port starting from start_port.
    
    Args:
        start_port: Starting port number to check
        max_attempts: Maximum number of ports to check
        
    Returns:
        Available port number or None if no port found
    """
    for port in range(start_port, start_port + max_attempts):
        if check_port_available(port):
            return port
    return None

# Global flag for memory agent availability
MEMORY_AVAILABLE = False

# Import memory bridge modules for centralized memory management
try:
    from .memory_bridge import AlphaAgentPoolMemoryBridge, MEMORY_BRIDGE_AVAILABLE
    from .enhanced_a2a_memory_bridge import EnhancedA2AMemoryBridge, get_memory_bridge, shutdown_memory_bridge
    logger.info("✅ Memory bridge modules successfully imported")
    ENHANCED_A2A_BRIDGE_AVAILABLE = True
    # Create alias for backward compatibility
    AlphaMemoryBridge = AlphaAgentPoolMemoryBridge
except ImportError as e:
    logger.warning(f"⚠️ Memory bridge modules not available: {e}")
    AlphaMemoryBridge = None
    AlphaAgentPoolMemoryBridge = None
    EnhancedA2AMemoryBridge = None
    MEMORY_BRIDGE_AVAILABLE = False
    ENHANCED_A2A_BRIDGE_AVAILABLE = False

# Try to import A2A memory coordinator
try:
    from .a2a_memory_coordinator import (
        AlphaPoolA2AMemoryCoordinator, 
        initialize_pool_coordinator, 
        get_pool_coordinator,
        shutdown_pool_coordinator
    )
    A2A_COORDINATOR_AVAILABLE = True
    logger.info("✅ A2A Memory Coordinator successfully imported")
except ImportError as e:
    logger.warning(f"⚠️ A2A Memory Coordinator not available: {e}")
    A2A_COORDINATOR_AVAILABLE = False

# Import agent manager for centralized agent coordination
try:
    from .agents.agent_manager import AlphaAgentManager, initialize_alpha_agents
    AGENT_MANAGER_AVAILABLE = True
    logger.info("✅ Alpha Agent Manager successfully imported")
except ImportError as e:
    logger.warning(f"⚠️ Alpha Agent Manager not available: {e}")
    AlphaAgentManager = None
    initialize_alpha_agents = None
    AGENT_MANAGER_AVAILABLE = False

# Try to import Alpha Strategy Research Framework
try:
    from .alpha_strategy_research import AlphaStrategyResearchFramework
    ALPHA_STRATEGY_RESEARCH_AVAILABLE = True
except ImportError:
    ALPHA_STRATEGY_RESEARCH_AVAILABLE = False

# Try to import enhanced MCP lifecycle management
try:
    from .enhanced_mcp_lifecycle import (
        EnhancedMCPLifecycleManager,
        create_enhanced_mcp_server
    )
    ENHANCED_MCP_AVAILABLE = True
except ImportError:
    ENHANCED_MCP_AVAILABLE = False

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
    from FinAgents.agent_pools.alpha_agent_pool.alpha_memory_client import AlphaMemoryClient

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
        # Get logger for this method
        logger = logging.getLogger(__name__)
        
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
    """
    Alpha Agent Pool Model Context Protocol (MCP) Server
    
    Professional multi-agent system for systematic alpha factor research, strategy development,
    and portfolio optimization. This implementation follows academic quantitative finance
    principles and provides institutional-grade tools for alpha generation.
    
    Core Capabilities:
    - Systematic Alpha Factor Discovery using academic methodologies
    - Multi-Factor Strategy Configuration with risk management
    - Comprehensive Backtesting with performance attribution
    - Enhanced A2A Memory Coordination for distributed learning
    - Cross-Agent Knowledge Transfer and strategy optimization
    
    Academic Foundation:
    Built on established principles from multi-agent systems theory, quantitative
    finance literature, and modern MLOps practices for algorithmic trading systems.
    
    Architecture:
    - MCP Server (port 8081) with 6 comprehensive alpha research tools
    - Enhanced A2A Memory Bridge with multi-server failover
    - Strategy Research Framework with peer-reviewed methodologies
    - Distributed memory coordination for institutional-scale operations
    """
    
    def __init__(self, host="0.0.0.0", port=8081, enable_enhanced_lifecycle=True):
        """
        Initialize the Alpha Agent Pool MCP Server with comprehensive capabilities.
        
        This constructor establishes the foundational architecture for multi-agent
        orchestration, including A2A protocol integration, memory coordination,
        and strategy performance tracking systems.
        
        Args:
            host (str): Network interface binding address for the MCP server
            port (int): TCP port number for service endpoint exposure  
            enable_enhanced_lifecycle (bool): Enable advanced lifecycle management features
        """
        self.host = host
        self.port = port
        self.enable_enhanced_lifecycle = enable_enhanced_lifecycle
        
        # Initialize enhanced MCP server or fallback to basic FastMCP
        if enable_enhanced_lifecycle and ENHANCED_MCP_AVAILABLE:
            self.pool_server, self.lifecycle_manager = create_enhanced_mcp_server(
                pool_id=f"alpha_pool_{port}"
            )
            self.logger = logging.getLogger(__name__)
            self.logger.info("Enhanced MCP lifecycle management enabled")
        else:
            self.pool_server = FastMCP("AlphaAgentPoolMCPServer")
            self.lifecycle_manager = None
            self.logger = logging.getLogger(__name__)
            if enable_enhanced_lifecycle:
                self.logger.warning("Enhanced MCP lifecycle management not available, using basic MCP")
        
        # Agent registry for tracking active instances
        self.agent_registry = {}  # agent_id -> (agent, process/thread)
        self.config_dir = os.path.join(os.path.dirname(__file__), "config")
        
        # Initialize the Command Planner for DAG-based execution
        self.planner = CommandPlanner(self)
        
        # Initialize A2A Memory Coordinator for pool-level memory operations
        self.a2a_coordinator: Optional["AlphaPoolA2AMemoryCoordinator"] = None
        self._coordinator_initialization_task = None
        
        if A2A_COORDINATOR_AVAILABLE:
            self.logger.info("A2A Memory Coordinator available, scheduling initialization")
        else:
            self.logger.warning("A2A Memory Coordinator not available, falling back to legacy memory")
        
        # Initialize memory bridge for comprehensive strategy tracking
        self.memory_bridge: Optional["AlphaAgentPoolMemoryBridge"] = None
        # Note: Memory bridge will be initialized asynchronously when needed
        
        # Initialize AlphaMemoryClient for MCP-based memory logging (temporarily disabled)
        # self.alpha_memory_client = AlphaMemoryClient(agent_id="alpha_agent_pool_server")
        self.alpha_memory_client = None  # Temporarily disabled until AlphaMemoryClient is available
        
        # Load historical market data into memory unit with automatic reset
        csv_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            "../../../data/cache/AAPL_2022-01-01_2024-12-31_1d.csv"
        ))
        self.memory = MemoryUnit(
            os.path.join(os.path.dirname(__file__), "memory_unit.json"), 
            autoload_csv_path=csv_path, 
            reset_on_init=True,
            memory_bridge=self.memory_bridge
        )
        
        # Initialize legacy memory agent if available (deprecated in favor of A2A)
        self.memory_agent: Optional["ExternalMemoryAgent"] = None
        self.session_id = None
        self._initialize_memory_agent()  # Initialize memory agent synchronously
        
        # Strategy performance tracking and analytics
        self.strategy_performance_cache = {}
        self.signal_generation_history = []
        
        # Import strategy research framework
        try:
            from .alpha_strategy_research import (
                AlphaStrategyResearchFramework,
                AlphaFactorCategory,
                RiskLevel
            )
            STRATEGY_RESEARCH_AVAILABLE = True
        except ImportError:
            STRATEGY_RESEARCH_AVAILABLE = False
            self.logger.warning("Alpha Strategy Research Framework not available")
        
        # Initialize strategy research framework
        self.strategy_research_framework = None
        if STRATEGY_RESEARCH_AVAILABLE:
            self.strategy_research_framework = AlphaStrategyResearchFramework(
                strategy_universe=["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
                benchmark="SPY",
                research_period="5Y",
                a2a_coordinator=None  # Will be set after A2A coordinator initialization
            )
            self.logger.info("Alpha Strategy Research Framework initialized")
        
        # Initialize Enhanced A2A Memory Bridge
        self.enhanced_memory_bridge = None
        self._bridge_initialization_pending = ENHANCED_A2A_BRIDGE_AVAILABLE
        if ENHANCED_A2A_BRIDGE_AVAILABLE:
            self.logger.info("Enhanced A2A Memory Bridge scheduled for initialization")
        
        # Register all pool management tools
        self._register_pool_tools()

    def _start_momentum_agent(self):
        """
        Starts the momentum agent as a separate process by correctly importing and instantiating it.
        Includes port conflict detection and automatic port resolution.
        """
        # Import agent-specific components here to avoid top-level import errors
        from FinAgents.agent_pools.alpha_agent_pool.schema.theory_driven_schema import MomentumAgentConfig
        from multiprocessing import Process

        config_path = os.path.join(self.config_dir, "momentum.yaml")
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Check if configured port is available
        configured_port = config_data.get('execution', {}).get('port', 5051)
        if not check_port_available(configured_port):
            self.logger.warning(f"⚠️ Momentum agent port {configured_port} is occupied, finding alternative...")
            
            # Find an available port
            alternative_port = find_available_port(configured_port + 1)
            if alternative_port:
                self.logger.info(f"🔄 Using alternative port {alternative_port} for momentum agent")
                config_data['execution']['port'] = alternative_port
            else:
                self.logger.error(f"❌ No available ports found starting from {configured_port + 1}")
                raise RuntimeError(f"No available ports for momentum agent")
        
        config = MomentumAgentConfig(**config_data)
        
        # Use a helper function to be the target of the process
        process = Process(target=run_momentum_agent_process, args=(config,))
        process.start()
        return config, process

    def _start_autonomous_agent(self):
        """
        Starts the autonomous agent as a separate process.
        Includes port conflict detection and automatic port resolution.
        """
        # Import agent-specific components here
        from FinAgents.agent_pools.alpha_agent_pool.agents.autonomous.autonomous_agent import run_autonomous_agent
        from multiprocessing import Process

        # Load configuration for autonomous agent
        config_path = os.path.join(self.config_dir, "autonomous.yaml")
        
        # Load config to check port
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        configured_port = config_data.get('execution', {}).get('port', 5052)
        port_override = None
        
        # Check if configured port is available
        if not check_port_available(configured_port):
            self.logger.warning(f"⚠️ Autonomous agent port {configured_port} is occupied, finding alternative...")
            
            # Find an available port
            alternative_port = find_available_port(configured_port + 1)
            if alternative_port:
                self.logger.info(f"🔄 Using alternative port {alternative_port} for autonomous agent")
                port_override = alternative_port
            else:
                self.logger.error(f"❌ No available ports found starting from {configured_port + 1}")
                raise RuntimeError(f"No available ports for autonomous agent")
        
        process = Process(target=run_autonomous_agent, args=(config_path, port_override))
        process.start()
        return process

    def start_agent_sync(self, agent_name: str):
        """
        Synchronously start a specific agent without asyncio event loop conflicts.
        Used by the synchronous start() method to avoid event loop issues.
        """
        if agent_name == "momentum_agent":
            try:
                self.logger.info(f"🚀 Starting {agent_name} synchronously...")
                config, process = self._start_momentum_agent()
                self.logger.info(f"✅ {agent_name} started successfully on port {config['port']}")
            except Exception as e:
                self.logger.error(f"❌ Failed to start {agent_name}: {e}")
                raise
        elif agent_name == "autonomous_agent":
            try:
                self.logger.info(f"🚀 Starting {agent_name} synchronously...")
                process = self._start_autonomous_agent()
                self.logger.info(f"✅ {agent_name} started successfully")
            except Exception as e:
                self.logger.error(f"❌ Failed to start {agent_name}: {e}")
                raise
        else:
            self.logger.error(f"❌ Unknown agent name: {agent_name}")
            raise ValueError(f"Unknown agent name: {agent_name}")

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
            
            # Read config to determine actual port used
            config_path = os.path.join(self.config_dir, "autonomous.yaml")
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            configured_port = config_data.get('execution', {}).get('port', 5052)
            
            # Determine actual port used (could be different if there was a conflict)
            actual_port = configured_port
            if not check_port_available(configured_port):
                alternative_port = find_available_port(configured_port + 1)
                if alternative_port:
                    actual_port = alternative_port
            
            details = f"Autonomous agent started on port {actual_port}"
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
        global MEMORY_AVAILABLE
        try:
            from FinAgents.memory.external_memory_interface import ExternalMemoryAgent
            MEMORY_AVAILABLE = True
        except ImportError:
            MEMORY_AVAILABLE = False

        if not MEMORY_AVAILABLE:
            self.logger.warning("External memory agent not available. Continuing without it.")
            return
        try:
            self.memory_agent = ExternalMemoryAgent()
            self.session_id = f"alpha_pool_session_{int(time.time())}"
            logger.info("External memory agent initialized for Alpha Agent Pool")
        except Exception as e:
            logger.error(f"Failed to initialize memory agent: {e}")
            self.memory_agent = None

    async def _initialize_enhanced_memory_bridge(self):
        """
        Initialize the Enhanced A2A Memory Bridge for improved memory coordination.
        
        This method sets up the enhanced memory bridge that provides robust
        connectivity to multiple memory services with automatic fallback.
        """
        try:
            if ENHANCED_A2A_BRIDGE_AVAILABLE:
                from .enhanced_a2a_memory_bridge import get_memory_bridge
                
                # Get or create the global memory bridge
                self.enhanced_memory_bridge = await get_memory_bridge(
                    pool_id=f"alpha_pool_{self.port}"
                )
                
                # Test the connection
                health_status = await self.enhanced_memory_bridge.health_check()
                if health_status.get("bridge_status") == "healthy":
                    self.logger.info("✅ Enhanced A2A Memory Bridge initialized successfully")
                    
                    # Update strategy research framework with memory bridge
                    if self.strategy_research_framework:
                        self.strategy_research_framework.memory_bridge = self.enhanced_memory_bridge
                        self.logger.info("✅ Strategy Research Framework connected to memory bridge")
                else:
                    self.logger.warning("⚠️  Enhanced A2A Memory Bridge initialized but connection unhealthy")
            else:
                self.logger.warning("Enhanced A2A Memory Bridge not available")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Enhanced A2A Memory Bridge: {e}")

    async def _store_performance_via_bridge(self, 
                                          agent_id: str, 
                                          performance_data: dict) -> bool:
        """
        Store agent performance data via Enhanced A2A Memory Bridge.
        
        Args:
            agent_id: Identifier of the agent
            performance_data: Performance metrics to store
            
        Returns:
            bool: True if storage successful
        """
        if self.enhanced_memory_bridge:
            return await self.enhanced_memory_bridge.store_agent_performance(
                agent_id=agent_id,
                performance_data=performance_data
            )
        return False

    async def _store_strategy_insights_via_bridge(self, 
                                                strategy_id: str, 
                                                insights_data: dict) -> bool:
        """
        Store strategy insights via Enhanced A2A Memory Bridge.
        
        Args:
            strategy_id: Identifier of the strategy
            insights_data: Strategy insights and configuration data
            
        Returns:
            bool: True if storage successful
        """
        if self.enhanced_memory_bridge:
            return await self.enhanced_memory_bridge.store_strategy_insights(
                strategy_id=strategy_id,
                insights_data=insights_data
            )
        return False

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

    async def store_event_in_memory(self, event_type, summary, keywords, details, log_level="INFO", session_id=None, correlation_id=None):
        """Store an event in the MCP memory server using AlphaMemoryClient."""
        if self.alpha_memory_client is None:
            logger.warning("AlphaMemoryClient not available, skipping event storage")
            return {"status": "skipped", "reason": "client_not_available"}
            
        return await self.alpha_memory_client.store_event(
            event_type=event_type,
            summary=summary,
            keywords=keywords,
            details=details,
            log_level=log_level,
            session_id=session_id,
            correlation_id=correlation_id
        )

    def _register_pool_tools(self):
        """
        Register management tools for the agent pool, including starting and listing sub-agents.
        Enhanced with comprehensive memory bridge integration and strategy flow tracking.
        """
        @self.pool_server.tool(name="start_agent", description="Start the specified sub-agent service.")
        def start_agent(agent_id: str) -> str:
            result = self.start_agent_sync(agent_id)
            # Log the event asynchronously to the MCP memory server
            if self.alpha_memory_client is not None:
                asyncio.create_task(
                    self.alpha_memory_client.store_event(
                        event_type="AGENT_ACTION",
                        summary=f"Started agent {agent_id}",
                        keywords=["start_agent", agent_id],
                        details={"agent_id": agent_id, "result": result}
                    )
                )
            return result

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

        @self.pool_server.tool(name="momentum_health", description="Check the health of the momentum agent.")
        async def momentum_health() -> dict:
            """
            Returns health status of the momentum agent, including process status and agent endpoint check.
            """
            import httpx
            status = self.check_agent_status("momentum_agent").name
            endpoint = "http://127.0.0.1:5051/health"
            endpoint_status = "unknown"
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.get(endpoint)
                    if resp.status_code == 200:
                        endpoint_status = "healthy"
                    else:
                        endpoint_status = f"unhealthy ({resp.status_code})"
            except Exception as e:
                endpoint_status = f"error: {e}"
            return {
                "process_status": status,
                "endpoint_status": endpoint_status
            }

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

        @self.pool_server.tool(name="generate_alpha_signals", description="Generate alpha signals based on market data with A2A memory coordination")
        async def generate_alpha_signals(symbol: str = None, symbols: list = None, date: str = None, lookback_period: int = 20, price: Optional[float] = None, memory: dict = None) -> dict:
            """
            Generate alpha signals using momentum strategy with comprehensive A2A memory coordination.
            
            This function implements academic-standard alpha signal generation with integrated
            memory coordination via the A2A protocol. Signal generation results are automatically
            stored in the distributed memory system for cross-agent learning and strategy optimization.
            
            Args:
                symbol: Individual stock symbol for signal generation
                symbols: List of stock symbols for batch processing
                date: Target date for signal generation (ISO format)
                lookback_period: Historical data window for momentum calculation
                price: Current price override (optional)
                memory: Alternative memory source (optional)
                
            Returns:
                Dict containing generated alpha signals with metadata and performance tracking
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
                
                # Use current date if not specified
                if not date:
                    date = "2024-01-30"  # Default date for testing
                
                results = {}
                signal_generation_metadata = {
                    "generation_timestamp": datetime.utcnow().isoformat(),
                    "lookback_period": lookback_period,
                    "target_date": date,
                    "total_symbols": len(target_symbols)
                }
                
                for target_symbol in target_symbols:
                    logger.info(f"Generating alpha signal for symbol: {target_symbol}")
                    
                    # Price data collection and validation
                    price_list = []
                    current_price = price
                    
                    if current_price is None or not isinstance(current_price, (float, int)):
                        memory_source = memory if memory is not None else self.memory
                        price_key = f"{target_symbol}_close_{date}"
                        current_price_str = memory_source.get(price_key) if hasattr(memory_source, 'get') else memory_source.get(price_key) if memory_source else None
                        
                        if current_price_str is None:
                            logger.warning(f"No price data for {target_symbol} on {date}, using default price 100.0")
                            current_price = 100.0
                        else:
                            try:
                                current_price = float(current_price_str)
                            except Exception:
                                logger.warning(f"Invalid price data for {target_symbol} on {date}, using default price 100.0")
                                current_price = 100.0

                    # Collect historical price data for momentum calculation
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

                    # Ensure chronological order and add current price
                    price_list.reverse()
                    price_list.append(current_price)

                    logger.info(f"Executing momentum strategy with {len(price_list)} price points for {target_symbol}")

                    # Initialize default response structure
                    response = {"signal": "HOLD", "confidence": 0.0, "error": "Unknown error"}
                    
                    # Attempt signal generation via MCP SSE client
                    base_url = "http://127.0.0.1:5051/sse"
                    try:
                        logger.info(f"Establishing SSE connection to momentum agent for {target_symbol}")
                        
                        async with sse_client(base_url, timeout=10) as (read, write):
                            logger.info(f"SSE connection established for {target_symbol}")
                            
                            async with ClientSession(read, write) as session:
                                await session.initialize()
                                logger.info(f"SSE session initialized for {target_symbol}")
                                
                                request_params = {
                                    "symbol": target_symbol,
                                    "price_list": price_list
                                }
                                
                                logger.info(f"Calling generate_signal tool for {target_symbol}")
                                
                                response_parts = await session.call_tool(
                                    "generate_signal",
                                    request_params
                                )
                                
                                # Parse response from momentum agent
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
                                    
                    except Exception as mcp_error:
                        logger.warning(f"MCP SSE client failed for {target_symbol}: {mcp_error}")
                        
                        # Fallback to simple HTTP client
                        try:
                            from FinAgents.agent_pools.alpha_agent_pool.simple_momentum_client import SimpleMomentumClient
                            simple_client = SimpleMomentumClient()
                            response = await simple_client.call_generate_signal(target_symbol, price_list)
                            logger.info(f"Fallback HTTP client succeeded for {target_symbol}")
                        except Exception as fallback_error:
                            logger.error(f"Both MCP and HTTP clients failed for {target_symbol}: {fallback_error}")
                            response = {"signal": "HOLD", "confidence": 0.0, "error": f"All clients failed: {mcp_error}"}

                    logger.info(f"Generated signal for {target_symbol}: {response}")
                    results[target_symbol] = response
                    
                    # Store signal generation event in A2A memory coordinator if available
                    if self.a2a_coordinator and response.get("signal") != "HOLD":
                        try:
                            await self.a2a_coordinator.a2a_client.store_strategy_performance(
                                agent_id="momentum_agent",
                                strategy_id=f"momentum_{target_symbol}_{date}",
                                performance_metrics={
                                    "symbol": target_symbol,
                                    "signal": response.get("signal"),
                                    "confidence": response.get("confidence", 0.0),
                                    "generation_date": date,
                                    "price_points_analyzed": len(price_list),
                                    "lookback_period": lookback_period
                                }
                            )
                            logger.info(f"✅ Signal stored in A2A memory coordinator for {target_symbol}")
                        except Exception as storage_error:
                            logger.warning(f"⚠️ Failed to store signal in A2A coordinator: {storage_error}")
                
                # Compile comprehensive response with academic metadata
                final_response = {
                    "status": "success",
                    "alpha_signals": {
                        "signals": results,
                        "metadata": signal_generation_metadata,
                        "generation_summary": {
                            "total_symbols_processed": len(results),
                            "successful_signals": len([r for r in results.values() if r.get("signal") != "HOLD"]),
                            "average_confidence": sum(r.get("confidence", 0) for r in results.values()) / len(results) if results else 0,
                            "memory_coordination_active": self.a2a_coordinator is not None
                        }
                    }
                }
                
                logger.info(f"✅ Alpha signal generation completed for {len(target_symbols)} symbols")
                logger.info(f"✅ Alpha signal generation completed for {len(target_symbols)} symbols")
                return final_response
                
            except Exception as top_level_error:
                logger.error(f"❌ Critical error in alpha signal generation: {top_level_error}", exc_info=True)
                import traceback
                error_traceback = traceback.format_exc()
                logger.error(f"Full error traceback: {error_traceback}")
                
                return {
                    "status": "error",
                    "message": f"Critical error in signal generation: {top_level_error}",
                    "alpha_signals": {"signals": {}}
                }

        @self.pool_server.tool(name="run_rl_backtest_and_update", description="Run RL backtest and update agent policy for a given symbol and market data.")
        async def run_rl_backtest_and_update(symbol: str, market_data: list, lookback_period: int = 30, initial_cash: float = 100000.0) -> dict:
            """
            Remotely call the momentum agent's RL backtest and policy update tool via SSE.
            Args:
                symbol (str): The symbol to backtest.
                market_data (list): List of dicts with 'date' and 'price'.
                lookback_period (int): Lookback window for momentum.
                initial_cash (float): Initial cash for backtest.
            Returns:
                dict: RL backtest and update results.
            """
            try:
                from mcp.client.sse import sse_client
                from mcp import ClientSession
                base_url = "http://127.0.0.1:5051/sse"
                logger.info(f"Connecting to momentum_agent for RL backtest/update: {symbol}")
                async with sse_client(base_url, timeout=15) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        params = {
                            "symbol": symbol,
                            "market_data": market_data,
                            "lookback_period": lookback_period,
                            "initial_cash": initial_cash
                        }
                        logger.info(f"Calling run_rl_backtest_and_update tool for {symbol}")
                        response_parts = await session.call_tool("run_rl_backtest_and_update", params)
                        if response_parts and hasattr(response_parts, 'content') and response_parts.content:
                            content_list = response_parts.content
                            if len(content_list) > 0 and hasattr(content_list[0], 'text'):
                                content_str = content_list[0].text
                                try:
                                    return json.loads(content_str)
                                except Exception:
                                    return {"status": "error", "raw_response": content_str}
                        return {"status": "error", "message": "No response from momentum agent."}
            except Exception as e:
                logger.error(f"Error in run_rl_backtest_and_update: {e}", exc_info=True)
                return {"status": "error", "message": str(e)}
        
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

        @self.pool_server.tool(name="discover_alpha_factors", description="Systematic alpha factor discovery and validation using academic methodologies")
        async def discover_alpha_factors(factor_categories: List[str] = None, significance_threshold: float = 0.05) -> dict:
            """
            Execute systematic alpha factor discovery following academic research protocols.
            
            This tool implements comprehensive factor mining using established quantitative
            finance methodologies for systematic alpha generation and statistical validation.
            
            Args:
                factor_categories: List of factor categories to investigate (momentum, mean_reversion, technical, volatility)
                significance_threshold: Statistical significance threshold for factor validation (default: 0.05)
                
            Returns:
                Dictionary containing discovered and validated alpha factors with academic metrics
            """
            if not self.strategy_research_framework:
                return {
                    "status": "error",
                    "message": "Strategy Research Framework not available",
                    "factors": {}
                }
            
            try:
                # Convert string categories to enum
                if factor_categories:
                    from .alpha_strategy_research import AlphaFactorCategory
                    category_enums = []
                    for cat in factor_categories:
                        try:
                            category_enums.append(AlphaFactorCategory(cat.lower()))
                        except ValueError:
                            self.logger.warning(f"Unknown factor category: {cat}")
                else:
                    category_enums = None
                
                # Execute factor discovery
                discovered_factors = await self.strategy_research_framework.discover_alpha_factors(
                    factor_categories=category_enums,
                    significance_threshold=significance_threshold
                )
                
                # Store discovery results via enhanced memory bridge
                discovery_performance = {
                    "total_factors_discovered": len(discovered_factors),
                    "significance_threshold": significance_threshold,
                    "avg_ir": sum(f.expected_ir for f in discovered_factors.values()) / len(discovered_factors) if discovered_factors else 0,
                    "discovery_timestamp": datetime.utcnow().isoformat(),
                    "methodology": "Systematic alpha factor discovery with academic validation"
                }
                
                await self._store_performance_via_bridge(
                    agent_id="alpha_factor_discovery_engine",
                    performance_data=discovery_performance
                )
                
                # Format response for academic reporting
                factor_summary = {}
                for factor_id, factor in discovered_factors.items():
                    factor_summary[factor_id] = {
                        "name": factor.factor_name,
                        "category": factor.category.value,
                        "expected_ir": factor.expected_ir,
                        "statistical_significance": factor.statistical_significance,
                        "robustness_score": factor.robustness_score,
                        "capacity_estimate": factor.capacity_estimate,
                        "academic_references": factor.academic_references[:2]
                    }
                
                return {
                    "status": "success",
                    "discovery_timestamp": datetime.utcnow().isoformat(),
                    "total_factors_discovered": len(discovered_factors),
                    "significance_threshold_used": significance_threshold,
                    "factors": factor_summary,
                    "methodology": "Systematic alpha factor discovery with academic validation",
                    "next_steps": ["develop_strategy_configuration", "run_comprehensive_backtest"]
                }
                
            except Exception as e:
                self.logger.error(f"Error in alpha factor discovery: {e}")
                return {
                    "status": "error",
                    "message": f"Factor discovery failed: {str(e)}",
                    "factors": {}
                }

        @self.pool_server.tool(name="develop_strategy_configuration", description="Develop institutional-grade strategy configuration from discovered alpha factors")
        async def develop_strategy_configuration(risk_level: str = "moderate", target_volatility: float = 0.15) -> dict:
            """
            Develop comprehensive strategy configuration following academic portfolio construction principles.
            
            This tool creates institutional-grade strategy specifications with comprehensive
            risk management, regime adaptation, and emergency procedures based on validated alpha factors.
            
            Args:
                risk_level: Target risk level (conservative, moderate, aggressive, institutional)
                target_volatility: Target annualized volatility (default: 0.15)
                
            Returns:
                Complete strategy configuration with implementation details and academic rationale
            """
            if not self.strategy_research_framework:
                return {
                    "status": "error",
                    "message": "Strategy Research Framework not available"
                }
            
            if not self.strategy_research_framework.discovered_factors:
                return {
                    "status": "error",
                    "message": "No alpha factors available. Please run discover_alpha_factors first."
                }
            
            try:
                from .alpha_strategy_research import RiskLevel
                
                # Convert string to risk level enum
                risk_level_enum = RiskLevel(risk_level.lower())
                
                # Develop strategy configuration
                strategy_config = await self.strategy_research_framework.develop_strategy_configuration(
                    factors=self.strategy_research_framework.discovered_factors,
                    risk_level=risk_level_enum,
                    target_volatility=target_volatility
                )
                
                # Store strategy configuration via enhanced memory bridge
                strategy_insights = {
                    "strategy_id": strategy_config.strategy_id,
                    "strategy_name": strategy_config.strategy_name,
                    "primary_factors": len(strategy_config.primary_alpha_factors),
                    "secondary_factors": len(strategy_config.secondary_alpha_factors),
                    "target_volatility": strategy_config.target_volatility,
                    "target_tracking_error": strategy_config.target_tracking_error,
                    "max_drawdown_limit": strategy_config.maximum_drawdown_limit,
                    "expected_capacity": strategy_config.expected_capacity,
                    "risk_level": risk_level,
                    "configuration_timestamp": datetime.utcnow().isoformat()
                }
                
                await self._store_strategy_insights_via_bridge(
                    strategy_id=strategy_config.strategy_id,
                    insights_data=strategy_insights
                )
                
                # Format response for academic presentation
                return {
                    "status": "success",
                    "strategy_id": strategy_config.strategy_id,
                    "strategy_name": strategy_config.strategy_name,
                    "configuration_timestamp": datetime.utcnow().isoformat(),
                    "primary_factors": len(strategy_config.primary_alpha_factors),
                    "secondary_factors": len(strategy_config.secondary_alpha_factors),
                    "target_metrics": {
                        "volatility": strategy_config.target_volatility,
                        "tracking_error": strategy_config.target_tracking_error,
                        "max_drawdown_limit": strategy_config.maximum_drawdown_limit
                    },
                    "risk_management": {
                        "position_limits": strategy_config.risk_management_rules["position_limits"],
                        "volatility_management": strategy_config.risk_management_rules["volatility_management"],
                        "drawdown_controls": strategy_config.risk_management_rules["drawdown_controls"]
                    },
                    "regime_adaptation": len(strategy_config.regime_adaptation_rules),
                    "emergency_procedures": list(strategy_config.emergency_procedures.keys()),
                    "expected_capacity": strategy_config.expected_capacity,
                    "academic_rationale": strategy_config.academic_rationale[:500] + "...",
                    "next_steps": ["run_comprehensive_backtest", "submit_strategy_to_memory"]
                }
                
            except Exception as e:
                self.logger.error(f"Error in strategy configuration development: {e}")
                return {
                    "status": "error",
                    "message": f"Strategy configuration failed: {str(e)}"
                }

        @self.pool_server.tool(name="run_comprehensive_backtest", description="Execute institutional-grade backtesting with full performance attribution")
        async def run_comprehensive_backtest(strategy_id: str, start_date: str = "2018-01-01", end_date: str = "2023-12-31") -> dict:
            """
            Execute comprehensive academic-standard backtesting with full performance attribution.
            
            This tool implements institutional-grade backtesting following academic standards
            for quantitative strategy validation and performance measurement with comprehensive
            risk analytics and statistical testing.
            
            Args:
                strategy_id: ID of the strategy configuration to backtest
                start_date: Backtest start date (ISO format, default: 2018-01-01)
                end_date: Backtest end date (ISO format, default: 2023-12-31)
                
            Returns:
                Comprehensive backtest results with academic performance metrics
            """
            if not self.strategy_research_framework:
                return {
                    "status": "error",
                    "message": "Strategy Research Framework not available"
                }
            
            # Find strategy configuration
            strategy_config = None
            for config in self.strategy_research_framework.strategy_configurations.values():
                if config.strategy_id == strategy_id:
                    strategy_config = config
                    break
            
            if not strategy_config:
                return {
                    "status": "error",
                    "message": f"Strategy configuration {strategy_id} not found. Please run develop_strategy_configuration first."
                }
            
            try:
                # Execute comprehensive backtest
                backtest_results = await self.strategy_research_framework.run_comprehensive_backtest(
                    strategy_config=strategy_config,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Store backtest results via enhanced memory bridge
                backtest_performance = {
                    "backtest_id": backtest_results.backtest_id,
                    "strategy_id": backtest_results.strategy_id,
                    "total_return": backtest_results.total_return,
                    "annualized_return": backtest_results.annualized_return,
                    "volatility": backtest_results.volatility,
                    "sharpe_ratio": backtest_results.sharpe_ratio,
                    "information_ratio": backtest_results.information_ratio,
                    "maximum_drawdown": backtest_results.maximum_drawdown,
                    "win_rate": backtest_results.win_rate,
                    "backtest_period": f"{start_date} to {end_date}",
                    "validation_status": "PASSED" if backtest_results.sharpe_ratio >= 1.0 else "REVIEW_REQUIRED",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                await self._store_performance_via_bridge(
                    agent_id="comprehensive_backtest_engine",
                    performance_data=backtest_performance
                )
                
                # Format response for academic presentation
                return {
                    "status": "success",
                    "backtest_id": backtest_results.backtest_id,
                    "strategy_id": backtest_results.strategy_id,
                    "backtest_period": f"{start_date} to {end_date}",
                    "performance_summary": {
                        "total_return": f"{backtest_results.total_return:.1%}",
                        "annualized_return": f"{backtest_results.annualized_return:.1%}",
                        "volatility": f"{backtest_results.volatility:.1%}",
                        "sharpe_ratio": f"{backtest_results.sharpe_ratio:.2f}",
                        "information_ratio": f"{backtest_results.information_ratio:.2f}",
                        "maximum_drawdown": f"{backtest_results.maximum_drawdown:.1%}",
                        "win_rate": f"{backtest_results.win_rate:.1%}"
                    },
                    "risk_metrics": {
                        "var_95": f"{backtest_results.var_95:.1%}",
                        "cvar_95": f"{backtest_results.cvar_95:.1%}",
                        "beta": f"{backtest_results.beta:.2f}",
                        "alpha": f"{backtest_results.alpha:.1%}",
                        "tracking_error": f"{backtest_results.tracking_error:.1%}"
                    },
                    "trading_metrics": {
                        "total_trades": backtest_results.total_trades,
                        "turnover": f"{backtest_results.turnover:.1f}x",
                        "transaction_costs": f"{backtest_results.transaction_costs:.1%}",
                        "profit_factor": f"{backtest_results.profit_factor:.2f}"
                    },
                    "regime_performance": {
                        regime.value: {
                            "return": f"{metrics['return']:.1%}",
                            "volatility": f"{metrics['volatility']:.1%}"
                        }
                        for regime, metrics in backtest_results.regime_performance.items()
                    },
                    "validation_status": "PASSED" if backtest_results.sharpe_ratio >= 1.0 else "REVIEW_REQUIRED",
                    "next_steps": ["submit_strategy_to_memory", "generate_strategy_report"]
                }
                
            except Exception as e:
                self.logger.error(f"Error in comprehensive backtesting: {e}")
                return {
                    "status": "error",
                    "message": f"Backtesting failed: {str(e)}"
                }

        @self.pool_server.tool(name="submit_strategy_to_memory", description="Submit complete strategy package to memory agent via A2A protocol")
        async def submit_strategy_to_memory(strategy_id: str, backtest_id: str = None) -> dict:
            """
            Submit complete strategy research package to memory agent via A2A protocol.
            
            This tool packages the complete strategy research, configuration, and validation
            results for storage in the distributed memory system following academic
            documentation standards and institutional reporting requirements.
            
            Args:
                strategy_id: ID of the strategy configuration to submit
                backtest_id: ID of the backtest results to include (optional, uses latest if not specified)
                
            Returns:
                Memory storage confirmation with submission details
            """
            if not self.strategy_research_framework:
                return {
                    "status": "error",
                    "message": "Strategy Research Framework not available"
                }
            
            # Find strategy configuration
            strategy_config = None
            for config in self.strategy_research_framework.strategy_configurations.values():
                if config.strategy_id == strategy_id:
                    strategy_config = config
                    break
            
            if not strategy_config:
                return {
                    "status": "error",
                    "message": f"Strategy configuration {strategy_id} not found"
                }
            
            # Find backtest results
            backtest_results = None
            if backtest_id:
                backtest_results = self.strategy_research_framework.backtest_results.get(backtest_id)
            else:
                # Use latest backtest for this strategy
                for bt_id, bt_results in self.strategy_research_framework.backtest_results.items():
                    if bt_results.strategy_id == strategy_id:
                        backtest_results = bt_results
                        break
            
            if not backtest_results:
                return {
                    "status": "error",
                    "message": f"No backtest results found for strategy {strategy_id}. Please run backtest first."
                }
            
            try:
                # Submit strategy package to memory
                storage_id = await self.strategy_research_framework.submit_strategy_to_memory(
                    strategy_config=strategy_config,
                    backtest_results=backtest_results
                )
                
                # Store strategy submission via enhanced memory bridge
                submission_data = {
                    "storage_id": storage_id,
                    "strategy_id": strategy_id,
                    "backtest_id": backtest_id,
                    "strategy_name": strategy_config.strategy_name,
                    "sharpe_ratio": backtest_results.sharpe_ratio if backtest_results else None,
                    "total_return": backtest_results.total_return if backtest_results else None,
                    "submission_timestamp": datetime.utcnow().isoformat(),
                    "validation_status": "complete"
                }
                
                await self._store_strategy_insights_via_bridge(
                    strategy_id=f"submitted_{strategy_id}",
                    insights_data=submission_data
                )
                
                return {
                    "status": "success",
                    "storage_id": storage_id,
                    "submission_timestamp": datetime.utcnow().isoformat(),
                    "strategy_name": strategy_config.strategy_name,
                    "package_contents": {
                        "strategy_configuration": "included",
                        "backtest_results": "included", 
                        "alpha_factors": len(self.strategy_research_framework.discovered_factors),
                        "academic_validation": "included",
                        "implementation_readiness": "included"
                    },
                    "memory_coordination": "A2A protocol" if self.a2a_coordinator else "local_storage",
                    "next_steps": ["generate_strategy_report", "begin_implementation_review"]
                }
                
            except Exception as e:
                self.logger.error(f"Error submitting strategy to memory: {e}")
                return {
                    "status": "error",
                    "message": f"Strategy submission failed: {str(e)}"
                }

        @self.pool_server.tool(name="generate_strategy_report", description="Generate comprehensive academic-style strategy research report")
        async def generate_strategy_report(strategy_id: str, backtest_id: str = None) -> dict:
            """
            Generate comprehensive academic-style strategy research report.
            
            This tool creates institutional-grade documentation suitable for academic
            publication, regulatory submission, and professional implementation following
            established standards for quantitative strategy research reporting.
            
            Args:
                strategy_id: ID of the strategy configuration
                backtest_id: ID of the backtest results (optional, uses latest if not specified)
                
            Returns:
                Generated research report with academic formatting and comprehensive analysis
            """
            if not self.strategy_research_framework:
                return {
                    "status": "error",
                    "message": "Strategy Research Framework not available"
                }
            
            # Find strategy configuration and backtest results
            strategy_config = None
            for config in self.strategy_research_framework.strategy_configurations.values():
                if config.strategy_id == strategy_id:
                    strategy_config = config
                    break
            
            if not strategy_config:
                return {
                    "status": "error",
                    "message": f"Strategy configuration {strategy_id} not found"
                }
            
            backtest_results = None
            if backtest_id:
                backtest_results = self.strategy_research_framework.backtest_results.get(backtest_id)
            else:
                # Use latest backtest for this strategy
                for bt_id, bt_results in self.strategy_research_framework.backtest_results.items():
                    if bt_results.strategy_id == strategy_id:
                        backtest_results = bt_results
                        break
            
            if not backtest_results:
                return {
                    "status": "error",
                    "message": f"No backtest results found for strategy {strategy_id}"
                }
            
            try:
                # Generate comprehensive research report
                research_report = await self.strategy_research_framework.generate_strategy_report(
                    strategy_config=strategy_config,
                    backtest_results=backtest_results
                )
                
                # Save report to file
                report_filename = f"strategy_report_{strategy_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"
                report_path = Path(report_filename)
                
                with open(report_path, 'w') as f:
                    f.write(research_report)
                
                return {
                    "status": "success",
                    "report_generated": True,
                    "report_filename": report_filename,
                    "report_sections": [
                        "Executive Summary",
                        "Alpha Factor Analysis", 
                        "Performance Analysis",
                        "Risk Management Framework",
                        "Academic Rationale",
                        "Implementation Recommendations"
                    ],
                    "report_length": len(research_report),
                    "academic_standards": "Institutional-grade documentation suitable for publication",
                    "next_steps": ["regulatory_review", "implementation_planning", "capacity_analysis"]
                }
                
            except Exception as e:
                self.logger.error(f"Error generating strategy report: {e}")
                return {
                    "status": "error",
                    "message": f"Report generation failed: {str(e)}"
                }

    def start(self):
        """
        Initialize and start the Alpha Agent Pool MCP Server with comprehensive capabilities.
        
        This method orchestrates the complete startup sequence for the multi-agent system,
        including A2A protocol coordination, memory bridge initialization, and agent
        pre-registration. The startup follows academic best practices for distributed
        financial computing systems.
        
        Startup Sequence:
        1. Enhanced MCP lifecycle manager initialization
        2. A2A Memory Coordinator establishment
        3. Legacy memory bridge fallback setup
        4. Agent registration and pre-start procedures
        5. MCP server endpoint activation
        """
        self.logger.info("Initiating comprehensive startup of Alpha Agent Pool MCP Server...")
        
        # Initialize memory bridge if available
        self.memory_bridge = self._initialize_memory_bridge()
        if not self.memory_bridge:
            self.logger.warning("Memory bridge not available. Continuing without it.")
        
        self.logger.info("Initiating agent pre-registration and startup sequence...")
        
        # Pre-start critical momentum agent for alpha signal generation
        self.start_agent_sync("momentum_agent")
        
        # Allow sufficient initialization time for agent stabilization
        import time
        time.sleep(5)

        self.logger.info(f"Starting AlphaAgentPoolMCPServer on {self.host}:{self.port}")
        self.pool_server.settings.host = self.host
        self.pool_server.settings.port = self.port
        # The run method is blocking, so it will keep the server alive.
        self.pool_server.run(transport="sse")

    async def _async_start(self):
        """
        Asynchronous startup method to handle all async initialization properly.
        """
        # Initialize enhanced lifecycle manager for advanced orchestration
        if self.lifecycle_manager:
            try:
                await self.lifecycle_manager.initialize()
                self.logger.info("✅ Enhanced MCP lifecycle manager initialized successfully")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize enhanced MCP lifecycle manager: {e}")
        
        # Initialize A2A Memory Coordinator for distributed memory operations
        if A2A_COORDINATOR_AVAILABLE:
            try:
                # Asynchronous A2A coordinator initialization
                self.a2a_coordinator = await initialize_pool_coordinator(
                    pool_id="alpha_agent_pool",
                    memory_url="http://127.0.0.1:8010"
                )
                self.logger.info("✅ A2A Memory Coordinator initialized and connected to memory agent")
                
                # Connect strategy research framework to A2A coordinator
                if self.strategy_research_framework:
                    self.strategy_research_framework.a2a_coordinator = self.a2a_coordinator
                    self.logger.info("✅ Strategy Research Framework connected to A2A coordinator")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize A2A Memory Coordinator: {e}")
                self.a2a_coordinator = None
        
        # Initialize memory bridge as fallback
        self.memory_bridge = self._initialize_memory_bridge()
        if not self.memory_bridge:
            self.logger.warning("Memory bridge not available. Continuing without it.")
        
        self.logger.info("Initiating agent pre-registration and startup sequence...")
        
        # Pre-register momentum agent with A2A memory coordinator
        if self.a2a_coordinator:
            try:
                await self.a2a_coordinator.register_agent(
                    agent_id="momentum_agent",
                    agent_type="theory_driven_momentum",
                    agent_config={
                        "strategy_type": "momentum", 
                        "lookback_window": 20,
                        "signal_threshold": 0.05,
                        "risk_adjustment": True
                    }
                )
                self.logger.info("✅ Momentum agent registered with A2A memory coordinator")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to register momentum agent with A2A coordinator: {e}")
        
        # Pre-start critical momentum agent for alpha signal generation
        self.start_agent_sync("momentum_agent")
        
        # Allow sufficient initialization time for agent stabilization
        await asyncio.sleep(5)

        self.logger.info(f"Starting AlphaAgentPoolMCPServer on {self.host}:{self.port}")
        
        # Start server using async method to avoid event loop conflicts
        await self._start_server_async()

    async def _start_server_async(self):
        """
        Start the MCP server in async mode to avoid event loop conflicts.
        """
        self.pool_server.settings.host = self.host
        self.pool_server.settings.port = self.port
        
        # Use the async server startup method instead of sync run()
        await self.pool_server.run_sse_async(mount_path="/sse")

    def start_sync(self):
        """
        Synchronous startup method for use when no asyncio loop is running.
        This is the original start() method behavior.
        """
        # Initialize enhanced lifecycle manager for advanced orchestration
        if self.lifecycle_manager:
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.lifecycle_manager.initialize())
                loop.close()
                self.logger.info("✅ Enhanced MCP lifecycle manager initialized successfully")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize enhanced MCP lifecycle manager: {e}")
        
        # Initialize A2A Memory Coordinator for distributed memory operations
        if A2A_COORDINATOR_AVAILABLE:
            try:
                # Asynchronous A2A coordinator initialization with proper event loop management
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self.a2a_coordinator = loop.run_until_complete(
                    initialize_pool_coordinator(
                        pool_id="alpha_agent_pool",
                        memory_url="http://127.0.0.1:8010"
                    )
                )
                loop.close()
                self.logger.info("✅ A2A Memory Coordinator initialized and connected to memory agent")
                
                # Connect strategy research framework to A2A coordinator
                if self.strategy_research_framework:
                    self.strategy_research_framework.a2a_coordinator = self.a2a_coordinator
                    self.logger.info("✅ Strategy Research Framework connected to A2A coordinator")
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize A2A Memory Coordinator: {e}")
                self.a2a_coordinator = None
        
        # Initialize legacy memory bridge as fallback system
        self.memory_bridge = self._initialize_memory_bridge()

        self.logger.info("Initiating agent pre-registration and startup sequence...")
        
        # Register momentum agent with A2A coordinator for distributed memory coordination
        if self.a2a_coordinator:
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    self.a2a_coordinator.register_agent(
                        agent_id="momentum_agent",
                        agent_type="theory_driven_momentum",
                        agent_config={
                            "strategy_type": "momentum", 
                            "lookback_window": 20,
                            "signal_threshold": 0.05,
                            "risk_adjustment": True
                        }
                    )
                )
                loop.close()
                self.logger.info("✅ Momentum agent registered with A2A memory coordinator")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to register momentum agent with A2A coordinator: {e}")
        
        # Pre-start critical momentum agent for alpha signal generation
        self.start_agent_sync("momentum_agent")
        
        # Allow sufficient initialization time for agent stabilization
        time.sleep(5) 

        self.logger.info(f"Starting AlphaAgentPoolMCPServer on {self.host}:{self.port}")
        self.pool_server.settings.host = self.host
        self.pool_server.settings.port = self.port
        # The run method is blocking, so it will keep the server alive.
        self.pool_server.run(transport="sse")

    async def start_async(self):
        """
        Asynchronous startup method for use when an asyncio loop is already running.
        This method properly handles async initialization without creating event loop conflicts.
        """
        # Initialize enhanced lifecycle manager for advanced orchestration
        if self.lifecycle_manager:
            try:
                await self.lifecycle_manager.initialize()
                self.logger.info("✅ Enhanced MCP lifecycle manager initialized successfully")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize enhanced MCP lifecycle manager: {e}")
        
        # Initialize A2A Memory Coordinator for distributed memory operations
        if A2A_COORDINATOR_AVAILABLE:
            try:
                self.a2a_coordinator = await initialize_pool_coordinator(
                    pool_id="alpha_agent_pool",
                    server_port=self.port,
                    memory_url="http://127.0.0.1:8002"
                )
                self.logger.info("✅ A2A Memory Coordinator initialized and connected to memory agent")
                
                # Connect strategy research framework to A2A coordinator
                if self.strategy_research_framework:
                    self.strategy_research_framework.a2a_coordinator = self.a2a_coordinator
                    self.logger.info("✅ Strategy Research Framework connected to A2A coordinator")
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize A2A Memory Coordinator: {e}")
                self.a2a_coordinator = None
        
        # Initialize legacy memory bridge as fallback system
        self.memory_bridge = self._initialize_memory_bridge()

        self.logger.info("Initiating agent pre-registration and startup sequence...")
        
        # Register momentum agent with A2A coordinator for distributed memory coordination
        if self.a2a_coordinator:
            try:
                await self.a2a_coordinator.register_agent(
                    agent_id="momentum_agent",
                    agent_type="theory_driven_momentum",
                    agent_config={
                        "strategy_type": "momentum", 
                        "lookback_window": 20,
                        "signal_threshold": 0.05,
                        "risk_adjustment": True
                    }
                )
                self.logger.info("✅ Momentum agent registered with A2A memory coordinator")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to register momentum agent with A2A coordinator: {e}")
        
        # Pre-start critical momentum agent for alpha signal generation
        self.start_agent_sync("momentum_agent")
        
        # Allow sufficient initialization time for agent stabilization
        await asyncio.sleep(5)

        self.logger.info(f"Starting AlphaAgentPoolMCPServer on {self.host}:{self.port}")
        
        # Start server using async method to avoid event loop conflicts
        await self._start_server_async()

    async def _start_server_async(self):
        """
        Start the MCP server in async mode to avoid event loop conflicts.
        """
        self.pool_server.settings.host = self.host
        self.pool_server.settings.port = self.port
        
        # Use the async server startup method instead of sync run()
        await self.pool_server.run_sse_async(mount_path="/sse")

    def stop(self):
        """
        Gracefully terminate the Alpha Agent Pool MCP Server and associated subsystems.
        
        This method implements a comprehensive shutdown sequence following academic
        best practices for distributed system termination. The shutdown process
        ensures proper resource cleanup, data persistence, and graceful agent
        process termination.
        
        Shutdown Sequence:
        1. Enhanced MCP lifecycle manager termination
        2. A2A Memory Coordinator disconnection and cleanup
        3. Individual agent process termination with timeout protection
        4. Registry cleanup and resource deallocation
        """
        self.logger.info("Initiating comprehensive shutdown of Alpha Agent Pool MCP Server...")
        
        # Terminate enhanced lifecycle manager with proper resource cleanup
        if self.lifecycle_manager:
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.lifecycle_manager.shutdown())
                loop.close()
                self.logger.info("✅ Enhanced MCP lifecycle manager terminated successfully")
            except Exception as e:
                self.logger.error(f"❌ Failed to terminate enhanced MCP lifecycle manager: {e}")
        
        # Disconnect and shutdown A2A Memory Coordinator with proper protocol cleanup
        if self.a2a_coordinator:
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(shutdown_pool_coordinator())
                loop.close()
                self.logger.info("✅ A2A Memory Coordinator disconnected and shut down successfully")
            except Exception as e:
                self.logger.error(f"❌ Failed to shutdown A2A Memory Coordinator: {e}")
        
        # Graceful termination of all registered agent processes
        for agent_id in list(self.agent_registry.keys()):
            try:
                self.logger.info(f"Terminating agent process '{agent_id}'...")
                
                # Retrieve agent process handle
                process = self.agent_registry[agent_id]
                
                # Attempt graceful termination first
                if hasattr(process, 'terminate'):
                    process.terminate()
                elif hasattr(process, 'kill'):
                    process.kill()
                
                # Wait for process termination with timeout protection
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
        
        # Initialize enhanced memory bridge if available
        if hasattr(server, '_bridge_initialization_pending') and server._bridge_initialization_pending:
            import asyncio
            asyncio.create_task(server._initialize_enhanced_memory_bridge())
        
        logger.info("Starting server...")
        server.start()
    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Stopping server...")
    finally:
        if server:
            server.stop()
        logger.info("Server has been stopped.")
