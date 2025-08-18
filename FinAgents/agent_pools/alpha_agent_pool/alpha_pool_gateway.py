#!/usr/bin/env python3
"""
Alpha Agent Pool Gateway: Marginal Gateway for External Orchestration

This module implements a dual-role gateway that serves as:
1. MCP Server for external orchestration systems
2. MCP Client for internal Alpha Agents

The gateway provides a unified interface for external systems while coordinating
with internal alpha agents through MCP protocol, enabling seamless integration
between external orchestration and internal agent management.

Architecture:
- External Interface: MCP Server (port 8082) for external orchestration
- Internal Interface: MCP Client connections to internal alpha agents
- Memory Coordination: A2A protocol integration for cross-agent learning
- Strategy Orchestration: Unified strategy management across agent pool

Author: FinAgent Research Team
License: Open Source Research License
Created: 2025-08-06
"""

import asyncio
import json
import logging
import os
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, TypedDict
from enum import Enum
from dataclasses import dataclass

# MCP Server and Client imports
from mcp.server.fastmcp import FastMCP
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.stdio import stdio_client

# Configure logging
logger = logging.getLogger(__name__)

# Import internal components
try:
    from .core import AlphaAgentPoolMCPServer, MemoryUnit, AgentStatus
    CORE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Core module not available: {e}")
    CORE_AVAILABLE = False

try:
    from .a2a_memory_coordinator import get_pool_coordinator, shutdown_pool_coordinator
    A2A_COORDINATOR_AVAILABLE = True
except ImportError:
    A2A_COORDINATOR_AVAILABLE = False

try:
    from .alpha_strategy_research import AlphaStrategyResearchFramework
    STRATEGY_RESEARCH_AVAILABLE = True
except ImportError:
    STRATEGY_RESEARCH_AVAILABLE = False

# Gateway configuration
GATEWAY_DEFAULT_PORT = 8082
INTERNAL_AGENTS = {
    "momentum_agent": "http://127.0.0.1:5051/mcp",
    "autonomous_agent": "http://127.0.0.1:5052/mcp", 
    "alpha_pool_server": "http://127.0.0.1:8081/mcp"
}

class GatewayStatus(Enum):
    """Gateway operational status enumeration"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class AgentConnectionStatus(Enum):
    """Internal agent connection status"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"

@dataclass
class AgentConnection:
    """Represents connection to an internal agent"""
    agent_id: str
    endpoint: str
    client: Optional[ClientSession] = None
    status: AgentConnectionStatus = AgentConnectionStatus.DISCONNECTED
    last_ping: Optional[datetime] = None
    error_count: int = 0

class OrchestrationTask(TypedDict):
    """Structure for orchestration tasks"""
    task_id: str
    task_type: str
    target_agents: List[str]
    parameters: Dict[str, Any]
    status: str
    created_at: str
    updated_at: str

class AlphaPoolGateway:
    """
    Alpha Agent Pool Gateway
    
    Serves as a dual-role gateway providing:
    - MCP Server interface for external orchestration systems
    - MCP Client connections to internal alpha agents
    - Unified strategy coordination and memory management
    - Cross-agent communication and task orchestration
    """
    
    def __init__(self, 
                 host: str = "0.0.0.0", 
                 port: int = GATEWAY_DEFAULT_PORT,
                 enable_internal_pool: bool = True):
        """
        Initialize the Alpha Pool Gateway
        
        Args:
            host: Network interface binding address
            port: TCP port for external MCP server
            enable_internal_pool: Whether to start internal agent pool
        """
        self.host = host
        self.port = port
        self.enable_internal_pool = enable_internal_pool
        
        # Initialize MCP server for external interface
        self.gateway_server = FastMCP("AlphaPoolGateway")
        
        # Gateway state management
        self.status = GatewayStatus.INITIALIZING
        self.session_id = f"gateway_{int(time.time())}"
        
        # Internal agent connections
        self.agent_connections: Dict[str, AgentConnection] = {}
        self.connection_lock = asyncio.Lock()
        
        # Task orchestration
        self.active_tasks: Dict[str, OrchestrationTask] = {}
        self.task_results: Dict[str, Any] = {}
        
        # Memory and coordination
        self.memory_unit = None
        self.a2a_coordinator = None
        self.strategy_framework = None
        
        # Internal alpha pool server (optional)
        self.internal_pool: Optional[AlphaAgentPoolMCPServer] = None
        
        # Performance tracking
        self.performance_metrics = {
            "requests_processed": 0,
            "tasks_orchestrated": 0,
            "agents_coordinated": 0,
            "errors_encountered": 0,
            "uptime_start": datetime.now(timezone.utc)
        }
        
        # Initialize components
        self._initialize_gateway()
        
    def _initialize_gateway(self):
        """Initialize gateway components and register tools"""
        logger.info(f"Initializing Alpha Pool Gateway on {self.host}:{self.port}")
        
        # Initialize memory unit
        memory_path = os.path.join(os.path.dirname(__file__), "gateway_memory.json")
        self.memory_unit = MemoryUnit(memory_path, reset_on_init=False)
        
        # Register gateway tools
        self._register_gateway_tools()
        
        # Initialize internal agent connections
        for agent_id, endpoint in INTERNAL_AGENTS.items():
            self.agent_connections[agent_id] = AgentConnection(
                agent_id=agent_id,
                endpoint=endpoint
            )
        
        logger.info("Gateway initialization completed")
        
    def _register_gateway_tools(self):
        """Register MCP tools for external orchestration"""
        
        @self.gateway_server.tool(
            name="gateway_status",
            description="Get comprehensive gateway status and health information"
        )
        async def gateway_status() -> Dict[str, Any]:
            """Return comprehensive gateway status"""
            return await self._get_gateway_status()
        
        @self.gateway_server.tool(
            name="list_internal_agents", 
            description="List all internal agents and their connection status"
        )
        async def list_internal_agents() -> Dict[str, Any]:
            """List internal agents with their connection status"""
            return await self._list_internal_agents()
        
        @self.gateway_server.tool(
            name="orchestrate_task",
            description="Orchestrate a task across multiple internal agents"
        )
        async def orchestrate_task(
            task_type: str,
            target_agents: List[str],
            parameters: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Orchestrate a task across internal agents"""
            return await self._orchestrate_task(task_type, target_agents, parameters)
        
        @self.gateway_server.tool(
            name="get_task_status",
            description="Get the status and results of an orchestrated task"
        )
        async def get_task_status(task_id: str) -> Dict[str, Any]:
            """Get task status and results"""
            return await self._get_task_status(task_id)
        
        @self.gateway_server.tool(
            name="call_internal_agent",
            description="Make direct call to internal agent via MCP"
        )
        async def call_internal_agent(
            agent_id: str,
            tool_name: str,
            parameters: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Call internal agent tool directly"""
            return await self._call_internal_agent(agent_id, tool_name, parameters)
        
        @self.gateway_server.tool(
            name="generate_alpha_strategy",
            description="Generate comprehensive alpha strategy using internal agents"
        )
        async def generate_alpha_strategy(
            symbols: List[str],
            strategy_type: str = "momentum",
            timeframe: str = "1d",
            lookback_period: int = 20
        ) -> Dict[str, Any]:
            """Generate alpha strategy using coordinated agents"""
            return await self._generate_alpha_strategy(symbols, strategy_type, timeframe, lookback_period)
        
        @self.gateway_server.tool(
            name="coordinate_memory_sync",
            description="Coordinate memory synchronization across agents"
        )
        async def coordinate_memory_sync() -> Dict[str, Any]:
            """Coordinate A2A memory synchronization"""
            return await self._coordinate_memory_sync()
        
        @self.gateway_server.tool(
            name="get_performance_metrics",
            description="Get gateway and agent performance metrics"
        )
        async def get_performance_metrics() -> Dict[str, Any]:
            """Get comprehensive performance metrics"""
            return await self._get_performance_metrics()
        
        @self.gateway_server.tool(
            name="emergency_shutdown",
            description="Emergency shutdown of gateway and internal agents"
        )
        async def emergency_shutdown() -> Dict[str, Any]:
            """Perform emergency shutdown"""
            return await self._emergency_shutdown()

    async def _get_gateway_status(self) -> Dict[str, Any]:
        """Get comprehensive gateway status"""
        self.performance_metrics["requests_processed"] += 1
        
        # Check internal agent connections
        connection_summary = {}
        for agent_id, conn in self.agent_connections.items():
            connection_summary[agent_id] = {
                "status": conn.status.value,
                "last_ping": conn.last_ping.isoformat() if conn.last_ping else None,
                "error_count": conn.error_count
            }
        
        return {
            "gateway_status": self.status.value,
            "session_id": self.session_id,
            "uptime_seconds": (datetime.now(timezone.utc) - self.performance_metrics["uptime_start"]).total_seconds(),
            "agent_connections": connection_summary,
            "active_tasks": len(self.active_tasks),
            "performance_metrics": self.performance_metrics,
            "memory_keys_count": len(self.memory_unit.keys()) if self.memory_unit else 0,
            "internal_pool_active": self.internal_pool is not None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def _list_internal_agents(self) -> Dict[str, Any]:
        """List internal agents and their capabilities"""
        self.performance_metrics["requests_processed"] += 1
        
        agents_info = {}
        
        for agent_id, conn in self.agent_connections.items():
            agent_info = {
                "agent_id": agent_id,
                "endpoint": conn.endpoint,
                "connection_status": conn.status.value,
                "capabilities": []
            }
            
            # Try to get agent capabilities if connected
            if conn.status == AgentConnectionStatus.CONNECTED and conn.client:
                try:
                    # Get available tools from the agent
                    tools_result = await conn.client.list_tools()
                    if tools_result and hasattr(tools_result, 'tools'):
                        agent_info["capabilities"] = [tool.name for tool in tools_result.tools]
                except Exception as e:
                    logger.warning(f"Failed to get capabilities for {agent_id}: {e}")
                    agent_info["error"] = str(e)
            
            agents_info[agent_id] = agent_info
        
        return {
            "agents": agents_info,
            "total_agents": len(agents_info),
            "connected_agents": len([a for a in self.agent_connections.values() 
                                   if a.status == AgentConnectionStatus.CONNECTED]),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def _orchestrate_task(self, 
                               task_type: str, 
                               target_agents: List[str], 
                               parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate a task across multiple internal agents"""
        self.performance_metrics["requests_processed"] += 1
        self.performance_metrics["tasks_orchestrated"] += 1
        
        # Generate task ID
        task_id = f"task_{int(time.time())}_{len(self.active_tasks)}"
        
        # Create task record
        task = OrchestrationTask(
            task_id=task_id,
            task_type=task_type,
            target_agents=target_agents,
            parameters=parameters,
            status="initiated",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat()
        )
        
        self.active_tasks[task_id] = task
        
        try:
            # Execute task based on type
            if task_type == "alpha_signal_generation":
                result = await self._execute_alpha_signal_task(task_id, target_agents, parameters)
            elif task_type == "strategy_backtesting":
                result = await self._execute_backtesting_task(task_id, target_agents, parameters)
            elif task_type == "memory_coordination":
                result = await self._execute_memory_coordination_task(task_id, target_agents, parameters)
            else:
                result = await self._execute_generic_task(task_id, target_agents, parameters)
            
            # Update task status
            task["status"] = "completed"
            task["updated_at"] = datetime.now(timezone.utc).isoformat()
            self.task_results[task_id] = result
            
            return {
                "task_id": task_id,
                "status": "completed",
                "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            # Handle task execution error
            task["status"] = "error"
            task["updated_at"] = datetime.now(timezone.utc).isoformat()
            self.performance_metrics["errors_encountered"] += 1
            
            error_result = {
                "error": str(e),
                "error_type": type(e).__name__
            }
            self.task_results[task_id] = error_result
            
            return {
                "task_id": task_id,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def _execute_alpha_signal_task(self, 
                                        task_id: str, 
                                        target_agents: List[str], 
                                        parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute alpha signal generation task"""
        logger.info(f"Executing alpha signal task {task_id} with agents: {target_agents}")
        
        results = {}
        
        # Call momentum agent for signal generation
        if "momentum_agent" in target_agents:
            momentum_result = await self._call_internal_agent(
                "momentum_agent",
                "generate_momentum_signal",
                parameters
            )
            results["momentum_signals"] = momentum_result
        
        # Call alpha pool server for comprehensive analysis
        if "alpha_pool_server" in target_agents:
            pool_result = await self._call_internal_agent(
                "alpha_pool_server",
                "generate_alpha_signals",
                parameters
            )
            results["alpha_pool_signals"] = pool_result
        
        # Aggregate results
        aggregated_result = {
            "task_id": task_id,
            "task_type": "alpha_signal_generation",
            "individual_results": results,
            "aggregated_signals": self._aggregate_signals(results),
            "execution_time": datetime.now(timezone.utc).isoformat()
        }
        
        return aggregated_result
    
    async def _execute_backtesting_task(self, 
                                       task_id: str, 
                                       target_agents: List[str], 
                                       parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute strategy backtesting task"""
        logger.info(f"Executing backtesting task {task_id} with agents: {target_agents}")
        
        # Implement backtesting coordination logic
        results = {}
        
        for agent_id in target_agents:
            if agent_id in self.agent_connections:
                try:
                    result = await self._call_internal_agent(
                        agent_id,
                        "run_backtest",
                        parameters
                    )
                    results[agent_id] = result
                except Exception as e:
                    results[agent_id] = {"error": str(e)}
        
        return {
            "task_id": task_id,
            "task_type": "strategy_backtesting",
            "backtest_results": results,
            "execution_time": datetime.now(timezone.utc).isoformat()
        }
    
    async def _execute_memory_coordination_task(self, 
                                               task_id: str, 
                                               target_agents: List[str], 
                                               parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute memory coordination task"""
        logger.info(f"Executing memory coordination task {task_id} with agents: {target_agents}")
        
        # Coordinate A2A memory synchronization
        coordination_results = {}
        
        if self.a2a_coordinator:
            try:
                sync_result = await self.a2a_coordinator.synchronize_memories(target_agents)
                coordination_results["a2a_sync"] = sync_result
            except Exception as e:
                coordination_results["a2a_sync"] = {"error": str(e)}
        
        return {
            "task_id": task_id,
            "task_type": "memory_coordination", 
            "coordination_results": coordination_results,
            "execution_time": datetime.now(timezone.utc).isoformat()
        }
    
    async def _execute_generic_task(self, 
                                   task_id: str, 
                                   target_agents: List[str], 
                                   parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute generic task across agents"""
        logger.info(f"Executing generic task {task_id} with agents: {target_agents}")
        
        results = {}
        tool_name = parameters.get("tool_name", "health_check")
        tool_params = parameters.get("tool_params", {})
        
        for agent_id in target_agents:
            if agent_id in self.agent_connections:
                try:
                    result = await self._call_internal_agent(agent_id, tool_name, tool_params)
                    results[agent_id] = result
                except Exception as e:
                    results[agent_id] = {"error": str(e)}
        
        return {
            "task_id": task_id,
            "task_type": "generic",
            "agent_results": results,
            "execution_time": datetime.now(timezone.utc).isoformat()
        }
    
    def _aggregate_signals(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate signals from multiple agents"""
        # Simple aggregation logic - can be enhanced
        aggregated = {
            "signal_strength": 0.0,
            "confidence": 0.0,
            "recommendation": "HOLD",
            "contributing_agents": list(results.keys())
        }
        
        signal_count = 0
        total_strength = 0.0
        total_confidence = 0.0
        
        for agent_id, result in results.items():
            if isinstance(result, dict) and "signal" in result:
                signal_data = result["signal"]
                if "strength" in signal_data:
                    total_strength += float(signal_data["strength"])
                    signal_count += 1
                if "confidence" in signal_data:
                    total_confidence += float(signal_data["confidence"])
        
        if signal_count > 0:
            aggregated["signal_strength"] = total_strength / signal_count
            aggregated["confidence"] = total_confidence / signal_count
            
            # Determine recommendation based on average signal strength
            if aggregated["signal_strength"] > 0.6:
                aggregated["recommendation"] = "BUY"
            elif aggregated["signal_strength"] < -0.6:
                aggregated["recommendation"] = "SELL"
        
        return aggregated
    
    async def _get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status and results"""
        self.performance_metrics["requests_processed"] += 1
        
        if task_id not in self.active_tasks:
            return {
                "error": f"Task {task_id} not found",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        task = self.active_tasks[task_id]
        result = {
            "task_id": task_id,
            "status": task["status"],
            "task_type": task["task_type"],
            "target_agents": task["target_agents"],
            "created_at": task["created_at"],
            "updated_at": task["updated_at"]
        }
        
        if task_id in self.task_results:
            result["results"] = self.task_results[task_id]
        
        return result
    
    async def _call_internal_agent(self, 
                                  agent_id: str, 
                                  tool_name: str, 
                                  parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Make direct call to internal agent"""
        if agent_id not in self.agent_connections:
            raise ValueError(f"Agent {agent_id} not found in connections")
        
        conn = self.agent_connections[agent_id]
        
        # Ensure agent is connected
        if conn.status != AgentConnectionStatus.CONNECTED:
            await self._connect_to_agent(agent_id)
        
        if conn.client is None:
            raise RuntimeError(f"Failed to establish connection to agent {agent_id}")
        
        try:
            # Call the tool on the agent
            result = await conn.client.call_tool(tool_name, parameters)
            
            # Update connection health
            conn.last_ping = datetime.now(timezone.utc)
            conn.error_count = 0
            
            return {
                "agent_id": agent_id,
                "tool_name": tool_name,
                "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            conn.error_count += 1
            logger.error(f"Error calling {tool_name} on {agent_id}: {e}")
            
            # Reconnect if too many errors
            if conn.error_count >= 3:
                conn.status = AgentConnectionStatus.ERROR
                conn.client = None
            
            raise RuntimeError(f"Failed to call {tool_name} on {agent_id}: {e}")
    
    async def _connect_to_agent(self, agent_id: str) -> bool:
        """Establish connection to internal agent"""
        if agent_id not in self.agent_connections:
            return False
        
        conn = self.agent_connections[agent_id]
        
        async with self.connection_lock:
            if conn.status == AgentConnectionStatus.CONNECTED:
                return True
            
            conn.status = AgentConnectionStatus.CONNECTING
            
            try:
                # Connect via HTTP MCP client
                client = streamablehttp_client(conn.endpoint)
                read, write, _ = await client.__aenter__()
                session = ClientSession(read, write)
                await session.initialize()
                
                conn.client = session
                conn.status = AgentConnectionStatus.CONNECTED
                conn.last_ping = datetime.now(timezone.utc)
                conn.error_count = 0
                
                logger.info(f"Successfully connected to agent {agent_id} at {conn.endpoint}")
                return True
                
            except Exception as e:
                conn.status = AgentConnectionStatus.ERROR
                conn.client = None
                conn.error_count += 1
                
                logger.error(f"Failed to connect to agent {agent_id}: {e}")
                return False
    
    async def _generate_alpha_strategy(self, 
                                      symbols: List[str], 
                                      strategy_type: str, 
                                      timeframe: str, 
                                      lookback_period: int) -> Dict[str, Any]:
        """Generate comprehensive alpha strategy"""
        self.performance_metrics["requests_processed"] += 1
        
        # Orchestrate strategy generation across multiple agents
        task_params = {
            "symbols": symbols,
            "strategy_type": strategy_type,
            "timeframe": timeframe,
            "lookback_period": lookback_period
        }
        
        # Use orchestration to coordinate strategy generation
        result = await self._orchestrate_task(
            task_type="alpha_signal_generation",
            target_agents=["momentum_agent", "alpha_pool_server"],
            parameters=task_params
        )
        
        # Enhance with strategy framework if available
        if self.strategy_framework:
            try:
                strategy_analysis = await self.strategy_framework.analyze_strategy(
                    symbols=symbols,
                    strategy_type=strategy_type
                )
                result["strategy_analysis"] = strategy_analysis
            except Exception as e:
                logger.warning(f"Strategy framework analysis failed: {e}")
        
        return result
    
    async def _coordinate_memory_sync(self) -> Dict[str, Any]:
        """Coordinate A2A memory synchronization"""
        self.performance_metrics["requests_processed"] += 1
        
        sync_results = {}
        
        if A2A_COORDINATOR_AVAILABLE:
            try:
                if not self.a2a_coordinator:
                    self.a2a_coordinator = await get_pool_coordinator(f"gateway_{self.port}")
                
                # Perform memory synchronization
                agents_to_sync = list(self.agent_connections.keys())
                sync_result = await self._orchestrate_task(
                    task_type="memory_coordination",
                    target_agents=agents_to_sync,
                    parameters={"sync_type": "full"}
                )
                
                sync_results["coordination_result"] = sync_result
                
            except Exception as e:
                sync_results["error"] = str(e)
        else:
            sync_results["error"] = "A2A coordinator not available"
        
        return sync_results
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        self.performance_metrics["requests_processed"] += 1
        
        # Calculate uptime
        uptime = datetime.now(timezone.utc) - self.performance_metrics["uptime_start"]
        
        # Get agent connection health
        agent_health = {}
        for agent_id, conn in self.agent_connections.items():
            agent_health[agent_id] = {
                "status": conn.status.value,
                "error_count": conn.error_count,
                "last_ping": conn.last_ping.isoformat() if conn.last_ping else None
            }
        
        return {
            "gateway_metrics": {
                **self.performance_metrics,
                "uptime_seconds": uptime.total_seconds(),
                "uptime_formatted": str(uptime)
            },
            "agent_health": agent_health,
            "task_statistics": {
                "active_tasks": len(self.active_tasks),
                "completed_tasks": len(self.task_results),
                "task_types": list(set(task["task_type"] for task in self.active_tasks.values()))
            },
            "memory_statistics": {
                "memory_keys": len(self.memory_unit.keys()) if self.memory_unit else 0,
                "a2a_coordinator_active": self.a2a_coordinator is not None
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def _emergency_shutdown(self) -> Dict[str, Any]:
        """Perform emergency shutdown"""
        logger.warning("Emergency shutdown initiated")
        
        self.status = GatewayStatus.SHUTDOWN
        shutdown_results = {}
        
        try:
            # Disconnect from all agents
            for agent_id, conn in self.agent_connections.items():
                if conn.client:
                    try:
                        await conn.client.close()
                        shutdown_results[f"{agent_id}_disconnected"] = True
                    except Exception as e:
                        shutdown_results[f"{agent_id}_disconnect_error"] = str(e)
            
            # Shutdown A2A coordinator
            if self.a2a_coordinator:
                try:
                    await shutdown_pool_coordinator(f"gateway_{self.port}")
                    shutdown_results["a2a_coordinator_shutdown"] = True
                except Exception as e:
                    shutdown_results["a2a_coordinator_shutdown_error"] = str(e)
            
            # Shutdown internal pool if running
            if self.internal_pool:
                try:
                    # Add shutdown logic for internal pool
                    shutdown_results["internal_pool_shutdown"] = True
                except Exception as e:
                    shutdown_results["internal_pool_shutdown_error"] = str(e)
            
            return {
                "status": "shutdown_completed",
                "shutdown_results": shutdown_results,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "status": "shutdown_error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def start_gateway(self):
        """Start the gateway server and initialize connections"""
        logger.info(f"Starting Alpha Pool Gateway on {self.host}:{self.port}")
        
        try:
            # Initialize A2A coordinator if available
            if A2A_COORDINATOR_AVAILABLE:
                try:
                    self.a2a_coordinator = await get_pool_coordinator(f"gateway_{self.port}")
                    logger.info("A2A Memory Coordinator initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize A2A coordinator: {e}")
            
            # Initialize strategy framework if available
            if STRATEGY_RESEARCH_AVAILABLE:
                try:
                    self.strategy_framework = AlphaStrategyResearchFramework()
                    logger.info("Strategy Research Framework initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize strategy framework: {e}")
            
            # Start internal pool if enabled
            if self.enable_internal_pool and CORE_AVAILABLE:
                try:
                    self.internal_pool = AlphaAgentPoolMCPServer(
                        host="127.0.0.1", 
                        port=8081
                    )
                    # Start internal pool in background
                    asyncio.create_task(self.internal_pool.start())
                    logger.info("Internal Alpha Pool Server started")
                except Exception as e:
                    logger.warning(f"Failed to start internal pool: {e}")
            
            # Connect to internal agents
            connection_tasks = []
            for agent_id in self.agent_connections.keys():
                task = asyncio.create_task(self._connect_to_agent(agent_id))
                connection_tasks.append(task)
            
            # Wait for initial connections (with timeout)
            try:
                await asyncio.wait_for(
                    asyncio.gather(*connection_tasks, return_exceptions=True),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.warning("Some agent connections timed out during startup")
            
            # Set gateway status to active
            self.status = GatewayStatus.ACTIVE
            
            # Start the MCP server
            await self.gateway_server.start_server_async(host=self.host, port=self.port)
            
        except Exception as e:
            logger.error(f"Failed to start gateway: {e}")
            self.status = GatewayStatus.ERROR
            raise
    
    async def stop_gateway(self):
        """Stop the gateway and cleanup resources"""
        logger.info("Stopping Alpha Pool Gateway")
        
        self.status = GatewayStatus.SHUTDOWN
        
        # Disconnect from all agents
        for agent_id, conn in self.agent_connections.items():
            if conn.client:
                try:
                    await conn.client.close()
                    logger.info(f"Disconnected from agent {agent_id}")
                except Exception as e:
                    logger.error(f"Error disconnecting from {agent_id}: {e}")
        
        # Cleanup A2A coordinator
        if self.a2a_coordinator:
            try:
                await shutdown_pool_coordinator(f"gateway_{self.port}")
                logger.info("A2A coordinator shutdown")
            except Exception as e:
                logger.error(f"Error shutting down A2A coordinator: {e}")
        
        # Stop internal pool if running
        if self.internal_pool:
            try:
                # Add internal pool shutdown logic if available
                logger.info("Internal pool shutdown")
            except Exception as e:
                logger.error(f"Error shutting down internal pool: {e}")
        
        logger.info("Gateway shutdown completed")

def check_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a port is available for binding"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            return result != 0
    except Exception:
        return False

def find_available_port(start_port: int, max_attempts: int = 10) -> Optional[int]:
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        if check_port_available(port):
            return port
    return None

async def main():
    """Main entry point for the gateway"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Find available port if default is occupied
    port = GATEWAY_DEFAULT_PORT
    if not check_port_available(port):
        logger.warning(f"Default port {port} is occupied, finding alternative...")
        alternative_port = find_available_port(port + 1)
        if alternative_port:
            port = alternative_port
            logger.info(f"Using alternative port {port}")
        else:
            logger.error("No available ports found")
            return
    
    # Create and start gateway
    gateway = AlphaPoolGateway(port=port)
    
    try:
        await gateway.start_gateway()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Gateway error: {e}")
    finally:
        await gateway.stop_gateway()

if __name__ == "__main__":
    asyncio.run(main())
