"""
Enhanced MCP Lifecycle Management for Alpha Agent Pool

This module extends the FastMCP server with comprehensive lifecycle management,
health monitoring, and resource management capabilities for the Alpha Agent Pool.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from enum import Enum
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import os

from mcp.server.fastmcp import FastMCP
from mcp.types import JSONRPCMessage, RequestId


logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Enum representing the possible states of an agent."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPING = "stopping"
    STOPPED = "stopped"


class HealthStatus(Enum):
    """Enum representing health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class AgentMetrics:
    """Metrics for an individual agent."""
    agent_id: str
    state: AgentState
    health_status: HealthStatus
    last_heartbeat: datetime
    total_requests: int
    failed_requests: int
    average_response_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    uptime_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        data = asdict(self)
        data['state'] = self.state.value
        data['health_status'] = self.health_status.value
        data['last_heartbeat'] = self.last_heartbeat.isoformat()
        return data


@dataclass
class PoolMetrics:
    """Metrics for the entire agent pool."""
    pool_id: str
    total_agents: int
    running_agents: int
    healthy_agents: int
    total_pool_requests: int
    pool_uptime_seconds: float
    memory_coordinator_status: str
    a2a_connections: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pool metrics to dictionary format."""
        return asdict(self)


class EnhancedMCPLifecycleManager:
    """
    Enhanced MCP lifecycle manager providing comprehensive agent pool management.
    
    This class extends the basic MCP server functionality with:
    - Agent state management and monitoring
    - Health checking and alerting
    - Resource usage tracking
    - Performance metrics collection
    - Graceful shutdown and recovery
    - A2A protocol coordination
    """
    
    def __init__(self, mcp_server: FastMCP, pool_id: str = "alpha_pool"):
        """
        Initialize the enhanced lifecycle manager.
        
        Args:
            mcp_server: The FastMCP server instance
            pool_id: Identifier for this agent pool
        """
        self.mcp_server = mcp_server
        self.pool_id = pool_id
        self.start_time = datetime.now()
        
        # Agent management
        self.agents: Dict[str, AgentMetrics] = {}
        self.agent_tasks: Dict[str, asyncio.Task] = {}
        self.agent_states: Dict[str, AgentState] = {}
        
        # Health monitoring
        self.health_check_interval = 30  # seconds
        self.health_check_task: Optional[asyncio.Task] = None
        self.critical_alerts: Set[str] = set()
        
        # Performance tracking
        self.request_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
        
        # A2A coordinator reference
        self.a2a_coordinator = None
        
        # Shutdown event
        self.shutdown_event = asyncio.Event()
        
        logger.info(f"Enhanced MCP lifecycle manager initialized for pool {pool_id}")
    
    async def initialize(self):
        """Initialize the lifecycle manager and start background tasks."""
        logger.info("Initializing enhanced MCP lifecycle manager...")
        
        # Register MCP tools for lifecycle management
        await self._register_mcp_tools()
        
        # Start health monitoring
        self.health_check_task = asyncio.create_task(self._health_monitor_loop())
        
        # Initialize A2A coordinator if available
        try:
            from .a2a_memory_coordinator import initialize_pool_coordinator, get_pool_coordinator
            memory_url = os.getenv("ALPHA_POOL_MEMORY_URL", "http://127.0.0.1:8010")
            self.a2a_coordinator = await initialize_pool_coordinator(
                pool_id=self.pool_id,
                memory_url=memory_url
            )
            logger.info("A2A coordinator initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize A2A coordinator: {e}")
        
        logger.info("Enhanced MCP lifecycle manager initialization completed")
    
    async def _register_mcp_tools(self):
        """Register MCP tools for lifecycle management."""
        
        @self.mcp_server.tool()
        async def get_agent_status(agent_id: str = None) -> Dict[str, Any]:
            """
            Get the status of a specific agent or all agents.
            
            Args:
                agent_id: Optional agent ID. If not provided, returns all agents.
            
            Returns:
                Dictionary containing agent status information.
            """
            if agent_id:
                if agent_id in self.agents:
                    return self.agents[agent_id].to_dict()
                else:
                    return {"error": f"Agent {agent_id} not found"}
            else:
                return {
                    "agents": {aid: metrics.to_dict() for aid, metrics in self.agents.items()},
                    "pool_metrics": self._get_pool_metrics().to_dict()
                }
        
        @self.mcp_server.tool()
        async def start_agent(agent_id: str, agent_type: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
            """
            Start a new agent in the pool.
            
            Args:
                agent_id: Unique identifier for the agent
                agent_type: Type of agent to start (e.g., 'momentum', 'trend_following')
                config: Optional configuration for the agent
            
            Returns:
                Dictionary containing the result of the start operation.
            """
            try:
                if agent_id in self.agents:
                    return {"error": f"Agent {agent_id} already exists"}
                
                # Create initial metrics
                metrics = AgentMetrics(
                    agent_id=agent_id,
                    state=AgentState.INITIALIZING,
                    health_status=HealthStatus.UNKNOWN,
                    last_heartbeat=datetime.now(),
                    total_requests=0,
                    failed_requests=0,
                    average_response_time=0.0,
                    memory_usage_mb=0.0,
                    cpu_usage_percent=0.0,
                    uptime_seconds=0.0
                )
                
                self.agents[agent_id] = metrics
                self.agent_states[agent_id] = AgentState.INITIALIZING
                
                # Start agent task
                task = asyncio.create_task(self._run_agent(agent_id, agent_type, config or {}))
                self.agent_tasks[agent_id] = task
                
                # Register with A2A coordinator if available
                if self.a2a_coordinator:
                    await self.a2a_coordinator.register_agent(agent_id)
                
                logger.info(f"Started agent {agent_id} of type {agent_type}")
                
                return {
                    "success": True,
                    "agent_id": agent_id,
                    "state": AgentState.INITIALIZING.value
                }
                
            except Exception as e:
                logger.error(f"Failed to start agent {agent_id}: {e}")
                return {"error": str(e)}
        
        @self.mcp_server.tool()
        async def stop_agent(agent_id: str, graceful: bool = True) -> Dict[str, Any]:
            """
            Stop a running agent.
            
            Args:
                agent_id: ID of the agent to stop
                graceful: Whether to perform a graceful shutdown
            
            Returns:
                Dictionary containing the result of the stop operation.
            """
            try:
                if agent_id not in self.agents:
                    return {"error": f"Agent {agent_id} not found"}
                
                # Update state
                self.agent_states[agent_id] = AgentState.STOPPING
                self.agents[agent_id].state = AgentState.STOPPING
                
                # Cancel agent task
                if agent_id in self.agent_tasks:
                    task = self.agent_tasks[agent_id]
                    if not task.done():
                        if graceful:
                            task.cancel()
                            try:
                                await asyncio.wait_for(task, timeout=10.0)
                            except asyncio.TimeoutError:
                                logger.warning(f"Graceful shutdown timeout for agent {agent_id}")
                        else:
                            task.cancel()
                    
                    del self.agent_tasks[agent_id]
                
                # Update final state
                self.agent_states[agent_id] = AgentState.STOPPED
                self.agents[agent_id].state = AgentState.STOPPED
                
                # Unregister from A2A coordinator
                if self.a2a_coordinator:
                    await self.a2a_coordinator.unregister_agent(agent_id)
                
                logger.info(f"Stopped agent {agent_id}")
                
                return {
                    "success": True,
                    "agent_id": agent_id,
                    "state": AgentState.STOPPED.value
                }
                
            except Exception as e:
                logger.error(f"Failed to stop agent {agent_id}: {e}")
                return {"error": str(e)}
        
        @self.mcp_server.tool()
        async def restart_agent(agent_id: str) -> Dict[str, Any]:
            """
            Restart an existing agent.
            
            Args:
                agent_id: ID of the agent to restart
            
            Returns:
                Dictionary containing the result of the restart operation.
            """
            try:
                if agent_id not in self.agents:
                    return {"error": f"Agent {agent_id} not found"}
                
                # Stop the agent first
                stop_result = await stop_agent(agent_id, graceful=True)
                if not stop_result.get("success"):
                    return stop_result
                
                # Wait a moment
                await asyncio.sleep(1.0)
                
                # Start the agent again (assuming we have the config stored)
                # For now, use default config
                start_result = await start_agent(agent_id, "momentum", {})
                
                return start_result
                
            except Exception as e:
                logger.error(f"Failed to restart agent {agent_id}: {e}")
                return {"error": str(e)}
        
        @self.mcp_server.tool()
        async def get_pool_health() -> Dict[str, Any]:
            """
            Get comprehensive health information for the entire pool.
            
            Returns:
                Dictionary containing pool health metrics and status.
            """
            pool_metrics = self._get_pool_metrics()
            
            # Calculate health score
            health_score = 100.0
            health_status = HealthStatus.HEALTHY
            
            if pool_metrics.total_agents == 0:
                health_score = 0.0
                health_status = HealthStatus.CRITICAL
            else:
                healthy_ratio = pool_metrics.healthy_agents / pool_metrics.total_agents
                health_score = healthy_ratio * 100
                
                if health_score < 50:
                    health_status = HealthStatus.CRITICAL
                elif health_score < 80:
                    health_status = HealthStatus.WARNING
            
            return {
                "pool_metrics": pool_metrics.to_dict(),
                "health_score": health_score,
                "health_status": health_status.value,
                "critical_alerts": list(self.critical_alerts),
                "uptime_hours": pool_metrics.pool_uptime_seconds / 3600,
                "a2a_coordinator_active": self.a2a_coordinator is not None
            }
        
        @self.mcp_server.tool()
        async def get_performance_metrics() -> Dict[str, Any]:
            """
            Get detailed performance metrics for the pool.
            
            Returns:
                Dictionary containing performance analytics.
            """
            try:
                # Calculate aggregate metrics
                total_requests = sum(agent.total_requests for agent in self.agents.values())
                total_failures = sum(agent.failed_requests for agent in self.agents.values())
                avg_response_time = sum(agent.average_response_time for agent in self.agents.values()) / len(self.agents) if self.agents else 0
                
                # Memory and CPU usage
                total_memory = sum(agent.memory_usage_mb for agent in self.agents.values())
                avg_cpu = sum(agent.cpu_usage_percent for agent in self.agents.values()) / len(self.agents) if self.agents else 0
                
                return {
                    "aggregate_metrics": {
                        "total_requests": total_requests,
                        "total_failures": total_failures,
                        "success_rate": (total_requests - total_failures) / total_requests if total_requests > 0 else 0,
                        "average_response_time_ms": avg_response_time,
                        "total_memory_usage_mb": total_memory,
                        "average_cpu_usage_percent": avg_cpu
                    },
                    "agent_metrics": {
                        agent_id: metrics.to_dict() 
                        for agent_id, metrics in self.agents.items()
                    },
                    "recent_requests": self.request_history[-50:] if self.request_history else []
                }
                
            except Exception as e:
                logger.error(f"Failed to get performance metrics: {e}")
                return {"error": str(e)}
        
        @self.mcp_server.tool()
        async def configure_agent(agent_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
            """
            Update configuration for a running agent.
            
            Args:
                agent_id: ID of the agent to configure
                config: New configuration parameters
            
            Returns:
                Dictionary containing the result of the configuration update.
            """
            try:
                if agent_id not in self.agents:
                    return {"error": f"Agent {agent_id} not found"}
                
                # For now, this is a placeholder
                # In a real implementation, you would send the config to the agent
                logger.info(f"Configuration updated for agent {agent_id}: {config}")
                
                return {
                    "success": True,
                    "agent_id": agent_id,
                    "updated_config": config
                }
                
            except Exception as e:
                logger.error(f"Failed to configure agent {agent_id}: {e}")
                return {"error": str(e)}
    
    async def _run_agent(self, agent_id: str, agent_type: str, config: Dict[str, Any]):
        """
        Run an individual agent in a dedicated task.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent to run
            config: Agent configuration
        """
        try:
            logger.info(f"Starting agent {agent_id} of type {agent_type}")
            
            # Update state to running
            self.agent_states[agent_id] = AgentState.RUNNING
            self.agents[agent_id].state = AgentState.RUNNING
            self.agents[agent_id].health_status = HealthStatus.HEALTHY
            
            start_time = time.time()
            
            # Simulate agent work (replace with actual agent logic)
            while not self.shutdown_event.is_set():
                try:
                    # Update heartbeat
                    self.agents[agent_id].last_heartbeat = datetime.now()
                    self.agents[agent_id].uptime_seconds = time.time() - start_time
                    
                    # Simulate some work
                    await asyncio.sleep(1.0)
                    
                    # Update metrics (simulation)
                    self.agents[agent_id].total_requests += 1
                    
                except asyncio.CancelledError:
                    logger.info(f"Agent {agent_id} received cancellation signal")
                    break
                except Exception as e:
                    logger.error(f"Error in agent {agent_id}: {e}")
                    self.agents[agent_id].failed_requests += 1
                    self.agents[agent_id].health_status = HealthStatus.WARNING
                    
                    # Sleep a bit before retrying
                    await asyncio.sleep(5.0)
                    
        except Exception as e:
            logger.error(f"Fatal error in agent {agent_id}: {e}")
            self.agent_states[agent_id] = AgentState.ERROR
            self.agents[agent_id].state = AgentState.ERROR
            self.agents[agent_id].health_status = HealthStatus.CRITICAL
        finally:
            logger.info(f"Agent {agent_id} task completed")
    
    async def _health_monitor_loop(self):
        """Background task for monitoring agent health."""
        while not self.shutdown_event.is_set():
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(5.0)
    
    async def _perform_health_checks(self):
        """Perform health checks on all agents."""
        current_time = datetime.now()
        stale_threshold = timedelta(seconds=60)  # 1 minute
        
        for agent_id, metrics in self.agents.items():
            # Check if agent is stale
            if current_time - metrics.last_heartbeat > stale_threshold:
                if metrics.health_status != HealthStatus.CRITICAL:
                    logger.warning(f"Agent {agent_id} appears stale")
                    metrics.health_status = HealthStatus.CRITICAL
                    self.critical_alerts.add(f"Agent {agent_id} is not responding")
            
            # Check error rate
            if metrics.total_requests > 10:
                error_rate = metrics.failed_requests / metrics.total_requests
                if error_rate > 0.1:  # 10% error rate
                    if metrics.health_status == HealthStatus.HEALTHY:
                        logger.warning(f"Agent {agent_id} has high error rate: {error_rate:.2%}")
                        metrics.health_status = HealthStatus.WARNING
    
    def _get_pool_metrics(self) -> PoolMetrics:
        """Calculate current pool metrics."""
        total_agents = len(self.agents)
        running_agents = sum(1 for state in self.agent_states.values() if state == AgentState.RUNNING)
        healthy_agents = sum(1 for metrics in self.agents.values() if metrics.health_status == HealthStatus.HEALTHY)
        total_requests = sum(metrics.total_requests for metrics in self.agents.values())
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return PoolMetrics(
            pool_id=self.pool_id,
            total_agents=total_agents,
            running_agents=running_agents,
            healthy_agents=healthy_agents,
            total_pool_requests=total_requests,
            pool_uptime_seconds=uptime,
            memory_coordinator_status="active" if self.a2a_coordinator else "inactive",
            a2a_connections=1 if self.a2a_coordinator else 0
        )
    
    def record_request(self, request_id: str, method: str, duration: float, success: bool):
        """Record a request for performance tracking."""
        request_record = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "method": method,
            "duration_ms": duration * 1000,
            "success": success
        }
        
        self.request_history.append(request_record)
        
        # Trim history if too large
        if len(self.request_history) > self.max_history_size:
            self.request_history = self.request_history[-self.max_history_size:]
    
    async def shutdown(self):
        """Gracefully shutdown the lifecycle manager."""
        logger.info("Shutting down enhanced MCP lifecycle manager...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Stop health monitoring
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        # Stop all agents
        for agent_id in list(self.agent_tasks.keys()):
            await self.stop_agent(agent_id, graceful=True)
        
        # Shutdown A2A coordinator
        if self.a2a_coordinator:
            try:
                from .a2a_memory_coordinator import shutdown_pool_coordinator
                await shutdown_pool_coordinator()
            except Exception as e:
                logger.error(f"Error shutting down A2A coordinator: {e}")
        
        logger.info("Enhanced MCP lifecycle manager shutdown completed")


def create_enhanced_mcp_server(pool_id: str = "alpha_pool") -> tuple[FastMCP, EnhancedMCPLifecycleManager]:
    """
    Create an enhanced MCP server with lifecycle management.
    
    Args:
        pool_id: Identifier for the agent pool
    
    Returns:
        Tuple of (FastMCP server, EnhancedMCPLifecycleManager)
    """
    # Create the MCP server
    mcp = FastMCP("AlphaAgentPool Enhanced")
    
    # Create the lifecycle manager
    lifecycle_manager = EnhancedMCPLifecycleManager(mcp, pool_id)
    
    # Add request monitoring middleware (simplified due to FastMCP API changes)
    try:
        # Try to access the internal MCP server if available
        if hasattr(mcp, '_mcp_server') and hasattr(mcp._mcp_server, '_handle_request'):
            original_handle_request = mcp._mcp_server._handle_request
            
            async def enhanced_handle_request(request: JSONRPCMessage) -> None:
                start_time = time.time()
                request_id = str(request.id) if hasattr(request, 'id') else "unknown"
                method = getattr(request, 'method', 'unknown')
                
                try:
                    result = await original_handle_request(request)
                    duration = time.time() - start_time
                    lifecycle_manager.record_request(request_id, method, duration, True)
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    lifecycle_manager.record_request(request_id, method, duration, False, str(e))
                    raise
            
            mcp._mcp_server._handle_request = enhanced_handle_request
            logger.info("Request monitoring middleware installed successfully")
        else:
            # Fallback: just log that monitoring is not available
            logger.warning("Request monitoring unavailable due to FastMCP API changes")
    except Exception as e:
        logger.warning(f"Failed to setup request monitoring: {e}")
    
    return mcp, lifecycle_manager
