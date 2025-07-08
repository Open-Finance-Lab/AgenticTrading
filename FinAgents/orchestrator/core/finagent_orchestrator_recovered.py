"""
FinAgent Orchestrator Core - Main Orchestration Engine

This module implements the central orchestration engine that coordinates all agent pools,
manages task execution, handles memory integration, and provides RL-enhanced backtesting.

Key Features:
- Multi-agent pool coordination
- DAG-based task execution
- Memory-enhanced decision making  
- Reinforcement learning integration
- Comprehensive backtesting framework
- Real-time monitoring and management

Author: FinAgent Team
Version: 1.0.0
"""

import asyncio
import concurrent.futures
import logging
import sys
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.client.sse import sse_client
from mcp import ClientSession

from .dag_planner import DAGPlanner, TaskNode, TaskStatus, AgentPoolType, TradingStrategy, BacktestConfiguration

# Add memory module to path
memory_path = Path(__file__).parent.parent.parent / "memory"
sys.path.insert(0, str(memory_path))

try:
    from ...memory.external_memory_agent import ExternalMemoryAgent, EventType, LogLevel
    MEMORY_AVAILABLE = True
except ImportError:
    # Graceful fallback if memory agent is not available
    ExternalMemoryAgent = None
    EventType = None
    LogLevel = None
    MEMORY_AVAILABLE = False

# Configure logging
logger = logging.getLogger("FinAgentOrchestrator")


class OrchestratorStatus(Enum):
    """Orchestrator operational status"""
    INITIALIZING = "initializing"
    READY = "ready"
    EXECUTING = "executing"
    BACKTESTING = "backtesting"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class AgentPoolConnection:
    """Configuration for connecting to agent pools"""
    pool_type: AgentPoolType
    endpoint: str
    port: int
    health_check_interval: int = 30
    timeout: int = 30
    max_retries: int = 3
    is_connected: bool = False
    last_health_check: Optional[datetime] = None


@dataclass
class ExecutionContext:
    """Context for strategy execution"""
    execution_id: str
    strategy: TradingStrategy
    start_time: datetime
    status: str = "running"
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Results from backtesting execution"""
    backtest_id: str
    strategy: TradingStrategy
    configuration: BacktestConfiguration
    start_time: datetime
    end_time: datetime
    performance_metrics: Dict[str, float]
    trade_history: List[Dict[str, Any]]
    portfolio_values: List[Dict[str, Any]]
    risk_metrics: Dict[str, float]
    memory_insights: Dict[str, Any]
    rl_performance: Optional[Dict[str, Any]] = None


class FinAgentOrchestrator:
    """
    Main orchestration engine for the FinAgent ecosystem.
    
    This class coordinates all agent pools, manages execution flows, and provides
    comprehensive backtesting and RL capabilities with memory integration.
    """

    def __init__(self, 
                 host: str = "0.0.0.0", 
                 port: int = 9000,
                 enable_rl: bool = True,
                 enable_memory: bool = True,
                 enable_monitoring: bool = True):
        """
        Initialize the FinAgent Orchestrator.
        
        Args:
            host: Host address for the orchestrator MCP server
            port: Port for the orchestrator MCP server
            enable_rl: Whether to enable RL capabilities
            enable_memory: Whether to enable memory integration
            enable_monitoring: Whether to enable agent pool monitoring
        """
        self.host = host
        self.port = port
        self.enable_rl = enable_rl
        self.enable_memory = enable_memory
        self.enable_monitoring = enable_monitoring
        self.status = OrchestratorStatus.INITIALIZING
        
        # Core components
        self.dag_planner = DAGPlanner()
        self.mcp_server = FastMCP("FinAgentOrchestrator")
        
        # Agent pool connections
        self.agent_pools = {
            AgentPoolType.DATA_AGENT_POOL: AgentPoolConnection(
                pool_type=AgentPoolType.DATA_AGENT_POOL,
                endpoint="http://localhost:8001/sse",
                port=8001
            ),
            AgentPoolType.ALPHA_AGENT_POOL: AgentPoolConnection(
                pool_type=AgentPoolType.ALPHA_AGENT_POOL,
                endpoint="http://localhost:5050/sse",
                port=5050
            ),
            AgentPoolType.EXECUTION_AGENT_POOL: AgentPoolConnection(
                pool_type=AgentPoolType.EXECUTION_AGENT_POOL,
                endpoint="http://localhost:8004/sse",
                port=8004
            ),
            AgentPoolType.MEMORY_AGENT: AgentPoolConnection(
                pool_type=AgentPoolType.MEMORY_AGENT,
                endpoint="http://localhost:8000/sse",
                port=8000
            )
        }
        
        # Execution management
        self.active_executions: Dict[str, ExecutionContext] = {}
        self.completed_executions: Dict[str, ExecutionContext] = {}
        self.task_executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        
        # Backtesting and RL
        self.backtest_results: Dict[str, BacktestResult] = {}
        self.rl_policies: Dict[str, Any] = {}
        self.sandbox_environments: Dict[str, Any] = {}
        self.current_session_id = None
        
        # Performance monitoring
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "avg_execution_time": 0.0,
            "active_agents": 0,
            "memory_events_logged": 0
        }
        
        # Initialize memory agent
        self.memory_agent = None
        if self.enable_memory and MEMORY_AVAILABLE:
            self.memory_agent = ExternalMemoryAgent(
                enable_real_time_hooks=True
            )
            logger.info("Memory agent initialized")
        elif self.enable_memory and not MEMORY_AVAILABLE:
            logger.warning("Memory integration requested but not available")
        
        # Register orchestrator tools
        self._register_orchestrator_tools()
        
        logger.info("FinAgent Orchestrator initialized")

    async def initialize(self):
        """Initialize the orchestrator and all components"""
        try:
            self.status = OrchestratorStatus.INITIALIZING
            
            # Initialize memory agent if enabled
            if self.memory_agent:
                await self._ensure_memory_agent_initialized()
            
            # Initialize DAG planner
            await self.dag_planner.initialize()
            
            # Health check all agent pools
            if self.enable_monitoring:
                await self._health_check_all_pools()
            
            self.status = OrchestratorStatus.READY
            logger.info("FinAgent Orchestrator ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            self.status = OrchestratorStatus.ERROR
            raise

    async def _ensure_memory_agent_initialized(self):
        """Ensure memory agent is properly initialized"""
        if not self.memory_agent:
            return
            
        try:
            if not hasattr(self.memory_agent, '_initialized') or not self.memory_agent._initialized:
                await self.memory_agent.initialize()
                self.current_session_id = f"orchestrator_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                logger.info(f"✅ Memory agent initialized with session: {self.current_session_id}")
        except Exception as e:
            logger.error(f"Failed to initialize memory agent: {e}")
            self.memory_agent = None

    async def _log_memory_event(self,
                               event_type,
                               log_level, 
                               title: str,
                               content: str,
                               tags: set = None,
                               metadata: dict = None,
                               correlation_id: str = None):
        """Log an event to the memory system"""
        if not self.memory_agent or not MEMORY_AVAILABLE:
            return None
            
        try:
            # Ensure memory agent is initialized
            await self._ensure_memory_agent_initialized()
            if not self.memory_agent:
                return None
                
            return await self.memory_agent.log_event(
                event_type=event_type,
                log_level=log_level,
                source_agent_pool="orchestrator",
                source_agent_id="finagent_orchestrator",
                title=title,
                content=content,
                tags=tags or set(),
                metadata=metadata or {},
                session_id=self.current_session_id,
                correlation_id=correlation_id
            )
        except Exception as e:
            logger.error(f"Failed to log memory event: {e}")
            return None

    async def _health_check_all_pools(self):
        """Perform health checks on all agent pools"""
        for pool_type, connection in self.agent_pools.items():
            try:
                # Simple connection test
                async with sse_client(connection.endpoint, timeout=5) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        connection.is_connected = True
                        connection.last_health_check = datetime.now()
                        logger.info(f"✅ {pool_type.value} pool healthy")
            except Exception as e:
                connection.is_connected = False
                logger.warning(f"❌ {pool_type.value} pool unhealthy: {e}")

    def _register_orchestrator_tools(self):
        """Register MCP tools for the orchestrator"""
        
        @self.mcp_server.tool(name="execute_strategy", description="Execute a trading strategy")
        async def execute_strategy(strategy_config: dict) -> dict:
            """Execute a trading strategy using DAG planning"""
            try:
                strategy = TradingStrategy(**strategy_config)
                execution_id = str(uuid.uuid4())
                
                # Create execution context
                context = ExecutionContext(
                    execution_id=execution_id,
                    strategy=strategy,
                    start_time=datetime.now()
                )
                
                self.active_executions[execution_id] = context
                
                # Plan DAG execution
                dag_plan = await self.dag_planner.create_dag_plan(strategy)
                
                # Execute DAG
                results = await self.dag_planner.execute_dag(dag_plan, self.agent_pools)
                
                # Update context
                context.status = "completed"
                context.results = results
                self.completed_executions[execution_id] = context
                del self.active_executions[execution_id]
                
                self.metrics["total_executions"] += 1
                self.metrics["successful_executions"] += 1
                
                # Log to memory
                await self._log_memory_event(
                    event_type=EventType.OPTIMIZATION,
                    log_level=LogLevel.INFO,
                    title=f"Strategy executed: {strategy.name}",
                    content=f"Successfully executed strategy '{strategy.name}' with {len(strategy.symbols)} symbols",
                    tags={"strategy", "execution", "success"},
                    metadata={
                        "strategy_id": strategy.strategy_id,
                        "execution_id": execution_id,
                        "symbols": strategy.symbols,
                        "strategy_type": strategy.strategy_type
                    },
                    correlation_id=execution_id
                )
                
                return {
                    "status": "success",
                    "execution_id": execution_id,
                    "results": results
                }
                
            except Exception as e:
                logger.error(f"Strategy execution failed: {e}")
                self.metrics["failed_executions"] += 1
                return {
                    "status": "error",
                    "error": str(e)
                }

        @self.mcp_server.tool(name="run_backtest", description="Run comprehensive backtest")
        async def run_backtest(config: dict) -> dict:
            """Run a comprehensive backtest with memory and RL integration"""
            try:
                backtest_config = BacktestConfiguration(**config)
                backtest_id = str(uuid.uuid4())
                
                self.status = OrchestratorStatus.BACKTESTING
                
                # Log backtest start
                await self._log_memory_event(
                    event_type=EventType.SYSTEM,
                    log_level=LogLevel.INFO,
                    title=f"Backtest started: {backtest_config.config_id}",
                    content=f"Starting backtest from {backtest_config.start_date} to {backtest_config.end_date}",
                    tags={"backtest", "start"},
                    metadata={
                        "backtest_id": backtest_id,
                        "config_id": backtest_config.config_id,
                        "start_date": backtest_config.start_date.isoformat(),
                        "end_date": backtest_config.end_date.isoformat(),
                        "initial_capital": backtest_config.initial_capital
                    },
                    correlation_id=backtest_id
                )
                
                # Run backtest simulation
                result = await self._run_backtest_simulation(backtest_config, backtest_id)
                
                # Store results
                self.backtest_results[backtest_id] = result
                
                self.status = OrchestratorStatus.READY
                
                # Log backtest completion
                await self._log_memory_event(
                    event_type=EventType.OPTIMIZATION,
                    log_level=LogLevel.INFO,
                    title=f"Backtest completed: {backtest_config.config_id}",
                    content=f"Backtest completed with total return: {result.performance_metrics.get('total_return', 0):.2%}",
                    tags={"backtest", "completed", "performance"},
                    metadata={
                        "backtest_id": backtest_id,
                        "total_return": result.performance_metrics.get('total_return', 0),
                        "sharpe_ratio": result.performance_metrics.get('sharpe_ratio', 0),
                        "max_drawdown": result.performance_metrics.get('max_drawdown', 0),
                        "total_trades": len(result.trade_history)
                    },
                    correlation_id=backtest_id
                )
                
                return {
                    "status": "success",
                    "backtest_id": backtest_id,
                    "performance_metrics": result.performance_metrics,
                    "summary": {
                        "total_trades": len(result.trade_history),
                        "total_return": result.performance_metrics.get('total_return', 0),
                        "sharpe_ratio": result.performance_metrics.get('sharpe_ratio', 0)
                    }
                }
                
            except Exception as e:
                logger.error(f"Backtest failed: {e}")
                self.status = OrchestratorStatus.ERROR
                return {
                    "status": "error",
                    "error": str(e)
                }

        @self.mcp_server.tool(name="get_orchestrator_status", description="Get orchestrator status and metrics")
        async def get_orchestrator_status() -> dict:
            """Get current orchestrator status and performance metrics"""
            return {
                "status": self.status.value,
                "metrics": self.metrics,
                "active_executions": len(self.active_executions),
                "completed_executions": len(self.completed_executions),
                "agent_pool_status": {
                    pool_type.value: {
                        "connected": connection.is_connected,
                        "last_check": connection.last_health_check.isoformat() if connection.last_health_check else None
                    }
                    for pool_type, connection in self.agent_pools.items()
                },
                "memory_enabled": self.memory_agent is not None,
                "rl_enabled": self.enable_rl,
                "session_id": self.current_session_id
            }

        @self.mcp_server.tool(name="start_agent_pool", description="Start a specific agent pool")
        async def start_agent_pool(pool_type: str, port: int = None) -> dict:
            """Start a specific agent pool and verify its connectivity"""
            import subprocess
            import time
            import os
            
            try:
                # Map pool types to their modules and default ports
                pool_config = {
                    "data": {
                        "module": "FinAgents.agent_pools.data_agent_pool.core",
                        "port": port or 8080,
                        "health_path": "/health"
                    },
                    "alpha": {
                        "module": "FinAgents.agent_pools.alpha_agent_pool.core", 
                        "port": port or 8081,
                        "health_path": "/health"
                    },
                    "portfolio": {
                        "module": "FinAgents.agent_pools.portfolio_construction_agent_pool.core",
                        "port": port or 8083,
                        "health_path": "/health"
                    },
                    "risk": {
                        "module": "FinAgents.agent_pools.risk_agent_pool.core",
                        "port": port or 8084,
                        "health_path": "/health"
                    },
                    "transaction_cost": {
                        "module": "FinAgents.agent_pools.transaction_cost_agent_pool.core",
                        "port": port or 8085,
                        "health_path": "/health"
                    },
                    "memory": {
                        "module": "FinAgents.memory.memory_server",
                        "port": port or 8086,
                        "health_path": "/health"
                    }
                }
                
                if pool_type not in pool_config:
                    return {
                        "status": "error",
                        "error": f"Unknown pool type: {pool_type}",
                        "available_types": list(pool_config.keys())
                    }
                
                config = pool_config[pool_type]
                
                # Check if already running
                try:
                    import httpx
                    async with httpx.AsyncClient(timeout=5) as client:
                        response = await client.get(f"http://localhost:{config['port']}{config['health_path']}")
                        if response.status_code == 200:
                            return {
                                "status": "already_running",
                                "pool_type": pool_type,
                                "port": config['port'],
                                "message": f"{pool_type} agent pool is already running on port {config['port']}"
                            }
                except:
                    pass  # Not running, proceed to start
                
                # Start the agent pool
                workspace_dir = "/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration"
                log_file = f"{workspace_dir}/logs/{pool_type}_agent_pool.log"
                pid_file = f"{workspace_dir}/logs/{pool_type}_agent_pool.pid"
                
                # Ensure log directory exists
                os.makedirs(f"{workspace_dir}/logs", exist_ok=True)
                
                # Start the process
                cmd = f"cd {workspace_dir} && python -m {config['module']} > {log_file} 2>&1 & echo $! > {pid_file}"
                
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    cwd=workspace_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Give it time to start
                await asyncio.sleep(3)
                
                # Verify it started successfully
                try:
                    import httpx
                    async with httpx.AsyncClient(timeout=10) as client:
                        for attempt in range(5):
                            try:
                                response = await client.get(f"http://localhost:{config['port']}{config['health_path']}")
                                if response.status_code == 200:
                                    # Update orchestrator's agent pool connection
                                    pool_enum_map = {
                                        "data": AgentPoolType.DATA_AGENT_POOL,
                                        "alpha": AgentPoolType.ALPHA_AGENT_POOL,
                                        "portfolio": AgentPoolType.EXECUTION_AGENT_POOL,
                                        "risk": AgentPoolType.EXECUTION_AGENT_POOL,
                                        "transaction_cost": AgentPoolType.EXECUTION_AGENT_POOL,
                                        "memory": AgentPoolType.MEMORY_AGENT
                                    }
                                    
                                    if pool_type in pool_enum_map:
                                        pool_enum = pool_enum_map[pool_type]
                                        if pool_enum in self.agent_pools:
                                            self.agent_pools[pool_enum].is_connected = True
                                            self.agent_pools[pool_enum].last_health_check = datetime.now()
                                    
                                    return {
                                        "status": "success",
                                        "pool_type": pool_type,
                                        "port": config['port'],
                                        "message": f"{pool_type} agent pool started successfully",
                                        "health_check": "passed"
                                    }
                            except Exception as e:
                                if attempt < 4:
                                    await asyncio.sleep(2)
                                    continue
                                else:
                                    break
                    
                    # If we get here, health check failed
                    return {
                        "status": "started_but_unhealthy",
                        "pool_type": pool_type,
                        "port": config['port'],
                        "message": f"{pool_type} agent pool started but health check failed",
                        "log_file": log_file
                    }
                    
                except Exception as e:
                    return {
                        "status": "start_failed",
                        "pool_type": pool_type,
                        "error": str(e),
                        "log_file": log_file
                    }
                
            except Exception as e:
                logger.error(f"Failed to start {pool_type} agent pool: {e}")
                return {
                    "status": "error",
                    "pool_type": pool_type,
                    "error": str(e)
                }

        @self.mcp_server.tool(name="stop_agent_pool", description="Stop a specific agent pool")
        async def stop_agent_pool(pool_type: str) -> dict:
            """Stop a specific agent pool"""
            import subprocess
            import os
            
            try:
                workspace_dir = "/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration"
                pid_file = f"{workspace_dir}/logs/{pool_type}_agent_pool.pid"
                
                # Try to read PID file
                if os.path.exists(pid_file):
                    try:
                        with open(pid_file, 'r') as f:
                            pid = int(f.read().strip())
                        
                        # Kill the process
                        subprocess.run(f"kill {pid}", shell=True, check=False)
                        os.remove(pid_file)
                        
                        # Update orchestrator's agent pool connection
                        pool_enum_map = {
                            "data": AgentPoolType.DATA_AGENT_POOL,
                            "alpha": AgentPoolType.ALPHA_AGENT_POOL,
                            "portfolio": AgentPoolType.EXECUTION_AGENT_POOL,
                            "risk": AgentPoolType.EXECUTION_AGENT_POOL,
                            "transaction_cost": AgentPoolType.EXECUTION_AGENT_POOL,
                            "memory": AgentPoolType.MEMORY_AGENT
                        }
                        
                        if pool_type in pool_enum_map:
                            pool_enum = pool_enum_map[pool_type]
                            if pool_enum in self.agent_pools:
                                self.agent_pools[pool_enum].is_connected = False
                        
                        return {
                            "status": "success",
                            "pool_type": pool_type,
                            "message": f"{pool_type} agent pool stopped"
                        }
                    except Exception as e:
                        return {
                            "status": "error",
                            "pool_type": pool_type,
                            "error": f"Failed to stop process: {e}"
                        }
                else:
                    # Try to kill by process name
                    result = subprocess.run(
                        f"pkill -f {pool_type}_agent_pool", 
                        shell=True, 
                        capture_output=True, 
                        text=True
                    )
                    
                    return {
                        "status": "attempted",
                        "pool_type": pool_type,
                        "message": f"Attempted to stop {pool_type} agent pool (no PID file found)"
                    }
                    
            except Exception as e:
                logger.error(f"Failed to stop {pool_type} agent pool: {e}")
                return {
                    "status": "error",
                    "pool_type": pool_type,
                    "error": str(e)
                }

        @self.mcp_server.tool(name="check_agent_pool_health", description="Check health of all or specific agent pools")
        async def check_agent_pool_health(pool_type: str = None) -> dict:
            """Check health status of agent pools"""
            import httpx
            
            # Define all pools and their ports
            pools_to_check = {
                "data": 8080,
                "alpha": 8081, 
                "portfolio": 8083,
                "risk": 8084,
                "transaction_cost": 8085,
                "memory": 8086
            }
            
            if pool_type:
                if pool_type not in pools_to_check:
                    return {
                        "status": "error",
                        "error": f"Unknown pool type: {pool_type}",
                        "available_types": list(pools_to_check.keys())
                    }
                pools_to_check = {pool_type: pools_to_check[pool_type]}
            
            results = {}
            
            async with httpx.AsyncClient(timeout=5) as client:
                for pool_name, port in pools_to_check.items():
                    try:
                        response = await client.get(f"http://localhost:{port}/health")
                        results[pool_name] = {
                            "status": "healthy" if response.status_code == 200 else "unhealthy",
                            "port": port,
                            "response_code": response.status_code,
                            "response_time_ms": response.elapsed.total_seconds() * 1000
                        }
                        
                        # Update orchestrator's tracking
                        pool_enum_map = {
                            "data": AgentPoolType.DATA_AGENT_POOL,
                            "alpha": AgentPoolType.ALPHA_AGENT_POOL,
                            "portfolio": AgentPoolType.EXECUTION_AGENT_POOL,
                            "risk": AgentPoolType.EXECUTION_AGENT_POOL,
                            "transaction_cost": AgentPoolType.EXECUTION_AGENT_POOL,
                            "memory": AgentPoolType.MEMORY_AGENT
                        }
                        
                        if pool_name in pool_enum_map:
                            pool_enum = pool_enum_map[pool_name]
                            if pool_enum in self.agent_pools:
                                self.agent_pools[pool_enum].is_connected = (response.status_code == 200)
                                self.agent_pools[pool_enum].last_health_check = datetime.now()
                            
                    except Exception as e:
                        results[pool_name] = {
                            "status": "unreachable",
                            "port": port,
                            "error": str(e)
                        }
                        
                        # Update orchestrator's tracking
                        pool_enum_map = {
                            "data": AgentPoolType.DATA_AGENT_POOL,
                            "alpha": AgentPoolType.ALPHA_AGENT_POOL,
                            "portfolio": AgentPoolType.EXECUTION_AGENT_POOL,
                            "risk": AgentPoolType.EXECUTION_AGENT_POOL,
                            "transaction_cost": AgentPoolType.EXECUTION_AGENT_POOL,
                            "memory": AgentPoolType.MEMORY_AGENT
                        }
                        
                        if pool_name in pool_enum_map:
                            pool_enum = pool_enum_map[pool_name]
                            if pool_enum in self.agent_pools:
                                self.agent_pools[pool_enum].is_connected = False
            
            # Summary
            healthy_count = len([r for r in results.values() if r["status"] == "healthy"])
            total_count = len(results)
            
            return {
                "summary": {
                    "healthy": healthy_count,
                    "total": total_count,
                    "health_percentage": (healthy_count / total_count) * 100 if total_count > 0 else 0
                },
                "details": results
            }

        @self.mcp_server.tool(name="diagnose_agent_pool", description="Diagnose issues with a specific agent pool")
        async def diagnose_agent_pool(pool_type: str) -> dict:
            """Diagnose issues with a specific agent pool"""
            import subprocess
            import os
            
            workspace_dir = "/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration"
            log_file = f"{workspace_dir}/logs/{pool_type}_agent_pool.log"
            pid_file = f"{workspace_dir}/logs/{pool_type}_agent_pool.pid"
            
            diagnosis = {
                "pool_type": pool_type,
                "timestamp": datetime.now().isoformat(),
                "checks": {}
            }
            
            # Check if PID file exists
            diagnosis["checks"]["pid_file_exists"] = os.path.exists(pid_file)
            
            # Check if process is running
            if os.path.exists(pid_file):
                try:
                    with open(pid_file, 'r') as f:
                        pid = int(f.read().strip())
                    
                    # Check if process exists
                    result = subprocess.run(f"ps -p {pid}", shell=True, capture_output=True)
                    diagnosis["checks"]["process_running"] = result.returncode == 0
                    diagnosis["checks"]["pid"] = pid
                except:
                    diagnosis["checks"]["process_running"] = False
                    diagnosis["checks"]["pid_file_readable"] = False
            
            # Check log file
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                    
                    diagnosis["checks"]["log_file_exists"] = True
                    diagnosis["checks"]["log_size_bytes"] = len(log_content)
                    
                    # Look for common error patterns
                    error_patterns = {
                        "import_error": "ImportError" in log_content or "ModuleNotFoundError" in log_content,
                        "port_conflict": "Address already in use" in log_content or "port" in log_content.lower(),
                        "async_error": "coroutine" in log_content and ("not awaited" in log_content or "running" in log_content),
                        "memory_error": "MemoryError" in log_content or "out of memory" in log_content.lower(),
                        "connection_error": "ConnectionError" in log_content or "Connection refused" in log_content,
                        "syntax_error": "SyntaxError" in log_content,
                        "attribute_error": "AttributeError" in log_content
                    }
                    
                    diagnosis["checks"]["error_patterns"] = {k: v for k, v in error_patterns.items() if v}
                    
                    # Get last few lines of log
                    log_lines = log_content.strip().split('\n')
                    diagnosis["checks"]["last_log_lines"] = log_lines[-10:] if log_lines else []
                    
                except Exception as e:
                    diagnosis["checks"]["log_file_readable"] = False
                    diagnosis["checks"]["log_read_error"] = str(e)
            else:
                diagnosis["checks"]["log_file_exists"] = False
            
            # Check port availability
            pool_ports = {
                "data": 8080,
                "alpha": 8081,
                "portfolio": 8083, 
                "risk": 8084,
                "transaction_cost": 8085,
                "memory": 8086
            }
            
            if pool_type in pool_ports:
                port = pool_ports[pool_type]
                try:
                    import httpx
                    async with httpx.AsyncClient(timeout=3) as client:
                        response = await client.get(f"http://localhost:{port}/health")
                        diagnosis["checks"]["health_endpoint"] = {
                            "reachable": True,
                            "status_code": response.status_code,
                            "healthy": response.status_code == 200
                        }
                except Exception as e:
                    diagnosis["checks"]["health_endpoint"] = {
                        "reachable": False,
                        "error": str(e)
                    }
                
                # Check if port is in use by another process
                result = subprocess.run(f"lsof -i :{port}", shell=True, capture_output=True, text=True)
                diagnosis["checks"]["port_in_use"] = result.returncode == 0
                if result.returncode == 0:
                    diagnosis["checks"]["port_usage"] = result.stdout.strip().split('\n')
            
            # Generate recommendations
            recommendations = []
            
            if not diagnosis["checks"].get("process_running", False):
                recommendations.append("Process is not running - try starting the agent pool")
            
            if diagnosis["checks"].get("error_patterns", {}).get("import_error"):
                recommendations.append("Import errors detected - check Python path and dependencies")
            
            if diagnosis["checks"].get("error_patterns", {}).get("port_conflict"):
                recommendations.append("Port conflict detected - check if another process is using the port")
                
            if diagnosis["checks"].get("error_patterns", {}).get("async_error"):
                recommendations.append("Async/await errors detected - review coroutine usage")
            
            if not diagnosis["checks"].get("health_endpoint", {}).get("healthy", False):
                recommendations.append("Health endpoint not responding - check server startup")
            
            diagnosis["recommendations"] = recommendations
            
            return diagnosis

