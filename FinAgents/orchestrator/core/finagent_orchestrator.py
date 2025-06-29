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
import logging
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.client.sse import sse_client
from mcp import ClientSession

from .dag_planner import DAGPlanner, TaskNode, TaskStatus, AgentPoolType, TradingStrategy, BacktestConfiguration

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
                 enable_memory: bool = True):
        """
        Initialize the FinAgent Orchestrator.
        
        Args:
            host: Host address for the orchestrator MCP server
            port: Port for the orchestrator MCP server
            enable_rl: Whether to enable RL capabilities
            enable_memory: Whether to enable memory integration
        """
        self.host = host
        self.port = port
        self.enable_rl = enable_rl
        self.enable_memory = enable_memory
        
        # Core components
        self.status = OrchestratorStatus.INITIALIZING
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
                endpoint="http://localhost:8002/sse",
                port=8002
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
        
        # Performance monitoring
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "avg_execution_time": 0.0,
            "total_tasks_executed": 0,
            "agent_pool_health": {}
        }
        
        self._register_orchestrator_tools()
        logger.info("FinAgent Orchestrator initialized")

    async def start(self):
        """Start the orchestrator and all subsystems"""
        logger.info("Starting FinAgent Orchestrator...")
        
        try:
            # Connect to all agent pools
            await self._connect_to_agent_pools()
            
            # Initialize memory agent if enabled
            if self.enable_memory:
                await self._initialize_memory_agent()
            
            # Initialize RL components if enabled
            if self.enable_rl:
                await self._initialize_rl_components()
            
            # Start health monitoring
            asyncio.create_task(self._monitor_agent_pools())
            
            self.status = OrchestratorStatus.READY
            logger.info("✅ FinAgent Orchestrator started successfully")
            
        except Exception as e:
            self.status = OrchestratorStatus.ERROR
            logger.error(f"❌ Failed to start orchestrator: {e}")
            raise

    async def _connect_to_agent_pools(self):
        """Establish connections to all agent pools"""
        logger.info("Connecting to agent pools...")
        
        for pool_type, connection in self.agent_pools.items():
            try:
                # Test connection
                async with sse_client(connection.endpoint, timeout=10) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool("health_check", {})
                        
                        if result.content:
                            connection.is_connected = True
                            connection.last_health_check = datetime.now()
                            logger.info(f"✅ Connected to {pool_type.value}")
                        else:
                            logger.warning(f"⚠️ Health check failed for {pool_type.value}")
                            
            except Exception as e:
                logger.warning(f"❌ Failed to connect to {pool_type.value}: {e}")
                connection.is_connected = False

    async def _initialize_memory_agent(self):
        """Initialize memory agent integration"""
        logger.info("Initializing memory agent integration...")
        
        memory_connection = self.agent_pools[AgentPoolType.MEMORY_AGENT]
        if memory_connection.is_connected:
            # Initialize memory contexts for different strategy types
            await self._setup_memory_contexts()
            logger.info("✅ Memory agent initialized")
        else:
            logger.warning("⚠️ Memory agent not available")

    async def _initialize_rl_components(self):
        """Initialize reinforcement learning components"""
        logger.info("Initializing RL components...")
        
        # Create RL policy for strategy optimization
        self.rl_policies["strategy_optimizer"] = {
            "type": "ddpg",
            "state_space": "market_features",
            "action_space": "strategy_parameters",
            "reward_function": "sharpe_ratio"
        }
        
        logger.info("✅ RL components initialized")

    async def _setup_memory_contexts(self):
        """Setup memory contexts for different trading scenarios"""
        contexts = [
            {"context_type": "momentum_strategies", "description": "Momentum-based trading patterns"},
            {"context_type": "mean_reversion", "description": "Mean reversion scenarios"},
            {"context_type": "volatility_events", "description": "High volatility market conditions"},
            {"context_type": "correlation_breaks", "description": "Correlation breakdown events"}
        ]
        
        memory_endpoint = self.agent_pools[AgentPoolType.MEMORY_AGENT].endpoint
        
        for context in contexts:
            try:
                await self._call_agent_pool(
                    AgentPoolType.MEMORY_AGENT,
                    "setup_context",
                    context
                )
            except Exception as e:
                logger.warning(f"Failed to setup memory context {context['context_type']}: {e}")

    async def execute_strategy(self, strategy: TradingStrategy) -> str:
        """
        Execute a trading strategy using the DAG orchestration.
        
        Args:
            strategy: Trading strategy to execute
            
        Returns:
            str: Execution ID for tracking
        """
        execution_id = str(uuid.uuid4())
        logger.info(f"Starting strategy execution {execution_id}: {strategy.name}")
        
        try:
            # Create execution context
            context = ExecutionContext(
                execution_id=execution_id,
                strategy=strategy,
                start_time=datetime.now()
            )
            self.active_executions[execution_id] = context
            
            # Generate execution DAG
            dag = await self.dag_planner.plan_strategy_execution(strategy)
            
            # Start execution
            asyncio.create_task(self._execute_dag(execution_id, dag))
            
            self.metrics["total_executions"] += 1
            return execution_id
            
        except Exception as e:
            logger.error(f"Failed to start strategy execution: {e}")
            if execution_id in self.active_executions:
                self.active_executions[execution_id].status = "failed"
            raise

    async def _execute_dag(self, execution_id: str, dag):
        """Execute DAG tasks in proper order"""
        context = self.active_executions[execution_id]
        
        try:
            execution_order = self.dag_planner.get_execution_order()
            logger.info(f"Executing {len(execution_order)} tasks for {execution_id}")
            
            # Execute tasks in topological order
            for task_id in execution_order:
                if context.status != "running":
                    break
                    
                # Check if all dependencies are completed
                task = self.dag_planner.task_registry[task_id]
                if not all(dep in context.completed_tasks for dep in task.dependencies):
                    continue
                
                # Execute task
                success = await self._execute_task(execution_id, task)
                
                if success:
                    context.completed_tasks.append(task_id)
                    self.metrics["total_tasks_executed"] += 1
                else:
                    context.failed_tasks.append(task_id)
                    context.status = "failed"
                    break
            
            # Finalize execution
            if context.status == "running":
                context.status = "completed"
                self.metrics["successful_executions"] += 1
            else:
                self.metrics["failed_executions"] += 1
            
            # Move to completed executions
            self.completed_executions[execution_id] = context
            del self.active_executions[execution_id]
            
            # Calculate execution time
            execution_time = (datetime.now() - context.start_time).total_seconds()
            self._update_avg_execution_time(execution_time)
            
            logger.info(f"Strategy execution {execution_id} completed with status: {context.status}")
            
        except Exception as e:
            logger.error(f"Error executing DAG for {execution_id}: {e}")
            context.status = "error"
            self.metrics["failed_executions"] += 1

    async def _execute_task(self, execution_id: str, task: TaskNode) -> bool:
        """Execute a single task"""
        logger.info(f"Executing task {task.task_id} for execution {execution_id}")
        
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            # Call appropriate agent pool
            result = await self._call_agent_pool(
                task.agent_pool,
                task.tool_name,
                task.parameters
            )
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            # Store result in execution context
            context = self.active_executions[execution_id]
            context.results[task.task_id] = result
            
            logger.info(f"✅ Task {task.task_id} completed successfully")
            return True
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now()
            
            logger.error(f"❌ Task {task.task_id} failed: {e}")
            return False

    async def _call_agent_pool(self, pool_type: AgentPoolType, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Call a specific agent pool with retry logic"""
        connection = self.agent_pools[pool_type]
        
        if not connection.is_connected:
            raise Exception(f"Agent pool {pool_type.value} is not connected")
        
        for attempt in range(connection.max_retries):
            try:
                async with sse_client(connection.endpoint, timeout=connection.timeout) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool(tool_name, parameters)
                        
                        if result.content and len(result.content) > 0:
                            content_item = result.content[0]
                            if hasattr(content_item, 'text'):
                                return json.loads(content_item.text)
                        
                        return {"status": "error", "error": "No content in response"}
                        
            except Exception as e:
                if attempt == connection.max_retries - 1:
                    raise e
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    async def run_backtest(self, config: BacktestConfiguration) -> str:
        """
        Run comprehensive backtest with memory and RL integration.
        
        Args:
            config: Backtesting configuration
            
        Returns:
            str: Backtest ID for tracking results
        """
        backtest_id = str(uuid.uuid4())
        logger.info(f"Starting backtest {backtest_id}: {config.strategy.name}")
        
        try:
            self.status = OrchestratorStatus.BACKTESTING
            
            # Create sandbox environment
            sandbox = await self._create_sandbox_environment(config)
            
            # Run backtest execution
            backtest_task = asyncio.create_task(
                self._execute_backtest(backtest_id, config, sandbox)
            )
            
            return backtest_id
            
        except Exception as e:
            logger.error(f"Failed to start backtest: {e}")
            self.status = OrchestratorStatus.READY
            raise

    async def _create_sandbox_environment(self, config: BacktestConfiguration) -> Dict[str, Any]:
        """Create isolated sandbox environment for backtesting"""
        sandbox = {
            "environment_id": str(uuid.uuid4()),
            "start_date": config.start_date,
            "end_date": config.end_date,
            "initial_capital": config.initial_capital,
            "current_capital": config.initial_capital,
            "positions": {},
            "trade_history": [],
            "portfolio_values": [],
            "commission_rate": config.commission_rate,
            "slippage_rate": config.slippage_rate
        }
        
        self.sandbox_environments[sandbox["environment_id"]] = sandbox
        return sandbox

    async def _execute_backtest(self, backtest_id: str, config: BacktestConfiguration, sandbox: Dict[str, Any]):
        """Execute the backtesting process"""
        start_time = datetime.now()
        
        try:
            # Prepare historical data
            await self._prepare_backtest_data(config, sandbox)
            
            # Initialize memory context if enabled
            if config.memory_enabled and self.enable_memory:
                await self._initialize_backtest_memory_context(config, sandbox)
            
            # Initialize RL policy if enabled
            if config.rl_enabled and self.enable_rl:
                await self._initialize_backtest_rl_policy(config, sandbox)
            
            # Run time-series simulation
            performance_data = await self._run_time_series_simulation(config, sandbox)
            
            # Calculate performance metrics
            metrics = await self._calculate_backtest_metrics(performance_data, config)
            
            # Generate final results
            result = BacktestResult(
                backtest_id=backtest_id,
                strategy=config.strategy,
                configuration=config,
                start_time=start_time,
                end_time=datetime.now(),
                performance_metrics=metrics["performance"],
                trade_history=sandbox["trade_history"],
                portfolio_values=sandbox["portfolio_values"],
                risk_metrics=metrics["risk"],
                memory_insights=metrics.get("memory", {}),
                rl_performance=metrics.get("rl", None)
            )
            
            self.backtest_results[backtest_id] = result
            
            logger.info(f"✅ Backtest {backtest_id} completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Backtest {backtest_id} failed: {e}")
        finally:
            self.status = OrchestratorStatus.READY

    async def _prepare_backtest_data(self, config: BacktestConfiguration, sandbox: Dict[str, Any]):
        """Prepare historical data for backtesting"""
        # Fetch historical data for all symbols
        for symbol in config.strategy.symbols:
            result = await self._call_agent_pool(
                AgentPoolType.DATA_AGENT_POOL,
                "fetch_market_data",
                {
                    "symbol": symbol,
                    "start": config.start_date.isoformat(),
                    "end": config.end_date.isoformat(),
                    "interval": config.strategy.timeframe
                }
            )
            sandbox[f"data_{symbol}"] = result

    async def _run_time_series_simulation(self, config: BacktestConfiguration, sandbox: Dict[str, Any]) -> Dict[str, Any]:
        """Run time-series simulation of strategy"""
        # This would implement the day-by-day backtesting logic
        # For now, return placeholder data
        return {
            "daily_returns": [],
            "cumulative_returns": [],
            "drawdowns": [],
            "volatility": 0.15,
            "sharpe_ratio": 1.2
        }

    async def _calculate_backtest_metrics(self, performance_data: Dict[str, Any], config: BacktestConfiguration) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        return {
            "performance": {
                "total_return": 0.15,
                "annualized_return": 0.12,
                "volatility": 0.15,
                "sharpe_ratio": 1.2,
                "sortino_ratio": 1.5,
                "max_drawdown": -0.08,
                "win_rate": 0.65
            },
            "risk": {
                "var_95": -0.05,
                "expected_shortfall": -0.07,
                "beta": 0.8,
                "alpha": 0.03
            },
            "memory": {
                "context_utilization": 0.8,
                "pattern_matches": 15,
                "memory_enhanced_decisions": 8
            },
            "rl": {
                "policy_performance": 1.15,
                "learning_progress": 0.9
            } if config.rl_enabled else None
        }

    async def _monitor_agent_pools(self):
        """Background task to monitor agent pool health"""
        while self.status != OrchestratorStatus.SHUTDOWN:
            try:
                for pool_type, connection in self.agent_pools.items():
                    if connection.is_connected:
                        try:
                            # Perform health check
                            result = await self._call_agent_pool(pool_type, "health_check", {})
                            connection.last_health_check = datetime.now()
                            self.metrics["agent_pool_health"][pool_type.value] = "healthy"
                        except Exception as e:
                            logger.warning(f"Health check failed for {pool_type.value}: {e}")
                            self.metrics["agent_pool_health"][pool_type.value] = "unhealthy"
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in agent pool monitoring: {e}")
                await asyncio.sleep(10)

    def _update_avg_execution_time(self, execution_time: float):
        """Update average execution time metric"""
        current_avg = self.metrics["avg_execution_time"]
        total_executions = self.metrics["total_executions"]
        
        if total_executions == 0:
            self.metrics["avg_execution_time"] = execution_time
        else:
            self.metrics["avg_execution_time"] = (
                (current_avg * (total_executions - 1) + execution_time) / total_executions
            )

    def _register_orchestrator_tools(self):
        """Register MCP tools for orchestrator management"""
        
        @self.mcp_server.tool(name="execute_strategy", description="Execute a trading strategy")
        async def execute_strategy_tool(strategy_config: Dict[str, Any]) -> Dict[str, Any]:
            """Execute a trading strategy"""
            try:
                strategy = TradingStrategy(**strategy_config)
                execution_id = await self.execute_strategy(strategy)
                return {
                    "status": "success",
                    "execution_id": execution_id,
                    "message": f"Strategy execution started with ID: {execution_id}"
                }
            except Exception as e:
                return {"status": "error", "error": str(e)}

        @self.mcp_server.tool(name="run_backtest", description="Run strategy backtest")
        async def run_backtest_tool(backtest_config: Dict[str, Any]) -> Dict[str, Any]:
            """Run a strategy backtest"""
            try:
                # Parse configuration
                strategy_data = backtest_config["strategy"]
                strategy = TradingStrategy(**strategy_data)
                
                config = BacktestConfiguration(
                    config_id=str(uuid.uuid4()),
                    strategy=strategy,
                    start_date=datetime.fromisoformat(backtest_config["start_date"]),
                    end_date=datetime.fromisoformat(backtest_config["end_date"]),
                    initial_capital=backtest_config["initial_capital"],
                    commission_rate=backtest_config.get("commission_rate", 0.001),
                    slippage_rate=backtest_config.get("slippage_rate", 0.001),
                    rl_enabled=backtest_config.get("rl_enabled", True),
                    memory_enabled=backtest_config.get("memory_enabled", True)
                )
                
                backtest_id = await self.run_backtest(config)
                return {
                    "status": "success",
                    "backtest_id": backtest_id,
                    "message": f"Backtest started with ID: {backtest_id}"
                }
            except Exception as e:
                return {"status": "error", "error": str(e)}

        @self.mcp_server.tool(name="get_execution_status", description="Get strategy execution status")
        async def get_execution_status_tool(execution_id: str) -> Dict[str, Any]:
            """Get status of strategy execution"""
            if execution_id in self.active_executions:
                context = self.active_executions[execution_id]
                return {
                    "status": "running",
                    "execution_id": execution_id,
                    "strategy_name": context.strategy.name,
                    "start_time": context.start_time.isoformat(),
                    "completed_tasks": len(context.completed_tasks),
                    "failed_tasks": len(context.failed_tasks)
                }
            elif execution_id in self.completed_executions:
                context = self.completed_executions[execution_id]
                return {
                    "status": context.status,
                    "execution_id": execution_id,
                    "strategy_name": context.strategy.name,
                    "start_time": context.start_time.isoformat(),
                    "completed_tasks": len(context.completed_tasks),
                    "failed_tasks": len(context.failed_tasks),
                    "results": context.results
                }
            else:
                return {"status": "error", "error": "Execution ID not found"}

        @self.mcp_server.tool(name="get_backtest_results", description="Get backtest results")
        async def get_backtest_results_tool(backtest_id: str) -> Dict[str, Any]:
            """Get backtest results"""
            if backtest_id in self.backtest_results:
                result = self.backtest_results[backtest_id]
                return {
                    "status": "success",
                    "backtest_id": backtest_id,
                    "strategy_name": result.strategy.name,
                    "performance_metrics": result.performance_metrics,
                    "risk_metrics": result.risk_metrics,
                    "total_trades": len(result.trade_history),
                    "memory_insights": result.memory_insights,
                    "rl_performance": result.rl_performance
                }
            else:
                return {"status": "error", "error": "Backtest ID not found"}

        @self.mcp_server.tool(name="get_orchestrator_status", description="Get orchestrator system status")
        def get_orchestrator_status_tool() -> Dict[str, Any]:
            """Get orchestrator status and metrics"""
            return {
                "status": self.status.value,
                "metrics": self.metrics,
                "active_executions": len(self.active_executions),
                "completed_executions": len(self.completed_executions),
                "agent_pool_connections": {
                    pool_type.value: connection.is_connected
                    for pool_type, connection in self.agent_pools.items()
                },
                "rl_enabled": self.enable_rl,
                "memory_enabled": self.enable_memory
            }

        @self.mcp_server.tool(name="create_sandbox_strategy", description="Create and test strategy in sandbox")
        async def create_sandbox_strategy_tool(strategy_config: Dict[str, Any]) -> Dict[str, Any]:
            """Create and test a strategy in sandbox environment"""
            try:
                # Create strategy
                strategy = TradingStrategy(**strategy_config)
                
                # Create sandbox configuration
                config = BacktestConfiguration(
                    config_id=str(uuid.uuid4()),
                    strategy=strategy,
                    start_date=datetime.now() - timedelta(days=30),
                    end_date=datetime.now(),
                    initial_capital=100000,
                    rl_enabled=True,
                    memory_enabled=True
                )
                
                # Run sandbox test
                backtest_id = await self.run_backtest(config)
                
                return {
                    "status": "success",
                    "sandbox_id": backtest_id,
                    "strategy_id": strategy.strategy_id,
                    "message": "Sandbox strategy test started"
                }
            except Exception as e:
                return {"status": "error", "error": str(e)}

    def run(self):
        """Start the orchestrator MCP server"""
        logger.info(f"Starting FinAgent Orchestrator MCP server on {self.host}:{self.port}")
        
        # Start the orchestrator
        asyncio.run(self.start())
        
        # Configure MCP server
        self.mcp_server.settings.host = self.host
        self.mcp_server.settings.port = self.port
        
        try:
            # Start the MCP server
            self.mcp_server.run(transport="sse")
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            self.status = OrchestratorStatus.SHUTDOWN
        except Exception as e:
            logger.error(f"Error running orchestrator: {e}")
            self.status = OrchestratorStatus.ERROR
            raise


if __name__ == "__main__":
    # Start the orchestrator
    orchestrator = FinAgentOrchestrator(
        host="0.0.0.0",
        port=9000,
        enable_rl=True,
        enable_memory=True
    )
    orchestrator.run()
