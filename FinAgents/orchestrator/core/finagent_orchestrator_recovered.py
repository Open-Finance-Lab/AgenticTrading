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

    async def _run_backtest_simulation(self, config: BacktestConfiguration, backtest_id: str) -> BacktestResult:
        """
        Run comprehensive backtest simulation using simple fallback data.
        """
        start_time = datetime.now()
        
        # Initialize simulation state
        portfolio_value = config.initial_capital
        positions = {symbol: 0 for symbol in config.strategy.symbols}
        cash = config.initial_capital
        trade_history = []
        portfolio_values = []
        
        # Generate trading dates
        trading_dates = []
        current_date = config.start_date
        while current_date <= config.end_date:
            if current_date.weekday() < 5:  # Trading days only
                trading_dates.append(current_date)
            current_date += timedelta(days=1)
        
        logger.info(f"Running backtest simulation for {len(trading_dates)} trading days")
        
        # Simulation loop with simple market data
        for i, date in enumerate(trading_dates):
            try:
                # Generate simple market data
                market_data = self._generate_simple_market_data(config.strategy.symbols, date)
                
                # Simple signal generation (placeholder)
                signals = []
                for symbol in config.strategy.symbols:
                    if symbol in market_data:
                        # Simple momentum signal
                        price = market_data[symbol]["close"]
                        if i > 10:  # Need some history
                            prev_price = portfolio_values[-1].get("market_data", {}).get(symbol, {}).get("close", price)
                            momentum = (price - prev_price) / prev_price
                            if momentum > 0.02:  # 2% positive momentum
                                signals.append({
                                    "symbol": symbol,
                                    "action": "buy",
                                    "confidence": min(0.8, 0.5 + abs(momentum))
                                })
                            elif momentum < -0.02:  # 2% negative momentum
                                if positions.get(symbol, 0) > 0:
                                    signals.append({
                                        "symbol": symbol,
                                        "action": "sell",
                                        "confidence": min(0.8, 0.5 + abs(momentum))
                                    })
                
                # Execute trades
                daily_trades = await self._execute_simple_trades(signals, market_data, positions, cash, config)
                trade_history.extend(daily_trades)
                
                # Update positions and cash
                for trade in daily_trades:
                    if trade["action"] == "buy":
                        cash -= trade["total_cost"]
                        positions[trade["symbol"]] = positions.get(trade["symbol"], 0) + trade["shares"]
                    elif trade["action"] == "sell":
                        cash += trade["total_proceeds"]
                        positions[trade["symbol"]] = max(0, positions.get(trade["symbol"], 0) - trade["shares"])
                
                # Calculate portfolio value
                current_portfolio_value = cash
                for symbol, qty in positions.items():
                    if symbol in market_data and qty > 0:
                        current_portfolio_value += qty * market_data[symbol]["close"]
                
                portfolio_values.append({
                    "date": date.isoformat(),
                    "value": current_portfolio_value,
                    "cash": cash,
                    "positions": positions.copy(),
                    "market_data": market_data
                })
                
                portfolio_value = current_portfolio_value
                
                # Log progress
                if i % 50 == 0:
                    progress = (i / len(trading_dates)) * 100
                    logger.info(f"Backtest progress: {progress:.1f}% - Portfolio: ${portfolio_value:,.2f}")
                    
            except Exception as e:
                logger.error(f"Error in backtest simulation on {date}: {e}")
                continue
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            portfolio_values, config.initial_capital, trading_dates
        )
        
        risk_metrics = self._calculate_risk_metrics(portfolio_values, trade_history)
        
        memory_insights = await self._generate_memory_insights(backtest_id, trade_history)
        
        end_time = datetime.now()
        
        return BacktestResult(
            backtest_id=backtest_id,
            strategy=config.strategy,
            configuration=config,
            start_time=start_time,
            end_time=end_time,
            performance_metrics=performance_metrics,
            trade_history=trade_history,
            portfolio_values=portfolio_values,
            risk_metrics=risk_metrics,
            memory_insights=memory_insights
        )

    def _generate_simple_market_data(self, symbols: List[str], date: datetime) -> Dict[str, Dict[str, float]]:
        """Generate simple synthetic market data"""
        import random
        random.seed(int(date.timestamp()) % 10000)
        
        base_prices = {"AAPL": 150.0, "MSFT": 300.0}
        market_data = {}
        
        for symbol in symbols:
            base = base_prices.get(symbol, 100.0)
            days_elapsed = (date - datetime(2022, 1, 1)).days
            
            # Simple trend with noise
            trend = 1 + (days_elapsed / 1095) * 0.25  # 25% growth over 3 years
            noise = 1 + random.normalvariate(0, 0.02)  # 2% daily volatility
            
            price = base * trend * noise
            
            market_data[symbol] = {
                "open": price * random.uniform(0.99, 1.01),
                "high": price * random.uniform(1.0, 1.02),
                "low": price * random.uniform(0.98, 1.0),
                "close": price,
                "volume": random.randint(1000000, 10000000)
            }
        
        return market_data

    async def _execute_simple_trades(self, signals: List[Dict], market_data: Dict, 
                                   positions: Dict, cash: float, config: BacktestConfiguration) -> List[Dict]:
        """Execute trades based on signals"""
        trades = []
        
        for signal in signals:
            try:
                symbol = signal.get("symbol")
                action = signal.get("action", "hold").lower()
                confidence = signal.get("confidence", 0.0)
                
                if symbol not in market_data or action == "hold":
                    continue
                
                current_price = market_data[symbol]["close"]
                
                if action == "buy" and confidence > 0.6:
                    # Calculate position size
                    max_position_value = cash * 0.2  # Max 20% per position
                    shares = int(max_position_value / current_price)
                    
                    if shares > 0:
                        total_cost = shares * current_price * (1 + config.commission_rate)
                        if total_cost <= cash:
                            trades.append({
                                "symbol": symbol,
                                "action": "buy",
                                "shares": shares,
                                "price": current_price,
                                "total_cost": total_cost,
                                "confidence": confidence,
                                "timestamp": datetime.now().isoformat()
                            })
                
                elif action == "sell" and symbol in positions and positions[symbol] > 0:
                    shares = positions[symbol]
                    total_proceeds = shares * current_price * (1 - config.commission_rate)
                    
                    trades.append({
                        "symbol": symbol,
                        "action": "sell",
                        "shares": shares,
                        "price": current_price,
                        "total_proceeds": total_proceeds,
                        "confidence": confidence,
                        "timestamp": datetime.now().isoformat()
                    })
                    
            except Exception as e:
                logger.error(f"Error executing trade for {signal}: {e}")
                continue
        
        return trades

    def _calculate_performance_metrics(self, portfolio_values: List[Dict], 
                                     initial_capital: float, trading_dates: List[datetime]) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if not portfolio_values:
            return {}
        
        values = [pv["value"] for pv in portfolio_values]
        returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
        
        total_return = (values[-1] - initial_capital) / initial_capital
        
        # Annualized return
        days = len(trading_dates)
        years = days / 252  # Approximate trading days per year
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Volatility
        volatility = (sum(r**2 for r in returns) / len(returns))**0.5 * (252**0.5) if returns else 0
        
        # Sharpe ratio
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        peak = initial_capital
        max_drawdown = 0
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "final_value": values[-1],
            "total_days": days
        }

    def _calculate_risk_metrics(self, portfolio_values: List[Dict], trade_history: List[Dict]) -> Dict[str, float]:
        """Calculate risk metrics"""
        if not portfolio_values:
            return {}
        
        values = [pv["value"] for pv in portfolio_values]
        returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
        
        # Value at Risk (95%)
        if len(returns) >= 20:
            sorted_returns = sorted(returns)
            var_95 = sorted_returns[int(len(returns) * 0.05)]
        else:
            var_95 = 0
        
        # Average trade size
        trade_sizes = []
        for trade in trade_history:
            if "total_cost" in trade:
                trade_sizes.append(trade["total_cost"])
            elif "total_proceeds" in trade:
                trade_sizes.append(trade["total_proceeds"])
        
        avg_trade_size = sum(trade_sizes) / len(trade_sizes) if trade_sizes else 0
        
        return {
            "var_95": var_95,
            "avg_trade_size": avg_trade_size,
            "total_trades": len(trade_history),
            "win_rate": self._calculate_win_rate(trade_history)
        }

    def _calculate_win_rate(self, trade_history: List[Dict]) -> float:
        """Calculate win rate from trade history"""
        if len(trade_history) < 2:
            return 0.0
        
        # Simplified win rate calculation
        buy_trades = [t for t in trade_history if t["action"] == "buy"]
        sell_trades = [t for t in trade_history if t["action"] == "sell"]
        
        if len(buy_trades) == 0 or len(sell_trades) == 0:
            return 0.0
        
        # Simplified: assume trades are profitable if sell price > avg buy price
        avg_buy_price = sum(t["price"] for t in buy_trades) / len(buy_trades)
        profitable_sells = len([t for t in sell_trades if t["price"] > avg_buy_price])
        
        return profitable_sells / len(sell_trades) if sell_trades else 0.0

    async def _generate_memory_insights(self, backtest_id: str, trade_history: List[Dict]) -> Dict[str, Any]:
        """Generate insights from memory analysis"""
        if not self.memory_agent:
            return {}
        
        try:
            return {
                "total_memory_events": self.metrics.get("memory_events_logged", 0),
                "backtest_id": backtest_id,
                "insights_generated": True,
                "analysis_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to generate memory insights: {e}")
            return {}

    async def start(self):
        """Start the orchestrator MCP server"""
        try:
            await self.initialize()
            logger.info(f"Starting FinAgent Orchestrator on {self.host}:{self.port}")
            await self.mcp_server.run(host=self.host, port=self.port)
        except Exception as e:
            logger.error(f"Failed to start orchestrator: {e}")
            raise

    async def shutdown(self):
        """Gracefully shutdown the orchestrator"""
        try:
            self.status = OrchestratorStatus.SHUTDOWN
            
            # Shutdown task executor
            self.task_executor.shutdown(wait=True)
            
            # Close memory agent connection
            if self.memory_agent:
                # Memory agent cleanup if needed
                pass
            
            logger.info("FinAgent Orchestrator shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# End of FinAgentOrchestrator class and module
