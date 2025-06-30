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
        Run comprehensive backtest simulation using real data from memory agent and agent pools.
        
        This method implements the complete data-driven workflow:
        1. Load historical data from memory agent
        2. Validate and supplement data via data agent pool
        3. Generate strategies via alpha agent pool with memory context
        4. Execute trades with full attribution tracking
        5. Enable memory-guided DAG replanning and agent tuning
        """
        start_time = datetime.now()
        
        # Initialize simulation state
        portfolio_value = config.initial_capital
        positions = {symbol: 0 for symbol in config.strategy.symbols}
        cash = config.initial_capital
        trade_history = []
        portfolio_values = []
        decision_chain = []  # Track decision attribution
        
        # Step 1: Load historical data from memory agent
        logger.info("🔍 Step 1: Loading historical market data from memory agent...")
        historical_data = await self._load_historical_data_from_memory(
            symbols=config.strategy.symbols,
            start_date=config.start_date,
            end_date=config.end_date,
            backtest_id=backtest_id
        )
        
        # Step 2: Validate data completeness via data agent pool
        logger.info("✅ Step 2: Validating data completeness via data agent pool...")
        data_validation_result = await self._validate_and_supplement_data(
            historical_data=historical_data,
            symbols=config.strategy.symbols,
            date_range=(config.start_date, config.end_date),
            backtest_id=backtest_id
        )
        
        # Use validated/supplemented data
        market_data_cache = data_validation_result.get("market_data", historical_data)
        
        # Generate trading dates
        trading_dates = []
        current_date = config.start_date
        while current_date <= config.end_date:
            if current_date.weekday() < 5:  # Trading days only
                trading_dates.append(current_date)
            current_date += timedelta(days=1)
        
        logger.info(f"🚀 Running memory-driven backtest simulation for {len(trading_dates)} trading days")
        
        # Step 3: Initialize strategy context with memory insights
        strategy_context = await self._initialize_strategy_context_from_memory(
            strategy=config.strategy,
            backtest_id=backtest_id
        )
        
        # Simulation loop
        for i, date in enumerate(trading_dates):
            try:
                # Get market data for this date
                daily_market_data = self._extract_daily_market_data(market_data_cache, date)
                
                if not daily_market_data:
                    # Fallback to synthetic data if no real data available
                    daily_market_data = self._generate_simple_market_data(config.strategy.symbols, date)
                    
                    await self._log_memory_event(
                        event_type=EventType.DATA_RETRIEVAL,
                        log_level=LogLevel.WARNING,
                        title=f"Using synthetic data for {date}",
                        content=f"No historical data found for {date}, using fallback synthetic data",
                        tags={"data", "synthetic", "fallback"},
                        metadata={"date": date.isoformat(), "symbols": config.strategy.symbols},
                        correlation_id=backtest_id
                    )
                
                # Step 4: Generate strategy signals via alpha agent pool with memory context
                alpha_signals = await self._generate_alpha_signals_with_memory(
                    strategy=config.strategy,
                    market_data=daily_market_data,
                    strategy_context=strategy_context,
                    date=date,
                    backtest_id=backtest_id,
                    positions=positions,
                    cash=cash
                )
                
                # Step 5: Risk assessment and signal validation
                validated_signals = await self._validate_signals_with_risk_assessment(
                    signals=alpha_signals,
                    market_data=daily_market_data,
                    positions=positions,
                    cash=cash,
                    config=config,
                    date=date,
                    backtest_id=backtest_id
                )
                
                # Step 6: Execute trades with full attribution tracking
                daily_trades = await self._execute_attributed_trades(
                    signals=validated_signals,
                    market_data=daily_market_data,
                    positions=positions,
                    cash=cash,
                    config=config,
                    date=date,
                    backtest_id=backtest_id,
                    decision_chain=decision_chain
                )
                
                trade_history.extend(daily_trades)
                
                # Update cash and positions based on trades
                for trade in daily_trades:
                    if trade["action"] == "buy":
                        cash -= trade["total_cost"]
                        positions[trade["symbol"]] = positions.get(trade["symbol"], 0) + trade["shares"]
                    elif trade["action"] == "sell":
                        cash += trade["total_proceeds"]
                        positions[trade["symbol"]] = max(0, positions.get(trade["symbol"], 0) - trade["shares"])
                
                # Calculate portfolio value using current market data
                current_portfolio_value = cash
                for symbol, qty in positions.items():
                    if symbol in daily_market_data and qty > 0:
                        current_portfolio_value += qty * daily_market_data[symbol]["close"]
                
                portfolio_values.append({
                    "date": date.isoformat(),
                    "value": current_portfolio_value,
                    "cash": cash,
                    "positions": positions.copy(),
                    "decision_attribution": decision_chain[-1] if decision_chain else {},
                    "market_data": daily_market_data
                })
                
                portfolio_value = current_portfolio_value
                
                # Step 7: Performance analysis and adaptive learning
                if i > 0 and i % 20 == 0:  # Every 20 days, analyze performance
                    await self._analyze_performance_and_adapt(
                        portfolio_values=portfolio_values[-20:],
                        trade_history=trade_history[-50:],
                        decision_chain=decision_chain[-50:],
                        strategy_context=strategy_context,
                        backtest_id=backtest_id
                    )
                
                # Log progress periodically with detailed attribution
                if i % 50 == 0:
                    progress = (i / len(trading_dates)) * 100
                    recent_performance = self._calculate_recent_performance(portfolio_values[-10:]) if len(portfolio_values) >= 10 else {}
                    
                    await self._log_memory_event(
                        event_type=EventType.SYSTEM,
                        log_level=LogLevel.INFO,
                        title=f"Backtest Progress: {progress:.1f}%",
                        content=f"Portfolio: ${portfolio_value:,.2f}, Recent performance: {recent_performance}",
                        tags={"backtest", "progress", "performance"},
                        metadata={
                            "backtest_id": backtest_id,
                            "progress_pct": progress,
                            "portfolio_value": portfolio_value,
                            "recent_performance": recent_performance,
                            "total_trades": len(trade_history)
                        },
                        correlation_id=backtest_id
                    )
                    
                    logger.info(f"📊 Backtest progress: {progress:.1f}% - Portfolio: ${portfolio_value:,.2f}")
                    
            except Exception as e:
                logger.error(f"❌ Error in backtest simulation on {date}: {e}")
                # Log the error for attribution analysis
                await self._log_memory_event(
                    event_type=EventType.ERROR,
                    log_level=LogLevel.ERROR,
                    title=f"Backtest simulation error on {date}",
                    content=f"Error: {str(e)}",
                    tags={"backtest", "error", "simulation"},
                    metadata={
                        "backtest_id": backtest_id,
                        "error_date": date.isoformat(),
                        "error_type": type(e).__name__
                    },
                    correlation_id=backtest_id
                )
                continue
        
        # Final performance analysis with complete attribution
        logger.info("📈 Calculating comprehensive performance metrics with attribution...")
        
        performance_metrics = self._calculate_performance_metrics(
            portfolio_values, config.initial_capital, trading_dates
        )
        
        risk_metrics = self._calculate_risk_metrics(portfolio_values, trade_history)
        
        # Generate comprehensive memory insights with attribution analysis
        memory_insights = await self._generate_comprehensive_memory_insights(
            backtest_id=backtest_id,
            trade_history=trade_history,
            decision_chain=decision_chain,
            portfolio_values=portfolio_values,
            performance_metrics=performance_metrics
        )
        
        # Generate improvement recommendations based on analysis
        improvement_recommendations = await self._generate_improvement_recommendations(
            backtest_id=backtest_id,
            performance_metrics=performance_metrics,
            decision_chain=decision_chain,
            memory_insights=memory_insights
        )
        
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
            memory_insights=memory_insights,
            rl_performance=improvement_recommendations
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

# ================================
    # Memory-Driven Data Loading Methods
    # ================================
    
    async def _load_historical_data_from_memory(self, symbols: List[str], 
                                              start_date: datetime, end_date: datetime,
                                              backtest_id: str) -> Dict[str, Any]:
        """
        Load historical market data from memory agent storage.
        
        This method queries the memory agent for previously stored market data,
        including price data, volume, and any derived indicators.
        """
        if not self.memory_agent:
            logger.warning("⚠️ Memory agent not available, cannot load historical data")
            return {}
        
        try:
            await self._log_memory_event(
                event_type=EventType.DATA_RETRIEVAL,
                log_level=LogLevel.INFO,
                title="Loading historical data from memory",
                content=f"Querying memory for {len(symbols)} symbols from {start_date} to {end_date}",
                tags={"data", "historical", "memory", "load"},
                metadata={
                    "symbols": symbols,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "backtest_id": backtest_id
                },
                correlation_id=backtest_id
            )
            
            # Query memory agent for stored market data
            historical_data = {}
            for symbol in symbols:
                # Search for historical data events in memory
                symbol_data = await self.memory_agent.retrieve_events(
                    source_agent_pool="data_agent_pool",
                    event_types=[EventType.DATA_RETRIEVAL, EventType.MARKET_DATA],
                    tags={"symbol", symbol.lower(), "historical"},
                    start_time=start_date,
                    end_time=end_date,
                    limit=1000
                )
                
                if symbol_data:
                    # Process and structure the data
                    processed_data = self._process_memory_market_data(symbol_data, symbol)
                    historical_data[symbol] = processed_data
                    logger.info(f"📊 Loaded {len(processed_data)} data points for {symbol} from memory")
                else:
                    logger.warning(f"⚠️ No historical data found in memory for {symbol}")
            
            return historical_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load historical data from memory: {e}")
            return {}
    
    def _process_memory_market_data(self, memory_events: List[Dict], symbol: str) -> Dict[str, Any]:
        """Process market data events from memory into structured format"""
        processed_data = {}
        
        for event in memory_events:
            try:
                event_data = event.get('metadata', {})
                if 'price_data' in event_data:
                    date_str = event.get('timestamp', '')
                    if date_str:
                        processed_data[date_str] = event_data['price_data']
            except Exception as e:
                logger.warning(f"⚠️ Error processing market data event: {e}")
                continue
        
        return processed_data
    
    async def _validate_and_supplement_data(self, historical_data: Dict[str, Any], 
                                          symbols: List[str], date_range: Tuple[datetime, datetime],
                                          backtest_id: str) -> Dict[str, Any]:
        """
        Validate data completeness and supplement missing data via data agent pool.
        
        This method checks for data gaps and fetches missing data through the data agent pool.
        """
        start_date, end_date = date_range
        
        await self._log_memory_event(
            event_type=EventType.DATA_VALIDATION,
            log_level=LogLevel.INFO,
            title="Validating data completeness",
            content=f"Checking data coverage for {len(symbols)} symbols",
            tags={"data", "validation", "completeness"},
            metadata={
                "symbols": symbols,
                "date_range": [start_date.isoformat(), end_date.isoformat()],
                "backtest_id": backtest_id
            },
            correlation_id=backtest_id
        )
        
        try:
            # Connect to data agent pool for validation
            data_pool_connection = self.agent_pools.get(AgentPoolType.DATA_AGENT_POOL)
            if not data_pool_connection or not data_pool_connection.is_connected:
                logger.warning("⚠️ Data agent pool not available for validation")
                return {"market_data": historical_data, "validation_status": "partial"}
            
            # For demo purposes, simulate data validation
            # In real implementation, this would connect to actual data agent pool
            logger.info("🔍 Simulating data agent pool validation...")
            
            # Check for missing data (simplified)
            missing_symbols = []
            for symbol in symbols:
                if symbol not in historical_data or len(historical_data[symbol]) < 100:
                    missing_symbols.append(symbol)
            
            if missing_symbols:
                logger.warning(f"⚠️ Missing data detected for symbols: {missing_symbols}")
                await self._log_memory_event(
                    event_type=EventType.DATA_RETRIEVAL,
                    log_level=LogLevel.WARNING,
                    title="Data gaps detected",
                    content=f"Missing data for symbols: {missing_symbols}",
                    tags={"data", "gaps", "missing"},
                    metadata={
                        "missing_symbols": missing_symbols,
                        "backtest_id": backtest_id
                    },
                    correlation_id=backtest_id
                )
            
            return {
                "market_data": historical_data,
                "validation_status": "complete" if not missing_symbols else "partial",
                "missing_symbols": missing_symbols
            }
            
        except Exception as e:
            logger.error(f"❌ Data validation failed: {e}")
            return {"market_data": historical_data, "validation_status": "error", "error": str(e)}
    
    def _extract_daily_market_data(self, market_data_cache: Dict[str, Any], date: datetime) -> Dict[str, Dict[str, float]]:
        """Extract market data for a specific trading date"""
        daily_data = {}
        date_str = date.strftime('%Y-%m-%d')
        
        for symbol, symbol_data in market_data_cache.items():
            if date_str in symbol_data:
                daily_data[symbol] = symbol_data[date_str]
            else:
                # Try alternative date formats or closest date
                for stored_date, data in symbol_data.items():
                    if stored_date.startswith(date_str):
                        daily_data[symbol] = data
                        break
        
        return daily_data
    
    # ================================
    # Memory-Enhanced Strategy Generation
    # ================================
    
    async def _initialize_strategy_context_from_memory(self, strategy: TradingStrategy, backtest_id: str) -> Dict[str, Any]:
        """
        Initialize strategy execution context using insights from memory agent.
        
        This loads historical strategy performance, market patterns, and optimization insights.
        """
        if not self.memory_agent:
            return {"initialized": False}
        
        try:
            # Query memory for similar strategy executions
            similar_strategies = await self.memory_agent.retrieve_events(
                event_types=[EventType.OPTIMIZATION, EventType.STRATEGY_EXECUTION],
                tags={"strategy", strategy.strategy_type.lower()},
                limit=50
            )
            
            # Query for market pattern insights
            market_insights = await self.memory_agent.retrieve_events(
                event_types=[EventType.MARKET_ANALYSIS, EventType.PATTERN_DETECTION],
                tags=set(symbol.lower() for symbol in strategy.symbols),
                limit=100
            )
            
            # Process insights into strategy context
            strategy_context = {
                "strategy_id": strategy.strategy_id,
                "historical_performance": self._analyze_historical_strategy_performance(similar_strategies),
                "market_patterns": self._extract_market_patterns(market_insights),
                "optimization_history": self._extract_optimization_insights(similar_strategies),
                "risk_insights": await self._get_risk_insights_from_memory(strategy.symbols),
                "backtest_id": backtest_id,
                "initialized": True
            }
            
            await self._log_memory_event(
                event_type=EventType.SYSTEM,
                log_level=LogLevel.INFO,
                title="Strategy context initialized from memory",
                content=f"Loaded context for {strategy.name} with {len(similar_strategies)} historical executions",
                tags={"strategy", "context", "memory", "initialization"},
                metadata={
                    "strategy_id": strategy.strategy_id,
                    "historical_executions": len(similar_strategies),
                    "market_insights": len(market_insights),
                    "context_summary": strategy_context
                },
                correlation_id=backtest_id
            )
            
            return strategy_context
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize strategy context from memory: {e}")
            return {"initialized": False, "error": str(e)}
    
    def _analyze_historical_strategy_performance(self, strategy_events: List[Dict]) -> Dict[str, Any]:
        """Analyze historical performance of similar strategies"""
        if not strategy_events:
            return {"no_history": True}
        
        performance_data = []
        for event in strategy_events:
            metadata = event.get('metadata', {})
            if 'performance_metrics' in metadata:
                performance_data.append(metadata['performance_metrics'])
        
        if not performance_data:
            return {"no_performance_data": True}
        
        # Calculate aggregate statistics
        returns = [p.get('total_return', 0) for p in performance_data]
        sharpe_ratios = [p.get('sharpe_ratio', 0) for p in performance_data]
        max_drawdowns = [p.get('max_drawdown', 0) for p in performance_data]
        
        return {
            "historical_executions": len(performance_data),
            "avg_return": sum(returns) / len(returns) if returns else 0,
            "avg_sharpe": sum(sharpe_ratios) / len(sharpe_ratios) if sharpe_ratios else 0,
            "avg_max_drawdown": sum(max_drawdowns) / len(max_drawdowns) if max_drawdowns else 0,
            "success_rate": len([r for r in returns if r > 0]) / len(returns) if returns else 0
        }
    
    def _extract_market_patterns(self, market_events: List[Dict]) -> Dict[str, Any]:
        """Extract market patterns and insights from memory events"""
        patterns = {
            "volatility_patterns": [],
            "trend_patterns": [],
            "seasonal_patterns": [],
            "correlation_patterns": []
        }
        
        for event in market_events:
            metadata = event.get('metadata', {})
            if 'pattern_type' in metadata:
                pattern_type = metadata['pattern_type']
                if pattern_type in patterns:
                    patterns[pattern_type].append(metadata)
        
        return patterns
    
    def _extract_optimization_insights(self, strategy_events: List[Dict]) -> Dict[str, Any]:
        """Extract optimization insights from historical strategy executions"""
        optimizations = []
        
        for event in strategy_events:
            metadata = event.get('metadata', {})
            if 'optimization_result' in metadata:
                optimizations.append(metadata['optimization_result'])
        
        return {
            "optimization_count": len(optimizations),
            "optimizations": optimizations[-10:] if optimizations else []  # Keep last 10
        }
    
    async def _get_risk_insights_from_memory(self, symbols: List[str]) -> Dict[str, Any]:
        """Get risk insights for symbols from memory"""
        if not self.memory_agent:
            return {}
        
        try:
            risk_events = await self.memory_agent.retrieve_events(
                event_types=[EventType.RISK_ASSESSMENT, EventType.ERROR],
                tags=set(symbol.lower() for symbol in symbols),
                limit=100
            )
            
            risk_insights = {
                "total_risk_events": len(risk_events),
                "error_patterns": [],
                "risk_warnings": []
            }
            
            for event in risk_events:
                metadata = event.get('metadata', {})
                if event.get('event_type') == 'ERROR':
                    risk_insights["error_patterns"].append({
                        "error_type": metadata.get('error_type', 'unknown'),
                        "timestamp": event.get('timestamp'),
                        "description": event.get('content', '')
                    })
                elif 'risk_level' in metadata:
                    risk_insights["risk_warnings"].append(metadata)
            
            return risk_insights
            
        except Exception as e:
            logger.error(f"❌ Failed to get risk insights from memory: {e}")
            return {}
    
    async def _generate_alpha_signals_with_memory(self, strategy: TradingStrategy, 
                                                market_data: Dict[str, Dict[str, float]],
                                                strategy_context: Dict[str, Any],
                                                date: datetime, backtest_id: str,
                                                positions: Dict[str, int], cash: float) -> List[Dict[str, Any]]:
        """
        Generate trading signals via alpha agent pool with memory context.
        
        This method provides historical context and market insights to the alpha agent pool
        for enhanced signal generation.
        """
        try:
            alpha_pool_connection = self.agent_pools.get(AgentPoolType.ALPHA_AGENT_POOL)
            if not alpha_pool_connection or not alpha_pool_connection.is_connected:
                logger.warning("⚠️ Alpha agent pool not available, using fallback signals")
                return self._generate_fallback_alpha_signals(market_data, strategy, date)
            
            # For demo purposes, simulate alpha agent pool interaction
            # In real implementation, this would connect to actual alpha agent pool
            logger.info(f"🧠 Generating alpha signals with memory context for {date}")
            
            signals = []
            for symbol in strategy.symbols:
                if symbol in market_data:
                    # Enhanced signal generation using strategy context
                    historical_perf = strategy_context.get("historical_performance", {})
                    market_patterns = strategy_context.get("market_patterns", {})
                    
                    # Simple momentum with memory enhancement
                    price = market_data[symbol]["close"]
                    signal_strength = 0.5
                    
                    # Adjust based on historical performance
                    avg_return = historical_perf.get("avg_return", 0)
                    if avg_return > 0.1:  # Good historical performance
                        signal_strength += 0.2
                    
                    # Check market patterns
                    volatility_patterns = market_patterns.get("volatility_patterns", [])
                    if len(volatility_patterns) > 0:
                        signal_strength += 0.1
                    
                    # Generate signal based on simple momentum
                    if len(positions) > 0 and symbol in positions:
                        # Simple sell logic
                        if positions[symbol] > 0 and signal_strength < 0.6:
                            signals.append({
                                "symbol": symbol,
                                "action": "sell",
                                "confidence": signal_strength,
                                "reasoning": f"Memory-enhanced sell signal based on context",
                                "attribution": {
                                    "source": "alpha_agent_pool",
                                    "memory_context": True,
                                    "historical_perf": avg_return,
                                    "signal_strength": signal_strength
                                }
                            })
                    else:
                        # Simple buy logic
                        if cash > price * 10:  # Can afford at least 10 shares
                            signals.append({
                                "symbol": symbol,
                                "action": "buy",
                                "confidence": signal_strength,
                                "reasoning": f"Memory-enhanced buy signal based on context",
                                "attribution": {
                                    "source": "alpha_agent_pool",
                                    "memory_context": True,
                                    "historical_perf": avg_return,
                                    "signal_strength": signal_strength
                                }
                            })
            
            # Log signal generation to memory
            await self._log_memory_event(
                event_type=EventType.ALPHA_SIGNAL,
                log_level=LogLevel.INFO,
                title=f"Alpha signals generated for {date}",
                content=f"Generated {len(signals)} signals with memory context",
                tags={"alpha", "signals", "memory", "context"},
                metadata={
                    "date": date.isoformat(),
                    "signals_count": len(signals),
                    "signals": signals,
                    "strategy_context": strategy_context,
                    "backtest_id": backtest_id
                },
                correlation_id=backtest_id
            )
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Failed to generate alpha signals with memory: {e}")
            await self._log_memory_event(
                event_type=EventType.ERROR,
                log_level=LogLevel.ERROR,
                title=f"Alpha signal generation failed for {date}",
                content=f"Error: {str(e)}",
                tags={"alpha", "error", "signal_generation"},
                metadata={
                    "date": date.isoformat(),
                    "error_type": type(e).__name__,
                    "backtest_id": backtest_id
                },
                correlation_id=backtest_id
            )
            return []
    
    def _generate_fallback_alpha_signals(self, market_data: Dict[str, Dict[str, float]], 
                                       strategy: TradingStrategy, date: datetime) -> List[Dict[str, Any]]:
        """Generate fallback alpha signals when agent pool is not available"""
        signals = []
        
        for symbol in strategy.symbols:
            if symbol in market_data:
                price = market_data[symbol]["close"]
                # Simple momentum signal as fallback
                signals.append({
                    "symbol": symbol,
                    "action": "buy",
                    "confidence": 0.5,
                    "reasoning": "Fallback momentum signal",
                    "attribution": {
                        "source": "fallback_generator",
                        "memory_context": False
                    }
                })
        
        return signals
    
    # ================================
    # Risk Assessment and Trade Execution with Attribution
    # ================================
    
    async def _validate_signals_with_risk_assessment(self, signals: List[Dict[str, Any]], 
                                                   market_data: Dict[str, Dict[str, float]],
                                                   positions: Dict[str, int], cash: float,
                                                   config: BacktestConfiguration, date: datetime,
                                                   backtest_id: str) -> List[Dict[str, Any]]:
        """
        Validate trading signals through comprehensive risk assessment.
        
        This method performs risk checks and filters signals based on risk parameters
        and historical risk patterns from memory.
        """
        if not signals:
            return []
        
        try:
            execution_pool_connection = self.agent_pools.get(AgentPoolType.EXECUTION_AGENT_POOL)
            if not execution_pool_connection or not execution_pool_connection.is_connected:
                logger.warning("⚠️ Execution agent pool not available, using basic risk assessment")
                return self._basic_risk_assessment(signals, market_data, positions, cash, config)
            
            # For demo purposes, simulate risk assessment
            logger.info(f"🛡️ Performing risk assessment for {len(signals)} signals")
            
            validated_signals = []
            
            for signal in signals:
                risk_score = self._calculate_signal_risk_score(signal, market_data, positions, cash)
                
                if risk_score < 0.7:  # Risk threshold
                    signal["risk_validated"] = True
                    signal["risk_score"] = risk_score
                    validated_signals.append(signal)
                else:
                    # Log rejected signal
                    await self._log_memory_event(
                        event_type=EventType.RISK_ASSESSMENT,
                        log_level=LogLevel.WARNING,
                        title=f"Signal rejected due to high risk",
                        content=f"Signal for {signal.get('symbol')} rejected: risk_score={risk_score}",
                        tags={"risk", "rejection", "signal"},
                        metadata={
                            "signal": signal,
                            "risk_score": risk_score,
                            "date": date.isoformat(),
                            "backtest_id": backtest_id
                        },
                        correlation_id=backtest_id
                    )
            
            logger.info(f"✅ Risk validation: {len(validated_signals)}/{len(signals)} signals approved")
            
            return validated_signals
            
        except Exception as e:
            logger.error(f"❌ Risk assessment failed: {e}")
            await self._log_memory_event(
                event_type=EventType.ERROR,
                log_level=LogLevel.ERROR,
                title=f"Risk assessment error for {date}",
                content=f"Error: {str(e)}",
                tags={"risk", "error", "assessment"},
                metadata={
                    "date": date.isoformat(),
                    "error_type": type(e).__name__,
                    "signals_count": len(signals),
                    "backtest_id": backtest_id
                },
                correlation_id=backtest_id
            )
            return signals  # Return original signals if risk assessment fails
    
    def _basic_risk_assessment(self, signals: List[Dict[str, Any]], market_data: Dict[str, Dict[str, float]],
                              positions: Dict[str, int], cash: float, config: BacktestConfiguration) -> List[Dict[str, Any]]:
        """Basic risk assessment when execution agent pool is not available"""
        validated_signals = []
        
        for signal in signals:
            # Simple risk checks
            symbol = signal.get("symbol")
            action = signal.get("action")
            
            if action == "buy" and cash > 1000:  # Minimum cash requirement
                signal["risk_validated"] = True
                signal["risk_score"] = 0.5
                validated_signals.append(signal)
            elif action == "sell" and positions.get(symbol, 0) > 0:
                signal["risk_validated"] = True
                signal["risk_score"] = 0.3
                validated_signals.append(signal)
        
        return validated_signals
    
    def _calculate_signal_risk_score(self, signal: Dict[str, Any], market_data: Dict[str, Dict[str, float]],
                                   positions: Dict[str, int], cash: float) -> float:
        """Calculate risk score for a signal (0 = low risk, 1 = high risk)"""
        risk_score = 0.0
        
        symbol = signal.get("symbol")
        action = signal.get("action")
        confidence = signal.get("confidence", 0.5)
        
        # Risk based on confidence
        risk_score += (1 - confidence) * 0.3
        
        # Risk based on position concentration
        if symbol in positions:
            position_value = positions[symbol] * market_data.get(symbol, {}).get("close", 0)
            total_portfolio = cash + sum(positions.get(s, 0) * market_data.get(s, {}).get("close", 0) for s in positions)
            if total_portfolio > 0:
                concentration = position_value / total_portfolio
                if concentration > 0.3:  # More than 30% in one position
                    risk_score += 0.4
        
        # Risk based on available cash
        if action == "buy" and cash < 10000:  # Low cash warning
            risk_score += 0.2
        
        return min(risk_score, 1.0)
    
    async def _execute_attributed_trades(self, signals: List[Dict[str, Any]], 
                                       market_data: Dict[str, Dict[str, float]],
                                       positions: Dict[str, int], cash: float,
                                       config: BacktestConfiguration, date: datetime,
                                       backtest_id: str, decision_chain: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute trades with full attribution tracking.
        
        This method executes trades while recording the complete decision chain
        for later attribution analysis and learning.
        """
        trades = []
        decision_record = {
            "date": date.isoformat(),
            "signals_processed": len(signals),
            "decisions": [],
            "attribution_id": str(uuid.uuid4()),
            "backtest_id": backtest_id
        }
        
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
                            trade = {
                                "symbol": symbol,
                                "action": "buy",
                                "shares": shares,
                                "price": current_price,
                                "total_cost": total_cost,
                                "confidence": confidence,
                                "timestamp": datetime.now().isoformat(),
                                "attribution": {
                                    "signal_source": signal.get("attribution", {}),
                                    "decision_id": decision_record["attribution_id"],
                                    "risk_score": signal.get("risk_score", 0),
                                    "reasoning": signal.get("reasoning", "")
                                }
                            }
                            trades.append(trade)
                            
                            # Record decision
                            decision_record["decisions"].append({
                                "symbol": symbol,
                                "action": "buy",
                                "shares": shares,
                                "reasoning": signal.get("reasoning", ""),
                                "attribution_chain": signal.get("attribution", {})
                            })
                            
                            # Log trade to memory
                            await self._log_memory_event(
                                event_type=EventType.TRANSACTION,
                                log_level=LogLevel.INFO,
                                title=f"Trade executed: BUY {symbol}",
                                content=f"Bought {shares} shares of {symbol} at ${current_price:.2f}",
                                tags={"trade", "buy", "execution", symbol.lower()},
                                metadata={
                                    "trade": trade,
                                    "decision_attribution": decision_record["attribution_id"],
                                    "backtest_id": backtest_id
                                },
                                correlation_id=backtest_id
                            )
                
                elif action == "sell" and symbol in positions and positions[symbol] > 0:
                    shares = positions[symbol]
                    total_proceeds = shares * current_price * (1 - config.commission_rate)
                    
                    trade = {
                        "symbol": symbol,
                        "action": "sell",
                        "shares": shares,
                        "price": current_price,
                        "total_proceeds": total_proceeds,
                        "confidence": confidence,
                        "timestamp": datetime.now().isoformat(),
                        "attribution": {
                            "signal_source": signal.get("attribution", {}),
                            "decision_id": decision_record["attribution_id"],
                            "risk_score": signal.get("risk_score", 0),
                            "reasoning": signal.get("reasoning", "")
                        }
                    }
                    trades.append(trade)
                    
                    # Record decision
                    decision_record["decisions"].append({
                        "symbol": symbol,
                        "action": "sell",
                        "shares": shares,
                        "reasoning": signal.get("reasoning", ""),
                        "attribution_chain": signal.get("attribution", {})
                    })
                    
                    # Log trade to memory
                    await self._log_memory_event(
                        event_type=EventType.TRANSACTION,
                        log_level=LogLevel.INFO,
                        title=f"Trade executed: SELL {symbol}",
                        content=f"Sold {shares} shares of {symbol} at ${current_price:.2f}",
                        tags={"trade", "sell", "execution", symbol.lower()},
                        metadata={
                            "trade": trade,
                            "decision_attribution": decision_record["attribution_id"],
                            "backtest_id": backtest_id
                        },
                        correlation_id=backtest_id
                    )
                    
            except Exception as e:
                logger.error(f"❌ Error executing trade for {signal}: {e}")
                continue
        
        # Add decision record to chain
        decision_chain.append(decision_record)
        
        logger.info(f"💰 Executed {len(trades)} trades with full attribution tracking")
        
        # Log decision chain to memory
        await self._log_memory_event(
            event_type=EventType.DECISION_CHAIN,
            log_level=LogLevel.INFO,
            title=f"Decision chain recorded for {date}",
            content=f"Recorded {len(decision_record['decisions'])} decisions",
            tags={"decision", "chain", "attribution"},
            metadata={
                "decision_record": decision_record,
                "backtest_id": backtest_id
            },
            correlation_id=backtest_id
        )
        
        return trades
    
    # ================================
    # Performance Analysis and Adaptive Learning
    # ================================
    
    async def _analyze_performance_and_adapt(self, portfolio_values: List[Dict], 
                                           trade_history: List[Dict], decision_chain: List[Dict],
                                           strategy_context: Dict[str, Any], backtest_id: str):
        """
        Analyze recent performance and adapt strategy parameters.
        
        This method performs attribution analysis and suggests improvements.
        """
        try:
            if len(portfolio_values) < 10:
                return
            
            # Calculate recent performance
            recent_values = [pv["value"] for pv in portfolio_values[-10:]]
            recent_returns = [(recent_values[i] - recent_values[i-1]) / recent_values[i-1] 
                            for i in range(1, len(recent_values))]
            
            avg_return = sum(recent_returns) / len(recent_returns) if recent_returns else 0
            volatility = (sum(r**2 for r in recent_returns) / len(recent_returns))**0.5 if recent_returns else 0
            
            # Analyze decision attribution
            recent_decisions = decision_chain[-5:] if len(decision_chain) >= 5 else decision_chain
            successful_decisions = 0
            total_decisions = 0
            
            for decision_record in recent_decisions:
                for decision in decision_record.get("decisions", []):
                    total_decisions += 1
                    # Simple success metric based on confidence and recent returns
                    if avg_return > 0:
                        successful_decisions += 1
            
            success_rate = successful_decisions / total_decisions if total_decisions > 0 else 0
            
            # Generate adaptation recommendations
            recommendations = []
            
            if avg_return < -0.01:  # Poor performance
                recommendations.append("Consider reducing position sizes")
                recommendations.append("Increase confidence threshold for signals")
            
            if volatility > 0.05:  # High volatility
                recommendations.append("Implement volatility-based position sizing")
                recommendations.append("Consider risk parity approach")
            
            if success_rate < 0.5:  # Low success rate
                recommendations.append("Review alpha signal generation parameters")
                recommendations.append("Enhance risk assessment criteria")
            
            # Log performance analysis to memory
            await self._log_memory_event(
                event_type=EventType.PERFORMANCE_ANALYSIS,
                log_level=LogLevel.INFO,
                title="Performance analysis and adaptation",
                content=f"Avg return: {avg_return:.4f}, Success rate: {success_rate:.2%}",
                tags={"performance", "analysis", "adaptation"},
                metadata={
                    "avg_return": avg_return,
                    "volatility": volatility,
                    "success_rate": success_rate,
                    "recommendations": recommendations,
                    "backtest_id": backtest_id
                },
                correlation_id=backtest_id
            )
            
            logger.info(f"📊 Performance analysis: Avg return: {avg_return:.4f}, Success rate: {success_rate:.2%}")
            
        except Exception as e:
            logger.error(f"❌ Performance analysis failed: {e}")
    
    def _calculate_recent_performance(self, recent_portfolio_values: List[Dict]) -> Dict[str, float]:
        """Calculate recent performance metrics"""
        if len(recent_portfolio_values) < 2:
            return {}
        
        values = [pv["value"] for pv in recent_portfolio_values]
        returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
        
        return {
            "avg_return": sum(returns) / len(returns) if returns else 0,
            "volatility": (sum(r**2 for r in returns) / len(returns))**0.5 if returns else 0,
            "total_return": (values[-1] - values[0]) / values[0] if values[0] > 0 else 0
        }
    
    async def _generate_comprehensive_memory_insights(self, backtest_id: str, trade_history: List[Dict],
                                                    decision_chain: List[Dict], portfolio_values: List[Dict],
                                                    performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate comprehensive insights from memory analysis"""
        if not self.memory_agent:
            return {}
        
        try:
            # Query memory for all events related to this backtest
            all_events = await self.memory_agent.retrieve_events(
                correlation_id=backtest_id,
                limit=1000
            )
            
            insights = {
                "total_memory_events": len(all_events),
                "backtest_id": backtest_id,
                "trade_analysis": {
                    "total_trades": len(trade_history),
                    "buy_trades": len([t for t in trade_history if t["action"] == "buy"]),
                    "sell_trades": len([t for t in trade_history if t["action"] == "sell"])
                },
                "decision_analysis": {
                    "total_decisions": len(decision_chain),
                    "avg_decisions_per_day": len(decision_chain) / len(portfolio_values) if portfolio_values else 0
                },
                "performance_summary": performance_metrics,
                "event_breakdown": {},
                "insights_generated": True,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            # Categorize events
            event_types = {}
            for event in all_events:
                event_type = event.get('event_type', 'unknown')
                event_types[event_type] = event_types.get(event_type, 0) + 1
            
            insights["event_breakdown"] = event_types
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Failed to generate comprehensive memory insights: {e}")
            return {}
    
    async def _generate_improvement_recommendations(self, backtest_id: str, performance_metrics: Dict[str, float],
                                                  decision_chain: List[Dict], memory_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Generate improvement recommendations based on analysis"""
        try:
            recommendations = {
                "agent_tuning": [],
                "dag_planning": [],
                "risk_management": [],
                "data_quality": [],
                "overall_score": 0.0
            }
            
            # Analyze performance
            total_return = performance_metrics.get('total_return', 0)
            sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
            max_drawdown = performance_metrics.get('max_drawdown', 0)
            
            score = 50  # Base score
            
            # Performance-based recommendations
            if total_return < 0:
                recommendations["agent_tuning"].append("Alpha agents need parameter adjustment")
                recommendations["dag_planning"].append("Review signal generation workflow")
                score -= 20
            else:
                score += 10
            
            if sharpe_ratio < 0.5:
                recommendations["risk_management"].append("Improve risk-adjusted returns")
                recommendations["agent_tuning"].append("Fine-tune confidence thresholds")
                score -= 10
            else:
                score += 5
            
            if max_drawdown > 0.2:
                recommendations["risk_management"].append("Implement stricter drawdown controls")
                recommendations["dag_planning"].append("Add volatility-based position sizing")
                score -= 15
            else:
                score += 5
            
            # Memory insights-based recommendations
            event_breakdown = memory_insights.get("event_breakdown", {})
            error_count = event_breakdown.get("ERROR", 0)
            
            if error_count > 10:
                recommendations["data_quality"].append("Address data quality issues")
                recommendations["agent_tuning"].append("Improve error handling in agent pools")
                score -= 10
            
            recommendations["overall_score"] = max(0, min(100, score))
            
            # Log recommendations to memory
            await self._log_memory_event(
                event_type=EventType.OPTIMIZATION,
                log_level=LogLevel.INFO,
                title="Improvement recommendations generated",
                content=f"Generated recommendations with overall score: {recommendations['overall_score']}",
                tags={"recommendations", "improvement", "optimization"},
                metadata={
                    "recommendations": recommendations,
                    "performance_metrics": performance_metrics,
                    "backtest_id": backtest_id
                },
                correlation_id=backtest_id
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Failed to generate improvement recommendations: {e}")
            return {}
