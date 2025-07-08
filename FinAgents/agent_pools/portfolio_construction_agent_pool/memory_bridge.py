"""
Portfolio Construction Agent Pool Memory Bridge

This module provides comprehensive integration between the Portfolio Construction 
Agent Pool and the External Memory Agent system, enabling retrieval of alpha signals,
risk metrics, and transaction costs while storing portfolio construction results.

The memory bridge supports:
- Multi-agent event stream integration (alpha, risk, transaction cost)
- Portfolio optimization result storage and retrieval
- Performance tracking and backtesting support
- Real-time portfolio monitoring and rebalancing signals
- Cross-strategy correlation analysis for portfolio construction

Academic Framework:
This implementation follows modern portfolio theory and incorporates advanced
optimization techniques for quantitative portfolio management applications.

Author: Jifeng Li
License: openMDW
Created: 2025-06-30
"""

import logging
import asyncio
import json
import uuid
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

# Configure logging with academic formatting standards
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Enumeration of portfolio optimization types"""
    MEAN_VARIANCE = "mean_variance"
    BLACK_LITTERMAN = "black_litterman"
    RISK_PARITY = "risk_parity"
    MAXIMUM_DIVERSIFICATION = "maximum_diversification"
    MINIMUM_VARIANCE = "minimum_variance"
    EQUAL_WEIGHT = "equal_weight"
    FACTOR_BASED = "factor_based"
    ROBUST_OPTIMIZATION = "robust_optimization"


class PortfolioStatus(Enum):
    """Portfolio lifecycle status enumeration"""
    PROPOSED = "proposed"
    OPTIMIZING = "optimizing"
    VALIDATED = "validated"
    IMPLEMENTED = "implemented"
    ACTIVE = "active"
    REBALANCING = "rebalancing"
    CLOSED = "closed"
    ARCHIVED = "archived"


@dataclass
class PortfolioPosition:
    """Individual portfolio position representation"""
    asset_id: str
    asset_name: str
    asset_type: str  # equity, bond, commodity, currency, alternative
    target_weight: float  # Target allocation percentage (0-1)
    current_weight: float  # Current allocation percentage (0-1)
    target_quantity: float
    current_quantity: float
    target_value: float
    current_value: float
    currency: str = "USD"
    sector: Optional[str] = None
    region: Optional[str] = None
    expected_return: Optional[float] = None
    volatility: Optional[float] = None
    beta: Optional[float] = None


@dataclass
class PortfolioRecord:
    """Comprehensive portfolio construction record"""
    portfolio_id: str
    portfolio_name: str
    portfolio_type: OptimizationType
    status: PortfolioStatus
    positions: List[PortfolioPosition]
    total_value: float
    benchmark: str
    objective: str  # Description of investment objective
    constraints: Dict[str, Any]  # Investment constraints
    risk_budget: float  # Maximum acceptable risk level
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    tracking_error: Optional[float] = None
    information_ratio: Optional[float] = None
    maximum_drawdown: Optional[float] = None
    creation_timestamp: datetime = None
    last_rebalance: datetime = None
    rebalancing_frequency: str = "monthly"  # daily, weekly, monthly, quarterly
    portfolio_manager: str = "portfolio_construction_agent_pool"
    strategy_source: str = "multi_agent_optimization"
    
    def __post_init__(self):
        if self.creation_timestamp is None:
            self.creation_timestamp = datetime.now(timezone.utc)


@dataclass
class OptimizationResult:
    """Portfolio optimization result and analytics"""
    optimization_id: str
    portfolio_id: str
    optimization_type: OptimizationType
    input_signals: Dict[str, Any]  # Alpha, risk, transaction cost inputs
    optimal_weights: Dict[str, float]  # Asset -> weight mapping
    expected_metrics: Dict[str, float]  # Expected return, vol, Sharpe, etc.
    optimization_status: str  # success, failed, partial
    objective_value: float  # Optimization objective function value
    constraints_satisfied: bool
    optimization_time_seconds: float
    alpha_signals_used: List[str]  # Alpha signal IDs incorporated
    risk_metrics_used: List[str]  # Risk analysis IDs incorporated
    transaction_costs_used: List[str]  # Transaction cost analysis IDs
    timestamp: datetime = None
    convergence_details: Optional[Dict[str, Any]] = None
    sensitivity_analysis: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class PortfolioMetrics:
    """Real-time portfolio performance metrics"""
    portfolio_id: str
    timestamp: datetime
    total_return: float
    benchmark_return: float
    active_return: float  # Portfolio return - benchmark return
    volatility: float
    benchmark_volatility: float
    tracking_error: float
    sharpe_ratio: float
    information_ratio: float
    maximum_drawdown: float
    value_at_risk_95: float
    conditional_var_95: float
    beta: float
    alpha: float  # Jensen's alpha
    turnover: float  # Portfolio turnover rate
    transaction_costs: float
    attribution_analysis: Optional[Dict[str, float]] = None
    risk_attribution: Optional[Dict[str, float]] = None


class PortfolioConstructionMemoryBridge:
    """
    Advanced memory bridge for Portfolio Construction Agent Pool integration.
    
    This bridge orchestrates multi-agent event streams to support sophisticated
    portfolio construction and optimization workflows.
    """
    
    def __init__(self, 
                 external_memory_config: Optional[Dict[str, Any]] = None,
                 enable_real_time_monitoring: bool = True,
                 optimization_cache_enabled: bool = True,
                 performance_analytics_enabled: bool = True):
        """
        Initialize Portfolio Construction Memory Bridge.
        
        Args:
            external_memory_config: External memory agent configuration
            enable_real_time_monitoring: Enable real-time portfolio monitoring
            optimization_cache_enabled: Enable optimization result caching
            performance_analytics_enabled: Enable comprehensive performance analytics
        """
        self.namespace = "portfolio_construction_agent_pool"
        self.external_memory_config = external_memory_config or {}
        self.enable_real_time_monitoring = enable_real_time_monitoring
        self.optimization_cache_enabled = optimization_cache_enabled
        self.performance_analytics_enabled = performance_analytics_enabled
        
        # Initialize external memory agent connection
        self.external_memory_agent = None
        self.session_id = f"portfolio_session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        # Local caching for performance optimization
        self.local_portfolio_cache: Dict[str, PortfolioRecord] = {}
        self.local_optimization_cache: Dict[str, OptimizationResult] = {}
        self.alpha_signals_cache: Dict[str, Any] = {}
        self.risk_metrics_cache: Dict[str, Any] = {}
        self.transaction_cost_cache: Dict[str, Any] = {}
        
        # Bridge statistics for monitoring
        self.bridge_statistics = {
            'portfolios_created': 0,
            'optimizations_performed': 0,
            'alpha_signals_retrieved': 0,
            'risk_metrics_retrieved': 0,
            'transaction_costs_retrieved': 0,
            'last_activity': datetime.now(timezone.utc)
        }
        
        logger.info(f"Portfolio Construction Memory Bridge initialized with session: {self.session_id}")
    
    async def initialize(self) -> None:
        """
        Initialize external memory agent connection and verify system readiness.
        """
        try:
            # Import external memory agent (lazy loading for dependency management)
            from ...memory.external_memory_agent import (
                ExternalMemoryAgent, EventType, LogLevel, SQLiteStorageBackend
            )
            
            # Initialize external memory agent with optimized configuration
            memory_db_path = Path("./FinAgents/memory/portfolio_construction_memory_storage") / f"portfolio_construction_memory_{self.session_id}.db"
            memory_db_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            
            storage_backend = SQLiteStorageBackend(
                db_path=str(memory_db_path)
            )
            
            self.external_memory_agent = ExternalMemoryAgent(
                storage_backend=storage_backend,
                enable_real_time_hooks=self.enable_real_time_monitoring,
                max_batch_size=1000
            )
            
            await self.external_memory_agent.initialize()
            
            # Log initialization event
            await self._log_system_event(
                event_type=EventType.SYSTEM,
                log_level=LogLevel.INFO,
                title="Portfolio Construction Memory Bridge Initialized",
                content=f"Successfully initialized memory bridge with session {self.session_id}",
                metadata={
                    "real_time_monitoring": self.enable_real_time_monitoring,
                    "optimization_cache_enabled": self.optimization_cache_enabled,
                    "performance_analytics_enabled": self.performance_analytics_enabled
                }
            )
            
            logger.info("External memory agent successfully initialized for Portfolio Construction Agent Pool")
            
        except ImportError as e:
            logger.warning(f"External memory agent not available: {e}")
            self.external_memory_agent = None
        except Exception as e:
            logger.error(f"Failed to initialize portfolio construction memory bridge: {e}")
            self.external_memory_agent = None
            raise
    
    async def close(self) -> None:
        """
        Close the memory bridge and clean up resources.
        """
        try:
            if self.external_memory_agent:
                await self.external_memory_agent.close()
                self.external_memory_agent = None
            
            # Clear local caches
            self.local_portfolio_cache.clear()
            self.local_optimization_cache.clear()
            self.alpha_signals_cache.clear()
            self.risk_metrics_cache.clear()
            self.transaction_cost_cache.clear()
            
            logger.info("Portfolio Construction Memory Bridge closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing portfolio construction memory bridge: {e}")
    
    async def retrieve_alpha_signals(self, 
                                   filters: Optional[Dict[str, Any]] = None,
                                   time_range: Optional[timedelta] = None,
                                   limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve alpha signals from memory for portfolio construction.
        
        Args:
            filters: Filter criteria for alpha signals
            time_range: Time window for signal retrieval
            limit: Maximum number of signals to retrieve
            
        Returns:
            List[Dict[str, Any]]: Filtered alpha signals
        """
        try:
            if time_range is None:
                time_range = timedelta(hours=24)
            
            # Query alpha agent pool events from memory
            query_filters = {
                "namespace": "alpha_agent_pool",
                "event_type": "SIGNAL_GENERATED",
                "time_range": time_range
            }
            
            # Add additional filters if provided
            if filters:
                query_filters.update(filters)
            
            if self.external_memory_agent:
                alpha_events = await self.external_memory_agent.query_events(query_filters)
                
                # Process and cache alpha signals
                alpha_signals = []
                for event in alpha_events[:limit]:
                    signal_data = event.get("content", {})
                    signal_id = event.get("event_id")
                    
                    # Cache for future use
                    self.alpha_signals_cache[signal_id] = signal_data
                    alpha_signals.append(signal_data)
                
                self.bridge_statistics['alpha_signals_retrieved'] += len(alpha_signals)
                
                logger.info(f"Retrieved {len(alpha_signals)} alpha signals from memory")
                return alpha_signals
            else:
                logger.warning("External memory agent not available for alpha signal retrieval")
                return []
                
        except Exception as e:
            logger.error(f"Failed to retrieve alpha signals: {e}")
            return []
    
    async def retrieve_risk_metrics(self, 
                                  filters: Optional[Dict[str, Any]] = None,
                                  time_range: Optional[timedelta] = None,
                                  limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve risk analysis results from memory for portfolio construction.
        
        Args:
            filters: Filter criteria for risk metrics
            time_range: Time window for risk metric retrieval
            limit: Maximum number of risk analyses to retrieve
            
        Returns:
            List[Dict[str, Any]]: Filtered risk analysis results
        """
        try:
            if time_range is None:
                time_range = timedelta(hours=24)
            
            # Query risk agent pool events from memory
            query_filters = {
                "namespace": "risk_agent_pool",
                "event_type": "ANALYSIS_COMPLETED",
                "time_range": time_range
            }
            
            # Add additional filters if provided
            if filters:
                query_filters.update(filters)
            
            if self.external_memory_agent:
                risk_events = await self.external_memory_agent.query_events(query_filters)
                
                # Process and cache risk metrics
                risk_metrics = []
                for event in risk_events[:limit]:
                    risk_data = event.get("content", {})
                    risk_id = event.get("event_id")
                    
                    # Cache for future use
                    self.risk_metrics_cache[risk_id] = risk_data
                    risk_metrics.append(risk_data)
                
                self.bridge_statistics['risk_metrics_retrieved'] += len(risk_metrics)
                
                logger.info(f"Retrieved {len(risk_metrics)} risk analysis results from memory")
                return risk_metrics
            else:
                logger.warning("External memory agent not available for risk metrics retrieval")
                return []
                
        except Exception as e:
            logger.error(f"Failed to retrieve risk metrics: {e}")
            return []
    
    async def retrieve_transaction_costs(self, 
                                       filters: Optional[Dict[str, Any]] = None,
                                       time_range: Optional[timedelta] = None,
                                       limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve transaction cost analysis from memory for portfolio construction.
        
        Args:
            filters: Filter criteria for transaction cost data
            time_range: Time window for transaction cost retrieval
            limit: Maximum number of transaction cost analyses to retrieve
            
        Returns:
            List[Dict[str, Any]]: Filtered transaction cost data
        """
        try:
            if time_range is None:
                time_range = timedelta(hours=24)
            
            # Query transaction cost agent pool events from memory
            query_filters = {
                "namespace": "transaction_cost_agent_pool",
                "event_type": "COST_ANALYSIS_COMPLETED",
                "time_range": time_range
            }
            
            # Add additional filters if provided
            if filters:
                query_filters.update(filters)
            
            if self.external_memory_agent:
                cost_events = await self.external_memory_agent.query_events(query_filters)
                
                # Process and cache transaction costs
                transaction_costs = []
                for event in cost_events[:limit]:
                    cost_data = event.get("content", {})
                    cost_id = event.get("event_id")
                    
                    # Cache for future use
                    self.transaction_cost_cache[cost_id] = cost_data
                    transaction_costs.append(cost_data)
                
                self.bridge_statistics['transaction_costs_retrieved'] += len(transaction_costs)
                
                logger.info(f"Retrieved {len(transaction_costs)} transaction cost analyses from memory")
                return transaction_costs
            else:
                logger.warning("External memory agent not available for transaction cost retrieval")
                return []
                
        except Exception as e:
            logger.error(f"Failed to retrieve transaction costs: {e}")
            return []
    
    async def store_portfolio_record(self, portfolio_record: PortfolioRecord) -> str:
        """
        Store portfolio construction record in memory system.
        
        Args:
            portfolio_record: Complete portfolio record to store
            
        Returns:
            str: Storage identifier for the portfolio record
        """
        try:
            storage_key = f"portfolio:{portfolio_record.portfolio_id}:{portfolio_record.creation_timestamp.isoformat()}"
            
            # Store in local cache for quick access
            self.local_portfolio_cache[storage_key] = portfolio_record
            
            # Store in external memory if available
            if self.external_memory_agent:
                from FinAgents.memory.external_memory_agent import MemoryEvent, EventType, LogLevel
                
                event = MemoryEvent(
                    event_id=storage_key,
                    timestamp=portfolio_record.timestamp,
                    event_type=EventType.PORTFOLIO_UPDATE,
                    log_level=LogLevel.INFO,
                    source_agent_pool="portfolio_construction_agent_pool",
                    source_agent_id=self.pool_id,
                    title=f"Portfolio Record Stored: {portfolio_record.portfolio_id}",
                    content=json.dumps(asdict(portfolio_record), default=str),
                    tags={"portfolio", "construction", portfolio_record.portfolio_type.value},
                    metadata={"portfolio_id": portfolio_record.portfolio_id, "type": "portfolio_record"},
                    session_id=self.session_id
                )
                
                await self.external_memory_agent.store_event(event)
                
                # Log portfolio creation event
                await self._log_portfolio_event(
                    portfolio_record=portfolio_record,
                    event_type="PORTFOLIO_CREATED"
                )
            
            # Update statistics
            self.bridge_statistics['portfolios_created'] += 1
            self.bridge_statistics['last_activity'] = datetime.now(timezone.utc)
            
            logger.info(f"Portfolio stored: {portfolio_record.portfolio_name} "
                       f"(Value: ${portfolio_record.total_value:,.2f}, "
                       f"Positions: {len(portfolio_record.positions)})")
            
            return storage_key
            
        except Exception as e:
            logger.error(f"Failed to store portfolio record: {e}")
            raise
    
    async def store_optimization_result(self, optimization_result: OptimizationResult) -> str:
        """
        Store portfolio optimization result in memory system.
        
        Args:
            optimization_result: Optimization result to store
            
        Returns:
            str: Storage identifier for the optimization result
        """
        try:
            storage_key = f"optimization:{optimization_result.optimization_id}:{optimization_result.timestamp.isoformat()}"
            
            # Store in local cache for quick access
            if self.optimization_cache_enabled:
                self.local_optimization_cache[storage_key] = optimization_result
            
            # Store in external memory if available
            if self.external_memory_agent:
                from FinAgents.memory.external_memory_agent import MemoryEvent, EventType, LogLevel
                
                event = MemoryEvent(
                    event_id=storage_key,
                    timestamp=optimization_result.timestamp,
                    event_type=EventType.OPTIMIZATION,
                    log_level=LogLevel.INFO,
                    source_agent_pool="portfolio_construction_agent_pool",
                    source_agent_id=self.pool_id,
                    title=f"Optimization Result Stored: {optimization_result.optimization_id}",
                    content=json.dumps(asdict(optimization_result), default=str),
                    tags={"optimization", optimization_result.optimization_type.value},
                    metadata={"optimization_id": optimization_result.optimization_id, "type": "optimization_result"},
                    session_id=self.session_id
                )
                
                await self.external_memory_agent.store_event(event)
                
                # Log optimization event
                await self._log_optimization_event(
                    optimization_result=optimization_result,
                    event_type="OPTIMIZATION_COMPLETED"
                )
            
            # Update statistics
            self.bridge_statistics['optimizations_performed'] += 1
            self.bridge_statistics['last_activity'] = datetime.now(timezone.utc)
            
            logger.info(f"Optimization result stored: {optimization_result.optimization_type.value} "
                       f"(Status: {optimization_result.optimization_status}, "
                       f"Objective: {optimization_result.objective_value:.6f})")
            
            return storage_key
            
        except Exception as e:
            logger.error(f"Failed to store optimization result: {e}")
            raise
    
    async def store_portfolio_metrics(self, portfolio_metrics: PortfolioMetrics) -> str:
        """
        Store portfolio performance metrics in memory system.
        
        Args:
            portfolio_metrics: Portfolio performance metrics to store
            
        Returns:
            str: Storage identifier for the metrics
        """
        try:
            storage_key = f"metrics:{portfolio_metrics.portfolio_id}:{portfolio_metrics.timestamp.isoformat()}"
            
            # Store in external memory if available
            if self.external_memory_agent:
                from FinAgents.memory.external_memory_agent import MemoryEvent, EventType, LogLevel
                
                event = MemoryEvent(
                    event_id=storage_key,
                    timestamp=portfolio_metrics.timestamp,
                    event_type=EventType.INFO,
                    log_level=LogLevel.INFO,
                    source_agent_pool="portfolio_construction_agent_pool",
                    source_agent_id=self.pool_id,
                    title=f"Portfolio Metrics Stored: {portfolio_metrics.portfolio_id}",
                    content=json.dumps(asdict(portfolio_metrics), default=str),
                    tags={"metrics", "performance"},
                    metadata={"portfolio_id": portfolio_metrics.portfolio_id, "type": "portfolio_metrics"},
                    session_id=self.session_id
                )
                
                await self.external_memory_agent.store_event(event)
                
                # Log performance metrics event
                await self._log_performance_event(
                    portfolio_metrics=portfolio_metrics,
                    event_type="PERFORMANCE_UPDATED"
                )
            
            self.bridge_statistics['last_activity'] = datetime.now(timezone.utc)
            
            logger.info(f"Portfolio metrics stored: {portfolio_metrics.portfolio_id} "
                       f"(Return: {portfolio_metrics.total_return:.4f}, "
                       f"Sharpe: {portfolio_metrics.sharpe_ratio:.3f})")
            
            return storage_key
            
        except Exception as e:
            logger.error(f"Failed to store portfolio metrics: {e}")
            raise
    
    async def retrieve_multi_agent_inputs(self, 
                                        investment_universe: List[str],
                                        time_horizon: str = "daily") -> Dict[str, Any]:
        """
        Retrieve comprehensive multi-agent inputs for portfolio construction.
        
        Args:
            investment_universe: List of assets to consider
            time_horizon: Time horizon for analysis (daily, weekly, monthly)
            
        Returns:
            Dict[str, Any]: Comprehensive input data from all agent pools
        """
        try:
            # Define time ranges based on horizon
            time_ranges = {
                "daily": timedelta(days=1),
                "weekly": timedelta(days=7),
                "monthly": timedelta(days=30)
            }
            
            time_range = time_ranges.get(time_horizon, timedelta(days=1))
            
            # Retrieve data from all agent pools concurrently
            alpha_task = self.retrieve_alpha_signals(
                filters={"symbols": investment_universe},
                time_range=time_range
            )
            
            risk_task = self.retrieve_risk_metrics(
                filters={"assets": investment_universe},
                time_range=time_range
            )
            
            cost_task = self.retrieve_transaction_costs(
                filters={"assets": investment_universe},
                time_range=time_range
            )
            
            # Execute all retrievals concurrently
            alpha_signals, risk_metrics, transaction_costs = await asyncio.gather(
                alpha_task, risk_task, cost_task
            )
            
            # Compile comprehensive input data
            multi_agent_inputs = {
                "alpha_signals": alpha_signals,
                "risk_metrics": risk_metrics,
                "transaction_costs": transaction_costs,
                "investment_universe": investment_universe,
                "time_horizon": time_horizon,
                "retrieval_timestamp": datetime.now(timezone.utc),
                "data_summary": {
                    "alpha_signals_count": len(alpha_signals),
                    "risk_analyses_count": len(risk_metrics),
                    "transaction_cost_analyses_count": len(transaction_costs)
                }
            }
            
            logger.info(f"Retrieved multi-agent inputs: {len(alpha_signals)} alpha signals, "
                       f"{len(risk_metrics)} risk analyses, {len(transaction_costs)} cost analyses")
            
            return multi_agent_inputs
            
        except Exception as e:
            logger.error(f"Failed to retrieve multi-agent inputs: {e}")
            raise
    
    async def get_portfolio_performance_analytics(self, 
                                                portfolio_id: str,
                                                time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Generate comprehensive portfolio performance analytics.
        
        Args:
            portfolio_id: Portfolio identifier
            time_range: Time range for analytics (default: last 30 days)
            
        Returns:
            Dict[str, Any]: Comprehensive performance analytics
        """
        try:
            if time_range is None:
                time_range = timedelta(days=30)
            
            if not self.performance_analytics_enabled:
                logger.warning("Performance analytics disabled")
                return {}
            
            # Retrieve historical portfolio metrics
            query_filters = {
                "namespace": self.namespace,
                "portfolio_id": portfolio_id,
                "time_range": time_range
            }
            
            historical_metrics = []
            if self.external_memory_agent:
                events = await self.external_memory_agent.query_events(query_filters)
                historical_metrics = [event.get("content", {}) for event in events]
            
            # Calculate analytics
            analytics = {
                "portfolio_id": portfolio_id,
                "analysis_period": time_range.days,
                "metrics_count": len(historical_metrics),
                "performance_summary": self._calculate_performance_summary(historical_metrics),
                "risk_analytics": self._calculate_risk_analytics(historical_metrics),
                "attribution_analysis": self._calculate_attribution_analysis(historical_metrics),
                "benchmark_comparison": self._calculate_benchmark_comparison(historical_metrics),
                "generated_at": datetime.now(timezone.utc)
            }
            
            logger.info(f"Generated performance analytics for portfolio: {portfolio_id}")
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to generate performance analytics: {e}")
            return {}
    
    async def _log_system_event(self, event_type, log_level, title: str, content: str, metadata: Dict[str, Any] = None):
        """Log system events to external memory"""
        if self.external_memory_agent:
            try:
                await self.external_memory_agent.log_event(
                    event_type=event_type,
                    log_level=log_level,
                    source_agent_pool="portfolio_construction_agent_pool",
                    source_agent_id=self.pool_id,
                    title=title,
                    content=content,
                    metadata={
                        "namespace": self.namespace,
                        "session_id": self.session_id,
                        **(metadata or {})
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to log system event: {e}")
    
    async def _log_portfolio_event(self, portfolio_record: PortfolioRecord, event_type: str):
        """Log portfolio-related events to external memory"""
        if self.external_memory_agent:
            try:
                from ...memory.external_memory_agent import EventType, LogLevel
                
                await self.external_memory_agent.log_event(
                    event_type=EventType.BUSINESS,
                    log_level=LogLevel.INFO,
                    source_agent_pool="portfolio_construction_agent_pool",
                    source_agent_id=self.pool_id,
                    title=f"Portfolio {event_type}",
                    content=f"Portfolio {portfolio_record.portfolio_name} ({portfolio_record.portfolio_type.value})",
                    metadata={
                        "namespace": self.namespace,
                        "portfolio_id": portfolio_record.portfolio_id,
                        "portfolio_type": portfolio_record.portfolio_type.value,
                        "total_value": portfolio_record.total_value,
                        "positions_count": len(portfolio_record.positions),
                        "event_type": event_type
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to log portfolio event: {e}")
    
    async def _log_optimization_event(self, optimization_result: OptimizationResult, event_type: str):
        """Log optimization-related events to external memory"""
        if self.external_memory_agent:
            try:
                from ...memory.external_memory_agent import EventType, LogLevel
                
                await self.external_memory_agent.log_event(
                    event_type=EventType.BUSINESS,
                    log_level=LogLevel.INFO,
                    source_agent_pool="portfolio_construction_agent_pool",
                    source_agent_id=self.pool_id,
                    title=f"Optimization {event_type}",
                    content=f"Portfolio optimization completed: {optimization_result.optimization_type.value}",
                    metadata={
                        "namespace": self.namespace,
                        "optimization_id": optimization_result.optimization_id,
                        "portfolio_id": optimization_result.portfolio_id,
                        "optimization_type": optimization_result.optimization_type.value,
                        "optimization_status": optimization_result.optimization_status,
                        "objective_value": optimization_result.objective_value,
                        "event_type": event_type
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to log optimization event: {e}")
    
    async def _log_performance_event(self, portfolio_metrics: PortfolioMetrics, event_type: str):
        """Log performance-related events to external memory"""
        if self.external_memory_agent:
            try:
                from ...memory.external_memory_agent import EventType, LogLevel
                
                await self.external_memory_agent.log_event(
                    event_type=EventType.BUSINESS,
                    log_level=LogLevel.INFO,
                    source_agent_pool="portfolio_construction_agent_pool",
                    source_agent_id=self.pool_id,
                    title=f"Performance {event_type}",
                    content=f"Portfolio performance metrics updated",
                    metadata={
                        "namespace": self.namespace,
                        "portfolio_id": portfolio_metrics.portfolio_id,
                        "total_return": portfolio_metrics.total_return,
                        "sharpe_ratio": portfolio_metrics.sharpe_ratio,
                        "information_ratio": portfolio_metrics.information_ratio,
                        "event_type": event_type
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to log performance event: {e}")
    
    def _calculate_performance_summary(self, historical_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate portfolio performance summary statistics"""
        if not historical_metrics:
            return {}
        
        returns = [m.get("total_return", 0) for m in historical_metrics if "total_return" in m]
        
        if not returns:
            return {}
        
        import statistics
        
        return {
            "average_return": statistics.mean(returns),
            "volatility": statistics.stdev(returns) if len(returns) > 1 else 0,
            "best_return": max(returns),
            "worst_return": min(returns),
            "total_observations": len(returns)
        }
    
    def _calculate_risk_analytics(self, historical_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate portfolio risk analytics"""
        if not historical_metrics:
            return {}
        
        # Extract risk metrics
        vars = [m.get("value_at_risk_95", 0) for m in historical_metrics if "value_at_risk_95" in m]
        tracking_errors = [m.get("tracking_error", 0) for m in historical_metrics if "tracking_error" in m]
        max_drawdowns = [m.get("maximum_drawdown", 0) for m in historical_metrics if "maximum_drawdown" in m]
        
        import statistics
        
        return {
            "average_var_95": statistics.mean(vars) if vars else 0,
            "average_tracking_error": statistics.mean(tracking_errors) if tracking_errors else 0,
            "maximum_drawdown": max(max_drawdowns) if max_drawdowns else 0,
            "risk_observations": len(vars)
        }
    
    def _calculate_attribution_analysis(self, historical_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate portfolio return attribution analysis"""
        if not historical_metrics:
            return {}
        
        # This would typically involve more sophisticated attribution calculations
        # For now, return basic attribution structure
        return {
            "security_selection_effect": 0.0,
            "asset_allocation_effect": 0.0,
            "interaction_effect": 0.0,
            "currency_effect": 0.0,
            "analysis_available": len(historical_metrics) > 0
        }
    
    def _calculate_benchmark_comparison(self, historical_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate portfolio benchmark comparison metrics"""
        if not historical_metrics:
            return {}
        
        portfolio_returns = [m.get("total_return", 0) for m in historical_metrics]
        benchmark_returns = [m.get("benchmark_return", 0) for m in historical_metrics]
        
        if not portfolio_returns or not benchmark_returns:
            return {}
        
        active_returns = [p - b for p, b in zip(portfolio_returns, benchmark_returns)]
        
        import statistics
        
        return {
            "average_active_return": statistics.mean(active_returns),
            "active_return_volatility": statistics.stdev(active_returns) if len(active_returns) > 1 else 0,
            "hit_rate": sum(1 for ar in active_returns if ar > 0) / len(active_returns),
            "comparison_periods": len(active_returns)
        }


# Convenience functions for creating portfolio construction records
def create_portfolio_record(portfolio_name: str, optimization_type: OptimizationType,
                          positions: List[PortfolioPosition], total_value: float,
                          benchmark: str, objective: str, **kwargs) -> PortfolioRecord:
    """
    Convenience function to create portfolio records with proper validation.
    
    Args:
        portfolio_name: Name of the portfolio
        optimization_type: Type of optimization used
        positions: List of portfolio positions
        total_value: Total portfolio value
        benchmark: Benchmark identifier
        objective: Investment objective description
        **kwargs: Additional optional parameters
        
    Returns:
        PortfolioRecord: Validated portfolio record
    """
    # Calculate portfolio metrics
    total_weight = sum(pos.target_weight for pos in positions)
    expected_return = sum(pos.target_weight * (pos.expected_return or 0) for pos in positions)
    expected_volatility = kwargs.get('expected_volatility', 0.15)  # Default volatility
    
    # Calculate Sharpe ratio estimate
    risk_free_rate = kwargs.get('risk_free_rate', 0.02)
    sharpe_ratio = (expected_return - risk_free_rate) / max(expected_volatility, 0.001)
    
    return PortfolioRecord(
        portfolio_id=kwargs.get('portfolio_id', str(uuid.uuid4())),
        portfolio_name=portfolio_name,
        portfolio_type=optimization_type,
        status=kwargs.get('status', PortfolioStatus.PROPOSED),
        positions=positions,
        total_value=total_value,
        benchmark=benchmark,
        objective=objective,
        constraints=kwargs.get('constraints', {}),
        risk_budget=kwargs.get('risk_budget', 0.20),
        expected_return=expected_return,
        expected_volatility=expected_volatility,
        sharpe_ratio=sharpe_ratio,
        tracking_error=kwargs.get('tracking_error'),
        information_ratio=kwargs.get('information_ratio'),
        maximum_drawdown=kwargs.get('maximum_drawdown'),
        rebalancing_frequency=kwargs.get('rebalancing_frequency', 'monthly'),
        portfolio_manager=kwargs.get('portfolio_manager', 'portfolio_construction_agent_pool'),
        strategy_source=kwargs.get('strategy_source', 'multi_agent_optimization')
    )


def create_optimization_result(optimization_type: OptimizationType, portfolio_id: str,
                             optimal_weights: Dict[str, float], input_signals: Dict[str, Any],
                             **kwargs) -> OptimizationResult:
    """
    Convenience function to create optimization results with proper validation.
    
    Args:
        optimization_type: Type of optimization performed
        portfolio_id: Portfolio identifier
        optimal_weights: Optimal asset weights
        input_signals: Input signals used for optimization
        **kwargs: Additional optional parameters
        
    Returns:
        OptimizationResult: Validated optimization result
    """
    return OptimizationResult(
        optimization_id=kwargs.get('optimization_id', str(uuid.uuid4())),
        portfolio_id=portfolio_id,
        optimization_type=optimization_type,
        input_signals=input_signals,
        optimal_weights=optimal_weights,
        expected_metrics=kwargs.get('expected_metrics', {}),
        optimization_status=kwargs.get('optimization_status', 'success'),
        objective_value=kwargs.get('objective_value', 0.0),
        constraints_satisfied=kwargs.get('constraints_satisfied', True),
        optimization_time_seconds=kwargs.get('optimization_time_seconds', 0.0),
        alpha_signals_used=kwargs.get('alpha_signals_used', []),
        risk_metrics_used=kwargs.get('risk_metrics_used', []),
        transaction_costs_used=kwargs.get('transaction_costs_used', []),
        convergence_details=kwargs.get('convergence_details'),
        sensitivity_analysis=kwargs.get('sensitivity_analysis')
    )


def create_portfolio_metrics_record(portfolio_id: str, total_return: float,
                                  benchmark_return: float, **kwargs) -> PortfolioMetrics:
    """
    Convenience function to create portfolio metrics records.
    
    Args:
        portfolio_id: Portfolio identifier
        total_return: Portfolio total return
        benchmark_return: Benchmark return
        **kwargs: Additional optional metrics
        
    Returns:
        PortfolioMetrics: Validated portfolio metrics record
    """
    return PortfolioMetrics(
        portfolio_id=portfolio_id,
        timestamp=kwargs.get('timestamp', datetime.now(timezone.utc)),
        total_return=total_return,
        benchmark_return=benchmark_return,
        active_return=total_return - benchmark_return,
        volatility=kwargs.get('volatility', 0.15),
        benchmark_volatility=kwargs.get('benchmark_volatility', 0.12),
        tracking_error=kwargs.get('tracking_error', 0.05),
        sharpe_ratio=kwargs.get('sharpe_ratio', (total_return - 0.02) / 0.15),
        information_ratio=kwargs.get('information_ratio', (total_return - benchmark_return) / 0.05),
        maximum_drawdown=kwargs.get('maximum_drawdown', 0.08),
        value_at_risk_95=kwargs.get('value_at_risk_95', 0.025),
        conditional_var_95=kwargs.get('conditional_var_95', 0.035),
        beta=kwargs.get('beta', 1.0),
        alpha=kwargs.get('alpha', total_return - benchmark_return),
        turnover=kwargs.get('turnover', 0.50),
        transaction_costs=kwargs.get('transaction_costs', 0.001),
        attribution_analysis=kwargs.get('attribution_analysis'),
        risk_attribution=kwargs.get('risk_attribution')
    )
