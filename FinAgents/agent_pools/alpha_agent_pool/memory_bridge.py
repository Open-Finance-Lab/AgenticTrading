"""
Alpha Agent Pool Memory Bridge - Integration with External Memory Agent

This module provides comprehensive integration between the Alpha Agent Pool and the 
External Memory Agent system, enabling storage, retrieval, and management of alpha 
generation strategies, signal events, performance metrics, and strategic patterns.

The memory bridge supports:
- Alpha signal generation and storage
- Strategy performance tracking and analysis
- Historical pattern recognition and retrieval
- Cross-strategy correlation analysis
- Real-time signal streaming and event logging
- Machine learning pattern storage for strategy enhancement

Academic Framework:
This implementation follows established financial engineering principles for 
algorithmic trading systems and incorporates modern MLOps practices for 
quantitative finance applications.

Author: Jifeng Li
License: openMDW
Created: 2025-06-30
"""

import logging
import asyncio
import json
import uuid
from typing import Dict, List, Optional, Any, Union, Set
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
from pathlib import Path

# Configure logging with academic formatting standards
logger = logging.getLogger(__name__)


@dataclass
class AlphaSignalRecord:
    """
    Data structure for alpha signal records following academic quantitative finance standards.
    
    This class encapsulates the essential components of alpha generation signals
    as commonly defined in academic literature on algorithmic trading systems.
    """
    signal_id: str
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    confidence_score: float  # [0, 1] confidence interval
    predicted_return: float  # Expected return percentage
    risk_estimate: float  # Estimated risk metric (volatility proxy)
    execution_weight: float  # Portfolio allocation weight [-1, 1]
    timestamp: datetime
    strategy_source: str  # Strategy that generated the signal
    agent_id: str
    market_regime: Optional[str] = None  # Bull, Bear, Sideways market classification
    feature_vector: Optional[Dict[str, float]] = None  # Technical indicators
    metadata: Optional[Dict[str, Any]] = None


@dataclass
@dataclass
class StrategyPerformanceMetrics:
    """
    Comprehensive performance metrics for alpha generation strategies.
    
    Following academic standards for strategy evaluation in quantitative finance,
    including risk-adjusted returns and statistical significance measures.
    """
    strategy_id: str
    agent_id: str
    evaluation_period: timedelta
    total_signals_generated: int
    successful_predictions: int
    prediction_accuracy: float  # Hit rate
    sharpe_ratio: float
    information_ratio: float
    maximum_drawdown: float
    average_return: float
    volatility: float
    calmar_ratio: float  # Return/Maximum Drawdown
    sortino_ratio: float  # Downside risk-adjusted return
    beta: Optional[float] = None  # Market beta coefficient
    alpha: Optional[float] = None  # Jensen's alpha
    tracking_error: Optional[float] = None
    value_at_risk_95: Optional[float] = None  # 95% VaR
    conditional_var: Optional[float] = None  # Expected Shortfall
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class MemoryPatternRecord:
    """
    Pattern storage for machine learning enhancement of alpha generation.
    
    This structure captures recurring market patterns and their associated
    alpha generation outcomes for reinforcement learning applications.
    """
    pattern_id: str
    pattern_type: str  # technical, fundamental, sentiment, macro
    pattern_features: Dict[str, float]  # Feature vector representation
    associated_outcomes: List[Dict[str, Any]]  # Historical signal outcomes
    success_rate: float
    pattern_frequency: int  # Occurrence frequency
    market_conditions: Dict[str, Any]  # Associated market regime characteristics
    discovery_timestamp: datetime
    last_occurrence: datetime
    statistical_significance: float  # p-value for pattern validity
    agent_source: str
    metadata: Optional[Dict[str, Any]] = None


class AlphaAgentPoolMemoryBridge:
    """
    Advanced memory bridge implementing state-of-the-art practices for 
    alpha generation strategy storage and retrieval in quantitative finance.
    
    This class provides a comprehensive interface between alpha generation agents
    and the external memory system, supporting both real-time signal processing
    and historical pattern analysis for strategy enhancement.
    
    Key Features:
    - Real-time signal event streaming
    - Performance analytics and risk metrics storage
    - Pattern recognition for strategy enhancement
    - Cross-strategy correlation analysis
    - Automated model retraining triggers
    """
    
    def __init__(self, 
                 external_memory_config: Optional[Dict[str, Any]] = None,
                 enable_pattern_learning: bool = True,
                 performance_tracking_enabled: bool = True,
                 real_time_logging: bool = True):
        """
        Initialize Alpha Agent Pool Memory Bridge with academic-grade configuration.
        
        Args:
            external_memory_config: Configuration for external memory agent connectivity
            enable_pattern_learning: Enable machine learning pattern storage and retrieval
            performance_tracking_enabled: Enable comprehensive performance analytics
            real_time_logging: Enable real-time event streaming to memory system
        """
        self.namespace = "alpha_agent_pool"
        self.external_memory_config = external_memory_config or {}
        self.enable_pattern_learning = enable_pattern_learning
        self.performance_tracking_enabled = performance_tracking_enabled
        self.real_time_logging = real_time_logging
        
        # Initialize external memory agent connection
        self.external_memory_agent = None
        self.session_id = f"alpha_session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        # Local caching for performance optimization
        self.local_signal_cache: Dict[str, AlphaSignalRecord] = {}
        self.local_pattern_cache: Dict[str, MemoryPatternRecord] = {}
        self.performance_cache: Dict[str, StrategyPerformanceMetrics] = {}
        
        # Statistics tracking
        self.bridge_statistics = {
            'signals_stored': 0,
            'patterns_discovered': 0,
            'performance_evaluations': 0,
            'memory_retrievals': 0,
            'last_activity': datetime.now(timezone.utc)
        }
        
        logger.info(f"Alpha Agent Pool Memory Bridge initialized with session: {self.session_id}")
    
    async def initialize(self) -> None:
        """
        Initialize external memory agent connection and verify system readiness.
        
        This method establishes the connection to the external memory system
        and performs initial health checks for operational readiness.
        """
        try:
            # Import external memory agent (lazy loading for dependency management)
            from ...memory.external_memory_agent import (
                ExternalMemoryAgent, EventType, LogLevel, SQLiteStorageBackend
            )
            
            # Initialize external memory agent with optimized configuration
            storage_backend = SQLiteStorageBackend(
                db_path=f"alpha_agent_pool_memory_{self.session_id}.db"
            )
            
            self.external_memory_agent = ExternalMemoryAgent(
                storage_backend=storage_backend,
                enable_real_time_hooks=self.real_time_logging,
                max_batch_size=1000
            )
            
            await self.external_memory_agent.initialize()
            
            # Log initialization event
            await self._log_system_event(
                event_type=EventType.SYSTEM,
                log_level=LogLevel.INFO,
                title="Alpha Agent Pool Memory Bridge Initialized",
                content=f"Successfully initialized memory bridge with session {self.session_id}",
                metadata={
                    "pattern_learning_enabled": self.enable_pattern_learning,
                    "performance_tracking_enabled": self.performance_tracking_enabled,
                    "real_time_logging": self.real_time_logging
                }
            )
            
            logger.info("External memory agent successfully initialized for Alpha Agent Pool")
            
        except ImportError as e:
            logger.warning(f"External memory agent not available: {e}")
            self.external_memory_agent = None
        except Exception as e:
            logger.error(f"Failed to initialize memory bridge: {e}")
            self.external_memory_agent = None
            raise
    
    async def close(self) -> None:
        """
        Close the memory bridge and clean up resources.
        
        This method properly closes the external memory agent connection
        and performs necessary cleanup operations.
        """
        try:
            if self.external_memory_agent:
                await self.external_memory_agent.close()
                self.external_memory_agent = None
            
            # Clear local caches
            self.local_signal_cache.clear()
            self.local_pattern_cache.clear()
            
            logger.info("Alpha Agent Pool Memory Bridge closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing memory bridge: {e}")

    async def store_alpha_signal(self, signal_record: AlphaSignalRecord) -> str:
        """
        Store alpha generation signals with comprehensive metadata for analysis.
        
        This method implements best practices for alpha signal storage in 
        quantitative finance systems, including proper categorization and
        metadata enrichment for subsequent analysis.
        
        Args:
            signal_record: Complete alpha signal record with metrics
            
        Returns:
            str: Unique storage identifier for the signal record
        """
        try:
            storage_key = f"{self.namespace}:signals:{signal_record.signal_id}"
            
            # Cache locally for immediate access
            self.local_signal_cache[signal_record.signal_id] = signal_record
            
            # Store in external memory system if available
            if self.external_memory_agent:
                await self._log_signal_event(
                    signal_record=signal_record,
                    event_type="SIGNAL_GENERATED"
                )
            
            # Update statistics
            self.bridge_statistics['signals_stored'] += 1
            self.bridge_statistics['last_activity'] = datetime.now(timezone.utc)
            
            logger.info(f"Alpha signal stored: {signal_record.symbol} - {signal_record.signal_type} "
                       f"(Confidence: {signal_record.confidence_score:.3f})")
            
            return storage_key
            
        except Exception as e:
            logger.error(f"Failed to store alpha signal: {e}")
            await self._log_error_event(
                error_type="SIGNAL_STORAGE_ERROR",
                error_message=str(e),
                context={"signal_id": signal_record.signal_id}
            )
            raise
    
    async def retrieve_alpha_signals(self, 
                                   filters: Optional[Dict[str, Any]] = None,
                                   time_range: Optional[timedelta] = None,
                                   limit: int = 100) -> List[AlphaSignalRecord]:
        """
        Retrieve alpha signals with advanced filtering capabilities.
        
        Implements sophisticated querying mechanisms for alpha signal retrieval,
        supporting both real-time and historical analysis workflows.
        
        Args:
            filters: Dictionary of filter criteria (symbol, strategy, confidence_threshold, etc.)
            time_range: Time window for signal retrieval (default: last 24 hours)
            limit: Maximum number of signals to retrieve
            
        Returns:
            List[AlphaSignalRecord]: Filtered list of alpha signal records
        """
        try:
            # Set default time range if not specified
            if time_range is None:
                time_range = timedelta(hours=24)
            
            cutoff_time = datetime.now(timezone.utc) - time_range
            
            # First, check local cache for recent signals
            cached_signals = []
            for signal in self.local_signal_cache.values():
                if signal.timestamp >= cutoff_time:
                    if self._signal_matches_filters(signal, filters):
                        cached_signals.append(signal)
            
            # Limit cache results
            cached_signals = sorted(cached_signals, key=lambda x: x.timestamp, reverse=True)[:limit]
            
            # If external memory is available, supplement with stored signals
            if self.external_memory_agent and len(cached_signals) < limit:
                # TODO: Implement sophisticated querying through external memory agent
                # This would involve converting filters to QueryFilter format
                pass
            
            # Update retrieval statistics
            self.bridge_statistics['memory_retrievals'] += 1
            
            logger.info(f"Retrieved {len(cached_signals)} alpha signals with filters: {filters}")
            return cached_signals
            
        except Exception as e:
            logger.error(f"Failed to retrieve alpha signals: {e}")
            await self._log_error_event(
                error_type="SIGNAL_RETRIEVAL_ERROR",
                error_message=str(e),
                context={"filters": filters}
            )
            return []
    
    async def store_strategy_performance(self, performance_metrics: StrategyPerformanceMetrics) -> str:
        """
        Store comprehensive strategy performance metrics for academic analysis.
        
        This method captures and stores detailed performance analytics following
        academic standards for quantitative strategy evaluation, enabling
        rigorous performance attribution and risk analysis.
        
        Args:
            performance_metrics: Complete performance metrics record
            
        Returns:
            str: Storage identifier for performance record
        """
        try:
            storage_key = f"{self.namespace}:performance:{performance_metrics.strategy_id}:{performance_metrics.timestamp.isoformat()}"
            
            # Cache performance metrics locally
            self.performance_cache[performance_metrics.strategy_id] = performance_metrics
            
            # Log performance event to external memory
            if self.external_memory_agent:
                await self._log_performance_event(performance_metrics)
            
            # Update statistics
            self.bridge_statistics['performance_evaluations'] += 1
            
            logger.info(f"Strategy performance stored: {performance_metrics.strategy_id} "
                       f"(Sharpe: {performance_metrics.sharpe_ratio:.3f}, "
                       f"Accuracy: {performance_metrics.prediction_accuracy:.3f})")
            
            return storage_key
            
        except Exception as e:
            logger.error(f"Failed to store strategy performance: {e}")
            await self._log_error_event(
                error_type="PERFORMANCE_STORAGE_ERROR",
                error_message=str(e),
                context={"strategy_id": performance_metrics.strategy_id}
            )
            raise
    
    async def discover_and_store_pattern(self, pattern_record: MemoryPatternRecord) -> str:
        """
        Store discovered market patterns for machine learning enhancement.
        
        This method implements pattern storage following academic best practices
        for market microstructure analysis and algorithmic trading enhancement.
        
        Args:
            pattern_record: Discovered pattern with statistical validation
            
        Returns:
            str: Pattern storage identifier
        """
        try:
            if not self.enable_pattern_learning:
                logger.warning("Pattern learning is disabled")
                return ""
            
            storage_key = f"{self.namespace}:patterns:{pattern_record.pattern_id}"
            
            # Cache pattern locally
            self.local_pattern_cache[pattern_record.pattern_id] = pattern_record
            
            # Log pattern discovery event
            if self.external_memory_agent:
                await self._log_pattern_discovery_event(pattern_record)
            
            # Update statistics
            self.bridge_statistics['patterns_discovered'] += 1
            
            logger.info(f"Market pattern discovered and stored: {pattern_record.pattern_type} "
                       f"(Success Rate: {pattern_record.success_rate:.3f}, "
                       f"Significance: {pattern_record.statistical_significance:.4f})")
            
            return storage_key
            
        except Exception as e:
            logger.error(f"Failed to store market pattern: {e}")
            await self._log_error_event(
                error_type="PATTERN_STORAGE_ERROR",
                error_message=str(e),
                context={"pattern_id": pattern_record.pattern_id}
            )
            raise
    
    async def retrieve_relevant_patterns(self, 
                                       market_conditions: Dict[str, Any],
                                       pattern_types: Optional[List[str]] = None,
                                       min_success_rate: float = 0.6,
                                       min_significance: float = 0.05) -> List[MemoryPatternRecord]:
        """
        Retrieve market patterns relevant to current market conditions.
        
        Implements sophisticated pattern matching algorithms for real-time
        alpha generation enhancement based on historical market patterns.
        
        Args:
            market_conditions: Current market regime characteristics
            pattern_types: Specific pattern types to retrieve
            min_success_rate: Minimum historical success rate threshold
            min_significance: Minimum statistical significance threshold
            
        Returns:
            List[MemoryPatternRecord]: Relevant patterns for current conditions
        """
        try:
            if not self.enable_pattern_learning:
                return []
            
            relevant_patterns = []
            
            # Search local pattern cache
            for pattern in self.local_pattern_cache.values():
                if self._pattern_matches_conditions(pattern, market_conditions, 
                                                   pattern_types, min_success_rate, min_significance):
                    relevant_patterns.append(pattern)
            
            # Sort by relevance (success rate * statistical significance)
            relevant_patterns.sort(
                key=lambda p: p.success_rate * (1 - p.statistical_significance),
                reverse=True
            )
            
            logger.info(f"Retrieved {len(relevant_patterns)} relevant patterns for current market conditions")
            return relevant_patterns
            
        except Exception as e:
            logger.error(f"Failed to retrieve relevant patterns: {e}")
            await self._log_error_event(
                error_type="PATTERN_RETRIEVAL_ERROR",
                error_message=str(e),
                context={"market_conditions": market_conditions}
            )
            return []
    
    async def get_strategy_performance_analytics(self, 
                                               strategy_id: str,
                                               analysis_period: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance analytics for strategy evaluation.
        
        This method provides detailed performance attribution analysis following
        academic standards for quantitative strategy evaluation and risk management.
        
        Args:
            strategy_id: Identifier for the strategy to analyze
            analysis_period: Time period for analysis (default: last 30 days)
            
        Returns:
            Dict[str, Any]: Comprehensive performance analytics report
        """
        try:
            if analysis_period is None:
                analysis_period = timedelta(days=30)
            
            # Retrieve strategy performance history
            performance_history = await self._get_strategy_performance_history(
                strategy_id, analysis_period
            )
            
            if not performance_history:
                return {"error": "No performance data available for specified period"}
            
            # Generate analytics report
            analytics_report = {
                "strategy_id": strategy_id,
                "analysis_period": analysis_period.days,
                "performance_summary": {
                    "total_evaluations": len(performance_history),
                    "average_sharpe_ratio": sum(p.sharpe_ratio for p in performance_history) / len(performance_history),
                    "average_accuracy": sum(p.prediction_accuracy for p in performance_history) / len(performance_history),
                    "best_sharpe": max(p.sharpe_ratio for p in performance_history),
                    "worst_drawdown": min(p.maximum_drawdown for p in performance_history),
                    "consistency_score": self._calculate_consistency_score(performance_history)
                },
                "risk_metrics": {
                    "average_volatility": sum(p.volatility for p in performance_history) / len(performance_history),
                    "max_var_95": max(p.value_at_risk_95 for p in performance_history if p.value_at_risk_95),
                    "average_tracking_error": sum(p.tracking_error for p in performance_history if p.tracking_error) / len([p for p in performance_history if p.tracking_error]) if any(p.tracking_error for p in performance_history) else None
                },
                "trend_analysis": {
                    "performance_trend": self._analyze_performance_trend(performance_history),
                    "improvement_rate": self._calculate_improvement_rate(performance_history)
                },
                "recommendations": self._generate_strategy_recommendations(performance_history)
            }
            
            # Log analytics generation
            if self.external_memory_agent:
                await self._log_analytics_event(strategy_id, analytics_report)
            
            logger.info(f"Generated performance analytics for strategy: {strategy_id}")
            return analytics_report
            
        except Exception as e:
            logger.error(f"Failed to generate performance analytics: {e}")
            await self._log_error_event(
                error_type="ANALYTICS_GENERATION_ERROR",
                error_message=str(e),
                context={"strategy_id": strategy_id}
            )
            return {"error": str(e)}
    
    async def get_bridge_statistics(self) -> Dict[str, Any]:
        """
        Retrieve comprehensive statistics about memory bridge operations.
        
        Returns:
            Dict[str, Any]: Detailed operational statistics and health metrics
        """
        bridge_stats = self.bridge_statistics.copy()
        bridge_stats.update({
            "session_id": self.session_id,
            "local_cache_sizes": {
                "signals": len(self.local_signal_cache),
                "patterns": len(self.local_pattern_cache),
                "performance_records": len(self.performance_cache)
            },
            "external_memory_available": self.external_memory_agent is not None,
            "features_enabled": {
                "pattern_learning": self.enable_pattern_learning,
                "performance_tracking": self.performance_tracking_enabled,
                "real_time_logging": self.real_time_logging
            }
        })
        
        # Add external memory statistics if available
        if self.external_memory_agent:
            external_stats = await self.external_memory_agent.get_statistics()
            bridge_stats["external_memory_stats"] = external_stats
        
        return bridge_stats
    
    # Private helper methods for internal operations
    
    async def _log_system_event(self, event_type, log_level, title: str, content: str, 
                               metadata: Optional[Dict[str, Any]] = None):
        """Log system events to external memory agent"""
        if self.external_memory_agent:
            try:
                await self.external_memory_agent.log_event(
                    event_type=event_type,
                    log_level=log_level,
                    source_agent_pool=self.namespace,
                    source_agent_id="memory_bridge",
                    title=title,
                    content=content,
                    tags={"system", "memory_bridge", "alpha_agent_pool"},
                    metadata=metadata,
                    session_id=self.session_id
                )
            except Exception as e:
                logger.warning(f"Failed to log system event: {e}")
    
    async def _log_signal_event(self, signal_record: AlphaSignalRecord, event_type: str):
        """Log alpha signal events to external memory"""
        if self.external_memory_agent:
            try:
                from ...memory.external_memory_agent import EventType, LogLevel
                
                await self.external_memory_agent.log_event(
                    event_type=EventType.OPTIMIZATION,
                    log_level=LogLevel.INFO,
                    source_agent_pool=self.namespace,
                    source_agent_id=signal_record.agent_id,
                    title=f"Alpha Signal: {signal_record.symbol} - {signal_record.signal_type}",
                    content=f"Generated {signal_record.signal_type} signal for {signal_record.symbol} "
                           f"with confidence {signal_record.confidence_score:.3f} "
                           f"and predicted return {signal_record.predicted_return:.4f}",
                    tags={"alpha_signal", signal_record.signal_type.lower(), signal_record.symbol, 
                          signal_record.strategy_source},
                    metadata={
                        "signal_id": signal_record.signal_id,
                        "symbol": signal_record.symbol,
                        "signal_type": signal_record.signal_type,
                        "confidence_score": signal_record.confidence_score,
                        "predicted_return": signal_record.predicted_return,
                        "risk_estimate": signal_record.risk_estimate,
                        "execution_weight": signal_record.execution_weight,
                        "strategy_source": signal_record.strategy_source,
                        "market_regime": signal_record.market_regime,
                        "feature_vector": signal_record.feature_vector
                    },
                    session_id=self.session_id
                )
            except Exception as e:
                logger.warning(f"Failed to log signal event: {e}")
    
    async def _log_performance_event(self, performance_metrics: StrategyPerformanceMetrics):
        """Log performance metrics to external memory"""
        if self.external_memory_agent:
            try:
                from ...memory.external_memory_agent import EventType, LogLevel
                
                await self.external_memory_agent.log_event(
                    event_type=EventType.OPTIMIZATION,
                    log_level=LogLevel.INFO,
                    source_agent_pool=self.namespace,
                    source_agent_id=performance_metrics.agent_id,
                    title=f"Strategy Performance: {performance_metrics.strategy_id}",
                    content=f"Performance evaluation for {performance_metrics.strategy_id}: "
                           f"Sharpe Ratio {performance_metrics.sharpe_ratio:.3f}, "
                           f"Accuracy {performance_metrics.prediction_accuracy:.3f}, "
                           f"Max Drawdown {performance_metrics.maximum_drawdown:.3f}",
                    tags={"performance_metrics", "strategy_evaluation", 
                          performance_metrics.strategy_id, performance_metrics.agent_id},
                    metadata=asdict(performance_metrics),
                    session_id=self.session_id
                )
            except Exception as e:
                logger.warning(f"Failed to log performance event: {e}")
    
    async def _log_pattern_discovery_event(self, pattern_record: MemoryPatternRecord):
        """Log pattern discovery events to external memory"""
        if self.external_memory_agent:
            try:
                from ...memory.external_memory_agent import EventType, LogLevel
                
                await self.external_memory_agent.log_event(
                    event_type=EventType.SYSTEM,
                    log_level=LogLevel.INFO,
                    source_agent_pool=self.namespace,
                    source_agent_id=pattern_record.agent_source,
                    title=f"Pattern Discovered: {pattern_record.pattern_type}",
                    content=f"Discovered {pattern_record.pattern_type} pattern with "
                           f"success rate {pattern_record.success_rate:.3f} "
                           f"and significance {pattern_record.statistical_significance:.4f}",
                    tags={"pattern_discovery", pattern_record.pattern_type, 
                          "machine_learning", "market_analysis"},
                    metadata=asdict(pattern_record),
                    session_id=self.session_id
                )
            except Exception as e:
                logger.warning(f"Failed to log pattern discovery event: {e}")
    
    async def _log_error_event(self, error_type: str, error_message: str, 
                              context: Optional[Dict[str, Any]] = None):
        """Log error events to external memory"""
        if self.external_memory_agent:
            try:
                from ...memory.external_memory_agent import EventType, LogLevel
                
                await self.external_memory_agent.log_event(
                    event_type=EventType.ERROR,
                    log_level=LogLevel.ERROR,
                    source_agent_pool=self.namespace,
                    source_agent_id="memory_bridge",
                    title=f"Memory Bridge Error: {error_type}",
                    content=f"Error occurred in memory bridge: {error_message}",
                    tags={"error", "memory_bridge", error_type.lower()},
                    metadata={
                        "error_type": error_type,
                        "error_message": error_message,
                        "context": context or {}
                    },
                    session_id=self.session_id
                )
            except Exception as e:
                logger.warning(f"Failed to log error event: {e}")
    
    async def _log_analytics_event(self, strategy_id: str, analytics_report: Dict[str, Any]):
        """Log analytics generation events"""
        if self.external_memory_agent:
            try:
                from ...memory.external_memory_agent import EventType, LogLevel
                
                await self.external_memory_agent.log_event(
                    event_type=EventType.SYSTEM,
                    log_level=LogLevel.INFO,
                    source_agent_pool=self.namespace,
                    source_agent_id="memory_bridge",
                    title=f"Performance Analytics Generated: {strategy_id}",
                    content=f"Generated comprehensive performance analytics for strategy {strategy_id}",
                    tags={"analytics", "performance_analysis", strategy_id, "reporting"},
                    metadata=analytics_report,
                    session_id=self.session_id
                )
            except Exception as e:
                logger.warning(f"Failed to log analytics event: {e}")
    
    def _signal_matches_filters(self, signal: AlphaSignalRecord, 
                               filters: Optional[Dict[str, Any]]) -> bool:
        """Check if signal matches provided filters"""
        if not filters:
            return True
        
        # Symbol filter
        if 'symbol' in filters and signal.symbol != filters['symbol']:
            return False
        
        # Signal type filter
        if 'signal_type' in filters and signal.signal_type != filters['signal_type']:
            return False
        
        # Confidence threshold filter
        if 'min_confidence' in filters and signal.confidence_score < filters['min_confidence']:
            return False
        
        # Strategy source filter
        if 'strategy_source' in filters and signal.strategy_source != filters['strategy_source']:
            return False
        
        # Agent ID filter
        if 'agent_id' in filters and signal.agent_id != filters['agent_id']:
            return False
        
        return True
    
    def _pattern_matches_conditions(self, pattern: MemoryPatternRecord,
                                   market_conditions: Dict[str, Any],
                                   pattern_types: Optional[List[str]],
                                   min_success_rate: float,
                                   min_significance: float) -> bool:
        """Check if pattern matches current market conditions and criteria"""
        # Basic threshold checks
        if pattern.success_rate < min_success_rate:
            return False
        
        if pattern.statistical_significance > min_significance:
            return False
        
        # Pattern type filter
        if pattern_types and pattern.pattern_type not in pattern_types:
            return False
        
        # TODO: Implement sophisticated market condition matching
        # This would involve comparing current market regime characteristics
        # with the pattern's associated market conditions
        
        return True
    
    async def _get_strategy_performance_history(self, strategy_id: str, 
                                              period: timedelta) -> List[StrategyPerformanceMetrics]:
        """Retrieve performance history for a strategy"""
        # For now, return cached performance if available
        if strategy_id in self.performance_cache:
            return [self.performance_cache[strategy_id]]
        
        # TODO: Implement retrieval from external memory system
        return []
    
    def _calculate_consistency_score(self, performance_history: List[StrategyPerformanceMetrics]) -> float:
        """Calculate consistency score based on performance variance"""
        if len(performance_history) < 2:
            return 0.0
        
        sharpe_ratios = [p.sharpe_ratio for p in performance_history]
        mean_sharpe = sum(sharpe_ratios) / len(sharpe_ratios)
        variance = sum((sr - mean_sharpe) ** 2 for sr in sharpe_ratios) / len(sharpe_ratios)
        
        # Consistency score (higher is more consistent)
        return max(0.0, 1.0 - (variance / max(abs(mean_sharpe), 0.1)))
    
    def _analyze_performance_trend(self, performance_history: List[StrategyPerformanceMetrics]) -> str:
        """Analyze performance trend over time"""
        if len(performance_history) < 2:
            return "insufficient_data"
        
        sorted_history = sorted(performance_history, key=lambda x: x.timestamp)
        recent_performance = sum(p.sharpe_ratio for p in sorted_history[-3:]) / min(3, len(sorted_history))
        early_performance = sum(p.sharpe_ratio for p in sorted_history[:3]) / min(3, len(sorted_history))
        
        if recent_performance > early_performance * 1.1:
            return "improving"
        elif recent_performance < early_performance * 0.9:
            return "declining"
        else:
            return "stable"
    
    def _calculate_improvement_rate(self, performance_history: List[StrategyPerformanceMetrics]) -> float:
        """Calculate rate of performance improvement"""
        if len(performance_history) < 2:
            return 0.0
        
        sorted_history = sorted(performance_history, key=lambda x: x.timestamp)
        first_sharpe = sorted_history[0].sharpe_ratio
        last_sharpe = sorted_history[-1].sharpe_ratio
        
        if first_sharpe == 0:
            return 0.0
        
        return (last_sharpe - first_sharpe) / abs(first_sharpe)
    
    def _generate_strategy_recommendations(self, performance_history: List[StrategyPerformanceMetrics]) -> List[str]:
        """Generate actionable recommendations based on performance analysis"""
        recommendations = []
        
        if not performance_history:
            return ["Insufficient performance data for recommendations"]
        
        latest_performance = performance_history[-1]
        
        # Sharpe ratio recommendations
        if latest_performance.sharpe_ratio < 0.5:
            recommendations.append("Consider strategy parameter optimization to improve risk-adjusted returns")
        
        # Accuracy recommendations
        if latest_performance.prediction_accuracy < 0.6:
            recommendations.append("Investigate signal quality and consider ensemble methods for improved accuracy")
        
        # Drawdown recommendations
        if latest_performance.maximum_drawdown > 0.2:
            recommendations.append("Implement stronger risk management controls to reduce maximum drawdown")
        
        # Information ratio recommendations
        if latest_performance.information_ratio < 0.3:
            recommendations.append("Focus on alpha generation independent of market beta")
        
        if not recommendations:
            recommendations.append("Strategy performance is within acceptable parameters")
        
        return recommendations


# Convenience functions for backward compatibility and ease of use

async def create_alpha_memory_bridge(config: Optional[Dict[str, Any]] = None) -> AlphaAgentPoolMemoryBridge:
    """
    Factory function to create and initialize Alpha Agent Pool Memory Bridge.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        AlphaAgentPoolMemoryBridge: Initialized memory bridge instance
    """
    bridge = AlphaAgentPoolMemoryBridge(
        external_memory_config=config,
        enable_pattern_learning=True,
        performance_tracking_enabled=True,
        real_time_logging=True
    )
    
    await bridge.initialize()
    return bridge


def create_alpha_signal_record(symbol: str, signal_type: str, confidence: float,
                             predicted_return: float, risk_estimate: float,
                             execution_weight: float, strategy_source: str,
                             agent_id: str, **kwargs) -> AlphaSignalRecord:
    """
    Convenience function to create alpha signal records with proper validation.
    
    Args:
        symbol: Trading symbol (e.g., 'AAPL', 'MSFT')
        signal_type: Signal type ('BUY', 'SELL', 'HOLD')
        confidence: Confidence score [0, 1]
        predicted_return: Expected return percentage
        risk_estimate: Risk metric (volatility proxy)
        execution_weight: Portfolio weight [-1, 1]
        strategy_source: Strategy identifier
        agent_id: Agent identifier
        **kwargs: Additional optional parameters
        
    Returns:
        AlphaSignalRecord: Validated alpha signal record
    """
    return AlphaSignalRecord(
        signal_id=str(uuid.uuid4()),
        symbol=symbol.strip().upper(),
        signal_type=signal_type.strip().upper(),
        confidence_score=max(0.0, min(1.0, confidence)),  # Clamp to [0,1]
        predicted_return=predicted_return,
        risk_estimate=abs(risk_estimate),  # Ensure positive risk
        execution_weight=max(-1.0, min(1.0, execution_weight)),  # Clamp to [-1,1]
        timestamp=datetime.now(timezone.utc),
        strategy_source=strategy_source,
        agent_id=agent_id,
        **kwargs
    )


def create_performance_metrics_record(strategy_id: str, agent_id: str,
                                    signals_generated: int, successful_predictions: int,
                                    sharpe_ratio: float, information_ratio: float,
                                    max_drawdown: float, avg_return: float,
                                    volatility: float, **kwargs) -> StrategyPerformanceMetrics:
    """
    Convenience function to create performance metrics records.
    
    Args:
        strategy_id: Strategy identifier
        agent_id: Agent identifier
        signals_generated: Total number of signals generated
        successful_predictions: Number of successful predictions
        sharpe_ratio: Sharpe ratio
        information_ratio: Information ratio
        max_drawdown: Maximum drawdown
        avg_return: Average return
        volatility: Return volatility
        **kwargs: Additional optional metrics
        
    Returns:
        StrategyPerformanceMetrics: Validated performance metrics record
    """
    accuracy = successful_predictions / max(signals_generated, 1)
    calmar_ratio = kwargs.get('calmar_ratio', avg_return / max(abs(max_drawdown), 0.001))
    
    return StrategyPerformanceMetrics(
        strategy_id=strategy_id,
        agent_id=agent_id,
        evaluation_period=timedelta(days=30),  # Default evaluation period
        total_signals_generated=signals_generated,
        successful_predictions=successful_predictions,
        prediction_accuracy=accuracy,
        sharpe_ratio=sharpe_ratio,
        information_ratio=information_ratio,
        maximum_drawdown=abs(max_drawdown),
        average_return=avg_return,
        volatility=abs(volatility),
        calmar_ratio=calmar_ratio,
        sortino_ratio=kwargs.get('sortino_ratio', sharpe_ratio * 1.1),  # Approximate if not provided
        **{k: v for k, v in kwargs.items() if k not in ['sortino_ratio', 'calmar_ratio']}
    )
