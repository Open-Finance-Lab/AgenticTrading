"""
Timing Optimizer

This agent optimizes the timing of trade execution to minimize transaction costs
and market impact while maximizing execution quality.
"""

import logging
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
import asyncio

from ...schema.optimization_schema import (
    OptimizationRequest,
    ExecutionRecommendation
)


logger = logging.getLogger(__name__)


class TimingStrategy(Enum):
    """Different timing strategies for execution."""
    IMMEDIATE = "immediate"
    SPREAD_EVENLY = "spread_evenly"
    VOLUME_WEIGHTED = "volume_weighted"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    MOMENTUM_BASED = "momentum_based"
    CONTRARIAN = "contrarian"
    ADAPTIVE = "adaptive"


class MarketSession(Enum):
    """Different market sessions."""
    PRE_MARKET = "pre_market"
    MARKET_OPEN = "market_open"
    MORNING_SESSION = "morning_session"
    MIDDAY = "midday"
    AFTERNOON_SESSION = "afternoon_session"
    MARKET_CLOSE = "market_close"
    AFTER_HOURS = "after_hours"


@dataclass
class TimingWindow:
    """Represents a timing window for execution."""
    start_time: datetime
    end_time: datetime
    expected_volume_pct: float
    expected_volatility: float
    expected_spread_bps: float
    expected_market_impact_bps: float
    liquidity_score: float
    session: MarketSession
    priority_score: float


@dataclass
class TimingRecommendation:
    """Recommendation for execution timing."""
    order_id: str
    symbol: str
    recommended_strategy: TimingStrategy
    optimal_timing_windows: List[TimingWindow]
    slice_schedule: List[Dict[str, Any]]
    expected_cost_reduction_bps: float
    expected_market_impact_reduction_bps: float
    risk_metrics: Dict[str, float]
    contingency_timing: List[TimingWindow]
    monitoring_triggers: List[str]


@dataclass
class TimingOptimizationResult:
    """Result of timing optimization."""
    optimization_id: str
    timestamp: datetime
    timing_recommendations: List[TimingRecommendation]
    market_timing_analysis: Dict[str, Any]
    expected_cost_savings_bps: float
    expected_risk_reduction: float
    implementation_schedule: Dict[str, Any]
    success_probability: float


class TimingOptimizer:
    """
    Advanced timing optimizer for minimizing transaction costs through optimal execution timing.
    
    This optimizer provides comprehensive timing optimization including:
    - Intraday timing pattern analysis
    - Volume and volatility-based scheduling
    - Market microstructure optimization
    - Momentum and mean-reversion strategies
    - Real-time timing adjustments
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the timing optimizer.
        
        Args:
            config: Configuration parameters for the optimizer
        """
        self.config = config or {}
        self.timing_models = {}
        self.historical_patterns = {}
        self.market_calendar = None
        
        # Timing parameters
        self.max_execution_windows = self.config.get('max_execution_windows', 6)
        self.min_window_size_minutes = self.config.get('min_window_size_minutes', 15)
        self.max_delay_hours = self.config.get('max_delay_hours', 4)
        
        # Market timing patterns
        self.intraday_patterns = self._initialize_intraday_patterns()
        
        # Risk parameters
        self.max_timing_risk = self.config.get('max_timing_risk', 0.15)
        self.opportunity_cost_weight = self.config.get('opportunity_cost_weight', 0.3)
        
    def _initialize_intraday_patterns(self) -> Dict[str, Dict[str, float]]:
        """Initialize typical intraday trading patterns."""
        
        return {
            'volume_pattern': {
                9: 1.8,   # Market open - high volume
                10: 1.4,  # Post-open
                11: 0.9,  # Morning quiet
                12: 0.8,  # Midday low
                13: 0.9,  # Early afternoon
                14: 1.1,  # Afternoon pickup
                15: 1.3,  # Pre-close
                16: 1.6   # Market close - high volume
            },
            'volatility_pattern': {
                9: 1.5,   # High volatility at open
                10: 1.2,
                11: 0.9,
                12: 0.8,  # Lowest volatility midday
                13: 0.8,
                14: 0.9,
                15: 1.1,
                16: 1.4   # High volatility at close
            },
            'spread_pattern': {
                9: 1.3,   # Wider spreads at open
                10: 1.1,
                11: 0.9,
                12: 0.8,  # Tightest spreads midday
                13: 0.8,
                14: 0.9,
                15: 1.0,
                16: 1.2   # Wider spreads at close
            },
            'market_impact_pattern': {
                9: 1.4,   # Higher impact at open
                10: 1.1,
                11: 0.9,
                12: 0.8,  # Lowest impact midday
                13: 0.8,
                14: 0.9,
                15: 1.1,
                16: 1.3   # Higher impact at close
            }
        }
    
    async def optimize_execution_timing(
        self, 
        request: OptimizationRequest,
        market_conditions: Optional[Dict[str, Any]] = None,
        optimization_id: Optional[str] = None
    ) -> TimingOptimizationResult:
        """
        Optimize execution timing for given orders.
        
        Args:
            request: Optimization request containing orders and parameters
            market_conditions: Current market conditions
            optimization_id: Unique identifier for this optimization
            
        Returns:
            Comprehensive timing optimization results
        """
        try:
            optimization_id = optimization_id or f"timing_opt_{datetime.utcnow().isoformat()}"
            logger.info(f"Starting timing optimization {optimization_id}")
            
            # Analyze market timing conditions
            if not market_conditions:
                market_conditions = await self._analyze_market_timing_conditions()
            
            # Generate timing windows
            timing_windows = await self._generate_timing_windows(market_conditions)
            
            # Optimize timing for each order
            timing_recommendations = []
            for order in request.orders:
                recommendation = await self._optimize_single_order_timing(
                    order, timing_windows, market_conditions, request
                )
                timing_recommendations.append(recommendation)
            
            # Perform market timing analysis
            market_analysis = await self._perform_market_timing_analysis(
                timing_windows, market_conditions
            )
            
            # Calculate expected improvements
            improvements = await self._calculate_timing_improvements(
                timing_recommendations, request
            )
            
            # Generate implementation schedule
            implementation_schedule = await self._generate_implementation_schedule(
                timing_recommendations
            )
            
            # Calculate success probability
            success_probability = await self._calculate_success_probability(
                timing_recommendations, market_conditions
            )
            
            # Compile results
            result = TimingOptimizationResult(
                optimization_id=optimization_id,
                timestamp=datetime.utcnow(),
                timing_recommendations=timing_recommendations,
                market_timing_analysis=market_analysis,
                expected_cost_savings_bps=improvements['cost_savings_bps'],
                expected_risk_reduction=improvements['risk_reduction'],
                implementation_schedule=implementation_schedule,
                success_probability=success_probability
            )
            
            logger.info(f"Timing optimization {optimization_id} completed")
            return result
            
        except Exception as e:
            logger.error(f"Error in timing optimization: {str(e)}")
            raise
    
    async def _analyze_market_timing_conditions(self) -> Dict[str, Any]:
        """Analyze current market conditions for timing optimization."""
        
        current_time = datetime.utcnow()
        current_hour = current_time.hour
        
        # Determine market session
        if 4 <= current_hour < 9:
            session = MarketSession.PRE_MARKET
        elif 9 <= current_hour < 10:
            session = MarketSession.MARKET_OPEN
        elif 10 <= current_hour < 12:
            session = MarketSession.MORNING_SESSION
        elif 12 <= current_hour < 14:
            session = MarketSession.MIDDAY
        elif 14 <= current_hour < 16:
            session = MarketSession.AFTERNOON_SESSION
        elif 16 <= current_hour < 17:
            session = MarketSession.MARKET_CLOSE
        else:
            session = MarketSession.AFTER_HOURS
        
        return {
            'current_session': session,
            'current_hour': current_hour,
            'market_stress_level': 0.3,  # 0-1 scale
            'overall_volatility': 0.25,  # Annualized volatility
            'liquidity_conditions': 'normal',  # poor, fair, normal, good, excellent
            'news_flow_intensity': 'low',  # low, medium, high
            'earnings_season': False,
            'economic_events_today': [],
            'sector_rotation_activity': 'moderate',
            'intraday_momentum': 'neutral',  # bearish, neutral, bullish
            'volume_vs_average': 1.1,  # Current volume vs. average
            'volatility_vs_average': 0.9  # Current volatility vs. average
        }
    
    async def _generate_timing_windows(
        self, 
        market_conditions: Dict[str, Any]
    ) -> List[TimingWindow]:
        """Generate optimal timing windows based on market conditions."""
        
        windows = []
        current_time = datetime.utcnow()
        
        # Generate windows for remaining trading hours
        for hour in range(current_time.hour, 17):  # Until market close
            # Skip if current hour is almost over
            if hour == current_time.hour and current_time.minute > 45:
                continue
            
            # Create 30-minute windows
            for period in [0, 30]:
                start_time = current_time.replace(hour=hour, minute=period, second=0, microsecond=0)
                end_time = start_time + timedelta(minutes=30)
                
                # Skip past windows
                if start_time <= current_time:
                    continue
                
                # Get market session
                session = self._get_market_session(hour)
                
                # Calculate window characteristics
                volume_pct = self.intraday_patterns['volume_pattern'].get(hour, 1.0)
                volatility = self.intraday_patterns['volatility_pattern'].get(hour, 1.0)
                spread_bps = self.intraday_patterns['spread_pattern'].get(hour, 1.0) * 5.0  # Base 5 bps
                market_impact_bps = self.intraday_patterns['market_impact_pattern'].get(hour, 1.0) * 3.0  # Base 3 bps
                
                # Adjust for market conditions
                stress_level = market_conditions.get('market_stress_level', 0.3)
                volume_pct *= (1.0 + market_conditions.get('volume_vs_average', 1.0) - 1.0)
                volatility *= (1.0 + stress_level * 0.5)
                spread_bps *= (1.0 + stress_level * 0.3)
                market_impact_bps *= (1.0 + stress_level * 0.4)
                
                # Calculate liquidity score (inverse of cost)
                liquidity_score = 1.0 / (1.0 + spread_bps/10 + market_impact_bps/10)
                
                # Calculate priority score (higher is better)
                priority_score = (
                    volume_pct * 0.3 +
                    liquidity_score * 0.4 +
                    (1.0 / volatility) * 0.2 +
                    (1.0 / (market_impact_bps + 1e-6)) * 0.1
                )
                
                window = TimingWindow(
                    start_time=start_time,
                    end_time=end_time,
                    expected_volume_pct=volume_pct,
                    expected_volatility=volatility,
                    expected_spread_bps=spread_bps,
                    expected_market_impact_bps=market_impact_bps,
                    liquidity_score=liquidity_score,
                    session=session,
                    priority_score=priority_score
                )
                
                windows.append(window)
        
        # Sort by priority score (descending)
        windows.sort(key=lambda w: w.priority_score, reverse=True)
        
        return windows[:self.max_execution_windows]
    
    def _get_market_session(self, hour: int) -> MarketSession:
        """Get market session for a given hour."""
        if hour < 9:
            return MarketSession.PRE_MARKET
        elif hour == 9:
            return MarketSession.MARKET_OPEN
        elif 10 <= hour < 12:
            return MarketSession.MORNING_SESSION
        elif 12 <= hour < 14:
            return MarketSession.MIDDAY
        elif 14 <= hour < 16:
            return MarketSession.AFTERNOON_SESSION
        elif hour == 16:
            return MarketSession.MARKET_CLOSE
        else:
            return MarketSession.AFTER_HOURS
    
    async def _optimize_single_order_timing(
        self, 
        order,
        timing_windows: List[TimingWindow],
        market_conditions: Dict[str, Any],
        request: OptimizationRequest
    ) -> TimingRecommendation:
        """Optimize timing for a single order."""
        
        # Determine optimal timing strategy
        strategy = await self._select_timing_strategy(order, market_conditions, request)
        
        # Select optimal timing windows
        optimal_windows = await self._select_optimal_windows(
            order, timing_windows, strategy, market_conditions
        )
        
        # Generate slice schedule
        slice_schedule = await self._generate_slice_schedule(
            order, optimal_windows, strategy
        )
        
        # Calculate expected improvements
        cost_reduction = await self._calculate_cost_reduction(
            order, optimal_windows, strategy
        )
        impact_reduction = await self._calculate_impact_reduction(
            order, optimal_windows, strategy
        )
        
        # Calculate risk metrics
        risk_metrics = await self._calculate_timing_risk_metrics(
            order, optimal_windows, strategy, market_conditions
        )
        
        # Generate contingency timing
        contingency_windows = await self._generate_contingency_timing(
            optimal_windows, timing_windows
        )
        
        # Generate monitoring triggers
        monitoring_triggers = await self._generate_monitoring_triggers(
            order, optimal_windows, strategy
        )
        
        return TimingRecommendation(
            order_id=order.order_id,
            symbol=order.symbol,
            recommended_strategy=strategy,
            optimal_timing_windows=optimal_windows,
            slice_schedule=slice_schedule,
            expected_cost_reduction_bps=cost_reduction,
            expected_market_impact_reduction_bps=impact_reduction,
            risk_metrics=risk_metrics,
            contingency_timing=contingency_windows,
            monitoring_triggers=monitoring_triggers
        )
    
    async def _select_timing_strategy(
        self, 
        order,
        market_conditions: Dict[str, Any],
        request: OptimizationRequest
    ) -> TimingStrategy:
        """Select optimal timing strategy for an order."""
        
        # Consider order characteristics
        order_size_factor = min(order.quantity / 10000, 2.0)  # Normalize to 10k shares
        
        # Consider market conditions
        volatility = market_conditions.get('overall_volatility', 0.25)
        stress_level = market_conditions.get('market_stress_level', 0.3)
        momentum = market_conditions.get('intraday_momentum', 'neutral')
        
        # Consider urgency
        urgency = request.urgency_level
        
        # Strategy selection logic
        if urgency == "high" or stress_level > 0.7:
            return TimingStrategy.IMMEDIATE
        
        elif order_size_factor > 1.5:  # Large orders
            if volatility > 0.3:
                return TimingStrategy.VOLATILITY_ADJUSTED
            else:
                return TimingStrategy.VOLUME_WEIGHTED
        
        elif momentum == "bullish" and order.side == "BUY":
            return TimingStrategy.MOMENTUM_BASED
        elif momentum == "bearish" and order.side == "SELL":
            return TimingStrategy.MOMENTUM_BASED
        elif momentum in ["bullish", "bearish"]:
            return TimingStrategy.CONTRARIAN
        
        elif volatility > 0.35:
            return TimingStrategy.VOLATILITY_ADJUSTED
        
        else:
            return TimingStrategy.ADAPTIVE
    
    async def _select_optimal_windows(
        self, 
        order,
        timing_windows: List[TimingWindow],
        strategy: TimingStrategy,
        market_conditions: Dict[str, Any]
    ) -> List[TimingWindow]:
        """Select optimal timing windows for an order."""
        
        if strategy == TimingStrategy.IMMEDIATE:
            # Return the first available window
            return timing_windows[:1] if timing_windows else []
        
        elif strategy == TimingStrategy.VOLUME_WEIGHTED:
            # Select windows with highest volume
            return sorted(timing_windows, key=lambda w: w.expected_volume_pct, reverse=True)[:3]
        
        elif strategy == TimingStrategy.VOLATILITY_ADJUSTED:
            # Select windows with lower volatility
            return sorted(timing_windows, key=lambda w: w.expected_volatility)[:3]
        
        elif strategy == TimingStrategy.SPREAD_EVENLY:
            # Select evenly spaced windows
            if len(timing_windows) >= 3:
                step = len(timing_windows) // 3
                return [timing_windows[i*step] for i in range(3)]
            else:
                return timing_windows
        
        else:  # ADAPTIVE or others
            # Use priority score
            return timing_windows[:3]
    
    async def _generate_slice_schedule(
        self, 
        order,
        timing_windows: List[TimingWindow],
        strategy: TimingStrategy
    ) -> List[Dict[str, Any]]:
        """Generate detailed slice schedule for order execution."""
        
        if not timing_windows:
            return []
        
        schedule = []
        remaining_quantity = order.quantity
        
        for i, window in enumerate(timing_windows):
            # Calculate slice size
            if i == len(timing_windows) - 1:  # Last slice
                slice_size = remaining_quantity
            else:
                # Distribute based on strategy
                if strategy == TimingStrategy.VOLUME_WEIGHTED:
                    total_volume = sum(w.expected_volume_pct for w in timing_windows)
                    slice_pct = window.expected_volume_pct / total_volume
                else:
                    slice_pct = 1.0 / len(timing_windows)
                
                slice_size = min(remaining_quantity, int(order.quantity * slice_pct))
            
            # Create slice entry
            slice_entry = {
                'slice_number': i + 1,
                'execution_time': window.start_time,
                'end_time': window.end_time,
                'quantity': slice_size,
                'percentage_of_order': (slice_size / order.quantity) * 100,
                'expected_cost_bps': window.expected_spread_bps + window.expected_market_impact_bps,
                'expected_volume_pct': window.expected_volume_pct,
                'market_session': window.session.value,
                'priority': len(timing_windows) - i  # Higher number = higher priority
            }
            
            schedule.append(slice_entry)
            remaining_quantity -= slice_size
            
            if remaining_quantity <= 0:
                break
        
        return schedule
    
    async def _calculate_cost_reduction(
        self, 
        order,
        timing_windows: List[TimingWindow],
        strategy: TimingStrategy
    ) -> float:
        """Calculate expected cost reduction from timing optimization."""
        
        if not timing_windows:
            return 0.0
        
        # Calculate weighted average cost of selected windows
        total_weight = sum(w.expected_volume_pct for w in timing_windows)
        if total_weight == 0:
            return 0.0
        
        weighted_cost = sum(
            (w.expected_spread_bps + w.expected_market_impact_bps) * w.expected_volume_pct
            for w in timing_windows
        ) / total_weight
        
        # Baseline cost (immediate execution)
        baseline_cost = 8.0  # Assume 8 bps for immediate execution
        
        # Calculate reduction
        cost_reduction = max(0, baseline_cost - weighted_cost)
        
        # Apply strategy modifier
        if strategy == TimingStrategy.IMMEDIATE:
            cost_reduction = 0.0
        elif strategy == TimingStrategy.VOLATILITY_ADJUSTED:
            cost_reduction *= 1.2  # 20% bonus for volatility timing
        elif strategy == TimingStrategy.VOLUME_WEIGHTED:
            cost_reduction *= 1.1  # 10% bonus for volume timing
        
        return cost_reduction
    
    async def _calculate_impact_reduction(
        self, 
        order,
        timing_windows: List[TimingWindow],
        strategy: TimingStrategy
    ) -> float:
        """Calculate expected market impact reduction."""
        
        if not timing_windows:
            return 0.0
        
        # Calculate weighted average impact of selected windows
        total_weight = sum(w.expected_volume_pct for w in timing_windows)
        if total_weight == 0:
            return 0.0
        
        weighted_impact = sum(
            w.expected_market_impact_bps * w.expected_volume_pct
            for w in timing_windows
        ) / total_weight
        
        # Baseline impact (immediate execution)
        baseline_impact = 5.0  # Assume 5 bps for immediate execution
        
        return max(0, baseline_impact - weighted_impact)
    
    async def _calculate_timing_risk_metrics(
        self, 
        order,
        timing_windows: List[TimingWindow],
        strategy: TimingStrategy,
        market_conditions: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate risk metrics for timing strategy."""
        
        risk_metrics = {}
        
        # Timing risk (risk of adverse price movement)
        execution_duration_hours = 0
        if timing_windows:
            start_time = timing_windows[0].start_time
            end_time = timing_windows[-1].end_time
            execution_duration_hours = (end_time - start_time).total_seconds() / 3600
        
        # Estimate timing risk based on volatility and duration
        volatility = market_conditions.get('overall_volatility', 0.25)
        timing_risk = volatility * np.sqrt(execution_duration_hours / 24) * 10000  # In bps
        risk_metrics['timing_risk_bps'] = timing_risk
        
        # Opportunity cost risk
        opportunity_cost = timing_risk * self.opportunity_cost_weight
        risk_metrics['opportunity_cost_bps'] = opportunity_cost
        
        # Execution risk (risk of non-execution)
        if timing_windows:
            avg_liquidity = sum(w.liquidity_score for w in timing_windows) / len(timing_windows)
            execution_risk = 1.0 - avg_liquidity
        else:
            execution_risk = 0.5  # Medium risk if no windows
        risk_metrics['execution_risk'] = execution_risk
        
        # Market impact risk
        if strategy in [TimingStrategy.VOLUME_WEIGHTED, TimingStrategy.ADAPTIVE]:
            impact_risk = 0.2  # Lower risk for smart strategies
        elif strategy == TimingStrategy.IMMEDIATE:
            impact_risk = 0.8  # Higher risk for immediate execution
        else:
            impact_risk = 0.4  # Medium risk
        risk_metrics['market_impact_risk'] = impact_risk
        
        # Overall risk score
        overall_risk = (
            (timing_risk / 50) * 0.3 +      # Normalize timing risk
            execution_risk * 0.3 +
            impact_risk * 0.2 +
            (opportunity_cost / 20) * 0.2    # Normalize opportunity cost
        )
        risk_metrics['overall_risk_score'] = min(1.0, overall_risk)
        
        return risk_metrics
    
    async def _generate_contingency_timing(
        self, 
        optimal_windows: List[TimingWindow],
        all_windows: List[TimingWindow]
    ) -> List[TimingWindow]:
        """Generate contingency timing windows."""
        
        # Find backup windows not in optimal set
        optimal_times = {w.start_time for w in optimal_windows}
        contingency_windows = [
            w for w in all_windows 
            if w.start_time not in optimal_times
        ]
        
        # Sort by priority and return top 2
        contingency_windows.sort(key=lambda w: w.priority_score, reverse=True)
        return contingency_windows[:2]
    
    async def _generate_monitoring_triggers(
        self, 
        order,
        timing_windows: List[TimingWindow],
        strategy: TimingStrategy
    ) -> List[str]:
        """Generate monitoring triggers for timing strategy."""
        
        triggers = [
            "Monitor market volatility vs. expectations",
            "Track volume patterns vs. historical norms",
            "Alert if market stress level increases significantly",
            "Monitor execution progress vs. schedule"
        ]
        
        # Strategy-specific triggers
        if strategy == TimingStrategy.VOLUME_WEIGHTED:
            triggers.append("Alert if volume drops below 70% of expected")
        elif strategy == TimingStrategy.VOLATILITY_ADJUSTED:
            triggers.append("Alert if volatility exceeds 150% of expected")
        elif strategy == TimingStrategy.MOMENTUM_BASED:
            triggers.append("Monitor price momentum reversal signals")
        
        # Window-specific triggers
        if timing_windows:
            if any(w.session == MarketSession.MARKET_CLOSE for w in timing_windows):
                triggers.append("Ensure completion before market close")
            if any(w.liquidity_score < 0.5 for w in timing_windows):
                triggers.append("Monitor liquidity conditions closely in low-liquidity windows")
        
        return triggers
    
    async def _perform_market_timing_analysis(
        self, 
        timing_windows: List[TimingWindow],
        market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform comprehensive market timing analysis."""
        
        analysis = {
            'optimal_execution_periods': [],
            'periods_to_avoid': [],
            'intraday_cost_curve': {},
            'liquidity_profile': {},
            'volatility_forecast': {},
            'risk_assessment': {}
        }
        
        # Identify optimal and suboptimal periods
        high_priority_windows = [w for w in timing_windows if w.priority_score > 0.8]
        low_priority_windows = [w for w in timing_windows if w.priority_score < 0.4]
        
        analysis['optimal_execution_periods'] = [
            {
                'time': w.start_time.strftime('%H:%M'),
                'session': w.session.value,
                'priority_score': w.priority_score,
                'expected_cost_bps': w.expected_spread_bps + w.expected_market_impact_bps
            }
            for w in high_priority_windows
        ]
        
        analysis['periods_to_avoid'] = [
            {
                'time': w.start_time.strftime('%H:%M'),
                'session': w.session.value,
                'reason': 'High cost and low liquidity'
            }
            for w in low_priority_windows
        ]
        
        # Create intraday profiles
        for hour in range(9, 17):
            session_windows = [w for w in timing_windows if w.start_time.hour == hour]
            if session_windows:
                avg_cost = sum(w.expected_spread_bps + w.expected_market_impact_bps for w in session_windows) / len(session_windows)
                avg_liquidity = sum(w.liquidity_score for w in session_windows) / len(session_windows)
                avg_volatility = sum(w.expected_volatility for w in session_windows) / len(session_windows)
            else:
                avg_cost = 8.0  # Default
                avg_liquidity = 0.5
                avg_volatility = 1.0
            
            hour_str = f"{hour:02d}:00"
            analysis['intraday_cost_curve'][hour_str] = avg_cost
            analysis['liquidity_profile'][hour_str] = avg_liquidity
            analysis['volatility_forecast'][hour_str] = avg_volatility
        
        # Risk assessment
        analysis['risk_assessment'] = {
            'market_timing_risk': 'medium',
            'liquidity_risk': 'low' if market_conditions.get('liquidity_conditions') == 'normal' else 'medium',
            'volatility_risk': 'high' if market_conditions.get('overall_volatility', 0.25) > 0.35 else 'medium',
            'execution_risk': 'low'
        }
        
        return analysis
    
    async def _calculate_timing_improvements(
        self, 
        timing_recommendations: List[TimingRecommendation],
        request: OptimizationRequest
    ) -> Dict[str, float]:
        """Calculate expected improvements from timing optimization."""
        
        if not timing_recommendations:
            return {'cost_savings_bps': 0.0, 'risk_reduction': 0.0}
        
        # Calculate average improvements
        avg_cost_savings = sum(rec.expected_cost_reduction_bps for rec in timing_recommendations) / len(timing_recommendations)
        
        # Calculate risk reduction (based on risk metrics)
        avg_risk_reduction = 0.0
        for rec in timing_recommendations:
            if rec.risk_metrics:
                # Lower overall risk score = higher risk reduction
                baseline_risk = 0.6  # Assume baseline risk of 0.6
                current_risk = rec.risk_metrics.get('overall_risk_score', 0.6)
                risk_reduction = max(0, baseline_risk - current_risk)
                avg_risk_reduction += risk_reduction
        
        avg_risk_reduction /= len(timing_recommendations)
        
        return {
            'cost_savings_bps': avg_cost_savings,
            'risk_reduction': avg_risk_reduction
        }
    
    async def _generate_implementation_schedule(
        self, 
        timing_recommendations: List[TimingRecommendation]
    ) -> Dict[str, Any]:
        """Generate implementation schedule for timing recommendations."""
        
        schedule = {
            'immediate_actions': [],
            'scheduled_executions': [],
            'monitoring_checkpoints': [],
            'contingency_triggers': []
        }
        
        # Immediate actions
        schedule['immediate_actions'] = [
            "Set up execution algorithms with timing parameters",
            "Configure monitoring systems for market conditions",
            "Prepare contingency execution plans"
        ]
        
        # Scheduled executions
        all_slices = []
        for rec in timing_recommendations:
            for slice_info in rec.slice_schedule:
                all_slices.append({
                    'order_id': rec.order_id,
                    'symbol': rec.symbol,
                    'execution_time': slice_info['execution_time'],
                    'quantity': slice_info['quantity'],
                    'strategy': rec.recommended_strategy.value
                })
        
        # Sort by execution time
        all_slices.sort(key=lambda x: x['execution_time'])
        schedule['scheduled_executions'] = all_slices
        
        # Monitoring checkpoints
        unique_times = sorted(set(slice['execution_time'] for slice in all_slices))
        schedule['monitoring_checkpoints'] = [
            {
                'time': execution_time,
                'actions': [
                    "Check market conditions vs. expectations",
                    "Verify algorithm parameters",
                    "Confirm contingency plans are ready"
                ]
            }
            for execution_time in unique_times[:3]  # First 3 checkpoints
        ]
        
        # Contingency triggers
        schedule['contingency_triggers'] = [
            "Market volatility exceeds 150% of expected",
            "Volume drops below 60% of expected",
            "Market stress indicators spike",
            "Technical execution issues arise"
        ]
        
        return schedule
    
    async def _calculate_success_probability(
        self, 
        timing_recommendations: List[TimingRecommendation],
        market_conditions: Dict[str, Any]
    ) -> float:
        """Calculate probability of successful timing optimization."""
        
        if not timing_recommendations:
            return 0.5
        
        # Base success probability
        base_probability = 0.8
        
        # Adjust for market conditions
        stress_level = market_conditions.get('market_stress_level', 0.3)
        volatility = market_conditions.get('overall_volatility', 0.25)
        
        # Penalty for stressed/volatile conditions
        stress_penalty = stress_level * 0.2
        volatility_penalty = max(0, (volatility - 0.3) * 0.3)
        
        # Bonus for good strategies
        strategy_bonus = 0.1  # Average bonus for good timing strategies
        
        # Calculate overall probability
        success_probability = base_probability - stress_penalty - volatility_penalty + strategy_bonus
        
        return max(0.3, min(0.95, success_probability))


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from datetime import datetime
    
    async def test_timing_optimizer():
        """Test the timing optimizer with sample data."""
        
        # Create sample optimization request
        from ...schema.optimization_schema import OptimizationRequest, OrderToOptimize
        
        request = OptimizationRequest(
            request_id="timing_test_001",
            timestamp=datetime.utcnow(),
            orders=[
                OrderToOptimize(
                    order_id="test_order_1",
                    symbol="AAPL",
                    side="BUY",
                    quantity=20000,  # Large order - should benefit from timing
                    limit_price=150.00,
                    time_in_force="DAY"
                ),
                OrderToOptimize(
                    order_id="test_order_2",
                    symbol="MSFT",
                    side="SELL",
                    quantity=8000,
                    limit_price=340.00,
                    time_in_force="DAY"
                )
            ],
            objective="minimize_cost",
            urgency_level="medium",
            risk_tolerance="moderate"
        )
        
        # Initialize optimizer and run optimization
        optimizer = TimingOptimizer()
        result = await optimizer.optimize_execution_timing(request)
        
        print("=== Timing Optimization Results ===")
        print(f"Optimization ID: {result.optimization_id}")
        print(f"Expected Cost Savings: {result.expected_cost_savings_bps:.2f} bps")
        print(f"Expected Risk Reduction: {result.expected_risk_reduction:.2%}")
        print(f"Success Probability: {result.success_probability:.1%}")
        
        print("\n=== Timing Recommendations ===")
        for rec in result.timing_recommendations:
            print(f"\nOrder: {rec.symbol} ({rec.order_id})")
            print(f"Strategy: {rec.recommended_strategy.value}")
            print(f"Expected Cost Reduction: {rec.expected_cost_reduction_bps:.2f} bps")
            print(f"Expected Impact Reduction: {rec.expected_market_impact_reduction_bps:.2f} bps")
            
            print("Execution Schedule:")
            for slice_info in rec.slice_schedule:
                print(f"  - {slice_info['execution_time'].strftime('%H:%M')}: "
                      f"{slice_info['quantity']:,} shares "
                      f"({slice_info['percentage_of_order']:.1f}%)")
        
        print("\n=== Market Timing Analysis ===")
        optimal_periods = result.market_timing_analysis['optimal_execution_periods']
        print("Optimal Execution Periods:")
        for period in optimal_periods:
            print(f"- {period['time']} ({period['session']}): "
                  f"{period['expected_cost_bps']:.2f} bps")
        
        print("\n=== Implementation Schedule ===")
        immediate_actions = result.implementation_schedule['immediate_actions']
        print("Immediate Actions:")
        for action in immediate_actions:
            print(f"- {action}")
    
    # Run the test
    asyncio.run(test_timing_optimizer())
