"""
Transaction Cost Optimizer

This agent optimizes transaction costs by recommending optimal execution strategies,
order types, sizing, and timing based on market conditions and historical performance.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
import asyncio

from ...schema.optimization_schema import (
    OptimizationRequest,
    OptimizationResult,
    OptimizationStrategy,
    ExecutionRecommendation
)
from ...schema.cost_models import TransactionCostBreakdown, CostEstimate


logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """Optimization objectives for transaction cost optimization."""
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_RISK = "minimize_risk"
    MINIMIZE_MARKET_IMPACT = "minimize_market_impact"
    MAXIMIZE_FILL_RATE = "maximize_fill_rate"
    BALANCE_COST_RISK = "balance_cost_risk"


@dataclass
class OptimizationConstraints:
    """Constraints for the optimization process."""
    max_market_impact_bps: Optional[float] = None
    max_cost_bps: Optional[float] = None
    min_fill_rate: Optional[float] = None
    max_execution_time_minutes: Optional[int] = None
    allowed_venues: Optional[List[str]] = None
    allowed_order_types: Optional[List[str]] = None
    max_order_size: Optional[int] = None


@dataclass
class OptimizedStrategy:
    """Result of cost optimization analysis."""
    strategy_id: str
    objective: OptimizationObjective
    recommended_actions: List[ExecutionRecommendation]
    expected_cost_reduction_bps: float
    expected_risk_reduction: float
    confidence_score: float
    implementation_priority: str
    estimated_savings_usd: float
    strategy_description: str
    constraints_satisfied: bool


class CostOptimizer:
    """
    Advanced transaction cost optimizer that recommends optimal execution strategies.
    
    This optimizer provides comprehensive cost optimization including:
    - Multi-objective optimization (cost, risk, market impact)
    - Dynamic strategy selection based on market conditions
    - Order sizing and timing optimization
    - Venue and routing optimization
    - Real-time strategy adaptation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the cost optimizer.
        
        Args:
            config: Configuration parameters for the optimizer
        """
        self.config = config or {}
        self.optimization_cache = {}
        self.strategy_performance = {}
        self.market_regime_detector = None
        
        # Optimization parameters
        self.optimization_horizon = self.config.get('optimization_horizon_minutes', 30)
        self.max_iterations = self.config.get('max_iterations', 1000)
        self.convergence_threshold = self.config.get('convergence_threshold', 0.001)
        
        # Strategy parameters
        self.strategy_universe = self.config.get('strategy_universe', [
            'TWAP', 'VWAP', 'POV', 'IS', 'ADAPTIVE'
        ])
        
        # Risk parameters
        self.risk_aversion = self.config.get('risk_aversion', 0.5)  # 0 = risk neutral, 1 = very risk averse
        self.max_concentration = self.config.get('max_concentration', 0.1)  # Max % of ADV
        
    async def optimize_execution_strategy(
        self, 
        request: OptimizationRequest,
        constraints: Optional[OptimizationConstraints] = None,
        optimization_id: Optional[str] = None
    ) -> OptimizationResult:
        """
        Optimize execution strategy for given orders and market conditions.
        
        Args:
            request: Optimization request containing orders and parameters
            constraints: Optional constraints for the optimization
            optimization_id: Unique identifier for this optimization
            
        Returns:
            Comprehensive optimization results with recommended strategies
        """
        try:
            optimization_id = optimization_id or f"opt_{datetime.utcnow().isoformat()}"
            logger.info(f"Starting cost optimization {optimization_id}")
            
            # Analyze current market conditions
            market_conditions = await self._analyze_market_conditions(request)
            
            # Generate strategy alternatives
            strategy_alternatives = await self._generate_strategy_alternatives(
                request, market_conditions, constraints
            )
            
            # Evaluate each strategy
            evaluated_strategies = []
            for strategy in strategy_alternatives:
                evaluation = await self._evaluate_strategy(
                    strategy, request, market_conditions, constraints
                )
                evaluated_strategies.append(evaluation)
            
            # Select optimal strategy
            optimal_strategy = await self._select_optimal_strategy(
                evaluated_strategies, request.objective
            )
            
            # Generate detailed recommendations
            recommendations = await self._generate_detailed_recommendations(
                optimal_strategy, request, market_conditions
            )
            
            # Calculate expected improvements
            improvements = await self._calculate_expected_improvements(
                optimal_strategy, request
            )
            
            # Compile results
            result = OptimizationResult(
                request_id=request.request_id,
                optimization_id=optimization_id,
                timestamp=datetime.utcnow(),
                optimal_strategy=optimal_strategy,
                alternative_strategies=evaluated_strategies[:3],  # Top 3 alternatives
                recommendations=recommendations,
                expected_cost_savings_bps=improvements['cost_savings_bps'],
                expected_risk_reduction=improvements['risk_reduction'],
                implementation_timeline=improvements['implementation_timeline'],
                confidence_level=optimal_strategy.confidence_score
            )
            
            logger.info(f"Cost optimization {optimization_id} completed")
            return result
            
        except Exception as e:
            logger.error(f"Error in cost optimization: {str(e)}")
            raise
    
    async def _analyze_market_conditions(self, request: OptimizationRequest) -> Dict[str, Any]:
        """Analyze current market conditions for optimization."""
        
        market_conditions = {
            'volatility_regime': 'normal',  # normal, high, low
            'liquidity_conditions': 'good',  # good, fair, poor
            'market_stress': 0.3,  # 0-1 scale
            'time_of_day_factor': 1.0,  # Multiplier based on time
            'symbol_specific_factors': {}
        }
        
        # Analyze each symbol in the request
        for order in request.orders:
            symbol_analysis = {
                'estimated_adv': 1000000,  # Average daily volume
                'typical_spread_bps': 5.0,
                'volatility_percentile': 50,
                'recent_news_impact': 'low',
                'earnings_proximity': False
            }
            market_conditions['symbol_specific_factors'][order.symbol] = symbol_analysis
        
        # Time-of-day analysis
        current_hour = datetime.utcnow().hour
        if 9 <= current_hour <= 10 or 15 <= current_hour <= 16:
            market_conditions['time_of_day_factor'] = 1.2  # Higher impact during open/close
        elif 11 <= current_hour <= 14:
            market_conditions['time_of_day_factor'] = 0.9  # Lower impact during midday
        
        return market_conditions
    
    async def _generate_strategy_alternatives(
        self, 
        request: OptimizationRequest,
        market_conditions: Dict[str, Any],
        constraints: Optional[OptimizationConstraints]
    ) -> List[OptimizationStrategy]:
        """Generate alternative execution strategies."""
        
        strategies = []
        
        # TWAP Strategy
        twap_strategy = OptimizationStrategy(
            strategy_id="TWAP_001",
            strategy_type="TWAP",
            parameters={
                'duration_minutes': self.optimization_horizon,
                'slice_size_pct': 5.0,  # % of order size per slice
                'order_type': 'LIMIT',
                'participation_rate': 0.1  # % of volume
            },
            expected_cost_bps=4.5,
            expected_risk_score=0.3,
            market_impact_bps=2.0,
            description="Time-Weighted Average Price strategy with regular intervals"
        )
        strategies.append(twap_strategy)
        
        # VWAP Strategy
        vwap_strategy = OptimizationStrategy(
            strategy_id="VWAP_001",
            strategy_type="VWAP",
            parameters={
                'duration_minutes': self.optimization_horizon,
                'volume_profile': 'historical',
                'order_type': 'LIMIT',
                'participation_rate': 0.15
            },
            expected_cost_bps=4.0,
            expected_risk_score=0.25,
            market_impact_bps=1.8,
            description="Volume-Weighted Average Price strategy following historical volume patterns"
        )
        strategies.append(vwap_strategy)
        
        # POV (Percentage of Volume) Strategy
        pov_strategy = OptimizationStrategy(
            strategy_id="POV_001",
            strategy_type="POV",
            parameters={
                'target_participation': 0.12,
                'max_participation': 0.20,
                'order_type': 'ADAPTIVE',
                'urgency_level': 'medium'
            },
            expected_cost_bps=3.8,
            expected_risk_score=0.35,
            market_impact_bps=2.2,
            description="Percentage of Volume strategy targeting specific participation rate"
        )
        strategies.append(pov_strategy)
        
        # Implementation Shortfall Strategy
        is_strategy = OptimizationStrategy(
            strategy_id="IS_001",
            strategy_type="IS",
            parameters={
                'risk_aversion': self.risk_aversion,
                'market_impact_sensitivity': 1.0,
                'timing_risk_weight': 0.5,
                'order_type': 'ADAPTIVE'
            },
            expected_cost_bps=3.5,
            expected_risk_score=0.28,
            market_impact_bps=1.5,
            description="Implementation Shortfall strategy optimizing cost vs. timing risk"
        )
        strategies.append(is_strategy)
        
        # Adaptive Strategy (Market Condition Dependent)
        adaptive_params = self._adapt_strategy_to_market(market_conditions)
        adaptive_strategy = OptimizationStrategy(
            strategy_id="ADAPTIVE_001",
            strategy_type="ADAPTIVE",
            parameters=adaptive_params,
            expected_cost_bps=3.2,
            expected_risk_score=0.22,
            market_impact_bps=1.3,
            description="Adaptive strategy that adjusts to real-time market conditions"
        )
        strategies.append(adaptive_strategy)
        
        # Filter strategies based on constraints
        if constraints:
            strategies = self._apply_constraints(strategies, constraints)
        
        return strategies
    
    async def _evaluate_strategy(
        self, 
        strategy: OptimizationStrategy,
        request: OptimizationRequest,
        market_conditions: Dict[str, Any],
        constraints: Optional[OptimizationConstraints]
    ) -> OptimizedStrategy:
        """Evaluate a specific strategy and return optimized version."""
        
        # Calculate expected performance metrics
        expected_cost = await self._calculate_expected_cost(strategy, request, market_conditions)
        expected_risk = await self._calculate_expected_risk(strategy, request, market_conditions)
        market_impact = await self._calculate_market_impact(strategy, request, market_conditions)
        
        # Generate execution recommendations
        recommendations = await self._generate_strategy_recommendations(
            strategy, request, market_conditions
        )
        
        # Calculate confidence score
        confidence = await self._calculate_confidence_score(strategy, market_conditions)
        
        # Estimate savings
        baseline_cost = 6.0  # Assume 6 bps baseline
        cost_savings = max(0, baseline_cost - expected_cost)
        
        # Estimate dollar savings
        total_notional = sum(order.quantity * order.limit_price for order in request.orders if order.limit_price)
        dollar_savings = (cost_savings / 10000) * total_notional if total_notional > 0 else 0
        
        # Check constraint satisfaction
        constraints_satisfied = self._check_constraints(strategy, constraints) if constraints else True
        
        # Determine implementation priority
        priority = self._determine_priority(cost_savings, expected_risk, confidence)
        
        return OptimizedStrategy(
            strategy_id=strategy.strategy_id,
            objective=request.objective,
            recommended_actions=recommendations,
            expected_cost_reduction_bps=cost_savings,
            expected_risk_reduction=max(0, 0.4 - expected_risk),  # vs. baseline risk
            confidence_score=confidence,
            implementation_priority=priority,
            estimated_savings_usd=dollar_savings,
            strategy_description=strategy.description,
            constraints_satisfied=constraints_satisfied
        )
    
    async def _select_optimal_strategy(
        self, 
        strategies: List[OptimizedStrategy],
        objective: OptimizationObjective
    ) -> OptimizedStrategy:
        """Select the optimal strategy based on the objective."""
        
        if not strategies:
            raise ValueError("No strategies to evaluate")
        
        # Score strategies based on objective
        scored_strategies = []
        for strategy in strategies:
            if not strategy.constraints_satisfied:
                continue  # Skip strategies that don't satisfy constraints
                
            score = self._calculate_objective_score(strategy, objective)
            scored_strategies.append((score, strategy))
        
        if not scored_strategies:
            # If no strategies satisfy constraints, return best overall
            logger.warning("No strategies satisfy constraints, returning best overall")
            scored_strategies = [(self._calculate_objective_score(s, objective), s) for s in strategies]
        
        # Sort by score (higher is better)
        scored_strategies.sort(key=lambda x: x[0], reverse=True)
        
        return scored_strategies[0][1]
    
    def _calculate_objective_score(self, strategy: OptimizedStrategy, objective: OptimizationObjective) -> float:
        """Calculate score for a strategy based on optimization objective."""
        
        if objective == OptimizationObjective.MINIMIZE_COST:
            return strategy.expected_cost_reduction_bps * strategy.confidence_score
        
        elif objective == OptimizationObjective.MINIMIZE_RISK:
            return strategy.expected_risk_reduction * strategy.confidence_score
        
        elif objective == OptimizationObjective.MINIMIZE_MARKET_IMPACT:
            # Infer market impact reduction from cost reduction
            market_impact_score = strategy.expected_cost_reduction_bps * 0.5  # Rough approximation
            return market_impact_score * strategy.confidence_score
        
        elif objective == OptimizationObjective.BALANCE_COST_RISK:
            cost_score = strategy.expected_cost_reduction_bps
            risk_score = strategy.expected_risk_reduction * 10  # Scale to match cost units
            balanced_score = (cost_score + risk_score) / 2
            return balanced_score * strategy.confidence_score
        
        else:  # Default case
            return (strategy.expected_cost_reduction_bps + strategy.expected_risk_reduction * 5) * strategy.confidence_score
    
    async def _generate_detailed_recommendations(
        self, 
        optimal_strategy: OptimizedStrategy,
        request: OptimizationRequest,
        market_conditions: Dict[str, Any]
    ) -> List[ExecutionRecommendation]:
        """Generate detailed execution recommendations."""
        
        recommendations = []
        
        for i, order in enumerate(request.orders):
            # Order-specific recommendation
            recommendation = ExecutionRecommendation(
                order_id=f"order_{i+1}",
                symbol=order.symbol,
                recommended_strategy=optimal_strategy.strategy_id,
                recommended_venue="SMART",  # Smart order routing
                recommended_order_type="LIMIT" if "LIMIT" in optimal_strategy.strategy_description else "ADAPTIVE",
                recommended_slice_size=min(order.quantity // 10, 5000),  # Max 5000 shares per slice
                estimated_execution_time_minutes=self.optimization_horizon,
                expected_cost_bps=optimal_strategy.expected_cost_reduction_bps,
                confidence_level=optimal_strategy.confidence_score,
                special_instructions=[
                    f"Use {optimal_strategy.strategy_id} algorithm",
                    "Monitor market impact closely",
                    "Adjust participation rate if conditions change"
                ]
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    async def _calculate_expected_improvements(
        self, 
        optimal_strategy: OptimizedStrategy,
        request: OptimizationRequest
    ) -> Dict[str, Any]:
        """Calculate expected improvements from optimization."""
        
        return {
            'cost_savings_bps': optimal_strategy.expected_cost_reduction_bps,
            'risk_reduction': optimal_strategy.expected_risk_reduction,
            'implementation_timeline': {
                'immediate': ['Switch to optimized algorithm'],
                'short_term': ['Monitor performance', 'Fine-tune parameters'],
                'long_term': ['Evaluate strategy effectiveness', 'Consider further optimization']
            },
            'success_probability': optimal_strategy.confidence_score
        }
    
    def _adapt_strategy_to_market(self, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt strategy parameters based on market conditions."""
        
        base_params = {
            'participation_rate': 0.12,
            'slice_size_pct': 5.0,
            'order_type': 'LIMIT',
            'urgency_multiplier': 1.0
        }
        
        # Adjust for market stress
        stress_level = market_conditions.get('market_stress', 0.3)
        if stress_level > 0.7:
            base_params['participation_rate'] *= 0.8  # Reduce participation in stressed markets
            base_params['order_type'] = 'ADAPTIVE'
        
        # Adjust for time of day
        time_factor = market_conditions.get('time_of_day_factor', 1.0)
        base_params['urgency_multiplier'] = time_factor
        
        # Adjust for liquidity
        liquidity = market_conditions.get('liquidity_conditions', 'good')
        if liquidity == 'poor':
            base_params['slice_size_pct'] *= 0.5  # Smaller slices in poor liquidity
            base_params['participation_rate'] *= 0.7
        
        return base_params
    
    def _apply_constraints(
        self, 
        strategies: List[OptimizationStrategy], 
        constraints: OptimizationConstraints
    ) -> List[OptimizationStrategy]:
        """Filter strategies based on constraints."""
        
        filtered_strategies = []
        
        for strategy in strategies:
            # Check cost constraint
            if constraints.max_cost_bps and strategy.expected_cost_bps > constraints.max_cost_bps:
                continue
            
            # Check market impact constraint
            if constraints.max_market_impact_bps and strategy.market_impact_bps > constraints.max_market_impact_bps:
                continue
            
            # Check venue constraints
            if constraints.allowed_venues:
                # This would need more detailed venue routing logic
                pass
            
            # Check order type constraints
            if constraints.allowed_order_types:
                strategy_order_type = strategy.parameters.get('order_type', 'LIMIT')
                if strategy_order_type not in constraints.allowed_order_types:
                    continue
            
            filtered_strategies.append(strategy)
        
        return filtered_strategies
    
    async def _calculate_expected_cost(
        self, 
        strategy: OptimizationStrategy,
        request: OptimizationRequest,
        market_conditions: Dict[str, Any]
    ) -> float:
        """Calculate expected execution cost for a strategy."""
        
        base_cost = strategy.expected_cost_bps
        
        # Adjust for market conditions
        stress_adjustment = market_conditions.get('market_stress', 0.3) * 2.0
        time_adjustment = (market_conditions.get('time_of_day_factor', 1.0) - 1.0) * 1.0
        
        adjusted_cost = base_cost + stress_adjustment + time_adjustment
        
        return max(0.5, adjusted_cost)  # Minimum 0.5 bps
    
    async def _calculate_expected_risk(
        self, 
        strategy: OptimizationStrategy,
        request: OptimizationRequest,
        market_conditions: Dict[str, Any]
    ) -> float:
        """Calculate expected execution risk for a strategy."""
        
        base_risk = strategy.expected_risk_score
        
        # Adjust for market volatility
        volatility_adjustment = 0.1 if market_conditions.get('volatility_regime') == 'high' else 0.0
        
        # Adjust for liquidity
        liquidity_adjustment = 0.15 if market_conditions.get('liquidity_conditions') == 'poor' else 0.0
        
        adjusted_risk = base_risk + volatility_adjustment + liquidity_adjustment
        
        return min(1.0, max(0.0, adjusted_risk))
    
    async def _calculate_market_impact(
        self, 
        strategy: OptimizationStrategy,
        request: OptimizationRequest,
        market_conditions: Dict[str, Any]
    ) -> float:
        """Calculate expected market impact for a strategy."""
        
        base_impact = strategy.market_impact_bps
        
        # Adjust for order size vs. ADV
        total_quantity = sum(order.quantity for order in request.orders)
        # Simplified: assume 1M average daily volume
        participation_rate = total_quantity / 1000000
        
        if participation_rate > 0.1:  # > 10% of ADV
            size_adjustment = (participation_rate - 0.1) * 20  # 20 bps per 1% above 10%
        else:
            size_adjustment = 0
        
        adjusted_impact = base_impact + size_adjustment
        
        return max(0.0, adjusted_impact)
    
    async def _generate_strategy_recommendations(
        self, 
        strategy: OptimizationStrategy,
        request: OptimizationRequest,
        market_conditions: Dict[str, Any]
    ) -> List[ExecutionRecommendation]:
        """Generate recommendations for a specific strategy."""
        
        recommendations = []
        # This would be implemented with detailed order-by-order recommendations
        # For now, return empty list as it's handled in the main flow
        return recommendations
    
    async def _calculate_confidence_score(
        self, 
        strategy: OptimizationStrategy,
        market_conditions: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for a strategy."""
        
        base_confidence = 0.8
        
        # Reduce confidence in stressed markets
        stress_penalty = market_conditions.get('market_stress', 0.3) * 0.2
        
        # Reduce confidence in poor liquidity
        liquidity_penalty = 0.1 if market_conditions.get('liquidity_conditions') == 'poor' else 0.0
        
        confidence = base_confidence - stress_penalty - liquidity_penalty
        
        return max(0.3, min(0.95, confidence))
    
    def _check_constraints(
        self, 
        strategy: OptimizationStrategy, 
        constraints: OptimizationConstraints
    ) -> bool:
        """Check if strategy satisfies constraints."""
        
        # This is a simplified check - would be more comprehensive in practice
        if constraints.max_cost_bps and strategy.expected_cost_reduction_bps < 0:
            return False
        
        return True
    
    def _determine_priority(self, cost_savings: float, risk: float, confidence: float) -> str:
        """Determine implementation priority for a strategy."""
        
        score = cost_savings * confidence - risk * 2
        
        if score > 2.0:
            return "HIGH"
        elif score > 1.0:
            return "MEDIUM"
        else:
            return "LOW"


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from datetime import datetime
    
    async def test_cost_optimizer():
        """Test the cost optimizer with sample data."""
        
        # Create sample optimization request
        from ...schema.optimization_schema import OptimizationRequest, OrderToOptimize
        
        request = OptimizationRequest(
            request_id="opt_test_001",
            timestamp=datetime.utcnow(),
            orders=[
                OrderToOptimize(
                    order_id="test_order_1",
                    symbol="AAPL",
                    side="BUY",
                    quantity=10000,
                    limit_price=150.00,
                    time_in_force="DAY"
                ),
                OrderToOptimize(
                    order_id="test_order_2",
                    symbol="MSFT",
                    side="SELL",
                    quantity=5000,
                    limit_price=340.00,
                    time_in_force="DAY"
                )
            ],
            objective=OptimizationObjective.MINIMIZE_COST,
            urgency_level="medium",
            risk_tolerance="moderate"
        )
        
        # Create constraints
        constraints = OptimizationConstraints(
            max_cost_bps=8.0,
            max_market_impact_bps=5.0,
            min_fill_rate=0.95,
            allowed_order_types=["LIMIT", "ADAPTIVE"]
        )
        
        # Initialize optimizer and run optimization
        optimizer = CostOptimizer()
        result = await optimizer.optimize_execution_strategy(request, constraints)
        
        print("=== Cost Optimization Results ===")
        print(f"Optimization ID: {result.optimization_id}")
        print(f"Optimal Strategy: {result.optimal_strategy.strategy_id}")
        print(f"Expected Cost Savings: {result.expected_cost_savings_bps:.2f} bps")
        print(f"Expected Risk Reduction: {result.expected_risk_reduction:.2f}")
        print(f"Confidence Level: {result.confidence_level:.1%}")
        print(f"Estimated Dollar Savings: ${result.optimal_strategy.estimated_savings_usd:,.2f}")
        
        print("\n=== Implementation Recommendations ===")
        for rec in result.recommendations:
            print(f"- {rec.symbol}: Use {rec.recommended_strategy} with {rec.recommended_order_type} orders")
            print(f"  Slice size: {rec.recommended_slice_size} shares")
            print(f"  Expected cost: {rec.expected_cost_bps:.2f} bps")
        
        print("\n=== Alternative Strategies ===")
        for alt in result.alternative_strategies:
            print(f"- {alt.strategy_id}: {alt.expected_cost_reduction_bps:.2f} bps savings "
                  f"(confidence: {alt.confidence_score:.1%})")
    
    # Run the test
    asyncio.run(test_cost_optimizer())
