"""
Routing Optimizer

This agent optimizes order routing decisions to minimize transaction costs
and maximize execution quality across different venues and liquidity pools.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum

from ...schema.optimization_schema import (
    OptimizationRequest,
    ExecutionRecommendation
)


logger = logging.getLogger(__name__)


class VenueType(Enum):
    """Types of trading venues."""
    EXCHANGE = "exchange"
    DARK_POOL = "dark_pool"
    ECN = "ecn"
    MARKET_MAKER = "market_maker"
    CROSSING_NETWORK = "crossing_network"


@dataclass
class VenueCharacteristics:
    """Characteristics of a trading venue."""
    venue_id: str
    venue_type: VenueType
    typical_cost_bps: float
    typical_fill_rate: float
    typical_speed_ms: float
    min_order_size: int
    max_order_size: int
    supports_hidden_orders: bool
    supports_iceberg_orders: bool
    market_share_pct: float
    liquidity_score: float
    adverse_selection_score: float


@dataclass
class RoutingRecommendation:
    """Recommendation for order routing."""
    order_id: str
    symbol: str
    primary_venue: str
    secondary_venues: List[str]
    allocation_percentages: Dict[str, float]
    expected_cost_bps: float
    expected_fill_rate: float
    expected_execution_time_minutes: float
    routing_logic: str
    risk_assessment: str
    contingency_plan: List[str]


@dataclass
class RoutingOptimizationResult:
    """Result of routing optimization."""
    optimization_id: str
    timestamp: datetime
    routing_recommendations: List[RoutingRecommendation]
    venue_analysis: Dict[str, Dict[str, float]]
    expected_cost_improvement_bps: float
    expected_fill_rate_improvement: float
    implementation_complexity: str
    monitoring_requirements: List[str]


class RoutingOptimizer:
    """
    Advanced order routing optimizer for minimizing transaction costs.
    
    This optimizer provides comprehensive routing optimization including:
    - Multi-venue cost analysis and comparison
    - Dynamic routing based on real-time conditions
    - Smart order routing with fallback strategies
    - Dark pool vs. lit venue optimization
    - Liquidity aggregation strategies
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the routing optimizer.
        
        Args:
            config: Configuration parameters for the optimizer
        """
        self.config = config or {}
        self.venue_universe = {}
        self.routing_cache = {}
        self.performance_tracker = {}
        
        # Initialize venue universe
        self._initialize_venue_universe()
        
        # Optimization parameters
        self.max_venues_per_order = self.config.get('max_venues_per_order', 3)
        self.min_allocation_pct = self.config.get('min_allocation_pct', 10.0)
        self.routing_frequency_ms = self.config.get('routing_frequency_ms', 100)
        
        # Risk parameters
        self.max_venue_concentration = self.config.get('max_venue_concentration', 0.5)
        self.adverse_selection_weight = self.config.get('adverse_selection_weight', 0.3)
        
    def _initialize_venue_universe(self):
        """Initialize the universe of available trading venues."""
        
        # Major exchanges
        self.venue_universe['NYSE'] = VenueCharacteristics(
            venue_id='NYSE',
            venue_type=VenueType.EXCHANGE,
            typical_cost_bps=2.5,
            typical_fill_rate=0.85,
            typical_speed_ms=1.2,
            min_order_size=1,
            max_order_size=1000000,
            supports_hidden_orders=True,
            supports_iceberg_orders=True,
            market_share_pct=15.5,
            liquidity_score=0.9,
            adverse_selection_score=0.4
        )
        
        self.venue_universe['NASDAQ'] = VenueCharacteristics(
            venue_id='NASDAQ',
            venue_type=VenueType.EXCHANGE,
            typical_cost_bps=2.3,
            typical_fill_rate=0.87,
            typical_speed_ms=1.1,
            min_order_size=1,
            max_order_size=1000000,
            supports_hidden_orders=True,
            supports_iceberg_orders=True,
            market_share_pct=18.2,
            liquidity_score=0.92,
            adverse_selection_score=0.38
        )
        
        self.venue_universe['ARCA'] = VenueCharacteristics(
            venue_id='ARCA',
            venue_type=VenueType.ECN,
            typical_cost_bps=2.1,
            typical_fill_rate=0.82,
            typical_speed_ms=0.9,
            min_order_size=1,
            max_order_size=500000,
            supports_hidden_orders=True,
            supports_iceberg_orders=True,
            market_share_pct=12.3,
            liquidity_score=0.85,
            adverse_selection_score=0.35
        )
        
        # Dark pools
        self.venue_universe['DARK_POOL_1'] = VenueCharacteristics(
            venue_id='DARK_POOL_1',
            venue_type=VenueType.DARK_POOL,
            typical_cost_bps=1.8,
            typical_fill_rate=0.65,
            typical_speed_ms=5.0,
            min_order_size=100,
            max_order_size=250000,
            supports_hidden_orders=True,
            supports_iceberg_orders=False,
            market_share_pct=8.5,
            liquidity_score=0.7,
            adverse_selection_score=0.25
        )
        
        self.venue_universe['DARK_POOL_2'] = VenueCharacteristics(
            venue_id='DARK_POOL_2',
            venue_type=VenueType.DARK_POOL,
            typical_cost_bps=1.9,
            typical_fill_rate=0.68,
            typical_speed_ms=4.5,
            min_order_size=100,
            max_order_size=300000,
            supports_hidden_orders=True,
            supports_iceberg_orders=False,
            market_share_pct=7.2,
            liquidity_score=0.72,
            adverse_selection_score=0.22
        )
        
        # ECNs
        self.venue_universe['ECN_1'] = VenueCharacteristics(
            venue_id='ECN_1',
            venue_type=VenueType.ECN,
            typical_cost_bps=2.0,
            typical_fill_rate=0.78,
            typical_speed_ms=1.5,
            min_order_size=1,
            max_order_size=750000,
            supports_hidden_orders=True,
            supports_iceberg_orders=True,
            market_share_pct=9.8,
            liquidity_score=0.8,
            adverse_selection_score=0.42
        )
    
    async def optimize_routing(
        self, 
        request: OptimizationRequest,
        market_conditions: Optional[Dict[str, Any]] = None,
        optimization_id: Optional[str] = None
    ) -> RoutingOptimizationResult:
        """
        Optimize order routing across available venues.
        
        Args:
            request: Optimization request containing orders to route
            market_conditions: Current market conditions
            optimization_id: Unique identifier for this optimization
            
        Returns:
            Comprehensive routing optimization results
        """
        try:
            optimization_id = optimization_id or f"routing_opt_{datetime.utcnow().isoformat()}"
            logger.info(f"Starting routing optimization {optimization_id}")
            
            # Analyze market conditions
            if not market_conditions:
                market_conditions = await self._analyze_current_market_conditions()
            
            # Update venue characteristics based on current conditions
            updated_venues = await self._update_venue_characteristics(market_conditions)
            
            # Generate routing recommendations for each order
            routing_recommendations = []
            for order in request.orders:
                recommendation = await self._optimize_single_order_routing(
                    order, updated_venues, market_conditions
                )
                routing_recommendations.append(recommendation)
            
            # Perform venue analysis
            venue_analysis = await self._perform_venue_analysis(updated_venues, request)
            
            # Calculate expected improvements
            improvements = await self._calculate_routing_improvements(
                routing_recommendations, request
            )
            
            # Assess implementation complexity
            complexity = await self._assess_implementation_complexity(routing_recommendations)
            
            # Generate monitoring requirements
            monitoring_requirements = await self._generate_monitoring_requirements(
                routing_recommendations
            )
            
            # Compile results
            result = RoutingOptimizationResult(
                optimization_id=optimization_id,
                timestamp=datetime.utcnow(),
                routing_recommendations=routing_recommendations,
                venue_analysis=venue_analysis,
                expected_cost_improvement_bps=improvements['cost_improvement_bps'],
                expected_fill_rate_improvement=improvements['fill_rate_improvement'],
                implementation_complexity=complexity,
                monitoring_requirements=monitoring_requirements
            )
            
            logger.info(f"Routing optimization {optimization_id} completed")
            return result
            
        except Exception as e:
            logger.error(f"Error in routing optimization: {str(e)}")
            raise
    
    async def _analyze_current_market_conditions(self) -> Dict[str, Any]:
        """Analyze current market conditions for routing optimization."""
        
        return {
            'market_volatility': 0.25,  # 0-1 scale
            'overall_liquidity': 0.8,   # 0-1 scale
            'market_stress': 0.3,       # 0-1 scale
            'time_of_day_factor': 1.0,  # Multiplier
            'venue_outages': [],        # List of venues with issues
            'venue_performance_adjustments': {
                'NYSE': 1.0,
                'NASDAQ': 1.05,  # Slightly better performance
                'ARCA': 0.95,
                'DARK_POOL_1': 0.9,
                'DARK_POOL_2': 0.85
            }
        }
    
    async def _update_venue_characteristics(
        self, 
        market_conditions: Dict[str, Any]
    ) -> Dict[str, VenueCharacteristics]:
        """Update venue characteristics based on current market conditions."""
        
        updated_venues = {}
        
        for venue_id, venue in self.venue_universe.items():
            # Skip venues with outages
            if venue_id in market_conditions.get('venue_outages', []):
                continue
            
            # Create updated venue characteristics
            updated_venue = VenueCharacteristics(
                venue_id=venue.venue_id,
                venue_type=venue.venue_type,
                typical_cost_bps=venue.typical_cost_bps,
                typical_fill_rate=venue.typical_fill_rate,
                typical_speed_ms=venue.typical_speed_ms,
                min_order_size=venue.min_order_size,
                max_order_size=venue.max_order_size,
                supports_hidden_orders=venue.supports_hidden_orders,
                supports_iceberg_orders=venue.supports_iceberg_orders,
                market_share_pct=venue.market_share_pct,
                liquidity_score=venue.liquidity_score,
                adverse_selection_score=venue.adverse_selection_score
            )
            
            # Apply market condition adjustments
            performance_adj = market_conditions.get('venue_performance_adjustments', {}).get(venue_id, 1.0)
            updated_venue.typical_cost_bps *= (2.0 - performance_adj)  # Inverse relationship
            updated_venue.typical_fill_rate *= performance_adj
            
            # Adjust for market stress
            stress_factor = market_conditions.get('market_stress', 0.3)
            if venue.venue_type == VenueType.DARK_POOL:
                # Dark pools may have reduced liquidity in stressed markets
                updated_venue.typical_fill_rate *= (1.0 - stress_factor * 0.3)
                updated_venue.liquidity_score *= (1.0 - stress_factor * 0.2)
            
            # Adjust for volatility
            volatility = market_conditions.get('market_volatility', 0.25)
            if volatility > 0.5:  # High volatility
                updated_venue.adverse_selection_score *= 1.2  # Higher adverse selection risk
                if venue.venue_type == VenueType.EXCHANGE:
                    updated_venue.typical_cost_bps *= 1.1  # Higher costs on exchanges
            
            updated_venues[venue_id] = updated_venue
        
        return updated_venues
    
    async def _optimize_single_order_routing(
        self, 
        order,
        venues: Dict[str, VenueCharacteristics],
        market_conditions: Dict[str, Any]
    ) -> RoutingRecommendation:
        """Optimize routing for a single order."""
        
        # Score all venues for this order
        venue_scores = await self._score_venues_for_order(order, venues, market_conditions)
        
        # Select top venues
        top_venues = sorted(venue_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
        selected_venues = top_venues[:self.max_venues_per_order]
        
        # Calculate allocation percentages
        allocations = await self._calculate_venue_allocations(selected_venues, order)
        
        # Generate routing logic
        routing_logic = await self._generate_routing_logic(selected_venues, allocations, order)
        
        # Assess risk
        risk_assessment = await self._assess_routing_risk(selected_venues, allocations)
        
        # Generate contingency plan
        contingency_plan = await self._generate_contingency_plan(selected_venues, order)
        
        # Calculate expected metrics
        expected_cost = self._calculate_weighted_cost(selected_venues, allocations)
        expected_fill_rate = self._calculate_weighted_fill_rate(selected_venues, allocations)
        expected_time = self._calculate_weighted_execution_time(selected_venues, allocations)
        
        return RoutingRecommendation(
            order_id=order.order_id,
            symbol=order.symbol,
            primary_venue=selected_venues[0][0],
            secondary_venues=[venue[0] for venue in selected_venues[1:]],
            allocation_percentages=allocations,
            expected_cost_bps=expected_cost,
            expected_fill_rate=expected_fill_rate,
            expected_execution_time_minutes=expected_time,
            routing_logic=routing_logic,
            risk_assessment=risk_assessment,
            contingency_plan=contingency_plan
        )
    
    async def _score_venues_for_order(
        self, 
        order,
        venues: Dict[str, VenueCharacteristics],
        market_conditions: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Score venues for a specific order."""
        
        venue_scores = {}
        
        for venue_id, venue in venues.items():
            # Check if venue can handle the order
            if order.quantity < venue.min_order_size or order.quantity > venue.max_order_size:
                continue
            
            # Calculate component scores
            cost_score = 1.0 / (venue.typical_cost_bps + 1e-6)  # Lower cost = higher score
            fill_rate_score = venue.typical_fill_rate
            speed_score = 1.0 / (venue.typical_speed_ms + 1e-6)  # Faster = higher score
            liquidity_score = venue.liquidity_score
            adverse_selection_score = 1.0 - venue.adverse_selection_score  # Lower adverse selection = higher score
            
            # Order size compatibility
            size_compatibility = min(1.0, venue.max_order_size / order.quantity)
            
            # Venue type preference based on order characteristics
            type_preference = self._get_venue_type_preference(order, venue)
            
            # Combine scores with weights
            total_score = (
                cost_score * 0.3 +
                fill_rate_score * 0.25 +
                liquidity_score * 0.2 +
                adverse_selection_score * 0.15 +
                size_compatibility * 0.05 +
                type_preference * 0.05
            )
            
            venue_scores[venue_id] = {
                'total_score': total_score,
                'cost_score': cost_score,
                'fill_rate_score': fill_rate_score,
                'speed_score': speed_score,
                'liquidity_score': liquidity_score,
                'adverse_selection_score': adverse_selection_score,
                'size_compatibility': size_compatibility,
                'type_preference': type_preference
            }
        
        return venue_scores
    
    def _get_venue_type_preference(self, order, venue: VenueCharacteristics) -> float:
        """Get venue type preference based on order characteristics."""
        
        # Large orders prefer dark pools
        if order.quantity > 10000:
            if venue.venue_type == VenueType.DARK_POOL:
                return 1.0
            elif venue.venue_type == VenueType.EXCHANGE:
                return 0.7
            else:
                return 0.8
        
        # Medium orders prefer ECNs
        elif order.quantity > 1000:
            if venue.venue_type == VenueType.ECN:
                return 1.0
            elif venue.venue_type == VenueType.EXCHANGE:
                return 0.9
            else:
                return 0.8
        
        # Small orders can use any venue
        else:
            return 0.9
    
    async def _calculate_venue_allocations(
        self, 
        selected_venues: List[Tuple[str, Dict[str, float]]],
        order
    ) -> Dict[str, float]:
        """Calculate allocation percentages for selected venues."""
        
        if not selected_venues:
            return {}
        
        # Calculate weights based on scores
        total_score = sum(venue[1]['total_score'] for venue in selected_venues)
        
        allocations = {}
        remaining_allocation = 100.0
        
        for i, (venue_id, scores) in enumerate(selected_venues):
            if i == len(selected_venues) - 1:  # Last venue gets remaining
                allocation = remaining_allocation
            else:
                # Allocate based on score, but with minimum allocation
                base_allocation = (scores['total_score'] / total_score) * 100
                allocation = max(self.min_allocation_pct, base_allocation)
                allocation = min(allocation, remaining_allocation - 
                               (len(selected_venues) - i - 1) * self.min_allocation_pct)
            
            allocations[venue_id] = allocation
            remaining_allocation -= allocation
        
        return allocations
    
    async def _generate_routing_logic(
        self, 
        selected_venues: List[Tuple[str, Dict[str, float]]],
        allocations: Dict[str, float],
        order
    ) -> str:
        """Generate routing logic description."""
        
        primary_venue = selected_venues[0][0]
        primary_allocation = allocations[primary_venue]
        
        logic_parts = [
            f"Route {primary_allocation:.0f}% to {primary_venue} (primary venue)"
        ]
        
        for venue_id, allocation in list(allocations.items())[1:]:
            logic_parts.append(f"Route {allocation:.0f}% to {venue_id}")
        
        logic_parts.append("Use smart order routing with dynamic rebalancing")
        logic_parts.append("Monitor fill rates and adjust routing if needed")
        
        return "; ".join(logic_parts)
    
    async def _assess_routing_risk(
        self, 
        selected_venues: List[Tuple[str, Dict[str, float]]],
        allocations: Dict[str, float]
    ) -> str:
        """Assess risk of the routing strategy."""
        
        # Calculate concentration risk
        max_allocation = max(allocations.values())
        if max_allocation > 60:
            concentration_risk = "HIGH"
        elif max_allocation > 40:
            concentration_risk = "MEDIUM"
        else:
            concentration_risk = "LOW"
        
        # Assess venue diversity
        venue_types = set()
        for venue_id, _ in selected_venues:
            venue = self.venue_universe[venue_id]
            venue_types.add(venue.venue_type)
        
        if len(venue_types) > 2:
            diversity_risk = "LOW"
        elif len(venue_types) == 2:
            diversity_risk = "MEDIUM"
        else:
            diversity_risk = "HIGH"
        
        return f"Concentration risk: {concentration_risk}, Venue diversity risk: {diversity_risk}"
    
    async def _generate_contingency_plan(
        self, 
        selected_venues: List[Tuple[str, Dict[str, float]]],
        order
    ) -> List[str]:
        """Generate contingency plan for routing failures."""
        
        plan = [
            "If primary venue fails, redistribute to secondary venues",
            "If fill rate drops below 70%, increase allocation to exchanges",
            "If market conditions deteriorate, reduce dark pool allocation",
            "Have backup venues ready for immediate activation"
        ]
        
        # Add venue-specific contingencies
        primary_venue = selected_venues[0][0]
        if self.venue_universe[primary_venue].venue_type == VenueType.DARK_POOL:
            plan.append("If dark pool liquidity dries up, route to lit venues immediately")
        
        return plan
    
    def _calculate_weighted_cost(
        self, 
        selected_venues: List[Tuple[str, Dict[str, float]]],
        allocations: Dict[str, float]
    ) -> float:
        """Calculate weighted average expected cost."""
        
        weighted_cost = 0.0
        total_allocation = sum(allocations.values())
        
        for venue_id, allocation in allocations.items():
            venue = self.venue_universe[venue_id]
            weight = allocation / total_allocation
            weighted_cost += venue.typical_cost_bps * weight
        
        return weighted_cost
    
    def _calculate_weighted_fill_rate(
        self, 
        selected_venues: List[Tuple[str, Dict[str, float]]],
        allocations: Dict[str, float]
    ) -> float:
        """Calculate weighted average expected fill rate."""
        
        weighted_fill_rate = 0.0
        total_allocation = sum(allocations.values())
        
        for venue_id, allocation in allocations.items():
            venue = self.venue_universe[venue_id]
            weight = allocation / total_allocation
            weighted_fill_rate += venue.typical_fill_rate * weight
        
        return weighted_fill_rate
    
    def _calculate_weighted_execution_time(
        self, 
        selected_venues: List[Tuple[str, Dict[str, float]]],
        allocations: Dict[str, float]
    ) -> float:
        """Calculate weighted average execution time."""
        
        # Convert speed from ms to minutes and invert for execution time
        weighted_time = 0.0
        total_allocation = sum(allocations.values())
        
        for venue_id, allocation in allocations.items():
            venue = self.venue_universe[venue_id]
            weight = allocation / total_allocation
            # Estimate execution time based on speed and fill rate
            estimated_time_minutes = (venue.typical_speed_ms / 1000 / 60) / venue.typical_fill_rate
            weighted_time += estimated_time_minutes * weight
        
        return max(0.1, weighted_time)  # Minimum 0.1 minutes
    
    async def _perform_venue_analysis(
        self, 
        venues: Dict[str, VenueCharacteristics],
        request: OptimizationRequest
    ) -> Dict[str, Dict[str, float]]:
        """Perform comprehensive venue analysis."""
        
        analysis = {}
        
        for venue_id, venue in venues.items():
            analysis[venue_id] = {
                'cost_ranking': 0,  # Will be filled below
                'fill_rate_ranking': 0,
                'liquidity_ranking': 0,
                'overall_score': 0,
                'suitability_for_large_orders': 0,
                'suitability_for_small_orders': 0
            }
        
        # Calculate rankings
        venues_list = list(venues.items())
        
        # Cost ranking (lower cost = better ranking)
        cost_sorted = sorted(venues_list, key=lambda x: x[1].typical_cost_bps)
        for i, (venue_id, _) in enumerate(cost_sorted):
            analysis[venue_id]['cost_ranking'] = i + 1
        
        # Fill rate ranking
        fill_rate_sorted = sorted(venues_list, key=lambda x: x[1].typical_fill_rate, reverse=True)
        for i, (venue_id, _) in enumerate(fill_rate_sorted):
            analysis[venue_id]['fill_rate_ranking'] = i + 1
        
        # Liquidity ranking
        liquidity_sorted = sorted(venues_list, key=lambda x: x[1].liquidity_score, reverse=True)
        for i, (venue_id, _) in enumerate(liquidity_sorted):
            analysis[venue_id]['liquidity_ranking'] = i + 1
        
        # Calculate overall scores and suitability
        for venue_id, venue in venues.items():
            # Overall score (weighted combination)
            cost_score = 1.0 / venue.typical_cost_bps
            fill_rate_score = venue.typical_fill_rate
            liquidity_score = venue.liquidity_score
            
            overall_score = (cost_score * 0.4 + fill_rate_score * 0.3 + liquidity_score * 0.3)
            analysis[venue_id]['overall_score'] = overall_score
            
            # Suitability for different order sizes
            if venue.venue_type == VenueType.DARK_POOL:
                analysis[venue_id]['suitability_for_large_orders'] = 0.9
                analysis[venue_id]['suitability_for_small_orders'] = 0.6
            elif venue.venue_type == VenueType.EXCHANGE:
                analysis[venue_id]['suitability_for_large_orders'] = 0.7
                analysis[venue_id]['suitability_for_small_orders'] = 0.9
            else:  # ECN
                analysis[venue_id]['suitability_for_large_orders'] = 0.8
                analysis[venue_id]['suitability_for_small_orders'] = 0.8
        
        return analysis
    
    async def _calculate_routing_improvements(
        self, 
        routing_recommendations: List[RoutingRecommendation],
        request: OptimizationRequest
    ) -> Dict[str, float]:
        """Calculate expected improvements from routing optimization."""
        
        # Baseline metrics (single venue routing)
        baseline_cost_bps = 5.0  # Typical single-venue cost
        baseline_fill_rate = 0.75  # Typical single-venue fill rate
        
        # Calculate optimized metrics
        total_cost = sum(rec.expected_cost_bps for rec in routing_recommendations)
        avg_optimized_cost = total_cost / len(routing_recommendations) if routing_recommendations else baseline_cost_bps
        
        total_fill_rate = sum(rec.expected_fill_rate for rec in routing_recommendations)
        avg_optimized_fill_rate = total_fill_rate / len(routing_recommendations) if routing_recommendations else baseline_fill_rate
        
        return {
            'cost_improvement_bps': baseline_cost_bps - avg_optimized_cost,
            'fill_rate_improvement': avg_optimized_fill_rate - baseline_fill_rate
        }
    
    async def _assess_implementation_complexity(
        self, 
        routing_recommendations: List[RoutingRecommendation]
    ) -> str:
        """Assess the complexity of implementing routing recommendations."""
        
        # Count unique venues
        unique_venues = set()
        total_allocations = 0
        
        for rec in routing_recommendations:
            unique_venues.add(rec.primary_venue)
            unique_venues.update(rec.secondary_venues)
            total_allocations += len(rec.allocation_percentages)
        
        # Assess complexity
        if len(unique_venues) <= 3 and total_allocations <= 6:
            return "LOW"
        elif len(unique_venues) <= 5 and total_allocations <= 12:
            return "MEDIUM"
        else:
            return "HIGH"
    
    async def _generate_monitoring_requirements(
        self, 
        routing_recommendations: List[RoutingRecommendation]
    ) -> List[str]:
        """Generate monitoring requirements for routing strategy."""
        
        requirements = [
            "Monitor fill rates by venue in real-time",
            "Track execution costs vs. expectations",
            "Alert on venue outages or performance degradation",
            "Review routing effectiveness daily",
            "Benchmark against industry routing performance"
        ]
        
        # Add specific requirements based on recommendations
        venues_used = set()
        for rec in routing_recommendations:
            venues_used.add(rec.primary_venue)
            venues_used.update(rec.secondary_venues)
        
        if any(self.venue_universe[v].venue_type == VenueType.DARK_POOL for v in venues_used):
            requirements.append("Monitor dark pool liquidity levels")
            requirements.append("Track adverse selection in dark pools")
        
        if len(venues_used) > 4:
            requirements.append("Implement venue performance dashboard")
            requirements.append("Set up automated rebalancing triggers")
        
        return requirements


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from datetime import datetime
    
    async def test_routing_optimizer():
        """Test the routing optimizer with sample data."""
        
        # Create sample optimization request
        from ...schema.optimization_schema import OptimizationRequest, OrderToOptimize
        
        request = OptimizationRequest(
            request_id="routing_test_001",
            timestamp=datetime.utcnow(),
            orders=[
                OrderToOptimize(
                    order_id="test_order_1",
                    symbol="AAPL",
                    side="BUY",
                    quantity=15000,  # Large order - should prefer dark pools
                    limit_price=150.00,
                    time_in_force="DAY"
                ),
                OrderToOptimize(
                    order_id="test_order_2",
                    symbol="MSFT",
                    side="SELL",
                    quantity=2000,   # Medium order - should prefer ECNs
                    limit_price=340.00,
                    time_in_force="DAY"
                )
            ],
            objective="minimize_cost",
            urgency_level="medium",
            risk_tolerance="moderate"
        )
        
        # Initialize optimizer and run optimization
        optimizer = RoutingOptimizer()
        result = await optimizer.optimize_routing(request)
        
        print("=== Routing Optimization Results ===")
        print(f"Optimization ID: {result.optimization_id}")
        print(f"Expected Cost Improvement: {result.expected_cost_improvement_bps:.2f} bps")
        print(f"Expected Fill Rate Improvement: {result.expected_fill_rate_improvement:.2%}")
        print(f"Implementation Complexity: {result.implementation_complexity}")
        
        print("\n=== Routing Recommendations ===")
        for rec in result.routing_recommendations:
            print(f"\nOrder: {rec.symbol} ({rec.order_id})")
            print(f"Primary Venue: {rec.primary_venue}")
            print(f"Secondary Venues: {rec.secondary_venues}")
            print(f"Allocations: {rec.allocation_percentages}")
            print(f"Expected Cost: {rec.expected_cost_bps:.2f} bps")
            print(f"Expected Fill Rate: {rec.expected_fill_rate:.1%}")
            print(f"Routing Logic: {rec.routing_logic}")
        
        print("\n=== Venue Analysis ===")
        for venue, metrics in result.venue_analysis.items():
            print(f"{venue}: Cost Rank #{metrics['cost_ranking']:.0f}, "
                  f"Fill Rate Rank #{metrics['fill_rate_ranking']:.0f}, "
                  f"Overall Score: {metrics['overall_score']:.3f}")
        
        print("\n=== Monitoring Requirements ===")
        for req in result.monitoring_requirements:
            print(f"- {req}")
    
    # Run the test
    asyncio.run(test_routing_optimizer())
