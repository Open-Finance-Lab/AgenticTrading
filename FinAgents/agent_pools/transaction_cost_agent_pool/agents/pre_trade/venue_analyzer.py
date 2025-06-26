"""
Venue Analyzer Agent

This agent provides comprehensive venue analysis capabilities for optimal
execution venue selection, analyzing costs, liquidity, and execution quality
across different market venues and execution strategies.

Key Features:
- Multi-venue cost comparison and analysis
- Liquidity assessment across venues
- Historical execution quality tracking
- Smart order routing recommendations
- Real-time venue performance monitoring

Author: FinAgent Development Team
License: OpenMDW
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
from decimal import Decimal
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

class VenueCategory(str, Enum):
    """Categories of execution venues."""
    EXCHANGE = "exchange"
    DARK_POOL = "dark_pool"
    ECN = "ecn"
    MARKET_MAKER = "market_maker"
    CROSSING_NETWORK = "crossing_network"
    ALTERNATIVE_TRADING_SYSTEM = "ats"

class LiquidityTier(str, Enum):
    """Liquidity tiers for venue classification."""
    TIER_1 = "tier_1"  # Highest liquidity
    TIER_2 = "tier_2"  # Good liquidity
    TIER_3 = "tier_3"  # Moderate liquidity
    TIER_4 = "tier_4"  # Limited liquidity

@dataclass
class VenueCharacteristics:
    """Comprehensive venue characteristics and capabilities."""
    venue_id: str
    venue_name: str
    venue_category: VenueCategory
    liquidity_tier: LiquidityTier
    
    # Cost structure
    maker_fee: float = 0.0
    taker_fee: float = 0.0
    flat_fee: float = 0.0
    minimum_fee: float = 0.0
    maximum_fee: float = 0.0
    
    # Operational characteristics
    min_order_size: float = 1.0
    max_order_size: Optional[float] = None
    tick_size: float = 0.01
    
    # Timing characteristics
    market_hours_start: str = "09:30"
    market_hours_end: str = "16:00"
    extended_hours: bool = False
    
    # Execution characteristics
    supports_hidden_orders: bool = False
    supports_iceberg_orders: bool = False
    supports_stop_orders: bool = False
    supports_algorithms: List[str] = None
    
    # Performance metrics
    average_fill_rate: float = 0.95
    average_execution_speed: float = 100.0  # milliseconds
    market_share: float = 0.0
    
    # Risk characteristics
    credit_risk_rating: str = "AAA"
    operational_risk_score: float = 1.0
    
    def __post_init__(self):
        if self.supports_algorithms is None:
            self.supports_algorithms = []

@dataclass
class VenuePerformanceMetrics:
    """Performance metrics for venue evaluation."""
    venue_id: str
    evaluation_period: str
    
    # Execution quality metrics
    fill_rate: float
    partial_fill_rate: float
    average_fill_time: float  # seconds
    price_improvement_rate: float
    
    # Cost metrics
    effective_spread: float
    realized_spread: float
    implementation_shortfall: float
    total_cost_bps: float
    
    # Volume metrics
    market_share: float
    daily_volume: float
    order_count: int
    
    # Quality scores
    execution_quality_score: float  # 0-100
    reliability_score: float  # 0-100
    innovation_score: float  # 0-100
    
    # Timestamp
    last_updated: datetime = datetime.utcnow()

class VenueAnalyzer:
    """
    Comprehensive venue analysis agent for optimal execution venue selection.
    
    This agent analyzes execution venues across multiple dimensions including
    cost, liquidity, execution quality, and operational characteristics to
    provide optimal venue selection recommendations.
    """
    
    def __init__(self, agent_id: str = "venue_analyzer"):
        """
        Initialize the Venue Analyzer agent.
        
        Args:
            agent_id: Unique agent identifier
        """
        self.agent_id = agent_id
        self.venues: Dict[str, VenueCharacteristics] = {}
        self.performance_history: Dict[str, List[VenuePerformanceMetrics]] = {}
        self.venue_rankings: Dict[str, Dict[str, float]] = {}
        
        # Initialize default venues
        self._initialize_default_venues()
        
        # Performance tracking
        self.analysis_count = 0
        self.recommendation_accuracy = 0.0
        
        logger.info(f"Venue Analyzer agent initialized: {agent_id}")
    
    def _initialize_default_venues(self):
        """Initialize default venue configurations."""
        try:
            # Major exchanges
            self.venues["NYSE"] = VenueCharacteristics(
                venue_id="NYSE",
                venue_name="New York Stock Exchange",
                venue_category=VenueCategory.EXCHANGE,
                liquidity_tier=LiquidityTier.TIER_1,
                maker_fee=-0.0015,  # Rebate
                taker_fee=0.0030,
                market_hours_start="09:30",
                market_hours_end="16:00",
                extended_hours=True,
                supports_hidden_orders=True,
                supports_iceberg_orders=True,
                supports_stop_orders=True,
                supports_algorithms=["TWAP", "VWAP", "Implementation Shortfall"],
                average_fill_rate=0.98,
                average_execution_speed=95.0,
                market_share=0.25,
                credit_risk_rating="AAA"
            )
            
            self.venues["NASDAQ"] = VenueCharacteristics(
                venue_id="NASDAQ",
                venue_name="NASDAQ Stock Market",
                venue_category=VenueCategory.EXCHANGE,
                liquidity_tier=LiquidityTier.TIER_1,
                maker_fee=-0.0020,
                taker_fee=0.0035,
                extended_hours=True,
                supports_hidden_orders=True,
                supports_iceberg_orders=True,
                supports_stop_orders=True,
                supports_algorithms=["TWAP", "VWAP", "POV", "Implementation Shortfall"],
                average_fill_rate=0.97,
                average_execution_speed=92.0,
                market_share=0.22,
                credit_risk_rating="AAA"
            )
            
            # Dark pools
            self.venues["DARK_POOL_1"] = VenueCharacteristics(
                venue_id="DARK_POOL_1",
                venue_name="Institutional Dark Pool",
                venue_category=VenueCategory.DARK_POOL,
                liquidity_tier=LiquidityTier.TIER_2,
                flat_fee=0.0010,
                min_order_size=100,
                supports_hidden_orders=True,
                supports_iceberg_orders=True,
                supports_algorithms=["TWAP", "VWAP"],
                average_fill_rate=0.85,
                average_execution_speed=150.0,
                market_share=0.08,
                credit_risk_rating="AA+"
            )
            
            # ECNs
            self.venues["BATS"] = VenueCharacteristics(
                venue_id="BATS",
                venue_name="BATS Exchange",
                venue_category=VenueCategory.ECN,
                liquidity_tier=LiquidityTier.TIER_2,
                maker_fee=-0.0025,
                taker_fee=0.0028,
                supports_hidden_orders=True,
                supports_algorithms=["TWAP", "VWAP"],
                average_fill_rate=0.92,
                average_execution_speed=105.0,
                market_share=0.12,
                credit_risk_rating="AA"
            )
            
            logger.info(f"Initialized {len(self.venues)} default venues")
            
        except Exception as e:
            logger.error(f"Venue initialization failed: {str(e)}")
            raise
    
    def analyze_venue_costs(
        self,
        symbol: str,
        quantity: float,
        side: str,
        order_type: str = "market",
        target_venues: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze costs across different venues for a given trade.
        
        Args:
            symbol: Trading symbol
            quantity: Trade quantity
            side: Order side (buy/sell)
            order_type: Order type
            target_venues: Specific venues to analyze (None for all)
            
        Returns:
            Dict[str, Any]: Comprehensive venue cost analysis
        """
        try:
            venues_to_analyze = target_venues or list(self.venues.keys())
            venue_costs = {}
            
            for venue_id in venues_to_analyze:
                if venue_id not in self.venues:
                    logger.warning(f"Venue {venue_id} not found, skipping")
                    continue
                
                venue = self.venues[venue_id]
                cost_analysis = self._calculate_venue_cost(
                    venue, symbol, quantity, side, order_type
                )
                venue_costs[venue_id] = cost_analysis
            
            # Rank venues by cost
            cost_ranking = self._rank_venues_by_cost(venue_costs)
            
            # Generate recommendations
            recommendations = self._generate_cost_recommendations(
                venue_costs, symbol, quantity, side
            )
            
            result = {
                "analysis_id": f"VENUE_COST_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "trade_specification": {
                    "symbol": symbol,
                    "quantity": quantity,
                    "side": side,
                    "order_type": order_type
                },
                "venue_costs": venue_costs,
                "cost_ranking": cost_ranking,
                "recommendations": recommendations,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "venues_analyzed": len(venue_costs)
            }
            
            self.analysis_count += 1
            logger.info(f"Analyzed costs for {len(venue_costs)} venues for {symbol}")
            
            return result
            
        except Exception as e:
            logger.error(f"Venue cost analysis failed: {str(e)}")
            raise
    
    def _calculate_venue_cost(
        self,
        venue: VenueCharacteristics,
        symbol: str,
        quantity: float,
        side: str,
        order_type: str
    ) -> Dict[str, Any]:
        """Calculate comprehensive cost for a specific venue."""
        try:
            # Estimate trade value (using placeholder price)
            estimated_price = 100.0  # Would be replaced with real price
            trade_value = quantity * estimated_price
            
            # Calculate fees based on venue structure
            if venue.flat_fee > 0:
                # Flat fee structure
                fee_cost = trade_value * venue.flat_fee / 10000
                fee_structure = "flat"
            else:
                # Maker/taker structure
                if order_type == "limit":
                    # Assume limit orders are makers
                    fee_cost = trade_value * abs(venue.maker_fee) / 10000
                    if venue.maker_fee < 0:
                        fee_cost = -fee_cost  # Rebate
                    fee_structure = "maker"
                else:
                    # Market orders are takers
                    fee_cost = trade_value * venue.taker_fee / 10000
                    fee_structure = "taker"
            
            # Apply minimum/maximum fee constraints
            if venue.minimum_fee > 0:
                fee_cost = max(fee_cost, venue.minimum_fee)
            if venue.maximum_fee > 0:
                fee_cost = min(fee_cost, venue.maximum_fee)
            
            # Calculate basis points
            fee_bps = (fee_cost / trade_value) * 10000
            
            # Estimate execution quality impact
            execution_quality_adjustment = self._estimate_execution_quality_cost(
                venue, quantity, trade_value
            )
            
            # Total cost
            total_cost = fee_cost + execution_quality_adjustment
            total_cost_bps = (total_cost / trade_value) * 10000
            
            return {
                "venue_id": venue.venue_id,
                "venue_name": venue.venue_name,
                "venue_category": venue.venue_category.value,
                "fee_structure": fee_structure,
                "costs": {
                    "fee_cost": fee_cost,
                    "fee_cost_bps": fee_bps,
                    "execution_quality_adjustment": execution_quality_adjustment,
                    "execution_quality_adjustment_bps": (execution_quality_adjustment / trade_value) * 10000,
                    "total_cost": total_cost,
                    "total_cost_bps": total_cost_bps
                },
                "execution_characteristics": {
                    "expected_fill_rate": venue.average_fill_rate,
                    "expected_execution_speed": venue.average_execution_speed,
                    "supports_order_type": self._supports_order_type(venue, order_type),
                    "market_share": venue.market_share
                },
                "risk_factors": self._identify_venue_risks(venue, quantity)
            }
            
        except Exception as e:
            logger.error(f"Venue cost calculation failed for {venue.venue_id}: {str(e)}")
            raise
    
    def _estimate_execution_quality_cost(
        self,
        venue: VenueCharacteristics,
        quantity: float,
        trade_value: float
    ) -> float:
        """Estimate cost impact from execution quality differences."""
        try:
            # Base execution quality cost (simplified model)
            base_cost = 0.0
            
            # Adjust for fill rate
            if venue.average_fill_rate < 0.95:
                # Cost of partial fills
                partial_fill_penalty = (0.95 - venue.average_fill_rate) * trade_value * 0.0005
                base_cost += partial_fill_penalty
            
            # Adjust for execution speed
            if venue.average_execution_speed > 120.0:  # milliseconds
                # Cost of slower execution
                speed_penalty = (venue.average_execution_speed - 120.0) / 1000.0 * trade_value * 0.0001
                base_cost += speed_penalty
            
            # Adjust for market share (liquidity proxy)
            if venue.market_share < 0.05:
                # Cost of lower liquidity
                liquidity_penalty = (0.05 - venue.market_share) * trade_value * 0.001
                base_cost += liquidity_penalty
            
            return base_cost
            
        except Exception as e:
            logger.warning(f"Execution quality cost estimation failed: {str(e)}")
            return 0.0
    
    def _supports_order_type(self, venue: VenueCharacteristics, order_type: str) -> bool:
        """Check if venue supports the specified order type."""
        if order_type == "market":
            return True
        elif order_type == "limit":
            return True
        elif order_type == "stop" or order_type == "stop_limit":
            return venue.supports_stop_orders
        elif order_type == "iceberg":
            return venue.supports_iceberg_orders
        else:
            return False
    
    def _identify_venue_risks(
        self,
        venue: VenueCharacteristics,
        quantity: float
    ) -> List[str]:
        """Identify risk factors for venue execution."""
        risks = []
        
        # Size constraints
        if quantity < venue.min_order_size:
            risks.append(f"Order size below minimum ({venue.min_order_size})")
        
        if venue.max_order_size and quantity > venue.max_order_size:
            risks.append(f"Order size above maximum ({venue.max_order_size})")
        
        # Liquidity risks
        if venue.liquidity_tier in [LiquidityTier.TIER_3, LiquidityTier.TIER_4]:
            risks.append("Limited liquidity venue")
        
        # Credit risks
        if venue.credit_risk_rating not in ["AAA", "AA+", "AA"]:
            risks.append(f"Credit risk concern ({venue.credit_risk_rating})")
        
        # Operational risks
        if venue.operational_risk_score > 2.0:
            risks.append("Elevated operational risk")
        
        return risks
    
    def _rank_venues_by_cost(self, venue_costs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank venues by total cost (lowest to highest)."""
        try:
            ranking = []
            
            for venue_id, cost_data in venue_costs.items():
                ranking.append({
                    "venue_id": venue_id,
                    "venue_name": cost_data["venue_name"],
                    "total_cost_bps": cost_data["costs"]["total_cost_bps"],
                    "fee_cost_bps": cost_data["costs"]["fee_cost_bps"],
                    "execution_quality_score": cost_data["execution_characteristics"]["expected_fill_rate"] * 100,
                    "risk_factor_count": len(cost_data["risk_factors"])
                })
            
            # Sort by total cost
            ranking.sort(key=lambda x: x["total_cost_bps"])
            
            # Add rank
            for i, item in enumerate(ranking):
                item["rank"] = i + 1
            
            return ranking
            
        except Exception as e:
            logger.error(f"Venue ranking failed: {str(e)}")
            return []
    
    def _generate_cost_recommendations(
        self,
        venue_costs: Dict[str, Any],
        symbol: str,
        quantity: float,
        side: str
    ) -> List[str]:
        """Generate venue selection recommendations."""
        recommendations = []
        
        try:
            # Find lowest cost venue
            lowest_cost_venue = min(
                venue_costs.items(),
                key=lambda x: x[1]["costs"]["total_cost_bps"]
            )
            
            recommendations.append(
                f"Lowest cost venue: {lowest_cost_venue[1]['venue_name']} "
                f"({lowest_cost_venue[1]['costs']['total_cost_bps']:.2f} bps)"
            )
            
            # Check for dark pool suitability
            large_order_threshold = 10000  # shares
            if quantity > large_order_threshold:
                dark_pools = [
                    (venue_id, data) for venue_id, data in venue_costs.items()
                    if data["venue_category"] == "dark_pool"
                ]
                
                if dark_pools:
                    best_dark_pool = min(dark_pools, key=lambda x: x[1]["costs"]["total_cost_bps"])
                    recommendations.append(
                        f"For large orders, consider dark pool: {best_dark_pool[1]['venue_name']} "
                        f"({best_dark_pool[1]['costs']['total_cost_bps']:.2f} bps)"
                    )
            
            # Speed recommendations
            fast_venues = [
                (venue_id, data) for venue_id, data in venue_costs.items()
                if data["execution_characteristics"]["expected_execution_speed"] < 100.0
            ]
            
            if fast_venues:
                fastest_venue = min(fast_venues, key=lambda x: x[1]["execution_characteristics"]["expected_execution_speed"])
                recommendations.append(
                    f"Fastest execution: {fastest_venue[1]['venue_name']} "
                    f"({fastest_venue[1]['execution_characteristics']['expected_execution_speed']:.1f}ms)"
                )
            
            # Risk-adjusted recommendations
            low_risk_venues = [
                (venue_id, data) for venue_id, data in venue_costs.items()
                if len(data["risk_factors"]) == 0
            ]
            
            if low_risk_venues:
                best_low_risk = min(low_risk_venues, key=lambda x: x[1]["costs"]["total_cost_bps"])
                recommendations.append(
                    f"Lowest risk option: {best_low_risk[1]['venue_name']} "
                    f"({best_low_risk[1]['costs']['total_cost_bps']:.2f} bps, no risk factors)"
                )
            
        except Exception as e:
            logger.warning(f"Recommendation generation failed: {str(e)}")
            recommendations.append("Unable to generate specific recommendations")
        
        return recommendations
    
    def find_optimal_venues(
        self,
        symbol: str,
        quantity: float,
        side: str,
        optimization_criteria: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Find optimal venues based on multi-criteria optimization.
        
        Args:
            symbol: Trading symbol
            quantity: Trade quantity
            side: Order side
            optimization_criteria: Weights for different criteria
            
        Returns:
            Dict[str, Any]: Optimal venue recommendations
        """
        try:
            # Default optimization criteria
            if optimization_criteria is None:
                optimization_criteria = {
                    "cost": 0.4,
                    "execution_quality": 0.3,
                    "speed": 0.2,
                    "risk": 0.1
                }
            
            # Get venue cost analysis
            cost_analysis = self.analyze_venue_costs(symbol, quantity, side)
            venue_costs = cost_analysis["venue_costs"]
            
            # Calculate composite scores
            venue_scores = {}
            for venue_id, data in venue_costs.items():
                score = self._calculate_composite_score(data, optimization_criteria)
                venue_scores[venue_id] = {
                    "venue_name": data["venue_name"],
                    "composite_score": score,
                    "cost_bps": data["costs"]["total_cost_bps"],
                    "execution_quality": data["execution_characteristics"]["expected_fill_rate"],
                    "speed": data["execution_characteristics"]["expected_execution_speed"],
                    "risk_factors": len(data["risk_factors"])
                }
            
            # Rank by composite score (higher is better)
            ranked_venues = sorted(
                venue_scores.items(),
                key=lambda x: x[1]["composite_score"],
                reverse=True
            )
            
            # Generate optimization report
            optimization_report = {
                "optimization_id": f"OPT_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "trade_specification": {
                    "symbol": symbol,
                    "quantity": quantity,
                    "side": side
                },
                "optimization_criteria": optimization_criteria,
                "optimal_venue": {
                    "venue_id": ranked_venues[0][0],
                    "venue_name": ranked_venues[0][1]["venue_name"],
                    "composite_score": ranked_venues[0][1]["composite_score"],
                    "expected_cost_bps": ranked_venues[0][1]["cost_bps"]
                },
                "venue_rankings": [
                    {
                        "rank": i + 1,
                        "venue_id": venue_id,
                        "venue_name": data["venue_name"],
                        "composite_score": data["composite_score"],
                        "cost_bps": data["cost_bps"],
                        "execution_quality": data["execution_quality"],
                        "speed_ms": data["speed"],
                        "risk_factors": data["risk_factors"]
                    }
                    for i, (venue_id, data) in enumerate(ranked_venues)
                ],
                "recommendations": self._generate_optimization_recommendations(ranked_venues),
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
            return optimization_report
            
        except Exception as e:
            logger.error(f"Optimal venue finding failed: {str(e)}")
            raise
    
    def _calculate_composite_score(
        self,
        venue_data: Dict[str, Any],
        criteria: Dict[str, float]
    ) -> float:
        """Calculate composite score based on multiple criteria."""
        try:
            score = 0.0
            
            # Cost score (lower cost = higher score)
            cost_bps = venue_data["costs"]["total_cost_bps"]
            cost_score = max(0, 100 - cost_bps)  # Simplified scoring
            score += cost_score * criteria.get("cost", 0.0)
            
            # Execution quality score
            fill_rate = venue_data["execution_characteristics"]["expected_fill_rate"]
            quality_score = fill_rate * 100
            score += quality_score * criteria.get("execution_quality", 0.0)
            
            # Speed score (faster = higher score)
            speed_ms = venue_data["execution_characteristics"]["expected_execution_speed"]
            speed_score = max(0, 200 - speed_ms)  # Simplified scoring
            score += speed_score * criteria.get("speed", 0.0)
            
            # Risk score (fewer risks = higher score)
            risk_count = len(venue_data["risk_factors"])
            risk_score = max(0, 100 - risk_count * 20)
            score += risk_score * criteria.get("risk", 0.0)
            
            return score
            
        except Exception as e:
            logger.warning(f"Composite score calculation failed: {str(e)}")
            return 0.0
    
    def _generate_optimization_recommendations(
        self,
        ranked_venues: List[Tuple[str, Dict[str, Any]]]
    ) -> List[str]:
        """Generate recommendations based on optimization results."""
        recommendations = []
        
        if not ranked_venues:
            return ["No venues available for optimization"]
        
        # Top venue recommendation
        top_venue = ranked_venues[0]
        recommendations.append(
            f"Recommended venue: {top_venue[1]['venue_name']} "
            f"(score: {top_venue[1]['composite_score']:.1f}, "
            f"cost: {top_venue[1]['cost_bps']:.2f} bps)"
        )
        
        # Alternative recommendation
        if len(ranked_venues) > 1:
            alt_venue = ranked_venues[1]
            recommendations.append(
                f"Alternative venue: {alt_venue[1]['venue_name']} "
                f"(score: {alt_venue[1]['composite_score']:.1f}, "
                f"cost: {alt_venue[1]['cost_bps']:.2f} bps)"
            )
        
        # Special considerations
        if top_venue[1]['risk_factors'] > 0:
            recommendations.append(
                f"Note: Recommended venue has {top_venue[1]['risk_factors']} risk factor(s) to consider"
            )
        
        return recommendations
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status information."""
        return {
            "agent_id": self.agent_id,
            "status": "active",
            "registered_venues": len(self.venues),
            "venue_categories": list(set(venue.venue_category.value for venue in self.venues.values())),
            "analyses_performed": self.analysis_count,
            "capabilities": [
                "Multi-venue cost analysis",
                "Execution quality assessment",
                "Risk factor identification",
                "Multi-criteria optimization",
                "Smart routing recommendations"
            ],
            "supported_order_types": [
                "market", "limit", "stop", "stop_limit", "iceberg"
            ],
            "last_updated": datetime.utcnow().isoformat()
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the venue analyzer
    analyzer = VenueAnalyzer()
    
    try:
        # Test venue cost analysis
        cost_analysis = analyzer.analyze_venue_costs(
            symbol="AAPL",
            quantity=5000,
            side="buy",
            order_type="market"
        )
        
        print("Venue Cost Analysis:")
        print(f"Venues analyzed: {cost_analysis['venues_analyzed']}")
        print("\nCost Rankings:")
        for venue in cost_analysis['cost_ranking'][:3]:
            print(f"  {venue['rank']}. {venue['venue_name']}: {venue['total_cost_bps']:.2f} bps")
        
        print("\nRecommendations:")
        for rec in cost_analysis['recommendations']:
            print(f"  - {rec}")
        
        # Test optimal venue finding
        print("\n" + "="*50)
        optimization = analyzer.find_optimal_venues(
            symbol="AAPL",
            quantity=5000,
            side="buy",
            optimization_criteria={"cost": 0.5, "execution_quality": 0.3, "speed": 0.2}
        )
        
        print("Optimal Venue Analysis:")
        optimal = optimization['optimal_venue']
        print(f"Optimal venue: {optimal['venue_name']}")
        print(f"Composite score: {optimal['composite_score']:.1f}")
        print(f"Expected cost: {optimal['expected_cost_bps']:.2f} bps")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
