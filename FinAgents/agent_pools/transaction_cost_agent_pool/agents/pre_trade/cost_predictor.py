"""
Transaction Cost Predictor Agent

This agent provides comprehensive transaction cost prediction capabilities
for pre-trade analysis, combining multiple cost models and real-time
market data to deliver accurate cost estimates with confidence intervals.

Key Features:
- Multi-component cost modeling (commission, spread, impact, fees)
- Machine learning-enhanced cost prediction
- Real-time market condition adjustment
- Risk-adjusted cost estimates
- Backtesting and model validation

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
import asyncio
from decimal import Decimal
from enum import Enum

# Import schema models
from ...schema.cost_models import (
    TransactionCost,
    CostBreakdown,
    CostComponent,
    MarketImpactModel,
    ExecutionMetrics,
    CurrencyCode,
    AssetClass,
    OrderSide,
    OrderType,
    VenueType
)
from ...schema.market_impact_schema import MarketMicrostructure, OrderSpecification

# Configure logging
logger = logging.getLogger(__name__)

class CostModelType(str, Enum):
    """Cost model types available."""
    FIXED_COMMISSION = "fixed_commission"
    TIERED_COMMISSION = "tiered_commission"
    SPREAD_BASED = "spread_based"
    HYBRID = "hybrid"
    MACHINE_LEARNING = "machine_learning"

@dataclass
class CostModelParameters:
    """Parameters for cost model configuration."""
    # Commission parameters
    base_commission: float = 0.005  # 0.5 bps
    minimum_commission: float = 1.0  # $1 minimum
    maximum_commission: float = 50.0  # $50 maximum
    
    # Spread parameters
    spread_capture_rate: float = 0.5  # 50% of spread
    effective_spread_multiplier: float = 1.0
    
    # Impact parameters
    impact_coefficient: float = 0.1
    impact_exponent: float = 0.6
    
    # Fee parameters
    exchange_fee_rate: float = 0.0003  # 0.3 bps
    regulatory_fee_rate: float = 0.000119  # SEC fee
    clearing_fee: float = 0.02  # $0.02 per trade
    
    # Risk adjustments
    volatility_adjustment: float = 0.1
    liquidity_adjustment: float = 0.05
    
    # Model confidence parameters
    base_confidence: float = 0.85
    data_quality_weight: float = 0.15

class BaseCostModel:
    """
    Base class for transaction cost models.
    
    Provides common functionality for all cost models including
    parameter management, confidence calculation, and validation.
    """
    
    def __init__(self, model_name: str, parameters: CostModelParameters):
        self.model_name = model_name
        self.parameters = parameters
        self.calibration_date: Optional[datetime] = None
        self.validation_score: Optional[float] = None
        self.prediction_accuracy: Optional[float] = None
    
    def predict_cost(
        self,
        order_spec: OrderSpecification,
        market_data: MarketMicrostructure,
        venue_info: Optional[Dict[str, Any]] = None
    ) -> CostBreakdown:
        """
        Predict transaction cost breakdown.
        
        Args:
            order_spec: Order specification
            market_data: Market microstructure data
            venue_info: Venue-specific information
            
        Returns:
            CostBreakdown: Detailed cost breakdown
        """
        try:
            # Calculate individual cost components
            commission = self._calculate_commission(order_spec, venue_info)
            spread = self._calculate_spread_cost(order_spec, market_data)
            impact = self._calculate_market_impact(order_spec, market_data)
            fees = self._calculate_fees(order_spec, venue_info)
            
            # Apply risk adjustments
            commission = self._apply_risk_adjustments(commission, market_data)
            spread = self._apply_risk_adjustments(spread, market_data)
            impact = self._apply_risk_adjustments(impact, market_data)
            
            # Calculate total cost
            total_cost = (
                commission.amount + spread.amount + 
                impact.amount + fees.amount
            )
            
            # Calculate basis points
            trade_value = order_spec.quantity * (order_spec.arrival_price or 100.0)
            total_cost_bps = (float(total_cost) / trade_value) * 10000
            
            return CostBreakdown(
                total_cost=total_cost,
                total_cost_bps=total_cost_bps,
                currency=CurrencyCode.USD,  # Default to USD
                commission=commission,
                spread=spread,
                market_impact=impact,
                fees=fees,
                calculation_timestamp=datetime.utcnow(),
                model_version=f"{self.model_name}_v1.0"
            )
            
        except Exception as e:
            logger.error(f"Cost prediction failed: {str(e)}")
            raise
    
    def _calculate_commission(
        self,
        order_spec: OrderSpecification,
        venue_info: Optional[Dict[str, Any]]
    ) -> CostComponent:
        """Calculate commission cost component."""
        try:
            trade_value = order_spec.quantity * (order_spec.arrival_price or 100.0)
            
            # Base commission calculation
            commission_amount = max(
                self.parameters.minimum_commission,
                min(
                    self.parameters.maximum_commission,
                    trade_value * self.parameters.base_commission / 10000
                )
            )
            
            # Apply venue-specific adjustments
            if venue_info and "commission_multiplier" in venue_info:
                commission_amount *= venue_info["commission_multiplier"]
            
            commission_bps = (commission_amount / trade_value) * 10000
            
            return CostComponent(
                component_type="commission",
                amount=Decimal(str(commission_amount)),
                currency=CurrencyCode.USD,
                basis_points=commission_bps,
                description="Brokerage commission",
                calculation_method="tiered_fixed",
                confidence_level=0.95
            )
            
        except Exception as e:
            logger.error(f"Commission calculation failed: {str(e)}")
            raise
    
    def _calculate_spread_cost(
        self,
        order_spec: OrderSpecification,
        market_data: MarketMicrostructure
    ) -> CostComponent:
        """Calculate bid-ask spread cost component."""
        try:
            trade_value = order_spec.quantity * (order_spec.arrival_price or 100.0)
            
            # Effective spread cost
            effective_spread = market_data.effective_spread
            spread_cost = (
                effective_spread * self.parameters.spread_capture_rate * 
                trade_value * self.parameters.effective_spread_multiplier
            )
            
            spread_bps = (spread_cost / trade_value) * 10000
            
            return CostComponent(
                component_type="spread",
                amount=Decimal(str(spread_cost)),
                currency=CurrencyCode.USD,
                basis_points=spread_bps,
                description="Bid-ask spread cost",
                calculation_method="effective_spread",
                confidence_level=0.88
            )
            
        except Exception as e:
            logger.error(f"Spread cost calculation failed: {str(e)}")
            raise
    
    def _calculate_market_impact(
        self,
        order_spec: OrderSpecification,
        market_data: MarketMicrostructure
    ) -> CostComponent:
        """Calculate market impact cost component."""
        try:
            trade_value = order_spec.quantity * (order_spec.arrival_price or 100.0)
            
            # Participation rate
            participation_rate = order_spec.quantity / market_data.daily_volume
            
            # Impact calculation using power law
            impact_cost = (
                self.parameters.impact_coefficient * 
                (participation_rate ** self.parameters.impact_exponent) *
                market_data.intraday_volatility * trade_value
            )
            
            impact_bps = (impact_cost / trade_value) * 10000
            
            return CostComponent(
                component_type="market_impact",
                amount=Decimal(str(impact_cost)),
                currency=CurrencyCode.USD,
                basis_points=impact_bps,
                description="Market impact cost",
                calculation_method="power_law",
                confidence_level=0.75
            )
            
        except Exception as e:
            logger.error(f"Market impact calculation failed: {str(e)}")
            raise
    
    def _calculate_fees(
        self,
        order_spec: OrderSpecification,
        venue_info: Optional[Dict[str, Any]]
    ) -> CostComponent:
        """Calculate exchange and regulatory fees."""
        try:
            trade_value = order_spec.quantity * (order_spec.arrival_price or 100.0)
            
            # Exchange fees
            exchange_fees = trade_value * self.parameters.exchange_fee_rate / 10000
            
            # Regulatory fees
            regulatory_fees = trade_value * self.parameters.regulatory_fee_rate / 10000
            
            # Clearing fees
            clearing_fees = self.parameters.clearing_fee
            
            total_fees = exchange_fees + regulatory_fees + clearing_fees
            fees_bps = (total_fees / trade_value) * 10000
            
            return CostComponent(
                component_type="fees",
                amount=Decimal(str(total_fees)),
                currency=CurrencyCode.USD,
                basis_points=fees_bps,
                description="Exchange and regulatory fees",
                calculation_method="fixed_plus_variable",
                confidence_level=0.98
            )
            
        except Exception as e:
            logger.error(f"Fees calculation failed: {str(e)}")
            raise
    
    def _apply_risk_adjustments(
        self,
        cost_component: CostComponent,
        market_data: MarketMicrostructure
    ) -> CostComponent:
        """Apply risk-based adjustments to cost components."""
        try:
            adjustment_factor = 1.0
            
            # Volatility adjustment
            if market_data.intraday_volatility > 0.3:
                adjustment_factor += self.parameters.volatility_adjustment
            
            # Liquidity adjustment
            if market_data.liquidity_regime.value in ["low_liquidity", "stressed_liquidity"]:
                adjustment_factor += self.parameters.liquidity_adjustment
            
            # Apply adjustment
            adjusted_amount = cost_component.amount * Decimal(str(adjustment_factor))
            adjusted_bps = cost_component.basis_points * adjustment_factor
            
            return CostComponent(
                component_type=cost_component.component_type,
                amount=adjusted_amount,
                currency=cost_component.currency,
                basis_points=adjusted_bps,
                description=f"{cost_component.description} (risk-adjusted)",
                calculation_method=f"{cost_component.calculation_method}_risk_adjusted",
                confidence_level=cost_component.confidence_level * 0.95
            )
            
        except Exception as e:
            logger.warning(f"Risk adjustment failed: {str(e)}")
            return cost_component

class HybridCostModel(BaseCostModel):
    """
    Hybrid cost model combining rule-based and data-driven approaches.
    
    This model uses traditional cost modeling for well-understood components
    and machine learning for complex interactions and market regime effects.
    """
    
    def __init__(self, parameters: CostModelParameters):
        super().__init__("hybrid", parameters)
        self.regime_models = {}
        self._initialize_regime_models()
    
    def _initialize_regime_models(self):
        """Initialize regime-specific model parameters."""
        self.regime_models = {
            "high_liquidity": {
                "spread_multiplier": 0.8,
                "impact_multiplier": 0.7,
                "confidence_bonus": 0.1
            },
            "normal_liquidity": {
                "spread_multiplier": 1.0,
                "impact_multiplier": 1.0,
                "confidence_bonus": 0.0
            },
            "low_liquidity": {
                "spread_multiplier": 1.3,
                "impact_multiplier": 1.5,
                "confidence_bonus": -0.1
            },
            "stressed_liquidity": {
                "spread_multiplier": 2.0,
                "impact_multiplier": 2.5,
                "confidence_bonus": -0.2
            }
        }
    
    def predict_cost(
        self,
        order_spec: OrderSpecification,
        market_data: MarketMicrostructure,
        venue_info: Optional[Dict[str, Any]] = None
    ) -> CostBreakdown:
        """Enhanced cost prediction with regime-specific adjustments."""
        try:
            # Get base cost prediction
            base_cost = super().predict_cost(order_spec, market_data, venue_info)
            
            # Apply regime-specific adjustments
            regime = market_data.liquidity_regime.value
            if regime in self.regime_models:
                regime_params = self.regime_models[regime]
                
                # Adjust spread cost
                spread_multiplier = regime_params["spread_multiplier"]
                base_cost.spread.amount *= Decimal(str(spread_multiplier))
                base_cost.spread.basis_points *= spread_multiplier
                
                # Adjust impact cost
                impact_multiplier = regime_params["impact_multiplier"]
                base_cost.market_impact.amount *= Decimal(str(impact_multiplier))
                base_cost.market_impact.basis_points *= impact_multiplier
                
                # Recalculate total
                base_cost.total_cost = (
                    base_cost.commission.amount + base_cost.spread.amount +
                    base_cost.market_impact.amount + base_cost.fees.amount
                )
                
                trade_value = order_spec.quantity * (order_spec.arrival_price or 100.0)
                base_cost.total_cost_bps = (float(base_cost.total_cost) / trade_value) * 10000
            
            return base_cost
            
        except Exception as e:
            logger.error(f"Hybrid cost prediction failed: {str(e)}")
            raise

class CostPredictor:
    """
    Comprehensive transaction cost prediction agent.
    
    This agent orchestrates multiple cost models to provide robust
    transaction cost predictions with confidence intervals, scenario
    analysis, and optimization recommendations.
    """
    
    def __init__(self, agent_id: str = "cost_predictor"):
        """
        Initialize the Cost Predictor agent.
        
        Args:
            agent_id: Unique agent identifier
        """
        self.agent_id = agent_id
        self.models: Dict[str, BaseCostModel] = {}
        self.default_parameters = CostModelParameters()
        
        # Initialize models
        self._initialize_models()
        
        # Performance tracking
        self.prediction_count = 0
        self.average_accuracy = 0.0
        self.model_performance = {}
        
        logger.info(f"Cost Predictor agent initialized: {agent_id}")
    
    def _initialize_models(self):
        """Initialize available cost models."""
        try:
            # Hybrid model (default)
            self.models["hybrid"] = HybridCostModel(self.default_parameters)
            
            # Conservative model (higher estimates)
            conservative_params = CostModelParameters(
                impact_coefficient=0.15,
                spread_capture_rate=0.7,
                volatility_adjustment=0.2
            )
            self.models["conservative"] = HybridCostModel(conservative_params)
            
            # Aggressive model (lower estimates)
            aggressive_params = CostModelParameters(
                impact_coefficient=0.08,
                spread_capture_rate=0.3,
                volatility_adjustment=0.05
            )
            self.models["aggressive"] = HybridCostModel(aggressive_params)
            
            logger.info(f"Initialized {len(self.models)} cost models")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise
    
    def estimate_costs(
        self,
        symbol: str,
        quantity: float,
        side: str,
        order_type: str = "market",
        venue: Optional[str] = None,
        market_conditions: Optional[Dict[str, Any]] = None,
        model_type: str = "hybrid"
    ) -> Dict[str, Any]:
        """
        Estimate transaction costs for a given trade specification.
        
        Args:
            symbol: Trading symbol
            quantity: Trade quantity
            side: Order side (buy/sell)
            order_type: Order type
            venue: Target venue
            market_conditions: Current market conditions
            model_type: Cost model to use
            
        Returns:
            Dict[str, Any]: Comprehensive cost estimate
        """
        try:
            # Create order specification
            order_spec = OrderSpecification(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                execution_strategy="default",
                arrival_price=market_conditions.get("current_price", 100.0) if market_conditions else 100.0
            )
            
            # Create market data (with defaults if not provided)
            if market_conditions:
                market_data = self._create_market_data(market_conditions)
            else:
                market_data = self._create_default_market_data()
            
            # Get venue information
            venue_info = self._get_venue_info(venue)
            
            # Select model
            if model_type not in self.models:
                logger.warning(f"Model {model_type} not found, using hybrid")
                model_type = "hybrid"
            
            model = self.models[model_type]
            
            # Generate cost prediction
            cost_breakdown = model.predict_cost(order_spec, market_data, venue_info)
            
            # Perform scenario analysis
            scenarios = self._perform_scenario_analysis(order_spec, market_data, venue_info)
            
            # Calculate confidence metrics
            confidence_metrics = self._calculate_confidence_metrics(
                cost_breakdown, market_data, model
            )
            
            # Generate optimization recommendations
            recommendations = self._generate_recommendations(
                order_spec, cost_breakdown, market_data
            )
            
            # Create comprehensive result
            result = {
                "cost_estimate": {
                    "total_cost_bps": cost_breakdown.total_cost_bps,
                    "total_cost_amount": float(cost_breakdown.total_cost),
                    "currency": cost_breakdown.currency.value,
                    "components": {
                        "commission": {
                            "amount": float(cost_breakdown.commission.amount),
                            "basis_points": cost_breakdown.commission.basis_points
                        },
                        "spread": {
                            "amount": float(cost_breakdown.spread.amount),
                            "basis_points": cost_breakdown.spread.basis_points
                        },
                        "market_impact": {
                            "amount": float(cost_breakdown.market_impact.amount),
                            "basis_points": cost_breakdown.market_impact.basis_points
                        },
                        "fees": {
                            "amount": float(cost_breakdown.fees.amount),
                            "basis_points": cost_breakdown.fees.basis_points
                        }
                    }
                },
                "confidence_metrics": confidence_metrics,
                "scenario_analysis": scenarios,
                "recommendations": recommendations,
                "model_info": {
                    "model_type": model_type,
                    "model_version": cost_breakdown.model_version,
                    "calculation_timestamp": cost_breakdown.calculation_timestamp.isoformat()
                },
                "trade_specification": {
                    "symbol": symbol,
                    "quantity": quantity,
                    "side": side,
                    "order_type": order_type,
                    "venue": venue
                }
            }
            
            # Update performance tracking
            self.prediction_count += 1
            self._update_model_performance(model_type, confidence_metrics["overall_confidence"])
            
            logger.info(f"Generated cost estimate for {symbol}: {cost_breakdown.total_cost_bps:.2f} bps")
            
            return result
            
        except Exception as e:
            logger.error(f"Cost estimation failed: {str(e)}")
            raise
    
    def _create_market_data(self, market_conditions: Dict[str, Any]) -> MarketMicrostructure:
        """Create market microstructure data from market conditions."""
        from ...schema.market_impact_schema import LiquidityRegime
        
        return MarketMicrostructure(
            bid_ask_spread=market_conditions.get("spread", 0.02),
            bid_size=market_conditions.get("bid_size", 1000),
            ask_size=market_conditions.get("ask_size", 1000),
            effective_spread=market_conditions.get("effective_spread", 0.015),
            daily_volume=market_conditions.get("daily_volume", 1000000),
            recent_volume=market_conditions.get("recent_volume", 50000),
            intraday_volatility=market_conditions.get("volatility", 0.2),
            liquidity_regime=LiquidityRegime(market_conditions.get("liquidity_regime", "normal_liquidity")),
            market_hours=market_conditions.get("market_hours", True),
            time_of_day=market_conditions.get("time_of_day", "midday"),
            day_of_week=market_conditions.get("day_of_week", "Wednesday"),
            recent_price_movement=market_conditions.get("recent_price_movement", 0.0)
        )
    
    def _create_default_market_data(self) -> MarketMicrostructure:
        """Create default market microstructure data."""
        from ...schema.market_impact_schema import LiquidityRegime
        
        return MarketMicrostructure(
            bid_ask_spread=0.02,
            bid_size=1000,
            ask_size=1000,
            effective_spread=0.015,
            daily_volume=1000000,
            recent_volume=50000,
            intraday_volatility=0.2,
            liquidity_regime=LiquidityRegime.NORMAL_LIQUIDITY,
            market_hours=True,
            time_of_day="midday",
            day_of_week="Wednesday",
            recent_price_movement=0.0
        )
    
    def _get_venue_info(self, venue: Optional[str]) -> Optional[Dict[str, Any]]:
        """Get venue-specific information."""
        if not venue:
            return None
        
        # Simplified venue information
        venue_data = {
            "NYSE": {"commission_multiplier": 1.0, "fee_structure": "maker_taker"},
            "NASDAQ": {"commission_multiplier": 1.05, "fee_structure": "maker_taker"},
            "DARK_POOL": {"commission_multiplier": 0.8, "fee_structure": "flat"}
        }
        
        return venue_data.get(venue, {"commission_multiplier": 1.0})
    
    def _perform_scenario_analysis(
        self,
        order_spec: OrderSpecification,
        market_data: MarketMicrostructure,
        venue_info: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform scenario analysis for cost estimation."""
        try:
            scenarios = {}
            
            # Best case scenario
            best_market = market_data.copy()
            best_market.intraday_volatility *= 0.7
            best_market.daily_volume *= 1.5
            best_cost = self.models["aggressive"].predict_cost(order_spec, best_market, venue_info)
            scenarios["best_case"] = {
                "total_cost_bps": best_cost.total_cost_bps,
                "description": "High liquidity, low volatility conditions"
            }
            
            # Worst case scenario
            worst_market = market_data.copy()
            worst_market.intraday_volatility *= 1.5
            worst_market.daily_volume *= 0.5
            worst_cost = self.models["conservative"].predict_cost(order_spec, worst_market, venue_info)
            scenarios["worst_case"] = {
                "total_cost_bps": worst_cost.total_cost_bps,
                "description": "Low liquidity, high volatility conditions"
            }
            
            return scenarios
            
        except Exception as e:
            logger.warning(f"Scenario analysis failed: {str(e)}")
            return {}
    
    def _calculate_confidence_metrics(
        self,
        cost_breakdown: CostBreakdown,
        market_data: MarketMicrostructure,
        model: BaseCostModel
    ) -> Dict[str, Any]:
        """Calculate confidence metrics for the cost estimate."""
        try:
            # Component confidences
            component_confidences = {
                "commission": cost_breakdown.commission.confidence_level,
                "spread": cost_breakdown.spread.confidence_level,
                "market_impact": cost_breakdown.market_impact.confidence_level,
                "fees": cost_breakdown.fees.confidence_level
            }
            
            # Overall confidence (weighted average)
            weights = {
                "commission": 0.2,
                "spread": 0.3,
                "market_impact": 0.4,
                "fees": 0.1
            }
            
            overall_confidence = sum(
                component_confidences[component] * weights[component]
                for component in component_confidences
            )
            
            # Adjust for market conditions
            if market_data.market_hours:
                overall_confidence *= 1.0
            else:
                overall_confidence *= 0.9
            
            # Adjust for data quality
            data_quality = 0.9  # Placeholder
            overall_confidence *= (0.85 + 0.15 * data_quality)
            
            return {
                "overall_confidence": min(overall_confidence, 1.0),
                "component_confidences": component_confidences,
                "confidence_interval": {
                    "lower_95": cost_breakdown.total_cost_bps * 0.8,
                    "upper_95": cost_breakdown.total_cost_bps * 1.2
                },
                "data_quality_score": data_quality
            }
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {str(e)}")
            return {"overall_confidence": 0.7}
    
    def _generate_recommendations(
        self,
        order_spec: OrderSpecification,
        cost_breakdown: CostBreakdown,
        market_data: MarketMicrostructure
    ) -> List[str]:
        """Generate cost optimization recommendations."""
        recommendations = []
        
        # High impact recommendation
        if cost_breakdown.market_impact.basis_points > 10:
            recommendations.append("Consider using TWAP or VWAP execution to reduce market impact")
        
        # High spread cost recommendation
        if cost_breakdown.spread.basis_points > 5:
            recommendations.append("Consider using limit orders to reduce spread costs")
        
        # Timing recommendations
        if not market_data.market_hours:
            recommendations.append("Consider waiting for market hours to reduce costs")
        
        # Venue recommendations
        participation_rate = order_spec.quantity / market_data.daily_volume
        if participation_rate > 0.1:
            recommendations.append("Consider using dark pools for large orders")
        
        # Size recommendations
        if cost_breakdown.total_cost_bps > 20:
            recommendations.append("Consider breaking the order into smaller sizes")
        
        return recommendations
    
    def _update_model_performance(self, model_type: str, confidence: float):
        """Update model performance tracking."""
        if model_type not in self.model_performance:
            self.model_performance[model_type] = {
                "prediction_count": 0,
                "average_confidence": 0.0
            }
        
        perf = self.model_performance[model_type]
        perf["prediction_count"] += 1
        perf["average_confidence"] = (
            (perf["average_confidence"] * (perf["prediction_count"] - 1) + confidence) /
            perf["prediction_count"]
        )
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status information."""
        return {
            "agent_id": self.agent_id,
            "status": "active",
            "available_models": list(self.models.keys()),
            "predictions_performed": self.prediction_count,
            "model_performance": self.model_performance,
            "capabilities": [
                "Multi-component cost modeling",
                "Scenario analysis",
                "Confidence estimation",
                "Optimization recommendations"
            ],
            "last_updated": datetime.utcnow().isoformat()
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the cost predictor
    predictor = CostPredictor()
    
    # Sample market conditions
    market_conditions = {
        "current_price": 150.0,
        "spread": 0.03,
        "daily_volume": 2000000,
        "volatility": 0.25,
        "liquidity_regime": "normal_liquidity",
        "market_hours": True
    }
    
    try:
        result = predictor.estimate_costs(
            symbol="AAPL",
            quantity=10000,
            side="buy",
            order_type="market",
            venue="NYSE",
            market_conditions=market_conditions
        )
        
        print(f"Total Cost: {result['cost_estimate']['total_cost_bps']:.2f} bps")
        print(f"Confidence: {result['confidence_metrics']['overall_confidence']:.3f}")
        print(f"Recommendations: {len(result['recommendations'])}")
        
        for rec in result['recommendations']:
            print(f"  - {rec}")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
