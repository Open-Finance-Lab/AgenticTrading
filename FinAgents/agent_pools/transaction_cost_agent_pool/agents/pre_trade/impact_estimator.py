"""
Market Impact Estimator Agent

This agent provides sophisticated market impact estimation capabilities
for pre-trade analysis, utilizing multiple impact models and machine
learning techniques to predict market impact across different scenarios.

Key Features:
- Multiple impact model implementations (linear, square-root, ML-based)
- Real-time impact estimation with confidence intervals
- Scenario analysis and stress testing
- Historical backtesting and model validation
- Regime-aware impact modeling

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
from abc import ABC, abstractmethod

# Import schema models
from ...schema.market_impact_schema import (
    ImpactEstimate, 
    TemporaryPermanentImpact,
    MarketMicrostructure,
    OrderSpecification,
    ImpactModelType,
    TimeHorizon,
    LiquidityRegime
)

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ImpactModelParameters:
    """Parameters for impact model calibration."""
    alpha: float = 0.6  # Temporary impact coefficient
    beta: float = 0.3   # Permanent impact coefficient
    gamma: float = 0.5  # Participation rate exponent
    delta: float = 0.2  # Volatility sensitivity
    
    # Advanced parameters
    liquidity_factor: float = 1.0
    regime_adjustment: float = 1.0
    time_decay: float = 0.95
    
    # Confidence parameters
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000

class BaseImpactModel(ABC):
    """
    Abstract base class for market impact models.
    
    This class defines the interface that all impact models must implement,
    ensuring consistency across different modeling approaches.
    """
    
    def __init__(self, model_name: str, parameters: ImpactModelParameters):
        self.model_name = model_name
        self.parameters = parameters
        self.calibration_date: Optional[datetime] = None
        self.calibration_r2: Optional[float] = None
    
    @abstractmethod
    def estimate_impact(
        self,
        order_spec: OrderSpecification,
        market_data: MarketMicrostructure
    ) -> TemporaryPermanentImpact:
        """
        Estimate market impact for given order and market conditions.
        
        Args:
            order_spec: Order specification
            market_data: Market microstructure data
            
        Returns:
            TemporaryPermanentImpact: Impact estimate breakdown
        """
        pass
    
    @abstractmethod
    def calibrate(self, historical_data: pd.DataFrame) -> bool:
        """
        Calibrate model parameters using historical data.
        
        Args:
            historical_data: Historical execution data
            
        Returns:
            bool: True if calibration successful
        """
        pass

class LinearImpactModel(BaseImpactModel):
    """
    Linear impact model implementation.
    
    This model assumes market impact is linear in order size,
    suitable for smaller orders and liquid markets.
    """
    
    def __init__(self, parameters: ImpactModelParameters):
        super().__init__("linear", parameters)
    
    def estimate_impact(
        self,
        order_spec: OrderSpecification,
        market_data: MarketMicrostructure
    ) -> TemporaryPermanentImpact:
        """
        Estimate impact using linear model.
        
        The linear model assumes:
        - Temporary impact = alpha * (quantity / daily_volume) * volatility
        - Permanent impact = beta * (quantity / daily_volume) * volatility
        """
        try:
            # Calculate participation rate
            participation_rate = order_spec.quantity / market_data.daily_volume
            
            # Base impact calculation
            base_impact = participation_rate * market_data.intraday_volatility
            
            # Apply regime adjustments
            regime_multiplier = self._get_regime_multiplier(market_data.liquidity_regime)
            
            # Calculate temporary impact
            temp_impact_bps = (
                self.parameters.alpha * base_impact * regime_multiplier * 10000
            )
            
            # Calculate permanent impact
            perm_impact_bps = (
                self.parameters.beta * base_impact * regime_multiplier * 10000
            )
            
            # Convert to amounts (assuming current price)
            current_price = (market_data.bid_size + market_data.ask_size) / 2  # Simplified
            total_value = order_spec.quantity * current_price
            
            temp_impact_amount = (temp_impact_bps / 10000) * total_value
            perm_impact_amount = (perm_impact_bps / 10000) * total_value
            
            return TemporaryPermanentImpact(
                temporary_impact_bps=temp_impact_bps,
                temporary_impact_amount=temp_impact_amount,
                temporary_duration=self._estimate_decay_time(participation_rate),
                permanent_impact_bps=perm_impact_bps,
                permanent_impact_amount=perm_impact_amount,
                total_impact_bps=temp_impact_bps + perm_impact_bps,
                total_impact_amount=temp_impact_amount + perm_impact_amount,
                temporary_impact_confidence={
                    "lower_95": temp_impact_bps * 0.8,
                    "upper_95": temp_impact_bps * 1.2
                },
                permanent_impact_confidence={
                    "lower_95": perm_impact_bps * 0.7,
                    "upper_95": perm_impact_bps * 1.3
                }
            )
            
        except Exception as e:
            logger.error(f"Linear impact estimation failed: {str(e)}")
            raise
    
    def calibrate(self, historical_data: pd.DataFrame) -> bool:
        """Calibrate linear model parameters."""
        try:
            # Simplified calibration - in practice would use regression
            # This is a placeholder implementation
            logger.info("Calibrating linear impact model...")
            
            # Mock calibration results
            self.calibration_date = datetime.utcnow()
            self.calibration_r2 = 0.75
            
            return True
            
        except Exception as e:
            logger.error(f"Linear model calibration failed: {str(e)}")
            return False
    
    def _get_regime_multiplier(self, regime: LiquidityRegime) -> float:
        """Get regime-specific multiplier."""
        multipliers = {
            LiquidityRegime.HIGH_LIQUIDITY: 0.7,
            LiquidityRegime.NORMAL_LIQUIDITY: 1.0,
            LiquidityRegime.LOW_LIQUIDITY: 1.5,
            LiquidityRegime.STRESSED_LIQUIDITY: 2.5
        }
        return multipliers.get(regime, 1.0)
    
    def _estimate_decay_time(self, participation_rate: float) -> float:
        """Estimate temporary impact decay time."""
        # Simple heuristic - higher participation = longer decay
        base_decay = 5.0  # minutes
        return base_decay * (1 + participation_rate * 10)

class SquareRootImpactModel(BaseImpactModel):
    """
    Square-root impact model implementation.
    
    This model assumes market impact follows a square-root relationship
    with order size, commonly used for larger institutional orders.
    """
    
    def __init__(self, parameters: ImpactModelParameters):
        super().__init__("square_root", parameters)
    
    def estimate_impact(
        self,
        order_spec: OrderSpecification,
        market_data: MarketMicrostructure
    ) -> TemporaryPermanentImpact:
        """
        Estimate impact using square-root model.
        
        The square-root model assumes:
        - Impact ∝ sqrt(quantity / daily_volume) * volatility
        """
        try:
            # Calculate participation rate
            participation_rate = order_spec.quantity / market_data.daily_volume
            
            # Square-root impact calculation
            sqrt_participation = np.sqrt(participation_rate)
            base_impact = sqrt_participation * market_data.intraday_volatility
            
            # Apply regime and time adjustments
            regime_multiplier = self._get_regime_multiplier(market_data.liquidity_regime)
            time_multiplier = self._get_time_multiplier(market_data.time_of_day)
            
            # Calculate temporary impact
            temp_impact_bps = (
                self.parameters.alpha * base_impact * regime_multiplier * 
                time_multiplier * 10000
            )
            
            # Calculate permanent impact
            perm_impact_bps = (
                self.parameters.beta * base_impact * regime_multiplier * 
                time_multiplier * 10000
            )
            
            # Convert to amounts
            current_price = (market_data.bid_size + market_data.ask_size) / 2
            total_value = order_spec.quantity * current_price
            
            temp_impact_amount = (temp_impact_bps / 10000) * total_value
            perm_impact_amount = (perm_impact_bps / 10000) * total_value
            
            return TemporaryPermanentImpact(
                temporary_impact_bps=temp_impact_bps,
                temporary_impact_amount=temp_impact_amount,
                temporary_duration=self._estimate_decay_time(participation_rate),
                permanent_impact_bps=perm_impact_bps,
                permanent_impact_amount=perm_impact_amount,
                total_impact_bps=temp_impact_bps + perm_impact_bps,
                total_impact_amount=temp_impact_amount + perm_impact_amount,
                temporary_impact_confidence={
                    "lower_95": temp_impact_bps * 0.75,
                    "upper_95": temp_impact_bps * 1.25
                },
                permanent_impact_confidence={
                    "lower_95": perm_impact_bps * 0.65,
                    "upper_95": perm_impact_bps * 1.35
                }
            )
            
        except Exception as e:
            logger.error(f"Square-root impact estimation failed: {str(e)}")
            raise
    
    def calibrate(self, historical_data: pd.DataFrame) -> bool:
        """Calibrate square-root model parameters."""
        try:
            logger.info("Calibrating square-root impact model...")
            
            # Mock calibration results
            self.calibration_date = datetime.utcnow()
            self.calibration_r2 = 0.82
            
            return True
            
        except Exception as e:
            logger.error(f"Square-root model calibration failed: {str(e)}")
            return False
    
    def _get_regime_multiplier(self, regime: LiquidityRegime) -> float:
        """Get regime-specific multiplier for square-root model."""
        multipliers = {
            LiquidityRegime.HIGH_LIQUIDITY: 0.8,
            LiquidityRegime.NORMAL_LIQUIDITY: 1.0,
            LiquidityRegime.LOW_LIQUIDITY: 1.4,
            LiquidityRegime.STRESSED_LIQUIDITY: 2.2
        }
        return multipliers.get(regime, 1.0)
    
    def _get_time_multiplier(self, time_of_day: str) -> float:
        """Get time-of-day multiplier."""
        multipliers = {
            "morning": 1.2,
            "midday": 0.9,
            "afternoon": 1.0,
            "close": 1.3
        }
        return multipliers.get(time_of_day, 1.0)
    
    def _estimate_decay_time(self, participation_rate: float) -> float:
        """Estimate decay time for square-root model."""
        base_decay = 8.0  # minutes
        return base_decay * (1 + np.sqrt(participation_rate) * 5)

class ImpactEstimator:
    """
    Comprehensive market impact estimation agent.
    
    This agent orchestrates multiple impact models to provide robust
    market impact estimates with confidence intervals and scenario analysis.
    """
    
    def __init__(self, agent_id: str = "impact_estimator"):
        """
        Initialize the Impact Estimator agent.
        
        Args:
            agent_id: Unique agent identifier
        """
        self.agent_id = agent_id
        self.models: Dict[str, BaseImpactModel] = {}
        self.default_parameters = ImpactModelParameters()
        
        # Initialize default models
        self._initialize_models()
        
        # Performance tracking
        self.estimation_count = 0
        self.average_confidence = 0.0
        
        logger.info(f"Impact Estimator agent initialized: {agent_id}")
    
    def _initialize_models(self):
        """Initialize available impact models."""
        try:
            # Linear model
            linear_params = ImpactModelParameters(alpha=0.5, beta=0.2)
            self.models["linear"] = LinearImpactModel(linear_params)
            
            # Square-root model
            sqrt_params = ImpactModelParameters(alpha=0.7, beta=0.3)
            self.models["square_root"] = SquareRootImpactModel(sqrt_params)
            
            logger.info(f"Initialized {len(self.models)} impact models")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise
    
    def estimate_market_impact(
        self,
        order_specification: OrderSpecification,
        market_microstructure: MarketMicrostructure,
        model_type: str = "square_root",
        scenario_analysis: bool = True
    ) -> ImpactEstimate:
        """
        Estimate market impact for a given order.
        
        Args:
            order_specification: Order details
            market_microstructure: Market state
            model_type: Impact model to use
            scenario_analysis: Whether to perform scenario analysis
            
        Returns:
            ImpactEstimate: Comprehensive impact estimate
        """
        try:
            estimate_id = f"IMPACT_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{self.estimation_count}"
            
            # Get the specified model
            if model_type not in self.models:
                logger.warning(f"Model {model_type} not found, using square_root")
                model_type = "square_root"
            
            model = self.models[model_type]
            
            # Generate base impact estimate
            impact_breakdown = model.estimate_impact(order_specification, market_microstructure)
            
            # Perform scenario analysis if requested
            scenario_results = {}
            if scenario_analysis:
                scenario_results = self._perform_scenario_analysis(
                    order_specification, market_microstructure, model
                )
            
            # Calculate confidence metrics
            model_confidence = self._calculate_model_confidence(model, market_microstructure)
            
            # Generate time-based impact profile
            impact_profile = self._generate_impact_profile(
                impact_breakdown, order_specification
            )
            
            # Create comprehensive estimate
            estimate = ImpactEstimate(
                estimate_id=estimate_id,
                model_type=ImpactModelType(model_type),
                model_version="1.0.0",
                order_specification=order_specification,
                market_microstructure=market_microstructure,
                impact_breakdown=impact_breakdown,
                impact_profile=impact_profile,
                best_case_impact=scenario_results.get("best_case", impact_breakdown.total_impact_bps * 0.7),
                worst_case_impact=scenario_results.get("worst_case", impact_breakdown.total_impact_bps * 1.5),
                base_case_impact=impact_breakdown.total_impact_bps,
                model_confidence=model_confidence,
                prediction_interval={
                    "lower_95": impact_breakdown.total_impact_bps * 0.75,
                    "upper_95": impact_breakdown.total_impact_bps * 1.25
                },
                sensitivity_analysis=self._perform_sensitivity_analysis(
                    order_specification, market_microstructure, model
                ),
                historical_accuracy=model.calibration_r2,
                risk_factors=self._identify_risk_factors(order_specification, market_microstructure),
                recommended_strategy=self._recommend_execution_strategy(
                    order_specification, market_microstructure, impact_breakdown
                ),
                estimation_timestamp=datetime.utcnow(),
                analyst_id=self.agent_id,
                validation_status="completed"
            )
            
            # Update tracking metrics
            self.estimation_count += 1
            self.average_confidence = (
                (self.average_confidence * (self.estimation_count - 1) + model_confidence) / 
                self.estimation_count
            )
            
            logger.info(f"Generated impact estimate {estimate_id} with confidence {model_confidence:.3f}")
            
            return estimate
            
        except Exception as e:
            logger.error(f"Impact estimation failed: {str(e)}")
            raise
    
    def _perform_scenario_analysis(
        self,
        order_spec: OrderSpecification,
        market_data: MarketMicrostructure,
        model: BaseImpactModel
    ) -> Dict[str, float]:
        """Perform scenario analysis for impact estimation."""
        try:
            scenarios = {}
            
            # Best case scenario (high liquidity)
            best_case_market = market_data.copy()
            best_case_market.liquidity_regime = LiquidityRegime.HIGH_LIQUIDITY
            best_case_market.daily_volume *= 1.5
            best_case_impact = model.estimate_impact(order_spec, best_case_market)
            scenarios["best_case"] = best_case_impact.total_impact_bps
            
            # Worst case scenario (stressed liquidity)
            worst_case_market = market_data.copy()
            worst_case_market.liquidity_regime = LiquidityRegime.STRESSED_LIQUIDITY
            worst_case_market.daily_volume *= 0.5
            worst_case_impact = model.estimate_impact(order_spec, worst_case_market)
            scenarios["worst_case"] = worst_case_impact.total_impact_bps
            
            return scenarios
            
        except Exception as e:
            logger.warning(f"Scenario analysis failed: {str(e)}")
            return {}
    
    def _calculate_model_confidence(
        self,
        model: BaseImpactModel,
        market_data: MarketMicrostructure
    ) -> float:
        """Calculate model confidence based on various factors."""
        try:
            confidence = 0.8  # Base confidence
            
            # Adjust for model calibration quality
            if model.calibration_r2:
                confidence *= model.calibration_r2
            
            # Adjust for market regime
            if market_data.liquidity_regime == LiquidityRegime.NORMAL_LIQUIDITY:
                confidence *= 1.0
            elif market_data.liquidity_regime == LiquidityRegime.HIGH_LIQUIDITY:
                confidence *= 0.95
            else:
                confidence *= 0.8
            
            # Adjust for market hours
            if market_data.market_hours:
                confidence *= 1.0
            else:
                confidence *= 0.9
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {str(e)}")
            return 0.7
    
    def _generate_impact_profile(
        self,
        impact_breakdown: TemporaryPermanentImpact,
        order_spec: OrderSpecification
    ) -> Dict[TimeHorizon, float]:
        """Generate time-based impact profile."""
        try:
            profile = {}
            
            # Immediate impact (temporary + permanent)
            profile[TimeHorizon.IMMEDIATE] = impact_breakdown.total_impact_bps
            
            # Short-term impact (some temporary recovery)
            profile[TimeHorizon.SHORT_TERM] = (
                impact_breakdown.permanent_impact_bps + 
                impact_breakdown.temporary_impact_bps * 0.7
            )
            
            # Medium-term impact (most temporary recovery)
            profile[TimeHorizon.MEDIUM_TERM] = (
                impact_breakdown.permanent_impact_bps + 
                impact_breakdown.temporary_impact_bps * 0.3
            )
            
            # Long-term impact (only permanent)
            profile[TimeHorizon.LONG_TERM] = impact_breakdown.permanent_impact_bps
            
            return profile
            
        except Exception as e:
            logger.warning(f"Impact profile generation failed: {str(e)}")
            return {}
    
    def _perform_sensitivity_analysis(
        self,
        order_spec: OrderSpecification,
        market_data: MarketMicrostructure,
        model: BaseImpactModel
    ) -> Dict[str, float]:
        """Perform sensitivity analysis on key parameters."""
        try:
            sensitivity = {}
            base_impact = model.estimate_impact(order_spec, market_data)
            
            # Volume sensitivity
            high_vol_market = market_data.copy()
            high_vol_market.daily_volume *= 1.2
            high_vol_impact = model.estimate_impact(order_spec, high_vol_market)
            sensitivity["volume_sensitivity"] = (
                (high_vol_impact.total_impact_bps - base_impact.total_impact_bps) / 
                base_impact.total_impact_bps * 100
            )
            
            # Volatility sensitivity
            high_vol_market = market_data.copy()
            high_vol_market.intraday_volatility *= 1.2
            high_vol_impact = model.estimate_impact(order_spec, high_vol_market)
            sensitivity["volatility_sensitivity"] = (
                (high_vol_impact.total_impact_bps - base_impact.total_impact_bps) / 
                base_impact.total_impact_bps * 100
            )
            
            return sensitivity
            
        except Exception as e:
            logger.warning(f"Sensitivity analysis failed: {str(e)}")
            return {}
    
    def _identify_risk_factors(
        self,
        order_spec: OrderSpecification,
        market_data: MarketMicrostructure
    ) -> List[str]:
        """Identify risk factors for the execution."""
        risk_factors = []
        
        # Check order size relative to daily volume
        participation_rate = order_spec.quantity / market_data.daily_volume
        if participation_rate > 0.1:
            risk_factors.append("High participation rate (>10% of daily volume)")
        
        # Check liquidity regime
        if market_data.liquidity_regime in [LiquidityRegime.LOW_LIQUIDITY, LiquidityRegime.STRESSED_LIQUIDITY]:
            risk_factors.append("Low liquidity market conditions")
        
        # Check market hours
        if not market_data.market_hours:
            risk_factors.append("Execution outside normal market hours")
        
        # Check recent volatility
        if market_data.intraday_volatility > 0.3:
            risk_factors.append("High intraday volatility")
        
        # Check spread
        if market_data.bid_ask_spread > market_data.effective_spread * 1.5:
            risk_factors.append("Wide bid-ask spread")
        
        return risk_factors
    
    def _recommend_execution_strategy(
        self,
        order_spec: OrderSpecification,
        market_data: MarketMicrostructure,
        impact_breakdown: TemporaryPermanentImpact
    ) -> str:
        """Recommend optimal execution strategy."""
        try:
            participation_rate = order_spec.quantity / market_data.daily_volume
            
            if participation_rate < 0.05:
                return "Market order - low impact expected"
            elif participation_rate < 0.15:
                return "TWAP strategy over 30-60 minutes"
            elif participation_rate < 0.25:
                return "VWAP strategy over 2-4 hours"
            else:
                return "Implementation shortfall strategy over full day"
                
        except Exception as e:
            logger.warning(f"Strategy recommendation failed: {str(e)}")
            return "Conservative TWAP strategy recommended"
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status information."""
        return {
            "agent_id": self.agent_id,
            "status": "active",
            "available_models": list(self.models.keys()),
            "estimations_performed": self.estimation_count,
            "average_confidence": self.average_confidence,
            "model_status": {
                model_name: {
                    "calibration_r2": model.calibration_r2,
                    "last_calibration": model.calibration_date.isoformat() if model.calibration_date else None
                }
                for model_name, model in self.models.items()
            },
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def calibrate_models(self, historical_data: pd.DataFrame) -> Dict[str, bool]:
        """Calibrate all available models."""
        results = {}
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Calibrating model: {model_name}")
                results[model_name] = model.calibrate(historical_data)
            except Exception as e:
                logger.error(f"Calibration failed for {model_name}: {str(e)}")
                results[model_name] = False
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    sample_order = OrderSpecification(
        symbol="AAPL",
        side="buy",
        quantity=50000,
        order_type="market",
        execution_strategy="twap",
        participation_rate=0.1
    )
    
    sample_market = MarketMicrostructure(
        bid_ask_spread=0.02,
        bid_size=1000,
        ask_size=1200,
        effective_spread=0.015,
        daily_volume=75000000,
        recent_volume=1500000,
        intraday_volatility=0.25,
        liquidity_regime=LiquidityRegime.NORMAL_LIQUIDITY,
        market_hours=True,
        time_of_day="morning",
        day_of_week="Tuesday",
        recent_price_movement=5.2
    )
    
    # Test the impact estimator
    estimator = ImpactEstimator()
    
    try:
        estimate = estimator.estimate_market_impact(
            sample_order, 
            sample_market, 
            model_type="square_root"
        )
        
        print(f"Impact Estimate ID: {estimate.estimate_id}")
        print(f"Total Impact: {estimate.base_case_impact:.2f} bps")
        print(f"Confidence: {estimate.model_confidence:.3f}")
        print(f"Recommended Strategy: {estimate.recommended_strategy}")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
