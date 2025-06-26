"""
Market Impact Schema - Market Impact Analysis Models

This module defines data structures for market impact estimation and analysis,
including temporary/permanent impact decomposition and microstructure modeling.

Key Models:
- ImpactEstimate: Market impact estimation results
- TemporaryPermanentImpact: Impact decomposition
- MarketMicrostructure: Market microstructure data
- OrderSpecification: Order specification for impact analysis

Author: FinAgent Development Team
License: OpenMDW
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from decimal import Decimal
from enum import Enum

class ImpactModelType(str, Enum):
    """Market impact model types."""
    LINEAR = "linear"
    SQUARE_ROOT = "square_root"
    LOGARITHMIC = "logarithmic"
    POWER_LAW = "power_law"
    MACHINE_LEARNING = "machine_learning"

class TimeHorizon(str, Enum):
    """Time horizons for impact analysis."""
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"    # Minutes to hours
    MEDIUM_TERM = "medium_term"  # Hours to days
    LONG_TERM = "long_term"      # Days to weeks

class LiquidityRegime(str, Enum):
    """Market liquidity regime classification."""
    HIGH_LIQUIDITY = "high_liquidity"
    NORMAL_LIQUIDITY = "normal_liquidity"
    LOW_LIQUIDITY = "low_liquidity"
    STRESSED_LIQUIDITY = "stressed_liquidity"

class OrderSpecification(BaseModel):
    """
    Detailed order specification for market impact analysis.
    
    This model captures all relevant order characteristics that influence
    market impact estimation and analysis.
    """
    # Basic order details
    symbol: str = Field(..., description="Trading symbol")
    side: str = Field(..., description="Order side (buy/sell)")
    quantity: float = Field(..., description="Order quantity")
    order_type: str = Field(..., description="Order type")
    
    # Execution parameters
    execution_strategy: str = Field(..., description="Execution strategy")
    participation_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Market participation rate")
    execution_duration: Optional[float] = Field(None, description="Planned execution duration (minutes)")
    
    # Timing parameters
    start_time: Optional[datetime] = Field(None, description="Execution start time")
    end_time: Optional[datetime] = Field(None, description="Execution end time")
    urgency_level: str = Field("normal", description="Execution urgency level")
    
    # Price parameters
    limit_price: Optional[float] = Field(None, description="Limit price if applicable")
    arrival_price: Optional[float] = Field(None, description="Arrival price reference")
    
    # Additional constraints
    minimum_fill_size: Optional[float] = Field(None, description="Minimum fill size")
    maximum_fill_size: Optional[float] = Field(None, description="Maximum fill size")
    venue_constraints: List[str] = Field(default_factory=list, description="Venue constraints")
    
    @validator('quantity')
    def validate_quantity(cls, v):
        if v <= 0:
            raise ValueError('Order quantity must be positive')
        return v

class MarketMicrostructure(BaseModel):
    """
    Market microstructure data for impact analysis.
    
    Captures the current state of market microstructure that affects
    the expected market impact of order execution.
    """
    # Liquidity metrics
    bid_ask_spread: float = Field(..., description="Current bid-ask spread")
    bid_size: float = Field(..., description="Bid size at best level")
    ask_size: float = Field(..., description="Ask size at best level")
    
    # Order book depth
    order_book_depth: Dict[str, List[Dict[str, float]]] = Field(
        default_factory=dict, 
        description="Order book depth (bids/asks)"
    )
    effective_spread: float = Field(..., description="Effective spread")
    
    # Volume and volatility
    daily_volume: float = Field(..., description="Average daily volume")
    recent_volume: float = Field(..., description="Recent trading volume")
    intraday_volatility: float = Field(..., description="Intraday volatility")
    
    # Market regime
    liquidity_regime: LiquidityRegime = Field(..., description="Current liquidity regime")
    market_hours: bool = Field(..., description="Whether market is in normal hours")
    
    # Timing factors
    time_of_day: str = Field(..., description="Time of day classification")
    day_of_week: str = Field(..., description="Day of week")
    is_expiry_day: bool = Field(False, description="Whether it's an expiry day")
    
    # Recent activity
    recent_price_movement: float = Field(..., description="Recent price movement (bps)")
    recent_volume_spike: bool = Field(False, description="Whether recent volume spike occurred")
    
    # Measurement timestamp
    measurement_time: datetime = Field(default_factory=datetime.utcnow, description="Measurement timestamp")

class TemporaryPermanentImpact(BaseModel):
    """
    Decomposition of market impact into temporary and permanent components.
    
    This model provides detailed breakdown of market impact into its
    temporary and permanent components, essential for cost optimization.
    """
    # Temporary impact (recovers over time)
    temporary_impact_bps: float = Field(..., description="Temporary impact in basis points")
    temporary_impact_amount: Decimal = Field(..., description="Temporary impact amount")
    temporary_duration: float = Field(..., description="Expected temporary impact duration (minutes)")
    
    # Permanent impact (price discovery)
    permanent_impact_bps: float = Field(..., description="Permanent impact in basis points")
    permanent_impact_amount: Decimal = Field(..., description="Permanent impact amount")
    
    # Total impact
    total_impact_bps: float = Field(..., description="Total impact in basis points")
    total_impact_amount: Decimal = Field(..., description="Total impact amount")
    
    # Confidence intervals
    temporary_impact_confidence: Dict[str, float] = Field(
        default_factory=dict, 
        description="Confidence intervals for temporary impact"
    )
    permanent_impact_confidence: Dict[str, float] = Field(
        default_factory=dict,
        description="Confidence intervals for permanent impact"
    )
    
    # Model attribution
    model_contribution: Dict[str, float] = Field(
        default_factory=dict,
        description="Contribution of different model factors"
    )
    
    @validator('temporary_impact_bps', 'permanent_impact_bps')
    def validate_non_negative_impact(cls, v):
        if v < 0:
            raise ValueError('Impact values must be non-negative')
        return v

class ImpactEstimate(BaseModel):
    """
    Comprehensive market impact estimation results.
    
    This model encapsulates the complete market impact analysis,
    including estimates, confidence intervals, and model diagnostics.
    """
    # Estimation metadata
    estimate_id: str = Field(..., description="Unique estimate identifier")
    model_type: ImpactModelType = Field(..., description="Impact model type used")
    model_version: str = Field(..., description="Model version")
    
    # Order and market context
    order_specification: OrderSpecification = Field(..., description="Order specification")
    market_microstructure: MarketMicrostructure = Field(..., description="Market microstructure")
    
    # Impact decomposition
    impact_breakdown: TemporaryPermanentImpact = Field(..., description="Impact breakdown")
    
    # Time-based impact profile
    impact_profile: Dict[TimeHorizon, float] = Field(
        default_factory=dict,
        description="Impact estimates by time horizon"
    )
    
    # Scenario analysis
    best_case_impact: float = Field(..., description="Best case impact estimate (bps)")
    worst_case_impact: float = Field(..., description="Worst case impact estimate (bps)")
    base_case_impact: float = Field(..., description="Base case impact estimate (bps)")
    
    # Model diagnostics
    model_confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")
    prediction_interval: Dict[str, float] = Field(
        default_factory=dict,
        description="Prediction interval bounds"
    )
    
    # Sensitivity analysis
    sensitivity_analysis: Dict[str, float] = Field(
        default_factory=dict,
        description="Sensitivity to key parameters"
    )
    
    # Historical comparison
    historical_accuracy: Optional[float] = Field(None, description="Historical model accuracy")
    similar_trades_impact: Optional[List[float]] = Field(None, description="Impact of similar historical trades")
    
    # Risk factors
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors")
    risk_adjustments: Dict[str, float] = Field(
        default_factory=dict,
        description="Risk-based adjustments"
    )
    
    # Execution recommendations
    recommended_strategy: Optional[str] = Field(None, description="Recommended execution strategy")
    alternative_strategies: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Alternative execution strategies"
    )
    
    # Timing and metadata
    estimation_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Estimation timestamp")
    analyst_id: Optional[str] = Field(None, description="Analyst identifier")
    
    # Validation
    validation_status: str = Field("pending", description="Validation status")
    validation_notes: Optional[str] = Field(None, description="Validation notes")

    class Config:
        """Pydantic model configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v)
        }
        
        schema_extra = {
            "example": {
                "estimate_id": "IMPACT_20241225_001",
                "model_type": "square_root",
                "model_version": "2.1.0",
                "order_specification": {
                    "symbol": "AAPL",
                    "side": "buy",
                    "quantity": 50000,
                    "order_type": "market",
                    "execution_strategy": "twap",
                    "participation_rate": 0.1
                },
                "market_microstructure": {
                    "bid_ask_spread": 0.02,
                    "bid_size": 1000,
                    "ask_size": 1200,
                    "effective_spread": 0.015,
                    "daily_volume": 75000000,
                    "recent_volume": 1500000,
                    "intraday_volatility": 0.25,
                    "liquidity_regime": "normal_liquidity",
                    "market_hours": True,
                    "time_of_day": "morning",
                    "day_of_week": "Tuesday",
                    "recent_price_movement": 5.2
                },
                "impact_breakdown": {
                    "temporary_impact_bps": 12.5,
                    "temporary_impact_amount": 185.75,
                    "temporary_duration": 15.0,
                    "permanent_impact_bps": 8.2,
                    "permanent_impact_amount": 121.90,
                    "total_impact_bps": 20.7,
                    "total_impact_amount": 307.65
                },
                "best_case_impact": 15.2,
                "worst_case_impact": 28.1,
                "base_case_impact": 20.7,
                "model_confidence": 0.87
            }
        }
