"""
Transaction Cost Models - Core Data Structures

This module defines the fundamental data models for transaction cost analysis,
including cost breakdowns, market impact modeling, and performance metrics.

Key Models:
- TransactionCost: Comprehensive transaction cost representation
- CostBreakdown: Detailed cost component analysis
- MarketImpactModel: Market impact estimation models
- ExecutionMetrics: Execution quality measurements
- PerformanceBenchmark: Performance comparison standards

Author: FinAgent Development Team
License: OpenMDW
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from decimal import Decimal
from enum import Enum

class CurrencyCode(str, Enum):
    """Standard currency codes for international markets."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CHF = "CHF"
    CAD = "CAD"
    AUD = "AUD"

class AssetClass(str, Enum):
    """Supported asset classes for transaction cost analysis."""
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    DERIVATIVES = "derivatives"
    FOREX = "forex"
    COMMODITY = "commodity"
    CRYPTOCURRENCY = "cryptocurrency"

class OrderSide(str, Enum):
    """Order side specification."""
    BUY = "buy"
    SELL = "sell"

class OrderType(str, Enum):
    """Supported order types for cost analysis."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"

class VenueType(str, Enum):
    """Market venue types."""
    EXCHANGE = "exchange"
    DARK_POOL = "dark_pool"
    ECN = "ecn"
    MARKET_MAKER = "market_maker"
    CROSSING_NETWORK = "crossing_network"

class CostComponent(BaseModel):
    """
    Individual cost component within a transaction cost breakdown.
    
    Attributes:
        component_type (str): Type of cost component (commission, spread, impact, etc.)
        amount (Decimal): Cost amount in base currency
        currency (CurrencyCode): Currency denomination
        basis_points (float): Cost expressed in basis points
        description (str): Human-readable description
        calculation_method (str): Method used to calculate this component
        confidence_level (float): Confidence in the calculation (0.0-1.0)
    """
    component_type: str = Field(..., description="Cost component type identifier")
    amount: Decimal = Field(..., description="Cost amount in base currency")
    currency: CurrencyCode = Field(..., description="Currency denomination")
    basis_points: float = Field(..., description="Cost in basis points")
    description: str = Field("", description="Component description")
    calculation_method: str = Field("", description="Calculation methodology")
    confidence_level: float = Field(1.0, ge=0.0, le=1.0, description="Calculation confidence")

class CostBreakdown(BaseModel):
    """
    Comprehensive breakdown of transaction costs into individual components.
    
    This model provides detailed decomposition of all cost elements
    associated with a trade execution, enabling precise cost attribution
    and optimization opportunities identification.
    """
    total_cost: Decimal = Field(..., description="Total transaction cost")
    total_cost_bps: float = Field(..., description="Total cost in basis points")
    currency: CurrencyCode = Field(..., description="Base currency")
    
    # Core cost components
    commission: CostComponent = Field(..., description="Brokerage commission")
    spread: CostComponent = Field(..., description="Bid-ask spread cost")
    market_impact: CostComponent = Field(..., description="Market impact cost")
    
    # Optional components
    taxes: Optional[CostComponent] = Field(None, description="Transaction taxes")
    fees: Optional[CostComponent] = Field(None, description="Exchange/venue fees")
    borrowing_cost: Optional[CostComponent] = Field(None, description="Securities borrowing cost")
    opportunity_cost: Optional[CostComponent] = Field(None, description="Timing opportunity cost")
    
    # Additional components for flexible cost modeling
    other_components: List[CostComponent] = Field(default_factory=list, description="Additional cost components")
    
    calculation_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Cost calculation time")
    model_version: str = Field("1.0.0", description="Cost model version")

class MarketImpactParameters(BaseModel):
    """
    Parameters for market impact modeling and estimation.
    
    These parameters control the behavior of market impact models,
    including both temporary and permanent impact components.
    """
    temporary_impact_rate: float = Field(..., description="Temporary impact rate parameter")
    permanent_impact_rate: float = Field(..., description="Permanent impact rate parameter")
    participation_rate: float = Field(..., description="Order participation rate")
    volatility: float = Field(..., description="Asset volatility")
    daily_volume: float = Field(..., description="Average daily volume")
    
    # Model-specific parameters
    model_parameters: Dict[str, Any] = Field(default_factory=dict, description="Model-specific parameters")

class MarketImpactModel(BaseModel):
    """
    Market impact model specification and estimation results.
    
    This model encapsulates both the configuration of market impact
    models and their estimation outputs for transaction cost analysis.
    """
    model_name: str = Field(..., description="Impact model identifier")
    model_type: str = Field(..., description="Model type (linear, square-root, etc.)")
    parameters: MarketImpactParameters = Field(..., description="Model parameters")
    
    # Impact estimates
    temporary_impact: float = Field(..., description="Temporary impact estimate (bps)")
    permanent_impact: float = Field(..., description="Permanent impact estimate (bps)")
    total_impact: float = Field(..., description="Total impact estimate (bps)")
    
    # Confidence and validation metrics
    confidence_interval: Dict[str, float] = Field(default_factory=dict, description="Confidence intervals")
    model_accuracy: Optional[float] = Field(None, description="Historical model accuracy")
    last_calibration: Optional[datetime] = Field(None, description="Last model calibration date")

class ExecutionMetrics(BaseModel):
    """
    Comprehensive execution quality metrics for post-trade analysis.
    
    This model captures detailed metrics used to evaluate the quality
    of trade execution against various benchmarks and standards.
    """
    # Primary execution metrics
    implementation_shortfall: float = Field(..., description="Implementation shortfall (bps)")
    arrival_price_deviation: float = Field(..., description="Deviation from arrival price (bps)")
    volume_weighted_price_deviation: float = Field(..., description="VWAP deviation (bps)")
    time_weighted_price_deviation: float = Field(..., description="TWAP deviation (bps)")
    
    # Execution timing metrics
    execution_duration: float = Field(..., description="Total execution time (minutes)")
    fill_rate: float = Field(..., description="Order fill rate (0.0-1.0)")
    average_fill_size: float = Field(..., description="Average fill size")
    number_of_fills: int = Field(..., description="Total number of fills")
    
    # Market impact measurements
    measured_impact: float = Field(..., description="Measured market impact (bps)")
    impact_decay_rate: Optional[float] = Field(None, description="Impact decay rate")
    
    # Risk-adjusted metrics
    sharpe_ratio: Optional[float] = Field(None, description="Execution Sharpe ratio")
    information_ratio: Optional[float] = Field(None, description="Information ratio vs benchmark")
    
    # Venue and routing metrics
    venue_distribution: Dict[str, float] = Field(default_factory=dict, description="Volume by venue")
    routing_efficiency: Optional[float] = Field(None, description="Routing efficiency score")

class PerformanceBenchmark(BaseModel):
    """
    Performance benchmark specification for execution quality comparison.
    
    Defines the benchmarks against which execution performance
    is measured and evaluated.
    """
    benchmark_name: str = Field(..., description="Benchmark identifier")
    benchmark_type: str = Field(..., description="Benchmark type (TWAP, VWAP, etc.)")
    
    # Benchmark values
    benchmark_price: float = Field(..., description="Benchmark price")
    benchmark_cost: float = Field(..., description="Benchmark cost (bps)")
    
    # Calculation parameters
    calculation_period: str = Field(..., description="Benchmark calculation period")
    market_participation_rate: Optional[float] = Field(None, description="Market participation rate")
    
    # Performance comparison
    actual_vs_benchmark: float = Field(..., description="Actual performance vs benchmark (bps)")
    percentile_rank: Optional[float] = Field(None, description="Percentile rank vs historical")
    
    benchmark_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Benchmark calculation time")

class CostAttribute(BaseModel):
    """
    Cost attribution analysis for identifying cost sources and optimization opportunities.
    
    This model provides detailed attribution of transaction costs to various
    factors, enabling targeted optimization strategies.
    """
    attribution_factor: str = Field(..., description="Cost attribution factor")
    attributed_cost: float = Field(..., description="Cost attributed to this factor (bps)")
    attribution_percentage: float = Field(..., description="Percentage of total cost")
    
    # Factor-specific details
    factor_description: str = Field("", description="Factor description")
    optimization_potential: Optional[float] = Field(None, description="Potential cost reduction (bps)")
    optimization_strategy: Optional[str] = Field(None, description="Suggested optimization approach")
    
    # Supporting data
    supporting_metrics: Dict[str, Any] = Field(default_factory=dict, description="Supporting metrics")

class TransactionCost(BaseModel):
    """
    Comprehensive transaction cost model encompassing all aspects of trade execution costs.
    
    This is the primary model for representing complete transaction cost information,
    including predictions, measurements, and analysis results.
    """
    # Trade identification
    trade_id: str = Field(..., description="Unique trade identifier")
    symbol: str = Field(..., description="Trading symbol")
    asset_class: AssetClass = Field(..., description="Asset class")
    
    # Order specification
    order_side: OrderSide = Field(..., description="Order side (buy/sell)")
    order_type: OrderType = Field(..., description="Order type")
    quantity: float = Field(..., description="Order quantity")
    currency: CurrencyCode = Field(..., description="Base currency")
    
    # Venue information
    venue: Optional[str] = Field(None, description="Execution venue")
    venue_type: Optional[VenueType] = Field(None, description="Venue type")
    
    # Cost analysis
    cost_breakdown: CostBreakdown = Field(..., description="Detailed cost breakdown")
    market_impact_model: Optional[MarketImpactModel] = Field(None, description="Market impact analysis")
    execution_metrics: Optional[ExecutionMetrics] = Field(None, description="Execution quality metrics")
    
    # Performance benchmarking
    benchmarks: List[PerformanceBenchmark] = Field(default_factory=list, description="Performance benchmarks")
    cost_attribution: List[CostAttribute] = Field(default_factory=list, description="Cost attribution analysis")
    
    # Metadata
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")
    analyst_id: Optional[str] = Field(None, description="Analyst identifier")
    analysis_version: str = Field("1.0.0", description="Analysis version")
    
    # Validation and quality assurance
    data_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Data quality score")
    validation_status: str = Field("pending", description="Validation status")
    validation_notes: Optional[str] = Field(None, description="Validation notes")

    @validator('quantity')
    def validate_quantity(cls, v):
        if v <= 0:
            raise ValueError('Quantity must be positive')
        return v

    @validator('cost_breakdown')
    def validate_cost_consistency(cls, v, values):
        """Ensure cost breakdown components sum to total cost."""
        if v is None:
            return v
        
        # Calculate sum of all components
        component_sum = (
            v.commission.amount + 
            v.spread.amount + 
            v.market_impact.amount
        )
        
        # Add optional components
        if v.taxes:
            component_sum += v.taxes.amount
        if v.fees:
            component_sum += v.fees.amount
        if v.borrowing_cost:
            component_sum += v.borrowing_cost.amount
        if v.opportunity_cost:
            component_sum += v.opportunity_cost.amount
        
        # Add other components
        for component in v.other_components:
            component_sum += component.amount
        
        # Allow for small rounding differences
        tolerance = Decimal('0.01')
        if abs(component_sum - v.total_cost) > tolerance:
            raise ValueError(
                f'Cost breakdown components ({component_sum}) do not sum to total cost ({v.total_cost})'
            )
        
        return v

    class Config:
        """Pydantic model configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v)
        }
        
        schema_extra = {
            "example": {
                "trade_id": "TRADE_20241225_001",
                "symbol": "AAPL",
                "asset_class": "equity",
                "order_side": "buy",
                "order_type": "market",
                "quantity": 10000.0,
                "currency": "USD",
                "venue": "NYSE",
                "venue_type": "exchange",
                "cost_breakdown": {
                    "total_cost": 125.50,
                    "total_cost_bps": 8.5,
                    "currency": "USD",
                    "commission": {
                        "component_type": "commission",
                        "amount": 25.00,
                        "currency": "USD",
                        "basis_points": 1.7,
                        "description": "Brokerage commission"
                    },
                    "spread": {
                        "component_type": "spread",
                        "amount": 50.25,
                        "currency": "USD", 
                        "basis_points": 3.4,
                        "description": "Bid-ask spread cost"
                    },
                    "market_impact": {
                        "component_type": "market_impact",
                        "amount": 50.25,
                        "currency": "USD",
                        "basis_points": 3.4,
                        "description": "Market impact cost"
                    }
                }
            }
        }

# Alias for backward compatibility and simplified usage
TransactionCostBreakdown = CostBreakdown

# Additional missing classes that might be needed
class CostEstimate(BaseModel):
    """
    Cost estimation model for pre-trade analysis.
    
    Provides estimated transaction costs before trade execution,
    including confidence intervals and model assumptions.
    """
    estimate_id: str = Field(..., description="Unique estimate identifier")
    symbol: str = Field(..., description="Trading symbol")
    quantity: float = Field(..., description="Estimated quantity")
    side: OrderSide = Field(..., description="Order side")
    
    # Cost estimates
    estimated_cost_bps: float = Field(..., description="Estimated cost (bps)")
    estimated_cost_amount: Decimal = Field(..., description="Estimated cost amount")
    currency: CurrencyCode = Field(..., description="Currency")
    
    # Confidence intervals
    confidence_level: float = Field(0.95, description="Confidence level")
    lower_bound_bps: float = Field(..., description="Lower confidence bound (bps)")
    upper_bound_bps: float = Field(..., description="Upper confidence bound (bps)")
    
    # Estimate breakdown
    commission_estimate: float = Field(..., description="Commission estimate (bps)")
    spread_estimate: float = Field(..., description="Spread estimate (bps)")
    impact_estimate: float = Field(..., description="Impact estimate (bps)")
    
    # Model information
    model_name: str = Field(..., description="Estimation model name")
    model_version: str = Field("1.0.0", description="Model version")
    
    # Market context
    market_conditions: Dict[str, Any] = Field(default_factory=dict, description="Market conditions")
    volatility: Optional[float] = Field(None, description="Asset volatility")
    liquidity_score: Optional[float] = Field(None, description="Liquidity score")
    
    # Timing
    estimate_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Estimate timestamp")
    valid_until: Optional[datetime] = Field(None, description="Estimate validity period")
    
    @validator('quantity')
    def validate_quantity(cls, v):
        if v <= 0:
            raise ValueError('Quantity must be positive')
        return v
