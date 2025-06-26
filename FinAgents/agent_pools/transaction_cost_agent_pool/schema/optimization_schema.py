"""
Optimization Schema - Execution Optimization Models

This module defines data structures for transaction cost optimization,
including optimization parameters, execution strategies, and results.

Key Models:
- OptimizationParameters: Optimization configuration
- ExecutionStrategy: Strategy specification
- PortfolioTrade: Portfolio-level trade representation
- OptimizationResult: Optimization outcome analysis

Author: FinAgent Development Team
License: OpenMDW
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum

class OptimizationObjective(str, Enum):
    """Optimization objectives for execution strategies."""
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_RISK = "minimize_risk"
    MINIMIZE_IMPACT = "minimize_impact"
    MAXIMIZE_ALPHA = "maximize_alpha"
    MINIMIZE_TRACKING_ERROR = "minimize_tracking_error"
    CUSTOM = "custom"

class ExecutionAlgorithm(str, Enum):
    """Available execution algorithms."""
    TWAP = "twap"
    VWAP = "vwap"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    ARRIVAL_PRICE = "arrival_price"
    PERCENT_OF_VOLUME = "percent_of_volume"
    ICEBERG = "iceberg"
    SMART_ORDER_ROUTER = "smart_order_router"

class ConstraintType(str, Enum):
    """Types of optimization constraints."""
    RISK_LIMIT = "risk_limit"
    POSITION_LIMIT = "position_limit"
    COST_LIMIT = "cost_limit"
    TIME_LIMIT = "time_limit"
    VENUE_CONSTRAINT = "venue_constraint"
    LIQUIDITY_CONSTRAINT = "liquidity_constraint"

class OptimizationStatus(str, Enum):
    """Optimization process status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Constraint(BaseModel):
    """
    Individual optimization constraint specification.
    
    Represents a single constraint that must be satisfied
    during the optimization process.
    """
    constraint_id: str = Field(..., description="Unique constraint identifier")
    constraint_type: ConstraintType = Field(..., description="Type of constraint")
    constraint_name: str = Field(..., description="Human-readable constraint name")
    
    # Constraint parameters
    limit_value: Optional[float] = Field(None, description="Constraint limit value")
    target_value: Optional[float] = Field(None, description="Constraint target value")
    tolerance: Optional[float] = Field(None, description="Constraint tolerance")
    
    # Applicability
    applies_to: List[str] = Field(default_factory=list, description="Assets/trades this constraint applies to")
    is_soft_constraint: bool = Field(False, description="Whether constraint is soft (can be violated with penalty)")
    penalty_factor: Optional[float] = Field(None, description="Penalty factor for soft constraint violations")
    
    # Priority and weighting
    priority: int = Field(1, description="Constraint priority (1=highest)")
    weight: float = Field(1.0, description="Constraint weight in optimization")
    
    # Validation
    is_active: bool = Field(True, description="Whether constraint is active")
    description: Optional[str] = Field(None, description="Detailed constraint description")

class OptimizationParameters(BaseModel):
    """
    Comprehensive optimization parameters and configuration.
    
    This model defines all parameters controlling the optimization
    process for transaction cost minimization.
    """
    # Basic optimization settings
    optimization_id: str = Field(..., description="Unique optimization identifier")
    objective: OptimizationObjective = Field(..., description="Primary optimization objective")
    time_horizon: int = Field(..., description="Optimization time horizon (minutes)")
    
    # Multi-objective optimization
    secondary_objectives: List[OptimizationObjective] = Field(
        default_factory=list, 
        description="Secondary optimization objectives"
    )
    objective_weights: Dict[str, float] = Field(
        default_factory=dict, 
        description="Weights for multi-objective optimization"
    )
    
    # Constraints
    constraints: List[Constraint] = Field(default_factory=list, description="Optimization constraints")
    
    # Risk parameters
    max_risk_limit: Optional[float] = Field(None, description="Maximum risk limit")
    risk_aversion_factor: float = Field(1.0, description="Risk aversion factor")
    
    # Cost parameters
    max_cost_limit: Optional[float] = Field(None, description="Maximum cost limit (bps)")
    cost_penalty_factor: float = Field(1.0, description="Cost penalty factor")
    
    # Execution parameters
    allowed_algorithms: List[ExecutionAlgorithm] = Field(
        default_factory=list, 
        description="Allowed execution algorithms"
    )
    preferred_algorithm: Optional[ExecutionAlgorithm] = Field(None, description="Preferred execution algorithm")
    
    # Venue preferences
    allowed_venues: List[str] = Field(default_factory=list, description="Allowed execution venues")
    venue_preferences: Dict[str, float] = Field(default_factory=dict, description="Venue preference weights")
    
    # Market impact settings
    impact_model: str = Field("square_root", description="Market impact model to use")
    impact_aversion: float = Field(1.0, description="Market impact aversion factor")
    
    # Timing parameters
    start_time: Optional[datetime] = Field(None, description="Earliest execution start time")
    end_time: Optional[datetime] = Field(None, description="Latest execution end time")
    urgency_level: str = Field("normal", description="Execution urgency level")
    
    # Portfolio-level parameters
    portfolio_correlation_matrix: Optional[Dict[str, Dict[str, float]]] = Field(
        None, 
        description="Asset correlation matrix for portfolio optimization"
    )
    
    # Solver parameters
    solver_type: str = Field("mixed_integer", description="Optimization solver type")
    max_iterations: int = Field(1000, description="Maximum solver iterations")
    convergence_tolerance: float = Field(1e-6, description="Convergence tolerance")
    time_limit: Optional[int] = Field(None, description="Solver time limit (seconds)")
    
    # Advanced parameters
    custom_parameters: Dict[str, Any] = Field(default_factory=dict, description="Custom optimization parameters")

class ExecutionStrategy(BaseModel):
    """
    Detailed execution strategy specification.
    
    This model describes a specific execution strategy resulting
    from the optimization process.
    """
    strategy_id: str = Field(..., description="Unique strategy identifier")
    strategy_name: str = Field(..., description="Strategy name")
    algorithm: ExecutionAlgorithm = Field(..., description="Execution algorithm")
    
    # Strategy parameters
    participation_rate: Optional[float] = Field(None, description="Market participation rate")
    execution_duration: Optional[int] = Field(None, description="Execution duration (minutes)")
    
    # Venue routing
    venue_allocation: Dict[str, float] = Field(default_factory=dict, description="Venue allocation percentages")
    primary_venue: Optional[str] = Field(None, description="Primary execution venue")
    
    # Timing strategy
    start_time: Optional[datetime] = Field(None, description="Strategy start time")
    end_time: Optional[datetime] = Field(None, description="Strategy end time")
    time_distribution: Optional[Dict[str, float]] = Field(None, description="Time-based execution distribution")
    
    # Order sizing
    order_sizing_method: str = Field("equal_weight", description="Order sizing method")
    min_order_size: Optional[float] = Field(None, description="Minimum order size")
    max_order_size: Optional[float] = Field(None, description="Maximum order size")
    
    # Risk management
    stop_loss_threshold: Optional[float] = Field(None, description="Stop loss threshold (bps)")
    profit_taking_threshold: Optional[float] = Field(None, description="Profit taking threshold (bps)")
    
    # Expected performance
    expected_cost_bps: float = Field(..., description="Expected transaction cost (bps)")
    expected_impact_bps: float = Field(..., description="Expected market impact (bps)")
    expected_completion_time: Optional[int] = Field(None, description="Expected completion time (minutes)")
    
    # Confidence metrics
    strategy_confidence: float = Field(..., ge=0.0, le=1.0, description="Strategy confidence score")
    risk_score: float = Field(..., ge=0.0, le=10.0, description="Strategy risk score (0-10)")
    
    # Alternative strategies
    alternatives: List[str] = Field(default_factory=list, description="Alternative strategy identifiers")

class PortfolioTrade(BaseModel):
    """
    Portfolio-level trade specification for optimization.
    
    Represents a single trade within a portfolio context,
    including correlations and dependencies with other trades.
    """
    trade_id: str = Field(..., description="Unique trade identifier")
    symbol: str = Field(..., description="Trading symbol")
    side: str = Field(..., description="Trade side (buy/sell)")
    quantity: float = Field(..., description="Trade quantity")
    
    # Trade characteristics
    trade_type: str = Field("equity", description="Trade type")
    currency: str = Field("USD", description="Trade currency")
    
    # Timing preferences
    desired_start_time: Optional[datetime] = Field(None, description="Desired start time")
    desired_end_time: Optional[datetime] = Field(None, description="Desired end time")
    urgency: str = Field("normal", description="Trade urgency level")
    
    # Cost and risk targets
    target_cost_bps: Optional[float] = Field(None, description="Target cost (bps)")
    max_cost_bps: Optional[float] = Field(None, description="Maximum acceptable cost (bps)")
    risk_limit: Optional[float] = Field(None, description="Risk limit for this trade")
    
    # Portfolio context
    portfolio_weight: float = Field(..., description="Weight in portfolio")
    sector: Optional[str] = Field(None, description="Sector classification")
    market_cap: Optional[str] = Field(None, description="Market cap classification")
    
    # Dependencies
    dependent_trades: List[str] = Field(default_factory=list, description="Dependent trade identifiers")
    correlation_factors: Dict[str, float] = Field(default_factory=dict, description="Correlation with other trades")
    
    # Execution constraints
    venue_constraints: List[str] = Field(default_factory=list, description="Venue constraints")
    algorithm_constraints: List[ExecutionAlgorithm] = Field(
        default_factory=list, 
        description="Algorithm constraints"
    )
    
    # Current market data
    current_price: Optional[float] = Field(None, description="Current market price")
    bid_ask_spread: Optional[float] = Field(None, description="Current bid-ask spread")
    daily_volume: Optional[float] = Field(None, description="Average daily volume")
    
    @validator('quantity')
    def validate_quantity(cls, v):
        if v <= 0:
            raise ValueError('Trade quantity must be positive')
        return v
    
    @validator('portfolio_weight')
    def validate_portfolio_weight(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Portfolio weight must be between 0 and 1')
        return v

class OptimizationResult(BaseModel):
    """
    Comprehensive optimization result analysis.
    
    This model captures the complete results of the optimization
    process, including optimal strategies and performance analysis.
    """
    # Result identification
    result_id: str = Field(..., description="Unique result identifier")
    optimization_id: str = Field(..., description="Associated optimization identifier")
    
    # Optimization outcome
    status: OptimizationStatus = Field(..., description="Optimization status")
    objective_value: float = Field(..., description="Achieved objective value")
    
    # Optimal strategies
    optimal_strategies: Dict[str, ExecutionStrategy] = Field(
        default_factory=dict, 
        description="Optimal strategies by trade ID"
    )
    
    # Portfolio-level results
    total_expected_cost_bps: float = Field(..., description="Total expected cost (bps)")
    total_expected_impact_bps: float = Field(..., description="Total expected impact (bps)")
    portfolio_risk_score: float = Field(..., description="Portfolio risk score")
    
    # Performance improvement
    cost_improvement_bps: Optional[float] = Field(None, description="Cost improvement vs baseline (bps)")
    risk_improvement: Optional[float] = Field(None, description="Risk improvement vs baseline")
    
    # Execution schedule
    execution_schedule: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Detailed execution schedule"
    )
    
    # Constraint analysis
    constraint_violations: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Constraint violations if any"
    )
    constraint_slack: Dict[str, float] = Field(
        default_factory=dict, 
        description="Constraint slack values"
    )
    
    # Sensitivity analysis
    sensitivity_analysis: Dict[str, float] = Field(
        default_factory=dict, 
        description="Sensitivity to parameter changes"
    )
    
    # Alternative solutions
    alternative_solutions: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Alternative near-optimal solutions"
    )
    
    # Solver information
    solver_iterations: int = Field(..., description="Number of solver iterations")
    solver_time: float = Field(..., description="Solver computation time (seconds)")
    convergence_status: str = Field(..., description="Solver convergence status")
    
    # Quality metrics
    solution_quality: float = Field(..., ge=0.0, le=1.0, description="Solution quality score")
    confidence_level: float = Field(..., ge=0.0, le=1.0, description="Result confidence level")
    
    # Recommendations
    implementation_recommendations: List[str] = Field(
        default_factory=list, 
        description="Implementation recommendations"
    )
    risk_warnings: List[str] = Field(default_factory=list, description="Risk warnings")
    
    # Metadata
    optimization_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Optimization timestamp")
    analyst_id: Optional[str] = Field(None, description="Analyst identifier")
    model_version: str = Field("1.0.0", description="Optimization model version")
    
    # Validation
    validation_status: str = Field("pending", description="Result validation status")
    validation_notes: Optional[str] = Field(None, description="Validation notes")

    class Config:
        """Pydantic model configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v)
        }
        
        schema_extra = {
            "example": {
                "result_id": "OPT_RESULT_20241225_001",
                "optimization_id": "OPT_20241225_001",
                "status": "completed",
                "objective_value": 23.7,
                "optimal_strategies": {
                    "TRADE_001": {
                        "strategy_id": "STRAT_001",
                        "strategy_name": "TWAP_Optimized",
                        "algorithm": "twap",
                        "participation_rate": 0.08,
                        "execution_duration": 120,
                        "expected_cost_bps": 8.5,
                        "expected_impact_bps": 6.2,
                        "strategy_confidence": 0.91,
                        "risk_score": 3.2
                    }
                },
                "total_expected_cost_bps": 23.7,
                "total_expected_impact_bps": 18.4,
                "portfolio_risk_score": 4.1,
                "cost_improvement_bps": 5.8,
                "solver_iterations": 847,
                "solver_time": 12.3,
                "convergence_status": "optimal",
                "solution_quality": 0.94,
                "confidence_level": 0.89
            }
        }

class OrderToOptimize(BaseModel):
    """
    Individual order specification within an optimization request.
    
    Represents a single order that needs to be optimized as part
    of a larger optimization request.
    """
    order_id: str = Field(..., description="Unique order identifier")
    symbol: str = Field(..., description="Trading symbol")
    side: str = Field(..., description="Order side (buy/sell)")
    quantity: float = Field(..., description="Order quantity")
    
    # Order characteristics
    order_type: str = Field("market", description="Order type")
    currency: str = Field("USD", description="Order currency")
    asset_class: str = Field("equity", description="Asset class")
    
    # Timing preferences
    desired_start_time: Optional[datetime] = Field(None, description="Desired start time")
    desired_end_time: Optional[datetime] = Field(None, description="Desired end time")
    max_execution_time: Optional[int] = Field(None, description="Maximum execution time (minutes)")
    
    # Cost and risk constraints
    max_cost_bps: Optional[float] = Field(None, description="Maximum acceptable cost (bps)")
    max_impact_bps: Optional[float] = Field(None, description="Maximum acceptable impact (bps)")
    risk_tolerance: Optional[float] = Field(None, description="Risk tolerance level")
    
    # Execution preferences
    preferred_venues: List[str] = Field(default_factory=list, description="Preferred execution venues")
    allowed_algorithms: List[ExecutionAlgorithm] = Field(
        default_factory=list, 
        description="Allowed execution algorithms"
    )
    
    # Market context
    current_price: Optional[float] = Field(None, description="Current market price")
    volatility: Optional[float] = Field(None, description="Asset volatility")
    average_daily_volume: Optional[float] = Field(None, description="Average daily volume")
    
    # Priority and urgency
    priority: int = Field(5, ge=1, le=10, description="Order priority (1=highest, 10=lowest)")
    urgency_level: str = Field("normal", description="Urgency level (low/normal/high/urgent)")
    
    @validator('quantity')
    def validate_quantity(cls, v):
        if v <= 0:
            raise ValueError('Order quantity must be positive')
        return v

class OptimizationRequest(BaseModel):
    """
    Comprehensive optimization request encompassing multiple orders and parameters.
    
    This is the primary model for requesting transaction cost optimization
    across a portfolio of orders with specific constraints and objectives.
    """
    # Request identification
    request_id: str = Field(..., description="Unique request identifier")
    requester_id: str = Field(..., description="Requester identifier")
    
    # Orders to optimize
    orders: List[OrderToOptimize] = Field(..., description="Orders to be optimized")
    
    # Optimization configuration
    optimization_parameters: OptimizationParameters = Field(..., description="Optimization parameters")
    
    # Portfolio context
    portfolio_id: Optional[str] = Field(None, description="Portfolio identifier")
    portfolio_value: Optional[float] = Field(None, description="Total portfolio value")
    benchmark: Optional[str] = Field(None, description="Benchmark identifier")
    
    # Risk management
    portfolio_risk_limit: Optional[float] = Field(None, description="Portfolio-level risk limit")
    max_concentration: Optional[float] = Field(None, description="Maximum concentration in single asset")
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = Field(
        None, 
        description="Asset correlation matrix"
    )
    
    # Market conditions
    market_regime: Optional[str] = Field(None, description="Current market regime")
    volatility_regime: Optional[str] = Field(None, description="Current volatility regime")
    liquidity_conditions: Optional[Dict[str, Any]] = Field(None, description="Market liquidity conditions")
    
    # Business constraints
    settlement_date: Optional[datetime] = Field(None, description="Required settlement date")
    regulatory_constraints: List[str] = Field(default_factory=list, description="Regulatory constraints")
    compliance_requirements: List[str] = Field(default_factory=list, description="Compliance requirements")
    
    # Performance targets
    target_performance_bps: Optional[float] = Field(None, description="Target performance vs benchmark (bps)")
    max_tracking_error: Optional[float] = Field(None, description="Maximum tracking error")
    
    # Execution preferences
    execution_style: str = Field("balanced", description="Execution style (aggressive/balanced/passive)")
    allow_partial_fills: bool = Field(True, description="Whether to allow partial fills")
    dark_pool_preference: Optional[float] = Field(None, description="Dark pool preference (0.0-1.0)")
    
    # Advanced features
    allow_netting: bool = Field(True, description="Allow netting of opposing positions")
    enable_cross_trading: bool = Field(False, description="Enable cross trading optimization")
    use_ml_predictions: bool = Field(True, description="Use ML predictions in optimization")
    
    # Callback and notification
    callback_url: Optional[str] = Field(None, description="Callback URL for results")
    notification_preferences: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Notification preferences"
    )
    
    # Request metadata
    request_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")
    expected_response_time: Optional[int] = Field(None, description="Expected response time (seconds)")
    priority: int = Field(5, ge=1, le=10, description="Request priority")
    
    # Validation and quality
    validate_inputs: bool = Field(True, description="Whether to validate inputs")
    quality_checks: List[str] = Field(default_factory=list, description="Quality checks to perform")
    
    @validator('orders')
    def validate_orders(cls, v):
        if not v:
            raise ValueError('At least one order must be provided')
        return v
    
    @validator('portfolio_value')
    def validate_portfolio_value(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Portfolio value must be positive')
        return v

    class Config:
        """Pydantic model configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v)
        }
        
        schema_extra = {
            "example": {
                "request_id": "OPT_REQ_20241225_001",
                "requester_id": "TRADER_001",
                "orders": [
                    {
                        "order_id": "ORDER_001",
                        "symbol": "AAPL",
                        "side": "buy",
                        "quantity": 10000,
                        "order_type": "market",
                        "max_cost_bps": 15.0,
                        "priority": 3,
                        "urgency_level": "normal"
                    }
                ],
                "optimization_parameters": {
                    "optimization_id": "OPT_20241225_001",
                    "objective": "minimize_cost",
                    "time_horizon": 120,
                    "max_cost_limit": 20.0,
                    "allowed_algorithms": ["twap", "vwap"],
                    "solver_type": "mixed_integer"
                },
                "portfolio_id": "PORTFOLIO_001",
                "execution_style": "balanced",
                "priority": 5
            }
        }

class ExecutionRecommendation(BaseModel):
    """
    Execution recommendation for optimized trade execution.
    
    This model provides specific recommendations for executing
    trades based on optimization analysis.
    """
    recommendation_id: str = Field(..., description="Unique recommendation identifier")
    trade_id: str = Field(..., description="Associated trade identifier")
    
    # Recommendation type and priority
    recommendation_type: str = Field(..., description="Type of recommendation")
    priority: int = Field(..., ge=1, le=10, description="Recommendation priority (1=highest)")
    urgency: str = Field("normal", description="Urgency level")
    
    # Execution strategy recommendation
    recommended_algorithm: ExecutionAlgorithm = Field(..., description="Recommended execution algorithm")
    recommended_venues: List[str] = Field(default_factory=list, description="Recommended venues")
    
    # Timing recommendations
    recommended_start_time: Optional[datetime] = Field(None, description="Recommended start time")
    recommended_duration: Optional[int] = Field(None, description="Recommended duration (minutes)")
    participation_rate: Optional[float] = Field(None, description="Recommended participation rate")
    
    # Cost and risk expectations
    expected_cost_bps: float = Field(..., description="Expected cost (bps)")
    expected_savings_bps: Optional[float] = Field(None, description="Expected savings vs baseline (bps)")
    risk_score: float = Field(..., ge=0.0, le=10.0, description="Risk score (0-10)")
    
    # Rationale and supporting information
    rationale: str = Field(..., description="Rationale for recommendation")
    supporting_metrics: Dict[str, Any] = Field(default_factory=dict, description="Supporting metrics")
    
    # Implementation details
    implementation_notes: List[str] = Field(default_factory=list, description="Implementation notes")
    monitoring_requirements: List[str] = Field(default_factory=list, description="Monitoring requirements")
    
    # Confidence and validity
    confidence_level: float = Field(..., ge=0.0, le=1.0, description="Confidence in recommendation")
    valid_until: Optional[datetime] = Field(None, description="Recommendation validity period")
    
    # Metadata
    created_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    analyst_id: Optional[str] = Field(None, description="Analyst identifier")
    model_version: str = Field("1.0.0", description="Model version")

# Aliases for backward compatibility and simplified usage
OptimizationStrategy = ExecutionStrategy
