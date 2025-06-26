"""
Execution Schema - Execution Quality Analysis Models

This module defines data structures for post-trade execution analysis,
including execution reports, quality metrics, and benchmark comparisons.

Key Models:
- ExecutionReport: Comprehensive execution analysis
- TradeExecution: Individual trade execution details
- QualityMetrics: Execution quality measurements
- BenchmarkComparison: Performance vs benchmarks

Author: FinAgent Development Team
License: OpenMDW
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from decimal import Decimal
from enum import Enum

class ExecutionStatus(str, Enum):
    """Trade execution status."""
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class BenchmarkType(str, Enum):
    """Benchmark types for execution comparison."""
    ARRIVAL_PRICE = "arrival_price"
    TWAP = "twap"
    VWAP = "vwap"
    CLOSE_PRICE = "close_price"
    OPEN_PRICE = "open_price"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"

class ExecutionQuality(str, Enum):
    """Execution quality classification."""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    VERY_POOR = "very_poor"

class Fill(BaseModel):
    """
    Individual trade fill within an execution.
    
    Represents a single fill that contributes to the overall
    execution of a parent order.
    """
    fill_id: str = Field(..., description="Unique fill identifier")
    fill_time: datetime = Field(..., description="Fill timestamp")
    fill_price: float = Field(..., description="Fill price")
    fill_quantity: float = Field(..., description="Fill quantity")
    venue: str = Field(..., description="Execution venue")
    
    # Fill characteristics
    is_aggressive: bool = Field(..., description="Whether fill was aggressive (market order)")
    counterparty_type: Optional[str] = Field(None, description="Counterparty type if known")
    
    # Cost components for this fill
    commission: Decimal = Field(..., description="Commission for this fill")
    fees: Decimal = Field(default=Decimal('0'), description="Fees for this fill")
    
    # Market context at time of fill
    mid_price_at_fill: Optional[float] = Field(None, description="Mid price at time of fill")
    spread_at_fill: Optional[float] = Field(None, description="Spread at time of fill")
    
    @validator('fill_quantity')
    def validate_fill_quantity(cls, v):
        if v <= 0:
            raise ValueError('Fill quantity must be positive')
        return v

class TradeExecution(BaseModel):
    """
    Comprehensive trade execution details.
    
    This model captures complete information about how an order
    was executed, including all fills and execution metrics.
    """
    # Execution identification
    execution_id: str = Field(..., description="Unique execution identifier")
    parent_order_id: str = Field(..., description="Parent order identifier")
    
    # Order details
    symbol: str = Field(..., description="Trading symbol")
    side: str = Field(..., description="Order side (buy/sell)")
    order_type: str = Field(..., description="Order type")
    
    # Execution summary
    total_quantity: float = Field(..., description="Total order quantity")
    executed_quantity: float = Field(..., description="Executed quantity")
    remaining_quantity: float = Field(..., description="Remaining quantity")
    
    # Pricing
    average_fill_price: float = Field(..., description="Volume-weighted average fill price")
    arrival_price: float = Field(..., description="Price at order arrival")
    
    # Timing
    order_time: datetime = Field(..., description="Order submission time")
    first_fill_time: Optional[datetime] = Field(None, description="First fill time")
    last_fill_time: Optional[datetime] = Field(None, description="Last fill time")
    completion_time: Optional[datetime] = Field(None, description="Order completion time")
    
    # Execution status
    status: ExecutionStatus = Field(..., description="Current execution status")
    cancellation_reason: Optional[str] = Field(None, description="Cancellation reason if applicable")
    
    # Fill details
    fills: List[Fill] = Field(default_factory=list, description="Individual fills")
    number_of_venues: int = Field(..., description="Number of venues used")
    venue_distribution: Dict[str, float] = Field(default_factory=dict, description="Quantity by venue")
    
    # Cost summary
    total_commission: Decimal = Field(..., description="Total commission")
    total_fees: Decimal = Field(..., description="Total fees")
    estimated_market_impact: Optional[float] = Field(None, description="Estimated market impact (bps)")
    
    @validator('executed_quantity')
    def validate_executed_quantity(cls, v, values):
        if 'total_quantity' in values and v > values['total_quantity']:
            raise ValueError('Executed quantity cannot exceed total quantity')
        return v

class QualityMetrics(BaseModel):
    """
    Comprehensive execution quality metrics.
    
    This model provides detailed metrics for evaluating the
    quality of trade execution across multiple dimensions.
    """
    # Primary quality measures
    implementation_shortfall_bps: float = Field(..., description="Implementation shortfall (bps)")
    price_improvement_bps: float = Field(..., description="Price improvement vs arrival (bps)")
    
    # Benchmark deviations
    twap_deviation_bps: float = Field(..., description="TWAP deviation (bps)")
    vwap_deviation_bps: float = Field(..., description="VWAP deviation (bps)")
    close_deviation_bps: float = Field(..., description="Close price deviation (bps)")
    
    # Execution efficiency
    fill_rate: float = Field(..., ge=0.0, le=1.0, description="Order fill rate")
    execution_rate: float = Field(..., description="Shares per minute execution rate")
    participation_rate: float = Field(..., description="Market participation rate")
    
    # Timing metrics
    time_to_first_fill: Optional[float] = Field(None, description="Time to first fill (seconds)")
    total_execution_time: float = Field(..., description="Total execution time (minutes)")
    
    # Market impact measures
    measured_impact_bps: float = Field(..., description="Measured market impact (bps)")
    impact_duration: Optional[float] = Field(None, description="Impact duration (minutes)")
    price_reversal_bps: Optional[float] = Field(None, description="Price reversal after execution (bps)")
    
    # Venue efficiency
    effective_spread_captured: float = Field(..., description="Effective spread captured")
    venue_selection_score: Optional[float] = Field(None, description="Venue selection efficiency score")
    
    # Risk metrics
    tracking_error: Optional[float] = Field(None, description="Tracking error vs benchmark")
    sharpe_ratio: Optional[float] = Field(None, description="Risk-adjusted return ratio")
    
    # Overall quality score
    overall_quality_score: float = Field(..., ge=0.0, le=100.0, description="Overall quality score (0-100)")
    quality_classification: ExecutionQuality = Field(..., description="Quality classification")
    
    # Confidence and reliability
    metric_confidence: float = Field(..., ge=0.0, le=1.0, description="Metric reliability confidence")
    data_completeness: float = Field(..., ge=0.0, le=1.0, description="Data completeness score")

class BenchmarkComparison(BaseModel):
    """
    Comparison of execution performance against various benchmarks.
    
    This model provides detailed comparison of actual execution
    performance against industry-standard benchmarks.
    """
    benchmark_type: BenchmarkType = Field(..., description="Benchmark type")
    benchmark_name: str = Field(..., description="Benchmark identifier")
    
    # Benchmark values
    benchmark_price: float = Field(..., description="Benchmark price")
    actual_price: float = Field(..., description="Actual execution price")
    
    # Performance comparison
    outperformance_bps: float = Field(..., description="Outperformance vs benchmark (bps)")
    outperformance_amount: Decimal = Field(..., description="Outperformance amount")
    
    # Statistical measures
    percentile_rank: Optional[float] = Field(None, description="Percentile rank vs peer universe")
    z_score: Optional[float] = Field(None, description="Z-score vs historical distribution")
    
    # Benchmark calculation details
    calculation_period: str = Field(..., description="Benchmark calculation period")
    market_participation_weight: Optional[float] = Field(None, description="Market participation weighting")
    
    # Confidence and validity
    benchmark_confidence: float = Field(..., ge=0.0, le=1.0, description="Benchmark reliability")
    is_valid_comparison: bool = Field(..., description="Whether comparison is statistically valid")
    
    comparison_notes: Optional[str] = Field(None, description="Additional comparison notes")

class ExecutionReport(BaseModel):
    """
    Comprehensive execution analysis report.
    
    This is the primary model for post-trade execution analysis,
    providing complete evaluation of execution quality and performance.
    """
    # Report identification
    report_id: str = Field(..., description="Unique report identifier")
    trade_id: str = Field(..., description="Associated trade identifier")
    
    # Execution details
    execution: TradeExecution = Field(..., description="Detailed execution information")
    
    # Quality analysis
    quality_metrics: QualityMetrics = Field(..., description="Comprehensive quality metrics")
    
    # Benchmark analysis
    benchmark_comparisons: List[BenchmarkComparison] = Field(
        default_factory=list, 
        description="Benchmark performance comparisons"
    )
    
    # Cost analysis
    total_transaction_cost_bps: float = Field(..., description="Total transaction cost (bps)")
    cost_breakdown: Dict[str, float] = Field(default_factory=dict, description="Cost component breakdown")
    
    # Market context
    market_conditions: Dict[str, Any] = Field(default_factory=dict, description="Market conditions during execution")
    liquidity_analysis: Optional[Dict[str, Any]] = Field(None, description="Liquidity analysis")
    
    # Attribution analysis
    performance_attribution: Dict[str, float] = Field(
        default_factory=dict, 
        description="Performance attribution by factor"
    )
    
    # Recommendations
    optimization_opportunities: List[str] = Field(
        default_factory=list, 
        description="Identified optimization opportunities"
    )
    recommended_improvements: List[str] = Field(
        default_factory=list,
        description="Recommended execution improvements"
    )
    
    # Report metadata
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")
    analyst_id: Optional[str] = Field(None, description="Analyst identifier")
    report_version: str = Field("1.0.0", description="Report version")
    
    # Quality assurance
    data_quality_issues: List[str] = Field(default_factory=list, description="Identified data quality issues")
    analysis_limitations: List[str] = Field(default_factory=list, description="Analysis limitations")
    
    # Validation
    validation_status: str = Field("pending", description="Report validation status")
    peer_review_status: Optional[str] = Field(None, description="Peer review status")

    class Config:
        """Pydantic model configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v)
        }
        
        schema_extra = {
            "example": {
                "report_id": "EXEC_RPT_20241225_001",
                "trade_id": "TRADE_20241225_001",
                "execution": {
                    "execution_id": "EXEC_20241225_001",
                    "parent_order_id": "ORDER_20241225_001",
                    "symbol": "AAPL",
                    "side": "buy",
                    "order_type": "market",
                    "total_quantity": 10000,
                    "executed_quantity": 10000,
                    "remaining_quantity": 0,
                    "average_fill_price": 185.75,
                    "arrival_price": 185.50,
                    "status": "filled",
                    "number_of_venues": 3,
                    "total_commission": 25.00,
                    "total_fees": 5.50
                },
                "quality_metrics": {
                    "implementation_shortfall_bps": 13.5,
                    "price_improvement_bps": -13.5,
                    "twap_deviation_bps": 8.2,
                    "vwap_deviation_bps": 5.7,
                    "fill_rate": 1.0,
                    "execution_rate": 500.0,
                    "participation_rate": 0.08,
                    "total_execution_time": 20.0,
                    "measured_impact_bps": 12.1,
                    "overall_quality_score": 78.5,
                    "quality_classification": "good",
                    "metric_confidence": 0.92,
                    "data_completeness": 0.98
                },
                "total_transaction_cost_bps": 16.4
            }
        }

class ExecutionAnalysisRequest(BaseModel):
    """
    Request model for post-trade execution analysis.
    
    This model defines the parameters and specifications for
    analyzing the quality and performance of trade executions.
    """
    # Request identification
    request_id: str = Field(..., description="Unique request identifier")
    requester_id: str = Field(..., description="Requester identifier")
    
    # Execution to analyze
    trade_id: str = Field(..., description="Trade identifier to analyze")
    execution_id: Optional[str] = Field(None, description="Specific execution identifier")
    
    # Analysis scope
    analysis_type: str = Field("comprehensive", description="Analysis type (basic/comprehensive/custom)")
    include_benchmarks: bool = Field(True, description="Include benchmark comparisons")
    include_attribution: bool = Field(True, description="Include performance attribution")
    
    # Benchmark specifications
    benchmarks_to_include: List[BenchmarkType] = Field(
        default_factory=lambda: [BenchmarkType.TWAP, BenchmarkType.VWAP, BenchmarkType.ARRIVAL_PRICE],
        description="Benchmarks to include in analysis"
    )
    
    # Analysis parameters
    market_data_source: str = Field("primary", description="Market data source")
    time_zone: str = Field("UTC", description="Time zone for analysis")
    
    # Peer comparison
    include_peer_comparison: bool = Field(False, description="Include peer universe comparison")
    peer_universe: Optional[str] = Field(None, description="Peer universe identifier")
    
    # Custom analysis parameters
    custom_metrics: List[str] = Field(default_factory=list, description="Custom metrics to calculate")
    analysis_filters: Dict[str, Any] = Field(default_factory=dict, description="Analysis filters")
    
    # Output preferences
    include_charts: bool = Field(False, description="Include visualization charts")
    output_format: str = Field("json", description="Output format (json/pdf/excel)")
    detail_level: str = Field("standard", description="Detail level (summary/standard/detailed)")
    
    # Request metadata
    request_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")
    priority: int = Field(5, ge=1, le=10, description="Request priority")
    expected_completion_time: Optional[int] = Field(None, description="Expected completion time (minutes)")

class ExecutionAnalysisResult(BaseModel):
    """
    Comprehensive result model for execution analysis.
    
    This model contains the complete results of post-trade
    execution analysis including quality metrics and recommendations.
    """
    # Result identification
    result_id: str = Field(..., description="Unique result identifier")
    request_id: str = Field(..., description="Associated request identifier")
    
    # Analysis summary
    analysis_status: str = Field(..., description="Analysis completion status")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Overall execution score")
    quality_classification: ExecutionQuality = Field(..., description="Quality classification")
    
    # Core execution analysis
    execution_report: ExecutionReport = Field(..., description="Detailed execution report")
    
    # Performance summary
    cost_performance_bps: float = Field(..., description="Cost performance (bps)")
    time_performance: float = Field(..., description="Time performance score")
    market_impact_bps: float = Field(..., description="Measured market impact (bps)")
    
    # Benchmark analysis
    benchmark_results: List[BenchmarkComparison] = Field(
        default_factory=list,
        description="Benchmark comparison results"
    )
    
    # Attribution analysis
    performance_attribution: Dict[str, float] = Field(
        default_factory=dict,
        description="Performance attribution by factor"
    )
    
    # Improvement opportunities
    cost_savings_potential: float = Field(..., description="Potential cost savings (bps)")
    optimization_recommendations: List[str] = Field(
        default_factory=list,
        description="Optimization recommendations"
    )
    
    # Risk analysis
    execution_risk_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Execution risk metrics"
    )
    
    # Market context
    market_conditions_impact: Dict[str, Any] = Field(
        default_factory=dict,
        description="Market conditions impact analysis"
    )
    
    # Statistical analysis
    confidence_intervals: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Confidence intervals for key metrics"
    )
    peer_comparison: Optional[Dict[str, float]] = Field(None, description="Peer comparison results")
    
    # Analysis metadata
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")
    analyst_id: Optional[str] = Field(None, description="Analyst identifier")
    analysis_version: str = Field("1.0.0", description="Analysis version")
    
    # Data quality
    data_quality_score: float = Field(..., ge=0.0, le=1.0, description="Data quality score")
    data_completeness: float = Field(..., ge=0.0, le=1.0, description="Data completeness")
    analysis_limitations: List[str] = Field(default_factory=list, description="Analysis limitations")
    
    # Validation
    validation_status: str = Field("pending", description="Result validation status")
    quality_assurance_notes: Optional[str] = Field(None, description="QA notes")

    class Config:
        """Pydantic model configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v)
        }
