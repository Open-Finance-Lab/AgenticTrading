"""
Transaction Cost Agent Pool - Schema Package Initialization

This module initializes the schema package, providing access to all data models
and validation schemas used throughout the transaction cost analysis system.

Author: FinAgent Development Team  
License: OpenMDW
"""

from .cost_models import (
    TransactionCost,
    CostBreakdown,
    MarketImpactModel,
    ExecutionMetrics,
    CostAttribute,
    PerformanceBenchmark
)

from .market_impact_schema import (
    ImpactEstimate,
    TemporaryPermanentImpact,
    MarketMicrostructure,
    OrderSpecification
)

from .execution_schema import (
    ExecutionReport,
    TradeExecution,
    QualityMetrics,
    BenchmarkComparison
)

from .optimization_schema import (
    OptimizationParameters,
    ExecutionStrategy,
    PortfolioTrade,
    OptimizationResult
)

__all__ = [
    # Core cost models
    "TransactionCost",
    "CostBreakdown", 
    "MarketImpactModel",
    "ExecutionMetrics",
    "CostAttribute",
    "PerformanceBenchmark",
    
    # Market impact models
    "ImpactEstimate",
    "TemporaryPermanentImpact",
    "MarketMicrostructure",
    "OrderSpecification",
    
    # Execution models
    "ExecutionReport",
    "TradeExecution",
    "QualityMetrics",
    "BenchmarkComparison",
    
    # Optimization models
    "OptimizationParameters",
    "ExecutionStrategy",
    "PortfolioTrade",
    "OptimizationResult"
]
