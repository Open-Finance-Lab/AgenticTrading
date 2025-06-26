"""
Transaction Cost Agent Pool - Agents Package Initialization

This module initializes the agents package, providing access to all
transaction cost analysis agents across different categories.

Author: FinAgent Development Team
License: OpenMDW
"""

# Pre-trade agents
from .pre_trade.impact_estimator import ImpactEstimator
from .pre_trade.cost_predictor import CostPredictor
from .pre_trade.venue_analyzer import VenueAnalyzer

# Post-trade agents (will be implemented)
# from .post_trade.execution_analyzer import ExecutionAnalyzer
# from .post_trade.slippage_analyzer import SlippageAnalyzer
# from .post_trade.attribution_engine import AttributionEngine

# Optimization agents (will be implemented)
# from .optimization.portfolio_optimizer import PortfolioOptimizer
# from .optimization.routing_optimizer import RoutingOptimizer
# from .optimization.timing_optimizer import TimingOptimizer

# Risk-adjusted agents (will be implemented)
# from .risk_adjusted.var_adjusted_cost import VaRAdjustedCostAnalyzer
# from .risk_adjusted.sharpe_cost_ratio import SharpeCostRatioAnalyzer
# from .risk_adjusted.drawdown_cost_impact import DrawdownCostImpactAnalyzer

__all__ = [
    # Pre-trade agents
    "ImpactEstimator",
    "CostPredictor", 
    "VenueAnalyzer",
    
    # Post-trade agents (to be uncommented when implemented)
    # "ExecutionAnalyzer",
    # "SlippageAnalyzer", 
    # "AttributionEngine",
    
    # Optimization agents (to be uncommented when implemented)
    # "PortfolioOptimizer",
    # "RoutingOptimizer",
    # "TimingOptimizer",
    
    # Risk-adjusted agents (to be uncommented when implemented)
    # "VaRAdjustedCostAnalyzer",
    # "SharpeCostRatioAnalyzer",
    # "DrawdownCostImpactAnalyzer"
]
