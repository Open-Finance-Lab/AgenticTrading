"""
Backtesting Module Initialization

This module provides imports for all backtesting framework components.
"""

from .event_driven_backtester import (
    EventDrivenBacktester,
    AlphaFactorEvaluator,
    TransactionCostModel,
    PerformanceMetrics,
    MarketEvent,
    OrderEvent,
    FillEvent,
    EventType
)

__all__ = [
    "EventDrivenBacktester",
    "AlphaFactorEvaluator", 
    "TransactionCostModel",
    "PerformanceMetrics",
    "MarketEvent",
    "OrderEvent",
    "FillEvent",
    "EventType"
]
