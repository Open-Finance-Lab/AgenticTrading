"""
Qlib Backtesting Pipeline

A simplified pipeline for backtesting factors and models using Qlib.
Inspired by Microsoft RD-Agent Qlib scenarios.
"""

from .interfaces import (
    BacktestInterface,
    FactorInterface,
    ModelInterface,
    EvaluationMetrics
)

from .factor_pipeline import (
    FactorBacktester,
    FactorEvaluator
)

from .model_pipeline import (
    ModelBacktester,
    ModelEvaluator
)

from .utils import (
    QlibConfig,
    DataProcessor,
    ResultProcessor
)

__version__ = "1.0.0"
__all__ = [
    "BacktestInterface",
    "FactorInterface", 
    "ModelInterface",
    "EvaluationMetrics",
    "FactorBacktester",
    "FactorEvaluator",
    "ModelBacktester",
    "ModelEvaluator",
    "QlibConfig",
    "DataProcessor",
    "ResultProcessor"
]
