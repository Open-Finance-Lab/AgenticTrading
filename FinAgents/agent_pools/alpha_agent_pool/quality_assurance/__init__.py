"""
Alpha Agent Pool Quality Assurance Framework

This module provides comprehensive testing and validation infrastructure for alpha agent pool,
focusing on alpha factor discovery, strategy validation, and reinforcement learning updates.

Core Components:
- Alpha factor quality metrics and statistical testing
- Event-driven backtesting framework with transaction cost modeling
- Performance attribution and risk decomposition
- Reinforcement learning validation and policy update testing
- Cross-agent collaboration validation

Architecture:
- tests/: Unit and integration tests for agent functionality
- backtesting/: Event-driven backtesting framework with performance analytics
- metrics/: Alpha factor quality metrics and risk-adjusted performance measures
- reports/: Automated report generation and performance visualization

Author: FinAgent Quality Assurance Team
License: Open Source Research License
Created: 2025-07-25
"""

__version__ = "1.0.0"
__author__ = "FinAgent Quality Assurance Team"

# Quality assurance framework components
from .tests import (
    AlphaFactorQualityTests,
    AgentInteractionTests,
    PerformanceValidationTests
)

from .backtesting import (
    EventDrivenBacktester,
    AlphaFactorEvaluator,
    TransactionCostModel,
    PerformanceMetrics
)

from .metrics import (
    AlphaFactorMetrics,
    RiskAdjustedMetrics,
    CrossSectionalMetrics
)

__all__ = [
    "AlphaFactorQualityTests",
    "AgentInteractionTests", 
    "PerformanceValidationTests",
    "EventDrivenBacktester",
    "AlphaFactorEvaluator",
    "PerformanceAnalyzer",
    "AlphaFactorMetrics",
    "RiskAdjustedMetrics",
    "CrossSectionalMetrics"
]
