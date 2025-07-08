"""
Portfolio Construction Agent Pool

This module provides sophisticated portfolio construction capabilities by integrating
signals from alpha generation, risk management, and transaction cost analysis.
The pool leverages machine learning and optimization techniques to construct
optimal portfolios based on multi-agent inputs.

Author: Jifeng Li
Created: 2025-06-30
License: openMDW
"""

from .core import PortfolioConstructionAgentPool, PortfolioConstructionMCPServer
from .memory_bridge import (
    PortfolioConstructionMemoryBridge,
    PortfolioRecord,
    OptimizationResult,
    PortfolioMetrics
)

__all__ = [
    'PortfolioConstructionAgentPool',
    'PortfolioConstructionMCPServer', 
    'PortfolioConstructionMemoryBridge',
    'PortfolioRecord',
    'OptimizationResult',
    'PortfolioMetrics'
]
