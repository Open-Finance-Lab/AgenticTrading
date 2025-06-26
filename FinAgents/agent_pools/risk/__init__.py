"""
Risk Agent Pool - Financial Risk Management System

This package provides a comprehensive risk management system for financial portfolios,
incorporating multiple risk models, measurement techniques, and analysis capabilities.

Author: Jifeng Li
License: openMDW
"""

from .core import RiskAgentPool
from .registry import AGENT_REGISTRY

__all__ = ["RiskAgentPool", "AGENT_REGISTRY"]
