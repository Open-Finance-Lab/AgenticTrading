"""
Risk Agents Package - Specialized Risk Analysis Agents

This package contains specialized agents for different types of risk analysis
including market risk, credit risk, operational risk, and liquidity risk.

Author: Jifeng Li
License: openMDW
"""

from .market_risk import MarketRiskAnalyzer
from .volatility import VolatilityAnalyzer
from .var_calculator import VaRCalculator
from .credit_risk import CreditRiskAnalyzer
from .liquidity_risk import LiquidityRiskAnalyzer
from .operational_risk import OperationalRiskAnalyzer
from .stress_testing import StressTester
from .model_risk import ModelRiskManager

__all__ = [
    "MarketRiskAnalyzer",
    "VolatilityAnalyzer", 
    "VaRCalculator",
    "CreditRiskAnalyzer",
    "LiquidityRiskAnalyzer",
    "OperationalRiskAnalyzer",
    "StressTester",
    "ModelRiskManager"
]
