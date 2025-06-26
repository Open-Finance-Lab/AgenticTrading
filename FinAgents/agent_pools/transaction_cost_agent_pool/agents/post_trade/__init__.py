"""
Post-trade Agents - Package Initialization

This module initializes the post-trade agents package, providing access
to all agents responsible for post-trade execution analysis and attribution.

Author: FinAgent Development Team
License: OpenMDW
"""

from .execution_analyzer import ExecutionAnalyzer
from .slippage_analyzer import SlippageAnalyzer
from .attribution_engine import AttributionEngine

__all__ = [
    'ExecutionAnalyzer',
    'SlippageAnalyzer', 
    'AttributionEngine'
]

from .execution_analyzer import ExecutionAnalyzer
from .slippage_analyzer import SlippageAnalyzer
from .attribution_engine import AttributionEngine

__all__ = [
    "ExecutionAnalyzer",
    "SlippageAnalyzer", 
    "AttributionEngine"
]
