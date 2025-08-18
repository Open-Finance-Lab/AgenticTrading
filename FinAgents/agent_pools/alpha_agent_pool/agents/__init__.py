"""
Alpha Agent Pool - Agents Module
================================

This module contains all the alpha generation agents for the FinAgent system.

Structure:
- agent_manager.py: Manages and coordinates all agents
- autonomous/: Self-directed agents that can operate independently  
- empirical/: Data-driven agents that discover patterns from market data
- theory_driven/: Agents that implement known financial theories and models
"""

# Import key agent classes for easy access
try:
    from .agent_manager import AlphaAgentManager
except ImportError:
    pass

try:
    from .autonomous.autonomous_agent import AutonomousAgent
except ImportError:
    pass

try:
    from .empirical.data_mining_agent import DataMiningAgent
    from .empirical.ml_pattern_agent import MLPatternAgent
except ImportError:
    pass

try:
    from .theory_driven.momentum_agent import MomentumAgent
    from .theory_driven.mean_reversion_agent import MeanReversionAgent
except ImportError:
    pass

__all__ = [
    'AlphaAgentManager',
    'AutonomousAgent', 
    'DataMiningAgent',
    'MLPatternAgent',
    'MomentumAgent',
    'MeanReversionAgent'
]
