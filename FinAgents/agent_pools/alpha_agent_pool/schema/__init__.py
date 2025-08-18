"""
Alpha Agent Pool Schema Module
==============================

This module contains all schema definitions for the Alpha Agent Pool,
including configuration schemas, data models, and type definitions.

Author: FinAgent Development Team
Created: 2025-07-25
"""

# Import all schema classes for easy access
try:
    from .theory_driven_schema import (
        MomentumAgentConfig as TheoryMomentumConfig,
        MomentumSignalRequest,
        AlphaStrategyFlow,
        MarketContext,
        Decision,
        Action,
        PerformanceFeedback,
        Metadata
    )
except ImportError:
    pass

try:
    from .agent_config import *
except ImportError:
    pass

try:
    from .config_schema import (
        AgentType,
        FactorCategory,
        AgentConfig,
        MomentumAgentConfig,
        MeanReversionAgentConfig,
        EmpiricalAgentConfig,
        AutonomousAgentConfig,
        AlphaPoolConfig,
        create_default_config,
        create_research_config,
        create_production_config,
        validate_config
    )
except ImportError:
    pass

__all__ = [
    # Theory-driven schemas
    'TheoryMomentumConfig',
    'MomentumSignalRequest',
    'AlphaStrategyFlow',
    'MarketContext',
    'Decision',
    'Action',
    'PerformanceFeedback',
    'Metadata',
    
    # Configuration schemas
    'AgentType',
    'FactorCategory',
    'AgentConfig',
    'MomentumAgentConfig',
    'MeanReversionAgentConfig', 
    'EmpiricalAgentConfig',
    'AutonomousAgentConfig',
    'AlphaPoolConfig',
    'create_default_config',
    'create_research_config',
    'create_production_config',
    'validate_config'
]
