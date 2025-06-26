"""
Transaction Cost Agent Pool - Core Package Initialization

This module initializes the Transaction Cost Agent Pool package, providing
centralized access to all core components and ensuring proper module loading.

Author: FinAgent Development Team
Version: 1.0.0
License: OpenMDW
"""

from typing import Dict, Any, Optional
import logging

# Configure package-level logging
logger = logging.getLogger(__name__)

# Package metadata
__version__ = "1.0.0"
__author__ = "FinAgent Development Team"
__license__ = "OpenMDW"
__description__ = "Enterprise-grade transaction cost analysis and optimization"

# Core component imports
from .core import TransactionCostAgentPool
from .registry import AGENT_REGISTRY, register_agent, get_agent

# Memory bridge functions - updated for External Memory Agent compatibility
try:
    from .memory_bridge import (
        TransactionCostMemoryBridge,
        create_transaction_cost_memory_bridge,
        log_transaction_cost_event,
        get_cost_analysis_history
    )
    # Legacy function aliases for backward compatibility
    def record_cost_event(*args, **kwargs):
        """Legacy function for recording cost events."""
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("record_cost_event is deprecated. Use TransactionCostMemoryBridge instead.")
        return None
    
    def retrieve_cost_history(*args, **kwargs):
        """Legacy function for retrieving cost history."""
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("retrieve_cost_history is deprecated. Use TransactionCostMemoryBridge.query_historical_executions instead.")
        return []
        
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Enhanced memory bridge not available: {e}")
    
    # Fallback implementations
    def record_cost_event(*args, **kwargs):
        logger.warning("Memory bridge not available - event not recorded")
        return None
    
    def retrieve_cost_history(*args, **kwargs):
        logger.warning("Memory bridge not available - returning empty history")
        return []

# Schema and model imports
from .schema.cost_models import (
    TransactionCost,
    MarketImpactModel,
    ExecutionMetrics,
    CostAttribute
)

# Agent imports
from .agents.pre_trade.impact_estimator import ImpactEstimator
from .agents.pre_trade.cost_predictor import CostPredictor
from .agents.post_trade.execution_analyzer import ExecutionAnalyzer
from .agents.optimization.cost_optimizer import CostOptimizer

__all__ = [
    # Core components
    "TransactionCostAgentPool",
    "AGENT_REGISTRY",
    "register_agent",
    "get_agent",
    
    # Memory and event handling
    "record_cost_event",
    "retrieve_cost_history",
    
    # Data models
    "TransactionCost",
    "MarketImpactModel", 
    "ExecutionMetrics",
    "CostAttribute",
    
    # Core agents
    "ImpactEstimator",
    "CostPredictor",
    "ExecutionAnalyzer",
    "CostOptimizer",
    
    # Package metadata
    "__version__",
    "__author__",
    "__license__",
    "__description__"
]

# Package initialization logging
logger.info(f"Transaction Cost Agent Pool v{__version__} initialized")
logger.info(f"Available agents: {len(AGENT_REGISTRY)} registered")

def get_version() -> str:
    """
    Retrieve the current package version.
    
    Returns:
        str: Version string in semantic versioning format
    """
    return __version__

def get_package_info() -> Dict[str, str]:
    """
    Retrieve comprehensive package information.
    
    Returns:
        Dict[str, str]: Package metadata including version, author, and license
    """
    return {
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "description": __description__
    }

def list_available_agents() -> Dict[str, Any]:
    """
    List all available agents in the registry.
    
    Returns:
        Dict[str, Any]: Dictionary of available agents with their configurations
    """
    return {
        agent_id: {
            "class": agent_config.get("class", "Unknown"),
            "description": agent_config.get("description", "No description available"),
            "capabilities": agent_config.get("capabilities", [])
        }
        for agent_id, agent_config in AGENT_REGISTRY.items()
    }
