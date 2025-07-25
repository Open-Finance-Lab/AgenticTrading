"""
Alpha Agent Pool Module

This module provides comprehensive multi-agent orchestration capabilities for
quantitative alpha generation and trading strategy management. The system
implements Agent-to-Agent (A2A) protocol integration for distributed memory
coordination and cross-agent learning facilitation.

Core Components:
- AlphaAgentPoolMCPServer: Central orchestration server
- A2AMemoryCoordinator: Distributed memory coordination system
- MemoryBridge: Legacy memory integration interface
- IntegrationTestSuite: Comprehensive testing framework

Academic Framework:
This implementation follows established principles from multi-agent systems
theory, quantitative finance literature, and modern MLOps practices for
algorithmic trading system deployment.

Author: FinAgent Research Team
License: Open Source Research License
Created: 2025-07-25
"""

# Core server and orchestration components
from .core import AlphaAgentPoolMCPServer

# A2A protocol integration components
try:
    from .a2a_memory_coordinator import (
        AlphaPoolA2AMemoryCoordinator,
        initialize_pool_coordinator,
        get_pool_coordinator,
        shutdown_pool_coordinator
    )
    A2A_COMPONENTS_AVAILABLE = True
except ImportError:
    A2A_COMPONENTS_AVAILABLE = False

# Memory bridge and integration components
try:
    from .memory_bridge import (
        AlphaAgentPoolMemoryBridge,
        create_alpha_memory_bridge,
        AlphaSignalRecord,
        StrategyPerformanceMetrics
    )
    MEMORY_BRIDGE_AVAILABLE = True
except ImportError:
    MEMORY_BRIDGE_AVAILABLE = False

# Testing and validation components
try:
    from .a2a_integration_test import A2AIntegrationTestSuite
    INTEGRATION_TESTS_AVAILABLE = True
except ImportError:
    INTEGRATION_TESTS_AVAILABLE = False

# Package metadata
__version__ = "1.0.0"
__author__ = "FinAgent Research Team"
__license__ = "Open Source Research License"

# Public API exports
__all__ = [
    # Core server
    "AlphaAgentPoolMCPServer",
    
    # A2A coordination (if available)
    "AlphaPoolA2AMemoryCoordinator",
    "initialize_pool_coordinator", 
    "get_pool_coordinator",
    "shutdown_pool_coordinator",
    
    # Memory bridge (if available)
    "AlphaAgentPoolMemoryBridge",
    "create_alpha_memory_bridge",
    "AlphaSignalRecord",
    "StrategyPerformanceMetrics",
    
    # Testing framework (if available)
    "A2AIntegrationTestSuite",
    
    # Availability flags
    "A2A_COMPONENTS_AVAILABLE",
    "MEMORY_BRIDGE_AVAILABLE", 
    "INTEGRATION_TESTS_AVAILABLE"
]

# Module-level availability reporting
def get_available_components():
    """
    Report availability of optional components.
    
    Returns:
        Dict containing availability status of optional components
    """
    return {
        "a2a_coordination": A2A_COMPONENTS_AVAILABLE,
        "memory_bridge": MEMORY_BRIDGE_AVAILABLE,
        "integration_tests": INTEGRATION_TESTS_AVAILABLE,
        "core_server": True  # Always available
    }

# Academic-style module documentation
def get_module_info():
    """
    Provide comprehensive module information for academic documentation.
    
    Returns:
        Dict containing module metadata and component information
    """
    return {
        "module_name": "FinAgents.agent_pools.alpha_agent_pool",
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "description": "Multi-agent alpha generation and trading strategy orchestration",
        "components": get_available_components(),
        "academic_framework": "Multi-agent systems theory with quantitative finance applications",
        "documentation": __doc__
    }