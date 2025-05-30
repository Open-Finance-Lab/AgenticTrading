"""
Protocol adapters for the FinAgent Orchestration system.
This module provides adapters for MCP, A2A, ANP, and ACP protocols.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class ProtocolAdapter(ABC):
    """Base class for all protocol adapters."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the protocol adapter."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the protocol adapter."""
        pass

class MCPAdapter(ProtocolAdapter):
    """Adapter for Model Context Protocol."""
    pass

class A2AAdapter(ProtocolAdapter):
    """Adapter for Agent-to-Agent Protocol."""
    pass

class ANPAdapter(ProtocolAdapter):
    """Adapter for Agent Notification Protocol."""
    pass

class ACPAdapter(ProtocolAdapter):
    """Adapter for Agent Communication Protocol."""
    pass

__all__ = ['ProtocolAdapter', 'MCPAdapter', 'A2AAdapter', 'ANPAdapter', 'ACPAdapter']
