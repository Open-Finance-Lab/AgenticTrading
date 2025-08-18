from __future__ import annotations

from typing import Protocol, Dict, Any, List, Optional


class MemoryPort(Protocol):
    """Port for A2A memory coordinator and peer knowledge exchange.
    
    This port defines the interface for distributed memory operations
    supporting cross-agent learning and knowledge sharing.
    """

    async def retrieve(self, query: Dict[str, Any], scope: str = "global") -> Dict[str, Any]:
        """Retrieve knowledge from distributed memory.
        
        Args:
            query: Query parameters for knowledge retrieval
            scope: Scope of retrieval (global, agent, pool)
            
        Returns:
            Retrieved knowledge data
            
        Raises:
            RuntimeError: If memory system is unavailable
            TimeoutError: If retrieval times out
        """
        ...

    async def append(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Append new event to distributed memory.
        
        Args:
            event: Event data to store
            
        Returns:
            Storage confirmation with event ID
            
        Raises:
            ValueError: If event format is invalid
            RuntimeError: If storage fails
        """
        ...

    async def lockless_suggest(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get lockless suggestions based on context.
        
        Args:
            context: Context for generating suggestions
            
        Returns:
            Suggested actions or parameters
        """
        ...

    async def store_strategy_performance(self, 
                                       agent_id: str,
                                       strategy_id: str, 
                                       performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Store strategy performance data.
        
        Args:
            agent_id: Agent identifier
            strategy_id: Strategy identifier  
            performance_metrics: Performance data
            
        Returns:
            Storage confirmation
        """
        ...

    async def get_agent_history(self, agent_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get historical data for an agent.
        
        Args:
            agent_id: Agent identifier
            limit: Maximum number of records
            
        Returns:
            List of historical events
        """
        ...

    async def search_similar_strategies(self, 
                                      strategy_signature: Dict[str, Any], 
                                      threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Search for similar strategies.
        
        Args:
            strategy_signature: Strategy characteristics to match
            threshold: Similarity threshold (0.0 to 1.0)
            
        Returns:
            List of similar strategies with metadata
        """
        ...

    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status.
        
        Returns:
            Connection status and health metrics
        """
        ...

