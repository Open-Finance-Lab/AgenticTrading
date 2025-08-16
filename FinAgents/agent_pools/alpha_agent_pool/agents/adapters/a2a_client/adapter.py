from __future__ import annotations

from typing import Any, Dict, List

from corepkg.ports.memory import MemoryPort
from corepkg.observability.logger import get_logger
from corepkg.observability.metrics import increment_counter


class A2AClientAdapter(MemoryPort):
    """Enhanced A2A client adapter with complete MemoryPort implementation."""

    def __init__(self, coordinator):
        self._coordinator = coordinator
        self._logger = get_logger("a2a_adapter")

    def _get_client(self):
        """Get A2A client with error handling."""
        if not self._coordinator:
            return None
        return getattr(self._coordinator, "a2a_client", None)

    async def retrieve(self, query: Dict[str, Any], scope: str = "global") -> Dict[str, Any]:
        """Retrieve knowledge from A2A memory."""
        client = self._get_client()
        if not client:
            increment_counter("a2a_client_unavailable", labels={"operation": "retrieve"})
            return {}
        
        try:
            fn = getattr(client, "retrieve", None)
            if callable(fn):
                result = await fn(query=query, scope=scope)
                increment_counter("a2a_retrieve_success")
                return result
            
            # Fallback methods
            for method_name in ["get", "query", "search"]:
                fn = getattr(client, method_name, None)
                if callable(fn):
                    result = await fn(query)
                    increment_counter("a2a_retrieve_fallback", labels={"method": method_name})
                    return result
            
            increment_counter("a2a_retrieve_no_method")
            return {}
            
        except Exception as e:
            self._logger.error("A2A retrieve failed", error=str(e), query=query)
            increment_counter("a2a_retrieve_error")
            return {}

    async def append(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Append event to A2A memory."""
        client = self._get_client()
        if not client:
            increment_counter("a2a_client_unavailable", labels={"operation": "append"})
            return {"status": "noop"}
        
        try:
            fn = getattr(client, "append_event", None) or getattr(client, "append", None)
            if callable(fn):
                result = await fn(event)
                increment_counter("a2a_append_success")
                return result if isinstance(result, dict) else {"status": "success"}
            
            # Fallback methods
            for method_name in ["store", "add", "insert"]:
                fn = getattr(client, method_name, None)
                if callable(fn):
                    await fn(event)
                    increment_counter("a2a_append_fallback", labels={"method": method_name})
                    return {"status": "success"}
            
            increment_counter("a2a_append_no_method")
            return {"status": "noop"}
            
        except Exception as e:
            self._logger.error("A2A append failed", error=str(e), event_type=event.get("type"))
            increment_counter("a2a_append_error")
            return {"status": "error", "message": str(e)}

    async def lockless_suggest(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get lockless suggestions from A2A memory."""
        client = self._get_client()
        if not client:
            increment_counter("a2a_client_unavailable", labels={"operation": "suggest"})
            return {}
        
        try:
            fn = getattr(client, "suggest", None) or getattr(client, "lockless_suggest", None)
            if callable(fn):
                result = await fn(context)
                increment_counter("a2a_suggest_success")
                return result
            
            increment_counter("a2a_suggest_no_method")
            return {}
            
        except Exception as e:
            self._logger.error("A2A suggest failed", error=str(e))
            increment_counter("a2a_suggest_error")
            return {}

    async def store_strategy_performance(self, 
                                       agent_id: str,
                                       strategy_id: str, 
                                       performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Store strategy performance data."""
        event = {
            "type": "strategy_performance",
            "agent_id": agent_id,
            "strategy_id": strategy_id,
            "metrics": performance_metrics,
            "timestamp": __import__("datetime").datetime.now().isoformat()
        }
        
        return await self.append(event)

    async def get_agent_history(self, agent_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get historical data for an agent."""
        query = {
            "type": "agent_history",
            "agent_id": agent_id,
            "limit": limit
        }
        
        result = await self.retrieve(query, scope="agent")
        return result.get("history", [])

    async def search_similar_strategies(self, 
                                      strategy_signature: Dict[str, Any], 
                                      threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Search for similar strategies."""
        query = {
            "type": "similarity_search",
            "signature": strategy_signature,
            "threshold": threshold
        }
        
        result = await self.retrieve(query, scope="global")
        return result.get("similar_strategies", [])

    def get_connection_status(self) -> Dict[str, Any]:
        """Get current A2A connection status."""
        client = self._get_client()
        
        if not client:
            return {
                "connected": False,
                "status": "no_coordinator",
                "client_available": False
            }
        
        # Try to get status from client
        try:
            if hasattr(client, "get_status"):
                return client.get_status()
            elif hasattr(client, "health_check"):
                health = client.health_check()
                return {"connected": True, "health": health}
            else:
                return {
                    "connected": True,
                    "status": "unknown",
                    "client_available": True,
                    "methods": [method for method in dir(client) if not method.startswith("_")]
                }
        except Exception as e:
            return {
                "connected": False,
                "status": "error",
                "error": str(e)
            }

