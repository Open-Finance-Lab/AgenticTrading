"""
Alpha Memory Client

Client for interacting with the memory system from alpha agents.
Provides methods for storing and retrieving alpha-related data.
"""

import httpx
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


class AlphaMemoryClient:
    """
    Client for alpha agent memory operations.
    
    This client provides a simplified interface for alpha agents to interact
    with the memory system, including storing performance data, strategy insights,
    and retrieving historical information.
    """
    
    def __init__(self, agent_id: str, memory_url: str = "http://127.0.0.1:8000"):
        self.agent_id = agent_id
        self.memory_url = memory_url
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")
        self.timeout = 10.0
    
    async def store_event(self, event_type: str, summary: str, keywords: List[str], 
                         details: str, log_level: str = "INFO", 
                         session_id: Optional[str] = None, 
                         correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Store an event in the memory system.
        
        Args:
            event_type: Type of event (e.g., 'ALPHA_SIGNAL', 'STRATEGY_UPDATE')
            summary: Brief summary of the event
            keywords: List of relevant keywords
            details: Detailed description
            log_level: Logging level (INFO, WARNING, ERROR)
            session_id: Optional session identifier
            correlation_id: Optional correlation identifier
            
        Returns:
            Result of the storage operation
        """
        try:
            payload = {
                "query": f"{event_type}: {summary}",
                "keywords": keywords,
                "summary": summary,
                "agent_id": self.agent_id,
                "event_type": event_type,
                "log_level": log_level,
                "details": details,
                "timestamp": datetime.now().isoformat()
            }
            
            if session_id:
                payload["session_id"] = session_id
            if correlation_id:
                payload["correlation_id"] = correlation_id
            
            # Try MCP endpoint first
            mcp_url = f"{self.memory_url}/mcp"
            mcp_payload = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "store_graph_memory",
                    "arguments": payload
                },
                "id": 1
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(mcp_url, json=mcp_payload)
                
                if response.status_code == 200:
                    result = response.json()
                    if "result" in result:
                        return {
                            "status": "success",
                            "message": "Event stored successfully",
                            "result": result["result"]
                        }
                
                # Fallback to direct memory endpoint
                memory_url = f"{self.memory_url}/memory/store"
                response = await client.post(memory_url, json=payload)
                
                if response.status_code == 200:
                    return {
                        "status": "success",
                        "message": "Event stored via fallback endpoint",
                        "result": response.json()
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"Failed to store event: HTTP {response.status_code}",
                        "error": response.text
                    }
                    
        except Exception as e:
            self.logger.error(f"Error storing event: {e}")
            return {
                "status": "error",
                "message": f"Exception during event storage: {str(e)}"
            }
    
    async def store_alpha_signal(self, symbol: str, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store alpha signal data.
        
        Args:
            symbol: Stock symbol
            signal_data: Signal information including strength, direction, etc.
            
        Returns:
            Storage result
        """
        keywords = [symbol, "alpha_signal", signal_data.get("direction", "unknown")]
        summary = f"Alpha signal for {symbol}: {signal_data.get('signal', 'N/A')}"
        details = json.dumps(signal_data, indent=2)
        
        return await self.store_event(
            event_type="ALPHA_SIGNAL",
            summary=summary,
            keywords=keywords,
            details=details,
            log_level="INFO"
        )
    
    async def store_performance_data(self, strategy_id: str, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store strategy performance data.
        
        Args:
            strategy_id: Strategy identifier
            performance_metrics: Performance metrics dict
            
        Returns:
            Storage result
        """
        keywords = [strategy_id, "performance", "strategy_metrics"]
        summary = f"Performance update for strategy {strategy_id}"
        details = json.dumps(performance_metrics, indent=2)
        
        return await self.store_event(
            event_type="PERFORMANCE_DATA",
            summary=summary,
            keywords=keywords,
            details=details,
            log_level="INFO"
        )
    
    async def retrieve_similar_strategies(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Retrieve strategies similar to the given query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            Search results
        """
        try:
            # Try MCP search endpoint
            mcp_url = f"{self.memory_url}/mcp"
            mcp_payload = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "semantic_search",
                    "arguments": {
                        "query": query,
                        "limit": limit,
                        "filter": {"agent_id": self.agent_id}
                    }
                },
                "id": 2
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(mcp_url, json=mcp_payload)
                
                if response.status_code == 200:
                    result = response.json()
                    if "result" in result:
                        return {
                            "status": "success",
                            "results": result["result"]
                        }
                
                # Fallback to basic search
                return {
                    "status": "limited",
                    "message": "Search functionality limited",
                    "results": []
                }
                
        except Exception as e:
            self.logger.error(f"Error retrieving strategies: {e}")
            return {
                "status": "error",
                "message": str(e),
                "results": []
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check memory system health.
        
        Returns:
            Health status
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.memory_url}/health")
                
                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "memory_system": response.json()
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "error": f"HTTP {response.status_code}"
                    }
                    
        except Exception as e:
            return {
                "status": "unreachable",
                "error": str(e)
            }


# Convenience function
async def create_alpha_memory_client(agent_id: str, memory_url: str = "http://127.0.0.1:8000") -> AlphaMemoryClient:
    """Create and test an alpha memory client."""
    client = AlphaMemoryClient(agent_id, memory_url)
    
    # Test connection
    health = await client.health_check()
    if health.get("status") != "healthy":
        logger.warning(f"Memory system health check failed: {health}")
    
    return client
