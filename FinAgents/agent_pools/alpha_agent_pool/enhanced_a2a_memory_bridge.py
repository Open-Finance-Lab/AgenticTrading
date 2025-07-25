"""
Enhanced A2A Memory Bridge for Alpha Agent Pool

This module provides a practical A2A memory bridge that connects the Alpha Agent Pool
with the memory services, focusing on functional integration rather than strict
protocol compliance.

Academic Framework:
- Memory-augmented agent coordination
- Cross-pool learning facilitation  
- Performance analytics storage and retrieval
- Strategy knowledge management

Author: FinAgent Research Team
Created: 2025-07-25
"""
import asyncio
import httpx
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import uuid

logger = logging.getLogger(__name__)

class EnhancedA2AMemoryBridge:
    """
    Enhanced A2A memory bridge for Alpha Agent Pool.
    
    This bridge provides practical memory coordination functionality between
    the Alpha Agent Pool and memory services, implementing key features
    for cross-agent learning and performance tracking.
    """
    
    def __init__(self, 
                 pool_id: str,
                 memory_server_url: str = "http://127.0.0.1:8002",
                 mcp_server_url: str = "http://127.0.0.1:8001",
                 legacy_server_url: str = "http://127.0.0.1:8000",
                 timeout: float = 30.0):
        """
        Initialize enhanced A2A memory bridge.
        
        Args:
            pool_id: Unique identifier for the alpha agent pool
            memory_server_url: A2A Memory Server URL (port 8002)
            mcp_server_url: MCP Memory Server URL (port 8001) 
            legacy_server_url: Legacy Memory Server URL (port 8000)
            timeout: Request timeout in seconds
        """
        self.pool_id = pool_id
        self.memory_server_url = memory_server_url.rstrip('/')
        self.mcp_server_url = mcp_server_url.rstrip('/')
        self.legacy_server_url = legacy_server_url.rstrip('/')
        self.timeout = timeout
        
        # HTTP client for memory operations
        self.http_client = httpx.AsyncClient(timeout=timeout)
        
        # Track connection status
        self.is_connected = False
        self.last_health_check = None
        self.preferred_server = None
        
        # Memory operation statistics
        self.operation_stats = {
            "store_operations": 0,
            "retrieve_operations": 0,
            "failed_operations": 0,
            "total_operations": 0
        }
        
        logger.info(f"Enhanced A2A Memory Bridge initialized for pool {pool_id}")
    
    async def initialize_connection(self) -> bool:
        """
        Initialize connection to memory services with fallback strategy.
        
        Returns:
            bool: True if connection established successfully
        """
        logger.info("Initializing memory service connections...")
        
        # Try A2A Memory Server first (port 8002)
        if await self._test_server_connection(self.memory_server_url, "A2A Memory Server"):
            self.preferred_server = self.memory_server_url
            self.is_connected = True
            logger.info("✅ Connected to A2A Memory Server (primary)")
            return True
        
        # Fallback to MCP Memory Server (port 8001)
        if await self._test_server_connection(self.mcp_server_url, "MCP Memory Server"):
            self.preferred_server = self.mcp_server_url
            self.is_connected = True
            logger.info("✅ Connected to MCP Memory Server (fallback)")
            return True
        
        # Fallback to Legacy Memory Server (port 8000)
        if await self._test_server_connection(self.legacy_server_url, "Legacy Memory Server"):
            self.preferred_server = self.legacy_server_url
            self.is_connected = True
            logger.info("✅ Connected to Legacy Memory Server (fallback)")
            return True
        
        logger.error("❌ Failed to connect to any memory server")
        self.is_connected = False
        return False
    
    async def _test_server_connection(self, server_url: str, server_name: str) -> bool:
        """Test connection to a specific memory server."""
        try:
            # Try basic connectivity test
            response = await self.http_client.get(f"{server_url}/", timeout=5.0)
            # Accept any response (including 404, 405) as proof server is running
            if response.status_code in [200, 404, 405]:
                logger.info(f"✅ {server_name} is accessible")
                return True
            else:
                logger.warning(f"⚠️  {server_name} returned status {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"❌ {server_name} not accessible: {e}")
            return False
    
    async def store_agent_performance(self, 
                                    agent_id: str,
                                    performance_data: Dict[str, Any]) -> bool:
        """
        Store agent performance data in memory.
        
        Args:
            agent_id: Identifier of the agent
            performance_data: Performance metrics and data
            
        Returns:
            bool: True if storage successful
        """
        if not self.is_connected:
            await self.initialize_connection()
        
        if not self.is_connected:
            logger.error("No memory server connection available for performance storage")
            self.operation_stats["failed_operations"] += 1
            return False
        
        try:
            # Prepare memory storage payload
            storage_payload = {
                "type": "agent_performance",
                "agent_id": agent_id,
                "pool_id": self.pool_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": performance_data,
                "correlation_id": str(uuid.uuid4())
            }
            
            # Try to store via HTTP POST to memory server
            response = await self.http_client.post(
                f"{self.preferred_server}/",
                json=storage_payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code in [200, 201, 202]:
                logger.info(f"✅ Stored performance data for agent {agent_id}")
                self.operation_stats["store_operations"] += 1
                self.operation_stats["total_operations"] += 1
                return True
            else:
                logger.warning(f"⚠️  Performance storage returned status {response.status_code}")
                self.operation_stats["failed_operations"] += 1
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to store agent performance: {e}")
            self.operation_stats["failed_operations"] += 1
            return False
    
    async def store_strategy_insights(self, 
                                    strategy_id: str,
                                    insights_data: Dict[str, Any]) -> bool:
        """
        Store strategy insights and research data in memory.
        
        Args:
            strategy_id: Identifier of the strategy
            insights_data: Strategy insights and analysis data
            
        Returns:
            bool: True if storage successful
        """
        if not self.is_connected:
            await self.initialize_connection()
        
        if not self.is_connected:
            logger.error("No memory server connection available for strategy storage")
            return False
        
        try:
            # Prepare strategy storage payload
            storage_payload = {
                "type": "strategy_insights",
                "strategy_id": strategy_id,
                "pool_id": self.pool_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": insights_data,
                "correlation_id": str(uuid.uuid4())
            }
            
            response = await self.http_client.post(
                f"{self.preferred_server}/",
                json=storage_payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code in [200, 201, 202]:
                logger.info(f"✅ Stored strategy insights for {strategy_id}")
                self.operation_stats["store_operations"] += 1
                self.operation_stats["total_operations"] += 1
                return True
            else:
                logger.warning(f"⚠️  Strategy storage returned status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to store strategy insights: {e}")
            return False
    
    async def retrieve_similar_strategies(self, 
                                        query: str,
                                        limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve similar strategies and patterns from memory.
        
        Args:
            query: Search query for similar strategies
            limit: Maximum number of results to return
            
        Returns:
            List of similar strategy data
        """
        if not self.is_connected:
            await self.initialize_connection()
        
        if not self.is_connected:
            logger.error("No memory server connection available for retrieval")
            return []
        
        try:
            # Prepare retrieval query payload
            query_payload = {
                "type": "strategy_search",
                "query": query,
                "pool_id": self.pool_id,
                "limit": limit,
                "timestamp": datetime.utcnow().isoformat(),
                "correlation_id": str(uuid.uuid4())
            }
            
            response = await self.http_client.post(
                f"{self.preferred_server}/",
                json=query_payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result_data = response.json()
                strategies = result_data.get('strategies', [])
                logger.info(f"✅ Retrieved {len(strategies)} similar strategies")
                self.operation_stats["retrieve_operations"] += 1
                self.operation_stats["total_operations"] += 1
                return strategies
            else:
                logger.warning(f"⚠️  Strategy retrieval returned status {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Failed to retrieve similar strategies: {e}")
            return []
    
    async def store_cross_agent_learning(self, 
                                       learning_data: Dict[str, Any]) -> bool:
        """
        Store cross-agent learning patterns and knowledge transfer data.
        
        Args:
            learning_data: Cross-agent learning and knowledge transfer data
            
        Returns:
            bool: True if storage successful
        """
        if not self.is_connected:
            await self.initialize_connection()
        
        if not self.is_connected:
            logger.error("No memory server connection available for learning storage")
            return False
        
        try:
            # Prepare learning storage payload
            storage_payload = {
                "type": "cross_agent_learning",
                "pool_id": self.pool_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": learning_data,
                "correlation_id": str(uuid.uuid4())
            }
            
            response = await self.http_client.post(
                f"{self.preferred_server}/",
                json=storage_payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code in [200, 201, 202]:
                logger.info("✅ Stored cross-agent learning data")
                self.operation_stats["store_operations"] += 1
                self.operation_stats["total_operations"] += 1
                return True
            else:
                logger.warning(f"⚠️  Learning storage returned status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to store cross-agent learning: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on memory bridge and services.
        
        Returns:
            Health status information
        """
        health_data = {
            "bridge_status": "healthy" if self.is_connected else "disconnected",
            "preferred_server": self.preferred_server,
            "last_health_check": datetime.utcnow().isoformat(),
            "operation_stats": self.operation_stats.copy()
        }
        
        # Test current connection
        if self.preferred_server:
            try:
                response = await self.http_client.get(f"{self.preferred_server}/", timeout=5.0)
                health_data["server_responsive"] = response.status_code in [200, 404, 405]
            except Exception as e:
                health_data["server_responsive"] = False
                health_data["server_error"] = str(e)
        
        self.last_health_check = datetime.utcnow()
        return health_data
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive memory bridge statistics.
        
        Returns:
            Memory bridge usage and performance statistics
        """
        stats = {
            "pool_id": self.pool_id,
            "connection_status": "connected" if self.is_connected else "disconnected",
            "preferred_server": self.preferred_server,
            "operation_stats": self.operation_stats.copy(),
            "uptime": "unknown",  # Could track initialization time
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None
        }
        
        # Calculate success rate
        total_ops = self.operation_stats["total_operations"]
        failed_ops = self.operation_stats["failed_operations"]
        if total_ops > 0:
            stats["success_rate"] = (total_ops - failed_ops) / total_ops
        else:
            stats["success_rate"] = 0.0
        
        return stats
    
    async def close(self):
        """Clean up resources and close connections."""
        if self.http_client:
            await self.http_client.aclose()
        logger.info("Enhanced A2A Memory Bridge closed")


# Global memory bridge instance for Alpha Agent Pool
_global_memory_bridge: Optional[EnhancedA2AMemoryBridge] = None

async def get_memory_bridge(pool_id: str = "alpha_pool_8081") -> EnhancedA2AMemoryBridge:
    """
    Get or create the global memory bridge instance.
    
    Args:
        pool_id: Alpha agent pool identifier
        
    Returns:
        Enhanced A2A memory bridge instance
    """
    global _global_memory_bridge
    
    if _global_memory_bridge is None:
        _global_memory_bridge = EnhancedA2AMemoryBridge(pool_id=pool_id)
        await _global_memory_bridge.initialize_connection()
    
    return _global_memory_bridge

async def shutdown_memory_bridge():
    """Shutdown the global memory bridge."""
    global _global_memory_bridge
    
    if _global_memory_bridge:
        await _global_memory_bridge.close()
        _global_memory_bridge = None
        logger.info("Global memory bridge shutdown completed")
