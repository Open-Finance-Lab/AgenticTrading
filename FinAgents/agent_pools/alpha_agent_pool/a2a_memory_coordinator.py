"""
Alpha Agent Pool A2A Memory Coordinator

This module provides pool-level coordination between alpha agents and the memory system
using the official A2A protocol. It serves as a central hub for memory operations
within the alpha agent pool, managing communication between individual alpha agents
and the external memory agent.

Key Features:
- Pool-level A2A protocol coordination
- Centralized memory operations management
- Agent performance aggregation and analysis
- Strategy pattern recognition and sharing
- Cross-agent learning facilitation

Author: FinAgent Team
License: Open Source
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from pathlib import Path

try:
    from .agents.theory_driven.a2a_client import AlphaAgentA2AClient, create_alpha_pool_a2a_client
    A2A_CLIENT_AVAILABLE = True
except ImportError:
    A2A_CLIENT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("A2A client not available")

logger = logging.getLogger(__name__)


class AlphaPoolA2AMemoryCoordinator:
    """
    Alpha Agent Pool A2A Memory Coordinator.
    
    This class serves as the central coordination hub for memory operations
    within the alpha agent pool, using the official A2A protocol to communicate
    with external memory agents.
    """
    
    def __init__(self, 
                 pool_id: str = "alpha_agent_pool",
                 memory_agent_url: str = "http://127.0.0.1:8002",
                 coordination_interval: float = 60.0):
        """
        Initialize the A2A memory coordinator.
        
        Args:
            pool_id: Unique identifier for the alpha agent pool
            memory_agent_url: URL of the external memory agent
            coordination_interval: Interval for coordination tasks (seconds)
        """
        self.pool_id = pool_id
        self.memory_agent_url = memory_agent_url
        self.coordination_interval = coordination_interval
        
        # Initialize A2A client if available
        self.a2a_client = None
        if A2A_CLIENT_AVAILABLE:
            self.a2a_client = create_alpha_pool_a2a_client(
                agent_pool_id=pool_id,
                memory_url=memory_agent_url
            )
        
        # Track registered agents and their performance
        self.registered_agents: Dict[str, Dict[str, Any]] = {}
        self.agent_performance_cache: Dict[str, List[Dict[str, Any]]] = {}
        
        # Coordination task handle
        self.coordination_task = None
        self.is_running = False
        
        logger.info(f"Alpha Pool A2A Memory Coordinator initialized for pool {pool_id}")
    
    async def start_coordination(self):
        """Start the memory coordination service."""
        if self.is_running:
            logger.warning("Coordination already running")
            return
        
        self.is_running = True
        
        # Connect to memory agent via A2A protocol
        if self.a2a_client:
            try:
                # Use direct connection instead of async context manager
                connection_success = await self.a2a_client.connect()
                if connection_success:
                    logger.info("Successfully connected to memory agent via A2A protocol")
                else:
                    logger.warning("A2A connection failed, using fallback methods")
            except Exception as e:
                logger.error(f"Failed to connect to memory agent: {e}")
        
        # Start coordination background task
        self.coordination_task = asyncio.create_task(self._coordination_loop())
        logger.info("Alpha Pool A2A Memory Coordinator started")
    
    async def stop_coordination(self):
        """Stop the memory coordination service."""
        self.is_running = False
        
        if self.coordination_task:
            self.coordination_task.cancel()
            try:
                await self.coordination_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Alpha Pool A2A Memory Coordinator stopped")
    
    async def register_agent(self, 
                           agent_id: str, 
                           agent_type: str, 
                           agent_config: Dict[str, Any]):
        """
        Register an alpha agent with the coordination system.
        
        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent (momentum, mean_reversion, etc.)
            agent_config: Agent configuration parameters
        """
        registration_data = {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "config": agent_config,
            "registration_time": datetime.utcnow().isoformat(),
            "status": "active"
        }
        
        self.registered_agents[agent_id] = registration_data
        
        # Store registration in memory via A2A protocol
        if self.a2a_client:
            try:
                await self.a2a_client.store_learning_feedback(
                    agent_id=agent_id,
                    feedback_type="AGENT_REGISTRATION",
                    feedback_data=registration_data
                )
                logger.info(f"[A2A] Registered agent {agent_id} via A2A protocol")
            except Exception as e:
                logger.warning(f"[A2A] Failed to store agent registration: {e}")
        
        logger.info(f"Alpha agent {agent_id} ({agent_type}) registered with pool coordinator")
    
    async def aggregate_pool_performance(self) -> Dict[str, Any]:
        """
        Aggregate performance metrics across all agents in the pool.
        
        Returns:
            Dict containing pool-wide performance analytics
        """
        pool_metrics = {
            "pool_id": self.pool_id,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "total_agents": len(self.registered_agents),
            "active_agents": len([a for a in self.registered_agents.values() if a["status"] == "active"]),
            "agent_performance": {}
        }
        
        # Collect performance data from each agent
        for agent_id, agent_data in self.registered_agents.items():
            agent_performance = self.agent_performance_cache.get(agent_id, [])
            
            if agent_performance:
                latest_performance = agent_performance[-1]
                pool_metrics["agent_performance"][agent_id] = {
                    "agent_type": agent_data["agent_type"],
                    "latest_ic": latest_performance.get("IC"),
                    "latest_ir": latest_performance.get("IR"),
                    "total_signals": latest_performance.get("num_trades", 0),
                    "win_rate": latest_performance.get("win_rate"),
                    "avg_return": latest_performance.get("avg_return")
                }
        
        # Store aggregated metrics via A2A protocol
        if self.a2a_client:
            try:
                await self.a2a_client.store_strategy_performance(
                    agent_id=self.pool_id,
                    strategy_id=f"pool_aggregated_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    performance_metrics=pool_metrics
                )
                logger.info("[A2A] Stored pool aggregated performance via A2A protocol")
            except Exception as e:
                logger.warning(f"[A2A] Failed to store pool performance: {e}")
        
        return pool_metrics
    
    async def retrieve_cross_agent_insights(self, 
                                          query: str,
                                          limit: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieve insights from memory that could benefit multiple agents.
        
        Args:
            query: Natural language query for insights
            limit: Maximum number of insights to retrieve
            
        Returns:
            List of relevant insights and patterns
        """
        if not self.a2a_client:
            logger.warning("A2A client not available for insight retrieval")
            return []
        
        try:
            insights = await self.a2a_client.retrieve_strategy_insights(
                search_query=query,
                limit=limit
            )
            
            logger.info(f"[A2A] Retrieved {len(insights)} cross-agent insights")
            return insights
            
        except Exception as e:
            logger.error(f"[A2A] Failed to retrieve cross-agent insights: {e}")
            return []
    
    async def facilitate_agent_learning(self, 
                                      source_agent_id: str,
                                      target_agent_ids: List[str],
                                      learning_pattern: Dict[str, Any]):
        """
        Facilitate learning transfer between agents using A2A protocol.
        
        Args:
            source_agent_id: Agent that discovered the pattern
            target_agent_ids: Agents that should learn from the pattern
            learning_pattern: Pattern or strategy to share
        """
        transfer_data = {
            "source_agent": source_agent_id,
            "target_agents": target_agent_ids,
            "learning_pattern": learning_pattern,
            "transfer_timestamp": datetime.utcnow().isoformat(),
            "pool_id": self.pool_id
        }
        
        if self.a2a_client:
            try:
                await self.a2a_client.store_learning_feedback(
                    agent_id=self.pool_id,
                    feedback_type="CROSS_AGENT_LEARNING_TRANSFER",
                    feedback_data=transfer_data
                )
                logger.info(f"[A2A] Facilitated learning transfer from {source_agent_id} to {len(target_agent_ids)} agents")
            except Exception as e:
                logger.warning(f"[A2A] Failed to store learning transfer: {e}")
    
    async def _coordination_loop(self):
        """Background coordination loop for memory management."""
        while self.is_running:
            try:
                # Perform periodic coordination tasks
                await self._periodic_health_check()
                await self._update_agent_statistics()
                await self._detect_learning_opportunities()
                
                # Wait for next coordination cycle
                await asyncio.sleep(self.coordination_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")
                await asyncio.sleep(self.coordination_interval)
    
    async def _periodic_health_check(self):
        """Perform health checks on A2A connections."""
        if self.a2a_client:
            try:
                healthy = await self.a2a_client.healthcheck()
                if not healthy:
                    logger.warning("A2A connection health check failed")
            except Exception as e:
                logger.error(f"A2A health check error: {e}")
    
    async def _update_agent_statistics(self):
        """Update statistics for all registered agents."""
        # This could fetch latest performance data from each agent
        # For now, we'll just log the coordination activity
        logger.debug(f"Coordinating memory for {len(self.registered_agents)} agents")
    
    async def _detect_learning_opportunities(self):
        """Detect opportunities for cross-agent learning."""
        if len(self.registered_agents) < 2:
            return
        
        # Analyze agent performance patterns to identify learning opportunities
        # This is a placeholder for more sophisticated pattern detection
        logger.debug("Analyzing cross-agent learning opportunities")


# Global coordinator instance for the alpha agent pool
_pool_coordinator: Optional[AlphaPoolA2AMemoryCoordinator] = None


def get_pool_coordinator() -> Optional[AlphaPoolA2AMemoryCoordinator]:
    """Get the global pool coordinator instance."""
    return _pool_coordinator


async def initialize_pool_coordinator(pool_id: str = "alpha_agent_pool",
                                    memory_url: str = "http://127.0.0.1:8002") -> AlphaPoolA2AMemoryCoordinator:
    """
    Initialize the global pool coordinator.
    
    Args:
        pool_id: Alpha agent pool identifier
        memory_url: Memory agent URL
        
    Returns:
        Initialized coordinator instance
    """
    global _pool_coordinator
    
    if _pool_coordinator is None:
        _pool_coordinator = AlphaPoolA2AMemoryCoordinator(
            pool_id=pool_id,
            memory_agent_url=memory_url
        )
        await _pool_coordinator.start_coordination()
        logger.info("Global alpha pool coordinator initialized")
    
    return _pool_coordinator


async def shutdown_pool_coordinator():
    """Shutdown the global pool coordinator."""
    global _pool_coordinator
    
    if _pool_coordinator:
        await _pool_coordinator.stop_coordination()
        _pool_coordinator = None
        logger.info("Global alpha pool coordinator shutdown")
