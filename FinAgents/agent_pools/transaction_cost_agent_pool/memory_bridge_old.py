"""
Memory Bridge for Transaction Cost Agent Pool

This module provides integration with both the legacy FinAgent memory system
and the new External Memory Agent, enabling transaction cost agents to store,
retrieve, and manage historical cost data, model parameters, and optimization results.

The memory bridge supports:
- Cost model persistence and versioning
- Historical execution data storage
- Optimization result caching
- Agent state management
- Cross-pool data sharing
- Unified logging through External Memory Agent

Author: FinAgent Development Team
Created: 2025-06-25
Updated: 2024 - Added External Memory Agent integration
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import asyncio
import time

# Set up logging first
logger = logging.getLogger(__name__)

# Legacy memory system import
try:
    from ...memory.memory_agent import MemoryAgent as LegacyMemoryAgent
    LEGACY_MEMORY_AVAILABLE = True
except ImportError:
    LEGACY_MEMORY_AVAILABLE = False
    logger.warning("Legacy memory agent not available")

# New External Memory Agent import
try:
    from ...memory.external_memory_interface import (
        create_memory_agent,
        EventType,
        LogLevel,
        QueryFilter,
        create_transaction_event,
        create_optimization_event,
        create_error_event
    )
    EXTERNAL_MEMORY_AVAILABLE = True
except ImportError:
    EXTERNAL_MEMORY_AVAILABLE = False
    logger.warning("External Memory Agent not available")

from .schema.cost_models import (
    TransactionCostModel,
    MarketImpactModel,
    ExecutionResult
)


logger = logging.getLogger(__name__)


@dataclass
class MemoryQuery:
    """Query specification for memory operations."""
    
    agent_type: str
    symbol: Optional[str] = None
    venue: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    tags: Optional[List[str]] = None
    limit: int = 100


@dataclass
class MemoryQuery:
    """Query specification for memory operations."""
    
    agent_type: str
    symbol: Optional[str] = None
    venue: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    tags: Optional[List[str]] = None
    limit: int = 100


class TransactionCostMemoryBridge:
    """
    Enhanced Memory bridge for transaction cost agent pool.
    
    Provides seamless integration with both legacy FinAgent memory system
    and the new External Memory Agent, enabling agents to persist and retrieve
    cost-related data with unified logging capabilities.
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 use_external_memory: bool = True,
                 storage_path: Optional[str] = None):
        """
        Initialize memory bridge with dual memory system support.
        
        Args:
            config: Configuration dictionary for legacy system
            use_external_memory: Whether to use the new External Memory Agent
            storage_path: Optional custom storage path for External Memory Agent
        """
        self.config = config or {}
        self.use_external_memory = use_external_memory and EXTERNAL_MEMORY_AVAILABLE
        self.namespace = "transaction_cost"
        self.pool_name = "transaction_cost_agent_pool"
        self.session_id = f"tc_session_{int(time.time())}"
        
        # Initialize External Memory Agent if available and enabled
        if self.use_external_memory:
            try:
                storage_path = storage_path or "transaction_cost_memory_storage"
                self.external_memory = create_memory_agent(storage_path)
                logger.info(f"External Memory Agent initialized for transaction cost pool")
            except Exception as e:
                logger.error(f"Failed to initialize External Memory Agent: {e}")
                self.use_external_memory = False
                self.external_memory = None
        else:
            self.external_memory = None
        
        # Initialize legacy memory manager if available
        if LEGACY_MEMORY_AVAILABLE and not self.use_external_memory:
            try:
                # Legacy initialization would go here
                self.legacy_memory = None  # Placeholder
                logger.info("Legacy memory system initialized")
            except Exception as e:
                logger.error(f"Failed to initialize legacy memory system: {e}")
                self.legacy_memory = None
        else:
            self.legacy_memory = None
    
    def _log_event(self, 
                   event_type: 'EventType', 
                   log_level: 'LogLevel',
                   agent_id: str,
                   title: str,
                   content: str,
                   **kwargs) -> Optional[str]:
        """
        Helper method for consistent event logging to External Memory Agent.
        
        Args:
            event_type: Type of event
            log_level: Log level
            agent_id: Source agent ID
            title: Event title
            content: Event content
            **kwargs: Additional arguments for logging
            
        Returns:
            Event ID if successful, None otherwise
        """
        if not self.use_external_memory or not self.external_memory:
            return None
        
        try:
            return self.external_memory.log_event(
                event_type=event_type,
                log_level=log_level,
                source_agent_pool=self.pool_name,
                source_agent_id=agent_id,
                title=title,
                content=content,
                session_id=self.session_id,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to log event to External Memory Agent: {e}")
            return None
    
    async def store_cost_model(
        self,
        model: TransactionCostModel,
        agent_id: str,
        version: str = "latest"
    ) -> str:
        """
        Store transaction cost model in memory with enhanced logging.
        
        Args:
            model: Cost model to store
            agent_id: ID of the agent storing the model
            version: Model version identifier
            
        Returns:
            Storage key for the stored model
        """
        try:
            storage_key = f"{self.namespace}:models:{agent_id}:{version}"
            
            model_data = {
                "model": model.dict() if hasattr(model, 'dict') else str(model),
                "agent_id": agent_id,
                "version": version,
                "timestamp": datetime.utcnow().isoformat(),
                "type": "cost_model"
            }
            
            # Log to External Memory Agent
            self._log_event(
                event_type=EventType.SYSTEM,
                log_level=LogLevel.INFO,
                agent_id=agent_id,
                title=f"Cost Model Stored - Version {version}",
                content=f"Stored transaction cost model for agent {agent_id}, version {version}",
                tags={"cost_model", "storage", agent_id, version},
                metadata={
                    "storage_key": storage_key,
                    "model_type": type(model).__name__,
                    "version": version,
                    "agent_id": agent_id
                }
            )
            
            # Store in legacy system if available
            if self.legacy_memory:
                # Legacy storage implementation would go here
                pass
            
            logger.info(f"Stored cost model for agent {agent_id}, version {version}")
            return storage_key
            
        except Exception as e:
            # Log error to External Memory Agent
            self._log_event(
                event_type=EventType.ERROR,
                log_level=LogLevel.ERROR,
                agent_id=agent_id,
                title="Cost Model Storage Failed",
                content=f"Failed to store cost model: {str(e)}",
                tags={"cost_model", "storage", "error"},
                metadata={
                    "error_type": type(e).__name__,
                    "agent_id": agent_id,
                    "version": version
                }
            )
            logger.error(f"Failed to store cost model: {e}")
            raise
    
    async def retrieve_cost_model(
        self,
        agent_id: str,
        version: str = "latest"
    ) -> Optional[TransactionCostModel]:
        """
        Retrieve cost model from memory with logging.
        
        Args:
            agent_id: ID of the agent
            version: Model version to retrieve
            
        Returns:
            Retrieved cost model or None
        """
        try:
            storage_key = f"{self.namespace}:models:{agent_id}:{version}"
            
            # Log retrieval attempt
            self._log_event(
                event_type=EventType.SYSTEM,
                log_level=LogLevel.INFO,
                agent_id=agent_id,
                title=f"Cost Model Retrieval - Version {version}",
                content=f"Retrieving transaction cost model for agent {agent_id}, version {version}",
                tags={"cost_model", "retrieval", agent_id, version},
                metadata={
                    "storage_key": storage_key,
                    "version": version,
                    "agent_id": agent_id
                }
            )
            
            # Implement retrieval logic based on available systems
            model_data = None
            
            if self.legacy_memory:
                # Legacy retrieval would go here
                pass
            
            if model_data:
                logger.info(f"Retrieved cost model for agent {agent_id}, version {version}")
                return TransactionCostModel(**model_data['model'])
            else:
                logger.warning(f"No cost model found for agent {agent_id}, version {version}")
                return None
                
        except Exception as e:
            # Log error
            self._log_event(
                event_type=EventType.ERROR,
                log_level=LogLevel.ERROR,
                agent_id=agent_id,
                title="Cost Model Retrieval Failed",
                content=f"Failed to retrieve cost model: {str(e)}",
                tags={"cost_model", "retrieval", "error"},
                metadata={
                    "error_type": type(e).__name__,
                    "agent_id": agent_id,
                    "version": version
                }
            )
            logger.error(f"Failed to retrieve cost model: {e}")
            raise
    
    async def log_execution_result(
        self,
        result: ExecutionResult,
        agent_id: str,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Log execution result with comprehensive event tracking.
        
        Args:
            result: Execution result to log
            agent_id: ID of the executing agent
            correlation_id: Optional correlation ID for related events
            
        Returns:
            Event ID from External Memory Agent
        """
        try:
            # Create transaction event using helper function
            if hasattr(result, 'symbol') and hasattr(result, 'quantity'):
                transaction_event = create_transaction_event(
                    agent_pool=self.pool_name,
                    agent_id=agent_id,
                    transaction_type=getattr(result, 'side', 'unknown'),
                    symbol=result.symbol,
                    quantity=result.quantity,
                    price=getattr(result, 'executed_price', 0.0),
                    cost=getattr(result, 'total_cost', 0.0),
                    session_id=self.session_id
                )
                
                # Add execution-specific metadata
                transaction_event['metadata'].update({
                    'execution_id': getattr(result, 'execution_id', None),
                    'venue': getattr(result, 'venue', None),
                    'market_impact': getattr(result, 'market_impact', None),
                    'slippage': getattr(result, 'slippage', None),
                    'execution_time': getattr(result, 'execution_time', None)
                })
                
                if correlation_id:
                    transaction_event['correlation_id'] = correlation_id
                
                # Log using batch method
                if self.external_memory:
                    event_ids = self.external_memory.log_events_batch([transaction_event])
                    if event_ids:
                        logger.info(f"Logged execution result for {result.symbol}")
                        return event_ids[0]
            
            # Fallback to general event logging
            return self._log_event(
                event_type=EventType.TRANSACTION,
                log_level=LogLevel.INFO,
                agent_id=agent_id,
                title="Execution Result",
                content=f"Transaction execution completed",
                tags={"execution", "result"},
                metadata={"result": str(result)},
                correlation_id=correlation_id
            ) or ""
            
        except Exception as e:
            # Log error
            self._log_event(
                event_type=EventType.ERROR,
                log_level=LogLevel.ERROR,
                agent_id=agent_id,
                title="Execution Result Logging Failed",
                content=f"Failed to log execution result: {str(e)}",
                tags={"execution", "logging", "error"},
                metadata={"error_type": type(e).__name__}
            )
            logger.error(f"Failed to log execution result: {e}")
            raise
    
    async def log_optimization_event(
        self,
        agent_id: str,
        optimization_type: str,
        result: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Log optimization event with structured data.
        
        Args:
            agent_id: ID of the optimizing agent
            optimization_type: Type of optimization performed
            result: Optimization results
            correlation_id: Optional correlation ID
            
        Returns:
            Event ID from External Memory Agent
        """
        try:
            optimization_event = create_optimization_event(
                agent_pool=self.pool_name,
                agent_id=agent_id,
                optimization_type=optimization_type,
                result=result,
                session_id=self.session_id
            )
            
            if correlation_id:
                optimization_event['correlation_id'] = correlation_id
            
            if self.external_memory:
                event_ids = self.external_memory.log_events_batch([optimization_event])
                if event_ids:
                    logger.info(f"Logged optimization event: {optimization_type}")
                    return event_ids[0]
            
            return ""
            
        except Exception as e:
            logger.error(f"Failed to log optimization event: {e}")
            raise
    
    def query_historical_executions(
        self,
        query: MemoryQuery
    ) -> List[Dict[str, Any]]:
        """
        Query historical execution data with External Memory Agent support.
        
        Args:
            query: Query specification
            
        Returns:
            List of historical execution records
        """
        try:
            if self.use_external_memory and self.external_memory:
                # Build query filter for External Memory Agent
                query_filter = QueryFilter(
                    start_time=query.start_date,
                    end_time=query.end_date,
                    event_types=[EventType.TRANSACTION],
                    source_agent_pools=[self.pool_name],
                    tags={query.symbol} if query.symbol else None,
                    limit=query.limit
                )
                
                if query.agent_type:
                    query_filter.source_agent_ids = [query.agent_type]
                
                result = self.external_memory.query_events(query_filter)
                
                # Convert events to execution records
                executions = []
                for event in result.events:
                    execution_record = {
                        'event_id': event.event_id,
                        'timestamp': event.timestamp,
                        'symbol': event.metadata.get('symbol'),
                        'quantity': event.metadata.get('quantity'),
                        'price': event.metadata.get('price'),
                        'cost': event.metadata.get('cost'),
                        'agent_id': event.source_agent_id,
                        'execution_data': event.metadata
                    }
                    executions.append(execution_record)
                
                logger.info(f"Retrieved {len(executions)} historical executions")
                return executions
            
            # Fallback to legacy system or empty result
            logger.warning("No memory system available for historical query")
            return []
            
        except Exception as e:
            logger.error(f"Failed to query historical executions: {e}")
            return []
    
    def get_recent_optimization_results(
        self,
        agent_id: Optional[str] = None,
        hours: int = 24,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get recent optimization results from External Memory Agent.
        
        Args:
            agent_id: Optional specific agent ID filter
            hours: Number of hours to look back
            limit: Maximum number of results
            
        Returns:
            List of recent optimization results
        """
        try:
            if self.use_external_memory and self.external_memory:
                query_filter = QueryFilter(
                    start_time=datetime.now() - timedelta(hours=hours),
                    event_types=[EventType.OPTIMIZATION],
                    source_agent_pools=[self.pool_name],
                    source_agent_ids=[agent_id] if agent_id else None,
                    limit=limit
                )
                
                result = self.external_memory.query_events(query_filter)
                
                optimization_results = []
                for event in result.events:
                    opt_result = {
                        'event_id': event.event_id,
                        'timestamp': event.timestamp,
                        'optimization_type': event.metadata.get('optimization_type'),
                        'result': event.metadata.get('result', {}),
                        'agent_id': event.source_agent_id
                    }
                    optimization_results.append(opt_result)
                
                logger.info(f"Retrieved {len(optimization_results)} recent optimization results")
                return optimization_results
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get recent optimization results: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics from both memory systems.
        
        Returns:
            Dictionary containing memory system statistics
        """
        stats = {
            'bridge_config': {
                'use_external_memory': self.use_external_memory,
                'external_memory_available': EXTERNAL_MEMORY_AVAILABLE,
                'legacy_memory_available': LEGACY_MEMORY_AVAILABLE,
                'pool_name': self.pool_name,
                'session_id': self.session_id
            }
        }
        
        # Get External Memory Agent statistics
        if self.use_external_memory and self.external_memory:
            try:
                external_stats = self.external_memory.get_statistics()
                stats['external_memory'] = external_stats
            except Exception as e:
                logger.error(f"Failed to get external memory statistics: {e}")
                stats['external_memory'] = {'error': str(e)}
        
        # Get legacy memory statistics if available
        if self.legacy_memory:
            try:
                # Legacy stats would go here
                stats['legacy_memory'] = {'status': 'available'}
            except Exception as e:
                logger.error(f"Failed to get legacy memory statistics: {e}")
                stats['legacy_memory'] = {'error': str(e)}
        
        return stats
    
    def cleanup(self) -> None:
        """Clean up memory bridge resources."""
        try:
            if self.external_memory:
                self.external_memory.cleanup()
            
            if self.legacy_memory:
                # Legacy cleanup would go here
                pass
            
            logger.info("Transaction cost memory bridge cleanup completed")
        except Exception as e:
            logger.error(f"Error during memory bridge cleanup: {e}")


# Legacy compatibility functions
async def create_transaction_cost_memory_bridge(
    config: Optional[Dict[str, Any]] = None,
    use_external_memory: bool = True
) -> TransactionCostMemoryBridge:
    """
    Create and initialize a transaction cost memory bridge.
    
    Args:
        config: Configuration for memory systems
        use_external_memory: Whether to use External Memory Agent
        
    Returns:
        Initialized memory bridge
    """
    return TransactionCostMemoryBridge(
        config=config,
        use_external_memory=use_external_memory
    )
            Retrieved cost model or None if not found
        """
        try:
            storage_key = f"{self.namespace}:models:{agent_id}:{version}"
            
            stored_data = await self.memory_manager.retrieve(storage_key)
            if not stored_data:
                return None
                
            model_data = stored_data.get("model")
            if model_data:
                return TransactionCostModel(**model_data)
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve cost model: {e}")
            return None
    
    async def store_execution_result(
        self,
        result: ExecutionResult,
        agent_id: str
    ) -> str:
        """
        Store execution result in memory.
        
        Args:
            result: Execution result to store
            agent_id: ID of the agent storing the result
            
        Returns:
            Storage key for the stored result
        """
        try:
            timestamp = datetime.utcnow()
            storage_key = f"{self.namespace}:executions:{agent_id}:{timestamp.isoformat()}"
            
            result_data = {
                "result": result.dict(),
                "agent_id": agent_id,
                "timestamp": timestamp.isoformat(),
                "type": "execution_result"
            }
            
            await self.memory_manager.store(
                key=storage_key,
                data=result_data,
                tags=["execution_result", agent_id, result.symbol, result.venue]
            )
            
            logger.info(f"Stored execution result for {result.symbol} at {result.venue}")
            return storage_key
            
        except Exception as e:
            logger.error(f"Failed to store execution result: {e}")
            raise
    
    async def query_execution_history(
        self,
        query: MemoryQuery
    ) -> List[ExecutionResult]:
        """
        Query historical execution results.
        
        Args:
            query: Query specification
            
        Returns:
            List of matching execution results
        """
        try:
            search_tags = ["execution_result"]
            if query.agent_type:
                search_tags.append(query.agent_type)
            if query.symbol:
                search_tags.append(query.symbol)
            if query.venue:
                search_tags.append(query.venue)
            
            stored_results = await self.memory_manager.search(
                tags=search_tags,
                limit=query.limit
            )
            
            execution_results = []
            for stored_data in stored_results:
                if stored_data.get("type") == "execution_result":
                    result_data = stored_data.get("result")
                    if result_data:
                        # Filter by date range if specified
                        if query.start_date or query.end_date:
                            result_time = datetime.fromisoformat(
                                result_data.get("execution_time", "")
                            )
                            if query.start_date and result_time < query.start_date:
                                continue
                            if query.end_date and result_time > query.end_date:
                                continue
                        
                        execution_results.append(ExecutionResult(**result_data))
            
            return execution_results
            
        except Exception as e:
            logger.error(f"Failed to query execution history: {e}")
            return []
    
    async def store_optimization_result(
        self,
        optimization_data: Dict[str, Any],
        agent_id: str,
        optimization_type: str
    ) -> str:
        """
        Store optimization result in memory.
        
        Args:
            optimization_data: Optimization result data
            agent_id: ID of the agent storing the result
            optimization_type: Type of optimization (cost, routing, timing)
            
        Returns:
            Storage key for the stored result
        """
        try:
            timestamp = datetime.utcnow()
            storage_key = f"{self.namespace}:optimization:{optimization_type}:{agent_id}:{timestamp.isoformat()}"
            
            result_data = {
                "optimization_data": optimization_data,
                "agent_id": agent_id,
                "optimization_type": optimization_type,
                "timestamp": timestamp.isoformat(),
                "type": "optimization_result"
            }
            
            await self.memory_manager.store(
                key=storage_key,
                data=result_data,
                tags=["optimization_result", agent_id, optimization_type]
            )
            
            logger.info(f"Stored {optimization_type} optimization result for agent {agent_id}")
            return storage_key
            
        except Exception as e:
            logger.error(f"Failed to store optimization result: {e}")
            raise
    
    async def get_agent_state(
        self,
        agent_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve agent state from memory.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Agent state data or None if not found
        """
        try:
            storage_key = f"{self.namespace}:agent_state:{agent_id}"
            return await self.memory_manager.retrieve(storage_key)
            
        except Exception as e:
            logger.error(f"Failed to retrieve agent state: {e}")
            return None
    
    async def save_agent_state(
        self,
        agent_id: str,
        state_data: Dict[str, Any]
    ) -> None:
        """
        Save agent state to memory.
        
        Args:
            agent_id: ID of the agent
            state_data: State data to save
        """
        try:
            storage_key = f"{self.namespace}:agent_state:{agent_id}"
            
            state_record = {
                "state_data": state_data,
                "agent_id": agent_id,
                "timestamp": datetime.utcnow().isoformat(),
                "type": "agent_state"
            }
            
            await self.memory_manager.store(
                key=storage_key,
                data=state_record,
                tags=["agent_state", agent_id]
            )
            
            logger.info(f"Saved state for agent {agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to save agent state: {e}")
            raise
    
    async def cleanup_old_data(
        self,
        days_to_keep: int = 30
    ) -> int:
        """
        Clean up old data from memory.
        
        Args:
            days_to_keep: Number of days of data to retain
            
        Returns:
            Number of records cleaned up
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            # Search for old records
            old_records = await self.memory_manager.search(
                tags=[self.namespace],
                limit=1000
            )
            
            cleanup_count = 0
            for record in old_records:
                record_time = datetime.fromisoformat(
                    record.get("timestamp", datetime.utcnow().isoformat())
                )
                
                if record_time < cutoff_date:
                    # Delete old record
                    storage_key = record.get("key")
                    if storage_key:
                        await self.memory_manager.delete(storage_key)
                        cleanup_count += 1
            
            logger.info(f"Cleaned up {cleanup_count} old records")
            return cleanup_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return 0
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory usage statistics for transaction cost data.
        
        Returns:
            Dictionary containing usage statistics
        """
        try:
            # Count different types of stored data
            all_records = await self.memory_manager.search(
                tags=[self.namespace],
                limit=10000
            )
            
            stats = {
                "total_records": len(all_records),
                "cost_models": 0,
                "execution_results": 0,
                "optimization_results": 0,
                "agent_states": 0,
                "oldest_record": None,
                "newest_record": None
            }
            
            timestamps = []
            for record in all_records:
                record_type = record.get("type", "unknown")
                if record_type == "cost_model":
                    stats["cost_models"] += 1
                elif record_type == "execution_result":
                    stats["execution_results"] += 1
                elif record_type == "optimization_result":
                    stats["optimization_results"] += 1
                elif record_type == "agent_state":
                    stats["agent_states"] += 1
                
                timestamp = record.get("timestamp")
                if timestamp:
                    timestamps.append(datetime.fromisoformat(timestamp))
            
            if timestamps:
                stats["oldest_record"] = min(timestamps).isoformat()
                stats["newest_record"] = max(timestamps).isoformat()
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get memory statistics: {e}")
            return {"error": str(e)}


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from unittest.mock import Mock
    
    async def test_memory_bridge():
        """Test the memory bridge functionality."""
        # Mock memory manager for testing
        mock_memory = Mock()
        mock_memory.store = Mock(return_value=asyncio.Future())
        mock_memory.store.return_value.set_result("stored")
        
        bridge = TransactionCostMemoryBridge(mock_memory)
        
        # Test cost model storage
        cost_model = TransactionCostModel(
            symbol="AAPL",
            venue="NYSE",
            fixed_cost=0.005,
            variable_cost_rate=0.001,
            market_impact_model=MarketImpactModel(
                symbol="AAPL",
                temporary_impact_rate=0.0005,
                permanent_impact_rate=0.0001
            )
        )
        
        storage_key = await bridge.store_cost_model(
            model=cost_model,
            agent_id="test_agent",
            version="v1.0"
        )
        
        print(f"✓ Cost model stored with key: {storage_key}")
        
        # Test execution result storage
        execution_result = ExecutionResult(
            order_id="ORDER123",
            symbol="AAPL",
            venue="NYSE",
            quantity=1000,
            executed_price=150.25,
            execution_time=datetime.utcnow(),
            total_cost=15.50,
            market_impact=2.30
        )
        
        result_key = await bridge.store_execution_result(
            result=execution_result,
            agent_id="test_agent"
        )
        
        print(f"✓ Execution result stored with key: {result_key}")
        
        # Test query construction
        query = MemoryQuery(
            agent_type="impact_estimator",
            symbol="AAPL",
            start_date=datetime.utcnow() - timedelta(days=30),
            limit=50
        )
        
        print(f"✓ Memory query created: {query}")
        
        print("Memory bridge test completed successfully!")
    
    # Run the test
    asyncio.run(test_memory_bridge())
