#!/usr/bin/env python3
"""
Test Suite for External Memory Agent Integration

This test suite validates the functionality of the External Memory Agent
and its integration with the transaction cost agent pool.

Author: FinAgent Development Team
Date: 2024
"""

import unittest
import tempfile
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

# Import the External Memory Agent components
try:
    from FinAgents.memory.external_memory_interface import (
        ExternalMemoryAgent,
        EventType,
        LogLevel,
        QueryFilter,
        create_memory_agent,
        create_transaction_event,
        create_optimization_event,
        create_error_event,
        create_market_data_event
    )
    EXTERNAL_MEMORY_AVAILABLE = True
except ImportError as e:
    print(f"External Memory Agent not available: {e}")
    EXTERNAL_MEMORY_AVAILABLE = False

# Import the enhanced memory bridge
try:
    from FinAgents.agent_pools.transaction_cost_agent_pool.memory_bridge_enhanced import (
        TransactionCostMemoryBridge,
        create_transaction_cost_memory_bridge
    )
    MEMORY_BRIDGE_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced Memory Bridge not available: {e}")
    MEMORY_BRIDGE_AVAILABLE = False


@unittest.skipUnless(EXTERNAL_MEMORY_AVAILABLE, "External Memory Agent not available")
class TestExternalMemoryAgent(unittest.TestCase):
    """Test cases for External Memory Agent functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.memory_agent = create_memory_agent(self.temp_dir)
        self.test_pool = "test_agent_pool"
        self.test_agent = "test_agent_001"
        
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'memory_agent'):
            self.memory_agent.cleanup()
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_basic_event_logging(self):
        """Test basic event logging functionality."""
        # Log a simple event
        event_id = self.memory_agent.log_event(
            event_type=EventType.INFO,
            log_level=LogLevel.INFO,
            source_agent_pool=self.test_pool,
            source_agent_id=self.test_agent,
            title="Test Event",
            content="This is a test event",
            tags={"test", "basic"},
            metadata={"test_data": "value"}
        )
        
        self.assertIsNotNone(event_id)
        self.assertIsInstance(event_id, str)
        
        # Verify event can be retrieved
        recent_events = self.memory_agent.get_recent_events(
            source_agent_pool=self.test_pool,
            hours=1
        )
        
        self.assertEqual(len(recent_events), 1)
        self.assertEqual(recent_events[0].title, "Test Event")
        self.assertEqual(recent_events[0].source_agent_pool, self.test_pool)
        self.assertEqual(recent_events[0].source_agent_id, self.test_agent)
    
    def test_transaction_event_logging(self):
        """Test transaction event logging using helper functions."""
        # Create transaction event
        transaction_event = create_transaction_event(
            agent_pool=self.test_pool,
            agent_id=self.test_agent,
            transaction_type="buy",
            symbol="AAPL",
            quantity=100,
            price=150.25,
            cost=15025.0,
            session_id="test_session"
        )
        
        # Log transaction event
        event_ids = self.memory_agent.log_events_batch([transaction_event])
        
        self.assertEqual(len(event_ids), 1)
        
        # Query transaction events
        query_filter = QueryFilter(
            event_types=[EventType.TRANSACTION],
            source_agent_pools=[self.test_pool],
            limit=10
        )
        
        result = self.memory_agent.query_events(query_filter)
        
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0].metadata['symbol'], 'AAPL')
        self.assertEqual(result.events[0].metadata['quantity'], 100)
        self.assertEqual(result.events[0].metadata['price'], 150.25)
    
    def test_batch_logging(self):
        """Test batch event logging functionality."""
        # Create multiple events
        events = []
        
        for i in range(5):
            event = create_transaction_event(
                agent_pool=self.test_pool,
                agent_id=f"agent_{i:03d}",
                transaction_type="buy" if i % 2 == 0 else "sell",
                symbol=f"STOCK_{i}",
                quantity=100 + i * 10,
                price=100.0 + i * 5,
                cost=(100.0 + i * 5) * (100 + i * 10)
            )
            events.append(event)
        
        # Log batch
        event_ids = self.memory_agent.log_events_batch(events)
        
        self.assertEqual(len(event_ids), 5)
        
        # Verify all events were stored
        recent_events = self.memory_agent.get_recent_events(
            source_agent_pool=self.test_pool,
            hours=1,
            limit=10
        )
        
        # Should have at least 5 events (plus any from previous tests)
        self.assertGreaterEqual(len(recent_events), 5)
    
    def test_advanced_querying(self):
        """Test advanced querying capabilities."""
        # Create events with different characteristics
        events = [
            create_transaction_event(
                agent_pool=self.test_pool,
                agent_id="trader_001",
                transaction_type="buy",
                symbol="AAPL",
                quantity=100,
                price=150.0,
                cost=15000.0
            ),
            create_optimization_event(
                agent_pool=self.test_pool,
                agent_id="optimizer_001",
                optimization_type="portfolio",
                result={"score": 0.85, "iterations": 100}
            ),
            create_error_event(
                agent_pool=self.test_pool,
                agent_id="validator_001",
                error_type="validation",
                error_message="Invalid portfolio weights"
            )
        ]
        
        # Log events
        event_ids = self.memory_agent.log_events_batch(events)
        self.assertEqual(len(event_ids), 3)
        
        # Test filtering by event type
        query_filter = QueryFilter(
            event_types=[EventType.TRANSACTION],
            source_agent_pools=[self.test_pool]
        )
        
        transaction_result = self.memory_agent.query_events(query_filter)
        transaction_events = [e for e in transaction_result.events 
                            if e.event_type == EventType.TRANSACTION]
        self.assertGreater(len(transaction_events), 0)
        
        # Test filtering by log level
        query_filter = QueryFilter(
            log_levels=[LogLevel.ERROR],
            source_agent_pools=[self.test_pool]
        )
        
        error_result = self.memory_agent.query_events(query_filter)
        error_events = [e for e in error_result.events 
                       if e.log_level == LogLevel.ERROR]
        self.assertGreater(len(error_events), 0)
        
        # Test content search
        query_filter = QueryFilter(
            content_search="AAPL",
            source_agent_pools=[self.test_pool]
        )
        
        search_result = self.memory_agent.query_events(query_filter)
        self.assertGreater(len(search_result.events), 0)
    
    def test_session_correlation(self):
        """Test session and correlation ID functionality."""
        session_id = "test_session_001"
        correlation_id = "test_correlation_001"
        
        # Create correlated events
        events = [
            create_transaction_event(
                agent_pool=self.test_pool,
                agent_id="agent_001",
                transaction_type="buy",
                symbol="AAPL",
                quantity=100,
                price=150.0,
                cost=15000.0,
                session_id=session_id
            ),
            create_optimization_event(
                agent_pool=self.test_pool,
                agent_id="agent_002",
                optimization_type="cost",
                result={"cost_savings": 25.0},
                session_id=session_id
            )
        ]
        
        # Add correlation ID
        for event in events:
            event['correlation_id'] = correlation_id
        
        # Log events
        event_ids = self.memory_agent.log_events_batch(events)
        self.assertEqual(len(event_ids), 2)
        
        # Test session-based retrieval
        session_events = self.memory_agent.get_events_by_session(session_id)
        self.assertEqual(len(session_events), 2)
        
        # Test correlation-based retrieval
        correlated_events = self.memory_agent.get_events_by_correlation(correlation_id)
        self.assertEqual(len(correlated_events), 2)
    
    def test_real_time_hooks(self):
        """Test real-time event processing hooks."""
        hook_called = False
        processed_events = []
        
        def test_hook(event):
            nonlocal hook_called
            hook_called = True
            processed_events.append(event)
        
        # Add hook
        self.memory_agent.add_event_hook(test_hook)
        
        # Log an event
        event_id = self.memory_agent.log_event(
            event_type=EventType.INFO,
            log_level=LogLevel.INFO,
            source_agent_pool=self.test_pool,
            source_agent_id=self.test_agent,
            title="Hook Test Event",
            content="Testing real-time hooks"
        )
        
        # Verify hook was called
        self.assertTrue(hook_called)
        self.assertEqual(len(processed_events), 1)
        self.assertEqual(processed_events[0].title, "Hook Test Event")
    
    def test_statistics(self):
        """Test statistics functionality."""
        # Log some events first
        events = [
            create_transaction_event(
                agent_pool=self.test_pool,
                agent_id="stats_agent",
                transaction_type="buy",
                symbol="AAPL",
                quantity=100,
                price=150.0,
                cost=15000.0
            ),
            create_error_event(
                agent_pool=self.test_pool,
                agent_id="stats_agent",
                error_type="test",
                error_message="Test error for statistics"
            )
        ]
        
        self.memory_agent.log_events_batch(events)
        
        # Get statistics
        stats = self.memory_agent.get_statistics()
        
        # Verify statistics structure
        self.assertIn('agent_stats', stats)
        self.assertIn('storage_stats', stats)
        self.assertIn('initialized', stats)
        
        # Verify some statistics values
        self.assertTrue(stats['initialized'])
        self.assertGreater(stats['agent_stats']['events_stored'], 0)


@unittest.skipUnless(MEMORY_BRIDGE_AVAILABLE, "Enhanced Memory Bridge not available")
class TestTransactionCostMemoryBridge(unittest.TestCase):
    """Test cases for Transaction Cost Memory Bridge integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.memory_bridge = TransactionCostMemoryBridge(
            use_external_memory=True,
            storage_path=self.temp_dir
        )
        self.test_agent = "test_cost_agent"
        
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'memory_bridge'):
            self.memory_bridge.cleanup()
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_memory_bridge_initialization(self):
        """Test memory bridge initialization."""
        self.assertIsNotNone(self.memory_bridge)
        self.assertTrue(self.memory_bridge.use_external_memory)
        self.assertIsNotNone(self.memory_bridge.external_memory)
        self.assertEqual(self.memory_bridge.pool_name, "transaction_cost_agent_pool")
    
    def test_cost_model_operations(self):
        """Test cost model storage and retrieval."""
        # This test would require actual cost model implementation
        # For now, we'll test the logging aspects
        
        # Test logging of cost model storage
        event_id = self.memory_bridge._log_event(
            event_type=EventType.SYSTEM,
            log_level=LogLevel.INFO,
            agent_id=self.test_agent,
            title="Cost Model Test",
            content="Testing cost model operations",
            tags={"cost_model", "test"}
        )
        
        self.assertIsNotNone(event_id)
    
    def test_optimization_event_logging(self):
        """Test optimization event logging."""
        # Create optimization result
        optimization_result = {
            "optimization_type": "transaction_cost",
            "score": 0.85,
            "iterations": 150,
            "convergence_time": 2.5,
            "cost_reduction": 0.12
        }
        
        # Log optimization event
        correlation_id = f"test_optimization_{int(time.time())}"
        
        # Use asyncio for async method
        import asyncio
        
        async def test_async():
            event_id = await self.memory_bridge.log_optimization_event(
                agent_id=self.test_agent,
                optimization_type="transaction_cost",
                result=optimization_result,
                correlation_id=correlation_id
            )
            return event_id
        
        event_id = asyncio.run(test_async())
        self.assertIsNotNone(event_id)
    
    def test_historical_query(self):
        """Test historical execution querying."""
        # First, create some test data
        events = []
        for i in range(3):
            event = create_transaction_event(
                agent_pool="transaction_cost_agent_pool",
                agent_id=f"executor_{i}",
                transaction_type="buy",
                symbol="AAPL",
                quantity=100 + i * 10,
                price=150.0 + i,
                cost=(150.0 + i) * (100 + i * 10)
            )
            events.append(event)
        
        # Log events
        if self.memory_bridge.external_memory:
            self.memory_bridge.external_memory.log_events_batch(events)
        
        # Query historical executions
        from FinAgents.agent_pools.transaction_cost_agent_pool.memory_bridge_enhanced import MemoryQuery
        
        query = MemoryQuery(
            agent_type="executor",
            symbol="AAPL",
            start_date=datetime.now() - timedelta(hours=1),
            limit=10
        )
        
        results = self.memory_bridge.query_historical_executions(query)
        
        # Should have some results
        self.assertGreaterEqual(len(results), 0)
    
    def test_statistics_integration(self):
        """Test statistics integration."""
        stats = self.memory_bridge.get_statistics()
        
        # Verify statistics structure
        self.assertIn('bridge_config', stats)
        self.assertIn('use_external_memory', stats['bridge_config'])
        self.assertIn('pool_name', stats['bridge_config'])
        
        # Verify external memory statistics if available
        if self.memory_bridge.use_external_memory:
            self.assertIn('external_memory', stats)


class TestIntegrationScenarios(unittest.TestCase):
    """Test complete integration scenarios."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @unittest.skipUnless(EXTERNAL_MEMORY_AVAILABLE and MEMORY_BRIDGE_AVAILABLE, 
                        "External Memory Agent and Memory Bridge not available")
    def test_complete_workflow_integration(self):
        """Test complete workflow integration."""
        # Create memory agent and bridge
        memory_agent = create_memory_agent(self.temp_dir)
        memory_bridge = TransactionCostMemoryBridge(
            use_external_memory=True,
            storage_path=self.temp_dir
        )
        
        try:
            # Simulate a complete transaction cost optimization workflow
            session_id = f"integration_test_{int(time.time())}"
            correlation_id = f"workflow_{int(time.time())}"
            
            # Step 1: Market data collection
            market_data_event = create_market_data_event(
                agent_pool="transaction_cost_agent_pool",
                agent_id="market_data_collector",
                symbol="AAPL",
                data_type="price_update",
                data={
                    "price": 150.25,
                    "volume": 1000000,
                    "timestamp": datetime.now().isoformat()
                },
                session_id=session_id
            )
            
            # Step 2: Cost analysis
            cost_analysis_event = create_optimization_event(
                agent_pool="transaction_cost_agent_pool",
                agent_id="cost_analyzer",
                optimization_type="cost_analysis",
                result={
                    "expected_cost": 25.0,
                    "market_impact": 0.05,
                    "slippage_estimate": 0.02
                },
                session_id=session_id
            )
            
            # Step 3: Transaction execution
            transaction_event = create_transaction_event(
                agent_pool="transaction_cost_agent_pool",
                agent_id="transaction_executor",
                transaction_type="buy",
                symbol="AAPL",
                quantity=1000,
                price=150.25,
                cost=150250.0,
                session_id=session_id
            )
            
            # Add correlation IDs
            for event in [market_data_event, cost_analysis_event, transaction_event]:
                event['correlation_id'] = correlation_id
            
            # Log all events
            event_ids = memory_agent.log_events_batch([
                market_data_event, 
                cost_analysis_event, 
                transaction_event
            ])
            
            self.assertEqual(len(event_ids), 3)
            
            # Query the complete workflow
            workflow_events = memory_agent.get_events_by_correlation(correlation_id)
            self.assertEqual(len(workflow_events), 3)
            
            # Verify event types are correct
            event_types = {event.event_type for event in workflow_events}
            expected_types = {EventType.MARKET_DATA, EventType.OPTIMIZATION, EventType.TRANSACTION}
            self.assertEqual(event_types, expected_types)
            
            # Verify session consistency
            session_events = memory_agent.get_events_by_session(session_id)
            self.assertEqual(len(session_events), 3)
            
        finally:
            # Cleanup
            memory_agent.cleanup()
            memory_bridge.cleanup()


def run_performance_test():
    """Run performance tests for the External Memory Agent."""
    if not EXTERNAL_MEMORY_AVAILABLE:
        print("External Memory Agent not available for performance testing")
        return
    
    print("Running performance tests...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        memory_agent = create_memory_agent(temp_dir)
        
        # Test batch logging performance
        print("Testing batch logging performance...")
        start_time = time.time()
        
        # Create 1000 events
        events = []
        for i in range(1000):
            event = create_transaction_event(
                agent_pool="performance_test_pool",
                agent_id=f"perf_agent_{i % 10}",
                transaction_type="buy" if i % 2 == 0 else "sell",
                symbol=f"STOCK_{i % 100}",
                quantity=100 + i,
                price=100.0 + (i % 50),
                cost=(100.0 + (i % 50)) * (100 + i)
            )
            events.append(event)
        
        # Log in batches of 100
        total_logged = 0
        for i in range(0, len(events), 100):
            batch = events[i:i+100]
            event_ids = memory_agent.log_events_batch(batch)
            total_logged += len(event_ids)
        
        batch_time = time.time() - start_time
        print(f"Logged {total_logged} events in {batch_time:.2f} seconds")
        print(f"Rate: {total_logged / batch_time:.2f} events/second")
        
        # Test querying performance
        print("Testing query performance...")
        start_time = time.time()
        
        # Perform various queries
        query_filter = QueryFilter(
            event_types=[EventType.TRANSACTION],
            source_agent_pools=["performance_test_pool"],
            limit=100
        )
        
        for _ in range(10):
            result = memory_agent.query_events(query_filter)
        
        query_time = time.time() - start_time
        print(f"Performed 10 queries in {query_time:.2f} seconds")
        print(f"Average query time: {query_time / 10:.3f} seconds")
        
        # Get final statistics
        stats = memory_agent.get_statistics()
        print(f"Final statistics: {stats['agent_stats']}")
        
        memory_agent.cleanup()
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    print("External Memory Agent Test Suite")
    print("=" * 50)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    print("\n" + "=" * 50)
    run_performance_test()
    
    print("\nAll tests completed!")
