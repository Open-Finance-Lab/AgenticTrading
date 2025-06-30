#!/usr/bin/env python3
"""
Test script for External Memory Agent

This script tests the basic functionality of the External Memory Agent
including storing events, querying, and batch operations.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add the memory directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from external_memory_agent import (
    ExternalMemoryAgent, 
    EventType, 
    LogLevel, 
    QueryFilter,
    SQLiteStorageBackend
)

async def test_basic_functionality():
    """Test basic memory agent functionality."""
    print("🧪 Testing External Memory Agent - Basic Functionality")
    print("=" * 60)
    
    # Initialize memory agent with SQLite backend
    memory_agent = ExternalMemoryAgent(
        storage_backend=SQLiteStorageBackend("test_memory.db"),
        enable_real_time_hooks=True
    )
    
    try:
        # Initialize the agent
        print("1. Initializing memory agent...")
        await memory_agent.initialize()
        print("✅ Memory agent initialized successfully")
        
        # Test single event logging
        print("\n2. Testing single event logging...")
        event_id = await memory_agent.log_event(
            event_type=EventType.MARKET_DATA,
            log_level=LogLevel.INFO,
            source_agent_pool="data_agent_pool",
            source_agent_id="polygon_agent",
            title="Market data retrieved",
            content="Successfully retrieved AAPL daily data for 2024-01-01 to 2024-12-31",
            tags={"market_data", "AAPL", "daily"},
            metadata={
                "symbol": "AAPL",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "data_points": 252
            },
            session_id="test_session_001"
        )
        print(f"✅ Single event logged with ID: {event_id}")
        
        # Test batch event logging
        print("\n3. Testing batch event logging...")
        batch_events = [
            {
                "event_type": "optimization",
                "log_level": "info",
                "source_agent_pool": "alpha_agent_pool",
                "source_agent_id": "momentum_agent",
                "title": "Portfolio optimization completed",
                "content": "Optimized portfolio allocation for maximum Sharpe ratio",
                "tags": ["optimization", "portfolio", "sharpe"],
                "metadata": {"sharpe_ratio": 1.85, "volatility": 0.15},
                "session_id": "test_session_001"
            },
            {
                "event_type": "transaction",
                "log_level": "info",
                "source_agent_pool": "execution_agent_pool",
                "source_agent_id": "order_manager",
                "title": "Order executed",
                "content": "Buy order for 100 shares of AAPL executed at $150.25",
                "tags": ["transaction", "buy", "AAPL"],
                "metadata": {"symbol": "AAPL", "quantity": 100, "price": 150.25},
                "session_id": "test_session_001"
            },
            {
                "event_type": "error",
                "log_level": "error",
                "source_agent_pool": "risk_agent_pool",
                "source_agent_id": "var_calculator",
                "title": "VaR calculation failed",
                "content": "Insufficient historical data for VaR calculation",
                "tags": ["error", "var", "risk"],
                "metadata": {"error_code": "INSUFFICIENT_DATA"},
                "session_id": "test_session_001"
            }
        ]
        
        batch_ids = await memory_agent.log_events_batch(batch_events)
        print(f"✅ Batch events logged: {len(batch_ids)} events")
        
        # Test querying recent events
        print("\n4. Testing event querying...")
        
        # Query all events from current session
        session_events = await memory_agent.get_events_by_session("test_session_001")
        print(f"✅ Found {len(session_events)} events in test session")
        
        # Query events by type
        query_filter = QueryFilter(
            event_types=[EventType.MARKET_DATA, EventType.OPTIMIZATION],
            limit=10
        )
        query_result = await memory_agent.query_events(query_filter)
        print(f"✅ Query completed: {len(query_result.events)} events found in {query_result.query_time_ms:.2f}ms")
        
        # Query recent events
        recent_events = await memory_agent.get_recent_events(hours=1, limit=10)
        print(f"✅ Found {len(recent_events)} recent events")
        
        # Test content search
        print("\n5. Testing content search...")
        content_filter = QueryFilter(
            content_search="AAPL",
            limit=5
        )
        search_result = await memory_agent.query_events(content_filter)
        print(f"✅ Content search for 'AAPL' found {len(search_result.events)} events")
        
        # Get statistics
        print("\n6. Getting statistics...")
        stats = await memory_agent.get_statistics()
        print("✅ Memory agent statistics:")
        print(f"   - Total events stored: {stats['agent_stats']['events_stored']}")
        print(f"   - Total events retrieved: {stats['agent_stats']['events_retrieved']}")
        print(f"   - Total batch operations: {stats['agent_stats']['batch_operations']}")
        print(f"   - Storage stats: {stats['storage_stats']}")
        
        # Display some sample events
        print("\n7. Sample events from session:")
        for i, event in enumerate(session_events[:3]):
            print(f"   Event {i+1}:")
            print(f"     - Type: {event.event_type.value}")
            print(f"     - Title: {event.title}")
            print(f"     - Source: {event.source_agent_pool}.{event.source_agent_id}")
            print(f"     - Tags: {', '.join(event.tags)}")
            print(f"     - Timestamp: {event.timestamp}")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        await memory_agent.cleanup()
        print("🧹 Cleanup completed")

async def test_real_time_hooks():
    """Test real-time event processing hooks."""
    print("\n🧪 Testing Real-time Event Hooks")
    print("=" * 40)
    
    # Create a memory agent with hooks enabled
    memory_agent = ExternalMemoryAgent(enable_real_time_hooks=True)
    
    # Create some hook functions
    events_processed = []
    
    def event_hook(event):
        """Hook that processes individual events."""
        events_processed.append(event.event_id)
        print(f"🪝 Event hook triggered for: {event.title}")
    
    def batch_hook(events):
        """Hook that processes batches of events."""
        print(f"🪝 Batch hook triggered for {len(events)} events")
    
    # Add hooks
    memory_agent.add_event_hook(event_hook)
    memory_agent.add_batch_hook(batch_hook)
    
    try:
        await memory_agent.initialize()
        
        # Log some events to trigger hooks
        await memory_agent.log_event(
            event_type=EventType.SYSTEM,
            log_level=LogLevel.INFO,
            source_agent_pool="test_pool",
            source_agent_id="test_agent",
            title="Hook test event 1",
            content="Testing real-time hooks functionality"
        )
        
        # Log batch events
        batch_events = [
            {
                "event_type": "system",
                "log_level": "info",
                "source_agent_pool": "test_pool",
                "source_agent_id": "test_agent",
                "title": f"Hook test batch event {i}",
                "content": "Testing batch hooks functionality"
            }
            for i in range(3)
        ]
        
        await memory_agent.log_events_batch(batch_events)
        
        print(f"✅ Real-time hooks test completed. Processed {len(events_processed)} individual events")
        
    finally:
        await memory_agent.cleanup()

async def test_performance():
    """Test memory agent performance with larger datasets."""
    print("\n🧪 Testing Performance with Larger Dataset")
    print("=" * 45)
    
    memory_agent = ExternalMemoryAgent()
    
    try:
        await memory_agent.initialize()
        
        # Test batch performance
        print("Testing batch insert performance...")
        start_time = datetime.now()
        
        # Create a large batch of events
        large_batch = []
        for i in range(1000):
            large_batch.append({
                "event_type": "market_data",
                "log_level": "info",
                "source_agent_pool": "performance_test",
                "source_agent_id": f"agent_{i % 10}",
                "title": f"Performance test event {i}",
                "content": f"This is performance test event number {i} with some content",
                "tags": ["performance", "test", f"batch_{i // 100}"],
                "metadata": {"index": i, "batch": i // 100}
            })
        
        batch_ids = await memory_agent.log_events_batch(large_batch)
        batch_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ Stored {len(batch_ids)} events in {batch_time:.2f} seconds")
        print(f"   Rate: {len(batch_ids) / batch_time:.1f} events/second")
        
        # Test query performance
        print("Testing query performance...")
        start_time = datetime.now()
        
        query_filter = QueryFilter(
            source_agent_pools=["performance_test"],
            limit=100
        )
        result = await memory_agent.query_events(query_filter)
        query_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ Retrieved {len(result.events)} events in {query_time:.3f} seconds")
        print(f"   Query reported time: {result.query_time_ms:.2f}ms")
        
    finally:
        await memory_agent.cleanup()

async def main():
    """Run all tests."""
    print("🚀 Starting External Memory Agent Tests")
    print("=" * 50)
    
    # Run basic functionality tests
    await test_basic_functionality()
    
    # Run real-time hooks tests
    await test_real_time_hooks()
    
    # Run performance tests
    await test_performance()
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    # Ensure we're in the correct directory
    os.chdir(Path(__file__).parent)
    
    # Run the tests
    asyncio.run(main())
