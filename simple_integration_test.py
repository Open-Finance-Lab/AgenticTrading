#!/usr/bin/env python3
"""
Simple integration test for External Memory Agent

This script performs basic integration tests for the External Memory Agent
without requiring the full test framework.
"""

import sys
import os
import tempfile
import shutil
from datetime import datetime, timedelta

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_external_memory_agent():
    """Test basic External Memory Agent functionality."""
    print("Testing External Memory Agent...")
    
    try:
        from FinAgents.memory.external_memory_interface import (
            create_memory_agent,
            EventType,
            LogLevel,
            QueryFilter,
            create_transaction_event
        )
        
        # Create temporary storage
        temp_dir = tempfile.mkdtemp()
        try:
            # Initialize memory agent
            memory_agent = create_memory_agent(temp_dir)
            print("✓ Memory agent created successfully")
            
            # Log a test event
            event_id = memory_agent.log_event(
                event_type=EventType.TRANSACTION,
                log_level=LogLevel.INFO,
                source_agent_pool="test_pool",
                source_agent_id="test_agent",
                title="Test Transaction",
                content="This is a test transaction",
                tags={"test", "transaction"},
                metadata={"symbol": "TEST", "quantity": 100}
            )
            print(f"✓ Event logged with ID: {event_id[:8]}...")
            
            # Query events
            recent_events = memory_agent.get_recent_events(
                source_agent_pool="test_pool",
                hours=1
            )
            print(f"✓ Retrieved {len(recent_events)} recent events")
            
            # Test batch logging
            batch_events = [
                create_transaction_event(
                    agent_pool="test_pool",
                    agent_id="batch_agent",
                    transaction_type="buy",
                    symbol="AAPL",
                    quantity=100,
                    price=150.0,
                    cost=15000.0
                ),
                create_transaction_event(
                    agent_pool="test_pool",
                    agent_id="batch_agent",
                    transaction_type="sell",
                    symbol="GOOGL",
                    quantity=50,
                    price=2750.0,
                    cost=137500.0
                )
            ]
            
            event_ids = memory_agent.log_events_batch(batch_events)
            print(f"✓ Batch logged {len(event_ids)} events")
            
            # Test querying with filters
            query_filter = QueryFilter(
                event_types=[EventType.TRANSACTION],
                source_agent_pools=["test_pool"],
                limit=10
            )
            
            result = memory_agent.query_events(query_filter)
            print(f"✓ Query returned {len(result.events)} events")
            
            # Test statistics
            stats = memory_agent.get_statistics()
            print(f"✓ Statistics: {stats['storage_stats']['total_events']} total events")
            
            # Cleanup
            memory_agent.cleanup()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        print("✓ External Memory Agent test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ External Memory Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_bridge_integration():
    """Test Memory Bridge integration."""
    print("\nTesting Memory Bridge integration...")
    
    try:
        from FinAgents.agent_pools.transaction_cost_agent_pool.memory_bridge import (
            TransactionCostMemoryBridge
        )
        
        # Create temporary storage
        temp_dir = tempfile.mkdtemp()
        try:
            # Initialize memory bridge
            memory_bridge = TransactionCostMemoryBridge(
                use_external_memory=True,
                storage_path=temp_dir
            )
            print("✓ Memory bridge created successfully")
            
            # Test basic logging
            event_id = memory_bridge._log_event(
                event_type="TRANSACTION",  # Use string instead of enum for compatibility
                log_level="INFO",
                agent_id="test_cost_agent",
                title="Test Cost Event",
                content="Testing cost event logging",
                tags={"cost", "test"}
            )
            print(f"✓ Cost event logged: {event_id is not None}")
            
            # Test statistics
            stats = memory_bridge.get_statistics()
            print(f"✓ Bridge statistics retrieved: {stats['bridge_config']['use_external_memory']}")
            
            # Cleanup
            memory_bridge.cleanup()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        print("✓ Memory Bridge integration test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Memory Bridge integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transaction_cost_pool_import():
    """Test transaction cost pool import."""
    print("\nTesting Transaction Cost Pool import...")
    
    try:
        from FinAgents.agent_pools.transaction_cost_agent_pool import (
            TransactionCostAgentPool,
            AGENT_REGISTRY,
            get_version
        )
        
        print(f"✓ Transaction Cost Pool imported successfully")
        print(f"✓ Version: {get_version()}")
        print(f"✓ Registry has {len(AGENT_REGISTRY)} agents")
        
        return True
        
    except Exception as e:
        print(f"✗ Transaction Cost Pool import failed: {e}")
        # This is expected to fail due to missing schema components
        print("  Note: This failure is expected due to incomplete schema models")
        return False


def main():
    """Run all integration tests."""
    print("External Memory Agent Integration Tests")
    print("=" * 50)
    
    tests = [
        ("External Memory Agent Basic Test", test_external_memory_agent),
        ("Memory Bridge Integration Test", test_memory_bridge_integration),
        ("Transaction Cost Pool Import Test", test_transaction_cost_pool_import),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! External Memory Agent is ready for use.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
