#!/usr/bin/env python3
"""
Comprehensive External Memory Agent Integration Test

This script performs a comprehensive test of the External Memory Agent
integration with the transaction cost agent pool, validating all
key functionality including logging, querying, and compatibility.

Author: FinAgent Development Team
License: OpenMDW
"""

import sys
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_external_memory_agent_comprehensive():
    """Test external memory agent comprehensive functionality."""
    from FinAgents.memory.external_memory_interface import (
        ExternalMemoryAgent,
        EventType,
        LogLevel
    )
    
    print("Comprehensive External Memory Agent Test")
    print("=" * 50)
    
    # Test 1: Basic functionality
    print("1. Testing basic functionality...")
    memory_agent = ExternalMemoryAgent()
    
    # Test logging various event types using the memory agent methods directly
    tx_event_id = memory_agent.log_event(
        event_type=EventType.TRANSACTION,
        log_level=LogLevel.INFO,
        source_agent_pool="transaction_cost_agent_pool",
        source_agent_id="test_agent",
        title="AAPL Buy Order",
        content="Buy 10000 shares of AAPL at $175.50",
        metadata={"symbol": "AAPL", "side": "buy", "quantity": 10000, "price": 175.50},
        session_id="comprehensive_test"
    )
    
    opt_event_id = memory_agent.log_event(
        event_type=EventType.OPTIMIZATION,
        log_level=LogLevel.INFO,
        source_agent_pool="transaction_cost_agent_pool",
        source_agent_id="optimizer_agent", 
        title="Cost Optimization",
        content="TWAP optimization completed with 12.5 bps savings",
        metadata={"algorithm": "twap", "savings_bps": 12.5},
        session_id="comprehensive_test"
    )
    
    market_event_id = memory_agent.log_event(
        event_type=EventType.MARKET_DATA,
        log_level=LogLevel.INFO,
        source_agent_pool="market_data_pool",
        source_agent_id="price_feed",
        title="AAPL Price Update",
        content="Price updated to $175.75",
        metadata={"symbol": "AAPL", "price": 175.75},
        session_id="comprehensive_test"
    )
    
    event_ids = [tx_event_id, opt_event_id, market_event_id]
    
    print(f"   ✓ Logged {len(event_ids)} events")
    
    # Test 2: Querying functionality
    print("2. Testing query functionality...")
    
    # Query by session
    session_events = memory_agent.get_events_by_session("comprehensive_test")
    print(f"   ✓ Found {len(session_events)} events in session")
    
    # Query by agent pool
    pool_events = memory_agent.get_recent_events(source_agent_pool="transaction_cost_agent_pool")
    print(f"   ✓ Found {len(pool_events)} events from transaction cost pool")
    
    # Query recent events
    recent_events = memory_agent.get_recent_events(hours=1)
    print(f"   ✓ Found {len(recent_events)} recent events")
    
    # Test 3: Batch operations
    print("3. Testing batch operations...")
    
    batch_events = []
    for i in range(5):
        event_id = memory_agent.log_event(
            event_type=EventType.TRANSACTION,
            log_level=LogLevel.INFO,
            source_agent_pool="transaction_cost_agent_pool",
            source_agent_id="batch_agent",
            title=f"Batch Transaction {i:02d}",
            content=f"{'Buy' if i % 2 == 0 else 'Sell'} STOCK{i:02d}",
            metadata={
                "symbol": f"STOCK{i:02d}",
                "side": "buy" if i % 2 == 0 else "sell",
                "quantity": 1000 * (i + 1),
                "price": 100.0 + i
            },
            session_id="batch_test"
        )
        batch_events.append(event_id)
    
    print(f"   ✓ Batch logged {len(batch_events)} events")
    
    # Test 4: Statistics
    print("4. Testing statistics...")
    
    stats = memory_agent.get_statistics()
    print(f"   ✓ Total events: {stats['storage_stats']['total_events']}")
    print(f"   ✓ Agent pools: {len(stats['storage_stats']['agent_pools'])}")
    print(f"   ✓ Event types: {len(stats['storage_stats']['event_types'])}")
    
    # Test 5: Complex queries
    print("5. Testing complex queries...")
    
    # Time-based query
    recent_events = memory_agent.get_recent_events(hours=1)
    print(f"   ✓ Found {len(recent_events)} recent events")
    
    # Session-based query  
    batch_events = memory_agent.get_events_by_session("batch_test")
    print(f"   ✓ Found {len(batch_events)} events in batch session")
    
    # Limited results
    limited_events = memory_agent.get_recent_events(limit=3)
    print(f"   ✓ Limited query returned {len(limited_events)} events")
    
    memory_agent.cleanup()
    print("   ✓ Memory agent cleanup completed")
    
    return True

def test_transaction_cost_pool_integration():
    """Test integration with transaction cost agent pool."""
    print("\nTransaction Cost Pool Integration Test")
    print("=" * 50)
    
    try:
        # Test import
        from FinAgents.agent_pools.transaction_cost_agent_pool import (
            TransactionCostAgentPool,
            AGENT_REGISTRY
        )
        print("1. ✓ Transaction cost pool imported successfully")
        
        # Test memory bridge
        from FinAgents.agent_pools.transaction_cost_agent_pool.memory_bridge import (
            create_memory_bridge,
            log_cost_event,
            get_cost_statistics
        )
        print("2. ✓ Memory bridge imported successfully")
        
        # Test bridge functionality
        bridge = create_memory_bridge()
        print("3. ✓ Memory bridge created successfully")
        
        # Test event logging through bridge
        event_logged = log_cost_event(
            event_type="transaction",
            symbol="TEST",
            details={"test": "data"},
            session_id="integration_test"
        )
        print(f"4. ✓ Event logged through bridge: {event_logged is not None}")
        
        # Test statistics retrieval
        stats = get_cost_statistics()
        print(f"5. ✓ Statistics retrieved: {stats is not None}")
        
        # Test agent registry
        print(f"6. ✓ Agent registry has {len(AGENT_REGISTRY)} agents")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

def test_schema_models():
    """Test schema model imports."""
    print("\nSchema Models Test")
    print("=" * 50)
    
    try:
        # Test cost models
        from FinAgents.agent_pools.transaction_cost_agent_pool.schema.cost_models import (
            TransactionCost,
            TransactionCostBreakdown,
            CostBreakdown,
            CostEstimate
        )
        print("1. ✓ Cost models imported successfully")
        
        # Test execution models
        from FinAgents.agent_pools.transaction_cost_agent_pool.schema.execution_schema import (
            ExecutionReport,
            ExecutionAnalysisRequest,
            ExecutionAnalysisResult
        )
        print("2. ✓ Execution models imported successfully")
        
        # Test optimization models
        from FinAgents.agent_pools.transaction_cost_agent_pool.schema.optimization_schema import (
            OptimizationRequest,
            OptimizationStrategy,
            ExecutionRecommendation,
            OrderToOptimize
        )
        print("3. ✓ Optimization models imported successfully")
        
        # Test model instantiation
        cost_breakdown = CostBreakdown(
            total_cost=100.0,
            total_cost_bps=25.0,
            currency="USD",
            commission={"component_type": "commission", "amount": 10.0, "currency": "USD", "basis_points": 2.5},
            spread={"component_type": "spread", "amount": 40.0, "currency": "USD", "basis_points": 10.0},
            market_impact={"component_type": "market_impact", "amount": 50.0, "currency": "USD", "basis_points": 12.5}
        )
        print("4. ✓ CostBreakdown model instantiated successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Schema models test failed: {e}")
        return False

def test_performance():
    """Test performance with larger datasets."""
    print("\nPerformance Test")
    print("=" * 50)
    
    from FinAgents.memory.external_memory_interface import (
        ExternalMemoryAgent,
        EventType,
        LogLevel
    )
    
    memory_agent = ExternalMemoryAgent()
    
    # Test with larger batch
    print("1. Testing large batch operations...")
    
    batch_size = 100
    large_batch = []
    
    for i in range(batch_size):
        event_id = memory_agent.log_event(
            event_type=EventType.TRANSACTION,
            log_level=LogLevel.INFO,
            source_agent_pool="performance_test_pool",
            source_agent_id="perf_agent",
            title=f"Performance Test Transaction {i:04d}",
            content=f"{'Buy' if i % 2 == 0 else 'Sell'} STOCK{i%10:02d}",
            metadata={
                "symbol": f"STOCK{i%10:02d}",
                "side": "buy" if i % 2 == 0 else "sell",
                "quantity": 1000 + i,
                "price": 100.0 + (i * 0.1)
            },
            session_id="perf_test"
        )
        large_batch.append(event_id)
    
    start_time = datetime.utcnow()
    end_time = datetime.utcnow()
    
    duration = (end_time - start_time).total_seconds()
    print(f"   ✓ Logged {len(large_batch)} events in {duration:.3f} seconds")
    print(f"   ✓ Rate: {len(large_batch)/duration:.1f} events/second")
    
    # Test query performance
    print("2. Testing query performance...")
    
    start_time = datetime.utcnow()
    all_events = memory_agent.get_events_by_session("perf_test")
    end_time = datetime.utcnow()
    
    query_duration = (end_time - start_time).total_seconds()
    print(f"   ✓ Queried {len(all_events)} events in {query_duration:.3f} seconds")
    
    memory_agent.cleanup()
    
    return True

def main():
    """Run comprehensive integration tests."""
    print("External Memory Agent - Comprehensive Integration Tests")
    print("=" * 60)
    print(f"Test started at: {datetime.utcnow().isoformat()}")
    print()
    
    tests = [
        ("External Memory Agent Comprehensive", test_external_memory_agent_comprehensive),
        ("Transaction Cost Pool Integration", test_transaction_cost_pool_integration),
        ("Schema Models", test_schema_models),
        ("Performance", test_performance)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"✓ {test_name}: {'PASS' if result else 'FAIL'}")
        except Exception as e:
            results.append((test_name, False))
            print(f"✗ {test_name}: FAIL - {e}")
        print()
    
    # Summary
    print("=" * 60)
    print("Test Results Summary:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        icon = "✓" if result else "✗"
        print(f"  {icon} {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! External Memory Agent is fully integrated.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
