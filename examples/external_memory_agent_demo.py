#!/usr/bin/env python3
"""
Example usage of the External Memory Agent for FinAgent-Orchestration System

This example demonstrates how to use the unified memory agent for logging
and retrieving events from different agent pools in the system.

Author: FinAgent-Orchestration Team
Date: 2024
"""

import time
from datetime import datetime, timedelta
from FinAgents.memory.external_memory_interface import (
    ExternalMemoryAgent,
    EventType,
    LogLevel,
    QueryFilter,
    create_memory_agent,
    create_transaction_event,
    create_optimization_event,
    create_market_data_event,
    create_error_event
)


def demo_basic_logging():
    """Demonstrate basic event logging functionality."""
    print("=== Basic Event Logging Demo ===")
    
    # Create memory agent instance
    memory_agent = create_memory_agent("demo_memory_storage")
    
    # Log individual events
    print("\n1. Logging individual events:")
    
    # Transaction event
    event_id_1 = memory_agent.log_event(
        event_type=EventType.TRANSACTION,
        log_level=LogLevel.INFO,
        source_agent_pool="transaction_cost_agent_pool",
        source_agent_id="cost_optimizer_001",
        title="AAPL Buy Order",
        content="Executed buy order for 100 shares of AAPL at $150.25",
        tags={"transaction", "buy", "AAPL"},
        metadata={
            "symbol": "AAPL",
            "quantity": 100,
            "price": 150.25,
            "total_cost": 15025.0
        },
        session_id="session_001"
    )
    print(f"  Transaction event logged: {event_id_1}")
    
    # Market data event
    event_id_2 = memory_agent.log_event(
        event_type=EventType.MARKET_DATA,
        log_level=LogLevel.INFO,
        source_agent_pool="market_data_agent_pool",
        source_agent_id="data_collector_001",
        title="AAPL Price Update",
        content="Real-time price update for AAPL",
        tags={"market_data", "price", "AAPL"},
        metadata={
            "symbol": "AAPL",
            "price": 150.75,
            "volume": 1000000,
            "timestamp": datetime.now().isoformat()
        },
        session_id="session_001"
    )
    print(f"  Market data event logged: {event_id_2}")
    
    # Error event
    event_id_3 = memory_agent.log_event(
        event_type=EventType.ERROR,
        log_level=LogLevel.ERROR,
        source_agent_pool="portfolio_agent_pool",
        source_agent_id="portfolio_manager_001",
        title="Portfolio Calculation Error",
        content="Failed to calculate portfolio risk metrics due to missing data",
        tags={"error", "portfolio", "risk"},
        metadata={
            "error_code": "RISK_001",
            "missing_data": ["volatility", "correlation_matrix"]
        },
        session_id="session_001"
    )
    print(f"  Error event logged: {event_id_3}")
    
    print(f"\nTotal events logged: 3")


def demo_batch_logging():
    """Demonstrate batch event logging functionality."""
    print("\n=== Batch Event Logging Demo ===")
    
    memory_agent = create_memory_agent("demo_memory_storage")
    
    # Create multiple events using helper functions
    print("\n2. Creating batch events using helper functions:")
    
    batch_events = []
    
    # Transaction events
    for i in range(5):
        symbol = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"][i]
        transaction_event = create_transaction_event(
            agent_pool="transaction_cost_agent_pool",
            agent_id=f"cost_optimizer_{i:03d}",
            transaction_type="buy" if i % 2 == 0 else "sell",
            symbol=symbol,
            quantity=100 + i * 10,
            price=150.0 + i * 5,
            cost=(150.0 + i * 5) * (100 + i * 10),
            session_id="batch_session_001"
        )
        batch_events.append(transaction_event)
    
    # Optimization events
    for i in range(3):
        optimization_event = create_optimization_event(
            agent_pool="portfolio_optimization_pool",
            agent_id=f"optimizer_{i:03d}",
            optimization_type=["portfolio", "risk", "cost"][i],
            result={
                "optimization_score": 0.85 + i * 0.05,
                "iterations": 100 + i * 50,
                "convergence_time": 2.5 + i * 0.5
            },
            session_id="batch_session_001"
        )
        batch_events.append(optimization_event)
    
    # Market data events
    for i, symbol in enumerate(["AAPL", "GOOGL", "MSFT"]):
        market_data_event = create_market_data_event(
            agent_pool="market_data_pool",
            agent_id=f"data_collector_{i:03d}",
            symbol=symbol,
            data_type="price_update",
            data={
                "open": 150.0 + i * 10,
                "high": 155.0 + i * 10,
                "low": 148.0 + i * 10,
                "close": 152.0 + i * 10,
                "volume": 1000000 + i * 100000
            },
            session_id="batch_session_001"
        )
        batch_events.append(market_data_event)
    
    # Log batch events
    event_ids = memory_agent.log_events_batch(batch_events)
    print(f"  Batch logged {len(event_ids)} events")
    print(f"  Sample event IDs: {event_ids[:3]}...")


def demo_querying():
    """Demonstrate event querying functionality."""
    print("\n=== Event Querying Demo ===")
    
    memory_agent = create_memory_agent("demo_memory_storage")
    
    print("\n3. Querying events with different filters:")
    
    # Query all events from session_001
    print("\n  a) All events from session_001:")
    session_events = memory_agent.get_events_by_session("session_001", limit=10)
    for event in session_events:
        print(f"    {event.timestamp.strftime('%H:%M:%S')} - {event.title} ({event.source_agent_pool})")
    
    # Query transaction events only
    print("\n  b) Transaction events only:")
    query_filter = QueryFilter(
        event_types=[EventType.TRANSACTION],
        limit=5
    )
    transaction_result = memory_agent.query_events(query_filter)
    print(f"    Found {transaction_result.total_count} transaction events")
    for event in transaction_result.events:
        print(f"    {event.timestamp.strftime('%H:%M:%S')} - {event.title}")
    
    # Query events from specific agent pool
    print("\n  c) Events from transaction_cost_agent_pool:")
    query_filter = QueryFilter(
        source_agent_pools=["transaction_cost_agent_pool"],
        limit=5
    )
    pool_result = memory_agent.query_events(query_filter)
    print(f"    Found {pool_result.total_count} events from transaction cost pool")
    for event in pool_result.events:
        print(f"    {event.timestamp.strftime('%H:%M:%S')} - {event.title}")
    
    # Query recent events (last hour)
    print("\n  d) Recent events (last hour):")
    recent_events = memory_agent.get_recent_events(hours=1, limit=10)
    print(f"    Found {len(recent_events)} recent events")
    for event in recent_events:
        print(f"    {event.timestamp.strftime('%H:%M:%S')} - {event.title} [{event.log_level.value}]")
    
    # Content search
    print("\n  e) Content search for 'AAPL':")
    query_filter = QueryFilter(
        content_search="AAPL",
        limit=5
    )
    search_result = memory_agent.query_events(query_filter)
    print(f"    Found {search_result.total_count} events containing 'AAPL'")
    for event in search_result.events:
        print(f"    {event.timestamp.strftime('%H:%M:%S')} - {event.title}")


def demo_statistics():
    """Demonstrate statistics and monitoring functionality."""
    print("\n=== Statistics and Monitoring Demo ===")
    
    memory_agent = create_memory_agent("demo_memory_storage")
    
    print("\n4. Memory agent statistics:")
    stats = memory_agent.get_statistics()
    
    print(f"  Agent Statistics:")
    print(f"    Events stored: {stats['agent_stats']['events_stored']}")
    print(f"    Events retrieved: {stats['agent_stats']['events_retrieved']}")
    print(f"    Batch operations: {stats['agent_stats']['batch_operations']}")
    print(f"    Errors: {stats['agent_stats']['errors']}")
    
    print(f"\n  Storage Statistics:")
    print(f"    Total events: {stats['storage_stats']['total_events']}")
    print(f"    Agent pools: {stats['storage_stats']['agent_pools']}")
    print(f"    Event types: {stats['storage_stats']['event_types']}")
    print(f"    Last updated: {stats['storage_stats']['last_updated']}")


def demo_real_time_hooks():
    """Demonstrate real-time event processing hooks."""
    print("\n=== Real-time Hooks Demo ===")
    
    memory_agent = create_memory_agent("demo_memory_storage")
    
    # Define event processing hooks
    def transaction_hook(event):
        if event.event_type == EventType.TRANSACTION:
            print(f"  🔔 Transaction Alert: {event.title} - ${event.metadata.get('total_cost', 0):.2f}")
    
    def error_hook(event):
        if event.event_type == EventType.ERROR:
            print(f"  ⚠️  Error Alert: {event.title} in {event.source_agent_pool}")
    
    def batch_analysis_hook(events):
        transaction_count = sum(1 for e in events if e.event_type == EventType.TRANSACTION)
        error_count = sum(1 for e in events if e.event_type == EventType.ERROR)
        print(f"  📊 Batch Analysis: {len(events)} events ({transaction_count} transactions, {error_count} errors)")
    
    # Add hooks
    memory_agent.add_event_hook(transaction_hook)
    memory_agent.add_event_hook(error_hook)
    memory_agent.add_batch_hook(batch_analysis_hook)
    
    print("\n5. Testing real-time hooks:")
    
    # Log some events to trigger hooks
    memory_agent.log_event(
        event_type=EventType.TRANSACTION,
        log_level=LogLevel.INFO,
        source_agent_pool="demo_pool",
        source_agent_id="demo_agent",
        title="TSLA Buy Order",
        content="Demo transaction",
        metadata={"total_cost": 25000.0}
    )
    
    memory_agent.log_event(
        event_type=EventType.ERROR,
        log_level=LogLevel.ERROR,
        source_agent_pool="demo_pool",
        source_agent_id="demo_agent",
        title="Demo Error",
        content="Demo error message"
    )
    
    # Test batch hook
    batch_events = [
        create_transaction_event("demo_pool", "demo_agent", "buy", "NVDA", 50, 800.0, 40000.0),
        create_error_event("demo_pool", "demo_agent", "validation", "Demo validation error")
    ]
    memory_agent.log_events_batch(batch_events)


def demo_integration_with_transaction_cost_pool():
    """Demonstrate integration with the transaction cost agent pool."""
    print("\n=== Integration with Transaction Cost Agent Pool Demo ===")
    
    memory_agent = create_memory_agent("demo_memory_storage")
    
    print("\n6. Simulating transaction cost agent pool integration:")
    
    # Simulate transaction cost optimization workflow
    session_id = f"tc_session_{int(time.time())}"
    correlation_id = f"tc_workflow_{int(time.time())}"
    
    # Step 1: Market data collection
    memory_agent.log_event(
        event_type=EventType.MARKET_DATA,
        log_level=LogLevel.INFO,
        source_agent_pool="transaction_cost_agent_pool",
        source_agent_id="market_data_collector",
        title="Market Data Collection Started",
        content="Collecting real-time market data for cost optimization",
        tags={"market_data", "collection", "start"},
        session_id=session_id,
        correlation_id=correlation_id
    )
    
    # Step 2: Cost analysis
    memory_agent.log_event(
        event_type=EventType.OPTIMIZATION,
        log_level=LogLevel.INFO,
        source_agent_pool="transaction_cost_agent_pool",
        source_agent_id="cost_analyzer",
        title="Transaction Cost Analysis",
        content="Analyzing transaction costs for optimal execution strategy",
        tags={"cost_analysis", "optimization"},
        metadata={
            "symbols": ["AAPL", "GOOGL", "MSFT"],
            "analysis_type": "slippage_impact",
            "market_impact_threshold": 0.05
        },
        session_id=session_id,
        correlation_id=correlation_id
    )
    
    # Step 3: Optimization results
    memory_agent.log_event(
        event_type=EventType.OPTIMIZATION,
        log_level=LogLevel.INFO,
        source_agent_pool="transaction_cost_agent_pool",
        source_agent_id="cost_optimizer",
        title="Cost Optimization Completed",
        content="Transaction cost optimization completed with recommended execution strategy",
        tags={"optimization", "completed", "execution_strategy"},
        metadata={
            "optimal_strategy": "TWAP",
            "expected_cost_reduction": 0.15,
            "execution_time_horizon": 30,  # minutes
            "recommended_order_size": 1000
        },
        session_id=session_id,
        correlation_id=correlation_id
    )
    
    # Step 4: Transaction execution
    transactions = [
        ("AAPL", "buy", 500, 150.25),
        ("GOOGL", "sell", 200, 2750.50),
        ("MSFT", "buy", 300, 380.75)
    ]
    
    for symbol, action, quantity, price in transactions:
        event = create_transaction_event(
            agent_pool="transaction_cost_agent_pool",
            agent_id="transaction_executor",
            transaction_type=action,
            symbol=symbol,
            quantity=quantity,
            price=price,
            cost=quantity * price,
            session_id=session_id
        )
        event['correlation_id'] = correlation_id
        memory_agent.log_events_batch([event])
    
    print(f"  Logged complete transaction cost optimization workflow")
    print(f"  Session ID: {session_id}")
    print(f"  Correlation ID: {correlation_id}")
    
    # Query the complete workflow
    print("\n  Workflow events:")
    workflow_events = memory_agent.get_events_by_correlation(correlation_id)
    for i, event in enumerate(workflow_events, 1):
        print(f"    {i}. {event.timestamp.strftime('%H:%M:%S')} - {event.title} ({event.event_type.value})")


def main():
    """Main demo function."""
    print("External Memory Agent Demo")
    print("=" * 50)
    
    try:
        # Run all demo functions
        demo_basic_logging()
        demo_batch_logging()
        demo_querying()
        demo_statistics()
        demo_real_time_hooks()
        demo_integration_with_transaction_cost_pool()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nThe External Memory Agent is ready for integration with all agent pools.")
        print("Check the 'demo_memory_storage' directory for stored events.")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
