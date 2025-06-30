#!/usr/bin/env python3
"""
Memory System Demo and Startup Script

This script demonstrates how to start and use the FinAgent Memory System.
It shows both the External Memory Agent (SQLite-based) and provides guidance
for the Neo4j-based Memory Server.
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime

# Add the memory directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from external_memory_agent import (
    ExternalMemoryAgent, 
    EventType, 
    LogLevel, 
    QueryFilter
)

class MemorySystemDemo:
    """Demo class for the FinAgent Memory System."""
    
    def __init__(self):
        self.memory_agent = None
    
    async def initialize_memory_system(self):
        """Initialize the memory system."""
        print("🧠 Initializing FinAgent Memory System")
        print("=" * 50)
        
        # Initialize External Memory Agent
        self.memory_agent = ExternalMemoryAgent(
            enable_real_time_hooks=True,
            max_batch_size=1000
        )
        
        await self.memory_agent.initialize()
        print("✅ External Memory Agent initialized successfully")
        
        # Add some demo hooks
        def event_monitor(event):
            if event.log_level == LogLevel.ERROR:
                print(f"🚨 ERROR detected: {event.title} from {event.source_agent_pool}")
        
        def batch_monitor(events):
            if len(events) > 10:
                print(f"📊 Large batch processed: {len(events)} events")
        
        self.memory_agent.add_event_hook(event_monitor)
        self.memory_agent.add_batch_hook(batch_monitor)
        
        return self.memory_agent
    
    async def demo_agent_pool_integration(self):
        """Demonstrate how agent pools can integrate with the memory system."""
        print("\n🔗 Demonstrating Agent Pool Integration")
        print("=" * 50)
        
        # Simulate events from different agent pools
        demo_events = [
            # Data Agent Pool Events
            {
                "event_type": EventType.MARKET_DATA,
                "log_level": LogLevel.INFO,
                "source_agent_pool": "data_agent_pool",
                "source_agent_id": "polygon_agent",
                "title": "Market data fetched",
                "content": "Successfully retrieved daily OHLCV data for AAPL",
                "tags": {"market_data", "AAPL", "daily"},
                "metadata": {"symbol": "AAPL", "records": 252}
            },
            # Alpha Agent Pool Events
            {
                "event_type": EventType.OPTIMIZATION,
                "log_level": LogLevel.INFO,
                "source_agent_pool": "alpha_agent_pool",
                "source_agent_id": "momentum_agent",
                "title": "Strategy signal generated",
                "content": "Generated BUY signal for AAPL based on momentum indicators",
                "tags": {"strategy", "momentum", "BUY", "AAPL"},
                "metadata": {"signal": "BUY", "confidence": 0.85}
            },
            # Risk Agent Pool Events
            {
                "event_type": EventType.PORTFOLIO_UPDATE,
                "log_level": LogLevel.WARNING,
                "source_agent_pool": "risk_agent_pool",
                "source_agent_id": "var_calculator",
                "title": "VaR threshold exceeded",
                "content": "Portfolio VaR exceeded risk limits: 2.5% vs 2.0% limit",
                "tags": {"risk", "var", "threshold", "warning"},
                "metadata": {"current_var": 0.025, "limit": 0.020}
            },
            # Execution Agent Pool Events
            {
                "event_type": EventType.TRANSACTION,
                "log_level": LogLevel.INFO,
                "source_agent_pool": "execution_agent_pool",
                "source_agent_id": "order_manager",
                "title": "Order executed",
                "content": "Executed buy order: 100 shares AAPL at $150.25",
                "tags": {"execution", "buy", "AAPL"},
                "metadata": {"symbol": "AAPL", "quantity": 100, "price": 150.25}
            }
        ]
        
        # Log each event
        session_id = f"demo_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        for event_data in demo_events:
            await self.memory_agent.log_event(
                event_type=event_data["event_type"],
                log_level=event_data["log_level"],
                source_agent_pool=event_data["source_agent_pool"],
                source_agent_id=event_data["source_agent_id"],
                title=event_data["title"],
                content=event_data["content"],
                tags=event_data["tags"],
                metadata=event_data["metadata"],
                session_id=session_id
            )
            print(f"✅ Logged: {event_data['title']}")
        
        return session_id
    
    async def demo_querying_capabilities(self, session_id):
        """Demonstrate memory querying capabilities."""
        print("\n🔍 Demonstrating Query Capabilities")
        print("=" * 50)
        
        # Query 1: Get all events from the demo session
        print("1. Querying events from demo session...")
        session_events = await self.memory_agent.get_events_by_session(session_id)
        print(f"   Found {len(session_events)} events in session")
        
        # Query 2: Search by content
        print("\n2. Searching for AAPL-related events...")
        aapl_filter = QueryFilter(content_search="AAPL", limit=10)
        aapl_result = await self.memory_agent.query_events(aapl_filter)
        print(f"   Found {len(aapl_result.events)} AAPL-related events")
        
        # Query 3: Filter by event type
        print("\n3. Filtering by event types...")
        type_filter = QueryFilter(
            event_types=[EventType.MARKET_DATA, EventType.TRANSACTION],
            limit=10
        )
        type_result = await self.memory_agent.query_events(type_filter)
        print(f"   Found {len(type_result.events)} market data and transaction events")
        
        # Query 4: Filter by agent pool
        print("\n4. Filtering by agent pool...")
        pool_filter = QueryFilter(
            source_agent_pools=["risk_agent_pool"],
            limit=10
        )
        pool_result = await self.memory_agent.query_events(pool_filter)
        print(f"   Found {len(pool_result.events)} events from risk agent pool")
        
        # Query 5: Recent events
        print("\n5. Getting recent events...")
        recent_events = await self.memory_agent.get_recent_events(hours=1, limit=10)
        print(f"   Found {len(recent_events)} recent events")
        
        return session_events
    
    async def demo_statistics(self):
        """Demonstrate statistics collection."""
        print("\n📊 Memory System Statistics")
        print("=" * 50)
        
        stats = await self.memory_agent.get_statistics()
        
        print("Agent Statistics:")
        agent_stats = stats['agent_stats']
        for key, value in agent_stats.items():
            print(f"  - {key}: {value}")
        
        print("\nStorage Statistics:")
        storage_stats = stats['storage_stats']
        for key, value in storage_stats.items():
            print(f"  - {key}: {value}")
    
    async def demo_event_details(self, events):
        """Show detailed event information."""
        print("\n📋 Sample Event Details")
        print("=" * 50)
        
        for i, event in enumerate(events[:2]):  # Show first 2 events
            print(f"\nEvent {i+1}:")
            print(f"  ID: {event.event_id}")
            print(f"  Type: {event.event_type.value}")
            print(f"  Level: {event.log_level.value}")
            print(f"  Source: {event.source_agent_pool}.{event.source_agent_id}")
            print(f"  Title: {event.title}")
            print(f"  Content: {event.content}")
            print(f"  Tags: {', '.join(event.tags)}")
            print(f"  Metadata: {event.metadata}")
            print(f"  Timestamp: {event.timestamp}")
    
    async def cleanup(self):
        """Clean up resources."""
        if self.memory_agent:
            await self.memory_agent.cleanup()
            print("🧹 Memory system cleanup completed")

async def main():
    """Main demo function."""
    print("🚀 FinAgent Memory System Demo")
    print("=" * 40)
    
    demo = MemorySystemDemo()
    
    try:
        # Initialize the memory system
        await demo.initialize_memory_system()
        
        # Demonstrate agent pool integration
        session_id = await demo.demo_agent_pool_integration()
        
        # Demonstrate querying capabilities
        events = await demo.demo_querying_capabilities(session_id)
        
        # Show statistics
        await demo.demo_statistics()
        
        # Show event details
        await demo.demo_event_details(events)
        
        print("\n" + "=" * 60)
        print("🎉 Memory System Demo Completed Successfully!")
        print("=" * 60)
        
        print("\n📚 Usage Guide:")
        print("1. Import: from memory.external_memory_agent import ExternalMemoryAgent")
        print("2. Initialize: memory_agent = ExternalMemoryAgent()")
        print("3. Start: await memory_agent.initialize()")
        print("4. Log events: await memory_agent.log_event(...)")
        print("5. Query: await memory_agent.query_events(filter)")
        print("6. Cleanup: await memory_agent.cleanup()")
        
        print("\n🔧 Integration with Agent Pools:")
        print("- Add memory_agent to your agent pool's __init__")
        print("- Call log_event() for significant actions")
        print("- Use session_id for workflow tracking")
        print("- Use correlation_id for related events")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await demo.cleanup()

if __name__ == "__main__":
    # Ensure we're in the correct directory
    os.chdir(Path(__file__).parent)
    
    # Run the demo
    asyncio.run(main())
