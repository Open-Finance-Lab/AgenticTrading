#!/usr/bin/env python3
"""
Quick A2A Integration Test Example

This script demonstrates the A2A integration between the momentum agent
and memory agent with a simplified test workflow.

Usage:
    python examples/quick_a2a_integration_test.py
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_a2a_integration():
    """
    Quick test of A2A integration between momentum agent and memory agent.
    """
    print("🧪 Quick A2A Integration Test")
    print("="*50)
    
    # Test 1: Import all A2A components
    print("\n1. Testing imports...")
    try:
        from FinAgents.agent_pools.alpha_agent_pool.agents.theory_driven.a2a_client import (
            create_alpha_pool_a2a_client
        )
        from FinAgents.agent_pools.alpha_agent_pool.a2a_memory_coordinator import (
            initialize_pool_coordinator, shutdown_pool_coordinator
        )
        print("✅ All A2A components imported successfully")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 2: Create A2A client
    print("\n2. Creating A2A client...")
    try:
        a2a_client = await create_alpha_pool_a2a_client(
            agent_pool_id="test_pool",
            memory_url="http://127.0.0.1:8011"
        )
        print("✅ A2A client created successfully")
    except Exception as e:
        print(f"❌ A2A client creation failed: {e}")
        return False
    
    # Test 3: Test A2A memory operations (mock)
    print("\n3. Testing A2A memory operations...")
    try:
        # Test storing a signal
        result = await a2a_client.store_alpha_signal_event(
            agent_id="momentum_agent",
            signal="BUY",
            confidence=0.8,
            symbol="AAPL",
            reasoning="Strong momentum signal detected with high confidence",
            market_context={
                "signal_type": "momentum",
                "strength": 0.75,
                "direction": "long",
                "timeframe": "1d"
            }
        )
        print(f"✅ Signal stored: {result}")
        
        # Test retrieving strategy insights
        insights = await a2a_client.retrieve_strategy_insights(
            search_query="momentum strategy insights for AAPL",
            limit=5
        )
        print(f"✅ Strategy insights retrieved: {len(insights)} items")
        
    except Exception as e:
        print(f"❌ A2A memory operations failed: {e}")
        return False
    
    # Test 4: Test pool coordinator
    print("\n4. Testing pool coordinator...")
    try:
        coordinator = await initialize_pool_coordinator(
            pool_id="test_pool",
            memory_url="http://127.0.0.1:8010"
        )
        
        # Register a test agent
        await coordinator.register_agent(
            agent_id="test_momentum_agent",
            agent_type="momentum",
            agent_config={"window": 20}
        )
        print("✅ Agent registered with coordinator")
        
        # Test background coordination
        await asyncio.sleep(1.0)  # Let background tasks run
        
        # Shutdown coordinator
        await shutdown_pool_coordinator()
        print("✅ Pool coordinator shut down successfully")
        
    except Exception as e:
        print(f"❌ Pool coordinator test failed: {e}")
        return False
    
    # Test 5: Enhanced MCP lifecycle (if available)
    print("\n5. Testing enhanced MCP lifecycle...")
    try:
        from FinAgents.agent_pools.alpha_agent_pool.enhanced_mcp_lifecycle import (
            create_enhanced_mcp_server
        )
        
        mcp_server, lifecycle_manager = create_enhanced_mcp_server("test_pool")
        print("✅ Enhanced MCP server created")
        
        # Initialize lifecycle manager
        await lifecycle_manager.initialize()
        print("✅ Lifecycle manager initialized")
        
        # Test agent management
        start_result = await lifecycle_manager.mcp_server.call_tool(
            "start_agent",
            {"agent_id": "test_agent", "agent_type": "momentum"}
        )
        print(f"✅ Agent start test: {start_result}")
        
        # Get status
        status_result = await lifecycle_manager.mcp_server.call_tool(
            "get_agent_status",
            {}
        )
        print(f"✅ Status check: {len(status_result.get('agents', {}))} agents")
        
        # Shutdown lifecycle manager
        await lifecycle_manager.shutdown()
        print("✅ Lifecycle manager shut down")
        
    except ImportError:
        print("⚠️ Enhanced MCP lifecycle not available (optional)")
    except Exception as e:
        print(f"❌ Enhanced MCP lifecycle test failed: {e}")
        return False
    
    print("\n🎉 All tests passed! A2A integration is working correctly.")
    return True


async def test_neo4j_setup():
    """Test Neo4j database setup."""
    print("\n🗄️  Testing Neo4j Setup")
    print("="*30)
    
    try:
        from scripts.setup_neo4j import Neo4jDatabaseManager
        
        # Create database manager
        db_manager = Neo4jDatabaseManager(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="finagent123",
            database="finagent"
        )
        
        # Test connection
        connected = await db_manager.connect()
        if not connected:
            print("❌ Failed to connect to Neo4j (make sure it's running)")
            return False
        
        print("✅ Connected to Neo4j")
        
        # Test schema initialization
        schema_ok = await db_manager.initialize_schema()
        print(f"✅ Schema initialized: {schema_ok}")
        
        # Test health check
        health = await db_manager.health_check()
        print(f"✅ Health check: {health['status']}")
        
        # Test sample data
        sample_ok = await db_manager.create_sample_data()
        print(f"✅ Sample data created: {sample_ok}")
        
        # Close connection
        db_manager.close()
        print("✅ Database connection closed")
        
        return True
        
    except ImportError as e:
        print(f"❌ Neo4j setup imports failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Neo4j test failed: {e}")
        return False


def print_setup_instructions():
    """Print setup instructions for the test."""
    print("\n📋 Setup Instructions")
    print("="*40)
    print("1. Make sure Neo4j is running:")
    print("   docker run -d --name neo4j-finagent \\")
    print("     -p 7474:7474 -p 7687:7687 \\")
    print("     -e NEO4J_AUTH=neo4j/finagent123 \\")
    print("     neo4j:5.15")
    print()
    print("2. Start the memory agent (if available):")
    print("   python start_memory_agent.py")
    print()
    print("3. Install required dependencies:")
    print("   pip install neo4j httpx fastapi uvicorn")
    print()
    print("4. Run this test:")
    print("   python examples/quick_a2a_integration_test.py")
    print()


async def main():
    """Main test function."""
    print_setup_instructions()
    
    # Run Neo4j tests
    neo4j_ok = await test_neo4j_setup()
    
    # Run A2A integration tests
    a2a_ok = await test_a2a_integration()
    
    # Final result
    print("\n" + "="*60)
    if neo4j_ok and a2a_ok:
        print("🎊 ALL TESTS PASSED! Your A2A integration is ready!")
    else:
        print("⚠️  Some tests failed. Check the output above.")
        if not neo4j_ok:
            print("   - Neo4j setup issues")
        if not a2a_ok:
            print("   - A2A integration issues")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
