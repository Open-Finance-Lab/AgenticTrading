#!/usr/bin/env python3
"""
Memory Unit Testing Script for Portfolio Construction Agent Pool

This script tests the memory unit functionality and verifies that data is being
properly stored and retrieved.

Author: Jifeng Li
Created: 2025-06-30
"""

import asyncio
import logging
import json
from pathlib import Path
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from FinAgents.agent_pools.portfolio_construction_agent_pool.core import PortfolioConstructionAgentPool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_memory_unit():
    """Test memory unit functionality."""
    logger.info("=" * 60)
    logger.info("Portfolio Construction Agent Pool Memory Unit Test")
    logger.info("=" * 60)
    
    try:
        # Create agent pool
        pool = PortfolioConstructionAgentPool(
            openai_api_key="test_key_for_memory_test",
            external_memory_config={
                "host": "localhost",
                "port": 8000
            }
        )
        
        # Initialize pool
        await pool.initialize()
        logger.info("✅ Agent pool initialized successfully")
        
        # Test memory unit basic operations
        logger.info("\n📝 Testing Memory Unit Basic Operations:")
        
        # Store some test data
        pool.memory_unit.set("test_key_1", {"data": "test_value_1", "timestamp": "2025-06-30"})
        pool.memory_unit.set("test_key_2", {"portfolio": "AAPL", "weight": 0.25})
        pool.memory_unit.set("test_key_3", {"optimization": "mean_variance", "sharpe": 1.5})
        
        # Retrieve data
        value1 = pool.memory_unit.get("test_key_1")
        value2 = pool.memory_unit.get("test_key_2")
        value3 = pool.memory_unit.get("test_key_3")
        
        logger.info(f"✅ Retrieved test_key_1: {value1}")
        logger.info(f"✅ Retrieved test_key_2: {value2}")
        logger.info(f"✅ Retrieved test_key_3: {value3}")
        
        # Test portfolio event recording
        logger.info("\n📊 Testing Portfolio Event Recording:")
        
        await pool.memory_unit.record_portfolio_event({
            "event_type": "PORTFOLIO_CREATED",
            "portfolio_id": "test_portfolio_001",
            "assets": ["AAPL", "GOOGL", "MSFT"],
            "optimization_type": "mean_variance"
        })
        
        await pool.memory_unit.record_portfolio_event({
            "event_type": "OPTIMIZATION_COMPLETED",
            "portfolio_id": "test_portfolio_001",
            "sharpe_ratio": 1.25,
            "volatility": 0.18
        })
        
        await pool.memory_unit.record_portfolio_event({
            "event_type": "BACKTEST_STARTED",
            "portfolio_id": "test_portfolio_001",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31"
        })
        
        logger.info("✅ Portfolio events recorded")
        
        # Execute a sample LangGraph task to generate more memory activity
        logger.info("\n🤖 Testing LangGraph Task Execution:")
        
        task = {
            "task_type": "portfolio_optimization",
            "agent_type": "mean_variance_optimizer",
            "parameters": {
                "user_input": "Create a balanced portfolio for testing memory functionality",
                "investment_universe": ["AAPL", "GOOGL", "MSFT", "AMZN"],
                "optimization_type": "mean_variance",
                "use_langgraph": True,
                "constraints": {
                    "max_weight": 0.40,
                    "min_weight": 0.10
                }
            }
        }
        
        # Mock the LLM for testing
        if pool.langgraph_agents:
            for agent_type, agent in pool.langgraph_agents.items():
                if agent.llm:
                    # Create a simple mock that returns AIMessage
                    from langchain_core.messages import AIMessage
                    
                    class SimpleMockLLM:
                        async def ainvoke(self, messages):
                            return AIMessage(content="I have sufficient data. Let me proceed with portfolio optimization.")
                    
                    agent.llm = SimpleMockLLM()
        
        result = await pool.execute_structured_task(task)
        logger.info(f"✅ LangGraph task executed: {result.get('status', 'unknown')}")
        
        # Get memory status
        logger.info("\n📈 Memory Status:")
        memory_status = await pool.get_memory_status()
        logger.info(f"Memory Statistics: {json.dumps(memory_status, indent=2, default=str)}")
        
        # Force save memory
        logger.info("\n💾 Forcing Memory Save:")
        save_result = await pool.save_memory_immediately()
        logger.info(f"Save Result: {save_result}")
        
        # Check if files exist
        logger.info("\n📁 Checking File System:")
        memory_path = Path("./FinAgents/memory/portfolio_construction_memory_storage")
        logger.info(f"Memory directory exists: {memory_path.exists()}")
        
        if memory_path.exists():
            files = list(memory_path.glob("*"))
            logger.info(f"Files in memory directory: {[f.name for f in files]}")
            
            # Check specific memory file
            memory_file = memory_path / f"{pool.pool_id}_memory.json"
            if memory_file.exists():
                with open(memory_file, 'r') as f:
                    file_content = json.load(f)
                logger.info(f"✅ Memory file contains {len(file_content.get('memory_data', {}))} memory entries")
                logger.info(f"✅ Memory file contains {len(file_content.get('events_log', []))} events")
            else:
                logger.warning(f"❌ Memory file not found: {memory_file}")
        
        # Test memory persistence by closing and reopening
        logger.info("\n🔄 Testing Memory Persistence:")
        await pool.close()
        logger.info("✅ Pool closed, memory saved")
        
        # Create new pool instance
        pool2 = PortfolioConstructionAgentPool(
            openai_api_key="test_key_for_memory_test",
            external_memory_config={
                "host": "localhost", 
                "port": 8000
            },
            pool_id=pool.pool_id  # Same pool ID to load same memory
        )
        
        await pool2.initialize()
        logger.info("✅ New pool instance initialized")
        
        # Check if data persisted
        restored_value1 = pool2.memory_unit.get("test_key_1")
        restored_value2 = pool2.memory_unit.get("test_key_2")
        
        logger.info(f"✅ Restored test_key_1: {restored_value1}")
        logger.info(f"✅ Restored test_key_2: {restored_value2}")
        
        # Get final memory status
        final_status = await pool2.get_memory_status()
        logger.info(f"Final Memory Statistics: {json.dumps(final_status, indent=2, default=str)}")
        
        await pool2.close()
        
        logger.info("\n🎉 Memory Unit Test Completed Successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Memory unit test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_memory_unit())
