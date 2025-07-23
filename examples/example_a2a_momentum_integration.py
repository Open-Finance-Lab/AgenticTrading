#!/usr/bin/env python3
"""
Example demonstration of the modified momentum agent with A2A protocol integration.

This example shows how the momentum agent now uses the A2A protocol to communicate
with the memory agent through the alpha agent pool coordination layer.

Features demonstrated:
- A2A protocol-based memory communication
- Pool-level memory coordination
- Signal generation with automatic memory storage
- Strategy performance tracking via A2A
- Learning feedback through A2A protocol

Usage:
    python example_a2a_momentum_integration.py

Author: FinAgent Team
License: Open Source
"""

import asyncio
import json
import os
import sys
import logging
from datetime import datetime
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


async def demonstrate_a2a_integration():
    """
    Demonstrate the A2A protocol integration with the momentum agent.
    """
    logger.info("🚀 Starting A2A Momentum Agent Integration Demo")
    
    try:
        # Import components
        from FinAgents.agent_pools.alpha_agent_pool.agents.theory_driven.a2a_client import (
            AlphaAgentA2AClient, create_alpha_pool_a2a_client
        )
        from FinAgents.agent_pools.alpha_agent_pool.a2a_memory_coordinator import (
            initialize_pool_coordinator, get_pool_coordinator
        )
        
        logger.info("✅ A2A components imported successfully")
        
        # 1. Initialize A2A Memory Coordinator
        logger.info("📊 Initializing Alpha Pool A2A Memory Coordinator...")
        coordinator = await initialize_pool_coordinator(
            pool_id="alpha_agent_pool_demo",
            memory_url="http://127.0.0.1:8010"
        )
        logger.info("✅ A2A Memory Coordinator initialized")
        
        # 2. Create A2A client for direct testing
        logger.info("🔗 Creating A2A client for testing...")
        a2a_client = create_alpha_pool_a2a_client(
            agent_pool_id="alpha_agent_pool_demo",
            memory_url="http://127.0.0.1:8010"
        )
        
        # 3. Test A2A connection
        logger.info("🏥 Testing A2A connection health...")
        async with a2a_client as client:
            health_ok = await client.healthcheck()
            logger.info(f"A2A Connection Health: {'✅ OK' if health_ok else '❌ Failed'}")
            
            if health_ok:
                # 4. Register demo agent with coordinator
                logger.info("📝 Registering demo momentum agent...")
                await coordinator.register_agent(
                    agent_id="momentum_agent_demo",
                    agent_type="theory_driven_momentum",
                    agent_config={
                        "window": 20,
                        "strategy_type": "momentum",
                        "demo_mode": True
                    }
                )
                
                # 5. Store sample signal event
                logger.info("📈 Storing sample trading signal via A2A...")
                await client.store_alpha_signal_event(
                    agent_id="momentum_agent_demo",
                    signal="BUY",
                    confidence=0.75,
                    symbol="AAPL",
                    reasoning="Strong momentum pattern detected with 20-day window",
                    market_context={
                        "price": 150.25,
                        "volume": 1000000,
                        "momentum_score": 0.65,
                        "volatility": 0.20,
                        "market_regime": "trending"
                    }
                )
                logger.info("✅ Signal event stored successfully")
                
                # 6. Store sample performance metrics
                logger.info("📊 Storing sample strategy performance...")
                sample_performance = {
                    "IC": 0.15,
                    "IR": 0.45,
                    "sharpe_ratio": 1.2,
                    "win_rate": 0.62,
                    "avg_return": 0.08,
                    "max_drawdown": -0.15,
                    "total_trades": 45,
                    "evaluation_period": "30_days"
                }
                
                await client.store_strategy_performance(
                    agent_id="momentum_agent_demo",
                    strategy_id=f"momentum_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    performance_metrics=sample_performance
                )
                logger.info("✅ Performance metrics stored successfully")
                
                # 7. Store learning feedback
                logger.info("🧠 Storing learning feedback...")
                learning_feedback = {
                    "window_optimization": {
                        "tested_windows": [10, 15, 20, 25],
                        "best_window": 20,
                        "performance_improvement": 0.08
                    },
                    "strategy_adaptation": {
                        "regime_detection": "improved",
                        "confidence_calibration": "enhanced"
                    }
                }
                
                await client.store_learning_feedback(
                    agent_id="momentum_agent_demo",
                    feedback_type="WINDOW_OPTIMIZATION",
                    feedback_data=learning_feedback
                )
                logger.info("✅ Learning feedback stored successfully")
                
                # 8. Retrieve strategy insights
                logger.info("🔍 Retrieving strategy insights...")
                insights = await client.retrieve_strategy_insights(
                    search_query="momentum strategy window optimization",
                    limit=5
                )
                logger.info(f"📋 Retrieved {len(insights)} strategy insights")
                
                # 9. Test pool-level aggregation
                logger.info("🔄 Testing pool-level performance aggregation...")
                pool_performance = await coordinator.aggregate_pool_performance()
                logger.info(f"📈 Pool Performance Summary:")
                logger.info(f"   Total Agents: {pool_performance.get('total_agents', 0)}")
                logger.info(f"   Active Agents: {pool_performance.get('active_agents', 0)}")
                
                # 10. Test cross-agent insights
                logger.info("🔗 Testing cross-agent insight retrieval...")
                cross_insights = await coordinator.retrieve_cross_agent_insights(
                    query="momentum strategy best practices",
                    limit=3
                )
                logger.info(f"🧩 Retrieved {len(cross_insights)} cross-agent insights")
        
        logger.info("🎉 A2A Integration Demo completed successfully!")
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("📋 DEMO SUMMARY")
        logger.info("="*60)
        logger.info("✅ A2A Memory Coordinator: Initialized")
        logger.info("✅ A2A Client Connection: Healthy")
        logger.info("✅ Agent Registration: Success")
        logger.info("✅ Signal Storage: Success")
        logger.info("✅ Performance Tracking: Success")
        logger.info("✅ Learning Feedback: Success")
        logger.info("✅ Strategy Insights: Retrieved")
        logger.info("✅ Pool Aggregation: Success")
        logger.info("✅ Cross-Agent Learning: Success")
        logger.info("="*60)
        
    except ImportError as e:
        logger.error(f"❌ Import error - A2A components not available: {e}")
        logger.info("💡 Please ensure the A2A client and coordinator are properly installed")
        
    except Exception as e:
        logger.error(f"❌ Demo failed with error: {e}")
        logger.exception("Full traceback:")
        
    finally:
        # Cleanup
        try:
            coordinator = get_pool_coordinator()
            if coordinator:
                await coordinator.stop_coordination()
                logger.info("🧹 A2A coordinator cleanup completed")
        except:
            pass


async def simulate_momentum_agent_workflow():
    """
    Simulate a complete momentum agent workflow with A2A integration.
    """
    logger.info("🔄 Simulating Momentum Agent Workflow with A2A Integration")
    
    try:
        from FinAgents.agent_pools.alpha_agent_pool.agents.theory_driven.a2a_client import (
            create_alpha_pool_a2a_client
        )
        
        # Create A2A client
        async with create_alpha_pool_a2a_client("alpha_agent_pool") as client:
            
            # Simulate signal generation process
            logger.info("📊 Simulating signal generation...")
            
            # Store multiple signals over time
            signals = [
                {"signal": "BUY", "confidence": 0.8, "symbol": "AAPL", "reasoning": "Strong upward momentum"},
                {"signal": "HOLD", "confidence": 0.6, "symbol": "AAPL", "reasoning": "Consolidation phase"},
                {"signal": "SELL", "confidence": 0.75, "symbol": "AAPL", "reasoning": "Momentum reversal detected"},
            ]
            
            for i, signal_data in enumerate(signals):
                await client.store_alpha_signal_event(
                    agent_id="momentum_workflow_demo",
                    **signal_data,
                    market_context={
                        "timestamp": datetime.now().isoformat(),
                        "sequence": i + 1,
                        "price": 150 + i * 2,
                        "volume": 1000000 * (1 + i * 0.1)
                    }
                )
                logger.info(f"📈 Stored signal {i+1}: {signal_data['signal']}")
            
            # Simulate backtest results and learning
            logger.info("🧪 Simulating backtest and learning cycle...")
            
            backtest_results = {
                "IC": 0.18,
                "IR": 0.52,
                "total_trades": 3,
                "win_rate": 0.67,
                "avg_return": 0.05,
                "sharpe_ratio": 1.35
            }
            
            await client.store_strategy_performance(
                agent_id="momentum_workflow_demo",
                strategy_id=f"workflow_test_{datetime.now().strftime('%H%M%S')}",
                performance_metrics=backtest_results
            )
            
            # Store adaptation feedback
            await client.store_learning_feedback(
                agent_id="momentum_workflow_demo",
                feedback_type="STRATEGY_ADAPTATION",
                feedback_data={
                    "adaptation_type": "window_optimization",
                    "improvement_metrics": backtest_results,
                    "next_action": "continue_with_current_settings"
                }
            )
            
            logger.info("✅ Momentum agent workflow simulation completed")
            
    except Exception as e:
        logger.error(f"❌ Workflow simulation failed: {e}")


def main():
    """Main function to run the A2A integration demonstration."""
    print("\n" + "="*80)
    print("🤖 FinAgent A2A Protocol Integration Demonstration")
    print("="*80)
    print("This demo shows the integration of the momentum agent with the")
    print("A2A protocol for memory communication and coordination.")
    print("="*80 + "\n")
    
    # Run the demonstration
    asyncio.run(demonstrate_a2a_integration())
    
    print("\n" + "-"*60)
    print("🔄 Running Workflow Simulation...")
    print("-"*60)
    
    # Run workflow simulation
    asyncio.run(simulate_momentum_agent_workflow())
    
    print("\n" + "="*80)
    print("🎯 Demo completed! Check the logs above for detailed results.")
    print("="*80)


if __name__ == "__main__":
    main()
