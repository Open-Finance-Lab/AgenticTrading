#!/usr/bin/env python3
"""
LangGraph Portfolio Construction Agent Demo

This script demonstrates the ReAct-style portfolio construction agents
using LangGraph for natural language processing and agent orchestration.
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the LangGraph portfolio construction agents
from FinAgents.agent_pools.portfolio_construction_agent_pool.core import (
    PortfolioConstructionAgentPool,
    PortfolioConstructionAgent,
    PortfolioTools
)


class MockOpenAIForDemo:
    """Mock OpenAI client for demo purposes"""
    
    def __init__(self):
        self.call_count = 0
    
    async def ainvoke(self, messages):
        """Mock async invoke for demonstration"""
        self.call_count += 1
        
        # Simple logic to progress through the portfolio construction process
        if self.call_count <= 2:
            content = "I need to gather market data for the investment universe. Let me use the get_market_data tool."
        elif self.call_count <= 4:
            content = "Now I need alpha signals to understand expected returns. Let me use the get_alpha_signals tool."
        elif self.call_count <= 6:
            content = "I should analyze portfolio risk. Let me use the analyze_portfolio_risk tool."
        elif self.call_count <= 8:
            content = "I have sufficient data. Let me proceed with portfolio optimization."
        else:
            content = "The optimization results look good. All validation checks pass. The portfolio is ready."
        
        # Return mock message
        from langchain_core.messages import AIMessage
        return AIMessage(content=content)


async def demo_langgraph_portfolio_construction():
    """Demonstrate LangGraph portfolio construction agents"""
    
    print("🚀 LangGraph Portfolio Construction Agent Demo")
    print("=" * 50)
    
    try:
        # Create portfolio construction agent pool
        print("\n1. Initializing Portfolio Construction Agent Pool...")
        pool = PortfolioConstructionAgentPool(
            openai_api_key="demo_key",  # Mock key for demo
            external_memory_config={
                "host": "localhost",
                "port": 8000
            },
            enable_real_time_monitoring=False
        )
        
        # Replace LLM with mock for demo
        mock_llm = MockOpenAIForDemo()
        for agent_type, agent in pool.langgraph_agents.items():
            if agent.llm:
                agent.llm = mock_llm
        
        await pool.initialize()
        print(f"✅ Pool initialized with {len(pool.langgraph_agents)} LangGraph agents")
        
        # Demo 1: Basic portfolio optimization
        print("\n2. Demo: Basic Mean-Variance Optimization")
        print("-" * 40)
        
        task = {
            "task_type": "portfolio_optimization",
            "agent_type": "mean_variance_optimizer",
            "parameters": {
                "user_input": "Create a mean-variance optimized portfolio for technology stocks",
                "investment_universe": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
                "optimization_type": "mean_variance",
                "use_langgraph": True,
                "constraints": {
                    "max_weight": 0.40,
                    "min_weight": 0.05
                }
            }
        }
        
        result = await pool.execute_structured_task(task)
        
        if result["status"] == "success":
            print("✅ Portfolio optimization successful!")
            optimization_result = result.get("optimization_result", {})
            optimal_weights = optimization_result.get("optimal_weights", {})
            
            print("\nOptimal Portfolio Weights:")
            for asset, weight in optimal_weights.items():
                print(f"  {asset}: {weight:.1%}")
            
            portfolio_metrics = optimization_result.get("portfolio_metrics", {})
            if portfolio_metrics:
                print(f"\nPortfolio Metrics:")
                print(f"  Expected Return: {portfolio_metrics.get('expected_return', 0):.2%}")
                print(f"  Portfolio Risk: {portfolio_metrics.get('portfolio_risk', 0):.2%}")
                print(f"  Sharpe Ratio: {portfolio_metrics.get('sharpe_ratio', 0):.2f}")
        else:
            print(f"❌ Portfolio optimization failed: {result.get('error')}")
        
        # Demo 2: Natural language processing
        print("\n3. Demo: Natural Language Portfolio Requests")
        print("-" * 40)
        
        natural_language_requests = [
            "Create a conservative portfolio for retirement with low risk",
            "Build an aggressive growth portfolio using risk parity",
            "Construct a factor-based momentum portfolio for tech stocks"
        ]
        
        for i, user_request in enumerate(natural_language_requests, 1):
            print(f"\nRequest {i}: {user_request}")
            
            # Reset mock counter for each request
            mock_llm.call_count = 0
            
            result = await pool.process_natural_language_with_langgraph(
                user_input=user_request,
                investment_universe=["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA"]
            )
            
            if result["status"] == "success":
                print(f"✅ Selected agent: {result.get('selected_agent')}")
                optimization_result = result.get("optimization_result", {})
                optimal_weights = optimization_result.get("optimal_weights", {})
                
                print("Portfolio weights:")
                for asset, weight in optimal_weights.items():
                    print(f"  {asset}: {weight:.1%}")
            else:
                print(f"❌ Request failed: {result.get('error')}")
        
        # Demo 3: Portfolio tools demonstration
        print("\n4. Demo: Portfolio Construction Tools")
        print("-" * 40)
        
        tools = PortfolioTools(pool)
        
        # Demo market data tool
        market_data_tool = tools.create_market_data_tool()
        market_data = market_data_tool.func("AAPL,GOOGL,MSFT")
        print("\nMarket Data Tool Result (sample):")
        market_data_json = json.loads(market_data)
        for symbol in list(market_data_json.keys())[:2]:  # Show first 2 symbols
            data = market_data_json[symbol]
            print(f"  {symbol}: Price=${data['current_price']:.2f}, Vol={data['volatility']:.1%}")
        
        # Demo alpha signals tool
        alpha_tool = tools.create_alpha_signals_tool()
        alpha_signals = alpha_tool.func("AAPL,GOOGL")
        print("\nAlpha Signals Tool Result (sample):")
        alpha_json = json.loads(alpha_signals)
        for symbol, signal in alpha_json.items():
            print(f"  {symbol}: {signal['signal_type']} (confidence: {signal['confidence_score']:.1%})")
        
        # Demo optimization tool
        optimization_tool = tools.create_optimization_tool()
        optimization_result = optimization_tool.func("equal_weight", "AAPL,GOOGL,MSFT", '{"max_weight": 0.4}')
        print("\nOptimization Tool Result:")
        opt_json = json.loads(optimization_result)
        for symbol, weight in opt_json["optimal_weights"].items():
            print(f"  {symbol}: {weight:.1%}")
        
        print("\n5. Summary")
        print("-" * 40)
        print("✅ LangGraph Portfolio Construction Agents successfully demonstrated:")
        print("  • ReAct-style reasoning and tool usage")
        print("  • Natural language processing and routing")
        print("  • Multi-step portfolio optimization workflow")
        print("  • Integration with mock market data and signals")
        print("  • Validation and error handling")
        
        await pool.close()
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")


async def demo_individual_agent():
    """Demonstrate individual LangGraph agent functionality"""
    
    print("\n🔧 Individual LangGraph Agent Demo")
    print("=" * 40)
    
    try:
        # Create a minimal agent pool for demo
        pool = PortfolioConstructionAgentPool(
            openai_api_key="demo_key",
            external_memory_config={}
        )
        
        # Create individual agent
        agent = PortfolioConstructionAgent(pool, "mean_variance_optimizer")
        
        # Replace LLM with mock
        mock_llm = MockOpenAIForDemo()
        agent.llm = mock_llm
        agent._create_agent_graph()
        
        print("✅ Individual LangGraph agent created")
        
        # Test portfolio construction
        request = {
            "user_input": "Create an optimal portfolio for tech stocks",
            "investment_universe": ["AAPL", "GOOGL", "MSFT"],
            "optimization_params": {
                "optimization_type": "mean_variance",
                "constraints": {"max_weight": 0.5, "min_weight": 0.1},
                "objective": "maximize_sharpe"
            }
        }
        
        print("\nTesting portfolio construction...")
        result = await agent.construct_portfolio(request)
        
        if result["status"] == "success":
            print("✅ Individual agent portfolio construction successful!")
            print(f"Thread ID: {result.get('thread_id')}")
            print(f"Agent Type: {result.get('agent_type')}")
        else:
            print(f"❌ Individual agent failed: {result.get('error')}")
        
    except Exception as e:
        logger.error(f"Individual agent demo failed: {e}")
        print(f"❌ Individual agent demo failed: {e}")


if __name__ == "__main__":
    print("🎯 Starting LangGraph Portfolio Construction Demo")
    
    # Run main demo
    asyncio.run(demo_langgraph_portfolio_construction())
    
    # Run individual agent demo
    asyncio.run(demo_individual_agent())
    
    print("\n🎉 Demo completed successfully!")
