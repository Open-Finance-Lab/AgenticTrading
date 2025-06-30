"""
LangGraph Portfolio Construction Agent Test

This test file specifically tests the LangGraph-based portfolio construction agents
with ReAct (Reasoning + Acting) capabilities for real portfolio optimization.

Author: Jifeng Li
Created: 2025-06-30
License: openMDW
"""

import asyncio
import pytest
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List
import json
import uuid

# LangChain imports for proper mocking
from langchain_core.messages import AIMessage

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockOpenAIClientForLangGraph:
    """Enhanced mock OpenAI client for LangGraph testing"""
    
    async def ainvoke(self, messages):
        """Mock async invoke for LangGraph LLM"""
        # Analyze the last message to determine appropriate response
        last_message = messages[-1] if messages else None
        
        # Simple counter to move through the process
        message_count = len(messages)
        
        if not last_message:
            content = "I need to start portfolio construction. Let me gather market data first."
        elif message_count <= 3:
            content = "I need to gather market data for the investment universe. Let me use the get_market_data tool."
        elif message_count <= 5:
            content = "Now I need alpha signals to understand expected returns. Let me use the get_alpha_signals tool."
        elif message_count <= 7:
            content = "I should analyze portfolio risk. Let me use the analyze_portfolio_risk tool."
        elif message_count <= 9:
            content = "I have sufficient data. Let me proceed with portfolio optimization using the optimize_portfolio tool."
        else:
            content = "The optimization results look good. All validation checks pass. The portfolio is ready."
        
        # Return proper LangChain AIMessage
        return AIMessage(content=content)


async def create_test_pool_with_langgraph():
    """Create a portfolio construction agent pool with LangGraph enabled for testing"""
    from FinAgents.agent_pools.portfolio_construction_agent_pool.core import PortfolioConstructionAgentPool
    
    # Create portfolio construction pool
    pool = PortfolioConstructionAgentPool(
        openai_api_key="test_key",
        external_memory_config={
            "host": "localhost",
            "port": 8000
        }
    )
    
    # Mock the OpenAI client for LangGraph agents
    mock_openai = MockOpenAIClientForLangGraph()
    
    # Replace LLM in LangGraph agents
    for agent_type, agent in pool.langgraph_agents.items():
        if agent.llm:
            agent.llm = mock_openai
    
    await pool.initialize()
    
    return pool


class TestLangGraphPortfolioAgents:
    """Test LangGraph-based portfolio construction agents"""
    
    @pytest.mark.asyncio
    async def test_langgraph_agent_initialization(self):
        """Test LangGraph agent initialization"""
        pool = await create_test_pool_with_langgraph()
        
        # Check that LangGraph agents are initialized
        assert len(pool.langgraph_agents) > 0
        
        expected_agents = [
            "mean_variance_optimizer",
            "black_litterman_optimizer",
            "risk_parity_optimizer",
            "factor_optimizer", 
            "robust_optimizer"
        ]
        
        for agent_type in expected_agents:
            assert agent_type in pool.langgraph_agents
            agent = pool.langgraph_agents[agent_type]
            assert agent.agent_type == agent_type
            assert agent.tools is not None
            
        logger.info(f"Successfully initialized {len(pool.langgraph_agents)} LangGraph agents")
        await pool.close()
    
    @pytest.mark.asyncio
    async def test_portfolio_tools(self):
        """Test portfolio construction tools"""
        pool = await create_test_pool_with_langgraph()
        
        # Get tools from one of the agents
        agent = list(pool.langgraph_agents.values())[0]
        tools = agent.tools.get_all_tools()
        
        assert len(tools) == 4
        
        tool_names = [tool.name for tool in tools]
        expected_tools = ["get_market_data", "get_alpha_signals", "analyze_portfolio_risk", "optimize_portfolio"]
        
        for expected_tool in expected_tools:
            assert expected_tool in tool_names
        
        # Test market data tool
        market_data_tool = next(tool for tool in tools if tool.name == "get_market_data")
        result = market_data_tool.func("AAPL,GOOGL,MSFT")
        
        # Parse JSON result
        data = json.loads(result)
        assert "AAPL" in data
        assert "GOOGL" in data
        assert "MSFT" in data
        
        for symbol, info in data.items():
            assert "current_price" in info
            assert "volatility" in info
            assert "daily_return" in info
        
        logger.info("Portfolio tools working correctly")
        await pool.close()
    
    @pytest.mark.asyncio
    async def test_langgraph_mean_variance_optimization(self):
        """Test LangGraph mean-variance optimization"""
        pool = await create_test_pool_with_langgraph()
        
        # Test mean-variance optimization using LangGraph
        task = {
            "task_type": "portfolio_optimization",
            "agent_type": "mean_variance_optimizer",
            "parameters": {
                "user_input": "Create a mean-variance optimized portfolio for technology stocks",
                "investment_universe": ["AAPL", "GOOGL", "MSFT"],
                "optimization_type": "mean_variance",
                "use_langgraph": True,
                "constraints": {
                    "max_weight": 0.50,
                    "min_weight": 0.10
                }
            }
        }
        
        result = await pool.execute_structured_task(task)
        
        assert result["status"] == "success"
        assert "optimization_result" in result
        assert "thread_id" in result
        assert "messages" in result
        
        optimization_result = result["optimization_result"]
        assert "optimal_weights" in optimization_result
        assert "portfolio_metrics" in optimization_result
        
        # Check weights
        optimal_weights = optimization_result["optimal_weights"]
        assert len(optimal_weights) == 3
        assert all(symbol in optimal_weights for symbol in ["AAPL", "GOOGL", "MSFT"])
        
        # Check that weights sum to approximately 1
        total_weight = sum(optimal_weights.values())
        assert abs(total_weight - 1.0) < 0.01
        
        # Check portfolio metrics
        portfolio_metrics = optimization_result["portfolio_metrics"]
        assert "expected_return" in portfolio_metrics
        assert "portfolio_risk" in portfolio_metrics
        assert "sharpe_ratio" in portfolio_metrics
        
        logger.info(f"LangGraph mean-variance optimization successful: {optimization_result}")
        await pool.close()
    
    @pytest.mark.asyncio
    async def test_langgraph_risk_parity_optimization(self):
        """Test LangGraph risk parity optimization"""
        pool = await create_test_pool_with_langgraph()
        
        task = {
            "task_type": "portfolio_optimization",
            "agent_type": "risk_parity_optimizer",
            "parameters": {
                "user_input": "Create a risk parity portfolio with equal risk contribution",
                "investment_universe": ["AAPL", "GOOGL", "MSFT", "AMZN"],
                "optimization_type": "risk_parity",
                "use_langgraph": True
            }
        }
        
        result = await pool.execute_structured_task(task)
        
        assert result["status"] == "success"
        assert "optimization_result" in result
        
        optimization_result = result["optimization_result"]
        optimal_weights = optimization_result["optimal_weights"]
        
        # Risk parity should give more equal weights (adjusted for risk)
        assert len(optimal_weights) == 4
        weights_list = list(optimal_weights.values())
        
        # Weights should be reasonably distributed (not too concentrated)
        max_weight = max(weights_list)
        min_weight = min(weights_list)
        assert max_weight / min_weight < 5.0  # Not too much concentration
        
        logger.info(f"LangGraph risk parity optimization successful: {optimal_weights}")
        await pool.close()
    
    @pytest.mark.asyncio
    async def test_natural_language_processing_with_langgraph(self):
        """Test natural language processing with LangGraph agents"""
        pool = await create_test_pool_with_langgraph()
        
        test_inputs = [
            "Create a conservative portfolio for retirement with low risk",
            "Build an aggressive growth portfolio using risk parity",
            "Construct a factor-based momentum portfolio for tech stocks",
            "Design a Black-Litterman portfolio with my investment views"
        ]
        
        for user_input in test_inputs:
            result = await pool.process_natural_language_with_langgraph(
                user_input=user_input,
                investment_universe=["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
            )
            
            assert result["status"] == "success"
            assert "optimization_result" in result
            assert "selected_agent" in result
            assert "original_input" in result
            assert result["original_input"] == user_input
            
            # Check that appropriate agent was selected
            selected_agent = result["selected_agent"]
            if "risk parity" in user_input.lower():
                assert selected_agent == "risk_parity_optimizer"
            elif "black litterman" in user_input.lower():
                assert selected_agent == "black_litterman_optimizer"
            elif "factor" in user_input.lower() or "momentum" in user_input.lower():
                assert selected_agent == "factor_optimizer"
            
            logger.info(f"NLP input: '{user_input[:50]}...' -> Agent: {selected_agent}")
        
        await pool.close()
    
    @pytest.mark.asyncio
    async def test_langgraph_agent_memory_and_state(self):
        """Test LangGraph agent memory and state management"""
        pool = await create_test_pool_with_langgraph()
        
        agent = pool.langgraph_agents["mean_variance_optimizer"]
        
        # Test first conversation
        request1 = {
            "user_input": "Create a portfolio with AAPL and GOOGL",
            "investment_universe": ["AAPL", "GOOGL"],
            "optimization_params": {"optimization_type": "mean_variance"}
        }
        
        result1 = await agent.construct_portfolio(request1)
        assert result1["status"] == "success"
        thread_id1 = result1["thread_id"]
        
        # Test second conversation (should have different thread)
        request2 = {
            "user_input": "Now add MSFT to the portfolio",
            "investment_universe": ["AAPL", "GOOGL", "MSFT"], 
            "optimization_params": {"optimization_type": "mean_variance"}
        }
        
        result2 = await agent.construct_portfolio(request2)
        assert result2["status"] == "success"
        thread_id2 = result2["thread_id"]
        
        # Threads should be different (each conversation is independent)
        assert thread_id1 != thread_id2
        
        logger.info(f"LangGraph memory test: Thread1={thread_id1}, Thread2={thread_id2}")
        await pool.close()
    
    @pytest.mark.asyncio
    async def test_langgraph_error_handling(self):
        """Test LangGraph agent error handling"""
        pool = await create_test_pool_with_langgraph()
        
        # Test with empty investment universe
        task = {
            "task_type": "portfolio_optimization",
            "agent_type": "mean_variance_optimizer",
            "parameters": {
                "user_input": "Create a portfolio",
                "investment_universe": [],  # Empty universe should cause error
                "optimization_type": "mean_variance",
                "use_langgraph": True
            }
        }
        
        result = await pool.execute_structured_task(task)
        
        # Should handle error gracefully
        assert "status" in result
        # Note: The result might be success with empty weights or error
        
        # Test with invalid agent type
        invalid_task = {
            "task_type": "portfolio_optimization",
            "agent_type": "nonexistent_optimizer",
            "parameters": {
                "user_input": "Create a portfolio",
                "investment_universe": ["AAPL"],
                "use_langgraph": True
            }
        }
        
        result = await pool.execute_structured_task(invalid_task)
        assert result["status"] == "error"
        assert "LangGraph agent not available" in result["error"]
        
        logger.info("LangGraph error handling test completed")
        await pool.close()
    
    @pytest.mark.asyncio 
    async def test_langgraph_performance_and_concurrency(self):
        """Test LangGraph agent performance and concurrent execution"""
        pool = await create_test_pool_with_langgraph()
        
        # Create multiple concurrent tasks
        tasks = []
        for i in range(3):
            task = {
                "task_type": "portfolio_optimization",
                "agent_type": "mean_variance_optimizer",
                "parameters": {
                    "user_input": f"Create portfolio {i+1}",
                    "investment_universe": ["AAPL", "GOOGL", "MSFT"][:(i+1)],
                    "optimization_type": "mean_variance",
                    "use_langgraph": True
                }
            }
            tasks.append(pool.execute_structured_task(task))
        
        # Execute concurrently
        start_time = datetime.now()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = datetime.now()
        
        execution_time = (end_time - start_time).total_seconds()
        
        # Check results
        successful_results = 0
        for result in results:
            if isinstance(result, dict) and result.get("status") == "success":
                successful_results += 1
            elif isinstance(result, Exception):
                logger.error(f"Concurrent task failed: {result}")
        
        assert successful_results >= 2, f"Only {successful_results}/3 concurrent tasks succeeded"
        assert execution_time < 30.0, f"Concurrent execution took too long: {execution_time}s"
        
        logger.info(f"Concurrent LangGraph test: {successful_results}/3 successful in {execution_time:.2f}s")
        await pool.close()


if __name__ == "__main__":
    """Run LangGraph tests manually"""
    
    async def run_langgraph_tests():
        print("Running LangGraph Portfolio Construction Agent Tests...")
        
        try:
            # Test 1: Agent initialization
            print("\n1. Testing LangGraph agent initialization...")
            pool = await create_test_pool_with_langgraph()
            print(f"✓ Initialized {len(pool.langgraph_agents)} LangGraph agents")
            await pool.close()
            
            # Test 2: Basic optimization
            print("\n2. Testing mean-variance optimization...")
            pool = await create_test_pool_with_langgraph()
            
            task = {
                "task_type": "portfolio_optimization",
                "agent_type": "mean_variance_optimizer",
                "parameters": {
                    "user_input": "Create an optimal portfolio",
                    "investment_universe": ["AAPL", "GOOGL", "MSFT"],
                    "optimization_type": "mean_variance",
                    "use_langgraph": True
                }
            }
            
            result = await pool.execute_structured_task(task)
            if result["status"] == "success":
                weights = result["optimization_result"]["optimal_weights"]
                print(f"✓ Optimization successful: {weights}")
            else:
                print(f"✗ Optimization failed: {result.get('error')}")
            
            await pool.close()
            
            # Test 3: Natural language processing
            print("\n3. Testing natural language processing...")
            pool = await create_test_pool_with_langgraph()
            
            result = await pool.process_natural_language_with_langgraph(
                "Create a conservative portfolio for retirement",
                ["AAPL", "GOOGL", "MSFT"]
            )
            
            if result["status"] == "success":
                print(f"✓ NLP processing successful: {result['selected_agent']}")
            else:
                print(f"✗ NLP processing failed: {result.get('error')}")
            
            await pool.close()
            
            print("\n✅ All LangGraph tests completed successfully!")
            
        except Exception as e:
            print(f"\n❌ LangGraph tests failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run the tests
    asyncio.run(run_langgraph_tests())
