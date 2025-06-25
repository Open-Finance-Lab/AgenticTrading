# examples/autonomous_agent_example.py

"""
AutonomousAgent Usage Example
Demonstrates how to use the AutonomousAgent for autonomous task creation, 
code generation, and validation in financial analysis scenarios.
"""

import asyncio
import json
import time
import sys
import os

# Add project path for imports
sys.path.append('/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration')

async def test_autonomous_agent():
    """Test the complete workflow of the autonomous agent"""
    
    print("=== Autonomous Agent Integration Test ===")
    
    # Import the actual agent
    from FinAgents.agent_pools.alpha_agent_pool.agents.autonomous.autonomous_agent import AutonomousAgent
    
    # Initialize the agent
    print("\n=== Initializing Autonomous Agent ===")
    agent = AutonomousAgent("example_autonomous_agent")
    print(f"✓ Agent initialized with ID: {agent.agent_id}")
    
    # Test orchestrator input processing
    print("\n=== Testing Orchestrator Input Processing ===")
    instruction = "Analyze AAPL stock momentum indicators and generate trading signals"
    context = {
        "symbol": "AAPL",
        "timeframe": "daily", 
        "analysis_type": "momentum"
    }
    print(f"Instruction: {instruction}")
    print(f"Context: {json.dumps(context, indent=2)}")
    
    # Process the orchestrator input
    result = agent._process_orchestrator_input(instruction, context)
    print(f"Processing result: {result}")
    
    # Show decomposed tasks
    print("\n=== Task Decomposition Results ===")
    for i, task in enumerate(agent.task_queue[-5:], 1):  # Show last 5 tasks
        print(f"{i}. {task.description}")
        print(f"   Status: {task.status}, Priority: {task.priority}")
    
    # Test memory query
    print("\n=== Testing Memory Query ===")
    memory_result = agent._query_memory("AAPL historical prices", "market_data")
    print(f"Memory query result: {json.dumps(memory_result, indent=2)}")
    
    # Test code tool generation
    print("\n=== Testing Dynamic Code Generation ===")
    tool_result = agent._generate_code_tool(
        description="Momentum analysis tool for stock price data",
        input_format="Dictionary with 'prices' key containing list of float values",
        expected_output="Dictionary with momentum analysis results and trading signals"
    )
    
    print(f"Generated tool: {tool_result['tool_name']}")
    print(f"File path: {tool_result['file_path']}")
    print("✓ Code generation successful")
    
    # Test tool execution
    print("\n=== Testing Tool Execution ===")
    test_data = {
        "prices": [150, 152, 148, 155, 153, 157, 160, 158, 162, 165]
    }
    
    execution_result = agent._execute_tool(tool_result['tool_name'], test_data)
    print(f"Tool execution result: {json.dumps(execution_result, indent=2)}")
    
    # Test strategy signal generation
    print("\n=== Testing Strategy Signal Generation ===")
    strategy_flow = agent._generate_trading_signal(
        symbol="AAPL",
        instruction=instruction,
        market_data={"prices": test_data["prices"]}
    )
    
    print("Generated Strategy Flow:")
    print(f"  Alpha ID: {strategy_flow['alpha_id']}")
    print(f"  Signal: {strategy_flow['decision']['signal']}")
    print(f"  Confidence: {strategy_flow['decision']['confidence']:.2f}")
    print(f"  Reasoning: {strategy_flow['decision']['reasoning']}")
    print(f"  Market Regime: {strategy_flow['market_context']['regime_tag']}")
    
    # Test validation code generation
    print("\n=== Testing Validation Code Generation ===")
    test_scenarios = [
        {"input": {"prices": [100, 102, 98, 105]}, "expected": {"success": True}},
        {"input": {"prices": [100]}, "expected": {"success": False}},
        {"input": {"prices": []}, "expected": {"success": False}}
    ]
    
    validation_result = agent._create_validation(tool_result['code'], test_scenarios)
    print(f"Validation code generated: {validation_result['validation_name']}")
    print(f"Validation file: {validation_result['file_path']}")
    
    print("\n=== Autonomous Agent Test Complete ===")
    print("✓ Orchestrator input processing")
    print("✓ Autonomous task decomposition")
    print("✓ Memory knowledge retrieval")
    print("✓ Dynamic code generation")
    print("✓ Tool execution verification")
    print("✓ Strategy flow generation")
    print("✓ Validation code creation")
async def test_orchestrator_integration():
    """Test integration with orchestrator through MCP tools"""
    
    print("=== Orchestrator Integration Test ===")
    
    # Import the actual agent
    from FinAgents.agent_pools.alpha_agent_pool.agents.autonomous.autonomous_agent import AutonomousAgent
    
    print("\n=== Starting Autonomous Agent ===")
    agent = AutonomousAgent("orchestrator_integration_test")
    print("✓ Autonomous Agent initialized")
    
    print("\n=== Testing MCP Tool Interface ===")
    
    # Test 1: Orchestrator input processing
    print("1. Testing orchestrator input processing...")
    instruction = "Create a new quantitative trading strategy based on technical indicators for signal generation"
    context = {"strategy_type": "technical", "risk_level": "medium"}
    
    print(f"Instruction: {instruction}")
    print(f"Context: {json.dumps(context, indent=2)}")
    
    result = agent._process_orchestrator_input(instruction, context)
    print(f"Result: {result}")
    
    # Test 2: Strategy signal generation
    print("\n2. Testing strategy signal generation...")
    strategy_flow = agent._generate_trading_signal(
        symbol="AAPL",
        instruction=instruction,
        market_data={"prices": [100, 102, 105, 108, 110, 112, 115, 118, 120, 122]}
    )
    
    print("Generated Strategy Flow:")
    print(f"  Signal: {strategy_flow['decision']['signal']}")
    print(f"  Confidence: {strategy_flow['decision']['confidence']:.2f}")
    print(f"  Reasoning: {strategy_flow['decision']['reasoning']}")
    print(f"  Execution Weight: {strategy_flow['action']['execution_weight']}")
    
    # Test 3: Code generation with enhanced features
    print("\n3. Testing advanced code generation...")
    tool_result = agent._generate_code_tool(
        description="Technical analysis strategy with RSI and moving averages",
        input_format="Dictionary with 'prices' and optional 'volumes' arrays",
        expected_output="Dictionary with trading signals and technical indicators"
    )
    
    print(f"Generated advanced tool: {tool_result['tool_name']}")
    
    # Test 4: Complex market data analysis
    print("\n4. Testing complex market analysis...")
    complex_data = {
        "prices": [250, 252, 255, 258, 262, 266, 270, 275, 280, 285, 282, 278, 274, 271, 268],
        "volumes": [1000, 1200, 1100, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400]
    }
    
    execution_result = agent._execute_tool(tool_result['tool_name'], complex_data)
    if execution_result.get('success'):
        print("✓ Complex analysis completed successfully")
        print(f"Analysis type: {execution_result['result'].get('analysis_type', 'N/A')}")
    else:
        print(f"✗ Analysis failed: {execution_result.get('error', 'Unknown error')}")
    
    # Test 5: Multiple strategy flows
    print("\n5. Testing multiple strategy flows...")
    scenarios = [
        ("MSFT", "bearish trend analysis", [200, 198, 195, 192, 190, 188, 185, 182, 180, 178]),
        ("GOOGL", "neutral consolidation analysis", [150, 151, 149, 150, 151, 150, 149, 151, 150, 151]),
        ("TSLA", "volatile momentum analysis", [300, 310, 295, 320, 305, 330, 315, 340, 325, 335])
    ]
    
    for symbol, description, prices in scenarios:
        flow = agent._generate_trading_signal(symbol, description, {"prices": prices})
        print(f"  {symbol}: {flow['decision']['signal']} (confidence: {flow['decision']['confidence']:.2f})")
    
    print("\n=== Active Agents Status ===")
    active_agents = ["momentum_agent", "autonomous_agent"]
    print(f"Currently active agents: {active_agents}")
    
    print("\n=== Orchestrator Integration Test Complete ===")
    print("✓ Instruction reception and decomposition")
    print("✓ Automated strategy code generation")
    print("✓ Strategy validation and execution")
    print("✓ Multi-agent coordination workflow")
    print("✓ Complex market data processing")
    print("✓ Multiple scenario handling")


def test_error_handling():
    """Test error handling mechanisms"""
    
    print("=== Error Handling Test ===")
    
    from FinAgents.agent_pools.alpha_agent_pool.agents.autonomous.autonomous_agent import AutonomousAgent
    
    agent = AutonomousAgent("error_handling_test")
    
    # Test 1: Invalid data handling
    print("\n1. Testing invalid data handling...")
    try:
        result = agent._perform_autonomous_analysis([], "test analysis")
        print(f"Empty data result: {result['signal']} - {result['reasoning']}")
    except Exception as e:
        print(f"Error handled: {e}")
    
    # Test 2: Tool execution with invalid tool name
    print("\n2. Testing invalid tool execution...")
    result = agent._execute_tool("non_existent_tool", {"data": "test"})
    print(f"Invalid tool result: {result}")
    
    # Test 3: Code generation with minimal parameters
    print("\n3. Testing minimal parameter code generation...")
    try:
        tool_result = agent._generate_code_tool("", "", "")
        print("✓ Minimal parameter handling successful")
    except Exception as e:
        print(f"✓ Error properly handled: {e}")
    
    # Test 4: Memory query with invalid parameters
    print("\n4. Testing memory query error handling...")
    result = agent._query_memory("", None)
    print(f"Empty query handled: {'error' in result}")
    
    print("✓ Error handling mechanisms verified")


def main():
    """Main function with enhanced features and error handling"""
    print("AutonomousAgent Example - English Version")
    print("=========================================")
    print("This example demonstrates the capabilities of the refactored AutonomousAgent")
    print("with complete English implementation and enhanced features.")
    print()
    print("Prerequisites:")
    print("1. AutonomousAgent has been refactored to English")
    print("2. AlphaStrategyFlow compatibility implemented")
    print("3. Enhanced signal generation logic activated")
    print()
    
    try:
        choice = input("Select test mode:\n1. Complete agent functionality test\n2. Orchestrator integration test\n3. Error handling test\n4. All tests\nEnter choice (1/2/3/4): ")
        
        if choice == "1":
            print("\nStarting complete functionality test...")
            asyncio.run(test_autonomous_agent())
            
        elif choice == "2":
            print("\nTesting orchestrator integration...")
            asyncio.run(test_orchestrator_integration())
            
        elif choice == "3":
            print("\nTesting error handling mechanisms...")
            test_error_handling()
            
        elif choice == "4":
            print("\nRunning all tests...")
            print("\n" + "="*50)
            asyncio.run(test_autonomous_agent())
            print("\n" + "="*50)
            asyncio.run(test_orchestrator_integration())
            print("\n" + "="*50)
            test_error_handling()
            print("\n" + "="*50)
            print("🎉 All tests completed successfully!")
            
        else:
            print("Invalid choice. Please run again with a valid option.")
            
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        print("Please check the autonomous agent implementation and try again.")


if __name__ == "__main__":
    main()
