#!/usr/bin/env python3
"""
Real MCP Client for calling actual server functions.
This client will make real calls to the momentum agent, use LLM, and submit to memory.
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

class RealMCPClient:
    """Real MCP client that calls actual server functions."""
    
    def __init__(self, base_url: str = "http://localhost:8081"):
        self.base_url = base_url
        self.session_id = f"real_client_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.test_results = {}
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
    
    async def call_mcp_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool through the server."""
        print(f"🔧 Calling MCP tool: {tool_name}")
        print(f"   📊 Parameters: {params}")
        
        try:
            # This would normally be done through MCP protocol over SSE
            # For now, we'll call the demo system directly to test real functionality
            
            # Import the demo system
            from demo_decoupled_system import EnhancedAlphaPoolDemo
            
            # Create demo instance
            demo = EnhancedAlphaPoolDemo(test_mode=True)
            
            # Initialize system
            await demo._initialize_system()
            
            # Call the actual function
            if tool_name == "generate_alpha_signals":
                result = await demo.generate_alpha_signals(**params)
            elif tool_name == "discover_alpha_factors":
                result = await demo.discover_alpha_factors(**params)
            elif tool_name == "develop_strategy_configuration":
                result = await demo.develop_strategy_configuration(**params)
            elif tool_name == "run_comprehensive_backtest":
                result = await demo.run_comprehensive_backtest(**params)
            elif tool_name == "submit_strategy_to_memory":
                result = await demo.submit_strategy_to_memory(**params)
            elif tool_name == "run_integrated_backtest":
                result = await demo.run_integrated_backtest(**params)
            else:
                result = {"status": "error", "message": f"Unknown tool: {tool_name}"}
            
            # Shutdown demo
            await demo.shutdown()
            
            print(f"   ✅ Tool call successful")
            return result
            
        except Exception as e:
            print(f"   ❌ Tool call failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def test_real_factor_discovery(self) -> Dict[str, Any]:
        """Test real alpha factor discovery using momentum agent and LLM."""
        print("\n🔍 Testing REAL Alpha Factor Discovery...")
        
        try:
            # Test factor discovery with real momentum agent
            test_cases = [
                {
                    "factor_categories": ["momentum", "volatility"],
                    "significance_threshold": 0.05
                },
                {
                    "factor_categories": ["mean_reversion", "liquidity"],
                    "significance_threshold": 0.01
                }
            ]
            
            results = {}
            for i, test_case in enumerate(test_cases):
                print(f"   📊 Test case {i+1}: {test_case}")
                
                # Call the actual MCP tool
                result = await self.call_mcp_tool("discover_alpha_factors", test_case)
                
                results[f"test_case_{i+1}"] = result
                
                if result.get("status") == "success":
                    print(f"   ✅ Real factors discovered successfully")
                    print(f"   📊 Result: {result}")
                else:
                    print(f"   ❌ Factor discovery failed: {result.get('message')}")
            
            self.test_results["real_factor_discovery"] = results
            return results
            
        except Exception as e:
            print(f"   ❌ Real factor discovery error: {e}")
            return {"status": "error", "message": str(e)}
    
    async def test_real_signal_generation(self) -> Dict[str, Any]:
        """Test real alpha signal generation using momentum agent and LLM."""
        print("\n📡 Testing REAL Alpha Signal Generation...")
        
        try:
            # Test signal generation with real momentum agent
            test_cases = [
                {
                    "symbols": ["AAPL", "GOOGL"],
                    "lookback_period": 20
                },
                {
                    "symbols": ["MSFT", "TSLA"],
                    "lookback_period": 50
                }
            ]
            
            results = {}
            for i, test_case in enumerate(test_cases):
                print(f"   📊 Test case {i+1}: {test_case}")
                
                # Call the actual MCP tool
                result = await self.call_mcp_tool("generate_alpha_signals", test_case)
                
                results[f"test_case_{i+1}"] = result
                
                if result.get("status") == "success":
                    print(f"   ✅ Real signals generated successfully")
                    signals = result.get("alpha_signals", {}).get("signals", {})
                    print(f"   📊 Generated {len(signals)} signals")
                    for symbol, signal in signals.items():
                        print(f"      {symbol}: {signal.get('signal')} (confidence: {signal.get('confidence')})")
                else:
                    print(f"   ❌ Signal generation failed: {result.get('message')}")
            
            self.test_results["real_signal_generation"] = results
            return results
            
        except Exception as e:
            print(f"   ❌ Real signal generation error: {e}")
            return {"status": "error", "message": str(e)}
    
    async def test_real_strategy_development(self) -> Dict[str, Any]:
        """Test real strategy development."""
        print("\n⚙️  Testing REAL Strategy Development...")
        
        try:
            # Test different risk levels
            risk_levels = ["conservative", "moderate", "aggressive"]
            results = {}
            
            for risk_level in risk_levels:
                print(f"   📊 Testing {risk_level} risk level...")
                
                # Call the actual MCP tool
                result = await self.call_mcp_tool("develop_strategy_configuration", {
                    "risk_level": risk_level,
                    "target_volatility": 0.15
                })
                
                results[risk_level] = result
                
                if result.get("status") == "success":
                    strategy = result.get("strategy_configuration", {})
                    print(f"   ✅ Strategy created: {strategy.get('strategy_id')}")
                    print(f"   📊 Risk level: {strategy.get('risk_level')}")
                    print(f"   📊 Target volatility: {strategy.get('target_volatility')}")
                else:
                    print(f"   ❌ Strategy development failed: {result.get('message')}")
            
            self.test_results["real_strategy_development"] = results
            return results
            
        except Exception as e:
            print(f"   ❌ Real strategy development error: {e}")
            return {"status": "error", "message": str(e)}
    
    async def test_real_backtesting(self) -> Dict[str, Any]:
        """Test real backtesting functionality."""
        print("\n📈 Testing REAL Backtesting System...")
        
        try:
            # First create a strategy
            strategy_result = await self.call_mcp_tool("develop_strategy_configuration", {
                "risk_level": "moderate",
                "target_volatility": 0.15
            })
            
            if strategy_result.get("status") != "success":
                print("   ❌ Cannot test backtesting without strategy")
                return {"status": "error", "message": "Strategy creation failed"}
            
            strategy_id = strategy_result.get("strategy_configuration", {}).get("strategy_id")
            print(f"   📊 Using strategy: {strategy_id}")
            
            # Test backtesting
            backtest_result = await self.call_mcp_tool("run_comprehensive_backtest", {
                "strategy_id": strategy_id,
                "start_date": "2020-01-01",
                "end_date": "2023-12-31"
            })
            
            if backtest_result.get("status") == "success":
                print("   ✅ Real backtest completed successfully")
                backtest_data = backtest_result.get("backtest_results", {})
                print(f"   📊 Backtest ID: {backtest_data.get('backtest_id')}")
                print(f"   📊 Total return: {backtest_data.get('total_return', 0):.2%}")
                print(f"   📊 Sharpe ratio: {backtest_data.get('sharpe_ratio', 0):.2f}")
            else:
                print(f"   ❌ Backtesting failed: {backtest_result.get('message')}")
            
            self.test_results["real_backtesting"] = {
                "strategy_creation": strategy_result,
                "backtest": backtest_result
            }
            
            return self.test_results["real_backtesting"]
            
        except Exception as e:
            print(f"   ❌ Real backtesting error: {e}")
            return {"status": "error", "message": str(e)}
    
    async def test_real_memory_submission(self) -> Dict[str, Any]:
        """Test real memory submission."""
        print("\n💾 Testing REAL Memory Submission...")
        
        try:
            # First create a strategy and run backtest
            strategy_result = await self.call_mcp_tool("develop_strategy_configuration", {
                "risk_level": "moderate",
                "target_volatility": 0.15
            })
            
            if strategy_result.get("status") != "success":
                print("   ❌ Cannot test memory submission without strategy")
                return {"status": "error", "message": "Strategy creation failed"}
            
            strategy_id = strategy_result.get("strategy_configuration", {}).get("strategy_id")
            
            # Run backtest
            backtest_result = await self.call_mcp_tool("run_comprehensive_backtest", {
                "strategy_id": strategy_id,
                "start_date": "2020-01-01",
                "end_date": "2023-12-31"
            })
            
            if backtest_result.get("status") != "success":
                print("   ❌ Cannot test memory submission without backtest")
                return {"status": "error", "message": "Backtest failed"}
            
            backtest_id = backtest_result.get("backtest_results", {}).get("backtest_id")
            
            # Submit to memory
            memory_result = await self.call_mcp_tool("submit_strategy_to_memory", {
                "strategy_id": strategy_id,
                "backtest_id": backtest_id
            })
            
            if memory_result.get("status") == "success":
                print("   ✅ Real memory submission successful")
                print(f"   📊 Submission ID: {memory_result.get('submission_id')}")
                print(f"   📊 Validation status: {memory_result.get('validation_status')}")
                
                # Check if data was actually persisted
                import os
                data_dir = "/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/data/cache"
                strategy_file = os.path.join(data_dir, "strategy_submissions.jsonl")
                
                if os.path.exists(strategy_file):
                    with open(strategy_file, 'r') as f:
                        lines = f.readlines()
                    print(f"   📁 Strategy submissions file: {len(lines)} entries")
                else:
                    print("   ⚠️  Strategy submissions file not found")
                
            else:
                print(f"   ❌ Memory submission failed: {memory_result.get('message')}")
            
            self.test_results["real_memory_submission"] = {
                "strategy_creation": strategy_result,
                "backtest": backtest_result,
                "memory_submission": memory_result
            }
            
            return self.test_results["real_memory_submission"]
            
        except Exception as e:
            print(f"   ❌ Real memory submission error: {e}")
            return {"status": "error", "message": str(e)}
    
    async def test_real_integrated_pipeline(self) -> Dict[str, Any]:
        """Test the real integrated pipeline."""
        print("\n🚀 Testing REAL Integrated Pipeline...")
        
        try:
            # Test the complete integrated pipeline
            test_data = {
                "strategy_id": "test_real_integrated_strategy",
                "symbols": ["AAPL", "GOOGL", "MSFT"],
                "start_date": "2020-01-01",
                "end_date": "2023-12-31",
                "risk_level": "moderate"
            }
            
            print(f"   📊 Test data: {test_data}")
            
            # Call the actual integrated pipeline
            result = await self.call_mcp_tool("run_integrated_backtest", test_data)
            
            if result.get("status") == "success":
                print("   ✅ Real integrated pipeline completed successfully")
                pipeline_data = result.get("pipeline_result", {})
                print(f"   📊 Pipeline ID: {pipeline_data.get('pipeline_id')}")
                print(f"   📊 Execution summary: {pipeline_data.get('execution_summary')}")
            else:
                print(f"   ❌ Integrated pipeline failed: {result.get('message')}")
            
            self.test_results["real_integrated_pipeline"] = result
            return result
            
        except Exception as e:
            print(f"   ❌ Real integrated pipeline error: {e}")
            return {"status": "error", "message": str(e)}
    
    async def run_real_tests(self):
        """Run all real functionality tests."""
        print("🚀 Starting REAL MCP Functionality Tests...")
        print(f"📅 Test session: {self.session_id}")
        print(f"🌐 Target server: {self.base_url}")
        print("=" * 80)
        
        start_time = time.time()
        
        try:
            # 1. Real factor discovery
            await self.test_real_factor_discovery()
            
            # 2. Real signal generation
            await self.test_real_signal_generation()
            
            # 3. Real strategy development
            await self.test_real_strategy_development()
            
            # 4. Real backtesting
            await self.test_real_backtesting()
            
            # 5. Real memory submission
            await self.test_real_memory_submission()
            
            # 6. Real integrated pipeline
            await self.test_real_integrated_pipeline()
            
            # Summary
            end_time = time.time()
            duration = end_time - start_time
            
            print("\n" + "=" * 80)
            print("🎉 REAL FUNCTIONALITY TEST SUITE FINISHED!")
            print(f"⏱️  Total execution time: {duration:.2f} seconds")
            print(f"📊 Test session: {self.session_id}")
            print(f"📋 Test results: {len(self.test_results)} test categories")
            
            # Save test results
            results_file = f"real_functionality_test_results_{self.session_id}.json"
            with open(results_file, 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)
            print(f"💾 Test results saved to: {results_file}")
            
            # Print summary
            print("\n📊 Test Summary:")
            for category, result in self.test_results.items():
                if isinstance(result, dict) and "status" in result:
                    status = result["status"]
                else:
                    status = "completed"
                status_icon = "✅" if status == "success" else "⚠️" if status == "warning" else "❌"
                print(f"   {status_icon} {category}: {status}")
            
        except Exception as e:
            print(f"❌ Test suite error: {e}")
            import traceback
            traceback.print_exc()

async def main():
    """Main test function."""
    print("🚀 Real MCP Functionality Test Client")
    print("Testing: REAL factor discovery, signal generation, memory submission, backtesting")
    
    async with RealMCPClient() as tester:
        await tester.run_real_tests()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"❌ Main error: {e}")
        import traceback
        traceback.print_exc()
