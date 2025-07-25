#!/usr/bin/env python3
"""
Alpha Agent Pool End-to-End Integration Test

Comprehensive test suite for Alpha Agent Pool with Enhanced A2A Memory Bridge integration.
This test covers the complete alpha strategy research workflow from factor discovery 
to strategy submission and memory coordination.

Author: FinAgent Orchestration Team
Date: July 2025
"""
import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlphaPoolEndToEndTester:
    """Comprehensive end-to-end test suite for Alpha Agent Pool."""
    
    def __init__(self):
        self.test_results = {}
        self.bridge = None
        
    async def setup(self):
        """Initialize test environment and Enhanced A2A Memory Bridge."""
        print("🚀 Initializing Alpha Agent Pool End-to-End Test Suite")
        print("=" * 70)
        
        try:
            from FinAgents.agent_pools.alpha_agent_pool.enhanced_a2a_memory_bridge import (
                get_memory_bridge
            )
            
            self.bridge = await get_memory_bridge(pool_id="e2e_test_suite")
            health = await self.bridge.health_check()
            
            if health.get('bridge_status') == 'healthy':
                print("✅ Enhanced A2A Memory Bridge initialized successfully")
                return True
            else:
                print("❌ Memory Bridge health check failed")
                return False
                
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    async def test_memory_services_connectivity(self):
        """Test connectivity to all memory services."""
        print("\n🔗 Testing Memory Services Connectivity")
        print("-" * 50)
        
        memory_services = [
            {"name": "A2A Memory Server", "port": 8002},
            {"name": "MCP Memory Server", "port": 8001},
            {"name": "Legacy Memory Server", "port": 8000}
        ]
        
        results = {}
        for service in memory_services:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"http://127.0.0.1:{service['port']}", timeout=5)
                    status = "accessible" if response.status_code in [200, 405] else f"status_{response.status_code}"
                    results[service['name']] = status
                    print(f"   {service['name']} (port {service['port']}): ✅ {status}")
            except Exception as e:
                results[service['name']] = f"error: {str(e)}"
                print(f"   {service['name']} (port {service['port']}): ❌ {str(e)}")
        
        self.test_results['memory_connectivity'] = results
        return all("accessible" in status for status in results.values())
    
    async def test_alpha_factor_discovery(self):
        """Test alpha factor discovery with memory storage."""
        print("\n🔍 Testing Alpha Factor Discovery")
        print("-" * 50)
        
        try:
            discovery_performance = {
                "total_factors_discovered": 12,
                "significance_threshold": 0.05,
                "avg_ir": 0.72,
                "discovery_timestamp": datetime.utcnow().isoformat(),
                "methodology": "Multi-variate factor analysis with regime detection",
                "top_factors": [
                    "momentum_volume_composite",
                    "mean_reversion_volatility", 
                    "cross_sectional_momentum",
                    "volatility_risk_premium"
                ]
            }
            
            success = await self.bridge.store_agent_performance(
                agent_id="alpha_factor_discovery_engine",
                performance_data=discovery_performance
            )
            
            if success:
                print("   ✅ Alpha factor discovery data stored successfully")
                self.test_results['factor_discovery'] = "success"
                return True
            else:
                print("   ❌ Failed to store alpha factor discovery data")
                self.test_results['factor_discovery'] = "failed"
                return False
                
        except Exception as e:
            print(f"   ❌ Alpha factor discovery test failed: {e}")
            self.test_results['factor_discovery'] = f"error: {str(e)}"
            return False
    
    async def test_strategy_configuration(self):
        """Test strategy configuration development with insights storage."""
        print("\n⚙️ Testing Strategy Configuration Development")
        print("-" * 50)
        
        try:
            strategy_insights = {
                "strategy_id": "enhanced_multi_factor_momentum_v3",
                "strategy_name": "Enhanced Multi-Factor Momentum Strategy",
                "primary_factors": ["momentum_volume_composite", "cross_sectional_momentum"],
                "secondary_factors": ["mean_reversion_volatility", "volatility_risk_premium"],
                "target_volatility": 0.16,
                "target_tracking_error": 0.09,
                "max_drawdown_limit": 0.15,
                "expected_capacity": 100000000,  # $100M
                "risk_level": "moderate_aggressive",
                "configuration_timestamp": datetime.utcnow().isoformat(),
                "academic_validation": "peer_reviewed_methodology"
            }
            
            success = await self.bridge.store_strategy_insights(
                strategy_id="enhanced_multi_factor_momentum_v3",
                insights_data=strategy_insights
            )
            
            if success:
                print("   ✅ Strategy configuration insights stored successfully")
                self.test_results['strategy_configuration'] = "success"
                return True
            else:
                print("   ❌ Failed to store strategy configuration insights")
                self.test_results['strategy_configuration'] = "failed"
                return False
                
        except Exception as e:
            print(f"   ❌ Strategy configuration test failed: {e}")
            self.test_results['strategy_configuration'] = f"error: {str(e)}"
            return False
    
    async def test_comprehensive_backtest(self):
        """Test comprehensive backtest with performance storage."""
        print("\n📊 Testing Comprehensive Backtest")
        print("-" * 50)
        
        try:
            backtest_performance = {
                "backtest_id": "bt_enhanced_multi_factor_v3_20250725",
                "strategy_id": "enhanced_multi_factor_momentum_v3",
                "total_return": 0.34,
                "annualized_return": 0.21,
                "volatility": 0.159,
                "sharpe_ratio": 1.48,
                "information_ratio": 1.25,
                "maximum_drawdown": -0.112,
                "win_rate": 0.64,
                "backtest_period": "2020-01-01 to 2024-12-31",
                "validation_status": "PASSED",
                "timestamp": datetime.utcnow().isoformat(),
                "risk_metrics": {
                    "var_95": -0.023,
                    "cvar_95": -0.035,
                    "beta": 0.87,
                    "alpha": 0.089
                }
            }
            
            success = await self.bridge.store_agent_performance(
                agent_id="comprehensive_backtest_engine",
                performance_data=backtest_performance
            )
            
            if success:
                print("   ✅ Comprehensive backtest results stored successfully")
                self.test_results['comprehensive_backtest'] = "success"
                return True
            else:
                print("   ❌ Failed to store backtest results")
                self.test_results['comprehensive_backtest'] = "failed"
                return False
                
        except Exception as e:
            print(f"   ❌ Comprehensive backtest test failed: {e}")
            self.test_results['comprehensive_backtest'] = f"error: {str(e)}"
            return False
    
    async def test_cross_agent_learning(self):
        """Test cross-agent learning pattern storage."""
        print("\n🤝 Testing Cross-Agent Learning")
        print("-" * 50)
        
        try:
            learning_data = {
                "source_agent": "enhanced_multi_factor_momentum_v3",
                "target_agents": ["momentum_agent_v4", "volatility_agent_v2", "regime_agent_v1"],
                "learning_pattern": {
                    "pattern_type": "dynamic_factor_weighting",
                    "description": "Adaptive factor weight optimization based on market microstructure",
                    "parameters": {
                        "regime_detection_method": "hmm_volatility_clustering",
                        "rebalance_frequency": "daily",
                        "momentum_weight_range": [0.3, 0.9],
                        "mean_reversion_weight_range": [0.1, 0.7],
                        "volatility_adjustment": True
                    },
                    "performance_improvement": 0.28,
                    "statistical_significance": 0.001
                },
                "transfer_timestamp": datetime.utcnow().isoformat(),
                "validation_metrics": {
                    "oos_sharpe_improvement": 0.23,
                    "drawdown_reduction": 0.18,
                    "capacity_increase": 35000000,  # $35M additional
                    "robustness_score": 0.89
                }
            }
            
            success = await self.bridge.store_cross_agent_learning(learning_data)
            
            if success:
                print("   ✅ Cross-agent learning data stored successfully")
                self.test_results['cross_agent_learning'] = "success"
                return True
            else:
                print("   ❌ Failed to store cross-agent learning data")
                self.test_results['cross_agent_learning'] = "failed"
                return False
                
        except Exception as e:
            print(f"   ❌ Cross-agent learning test failed: {e}")
            self.test_results['cross_agent_learning'] = f"error: {str(e)}"
            return False
    
    async def test_strategy_retrieval(self):
        """Test strategy retrieval and comparison."""
        print("\n🔍 Testing Strategy Retrieval")
        print("-" * 50)
        
        try:
            similar_strategies = await self.bridge.retrieve_similar_strategies(
                query="multi-factor momentum strategy with volatility adjustment and regime detection",
                limit=5
            )
            
            print(f"   ✅ Retrieved {len(similar_strategies)} similar strategies")
            self.test_results['strategy_retrieval'] = {
                "status": "success",
                "strategies_found": len(similar_strategies)
            }
            return True
            
        except Exception as e:
            print(f"   ❌ Strategy retrieval test failed: {e}")
            self.test_results['strategy_retrieval'] = f"error: {str(e)}"
            return False
    
    async def test_memory_bridge_statistics(self):
        """Test memory bridge statistics and health monitoring."""
        print("\n📈 Testing Memory Bridge Statistics")
        print("-" * 50)
        
        try:
            stats = await self.bridge.get_memory_statistics()
            
            print(f"   Bridge Status: {stats.get('bridge_status', 'unknown')}")
            print(f"   Success Rate: {stats.get('success_rate', 0.0):.1%}")
            print(f"   Total Operations: {stats.get('operation_stats', {}).get('total_operations', 0)}")
            print(f"   Store Operations: {stats.get('operation_stats', {}).get('store_operations', 0)}")
            print(f"   Retrieve Operations: {stats.get('operation_stats', {}).get('retrieve_operations', 0)}")
            
            self.test_results['memory_statistics'] = {
                "status": "success",
                "success_rate": stats.get('success_rate', 0.0),
                "total_operations": stats.get('operation_stats', {}).get('total_operations', 0)
            }
            
            print("   ✅ Memory bridge statistics retrieved successfully")
            return True
            
        except Exception as e:
            print(f"   ❌ Memory bridge statistics test failed: {e}")
            self.test_results['memory_statistics'] = f"error: {str(e)}"
            return False
    
    async def cleanup(self):
        """Cleanup test resources."""
        try:
            if self.bridge:
                await self.bridge.close()
                
            from FinAgents.agent_pools.alpha_agent_pool.enhanced_a2a_memory_bridge import (
                shutdown_memory_bridge
            )
            await shutdown_memory_bridge()
            print("\n🧹 Test cleanup completed successfully")
            
        except Exception as e:
            print(f"\n⚠️ Cleanup warning: {e}")
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        print("\n📋 Alpha Agent Pool End-to-End Test Report")
        print("=" * 70)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() 
                             if isinstance(result, str) and result == "success" or
                                isinstance(result, dict) and result.get("status") == "success")
        
        print(f"Total Tests: {total_tests}")
        print(f"Successful Tests: {successful_tests}")
        print(f"Success Rate: {successful_tests/total_tests:.1%}")
        
        print("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status_icon = "✅" if (isinstance(result, str) and result == "success" or
                                isinstance(result, dict) and result.get("status") == "success") else "❌"
            print(f"   {status_icon} {test_name}: {result}")
        
        # Save results to file
        report_data = {
            "test_timestamp": datetime.utcnow().isoformat(),
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": successful_tests/total_tests,
            "detailed_results": self.test_results
        }
        
        report_file = Path(__file__).parent / f"test_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Test report saved to: {report_file}")
        
        return successful_tests == total_tests

async def main():
    """Main test execution function."""
    tester = AlphaPoolEndToEndTester()
    
    try:
        # Setup
        if not await tester.setup():
            print("❌ Test setup failed, aborting")
            return False
        
        # Execute test suite
        test_methods = [
            tester.test_memory_services_connectivity,
            tester.test_alpha_factor_discovery,
            tester.test_strategy_configuration,
            tester.test_comprehensive_backtest,
            tester.test_cross_agent_learning,
            tester.test_strategy_retrieval,
            tester.test_memory_bridge_statistics
        ]
        
        all_passed = True
        for test_method in test_methods:
            result = await test_method()
            all_passed = all_passed and result
        
        # Generate report
        report_success = tester.generate_test_report()
        
        if all_passed and report_success:
            print("\n🎉 All tests passed successfully!")
            print("🔥 Alpha Agent Pool Enhanced A2A Memory Bridge Integration: FULLY OPERATIONAL")
        else:
            print("\n⚠️ Some tests failed or encountered issues")
        
        return all_passed
        
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
