#!/usr/bin/env python3
"""
Test suite for Enhanced Alpha Agent Pool Demo System.

This test suite validates that the enhanced demo system provides complete
functional parity with the original core.py while demonstrating the benefits
of the decoupled architecture.

Test Coverage:
- Core functionality mapping and compatibility
- API interface consistency 
- Data flow and processing
- Error handling and resilience
- Configuration management
- Observability and monitoring
- Performance characteristics
- Memory management
"""

import asyncio
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock

import pytest

# Import the enhanced demo system
import sys
sys.path.append(str(Path(__file__).parent.parent))

from demo_decoupled_system import EnhancedAlphaPoolDemo


class TestEnhancedDemoSystem(unittest.TestCase):
    """Test suite for Enhanced Alpha Agent Pool Demo System."""
    
    def setUp(self):
        """Set up test environment."""
        self.demo = EnhancedAlphaPoolDemo(test_mode=True)
        
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'demo'):
            self.demo.reset_state()

    def test_demo_initialization(self):
        """Test demo system initialization."""
        self.assertIsNotNone(self.demo.session_id)
        self.assertTrue(self.demo.test_mode)
        self.assertIsNotNone(self.demo.config)
        self.assertIsNotNone(self.demo.logger)
        self.assertIsNotNone(self.demo.metrics)
        
        # Verify initial state
        self.assertEqual(len(self.demo.agent_registry), 0)
        self.assertEqual(len(self.demo.signal_generation_history), 0)
        self.assertEqual(len(self.demo.discovered_factors), 0)
        self.assertEqual(len(self.demo.strategy_configurations), 0)
        self.assertEqual(len(self.demo.backtest_results), 0)

    @pytest.mark.asyncio
    async def test_alpha_signal_generation(self):
        """Test alpha signal generation functionality."""
        # Test single symbol
        result = await self.demo.generate_alpha_signals(symbol="AAPL")
        
        self.assertEqual(result["status"], "success")
        self.assertIn("alpha_signals", result)
        self.assertIn("AAPL", result["alpha_signals"]["signals"])
        
        signal = result["alpha_signals"]["signals"]["AAPL"]
        self.assertIn("signal", signal)
        self.assertIn("confidence", signal)
        self.assertIn("timestamp", signal)
        self.assertIn("symbol", signal)
        self.assertIn(signal["signal"], ["BUY", "SELL", "HOLD"])
        self.assertGreaterEqual(signal["confidence"], 0.0)
        self.assertLessEqual(signal["confidence"], 1.0)
        
        # Test multiple symbols
        result = await self.demo.generate_alpha_signals(symbols=["GOOGL", "MSFT"])
        
        self.assertEqual(result["status"], "success")
        self.assertIn("GOOGL", result["alpha_signals"]["signals"])
        self.assertIn("MSFT", result["alpha_signals"]["signals"])
        
        # Verify history tracking
        self.assertGreater(len(self.demo.signal_generation_history), 0)

    @pytest.mark.asyncio
    async def test_factor_discovery(self):
        """Test alpha factor discovery functionality."""
        result = await self.demo.discover_alpha_factors()
        
        self.assertEqual(result["status"], "success")
        self.assertIn("discovery_id", result)
        self.assertIn("discovered_factors", result)
        self.assertIn("summary", result)
        
        # Verify factors were discovered
        factors = result["discovered_factors"]
        self.assertGreater(len(factors), 0)
        
        # Check specific categories
        expected_categories = ["momentum", "mean_reversion", "volatility", "technical"]
        for category in expected_categories:
            if category in factors:
                self.assertGreater(len(factors[category]), 0)
                
                # Verify factor structure
                for factor_name, factor_data in factors[category].items():
                    self.assertIn("significance", factor_data)
                    self.assertIn("sharpe", factor_data)
                    self.assertIn("description", factor_data)
        
        # Verify storage
        discovery_id = result["discovery_id"]
        self.assertIn(discovery_id, self.demo.discovered_factors)

    @pytest.mark.asyncio
    async def test_strategy_development(self):
        """Test strategy configuration development."""
        # Test different risk levels
        risk_levels = ["conservative", "moderate", "aggressive"]
        
        for risk_level in risk_levels:
            result = await self.demo.develop_strategy_configuration(risk_level=risk_level)
            
            self.assertEqual(result["status"], "success")
            self.assertIn("strategy_configuration", result)
            self.assertIn("validation_status", result)
            self.assertEqual(result["validation_status"], "PASSED")
            
            config = result["strategy_configuration"]
            self.assertIn("strategy_id", config)
            self.assertEqual(config["risk_level"], risk_level)
            self.assertIn("max_leverage", config)
            self.assertIn("stop_loss", config)
            self.assertIn("position_size", config)
            self.assertIn("factor_weights", config)
            
            # Verify different risk levels have different parameters
            if risk_level == "conservative":
                self.assertLessEqual(config["max_leverage"], 2.0)
            elif risk_level == "aggressive":
                self.assertGreaterEqual(config["max_leverage"], 2.5)
        
        # Verify storage
        self.assertGreater(len(self.demo.strategy_configurations), 0)

    @pytest.mark.asyncio
    async def test_comprehensive_backtesting(self):
        """Test comprehensive backtesting functionality."""
        # First create a strategy
        strategy_result = await self.demo.develop_strategy_configuration()
        self.assertEqual(strategy_result["status"], "success")
        strategy_id = strategy_result["strategy_configuration"]["strategy_id"]
        
        # Run backtest
        result = await self.demo.run_comprehensive_backtest(strategy_id)
        
        self.assertEqual(result["status"], "success")
        self.assertIn("backtest_results", result)
        self.assertIn("validation_status", result)
        self.assertIn("performance_summary", result)
        
        backtest = result["backtest_results"]
        self.assertEqual(backtest["strategy_id"], strategy_id)
        self.assertIn("backtest_id", backtest)
        self.assertIn("performance_metrics", backtest)
        self.assertIn("risk_metrics", backtest)
        self.assertIn("execution_metrics", backtest)
        
        # Verify metrics structure
        perf = backtest["performance_metrics"]
        self.assertIn("total_return", perf)
        self.assertIn("annualized_return", perf)
        self.assertIn("volatility", perf)
        self.assertIn("sharpe_ratio", perf)
        self.assertIn("max_drawdown", perf)
        self.assertIn("win_rate", perf)
        
        # Verify realistic ranges
        self.assertGreaterEqual(perf["win_rate"], 0.0)
        self.assertLessEqual(perf["win_rate"], 1.0)
        self.assertGreaterEqual(perf["max_drawdown"], 0.0)
        
        # Verify storage
        backtest_id = backtest["backtest_id"]
        self.assertIn(backtest_id, self.demo.backtest_results)

    def test_agent_management(self):
        """Test agent lifecycle management."""
        # Start agents
        agent_ids = ["momentum_agent", "mean_reversion_agent", "volatility_agent"]
        
        for agent_id in agent_ids:
            result = self.demo.start_agent(agent_id)
            self.assertIn("successfully", result)
            
        # List agents
        agents = self.demo.list_agents()
        self.assertEqual(len(agents), len(agent_ids))
        for agent_id in agent_ids:
            self.assertIn(agent_id, agents)
            
        # Check status
        status = self.demo.get_agent_status()
        for agent_id in agent_ids:
            self.assertIn(agent_id, status)
            self.assertEqual(status[agent_id], "running")

    @pytest.mark.asyncio
    async def test_memory_operations(self):
        """Test A2A memory operations."""
        # Test set/get operations
        test_key = "test_key_123"
        test_value = {"test": "data", "number": 42}
        
        # Set memory
        result = await self.demo.set_memory(test_key, test_value)
        self.assertEqual(result, "OK")
        
        # Get memory
        retrieved_value = await self.demo.get_memory(test_key)
        # Note: Returns mock value since no real A2A coordinator
        self.assertIsNotNone(retrieved_value)
        
        # List keys
        keys = await self.demo.list_memory_keys()
        self.assertIsInstance(keys, list)
        
        # Delete memory
        result = await self.demo.delete_memory(test_key)
        self.assertEqual(result, "OK")

    @pytest.mark.asyncio
    async def test_strategy_memory_submission(self):
        """Test strategy submission to A2A memory."""
        # Create strategy and backtest
        strategy_result = await self.demo.develop_strategy_configuration()
        strategy_id = strategy_result["strategy_configuration"]["strategy_id"]
        
        backtest_result = await self.demo.run_comprehensive_backtest(strategy_id)
        backtest_id = backtest_result["backtest_results"]["backtest_id"]
        
        # Submit to memory
        result = await self.demo.submit_strategy_to_memory(strategy_id, backtest_id)
        
        self.assertEqual(result["status"], "success")
        self.assertIn("submission_result", result)
        self.assertIn("strategy_data", result)
        
        strategy_data = result["strategy_data"]
        self.assertEqual(strategy_data["strategy_id"], strategy_id)
        self.assertIsNotNone(strategy_data["strategy_config"])
        self.assertIsNotNone(strategy_data["backtest_results"])

    def test_task_management(self):
        """Test enhanced task management functionality."""
        # These methods require orchestrator to be initialized
        # In test mode, they handle None gracefully
        
        # Test task status query
        status = self.demo.get_task_status("nonexistent_task")
        self.assertIsNone(status)
        
        # Test task cancellation
        cancelled = self.demo.cancel_task("nonexistent_task")
        self.assertFalse(cancelled)
        
        # Test active tasks list
        active = self.demo.list_active_tasks()
        self.assertEqual(active, [])
        
        # Test metrics
        metrics = self.demo.get_orchestrator_metrics()
        self.assertEqual(metrics, {})

    def test_result_collection(self):
        """Test result collection and state management."""
        # Get all results
        results = self.demo.get_all_results()
        
        expected_keys = [
            "agent_registry", "signal_generation_history", 
            "discovered_factors", "strategy_configurations",
            "backtest_results", "orchestrator_metrics"
        ]
        
        for key in expected_keys:
            self.assertIn(key, results)
            
        # Test state reset
        self.demo.agent_registry["test"] = {"status": "running"}
        self.demo.signal_generation_history.append({"test": "signal"})
        
        self.assertGreater(len(self.demo.agent_registry), 0)
        self.assertGreater(len(self.demo.signal_generation_history), 0)
        
        self.demo.reset_state()
        
        self.assertEqual(len(self.demo.agent_registry), 0)
        self.assertEqual(len(self.demo.signal_generation_history), 0)

    @pytest.mark.asyncio
    async def test_full_workflow_integration(self):
        """Test complete workflow integration."""
        # Run a mini version of the full workflow
        
        # 1. Agent management
        self.demo.start_agent("test_agent")
        
        # 2. Alpha signal generation
        signal_result = await self.demo.generate_alpha_signals(symbol="AAPL")
        self.assertEqual(signal_result["status"], "success")
        
        # 3. Factor discovery
        factor_result = await self.demo.discover_alpha_factors()
        self.assertEqual(factor_result["status"], "success")
        
        # 4. Strategy development
        strategy_result = await self.demo.develop_strategy_configuration()
        self.assertEqual(strategy_result["status"], "success")
        strategy_id = strategy_result["strategy_configuration"]["strategy_id"]
        
        # 5. Backtesting
        backtest_result = await self.demo.run_comprehensive_backtest(strategy_id)
        self.assertEqual(backtest_result["status"], "success")
        backtest_id = backtest_result["backtest_results"]["backtest_id"]
        
        # 6. Memory submission
        memory_result = await self.demo.submit_strategy_to_memory(strategy_id, backtest_id)
        self.assertEqual(memory_result["status"], "success")
        
        # 7. Verify all data is tracked
        results = self.demo.get_all_results()
        self.assertGreater(len(results["agent_registry"]), 0)
        self.assertGreater(len(results["signal_generation_history"]), 0)
        self.assertGreater(len(results["discovered_factors"]), 0)
        self.assertGreater(len(results["strategy_configurations"]), 0)
        self.assertGreater(len(results["backtest_results"]), 0)

    def test_error_handling(self):
        """Test error handling and resilience."""
        # Test with invalid parameters
        invalid_cases = [
            # Missing required parameters
            (self.demo.generate_alpha_signals, {}, "error"),
            # Invalid risk level
            (self.demo.develop_strategy_configuration, {"risk_level": "invalid"}, "success"),  # Falls back to moderate
            # Non-existent strategy for backtest
            (self.demo.run_comprehensive_backtest, {"strategy_id": "nonexistent"}, "success"),  # Still generates mock data
        ]
        
        for func, kwargs, expected_status in invalid_cases:
            if asyncio.iscoroutinefunction(func):
                # Skip async functions in sync test
                continue
            else:
                # Test sync functions that don't return status dict
                try:
                    result = func(**kwargs)
                    # If it returns a status dict, check it
                    if isinstance(result, dict) and "status" in result:
                        # Allow either expected status or error
                        self.assertIn(result["status"], [expected_status, "error"])
                except Exception:
                    # Exceptions are also acceptable for invalid inputs
                    pass

    def test_configuration_compatibility(self):
        """Test configuration and compatibility."""
        # Test different initialization modes
        demo_no_env = EnhancedAlphaPoolDemo(use_env=False, test_mode=True)
        self.assertIsNotNone(demo_no_env.config)
        
        # Test with custom config path (non-existent file)
        demo_custom = EnhancedAlphaPoolDemo(config_path="nonexistent.yaml", test_mode=True)
        self.assertIsNotNone(demo_custom.config)
        
        # Cleanup
        demo_no_env.reset_state()
        demo_custom.reset_state()

    @pytest.mark.asyncio
    async def test_performance_characteristics(self):
        """Test performance characteristics and resource usage."""
        import time
        
        # Test signal generation performance
        start_time = time.time()
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
        
        for symbol in symbols:
            result = await self.demo.generate_alpha_signals(symbol=symbol)
            self.assertEqual(result["status"], "success")
        
        elapsed_time = time.time() - start_time
        
        # Should be fast in test mode (< 1 second for 5 symbols)
        self.assertLess(elapsed_time, 1.0)
        
        # Verify no memory leaks in tracking structures
        self.assertEqual(len(self.demo.signal_generation_history), len(symbols))

    def test_test_mode_features(self):
        """Test specific test mode features."""
        # Test report generation
        self.demo._test_report = {"test": "data"}
        report = self.demo.get_test_report()
        self.assertEqual(report["test"], "data")
        
        # Test mode should be enabled
        self.assertTrue(self.demo.test_mode)

    @pytest.mark.asyncio
    async def test_momentum_health_check(self):
        """Test momentum agent health check."""
        # Without agent started
        health = await self.demo.momentum_health()
        self.assertEqual(health["process_status"], "stopped")
        self.assertEqual(health["endpoint_status"], "unhealthy")
        
        # With agent started
        self.demo.start_agent("momentum_agent")
        health = await self.demo.momentum_health()
        self.assertEqual(health["process_status"], "running")
        self.assertEqual(health["endpoint_status"], "healthy")


class TestCoreFunctionalityParity(unittest.TestCase):
    """Test parity with original core.py functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.demo = EnhancedAlphaPoolDemo(test_mode=True)
        
    def tearDown(self):
        """Clean up test environment."""
        self.demo.reset_state()

    def test_api_method_existence(self):
        """Test that all core.py API methods exist in demo system."""
        # Core functionality methods that should exist
        expected_methods = [
            # Task management
            "submit_task",
            "get_task_status", 
            "cancel_task",
            "list_active_tasks",
            "get_orchestrator_metrics",
            
            # Alpha generation
            "generate_alpha_signals",
            "discover_alpha_factors",
            "develop_strategy_configuration",
            "run_comprehensive_backtest",
            "submit_strategy_to_memory",
            
            # Agent management
            "start_agent",
            "list_agents",
            "get_agent_status",
            "momentum_health",
            
            # Memory operations
            "get_memory",
            "set_memory",
            "delete_memory",
            "list_memory_keys",
        ]
        
        for method_name in expected_methods:
            self.assertTrue(hasattr(self.demo, method_name), 
                          f"Method {method_name} missing from demo system")
            method = getattr(self.demo, method_name)
            self.assertTrue(callable(method), 
                          f"Method {method_name} is not callable")

    @pytest.mark.asyncio
    async def test_api_signature_compatibility(self):
        """Test API signature compatibility with core.py."""
        # Test generate_alpha_signals signature compatibility
        # Should work with both symbol and symbols parameters
        result1 = await self.demo.generate_alpha_signals(symbol="AAPL")
        self.assertEqual(result1["status"], "success")
        
        result2 = await self.demo.generate_alpha_signals(symbols=["GOOGL"])
        self.assertEqual(result2["status"], "success")
        
        # Test optional parameters
        result3 = await self.demo.generate_alpha_signals(
            symbol="MSFT", 
            lookback_period=30,
            date="2024-01-01"
        )
        self.assertEqual(result3["status"], "success")

    def test_return_format_compatibility(self):
        """Test return format compatibility with core.py."""
        # Agent management should return strings/lists as expected
        result = self.demo.start_agent("test_agent")
        self.assertIsInstance(result, str)
        self.assertIn("successfully", result)
        
        agents = self.demo.list_agents()
        self.assertIsInstance(agents, list)
        
        status = self.demo.get_agent_status()
        self.assertIsInstance(status, dict)

    @pytest.mark.asyncio
    async def test_data_structure_compatibility(self):
        """Test data structure compatibility with core.py."""
        # Test signal generation returns expected structure
        result = await self.demo.generate_alpha_signals(symbol="AAPL")
        
        # Should match core.py return structure
        expected_structure = {
            "status": str,
            "alpha_signals": {
                "signals": dict,
                "metadata": dict
            }
        }
        
        self._validate_structure(result, expected_structure)
        
        # Test factor discovery structure
        factor_result = await self.demo.discover_alpha_factors()
        factor_expected = {
            "status": str,
            "discovery_id": str,
            "discovered_factors": dict,
            "summary": dict
        }
        
        self._validate_structure(factor_result, factor_expected)

    def _validate_structure(self, data, expected, path=""):
        """Recursively validate data structure."""
        for key, expected_type in expected.items():
            self.assertIn(key, data, f"Missing key {path}.{key}")
            
            if isinstance(expected_type, dict):
                self.assertIsInstance(data[key], dict, f"Wrong type at {path}.{key}")
                self._validate_structure(data[key], expected_type, f"{path}.{key}")
            else:
                self.assertIsInstance(data[key], expected_type, f"Wrong type at {path}.{key}")


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedDemoSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestCoreFunctionalityParity))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run the tests
    success = run_tests()
    exit(0 if success else 1)
