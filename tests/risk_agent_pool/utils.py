"""
Test Utilities for Risk Agent Pool Tests

Author: Jifeng Li
License: openMDW

This module provides utility functions and helper classes for testing
the Risk Agent Pool functionality.
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import statistics


logger = logging.getLogger(__name__)


class TestTimer:
    """Timer utility for performance testing"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def start(self):
        """Start the timer"""
        self.start_time = time.perf_counter()
        return self
    
    def stop(self):
        """Stop the timer and calculate duration"""
        if self.start_time is None:
            raise ValueError("Timer not started")
        
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        return self.duration
    
    def __enter__(self):
        """Context manager entry"""
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
    
    def get_duration(self) -> float:
        """Get the measured duration"""
        if self.duration is None:
            raise ValueError("Timer not completed")
        return self.duration


class PerformanceMetrics:
    """Collect and analyze performance metrics"""
    
    def __init__(self):
        self.measurements = []
        self.labels = []
    
    def add_measurement(self, duration: float, label: str = ""):
        """Add a timing measurement"""
        self.measurements.append(duration)
        self.labels.append(label)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistical analysis of measurements"""
        if not self.measurements:
            return {}
        
        return {
            "mean": statistics.mean(self.measurements),
            "median": statistics.median(self.measurements),
            "min": min(self.measurements),
            "max": max(self.measurements),
            "std_dev": statistics.stdev(self.measurements) if len(self.measurements) > 1 else 0.0,
            "count": len(self.measurements)
        }
    
    def print_summary(self):
        """Print performance summary"""
        stats = self.get_statistics()
        print(f"\nPerformance Summary:")
        print(f"  Measurements: {stats['count']}")
        print(f"  Mean: {stats['mean']:.4f}s")
        print(f"  Median: {stats['median']:.4f}s")
        print(f"  Min: {stats['min']:.4f}s")
        print(f"  Max: {stats['max']:.4f}s")
        print(f"  Std Dev: {stats['std_dev']:.4f}s")


class TestValidator:
    """Validation utilities for test assertions"""
    
    @staticmethod
    def validate_risk_analysis_result(result: Dict[str, Any]) -> bool:
        """Validate standard risk analysis result format"""
        required_fields = ["agent", "risk_type", "status"]
        
        for field in required_fields:
            if field not in result:
                logger.error(f"Missing required field: {field}")
                return False
        
        if result["status"] not in ["success", "error"]:
            logger.error(f"Invalid status: {result['status']}")
            return False
        
        if result["status"] == "success" and "results" not in result:
            logger.error("Success result missing 'results' field")
            return False
        
        if result["status"] == "error" and "error" not in result:
            logger.error("Error result missing 'error' field")
            return False
        
        return True
    
    @staticmethod
    def validate_var_result(var_result: Dict[str, Any]) -> bool:
        """Validate VaR calculation result"""
        if not isinstance(var_result, dict):
            return False
        
        # Check for required VaR fields
        required_fields = ["time_horizon", "calculation_method"]
        for field in required_fields:
            if field not in var_result:
                logger.error(f"VaR result missing field: {field}")
                return False
        
        # Check for at least one VaR value
        var_fields = [key for key in var_result.keys() if key.startswith("var_")]
        if not var_fields:
            logger.error("VaR result contains no VaR values")
            return False
        
        # Validate VaR values are numeric and positive
        for field in var_fields:
            value = var_result[field]
            if not isinstance(value, (int, float)) or value < 0:
                logger.error(f"Invalid VaR value for {field}: {value}")
                return False
        
        return True
    
    @staticmethod
    def validate_portfolio_data(portfolio_data: Dict[str, Any]) -> bool:
        """Validate portfolio data structure"""
        if not isinstance(portfolio_data, dict):
            return False
        
        if "positions" not in portfolio_data:
            logger.error("Portfolio data missing 'positions'")
            return False
        
        positions = portfolio_data["positions"]
        if not isinstance(positions, list) or len(positions) == 0:
            logger.error("Portfolio positions must be non-empty list")
            return False
        
        # Validate each position
        required_position_fields = ["asset_id", "quantity", "current_price"]
        for i, position in enumerate(positions):
            for field in required_position_fields:
                if field not in position:
                    logger.error(f"Position {i} missing field: {field}")
                    return False
            
            # Validate numeric fields
            if position["quantity"] <= 0 or position["current_price"] <= 0:
                logger.error(f"Position {i} has invalid numeric values")
                return False
        
        return True
    
    @staticmethod
    def validate_correlation_matrix(matrix: List[List[float]]) -> bool:
        """Validate correlation matrix properties"""
        if not matrix or not matrix[0]:
            return False
        
        size = len(matrix)
        
        # Check square matrix
        for row in matrix:
            if len(row) != size:
                logger.error("Correlation matrix is not square")
                return False
        
        # Check diagonal elements are 1.0
        for i in range(size):
            if abs(matrix[i][i] - 1.0) > 1e-6:
                logger.error(f"Diagonal element [{i}][{i}] is not 1.0")
                return False
        
        # Check symmetry
        for i in range(size):
            for j in range(size):
                if abs(matrix[i][j] - matrix[j][i]) > 1e-6:
                    logger.error(f"Matrix not symmetric at [{i}][{j}]")
                    return False
        
        # Check correlation bounds [-1, 1]
        for i in range(size):
            for j in range(size):
                if matrix[i][j] < -1.0 or matrix[i][j] > 1.0:
                    logger.error(f"Correlation value out of bounds at [{i}][{j}]: {matrix[i][j]}")
                    return False
        
        return True


class MockDataGenerator:
    """Generate realistic mock data for testing"""
    
    @staticmethod
    def generate_price_series(
        initial_price: float = 100.0,
        num_periods: int = 252,
        volatility: float = 0.2,
        drift: float = 0.05
    ) -> List[float]:
        """Generate realistic price series using geometric Brownian motion"""
        import random
        import math
        
        prices = [initial_price]
        dt = 1.0 / 252  # Daily time step
        
        for _ in range(num_periods - 1):
            random_shock = random.gauss(0, 1)
            price_change = drift * dt + volatility * math.sqrt(dt) * random_shock
            new_price = prices[-1] * math.exp(price_change)
            prices.append(new_price)
        
        return prices
    
    @staticmethod
    def generate_returns_from_prices(prices: List[float]) -> List[float]:
        """Calculate returns from price series"""
        if len(prices) < 2:
            return []
        
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)
        
        return returns
    
    @staticmethod
    def generate_correlated_returns(
        base_returns: List[float],
        correlation: float = 0.5,
        volatility_multiplier: float = 1.0
    ) -> List[float]:
        """Generate correlated return series"""
        import random
        import math
        
        if not base_returns:
            return []
        
        correlated_returns = []
        base_vol = TestValidator._calculate_volatility(base_returns)
        
        for base_ret in base_returns:
            # Decompose into correlated and independent components
            independent_shock = random.gauss(0, base_vol * volatility_multiplier)
            correlated_component = correlation * base_ret
            independent_component = math.sqrt(1 - correlation**2) * independent_shock
            
            correlated_ret = correlated_component + independent_component
            correlated_returns.append(correlated_ret)
        
        return correlated_returns
    
    @staticmethod
    def _calculate_volatility(returns: List[float]) -> float:
        """Calculate volatility of returns"""
        if len(returns) < 2:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return)**2 for r in returns) / (len(returns) - 1)
        return variance**0.5


class AsyncTestRunner:
    """Utility for running async tests with various patterns"""
    
    @staticmethod
    async def run_concurrent_tasks(
        tasks: List[Callable],
        max_concurrent: int = 5
    ) -> List[Any]:
        """Run tasks concurrently with concurrency limit"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_with_semaphore(task):
            async with semaphore:
                return await task()
        
        # Create coroutines with semaphore
        coroutines = [run_with_semaphore(task) for task in tasks]
        
        # Run all tasks
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        return results
    
    @staticmethod
    async def measure_throughput(
        task_factory: Callable,
        duration_seconds: float = 10.0,
        max_concurrent: int = 10
    ) -> Dict[str, float]:
        """Measure task throughput over time period"""
        start_time = time.perf_counter()
        end_time = start_time + duration_seconds
        
        completed_tasks = 0
        errors = 0
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_task():
            nonlocal completed_tasks, errors
            async with semaphore:
                try:
                    await task_factory()
                    completed_tasks += 1
                except Exception:
                    errors += 1
        
        # Launch tasks continuously until time limit
        tasks = []
        while time.perf_counter() < end_time:
            task = asyncio.create_task(run_task())
            tasks.append(task)
            
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.01)
        
        # Wait for remaining tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        actual_duration = time.perf_counter() - start_time
        
        return {
            "duration": actual_duration,
            "completed_tasks": completed_tasks,
            "errors": errors,
            "tasks_per_second": completed_tasks / actual_duration,
            "error_rate": errors / (completed_tasks + errors) if (completed_tasks + errors) > 0 else 0.0
        }


class TestReportGenerator:
    """Generate test reports and summaries"""
    
    def __init__(self):
        self.test_results = []
        self.performance_data = []
    
    def add_test_result(
        self,
        test_name: str,
        status: str,
        duration: float,
        details: Dict[str, Any] = None
    ):
        """Add test result to report"""
        result = {
            "test_name": test_name,
            "status": status,
            "duration": duration,
            "timestamp": datetime.now(),
            "details": details or {}
        }
        self.test_results.append(result)
    
    def add_performance_data(
        self,
        test_name: str,
        metrics: Dict[str, Any]
    ):
        """Add performance data to report"""
        perf_data = {
            "test_name": test_name,
            "timestamp": datetime.now(),
            "metrics": metrics
        }
        self.performance_data.append(perf_data)
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate test summary report"""
        if not self.test_results:
            return {"status": "no_tests_run"}
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["status"] == "passed"])
        failed_tests = len([r for r in self.test_results if r["status"] == "failed"])
        
        total_duration = sum(r["duration"] for r in self.test_results)
        avg_duration = total_duration / total_tests
        
        return {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "pass_rate": passed_tests / total_tests,
            "total_duration": total_duration,
            "average_duration": avg_duration,
            "timestamp": datetime.now()
        }
    
    def generate_detailed_report(self) -> str:
        """Generate detailed test report"""
        summary = self.generate_summary()
        
        report_lines = [
            "=" * 60,
            "RISK AGENT POOL TEST REPORT",
            "=" * 60,
            f"Test Summary:",
            f"  Total Tests: {summary.get('total_tests', 0)}",
            f"  Passed: {summary.get('passed', 0)}",
            f"  Failed: {summary.get('failed', 0)}",
            f"  Pass Rate: {summary.get('pass_rate', 0):.2%}",
            f"  Total Duration: {summary.get('total_duration', 0):.2f}s",
            f"  Average Duration: {summary.get('average_duration', 0):.2f}s",
            "",
            "Test Details:",
            "-" * 40
        ]
        
        for result in self.test_results:
            status_symbol = "✓" if result["status"] == "passed" else "✗"
            report_lines.append(
                f"{status_symbol} {result['test_name']} ({result['duration']:.3f}s)"
            )
        
        if self.performance_data:
            report_lines.extend([
                "",
                "Performance Data:",
                "-" * 40
            ])
            
            for perf in self.performance_data:
                report_lines.append(f"📊 {perf['test_name']}:")
                for metric, value in perf['metrics'].items():
                    if isinstance(value, float):
                        report_lines.append(f"    {metric}: {value:.4f}")
                    else:
                        report_lines.append(f"    {metric}: {value}")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)


class ErrorInjector:
    """Utility for injecting errors during testing"""
    
    def __init__(self):
        self.error_probability = 0.0
        self.error_types = []
        self.call_count = 0
    
    def set_error_probability(self, probability: float):
        """Set probability of error injection (0.0 to 1.0)"""
        self.error_probability = max(0.0, min(1.0, probability))
    
    def add_error_type(self, error_class, message: str = "Injected test error"):
        """Add error type to inject"""
        self.error_types.append((error_class, message))
    
    def maybe_inject_error(self):
        """Randomly inject an error based on probability"""
        import random
        
        self.call_count += 1
        
        if random.random() < self.error_probability and self.error_types:
            error_class, message = random.choice(self.error_types)
            raise error_class(f"{message} (call #{self.call_count})")
    
    def reset(self):
        """Reset error injector state"""
        self.call_count = 0
        self.error_probability = 0.0
        self.error_types.clear()


@asynccontextmanager
async def temporary_config_override(obj, config_updates: Dict[str, Any]):
    """Temporarily override object configuration for testing"""
    original_config = getattr(obj, 'config', {}).copy()
    
    try:
        # Apply updates
        if hasattr(obj, 'config'):
            obj.config.update(config_updates)
        else:
            obj.config = config_updates
        
        yield obj
        
    finally:
        # Restore original configuration
        if hasattr(obj, 'config'):
            obj.config = original_config


def assert_approx_equal(actual: float, expected: float, tolerance: float = 1e-6):
    """Assert that two floating point numbers are approximately equal"""
    if abs(actual - expected) > tolerance:
        raise AssertionError(
            f"Values not approximately equal: {actual} != {expected} "
            f"(tolerance: {tolerance})"
        )


def assert_result_structure(result: Dict[str, Any], expected_keys: List[str]):
    """Assert that result has expected structure"""
    for key in expected_keys:
        if key not in result:
            raise AssertionError(f"Missing expected key in result: {key}")


def load_test_config(config_file: str = "test_config.json") -> Dict[str, Any]:
    """Load test configuration from file"""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Test config file {config_file} not found, using defaults")
        return {
            "timeout": 30.0,
            "max_concurrent": 10,
            "performance_threshold": 5.0,
            "error_tolerance": 0.01
        }


def save_test_artifacts(
    results: Dict[str, Any],
    artifacts_dir: str = "test_artifacts"
):
    """Save test artifacts for analysis"""
    import os
    import json
    
    # Create artifacts directory
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_results_{timestamp}.json"
    filepath = os.path.join(artifacts_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Test artifacts saved to {filepath}")


# Global test utilities instance
test_timer = TestTimer()
test_validator = TestValidator()
mock_data_generator = MockDataGenerator()
async_test_runner = AsyncTestRunner()
error_injector = ErrorInjector()
