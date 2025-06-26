"""
Pytest configuration for Risk Agent Pool tests.

Author: Jifeng Li
License: openMDW
"""

import pytest
import asyncio
import os
import sys
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import all fixtures from fixtures.py
from .fixtures import *

# Configure asyncio for tests
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Custom markers for test categorization
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for end-to-end workflows"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and load tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take longer to execute"
    )
    config.addinivalue_line(
        "markers", "requires_openai: Tests that require OpenAI API access"
    )
    config.addinivalue_line(
        "markers", "requires_memory: Tests that require external memory storage"
    )


# Test session setup and teardown
@pytest.fixture(scope="session", autouse=True)
def test_session_setup():
    """Setup for the entire test session."""
    print("\n" + "=" * 60)
    print("Starting Risk Agent Pool Test Suite")
    print(f"Test session started at: {datetime.now()}")
    print("=" * 60)
    
    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "WARNING"
    
    yield
    
    # Cleanup after all tests
    print("\n" + "=" * 60)
    print("Risk Agent Pool Test Suite Complete")
    print(f"Test session ended at: {datetime.now()}")
    print("=" * 60)


@pytest.fixture(scope="function", autouse=True)
def test_function_setup():
    """Setup and teardown for each test function."""
    # Setup before each test
    import gc
    gc.collect()  # Clean up memory before each test
    
    yield
    
    # Cleanup after each test
    gc.collect()  # Clean up memory after each test


# Timeout configuration for async tests
@pytest.fixture(scope="function")
def async_timeout():
    """Default timeout for async tests."""
    return 30  # 30 seconds


# Test data cleanup
@pytest.fixture(scope="function")
def temp_data_dir(tmp_path):
    """Create temporary directory for test data."""
    test_data_dir = tmp_path / "test_data"
    test_data_dir.mkdir()
    
    yield test_data_dir
    
    # Cleanup is automatic with tmp_path


# Mock configurations
@pytest.fixture(scope="session")
def mock_config():
    """Configuration for mock services."""
    return {
        "openai": {
            "model": "gpt-4",
            "temperature": 0.1,
            "max_tokens": 2000,
            "timeout": 30
        },
        "memory": {
            "storage_type": "mock",
            "cache_ttl": 3600,
            "batch_size": 100
        },
        "risk_pool": {
            "max_concurrent_analyses": 50,
            "default_confidence_level": 0.95,
            "cache_results": True
        }
    }


# Performance test configuration
@pytest.fixture(scope="session")
def performance_config():
    """Configuration for performance tests."""
    return {
        "load_test": {
            "max_concurrent_requests": 50,
            "test_duration_seconds": 60,
            "target_requests_per_second": 10
        },
        "memory_test": {
            "max_memory_growth_mb": 50,
            "gc_frequency": 10
        },
        "latency_test": {
            "max_avg_latency_ms": 1000,
            "max_p95_latency_ms": 2000
        }
    }


# Error injection for resilience testing
@pytest.fixture
def error_injector():
    """Utility for injecting errors in tests."""
    class ErrorInjector:
        def __init__(self):
            self.failure_rate = 0.0
            self.call_count = 0
        
        def set_failure_rate(self, rate):
            """Set the failure rate (0.0 to 1.0)."""
            self.failure_rate = rate
        
        def should_fail(self):
            """Check if the current call should fail."""
            self.call_count += 1
            import random
            return random.random() < self.failure_rate
        
        def reset(self):
            """Reset the error injector."""
            self.failure_rate = 0.0
            self.call_count = 0
    
    injector = ErrorInjector()
    yield injector
    injector.reset()


# Test result collection
@pytest.fixture(scope="session")
def test_results():
    """Collect test results for reporting."""
    results = {
        "passed": [],
        "failed": [],
        "skipped": [],
        "errors": [],
        "performance_metrics": {}
    }
    
    yield results
    
    # Generate summary report
    total_tests = len(results["passed"]) + len(results["failed"]) + len(results["skipped"])
    if total_tests > 0:
        print(f"\nTest Summary:")
        print(f"  Passed: {len(results['passed'])}")
        print(f"  Failed: {len(results['failed'])}")
        print(f"  Skipped: {len(results['skipped'])}")
        print(f"  Success Rate: {len(results['passed'])/total_tests:.1%}")


# Pytest hooks for custom behavior
def pytest_runtest_makereport(item, call):
    """Hook to customize test reporting."""
    if call.when == "call":
        # Log test execution
        test_name = item.nodeid
        if call.excinfo is None:
            print(f"✅ {test_name}")
        else:
            print(f"❌ {test_name}: {call.excinfo.value}")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add custom markers."""
    for item in items:
        # Add markers based on test file names
        if "test_performance" in item.fspath.basename:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        elif "test_integration" in item.fspath.basename:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
        
        # Add markers for tests requiring external services
        if "openai" in item.name.lower():
            item.add_marker(pytest.mark.requires_openai)
        if "memory" in item.name.lower():
            item.add_marker(pytest.mark.requires_memory)


# Skip tests based on environment
def pytest_runtest_setup(item):
    """Setup before running each test."""
    # Skip performance tests if not explicitly requested
    if item.get_closest_marker("performance"):
        if not item.config.getoption("-m") or "performance" not in item.config.getoption("-m"):
            pytest.skip("Performance tests skipped (use -m performance to run)")
    
    # Skip tests requiring external services in CI
    if os.environ.get("CI") == "true":
        if item.get_closest_marker("requires_openai"):
            pytest.skip("OpenAI tests skipped in CI")
        if item.get_closest_marker("requires_memory"):
            pytest.skip("Memory tests skipped in CI")


# Command line options
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-performance",
        action="store_true",
        default=False,
        help="Run performance tests"
    )
    parser.addoption(
        "--run-slow",
        action="store_true", 
        default=False,
        help="Run slow tests"
    )
    parser.addoption(
        "--mock-external",
        action="store_true",
        default=True,
        help="Mock external services (default: True)"
    )
