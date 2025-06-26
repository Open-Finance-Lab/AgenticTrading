"""
Test runner script for Risk Agent Pool test suite.

Author: Jifeng Li
License: openMDW
"""

import os
import sys
import pytest
import argparse
from datetime import datetime


def run_tests(test_type="all", verbose=False, coverage=False, parallel=False):
    """
    Run the Risk Agent Pool test suite.
    
    Args:
        test_type (str): Type of tests to run ("unit", "integration", "performance", "all")
        verbose (bool): Enable verbose output
        coverage (bool): Enable coverage reporting
        parallel (bool): Run tests in parallel
    """
    # Base pytest arguments
    pytest_args = []
    
    # Set test directory based on type
    test_dir = "tests/risk_agent_pool/"
    
    if test_type == "unit":
        pytest_args.extend([
            f"{test_dir}test_registry.py",
            f"{test_dir}test_core.py",
            f"{test_dir}test_memory_bridge.py",
            f"{test_dir}test_agents.py"
        ])
    elif test_type == "integration":
        pytest_args.append(f"{test_dir}test_integration.py")
    elif test_type == "performance":
        pytest_args.append(f"{test_dir}test_performance.py")
    elif test_type == "all":
        pytest_args.append(test_dir)
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    # Add verbose output if requested
    if verbose:
        pytest_args.append("-v")
    
    # Add coverage if requested
    if coverage:
        pytest_args.extend([
            "--cov=FinAgents.agent_pools.risk",
            "--cov-report=html:tests/risk_agent_pool/htmlcov",
            "--cov-report=term-missing"
        ])
    
    # Add parallel execution if requested
    if parallel and test_type != "performance":  # Don't parallelize performance tests
        pytest_args.extend(["-n", "auto"])
    
    # Add markers for different test types
    if test_type == "performance":
        pytest_args.extend(["-m", "not slow"])  # Skip slow tests in performance mode
    
    # Additional pytest configuration
    pytest_args.extend([
        "--tb=short",  # Shorter tracebacks
        "--strict-markers",  # Strict marker validation
        "--disable-warnings"  # Disable warnings for cleaner output
    ])
    
    print(f"Running {test_type} tests for Risk Agent Pool...")
    print(f"Test command: pytest {' '.join(pytest_args)}")
    print("-" * 60)
    
    # Run the tests
    exit_code = pytest.main(pytest_args)
    
    # Print summary
    print("-" * 60)
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed.")
    
    return exit_code


def run_specific_test(test_file, test_function=None, verbose=False):
    """
    Run a specific test file or function.
    
    Args:
        test_file (str): Path to the test file
        test_function (str): Specific test function to run
        verbose (bool): Enable verbose output
    """
    pytest_args = []
    
    if test_function:
        pytest_args.append(f"{test_file}::{test_function}")
    else:
        pytest_args.append(test_file)
    
    if verbose:
        pytest_args.extend(["-v", "-s"])
    
    return pytest.main(pytest_args)


def setup_test_environment():
    """Setup the test environment."""
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Set environment variables for testing
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "WARNING"  # Reduce log noise during tests
    
    # Create test output directories
    test_output_dir = "tests/risk_agent_pool/test_output"
    os.makedirs(test_output_dir, exist_ok=True)
    
    print("Test environment setup complete.")


def generate_test_report():
    """Generate a test report summary."""
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
# Risk Agent Pool Test Report
Generated: {report_time}

## Test Coverage
- Unit Tests: test_registry.py, test_core.py, test_memory_bridge.py, test_agents.py
- Integration Tests: test_integration.py
- Performance Tests: test_performance.py

## Test Categories
1. **Registry Tests**: Agent registration, instantiation, lifecycle
2. **Core Tests**: Risk analysis, OpenAI integration, MCP server
3. **Memory Bridge Tests**: Data storage, retrieval, caching
4. **Agent Tests**: Individual agent functionality (market, credit, VaR, etc.)
5. **Integration Tests**: End-to-end workflows, error handling
6. **Performance Tests**: Load testing, memory usage, concurrency

## Running Tests
```bash
# Run all tests
python tests/risk_agent_pool/run_tests.py

# Run specific test types
python tests/risk_agent_pool/run_tests.py --type unit
python tests/risk_agent_pool/run_tests.py --type integration
python tests/risk_agent_pool/run_tests.py --type performance

# Run with coverage
python tests/risk_agent_pool/run_tests.py --coverage

# Run in parallel
python tests/risk_agent_pool/run_tests.py --parallel

# Verbose output
python tests/risk_agent_pool/run_tests.py --verbose
```

## Test Data
The test suite uses comprehensive mock data including:
- Sample portfolio data with multiple positions
- Market data with prices, volatilities, and correlations
- Risk scenarios for stress testing
- Mock OpenAI and memory bridge responses

## Expected Performance
- Unit tests: < 30 seconds
- Integration tests: < 60 seconds  
- Performance tests: < 5 minutes
- All tests with coverage: < 10 minutes
"""
    
    with open("tests/risk_agent_pool/TEST_REPORT.md", "w") as f:
        f.write(report)
    
    print("Test report generated: tests/risk_agent_pool/TEST_REPORT.md")


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="Risk Agent Pool Test Runner")
    
    parser.add_argument(
        "--type", "-t",
        choices=["unit", "integration", "performance", "all"],
        default="all",
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Enable coverage reporting"
    )
    
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--specific", "-s",
        help="Run specific test file or function (e.g., test_core.py::test_analyze_risk)"
    )
    
    parser.add_argument(
        "--report", "-r",
        action="store_true",
        help="Generate test report"
    )
    
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Only setup test environment"
    )
    
    args = parser.parse_args()
    
    # Setup test environment
    setup_test_environment()
    
    if args.setup_only:
        print("Test environment setup complete. Exiting.")
        return 0
    
    if args.report:
        generate_test_report()
        return 0
    
    if args.specific:
        # Parse specific test
        if "::" in args.specific:
            test_file, test_function = args.specific.split("::", 1)
        else:
            test_file, test_function = args.specific, None
        
        return run_specific_test(
            f"tests/risk_agent_pool/{test_file}",
            test_function,
            args.verbose
        )
    
    # Run standard test suite
    return run_tests(
        test_type=args.type,
        verbose=args.verbose,
        coverage=args.coverage,
        parallel=args.parallel
    )


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
