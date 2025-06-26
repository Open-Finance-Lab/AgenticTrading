# Risk Agent Pool Test Suite

This directory contains a comprehensive test suite for the Risk Agent Pool, designed to validate all functionality including unit tests, integration tests, and performance tests.

## Author
Jifeng Li

## License
openMDW

## Test Structure

```
tests/risk_agent_pool/
├── __init__.py                 # Test package initialization
├── conftest.py                # Pytest configuration and fixtures
├── fixtures.py                # Shared test fixtures and mock data
├── utils.py                   # Test utilities and helpers
├── run_tests.py              # Test runner script
├── test_registry.py          # Agent registry tests
├── test_core.py              # Core functionality tests
├── test_memory_bridge.py     # Memory bridge tests
├── test_agents.py            # Individual agent tests
├── test_integration.py       # End-to-end integration tests
├── test_performance.py       # Performance and load tests
└── README.md                 # This file
```

## Test Categories

### 1. Unit Tests
- **test_registry.py**: Tests for agent registration, instantiation, and lifecycle management
- **test_core.py**: Tests for the main RiskAgentPool class functionality
- **test_memory_bridge.py**: Tests for memory storage and retrieval operations
- **test_agents.py**: Tests for individual risk agent implementations

### 2. Integration Tests
- **test_integration.py**: End-to-end workflow tests, error handling, and system integration

### 3. Performance Tests
- **test_performance.py**: Load testing, memory usage analysis, and performance benchmarking

## Test Coverage

The test suite covers:

### Core Functionality
- ✅ Risk agent registration and management
- ✅ OpenAI LLM integration
- ✅ External memory bridge integration
- ✅ MCP server functionality
- ✅ Context decompression and processing
- ✅ Error handling and recovery

### Risk Agents
- ✅ Market Risk Agent (beta, volatility, sector analysis)
- ✅ Volatility Agent (historical, GARCH modeling)
- ✅ VaR Calculator (historical, parametric, Monte Carlo)
- ✅ Credit Risk Agent (default probability, concentration)
- ✅ Liquidity Risk Agent (bid-ask impact, liquidation time)
- ✅ Operational Risk Agent (system reliability, process risk)
- ✅ Stress Testing Agent (scenario analysis, survival probability)
- ✅ Model Risk Agent (validation, performance monitoring)

### Integration Scenarios
- ✅ Complete risk analysis workflows
- ✅ Portfolio stress testing
- ✅ Real-time risk monitoring
- ✅ Historical analysis comparison
- ✅ Custom agent integration
- ✅ Concurrent analysis handling
- ✅ Error recovery and resilience

### Performance Metrics
- ✅ Single analysis latency
- ✅ Concurrent load handling
- ✅ Memory usage stability
- ✅ Sustained load performance
- ✅ Agent switching efficiency
- ✅ Error recovery impact

## Running Tests

### Prerequisites
```bash
pip install pytest pytest-asyncio pytest-cov pytest-xdist psutil
```

### Basic Usage

#### Run All Tests
```bash
python tests/risk_agent_pool/run_tests.py
```

#### Run Specific Test Categories
```bash
# Unit tests only
python tests/risk_agent_pool/run_tests.py --type unit

# Integration tests only
python tests/risk_agent_pool/run_tests.py --type integration

# Performance tests only
python tests/risk_agent_pool/run_tests.py --type performance
```

#### Advanced Options
```bash
# Run with coverage reporting
python tests/risk_agent_pool/run_tests.py --coverage

# Run tests in parallel (faster execution)
python tests/risk_agent_pool/run_tests.py --parallel

# Run with verbose output
python tests/risk_agent_pool/run_tests.py --verbose

# Run specific test file
python tests/risk_agent_pool/run_tests.py --specific test_core.py

# Run specific test function
python tests/risk_agent_pool/run_tests.py --specific test_core.py::test_analyze_risk_basic
```

### Direct Pytest Usage
```bash
# From project root
cd /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration

# Run all tests
pytest tests/risk_agent_pool/ -v

# Run with coverage
pytest tests/risk_agent_pool/ --cov=FinAgents.agent_pools.risk --cov-report=html

# Run specific markers
pytest tests/risk_agent_pool/ -m "unit"
pytest tests/risk_agent_pool/ -m "integration"
pytest tests/risk_agent_pool/ -m "performance"

# Run in parallel
pytest tests/risk_agent_pool/ -n auto
```

## Test Configuration

### Environment Variables
- `TESTING=true`: Enables test mode
- `LOG_LEVEL=WARNING`: Reduces log noise during testing
- `CI=true`: Enables CI-specific test configurations

### Mock Configuration
Tests use comprehensive mocking for:
- OpenAI API responses
- External memory storage
- Market data feeds
- Network communications

### Performance Thresholds
- Single analysis: < 1 second
- Concurrent load (50 requests): < 10 seconds
- Memory growth: < 50MB during sustained load
- Error recovery overhead: < 20% performance impact

## Sample Test Data

The test suite includes realistic sample data:

### Portfolio Data
```python
{
    "positions": [
        {"symbol": "AAPL", "quantity": 1000, "market_value": 150000},
        {"symbol": "GOOGL", "quantity": 500, "market_value": 125000},
        # ... more positions
    ],
    "total_value": 1000000,
    "currency": "USD"
}
```

### Market Data
```python
{
    "symbols": ["AAPL", "GOOGL", "MSFT"],
    "prices": [150.00, 2500.00, 300.00],
    "volatilities": [0.25, 0.30, 0.20],
    "correlations": [[1.0, 0.7, 0.8], [0.7, 1.0, 0.6], [0.8, 0.6, 1.0]]
}
```

### Risk Scenarios
```python
{
    "scenarios": [
        {"name": "Market Crash", "market_shock": -0.30},
        {"name": "Interest Rate Spike", "rate_shock": 0.02},
        {"name": "Credit Crisis", "credit_spread": 0.05}
    ]
}
```

## Expected Test Results

### Typical Test Execution Times
- Unit tests: 20-30 seconds
- Integration tests: 45-60 seconds
- Performance tests: 3-5 minutes
- All tests with coverage: 8-10 minutes

### Success Criteria
- All unit tests pass: 100%
- Integration tests pass: ≥95%
- Performance tests meet thresholds: ≥90%
- Code coverage: ≥85%

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure project root is in Python path
export PYTHONPATH=/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration:$PYTHONPATH
```

#### Async Test Failures
```bash
# Install required async testing packages
pip install pytest-asyncio
```

#### Performance Test Timeouts
```bash
# Run performance tests with increased timeout
pytest tests/risk_agent_pool/test_performance.py --timeout=300
```

### Debug Mode
```bash
# Run with detailed debugging
python tests/risk_agent_pool/run_tests.py --verbose --specific test_core.py::test_analyze_risk_basic
```

### Memory Issues
```bash
# Run with memory profiling
pytest tests/risk_agent_pool/ --memray
```

## Continuous Integration

For CI/CD pipelines:

```yaml
# Example GitHub Actions configuration
- name: Run Risk Agent Pool Tests
  run: |
    cd /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration
    python tests/risk_agent_pool/run_tests.py --type unit --coverage
    python tests/risk_agent_pool/run_tests.py --type integration
```

## Test Reports

Generate comprehensive test reports:
```bash
# Generate detailed test report
python tests/risk_agent_pool/run_tests.py --report

# Coverage report (HTML)
python tests/risk_agent_pool/run_tests.py --coverage
# Open: tests/risk_agent_pool/htmlcov/index.html
```

## Contributing

When adding new tests:

1. Follow the existing test structure and naming conventions
2. Use appropriate test markers (`@pytest.mark.unit`, `@pytest.mark.integration`, etc.)
3. Include comprehensive docstrings and comments in English
4. Add mock data to `fixtures.py` for reusability
5. Ensure new tests pass in isolation and with the full suite
6. Update this README if adding new test categories

## Support

For test-related issues:
- Check the test output for specific error messages
- Review the test logs for detailed execution information
- Ensure all dependencies are installed and up to date
- Verify the Risk Agent Pool implementation is complete and correct
