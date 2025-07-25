# Alpha Agent Pool and Memory Agent A2A Connection Test

This directory contains a comprehensive test suite for validating the A2A (Agent-to-Agent) protocol connection between the Alpha Agent Pool and Memory Agent services.

## Test Files

### `test_alpha_memory_a2a_connection.py`
The main test script that performs comprehensive A2A connectivity testing including:
- Service health checks for all required components
- A2A protocol endpoint validation
- Message exchange testing
- Memory operations through A2A interface
- Performance metrics and latency analysis

### `run_a2a_connection_test.sh`
A convenient shell script that:
- Activates the correct conda environment
- Checks service availability
- Runs the comprehensive test suite
- Provides clear status reporting

## Prerequisites

Before running the tests, ensure the following services are started:

### 1. Memory Services
```bash
cd /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/memory
./start_memory_services.sh all
```

This starts:
- Memory Server (port 8000)
- MCP Protocol Server (port 8001) 
- A2A Protocol Server (port 8002)

### 2. Alpha Agent Pool
```bash
cd /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/tests
./start_agent_pools.sh
```

This starts the Alpha Agent Pool service (port 8081).

### 3. Environment Requirements
- Conda environment named 'agent' with all dependencies installed
- Neo4j database running (if memory persistence is enabled)
- OpenAI API key configured (if LLM services are tested)

## Running the Tests

### Option 1: Using the convenience script (Recommended)
```bash
cd /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/tests
./run_a2a_connection_test.sh
```

### Option 2: Direct Python execution
```bash
cd /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration
conda activate agent
python tests/test_alpha_memory_a2a_connection.py
```

## Test Coverage

The test suite validates:

1. **Service Health Checks**: Verifies all required services are running and responding
2. **A2A Protocol Endpoints**: Tests A2A server connectivity and basic protocol compliance
3. **Message Exchange**: Validates A2A message routing and response handling
4. **Memory Operations**: Tests memory storage/retrieval through A2A interface
5. **Alpha Pool Integration**: Verifies Alpha Agent Pool A2A capabilities
6. **Performance Metrics**: Measures latency and success rates for A2A communications

## Test Results

Results are automatically saved to:
- Console output with real-time progress
- Log files in `/logs/` directory
- JSON summary report with detailed metrics

## Expected Outcomes

A successful test run should show:
- All services responding to health checks
- A2A protocol endpoints accessible
- Message exchange functioning (even if specific endpoints vary)
- Memory server connectivity through A2A
- Acceptable performance metrics (< 2000ms average latency, > 60% success rate)

## Troubleshooting

### Common Issues

1. **Services not running**: Ensure all prerequisite services are started before testing
2. **Port conflicts**: Check that ports 8000, 8002, and 8081 are available
3. **Environment issues**: Verify conda 'agent' environment is properly configured
4. **Network connectivity**: Ensure localhost networking is functional

### Debugging

Check the logs directory for detailed error information:
```bash
tail -f logs/a2a_connection_test.log
tail -f logs/memory_server.log
tail -f logs/a2a_server.log
```

## Integration Notes

This test suite is designed to validate the foundational A2A connectivity required for:
- Alpha signal sharing between agent pools
- Memory persistence and retrieval coordination
- Inter-agent communication protocols
- Real-time strategy coordination

The tests are intentionally flexible to accommodate different A2A protocol implementations while ensuring core connectivity requirements are met.
