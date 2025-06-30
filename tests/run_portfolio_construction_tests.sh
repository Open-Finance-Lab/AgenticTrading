#!/bin/bash

# Portfolio Construction Agent Pool Memory Integration Test Runner
# This script runs comprehensive tests for the Portfolio Construction Agent Pool
# including multi-agent integration and memory bridge functionality

set -e  # Exit on any error

# Get the absolute path of the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo "========================================"
echo "Portfolio Construction Agent Pool Tests"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo "Current directory: $(pwd)"

# Change to project root directory
cd "$PROJECT_ROOT"

# Check if required directories exist
echo "Checking directory structure..."
if [ ! -d "FinAgents/agent_pools/portfolio_construction_agent_pool" ]; then
    echo "Error: Portfolio Construction Agent Pool directory not found"
    exit 1
fi

if [ ! -d "FinAgents/memory" ]; then
    echo "Error: Memory directory not found"
    exit 1
fi

if [ ! -d "tests" ]; then
    echo "Error: Tests directory not found"
    exit 1
fi

echo "All required directories found."

# Set PYTHONPATH to include project root and modules
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/FinAgents:$PROJECT_ROOT/FinAgents/agent_pools:$PYTHONPATH"

echo "PYTHONPATH set to: $PYTHONPATH"

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp for log files
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="logs/portfolio_construction_test_${TIMESTAMP}.log"
REPORT_FILE="logs/portfolio_construction_test_report_${TIMESTAMP}.html"

echo "Starting Portfolio Construction Agent Pool tests..."
echo "Log file: $LOG_FILE"
echo "Report file: $REPORT_FILE"

# Run the Portfolio Construction Agent Pool integration test
echo "Running Portfolio Construction Agent Pool memory integration test..."
python -m pytest \
    FinAgents/agent_pools/portfolio_construction_agent_pool/test_integration.py \
    -v \
    --tb=short \
    --capture=no \
    --log-cli-level=INFO \
    2>&1 | tee "$LOG_FILE"

# Check if tests passed
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "=================================="
    echo "✅ All tests passed successfully!"
    echo "=================================="
    echo "Log file: $LOG_FILE"
    echo "Report file: $REPORT_FILE"
    exit 0
else
    echo "=================================="
    echo "❌ Some tests failed!"
    echo "=================================="
    echo "Check the log file for details: $LOG_FILE"
    echo "Check the report file for details: $REPORT_FILE"
    exit 1
fi
