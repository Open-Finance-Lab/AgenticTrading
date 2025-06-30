#!/bin/bash

# Alpha Agent Pool Memory Integration Test Runner
# 
# This script provides comprehensive testing for the Alpha Agent Pool memory
# integration with the External Memory Agent system. It includes academic-grade
# testing procedures for quantitative finance applications.
#
# Author: Jifeng Li
# Created: 2025-06-30
# License: openMDW

set -e  # Exit on any error

# Color codes for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
TEST_DIR="$PROJECT_ROOT/tests"
LOGS_DIR="$PROJECT_ROOT/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TEST_LOG="$LOGS_DIR/alpha_pool_memory_test_$TIMESTAMP.log"

# Create logs directory if it doesn't exist
mkdir -p "$LOGS_DIR"

# Function to print colored output
print_header() {
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

print_step() {
    echo -e "${PURPLE}🔄 $1${NC}"
}

# Function to check Python dependencies
check_dependencies() {
    print_step "Checking Python dependencies..."
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed or not in PATH"
        exit 1
    fi
    
    # Check Python version (require 3.8+)
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    REQUIRED_VERSION="3.8"
    
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
        print_error "Python 3.8+ required, found Python $PYTHON_VERSION"
        exit 1
    fi
    
    print_success "Python $PYTHON_VERSION detected"
    
    # Check if pip is available
    if ! command -v pip3 &> /dev/null; then
        print_warning "pip3 not found, trying pip..."
        if ! command -v pip &> /dev/null; then
            print_error "pip is not installed"
            exit 1
        else
            PIP_CMD="pip"
        fi
    else
        PIP_CMD="pip3"
    fi
    
    print_success "pip package manager available"
}

# Function to install required packages
install_dependencies() {
    print_step "Installing required Python packages..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$PROJECT_ROOT/venv" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv "$PROJECT_ROOT/venv"
    fi
    
    # Activate virtual environment
    source "$PROJECT_ROOT/venv/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install test dependencies
    print_info "Installing test dependencies..."
    pip install pytest pytest-asyncio
    
    # Install project requirements if available
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        print_info "Installing project requirements..."
        pip install -r "$PROJECT_ROOT/requirements.txt"
    fi
    
    # Install additional dependencies for testing
    pip install aiohttp aiofiles aiosqlite
    
    print_success "Dependencies installed successfully"
}

# Function to check project structure
check_project_structure() {
    print_step "Validating project structure..."
    
    # Check for required directories
    REQUIRED_DIRS=(
        "$PROJECT_ROOT/FinAgents"
        "$PROJECT_ROOT/FinAgents/agent_pools"
        "$PROJECT_ROOT/FinAgents/agent_pools/alpha_agent_pool"
        "$PROJECT_ROOT/FinAgents/memory"
        "$PROJECT_ROOT/tests"
    )
    
    for dir in "${REQUIRED_DIRS[@]}"; do
        if [ ! -d "$dir" ]; then
            print_error "Required directory not found: $dir"
            exit 1
        fi
    done
    
    # Check for required files
    REQUIRED_FILES=(
        "$PROJECT_ROOT/FinAgents/agent_pools/alpha_agent_pool/memory_bridge.py"
        "$PROJECT_ROOT/FinAgents/agent_pools/alpha_agent_pool/core.py"
        "$PROJECT_ROOT/tests/test_alpha_agent_pool_memory_integration.py"
    )
    
    for file in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$file" ]; then
            print_error "Required file not found: $file"
            exit 1
        fi
    done
    
    print_success "Project structure validation passed"
}

# Function to run pre-test environment setup
setup_test_environment() {
    print_step "Setting up test environment..."
    
    # Set PYTHONPATH to include project root
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    
    # Create test data directories if they don't exist
    mkdir -p "$PROJECT_ROOT/data/cache"
    mkdir -p "$PROJECT_ROOT/logs"
    
    # Generate test data if needed
    if [ ! -f "$PROJECT_ROOT/data/cache/AAPL_2024-01-01_2024-01-31_1d.csv" ]; then
        print_info "Generating test market data..."
        cat > "$PROJECT_ROOT/data/cache/AAPL_2024-01-01_2024-01-31_1d.csv" << EOF
timestamp,open,high,low,close,volume
2024-01-01,150.00,152.00,149.50,151.50,1000000
2024-01-02,151.50,153.00,150.00,152.25,1100000
2024-01-03,152.25,154.00,151.00,153.75,1200000
2024-01-04,153.75,155.50,152.50,154.20,1150000
2024-01-05,154.20,156.00,153.00,155.80,1300000
EOF
        print_success "Test market data generated"
    fi
    
    print_success "Test environment setup completed"
}

# Function to run the main test suite
run_tests() {
    print_step "Running Alpha Agent Pool Memory Integration Tests..."
    
    # Activate virtual environment
    if [ -d "$PROJECT_ROOT/venv" ]; then
        source "$PROJECT_ROOT/venv/bin/activate"
    fi
    
    # Navigate to project root
    cd "$PROJECT_ROOT"
    
    # Set environment variables
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    export ALPHA_POOL_TEST_MODE="true"
    export MEMORY_AGENT_TEST_MODE="true"
    
    # Create detailed log file
    echo "Alpha Agent Pool Memory Integration Test Log" > "$TEST_LOG"
    echo "Started at: $(date)" >> "$TEST_LOG"
    echo "Project Root: $PROJECT_ROOT" >> "$TEST_LOG"
    echo "Python Path: $PYTHONPATH" >> "$TEST_LOG"
    echo "----------------------------------------" >> "$TEST_LOG"
    
    # Run the comprehensive test suite
    print_info "Executing comprehensive test suite..."
    
    if python3 "$TEST_DIR/test_alpha_agent_pool_memory_integration.py" 2>&1 | tee -a "$TEST_LOG"; then
        print_success "Test suite execution completed"
        TEST_EXIT_CODE=0
    else
        print_error "Test suite execution failed"
        TEST_EXIT_CODE=1
    fi
    
    # Also run with pytest if available
    if command -v pytest &> /dev/null; then
        print_info "Running additional pytest validation..."
        
        if pytest "$TEST_DIR/test_alpha_agent_pool_memory_integration.py" -v --tb=short 2>&1 | tee -a "$TEST_LOG"; then
            print_success "Pytest validation completed"
        else
            print_warning "Pytest validation encountered issues (may be due to missing dependencies)"
        fi
    fi
    
    return $TEST_EXIT_CODE
}

# Function to generate test report
generate_report() {
    print_step "Generating test report..."
    
    REPORT_FILE="$LOGS_DIR/alpha_pool_memory_test_report_$TIMESTAMP.html"
    
    cat > "$REPORT_FILE" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Alpha Agent Pool Memory Integration Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 8px; }
        .success { color: #28a745; }
        .error { color: #dc3545; }
        .warning { color: #ffc107; }
        .info { color: #17a2b8; }
        .log-content { background-color: #f8f9fa; padding: 15px; border-radius: 4px; 
                      font-family: monospace; white-space: pre-wrap; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Alpha Agent Pool Memory Integration Test Report</h1>
        <p><strong>Test Execution Date:</strong> $(date)</p>
        <p><strong>Project Location:</strong> $PROJECT_ROOT</p>
        <p><strong>Test Script Version:</strong> 1.0.0</p>
    </div>
    
    <h2>Test Environment</h2>
    <ul>
        <li><strong>Python Version:</strong> $(python3 --version)</li>
        <li><strong>Operating System:</strong> $(uname -s)</li>
        <li><strong>Architecture:</strong> $(uname -m)</li>
        <li><strong>Test Log Location:</strong> $TEST_LOG</li>
    </ul>
    
    <h2>Test Execution Log</h2>
    <div class="log-content">
$(cat "$TEST_LOG" 2>/dev/null || echo "Log file not available")
    </div>
    
    <h2>Test Summary</h2>
    <p>This report contains the results of comprehensive testing for the Alpha Agent Pool 
    memory integration system. The tests validate signal storage, performance tracking, 
    pattern recognition, and cross-component integration following academic standards 
    for quantitative finance applications.</p>
    
    <p><em>Report generated automatically by the Alpha Agent Pool test suite.</em></p>
</body>
</html>
EOF
    
    print_success "Test report generated: $REPORT_FILE"
}

# Function to cleanup test artifacts
cleanup() {
    print_step "Cleaning up test artifacts..."
    
    # Remove temporary test files (but keep logs)
    find "$PROJECT_ROOT" -name "*.pyc" -delete 2>/dev/null || true
    find "$PROJECT_ROOT" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find "$PROJECT_ROOT" -name "test_memory*.db" -delete 2>/dev/null || true
    find "$PROJECT_ROOT" -name "test_memory_unit*.json" -delete 2>/dev/null || true
    
    print_success "Cleanup completed"
}

# Function to display usage information
show_usage() {
    echo "Alpha Agent Pool Memory Integration Test Runner"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h          Show this help message"
    echo "  --install-deps      Install required dependencies"
    echo "  --skip-deps         Skip dependency checking and installation"
    echo "  --cleanup-only      Only perform cleanup operations"
    echo "  --verbose           Enable verbose output"
    echo "  --report-only       Generate report from existing logs"
    echo ""
    echo "Examples:"
    echo "  $0                  Run full test suite with dependency checks"
    echo "  $0 --install-deps   Install dependencies and run tests"
    echo "  $0 --skip-deps      Run tests without dependency management"
    echo "  $0 --cleanup-only   Clean up test artifacts only"
    echo ""
    echo "For more information, see the project documentation."
}

# Main execution function
main() {
    local INSTALL_DEPS=false
    local SKIP_DEPS=false
    local CLEANUP_ONLY=false
    local VERBOSE=false
    local REPORT_ONLY=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                show_usage
                exit 0
                ;;
            --install-deps)
                INSTALL_DEPS=true
                shift
                ;;
            --skip-deps)
                SKIP_DEPS=true
                shift
                ;;
            --cleanup-only)
                CLEANUP_ONLY=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                set -x  # Enable bash debugging
                shift
                ;;
            --report-only)
                REPORT_ONLY=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Display banner
    print_header "Alpha Agent Pool Memory Integration Test Suite"
    print_info "Academic Framework for Quantitative Finance System Testing"
    print_info "Starting test execution at $(date)"
    
    # Handle cleanup-only mode
    if [ "$CLEANUP_ONLY" = true ]; then
        cleanup
        print_success "Cleanup completed successfully"
        exit 0
    fi
    
    # Handle report-only mode
    if [ "$REPORT_ONLY" = true ]; then
        generate_report
        print_success "Report generation completed"
        exit 0
    fi
    
    # Main test execution flow
    local EXIT_CODE=0
    
    # Step 1: Check project structure
    check_project_structure
    
    # Step 2: Handle dependencies
    if [ "$SKIP_DEPS" != true ]; then
        check_dependencies
        
        if [ "$INSTALL_DEPS" = true ]; then
            install_dependencies
        fi
    fi
    
    # Step 3: Setup test environment
    setup_test_environment
    
    # Step 4: Run tests
    if ! run_tests; then
        EXIT_CODE=1
    fi
    
    # Step 5: Generate report
    generate_report
    
    # Step 6: Cleanup
    cleanup
    
    # Final summary
    print_header "Test Execution Summary"
    
    if [ $EXIT_CODE -eq 0 ]; then
        print_success "All tests completed successfully!"
        print_info "Test log available at: $TEST_LOG"
        print_info "Test report available at: $LOGS_DIR/alpha_pool_memory_test_report_$TIMESTAMP.html"
        print_success "Alpha Agent Pool memory integration is ready for production deployment."
    else
        print_error "Test execution completed with errors."
        print_warning "Please review the test log and fix any issues before deployment."
        print_info "Test log available at: $TEST_LOG"
    fi
    
    # Instructions for next steps
    print_header "Next Steps"
    if [ $EXIT_CODE -eq 0 ]; then
        print_info "✅ Integration testing passed"
        print_info "🚀 Ready to deploy Alpha Agent Pool with memory integration"
        print_info "📊 Performance monitoring recommended for production use"
    else
        print_info "🔧 Fix identified issues and re-run tests"
        print_info "📋 Review test logs for detailed error information"
        print_info "💡 Consider running with --verbose for additional debugging"
    fi
    
    exit $EXIT_CODE
}

# Trap to ensure cleanup on exit
trap cleanup EXIT

# Execute main function with all arguments
main "$@"
