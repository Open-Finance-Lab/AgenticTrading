# FinAgent Memory System - Testing Framework

This directory contains all testing components for the FinAgent Memory System.

## Test File Structure

- `memory_test.py` - Unified memory system test suite (24 comprehensive tests)
- `database_test.py` - Standalone database operations test script (10 database tests)
- `test_system.sh` - Automated test execution script
- `test_summary.py` - Test results analysis and reporting tool
- `neo4j_diagnostic.py` - Neo4j connection diagnostic utility

## Usage

### Direct Test Execution
```bash
# Run complete test suite from memory directory
python tests/memory_test.py --verbose

# Run standalone database tests
python tests/database_test.py --verbose

# Use automated test script
./tests/test_system.sh
```

### Specialized Database Testing
```bash
# Run comprehensive database tests
python tests/database_test.py --verbose

# Cleanup test data only
python tests/database_test.py --cleanup

# Save test results
python tests/database_test.py --output database_results.json
```

### Additional Utilities
```bash
# View test results summary
python tests/test_summary.py

# Neo4j connection diagnostics
python tests/neo4j_diagnostic.py

# Automated test system
./tests/test_system.sh
```

## Test Coverage

1. **Port Connectivity Tests** - Validates service port accessibility
2. **Database Operations Tests** - Direct Neo4j database operation testing
   - Database connection testing
   - Database schema validation
   - Memory storage operations
   - Memory retrieval operations
   - Memory search functionality
   - Database statistics collection
   - Data cleanup operations
3. **Memory Server Tests** - HTTP REST API functionality testing
4. **MCP Server Tests** - Model Context Protocol functionality testing
5. **A2A Server Tests** - Agent-to-Agent Protocol functionality testing
6. **Performance Tests** - A2A server performance benchmarking

## Database Testing Details

Database tests connect directly to Neo4j database and perform the following operations:

- 🔗 **Connection Test**: Validates Neo4j database connectivity
- 📋 **Schema Check**: Verifies database indexes and constraints
- 💾 **Storage Test**: Creates test memory nodes
- 🔍 **Retrieval Test**: Queries and retrieves memory data
- 🔎 **Search Test**: Keyword-based memory searching
- 📊 **Statistics Test**: Database statistics collection
- 🧹 **Cleanup Test**: Test data removal

**Database Configuration Requirements**:
- Neo4j service running on `bolt://localhost:7687`
- Username: `neo4j`
- Password: `finagent123`
- Database: `neo4j`

## Test Results Interpretation

- ✅ **100% Success Rate** - All services and database operations functioning normally
- 📊 **Test Metrics** - Includes response times, throughput, and performance data
- 🌐 **Service Status** - Online status of each port service
- 🗄️ **Database Status** - Neo4j database connectivity and operation status

All test results are automatically saved to JSON files for subsequent analysis and monitoring.

## Expected Test Results

**Total Tests**: 24 comprehensive tests
- **Port Connectivity**: 3/3 ✅
- **Database Operations**: 7/7 ✅
- **Memory Server**: 2/2 ✅
- **MCP Server**: 6/6 ✅
- **A2A Server**: 6/6 ✅

**Performance Benchmarks**:
- A2A Protocol: 580+ operations/second
- Database Operations: Sub-second response times
- Service Startup: < 5 seconds for all services
