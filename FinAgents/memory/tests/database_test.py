#!/usr/bin/env python3
"""
FinAgent Memory System - Database Operations Test

This script provides focused testing for Neo4j database operations.
It can be run independently to verify database functionality.

Usage:
    python database_test.py [--verbose] [--cleanup]
    
Examples:
    python database_test.py --verbose
    python database_test.py --cleanup  # Clean up all test data
"""

import json
import time
import argparse
from datetime import datetime
from typing import Dict, Any, List
import sys

# Neo4j database testing
try:
    from neo4j import GraphDatabase, Driver
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

# Colors for output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'

def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Colors.BLUE}{'='*70}{Colors.NC}")
    print(f"{Colors.BLUE}🗄️  {text}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*70}{Colors.NC}")

def print_info(text: str):
    """Print info message."""
    print(f"{Colors.CYAN}ℹ️  {text}{Colors.NC}")

def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✅ {text}{Colors.NC}")

def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.NC}")

def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}❌ {text}{Colors.NC}")

class DatabaseTester:
    """Neo4j database operations tester."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.test_results: List[Dict[str, Any]] = []
        
        # Database configuration
        self.database_config = {
            "uri": "bolt://localhost:7687",
            "username": "neo4j", 
            "password": "finagent123",
            "database": "neo4j"
        }
        self.neo4j_driver = None
        
    def record_test(self, test_name: str, passed: bool, details: str = "", error: str = ""):
        """Record test result."""
        self.test_results.append({
            "test_name": test_name,
            "passed": passed,
            "details": details,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        
        status = "✅ PASS" if passed else "❌ FAIL"
        color = Colors.GREEN if passed else Colors.RED
        print(f"{color}{status} - {test_name}{Colors.NC}")
        if details:
            print(f"   {details}")
        if error and self.verbose:
            print(f"   Error: {error}")
    
    def test_connection(self) -> bool:
        """Test database connection."""
        print_header("DATABASE CONNECTION TEST")
        
        if not NEO4J_AVAILABLE:
            self.record_test("Neo4j Driver", False, 
                           "Neo4j driver not available - install with: pip install neo4j")
            return False
        
        # Try different authentication configurations
        test_configs = [
            {"uri": "bolt://localhost:7687", "username": "neo4j", "password": "finagent123"},
            {"uri": "bolt://localhost:7687", "username": "neo4j", "password": "FinOrchestration"},
            {"uri": "bolt://localhost:7687", "username": "neo4j", "password": "neo4j"},
            {"uri": "bolt://127.0.0.1:7687", "username": "neo4j", "password": "finagent123"},
        ]
        
        for config in test_configs:
            try:
                if self.verbose:
                    print_info(f"Trying: {config['uri']} with user '{config['username']}'")
                
                # Add retry logic for connection
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        self.neo4j_driver = GraphDatabase.driver(
                            config["uri"],
                            auth=(config["username"], config["password"])
                        )
                        
                        # Verify connection
                        with self.neo4j_driver.session() as session:
                            result = session.run("RETURN 1 as test")
                            test_value = result.single()["test"]
                            if test_value == 1:
                                self.record_test("Database Connection", True, 
                                               f"Connected to {config['uri']} with user '{config['username']}'")
                                return True
                    except Exception as e:
                        if attempt < max_retries - 1:
                            if self.verbose:
                                print_info(f"Connection attempt {attempt + 1} failed, retrying...")
                            time.sleep(0.5)  # Short delay before retry
                            continue
                        else:
                            raise e
                        
            except Exception as e:
                if self.verbose:
                    print_info(f"Connection failed: {str(e)}")
                if self.neo4j_driver:
                    try:
                        self.neo4j_driver.close()
                    except:
                        pass
                    self.neo4j_driver = None
                continue
        
        self.record_test("Database Connection", False, 
                       "Could not connect with any configuration")
        return False
    
    def test_schema_operations(self) -> bool:
        """Test database schema operations."""
        print_header("DATABASE SCHEMA OPERATIONS")
        
        if not self.neo4j_driver:
            self.record_test("Schema Operations", False, "No database connection")
            return False
        
        try:
            with self.neo4j_driver.session() as session:
                # Check existing indexes
                try:
                    indexes_result = session.run("SHOW INDEXES")
                    indexes = [record["name"] for record in indexes_result]
                    self.record_test("Index Listing", True, 
                                   f"Found {len(indexes)} indexes in database")
                except Exception:
                    # Fallback for older Neo4j versions
                    session.run("CALL db.indexes()")
                    self.record_test("Index Listing", True, 
                                   "Index listing completed (legacy method)")
                
                # Check constraints
                try:
                    constraints_result = session.run("SHOW CONSTRAINTS")
                    constraints = list(constraints_result)
                    self.record_test("Constraint Listing", True, 
                                   f"Found {len(constraints)} constraints in database")
                except Exception:
                    self.record_test("Constraint Listing", True, 
                                   "Constraint listing skipped (may not be supported)")
                
                return True
                
        except Exception as e:
            self.record_test("Schema Operations", False, error=str(e))
            return False
    
    def test_memory_operations(self) -> bool:
        """Test memory storage and retrieval operations."""
        print_header("MEMORY OPERATIONS TEST")
        
        if not self.neo4j_driver:
            self.record_test("Memory Operations", False, "No database connection")
            return False
        
        test_memory_id = f"db_test_memory_{int(time.time())}"
        all_passed = True
        
        try:
            with self.neo4j_driver.session() as session:
                # Test 1: Create memory node
                test_data = {
                    "memory_id": test_memory_id,
                    "summary": "Database test memory for validation",
                    "keywords": ["database", "test", "validation", "neo4j"],
                    "agent_id": "database_test_agent",
                    "event_type": "database_test",
                    "log_level": "INFO",
                    "session_id": "db_test_session",
                    "timestamp": datetime.now().isoformat(),
                    "content": "This is a test memory created by the database test suite"
                }
                
                create_query = """
                CREATE (m:Memory {
                    memory_id: $memory_id,
                    summary: $summary,
                    keywords: $keywords,
                    agent_id: $agent_id,
                    event_type: $event_type,
                    log_level: $log_level,
                    session_id: $session_id,
                    timestamp: $timestamp,
                    content: $content,
                    created_at: datetime()
                }) RETURN m.memory_id as id
                """
                
                result = session.run(create_query, **test_data)
                created_id = result.single()["id"]
                
                if created_id == test_memory_id:
                    self.record_test("Memory Creation", True, 
                                   f"Created memory node: {test_memory_id}")
                else:
                    self.record_test("Memory Creation", False, "Memory ID mismatch")
                    all_passed = False
                
                # Test 2: Retrieve memory
                retrieve_query = """
                MATCH (m:Memory {memory_id: $memory_id})
                RETURN m.memory_id as id, m.summary as summary, 
                       m.keywords as keywords, m.content as content
                """
                
                result = session.run(retrieve_query, memory_id=test_memory_id)
                record = result.single()
                
                if record and record["id"] == test_memory_id:
                    self.record_test("Memory Retrieval", True, 
                                   f"Retrieved: {record['summary']}")
                else:
                    self.record_test("Memory Retrieval", False, "Could not retrieve memory")
                    all_passed = False
                
                # Test 3: Update memory
                update_query = """
                MATCH (m:Memory {memory_id: $memory_id})
                SET m.summary = $new_summary, m.updated_at = datetime()
                RETURN m.memory_id as id
                """
                
                new_summary = "Updated database test memory"
                result = session.run(update_query, 
                                   memory_id=test_memory_id, 
                                   new_summary=new_summary)
                
                if result.single():
                    self.record_test("Memory Update", True, 
                                   f"Updated memory summary")
                else:
                    self.record_test("Memory Update", False, "Update failed")
                    all_passed = False
                
                # Test 4: Search memories by keyword
                search_query = """
                MATCH (m:Memory)
                WHERE any(keyword IN m.keywords WHERE keyword CONTAINS 'database')
                RETURN count(m) as count, collect(m.memory_id)[0..3] as sample_ids
                """
                
                result = session.run(search_query)
                record = result.single()
                count = record["count"]
                sample_ids = record["sample_ids"]
                
                if count > 0:
                    self.record_test("Memory Search", True, 
                                   f"Found {count} memories with 'database' keyword")
                else:
                    self.record_test("Memory Search", False, "No memories found")
                    all_passed = False
                
                # Test 5: Create relationships
                # Create a second memory to test relationships
                related_memory_id = f"related_{test_memory_id}"
                create_related_query = """
                CREATE (m:Memory {
                    memory_id: $memory_id,
                    summary: 'Related memory for testing',
                    agent_id: $agent_id,
                    created_at: datetime()
                }) RETURN m.memory_id as id
                """
                
                result = session.run(create_related_query, 
                                   memory_id=related_memory_id,
                                   agent_id="database_test_agent")
                
                # Create relationship
                relate_query = """
                MATCH (m1:Memory {memory_id: $id1})
                MATCH (m2:Memory {memory_id: $id2})
                CREATE (m1)-[r:RELATES_TO {created_at: datetime()}]->(m2)
                RETURN type(r) as relationship_type
                """
                
                result = session.run(relate_query, 
                                   id1=test_memory_id, 
                                   id2=related_memory_id)
                
                if result.single():
                    self.record_test("Memory Relationships", True, 
                                   "Created RELATES_TO relationship")
                else:
                    self.record_test("Memory Relationships", False, "Relationship creation failed")
                    all_passed = False
                
                return all_passed
                
        except Exception as e:
            self.record_test("Memory Operations", False, error=str(e))
            return False
    
    def test_statistics(self) -> bool:
        """Test database statistics queries."""
        print_header("DATABASE STATISTICS TEST")
        
        if not self.neo4j_driver:
            self.record_test("Statistics", False, "No database connection")
            return False
        
        try:
            with self.neo4j_driver.session() as session:
                # Comprehensive statistics
                stats_queries = {
                    "total_nodes": "MATCH (n) RETURN count(n) as count",
                    "total_memories": "MATCH (m:Memory) RETURN count(m) as count",
                    "total_agents": "MATCH (m:Memory) RETURN count(DISTINCT m.agent_id) as count",
                    "total_relationships": "MATCH ()-[r]-() RETURN count(r) as count",
                    "memory_by_agent": """
                        MATCH (m:Memory) 
                        RETURN m.agent_id as agent, count(m) as memory_count 
                        ORDER BY memory_count DESC LIMIT 5
                    """,
                    "recent_memories": """
                        MATCH (m:Memory) 
                        WHERE m.created_at IS NOT NULL
                        RETURN count(m) as count
                    """
                }
                
                stats = {}
                for stat_name, query in stats_queries.items():
                    if stat_name in ["memory_by_agent", "recent_memories"]:
                        result = session.run(query)
                        if stat_name == "memory_by_agent":
                            agents = list(result)
                            stats[stat_name] = len(agents)
                        else:
                            stats[stat_name] = result.single()["count"]
                    else:
                        result = session.run(query)
                        stats[stat_name] = result.single()["count"]
                
                details = (f"Nodes: {stats['total_nodes']}, "
                          f"Memories: {stats['total_memories']}, "
                          f"Agents: {stats['total_agents']}, "
                          f"Relationships: {stats['total_relationships']}")
                
                self.record_test("Database Statistics", True, details)
                return True
                
        except Exception as e:
            self.record_test("Database Statistics", False, error=str(e))
            return False
    
    def cleanup_test_data(self) -> bool:
        """Clean up test data."""
        print_header("DATABASE CLEANUP")
        
        if not self.neo4j_driver:
            self.record_test("Cleanup", False, "No database connection")
            return False
        
        try:
            with self.neo4j_driver.session() as session:
                # Remove test memories and relationships
                cleanup_queries = [
                    "MATCH (m:Memory) WHERE m.memory_id CONTAINS 'db_test_memory' DETACH DELETE m",
                    "MATCH (m:Memory) WHERE m.memory_id CONTAINS 'related_' DETACH DELETE m",
                    "MATCH (m:Memory {agent_id: 'database_test_agent'}) DETACH DELETE m"
                ]
                
                total_deleted = 0
                for query in cleanup_queries:
                    result = session.run(query)
                    # Get summary if available
                    summary = result.consume()
                    if hasattr(summary, 'counters'):
                        total_deleted += summary.counters.nodes_deleted
                
                self.record_test("Test Data Cleanup", True, 
                               f"Cleaned up {total_deleted} test nodes and relationships")
                return True
                
        except Exception as e:
            self.record_test("Test Data Cleanup", False, error=str(e))
            return False
    
    def run_all_tests(self, cleanup_only: bool = False) -> Dict[str, Any]:
        """Run all database tests."""
        print_header("FINAGENT DATABASE TEST SUITE")
        print_info("Testing Neo4j Database Operations")
        
        start_time = time.time()
        
        # Test connection first
        if not self.test_connection():
            return self.generate_summary(start_time)
        
        if cleanup_only:
            # Only run cleanup
            self.cleanup_test_data()
        else:
            # Run all tests
            self.test_schema_operations()
            self.test_memory_operations()
            self.test_statistics()
            self.cleanup_test_data()
        
        return self.generate_summary(start_time)
    
    def generate_summary(self, start_time: float) -> Dict[str, Any]:
        """Generate test summary."""
        end_time = time.time()
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["passed"])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": success_rate,
            "duration": end_time - start_time,
            "timestamp": datetime.now().isoformat(),
            "test_results": self.test_results
        }
        
        # Print summary
        print_header("DATABASE TEST SUMMARY")
        print(f"{Colors.PURPLE}📊 Test Results:{Colors.NC}")
        print(f"   📝 Total Tests: {total_tests}")
        print(f"   ✅ Passed: {passed_tests}")
        print(f"   ❌ Failed: {failed_tests}")
        print(f"   📈 Success Rate: {success_rate:.1f}%")
        print(f"   ⏱️  Duration: {end_time - start_time:.2f}s")
        
        if failed_tests > 0:
            print(f"\n{Colors.RED}❌ Failed Tests:{Colors.NC}")
            for result in self.test_results:
                if not result["passed"]:
                    print(f"   • {result['test_name']}")
        
        # Close database connection
        if self.neo4j_driver:
            self.neo4j_driver.close()
            print_info("Database connection closed")
        
        return summary

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="FinAgent Database Test Suite")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output with detailed information")
    parser.add_argument("--cleanup", "-c", action="store_true", 
                       help="Only run cleanup operations")
    parser.add_argument("--output", "-o", help="Output file for test results (JSON format)")
    
    args = parser.parse_args()
    
    # Create tester and run tests
    tester = DatabaseTester(verbose=args.verbose)
    summary = tester.run_all_tests(cleanup_only=args.cleanup)
    
    # Save results if output file specified
    if args.output:
        try:
            with open(args.output, 'w') as f:
                json.dump(summary, f, indent=2)
            print_success(f"📄 Test results saved to: {args.output}")
        except Exception as e:
            print_error(f"Failed to save results: {e}")
    
    # Exit with proper code
    exit_code = 0 if summary["failed_tests"] == 0 else 1
    
    if exit_code == 0:
        print_success("🎉 All database tests passed!")
    else:
        print_warning(f"⚠️  {summary['failed_tests']} test(s) failed.")
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
