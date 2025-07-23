#!/usr/bin/env python3
"""
Neo4j Database Setup and Initialization Script for FinAgent Memory System

This script handles the complete setup of Neo4j database for the FinAgent memory system,
including database creation, schema setup, and connection testing.

Features:
- Automated Neo4j database initialization
- Schema creation with proper indexes and constraints
- Connection testing and validation
- Sample data insertion for testing
- Health check endpoints

Usage:
    python scripts/setup_neo4j.py --action [init|test|reset|health]

Author: FinAgent Team
License: Open Source
"""

import os
import sys
import json
import asyncio
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, AuthError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Neo4jDatabaseManager:
    """
    Neo4j Database Manager for FinAgent Memory System.
    
    Handles database initialization, schema setup, and basic operations
    for the memory agent's graph database backend.
    """
    
    def __init__(self, 
                 uri: str = "bolt://localhost:7687",
                 username: str = "neo4j",
                 password: str = "password",
                 database: str = "finagent"):
        """
        Initialize Neo4j database manager.
        
        Args:
            uri: Neo4j database URI
            username: Database username
            password: Database password
            database: Database name
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver = None
        
        if not NEO4J_AVAILABLE:
            logger.error("Neo4j Python driver not available. Install with: pip install neo4j")
            raise ImportError("Neo4j driver not available")
    
    async def connect(self) -> bool:
        """
        Establish connection to Neo4j database.
        
        Returns:
            bool: True if connection successful
        """
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            
            # Test connection
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                
            if test_value == 1:
                logger.info(f"✅ Successfully connected to Neo4j at {self.uri}")
                return True
            else:
                logger.error("❌ Neo4j connection test failed")
                return False
                
        except ServiceUnavailable:
            logger.error(f"❌ Neo4j service not available at {self.uri}")
            return False
        except AuthError:
            logger.error(f"❌ Authentication failed for user {self.username}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            return False
    
    async def initialize_schema(self) -> bool:
        """
        Initialize the database schema with constraints and indexes.
        
        Returns:
            bool: True if schema initialization successful
        """
        if not self.driver:
            logger.error("Database not connected")
            return False
        
        schema_commands = [
            # Node constraints
            "CREATE CONSTRAINT agent_id_unique IF NOT EXISTS FOR (a:Agent) REQUIRE a.agent_id IS UNIQUE",
            "CREATE CONSTRAINT signal_id_unique IF NOT EXISTS FOR (s:Signal) REQUIRE s.signal_id IS UNIQUE",
            "CREATE CONSTRAINT strategy_id_unique IF NOT EXISTS FOR (st:Strategy) REQUIRE st.strategy_id IS UNIQUE",
            "CREATE CONSTRAINT memory_id_unique IF NOT EXISTS FOR (m:Memory) REQUIRE m.memory_id IS UNIQUE",
            
            # Indexes for performance
            "CREATE INDEX agent_type_index IF NOT EXISTS FOR (a:Agent) ON (a.agent_type)",
            "CREATE INDEX signal_timestamp_index IF NOT EXISTS FOR (s:Signal) ON (s.timestamp)",
            "CREATE INDEX strategy_performance_index IF NOT EXISTS FOR (st:Strategy) ON (st.performance_score)",
            "CREATE INDEX memory_keyword_index IF NOT EXISTS FOR (m:Memory) ON (m.keywords)",
            "CREATE INDEX memory_timestamp_index IF NOT EXISTS FOR (m:Memory) ON (m.timestamp)",
            
            # Full-text search indexes
            "CREATE FULLTEXT INDEX memory_content_search IF NOT EXISTS FOR (m:Memory) ON EACH [m.content, m.summary, m.reasoning]",
            "CREATE FULLTEXT INDEX signal_reasoning_search IF NOT EXISTS FOR (s:Signal) ON EACH [s.reasoning, s.market_context]"
        ]
        
        try:
            with self.driver.session(database=self.database) as session:
                for command in schema_commands:
                    try:
                        session.run(command)
                        logger.info(f"✅ Executed: {command}")
                    except Exception as e:
                        if "already exists" in str(e).lower():
                            logger.info(f"⏭️  Already exists: {command}")
                        else:
                            logger.warning(f"⚠️  Failed to execute: {command} - {e}")
            
            logger.info("✅ Database schema initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Schema initialization failed: {e}")
            return False
    
    async def create_sample_data(self) -> bool:
        """
        Create sample data for testing purposes.
        
        Returns:
            bool: True if sample data creation successful
        """
        if not self.driver:
            logger.error("Database not connected")
            return False
        
        sample_data_commands = [
            # Create sample agents
            """
            MERGE (a1:Agent {
                agent_id: 'momentum_agent_001',
                agent_type: 'momentum',
                pool_id: 'alpha_agent_pool',
                status: 'active',
                created_at: datetime(),
                config: '{"window": 20, "strategy_type": "momentum"}'
            })
            """,
            
            """
            MERGE (a2:Agent {
                agent_id: 'mean_reversion_agent_001',
                agent_type: 'mean_reversion',
                pool_id: 'alpha_agent_pool',
                status: 'active',
                created_at: datetime(),
                config: '{"window": 15, "strategy_type": "mean_reversion"}'
            })
            """,
            
            # Create sample strategy
            """
            MERGE (s1:Strategy {
                strategy_id: 'momentum_strategy_20250722',
                agent_id: 'momentum_agent_001',
                strategy_type: 'momentum',
                performance_score: 0.75,
                ic_score: 0.15,
                ir_score: 0.45,
                sharpe_ratio: 1.2,
                created_at: datetime()
            })
            """,
            
            # Create sample memory
            """
            MERGE (m1:Memory {
                memory_id: 'memory_001',
                agent_id: 'momentum_agent_001',
                event_type: 'TRADING_SIGNAL',
                content: 'Generated BUY signal for AAPL with 75% confidence based on momentum analysis',
                summary: 'AAPL BUY signal - momentum strategy',
                keywords: ['trading_signal', 'momentum', 'AAPL', 'BUY'],
                timestamp: datetime(),
                correlation_id: 'test_001'
            })
            """,
            
            # Create relationships
            """
            MATCH (a:Agent {agent_id: 'momentum_agent_001'})
            MATCH (s:Strategy {strategy_id: 'momentum_strategy_20250722'})
            MERGE (a)-[:USES_STRATEGY]->(s)
            """,
            
            """
            MATCH (a:Agent {agent_id: 'momentum_agent_001'})
            MATCH (m:Memory {memory_id: 'memory_001'})
            MERGE (a)-[:CREATED_MEMORY]->(m)
            """
        ]
        
        try:
            with self.driver.session(database=self.database) as session:
                for command in sample_data_commands:
                    session.run(command)
                    
            logger.info("✅ Sample data created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create sample data: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check on the database.
        
        Returns:
            Dict containing health check results
        """
        health_info = {
            "connected": False,
            "database": self.database,
            "uri": self.uri,
            "node_counts": {},
            "relationship_counts": {},
            "indexes": [],
            "constraints": [],
            "last_check": datetime.utcnow().isoformat()
        }
        
        if not self.driver:
            return health_info
        
        try:
            with self.driver.session(database=self.database) as session:
                health_info["connected"] = True
                
                # Get node counts
                node_types = ["Agent", "Strategy", "Signal", "Memory"]
                for node_type in node_types:
                    result = session.run(f"MATCH (n:{node_type}) RETURN count(n) as count")
                    health_info["node_counts"][node_type] = result.single()["count"]
                
                # Get relationship counts
                rel_result = session.run("MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count")
                for record in rel_result:
                    health_info["relationship_counts"][record["rel_type"]] = record["count"]
                
                # Get indexes
                index_result = session.run("SHOW INDEXES")
                health_info["indexes"] = [dict(record) for record in index_result]
                
                # Get constraints
                constraint_result = session.run("SHOW CONSTRAINTS")
                health_info["constraints"] = [dict(record) for record in constraint_result]
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_info["error"] = str(e)
        
        return health_info
    
    async def reset_database(self) -> bool:
        """
        Reset the database by removing all data (use with caution).
        
        Returns:
            bool: True if reset successful
        """
        if not self.driver:
            logger.error("Database not connected")
            return False
        
        try:
            with self.driver.session(database=self.database) as session:
                # Remove all relationships first
                session.run("MATCH ()-[r]->() DELETE r")
                
                # Remove all nodes
                session.run("MATCH (n) DELETE n")
                
            logger.info("✅ Database reset completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database reset failed: {e}")
            return False
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Database connection closed")


async def main():
    """Main function for Neo4j setup operations."""
    parser = argparse.ArgumentParser(description="Neo4j Database Setup for FinAgent")
    parser.add_argument(
        "--action", 
        choices=["init", "test", "reset", "health", "sample"],
        default="init",
        help="Action to perform"
    )
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--username", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="password", help="Neo4j password")
    parser.add_argument("--database", default="finagent", help="Database name")
    
    args = parser.parse_args()
    
    # Get password from environment if available
    password = os.getenv("NEO4J_PASSWORD", args.password)
    
    print("\n" + "="*80)
    print("🗄️  FinAgent Neo4j Database Setup")
    print("="*80)
    
    db_manager = Neo4jDatabaseManager(
        uri=args.uri,
        username=args.username,
        password=password,
        database=args.database
    )
    
    try:
        if args.action == "init":
            logger.info("🚀 Initializing Neo4j database...")
            
            if await db_manager.connect():
                if await db_manager.initialize_schema():
                    logger.info("✅ Database initialization completed successfully")
                else:
                    logger.error("❌ Schema initialization failed")
                    return 1
            else:
                logger.error("❌ Database connection failed")
                return 1
        
        elif args.action == "test":
            logger.info("🧪 Testing Neo4j connection...")
            
            if await db_manager.connect():
                health = await db_manager.health_check()
                print(f"\n📊 Database Health Report:")
                print(f"   Connected: {health['connected']}")
                print(f"   Database: {health['database']}")
                print(f"   Node Counts: {health['node_counts']}")
                print(f"   Relationship Counts: {health['relationship_counts']}")
                print(f"   Indexes: {len(health['indexes'])}")
                print(f"   Constraints: {len(health['constraints'])}")
            else:
                logger.error("❌ Connection test failed")
                return 1
        
        elif args.action == "reset":
            logger.warning("⚠️  Resetting database (all data will be lost)...")
            response = input("Are you sure? Type 'yes' to confirm: ")
            
            if response.lower() == "yes":
                if await db_manager.connect():
                    if await db_manager.reset_database():
                        logger.info("✅ Database reset completed")
                    else:
                        logger.error("❌ Database reset failed")
                        return 1
            else:
                logger.info("Reset cancelled")
        
        elif args.action == "health":
            logger.info("🏥 Performing health check...")
            
            if await db_manager.connect():
                health = await db_manager.health_check()
                print(json.dumps(health, indent=2))
            else:
                logger.error("❌ Health check failed")
                return 1
        
        elif args.action == "sample":
            logger.info("📝 Creating sample data...")
            
            if await db_manager.connect():
                if await db_manager.create_sample_data():
                    logger.info("✅ Sample data created successfully")
                else:
                    logger.error("❌ Sample data creation failed")
                    return 1
    
    finally:
        db_manager.close()
    
    print("\n" + "="*80)
    print("✅ Neo4j setup operation completed!")
    print("="*80)
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
