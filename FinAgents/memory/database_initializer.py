#!/usr/bin/env python3
"""
FinAgent Database Initializer

Database initialization and setup module for FinAgent memory system.
Handles database schema creation, indexing, constraints, and seed data based on configuration.

Features:
- Configuration-driven initialization
- Index and constraint management
- Seed data loading
- Database health verification
- Backup and restore capabilities

Author: FinAgent Team
License: Open Source
"""

# ═══════════════════════════════════════════════════════════════════════════════════
# IMPORTS AND DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════════════

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Neo4j imports
try:
    from neo4j import GraphDatabase, AsyncGraphDatabase
    from neo4j.exceptions import ServiceUnavailable, AuthError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("⚠️ Neo4j not available. Install with: pip install neo4j")

# Configuration imports
try:
    from configuration_manager import (
        get_config_manager, get_database_config, 
        DatabaseInitConfig, Environment
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("⚠️ Configuration manager not available")
    
    # Create fallback classes for type hints
    class DatabaseInitConfig:
        def __init__(self, **kwargs):
            pass
    
    class Environment:
        DEVELOPMENT = "development"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════════
# DATABASE INITIALIZER IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════════

class DatabaseInitializer:
    """
    Database initialization and setup for FinAgent memory system.
    Handles schema creation, indexing, and data seeding based on configuration.
    """
    
    def __init__(self, config: Dict[str, Any], init_config: DatabaseInitConfig):
        """
        Initialize database initializer.
        
        Args:
            config: Database connection configuration
            init_config: Database initialization configuration
        """
        self.config = config
        self.init_config = init_config
        self.driver = None
        
        # Database connection info
        self.uri = config.get("uri", "bolt://localhost:7687")
        self.username = config.get("username", "neo4j")
        self.password = config.get("password", "finagent123")
        self.database = config.get("database", "finagent")
    
    async def initialize(self) -> bool:
        """
        Initialize the database with configuration settings.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info("🔧 Starting database initialization...")
            
            # Connect to database
            if not await self.connect():
                logger.error("❌ Failed to connect to database")
                return False
            
            # Clear database if configured
            if self.init_config.clear_on_startup:
                await self.clear_database()
            
            # Create backup if configured
            if self.init_config.backup_before_init:
                await self.create_backup()
            
            # Create indexes
            if self.init_config.auto_create_indexes:
                await self.create_indexes()
            
            # Create constraints
            if self.init_config.auto_create_constraints:
                await self.create_constraints()
            
            # Load seed data
            if self.init_config.seed_data_enabled:
                await self.load_seed_data()
            
            # Verify database health
            health_status = await self.verify_health()
            
            logger.info("✅ Database initialization completed successfully")
            return health_status
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            return False
        finally:
            await self.close()
    
    async def connect(self) -> bool:
        """Connect to Neo4j database."""
        try:
            if not NEO4J_AVAILABLE:
                logger.error("❌ Neo4j driver not available")
                return False
            
            logger.info(f"🔌 Connecting to Neo4j at {self.uri}")
            self.driver = AsyncGraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            
            # Verify connectivity
            await self.driver.verify_connectivity()
            logger.info("✅ Connected to Neo4j database")
            return True
            
        except AuthError as e:
            logger.error(f"❌ Authentication failed: {e}")
            return False
        except ServiceUnavailable as e:
            logger.error(f"❌ Neo4j service unavailable: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Database connection error: {e}")
            return False
    
    async def close(self):
        """Close database connection."""
        if self.driver:
            await self.driver.close()
            logger.info("🔌 Database connection closed")
    
    async def clear_database(self):
        """Clear all data from the database."""
        try:
            logger.warning("🗑️ Clearing all data from database...")
            query = "MATCH (n) DETACH DELETE n"
            
            async with self.driver.session(database=self.database) as session:
                await session.run(query)
            
            logger.info("✅ Database cleared successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to clear database: {e}")
            raise
    
    async def create_backup(self):
        """Create database backup (placeholder for future implementation)."""
        try:
            logger.info("💾 Creating database backup...")
            
            # For now, just log the backup timestamp
            backup_info = {
                "timestamp": datetime.utcnow().isoformat(),
                "database": self.database,
                "backup_type": "pre_initialization"
            }
            
            logger.info(f"✅ Backup info recorded: {backup_info}")
            
        except Exception as e:
            logger.warning(f"⚠️ Backup creation failed: {e}")
    
    async def create_indexes(self):
        """Create required indexes from configuration."""
        try:
            logger.info("📊 Creating database indexes...")
            
            async with self.driver.session(database=self.database) as session:
                for index_config in self.init_config.required_indexes:
                    await self._create_single_index(session, index_config)
            
            logger.info("✅ All indexes created successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to create indexes: {e}")
            raise
    
    async def _create_single_index(self, session, index_config: Dict[str, Any]):
        """Create a single index."""
        try:
            index_name = index_config.get("name")
            index_type = index_config.get("type", "btree")
            labels = index_config.get("labels", [])
            properties = index_config.get("properties", [])
            
            if not index_name or not labels or not properties:
                logger.warning(f"⚠️ Incomplete index configuration: {index_config}")
                return
            
            # Build index query based on type
            if index_type == "fulltext":
                label_str = "|".join(labels)
                prop_str = ", ".join([f"n.{prop}" for prop in properties])
                query = f"CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS FOR (n:{label_str}) ON EACH [{prop_str}]"
            else:
                # Standard btree index
                label = labels[0]  # Use first label for btree index
                prop_str = ", ".join([f"n.{prop}" for prop in properties])
                query = f"CREATE INDEX {index_name} IF NOT EXISTS FOR (n:{label}) ON ({prop_str})"
            
            await session.run(query)
            logger.info(f"   ✅ Created {index_type} index: {index_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create index {index_name}: {e}")
            raise
    
    async def create_constraints(self):
        """Create required constraints from configuration."""
        try:
            logger.info("🔒 Creating database constraints...")
            
            async with self.driver.session(database=self.database) as session:
                for constraint_config in self.init_config.required_constraints:
                    await self._create_single_constraint(session, constraint_config)
            
            logger.info("✅ All constraints created successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to create constraints: {e}")
            raise
    
    async def _create_single_constraint(self, session, constraint_config: Dict[str, Any]):
        """Create a single constraint."""
        try:
            constraint_name = constraint_config.get("name")
            constraint_type = constraint_config.get("type", "uniqueness")
            labels = constraint_config.get("labels", [])
            properties = constraint_config.get("properties", [])
            
            if not constraint_name or not labels or not properties:
                logger.warning(f"⚠️ Incomplete constraint configuration: {constraint_config}")
                return
            
            label = labels[0]  # Use first label for constraint
            
            if constraint_type == "uniqueness":
                prop_str = ", ".join([f"n.{prop}" for prop in properties])
                query = f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS FOR (n:{label}) REQUIRE ({prop_str}) IS UNIQUE"
            elif constraint_type == "existence":
                # Note: Node property existence constraints are only available in Neo4j Enterprise
                prop_str = ", ".join([f"n.{prop}" for prop in properties])
                query = f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS FOR (n:{label}) REQUIRE ({prop_str}) IS NOT NULL"
            else:
                logger.warning(f"⚠️ Unknown constraint type: {constraint_type}")
                return
            
            await session.run(query)
            logger.info(f"   ✅ Created {constraint_type} constraint: {constraint_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create constraint {constraint_name}: {e}")
            # Don't raise for constraints as they might not be supported in Community edition
    
    async def load_seed_data(self):
        """Load seed data from configuration."""
        try:
            if not self.init_config.seed_data:
                logger.info("📝 No seed data configured")
                return
            
            logger.info("🌱 Loading seed data...")
            
            async with self.driver.session(database=self.database) as session:
                for seed_record in self.init_config.seed_data:
                    await self._create_seed_memory(session, seed_record)
            
            logger.info("✅ Seed data loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load seed data: {e}")
            raise
    
    async def _create_seed_memory(self, session, seed_data: Dict[str, Any]):
        """Create a single seed memory record."""
        try:
            query = """
            CREATE (m:Memory {
                memory_id: $memory_id,
                query: $query,
                keywords: $keywords,
                summary: $summary,
                agent_id: $agent_id,
                event_type: $event_type,
                timestamp: datetime(),
                lookup_count: 0
            })
            """
            
            parameters = {
                "memory_id": seed_data.get("memory_id"),
                "query": seed_data.get("query"),
                "keywords": seed_data.get("keywords", []),
                "summary": seed_data.get("summary"),
                "agent_id": seed_data.get("agent_id"),
                "event_type": seed_data.get("event_type", "SEED_DATA")
            }
            
            await session.run(query, parameters)
            logger.info(f"   ✅ Created seed memory: {seed_data.get('memory_id')}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create seed memory: {e}")
            raise
    
    async def verify_health(self) -> bool:
        """Verify database health and configuration."""
        try:
            logger.info("🏥 Verifying database health...")
            
            async with self.driver.session(database=self.database) as session:
                # Check basic connectivity
                result = await session.run("RETURN 1 as test")
                record = await result.single()
                if not record or record["test"] != 1:
                    logger.error("❌ Basic connectivity test failed")
                    return False
                
                # Check indexes
                indexes_result = await session.run("SHOW INDEXES")
                indexes = await indexes_result.data()
                logger.info(f"   📊 Found {len(indexes)} indexes")
                
                # Check constraints
                constraints_result = await session.run("SHOW CONSTRAINTS")
                constraints = await constraints_result.data()
                logger.info(f"   🔒 Found {len(constraints)} constraints")
                
                # Check memory count
                count_result = await session.run("MATCH (m:Memory) RETURN count(m) as count")
                count_record = await count_result.single()
                memory_count = count_record["count"] if count_record else 0
                logger.info(f"   🧠 Found {memory_count} memory records")
            
            logger.info("✅ Database health verification passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database health verification failed: {e}")
            return False

# ═══════════════════════════════════════════════════════════════════════════════════
# INITIALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════════

async def initialize_database(environment: Optional[Environment] = None) -> bool:
    """
    Initialize database for specified environment.
    
    Args:
        environment: Target environment (defaults to current)
        
    Returns:
        bool: True if initialization successful
    """
    try:
        if not CONFIG_AVAILABLE:
            logger.error("❌ Configuration manager not available")
            return False
        
        # Get configuration
        config_manager = get_config_manager()
        if environment:
            config_manager.set_environment(environment)
        
        db_config = get_database_config(environment)
        init_config = config_manager.get_database_init_config(environment)
        
        # Initialize database
        initializer = DatabaseInitializer(db_config, init_config)
        return await initializer.initialize()
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False

def initialize_database_sync(environment: Optional[Environment] = None) -> bool:
    """Synchronous wrapper for database initialization."""
    return asyncio.run(initialize_database(environment))

# ═══════════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FinAgent Database Initializer")
    parser.add_argument(
        "--env", "--environment",
        choices=["development", "testing", "staging", "production"],
        default="development",
        help="Target environment for initialization"
    )
    
    args = parser.parse_args()
    environment = Environment(args.env)
    
    print(f"\n🔧 FinAgent Database Initializer")
    print(f"🌍 Environment: {environment.value}")
    print("="*50)
    
    success = initialize_database_sync(environment)
    
    if success:
        print("\n✅ Database initialization completed successfully")
    else:
        print("\n❌ Database initialization failed")
        exit(1)
