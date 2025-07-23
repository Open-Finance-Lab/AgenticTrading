#!/usr/bin/env python3
"""
Core Service Manager - Non-LLM Core Services
=============================================

This module manages core FinAgent services that operate independently 
of LLM interactions. It provides:

1. Database-only memory operations
2. Direct MCP protocol services  
3. System health monitoring
4. Configuration management
5. Agent-to-Agent communication

The goal is to isolate LLM usage to specific research and analysis tasks
while keeping core system operations LLM-free for performance and reliability.
"""

import asyncio
import logging
import signal
import sys
from typing import Dict, Any, Optional
from pathlib import Path
import uvicorn
from contextlib import asynccontextmanager

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from FinAgents.memory.configuration_manager import ConfigurationManager
from FinAgents.memory.database_initializer import DatabaseInitializer
from FinAgents.memory.unified_database_manager import create_database_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CoreServiceManager:
    """
    Manages core FinAgent services without LLM dependencies.
    
    This manager handles:
    - Database connections and operations
    - MCP protocol server
    - System health monitoring  
    - Configuration management
    - Service lifecycle management
    """
    
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.database_manager = None
        self.services = {}
        self.running = False
        
    async def initialize_database(self) -> bool:
        """Initialize database connection and schema."""
        try:
            logger.info("🔧 Initializing database connection...")
            
            # Initialize database schema
            db_initializer = DatabaseInitializer(self.config_manager)
            await db_initializer.initialize_database()
            
            # Create database manager
            self.database_manager = await create_database_manager(self.config_manager)
            
            logger.info("✅ Database initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            return False
    
    async def start_memory_server(self) -> bool:
        """Start the core memory server (non-LLM)."""
        try:
            logger.info("🚀 Starting core memory server...")
            
            # Import memory server module
            from FinAgents.memory import memory_server
            
            # Configure server for non-LLM operation
            memory_server.LLM_ENABLED = False
            
            # Get server configuration
            server_config = self.config_manager.get_server_config()
            
            # Start server
            config = uvicorn.Config(
                "FinAgents.memory.memory_server:app",
                host="0.0.0.0", 
                port=server_config.memory_server,
                log_level="info"
            )
            
            self.services['memory_server'] = uvicorn.Server(config)
            logger.info(f"✅ Memory server configured on port {server_config.memory_server}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Memory server startup failed: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        health_status = {
            "timestamp": asyncio.get_event_loop().time(),
            "overall_status": "healthy",
            "services": {}
        }
        
        try:
            # Check database
            if self.database_manager:
                db_health = await self.database_manager.health_check()
                health_status["services"]["database"] = {
                    "status": "healthy" if db_health else "unhealthy",
                    "connected": db_health
                }
            else:
                health_status["services"]["database"] = {
                    "status": "not_initialized",
                    "connected": False
                }
            
            # Check memory server
            memory_status = "running" if 'memory_server' in self.services else "stopped"
            health_status["services"]["memory_server"] = {
                "status": memory_status,
                "port": self.config_manager.get_server_config().memory_server
            }
            
            # Determine overall status
            service_statuses = [svc["status"] for svc in health_status["services"].values()]
            if any(status in ["unhealthy", "not_initialized"] for status in service_statuses):
                health_status["overall_status"] = "degraded"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status["overall_status"] = "error"
            health_status["error"] = str(e)
            return health_status
    
    async def start_all_services(self):
        """Start all core services."""
        logger.info("🚀 Starting FinAgent Core Services (Non-LLM)")
        
        # Initialize database
        if not await self.initialize_database():
            logger.error("❌ Failed to initialize database, aborting startup")
            return False
        
        # Start memory server
        if not await self.start_memory_server():
            logger.error("❌ Failed to start memory server, aborting startup")
            return False
        
        self.running = True
        logger.info("✅ All core services started successfully")
        
        # Print service status
        await self.print_service_status()
        
        return True
    
    async def print_service_status(self):
        """Print current service status."""
        health = await self.health_check()
        
        print("\n" + "="*80)
        print("🎯 FinAgent Core Services Status (Non-LLM)")
        print("="*80)
        
        for service_name, service_info in health["services"].items():
            status_icon = "✅" if service_info["status"] == "healthy" or service_info["status"] == "running" else "❌"
            print(f"   {status_icon} {service_name.title()}: {service_info['status']}")
            
            if "port" in service_info:
                print(f"      Port: {service_info['port']}")
            if "connected" in service_info:
                print(f"      Connected: {service_info['connected']}")
        
        print(f"\n🌍 Overall Status: {health['overall_status'].upper()}")
        print("="*80)
    
    async def stop_all_services(self):
        """Stop all services gracefully."""
        logger.info("🛑 Stopping all core services...")
        
        # Stop memory server
        if 'memory_server' in self.services:
            self.services['memory_server'].should_exit = True
            del self.services['memory_server']
        
        # Close database connections
        if self.database_manager:
            await self.database_manager.close()
            self.database_manager = None
        
        self.running = False
        logger.info("✅ All services stopped")

# Global service manager instance
core_service_manager = CoreServiceManager()

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print(f"\n🛑 Received signal {signum}, initiating shutdown...")
    asyncio.create_task(core_service_manager.stop_all_services())
    sys.exit(0)

async def main():
    """Main entry point for core service manager."""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start all services
        success = await core_service_manager.start_all_services()
        
        if not success:
            logger.error("❌ Failed to start core services")
            return 1
        
        # Run memory server
        if 'memory_server' in core_service_manager.services:
            await core_service_manager.services['memory_server'].serve()
        
        # Keep running
        while core_service_manager.running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ Core service manager error: {e}")
        return 1
    finally:
        await core_service_manager.stop_all_services()
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
