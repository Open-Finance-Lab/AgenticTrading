#!/usr/bin/env python3
"""
FinAgent MCP Server

Dedicated MCP (Model Context Protocol) server implementation for FinAgent Memory operations.
This server focuses exclusively on MCP protocol compliance and tool execution,
using the unified database and interface managers for actual operations.

Features:
- Pure MCP protocol implementation
- Unified architecture integration
- Comprehensive tool definitions
- Enhanced error handling and logging
- Performance monitoring and health checks

Author: FinAgent Team
License: Open Source
"""

# ═══════════════════════════════════════════════════════════════════════════════════
# IMPORTS AND DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════════════

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# MCP server imports
try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️ MCP server not available. Install with: pip install mcp")

# Unified components
try:
    from unified_database_manager import create_database_manager
    from unified_interface_manager import create_interface_manager, UnifiedInterfaceManager
    UNIFIED_COMPONENTS_AVAILABLE = True
except ImportError:
    UNIFIED_COMPONENTS_AVAILABLE = False
    UnifiedInterfaceManager = object  # Fallback for type hints
    print("⚠️ Unified components not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════════

MCP_SERVER_NAME = "FinAgent-MCP-Server"
MCP_SERVER_VERSION = "2.0.0"

DATABASE_CONFIG = {
    "uri": "bolt://localhost:7687",
    "username": "neo4j", 
    "password": "FinOrchestration",
    "database": "neo4j"
}

# Global components
interface_manager = None

# ═══════════════════════════════════════════════════════════════════════════════════
# MCP SERVER LIFECYCLE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════════

async def initialize_mcp_server() -> bool:
    """Initialize the MCP server with unified architecture."""
    global interface_manager
    
    try:
        if not UNIFIED_COMPONENTS_AVAILABLE:
            logger.error("❌ Unified components not available for MCP server")
            return False
        
        logger.info("🚀 Initializing FinAgent MCP Server...")
        
        # Create and initialize interface manager
        interface_manager = create_interface_manager(DATABASE_CONFIG)
        
        if not await interface_manager.initialize():
            logger.error("❌ Failed to initialize interface manager")
            return False
        
        logger.info("✅ FinAgent MCP Server initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ MCP server initialization failed: {e}")
        return False

async def shutdown_mcp_server():
    """Shutdown the MCP server gracefully."""
    try:
        if interface_manager:
            await interface_manager.shutdown()
        logger.info("✅ FinAgent MCP Server shutdown complete")
    except Exception as e:
        logger.error(f"❌ MCP server shutdown error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════════
# MCP SERVER IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════════

if MCP_AVAILABLE and UNIFIED_COMPONENTS_AVAILABLE:
    from contextlib import asynccontextmanager
    from collections.abc import AsyncIterator
    
    @asynccontextmanager
    async def mcp_lifespan(server: FastMCP) -> AsyncIterator[None]:
        """MCP server lifecycle management."""
        logger.info("🚀 [MCP-SERVER] Starting FinAgent MCP Server...")
        
        if await initialize_mcp_server():
            logger.info("✅ [MCP-SERVER] Server startup complete")
            try:
                yield
            finally:
                logger.info("🛑 [MCP-SERVER] Shutting down...")
                await shutdown_mcp_server()
        else:
            logger.error("❌ [MCP-SERVER] Server initialization failed")
            yield
    
    # Create MCP server instance
    mcp_server = FastMCP(
        MCP_SERVER_NAME,
        lifespan=mcp_lifespan,
        stateless_http=True,
        debug=True
    )
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # MCP TOOL DEFINITIONS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @mcp_server.tool(
        name="store_memory",
        description="Store a memory record with intelligent linking and semantic indexing"
    )
    async def mcp_store_memory(
        query: str,
        keywords: List[str],
        summary: str,
        agent_id: str,
        event_type: str = "USER_QUERY",
        log_level: str = "INFO",
        session_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """MCP tool for storing memories."""
        if not interface_manager:
            return json.dumps({"status": "error", "message": "Interface manager not initialized"})
        
        try:
            result = await interface_manager.execute_tool("store_graph_memory", {
                "query": query,
                "keywords": keywords,
                "summary": summary,
                "agent_id": agent_id,
                "event_type": event_type,
                "log_level": log_level,
                "session_id": session_id,
                "correlation_id": correlation_id
            })
            
            return json.dumps(result)
            
        except Exception as e:
            logger.error(f"❌ MCP store_memory failed: {e}")
            return json.dumps({"status": "error", "message": str(e)})
    
    @mcp_server.tool(
        name="retrieve_memory",
        description="Retrieve memories using enhanced search capabilities"
    )
    async def mcp_retrieve_memory(
        search_query: str,
        limit: int = 5
    ) -> str:
        """MCP tool for retrieving memories."""
        if not interface_manager:
            return json.dumps({"status": "error", "message": "Interface manager not initialized"})
        
        try:
            result = await interface_manager.execute_tool("retrieve_graph_memory", {
                "search_query": search_query,
                "limit": limit
            })
            
            return json.dumps(result)
            
        except Exception as e:
            logger.error(f"❌ MCP retrieve_memory failed: {e}")
            return json.dumps({"status": "error", "message": str(e)})
    
    @mcp_server.tool(
        name="semantic_search",
        description="Perform AI-powered semantic search across memories"
    )
    async def mcp_semantic_search(
        query: str,
        limit: int = 10,
        similarity_threshold: float = 0.3
    ) -> str:
        """MCP tool for semantic search."""
        if not interface_manager:
            return json.dumps({"status": "error", "message": "Interface manager not initialized"})
        
        try:
            result = await interface_manager.execute_tool("semantic_search_memories", {
                "query": query,
                "limit": limit,
                "similarity_threshold": similarity_threshold
            })
            
            return json.dumps(result)
            
        except Exception as e:
            logger.error(f"❌ MCP semantic_search failed: {e}")
            return json.dumps({"status": "error", "message": str(e)})
    
    @mcp_server.tool(
        name="get_statistics",
        description="Get comprehensive system statistics and health information"
    )
    async def mcp_get_statistics() -> str:
        """MCP tool for getting statistics."""
        if not interface_manager:
            return json.dumps({"status": "error", "message": "Interface manager not initialized"})
        
        try:
            result = await interface_manager.execute_tool("get_graph_memory_statistics", {})
            return json.dumps(result)
            
        except Exception as e:
            logger.error(f"❌ MCP get_statistics failed: {e}")
            return json.dumps({"status": "error", "message": str(e)})
    
    @mcp_server.tool(
        name="health_check",
        description="Perform comprehensive health check of all system components"
    )
    async def mcp_health_check() -> str:
        """MCP tool for health checking."""
        if not interface_manager:
            return json.dumps({"status": "error", "message": "Interface manager not initialized"})
        
        try:
            result = await interface_manager.execute_tool("health_check", {})
            return json.dumps(result)
            
        except Exception as e:
            logger.error(f"❌ MCP health_check failed: {e}")
            return json.dumps({"status": "error", "message": str(e)})
    
    @mcp_server.tool(
        name="create_relationship", 
        description="Create intelligent relationships between memory nodes"
    )
    async def mcp_create_relationship(
        source_memory_id: str,
        target_memory_id: str,
        relationship_type: str
    ) -> str:
        """MCP tool for creating relationships."""
        if not interface_manager:
            return json.dumps({"status": "error", "message": "Interface manager not initialized"})
        
        try:
            result = await interface_manager.execute_tool("create_relationship", {
                "source_memory_id": source_memory_id,
                "target_memory_id": target_memory_id,
                "relationship_type": relationship_type
            })
            
            return json.dumps(result)
            
        except Exception as e:
            logger.error(f"❌ MCP create_relationship failed: {e}")
            return json.dumps({"status": "error", "message": str(e)})
    
    # Create app instance
    app = mcp_server.streamable_http_app()
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # SERVER INFORMATION AND UTILITIES
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def print_mcp_server_info():
        """Print MCP server information."""
        print("\n" + "="*80)
        print("🚀 FINAGENT MCP SERVER - DEDICATED IMPLEMENTATION")
        print("="*80)
        print(f"📋 Server Info:")
        print(f"   🏷️  Name: {MCP_SERVER_NAME}")
        print(f"   📦 Version: {MCP_SERVER_VERSION}")
        print(f"   🔧 Protocol: Model Context Protocol (MCP)")
        print(f"   🗄️  Architecture: Unified Components")
        print("\n📡 Available MCP Tools:")
        print("   • store_memory - Enhanced memory storage")
        print("   • retrieve_memory - Intelligent memory retrieval")
        print("   • semantic_search - AI-powered semantic search")
        print("   • get_statistics - Comprehensive analytics")
        print("   • health_check - System health monitoring")
        print("   • create_relationship - Relationship management")
        print("\n🔧 Server Configuration:")
        print(f"   📍 Database URI: {DATABASE_CONFIG['uri']}")
        print(f"   👤 Database User: {DATABASE_CONFIG['username']}")
        print(f"   🗄️  Database Name: {DATABASE_CONFIG['database']}")
        print("="*80)

else:
    def print_mcp_server_info():
        print("❌ MCP Server not available - missing dependencies")
        print("   Install with: pip install mcp")
    
    app = None

# ═══════════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_mcp_server_info()
    if app:
        print("\n🚀 Starting FinAgent MCP Server...")
        print("   Use: uvicorn mcp_server:app --host 0.0.0.0 --port 8001")
        print("="*80)
    else:
        print("\n❌ Cannot start MCP server - dependencies missing")
        print("="*80)
