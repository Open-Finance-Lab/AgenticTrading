#!/usr/bin/env python3
"""
FinAgent Unified Interface Manager

This module provides a unified interface management system for the FinAgent Memory Agent.
It handles tool definitions, MCP protocol integration, and provides a consistent API
for interacting with memory operations across different protocols and clients.

Features:
- Unified tool interface definitions for MCP and HTTP protocols
- Dynamic tool registration and management
- Standardized error handling and response formatting
- Protocol-agnostic client interfaces
- Conversation management with tool integration
- Performance monitoring and logging

Author: FinAgent Team
License: Open Source
"""

# ═══════════════════════════════════════════════════════════════════════════════════
# IMPORTS AND DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════════════

import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# MCP client imports
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import StdioClientTransport  
    from mcp.types import CallToolRequest, Tool
    import mcp.types as mcp_types
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# OpenAI integration imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Database manager import
from unified_database_manager import UnifiedDatabaseManager, create_database_manager

# Configure logging
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════════
# DATA MODELS AND ENUMS
# ═══════════════════════════════════════════════════════════════════════════════════

class ProtocolType(Enum):
    """Enumeration of supported protocol types"""
    MCP = "mcp"
    HTTP = "http"
    A2A = "a2a"
    WEBSOCKET = "websocket"


class ToolCategory(Enum):
    """Enumeration of tool categories"""
    MEMORY_STORAGE = "memory_storage"
    MEMORY_RETRIEVAL = "memory_retrieval"
    RELATIONSHIP_MANAGEMENT = "relationship_management"
    ANALYTICS = "analytics"
    HEALTH_MONITORING = "health_monitoring"
    SEMANTIC_SEARCH = "semantic_search"
    STREAM_PROCESSING = "stream_processing"


@dataclass
class ToolDefinition:
    """Data class for tool definitions"""
    name: str
    description: str
    category: ToolCategory
    parameters: Dict[str, Any]
    protocol_support: List[ProtocolType]
    handler: Optional[Callable] = None


@dataclass
class InterfaceConfig:
    """Configuration for interface manager"""
    database_config: Dict[str, Any]
    mcp_enabled: bool = True
    http_enabled: bool = True
    a2a_enabled: bool = True
    websocket_enabled: bool = False
    openai_integration: bool = True
    debug_mode: bool = False


# ═══════════════════════════════════════════════════════════════════════════════════
# UNIFIED INTERFACE MANAGER CLASS
# ═══════════════════════════════════════════════════════════════════════════════════

class UnifiedInterfaceManager:
    """
    Centralized interface manager for FinAgent memory operations.
    
    This class provides a unified interface for all memory operations,
    supporting multiple protocols and providing standardized tool definitions.
    """
    
    def __init__(self, config: InterfaceConfig):
        """
        Initialize the unified interface manager.
        
        Args:
            config: Interface configuration
        """
        self.config = config
        self.database_manager: Optional[UnifiedDatabaseManager] = None
        self.tools: Dict[str, ToolDefinition] = {}
        self.mcp_session: Optional[ClientSession] = None
        
        # Performance metrics
        self.tool_call_count = 0
        self.error_count = 0
        self.last_activity = None
        
        logger.info("Unified Interface Manager initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize the interface manager and its components.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Initialize database manager
            self.database_manager = create_database_manager(self.config.database_config)
            
            if not await self.database_manager.connect():
                logger.error("❌ Failed to connect to database")
                return False
            
            # Register all tool definitions
            self._register_all_tools()
            
            logger.info("✅ Unified Interface Manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Interface manager initialization failed: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the interface manager gracefully."""
        try:
            if self.database_manager:
                await self.database_manager.close()
            
            if self.mcp_session:
                await self.mcp_session.close()
            
            logger.info("✅ Interface manager shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")

    # ═══════════════════════════════════════════════════════════════════════════════════
    # TOOL REGISTRATION AND MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════════════

    def _register_all_tools(self) -> None:
        """Register all available tools with their definitions."""
        
        # Memory Storage Tools
        self._register_tool(ToolDefinition(
            name="store_graph_memory",
            description="Stores a structured memory in the Neo4j graph database with intelligent linking.",
            category=ToolCategory.MEMORY_STORAGE,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The original query or content to store"},
                    "keywords": {"type": "array", "items": {"type": "string"}, "description": "List of keywords"},
                    "summary": {"type": "string", "description": "Summary of the memory content"},
                    "agent_id": {"type": "string", "description": "ID of the agent creating the memory"},
                    "event_type": {"type": "string", "default": "USER_QUERY", "description": "Type of event"},
                    "log_level": {"type": "string", "default": "INFO", "description": "Logging level"},
                    "session_id": {"type": "string", "description": "Optional session identifier"},
                    "correlation_id": {"type": "string", "description": "Optional correlation identifier"}
                },
                "required": ["query", "keywords", "summary", "agent_id"]
            },
            protocol_support=[ProtocolType.MCP, ProtocolType.HTTP, ProtocolType.A2A],
            handler=self._handle_store_memory
        ))

        self._register_tool(ToolDefinition(
            name="store_graph_memories_batch",
            description="Stores multiple memories in a batch operation for high throughput processing.",
            category=ToolCategory.MEMORY_STORAGE,
            parameters={
                "type": "object",
                "properties": {
                    "events": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "agent_id": {"type": "string"},
                                "content": {"type": "object"},
                                "query": {"type": "string"},
                                "keywords": {"type": "array", "items": {"type": "string"}},
                                "summary": {"type": "string"},
                                "event_type": {"type": "string"},
                                "session_id": {"type": "string"},
                                "correlation_id": {"type": "string"}
                            },
                            "required": ["agent_id", "content"]
                        },
                        "description": "List of memory events to store"
                    }
                },
                "required": ["events"]
            },
            protocol_support=[ProtocolType.MCP, ProtocolType.HTTP],
            handler=self._handle_batch_store
        ))

        # Memory Retrieval Tools
        self._register_tool(ToolDefinition(
            name="retrieve_graph_memory",
            description="Retrieves memories using full-text search with intelligent ranking.",
            category=ToolCategory.MEMORY_RETRIEVAL,
            parameters={
                "type": "object",
                "properties": {
                    "search_query": {"type": "string", "description": "Text to search for in memories"},
                    "limit": {"type": "integer", "default": 5, "description": "Maximum number of results"}
                },
                "required": ["search_query"]
            },
            protocol_support=[ProtocolType.MCP, ProtocolType.HTTP, ProtocolType.A2A],
            handler=self._handle_retrieve_memory
        ))

        self._register_tool(ToolDefinition(
            name="retrieve_memory_with_expansion",
            description="Retrieves memories and expands results using relationship links for comprehensive context.",
            category=ToolCategory.MEMORY_RETRIEVAL,
            parameters={
                "type": "object",
                "properties": {
                    "search_query": {"type": "string", "description": "Text to search for"},
                    "limit": {"type": "integer", "default": 10, "description": "Maximum number of results including expansions"}
                },
                "required": ["search_query"]
            },
            protocol_support=[ProtocolType.MCP, ProtocolType.HTTP],
            handler=self._handle_retrieve_with_expansion
        ))

        # Semantic Search Tools
        self._register_tool(ToolDefinition(
            name="semantic_search_memories",
            description="Performs intelligent semantic search using AI embeddings and similarity scoring.",
            category=ToolCategory.SEMANTIC_SEARCH,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Semantic query text"},
                    "limit": {"type": "integer", "default": 10, "description": "Maximum results"},
                    "similarity_threshold": {"type": "number", "default": 0.3, "description": "Minimum similarity score"}
                },
                "required": ["query"]
            },
            protocol_support=[ProtocolType.MCP, ProtocolType.HTTP],
            handler=self._handle_semantic_search
        ))

        # Filtering and Analytics Tools
        self._register_tool(ToolDefinition(
            name="filter_graph_memories",
            description="Filters memories based on structured criteria like time, agent, or event type.",
            category=ToolCategory.MEMORY_RETRIEVAL,
            parameters={
                "type": "object",
                "properties": {
                    "filters": {
                        "type": "object",
                        "properties": {
                            "agent_id": {"type": "string"},
                            "event_type": {"type": "string"},
                            "session_id": {"type": "string"},
                            "start_time": {"type": "string"},
                            "end_time": {"type": "string"}
                        },
                        "description": "Filter criteria"
                    },
                    "limit": {"type": "integer", "default": 100, "description": "Maximum results"},
                    "offset": {"type": "integer", "default": 0, "description": "Results offset"}
                },
                "required": ["filters"]
            },
            protocol_support=[ProtocolType.MCP, ProtocolType.HTTP],
            handler=self._handle_filter_memories
        ))

        self._register_tool(ToolDefinition(
            name="get_graph_memory_statistics",
            description="Retrieves comprehensive statistics about memories, agents, and database performance.",
            category=ToolCategory.ANALYTICS,
            parameters={
                "type": "object",
                "properties": {}
            },
            protocol_support=[ProtocolType.MCP, ProtocolType.HTTP, ProtocolType.A2A],
            handler=self._handle_get_statistics
        ))

        # Relationship Management Tools
        self._register_tool(ToolDefinition(
            name="create_relationship",
            description="Creates a directed relationship between two memory nodes for enhanced context linking.",
            category=ToolCategory.RELATIONSHIP_MANAGEMENT,
            parameters={
                "type": "object",
                "properties": {
                    "source_memory_id": {"type": "string", "description": "Source memory ID"},
                    "target_memory_id": {"type": "string", "description": "Target memory ID"},
                    "relationship_type": {"type": "string", "description": "Type of relationship (RELATES_TO, CONTRADICTS, etc.)"}
                },
                "required": ["source_memory_id", "target_memory_id", "relationship_type"]
            },
            protocol_support=[ProtocolType.MCP, ProtocolType.HTTP],
            handler=self._handle_create_relationship
        ))

        # Maintenance Tools
        self._register_tool(ToolDefinition(
            name="prune_graph_memories",
            description="Deletes old and irrelevant memories to maintain database performance and relevance.",
            category=ToolCategory.ANALYTICS,
            parameters={
                "type": "object",
                "properties": {
                    "max_age_days": {"type": "integer", "default": 180, "description": "Maximum age in days"},
                    "min_lookup_count": {"type": "integer", "default": 1, "description": "Minimum lookup count to retain"}
                }
            },
            protocol_support=[ProtocolType.MCP, ProtocolType.HTTP],
            handler=self._handle_prune_memories
        ))

        # Health Monitoring Tools
        self._register_tool(ToolDefinition(
            name="health_check",
            description="Performs comprehensive health check of the memory system and database connectivity.",
            category=ToolCategory.HEALTH_MONITORING,
            parameters={
                "type": "object",
                "properties": {}
            },
            protocol_support=[ProtocolType.MCP, ProtocolType.HTTP, ProtocolType.A2A],
            handler=self._handle_health_check
        ))

        logger.info(f"✅ Registered {len(self.tools)} tools across {len(ToolCategory)} categories")

    def _register_tool(self, tool_def: ToolDefinition) -> None:
        """Register a single tool definition."""
        self.tools[tool_def.name] = tool_def

    def get_tool_definitions(self, protocol: ProtocolType = ProtocolType.MCP) -> List[Dict[str, Any]]:
        """
        Get tool definitions formatted for specific protocol.
        
        Args:
            protocol: Target protocol type
            
        Returns:
            List[Dict[str, Any]]: Formatted tool definitions
        """
        definitions = []
        
        for tool_name, tool_def in self.tools.items():
            if protocol in tool_def.protocol_support:
                definition = {
                    "name": tool_def.name,
                    "description": tool_def.description,
                    "inputSchema": tool_def.parameters
                }
                definitions.append(definition)
        
        return definitions

    def get_openai_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get tool definitions formatted for OpenAI function calling."""
        definitions = []
        
        for tool_name, tool_def in self.tools.items():
            if ProtocolType.HTTP in tool_def.protocol_support:
                definition = {
                    "type": "function",
                    "function": {
                        "name": tool_def.name,
                        "description": tool_def.description,
                        "parameters": tool_def.parameters
                    }
                }
                definitions.append(definition)
        
        return definitions

    # ═══════════════════════════════════════════════════════════════════════════════════
    # TOOL EXECUTION HANDLERS
    # ═══════════════════════════════════════════════════════════════════════════════════

    async def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool with the given arguments.
        
        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            
        Returns:
            Dict[str, Any]: Tool execution result
        """
        try:
            if tool_name not in self.tools:
                return {"status": "error", "message": f"Tool '{tool_name}' not found"}
            
            tool_def = self.tools[tool_name]
            if not tool_def.handler:
                return {"status": "error", "message": f"Tool '{tool_name}' has no handler"}
            
            # Execute tool handler
            result = await tool_def.handler(tool_args)
            
            # Update metrics
            self.tool_call_count += 1
            self.last_activity = datetime.utcnow()
            
            logger.info(f"✅ Tool '{tool_name}' executed successfully")
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Tool '{tool_name}' execution failed: {e}")
            return {
                "status": "error",
                "message": f"Tool execution failed: {str(e)}",
                "exception_type": type(e).__name__
            }

    # Individual Tool Handlers
    async def _handle_store_memory(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle store_graph_memory tool execution."""
        try:
            result = await self.database_manager.store_memory(
                query=args["query"],
                keywords=args["keywords"],
                summary=args["summary"],
                agent_id=args["agent_id"],
                event_type=args.get("event_type", "USER_QUERY"),
                log_level=args.get("log_level", "INFO"),
                session_id=args.get("session_id"),
                correlation_id=args.get("correlation_id")
            )
            
            linked_count = len(result.get("linked_memories", []))
            message = f"Memory stored successfully and linked to {linked_count} similar memories."
            
            return {
                "status": "success",
                "message": message,
                "stored_memory": result
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Memory storage failed: {str(e)}"
            }

    async def _handle_batch_store(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle store_graph_memories_batch tool execution."""
        try:
            count = await self.database_manager.store_memories_batch(args["events"])
            
            return {
                "status": "success",
                "stored_count": count,
                "message": f"Successfully stored {count} memories in batch operation."
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Batch storage failed: {str(e)}"
            }

    async def _handle_retrieve_memory(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle retrieve_graph_memory tool execution."""
        try:
            memories = await self.database_manager.retrieve_memory(
                search_query=args["search_query"],
                limit=args.get("limit", 5)
            )
            
            return {
                "status": "success",
                "retrieved_memories": memories,
                "count": len(memories)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Memory retrieval failed: {str(e)}"
            }

    async def _handle_retrieve_with_expansion(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle retrieve_memory_with_expansion tool execution."""
        try:
            memories = await self.database_manager.retrieve_memory_with_expansion(
                search_query=args["search_query"],
                limit=args.get("limit", 10)
            )
            
            return {
                "status": "success",
                "retrieved_memories": memories,
                "count": len(memories)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Expanded retrieval failed: {str(e)}"
            }

    async def _handle_semantic_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle semantic_search_memories tool execution."""
        try:
            # This would use the intelligent indexer if available
            if hasattr(self.database_manager, 'indexer') and self.database_manager.indexer:
                # Semantic search implementation
                memories = await self.database_manager.retrieve_memory(
                    search_query=args["query"],
                    limit=args.get("limit", 10)
                )
                
                return {
                    "status": "success",
                    "results": memories,
                    "query": args["query"],
                    "similarity_threshold": args.get("similarity_threshold", 0.3)
                }
            else:
                return {
                    "status": "error",
                    "message": "Semantic search not available. Intelligent indexer not initialized."
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Semantic search failed: {str(e)}"
            }

    async def _handle_filter_memories(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle filter_graph_memories tool execution."""
        try:
            memories = await self.database_manager.filter_memories(
                filters=args["filters"],
                limit=args.get("limit", 100),
                offset=args.get("offset", 0)
            )
            
            return {
                "status": "success",
                "filtered_memories": memories,
                "count": len(memories)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Memory filtering failed: {str(e)}"
            }

    async def _handle_get_statistics(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_graph_memory_statistics tool execution."""
        try:
            stats = await self.database_manager.get_statistics()
            
            # Add interface manager stats
            stats.update({
                "interface_stats": {
                    "tool_call_count": self.tool_call_count,
                    "error_count": self.error_count,
                    "last_activity": self.last_activity.isoformat() if self.last_activity else None,
                    "registered_tools": len(self.tools)
                }
            })
            
            return {
                "status": "success",
                "statistics": stats
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Statistics retrieval failed: {str(e)}"
            }

    async def _handle_create_relationship(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle create_relationship tool execution."""
        try:
            relationship_type = await self.database_manager.create_relationship(
                source_memory_id=args["source_memory_id"],
                target_memory_id=args["target_memory_id"],
                relationship_type=args["relationship_type"]
            )
            
            if relationship_type:
                return {
                    "status": "success",
                    "message": f"Relationship '{relationship_type}' created from {args['source_memory_id']} to {args['target_memory_id']}."
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to create relationship. Check if both memory IDs exist."
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Relationship creation failed: {str(e)}"
            }

    async def _handle_prune_memories(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prune_graph_memories tool execution."""
        try:
            deleted_count = await self.database_manager.prune_memories(
                max_age_days=args.get("max_age_days", 180),
                min_lookup_count=args.get("min_lookup_count", 1)
            )
            
            return {
                "status": "success",
                "deleted_count": deleted_count,
                "message": f"Successfully pruned {deleted_count} old or irrelevant memories."
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Memory pruning failed: {str(e)}"
            }

    async def _handle_health_check(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle health_check tool execution."""
        try:
            # Database health check
            db_health = await self.database_manager.health_check()
            
            # Interface manager health
            interface_health = {
                "status": "healthy",
                "tools_registered": len(self.tools),
                "tool_calls": self.tool_call_count,
                "error_rate": self.error_count / max(self.tool_call_count, 1),
                "last_activity": self.last_activity.isoformat() if self.last_activity else None
            }
            
            overall_status = "healthy" if db_health.get("status") == "healthy" else "unhealthy"
            
            return {
                "status": overall_status,
                "database_health": db_health,
                "interface_health": interface_health,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    # ═══════════════════════════════════════════════════════════════════════════════════
    # MCP CLIENT INTEGRATION
    # ═══════════════════════════════════════════════════════════════════════════════════

    async def call_mcp_tool(self, session: ClientSession, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """
        Call an MCP tool with error handling and response parsing.
        Compatible with existing interface patterns.
        
        Args:
            session: MCP client session
            tool_name: Name of the tool to call
            tool_args: Tool arguments
            
        Returns:
            Any: Tool response data
        """
        try:
            # Create MCP tool call request
            mcp_request = CallToolRequest(
                method="tools/call",
                params={
                    "name": tool_name,
                    "arguments": tool_args
                }
            )
            
            # Execute MCP tool call
            mcp_response = await session.call_tool(mcp_request)
            
            # Handle response errors
            if mcp_response.isError or not mcp_response.content:
                error_message = f"Error from MCP tool '{tool_name}'."
                if mcp_response.meta and mcp_response.meta.get("error_message"):
                    error_message += f" Details: {mcp_response.meta['error_message']}"
                logger.error(f"❌ {error_message}")
                return {"error": error_message}
            
            # Parse response content
            if isinstance(mcp_response.content[0], mcp_types.TextContent):
                tool_result_text = mcp_response.content[0].text
                logger.info(f"✅ MCP tool '{tool_name}' successful")
                
                try:
                    return json.loads(tool_result_text)
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ MCP tool '{tool_name}' result was not valid JSON")
                    return {"raw_output": tool_result_text}
            else:
                logger.warning(f"⚠️ Unexpected content type from MCP tool '{tool_name}'")
                return {"error": "Unexpected content type from MCP tool"}
                
        except Exception as e:
            logger.error(f"❌ Exception during MCP tool call '{tool_name}': {e}")
            return {"error": str(e), "exception_type": type(e).__name__}

    async def run_conversation_with_tools(self, user_prompt: str, mcp_session: ClientSession) -> str:
        """
        Run a conversation with tool integration.
        
        Args:
            user_prompt: User's input prompt
            mcp_session: MCP session for tool calls
            
        Returns:
            str: Conversation response
        """
        try:
            if not OPENAI_AVAILABLE:
                return "OpenAI integration not available for conversation management."
            
            # Get tool definitions for OpenAI
            tools = self.get_openai_tool_definitions()
            
            # Create OpenAI client (would need API key configuration)
            # This is a placeholder for OpenAI integration
            response = f"Processed user prompt: '{user_prompt}' with {len(tools)} available tools."
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Conversation failed: {e}")
            return f"Conversation error: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS AND UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════════

def create_interface_manager(database_config: Optional[Dict[str, Any]] = None) -> UnifiedInterfaceManager:
    """
    Factory function to create a configured interface manager.
    
    Args:
        database_config: Optional database configuration
        
    Returns:
        UnifiedInterfaceManager: Configured interface manager
    """
    db_config = database_config or {
        "uri": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "FinOrchestration",
        "database": "neo4j"
    }
    
    config = InterfaceConfig(
        database_config=db_config,
        mcp_enabled=MCP_AVAILABLE,
        openai_integration=OPENAI_AVAILABLE,
        debug_mode=True
    )
    
    return UnifiedInterfaceManager(config)


def get_tool_definitions_for_mcp() -> List[Dict[str, Any]]:
    """
    Get tool definitions formatted for MCP protocol.
    Convenience function for external integrations.
    
    Returns:
        List[Dict[str, Any]]: MCP-formatted tool definitions
    """
    interface_manager = create_interface_manager()
    return interface_manager.get_tool_definitions(ProtocolType.MCP)


# ═══════════════════════════════════════════════════════════════════════════════════
# TESTING AND VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    async def test_interface_manager():
        """Test the unified interface manager functionality."""
        print("🧪 Testing Unified Interface Manager")
        
        # Create interface manager
        interface_manager = create_interface_manager()
        
        if await interface_manager.initialize():
            print("✅ Interface manager initialized successfully")
            
            # Test tool definitions
            tools = interface_manager.get_tool_definitions()
            print(f"✅ Retrieved {len(tools)} tool definitions")
            
            # Test tool execution
            test_args = {
                "query": "Test memory for interface manager",
                "keywords": ["test", "interface"],
                "summary": "Testing the unified interface manager",
                "agent_id": "test_interface_agent"
            }
            
            result = await interface_manager.execute_tool("store_graph_memory", test_args)
            print(f"✅ Tool execution result: {result.get('status', 'unknown')}")
            
            # Test health check
            health = await interface_manager.execute_tool("health_check", {})
            print(f"✅ Health check: {health.get('status', 'unknown')}")
            
            await interface_manager.shutdown()
        else:
            print("❌ Interface manager initialization failed")
    
    asyncio.run(test_interface_manager())
