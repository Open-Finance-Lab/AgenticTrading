#!/usr/bin/env python3
"""
FinAgent Memory MCP Server

This module implements the MCP (Model Context Protocol) server for FinAgent Memory operations.
It provides a clean interface for memory storage, retrieval, and management using the
unified database manager and interface components.

Features:
- MCP protocol compliance for agent lifecycle management
- Unified database operations through centralized manager
- Standardized tool definitions and error handling
- Enhanced memory operations with intelligent linking
- Real-time streaming and semantic search capabilities

Author: FinAgent Team
License: Open Source
"""

# ═══════════════════════════════════════════════════════════════════════════════════
# IMPORTS AND DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════════════

from mcp.server.fastmcp import FastMCP
import uuid
import json
import sys
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

# Import unified components
try:
    from unified_database_manager import UnifiedDatabaseManager, TradingGraphMemory, create_database_manager
    from unified_interface_manager import UnifiedInterfaceManager, create_interface_manager
    UNIFIED_COMPONENTS_AVAILABLE = True
except ImportError:
    # Fallback to original database for compatibility
    from database import TradingGraphMemory
    UNIFIED_COMPONENTS_AVAILABLE = False

# Import intelligent indexer and stream processor
try:
    from intelligent_memory_indexer import IntelligentMemoryIndexer
    INTELLIGENT_INDEXER_AVAILABLE = True
except ImportError:
    INTELLIGENT_INDEXER_AVAILABLE = False

try:
    from realtime_stream_processor import StreamProcessor, ReactiveMemoryManager
    STREAM_PROCESSOR_AVAILABLE = True
except ImportError:
    STREAM_PROCESSOR_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION AND GLOBAL VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════════

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "FinOrchestration"

# Global instances for unified architecture
UNIFIED_DATABASE_MANAGER: Optional[UnifiedDatabaseManager] = None
UNIFIED_INTERFACE_MANAGER: Optional[UnifiedInterfaceManager] = None

# Legacy compatibility
GRAPH_DB_INSTANCE: Optional[TradingGraphMemory] = None

# Enhanced components
INTELLIGENT_INDEXER: Optional[IntelligentMemoryIndexer] = None
STREAM_PROCESSOR: Optional[StreamProcessor] = None
REACTIVE_MANAGER: Optional[ReactiveMemoryManager] = None

# ═══════════════════════════════════════════════════════════════════════════════════
# APPLICATION LIFECYCLE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[None]:
    """
    Application lifecycle manager with enhanced initialization.
    Handles both unified architecture and legacy compatibility.
    """
    global UNIFIED_DATABASE_MANAGER, UNIFIED_INTERFACE_MANAGER, GRAPH_DB_INSTANCE
    global INTELLIGENT_INDEXER, STREAM_PROCESSOR, REACTIVE_MANAGER
    
    print("🚀 [SERVER] ═══════════════════════════════════════════════════════════")
    print("🚀 [SERVER] FINAGENT MEMORY SERVER INITIALIZATION")
    print("🚀 [SERVER] ═══════════════════════════════════════════════════════════")

    try:
        # ═══════════════════════════════════════════════════════════════════════════════════
        # UNIFIED ARCHITECTURE INITIALIZATION (PREFERRED)
        # ═══════════════════════════════════════════════════════════════════════════════════
        
        if UNIFIED_COMPONENTS_AVAILABLE:
            print("🔧 [SERVER] Initializing unified architecture components...")
            
            # Initialize unified database manager
            database_config = {
                "uri": NEO4J_URI,
                "username": NEO4J_USER,
                "password": NEO4J_PASSWORD,
                "database": "neo4j"
            }
            
            UNIFIED_DATABASE_MANAGER = create_database_manager(database_config)
            
            if await UNIFIED_DATABASE_MANAGER.connect():
                print("✅ [SERVER] Unified database manager connected successfully")
                
                # Initialize unified interface manager
                UNIFIED_INTERFACE_MANAGER = create_interface_manager(database_config)
                
                if await UNIFIED_INTERFACE_MANAGER.initialize():
                    print("✅ [SERVER] Unified interface manager initialized successfully")
                else:
                    print("⚠️ [SERVER] Unified interface manager initialization failed, using legacy mode")
                    UNIFIED_INTERFACE_MANAGER = None
            else:
                print("⚠️ [SERVER] Unified database manager connection failed, falling back to legacy")
                UNIFIED_DATABASE_MANAGER = None

        # ═══════════════════════════════════════════════════════════════════════════════════
        # LEGACY COMPATIBILITY INITIALIZATION (FALLBACK)
        # ═══════════════════════════════════════════════════════════════════════════════════
        
        if not UNIFIED_DATABASE_MANAGER:
            print("🔧 [SERVER] Initializing legacy TradingGraphMemory for compatibility...")
            
            GRAPH_DB_INSTANCE = TradingGraphMemory(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

            if GRAPH_DB_INSTANCE and GRAPH_DB_INSTANCE.driver:
                print("🔗 [SERVER] Creating full-text search index...")
                await GRAPH_DB_INSTANCE.create_memory_index()
                print("🔗 [SERVER] Creating structured property indexes...")
                await GRAPH_DB_INSTANCE.create_structured_indexes()
                print("✅ [SERVER] Legacy database initialization complete")
            else:
                print("❌ [SERVER] ERROR: Could not connect to Neo4j database")

        # ═══════════════════════════════════════════════════════════════════════════════════
        # ENHANCED FEATURES INITIALIZATION (OPTIONAL)
        # ═══════════════════════════════════════════════════════════════════════════════════

        # Initialize intelligent indexer
        if INTELLIGENT_INDEXER_AVAILABLE:
            try:
                INTELLIGENT_INDEXER = IntelligentMemoryIndexer()
                print("🧠 [SERVER] Intelligent memory indexer initialized successfully")
            except Exception as e:
                print(f"⚠️ [SERVER] Failed to initialize intelligent indexer: {e}")

        # Initialize real-time stream processor
        if STREAM_PROCESSOR_AVAILABLE:
            try:
                STREAM_PROCESSOR = StreamProcessor()
                REACTIVE_MANAGER = ReactiveMemoryManager(STREAM_PROCESSOR)
                print("⚡ [SERVER] Real-time stream processor initialized successfully")
            except Exception as e:
                print(f"⚠️ [SERVER] Failed to initialize stream processor: {e}")

        print("� [SERVER] ═══════════════════════════════════════════════════════════")
        print("🚀 [SERVER] MEMORY SERVER STARTUP COMPLETE")
        print("🚀 [SERVER] ═══════════════════════════════════════════════════════════")
        
        # Component status summary
        print(f"📊 [SERVER] Component Status:")
        print(f"   🗄️  Unified Database: {'✅ Active' if UNIFIED_DATABASE_MANAGER else '❌ Unavailable'}")
        print(f"   🔧 Unified Interface: {'✅ Active' if UNIFIED_INTERFACE_MANAGER else '❌ Unavailable'}")
        print(f"   📚 Legacy Database: {'✅ Active' if GRAPH_DB_INSTANCE else '❌ Unavailable'}")
        print(f"   🧠 Intelligent Indexer: {'✅ Active' if INTELLIGENT_INDEXER else '❌ Unavailable'}")
        print(f"   ⚡ Stream Processor: {'✅ Active' if STREAM_PROCESSOR else '❌ Unavailable'}")
        
        yield
        
    finally:
        print("🛑 [SERVER] ═══════════════════════════════════════════════════════════")
        print("🛑 [SERVER] MEMORY SERVER SHUTDOWN INITIATED")
        print("🛑 [SERVER] ═══════════════════════════════════════════════════════════")
        
        # Clean shutdown of all components
        if UNIFIED_DATABASE_MANAGER:
            await UNIFIED_DATABASE_MANAGER.close()
            print("✅ [SERVER] Unified database manager closed")
            
        if UNIFIED_INTERFACE_MANAGER:
            await UNIFIED_INTERFACE_MANAGER.shutdown()
            print("✅ [SERVER] Unified interface manager shut down")
            
        if GRAPH_DB_INSTANCE:
            await GRAPH_DB_INSTANCE.close()
            print("✅ [SERVER] Legacy database connection closed")
            
        if STREAM_PROCESSOR:
            await STREAM_PROCESSOR.shutdown()
            print("✅ [SERVER] Stream processor shut down")
            
        print("🛑 [SERVER] Memory server shutdown complete")

# ═══════════════════════════════════════════════════════════════════════════════════
# MCP SERVER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════════

mcp = FastMCP(
    "FinAgentMemoryServer",
    lifespan=app_lifespan,
    stateless_http=True,
    debug=True
)

# ═══════════════════════════════════════════════════════════════════════════════════
# ENHANCED MEMORY STORAGE TOOLS
# ═══════════════════════════════════════════════════════════════════════════════════

@mcp.tool(name="store_graph_memory",
          description="Stores a structured memory in the Neo4j graph database with intelligent linking and semantic indexing.")
async def store_graph_memory(
    query: str,
    keywords: list,
    summary: str,
    agent_id: str,
    event_type: Optional[str] = 'USER_QUERY',
    log_level: Optional[str] = 'INFO',
    session_id: Optional[str] = None,
    correlation_id: Optional[str] = None
):
    """Enhanced memory storage with unified architecture support."""
    print(f"🛠️ [SERVER] ═══════ TOOL: store_graph_memory ═══════")
    
    # Use unified database manager if available
    if UNIFIED_DATABASE_MANAGER:
        try:
            stored_data = await UNIFIED_DATABASE_MANAGER.store_memory(
                query=query,
                keywords=keywords,
                summary=summary,
                agent_id=agent_id,
                event_type=event_type,
                log_level=log_level,
                session_id=session_id,
                correlation_id=correlation_id
            )
            
            if stored_data:
                linked_count = len(stored_data.get('linked_memories', []))
                message = f"Memory stored with unified manager and linked to {linked_count} similar memories."
                print(f"   ✅ [SERVER] {message}")
                
                # Publish to stream processor if available
                if REACTIVE_MANAGER:
                    await REACTIVE_MANAGER.handle_memory_event({
                        "event_type": "memory_stored",
                        "memory_id": stored_data.get("memory_id"),
                        "agent_id": agent_id,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                response_data = {
                    "status": "success",
                    "message": message,
                    "stored_memory": stored_data,
                    "architecture": "unified"
                }
                return json.dumps(response_data)
            else:
                raise Exception("Unified database manager returned None")
                
        except Exception as e:
            print(f"   ❌ [SERVER] ERROR in unified store_graph_memory: {e}")
            # Fall through to legacy handling
    
    # Legacy fallback
    if not GRAPH_DB_INSTANCE:
        raise Exception("No database connection available (unified or legacy).")

    try:
        stored_data = await GRAPH_DB_INSTANCE.store_memory(
            query, keywords, summary, agent_id, event_type, log_level, session_id, correlation_id
        )
        
        if stored_data:
            linked_count = len(stored_data.get('linked_memories', []))
            message = f"Memory stored with legacy manager and linked to {linked_count} similar memories."
            print(f"   ✅ [SERVER] {message}")
            
            response_data = {
                "status": "success",
                "message": message,
                "stored_memory": stored_data,
                "architecture": "legacy"
            }
            return json.dumps(response_data)
        else:
            raise Exception("Legacy database manager returned None")
            
    except Exception as e:
        print(f"   ❌ [SERVER] ERROR in legacy store_graph_memory: {e}")
        error_response = {
            "status": "error",
            "message": f"Memory storage failed: {str(e)}",
            "architecture": "legacy"
        }
        return json.dumps(error_response)

@mcp.tool(name="store_graph_memories_batch",
          description="Stores multiple memories in a batch operation for high-throughput processing with enhanced performance monitoring.")
async def store_graph_memories_batch(events: List[Dict[str, Any]]):
    """Enhanced batch storage with unified architecture support."""
    print(f"🛠️ [SERVER] ═══════ TOOL: store_graph_memories_batch ═══════")
    
    # Use unified database manager if available
    if UNIFIED_DATABASE_MANAGER:
        try:
            count = await UNIFIED_DATABASE_MANAGER.store_memories_batch(events)
            message = f"Successfully stored {count} memories using unified batch operation."
            print(f"   ✅ [SERVER] {message}")
            
            response_data = {
                "status": "success", 
                "stored_count": count, 
                "message": message,
                "architecture": "unified"
            }
            return json.dumps(response_data)
            
        except Exception as e:
            print(f"   ❌ [SERVER] ERROR in unified batch storage: {e}")
            # Fall through to legacy handling
    
    # Legacy fallback
    if not GRAPH_DB_INSTANCE:
        raise Exception("No database connection available for batch operation.")
    
    try:
        count = await GRAPH_DB_INSTANCE.store_memories_batch(events)
        message = f"Successfully stored {count} memories using legacy batch operation."
        print(f"   ✅ [SERVER] {message}")
        
        response_data = {
            "status": "success", 
            "stored_count": count, 
            "message": message,
            "architecture": "legacy"
        }
        return json.dumps(response_data)
        
    except Exception as e:
        print(f"   ❌ [SERVER] ERROR in legacy batch storage: {e}")
        return json.dumps({
            "status": "error", 
            "message": f"Batch storage failed: {str(e)}",
            "architecture": "legacy"
        })

# ═══════════════════════════════════════════════════════════════════════════════════
# ENHANCED MEMORY RETRIEVAL TOOLS
# ═══════════════════════════════════════════════════════════════════════════════════

@mcp.tool(name="retrieve_graph_memory",
          description="Retrieves memories using enhanced full-text search with intelligent ranking and semantic capabilities.")
async def retrieve_graph_memory(
    search_query: str, 
    limit: int = 5
):
    """Enhanced memory retrieval with unified architecture support."""
    print(f"🛠️ [SERVER] ═══════ TOOL: retrieve_graph_memory ═══════")
    
    # Use unified database manager if available
    if UNIFIED_DATABASE_MANAGER:
        try:
            search_results = await UNIFIED_DATABASE_MANAGER.retrieve_memory(search_query, limit)
            print(f"   ✅ [SERVER] Retrieved {len(search_results)} memories using unified manager.")
            
            response_data = {
                "status": "success", 
                "retrieved_memories": search_results,
                "architecture": "unified",
                "enhanced_features": {
                    "semantic_search": INTELLIGENT_INDEXER is not None,
                    "real_time_processing": STREAM_PROCESSOR is not None
                }
            }
            return json.dumps(response_data)
            
        except Exception as e:
            print(f"   ❌ [SERVER] ERROR in unified retrieval: {e}")
            # Fall through to legacy handling
    
    # Legacy fallback
    if not GRAPH_DB_INSTANCE:
        error_response = {"status": "error", "message": "Database connection is not available."}
        print("   ❌ [SERVER] ERROR: No database connection available.")
        return json.dumps(error_response)

    try:
        search_results = await GRAPH_DB_INSTANCE.retrieve_memory(search_query, limit)
        print(f"   ✅ [SERVER] Retrieved {len(search_results)} memories using legacy manager.")
        
        response_data = {
            "status": "success", 
            "retrieved_memories": search_results,
            "architecture": "legacy"
        }
        return json.dumps(response_data)
        
    except Exception as e:
        print(f"   ❌ [SERVER] ERROR in legacy retrieval: {e}")
        error_response = {
            "status": "error", 
            "message": f"Memory retrieval failed: {str(e)}",
            "exception_type": type(e).__name__,
            "architecture": "legacy"
        }
        return json.dumps(error_response)

# ═══════════════════════════════════════════════════════════════════════════════════
# ADVANCED FILTERING AND ANALYTICS TOOLS
# ═══════════════════════════════════════════════════════════════════════════════════

@mcp.tool(name="filter_graph_memories",
          description="Filters memories based on structured criteria with enhanced query capabilities and performance optimization.")
async def filter_graph_memories(
    filters: Dict[str, Any], 
    limit: int = 100,
    offset: int = 0
):
    """Enhanced memory filtering with unified architecture support."""
    print(f"🛠️ [SERVER] ═══════ TOOL: filter_graph_memories ═══════")
    
    # Use unified database manager if available
    if UNIFIED_DATABASE_MANAGER:
        try:
            results = await UNIFIED_DATABASE_MANAGER.filter_memories(filters, limit, offset)
            message = f"Unified filter query returned {len(results)} memories."
            print(f"   ✅ [SERVER] {message}")
            
            response_data = {
                "status": "success", 
                "filtered_memories": results,
                "message": message,
                "architecture": "unified",
                "filter_criteria": filters
            }
            return json.dumps(response_data)
            
        except Exception as e:
            print(f"   ❌ [SERVER] ERROR in unified filtering: {e}")
            # Fall through to legacy handling
    
    # Legacy fallback
    if not GRAPH_DB_INSTANCE:
        raise Exception("Database connection is not available for filtering.")

    try:
        results = await GRAPH_DB_INSTANCE.filter_memories(filters, limit, offset)
        message = f"Legacy filter query returned {len(results)} memories."
        print(f"   ✅ [SERVER] {message}")
        
        response_data = {
            "status": "success", 
            "filtered_memories": results,
            "message": message,
            "architecture": "legacy"
        }
        return json.dumps(response_data)
        
    except Exception as e:
        print(f"   ❌ [SERVER] ERROR in legacy filtering: {e}")
        return json.dumps({
            "status": "error", 
            "message": f"Filtering failed: {str(e)}",
            "architecture": "legacy"
        })

@mcp.tool(name="get_graph_memory_statistics",
          description="Retrieves comprehensive statistics with enhanced metrics including performance analytics and system health.")
async def get_graph_memory_statistics():
    """Enhanced statistics retrieval with unified architecture support."""
    print(f"🛠️ [SERVER] ═══════ TOOL: get_graph_memory_statistics ═══════")
    
    # Use unified database manager if available
    if UNIFIED_DATABASE_MANAGER:
        try:
            stats = await UNIFIED_DATABASE_MANAGER.get_statistics()
            print(f"   ✅ [SERVER] Successfully retrieved enhanced statistics.")
            
            # Add architecture information
            stats.update({
                "architecture": "unified",
                "enhanced_features": {
                    "intelligent_indexing": INTELLIGENT_INDEXER is not None,
                    "real_time_processing": STREAM_PROCESSOR is not None,
                    "unified_interface": UNIFIED_INTERFACE_MANAGER is not None
                }
            })
            
            response_data = {
                "status": "success", 
                "statistics": stats
            }
            return json.dumps(response_data)
            
        except Exception as e:
            print(f"   ❌ [SERVER] ERROR in unified statistics: {e}")
            # Fall through to legacy handling
    
    # Legacy fallback
    if not GRAPH_DB_INSTANCE:
        raise Exception("Database connection is not available for statistics.")
    
    try:
        stats = await GRAPH_DB_INSTANCE.get_statistics()
        stats.update({"architecture": "legacy"})
        print(f"   ✅ [SERVER] Successfully retrieved legacy statistics.")
        
        response_data = {
            "status": "success", 
            "statistics": stats
        }
        return json.dumps(response_data)
        
    except Exception as e:
        print(f"   ❌ [SERVER] ERROR in legacy statistics: {e}")
        return json.dumps({
            "status": "error", 
            "message": f"Statistics retrieval failed: {str(e)}",
            "architecture": "legacy"
        })

# ═══════════════════════════════════════════════════════════════════════════════════
# ADVANCED RETRIEVAL WITH RELATIONSHIP EXPANSION
# ═══════════════════════════════════════════════════════════════════════════════════

@mcp.tool(name="retrieve_memory_with_expansion",
          description="Retrieves memories with intelligent relationship expansion for comprehensive context discovery.")
async def retrieve_memory_with_expansion(
    search_query: str,
    limit: int = 10
):
    """Enhanced memory retrieval with relationship expansion."""
    print(f"🛠️ [SERVER] ═══════ TOOL: retrieve_memory_with_expansion ═══════")
    
    # Use unified database manager if available
    if UNIFIED_DATABASE_MANAGER:
        try:
            search_results = await UNIFIED_DATABASE_MANAGER.retrieve_memory_with_expansion(search_query, limit)
            print(f"   ✅ [SERVER] Retrieved {len(search_results)} memories with unified expansion.")
            
            response_data = {
                "status": "success", 
                "retrieved_memories": search_results,
                "architecture": "unified",
                "expansion_enabled": True
            }
            return json.dumps(response_data)
            
        except Exception as e:
            print(f"   ❌ [SERVER] ERROR in unified expansion retrieval: {e}")
            # Fall through to legacy handling
    
    # Legacy fallback
    if not GRAPH_DB_INSTANCE:
        raise Exception("Database connection is not available for expansion retrieval.")
    
    try:
        search_results = await GRAPH_DB_INSTANCE.retrieve_memory_with_expansion(search_query, limit)
        print(f"   ✅ [SERVER] Retrieved {len(search_results)} memories with legacy expansion.")
        
        response_data = {
            "status": "success", 
            "retrieved_memories": search_results,
            "architecture": "legacy"
        }
        return json.dumps(response_data)
        
    except Exception as e:
        print(f"   ❌ [SERVER] ERROR in legacy expansion retrieval: {e}")
        error_response = {
            "status": "error", 
            "message": f"Expansion retrieval failed: {str(e)}",
            "architecture": "legacy"
        }
        return json.dumps(error_response)

# ═══════════════════════════════════════════════════════════════════════════════════
# ENHANCED SEMANTIC SEARCH TOOLS
# ═══════════════════════════════════════════════════════════════════════════════════

@mcp.tool(name="semantic_search_memories",
          description="Performs intelligent semantic search using AI embeddings, similarity scoring, and context understanding.")
async def semantic_search_memories(
    query: str,
    limit: int = 10,
    similarity_threshold: float = 0.3
):
    """Advanced semantic search with intelligent indexing."""
    print(f"🛠️ [SERVER] ═══════ TOOL: semantic_search_memories ═══════")
    
    if not INTELLIGENT_INDEXER:
        return json.dumps({
            "status": "error", 
            "message": "Intelligent indexer not available. Install sentence-transformers for semantic search.",
            "feature": "semantic_search"
        })
    
    # Use unified database manager if available
    database_instance = UNIFIED_DATABASE_MANAGER or GRAPH_DB_INSTANCE
    
    if not database_instance:
        raise Exception("No database connection available for semantic search.")

    try:
        # Get memories for semantic search
        if UNIFIED_DATABASE_MANAGER:
            all_memories = await UNIFIED_DATABASE_MANAGER.retrieve_memory("", limit=1000)
        else:
            all_memories = await GRAPH_DB_INSTANCE.retrieve_memory("", limit=1000)
        
        if not all_memories:
            return json.dumps({
                "status": "success",
                "results": [],
                "message": "No memories found in database for semantic search."
            })

        # Perform semantic search using intelligent indexer
        search_results = INTELLIGENT_INDEXER.semantic_search(
            query=query,
            memories=all_memories,
            top_k=limit,
            similarity_threshold=similarity_threshold
        )
        
        # Process results and handle numpy types
        processed_results = []
        for result in search_results:
            processed_result = dict(result)
            if 'similarity_score' in processed_result:
                processed_result['similarity_score'] = float(processed_result['similarity_score'])
            processed_results.append(processed_result)
        
        print(f"   ✅ [SERVER] Semantic search returned {len(processed_results)} results.")
        
        # Publish search event to stream processor
        if REACTIVE_MANAGER:
            await REACTIVE_MANAGER.handle_search_event({
                "query": query,
                "results_count": len(processed_results),
                "timestamp": datetime.utcnow().isoformat(),
                "search_type": "semantic"
            })
        
        response_data = {
            "status": "success",
            "results": processed_results,
            "query": query,
            "similarity_threshold": similarity_threshold,
            "architecture": "unified" if UNIFIED_DATABASE_MANAGER else "legacy",
            "search_type": "semantic"
        }
        return json.dumps(response_data)
        
    except Exception as e:
        print(f"   ❌ [SERVER] ERROR in semantic search: {e}")
        return json.dumps({
            "status": "error", 
            "message": f"Semantic search failed: {str(e)}",
            "search_type": "semantic"
        })

@mcp.tool(name="get_trending_keywords",
          description="Extracts and analyzes trending keywords from recent memories using intelligent text processing.")
async def get_trending_keywords(
    days_back: int = 7,
    top_k: int = 20
):
    """Extract trending keywords using intelligent analysis."""
    print(f"🛠️ [SERVER] ═══════ TOOL: get_trending_keywords ═══════")
    
    if not INTELLIGENT_INDEXER:
        return json.dumps({
            "status": "error",
            "message": "Intelligent indexer not available for keyword analysis.",
            "feature": "keyword_analysis"
        })
    
    database_instance = UNIFIED_DATABASE_MANAGER or GRAPH_DB_INSTANCE
    
    if not database_instance:
        raise Exception("Database connection is not available for keyword analysis.")

    try:
        # Get recent memories
        if UNIFIED_DATABASE_MANAGER:
            recent_memories = await UNIFIED_DATABASE_MANAGER.retrieve_memory("", limit=500)
        else:
            recent_memories = await GRAPH_DB_INSTANCE.retrieve_memory("", limit=500)
        
        if not recent_memories:
            return json.dumps({
                "status": "success",
                "keywords": [],
                "message": "No recent memories found for keyword analysis."
            })

        # Extract trending keywords
        keywords = INTELLIGENT_INDEXER.extract_trending_keywords(
            memories=recent_memories,
            top_k=top_k
        )
        
        print(f"   ✅ [SERVER] Extracted {len(keywords)} trending keywords.")
        
        response_data = {
            "status": "success",
            "keywords": keywords,
            "days_analyzed": days_back,
            "total_memories": len(recent_memories),
            "architecture": "unified" if UNIFIED_DATABASE_MANAGER else "legacy"
        }
        return json.dumps(response_data)
        
    except Exception as e:
        print(f"   ❌ [SERVER] ERROR in keyword analysis: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Keyword extraction failed: {str(e)}"
        })

# ═══════════════════════════════════════════════════════════════════════════════════
# DATABASE MAINTENANCE AND OPTIMIZATION TOOLS
# ═══════════════════════════════════════════════════════════════════════════════════

@mcp.tool(name="prune_graph_memories",
          description="Intelligently deletes old and irrelevant memories with enhanced criteria and safety checks.")
async def prune_graph_memories(
    max_age_days: int = 180,
    min_lookup_count: int = 1
):
    """Enhanced memory pruning with unified architecture support."""
    print(f"🛠️ [SERVER] ═══════ TOOL: prune_graph_memories ═══════")
    
    # Use unified database manager if available
    if UNIFIED_DATABASE_MANAGER:
        try:
            deleted_count = await UNIFIED_DATABASE_MANAGER.prune_memories(max_age_days, min_lookup_count)
            message = f"Unified pruning successfully removed {deleted_count} old or irrelevant memories."
            print(f"   ✅ [SERVER] {message}")
            
            response_data = {
                "status": "success", 
                "deleted_count": deleted_count, 
                "message": message,
                "architecture": "unified",
                "criteria": {
                    "max_age_days": max_age_days,
                    "min_lookup_count": min_lookup_count
                }
            }
            return json.dumps(response_data)
            
        except Exception as e:
            print(f"   ❌ [SERVER] ERROR in unified pruning: {e}")
            # Fall through to legacy handling
    
    # Legacy fallback
    if not GRAPH_DB_INSTANCE:
        raise Exception("Database connection is not available for pruning.")

    try:
        deleted_count = await GRAPH_DB_INSTANCE.prune_memories(max_age_days, min_lookup_count)
        message = f"Legacy pruning successfully removed {deleted_count} old or irrelevant memories."
        print(f"   ✅ [SERVER] {message}")
        
        response_data = {
            "status": "success", 
            "deleted_count": deleted_count, 
            "message": message,
            "architecture": "legacy"
        }
        return json.dumps(response_data)
        
    except Exception as e:
        print(f"   ❌ [SERVER] ERROR in legacy pruning: {e}")
        error_response = {
            "status": "error", 
            "message": f"Memory pruning failed: {str(e)}",
            "architecture": "legacy"
        }
        return json.dumps(error_response)

# ═══════════════════════════════════════════════════════════════════════════════════
# ADVANCED RELATIONSHIP MANAGEMENT TOOLS
# ═══════════════════════════════════════════════════════════════════════════════════

@mcp.tool(name="create_relationship",
          description="Creates intelligent relationships between memory nodes with enhanced context linking and validation.")
async def create_relationship(
    source_memory_id: str,
    target_memory_id: str,
    relationship_type: str
):
    """Enhanced relationship creation with unified architecture support."""
    print(f"🛠️ [SERVER] ═══════ TOOL: create_relationship ═══════")
    
    # Use unified database manager if available
    if UNIFIED_DATABASE_MANAGER:
        try:
            rel_type = await UNIFIED_DATABASE_MANAGER.create_relationship(
                source_memory_id, target_memory_id, relationship_type
            )
            
            if rel_type:
                message = f"Unified relationship '{rel_type}' created from {source_memory_id} to {target_memory_id}."
                print(f"   ✅ [SERVER] {message}")
                
                response_data = {
                    "status": "success", 
                    "message": message,
                    "relationship_type": rel_type,
                    "source_id": source_memory_id,
                    "target_id": target_memory_id,
                    "architecture": "unified"
                }
                return json.dumps(response_data)
            else:
                raise Exception("Unified relationship creation failed. Check if both memory IDs exist.")
                
        except Exception as e:
            print(f"   ❌ [SERVER] ERROR in unified relationship creation: {e}")
            # Fall through to legacy handling
    
    # Legacy fallback
    if not GRAPH_DB_INSTANCE:
        raise Exception("Database connection is not available for relationship creation.")

    try:
        rel_type = await GRAPH_DB_INSTANCE.create_relationship(source_memory_id, target_memory_id, relationship_type)
        
        if rel_type:
            message = f"Legacy relationship '{rel_type}' created from {source_memory_id} to {target_memory_id}."
            print(f"   ✅ [SERVER] {message}")
            
            response_data = {
                "status": "success", 
                "message": message,
                "relationship_type": rel_type,
                "architecture": "legacy"
            }
            return json.dumps(response_data)
        else:
            raise Exception("Legacy relationship creation failed. Check if both memory IDs exist.")
            
    except Exception as e:
        print(f"   ❌ [SERVER] ERROR in legacy relationship creation: {e}")
        error_response = {
            "status": "error", 
            "message": f"Relationship creation failed: {str(e)}",
            "architecture": "legacy"
        }
        return json.dumps(error_response)

# ═══════════════════════════════════════════════════════════════════════════════════
# REAL-TIME STREAMING AND EVENT PROCESSING TOOLS
# ═══════════════════════════════════════════════════════════════════════════════════

@mcp.tool(name="publish_memory_event",
          description="Publishes memory-related events to the real-time stream processing system for reactive analytics.")
async def publish_memory_event(
    event_type: str,
    event_data: Dict[str, Any],
    priority: str = "normal"
):
    """Publish events to real-time stream processor."""
    print(f"🛠️ [SERVER] ═══════ TOOL: publish_memory_event ═══════")
    
    if not STREAM_PROCESSOR:
        return json.dumps({
            "status": "error",
            "message": "Stream processor not available. Install redis and websockets for real-time processing.",
            "feature": "real_time_streaming"
        })

    try:
        event_id = await STREAM_PROCESSOR.publish_event(
            event_type=event_type,
            data=event_data,
            priority=priority
        )
        
        print(f"   ✅ [SERVER] Published event {event_id} to stream processor.")
        
        response_data = {
            "status": "success",
            "event_id": event_id,
            "event_type": event_type,
            "message": f"Event published successfully with ID: {event_id}",
            "priority": priority
        }
        return json.dumps(response_data)
        
    except Exception as e:
        print(f"   ❌ [SERVER] ERROR in event publishing: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Failed to publish event: {str(e)}"
        })

@mcp.tool(name="get_stream_statistics",
          description="Returns comprehensive real-time statistics from the stream processing system and event analytics.")
async def get_stream_statistics():
    """Get real-time stream processing statistics."""
    print(f"🛠️ [SERVER] ═══════ TOOL: get_stream_statistics ═══════")
    
    if not STREAM_PROCESSOR:
        return json.dumps({
            "status": "error",
            "message": "Stream processor not available for statistics.",
            "feature": "stream_analytics"
        })

    try:
        stats = STREAM_PROCESSOR.get_statistics()
        
        print(f"   ✅ [SERVER] Retrieved stream processor statistics.")
        
        response_data = {
            "status": "success",
            "statistics": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        return json.dumps(response_data)
        
    except Exception as e:
        print(f"   ❌ [SERVER] ERROR in stream statistics: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Failed to get stream statistics: {str(e)}"
        })

# ═══════════════════════════════════════════════════════════════════════════════════
# SYSTEM HEALTH AND MONITORING TOOLS
# ═══════════════════════════════════════════════════════════════════════════════════

@mcp.tool(name="health_check",
          description="Performs comprehensive health check of all system components including database, indexing, and streaming.")
async def health_check():
    """Comprehensive system health check."""
    print(f"🛠️ [SERVER] ═══════ TOOL: health_check ═══════")
    
    try:
        health_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "components": {}
        }
        
        # Check unified database manager
        if UNIFIED_DATABASE_MANAGER:
            try:
                db_health = await UNIFIED_DATABASE_MANAGER.health_check()
                health_report["components"]["unified_database"] = db_health
            except Exception as e:
                health_report["components"]["unified_database"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_report["overall_status"] = "degraded"
        
        # Check legacy database
        if GRAPH_DB_INSTANCE:
            try:
                # Simple health check for legacy database
                health_report["components"]["legacy_database"] = {
                    "status": "healthy" if GRAPH_DB_INSTANCE.driver else "unhealthy",
                    "connected": GRAPH_DB_INSTANCE.driver is not None
                }
            except Exception as e:
                health_report["components"]["legacy_database"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        # Check intelligent indexer
        health_report["components"]["intelligent_indexer"] = {
            "status": "available" if INTELLIGENT_INDEXER else "unavailable",
            "enabled": INTELLIGENT_INDEXER is not None
        }
        
        # Check stream processor
        health_report["components"]["stream_processor"] = {
            "status": "available" if STREAM_PROCESSOR else "unavailable",
            "enabled": STREAM_PROCESSOR is not None
        }
        
        # Check unified interface manager
        health_report["components"]["unified_interface"] = {
            "status": "available" if UNIFIED_INTERFACE_MANAGER else "unavailable",
            "enabled": UNIFIED_INTERFACE_MANAGER is not None
        }
        
        # Determine overall status
        component_statuses = [comp.get("status", "unknown") for comp in health_report["components"].values()]
        if any(status == "unhealthy" for status in component_statuses):
            health_report["overall_status"] = "unhealthy"
        elif any(status in ["unavailable", "degraded"] for status in component_statuses):
            health_report["overall_status"] = "degraded"
        
        print(f"   ✅ [SERVER] Health check completed: {health_report['overall_status']}")
        
        response_data = {
            "status": "success",
            "health_report": health_report
        }
        return json.dumps(response_data)
        
    except Exception as e:
        print(f"   ❌ [SERVER] ERROR in health check: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Health check failed: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        })

# ═══════════════════════════════════════════════════════════════════════════════════
# APPLICATION ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════════

from starlette.routing import Route
from starlette.responses import JSONResponse

app = mcp.streamable_http_app()

# Add health check handler
async def health_handler(request):
    """Health check handler for Starlette."""
    try:
        health_result = await health_check()
        return JSONResponse({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "FinAgent Memory Server",
            "details": health_result
        })
    except Exception as e:
        return JSONResponse({
            "status": "unhealthy", 
            "timestamp": datetime.now().isoformat(),
            "service": "FinAgent Memory Server",
            "error": str(e)
        }, status_code=500)

# Add health route to the app
app.router.routes.append(Route("/health", health_handler, methods=["GET"]))

# ═══════════════════════════════════════════════════════════════════════════════════
# DEVELOPMENT AND TESTING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════════

def print_server_info():
    """Print server information and available features."""
    print("\n" + "="*80)
    print("🚀 FINAGENT MEMORY SERVER - ENHANCED ARCHITECTURE")
    print("="*80)
    print("📋 Available Features:")
    print(f"   🗄️  Unified Database Manager: {'✅ Available' if UNIFIED_COMPONENTS_AVAILABLE else '❌ Unavailable'}")
    print(f"   🔧 Unified Interface Manager: {'✅ Available' if UNIFIED_COMPONENTS_AVAILABLE else '❌ Unavailable'}")
    print(f"   🧠 Intelligent Indexer: {'✅ Available' if INTELLIGENT_INDEXER_AVAILABLE else '❌ Unavailable'}")
    print(f"   ⚡ Stream Processor: {'✅ Available' if STREAM_PROCESSOR_AVAILABLE else '❌ Unavailable'}")
    print(f"   📚 Legacy Compatibility: ✅ Maintained")
    print("\n📡 Available Tools:")
    print("   • store_graph_memory - Enhanced memory storage with intelligent linking")
    print("   • store_graph_memories_batch - High-throughput batch operations")
    print("   • retrieve_graph_memory - Enhanced full-text search with ranking")
    print("   • retrieve_memory_with_expansion - Relationship-based expansion")
    print("   • semantic_search_memories - AI-powered semantic search")
    print("   • filter_graph_memories - Advanced filtering capabilities")
    print("   • get_graph_memory_statistics - Comprehensive analytics")
    print("   • create_relationship - Intelligent relationship management")
    print("   • prune_graph_memories - Smart memory cleanup")
    print("   • get_trending_keywords - Intelligent keyword analysis")
    print("   • publish_memory_event - Real-time event publishing")
    print("   • get_stream_statistics - Stream processing analytics")
    print("   • health_check - Comprehensive system health monitoring")
    print("="*80)

if __name__ == "__main__":
    print_server_info()
    print("\n🔧 Server Configuration:")
    print(f"   📍 Neo4j URI: {NEO4J_URI}")
    print(f"   👤 Neo4j User: {NEO4J_USER}")
    print(f"   🏷️  Server Name: FinAgentMemoryServer")
    print("\n🚀 Starting FinAgent Memory Server...")
    print("   Use uvicorn to run: uvicorn memory_server:app --host 0.0.0.0 --port 8000")
    print("="*80)
