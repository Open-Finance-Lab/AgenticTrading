from mcp.server.fastmcp import FastMCP
from database import TradingGraphMemory
import uuid
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "FinOrchestration"

GRAPH_DB_INSTANCE: Optional[TradingGraphMemory] = None

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[None]:
    global GRAPH_DB_INSTANCE
    print("🚀 [SERVER] Lifespan event: Initializing application context.")

    GRAPH_DB_INSTANCE = TradingGraphMemory(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    if GRAPH_DB_INSTANCE and GRAPH_DB_INSTANCE.driver:
        print("🔗 [SERVER] Lifespan event: Ensuring full-text search index is created...")
        await GRAPH_DB_INSTANCE.create_memory_index()
        print("🔗 [SERVER] Lifespan event: Ensuring structured property indexes are created...")
        await GRAPH_DB_INSTANCE.create_structured_indexes()
    else:
        print("❌ [SERVER] Lifespan event ERROR: Could not connect to Neo4j.")

    try:
        print("✅ [SERVER] Lifespan startup complete.")
        yield
    finally:
        print("🛑 [SERVER] Lifespan event: Cleaning up and closing Neo4j connection.")
        if GRAPH_DB_INSTANCE:
            await GRAPH_DB_INSTANCE.close()

mcp = FastMCP(
    "Neo4jMemoryAgent",
    lifespan=app_lifespan,
    stateless_http=True,
    debug=True
)

@mcp.tool(name="store_graph_memory",
          description="Stores a structured memory in the Neo4j graph database.")
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
    print(f"🛠️ [SERVER] --- Tool: store_graph_memory ---")
    if not GRAPH_DB_INSTANCE:
        raise Exception("Database connection is not available.")

    try:
        stored_data = await GRAPH_DB_INSTANCE.store_memory(
            query, keywords, summary, agent_id, event_type, log_level, session_id, correlation_id
        )
        if stored_data:
            print(f"   - ✅ [SERVER] Memory stored successfully in DB.")
            linked_count = len(stored_data.get('linked_memories', []))
            message = f"Memory stored in Neo4j graph and linked to {linked_count} similar memories."

            response_data = { "status": "success", "message": message, "stored_memory": stored_data }
            return json.dumps(response_data)
        else:
            raise Exception("Failed to store memory in Neo4j, store_memory method returned None.")
    except Exception as e:
        print(f"   - ❌ [SERVER] ERROR during store_graph_memory: {e}")
        error_response = { "status": "error", "message": f"An internal error occurred in store_graph_memory: {str(e)}" }
        return json.dumps(error_response)

@mcp.tool(name="store_graph_memories_batch",
          description="Stores a batch of event-like memories in the Neo4j graph database. Optimized for high-throughput.")
async def store_graph_memories_batch(events: List[Dict[str, Any]]):
    print(f"🛠️ [SERVER] --- Tool: store_graph_memories_batch ---")
    if not GRAPH_DB_INSTANCE:
        raise Exception("Database connection is not available.")
    
    try:
        count = await GRAPH_DB_INSTANCE.store_memories_batch(events)
        message = f"Successfully stored {count} memories in a batch operation."
        print(f"   - ✅ [SERVER] {message}")
        return json.dumps({"status": "success", "stored_count": count, "message": message})
    except Exception as e:
        print(f"   - ❌ [SERVER] ERROR during store_graph_memories_batch: {e}")
        return json.dumps({"status": "error", "message": str(e)})

@mcp.tool(name="filter_graph_memories",
          description="Filters memories based on structured criteria like time, event type, or session ID, not on semantic content.")
async def filter_graph_memories(
    filters: Dict[str, Any], 
    limit: int = 100,
    offset: int = 0
):
    print(f"🛠️ [SERVER] --- Tool: filter_graph_memories ---")
    if not GRAPH_DB_INSTANCE:
        raise Exception("Database connection is not available.")

    try:
        results = await GRAPH_DB_INSTANCE.filter_memories(filters, limit, offset)
        message = f"Filter query returned {len(results)} memories."
        print(f"   - ✅ [SERVER] {message}")
        return json.dumps({"status": "success", "filtered_memories": results})
    except Exception as e:
        print(f"   - ❌ [SERVER] ERROR during filter_graph_memories: {e}")
        return json.dumps({"status": "error", "message": str(e)})

@mcp.tool(name="get_graph_memory_statistics",
          description="Retrieves operational statistics about the memories in the graph database.")
async def get_graph_memory_statistics():
    print(f"🛠️ [SERVER] --- Tool: get_graph_memory_statistics ---")
    if not GRAPH_DB_INSTANCE:
        raise Exception("Database connection is not available.")
    
    try:
        stats = await GRAPH_DB_INSTANCE.get_statistics()
        print(f"   - ✅ [SERVER] Successfully retrieved statistics.")
        return json.dumps({"status": "success", "statistics": stats})
    except Exception as e:
        print(f"   - ❌ [SERVER] ERROR during get_graph_memory_statistics: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool(name="retrieve_graph_memory",
          description="Retrieves memories from the Neo4j graph database using a direct full-text search.")
async def retrieve_graph_memory(
    search_query: str, 
    limit: int = 5
):
    print(f"🛠️ [SERVER] --- Tool: retrieve_graph_memory ---")
    if not GRAPH_DB_INSTANCE:
        print("   - ❌ [SERVER] ERROR: Database connection is not available.")
        error_response = { "status": "error", "message": "Database connection is not available." }
        return json.dumps(error_response)

    try:
        search_results = await GRAPH_DB_INSTANCE.retrieve_memory(search_query, limit)
        print(f"   - ✅ [SERVER] Retrieved {len(search_results)} memories from DB.")
        response_data = { "status": "success", "retrieved_memories": search_results }
        return json.dumps(response_data)
    except Exception as e:
        print(f"   - ❌ [SERVER] ERROR during retrieve_graph_memory: {e}")
        error_message = f"An internal error occurred in retrieve_graph_memory: {str(e)}"
        error_response = { "status": "error", "message": error_message, "exception_type": type(e).__name__ }
        return json.dumps(error_response)

@mcp.tool(name="retrieve_memory_with_expansion",
          description="Retrieves memories by searching and then expanding to find related memories via SIMILAR_TO links.")
async def retrieve_memory_with_expansion(
    search_query: str,
    limit: int = 10
):

    print(f"🛠️ [SERVER] --- Tool: retrieve_memory_with_expansion ---")
    if not GRAPH_DB_INSTANCE:
        raise Exception("Database connection is not available.")
    
    try:
        search_results = await GRAPH_DB_INSTANCE.retrieve_memory_with_expansion(search_query, limit)
        print(f"   - ✅ [SERVER] Retrieved {len(search_results)} memories using expansion search.")
        response_data = {"status": "success", "retrieved_memories": search_results}
        return json.dumps(response_data)
    except Exception as e:
        print(f"   - ❌ [SERVER] ERROR during retrieve_memory_with_expansion: {e}")
        error_response = {"status": "error", "message": f"An internal error occurred: {str(e)}"}
        return json.dumps(error_response)

@mcp.tool(name="prune_graph_memories",
          description="Deletes old and irrelevant memories to keep the database clean and efficient.")
async def prune_graph_memories(
    max_age_days: int = 180,
    min_lookup_count: int = 1
):

    print(f"🛠️ [SERVER] --- Tool: prune_graph_memories ---")
    if not GRAPH_DB_INSTANCE:
        raise Exception("Database connection is not available.")

    try:
        deleted_count = await GRAPH_DB_INSTANCE.prune_memories(max_age_days, min_lookup_count)
        message = f"Successfully pruned {deleted_count} old or irrelevant memories."
        print(f"   - ✅ [SERVER] {message}")
        response_data = {"status": "success", "deleted_count": deleted_count, "message": message}
        return json.dumps(response_data)
    except Exception as e:
        print(f"   - ❌ [SERVER] ERROR during prune_graph_memories: {e}")
        error_response = {"status": "error", "message": f"An internal error occurred during pruning: {str(e)}"}
        return json.dumps(error_response)


@mcp.tool(name="create_relationship",
          description="Creates a directed relationship between two existing memory nodes to link them contextually.")
async def create_relationship(
    source_memory_id: str,
    target_memory_id: str,
    relationship_type: str
):

    print(f"🛠️ [SERVER] --- Tool: create_relationship ---")
    if not GRAPH_DB_INSTANCE:
        raise Exception("Database connection is not available.")

    try:
        rel_type = await GRAPH_DB_INSTANCE.create_relationship(source_memory_id, target_memory_id, relationship_type)
        if rel_type:
            print(f"   - ✅ [SERVER] Relationship '{rel_type}' created successfully.")
            response_data = { "status": "success", "message": f"Relationship '{rel_type}' created from {source_memory_id} to {target_memory_id}." }
            return json.dumps(response_data)
        else:
            raise Exception("Failed to create relationship. Check if both memory IDs exist.")
    except Exception as e:
        print(f"   - ❌ [SERVER] ERROR during create_relationship: {e}")
        error_response = { "status": "error", "message": f"An internal error occurred in create_relationship: {str(e)}" }
        return json.dumps(error_response)

app = mcp.streamable_http_app()
