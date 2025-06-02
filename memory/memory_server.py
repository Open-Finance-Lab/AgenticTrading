from mcp.server.fastmcp import FastMCP
from chroma_retriever import ChromaRetriever
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator 
from dataclasses import dataclass


retriever = ChromaRetriever(collection_name="mcp_agent_memory")


@dataclass
class AppContext:
    chroma_service: ChromaRetriever 

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with type-safe context"""
    # The 'retriever' instance is already created globally.
    # For many ChromaDB client setups (especially in-memory), explicit connect/disconnect
    # after __init__ isn't standard via separate methods on a wrapper like ChromaRetriever.
    # The client is initialized within ChromaRetriever's __init__.
    # If specific setup/teardown for the underlying chromadb.Client were needed,
    # you'd add methods to ChromaRetriever and call them here.

    print("Lifespan event: Initializing application context.")
    # For now, we'll assume the globally initialized 'retriever' is the service
    # we want to make available through the lifespan context.
    current_retriever_instance = retriever 

    try:
        yield AppContext(chroma_service=current_retriever_instance)
    finally:
        # For example: current_retriever_instance.client.reset() # if you want to clear it
        print("Lifespan event: Cleaning up application context.")

mcp = FastMCP(
    "Memory",
    lifespan=app_lifespan, 
    stateless_http=True,  
    debug=True            
)

@mcp.tool(name="store_memory",
          description="Call to Agent to Store Sent Memory in its Database")
def store_memory(content: str, category: str,
                 source_agent_id: Optional[str],
                 timestamp: Optional[str],
                 additional_metadata: Optional[Dict[str, Any]]):

    print(f"🛠️ Called store memory: {content[:50]}")
    actual_timestamp = timestamp or datetime.now().isoformat()
    memory_id = str(uuid.uuid4())
    metadata_for_chroma = {
        "category": category,
        "source_agent_id": source_agent_id or "unknown",
        "original_timestamp": actual_timestamp,
        **(additional_metadata or {})
    }
    try:
        retriever.add_document(
            document=content,
            metadata=metadata_for_chroma,
            doc_id=memory_id
        )
        print(f"✅ Document added to ChromaDB successfully. ID: {memory_id}")
        return {
            "memory_id": memory_id,
            "status_message": "Memory content processed and stored successfully in Vector DB.",
            "generated_keywords": metadata_for_chroma.get("keywords", []),
            "generated_context": metadata_for_chroma.get("context_summary", "")
        }
    except Exception as e:
        print(f"❌ ERROR during retriever.add_document: {e}")
        raise e

@mcp.tool(name="retrieve_memory",
          description="Call to Agent to retrieve from Database")
def retrieve_memory(query: str, k: int):
    print(f"🛠️ Called Retreive Memory: {query[:50]}")

    try:
        search_results = retriever.search(query=query,k=k)

        retrieved_items: List[Dict[str, Any]] = []

        if search_results and search_results.get('ids') and search_results['ids'][0]:
            ids_list = search_results['ids'][0]
            docs_list = search_results.get('documents', [[]])[0]
            metadatas_list = search_results.get('metadatas', [[]])[0]
            distances_list = search_results.get('distances', [[]])[0]
            
            for i in range(len(ids_list)):
                retrieved_items.append({
                    "id": ids_list[i],
                    "document": docs_list[i] if i < len(docs_list) else "",
                    "metadata": metadatas_list[i] if i < len(metadatas_list) else {},
                    "distance": distances_list[i] if i < len(distances_list) else None
                })
        
        print(f"✅ Formatted {len(retrieved_items)} retrieved items.")
        
        return {
            "retrieved_memories": retrieved_items,
            "status_message": f"Retrieved {len(retrieved_items)} memories."
        }
        
    except Exception as e:
        print(f"❌ ERROR during retriever.search: {e}")
        raise e
# What Uvi rus
app = mcp.streamable_http_app()