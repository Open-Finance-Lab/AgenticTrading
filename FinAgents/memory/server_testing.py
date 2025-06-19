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
    print("Lifespan event: Initializing application context.")

    current_retriever_instance = retriever 

    try:
        yield AppContext(chroma_service=current_retriever_instance)
    finally:
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
        print(f"Document added to ChromaDB successfully. ID: {memory_id}")
        return {
            "memory_id": memory_id,
            "status_message": "Memory content processed and stored successfully in Vector DB.",
            "generated_keywords": metadata_for_chroma.get("keywords", []),
            "generated_context": metadata_for_chroma.get("context_summary", "")
        }
    except Exception as e:
        print(e)
        raise e


app = mcp.streamable_http_app()