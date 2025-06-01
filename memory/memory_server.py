from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from datetime import datetime
import uuid
import json
from typing import Dict, Any, Optional, List, Union
import traceback 
from chroma_retriever import ChromaRetriever
from pydantic_models import (
    MCPRequest, StoreMemoryInput, StoreMemoryResult, StoreMemoryResponse,
    RetrieveMemoriesInput, RetrievedMemoryItem, RetrieveMemoriesResult, RetrieveMemoriesResponse,
    MCPError, MCPErrorResponse, MCPErrorData, MCPResponseMeta
)

app = FastAPI()

retriever = ChromaRetriever(collection_name="mcp_agent_memory")
print(f"--- ChromaRetriever initialized: {retriever.collection.name}")


def create_error_response(request_id: Optional[Union[str, int]], code: int, message: str, agent_id: Optional[str] = None, details: Optional[str] = None) -> MCPErrorResponse:
    """Creates standardized MCPErrorResponse object."""
    return MCPErrorResponse(
        error=MCPError(
            code=code,
            message=message,
            data=MCPErrorData(agent_id=agent_id, details=details) if agent_id or details else None
        ),
        id=request_id
    )

def handle_store_memory(params_input: Dict[str, Any]) -> StoreMemoryResult:
    """
    Handles the logic for storing a memory.
    Validates input, processes metadata, and adds document to ChromaDB.
    """
    print(f"--- In handle_store_memory, received raw input for validation: {str(params_input)[:500]}...")
    if retriever is None:
        print("!!! Error in handle_store_memory: ChromaRetriever is not available.")
        raise HTTPException(status_code=500, detail="Memory storage (ChromaRetriever) is not available.")
    
    try:
        store_input = StoreMemoryInput(**params_input)
        print(f"--- Pydantic validation for StoreMemoryInput successful.")
    except ValidationError as e:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!!! DETAILED PYDANTIC VALIDATION ERROR for store_memory (json):")
        print(e.json(indent=2))
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        raise HTTPException(status_code=400, detail=e.errors())
    except Exception as e:
        print(f"!!! UNEXPECTED ERROR during StoreMemoryInput model creation: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error processing store_memory input: {str(e)}")

    memory_id = str(uuid.uuid4())
    timestamp = store_input.timestamp or datetime.now().isoformat()

    metadata_for_chroma = {
        "category": store_input.category,
        "source_agent_id": store_input.source_agent_id,
        "original_timestamp": timestamp,
        **(store_input.additional_metadata or {})
    }

    generated_keywords = metadata_for_chroma.get("keywords", []) 
    generated_context = metadata_for_chroma.get("context_summary", "")

    try:
        print(f"--- Attempting to add document to ChromaDB. ID: {memory_id}, Content: '{store_input.content[:100]}...'")
        retriever.add_document(
            document=store_input.content,
            metadata=metadata_for_chroma,
            doc_id=memory_id
        )
        print(f"--- Document added to ChromaDB successfully. ID: {memory_id}")
    except Exception as e:
        print(f"!!! ERROR during retriever.add_document: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to store memory in ChromaDB: {str(e)}")

    return StoreMemoryResult(
        memory_id=memory_id,
        status_message="Memory content processed and stored successfully in Vector DB.",
        generated_keywords=generated_keywords,
        generated_context=generated_context 
    )


def handle_retrieve_memories(params_input: Dict[str, Any]) -> RetrieveMemoriesResult:
    """
    Handles the logic for retrieving memories.
    Validates input, searches ChromaDB, and formats the results.
    """
    print(f"--- In handle_retrieve_memories, received raw input for validation: {params_input}")
    if retriever is None:
        print("!!! Error in handle_retrieve_memories: ChromaRetriever is not available.")
        raise HTTPException(status_code=500, detail="Memory retrieval (ChromaRetriever) is not available.")
    
    try:
        retrieve_input = RetrieveMemoriesInput(**params_input)
        print(f"--- Pydantic validation for RetrieveMemoriesInput successful.")
    except ValidationError as e:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!!! DETAILED PYDANTIC VALIDATION ERROR for retrieve_memories (json):")
        print(e.json(indent=2))
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        raise HTTPException(status_code=400, detail=e.errors())
    except Exception as e:
        print(f"!!! UNEXPECTED ERROR during RetrieveMemoriesInput model creation: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error processing retrieve_memories input: {str(e)}")

    try:
        print(f"--- Attempting to search ChromaDB. Query: '{retrieve_input.query}', k: {retrieve_input.k}")
        search_results_raw = retriever.search(query=retrieve_input.query, k=retrieve_input.k)
        print(f"--- ChromaDB search successful. Raw results snippet: {str(search_results_raw)[:200]}...")
    except Exception as e:
        print(f"!!! ERROR during retriever.search: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to retrieve memories from ChromaDB: {str(e)}")

    retrieved_items: List[RetrievedMemoryItem] = []

    if search_results_raw and search_results_raw.get('ids') and search_results_raw['ids'][0]:
        ids_list = search_results_raw['ids'][0]

        docs_list = search_results_raw.get('documents', [[]])[0]
        metadatas_list = search_results_raw.get('metadatas', [[]])[0]
        distances_list = search_results_raw.get('distances', [[]])[0]

        for i in range(len(ids_list)):
            retrieved_items.append(
                RetrievedMemoryItem(
                    id=ids_list[i],
                    document=docs_list[i] if i < len(docs_list) else "",
                    metadata=metadatas_list[i] if i < len(metadatas_list) else {},
                    distance=distances_list[i] if i < len(distances_list) else None
                )
            )
    print(f"--- Formatted {len(retrieved_items)} retrieved items.")
    return RetrieveMemoriesResult(
        retrieved_memories=retrieved_items,
        status_message=f"Retrieved {len(retrieved_items)} memories."
    )



@app.post("/mcp")
async def mcp_endpoint(request: MCPRequest):
    """Main MCP endpoint to execute agent functions."""
    print(f"--- MCP ENDPOINT CALLED. Request ID: {request.id if request else 'Unknown'}")
    try:
        print(f"--- MCPRequest parsed. Method: {request.method}, Agent ID: {request.params.agent_id}, Function: {request.params.function}, Input Snippet: {str(request.params.input)[:200]}...")
    except Exception as e:
        print(f"!!! ERROR during initial MCPRequest processing (after Pydantic validation by FastAPI): {e}")
        traceback.print_exc()
        req_id_for_error = request.id if request and hasattr(request, 'id') else "unknown"
        error_resp = create_error_response(
            req_id_for_error, -32700, "Parse error", None, "Failed to process initial MCPRequest structure."
        )
        return JSONResponse(status_code=400, content=error_resp.model_dump(exclude_none=True))

    request_id = request.id
    response_meta = MCPResponseMeta(status="success", timestamp=datetime.now().isoformat())
    response_payload = None 

    if request.method == "agent.execute":
        function_name = request.params.function
        params_input = request.params.input 

        try:
            if function_name == "store_memory":
                result = handle_store_memory(params_input)
                response_payload = StoreMemoryResponse(result=result, meta=response_meta, id=request_id)
            elif function_name == "retrieve_memories":
                result = handle_retrieve_memories(params_input)
                response_payload = RetrieveMemoriesResponse(result=result, meta=response_meta, id=request_id)
            else:
                print(f"!!! Function '{function_name}' not supported by this Memory Agent server.")
                error_resp = create_error_response(
                    request_id, -32601, "Method not found", request.params.agent_id, f"Function '{function_name}' not supported."
                )
                return JSONResponse(status_code=404, content=error_resp.model_dump(exclude_none=True))
            
            return JSONResponse(content=response_payload.model_dump(exclude_none=True))

        except HTTPException as http_exc:
            print(f"!!! HTTPException caught in mcp_endpoint for function '{function_name}': {http_exc.status_code} - Detail: {http_exc.detail}")
            error_detail_str = str(http_exc.detail)
            error_resp = create_error_response(
                request_id, -32000, "Agent function execution error", request.params.agent_id, error_detail_str
            )
            return JSONResponse(status_code=http_exc.status_code, content=error_resp.model_dump(exclude_none=True))
        except Exception as e:
            print(f"!!! UNEXPECTED SERVER ERROR in mcp_endpoint for function '{function_name}': {type(e).__name__} - {e}")
            traceback.print_exc()
            error_resp = create_error_response(
                request_id, -32603, "Internal server error", request.params.agent_id, f"An unexpected server error occurred: {str(e)}"
            )
            return JSONResponse(status_code=500, content=error_resp.model_dump(exclude_none=True))
    else:
        print(f"!!! MCP method '{request.method}' not supported.")
        error_resp = create_error_response(
            request_id, -32601, "Method not allowed", None, f"MCP method '{request.method}' not supported. Use 'agent.execute'."
        )
        return JSONResponse(status_code=405, content=error_resp.model_dump(exclude_none=True))

if __name__ == "__main__":
    import uvicorn
    print("--- Starting Memory Agent MCP Server ---")# to run uvicorn memory_server:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
    print("--- Server Shut Down ---")