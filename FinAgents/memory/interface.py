import openai
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import asyncio
import httpx 
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp import types as mcp_types


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
client = openai.OpenAI(api_key=OPENAI_API_KEY)


MCP_SERVER_URL = "http://127.0.0.1:8000/mcp" 

tools_definition = [
    {
        "type": "function",
        "function": {
            "name": "store_graph_memory",
            "description": "Stores a structured memory about a topic, query, or event into the Neo4j graph database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The original question or topic that this memory is about."},
                    "keywords": {"type": "array", "items": {"type": "string"}, "description": "A list of important keywords related to the memory."},
                    "summary": {"type": "string", "description": "A concise summary of the memory's content."},
                    "agent_id": {"type": "string", "description": "The unique identifier of the agent storing this memory."}
                },
                "required": ["query", "keywords", "summary", "agent_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_graph_memory",
            "description": "Retrieves memories from the Neo4j graph database using a fast, direct full-text search on keywords and summaries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {"type": "string", "description": "The search query, keywords, or question to find relevant memories for."},
                    "limit": {"type": "integer", "description": "The maximum number of memories to retrieve. Defaults to 5."}
                },
                "required": ["search_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_memory_with_expansion",
            "description": "A comprehensive search that first finds memories via full-text search, then expands to include related memories connected by 'SIMILAR_TO' links. Use this for broader, more contextual queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {"type": "string", "description": "The search query, keywords, or question to find relevant memories for."},
                    "limit": {"type": "integer", "description": "The maximum number of combined memories to retrieve. Defaults to 10."}
                },
                "required": ["search_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_relationship",
            "description": "Creates a directed relationship between two existing memory nodes to link them contextually (e.g., one memory clarifies, contradicts, or follows another).",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_memory_id": {"type": "string", "description": "The 'memory_id' of the memory node where the relationship starts."},
                    "target_memory_id": {"type": "string", "description": "The 'memory_id' of the memory node where the relationship ends."},
                    "relationship_type": {"type": "string", "description": "The type of the relationship in uppercase snake_case (e.g., 'RELATES_TO', 'CONTRADICTS', 'CLARIFIES')."}
                },
                "required": ["source_memory_id", "target_memory_id", "relationship_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "prune_graph_memories",
            "description": "Deletes old and irrelevant memories from the database. Should be used periodically for maintenance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_age_days": {
                        "type": "integer",
                        "description": "The maximum age in days for a memory to be kept, if it's irrelevant. Defaults to 180."
                    },
                    "min_lookup_count": {
                        "type": "integer",
                        "description": "The minimum lookup_count for a memory to be considered relevant and be kept, even if it's old. Defaults to 1."
                    }
                },
                "required": [],
            },
        },
    }
]

async def call_mcp_tool(session: ClientSession, tool_name: str, tool_args: Dict[str, Any]) -> Any:
    """Helper function to call a tool on the MCP server and handle the response."""
    print(f"📞 [CLIENT] Calling MCP tool: {tool_name} with args: {json.dumps(tool_args, indent=2)}")

    try:
        mcp_response = await session.call_tool(tool_name, tool_args)
        print(f"📦 [CLIENT] Received raw response object from MCP: {mcp_response}")


        if mcp_response.isError or not mcp_response.content:
            error_message = f"Error from MCP tool '{tool_name}'."
            if mcp_response.meta and mcp_response.meta.get("error_message"): 
                error_message += f" Details: {mcp_response.meta['error_message']}"
            print(f"❌ [CLIENT] {error_message}")
            return {"error": error_message}

        if isinstance(mcp_response.content[0], mcp_types.TextContent):
            tool_result_text = mcp_response.content[0].text
            print(f"✅ [CLIENT] MCP tool '{tool_name}' successful. Raw text result: {tool_result_text}")
            try:
                return json.loads(tool_result_text) 
            except json.JSONDecodeError:
                print(f"⚠️ [CLIENT] MCP tool '{tool_name}' result was not valid JSON: {tool_result_text}")
                return {"raw_output": tool_result_text}
        else:
            print(f"⚠️ [CLIENT] Unexpected content type from MCP tool '{tool_name}': {type(mcp_response.content[0])}")
            return {"error": "Unexpected content type from MCP tool"}

    except Exception as e:
        print(f"❌ [CLIENT] Exception during MCP tool call '{tool_name}': {e}")
        return {"error": str(e), "exception_type": type(e).__name__}


async def run_conversation_with_tools(user_prompt: str, mcp_session: ClientSession):
    """Orchestrates a conversation turn with OpenAI, allowing for tool calls."""
    messages = [{"role": "user", "content": user_prompt}]
    print(f"\n👤 [CLIENT] User Prompt: {user_prompt}")

    for i in range(5): 
        print(f"\n--- Iteration {i+1} ---")
        try:
            print("💬 [CLIENT] Sending request to OpenAI...")
            response = client.chat.completions.create(
                model="gpt-4o", 
                messages=messages,
                tools=tools_definition,
                tool_choice="auto", 
            )
            response_message = response.choices[0].message
            print(f"🤖 [CLIENT] Received response from OpenAI: Finish Reason = {response.choices[0].finish_reason}")

            tool_calls = response_message.tool_calls

            if tool_calls:
                print(f"🧠 [CLIENT] Assistant wants to call {len(tool_calls)} tool(s).")
                messages.append(response_message)

                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    function_response = await call_mcp_tool(mcp_session, function_name, function_args)
                    
                    print(f"📤 [CLIENT] Appending tool response to conversation history for tool_call_id: {tool_call.id}")
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(function_response),
                    })
            else:
                assistant_response = response_message.content
                print(f"\n🏁 [CLIENT] Final Agent Response: {assistant_response}")
                return assistant_response 

        except openai.APIError as e:
            print(f"❌ [CLIENT] OpenAI API Error: {e}")
            return f"Sorry, I encountered an API error: {e}"
        except Exception as e:
            print(f"❌ [CLIENT] An unexpected error occurred: {e}")
            return f"Sorry, an unexpected error occurred: {e}"
    
    return "Max tool iterations reached. The conversation might be stuck in a loop."


async def main():
    """Main function to connect to the MCP server and start the chat loop."""
    print(f"[CLIENT] Attempting to connect to MCP server at {MCP_SERVER_URL}...")
    try:
        async with streamablehttp_client(MCP_SERVER_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                print("✅ [CLIENT] Connected to MCP server's session.\n")
                
                while True:
                    try:
                        user_query = input("👤 You (or 'exit' to end): ")
                        if user_query.lower() in ["exit", "quit"]:
                            print("🤖 Agent: Goodbye!")
                            break
                        if not user_query.strip():
                            continue
                        
                        await run_conversation_with_tools(user_query, session)
                        print("-" * 60)
                    except (KeyboardInterrupt, EOFError):
                        print("\n🤖 Agent: Goodbye!")
                        break
                    except Exception as e: 
                        print(f"❌ [CLIENT] An error occurred in the input loop: {e}")
                        break

    except httpx.ConnectError as e:
        print(f"❌ [CLIENT] Connection Error: Failed to connect to MCP server at {MCP_SERVER_URL}.")
        print(f"   Is the server running? `uvicorn server:app --reload`")
    except Exception as e:
        print(f"❌ [CLIENT] An unexpected error occurred during MCP client setup: {e}")

if __name__ == "__main__":
    asyncio.run(main())