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

#.env has OpenAI key in main dir
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dotenv_path = os.path.join(project_root, '.env')
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)


MCP_SERVER_URL = "http://127.0.0.1:8000/mcp" 


tools_definition = [
    {
        "type": "function",
        "function": {
            "name": "store_memory", 
            "description": "Stores a piece of information or memory into the memory database. Use this to remember facts, user statements, or context for future reference.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The textual content of the memory to store."
                    },
                    "category": {
                        "type": "string",
                        "description": "A category for the memory (e.g., 'User Preference', 'Fact', 'Meeting Note'). Defaults to 'General'."
                    },
                    "source_agent_id": {
                        "type": "string",
                        "description": "The ID of the agent or source providing this memory. Defaults to 'openai_gpt4o_chat_agent'."
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "An ISO 8601 timestamp for when the memory occurred or was recorded. Can be null or omitted to auto-generate."
                    },
                    "additional_metadata": {
                        "type": "object",
                        "description": "Any other structured data to store with the memory as key-value pairs. Can be an empty object if no additional metadata.",
                        "additionalProperties": True
                    }
                },
                "required": ["content"], 
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_memory", 
            "description": "Retrieves relevant memories from the database based on a textual query. Use this to recall past information, facts, or context related to the current conversation to help answer a question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or question to find relevant memories for."
                    },
                    "k": {
                        "type": "integer",
                        "description": "The maximum number of memories to retrieve. Optional, server defaults to 3 if not provided."
                    }
                },
                "required": ["query"],
            },
        },
    }
]

async def call_mcp_tool(session: ClientSession, tool_name: str, tool_args: Dict[str, Any]) -> Any:
    print(f"📞 Calling MCP tool: {tool_name} with raw args from OpenAI: {json.dumps(tool_args, indent=2)}")

    if tool_name == "store_memory":
        if "category" not in tool_args or tool_args["category"] is None:
            tool_args["category"] = "General"
        if "source_agent_id" not in tool_args or tool_args["source_agent_id"] is None:
            tool_args["source_agent_id"] = "openai_gpt4o_chat_agent" 
        if "timestamp" not in tool_args:
             tool_args["timestamp"] = None
        if "additional_metadata" not in tool_args:
            tool_args["additional_metadata"] = {}
    elif tool_name == "retrieve_memory":
        if "k" not in tool_args or tool_args["k"] is None:
            tool_args["k"] = tool_args.get("k", 3) 


    print(f"   Processed args for MCP server: {json.dumps(tool_args, indent=2)}")

    try:
        mcp_response = await session.call_tool(tool_name, tool_args)

        if mcp_response.isError or not mcp_response.content:
            error_message = f"Error from MCP tool '{tool_name}'."
            if mcp_response.content and isinstance(mcp_response.content[0], mcp_types.TextContent):
                error_message += f" Details: {mcp_response.content[0].text}"
            elif mcp_response.meta and mcp_response.meta.get("error_message"): 
                error_message += f" Details: {mcp_response.meta['error_message']}"
            print(f"❌ {error_message}")
            return {"error": error_message, "mcp_tool_error_details": error_message} 

        
        if isinstance(mcp_response.content[0], mcp_types.TextContent):
            tool_result_text = mcp_response.content[0].text
            print(f"✅ MCP tool '{tool_name}' successful. Raw result: {tool_result_text}")
            try:

                return json.loads(tool_result_text) 
            except json.JSONDecodeError:
                print(f"⚠️ MCP tool '{tool_name}' result was not valid JSON: {tool_result_text}")
                return {"raw_output": tool_result_text} 
        else:
            print(f"⚠️ Unexpected content type from MCP tool '{tool_name}': {type(mcp_response.content[0])}")
            return {"error": "Unexpected content type from MCP tool"}

    except Exception as e:
        print(f"❌ Exception during MCP tool call '{tool_name}': {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "exception_type": type(e).__name__}


async def run_conversation_with_tools(user_prompt: str, mcp_session: ClientSession):
    messages = [{"role": "user", "content": user_prompt}]
    print(f"\n👤 User: {user_prompt}")

    for iteration_count in range(5): 
        try:
            response = client.chat.completions.create(
                model="gpt-4o", 
                messages=messages,
                tools=tools_definition,
                tool_choice="auto", 
            )
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            if tool_calls:
                messages.append(response_message) 

                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args_str = tool_call.function.arguments
                    try:
                        function_args = json.loads(function_args_str)
                    except json.JSONDecodeError:
                        print(f"Error: Could not decode arguments for {function_name}: {function_args_str}")
                        function_response_content = f"Error: Invalid arguments provided for {function_name}. Arguments string was: {function_args_str}"
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response_content,
                        })
                        continue 

                    print(f"   - Tool: {function_name}")
                    print(f"   - Arguments from OpenAI: {json.dumps(function_args, indent=2)}")

                    function_response = await call_mcp_tool(mcp_session, function_name, function_args)
                    
                    if isinstance(function_response, (dict, list)):
                        response_content_str = json.dumps(function_response)
                    elif isinstance(function_response, str):
                        response_content_str = function_response
                    elif function_response is None:
                         response_content_str = json.dumps({"status": "Tool executed but returned None."})
                    else:
                        response_content_str = str(function_response)


                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": response_content_str,
                    })
            else:
                assistant_response = response_message.content
                print(f"\n🤖 Agent: {assistant_response}")
                return assistant_response 

        except openai.APIError as e:
            print(f"OpenAI API Error: {e}")
            return f"Sorry, I encountered an API error: {e}"
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
            return f"Sorry, an unexpected error occurred: {e}"
    
    print("Max tool iterations reached. Returning last known state or error.")
    return "Max tool iterations reached. The conversation might be stuck in a loop or require more steps."


async def main():
    print(f"Attempting to connect to MCP server at {MCP_SERVER_URL}...")
    try:
        async with streamablehttp_client(MCP_SERVER_URL) as (read, write, transport_specific_context):
            async with ClientSession(read, write) as session:
                print("✅ Connected to MCP server's session.\n")
                
                while True:
                    try:
                        user_query = input("👤 You (exit to end): ")
                        if user_query.lower() in ["exit", "quit"]:
                            print("Agent: Goodbye!")
                            break
                        if not user_query.strip():
                            continue
                        
                        await run_conversation_with_tools(user_query, session)
                        print("-" * 60)
                    except Exception as e: 
                        print(f"An error occurred in the input loop: {e}")
                        break


    except httpx.ConnectError as e:
        print(f"❌ Connection Error: Failed to connect to MCP server at {MCP_SERVER_URL}.")
        print(f"   Details: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred during MCP client setup: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())