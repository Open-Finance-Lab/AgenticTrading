import openai
import requests
import json
import os
from datetime import datetime
from dotenv import load_dotenv 
from typing import List, Dict, Any

#.env has OpenAI key in main dir
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dotenv_path = os.path.join(project_root, '.env')

load_dotenv(dotenv_path=dotenv_path)

api_key = os.environ.get("OPENAI_API_KEY")
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
                        "description": "An ISO 8601 timestamp for when the memory occurred or was recorded. Will be auto-generated if not provided."
                    },
                    "additional_metadata": {
                        "type": "object",
                        "description": "Any other structured data to store with the memory as key-value pairs.",
                        "additionalProperties": True
                    }
                },
                "required": ["content"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_memories",
            "description": "Retrieves relevant memories from the database based on a textual query. "
            "Use this to recall past information, facts, or context related to the current conversation to help answer a question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or question to find relevant memories for."
                    },
                    "k": {
                        "type": "integer",
                        "description": "The maximum number of memories to retrieve. Optional, defaults to 3."
                    }
                },
                "required": ["query"],
            },
        }
    }
]





def call_mcp_server_tool(function_name: str, function_args: dict) -> dict:
    """Calls the specified tool on the MCP server."""
    print(f"Attempting to call MCP tool: {function_name} with args: {function_args}")
    payload = {
        "method": "agent.execute",
        "params": {
            "agent_id": "openai_gpt4o_chat_agent", 
            "function": function_name,
            "input": function_args
        },
        "id": f"chat-req-{datetime.now().timestamp()}"
    }


    #debugging/errors
    try:
        response = requests.post(MCP_SERVER_URL, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        print(f"MCP Server Response ({response.status_code}): {response.json()}")
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error calling MCP server for {function_name}: {http_err}")
        raw_response_text = http_err.response.text if http_err.response else "N/A"
        try:
            error_details = http_err.response.json()
        except json.JSONDecodeError:
            error_details = raw_response_text
        return {"error": {"message": str(http_err), "type": "mcp_http_error", "status_code": http_err.response.status_code, "details": error_details, "function_called": function_name}}
    except requests.exceptions.RequestException as e:
        print(f"Error calling MCP server for {function_name}: {e}")
        return {"error": {"message": str(e), "type": "mcp_request_failed", "function_called": function_name}}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from MCP server for {function_name}: {e}")
        return {"error": {"message": "Failed to decode JSON response.", "type": "mcp_response_invalid_json", "raw_response": response.text if 'response' in locals() else "N/A"}}




#main loop
def run_conversation():
    messages = [{"role": "system", "content": "You are a helpful financial memory assistant."
    " You have tools to store and retrieve memories. When retrieving memories, analyze the user's "
    "query to determine if the retrieved information should be used to directly answer the query or "
    "if further synthesis is needed. Always aim to be factual and use stored memories to enhance your responses."}]
    
    print("Agent: You are now chatting with memory agent that is connected to the DB. Type 'exit' to end.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})
        original_user_query_for_rag = user_input
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=tools_definition,
                tool_choice="auto",
            )
            response_message = response.choices[0].message
            messages.append(response_message)

            if response_message.tool_calls:
                print("Agent wants to use a tool.")
                tool_call_results = []

                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    print(f"System: Calling tool '{function_name}' with arguments: {function_args}")

                    if function_name == "store_memory":
                        if "source_agent_id" not in function_args:
                            function_args["source_agent_id"] = "openai_gpt4o_chat_agent"
                        
                        tool_response_data = call_mcp_server_tool(
                            function_name=function_name,
                            function_args=function_args
                        )
                        tool_call_results.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": json.dumps(tool_response_data), #send back json string of the mcp response
                        })

                    elif function_name == "retrieve_memories":
                        retrieved_data_from_mcp = call_mcp_server_tool(
                            function_name=function_name,
                            function_args=function_args
                        )

                        if retrieved_data_from_mcp.get("error") or \
                           not retrieved_data_from_mcp.get("result") or \
                           not retrieved_data_from_mcp["result"].get("retrieved_memories"):
                            
                            tool_call_results.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": json.dumps({"status": "No relevant memories found or error in retrieval.", "retrieved_memories": []}),
                            })
                            continue

                        retrieved_memories = retrieved_data_from_mcp["result"]["retrieved_memories"]
                        
                        #augment the prompt with retrieved memories
                        context_str = "Based on the following retrieved information from my memory:\n"
                        if retrieved_memories:
                            for i, mem_item in enumerate(retrieved_memories):
                                content = mem_item.get('document', '')
                                context_str += f"{i+1}. (Memory ID: {mem_item.get('id', 'N/A')}) {content}\n"
                        
                        rag_question = original_user_query_for_rag 
                        augmented_prompt = f"{context_str}\nNow, please answer the user's request: \"{rag_question}\""
                        
                        print(f"\n--- RAG: Augmented Prompt for LLM ---\n{augmented_prompt}\n------------------------------------")

                        rag_response_messages = [
                            {"role": "system", "content": "You are a helpful assistant. Base your answer *only* on the provided context. "
                            "If the context isn't sufficient or relevant to the question, clearly state that the provided information doesn't answer the question."},
                            {"role": "user", "content": augmented_prompt}
                        ]
                        
                        rag_llm_response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=rag_response_messages,
                            temperature=0.3 #more factual response with lower temperature.... less randomness
                        )
                        final_rag_answer = rag_llm_response.choices[0].message.content
                        print(f"RAG: LLM generated answer: {final_rag_answer}")

                        tool_call_results.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": json.dumps({
                                "status": "Successfully retrieved and synthesized information.", 
                                "synthesized_answer": final_rag_answer, #RAG results
                                "source_memory_ids": [mem.get('id') for mem in retrieved_memories if mem.get('id')] 
                            }), 
                        })
                
                if tool_call_results:
                    messages.extend(tool_call_results)
                    final_response_after_tools = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                    )
                    final_gpt_message = final_response_after_tools.choices[0].message
                    print(f"GPT (after tool processing): {final_gpt_message.content}")
                    messages.append(final_gpt_message)
            else:
                print(f"GPT: {response_message.content}")

        except Exception as e:
            print(f"An error occurred in the conversation loop: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_conversation()