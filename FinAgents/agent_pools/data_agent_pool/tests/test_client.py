# import asyncio
# from mcp.client.streamable_http import streamablehttp_client
# from mcp.client.session import ClientSession

# async def main():
#     # URL for the MCP endpoint
#     url = "http://localhost:8001/mcp"

#     # Establish a streamable HTTP connection to the MCP server
#     async with streamablehttp_client(url) as (read, write, _):
#         # Create a client session for JSON-RPC interaction
#         async with ClientSession(read, write) as session:
#             await session.initialize()
#             print("✅ Connected to MCP server.")

#             # -------------------
#             # 1. Test 'register' resource
#             # -------------------
#             print("\\n[REGISTER RESOURCE]")
#             register_response = await session.read_resource("register://test-agent")
#             print("Response:", register_response)

#             # -------------------
#             # 2. Test 'heartbeat' resource
#             # -------------------
#             print("\\n[HEARTBEAT RESOURCE]")
#             heartbeat_response = await session.read_resource("heartbeat://test-agent")
#             print("Response:", heartbeat_response)

#             # -------------------
#             # 3. Test 'agent.execute' tool
#             # -------------------
#             print("\\n[AGENT EXECUTE TOOL]")
#             tool_response = await session.call_tool(
#                 "agent.execute",
#                 {
#                     "agent_id": "test-agent",
#                     "function": "echo",
#                     "input": {"message": "Hello MCP"}
#                 }
#             )
#             print("Response:", tool_response)

# if __name__ == "__main__":
#     asyncio.run(main())
# test_client.py

import asyncio
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession

async def main():
    url = "http://localhost:8001/mcp"

    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            print("✅ Connected to MCP server.\n")

            # Step 1: Call agent.execute
            print("[AGENT EXECUTE TOOL]")
            try:
                response = await session.call_tool(
                    "agent.execute",
                    {
                        "agent_id": "binance_agent",
                        "function": "fetch_ohlcv",
                        "input": {
                            "symbol": "BTCUSDT",
                            "interval": "1h"
                        }
                    }
                )
                print("Response:", response)
            except Exception as e:
                print("❌ Error during agent.execute:", e)

if __name__ == "__main__":
    asyncio.run(main())
    