import asyncio
from mcp.client.sse import sse_client
from mcp import ClientSession

async def main():
    # Connect to MCP SSE service
    url = "http://localhost:5050/sse"
    async with sse_client(url, headers={}) as (read_stream, write_stream):
        session = ClientSession(read_stream, write_stream)
        async with session:
            # Initialize the session
            await session.initialize()
            print("Session initialized.")

            # Call the tool   
            result = await session.call_tool("start_agent", {"agent_id": "momentum_agent"})
            print("start_agent result:", result)

            result = await session.call_tool("list_agents", {})
            print("list_agents result:", result)

if __name__ == "__main__":
    asyncio.run(main())

