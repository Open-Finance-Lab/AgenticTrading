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

            # Test: List all memory keys (should include AAPL_close_... keys if autoloaded)
            keys = await session.call_tool("list_memory_keys", {})
            print("Memory keys:", keys)

            # Test: Get a specific close price from memory
            close_val = await session.call_tool("get_memory", {"key": "AAPL_close_2024-01-02 05:00:00"})
            print("AAPL_close_2024-01-02_05:00:00:", close_val)

            # Test: Set and get a custom value
            await session.call_tool("set_memory", {"key": "test_key", "value": 123.45})
            test_val = await session.call_tool("get_memory", {"key": "test_key"})
            print("test_key:", test_val)

            # Test: Delete the custom value
            await session.call_tool("delete_memory", {"key": "test_key"})
            deleted_val = await session.call_tool("get_memory", {"key": "test_key"})
            print("test_key after delete:", deleted_val)

if __name__ == "__main__":
    asyncio.run(main())

