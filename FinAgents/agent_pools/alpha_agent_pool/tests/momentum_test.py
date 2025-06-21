import asyncio
from mcp.client.sse import sse_client
from mcp import ClientSession

async def main():
    """
    Entry point for testing the MCP SSE client and tool invocation.
    Establishes a connection to the MCP SSE service, initializes the session,
    and invokes the 'generate_signal' tool with a sample request payload.
    """
    url = "http://localhost:5051/sse"  # URL of the MCP SSE service endpoint
    async with sse_client(url, headers={}) as (read_stream, write_stream):
        session = ClientSession(read_stream, write_stream)
        async with session:
            # Initialize the client session with the MCP server
            await session.initialize()
            print("Session initialized.")

            # Invoke the 'generate_signal' tool with a sample symbol and price list
            result = await session.call_tool("generate_signal", {
                "request": {
                    "symbol": "AAPL",
                    "price_list": [100.0, 101.5, 102.3, 104.0, 106.0]
                }
            })
            print("generate_signal result:", result)

if __name__ == "__main__":
    # Run the asynchronous main function as the script entry point
    asyncio.run(main())