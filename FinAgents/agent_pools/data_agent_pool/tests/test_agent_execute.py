import asyncio
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession

async def main():
    url = "http://localhost:8001/mcp"  # 注意：必须是原生 MCP 服务端口

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