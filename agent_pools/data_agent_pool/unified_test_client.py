# unified_test_client.py

import asyncio
import json
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession

async def call_tool(session, agent_id, function, input_args):
    print(f"\n🔧 Calling {agent_id}.{function} with args: {input_args}")
    try:
        result = await session.call_tool("agent.execute", {
            "agent_id": agent_id,
            "function": function,
            "input": input_args
        })
        if hasattr(result, "text"):
            print(json.dumps(json.loads(result.text), indent=2))
        else:
            print(result)
    except Exception as e:
        print(f"❌ Error calling {agent_id}.{function}: {e}")

async def main():
    url = "http://localhost:8001/mcp"

    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ Connected to MCP server.")

            # Test all registered agents and their default functions
            await call_tool(session, "binance_agent", "fetch_ohlcv", {
                "symbol": "BTCUSDT",
                "interval": "1h"
            })

            await call_tool(session, "coinbase_agent", "fetch_price", {
                "symbol": "ETHUSD"
            })

            await call_tool(session, "alpaca_agent", "fetch_equity_data", {
                "ticker": "AAPL"
            })

            await call_tool(session, "iex_agent", "get_quote", {
                "ticker": "TSLA"
            })

            await call_tool(session, "newsapi_agent", "fetch_headlines", {
                "topic": "economy"
            })

            await call_tool(session, "rss_agent", "pull_feed", {
                "feed_url": "https://example.com/rss"
            })

if __name__ == "__main__":
    asyncio.run(main())