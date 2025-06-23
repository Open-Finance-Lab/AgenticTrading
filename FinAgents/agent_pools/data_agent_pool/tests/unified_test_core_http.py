import asyncio
import httpx
import json

MCP_URL = "http://localhost:8000/mcp"

async def call_tool_http(method, arguments, id=1):
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": method,
            "arguments": arguments
        },
        "id": id
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(MCP_URL, json=payload, headers=headers)
        print(f"\n🔧 Call {method}({arguments})")
        print("Status:", resp.status_code)
        try:
            print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
        except Exception:
            print(resp.text)
        return resp

async def main():
    print("✅ Testing FastAPI core server via HTTP JSON-RPC...")

    # 1. 初始化 agent
    await call_tool_http("init_agent", {"agent_id": "binance_agent"}, id=1)
    await call_tool_http("init_agent", {"agent_id": "coinbase_agent"}, id=2)
    await call_tool_http("init_agent", {"agent_id": "alpaca_agent"}, id=3)
    await call_tool_http("init_agent", {"agent_id": "iex_agent"}, id=4)
    await call_tool_http("init_agent", {"agent_id": "newsapi_agent"}, id=5)
    await call_tool_http("init_agent", {"agent_id": "rss_agent"}, id=6)

    # 2. agent.execute 测试
    await call_tool_http("agent.execute", {
        "agent_id": "binance_agent",
        "function": "fetch_ohlcv",
        "input": {"symbol": "BTCUSDT", "interval": "1h"}
    }, id=10)

    await call_tool_http("agent.execute", {
        "agent_id": "coinbase_agent",
        "function": "fetch_price",
        "input": {"symbol": "ETHUSD"}
    }, id=11)

    await call_tool_http("agent.execute", {
        "agent_id": "alpaca_agent",
        "function": "fetch_equity_data",
        "input": {"ticker": "AAPL"}
    }, id=12)

    await call_tool_http("agent.execute", {
        "agent_id": "iex_agent",
        "function": "get_quote",
        "input": {"ticker": "TSLA"}
    }, id=13)

    await call_tool_http("agent.execute", {
        "agent_id": "newsapi_agent",
        "function": "fetch_headlines",
        "input": {"topic": "economy"}
    }, id=14)

    await call_tool_http("agent.execute", {
        "agent_id": "rss_agent",
        "function": "pull_feed",
        "input": {"feed_url": "https://example.com/rss"}
    }, id=15)

    # 3. 其它工具
    await call_tool_http("list_agents", {}, id=20)
    await call_tool_http("health_check", {}, id=21)


if __name__ == "__main__":
    asyncio.run(main())