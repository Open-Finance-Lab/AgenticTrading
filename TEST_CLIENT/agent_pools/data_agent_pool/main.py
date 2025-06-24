import os
import multiprocessing
from dotenv import load_dotenv
from servers.equity import polygon_mcp

def start_mcp(transport):
    polygon_mcp.run(transport)

if __name__ == "__main__":
    load_dotenv()
    transport = os.environ.get("MCP_TRANSPORT")

    server_proc = multiprocessing.Process(target=start_mcp, args=(transport,))
    server_proc.daemon = True
    server_proc.start()

    from agents.equity.polygon_agent import agent
    import asyncio

    async def call_agent():
        resp = await agent.ainvoke({
            "messages": [{"role": "user", "content": "Get the latest price for AAPL stock"}]
        })
        print(resp)

    asyncio.run(call_agent())
