

payload = {
    "content": "The sky is blue and the grass is green.",
    "category": "General Fact",
    "source_agent_id": "fact_importer",
    "timestamp": None,
    "additional_metadata": {
        "source_url": "http://facts.com",
        "verified": True
    }
}

payload2 = {
    "query": "What is the sky?",
    "k": 1
}

import asyncio
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession

async def main():
    url = "http://localhost:8000/mcp"

    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            print("Connected to MCP server.\n")


            try:
                response = await session.call_tool(
                    "store_memory",
                    payload
                )
                print("Response:", response)
            except Exception as e:
                print("Error during agent.execute:", e)



            try:
                response = await session.call_tool(
                    "retrieve_memory",
                    payload2
                )
                print("Response:", response)
            except Exception as e:
                print("Error during agent.execute:", e)

if __name__ == "__main__":
    asyncio.run(main())
    