import asyncio
import os
from dotenv import load_dotenv
from pprint import pprint

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_swarm, create_handoff_tool

load_dotenv()
os.getenv("OPENAI_API_KEY")

async def main():
    client = MultiServerMCPClient(
        {
            "DataTools": {
                "command": "python",
                "args": [
                    "/home/arnav-gr0ver/DEV/Research/GenAI-OpenFinance/"
                    "FinAgent-Orchestration/agent_pools/data_agent_pool/tool_server.py"
                ],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()
    ingest_tool    = next(t for t in tools if t.name == "ingest_ticker_data")
    validate_tool  = next(t for t in tools if t.name == "validate_data")
    transform_tool = next(t for t in tools if t.name == "transform_data")

    transfer_to_validation = create_handoff_tool(
        agent_name="validation_agent",
        description="Pass ingested data to validation_agent",
    )

    transfer_to_transformation = create_handoff_tool(
        agent_name="transformation_agent",
        description="Pass validated data to transformation_agent",
    )

    ingestion_agent = create_react_agent(
        model=ChatOpenAI(model="gpt-4o"),
        tools=[ingest_tool, transfer_to_validation],
        prompt=(
            "You are the **ingestion agent**. Your job:\n"
            "1. Extract **ticker**, **period**, and **interval** from the user's message.\n"
            "   - `ticker`: alphanumeric stock symbol (e.g. AAPL).\n"
            "   - If the user gave a date range, convert it into `period` (e.g. Feb 1–2 2024 → '2d').\n"
            "   - `interval`: a valid yfinance interval (e.g. '1d', '1h', '5m').\n"
            "   - Defaults: `period='5d'`, `interval='1d'` if none provided.\n\n"
            "2. Validate parsed values:\n"
            "   - If `ticker` is missing or malformed, respond with a clear error message (no tool call).\n"
            "3. Otherwise, call `ingest_ticker_data(ticker=..., period=..., interval=...)`.\n"
            "4. After receiving the JSON list of rows, immediately call `transfer_to_validation_agent()` (no arguments).\n"
        ),
        name="ingestion_agent",
    )

    validation_agent = create_react_agent(
        model=ChatOpenAI(model="gpt-4o"),
        tools=[validate_tool, transfer_to_transformation],
        prompt=(
            "You are the **validation agent**. Your job:\n"
            "1. The ingestion agent has already called `ingest_ticker_data` and provided a JSON array of rows under `data`.\n"
            "2. If `data` is empty or not passed, reply with an error (no tool call).\n"
            "3. Otherwise, call `validate_data(data=data)`.\n"
            "4. If `is_valid` is False, report the null counts and stop (no handoff).\n"
            "   If `is_valid` is True, immediately call `transfer_to_transformation_agent()` (no arguments).\n"
        ),
        name="validation_agent",
    )

    transformation_agent = create_react_agent(
        model=ChatOpenAI(model="gpt-4o"),
        tools=[transform_tool],
        prompt=(
            "You are the **transformation agent**. Your job:\n"
            "1. The validation agent has already confirmed that `data` (JSON list of rows) is valid.\n"
            "2. Verify that `data` is a non-empty list of dicts containing 'Close' and at least Date/Open/High/Low/Volume fields.\n"
            "   If the structure is missing or invalid, reply with an error (no tool call).\n"
            "3. Otherwise, call `transform_data(data=data)`.\n"
            "4. When `transform_data` returns a Markdown table of all rows, output that table as your final response.\n"
            "   Do NOT call any further tools.\n"
        ),
        name="transformation_agent",
    )

    swarm = create_swarm(
        agents=[ingestion_agent, validation_agent, transformation_agent],
        default_active_agent="ingestion_agent",
    ).compile()

    user_query = {
        "messages": [
            {
                "role": "user",
                "content": "Get AAPL data from February 1 to February 2, 2024 at hourly intervals."
            }
        ]
    }

    print("⏳ Starting swarm...")
    async for chunk in swarm.astream(user_query):
        pprint(chunk)
        print("-" * 80)
    print("✅ Swarm complete.")

if __name__ == "__main__":
    asyncio.run(main())
