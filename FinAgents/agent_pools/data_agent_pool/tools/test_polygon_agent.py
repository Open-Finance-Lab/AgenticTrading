import asyncio
from agent_pools.data_agent_pool.core import DataAgentPool
from dotenv import load_dotenv
load_dotenv()

async def main():
    # Initialize the agent pool
    pool = DataAgentPool()
    polygon_agent = pool.agents.get("polygon_agent")
    assert polygon_agent is not None, "PolygonAgent not found in agent pool"

    # Example natural language query
    query = (
        "Analyze the trading volume and price trend of AAPL and MSFT for the last 30 days. "
        "Highlight any unusual volume spikes and provide a summary."
    )

    # Call the LLM-driven intent interface
    result = await polygon_agent.process_intent(query)
    print("=== Execution Plan ===")
    print(result["execution_plan"])
    print("=== Result ===")
    print(result["result"])
    print("=== Metadata ===")
    print(result["metadata"])

if __name__ == "__main__":
    asyncio.run(main())