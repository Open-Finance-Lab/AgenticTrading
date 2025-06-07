# data_agent_pool/tools/fetch_tool.py
from langchain.tools import Tool
from agent_pools.data_agent_pool.core import DataAgentPool
from typing import Dict


def fetch_ticker_data_tool(pool: DataAgentPool) -> Tool:
    """
    Constructs a LangGraph-compatible Tool for fetching ticker data
    from the autonomous DataAgentPool system.

    Parameters:
    - pool (DataAgentPool): The centralized data orchestration object.

    Returns:
    - Tool: LangGraph Tool instance encapsulating data fetch functionality.
    """

    def _fetch_wrapper(input_dict: Dict) -> str:
        """
        Wrapper to extract inputs and execute the pool.fetch() method.

        Expected keys in input_dict: symbol, start, end, interval

        Returns:
        - str: Status summary of the fetch process.
        """
        df = pool.fetch(
            symbol=input_dict["symbol"],
            start=input_dict["start"],
            end=input_dict["end"],
            interval=input_dict.get("interval", "1h")
        )
        return f"Fetched {len(df)} rows for {input_dict['symbol']} from {input_dict['start']} to {input_dict['end']}"

    return Tool(
        name="fetch_ticker_data",
        func=_fetch_wrapper,
        description="Fetches OHLCV market data for a given symbol over a date range with a given interval."
    )