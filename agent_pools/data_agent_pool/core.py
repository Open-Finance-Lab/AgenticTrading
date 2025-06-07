# core.py

from agent_pools.data_agent_pool.registry import AGENT_REGISTRY, preload_default_agents
from agent_pools.data_agent_pool.memory_bridge import record_event
from agent_pools.data_agent_pool.agents.crypto.binance_agent import BinanceAgent
from datetime import datetime
from typing import Optional
import pandas as pd


class DataAgentPool:
    """
    This class serves as the centralized data access orchestrator within the FinAgent system. 
    It autonomously selects and invokes suitable data agents based on configuration and execution outcomes.
    """

    def __init__(self):
        """
        Initialize the pool by retrieving all active agents sorted by priority.

        Agents are defined in the registry and their status and behavior are
        controlled through associated configuration files.
        """
        preload_default_agents()
        self.agents = AGENT_REGISTRY  # Now a dict

    def fetch(self, 
              symbol: str, 
              start: str, 
              end: str, 
              interval: str = "1h",
              market: Optional[str] = None) -> pd.DataFrame:
        """
        Iteratively attempts to fetch market data using the available agents.
        If an agent fails, the next one in the priority list is tried.

        Parameters:
        - symbol (str): Asset ticker symbol.
        - start (str): Start datetime in ISO format.
        - end (str): End datetime in ISO format.
        - interval (str): Desired data resolution (e.g., '1m', '1h').
        - market (Optional[str]): Optional market segment (e.g., 'crypto', 'equity').

        Returns:
        - pd.DataFrame: Standardized dataframe containing OHLCV time series.

        Raises:
        - RuntimeError: If no agent successfully returns valid data.
        """
        for agent in self.agents.values():
            try:
                df = agent.fetch(symbol, start, end, interval)
                record_event(
                    agent_name=agent.__class__.__name__,
                    task="fetch",
                    input={"symbol": symbol, "start": start, "end": end, "interval": interval},
                    summary=f"Fetched {len(df)} rows from {agent.__class__.__name__}"
                )
                return df
            except Exception as e:
                record_event(
                    agent_name=agent.__class__.__name__,
                    task="fetch-failure",
                    input={"symbol": symbol, "start": start, "end": end, "interval": interval},
                    summary=f"Error: {str(e)}"
                )
                continue

        raise RuntimeError("All data sources failed to return a valid result.")

