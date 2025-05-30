# registry.py

import logging
from typing import Dict, Callable, Any

# Example interface for agents
class BaseAgent:
    def __init__(self, config: dict):
        self.config = config

    def execute(self, function_name: str, inputs: dict) -> Any:
        """Dispatch execution to agent methods dynamically."""
        method: Callable = getattr(self, function_name, None)
        if not method:
            raise AttributeError(f"Function '{function_name}' not found in agent.")
        return method(**inputs)

# Example agent: Binance data agent
class BinanceAgent(BaseAgent):
    def fetch_ohlcv(self, symbol: str, interval: str) -> dict:
        """Simulated OHLCV fetcher for crypto market."""
        # In real use: fetch from Binance API
        return {
            "symbol": symbol,
            "interval": interval,
            "data": [[1717000000, 68000, 68500, 67500, 68200, 1245.6]]
        }

# Global registry
AGENT_REGISTRY: Dict[str, BaseAgent] = {}
    
def register_agent(agent_id: str, agent_instance: BaseAgent):
    """Register agent instance to the global registry."""
    if agent_id in AGENT_REGISTRY:
        logging.warning(f"Agent '{agent_id}' already registered. Overwriting.")
    AGENT_REGISTRY[agent_id] = agent_instance
    logging.info(f"Agent '{agent_id}' registered with config: {agent_instance.config}")

# Optional: static load for demo or testing
def preload_default_agents():
    # Example config placeholder
    binance_cfg = {
        "name": "binance",
        "base_url": "https://api.binance.com"
    }
    register_agent("binance_agent", BinanceAgent(config=binance_cfg))