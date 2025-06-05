# import logging
# from typing import Dict, Callable, Any

# # === Base Interface for All Agents ===
# class BaseAgent:
#     def __init__(self, config: dict):
#         self.config = config

#     def execute(self, function_name: str, inputs: dict) -> Any:
#         """Dispatch execution to agent methods dynamically."""
#         method: Callable = getattr(self, function_name, None)
#         if not method:
#             raise AttributeError(f"Function '{function_name}' not found in agent.")
#         return method(**inputs)

# # === Global Agent Registry ===
# AGENT_REGISTRY: Dict[str, BaseAgent] = {}

# def register_agent(agent_id: str, agent_instance: BaseAgent):
#     """Register agent instance to the global registry."""
#     if agent_id in AGENT_REGISTRY:
#         logging.warning(f"Agent '{agent_id}' already registered. Overwriting.")
#     AGENT_REGISTRY[agent_id] = agent_instance
#     logging.info(f"Agent '{agent_id}' registered with config: {agent_instance.config}")

# # === Default Agents Loading ===

# # Import all actual agent classes
# from agents.crypto.binance_agent import BinanceAgent
# from agents.crypto.coinbase_agent import CoinbaseAgent
# from agents.equity.alpaca_agent import AlpacaAgent
# from agents.equity.iex_agent import IEXAgent
# from agents.news.newsapi_agent import NewsAPIAgent
# from agents.news.rss_agent import RSSAgent

# def preload_default_agents():
#     """Preload a standard set of data agents into the registry."""
#     register_agent("binance_agent", BinanceAgent({"name": "binance"}))
#     register_agent("coinbase_agent", CoinbaseAgent({"name": "coinbase"}))
#     register_agent("alpaca_agent", AlpacaAgent({"name": "alpaca"}))
#     register_agent("iex_agent", IEXAgent({"name": "iex"}))
#     register_agent("newsapi_agent", NewsAPIAgent({"name": "newsapi"}))
#     register_agent("rss_agent", RSSAgent({"name": "rss"}))
import logging
import os
import yaml
from typing import Dict, Callable, Any

# === Base Interface for All Agents ===
class BaseAgent:
    def __init__(self, config: dict):
        self.config = config

    def execute(self, function_name: str, inputs: dict) -> Any:
        """Dispatch execution to agent methods dynamically."""
        method: Callable = getattr(self, function_name, None)
        if not method:
            raise AttributeError(f"Function '{function_name}' not found in agent.")
        return method(**inputs)

# === Global Agent Registry ===
AGENT_REGISTRY: Dict[str, BaseAgent] = {}

def register_agent(agent_id: str, agent_instance: BaseAgent):
    """Register agent instance to the global registry."""
    if agent_id in AGENT_REGISTRY:
        logging.warning(f"Agent '{agent_id}' already registered. Overwriting.")
    AGENT_REGISTRY[agent_id] = agent_instance
    logging.info(f"Agent '{agent_id}' registered with config: {agent_instance.config}")

# === Agent Class Imports ===
from agents.crypto.binance_agent import BinanceAgent
from agents.crypto.coinbase_agent import CoinbaseAgent
from agents.equity.alpaca_agent import AlpacaAgent
from agents.equity.iex_agent import IEXAgent
from agents.news.newsapi_agent import NewsAPIAgent
from agents.news.rss_agent import RSSAgent

# === Schema Imports ===
from schema.crypto_schema import BinanceConfig, CoinbaseConfig
from schema.equity_schema import AlpacaConfig, IEXConfig
from schema.news_schema import NewsAPIConfig, RSSConfig

# === Config Loader ===
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")

def load_config(agent_id: str) -> dict:
    config_file = os.path.join(CONFIG_DIR, f"{agent_id.replace('_agent', '')}.yaml")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file for agent '{agent_id}' not found: {config_file}")
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

# === Agent Loader ===
def preload_default_agents():
    """Load all supported agents from schema-config YAML."""

    # crypto
    binance_cfg = BinanceConfig(**load_config("binance_agent"))
    coinbase_cfg = CoinbaseConfig(**load_config("coinbase_agent"))
    register_agent("binance_agent", BinanceAgent(binance_cfg))
    register_agent("coinbase_agent", CoinbaseAgent(coinbase_cfg))

    # equity
    alpaca_cfg = AlpacaConfig(**load_config("alpaca_agent"))
    iex_cfg = IEXConfig(**load_config("iex_agent"))
    register_agent("alpaca_agent", AlpacaAgent(alpaca_cfg))
    register_agent("iex_agent", IEXAgent(iex_cfg))

    # news
    newsapi_cfg = NewsAPIConfig(**load_config("newsapi_agent"))
    rss_cfg = RSSConfig(**load_config("rss_agent"))
    register_agent("newsapi_agent", NewsAPIAgent(newsapi_cfg))
    register_agent("rss_agent", RSSAgent(rss_cfg))