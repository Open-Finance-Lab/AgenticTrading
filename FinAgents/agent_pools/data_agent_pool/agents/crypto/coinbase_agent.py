from FinAgents.agent_pools.data_agent_pool.registry import BaseAgent
from FinAgents.agent_pools.data_agent_pool.schema.crypto_schema import CoinbaseConfig

class CoinbaseAgent(BaseAgent):
    def __init__(self, config: CoinbaseConfig):
        super().__init__(config.model_dump())

    def get_spot_price(self, symbol: str) -> dict:
        return {
            "symbol": symbol,
            "price": 68100.5
        }
        
    def fetch_price(self, symbol: str) -> dict:
        return {
            "symbol": symbol,
            "price": 3000.0  # 模拟值，可替换为API调用
        }