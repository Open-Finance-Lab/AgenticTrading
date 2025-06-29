from FinAgents.agent_pools.data_agent_pool.registry import BaseAgent
from FinAgents.agent_pools.data_agent_pool.schema.equity_schema import IEXConfig

class IEXAgent(BaseAgent):
    def __init__(self, config: IEXConfig):
        super().__init__(config.model_dump())

    def get_market_summary(self) -> dict:
        return {
            "summary": "Markets are up 1.5% today."
        }
    
    def get_quote(self, ticker: str) -> dict:
        return {
            "ticker": ticker,
            "quote": {
                "open": 720.0,
                "close": 730.0,
                "volume": 50000
            }
        }