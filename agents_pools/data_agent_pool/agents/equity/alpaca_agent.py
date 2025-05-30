from registry import BaseAgent
from schema.equity_schema import AlpacaConfig

class AlpacaAgent(BaseAgent):
    def __init__(self, config: AlpacaConfig):
        super().__init__(config.model_dump())

    def get_equity_quote(self, ticker: str) -> dict:
        return {
            "ticker": ticker,
            "price": 154.7
        }
        
    def fetch_equity_data(self, ticker: str) -> dict:
        return {
            "ticker": ticker,
            "price": 185.0,
            "volume": 100000
        }