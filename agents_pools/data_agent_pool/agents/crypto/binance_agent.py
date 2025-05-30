from registry import BaseAgent
from schema.crypto_schema import BinanceConfig

class BinanceAgent(BaseAgent):
    def __init__(self, config: BinanceConfig):
        super().__init__(config.model_dump())

    def fetch_ohlcv(self, symbol: str, interval: str) -> dict:
        return {
            "symbol": symbol,
            "interval": interval,
            "data": [[1717000000, 68000, 68500, 67500, 68200, 1245.6]]
        }