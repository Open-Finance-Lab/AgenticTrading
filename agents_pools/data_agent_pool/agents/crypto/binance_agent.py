
import random
from schema import load_agent_config

class BinanceAgent:
    def __init__(self):
        self.config = load_agent_config("binance_agent")

    def get_capabilities(self):
        return {
            "tools": ["fetch_spot_price", "fetch_ohlcv"],
            "resources": ["binance_api"],
            "schema": {
                "fetch_spot_price": {
                    "input": { "symbol": "string" },
                    "output": { "price": "float" }
                },
                "fetch_ohlcv": {
                    "input": { "symbol": "string", "interval": "string" },
                    "output": { "ohlcv": "list[list[float]]" }
                }
            }
        }

    def execute(self, function, inputs):
        if function == "fetch_spot_price":
            return self.fetch_spot_price(inputs["symbol"])
        elif function == "fetch_ohlcv":
            return self.fetch_ohlcv(inputs["symbol"], inputs.get("interval", self.config["api"]["default_interval"]))
        else:
            raise ValueError(f"Function '{function}' not supported by BinanceAgent")

    def fetch_spot_price(self, symbol):
        # Simulated price response (mock)
        return {
            "price": round(random.uniform(10000, 30000), 2),
            "symbol": symbol
        }

    def fetch_ohlcv(self, symbol, interval):
        # Simulated OHLCV data
        return {
            "symbol": symbol,
            "interval": interval,
            "ohlcv": [[random.uniform(10000, 30000) for _ in range(5)] for _ in range(10)]
        }
