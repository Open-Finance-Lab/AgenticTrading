from typing import Dict, List, Optional, Union, Any
import pandas as pd
from datetime import datetime
from agent_pools.data_agent_pool.registry import BaseAgent
from agent_pools.data_agent_pool.schema.crypto_schema import BinanceConfig

class BinanceAgent(BaseAgent):
    """
    Binance cryptocurrency exchange data agent implementation.
    """

    def __init__(self, config: BinanceConfig):
        """
        Initialize Binance agent with configuration.
        """
        super().__init__(config.model_dump())
        # Adapt to nested config
        self.api_base_url = self.config.get("api", {}).get("base_url", "https://api.binance.com")
        self.endpoints = self.config.get("api", {}).get("endpoints", {})
        self.default_interval = self.config.get("api", {}).get("default_interval", "1h")
        self.api_key = self.config.get("authentication", {}).get("api_key")
        self.api_secret = self.config.get("authentication", {}).get("secret_key")
        self.timeout = self.config.get("constraints", {}).get("timeout", 5)
        self.rate_limit = self.config.get("constraints", {}).get("rate_limit_per_minute", 120)
        self._validate_config()

    def fetch(self, 
             symbol: str,
             start: str,
             end: str,
             interval: str = None) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Binance.

        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            start (str): Start time in ISO format
            end (str): End time in ISO format
            interval (str, optional): Kline interval. Defaults to "1h"

        Returns:
            pd.DataFrame: OHLCV data with columns [timestamp, open, high, low, close, volume]

        Raises:
            ValueError: If input parameters are invalid
            RuntimeError: If API request fails
        """
        try:
            interval = interval or self.default_interval
            # Convert ISO timestamps to milliseconds
            start_ts = int(datetime.fromisoformat(start).timestamp() * 1000)
            end_ts = int(datetime.fromisoformat(end).timestamp() * 1000)

            # For demo, return sample data
            # In production, implement actual API call
            data = {
                "timestamp": pd.date_range(start=start, end=end, freq=interval),
                "open": [68000.0] * 24,
                "high": [68500.0] * 24,
                "low": [67500.0] * 24,
                "close": [68200.0] * 24,
                "volume": [1245.6] * 24
            }
            return pd.DataFrame(data)

        except ValueError as e:
            raise ValueError(f"Invalid input parameters: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to fetch data from Binance: {str(e)}")

    def get_current_price(self, symbol: str) -> Dict[str, float]:
        """
        Get real-time price for a trading pair.

        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')

        Returns:
            Dict[str, float]: Current price data
            
        Raises:
            RuntimeError: If price fetch fails
        """
        try:
            # Mock implementation for demo
            return {
                "symbol": symbol,
                "price": 68000.5,
                "timestamp": datetime.now().timestamp()
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get current price: {str(e)}")

    def _validate_config(self) -> None:
        """
        Validate agent configuration.

        Raises:
            ValueError: If required config parameters are missing
        """
        if not self.api_key or not self.api_secret:
            raise ValueError("Missing required config field: api_key or secret_key")