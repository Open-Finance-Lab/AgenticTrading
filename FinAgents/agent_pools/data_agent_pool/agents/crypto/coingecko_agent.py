from typing import Dict, List, Optional, Union, Any
import pandas as pd
from datetime import datetime
import requests
import time
from FinAgents.agent_pools.data_agent_pool.registry import BaseAgent
from FinAgents.agent_pools.data_agent_pool.schema.crypto_schema import CoinGeckoConfig


class CoinGeckoAgent(BaseAgent):
    """
    CoinGecko cryptocurrency market data agent implementation.
    """
    
    def __init__(self, config: CoinGeckoConfig):
        """
        Initialize CoinGecko agent with configuration.
        """
        super().__init__(config.model_dump())
        # Adapt to nested config
        self.api_base_url = self.config.get("api", {}).get("base_url", "https://api.coingecko.com/api/v3")
        self.endpoints = self.config.get("api", {}).get("endpoints", {})
        self.default_vs_currency = self.config.get("api", {}).get("default_vs_currency", "usd")
        self.api_key = self.config.get("authentication", {}).get("api_key")
        self.timeout = self.config.get("constraints", {}).get("timeout", 10)
        self.rate_limit = self.config.get("constraints", {}).get("rate_limit_per_minute", 50)
        self.pro_api = self.config.get("authentication", {}).get("pro_api", False)
        
        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 60 / self.rate_limit if self.rate_limit > 0 else 0
        
        self._validate_config()
    
    def _rate_limit_check(self):
        """Implement rate limiting between requests."""
        if self._min_request_interval > 0:
            elapsed = time.time() - self._last_request_time
            if elapsed < self._min_request_interval:
                time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make API request with rate limiting and error handling."""
        self._rate_limit_check()
        
        url = f"{self.api_base_url}/{endpoint}"
        headers = {}
        
        if self.api_key and self.pro_api:
            headers["x-cg-pro-api-key"] = self.api_key
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {str(e)}")
    
    def fetch(self,
              coin_id: str,
              start: str,
              end: str,
              vs_currency: str = None) -> pd.DataFrame:
        """
        Fetch historical price data from CoinGecko.
        
        Args:
            coin_id (str): CoinGecko coin ID (e.g., 'bitcoin', 'ethereum')
            start (str): Start time in ISO format
            end (str): End time in ISO format
            vs_currency (str, optional): Target currency. Defaults to "usd"
            
        Returns:
            pd.DataFrame: Historical price data with columns [timestamp, price, market_cap, total_volume]
            
        Raises:
            ValueError: If input parameters are invalid
            RuntimeError: If API request fails
        """
        try:
            vs_currency = vs_currency or self.default_vs_currency
            
            # Convert ISO timestamps to Unix timestamps
            start_ts = int(datetime.fromisoformat(start).timestamp())
            end_ts = int(datetime.fromisoformat(end).timestamp())
            
            # CoinGecko historical data endpoint
            endpoint = f"coins/{coin_id}/market_chart/range"
            params = {
                "vs_currency": vs_currency,
                "from": start_ts,
                "to": end_ts
            }
            
            data = self._make_request(endpoint, params)
            
            # Extract price, market cap, and volume data
            prices = data.get("prices", [])
            market_caps = data.get("market_caps", [])
            volumes = data.get("total_volumes", [])
            
            if not prices:
                raise ValueError(f"No price data found for {coin_id}")
            
            # Convert to DataFrame
            df_data = {
                "timestamp": [datetime.fromtimestamp(p[0] / 1000) for p in prices],
                "price": [p[1] for p in prices],
                "market_cap": [mc[1] if mc else None for mc in market_caps],
                "total_volume": [v[1] if v else None for v in volumes]
            }
            
            return pd.DataFrame(df_data)
            
        except ValueError as e:
            raise ValueError(f"Invalid input parameters: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to fetch data from CoinGecko: {str(e)}")
    
    def get_current_price(self, coin_ids: Union[str, List[str]], vs_currency: str = None) -> Dict[str, Any]:
        """
        Get real-time price for one or multiple cryptocurrencies.
        
        Args:
            coin_ids (Union[str, List[str]]): Single coin ID or list of coin IDs
            vs_currency (str, optional): Target currency. Defaults to "usd"
            
        Returns:
            Dict[str, Any]: Current price data with additional market information
            
        Raises:
            RuntimeError: If price fetch fails
        """
        try:
            vs_currency = vs_currency or self.default_vs_currency
            
            if isinstance(coin_ids, str):
                coin_ids = [coin_ids]
            
            endpoint = "simple/price"
            params = {
                "ids": ",".join(coin_ids),
                "vs_currencies": vs_currency,
                "include_market_cap": "true",
                "include_24hr_vol": "true",
                "include_24hr_change": "true",
                "include_last_updated_at": "true"
            }
            
            data = self._make_request(endpoint, params)
            
            # Add timestamp to response
            result = {
                "data": data,
                "timestamp": datetime.now().timestamp(),
                "vs_currency": vs_currency
            }
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to get current price: {str(e)}")
    
    def get_market_data(self, coin_id: str, vs_currency: str = None) -> Dict[str, Any]:
        """
        Get comprehensive market data for a cryptocurrency.
        
        Args:
            coin_id (str): CoinGecko coin ID
            vs_currency (str, optional): Target currency. Defaults to "usd"
            
        Returns:
            Dict[str, Any]: Comprehensive market data
            
        Raises:
            RuntimeError: If market data fetch fails
        """
        try:
            vs_currency = vs_currency or self.default_vs_currency
            
            endpoint = f"coins/{coin_id}"
            params = {
                "localization": "false",
                "tickers": "false",
                "market_data": "true",
                "community_data": "false",
                "developer_data": "false",
                "sparkline": "false"
            }
            
            data = self._make_request(endpoint, params)
            
            # Extract relevant market data
            market_data = data.get("market_data", {})
            
            result = {
                "id": data.get("id"),
                "name": data.get("name"),
                "symbol": data.get("symbol"),
                "current_price": market_data.get("current_price", {}).get(vs_currency),
                "market_cap": market_data.get("market_cap", {}).get(vs_currency),
                "total_volume": market_data.get("total_volume", {}).get(vs_currency),
                "price_change_24h": market_data.get("price_change_24h"),
                "price_change_percentage_24h": market_data.get("price_change_percentage_24h"),
                "market_cap_rank": market_data.get("market_cap_rank"),
                "total_supply": market_data.get("total_supply"),
                "max_supply": market_data.get("max_supply"),
                "circulating_supply": market_data.get("circulating_supply"),
                "last_updated": market_data.get("last_updated"),
                "timestamp": datetime.now().timestamp()
            }
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to get market data: {str(e)}")
    
    def get_trending_coins(self) -> Dict[str, Any]:
        """
        Get trending cryptocurrencies.
        
        Returns:
            Dict[str, Any]: Trending coins data
            
        Raises:
            RuntimeError: If trending data fetch fails
        """
        try:
            endpoint = "search/trending"
            data = self._make_request(endpoint)
            
            result = {
                "trending_coins": data.get("coins", []),
                "trending_nfts": data.get("nfts", []),
                "trending_categories": data.get("categories", []),
                "timestamp": datetime.now().timestamp()
            }
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to get trending coins: {str(e)}")
    
    def search_coins(self, query: str) -> Dict[str, Any]:
        """
        Search for cryptocurrencies by name or symbol.
        
        Args:
            query (str): Search query
            
        Returns:
            Dict[str, Any]: Search results
            
        Raises:
            RuntimeError: If search fails
        """
        try:
            endpoint = "search"
            params = {"query": query}
            
            data = self._make_request(endpoint, params)
            
            result = {
                "coins": data.get("coins", []),
                "exchanges": data.get("exchanges", []),
                "icos": data.get("icos", []),
                "categories": data.get("categories", []),
                "nfts": data.get("nfts", []),
                "query": query,
                "timestamp": datetime.now().timestamp()
            }
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to search coins: {str(e)}")
    
    def _validate_config(self) -> None:
        """
        Validate agent configuration.
        
        Raises:
            ValueError: If required config parameters are missing or invalid
        """
        if not self.api_base_url:
            raise ValueError("Missing required config field: api_base_url")
        
        if self.rate_limit <= 0:
            raise ValueError("Rate limit must be a positive number")
        
        if self.timeout <= 0:
            raise ValueError("Timeout must be a positive number")
        
        # Validate API key if pro API is enabled
        if self.pro_api and not self.api_key:
            raise ValueError("API key is required when pro_api is enabled")
    
    def get_supported_coins(self) -> List[Dict[str, Any]]:
        """
        Get list of all supported coins.
        
        Returns:
            List[Dict[str, Any]]: List of supported coins with id, symbol, and name
            
        Raises:
            RuntimeError: If coins list fetch fails
        """
        try:
            endpoint = "coins/list"
            data = self._make_request(endpoint)
            return data
            
        except Exception as e:
            raise RuntimeError(f"Failed to get supported coins: {str(e)}")