from typing import Dict, List, Optional, Union, Any, Callable
import pandas as pd
import requests
import os
from datetime import datetime, timedelta
from agent_pools.data_agent_pool.registry import BaseAgent
from agent_pools.data_agent_pool.schema.crypto_schema import CoinbaseConfig
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.agents import Tool
from dotenv import load_dotenv
import json
import logging
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import hmac
import hashlib
import base64

load_dotenv()


class CoinbaseAgent(BaseAgent):
    """
    Coinbase Pro/Advanced Trade API data agent implementation.
    
    Features:
    - Historical OHLCV data for crypto pairs
    - Real-time market data
    - Order book data
    - Trading statistics
    - Account information (with API keys)
    - Market ticker information
    - Currency information
    - Trading fees
    - Time and sales data
    """

    GRANULARITY_MAP = {
        '1m': 60, '5m': 300, '15m': 900, '1h': 3600,
        '6h': 21600, '1d': 86400
    }

    def __init__(self, config: CoinbaseConfig):
        """Initialize Coinbase data agent."""
        super().__init__(config.model_dump())
        self.config = config
        self.cache_dir = 'data/cache/coinbase'
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # API Configuration
        self.base_url = "https://api.exchange.coinbase.com"
        self.api_key = config.api_key if hasattr(config, 'api_key') else None
        self.api_secret = config.api_secret if hasattr(config, 'api_secret') else None
        self.passphrase = config.passphrase if hasattr(config, 'passphrase') else None
        
        self._validate_config()
        self._init_tools()
        self._init_analysis_chain()
        
        # Initialize thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        if not hasattr(self.config, "llm_enabled"):
            raise ValueError("Missing required config parameter: 'llm_enabled'. Please add 'llm_enabled: true/false' to your coinbase.yaml.")
        
        self.llm_enabled = bool(self.config.llm_enabled)
        print(f"Coinbase Agent - llm_enabled config value: {self.llm_enabled}")
        
        if self.llm_enabled:
            self._init_llm_interface()

    def _init_llm_interface(self):
        """Configure LLM interface for crypto market analysis."""
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1 
        )
        
        self.system_prompt = SystemMessage(content="""
You are a professional Coinbase crypto data agent planner.

Your task is to generate an execution plan as a valid JSON object with a "steps" field (a list of tasks). Each step should specify:
- "tool": the tool to use
- "parameters": the parameters for the tool
- "type": the type of data being requested

Available tools:
- "fetch_historical_data": Get historical OHLCV data for crypto pairs
- "get_real_time_quote": Get current market data for crypto pairs
- "get_order_book": Get order book data
- "get_market_stats": Get 24hr trading statistics
- "get_ticker_info": Get ticker information for all products
- "get_currencies": Get information about available currencies
- "get_products": Get information about available trading pairs
- "get_candles": Get candlestick data with custom parameters
- "get_trades": Get recent trades for a product
- "get_account_info": Get account information (requires API keys)
- "get_fees": Get current trading fees

**Only output a valid JSON object, and nothing else.**

Example:
{
  "steps": [
    {
      "tool": "fetch_historical_data",
      "parameters": {
        "product_id": "BTC-USD",
        "granularity": "1h",
        "start": "2024-01-01T00:00:00Z",
        "end": "2024-01-31T23:59:59Z"
      },
      "type": "historical_data"
    },
    {
      "tool": "get_real_time_quote",
      "parameters": {
        "product_id": "BTC-USD"
      },
      "type": "real_time_data"
    }
  ]
}
""")

    def _init_tools(self):
        """Register available Coinbase operations."""
        self.tools = [
            Tool(
                name="fetch_historical_data",
                func=self.fetch_historical_data,
                description="Retrieve historical market data (OHLCV) for crypto pairs"
            ),
            Tool(
                name="get_real_time_quote",
                func=self.get_real_time_quote,
                description="Get current market data for crypto pairs"
            ),
            Tool(
                name="get_order_book",
                func=self.get_order_book,
                description="Get order book data for a product"
            ),
            Tool(
                name="get_market_stats",
                func=self.get_market_stats,
                description="Get 24hr trading statistics for a product"
            ),
            Tool(
                name="get_ticker_info",
                func=self.get_ticker_info,
                description="Get ticker information for all or specific products"
            ),
            Tool(
                name="get_currencies",
                func=self.get_currencies,
                description="Get information about available currencies"
            ),
            Tool(
                name="get_products",
                func=self.get_products,
                description="Get information about available trading pairs"
            ),
            Tool(
                name="get_candles",
                func=self.get_candles,
                description="Get candlestick data with custom parameters"
            ),
            Tool(
                name="get_trades",
                func=self.get_trades,
                description="Get recent trades for a product"
            ),
            Tool(
                name="get_account_info",
                func=self.get_account_info,
                description="Get account information (requires API keys)"
            ),
            Tool(
                name="get_fees",
                func=self.get_fees,
                description="Get current trading fees"
            )
        ]

    def _make_request(self, method: str, endpoint: str, params: dict = None, auth_required: bool = False) -> dict:
        """Make authenticated or public API request to Coinbase."""
        url = f"{self.base_url}{endpoint}"
        headers = {'Content-Type': 'application/json'}
        
        if auth_required and self.api_key:
            timestamp = str(time.time())
            message = timestamp + method + endpoint + (json.dumps(params) if params and method != 'GET' else '')
            signature = base64.b64encode(
                hmac.new(
                    base64.b64decode(self.api_secret),
                    message.encode('utf-8'),
                    hashlib.sha256
                ).digest()
            ).decode('utf-8')
            
            headers.update({
                'CB-ACCESS-KEY': self.api_key,
                'CB-ACCESS-SIGN': signature,
                'CB-ACCESS-TIMESTAMP': timestamp,
                'CB-ACCESS-PASSPHRASE': self.passphrase
            })
        
        try:
            if method == 'GET':
                response = requests.get(url, params=params, headers=headers)
            elif method == 'POST':
                response = requests.post(url, json=params, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {str(e)}")

    def fetch_historical_data(self, 
                            product_id: str,
                            granularity: str = "1h",
                            start: str = None,
                            end: str = None,
                            force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch historical market data from Coinbase.
        
        Args:
            product_id: Trading pair (e.g., 'BTC-USD', 'ETH-BTC')
            granularity: Data granularity (1m, 5m, 15m, 1h, 6h, 1d)
            start: Start time in ISO 8601 format
            end: End time in ISO 8601 format
            force_refresh: Force refresh from API
        """
        try:
            # Validate granularity
            if granularity not in self.GRANULARITY_MAP:
                raise ValueError(f"Invalid granularity: {granularity}. Valid options: {list(self.GRANULARITY_MAP.keys())}")

            # Check cache
            cache_key = f"{product_id}_{granularity}_{start}_{end}"
            cache_file = os.path.join(self.cache_dir, f'{cache_key}.csv')
            
            if not force_refresh and os.path.exists(cache_file):
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                # Check if cache is recent
                cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
                max_age = timedelta(hours=1) if granularity in ['1m', '5m', '15m'] else timedelta(days=1)
                
                if cache_age < max_age:
                    return df

            # Prepare parameters
            params = {
                'granularity': self.GRANULARITY_MAP[granularity]
            }
            if start:
                params['start'] = start
            if end:
                params['end'] = end

            # Fetch from Coinbase API
            data = self._make_request('GET', f'/products/{product_id}/candles', params)
            
            if not data:
                raise ValueError(f"No data returned for {product_id}")

            # Convert to DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Convert to numeric
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
            
            # Add technical indicators
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std() * (365 ** 0.5)  # Annualized volatility
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['rsi'] = self._calculate_rsi(df['close'])
            
            # Cache results
            df.to_csv(cache_file)
            return df
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch historical data for {product_id}: {str(e)}")

    def get_real_time_quote(self, product_id: str) -> Dict[str, Any]:
        """Get current market data for a crypto pair."""
        try:
            # Get ticker data
            ticker_data = self._make_request('GET', f'/products/{product_id}/ticker')
            
            # Get 24hr stats
            stats_data = self._make_request('GET', f'/products/{product_id}/stats')
            
            return {
                "product_id": product_id,
                "price": float(ticker_data.get("price", 0)),
                "bid": float(ticker_data.get("bid", 0)),
                "ask": float(ticker_data.get("ask", 0)),
                "volume": float(ticker_data.get("volume", 0)),
                "time": ticker_data.get("time"),
                "open_24h": float(stats_data.get("open", 0)),
                "high_24h": float(stats_data.get("high", 0)),
                "low_24h": float(stats_data.get("low", 0)),
                "volume_24h": float(stats_data.get("volume", 0)),
                "volume_30d": float(stats_data.get("volume_30day", 0)),
                "change_24h": float(stats_data.get("open", 0)) - float(ticker_data.get("price", 0)) if stats_data.get("open") else 0,
                "change_percent_24h": ((float(ticker_data.get("price", 0)) - float(stats_data.get("open", 0))) / float(stats_data.get("open", 1))) * 100 if stats_data.get("open") else 0
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to get real-time quote for {product_id}: {str(e)}")

    def get_order_book(self, product_id: str, level: int = 2) -> Dict[str, Any]:
        """Get order book data for a product."""
        try:
            params = {'level': level}
            data = self._make_request('GET', f'/products/{product_id}/book', params)
            
            return {
                "product_id": product_id,
                "sequence": data.get("sequence"),
                "bids": [[float(price), float(size)] for price, size in data.get("bids", [])],
                "asks": [[float(price), float(size)] for price, size in data.get("asks", [])],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to get order book for {product_id}: {str(e)}")

    def get_market_stats(self, product_id: str) -> Dict[str, Any]:
        """Get 24hr trading statistics for a product."""
        try:
            data = self._make_request('GET', f'/products/{product_id}/stats')
            
            return {
                "product_id": product_id,
                "open": float(data.get("open", 0)),
                "high": float(data.get("high", 0)),
                "low": float(data.get("low", 0)),
                "volume": float(data.get("volume", 0)),
                "last": float(data.get("last", 0)),
                "volume_30day": float(data.get("volume_30day", 0))
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to get market stats for {product_id}: {str(e)}")

    def get_ticker_info(self, product_id: str = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get ticker information for all or specific products."""
        try:
            if product_id:
                data = self._make_request('GET', f'/products/{product_id}/ticker')
                return {
                    "product_id": product_id,
                    "price": float(data.get("price", 0)),
                    "size": float(data.get("size", 0)),
                    "bid": float(data.get("bid", 0)),
                    "ask": float(data.get("ask", 0)),
                    "volume": float(data.get("volume", 0)),
                    "time": data.get("time")
                }
            else:
                # Get all tickers
                products = self._make_request('GET', '/products')
                tickers = []
                for product in products[:20]:  # Limit to avoid rate limits
                    try:
                        ticker = self._make_request('GET', f'/products/{product["id"]}/ticker')
                        tickers.append({
                            "product_id": product["id"],
                            "price": float(ticker.get("price", 0)),
                            "volume": float(ticker.get("volume", 0)),
                            "bid": float(ticker.get("bid", 0)),
                            "ask": float(ticker.get("ask", 0))
                        })
                    except Exception:
                        continue
                return tickers
                
        except Exception as e:
            raise RuntimeError(f"Failed to get ticker info: {str(e)}")

    def get_currencies(self) -> List[Dict[str, Any]]:
        """Get information about available currencies."""
        try:
            data = self._make_request('GET', '/currencies')
            
            return [
                {
                    "id": currency.get("id"),
                    "name": currency.get("name"),
                    "min_size": float(currency.get("min_size", 0)),
                    "status": currency.get("status"),
                    "max_precision": currency.get("max_precision"),
                    "convertible_to": currency.get("convertible_to", []),
                    "details": currency.get("details", {})
                }
                for currency in data
            ]
            
        except Exception as e:
            raise RuntimeError(f"Failed to get currencies: {str(e)}")

    def get_products(self) -> List[Dict[str, Any]]:
        """Get information about available trading pairs."""
        try:
            data = self._make_request('GET', '/products')
            
            return [
                {
                    "id": product.get("id"),
                    "base_currency": product.get("base_currency"),
                    "quote_currency": product.get("quote_currency"),
                    "base_min_size": float(product.get("base_min_size", 0)),
                    "base_max_size": float(product.get("base_max_size", 0)),
                    "quote_increment": float(product.get("quote_increment", 0)),
                    "base_increment": float(product.get("base_increment", 0)),
                    "display_name": product.get("display_name"),
                    "min_market_funds": float(product.get("min_market_funds", 0)),
                    "max_market_funds": float(product.get("max_market_funds", 0)),
                    "margin_enabled": product.get("margin_enabled", False),
                    "post_only": product.get("post_only", False),
                    "limit_only": product.get("limit_only", False),
                    "cancel_only": product.get("cancel_only", False),
                    "trading_disabled": product.get("trading_disabled", False),
                    "status": product.get("status"),
                    "status_message": product.get("status_message")
                }
                for product in data
            ]
            
        except Exception as e:
            raise RuntimeError(f"Failed to get products: {str(e)}")

    def get_candles(self, 
                   product_id: str, 
                   start: str = None, 
                   end: str = None, 
                   granularity: str = "1h") -> pd.DataFrame:
        """Get candlestick data with custom parameters."""
        return self.fetch_historical_data(product_id, granularity, start, end)

    def get_trades(self, product_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades for a product."""
        try:
            params = {}
            if limit:
                params['limit'] = min(limit, 1000)  # API limit
                
            data = self._make_request('GET', f'/products/{product_id}/trades', params)
            
            return [
                {
                    "time": trade.get("time"),
                    "trade_id": trade.get("trade_id"),
                    "price": float(trade.get("price", 0)),
                    "size": float(trade.get("size", 0)),
                    "side": trade.get("side")
                }
                for trade in data
            ]
            
        except Exception as e:
            raise RuntimeError(f"Failed to get trades for {product_id}: {str(e)}")

    def get_account_info(self) -> Dict[str, Any]:
        """Get account information (requires API keys)."""
        try:
            if not self.api_key:
                return {"error": "API keys required for account information"}
                
            accounts = self._make_request('GET', '/accounts', auth_required=True)
            
            return {
                "accounts": [
                    {
                        "id": account.get("id"),
                        "currency": account.get("currency"),
                        "balance": float(account.get("balance", 0)),
                        "available": float(account.get("available", 0)),
                        "hold": float(account.get("hold", 0)),
                        "profile_id": account.get("profile_id"),
                        "trading_enabled": account.get("trading_enabled")
                    }
                    for account in accounts
                ]
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to get account info: {str(e)}")

    def get_fees(self) -> Dict[str, Any]:
        """Get current trading fees."""
        try:
            if not self.api_key:
                return {"error": "API keys required for fee information"}
                
            data = self._make_request('GET', '/fees', auth_required=True)
            
            return {
                "maker_fee_rate": float(data.get("maker_fee_rate", 0)),
                "taker_fee_rate": float(data.get("taker_fee_rate", 0)),
                "usd_volume": float(data.get("usd_volume", 0))
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to get fees: {str(e)}")

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if not hasattr(self.config, 'cache_enabled'):
            self.config.cache_enabled = True

    def _init_analysis_chain(self):
        """Initialize analysis chains for advanced crypto market analysis."""
        pass

    def _parse_intent(self, llm_output: str) -> dict:
        """Parse LLM output into a validated execution plan."""
        try:
            plan = json.loads(llm_output)
        except Exception:
            try:
                json_str = re.search(r'\{.*\}', llm_output, re.DOTALL).group()
                plan = json.loads(json_str)
            except Exception:
                logging.warning("LLM output is not valid JSON. Using default plan.")
                plan = {
                    "steps": [{
                        "tool": "get_real_time_quote",
                        "parameters": {"product_id": "BTC-USD"},
                        "type": "real_time_data"
                    }]
                }

        # Validation
        if "steps" in plan:
            if not isinstance(plan["steps"], list) or not plan["steps"]:
                raise ValueError("Execution plan 'steps' must be a non-empty list.")
            for step in plan["steps"]:
                for field in ["tool", "parameters"]:
                    if field not in step:
                        raise ValueError(f"Step missing required field: {field}")
        else:
            for field in ["tool", "parameters"]:
                if field not in plan:
                    raise ValueError(f"Execution plan missing required field: {field}")

        return plan

    async def process_intent(self, query: str) -> Dict[str, Any]:
        """Process natural language crypto market data requests."""
        if not getattr(self, "llm_enabled", True):
            # Default plan for testing
            plan = {
                "steps": [{
                    "tool": "get_real_time_quote",
                    "parameters": {"product_id": "BTC-USD"},
                    "type": "real_time_data"
                }]
            }
            result = await self._execute_strategy(plan)
            return {
                "execution_plan": plan,
                "result": result,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "query_type": plan["steps"][0].get("type") if "steps" in plan else plan.get("type"),
                    "llm_used": False
                }
            }

        # LLM-driven path
        intent_analysis = await self.llm.agenerate([
            [self.system_prompt, HumanMessage(content=query)]
        ])
        plan = self._parse_intent(intent_analysis.generations[0][0].text)
        print("=== Coinbase Execution Plan ===")
        print(json.dumps(plan, indent=2))
        result = await self._execute_strategy(plan)
        
        return {
            "execution_plan": plan,
            "result": result,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "query_type": plan["steps"][0].get("type") if "steps" in plan else plan.get("type"),
                "llm_used": True
            }
        }

    async def _execute_strategy(self, plan: Dict) -> Any:
        """Execute generated crypto market data strategy."""
        import inspect
        
        try:
            if "steps" in plan:
                # Multi-step plan
                results = []
                for step in plan["steps"]:
                    tool_name = step.get("tool")
                    tool = next((t for t in self.tools if t.name == tool_name), None)
                    if not tool:
                        available = [t.name for t in self.tools]
                        raise ValueError(f"Tool not found: {tool_name}. Available tools: {available}")
                    
                    func = tool.func
                    params = step.get("parameters", {})
                    
                    # Execute in thread pool if synchronous
                    if inspect.iscoroutinefunction(func):
                        result = await func(**params)
                    else:
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(self.executor, lambda: func(**params))
                    
                    results.append({
                        "step": tool_name,
                        "result": result
                    })
                return results
            else:
                # Single-step plan
                tool_name = plan.get("tool")
                tool = next((t for t in self.tools if t.name == tool_name), None)
                if not tool:
                    raise ValueError(f"Tool not found: {tool_name}")
                
                func = tool.func
                params = plan.get("parameters", {})
                
                if inspect.iscoroutinefunction(func):
                    return await func(**params)
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(self.executor, lambda: func(**params))
                    
        except Exception as e:
            raise RuntimeError(f"Strategy execution failed: {str(e)}")

    def __del__(self):
        """Clean up thread pool executor."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)