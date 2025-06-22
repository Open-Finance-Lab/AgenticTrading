from typing import Dict, List, Optional, Union, Any, Callable
import pandas as pd
import requests
import os
from datetime import datetime, timedelta
from agent_pools.data_agent_pool.registry import BaseAgent
from agent_pools.data_agent_pool.schema.equity_schema import CoinGeckoConfig
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.agents import Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json
import logging
import re
import time

load_dotenv()


class CoinGeckoAgent(BaseAgent):
    """
    Enhanced CoinGecko data agent implementation.
    
    Features:
    - Historical OHLCV data for cryptocurrencies
    - Coin information and metadata
    - Market data and statistics
    - Price data in multiple currencies
    - Top coins by market cap and volume
    - Market trends and global statistics
    """

    INTERVAL_MAP = {
        '1h': 'hourly',
        '1d': 'daily',
        '7d': 'weekly',
        '30d': 'monthly'
    }

    CURRENCY_OPTIONS = [
        'usd', 'eur', 'jpy', 'btc', 'eth', 'ltc', 'bch', 'bnb', 'eos', 'xrp', 'xlm', 
        'link', 'dot', 'yfi', 'gbp', 'aud', 'cad', 'chf', 'cny', 'krw', 'rub'
    ]

    def __init__(self, config: CoinGeckoConfig):
        """
        Initialize enhanced cryptocurrency data agent.
        """
        super().__init__(config.model_dump())
        self.config = config
        self.api_base_url = self.config.api.base_url
        self.cache_dir = 'data/cache/coingecko'
        os.makedirs(self.cache_dir, exist_ok=True)
        self._validate_config()
        self._init_tools()
        self._init_analysis_chain()
        
        if not hasattr(self.config, "llm_enabled"):
            raise ValueError("Missing required config parameter: 'llm_enabled'. Please add 'llm_enabled: true/false' to your coingecko.yaml.")
        
        self.llm_enabled = bool(self.config.llm_enabled)
        print(f"llm_enabled config value: {self.llm_enabled}")
        
        if self.llm_enabled:
            self._init_llm_interface()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 60 / self.config.constraints.rate_limit_per_minute

    def _init_llm_interface(self):
        """
        Configure LLM interface for cryptocurrency market analysis.
        """
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1
        )
        
        self.system_prompt = SystemMessage(content="""
You are a professional cryptocurrency data agent planner.

Your task is to generate an execution plan as a valid JSON object with a "steps" field (a list of tasks). Each step should specify:
- "tool": the tool to use, such as "fetch_crypto_data", "analyze_coin", "get_market_stats", or "identify_top_coins"
- "parameters": the parameters for the tool, such as "coin_id", "days", "currency", "interval"
- "type": the type of data, such as "price_data", "coin_info", "market_data", or "top_coins"

If the user asks for multiple cryptocurrencies or multiple types of data, include multiple steps in the "steps" list.
If the user requests a single task, output a single-step plan.

**Only output a valid JSON object, and nothing else.**

Example user input:
"Get daily price data for Bitcoin and Ethereum for the last 30 days in USD, and also provide coin information for both."

Example output:
{
  "steps": [
    {
      "tool": "fetch_crypto_data",
      "parameters": {
        "coin_id": "bitcoin",
        "days": 30,
        "currency": "usd",
        "interval": "daily"
      },
      "type": "price_data"
    },
    {
      "tool": "fetch_crypto_data",
      "parameters": {
        "coin_id": "ethereum",
        "days": 30,
        "currency": "usd",
        "interval": "daily"
      },
      "type": "price_data"
    },
    {
      "tool": "analyze_coin",
      "parameters": {
        "coin_id": "bitcoin"
      },
      "type": "coin_info"
    },
    {
      "tool": "analyze_coin",
      "parameters": {
        "coin_id": "ethereum"
      },
      "type": "coin_info"
    }
  ]
}

You can only use the following tools: "fetch_crypto_data", "analyze_coin", "get_market_stats", "identify_top_coins".
Do not invent or use any other tool names.

Common cryptocurrency IDs:
- bitcoin, ethereum, binancecoin, cardano, solana, polkadot, chainlink, litecoin, bitcoin-cash, stellar, dogecoin, etc.
- Use lowercase, hyphenated format (e.g., "bitcoin-cash" not "Bitcoin Cash")
""")

    def _init_tools(self):
        """
        Register available cryptocurrency data operations.
        """
        self.tools = [
            Tool(
                name="fetch_crypto_data",
                func=self.fetch,
                description="Retrieve historical cryptocurrency price data"
            ),
            Tool(
                name="analyze_coin",
                func=self.get_coin_info,
                description="Get comprehensive cryptocurrency information and metrics"
            ),
            Tool(
                name="get_market_stats",
                func=self.get_global_stats,
                description="Get global cryptocurrency market statistics"
            ),
            Tool(
                name="identify_top_coins",
                func=self.get_top_coins,
                description="Find top cryptocurrencies by market cap or volume"
            )
        ]

    def _rate_limit(self):
        """Implement rate limiting to respect API limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    def fetch(self, 
             coin_id: str,
             days: int = 30,
             currency: str = "usd",
             interval: str = "daily",
             force_refresh: bool = False,
             **kwargs) -> pd.DataFrame:
        """
        Fetch comprehensive cryptocurrency market data from CoinGecko.
        
        Args:
            coin_id: CoinGecko coin identifier (e.g., 'bitcoin', 'ethereum')
            days: Number of days of historical data (1-max)
            currency: Target currency (usd, eur, btc, etc.)
            interval: Data interval ('daily' for > 90 days, 'hourly' for <= 90 days)
            force_refresh: Force refresh cached data
        """
        # Validate inputs
        if currency.lower() not in self.CURRENCY_OPTIONS:
            raise ValueError(f"Unsupported currency: {currency}. Supported: {self.CURRENCY_OPTIONS}")
        
        # Check cache first
        cache_file = os.path.join(
            self.cache_dir, 
            f'{coin_id}_{days}d_{currency}_{interval}.csv'
        )
        
        if not force_refresh and os.path.exists(cache_file):
            # Check if cache is recent (within 1 hour for crypto data)
            cache_age = time.time() - os.path.getmtime(cache_file)
            if cache_age < 3600:  # 1 hour
                return pd.read_csv(cache_file, index_col=0, parse_dates=True)

        self._rate_limit()

        # Prepare API call
        url = f"{self.api_base_url}/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": currency.lower(),
            "days": days,
            "interval": "daily" if days > 90 else "hourly"
        }
        
        # Add API key if available
        if hasattr(self.config.authentication, 'api_key') and self.config.authentication.api_key:
            params["x_cg_demo_api_key"] = self.config.authentication.api_key

        print(f"Fetching data for {coin_id} - {days} days in {currency}")

        # Make API request
        response = requests.get(url, params=params, timeout=self.config.constraints.timeout)
        if response.status_code != 200:
            raise RuntimeError(f"CoinGecko API error: {response.status_code} {response.text}")
        
        data = response.json()
        
        if not all(key in data for key in ['prices', 'market_caps', 'total_volumes']):
            raise ValueError(f"Incomplete data returned for {coin_id}")

        # Convert to DataFrame
        df = pd.DataFrame({
            'timestamp': [datetime.fromtimestamp(item[0]/1000) for item in data['prices']],
            'price': [item[1] for item in data['prices']],
            'market_cap': [item[1] for item in data['market_caps']],
            'volume': [item[1] for item in data['total_volumes']]
        })
        
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # Add derived metrics
        df['price_change'] = df['price'].pct_change()
        df['volume_sma_7'] = df['volume'].rolling(window=7).mean()
        df['price_sma_7'] = df['price'].rolling(window=7).mean()
        df['price_sma_30'] = df['price'].rolling(window=30).mean()
        
        # Add volatility
        df['volatility'] = df['price_change'].rolling(window=7).std() * (365**0.5)

        # Cache results
        df.to_csv(cache_file)
        return df

    def get_coin_info(self, coin_id: str) -> Dict[str, Any]:
        """Get detailed cryptocurrency information."""
        self._rate_limit()
        
        try:
            url = f"{self.api_base_url}/coins/{coin_id}"
            params = {
                "localization": "false",
                "tickers": "false",
                "market_data": "true",
                "community_data": "true",
                "developer_data": "true"
            }
            
            # Add API key if available
            if hasattr(self.config.authentication, 'api_key') and self.config.authentication.api_key:
                params["x_cg_demo_api_key"] = self.config.authentication.api_key
            
            response = requests.get(url, params=params, timeout=self.config.constraints.timeout)
            if response.status_code != 200:
                raise RuntimeError(f"CoinGecko API error: {response.status_code}")

            data = response.json()
            market_data = data.get("market_data", {})
            
            return {
                "id": data["id"],
                "symbol": data["symbol"].upper(),
                "name": data["name"],
                "description": data.get("description", {}).get("en", "").split('.')[0] if data.get("description", {}).get("en") else "",
                "market_cap_rank": market_data.get("market_cap_rank"),
                "current_price_usd": market_data.get("current_price", {}).get("usd"),
                "market_cap_usd": market_data.get("market_cap", {}).get("usd"),
                "total_volume_usd": market_data.get("total_volume", {}).get("usd"),
                "price_change_24h": market_data.get("price_change_percentage_24h"),
                "price_change_7d": market_data.get("price_change_percentage_7d"),
                "price_change_30d": market_data.get("price_change_percentage_30d"),
                "circulating_supply": market_data.get("circulating_supply"),
                "total_supply": market_data.get("total_supply"),
                "max_supply": market_data.get("max_supply"),
                "ath": market_data.get("ath", {}).get("usd"),
                "ath_date": market_data.get("ath_date", {}).get("usd"),
                "atl": market_data.get("atl", {}).get("usd"),
                "atl_date": market_data.get("atl_date", {}).get("usd"),
                "homepage": data.get("links", {}).get("homepage", [None])[0],
                "blockchain_site": data.get("links", {}).get("blockchain_site", []),
                "categories": data.get("categories", [])
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get coin info for {coin_id}: {str(e)}")

    def get_global_stats(self) -> Dict[str, Any]:
        """Get global cryptocurrency market statistics."""
        self._rate_limit()
        
        try:
            url = f"{self.api_base_url}/global"
            params = {}
            
            # Add API key if available
            if hasattr(self.config.authentication, 'api_key') and self.config.authentication.api_key:
                params["x_cg_demo_api_key"] = self.config.authentication.api_key
            
            response = requests.get(url, params=params, timeout=self.config.constraints.timeout)
            if response.status_code != 200:
                raise RuntimeError(f"CoinGecko API error: {response.status_code}")

            data = response.json()["data"]
            
            return {
                "active_cryptocurrencies": data["active_cryptocurrencies"],
                "upcoming_icos": data["upcoming_icos"],
                "ongoing_icos": data["ongoing_icos"],
                "ended_icos": data["ended_icos"],
                "markets": data["markets"],
                "total_market_cap_usd": data["total_market_cap"].get("usd"),
                "total_volume_24h_usd": data["total_volume"].get("usd"),
                "market_cap_percentage": data["market_cap_percentage"],
                "market_cap_change_24h": data["market_cap_change_percentage_24h_usd"],
                "updated_at": data["updated_at"]
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get global stats: {str(e)}")

    def get_top_coins(self, 
                     limit: int = 10, 
                     currency: str = "usd",
                     order: str = "market_cap_desc") -> List[Dict[str, Any]]:
        """Get top cryptocurrencies by market cap, volume, or other metrics."""
        self._rate_limit()
        
        try:
            url = f"{self.api_base_url}/coins/markets"
            params = {
                "vs_currency": currency.lower(),
                "order": order,
                "per_page": limit,
                "page": 1,
                "sparkline": "false",
                "price_change_percentage": "1h,24h,7d,30d"
            }
            
            # Add API key if available
            if hasattr(self.config.authentication, 'api_key') and self.config.authentication.api_key:
                params["x_cg_demo_api_key"] = self.config.authentication.api_key
            
            response = requests.get(url, params=params, timeout=self.config.constraints.timeout)
            if response.status_code != 200:
                raise RuntimeError(f"CoinGecko API error: {response.status_code}")

            data = response.json()
            
            return [{
                "id": coin["id"],
                "symbol": coin["symbol"].upper(),
                "name": coin["name"],
                "market_cap_rank": coin["market_cap_rank"],
                "current_price": coin["current_price"],
                "market_cap": coin["market_cap"],
                "total_volume": coin["total_volume"],
                "price_change_1h": coin.get("price_change_percentage_1h_in_currency"),
                "price_change_24h": coin.get("price_change_percentage_24h"),
                "price_change_7d": coin.get("price_change_percentage_7d_in_currency"),
                "price_change_30d": coin.get("price_change_percentage_30d_in_currency"),
                "circulating_supply": coin["circulating_supply"],
                "total_supply": coin["total_supply"],
                "max_supply": coin["max_supply"]
            } for coin in data]
            
        except Exception as e:
            raise RuntimeError(f"Failed to get top coins: {str(e)}")

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if not self.config.api.base_url:
            raise ValueError("Missing required CoinGecko API base URL")
        
        # API key is optional for CoinGecko free tier
        if hasattr(self.config.authentication, 'api_key'):
            print("Using CoinGecko API key for higher rate limits")
        else:
            print("Using CoinGecko free tier (no API key)")

    def _init_analysis_chain(self):
        """
        Placeholder for initializing advanced analysis chains.
        """
        pass

    def _parse_intent(self, llm_output: str) -> dict:
        """
        Parse the LLM output into a validated execution plan.
        """
        import json, logging, re

        # Try to parse as JSON
        try:
            plan = json.loads(llm_output)
        except Exception:
            try:
                json_str = re.search(r'\{.*\}', llm_output, re.DOTALL).group()
                plan = json.loads(json_str)
            except Exception:
                logging.warning("LLM output is not valid JSON. Using default plan.")
                plan = {
                    "tool": "fetch_crypto_data",
                    "parameters": {
                        "coin_id": "bitcoin",
                        "days": 30,
                        "currency": "usd",
                        "interval": "daily"
                    },
                    "type": "price_data"
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
        """
        Process natural language cryptocurrency data requests.
        """
        if not getattr(self, "llm_enabled", True):
            plan = {
                "tool": "fetch_crypto_data",
                "parameters": {
                    "coin_id": "bitcoin",
                    "days": 30,
                    "currency": "usd",
                    "interval": "daily"
                },
                "type": "price_data"
            }
            result = await self._execute_strategy(plan)
            return {
                "execution_plan": plan,
                "result": result,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "query_type": plan.get("type"),
                    "data_points": len(result) if isinstance(result, pd.DataFrame) else 1,
                    "llm_used": False
                }
            }

        # LLM-driven path
        intent_analysis = await self.llm.agenerate([
            [self.system_prompt, HumanMessage(content=query)]
        ])
        plan = self._parse_intent(intent_analysis.generations[0][0].text)
        print("=== Execution Plan ===")
        print(json.dumps(plan, indent=2))
        result = await self._execute_strategy(plan)
        return {
            "execution_plan": plan,
            "result": result,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "query_type": plan.get("type") if "steps" not in plan else "multi_step",
                "data_points": len(result) if isinstance(result, pd.DataFrame) else 1,
                "llm_used": True
            }
        }

    async def _execute_strategy(self, plan: Dict) -> Any:
        """
        Execute generated cryptocurrency data strategy.
        """
        import inspect

        try:
            # Multi-step plan
            if "steps" in plan:
                results = []
                for step in plan["steps"]:
                    tool_name = step.get("tool")
                    tool = next((t for t in self.tools if t.name == tool_name), None)
                    if not tool:
                        available = [t.name for t in self.tools]
                        raise ValueError(f"Tool not found: {tool_name}. Available tools: {available}")
                    
                    func = tool.func
                    params = step.get("parameters", {})
                    
                    if inspect.iscoroutinefunction(func):
                        result = await func(**params)
                    else:
                        result = func(**params)
                    
                    results.append({
                        "step": tool_name,
                        "parameters": params,
                        "result": result
                    })
                return results
            
            # Single-step plan
            else:
                tool_name = plan.get("tool")
                tool = next((t for t in self.tools if t.name == tool_name), None)
                if not tool:
                    raise ValueError(f"Tool not found: {tool_name}")
                
                func = tool.func
                params = plan.get("parameters", {})
                
                if inspect.iscoroutinefunction(func):
                    return await func(**params)
                else:
                    return func(**params)
                    
        except Exception as e:
            raise RuntimeError(f"Strategy execution failed: {str(e)}")