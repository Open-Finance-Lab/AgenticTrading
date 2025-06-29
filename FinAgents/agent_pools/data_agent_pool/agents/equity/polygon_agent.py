from typing import Dict, List, Optional, Union, Any, Callable
import pandas as pd
import requests
import os
from datetime import datetime
from FinAgents.agent_pools.data_agent_pool.base import BaseAgent
from FinAgents.agent_pools.data_agent_pool.schema.equity_schema import PolygonConfig
from .ticker_selector import select_top_tickers
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.agents import Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json
import logging
import re

load_dotenv()


class PolygonAgent(BaseAgent):
    """
    Enhanced Polygon.io data agent implementation.
    
    Features:
    - Historical OHLCV data with VWAP
    - Company information
    - Pre/post market prices
    - Dividend and split data
    - Top ticker selection based on volume and volatility
    """

    INTERVAL_MAP = {
        '1m': (1, 'minute'),
        '5m': (5, 'minute'),
        '15m': (15, 'minute'),
        '30m': (30, 'minute'),
        '1h': (1, 'hour'),
        '1d': (1, 'day')
    }

    def __init__(self, config: PolygonConfig):
        """
        Initialize enhanced market data agent.
        """
        super().__init__(config.model_dump())
        self.config = config  # Ensure config is a PolygonConfig instance
        self.api_base_url = self.config.api.base_url
        self.cache_dir = 'data/cache'
        os.makedirs(self.cache_dir, exist_ok=True)
        self._validate_config()
        self._init_tools()
        self._init_analysis_chain()
        if not hasattr(self.config, "llm_enabled"):
            raise ValueError("Missing required config parameter: 'llm_enabled'. Please add 'llm_enabled: true/false' to your polygon.yaml.")
        self.llm_enabled = bool(self.config.llm_enabled)
        print(f"llm_enabled config value: {self.llm_enabled}")
        if self.llm_enabled:
            self._init_llm_interface()

    def _init_llm_interface(self):
        """
        Configure LLM interface for market analysis.

        Establishes:
        - Language model connection
        - System prompts
        - Context management
        """
        self.llm = ChatOpenAI(
            model_name="o4-mini",
            temperature=1 
        )
        # Enhanced system prompt for robust multi-step, multi-symbol, multi-type planning
        self.system_prompt = SystemMessage(content="""
You are a professional financial data agent planner.

Your task is to generate an execution plan as a valid JSON object with a "steps" field (a list of tasks). Each step should specify:
- "tool": the tool to use, such as "fetch_market_data", "analyze_company", or "identify_leaders"
- "parameters": the parameters for the tool, such as "symbol", "start", "end", "interval"
- "type": the type of data, such as "market_data", "company_info", or "top_tickers"

If the user asks for multiple stocks or multiple types of data, include multiple steps in the "steps" list. 
If the user requests a single task, output a single-step plan.

**Only output a valid JSON object, and nothing else.**

Example user input:
"Get daily price data for AAPL and MSFT for January 2024, and also provide company information for both."

Example output:
{
  "steps": [
    {
      "tool": "fetch_market_data",
      "parameters": {
        "symbol": "AAPL",
        "start": "2024-01-01",
        "end": "2024-01-31",
        "interval": "1d"
      },
      "type": "market_data"
    },
    {
      "tool": "fetch_market_data",
      "parameters": {
        "symbol": "MSFT",
        "start": "2024-01-01",
        "end": "2024-01-31",
        "interval": "1d"
      },
      "type": "market_data"
    },
    {
      "tool": "analyze_company",
      "parameters": {
        "symbol": "AAPL"
      },
      "type": "company_info"
    },
    {
      "tool": "analyze_company",
      "parameters": {
        "symbol": "MSFT"
      },
      "type": "company_info"
    }
  ]
}

If the user wants the steps to be executed in a specific order, or in parallel, add a top-level field "execution_mode" with value "sequential" or "parallel". For example:

{
  "execution_mode": "parallel",
  "steps": [
    ...
  ]
}

You can only use the following tools: "fetch_market_data", "analyze_company", "identify_leaders".
Do not invent or use any other tool names.
""")

    def _init_tools(self):
        """
        Register available market data operations.
        
        Tools include:
        - Historical data retrieval
        - Company information access
        - Market metrics calculation
        - Top performer identification
        """
        self.tools = [
            Tool(
                name="fetch_market_data",
                func=self.fetch,
                description="Retrieve historical market data with specified parameters"
            ),
            Tool(
                name="analyze_company",
                func=self.get_company_info,
                description="Get comprehensive company analytics and metrics"
            ),
            Tool(
                name="identify_leaders",
                func=self.get_top_tickers,
                description="Find market leading stocks based on performance metrics"
            )
        ]

    def fetch(self, 
             symbol: str,
             start: str = None,
             end: str = None,
             interval: str = "1h",
             force_refresh: bool = False,
             **kwargs) -> pd.DataFrame:
        """
        Fetch comprehensive market data from Polygon.io.
        """
        # Accept 'from' and 'to' as aliases for 'start' and 'end'
        if start is None:
            start = kwargs.get("from")
        if end is None:
            end = kwargs.get("to")
        if not start or not end:
            raise ValueError("Missing required parameters: start/end (or from/to)")

        # Convert start and end to YYYY-MM-DD format
        try:
            start = datetime.strptime(start, "%Y-%m-%d").date()
            end = datetime.strptime(end, "%Y-%m-%d").date()
        except Exception as e:
            raise ValueError(f"Invalid date format for start or end: {str(e)}")

        # Check cache first
        cache_file = os.path.join(
            self.cache_dir, 
            f'{symbol}_{start}_{end}_{interval}.csv'
        )
        if not force_refresh and os.path.exists(cache_file):
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)

        # Validate interval
        if interval not in self.INTERVAL_MAP:
            raise ValueError(f"Unsupported interval: {interval}")
        multiplier, timespan = self.INTERVAL_MAP[interval]

        # Prepare API call
        url = f"{self.api_base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start}/{end}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.config.authentication.api_key
        }

        print("Plan parameters:", params)

        # Get OHLCV data
        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise RuntimeError(f"API error: {response.status_code} {response.text}")
        
        data = response.json()
        if not data.get("results"):
            raise ValueError(f"No data returned for {symbol}")

        # Convert to DataFrame
        df = pd.DataFrame(data["results"])
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.rename(columns={
            'o': 'open', 'h': 'high', 'l': 'low',
            'c': 'close', 'v': 'volume', 'n': 'trades'
        })

        # Fetch additional data (VWAP, pre/post market, etc.)
        if interval in ['1d', '1h']:
            self._enrich_market_data(df, symbol)

        # Cache results
        df.to_csv(cache_file)
        return df

    def _enrich_market_data(self, df: pd.DataFrame, symbol: str) -> None:
        """Add VWAP, pre/post market prices, dividends and splits."""
        for date in df.index.date:
            date_str = date.strftime('%Y-%m-%d')
            url = f"{self.api_base_url}/v1/open-close/{symbol}/{date_str}"
            params = {"adjusted": "true", "apiKey": self.config.authentication.api_key}
            
            try:
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    mask = df.index.date == date
                    df.loc[mask, 'vwap'] = data.get('vwap')
                    df.loc[mask, 'pre_market'] = data.get('preMarket')
                    df.loc[mask, 'after_market'] = data.get('afterHours')
            except Exception as e:
                print(f"Failed to fetch daily details: {str(e)}")

    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """Get detailed company information."""
        try:
            url = f"{self.api_base_url}/v3/reference/tickers/{symbol}"
            params = {"apiKey": self.config.authentication.api_key}
            
            response = requests.get(url, params=params)
            if response.status_code != 200:
                raise RuntimeError(f"API error: {response.status_code}")

            data = response.json()["results"]
            return {
                "symbol": data["ticker"],
                "name": data["name"],
                "market": data["market"],
                "locale": data["locale"],
                "type": data["type"],
                "currency": data["currency_name"],
                "outstanding_shares": data.get("share_class_shares_outstanding"),
                "market_cap": data.get("market_cap"),
                "description": data.get("description")
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get company info: {str(e)}")

    def get_top_tickers(self, n: int = 5) -> List[str]:
        """Get top n tickers based on volume and volatility."""
        return select_top_tickers(n)

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if not getattr(self.config.authentication, "api_key", None):
            raise ValueError("Missing required Polygon API key")

    def _init_analysis_chain(self):
        """
        Placeholder for initializing advanced analysis chains.
        Extend this method to add custom LLM-based analysis pipelines.
        """
        pass

    def _parse_intent(self, llm_output: str) -> dict:
        """
        Parse the LLM output into a validated execution plan.
        Supports both single-step and multi-step (steps) plans.
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
                    "tool": "fetch_market_data",
                    "parameters": {
                        "symbol": "AAPL",
                        "start": "2024-01-01",
                        "end": "2024-01-31",
                        "interval": "1d"
                    },
                    "type": "market_data"
                }

        # --- Validation ---
        if "steps" in plan:
            # Multi-step plan
            if not isinstance(plan["steps"], list) or not plan["steps"]:
                raise ValueError("Execution plan 'steps' must be a non-empty list.")
            for step in plan["steps"]:
                for field in ["tool", "parameters"]:
                    if field not in step:
                        raise ValueError(f"Step missing required field: {field}")
        else:
            # Single-step plan
            for field in ["tool", "parameters"]:
                if field not in plan:
                    raise ValueError(f"Execution plan missing required field: {field}")

        return plan

    async def process_intent(self, query: str) -> Dict[str, Any]:
        """
        Process natural language market data requests.
        If LLM is disabled, return a default plan and result for testing.
        """
        if not getattr(self, "llm_enabled", True):
            plan = {
                "tool": "fetch_market_data",
                "parameters": {
                    "symbol": "AAPL",
                    "start": "2024-01-01",
                    "end": "2024-01-31",
                    "interval": "1d"
                },
                "type": "market_data"
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

        # Normal LLM-driven path
        intent_analysis = await self.llm.agenerate([
            [self.system_prompt, HumanMessage(content=query)]
        ])
        plan = self._parse_intent(intent_analysis.generations[0][0].text)
        print("=== Execution Plan ===")
        print(plan)
        result = await self._execute_strategy(plan)
        return {
            "execution_plan": plan,
            "result": result,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "query_type": plan.get("type"),
                "data_points": len(result) if isinstance(result, pd.DataFrame) else 1,
                "llm_used": True
            }
        }

    async def _execute_strategy(self, plan: Dict) -> Any:
        """
        Execute generated market data strategy.
        Supports both single-step and multi-step (steps) plans.
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

    def start_mcp_server(self, port: int = 8002, host: str = "0.0.0.0", transport: str = "sse"):
        """
        Start the MCP server for the PolygonAgent with natural language interface.
        
        Args:
            port (int): The port to bind the server to. Defaults to 8002.
            host (str): The host address to bind. Defaults to "0.0.0.0".
            transport (str): The transport protocol to use. Defaults to "sse".
        """
        from mcp.server.fastmcp import FastMCP
        
        # Create MCP server instance
        self.mcp_server = FastMCP("PolygonAgent")
        
        # Register natural language interface tool
        @self.mcp_server.tool(name="process_market_query", description="Process natural language market data queries")
        async def process_market_query(query: str) -> dict:
            """
            Process natural language market data requests.
            
            Args:
                query: Natural language query (e.g., "Get daily data for AAPL from 2024-01-01 to 2024-12-31")
            
            Returns:
                dict: Structured response with execution plan and results
            """
            try:
                result = await self.process_intent(query)
                return {
                    "status": "success",
                    "query": query,
                    "execution_plan": result.get("execution_plan"),
                    "result": result.get("result"),
                    "metadata": result.get("metadata")
                }
            except Exception as e:
                return {
                    "status": "error",
                    "query": query,
                    "error": str(e)
                }
        
        # Register direct data fetch tool
        @self.mcp_server.tool(name="fetch_market_data", description="Directly fetch market data for a symbol")
        def fetch_market_data(symbol: str, start: str, end: str, interval: str = "1d") -> dict:
            """
            Directly fetch market data for a specific symbol.
            
            Args:
                symbol: Stock symbol (e.g., 'AAPL')
                start: Start date (YYYY-MM-DD)
                end: End date (YYYY-MM-DD)
                interval: Time interval (1d, 1h, etc.)
            
            Returns:
                dict: Market data response
            """
            try:
                df = self.fetch(symbol=symbol, start=start, end=end, interval=interval)
                return {
                    "status": "success",
                    "symbol": symbol,
                    "data": df.to_dict(orient="records"),
                    "count": len(df)
                }
            except Exception as e:
                return {
                    "status": "error",
                    "symbol": symbol,
                    "error": str(e)
                }
        
        # Register company info tool
        @self.mcp_server.tool(name="get_company_info", description="Get company information for a symbol")
        def get_company_info_tool(symbol: str) -> dict:
            """
            Get company information for a specific symbol.
            
            Args:
                symbol: Stock symbol (e.g., 'AAPL')
            
            Returns:
                dict: Company information response
            """
            try:
                info = self.get_company_info(symbol)
                return {
                    "status": "success", 
                    "symbol": symbol,
                    "company_info": info
                }
            except Exception as e:
                return {
                    "status": "error",
                    "symbol": symbol,
                    "error": str(e)
                }
        
        # Register health check tool
        @self.mcp_server.tool(name="health_check", description="Health check for PolygonAgent")
        def health_check() -> dict:
            """
            Return the health status of the PolygonAgent.
            
            Returns:
                dict: Health status
            """
            return {
                "status": "ok",
                "agent": "PolygonAgent",
                "timestamp": datetime.now().isoformat(),
                "llm_enabled": getattr(self, "llm_enabled", True),
                "api_configured": bool(self.config.authentication.api_key)
            }
        
        # Start the server
        self.mcp_server.settings.host = host
        self.mcp_server.settings.port = port
        print(f"Starting PolygonAgent MCP server on {host}:{port} (transport={transport}) ...")
        print("=== Registered Tools ===")
        import asyncio
        tools = asyncio.run(self.mcp_server.list_tools())
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")
        
        self.mcp_server.run(transport=transport)

