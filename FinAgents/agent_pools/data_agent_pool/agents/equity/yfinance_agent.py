from typing import Dict, List, Optional, Union, Any, Callable
import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timedelta
from FinAgents.agent_pools.data_agent_pool.registry import BaseAgent
from FinAgents.agent_pools.data_agent_pool.schema.equity_schema import YFinanceConfig
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.agents import Tool
from dotenv import load_dotenv
import json
import logging
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor

load_dotenv()


class YFinanceAgent(BaseAgent):
    """
    Yahoo Finance data agent implementation using yfinance library.
    
    Features:
    - Historical OHLCV data
    - Real-time quotes
    - Company information and fundamentals
    - Financial statements (income, balance sheet, cash flow)
    - Options data
    - Dividend and split history
    - Market news
    - Analyst recommendations
    - ESG scores
    - Institutional holdings
    """

    PERIOD_MAP = {
        '1d': '1d', '5d': '5d', '1mo': '1mo', '3mo': '3mo',
        '6mo': '6mo', '1y': '1y', '2y': '2y', '5y': '5y',
        '10y': '10y', 'ytd': 'ytd', 'max': 'max'
    }

    INTERVAL_MAP = {
        '1m': '1m', '2m': '2m', '5m': '5m', '15m': '15m',
        '30m': '30m', '60m': '60m', '90m': '90m', '1h': '1h',
        '1d': '1d', '5d': '5d', '1wk': '1wk', '1mo': '1mo', '3mo': '3mo'
    }

    def __init__(self, config: YFinanceConfig):
        """Initialize Yahoo Finance data agent."""
        super().__init__(config.model_dump())
        self.config = config
        self.cache_dir = 'data/cache/yfinance'
        os.makedirs(self.cache_dir, exist_ok=True)
        self._validate_config()
        self._init_tools()
        self._init_analysis_chain()
        
        # Initialize thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        if not hasattr(self.config, "llm_enabled"):
            raise ValueError("Missing required config parameter: 'llm_enabled'. Please add 'llm_enabled: true/false' to your yfinance.yaml.")
        
        self.llm_enabled = bool(self.config.llm_enabled)
        print(f"YFinance Agent - llm_enabled config value: {self.llm_enabled}")
        
        if self.llm_enabled:
            self._init_llm_interface()

    def _init_llm_interface(self):
        """Configure LLM interface for market analysis."""
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1 
        )
        
        self.system_prompt = SystemMessage(content="""
You are a professional Yahoo Finance data agent planner.

Your task is to generate an execution plan as a valid JSON object with a "steps" field (a list of tasks). Each step should specify:
- "tool": the tool to use
- "parameters": the parameters for the tool
- "type": the type of data being requested

Available tools:
- "fetch_historical_data": Get historical OHLCV data
- "get_company_info": Get company information and fundamentals
- "get_financial_statements": Get income statement, balance sheet, cash flow
- "get_real_time_quote": Get current market data
- "get_options_data": Get options chain data
- "get_dividends_splits": Get dividend and split history
- "get_news": Get latest news for a symbol
- "get_recommendations": Get analyst recommendations
- "get_esg_scores": Get ESG (Environmental, Social, Governance) data
- "get_institutional_holders": Get institutional ownership data
- "get_market_summary": Get market indices summary

**Only output a valid JSON object, and nothing else.**

Example:
{
  "steps": [
    {
      "tool": "fetch_historical_data",
      "parameters": {
        "symbol": "AAPL",
        "period": "1y",
        "interval": "1d"
      },
      "type": "historical_data"
    },
    {
      "tool": "get_company_info",
      "parameters": {
        "symbol": "AAPL"
      },
      "type": "company_info"
    }
  ]
}
""")

    def _init_tools(self):
        """Register available Yahoo Finance operations."""
        self.tools = [
            Tool(
                name="fetch_historical_data",
                func=self.fetch_historical_data,
                description="Retrieve historical market data (OHLCV) for a symbol"
            ),
            Tool(
                name="get_company_info",
                func=self.get_company_info,
                description="Get comprehensive company information and fundamentals"
            ),
            Tool(
                name="get_financial_statements",
                func=self.get_financial_statements,
                description="Get financial statements (income, balance sheet, cash flow)"
            ),
            Tool(
                name="get_real_time_quote",
                func=self.get_real_time_quote,
                description="Get current market data and quote information"
            ),
            Tool(
                name="get_options_data",
                func=self.get_options_data,
                description="Get options chain data for a symbol"
            ),
            Tool(
                name="get_dividends_splits",
                func=self.get_dividends_splits,
                description="Get dividend and stock split history"
            ),
            Tool(
                name="get_news",
                func=self.get_news,
                description="Get latest news articles for a symbol"
            ),
            Tool(
                name="get_recommendations",
                func=self.get_recommendations,
                description="Get analyst recommendations and ratings"
            ),
            Tool(
                name="get_esg_scores",
                func=self.get_esg_scores,
                description="Get ESG (Environmental, Social, Governance) scores"
            ),
            Tool(
                name="get_institutional_holders",
                func=self.get_institutional_holders,
                description="Get institutional ownership information"
            ),
            Tool(
                name="get_market_summary",
                func=self.get_market_summary,
                description="Get market indices summary and overview"
            )
        ]

    def fetch_historical_data(self, 
                            symbol: str,
                            period: str = "1y",
                            interval: str = "1d",
                            start: str = None,
                            end: str = None,
                            force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch historical market data from Yahoo Finance.
        
        Args:
            symbol: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            start: Start date (YYYY-MM-DD format)
            end: End date (YYYY-MM-DD format)
            force_refresh: Force refresh from API
        """
        try:
            # Validate parameters
            if period not in self.PERIOD_MAP:
                raise ValueError(f"Invalid period: {period}. Valid periods: {list(self.PERIOD_MAP.keys())}")
            
            if interval not in self.INTERVAL_MAP:
                raise ValueError(f"Invalid interval: {interval}. Valid intervals: {list(self.INTERVAL_MAP.keys())}")

            # Check cache
            cache_key = f"{symbol}_{period}_{interval}_{start}_{end}"
            cache_file = os.path.join(self.cache_dir, f'{cache_key}.csv')
            
            if not force_refresh and os.path.exists(cache_file):
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                # Check if cache is recent (within last hour for intraday, last day for daily+)
                cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
                max_age = timedelta(hours=1) if interval.endswith('m') else timedelta(days=1)
                
                if cache_age < max_age:
                    return df

            # Fetch from Yahoo Finance
            ticker = yf.Ticker(symbol)
            
            if start and end:
                df = ticker.history(start=start, end=end, interval=interval)
            else:
                df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data returned for {symbol}")

            # Add additional metrics
            df['returns'] = df['Close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std() * (252 ** 0.5)  # Annualized volatility
            df['sma_20'] = df['Close'].rolling(window=20).mean()
            df['sma_50'] = df['Close'].rolling(window=50).mean()
            df['rsi'] = self._calculate_rsi(df['Close'])
            
            # Cache results
            df.to_csv(cache_file)
            return df
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch historical data for {symbol}: {str(e)}")

    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive company information."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                "symbol": symbol,
                "name": info.get("longName", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": info.get("marketCap", 0),
                "enterprise_value": info.get("enterpriseValue", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "forward_pe": info.get("forwardPE", 0),
                "peg_ratio": info.get("pegRatio", 0),
                "price_to_book": info.get("priceToBook", 0),
                "price_to_sales": info.get("priceToSalesTrailing12Months", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "beta": info.get("beta", 0),
                "52_week_high": info.get("fiftyTwoWeekHigh", 0),
                "52_week_low": info.get("fiftyTwoWeekLow", 0),
                "current_price": info.get("currentPrice", 0),
                "target_price": info.get("targetMeanPrice", 0),
                "recommendation": info.get("recommendationKey", "N/A"),
                "employees": info.get("fullTimeEmployees", 0),
                "website": info.get("website", "N/A"),
                "business_summary": info.get("longBusinessSummary", "N/A")
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to get company info for {symbol}: {str(e)}")

    def get_financial_statements(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Get financial statements (income statement, balance sheet, cash flow)."""
        try:
            ticker = yf.Ticker(symbol)
            
            return {
                "income_statement": ticker.financials,
                "balance_sheet": ticker.balance_sheet,
                "cash_flow": ticker.cashflow,
                "quarterly_income": ticker.quarterly_financials,
                "quarterly_balance": ticker.quarterly_balance_sheet,
                "quarterly_cashflow": ticker.quarterly_cashflow
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to get financial statements for {symbol}: {str(e)}")

    def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """Get current market data and quote information."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get latest price data
            hist = ticker.history(period="1d", interval="1m")
            latest = hist.iloc[-1] if not hist.empty else {}
            
            return {
                "symbol": symbol,
                "current_price": info.get("currentPrice", latest.get("Close", 0)),
                "previous_close": info.get("previousClose", 0),
                "open": info.get("open", latest.get("Open", 0)),
                "day_high": info.get("dayHigh", latest.get("High", 0)),
                "day_low": info.get("dayLow", latest.get("Low", 0)),
                "volume": info.get("volume", latest.get("Volume", 0)),
                "avg_volume": info.get("averageVolume", 0),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "change": info.get("currentPrice", 0) - info.get("previousClose", 0),
                "change_percent": ((info.get("currentPrice", 0) - info.get("previousClose", 0)) / info.get("previousClose", 1)) * 100,
                "bid": info.get("bid", 0),
                "ask": info.get("ask", 0),
                "bid_size": info.get("bidSize", 0),
                "ask_size": info.get("askSize", 0)
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to get real-time quote for {symbol}: {str(e)}")

    def get_options_data(self, symbol: str, expiration_date: str = None) -> Dict[str, pd.DataFrame]:
        """Get options chain data."""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get available expiration dates
            expirations = ticker.options
            
            if not expirations:
                return {"error": "No options data available"}
            
            # Use first available expiration if none specified
            exp_date = expiration_date if expiration_date in expirations else expirations[0]
            
            # Get options chain
            options_chain = ticker.option_chain(exp_date)
            
            return {
                "expiration_date": exp_date,
                "available_expirations": list(expirations),
                "calls": options_chain.calls,
                "puts": options_chain.puts
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to get options data for {symbol}: {str(e)}")

    def get_dividends_splits(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Get dividend and stock split history."""
        try:
            ticker = yf.Ticker(symbol)
            
            return {
                "dividends": ticker.dividends,
                "splits": ticker.splits,
                "actions": ticker.actions  # Combined dividends and splits
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to get dividends/splits for {symbol}: {str(e)}")

    def get_news(self, symbol: str, max_items: int = 10) -> List[Dict[str, Any]]:
        """Get latest news articles for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            return [
                {
                    "title": article.get("title", ""),
                    "link": article.get("link", ""),
                    "published": article.get("providerPublishTime", 0),
                    "publisher": article.get("publisher", ""),
                    "summary": article.get("summary", "")
                }
                for article in news[:max_items]
            ]
            
        except Exception as e:
            raise RuntimeError(f"Failed to get news for {symbol}: {str(e)}")

    def get_recommendations(self, symbol: str) -> pd.DataFrame:
        """Get analyst recommendations and ratings."""
        try:
            ticker = yf.Ticker(symbol)
            return ticker.recommendations
            
        except Exception as e:
            raise RuntimeError(f"Failed to get recommendations for {symbol}: {str(e)}")

    def get_esg_scores(self, symbol: str) -> Dict[str, Any]:
        """Get ESG (Environmental, Social, Governance) scores."""
        try:
            ticker = yf.Ticker(symbol)
            sustainability = ticker.sustainability
            
            if sustainability is None or sustainability.empty:
                return {"error": "No ESG data available"}
            
            return sustainability.to_dict()
            
        except Exception as e:
            raise RuntimeError(f"Failed to get ESG scores for {symbol}: {str(e)}")

    def get_institutional_holders(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Get institutional ownership information."""
        try:
            ticker = yf.Ticker(symbol)
            
            return {
                "institutional_holders": ticker.institutional_holders,
                "major_holders": ticker.major_holders,
                "mutual_fund_holders": ticker.mutualfund_holders
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to get institutional holders for {symbol}: {str(e)}")

    def get_market_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get market indices summary."""
        try:
            indices = {
                "S&P 500": "^GSPC",
                "Dow Jones": "^DJI", 
                "NASDAQ": "^IXIC",
                "Russell 2000": "^RUT",
                "VIX": "^VIX"
            }
            
            summary = {}
            for name, symbol in indices.items():
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    hist = ticker.history(period="2d")
                    
                    if not hist.empty:
                        current = hist.iloc[-1]["Close"]
                        previous = hist.iloc[-2]["Close"] if len(hist) > 1 else current
                        change = current - previous
                        change_pct = (change / previous) * 100 if previous != 0 else 0
                        
                        summary[name] = {
                            "symbol": symbol,
                            "current_price": current,
                            "change": change,
                            "change_percent": change_pct,
                            "volume": hist.iloc[-1].get("Volume", 0)
                        }
                except Exception as e:
                    summary[name] = {"error": str(e)}
            
            return summary
            
        except Exception as e:
            raise RuntimeError(f"Failed to get market summary: {str(e)}")

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
        # YFinance doesn't require API keys, so minimal validation
        if not hasattr(self.config, 'cache_enabled'):
            self.config.cache_enabled = True

    def _init_analysis_chain(self):
        """Initialize analysis chains for advanced market analysis."""
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
                        "tool": "get_company_info",
                        "parameters": {"symbol": "AAPL"},
                        "type": "company_info"
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
        """Process natural language market data requests."""
        if not getattr(self, "llm_enabled", True):
            # Default plan for testing
            plan = {
                "steps": [{
                    "tool": "get_company_info",
                    "parameters": {"symbol": "AAPL"},
                    "type": "company_info"
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
        print("=== YFinance Execution Plan ===")
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
        """Execute generated market data strategy."""
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