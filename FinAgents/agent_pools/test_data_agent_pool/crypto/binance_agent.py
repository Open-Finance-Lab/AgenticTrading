from typing import Dict, List, Optional, Union, Any, Callable
import pandas as pd
import os
from datetime import datetime, timedelta
from agent_pools.data_agent_pool.registry import BaseAgent
from agent_pools.data_agent_pool.schema.crypto_schema import BinanceConfig
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.agents import Tool
from dotenv import load_dotenv
import json
import logging
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
import ccxt
import numpy as np
from decimal import Decimal, ROUND_DOWN
import time

load_dotenv()


class BinanceAgent(BaseAgent):
    """
    Binance trading agent implementation using ccxt library.
    
    Features:
    - Spot and Futures trading
    - Historical OHLCV data
    - Real-time market data
    - Account management
    - Order management (market, limit, stop-loss, take-profit)
    - Portfolio tracking
    - Risk management
    - Technical indicators
    - Market analysis
    - News and announcements
    - Funding rates and perpetual data
    """

    TIMEFRAME_MAP = {
        '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m',
        '30m': '30m', '1h': '1h', '2h': '2h', '4h': '4h',
        '6h': '6h', '8h': '8h', '12h': '12h', '1d': '1d',
        '3d': '3d', '1w': '1w', '1M': '1M'
    }

    ORDER_TYPES = ['market', 'limit', 'stop_market', 'stop_limit', 'take_profit', 'take_profit_limit']
    
    def __init__(self, config: BinanceConfig):
        """Initialize Binance trading agent."""
        super().__init__(config.model_dump())
        self.config = config
        self.cache_dir = 'data/cache/binance'
        os.makedirs(self.cache_dir, exist_ok=True)
        self._validate_config()
        self._init_exchange()
        self._init_tools()
        self._init_analysis_chain()
        
        # Initialize thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        if not hasattr(self.config, "llm_enabled"):
            raise ValueError("Missing required config parameter: 'llm_enabled'. Please add 'llm_enabled: true/false' to your binance.yaml.")
        
        self.llm_enabled = bool(self.config.llm_enabled)
        print(f"Binance Agent - llm_enabled config value: {self.llm_enabled}")
        
        if self.llm_enabled:
            self._init_llm_interface()

    def _init_exchange(self):
        """Initialize Binance exchange connection."""
        try:
            self.exchange = ccxt.binance({
                'apiKey': self.config.api_key,
                'secret': self.config.api_secret,
                'sandbox': getattr(self.config, 'sandbox', False),
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',  # 'spot' or 'future'
                }
            })
            
            # Test connection
            if self.config.api_key and self.config.api_secret:
                self.exchange.check_required_credentials()
                print("✅ Binance API connection established")
            else:
                print("⚠️ No API credentials provided - running in data-only mode")
                
        except Exception as e:
            print(f"❌ Failed to initialize Binance connection: {str(e)}")
            raise

    def _init_llm_interface(self):
        """Configure LLM interface for trading analysis."""
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1 
        )
        
        self.system_prompt = SystemMessage(content="""
You are a professional Binance trading agent planner.

Your task is to generate an execution plan as a valid JSON object with a "steps" field (a list of tasks). Each step should specify:
- "tool": the tool to use
- "parameters": the parameters for the tool
- "type": the type of operation being requested

Available tools:
- "fetch_ohlcv_data": Get historical OHLCV data
- "get_ticker": Get real-time ticker data
- "get_order_book": Get order book data
- "get_account_balance": Get account balance information
- "place_order": Place a trading order
- "cancel_order": Cancel an existing order
- "get_open_orders": Get all open orders
- "get_trade_history": Get trading history
- "get_market_summary": Get market overview
- "get_funding_rates": Get futures funding rates
- "get_perpetual_positions": Get perpetual futures positions
- "calculate_technical_indicators": Calculate technical analysis indicators
- "perform_risk_analysis": Analyze portfolio risk
- "get_top_gainers_losers": Get top performing cryptocurrencies
- "search_trading_pairs": Search for available trading pairs
- "get_24h_stats": Get 24-hour trading statistics

**Only output a valid JSON object, and nothing else.**

Example:
{
  "steps": [
    {
      "tool": "fetch_ohlcv_data",
      "parameters": {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "limit": 100
      },
      "type": "market_data"
    },
    {
      "tool": "get_ticker",
      "parameters": {
        "symbol": "BTC/USDT"
      },
      "type": "real_time_data"
    }
  ]
}
""")

    def _init_tools(self):
        """Register available Binance operations."""
        self.tools = [
            Tool(
                name="fetch_ohlcv_data",
                func=self.fetch_ohlcv_data,
                description="Retrieve historical OHLCV data for a trading pair"
            ),
            Tool(
                name="get_ticker",
                func=self.get_ticker,
                description="Get real-time ticker data for a symbol"
            ),
            Tool(
                name="get_order_book",
                func=self.get_order_book,
                description="Get order book data for a trading pair"
            ),
            Tool(
                name="get_account_balance",
                func=self.get_account_balance,
                description="Get account balance information"
            ),
            Tool(
                name="place_order",
                func=self.place_order,
                description="Place a trading order (market, limit, stop-loss, etc.)"
            ),
            Tool(
                name="cancel_order",
                func=self.cancel_order,
                description="Cancel an existing order"
            ),
            Tool(
                name="get_open_orders",
                func=self.get_open_orders,
                description="Get all open orders"
            ),
            Tool(
                name="get_trade_history",
                func=self.get_trade_history,
                description="Get trading history"
            ),
            Tool(
                name="get_market_summary",
                func=self.get_market_summary,
                description="Get comprehensive market overview"
            ),
            Tool(
                name="get_funding_rates",
                func=self.get_funding_rates,
                description="Get futures funding rates"
            ),
            Tool(
                name="get_perpetual_positions",
                func=self.get_perpetual_positions,
                description="Get perpetual futures positions"
            ),
            Tool(
                name="calculate_technical_indicators",
                func=self.calculate_technical_indicators,
                description="Calculate technical analysis indicators"
            ),
            Tool(
                name="perform_risk_analysis",
                func=self.perform_risk_analysis,
                description="Analyze portfolio risk metrics"
            ),
            Tool(
                name="get_top_gainers_losers",
                func=self.get_top_gainers_losers,
                description="Get top performing cryptocurrencies"
            ),
            Tool(
                name="search_trading_pairs",
                func=self.search_trading_pairs,
                description="Search for available trading pairs"
            ),
            Tool(
                name="get_24h_stats",
                func=self.get_24h_stats,
                description="Get 24-hour trading statistics"
            )
        ]

    def fetch_ohlcv_data(self, 
                        symbol: str,
                        timeframe: str = "1h",
                        limit: int = 100,
                        since: int = None,
                        force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Binance.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            limit: Number of candles to fetch (max 1000)
            since: Timestamp to fetch data from
            force_refresh: Force refresh from API
        """
        try:
            # Validate parameters
            if timeframe not in self.TIMEFRAME_MAP:
                raise ValueError(f"Invalid timeframe: {timeframe}. Valid timeframes: {list(self.TIMEFRAME_MAP.keys())}")
            
            if limit > 1000:
                limit = 1000
                print("⚠️ Limit reduced to 1000 (Binance API maximum)")

            # Check cache
            cache_key = f"{symbol.replace('/', '_')}_{timeframe}_{limit}_{since}"
            cache_file = os.path.join(self.cache_dir, f'{cache_key}.csv')
            
            if not force_refresh and os.path.exists(cache_file):
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                # Check if cache is recent (within last 5 minutes for short timeframes)
                cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
                max_age = timedelta(minutes=5) if timeframe.endswith('m') else timedelta(hours=1)
                
                if cache_age < max_age and len(df) > 0:
                    return df

            # Fetch from Binance
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            
            if not ohlcv:
                raise ValueError(f"No data returned for {symbol}")

            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add technical indicators
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(24)  # Daily volatility for hourly data
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            df['rsi'] = self._calculate_rsi(df['close'])
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Cache results
            df.to_csv(cache_file)
            return df
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch OHLCV data for {symbol}: {str(e)}")

    def get_ticker(self, symbol: str = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get real-time ticker data."""
        try:
            if symbol:
                ticker = self.exchange.fetch_ticker(symbol)
                return {
                    "symbol": ticker['symbol'],
                    "last_price": ticker['last'],
                    "bid": ticker['bid'],
                    "ask": ticker['ask'],
                    "high_24h": ticker['high'],
                    "low_24h": ticker['low'],
                    "volume_24h": ticker['baseVolume'],
                    "quote_volume_24h": ticker['quoteVolume'],
                    "change_24h": ticker['change'],
                    "change_percent_24h": ticker['percentage'],
                    "timestamp": ticker['timestamp']
                }
            else:
                # Get all tickers
                tickers = self.exchange.fetch_tickers()
                return [
                    {
                        "symbol": ticker['symbol'],
                        "last_price": ticker['last'],
                        "change_percent_24h": ticker['percentage'],
                        "volume_24h": ticker['baseVolume']
                    }
                    for ticker in tickers.values()
                ]
        except Exception as e:
            raise RuntimeError(f"Failed to get ticker data: {str(e)}")

    def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Get order book data."""
        try:
            order_book = self.exchange.fetch_order_book(symbol, limit)
            
            return {
                "symbol": symbol,
                "bids": order_book['bids'][:10],  # Top 10 bids
                "asks": order_book['asks'][:10],  # Top 10 asks
                "timestamp": order_book['timestamp'],
                "bid_total_volume": sum([bid[1] for bid in order_book['bids']]),
                "ask_total_volume": sum([ask[1] for ask in order_book['asks']]),
                "spread": order_book['asks'][0][0] - order_book['bids'][0][0] if order_book['asks'] and order_book['bids'] else 0,
                "spread_percent": ((order_book['asks'][0][0] - order_book['bids'][0][0]) / order_book['bids'][0][0]) * 100 if order_book['asks'] and order_book['bids'] else 0
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get order book for {symbol}: {str(e)}")

    def get_account_balance(self) -> Dict[str, Any]:
        """Get account balance information."""
        try:
            if not self.config.api_key:
                return {"error": "API credentials required for account operations"}
            
            balance = self.exchange.fetch_balance()
            
            # Process balances
            total_balance = {}
            free_balance = {}
            used_balance = {}
            
            for currency, amounts in balance.items():
                if currency not in ['info', 'free', 'used', 'total']:
                    if amounts['total'] > 0:
                        total_balance[currency] = amounts['total']
                        free_balance[currency] = amounts['free']
                        used_balance[currency] = amounts['used']
            
            return {
                "total_balance": total_balance,
                "free_balance": free_balance,
                "used_balance": used_balance,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get account balance: {str(e)}")

    def place_order(self,
                   symbol: str,
                   side: str,  # 'buy' or 'sell'
                   order_type: str,  # 'market', 'limit', 'stop_market', etc.
                   amount: float,
                   price: float = None,
                   stop_price: float = None,
                   params: Dict = None) -> Dict[str, Any]:
        """Place a trading order."""
        try:
            if not self.config.api_key:
                return {"error": "API credentials required for trading operations"}
            
            if order_type not in self.ORDER_TYPES:
                raise ValueError(f"Invalid order type: {order_type}. Valid types: {self.ORDER_TYPES}")
            
            if side not in ['buy', 'sell']:
                raise ValueError("Side must be 'buy' or 'sell'")
            
            # Prepare order parameters
            order_params = params or {}
            
            if order_type == 'market':
                order = self.exchange.create_market_order(symbol, side, amount, None, None, order_params)
            elif order_type == 'limit':
                if not price:
                    raise ValueError("Price required for limit orders")
                order = self.exchange.create_limit_order(symbol, side, amount, price, None, order_params)
            elif order_type in ['stop_market', 'stop_limit']:
                if not stop_price:
                    raise ValueError("Stop price required for stop orders")
                order_params['stopPrice'] = stop_price
                if order_type == 'stop_limit' and not price:
                    raise ValueError("Price required for stop limit orders")
                order = self.exchange.create_order(symbol, order_type, side, amount, price, None, None, order_params)
            else:
                order = self.exchange.create_order(symbol, order_type, side, amount, price, None, None, order_params)
            
            return {
                "order_id": order['id'],
                "symbol": order['symbol'],
                "side": order['side'],
                "type": order['type'],
                "amount": order['amount'],
                "price": order.get('price'),
                "status": order['status'],
                "timestamp": order['timestamp'],
                "info": order.get('info', {})
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to place order: {str(e)}")

    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Cancel an existing order."""
        try:
            if not self.config.api_key:
                return {"error": "API credentials required for trading operations"}
            
            result = self.exchange.cancel_order(order_id, symbol)
            return {
                "order_id": result['id'],
                "symbol": result['symbol'],
                "status": result['status'],
                "timestamp": result['timestamp']
            }
        except Exception as e:
            raise RuntimeError(f"Failed to cancel order {order_id}: {str(e)}")

    def get_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Get all open orders."""
        try:
            if not self.config.api_key:
                return {"error": "API credentials required for trading operations"}
            
            orders = self.exchange.fetch_open_orders(symbol)
            
            return [
                {
                    "order_id": order['id'],
                    "symbol": order['symbol'],
                    "side": order['side'],
                    "type": order['type'],
                    "amount": order['amount'],
                    "price": order.get('price'),
                    "status": order['status'],
                    "timestamp": order['timestamp']
                }
                for order in orders
            ]
        except Exception as e:
            raise RuntimeError(f"Failed to get open orders: {str(e)}")

    def get_trade_history(self, symbol: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trading history."""
        try:
            if not self.config.api_key:
                return {"error": "API credentials required for trading operations"}
            
            trades = self.exchange.fetch_my_trades(symbol, None, limit)
            
            return [
                {
                    "trade_id": trade['id'],
                    "order_id": trade['order'],
                    "symbol": trade['symbol'],
                    "side": trade['side'],
                    "amount": trade['amount'],
                    "price": trade['price'],
                    "cost": trade['cost'],
                    "fee": trade['fee'],
                    "timestamp": trade['timestamp']
                }
                for trade in trades
            ]
        except Exception as e:
            raise RuntimeError(f"Failed to get trade history: {str(e)}")

    def get_market_summary(self) -> Dict[str, Any]:
        """Get comprehensive market overview."""
        try:
            # Get market statistics
            tickers = self.exchange.fetch_tickers()
            
            # Calculate market metrics
            total_volume = sum([ticker['quoteVolume'] for ticker in tickers.values() if ticker['quoteVolume']])
            gainers = sorted([t for t in tickers.values() if t['percentage'] and t['percentage'] > 0], 
                           key=lambda x: x['percentage'], reverse=True)[:10]
            losers = sorted([t for t in tickers.values() if t['percentage'] and t['percentage'] < 0], 
                          key=lambda x: x['percentage'])[:10]
            
            return {
                "total_markets": len(tickers),
                "total_volume_24h": total_volume,
                "top_gainers": [
                    {
                        "symbol": t['symbol'],
                        "change_percent": t['percentage'],
                        "price": t['last'],
                        "volume": t['baseVolume']
                    }
                    for t in gainers
                ],
                "top_losers": [
                    {
                        "symbol": t['symbol'],
                        "change_percent": t['percentage'],
                        "price": t['last'],
                        "volume": t['baseVolume']
                    }
                    for t in losers
                ],
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get market summary: {str(e)}")

    def get_funding_rates(self, symbol: str = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get futures funding rates."""
        try:
            if symbol:
                funding = self.exchange.fetch_funding_rate(symbol)
                return {
                    "symbol": funding['symbol'],
                    "funding_rate": funding['fundingRate'],
                    "next_funding_time": funding['fundingDatetime'],
                    "timestamp": funding['timestamp']
                }
            else:
                funding_rates = self.exchange.fetch_funding_rates()
                return [
                    {
                        "symbol": rate['symbol'],
                        "funding_rate": rate['fundingRate'],
                        "next_funding_time": rate['fundingDatetime']
                    }
                    for rate in funding_rates.values()
                    if rate['fundingRate'] is not None
                ]
        except Exception as e:
            raise RuntimeError(f"Failed to get funding rates: {str(e)}")

    def get_perpetual_positions(self) -> List[Dict[str, Any]]:
        """Get perpetual futures positions."""
        try:
            if not self.config.api_key:
                return {"error": "API credentials required for position data"}
            
            positions = self.exchange.fetch_positions()
            active_positions = [pos for pos in positions if pos['contracts'] != 0]
            
            return [
                {
                    "symbol": pos['symbol'],
                    "side": pos['side'],
                    "size": pos['contracts'],
                    "notional": pos['notional'],
                    "entry_price": pos['entryPrice'],
                    "mark_price": pos['markPrice'],
                    "unrealized_pnl": pos['unrealizedPnl'],
                    "percentage": pos['percentage'],
                    "margin": pos['initialMargin']
                }
                for pos in active_positions
            ]
        except Exception as e:
            raise RuntimeError(f"Failed to get positions: {str(e)}")

    def calculate_technical_indicators(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> Dict[str, Any]:
        """Calculate comprehensive technical analysis indicators."""
        try:
            df = self.fetch_ohlcv_data(symbol, timeframe, limit)
            
            if df.empty:
                raise ValueError(f"No data available for {symbol}")
            
            latest = df.iloc[-1]
            
            # Additional indicators
            stoch_k, stoch_d = self._calculate_stochastic(df['high'], df['low'], df['close'])
            atr = self._calculate_atr(df['high'], df['low'], df['close'])
            
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": latest.name.isoformat(),
                "price": latest['close'],
                "indicators": {
                    "sma_20": latest['sma_20'],
                    "sma_50": latest['sma_50'],
                    "ema_12": latest['ema_12'],
                    "ema_26": latest['ema_26'],
                    "rsi": latest['rsi'],
                    "macd": latest['macd'],
                    "macd_signal": latest['macd_signal'],
                    "macd_histogram": latest['macd_histogram'],
                    "bb_upper": latest['bb_upper'],
                    "bb_middle": latest['bb_middle'],
                    "bb_lower": latest['bb_lower'],
                    "stoch_k": stoch_k.iloc[-1] if len(stoch_k) > 0 else None,
                    "stoch_d": stoch_d.iloc[-1] if len(stoch_d) > 0 else None,
                    "atr": atr.iloc[-1] if len(atr) > 0 else None,
                    "volatility": latest['volatility']
                },
                "signals": {
                    "trend": "bullish" if latest['sma_20'] > latest['sma_50'] else "bearish",
                    "rsi_signal": "overbought" if latest['rsi'] > 70 else "oversold" if latest['rsi'] < 30 else "neutral",
                    "bb_signal": "upper" if latest['close'] > latest['bb_upper'] else "lower" if latest['close'] < latest['bb_lower'] else "middle",
                    "macd_signal": "bullish" if latest['macd'] > latest['macd_signal'] else "bearish"
                }
            }
        except Exception as e:
            raise RuntimeError(f"Failed to calculate technical indicators: {str(e)}")

    def perform_risk_analysis(self) -> Dict[str, Any]:
        """Analyze portfolio risk metrics."""
        try:
            if not self.config.api_key:
                return {"error": "API credentials required for risk analysis"}
            
            balance = self.get_account_balance()
            positions = self.get_perpetual_positions()
            
            if isinstance(positions, dict) and "error" in positions:
                positions = []
            
            # Calculate portfolio metrics
            total_equity = sum(balance.get('total_balance', {}).values())
            total_margin = sum([pos.get('margin', 0) for pos in positions])
            total_unrealized_pnl = sum([pos.get('unrealized_pnl', 0) for pos in positions])
            
            # Risk metrics
            margin_ratio = (total_margin / total_equity) * 100 if total_equity > 0 else 0
            
            return {
                "portfolio_value": total_equity,
                "total_margin_used": total_margin,
                "margin_ratio": margin_ratio,
                "free_margin": total_equity - total_margin,
                "unrealized_pnl": total_unrealized_pnl,
                "pnl_percentage": (total_unrealized_pnl / total_equity) * 100 if total_equity > 0 else 0,
                "active_positions": len(positions),
                "risk_level": "high" if margin_ratio > 75 else "medium" if margin_ratio > 50 else "low",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            raise RuntimeError(f"Failed to perform risk analysis: {str(e)}")

    def get_top_gainers_losers(self, limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Get top performing cryptocurrencies."""
        try:
            tickers = self.exchange.fetch_tickers()
            
            # Filter and sort
            valid_tickers = [t for t in tickers.values() if t['percentage'] is not None and t['baseVolume'] and t['baseVolume'] > 1000]
            
            gainers = sorted(valid_tickers, key=lambda x: x['percentage'], reverse=True)[:limit]
            losers = sorted(valid_tickers, key=lambda x: x['percentage'])[:limit]
            
            return {
                "top_gainers": [
                    {
                        "symbol": t['symbol'],
                        "price": t['last'],
                        "change_percent": t['percentage'],
                        "volume_24h": t['baseVolume']
                    }
                    for t in gainers
                ],
                "top_losers": [
                    {
                        "symbol": t['symbol'],
                        "price": t['last'],
                        "change_percent": t['percentage'],
                        "volume_24h": t['baseVolume']
                    }
                    for t in losers
                ]
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get top gainers/losers: {str(e)}")

    def search_trading_pairs(self, base_currency: str = None, quote_currency: str = None) -> List[str]:
        """Search for available trading pairs."""
        try:
            markets = self.exchange.load_markets()
            symbols = list(markets.keys())
            
            if base_currency:
                symbols = [s for s in symbols if s.split('/')[0].upper() == base_currency.upper()]
            
            if quote_currency:
                symbols = [s for s in symbols if s.split('/')[1].upper() == quote_currency.upper()]
            
            return sorted(symbols)
        except Exception as e:
            raise RuntimeError(f"Failed to search trading pairs: {str(e)}")

    def get_24h_stats(self, symbol: str = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get 24-hour trading statistics."""
        try:
            if symbol:
                ticker = self.exchange.fetch_ticker(symbol)
                return {
                    "symbol": ticker['symbol'],
                    "open_24h": ticker['open'],
                    "high_24h": ticker['high'],
                    "low_24h": ticker['low'],
                    "close_price": ticker['last'],
                    "volume_24h": ticker['baseVolume'],
                    "quote_volume_24h": ticker['quoteVolume'],
                    "change_24h": ticker['change'],
                    "change_percent_24h": ticker['percentage'],
                    "vwap": ticker.get('vwap'),
                    "count": ticker.get('count'),
                    "timestamp": ticker['timestamp']
                }
            else:
                tickers = self.exchange.fetch_tickers()
                return [
                    {
                        "symbol": ticker['symbol'],
                        "high_24h": ticker['high'],
                        "low_24h": ticker['low'],
                        "volume_24h": ticker['baseVolume'],
                        "change_percent_24h": ticker['percentage'],
                        "last_price": ticker['last']
                    }
                    for ticker in tickers.values()
                    if ticker['baseVolume'] and ticker['baseVolume'] > 0
                ]
        except Exception as e:
            raise RuntimeError(f"Failed to get 24h stats: {str(e)}")

    # Technical Analysis Helper Methods
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2):
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, sma, lower

    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3):
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent

    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14):
        """Calculate Average True Range."""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = pd.Series(true_range).rolling(window=window).mean()
        return atr

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_fields = ['api_key', 'api_secret']
        for field in required_fields:
            if not hasattr(self.config, field):
                print(f"⚠️ Missing {field} - some features will be unavailable")

        if not hasattr(self.config, 'cache_enabled'):
            self.config.cache_enabled = True

        if not hasattr(self.config, 'sandbox'):
            self.config.sandbox = False

    def _init_analysis_chain(self):
        """Initialize analysis chains for advanced trading analysis."""
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
                        "tool": "get_ticker",
                        "parameters": {"symbol": "BTC/USDT"},
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
        """Process natural language trading requests."""
        if not getattr(self, "llm_enabled", True):
            # Default plan for testing
            plan = {
                "steps": [{
                    "tool": "get_ticker",
                    "parameters": {"symbol": "BTC/USDT"},
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
        print("=== Binance Execution Plan ===")
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
        """Execute generated trading strategy."""
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

    # Additional utility methods for advanced trading
    def calculate_position_size(self, 
                               symbol: str, 
                               risk_percent: float = 1.0, 
                               stop_loss_price: float = None,
                               account_balance: float = None) -> Dict[str, Any]:
        """Calculate optimal position size based on risk management."""
        try:
            if not account_balance:
                balance_info = self.get_account_balance()
                if isinstance(balance_info, dict) and "error" in balance_info:
                    raise ValueError("Cannot calculate position size without account balance")
                account_balance = sum(balance_info.get('total_balance', {}).values())
            
            ticker = self.get_ticker(symbol)
            current_price = ticker['last_price']
            
            risk_amount = account_balance * (risk_percent / 100)
            
            if stop_loss_price:
                price_diff = abs(current_price - stop_loss_price)
                position_size = risk_amount / price_diff
            else:
                # Use 2% price movement as default risk
                price_diff = current_price * 0.02
                position_size = risk_amount / price_diff
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "account_balance": account_balance,
                "risk_amount": risk_amount,
                "risk_percent": risk_percent,
                "recommended_position_size": position_size,
                "stop_loss_price": stop_loss_price,
                "potential_loss": risk_amount
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to calculate position size: {str(e)}")

    def get_market_sentiment(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Analyze market sentiment based on price movements and volume."""
        try:
            if not symbols:
                symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT"]
            
            sentiment_data = []
            for symbol in symbols:
                try:
                    ticker = self.get_ticker(symbol)
                    change_24h = ticker.get('change_percent_24h', 0)
                    volume_24h = ticker.get('volume_24h', 0)
                    
                    sentiment_data.append({
                        "symbol": symbol,
                        "change_24h": change_24h,
                        "volume_24h": volume_24h,
                        "sentiment": "bullish" if change_24h > 2 else "bearish" if change_24h < -2 else "neutral"
                    })
                except Exception:
                    continue
            
            bullish_count = sum(1 for d in sentiment_data if d['sentiment'] == 'bullish')
            bearish_count = sum(1 for d in sentiment_data if d['sentiment'] == 'bearish')
            neutral_count = sum(1 for d in sentiment_data if d['sentiment'] == 'neutral')
            
            total_symbols = len(sentiment_data)
            overall_sentiment = "bullish" if bullish_count > bearish_count else "bearish" if bearish_count > bullish_count else "neutral"
            
            return {
                "overall_sentiment": overall_sentiment,
                "sentiment_score": (bullish_count - bearish_count) / total_symbols if total_symbols > 0 else 0,
                "bullish_count": bullish_count,
                "bearish_count": bearish_count,
                "neutral_count": neutral_count,
                "analyzed_symbols": sentiment_data,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to analyze market sentiment: {str(e)}")

    def __del__(self):
        """Clean up thread pool executor."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)