"""
Shared Baseline Generator for Backtesting and Paper Trading

This module generates baseline equity curves (Buy & Hold, Index) for a given:
- Date range
- Symbol list
- Mode (backtest or paper)

Can be called by:
- Backtest script (historical data, mode="backtest")
- Paper trading service (live data, mode="paper")

Same logic, different contexts.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

from dashboard.backend.paths import CREDENTIALS_DIR
from dashboard.backend.domain.backtesting.constants import INITIAL_CAPITAL
from dashboard.backend.infrastructure.market_data.alpaca_bars import (
    MarketDataUnavailableError,
)

# NOTE: domain.leaderboard.strategies._common is imported lazily inside the
# methods that need it — the strategies package imports this module back
# (buy_hold et al.), so a top-level import here is a circular import.

try:
    import pandas as pd
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "pandas"])
    import pandas as pd


class BaselineGenerator:
    """Generates baseline equity curves from real historical data."""
    
    def __init__(self):
        """Initialize without touching credentials or the network."""
        self.api_key = None
        self.secret_key = None
        self.headers = None

    def _ensure_credentials(self):
        """Load Alpaca credentials only for methods that fetch remote bars."""
        if self.api_key and self.secret_key:
            return
        self._load_credentials()
    
    def _load_credentials(self):
        """Load Alpaca credentials from environment variables or file."""
        # Try environment variables first (for Render, Docker, etc.)
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if self.api_key and self.secret_key:
            print("✅ Loaded Alpaca credentials from environment variables")
            self.headers = {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key,
            }
            return
        
        # Fall back to credentials file (for local development)
        creds_path = CREDENTIALS_DIR / "alpaca.json"
        try:
            with open(creds_path, 'r') as f:
                creds = json.load(f)
                self.api_key = creds.get('api_key')
                self.secret_key = creds.get('secret_key')
                
                if not self.api_key or not self.secret_key:
                    raise ValueError("Missing Alpaca credentials in file")
                
                print(f"✅ Loaded Alpaca credentials from {creds_path}")
                self.headers = {
                    "APCA-API-KEY-ID": self.api_key,
                    "APCA-API-SECRET-KEY": self.secret_key,
                }
        except Exception as e:
            print(f"❌ Failed to load credentials from file: {e}")
            print("   Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
            # A plain exception, not sys.exit(1): baselines are generated inside
            # the server (paper init, leaderboard strategies, backtest finalize)
            # where SystemExit would evade `except Exception` (the B0 class).
            raise MarketDataUnavailableError(
                "Alpaca credentials not found (set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY, or provide credentials/alpaca.json)"
            ) from e
    
    def _fetch_bars_for_symbol(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch REAL historical bars from Alpaca API.
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with OHLCV data, indexed by timestamp
        """
        self._ensure_credentials()

        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
        except ImportError as e:
            print("❌ alpaca-py not installed. Install with: pip install alpaca-py")
            raise MarketDataUnavailableError(
                "alpaca-py is not installed (pip install alpaca-py)"
            ) from e
        
        try:
            client = StockHistoricalDataClient(self.api_key, self.secret_key)
            
            request = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Hour,
                start=start_date,
                end=end_date,
            )
            
            bars_data = client.get_stock_bars(request)
            
            if symbol in bars_data.df.index.get_level_values(0):
                df = bars_data.df.loc[symbol].copy()
                return df
            else:
                return None
        
        except Exception as e:
            print(f"⚠️ Error fetching {symbol}: {e}")
            return None
    
    def generate_buyhold_baseline(
        self, 
        bars_by_symbol: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        initial_capital: float = INITIAL_CAPITAL,
        symbols_to_buy: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Generate Buy & Hold baseline curve.
        
        Strategy: Buy equal amounts of specified symbols at start, hold until end.
        
        Args:
            bars_by_symbol: Dict of {symbol: DataFrame with OHLCV}
            start_date: Start date string
            end_date: End date string
            initial_capital: Initial portfolio value
            symbols_to_buy: List of symbols to buy (default: all in bars_by_symbol)
        
        Returns:
            List of equity points: [{timestamp, equity, cash, positions_value}, ...]
        """
        if not bars_by_symbol:
            return []
        
        # Filter to only requested symbols
        if symbols_to_buy is None:
            bars_subset = bars_by_symbol
        else:
            bars_subset = {k: v for k, v in bars_by_symbol.items() if k in symbols_to_buy}
        
        if not bars_subset:
            return []
        
        # Get all timestamps across all symbols
        all_timestamps = set()
        for df in bars_subset.values():
            all_timestamps.update(df.index)
        all_timestamps = sorted(all_timestamps)
        
        if not all_timestamps:
            return []
        
        # Filter: only keep regular market hours (9:30 AM - 4:00 PM ET)
        import pytz
        et_tz = pytz.timezone('US/Eastern')
        market_hours_only = []
        
        for ts in all_timestamps:
            ts_et = ts.astimezone(et_tz)
            hour = ts_et.hour
            minute = ts_et.minute
            
            # Market hours: 9:30 AM through 4:00 PM ET
            is_market_hours = (hour > 9 and hour < 16) or \
                             (hour == 9 and minute >= 30) or \
                             (hour == 16 and minute == 0)
            
            if is_market_hours:
                market_hours_only.append(ts)
        
        all_timestamps = market_hours_only
        from dashboard.backend.domain.leaderboard.strategies._common import (
            timestamps_in_contest,
        )
        all_timestamps = timestamps_in_contest(all_timestamps, start_date, end_date)

        if not all_timestamps:
            return []

        first_ts = all_timestamps[0]

        # Buy equal amounts of available stocks
        positions = {}
        cash = initial_capital
        num_symbols = len(bars_subset)
        
        print(f"\n   📋 Baseline buying {num_symbols} stocks equally:")
        print(f"      Allocation per stock: ${initial_capital / num_symbols:,.0f}")
        
        for symbol, df in bars_subset.items():
            if first_ts not in df.index:
                continue
            
            price = df.loc[first_ts, "close"]
            allocation = initial_capital / num_symbols
            shares = int(allocation / price)
            
            if shares > 0:
                positions[symbol] = shares
                cash -= shares * price
        
        print(f"      Stocks bought: {len(positions)} ({', '.join(sorted(positions.keys())[:10])}{'...' if len(positions) > 10 else ''})")
        print(f"      Total invested: ${initial_capital - cash:,.0f}")
        print(f"      Cash remaining: ${cash:,.0f}")
        
        # Build forward-filled price cache for smooth equity curve
        price_cache = {}
        for symbol, df in bars_subset.items():
            if symbol not in positions:
                continue
            
            price_cache[symbol] = {}
            last_price = df.loc[first_ts, "close"] if first_ts in df.index else None
            
            if last_price is None:
                continue
            
            for timestamp in all_timestamps:
                if timestamp in df.index:
                    last_price = df.loc[timestamp, "close"]
                # Forward-fill missing data
                price_cache[symbol][timestamp] = last_price
        
        # Calculate equity at each timestamp
        equity_curve = []
        for timestamp in all_timestamps:
            positions_value = 0
            
            for symbol, shares in positions.items():
                if symbol in price_cache and timestamp in price_cache[symbol]:
                    positions_value += shares * price_cache[symbol][timestamp]
            
            total_equity = cash + positions_value
            
            equity_curve.append({
                "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                "equity": round(total_equity, 2),
                "cash": round(cash, 2),
                "positions_value": round(positions_value, 2),
                "daily_return": 0
            })
        
        return equity_curve
    
    def generate_index_baseline(
        self,
        bars_by_symbol: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        initial_capital: float = INITIAL_CAPITAL,
        symbols_to_track: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Generate Index baseline curve (equal-weight index).
        
        Strategy: Equal-weight portfolio of specified symbols, rebalanced daily.
        
        Args:
            bars_by_symbol: Dict of {symbol: DataFrame with OHLCV}
            start_date: Start date string
            end_date: End date string
            initial_capital: Initial portfolio value
            symbols_to_track: List of symbols to track (default: all in bars_by_symbol)
        
        Returns:
            List of equity points: [{timestamp, equity, cash, positions_value}, ...]
        """
        if not bars_by_symbol:
            return []
        
        # Filter to only requested symbols
        if symbols_to_track is None:
            bars_subset = bars_by_symbol
        else:
            bars_subset = {k: v for k, v in bars_by_symbol.items() if k in symbols_to_track}
        
        # Get all timestamps
        all_timestamps = set()
        for df in bars_subset.values():
            all_timestamps.update(df.index)
        all_timestamps = sorted(all_timestamps)
        
        if not all_timestamps:
            return []
        
        # Filter: only keep regular market hours (9:30 AM - 4:00 PM ET)
        import pytz
        et_tz = pytz.timezone('US/Eastern')
        market_hours_only = []
        
        for ts in all_timestamps:
            ts_et = ts.astimezone(et_tz)
            hour = ts_et.hour
            minute = ts_et.minute
            
            # Market hours: 9:30 AM through 4:00 PM ET
            is_market_hours = (hour > 9 and hour < 16) or \
                             (hour == 9 and minute >= 30) or \
                             (hour == 16 and minute == 0)
            
            if is_market_hours:
                market_hours_only.append(ts)
        
        all_timestamps = market_hours_only
        from dashboard.backend.domain.leaderboard.strategies._common import (
            timestamps_in_contest,
        )
        all_timestamps = timestamps_in_contest(all_timestamps, start_date, end_date)

        if not all_timestamps:
            return []

        first_ts = all_timestamps[0]

        # Get initial prices
        initial_prices = {}
        for symbol, df in bars_subset.items():
            if first_ts in df.index:
                initial_prices[symbol] = df.loc[first_ts, "close"]
        
        if not initial_prices:
            return []
        
        # Build forward-filled price cache
        price_cache = {}
        for symbol, df in bars_subset.items():
            if symbol not in initial_prices:
                continue
            
            price_cache[symbol] = {}
            last_price = df.loc[first_ts, "close"]
            
            for timestamp in all_timestamps:
                if timestamp in df.index:
                    last_price = df.loc[timestamp, "close"]
                # Forward-fill
                price_cache[symbol][timestamp] = last_price
        
        # Calculate index equity at each timestamp
        equity_curve = []
        num_symbols = len(initial_prices)
        
        print(f"\n   📋 Index baseline tracking {num_symbols} stocks equally (equal-weight):")
        print(f"      Stocks tracked: {', '.join(sorted(initial_prices.keys())[:10])}{'...' if len(initial_prices) > 10 else ''}")
        print(f"      Initial capital: ${initial_capital:,.0f}")
        print(f"      Portfolio: 100% invested in {num_symbols}-stock equal-weight index")
        print()
        
        for timestamp in all_timestamps:
            index_return = 0
            valid_count = 0
            
            for symbol in initial_prices:
                if symbol in price_cache and timestamp in price_cache[symbol]:
                    current_price = price_cache[symbol][timestamp]
                    symbol_return = (current_price / initial_prices[symbol]) - 1
                    index_return += symbol_return
                    valid_count += 1
            
            if valid_count > 0:
                avg_return = index_return / valid_count
                total_equity = initial_capital * (1 + avg_return)
                positions_value = total_equity  # All in positions, no cash
            else:
                total_equity = initial_capital
                positions_value = 0
            
            equity_curve.append({
                "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                "equity": round(total_equity, 2),
                "cash": 0,
                "positions_value": round(positions_value, 2),
                "daily_return": 0
            })
        
        return equity_curve


def generate_baselines(
    bars_by_symbol: Dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
    initial_capital: float = INITIAL_CAPITAL,
    symbols_list: Optional[List[str]] = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate both baselines (Buy & Hold, Index).
    
    Args:
        bars_by_symbol: Dict of {symbol: DataFrame with OHLCV}
        start_date: Start date string
        end_date: End date string
        initial_capital: Initial portfolio value
        symbols_list: List of symbols to use (default: all in bars_by_symbol)
    
    Returns:
        Tuple of (buyhold_curve, index_curve)
    """
    generator = BaselineGenerator()
    
    buyhold_curve = generator.generate_buyhold_baseline(
        bars_by_symbol, start_date, end_date, initial_capital, symbols_list
    )
    
    index_curve = generator.generate_index_baseline(
        bars_by_symbol, start_date, end_date, initial_capital, symbols_list
    )
    
    return buyhold_curve, index_curve
