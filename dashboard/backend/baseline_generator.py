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
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, time

from dashboard.backend.paths import CREDENTIALS_DIR
from dashboard.backend.domain.backtesting.constants import INITIAL_CAPITAL
from dashboard.backend.domain.backtesting.currency import CurrencyContext
from dashboard.backend.domain.trading.execution import calculate_transaction_costs
from dashboard.backend.infrastructure.market_data.alpaca_bars import (
    MarketDataUnavailableError,
)

# The baseline calculations operate on already-normalized bars and keep their
# market-session filtering local so A-share timestamps are not interpreted as
# US/Eastern dates.

try:
    import pandas as pd
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "pandas"])
    import pandas as pd


def _market_hours_only(timestamps, market_timezone: str):
    """Keep regular sessions in the timezone belonging to the market profile."""
    import pytz

    market_tz = pytz.timezone(market_timezone)
    kept = []
    for timestamp in timestamps:
        local = timestamp.astimezone(market_tz)
        local_time = local.time()
        if market_timezone == "Asia/Shanghai":
            in_session = (
                time(9, 30) <= local_time <= time(11, 30)
                or time(13, 0) <= local_time <= time(15, 0)
            )
        else:
            in_session = (
                (local.hour > 9 and local.hour < 16)
                or (local.hour == 9 and local.minute >= 30)
                or (local.hour == 16 and local.minute == 0)
            )
        if in_session:
            kept.append(timestamp)
    return kept


def _timestamps_in_window(timestamps, start_date: str, end_date: str, market_timezone: str):
    import pytz

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    market_tz = pytz.timezone(market_timezone)
    return [
        timestamp
        for timestamp in timestamps
        if start <= timestamp.astimezone(market_tz).date() <= end
    ]


def _equity_point(
    timestamp,
    equity: float,
    cash: float,
    positions_value: float,
    currency_context: CurrencyContext | None,
) -> Dict:
    record = {
        "timestamp": timestamp,
        "equity": equity,
        "cash": cash,
        "positions_value": positions_value,
        "daily_return": 0,
    }
    if currency_context is not None:
        record = currency_context.reporting_equity_record(record)
    record["timestamp"] = (
        timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp)
    )
    for field in (
        "equity",
        "cash",
        "positions_value",
        "native_equity",
        "native_cash",
        "native_positions_value",
    ):
        if field in record:
            record[field] = round(float(record[field]), 2)
    return record


# Backstop for the lot top-up sweep below. Reached only if a lot were free,
# which the cost model forbids; it exists so a future zero-price bar cannot
# spin the loop forever.
_MAX_TOPUP_LOTS = 100_000


def _buy_order_cash_out(
    price: float,
    shares: int,
    transaction_cost_profile,
) -> float:
    """Cash a single buy order of ``shares`` removes, fees included.

    Always costs the WHOLE order rather than an increment, because the A-share
    commission carries a per-order minimum (¥5): ten lots priced one at a time
    would be charged that floor ten times. Callers that grow a position take
    the difference between two whole-order quotes instead.
    """
    if shares <= 0:
        return 0.0
    costs = calculate_transaction_costs(
        side="buy",
        reference_price=price,
        shares=shares,
        transaction_cost_profile=transaction_cost_profile,
    )
    return -costs["net_cash_impact"]


def _plan_buyhold_allocation(
    prices: Dict[str, float],
    capital: float,
    num_symbols: int,
    lot_size: int,
    transaction_cost_profile,
) -> Dict[str, int]:
    """Share counts for an equal-weight buy & hold sleeve.

    Two passes, because equal weight and whole lots do not compose. An equal
    slice of a small account is routinely worth less than one 100-share A-share
    lot, so flooring each slice on its own strands most of the capital in cash
    and turns the benchmark every agent is scored against into a flat line.

    1. Equal weight: each symbol gets its own slice, floored to whole lots and
       capped by cash actually left. A symbol never borrows from its neighbours'
       slices, so composition does not depend on iteration order.
    2. Top-up: whatever pass 1 could not place is swept into further lots,
       poorest symbol first, so the sleeve stays invested and stays as close to
       equal weight as whole lots allow.
    """
    if not prices or capital <= 0 or num_symbols <= 0:
        return {}

    lot_size = max(1, int(lot_size or 1))
    # Denominator is the full sleeve, not the priced subset: a symbol with no
    # bar at the open forfeits its slice rather than redistributing it, which
    # is what the equal-weight baseline has always done.
    allocation = capital / num_symbols
    planned: Dict[str, int] = {}
    spent = 0.0

    for symbol, price in prices.items():
        # Rejects NaN as well as zero/negative: a bad bar must not divide, and
        # a free lot would spin the top-up sweep below.
        if not price > 0:
            continue
        shares = int(allocation / price)
        shares -= shares % lot_size
        # Fees push a naively-affordable order over its slice; shrink a lot at a
        # time until it fits both the slice and the cash on hand.
        budget = min(allocation, capital - spent)
        while shares > 0 and _buy_order_cash_out(
            price, shares, transaction_cost_profile
        ) > budget:
            shares -= lot_size
        if shares > 0:
            planned[symbol] = shares
            spent += _buy_order_cash_out(price, shares, transaction_cost_profile)

    if lot_size <= 1:
        # Without lot rounding pass 1 already places the whole slice, so a
        # top-up would only push the benchmark past equal weight.
        return planned

    for _ in range(_MAX_TOPUP_LOTS):
        remaining = capital - spent
        # Poorest-first keeps the sleeve balanced and gets an unrepresented
        # symbol its first lot before any symbol gets a second one.
        candidates = sorted(
            (symbol for symbol, price in prices.items() if price > 0),
            key=lambda s: (planned.get(s, 0) * float(prices[s]), float(prices[s]), s),
        )
        for symbol in candidates:
            price = prices[symbol]
            held = planned.get(symbol, 0)
            delta = _buy_order_cash_out(
                price, held + lot_size, transaction_cost_profile
            ) - _buy_order_cash_out(price, held, transaction_cost_profile)
            if delta <= remaining:
                planned[symbol] = held + lot_size
                spent += delta
                break
        else:
            break

    return planned


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
        symbols_to_buy: Optional[List[str]] = None,
        market_timezone: str = "US/Eastern",
        currency_context: CurrencyContext | None = None,
        transaction_cost_profile=None,
        transaction_cost_totals: Optional[Dict[str, float]] = None,
        lot_size: int = 1,
        allocation_summary: Optional[Dict[str, Any]] = None,
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
            transaction_cost_profile: Charges the sleeve real fees when set.
            transaction_cost_totals: Out-dict; fee totals are added into it.
            lot_size: Board lot the market enforces (100 for A-shares, 1
                elsewhere). Comes from ``MarketProfile.lot_size`` — it is a
                separate market rule from ``transaction_cost_profile`` and must
                not be inferred from it.
            allocation_summary: Out-dict recording how much of the sleeve
                actually filled, so a partly-placed benchmark is visible in run
                metadata instead of silently reading as a real flat curve.

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
        
        all_timestamps = _market_hours_only(all_timestamps, market_timezone)
        all_timestamps = _timestamps_in_window(
            all_timestamps, start_date, end_date, market_timezone
        )

        if not all_timestamps:
            return []

        first_ts = all_timestamps[0]

        # Buy equal amounts of available stocks
        native_initial_capital = (
            currency_context.to_native(initial_capital, first_ts)
            if currency_context is not None
            else initial_capital
        )
        native_symbol = (
            "¥"
            if currency_context is not None
            and currency_context.native_currency == "CNY"
            else "$"
        )
        cash = native_initial_capital
        num_symbols = len(bars_subset)
        
        print(f"\n   📋 Baseline buying {num_symbols} stocks equally:")
        print(
            f"      Allocation per stock: {native_symbol}"
            f"{native_initial_capital / num_symbols:,.0f}"
        )
        
        open_prices = {
            symbol: float(df.loc[first_ts, "close"])
            for symbol, df in bars_subset.items()
            if first_ts in df.index
        }
        positions = _plan_buyhold_allocation(
            open_prices,
            native_initial_capital,
            num_symbols,
            lot_size,
            transaction_cost_profile,
        )

        # Cost the plan once per symbol. The order-level pass matters for the
        # A-share ¥5 minimum commission: it is charged per submitted order, so
        # a symbol's whole sleeve must be priced as one order.
        for symbol, shares in positions.items():
            costs = calculate_transaction_costs(
                side="buy",
                reference_price=open_prices[symbol],
                shares=shares,
                transaction_cost_profile=transaction_cost_profile,
            )
            cash += costs["net_cash_impact"]
            if transaction_cost_totals is not None:
                for field in (
                    "gross_value",
                    "slippage_amount",
                    "commission",
                    "stamp_duty",
                    "transfer_fee",
                    "total_fees",
                ):
                    transaction_cost_totals[field] = (
                        transaction_cost_totals.get(field, 0.0) + costs[field]
                    )

        invested = native_initial_capital - cash
        effective_lot = max(1, int(lot_size or 1))
        if allocation_summary is not None:
            # Absent is not the same as broken: without these a benchmark that
            # placed nothing looks exactly like one that correctly held cash.
            allocation_summary.update(
                {
                    "symbols_requested": num_symbols,
                    "symbols_priced": len(open_prices),
                    "symbols_bought": len(positions),
                    "symbols_skipped": num_symbols - len(positions),
                    "lot_size": effective_lot,
                    "invested_ratio": (
                        invested / native_initial_capital
                        if native_initial_capital
                        else 0.0
                    ),
                }
            )
        # Only an affordability shortfall is worth shouting about; a symbol with
        # no bar at the open is an ordinary data gap the loop above already
        # skipped, and the counts above still record it.
        if len(positions) < len(open_prices):
            print(
                f"      ⚠️  {len(open_prices) - len(positions)} of "
                f"{len(open_prices)} priced symbols bought nothing — an equal "
                f"slice is worth less than one {effective_lot}-share lot at "
                f"this capital"
            )

        print(f"      Stocks bought: {len(positions)} ({', '.join(sorted(positions.keys())[:10])}{'...' if len(positions) > 10 else ''})")
        print(f"      Total invested: {native_symbol}{native_initial_capital - cash:,.0f}")
        print(f"      Cash remaining: {native_symbol}{cash:,.0f}")
        
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
            
            equity_curve.append(
                _equity_point(
                    timestamp,
                    total_equity,
                    cash,
                    positions_value,
                    currency_context,
                )
            )
        
        return equity_curve
    
    def generate_index_baseline(
        self,
        bars_by_symbol: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        initial_capital: float = INITIAL_CAPITAL,
        symbols_to_track: Optional[List[str]] = None,
        market_timezone: str = "US/Eastern",
        currency_context: CurrencyContext | None = None,
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
        
        all_timestamps = _market_hours_only(all_timestamps, market_timezone)
        all_timestamps = _timestamps_in_window(
            all_timestamps, start_date, end_date, market_timezone
        )

        if not all_timestamps:
            return []

        first_ts = all_timestamps[0]
        native_initial_capital = (
            currency_context.to_native(initial_capital, first_ts)
            if currency_context is not None
            else initial_capital
        )
        native_symbol = (
            "¥"
            if currency_context is not None
            and currency_context.native_currency == "CNY"
            else "$"
        )

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
        print(f"      Initial capital: {native_symbol}{native_initial_capital:,.0f}")
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
                total_equity = native_initial_capital * (1 + avg_return)
                positions_value = total_equity  # All in positions, no cash
            else:
                total_equity = native_initial_capital
                positions_value = 0
            
            equity_curve.append(
                _equity_point(
                    timestamp,
                    total_equity,
                    0,
                    positions_value,
                    currency_context,
                )
            )
        
        return equity_curve


def generate_baselines(
    bars_by_symbol: Dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
    initial_capital: float = INITIAL_CAPITAL,
    symbols_list: Optional[List[str]] = None,
    market_timezone: str = "US/Eastern",
    currency_context: CurrencyContext | None = None,
    transaction_cost_profile=None,
    transaction_cost_totals: Optional[Dict[str, float]] = None,
    lot_size: int = 1,
    allocation_summary: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate both baselines (Buy & Hold, Index).

    Args:
        bars_by_symbol: Dict of {symbol: DataFrame with OHLCV}
        start_date: Start date string
        end_date: End date string
        initial_capital: Initial portfolio value
        symbols_list: List of symbols to use (default: all in bars_by_symbol)
        lot_size: Board lot the market enforces; see
            ``BaselineGenerator.generate_buyhold_baseline``.
        allocation_summary: Out-dict describing how much of the buy & hold
            sleeve actually filled.

    Returns:
        Tuple of (buyhold_curve, index_curve)
    """
    generator = BaselineGenerator()

    buyhold_curve = generator.generate_buyhold_baseline(
        bars_by_symbol,
        start_date,
        end_date,
        initial_capital,
        symbols_list,
        market_timezone,
        currency_context,
        transaction_cost_profile,
        transaction_cost_totals,
        lot_size,
        allocation_summary,
    )
    
    index_curve = generator.generate_index_baseline(
        bars_by_symbol,
        start_date,
        end_date,
        initial_capital,
        symbols_list,
        market_timezone,
        currency_context,
    )
    
    return buyhold_curve, index_curve
