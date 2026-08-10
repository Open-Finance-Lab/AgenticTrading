#!/usr/bin/env python3
"""
Backtest Real Trading Agent vs. Baselines

Compares real trading strategies using actual historical price data from Alpaca API:
1. Agent (AI trading strategy with momentum + mean reversion)
2. Buy & Hold (simple baseline - buy and hold all symbols)
3. DJIA Index (market index tracking)

Results are saved to SQLite database for dashboard visualization.

Real Data:
- Uses Alpaca API to fetch actual historical bar data (OHLCV)
- Requires valid Alpaca credentials in credentials/alpaca.json
- Trades on real price movements
- Calculates real Sharpe ratio, max drawdown, returns

Usage:
    python3 backtest.py --start 2024-01-01 --end 2024-12-31
    python3 backtest.py --symbols AAPL MSFT NVDA --start 2026-01-01 --end 2026-04-22
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import argparse
import uuid
import requests

# Try to import numpy for calculations
try:
    import numpy as np
except ImportError:
    print("Installing numpy...")
    subprocess.check_call(["pip", "install", "numpy"])
    import numpy as np

from backtest_engine import BacktestEngine

# Direct-execution bootstrap: make the repo root importable so canonical
# `dashboard.backend.*` imports resolve (no-op when run as part of the package).
from _bootstrap import ensure_repo_root

ensure_repo_root()
from dashboard.backend.paths import CREDENTIALS_DIR, DATA_DIR
from dashboard.backend.database import db
from dashboard.backend.infrastructure.llm.validator import DJIA_30, TOP_10_STOCKS

# ============================================================================
# Configuration
# ============================================================================

# Canonical Dow-30 (single source: validator.DJIA_30, guarded by
# tests/test_djia30_universe.py), kept under the historical local name.
DJIA_SYMBOLS = list(DJIA_30)

SCRIPT_DIR = Path(__file__).parent
ALPACA_CLI = None  # Not used - we'll use yfinance instead

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = DATA_DIR / "backtest_results.json"


# ============================================================================
# Data Fetching (Real Historical Prices from Alpaca)
# ============================================================================

class PriceFetcher:
    """Fetches real historical price data from Alpaca API."""
    
    def __init__(self):
        self.cache: Dict[str, Dict[str, float]] = {}
        self._load_credentials()
    
    def _load_credentials(self):
        """Load Alpaca credentials from credentials/alpaca.json"""
        creds_path = CREDENTIALS_DIR / "alpaca.json"
        try:
            with open(creds_path, 'r') as f:
                creds = json.load(f)
                self.api_key = creds.get('api_key')
                self.secret_key = creds.get('secret_key')
                self.base_url = creds.get('base_url', 'https://data.alpaca.markets')
                
                if not self.api_key or not self.secret_key:
                    raise ValueError("Missing API credentials")
                
                self.headers = {
                    "APCA-API-KEY-ID": self.api_key,
                    "APCA-API-SECRET-KEY": self.secret_key,
                }
                print("✅ Alpaca credentials loaded")
        except Exception as e:
            print(f"❌ Failed to load Alpaca credentials: {e}")
            self.api_key = None
            self.secret_key = None
    
    def get_historical(self, symbol: str, start_date: str, end_date: str) -> Dict[str, float]:
        """
        Fetch real historical daily prices from Alpaca API.
        Format: {"2024-01-01": 150.25, "2024-01-02": 151.50, ...}
        Returns closing prices for each trading day.
        """
        if not self.api_key:
            print(f"  ⚠️ No Alpaca credentials, skipping {symbol}")
            return {}
        
        try:
            print(f"  Fetching {symbol} from {start_date} to {end_date}...")
            
            url = f"{self.base_url}/v2/stocks/{symbol}/bars"
            params = {
                "start": start_date,
                "end": end_date,
                "timeframe": "1D"
            }
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            
            if response.status_code != 200:
                print(f"    ❌ Alpaca error {response.status_code}: {response.text[:100]}")
                return {}
            
            bars = response.json().get("bars", [])
            if not bars:
                print(f"    ⚠️ No bars found for {symbol}")
                return {}
            
            # Convert to dict: date -> close price
            prices = {}
            for bar in bars:
                # Alpaca returns timestamp in ISO format
                date_str = bar['t'][:10]  # Extract YYYY-MM-DD
                prices[date_str] = float(bar['c'])  # 'c' is close price
            
            print(f"    ✅ Got {len(prices)} trading days for {symbol}")
            return prices
        
        except Exception as e:
            print(f"    ❌ Error fetching {symbol}: {e}")
            return {}


# ============================================================================
# Trading Strategies
# ============================================================================

class ClawdyStrategy:
    """
    Clawdy Trading Agent Strategy
    
    Simple momentum + mean reversion strategy:
    - Buy when price dips 3% below 20-day MA
    - Sell when price rises 5% above entry
    - Risk management: max 5% per symbol
    """
    
    def __init__(self, backtest_engine: BacktestEngine):
        self.engine = backtest_engine
        self.position_entries: Dict[str, float] = {}
        self.symbol_ma_20: Dict[str, List[float]] = {}
    
    def process_day(self, date: str, prices: Dict[str, float]):
        """Process trading signals for a day."""
        # Update 20-day moving averages
        for symbol in prices:
            if symbol not in self.symbol_ma_20:
                self.symbol_ma_20[symbol] = []
            
            self.symbol_ma_20[symbol].append(prices[symbol])
            if len(self.symbol_ma_20[symbol]) > 20:
                self.symbol_ma_20[symbol].pop(0)
        
        # Check for buy signals
        for symbol in prices:
            price = prices[symbol]
            ma_20 = np.mean(self.symbol_ma_20[symbol]) if self.symbol_ma_20[symbol] else price
            
            # Buy signal: price dips 3% below MA and not already holding
            if price < ma_20 * 0.97 and symbol not in self.position_entries:
                # Calculate position size (risk 5% of capital per position)
                portfolio = self.engine.portfolios["Clawdy"]
                risk_amount = portfolio.get_total_equity(prices) * 0.05
                shares = int(risk_amount / price)
                
                if shares > 0:
                    self.engine.execute_trade(
                        "Clawdy", symbol, "BUY", shares, price,
                        f"Price dip: ${price:.2f} vs MA ${ma_20:.2f}"
                    )
                    self.position_entries[symbol] = price
            
            # Sell signal: price rises 5% above entry or MA resistance
            elif symbol in self.position_entries:
                entry = self.position_entries[symbol]
                
                if price > entry * 1.05 or price > ma_20 * 1.03:
                    portfolio = self.engine.portfolios["Clawdy"]
                    qty = portfolio.positions.get(symbol, 0)
                    
                    if qty > 0:
                        self.engine.execute_trade(
                            "Clawdy", symbol, "SELL", qty, price,
                            f"Take profit: +{((price/entry - 1) * 100):.1f}%"
                        )
                        del self.position_entries[symbol]


class BuyHoldStrategy:
    """Buy and hold baseline - buy at start, hold until end."""
    
    def __init__(self, backtest_engine: BacktestEngine, symbols: List[str]):
        self.engine = backtest_engine
        self.symbols = symbols
        self.initialized = False
    
    def process_day(self, date: str, prices: Dict[str, float]):
        """Initialize on first day."""
        if not self.initialized:
            portfolio = self.engine.portfolios["BuyAndHold"]
            
            # Allocate capital equally across symbols
            symbols_available = [s for s in self.symbols if s in prices]
            allocation_per_symbol = portfolio.get_total_equity(prices) / len(symbols_available)
            
            for symbol in symbols_available:
                price = prices[symbol]
                shares = int(allocation_per_symbol / price)
                
                if shares > 0:
                    self.engine.execute_trade(
                        "BuyAndHold", symbol, "BUY", shares, price,
                        "Initial buy & hold allocation"
                    )
            
            self.initialized = True


class DJIAIndexStrategy:
    """DJIA equal-weight basket - buy all 30 DJIA stocks equally."""
    
    def __init__(self, backtest_engine: BacktestEngine):
        self.engine = backtest_engine
        self.initialized = False
    
    def process_day(self, date: str, prices: Dict[str, float]):
        """Initialize on first day."""
        if not self.initialized:
            portfolio = self.engine.portfolios["DJIAIndex"]
            available = [s for s in DJIA_SYMBOLS if s in prices]
            
            # Equal weight allocation
            allocation_per_symbol = portfolio.get_total_equity(prices) / len(available)
            
            for symbol in available:
                price = prices[symbol]
                shares = int(allocation_per_symbol / price)
                
                if shares > 0:
                    self.engine.execute_trade(
                        "DJIAIndex", symbol, "BUY", shares, price,
                        "DJIA index equal-weight"
                    )
            
            self.initialized = True


# ============================================================================
# Main Backtest
# ============================================================================

def run_backtest(start_date: str, end_date: str) -> Dict:
    """Run complete backtest."""
    print(f"\n{'='*70}")
    print(f"🧪 BACKTESTING: Clawdy vs. Baselines")
    print(f"{'='*70}")
    print(f"Period: {start_date} → {end_date}\n")
    
    # Initialize engine
    engine = BacktestEngine(initial_capital=100000)
    engine.register_agent("Clawdy")
    engine.register_agent("BuyAndHold")
    engine.register_agent("DJIAIndex")
    
    # Initialize strategies
    clawdy_strategy = ClawdyStrategy(engine)
    buy_hold_strategy = BuyHoldStrategy(engine, TOP_10_STOCKS)  # canonical top-10 basket
    djia_strategy = DJIAIndexStrategy(engine)
    
    # Fetch price data
    print(f"📊 Fetching historical prices...")
    fetcher = PriceFetcher()
    
    all_dates = {}
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    current = start
    total_days = 0
    
    while current <= end:
        if current.weekday() < 5:  # Only trading days
            date_str = current.strftime("%Y-%m-%d")
            all_dates[date_str] = {}
            total_days += 1
        current += timedelta(days=1)
    
    print(f"   Simulating prices for {len(all_dates)} trading days...")
    
    # Fetch prices for all symbols
    symbols_to_fetch = list(set(DJIA_SYMBOLS + ["SPY", "QQQ"]))
    
    for symbol in symbols_to_fetch[:5]:  # Sample a few to show progress
        prices = fetcher.get_historical(symbol, start_date, end_date)
        for date_str, price in prices.items():
            if date_str in all_dates:
                all_dates[date_str][symbol] = price
    
    print(f"   ✓ Data fetched for {len([d for d in all_dates.values() if d])} days\n")
    
    # Run simulation day by day
    print(f"🚀 Running simulation...")
    for i, (date_str, prices) in enumerate(sorted(all_dates.items())):
        if prices:  # Only process if we have price data
            clawdy_strategy.process_day(date_str, prices)
            buy_hold_strategy.process_day(date_str, prices)
            djia_strategy.process_day(date_str, prices)
            
            engine.record_daily_snapshot(date_str, prices)
        
        # Progress indicator
        if (i + 1) % max(1, len(all_dates) // 10) == 0:
            pct = (i + 1) / len(all_dates) * 100
            print(f"   {pct:.0f}% complete...", end="\r")
    
    print(f"   ✓ Simulation complete        \n")
    
    # Calculate metrics
    print(f"📈 Performance Metrics:\n")
    
    agents = ["Clawdy", "BuyAndHold", "DJIAIndex"]
    results = {}
    
    for agent in agents:
        metrics = engine.calculate_metrics(agent)
        equity_curve = engine.get_equity_curve(agent)
        
        results[agent] = {
            "metrics": {
                "total_return": metrics.total_return,
                "annual_return": metrics.annual_return,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": metrics.max_drawdown,
                "win_rate": metrics.win_rate,
                "num_trades": metrics.num_trades,
                "avg_trade_return": metrics.avg_trade_return,
            },
            "equity_curve": equity_curve
        }
        
        # Final equity
        final_equity = equity_curve[-1]["equity"] if equity_curve else 100000
        
        print(f"   {agent}")
        print(f"      Return: {metrics.total_return:+.2f}%")
        print(f"      Final Equity: ${final_equity:,.0f}")
        print(f"      Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"      Max Drawdown: {metrics.max_drawdown:.2f}%")
        print(f"      Trades: {metrics.num_trades}")
        print()
    
    # Determine winner
    returns = {agent: results[agent]["metrics"]["total_return"] for agent in agents}
    winner = max(returns, key=returns.get)
    
    print(f"🏆 Winner: {winner} (+{returns[winner]:.2f}%)\n")
    
    return results


# ============================================================================
# Export Results to Database & JSON
# ============================================================================

def save_backtest_to_database(results: Dict, start_date: str, end_date: str):
    """Save backtest results to SQLite database."""
    for agent, data in results.items():
        # Generate unique run ID
        run_id = f"backtest_{agent}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        metrics = data["metrics"]
        equity_curve = data["equity_curve"]
        
        # Insert run metadata. This standalone CLI has no session of its own,
        # so it uses the same grouping key every other sessionless writer uses
        # (engine.BacktestEngine's default and backtest_hourly_agent's
        # --session-id default), which is also what database.py backfills onto
        # pre-session rows. A per-run session id would instead mint one orphan
        # session per invocation, unreachable by every session-scoped query.
        #
        # BacktestMetrics reports return/drawdown in PERCENT (backtest_engine
        # multiplies by 100), but agent_runs stores FRACTIONS — every other
        # insert_run caller writes (final - initial) / initial. The frontend
        # disambiguates by magnitude (`Math.abs(x) <= 1 ? x * 100 : x`), so
        # storing percent here would render a +0.4% run as "+40.00%".
        db.insert_run(
            run_id=run_id,
            session_id="legacy-demo-session",
            agent_name=agent,
            mode="backtest",
            start_date=start_date,
            end_date=end_date,
            initial_equity=100000,
            final_equity=equity_curve[-1]["equity"] if equity_curve else 100000,
            total_return=metrics["total_return"] / 100.0,
            sharpe_ratio=metrics["sharpe_ratio"],
            max_drawdown=metrics["max_drawdown"] / 100.0,
            num_trades=metrics["num_trades"]
        )
        
        # Insert equity curve points
        db.insert_equity_points(run_id, equity_curve)
        
        print(f"   ✅ {agent} saved (run_id: {run_id})")


def export_results(results: Dict, start_date: str, end_date: str):
    """Export backtest results to JSON and database."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable format
    export_data = {}
    for agent, data in results.items():
        export_data[agent] = {
            "metrics": data["metrics"],
            "equity_curve": data["equity_curve"]
        }
    
    RESULTS_FILE.write_text(json.dumps(export_data, indent=2))
    print(f"✅ Results saved to JSON: {RESULTS_FILE}")
    
    # Save to database for dashboard
    print(f"📊 Saving to database...")
    save_backtest_to_database(results, start_date, end_date)
    print()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Backtest Clawdy Trading Agent")
    parser.add_argument("--start", default="2026-03-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-04-13", help="End date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    results = run_backtest(args.start, args.end)
    export_results(results, args.start, args.end)


if __name__ == "__main__":
    main()
