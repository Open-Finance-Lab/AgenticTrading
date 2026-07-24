from dashboard.backend.domain.backtesting.constants import INITIAL_CAPITAL
"""
Paper trading baselines - calculate independently from backtesting.
Creates DJIA Index and Buy & Hold baselines for the paper account's date range.

Canonical location (Phase 3C5B). Moved verbatim from
``dashboard/backend/paper_baselines.py`` (now a thin compatibility re-export
shim). Baseline formulas, symbols, date ranges, data URLs, output formats,
exceptions, and logging are unchanged; only the module location moved. The
Alpaca provider adapter continues to come from
``dashboard.backend.infrastructure.brokers.alpaca_paper``.
"""

import json
import requests
import os
import uuid
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from dashboard.backend.paths import CREDENTIALS_DIR

from dashboard.backend.database import db
from dashboard.backend.infrastructure.brokers.alpaca_paper import AlpacaPaperTradingClient
from dashboard.backend.infrastructure.llm.validator import DJIA_30, TOP_10_STOCKS

# Canonical Dow-30 (single source: validator.DJIA_30, guarded by
# tests/test_djia30_universe.py), kept under the historical local name.
DJIA_SYMBOLS = list(DJIA_30)


class PaperTradingBaselineCalculator:
    """Calculate baselines for paper trading from Alpaca data."""
    
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        """Initialize with Alpaca credentials."""
        # Always initialize these first
        self.api_key = None
        self.secret_key = None
        
        if api_key is None:
            self._load_credentials()
        else:
            self.api_key = api_key
            self.secret_key = secret_key
        
        self.data_api_url = "https://data.alpaca.markets"
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }
    
    def _load_credentials(self):
        """Load credentials from environment variables or file."""
        # Try environment variables first (for Render, Docker, etc.)
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        print(f"🔍 DEBUG: ALPACA_API_KEY env var = {self.api_key is not None}")
        print(f"🔍 DEBUG: ALPACA_SECRET_KEY env var = {self.secret_key is not None}")
        
        if self.api_key and self.secret_key:
            print("✅ Loaded Alpaca credentials from environment variables")
            return
        
        # Fall back to credentials file (for local development)
        creds_path = CREDENTIALS_DIR / "alpaca.json"
        if creds_path.exists():
            try:
                with open(creds_path, 'r') as f:
                    creds = json.load(f)
                    self.api_key = creds.get('api_key') or creds.get('apiKey')
                    self.secret_key = creds.get('secret_key') or creds.get('secretKey')
                    print("✅ Loaded Alpaca credentials from credentials/alpaca.json")
            except Exception as e:
                print(f"⚠️ Error loading credentials file: {e}")
                self.api_key = None
                self.secret_key = None
        else:
            # No credentials found anywhere
            self.api_key = None
            self.secret_key = None
            print("⚠️ Warning: Alpaca credentials not found in environment variables or credentials/alpaca.json")
    
    def get_paper_account_date_range(self) -> Optional[tuple]:
        """
        Get date range from paper account portfolio history.
        Returns (start_date, end_date) as datetime objects.
        """
        if not self.api_key or not self.secret_key:
            print("⚠️ Alpaca credentials not configured - skipping paper baseline initialization")
            return None
        
        try:
            client = AlpacaPaperTradingClient(self.api_key, self.secret_key)
            history = client.get_portfolio_history(timeframe="1D")
            
            if history and "timestamp" in history:
                timestamps = history.get("timestamp", [])
                if timestamps:
                    # Convert timestamps to datetime
                    start_ts = timestamps[0]
                    end_ts = timestamps[-1]
                    
                    start_date = datetime.fromtimestamp(start_ts) if isinstance(start_ts, (int, float)) else datetime.fromisoformat(str(start_ts))
                    end_date = datetime.fromtimestamp(end_ts) if isinstance(end_ts, (int, float)) else datetime.fromisoformat(str(end_ts))
                    
                    return (start_date, end_date)
        except Exception as e:
            print(f"Error getting paper account date range: {e}")
        
        return None
    
    def fetch_djia_historical(self, start_date: datetime, end_date: datetime) -> Optional[List[Dict]]:
        """
        Fetch DJIA historical data for a specific date range.
        Uses SPY as proxy.
        """
        try:
            url = f"{self.data_api_url}/v2/stocks/SPY/bars"
            params = {
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d"),
                "timeframe": "1D"
            }
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            
            if response.status_code != 200:
                print(f"Error fetching SPY: {response.status_code}")
                return None
            
            bars = response.json().get("bars", [])
            if not bars:
                return None
            
            # Create equity curve starting at INITIAL_CAPITAL
            initial_price = bars[0]["c"]
            equity_curve = []
            
            for bar in bars:
                price = bar["c"]
                equity = INITIAL_CAPITAL * (price / initial_price)
                equity_curve.append({
                    "timestamp": bar["t"],
                    "equity": round(equity, 2),
                    "cash": round(equity * 0.3, 2),
                    "positions_value": round(equity * 0.7, 2),
                    "daily_return": (price / initial_price) - 1
                })
            
            return equity_curve
        
        except Exception as e:
            print(f"Error in fetch_djia_historical: {e}")
            return None
    
    def fetch_buy_and_hold_djia(self, start_date: datetime, end_date: datetime) -> Optional[List[Dict]]:
        """
        Calculate equal-weighted DJIA buy-and-hold for date range.
        """
        try:
            # Use the canonical 10-stock basket (faster than all 30)
            sample_symbols = TOP_10_STOCKS
            all_bars = {}
            
            for symbol in sample_symbols:
                try:
                    url = f"{self.data_api_url}/v2/stocks/{symbol}/bars"
                    params = {
                        "start": start_date.strftime("%Y-%m-%d"),
                        "end": end_date.strftime("%Y-%m-%d"),
                        "timeframe": "1D"
                    }
                    
                    response = requests.get(url, headers=self.headers, params=params, timeout=5)
                    
                    if response.status_code == 200:
                        bars = response.json().get("bars", [])
                        if bars:
                            all_bars[symbol] = bars
                except Exception as e:
                    print(f"Error fetching {symbol}: {e}")
                    continue
            
            if not all_bars:
                return None
            
            # Build equal-weighted portfolio
            first_symbol = next(iter(all_bars.keys()))
            timestamps = [bar["t"] for bar in all_bars[first_symbol]]
            
            equity_curve = []
            
            for bar_idx, timestamp in enumerate(timestamps):
                total_return = 0
                count = 0
                
                for symbol in all_bars:
                    bars = all_bars[symbol]
                    if bar_idx < len(bars):
                        price = bars[bar_idx]["c"]
                        initial_price = bars[0]["c"]
                        daily_return = (price / initial_price) - 1
                        
                        total_return += daily_return
                        count += 1
                
                avg_return = total_return / count if count > 0 else 0
                equity = INITIAL_CAPITAL * (1 + avg_return)
                
                equity_curve.append({
                    "timestamp": timestamp,
                    "equity": round(equity, 2),
                    "cash": round(equity * 0.3, 2),
                    "positions_value": round(equity * 0.7, 2),
                    "daily_return": avg_return
                })
            
            return equity_curve
        
        except Exception as e:
            print(f"Error in fetch_buy_and_hold_djia: {e}")
            return None
    
    def create_paper_baselines(self) -> bool:
        """
        Create baselines for paper trading account date range.
        Stores as mode='paper_baseline' in database.
        """
        print("📊 Creating baselines for paper trading...")
        
        # Get paper account date range
        date_range = self.get_paper_account_date_range()
        if not date_range:
            print("❌ Could not determine paper account date range")
            return False
        
        start_date, end_date = date_range
        print(f"  Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Fetch DJIA baseline
        print("  Fetching DJIA Index...")
        djia_curve = self.fetch_djia_historical(start_date, end_date)
        
        if not djia_curve:
            print("  ⚠️ Could not fetch DJIA (trying synthetic)")
            # Fall back to synthetic
            djia_curve = self._create_synthetic_curve(start_date, end_date, drift=0.0004, vol=0.012)
            if not djia_curve:
                print("  ❌ Failed to create DJIA baseline")
                return False
        
        print(f"  ✅ DJIA: {len(djia_curve)} points")
        
        # Fetch Buy-and-Hold baseline
        print("  Fetching Buy & Hold DJIA...")
        bah_curve = self.fetch_buy_and_hold_djia(start_date, end_date)
        
        if not bah_curve:
            print("  ⚠️ Could not fetch B&H (trying synthetic)")
            bah_curve = self._create_synthetic_curve(start_date, end_date, drift=0.0003, vol=0.015)
            if not bah_curve:
                print("  ❌ Failed to create B&H baseline")
                return False
        
        print(f"  ✅ Buy & Hold: {len(bah_curve)} points")
        
        # Store in database
        now = datetime.now()
        
        # DJIA run
        djia_run_id = f"djia_paper_baseline_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        djia_initial = djia_curve[0]["equity"]
        djia_final = djia_curve[-1]["equity"]
        djia_return = (djia_final - djia_initial) / djia_initial
        
        db.insert_run(
            run_id=djia_run_id,
            session_id="baseline-demo",
            agent_name="DJIA Index",
            mode="paper_baseline",
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            initial_equity=djia_initial,
            final_equity=djia_final,
            total_return=djia_return,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            num_trades=0
        )
        db.insert_equity_points(djia_run_id, djia_curve)
        print(f"  ✅ Stored DJIA baseline")
        
        # Buy-and-Hold run
        bah_run_id = f"bah_paper_baseline_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        bah_initial = bah_curve[0]["equity"]
        bah_final = bah_curve[-1]["equity"]
        bah_return = (bah_final - bah_initial) / bah_initial
        
        db.insert_run(
            run_id=bah_run_id,
            session_id="baseline-demo",
            agent_name="Buy & Hold DJIA",
            mode="paper_baseline",
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            initial_equity=bah_initial,
            final_equity=bah_final,
            total_return=bah_return,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            num_trades=0
        )
        db.insert_equity_points(bah_run_id, bah_curve)
        print(f"  ✅ Stored Buy & Hold baseline")
        
        return True
    
    def _create_synthetic_curve(self, start_date: datetime, end_date: datetime, 
                                drift: float = 0.0004, vol: float = 0.012) -> List[Dict]:
        """Create synthetic equity curve for fallback."""
        import random
        
        curve = []
        equity = INITIAL_CAPITAL
        current_date = start_date
        
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Weekdays only
                daily_return = drift + vol * random.gauss(0, 1)
                equity *= (1 + daily_return)
                
                curve.append({
                    "timestamp": current_date.isoformat(),
                    "equity": round(equity, 2),
                    "cash": round(equity * 0.3, 2),
                    "positions_value": round(equity * 0.7, 2),
                    "daily_return": daily_return
                })
            
            current_date += timedelta(days=1)
        
        return curve


def create_paper_baselines_if_not_exists() -> bool:
    """
    Create paper trading baselines if they don't exist.
    Fetches them for the current paper account date range.
    """
    try:
        # Check if baselines already exist
        existing = db.get_runs_by_mode("paper_baseline")
        if existing and len(existing) >= 2:
            print(f"✅ Paper baselines already exist ({len(existing)} runs)")
            return True
        
        calculator = PaperTradingBaselineCalculator()
        return calculator.create_paper_baselines()
    
    except Exception as e:
        print(f"⚠️ Error creating paper baselines: {e}")
        return False
