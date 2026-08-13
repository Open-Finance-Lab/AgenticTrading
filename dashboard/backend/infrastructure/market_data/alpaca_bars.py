"""Alpaca historical bar loader.

Extracted (Phase 2B1) from ``AlpacaDataLoader`` in
``dashboard/scripts/backtest_hourly_agent.py``. One deliberate behavior change
since the move (B0/H4 deep fix): missing credentials or a missing alpaca-py SDK
raise :class:`MarketDataUnavailableError` instead of ``sys.exit(1)``. SystemExit
is a BaseException — it sailed past ``except Exception`` at every server call
site, silently killed daemon loader threads, and wedged the ASGI loop (the
original B0 hang). A plain exception is catchable everywhere; only CLI
entrypoints translate it back into an exit code.

This is intentionally NOT merged with ``dashboard/backend/market_data.py``; that
consolidation belongs to a later domain-migration phase. The Alpaca SDK imports
remain lazy (inside ``__init__``) so importing this module performs no network
requests.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from dashboard.backend.paths import CREDENTIALS_DIR

# Basic plan may query SIP historical bars, but not the most recent window.
# Docs: https://docs.alpaca.markets/docs/market-data-faq
# ("end must be at least 15 minutes old to query SIP without a subscription").
DEFAULT_SIP_DELAY_MINUTES = 15


class MarketDataUnavailableError(RuntimeError):
    """Market data cannot be loaded (missing credentials, SDK, or data).

    Deliberately a plain Exception subclass: server code catches it with
    ``except Exception``; CLI entrypoints convert it to ``sys.exit(1)``.
    """


class AlpacaCredentialsError(MarketDataUnavailableError):
    """Raised when Alpaca API credentials are not configured."""


def sip_delay_minutes() -> int:
    raw = (os.getenv("ALPACA_SIP_DELAY_MINUTES") or "").strip()
    if not raw:
        return DEFAULT_SIP_DELAY_MINUTES
    try:
        return max(0, int(raw))
    except ValueError:
        print(
            f"WARNING: ALPACA_SIP_DELAY_MINUTES={raw!r} is not an integer; "
            f"using {DEFAULT_SIP_DELAY_MINUTES}"
        )
        return DEFAULT_SIP_DELAY_MINUTES


def allow_recent_sip() -> bool:
    raw = (os.getenv("ALPACA_ALLOW_RECENT_SIP") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def parse_alpaca_end(end: Union[str, datetime]) -> datetime:
    """Normalize an Alpaca ``end`` to an aware UTC datetime."""
    if isinstance(end, datetime):
        dt = end
    else:
        text = str(end).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def clamp_end_for_sip(
    end: Union[str, datetime],
    *,
    now: Optional[datetime] = None,
    delay_minutes: Optional[int] = None,
) -> datetime:
    """Cap ``end`` so Basic-plan SIP queries stay outside the recent window."""
    end_dt = parse_alpaca_end(end)
    minutes = DEFAULT_SIP_DELAY_MINUTES if delay_minutes is None else delay_minutes
    if minutes <= 0 or allow_recent_sip():
        return end_dt
    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=timezone.utc)
    cutoff = clock.astimezone(timezone.utc) - timedelta(minutes=minutes)
    return min(end_dt, cutoff)


def resolve_alpaca_data_feed(data_feed_enum: Any):
    """Resolve ``ALPACA_DATA_FEED``; default SIP (full tape) for backtests.

    Basic accounts can use SIP for history older than ~15 minutes. IEX is only
    ~2.5% of volume and is the wrong default for DJIA / multi-name backtests.
    """
    raw = (os.getenv("ALPACA_DATA_FEED") or "sip").strip().lower()
    mapping = {
        "iex": data_feed_enum.IEX,
        "sip": data_feed_enum.SIP,
        "delayed_sip": getattr(data_feed_enum, "DELAYED_SIP", data_feed_enum.SIP),
        "otc": getattr(data_feed_enum, "OTC", data_feed_enum.IEX),
    }
    if raw not in mapping:
        print(f"WARNING: ALPACA_DATA_FEED={raw!r} is not recognized; using sip")
        return data_feed_enum.SIP
    return mapping[raw]


class AlpacaDataLoader:
    """Fetches historical hourly bars from Alpaca API."""

    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        """Initialize with Alpaca credentials."""
        if not api_key or not secret_key:
            creds = self._load_credentials()
            api_key = creds.get("api_key")
            secret_key = creds.get("secret_key")

        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://data.alpaca.markets"

        try:
            from alpaca.data.enums import DataFeed
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            self.client = StockHistoricalDataClient(self.api_key, self.secret_key)
            self.StockBarsRequest = StockBarsRequest
            self.TimeFrame = TimeFrame
            self.DataFeed = DataFeed
            print("✅ Alpaca credentials loaded")
        except ImportError as e:
            print(f"❌ alpaca-py not installed: {e}")
            print("   Run: pip install alpaca-py")
            raise MarketDataUnavailableError(
                "alpaca-py is not installed (pip install alpaca-py)"
            ) from e

    def _resolve_data_feed(self):
        return resolve_alpaca_data_feed(self.DataFeed)

    def _effective_end(self, end: str, feed) -> Union[str, datetime]:
        """For SIP feeds on Basic, clamp ``end`` outside the recent window."""
        if feed == self.DataFeed.IEX:
            return end
        clamped = clamp_end_for_sip(end, delay_minutes=sip_delay_minutes())
        original = parse_alpaca_end(end)
        if clamped < original:
            print(
                f"   Clamping SIP end {original.isoformat()} → {clamped.isoformat()} "
                f"(Basic plan blocks recent SIP; set ALPACA_ALLOW_RECENT_SIP=1 if paid)"
            )
        return clamped

    def _load_credentials(self) -> Dict:
        """Load Alpaca credentials from environment variables or file."""
        # Try environment variables first (for Render, Docker, etc.)
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')

        if api_key and secret_key:
            print("✅ Loaded Alpaca credentials from environment variables")
            return {"api_key": api_key, "secret_key": secret_key}

        # Fall back to credentials file (for local development)
        creds_path = CREDENTIALS_DIR / "alpaca.json"
        if not creds_path.exists():
            print(f"❌ Credentials not found in environment variables or file: {creds_path}")
            print("   Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
            raise AlpacaCredentialsError(
                "Alpaca credentials not found (set ALPACA_API_KEY and "
                f"ALPACA_SECRET_KEY, or provide {creds_path})"
            )

        print(f"✅ Loaded Alpaca credentials from {creds_path}")
        with open(creds_path) as f:
            return json.load(f)

    def _bars_to_frames(self, bars, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        data = {}
        for symbol in symbols:
            if symbol in bars.df.index.get_level_values(0):
                df = bars.df.xs(symbol).reset_index()
                df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
                data[symbol] = df.sort_index()
                print(f"  ✅ {symbol}: {len(df)} hourly bars")
            else:
                print(f"  ⚠️  {symbol}: No data available")
        return data

    def fetch_bars(self, symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch hourly OHLCV data from Alpaca API.

        Args:
            symbols: List of stock symbols
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)

        Returns:
            {symbol: DataFrame with timestamp, open, high, low, close, volume}
        """
        if not self.client:
            print("⚠️ Alpaca not configured — skipping bar fetch")
            return {}

        print(f"\n📊 Fetching {len(symbols)} symbols from {start} to {end}...")
        feed = self._resolve_data_feed()
        effective_end = self._effective_end(end, feed)
        print(
            f"   Timeframe: Hourly (1h) feed={feed.value} "
            f"end={effective_end} with forward-filled price cache\n"
        )

        request = self.StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=self.TimeFrame.Hour,
            start=start,
            end=effective_end,
            feed=feed,
        )

        try:
            bars = self.client.get_stock_bars(request)
            return self._bars_to_frames(bars, symbols)

        except Exception as e:
            message = str(e)
            print(f"❌ Error fetching bars ({feed.value}): {message}")
            # Basic SIP refusal on an unclamped window — retry once on IEX so
            # a local mis-set end does not wipe the whole backtest.
            if (
                "subscription does not permit" in message.lower()
                and feed != self.DataFeed.IEX
            ):
                print("   Retrying with feed=iex…")
                try:
                    retry = self.StockBarsRequest(
                        symbol_or_symbols=symbols,
                        timeframe=self.TimeFrame.Hour,
                        start=start,
                        end=end,
                        feed=self.DataFeed.IEX,
                    )
                    bars = self.client.get_stock_bars(retry)
                    return self._bars_to_frames(bars, symbols)
                except Exception as retry_exc:
                    print(f"❌ IEX retry also failed: {retry_exc}")
                    return {}
            if "subscription does not permit" not in message.lower():
                import traceback
                traceback.print_exc()
            return {}
