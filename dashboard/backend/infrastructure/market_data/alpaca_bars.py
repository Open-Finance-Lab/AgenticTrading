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

# Which tape prices every backtest, baseline and leaderboard curve. SIP is the
# full consolidated tape; IEX is ~2.5% of volume. Changing this changes results,
# so it is recorded per run (see ``feed_provenance``) rather than left implicit.
DEFAULT_ALPACA_FEED = "sip"

# Canonical feed names, matching ``alpaca.data.enums.DataFeed`` ``.value``.
SUPPORTED_ALPACA_FEEDS = ("iex", "sip", "delayed_sip", "otc")

# alpaca-py 0.43.2 issues every HTTP request via
# ``self._session.request(method, url, **opts)`` with no ``timeout`` in
# ``opts``, so a stalled socket blocks ``requests`` forever and permanently
# leaks a threadpool thread -- this binds at concurrency >= 1, not just under
# burst load. Read once at import, like MAX_ACTIVE_RUNS_PER_AGENT.
#
# Parsed defensively, same shape as ``_max_active_dashboard_backtests`` in
# ``api/routers/backtests.py``: a typo'd operator value must not take the
# whole app down at import (this module is on the boot path), so a bad value
# falls back to the default with a log line instead of raising. No range
# check on the fallen-back-to value -- a negative timeout is already rejected
# loudly by urllib3 on the first request, so silently clamping it here would
# hide the operator's mistake instead of surfacing it.
_DEFAULT_ALPACA_HTTP_TIMEOUT_SECONDS = 60.0
_DEFAULT_ALPACA_HTTP_CONNECT_TIMEOUT_SECONDS = 10.0


def _alpaca_http_timeout_seconds() -> float:
    raw = os.getenv("ALPACA_HTTP_TIMEOUT_SECONDS")
    if raw is None or not str(raw).strip():
        return _DEFAULT_ALPACA_HTTP_TIMEOUT_SECONDS
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        print(
            "ALPACA_HTTP_TIMEOUT_SECONDS is not a number "
            f"({raw!r}); using {_DEFAULT_ALPACA_HTTP_TIMEOUT_SECONDS}",
            flush=True,
        )
        return _DEFAULT_ALPACA_HTTP_TIMEOUT_SECONDS


def _alpaca_http_connect_timeout_seconds() -> float:
    raw = os.getenv("ALPACA_HTTP_CONNECT_TIMEOUT_SECONDS")
    if raw is None or not str(raw).strip():
        return _DEFAULT_ALPACA_HTTP_CONNECT_TIMEOUT_SECONDS
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        print(
            "ALPACA_HTTP_CONNECT_TIMEOUT_SECONDS is not a number "
            f"({raw!r}); using {_DEFAULT_ALPACA_HTTP_CONNECT_TIMEOUT_SECONDS}",
            flush=True,
        )
        return _DEFAULT_ALPACA_HTTP_CONNECT_TIMEOUT_SECONDS


ALPACA_HTTP_TIMEOUT_SECONDS = _alpaca_http_timeout_seconds()
ALPACA_HTTP_CONNECT_TIMEOUT_SECONDS = _alpaca_http_connect_timeout_seconds()

# Stamped on each returned frame so a rare IEX fallback is visible to callers
# (return type stays Dict[str, DataFrame] for the MarketDataProvider contract).
FRAME_ATTR_FEED = "alpaca_feed"
FRAME_ATTR_SIP_FALLBACK = "alpaca_sip_fallback"
FRAME_ATTR_END_CLAMPED = "alpaca_end_clamped"


class MarketDataUnavailableError(RuntimeError):
    """Market data cannot be loaded (missing credentials, SDK, or data).

    Deliberately a plain Exception subclass: server code catches it with
    ``except Exception``; CLI entrypoints convert it to ``sys.exit(1)``.
    """


class AlpacaCredentialsError(MarketDataUnavailableError):
    """Raised when Alpaca API credentials are not configured."""


class AlpacaFeedConfigError(MarketDataUnavailableError):
    """``ALPACA_DATA_FEED`` names a feed we cannot serve.

    Deliberately fatal rather than a warning-and-default: this variable selects
    which tape priced a published leaderboard run, so silently substituting a
    different feed for a typo'd one ("IEXX" meant to force IEX) would ship the
    opposite of the operator's intent, and in prod the warning would be one log
    line nobody reads.
    """


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


def _parse_alpaca_end_or_none(value: Union[str, datetime]) -> Optional[datetime]:
    """``parse_alpaca_end`` for values we are willing to ignore when malformed."""
    try:
        return parse_alpaca_end(value)
    except (TypeError, ValueError):
        return None


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
    start: Optional[Union[str, datetime]] = None,
    now: Optional[datetime] = None,
    delay_minutes: Optional[int] = None,
) -> Union[str, datetime]:
    """Cap ``end`` so Basic-plan SIP queries stay outside the recent window.

    ``start`` (when given) floors the cutoff: a same-day request dispatched
    shortly after 00:00 UTC would otherwise be clamped to ``now−15m`` — i.e.
    *yesterday*, an inverted range that Alpaca answers with nothing and that
    the caller's negative cache then pins as a hard failure. Clamping is an
    accommodation for the Basic plan, never a reason to invert a window.

    An ``end`` this module cannot parse is returned unchanged rather than
    raising: the clamp is an optimization, and the SDK's own validation gives a
    better error than a bare ``ValueError`` from here.
    """
    end_dt = _parse_alpaca_end_or_none(end)
    if end_dt is None:
        return end
    minutes = DEFAULT_SIP_DELAY_MINUTES if delay_minutes is None else delay_minutes
    if minutes <= 0 or allow_recent_sip():
        return end_dt
    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=timezone.utc)
    cutoff = clock.astimezone(timezone.utc) - timedelta(minutes=minutes)
    if start is not None:
        start_dt = _parse_alpaca_end_or_none(start)
        if start_dt is not None:
            cutoff = max(cutoff, start_dt)
    return min(end_dt, cutoff)


def configured_feed_name() -> str:
    """Canonical name of the tape selected by ``ALPACA_DATA_FEED``.

    Kept SDK-free so callers that only need to *record* or compare the feed
    (leaderboard run provenance) do not have to import alpaca-py.
    """
    raw = (os.getenv("ALPACA_DATA_FEED") or DEFAULT_ALPACA_FEED).strip().lower()
    if raw not in SUPPORTED_ALPACA_FEEDS:
        raise AlpacaFeedConfigError(
            f"ALPACA_DATA_FEED={raw!r} is not a known Alpaca feed; "
            f"expected one of {', '.join(SUPPORTED_ALPACA_FEEDS)}"
        )
    return raw


def resolve_alpaca_data_feed(data_feed_enum: Any):
    """Resolve ``ALPACA_DATA_FEED``; default SIP (full tape) for backtests.

    Basic accounts can use SIP for history older than ~15 minutes. IEX is only
    ~2.5% of volume and is the wrong default for DJIA / multi-name backtests.

    An unknown name — or one this SDK build's enum lacks — raises rather than
    falling back, so a typo cannot silently price a published run off the wrong
    tape. See :class:`AlpacaFeedConfigError`.
    """
    name = configured_feed_name()
    member = getattr(data_feed_enum, name.upper(), None)
    if member is None:
        raise AlpacaFeedConfigError(
            f"ALPACA_DATA_FEED={name!r} is not supported by the installed "
            "alpaca-py DataFeed enum"
        )
    return member


def feed_provenance(bars: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
    """Which tape produced these frames, read back off the stamps.

    Returns ``None`` for an empty or unstamped mapping (e.g. an index strategy
    priced from Yahoo, which never touches Alpaca) so callers can tell "not
    applicable" from "IEX fallback".
    """
    for frame in bars.values():
        attrs = getattr(frame, "attrs", None) or {}
        if FRAME_ATTR_FEED in attrs:
            return {
                "market_data_feed": attrs.get(FRAME_ATTR_FEED),
                "sip_fallback_to_iex": bool(attrs.get(FRAME_ATTR_SIP_FALLBACK)),
                "end_clamped": bool(attrs.get(FRAME_ATTR_END_CLAMPED)),
            }
    return None


def _apply_default_timeout(client: Any) -> None:
    """Wrap ``client._session.request`` so every call gets a default timeout.

    alpaca-py builds ``requests.Session.request`` with no ``timeout`` kwarg,
    so a stalled socket hangs forever and permanently leaks a threadpool
    thread. If a future alpaca-py release renames or drops ``_session``, fail
    open (warn, don't raise) rather than silently restoring the unbounded
    behavior this exists to prevent -- but make that loud, since a quiet
    warning nobody reads is exactly how the original bug went unnoticed.
    """
    session = getattr(client, "_session", None)
    original_request = getattr(session, "request", None)
    if session is None or original_request is None:
        print(
            "WARNING: Alpaca client has no usable _session.request "
            "(likely an alpaca-py version change) -- "
            "ALPACA_HTTP_TIMEOUT_SECONDS default timeout was NOT applied; "
            "Alpaca HTTP requests are unbounded until this is fixed"
        )
        return

    if getattr(original_request, "_atl_default_timeout_applied", False):
        return

    def _request_with_default_timeout(*args, **kwargs):
        kwargs.setdefault(
            "timeout", (ALPACA_HTTP_CONNECT_TIMEOUT_SECONDS, ALPACA_HTTP_TIMEOUT_SECONDS)
        )
        return original_request(*args, **kwargs)

    _request_with_default_timeout._atl_default_timeout_applied = True
    session.request = _request_with_default_timeout


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
            _apply_default_timeout(self.client)
            self.StockBarsRequest = StockBarsRequest
            self.TimeFrame = TimeFrame
            self.DataFeed = DataFeed
            self.last_fetch: Optional[Dict[str, Any]] = None
            print("✅ Alpaca credentials loaded")
        except ImportError as e:
            print(f"❌ alpaca-py not installed: {e}")
            print("   Run: pip install alpaca-py")
            raise MarketDataUnavailableError(
                "alpaca-py is not installed (pip install alpaca-py)"
            ) from e

    def _resolve_data_feed(self):
        return resolve_alpaca_data_feed(self.DataFeed)

    def _effective_end(
        self, end: str, feed, start: Optional[str] = None
    ) -> tuple[Union[str, datetime], bool]:
        """For SIP feeds on Basic, clamp ``end`` outside the recent window.

        Returns ``(effective_end, was_clamped)``; the flag is recorded as run
        provenance so a shortened window is identifiable after the fact.

        Alpaca ``end`` is exclusive and filters on each bar's *opening*
        timestamp — the left edge of the interval, per the market-data FAQ.
        That is why ``fetch_hourly_bars`` bumps ``end_date`` by a day: bars on
        ``end_date`` open after midnight and would otherwise be dropped. It is
        also why the clamp does not truncate a just-closed session: at 16:05 ET
        the cutoff is 15:50 ET, still later than the 15:00 ET open of the final
        RTH hourly bar, so that bar is returned whole.

        The margin is one bar wide, though — a ``ALPACA_SIP_DELAY_MINUTES``
        above ~65 pushes the cutoff below 15:00 ET and *does* drop the closing
        hour, which the daily board would then cache for the rest of the day.
        ``test_clamp_keeps_final_rth_bar_after_close`` pins the default.
        """
        if feed == self.DataFeed.IEX:
            return end, False
        clamped = clamp_end_for_sip(
            end, start=start, delay_minutes=sip_delay_minutes()
        )
        original = _parse_alpaca_end_or_none(end)
        if original is None:
            return clamped, False
        if not isinstance(clamped, datetime) or clamped >= original:
            return clamped, False
        print(
            f"   Clamping SIP end {original.isoformat()} → {clamped.isoformat()} "
            f"(Basic plan blocks recent SIP; set ALPACA_ALLOW_RECENT_SIP=1 if paid)"
        )
        return clamped, True

    def _record_fetch(
        self,
        *,
        feed,
        requested_end: str,
        effective_end: Union[str, datetime],
        sip_fallback_to_iex: bool,
        end_clamped: bool = False,
    ) -> None:
        self.last_fetch = {
            "feed": getattr(feed, "value", str(feed)),
            "requested_end": requested_end,
            "effective_end": effective_end,
            "sip_fallback_to_iex": sip_fallback_to_iex,
            "end_clamped": end_clamped,
        }

    def _stamp_frames(
        self,
        data: Dict[str, pd.DataFrame],
        *,
        feed,
        sip_fallback_to_iex: bool,
        end_clamped: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        feed_name = getattr(feed, "value", str(feed))
        for frame in data.values():
            frame.attrs[FRAME_ATTR_FEED] = feed_name
            frame.attrs[FRAME_ATTR_SIP_FALLBACK] = sip_fallback_to_iex
            frame.attrs[FRAME_ATTR_END_CLAMPED] = end_clamped
        return data

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
            self.last_fetch = None
            return {}

        print(f"\n📊 Fetching {len(symbols)} symbols from {start} to {end}...")
        feed = self._resolve_data_feed()
        effective_end, end_clamped = self._effective_end(end, feed, start)
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
            self._record_fetch(
                feed=feed,
                requested_end=end,
                effective_end=effective_end,
                sip_fallback_to_iex=False,
                end_clamped=end_clamped,
            )
            return self._stamp_frames(
                self._bars_to_frames(bars, symbols),
                feed=feed,
                sip_fallback_to_iex=False,
                end_clamped=end_clamped,
            )

        except Exception as e:
            message = str(e)
            print(f"❌ Error fetching bars ({feed.value}): {message}")
            # Clamp should keep Basic SIP outside the recent window. If Alpaca
            # still refuses, retry once on IEX so a local mis-set end does not
            # wipe the backtest — but mark the result so callers can see it
            # is not full-tape SIP. IEX allows recent data, so retry uses the
            # original unclamped ``end``.
            if (
                "subscription does not permit" in message.lower()
                and feed != self.DataFeed.IEX
            ):
                print(
                    "WARNING: SIP refused; retrying feed=iex. "
                    "IEX is ~2.5% of volume, not the SIP tape. "
                    "Frames are stamped alpaca_sip_fallback=True."
                )
                try:
                    retry = self.StockBarsRequest(
                        symbol_or_symbols=symbols,
                        timeframe=self.TimeFrame.Hour,
                        start=start,
                        end=end,
                        feed=self.DataFeed.IEX,
                    )
                    bars = self.client.get_stock_bars(retry)
                    self._record_fetch(
                        feed=self.DataFeed.IEX,
                        requested_end=end,
                        effective_end=end,
                        sip_fallback_to_iex=True,
                    )
                    return self._stamp_frames(
                        self._bars_to_frames(bars, symbols),
                        feed=self.DataFeed.IEX,
                        sip_fallback_to_iex=True,
                    )
                except Exception as retry_exc:
                    print(f"❌ IEX retry also failed: {retry_exc}")
                    self.last_fetch = None
                    return {}
            if "subscription does not permit" not in message.lower():
                import traceback
                traceback.print_exc()
            self.last_fetch = None
            return {}
