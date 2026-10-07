"""Market-data provider contract, selection, and feature gating."""

from __future__ import annotations

import importlib.util
import os
from time import perf_counter
from datetime import date, datetime, timedelta
from typing import Protocol

import pandas as pd
import pytz

from .alpaca_bars import AlpacaDataLoader
from .frequency import normalize_bar_timeframe
from .profiles import ALPACA, IFIND_ASHARE, VNPY_SIMULATION
from .sessions import timezone_for_market


SUPPORTED_DATA_SOURCES = (ALPACA, VNPY_SIMULATION, IFIND_ASHARE)

_TRUTHY = {"1", "true", "yes", "on"}


def exclusive_end(end_date: str) -> str:
    """The provider ``end`` that covers the inclusive ``end_date``.

    Every provider reads ``end`` as half-open, while a backtest's ``end_date``
    is the last day to trade.

    Parsed with the same ``strptime`` the route and the engine validate with,
    so an unpadded ``2026-9-11`` they accept is bumped here too, and returned
    zero-padded. Anything that does not parse RAISES: passing it through
    unchanged let the provider read it as the exclusive bound again, which
    silently dropped the last day -- the bug this function exists to fix.
    """
    return (parse_ymd(end_date) + timedelta(days=1)).isoformat()


#: Calendar days of history a dashboard backtest fetches before its
#: ``start_date``, so its indicators are warm on the first decision bar
#: (#540). The longest window is sma50; at A-share's four sessions a day that
#: is ~13 trading days, and 30 calendar days still clears it with a week-long
#: exchange closure (National Day, Spring Festival) inside the pad.
INDICATOR_WARMUP_CALENDAR_DAYS = 30


def warmup_fetch_start(start_date: str) -> str:
    """The provider ``start`` that covers ``start_date`` plus its indicator pad.

    One owner for the engine's fetch and the boot-time bar-cache warm: the
    cache is keyed on the requested ``start``, so a warm that padded by a
    different amount would fill entries no run reads. Zero-padded and raising
    on a malformed date, like ``exclusive_end``.
    """
    return (
        parse_ymd(start_date) - timedelta(days=INDICATOR_WARMUP_CALENDAR_DAYS)
    ).isoformat()


def parse_ymd(value: object) -> date:
    """A ``YYYY-MM-DD`` date, parsed as leniently as the route parses it."""
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"expected a YYYY-MM-DD date, got {value!r}") from exc


def market_today(market: object = None) -> date:
    """Today's date on ``market``'s own clock."""
    return datetime.now(pytz.timezone(timezone_for_market(market))).date()


def settled_exclusive_end(
    end_date: str,
    *,
    market: object = None,
    today: date | None = None,
) -> str:
    """``exclusive_end``, never past the start of the market's current day.

    A backtest over a window that reaches today would otherwise trade today's
    session while it is still forming: no provider is asked for bars past
    their close (Alpaca returns the hour whose OPEN is before ``end`` with
    whatever volume it has so far, even under the SIP delay clamp), so the
    last bars' OHLCV are partial and the same labelled window re-run an hour
    later draws a different curve. Before inclusive ends, today was excluded
    by accident; this keeps it excluded on purpose.

    Not used by ``leaderboard/baselines.fetch_hourly_bars``: the rolling daily
    board deliberately reads today up to the SIP clamp.
    """
    bound = exclusive_end(end_date)
    ceiling = (today or market_today(market)).isoformat()
    return min(bound, ceiling)


def window_provenance(end_date: str, provider_end: str) -> dict[str, object]:
    """What a run's recorded window actually covered, for ``agent_runs.metadata``.

    Rows written before inclusive ends stopped a day short of the same
    ``end_date`` label; ``end_date_inclusive`` is what tells the two apart
    after the fact. ``open_session_excluded`` marks a window that reached a
    session still in progress, whose last day was therefore not traded.
    """
    return {
        "end_date_inclusive": True,
        "provider_end_date": provider_end,
        "open_session_excluded": provider_end < exclusive_end(end_date),
    }


class MarketDataProvider(Protocol):
    """Normalized market-data input consumed by backtests."""

    def fetch_bars(
        self,
        symbols: list[str],
        start: str,
        end: str,
    ) -> dict[str, pd.DataFrame]:
        """Return symbol-keyed OHLCV frames for the half-open window ``[start, end)``."""


class TraceAwareMarketDataProvider:
    """Observability wrapper that preserves the provider contract."""

    def __init__(self, provider: MarketDataProvider, *, run_id: str, source: str):
        self._provider = provider
        self._run_id = run_id
        self._source = source

    def __getattr__(self, name: str):
        return getattr(self._provider, name)

    def fetch_bars(self, symbols, start, end, *, depth_start=None):
        from dashboard.backend.domain.traces import service as trace_service

        started = perf_counter()
        symbols = list(symbols)
        query = {
            "symbols": [str(symbol) for symbol in symbols],
            "start": str(start), "end": str(end),
        }
        kwargs = {} if depth_start is None else {"depth_start": depth_start}
        query.update({key: str(value) for key, value in kwargs.items()})
        key = f"{self._source}:{start}:{end}:{','.join(query['symbols'])}"
        def record(**event):
            try:
                trace_service.record_data_retrieval(**event)
            except Exception:
                # Trace persistence is observational; provider results and
                # provider errors must remain the source of truth.
                return
        try:
            frames = self._provider.fetch_bars(symbols, start, end, **kwargs)
            record(
                run_id=self._run_id, source=self._source, query_summary=query,
                result_summary={
                    "symbol_count": len(frames or {}),
                    "bar_counts": {str(symbol): int(len(frame)) for symbol, frame in (frames or {}).items()},
                }, duration_ms=(perf_counter() - started) * 1000, idempotency_key=key,
            )
            return frames
        except Exception:
            record(
                run_id=self._run_id, source=self._source, query_summary=query,
                result_summary={}, duration_ms=(perf_counter() - started) * 1000,
                idempotency_key=key, outcome="failed",
            )
            raise


def with_trace(provider: MarketDataProvider, *, run_id: str, source: str) -> MarketDataProvider:
    return TraceAwareMarketDataProvider(provider, run_id=run_id, source=source)


class UnsupportedMarketDataSource(ValueError):
    """Raised when a request names a data source outside the allow-list."""


class MarketDataSourceDisabled(RuntimeError):
    """Raised when a known data source is disabled by configuration."""


class MarketDataDependencyError(RuntimeError):
    """Raised when an optional provider dependency is not installed."""


class MarketDataCredentialsError(RuntimeError):
    """Raised when a selected provider lacks required credentials."""


def _feature_enabled(environment_variable: str) -> bool:
    value = os.getenv(environment_variable, "")
    return value.strip().lower() in _TRUTHY


def vnpy_simulation_enabled() -> bool:
    """Return whether the development-only vn.py simulator is enabled."""
    return _feature_enabled("ENABLE_VNPY_SIMULATION")


def ifind_ashare_enabled() -> bool:
    """Return whether the iFinD A-share provider is enabled."""
    return _feature_enabled("ENABLE_IFIND_ASHARE")


def validate_market_data_source(data_source: str) -> None:
    """Validate the source name and feature gate without creating a client."""
    if data_source not in SUPPORTED_DATA_SOURCES:
        raise UnsupportedMarketDataSource(
            f"Unknown market data source: {data_source!r}"
        )
    if data_source == VNPY_SIMULATION and not vnpy_simulation_enabled():
        raise MarketDataSourceDisabled(
            "vn.py simulation is disabled; set ENABLE_VNPY_SIMULATION=true"
        )
    if data_source == IFIND_ASHARE and not ifind_ashare_enabled():
        raise MarketDataSourceDisabled(
            "iFinD A-share market data is disabled; "
            "set ENABLE_IFIND_ASHARE=true"
        )


def ensure_market_data_source_available(data_source: str) -> None:
    """Validate configuration and optional dependencies without importing vn.py."""
    validate_market_data_source(data_source)
    if data_source == VNPY_SIMULATION and importlib.util.find_spec("vnpy") is None:
        raise MarketDataDependencyError(
            "vn.py is not installed; run pip install -r requirements-vnpy.txt"
        )
    if data_source == IFIND_ASHARE and not any(
        os.getenv(name, "").strip()
        for name in ("IFIND_REFRESH_TOKEN", "IFIND_ACCESS_TOKEN")
    ):
        raise MarketDataCredentialsError(
            "iFinD credentials are not configured; "
            "set IFIND_REFRESH_TOKEN or IFIND_ACCESS_TOKEN"
        )


def create_market_data_provider(
    data_source: str = ALPACA,
    universe: str | None = None,
    *,
    source_timeframe: str | None = None,
) -> MarketDataProvider:
    """Create a provider while keeping optional imports isolated.

    ``source_timeframe`` is intentionally explicit.  Omitting it preserves
    the legacy behavior of requesting the profile's decision timeframe, so an
    existing hourly backtest cannot accidentally start making decisions on
    every minute bar.  Phase 2 will pass ``profile.source_timeframe`` after
    the minute-to-decision-bar aggregation path is connected.
    """
    from .profiles import get_market_profile

    profile = get_market_profile(data_source, universe)
    ensure_market_data_source_available(data_source)
    requested_timeframe = normalize_bar_timeframe(
        profile.timeframe if source_timeframe is None else source_timeframe
    )

    if data_source == ALPACA:
        loader = AlpacaDataLoader()
        # Configure after construction to keep compatibility with lightweight
        # test doubles and legacy integrations that replace AlpacaDataLoader
        # with a zero-argument class.
        configure_market_data_provider(loader, requested_timeframe)
        return loader

    if data_source == IFIND_ASHARE:
        if requested_timeframe != profile.timeframe:
            raise ValueError(
                "iFinD A-share provider currently supports only its profile "
                f"timeframe {profile.timeframe!r}"
            )
        from .ifind_ashare import IFindAshareProvider

        return IFindAshareProvider(profile=profile)

    if requested_timeframe != profile.timeframe:
        raise ValueError(
            "vn.py simulation provider currently supports only its profile "
            f"timeframe {profile.timeframe!r}"
        )

    try:
        from .vnpy_simulation import VnpySimulationProvider
    except ModuleNotFoundError as exc:
        if exc.name == "vnpy" or (exc.name and exc.name.startswith("vnpy.")):
            raise MarketDataDependencyError(
                "vn.py is not installed; run "
                "pip install -r requirements-vnpy.txt"
            ) from exc
        raise

    return VnpySimulationProvider()


def configure_market_data_provider(
    provider: MarketDataProvider,
    source_timeframe: str,
) -> MarketDataProvider:
    """Configure a provider's source timeframe when it supports the feature.

    This small compatibility boundary lets the factory configure the real
    Alpaca loader without requiring every existing injected provider or test
    double to change its constructor signature.
    """
    canonical = normalize_bar_timeframe(source_timeframe)
    configure = getattr(provider, "configure_source_timeframe", None)
    if callable(configure):
        configure(canonical)
    else:
        # A replacement provider may not expose the optional capability. Keep
        # the attribute visible for diagnostics while leaving its own fetch
        # implementation untouched.
        try:
            setattr(provider, "source_timeframe", canonical)
        except (AttributeError, TypeError):
            pass
    return provider
