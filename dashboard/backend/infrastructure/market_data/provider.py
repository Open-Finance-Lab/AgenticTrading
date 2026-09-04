"""Market-data provider contract, selection, and feature gating."""

from __future__ import annotations

import importlib.util
import os
from typing import Protocol

import pandas as pd

from .alpaca_bars import AlpacaDataLoader
from .frequency import normalize_bar_timeframe
from .profiles import ALPACA, IFIND_ASHARE, VNPY_SIMULATION


SUPPORTED_DATA_SOURCES = (ALPACA, VNPY_SIMULATION, IFIND_ASHARE)

_TRUTHY = {"1", "true", "yes", "on"}


class MarketDataProvider(Protocol):
    """Normalized market-data input consumed by backtests."""

    def fetch_bars(
        self,
        symbols: list[str],
        start: str,
        end: str,
    ) -> dict[str, pd.DataFrame]:
        """Return symbol-keyed OHLCV frames for the requested date window."""


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
