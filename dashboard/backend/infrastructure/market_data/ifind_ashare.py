"""Fixed-universe iFinD A-share provider for ATL backtests."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from .ifind_adapter import response_to_frames
from .ifind_client import IFindHttpClient
from .ifind_fx import IFindHistoricalFxProvider
from .ifind_market_rules import response_to_market_rules
from .profiles import IFIND_ASHARE, MarketProfile, get_market_profile


DateInput = str | date | datetime
Adapter = Callable[..., dict[str, pd.DataFrame]]
MarketRuleAdapter = Callable[..., object]
_MARKET_TIMEZONE = ZoneInfo("Asia/Shanghai")
_MINIMUM_BARS = 50


class IFindUniverseError(ValueError):
    """Raised when a caller does not request the selected registered universe."""


class IFindDateInputError(ValueError):
    """Raised when provider date inputs cannot form a half-open date window."""


class IFindAshareProvider:
    """Fetch and adapt one backend-owned registered A-share universe."""

    def __init__(
        self,
        *,
        profile: MarketProfile | None = None,
        client: IFindHttpClient | None = None,
        adapter: Adapter = response_to_frames,
        fx_provider: IFindHistoricalFxProvider | None = None,
        market_rule_adapter: MarketRuleAdapter = response_to_market_rules,
    ) -> None:
        self.profile = profile or get_market_profile(IFIND_ASHARE)
        if self.profile.data_source != IFIND_ASHARE:
            raise ValueError("iFinD provider requires an iFinD market profile")
        self._client = client if client is not None else IFindHttpClient()
        self._adapter = adapter
        self._market_rule_adapter = market_rule_adapter
        self._fx_provider = (
            fx_provider
            if fx_provider is not None
            else IFindHistoricalFxProvider(client=self._client)
        )

    def fetch_bars(
        self,
        symbols: Sequence[str],
        start: DateInput,
        end: DateInput,
    ) -> dict[str, pd.DataFrame]:
        """Fetch one canonical batch and return validated OHLCV frames."""
        canonical_symbols = self._validate_universe(symbols)
        start_date = self._as_market_date(start)
        end_date = self._as_market_date(end)
        if end_date <= start_date:
            raise IFindDateInputError("iFinD end date must be after start date")

        payload = self._client.fetch_hourly_bars(
            canonical_symbols,
            start_date,
            end_date,
        )
        return self._adapter(
            payload,
            expected_symbols=canonical_symbols,
            start=start_date,
            end=end_date,
            min_bars=_MINIMUM_BARS,
        )

    def fetch_usd_cny(
        self,
        symbols: Sequence[str],
        start: DateInput,
        end: DateInput,
    ) -> dict[date, float]:
        """Return validated iFinD historical CNY-per-USD rates."""
        canonical_symbols = self._validate_universe(symbols)
        start_date = self._as_market_date(start)
        end_date = self._as_market_date(end)
        if end_date <= start_date:
            raise IFindDateInputError("iFinD end date must be after start date")
        return self._fx_provider.fetch_usd_cny(
            canonical_symbols,
            start_date,
            end_date,
        )

    def fetch_market_rules(
        self,
        symbols: Sequence[str],
        start: DateInput,
        end: DateInput,
        *,
        bars_by_symbol: dict[str, pd.DataFrame],
    ):
        """Return the official validated rule calendar for loaded A-share bars."""
        canonical_symbols = self._validate_universe(symbols)
        start_date = self._as_market_date(start)
        end_date = self._as_market_date(end)
        if end_date <= start_date:
            raise IFindDateInputError("iFinD end date must be after start date")

        required_dates = sorted({
            timestamp.date()
            for frame in bars_by_symbol.values()
            for timestamp in frame.index
        })
        payload = self._client.fetch_daily_market_rules(
            canonical_symbols,
            start_date,
            end_date,
        )
        price_tick = (
            self.profile.transaction_cost_profile.price_tick
            if self.profile.transaction_cost_profile is not None
            else 0.01
        )
        return self._market_rule_adapter(
            payload,
            expected_symbols=canonical_symbols,
            required_dates=required_dates,
            bars_by_symbol=bars_by_symbol,
            fetch_basic_status=self._client.fetch_basic_market_status,
            price_tick=price_tick,
        )

    def _validate_universe(self, symbols: Sequence[str]) -> tuple[str, ...]:
        universe = self.profile.universe
        if isinstance(symbols, (str, bytes)):
            raise IFindUniverseError(
                f"iFinD provider requires the complete {universe} universe"
            )
        try:
            requested = tuple(symbols)
        except TypeError:
            raise IFindUniverseError(
                f"iFinD provider requires the complete {universe} universe"
            ) from None

        expected = self.profile.symbols
        valid = (
            len(requested) == len(expected)
            and all(isinstance(symbol, str) for symbol in requested)
            and set(requested) == set(expected)
        )
        if not valid:
            raise IFindUniverseError(
                f"iFinD provider requires the complete {universe} universe"
            )
        return expected

    @staticmethod
    def _as_market_date(value: DateInput) -> date:
        if isinstance(value, datetime):
            if value.tzinfo is not None and value.utcoffset() is not None:
                value = value.astimezone(_MARKET_TIMEZONE)
            return value.date()
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                parsed = date.fromisoformat(value)
            except ValueError:
                raise IFindDateInputError(
                    "iFinD dates must use YYYY-MM-DD"
                ) from None
            if parsed.isoformat() == value:
                return parsed
        raise IFindDateInputError("iFinD dates must use YYYY-MM-DD")
