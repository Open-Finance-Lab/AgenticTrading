"""Fixed-universe iFinD A-share provider for ATL backtests."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta
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
# A-share cash equities trade four 60m bars a session: 09:30-11:30 and
# 13:00-15:00. The floor below is expressed against that, not as a flat count.
_SESSIONS_PER_TRADING_DAY = 4
# Half the weekday-derived expectation. It has to absorb exchange holidays --
# Qingming and Labour Day both fall inside the windows this universe ships
# with -- and a weekday count cannot see them, so a stricter fraction would
# refuse good data every May. Half still catches the failure this guard exists
# for: an empty, one-day, or wholesale-truncated upstream reply.
#
# ⚠ Half is calibrated against the one- and two-day closures, and it does NOT
# cover the week-long ones. National Day (Oct 1-7) and Spring Festival each
# shut the exchange for most of a trading week, so a window that is legal
# under MAX_BACKTEST_DAYS and whose data is perfectly good can still land
# under this floor and be refused as incomplete -- a false refusal, not a
# missed detection, so it fails in the safe direction and says so loudly
# rather than charting a short curve as real. The honest fix is a CN trading
# calendar, which this module deliberately does not carry: inventing one here
# would put a second, unversioned holiday table in the codebase next to the
# exchange's own. Lowering the fraction instead would trade a visible false
# refusal for a silent acceptance of a genuinely truncated reply, which is
# the trade this guard exists to refuse. Tracked as a follow-up on #474.
_MINIMUM_BAR_COMPLETENESS = 0.5
# One full session day. Keeps a very short window from deriving a floor of
# zero, which would disable the check entirely.
_ABSOLUTE_MINIMUM_BARS = _SESSIONS_PER_TRADING_DAY


class IFindUniverseError(ValueError):
    """Raised when a caller does not request the selected registered universe."""


class IFindDateInputError(ValueError):
    """Raised when provider date inputs cannot form a half-open date window."""


def minimum_bars_for_window(start: date, end: date) -> int:
    """Return the fewest valid bars a complete response may hold.

    Derived from the requested half-open window rather than fixed, because a
    flat count is simultaneously a hidden minimum window (it was 50, i.e. ~13
    trading days) and unreachable once MAX_BACKTEST_DAYS fell to 14 -- a
    14-day window holds at most 10 weekdays, or 40 bars.

    Public because the same depth question is asked twice on this path, at two
    layers: here, against one upstream response, and again in
    ``domain/backtesting/engine.py``'s ``_validate_ifind_loaded_data``, against
    the assembled per-symbol frames. Both used to hardcode 50 independently,
    which is how the second one outlived the first being fixed.
    """
    weekdays = sum(
        1
        for offset in range((end - start).days)
        if (start + timedelta(days=offset)).weekday() < 5
    )
    expected = weekdays * _SESSIONS_PER_TRADING_DAY
    return max(_ABSOLUTE_MINIMUM_BARS, int(expected * _MINIMUM_BAR_COMPLETENESS))


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
            min_bars=minimum_bars_for_window(start_date, end_date),
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
