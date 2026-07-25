"""Infer historical USD/CNY rates from iFinD dual-currency daily closes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, timedelta
import math
from statistics import median
from typing import Any

from .ifind_client import IFindHttpClient


LOOKBACK_DAYS = 14
MIN_SYMBOLS_PER_DAY = 2
MAX_RELATIVE_DEVIATION = 0.0025
MIN_CNY_PER_USD = 1.0
MAX_CNY_PER_USD = 20.0


class IFindFxError(ValueError):
    """Base error for sanitized iFinD historical FX inference failures."""


class IFindFxResponseError(IFindFxError):
    """Raised when a daily-close response violates the documented schema."""


class IFindFxValidationError(IFindFxError):
    """Raised when closes cannot produce a trustworthy USD/CNY rate."""


class IFindHistoricalFxProvider:
    """Recover iFinD's daily CNY-per-USD conversion from paired stock closes."""

    def __init__(
        self,
        *,
        client: IFindHttpClient | None = None,
        lookback_days: int = LOOKBACK_DAYS,
        min_symbols_per_day: int = MIN_SYMBOLS_PER_DAY,
        max_relative_deviation: float = MAX_RELATIVE_DEVIATION,
    ) -> None:
        self._client = client if client is not None else IFindHttpClient()
        self._lookback_days = lookback_days
        self._min_symbols_per_day = min_symbols_per_day
        self._max_relative_deviation = max_relative_deviation

    def fetch_usd_cny(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
    ) -> dict[date, float]:
        """Return sorted daily rates where one USD equals the returned CNY value."""
        normalized_symbols = tuple(symbols)
        request_start = start - timedelta(days=self._lookback_days)
        rmb_payload = self._client.fetch_daily_closes(
            normalized_symbols,
            request_start,
            end,
            currency="RMB",
        )
        usd_payload = self._client.fetch_daily_closes(
            normalized_symbols,
            request_start,
            end,
            currency="MHB",
        )
        rmb = _parse_daily_closes(
            rmb_payload,
            normalized_symbols,
            request_start,
            end,
            "RMB",
        )
        usd = _parse_daily_closes(
            usd_payload,
            normalized_symbols,
            request_start,
            end,
            "MHB",
        )

        available_dates = sorted(
            {
                observed_date
                for symbol_values in (*rmb.values(), *usd.values())
                for observed_date in symbol_values
            }
        )
        rates: dict[date, float] = {}
        for observed_date in available_dates:
            observations = []
            for symbol in normalized_symbols:
                rmb_close = rmb.get(symbol, {}).get(observed_date)
                usd_close = usd.get(symbol, {}).get(observed_date)
                if rmb_close is None or usd_close is None:
                    continue
                observations.append(rmb_close / usd_close)

            if len(observations) < self._min_symbols_per_day:
                raise IFindFxValidationError(
                    "iFinD historical FX requires at least "
                    f"{self._min_symbols_per_day} matched symbols per date"
                )

            daily_rate = float(median(observations))
            if (
                not math.isfinite(daily_rate)
                or not MIN_CNY_PER_USD <= daily_rate <= MAX_CNY_PER_USD
            ):
                raise IFindFxValidationError(
                    "iFinD historical FX rate has an invalid direction or value"
                )
            max_deviation = max(
                abs(observation / daily_rate - 1.0)
                for observation in observations
            )
            if max_deviation > self._max_relative_deviation:
                raise IFindFxValidationError(
                    "iFinD dual-currency symbol rates disagree"
                )
            rates[observed_date] = daily_rate

        if not rates:
            raise IFindFxValidationError(
                "iFinD historical FX returned no usable daily rates"
            )
        return rates


def _parse_daily_closes(
    payload: Any,
    expected_symbols: Sequence[str],
    start: date,
    end: date,
    currency: str,
) -> dict[str, dict[date, float]]:
    if not isinstance(payload, Mapping):
        raise IFindFxResponseError("iFinD daily-close response must be an object")
    errorcode = payload.get("errorcode")
    if isinstance(errorcode, bool) or not isinstance(errorcode, int):
        raise IFindFxResponseError("iFinD daily-close errorcode must be an integer")
    if errorcode != 0:
        raise IFindFxResponseError(
            f"iFinD daily-close business response failed currency={currency}"
        )
    tables = payload.get("tables")
    if not isinstance(tables, list):
        raise IFindFxResponseError("iFinD daily-close tables must be an array")

    expected = set(expected_symbols)
    parsed: dict[str, dict[date, float]] = {}
    for entry in tables:
        if not isinstance(entry, Mapping):
            raise IFindFxResponseError("iFinD daily-close table must be an object")
        symbol = entry.get("thscode")
        if not isinstance(symbol, str) or symbol not in expected:
            raise IFindFxResponseError("iFinD daily-close returned an unexpected symbol")
        if symbol in parsed:
            raise IFindFxResponseError("iFinD daily-close returned a duplicate symbol")

        raw_times = entry.get("time")
        raw_table = entry.get("table")
        raw_closes = raw_table.get("close") if isinstance(raw_table, Mapping) else None
        if not isinstance(raw_times, list) or not isinstance(raw_closes, list):
            raise IFindFxResponseError("iFinD daily-close fields must be arrays")
        if len(raw_times) != len(raw_closes):
            raise IFindFxResponseError("iFinD daily-close array length mismatch")

        values: dict[date, float] = {}
        for raw_time, raw_close in zip(raw_times, raw_closes):
            observed_date = _parse_date(raw_time)
            if not start <= observed_date < end:
                raise IFindFxValidationError(
                    "iFinD daily-close date is outside the requested window"
                )
            if observed_date in values:
                raise IFindFxValidationError(
                    "iFinD daily-close response contains a duplicate date"
                )
            if isinstance(raw_close, bool):
                raise IFindFxValidationError(
                    "iFinD daily-close value must be positive and finite"
                )
            try:
                close = float(raw_close)
            except (TypeError, ValueError, OverflowError):
                raise IFindFxValidationError(
                    "iFinD daily-close value must be positive and finite"
                ) from None
            if not math.isfinite(close) or close <= 0:
                raise IFindFxValidationError(
                    "iFinD daily-close value must be positive and finite"
                )
            values[observed_date] = close
        parsed[symbol] = values
    return parsed


def _parse_date(raw_value: object) -> date:
    if not isinstance(raw_value, str):
        raise IFindFxValidationError("iFinD daily-close date must be a string")
    try:
        return date.fromisoformat(raw_value)
    except ValueError:
        raise IFindFxValidationError(
            "iFinD daily-close date must use YYYY-MM-DD"
        ) from None
