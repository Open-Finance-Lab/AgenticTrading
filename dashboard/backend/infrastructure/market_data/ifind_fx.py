"""Infer historical USD/CNY rates from iFinD dual-currency daily closes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
import logging
import math
from statistics import median
from typing import Any

from .ifind_client import IFindHttpClient


logger = logging.getLogger(__name__)

LOOKBACK_DAYS = 14
MIN_SYMBOLS_PER_DAY = 2
MAX_RELATIVE_DEVIATION = 0.0025
MIN_CNY_PER_USD = 1.0
MAX_CNY_PER_USD = 20.0
# Fraction of candidate dates that may be skipped before the gaps stop looking
# like ordinary trading halts and start looking like a broken contract.
MAX_SKIPPED_DATE_RATIO = 0.5


class IFindFxError(ValueError):
    """Base error for sanitized iFinD historical FX inference failures."""


class IFindFxResponseError(IFindFxError):
    """Raised when a daily-close response violates the documented schema."""


class IFindFxValidationError(IFindFxError):
    """Raised when closes cannot produce a trustworthy USD/CNY rate."""


@dataclass
class FxInferenceStats:
    """Counts of everything dropped while inferring the rate series.

    Holds counts only — never a raw close, symbol price or response body — so
    it stays safe to log and to embed in an operator-facing error message.
    """

    candidate_dates: int = 0
    resolved_dates: int = 0
    blank_closes: int = 0
    unusable_closes: int = 0
    dates_below_min_symbols: int = 0
    dates_out_of_range: int = 0
    dates_inconsistent: int = 0
    outliers_dropped_by_symbol: dict[str, int] = field(default_factory=dict)
    symbols_without_usable_closes: list[str] = field(default_factory=list)

    @property
    def skipped_dates(self) -> int:
        return (
            self.dates_below_min_symbols
            + self.dates_out_of_range
            + self.dates_inconsistent
        )

    def summary(self) -> str:
        parts = [
            f"resolved={self.resolved_dates}/{self.candidate_dates} dates",
            f"blank_closes={self.blank_closes}",
            f"unusable_closes={self.unusable_closes}",
            f"skipped_below_min_symbols={self.dates_below_min_symbols}",
            f"skipped_out_of_range={self.dates_out_of_range}",
            f"skipped_inconsistent={self.dates_inconsistent}",
        ]
        if self.outliers_dropped_by_symbol:
            dropped = ", ".join(
                f"{symbol}={count}"
                for symbol, count in sorted(self.outliers_dropped_by_symbol.items())
            )
            parts.append(f"outliers_dropped={{{dropped}}}")
        if self.symbols_without_usable_closes:
            parts.append(
                "symbols_without_usable_closes="
                f"{sorted(self.symbols_without_usable_closes)}"
            )
        return "; ".join(parts)


class IFindHistoricalFxProvider:
    """Recover iFinD's daily CNY-per-USD conversion from paired stock closes.

    A-share names are routinely suspended (停牌) for a day or more, and the
    daily-close request deliberately asks iFinD to leave those days blank, so
    a gap in one symbol is expected data — not a broken response. Gaps, single
    bad quotes and days that cannot be corroborated are therefore dropped from
    the series rather than aborting the run; the caller decides whether what
    survives actually covers its market window (``CurrencyContext.rate_at``).
    Only a wholesale failure — no usable dates at all, or more than
    ``MAX_SKIPPED_DATE_RATIO`` of them dropped — is fatal, because that is the
    signature of a contract change rather than a trading halt.
    """

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
        # Never below 1: the corroboration loop takes a median of the survivors,
        # and median([]) raises.
        self._min_symbols_per_day = max(int(min_symbols_per_day), 1)
        self._max_relative_deviation = max_relative_deviation
        self.last_stats: FxInferenceStats | None = None

    def fetch_usd_cny(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
    ) -> dict[date, float]:
        """Return sorted daily rates where one USD equals the returned CNY value."""
        normalized_symbols = tuple(symbols)
        request_start = start - timedelta(days=self._lookback_days)
        stats = FxInferenceStats()
        self.last_stats = stats

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
            stats,
        )
        usd = _parse_daily_closes(
            usd_payload,
            normalized_symbols,
            request_start,
            end,
            "MHB",
            stats,
        )
        stats.symbols_without_usable_closes = [
            symbol
            for symbol in normalized_symbols
            if not rmb.get(symbol) or not usd.get(symbol)
        ]

        available_dates = sorted(
            {
                observed_date
                for symbol_values in (*rmb.values(), *usd.values())
                for observed_date in symbol_values
            }
        )
        stats.candidate_dates = len(available_dates)

        rates: dict[date, float] = {}
        for observed_date in available_dates:
            observations: list[tuple[str, float]] = []
            for symbol in normalized_symbols:
                rmb_close = rmb.get(symbol, {}).get(observed_date)
                usd_close = usd.get(symbol, {}).get(observed_date)
                if rmb_close is None or usd_close is None:
                    continue
                observations.append((symbol, rmb_close / usd_close))

            daily_rate = self._resolve_daily_rate(observations, stats)
            if daily_rate is None:
                continue
            rates[observed_date] = daily_rate

        stats.resolved_dates = len(rates)
        self._report(stats)

        if not rates:
            raise IFindFxValidationError(
                "iFinD historical FX returned no usable daily rates "
                f"({stats.summary()})"
            )
        if (
            stats.candidate_dates
            and stats.skipped_dates > stats.candidate_dates * MAX_SKIPPED_DATE_RATIO
        ):
            raise IFindFxValidationError(
                "iFinD historical FX dropped most candidate dates, which points "
                f"at a changed daily-close contract rather than trading halts "
                f"({stats.summary()})"
            )
        return rates

    def _resolve_daily_rate(
        self,
        observations: Sequence[tuple[str, float]],
        stats: FxInferenceStats,
    ) -> float | None:
        """Return one corroborated rate for a date, or ``None`` to skip it.

        Discards the single worst-disagreeing symbol at a time rather than
        failing the date outright: one stale or mis-converted quote should not
        cost the day when the remaining symbols still agree with each other.
        """
        kept = list(observations)
        if len(kept) < self._min_symbols_per_day:
            stats.dates_below_min_symbols += 1
            return None

        while len(kept) >= self._min_symbols_per_day:
            daily_rate = float(median(value for _symbol, value in kept))
            if (
                not math.isfinite(daily_rate)
                or not MIN_CNY_PER_USD <= daily_rate <= MAX_CNY_PER_USD
            ):
                stats.dates_out_of_range += 1
                return None

            deviations = [
                (abs(value / daily_rate - 1.0), symbol) for symbol, value in kept
            ]
            worst_deviation, worst_symbol = max(deviations)
            if worst_deviation <= self._max_relative_deviation:
                return daily_rate

            worst_index = next(
                index
                for index, (symbol, _value) in enumerate(kept)
                if symbol == worst_symbol
            )
            kept.pop(worst_index)
            stats.outliers_dropped_by_symbol[worst_symbol] = (
                stats.outliers_dropped_by_symbol.get(worst_symbol, 0) + 1
            )

        stats.dates_inconsistent += 1
        return None

    def _report(self, stats: FxInferenceStats) -> None:
        """Make dropped data visible; a silent skip is the failure mode here.

        Escalation is deliberate. A symbol that contributed nothing all window,
        or a majority of dates dropped, reads the same whether the symbol was
        suspended or the upstream field was renamed — that ambiguity is what
        earns an ERROR. Ordinary halts stay at INFO so the real signal is not
        buried under them.
        """
        wholesale = bool(
            stats.symbols_without_usable_closes
            or (
                stats.candidate_dates
                and stats.skipped_dates
                > stats.candidate_dates * MAX_SKIPPED_DATE_RATIO
            )
        )
        if wholesale:
            logger.error(
                "iFinD historical FX dropped data wholesale; treat as a "
                "possible daily-close contract change: %s",
                stats.summary(),
            )
        elif (
            stats.skipped_dates
            or stats.unusable_closes
            or stats.outliers_dropped_by_symbol
        ):
            logger.warning("iFinD historical FX dropped some data: %s", stats.summary())
        elif stats.blank_closes:
            logger.info("iFinD historical FX: %s", stats.summary())


def _parse_daily_closes(
    payload: Any,
    expected_symbols: Sequence[str],
    start: date,
    end: date,
    currency: str,
    stats: FxInferenceStats,
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
            # The request asks for Fill=Blank, so a halted symbol legitimately
            # reports nothing for that day. Skip the datum; the day survives if
            # the other symbols still corroborate a rate.
            if _is_blank_close(raw_close):
                stats.blank_closes += 1
                continue
            close = _coerce_close(raw_close)
            if close is None:
                stats.unusable_closes += 1
                continue
            values[observed_date] = close
        parsed[symbol] = values
    return parsed


def _is_blank_close(raw_close: object) -> bool:
    if raw_close is None:
        return True
    return isinstance(raw_close, str) and not raw_close.strip()


def _coerce_close(raw_close: object) -> float | None:
    """Return a usable positive close, or ``None`` when the datum is unusable."""
    if isinstance(raw_close, bool):
        return None
    try:
        close = float(raw_close)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(close) or close <= 0:
        return None
    return close


def _parse_date(raw_value: object) -> date:
    if not isinstance(raw_value, str):
        raise IFindFxValidationError("iFinD daily-close date must be a string")
    try:
        return date.fromisoformat(raw_value)
    except ValueError:
        raise IFindFxValidationError(
            "iFinD daily-close date must use YYYY-MM-DD"
        ) from None
