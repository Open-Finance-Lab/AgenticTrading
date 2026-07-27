"""Convert official iFinD table responses into validated ATL OHLCV frames."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from typing import Any

import numpy as np
import pandas as pd


OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
PRICE_COLUMNS = ("open", "high", "low", "close")
MARKET_TIMEZONE = "Asia/Shanghai"
_MORNING_SESSION = (time(9, 30), time(11, 30))
_AFTERNOON_SESSION = (time(13, 0), time(15, 0))


class IFindAdapterError(ValueError):
    """Base error for sanitized iFinD response conversion failures."""


class IFindBusinessResponseError(IFindAdapterError):
    """Raised when a response contains a non-zero numeric business code."""

    def __init__(self, message: str, errorcode: int):
        super().__init__(message)
        self.errorcode = errorcode


class IFindResponseSchemaError(IFindAdapterError):
    """Raised when a response does not match the official tables structure."""


class IFindBarValidationError(IFindAdapterError):
    """Raised when timestamps or OHLCV values violate market invariants."""


def response_to_frames(
    payload: Mapping[str, object],
    expected_symbols: Sequence[str],
    start: datetime | date,
    end: datetime | date,
    min_bars: int = 50,
) -> dict[str, pd.DataFrame]:
    """Convert one official iFinD response into symbol-keyed OHLCV frames."""
    symbols = _validate_expected_symbols(expected_symbols)
    start_timestamp = _normalize_boundary(start, "start")
    end_timestamp = _normalize_boundary(end, "end")
    if end_timestamp <= start_timestamp:
        raise IFindBarValidationError("iFinD date window requires end after start")
    if isinstance(min_bars, bool) or not isinstance(min_bars, int) or min_bars < 0:
        raise IFindBarValidationError("min_bars must be a non-negative integer")

    tables = _validate_top_level(payload)
    entries = _index_tables(tables)
    returned_symbols = set(entries)
    expected_set = set(symbols)
    if returned_symbols != expected_set:
        missing = [symbol for symbol in symbols if symbol not in returned_symbols]
        unexpected_count = len(returned_symbols - expected_set)
        raise IFindResponseSchemaError(
            "iFinD symbol set mismatch "
            f"missing={missing!r} unexpected_count={unexpected_count}"
        )

    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frames[symbol] = _table_to_frame(
            entries[symbol],
            symbol,
            start_timestamp,
            end_timestamp,
            min_bars,
        )
    return frames


def _validate_expected_symbols(expected_symbols: Sequence[str]) -> tuple[str, ...]:
    if isinstance(expected_symbols, (str, bytes)):
        raise IFindResponseSchemaError("expected_symbols must be a sequence")
    symbols = tuple(expected_symbols)
    if not symbols or any(not isinstance(symbol, str) or not symbol for symbol in symbols):
        raise IFindResponseSchemaError(
            "expected_symbols must contain non-empty strings"
        )
    if len(set(symbols)) != len(symbols):
        raise IFindResponseSchemaError("expected_symbols contains duplicates")
    return symbols


def _normalize_boundary(value: datetime | date, name: str) -> pd.Timestamp:
    if not isinstance(value, (datetime, date)):
        raise IFindBarValidationError(f"{name} must be a date or datetime")
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise IFindBarValidationError(f"{name} must be a valid date or datetime")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(MARKET_TIMEZONE)
    return timestamp.tz_convert(MARKET_TIMEZONE)


def _validate_top_level(payload: Any) -> list[object]:
    if not isinstance(payload, Mapping):
        raise IFindResponseSchemaError("iFinD response must be a JSON object")
    if "errorcode" not in payload:
        raise IFindResponseSchemaError("iFinD response is missing errorcode")

    errorcode = payload["errorcode"]
    if isinstance(errorcode, bool) or not isinstance(errorcode, int):
        raise IFindResponseSchemaError("iFinD errorcode must be an integer")
    if errorcode != 0:
        raise IFindBusinessResponseError(
            f"iFinD business response failed errorcode={errorcode}",
            errorcode,
        )

    tables = payload.get("tables")
    if not isinstance(tables, list):
        raise IFindResponseSchemaError("iFinD response tables must be an array")
    return tables


def _index_tables(tables: list[object]) -> dict[str, Mapping[str, object]]:
    entries: dict[str, Mapping[str, object]] = {}
    for entry in tables:
        if not isinstance(entry, Mapping):
            raise IFindResponseSchemaError("each iFinD table must be an object")
        symbol = entry.get("thscode")
        if not isinstance(symbol, str) or not symbol:
            raise IFindResponseSchemaError(
                "each iFinD table must contain a string thscode"
            )
        if symbol in entries:
            raise IFindResponseSchemaError("iFinD response has a duplicate table")
        entries[symbol] = entry
    return entries


def _table_to_frame(
    entry: Mapping[str, object],
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_bars: int,
) -> pd.DataFrame:
    raw_times = _require_array(entry.get("time"), symbol, "time")
    raw_table = entry.get("table")
    if not isinstance(raw_table, Mapping):
        raise IFindResponseSchemaError(
            f"iFinD table for symbol={symbol} must contain a table object"
        )

    raw_columns: dict[str, list[object]] = {}
    for field in OHLCV_COLUMNS:
        raw_columns[field] = _require_array(raw_table.get(field), symbol, field)

    lengths = {len(raw_times), *(len(values) for values in raw_columns.values())}
    if len(lengths) != 1:
        raise IFindResponseSchemaError(
            f"iFinD array lengths differ for symbol={symbol}"
        )

    index = _parse_timestamps(raw_times, symbol)
    _validate_timestamps(index, symbol, start, end)

    values = {
        field: _numeric_values(raw_columns[field], symbol, field, index)
        for field in OHLCV_COLUMNS
    }
    _validate_prices(values, symbol, index)

    frame = pd.DataFrame(values, index=index).loc[:, list(OHLCV_COLUMNS)]
    if len(frame) < min_bars:
        raise IFindBarValidationError(
            f"symbol={symbol} has {len(frame)} valid bars; minimum={min_bars}"
        )
    return frame


def _require_array(value: object, symbol: str, field: str) -> list[object]:
    if not isinstance(value, list):
        raise IFindResponseSchemaError(
            f"iFinD field={field} for symbol={symbol} must be an array"
        )
    return value


def _parse_timestamps(values: list[object], symbol: str) -> pd.DatetimeIndex:
    timestamps: list[pd.Timestamp] = []
    for row, raw_value in enumerate(values):
        if not isinstance(raw_value, str):
            raise IFindBarValidationError(
                f"invalid timestamp for symbol={symbol} row={row}"
            )
        try:
            timestamp = pd.Timestamp(raw_value)
        except (TypeError, ValueError):
            raise IFindBarValidationError(
                f"invalid timestamp for symbol={symbol} row={row}"
            ) from None
        if pd.isna(timestamp):
            raise IFindBarValidationError(
                f"invalid timestamp for symbol={symbol} row={row}"
            )
        if timestamp.tzinfo is not None:
            raise IFindBarValidationError(
                "iFinD timestamp must use LocalTime without an offset "
                f"for symbol={symbol} row={row}"
            )
        timestamps.append(timestamp.tz_localize(MARKET_TIMEZONE))

    if timestamps:
        index = pd.DatetimeIndex(timestamps, name="timestamp")
    else:
        index = pd.DatetimeIndex(
            [], name="timestamp", tz=MARKET_TIMEZONE
        )
    if index.has_duplicates:
        raise IFindBarValidationError(
            f"duplicate timestamp in iFinD bars for symbol={symbol}"
        )
    if not index.is_monotonic_increasing:
        raise IFindBarValidationError(
            f"timestamps must be strictly increasing for symbol={symbol}"
        )
    return index


def _validate_timestamps(
    index: pd.DatetimeIndex,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    for row, timestamp in enumerate(index):
        if timestamp < start or timestamp >= end:
            raise IFindBarValidationError(
                "timestamp outside requested window "
                f"for symbol={symbol} row={row}"
            )
        if timestamp.weekday() >= 5 or not _in_trading_session(timestamp.time()):
            raise IFindBarValidationError(
                "timestamp outside A-share trading session "
                f"for symbol={symbol} row={row}"
            )


def _in_trading_session(value: time) -> bool:
    morning = _MORNING_SESSION[0] <= value <= _MORNING_SESSION[1]
    afternoon = _AFTERNOON_SESSION[0] <= value <= _AFTERNOON_SESSION[1]
    return morning or afternoon


def _numeric_values(
    raw_values: list[object],
    symbol: str,
    field: str,
    index: pd.DatetimeIndex,
) -> np.ndarray:
    if any(isinstance(value, bool) for value in raw_values):
        raise IFindBarValidationError(
            f"invalid {field} for symbol={symbol} row=0"
        )
    converted = pd.to_numeric(pd.Series(raw_values, dtype="object"), errors="coerce")
    values = converted.to_numpy(dtype=float)
    invalid_rows = np.flatnonzero(~np.isfinite(values))
    if invalid_rows.size:
        row = int(invalid_rows[0])
        raise IFindBarValidationError(
            f"invalid {field} for {_bar_context(symbol, row, index)}"
        )
    return values


def _validate_prices(
    values: Mapping[str, np.ndarray],
    symbol: str,
    index: pd.DatetimeIndex,
) -> None:
    for field in PRICE_COLUMNS:
        invalid_rows = np.flatnonzero(values[field] <= 0)
        if invalid_rows.size:
            row = int(invalid_rows[0])
            raise IFindBarValidationError(
                f"invalid {field} price for {_bar_context(symbol, row, index)}"
            )

    invalid_high = np.flatnonzero(
        values["high"] < np.maximum(values["open"], values["close"])
    )
    if invalid_high.size:
        row = int(invalid_high[0])
        raise IFindBarValidationError(
            "invalid high price; high must cover open and close for "
            f"{_bar_context(symbol, row, index)}"
        )

    invalid_low = np.flatnonzero(
        values["low"] > np.minimum(values["open"], values["close"])
    )
    if invalid_low.size:
        row = int(invalid_low[0])
        raise IFindBarValidationError(
            "invalid low price; low must cover open and close for "
            f"{_bar_context(symbol, row, index)}"
        )

    invalid_volume = np.flatnonzero(values["volume"] < 0)
    if invalid_volume.size:
        row = int(invalid_volume[0])
        raise IFindBarValidationError(
            f"invalid volume for {_bar_context(symbol, row, index)}"
        )


def _bar_context(symbol: str, row: int, index: pd.DatetimeIndex) -> str:
    timestamp = index[row].isoformat() if row < len(index) else "unavailable"
    return f"symbol={symbol} row={row} timestamp={timestamp}"
