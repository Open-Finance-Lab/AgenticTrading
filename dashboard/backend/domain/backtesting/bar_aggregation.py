"""Session-aware aggregation of source bars into decision bars.

The market-data provider returns bars at the configured source resolution.  A
strategy must only see a completed decision bar, so this module labels each
bucket at its *right* edge.  For example, US 5-minute bars from 09:30 through
10:25 become the 10:30 decision bar.  The next source bar, opening at 10:30,
can therefore be used as the execution bar without look-ahead.

This is intentionally independent of any provider SDK.  It also avoids a
plain pandas ``resample`` because exchange sessions do not begin at midnight
and some markets have a lunch break.
"""

from __future__ import annotations

from datetime import time
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd

from dashboard.backend.infrastructure.market_data.frequency import (
    normalize_bar_timeframe,
    timeframe_minutes,
)


class BarAggregationError(ValueError):
    """Raised when source bars cannot be safely aggregated."""


_QUALITY_COUNT_COLUMNS = (
    "missing_source_bars",
    "duplicate_source_bars",
    "off_grid_source_bars",
    "invalid_source_bars",
)


def _session_windows(market: str) -> tuple[tuple[time, time], ...]:
    canonical = str(market or "US").strip().upper()
    if canonical == "CN":
        return ((time(9, 30), time(11, 30)), (time(13, 0), time(15, 0)))
    return ((time(9, 30), time(16, 0)),)


def _as_local_index(frame: pd.DataFrame, timezone: str) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise BarAggregationError("source bars must use a DatetimeIndex")
    result = frame.copy()
    if result.index.tz is None:
        result.index = result.index.tz_localize(timezone)
    else:
        result.index = result.index.tz_convert(timezone)
    return result.sort_index()


def _session_for_timestamp(
    timestamp: pd.Timestamp,
    windows: Iterable[tuple[time, time]],
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    local_date = timestamp.normalize()
    for start_time, end_time in windows:
        start = local_date + pd.Timedelta(
            hours=start_time.hour, minutes=start_time.minute
        )
        end = local_date + pd.Timedelta(
            hours=end_time.hour, minutes=end_time.minute
        )
        if start <= timestamp < end:
            return start, end
    return None


def _weighted_vwap(group: pd.DataFrame, close: float) -> float:
    if "vwap" not in group.columns:
        return close
    values = pd.to_numeric(group["vwap"], errors="coerce")
    volumes = pd.to_numeric(group["volume"], errors="coerce").fillna(0.0)
    valid = values.notna() & volumes.gt(0)
    if valid.any() and float(volumes[valid].sum()) > 0:
        return float((values[valid] * volumes[valid]).sum() / volumes[valid].sum())
    return close


def aggregate_bars(
    frame: pd.DataFrame,
    *,
    source_timeframe: str,
    decision_timeframe: str = "60m",
    market: str = "US",
    timezone: str = "US/Eastern",
) -> pd.DataFrame:
    """Aggregate one symbol's source bars into completed session bars.

    The returned index is timezone-aware UTC, matching the canonical provider
    boundary.  Incomplete or missing source bars are not synthesized; quality
    columns make the gap visible to callers.
    """
    source = normalize_bar_timeframe(source_timeframe)
    decision = normalize_bar_timeframe(decision_timeframe)
    source_minutes = timeframe_minutes(source)
    decision_minutes = timeframe_minutes(decision)
    if source_minutes >= decision_minutes:
        raise BarAggregationError(
            "aggregation requires source_timeframe to be finer than "
            "decision_timeframe"
        )
    required = ("open", "high", "low", "close", "volume")
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise BarAggregationError(
            f"source bars are missing required columns: {', '.join(missing)}"
        )
    if frame.empty:
        return frame.copy()

    local = _as_local_index(frame, timezone)
    windows = _session_windows(market)
    buckets: dict[pd.Timestamp, list[pd.Series]] = {}
    bucket_ends: dict[pd.Timestamp, pd.Timestamp] = {}
    for timestamp, row in local.iterrows():
        session = _session_for_timestamp(timestamp, windows)
        if session is None:
            continue
        session_start, session_end = session
        elapsed_minutes = int((timestamp - session_start).total_seconds() // 60)
        offset_minutes = (elapsed_minutes // decision_minutes) * decision_minutes
        bucket_start = session_start + pd.Timedelta(minutes=offset_minutes)
        bucket_end = min(
            bucket_start + pd.Timedelta(minutes=decision_minutes), session_end
        )
        # A source bar can only belong to a decision bucket that has not ended.
        if bucket_start >= bucket_end:
            continue
        buckets.setdefault(bucket_start, []).append(row)
        bucket_ends[bucket_start] = bucket_end

    records: list[dict] = []
    for bucket_start in sorted(buckets):
        group = pd.DataFrame(buckets[bucket_start])
        group = group.sort_index()
        bucket_end = bucket_ends[bucket_start]
        expected = int(
            (bucket_end - bucket_start).total_seconds() // (source_minutes * 60)
        )
        expected_index = pd.date_range(
            bucket_start,
            periods=expected,
            freq=f"{source_minutes}min",
        )
        actual_index = pd.DatetimeIndex(group.index)
        unique_actual_index = actual_index.unique()
        missing_source_bars = len(expected_index.difference(unique_actual_index))
        duplicate_source_bars = len(actual_index) - len(unique_actual_index)
        off_grid_source_bars = len(unique_actual_index.difference(expected_index))
        required_values = group.loc[:, list(required)].apply(
            pd.to_numeric, errors="coerce"
        )
        invalid_source_bars = int(
            (~np.isfinite(required_values.to_numpy(dtype=float)).all(axis=1)).sum()
        )
        is_complete = not any(
            (
                missing_source_bars,
                duplicate_source_bars,
                off_grid_source_bars,
                invalid_source_bars,
            )
        )
        volume = float(pd.to_numeric(group["volume"], errors="coerce").fillna(0).sum())
        close = float(group["close"].iloc[-1])
        record = {
            "timestamp": bucket_end.tz_convert("UTC"),
            "open": float(group["open"].iloc[0]),
            "high": float(pd.to_numeric(group["high"], errors="coerce").max()),
            "low": float(pd.to_numeric(group["low"], errors="coerce").min()),
            "close": close,
            "volume": volume,
            "source_bar_count": int(len(group)),
            "expected_source_bars": expected,
            "missing_source_bars": int(missing_source_bars),
            "duplicate_source_bars": int(duplicate_source_bars),
            "off_grid_source_bars": int(off_grid_source_bars),
            "invalid_source_bars": invalid_source_bars,
            "is_complete": is_complete,
            "has_gap": not is_complete,
        }
        if "trade_count" in group.columns:
            record["trade_count"] = float(
                pd.to_numeric(group["trade_count"], errors="coerce")
                .fillna(0)
                .sum()
            )
        if "vwap" in group.columns:
            record["vwap"] = _weighted_vwap(group, close)
        records.append(record)

    if not records:
        columns = ["open", "high", "low", "close", "volume"]
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], tz="UTC"))

    result = pd.DataFrame.from_records(records).set_index("timestamp").sort_index()
    result.attrs.update(dict(getattr(frame, "attrs", {}) or {}))
    result.attrs.update(
        {
            "aggregation_source_timeframe": source,
            "aggregation_decision_timeframe": decision,
            "aggregation_market": str(market or "US").strip().upper(),
            "aggregation_timezone": timezone,
        }
    )
    return result


def aggregate_bars_by_symbol(
    bars_by_symbol: Mapping[str, pd.DataFrame],
    *,
    source_timeframe: str,
    decision_timeframe: str = "60m",
    market: str = "US",
    timezone: str = "US/Eastern",
) -> Dict[str, pd.DataFrame]:
    """Aggregate each symbol independently, preserving the symbol mapping."""
    return {
        symbol: aggregate_bars(
            frame,
            source_timeframe=source_timeframe,
            decision_timeframe=decision_timeframe,
            market=market,
            timezone=timezone,
        )
        for symbol, frame in bars_by_symbol.items()
    }


def summarize_aggregation_quality(
    bars_by_symbol: Mapping[str, pd.DataFrame],
) -> Dict[str, Any]:
    """Return a JSON-safe audit summary before incomplete bars are dropped.

    Counts are symbol-bar counts: the same decision timestamp contributes once
    for each symbol.  Keeping this summary before filtering makes a completed
    run distinguishable from one that silently lost source observations.
    """

    summary: Dict[str, Any] = {
        "policy": "drop_incomplete_decision_bars",
        "decision_timestamp_min_symbol_coverage": 0.8,
        "total_decision_bars": 0,
        "usable_decision_bars": 0,
        "dropped_decision_bars": 0,
        **{column: 0 for column in _QUALITY_COUNT_COLUMNS},
        "symbols": {},
    }
    for symbol, frame in bars_by_symbol.items():
        total = int(len(frame))
        if "is_complete" in frame.columns:
            usable = int(frame["is_complete"].fillna(False).astype(bool).sum())
        else:
            usable = total
        symbol_summary: Dict[str, Any] = {
            "total_decision_bars": total,
            "usable_decision_bars": usable,
            "dropped_decision_bars": total - usable,
        }
        for column in _QUALITY_COUNT_COLUMNS:
            value = (
                int(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())
                if column in frame.columns
                else 0
            )
            symbol_summary[column] = value
            summary[column] += value
        summary["symbols"][symbol] = symbol_summary
        summary["total_decision_bars"] += total
        summary["usable_decision_bars"] += usable
        summary["dropped_decision_bars"] += total - usable
    return summary
