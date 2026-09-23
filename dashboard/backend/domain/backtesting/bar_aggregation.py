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

from bisect import bisect_left
from datetime import date, time
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple

import numpy as np
import pandas as pd

from dashboard.backend.infrastructure.market_data.frequency import (
    normalize_bar_timeframe,
    timeframe_minutes,
)
# Re-exported: the session bounds have one owner, and it is not this module.
from dashboard.backend.infrastructure.market_data.sessions import (
    FRAME_ATTR_OPEN_STAMPED_MINUTES,
    is_session_close,
    market_local,
    session_windows,
)


class BarAggregationError(ValueError):
    """Raised when source bars cannot be safely aggregated."""


_QUALITY_COUNT_COLUMNS = (
    "missing_source_bars",
    "duplicate_source_bars",
    "off_grid_source_bars",
    "invalid_source_bars",
)


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
        result = frame.copy()
        result.attrs.pop(FRAME_ATTR_OPEN_STAMPED_MINUTES, None)
        return result
    # Bucketing below reads a source stamp as its bar's OPEN (09:30-10:25 ->
    # the 10:30 bar), and ``plan_execution_fills`` then fills at the bar opening
    # on a decision's close. A close-stamped source would land one bar late in
    # every bucket with nothing to show for it, so it is refused, not guessed.
    stamped = frame.attrs.get(FRAME_ATTR_OPEN_STAMPED_MINUTES)
    if stamped != source_minutes:
        raise BarAggregationError(
            f"aggregation requires {source}-bars stamped at their open; the "
            f"frame is stamped {'at the close' if stamped is None else f'{stamped}m at the open'}"
        )

    local = _as_local_index(frame, timezone)
    windows = session_windows(market)
    # Walk the index, not the rows: iterrows builds a Series per source bar,
    # and the onboarding shape has ~16k of them (7 weekdays x 78 five-minute
    # bars x 30 symbols). That is the cheapest thing here to remove, not the
    # expensive one, and the comment first drafted for this block claimed the
    # opposite. Measured 2026-09-21 under cProfile: iterrows is ~0.8s of this
    # function's ~9.2s cumulative, while the per-bucket work carries the rest
    # (_weighted_vwap ~1.9s, group.apply(pd.to_numeric) ~1.3s, plus a
    # DataFrame and a pd.date_range built for each of 1,470 buckets). End to
    # end this buys ~11%. It also lands in `loading_bars`, not `indicators`:
    # aggregate_bars_by_symbol is called from load_data (engine.py:989). The
    # per-bucket arithmetic below -- quality counts, OHLCV, turnover, vwap -- is
    # byte-for-byte what it was; only how a row finds its bucket, and how that
    # bucket's end is looked up, changed.
    keep: list[bool] = []
    starts: list[pd.Timestamp] = []
    # One entry per bucket, not one per source bar. `bucket_end` is a pure
    # function of `bucket_start` -- min(start + decision, session_end) -- and a
    # start belongs to exactly one session, because sessions on a date do not
    # overlap and the date is part of the start. The previous shape built an
    # `ends` list and a `_bucket_end` column as long as the kept rows, then read
    # `.iloc[0]` of each group and discarded the rest.
    ends_by_start: dict[pd.Timestamp, pd.Timestamp] = {}
    for timestamp in local.index:
        session = _session_for_timestamp(timestamp, windows)
        if session is None:
            keep.append(False)
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
            keep.append(False)
            continue
        keep.append(True)
        starts.append(bucket_start)
        # Checked, not assumed. Collapsing one end per bucket is only sound
        # while the mapping really is a function; a market whose windows put
        # two sessions on the same bucket start would otherwise silently take
        # whichever end arrived first and mis-size that bucket's expected bar
        # count. One dict op either way, and a wrong number here is invisible
        # downstream -- it lands as a quality count, not as a crash.
        if ends_by_start.setdefault(bucket_start, bucket_end) != bucket_end:
            raise BarAggregationError(
                "two sessions produced decision bucket "
                f"{bucket_start} with different ends "
                f"({ends_by_start[bucket_start]} and {bucket_end}); "
                "bucket_end is no longer a function of bucket_start"
            )

    # Boolean-mask `.loc` already returns a new frame, and grouping on an
    # external key array never writes to it, so the defensive `.copy()` -- a
    # third full copy of the bars, after `_as_local_index`'s `frame.copy()` and
    # its `sort_index()` -- bought nothing here. Positional alignment holds
    # because `starts` gains exactly one entry for every True appended to
    # `keep`.
    kept = local.loc[keep]
    bucket_keys = pd.DatetimeIndex(starts)

    records: list[dict] = []
    for bucket_start, group in kept.groupby(bucket_keys, sort=True):
        bucket_end = ends_by_start[bucket_start]
        group = group.sort_index()
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
        source_volume = pd.to_numeric(group["volume"], errors="coerce").fillna(0.0)
        turnover_prices = pd.to_numeric(group["close"], errors="coerce")
        if "vwap" in group.columns:
            source_vwap = pd.to_numeric(group["vwap"], errors="coerce")
            turnover_prices = source_vwap.where(
                np.isfinite(source_vwap), turnover_prices
            )
        record = {
            "timestamp": bucket_end.tz_convert("UTC"),
            "open": float(group["open"].iloc[0]),
            "high": float(pd.to_numeric(group["high"], errors="coerce").max()),
            "low": float(pd.to_numeric(group["low"], errors="coerce").min()),
            "close": close,
            "volume": volume,
            # Alpaca's per-bar VWAP times share volume is the traded notional.
            # Falling back to each source bar's close keeps older fixtures and
            # providers useful without introducing an hour-end-price estimate.
            "turnover": float((turnover_prices * source_volume).sum()),
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
    # A decision bar is stamped at its close, whatever its source was.
    result.attrs.pop(FRAME_ATTR_OPEN_STAMPED_MINUTES, None)
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


class ExecutionFill(NamedTuple):
    """How one decision fills: the source ``bar`` it is priced from, which of
    that bar's prices (``price_field``), and the instant it fills
    (``filled_at``). One record rather than parallel lists, so a bar can never
    be paired with another step's price field."""

    bar: Any
    price_field: str
    filled_at: Any


def plan_execution_fills(
    decision_timestamps: Iterable[Any],
    source_timestamps: List[Any],
    *,
    source_minutes: int,
    market: str,
    timezone: str,
) -> Dict[Any, ExecutionFill]:
    """Map each decision bar to the :class:`ExecutionFill` it fills on.

    ``source_timestamps`` are open-stamped bars of ``source_minutes`` (the only
    kind :func:`aggregate_bars` accepts), sorted. A decision closes at its
    stamp, so it fills at the ``open`` of the source bar opening at that
    instant -- the first fill without look-ahead. A session's final bucket
    (16:00 ET) has no such bar once after-hours bars are out of the source set,
    so it fills at the ``close`` of the source bar ending at that instant,
    priced at the last regular-hours trade and stamped at that bar's close
    rather than its open, so the trade never predates the decision.
    Filling it at the 16:00 bar instead made the day's closing decision depend
    on whether the tape served an after-hours bar at all.

    One planner for the engine and the protocol path's dataset store, which
    each used to carry their own copy of the exact-match rule. Decisions with
    no fill are absent.
    """
    by_day: Dict[date, List[Any]] = {}
    for timestamp in source_timestamps:
        by_day.setdefault(market_local(timestamp, timezone).date(), []).append(
            timestamp
        )
    span = pd.Timedelta(minutes=source_minutes)
    fills: Dict[Any, ExecutionFill] = {}
    for timestamp in decision_timestamps:
        same_day = by_day.get(market_local(timestamp, timezone).date(), [])
        index = bisect_left(same_day, timestamp)
        if index < len(same_day) and same_day[index] == timestamp:
            # The source's own object, not the equal decision stamp: they can
            # differ in tz, and the fill bar is what a trade is stamped with.
            bar = same_day[index]
            fills[timestamp] = ExecutionFill(bar, "open", bar)
        elif (
            index > 0
            and same_day[index - 1] + span == timestamp
            and is_session_close(timestamp, market=market, timezone=timezone)
        ):
            # Only the bar closing AT the session close: an earlier one's close
            # predates the decision it would fill.
            bar = same_day[index - 1]
            fills[timestamp] = ExecutionFill(bar, "close", bar + span)
    return fills
