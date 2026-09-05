"""Shared, side-effect-free helpers for baseline strategies.

These utilities only build price series and equity curves from already-fetched
bars. They contain no strategy logic, so individual strategies stay independent.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

import pandas as pd
import pytz

_ET = pytz.timezone("US/Eastern")


def parse_config_date(date_str: str) -> dt.date:
    return dt.datetime.strptime(date_str, "%Y-%m-%d").date()


def reference_start_date(contest_start: str, config: Optional[Dict[str, Any]] = None) -> str:
    """Prior-month reference window start (defaults to one calendar month before contest)."""
    if config and config.get("reference_start_date"):
        return str(config["reference_start_date"])
    start = parse_config_date(contest_start)
    month = start.month - 1
    year = start.year
    if month < 1:
        month = 12
        year -= 1
    return f"{year}-{month:02d}-{start.day:02d}"


def timestamp_date(ts: Any) -> dt.date:
    if hasattr(ts, "astimezone"):
        return ts.astimezone(_ET).date()
    return parse_config_date(str(ts)[:10])


def timestamps_in_contest(
    timestamps: List[Any],
    start_date: str,
    end_date: str,
) -> List[Any]:
    """Keep timestamps whose calendar date falls within [start_date, end_date]."""
    start = parse_config_date(start_date)
    end = parse_config_date(end_date)
    return [ts for ts in timestamps if start <= timestamp_date(ts) <= end]


def timestamps_in_reference(
    timestamps: List[Any],
    reference_start: str,
    contest_start: str,
) -> List[Any]:
    """Keep timestamps in [reference_start, contest_start) — prior month only."""
    ref_start = parse_config_date(reference_start)
    contest = parse_config_date(contest_start)
    return [ts for ts in timestamps if ref_start <= timestamp_date(ts) < contest]


def filter_market_hours(timestamps: List[Any]) -> List[Any]:
    """Keep only regular US market-hours timestamps (9:30–16:00 ET)."""
    kept = []
    for ts in timestamps:
        ts_et = ts.astimezone(_ET)
        hour, minute = ts_et.hour, ts_et.minute
        is_market_hours = (
            (hour > 9 and hour < 16)
            or (hour == 9 and minute >= 30)
            or (hour == 16 and minute == 0)
        )
        if is_market_hours:
            kept.append(ts)
    return kept


def market_timestamps(bars_subset: Dict[str, pd.DataFrame]) -> List[Any]:
    """Sorted, market-hours-only union of timestamps across the given symbols."""
    all_ts = set()
    for df in bars_subset.values():
        all_ts.update(df.index)
    return filter_market_hours(sorted(all_ts))


def build_price_cache(
    bars_subset: Dict[str, pd.DataFrame],
    timestamps: List[Any],
) -> Dict[str, Dict[Any, float]]:
    """Forward-filled close price per symbol over the supplied timestamps.

    A symbol is priced from its own first bar onward: timestamps before it has
    any bar are simply absent from its entry (callers treat a missing price as
    "not tradable yet"). A symbol with no bars at any of the timestamps gets no
    entry at all.
    """
    if not timestamps:
        return {}

    cache: Dict[str, Dict[Any, float]] = {}
    for symbol, df in bars_subset.items():
        # dict lookup per timestamp; a duplicated index timestamp resolves to its
        # last bar instead of surfacing as a Series.
        closes = df["close"].to_dict()
        last_price: Optional[float] = None
        per_ts: Dict[Any, float] = {}
        for ts in timestamps:
            if ts in closes:
                last_price = closes[ts]
            if last_price is not None:
                per_ts[ts] = last_price
        if per_ts:
            cache[symbol] = per_ts
    return cache


def equity_curve_from_positions(
    positions: Dict[str, int],
    cash: float,
    price_cache: Dict[str, Dict[Any, float]],
    timestamps: List[Any],
) -> List[Dict[str, Any]]:
    """Build an equity curve given fixed share counts and a price cache."""
    curve: List[Dict[str, Any]] = []
    for ts in timestamps:
        positions_value = 0.0
        for symbol, shares in positions.items():
            prices = price_cache.get(symbol)
            if prices and ts in prices:
                positions_value += shares * prices[ts]
        total_equity = cash + positions_value
        curve.append(
            {
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "equity": round(total_equity, 2),
                "cash": round(cash, 2),
                "positions_value": round(positions_value, 2),
                "daily_return": 0,
            }
        )
    return curve


def subset_bars(
    bars_by_symbol: Dict[str, pd.DataFrame],
    symbols: List[str],
) -> Dict[str, pd.DataFrame]:
    """Return only the bars for symbols that are both requested and available."""
    return {s: bars_by_symbol[s] for s in symbols if s in bars_by_symbol}
