"""Leaderboard baseline strategy plumbing.

Strategy *logic* lives in the leaderboard ``strategies`` package — one file per
strategy. This module just handles data fetching, dispatch, and metrics.

Canonical location (Phase 3C3). Moved from
``dashboard/backend/engines/leaderboard_baselines.py``; the original module was
removed in Phase 4A. The reusable backtest metric helpers and the
Alpaca market-data provider are imported from their canonical homes (not
duplicated here); only the leaderboard-specific contest-window fetch, metric
aggregation schema, and daily downsample policy live here.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd

from dashboard.backend.domain.leaderboard.strategies import get_strategy
from dashboard.backend.domain.backtesting.constants import INITIAL_CAPITAL
from dashboard.backend.domain.backtesting.metrics import (
    calculate_max_drawdown,
    calculate_sharpe,
)
from dashboard.backend.infrastructure.market_data.alpaca_bars import AlpacaDataLoader


def fetch_hourly_bars(symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """Load Alpaca hourly bars for the contest window.

    Alpaca treats ``end`` as exclusive, which would drop the final window day
    (e.g. bars on ``end_date`` start after midnight). We bump it by one day so
    the Alpaca strategies cover the same last day as the Yahoo index series.

    For the rolling *daily* board, ``end_date`` is often "today", so this bump
    lands on tomorrow 00:00 UTC — inside Basic's forbidden recent-SIP window.
    ``AlpacaDataLoader.fetch_bars`` clamps SIP ``end`` to now−15m for that case.

    The returned frames carry the tape they came from in ``.attrs``; callers
    that persist a curve should read it with ``feed_provenance`` and store it,
    because the log line below outlives nothing.
    """
    loader = AlpacaDataLoader()
    end_inclusive = (
        datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")
    data = loader.fetch_bars(symbols, start_date, end_inclusive)
    meta = loader.last_fetch or {}
    if meta.get("sip_fallback_to_iex"):
        print(
            "WARNING: leaderboard bars served from IEX after SIP refusal "
            f"(requested_end={meta.get('requested_end')}, "
            f"effective_end={meta.get('effective_end')}). "
            "IEX is not the SIP tape."
        )
    elif meta.get("end_clamped"):
        print(
            "NOTE: leaderboard bars fetched with a clamped SIP end "
            f"(requested_end={meta.get('requested_end')}, "
            f"effective_end={meta.get('effective_end')})."
        )
    return data


def compute_equity_curve(
    strategy: Dict[str, Any],
    bars_by_symbol: Dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
    initial_capital: float = INITIAL_CAPITAL,
) -> List[Dict[str, Any]]:
    """Run one leaderboard baseline strategy via the strategy registry."""
    return get_strategy(strategy).run(
        bars_by_symbol, start_date, end_date, initial_capital
    )


def calc_metrics(equity_curve: List[Dict[str, Any]], initial_capital: float) -> Dict[str, float]:
    """Sharpe, drawdown, return from an hourly equity curve."""
    if not equity_curve:
        return {
            "initial_equity": initial_capital,
            "final_equity": initial_capital,
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }

    initial_eq = equity_curve[0].get("equity", initial_capital)
    final_eq = equity_curve[-1].get("equity", initial_capital)
    total_return = (final_eq - initial_capital) / initial_capital if initial_capital else 0.0

    return {
        "initial_equity": float(initial_eq),
        "final_equity": float(final_eq),
        "total_return": float(total_return),
        "sharpe_ratio": float(calculate_sharpe(equity_curve)),
        "max_drawdown": float(calculate_max_drawdown(equity_curve)),
    }


def downsample_daily(equity_curve: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep last equity point per calendar day for chart display."""
    by_day: Dict[str, Dict[str, Any]] = {}
    for point in equity_curve:
        ts = str(point.get("timestamp", ""))
        day = ts[:10] if len(ts) >= 10 else ts
        if day:
            by_day[day] = point
    return [by_day[day] for day in sorted(by_day.keys())]


def chart_equity_curve(
    equity_hourly: List[Dict[str, Any]],
    *,
    initial_equity: float,
    start_date: str,
) -> List[Dict[str, Any]]:
    """Build the Competition-chart series: open tick + full hourly path.

    Prepends a synthetic point at ``{start_date}T00:00:00+00:00`` with
    ``initial_equity`` so every series shares the same leftmost x and opens at
    starting capital. The stored hourly points follow unchanged (no daily
    downsample).
    """
    open_ts = f"{start_date}T00:00:00+00:00"
    open_point: Dict[str, Any] = {
        "timestamp": open_ts,
        "equity": float(initial_equity),
        "cash": float(initial_equity),
        "positions_value": 0.0,
    }
    if not equity_hourly:
        return [open_point]

    # Drop a stored point that already sits on the open timestamp so we don't
    # double-plot the same x with a different equity.
    rest = [
        dict(pt)
        for pt in equity_hourly
        if str(pt.get("timestamp", "")) != open_ts
    ]
    return [open_point, *rest]


def _timestamp_key(ts: Any) -> str:
    """Comparable timestamp key (drop timezone suffix so Alpaca/Yahoo mix sorts)."""
    s = str(ts or "")
    if len(s) >= 19 and s[10] == "T":
        return s[:19]
    return s


def _timestamp_minute(ts: Any) -> int:
    key = _timestamp_key(ts)
    if len(key) >= 16 and key[13] == ":":
        try:
            return int(key[14:16])
        except ValueError:
            return -1
    return -1


def pick_reference_timestamps(curves: List[List[Dict[str, Any]]]) -> List[str]:
    """Choose the shared chart axis: prefer Alpaca-style :00 hours (longest).

    Yahoo index hours are typically :30 (US cash open 9:30 ET). Agent / stock
    baselines from Alpaca are usually :00. The Competition chart aligns everyone
    onto the Alpaca grid when available.
    """
    best_ts: List[str] = []
    best_score = (-1, -1)  # (count of :00 minutes, length)
    for curve in curves:
        if not curve:
            continue
        timestamps = [str(p.get("timestamp", "")) for p in curve if p.get("timestamp")]
        if not timestamps:
            continue
        on_the_hour = sum(1 for t in timestamps if _timestamp_minute(t) == 0)
        score = (on_the_hour, len(timestamps))
        if score > best_score:
            best_score = score
            best_ts = timestamps
    return best_ts


def align_equity_curve_asof(
    curve: List[Dict[str, Any]],
    reference_timestamps: List[str],
) -> List[Dict[str, Any]]:
    """Reindex ``curve`` onto ``reference_timestamps`` via as-of (forward) fill."""
    if not reference_timestamps:
        return [dict(pt) for pt in curve]
    if not curve:
        return []

    sources = sorted(curve, key=lambda p: _timestamp_key(p.get("timestamp")))
    out: List[Dict[str, Any]] = []
    i = 0
    last: Dict[str, Any] | None = None
    for ref in reference_timestamps:
        ref_key = _timestamp_key(ref)
        while i < len(sources) and _timestamp_key(sources[i].get("timestamp")) <= ref_key:
            last = sources[i]
            i += 1
        if last is None:
            continue
        out.append(
            {
                "timestamp": ref,
                "equity": last.get("equity"),
                "cash": last.get("cash"),
                "positions_value": last.get("positions_value"),
                "daily_return": last.get("daily_return", 0),
            }
        )
    return out


def align_equity_curves(
    curves: List[List[Dict[str, Any]]],
) -> List[List[Dict[str, Any]]]:
    """Align every curve onto one shared timestamp axis (as-of fill)."""
    ref = pick_reference_timestamps(curves)
    if not ref:
        return [[dict(pt) for pt in c] for c in curves]
    return [align_equity_curve_asof(c, ref) for c in curves]
