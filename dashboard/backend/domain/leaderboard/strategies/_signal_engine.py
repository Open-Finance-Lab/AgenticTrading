"""Shared engine for baseline strategies whose signals are computed on daily
closes but must return an hourly equity curve, since ``base.py``'s contract
promises one and the leaderboard/contest engine feeds ``run()`` hourly bars
throughout (see ``baselines.py::fetch_hourly_bars``, ``TimeFrame.Hour``).

Resamples ``bars_by_symbol`` to one daily close per symbol, evaluates a
strategy's own weight function once per trading day using only closes
strictly before that day (no look-ahead), converts the resulting weights to
share counts, and marks equity every hour in between using intraday prices.

Lookback caveat: the leaderboard's contest window is intentionally short
(one month) with a matching one-month reference buffer before it (see
``leaderboard.json``'s top-level ``reference_start_date``) -- nowhere near
the full year these strategies were originally validated against in the
"Strategy Lab" backtest report. Every ``weight_fn`` here is expected to cap
its own lookback at whatever history is actually available (see
``available_window`` below) rather than require a fixed number of days and
silently produce ``NaN``/empty weights when the contest window is shorter
than that. On the real 1-month contest this means a strategy nominally
described as "252-day momentum" effectively becomes a much shorter-window
version of itself -- an honest consequence of the contest's short window,
not a bug, and documented per-strategy.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from ._common import build_price_cache, market_timestamps, timestamp_date, timestamps_in_contest


@dataclass
class DailyHistory:
    """Daily-resampled OHLCV, one column per symbol, sliced to strictly
    BEFORE the current day (no look-ahead) before being handed to a
    strategy's weight_fn."""

    close: pd.DataFrame
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame

    def __len__(self) -> int:
        return len(self.close)

    def before(self, cur_date: dt.date) -> "DailyHistory":
        cutoff = pd.Timestamp(cur_date)
        return DailyHistory(
            close=self.close[self.close.index < cutoff],
            open=self.open[self.open.index < cutoff],
            high=self.high[self.high.index < cutoff],
            low=self.low[self.low.index < cutoff],
            volume=self.volume[self.volume.index < cutoff],
        )


WeightFn = Callable[[DailyHistory, dt.date, int], Dict[str, float]]
# weight_fn(history, current_date, day_index) -> {symbol: weight (0..1)}
# `history` holds only daily OHLCV strictly BEFORE `current_date` (no
# look-ahead). `day_index` counts trading days from the first bar seen in
# `bars_by_symbol` (0-based), so a strategy can use e.g. `day_index % 21 == 0`
# for a monthly cadence, or ignore it for a daily one.


def available_window(history: DailyHistory, desired: int) -> int:
    """Cap a strategy's own desired lookback at what's actually available."""
    return max(0, min(desired, len(history)))


def _resample_field(bars_by_symbol: Dict[str, pd.DataFrame], field: str, how: str) -> pd.DataFrame:
    cols: Dict[str, pd.Series] = {}
    for sym, df in bars_by_symbol.items():
        if df is None or df.empty or field not in df.columns:
            continue
        grouped = df[field].groupby(df.index.map(timestamp_date))
        cols[sym] = getattr(grouped, how)()
    if not cols:
        return pd.DataFrame()
    frame = pd.DataFrame(cols).sort_index()
    frame.index = pd.to_datetime(frame.index)
    return frame


def daily_history(bars_by_symbol: Dict[str, pd.DataFrame]) -> DailyHistory:
    """Resample hourly bars_by_symbol to one daily OHLCV row per symbol."""
    return DailyHistory(
        close=_resample_field(bars_by_symbol, "close", "last"),
        open=_resample_field(bars_by_symbol, "open", "first"),
        high=_resample_field(bars_by_symbol, "high", "max"),
        low=_resample_field(bars_by_symbol, "low", "min"),
        volume=_resample_field(bars_by_symbol, "volume", "sum"),
    )


def make_entry_exit_weight_fn(
    entry_fn: Callable[[DailyHistory, str], bool],
    exit_fn: Callable[[DailyHistory, str], bool],
    symbols: List[str],
    max_positions: int = 8,
    min_history: int = 2,
) -> WeightFn:
    """Shared 'held set' bookkeeping for strategies that decide per-symbol
    entry/exit each day from the latest available history, rather than
    re-ranking a fresh target set every rebalance (e.g. Bandtastic,
    Supertrend, hlhb). `entry_fn(history, symbol)` / `exit_fn(history, symbol)`
    look only at `history` (already sliced to strictly before today) and
    should read whatever trailing window they need off its tail themselves."""
    held: set = set()

    def weight_fn(history: DailyHistory, cur_date, day_index):
        if len(history) < min_history:
            return {s: 1.0 / len(held) for s in held} if held else {}
        for s in list(held):
            if exit_fn(history, s):
                held.discard(s)
        if len(held) < max_positions:
            for s in symbols:
                if len(held) >= max_positions:
                    break
                if s not in held and entry_fn(history, s):
                    held.add(s)
        if not held:
            return {}
        return {s: 1.0 / len(held) for s in held}

    return weight_fn


def run_daily_signal_strategy(
    bars_by_symbol: Dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
    initial_capital: float,
    weight_fn: WeightFn,
    rebalance_every_days: int = 1,
    lot_size: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Returns (equity_curve, n_trades). `n_trades` counts genuine buy/sell
    share-count changes, not every day's mark-to-market."""
    symbols = list(bars_by_symbol.keys())
    history_full = daily_history(bars_by_symbol)
    if history_full.close.empty:
        return [], 0

    all_ts = market_timestamps(bars_by_symbol)
    contest_ts = timestamps_in_contest(all_ts, start_date, end_date)
    if not contest_ts:
        return [], 0
    price_cache = build_price_cache(bars_by_symbol, all_ts)

    cash = float(initial_capital)
    shares: Dict[str, float] = {s: 0.0 for s in symbols}
    curve: List[Dict[str, Any]] = []
    weights: Dict[str, float] = {}
    last_date: Optional[dt.date] = None
    day_index = -1
    n_trades = 0

    for ts in contest_ts:
        cur_date = timestamp_date(ts)
        if cur_date != last_date:
            last_date = cur_date
            day_index += 1
            if day_index % max(rebalance_every_days, 1) == 0:
                history = history_full.before(cur_date)
                weights = weight_fn(history, cur_date, day_index) or {}

                prices = {s: price_cache.get(s, {}).get(ts) for s in symbols}
                port_val = cash + sum(
                    shares[s] * prices[s] for s in symbols if prices.get(s) is not None
                )
                target: Dict[str, float] = {}
                for s in symbols:
                    w = weights.get(s, 0.0)
                    p = prices.get(s)
                    if not p or p <= 0 or w <= 0:
                        target[s] = 0.0
                        continue
                    dollars = w * port_val
                    if lot_size:
                        dollars = round(dollars / lot_size) * lot_size
                    target[s] = dollars
                total_target = sum(target.values())
                if lot_size and total_target > port_val > 0:
                    scale = port_val / total_target
                    target = {s: round(v * scale / lot_size) * lot_size for s, v in target.items()}

                new_cash = port_val
                for s in symbols:
                    p = prices.get(s)
                    if not p or p <= 0:
                        continue
                    new_sh = target[s] / p
                    if abs(new_sh - shares[s]) > 1e-9:
                        n_trades += 1
                    shares[s] = new_sh
                    new_cash -= new_sh * p
                cash = new_cash

        prices_now = {s: price_cache.get(s, {}).get(ts) for s in symbols}
        positions_value = sum(
            shares[s] * prices_now[s] for s in symbols if prices_now.get(s) is not None
        )
        curve.append(
            {
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "equity": round(cash + positions_value, 2),
                "cash": round(cash, 2),
                "positions_value": round(positions_value, 2),
                "daily_return": 0,
            }
        )
    return curve, n_trades
