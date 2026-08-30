"""TrendRiderStrategy (simplified), from the freqtrade-strategies community
repository (github.com/freqtrade/freqtrade-strategies, user_data/strategies
/TrendRiderStrategy.py).

Enters on any of three trend-confirmation signals (a golden cross of the
10/50-day EMA, an RSI bounce off oversold while price holds above the
200-day average, or a MACD histogram turning positive); exits on RSI
overheating, a bearish EMA cross, or price falling back below the 200-day
average.

Simplification: the original strategy has 6 entry patterns and 4 exits, plus
a cascading time-based profit exit -- reduced here to the 3 clearest
generic entries and 3 clearest exits. The original also gates every entry on
BTC RSI>35 and a crypto Fear & Greed Index band; both are crypto-market-
structure-specific regime filters with no equities equivalent, and are
dropped here rather than replaced with an invented substitute.

Lookback: needs ~200 days for the SMA-200 term (capped to whatever is
available -- see the module docstring in ``_signal_engine.py`` for why the
leaderboard's short contest window makes this term behave differently than
in the full-year "Strategy Lab" backtest).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .base import BaselineStrategy
from ._common import subset_bars
from ._indicators import ema, macd, rsi
from ._signal_engine import (
    DailyHistory,
    daily_history,
    make_entry_exit_weight_fn,
    run_daily_signal_strategy,
)

_MAX_POSITIONS = 8
_MIN_HISTORY = 25
_SMA_DAYS = 200


def _sma_last(close: pd.Series) -> float:
    """SMA over the available-capped 200-day window, at the last row of `close`.

    The window is capped at ``len(close)`` -- this symbol's own non-NaN daily
    rows, not ``len(history)`` (the union frame across symbols): one gap day
    for this symbol would make a union-sized window exceed its rows and turn
    the SMA permanently NaN, disabling both the RSI-bounce entry and the
    SMA-breakdown exit for it. The window grows day by day on a short contest,
    so unlike the fixed-span indicators it is one short mean per call."""
    return float(close.iloc[-_SMA_DAYS:].mean())


class TrendRiderStrategy(BaselineStrategy):
    key = "trendrider"

    def run(
        self,
        bars_by_symbol: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        initial_capital: float,
    ) -> List[Dict[str, Any]]:
        symbols = self.required_symbols()
        bars_subset = subset_bars(bars_by_symbol, symbols)
        if not bars_subset:
            return []

        # Fixed-span indicators are computed once per symbol over the full daily
        # series and read back at the position of the last row of the
        # (strictly-before-today) history the engine hands entry/exit. ema/macd/
        # rsi are causal (ewm), so position i of the full series equals the last
        # value of the same indicator computed on the first i+1 rows alone -- no
        # look-ahead. The MACD histogram is read only by entry(); exit_() does
        # not pay for it.
        history_full = daily_history(bars_subset)
        cache: Dict[str, Tuple[np.ndarray, ...]] = {}

        def _series(sym: str) -> Tuple[np.ndarray, ...]:
            if sym not in cache:
                close = history_full.close[sym].dropna().to_frame()
                _, _, hist_line = macd(close)
                cache[sym] = tuple(
                    frame.iloc[:, 0].to_numpy()
                    for frame in (close, ema(close, 10), ema(close, 50), hist_line, rsi(close, 14))
                )
            return cache[sym]

        def entry(history: DailyHistory, sym: str) -> bool:
            close = history.close[sym].dropna()
            i = len(close) - 1
            if i < _MIN_HISTORY:
                return False
            c, ema10, ema50, hist_line, rsi14 = _series(sym)
            golden_cross = ema10[i] > ema50[i] and ema10[i - 1] <= ema50[i - 1]
            rsi_bounce = rsi14[i - 1] < 30 and rsi14[i] >= 30 and c[i] > _sma_last(close)
            macd_cross = hist_line[i] > 0 and hist_line[i - 1] <= 0
            return bool(golden_cross or rsi_bounce or macd_cross)

        def exit_(history: DailyHistory, sym: str) -> bool:
            close = history.close[sym].dropna()
            i = len(close) - 1
            if i < _MIN_HISTORY:
                return False
            c, ema10, ema50, _, rsi14 = _series(sym)
            bearish_cross = ema10[i] < ema50[i]
            return bool(
                rsi14[i] > 78
                or bearish_cross
                or c[i] < _sma_last(close) * 0.99
            )

        weight_fn = make_entry_exit_weight_fn(entry, exit_, symbols, _MAX_POSITIONS, _MIN_HISTORY)
        curve, n_trades = run_daily_signal_strategy(
            bars_subset, start_date, end_date, initial_capital, weight_fn,
            rebalance_every_days=1, history=history_full,
        )
        self._num_trades = n_trades
        return curve
