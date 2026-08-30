"""Supertrend (triple-confirmation), from the freqtrade-strategies community
repository (github.com/freqtrade/freqtrade-strategies, user_data/strategies
/Supertrend.py).

Combines three ATR-based Supertrend indicators at different sensitivities
(7/3, 10/3, 14/4 period/multiplier); only buys a stock when all three agree
it's trending up, and exits when all three flip to down -- triple-
confirmation trend-following. ATR-based Supertrend is asset-class agnostic
and needed no adaptation from freqtrade's crypto pairs.

Lookback: needs ~20 days for the slowest ATR to stabilize; below that, holds
whatever is already held and takes no new positions.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .base import BaselineStrategy
from ._common import subset_bars
from ._indicators import supertrend_single
from ._signal_engine import DailyHistory, daily_history, make_entry_exit_weight_fn, run_daily_signal_strategy

_MAX_POSITIONS = 8
_MIN_HISTORY = 20
_VARIANTS = [(7, 3.0), (10, 3.0), (14, 4.0)]


class SupertrendTripleStrategy(BaselineStrategy):
    key = "supertrend_triple"

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

        # The three trend series are computed once per symbol over the full daily
        # series and read back at the position of the last row of the
        # (strictly-before-today) history the engine hands entry/exit. Supertrend
        # is a forward recursion over ATR (itself an ewm), so position i of the
        # full series equals the last value computed on the first i+1 rows alone
        # -- no look-ahead.
        history_full = daily_history(bars_subset)
        cache: Dict[str, List[np.ndarray]] = {}

        def _all_trends(sym: str) -> List[np.ndarray]:
            if sym not in cache:
                close = history_full.close[sym].dropna()
                high = history_full.high[sym].reindex(close.index)
                low = history_full.low[sym].reindex(close.index)
                cache[sym] = [
                    supertrend_single(high, low, close, period, mult).to_numpy()
                    for period, mult in _VARIANTS
                ]
            return cache[sym]

        def _last_row(history: DailyHistory, sym: str) -> int:
            """Position, in the full series, of the last daily row in `history`."""
            return int(history.close[sym].count()) - 1

        def entry(history: DailyHistory, sym: str) -> bool:
            if len(history) < _MIN_HISTORY:
                return False
            i = _last_row(history, sym)
            return i >= 0 and all(trend[i] == "up" for trend in _all_trends(sym))

        def exit_(history: DailyHistory, sym: str) -> bool:
            if len(history) < _MIN_HISTORY:
                return False
            i = _last_row(history, sym)
            return i >= 0 and all(trend[i] == "down" for trend in _all_trends(sym))

        weight_fn = make_entry_exit_weight_fn(entry, exit_, symbols, _MAX_POSITIONS, _MIN_HISTORY)
        curve, n_trades = run_daily_signal_strategy(
            bars_subset, start_date, end_date, initial_capital, weight_fn,
            rebalance_every_days=1, history=history_full,
        )
        self._num_trades = n_trades
        return curve
