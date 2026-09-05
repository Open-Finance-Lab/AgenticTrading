"""hlhb, from the freqtrade-strategies community repository
(github.com/freqtrade/freqtrade-strategies, user_data/strategies/hlhb.py).

Buys when RSI crosses above 50, the 5-day EMA rises above the 10-day EMA,
and ADX confirms a real trend (>25) -- three signals confirming a fresh
uptrend simultaneously; sells on the mirrored bearish combination. Fully
generic trend/momentum confirmation stack, portable without modification.

Lookback: needs ~28 days for ADX to produce its first value (two chained
14-day Wilder smoothings); below that, holds whatever is already held and
takes no new positions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .base import BaselineStrategy
from ._common import subset_bars
from ._indicators import adx, ema, rsi
from ._signal_engine import DailyHistory, daily_history, make_entry_exit_weight_fn, run_daily_signal_strategy

_MAX_POSITIONS = 8
_MIN_HISTORY = 27  # entry/exit need one row more than this for the crossover


class HlhbStrategy(BaselineStrategy):
    key = "hlhb"

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

        # Indicators are computed once per symbol over the full daily series and
        # read back at the position of the last row of the (strictly-before-today)
        # history the engine hands entry/exit. rsi/ema/adx are causal (ewm), so
        # position i of the full series equals the last value of the same
        # indicator computed on the first i+1 rows alone -- no look-ahead.
        history_full = daily_history(bars_subset)
        cache: Dict[str, Tuple[np.ndarray, ...]] = {}

        def _series(sym: str) -> Tuple[np.ndarray, ...]:
            if sym not in cache:
                close = history_full.close[sym].dropna().to_frame()
                high = history_full.high[sym].reindex(close.index).to_frame()
                low = history_full.low[sym].reindex(close.index).to_frame()
                adx14, _, _ = adx(high, low, close, 14)
                cache[sym] = tuple(
                    frame.iloc[:, 0].to_numpy()
                    for frame in (rsi(close, 14), ema(close, 5), ema(close, 10), adx14)
                )
            return cache[sym]

        def _last_row(history: DailyHistory, sym: str) -> int:
            """Position, in the full series, of the last daily row in `history`."""
            return int(history.close[sym].count()) - 1

        def entry(history: DailyHistory, sym: str) -> bool:
            i = _last_row(history, sym)
            if i < _MIN_HISTORY:
                return False
            rsi14, ema5, ema10, adx14 = _series(sym)
            return bool(
                rsi14[i] > 50
                and rsi14[i - 1] <= 50
                and ema5[i] > ema10[i]
                and adx14[i] > 25
            )

        def exit_(history: DailyHistory, sym: str) -> bool:
            i = _last_row(history, sym)
            if i < _MIN_HISTORY:
                return False
            rsi14, ema5, ema10, adx14 = _series(sym)
            return bool(
                rsi14[i] < 50
                and rsi14[i - 1] >= 50
                and ema5[i] < ema10[i]
                and adx14[i] > 25
            )

        weight_fn = make_entry_exit_weight_fn(entry, exit_, symbols, _MAX_POSITIONS, _MIN_HISTORY)
        curve, n_trades = run_daily_signal_strategy(
            bars_subset, start_date, end_date, initial_capital, weight_fn,
            rebalance_every_days=1, history=history_full,
        )
        self._num_trades = n_trades
        return curve
