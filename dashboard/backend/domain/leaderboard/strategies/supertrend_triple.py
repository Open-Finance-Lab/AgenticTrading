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

import pandas as pd

from dashboard.backend.infrastructure.llm.validator import DJIA_30

from .base import BaselineStrategy
from ._indicators import supertrend_single
from ._signal_engine import DailyHistory, make_entry_exit_weight_fn, run_daily_signal_strategy

_MAX_POSITIONS = 8
_MIN_HISTORY = 20
_VARIANTS = [(7, 3.0), (10, 3.0), (14, 4.0)]


def _all_trends(history: DailyHistory, sym: str) -> List[str]:
    close = history.close[sym].dropna()
    high = history.high[sym].reindex(close.index)
    low = history.low[sym].reindex(close.index)
    return [supertrend_single(high, low, close, period, mult).iloc[-1] for period, mult in _VARIANTS]


class SupertrendTripleStrategy(BaselineStrategy):
    key = "supertrend_triple"

    def required_symbols(self) -> List[str]:
        symbols = self.config.get("symbols")
        return list(symbols) if symbols else list(DJIA_30)

    def run(
        self,
        bars_by_symbol: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        initial_capital: float,
    ) -> List[Dict[str, Any]]:
        symbols = self.required_symbols()
        bars_subset = {s: bars_by_symbol[s] for s in symbols if s in bars_by_symbol}
        if not bars_subset:
            return []

        def entry(history: DailyHistory, sym: str) -> bool:
            if len(history) < _MIN_HISTORY:
                return False
            return all(t == "up" for t in _all_trends(history, sym))

        def exit_(history: DailyHistory, sym: str) -> bool:
            if len(history) < _MIN_HISTORY:
                return False
            return all(t == "down" for t in _all_trends(history, sym))

        weight_fn = make_entry_exit_weight_fn(entry, exit_, symbols, _MAX_POSITIONS, _MIN_HISTORY)
        curve, n_trades = run_daily_signal_strategy(
            bars_subset, start_date, end_date, initial_capital, weight_fn,
            rebalance_every_days=1,
        )
        self._num_trades = n_trades
        return curve

    def num_trades(self) -> int:
        return getattr(self, "_num_trades", 0)
