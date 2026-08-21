"""hlhb, from the freqtrade-strategies community repository
(github.com/freqtrade/freqtrade-strategies, user_data/strategies/hlhb.py).

Buys when RSI crosses above 50, the 5-day EMA rises above the 10-day EMA,
and ADX confirms a real trend (>25) -- three signals confirming a fresh
uptrend simultaneously; sells on the mirrored bearish combination. Fully
generic trend/momentum confirmation stack, portable without modification.

Lookback: needs ~25 days for ADX/EMA to stabilize; below that, holds
whatever is already held and takes no new positions.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from dashboard.backend.infrastructure.llm.validator import DJIA_30

from .base import BaselineStrategy
from ._indicators import adx, ema, rsi
from ._signal_engine import DailyHistory, make_entry_exit_weight_fn, run_daily_signal_strategy

_MAX_POSITIONS = 8
_MIN_HISTORY = 25


class HlhbStrategy(BaselineStrategy):
    key = "hlhb"

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

        def _series(history: DailyHistory, sym: str):
            close = history.close[sym].dropna().to_frame()
            high = history.high[sym].reindex(close.index).to_frame()
            low = history.low[sym].reindex(close.index).to_frame()
            rsi14 = rsi(close, 14)
            ema5 = ema(close, 5)
            ema10 = ema(close, 10)
            adx14, _, _ = adx(high, low, close, 14)
            return rsi14, ema5, ema10, adx14

        def entry(history: DailyHistory, sym: str) -> bool:
            close = history.close[sym].dropna()
            if len(close) < _MIN_HISTORY + 1:
                return False
            rsi14, ema5, ema10, adx14 = _series(history, sym)
            return bool(
                rsi14.iloc[-1, 0] > 50
                and rsi14.iloc[-2, 0] <= 50
                and ema5.iloc[-1, 0] > ema10.iloc[-1, 0]
                and adx14.iloc[-1, 0] > 25
            )

        def exit_(history: DailyHistory, sym: str) -> bool:
            close = history.close[sym].dropna()
            if len(close) < _MIN_HISTORY + 1:
                return False
            rsi14, ema5, ema10, adx14 = _series(history, sym)
            return bool(
                rsi14.iloc[-1, 0] < 50
                and rsi14.iloc[-2, 0] >= 50
                and ema5.iloc[-1, 0] < ema10.iloc[-1, 0]
                and adx14.iloc[-1, 0] > 25
            )

        weight_fn = make_entry_exit_weight_fn(entry, exit_, symbols, _MAX_POSITIONS, _MIN_HISTORY)
        curve, n_trades = run_daily_signal_strategy(
            bars_subset, start_date, end_date, initial_capital, weight_fn,
            rebalance_every_days=1,
        )
        self._num_trades = n_trades
        return curve

    def num_trades(self) -> int:
        return getattr(self, "_num_trades", 0)
