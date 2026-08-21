"""Bandtastic, from the freqtrade-strategies community repository
(github.com/freqtrade/freqtrade-strategies, user_data/strategies/Bandtastic.py).

Buys a stock when its price falls below its 20-day Bollinger lower band
while RSI stays under 52 and its 10-day EMA is above its 50-day EMA (an
uptrend-confirmed dip); sells on the mirror-image condition at the upper
band. Pure generic technical-analysis logic (Bollinger/RSI/EMA), portable
from freqtrade's crypto pairs to equities without modification.

Lookback: needs ~50 days for the slower EMA; below that, holds whatever is
already held and takes no new positions.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from dashboard.backend.infrastructure.llm.validator import DJIA_30

from .base import BaselineStrategy
from ._indicators import bollinger, ema, rsi
from ._signal_engine import DailyHistory, make_entry_exit_weight_fn, run_daily_signal_strategy

_MAX_POSITIONS = 8
_MIN_HISTORY = 50


class BandtasticStrategy(BaselineStrategy):
    key = "bandtastic"

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
            close = history.close[sym].dropna()
            if len(close) < _MIN_HISTORY:
                return False
            _, _, bb_lower, _ = bollinger(close.to_frame(), 20, 2.0)
            rsi14 = rsi(close.to_frame(), 14)
            ema10 = ema(close.to_frame(), 10)
            ema50 = ema(close.to_frame(), 50)
            return bool(
                close.iloc[-1] < bb_lower.iloc[-1, 0]
                and rsi14.iloc[-1, 0] < 52
                and ema10.iloc[-1, 0] > ema50.iloc[-1, 0]
            )

        def exit_(history: DailyHistory, sym: str) -> bool:
            close = history.close[sym].dropna()
            if len(close) < _MIN_HISTORY:
                return False
            bb_upper, _, _, _ = bollinger(close.to_frame(), 20, 2.0)
            rsi14 = rsi(close.to_frame(), 14)
            ema10 = ema(close.to_frame(), 10)
            ema50 = ema(close.to_frame(), 50)
            return bool(
                close.iloc[-1] > bb_upper.iloc[-1, 0]
                and rsi14.iloc[-1, 0] > 57
                and ema10.iloc[-1, 0] < ema50.iloc[-1, 0]
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
