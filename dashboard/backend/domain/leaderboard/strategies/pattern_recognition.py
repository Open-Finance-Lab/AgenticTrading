"""PatternRecognition, from the freqtrade-strategies community repository
(github.com/freqtrade/freqtrade-strategies, user_data/strategies
/PatternRecognition.py).

Buys when a stock forms a 'high wave' candle (a small price-change body with
long shadows both above and below, signaling indecision) at a fresh 10-day
low. TA-Lib's CDLHIGHWAVE definition is reproduced directly from its OHLC
ratios rather than depending on the TA-Lib C library. The original strategy
has no sell logic of its own (it relies on an ROI table/stoploss not modeled
here), so a fixed 10-trading-day hold is used as the exit -- a default this
strategy needed and the source didn't specify.

Lookback: needs ~10 days for the rolling low; below that, holds whatever is
already held and takes no new positions.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from dashboard.backend.infrastructure.llm.validator import DJIA_30

from .base import BaselineStrategy
from ._signal_engine import DailyHistory, make_entry_exit_weight_fn, run_daily_signal_strategy

_MAX_POSITIONS = 8
_MIN_HISTORY = 10
_HOLD_DAYS = 10


class PatternRecognitionStrategy(BaselineStrategy):
    key = "pattern_recognition"

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

        entry_day: Dict[str, int] = {}

        def entry(history: DailyHistory, sym: str) -> bool:
            close = history.close[sym].dropna()
            if len(close) < _MIN_HISTORY:
                return False
            open_ = history.open[sym].reindex(close.index)
            high = history.high[sym].reindex(close.index)
            low = history.low[sym].reindex(close.index)
            rng = high.iloc[-1] - low.iloc[-1]
            if not rng:
                return False
            body = abs(close.iloc[-1] - open_.iloc[-1]) / rng
            upper_shadow = (high.iloc[-1] - max(open_.iloc[-1], close.iloc[-1])) / rng
            lower_shadow = (min(open_.iloc[-1], close.iloc[-1]) - low.iloc[-1]) / rng
            is_high_wave = body < 0.3 and upper_shadow > 0.3 and lower_shadow > 0.3
            at_recent_low = close.iloc[-1] <= close.iloc[-_MIN_HISTORY:].min() * 1.01
            signal = bool(is_high_wave and at_recent_low)
            if signal:
                entry_day[sym] = len(close)
            return signal

        def exit_(history: DailyHistory, sym: str) -> bool:
            close = history.close[sym].dropna()
            held_since = entry_day.get(sym)
            if held_since is None:
                return False
            return len(close) - held_since >= _HOLD_DAYS

        weight_fn = make_entry_exit_weight_fn(entry, exit_, symbols, _MAX_POSITIONS, _MIN_HISTORY)
        curve, n_trades = run_daily_signal_strategy(
            bars_subset, start_date, end_date, initial_capital, weight_fn,
            rebalance_every_days=1,
        )
        self._num_trades = n_trades
        return curve

    def num_trades(self) -> int:
        return getattr(self, "_num_trades", 0)
