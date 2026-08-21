"""UniversalMACD (zero-cross), from the freqtrade-strategies community
repository (github.com/freqtrade/freqtrade-strategies, user_data/strategies
/UniversalMACD.py).

Buys when a normalized MACD ratio (12-day EMA divided by 26-day EMA, minus 1)
crosses from negative to positive, sells on the reverse cross -- a classic
MACD bullish/bearish signal. The original's specific numeric entry/exit
bands were hyperopt-fit to a particular crypto pair's volatility; those
literal numbers are not reused here (re-fitting them for equities would mean
inventing new thresholds), reduced instead to the underlying zero-cross
mechanism, which transfers directly.

Lookback: needs ~27 days for the slower EMA; below that, holds whatever is
already held and takes no new positions.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from dashboard.backend.infrastructure.llm.validator import DJIA_30

from .base import BaselineStrategy
from ._indicators import ema
from ._signal_engine import DailyHistory, make_entry_exit_weight_fn, run_daily_signal_strategy

_MAX_POSITIONS = 8
_MIN_HISTORY = 27


def _umacd(history: DailyHistory, sym: str) -> pd.DataFrame:
    close = history.close[sym].dropna().to_frame()
    return ema(close, 12) / ema(close, 26) - 1


class UniversalMACDStrategy(BaselineStrategy):
    key = "universal_macd"

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
            if len(close) < _MIN_HISTORY + 1:
                return False
            u = _umacd(history, sym)
            return bool(u.iloc[-1, 0] > 0 and u.iloc[-2, 0] <= 0)

        def exit_(history: DailyHistory, sym: str) -> bool:
            close = history.close[sym].dropna()
            if len(close) < _MIN_HISTORY + 1:
                return False
            u = _umacd(history, sym)
            return bool(u.iloc[-1, 0] < 0 and u.iloc[-2, 0] >= 0)

        weight_fn = make_entry_exit_weight_fn(entry, exit_, symbols, _MAX_POSITIONS, _MIN_HISTORY)
        curve, n_trades = run_daily_signal_strategy(
            bars_subset, start_date, end_date, initial_capital, weight_fn,
            rebalance_every_days=1,
        )
        self._num_trades = n_trades
        return curve

    def num_trades(self) -> int:
        return getattr(self, "_num_trades", 0)
