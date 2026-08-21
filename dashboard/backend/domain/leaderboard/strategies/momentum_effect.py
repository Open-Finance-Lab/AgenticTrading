"""Momentum Effect in Stocks, from QuantConnect's public Investment Strategy
Library (quantconnect.com/learning/articles/investment-strategy-library
/momentum-effect-in-stocks).

Ranks stocks by trailing 12-month total return and holds the top 10 equally
weighted, rebalanced monthly -- the classic academic momentum-factor
strategy.

Lookback caveat: wants a full 252-day trailing return. On the leaderboard's
~1-month contest window (with a matching ~1-month reference buffer), this
degrades to whatever trailing return the available history supports -- a
materially shorter-window momentum signal than the 12-month version "Strategy
Lab" backtested over a full year.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from dashboard.backend.infrastructure.llm.validator import DJIA_30

from .base import BaselineStrategy
from ._signal_engine import DailyHistory, run_daily_signal_strategy

_TOP_N = 10
_DESIRED_LOOKBACK_DAYS = 252
_MIN_HISTORY = 5
_REBALANCE_DAYS = 21


class MomentumEffectStrategy(BaselineStrategy):
    key = "momentum_effect"

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

        def weight_fn(history: DailyHistory, cur_date, day_index):
            n = len(history)
            if n < _MIN_HISTORY:
                return {}
            lookback = min(_DESIRED_LOOKBACK_DAYS, n - 1)
            close = history.close
            trailing_return = close.iloc[-1] / close.iloc[-1 - lookback] - 1
            top = trailing_return.dropna().sort_values(ascending=False).head(_TOP_N)
            if top.empty:
                return {}
            return {sym: 1.0 / len(top) for sym in top.index}

        curve, n_trades = run_daily_signal_strategy(
            bars_subset, start_date, end_date, initial_capital, weight_fn,
            rebalance_every_days=_REBALANCE_DAYS,
        )
        self._num_trades = n_trades
        return curve

    def num_trades(self) -> int:
        return getattr(self, "_num_trades", 0)
