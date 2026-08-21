"""Volatility Effect in Stocks, from QuantConnect's public Investment
Strategy Library (quantconnect.com/learning/articles/investment-strategy-library
/volatility-effect-in-stocks).

Computes each stock's trailing return volatility and holds the 5
lowest-volatility names equally weighted, rebalanced monthly -- a defensive,
low-volatility-anomaly strategy.

Lookback caveat: wants a full 252-day trailing volatility window. On the
leaderboard's ~1-month contest window (with a matching ~1-month reference
buffer), this degrades to whatever window the available history supports.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from dashboard.backend.infrastructure.llm.validator import DJIA_30

from .base import BaselineStrategy
from ._signal_engine import DailyHistory, run_daily_signal_strategy

_TOP_N = 5
_DESIRED_LOOKBACK_DAYS = 252
_MIN_HISTORY = 10
_REBALANCE_DAYS = 21


class VolatilityEffectStrategy(BaselineStrategy):
    key = "volatility_effect"

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
            lookback = min(_DESIRED_LOOKBACK_DAYS, n)
            returns = history.close.pct_change().iloc[-lookback:]
            vol = returns.std().dropna()
            bottom = vol.sort_values().head(_TOP_N)
            if bottom.empty:
                return {}
            return {sym: 1.0 / len(bottom) for sym in bottom.index}

        curve, n_trades = run_daily_signal_strategy(
            bars_subset, start_date, end_date, initial_capital, weight_fn,
            rebalance_every_days=_REBALANCE_DAYS,
        )
        self._num_trades = n_trades
        return curve

    def num_trades(self) -> int:
        return getattr(self, "_num_trades", 0)
