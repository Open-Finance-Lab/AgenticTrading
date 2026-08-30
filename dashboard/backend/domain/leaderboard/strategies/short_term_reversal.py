"""Short-Term Reversal (long-only adaptation), from QuantConnect's public
Investment Strategy Library (quantconnect.com/learning/articles/investment
-strategy-library/short-term-reversal).

Ranks stocks by their prior month's return and holds the 10 worst performers
equally weighted, rebalanced monthly, betting on short-term mean reversion.
The original strategy also shorts the 10 best performers; that leg is
dropped here since every strategy in this leaderboard (and the rest of the
dashboard's backtest engine) is long-only.

Lookback caveat: wants a 21-trading-day prior-month return, capped to
whatever history is available.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from .base import BaselineStrategy
from ._common import subset_bars
from ._signal_engine import DailyHistory, run_daily_signal_strategy

_TOP_N = 10
_DESIRED_LOOKBACK_DAYS = 21
_MIN_HISTORY = 5
_REBALANCE_DAYS = 21


class ShortTermReversalStrategy(BaselineStrategy):
    key = "short_term_reversal"

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

        def weight_fn(history: DailyHistory, cur_date, day_index):
            n = len(history)
            if n < _MIN_HISTORY:
                return {}
            lookback = min(_DESIRED_LOOKBACK_DAYS, n - 1)
            close = history.close
            prior_month_return = close.iloc[-1] / close.iloc[-1 - lookback] - 1
            bottom = prior_month_return.dropna().sort_values().head(_TOP_N)
            if bottom.empty:
                return {}
            return {sym: 1.0 / len(bottom) for sym in bottom.index}

        curve, n_trades = run_daily_signal_strategy(
            bars_subset, start_date, end_date, initial_capital, weight_fn,
            rebalance_every_days=_REBALANCE_DAYS,
        )
        self._num_trades = n_trades
        return curve
