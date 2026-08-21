"""CAPM Alpha Ranking on Dow 30, from QuantConnect's public Investment
Strategy Library (quantconnect.com/learning/articles/investment-strategy-library
/capm-alpha-ranking-strategy-on-dow-30-companies).

Runs a rolling CAPM regression of each stock's daily returns against the
equal-weight Dow-30 market return, ranks by alpha (the return unexplained by
market movement), and holds the top 2 alpha generators equally weighted,
rebalanced monthly.

Lookback: uses up to 60 trading days of returns for the regression, capped to
whatever history is available. On the leaderboard's ~1-month contest window
this typically has enough of the ~1-month reference buffer to run, unlike the
longer-lookback strategies in this same batch.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from dashboard.backend.infrastructure.llm.validator import DJIA_30

from .base import BaselineStrategy
from ._signal_engine import DailyHistory, available_window, run_daily_signal_strategy

_LOOKBACK_DAYS = 60
_MIN_HISTORY = 30
_REBALANCE_DAYS = 21


class CAPMAlphaRankingStrategy(BaselineStrategy):
    key = "capm_alpha_ranking"

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
            window = available_window(history, _LOOKBACK_DAYS)
            if window < _MIN_HISTORY:
                return {}
            close = history.close.iloc[-window:]
            returns = close.pct_change().dropna(how="all")
            if returns.shape[0] < _MIN_HISTORY - 1:
                return {}
            mkt_ret = returns.mean(axis=1)
            if mkt_ret.std() == 0:
                return {}
            x_c = mkt_ret - mkt_ret.mean()
            var_x = (x_c ** 2).sum()

            alphas: Dict[str, float] = {}
            for sym in returns.columns:
                ys = returns[sym]
                valid = ys.notna() & mkt_ret.notna()
                if valid.sum() < _MIN_HISTORY - 1:
                    continue
                beta = (x_c[valid] * (ys[valid] - ys[valid].mean())).sum() / var_x
                alphas[sym] = ys[valid].mean() - beta * mkt_ret[valid].mean()
            if not alphas:
                return {}
            top2 = pd.Series(alphas).sort_values(ascending=False).head(2)
            return {sym: 0.5 for sym in top2.index}

        curve, n_trades = run_daily_signal_strategy(
            bars_subset, start_date, end_date, initial_capital, weight_fn,
            rebalance_every_days=_REBALANCE_DAYS,
        )
        self._num_trades = n_trades
        return curve

    def num_trades(self) -> int:
        return getattr(self, "_num_trades", 0)
