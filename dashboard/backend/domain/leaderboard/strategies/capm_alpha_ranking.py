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

from .base import BaselineStrategy
from ._common import subset_bars
from ._signal_engine import DailyHistory, available_window, run_daily_signal_strategy

_LOOKBACK_DAYS = 60
_MIN_HISTORY = 30
_REBALANCE_DAYS = 21


class CAPMAlphaRankingStrategy(BaselineStrategy):
    key = "capm_alpha_ranking"

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
            window = available_window(history, _LOOKBACK_DAYS)
            if window < _MIN_HISTORY:
                return {}
            close = history.close.iloc[-window:]
            # fill_method=None: a missing close must not be padded into a 0%
            # return day (pandas' default pads), which would bias the regression.
            returns = close.pct_change(fill_method=None).dropna(how="all")
            if returns.shape[0] < _MIN_HISTORY - 1:
                return {}
            mkt_ret = returns.mean(axis=1)
            if mkt_ret.std() == 0:
                return {}
            alphas: Dict[str, float] = {}
            for sym in returns.columns:
                ys = returns[sym]
                valid = ys.notna() & mkt_ret.notna()
                if valid.sum() < _MIN_HISTORY - 1:
                    continue
                # OLS on the symbol's own valid days: centring and variance must
                # use the same rows as the covariance, or a symbol with gaps has
                # its beta divided by the full window's variance and understated.
                x_v = mkt_ret[valid]
                y_v = ys[valid]
                x_c = x_v - x_v.mean()
                var_x = (x_c ** 2).sum()
                if var_x <= 0:
                    continue
                beta = (x_c * (y_v - y_v.mean())).sum() / var_x
                alphas[sym] = y_v.mean() - beta * x_v.mean()
            if not alphas:
                return {}
            top2 = pd.Series(alphas).sort_values(ascending=False).head(2)
            if top2.empty:
                return {}
            # Equal weight over however many made the cut, so a single
            # qualifying name gets the whole book rather than leaving half in cash.
            return {sym: 1.0 / len(top2) for sym in top2.index}

        curve, n_trades = run_daily_signal_strategy(
            bars_subset, start_date, end_date, initial_capital, weight_fn,
            rebalance_every_days=_REBALANCE_DAYS,
        )
        self._num_trades = n_trades
        return curve
