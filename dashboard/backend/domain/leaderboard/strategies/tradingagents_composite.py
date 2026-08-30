"""Technical-composite proxy for Tauric Research's TradingAgents multi-agent
framework (https://github.com/TauricResearch/TradingAgents).

TradingAgents is a LangGraph pipeline (fundamental/sentiment/technical/news
analyst agents -> bull/bear researcher debate -> trader -> risk-management
debate -> portfolio manager), not a catalog of named strategies, and it
discloses no numeric decision formula anywhere in its source -- every stage
is an LLM judgment call over qualitative reports. This proxy instead
composites the CONCRETE signals its own market_analyst tool actually computes
(RSI, MACD histogram, 50/200-day SMA position, Bollinger %B) into a single
score, holding the top 8 Dow names by that score. Its sentiment analyst pulls
StockTwits/Reddit data with no data source available here, and is dropped
rather than approximated.

Lookback caveat: uses up to 200 days of history for the SMA-200 term, capped
to whatever is actually available (see ``available_window``). On the
leaderboard's ~1-month contest window (with a matching ~1-month reference
buffer), the 200-day and 50-day terms both degrade to "as much history as
exists," which is a materially shorter-window version of the strategy
"Strategy Lab" backtested over a full year.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from .base import BaselineStrategy
from ._common import subset_bars
from ._indicators import bollinger, macd, rsi, zscore_row
from ._signal_engine import DailyHistory, available_window, run_daily_signal_strategy

_TOP_N = 8
_REBALANCE_DAYS = 5


class TradingAgentsCompositeStrategy(BaselineStrategy):
    key = "tradingagents_composite"

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
            if n < 25:
                return {}
            close = history.close
            rsi14 = rsi(close, 14)  # n >= 25 above, so the 14-day window always fits
            _, _, hist_line = macd(close)
            sma50 = close.rolling(available_window(history, 50)).mean()
            sma200 = close.rolling(available_window(history, 200)).mean()
            _, _, _, pctb = bollinger(close, available_window(history, 20), 2.0)

            last_rsi = rsi14.iloc[-1]
            last_hist = hist_line.iloc[-1]
            last_close = close.iloc[-1]
            last_sma50 = sma50.iloc[-1]
            last_sma200 = sma200.iloc[-1]
            last_pctb = pctb.iloc[-1]

            rsi_signed = 50 - last_rsi
            sma_pos = (last_close - last_sma50) / last_sma50 + (last_close - last_sma200) / last_sma200
            bb = last_pctb - 0.5

            frame = pd.DataFrame(
                {
                    "rsi": zscore_row(rsi_signed),
                    "macd": zscore_row(last_hist),
                    "trend": zscore_row(sma_pos),
                    "bb": zscore_row(-bb),
                }
            )
            composite = frame.mean(axis=1).dropna()
            top = composite.sort_values(ascending=False).head(_TOP_N)
            top = top[top > 0]
            if top.empty:
                return {}
            shifted = top - top.min() + 0.1
            return (shifted / shifted.sum()).to_dict()

        curve, n_trades = run_daily_signal_strategy(
            bars_subset, start_date, end_date, initial_capital, weight_fn,
            rebalance_every_days=_REBALANCE_DAYS,
        )
        self._num_trades = n_trades
        return curve
