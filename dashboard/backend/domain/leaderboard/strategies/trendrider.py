"""TrendRiderStrategy (simplified), from the freqtrade-strategies community
repository (github.com/freqtrade/freqtrade-strategies, user_data/strategies
/TrendRiderStrategy.py).

Enters on any of three trend-confirmation signals (a golden cross of the
10/50-day EMA, an RSI bounce off oversold while price holds above the
200-day average, or a MACD histogram turning positive); exits on RSI
overheating, a bearish EMA cross, or price falling back below the 200-day
average.

Simplification: the original strategy has 6 entry patterns and 4 exits, plus
a cascading time-based profit exit -- reduced here to the 3 clearest
generic entries and 3 clearest exits. The original also gates every entry on
BTC RSI>35 and a crypto Fear & Greed Index band; both are crypto-market-
structure-specific regime filters with no equities equivalent, and are
dropped here rather than replaced with an invented substitute.

Lookback: needs ~200 days for the SMA-200 term (capped to whatever is
available -- see the module docstring in ``_signal_engine.py`` for why the
leaderboard's short contest window makes this term behave differently than
in the full-year "Strategy Lab" backtest).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from dashboard.backend.infrastructure.llm.validator import DJIA_30

from .base import BaselineStrategy
from ._indicators import ema, macd, rsi
from ._signal_engine import DailyHistory, available_window, make_entry_exit_weight_fn, run_daily_signal_strategy

_MAX_POSITIONS = 8
_MIN_HISTORY = 25


def _series(history: DailyHistory, sym: str):
    close = history.close[sym].dropna().to_frame()
    ema10 = ema(close, 10)
    ema50 = ema(close, 50)
    sma200 = close.rolling(available_window(history, 200)).mean()
    _, _, hist_line = macd(close)
    rsi14 = rsi(close, 14)
    return close, ema10, ema50, sma200, hist_line, rsi14


class TrendRiderStrategy(BaselineStrategy):
    key = "trendrider"

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
            c, ema10, ema50, sma200, hist_line, rsi14 = _series(history, sym)
            golden_cross = (
                ema10.iloc[-1, 0] > ema50.iloc[-1, 0] and ema10.iloc[-2, 0] <= ema50.iloc[-2, 0]
            )
            rsi_bounce = (
                rsi14.iloc[-2, 0] < 30 and rsi14.iloc[-1, 0] >= 30 and c.iloc[-1, 0] > sma200.iloc[-1, 0]
            )
            macd_cross = hist_line.iloc[-1, 0] > 0 and hist_line.iloc[-2, 0] <= 0
            return bool(golden_cross or rsi_bounce or macd_cross)

        def exit_(history: DailyHistory, sym: str) -> bool:
            close = history.close[sym].dropna()
            if len(close) < _MIN_HISTORY + 1:
                return False
            c, ema10, ema50, sma200, _, rsi14 = _series(history, sym)
            bearish_cross = ema10.iloc[-1, 0] < ema50.iloc[-1, 0]
            return bool(
                rsi14.iloc[-1, 0] > 78
                or bearish_cross
                or c.iloc[-1, 0] < sma200.iloc[-1, 0] * 0.99
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
