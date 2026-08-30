"""Bandtastic, from the freqtrade-strategies community repository
(github.com/freqtrade/freqtrade-strategies, user_data/strategies/Bandtastic.py).

Buys a stock when its price falls below its 20-day Bollinger lower band
while RSI stays under 52 and its 10-day EMA is above its 50-day EMA (an
uptrend-confirmed dip); sells on the mirror-image condition at the upper
band. Pure generic technical-analysis logic (Bollinger/RSI/EMA), portable
from freqtrade's crypto pairs to equities without modification.

Lookback: wants 50 days for the slower EMA, but the leaderboard's reference
buffer plus contest window is ~44 trading days, so a hard 50-day floor would
never trade. The slow EMA's span is capped at the history actually available
(see ``_signal_engine``'s lookback caveat); the 20-day Bollinger window is the
real floor, below which the strategy holds whatever it holds and takes no new
positions.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from .base import BaselineStrategy
from ._common import subset_bars
from ._indicators import bollinger, ema, rsi
from ._signal_engine import DailyHistory, make_entry_exit_weight_fn, run_daily_signal_strategy

_MAX_POSITIONS = 8
_SLOW_EMA_DAYS = 50
_MIN_HISTORY = 20  # the Bollinger window; the slow EMA degrades down to this


def _slow_ema_span(close: pd.Series) -> int:
    """The slow EMA's span capped at this symbol's own non-NaN daily rows (not the
    union frame's length -- ``ema`` sets ``min_periods=span``, so a span above
    the row count is all-NaN and every comparison against it is False)."""
    return min(_SLOW_EMA_DAYS, len(close))


class BandtasticStrategy(BaselineStrategy):
    key = "bandtastic"

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

        def entry(history: DailyHistory, sym: str) -> bool:
            close = history.close[sym].dropna()
            if len(close) < _MIN_HISTORY:
                return False
            _, _, bb_lower, _ = bollinger(close.to_frame(), 20, 2.0)
            rsi14 = rsi(close.to_frame(), 14)
            ema10 = ema(close.to_frame(), 10)
            ema50 = ema(close.to_frame(), _slow_ema_span(close))
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
            ema50 = ema(close.to_frame(), _slow_ema_span(close))
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
