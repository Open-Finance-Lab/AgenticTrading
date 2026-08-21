"""Overnight Anomaly, from QuantConnect's public Investment Strategy Library
(quantconnect.com/learning/articles/investment-strategy-library/overnight-anomaly).

Buys SPY at every day's close and sells at the next day's open, capturing
only the overnight return and skipping the regular trading session entirely
-- tests whether stock gains concentrate outside market hours.

Single-instrument, precise intraday timing (buy at the literal close price,
sell at the literal open price), so this runs directly against the hourly
bars rather than through the shared daily-signal engine used by the other
strategies in this batch -- there is no periodic weight to compute, just two
scheduled trades a day.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from .base import BaselineStrategy
from ._common import market_timestamps, timestamp_date, timestamps_in_contest

_DEFAULT_SYMBOL = "SPY"


class OvernightAnomalyStrategy(BaselineStrategy):
    key = "overnight_anomaly"

    def _symbol(self) -> str:
        symbols = self.config.get("symbols")
        return symbols[0] if symbols else _DEFAULT_SYMBOL

    def required_symbols(self) -> List[str]:
        return [self._symbol()]

    def run(
        self,
        bars_by_symbol: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        initial_capital: float,
    ) -> List[Dict[str, Any]]:
        symbol = self._symbol()
        df = bars_by_symbol.get(symbol)
        if df is None or df.empty:
            return []

        all_ts = market_timestamps({symbol: df})
        contest_ts = timestamps_in_contest(all_ts, start_date, end_date)
        if not contest_ts:
            return []

        first_of_day: Dict[Any, Any] = {}
        last_of_day: Dict[Any, Any] = {}
        for ts in contest_ts:
            d = timestamp_date(ts)
            first_of_day.setdefault(d, ts)
            last_of_day[d] = ts

        cash = float(initial_capital)
        shares = 0.0
        curve: List[Dict[str, Any]] = []
        n_trades = 0

        for ts in contest_ts:
            d = timestamp_date(ts)
            if ts == first_of_day[d] and shares > 0:
                sell_price = float(df.loc[ts, "open"])
                cash = shares * sell_price
                shares = 0.0
                n_trades += 1
            if ts == last_of_day[d] and shares == 0:
                buy_price = float(df.loc[ts, "close"])
                if buy_price > 0:
                    shares = cash / buy_price
                    cash = 0.0
                    n_trades += 1

            mark_price = float(df.loc[ts, "close"])
            positions_value = shares * mark_price
            curve.append(
                {
                    "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                    "equity": round(cash + positions_value, 2),
                    "cash": round(cash, 2),
                    "positions_value": round(positions_value, 2),
                    "daily_return": 0,
                }
            )

        self._num_trades = n_trades
        return curve

    def num_trades(self) -> int:
        return getattr(self, "_num_trades", 0)
