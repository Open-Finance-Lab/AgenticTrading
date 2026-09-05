"""Turn of the Month, from QuantConnect's public Investment Strategy Library
(quantconnect.com/learning/articles/investment-strategy-library/turn-of-the
-month-in-equity-indexes).

Buys SPY at the open on the last trading day of each month and holds for 3
trading days before selling at that third day's close -- tests the
documented tendency for stocks to rally around month-end.

Single-instrument, precise intraday timing (buy at the literal open of a
specific day, sell at the literal close of a day 3 trading days later), so
this runs directly against the hourly bars rather than through the shared
daily-signal engine used by the other strategies in this batch.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from .base import BaselineStrategy
from ._common import market_timestamps, timestamp_date, timestamps_in_contest

_DEFAULT_SYMBOL = "SPY"
_HOLD_TRADING_DAYS = 3


class TurnOfMonthStrategy(BaselineStrategy):
    key = "turn_of_month"

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
        # Scalar reads below (df.loc[ts, "open"] / "close") would come back as
        # a Series on a duplicated bar timestamp; keep the last bar per stamp.
        df = df[~df.index.duplicated(keep="last")].sort_index()

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
        unique_dates = sorted(first_of_day.keys())
        # A date is a month-end only when the next sampled date is in a later
        # month. The last sampled date is never one on that evidence alone: the
        # contest window usually ends mid-month, and treating its final day as
        # a month-end bought SPY on the last day of every contest.
        is_last_of_month = {
            d: (i + 1 < len(unique_dates) and unique_dates[i + 1].month != d.month)
            for i, d in enumerate(unique_dates)
        }

        cash = float(initial_capital)
        shares = 0.0
        hold_days_left = 0
        seen_date = None
        curve: List[Dict[str, Any]] = []
        n_trades = 0

        for ts in contest_ts:
            d = timestamp_date(ts)
            if d != seen_date:
                seen_date = d
                if hold_days_left == 0 and is_last_of_month[d] and shares == 0:
                    buy_price = float(df.loc[first_of_day[d], "open"])
                    if buy_price > 0:
                        shares = cash / buy_price
                        cash = 0.0
                        n_trades += 1
                        # Only a real fill starts the hold; a skipped buy must
                        # not block the next candidate day with a phantom hold.
                        hold_days_left = _HOLD_TRADING_DAYS
                elif hold_days_left > 0:
                    hold_days_left -= 1
                    if hold_days_left == 0 and shares > 0:
                        sell_price = float(df.loc[last_of_day[d], "close"])
                        cash = shares * sell_price
                        shares = 0.0
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
