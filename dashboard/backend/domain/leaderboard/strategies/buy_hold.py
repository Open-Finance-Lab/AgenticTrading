"""Buy & hold a fixed set of symbols (equal dollar allocation, held to the end).

Used for single-name buy & hold (e.g. SPY, AAPL) — set ``symbols`` in config.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from dashboard.backend.baseline_generator import BaselineGenerator

from .base import BaselineStrategy
from ._common import subset_bars


class BuyHoldStrategy(BaselineStrategy):
    key = "buy_hold"

    def required_symbols(self) -> List[str]:
        return list(self.config.get("symbols") or [])

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
        return BaselineGenerator().generate_buyhold_baseline(
            bars_subset, start_date, end_date, initial_capital, symbols
        )

    def num_trades(self) -> int:
        return len(self.required_symbols())
