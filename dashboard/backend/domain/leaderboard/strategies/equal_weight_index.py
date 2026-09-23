"""Equal-weight index across a universe (default DJIA 30), continuously rebalanced.

This tracks the average per-symbol return, i.e. an equal-weight index that is
effectively rebalanced every bar — distinct from equal-weight buy & hold whose
weights drift over time.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from dashboard.backend.baseline_generator import BaselineGenerator
from dashboard.backend.infrastructure.llm.validator import DJIA_30

from .base import BaselineStrategy


class EqualWeightIndexStrategy(BaselineStrategy):
    key = "equal_weight_index"

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
        return BaselineGenerator().generate_index_baseline(
            bars_by_symbol, start_date, end_date, initial_capital, symbols
        )
