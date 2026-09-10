"""Base class for leaderboard baseline strategies.

Each baseline strategy lives in its own module and subclasses ``BaselineStrategy``.
Strategies are independent: a bug in one must never affect another. Shared,
side-effect-free helpers live in ``_common.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd

from dashboard.backend.infrastructure.llm.validator import DJIA_30


class BaselineStrategy(ABC):
    """A single leaderboard baseline strategy.

    Subclasses declare a class-level ``key`` (used by the registry / config
    ``strategy`` field) and implement ``run``; they override
    ``required_symbols`` only when the DJIA-30 default universe doesn't apply.
    """

    key: str = ""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.id = self.config.get("id")
        self.name = self.config.get("name")

    def required_symbols(self) -> List[str]:
        """Symbols this strategy needs market data for: ``config["symbols"]``
        when set, else the DJIA 30 universe."""
        symbols = self.config.get("symbols")
        return list(symbols) if symbols else list(DJIA_30)

    @abstractmethod
    def run(
        self,
        bars_by_symbol: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        initial_capital: float,
    ) -> List[Dict[str, Any]]:
        """Return an hourly equity curve: [{timestamp, equity, cash, positions_value}, ...]."""

    def num_trades(self) -> int:
        """Number of trades this strategy executes (for display only).

        Strategies that count trades set ``self._num_trades`` in ``run()``.
        """
        return getattr(self, "_num_trades", 0)
