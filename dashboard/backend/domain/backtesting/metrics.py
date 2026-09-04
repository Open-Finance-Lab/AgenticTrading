"""Backtest performance metrics.

Extracted (Phase 2A) from ``HourlyBacktester._calc_sharpe`` and
``HourlyBacktester._calc_max_dd`` in ``dashboard/scripts/backtest_hourly_agent.py``.

These are pure functions over an equity curve represented as a list of dicts,
each containing an ``"equity"`` value. Inputs, outputs, edge-case behavior, and
the hourly annualization assumptions are identical to the original methods; the
legacy methods now delegate here.
"""

from typing import Dict, List, Optional

import numpy as np


def calculate_sharpe(
    equity_curve: List[Dict], periods_per_year: Optional[float] = None
) -> float:
    """
    Calculate Sharpe ratio from hourly equity curve.

    Formula:
        sharpe = (mean(returns) / std(returns)) * sqrt(periods_per_year)

    ``periods_per_year`` defaults to the legacy hourly assumption.  A 5-minute
    valuation curve can pass ``252 * 6.5 * 12`` without changing the historical
    hourly behavior.

    Returns: float
        Annualized Sharpe ratio. Returns 0 if insufficient data or zero volatility.
    """
    if len(equity_curve) < 2:
        return 0

    equities = np.array([e["equity"] for e in equity_curve])
    returns = np.diff(equities) / equities[:-1]

    if len(returns) == 0 or np.std(returns) == 0:
        return 0

    periods_per_year = 252 * 6.5 if periods_per_year is None else periods_per_year
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    annualization_factor = np.sqrt(periods_per_year)
    return (np.mean(returns) / np.std(returns)) * annualization_factor


def calculate_max_drawdown(equity_curve: List[Dict]) -> float:
    """Calculate max drawdown."""
    if not equity_curve:
        return 0

    equities = np.array([e["equity"] for e in equity_curve])
    running_max = equities[0]
    max_dd = 0

    for equity in equities:
        if equity > running_max:
            running_max = equity
        dd = (equity - running_max) / running_max
        if dd < max_dd:
            max_dd = dd

    return max_dd
