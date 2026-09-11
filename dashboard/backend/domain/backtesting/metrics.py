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
        Annualized Sharpe ratio. Returns 0 if insufficient data, zero volatility,
        or an undefined per-step return (see below).
    """
    if len(equity_curve) < 2:
        return 0

    equities = np.array([e["equity"] for e in equity_curve], dtype=float)
    # A step whose *opening* equity is 0 has no defined return: the $0 runs made
    # legal on 2026-09-10 produce an all-zero curve, so every step is 0/0.
    #
    # This cannot be left to the zero-volatility exit below, because
    # ``np.std(nan) == 0`` is False -- NaN would fall straight through it and be
    # written to ``agent_runs.sharpe_ratio``. That is invisible on SQLite, which
    # coerces NaN to NULL on write, and fatal on the Postgres run-history
    # backend (``AGENT_RUNS_DATABASE_URL``), which round-trips it: Starlette
    # encodes with ``allow_nan=False``, so every route returning a plain dict
    # rather than a ``response_model`` answers 500 for that run.
    #
    # 0 is the same convention ``fractional_return`` settled on for the return
    # itself: with no capital there is nothing to have been at risk, so the
    # honest answer to "how was it risk-adjusted" is "it wasn't".
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.diff(equities) / equities[:-1]

    if len(returns) == 0 or not np.all(np.isfinite(returns)):
        return 0
    if np.std(returns) == 0:
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
        # No peak means no drawdown from it. An unfunded run sits at 0 for its
        # whole curve, making this ``(0 - 0) / 0``. The old code already landed
        # on the right answer, but only by accident -- ``nan < max_dd`` is
        # False, so max_dd stayed 0 -- while numpy printed a RuntimeWarning into
        # the stdout of every $0 backtest.
        if running_max == 0:
            continue
        dd = (equity - running_max) / running_max
        if dd < max_dd:
            max_dd = dd

    return max_dd
