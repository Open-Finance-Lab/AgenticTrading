"""Shared backtesting / capital constants.

Canonical home (Phase 2C4) for backtesting constants used by backend consumers.
Moved from ``dashboard/scripts/backtest_hourly_agent.py`` and re-exported there
for backward compatibility (``bha.INITIAL_CAPITAL``).

``LLM_MODEL_NAME`` intentionally lives in
``dashboard.backend.infrastructure.llm.backtest_harness`` and is NOT duplicated
here.

Capital scale (product):
- Account portfolio equity: ``DEFAULT_PORTFOLIO_EQUITY`` ($10,000). For
  signed-in users this is a real ledger budget (unallocated cash ↔ agent
  sleeves via allocate/reclaim). Guests/demo still treat it as display-only.
- Per-agent sleeve (``cash_allocation``): default
  ``DEFAULT_AGENT_CASH_ALLOCATION`` ($1,000), max ``MAX_AGENT_CASH_ALLOCATION``
  ($3,000) — also capped by remaining unallocated cash. **Paper trading only**;
  backtests must not read or write this field.
- Backtest / protocol simulation capital: floor ``MIN_BACKTEST_INITIAL_CAPITAL``
  ($1), default ``INITIAL_CAPITAL`` ($1,000), max
  ``MAX_BACKTEST_INITIAL_CAPITAL`` ($3,000). Chosen per run via
  ``config.initial_cash`` / ``--initial-capital``, resolved by
  ``resolve_initial_capital()`` below, which only ever looks at the value
  passed in for *that* run — it does not read the agent row. The per-agent
  ``backtest_allocation`` field (set from the Configure screen's Allocated
  Capital card) is stored on the agent and supplied by the dashboard as that
  requested value when it launches a backtest for the agent; it is not a
  backend fallback. Either way, backtest capital is independent of the
  account ledger and agent sleeves.
"""

from __future__ import annotations

from typing import Any, Optional

# Default starting capital for a backtest when none is requested.
INITIAL_CAPITAL = 1000

# Hard floor on simulation capital for a single backtest / protocol run.
MIN_BACKTEST_INITIAL_CAPITAL = 1

# Hard cap on simulation capital for a single backtest / protocol run.
MAX_BACKTEST_INITIAL_CAPITAL = 3_000

# Account portfolio equity ($10k). Real ledger budget for signed-in users.
DEFAULT_PORTFOLIO_EQUITY = 10_000

# Per-agent cash allocation bounds (also enforced by the agents API).
# These bound the paper-trading sleeve only — not backtest capital.
DEFAULT_AGENT_CASH_ALLOCATION = 1000
MAX_AGENT_CASH_ALLOCATION = 3_000


def resolve_initial_capital(requested: Optional[Any] = None) -> float:
    """Resolve simulation capital for a backtest / protocol run.

    Independent of agent ``cash_allocation`` (portfolio sleeve). Uses
    ``requested`` when present and positive, clamped to
    ``MAX_BACKTEST_INITIAL_CAPITAL``. Otherwise returns ``INITIAL_CAPITAL``.
    """
    if requested is None:
        return float(INITIAL_CAPITAL)
    try:
        value = float(requested)
    except (TypeError, ValueError):
        return float(INITIAL_CAPITAL)
    if value <= 0:
        return float(INITIAL_CAPITAL)
    return float(min(value, MAX_BACKTEST_INITIAL_CAPITAL))
