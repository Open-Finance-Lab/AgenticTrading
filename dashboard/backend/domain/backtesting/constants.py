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
  ($0), default ``INITIAL_CAPITAL`` ($1,000), max
  ``MAX_BACKTEST_INITIAL_CAPITAL`` ($3,000). Chosen per run via
  ``config.initial_cash`` / ``--initial-capital``, resolved by
  ``resolve_initial_capital()`` below, which only ever looks at the value
  passed in for *that* run — it does not read the agent row. The per-agent
  ``backtest_allocation`` field (set from the Configure screen's Allocated
  Capital card) is stored on the agent and supplied by the dashboard as that
  requested value when it launches a backtest for the agent; it is not a
  backend fallback. Either way, backtest capital is independent of the
  account ledger and agent sleeves.

⚠ **$0 is a legal amount and zero is not absent.** The floor was $1 until
2026-09-10, which put the two halves of one Configure card in disagreement:
``cash_allocation`` took $0 everywhere and ``backtest_allocation`` refused it
at seven separate layers, the loudest a 422 and the quietest this module's own
``if value <= 0: return INITIAL_CAPITAL``. That last one is the shape to keep
out: it *accepted* the 0 and then ran $1,000, so the field and the run
disagreed with nothing to read. Absent (``None``, ``""``, unparseable) still
falls back to ``INITIAL_CAPITAL``; an explicit 0 does not, and every ``or``
chain on this path (``run["initial_equity"] or …``) collapses that distinction
by accident — use ``is None``.

A $0 run is degenerate but real: no cash means no fills, so every step holds,
the curve is flat at $0 and the honest return is 0.00%. Computing that return
is what ``fractional_return()`` exists for — see its own note.
"""

from __future__ import annotations

from typing import Any, Optional

# Default starting capital for a backtest when none is requested.
INITIAL_CAPITAL = 1000

# Hard floor on simulation capital for a single backtest / protocol run.
# Zero is deliberately inclusive; see the ⚠ note in the module docstring.
MIN_BACKTEST_INITIAL_CAPITAL = 0

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
    ``requested`` when it is a number at or above
    ``MIN_BACKTEST_INITIAL_CAPITAL``, clamped to
    ``MAX_BACKTEST_INITIAL_CAPITAL``. Otherwise returns ``INITIAL_CAPITAL``.

    ``0`` is a value and is returned as ``0.0``; only *absent* (``None``, ``""``,
    anything unparseable) and *negative* fall back to the default. The old
    ``value <= 0`` lumped zero in with those, so a 0 that reached here ran as
    $1,000 with nothing reporting the substitution.
    """
    if requested is None:
        return float(INITIAL_CAPITAL)
    try:
        value = float(requested)
    except (TypeError, ValueError):
        return float(INITIAL_CAPITAL)
    # NaN fails every comparison, so test for membership in the range rather
    # than for exclusion from it -- `value < MIN` is False for NaN and would
    # let it through to the clamp, where `min(nan, 3000)` returns nan.
    if not value >= MIN_BACKTEST_INITIAL_CAPITAL:
        return float(INITIAL_CAPITAL)
    return float(min(value, MAX_BACKTEST_INITIAL_CAPITAL))


def fractional_return(final_equity: float, initial_capital: float) -> float:
    """Total return as a fraction, safe at zero capital.

    ``(final - initial) / initial`` appears at five sites (``engine.py`` x4,
    ``external_run_service.py`` x1) and was unguarded at all five. The $1 floor
    on ``MIN_BACKTEST_INITIAL_CAPITAL`` was the only thing keeping a typed 0
    away from that division, so lifting the floor without this helper would
    have traded a 422 the user could read for a ``ZeroDivisionError`` partway
    through their run.

    Zero capital returns ``0.0`` rather than raising or reporting ``inf``: with
    no cash there are no fills, so the honest answer to "how did it do" is
    "it didn't". Reporting ``inf`` on the arithmetic technicality that
    ``final`` may be nonzero would put a garbage number on the run card.
    """
    initial = float(initial_capital)
    if initial == 0:
        return 0.0
    return (float(final_equity) - initial) / initial
