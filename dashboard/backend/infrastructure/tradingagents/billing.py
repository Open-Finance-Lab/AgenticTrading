"""ATL Credits metering for the hosted TradingAgents runtime.

The same unit and lifecycle as AI Hedge Fund (``ai_hedge_fund/billing.py``,
whose ``HostedDecisionBilling`` this reuses): upstream calls its models inside
its own interpreter, so one decision (one bridge subprocess) is reserved at a
ceiling before it runs and settled at the token usage the bridge's meter
reports after it. Only the ceiling is TradingAgents-specific, because the call
graph is.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from dashboard.backend.infrastructure.ai_hedge_fund.billing import (
    HostedDecisionBilling,
)
from dashboard.backend.infrastructure.llm.token_cost import (
    credits_micro_for_usd,
    estimate_cost_usd,
)

SOURCE = "tradingagents"

# One upstream pass, measured: 8 fixed nodes (bull, bear, research manager,
# trader, three risk debators, portfolio manager) at the pinned 1/1 debate
# rounds, plus each selected analyst's tool loop -- 3 calls for the market
# analyst, budgeted at 4 here because a tool loop's length is the model's
# choice. The SDK's generator retries a failed date once inside the same
# subprocess (``for attempt in (1, 2)``), so a decision can be two passes.
FIXED_CALLS_PER_PASS = 8
CALLS_PER_ANALYST = 4
PASSES_PER_DECISION = 2

# Per-call budget behind the ceiling. Generous on purpose, as in AI Hedge
# Fund's: a ceiling below the real spend is capped at settlement (the operator
# eats the rest) rather than recorded as an overage. TradingAgents prompts
# carry every earlier report and both debate histories, so the later nodes'
# inputs are several times the analysts'.
CEILING_INPUT_TOKENS_PER_CALL = 16_000
CEILING_OUTPUT_TOKENS_PER_CALL = 2_000


def decision_calls(analysts: Optional[Sequence[str]]) -> int:
    """The model calls one decision's ceiling budgets for."""
    analyst_count = max(len(analysts or ()), 1)
    return (FIXED_CALLS_PER_PASS + CALLS_PER_ANALYST * analyst_count) * (
        PASSES_PER_DECISION
    )


def decision_ceiling_micro(model: str, analysts: Optional[Sequence[str]]) -> int:
    """The most one decision is held for, in Credit micro-units.

    One formula for the launch preflight and the per-decision hold, so the
    route cannot admit a run whose decisions the child then refuses.
    """
    calls = decision_calls(analysts)
    return credits_micro_for_usd(
        estimate_cost_usd(
            model,
            calls * CEILING_INPUT_TOKENS_PER_CALL,
            calls * CEILING_OUTPUT_TOKENS_PER_CALL,
        )
    )


def billing_from_handoff(*, credits: Any, handoff: Any) -> HostedDecisionBilling:
    """The run's meter, recorded against the gateway the route selected.

    The handoff's provider is the gateway the route priced the run on, and the
    child selects its gateway from the same environment, so the hold and the
    account that actually pays name the same lane.
    """
    return HostedDecisionBilling.from_handoff(
        credits=credits,
        handoff=handoff,
        provider_id=handoff.provider_id,
        source=SOURCE,
    )
