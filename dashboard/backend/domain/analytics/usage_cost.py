"""The cost properties of a ``model_usage_recorded`` event, written and read in one place.

``cost_micro_usd`` is ATL cost: what the platform lane debited, in
micro-Credits ($1 = 1 Credit). A BYOK call debits nothing, so it carries 0.

``estimated_cost_micro_usd`` is BYOK-only: what the same tokens would have
debited at the price table's listed rate. It is omitted -- the call is
*unpriced* -- when the provider reported no usage or the table does not list
the model, and the admin Credits panel counts those calls instead of drawing
them as a real 0.

Both writers -- the live emitter (``LLMExecutionService._emit_model_usage``)
and the history backfill (``backfill._usage_candidate``) -- build the
properties here, so the two cannot drift apart. Events written between PR #572
and this split carried the BYOK estimate in ``cost_micro_usd`` itself; the two
readers below absorb that era, so no other reader has to know it existed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dashboard.backend.infrastructure.llm.execution.models import LLMUsage
from dashboard.backend.infrastructure.llm.token_cost import list_price_estimate_usd

ESTIMATED_COST_PROPERTY = "estimated_cost_micro_usd"


def _micro(usd: float) -> int:
    return max(0, round(usd * 1_000_000))


def model_usage_properties(
    *,
    billing_mode: str,
    model_id: str | None,
    input_tokens: int,
    output_tokens: int,
    usage_available: bool,
    provider_cost_usd: float | None,
    estimated_cost_usd: float | None,
) -> dict[str, int]:
    """The properties of one ``model_usage_recorded`` event.

    The platform lane records its debit: the provider-reported cost when there
    is one, else the call's own estimate -- the same formula the ledger uses.
    """
    properties = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_micro_usd": 0,
    }
    if billing_mode == "platform_credits":
        properties["cost_micro_usd"] = _micro(
            provider_cost_usd
            if provider_cost_usd is not None
            else estimated_cost_usd or 0.0
        )
    elif billing_mode == "byok":
        estimate = list_price_estimate_usd(
            model_id,
            LLMUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                usage_available=usage_available,
            ),
        )
        if estimate is not None:
            properties[ESTIMATED_COST_PROPERTY] = _micro(estimate)
    return properties


def atl_cost_micro(billing_mode: str | None, properties: Mapping[str, Any]) -> int:
    """ATL cost of one call: the platform debit, and 0 for every other lane."""
    if billing_mode != "platform_credits":
        return 0
    return int(properties.get("cost_micro_usd", 0))


def byok_estimate_micro(properties: Mapping[str, Any]) -> int | None:
    """A BYOK call's list-price estimate, or None when the call is unpriced."""
    if ESTIMATED_COST_PROPERTY in properties:
        return int(properties[ESTIMATED_COST_PROPERTY])
    # PR #572 wrote the estimate into cost_micro_usd; before it, BYOK had none.
    legacy = int(properties.get("cost_micro_usd", 0))
    return legacy if legacy > 0 else None
