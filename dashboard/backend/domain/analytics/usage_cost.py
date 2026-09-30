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
and this split carried the BYOK estimate in ``cost_micro_usd`` itself, so read
that property only through ``atl_cost_micro``: a reader summing it directly
without the lane check adds those estimates into ATL spend. The two readers
below absorb that era -- raw events in ``byok_estimate_micro``, rolled-up
days in ``byok_rollup_lane`` -- so no other reader has to know it existed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dashboard.backend.infrastructure.llm.execution.models import LLMUsage
from dashboard.backend.infrastructure.llm.pricing import listed_price_for_model
from dashboard.backend.infrastructure.llm.token_cost import (
    credits_micro_for_usd,
    list_price_estimate_usd,
)

ESTIMATED_COST_PROPERTY = "estimated_cost_micro_usd"


def _byok_estimate(
    model_id: str | None,
    input_tokens: int,
    output_tokens: int,
    *,
    usage_available: bool,
) -> int | None:
    estimate = list_price_estimate_usd(
        model_id,
        LLMUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usage_available=usage_available,
        ),
    )
    return None if estimate is None else credits_micro_for_usd(estimate)


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

    The platform lane records the call's cost exactly as the ledger books it
    (``token_cost.build_cost_evidence``'s ``provider_cost_credits_micro``):
    the provider-reported cost when there is one, else the call's own
    estimate, converted by ``credits_micro_for_usd`` -- and 0 when the
    provider reported no usage, which the ledger does not debit.
    """
    properties = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_micro_usd": 0,
    }
    if billing_mode == "platform_credits" and usage_available:
        properties["cost_micro_usd"] = credits_micro_for_usd(
            provider_cost_usd if provider_cost_usd is not None else estimated_cost_usd
        )
    elif billing_mode == "byok":
        estimate = _byok_estimate(
            model_id,
            input_tokens,
            output_tokens,
            usage_available=usage_available,
        )
        if estimate is not None:
            properties[ESTIMATED_COST_PROPERTY] = estimate
    return properties


def atl_cost_micro(billing_mode: str | None, properties: Mapping[str, Any]) -> int:
    """ATL cost of one call: the platform debit, and 0 for every other lane."""
    if billing_mode != "platform_credits":
        return 0
    return int(properties.get("cost_micro_usd", 0))


def byok_estimate_micro(
    properties: Mapping[str, Any],
    model_id: str | None,
) -> int | None:
    """A BYOK call's list-price estimate, or None when the call is unpriced."""
    if ESTIMATED_COST_PROPERTY in properties:
        return int(properties[ESTIMATED_COST_PROPERTY])
    # Before PR #572 BYOK carried no estimate, and under it a call with no
    # reported usage recorded 0: both unpriced. Otherwise #572 wrote the call's
    # own estimate into cost_micro_usd, priced off a snapshot that falls back
    # to $1/$5 for a model the table does not list -- so re-price the call's
    # own tokens at the listed rate, and an unlisted model reads as unpriced
    # exactly as it would if written today.
    if int(properties.get("cost_micro_usd", 0)) <= 0:
        return None
    return _byok_estimate(
        model_id,
        int(properties.get("input_tokens", 0)),
        int(properties.get("output_tokens", 0)),
        usage_available=True,
    )


def byok_rollup_lane(
    model_id: str | None,
    *,
    calls: int,
    estimate_micro: int,
    unpriced_calls: int | None,
) -> tuple[int, int]:
    """One provider/model's BYOK rollup for one day, as (estimate, unpriced calls).

    ``unpriced_calls`` is None for a day rolled up before unpriced calls were
    counted, whose gap cannot be split after the fact. With no estimate the day
    predates PR #572, when no BYOK call had one, so every call is unpriced.
    With one, #572 priced it off a snapshot that falls back to $1/$5 for a
    model the table does not list: it stands only where the table lists the
    model, and is otherwise every call unpriced -- the answer
    ``byok_estimate_micro`` gives the same calls read raw.
    """
    if unpriced_calls is not None:
        return estimate_micro, unpriced_calls
    if estimate_micro and listed_price_for_model(model_id) is not None:
        return estimate_micro, 0
    return 0, calls
