"""Token usage and dollar-cost estimation for agent runs.

External agents run their own LLM client side, so the backend never sees the
real token counts. Instead we estimate input tokens from the market context the
backend serves each hour and output tokens from the decisions the agent submits.
For server-side LLM calls (the internal hourly backtester) we can record the
real usage reported by the provider, so those numbers are exact.

The estimator is deliberately dependency-free (no tiktoken / network calls) so
it can run anywhere. It uses a characters-per-token heuristic that is a good
approximation for the JSON-heavy payloads this app exchanges.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Mapping

from dashboard.backend.infrastructure.llm import pricing as _pricing
from dashboard.backend.infrastructure.llm.execution.models import (
    BillingEvidence,
    BillingMode,
    LLMUsage,
    PricingSnapshot,
)

# JSON / structured text packs slightly more tokens per character than prose.
# ~3.8 chars/token tracks Claude + GPT tokenizers well for this payload shape.
CHARS_PER_TOKEN = 3.8

# The price table and the free-model sentinels live in ``pricing`` — a leaf —
# so ``execution/models`` can price a snapshot without importing this module
# back (CodeQL ``py/cyclic-import``). Explicit assignments rather than a bare
# ``from … import``: the names stay part of this module's public surface
# (``discord_bot`` imports ``is_free_model`` from here), and an assignment
# counts as a *use*, so CodeQL does not report the re-export as
# ``py/unused-import`` (see PR #213).
PRICING_SOURCE_VERSION = _pricing.PRICING_SOURCE_VERSION
is_free_model = _pricing.is_free_model
price_for_model = _pricing.price_for_model

USD_PER_CREDIT = Decimal("1")
CREDITS_MICRO_PER_CREDIT = 1_000_000


def estimate_tokens(value: Any) -> int:
    """Estimate the number of tokens in a string or JSON-serializable object."""
    if value is None:
        return 0
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, separators=(",", ":"), default=str)
        except (TypeError, ValueError):
            text = str(value)
    if not text:
        return 0
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def estimate_cost_usd(model: str | None, input_tokens: int, output_tokens: int) -> float:
    """Estimate the USD cost of a run given token counts and a model name."""
    in_price, out_price = price_for_model(model)
    cost = (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price
    return round(cost, 6)


def normalize_usage(payload: Any) -> LLMUsage:
    """Normalize provider usage shapes without treating missing values as zero."""

    if isinstance(payload, LLMUsage):
        return payload
    value = payload
    if isinstance(payload, Mapping) and isinstance(payload.get("usageMetadata"), Mapping):
        value = payload["usageMetadata"]

    def pick(*names: str) -> Any:
        for name in names:
            if isinstance(value, Mapping) and name in value:
                return value[name]
            candidate = getattr(value, name, None)
            if candidate is not None:
                return candidate
        return None

    input_value = pick("input_tokens", "prompt_tokens", "promptTokenCount")
    output_value = pick("output_tokens", "completion_tokens", "candidatesTokenCount")
    try:
        input_tokens = int(input_value) if input_value is not None else 0
        output_tokens = int(output_value) if output_value is not None else 0
    except (TypeError, ValueError, OverflowError):
        return LLMUsage(input_tokens=0, output_tokens=0, usage_available=False)
    if input_tokens < 0 or output_tokens < 0 or input_value is None or output_value is None:
        return LLMUsage(
            input_tokens=max(input_tokens, 0),
            output_tokens=max(output_tokens, 0),
            usage_available=False,
        )
    return LLMUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usage_available=True,
    )


def estimate_cost_from_snapshot(
    snapshot: PricingSnapshot,
    usage: LLMUsage,
) -> float | None:
    """Calculate exact six-decimal USD cost from the captured price snapshot."""

    if not usage.usage_available:
        return None
    try:
        input_cost = (
            Decimal(usage.input_tokens)
            * Decimal(str(snapshot.input_usd_per_million_tokens))
            / Decimal(1_000_000)
        )
        output_cost = (
            Decimal(usage.output_tokens)
            * Decimal(str(snapshot.output_usd_per_million_tokens))
            / Decimal(1_000_000)
        )
        total = (input_cost + output_cost).quantize(
            Decimal("0.000001"), rounding=ROUND_HALF_UP
        )
    except (InvalidOperation, TypeError, ValueError):
        return None
    return float(total)


def credits_micro_for_usd(cost_usd: float | Decimal | None) -> int:
    """Convert USD to ATL Credit micro-units at the fixed $1 = 1 Credit rate."""

    if cost_usd is None:
        return 0
    try:
        value = Decimal(str(cost_usd))
    except (InvalidOperation, TypeError, ValueError):
        return 0
    if value < 0:
        raise ValueError("cost_usd must not be negative")
    return int(
        (value / USD_PER_CREDIT * CREDITS_MICRO_PER_CREDIT).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    )


def build_cost_evidence(
    *,
    billing_mode: BillingMode,
    provider_id: str,
    model_id: str,
    usage: LLMUsage,
    provider_cost_usd: float | None,
    pricing_snapshot: PricingSnapshot,
) -> BillingEvidence:
    """Build serializable evidence for both billable and BYOK lanes."""

    if (
        pricing_snapshot.provider_id != provider_id
        or pricing_snapshot.model_id != model_id
    ):
        raise ValueError("pricing snapshot does not match provider and model")
    if provider_cost_usd is not None and (
        not math.isfinite(float(provider_cost_usd)) or provider_cost_usd < 0
    ):
        provider_cost_usd = None
    estimated = estimate_cost_from_snapshot(pricing_snapshot, usage)
    if not usage.usage_available:
        authority = "unavailable"
    elif provider_cost_usd is not None:
        authority = "provider_reported_cost"
    elif estimated is not None:
        authority = "provider_usage_pricing_snapshot"
    else:
        authority = "unavailable"
    cost_micro = credits_micro_for_usd(
        provider_cost_usd if provider_cost_usd is not None else estimated
    )
    return BillingEvidence(
        billing_source=billing_mode,
        usage_authority=authority,
        provider_cost_usd=provider_cost_usd,
        estimated_cost_usd=estimated,
        pricing_snapshot=pricing_snapshot,
        provider_cost_credits_micro=(cost_micro if usage.usage_available else 0),
        debited_credits_micro=(
            cost_micro
            if billing_mode is BillingMode.PLATFORM_CREDITS and usage.usage_available
            else 0
        ),
    )


def summarize(
    model: str | None,
    input_tokens: int,
    output_tokens: int,
    llm_calls: int = 0,
) -> dict[str, Any]:
    """Build a serializable token/cost summary for storage or API responses."""
    input_tokens = int(input_tokens or 0)
    output_tokens = int(output_tokens or 0)
    return {
        "model": model,
        "llm_calls": int(llm_calls or 0),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "est_cost_usd": estimate_cost_usd(model, input_tokens, output_tokens),
    }
