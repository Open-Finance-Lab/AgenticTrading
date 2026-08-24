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
from typing import Any, Mapping, Tuple

from dashboard.backend.infrastructure.llm.execution.models import (
    BillingEvidence,
    BillingMode,
    LLMUsage,
    PricingSnapshot,
)

# JSON / structured text packs slightly more tokens per character than prose.
# ~3.8 chars/token tracks Claude + GPT tokenizers well for this payload shape.
CHARS_PER_TOKEN = 3.8

# Approximate USD pricing per 1,000,000 tokens (input, output).
# Matched by substring against the run's model name (longest/most specific first).
# "Local" / rule-based models incur no API cost.
_PRICING_TABLE: list[Tuple[str, float, float]] = [
    # CommonStack-verified slugs (provider/model), rates from GET /v1/models on
    # 2026-06-24. Listed first so the specific slug wins over generic needles.
    ("openai/gpt-5.5", 5.0, 30.0),
    ("google/gemini-3.1-pro", 2.0, 12.0),
    ("anthropic/claude-sonnet-4-6", 3.0, 15.0),
    ("deepseek/deepseek-v4-pro", 0.435, 0.87),
    ("qwen/qwen3.7-plus", 0.40, 1.60),
    ("x-ai/grok-4.20-reasoning", 1.25, 2.50),  # listed but unavailable on our account (no channel)
    # OpenRouter-listed (provider/model). Rates from openrouter.ai model pages.
    ("nvidia/nemotron-3-nano-30b-a3b", 0.05, 0.20),
    ("claude-opus-4", 15.0, 75.0),
    ("claude-sonnet-4", 3.0, 15.0),
    ("claude-haiku-4", 1.0, 5.0),
    ("claude-3-7-sonnet", 3.0, 15.0),
    ("claude-3-5-sonnet", 3.0, 15.0),
    ("claude-3-5-haiku", 0.80, 4.0),
    ("claude-3-opus", 15.0, 75.0),
    ("claude-3-haiku", 0.25, 1.25),
    ("opus", 15.0, 75.0),
    ("sonnet", 3.0, 15.0),
    ("haiku", 1.0, 5.0),
    ("gpt-4o-mini", 0.15, 0.60),
    ("gpt-4o", 2.50, 10.0),
    ("gpt-4.1-mini", 0.40, 1.60),
    ("gpt-4.1", 2.0, 8.0),
    ("o3-mini", 1.10, 4.40),
    ("o3", 2.0, 8.0),
    ("gpt-4-turbo", 10.0, 30.0),
    ("gpt-4", 30.0, 60.0),
    ("gpt-3.5", 0.50, 1.50),
]

# Model names that represent no paid LLM call (cost = 0).
_FREE_MODEL_MARKERS = ("rule-based", "local-model", "local", "demo", "baseline", "none")

# Fallback pricing when a real-looking model name is not in the table.
_DEFAULT_PRICING: Tuple[float, float] = (1.0, 5.0)
PRICING_SOURCE_VERSION = "pricing-table-2026-08-24"
USD_PER_CREDIT = Decimal("1")
CREDITS_MICRO_PER_CREDIT = 1_000_000


def is_free_model(model: str | None) -> bool:
    """True when ``model`` names no real paid LLM: a sentinel / rule-based /
    local marker (e.g. ``'local-model'``, ``'rule-based'``) or nothing at all.

    Callers use this to treat such values as "no explicit model" rather than a
    real model id — e.g. the Discord bot must not forward the default
    ``'local-model'`` sentinel to the hosted-model API as if it were a model."""
    name = (model or "").strip().lower()
    if not name:
        return True
    return any(marker in name for marker in _FREE_MODEL_MARKERS)


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


def price_for_model(model: str | None) -> Tuple[float, float]:
    """Return (input_usd_per_mtok, output_usd_per_mtok) for a model name."""
    name = (model or "").strip().lower()
    if not name:
        return _DEFAULT_PRICING
    if any(marker in name for marker in _FREE_MODEL_MARKERS):
        return (0.0, 0.0)
    for needle, in_price, out_price in _PRICING_TABLE:
        if needle in name:
            return (in_price, out_price)
    return _DEFAULT_PRICING


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
    return BillingEvidence(
        billing_source=billing_mode,
        usage_authority=authority,
        provider_cost_usd=provider_cost_usd,
        estimated_cost_usd=estimated,
        pricing_snapshot=pricing_snapshot,
        debited_credits_micro=(
            credits_micro_for_usd(provider_cost_usd if provider_cost_usd is not None else estimated)
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
