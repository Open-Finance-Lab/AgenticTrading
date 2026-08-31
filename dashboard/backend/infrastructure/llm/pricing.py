"""Model price table and the free-model sentinels — a leaf module.

Both ``token_cost`` (estimation and billing evidence) and
``execution/models.py`` (``PricingSnapshot.from_model``) need the table, and
``token_cost`` also needs the pydantic models from ``execution/models`` — so
the table lives here, below both, rather than in either. Keep this module free
of ``dashboard`` imports: the cycle it replaced (CodeQL ``py/cyclic-import``)
only held together because one side imported lazily inside a method.
"""

from __future__ import annotations

from typing import Tuple

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

# Stamped on every ``PricingSnapshot`` so a stored cost can be traced to the
# table that priced it. Bump it whenever the rates above change.
PRICING_SOURCE_VERSION = "pricing-table-2026-08-24"


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
