"""Model price table and the free-model sentinels — a leaf module.

Both ``token_cost`` (estimation and billing evidence) and
``execution/models.py`` (``PricingSnapshot.from_model``) need the table, and
``token_cost`` also needs the pydantic models from ``execution/models`` — so
the table lives here, below both, rather than in either. Keep this module free
of ``dashboard`` imports: the cycle it replaced (CodeQL ``py/cyclic-import``)
only held together because one side imported lazily inside a method.
"""

from __future__ import annotations

import re
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
    ("anthropic/claude-haiku-4-5", 1.0, 5.0),
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


def _table_price(name: str) -> Tuple[float, float] | None:
    if any(marker in name for marker in _FREE_MODEL_MARKERS):
        return (0.0, 0.0)
    for needle, in_price, out_price in _PRICING_TABLE:
        if needle in name:
            return (in_price, out_price)
    return None


def price_for_model(model: str | None) -> Tuple[float, float]:
    """Return (input_usd_per_mtok, output_usd_per_mtok) for a model name."""
    name = (model or "").strip().lower()
    return (_table_price(name) if name else None) or _DEFAULT_PRICING


# A dated or preview snapshot of a model is that model: Anthropic's
# ``claude-3-5-haiku-20241022``, OpenAI's ``gpt-4o-2024-08-06``, the catalog's
# ``gemini-3.1-pro-preview``. Any other suffix names a different model.
_SNAPSHOT_SUFFIX = re.compile(r"-(?:\d{8}|\d{4}-\d{2}-\d{2}|preview)$")


def _listed_name(name: str) -> str:
    """``name`` without its provider prefix or a snapshot suffix."""
    return _SNAPSHOT_SUFFIX.sub("", name.rsplit("/", 1)[-1])


def _build_listed_prices() -> dict[str, Tuple[float, float]]:
    listed: dict[str, Tuple[float, float]] = {}
    for needle, in_price, out_price in _PRICING_TABLE:
        # First entry wins, as it does for the substring match.
        listed.setdefault(_listed_name(needle), (in_price, out_price))
    return listed


_LISTED_PRICES = _build_listed_prices()


def listed_price_for_model(model: str | None) -> Tuple[float, float] | None:
    """The table's price for ``model``, or None when the table does not list it.

    ``price_for_model`` never answers None: an unlisted name falls back to
    ``_DEFAULT_PRICING`` so a reservation always has a ceiling. That fallback is
    a guess, which is fine for bounding a hold and wrong for a figure published
    as what a call would have cost (the admin BYOK estimate), where an unlisted
    model must read as unpriced rather than as $1/$5.

    The substring match is a guess of the same kind: ``gpt-4.1-nano`` would
    borrow ``gpt-4.1``'s rate (~20x its own) and ``o3-pro`` ``o3``'s (~10x
    below it). So a model is listed here only when its name, less the
    provider prefix and a snapshot suffix, *is* a table entry's. OpenRouter's
    ``:free`` variants are listed at zero, not at their paid sibling's rate,
    and so are the free-model sentinels -- by exact name, since the substring
    markers would price any model with ``none`` or ``demo`` in it at zero.
    """
    name = (model or "").strip().lower()
    if not name:
        return None
    if name.endswith(":free") or name in _FREE_MODEL_MARKERS:
        return (0.0, 0.0)
    return _LISTED_PRICES.get(_listed_name(name))
