"""The cost properties of a model_usage_recorded event (analytics.usage_cost)."""

import pytest

from dashboard.backend.domain.analytics.usage_cost import (
    byok_estimate_micro,
    byok_rollup_lane,
    model_usage_properties,
)
from dashboard.backend.infrastructure.llm.execution.models import (
    BillingMode,
    LLMUsage,
    PricingSnapshot,
)
from dashboard.backend.infrastructure.llm.token_cost import build_cost_evidence


def _platform(provider_cost_usd, *, usage_available=True, estimated_cost_usd=None):
    return model_usage_properties(
        billing_mode="platform_credits",
        model_id="openai/gpt-5.5",
        input_tokens=10,
        output_tokens=5,
        usage_available=usage_available,
        provider_cost_usd=provider_cost_usd,
        estimated_cost_usd=estimated_cost_usd,
    )["cost_micro_usd"]


@pytest.mark.parametrize("provider_cost_usd", [0.0000125, 0.0000005, 0.0000015, 1.2345675])
def test_platform_cost_is_the_micro_figure_the_ledger_books(provider_cost_usd):
    """Float round() is round-half-even and the ledger rounds half-up in
    Decimal, so the two disagreed by a micro-Credit on every exact half."""
    snapshot = PricingSnapshot.from_model("openai/gpt-5.5", "openrouter")
    evidence = build_cost_evidence(
        billing_mode=BillingMode.PLATFORM_CREDITS,
        provider_id="openrouter",
        model_id="openai/gpt-5.5",
        usage=LLMUsage(input_tokens=10, output_tokens=5, usage_available=True),
        provider_cost_usd=provider_cost_usd,
        pricing_snapshot=snapshot,
    )
    assert _platform(provider_cost_usd) == evidence.provider_cost_credits_micro


def test_platform_cost_rounds_an_exact_half_up():
    # 12.5 micro-Credits: round() gave 12, the ledger books 13.
    assert _platform(0.0000125) == 13


def test_platform_cost_is_zero_when_no_usage_was_reported():
    """The ledger debits nothing without reported usage, whatever cost the
    provider sent; the event must not record a debit that never happened."""
    assert _platform(0.5, usage_available=False) == 0
    assert _platform(None, estimated_cost_usd=0.25) == 250_000


def test_legacy_byok_estimate_is_repriced_at_the_listed_rate():
    """A PR #572-era event carried its estimate in cost_micro_usd, priced off a
    snapshot that falls back to $1/$5 for an unlisted model."""
    legacy = {"input_tokens": 1_000, "output_tokens": 100, "cost_micro_usd": 1_500}
    assert byok_estimate_micro(legacy, "openai/gpt-5.5") == 8_000
    assert byok_estimate_micro(legacy, "openai/gpt-4.1-nano") is None
    assert byok_estimate_micro({**legacy, "cost_micro_usd": 0}, "openai/gpt-5.5") is None
    # The split property always wins over the legacy field.
    assert byok_estimate_micro({**legacy, "estimated_cost_micro_usd": 7}, "x") == 7


def test_rollup_lane_reads_a_legacy_rollup_per_model():
    counted = byok_rollup_lane("x", calls=4, estimate_micro=300, unpriced_calls=1)
    assert counted == (300, 1)
    # Before #572: no estimate, every call unpriced.
    assert byok_rollup_lane("openai/gpt-5.5", calls=5, estimate_micro=0, unpriced_calls=None) == (0, 5)
    # #572-era: the estimate stands for a listed model only.
    assert byok_rollup_lane("openai/gpt-5.5", calls=3, estimate_micro=90, unpriced_calls=None) == (90, 0)
    assert byok_rollup_lane("openai/gpt-4.1-nano", calls=2, estimate_micro=40, unpriced_calls=None) == (0, 2)


def test_cost_micro_usd_is_read_through_atl_cost_micro_only():
    """PR #572-era BYOK events carry an estimate in cost_micro_usd, so a reader
    summing the property without the lane check adds it into ATL spend. Every
    reader goes through atl_cost_micro. Allowed besides: the validator, and
    value_queries' defensive reader, which only type-checks the value before
    handing it over."""
    from pathlib import Path

    analytics = Path(__file__).resolve().parents[3] / "domain" / "analytics"
    backend = analytics.parents[1]
    allowed = {analytics / name for name in ("usage_cost.py", "models.py", "value_queries.py")}
    offenders = [
        f"{path.relative_to(backend)}:{number}"
        for path in backend.rglob("*.py")
        if "tests" not in path.parts and path not in allowed
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if 'get("cost_micro_usd"' in line or 'properties["cost_micro_usd"]' in line
    ]
    assert offenders == []
