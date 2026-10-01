"""Every dashboard-launchable model states its sampling policy.

The pipeline request builder sent model, max_tokens, system and messages and
nothing else, so two runs of one configuration were two draws from the
provider's default sampler, 49 bars deep, each prompt embedding the previous
bar's draw. The policy is per *model*, not per provider: the provider-level
`reasoning` capability cannot say whether this model rejects temperature
(OpenAI reasoning models do) or ignores it while thinking (DeepSeek does).
"""
import pytest

from dashboard.backend.domain.model_providers.execution_catalog import (
    ATL_EXECUTION_MODELS,
    PINNED_NO_THINKING,
    PINNED_REASONING_LOW,
    PINNED_TEMPERATURE,
    PROVIDER_DEFAULT,
    CatalogModel,
    SamplingPolicy,
    sampling_policy_for,
)

_EXPECTED = {
    "anthropic/claude-haiku-4-5": PINNED_TEMPERATURE,
    "anthropic/claude-sonnet-4-6": PINNED_TEMPERATURE,
    "openai/gpt-5.5": PINNED_REASONING_LOW,
    # Google advises against low temperatures on Gemini 3 (looping, worse
    # output); left at the provider default until a temperature-0 run is
    # measured.
    "google/gemini-3.1-pro-preview": PROVIDER_DEFAULT,
    "deepseek/deepseek-v4-pro": PINNED_NO_THINKING,
    "qwen/qwen3.7-plus": PINNED_NO_THINKING,
}


def test_the_table_covers_the_catalog_exactly():
    assert {m.catalog_id for m in ATL_EXECUTION_MODELS} == set(_EXPECTED)


@pytest.mark.parametrize("catalog_id", sorted(_EXPECTED))
def test_each_model_pins_the_documented_policy(catalog_id):
    model = next(m for m in ATL_EXECUTION_MODELS if m.catalog_id == catalog_id)
    assert model.sampling == _EXPECTED[catalog_id]
    assert sampling_policy_for(catalog_id) == _EXPECTED[catalog_id]


def test_the_three_policies_are_what_they_say():
    assert PINNED_TEMPERATURE == SamplingPolicy(temperature=0.0, reasoning_effort=None)
    assert PINNED_REASONING_LOW == SamplingPolicy(temperature=None, reasoning_effort="low")
    assert PINNED_NO_THINKING == SamplingPolicy(temperature=0.0, reasoning_effort="none")
    assert PROVIDER_DEFAULT == SamplingPolicy(temperature=None, reasoning_effort=None)
    assert [p.pinned for p in (PINNED_TEMPERATURE, PINNED_REASONING_LOW,
                               PINNED_NO_THINKING, PROVIDER_DEFAULT)] == [
        True, True, True, False
    ]


def test_a_model_outside_the_catalog_has_no_policy():
    """None, not PROVIDER_DEFAULT: nothing was resolved, so nothing is claimed."""
    assert sampling_policy_for("openai/o5") is None
    assert sampling_policy_for("") is None
    assert sampling_policy_for(" deepseek/deepseek-v4-pro ") == PINNED_NO_THINKING


def test_a_catalog_row_cannot_forget_to_state_its_policy():
    """`sampling` has no default, and the missing default *is* the guard.

    Every row in the catalog was written `CatalogModel("vendor/id", "Label",
    "vendor")` before this change -- three positional arguments and nothing
    else -- so that is the form the next one gets copied into. With a
    `PINNED_TEMPERATURE` default, the next OpenAI reasoning
    model added that way silently pins `temperature=0` on a route
    that returns 400 for any non-default temperature -- every backtest on it
    failing at the provider, with nothing local to see and a metadata row
    confidently reporting `policy: pinned_v1`. The table test above does not
    catch it either: whoever adds the row also adds it to `_EXPECTED`, and if
    they copy the default they wrote by accident, the assertion agrees with
    them. Only the construction site can ask the question.
    """
    with pytest.raises(TypeError):
        CatalogModel("openai/o5", "o5", "openai")
