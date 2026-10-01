"""Every dashboard-launchable model carries a pinned sampling policy.

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
    CatalogModel,
    ExecutionModelRoute,
    SamplingPolicy,
    list_execution_model_routes,
)
from dashboard.backend.domain.model_providers.models import ProviderRecord

_EXPECTED = {
    "anthropic/claude-haiku-4-5": PINNED_TEMPERATURE,
    "anthropic/claude-sonnet-4-6": PINNED_TEMPERATURE,
    "openai/gpt-5.5": PINNED_REASONING_LOW,
    "google/gemini-3.1-pro-preview": PINNED_TEMPERATURE,
    "deepseek/deepseek-v4-pro": PINNED_NO_THINKING,
    "qwen/qwen3.7-plus": PINNED_NO_THINKING,
}


def test_the_table_covers_the_catalog_exactly():
    assert {m.catalog_id for m in ATL_EXECUTION_MODELS} == set(_EXPECTED)


@pytest.mark.parametrize("catalog_id", sorted(_EXPECTED))
def test_each_model_pins_the_documented_policy(catalog_id):
    model = next(m for m in ATL_EXECUTION_MODELS if m.catalog_id == catalog_id)
    assert model.sampling == _EXPECTED[catalog_id]
    assert model.sampling.temperature is not None or model.sampling.reasoning_effort


def test_the_three_policies_are_what_they_say():
    assert PINNED_TEMPERATURE == SamplingPolicy(temperature=0.0, reasoning_effort=None)
    assert PINNED_REASONING_LOW == SamplingPolicy(temperature=None, reasoning_effort="low")
    assert PINNED_NO_THINKING == SamplingPolicy(temperature=0.0, reasoning_effort="none")


def test_routes_carry_their_models_policy():
    provider = ProviderRecord(
        provider_id="openrouter",
        display_name="OpenRouter",
        adapter_type="openrouter",
        approved_base_url="https://openrouter.ai/api/v1",
    )
    by_id = {r.catalog_id: r for r in list_execution_model_routes(provider)}
    assert by_id["openai/gpt-5.5"].sampling == PINNED_REASONING_LOW
    assert by_id["deepseek/deepseek-v4-pro"].sampling == PINNED_NO_THINKING
    assert by_id["anthropic/claude-sonnet-4-6"].sampling == PINNED_TEMPERATURE


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


def test_a_hand_built_route_claims_no_policy():
    """`ExecutionModelRoute.sampling` defaults to None, not to a policy.

    Only `list_execution_model_routes` builds a real route, and it always
    fills this from the model. The constructions that omit it are the six test
    stubs standing in for a preflight, and for those `None` is the truthful
    answer: nothing was resolved. A `PINNED_TEMPERATURE` default would have
    each of them claim a pinning nobody asked for, and the endpoint would then
    hand the child `--llm-temperature 0` for a route it never looked up.
    """
    route = ExecutionModelRoute(
        catalog_id="openai/gpt-5.5",
        label="GPT-5.5",
        provider_model_id="gpt-5.5",
    )
    assert route.sampling is None


def test_every_route_the_catalog_builds_carries_a_policy():
    """That None is reachable by hand only; the real builder never emits it."""
    provider = ProviderRecord(
        provider_id="openrouter",
        display_name="OpenRouter",
        adapter_type="openrouter",
        approved_base_url="https://openrouter.ai/api/v1",
    )
    routes = list_execution_model_routes(provider)
    assert routes
    assert all(route.sampling is not None for route in routes)
