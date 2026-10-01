"""Canonical ATL models and provider-specific request routes."""

from __future__ import annotations

from dataclasses import dataclass

from .models import ProviderRecord


class UnsupportedExecutionModel(ValueError):
    """The requested ATL model is not approved for this provider."""


@dataclass(frozen=True)
class SamplingPolicy:
    """What every backtest call asks the model to sample with.

    Per model, not per provider: the provider-level ``reasoning`` capability
    cannot say whether *this* model rejects a temperature (OpenAI reasoning
    models return 400) or ignores it while thinking (DeepSeek). A value of
    ``None`` is *not sent*, which keeps the request byte-identical to what
    every run before this sent for that field.

    Applied by ``AnthropicCompatibleExecutionClient``, which resolves it from
    the signed handoff's ``model_id`` (``sampling_policy_for``) and imposes it
    on every call the backtest child makes. Nothing upstream of that client
    carries it, so no call site can forget to.
    """

    temperature: float | None
    reasoning_effort: str | None

    @property
    def pinned(self) -> bool:
        """True when the policy sends anything at all."""
        return self.temperature is not None or bool(self.reasoning_effort)


PINNED_TEMPERATURE = SamplingPolicy(temperature=0.0, reasoning_effort=None)
PINNED_REASONING_LOW = SamplingPolicy(temperature=None, reasoning_effort="low")
# "none" switches thinking off rather than naming an effort level. The OpenAI
# adapter sends it to an openai_compatible provider (CommonStack) as
# `thinking: {type: "disabled"}`, the one reasoning control that lane honours.
PINNED_NO_THINKING = SamplingPolicy(temperature=0.0, reasoning_effort="none")
# Sends nothing: the provider's own defaults, exactly as before any policy.
# A stated choice, not an absence -- a row still has to name it.
PROVIDER_DEFAULT = SamplingPolicy(temperature=None, reasoning_effort=None)


@dataclass(frozen=True)
class CatalogModel:
    catalog_id: str
    label: str
    vendor: str
    # No default, deliberately. Every row below used to be three positional
    # arguments and nothing else, which is the form the next one gets copied
    # into; a default here would pin temperature 0 on the next OpenAI
    # reasoning model added that way -- a model that returns 400 for any
    # non-default temperature. The spec's rule is that a model not in this
    # catalog cannot be launched at all, "so there is no default row";
    # leaving the field required is that rule with a compiler behind it.
    sampling: SamplingPolicy


@dataclass(frozen=True)
class ExecutionModelRoute:
    catalog_id: str
    label: str
    provider_model_id: str


ATL_EXECUTION_MODELS = (
    CatalogModel(
        "anthropic/claude-haiku-4-5",
        "Claude Haiku 4.5",
        "anthropic",
        PINNED_TEMPERATURE,
    ),
    CatalogModel(
        "anthropic/claude-sonnet-4-6",
        "Claude Sonnet 4.6",
        "anthropic",
        PINNED_TEMPERATURE,
    ),
    # OpenAI reasoning models reject a non-default temperature outright.
    CatalogModel("openai/gpt-5.5", "GPT-5.5", "openai", PINNED_REASONING_LOW),
    # Provider default, not temperature 0: Google's Gemini 3 guidance is to
    # keep the default 1.0 and warns that lower values can loop or degrade
    # the answer. A loop fills the output ceiling and buys a 4096-token
    # recovery retry, so pinning 0 unmeasured risks doubling the billed tokens
    # per bar. Revisit once a temperature-0 Gemini run has been measured.
    CatalogModel(
        "google/gemini-3.1-pro-preview",
        "Gemini 3.1 Pro Preview",
        "google",
        PROVIDER_DEFAULT,
    ),
    # Thinking models served on CommonStack, which honours no graduated
    # reasoning control for these two: reasoning.effort, a top-level
    # reasoning_effort, reasoning.enabled=false and thinking.budget_tokens
    # were all ignored in the 2026-10-01 probe (#539). Left alone, each call
    # either thinks to the 2000-token ceiling or does not think at all.
    # Thinking off is the one control it honours, and with thinking off the
    # temperature applies as well.
    CatalogModel(
        "deepseek/deepseek-v4-pro",
        "DeepSeek V4 Pro",
        "deepseek",
        PINNED_NO_THINKING,
    ),
    CatalogModel(
        "qwen/qwen3.7-plus",
        "Qwen3.7 Plus",
        "qwen",
        PINNED_NO_THINKING,
    ),
)

_NATIVE_VENDOR = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "google",
}


def _provider_model_id(
    provider: ProviderRecord,
    model: CatalogModel,
) -> str | None:
    if provider.adapter_type == "openrouter":
        return model.catalog_id
    native_vendor = _NATIVE_VENDOR.get(provider.adapter_type)
    if native_vendor:
        if model.vendor != native_vendor:
            return None
        return model.catalog_id.split("/", 1)[1]
    if provider.adapter_type == "openai_compatible":
        return (
            model.catalog_id
            if model.catalog_id in provider.capabilities.model_allowlist
            else None
        )
    return None


def list_execution_model_routes(
    provider: ProviderRecord,
) -> tuple[ExecutionModelRoute, ...]:
    """Return ATL models that this registered provider can execute."""

    routes: list[ExecutionModelRoute] = []
    for model in ATL_EXECUTION_MODELS:
        provider_model_id = _provider_model_id(provider, model)
        if provider_model_id:
            routes.append(
                ExecutionModelRoute(
                    catalog_id=model.catalog_id,
                    label=model.label,
                    provider_model_id=provider_model_id,
                )
            )
    return tuple(routes)


def sampling_policy_for(catalog_id: str) -> SamplingPolicy | None:
    """The catalog model's policy, or None for an id the catalog lacks.

    None, not ``PROVIDER_DEFAULT``: a model outside the catalog cannot be
    launched by the dashboard at all, so reaching here with one is a caller
    that resolved nothing, and the honest answer is "no policy" rather than a
    policy nobody chose.
    """

    requested = str(catalog_id or "").strip()
    for model in ATL_EXECUTION_MODELS:
        if model.catalog_id == requested:
            return model.sampling
    return None


def resolve_execution_model_route(
    provider: ProviderRecord,
    catalog_id: str,
) -> ExecutionModelRoute:
    """Resolve one approved ATL model to the provider's request model id."""

    requested = str(catalog_id or "").strip()
    for route in list_execution_model_routes(provider):
        if route.catalog_id == requested:
            return route
    raise UnsupportedExecutionModel(
        "model is not available from this provider"
    )


__all__ = [
    "ATL_EXECUTION_MODELS",
    "CatalogModel",
    "ExecutionModelRoute",
    "PINNED_NO_THINKING",
    "PINNED_REASONING_LOW",
    "PINNED_TEMPERATURE",
    "PROVIDER_DEFAULT",
    "SamplingPolicy",
    "UnsupportedExecutionModel",
    "list_execution_model_routes",
    "resolve_execution_model_route",
    "sampling_policy_for",
]
