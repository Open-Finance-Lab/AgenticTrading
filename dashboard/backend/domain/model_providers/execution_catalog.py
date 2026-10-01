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
    """

    temperature: float | None
    reasoning_effort: str | None


PINNED_TEMPERATURE = SamplingPolicy(temperature=0.0, reasoning_effort=None)
PINNED_REASONING_LOW = SamplingPolicy(temperature=None, reasoning_effort="low")
# "none" switches thinking off rather than naming an effort level. The OpenAI
# adapter sends it to an openai_compatible provider (CommonStack) as
# `thinking: {type: "disabled"}`, the one reasoning control that lane honours.
PINNED_NO_THINKING = SamplingPolicy(temperature=0.0, reasoning_effort="none")


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
    # None, not a policy. Only `list_execution_model_routes` builds a real
    # route and it always fills this from the model; the constructions that
    # omit it are test stubs standing in for a preflight, and for those the
    # honest answer is "nothing was resolved". A pinned default would have a
    # hand-built route claim a policy nobody chose, and the endpoint would
    # then send `--llm-temperature 0` for a route it never looked up.
    sampling: SamplingPolicy | None = None


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
    CatalogModel(
        "google/gemini-3.1-pro-preview",
        "Gemini 3.1 Pro Preview",
        "google",
        PINNED_TEMPERATURE,
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
                    sampling=model.sampling,
                )
            )
    return tuple(routes)


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
    "SamplingPolicy",
    "UnsupportedExecutionModel",
    "list_execution_model_routes",
    "resolve_execution_model_route",
]
