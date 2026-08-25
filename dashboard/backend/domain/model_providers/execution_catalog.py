"""Canonical ATL models and provider-specific request routes."""

from __future__ import annotations

from dataclasses import dataclass

from .models import ProviderRecord


class UnsupportedExecutionModel(ValueError):
    """The requested ATL model is not approved for this provider."""


@dataclass(frozen=True)
class CatalogModel:
    catalog_id: str
    label: str
    vendor: str


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
    ),
    CatalogModel(
        "anthropic/claude-sonnet-4-6",
        "Claude Sonnet 4.6",
        "anthropic",
    ),
    CatalogModel("openai/gpt-5.5", "GPT-5.5", "openai"),
    CatalogModel(
        "google/gemini-3.1-pro-preview",
        "Gemini 3.1 Pro Preview",
        "google",
    ),
    CatalogModel(
        "deepseek/deepseek-v4-pro",
        "DeepSeek V4 Pro",
        "deepseek",
    ),
    CatalogModel("qwen/qwen3.7-plus", "Qwen3.7 Plus", "qwen"),
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
    "UnsupportedExecutionModel",
    "list_execution_model_routes",
    "resolve_execution_model_route",
]
