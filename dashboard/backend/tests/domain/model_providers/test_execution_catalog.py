"""Provider compatibility contracts for the canonical ATL model catalog."""

from __future__ import annotations

import pytest

from dashboard.backend.domain.model_providers.execution_catalog import (
    UnsupportedExecutionModel,
    list_execution_model_routes,
    resolve_execution_model_route,
)
from dashboard.backend.domain.model_providers.models import ProviderRecord
from dashboard.backend.domain.model_providers.repository import ModelProviderStore
from dashboard.backend.domain.model_providers.service import ModelProviderService


def _provider(
    adapter_type: str,
    *,
    allowlist: tuple[str, ...] = (),
) -> ProviderRecord:
    return ProviderRecord(
        provider_id=(
            "approved_compatible"
            if adapter_type == "openai_compatible"
            else adapter_type
        ),
        display_name="Provider",
        adapter_type=adapter_type,
        approved_base_url="https://provider.example/v1",
        capabilities={"model_allowlist": allowlist},
    )


def test_openrouter_can_run_the_full_atl_catalog():
    routes = list_execution_model_routes(_provider("openrouter"))

    assert [route.catalog_id for route in routes] == [
        "anthropic/claude-haiku-4-5",
        "anthropic/claude-sonnet-4-6",
        "openai/gpt-5.5",
        "google/gemini-3.1-pro-preview",
        "deepseek/deepseek-v4-pro",
        "qwen/qwen3.7-plus",
    ]
    assert all(route.provider_model_id == route.catalog_id for route in routes)


@pytest.mark.parametrize(
    ("adapter_type", "catalog_id", "provider_model_id"),
    [
        ("openai", "openai/gpt-5.5", "gpt-5.5"),
        (
            "anthropic",
            "anthropic/claude-haiku-4-5",
            "claude-haiku-4-5",
        ),
        (
            "anthropic",
            "anthropic/claude-sonnet-4-6",
            "claude-sonnet-4-6",
        ),
        (
            "gemini",
            "google/gemini-3.1-pro-preview",
            "gemini-3.1-pro-preview",
        ),
    ],
)
def test_native_provider_routes_strip_only_the_expected_vendor_prefix(
    adapter_type,
    catalog_id,
    provider_model_id,
):
    route = resolve_execution_model_route(_provider(adapter_type), catalog_id)

    assert route.catalog_id == catalog_id
    assert route.provider_model_id == provider_model_id


def test_native_provider_rejects_an_incompatible_model():
    with pytest.raises(UnsupportedExecutionModel):
        resolve_execution_model_route(
            _provider("openai"),
            "anthropic/claude-sonnet-4-6",
        )


def test_custom_provider_requires_an_explicit_allowlist():
    provider = _provider(
        "openai_compatible",
        allowlist=("openai/gpt-5.5",),
    )

    assert [
        route.catalog_id for route in list_execution_model_routes(provider)
    ] == ["openai/gpt-5.5"]
    with pytest.raises(UnsupportedExecutionModel):
        resolve_execution_model_route(provider, "deepseek/deepseek-v4-pro")


def test_custom_provider_allowlist_rejects_invalid_model_ids():
    with pytest.raises(ValueError, match="model_allowlist"):
        _provider("openai_compatible", allowlist=("not allowed?",))


def test_platform_candidates_prefer_openrouter_and_support_commonstack_only(
    tmp_path, monkeypatch
):
    store = ModelProviderStore(tmp_path / "providers.db")
    service = ModelProviderService(store=store)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-openrouter-test-abcd")
    monkeypatch.setenv("COMMONSTACK_API_KEY", "cs-commonstack-test-abcd")

    assert service.resolve_platform_execution_candidates(
        "qwen/qwen3.7-plus"
    ) == ("openrouter", "commonstack")

    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert service.resolve_platform_execution_candidates(
        "qwen/qwen3.7-plus"
    ) == ("commonstack",)
