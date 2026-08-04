"""Tests for parallel LLM gateway providers (CommonStack / OpenRouter / Anthropic)."""

from __future__ import annotations

import pytest

from dashboard.backend.infrastructure.llm import providers as providers_pkg
from dashboard.backend.infrastructure.llm.providers import (
    KNOWN_INTEGRATIONS,
    anthropic_native,
    commonstack,
    default_model_name,
    make_llm_client,
    openrouter,
    resolve_integration,
)


def test_known_integrations_are_parallel_siblings():
    assert set(KNOWN_INTEGRATIONS) == {"commonstack", "openrouter", "anthropic"}
    assert providers_pkg.PROVIDERS["commonstack"] is commonstack
    assert providers_pkg.PROVIDERS["openrouter"] is openrouter
    assert providers_pkg.PROVIDERS["anthropic"] is anthropic_native


def test_resolve_integration_explicit(monkeypatch):
    monkeypatch.delenv("COMMONSTACK_API_KEY", raising=False)
    assert resolve_integration("openrouter") == "openrouter"
    assert resolve_integration("CommonStack") == "commonstack"
    assert resolve_integration("ANTHROPIC") == "anthropic"


def test_resolve_integration_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown LLM integration"):
        resolve_integration("together")


def test_resolve_integration_auto_prefers_commonstack(monkeypatch):
    monkeypatch.setenv("COMMONSTACK_API_KEY", "cs-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")  # must NOT auto-pick OpenRouter
    assert resolve_integration(None) == "commonstack"
    assert resolve_integration("") == "commonstack"


def test_resolve_integration_auto_falls_back_to_anthropic(monkeypatch):
    monkeypatch.delenv("COMMONSTACK_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")  # still opt-in only
    assert resolve_integration(None) == "anthropic"


def test_default_model_name_per_integration(monkeypatch):
    monkeypatch.delenv("COMMONSTACK_API_KEY", raising=False)
    assert default_model_name("anthropic") == anthropic_native.DEFAULT_MODEL
    assert default_model_name("commonstack") == commonstack.DEFAULT_MODEL
    assert default_model_name("openrouter") == openrouter.DEFAULT_MODEL
    # CommonStack Anthropic provider stubbed greeting; DeepSeek is the hosted default.
    assert commonstack.DEFAULT_MODEL == "deepseek/deepseek-v4-pro"


def test_make_llm_client_openrouter_uses_openrouter_key(monkeypatch):
    if not providers_pkg.HAS_ANTHROPIC:
        pytest.skip("anthropic SDK not installed")

    captured = {}

    class _FakeAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.messages = object()

    monkeypatch.setattr(providers_pkg, "_Anthropic", _FakeAnthropic)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("COMMONSTACK_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://example.com")
    monkeypatch.setenv("OPENROUTER_APP_TITLE", "ATL Test")

    client = make_llm_client("openrouter")
    assert client is not None
    assert isinstance(client, openrouter.OpenRouterClient)
    assert captured["api_key"] == "sk-or-test"
    assert captured["base_url"] == openrouter.base_url()
    assert captured["default_headers"]["HTTP-Referer"] == "https://example.com"
    assert captured["default_headers"]["X-Title"] == "ATL Test"


def test_make_llm_client_passes_reasoning_only_to_openrouter(monkeypatch):
    captured = {}

    def _openrouter_client(anthropic_cls, *, reasoning_effort=None):
        captured["openrouter"] = (anthropic_cls, reasoning_effort)
        return "openrouter-client"

    def _commonstack_client(anthropic_cls):
        captured["commonstack"] = anthropic_cls
        return "commonstack-client"

    def _anthropic_client(anthropic_cls):
        captured["anthropic"] = anthropic_cls
        return "anthropic-client"

    monkeypatch.setattr(openrouter, "make_client", _openrouter_client)
    monkeypatch.setattr(commonstack, "make_client", _commonstack_client)
    monkeypatch.setattr(anthropic_native, "make_client", _anthropic_client)

    assert make_llm_client("openrouter", reasoning_effort="none") == "openrouter-client"
    assert captured["openrouter"] == (providers_pkg._Anthropic, "none")

    assert (
        make_llm_client("commonstack", reasoning_effort="none") == "commonstack-client"
    )
    assert captured["commonstack"] is providers_pkg._Anthropic

    assert make_llm_client("anthropic", reasoning_effort="none") == "anthropic-client"
    assert captured["anthropic"] is providers_pkg._Anthropic


def test_openrouter_messages_enable_medium_reasoning_by_default(monkeypatch):
    """Default maps medium → reasoning.max_tokens=2048 so JSON still fits."""
    monkeypatch.delenv("OPENROUTER_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("OPENROUTER_REASONING_MAX_TOKENS", raising=False)
    recorded = {}

    class _Inner:
        def create(self, **kwargs):
            recorded.update(kwargs)
            return "ok"

    proxy = openrouter._OpenRouterMessages(_Inner())
    assert proxy.create(model="nvidia/nemotron-3-nano-30b-a3b", max_tokens=8000) == "ok"
    assert recorded["extra_body"]["reasoning"] == {"max_tokens": 2048, "enabled": True}
    assert recorded["thinking"] == {"type": "enabled", "budget_tokens": 2048}


def test_openrouter_instance_reasoning_overrides_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "medium")
    recorded = {}

    class _Inner:
        def create(self, **kwargs):
            recorded.update(kwargs)
            return "ok"

    proxy = openrouter._OpenRouterMessages(_Inner(), reasoning_effort="none")
    assert proxy.create(model="nvidia/nemotron-3-nano-30b-a3b") == "ok"
    assert recorded["extra_body"]["reasoning"] == {
        "effort": "none",
        "enabled": False,
        "exclude": True,
    }
    assert recorded["thinking"] == {"type": "disabled"}


def test_openrouter_instances_keep_reasoning_overrides_isolated(monkeypatch):
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "high")
    disabled_recorded = {}
    medium_recorded = {}

    class _Inner:
        def __init__(self, recorder):
            self.recorder = recorder

        def create(self, **kwargs):
            self.recorder.update(kwargs)
            return "ok"

    disabled = openrouter._OpenRouterMessages(
        _Inner(disabled_recorded), reasoning_effort="none"
    )
    medium = openrouter._OpenRouterMessages(
        _Inner(medium_recorded), reasoning_effort="medium"
    )

    assert disabled.create(model="nemotron") == "ok"
    assert medium.create(model="nemotron") == "ok"

    assert disabled_recorded["thinking"] == {"type": "disabled"}
    assert medium_recorded["thinking"] == {
        "type": "enabled",
        "budget_tokens": 2048,
    }


def test_openrouter_messages_inject_reasoning_none_when_disabled(monkeypatch):
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "none")
    recorded = {}

    class _Inner:
        def create(self, **kwargs):
            recorded.update(kwargs)
            return "ok"

    proxy = openrouter._OpenRouterMessages(_Inner())
    assert proxy.create(model="nvidia/nemotron-3-nano-30b-a3b", max_tokens=2000) == "ok"
    assert recorded["extra_body"]["reasoning"]["effort"] == "none"
    assert recorded["extra_body"]["reasoning"]["enabled"] is False
    assert recorded["extra_body"]["reasoning"]["exclude"] is True
    assert recorded["thinking"] == {"type": "disabled"}


def test_openrouter_messages_respect_caller_reasoning(monkeypatch):
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "none")
    recorded = {}

    class _Inner:
        def create(self, **kwargs):
            recorded.update(kwargs)
            return "ok"

    proxy = openrouter._OpenRouterMessages(_Inner())
    proxy.create(
        extra_body={"reasoning": {"effort": "high"}}, thinking={"type": "enabled"}
    )
    assert recorded["extra_body"] == {"reasoning": {"effort": "high"}}
    assert recorded["thinking"] == {"type": "enabled"}


def test_openrouter_reasoning_effort_auto_skips_injection(monkeypatch):
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "auto")
    assert openrouter.reasoning_extra_body() is None
    assert openrouter.anthropic_thinking_kwarg() is None


def test_openrouter_reasoning_effort_medium_maps_to_max_tokens(monkeypatch):
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "medium")
    monkeypatch.delenv("OPENROUTER_REASONING_MAX_TOKENS", raising=False)
    body = openrouter.reasoning_extra_body()
    assert body == {"reasoning": {"max_tokens": 2048, "enabled": True}}
    assert openrouter.anthropic_thinking_kwarg() == {
        "type": "enabled",
        "budget_tokens": 2048,
    }


def test_openrouter_reasoning_max_tokens_env_overrides(monkeypatch):
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "medium")
    monkeypatch.setenv("OPENROUTER_REASONING_MAX_TOKENS", "1500")
    body = openrouter.reasoning_extra_body()
    assert body == {"reasoning": {"max_tokens": 1500, "enabled": True}}
    assert openrouter.anthropic_thinking_kwarg() == {
        "type": "enabled",
        "budget_tokens": 1500,
    }


def test_make_llm_client_commonstack_ignores_openrouter_key(monkeypatch):
    if not providers_pkg.HAS_ANTHROPIC:
        pytest.skip("anthropic SDK not installed")

    captured = {}

    class _FakeAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(providers_pkg, "_Anthropic", _FakeAnthropic)
    monkeypatch.setenv("COMMONSTACK_API_KEY", "cs-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    client = make_llm_client("commonstack")
    assert client is not None
    assert captured["api_key"] == "cs-key"
    assert captured["base_url"] == commonstack.base_url()


def test_make_llm_client_explicit_openrouter_missing_key_returns_none(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("COMMONSTACK_API_KEY", "cs-key")  # must not leak across
    assert make_llm_client("openrouter") is None


def test_ensure_llm_client_available_rejects_missing_sdk(monkeypatch):
    monkeypatch.setattr(providers_pkg, "HAS_ANTHROPIC", False)
    monkeypatch.setattr(providers_pkg, "_Anthropic", None)

    with pytest.raises(
        providers_pkg.LLMProviderConfigurationError,
        match="SDK",
    ):
        providers_pkg.ensure_llm_client_available()


def test_ensure_llm_client_available_rejects_missing_key_without_leaking_it(
    monkeypatch,
):
    secret = "secret-that-must-not-appear"
    monkeypatch.setattr(providers_pkg, "HAS_ANTHROPIC", True)
    monkeypatch.setattr(providers_pkg, "_Anthropic", object())
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    monkeypatch.setattr(providers_pkg, "make_llm_client", lambda _integration=None: None)

    with pytest.raises(providers_pkg.LLMProviderConfigurationError) as exc_info:
        providers_pkg.ensure_llm_client_available("anthropic")

    assert "anthropic" in str(exc_info.value)
    assert secret not in str(exc_info.value)


def test_ensure_llm_client_available_returns_constructed_client(monkeypatch):
    client = object()
    monkeypatch.setattr(providers_pkg, "HAS_ANTHROPIC", True)
    monkeypatch.setattr(providers_pkg, "_Anthropic", object())
    monkeypatch.setattr(providers_pkg, "make_llm_client", lambda _integration=None: client)

    assert providers_pkg.ensure_llm_client_available("commonstack") is client


def test_harness_reexports_provider_factory(monkeypatch):
    from dashboard.backend.infrastructure.llm import backtest_harness as bh

    monkeypatch.delenv("COMMONSTACK_API_KEY", raising=False)
    assert bh.default_model_name() == bh.LLM_MODEL_NAME
    assert bh.default_model_name("openrouter") == bh.OPENROUTER_MODEL_NAME
    assert bh.COMMONSTACK_MODEL_NAME == commonstack.DEFAULT_MODEL
