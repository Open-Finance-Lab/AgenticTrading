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


# --------------------------------------------------------------------------
# The thinking budget and the answer share one output ceiling
#
# ``anthropic_thinking_kwarg``'s whole purpose is that "thinking cannot consume
# the entire max_tokens ceiling" -- but the mapped budget was never checked
# against the ceiling the request actually carries, and the default pairing
# (effort ``medium`` → 2048 against the harness's ``max_tokens=2000``) made
# that guarantee false on *every first request*. A reply that cannot emit text
# is the empty response the retry paths downstream exist to recover from, at a
# billed call each, so this is the root cause under those retries.
# --------------------------------------------------------------------------


def _fresh_warn_state(monkeypatch):
    """The clamp warns once per (budget, ceiling); isolate that across tests."""
    monkeypatch.setattr(openrouter, "_WARNED_BUDGET_CEILINGS", set())


def test_the_budget_is_clamped_to_leave_room_for_an_answer(monkeypatch):
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "medium")
    monkeypatch.delenv("OPENROUTER_REASONING_MAX_TOKENS", raising=False)
    _fresh_warn_state(monkeypatch)

    expected = 2000 - openrouter._MIN_ANSWER_TOKENS
    assert openrouter.anthropic_thinking_kwarg(max_tokens=2000) == {
        "type": "enabled",
        "budget_tokens": expected,
    }
    assert openrouter.reasoning_extra_body(max_tokens=2000) == {
        "reasoning": {"max_tokens": expected, "enabled": True}
    }


def test_a_budget_that_already_fits_is_left_alone(monkeypatch):
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "medium")
    monkeypatch.delenv("OPENROUTER_REASONING_MAX_TOKENS", raising=False)
    _fresh_warn_state(monkeypatch)

    # The recovery ceiling exists precisely so reasoning and the JSON both fit.
    assert openrouter.anthropic_thinking_kwarg(max_tokens=4096) == {
        "type": "enabled",
        "budget_tokens": 2048,
    }


def test_an_unknown_ceiling_is_not_clamped_against(monkeypatch):
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "medium")
    monkeypatch.delenv("OPENROUTER_REASONING_MAX_TOKENS", raising=False)
    _fresh_warn_state(monkeypatch)

    for ceiling in (None, 0, -1, "2000", True):
        assert openrouter.anthropic_thinking_kwarg(max_tokens=ceiling) == {
            "type": "enabled",
            "budget_tokens": 2048,
        }, ceiling


def test_a_ceiling_too_small_for_any_legal_budget_disables_reasoning(monkeypatch):
    """``LLM_MAX_OUTPUT_TOKENS=600`` is a documented small-demo config.

    1024 is OpenRouter's floor for ``reasoning.max_tokens``, so there is no
    smaller budget to fall back to -- only a request that cannot answer.
    """
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "medium")
    monkeypatch.delenv("OPENROUTER_REASONING_MAX_TOKENS", raising=False)
    _fresh_warn_state(monkeypatch)

    assert openrouter.anthropic_thinking_kwarg(max_tokens=600) == {
        "type": "disabled"
    }
    body = openrouter.reasoning_extra_body(max_tokens=600)
    assert body["reasoning"]["enabled"] is False
    assert body["reasoning"]["effort"] == "none"


def test_the_env_budget_override_is_clamped_too(monkeypatch):
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "medium")
    monkeypatch.setenv("OPENROUTER_REASONING_MAX_TOKENS", "6000")
    _fresh_warn_state(monkeypatch)

    expected = 2000 - openrouter._MIN_ANSWER_TOKENS
    assert openrouter.anthropic_thinking_kwarg(max_tokens=2000) == {
        "type": "enabled",
        "budget_tokens": expected,
    }


def test_the_clamp_is_announced_once_per_pairing(monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "medium")
    monkeypatch.delenv("OPENROUTER_REASONING_MAX_TOKENS", raising=False)
    _fresh_warn_state(monkeypatch)

    for _ in range(3):
        openrouter.anthropic_thinking_kwarg(max_tokens=2000)
    out = capsys.readouterr().out
    assert out.count("leaves no room for an answer") == 1
    assert "max_tokens=2000" in out


def test_the_proxy_clamps_against_the_ceiling_it_is_sending(monkeypatch):
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "medium")
    monkeypatch.delenv("OPENROUTER_REASONING_MAX_TOKENS", raising=False)
    _fresh_warn_state(monkeypatch)
    recorded = {}

    class _Inner:
        def create(self, **kwargs):
            recorded.update(kwargs)
            return "ok"

    proxy = openrouter._OpenRouterMessages(_Inner())
    proxy.create(model="m", max_tokens=2000, messages=[])

    budget = recorded["thinking"]["budget_tokens"]
    assert budget + openrouter._MIN_ANSWER_TOKENS <= 2000
    assert recorded["extra_body"]["reasoning"]["max_tokens"] == budget


def test_the_proxy_leaves_a_recovery_call_unclamped(monkeypatch):
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "medium")
    monkeypatch.delenv("OPENROUTER_REASONING_MAX_TOKENS", raising=False)
    _fresh_warn_state(monkeypatch)
    recorded = {}

    class _Inner:
        def create(self, **kwargs):
            recorded.update(kwargs)
            return "ok"

    proxy = openrouter._OpenRouterMessages(_Inner())
    proxy.create(model="m", max_tokens=4096, messages=[])
    assert recorded["thinking"] == {"type": "enabled", "budget_tokens": 2048}


def test_the_shipping_default_pairing_can_return_text(monkeypatch):
    """The regression this clamp exists for, pinned across both modules.

    Neither constant is wrong on its own; they were only ever wrong together,
    so the guard has to read both -- a test that hardcodes 2048 and 2000 goes
    green again the moment one of them moves.
    """
    from dashboard.backend.infrastructure.llm.backtest_harness import (
        DEFAULT_MAX_OUTPUT_TOKENS,
    )

    monkeypatch.delenv("OPENROUTER_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("OPENROUTER_REASONING_MAX_TOKENS", raising=False)
    _fresh_warn_state(monkeypatch)

    thinking = openrouter.anthropic_thinking_kwarg(
        max_tokens=DEFAULT_MAX_OUTPUT_TOKENS
    )
    if thinking.get("type") == "enabled":
        assert thinking["budget_tokens"] < DEFAULT_MAX_OUTPUT_TOKENS


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
