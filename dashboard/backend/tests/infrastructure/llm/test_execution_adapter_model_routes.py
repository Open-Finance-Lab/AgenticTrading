"""Provider adapters translate ATL model ids only at the network boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import dashboard.backend.infrastructure.llm.execution.adapters.anthropic as anthropic_module
import dashboard.backend.infrastructure.llm.execution.adapters.gemini as gemini_module
import dashboard.backend.infrastructure.llm.execution.adapters.openai as openai_module
from dashboard.backend.domain.model_providers.models import (
    ProviderCapabilities,
    ProviderRecord,
)
from dashboard.backend.infrastructure.llm.execution.models import (
    LLMExecutionRequest,
    LLMMessage,
    UsagePolicy,
)


class _Closable:
    def close(self) -> None:
        return None


def _credential(provider_id: str):
    return SimpleNamespace(
        credential_id="credential-test-id",
        provider_id=provider_id,
        key_last_four="test",
        secret="sk-fake-adapter-test-only",
    )


def _provider(provider_id: str, adapter_type: str, base_url: str) -> ProviderRecord:
    return ProviderRecord(
        provider_id=provider_id,
        display_name=provider_id,
        adapter_type=adapter_type,
        approved_base_url=base_url,
    )


def _request(
    provider_id: str,
    model_id: str,
    *,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
) -> LLMExecutionRequest:
    return LLMExecutionRequest(
        user_id=7,
        run_id="run-model-route",
        call_index=0,
        billing_mode="byok",
        provider_id=provider_id,
        model_id=model_id,
        messages=(LLMMessage(role="user", content="Return one word."),),
        usage_policy=UsagePolicy(max_output_tokens=16),
        reasoning_effort=reasoning_effort,
        temperature=temperature,
    )


def _openai_response(model: str, finish_reason: str | None = None):
    choice = SimpleNamespace(message=SimpleNamespace(content="ok"))
    if finish_reason is not None:
        choice.finish_reason = finish_reason
    return SimpleNamespace(
        choices=[choice],
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
        model=model,
    )


def test_openai_uses_native_model_and_keeps_canonical_result(monkeypatch):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return _openai_response("gpt-5.5")

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        ),
        close=lambda: None,
    )
    monkeypatch.setattr(
        openai_module,
        "build_safe_http_client",
        lambda *_args, **_kwargs: _Closable(),
    )
    adapter = openai_module.OpenAIAdapter(
        client_factory=lambda **_kwargs: client,
    )
    request = _request("openai", "openai/gpt-5.5")

    result = adapter.complete(
        request,
        _credential("openai"),
        _provider("openai", "openai", "https://api.openai.com/v1"),
    )

    assert captured["model"] == "gpt-5.5"
    assert request.model_id == "openai/gpt-5.5"
    assert result.model_id == "openai/gpt-5.5"


def test_openrouter_keeps_provider_qualified_model(monkeypatch):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return _openai_response("openai/gpt-5.5")

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        ),
        close=lambda: None,
    )
    monkeypatch.setattr(
        openai_module,
        "build_safe_http_client",
        lambda *_args, **_kwargs: _Closable(),
    )
    adapter = openai_module.OpenRouterAdapter(
        client_factory=lambda **_kwargs: client,
    )
    request = _request("openrouter", "openai/gpt-5.5")

    result = adapter.complete(
        request,
        _credential("openrouter"),
        _provider(
            "openrouter",
            "openrouter",
            "https://openrouter.ai/api/v1",
        ),
    )

    assert captured["model"] == "openai/gpt-5.5"
    assert result.model_id == "openai/gpt-5.5"


def test_openrouter_reasoning_none_disables_reasoning(monkeypatch):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return _openai_response("qwen/qwen3.7-plus")

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        close=lambda: None,
    )
    monkeypatch.setattr(
        openai_module,
        "build_safe_http_client",
        lambda *_args, **_kwargs: _Closable(),
    )
    adapter = openai_module.OpenRouterAdapter(client_factory=lambda **_kwargs: client)

    adapter.complete(
        _request(
            "openrouter",
            "qwen/qwen3.7-plus",
            reasoning_effort="none",
        ),
        _credential("openrouter"),
        _provider("openrouter", "openrouter", "https://openrouter.ai/api/v1"),
    )

    assert captured["extra_body"] == {
        "reasoning": {"effort": "none", "enabled": False, "exclude": True}
    }


def test_commonstack_openai_compatible_route_preserves_reasoning_and_ceiling(
    monkeypatch,
):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return _openai_response("qwen/qwen3.7-plus")

    client = _openai_client(create)
    monkeypatch.setattr(
        openai_module,
        "build_safe_http_client",
        lambda *_args, **_kwargs: _Closable(),
    )
    adapter = openai_module.OpenAICompatibleAdapter(
        client_factory=lambda **_kwargs: client,
    )
    provider = ProviderRecord(
        provider_id="commonstack",
        display_name="CommonStack",
        adapter_type="openai_compatible",
        approved_base_url="https://api.commonstack.ai/v1",
        capabilities=ProviderCapabilities(
            model_allowlist=("qwen/qwen3.7-plus",),
            reasoning=True,
        ),
    )
    request = LLMExecutionRequest(
        user_id=7,
        run_id="run-commonstack-route",
        call_index=0,
        billing_mode="platform_credits",
        provider_id="commonstack",
        model_id="qwen/qwen3.7-plus",
        messages=(LLMMessage(role="user", content="Return one word."),),
        usage_policy=UsagePolicy(max_output_tokens=4096),
        temperature=0.2,
        reasoning_effort="high",
    )

    adapter.complete(request, _credential("commonstack"), provider)

    assert captured["model"] == "qwen/qwen3.7-plus"
    assert captured["max_tokens"] == 4096
    assert captured["temperature"] == 0.2
    assert captured["extra_body"] == {"reasoning": {"effort": "high"}}


def test_anthropic_uses_native_model_and_keeps_canonical_result(monkeypatch):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            usage=SimpleNamespace(input_tokens=2, output_tokens=1),
            model="claude-sonnet-4-6",
        )

    client = SimpleNamespace(
        messages=SimpleNamespace(create=create),
        close=lambda: None,
    )
    monkeypatch.setattr(
        anthropic_module,
        "build_safe_http_client",
        lambda *_args, **_kwargs: _Closable(),
    )
    adapter = anthropic_module.AnthropicExecutionAdapter(
        client_factory=lambda **_kwargs: client,
    )
    request = _request("anthropic", "anthropic/claude-sonnet-4-6")

    result = adapter.complete(
        request,
        _credential("anthropic"),
        _provider(
            "anthropic",
            "anthropic",
            "https://api.anthropic.com",
        ),
    )

    assert captured["model"] == "claude-sonnet-4-6"
    assert result.model_id == "anthropic/claude-sonnet-4-6"


def test_gemini_uses_native_model_in_endpoint_and_keeps_canonical_result(
    monkeypatch,
):
    captured = {}

    class _Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "candidates": [
                    {"content": {"parts": [{"text": "ok"}]}}
                ],
                "usageMetadata": {
                    "promptTokenCount": 2,
                    "candidatesTokenCount": 1,
                },
            }

    class _HTTPClient(_Closable):
        def post(self, url, *, headers, json):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return _Response()

    monkeypatch.setattr(
        gemini_module,
        "build_safe_http_client",
        lambda *_args, **_kwargs: _HTTPClient(),
    )
    request = _request("gemini", "google/gemini-3.1-pro-preview")

    result = gemini_module.GeminiExecutionAdapter().complete(
        request,
        _credential("gemini"),
        _provider(
            "gemini",
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta",
        ),
    )

    assert captured["url"].endswith(
        "/models/gemini-3.1-pro-preview:generateContent"
    )
    assert result.model_id == "google/gemini-3.1-pro-preview"


def _openai_client(create):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        close=lambda: None,
    )


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [("length", "max_tokens"), ("stop", "stop"), (None, None)],
)
def test_openai_reports_normalised_finish_reason(monkeypatch, finish_reason, expected):
    monkeypatch.setattr(
        openai_module,
        "build_safe_http_client",
        lambda *_args, **_kwargs: _Closable(),
    )
    client = _openai_client(lambda **_kwargs: _openai_response("gpt-5.5", finish_reason))
    adapter = openai_module.OpenAIAdapter(client_factory=lambda **_kwargs: client)

    result = adapter.complete(
        _request("openai", "openai/gpt-5.5"),
        _credential("openai"),
        _provider("openai", "openai", "https://api.openai.com/v1"),
    )

    assert result.finish_reason == expected


def test_gemini_reports_max_tokens_finish_reason(monkeypatch):
    class _Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "{\"orders\": ["}]},
                        "finishReason": "MAX_TOKENS",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1},
            }

    class _HTTPClient(_Closable):
        def post(self, url, *, headers, json):
            return _Response()

    monkeypatch.setattr(
        gemini_module,
        "build_safe_http_client",
        lambda *_args, **_kwargs: _HTTPClient(),
    )

    result = gemini_module.GeminiExecutionAdapter().complete(
        _request("gemini", "google/gemini-3.1-pro-preview"),
        _credential("gemini"),
        _provider("gemini", "gemini", "https://generativelanguage.googleapis.com/v1beta"),
    )

    assert result.finish_reason == "max_tokens"


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [("max_tokens", "max_tokens"), ("end_turn", "end_turn"), (None, None)],
)
def test_anthropic_reports_normalised_stop_reason(monkeypatch, stop_reason, expected):
    def create(**_kwargs):
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            usage=SimpleNamespace(input_tokens=2, output_tokens=1),
            model="claude-sonnet-4-6",
            stop_reason=stop_reason,
        )

    client = SimpleNamespace(
        messages=SimpleNamespace(create=create),
        close=lambda: None,
    )
    monkeypatch.setattr(
        anthropic_module,
        "build_safe_http_client",
        lambda *_args, **_kwargs: _Closable(),
    )
    adapter = anthropic_module.AnthropicExecutionAdapter(
        client_factory=lambda **_kwargs: client,
    )

    result = adapter.complete(
        _request("anthropic", "anthropic/claude-sonnet-4-6"),
        _credential("anthropic"),
        _provider("anthropic", "anthropic", "https://api.anthropic.com"),
    )

    assert result.finish_reason == expected


def test_normalize_finish_reason_folds_vendor_spellings_and_stays_bounded():
    from dashboard.backend.infrastructure.llm.execution.adapters.base import (
        normalize_finish_reason,
    )

    assert normalize_finish_reason("length") == "max_tokens"
    assert normalize_finish_reason("MAX_TOKENS") == "max_tokens"
    assert normalize_finish_reason(" max_tokens ") == "max_tokens"
    assert normalize_finish_reason("STOP") == "stop"
    assert normalize_finish_reason("") is None
    assert normalize_finish_reason(None) is None
    assert normalize_finish_reason(3) is None
    # An OpenAI-compatible provider may put anything here; it must never
    # exceed the result model's bound and fail a successful call.
    assert len(normalize_finish_reason("x" * 80)) == 32


def test_native_openai_sends_reasoning_effort_as_a_top_level_parameter(monkeypatch):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return _openai_response("gpt-5.5")

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        close=lambda: None,
    )
    monkeypatch.setattr(
        openai_module, "build_safe_http_client", lambda *_args, **_kwargs: _Closable()
    )
    adapter = openai_module.OpenAIAdapter(client_factory=lambda **_kwargs: client)

    adapter.complete(
        _request("openai", "openai/gpt-5.5", reasoning_effort="low"),
        _credential("openai"),
        _provider("openai", "openai", "https://api.openai.com/v1"),
    )

    assert captured["reasoning_effort"] == "low"
    assert "extra_body" not in captured
    assert "temperature" not in captured


def test_native_openai_sends_nothing_extra_when_no_effort_is_requested(monkeypatch):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return _openai_response("gpt-5.5")

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        close=lambda: None,
    )
    monkeypatch.setattr(
        openai_module, "build_safe_http_client", lambda *_args, **_kwargs: _Closable()
    )
    adapter = openai_module.OpenAIAdapter(client_factory=lambda **_kwargs: client)

    adapter.complete(
        _request("openai", "openai/gpt-5.5"),
        _credential("openai"),
        _provider("openai", "openai", "https://api.openai.com/v1"),
    )

    assert "reasoning_effort" not in captured
    assert "extra_body" not in captured


def _compatible_capture(
    monkeypatch, reasoning_effort, *, provider_id="commonstack", temperature=None
):
    """Send one DeepSeek call through an openai_compatible provider.

    Returns ``(kwargs sent to the SDK, adapter response)``.
    """
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return _openai_response("deepseek/deepseek-v4-pro")

    client = _openai_client(create)
    monkeypatch.setattr(
        openai_module, "build_safe_http_client", lambda *_args, **_kwargs: _Closable()
    )
    adapter = openai_module.OpenAICompatibleAdapter(
        client_factory=lambda **_kwargs: client,
    )
    provider = ProviderRecord(
        provider_id=provider_id,
        display_name=provider_id,
        adapter_type="openai_compatible",
        approved_base_url="https://api.commonstack.ai/v1",
        capabilities=ProviderCapabilities(
            model_allowlist=("deepseek/deepseek-v4-pro",),
            reasoning=True,
        ),
    )
    response = adapter.complete(
        _request(
            provider_id,
            "deepseek/deepseek-v4-pro",
            reasoning_effort=reasoning_effort,
            temperature=temperature,
        ),
        _credential(provider_id),
        provider,
    )
    return captured, response


def _commonstack_capture(monkeypatch, reasoning_effort):
    return _compatible_capture(monkeypatch, reasoning_effort)[0]


@pytest.mark.parametrize("off", ["none", "off", "false", "0", "disabled", "NONE"])
def test_commonstack_sends_thinking_disabled_for_an_off_value(monkeypatch, off):
    """CommonStack honours no graduated reasoning control for DeepSeek V4 Pro
    or Qwen3.7 Plus. The 2026-10-01 probe (#539) found `reasoning.effort`, a
    top-level `reasoning_effort`, `reasoning.enabled:false` and
    `thinking.budget_tokens` all ignored, and only `thinking: {type:
    "disabled"}` honoured. It is sent *instead of* `reasoning`, not beside it:
    a `reasoning` key next to it would be the request the probe showed does
    nothing."""
    captured = _commonstack_capture(monkeypatch, off)

    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning" not in captured["extra_body"]
    assert "reasoning_effort" not in captured


def test_commonstack_keeps_reasoning_effort_for_a_graduated_value(monkeypatch):
    """GPT-5.5 on CommonStack does honour `reasoning.effort` (`low` used 512
    reasoning tokens in the probe), so anything outside the off-set keeps
    today's shape."""
    captured = _commonstack_capture(monkeypatch, "low")

    assert captured["extra_body"] == {"reasoning": {"effort": "low"}}
    assert "thinking" not in captured["extra_body"]


def test_thinking_disabled_is_commonstack_only(monkeypatch):
    """Keyed on the provider, not on the openai_compatible adapter type.

    An admin-registered OpenAI-shaped server (DeepSeek's own API, vLLM,
    Together) was never probed with a `thinking` body field: a strict one
    400s on the unknown key, a lenient one thinks anyway while the run says
    thinking was off. It keeps the `reasoning` shape it was sent before.
    """
    captured, response = _compatible_capture(
        monkeypatch, "none", provider_id="deepseek_direct"
    )

    assert captured["extra_body"] == {"reasoning": {"effort": "none"}}
    assert response.sampling_wire == "reasoning.effort=none"


def test_commonstack_reports_the_wire_shape_it_sent(monkeypatch):
    _captured, response = _compatible_capture(monkeypatch, "none", temperature=0.0)

    assert response.sampling_wire == "temperature=0.0;thinking=disabled"


def test_an_adapter_that_sends_no_sampling_reports_none(monkeypatch):
    captured, response = _compatible_capture(monkeypatch, None)

    assert "extra_body" not in captured and "temperature" not in captured
    assert response.sampling_wire is None


def test_openrouter_reports_its_thinking_off_shape(monkeypatch):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return _openai_response("deepseek/deepseek-v4-pro")

    monkeypatch.setattr(
        openai_module, "build_safe_http_client", lambda *_args, **_kwargs: _Closable()
    )
    adapter = openai_module.OpenRouterAdapter(
        client_factory=lambda **_kwargs: _openai_client(create)
    )
    response = adapter.complete(
        _request("openrouter", "deepseek/deepseek-v4-pro", reasoning_effort="none"),
        _credential("openrouter"),
        _provider("openrouter", "openrouter", "https://openrouter.ai/api/v1"),
    )

    assert captured["extra_body"] == {
        "reasoning": {"effort": "none", "enabled": False, "exclude": True}
    }
    assert response.sampling_wire == "reasoning.effort=none,enabled=false"


def test_native_openai_reports_its_top_level_effort(monkeypatch):
    monkeypatch.setattr(
        openai_module, "build_safe_http_client", lambda *_args, **_kwargs: _Closable()
    )
    adapter = openai_module.OpenAIAdapter(
        client_factory=lambda **_kwargs: _openai_client(
            lambda **_kw: _openai_response("gpt-5.5")
        )
    )
    response = adapter.complete(
        _request("openai", "openai/gpt-5.5", reasoning_effort="low"),
        _credential("openai"),
        _provider("openai", "openai", "https://api.openai.com/v1"),
    )

    assert response.sampling_wire == "reasoning_effort=low"
