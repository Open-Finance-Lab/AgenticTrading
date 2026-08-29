"""Provider adapters translate ATL model ids only at the network boundary."""

from __future__ import annotations

from types import SimpleNamespace

import dashboard.backend.infrastructure.llm.execution.adapters.anthropic as anthropic_module
import dashboard.backend.infrastructure.llm.execution.adapters.gemini as gemini_module
import dashboard.backend.infrastructure.llm.execution.adapters.openai as openai_module
from dashboard.backend.domain.model_providers.models import ProviderRecord
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
    )


def _openai_response(model: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok"),
            )
        ],
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
