"""Provider credential validation tests with no real network or keys."""

from __future__ import annotations

import httpx
import pytest

from dashboard.backend.infrastructure.llm.adapters import get_adapter
from dashboard.backend.infrastructure.llm.adapters import safe_http
from dashboard.backend.infrastructure.llm.adapters import base as adapter_base


@pytest.mark.parametrize(
    ("adapter_type", "base_url", "expected_path", "header"),
    [
        ("openrouter", "https://openrouter.ai/api/v1", "/api/v1/key", "Authorization"),
        ("openai", "https://api.openai.com/v1", "/v1/models", "Authorization"),
        ("openai_compatible", "https://models.example.com/v1", "/v1/models", "Authorization"),
        ("anthropic", "https://api.anthropic.com", "/v1/models", "x-api-key"),
        ("gemini", "https://generativelanguage.googleapis.com", "/v1beta/models", "x-goog-api-key"),
    ],
)
def test_validation_uses_authenticated_provider_endpoint(adapter_type, base_url, expected_path, header):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        payload = (
            {"data": {"label": "test key"}}
            if adapter_type == "openrouter"
            else {
                "data": [{"id": "fake/model"}],
                "models": [{"name": "models/fake-gemini"}],
            }
        )
        return httpx.Response(
            200,
            json=payload,
        )

    adapter = get_adapter(adapter_type)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = adapter.validate(base_url, "sk-fake-validation-secret", client=client)

    assert result.status == "verified"
    assert calls and calls[0].method == "GET"
    assert calls[0].url.path == expected_path
    assert calls[0].headers[header]
    assert "sk-fake-validation-secret" in calls[0].headers[header]
    assert calls[0].url.query == b""
    assert all(call.method == "GET" for call in calls)


@pytest.mark.parametrize("status", [401, 403])
def test_authentication_failure_is_invalid(status):
    adapter = get_adapter("openrouter")
    transport = httpx.MockTransport(lambda _request: httpx.Response(status))
    with httpx.Client(transport=transport) as client:
        result = adapter.validate("https://openrouter.ai/api/v1", "sk-fake-invalid", client=client)
    assert result.status == "invalid"
    assert result.message == "The provider rejected this API key."


@pytest.mark.parametrize("adapter_type,base_url", [
    ("openai", "https://api.openai.com/v1"),
    ("anthropic", "https://api.anthropic.com"),
    ("gemini", "https://generativelanguage.googleapis.com"),
])
@pytest.mark.parametrize("status", [401, 403])
def test_each_native_provider_rejects_bad_auth(adapter_type, base_url, status):
    adapter = get_adapter(adapter_type)
    transport = httpx.MockTransport(lambda _request: httpx.Response(status))
    with httpx.Client(transport=transport) as client:
        result = adapter.validate(base_url, "sk-fake-invalid", client=client)
    assert result.status == "invalid"


@pytest.mark.parametrize("status", [429, 500, 503])
def test_transient_provider_failure_is_unavailable(status):
    adapter = get_adapter("openai")
    transport = httpx.MockTransport(lambda _request: httpx.Response(status))
    with httpx.Client(transport=transport) as client:
        result = adapter.validate("https://api.openai.com/v1", "sk-fake-transient", client=client)
    assert result.status == "verification_unavailable"


def test_redirect_is_not_followed_and_is_unavailable():
    adapter = get_adapter("openai")
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(302, headers={"location": "https://evil.example/models"})
    )
    with httpx.Client(transport=transport, follow_redirects=False) as client:
        result = adapter.validate("https://api.openai.com/v1", "sk-fake-redirect", client=client)
    assert result.status == "verification_unavailable"


def test_empty_model_discovery_response_is_not_verified():
    adapter = get_adapter("openai")
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, json={"data": []}))
    with httpx.Client(transport=transport, follow_redirects=False) as client:
        result = adapter.validate("https://api.openai.com/v1", "sk-fake-empty", client=client)
    assert result.status == "verification_unavailable"
    assert result.message == "Provider returned an invalid model list."


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(
            200,
            text="<html>not a model list</html>",
            headers={"content-type": "text/html"},
        ),
        httpx.Response(200, json={"error": {"message": "bad key"}}),
        httpx.Response(200, text="{not-json}"),
    ],
)
def test_malformed_discovery_response_is_not_verified(response):
    adapter = get_adapter("openai")
    transport = httpx.MockTransport(lambda _request: response)
    with httpx.Client(transport=transport, follow_redirects=False) as client:
        result = adapter.validate("https://api.openai.com/v1", "sk-fake-malformed", client=client)
    assert result.status == "verification_unavailable"


def test_private_dns_result_is_rejected_before_network(monkeypatch):
    monkeypatch.delenv("BROKER_CREDENTIAL_VERIFICATION_PROXY", raising=False)
    calls = []

    def fake_getaddrinfo(*args, **kwargs):
        calls.append((args, kwargs))
        return [
            (
                safe_http.socket.AF_INET,
                safe_http.socket.SOCK_STREAM,
                safe_http.socket.IPPROTO_TCP,
                "",
                ("127.0.0.1", 443),
            )
        ]

    monkeypatch.setattr(safe_http.socket, "getaddrinfo", fake_getaddrinfo)
    adapter = get_adapter("openai")
    result = adapter.validate("https://public.example/v1", "sk-fake-private")
    assert result.status == "verification_unavailable"
    assert result.message == "Provider address is not allowed."
    assert calls


@pytest.mark.parametrize(
    ("adapter_type", "official_url"),
    [
        ("openrouter", "https://openrouter.ai/api/v1/key"),
        ("openai", "https://api.openai.com/v1/models"),
        ("anthropic", "https://api.anthropic.com/v1/models"),
        (
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta/models",
        ),
    ],
)
def test_explicit_proxy_is_limited_to_native_official_origins(
    monkeypatch, adapter_type, official_url
):
    monkeypatch.setenv("BROKER_CREDENTIAL_VERIFICATION_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setattr(
        adapter_base,
        "build_explicit_proxy_transport",
        lambda proxy: ("proxy", proxy),
    )
    monkeypatch.setattr(
        adapter_base,
        "build_pinned_transport",
        lambda url: ("pinned", url),
    )

    assert get_adapter(adapter_type)._transport(official_url) == (
        "proxy",
        "http://127.0.0.1:7897",
    )
    assert get_adapter(adapter_type)._transport(
        "https://custom-provider.example/v1/models"
    ) == ("pinned", "https://custom-provider.example/v1/models")


def test_openai_compatible_never_uses_explicit_proxy(monkeypatch):
    monkeypatch.setenv("BROKER_CREDENTIAL_VERIFICATION_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setattr(
        adapter_base,
        "build_explicit_proxy_transport",
        lambda proxy: ("proxy", proxy),
    )
    monkeypatch.setattr(
        adapter_base,
        "build_pinned_transport",
        lambda url: ("pinned", url),
    )

    assert get_adapter("openai_compatible")._transport(
        "https://models.example.com/v1/models"
    ) == ("pinned", "https://models.example.com/v1/models")
