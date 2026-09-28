"""One provider request per attempt: the SDKs never replay, and the timeout is ours.

These drive the REAL openai/anthropic SDK clients (the pinned versions in
requirements.txt) over an ``httpx.MockTransport`` swapped in beneath
``build_safe_http_client``, so the timeout wiring, the SDK retry loop and the
adapters' error mapping all run for real. Only the socket is fake.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest
from anthropic import _base_client as anthropic_base_client

import dashboard.backend.infrastructure.llm.execution.adapters.anthropic as anthropic_module
import dashboard.backend.infrastructure.llm.execution.adapters.base as base_module
import dashboard.backend.infrastructure.llm.execution.adapters.gemini as gemini_module
import dashboard.backend.infrastructure.llm.execution.adapters.openai as openai_module
from dashboard.backend.domain.model_providers.models import (
    ProviderCapabilities,
    ProviderRecord,
)
from dashboard.backend.infrastructure.llm.execution import service as service_module
from dashboard.backend.infrastructure.llm.execution.errors import (
    ExecutionErrorCategory,
    RetryHint,
)
from dashboard.backend.infrastructure.llm.execution.models import (
    LLMExecutionRequest,
    LLMMessage,
    UsagePolicy,
)


SECRET = "sk-fake-retry-policy-test-only"
ADAPTERS_DIR = Path(base_module.__file__).resolve().parent
DEFAULT_WIRE_TIMEOUT = {"connect": 8.0, "read": 180.0, "write": 60.0, "pool": 60.0}


def _credential(provider_id: str):
    return SimpleNamespace(
        credential_id="credential-test-id",
        provider_id=provider_id,
        key_last_four="only",
        secret=SECRET,
    )


def _request(provider_id: str, model_id: str) -> LLMExecutionRequest:
    return LLMExecutionRequest(
        user_id=7,
        run_id="run-retry-policy",
        call_index=0,
        billing_mode="platform_credits",
        provider_id=provider_id,
        model_id=model_id,
        messages=(LLMMessage(role="user", content="Return one word."),),
        usage_policy=UsagePolicy(max_output_tokens=2000),
    )


_OPENAI_ROUTES = {
    "commonstack": (
        openai_module.OpenAICompatibleAdapter,
        ProviderRecord(
            provider_id="commonstack",
            display_name="CommonStack",
            adapter_type="openai_compatible",
            approved_base_url="https://api.commonstack.ai/v1",
            capabilities=ProviderCapabilities(model_allowlist=("qwen/qwen3.7-plus",)),
        ),
        "qwen/qwen3.7-plus",
    ),
    "openai": (
        openai_module.OpenAIAdapter,
        ProviderRecord(
            provider_id="openai",
            display_name="openai",
            adapter_type="openai",
            approved_base_url="https://api.openai.com/v1",
        ),
        "openai/gpt-5.5",
    ),
}
_ANTHROPIC_PROVIDER = ProviderRecord(
    provider_id="anthropic",
    display_name="anthropic",
    adapter_type="anthropic",
    approved_base_url="https://api.anthropic.com",
)
_ANTHROPIC_MODEL = "anthropic/claude-sonnet-4-6"
_GEMINI_PROVIDER = ProviderRecord(
    provider_id="gemini",
    display_name="gemini",
    adapter_type="gemini",
    approved_base_url="https://generativelanguage.googleapis.com/v1beta",
)
_GEMINI_MODEL = "google/gemini-3.1-pro-preview"

_CHAT_COMPLETION = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 0,
    "model": "qwen/qwen3.7-plus",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "HOLD"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
}
_ANTHROPIC_MESSAGE = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-6",
    "content": [{"type": "text", "text": "HOLD"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 12, "output_tokens": 3},
}


@pytest.fixture
def wire(monkeypatch):
    """Replace the pinned socket transport with a scripted one; record every send."""

    monkeypatch.delenv("BROKER_CREDENTIAL_VERIFICATION_PROXY", raising=False)
    state = SimpleNamespace(respond=None, sent=[], sdk_sleeps=[])

    def handler(request: httpx.Request) -> httpx.Response:
        state.sent.append(dict(request.extensions.get("timeout", {})))
        return state.respond(request)

    monkeypatch.setattr(
        base_module,
        "build_pinned_transport",
        lambda *_args, **_kwargs: httpx.MockTransport(handler),
    )
    # Both SDKs back off with ``time.sleep``; recording it makes any replay
    # visible and keeps a regression from sleeping for real.
    for sdk_client in (openai._base_client, anthropic_base_client):
        monkeypatch.setattr(sdk_client.time, "sleep", state.sdk_sleeps.append)
    return state


def _stall(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("generation still running", request=request)


@pytest.mark.parametrize("route", sorted(_OPENAI_ROUTES))
def test_real_openai_sdk_makes_one_request_on_read_timeout(wire, route):
    adapter_class, provider, model_id = _OPENAI_ROUTES[route]
    wire.respond = _stall

    with pytest.raises(base_module.ProviderExecutionError) as exc_info:
        adapter_class().complete(
            _request(route, model_id), _credential(route), provider
        )

    # On main this was 3 sends: the SDK regenerated the completion twice.
    assert len(wire.sent) == 1
    assert wire.sent[0] == DEFAULT_WIRE_TIMEOUT
    assert wire.sdk_sleeps == []
    error = exc_info.value
    assert error.category is ExecutionErrorCategory.PROVIDER_TIMEOUT
    assert error.timeout_phase == "read"
    assert error.retry_hint is RetryHint.NONE
    assert isinstance(error.__cause__, openai.APITimeoutError)
    assert isinstance(error.__cause__.__cause__, httpx.ReadTimeout)


def test_real_anthropic_sdk_makes_one_request_on_read_timeout(wire):
    wire.respond = _stall

    with pytest.raises(base_module.ProviderExecutionError) as exc_info:
        anthropic_module.AnthropicExecutionAdapter().complete(
            _request("anthropic", _ANTHROPIC_MODEL),
            _credential("anthropic"),
            _ANTHROPIC_PROVIDER,
        )

    assert len(wire.sent) == 1
    assert wire.sent[0] == DEFAULT_WIRE_TIMEOUT
    assert wire.sdk_sleeps == []
    assert exc_info.value.category is ExecutionErrorCategory.PROVIDER_TIMEOUT
    assert exc_info.value.timeout_phase == "read"
    assert exc_info.value.retry_hint is RetryHint.NONE


def test_openai_sdk_default_would_regenerate_three_times(wire):
    """Control: why the adapters pass ``max_retries`` and ``timeout`` explicitly.

    Built the way the adapter used to build it -- SDK defaults, our 60s
    http_client -- a single read timeout becomes three generations. If an SDK
    upgrade flips this, the explicit kwargs are still right; this test just
    stops documenting the reason.
    """

    wire.respond = _stall
    client = openai.OpenAI(
        api_key=SECRET,
        base_url="https://api.commonstack.ai/v1",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda request: (
                wire.sent.append(dict(request.extensions["timeout"])),
                _stall(request),
            )[1]),
            timeout=httpx.Timeout(60.0, connect=8.0),
        ),
    )

    with pytest.raises(openai.APITimeoutError):
        client.chat.completions.create(
            model="qwen/qwen3.7-plus",
            messages=[{"role": "user", "content": "Return one word."}],
            max_tokens=2000,
        )

    assert len(wire.sent) == 3, "SDK defaults no longer replay a read timeout"
    assert all(sent["read"] == 60.0 for sent in wire.sent)


def test_read_timeout_setting_reaches_the_wire(wire, monkeypatch):
    monkeypatch.setattr(base_module, "PROVIDER_READ_TIMEOUT_SECONDS", 45)
    wire.respond = _stall
    adapter_class, provider, model_id = _OPENAI_ROUTES["commonstack"]

    with pytest.raises(base_module.ProviderExecutionError):
        adapter_class().complete(
            _request("commonstack", model_id), _credential("commonstack"), provider
        )

    assert wire.sent == [{**DEFAULT_WIRE_TIMEOUT, "read": 45.0}]


def test_real_sdk_status_error_is_single_attempt_with_hints(wire):
    wire.respond = lambda request: httpx.Response(
        503,
        headers={"retry-after": "7"},
        json={"error": {"message": "overloaded"}},
        request=request,
    )
    adapter_class, provider, model_id = _OPENAI_ROUTES["commonstack"]

    with pytest.raises(base_module.ProviderExecutionError) as exc_info:
        adapter_class().complete(
            _request("commonstack", model_id), _credential("commonstack"), provider
        )

    assert len(wire.sent) == 1
    assert wire.sdk_sleeps == []
    error = exc_info.value
    assert error.category is ExecutionErrorCategory.PROVIDER_UNAVAILABLE
    assert error.retry_hint is RetryHint.REJECTED
    assert error.provider_status_code == 503
    assert error.retry_after_seconds == 7.0


def test_real_sdks_still_complete_through_the_new_wiring(wire):
    wire.respond = lambda request: httpx.Response(
        200, json=_CHAT_COMPLETION, request=request
    )
    adapter_class, provider, model_id = _OPENAI_ROUTES["commonstack"]
    openai_result = adapter_class().complete(
        _request("commonstack", model_id), _credential("commonstack"), provider
    )

    wire.respond = lambda request: httpx.Response(
        200, json=_ANTHROPIC_MESSAGE, request=request
    )
    anthropic_result = anthropic_module.AnthropicExecutionAdapter().complete(
        _request("anthropic", _ANTHROPIC_MODEL),
        _credential("anthropic"),
        _ANTHROPIC_PROVIDER,
    )

    assert (openai_result.text, openai_result.usage.output_tokens) == ("HOLD", 3)
    assert (anthropic_result.text, anthropic_result.usage.output_tokens) == ("HOLD", 3)
    assert wire.sent == [DEFAULT_WIRE_TIMEOUT, DEFAULT_WIRE_TIMEOUT]


class _Closable:
    def close(self) -> None:
        return None


@pytest.mark.parametrize(
    ("module", "adapter_factory", "provider", "model_id", "call"),
    [
        (
            openai_module,
            openai_module.OpenAIAdapter,
            _OPENAI_ROUTES["openai"][1],
            "openai/gpt-5.5",
            "chat",
        ),
        (
            openai_module,
            openai_module.OpenRouterAdapter,
            ProviderRecord(
                provider_id="openrouter",
                display_name="openrouter",
                adapter_type="openrouter",
                approved_base_url="https://openrouter.ai/api/v1",
            ),
            "openai/gpt-5.5",
            "chat",
        ),
        (
            openai_module,
            openai_module.OpenAICompatibleAdapter,
            _OPENAI_ROUTES["commonstack"][1],
            "qwen/qwen3.7-plus",
            "chat",
        ),
        (
            anthropic_module,
            anthropic_module.AnthropicExecutionAdapter,
            _ANTHROPIC_PROVIDER,
            _ANTHROPIC_MODEL,
            "messages",
        ),
    ],
)
def test_every_sdk_adapter_disables_sdk_retries(
    monkeypatch, module, adapter_factory, provider, model_id, call
):
    factory_kwargs = {}
    builder_kwargs = {}

    def create(**_kwargs):
        raise httpx.ReadTimeout("stalled")

    def client_factory(**kwargs):
        factory_kwargs.update(kwargs)
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            messages=SimpleNamespace(create=create),
            close=lambda: None,
        )

    def build(*_args, **kwargs):
        builder_kwargs.update(kwargs)
        return _Closable()

    monkeypatch.setattr(module, "build_safe_http_client", build)

    with pytest.raises(base_module.ProviderExecutionError):
        adapter_factory(client_factory=client_factory).complete(
            _request(provider.provider_id, model_id),
            _credential(provider.provider_id),
            provider,
        )

    assert factory_kwargs["max_retries"] == 0
    assert factory_kwargs["timeout"] == base_module.provider_http_timeout()
    assert builder_kwargs["timeout"] is factory_kwargs["timeout"]


def test_sdk_adapter_sources_pass_max_retries_and_timeout():
    """Source guard: a later edit cannot quietly drop the kwargs or re-enable retries."""

    assert base_module.SDK_MAX_RETRIES == 0
    for name in ("openai.py", "anthropic.py"):
        tree = ast.parse((ADAPTERS_DIR / name).read_text(encoding="utf-8"))
        factory_calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "client_factory":
                factory_calls.append(node)
            if isinstance(func, ast.Attribute) and func.attr in {"with_options", "copy"}:
                assert all(kw.arg != "max_retries" for kw in node.keywords), name
        assert factory_calls, f"{name}: no client_factory(...) call found"
        for node in factory_calls:
            keywords = {kw.arg: kw.value for kw in node.keywords}
            assert "timeout" in keywords, name
            retries = keywords.get("max_retries")
            assert (
                isinstance(retries, ast.Name) and retries.id == "SDK_MAX_RETRIES"
            ) or (isinstance(retries, ast.Constant) and retries.value == 0), name


def test_gemini_gets_provider_timeout_and_keeps_status_branch(wire):
    client = base_module.build_safe_http_client(_GEMINI_PROVIDER.approved_base_url)
    try:
        assert client.timeout == base_module.provider_http_timeout()
    finally:
        client.close()

    wire.respond = _stall
    with pytest.raises(base_module.ProviderExecutionError) as timed_out:
        gemini_module.GeminiExecutionAdapter().complete(
            _request("gemini", _GEMINI_MODEL), _credential("gemini"), _GEMINI_PROVIDER
        )
    assert len(wire.sent) == 1
    assert wire.sent[0] == DEFAULT_WIRE_TIMEOUT
    assert timed_out.value.category is ExecutionErrorCategory.PROVIDER_TIMEOUT
    assert timed_out.value.timeout_phase == "read"
    assert timed_out.value.retry_hint is RetryHint.NONE

    # The status branch raises its own error and carries no hint, so a Gemini
    # 5xx is never repeated -- unchanged from before the policy.
    wire.respond = lambda request: httpx.Response(503, json={}, request=request)
    with pytest.raises(base_module.ProviderExecutionError) as unavailable:
        gemini_module.GeminiExecutionAdapter().complete(
            _request("gemini", _GEMINI_MODEL), _credential("gemini"), _GEMINI_PROVIDER
        )
    assert unavailable.value.category is ExecutionErrorCategory.PROVIDER_UNAVAILABLE
    assert unavailable.value.retry_hint is RetryHint.NONE


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_read_timeout_parse_defaults_silently(raw, capsys):
    assert base_module._parse_provider_read_timeout(raw) == 180
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(("raw", "expected"), [("30", 30), ("600", 600), (" 120 ", 120)])
def test_read_timeout_parse_accepts_in_range(raw, expected, capsys):
    assert base_module._parse_provider_read_timeout(raw) == expected
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("raw", ["abc", "90.5", "0", "-5", "29", "601", "1800"])
def test_read_timeout_parse_rejects_junk_and_out_of_range(raw, capsys):
    assert base_module._parse_provider_read_timeout(raw) == 180
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("WARNING: LLM_PROVIDER_READ_TIMEOUT_SECONDS ")


def test_suite_runs_with_default_read_timeout():
    # conftest strips LLM_PROVIDER_READ_TIMEOUT_SECONDS before base.py imports.
    assert base_module.PROVIDER_READ_TIMEOUT_SECONDS == 180
    assert base_module.provider_http_timeout() == httpx.Timeout(
        connect=8.0, read=180.0, write=60.0, pool=60.0
    )


def test_attempt_failure_line_is_relay_safe(capsys):
    from dashboard.backend.api.routers import backtests

    error = base_module.ProviderExecutionError(
        ExecutionErrorCategory.PROVIDER_UNAVAILABLE,
        retry_hint=RetryHint.REJECTED,
        provider_status_code=503,
    )
    error.provider_elapsed_seconds = 1.25
    request = _request("commonstack", "qwen/qwen3.7-plus").model_copy(
        update={"run_id": "agent_20260928_024706_cbda3555", "call_index": 4}
    )
    service_module._report_provider_attempt_failed(request, 2, error, "retry")
    unsafe = request.model_copy(update={"run_id": "authorization=Bearer x"})
    service_module._report_provider_attempt_failed(unsafe, 0, error, "none")

    first, second = capsys.readouterr().out.splitlines()
    assert first == (
        "ERROR: llm.provider_attempt_failed run=agent_20260928_024706_cbda3555 "
        "call=4 attempt=2 provider=commonstack model=qwen/qwen3.7-plus "
        "billing=platform_credits category=provider_unavailable phase=- "
        "status=503 hint=rejected elapsed_s=1.2 read_timeout_s=180 next=retry"
    )
    # A run id that is not a plain identifier is dropped, never echoed.
    assert " run=- " in second
    for line in (first, second):
        assert line.startswith(backtests._RELAYED_CHILD_LINE_PREFIX)
        assert backtests._redact_credentials(line) == line
        assert SECRET not in line
