"""CommonStack thinking-off for the leaderboard's legacy harness path.

CommonStack's ``/v1/messages`` ignores every thinking control for DeepSeek V4
Pro and Qwen3.7 Plus; only ``/v1/chat/completions`` honours
``thinking: {type: "disabled"}`` (2026-10-02 probe). An entry that turns
reasoning off is therefore served on chat completions, with the response
reshaped so the harness reads it exactly as it reads an Anthropic reply.

The wire tests drive the real OpenAI SDK over an ``httpx.MockTransport``. The
retry, timeout and header behaviour under test belongs to the SDK, and a fake
client would only re-assert what the fake was written to do.
"""
import json
from datetime import datetime
from pathlib import Path

import httpx
import openai
import pytest

from dashboard.backend.domain.backtesting.portfolio_manager import PortfolioManager
from dashboard.backend.infrastructure.llm import backtest_harness as harness
from dashboard.backend.infrastructure.llm import http_policy, providers
from dashboard.backend.infrastructure.llm.backtest_harness import (
    extract_response_text,
    request_trading_decision,
)
from dashboard.backend.infrastructure.llm.pipeline_runner import truncation_reason
from dashboard.backend.infrastructure.llm.providers import commonstack
from dashboard.backend.infrastructure.llm.reasoning_controls import (
    UnsupportedReasoningEffort,
    is_reasoning_off,
)

_CONFIG = Path(__file__).resolve().parents[4] / "config" / "leaderboard.json"
_KEY = "sk-fake-commonstack-thinking-off"


def _completion(content, finish="stop", prompt_tokens=120, completion_tokens=40):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "deepseek/deepseek-v4-pro",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class _Wire:
    """Every request the SDK sent, and the scripted answers it got back."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.constructed: list[dict] = []
        self.sleeps: list[float] = []
        self.answers: list = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.answers.pop(0) if self.answers else _completion('{"actions": []}')
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, httpx.Response):
            return answer
        return httpx.Response(200, json=answer)

    def body(self, index=0) -> dict:
        return json.loads(self.requests[index].content)


@pytest.fixture
def wire(monkeypatch):
    """The real SDK, constructed by ``commonstack`` itself, on a mock transport."""
    monkeypatch.setenv("COMMONSTACK_API_KEY", _KEY)
    monkeypatch.delenv("COMMONSTACK_BASE_URL", raising=False)
    recorder = _Wire()
    real_openai = openai.OpenAI

    class _WiredOpenAI(real_openai):
        def __init__(self, **kwargs):
            recorder.constructed.append(kwargs)
            super().__init__(
                http_client=httpx.Client(transport=httpx.MockTransport(recorder.handle)),
                **kwargs,
            )

    monkeypatch.setattr(openai, "OpenAI", _WiredOpenAI)
    monkeypatch.setattr(commonstack, "_retry_sleep", recorder.sleeps.append)
    return recorder


def _client():
    return commonstack.make_client(object, reasoning_effort="none")


# --- Which client an effort selects ------------------------------------------


@pytest.mark.parametrize("effort", ["none", "off", "disabled", " NONE "])
def test_an_off_effort_selects_the_chat_completions_client(wire, effort):
    client = commonstack.make_client(object, reasoning_effort=effort)
    assert isinstance(client, commonstack.ChatCompletionsClient)


@pytest.mark.parametrize("effort", [None, "", "auto", "default"])
def test_no_effort_or_a_passthrough_keeps_the_messages_client(monkeypatch, effort):
    """Every caller that passes no effort keeps exactly the client it had."""
    monkeypatch.setenv("COMMONSTACK_API_KEY", _KEY)
    built = []

    def anthropic_cls(**kwargs):
        built.append(kwargs)
        return "anthropic-client"

    assert commonstack.make_client(anthropic_cls, reasoning_effort=effort) == (
        "anthropic-client"
    )
    assert built == [{"api_key": _KEY, "base_url": commonstack.base_url()}]


@pytest.mark.parametrize("key_set", [True, False])
@pytest.mark.parametrize("effort", ["low", "high", "medium"])
def test_a_graduated_effort_is_refused_not_dropped(monkeypatch, effort, key_set):
    """CommonStack honours on/off only. Recording an effort it never sent
    would label the run with a setting that did not run."""
    if key_set:
        monkeypatch.setenv("COMMONSTACK_API_KEY", _KEY)
    else:
        monkeypatch.delenv("COMMONSTACK_API_KEY", raising=False)
    with pytest.raises(UnsupportedReasoningEffort, match="on/off only"):
        commonstack.make_client(object, reasoning_effort=effort)


def test_no_key_builds_no_client(monkeypatch):
    monkeypatch.delenv("COMMONSTACK_API_KEY", raising=False)
    assert commonstack.make_client(object, reasoning_effort="none") is None


def test_the_dispatcher_hands_commonstack_the_effort(monkeypatch):
    """The effort has to reach commonstack.make_client to mean anything."""
    if not providers.HAS_ANTHROPIC:
        pytest.skip("anthropic SDK not installed")
    monkeypatch.setenv("COMMONSTACK_API_KEY", _KEY)
    seen = {}

    def make_client(anthropic_cls, *, reasoning_effort=None):
        seen["effort"] = reasoning_effort
        return "client"

    monkeypatch.setattr(commonstack, "make_client", make_client)
    assert providers.make_llm_client("commonstack", reasoning_effort="none") == "client"
    assert seen == {"effort": "none"}


# --- What reaches the wire -----------------------------------------------------


def test_the_request_disables_thinking_on_the_wire(wire):
    request_trading_decision(
        _client(), prompt="decide", model="deepseek/deepseek-v4-pro", temperature=0
    )
    (request,) = wire.requests
    assert str(request.url) == commonstack.chat_base_url() + "/chat/completions"
    body = wire.body()
    assert body["thinking"] == {"type": "disabled"}
    assert "reasoning" not in body
    assert body["model"] == "deepseek/deepseek-v4-pro"
    assert body["temperature"] == 0
    assert body["max_tokens"] == harness.DEFAULT_MAX_OUTPUT_TOKENS
    # The harness's system prompt rides as the first chat message.
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][0]["content"] == harness.system_prompt_for_market(None)
    assert body["messages"][1] == {"role": "user", "content": "decide"}


def test_temperature_is_omitted_when_the_entry_sets_none(wire):
    request_trading_decision(_client(), prompt="decide", model="m")
    assert "temperature" not in wire.body()


def test_openai_account_headers_never_reach_the_gateway(wire, monkeypatch):
    """The SDK reads these from the environment and sends them on every call."""
    monkeypatch.setenv("OPENAI_ORG_ID", "org-must-not-leak")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-must-not-leak")
    request_trading_decision(_client(), prompt="decide", model="m")
    headers = wire.requests[0].headers
    assert "openai-organization" not in headers
    assert "openai-project" not in headers
    assert headers["authorization"] == f"Bearer {_KEY}"


def test_the_sdk_never_retries_and_reads_the_timeout_setting(wire, monkeypatch):
    # A value no default produces, so a hard-coded timeout would fail here.
    monkeypatch.setattr(http_policy, "PROVIDER_READ_TIMEOUT_SECONDS", 77)
    request_trading_decision(_client(), prompt="decide", model="m")
    (kwargs,) = wire.constructed
    assert kwargs["max_retries"] == 0
    assert kwargs["timeout"] == http_policy.provider_http_timeout()
    assert wire.requests[0].extensions["timeout"]["read"] == 77


def test_a_read_timeout_is_one_generation_not_three(wire):
    """SDK defaults replayed a stalled call twice, each one a billed generation."""
    wire.answers = [httpx.ReadTimeout("stalled")]
    with pytest.raises(openai.APITimeoutError):
        request_trading_decision(_client(), prompt="decide", model="m")
    assert len(wire.requests) == 1
    assert wire.sleeps == []


def test_a_fast_refusal_is_repeated_by_the_retry_owner(wire):
    """max_retries=0 alone would fail a 503 that generated nothing."""
    wire.answers = [
        httpx.Response(503, json={"error": {"message": "busy"}}),
        _completion('{"actions": []}'),
    ]
    response = request_trading_decision(_client(), prompt="decide", model="m")
    assert extract_response_text(response) == '{"actions": []}'
    assert len(wire.requests) == 2
    assert wire.sleeps == [http_policy.SAME_PROVIDER_BACKOFF_SECONDS[0]]


def test_an_anthropic_only_argument_fails_loudly(wire):
    """Dropping an unknown argument would change the request silently."""
    with pytest.raises(TypeError):
        _client().messages.create(
            model="m", max_tokens=10, messages=[], thinking={"type": "enabled"}
        )
    assert wire.requests == []


# --- How the reply reads -------------------------------------------------------


def test_a_reply_reads_like_an_anthropic_message(wire):
    response = request_trading_decision(_client(), prompt="decide", model="m")
    assert extract_response_text(response) == '{"actions": []}'
    assert response.usage.input_tokens == 120
    assert response.usage.output_tokens == 40
    assert response.stop_reason == "stop"


@pytest.mark.parametrize("content", ["", "   ", None])
def test_an_empty_reply_takes_the_harness_empty_reply_path(wire, content):
    """The empty-reply retry keys on this exact error; keep it reachable."""
    wire.answers = [_completion(content)]
    response = request_trading_decision(_client(), prompt="decide", model="m")
    with pytest.raises(AttributeError, match="No text content"):
        extract_response_text(response)


def test_a_reply_cut_at_the_ceiling_is_recognised_as_truncated(wire):
    wire.answers = [_completion('{"actions": [', finish="length", completion_tokens=7)]
    response = request_trading_decision(_client(), prompt="decide", model="m")
    assert response.stop_reason == "max_tokens"
    assert truncation_reason(response, 7, '{"actions": [') is not None


def test_list_shaped_content_is_joined_not_read_as_empty():
    response = commonstack._as_anthropic_response(
        {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": '{"actions": '},
                            {"type": "text", "text": "[]}"},
                        ]
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
    )
    # Parts are stripped then joined, exactly as the execution adapter reads them.
    assert json.loads(extract_response_text(response)) == {"actions": []}
    assert (response.usage.input_tokens, response.usage.output_tokens) == (5, 3)


@pytest.mark.parametrize(
    "body",
    [
        {"error": {"message": "upstream exploded", "code": 500}},
        {"choices": []},
        {"choices": None},
    ],
)
def test_a_body_with_no_choice_is_an_error_not_an_empty_reply(body):
    """An empty reply is retried and billed as a model turn; this is neither."""
    with pytest.raises(commonstack.ChatCompletionsResponseError):
        commonstack._as_anthropic_response(body)


def test_a_gateway_error_costs_one_logged_step_not_five_billed_calls(wire, capsys):
    """Read as an empty reply, a broken body ran four retries and the rescue
    call, billed each as an LLM call, and only then fell back."""
    wire.answers = [httpx.Response(200, json={"error": {"message": "x"}})] * 6
    manager = PortfolioManager(initial_capital=1000.0)
    state = {
        "timestamp": datetime(2026, 4, 1, 15, 30),
        "cash": 1000.0,
        "positions": [],
        "positions_value": 0.0,
        "total_equity": 1000.0,
        "market_signals": {},
    }
    manager.make_trading_decision_with_llm(state, _client(), model="m")
    assert len(wire.requests) == 1
    assert manager.llm_calls == 0
    assert manager.llm_decisions == 0
    assert "returned no choices" in capsys.readouterr().out


@pytest.mark.parametrize(
    "configured, expected",
    [
        (None, "https://api.commonstack.ai/v1"),
        ("https://api.commonstack.ai/", "https://api.commonstack.ai/v1"),
        ("https://api.commonstack.ai/v1", "https://api.commonstack.ai/v1"),
        ("https://api.commonstack.ai/v1/", "https://api.commonstack.ai/v1"),
    ],
)
def test_the_chat_base_carries_one_v1(monkeypatch, configured, expected):
    if configured is None:
        monkeypatch.delenv("COMMONSTACK_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("COMMONSTACK_BASE_URL", configured)
    assert commonstack.chat_base_url() == expected


@pytest.mark.parametrize("entry_id", ["deepseek_v4_pro", "qwen3_7_plus"])
def test_the_thinking_models_run_thinking_off_at_temperature_zero(entry_id):
    """Same pin the dashboard catalog applies (PINNED_NO_THINKING)."""
    config = json.loads(_CONFIG.read_text(encoding="utf-8"))
    entry = next(e for e in config["strategies"] if e["id"] == entry_id)
    assert entry["integration"] == "commonstack"
    assert is_reasoning_off(entry["reasoning_effort"])
    assert entry["temperature"] == 0
