"""CommonStack thinking-off for the leaderboard's legacy harness path.

CommonStack's ``/v1/messages`` ignores every thinking control for DeepSeek V4
Pro and Qwen3.7 Plus; only ``/v1/chat/completions`` honours
``thinking: {type: "disabled"}`` (2026-10-02 probe). An entry that turns
reasoning off is therefore served on chat completions, with the response
reshaped so the harness reads it exactly as it reads an Anthropic reply.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dashboard.backend.infrastructure.llm import providers
from dashboard.backend.infrastructure.llm.backtest_harness import (
    extract_response_text,
    request_trading_decision,
)
from dashboard.backend.infrastructure.llm.pipeline_runner import truncation_reason
from dashboard.backend.infrastructure.llm.providers import commonstack

_CONFIG = Path(__file__).resolve().parents[4] / "config" / "leaderboard.json"


class _FakeOpenAI:
    """Records constructor args and every chat.completions.create call."""

    instances: list = []

    def __init__(self, reply=None, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self._reply = reply if reply is not None else _reply("{\"actions\": []}")
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        _FakeOpenAI.instances.append(self)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._reply


def _reply(content, finish="stop", prompt_tokens=120, completion_tokens=40):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason=finish
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
    )


def _bridge(monkeypatch, reply=None):
    monkeypatch.setenv("COMMONSTACK_API_KEY", "test-key")
    fake = {}

    def factory(**kwargs):
        fake["client"] = _FakeOpenAI(reply=reply, **kwargs)
        return fake["client"]

    client = commonstack.make_client(
        object, reasoning_effort="none", openai_cls=factory
    )
    return client, fake["client"]


@pytest.mark.parametrize("effort", ["none", "off", "disabled", " NONE "])
def test_an_off_effort_selects_the_chat_completions_client(monkeypatch, effort):
    monkeypatch.setenv("COMMONSTACK_API_KEY", "test-key")
    client = commonstack.make_client(
        object, reasoning_effort=effort, openai_cls=lambda **kw: _FakeOpenAI(**kw)
    )
    assert isinstance(client, commonstack.ChatCompletionsClient)


@pytest.mark.parametrize("effort", [None, "low", "high", "default"])
def test_any_other_effort_keeps_the_messages_client(monkeypatch, effort):
    """Every caller that passes no effort keeps exactly the client it had."""
    monkeypatch.setenv("COMMONSTACK_API_KEY", "test-key")
    built = []

    def anthropic_cls(**kwargs):
        built.append(kwargs)
        return "anthropic-client"

    def openai_cls(**kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("chat completions used without thinking off")

    client = commonstack.make_client(
        anthropic_cls, reasoning_effort=effort, openai_cls=openai_cls
    )
    assert client == "anthropic-client"
    assert built == [{"api_key": "test-key", "base_url": commonstack.base_url()}]


def test_no_key_builds_no_client(monkeypatch):
    monkeypatch.delenv("COMMONSTACK_API_KEY", raising=False)
    assert commonstack.make_client(object, reasoning_effort="none") is None


def test_the_request_disables_thinking_on_the_wire(monkeypatch):
    client, fake = _bridge(monkeypatch)
    request_trading_decision(
        client,
        prompt="decide",
        model="deepseek/deepseek-v4-pro",
        temperature=0,
    )
    (call,) = fake.calls
    assert call["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning" not in call and "reasoning" not in call["extra_body"]
    assert call["model"] == "deepseek/deepseek-v4-pro"
    assert call["temperature"] == 0
    assert call["max_tokens"] == 2000
    # The harness's system prompt rides as the first chat message.
    assert [m["role"] for m in call["messages"]] == ["system", "user"]
    assert call["messages"][0]["content"].startswith("You are an expert")
    assert call["messages"][1] == {"role": "user", "content": "decide"}
    assert fake.kwargs["base_url"] == "https://api.commonstack.ai/v1"


def test_temperature_is_omitted_when_the_entry_sets_none(monkeypatch):
    client, fake = _bridge(monkeypatch)
    request_trading_decision(client, prompt="decide", model="m")
    assert "temperature" not in fake.calls[0]


def test_an_anthropic_only_argument_fails_loudly(monkeypatch):
    """Dropping an unknown argument would change the request silently."""
    client, _ = _bridge(monkeypatch)
    with pytest.raises(TypeError):
        client.messages.create(
            model="m", max_tokens=10, messages=[], thinking={"type": "enabled"}
        )


def test_a_reply_reads_like_an_anthropic_message(monkeypatch):
    client, _ = _bridge(monkeypatch, reply=_reply('{"actions": []}'))
    response = request_trading_decision(client, prompt="decide", model="m")
    assert extract_response_text(response) == '{"actions": []}'
    assert response.usage.input_tokens == 120
    assert response.usage.output_tokens == 40
    assert response.stop_reason == "end_turn"


@pytest.mark.parametrize("content", ["", "   ", None])
def test_an_empty_reply_takes_the_harness_empty_reply_path(monkeypatch, content):
    """The empty-reply retry keys on this exact error; keep it reachable."""
    client, _ = _bridge(monkeypatch, reply=_reply(content))
    response = request_trading_decision(client, prompt="decide", model="m")
    with pytest.raises(AttributeError, match="No text content"):
        extract_response_text(response)


def test_a_reply_cut_at_the_ceiling_is_recognised_as_truncated(monkeypatch):
    client, _ = _bridge(
        monkeypatch, reply=_reply('{"actions": [', finish="length", completion_tokens=7)
    )
    response = request_trading_decision(client, prompt="decide", model="m")
    assert response.stop_reason == "max_tokens"
    assert truncation_reason(response, 7, '{"actions": [') is not None


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


def test_the_dispatcher_hands_commonstack_the_effort(monkeypatch):
    """The effort has to reach commonstack.make_client to mean anything."""
    if not providers.HAS_ANTHROPIC:
        pytest.skip("anthropic SDK not installed")
    monkeypatch.setenv("COMMONSTACK_API_KEY", "test-key")
    seen = {}

    def make_client(anthropic_cls, *, reasoning_effort=None):
        seen["effort"] = reasoning_effort
        return "client"

    monkeypatch.setattr(commonstack, "make_client", make_client)
    assert providers.make_llm_client("commonstack", reasoning_effort="none") == "client"
    assert seen == {"effort": "none"}


@pytest.mark.parametrize("entry_id", ["deepseek_v4_pro", "qwen3_7_plus"])
def test_the_thinking_models_run_thinking_off_at_temperature_zero(entry_id):
    """Same pin the dashboard catalog applies (PINNED_NO_THINKING)."""
    config = json.loads(_CONFIG.read_text(encoding="utf-8"))
    entry = next(e for e in config["strategies"] if e["id"] == entry_id)
    assert entry["integration"] == "commonstack"
    assert commonstack.thinking_is_off(entry["reasoning_effort"])
    assert entry["temperature"] == 0
