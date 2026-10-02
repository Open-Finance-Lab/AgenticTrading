"""CommonStack gateway integration.

CommonStack exposes OpenAI / Google / xAI / DeepSeek / Qwen / Anthropic models
behind one key on an Anthropic-compatible ``/v1/messages`` surface. Responses
keep Anthropic shape (``content[0].text`` + ``usage.{input,output}_tokens``),
so the shared backtest harness needs only a different ``base_url`` and a
``provider/model`` slug.

Thinking off is the exception. ``/v1/messages`` ignores every thinking control
for DeepSeek V4 Pro and Qwen3.7 Plus -- ``thinking: {type: "disabled"}``,
``enable_thinking``, ``chat_template_kwargs`` and ``reasoning.enabled`` all
left Qwen reasoning ~1k tokens for a ~100-token answer in the 2026-10-02 probe
-- while ``/v1/chat/completions`` honours ``thinking: {type: "disabled"}``
(61 tokens, 5s). Left on, DeepSeek's per-call coin flip into reasoning emptied
13 of 132 leaderboard steps even after the 4096-token rescue call, which fails
the H6 guard. So an entry that turns reasoning off is served by
``ChatCompletionsClient``, which speaks chat completions on the wire and hands
the harness an Anthropic-shaped response.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Optional

from dashboard.backend.infrastructure.llm.reasoning_controls import (
    REASONING_OFF_VALUES,
)

INTEGRATION_ID = "commonstack"
# Prefer DeepSeek over Anthropic slugs: CommonStack's Anthropic provider has
# been observed returning a canned "Hi! How can I help you today?" with
# ~10 input_tokens while ignoring the request body (breaks Discord /strategy
# and default LLM backtests). DeepSeek stays reliable on the same key.
DEFAULT_MODEL = "deepseek/deepseek-v4-pro"
DEFAULT_BASE_URL = "https://api.commonstack.ai"


def base_url() -> str:
    return os.getenv("COMMONSTACK_BASE_URL", DEFAULT_BASE_URL)


def default_model_name() -> str:
    return DEFAULT_MODEL


def chat_base_url() -> str:
    """OpenAI-SDK base for the same host: ``base_url()`` plus ``/v1``, once."""
    root = base_url().rstrip("/")
    return root if root.endswith("/v1") else root + "/v1"


def thinking_is_off(reasoning_effort: Optional[str]) -> bool:
    """True when a config-supplied effort turns thinking off.

    ``None`` keeps the provider default: the native ``/v1/messages`` client,
    exactly as every caller that passes no effort has always had.
    """
    if reasoning_effort is None:
        return False
    return str(reasoning_effort).strip().lower() in REASONING_OFF_VALUES


# Chat-completions finish reasons, in the Anthropic spelling the harness reads
# (``_hit_output_ceiling`` accepts both, but one vocabulary is less to audit).
_STOP_REASONS = {"length": "max_tokens", "stop": "end_turn"}


def _system_text(system: Any) -> Optional[str]:
    if system is None or isinstance(system, str):
        return system or None
    # Anthropic also takes a list of text blocks.
    parts = [
        block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
        for block in system
    ]
    return "\n".join(p for p in parts if p) or None


def _as_anthropic_response(response: Any) -> SimpleNamespace:
    """Anthropic ``Message`` shape from a chat-completions response.

    Empty or absent content becomes an empty ``content`` list, so
    ``extract_response_text`` raises its usual "No text content" error and the
    harness's empty-reply retry runs unchanged.
    """
    choices = getattr(response, "choices", None) or []
    first = choices[0] if choices else None
    message = getattr(first, "message", None)
    text = getattr(message, "content", None)
    content = (
        [SimpleNamespace(type="text", text=text)]
        if isinstance(text, str) and text.strip()
        else []
    )
    finish = getattr(first, "finish_reason", None)
    usage = getattr(response, "usage", None)
    return SimpleNamespace(
        content=content,
        stop_reason=_STOP_REASONS.get(finish, finish),
        usage=SimpleNamespace(
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        ),
    )


class _ChatCompletionsMessages:
    def __init__(self, client: Any):
        self._client = client

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list,
        system: Any = None,
        temperature: Optional[float] = None,
    ) -> SimpleNamespace:
        # Keyword-only with no **kwargs: an Anthropic-only argument a caller
        # adds later fails loudly here instead of being dropped on the wire.
        wire_messages = []
        system_text = _system_text(system)
        if system_text:
            wire_messages.append({"role": "system", "content": system_text})
        wire_messages.extend(messages)
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": wire_messages,
            # Sent instead of any ``reasoning`` field, as the execution
            # adapter does for this provider (THINKING_TOGGLE_PROVIDERS).
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        return _as_anthropic_response(self._client.chat.completions.create(**kwargs))


class ChatCompletionsClient:
    """``messages.create`` on CommonStack's chat-completions surface, thinking off.

    Only ``messages.create`` exists: the backtest harness is the one caller
    that passes an effort, and it uses nothing else.
    """

    def __init__(self, openai_client: Any):
        self.messages = _ChatCompletionsMessages(openai_client)


def _openai_cls() -> Any:
    from openai import OpenAI

    return OpenAI


def make_client(
    anthropic_cls: Any,
    *,
    reasoning_effort: Optional[str] = None,
    openai_cls: Optional[Any] = None,
) -> Optional[Any]:
    """Build an Anthropic-compatible client for CommonStack, or ``None``.

    With thinking off the client speaks chat completions (see the module
    docstring); otherwise the native ``/v1/messages`` client, unchanged.
    """
    key = os.getenv("COMMONSTACK_API_KEY")
    if not key:
        return None
    if thinking_is_off(reasoning_effort):
        try:
            # SDK defaults (retries, timeout) on purpose: the Anthropic client
            # this replaces runs on its defaults too, so the only thing this
            # changes is the surface the request reaches.
            client = (openai_cls or _openai_cls())(
                api_key=key, base_url=chat_base_url()
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"⚠️  Failed to init CommonStack chat client: {exc}")
            return None
        print("ℹ️  CommonStack: thinking disabled (chat completions)")
        return ChatCompletionsClient(client)
    try:
        return anthropic_cls(api_key=key, base_url=base_url())
    except Exception as exc:  # pragma: no cover - defensive
        print(f"⚠️  Failed to init CommonStack client: {exc}")
        return None
