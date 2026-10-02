"""CommonStack gateway integration (default multi-provider path).

CommonStack (https://commonstack.ai) exposes Anthropic, DeepSeek, Qwen, etc.
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

The wire body and the response reading are shared with the billed execution
adapter through two leaves (``reasoning_controls``, ``chat_completions``),
never by importing that layer, which would drag its registry, handoff and
credential code into every backtest process.
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace
from typing import Any, Optional

from dashboard.backend.infrastructure.llm.chat_completions import (
    first_choice,
    normalize_finish_reason,
    response_text,
    usage_counts,
    value_at,
)
from dashboard.backend.infrastructure.llm.http_policy import (
    SDK_MAX_RETRIES,
    call_with_retries,
    provider_http_timeout,
)
from dashboard.backend.infrastructure.llm.reasoning_controls import (
    thinking_disabled_body,
    wants_thinking_off,
)

INTEGRATION_ID = "commonstack"
# Prefer DeepSeek over Anthropic slugs: CommonStack's Anthropic provider has
# been observed returning a canned "Hi! How can I help you today?" with
# ~10 input_tokens while ignoring the request body (breaks Discord /strategy
# and default LLM backtests). DeepSeek stays reliable on the same key.
DEFAULT_MODEL = "deepseek/deepseek-v4-pro"
DEFAULT_BASE_URL = "https://api.commonstack.ai"

# Monkeypatched by tests; looked up per call, so a patch takes effect.
_retry_sleep = time.sleep


def base_url() -> str:
    return os.getenv("COMMONSTACK_BASE_URL", DEFAULT_BASE_URL)


def default_model_name() -> str:
    return DEFAULT_MODEL


def chat_base_url() -> str:
    """OpenAI-SDK base for the same host: ``base_url()`` plus ``/v1``, once."""
    root = base_url().rstrip("/")
    return root if root.endswith("/v1") else root + "/v1"


class ChatCompletionsResponseError(RuntimeError):
    """A 200 from chat completions that carried no choice at all.

    Not an empty reply: an empty reply is a choice with no text, which the
    harness retries and bills as a model turn. A body with no ``choices`` is a
    gateway error or a contract change. Raising sends the step through the
    harness's outer handler (one logged rule-based step) instead of five
    billed retries that all read the same broken body as "the model said
    nothing".
    """


def _system_text(system: Any) -> Optional[str]:
    if system is None or isinstance(system, str):
        return system or None
    # Anthropic also takes a list of text blocks.
    parts = [value_at(block, "text", "") for block in system]
    return "\n".join(p for p in parts if p) or None


def _as_anthropic_response(response: Any) -> SimpleNamespace:
    """Anthropic ``Message`` shape from a chat-completions response.

    A choice with no text becomes an empty ``content`` list, so
    ``extract_response_text`` raises its usual "No text content" error and the
    harness's empty-reply retry runs unchanged. ``stop_reason`` uses the
    execution layer's vendor-neutral tag, which is what the harness already
    reads on that path (``max_tokens`` at the ceiling).
    """
    choice = first_choice(response)
    if choice is None:
        raise ChatCompletionsResponseError(
            "CommonStack chat completions returned no choices "
            f"(response type {type(response).__name__}; "
            f"error field present: {value_at(response, 'error') is not None})"
        )
    text = response_text(response)
    counts = usage_counts(response) or (0, 0)
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)] if text else [],
        stop_reason=normalize_finish_reason(value_at(choice, "finish_reason")),
        usage=SimpleNamespace(
            input_tokens=int(counts[0] or 0),
            output_tokens=int(counts[1] or 0),
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
            "extra_body": thinking_disabled_body(),
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        # The SDK runs with max_retries=0, so this owns the retries: only a
        # failure that generated nothing is repeated, never a read timeout.
        response = call_with_retries(
            lambda: self._client.chat.completions.create(**kwargs),
            label="CommonStack chat completions",
            sleep=_retry_sleep,
        )
        return _as_anthropic_response(response)


class ChatCompletionsClient:
    """``messages.create`` on CommonStack's chat-completions surface, thinking off.

    Only ``messages.create`` exists: the backtest harness is the one caller
    that passes an effort, and it uses nothing else.
    """

    def __init__(self, openai_client: Any):
        self.messages = _ChatCompletionsMessages(openai_client)


def _chat_client(key: str) -> Any:
    from openai import OpenAI

    # SDK_MAX_RETRIES (0) and an explicit timeout, as every other SDK client
    # here: the SDK's own retry loop replays a read timeout with no
    # idempotency key, so one stall became three billed generations
    # (http_policy.py). The /v1/messages client below predates that policy.
    client = OpenAI(
        api_key=key,
        base_url=chat_base_url(),
        max_retries=SDK_MAX_RETRIES,
        timeout=provider_http_timeout(),
    )
    # The SDK fills these from OPENAI_ORG_ID / OPENAI_PROJECT_ID and sends them
    # as headers on every request. They belong to the OpenAI account, never to
    # a third-party gateway. Passing None to the constructor does not help,
    # because None means "read the environment".
    client.organization = None
    client.project = None
    return client


def make_client(
    anthropic_cls: Any,
    *,
    reasoning_effort: Optional[str] = None,
) -> Optional[Any]:
    """Build an Anthropic-compatible client for CommonStack, or ``None``.

    With thinking off the client speaks chat completions (see the module
    docstring). With no effort, or a passthrough one, it is the native
    ``/v1/messages`` client, unchanged. A graduated effort raises
    ``UnsupportedReasoningEffort``: CommonStack honours on/off only, and a run
    must not record an effort that never reached the wire.
    """
    thinking_off = wants_thinking_off(reasoning_effort, integration=INTEGRATION_ID)
    key = os.getenv("COMMONSTACK_API_KEY")
    if not key:
        return None
    if thinking_off:
        try:
            client = _chat_client(key)
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
