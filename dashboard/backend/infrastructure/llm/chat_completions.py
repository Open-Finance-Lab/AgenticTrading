"""Reading an OpenAI-shaped chat-completions response.

A leaf (standard library only) so the billed execution adapter
(``execution/adapters/openai.py``) and the legacy harness's CommonStack
thinking-off client (``providers/commonstack.py``) parse one wire shape one
way. The legacy client must not import the execution layer itself: that drags
its registry, handoff and credential code into every backtest process.
"""

from __future__ import annotations

from typing import Any

# Provider spellings of "the reply stopped at the output ceiling", folded to
# one value so callers above the adapters never see the vendor vocabulary.
_OUTPUT_CEILING_FINISH_REASONS = frozenset({"length", "max_tokens"})
FINISH_REASON_MAX_TOKENS = "max_tokens"
# ``LLMExecutionResult.finish_reason`` is bounded; an OpenAI-compatible
# provider may put anything in this field, and a long value must not turn a
# successful call into ``response_invalid`` when the result model rejects it.
_FINISH_REASON_MAX_LENGTH = 32


def value_at(value: Any, name: str, default: Any = None) -> Any:
    """``value[name]`` for a dict, ``value.name`` for an SDK object."""
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def normalize_finish_reason(value: Any) -> str | None:
    """Fold a provider stop/finish reason into a lowercase, vendor-neutral tag.

    ``length`` (OpenAI / OpenRouter), ``MAX_TOKENS`` (Gemini) and
    ``max_tokens`` (Anthropic) all become ``"max_tokens"``; any other string is
    passed through lowercased (and clamped to the result model's length bound)
    so it stays inspectable; anything else is ``None``.
    """
    if not isinstance(value, str):
        return None
    reason = value.strip().lower()
    if not reason:
        return None
    if reason in _OUTPUT_CEILING_FINISH_REASONS:
        return FINISH_REASON_MAX_TOKENS
    return reason[:_FINISH_REASON_MAX_LENGTH]


def first_choice(response: Any) -> Any:
    """The first choice, or ``None`` when the response carries none.

    ``None`` is not an empty reply: a body with no ``choices`` is a gateway
    error or a contract change, and callers must not bill it as a model turn.
    """
    choices = value_at(response, "choices", None)
    if not isinstance(choices, (list, tuple)) or not choices:
        return None
    return choices[0]


def response_text(response: Any) -> str:
    """The first choice's text, joined from parts when the content is a list."""
    message = value_at(first_choice(response), "message")
    content = value_at(message, "content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for block in content:
            text = value_at(block, "text")
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
        return "".join(chunks).strip()
    return ""


def usage_counts(response: Any) -> tuple[Any, Any] | None:
    """Raw ``(input, output)`` token counts, or ``None`` with no usage object.

    Values are returned as the provider sent them; each caller applies its
    own validation (``usage_from_fields`` in the adapter, ``int(... or 0)`` in
    the legacy harness, which has always read a missing count as zero).
    """
    usage = value_at(response, "usage")
    if usage is None:
        return None
    return (
        value_at(usage, "prompt_tokens", value_at(usage, "input_tokens")),
        value_at(usage, "completion_tokens", value_at(usage, "output_tokens")),
    )


__all__ = [
    "FINISH_REASON_MAX_TOKENS",
    "first_choice",
    "normalize_finish_reason",
    "response_text",
    "usage_counts",
    "value_at",
]
