"""Reasoning-effort spellings shared by every client that reads one.

One vocabulary and one predicate, imported by the legacy OpenRouter and
CommonStack clients (``providers/openrouter.py``, ``providers/commonstack.py``)
and the execution adapter (``execution/adapters/openai.py``), so they cannot
disagree about whether an effort string turns thinking off. The results panel
keeps a JavaScript copy of the off set (``formatBacktestSampling`` in
``frontend/app.js``), held equal to this one by
``tests/test_backtest_sampling_row.py``.
"""

from __future__ import annotations

from typing import Any, Optional

REASONING_OFF_VALUES = frozenset({"none", "off", "false", "0", "disabled"})
# "Send no reasoning control at all": the provider's own default applies.
REASONING_PASSTHROUGH_VALUES = frozenset({"auto", "default"})


def normalize_reasoning_effort(value: Optional[Any]) -> str:
    """An effort as every client compares it: stripped, lowercased, ``""`` for None."""
    return "" if value is None else str(value).strip().lower()


def is_reasoning_off(value: Optional[Any]) -> bool:
    """True when ``value`` turns thinking off. ``None`` and blank are not off."""
    return normalize_reasoning_effort(value) in REASONING_OFF_VALUES


class UnsupportedReasoningEffort(ValueError):
    """A configured effort that the chosen client cannot put on the wire.

    Raised rather than dropped. ``_llm_run_metadata`` records the configured
    effort as what was sent, and the results panel reads it back that way. An
    effort the client quietly ignored would publish a setting that never ran.
    """


def wants_thinking_off(value: Optional[Any], *, integration: str) -> bool:
    """For a client whose only reasoning control is on/off: is it asking for off?

    ``None``, blank and passthrough values mean "send nothing", so the
    provider default applies and the answer is False. An off value is True.
    Any graduated effort (``low``, ``high``, ...) raises: this client has no
    way to send it.
    """
    effort = normalize_reasoning_effort(value)
    if not effort or effort in REASONING_PASSTHROUGH_VALUES:
        return False
    if effort in REASONING_OFF_VALUES:
        return True
    raise UnsupportedReasoningEffort(
        f"reasoning_effort={value!r} cannot be applied on integration "
        f"{integration!r}: it honours thinking on/off only. Use one of "
        f"{sorted(REASONING_OFF_VALUES)} to turn thinking off, or drop the "
        "field to keep the provider default."
    )


def thinking_disabled_body() -> dict[str, Any]:
    """The one thinking control CommonStack honours: chat completions only.

    CommonStack ignores every graduated reasoning control for DeepSeek V4 Pro
    and Qwen3.7 Plus (``reasoning.effort``, ``reasoning.enabled=false``,
    ``thinking.budget_tokens``: #539, 2026-10-01), and its ``/v1/messages``
    surface ignores this one too (2026-10-02). Sent as ``extra_body`` by both
    the execution adapter and the legacy harness's CommonStack client; a fresh
    dict per call, so neither can mutate the other's request.
    """
    return {"thinking": {"type": "disabled"}}


__all__ = [
    "REASONING_OFF_VALUES",
    "REASONING_PASSTHROUGH_VALUES",
    "UnsupportedReasoningEffort",
    "is_reasoning_off",
    "normalize_reasoning_effort",
    "thinking_disabled_body",
    "wants_thinking_off",
]
