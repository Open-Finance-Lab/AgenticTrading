"""Anthropic-shaped compatibility client backed by the unified execution service."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

from dashboard.backend.infrastructure.llm.execution.errors import (
    ExecutionErrorCategory,
    LLMExecutionError,
)
from dashboard.backend.infrastructure.llm.execution.handoff import ExecutionHandoff
from dashboard.backend.infrastructure.llm.execution.models import (
    LLMExecutionRequest,
    LLMMessage,
    UsagePolicy,
)
from dashboard.backend.infrastructure.llm.execution.service import LLMExecutionService


def _text_content(value: Any) -> str:
    """Normalize the string content used by the existing backtest callers."""

    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        chunks: list[str] = []
        for block in value:
            if isinstance(block, Mapping):
                text = block.get("text")
            else:
                text = getattr(block, "text", None)
            if isinstance(text, str):
                chunks.append(text)
        return "".join(chunks)
    return ""


def _message(value: Any) -> LLMMessage:
    if isinstance(value, Mapping):
        role = value.get("role")
        content = value.get("content")
    else:
        role = getattr(value, "role", None)
        content = getattr(value, "content", None)
    text = _text_content(content)
    if role not in {"user", "assistant"} or not text.strip():
        raise LLMExecutionError(ExecutionErrorCategory.RESPONSE_INVALID)
    try:
        return LLMMessage(role=role, content=text)
    except Exception as exc:  # noqa: BLE001 - preserve the fixed public error
        raise LLMExecutionError(ExecutionErrorCategory.RESPONSE_INVALID) from exc


class _Messages:
    def __init__(self, client: "AnthropicCompatibleExecutionClient") -> None:
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create(**kwargs)


class AnthropicCompatibleExecutionClient:
    """Expose ``messages.create`` while routing each call through ATL billing."""

    def __init__(
        self,
        *,
        execution_service: LLMExecutionService,
        handoff: ExecutionHandoff,
    ) -> None:
        self.execution_service = execution_service
        self.handoff = handoff
        self.fail_closed = True
        self._next_call_index = 0
        self.messages = _Messages(self)

    def _create(self, **kwargs: Any) -> Any:
        model = kwargs.get("model")
        if not isinstance(model, str) or model.strip() != self.handoff.model_id:
            raise LLMExecutionError(ExecutionErrorCategory.RESPONSE_INVALID)

        max_tokens = kwargs.get("max_tokens")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise LLMExecutionError(ExecutionErrorCategory.RESPONSE_INVALID)

        raw_messages = kwargs.get("messages")
        if not isinstance(raw_messages, Sequence) or isinstance(
            raw_messages, (str, bytes, bytearray)
        ):
            raise LLMExecutionError(ExecutionErrorCategory.RESPONSE_INVALID)
        try:
            messages = tuple(_message(item) for item in raw_messages)
            system = kwargs.get("system")
            if system is not None and not isinstance(system, str):
                raise ValueError("system must be a string")
            temperature = kwargs.get("temperature")
            if temperature is not None and (
                isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            ):
                raise ValueError("temperature must be numeric")
            reasoning_effort = kwargs.get("reasoning_effort")
            if reasoning_effort is not None and not isinstance(reasoning_effort, str):
                raise ValueError("reasoning_effort must be a string")
            request = LLMExecutionRequest(
                user_id=self.handoff.user_id,
                run_id=self.handoff.run_id,
                call_index=self._next_call_index,
                billing_mode=self.handoff.billing_mode,
                provider_id=self.handoff.provider_id,
                model_id=self.handoff.model_id,
                system_message=system,
                messages=messages,
                usage_policy=UsagePolicy(max_output_tokens=max_tokens),
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
        except LLMExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001 - request details stay private
            raise LLMExecutionError(ExecutionErrorCategory.RESPONSE_INVALID) from exc

        # Reserve a unique id for every attempt, including a failed provider call.
        self._next_call_index += 1
        result = self.execution_service.execute(request)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=result.text)],
            model=result.model_id,
            usage=SimpleNamespace(
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
            ),
        )

    def close(self) -> None:
        """Match SDK clients; provider connections are owned by each adapter."""


__all__ = ["AnthropicCompatibleExecutionClient"]
