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
    BillingMode,
    LLMExecutionRequest,
    LLMExecutionResult,
    LLMMessage,
    LLMRunEvidence,
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
        self._completed_results: list[LLMExecutionResult] = []
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
                provider_ids=self.handoff.provider_ids,
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
        # Production services return the complete typed result. Older test and
        # extension doubles return only the SDK-compatible attributes; keep
        # those callers working, but do not invent billing evidence for them.
        if isinstance(result, LLMExecutionResult):
            self._completed_results.append(result)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=result.text)],
            model=result.model_id,
            usage=SimpleNamespace(
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
            ),
            # Same attribute the raw Anthropic SDK exposes, so callers can
            # tell a reply cut at the output ceiling from a malformed one
            # without knowing which client built the response.
            stop_reason=getattr(result, "finish_reason", None),
        )

    def execution_summary(self) -> LLMRunEvidence | None:
        """Return safe, backend-authoritative evidence for completed calls."""

        if not self._completed_results:
            return None

        for result in self._completed_results:
            if (
                result.billing.billing_source,
                result.model_id,
            ) != (self.handoff.billing_mode, self.handoff.model_id):
                raise RuntimeError("inconsistent LLM execution identity")

        requested_provider_ids = {
            result.requested_provider_id or self.handoff.provider_id
            for result in self._completed_results
        }
        if requested_provider_ids != {self.handoff.provider_id}:
            raise RuntimeError("inconsistent requested LLM provider identity")

        provider_ids = tuple(
            dict.fromkeys(result.provider_id for result in self._completed_results)
        )
        provider_mixed = len(provider_ids) > 1
        provider_id = "mixed" if provider_mixed else provider_ids[0]

        first = self._completed_results[0]

        snapshots = [
            result.billing.pricing_snapshot for result in self._completed_results
        ]
        pricing_snapshot = (
            None
            if provider_mixed
            else (
                snapshots[0]
                if all(snapshot == snapshots[0] for snapshot in snapshots[1:])
                else None
            )
        )

        usage_available = all(
            result.usage.usage_available for result in self._completed_results
        )
        provider_costs = [
            result.billing.provider_cost_usd for result in self._completed_results
        ]
        estimated_costs = [
            result.billing.estimated_cost_usd for result in self._completed_results
        ]
        outstanding = sum(
            result.billing.outstanding_credits_micro
            for result in self._completed_results
        )
        credential_ids = {result.credential_id for result in self._completed_results}
        credential_last_fours = {
            result.credential_key_last_four for result in self._completed_results
        }
        credential_id = (
            None
            if provider_mixed or len(credential_ids) != 1
            else credential_ids.pop()
        )
        credential_key_last_four = (
            None
            if provider_mixed or len(credential_last_fours) != 1
            else credential_last_fours.pop()
        )

        if not usage_available:
            outcome = "unavailable"
        elif self.handoff.billing_mode is BillingMode.BYOK:
            outcome = "byok"
        elif outstanding:
            outcome = "settled_overage"
        else:
            outcome = "settled"

        return LLMRunEvidence(
            billing_mode=self.handoff.billing_mode,
            provider_id=provider_id,
            requested_provider_id=self.handoff.provider_id,
            provider_ids=provider_ids,
            provider_mixed=provider_mixed,
            model_id=first.model_id,
            credential_id=credential_id,
            credential_key_last_four=credential_key_last_four,
            call_count=len(self._completed_results),
            input_tokens=sum(
                result.usage.input_tokens for result in self._completed_results
            ),
            output_tokens=sum(
                result.usage.output_tokens for result in self._completed_results
            ),
            usage_available=usage_available,
            provider_cost_usd=(
                round(
                    sum(float(value) for value in provider_costs if value is not None),
                    6,
                )
                if all(value is not None for value in provider_costs)
                else None
            ),
            estimated_cost_usd=(
                round(
                    sum(float(value) for value in estimated_costs if value is not None),
                    6,
                )
                if all(value is not None for value in estimated_costs)
                else None
            ),
            pricing_snapshot=pricing_snapshot,
            debited_credits_micro=sum(
                result.billing.debited_credits_micro
                for result in self._completed_results
            ),
            outstanding_credits_micro=outstanding,
            outcome=outcome,
        )

    def close(self) -> None:
        """Match SDK clients; provider connections are owned by each adapter."""


__all__ = ["AnthropicCompatibleExecutionClient"]
