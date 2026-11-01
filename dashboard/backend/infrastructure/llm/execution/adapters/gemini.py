"""Gemini REST execution adapter with pinned HTTP transport."""

from __future__ import annotations

from urllib.parse import quote

from dashboard.backend.domain.model_providers.execution_catalog import (
    UnsupportedExecutionModel,
    resolve_execution_model_route,
)
from dashboard.backend.domain.model_providers.models import ProviderRecord
from dashboard.backend.infrastructure.llm.execution.models import (
    LLMExecutionRequest,
    LLMUsage,
)

from .base import (
    AdapterResponse,
    CredentialMaterial,
    ProviderExecutionError,
    build_safe_http_client,
    describe_sampling_wire,
    map_provider_error,
    normalize_finish_reason,
    optional_nonnegative_float,
    usage_from_fields,
    value_at,
)


def _endpoint(base_url: str, model_id: str) -> str:
    origin = base_url.rstrip("/")
    version = "" if origin.endswith("/v1beta") else "/v1beta"
    model = model_id.removeprefix("models/")
    return f"{origin}{version}/models/{quote(model, safe='._/-')}:generateContent"


def _token_count(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _gemini_usage(usage_data: object) -> LLMUsage | None:
    """Billed ``(input, output)`` usage from ``usageMetadata``, thinking included.

    ``candidatesTokenCount`` excludes thinking, which Google reports apart in
    ``thoughtsTokenCount`` and bills as output (#657). A reply whose thinking
    spent the ceiling often carries no ``candidatesTokenCount`` at all, so a
    missing one counts as zero rather than dropping the usage.

    Some thinking replies omit ``thoughtsTokenCount`` too. Only then is
    thinking read off ``totalTokenCount``, less the prompt and any tool-use
    prompt it also counts (cached content is already part of the prompt).
    The two optional fields only ever add: a malformed one is ignored, never
    allowed to drop an otherwise valid count.
    """
    if usage_data is None:
        return None
    prompt = value_at(usage_data, "promptTokenCount")
    candidates = value_at(usage_data, "candidatesTokenCount")
    thoughts = _token_count(value_at(usage_data, "thoughtsTokenCount"))
    total = _token_count(value_at(usage_data, "totalTokenCount"))
    if candidates is None:
        if thoughts is None and total is None:
            return None
        candidates = 0
    parsed_candidates = _token_count(candidates)
    if parsed_candidates is None:
        return None
    output = parsed_candidates + (thoughts or 0)
    parsed_prompt = _token_count(prompt)
    if thoughts is None and total is not None and parsed_prompt is not None:
        tool_prompt = _token_count(value_at(usage_data, "toolUsePromptTokenCount")) or 0
        output = max(output, total - parsed_prompt - tool_prompt)
    return usage_from_fields(prompt, output)


class GeminiExecutionAdapter:
    def complete(
        self,
        request: LLMExecutionRequest,
        credential: CredentialMaterial,
        provider: ProviderRecord,
    ) -> AdapterResponse:
        contents = [
            {
                "role": "model" if message.role == "assistant" else message.role,
                "parts": [{"text": message.content}],
            }
            for message in request.messages
            if message.role != "system"
        ]
        payload: dict[str, object] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": request.usage_policy.max_output_tokens,
            },
        }
        if request.system_message:
            payload["systemInstruction"] = {
                "parts": [{"text": request.system_message}]
            }
        generation = payload["generationConfig"]
        wire: list[str] = []
        if request.temperature is not None:
            generation["temperature"] = request.temperature  # type: ignore[index]
            wire.append(f"temperature={request.temperature}")
        client = None
        try:
            try:
                provider_model_id = resolve_execution_model_route(
                    provider,
                    request.model_id,
                ).provider_model_id
            except UnsupportedExecutionModel as exc:
                raise ProviderExecutionError("provider_unavailable") from exc
            client = build_safe_http_client(provider.approved_base_url)
            response = client.post(
                _endpoint(provider.approved_base_url, provider_model_id),
                headers={"x-goog-api-key": credential.secret, "Accept": "application/json"},
                json=payload,
            )
            if response.status_code in {401, 403}:
                raise ProviderExecutionError("credential_invalid")
            if response.status_code == 408 or response.status_code == 429 or response.status_code >= 500:
                raise ProviderExecutionError("provider_unavailable")
            if response.status_code >= 400:
                raise ProviderExecutionError("provider_unavailable")
            data = response.json()
            candidates = data.get("candidates") if isinstance(data, dict) else None
            first = candidates[0] if isinstance(candidates, list) and candidates else None
            content = value_at(first, "content")
            parts = value_at(content, "parts", ())
            text = "".join(
                str(value_at(part, "text"))
                for part in parts or ()
                if isinstance(value_at(part, "text"), str)
            ).strip()
            usage_data = data.get("usageMetadata") if isinstance(data, dict) else None
            usage = _gemini_usage(usage_data)
            provider_cost_usd = optional_nonnegative_float(value_at(data, "cost"))
            if not text:
                raise ProviderExecutionError.for_empty_reply(
                    usage=usage, provider_cost_usd=provider_cost_usd
                )
            return AdapterResponse(
                text=text,
                model_id=request.model_id,
                usage=usage,
                provider_cost_usd=provider_cost_usd,
                finish_reason=normalize_finish_reason(
                    value_at(first, "finishReason")
                ),
                sampling_wire=describe_sampling_wire(wire),
            )
        except ProviderExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001 - mapped to a fixed safe category
            raise map_provider_error(exc) from exc
        finally:
            if client is not None:
                client.close()


__all__ = ["GeminiExecutionAdapter"]
