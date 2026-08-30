"""Gemini REST execution adapter with pinned HTTP transport."""

from __future__ import annotations

from urllib.parse import quote

from dashboard.backend.domain.model_providers.execution_catalog import (
    UnsupportedExecutionModel,
    resolve_execution_model_route,
)
from dashboard.backend.domain.model_providers.models import ProviderRecord
from dashboard.backend.infrastructure.llm.execution.models import LLMExecutionRequest

from .base import (
    AdapterResponse,
    CredentialMaterial,
    ProviderExecutionError,
    build_safe_http_client,
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
        if request.temperature is not None:
            generation["temperature"] = request.temperature  # type: ignore[index]
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
            if not text:
                raise ProviderExecutionError("response_invalid")
            usage_data = data.get("usageMetadata") if isinstance(data, dict) else None
            usage = usage_from_fields(
                value_at(usage_data, "promptTokenCount"),
                value_at(usage_data, "candidatesTokenCount"),
            ) if usage_data is not None else None
            return AdapterResponse(
                text=text,
                model_id=request.model_id,
                usage=usage,
                provider_cost_usd=optional_nonnegative_float(
                    value_at(data, "cost")
                ),
                finish_reason=normalize_finish_reason(
                    value_at(first, "finishReason")
                ),
            )
        except ProviderExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001 - mapped to a fixed safe category
            raise map_provider_error(exc) from exc
        finally:
            if client is not None:
                client.close()


__all__ = ["GeminiExecutionAdapter"]
