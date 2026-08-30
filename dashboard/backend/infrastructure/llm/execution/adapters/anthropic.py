"""Anthropic Messages execution adapter."""

from __future__ import annotations

from typing import Any

from dashboard.backend.domain.model_providers.execution_catalog import (
    UnsupportedExecutionModel,
    resolve_execution_model_route,
)
from dashboard.backend.domain.model_providers.models import ProviderRecord
from dashboard.backend.infrastructure.llm.execution.models import LLMExecutionRequest

from .base import (
    AdapterResponse,
    ClientFactory,
    CredentialMaterial,
    ProviderExecutionError,
    build_safe_http_client,
    map_provider_error,
    normalize_finish_reason,
    optional_nonnegative_float,
    usage_from_fields,
    value_at,
)


def _default_client_factory(**kwargs: Any) -> Any:
    try:
        from anthropic import Anthropic
    except ImportError as exc:  # pragma: no cover - deployment dependency guard
        raise ProviderExecutionError("provider_unavailable") from exc
    return Anthropic(**kwargs)


def _content_text(response: Any) -> str:
    blocks = value_at(response, "content", ())
    chunks: list[str] = []
    for block in blocks or ():
        if value_at(block, "type") not in {None, "text"}:
            continue
        text = value_at(block, "text")
        if isinstance(text, str) and text.strip():
            chunks.append(text.strip())
    return "".join(chunks).strip()


class AnthropicExecutionAdapter:
    def __init__(self, *, client_factory: ClientFactory = _default_client_factory) -> None:
        self.client_factory = client_factory

    def complete(
        self,
        request: LLMExecutionRequest,
        credential: CredentialMaterial,
        provider: ProviderRecord,
    ) -> AdapterResponse:
        messages = [message.model_dump() for message in request.messages if message.role != "system"]
        client = None
        owned_http_client = None
        try:
            try:
                provider_model_id = resolve_execution_model_route(
                    provider,
                    request.model_id,
                ).provider_model_id
            except UnsupportedExecutionModel as exc:
                raise ProviderExecutionError("provider_unavailable") from exc
            owned_http_client = build_safe_http_client(
                provider.approved_base_url,
                proxy_origin="https://api.anthropic.com",
            )
            client = self.client_factory(
                api_key=credential.secret,
                base_url=provider.approved_base_url,
                http_client=owned_http_client,
            )
            kwargs: dict[str, Any] = {
                "model": provider_model_id,
                "max_tokens": request.usage_policy.max_output_tokens,
                "messages": messages,
            }
            if request.system_message:
                kwargs["system"] = request.system_message
            if request.temperature is not None:
                kwargs["temperature"] = request.temperature
            response = client.messages.create(**kwargs)
            text = _content_text(response)
            if not text:
                raise ProviderExecutionError("response_invalid")
            usage = value_at(response, "usage")
            normalized_usage = usage_from_fields(
                value_at(usage, "input_tokens"),
                value_at(usage, "output_tokens"),
            ) if usage is not None else None
            return AdapterResponse(
                text=text,
                model_id=request.model_id,
                usage=normalized_usage,
                provider_cost_usd=optional_nonnegative_float(
                    value_at(response, "cost", value_at(usage, "cost"))
                ),
                finish_reason=normalize_finish_reason(
                    value_at(response, "stop_reason")
                ),
            )
        except ProviderExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001 - mapped to a fixed safe category
            raise map_provider_error(exc) from exc
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
            if owned_http_client is not None:
                owned_http_client.close()


__all__ = ["AnthropicExecutionAdapter"]
