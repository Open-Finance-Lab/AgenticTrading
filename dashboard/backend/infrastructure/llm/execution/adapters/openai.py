"""OpenAI-compatible chat-completions execution adapter."""

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
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - deployment dependency guard
        raise ProviderExecutionError("provider_unavailable") from exc
    return OpenAI(**kwargs)


def _first_choice(response: Any) -> Any:
    choices = value_at(response, "choices", ())
    return choices[0] if choices else None


def _response_text(response: Any) -> str:
    first = _first_choice(response)
    message = value_at(first, "message")
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


def _response_usage(response: Any):
    usage = value_at(response, "usage")
    if usage is None:
        return None
    return usage_from_fields(
        value_at(usage, "prompt_tokens", value_at(usage, "input_tokens")),
        value_at(usage, "completion_tokens", value_at(usage, "output_tokens")),
    )


class OpenAIExecutionAdapter:
    """Execute OpenAI, OpenRouter, and allowlisted OpenAI-compatible providers."""

    def __init__(
        self,
        *,
        client_factory: ClientFactory = _default_client_factory,
        proxy_origin: str | None = None,
    ) -> None:
        self.client_factory = client_factory
        self.proxy_origin = proxy_origin

    def complete(
        self,
        request: LLMExecutionRequest,
        credential: CredentialMaterial,
        provider: ProviderRecord,
    ) -> AdapterResponse:
        messages: list[dict[str, str]] = []
        if request.system_message:
            messages.append({"role": "system", "content": request.system_message})
        messages.extend(message.model_dump() for message in request.messages)
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
                proxy_origin=self.proxy_origin,
            )
            client = self.client_factory(
                api_key=credential.secret,
                base_url=provider.approved_base_url,
                http_client=owned_http_client,
            )
            kwargs: dict[str, Any] = {
                "model": provider_model_id,
                "messages": messages,
                "max_tokens": request.usage_policy.max_output_tokens,
            }
            if request.temperature is not None:
                kwargs["temperature"] = request.temperature
            if request.reasoning_effort and provider.adapter_type in {
                "openrouter",
                "openai_compatible",
            }:
                effort = request.reasoning_effort.strip().lower()
                reasoning = {"effort": request.reasoning_effort}
                if provider.adapter_type == "openrouter" and effort in {
                    "none",
                    "off",
                    "false",
                    "0",
                    "disabled",
                }:
                    reasoning.update({"enabled": False, "exclude": True})
                kwargs["extra_body"] = {
                    "reasoning": reasoning,
                }
            response = client.chat.completions.create(**kwargs)
            text = _response_text(response)
            if not text:
                raise ProviderExecutionError("response_invalid")
            usage = _response_usage(response)
            provider_cost_usd = optional_nonnegative_float(
                value_at(response, "cost", value_at(value_at(response, "usage"), "cost"))
            )
            return AdapterResponse(
                text=text,
                model_id=request.model_id,
                usage=usage,
                provider_cost_usd=provider_cost_usd,
                finish_reason=normalize_finish_reason(
                    value_at(_first_choice(response), "finish_reason")
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


class OpenAIAdapter(OpenAIExecutionAdapter):
    def __init__(self, *, client_factory: ClientFactory = _default_client_factory) -> None:
        super().__init__(
            client_factory=client_factory,
            proxy_origin="https://api.openai.com/v1",
        )


class OpenRouterAdapter(OpenAIExecutionAdapter):
    def __init__(self, *, client_factory: ClientFactory = _default_client_factory) -> None:
        super().__init__(
            client_factory=client_factory,
            proxy_origin="https://openrouter.ai/api/v1",
        )


class OpenAICompatibleAdapter(OpenAIExecutionAdapter):
    def __init__(self, *, client_factory: ClientFactory = _default_client_factory) -> None:
        super().__init__(client_factory=client_factory, proxy_origin=None)


__all__ = [
    "OpenAIAdapter",
    "OpenAICompatibleAdapter",
    "OpenAIExecutionAdapter",
    "OpenRouterAdapter",
]
