"""OpenAI-compatible chat-completions execution adapter."""

from __future__ import annotations

from typing import Any

from dashboard.backend.domain.model_providers.execution_catalog import (
    UnsupportedExecutionModel,
    resolve_execution_model_route,
)
from dashboard.backend.domain.model_providers.models import ProviderRecord
from dashboard.backend.infrastructure.llm.chat_completions import (
    first_choice,
    response_text,
    usage_counts,
)
from dashboard.backend.infrastructure.llm.execution.models import LLMExecutionRequest
from dashboard.backend.infrastructure.llm.reasoning_controls import (
    is_reasoning_off,
    thinking_disabled_body,
)

from .base import (
    SDK_MAX_RETRIES,
    AdapterResponse,
    ClientFactory,
    CredentialMaterial,
    ProviderExecutionError,
    build_safe_http_client,
    describe_sampling_wire,
    map_provider_error,
    normalize_finish_reason,
    optional_nonnegative_float,
    provider_http_timeout,
    usage_from_fields,
    value_at,
)

# Providers, by id, that take thinking off as ``thinking: {type: "disabled"}``.
# Keyed on the provider, not on ``adapter_type == "openai_compatible"``: that
# adapter type is the one an admin registers for any OpenAI-shaped server
# (DeepSeek's own API, vLLM, Together), and those were never probed with this
# field -- a strict one answers 400 to an unknown body key, a lenient one
# ignores it and thinks anyway while the run says thinking was off. They keep
# the ``reasoning`` shape they were sent before this existed.
THINKING_TOGGLE_PROVIDERS = frozenset({"commonstack"})


def _default_client_factory(**kwargs: Any) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - deployment dependency guard
        raise ProviderExecutionError("provider_unavailable") from exc
    return OpenAI(**kwargs)


def _response_usage(response: Any):
    counts = usage_counts(response)
    return None if counts is None else usage_from_fields(*counts)


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
            # One Timeout object for both layers, and no SDK retries: see the
            # provider-timeout block in base.py.
            timeout = provider_http_timeout()
            owned_http_client = build_safe_http_client(
                provider.approved_base_url,
                proxy_origin=self.proxy_origin,
                timeout=timeout,
            )
            client = self.client_factory(
                api_key=credential.secret,
                base_url=provider.approved_base_url,
                http_client=owned_http_client,
                timeout=timeout,
                max_retries=SDK_MAX_RETRIES,
            )
            kwargs: dict[str, Any] = {
                "model": provider_model_id,
                "messages": messages,
                "max_tokens": request.usage_policy.max_output_tokens,
            }
            wire: list[str] = []
            if request.temperature is not None:
                kwargs["temperature"] = request.temperature
                wire.append(f"temperature={request.temperature}")
            if request.reasoning_effort and provider.adapter_type in {
                "openrouter",
                "openai_compatible",
            }:
                reasoning_off = is_reasoning_off(request.reasoning_effort)
                if reasoning_off and provider.provider_id in THINKING_TOGGLE_PROVIDERS:
                    # CommonStack honours no graduated reasoning control for
                    # DeepSeek V4 Pro or Qwen3.7 Plus: reasoning.effort,
                    # reasoning.enabled=false and thinking.budget_tokens were
                    # all ignored in the 2026-10-01 probe (#539). Thinking
                    # on/off is the one control it honours. Sent instead of
                    # `reasoning`, not beside it.
                    kwargs["extra_body"] = thinking_disabled_body()
                    wire.append("thinking=disabled")
                else:
                    reasoning = {"effort": request.reasoning_effort}
                    if provider.adapter_type == "openrouter" and reasoning_off:
                        reasoning.update({"enabled": False, "exclude": True})
                    kwargs["extra_body"] = {
                        "reasoning": reasoning,
                    }
                    wire.append(
                        "reasoning.effort="
                        + request.reasoning_effort
                        + (",enabled=false" if "enabled" in reasoning else "")
                    )
            elif request.reasoning_effort and provider.adapter_type == "openai":
                # Chat Completions takes it as a top-level parameter; only
                # reasoning models accept it, and only the catalog's
                # reasoning-only policy ever asks for it here.
                kwargs["reasoning_effort"] = request.reasoning_effort
                wire.append(f"reasoning_effort={request.reasoning_effort}")
            response = client.chat.completions.create(**kwargs)
            text = response_text(response)
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
                    value_at(first_choice(response), "finish_reason")
                ),
                sampling_wire=describe_sampling_wire(wire),
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
