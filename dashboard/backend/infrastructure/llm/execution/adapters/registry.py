"""Provider-record to execution-adapter mapping."""

from __future__ import annotations

from dashboard.backend.domain.model_providers.models import ProviderRecord

from .anthropic import AnthropicExecutionAdapter
from .base import ProviderExecutionAdapter, ProviderExecutionError
from .gemini import GeminiExecutionAdapter
from .openai import (
    OpenAIAdapter,
    OpenAICompatibleAdapter,
    OpenRouterAdapter,
)


def get_execution_adapter(provider: ProviderRecord) -> ProviderExecutionAdapter:
    adapter_type = provider.adapter_type
    if adapter_type == "openai":
        return OpenAIAdapter()
    if adapter_type == "openrouter":
        return OpenRouterAdapter()
    if adapter_type == "anthropic":
        return AnthropicExecutionAdapter()
    if adapter_type == "gemini":
        return GeminiExecutionAdapter()
    if adapter_type == "openai_compatible":
        return OpenAICompatibleAdapter()
    raise ProviderExecutionError("provider_unavailable")


__all__ = ["get_execution_adapter"]
