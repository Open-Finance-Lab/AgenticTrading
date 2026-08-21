"""Allow-listed adapter registry; arbitrary URLs are never selected by ID."""

from __future__ import annotations

from .anthropic import AnthropicAdapter
from .base import ProviderAdapter
from .gemini import GeminiAdapter
from .openai import OpenAIAdapter
from .openrouter import OpenRouterAdapter

_ADAPTERS: dict[str, ProviderAdapter] = {
    "openrouter": OpenRouterAdapter(),
    "openai": OpenAIAdapter(),
    "openai_compatible": ProviderAdapter("openai_compatible", "/models"),
    "anthropic": AnthropicAdapter(),
    "gemini": GeminiAdapter(),
}


def get_adapter(adapter_type: str) -> ProviderAdapter:
    try:
        return _ADAPTERS[adapter_type]
    except KeyError as exc:
        raise ValueError("unsupported provider adapter") from exc
