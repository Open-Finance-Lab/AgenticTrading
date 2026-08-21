"""OpenRouter model-discovery adapter."""

from .base import ProviderAdapter


class OpenRouterAdapter(ProviderAdapter):
    def __init__(self) -> None:
        super().__init__("openrouter", "/models")
