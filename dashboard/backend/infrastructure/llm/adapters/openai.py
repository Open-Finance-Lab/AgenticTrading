"""Native OpenAI model-discovery adapter."""

from .base import ProviderAdapter


class OpenAIAdapter(ProviderAdapter):
    def __init__(self) -> None:
        super().__init__("openai", "/models")
