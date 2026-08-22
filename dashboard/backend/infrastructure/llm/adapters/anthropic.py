"""Anthropic authenticated model-discovery adapter."""

from .base import ProviderAdapter


class AnthropicAdapter(ProviderAdapter):
    def __init__(self) -> None:
        super().__init__("anthropic", "/v1/models", "https://api.anthropic.com")

    def build_request(self, base_url: str, secret: str) -> tuple[str, dict[str, str]]:
        return f"{base_url.rstrip('/')}{self.discovery_path}", {
            "x-api-key": secret,
            "anthropic-version": "2023-06-01",
            "Accept": "application/json",
        }
