"""OpenRouter API-key verification adapter."""

from typing import Any

from .base import ProviderAdapter


class OpenRouterAdapter(ProviderAdapter):
    def __init__(self) -> None:
        super().__init__("openrouter", "/key", "https://openrouter.ai")

    def validate_payload(self, payload: Any) -> tuple[bool, list[str], str]:
        # /key is an authenticated identity endpoint. It intentionally does not
        # call a generation route and does not need to enumerate billable models.
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            return False, [], "Provider returned an invalid key verification response."
        return True, [], "API key verified."
