"""Google Gemini authenticated model-discovery adapter."""

from .base import ProviderAdapter


class GeminiAdapter(ProviderAdapter):
    def __init__(self) -> None:
        super().__init__(
            "gemini",
            "/v1beta/models",
            "https://generativelanguage.googleapis.com",
        )

    def build_request(self, base_url: str, secret: str) -> tuple[str, dict[str, str]]:
        return f"{base_url.rstrip('/')}{self.discovery_path}", {
            "x-goog-api-key": secret,
            "Accept": "application/json",
        }

    def parse_models(self, payload):
        data = payload if isinstance(payload, dict) else {}
        models = data.get("models") or []
        result = []
        for item in models if isinstance(models, list) else []:
            value = item.get("name") if isinstance(item, dict) else None
            if isinstance(value, str) and value.startswith("models/"):
                result.append(value.removeprefix("models/")[:200])
        return sorted(set(result))[:200]
