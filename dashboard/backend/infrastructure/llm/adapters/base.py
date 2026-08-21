"""Small, bounded provider adapter contract for credential verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from dashboard.backend.domain.model_providers.models import CredentialValidation

from .safe_http import (
    ProviderAddressResolutionError,
    UnsafeProviderAddress,
    build_pinned_transport,
)


@dataclass(frozen=True)
class ProviderAdapter:
    """One provider's safe discovery endpoint and authentication headers."""

    adapter_type: str
    discovery_path: str

    def build_request(self, base_url: str, secret: str) -> tuple[str, dict[str, str]]:
        return f"{base_url.rstrip('/')}{self.discovery_path}", {
            "Authorization": f"Bearer {secret}",
            "Accept": "application/json",
        }

    def parse_models(self, payload: Any) -> list[str]:
        data = payload if isinstance(payload, dict) else {}
        raw = data.get("data") or data.get("models") or []
        result: list[str] = []
        if isinstance(raw, list):
            for item in raw:
                value = item.get("id") if isinstance(item, dict) else item
                if isinstance(value, str) and value.strip():
                    result.append(value.strip()[:200])
        return sorted(set(result))[:200]

    def validate(self, base_url: str, secret: str, *, client: httpx.Client | None = None) -> CredentialValidation:
        url, headers = self.build_request(base_url, secret)
        try:
            if client is None:
                transport = build_pinned_transport(url)
                with httpx.Client(
                    timeout=httpx.Timeout(8.0, connect=3.0),
                    follow_redirects=False,
                    trust_env=False,
                    transport=transport,
                ) as owned:
                    response = owned.get(url, headers=headers)
            else:
                response = client.get(url, headers=headers)
        except UnsafeProviderAddress:
            return CredentialValidation(
                status="invalid",
                message="Provider address is not allowed.",
            )
        except ProviderAddressResolutionError:
            return CredentialValidation(
                status="verification_unavailable",
                message="Provider address could not be resolved.",
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            return CredentialValidation(status="verification_unavailable", message="Provider verification timed out or was unavailable.")
        except httpx.HTTPError:
            return CredentialValidation(status="verification_unavailable", message="Provider verification was unavailable.")

        if response.status_code in {401, 403}:
            return CredentialValidation(status="invalid", message="The provider rejected this API key.")
        if 300 <= response.status_code < 400:
            return CredentialValidation(status="invalid", message="The provider returned an unexpected redirect.")
        if response.status_code == 429 or response.status_code >= 500:
            return CredentialValidation(status="verification_unavailable", message="The provider is temporarily unavailable.")
        if response.status_code >= 400:
            return CredentialValidation(status="invalid", message="The provider rejected this API key or request.")
        if response.status_code != 200:
            return CredentialValidation(status="invalid", message="Provider returned an unexpected response.")
        content_type = response.headers.get("content-type", "").lower()
        if content_type and "json" not in content_type:
            return CredentialValidation(status="invalid", message="Provider returned an invalid model list.")
        try:
            payload = response.json()
        except ValueError:
            return CredentialValidation(status="invalid", message="Provider returned an invalid model list.")
        models = self.parse_models(payload)
        if not models:
            return CredentialValidation(status="invalid", message="Provider returned an invalid model list.")
        return CredentialValidation(status="verified", message="API key verified.", models=models)
