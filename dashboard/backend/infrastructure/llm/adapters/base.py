"""Small, bounded provider adapter contract for credential verification."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any
from urllib.parse import urlsplit

import httpx

from dashboard.backend.domain.model_providers.models import CredentialValidation

from .safe_http import (
    ProviderAddressResolutionError,
    UnsafeProviderAddress,
    build_explicit_proxy_transport,
    build_pinned_transport,
)


def _https_origin(url: str) -> tuple[str, int] | None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        return None
    try:
        host = parsed.hostname.strip().rstrip(".").lower().encode("idna").decode("ascii")
        port = parsed.port or 443
    except (UnicodeError, ValueError):
        return None
    return host, port


@dataclass(frozen=True)
class ProviderAdapter:
    """One provider's safe discovery endpoint and authentication headers."""

    adapter_type: str
    discovery_path: str
    proxy_origin: str | None = None

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

    def validate_payload(self, payload: Any) -> tuple[bool, list[str], str]:
        """Validate a successful provider response without exposing its body."""

        models = self.parse_models(payload)
        if not models:
            return False, [], "Provider returned an invalid model list."
        return True, models, "API key verified."

    def _transport(self, url: str):
        proxy = (os.getenv("BROKER_CREDENTIAL_VERIFICATION_PROXY") or "").strip()
        if (
            proxy
            and self.proxy_origin
            and _https_origin(url) == _https_origin(self.proxy_origin)
        ):
            return build_explicit_proxy_transport(proxy)
        return build_pinned_transport(url)

    def validate(self, base_url: str, secret: str, *, client: httpx.Client | None = None) -> CredentialValidation:
        url, headers = self.build_request(base_url, secret)
        try:
            if client is None:
                transport = self._transport(url)
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
                status="verification_unavailable",
                message="Provider address is not allowed.",
            )
        except ProviderAddressResolutionError:
            return CredentialValidation(
                status="verification_unavailable",
                message="Provider address could not be resolved.",
            )
        except (httpx.TimeoutException, httpx.NetworkError):
            return CredentialValidation(status="verification_unavailable", message="Provider verification timed out or was unavailable.")
        except httpx.HTTPError:
            return CredentialValidation(status="verification_unavailable", message="Provider verification was unavailable.")

        if response.status_code in {401, 403}:
            return CredentialValidation(status="invalid", message="The provider rejected this API key.")
        if 300 <= response.status_code < 400:
            return CredentialValidation(status="verification_unavailable", message="The provider returned an unexpected redirect.")
        if response.status_code == 429 or response.status_code >= 500:
            return CredentialValidation(status="verification_unavailable", message="The provider is temporarily unavailable.")
        if response.status_code >= 400:
            return CredentialValidation(status="invalid", message="The provider rejected this API key or request.")
        if response.status_code != 200:
            return CredentialValidation(status="verification_unavailable", message="Provider returned an unexpected response.")
        content_type = response.headers.get("content-type", "").lower()
        if content_type and "json" not in content_type:
            return CredentialValidation(status="verification_unavailable", message="Provider returned an invalid verification response.")
        try:
            payload = response.json()
        except ValueError:
            return CredentialValidation(status="verification_unavailable", message="Provider returned an invalid verification response.")
        valid, models, message = self.validate_payload(payload)
        if not valid:
            return CredentialValidation(status="verification_unavailable", message=message)
        return CredentialValidation(status="verified", message=message, models=models)
