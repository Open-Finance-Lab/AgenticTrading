"""Shared validation and exceptions for model provider repositories."""

from __future__ import annotations

from urllib.parse import urlsplit
import base64
import hashlib
import hmac
import ipaddress
import json
import os
import re
from collections.abc import Mapping

from .models import AdapterType, ProviderCapabilities


class ModelProviderStoreError(RuntimeError):
    pass


class CredentialNotFoundError(ModelProviderStoreError):
    pass


class CredentialOwnershipError(ModelProviderStoreError):
    pass


class CredentialConflictError(ModelProviderStoreError):
    pass


class ProviderNotFoundError(ModelProviderStoreError):
    pass


class InvalidProviderOriginError(ModelProviderStoreError):
    pass


def canonical_request_digest(payload: Mapping[str, object]) -> str:
    """Return a stable digest for an admin mutation without retaining secrets."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def secret_fingerprint(secret: str) -> str:
    """Return an HMAC fingerprint keyed by the configured Fernet key."""

    configured = (os.getenv("BROKER_TOKEN_ENCRYPTION_KEY") or "").strip()
    if not configured:
        raise RuntimeError("BROKER_TOKEN_ENCRYPTION_KEY is not set")
    try:
        key = base64.urlsafe_b64decode(configured.encode("ascii"))
    except (ValueError, UnicodeEncodeError) as exc:
        raise RuntimeError("BROKER_TOKEN_ENCRYPTION_KEY is invalid") from exc
    return hmac.new(key, secret.encode("utf-8"), hashlib.sha256).hexdigest()


SUPPORTED_ADAPTER_TYPES: set[str] = {
    "openrouter",
    "openai",
    "anthropic",
    "gemini",
    "openai_compatible",
}

_PROVIDER_ID_PATTERN = re.compile(r"^[a-z0-9_]{2,64}$")


COMMONSTACK_MODEL_ALLOWLIST = (
    "openai/gpt-5.5",
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-sonnet-4-6",
    "deepseek/deepseek-v4-pro",
    "qwen/qwen3.7-plus",
)


SEEDED_PROVIDERS = (
    {
        "provider_id": "openrouter",
        "display_name": "OpenRouter",
        "adapter_type": "openrouter",
        "byok_enabled": True,
        "platform_enabled": True,
        "approved_base_url": "https://openrouter.ai/api/v1",
        "capabilities": ProviderCapabilities(
            model_discovery=True,
            system_messages=True,
            reasoning=True,
            cached_token_usage=True,
            reasoning_token_usage=True,
            reported_monetary_cost=True,
            supported_parameters=("temperature", "max_output_tokens", "reasoning_effort"),
        ),
    },
    {
        "provider_id": "commonstack",
        "display_name": "CommonStack",
        "adapter_type": "openai_compatible",
        "approved_base_url": "https://api.commonstack.ai/v1",
        "byok_enabled": False,
        "platform_enabled": True,
        "capabilities": ProviderCapabilities(
            model_discovery=True,
            system_messages=True,
            reasoning=True,
            supported_parameters=(
                "temperature",
                "max_output_tokens",
                "reasoning_effort",
            ),
            model_allowlist=COMMONSTACK_MODEL_ALLOWLIST,
        ),
    },
    {
        "provider_id": "openai",
        "display_name": "OpenAI",
        "adapter_type": "openai",
        "approved_base_url": "https://api.openai.com/v1",
        "capabilities": ProviderCapabilities(
            model_discovery=True,
            system_messages=True,
            reasoning=True,
            cached_token_usage=True,
            reasoning_token_usage=True,
            reported_monetary_cost=False,
            supported_parameters=("temperature", "max_output_tokens", "reasoning_effort"),
        ),
    },
    {
        "provider_id": "anthropic",
        "display_name": "Anthropic",
        "adapter_type": "anthropic",
        "approved_base_url": "https://api.anthropic.com",
        "capabilities": ProviderCapabilities(
            model_discovery=True,
            system_messages=True,
            reasoning=True,
            cached_token_usage=True,
            reasoning_token_usage=True,
            supported_parameters=("temperature", "max_output_tokens", "reasoning_effort"),
        ),
    },
    {
        "provider_id": "gemini",
        "display_name": "Google Gemini",
        "adapter_type": "gemini",
        "approved_base_url": "https://generativelanguage.googleapis.com",
        "capabilities": ProviderCapabilities(
            model_discovery=True,
            system_messages=True,
            reasoning=True,
            supported_parameters=("temperature", "max_output_tokens", "reasoning_effort"),
        ),
    },
)


def validate_approved_origin(value: str) -> str:
    value = str(value or "").strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname:
        raise InvalidProviderOriginError("provider origin must be an HTTPS URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise InvalidProviderOriginError("provider origin must not contain credentials or query data")
    host = parsed.hostname.lower()
    try:
        address = ipaddress.ip_address(host)
        if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
            raise InvalidProviderOriginError("private provider origins are not allowed")
    except ValueError:
        if host in {"localhost", "metadata.google.internal"} or host.endswith(".local"):
            raise InvalidProviderOriginError("private provider origins are not allowed")
    return value


def validate_adapter_type(value: str) -> AdapterType:
    if value not in SUPPORTED_ADAPTER_TYPES:
        raise ModelProviderStoreError("unsupported provider adapter")
    return value  # type: ignore[return-value]


def validate_provider_id(value: str) -> str:
    value = str(value or "").strip()
    if not _PROVIDER_ID_PATTERN.fullmatch(value):
        raise ModelProviderStoreError("invalid provider id")
    return value


def serialize_capabilities(value: ProviderCapabilities | dict) -> str:
    capabilities = value if isinstance(value, ProviderCapabilities) else ProviderCapabilities.model_validate(value)
    return json.dumps(capabilities.model_dump(), sort_keys=True, separators=(",", ":"))


def deserialize_capabilities(value: str | None) -> ProviderCapabilities:
    try:
        return ProviderCapabilities.model_validate(json.loads(value or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return ProviderCapabilities()
