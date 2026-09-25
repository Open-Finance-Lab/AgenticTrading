"""Shared validation and exceptions for model provider repositories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
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



@dataclass(frozen=True)
class DefaultCredentialFacts:
    """One user's default credentials, folded to what operational state needs.

    ``status`` follows get_operational_facts' worst-first precedence: invalid
    beats verification_unavailable beats verified. ``default_provider_ids``
    carries every provider a default credential points at, and
    ``verified_default_provider_counts`` the per-provider count of *verified*
    defaults -- the ``== 1`` test in ModelProviderService.list_execution_options
    (service.py:194) is a count, not a boolean, and collapsing it to one loses
    the two-defaults case.
    """

    status: str
    default_provider_ids: frozenset[str]
    verified_default_provider_counts: Mapping[str, int]


def fold_default_credential_rows(rows) -> dict[int, DefaultCredentialFacts]:
    """Shared by both twins: rows of (user_id, provider_id, status) -> facts."""
    statuses: dict[int, set[str]] = {}
    providers: dict[int, set[str]] = {}
    verified: dict[int, dict[str, int]] = {}
    for row in rows:
        user_id = int(row["user_id"])
        provider_id = str(row["provider_id"])
        status = str(row["status"])
        statuses.setdefault(user_id, set()).add(status)
        providers.setdefault(user_id, set()).add(provider_id)
        if status == "verified":
            counts = verified.setdefault(user_id, {})
            counts[provider_id] = counts.get(provider_id, 0) + 1
    result: dict[int, DefaultCredentialFacts] = {}
    for user_id, seen in statuses.items():
        if "invalid" in seen:
            status = "invalid"
        elif "verification_unavailable" in seen:
            status = "verification_unavailable"
        elif "verified" in seen:
            status = "verified"
        else:
            status = "missing"
        result[user_id] = DefaultCredentialFacts(
            status=status,
            default_provider_ids=frozenset(providers[user_id]),
            verified_default_provider_counts=dict(verified.get(user_id, {})),
        )
    return result

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
    "anthropic/claude-haiku-4-5",
)

# The seed below is ``ON CONFLICT DO NOTHING``, so an id appended to the
# allowlist above never reaches a deployment whose row already exists. Each
# addition therefore ships with a one-shot backfill keyed on this id; bump it
# (``-v2``, ...) with the next addition so the backfill runs again.
COMMONSTACK_ALLOWLIST_MIGRATION_ID = "commonstack-allowlist-v1"


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


def commonstack_allowlist_backfill(capabilities_json: str | None) -> str | None:
    """Return ``capabilities_json`` with the seeded CommonStack models appended.

    Only appends: an id an admin added stays and nothing is reordered. Returns
    None when there is nothing to add *or* the stored row cannot be read --
    ``deserialize_capabilities`` turns an unreadable row into default
    capabilities, and writing that back would erase whatever the row held.
    """
    try:
        capabilities = ProviderCapabilities.model_validate(
            json.loads(capabilities_json or "")
        )
    except (TypeError, ValueError):
        return None
    current = capabilities.model_allowlist
    missing = tuple(
        model_id
        for model_id in COMMONSTACK_MODEL_ALLOWLIST
        if model_id not in current
    )
    if not missing:
        return None
    return serialize_capabilities(
        capabilities.model_copy(update={"model_allowlist": current + missing})
    )


def deserialize_capabilities(value: str | None) -> ProviderCapabilities:
    try:
        return ProviderCapabilities.model_validate(json.loads(value or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return ProviderCapabilities()
