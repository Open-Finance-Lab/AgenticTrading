"""Execution adapter protocol and safe provider-network helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Callable, Protocol
from urllib.parse import urlsplit

import httpx

from dashboard.backend.domain.model_providers.models import ProviderRecord
from dashboard.backend.infrastructure.llm.adapters.safe_http import (
    ProviderAddressResolutionError,
    UnsafeProviderAddress,
    build_explicit_proxy_transport,
    build_pinned_transport,
)
from dashboard.backend.infrastructure.llm.execution.errors import (
    ExecutionErrorCategory,
    LLMExecutionError,
)
from dashboard.backend.infrastructure.llm.execution.models import (
    LLMExecutionRequest,
    LLMUsage,
)


class CredentialMaterial(Protocol):
    credential_id: str | None
    provider_id: str
    key_last_four: str
    secret: str


# Provider spellings of "the reply stopped at the output ceiling", folded to
# one value so callers above the adapters never see the vendor vocabulary.
_OUTPUT_CEILING_FINISH_REASONS = frozenset({"length", "max_tokens"})
FINISH_REASON_MAX_TOKENS = "max_tokens"
# ``LLMExecutionResult.finish_reason`` is bounded; an OpenAI-compatible
# provider may put anything in this field, and a long value must not turn a
# successful call into ``response_invalid`` when the result model rejects it.
_FINISH_REASON_MAX_LENGTH = 32

_PROVIDER_ERROR_PAYLOAD_MAX_BYTES = 4096
_QUOTA_ERROR_IDENTIFIERS = frozenset(
    {
        "in_flight_budget_exhausted",
        "insufficient_quota",
        "quota_exceeded",
        "quota_exhausted",
        "insufficient_balance",
        "credit_balance_exhausted",
    }
)
_QUOTA_ERROR_PHRASES = (
    "insufficient balance",
    "insufficient credits",
    "quota exceeded",
    "quota exhausted",
    "exceeded your current quota",
    "not enough credits",
)


def normalize_finish_reason(value: Any) -> str | None:
    """Fold a provider stop/finish reason into a lowercase, vendor-neutral tag.

    ``length`` (OpenAI / OpenRouter), ``MAX_TOKENS`` (Gemini) and
    ``max_tokens`` (Anthropic) all become ``"max_tokens"``; any other string is
    passed through lowercased (and clamped to the result model's length bound)
    so it stays inspectable; anything else is ``None``.
    """
    if not isinstance(value, str):
        return None
    reason = value.strip().lower()
    if not reason:
        return None
    if reason in _OUTPUT_CEILING_FINISH_REASONS:
        return FINISH_REASON_MAX_TOKENS
    return reason[:_FINISH_REASON_MAX_LENGTH]


@dataclass(frozen=True)
class AdapterResponse:
    text: str
    model_id: str
    usage: LLMUsage | None
    provider_cost_usd: float | None = None
    # Why the provider stopped generating, via ``normalize_finish_reason``.
    # ``"max_tokens"`` is the one value callers act on: the reply was cut at
    # the output ceiling, so an unparseable body is a truncation, not a
    # malformed answer. ``None`` when the provider reported nothing.
    finish_reason: str | None = None


class ProviderExecutionError(LLMExecutionError):
    """A fixed, secret-free error emitted by an execution adapter."""


class ProviderExecutionAdapter(Protocol):
    def complete(
        self,
        request: LLMExecutionRequest,
        credential: CredentialMaterial,
        provider: ProviderRecord,
    ) -> AdapterResponse:
        """Run one completion against ``provider`` and return its normalised reply."""


def value_at(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def optional_nonnegative_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def usage_from_fields(input_tokens: Any, output_tokens: Any) -> LLMUsage | None:
    if isinstance(input_tokens, bool) or isinstance(output_tokens, bool):
        return None
    try:
        parsed_input = int(input_tokens)
        parsed_output = int(output_tokens)
    except (TypeError, ValueError):
        return None
    if parsed_input < 0 or parsed_output < 0:
        return None
    return LLMUsage(input_tokens=parsed_input, output_tokens=parsed_output)


def _provider_status_codes(exc: Exception) -> tuple[int, ...]:
    """Read provider statuses without trusting arbitrary exception text."""

    statuses: list[int] = []
    for value in (
        getattr(exc, "status_code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ):
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            statuses.append(value)
    return tuple(statuses)


def _bounded_error_payload(exc: Exception) -> dict[str, Any]:
    """Parse only a small structured provider error body, if one is present."""

    response = getattr(exc, "response", None)
    content = getattr(response, "content", b"")
    if isinstance(content, str):
        content = content.encode("utf-8", errors="ignore")
    elif isinstance(content, bytearray):
        content = bytes(content)
    if not isinstance(content, bytes) or len(content) > _PROVIDER_ERROR_PAYLOAD_MAX_BYTES:
        return {}
    try:
        parsed = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _structured_quota_signal(payload: dict[str, Any]) -> bool:
    """Match allowlisted code/type/message fields only."""

    identifiers: list[Any] = [payload.get("code"), payload.get("type")]
    messages: list[Any] = [payload.get("message")]
    error = payload.get("error")
    if isinstance(error, dict):
        identifiers.extend((error.get("code"), error.get("type")))
        messages.append(error.get("message"))

    for value in identifiers:
        if not isinstance(value, str):
            continue
        normalized = value.strip().lower()
        if normalized in _QUOTA_ERROR_IDENTIFIERS:
            return True
    for value in messages:
        if isinstance(value, str) and any(
            phrase in value.strip().lower() for phrase in _QUOTA_ERROR_PHRASES
        ):
            return True
    return False


def map_provider_error(exc: Exception) -> ProviderExecutionError:
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)) or "timeout" in type(exc).__name__.lower():
        return ProviderExecutionError(ExecutionErrorCategory.PROVIDER_TIMEOUT)
    status_codes = _provider_status_codes(exc)
    if any(status in {401, 403} for status in status_codes):
        return ProviderExecutionError(ExecutionErrorCategory.CREDENTIAL_INVALID)
    if any(status == 402 for status in status_codes):
        return ProviderExecutionError(ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED)
    if not status_codes or any(400 <= status < 500 for status in status_codes):
        if _structured_quota_signal(_bounded_error_payload(exc)):
            return ProviderExecutionError(ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED)
    if isinstance(exc, (UnsafeProviderAddress, ProviderAddressResolutionError)):
        return ProviderExecutionError(ExecutionErrorCategory.PROVIDER_UNAVAILABLE)
    return ProviderExecutionError(ExecutionErrorCategory.PROVIDER_UNAVAILABLE)


def build_safe_http_client(
    base_url: str,
    *,
    proxy_origin: str | None = None,
    timeout_seconds: float = 60.0,
) -> httpx.Client:
    """Create an explicit-proxy official client or an IP-pinned custom client."""

    proxy = (os.getenv("BROKER_CREDENTIAL_VERIFICATION_PROXY") or "").strip()
    parsed = urlsplit(base_url)
    proxy_parsed = urlsplit(proxy_origin or "")
    same_official_origin = bool(
        proxy_origin
        and parsed.scheme == "https"
        and proxy_parsed.scheme == "https"
        and parsed.hostname == proxy_parsed.hostname
        and (parsed.port or 443) == (proxy_parsed.port or 443)
    )
    transport = (
        build_explicit_proxy_transport(proxy)
        if proxy and same_official_origin
        else build_pinned_transport(base_url)
    )
    return httpx.Client(
        timeout=httpx.Timeout(timeout_seconds, connect=8.0),
        follow_redirects=False,
        trust_env=False,
        transport=transport,
    )


ClientFactory = Callable[..., Any]


__all__ = [
    "FINISH_REASON_MAX_TOKENS",
    "AdapterResponse",
    "ClientFactory",
    "CredentialMaterial",
    "ProviderExecutionAdapter",
    "ProviderExecutionError",
    "build_safe_http_client",
    "map_provider_error",
    "normalize_finish_reason",
    "optional_nonnegative_float",
    "usage_from_fields",
    "value_at",
]
