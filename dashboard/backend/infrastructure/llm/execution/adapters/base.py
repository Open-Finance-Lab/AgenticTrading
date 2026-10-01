"""Execution adapter protocol and safe provider-network helpers."""

from __future__ import annotations

from dataclasses import dataclass
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
# The timeout and retry policy -- and why every SDK client runs with
# ``max_retries=0`` -- lives in the leaf ``http_policy`` so callers outside
# this layer can share it without importing it. Patch its globals there.
from dashboard.backend.infrastructure.llm.http_policy import (
    PRE_SEND_TIMEOUT_PHASES as _PRE_SEND_TIMEOUT_PHASES,
    SDK_MAX_RETRIES,
    RetryHint,
    bounded_error_payload as _bounded_error_payload,
    exception_chain as _exception_chain,
    is_timeout_error as _is_timeout_error,
    provider_http_timeout,
    provider_read_timeout_seconds,
    provider_status_codes as _provider_status_codes,
    response_headers as _response_headers,
    retry_after_seconds as _retry_after_seconds,
    status_retry_hint as _status_retry_hint,
    structured_quota_signal as _structured_quota_signal,
    timeout_phase as _timeout_phase,
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
    # What the adapter sent for sampling, via ``describe_sampling_wire``.
    sampling_wire: str | None = None


def describe_sampling_wire(controls: list[str]) -> str | None:
    """Join the sampling controls an adapter sent; ``None`` when it sent none."""

    return ";".join(controls) if controls else None


class ProviderExecutionError(LLMExecutionError):
    """A fixed, secret-free error emitted by an execution adapter.

    ``retry_hint`` tells ``LLMExecutionService`` whether the attempt may be
    repeated at the same provider; the other fields only feed its log line.
    The defaults describe "unknown, never repeat", so an error built without
    them (every scripted test adapter, the Gemini status branch) behaves
    exactly as before the retry policy existed.
    """

    def __init__(
        self,
        category: ExecutionErrorCategory | str,
        message: str | None = None,
        *,
        retry_hint: RetryHint | str = RetryHint.NONE,
        timeout_phase: str | None = None,
        provider_status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(category, message)
        self.retry_hint = RetryHint(retry_hint)
        self.timeout_phase = timeout_phase
        self.provider_status_code = provider_status_code
        self.retry_after_seconds = retry_after_seconds


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



def map_provider_error(exc: Exception) -> ProviderExecutionError:
    # Category order is load-bearing and predates the retry hints: timeout
    # first, then credential, quota, and everything else unavailable. The
    # hints only say whether ``LLMExecutionService`` may repeat the attempt
    # (see ``RetryHint``); they never change which category an error gets.
    chain = _exception_chain(exc)
    if _is_timeout_error(exc):
        phase = _timeout_phase(chain)
        return ProviderExecutionError(
            ExecutionErrorCategory.PROVIDER_TIMEOUT,
            retry_hint=(
                RetryHint.PRE_SEND
                if phase in _PRE_SEND_TIMEOUT_PHASES
                # A read timeout, or a timeout of unknown phase, is assumed
                # to have abandoned a generation the provider will bill.
                else RetryHint.NONE
            ),
            timeout_phase=phase,
        )
    status_codes = _provider_status_codes(exc)
    if any(status in {401, 403} for status in status_codes):
        return ProviderExecutionError(ExecutionErrorCategory.CREDENTIAL_INVALID)
    if any(status == 402 for status in status_codes):
        return ProviderExecutionError(ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED)
    if not status_codes or any(400 <= status < 500 for status in status_codes):
        if _structured_quota_signal(_bounded_error_payload(exc)):
            return ProviderExecutionError(ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED)
    if status_codes:
        headers = _response_headers(exc)
        return ProviderExecutionError(
            ExecutionErrorCategory.PROVIDER_UNAVAILABLE,
            retry_hint=_status_retry_hint(status_codes[0], headers),
            provider_status_code=status_codes[0],
            retry_after_seconds=_retry_after_seconds(headers),
        )
    if any(isinstance(item, UnsafeProviderAddress) for item in chain):
        # A policy refusal (non-public address); repeating it changes nothing.
        return ProviderExecutionError(ExecutionErrorCategory.PROVIDER_UNAVAILABLE)
    if any(
        isinstance(item, (ProviderAddressResolutionError, httpx.ConnectError))
        for item in chain
    ):
        return ProviderExecutionError(
            ExecutionErrorCategory.PROVIDER_UNAVAILABLE,
            retry_hint=RetryHint.PRE_SEND,
        )
    if any(
        isinstance(item, (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError))
        for item in chain
    ):
        # Sent, then the connection dropped: work may have started, so the
        # service repeats it only when it failed fast.
        return ProviderExecutionError(
            ExecutionErrorCategory.PROVIDER_UNAVAILABLE,
            retry_hint=RetryHint.REJECTED,
        )
    return ProviderExecutionError(ExecutionErrorCategory.PROVIDER_UNAVAILABLE)


def build_safe_http_client(
    base_url: str,
    *,
    proxy_origin: str | None = None,
    timeout: httpx.Timeout | None = None,
) -> httpx.Client:
    """Create an explicit-proxy official client or an IP-pinned custom client.

    ``timeout`` defaults to ``provider_http_timeout()``. SDK adapters build it
    once and pass the same object to the SDK constructor as well.
    """

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
        timeout=timeout if timeout is not None else provider_http_timeout(),
        follow_redirects=False,
        trust_env=False,
        transport=transport,
    )


ClientFactory = Callable[..., Any]


__all__ = [
    "FINISH_REASON_MAX_TOKENS",
    "SDK_MAX_RETRIES",
    "AdapterResponse",
    "describe_sampling_wire",
    "ClientFactory",
    "CredentialMaterial",
    "ProviderExecutionAdapter",
    "ProviderExecutionError",
    "build_safe_http_client",
    "map_provider_error",
    "normalize_finish_reason",
    "optional_nonnegative_float",
    "provider_http_timeout",
    "provider_read_timeout_seconds",
    "usage_from_fields",
    "value_at",
]
