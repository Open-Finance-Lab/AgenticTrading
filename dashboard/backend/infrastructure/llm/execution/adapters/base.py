"""Execution adapter protocol and safe provider-network helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
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
    RetryHint,
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

# Provider timeouts, and why the SDKs never retry.
#
# Both SDKs (openai 1.101 ``_base_client.py:963-1000``, anthropic 0.95 the
# same Stainless loop) default to ``max_retries=2`` behind ONE
# ``except httpx.TimeoutException`` that cannot tell a read timeout -- a
# whole generation in flight -- from a connect timeout, and they send no
# idempotency key (``_idempotency_header = None``). Every replay is a fresh,
# billable generation. Run agent_20260928_024706_cbda3555 shows the cost:
# with a 60s read timeout its calls took 112/100/58/176s (a 60s abandoned
# generation plus a regenerated one) until one took 185s = 3 x 60s and failed
# as ``provider_timeout``. A non-streaming provider sends no byte until the
# completion is done, so for this traffic the read timeout is a
# whole-generation deadline, and 60s sat just under what DeepSeek V4 needs
# to fill a 2000-token ceiling at its slowest healthy rate (~33 tok/s).
#
# So: ``SDK_MAX_RETRIES = 0`` on every SDK client, passed with an explicit
# ``timeout=`` (the SDKs adopt an http_client's timeout only when it differs
# from httpx's default -- do not lean on that). ``LLMExecutionService`` is the
# only retry owner: it repeats a failed attempt at the same provider only when
# ``map_provider_error`` says nothing was generated (``RetryHint``), and gives
# every repeat its own reservation row. Do not re-enable SDK retries, and
# never retry a read timeout.
SDK_MAX_RETRIES = 0
_CONNECT_TIMEOUT_SECONDS = 8.0
_WRITE_TIMEOUT_SECONDS = 60.0
_POOL_TIMEOUT_SECONDS = 60.0
# 180s: roughly today's per-candidate worst case (3 x 60s + backoff ~= 185s),
# so a hung call never waits longer than it did -- it just stops paying for
# three generations. It covers the 4096-token recovery ceiling down to about
# 23 tok/s. Tune per deployment with LLM_PROVIDER_READ_TIMEOUT_SECONDS.
_DEFAULT_PROVIDER_READ_TIMEOUT_SECONDS = 180
_MIN_PROVIDER_READ_TIMEOUT_SECONDS = 30
_MAX_PROVIDER_READ_TIMEOUT_SECONDS = 600
_RETRY_AFTER_HEADER_MAX_LENGTH = 32


def _parse_provider_read_timeout(raw: str | None) -> int:
    """Parse LLM_PROVIDER_READ_TIMEOUT_SECONDS; never raise.

    This module is imported at web boot (``backtests.py`` -> ``service.py``),
    and an unparseable env value read with a bare ``int()`` at module scope has
    killed app boot in this repo before. Junk and out-of-range values warn and
    fall back; the range rejects a dropped or doubled digit ("18", "1800").
    """

    default = _DEFAULT_PROVIDER_READ_TIMEOUT_SECONDS
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        print(
            "WARNING: LLM_PROVIDER_READ_TIMEOUT_SECONDS is not an integer "
            f"({raw!r}); using {default}",
            flush=True,
        )
        return default
    if not (
        _MIN_PROVIDER_READ_TIMEOUT_SECONDS
        <= value
        <= _MAX_PROVIDER_READ_TIMEOUT_SECONDS
    ):
        print(
            "WARNING: LLM_PROVIDER_READ_TIMEOUT_SECONDS is out of range "
            f"({value}; allowed {_MIN_PROVIDER_READ_TIMEOUT_SECONDS}-"
            f"{_MAX_PROVIDER_READ_TIMEOUT_SECONDS}); using {default}",
            flush=True,
        )
        return default
    return value


PROVIDER_READ_TIMEOUT_SECONDS = _parse_provider_read_timeout(
    os.getenv("LLM_PROVIDER_READ_TIMEOUT_SECONDS")
)


def provider_read_timeout_seconds() -> int:
    # Read at call time so tests can monkeypatch the global. Never
    # ``importlib.reload`` this module: that mints a second
    # ProviderExecutionError class the adapters' except clauses do not match.
    return PROVIDER_READ_TIMEOUT_SECONDS


def provider_http_timeout() -> httpx.Timeout:
    return httpx.Timeout(
        connect=_CONNECT_TIMEOUT_SECONDS,
        read=float(provider_read_timeout_seconds()),
        write=_WRITE_TIMEOUT_SECONDS,
        pool=_POOL_TIMEOUT_SECONDS,
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


_TIMEOUT_PHASES: tuple[tuple[type[BaseException], str], ...] = (
    (httpx.ConnectTimeout, "connect"),
    (httpx.ReadTimeout, "read"),
    (httpx.WriteTimeout, "write"),
    (httpx.PoolTimeout, "pool"),
)
# Nothing reached the provider in these phases, so nothing was generated.
_PRE_SEND_TIMEOUT_PHASES = frozenset({"connect", "write", "pool"})


def _exception_chain(exc: BaseException, depth: int = 4) -> tuple[BaseException, ...]:
    """``exc`` and its causes: both SDKs keep the httpx error as ``__cause__``."""

    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and len(chain) < depth and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    return tuple(chain)


def _timeout_phase(chain: tuple[BaseException, ...]) -> str | None:
    for item in chain:
        for exc_type, phase in _TIMEOUT_PHASES:
            if isinstance(item, exc_type):
                return phase
    return None


def _response_headers(exc: BaseException) -> Any:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    return headers if callable(getattr(headers, "get", None)) else {}


def _header_value(headers: Any, name: str) -> str | None:
    try:
        value = headers.get(name)
    except Exception:  # noqa: BLE001 - a malformed header map is just absent
        return None
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if 0 < len(value) <= _RETRY_AFTER_HEADER_MAX_LENGTH else None


def _retry_after_seconds(headers: Any) -> float | None:
    """``retry-after-ms`` wins, then a numeric ``retry-after``; an HTTP-date is ignored."""

    for name, scale in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        raw = _header_value(headers, name)
        if raw is None:
            continue
        try:
            value = float(raw) / scale
        except ValueError:
            return None
        return value if math.isfinite(value) and value >= 0 else None
    return None


def _status_retry_hint(status: int, headers: Any) -> RetryHint:
    should_retry = (_header_value(headers, "x-should-retry") or "").lower()
    if should_retry == "false":
        return RetryHint.NONE
    if should_retry == "true":
        return RetryHint.REJECTED
    if status in {408, 409, 429} or status >= 500:
        return RetryHint.REJECTED
    return RetryHint.NONE


def map_provider_error(exc: Exception) -> ProviderExecutionError:
    # Category order is load-bearing and predates the retry hints: timeout
    # first, then credential, quota, and everything else unavailable. The
    # hints only say whether ``LLMExecutionService`` may repeat the attempt
    # (see ``RetryHint``); they never change which category an error gets.
    chain = _exception_chain(exc)
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)) or "timeout" in type(exc).__name__.lower():
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
    "PROVIDER_READ_TIMEOUT_SECONDS",
    "SDK_MAX_RETRIES",
    "AdapterResponse",
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
