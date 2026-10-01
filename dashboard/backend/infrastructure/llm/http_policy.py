"""Timeout and retry policy for every LLM SDK client the backend builds.

A leaf on purpose: standard library and ``httpx`` only. The billed execution
layer (``execution/adapters/base.py``, ``execution/service.py``) and the two
clients outside it -- dashboard chat (``domain/chat/service.py``) and the
strategy-chat panel (``domain/backtesting/algo_service.py``) -- all read the
policy from here, so adopting it never drags the execution layer's registry,
handoff and credential code into a process that only wants a timeout.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any, TypeVar

import httpx


T = TypeVar("T")


class RetryHint(StrEnum):
    """Whether one failed provider attempt may be repeated against that provider.

    ``PRE_SEND`` — the request never reached the provider (DNS, connect, TLS,
    a stalled request body): repeating it cannot regenerate anything.
    ``REJECTED`` — the provider answered with a status that says "not now"
    (408/409/429/5xx, ``x-should-retry: true``) or dropped the connection; it
    may already have done work, so it is repeated only when it failed fast.
    ``NONE`` — never repeat: above all a read timeout, where a whole
    generation was in flight and, with no idempotency key, is billed again on
    every replay.
    """

    PRE_SEND = "pre_send"
    REJECTED = "rejected"
    NONE = "none"


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
# from httpx's default -- do not lean on that). Setting it is only half the
# policy: the caller must own the retries the SDK no longer makes, through
# ``same_provider_retry_delay`` -- ``LLMExecutionService`` for billed runs,
# ``call_with_retries``/``acall_with_retries`` for everything else. Without
# that owner, ``max_retries=0`` also drops the free retries of failures that
# generated nothing (a stale keep-alive, a 529). Never retry a read timeout.
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
    # Read at call time so tests can monkeypatch THIS module's global (not a
    # re-export of it: patching ``adapters.base.PROVIDER_READ_TIMEOUT_SECONDS``
    # changes nothing).
    return PROVIDER_READ_TIMEOUT_SECONDS


def provider_http_timeout() -> httpx.Timeout:
    return httpx.Timeout(
        connect=_CONNECT_TIMEOUT_SECONDS,
        read=float(provider_read_timeout_seconds()),
        write=_WRITE_TIMEOUT_SECONDS,
        pool=_POOL_TIMEOUT_SECONDS,
    )


# Same-provider retries. Repeat an attempt at the same provider (and model)
# only when the error says nothing was generated:
#   - ``RetryHint.PRE_SEND`` (DNS, connect, TLS, a stalled body), at any age;
#   - ``RetryHint.REJECTED`` (408/409/429/5xx, a dropped connection) only when
#     it came back within FAST_FAILURE_SECONDS. CommonStack's 2026-09-27 500s
#     arrived ~24s in, after the generation, and three SDK replays of them
#     bought nothing but 73s; a gateway that refuses outright answers in <2s,
#     and the fastest DeepSeek completion seen is ~24s. The gate is calibrated
#     on DeepSeek: a model that can finish in under 15s (Haiku, GPT-5.5) can
#     have a fast post-generation 5xx or cut-off body repeated. That is still
#     fewer replays than the SDK made unconditionally.
# A read timeout is never repeated.
MAX_SAME_PROVIDER_RETRIES = 2
SAME_PROVIDER_BACKOFF_SECONDS = (4.0, 12.0)
FAST_FAILURE_SECONDS = 15.0
# A stated Retry-After is waited out up to the SDKs' own ceiling (openai and
# anthropic ``_calculate_retry_timeout`` honour <= 60s), so no refusal the SDK
# used to wait out now fails the call. Above it the provider is saying "not
# this minute", and repeating early would only be refused again.
RETRY_AFTER_CAP_SECONDS = 60.0


def same_provider_retry_delay(
    hint: RetryHint | str,
    *,
    elapsed_seconds: float | None,
    retry_after_seconds: float | None,
    retries_done: int,
) -> float | None:
    """Seconds to wait before repeating a failed attempt, or None to stop."""

    if retries_done >= MAX_SAME_PROVIDER_RETRIES:
        return None
    hint = RetryHint(hint)
    if hint is RetryHint.PRE_SEND:
        pass
    elif (
        hint is RetryHint.REJECTED
        and isinstance(elapsed_seconds, float)
        and elapsed_seconds <= FAST_FAILURE_SECONDS
    ):
        pass
    else:
        # A read timeout, a slow rejection, or anything unclassified.
        return None
    delay = SAME_PROVIDER_BACKOFF_SECONDS[retries_done]
    if retry_after_seconds is not None:
        if retry_after_seconds > RETRY_AFTER_CAP_SECONDS:
            return None
        delay = max(delay, retry_after_seconds)
    return delay


# --- Reading an SDK/httpx exception. Pure functions of the exception object;
# --- never of its message text, which can carry upstream data.

_TIMEOUT_PHASES: tuple[tuple[type[BaseException], str], ...] = (
    (httpx.ConnectTimeout, "connect"),
    (httpx.ReadTimeout, "read"),
    (httpx.WriteTimeout, "write"),
    (httpx.PoolTimeout, "pool"),
)
# Nothing reached the provider in these phases, so nothing was generated.
PRE_SEND_TIMEOUT_PHASES = frozenset({"connect", "write", "pool"})

PROVIDER_ERROR_PAYLOAD_MAX_BYTES = 4096
QUOTA_ERROR_IDENTIFIERS = frozenset(
    {
        "in_flight_budget_exhausted",
        "insufficient_quota",
        "quota_exceeded",
        "quota_exhausted",
        "insufficient_balance",
        "credit_balance_exhausted",
    }
)
QUOTA_ERROR_PHRASES = (
    "insufficient balance",
    "insufficient credits",
    "quota exceeded",
    "quota exhausted",
    "exceeded your current quota",
    "not enough credits",
)


def exception_chain(exc: BaseException, depth: int = 4) -> tuple[BaseException, ...]:
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


def is_timeout_error(exc: BaseException) -> bool:
    return (
        isinstance(exc, (TimeoutError, httpx.TimeoutException))
        or "timeout" in type(exc).__name__.lower()
    )


def timeout_phase(chain: tuple[BaseException, ...]) -> str | None:
    for item in chain:
        for exc_type, phase in _TIMEOUT_PHASES:
            if isinstance(item, exc_type):
                return phase
    return None


def provider_status_codes(exc: BaseException) -> tuple[int, ...]:
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


def bounded_error_payload(exc: BaseException) -> dict[str, Any]:
    """Parse only a small structured provider error body, if one is present."""

    response = getattr(exc, "response", None)
    content = getattr(response, "content", b"")
    if isinstance(content, str):
        content = content.encode("utf-8", errors="ignore")
    elif isinstance(content, bytearray):
        content = bytes(content)
    if not isinstance(content, bytes) or len(content) > PROVIDER_ERROR_PAYLOAD_MAX_BYTES:
        return {}
    try:
        parsed = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def structured_quota_signal(payload: dict[str, Any]) -> bool:
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
        if normalized in QUOTA_ERROR_IDENTIFIERS:
            return True
    for value in messages:
        if isinstance(value, str) and any(
            phrase in value.strip().lower() for phrase in QUOTA_ERROR_PHRASES
        ):
            return True
    return False


def response_headers(exc: BaseException) -> Any:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    return headers if callable(getattr(headers, "get", None)) else {}


def header_value(headers: Any, name: str) -> str | None:
    try:
        value = headers.get(name)
    except Exception:  # noqa: BLE001 - a malformed header map is just absent
        return None
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if 0 < len(value) <= _RETRY_AFTER_HEADER_MAX_LENGTH else None


def retry_after_seconds(headers: Any) -> float | None:
    """``retry-after-ms`` wins, then a numeric ``retry-after``; an HTTP-date is ignored."""

    for name, scale in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        raw = header_value(headers, name)
        if raw is None:
            continue
        try:
            value = float(raw) / scale
        except ValueError:
            continue
        if math.isfinite(value) and value >= 0:
            return value
    return None


def status_retry_hint(status: int, headers: Any) -> RetryHint:
    should_retry = (header_value(headers, "x-should-retry") or "").lower()
    if should_retry == "false":
        return RetryHint.NONE
    if should_retry == "true":
        return RetryHint.REJECTED
    if status in {408, 409, 429} or status >= 500:
        return RetryHint.REJECTED
    return RetryHint.NONE


def sdk_error_retry_hint(exc: BaseException) -> tuple[RetryHint, float | None]:
    """``(hint, retry_after_seconds)`` for an exception raised by an SDK call.

    The retry half of ``adapters.base.map_provider_error``, for callers that
    hold a raw SDK exception rather than an adapter error. It agrees with that
    mapper's ``retry_hint`` on every SDK-shaped exception (pinned by
    ``test_chat_algo_client_policy.py``); it does not know the execution
    layer's own address-policy errors, which only its pinned transport raises.
    """

    chain = exception_chain(exc)
    if is_timeout_error(exc):
        phase = timeout_phase(chain)
        hint = RetryHint.PRE_SEND if phase in PRE_SEND_TIMEOUT_PHASES else RetryHint.NONE
        return hint, None
    statuses = provider_status_codes(exc)
    if any(status in {401, 402, 403} for status in statuses):
        return RetryHint.NONE, None
    if not statuses or any(400 <= status < 500 for status in statuses):
        if structured_quota_signal(bounded_error_payload(exc)):
            return RetryHint.NONE, None
    if statuses:
        headers = response_headers(exc)
        return status_retry_hint(statuses[0], headers), retry_after_seconds(headers)
    if any(isinstance(item, httpx.ConnectError) for item in chain):
        return RetryHint.PRE_SEND, None
    if any(
        isinstance(item, (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError))
        for item in chain
    ):
        return RetryHint.REJECTED, None
    return RetryHint.NONE, None


def _retry_delay_for(exc: BaseException, elapsed: float, retries_done: int) -> float | None:
    hint, retry_after = sdk_error_retry_hint(exc)
    return same_provider_retry_delay(
        hint,
        elapsed_seconds=elapsed,
        retry_after_seconds=retry_after,
        retries_done=retries_done,
    )


def _report_retry(label: str, exc: BaseException, delay: float, retries_done: int) -> None:
    # Type name only: an SDK exception's message can echo upstream data.
    print(
        f"{label}: {type(exc).__name__}; repeating at the same model in "
        f"{delay:.0f}s (retry {retries_done + 1}/{MAX_SAME_PROVIDER_RETRIES})",
        flush=True,
    )


def call_with_retries(
    call: Callable[[], T],
    *,
    label: str,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> T:
    """Run one SDK call, owning the retries ``SDK_MAX_RETRIES`` turned off."""

    retries_done = 0
    while True:
        started = clock()
        try:
            return call()
        except Exception as exc:
            delay = _retry_delay_for(exc, clock() - started, retries_done)
            if delay is None:
                raise
            _report_retry(label, exc, delay, retries_done)
        sleep(delay)
        retries_done += 1


async def acall_with_retries(
    call: Callable[[], Awaitable[T]],
    *,
    label: str,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> T:
    """Async twin of ``call_with_retries``."""

    retries_done = 0
    while True:
        started = clock()
        try:
            return await call()
        except Exception as exc:
            delay = _retry_delay_for(exc, clock() - started, retries_done)
            if delay is None:
                raise
            _report_retry(label, exc, delay, retries_done)
        await sleep(delay)
        retries_done += 1


__all__ = [
    "FAST_FAILURE_SECONDS",
    "MAX_SAME_PROVIDER_RETRIES",
    "PROVIDER_READ_TIMEOUT_SECONDS",
    "RETRY_AFTER_CAP_SECONDS",
    "SAME_PROVIDER_BACKOFF_SECONDS",
    "SDK_MAX_RETRIES",
    "RetryHint",
    "acall_with_retries",
    "call_with_retries",
    "provider_http_timeout",
    "provider_read_timeout_seconds",
    "same_provider_retry_delay",
    "sdk_error_retry_hint",
]
