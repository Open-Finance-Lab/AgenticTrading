"""Bounded, provider-neutral mapping for upstream execution errors."""

import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from dashboard.backend.infrastructure.llm.adapters.safe_http import (
    ProviderAddressResolutionError,
    UnsafeProviderAddress,
)
from dashboard.backend.infrastructure.llm.execution.adapters.base import (
    map_provider_error,
)
from dashboard.backend.infrastructure.llm.execution.errors import (
    ExecutionErrorCategory,
    RetryHint,
)


class _ProviderError(Exception):
    def __init__(
        self,
        status_code: int | None,
        payload: object,
        *,
        response_status: int | None = None,
    ):
        super().__init__("synthetic provider failure")
        content = json.dumps(payload).encode("utf-8")
        self.status_code = status_code
        self.response = SimpleNamespace(
            status_code=response_status if response_status is not None else status_code,
            content=content,
        )


def _error(
    status_code: int | None,
    payload: object,
    *,
    response_status: int | None = None,
) -> _ProviderError:
    return _ProviderError(status_code, payload, response_status=response_status)


@pytest.mark.parametrize(
    ("status_code", "payload"),
    [
        (402, {"error": {"message": "payment required"}}),
        (429, {"error": {"code": "in_flight_budget_exhausted"}}),
        (400, {"error": {"type": "insufficient_quota"}}),
        (400, {"error": {"message": "Insufficient balance for this request"}}),
        (429, {"error": {"message": "You exceeded your current quota"}}),
        (None, {"code": "quota_exhausted"}),
    ],
)
def test_explicit_balance_or_quota_errors_are_typed(status_code, payload):
    mapped = map_provider_error(_error(status_code, payload))
    assert mapped.category is ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED


def test_response_status_code_is_used_when_exception_has_no_status():
    mapped = map_provider_error(
        _error(None, {"error": {"message": "payment required"}}, response_status=402)
    )
    assert mapped.category is ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED


@pytest.mark.parametrize(
    ("status_code", "payload", "expected"),
    [
        (429, {"error": {"message": "rate limit exceeded"}}, "provider_unavailable"),
        (500, {"error": {"message": "insufficient capacity"}}, "provider_unavailable"),
        (500, {"error": {"code": "quota_exceeded"}}, "provider_unavailable"),
        (401, {"error": {"message": "invalid key"}}, "credential_invalid"),
        (401, {"error": {"code": "insufficient_quota"}}, "credential_invalid"),
        (503, {"error": {"message": "service unavailable"}}, "provider_unavailable"),
    ],
)
def test_non_quota_errors_do_not_become_failover_signals(
    status_code, payload, expected
):
    mapped = map_provider_error(_error(status_code, payload))
    assert mapped.category.value == expected


def test_malformed_or_oversized_payload_does_not_become_a_quota_signal():
    malformed = SimpleNamespace(status_code=429, content=b"{not-json")
    malformed_error = _ProviderError.__new__(_ProviderError)
    Exception.__init__(malformed_error, "synthetic provider failure")
    malformed_error.status_code = 429
    malformed_error.response = malformed
    assert (
        map_provider_error(malformed_error).category
        is ExecutionErrorCategory.PROVIDER_UNAVAILABLE
    )
    malformed_402 = _ProviderError.__new__(_ProviderError)
    Exception.__init__(malformed_402, "synthetic provider failure")
    malformed_402.status_code = 402
    malformed_402.response = SimpleNamespace(status_code=402, content=b"{not-json")
    assert (
        map_provider_error(malformed_402).category
        is ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED
    )

    oversized = _error(
        429,
        {"error": {"message": "x" * 4096 + " quota exceeded"}},
    )
    assert (
        map_provider_error(oversized).category
        is ExecutionErrorCategory.PROVIDER_UNAVAILABLE
    )


def test_quota_looking_text_outside_structured_error_fields_is_ignored():
    mapped = map_provider_error(
        _error(429, {"details": {"message": "insufficient balance"}})
    )
    assert mapped.category is ExecutionErrorCategory.PROVIDER_UNAVAILABLE


def test_timeout_precedes_quota_classification():
    class _TimeoutProviderError(TimeoutError):
        status_code = 402
        response = SimpleNamespace(status_code=402, content=b"{}")

    exc = _TimeoutProviderError("quota exceeded")
    assert map_provider_error(exc).category is ExecutionErrorCategory.PROVIDER_TIMEOUT


# --- Retry hints: whether LLMExecutionService may repeat the attempt --------
# The category of every error above is unchanged by the hints; these pin the
# second, independent axis the service's same-provider retry reads.

_WIRE_REQUEST = httpx.Request("POST", "https://api.commonstack.ai/v1/chat/completions")


def _raised_from(outer: BaseException, inner: BaseException) -> BaseException:
    """``outer`` raised ``from inner``, as both SDKs raise their wrappers."""
    try:
        try:
            raise inner
        except BaseException as cause:
            raise outer from cause
    except BaseException as raised:
        return raised


def _status_error(status: int, headers: dict[str, str] | None = None, body=None):
    response = httpx.Response(
        status, headers=headers or {}, json=body or {}, request=_WIRE_REQUEST
    )
    return openai.APIStatusError("synthetic", response=response, body=body)


@pytest.mark.parametrize(
    ("exc", "phase", "hint"),
    [
        (httpx.ReadTimeout("x"), "read", RetryHint.NONE),
        (
            _raised_from(openai.APITimeoutError(request=_WIRE_REQUEST), httpx.ReadTimeout("x")),
            "read",
            RetryHint.NONE,
        ),
        (httpx.ConnectTimeout("x"), "connect", RetryHint.PRE_SEND),
        (
            _raised_from(openai.APITimeoutError(request=_WIRE_REQUEST), httpx.ConnectTimeout("x")),
            "connect",
            RetryHint.PRE_SEND,
        ),
        (httpx.WriteTimeout("x"), "write", RetryHint.PRE_SEND),
        (httpx.PoolTimeout("x"), "pool", RetryHint.PRE_SEND),
        (TimeoutError("x"), None, RetryHint.NONE),
        (openai.APITimeoutError(request=_WIRE_REQUEST), None, RetryHint.NONE),
    ],
)
def test_timeout_phase_and_hint_from_exception_chain(exc, phase, hint):
    mapped = map_provider_error(exc)
    assert mapped.category is ExecutionErrorCategory.PROVIDER_TIMEOUT
    assert mapped.timeout_phase == phase
    assert mapped.retry_hint is hint


@pytest.mark.parametrize(
    ("exc", "hint"),
    [
        (
            _raised_from(openai.APIConnectionError(request=_WIRE_REQUEST), httpx.ConnectError("x")),
            RetryHint.PRE_SEND,
        ),
        (httpx.ConnectError("x"), RetryHint.PRE_SEND),
        (ProviderAddressResolutionError("dns"), RetryHint.PRE_SEND),
        (
            _raised_from(
                openai.APIConnectionError(request=_WIRE_REQUEST),
                httpx.RemoteProtocolError("x"),
            ),
            RetryHint.REJECTED,
        ),
        (
            _raised_from(openai.APIConnectionError(request=_WIRE_REQUEST), httpx.ReadError("x")),
            RetryHint.REJECTED,
        ),
        (httpx.WriteError("x"), RetryHint.REJECTED),
        (UnsafeProviderAddress("private"), RetryHint.NONE),
        (openai.APIConnectionError(request=_WIRE_REQUEST), RetryHint.NONE),
        (RuntimeError("adapter bug"), RetryHint.NONE),
    ],
)
def test_transport_failures_before_and_after_send(exc, hint):
    mapped = map_provider_error(exc)
    assert mapped.category is ExecutionErrorCategory.PROVIDER_UNAVAILABLE
    assert mapped.retry_hint is hint
    assert mapped.provider_status_code is None


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504, 529])
def test_status_rejections_carry_status_and_hint(status):
    mapped = map_provider_error(_status_error(status))
    assert mapped.category is ExecutionErrorCategory.PROVIDER_UNAVAILABLE
    assert mapped.retry_hint is RetryHint.REJECTED
    assert mapped.provider_status_code == status


@pytest.mark.parametrize("status", [400, 404, 422])
def test_other_client_errors_are_never_repeated(status):
    mapped = map_provider_error(_status_error(status))
    assert mapped.category is ExecutionErrorCategory.PROVIDER_UNAVAILABLE
    assert mapped.retry_hint is RetryHint.NONE


@pytest.mark.parametrize(
    ("status", "body", "category"),
    [
        (401, None, ExecutionErrorCategory.CREDENTIAL_INVALID),
        (403, None, ExecutionErrorCategory.CREDENTIAL_INVALID),
        (402, None, ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED),
        (429, {"error": {"code": "insufficient_quota"}}, ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED),
        (429, {"error": {"code": "in_flight_budget_exhausted"}}, ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED),
    ],
)
def test_lane_state_errors_keep_their_category_and_no_hint(status, body, category):
    mapped = map_provider_error(_status_error(status, body=body))
    assert mapped.category is category
    assert mapped.retry_hint is RetryHint.NONE


def test_x_should_retry_header_overrides_status():
    assert (
        map_provider_error(_status_error(503, {"x-should-retry": "false"})).retry_hint
        is RetryHint.NONE
    )
    assert (
        map_provider_error(_status_error(400, {"x-should-retry": "true"})).retry_hint
        is RetryHint.REJECTED
    )


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"retry-after": "7"}, 7.0),
        ({"retry-after": "0.5"}, 0.5),
        ({"retry-after-ms": "1500", "retry-after": "9"}, 1.5),
        ({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}, None),
        ({"retry-after": "-1"}, None),
        ({"retry-after": "nan"}, None),
        ({"retry-after": "inf"}, None),
        ({"retry-after": "9" * 64}, None),
        ({}, None),
    ],
)
def test_retry_after_parsing(headers, expected):
    assert map_provider_error(_status_error(503, headers)).retry_after_seconds == expected


def test_fake_response_without_headers_maps_without_raising():
    mapped = map_provider_error(_error(503, {"error": {"message": "down"}}))
    assert mapped.category is ExecutionErrorCategory.PROVIDER_UNAVAILABLE
    assert mapped.retry_hint is RetryHint.REJECTED
    assert mapped.retry_after_seconds is None
