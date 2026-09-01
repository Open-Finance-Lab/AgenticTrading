"""Bounded, provider-neutral mapping for upstream execution errors."""

import json
from types import SimpleNamespace

import pytest

from dashboard.backend.infrastructure.llm.execution.adapters.base import (
    map_provider_error,
)
from dashboard.backend.infrastructure.llm.execution.errors import (
    ExecutionErrorCategory,
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
