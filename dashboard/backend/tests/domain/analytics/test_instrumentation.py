"""Best-effort server-authoritative Analytics instrumentation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from dashboard.backend.domain.analytics import instrumentation
from dashboard.backend.domain.analytics.models import AppendEventResult


NOW = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)


class RecordingService:
    def __init__(self):
        self.calls: list[dict] = []

    def try_record_server_event(self, **kwargs):
        self.calls.append(kwargs)
        return AppendEventResult.model_construct(event=None, created=True)


def _service(monkeypatch) -> RecordingService:
    service = RecordingService()
    monkeypatch.setattr(instrumentation, "get_analytics_service", lambda: service)
    monkeypatch.setattr(instrumentation, "_snapshot_recalculator", lambda _user_id: None)
    return service


def test_credential_event_contains_only_safe_envelope_fields(monkeypatch):
    service = _service(monkeypatch)

    instrumentation.emit_credential_event(
        event_name="credential_verified",
        user_id=7,
        credential_id="cred-1",
        provider_id="openrouter",
        occurred_at=NOW,
        version="2026-08-26T12:00:00+00:00",
    )

    call = service.calls[0]
    assert call["event_name"] == "credential_verified"
    assert call["source_event_id"].startswith(
        "credential:credential_verified:cred-1:"
    )
    assert call["source_record_type"] == "credential"
    assert call["source_record_id"] == "cred-1"
    assert call["provider_id"] == "openrouter"
    assert call["properties"] == {}
    assert "api_key" not in repr(call)


def test_server_event_failure_never_escapes_or_prints_exception_text(
    monkeypatch,
    capsys,
):
    class FailingService:
        def try_record_server_event(self, **kwargs):
            raise RuntimeError("synthetic-secret-canary from provider body")

    monkeypatch.setattr(
        instrumentation,
        "get_analytics_service",
        lambda: FailingService(),
    )

    instrumentation.emit_run_event(
        event_name="backtest_requested",
        user_id=7,
        run_id="run-1",
        occurred_at=NOW,
    )

    output = capsys.readouterr().out
    assert "synthetic-secret-canary" not in output
    assert "run-1" not in output
    assert "analytics.instrumentation_failed" in output
    assert "category=RuntimeError" in output


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (TimeoutError(), "provider_timeout"),
        (PermissionError(), "credential_invalid"),
        (LookupError(), "credential_missing"),
        (ConnectionError(), "provider_unavailable"),
        (RuntimeError(), "internal_error"),
    ],
)
def test_safe_error_event_uses_stable_categories(
    monkeypatch,
    exc,
    expected,
):
    service = _service(monkeypatch)

    instrumentation.emit_safe_error_event(
        user_id=7,
        source_record_type="run",
        source_record_id="run-1",
        exc=exc,
        occurred_at=NOW,
    )

    call = service.calls[0]
    assert call["event_name"] == "safe_error_recorded"
    assert call["error_category"] == expected
    assert call["properties"] == {}


def test_model_usage_requires_exact_bounded_usage_properties(monkeypatch):
    service = _service(monkeypatch)

    instrumentation.emit_resource_event(
        event_name="model_usage_recorded",
        user_id=7,
        source_record_type="run",
        source_record_id="run-1",
        correlation_id="run-1",
        provider_id="openrouter",
        model_id="openai/gpt-5.5",
        billing_mode="byok",
        outcome="succeeded",
        properties={
            "input_tokens": 100,
            "output_tokens": 50,
            "cost_micro_usd": 0,
        },
        version="3",
        occurred_at=NOW,
    )

    call = service.calls[0]
    assert call["source_event_id"] == "resource:model_usage_recorded:run-1:3"
    assert call["properties"] == {
        "input_tokens": 100,
        "output_tokens": 50,
        "cost_micro_usd": 0,
    }


def test_unsupported_event_name_is_contained(monkeypatch, capsys):
    service = _service(monkeypatch)

    instrumentation.emit_agent_event(
        event_name="agent_prompt_recorded",
        user_id=7,
        agent_id="agent-1",
        occurred_at=NOW,
    )

    assert service.calls == []
    assert "category=ValueError" in capsys.readouterr().out


def test_stored_event_recalculates_snapshot_best_effort(monkeypatch):
    _service(monkeypatch)
    users = []
    monkeypatch.setattr(
        instrumentation,
        "_snapshot_recalculator",
        lambda user_id: users.append(user_id),
    )

    instrumentation.emit_agent_event(
        event_name="agent_created",
        user_id=7,
        agent_id="agent-1",
        occurred_at=NOW,
    )

    assert users == [7]


def test_snapshot_failure_never_escapes_or_logs_exception_text(
    monkeypatch,
    capsys,
):
    _service(monkeypatch)

    def fail(_user_id):
        raise RuntimeError("snapshot-secret-canary")

    monkeypatch.setattr(instrumentation, "_snapshot_recalculator", fail)

    instrumentation.emit_agent_event(
        event_name="agent_created",
        user_id=7,
        agent_id="agent-1",
        occurred_at=NOW,
    )

    output = capsys.readouterr().out
    assert "snapshot-secret-canary" not in output
    assert "category=RuntimeError" in output
