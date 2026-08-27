"""Analytics foundation service tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from dashboard.backend.domain.analytics.models import (
    AppendEventResult,
    FrontendAnalyticsEvent,
    RequestAnalyticsContext,
)
from dashboard.backend.domain.analytics.service import AnalyticsService


NOW = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)


class RecordingStore:
    def __init__(self):
        self.events = []
        self.exclusions = []
        self.accesses = []

    def append_event(self, event):
        self.events.append(event)
        return AppendEventResult(event=event, created=True)

    def set_subject_exclusion(self, user_id, **kwargs):
        value = {"user_id": user_id, **kwargs}
        self.exclusions.append(value)
        return value

    def record_admin_access(self, admin_user_id, subject_user_id, section):
        value = {
            "sequence": len(self.accesses) + 1,
            "admin_user_id": admin_user_id,
            "subject_user_id": subject_user_id,
            "section": section,
            "accessed_at": NOW.isoformat(),
        }
        self.accesses.append(value)
        return value


class FailingStore:
    def append_event(self, event):
        raise RuntimeError("synthetic-secret-canary from provider body")


def frontend_event(**overrides):
    value = {
        "event_id": str(uuid4()),
        "schema_version": 1,
        "event_name": "page_viewed",
        "session_id": str(uuid4()),
        "occurred_at": NOW,
        "page_view": "home",
        "properties": {},
    }
    value.update(overrides)
    return FrontendAnalyticsEvent.model_validate(value)


def safe_context():
    return RequestAnalyticsContext(
        country_code="US",
        device_category="desktop",
        browser_family="Chrome",
        network_hash="a" * 64,
    )


def test_frontend_event_uses_authenticated_identity_and_server_received_time():
    store = RecordingStore()
    service = AnalyticsService(store=store)
    result = service.accept_frontend_event(
        user={
            "id": 42,
            "role": "user",
            "email": "must-not-persist@example.test",
        },
        payload=frontend_event(),
        context=safe_context(),
        received_at=NOW + timedelta(seconds=1),
    )
    assert result.event.user_id == 42
    assert result.event.event_group == "experience"
    assert result.event.event_source == "frontend"
    assert result.event.received_at == NOW + timedelta(seconds=1)
    assert "email" not in result.event.model_dump()
    assert store.events == [result.event]


@pytest.mark.parametrize(
    "occurred_at",
    [NOW - timedelta(hours=24, seconds=1), NOW + timedelta(minutes=5, seconds=1)],
)
def test_frontend_event_rejects_large_clock_skew(occurred_at):
    service = AnalyticsService(store=RecordingStore())
    with pytest.raises(ValueError, match="occurred_at"):
        service.accept_frontend_event(
            user={"id": 42, "role": "user"},
            payload=frontend_event(occurred_at=occurred_at),
            context=safe_context(),
            received_at=NOW,
        )


def test_server_event_requires_server_name_and_deterministic_source_id():
    service = AnalyticsService(store=RecordingStore())
    with pytest.raises(ValueError, match="server event name"):
        service.record_server_event(
            event_name="page_viewed",
            user_id=42,
            source_event_id="page:unsafe",
        )
    with pytest.raises(ValueError, match="source_event_id"):
        service.record_server_event(
            event_name="agent_created",
            user_id=42,
            source_event_id="",
        )


def test_server_usage_event_is_normalized_for_future_instrumentation():
    store = RecordingStore()
    service = AnalyticsService(store=store)
    result = service.record_server_event(
        event_name="model_usage_recorded",
        user_id=42,
        source_event_id="usage:run-1:0",
        source_record_type="run",
        source_record_id="run-1",
        correlation_id="run-1",
        provider_id="openrouter",
        model_id="openai/gpt-5.5",
        billing_mode="platform_credits",
        outcome="succeeded",
        properties={
            "input_tokens": 100,
            "output_tokens": 50,
            "cost_micro_usd": 375,
        },
        occurred_at=NOW,
        received_at=NOW + timedelta(seconds=1),
    )
    assert result.event.event_group == "resource"
    assert result.event.event_source == "server"
    assert result.event.properties["cost_micro_usd"] == 375
    assert result.event.network_hash is None


def test_best_effort_server_event_never_raises_or_prints_sensitive_fields(capsys):
    failing = AnalyticsService(store=FailingStore())
    result = failing.try_record_server_event(
        event_name="credential_verified",
        user_id=42,
        source_event_id="credential:synthetic-id:verified",
        source_record_type="credential",
        source_record_id="synthetic-id",
        properties={},
    )
    assert result is None
    output = capsys.readouterr().out
    assert "synthetic-secret-canary" not in output
    assert "synthetic-id" not in output
    assert "42" not in output
    assert "analytics.append_failed" in output
    assert "category=RuntimeError" in output


def test_subject_exclusion_and_profile_access_require_admin_actor():
    store = RecordingStore()
    service = AnalyticsService(store=store)
    user = {"id": 9, "role": "user"}
    admin = {"id": 7, "role": "admin"}

    with pytest.raises(PermissionError, match="admin required"):
        service.set_subject_exclusion(
            actor=user,
            user_id=42,
            excluded=True,
            reason="Synthetic QA account.",
        )
    with pytest.raises(PermissionError, match="admin required"):
        service.record_admin_profile_access(
            actor=user,
            subject_user_id=42,
            section="overview",
        )

    setting = service.set_subject_exclusion(
        actor=admin,
        user_id=42,
        excluded=True,
        reason="Synthetic QA account.",
    )
    access = service.record_admin_profile_access(
        actor=admin,
        subject_user_id=42,
        section="overview",
    )
    assert setting["actor_user_id"] == 7
    assert access["admin_user_id"] == 7
    assert access["subject_user_id"] == 42
