"""Application service for safe Analytics event acceptance and auditing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from .models import (
    ALLOWED_SERVER_EVENT_NAMES,
    EVENT_GROUP_BY_NAME,
    AnalyticsEventRecord,
    AppendEventResult,
    FrontendAnalyticsEvent,
    RequestAnalyticsContext,
)
from .repository import analytics_store
from .repository_common import positive_user_id


def _aware_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return value.astimezone(timezone.utc)


def _required_source_event_id(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("source_event_id must be a trimmed non-empty string")
    if len(value) > 200:
        raise ValueError("source_event_id must be at most 200 characters")
    return value


class AnalyticsService:
    """Normalize trusted identities and safe event metadata before storage."""

    def __init__(self, store):
        self.store = store

    def accept_frontend_event(
        self,
        *,
        user: dict,
        payload: FrontendAnalyticsEvent,
        context: RequestAnalyticsContext,
        received_at: datetime | None = None,
    ) -> AppendEventResult:
        user_id = positive_user_id(user.get("id"))
        if not isinstance(payload, FrontendAnalyticsEvent):
            payload = FrontendAnalyticsEvent.model_validate(payload)
        if not isinstance(context, RequestAnalyticsContext):
            context = RequestAnalyticsContext.model_validate(context)
        received = _aware_utc(
            received_at or datetime.now(timezone.utc),
            "received_at",
        )
        occurred = _aware_utc(payload.occurred_at, "occurred_at")
        if occurred < received - timedelta(hours=24):
            raise ValueError("occurred_at is too old")
        if occurred > received + timedelta(minutes=5):
            raise ValueError("occurred_at is in the future")
        event = AnalyticsEventRecord(
            event_id=payload.event_id,
            schema_version=payload.schema_version,
            event_name=payload.event_name,
            event_group=EVENT_GROUP_BY_NAME[payload.event_name],
            user_id=user_id,
            session_id=payload.session_id,
            occurred_at=occurred,
            received_at=received,
            event_source="frontend",
            page_view=payload.page_view,
            country_code=context.country_code,
            device_category=context.device_category,
            browser_family=context.browser_family,
            network_hash=context.network_hash,
            properties=payload.properties,
        )
        return self.store.append_event(event)

    def record_server_event(
        self,
        *,
        event_name: str,
        user_id: int,
        source_event_id: str,
        source_record_type: str | None = None,
        source_record_id: str | None = None,
        correlation_id: str | None = None,
        session_id: str | None = None,
        provider_id: str | None = None,
        model_id: str | None = None,
        billing_mode: str | None = None,
        outcome: str | None = None,
        error_category: str | None = None,
        properties: dict[str, Any] | None = None,
        occurred_at: datetime | None = None,
        received_at: datetime | None = None,
    ) -> AppendEventResult:
        if event_name not in ALLOWED_SERVER_EVENT_NAMES:
            raise ValueError("event_name must be a supported server event name")
        subject_id = positive_user_id(user_id)
        source_id = _required_source_event_id(source_event_id)
        received = _aware_utc(
            received_at or datetime.now(timezone.utc),
            "received_at",
        )
        occurred = _aware_utc(occurred_at or received, "occurred_at")
        event = AnalyticsEventRecord(
            event_id=str(uuid4()),
            schema_version=1,
            event_name=event_name,
            event_group=EVENT_GROUP_BY_NAME[event_name],
            user_id=subject_id,
            session_id=session_id,
            occurred_at=occurred,
            received_at=received,
            event_source="server",
            source_event_id=source_id,
            source_record_type=source_record_type,
            source_record_id=source_record_id,
            correlation_id=correlation_id,
            provider_id=provider_id,
            model_id=model_id,
            billing_mode=billing_mode,
            outcome=outcome,
            error_category=error_category,
            properties=properties or {},
        )
        return self.store.append_event(event)

    def try_record_server_event(self, **kwargs) -> AppendEventResult | None:
        try:
            return self.record_server_event(**kwargs)
        except Exception as exc:
            event_name = kwargs.get("event_name")
            safe_event = (
                event_name
                if isinstance(event_name, str)
                and event_name in ALLOWED_SERVER_EVENT_NAMES
                else "unknown"
            )
            category = type(exc).__name__[:80]
            print(
                "WARNING: analytics.append_failed "
                f"event={safe_event} category={category}"
            )
            return None

    def set_subject_exclusion(
        self,
        *,
        actor: dict,
        user_id: int,
        excluded: bool,
        reason: str,
    ) -> dict[str, Any]:
        if actor.get("role") != "admin":
            raise PermissionError("admin required")
        return self.store.set_subject_exclusion(
            positive_user_id(user_id),
            excluded=excluded,
            actor_user_id=positive_user_id(actor.get("id"), "actor_user_id"),
            reason=reason,
        )

    def record_admin_profile_access(
        self,
        *,
        actor: dict,
        subject_user_id: int,
        section: str,
    ) -> dict[str, Any]:
        if actor.get("role") != "admin":
            raise PermissionError("admin required")
        return self.store.record_admin_access(
            positive_user_id(actor.get("id"), "admin_user_id"),
            positive_user_id(subject_user_id, "subject_user_id"),
            section,
        )


analytics_service = AnalyticsService(store=analytics_store)


def get_analytics_service() -> AnalyticsService:
    return analytics_service
