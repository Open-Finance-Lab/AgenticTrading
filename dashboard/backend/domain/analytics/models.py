"""Strict, versioned Analytics event models and property allowlists."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ANALYTICS_SCHEMA_VERSION = 1
MAX_PROPERTIES_BYTES = 1024

ALLOWED_FRONTEND_EVENT_NAMES = {
    "page_viewed",
    "page_hidden",
    "session_heartbeat",
}
ALLOWED_SERVER_EVENT_NAMES = {
    "account_signed_up",
    "authenticated_session_started",
    "credential_saved",
    "credential_verified",
    "credential_defaulted",
    "credential_reverified",
    "credential_revoked",
    "agent_created",
    "agent_updated",
    "agent_deleted",
    "backtest_requested",
    "backtest_queued",
    "backtest_started",
    "backtest_completed",
    "backtest_failed",
    "backtest_cancelled",
    "model_usage_recorded",
    "credits_reserved",
    "credits_settled",
    "credits_refunded",
    "safe_error_recorded",
}
ALLOWED_EVENT_NAMES = ALLOWED_FRONTEND_EVENT_NAMES | ALLOWED_SERVER_EVENT_NAMES
ALLOWED_PAGE_VIEWS = {
    "home",
    "agents",
    "agent_editor",
    "backtest",
    "paper_trading",
    "competition",
    "community",
    "credits",
    "account",
}
ALLOWED_BILLING_MODES = {"byok", "platform_credits"}
ALLOWED_OUTCOMES = {"succeeded", "failed", "cancelled"}
ALLOWED_ERROR_CATEGORIES = {
    "credential_invalid",
    "credential_missing",
    "provider_timeout",
    "provider_unavailable",
    "provider_quota_exhausted",
    "credits_unavailable",
    "model_not_allowed",
    "internal_error",
}

EVENT_GROUP_BY_NAME = {
    "page_viewed": "experience",
    "page_hidden": "experience",
    "session_heartbeat": "experience",
    "account_signed_up": "account",
    "authenticated_session_started": "account",
    "credential_saved": "credential",
    "credential_verified": "credential",
    "credential_defaulted": "credential",
    "credential_reverified": "credential",
    "credential_revoked": "credential",
    "agent_created": "agent",
    "agent_updated": "agent",
    "agent_deleted": "agent",
    "backtest_requested": "run",
    "backtest_queued": "run",
    "backtest_started": "run",
    "backtest_completed": "run",
    "backtest_failed": "run",
    "backtest_cancelled": "run",
    "model_usage_recorded": "resource",
    "credits_reserved": "resource",
    "credits_settled": "resource",
    "credits_refunded": "resource",
    "safe_error_recorded": "resource",
}

_EMPTY_SERVER_EVENTS = ALLOWED_SERVER_EVENT_NAMES - {
    "model_usage_recorded",
    "credits_reserved",
    "credits_settled",
    "credits_refunded",
}
_CREDIT_BUCKETS = {"grant", "purchased"}


def _canonical_uuid(value: str, field_name: str) -> str:
    try:
        parsed = UUID(value)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{field_name} must be a canonical UUID") from exc
    canonical = str(parsed)
    if canonical != value.lower():
        raise ValueError(f"{field_name} must be a canonical UUID")
    return canonical


def _bounded_properties(value: dict[str, Any]) -> dict[str, Any]:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("analytics properties must be JSON-safe") from exc
    if len(encoded) > MAX_PROPERTIES_BYTES:
        raise ValueError("analytics properties are too large")
    return value


def sanitize_frontend_properties(
    event_name: str,
    properties: object,
) -> dict[str, Any]:
    if event_name not in ALLOWED_FRONTEND_EVENT_NAMES:
        raise ValueError("unknown frontend analytics event")
    if not isinstance(properties, dict):
        raise ValueError("analytics properties must be an object")
    allowed_keys = set() if event_name == "page_viewed" else {"visible_ms"}
    if set(properties) - allowed_keys:
        raise ValueError("unknown frontend analytics property")
    cleaned: dict[str, Any] = {}
    if "visible_ms" in properties:
        visible_ms = properties["visible_ms"]
        if (
            isinstance(visible_ms, bool)
            or not isinstance(visible_ms, int)
            or not 0 <= visible_ms <= 1_800_000
        ):
            raise ValueError("visible_ms must be an integer from 0 through 1800000")
        cleaned["visible_ms"] = visible_ms
    return _bounded_properties(cleaned)


def _bounded_integer(
    value: object,
    field_name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ValueError(
            f"{field_name} must be an integer from {minimum} through {maximum}"
        )
    return value


def sanitize_server_properties(
    event_name: str,
    properties: object,
) -> dict[str, Any]:
    if event_name not in ALLOWED_SERVER_EVENT_NAMES:
        raise ValueError("unknown server analytics event")
    if not isinstance(properties, dict):
        raise ValueError("analytics properties must be an object")
    if event_name in _EMPTY_SERVER_EVENTS:
        if properties:
            raise ValueError("server analytics event does not accept properties")
        return {}
    if event_name == "model_usage_recorded":
        allowed = {"input_tokens", "output_tokens", "cost_micro_usd"}
        if set(properties) != allowed:
            raise ValueError("model usage properties are incomplete or unknown")
        cleaned = {
            "input_tokens": _bounded_integer(
                properties["input_tokens"],
                "input_tokens",
                minimum=0,
                maximum=2_000_000_000,
            ),
            "output_tokens": _bounded_integer(
                properties["output_tokens"],
                "output_tokens",
                minimum=0,
                maximum=2_000_000_000,
            ),
            "cost_micro_usd": _bounded_integer(
                properties["cost_micro_usd"],
                "cost_micro_usd",
                minimum=0,
                maximum=10_000_000_000,
            ),
        }
        return _bounded_properties(cleaned)

    allowed = {"amount_micro", "bucket"}
    if set(properties) != allowed:
        raise ValueError("Credits properties are incomplete or unknown")
    bucket = properties["bucket"]
    if not isinstance(bucket, str) or bucket not in _CREDIT_BUCKETS:
        raise ValueError("bucket must be grant or purchased")
    cleaned = {
        "amount_micro": _bounded_integer(
            properties["amount_micro"],
            "amount_micro",
            minimum=1,
            maximum=10_000_000_000,
        ),
        "bucket": bucket,
    }
    return _bounded_properties(cleaned)


class FrontendAnalyticsEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=36, max_length=36)
    schema_version: Literal[1]
    event_name: Literal["page_viewed", "page_hidden", "session_heartbeat"]
    session_id: str = Field(min_length=36, max_length=36)
    occurred_at: datetime
    page_view: str = Field(min_length=1, max_length=64)
    properties: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id", "session_id")
    @classmethod
    def validate_uuid(cls, value: str, info) -> str:
        return _canonical_uuid(value, info.field_name)

    @field_validator("occurred_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must include a timezone")
        return value

    @field_validator("page_view")
    @classmethod
    def validate_page_view(cls, value: str) -> str:
        if value not in ALLOWED_PAGE_VIEWS:
            raise ValueError("unknown page view")
        return value

    @model_validator(mode="after")
    def validate_properties(self) -> "FrontendAnalyticsEvent":
        self.properties = sanitize_frontend_properties(
            self.event_name,
            self.properties,
        )
        return self


class AnalyticsEventDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=36, max_length=36)
    schema_version: Literal[1] = 1
    event_name: str = Field(min_length=1, max_length=64)
    user_id: int = Field(gt=0)
    session_id: str | None = Field(default=None, min_length=36, max_length=36)
    occurred_at: datetime
    event_source: Literal["frontend", "server", "backfill"]
    source_event_id: str | None = Field(default=None, min_length=1, max_length=200)
    source_record_type: str | None = Field(default=None, min_length=1, max_length=64)
    source_record_id: str | None = Field(default=None, min_length=1, max_length=200)
    correlation_id: str | None = Field(default=None, min_length=1, max_length=200)
    page_view: str | None = Field(default=None, min_length=1, max_length=64)
    provider_id: str | None = Field(default=None, min_length=1, max_length=128)
    model_id: str | None = Field(default=None, min_length=1, max_length=256)
    billing_mode: str | None = Field(default=None, min_length=1, max_length=32)
    outcome: str | None = Field(default=None, min_length=1, max_length=32)
    error_category: str | None = Field(default=None, min_length=1, max_length=64)
    properties: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def validate_event_id(cls, value: str) -> str:
        return _canonical_uuid(value, "event_id")

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str | None) -> str | None:
        return _canonical_uuid(value, "session_id") if value is not None else None

    @field_validator("occurred_at")
    @classmethod
    def require_occurred_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must include a timezone")
        return value

    @field_validator("event_name")
    @classmethod
    def validate_event_name(cls, value: str) -> str:
        if value not in ALLOWED_EVENT_NAMES:
            raise ValueError("unknown analytics event")
        return value

    @model_validator(mode="after")
    def validate_event_fields(self) -> "AnalyticsEventDraft":
        if self.billing_mode is not None and self.billing_mode not in ALLOWED_BILLING_MODES:
            raise ValueError("unknown billing mode")
        if self.outcome is not None and self.outcome not in ALLOWED_OUTCOMES:
            raise ValueError("unknown outcome")
        if (
            self.error_category is not None
            and self.error_category not in ALLOWED_ERROR_CATEGORIES
        ):
            raise ValueError("unknown error category")
        if self.page_view is not None and self.page_view not in ALLOWED_PAGE_VIEWS:
            raise ValueError("unknown page view")
        if self.event_source == "frontend":
            if self.event_name not in ALLOWED_FRONTEND_EVENT_NAMES:
                raise ValueError("frontend source cannot claim server event")
            if self.source_event_id is not None:
                raise ValueError("frontend event cannot set source_event_id")
            self.properties = sanitize_frontend_properties(
                self.event_name,
                self.properties,
            )
        else:
            if self.event_name not in ALLOWED_SERVER_EVENT_NAMES:
                raise ValueError("server source cannot claim frontend event")
            if self.source_event_id is None:
                raise ValueError("server and backfill events require source_event_id")
            self.properties = sanitize_server_properties(
                self.event_name,
                self.properties,
            )
        return self


class AnalyticsEventRecord(AnalyticsEventDraft):
    event_group: Literal[
        "experience",
        "account",
        "credential",
        "agent",
        "run",
        "resource",
    ]
    received_at: datetime
    country_code: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    device_category: Literal["mobile", "tablet", "desktop", "unknown"] | None = None
    browser_family: Literal["Edge", "Chrome", "Firefox", "Safari", "Other"] | None = None
    network_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("received_at")
    @classmethod
    def require_received_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("received_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_event_group(self) -> "AnalyticsEventRecord":
        if self.event_group != EVENT_GROUP_BY_NAME[self.event_name]:
            raise ValueError("event group does not match event name")
        return self


class AppendEventResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: AnalyticsEventRecord
    created: bool


class RetentionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_events_deleted: int = Field(default=0, ge=0)
    access_rows_deleted: int = Field(default=0, ge=0)
    has_more_raw_events: bool = False
    has_more_access_rows: bool = False


class RequestAnalyticsContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    country_code: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    device_category: Literal["mobile", "tablet", "desktop", "unknown"]
    browser_family: Literal["Edge", "Chrome", "Firefox", "Safari", "Other"]
    network_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
