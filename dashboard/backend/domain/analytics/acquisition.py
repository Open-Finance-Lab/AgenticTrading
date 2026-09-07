"""Privacy-safe acquisition attribution and aggregation rules."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator


UTC = timezone.utc
AcquisitionSource = Literal["student", "community", "friend", "competition", "unknown"]
AttributionMethod = Literal["invite", "manual", "unknown"]
LifecycleFilter = Literal["new", "active", "at_risk", "dormant"]
_COHORT_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
_INVITE_MAX_AGE = timedelta(days=90)


def normalize_cohort_slug(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError("cohort must be a string or null")
    normalized = value.strip().lower()
    if not normalized or len(normalized) > 64 or not _COHORT_RE.fullmatch(normalized):
        raise ValueError("cohort must be a lower-case slug of 1 through 64 characters")
    return normalized


class InviteResolution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: AcquisitionSource = "unknown"
    cohort: str | None = None
    method: AttributionMethod = "unknown"

    @field_validator("cohort")
    @classmethod
    def validate_cohort(cls, value: str | None) -> str | None:
        return normalize_cohort_slug(value)


class AcquisitionAttribution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: int = Field(gt=0)
    source: AcquisitionSource = "unknown"
    cohort: str | None = None
    method: AttributionMethod = "unknown"
    attributed_at: datetime | None = None
    original_source: AcquisitionSource = "unknown"
    original_cohort: str | None = None
    last_corrected_at: datetime | None = None
    last_corrected_by_admin_id: int | None = Field(default=None, gt=0)

    @field_validator("cohort", "original_cohort")
    @classmethod
    def validate_cohort(cls, value: str | None) -> str | None:
        return normalize_cohort_slug(value)

    @field_validator("attributed_at", "last_corrected_at", mode="before")
    @classmethod
    def normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if not isinstance(value, datetime):
            value = datetime.fromisoformat(str(value))
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("attribution timestamp must include a timezone")
        return value.astimezone(UTC)


class AcquisitionFilters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: AcquisitionSource | None = None
    cohort: str | None = None
    lifecycle: LifecycleFilter | None = None
    blocked: bool | None = None
    paid: bool | None = None
    active: bool | None = None
    task_completed: bool | None = None
    repeat: bool | None = None
    paid_intent: bool | None = None
    include_internal: bool = False

    @field_validator("cohort")
    @classmethod
    def validate_cohort(cls, value: str | None) -> str | None:
        return normalize_cohort_slug(value)


class AcquisitionGroupFacts(BaseModel):
    """Display-safe per-user facts used to build a source/cohort group."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    active: bool = False
    task_completed: bool = False
    repeat: bool = False
    runs: int = Field(default=0, ge=0)
    atl_credits_settled_micro: int = Field(default=0, ge=0)
    paid_intent: bool = False
    paid: bool = False


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value.astimezone(UTC)


def _event_value(event: object, name: str, default: Any = None) -> Any:
    if isinstance(event, dict):
        return event.get(name, default)
    return getattr(event, name, default)


def _event_time(event: object) -> datetime:
    value = _event_value(event, "occurred_at")
    if isinstance(value, datetime):
        return _utc(value)
    parsed = datetime.fromisoformat(str(value))
    return _utc(parsed)


def resolve_invite_token(
    token: str | None,
    signing_key: str | bytes | None,
    now: datetime | None = None,
) -> InviteResolution:
    """Verify a compact ``base64url(payload).base64url(signature)`` invite.

    The payload is intentionally tiny and allowlisted. Any malformed, expired,
    or unverifiable token becomes ``unknown`` rather than blocking signup.
    """

    unknown = InviteResolution()
    if not isinstance(token, str) or not token or len(token) > 512:
        return unknown
    if not signing_key:
        return unknown
    key = signing_key.encode("utf-8") if isinstance(signing_key, str) else signing_key
    try:
        encoded_payload, encoded_signature = token.split(".", 1)
        raw_payload = base64.urlsafe_b64decode(
            encoded_payload + "=" * (-len(encoded_payload) % 4)
        )
        signature = base64.urlsafe_b64decode(
            encoded_signature + "=" * (-len(encoded_signature) % 4)
        )
        expected = hmac.new(
            key, encoded_payload.encode("ascii"), hashlib.sha256
        ).digest()
        if not hmac.compare_digest(signature, expected):
            return unknown
        payload = json.loads(raw_payload.decode("utf-8"))
        if not isinstance(payload, dict):
            return unknown
        source = payload.get("source")
        cohort = payload.get("cohort")
        issued_at = payload.get("iat")
        expires_at = payload.get("exp")
        if source not in {"student", "community", "friend", "competition"}:
            return unknown
        current = _utc(now or datetime.now(UTC))
        if not isinstance(expires_at, int) or expires_at < int(current.timestamp()):
            return unknown
        if issued_at is not None and (
            not isinstance(issued_at, int)
            or issued_at > int(current.timestamp())
            or int(current.timestamp()) - issued_at
            > int(_INVITE_MAX_AGE.total_seconds())
        ):
            return unknown
        return InviteResolution(source=source, cohort=cohort, method="invite")
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError, OverflowError):
        return unknown


def active_user_ids(
    events: Sequence[object], start: datetime, end: datetime
) -> set[int]:
    window_start = _utc(start)
    window_end = _utc(end)
    return {
        int(_event_value(event, "user_id"))
        for event in events
        if _event_value(event, "event_name") == "backtest_requested"
        and window_start <= _event_time(event) < window_end
        and int(_event_value(event, "user_id")) > 0
    }


def repeat_user_ids(success_events: Sequence[object]) -> set[int]:
    dates_by_user: dict[int, set[date]] = defaultdict(set)
    for event in success_events:
        if _event_value(event, "event_name") not in {None, "backtest_completed"}:
            continue
        user_id = int(_event_value(event, "user_id"))
        if user_id > 0:
            dates_by_user[user_id].add(_event_time(event).date())
    return {user_id for user_id, dates in dates_by_user.items() if len(dates) >= 2}


def runs_per_active_user(runs: int, active: int) -> float | None:
    if isinstance(runs, bool) or isinstance(active, bool) or runs < 0 or active < 0:
        raise ValueError("runs and active must be non-negative integers")
    return None if active == 0 else runs / active


def checkout_window_is_mature(checkout_at: datetime, as_of: datetime) -> bool:
    return _utc(as_of) >= _utc(checkout_at) + timedelta(days=7)


__all__ = [
    "AcquisitionAttribution",
    "AcquisitionFilters",
    "AcquisitionGroupFacts",
    "AcquisitionSource",
    "AttributionMethod",
    "InviteResolution",
    "active_user_ids",
    "checkout_window_is_mature",
    "normalize_cohort_slug",
    "repeat_user_ids",
    "resolve_invite_token",
    "runs_per_active_user",
]
