"""Shared validation and cursor helpers for Analytics repositories."""

from __future__ import annotations

import base64
import binascii
import json
from datetime import datetime, timezone

from .models import AnalyticsEventRecord


class AnalyticsStoreError(RuntimeError):
    """Base class for safe, expected Analytics persistence failures."""


class AnalyticsIdempotencyConflictError(AnalyticsStoreError):
    """An event idempotency key was replayed with different safe data."""


def utc_iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("analytics timestamp must include a timezone")
    return value.astimezone(timezone.utc).isoformat()


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def positive_user_id(value: object, name: str = "user_id") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def positive_limit(value: object, *, maximum: int = 100) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise ValueError(f"limit must be an integer from 1 through {maximum}")
    return value


def required_reason(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("reason must be a trimmed non-empty string")
    if len(value) > 500:
        raise ValueError("reason must be at most 500 characters")
    return value


def validate_access_section(value: object) -> str:
    allowed = {"overview", "timeline", "runs", "usage", "sessions"}
    if not isinstance(value, str) or value not in allowed:
        raise ValueError("section must be a supported Analytics profile section")
    return value


def encode_event_cursor(occurred_at: str, sequence: int) -> str:
    if not isinstance(occurred_at, str) or not occurred_at or len(occurred_at) > 64:
        raise ValueError("invalid analytics cursor")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
        raise ValueError("invalid analytics cursor")
    payload = json.dumps(
        [occurred_at, sequence],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def decode_event_cursor(cursor: str) -> tuple[str, int]:
    if not isinstance(cursor, str) or not cursor or len(cursor) > 256:
        raise ValueError("invalid analytics cursor")
    try:
        raw = base64.b64decode(
            cursor + "=" * (-len(cursor) % 4),
            altchars=b"-_",
            validate=True,
        )
        value = json.loads(raw.decode("utf-8"))
    except (ValueError, binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid analytics cursor") from exc
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("invalid analytics cursor")
    occurred_at, sequence = value
    if (
        not isinstance(occurred_at, str)
        or not occurred_at
        or len(occurred_at) > 64
        or isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence <= 0
    ):
        raise ValueError("invalid analytics cursor")
    return occurred_at, sequence


def canonical_event_payload(
    event: AnalyticsEventRecord,
    *,
    ignore_event_id: bool = False,
) -> dict[str, object]:
    value = event.model_dump(mode="json")
    value["occurred_at"] = utc_iso(event.occurred_at)
    value.pop("received_at", None)
    if ignore_event_id:
        value.pop("event_id", None)
    value["properties"] = json.loads(
        json.dumps(
            value["properties"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return value
