"""Pure contracts for privacy-safe acquisition analytics rules."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone

import pytest

from dashboard.backend.domain.analytics.acquisition import (
    AcquisitionAttribution,
    active_user_ids,
    checkout_window_is_mature,
    normalize_cohort_slug,
    repeat_user_ids,
    resolve_invite_token,
    runs_per_active_user,
)


UTC = timezone.utc
NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)
SIGNING_KEY = "synthetic-acquisition-test-key"


def _signed_invite(payload: dict[str, object]) -> str:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
    signature = hmac.new(
        SIGNING_KEY.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256
    ).digest()
    encoded_signature = base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")
    return f"{encoded}.{encoded_signature}"


@pytest.mark.parametrize("source", ["student", "community", "friend", "competition"])
def test_signed_invite_accepts_allowlisted_sources(source):
    token = _signed_invite(
        {
            "source": source,
            "cohort": "fall-2026",
            "iat": int(NOW.timestamp()),
            "exp": int((NOW + timedelta(days=7)).timestamp()),
        }
    )

    result = resolve_invite_token(token, SIGNING_KEY, NOW)

    assert result.source == source
    assert result.cohort == "fall-2026"
    assert result.method == "invite"


def test_invalid_expired_or_tampered_invite_defaults_to_unknown():
    expired = _signed_invite(
        {
            "source": "student",
            "cohort": "fall-2026",
            "iat": int((NOW - timedelta(days=8)).timestamp()),
            "exp": int((NOW - timedelta(seconds=1)).timestamp()),
        }
    )

    assert resolve_invite_token(None, SIGNING_KEY, NOW).source == "unknown"
    assert resolve_invite_token(expired, SIGNING_KEY, NOW).source == "unknown"
    assert resolve_invite_token(f"{expired}x", SIGNING_KEY, NOW).source == "unknown"
    assert resolve_invite_token(expired, None, NOW).source == "unknown"


@pytest.mark.parametrize("value", ["Bad/Slug", "x" * 65, "two words"])
def test_cohort_requires_a_bounded_lowercase_slug(value):
    with pytest.raises(ValueError, match="lower-case slug"):
        normalize_cohort_slug(value)


@pytest.mark.parametrize(
    ("value", "expected"), [("UPPER", "upper"), (" space ", "space")]
)
def test_cohort_normalizes_case_and_outer_whitespace(value, expected):
    assert normalize_cohort_slug(value) == expected


def test_attribution_defaults_and_timestamp_are_display_safe():
    attribution = AcquisitionAttribution(user_id=7, attributed_at=NOW.isoformat())

    assert attribution.source == "unknown"
    assert attribution.cohort is None
    assert attribution.method == "unknown"
    assert attribution.attributed_at == NOW
    assert "invite_token" not in AcquisitionAttribution.model_fields


def test_active_repeat_depth_and_checkout_maturity_formulas():
    events = [
        {"event_name": "backtest_requested", "user_id": 7, "occurred_at": NOW},
        {
            "event_name": "backtest_requested",
            "user_id": 8,
            "occurred_at": NOW - timedelta(days=8),
        },
    ]
    success_events = [
        {"event_name": "backtest_completed", "user_id": 7, "occurred_at": NOW},
        {
            "event_name": "backtest_completed",
            "user_id": 7,
            "occurred_at": NOW - timedelta(days=1),
        },
        {
            "event_name": "backtest_completed",
            "user_id": 8,
            "occurred_at": NOW,
        },
    ]

    assert active_user_ids(
        events, NOW - timedelta(days=1), NOW + timedelta(days=1)
    ) == {7}
    assert repeat_user_ids(success_events) == {7}
    assert runs_per_active_user(3, 2) == 1.5
    assert runs_per_active_user(0, 0) is None
    assert checkout_window_is_mature(NOW, NOW + timedelta(days=7)) is True
    assert checkout_window_is_mature(NOW, NOW + timedelta(days=7, seconds=-1)) is False
