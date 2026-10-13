"""Closed Analytics event-contract tests."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from dashboard.backend.domain.analytics.models import (
    ALLOWED_BILLING_MODES,
    ALLOWED_ERROR_CATEGORIES,
    ALLOWED_EVENT_NAMES,
    ALLOWED_FRONTEND_EVENT_NAMES,
    ALLOWED_PAGE_VIEWS,
    AnalyticsEventRecord,
    FrontendAnalyticsEvent,
    sanitize_server_properties,
    stored_properties,
)
from dashboard.backend.infrastructure.llm.execution.errors import (
    ExecutionErrorCategory,
    LLMExecutionError,
)


NOW = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)


def _frontend_payload(**overrides):
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
    return value


def _record_payload(**overrides):
    value = {
        "event_id": str(uuid4()),
        "schema_version": 1,
        "event_name": "page_viewed",
        "event_group": "experience",
        "user_id": 7,
        "session_id": str(uuid4()),
        "occurred_at": NOW,
        "received_at": NOW,
        "event_source": "frontend",
        "page_view": "home",
        "device_category": "desktop",
        "browser_family": "Chrome",
        "properties": {},
    }
    value.update(overrides)
    return value


def test_allowlists_are_closed_and_versioned():
    assert ALLOWED_FRONTEND_EVENT_NAMES == {
        "page_viewed",
        "page_hidden",
        "session_heartbeat",
    }
    assert ALLOWED_PAGE_VIEWS == {
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
    assert ALLOWED_BILLING_MODES == {"byok", "platform_credits"}
    assert {
        "credential_invalid",
        "credential_missing",
        "provider_timeout",
        "provider_unavailable",
        "credits_unavailable",
        "model_not_allowed",
        "internal_error",
        "run_timeout",
    } <= ALLOWED_ERROR_CATEGORIES
    assert ALLOWED_FRONTEND_EVENT_NAMES < ALLOWED_EVENT_NAMES


def test_run_timeout_is_its_own_error_category():
    """A budget exhaustion must be distinguishable from a crash.

    Before this, `_finalize_slot_locked` hardcoded `internal_error` for every
    non-None error, so the product's own analytics could not tell "we ran out of
    the hour we gave ourselves" from "something threw".
    """
    assert "run_timeout" in ALLOWED_ERROR_CATEGORIES
    assert "internal_error" in ALLOWED_ERROR_CATEGORIES


@pytest.mark.parametrize(
    "patch",
    [
        {"schema_version": 2},
        {"event_name": "backtest_completed"},
        {"event_name": "clicked_anything"},
        {"page_view": "admin"},
        {"session_id": "auth-token-shaped-value"},
        {"email": "not-accepted@example.test"},
        {"properties": {"prompt": "must not cross"}},
        {"properties": {"api_key": "synthetic-secret-canary"}},
    ],
)
def test_frontend_event_rejects_unknown_or_sensitive_shape(patch):
    with pytest.raises(ValidationError):
        FrontendAnalyticsEvent.model_validate(_frontend_payload(**patch))


def test_frontend_event_accepts_only_bounded_duration_metadata():
    event = FrontendAnalyticsEvent.model_validate(
        _frontend_payload(
            event_name="page_hidden",
            properties={"visible_ms": 12_500},
        )
    )
    assert event.properties == {"visible_ms": 12_500}

    for invalid in (-1, True, 1_800_001, "12500"):
        with pytest.raises(ValidationError):
            FrontendAnalyticsEvent.model_validate(
                _frontend_payload(
                    event_name="page_hidden",
                    properties={"visible_ms": invalid},
                )
            )


def test_server_properties_are_event_specific_and_bounded():
    assert sanitize_server_properties(
        "model_usage_recorded",
        {"input_tokens": 10, "output_tokens": 4, "cost_micro_usd": 25},
    ) == {"input_tokens": 10, "output_tokens": 4, "cost_micro_usd": 25}
    assert sanitize_server_properties(
        "credits_settled",
        {"amount_micro": 100, "bucket": "grant"},
    ) == {"amount_micro": 100, "bucket": "grant"}

    for event_name, properties in (
        ("agent_created", {"name": "must-not-cross"}),
        ("model_usage_recorded", {"input_tokens": True}),
        ("model_usage_recorded", {"cost_micro_usd": -1}),
        ("credits_settled", {"amount_micro": 0, "bucket": "grant"}),
        ("credits_settled", {"amount_micro": 100, "bucket": "unknown"}),
    ):
        with pytest.raises(ValueError):
            sanitize_server_properties(event_name, properties)


@pytest.mark.parametrize(
    "patch",
    [
        {"event_name": "unknown_event"},
        {"event_group": "run"},
        {"billing_mode": "free"},
        {"outcome": "queued"},
        {"error_category": "raw_provider_error"},
        {"provider_id": "x" * 129},
    ],
)
def test_stored_event_rejects_incoherent_or_oversized_fields(patch):
    with pytest.raises(ValidationError):
        AnalyticsEventRecord.model_validate(_record_payload(**patch))


def test_stored_event_accepts_safe_server_usage_metadata():
    record = AnalyticsEventRecord.model_validate(
        _record_payload(
            event_name="model_usage_recorded",
            event_group="resource",
            event_source="server",
            session_id=None,
            page_view=None,
            source_event_id="usage:run-1:0",
            provider_id="openrouter",
            model_id="openai/gpt-5.5",
            billing_mode="platform_credits",
            outcome="succeeded",
            properties={
                "input_tokens": 50,
                "output_tokens": 20,
                "cost_micro_usd": 123,
            },
        )
    )
    assert record.event_group == "resource"
    assert record.properties["cost_micro_usd"] == 123


@pytest.mark.parametrize(
    ("extra", "accepted"),
    [
        ({}, True),
        ({"estimated_cost_micro_usd": 900}, True),
        ({"estimated_cost_micro_usd": -1}, False),
        ({"estimated_cost_micro_usd": 10_000_000_001}, False),
        ({"provider_cost_micro_usd": 1}, False),
    ],
)
def test_usage_metadata_accepts_only_the_optional_byok_estimate(extra, accepted):
    """BYOK's list-price estimate is the one optional usage property, bounded
    like cost_micro_usd; any other key is still refused."""
    payload = _record_payload(
        event_name="model_usage_recorded",
        event_group="resource",
        event_source="server",
        session_id=None,
        page_view=None,
        source_event_id="usage:run-1:0",
        provider_id="openrouter",
        model_id="openai/gpt-5.5",
        billing_mode="byok",
        outcome="succeeded",
        properties={"input_tokens": 50, "output_tokens": 20, "cost_micro_usd": 0, **extra},
    )
    if accepted:
        assert AnalyticsEventRecord.model_validate(payload).properties == payload["properties"]
    else:
        with pytest.raises(ValidationError):
            AnalyticsEventRecord.model_validate(payload)


@pytest.mark.parametrize(
    ("billing_mode", "accepted"),
    [("byok", True), ("platform_credits", False), (None, False)],
)
def test_usage_estimate_is_accepted_on_the_byok_lane_only(billing_mode, accepted):
    """Every reader ignores the estimate outside BYOK, so one written on another
    lane would be stored and silently dropped. It is refused instead."""
    payload = _record_payload(
        event_name="model_usage_recorded",
        event_group="resource",
        event_source="server",
        session_id=None,
        page_view=None,
        source_event_id="usage:run-1:0",
        provider_id="openrouter",
        model_id="openai/gpt-5.5",
        billing_mode=billing_mode,
        outcome="succeeded",
        properties={
            "input_tokens": 50,
            "output_tokens": 20,
            "cost_micro_usd": 0,
            "estimated_cost_micro_usd": 900,
        },
    )
    if accepted:
        AnalyticsEventRecord.model_validate(payload)
    else:
        with pytest.raises(ValidationError, match="BYOK-only"):
            AnalyticsEventRecord.model_validate(payload)


def test_stored_properties_drop_keys_this_build_does_not_know(capsys):
    """Strict on read, a property added by a later build failed every read of
    its event -- after a revert, or on the old instance during a deploy. The
    unknown key is dropped and reported once; known keys are untouched, and
    still validated when the record is built."""
    usage = {"input_tokens": 1, "output_tokens": 2, "cost_micro_usd": 3}

    assert stored_properties("model_usage_recorded", {**usage, "future_key": 9}) == usage
    assert stored_properties("model_usage_recorded", {**usage, "future_key": 1}) == usage
    assert capsys.readouterr().out.count("property=future_key") == 1
    assert stored_properties("page_viewed", {"visible_ms": 5}) == {}
    assert stored_properties("page_hidden", {"visible_ms": 5}) == {"visible_ms": 5}
    # Nothing to reconcile against: left for the validator to refuse.
    assert stored_properties("not_an_event", {"x": 1}) == {"x": 1}
    assert stored_properties("model_usage_recorded", None) is None


def test_quota_exhausted_error_message_is_fixed():
    error = LLMExecutionError(ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED)
    assert str(error) == "The selected model provider has insufficient balance or quota."


def test_safe_error_accepts_provider_quota_exhausted():
    record = AnalyticsEventRecord.model_validate(
        _record_payload(
            event_name="safe_error_recorded",
            event_group="resource",
            event_source="server",
            session_id=None,
            page_view=None,
            source_event_id="safe-error:quota-test",
            error_category="provider_quota_exhausted",
        )
    )
    assert record.error_category == "provider_quota_exhausted"


def test_every_execution_failure_category_maps_into_the_allow_list():
    # `account_restricted` was mapped but not admitted, so emit_safe_error_event
    # raised inside its own guard and every restricted-account failure was
    # dropped with a WARNING line. Any category added to the mapping must be
    # admitted here in the same change.
    from dashboard.backend.infrastructure.llm.execution.errors import (
        ExecutionErrorCategory,
    )
    from dashboard.backend.infrastructure.llm.execution.service import (
        analytics_error_category,
    )

    for category in ExecutionErrorCategory:
        assert analytics_error_category(category) in ALLOWED_ERROR_CATEGORIES, category
