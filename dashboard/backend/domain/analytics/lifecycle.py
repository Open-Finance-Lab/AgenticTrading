"""Pure, explainable rules for Admin user-value analytics."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .models import AnalyticsEventRecord


LifecycleSegment = Literal[
    "new",
    "onboarding",
    "growing",
    "core",
    "at_risk",
    "dormant",
]
OperationalState = Literal["blocked", "needs_attention", "healthy"]
CommercialTier = Literal["unpaid", "starter", "invested", "high_value"]
CredentialStatus = Literal[
    "verified",
    "invalid",
    "verification_unavailable",
    "missing",
]


_LIFECYCLE_ACTIVITY_EVENTS = frozenset(
    {
        "agent_created",
        "agent_updated",
        "credential_saved",
        "credential_verified",
        "credential_reverified",
        "credential_defaulted",
        "backtest_requested",
        "backtest_queued",
        "backtest_started",
        "backtest_completed",
        "backtest_failed",
        "backtest_cancelled",
        "model_usage_recorded",
        "credits_reserved",
        "credits_settled",
    }
)

_LIFECYCLE_REASONS = {
    "new_no_successful_backtest": (
        "The account is less than seven UTC days old and has not completed a "
        "successful backtest."
    ),
    "onboarding_no_successful_backtest": (
        "The account has not completed a successful backtest."
    ),
    "growing_activated_below_core_threshold": (
        "The user activated recently but has not met both Core thresholds."
    ),
    "core_repeated_value": (
        "The user has at least three active days and three successful backtests "
        "in the trailing 30 UTC days."
    ),
    "at_risk_never_activated": (
        "The user has never activated and has been inactive for 8 to 29 UTC days."
    ),
    "at_risk_previously_activated": (
        "The user previously activated and has been inactive for 8 to 29 UTC days."
    ),
    "dormant_never_activated": (
        "The user has never activated and has been inactive for at least 30 UTC days."
    ),
    "dormant_previously_activated": (
        "The user previously activated and has been inactive for at least 30 UTC days."
    ),
}

_OPERATIONAL_REASONS = {
    "account_restricted": "The Credits account is restricted from model spending.",
    "billing_lane_unavailable": "No usable model billing lane is available.",
    "provider_disabled": "The selected model provider is disabled.",
    "invalid_default_credential": "The default model credential is invalid.",
    "three_consecutive_failed_runs": (
        "At least three consecutive terminal runs failed within 24 hours."
    ),
    "run_deadline_exceeded": "A run remains non-terminal beyond its safe deadline.",
    "no_supported_issue": "No supported current operational issue was detected.",
}

_OPERATIONAL_EVIDENCE = {
    "account_restricted": "Credits account restriction is unresolved.",
    "billing_lane_unavailable": "Usable billing lanes: 0.",
    "provider_disabled": "Selected provider enabled: no.",
    "invalid_default_credential": "Default credential status: invalid.",
    "run_deadline_exceeded": "A run is beyond its safe deadline.",
    "no_supported_issue": "All supported operational checks passed.",
}


def _require_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return value.astimezone(timezone.utc)


class LifecycleInputs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: int = Field(gt=0)
    created_at: datetime
    first_successful_backtest_at: datetime | None = None
    last_meaningful_activity_at: datetime | None = None
    active_days_30d: int = Field(ge=0, le=30)
    successful_backtests_30d: int = Field(ge=0)


class LifecycleResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    segment: LifecycleSegment
    reason_code: str
    reason: str
    evidence: Sequence[str]
    activated_at: datetime | None
    last_meaningful_activity_at: datetime | None
    inactive_days: int = Field(ge=0)
    active_days_30d: int = Field(ge=0, le=30)
    successful_backtests_30d: int = Field(ge=0)
    calculated_at: datetime


class OperationalSignals(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: int = Field(gt=0)
    account_restricted: bool = False
    usable_billing_lane: bool = True
    selected_provider_enabled: bool = True
    default_credential_status: CredentialStatus = "verified"
    failed_terminal_runs_24h: int = Field(default=0, ge=0)
    run_beyond_safe_deadline: bool = False


class OperationalResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state: OperationalState
    reason_code: str
    reason: str
    evidence: Sequence[str]
    calculated_at: datetime


def is_lifecycle_activity(event: AnalyticsEventRecord) -> bool:
    """Return whether an accepted event advances the lifecycle inactivity clock."""

    return event.event_name in _LIFECYCLE_ACTIVITY_EVENTS


def _day_evidence(label: str, days: int) -> str:
    if days == 0:
        return f"{label} today"
    unit = "day" if days == 1 else "days"
    return f"{label} {days} {unit} ago"


def _lifecycle_evidence(
    inputs: LifecycleInputs,
    *,
    account_age_days: int,
    inactive_days: int,
    activated_at: datetime | None,
) -> tuple[str, ...]:
    evidence = [
        f"{inputs.active_days_30d} active days in the trailing 30 days",
        (
            f"{inputs.successful_backtests_30d} successful backtests in the "
            "trailing 30 days"
        ),
        _day_evidence("last meaningful activity", inactive_days),
    ]
    if activated_at is None:
        evidence.append("no successful backtest recorded")
        evidence.append(_day_evidence("account created", account_age_days))
    else:
        evidence.append(
            f"first successful backtest on {activated_at.date().isoformat()}"
        )
    return tuple(evidence)


def _lifecycle_reason_code(
    segment: LifecycleSegment,
    *,
    activated: bool,
) -> str:
    if segment in {"at_risk", "dormant"}:
        qualifier = "previously_activated" if activated else "never_activated"
        return f"{segment}_{qualifier}"
    return {
        "new": "new_no_successful_backtest",
        "onboarding": "onboarding_no_successful_backtest",
        "growing": "growing_activated_below_core_threshold",
        "core": "core_repeated_value",
    }[segment]


def calculate_lifecycle(
    inputs: LifecycleInputs,
    as_of: datetime,
) -> LifecycleResult:
    """Classify one user from fixed UTC-day inputs with deterministic precedence."""

    calculation_time = _require_utc(as_of, "as_of")
    created_at = _require_utc(inputs.created_at, "created_at")
    activated_at = (
        _require_utc(
            inputs.first_successful_backtest_at,
            "first_successful_backtest_at",
        )
        if inputs.first_successful_backtest_at is not None
        else None
    )
    last_meaningful = (
        _require_utc(
            inputs.last_meaningful_activity_at,
            "last_meaningful_activity_at",
        )
        if inputs.last_meaningful_activity_at is not None
        else None
    )
    activity_candidates = [created_at]
    if activated_at is not None:
        activity_candidates.append(activated_at)
    if last_meaningful is not None:
        activity_candidates.append(last_meaningful)
    activity_anchor = max(activity_candidates)
    inactive_days = (calculation_time.date() - activity_anchor.date()).days
    account_age_days = (calculation_time.date() - created_at.date()).days
    if inactive_days < 0 or account_age_days < 0:
        raise ValueError("lifecycle evidence cannot occur after as_of")

    if inactive_days >= 30:
        segment: LifecycleSegment = "dormant"
    elif inactive_days >= 8:
        segment = "at_risk"
    elif activated_at is None:
        segment = "new" if account_age_days <= 6 else "onboarding"
    elif inputs.active_days_30d >= 3 and inputs.successful_backtests_30d >= 3:
        segment = "core"
    else:
        segment = "growing"

    reason_code = _lifecycle_reason_code(
        segment,
        activated=activated_at is not None,
    )
    return LifecycleResult(
        segment=segment,
        reason_code=reason_code,
        reason=_LIFECYCLE_REASONS[reason_code],
        evidence=_lifecycle_evidence(
            inputs,
            account_age_days=account_age_days,
            inactive_days=inactive_days,
            activated_at=activated_at,
        ),
        activated_at=activated_at,
        last_meaningful_activity_at=last_meaningful,
        inactive_days=inactive_days,
        active_days_30d=inputs.active_days_30d,
        successful_backtests_30d=inputs.successful_backtests_30d,
        calculated_at=calculation_time,
    )


def _operational_result(
    state: OperationalState,
    reason_code: str,
    calculation_time: datetime,
    *,
    evidence: str | None = None,
) -> OperationalResult:
    return OperationalResult(
        state=state,
        reason_code=reason_code,
        reason=_OPERATIONAL_REASONS[reason_code],
        evidence=(evidence or _OPERATIONAL_EVIDENCE[reason_code],),
        calculated_at=calculation_time,
    )


def calculate_operational_state(
    signals: OperationalSignals,
    as_of: datetime,
) -> OperationalResult:
    """Calculate the operational axis independently from lifecycle state."""

    calculation_time = _require_utc(as_of, "as_of")
    if signals.account_restricted:
        return _operational_result(
            "blocked",
            "account_restricted",
            calculation_time,
        )
    if not signals.usable_billing_lane:
        return _operational_result(
            "blocked",
            "billing_lane_unavailable",
            calculation_time,
        )
    if not signals.selected_provider_enabled:
        return _operational_result(
            "blocked",
            "provider_disabled",
            calculation_time,
        )
    if signals.default_credential_status == "invalid":
        return _operational_result(
            "needs_attention",
            "invalid_default_credential",
            calculation_time,
        )
    if signals.failed_terminal_runs_24h >= 3:
        return _operational_result(
            "needs_attention",
            "three_consecutive_failed_runs",
            calculation_time,
            evidence=(
                f"{signals.failed_terminal_runs_24h} consecutive terminal runs "
                "failed in the last 24 hours."
            ),
        )
    if signals.run_beyond_safe_deadline:
        return _operational_result(
            "needs_attention",
            "run_deadline_exceeded",
            calculation_time,
        )
    return _operational_result(
        "healthy",
        "no_supported_issue",
        calculation_time,
    )


def commercial_tier(net_purchased_micro: int) -> CommercialTier:
    """Classify lifetime net purchases; refunds cannot make revenue negative."""

    if isinstance(net_purchased_micro, bool) or not isinstance(
        net_purchased_micro,
        int,
    ):
        raise ValueError("net_purchased_micro must be an integer")
    value = max(0, net_purchased_micro)
    if value == 0:
        return "unpaid"
    if value < 5_000_000:
        return "starter"
    if value < 20_000_000:
        return "invested"
    return "high_value"


def activation_cohort_week(activated_at: datetime) -> date:
    """Return the UTC Monday date for a user's activation timestamp."""

    activated_day = _require_utc(activated_at, "activated_at").date()
    return activated_day - timedelta(days=activated_day.weekday())


__all__ = [
    "CommercialTier",
    "LifecycleInputs",
    "LifecycleResult",
    "LifecycleSegment",
    "OperationalResult",
    "OperationalSignals",
    "OperationalState",
    "activation_cohort_week",
    "calculate_lifecycle",
    "calculate_operational_state",
    "commercial_tier",
    "is_lifecycle_activity",
]
