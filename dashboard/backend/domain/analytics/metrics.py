"""Deterministic, UTC-bounded Analytics metric formulas."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import AnalyticsEventRecord


_MEANINGFUL_EXPERIENCE_EVENTS = {"page_viewed"}
_TERMINAL_SUCCESS = "backtest_completed"
_TERMINAL_FAILURE = "backtest_failed"


def _utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return value.astimezone(timezone.utc)


class AnalyticsMetricFilters(BaseModel):
    """Normalized query filters shared by rollups and Admin APIs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: datetime
    end: datetime
    billing_mode: str | None = None
    provider_id: str | None = Field(default=None, max_length=128)
    model_id: str | None = Field(default=None, max_length=256)
    include_internal: bool = False

    @model_validator(mode="after")
    def validate_range(self) -> "AnalyticsMetricFilters":
        start = _utc(self.start, "start")
        end = _utc(self.end, "end")
        if end <= start:
            raise ValueError("end must be after start")
        if self.billing_mode not in {None, "byok", "platform_credits"}:
            raise ValueError("billing_mode must be byok or platform_credits")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        return self

    @classmethod
    def default(cls, *, now: datetime | None = None, days: int = 30):
        current = _utc(now or datetime.now(timezone.utc), "now")
        if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= 180:
            raise ValueError("days must be an integer from 1 through 180")
        return cls(start=current - timedelta(days=days), end=current)


class AnalyticsOverviewMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    active_users_7d: int = 0
    mature_signup_cohort_users: int = 0
    first_success_within_7d_users: int = 0
    first_success_conversion: float | None = None
    completed_runs: int = 0
    failed_runs: int = 0
    backtest_success_rate: float | None = None
    users_with_first_success: int = 0
    users_with_repeat_success: int = 0
    repeat_run_rate: float | None = None
    platform_model_cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    affected_users: int = 0
    daily_active_users: dict[str, int] = Field(default_factory=dict)
    daily_completed_runs: dict[str, int] = Field(default_factory=dict)
    top_failure_categories: dict[str, int] = Field(default_factory=dict)


def backtest_success_rate(*, completed: int, failed: int) -> float | None:
    denominator = completed + failed
    return None if denominator == 0 else completed / denominator


def repeat_run_rate(
    *,
    users_with_first_success: int,
    users_with_repeat_success: int,
) -> float | None:
    if users_with_first_success == 0:
        return None
    return users_with_repeat_success / users_with_first_success


def is_meaningful_event(event: AnalyticsEventRecord) -> bool:
    if event.event_group == "experience":
        return event.event_name in _MEANINGFUL_EXPERIENCE_EVENTS
    return True


def _matches_dimensions(
    event: AnalyticsEventRecord,
    *,
    billing_mode: str | None,
    provider_id: str | None,
    model_id: str | None,
) -> bool:
    if billing_mode is not None and event.billing_mode != billing_mode:
        return False
    if provider_id is not None and event.provider_id != provider_id:
        return False
    if model_id is not None and event.model_id != model_id:
        return False
    return True


def calculate_overview_metrics(
    events: Iterable[AnalyticsEventRecord],
    *,
    start: datetime,
    end: datetime,
    excluded_user_ids: set[int] | None = None,
    billing_mode: str | None = None,
    provider_id: str | None = None,
    model_id: str | None = None,
) -> AnalyticsOverviewMetrics:
    """Calculate exact metrics from a bounded event set.

    Callers must include signup and successful-run context needed by cohort and
    repeat formulas. Values counted for the requested range still use the
    half-open interval ``[start, end)``.
    """

    range_start = _utc(start, "start")
    range_end = _utc(end, "end")
    if range_end <= range_start:
        raise ValueError("end must be after start")
    excluded = excluded_user_ids or set()
    eligible = [
        event
        for event in events
        if event.user_id not in excluded
        and _matches_dimensions(
            event,
            billing_mode=billing_mode,
            provider_id=provider_id,
            model_id=model_id,
        )
    ]
    in_range = [
        event
        for event in eligible
        if range_start <= event.occurred_at.astimezone(timezone.utc) < range_end
    ]

    active_start = range_end - timedelta(days=7)
    active_users = {
        event.user_id
        for event in eligible
        if active_start <= event.occurred_at.astimezone(timezone.utc) < range_end
        and is_meaningful_event(event)
    }
    affected_users = {
        event.user_id for event in in_range if is_meaningful_event(event)
    }

    successes_by_user: dict[int, list[datetime]] = defaultdict(list)
    for event in eligible:
        if event.event_name == _TERMINAL_SUCCESS:
            successes_by_user[event.user_id].append(
                event.occurred_at.astimezone(timezone.utc)
            )
    for values in successes_by_user.values():
        values.sort()

    mature_signups: dict[int, datetime] = {}
    for event in eligible:
        if event.event_name != "account_signed_up":
            continue
        signup_at = event.occurred_at.astimezone(timezone.utc)
        if range_start <= signup_at < range_end and signup_at + timedelta(days=7) <= range_end:
            mature_signups.setdefault(event.user_id, signup_at)
    converted = 0
    for user_id, signup_at in mature_signups.items():
        if any(
            signup_at <= success_at <= signup_at + timedelta(days=7)
            for success_at in successes_by_user.get(user_id, [])
        ):
            converted += 1

    users_with_first_success = len(successes_by_user)
    users_with_repeat = 0
    for success_times in successes_by_user.values():
        first = success_times[0]
        if any(
            first + timedelta(hours=24) <= later <= first + timedelta(days=30)
            for later in success_times[1:]
        ):
            users_with_repeat += 1

    completed = sum(event.event_name == _TERMINAL_SUCCESS for event in in_range)
    failed = sum(event.event_name == _TERMINAL_FAILURE for event in in_range)
    input_tokens = 0
    output_tokens = 0
    platform_cost_micro = 0
    daily_users: dict[date, set[int]] = defaultdict(set)
    daily_completed: Counter[date] = Counter()
    failures: Counter[str] = Counter()
    for event in in_range:
        event_day = event.occurred_at.astimezone(timezone.utc).date()
        if is_meaningful_event(event):
            daily_users[event_day].add(event.user_id)
        if event.event_name == _TERMINAL_SUCCESS:
            daily_completed[event_day] += 1
        if event.error_category:
            failures[event.error_category] += 1
        if event.event_name == "model_usage_recorded":
            input_tokens += int(event.properties.get("input_tokens", 0))
            output_tokens += int(event.properties.get("output_tokens", 0))
            if event.billing_mode == "platform_credits":
                platform_cost_micro += int(
                    event.properties.get("cost_micro_usd", 0)
                )

    return AnalyticsOverviewMetrics(
        active_users_7d=len(active_users),
        mature_signup_cohort_users=len(mature_signups),
        first_success_within_7d_users=converted,
        first_success_conversion=(
            None if not mature_signups else converted / len(mature_signups)
        ),
        completed_runs=completed,
        failed_runs=failed,
        backtest_success_rate=backtest_success_rate(
            completed=completed,
            failed=failed,
        ),
        users_with_first_success=users_with_first_success,
        users_with_repeat_success=users_with_repeat,
        repeat_run_rate=repeat_run_rate(
            users_with_first_success=users_with_first_success,
            users_with_repeat_success=users_with_repeat,
        ),
        platform_model_cost_usd=platform_cost_micro / 1_000_000,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        affected_users=len(affected_users),
        daily_active_users={
            day.isoformat(): len(users) for day, users in sorted(daily_users.items())
        },
        daily_completed_runs={
            day.isoformat(): count for day, count in sorted(daily_completed.items())
        },
        top_failure_categories=dict(failures.most_common(10)),
    )


__all__ = [
    "AnalyticsMetricFilters",
    "AnalyticsOverviewMetrics",
    "backtest_success_rate",
    "calculate_overview_metrics",
    "is_meaningful_event",
    "repeat_run_rate",
]
