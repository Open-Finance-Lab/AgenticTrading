"""Legacy and dual-axis Analytics snapshots with bounded repair logic."""

from __future__ import annotations

import json
from datetime import datetime, time, timedelta, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .lifecycle import (
    LifecycleInputs,
    OperationalSignals,
    calculate_lifecycle,
    calculate_operational_state,
    is_lifecycle_activity,
)
from .metrics import is_meaningful_event
from .models import AnalyticsEventRecord
from .repository import analytics_store
from .repository_common import positive_limit, positive_user_id, utc_iso
from .rollups import AnalyticsRollupStore
from .value_repository import (
    UserLifecycleDailySnapshot,
    UserValueSnapshot,
    ValueAnalyticsStore,
)


_REASONS = {
    "billing_lane_unavailable": "The latest attempted run has no usable billing lane.",
    "provider_disabled": "The selected model provider is currently unavailable.",
    "account_restricted": "The account is currently restricted from model spending.",
    "invalid_default_credential": "The default model credential is invalid.",
    "three_consecutive_failed_runs": "The three newest terminal runs failed within 24 hours.",
    "run_deadline_exceeded": "A run remains non-terminal beyond its safe deadline.",
    "no_meaningful_activity_30d": "No meaningful product activity was observed for 30 days.",
    "no_successful_run": "The user has not completed a successful backtest yet.",
    "recent_successful_run_and_activity": "A successful run and recent meaningful activity are present.",
    "successful_run_missing": "A successful run is expected but cannot be found.",
}


class UserAnalyticsSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: int = Field(gt=0)
    status: str
    reason_code: str = Field(min_length=1, max_length=100)
    human_readable_reason: str = Field(min_length=1, max_length=500)
    evidence_event_ids: list[str] = Field(default_factory=list, max_length=5)
    calculated_at: datetime

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        allowed = {"blocked", "needs_attention", "dormant", "onboarding", "active"}
        if value not in allowed:
            raise ValueError("unsupported Analytics user state")
        return value

    @field_validator("calculated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("calculated_at must include a timezone")
        return value.astimezone(timezone.utc)


def _parse_timestamp(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class AnalyticsStateStore:
    """Snapshot companion operations over PR 1 tables."""

    def __init__(self, base_store=analytics_store):
        self.base_store = base_store
        self.rollups = AnalyticsRollupStore(base_store)
        self.is_postgres = hasattr(base_store, "database_url")

    def get_user(self, user_id: int) -> dict[str, Any] | None:
        subject_id = positive_user_id(user_id)
        if self.is_postgres:
            with self.base_store._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM users WHERE id = %s", (subject_id,))
                    row = cur.fetchone()
        else:
            with self.base_store._get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM users WHERE id = ?", (subject_id,)
                ).fetchone()
        return dict(row) if row is not None else None

    def list_user_events(
        self,
        user_id: int,
        *,
        now: datetime,
        days: int = 180,
    ) -> list[AnalyticsEventRecord]:
        return self.rollups.list_events(
            start=now - timedelta(days=days),
            end=now + timedelta(microseconds=1),
            include_internal=True,
            user_id=positive_user_id(user_id),
        )

    def upsert_snapshot(
        self,
        snapshot: UserAnalyticsSnapshot,
    ) -> UserAnalyticsSnapshot:
        payload = (
            snapshot.user_id,
            snapshot.status,
            snapshot.reason_code,
            snapshot.human_readable_reason,
            json.dumps(snapshot.evidence_event_ids, separators=(",", ":")),
            utc_iso(snapshot.calculated_at),
        )
        if self.is_postgres:
            with self.base_store._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO user_analytics_snapshots (
                            user_id, status, reason_code, human_readable_reason,
                            evidence_event_ids_json, calculated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT(user_id) DO UPDATE SET
                            status = EXCLUDED.status,
                            reason_code = EXCLUDED.reason_code,
                            human_readable_reason = EXCLUDED.human_readable_reason,
                            evidence_event_ids_json = EXCLUDED.evidence_event_ids_json,
                            calculated_at = EXCLUDED.calculated_at
                        """,
                        payload,
                    )
        else:
            with self.base_store._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO user_analytics_snapshots (
                        user_id, status, reason_code, human_readable_reason,
                        evidence_event_ids_json, calculated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        status = excluded.status,
                        reason_code = excluded.reason_code,
                        human_readable_reason = excluded.human_readable_reason,
                        evidence_event_ids_json = excluded.evidence_event_ids_json,
                        calculated_at = excluded.calculated_at
                    """,
                    payload,
                )
        return snapshot

    def upsert_combined_snapshot(
        self,
        legacy: UserAnalyticsSnapshot,
        value: UserValueSnapshot,
    ) -> tuple[UserAnalyticsSnapshot, UserValueSnapshot]:
        """Persist compatibility and dual-axis fields in one database statement."""

        if legacy.user_id != value.user_id:
            raise ValueError("snapshot user IDs must match")
        if legacy.calculated_at != value.calculated_at:
            raise ValueError("snapshot calculation times must match")

        def evidence(items: Any) -> str:
            return json.dumps(list(items), separators=(",", ":"), ensure_ascii=True)

        payload = (
            legacy.user_id,
            legacy.status,
            legacy.reason_code,
            legacy.human_readable_reason,
            evidence(legacy.evidence_event_ids),
            value.lifecycle_segment,
            value.lifecycle_reason_code,
            value.lifecycle_reason,
            evidence(value.lifecycle_evidence),
            value.operational_state,
            value.operational_reason_code,
            value.operational_reason,
            evidence(value.operational_evidence),
            utc_iso(value.activated_at) if value.activated_at else None,
            (
                utc_iso(value.last_meaningful_activity_at)
                if value.last_meaningful_activity_at
                else None
            ),
            value.inactive_days,
            value.active_days_30d,
            value.successful_backtests_30d,
            utc_iso(value.calculated_at),
        )
        columns = """
            user_id, status, reason_code, human_readable_reason,
            evidence_event_ids_json, lifecycle_segment, lifecycle_reason_code,
            lifecycle_reason, lifecycle_evidence_json, operational_state,
            operational_reason_code, operational_reason,
            operational_evidence_json, activated_at,
            last_meaningful_activity_at, inactive_days, active_days_30d,
            successful_backtests_30d, calculated_at
        """
        updates = """
            status=excluded.status,
            reason_code=excluded.reason_code,
            human_readable_reason=excluded.human_readable_reason,
            evidence_event_ids_json=excluded.evidence_event_ids_json,
            lifecycle_segment=excluded.lifecycle_segment,
            lifecycle_reason_code=excluded.lifecycle_reason_code,
            lifecycle_reason=excluded.lifecycle_reason,
            lifecycle_evidence_json=excluded.lifecycle_evidence_json,
            operational_state=excluded.operational_state,
            operational_reason_code=excluded.operational_reason_code,
            operational_reason=excluded.operational_reason,
            operational_evidence_json=excluded.operational_evidence_json,
            activated_at=excluded.activated_at,
            last_meaningful_activity_at=excluded.last_meaningful_activity_at,
            inactive_days=excluded.inactive_days,
            active_days_30d=excluded.active_days_30d,
            successful_backtests_30d=excluded.successful_backtests_30d,
            calculated_at=excluded.calculated_at
        """
        placeholders = ", ".join(["%s" if self.is_postgres else "?"] * len(payload))
        sql = f"""
            INSERT INTO user_analytics_snapshots ({columns})
            VALUES ({placeholders})
            ON CONFLICT(user_id) DO UPDATE SET {updates}
        """
        with self.base_store._get_connection() as conn:
            if self.is_postgres:
                with conn.cursor() as cur:
                    cur.execute(sql, payload)
            else:
                conn.execute(sql, payload)
        return legacy, value

    def get_snapshot(self, user_id: int) -> UserAnalyticsSnapshot | None:
        subject_id = positive_user_id(user_id)
        if self.is_postgres:
            with self.base_store._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT * FROM user_analytics_snapshots WHERE user_id = %s",
                        (subject_id,),
                    )
                    row = cur.fetchone()
        else:
            with self.base_store._get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM user_analytics_snapshots WHERE user_id = ?",
                    (subject_id,),
                ).fetchone()
        if row is None:
            return None
        return UserAnalyticsSnapshot(
            user_id=int(row["user_id"]),
            status=str(row["status"]),
            reason_code=str(row["reason_code"]),
            human_readable_reason=str(row["human_readable_reason"]),
            evidence_event_ids=list(json.loads(str(row["evidence_event_ids_json"]))),
            calculated_at=_parse_timestamp(row["calculated_at"]),
        )

    def list_stale_user_ids(
        self,
        *,
        before: datetime | None = None,
        now: datetime | None = None,
        limit: int,
        include_time_transitions: bool = False,
    ) -> list[int]:
        page_size = positive_limit(limit)
        if before is None and now is None:
            raise ValueError("before or now is required")
        current = _require_utc(now, "now") if now is not None else None
        stale_before = before or (current - timedelta(minutes=15))
        cutoff = utc_iso(_require_utc(stale_before, "before"))
        transition_sql = ""
        transition_params: list[str] = []
        if include_time_transitions:
            if current is None:
                raise ValueError("now is required for time transitions")
            day_start = datetime.combine(current.date(), time.min, tzinfo=timezone.utc)
            placeholder = "%s" if self.is_postgres else "?"
            transition_sql = f"""
                OR snapshots.lifecycle_segment IS NULL
                OR snapshots.calculated_at < {placeholder}
                OR NOT EXISTS (
                    SELECT 1
                    FROM user_lifecycle_daily_snapshots AS daily
                    WHERE daily.user_id = users.id
                      AND daily.snapshot_date = {placeholder}
                )
            """
            transition_params = [utc_iso(day_start), current.date().isoformat()]
        if self.is_postgres:
            with self.base_store._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT users.id
                        FROM users
                        LEFT JOIN user_analytics_snapshots AS snapshots
                          ON snapshots.user_id = users.id
                        LEFT JOIN analytics_subject_settings AS settings
                          ON settings.user_id = users.id
                        WHERE users.role <> 'admin'
                          AND COALESCE(settings.excluded, FALSE) = FALSE
                          AND (
                              snapshots.user_id IS NULL
                              OR snapshots.calculated_at < %s
                              {transition_sql}
                          )
                        ORDER BY users.id
                        LIMIT %s
                        """,
                        (cutoff, *transition_params, page_size),
                    )
                    rows = cur.fetchall()
        else:
            with self.base_store._get_connection() as conn:
                rows = conn.execute(
                    f"""
                    SELECT users.id
                    FROM users
                    LEFT JOIN user_analytics_snapshots AS snapshots
                      ON snapshots.user_id = users.id
                    LEFT JOIN analytics_subject_settings AS settings
                      ON settings.user_id = users.id
                    WHERE users.role <> 'admin'
                      AND COALESCE(settings.excluded, 0) = 0
                      AND (
                          snapshots.user_id IS NULL
                          OR snapshots.calculated_at < ?
                          {transition_sql}
                      )
                    ORDER BY users.id
                    LIMIT ?
                    """,
                    (cutoff, *transition_params, page_size),
                ).fetchall()
        return [int(row["id"]) for row in rows]


def _ordered_evidence(events: list[AnalyticsEventRecord]) -> list[str]:
    by_id = sorted(events, key=lambda event: event.event_id)
    ordered = sorted(
        by_id,
        key=lambda event: event.occurred_at.astimezone(timezone.utc),
        reverse=True,
    )
    return [event.event_id for event in ordered[:5]]


def _snapshot(
    *,
    user_id: int,
    status: str,
    reason_code: str,
    evidence: list[AnalyticsEventRecord],
    now: datetime,
) -> UserAnalyticsSnapshot:
    return UserAnalyticsSnapshot(
        user_id=user_id,
        status=status,
        reason_code=reason_code,
        human_readable_reason=_REASONS[reason_code],
        evidence_event_ids=_ordered_evidence(evidence),
        calculated_at=now,
    )


def calculate_user_state(
    user_id: int,
    *,
    now: datetime | None = None,
    store: AnalyticsStateStore | None = None,
) -> UserAnalyticsSnapshot:
    subject_id = positive_user_id(user_id)
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("now must include a timezone")
    current = current.astimezone(timezone.utc)
    state_store = store or AnalyticsStateStore()
    user = state_store.get_user(subject_id)
    if user is None:
        raise LookupError("Analytics user was not found")
    events = state_store.list_user_events(subject_id, now=current)
    events_desc = sorted(events, key=lambda event: event.occurred_at, reverse=True)
    attempted_core = any(
        event.event_group == "run"
        or (
            event.event_name == "safe_error_recorded"
            and event.source_record_type == "run"
        )
        for event in events
    )

    blocking_categories = {
        "credential_missing",
        "provider_unavailable",
        "credits_unavailable",
        "model_not_allowed",
    }
    resolvers = {
        "credential_verified",
        "credential_reverified",
        "credential_defaulted",
        "credits_settled",
        "backtest_completed",
    }
    latest_resolver = next(
        (event for event in events_desc if event.event_name in resolvers),
        None,
    )
    blocker = next(
        (
            event
            for event in events_desc
            if event.event_name == "safe_error_recorded"
            and event.error_category in blocking_categories
        ),
        None,
    )
    if (
        attempted_core
        and blocker is not None
        and (
            latest_resolver is None or blocker.occurred_at > latest_resolver.occurred_at
        )
    ):
        reason = (
            "provider_disabled"
            if blocker.error_category in {"provider_unavailable", "model_not_allowed"}
            else "billing_lane_unavailable"
        )
        return _snapshot(
            user_id=subject_id,
            status="blocked",
            reason_code=reason,
            evidence=[blocker],
            now=current,
        )

    invalid_credential = next(
        (
            event
            for event in events_desc
            if event.event_name == "safe_error_recorded"
            and event.error_category == "credential_invalid"
        ),
        None,
    )
    credential_resolver = next(
        (
            event
            for event in events_desc
            if event.event_name
            in {"credential_verified", "credential_reverified", "credential_defaulted"}
        ),
        None,
    )
    if invalid_credential is not None and (
        credential_resolver is None
        or invalid_credential.occurred_at > credential_resolver.occurred_at
    ):
        return _snapshot(
            user_id=subject_id,
            status="needs_attention",
            reason_code="invalid_default_credential",
            evidence=[invalid_credential],
            now=current,
        )

    terminal_24h = [
        event
        for event in events_desc
        if event.event_name
        in {"backtest_completed", "backtest_failed", "backtest_cancelled"}
        and current - timedelta(hours=24) <= event.occurred_at <= current
    ]
    newest_three = terminal_24h[:3]
    if len(newest_three) == 3 and all(
        event.event_name == "backtest_failed" for event in newest_three
    ):
        return _snapshot(
            user_id=subject_id,
            status="needs_attention",
            reason_code="three_consecutive_failed_runs",
            evidence=newest_three,
            now=current,
        )

    meaningful = [event for event in events_desc if is_meaningful_event(event)]
    last_meaningful = meaningful[0] if meaningful else None
    created_at = _parse_timestamp(user["created_at"])
    if (
        last_meaningful is not None
        and last_meaningful.occurred_at < current - timedelta(days=30)
    ) or (last_meaningful is None and created_at < current - timedelta(days=30)):
        return _snapshot(
            user_id=subject_id,
            status="dormant",
            reason_code="no_meaningful_activity_30d",
            evidence=([last_meaningful] if last_meaningful else []),
            now=current,
        )

    successes = [
        event for event in events_desc if event.event_name == "backtest_completed"
    ]
    if not successes:
        signup = next(
            (event for event in events_desc if event.event_name == "account_signed_up"),
            None,
        )
        return _snapshot(
            user_id=subject_id,
            status="onboarding",
            reason_code="no_successful_run",
            evidence=([signup] if signup else []),
            now=current,
        )

    return _snapshot(
        user_id=subject_id,
        status="active",
        reason_code="recent_successful_run_and_activity",
        evidence=[successes[0]] + ([last_meaningful] if last_meaningful else []),
        now=current,
    )


def _require_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return value.astimezone(timezone.utc)


def _calculate_user_value_snapshot(
    user_id: int,
    *,
    now: datetime,
    state_store: AnalyticsStateStore,
    value_store: ValueAnalyticsStore,
) -> UserValueSnapshot:
    subject_id = positive_user_id(user_id)
    user = state_store.get_user(subject_id)
    if user is None:
        raise LookupError("Analytics user was not found")

    events = [
        event
        for event in state_store.list_user_events(subject_id, now=now)
        if event.occurred_at.astimezone(timezone.utc) <= now
    ]
    activity_start = now - timedelta(days=180)
    credit_activity = tuple(
        timestamp.astimezone(timezone.utc)
        for timestamp in value_store.list_credit_activity(
            [subject_id],
            start=activity_start,
            end=now + timedelta(microseconds=1),
        ).get(subject_id, ())
        if timestamp.astimezone(timezone.utc) <= now
    )
    previous = value_store.get_current_snapshot(subject_id)

    successes = [
        event.occurred_at.astimezone(timezone.utc)
        for event in events
        if event.event_name == "backtest_completed"
    ]
    activation_candidates = list(successes)
    if (
        previous is not None
        and previous.activated_at is not None
        and previous.activated_at <= now
    ):
        activation_candidates.append(previous.activated_at)
    activated_at = min(activation_candidates) if activation_candidates else None

    meaningful_activity = [
        event.occurred_at.astimezone(timezone.utc)
        for event in events
        if is_lifecycle_activity(event)
    ]
    meaningful_activity.extend(credit_activity)
    if (
        previous is not None
        and previous.last_meaningful_activity_at is not None
        and previous.last_meaningful_activity_at <= now
    ):
        meaningful_activity.append(previous.last_meaningful_activity_at)
    last_meaningful = max(meaningful_activity) if meaningful_activity else None

    window_start = datetime.combine(
        now.date() - timedelta(days=29),
        time.min,
        tzinfo=timezone.utc,
    )
    active_days = {
        timestamp.date()
        for timestamp in meaningful_activity
        if window_start <= timestamp <= now
    }
    successful_backtests = sum(
        1 for timestamp in successes if window_start <= timestamp <= now
    )
    lifecycle = calculate_lifecycle(
        LifecycleInputs(
            user_id=subject_id,
            created_at=_parse_timestamp(user["created_at"]),
            first_successful_backtest_at=activated_at,
            last_meaningful_activity_at=last_meaningful,
            active_days_30d=len(active_days),
            successful_backtests_30d=successful_backtests,
        ),
        now,
    )
    facts = value_store.get_operational_facts(subject_id, now=now)
    operational = calculate_operational_state(
        OperationalSignals(
            user_id=subject_id,
            account_restricted=facts.account_restricted,
            usable_billing_lane=facts.usable_billing_lane,
            selected_provider_enabled=facts.selected_provider_enabled,
            default_credential_status=facts.default_credential_status,
            failed_terminal_runs_24h=facts.failed_terminal_runs_24h,
            run_beyond_safe_deadline=facts.run_beyond_safe_deadline,
        ),
        now,
    )
    return UserValueSnapshot(
        user_id=subject_id,
        lifecycle_segment=lifecycle.segment,
        lifecycle_reason_code=lifecycle.reason_code,
        lifecycle_reason=lifecycle.reason,
        lifecycle_evidence=lifecycle.evidence,
        operational_state=operational.state,
        operational_reason_code=operational.reason_code,
        operational_reason=operational.reason,
        operational_evidence=operational.evidence,
        activated_at=lifecycle.activated_at,
        last_meaningful_activity_at=lifecycle.last_meaningful_activity_at,
        inactive_days=lifecycle.inactive_days,
        active_days_30d=lifecycle.active_days_30d,
        successful_backtests_30d=lifecycle.successful_backtests_30d,
        calculated_at=now,
    )


def calculate_user_value_snapshot(
    user_id: int,
    *,
    now: datetime | None = None,
    state_store: AnalyticsStateStore | None = None,
    value_store: ValueAnalyticsStore | None = None,
) -> UserValueSnapshot:
    current = _require_utc(now or datetime.now(timezone.utc), "now")
    states = state_store or AnalyticsStateStore()
    values = value_store or ValueAnalyticsStore(states.base_store)
    snapshot = _calculate_user_value_snapshot(
        user_id,
        now=current,
        state_store=states,
        value_store=values,
    )
    return values.upsert_current_snapshot(snapshot)


def recalculate_user_snapshots(
    user_id: int,
    *,
    now: datetime | None = None,
    state_store: AnalyticsStateStore | None = None,
    value_store: ValueAnalyticsStore | None = None,
) -> tuple[UserAnalyticsSnapshot, UserValueSnapshot]:
    """Recalculate legacy and dual-axis projections from the same evidence."""

    current = _require_utc(now or datetime.now(timezone.utc), "now")
    states = state_store or AnalyticsStateStore()
    values = value_store or ValueAnalyticsStore(states.base_store)
    legacy = calculate_user_state(user_id, now=current, store=states)
    value = _calculate_user_value_snapshot(
        user_id,
        now=current,
        state_store=states,
        value_store=values,
    )
    if getattr(values, "analytics_base", None) is states.base_store:
        return states.upsert_combined_snapshot(legacy, value)
    states.upsert_snapshot(legacy)
    values.upsert_current_snapshot(value)
    return legacy, value


def recalculate_user_snapshot(
    user_id: int,
    *,
    now: datetime | None = None,
    store: AnalyticsStateStore | None = None,
) -> UserAnalyticsSnapshot:
    state_store = store or AnalyticsStateStore()
    snapshot = calculate_user_state(user_id, now=now, store=state_store)
    return state_store.upsert_snapshot(snapshot)


def repair_stale_snapshots(
    *,
    now: datetime | None = None,
    limit: int = 100,
    stale_after: timedelta = timedelta(minutes=15),
    store: AnalyticsStateStore | None = None,
) -> int:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("now must include a timezone")
    state_store = store or AnalyticsStateStore()
    user_ids = state_store.list_stale_user_ids(
        before=current - stale_after,
        limit=limit,
    )
    repaired = 0
    for user_id in user_ids:
        try:
            recalculate_user_snapshot(user_id, now=current, store=state_store)
            repaired += 1
        except Exception as exc:
            print(
                "WARNING: analytics.snapshot_repair_failed "
                f"category={type(exc).__name__[:80]}"
            )
    return repaired


def repair_stale_value_snapshots(
    *,
    now: datetime | None = None,
    limit: int = 100,
    state_store: AnalyticsStateStore | None = None,
    value_store: ValueAnalyticsStore | None = None,
) -> int:
    current = _require_utc(now or datetime.now(timezone.utc), "now")
    states = state_store or AnalyticsStateStore()
    values = value_store or ValueAnalyticsStore(states.base_store)
    user_ids = states.list_stale_user_ids(
        now=current,
        limit=limit,
        include_time_transitions=True,
    )
    repaired = 0
    for user_id in user_ids:
        try:
            _legacy, snapshot = recalculate_user_snapshots(
                user_id,
                now=current,
                state_store=states,
                value_store=values,
            )
            values.upsert_daily_snapshot(
                UserLifecycleDailySnapshot(
                    snapshot_date=current.date(),
                    user_id=user_id,
                    lifecycle_segment=snapshot.lifecycle_segment,
                    lifecycle_reason_code=snapshot.lifecycle_reason_code,
                    data_quality="complete",
                    calculated_at=current,
                )
            )
            repaired += 1
        except Exception as exc:
            print(
                "WARNING: analytics.value_snapshot_repair_failed "
                f"category={type(exc).__name__[:80]}"
            )
    return repaired


__all__ = [
    "AnalyticsStateStore",
    "UserAnalyticsSnapshot",
    "calculate_user_value_snapshot",
    "calculate_user_state",
    "recalculate_user_snapshot",
    "recalculate_user_snapshots",
    "repair_stale_snapshots",
    "repair_stale_value_snapshots",
]
