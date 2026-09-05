"""Bounded, resumable lifecycle history reconstruction."""

from __future__ import annotations

import base64
import json
from datetime import date, datetime, time, timedelta, timezone
from typing import Literal, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .lifecycle import LifecycleInputs, calculate_lifecycle, is_lifecycle_activity
from .repository import analytics_store
from .repository_common import positive_limit, positive_user_id
from .rollups import AnalyticsRollupStore
from .value_repository import (
    ProjectionJob,
    UserLifecycleDailySnapshot,
    ValueAnalyticsStore,
)


UTC = timezone.utc
BACKFILL_JOB_NAME = "lifecycle_previous_8_weeks"
MAX_BACKFILL_DAYS = 56


def _utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return value.astimezone(UTC)


def _timestamp(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


class LifecycleBackfillUser(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: int = Field(gt=0)
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def require_created_timezone(cls, value: datetime) -> datetime:
        return _utc(value, "created_at")


class LifecycleBackfillEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_name: str = Field(min_length=1, max_length=64)
    occurred_at: datetime

    @field_validator("occurred_at")
    @classmethod
    def require_occurred_timezone(cls, value: datetime) -> datetime:
        return _utc(value, "occurred_at")


class LifecycleBackfillCursor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    last_user_id: int = Field(gt=0)
    window_start: date
    window_end: date

    def encode(self) -> str:
        payload = json.dumps(
            {
                "last_user_id": self.last_user_id,
                "window_start": self.window_start.isoformat(),
                "window_end": self.window_end.isoformat(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")

    @classmethod
    def decode(cls, value: str) -> "LifecycleBackfillCursor":
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError("cursor must be a non-empty encoded string")
        try:
            padded = value + "=" * (-len(value) % 4)
            payload = base64.b64decode(
                padded.encode("ascii"),
                altchars=b"-_",
                validate=True,
            )
            data = json.loads(payload.decode("ascii"))
            return cls.model_validate(data)
        except Exception as exc:
            raise ValueError("cursor is invalid") from exc


class LifecycleBackfillReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    processed_users: int = Field(ge=0)
    written_rows: int = Field(ge=0)
    partial_days: int = Field(ge=0)
    next_cursor: str | None = None
    complete: bool


class LifecycleBackfillSourceContract(Protocol):
    def list_users(
        self,
        *,
        after_user_id: int,
        limit: int,
    ) -> Sequence[LifecycleBackfillUser]: ...

    def list_evidence(
        self,
        user_id: int,
        *,
        start: datetime,
        end: datetime,
    ) -> Sequence[LifecycleBackfillEvidence]: ...

    def quality_for(
        self,
        snapshot_date: date,
        *,
        required_from: date,
    ) -> Literal["complete", "partial"]: ...


class LifecycleBackfillSource:
    """Read display-safe lifecycle evidence from authoritative local stores."""

    def __init__(
        self,
        *,
        analytics_base=None,
        value_store: ValueAnalyticsStore | None = None,
        complete_from: date | None = None,
    ) -> None:
        self.analytics_base = analytics_base or analytics_store
        self.value_store = value_store or ValueAnalyticsStore(self.analytics_base)
        self.rollups = AnalyticsRollupStore(self.analytics_base)
        self.complete_from = complete_from
        self.is_postgres = hasattr(self.analytics_base, "database_url")

    def list_users(
        self,
        *,
        after_user_id: int,
        limit: int,
    ) -> tuple[LifecycleBackfillUser, ...]:
        cursor = max(0, int(after_user_id))
        page_size = positive_limit(limit, maximum=501)
        placeholder = "%s" if self.is_postgres else "?"
        sql = f"""
            SELECT users.id, users.created_at
            FROM users
            LEFT JOIN analytics_subject_settings AS settings
              ON settings.user_id = users.id
            WHERE users.id > {placeholder}
              AND users.role <> 'admin'
              AND COALESCE(settings.excluded, {"FALSE" if self.is_postgres else "0"}) = {"FALSE" if self.is_postgres else "0"}
            ORDER BY users.id
            LIMIT {placeholder}
        """
        with self.analytics_base._get_connection() as conn:
            if self.is_postgres:
                with conn.cursor() as db_cursor:
                    db_cursor.execute(sql, (cursor, page_size))
                    rows = db_cursor.fetchall()
            else:
                rows = conn.execute(sql, (cursor, page_size)).fetchall()
        return tuple(
            LifecycleBackfillUser(
                user_id=int(row["id"]),
                created_at=_timestamp(row["created_at"]),
            )
            for row in rows
        )

    def list_evidence(
        self,
        user_id: int,
        *,
        start: datetime,
        end: datetime,
    ) -> tuple[LifecycleBackfillEvidence, ...]:
        subject_id = positive_user_id(user_id)
        window_start = _utc(start, "start")
        window_end = _utc(end, "end")
        if window_end <= window_start:
            raise ValueError("end must be later than start")
        events = self.rollups.list_events(
            start=window_start,
            end=window_end,
            include_internal=True,
            user_id=subject_id,
        )
        evidence = [
            LifecycleBackfillEvidence(
                event_name=event.event_name,
                occurred_at=event.occurred_at,
            )
            for event in events
            if event.event_name == "backtest_completed" or is_lifecycle_activity(event)
        ]
        credit_activity = self.value_store.list_credit_activity(
            [subject_id],
            start=window_start,
            end=window_end,
        ).get(subject_id, ())
        evidence.extend(
            LifecycleBackfillEvidence(
                event_name="credits_settled",
                occurred_at=timestamp,
            )
            for timestamp in credit_activity
        )
        return tuple(sorted(evidence, key=lambda item: item.occurred_at))

    def quality_for(
        self,
        snapshot_date: date,
        *,
        required_from: date,
    ) -> Literal["complete", "partial"]:
        del required_from
        if self.complete_from is None or snapshot_date < self.complete_from:
            return "partial"
        return "complete"


def _window_days(start: date, end: date) -> tuple[date, ...]:
    if not isinstance(start, date) or isinstance(start, datetime):
        raise ValueError("start must be a date")
    if not isinstance(end, date) or isinstance(end, datetime):
        raise ValueError("end must be a date")
    if end <= start:
        raise ValueError("end must be later than start")
    length = (end - start).days
    if length > MAX_BACKFILL_DAYS:
        raise ValueError("backfill window must contain at most 56 days")
    return tuple(start + timedelta(days=offset) for offset in range(length))


def _inputs_at(
    user: LifecycleBackfillUser,
    evidence: Sequence[LifecycleBackfillEvidence],
    *,
    cutoff: datetime,
) -> LifecycleInputs:
    available = [
        item for item in evidence if user.created_at <= item.occurred_at < cutoff
    ]
    successes = [
        item.occurred_at
        for item in available
        if item.event_name == "backtest_completed"
    ]
    meaningful = [item.occurred_at for item in available if is_lifecycle_activity(item)]
    window_start = cutoff - timedelta(days=30)
    return LifecycleInputs(
        user_id=user.user_id,
        created_at=user.created_at,
        first_successful_backtest_at=min(successes) if successes else None,
        last_meaningful_activity_at=max(meaningful) if meaningful else None,
        active_days_30d=len(
            {
                timestamp.date()
                for timestamp in meaningful
                if window_start <= timestamp < cutoff
            }
        ),
        successful_backtests_30d=sum(
            1 for timestamp in successes if window_start <= timestamp < cutoff
        ),
    )


def backfill_lifecycle_history(
    *,
    start: date,
    end: date,
    batch_size: int = 100,
    cursor: str | None = None,
    now: datetime | None = None,
    source: LifecycleBackfillSourceContract,
    store: ValueAnalyticsStore,
) -> LifecycleBackfillReport:
    days = _window_days(start, end)
    current = _utc(now or datetime.now(UTC), "now")
    if end > current.date() + timedelta(days=1):
        raise ValueError("backfill window cannot extend beyond the current UTC day")
    page_size = positive_limit(batch_size, maximum=500)
    after_user_id = 0
    if cursor is not None:
        decoded = LifecycleBackfillCursor.decode(cursor)
        if decoded.window_start != start or decoded.window_end != end:
            raise ValueError("cursor window does not match the requested window")
        after_user_id = decoded.last_user_id

    page = tuple(source.list_users(after_user_id=after_user_id, limit=page_size + 1))
    page_ids = [user.user_id for user in page]
    if (
        len(page) > page_size + 1
        or page_ids != sorted(set(page_ids))
        or any(user_id <= after_user_id for user_id in page_ids)
    ):
        raise ValueError("backfill source returned an invalid user page")
    users = page[:page_size]
    partial_dates: set[date] = set()
    written_rows = 0
    evidence_start = datetime.combine(
        start - timedelta(days=30),
        time.min,
        tzinfo=UTC,
    )
    evidence_end = min(
        datetime.combine(end, time.min, tzinfo=UTC),
        current + timedelta(microseconds=1),
    )
    existing = {
        (snapshot.snapshot_date, snapshot.user_id): snapshot
        for snapshot in store.list_daily_snapshots(
            start=start,
            end=end,
            user_ids=[user.user_id for user in users],
        )
    }
    for user in users:
        evidence = source.list_evidence(
            user.user_id,
            start=evidence_start,
            end=evidence_end,
        )
        for snapshot_date in days:
            cutoff = min(
                datetime.combine(
                    snapshot_date + timedelta(days=1),
                    time.min,
                    tzinfo=UTC,
                ),
                current,
            )
            if user.created_at >= cutoff:
                continue
            result = calculate_lifecycle(
                _inputs_at(user, evidence, cutoff=cutoff),
                cutoff,
            )
            quality = source.quality_for(
                snapshot_date,
                required_from=snapshot_date - timedelta(days=29),
            )
            prior = existing.get((snapshot_date, user.user_id))
            if (
                prior is not None
                and prior.data_quality == "complete"
                and quality == "partial"
            ):
                continue
            if quality == "partial":
                partial_dates.add(snapshot_date)
            store.upsert_daily_snapshot(
                UserLifecycleDailySnapshot(
                    snapshot_date=snapshot_date,
                    user_id=user.user_id,
                    lifecycle_segment=result.segment,
                    lifecycle_reason_code=result.reason_code,
                    data_quality=quality,
                    calculated_at=current,
                )
            )
            written_rows += 1

    complete = len(page) <= page_size
    next_cursor = None
    if not complete and users:
        next_cursor = LifecycleBackfillCursor(
            last_user_id=users[-1].user_id,
            window_start=start,
            window_end=end,
        ).encode()
    return LifecycleBackfillReport(
        processed_users=len(users),
        written_rows=written_rows,
        partial_days=len(partial_dates),
        next_cursor=next_cursor,
        complete=complete,
    )


def run_lifecycle_backfill_batch(
    *,
    now: datetime | None = None,
    batch_size: int = 100,
    source: LifecycleBackfillSourceContract | None = None,
    store: ValueAnalyticsStore | None = None,
) -> LifecycleBackfillReport:
    current = _utc(now or datetime.now(UTC), "now")
    values = store or ValueAnalyticsStore()
    history_source = source or LifecycleBackfillSource(
        analytics_base=values.analytics_base,
        value_store=values,
    )
    job = values.get_projection_job(BACKFILL_JOB_NAME)
    if job is None:
        end = current.date() + timedelta(days=1)
        job = ProjectionJob(
            job_name=BACKFILL_JOB_NAME,
            window_start=end - timedelta(days=MAX_BACKFILL_DAYS),
            window_end=end,
            status="pending",
            updated_at=current,
        )
    if job.status == "complete":
        return LifecycleBackfillReport(
            processed_users=0,
            written_rows=0,
            partial_days=0,
            complete=True,
        )
    report = backfill_lifecycle_history(
        start=job.window_start,
        end=job.window_end,
        batch_size=batch_size,
        cursor=job.cursor,
        now=current,
        source=history_source,
        store=values,
    )
    values.save_projection_job(
        ProjectionJob(
            job_name=job.job_name,
            window_start=job.window_start,
            window_end=job.window_end,
            cursor=report.next_cursor,
            status="complete" if report.complete else "running",
            updated_at=current,
        )
    )
    return report


__all__ = [
    "BACKFILL_JOB_NAME",
    "LifecycleBackfillCursor",
    "LifecycleBackfillEvidence",
    "LifecycleBackfillReport",
    "LifecycleBackfillSource",
    "LifecycleBackfillSourceContract",
    "LifecycleBackfillUser",
    "backfill_lifecycle_history",
    "run_lifecycle_backfill_batch",
]
