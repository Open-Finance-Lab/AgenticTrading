"""Bounded retention for raw Analytics events and Admin access records."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

from .models import RetentionResult
from .repository import analytics_store


RAW_EVENT_RETENTION_DAYS = 180
ADMIN_ACCESS_RETENTION_DAYS = 365
RETENTION_BATCH_SIZE = 1000
RETENTION_INTERVAL_SECONDS = 24 * 60 * 60
# Retention normally runs once a day. A capped run that reports more expired
# rows retries after a short interval instead of deferring the backlog for a
# full day.
RETENTION_BACKLOG_RETRY_SECONDS = 60
MAX_BATCHES_PER_RUN = 20


class AnalyticsRetentionService:
    """Delete expired detail in bounded batches while preserving aggregates."""

    def __init__(
        self,
        *,
        store,
        batch_size: int = RETENTION_BATCH_SIZE,
        max_batches: int = MAX_BATCHES_PER_RUN,
    ) -> None:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or not 1 <= batch_size <= 10_000
        ):
            raise ValueError("batch_size must be an integer from 1 through 10000")
        if (
            isinstance(max_batches, bool)
            or not isinstance(max_batches, int)
            or max_batches < 1
        ):
            raise ValueError("max_batches must be a positive integer")
        self.store = store
        self.batch_size = batch_size
        self.max_batches = max_batches

    def run_once(self, now: datetime | None = None) -> RetentionResult:
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError("now must include a timezone")
        current = current.astimezone(timezone.utc)
        raw_before = current - timedelta(days=RAW_EVENT_RETENTION_DAYS)
        access_before = current - timedelta(days=ADMIN_ACCESS_RETENTION_DAYS)
        raw_deleted = 0
        access_deleted = 0
        has_more_raw = False
        has_more_access = False

        for _batch in range(self.max_batches):
            result = self.store.delete_expired(
                raw_before=raw_before,
                access_before=access_before,
                batch_size=self.batch_size,
            )
            raw_deleted += result.raw_events_deleted
            access_deleted += result.access_rows_deleted
            has_more_raw = result.has_more_raw_events
            has_more_access = result.has_more_access_rows
            if not has_more_raw and not has_more_access:
                break

        return RetentionResult(
            raw_events_deleted=raw_deleted,
            access_rows_deleted=access_deleted,
            has_more_raw_events=has_more_raw,
            has_more_access_rows=has_more_access,
        )


class AnalyticsRetentionCoordinator:
    """Run retention daily, with bounded retries while expired rows remain."""

    def __init__(
        self,
        *,
        service: AnalyticsRetentionService,
        clock=time.monotonic,
        interval_seconds: float = RETENTION_INTERVAL_SECONDS,
    ) -> None:
        self.service = service
        self.clock = clock
        self.interval_seconds = interval_seconds
        self.consecutive_failures = 0
        self._next_run_at = 0.0
        self._lock = threading.Lock()

    def run_if_due(self) -> RetentionResult | None:
        if self.clock() < self._next_run_at:
            return None
        if not self._lock.acquire(blocking=False):
            return None
        try:
            now = self.clock()
            if now < self._next_run_at:
                return None
            self._next_run_at = now + self.interval_seconds
            try:
                result = self.service.run_once()
            except Exception as exc:
                self.consecutive_failures += 1
                print(
                    "WARNING: analytics.retention_failed "
                    f"consecutive_failures={self.consecutive_failures} "
                    f"category={type(exc).__name__}"
                )
                return None
            self.consecutive_failures = 0
            if result.has_more_raw_events or result.has_more_access_rows:
                self._next_run_at = now + min(
                    self.interval_seconds, RETENTION_BACKLOG_RETRY_SECONDS
                )
            return result
        finally:
            self._lock.release()


analytics_retention_service = AnalyticsRetentionService(store=analytics_store)
analytics_retention_coordinator = AnalyticsRetentionCoordinator(
    service=analytics_retention_service,
)
