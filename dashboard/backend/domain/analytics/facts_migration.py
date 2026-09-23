"""Startup migrations for the fact tables PR A creates.

Both are idempotent and both run again on every boot; the worker thread in
``daily_job.py`` calls ``run_startup_migrations`` once before its first tick.
PR B re-runs the same seed immediately before dropping the legacy tables,
because the legacy snapshot row keeps being maintained between the two PRs
(design SS6.9 "Migration of existing history", SS11).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


HISTORY_COPY_DAYS = 56


class StartupMigrationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    activity_seeded: int = Field(default=0, ge=0)
    history_copied: int = Field(default=0, ge=0)


def _utc_now(value: datetime | None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("now must include a timezone")
    return current.astimezone(timezone.utc)


def migrate_lifecycle_history(*, value_store: Any, now: datetime) -> int:
    """Copy the last eight weeks of legacy daily snapshots; returns rows inserted."""
    current = _utc_now(now)
    return value_store.copy_daily_snapshot_history(
        since=current.date() - timedelta(days=HISTORY_COPY_DAYS), now=current
    )


def run_startup_migrations(
    *,
    value_store: Any = None,
    now: datetime | None = None,
) -> StartupMigrationReport:
    """Seed ``user_activity`` from the legacy row, then copy the history."""
    current = _utc_now(now)
    if value_store is None:
        from .repository import analytics_store
        from .value_repository import build_value_analytics_store

        value_store = build_value_analytics_store(analytics_store)
    seeded = value_store.seed_activity_from_snapshots(now=current)
    copied = migrate_lifecycle_history(value_store=value_store, now=current)
    return StartupMigrationReport(activity_seeded=seeded, history_copied=copied)


__all__ = [
    "HISTORY_COPY_DAYS",
    "StartupMigrationReport",
    "migrate_lifecycle_history",
    "run_startup_migrations",
]
