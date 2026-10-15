"""Startup migration from the five-state snapshot tables to the fact tables.

Runs on every boot, twice: synchronously from ``app.py``'s startup hook and
again as the daily-facts worker's ``prepare`` step (``daily_job.py``). Both
steps are idempotent, so the repeat costs nothing.

``user_activity`` was first seeded from ``user_analytics_snapshots`` when it
was created; the legacy row kept being maintained until the snapshot sweep was
deleted, so the same seed runs again immediately before both legacy tables are
dropped (design SS6.9 "Migration of existing history", SS11). After the first
boot that drops them, the seed reads nothing and the drop is ``IF EXISTS``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StartupMigrationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    activity_seeded: int = Field(default=0, ge=0)


def _utc_now(value: datetime | None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("now must include a timezone")
    return current.astimezone(timezone.utc)


def run_startup_migrations(
    *,
    value_store: Any = None,
    now: datetime | None = None,
) -> StartupMigrationReport:
    """Re-seed ``user_activity`` from the legacy row, then drop the legacy tables.

    The order is the point: the drop must never run before the seed has read
    the final legacy row, or every pre-existing activation date is lost.
    """
    current = _utc_now(now)
    if value_store is None:
        from .repository import analytics_store
        from .value_repository import build_value_analytics_store

        value_store = build_value_analytics_store(analytics_store)
    seeded = value_store.seed_activity_from_snapshots(now=current)
    value_store.analytics_base.drop_legacy_snapshot_tables()
    return StartupMigrationReport(activity_seeded=seeded)


__all__ = [
    "StartupMigrationReport",
    "run_startup_migrations",
]
