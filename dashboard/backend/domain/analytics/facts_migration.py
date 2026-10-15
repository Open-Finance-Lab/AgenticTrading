"""Startup migration from the five-state snapshot tables to the fact tables.

Runs on every boot, twice: synchronously from ``app.py``'s startup hook and
again as the daily-facts worker's ``prepare`` step (``daily_job.py``). Both
steps are idempotent, so the repeat costs nothing.

``user_activity`` was first seeded from ``user_analytics_snapshots`` when it
was created; the legacy row kept being maintained until the snapshot sweep was
deleted, so the same seed runs again immediately before both legacy tables are
dropped (design SS6.9 "Migration of existing history", SS11). Between the two,
every ``user_lifecycle_daily_snapshots`` day that no fact row covers is rolled
up into the anonymous lifecycle rollups, the only copy of it the drop leaves.
After the first boot that drops them, both reads find nothing and the drop is
``IF EXISTS``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StartupMigrationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    activity_seeded: int = Field(default=0, ge=0)
    legacy_days_rolled_up: int = Field(default=0, ge=0)


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
    """Re-seed ``user_activity``, roll up uncovered legacy days, then drop.

    The order is the point: the drop must never run before the seed has read
    the final legacy row, or every pre-existing activation date is lost, nor
    before the legacy lifecycle days that no fact row covers are rolled up,
    or the movement history they hold is lost. Either step raising skips the
    drop; the next boot tries again.
    """
    current = _utc_now(now)
    if value_store is None:
        from .repository import analytics_store
        from .value_repository import build_value_analytics_store

        value_store = build_value_analytics_store(analytics_store)
    from .rollups import rollup_legacy_lifecycle_days

    seeded = value_store.seed_activity_from_snapshots(now=current)
    rolled_up = rollup_legacy_lifecycle_days(store=value_store)
    value_store.analytics_base.drop_legacy_snapshot_tables()
    return StartupMigrationReport(
        activity_seeded=seeded, legacy_days_rolled_up=len(rolled_up)
    )


__all__ = [
    "StartupMigrationReport",
    "run_startup_migrations",
]
