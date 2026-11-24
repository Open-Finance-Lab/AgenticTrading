"""Startup migration from the five-state snapshot tables to the fact tables.

Runs once per boot, as the daily-facts worker's ``prepare`` step
(``daily_job.py``), off the web process's event loop. A failed run is retried
before each later tick until one succeeds. Every step is idempotent, so a
retry, or the next boot, costs only the checks that find nothing to do.

``user_activity`` was first seeded from ``user_analytics_snapshots`` when it
was created; the legacy row kept being maintained until the snapshot sweep was
deleted, so the same seed runs again immediately before both legacy tables are
dropped (design SS6.9 "Migration of existing history", SS11). Between the two,
every ``user_lifecycle_daily_snapshots`` day that no fact row covers is rolled
up into the anonymous lifecycle rollups, the only copy of it the drop leaves,
and the fact days the PR A history copy wrote without a transition log get one.
After the first boot that drops them, those reads find nothing and the drop is
``IF EXISTS``. Last, rolled-up days that predate the ``activation_users``
rollup metric are filled from the raw events still inside retention.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StartupMigrationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    activity_seeded: int = Field(default=0, ge=0)
    legacy_days_rolled_up: int = Field(default=0, ge=0)
    transitions_backfilled: int = Field(default=0, ge=0)
    activation_days_backfilled: int = Field(default=0, ge=0)


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
    """Re-seed ``user_activity``, roll up and derive legacy history, drop, fill.

    The order is the point: the drop must never run before the seed has read
    the final legacy row, or every pre-existing activation date is lost, nor
    before the legacy lifecycle days that no fact row covers are rolled up,
    or the movement history they hold is lost, nor before the transition
    backfill has read the legacy day preceding the first copied fact day.
    Any of those raising skips the drop; the next attempt tries again.

    The activation backfill shares none of that chain, so it runs whether
    the chain succeeded or not -- a drop that keeps losing its lock must not
    also keep the funnel unavailable. The first error (chain, then backfill)
    is raised once both have had their turn, so the worker retries.
    """
    current = _utc_now(now)
    if value_store is None:
        from .repository import analytics_store
        from .value_repository import build_value_analytics_store

        value_store = build_value_analytics_store(analytics_store)
    from .retention import RAW_EVENT_RETENTION_DAYS
    from .rollups import AnalyticsRollupStore, rollup_legacy_lifecycle_days

    errors: list[Exception] = []
    seeded, rolled_up, transitions = 0, (), 0
    try:
        seeded = value_store.seed_activity_from_snapshots(now=current)
        rolled_up = rollup_legacy_lifecycle_days(store=value_store)
        transitions = value_store.backfill_transitions_from_history(now=current)
        value_store.analytics_base.drop_legacy_snapshot_tables()
    except Exception as exc:
        errors.append(exc)
    activation_days: tuple[Any, ...] = ()
    try:
        # The oldest day whose raw events are still whole: retention deletes
        # events older than RAW_EVENT_RETENTION_DAYS, so the boundary day may
        # be partly gone and is left alone.
        activation_days = AnalyticsRollupStore(
            value_store.analytics_base
        ).backfill_activation_users(
            since=current.date() - timedelta(days=RAW_EVENT_RETENTION_DAYS - 1),
            until=current.date(),
        )
    except Exception as exc:
        errors.append(exc)
    if errors:
        raise errors[0]
    return StartupMigrationReport(
        activity_seeded=seeded,
        legacy_days_rolled_up=len(rolled_up),
        transitions_backfilled=transitions,
        activation_days_backfilled=len(activation_days),
    )


__all__ = [
    "StartupMigrationReport",
    "run_startup_migrations",
]
