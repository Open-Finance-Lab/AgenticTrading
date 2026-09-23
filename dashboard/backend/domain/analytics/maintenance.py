"""Throttled repair of the legacy Analytics snapshot rows.

Admin layer redesign PR A moved the day rollup and the lifecycle history backfill off
this reaper tick: the daily-facts job (``daily_facts.py``, on its own worker
thread, ``daily_job.py``) owns the day rollup now, so the two can never write
the same rows (design SS6.9 step 1, D23), and the eight-week history copy in
``facts_migration.py`` replaces the history copy. What remains is the two
24-hour-throttled snapshot repairs PR 0 left, which keep
``user_analytics_snapshots`` maintained for the read paths that still use it.
PR B moves those read paths and deletes this module.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AnalyticsMaintenanceReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repaired_snapshots: int = Field(default=0, ge=0)
    repaired_value_snapshots: int = Field(default=0, ge=0)
    failures: int = Field(default=0, ge=0)


def _current_utc(value: datetime | None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("now must include a timezone")
    return current.astimezone(timezone.utc)


def run_analytics_maintenance(
    *,
    now: datetime | None = None,
    snapshot_limit: int = 100,
    repair_snapshots: Any | None = None,
    repair_value_snapshots: Any | None = None,
) -> AnalyticsMaintenanceReport:
    """Repair bounded batches of stale dual-axis and legacy snapshots."""

    current = _current_utc(now)
    if isinstance(snapshot_limit, bool) or not isinstance(snapshot_limit, int):
        raise ValueError("snapshot_limit must be an integer")
    page_size = max(1, min(snapshot_limit, 100))
    failures = 0

    if repair_snapshots is None:
        from .states import repair_stale_snapshots

        repair_snapshots = repair_stale_snapshots
    if repair_value_snapshots is None:
        from .states import repair_stale_value_snapshots

        repair_value_snapshots = repair_stale_value_snapshots

    repaired_values = 0
    try:
        repaired_values = int(
            repair_value_snapshots(
                now=current,
                limit=page_size,
            )
        )
    except Exception as exc:
        failures += 1
        print(
            "WARNING: analytics.value_snapshot_maintenance_failed "
            f"category={type(exc).__name__[:80]}"
        )

    repaired = 0
    try:
        repaired = int(
            repair_snapshots(
                now=current,
                limit=page_size,
            )
        )
    except Exception as exc:
        failures += 1
        print(
            "WARNING: analytics.snapshot_maintenance_failed "
            f"category={type(exc).__name__[:80]}"
        )

    return AnalyticsMaintenanceReport(
        repaired_snapshots=max(0, repaired),
        repaired_value_snapshots=max(0, repaired_values),
        failures=failures,
    )


__all__ = [
    "AnalyticsMaintenanceReport",
    "run_analytics_maintenance",
]
