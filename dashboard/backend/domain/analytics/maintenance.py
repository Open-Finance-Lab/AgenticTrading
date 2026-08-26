"""Bounded Analytics rollup and snapshot repair maintenance."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from threading import Lock
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AnalyticsMaintenanceReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rollup_days: tuple[date, ...]
    rollup_rebuilt: bool = False
    repaired_snapshots: int = Field(default=0, ge=0)
    failures: int = Field(default=0, ge=0)


_guard_lock = Lock()
_last_rollup_day: date | None = None


def _current_utc(value: datetime | None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("now must include a timezone")
    return current.astimezone(timezone.utc)


def reset_maintenance_guard_for_tests() -> None:
    """Reset only the process-local scheduling guard."""

    global _last_rollup_day
    with _guard_lock:
        _last_rollup_day = None


def run_analytics_maintenance(
    *,
    now: datetime | None = None,
    snapshot_limit: int = 100,
    rebuild_rollup: Any | None = None,
    repair_snapshots: Any | None = None,
) -> AnalyticsMaintenanceReport:
    """Rebuild yesterday once per process and repair one bounded snapshot batch."""

    global _last_rollup_day
    current = _current_utc(now)
    if isinstance(snapshot_limit, bool) or not isinstance(snapshot_limit, int):
        raise ValueError("snapshot_limit must be an integer")
    page_size = max(1, min(snapshot_limit, 100))
    completed_day = current.date() - timedelta(days=1)
    failures = 0
    rebuilt = False

    if rebuild_rollup is None:
        from .rollups import rollup_day

        rebuild_rollup = rollup_day
    if repair_snapshots is None:
        from .states import repair_stale_snapshots

        repair_snapshots = repair_stale_snapshots

    with _guard_lock:
        should_rebuild = _last_rollup_day != completed_day
        if should_rebuild:
            # Reserve the day before doing I/O so concurrent reaper passes do
            # not rebuild it twice.  A failure clears the reservation below.
            _last_rollup_day = completed_day
    if should_rebuild:
        try:
            rebuild_rollup(
                completed_day,
                now=datetime.combine(
                    current.date(),
                    datetime.min.time(),
                    tzinfo=timezone.utc,
                ),
            )
            rebuilt = True
        except Exception as exc:
            failures += 1
            with _guard_lock:
                if _last_rollup_day == completed_day:
                    _last_rollup_day = None
            print(
                "WARNING: analytics.rollup_maintenance_failed "
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
        rollup_days=(completed_day,),
        rollup_rebuilt=rebuilt,
        repaired_snapshots=max(0, repaired),
        failures=failures,
    )


__all__ = [
    "AnalyticsMaintenanceReport",
    "reset_maintenance_guard_for_tests",
    "run_analytics_maintenance",
]
