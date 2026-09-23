"""Analytics maintenance is bounded, idempotent, and safe to register."""

from __future__ import annotations

import inspect
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import dashboard.backend.app as app_module
import dashboard.backend.domain.analytics.maintenance as maintenance


NOW = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)


def test_maintenance_repairs_bounded_batches_and_nothing_else():
    repair_limits = []
    value_repair_limits = []
    repair_order = []

    def repair(**kwargs):
        repair_order.append("legacy")
        repair_limits.append(kwargs["limit"])
        return 25

    def repair_values(**kwargs):
        repair_order.append("value")
        value_repair_limits.append(kwargs["limit"])
        return 12

    report = maintenance.run_analytics_maintenance(
        now=NOW,
        snapshot_limit=250,
        repair_snapshots=repair,
        repair_value_snapshots=repair_values,
    )

    assert report.repaired_snapshots == 25
    assert report.repaired_value_snapshots == 12
    assert report.failures == 0
    assert repair_limits == [100]
    assert value_repair_limits == [100]
    assert repair_order == ["value", "legacy"]


def test_maintenance_isolates_repair_failures():
    def fail_repair(**_kwargs):
        raise RuntimeError("private user detail")

    def fail_value_repair(**_kwargs):
        raise RuntimeError("private value projection detail")

    report = maintenance.run_analytics_maintenance(
        now=NOW,
        repair_snapshots=fail_repair,
        repair_value_snapshots=fail_value_repair,
    )

    assert report.repaired_snapshots == 0
    assert report.repaired_value_snapshots == 0
    assert report.failures == 2


def test_maintenance_no_longer_owns_rollups_or_the_lifecycle_backfill():
    """Design SS6.9 step 1 / D23: the daily job owns rollup_day now, so the
    reaper tick must not be able to write the same rollup rows."""
    parameters = inspect.signature(maintenance.run_analytics_maintenance).parameters
    source = inspect.getsource(maintenance)
