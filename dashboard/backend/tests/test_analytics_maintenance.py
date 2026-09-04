"""Analytics maintenance is bounded, idempotent, and safe to register."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import dashboard.backend.domain.analytics.maintenance as maintenance


NOW = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)


def test_maintenance_rebuilds_one_day_and_bounds_snapshot_repairs():
    maintenance.reset_maintenance_guard_for_tests()
    rollup_calls = []
    repair_limits = []
    value_repair_limits = []
    repair_order = []
    repair_results = iter([25, 0])
    value_repair_results = iter([12, 0])

    def rebuild(day, **kwargs):
        rollup_calls.append((day, kwargs["now"]))

    def repair(**kwargs):
        repair_order.append("legacy")
        repair_limits.append(kwargs["limit"])
        return next(repair_results)

    def repair_values(**kwargs):
        repair_order.append("value")
        value_repair_limits.append(kwargs["limit"])
        return next(value_repair_results)

    first = maintenance.run_analytics_maintenance(
        now=NOW,
        snapshot_limit=250,
        rebuild_rollup=rebuild,
        repair_snapshots=repair,
        repair_value_snapshots=repair_values,
    )
    second = maintenance.run_analytics_maintenance(
        now=NOW,
        snapshot_limit=250,
        rebuild_rollup=rebuild,
        repair_snapshots=repair,
        repair_value_snapshots=repair_values,
    )

    assert first.rollup_days == (date(2026, 8, 25),)
    assert second.rollup_days == first.rollup_days
    assert first.rollup_rebuilt is True
    assert second.rollup_rebuilt is False
    assert first.repaired_snapshots == 25
    assert second.repaired_snapshots == 0
    assert first.repaired_value_snapshots == 12
    assert second.repaired_value_snapshots == 0
    assert repair_limits == [100, 100]
    assert value_repair_limits == [100, 100]
    assert repair_order == ["value", "legacy", "value", "legacy"]
    assert len(rollup_calls) == 1


def test_maintenance_isolates_rollup_and_snapshot_failures():
    maintenance.reset_maintenance_guard_for_tests()

    def fail_rollup(*_args, **_kwargs):
        raise RuntimeError("private database detail")

    def fail_repair(**_kwargs):
        raise RuntimeError("private user detail")

    def fail_value_repair(**_kwargs):
        raise RuntimeError("private value projection detail")

    report = maintenance.run_analytics_maintenance(
        now=NOW,
        rebuild_rollup=fail_rollup,
        repair_snapshots=fail_repair,
        repair_value_snapshots=fail_value_repair,
    )

    assert report.rollup_rebuilt is False
    assert report.repaired_snapshots == 0
    assert report.repaired_value_snapshots == 0
    assert report.failures == 3


def test_app_registers_analytics_maintenance_through_reaper():
    app_file = Path(__file__).resolve().parents[1] / "app.py"
    source = app_file.read_text(encoding="utf-8")

    assert "register_reaper_sweep(run_analytics_maintenance)" in source
    assert "analytics.maintenance_registration_failed" in source
