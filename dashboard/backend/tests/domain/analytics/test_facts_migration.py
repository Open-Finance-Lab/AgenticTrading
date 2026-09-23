"""Existing lifecycle history is carried into the fact table as labelled partial rows."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from dashboard.backend.domain.analytics.facts_migration import (
    HISTORY_COPY_DAYS,
    StartupMigrationReport,
    migrate_lifecycle_history,
    run_startup_migrations,
)
from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.value_repository import (
    UserDailyFact,
    UserLifecycleDailySnapshot,
    build_value_analytics_store,
)
from dashboard.backend.tests.domain.analytics.test_value_repository import (
    _value_snapshot,
)
from dashboard.backend.users import UserStore


NOW = datetime(2026, 9, 12, 0, 5, tzinfo=timezone.utc)


class MigrationFixture:
    def __init__(self, tmp_path):
        path = tmp_path / "migration.db"
        users = UserStore(db_path=path)
        self.user_id = int(users.create_user("m@example.test", "M", "SecurePass1!")["id"])
        users.apply_admin_patch(self.user_id, user_group="partner")
        self.analytics = AnalyticsStore(db_path=path)
        self.store = build_value_analytics_store(
            self.analytics,
            credits_base=object(),
            provider_base=object(),
            agent_base=object(),
            run_base=object(),
        )
        self.legacy_dates = [
            date(2026, 8, 20),
            date(2026, 9, 1),
            date(2026, 9, 10),
        ]
        for day in self.legacy_dates + [date(2026, 6, 1)]:  # June is older than 56 days
            self.store.upsert_daily_snapshot(
                UserLifecycleDailySnapshot(
                    snapshot_date=day,
                    user_id=self.user_id,
                    lifecycle_segment="growing",
                    lifecycle_reason_code="growing_activated_below_core_threshold",
                    data_quality="complete",
                    calculated_at=datetime.combine(
                        day + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
                    ),
                )
            )
        self.legacy_row_count = len(self.legacy_dates)

    def seed_complete_fact(self, day):
        self.store.upsert_daily_facts(
            [
                UserDailyFact(
                    snapshot_date=day,
                    user_id=self.user_id,
                    lifecycle_segment="core",
                    lifecycle_reason_code="core_repeated_value",
                    operational_state="healthy",
                    operational_reason_code=None,
                    tier="starter",
                    user_group="partner",
                    active=True,
                    runs_completed=3,
                    data_quality="complete",
                    calculated_at=NOW,
                )
            ]
        )

    def fact_for(self, day):
        return {row.user_id: row for row in self.store.list_facts_for_date(day)}[self.user_id]


def test_history_is_copied_as_partial(tmp_path):
    """Movement charts keep their history; the missing columns stay honest."""
    fixture = MigrationFixture(tmp_path)

    copied = migrate_lifecycle_history(value_store=fixture.store, now=NOW)

    assert copied == fixture.legacy_row_count
    for day in fixture.legacy_dates:
        row = fixture.fact_for(day)
        assert row.data_quality == "partial"
        assert row.lifecycle_segment == "growing"
        assert row.runs_completed == 0
        assert row.operator_cost_micro == 0
        assert row.tier == "unpaid"
        assert row.operational_state == "healthy"
        assert row.operational_reason_code is None
        assert row.user_group == "partner"  # read from users at the copy (D9)
        assert row.active is False
    assert fixture.store.list_facts_for_date(date(2026, 6, 1)) == []
    assert (NOW.date() - timedelta(days=HISTORY_COPY_DAYS)) > date(2026, 6, 1)


def test_migration_is_idempotent(tmp_path):
    fixture = MigrationFixture(tmp_path)
    migrate_lifecycle_history(value_store=fixture.store, now=NOW)

    second = migrate_lifecycle_history(value_store=fixture.store, now=NOW)

    assert second == 0


def test_migration_never_overwrites_a_complete_row(tmp_path):
    """A day the new job already computed wins over a migrated stub."""
    fixture = MigrationFixture(tmp_path)
    fixture.seed_complete_fact(date(2026, 9, 10))

    migrate_lifecycle_history(value_store=fixture.store, now=NOW)
    row = fixture.fact_for(date(2026, 9, 10))

    assert row.data_quality == "complete"
    assert row.lifecycle_segment == "core"
    assert row.tier == "starter"


def test_startup_migrations_seed_activity_and_copy_history(tmp_path):
    fixture = MigrationFixture(tmp_path)
    fixture.store.upsert_current_snapshot(_value_snapshot(fixture.user_id))

    report = run_startup_migrations(value_store=fixture.store, now=NOW)
    again = run_startup_migrations(value_store=fixture.store, now=NOW)

    assert report == StartupMigrationReport(activity_seeded=1, history_copied=3)
    assert again == StartupMigrationReport(activity_seeded=1, history_copied=0)
    assert fixture.store.get_activity(fixture.user_id).activated_at is not None
