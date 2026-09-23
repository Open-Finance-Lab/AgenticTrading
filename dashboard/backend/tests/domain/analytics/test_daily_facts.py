"""One set-based pass per UTC day: correctness properties of the daily job."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, time, timedelta, timezone

import pytest
from cryptography.fernet import Fernet

from dashboard.backend.database import BacktestDatabase
from dashboard.backend.domain.agents.repository import AgentStore
from dashboard.backend.domain.analytics.daily_facts import (
    DAILY_FACTS_JOB,
    DailyFactsReport,
    run_daily_facts,
)
from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.service import AnalyticsService
from dashboard.backend.domain.analytics.value_repository import (
    UserDailyFact,
    build_value_analytics_store,
)
from dashboard.backend.domain.brokers import repository as broker_repository
from dashboard.backend.domain.credits.repository import CreditsStore
from dashboard.backend.domain.model_providers.repository import ModelProviderStore
from dashboard.backend.domain.runs.repository import RunStore
from dashboard.backend.users import UserStore


D = date(2026, 9, 11)
NOW = datetime(2026, 9, 12, 0, 5, tzinfo=timezone.utc)  # the tick after midnight
END_OF_D = datetime.combine(D, time(23, 59, 59, 999999), tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _encryption_key(monkeypatch):
    monkeypatch.setenv("BROKER_TOKEN_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setattr(broker_repository, "_fernet_instance", None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("COMMONSTACK_API_KEY", raising=False)


class _NoRetention:
    def __init__(self):
        self.calls = 0

    def run_if_due(self):
        self.calls += 1
        return None


class DailyFixture:
    def __init__(self, path):
        self.path = path / "daily.db"
        self.runs_path = path / "runs.db"
        UserStore(db_path=self.path)
        self.analytics = AnalyticsStore(self.path)
        self.credits = CreditsStore(self.path)
        self.providers = ModelProviderStore(self.path)
        self.agents = AgentStore(self.path)
        self.protocol_runs = RunStore(self.path)
        self.run_history = BacktestDatabase(self.runs_path)
        self.store = build_value_analytics_store(
            self.analytics,
            credits_base=self.credits,
            provider_base=self.providers,
            agent_base=self.agents,
            run_base=self.protocol_runs,
        )
        self.service = AnalyticsService(
            self.analytics, value_store=self.store, maintain_activity=True
        )
        self.retention = _NoRetention()
        self._next_id = 1
        self.admin_id = self.create_user_at(NOW - timedelta(days=90), role="admin")
        self.excluded_id = self.create_user_at(NOW - timedelta(days=90))
        self.analytics.set_subject_exclusion(
            self.excluded_id, excluded=True, actor_user_id=self.admin_id, reason="test"
        )
        self.active_id = self.create_user_at(NOW - timedelta(days=60), user_group="organic")
        self.at_risk_id = self.create_user_at(NOW - timedelta(days=60))
        self.created_today_id = None
        self._seed()

    # -- helpers -------------------------------------------------------------

    def create_user_at(self, when, *, role="user", user_group="unknown"):
        user_id = self._next_id
        self._next_id += 1
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT INTO users (id, email, display_name, password_hash, role, "
                "user_group, created_at) VALUES (?, ?, ?, 'x', ?, ?, ?)",
                (user_id, f"user{user_id}@example.test", f"User {user_id}", role,
                 user_group, when.isoformat()),
            )
        if when > END_OF_D:
            self.created_today_id = user_id
        return user_id

    def append_event(self, user_id, *, event_name, occurred_at, received_at, index=None):
        suffix = index if index is not None else occurred_at.isoformat()
        return self.service.record_server_event(
            event_name=event_name,
            user_id=user_id,
            source_event_id=f"run:{event_name}:{user_id}:{suffix}",
            source_record_type="run",
            source_record_id=f"run-{user_id}-{suffix}",
            occurred_at=occurred_at,
            received_at=received_at,
        )

    def touch_activity(self, user_id, *, at):
        self.store.record_activity(user_id, occurred_at=at, activating=False, now=at)

    def seed_previous_segment(self, segment, *, user_id=None):
        self.store.upsert_daily_facts(
            [
                UserDailyFact(
                    snapshot_date=D - timedelta(days=1),
                    user_id=user_id or self.at_risk_id,
                    lifecycle_segment=segment,
                    lifecycle_reason_code="growing_activated_below_core_threshold",
                    operational_state="healthy",
                    operational_reason_code=None,
                    tier="unpaid",
                    user_group="unknown",
                    active=False,
                    data_quality="complete",
                    calculated_at=NOW - timedelta(days=1),
                )
            ]
        )

    def seed_fact(self, day, user_id, *, active, runs_completed):
        self.store.upsert_daily_facts(
            [
                UserDailyFact(
                    snapshot_date=day,
                    user_id=user_id,
                    lifecycle_segment="growing",
                    lifecycle_reason_code="growing_activated_below_core_threshold",
                    operational_state="healthy",
                    operational_reason_code=None,
                    tier="unpaid",
                    user_group="unknown",
                    active=active,
                    runs_requested=runs_completed,
                    runs_completed=runs_completed,
                    data_quality="complete",
                    calculated_at=NOW - timedelta(days=1),
                )
            ]
        )

    def seed_two_of_three_before_d(self):
        user_id = self.create_user_at(NOW - timedelta(days=60))
        self.touch_activity(user_id, at=NOW - timedelta(days=20))
        self.store.record_activity(
            user_id,
            occurred_at=NOW - timedelta(days=20),
            activating=True,
            now=NOW - timedelta(days=20),
        )
        self.seed_fact(D - timedelta(days=5), user_id, active=True, runs_completed=1)
        self.seed_fact(D - timedelta(days=3), user_id, active=True, runs_completed=1)
        return user_id

    def seed_success_on(self, day, user_id):
        at = datetime.combine(day, time(15, 0), tzinfo=timezone.utc)
        self.append_event(
            user_id, event_name="backtest_completed", occurred_at=at,
            received_at=at + timedelta(seconds=1),
        )

    def break_step(self, name):
        assert name == "ledger", "only the ledger step is breakable in this fixture"

        def _broken(*_args, **_kwargs):
            raise RuntimeError("private ledger detail")

        self.credits.aggregate_ledger_for_day = _broken

    def repair_step(self, name):
        assert name == "ledger"
        del self.credits.aggregate_ledger_for_day

    def fact_for(self, user_id, day=D):
        rows = {row.user_id: row for row in self.store.list_facts_for_date(day)}
        return rows[user_id]

    def transitions_for(self, user_id):
        with self.analytics._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM lifecycle_transitions WHERE user_id = ? ORDER BY snapshot_date",
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def run(self, *, now=NOW):
        return run_daily_facts(
            now=now,
            value_store=self.store,
            run_history_store=self.run_history,
            retention=self.retention,
        )

    # -- seed --------------------------------------------------------------

    def _seed(self):
        active_day = datetime.combine(D, time(10, 0), tzinfo=timezone.utc)
        for index, name in enumerate(
            ("backtest_requested", "backtest_completed", "backtest_requested", "backtest_failed")
        ):
            self.append_event(
                self.active_id,
                event_name=name,
                occurred_at=active_day + timedelta(minutes=index),
                received_at=active_day + timedelta(minutes=index, seconds=1),
                index=index,
            )
        self.run_history.insert_run(
            run_id="run-active",
            session_id="session-active",
            agent_name="Agent",
            mode="backtest",
            start_date="2026-09-01",
            end_date="2026-09-02",
            initial_equity=100000.0,
            est_cost_usd=1.25,
            owner_user_id=self.active_id,
        )
        conn = self.run_history._get_connection()
        try:
            conn.execute(
                "UPDATE agent_runs SET updated_at = ? WHERE run_id = ?",
                (f"{D.isoformat()} 10:30:00", "run-active"),
            )
            conn.commit()
        finally:
            conn.close()
        # at_risk: activated long ago, last meaningful activity ten days before D.
        self.store.record_activity(
            self.at_risk_id,
            occurred_at=NOW - timedelta(days=40),
            activating=True,
            now=NOW - timedelta(days=40),
        )
        self.touch_activity(self.at_risk_id, at=datetime.combine(D - timedelta(days=10), time(9), tzinfo=timezone.utc))


@pytest.fixture
def daily_fixture(tmp_path):
    return DailyFixture(tmp_path)


def test_a_day_is_written_once_and_only_once(daily_fixture):
    """The second call in the same UTC day does nothing."""
    first = daily_fixture.run()
    second = daily_fixture.run()

    assert isinstance(first, DailyFactsReport)
    assert first.claimed is True
    assert first.snapshot_date == D
    assert second.claimed is False
    assert second.users_written == 0
    job = daily_fixture.store.get_projection_job(DAILY_FACTS_JOB)
    assert job.cursor == D.isoformat()
    assert job.status == "complete"


def test_admins_and_excluded_users_get_no_rows(daily_fixture):
    daily_fixture.run()
    written = {row.user_id for row in daily_fixture.store.list_facts_for_date(D)}

    assert daily_fixture.admin_id not in written
    assert daily_fixture.excluded_id not in written
    assert daily_fixture.active_id in written
    assert daily_fixture.at_risk_id in written


def test_run_outcomes_cost_group_and_states_land_on_the_row(daily_fixture):
    daily_fixture.run()
    row = daily_fixture.fact_for(daily_fixture.active_id)

    assert row.runs_requested == 2
    assert row.runs_completed == 1
    assert row.runs_failed == 1
    assert row.runs_cancelled == 0
    assert row.operator_cost_micro == 1_250_000
    assert row.own_spend_micro == 0
    assert row.active is True
    assert row.user_group == "organic"
    assert row.tier == "unpaid"
    assert row.lifecycle_segment == "growing"
    assert row.operational_state == "blocked"  # no balance, no BYOK credential
    assert row.operational_reason_code == "billing_lane_unavailable"
    assert row.data_quality == "complete"


def test_the_at_risk_user_reads_from_the_clock_not_the_events(daily_fixture):
    daily_fixture.run()
    row = daily_fixture.fact_for(daily_fixture.at_risk_id)

    assert row.active is False
    assert row.lifecycle_segment == "at_risk"
    assert row.lifecycle_reason_code == "at_risk_previously_activated"


def test_a_segment_change_appends_exactly_one_transition(daily_fixture):
    daily_fixture.seed_previous_segment("growing")
    daily_fixture.run()
    daily_fixture.run()

    transitions = daily_fixture.transitions_for(daily_fixture.at_risk_id)
    assert len(transitions) == 1
    assert transitions[0]["from_segment"] == "growing"
    assert transitions[0]["to_segment"] == "at_risk"
    assert transitions[0]["inactive_days"] == 10
    assert transitions[0]["data_quality"] == "complete"


def test_no_previous_row_means_no_transition(daily_fixture):
    daily_fixture.run()

    assert daily_fixture.transitions_for(daily_fixture.active_id) == []


def test_a_failed_step_marks_the_day_partial_and_retries_it(daily_fixture, capsys):
    daily_fixture.break_step("ledger")
    first = daily_fixture.run()
    printed = capsys.readouterr().out

    assert first.claimed is True
    assert first.partial is True
    assert "ledger" in first.failed_steps
    assert "WARNING: analytics.daily_facts.ledger_failed category=RuntimeError" in printed
    assert "private ledger detail" not in printed
    assert daily_fixture.fact_for(daily_fixture.active_id).data_quality == "partial"
    job = daily_fixture.store.get_projection_job(DAILY_FACTS_JOB)
    assert job.status == "pending"  # released, not completed
    assert job.cursor is None

    daily_fixture.repair_step("ledger")
    retry = daily_fixture.run(now=NOW + timedelta(minutes=5))
    assert retry.claimed is True
    assert retry.partial is False
    assert daily_fixture.fact_for(daily_fixture.active_id).data_quality == "complete"


def test_a_retry_corrects_the_transition_the_partial_day_wrote(daily_fixture):
    """DO UPDATE, not DO NOTHING.

    The day a transition is rewritten is the day the first attempt was
    wrong. Freezing it leaves `lifecycle_transitions` permanently
    disagreeing with the `user_daily_facts` row the retry did correct.
    """
    daily_fixture.seed_previous_segment("growing")
    daily_fixture.break_step("ledger")
    daily_fixture.run()
    first = daily_fixture.transitions_for(daily_fixture.at_risk_id)[0]

    daily_fixture.repair_step("ledger")
    daily_fixture.run(now=NOW + timedelta(minutes=5))
    corrected = daily_fixture.transitions_for(daily_fixture.at_risk_id)

    assert len(corrected) == 1
    assert first["data_quality"] == "partial"
    assert corrected[0]["data_quality"] == "complete"


def test_a_user_active_after_midnight_does_not_abort_the_day(daily_fixture):
    """The regression that would have fired on the first night in prod.

    `user_activity` is overwritten in place, so a user who acts at 00:02
    has a `last_meaningful_activity_at` newer than the end of D. Fed to
    `calculate_lifecycle(as_of=<end of D>)` unclamped that raises, and the
    single try/except around the step turns one such user into zero fact
    rows for the entire population.
    """
    daily_fixture.touch_activity(daily_fixture.active_id, at=NOW)
    daily_fixture.create_user_at(NOW)

    report = daily_fixture.run()

    assert report.partial is False
    assert report.failed_steps == ()
    assert report.users_written >= 2
    written = {row.user_id for row in daily_fixture.store.list_facts_for_date(D)}
    assert daily_fixture.active_id in written
    # Created after D ended: no day D to describe.
    assert daily_fixture.created_today_id not in written


def test_the_stored_segment_counts_its_own_day(daily_fixture):
    """The trailing window is 30 dates ending at D, D included.

    Sitting at two active days and two successes before D, plus an active
    day with a success on D, is the `core` threshold exactly. Summing only
    D-29..D-1 writes `growing` and the read path answers `core` the next
    morning.
    """
    user_id = daily_fixture.seed_two_of_three_before_d()
    daily_fixture.seed_success_on(D, user_id)

    daily_fixture.run()

    assert daily_fixture.fact_for(user_id).lifecycle_segment == "core"


def test_an_event_arriving_after_the_day_was_written_is_picked_up(daily_fixture):
    """Late arrivals are recomputed, not lost.

    A run finishing at 23:59 can be appended minutes later, and the
    frontend route accepts `occurred_at` up to 24 hours old. Both land
    inside D after D's aggregate was taken, and the cursor has already
    moved past D. The next day's job finds them by `received_at`.
    """
    daily_fixture.run()
    before = daily_fixture.fact_for(daily_fixture.active_id).runs_completed

    daily_fixture.append_event(
        daily_fixture.active_id,
        event_name="backtest_completed",
        occurred_at=datetime(2026, 9, 11, 23, 59, tzinfo=timezone.utc),
        received_at=NOW + timedelta(minutes=5),
        index="late",
    )
    report = daily_fixture.run(now=NOW + timedelta(days=1))

    assert report.snapshot_date == D + timedelta(days=1)
    assert D in report.recomputed_dates
    assert daily_fixture.fact_for(daily_fixture.active_id).runs_completed == before + 1


def test_a_settled_day_is_not_recomputed_forever(daily_fixture):
    """The sweep is driven by evidence, not by a timer."""
    daily_fixture.run()
    report = daily_fixture.run(now=NOW + timedelta(days=1))

    assert report.recomputed_dates == ()


def test_the_retention_coordinator_runs_on_every_tick(daily_fixture):
    daily_fixture.run()
    daily_fixture.run(now=NOW + timedelta(minutes=5))  # idle tick

    assert daily_fixture.retention.calls == 2
