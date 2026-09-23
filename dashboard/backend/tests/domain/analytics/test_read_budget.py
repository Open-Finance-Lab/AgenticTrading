"""The architectural claim of the design, stated as tests (design SS6.12).

Every store the daily job touches is wrapped in a counting spy and the job is
driven against 200 and then 400 synthetic users. If the query count moves,
someone has put a user id into a query inside the job; if the wall-clock ratio
moves, a constant-count step hides a scan that grows with the population.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import date, datetime, time as clock_time, timedelta, timezone

import pytest
from cryptography.fernet import Fernet

from dashboard.backend.database import BacktestDatabase
from dashboard.backend.domain.agents.repository import AgentStore
from dashboard.backend.domain.analytics.daily_facts import run_daily_facts
from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.service import AnalyticsService
from dashboard.backend.domain.analytics.value_repository import (
    build_value_analytics_store,
)
from dashboard.backend.domain.brokers import repository as broker_repository
from dashboard.backend.domain.credits.repository import CreditsStore
from dashboard.backend.domain.model_providers.repository import ModelProviderStore
from dashboard.backend.domain.runs.repository import RunStore
from dashboard.backend.tests.domain.analytics._store_spies import CountingSpy, SpyBundle
from dashboard.backend.tests.test_credits_ledger_aggregates import _insert, _ledger_row
from dashboard.backend.users import UserStore


D = date(2026, 9, 11)
NOW = datetime(2026, 9, 12, 0, 5, tzinfo=timezone.utc)

# The constant the daily job commits to, per claimed tick, in public store-method
# calls (the unit CountingSpy measures). Task 9's table in the PR A plan is the
# source of this number; if it moves, either the job or the table drifted, and
# the table is authoritative until the design changes.
#   value store   12  claim, aggregate_events, record_activity_batch x2,
#                     list_operational_signals, list_activity, sum_recent_facts,
#                     upsert_daily_facts, list_facts_for_date,
#                     append_lifecycle_transitions, list_days_needing_recompute,
#                     complete_projection_day
#   analytics      2  list_excluded_user_ids (inside rollup_day), list_daily_subjects
#   credits        3  aggregate_ledger_for_day, get_balance_projections,
#                     list_account_billing_states
#   providers      3  list_default_credential_facts, list_all_providers,
#                     list_platform_credential_statuses
#   agents         1  list_agent_owners
#   protocol runs  1  list_terminal_runs_since
#   run history    1  aggregate_operator_cost_for_day
DAILY_JOB_CLAIMED_TICK_CALLS = 23
WALL_CLOCK_BUDGET_SECONDS_AT_200 = 20.0
WALL_CLOCK_RATIO_LIMIT = 2.5
# A ratio over tiny absolute times is noise, not a scan; floor the denominator.
WALL_CLOCK_FLOOR_SECONDS = 0.25


@pytest.fixture(autouse=True)
def _encryption_key(monkeypatch):
    monkeypatch.setenv("BROKER_TOKEN_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setattr(broker_repository, "_fernet_instance", None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("COMMONSTACK_API_KEY", raising=False)


class _NoRetention:
    def run_if_due(self):
        return None


class _DailyFixture:
    def __init__(self, path, *, users):
        path.mkdir(parents=True, exist_ok=True)
        self.path = path / "budget.db"
        UserStore(db_path=self.path)
        with sqlite3.connect(self.path) as conn:
            conn.executemany(
                "INSERT INTO users (id, email, display_name, password_hash, role, "
                "user_group, created_at) VALUES (?, ?, ?, 'x', 'user', 'unknown', ?)",
                [
                    (index, f"user{index}@example.test", f"User {index}",
                     (NOW - timedelta(days=60)).isoformat())
                    for index in range(1, users + 1)
                ],
            )
        self.user_ids = list(range(1, users + 1))
        analytics = AnalyticsStore(self.path)
        credits = CreditsStore(self.path)
        providers = ModelProviderStore(self.path)
        agents = AgentStore(self.path)
        protocol_runs = RunStore(self.path)
        run_history = BacktestDatabase(path / "runs.db")
        self.spies = SpyBundle(
            analytics=CountingSpy(analytics, "analytics"),
            credits=CountingSpy(credits, "credits"),
            providers=CountingSpy(providers, "providers"),
            agents=CountingSpy(agents, "agents"),
            runs=CountingSpy(protocol_runs, "runs"),
            run_history=CountingSpy(run_history, "run_history"),
        )
        real_value_store = build_value_analytics_store(
            self.spies.analytics,
            credits_base=self.spies.credits,
            provider_base=self.spies.providers,
            agent_base=self.spies.agents,
            run_base=self.spies.runs,
        )
        self.value_store = CountingSpy(real_value_store, "value_store")
        self.spies.spies["value_store"] = self.value_store
        # Seed through the real (unspied) stores so setup is not counted.
        service = AnalyticsService(analytics, value_store=real_value_store, maintain_activity=True)
        day_start = datetime.combine(D, clock_time(9, 0), tzinfo=timezone.utc)
        for user_id in self.user_ids:
            service.record_server_event(
                event_name="backtest_requested",
                user_id=user_id,
                source_event_id=f"run:backtest_requested:{user_id}",
                source_record_type="run",
                source_record_id=f"run-{user_id}",
                occurred_at=day_start + timedelta(seconds=user_id),
                received_at=day_start + timedelta(seconds=user_id + 1),
            )
            if user_id % 5 == 0:
                service.record_server_event(
                    event_name="backtest_completed",
                    user_id=user_id,
                    source_event_id=f"run:backtest_completed:{user_id}",
                    source_record_type="run",
                    source_record_id=f"run-{user_id}",
                    occurred_at=day_start + timedelta(minutes=1, seconds=user_id),
                    received_at=day_start + timedelta(minutes=1, seconds=user_id + 1),
                )
            if user_id % 2 == 0:
                agents.create_agent(name=f"Agent {user_id}", owner_user_id=user_id)
        _insert(
            self.path,
            "credit_ledger_entries",
            [
                _ledger_row(user_id, "admin_grant_assign", 1_000_000, NOW - timedelta(days=2), key=f"g{user_id}")
                for user_id in self.user_ids
                if user_id % 3 == 0
            ],
        )
        self.spies.reset()

    def run(self, *, now=NOW):
        return run_daily_facts(
            now=now,
            value_store=self.value_store,
            run_history_store=self.spies.run_history,
            retention=_NoRetention(),
        )


def _timed(action):
    started = time.perf_counter()
    result = action()
    return result, time.perf_counter() - started


def test_the_daily_job_costs_the_same_at_any_user_count(tmp_path):
    """Doubling the population changes nothing about the query count.

    This is the architectural claim of the whole design, stated as a test.
    If it ever fails, someone has put a user id into a query inside the job.
    """
    small = _DailyFixture(tmp_path / "small", users=200)
    large = _DailyFixture(tmp_path / "large", users=400)

    small_report = small.run()
    large_report = large.run()

    assert small_report.claimed and large_report.claimed
    assert small_report.partial is False and large_report.partial is False
    assert small_report.users_written == 200
    assert large_report.users_written == 400
    assert small.spies.total_calls == large.spies.total_calls, (
        small.spies.calls_by_store(),
        large.spies.calls_by_store(),
    )
    assert small.spies.total_calls == DAILY_JOB_CLAIMED_TICK_CALLS, small.spies.calls_by_store()
    assert small.spies.calls_with_scalar_user_id == []
    assert large.spies.calls_with_scalar_user_id == []


def test_the_daily_job_stays_inside_the_wall_clock_budget(tmp_path):
    """A constant query count can still hide a scan (design SS12 item 5)."""
    small = _DailyFixture(tmp_path / "small", users=200)
    large = _DailyFixture(tmp_path / "large", users=400)

    _report, small_elapsed = _timed(small.run)
    _report, large_elapsed = _timed(large.run)

    assert small_elapsed < WALL_CLOCK_BUDGET_SECONDS_AT_200, small_elapsed
    assert large_elapsed <= WALL_CLOCK_RATIO_LIMIT * max(small_elapsed, WALL_CLOCK_FLOOR_SECONDS), (
        small_elapsed,
        large_elapsed,
    )


def test_an_idle_tick_costs_one_query(tmp_path):
    """Most ticks in a day do nothing. They must cost (almost) nothing."""
    fixture = _DailyFixture(tmp_path, users=50)
    fixture.run()
    fixture.spies.reset()

    for tick in range(60):
        report = fixture.run(now=NOW + timedelta(minutes=5 * tick))
        assert report.claimed is False

    assert fixture.spies.total_calls == 60
    assert fixture.spies.calls_by_store()["value_store"] == 60
    assert all(
        count == 0
        for name, count in fixture.spies.calls_by_store().items()
        if name != "value_store"
    )
