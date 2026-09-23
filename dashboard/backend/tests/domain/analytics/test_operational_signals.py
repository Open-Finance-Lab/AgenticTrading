"""The batched operational read agrees with the per-user one.

Two implementations of one rule set drift silently. This test is the only
thing that keeps the daily job's answer equal to the profile's answer.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from cryptography.fernet import Fernet

from dashboard.backend.domain.agents.repository import AgentStore
from dashboard.backend.domain.analytics.lifecycle import (
    calculate_operational_state,
    consecutive_failed_terminal_runs,
)
from dashboard.backend.domain.analytics.repository import AnalyticsStore
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


NOW = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _encryption_key(monkeypatch):
    monkeypatch.setenv("BROKER_TOKEN_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setattr(broker_repository, "_fernet_instance", None)
    # The platform lane must be decided by the stored credential alone.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("COMMONSTACK_API_KEY", raising=False)


class OperationalFixture:
    def __init__(self, path, *, users):
        self.path = path
        UserStore(db_path=path)  # creates the users table
        with sqlite3.connect(path) as conn:
            conn.executemany(
                "INSERT INTO users (id, email, display_name, password_hash, role, "
                "user_group, created_at) VALUES (?, ?, ?, 'x', 'user', 'unknown', ?)",
                [
                    (index, f"user{index}@example.test", f"User {index}",
                     (NOW - timedelta(days=30)).isoformat())
                    for index in range(1, users + 1)
                ],
            )
        self.user_ids = list(range(1, users + 1))
        self.credits = CreditsStore(path)
        self.providers = ModelProviderStore(path)
        self.agents = AgentStore(path)
        self.runs = RunStore(path)
        self.analytics = AnalyticsStore(path)
        self.spies = SpyBundle(
            credits=CountingSpy(self.credits, "credits"),
            providers=CountingSpy(self.providers, "providers"),
            agents=CountingSpy(self.agents, "agents"),
            runs=CountingSpy(self.runs, "runs"),
        )
        self.store = build_value_analytics_store(
            self.analytics,
            credits_base=self.spies.credits,
            provider_base=self.spies.providers,
            agent_base=self.spies.agents,
            run_base=self.spies.runs,
        )
        self.interleaved_failures_id = 4
        self._seed()

    # -- seeding -----------------------------------------------------------

    def _provider(self, provider_id, *, status="enabled", platform_enabled=None):
        conn = self.providers._get_connection()
        try:
            conn.execute(
                "UPDATE provider_registry SET status = ? WHERE provider_id = ?",
                (status, provider_id),
            )
            if platform_enabled is not None:
                conn.execute(
                    "UPDATE provider_registry SET platform_enabled = ? WHERE provider_id = ?",
                    (int(platform_enabled), provider_id),
                )
            conn.commit()
        finally:
            conn.close()

    def _default_credential(self, user_id, provider_id, *, status="verified"):
        created = self.providers.create_user_credential(
            user_id=user_id,
            provider_id=provider_id,
            label=f"{provider_id}-{user_id}",
            secret="sk-synthetic-secret-1234",
            status="verified",
            set_default=True,
        )
        if status != "verified":
            # The store clears is_default when a credential stops being
            # verified, so an *invalid default* -- the exact state
            # ``default_credential_status == "invalid"`` describes -- can only
            # be reached by a row that went invalid underneath its flag. Write
            # that row directly; both readers see the same table.
            conn = self.providers._get_connection()
            try:
                conn.execute(
                    "UPDATE user_model_credentials SET status = ? WHERE credential_id = ?",
                    (status, created["credential_id"]),
                )
                conn.commit()
            finally:
                conn.close()

    def _run(self, agent, status, *, hours_ago):
        run = self.runs.create_run(
            agent_id=agent["agent_id"],
            agent_version_id=None,
            session_id=agent["session_id"],
            environment_id=None,
            environment_type="backtest",
            config={},
            status=status,
        )
        stamp = (NOW - timedelta(hours=hours_ago)).isoformat()
        conn = self.runs._get_connection()
        try:
            conn.execute(
                "UPDATE protocol_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
                (stamp, stamp, run["run_id"]),
            )
            conn.commit()
        finally:
            conn.close()

    def _grant(self, user_id, amount_micro, *, key):
        _insert(
            self.path,
            "credit_ledger_entries",
            [_ledger_row(user_id, "admin_grant_assign", amount_micro, NOW - timedelta(days=1), key=key)],
        )

    def _seed(self):
        # One platform lane is open for everyone with a balance.
        self._provider("openrouter", status="enabled", platform_enabled=True)
        self.providers.upsert_platform_credential(
            provider_id="openrouter", secret="sk-platform-secret-9999", status="verified"
        )
        for user_id in self.user_ids:
            self._grant(user_id, 1_000_000, key=f"grant-{user_id}")
        # 1: restricted account (blocked, highest precedence).
        self.credits.restrict_account(1, reason="llm_overage")
        # 2: invalid default credential (needs_attention).
        self._default_credential(2, "openai", status="invalid")
        # 3: three consecutive failed terminal runs inside 24h (needs_attention).
        agent_three = self.agents.create_agent(name="Three", owner_user_id=3)
        for hours in (1, 2, 3):
            self._run(agent_three, "failed", hours_ago=hours)
        # 4: failed, failed, succeeded, failed, failed -> consecutive count 2.
        agent_four = self.agents.create_agent(name="Four", owner_user_id=4)
        for status, hours in (("failed", 1), ("failed", 2), ("completed", 3), ("failed", 4), ("failed", 5)):
            self._run(agent_four, status, hours_ago=hours)
        # 5: failures spread across two agents; pooled newest-first is what counts.
        agent_five_a = self.agents.create_agent(name="Five A", owner_user_id=5)
        agent_five_b = self.agents.create_agent(name="Five B", owner_user_id=5)
        self._run(agent_five_a, "failed", hours_ago=1)
        self._run(agent_five_b, "failed", hours_ago=2)
        self._run(agent_five_a, "completed", hours_ago=3)
        self._run(agent_five_b, "failed", hours_ago=30)  # outside 24h
        # 6: default credential on a provider that is then disabled (blocked).
        # Order matters: create_user_credential refuses a disabled provider,
        # so the credential is created first and the provider disabled after.
        # "anthropic" is one of the four SEEDED_PROVIDERS (repository_common.py)
        # and is BYOK-enabled; nobody else in this fixture uses it.
        self._default_credential(6, "anthropic")
        self._provider("anthropic", status="disabled")
        # 7: no usable lane -- remove the grant so the balance is zero (blocked).
        with sqlite3.connect(self.path) as conn:
            conn.execute("DELETE FROM credit_ledger_entries WHERE user_id = 7")
        # 8+: clean accounts (healthy).

    def reset(self):
        self.spies.reset()

    @property
    def total_calls(self):
        return self.spies.total_calls

    @property
    def calls_with_scalar_user_id(self):
        return self.spies.calls_with_scalar_user_id


def _operational_fixture(path, *, users):
    return OperationalFixture(path / "operational.db", users=users)


@pytest.fixture
def operational_fixture(tmp_path):
    return _operational_fixture(tmp_path, users=8)


def test_batched_signals_match_the_per_user_facts(operational_fixture):
    """Same users, same instant, same answer -- and every seeded state is hit."""
    store, user_ids = operational_fixture.store, operational_fixture.user_ids

    batched = store.list_operational_signals(user_ids, now=NOW)

    states = {}
    for user_id in user_ids:
        facts = store.get_operational_facts(user_id, now=NOW)
        signals = batched[user_id]
        assert signals.account_restricted == facts.account_restricted, user_id
        assert signals.usable_billing_lane == facts.usable_billing_lane, user_id
        assert signals.selected_provider_enabled == facts.selected_provider_enabled, user_id
        assert signals.default_credential_status == facts.default_credential_status, user_id
        assert signals.failed_terminal_runs_24h == facts.failed_terminal_runs_24h, user_id
        states[user_id] = calculate_operational_state(signals, NOW)
        assert states[user_id].state == calculate_operational_state(
            type(signals)(
                user_id=user_id,
                account_restricted=facts.account_restricted,
                usable_billing_lane=facts.usable_billing_lane,
                selected_provider_enabled=facts.selected_provider_enabled,
                default_credential_status=facts.default_credential_status,
                failed_terminal_runs_24h=facts.failed_terminal_runs_24h,
                run_beyond_safe_deadline=False,
            ),
            NOW,
        ).state
    assert states[1].reason_code == "account_restricted"
    assert states[2].reason_code == "invalid_default_credential"
    assert states[3].reason_code == "three_consecutive_failed_runs"
    assert states[4].state == "healthy"
    assert states[6].reason_code == "provider_disabled"
    assert states[7].reason_code == "billing_lane_unavailable"
    assert states[8].state == "healthy"


def test_the_population_path_answers_for_the_given_ids_without_an_in_list(operational_fixture):
    store = operational_fixture.store
    ids = operational_fixture.user_ids

    wide = store.list_operational_signals(ids, now=NOW, population_wide=True)
    listed = store.list_operational_signals(ids, now=NOW)

    assert set(wide) == set(ids)
    assert wide == listed
    # User 7 has no row in any source; both modes still answer for them.
    assert wide[7].usable_billing_lane is False
    assert store.list_operational_signals([], now=NOW, population_wide=True) == {}


def test_batched_signals_do_not_query_per_user(tmp_path):
    """Query count is fixed, not proportional to the population.

    Asserting a literal ceiling pins an implementation detail this task
    cannot honour: the run count alone needs two statements, because
    protocol_runs and external_agents are in different databases. The
    property that matters is that neither number moves when the user count
    does.
    """
    small = _operational_fixture(tmp_path / "small", users=10)
    large = _operational_fixture(tmp_path / "large", users=40)

    small.reset()
    small.store.list_operational_signals(small.user_ids, now=NOW, population_wide=True)
    small_calls = small.total_calls

    large.reset()
    large.store.list_operational_signals(large.user_ids, now=NOW, population_wide=True)

    assert large.calls_with_scalar_user_id == []
    assert large.total_calls == small_calls
    # A loose absolute bound as well, so "zero queries because it silently
    # returned defaults" cannot pass the equality above.
    assert 4 <= small_calls <= 8


def test_the_batched_count_is_consecutive_not_total(operational_fixture):
    """failed, failed, succeeded, failed, failed inside 24h scores 2.

    This is the one place the two implementations could disagree while
    every other assertion stayed green, because both numbers are plausible
    and only one of them matches the reason code the UI renders.
    """
    user_id = operational_fixture.interleaved_failures_id

    signals = operational_fixture.store.list_operational_signals(
        [user_id], now=NOW
    )[user_id]

    assert signals.failed_terminal_runs_24h == 2
    assert calculate_operational_state(signals, NOW).state == "healthy"


def test_runs_are_pooled_across_an_owners_agents_before_counting(operational_fixture):
    signals = operational_fixture.store.list_operational_signals([5], now=NOW)[5]

    # Newest-first across both agents: failed(A,-1h), failed(B,-2h), completed(A,-3h).
    assert signals.failed_terminal_runs_24h == 2


def test_a_user_with_no_rows_agrees_with_the_per_user_reader(operational_fixture):
    """Absence is computed, not defaulted.

    A user with no ledger row, no credential and no agent has a zero balance
    and no BYOK lane, so both readers say `billing_lane_unavailable`. The
    superseded plan asserted `healthy` here against synthetic stores whose
    defaults were permissive; against real stores the honest answer is
    "blocked", and it must be the *same* honest answer on both paths.
    """
    store = operational_fixture.store
    absent = max(operational_fixture.user_ids) + 1
    # The user must exist -- get_account_billing_state lazily creates the
    # credit_accounts row and its FK points at users. "Absent" here means
    # exists as an account with no ledger row, no credential, no agent.
    import sqlite3 as _sqlite3
    with _sqlite3.connect(operational_fixture.path) as conn:
        conn.execute(
            "INSERT INTO users (id, email, display_name, password_hash, role, "
            "user_group, created_at) VALUES (?, 'absent@example.test', 'Absent', "
            "'x', 'user', 'unknown', ?)",
            (absent, (NOW - timedelta(days=30)).isoformat()),
        )

    signals = store.list_operational_signals([absent], now=NOW)[absent]
    facts = store.get_operational_facts(absent, now=NOW)

    assert signals.default_credential_status == "missing"
    assert signals.failed_terminal_runs_24h == 0
    assert signals.usable_billing_lane is facts.usable_billing_lane is False
    assert (
        calculate_operational_state(signals, NOW).reason_code
        == "billing_lane_unavailable"
    )


@pytest.mark.parametrize(
    "statuses,expected",
    [
        ((), 0),
        (("failed", "failed", "failed"), 3),
        (("failed", "timed_out", "completed", "failed"), 2),
        (("completed", "failed", "failed"), 0),
        (("cancelled", "failed"), 0),
    ],
)
def test_consecutive_failed_terminal_runs_counts_leading_failures(statuses, expected):
    assert consecutive_failed_terminal_runs(statuses) == expected