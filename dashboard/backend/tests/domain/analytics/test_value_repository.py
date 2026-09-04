"""Contracts for user-value projection persistence and safe source joins."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone

from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.states import (
    AnalyticsStateStore,
    UserAnalyticsSnapshot,
)
from dashboard.backend.domain.analytics.value_repository import (
    ProjectionJob,
    UserLifecycleDailySnapshot,
    UserValueSnapshot,
    ValueAnalyticsStore,
)
from dashboard.backend.users import UserStore


UTC = timezone.utc
NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
START = NOW - timedelta(days=7)
END = NOW + timedelta(seconds=1)


class SyntheticCreditsStore:
    def __init__(self, db_path):
        self.db_path = db_path
        self.balance_calls: list[tuple[int, ...]] = []
        self.select_statements: list[str] = []
        self.balances = {}
        self.billing = {
            1: {
                "account_status": "active",
                "restriction_reason": None,
                "outstanding_credits_micro": 0,
            }
        }
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE credit_ledger_entries (
                    user_id INTEGER NOT NULL,
                    entry_type TEXT NOT NULL,
                    amount_micro INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE credit_llm_usage_entries (
                    user_id INTEGER NOT NULL,
                    amount_micro INTEGER NOT NULL,
                    operation_key TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.set_trace_callback(
            lambda statement: (
                self.select_statements.append(statement)
                if statement.lstrip().upper().startswith("SELECT")
                else None
            )
        )
        return conn

    def get_balance_projections(self, user_ids):
        self.balance_calls.append(tuple(user_ids))
        return {
            user_id: self.balances.get(
                user_id,
                {
                    "grant_available_micro": 900_000,
                    "purchased_available_micro": 100_000,
                    "total_available_micro": 1_000_000,
                },
            )
            for user_id in user_ids
        }

    def get_account_billing_state(self, user_id):
        return self.billing.get(
            user_id,
            {
                "account_status": "active",
                "restriction_reason": None,
                "outstanding_credits_micro": 0,
            },
        )


class SyntheticProviderStore:
    def __init__(self, credentials, providers):
        self.credentials = credentials
        self.providers = providers
        self.secret_reads = 0

    def list_user_credentials(self, user_id):
        return list(self.credentials.get(user_id, ()))

    def list_all_providers(self):
        return list(self.providers)

    def get_user_credential_secret(self, *_args, **_kwargs):
        self.secret_reads += 1
        raise AssertionError("Analytics must not decrypt credentials")


class SyntheticAgentStore:
    def list_agents(self, *, owner_user_id):
        return [{"agent_id": f"agent-{owner_user_id}"}]


class SyntheticRunStore:
    def __init__(self, runs):
        self.runs = runs

    def list_runs(self, agent_id):
        return list(self.runs.get(agent_id, ()))


def _value_snapshot(user_id: int) -> UserValueSnapshot:
    return UserValueSnapshot(
        user_id=user_id,
        lifecycle_segment="core",
        lifecycle_reason_code="core_repeated_value",
        lifecycle_reason="The user repeatedly completed valuable work.",
        lifecycle_evidence=("3 active days in the trailing 30 days",),
        operational_state="healthy",
        operational_reason_code="no_supported_issue",
        operational_reason="No supported current operational issue was detected.",
        operational_evidence=("All supported operational checks passed.",),
        activated_at=NOW - timedelta(days=20),
        last_meaningful_activity_at=NOW - timedelta(days=2),
        inactive_days=2,
        active_days_30d=3,
        successful_backtests_30d=4,
        calculated_at=NOW,
    )


def _stores(tmp_path):
    analytics_path = tmp_path / "analytics.db"
    user = UserStore(db_path=analytics_path).create_user(
        "value-user@example.test",
        "Value User",
        "SecurePass1!",
    )
    analytics = AnalyticsStore(db_path=analytics_path)
    credits = SyntheticCreditsStore(tmp_path / "credits.db")
    return int(user["id"]), analytics, credits


def _value_store(analytics, credits):
    return ValueAnalyticsStore(
        analytics,
        credits,
        provider_base=SyntheticProviderStore({}, []),
        agent_base=SyntheticAgentStore(),
        run_base=SyntheticRunStore({}),
    )


def test_current_projection_round_trips_without_overwriting_legacy_state(tmp_path):
    user_id, analytics, credits = _stores(tmp_path)
    legacy = UserAnalyticsSnapshot(
        user_id=user_id,
        status="needs_attention",
        reason_code="invalid_default_credential",
        human_readable_reason="The default model credential is invalid.",
        evidence_event_ids=["synthetic-event"],
        calculated_at=NOW - timedelta(minutes=1),
    )
    AnalyticsStateStore(analytics).upsert_snapshot(legacy)
    store = _value_store(analytics, credits)

    expected = _value_snapshot(user_id)
    store.upsert_current_snapshot(expected)

    assert store.get_current_snapshot(user_id) == expected
    persisted_legacy = AnalyticsStateStore(analytics).get_snapshot(user_id)
    assert persisted_legacy is not None
    assert persisted_legacy.status == legacy.status
    assert persisted_legacy.reason_code == legacy.reason_code
    assert persisted_legacy.human_readable_reason == legacy.human_readable_reason
    assert persisted_legacy.evidence_event_ids == legacy.evidence_event_ids


def test_daily_snapshot_and_projection_job_upserts_are_idempotent(tmp_path):
    user_id, analytics, credits = _stores(tmp_path)
    store = _value_store(analytics, credits)
    first = UserLifecycleDailySnapshot(
        snapshot_date=NOW.date(),
        user_id=user_id,
        lifecycle_segment="growing",
        lifecycle_reason_code="growing_activated_below_core_threshold",
        data_quality="partial",
        calculated_at=NOW - timedelta(minutes=1),
    )
    replacement = first.model_copy(
        update={
            "lifecycle_segment": "core",
            "lifecycle_reason_code": "core_repeated_value",
            "data_quality": "complete",
            "calculated_at": NOW,
        }
    )

    store.upsert_daily_snapshot(first)
    store.upsert_daily_snapshot(replacement)
    rows = store.list_daily_snapshots(
        start=NOW.date(),
        end=NOW.date() + timedelta(days=1),
        user_ids=[user_id],
    )

    assert rows == [replacement]
    assert (
        store.list_daily_snapshots(
            start=NOW.date(),
            end=NOW.date() + timedelta(days=1),
            user_ids=[],
        )
        == []
    )

    job = ProjectionJob(
        job_name="lifecycle-backfill",
        window_start=date(2026, 7, 6),
        window_end=date(2026, 8, 30),
        cursor="user:40",
        status="running",
        updated_at=NOW,
    )
    store.save_projection_job(job)
    assert store.get_projection_job(job.job_name) == job


def test_commercial_value_keeps_revenue_grants_usage_and_balance_separate(tmp_path):
    user_id, analytics, credits = _stores(tmp_path)
    old = START - timedelta(days=1)
    with credits._get_connection() as conn:
        conn.executemany(
            "INSERT INTO credit_ledger_entries VALUES (?, ?, ?, ?)",
            [
                (user_id, "purchase", 10_000_000, old.isoformat()),
                (user_id, "purchase", 3_000_000, START.isoformat()),
                (
                    user_id,
                    "refund",
                    -6_000_000,
                    (START + timedelta(hours=1)).isoformat(),
                ),
                (
                    user_id,
                    "admin_grant_assign",
                    1_500_000,
                    (START + timedelta(hours=2)).isoformat(),
                ),
                (
                    user_id,
                    "admin_grant_reclaim",
                    -500_000,
                    (START + timedelta(hours=3)).isoformat(),
                ),
            ],
        )
        conn.executemany(
            "INSERT INTO credit_llm_usage_entries VALUES (?, ?, ?, ?)",
            [
                (
                    user_id,
                    -400_000,
                    "settle:normal",
                    (START + timedelta(hours=4)).isoformat(),
                ),
                (
                    user_id,
                    -100_000,
                    "settle:recovery:1",
                    (START + timedelta(hours=5)).isoformat(),
                ),
            ],
        )
    store = _value_store(analytics, credits)
    credits.select_statements.clear()

    fact = store.list_commercial_values(
        [user_id, user_id],
        start=START,
        end=END,
    )[user_id]

    assert fact.lifetime_net_purchased_micro == 7_000_000
    assert fact.commercial_tier == "invested"
    assert fact.purchased_micro == 3_000_000
    assert fact.refunded_micro == 6_000_000
    assert fact.admin_grant_activity_micro == 2_000_000
    assert fact.consumed_micro == 500_000
    assert fact.grant_available_micro == 900_000
    assert fact.purchased_available_micro == 100_000
    assert fact.total_available_micro == 1_000_000
    assert credits.balance_calls == [(user_id,)]
    assert len(credits.select_statements) == 3


def test_credit_activity_includes_purchase_and_consumption_only(tmp_path):
    user_id, analytics, credits = _stores(tmp_path)
    purchase_at = START + timedelta(hours=1)
    refund_at = START + timedelta(hours=2)
    grant_at = START + timedelta(hours=3)
    consumed_at = START + timedelta(hours=4)
    with credits._get_connection() as conn:
        conn.executemany(
            "INSERT INTO credit_ledger_entries VALUES (?, ?, ?, ?)",
            [
                (user_id, "purchase", 1_000_000, purchase_at.isoformat()),
                (user_id, "refund", -100_000, refund_at.isoformat()),
                (user_id, "admin_grant_assign", 500_000, grant_at.isoformat()),
            ],
        )
        conn.execute(
            "INSERT INTO credit_llm_usage_entries VALUES (?, ?, ?, ?)",
            (user_id, -50_000, "settle:1", consumed_at.isoformat()),
        )

    activity = _value_store(analytics, credits).list_credit_activity(
        [user_id],
        start=START,
        end=END,
    )

    assert activity[user_id] == (purchase_at, consumed_at)


def test_operational_facts_use_public_sources_without_reading_credentials(tmp_path):
    user_id, analytics, credits = _stores(tmp_path)
    providers = SyntheticProviderStore(
        {
            user_id: [
                {
                    "credential_id": "cred-public",
                    "provider_id": "openrouter",
                    "status": "invalid",
                    "is_default": True,
                },
                {
                    "credential_id": "cred-fallback",
                    "provider_id": "gemini",
                    "status": "verified",
                    "is_default": True,
                },
            ]
        },
        [
            {
                "provider_id": "openrouter",
                "status": "disabled",
                "byok_enabled": True,
                "platform_enabled": True,
            },
            {
                "provider_id": "gemini",
                "status": "enabled",
                "byok_enabled": True,
                "platform_enabled": False,
            },
        ],
    )
    runs = SyntheticRunStore(
        {
            f"agent-{user_id}": [
                {
                    "status": "failed",
                    "updated_at": (NOW - timedelta(hours=1)).isoformat(),
                },
                {
                    "status": "failed",
                    "updated_at": (NOW - timedelta(hours=2)).isoformat(),
                },
                {
                    "status": "failed",
                    "updated_at": (NOW - timedelta(hours=3)).isoformat(),
                },
                {
                    "status": "running",
                    "created_at": (NOW - timedelta(hours=2)).isoformat(),
                },
            ]
        }
    )
    store = ValueAnalyticsStore(
        analytics,
        credits,
        provider_base=providers,
        agent_base=SyntheticAgentStore(),
        run_base=runs,
    )

    facts = store.get_operational_facts(user_id, now=NOW)

    assert facts.account_restricted is False
    assert facts.usable_billing_lane is True
    assert facts.selected_provider_enabled is False
    assert facts.default_credential_status == "invalid"
    assert facts.failed_terminal_runs_24h == 3
    assert facts.run_beyond_safe_deadline is True
    assert providers.secret_reads == 0


def test_non_default_credential_does_not_create_a_usable_billing_lane(tmp_path):
    user_id, analytics, credits = _stores(tmp_path)
    credits.balances[user_id] = {
        "grant_available_micro": 0,
        "purchased_available_micro": 0,
        "total_available_micro": 0,
    }
    providers = SyntheticProviderStore(
        {
            user_id: [
                {
                    "credential_id": "cred-not-default",
                    "provider_id": "gemini",
                    "status": "verified",
                    "is_default": False,
                }
            ]
        },
        [
            {
                "provider_id": "gemini",
                "status": "enabled",
                "byok_enabled": True,
                "platform_enabled": False,
            }
        ],
    )
    store = ValueAnalyticsStore(
        analytics,
        credits,
        provider_base=providers,
        agent_base=SyntheticAgentStore(),
        run_base=SyntheticRunStore({}),
    )

    facts = store.get_operational_facts(user_id, now=NOW)

    assert facts.usable_billing_lane is False
    assert facts.default_credential_status == "missing"
