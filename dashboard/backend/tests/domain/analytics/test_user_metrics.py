"""One metric struct, two audiences (design §6.13)."""

from __future__ import annotations

import json
import sqlite3
from datetime import timedelta

import pytest

from dashboard.backend.domain.analytics.query_service import (
    AnalyticsQueryService,
    UserMetricsAdmin,
    UserMetricsShared,
)
from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.value_repository import ValueAnalyticsStore
from dashboard.backend.tests.domain.analytics.test_fact_reads import NOW, YESTERDAY, fact
from dashboard.backend.tests.domain.analytics.test_value_repository import (
    SyntheticAgentStore,
    SyntheticCreditsStore,
    SyntheticProviderStore,
    SyntheticRunStore,
)
from dashboard.backend.users import UserStore


_ADMIN_ONLY = {"active_days_30d", "lifecycle", "operational", "user_group", "group_badge", "access_log"}
_SHARED = {"user_id", "facts_as_of", "runs_7d", "runs_30d", "own_spend_micro", "operator_cost_micro", "tier", "credit_balance_micro", "data_quality"}


@pytest.fixture
def metrics_fixture(tmp_path):
    db_path = tmp_path / "metrics.db"
    users = UserStore(db_path=db_path)
    admin = users.create_user("admin@example.test", "Admin", "SecurePass1!")
    users.apply_admin_patch(int(admin["id"]), role="admin")
    subject = users.create_user("subject@example.test", "Subject", "SecurePass1!")
    users.apply_admin_patch(int(subject["id"]), user_group="partner")
    # create_user stamps the real clock, which is later than the fixed NOW;
    # lifecycle evidence may not postdate its as_of.
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE users SET created_at = ?",
            ((NOW - timedelta(days=90)).isoformat(),),
        )
    analytics = AnalyticsStore(db_path=db_path)
    credits = SyntheticCreditsStore(tmp_path / "credits.db")
    value_store = ValueAnalyticsStore(
        analytics,
        credits,
        # One enabled platform lane and a positive balance: operationally healthy.
        provider_base=SyntheticProviderStore(
            {}, [{"provider_id": "openrouter", "status": "enabled", "platform_enabled": True}]
        ),
        agent_base=SyntheticAgentStore(),
        run_base=SyntheticRunStore({}),
    )
    user_id = int(subject["id"])
    value_store.record_activity(user_id, occurred_at=NOW - timedelta(days=20), activating=True, now=NOW)
    value_store.record_activity(user_id, occurred_at=NOW - timedelta(days=1), activating=False, now=NOW)
    value_store.upsert_daily_facts(
        [fact(user_id, YESTERDAY - timedelta(days=offset), runs_requested=2, runs_completed=1,
              runs_failed=1, operator_cost_micro=100_000, own_spend_micro=40_000)
         for offset in range(10)]
    )
    analytics.record_admin_access(int(admin["id"]), user_id, "overview")
    service = AnalyticsQueryService(store=analytics, user_store=users, value_store=value_store)
    return service, user_id


def test_the_user_audience_never_sees_admin_fields(metrics_fixture):
    service, user_id = metrics_fixture

    metrics = service.get_user_metrics(user_id, audience="user", now=NOW)

    assert type(metrics) is UserMetricsShared
    # Absent, not null -- on every serialisation path, including the one
    # FastAPI uses, because the model has no such fields at all.
    for payload in (metrics.model_dump(), json.loads(metrics.model_dump_json())):
        assert _ADMIN_ONLY.isdisjoint(payload)
        assert set(payload) == _SHARED


def test_both_audiences_agree_on_the_shared_numbers(metrics_fixture):
    service, user_id = metrics_fixture

    admin = service.get_user_metrics(user_id, audience="admin", now=NOW).model_dump()
    user = service.get_user_metrics(user_id, audience="user", now=NOW).model_dump()

    for field in _SHARED:
        assert admin[field] == user[field], field
    assert admin["runs_7d"] == {"requested": 14, "completed": 7, "failed": 7, "cancelled": 0}
    assert admin["runs_30d"] == {"requested": 20, "completed": 10, "failed": 10, "cancelled": 0}
    assert admin["operator_cost_micro"] == 1_000_000
    assert admin["own_spend_micro"] == 400_000
    assert admin["facts_as_of"] == YESTERDAY
    assert admin["credit_balance_micro"] == 1_000_000  # SyntheticCreditsStore default
    assert admin["data_quality"] == "partial"  # 10 of the 30 trailing days have a row


def test_the_admin_audience_sees_every_field(metrics_fixture):
    service, user_id = metrics_fixture

    metrics = service.get_user_metrics(user_id, audience="admin", now=NOW)
    payload = metrics.model_dump()

    assert type(metrics) is UserMetricsAdmin
    assert _ADMIN_ONLY <= set(payload)
    assert payload["active_days_30d"] == 10
    assert payload["lifecycle"]["segment"] == "core"
    assert payload["operational"]["state"] == "healthy"
    assert payload["user_group"] == "partner"
    assert payload["group_badge"] == "partner"
    assert payload["access_log"] == ["overview"]


def test_get_user_metrics_rejects_an_unknown_audience(metrics_fixture):
    service, user_id = metrics_fixture
    with pytest.raises(ValueError):
        service.get_user_metrics(user_id, audience="public", now=NOW)


def test_get_user_metrics_raises_lookup_error_for_an_unknown_user(metrics_fixture):
    service, _user_id = metrics_fixture
    with pytest.raises(LookupError):
        service.get_user_metrics(404, audience="admin", now=NOW)
