"""Admin Analytics query-service and API contract coverage."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import date, datetime, timedelta, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from dashboard.backend import users as users_module
from dashboard.backend.app import app
from dashboard.backend.domain.analytics.metrics import AnalyticsMetricFilters
from dashboard.backend.domain.analytics.query_service import (
    AnalyticsOverview,
    AnalyticsQueryService,
    get_analytics_query_service,
    get_value_analytics_query_service,
)
from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.repository_common import AnalyticsStoreError
from dashboard.backend.domain.analytics.rollups import (
    AnalyticsRollupStore,
    DailyRollup,
)
from dashboard.backend.domain.analytics.service import AnalyticsService
from dashboard.backend.domain.analytics.service import get_analytics_service
from dashboard.backend.domain.analytics.models import (
    FrontendAnalyticsEvent,
    RequestAnalyticsContext,
)
from dashboard.backend.domain.analytics.value_queries import (
    CommercialAnalyticsResponse,
    GroupAnalyticsResponse,
    LifecycleAnalyticsResponse,
    OperationalAnalyticsResponse,
    PaginatedValueUsers,
    RetentionAnalyticsResponse,
    SectionAvailability,
    UserGroupSummary,
    UserValueFilters,
    ValueUserProfile,
)
from dashboard.backend.domain.analytics.value_repository import UserDailyFact
from dashboard.backend.domain.user_groups import (
    DEFAULT_USER_GROUP,
    USER_GROUP_LABELS,
    USER_GROUPS,
)
from dashboard.backend.users import UserStore


NOW = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "admin_analytics"


def _contract(name: str, model):
    payload = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
    return model.model_validate(payload)


class FixtureValueQueryService:
    def __init__(self, subject_id: int):
        self.subject_id = subject_id
        self.calls = []

    def get_lifecycle(self, **kwargs):
        self.calls.append(("lifecycle", kwargs))
        return _contract("lifecycle.json", LifecycleAnalyticsResponse)

    def get_retention(self, **kwargs):
        self.calls.append(("retention", kwargs))
        return _contract("retention.json", RetentionAnalyticsResponse)

    def get_commercial(self, **kwargs):
        self.calls.append(("commercial", kwargs))
        return _contract("commercial.json", CommercialAnalyticsResponse)

    def get_operational(self, **kwargs):
        self.calls.append(("operational", kwargs))
        return _contract("operational.json", OperationalAnalyticsResponse)

    def get_groups(self, **kwargs):
        self.calls.append(("groups", kwargs))
        selected = kwargs.get("user_group")
        return GroupAnalyticsResponse(
            as_of=NOW,
            groups=[
                UserGroupSummary(
                    group=group,
                    label=USER_GROUP_LABELS[group],
                    users=(1 if selected in {None, group} and group == DEFAULT_USER_GROUP else 0),
                    successful_run_users=0,
                    repeat_users=0,
                    total_runs=0,
                    atl_cost_micro_usd=0,
                    paid_users=0,
                )
                for group in USER_GROUPS
            ],
            selected_user_group=selected,
            availability=SectionAvailability(
                available=True,
                status="ready",
                coverage_start=date(2026, 8, 1),
                coverage_end=date(2026, 8, 30),
            ),
        )

    def list_users(self, **kwargs):
        self.calls.append(("users", kwargs))
        response = _contract("users.json", PaginatedValueUsers)
        item = response.items[0].model_copy(
            update={
                "user_id": self.subject_id,
                "profile_path": f"/admin/analytics/users/{self.subject_id}",
            }
        )
        return response.model_copy(
            update={
                "items": [item],
                "total": 1,
                "limit": kwargs["limit"],
                "offset": kwargs["offset"],
            }
        )

    def get_user_profile(self, **kwargs):
        self.calls.append(("profile", kwargs))
        if kwargs["user_id"] != self.subject_id:
            raise LookupError("synthetic missing user")
        response = _contract("user_detail.json", ValueUserProfile)
        return response.model_copy(
            update={
                "user_id": self.subject_id,
                "commercial": response.commercial.model_copy(
                    update={"user_id": self.subject_id}
                ),
            }
        )


class QueryUsers:
    def __init__(self, rows):
        self.rows = list(rows)

    def list_users_admin(self, *, limit=100, offset=0, query=None):
        rows = self.rows
        if query:
            needle = query.lower()
            rows = [
                row
                for row in rows
                if needle in row["email"].lower()
                or needle in row["display_name"].lower()
            ]
        return rows[offset : offset + limit]

    def get_user_admin(self, user_id):
        return next((row for row in self.rows if row["id"] == user_id), None)


def _fixture(tmp_path):
    path = tmp_path / "query.db"
    users = [
        {
            "id": 1,
            "email": "one@example.test",
            "display_name": "One",
            "role": "user",
            "created_at": (NOW - timedelta(days=20)).isoformat(),
        },
        {
            "id": 2,
            "email": "admin@example.test",
            "display_name": "Admin",
            "role": "admin",
            "created_at": (NOW - timedelta(days=30)).isoformat(),
        },
    ]
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                email TEXT NOT NULL,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            "INSERT INTO users VALUES (?, ?, ?, 'x', ?, ?)",
            [
                (
                    row["id"],
                    row["email"],
                    row["display_name"],
                    row["role"],
                    row["created_at"],
                )
                for row in users
            ],
        )
    analytics = AnalyticsStore(path)
    return (
        analytics,
        AnalyticsService(analytics),
        AnalyticsRollupStore(analytics),
        QueryUsers(users),
    )


def _event(service, name, at, source_id, **kwargs):
    return service.record_server_event(
        event_name=name,
        user_id=1,
        source_event_id=source_id,
        source_record_type=kwargs.pop("source_record_type", "run"),
        source_record_id=kwargs.pop("source_record_id", source_id.rsplit(":", 1)[-1]),
        occurred_at=at,
        **kwargs,
    ).event


def test_query_service_merges_completed_rollups_with_current_raw_day(tmp_path):
    analytics, events, rollups, users = _fixture(tmp_path)
    yesterday = NOW.date() - timedelta(days=1)
    rollups.replace_day(
        yesterday,
        [
            DailyRollup(
                rollup_date=yesterday,
                metric_name="terminal_completed",
                value_count=4,
                updated_at=datetime.combine(
                    NOW.date(), datetime.min.time(), tzinfo=timezone.utc
                ),
            ),
            DailyRollup(
                rollup_date=yesterday,
                metric_name="daily_active_users",
                value_count=3,
                updated_at=datetime.combine(
                    NOW.date(), datetime.min.time(), tzinfo=timezone.utc
                ),
            ),
        ],
    )
    _event(
        events,
        "backtest_completed",
        NOW - timedelta(minutes=2),
        "run:backtest_completed:today",
        outcome="succeeded",
    )
    service = AnalyticsQueryService(store=analytics, user_store=users)

    overview = service.get_overview(
        now=NOW,
        filters=AnalyticsMetricFilters(
            start=datetime.combine(
                yesterday, datetime.min.time(), tzinfo=timezone.utc
            ),
            end=NOW,
        ),
    )

    assert overview.completed_runs == 5
    assert overview.daily_completed_runs[yesterday.isoformat()] == 4
    assert overview.daily_completed_runs[NOW.date().isoformat()] == 1
    assert overview.last_updated == NOW
    assert overview.availability["growth"].available is True


def test_billing_lane_mix_counts_model_calls_from_rollups_and_today_honouring_filters(tmp_path):
    analytics, events, rollups, users = _fixture(tmp_path)
    yesterday = NOW.date() - timedelta(days=1)
    stamp = datetime.combine(NOW.date(), datetime.min.time(), tzinfo=timezone.utc)

    def usage_rollup(billing_mode, model_id, count):
        return DailyRollup(
            rollup_date=yesterday,
            metric_name="event_count",
            event_name="model_usage_recorded",
            billing_mode=billing_mode,
            provider_id="openrouter",
            model_id=model_id,
            value_count=count,
            updated_at=stamp,
        )

    rollups.replace_day(
        yesterday,
        [
            usage_rollup("platform_credits", "a", 3),
            usage_rollup("platform_credits", "b", 1),
            usage_rollup("byok", "a", 2),
            # One settlement per non-zero credit bucket: not a call counter.
            DailyRollup(
                rollup_date=yesterday,
                metric_name="event_count",
                event_name="credits_settled",
                billing_mode="platform_credits",
                value_count=8,
                updated_at=stamp,
            ),
        ],
    )
    for index, (billing_mode, model_id) in enumerate(
        (("platform_credits", "a"), ("byok", "a"), ("byok", "b"))
    ):
        _event(
            events,
            "model_usage_recorded",
            NOW - timedelta(minutes=10 + index),
            f"resource:model_usage_recorded:run-today:{index}",
            correlation_id="run-today",
            provider_id="openrouter",
            model_id=model_id,
            billing_mode=billing_mode,
            outcome="succeeded",
            properties={"input_tokens": 1, "output_tokens": 1, "cost_micro_usd": 1},
        )
    _event(
        events,
        "credits_settled",
        NOW - timedelta(minutes=5),
        "resource:credits_settled:reservation-today:grant",
        source_record_type="credit_reservation",
        source_record_id="reservation-today",
        correlation_id="run-today",
        billing_mode="platform_credits",
        properties={"amount_micro": 100, "bucket": "grant"},
    )
    service = AnalyticsQueryService(store=analytics, user_store=users)
    start = datetime.combine(yesterday, datetime.min.time(), tzinfo=timezone.utc)

    def mix(**filters):
        overview = service.get_overview(
            now=NOW,
            filters=AnalyticsMetricFilters(start=start, end=NOW, **filters),
        )
        assert overview.availability["growth"].available is True
        return [
            (row.day, row.platform_credits, row.byok)
            for row in overview.billing_lane_mix
        ]

    today = NOW.date().isoformat()
    assert mix() == [(yesterday.isoformat(), 4, 2), (today, 1, 2)]
    assert mix(model_id="a") == [(yesterday.isoformat(), 3, 2), (today, 1, 1)]
    assert mix(billing_mode="byok") == [(yesterday.isoformat(), 0, 2), (today, 0, 2)]


def test_billing_lane_mix_carries_each_lane_in_credits_honouring_filters(tmp_path):
    """Both lanes read cost off model_usage_recorded: completed days from the
    platform_model_cost_usd / byok_estimated_cost_usd rollups, today from raw
    events. The platform lane sums to the headline under every filter, and
    the BYOK estimate never enters it."""
    analytics, events, rollups, users = _fixture(tmp_path)
    yesterday = NOW.date() - timedelta(days=1)
    stamp = datetime.combine(NOW.date(), datetime.min.time(), tzinfo=timezone.utc)

    def cost_rollup(metric, billing_mode, sum_micro, provider_id="", model_id=""):
        return DailyRollup(
            rollup_date=yesterday,
            metric_name=metric,
            billing_mode=billing_mode,
            provider_id=provider_id,
            model_id=model_id,
            value_sum_micro=sum_micro,
            updated_at=stamp,
        )

    rollups.replace_day(
        yesterday,
        [
            # The day's undimensioned total plus its per-model split: summing
            # both would double the lane.
            cost_rollup("platform_model_cost_usd", "platform_credits", 5_000_000),
            cost_rollup("platform_model_cost_usd", "platform_credits", 3_000_000, "openrouter", "a"),
            cost_rollup("platform_model_cost_usd", "platform_credits", 2_000_000, "commonstack", "b"),
            cost_rollup("byok_estimated_cost_usd", "byok", 700_000, "openrouter", "a"),
            cost_rollup("byok_estimated_cost_usd", "byok", 300_000, "commonstack", "b"),
            # Written zero or not by the current rollup; without them the day
            # would read as one rolled up before unpriced calls were counted.
            cost_rollup("byok_unpriced_calls", "byok", 0, "openrouter", "a"),
            cost_rollup("byok_unpriced_calls", "byok", 0, "commonstack", "b"),
        ],
    )
    for index, (billing_mode, provider_id, model_id, cost) in enumerate(
        (
            ("platform_credits", "openrouter", "a", 400_000),
            ("byok", "openrouter", "a", 90_000),
            ("byok", "commonstack", "b", 10_000),
        )
    ):
        _event(
            events,
            "model_usage_recorded",
            NOW - timedelta(minutes=10 + index),
            f"resource:model_usage_recorded:run-today:{index}",
            correlation_id="run-today",
            provider_id=provider_id,
            model_id=model_id,
            billing_mode=billing_mode,
            outcome="succeeded",
            properties=(
                {"input_tokens": 1, "output_tokens": 1, "cost_micro_usd": cost}
                if billing_mode == "platform_credits"
                else {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "cost_micro_usd": 0,
                    "estimated_cost_micro_usd": cost,
                }
            ),
        )
    service = AnalyticsQueryService(store=analytics, user_store=users)
    start = datetime.combine(yesterday, datetime.min.time(), tzinfo=timezone.utc)

    def lanes(**filters):
        overview = service.get_overview(
            now=NOW,
            filters=AnalyticsMetricFilters(start=start, end=NOW, **filters),
        )
        assert overview.availability["growth"].available is True
        rows = [
            (row.day, row.platform_cost_micro, row.byok_estimated_micro)
            for row in overview.billing_lane_mix
        ]
        # The per-day platform lane and the headline are one figure, and so is
        # the headline read alone (what /commercial publishes).
        assert sum(row[1] for row in rows) == round(
            overview.platform_model_cost_usd * 1_000_000
        )
        assert service.get_platform_model_cost_micro(
            filters=AnalyticsMetricFilters(start=start, end=NOW, **filters), now=NOW
        ) == round(overview.platform_model_cost_usd * 1_000_000)
        return rows

    day, today = yesterday.isoformat(), NOW.date().isoformat()
    assert lanes() == [(day, 5_000_000, 1_000_000), (today, 400_000, 100_000)]
    assert lanes(provider_id="openrouter") == [
        (day, 3_000_000, 700_000),
        (today, 400_000, 90_000),
    ]
    assert lanes(model_id="b") == [(day, 2_000_000, 300_000), (today, 0, 10_000)]
    assert lanes(billing_mode="byok") == [(day, 0, 1_000_000), (today, 0, 100_000)]
    assert lanes(billing_mode="platform_credits") == [
        (day, 5_000_000, 0),
        (today, 400_000, 0),
    ]


def test_failed_raw_event_read_marks_growth_unavailable_not_a_rollup_only_range(
    tmp_path, monkeypatch, capsys
):
    """Today's lanes, today's platform cost and the token totals come from the
    raw events. When that read failed, growth stayed available and published
    the rollups alone as the whole range -- yesterday's cost as the range's,
    today's calls missing from every lane -- with status "ready"."""
    analytics, events, rollups, users = _fixture(tmp_path)
    yesterday = NOW.date() - timedelta(days=1)
    stamp = datetime.combine(NOW.date(), datetime.min.time(), tzinfo=timezone.utc)
    rollups.replace_day(
        yesterday,
        [
            DailyRollup(
                rollup_date=yesterday,
                metric_name="platform_model_cost_usd",
                billing_mode="platform_credits",
                value_sum_micro=5_000_000,
                updated_at=stamp,
            )
        ],
    )
    service = AnalyticsQueryService(store=analytics, user_store=users)

    def refuse(**_kwargs):
        raise AnalyticsStoreError("stored Analytics properties are invalid")

    monkeypatch.setattr(service.query_store, "list_metric_events", refuse)
    filters = AnalyticsMetricFilters(
        start=datetime.combine(yesterday, datetime.min.time(), tzinfo=timezone.utc),
        end=NOW,
    )
    overview = service.get_overview(now=NOW, filters=filters)

    assert overview.availability["growth"].available is False
    assert overview.platform_model_cost_usd is None
    assert overview.billing_lane_mix == []
    assert service.get_platform_model_cost_micro(filters=filters, now=NOW) is None
    assert "ERROR platform model cost unavailable" in capsys.readouterr().out


def test_billing_lane_mix_counts_unpriced_byok_calls_honouring_filters(tmp_path):
    """A BYOK call with no list-price estimate is counted, not summed as 0, so
    the panel can say how much of the estimate is missing. Completed days read
    byok_unpriced_calls. A day rolled up before that metric existed has none,
    and is read per provider/model: with no estimate the calls predate PR #572
    and are all unpriced; with one, #572 priced an unlisted model at the $1/$5
    fallback, so the estimate stands only for a listed model. Deciding that per
    day instead made the unfiltered total and a per-model filter disagree."""
    analytics, events, rollups, users = _fixture(tmp_path)
    stamp = datetime.combine(NOW.date(), datetime.min.time(), tzinfo=timezone.utc)
    counted, legacy, legacy_priced = (NOW.date() - timedelta(days=n) for n in (1, 2, 3))
    listed = "openai/gpt-5.5"

    def row(day, metric, model_id, *, count=0, sum_micro=0, event_name=""):
        return DailyRollup(
            rollup_date=day,
            metric_name=metric,
            event_name=event_name,
            billing_mode="byok",
            provider_id="openrouter",
            model_id=model_id,
            value_count=count,
            value_sum_micro=sum_micro,
            updated_at=stamp,
        )

    def calls(day, model_id, count):
        return row(day, "event_count", model_id, count=count, event_name="model_usage_recorded")

    rollups.replace_day(
        counted,
        [
            calls(counted, "a", 4),
            calls(counted, "b", 2),
            row(counted, "byok_estimated_cost_usd", "a", sum_micro=300),
            row(counted, "byok_unpriced_calls", "a", count=1),
            row(counted, "byok_estimated_cost_usd", "b", sum_micro=0),
            row(counted, "byok_unpriced_calls", "b", count=2),
        ],
    )
    rollups.replace_day(legacy, [calls(legacy, "a", 5)])
    rollups.replace_day(
        legacy_priced,
        [
            calls(legacy_priced, listed, 3),
            row(legacy_priced, "byok_estimated_cost_usd", listed, sum_micro=90),
            # "b" is not on the price list: #572's 40 is the $1/$5 guess.
            calls(legacy_priced, "b", 2),
            row(legacy_priced, "byok_estimated_cost_usd", "b", sum_micro=40),
        ],
    )
    for index, properties in enumerate(
        (
            {"cost_micro_usd": 0, "estimated_cost_micro_usd": 7},
            {"cost_micro_usd": 0},
            {"cost_micro_usd": 0},
        )
    ):
        _event(
            events,
            "model_usage_recorded",
            NOW - timedelta(minutes=10 + index),
            f"resource:model_usage_recorded:run-today:{index}",
            correlation_id="run-today",
            provider_id="openrouter",
            model_id="a" if index < 2 else "b",
            billing_mode="byok",
            outcome="succeeded",
            properties={"input_tokens": 1, "output_tokens": 1, **properties},
        )
    service = AnalyticsQueryService(store=analytics, user_store=users)
    start = datetime.combine(legacy_priced, datetime.min.time(), tzinfo=timezone.utc)

    def unpriced(**filters):
        overview = service.get_overview(
            now=NOW,
            filters=AnalyticsMetricFilters(start=start, end=NOW, **filters),
        )
        assert overview.availability["growth"].available is True
        return [
            (row.day, row.byok, row.byok_estimated_micro, row.byok_unpriced)
            for row in overview.billing_lane_mix
        ]

    today = NOW.date().isoformat()
    assert unpriced() == [
        (legacy_priced.isoformat(), 5, 90, 2),
        (legacy.isoformat(), 5, 0, 5),
        (counted.isoformat(), 6, 300, 3),
        (today, 3, 7, 2),
    ]
    assert unpriced(model_id="a") == [
        (legacy.isoformat(), 5, 0, 5),
        (counted.isoformat(), 4, 300, 1),
        (today, 2, 7, 1),
    ]
    # The filtered views partition the unfiltered one, legacy day included.
    assert unpriced(model_id=listed) == [(legacy_priced.isoformat(), 3, 90, 0)]
    assert unpriced(model_id="b") == [
        (legacy_priced.isoformat(), 2, 0, 2),
        (counted.isoformat(), 2, 0, 2),
        (today, 1, 0, 1),
    ]


def test_activity_reports_atl_cost_only_for_byok_usage(tmp_path):
    """The activity API's cost_micro_usd is ATL cost. A BYOK call costs ATL
    nothing -- including one from the PR #572 era, whose event carried the
    list-price estimate in that very field."""
    analytics, events, _rollups, users = _fixture(tmp_path)
    for index, (billing_mode, properties) in enumerate(
        (
            ("platform_credits", {"cost_micro_usd": 1_250}),
            ("byok", {"cost_micro_usd": 0, "estimated_cost_micro_usd": 900}),
            ("byok", {"cost_micro_usd": 420}),
        )
    ):
        _event(
            events,
            "model_usage_recorded",
            NOW - timedelta(minutes=10 + index),
            f"resource:model_usage_recorded:run-activity:{index}",
            correlation_id="run-activity",
            provider_id="openrouter",
            model_id="a",
            billing_mode=billing_mode,
            outcome="succeeded",
            properties={"input_tokens": 1, "output_tokens": 1, **properties},
        )
    service = AnalyticsQueryService(store=analytics, user_store=users)

    page = service.get_user_activity(user_id=1, section="usage", limit=10, cursor=None)

    assert sorted(
        (item.billing_mode, item.cost_micro_usd) for item in page.items
    ) == [("byok", 0), ("byok", 0), ("platform_credits", 1_250)]


def test_billing_lane_mix_skips_a_day_with_no_calls_and_no_cost(tmp_path):
    """Every rolled-up day carries a platform cost total, zero or not. A zero
    day is not activity: listing it would replace the panel's empty state with
    flat lines."""
    analytics, _events, rollups, users = _fixture(tmp_path)
    yesterday = NOW.date() - timedelta(days=1)
    stamp = datetime.combine(NOW.date(), datetime.min.time(), tzinfo=timezone.utc)
    rollups.replace_day(
        yesterday,
        [
            DailyRollup(
                rollup_date=yesterday,
                metric_name="platform_model_cost_usd",
                billing_mode="platform_credits",
                value_sum_micro=0,
                updated_at=stamp,
            )
        ],
    )
    service = AnalyticsQueryService(store=analytics, user_store=users)

    overview = service.get_overview(
        now=NOW,
        filters=AnalyticsMetricFilters(
            start=datetime.combine(yesterday, datetime.min.time(), tzinfo=timezone.utc),
            end=NOW,
        ),
    )

    assert overview.availability["growth"].available is True
    assert overview.billing_lane_mix == []


def test_profile_is_display_safe_and_reads_facts_not_the_event_window(tmp_path):
    analytics, events, _rollups, users = _fixture(tmp_path)
    from dashboard.backend.domain.analytics.value_repository import ValueAnalyticsStore
    from dashboard.backend.tests.domain.analytics.test_fact_reads import fact
    from dashboard.backend.tests.domain.analytics.test_value_repository import (
        SyntheticAgentStore, SyntheticCreditsStore, SyntheticProviderStore, SyntheticRunStore,
    )

    value_store = ValueAnalyticsStore(
        analytics, SyntheticCreditsStore(tmp_path / "credits.db"),
        # One enabled platform lane and a positive balance: operationally healthy.
        provider_base=SyntheticProviderStore(
            {}, [{"provider_id": "openrouter", "status": "enabled", "platform_enabled": True}]
        ),
        agent_base=SyntheticAgentStore(), run_base=SyntheticRunStore({}),
    )
    yesterday = NOW.date() - timedelta(days=1)
    value_store.record_activity(1, occurred_at=NOW - timedelta(hours=1), activating=True, now=NOW)
    value_store.upsert_daily_facts(
        [fact(1, yesterday, runs_requested=3, runs_completed=2, runs_failed=1, operator_cost_micro=250_000, own_spend_micro=100)]
    )
    _event(events, "model_usage_recorded", NOW - timedelta(minutes=50), "resource:model_usage_recorded:run-1:0",
           correlation_id="run-1", provider_id="openrouter", model_id="openai/gpt-5.5",
           billing_mode="platform_credits", outcome="succeeded",
           properties={"input_tokens": 120, "output_tokens": 30, "cost_micro_usd": 250_000})
    service = AnalyticsQueryService(store=analytics, user_store=users, value_store=value_store)

    def refuse_window_scan(**_kwargs):
        raise AssertionError("the profile must not scan the event window")

    service.query_store.rollups.list_events = refuse_window_scan
    recorder = _WindowRecordingQueryStore(service.query_store)
    service.query_store = recorder

    profile = service.get_user_profile(user_id=1, now=NOW, start=NOW - timedelta(days=30), end=NOW)
    serialized = profile.model_dump(mode="json")

    assert profile.state.status == "active"
    assert profile.state.evidence_event_ids == []
    assert profile.run_summary == {"requested": 3, "completed": 2, "failed": 1, "cancelled": 0}
    assert profile.platform_model_cost_usd == 0.25
    assert profile.credits_debited_micro == 100
    assert profile.input_tokens == 120
    assert profile.default_provider == "openrouter"
    assert profile.activation_milestones["backtest_completed"] == NOW - timedelta(hours=1)
    assert profile.last_meaningful_activity == NOW - timedelta(hours=1)
    assert recorder.windows == []  # no list_metric_events call: the only raw read is one timeline page
    assert "properties" not in str(serialized)
    assert "session_id" not in str(serialized)


def test_activity_sections_page_independently_and_hide_session_ids(tmp_path):
    analytics, events, _rollups, users = _fixture(tmp_path)
    for index in range(3):
        _event(
            events,
            "backtest_completed",
            NOW - timedelta(minutes=index + 1),
            f"run:backtest_completed:run-{index}",
            source_record_id=f"run-{index}",
            outcome="succeeded",
        )
    context = RequestAnalyticsContext(
        country_code="US",
        device_category="desktop",
        browser_family="Chrome",
    )
    session_id = str(uuid4())
    for index, name in enumerate(("page_viewed", "session_heartbeat")):
        events.accept_frontend_event(
            user={"id": 1},
            payload=FrontendAnalyticsEvent(
                event_id=str(uuid4()),
                schema_version=1,
                event_name=name,
                session_id=session_id,
                occurred_at=NOW - timedelta(minutes=10 - index),
                page_view="agents",
                properties=({} if name == "page_viewed" else {"visible_ms": 500}),
            ),
            context=context,
            received_at=NOW,
        )
    service = AnalyticsQueryService(store=analytics, user_store=users)

    first_runs = service.get_user_activity(
        user_id=1,
        section="runs",
        limit=2,
        cursor=None,
    )
    second_runs = service.get_user_activity(
        user_id=1,
        section="runs",
        limit=2,
        cursor=first_runs.next_cursor,
    )
    sessions = service.get_user_activity(
        user_id=1,
        section="sessions",
        limit=2,
        cursor=None,
        now=NOW,
    )

    assert len(first_runs.items) == 2
    assert len(second_runs.items) == 1
    assert first_runs.next_cursor is not None
    assert sessions.items[0].session_event_count == 2
    assert sessions.items[0].visible_ms == 500
    assert "session_id" not in str(sessions.model_dump(mode="json"))


def test_sessions_section_is_bounded_to_thirty_days(tmp_path):
    analytics, events, _rollups, users = _fixture(tmp_path)
    context = RequestAnalyticsContext(
        country_code="US", device_category="desktop", browser_family="Chrome"
    )
    for days_ago in (2, 45):
        occurred_at = NOW - timedelta(days=days_ago)
        events.accept_frontend_event(
            user={"id": 1},
            payload=FrontendAnalyticsEvent(
                event_id=str(uuid4()),
                schema_version=1,
                event_name="page_viewed",
                session_id=str(uuid4()),
                occurred_at=occurred_at,
                page_view="agents",
                properties={},
            ),
            context=context,
            # Ingest rejects occurred_at older than received_at - 24h.
            received_at=occurred_at,
        )
    service = AnalyticsQueryService(store=analytics, user_store=users)

    sessions = service.get_user_activity(
        user_id=1, section="sessions", limit=10, cursor=None, now=NOW
    )

    assert len(sessions.items) == 1
    assert sessions.items[0].occurred_at == NOW - timedelta(days=2)
    import inspect

    from dashboard.backend.domain.analytics.query_service import AnalyticsQueryStore

    assert "occurred_at >=" in inspect.getsource(AnalyticsQueryStore.list_session_rows)


def test_overview_marks_rollup_backed_panels_unavailable_when_rollups_fail(
    tmp_path, monkeypatch, capsys
):
    analytics, events, _rollups, users = _fixture(tmp_path)
    _event(
        events,
        "backtest_completed",
        NOW - timedelta(minutes=1),
        "run:backtest_completed:current",
        outcome="succeeded",
    )
    service = AnalyticsQueryService(store=analytics, user_store=users)
    # Yesterday's facts have landed, so attention has a day to read.
    service.value_store.upsert_daily_facts([_fact(1, NOW.date() - timedelta(days=1))])
    monkeypatch.setattr(
        service.query_store.rollups,
        "list_rollups",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("private detail")),
    )

    overview = service.get_overview(
        now=NOW,
        filters=AnalyticsMetricFilters(start=NOW - timedelta(days=1), end=NOW),
    )

    assert overview.active_users_7d is None
    assert overview.completed_runs is None
    assert overview.availability["snapshot"].available is False
    assert overview.availability["growth"].available is False
    assert overview.availability["funnel"].available is False
    assert overview.availability["growth"].error_code == "temporarily_unavailable"
    # M1: friction (top_failure_categories) is computed from the funnel's
    # reads, so it goes down with them; attention reads facts and stays up.
    assert overview.availability["friction"].available is False
    assert overview.availability["attention"].available is True
    # M2: the swallowed failure reaches the log, by class name only.
    out = capsys.readouterr().out
    assert "WARNING: analytics.read.overview_rollups_failed category=RuntimeError" in out
    assert "private detail" not in out


class _WindowRecordingQueryStore:
    """Wraps AnalyticsQueryStore and records every raw-event read window."""

    def __init__(self, inner):
        self.inner = inner
        self.windows = []

    def list_metric_events(self, *, start, end, include_internal):
        self.windows.append((start, end))
        return self.inner.list_metric_events(
            start=start, end=end, include_internal=include_internal
        )

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_overview_reads_raw_events_for_the_current_utc_day_only(tmp_path):
    """Design §6.15 item 2: one bounded one-day scan, never the filter window."""

    analytics, _events, _rollups, users = _fixture(tmp_path)
    service = AnalyticsQueryService(store=analytics, user_store=users)
    recorder = _WindowRecordingQueryStore(service.query_store)
    service.query_store = recorder

    service.get_overview(
        now=NOW,
        filters=AnalyticsMetricFilters(start=NOW - timedelta(days=30), end=NOW),
    )

    assert len(recorder.windows) == 1
    start, end = recorder.windows[0]
    assert start == datetime.combine(NOW.date(), datetime.min.time(), tzinfo=timezone.utc)
    assert end == NOW


def test_overview_skips_the_raw_scan_for_a_window_that_ends_before_today(tmp_path):
    analytics, _events, _rollups, users = _fixture(tmp_path)
    service = AnalyticsQueryService(store=analytics, user_store=users)
    recorder = _WindowRecordingQueryStore(service.query_store)
    service.query_store = recorder

    overview = service.get_overview(
        now=NOW,
        filters=AnalyticsMetricFilters(
            start=NOW - timedelta(days=10),
            end=datetime.combine(NOW.date(), datetime.min.time(), tzinfo=timezone.utc),
        ),
    )

    assert recorder.windows == []
    # Nothing to read is not a failed read.
    assert overview.availability["growth"].available is True
    assert overview.availability["funnel"].available is True


def test_overview_trailing_metrics_and_funnel_come_from_rollups(tmp_path):
    analytics, events, rollups, users = _fixture(tmp_path)
    yesterday = NOW.date() - timedelta(days=1)
    stamp = datetime.combine(NOW.date(), datetime.min.time(), tzinfo=timezone.utc)

    def rollup(metric_name, value_count, **dimensions):
        return DailyRollup(
            rollup_date=yesterday,
            metric_name=metric_name,
            value_count=value_count,
            updated_at=stamp,
            **dimensions,
        )

    rollups.replace_day(
        yesterday,
        [
            rollup("rolling_active_users_7d", 9),
            rollup("mature_signup_cohort_users", 4),
            rollup("first_success_within_7d_users", 1),
            rollup("users_with_first_success", 8),
            rollup("users_with_repeat_success_24h_30d", 2),
            rollup("activation_users", 5, event_name="account_signed_up"),
            rollup("activation_users", 2, event_name="backtest_completed"),
            rollup(
                "event_count",
                3,
                event_name="backtest_failed",
                outcome="failed",
                error_category="provider_timeout",
            ),
            rollup("input_tokens", 1000),
            rollup("output_tokens", 250),
        ],
    )
    _event(
        events,
        "backtest_completed",
        NOW - timedelta(minutes=2),
        "run:backtest_completed:today",
        outcome="succeeded",
    )
    _event(
        events,
        "backtest_failed",
        NOW - timedelta(minutes=1),
        "run:backtest_failed:today-fail",
        source_record_id="today-fail",
        outcome="failed",
        error_category="provider_timeout",
    )
    service = AnalyticsQueryService(store=analytics, user_store=users)

    overview = service.get_overview(
        now=NOW,
        filters=AnalyticsMetricFilters(
            start=datetime.combine(yesterday, datetime.min.time(), tzinfo=timezone.utc),
            end=NOW,
        ),
    )

    assert overview.active_users_7d == 9
    assert overview.first_success_conversion == 0.25
    assert overview.repeat_run_rate == 0.25
    assert overview.activation_funnel == {
        "account_signed_up": 5,
        "credential_verified": 0,
        "agent_created": 0,
        "backtest_completed": 3,
    }
    assert [
        (c.error_category, c.event_count) for c in overview.top_failure_categories
    ] == [("provider_timeout", 4)]
    assert overview.input_tokens == 1000
    assert overview.output_tokens == 250
    assert overview.availability["snapshot"].available is True
    assert overview.availability["funnel"].available is True


def test_overview_failure_categories_count_events_filtered_or_not(tmp_path):
    """Failure categories sum ``event_count`` rows plus today's events, with
    or without a dimension filter (#647). The per-category ``affected_users``
    rows older rollups still hold are not read."""

    analytics, events, rollups, users = _fixture(tmp_path)
    yesterday = NOW.date() - timedelta(days=1)
    stamp = datetime.combine(NOW.date(), datetime.min.time(), tzinfo=timezone.utc)

    def failed_rollup(model_id, count):
        return DailyRollup(
            rollup_date=yesterday,
            metric_name="event_count",
            event_name="backtest_failed",
            provider_id="openrouter",
            model_id=model_id,
            outcome="failed",
            error_category="provider_timeout",
            value_count=count,
            updated_at=stamp,
        )

    rollups.replace_day(
        yesterday,
        [
            failed_rollup("a", 3),
            failed_rollup("b", 7),
            DailyRollup(
                rollup_date=yesterday,
                metric_name="affected_users",
                error_category="provider_timeout",
                value_count=2,
                updated_at=stamp,
            ),
        ],
    )
    for index in range(2):
        _event(
            events,
            "backtest_failed",
            NOW - timedelta(minutes=index + 1),
            f"run:backtest_failed:today-{index}",
            source_record_id=f"today-{index}",
            provider_id="openrouter",
            model_id="a",
            outcome="failed",
            error_category="provider_timeout",
        )
    service = AnalyticsQueryService(store=analytics, user_store=users)
    start = datetime.combine(yesterday, datetime.min.time(), tzinfo=timezone.utc)

    def categories(**filters):
        overview = service.get_overview(
            now=NOW,
            filters=AnalyticsMetricFilters(start=start, end=NOW, **filters),
        )
        return [
            (c.error_category, c.event_count)
            for c in overview.top_failure_categories
        ]

    # Unfiltered: 3 + 7 yesterday, 2 today; the legacy row adds nothing.
    assert categories() == [("provider_timeout", 12)]
    # Filtered: 3 yesterday on model a, 2 today.
    assert categories(model_id="a") == [("provider_timeout", 5)]


def _fact(user_id, day, **overrides):
    values = {
        "snapshot_date": day,
        "user_id": user_id,
        "lifecycle_segment": "growing",
        "lifecycle_reason_code": "growing_activated_below_core_threshold",
        "operational_state": "healthy",
        "operational_reason_code": "no_supported_issue",
        "tier": "unpaid",
        "user_group": "unknown",
        "active": True,
        "data_quality": "complete",
        "calculated_at": NOW,
    }
    values.update(overrides)
    return UserDailyFact(**values)


def test_overview_attention_comes_from_yesterdays_facts(tmp_path):
    """Design D20: the overview carries no five-state counts; attention lists
    yesterday's blocked / needs-attention users with their 30-day run totals
    and their current activity clock."""

    analytics, _events, _rollups, users = _fixture(tmp_path)
    service = AnalyticsQueryService(store=analytics, user_store=users)
    yesterday = NOW.date() - timedelta(days=1)
    with analytics._get_connection() as conn:
        conn.executemany(
            "INSERT INTO users VALUES (?, ?, ?, 'x', 'user', ?)",
            [
                (user_id, f"u{user_id}@example.test", f"U{user_id}", NOW.isoformat())
                for user_id in (3, 4, 5)
            ],
        )
    last_seen = NOW - timedelta(days=3)
    service.value_store.record_activity(
        1, occurred_at=last_seen, activating=False, now=last_seen
    )
    service.value_store.upsert_daily_facts(
        [
            _fact(
                1,
                yesterday,
                operational_state="needs_attention",
                operational_reason_code="invalid_default_credential",
                runs_requested=2,
                runs_failed=1,
            ),
            # Ten days earlier, inside the 30-day window: adds to the totals.
            _fact(1, yesterday - timedelta(days=10), runs_requested=3, runs_failed=2),
            _fact(
                3,
                yesterday,
                operational_state="blocked",
                operational_reason_code="account_restricted",
            ),
            _fact(4, yesterday, lifecycle_segment="dormant",
                  lifecycle_reason_code="dormant_previously_activated"),
            _fact(5, yesterday, lifecycle_segment="onboarding",
                  lifecycle_reason_code="onboarding_signed_up_not_activated"),
            # Today's row (if one existed) is not yesterday's population.
            _fact(5, NOW.date(), operational_state="blocked",
                  operational_reason_code="account_restricted"),
        ]
    )

    overview = service.get_overview(
        now=NOW,
        filters=AnalyticsMetricFilters(start=NOW - timedelta(days=1), end=NOW),
    )

    assert [
        (item.user_id, item.status, item.reason_code, item.recent_runs, item.recent_failures)
        for item in overview.users_needing_attention
    ] == [
        (1, "needs_attention", "invalid_default_credential", 5, 3),
        (3, "blocked", "account_restricted", 0, 0),
    ]
    first = overview.users_needing_attention[0]
    assert first.human_readable_reason == "The default model credential is invalid."
    assert first.last_meaningful_activity == last_seen
    assert first.display_name == "One"
    assert overview.users_needing_attention[1].last_meaningful_activity is None
    assert overview.availability["friction"].available is True
    assert overview.availability["attention"].available is True


def test_overview_attention_falls_back_to_the_latest_landed_fact_day(tmp_path, capsys):
    """Review #3: before the night's job lands (or all day, when it fails)
    yesterday has no rows, and [] marked available read as "nobody needs
    attention". The list joins the latest landed day, as /users does."""
    analytics, _events, _rollups, users = _fixture(tmp_path)
    service = AnalyticsQueryService(store=analytics, user_store=users)
    landed = NOW.date() - timedelta(days=2)
    service.value_store.upsert_daily_facts(
        [
            _fact(
                1,
                landed,
                operational_state="blocked",
                operational_reason_code="account_restricted",
            )
        ]
    )

    overview = service.get_overview(
        now=NOW, filters=AnalyticsMetricFilters(start=NOW - timedelta(days=1), end=NOW)
    )

    assert [item.user_id for item in overview.users_needing_attention] == [1]
    assert overview.availability["attention"].available is True
    assert "WARNING: analytics.read.facts_stale" in capsys.readouterr().out


def test_overview_attention_is_unavailable_when_no_fact_day_has_landed(tmp_path):
    analytics, _events, _rollups, users = _fixture(tmp_path)
    service = AnalyticsQueryService(store=analytics, user_store=users)

    overview = service.get_overview(
        now=NOW, filters=AnalyticsMetricFilters(start=NOW - timedelta(days=1), end=NOW)
    )

    assert overview.users_needing_attention == []
    assert overview.availability["attention"].available is False


def test_overview_attention_refuses_a_fact_day_older_than_the_nightly_gap(tmp_path):
    """The attention panel has no as-of field: after days of failed jobs, an
    old blocked list must read unavailable, not current."""
    analytics, _events, _rollups, users = _fixture(tmp_path)
    service = AnalyticsQueryService(store=analytics, user_store=users)
    service.value_store.upsert_daily_facts(
        [
            _fact(
                1,
                NOW.date() - timedelta(days=3),
                operational_state="blocked",
                operational_reason_code="account_restricted",
            )
        ]
    )

    overview = service.get_overview(
        now=NOW, filters=AnalyticsMetricFilters(start=NOW - timedelta(days=1), end=NOW)
    )

    assert overview.users_needing_attention == []
    assert overview.availability["attention"].available is False


def _midnight(day):
    return datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)


def test_overview_funnel_is_unavailable_while_a_rolled_up_day_lacks_activation_rows(tmp_path):
    """Review #5: a day rolled up before activation_users existed summed as
    zero under an available funnel. Until the backfill fills it, the funnel
    says unavailable; friction, from the same reads, stays up."""
    analytics, _events, rollups, users = _fixture(tmp_path)
    yesterday = NOW.date() - timedelta(days=1)
    service = AnalyticsQueryService(store=analytics, user_store=users)
    filters = AnalyticsMetricFilters(start=_midnight(yesterday), end=NOW)
    pre_metric_day = [
        DailyRollup(
            rollup_date=yesterday,
            metric_name="daily_active_users",
            value_count=3,
            updated_at=_midnight(NOW.date()),
        )
    ]
    rollups.replace_day(yesterday, pre_metric_day)

    stale = service.get_overview(now=NOW, filters=filters)

    assert stale.availability["funnel"].available is False
    assert stale.activation_funnel == {}
    assert stale.availability["friction"].available is True

    rollups.replace_day(
        yesterday,
        pre_metric_day
        + [
            DailyRollup(
                rollup_date=yesterday,
                metric_name="activation_users",
                event_name=event_name,
                value_count=2 if event_name == "account_signed_up" else 0,
                updated_at=_midnight(NOW.date()),
            )
            for event_name in ("account_signed_up", "credential_verified", "agent_created", "backtest_completed")
        ],
    )

    filled = service.get_overview(now=NOW, filters=filters)

    assert filled.availability["funnel"].available is True
    assert filled.activation_funnel["account_signed_up"] == 2


def test_overview_include_internal_does_not_add_internal_accounts_to_today(tmp_path):
    """Review #6: completed days come from rollups written without internal
    accounts, so today's raw slice must leave them out too, whatever the
    flag; mixing them produced figures that matched neither setting."""
    analytics, events, _rollups, users = _fixture(tmp_path)
    for index, user_id in enumerate((1, 2)):  # user 2 is the admin account
        events.record_server_event(
            event_name="backtest_completed",
            user_id=user_id,
            source_event_id=f"run:backtest_completed:today-{index}",
            source_record_type="run",
            source_record_id=f"today-{index}",
            outcome="succeeded",
            occurred_at=NOW - timedelta(minutes=5 + index),
        )
    service = AnalyticsQueryService(store=analytics, user_store=users)
    start = _midnight(NOW.date())

    internal_free = service.get_overview(
        now=NOW, filters=AnalyticsMetricFilters(start=start, end=NOW)
    )
    widened = service.get_overview(
        now=NOW, filters=AnalyticsMetricFilters(start=start, end=NOW, include_internal=True)
    )

    assert internal_free.completed_runs == 1
    assert widened.completed_runs == internal_free.completed_runs
    assert widened.daily_completed_runs == internal_free.daily_completed_runs
    assert widened.filters.include_internal is True  # accepted and echoed


def test_platform_cost_read_alone_also_leaves_internal_accounts_out_of_today(tmp_path):
    """The figure /commercial publishes is the overview's, read alone; it
    must not widen today's slice under include_internal either."""
    analytics, events, _rollups, users = _fixture(tmp_path)
    for index, user_id in enumerate((1, 2)):  # user 2 is the admin account
        events.record_server_event(
            event_name="model_usage_recorded",
            user_id=user_id,
            source_event_id=f"resource:model_usage_recorded:today-{index}:0",
            source_record_type="resource",
            source_record_id=f"today-{index}",
            correlation_id=f"today-{index}",
            provider_id="openrouter",
            model_id="openai/gpt-5.5",
            billing_mode="platform_credits",
            outcome="succeeded",
            properties={"input_tokens": 10, "output_tokens": 5, "cost_micro_usd": 300_000},
            occurred_at=NOW - timedelta(minutes=5 + index),
        )
    service = AnalyticsQueryService(store=analytics, user_store=users)
    start = _midnight(NOW.date())

    def cost(include_internal):
        return service.get_platform_model_cost_micro(
            filters=AnalyticsMetricFilters(start=start, end=NOW, include_internal=include_internal),
            now=NOW,
        )

    assert cost(False) == 300_000
    assert cost(True) == cost(False)


def test_profile_milestones_are_first_occurrences_not_the_newest_page(tmp_path):
    """Review #8: min() over the newest-100 timeline page dropped a first
    verification older than the user's last hundred events."""
    analytics, events, _rollups, users = _fixture(tmp_path)
    first_verified = NOW - timedelta(days=40)
    events.record_server_event(
        event_name="credential_verified",
        user_id=1,
        source_event_id="credential:c-1:verified:first",
        source_record_type="credential",
        source_record_id="c-1",
        occurred_at=first_verified,
    )
    events.record_server_event(
        event_name="credential_verified",
        user_id=1,
        source_event_id="credential:c-1:verified:again",
        source_record_type="credential",
        source_record_id="c-1",
        occurred_at=NOW - timedelta(days=2),
    )
    for index in range(101):
        _event(
            events,
            "backtest_requested",
            NOW - timedelta(hours=1, minutes=index),
            f"run:backtest_requested:page-{index}",
            source_record_id=f"page-{index}",
        )
    from dashboard.backend.domain.analytics.value_repository import ValueAnalyticsStore
    from dashboard.backend.tests.domain.analytics.test_value_repository import (
        SyntheticAgentStore, SyntheticCreditsStore, SyntheticProviderStore, SyntheticRunStore,
    )

    value_store = ValueAnalyticsStore(
        analytics, SyntheticCreditsStore(tmp_path / "credits.db"),
        provider_base=SyntheticProviderStore(
            {}, [{"provider_id": "openrouter", "status": "enabled", "platform_enabled": True}]
        ),
        agent_base=SyntheticAgentStore(), run_base=SyntheticRunStore({}),
    )
    service = AnalyticsQueryService(store=analytics, user_store=users, value_store=value_store)

    profile = service.get_user_profile(
        user_id=1, now=NOW, start=NOW - timedelta(days=90), end=NOW
    )

    assert profile.activation_milestones["credential_verified"] == first_verified
    assert "agent_created" not in profile.activation_milestones
    # The selected range bounds it, as on main.
    narrow = service.get_user_profile(
        user_id=1, now=NOW, start=NOW - timedelta(days=10), end=NOW
    )
    assert narrow.activation_milestones["credential_verified"] == NOW - timedelta(days=2)


def test_a_session_crossing_the_30_day_cutoff_is_read_whole(tmp_path):
    """Review #13: the window selects sessions, not events. A session that
    began before the cutoff and continued after it keeps all its events; a
    session wholly before the cutoff stays out (the design's 30-day window)."""
    analytics, events, _rollups, users = _fixture(tmp_path)
    cutoff = NOW - timedelta(days=30)
    context = RequestAnalyticsContext(
        country_code="US", device_category="desktop", browser_family="Chrome"
    )
    straddling, older = str(uuid4()), str(uuid4())
    for session_id, name, at in (
        (straddling, "page_viewed", cutoff - timedelta(minutes=5)),
        (straddling, "session_heartbeat", cutoff + timedelta(minutes=5)),
        (older, "page_viewed", cutoff - timedelta(days=1)),
    ):
        events.accept_frontend_event(
            user={"id": 1},
            payload=FrontendAnalyticsEvent(
                event_id=str(uuid4()),
                schema_version=1,
                event_name=name,
                session_id=session_id,
                occurred_at=at,
                page_view="agents",
                properties=({} if name == "page_viewed" else {"visible_ms": 500}),
            ),
            context=context,
            received_at=at + timedelta(minutes=1),
        )
    service = AnalyticsQueryService(store=analytics, user_store=users)

    page = service.get_user_activity(
        user_id=1, section="sessions", limit=10, cursor=None, now=NOW
    )

    assert [item.session_event_count for item in page.items] == [2]
    assert page.items[0].visible_ms == 500


def test_overview_marks_fact_panels_unavailable_when_fact_reads_fail(tmp_path, capsys):
    analytics, _events, _rollups, users = _fixture(tmp_path)

    class BrokenFacts:
        def count_states_for_date(self, _day):
            raise AssertionError("friction availability no longer probes the facts")

        def latest_fact_date(self, *, on_or_before):
            return on_or_before

        def list_attention_candidates(self, **_kwargs):
            raise RuntimeError("private detail")

    service = AnalyticsQueryService(
        store=analytics, user_store=users, value_store=BrokenFacts()
    )

    overview = service.get_overview(
        now=NOW,
        filters=AnalyticsMetricFilters(start=NOW - timedelta(days=1), end=NOW),
    )

    assert overview.users_needing_attention == []
    # M1: friction is served from the funnel's reads, so it follows them.
    assert overview.availability["friction"].available is True
    assert overview.availability["attention"].available is False
    assert overview.availability["growth"].available is True
    # M2: the swallowed failure reaches the log, by class name only.
    out = capsys.readouterr().out
    assert "WARNING: analytics.read.overview_attention_failed category=RuntimeError" in out
    assert "private detail" not in out


def test_query_service_builds_its_value_store_only_on_first_use(tmp_path, monkeypatch):
    """The module-level service is built at import; an eager default would pull
    the credits / provider / agent / run singletons onto the boot path."""

    from dashboard.backend.domain.analytics import query_service as module

    analytics, _events, _rollups, users = _fixture(tmp_path)
    built = []
    real = module.build_value_analytics_store

    def recording(**kwargs):
        built.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(module, "build_value_analytics_store", recording)
    service = AnalyticsQueryService(store=analytics, user_store=users)
    assert built == []

    first = service.value_store
    assert service.value_store is first
    assert built == [{"analytics_base": analytics}]


@pytest.fixture
def admin_analytics_api(monkeypatch):
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "admin-analytics.db"
        users = UserStore(path)
        admin = users.create_user(
            "admin@example.test", "Analytics Admin", "SecurePass1!"
        )
        users.apply_admin_patch(admin["id"], role="admin")
        outsider = users.create_user(
            "outsider@example.test", "Outsider", "SecurePass1!"
        )
        subject = users.create_user(
            "subject@example.test", "Subject", "SecurePass1!"
        )
        analytics = AnalyticsStore(path)
        event_service = AnalyticsService(analytics)
        at = datetime.now(timezone.utc).replace(microsecond=0)
        event_service.record_server_event(
            event_name="account_signed_up",
            user_id=subject["id"],
            source_event_id=f"account:account_signed_up:{subject['id']}",
            source_record_type="user",
            source_record_id=str(subject["id"]),
            occurred_at=at - timedelta(days=10),
        )
        for index in range(3):
            event_service.record_server_event(
                event_name="backtest_completed",
                user_id=subject["id"],
                source_event_id=f"run:backtest_completed:api-run-{index}",
                source_record_type="run",
                source_record_id=f"api-run-{index}",
                correlation_id=f"api-run-{index}",
                outcome="succeeded",
                occurred_at=at - timedelta(minutes=index + 1),
            )
        query_service = AnalyticsQueryService(store=analytics, user_store=users)
        value_query_service = FixtureValueQueryService(int(subject["id"]))
        monkeypatch.setattr(users_module, "user_store", users)
        app.dependency_overrides[get_analytics_query_service] = lambda: query_service
        app.dependency_overrides[get_value_analytics_query_service] = (
            lambda: value_query_service
        )
        app.dependency_overrides[get_analytics_service] = lambda: event_service
        admin_token = users.create_session(admin["id"])
        outsider_token = users.create_session(outsider["id"])
        with TestClient(app) as client:
            yield {
                "client": client,
                "analytics": analytics,
                "event_service": event_service,
                "query_service": query_service,
                "value_query_service": value_query_service,
                "admin": admin,
                "subject": subject,
                "admin_headers": {"Authorization": f"Bearer {admin_token}"},
                "outsider_headers": {"Authorization": f"Bearer {outsider_token}"},
            }
        app.dependency_overrides.pop(get_analytics_query_service, None)
        app.dependency_overrides.pop(get_value_analytics_query_service, None)
        app.dependency_overrides.pop(get_analytics_service, None)


def test_non_admin_cannot_query_any_admin_analytics_route(admin_analytics_api):
    api = admin_analytics_api
    subject_id = api["subject"]["id"]
    calls = [
        ("/api/admin/analytics/overview", {}),
        ("/api/admin/analytics/lifecycle", {}),
        ("/api/admin/analytics/retention", {}),
        ("/api/admin/analytics/commercial", {}),
        ("/api/admin/analytics/operational", {}),
        ("/api/admin/analytics/groups", {}),
        ("/api/admin/analytics/users", {}),
        (f"/api/admin/analytics/users/{subject_id}", {}),
        (
            f"/api/admin/analytics/users/{subject_id}/activity",
            {"section": "runs"},
        ),
    ]

    for path, params in calls:
        response = api["client"].get(
            path,
            params=params,
            headers=api["outsider_headers"],
        )
        assert response.status_code == 403, (path, response.text)


@pytest.mark.parametrize(
    "section",
    ["lifecycle", "retention", "commercial", "operational"],
)
def test_admin_value_sections_have_independent_contracts(
    admin_analytics_api,
    section,
):
    api = admin_analytics_api
    params = {
        "from": "2026-08-01",
        "to": "2026-08-31",
        "include_internal": "false",
    }
    if section == "operational":
        params.update(
            {
                "billing_mode": "platform_credits",
                "provider": "provider_synthetic",
                "model": "model-synthetic-v1",
            }
        )

    response = api["client"].get(
        f"/api/admin/analytics/{section}",
        params=params,
        headers=api["admin_headers"],
    )

    assert response.status_code == 200, response.text
    name, call = api["value_query_service"].calls[-1]
    assert name == section
    assert call["start"] == date(2026, 8, 1)
    assert call["end"] == date(2026, 9, 1)
    assert call["include_internal"] is False
    if section == "operational":
        assert call["provider_id"] == "provider_synthetic"
        assert call["model_id"] == "model-synthetic-v1"
        assert call["billing_mode"] == "platform_credits"


def test_group_endpoint_propagates_filter_and_always_returns_six_rows(
    admin_analytics_api,
):
    api = admin_analytics_api
    response = api["client"].get(
        "/api/admin/analytics/groups",
        params={
            "from": "2026-08-01",
            "to": "2026-08-31",
            "user_group": "organic",
            "include_internal": "false",
        },
        headers=api["admin_headers"],
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["groups"]) == 6
    assert [row["group"] for row in body["groups"]] == list(USER_GROUPS)
    assert body["selected_user_group"] == "organic"
    name, call = api["value_query_service"].calls[-1]
    assert name == "groups"
    assert call["start"] == date(2026, 8, 1)
    assert call["end"] == date(2026, 9, 1)
    assert call["include_internal"] is False
    assert call["user_group"] == "organic"


def test_group_endpoint_rejects_unknown_group(admin_analytics_api):
    response = admin_analytics_api["client"].get(
        "/api/admin/analytics/groups",
        params={
            "from": "2026-09-01",
            "to": "2026-09-03",
            "user_group": "friends",
        },
        headers=admin_analytics_api["admin_headers"],
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid Analytics query."}


@pytest.mark.parametrize("movement_range", ["5d", "1w", "1m", "1y"])
def test_lifecycle_accepts_documented_movement_ranges(admin_analytics_api, movement_range):
    api = admin_analytics_api
    response = api["client"].get(
        "/api/admin/analytics/lifecycle",
        params={"from": "2026-08-01", "to": "2026-08-31", "movement_range": movement_range},
        headers=api["admin_headers"],
    )

    assert response.status_code == 200, response.text
    name, call = api["value_query_service"].calls[-1]
    assert name == "lifecycle"
    assert call["movement_range"] == movement_range


def test_lifecycle_rejects_unknown_movement_range(admin_analytics_api):
    response = admin_analytics_api["client"].get(
        "/api/admin/analytics/lifecycle",
        params={"movement_range": "2q"},
        headers=admin_analytics_api["admin_headers"],
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid Analytics query."}


def test_admin_overview_accepts_documented_filters(admin_analytics_api):
    api = admin_analytics_api
    response = api["client"].get(
        "/api/admin/analytics/overview",
        params={
            "from": "2026-08-01",
            "to": "2026-08-26",
            "billing_mode": "byok",
            "provider": "openrouter",
            "model": "openai/gpt-5.5",
            "include_internal": "false",
        },
        headers=api["admin_headers"],
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["filters"]["billing_mode"] == "byok"
    assert body["filters"]["provider_id"] == "openrouter"
    assert body["filters"]["model_id"] == "openai/gpt-5.5"


def test_profile_and_activity_reads_record_access_without_body(admin_analytics_api):
    api = admin_analytics_api
    subject_id = api["subject"]["id"]
    profile = api["client"].get(
        f"/api/admin/analytics/users/{subject_id}",
        params={"from": "2026-08-01", "to": "2026-08-31"},
        headers=api["admin_headers"],
    )
    activity = api["client"].get(
        f"/api/admin/analytics/users/{subject_id}/activity",
        params={"section": "runs", "limit": 2},
        headers=api["admin_headers"],
    )
    access = api["analytics"].list_admin_access(subject_id, limit=10)

    assert profile.status_code == 200, profile.text
    assert activity.status_code == 200, activity.text
    assert activity.json()["next_cursor"] is not None
    assert [row["section"] for row in access[:2]] == ["runs", "overview"]
    assert all("response" not in row for row in access)
    _name, call = api["value_query_service"].calls[-1]
    assert call["start"] == date(2026, 8, 1)
    assert call["end"] == date(2026, 9, 1)


def test_admin_analytics_rejects_invalid_queries_without_echo(admin_analytics_api):
    api = admin_analytics_api
    canary = "synthetic-secret-query-canary"
    response = api["client"].get(
        "/api/admin/analytics/overview",
        params={"provider": f"{canary}!"},
        headers=api["admin_headers"],
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid Analytics query."}
    assert canary not in response.text


def test_admin_user_list_accepts_documented_filters(admin_analytics_api):
    api = admin_analytics_api
    today = datetime.now(timezone.utc).date()
    response = api["client"].get(
        "/api/admin/analytics/users",
        params={
            "q": "Subject",
            "lifecycle_segment": "core",
            "operational_state": "healthy",
            "commercial_tier": "invested",
            "user_group": "partner",
            "activated": "true",
            "last_meaningful_activity_from": (today - timedelta(days=1)).isoformat(),
            "last_meaningful_activity_to": today.isoformat(),
            "priority": "false",
            "limit": "1",
            "offset": "0",
        },
        headers=api["admin_headers"],
    )

    assert response.status_code == 200, response.text
    assert response.json()["total"] == 1
    assert response.json()["items"][0]["user_id"] == api["subject"]["id"]
    name, call = api["value_query_service"].calls[-1]
    assert name == "users"
    assert call["limit"] == 1
    assert call["offset"] == 0
    filters = call["filters"]
    assert filters.lifecycle_segment == "core"
    assert filters.operational_state == "healthy"
    assert filters.commercial_tier == "invested"
    assert filters.user_group == "partner"
    assert filters.activated is True


def test_admin_analytics_maps_not_found_and_cursor_errors_safely(
    admin_analytics_api,
):
    api = admin_analytics_api
    canary = "synthetic-secret-cursor-canary"
    missing = api["client"].get(
        "/api/admin/analytics/users/999999",
        headers=api["admin_headers"],
    )
    invalid_cursor = api["client"].get(
        f"/api/admin/analytics/users/{api['subject']['id']}/activity",
        params={"section": "runs", "cursor": f"{canary}!"},
        headers=api["admin_headers"],
    )

    assert missing.status_code == 404
    assert missing.json() == {"detail": "Analytics user was not found."}
    assert invalid_cursor.status_code == 422
    assert invalid_cursor.json() == {"detail": "Invalid Analytics query."}
    assert canary not in invalid_cursor.text


def test_admin_analytics_maps_service_and_access_failures_safely(
    admin_analytics_api,
    monkeypatch,
):
    api = admin_analytics_api
    canary = "synthetic-secret-storage-canary"
    monkeypatch.setattr(
        api["query_service"],
        "get_overview",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError(canary)),
    )
    overview = api["client"].get(
        "/api/admin/analytics/overview",
        headers=api["admin_headers"],
    )

    monkeypatch.setattr(
        api["event_service"],
        "record_admin_profile_access",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError(canary)),
    )
    profile = api["client"].get(
        f"/api/admin/analytics/users/{api['subject']['id']}",
        headers=api["admin_headers"],
    )

    for response in (overview, profile):
        assert response.status_code == 503
        assert response.json() == {
            "detail": "Analytics is temporarily unavailable."
        }
        assert canary not in response.text


def test_value_section_failures_are_safe_and_independent(
    admin_analytics_api,
    monkeypatch,
):
    api = admin_analytics_api
    canary = "synthetic-secret-value-service-canary"
    monkeypatch.setattr(
        api["value_query_service"],
        "get_commercial",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError(canary)),
    )

    commercial = api["client"].get(
        "/api/admin/analytics/commercial",
        headers=api["admin_headers"],
    )
    lifecycle = api["client"].get(
        "/api/admin/analytics/lifecycle",
        headers=api["admin_headers"],
    )

    assert commercial.status_code == 503
    assert commercial.json() == {"detail": "Analytics is temporarily unavailable."}
    assert canary not in commercial.text
    assert lifecycle.status_code == 200


@pytest.mark.parametrize(
    "path,params",
    [
        ("/api/admin/analytics/lifecycle", {"unknown": "value"}),
        ("/api/admin/analytics/retention", [("from", "2026-08-01"), ("from", "2026-08-02")]),
        ("/api/admin/analytics/commercial", {"from": "2026-01-01", "to": "2026-08-01"}),
        ("/api/admin/analytics/operational", {"provider": "synthetic secret!"}),
        ("/api/admin/analytics/users", {"commercial_tier": "unsupported"}),
    ],
)
def test_value_routes_reject_unknown_duplicate_and_unsafe_queries(
    admin_analytics_api,
    path,
    params,
):
    api = admin_analytics_api
    response = api["client"].get(path, params=params, headers=api["admin_headers"])

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid Analytics query."}
    assert "synthetic secret" not in response.text


def test_overview_has_no_five_state_field_and_users_rejects_status(admin_analytics_api):
    """Design D20: the one permitted contract break before PR D."""
    api = admin_analytics_api

    overview = api["client"].get("/api/admin/analytics/overview", headers=api["admin_headers"])
    users = api["client"].get(
        "/api/admin/analytics/users", params={"status": "active"}, headers=api["admin_headers"]
    )

    assert overview.status_code == 200, overview.text
    assert "user_state_counts" not in overview.json()
    assert set(overview.json()["availability"]) == {"snapshot", "growth", "funnel", "friction", "attention"}
    assert users.status_code == 422
    assert users.json() == {"detail": "Invalid Analytics query."}
    assert "user_state_counts" not in AnalyticsOverview.model_fields
    assert "legacy_status" not in UserValueFilters.model_fields
