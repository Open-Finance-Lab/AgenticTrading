"""Set-based reads over user_daily_facts, lifecycle_transitions and user_activity.

Every read here is one statement over the whole population (or one page of
it). The SQLite cases run always; ``assert_fact_read_contract`` is also driven
by the Postgres twin from test_repository_postgres.py under @pg_only.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.value_repository import (
    MAX_USER_BATCH,
    LifecycleTransitionRow,
    UserDailyFact,
    UserPageQuery,
    ValueAnalyticsStore,
)
from dashboard.backend.users import UserStore


UTC = timezone.utc
NOW = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)
YESTERDAY = date(2026, 9, 14)


def fact(user_id: int, day: date, **overrides) -> UserDailyFact:
    values = dict(
        snapshot_date=day,
        user_id=user_id,
        lifecycle_segment="growing",
        lifecycle_reason_code="growing_activated_below_core_threshold",
        operational_state="healthy",
        operational_reason_code="no_supported_issue",
        tier="unpaid",
        user_group="unknown",
        active=True,
        runs_requested=1,
        runs_completed=1,
        runs_failed=0,
        runs_cancelled=0,
        operator_cost_micro=0,
        own_spend_micro=0,
        data_quality="complete",
        calculated_at=datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=UTC),
    )
    values.update(overrides)
    return UserDailyFact(**values)


def transition(user_id: int, day: date, from_segment: str, to_segment: str, **overrides):
    values = dict(
        user_id=user_id,
        snapshot_date=day,
        from_segment=from_segment,
        to_segment=to_segment,
        inactive_days=0,
        data_quality="complete",
        created_at=datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=UTC),
    )
    values.update(overrides)
    return LifecycleTransitionRow(**values)


def seed_population(store, create_user, *, admin_id: int) -> dict[str, int]:
    """Five users plus an admin. Returns the ids by role in the scenario.

    ``create_user(key, email, group) -> int`` makes one account on whichever
    dialect is under test: the SQLite fixture goes through ``UserStore``, the
    Postgres contract fixture (a stub ``users`` table with no default id)
    inserts explicit ids. Every assertion below counts only this population,
    so pre-seeded accounts on either tier contribute nothing.
    """

    ids = {}
    for key, email, group in (
        ("blocked", "blocked@example.test", "partner"),
        ("attention", "attention@example.test", "partner"),
        ("core", "core@example.test", "organic"),
        ("dormant", "dormant@example.test", "unknown"),
        ("fresh", "fresh@example.test", "organic"),
    ):
        ids[key] = create_user(key, email, group)
    ids["admin"] = admin_id
    two_days_ago = YESTERDAY - timedelta(days=1)
    store.upsert_daily_facts(
        [
            fact(ids["blocked"], YESTERDAY, operational_state="blocked",
                 operational_reason_code="billing_lane_unavailable",
                 lifecycle_segment="at_risk", user_group="partner",
                 runs_requested=2, runs_completed=0, runs_failed=2, active=True),
            fact(ids["blocked"], two_days_ago, operational_state="blocked",
                 operational_reason_code="billing_lane_unavailable",
                 lifecycle_segment="growing", user_group="partner",
                 runs_requested=1, runs_completed=0, runs_failed=1),
            fact(ids["attention"], YESTERDAY, operational_state="needs_attention",
                 operational_reason_code="three_consecutive_failed_runs",
                 lifecycle_segment="growing", user_group="partner",
                 runs_requested=3, runs_completed=0, runs_failed=3, tier="starter"),
            fact(ids["core"], YESTERDAY, lifecycle_segment="core", user_group="organic",
                 runs_requested=4, runs_completed=4, operator_cost_micro=250_000,
                 own_spend_micro=100_000, tier="invested"),
            fact(ids["core"], two_days_ago, lifecycle_segment="growing", user_group="organic",
                 runs_requested=1, runs_completed=1, operator_cost_micro=50_000),
            fact(ids["dormant"], YESTERDAY, lifecycle_segment="dormant", active=False,
                 runs_requested=0, runs_completed=0, data_quality="partial",
                 operational_reason_code=None),
        ]
    )
    store.append_lifecycle_transitions(
        [
            transition(ids["blocked"], YESTERDAY, "growing", "at_risk", inactive_days=8),
            transition(ids["core"], YESTERDAY, "growing", "core"),
        ]
    )
    for key, activated_days_ago, active_days_ago in (
        ("blocked", 20, 9),
        ("core", 30, 1),
        ("attention", 3, 1),
        ("admin", 40, 1),
    ):
        store.record_activity(
            ids[key],
            occurred_at=NOW - timedelta(days=activated_days_ago),
            activating=True,
            now=NOW,
        )
        store.record_activity(
            ids[key],
            occurred_at=NOW - timedelta(days=active_days_ago),
            activating=False,
            now=NOW,
        )
    return ids


def assert_fact_read_contract(store, ids: dict[str, int]) -> None:
    two_days_ago = YESTERDAY - timedelta(days=1)

    # I5: the latest landed day at or before a bound, one MAX() read.
    assert store.latest_fact_date(on_or_before=YESTERDAY + timedelta(days=5)) == YESTERDAY
    assert store.latest_fact_date(on_or_before=YESTERDAY) == YESTERDAY
    assert store.latest_fact_date(on_or_before=two_days_ago) == two_days_ago
    assert store.latest_fact_date(on_or_before=two_days_ago - timedelta(days=1)) is None

    states = store.count_states_for_date(YESTERDAY)
    by_state = {}
    by_reason = {}
    by_segment = {}
    for row in states:
        by_state[row.operational_state] = by_state.get(row.operational_state, 0) + row.users
        by_reason[row.operational_reason_code] = by_reason.get(row.operational_reason_code, 0) + row.users
        by_segment[row.lifecycle_segment] = by_segment.get(row.lifecycle_segment, 0) + row.users
    assert by_state == {"blocked": 1, "needs_attention": 1, "healthy": 2}
    assert by_reason["billing_lane_unavailable"] == 1
    assert by_segment == {"at_risk": 1, "growing": 1, "core": 1, "dormant": 1}
    assert any(row.partial_rows == 1 for row in states if row.lifecycle_segment == "dormant")
    # A NULL reason code (history-copy rows) is a value, not an error.
    assert None in by_reason

    segments = store.count_segments_by_date(start=two_days_ago, end=YESTERDAY + timedelta(days=1))
    assert {(row.snapshot_date, row.lifecycle_segment, row.users) for row in segments} == {
        (two_days_ago, "growing", 2),
        (YESTERDAY, "at_risk", 1),
        (YESTERDAY, "growing", 1),
        (YESTERDAY, "core", 1),
        (YESTERDAY, "dormant", 1),
    }

    transitions = store.count_transitions(start=YESTERDAY, end=YESTERDAY + timedelta(days=1))
    assert {(t.from_segment, t.to_segment, t.users) for t in transitions} == {
        ("growing", "at_risk", 1),
        ("growing", "core", 1),
    }

    attention = store.list_attention_candidates(
        day=YESTERDAY, window_start=YESTERDAY - timedelta(days=29), limit=10
    )
    assert [c.user_id for c in attention] == [ids["attention"], ids["blocked"]]
    assert attention[0].recent_failures == 3
    assert attention[1].recent_failures == 3  # 2 yesterday + 1 the day before
    assert attention[1].recent_runs == 3
    assert attention[1].operational_reason_code == "billing_lane_unavailable"
    assert attention[0].email == "attention@example.test"
    assert store.list_attention_candidates(
        day=YESTERDAY, window_start=YESTERDAY - timedelta(days=29), limit=1
    )[0].user_id == ids["attention"]

    groups = store.sum_facts_by_user_group(start=two_days_ago, end=YESTERDAY + timedelta(days=1))
    assert groups["organic"].total_runs == 5
    assert groups["organic"].operator_cost_micro == 300_000
    assert groups["organic"].successful_run_users == 1
    assert groups["organic"].repeat_users == 1  # core completed on two days
    assert groups["partner"].total_runs == 6
    assert groups["partner"].successful_run_users == 0
    assert groups["partner"].repeat_users == 0
    assert "internal" not in groups
    # The dormant user's partial row is the window's only one (I2).
    assert groups["unknown"].partial_rows == 1
    assert groups["organic"].partial_rows == 0
    assert groups["partner"].partial_rows == 0

    active = store.list_active_dates(
        [ids["core"], ids["dormant"], ids["blocked"]],
        start=two_days_ago,
        end=YESTERDAY + timedelta(days=1),
    )
    assert active[ids["core"]] == [two_days_ago, YESTERDAY]
    assert active[ids["dormant"]] == []
    assert active[ids["blocked"]] == [two_days_ago, YESTERDAY]

    activations = store.list_activations(
        start=NOW - timedelta(days=35), end=NOW, include_internal=False
    )
    assert set(activations) == {ids["blocked"], ids["core"], ids["attention"]}
    assert activations[ids["core"]] == NOW - timedelta(days=30)
    with_admin = store.list_activations(
        start=NOW - timedelta(days=45), end=NOW, include_internal=True
    )
    assert ids["admin"] in with_admin

    assert store.count_activated_users(include_internal=False) == 3
    assert store.count_activated_users(include_internal=True) == 4

    mine = store.list_user_transitions(
        ids["blocked"], start=two_days_ago, end=YESTERDAY + timedelta(days=1)
    )
    assert [(t.from_segment, t.to_segment) for t in mine] == [("growing", "at_risk")]
    assert store.list_user_transitions(
        ids["blocked"], start=two_days_ago, end=YESTERDAY
    ) == []

    for_day = store.list_transitions_for_date(YESTERDAY)
    assert {t.user_id for t in for_day} == {ids["blocked"], ids["core"]}
    assert store.list_transitions_for_date(two_days_ago) == []


def _page_query(**overrides) -> UserPageQuery:
    values = dict(
        q=None,
        lifecycle_segment=None,
        operational_state=None,
        commercial_tier=None,
        user_group=None,
        activated=None,
        last_meaningful_activity_from=None,
        last_meaningful_activity_to=None,
        priority=False,
        include_internal=False,
    )
    values.update(overrides)
    return UserPageQuery(**values)


def assert_users_page_contract(store, ids: dict[str, int]) -> None:
    rows, total = store.list_users_page(
        query=_page_query(), day=YESTERDAY, limit=100, offset=0
    )
    assert total == 5
    assert [row.user_id for row in rows] == sorted(
        ids[key] for key in ("blocked", "attention", "core", "dormant", "fresh")
    )
    fresh = next(row for row in rows if row.user_id == ids["fresh"])
    assert fresh.lifecycle_segment is None
    assert fresh.operational_state == "healthy"
    assert fresh.tier == "unpaid"
    assert fresh.activated_at is None
    # A NULL stored reason code (history-copy row) reaches the service as healthy.
    dormant = next(row for row in rows if row.user_id == ids["dormant"])
    assert dormant.operational_reason_code == "no_supported_issue"

    rows, total = store.list_users_page(
        query=_page_query(include_internal=True), day=YESTERDAY, limit=100, offset=0
    )
    assert total == 6
    assert ids["admin"] in {row.user_id for row in rows}

    rows, total = store.list_users_page(
        query=_page_query(), day=YESTERDAY, limit=2, offset=2
    )
    assert total == 5
    every_id = sorted(
        ids[key] for key in ("blocked", "attention", "core", "dormant", "fresh")
    )
    assert [row.user_id for row in rows] == every_id[2:4]

    rows, _total = store.list_users_page(
        query=_page_query(priority=True), day=YESTERDAY, limit=100, offset=0
    )
    assert [row.user_id for row in rows] == [ids["blocked"], ids["attention"]]

    rows, total = store.list_users_page(
        query=_page_query(user_group="partner", commercial_tier="starter"),
        day=YESTERDAY, limit=100, offset=0,
    )
    assert total == 1 and rows[0].user_id == ids["attention"]

    rows, total = store.list_users_page(
        query=_page_query(operational_state="blocked"), day=YESTERDAY, limit=100, offset=0
    )
    assert [row.user_id for row in rows] == [ids["blocked"]]
    assert rows[0].operational_reason_code == "billing_lane_unavailable"

    rows, total = store.list_users_page(
        query=_page_query(lifecycle_segment="core"), day=YESTERDAY, limit=100, offset=0
    )
    assert [row.user_id for row in rows] == [ids["core"]]

    rows, total = store.list_users_page(
        query=_page_query(activated=False), day=YESTERDAY, limit=100, offset=0
    )
    assert {row.user_id for row in rows} == {ids["dormant"], ids["fresh"]}

    rows, total = store.list_users_page(
        query=_page_query(
            last_meaningful_activity_from=NOW - timedelta(days=2),
            last_meaningful_activity_to=NOW,
        ),
        day=YESTERDAY, limit=100, offset=0,
    )
    assert {row.user_id for row in rows} == {ids["core"], ids["attention"]}

    rows, total = store.list_users_page(
        query=_page_query(q="ATTEN"), day=YESTERDAY, limit=100, offset=0
    )
    assert [row.user_id for row in rows] == [ids["attention"]]
    rows, total = store.list_users_page(
        query=_page_query(q="%"), day=YESTERDAY, limit=100, offset=0
    )
    assert total == 0  # escaped: no literal percent sign in any email or name


def assert_priority_sort_contract(store, create_user) -> None:
    """The priority order, pinned key by key: group, tier, inactivity, id.

    Users are created in an order that differs from the expected one, and the
    lower groups carry the better tier, so no key passes by accident. Only
    these users' relative order is asserted, which keeps the check valid on a
    store that already holds the contract population.
    """

    #   key, fact overrides, days since last meaningful activity (None = no row)
    specs = [
        ("pr_onboard_high", dict(lifecycle_segment="onboarding", tier="high_value"), 1),
        ("pr_at_risk", dict(lifecycle_segment="at_risk"), 1),
        ("pr_attention", dict(operational_state="needs_attention", lifecycle_segment="at_risk",
                              operational_reason_code="three_consecutive_failed_runs"), 1),
        ("pr_blocked_recent", dict(operational_state="blocked",
                                   operational_reason_code="billing_lane_unavailable"), 1),
        ("pr_blocked_tie_a", dict(operational_state="blocked",
                                  operational_reason_code="billing_lane_unavailable"), 5),
        ("pr_blocked_tie_b", dict(operational_state="blocked",
                                  operational_reason_code="billing_lane_unavailable"), 5),
        ("pr_blocked_old", dict(operational_state="blocked",
                                operational_reason_code="billing_lane_unavailable"), 20),
        ("pr_blocked_never", dict(operational_state="blocked",
                                  operational_reason_code="billing_lane_unavailable"), None),
        ("pr_blocked_starter", dict(operational_state="blocked", tier="starter",
                                    operational_reason_code="billing_lane_unavailable"), 1),
        ("pr_blocked_high", dict(operational_state="blocked", tier="high_value",
                                 operational_reason_code="billing_lane_unavailable"), 1),
    ]
    ids = {}
    for key, overrides, idle_days in specs:
        user_id = create_user(key, f"{key}@example.test", "unknown")
        ids[key] = user_id
        store.upsert_daily_facts([fact(user_id, YESTERDAY, **overrides)])
        if idle_days is not None:
            store.record_activity(
                user_id, occurred_at=NOW - timedelta(days=idle_days), activating=False, now=NOW
            )
    assert ids["pr_blocked_tie_a"] < ids["pr_blocked_tie_b"]

    rows, _total = store.list_users_page(
        query=_page_query(priority=True), day=YESTERDAY, limit=100, offset=0
    )

    mine = {user_id: key for key, user_id in ids.items()}
    assert [mine[row.user_id] for row in rows if row.user_id in mine] == [
        "pr_blocked_high",      # group blocked, tier high_value
        "pr_blocked_starter",   # tier starter
        "pr_blocked_never",     # unpaid; no activity row sorts first
        "pr_blocked_old",       # then the longest idle
        "pr_blocked_tie_a",     # equal idle time falls to the lower id
        "pr_blocked_tie_b",
        "pr_blocked_recent",
        "pr_attention",         # group needs_attention, though its segment is at_risk
        "pr_at_risk",
        "pr_onboard_high",      # last group, despite the best tier
    ]


def _sqlite_fixture(tmp_path):
    db_path = tmp_path / "fact-reads.db"
    users = UserStore(db_path=db_path)
    admin = users.create_user("admin@example.test", "Admin", "SecurePass1!")
    users.apply_admin_patch(int(admin["id"]), role="admin")
    analytics = AnalyticsStore(db_path=db_path)
    store = ValueAnalyticsStore(
        analytics,
        credits_base=object(),
        provider_base=object(),
        agent_base=object(),
        run_base=object(),
    )
    return store, users, int(admin["id"])


def _sqlite_user_factory(users):
    def create_user(key: str, email: str, group: str) -> int:
        user = users.create_user(email, key.title(), "SecurePass1!")
        users.apply_admin_patch(int(user["id"]), user_group=group)
        return int(user["id"])

    return create_user


def test_sqlite_fact_reads_follow_the_contract(tmp_path):
    store, users, admin_id = _sqlite_fixture(tmp_path)
    ids = seed_population(store, _sqlite_user_factory(users), admin_id=admin_id)
    assert_fact_read_contract(store, ids)


def test_fact_reads_are_one_statement_per_call(tmp_path):
    """No read here may issue a per-user statement (design §6.11 rule 6)."""

    store, users, admin_id = _sqlite_fixture(tmp_path)
    ids = seed_population(store, _sqlite_user_factory(users), admin_id=admin_id)
    statements: list[str] = []
    original = store.analytics_base._get_connection

    from contextlib import contextmanager

    @contextmanager
    def tracing_connection():
        with original() as conn:
            conn.set_trace_callback(
                lambda statement: statements.append(statement)
                if statement.lstrip().upper().startswith("SELECT")
                else None
            )
            yield conn

    store.analytics_base._get_connection = tracing_connection
    store.count_states_for_date(YESTERDAY)
    store.count_segments_by_date(start=YESTERDAY - timedelta(days=30), end=YESTERDAY + timedelta(days=1))
    store.count_transitions(start=YESTERDAY - timedelta(days=30), end=YESTERDAY + timedelta(days=1))
    store.list_attention_candidates(day=YESTERDAY, window_start=YESTERDAY - timedelta(days=29), limit=10)
    store.list_active_dates(list(ids.values()), start=YESTERDAY - timedelta(days=30), end=YESTERDAY + timedelta(days=1))
    store.list_activations(start=NOW - timedelta(days=60), end=NOW, include_internal=False)
    store.count_activated_users(include_internal=False)
    store.list_user_transitions(ids["core"], start=YESTERDAY - timedelta(days=30), end=YESTERDAY + timedelta(days=1))
    store.list_transitions_for_date(YESTERDAY)
    assert len(statements) == 9
    statements.clear()
    store.sum_facts_by_user_group(start=YESTERDAY - timedelta(days=30), end=YESTERDAY + timedelta(days=1))
    assert len(statements) == 2


def test_list_active_dates_spans_more_than_one_id_batch(tmp_path):
    """Retention cohorts can exceed MAX_USER_BATCH; the read chunks, not raises."""

    store, users, admin_id = _sqlite_fixture(tmp_path)
    ids = seed_population(store, _sqlite_user_factory(users), admin_id=admin_id)
    padding = list(range(10_000, 10_000 + MAX_USER_BATCH + 20))
    window = dict(start=YESTERDAY - timedelta(days=1), end=YESTERDAY + timedelta(days=1))

    result = store.list_active_dates([*padding, ids["core"]], **window)

    assert len(result) == len(padding) + 1
    assert result[ids["core"]] == [YESTERDAY - timedelta(days=1), YESTERDAY]
    assert all(result[user_id] == [] for user_id in padding)
    assert store.list_active_dates([], **window) == {}


def test_fact_reads_reject_malformed_windows(tmp_path):
    store, _users, _admin_id = _sqlite_fixture(tmp_path)
    with pytest.raises(ValueError):
        store.count_states_for_date(datetime(2026, 9, 14, tzinfo=UTC))
    with pytest.raises(ValueError):
        store.count_segments_by_date(start=YESTERDAY, end=YESTERDAY)
    with pytest.raises(ValueError):
        store.count_transitions(start=YESTERDAY, end=YESTERDAY - timedelta(days=1))
    with pytest.raises(ValueError):
        store.list_attention_candidates(
            day=YESTERDAY, window_start=YESTERDAY + timedelta(days=1), limit=5
        )
    with pytest.raises(ValueError):
        store.list_attention_candidates(day=YESTERDAY, window_start=YESTERDAY, limit=0)
    with pytest.raises(ValueError):
        store.list_user_transitions(0, start=YESTERDAY, end=YESTERDAY + timedelta(days=1))
    with pytest.raises(ValueError):
        store.list_transitions_for_date("2026-09-14")


def test_sqlite_users_page_follows_the_contract(tmp_path):
    store, users, admin_id = _sqlite_fixture(tmp_path)
    ids = seed_population(store, _sqlite_user_factory(users), admin_id=admin_id)
    assert_users_page_contract(store, ids)


def test_sqlite_users_page_priority_sort_follows_the_contract(tmp_path):
    store, users, _admin_id = _sqlite_fixture(tmp_path)
    assert_priority_sort_contract(store, _sqlite_user_factory(users))
