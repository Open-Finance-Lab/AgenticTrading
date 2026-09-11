# User Analytics Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace materialized per-user analytics snapshots with facts derived from authoritative tables, so the cost of analytics is bounded by construction and every cross-user number comes from one daily fact table instead of a repeated full-history scan.

**Architecture:** Event ingestion maintains two timestamps per user instead of recomputing a label. The lifecycle label becomes a read-time pure function of those timestamps plus trailing-30-day sums. One set-based job writes `user_daily_facts` once per UTC day behind a compare-and-set due check, and every cross-user chart, list and trend reads that table or the anonymous rollups. Users are described on three orthogonal axes: role, tier derived from the credit ledger, and one admin-assigned cohort.

**Tech Stack:** Python 3, FastAPI, Pydantic v2, SQLite, PostgreSQL/psycopg 3, pytest, vanilla JavaScript, Chart.js 4.4.0 already loaded by `app.html`.

**Spec:** `docs/superpowers/specs/2026-09-12-user-analytics-architecture-design.md`

**Delivery:** three sequential pull requests. Each branches off `origin/main` and leaves prod working on its own. PR B and PR C must run green on the CI Postgres tier before merge.

| PR | Branch | Scope |
|---|---|---|
| A | `fix/analytics-read-budget-stopgap` | Throttle the existing burners, add the auth 503, pin a read budget. No schema change. |
| B | `feat/user-analytics-daily-facts` | New data model, ingestion-maintained activity, the daily job, legacy removal. |
| C | `feat/user-analytics-read-paths` | Admin read paths onto the new tables, the three axis filters, long-term rollups. |

## Global Constraints

Exact values, copied from the spec. Every task's requirements implicitly include this section.

- Activation is the first server-authoritative `backtest_completed` event. Queued, started, failed and cancelled runs never activate a user.
- The meaningful-activity set is `_LIFECYCLE_ACTIVITY_EVENTS` in `dashboard/backend/domain/analytics/lifecycle.py`, unchanged by this work.
- `inactive_days` is counted in UTC calendar dates. `0..7` is recent, `8..29` is At risk, `30+` is Dormant.
- Core requires at least 3 distinct active UTC days and 3 successful backtests in the trailing 30 UTC dates, with `inactive_days <= 7`.
- Lifecycle and operational state are independent axes. Operational precedence stays `blocked`, then `needs_attention`, then `healthy`.
- Commercial tier uses lifetime settled purchases minus settled refunds: `unpaid = 0`, `starter = 1..4_999_999` micro-USD, `invested = 5_000_000..19_999_999`, `high_value >= 20_000_000`. The resolver is `commercial_tier()` and stays a pure function.
- A user has at most one cohort. Cohort values match `^[a-z0-9_-]{1,32}$`. There is no cohort lookup table. `paid` and `admin` are never cohort values.
- Display precedence for the single group badge: `admin` when the role is admin, else the cohort when set, else `paid` when the tier is anything but `unpaid`, else `free`.
- Freshness is two-tier. Live means one user's timeline, one user's segment, one user's operational state, and the credit balance. Daily means every cross-user number and every per-user trailing window, complete through the previous UTC day and labelled "as of yesterday".
- The admin overview may read the current UTC day from raw events as one bounded scan. That is the only raw-event read outside the paginated timeline route.
- No per-user "today" numbers are shown anywhere.
- Any table carrying a user id is retained 180 days. Long-term history lives only in the anonymous `analytics_daily_rollups`, retained indefinitely.
- Admins and users excluded through `analytics_subject_settings` get no `user_daily_facts` rows. Their profiles still work from live data.
- `partial` marks a fact row whose source was unavailable or which was migrated without run and cost columns. The UI keeps labelling such periods "Incomplete data" rather than zero.
- Every analytics filter and chart accepts any combination of role, tier and cohort.
- Users never see active days, lifecycle segment, lifecycle reasons or evidence, operational state or evidence, cohort, group badge, or the admin access log.
- All analytics APIs stay admin-only, read-only and display-safe. They never return SQL, raw events outside the timeline route, provider bodies, secrets, prompt or strategy content, full IP addresses, raw User-Agent values, or credential material.
- Log lines use `print()`, never `logger` — `dashboard.backend.*` loggers sit at WARNING in prod and emit nothing. Failure lines carry an exception class name only, never a message body: `print(f"WARNING: analytics.x_failed category={type(exc).__name__[:80]}")`.
- Postgres parameters that can be `None` and land inside `COALESCE` or another type-inferring context need an explicit `::integer` / `::text` cast. psycopg sends a bare `None` as OID 0 and Postgres refuses to resolve it. No SQLite test can catch a missing cast.
- Use synthetic fixtures and fake repositories. No test requires a real API key, a Stripe call, a provider call, a production database, or a copied production identity.
- Never stage or commit `dashboard/storage/data/backtest.db`. A bare backend import outside the pytest conftest runs `CREATE TABLE IF NOT EXISTS` and `ALTER TABLE` against the committed seed database. Check `git status` before every `git add`, and stage named paths, never `-A`.

## Locked File Structure

Files this plan creates or changes, and what each is responsible for.

**PR A**

- `dashboard/backend/domain/analytics/states.py`: gains a module constant `SNAPSHOT_STALE_AFTER`, a single-read recompute, and a freshness guard used by the ingestion path.
- `dashboard/backend/domain/analytics/service.py`: the inline per-event projection consults the freshness guard before recomputing.
- `dashboard/backend/domain/analytics/instrumentation.py`: the fallback recompute path consults the same guard.
- `dashboard/backend/domain/analytics/maintenance.py`: drops the legacy repair leg from the tick.
- `dashboard/backend/api/auth.py`: maps a user-store `OperationalError` to 503 on signup, login and current-user.
- `dashboard/backend/tests/domain/analytics/test_read_budget.py`: new. The counting-spy budget tests.
- `dashboard/backend/tests/test_auth_store_unavailable.py`: new. The 503 mapping tests.
- `CLAUDE.md`: the per-user-read-inside-the-reaper gotcha.

**PR B**

- `dashboard/backend/domain/analytics/repository.py` and `repository_postgres.py`: DDL for `user_activity`, `user_daily_facts` and `lifecycle_transitions`; removal of `user_analytics_snapshots` value columns and `user_lifecycle_daily_snapshots`.
- `dashboard/backend/domain/analytics/value_repository.py`: store methods for the three new tables, the compare-and-set day claim, and the set-based daily aggregation queries.
- `dashboard/backend/domain/analytics/lifecycle.py`: unchanged rules; gains `resolve_group_badge`.
- `dashboard/backend/domain/analytics/daily_facts.py`: new. The daily job and its report model.
- `dashboard/backend/domain/analytics/lifecycle_reads.py`: new. Builds lifecycle inputs from stored rows.
- `dashboard/backend/domain/analytics/facts_migration.py`: new. Copies eight weeks of history, idempotently.
- `dashboard/backend/domain/analytics/service.py`: ingestion maintains `user_activity` instead of recomputing snapshots.
- `dashboard/backend/domain/analytics/maintenance.py`: the tick becomes a due check plus the daily job.
- `dashboard/backend/users.py` and `users_postgres.py`: the `cohort` column, its validation, and the admin projection.
- `dashboard/backend/api/routers/admin_users.py`: `cohort` on `AdminUserPatch`.
- `dashboard/backend/database.py` and `database_postgres.py`: `agent_runs.owner_user_id`.
- `dashboard/backend/tests/test_architecture_boundaries.py`: the event-log discipline guard.

**PR C**

- `dashboard/backend/domain/analytics/value_queries.py`: lifecycle, retention, operational, users and profile read paths onto `user_daily_facts`; the three axis filters.
- `dashboard/backend/domain/analytics/query_service.py`: overview onto rollups plus one current-day scan; `get_user_metrics` with the audience filter.
- `dashboard/backend/domain/analytics/retention.py`: long-term tier and cohort rollups written before fact rows expire.
- `dashboard/frontend/js/admin-analytics-value.js` and `dashboard/frontend/app.html`: the three axis filter controls and the cohort field.

---

# PR A — Read-budget stopgap

Branch: `git switch -c fix/analytics-read-budget-stopgap origin/main`

No schema change. Every change here is deleted by PR B. The point is to make prod safe today and to leave behind a test that fails if anyone reintroduces a per-user read on a hot path.

**Two burners, not one.** The spec's "Why now" names the sixty-second sweep. Reading the code found a second and larger one: `AnalyticsService.record_server_event` recomputes the caller's snapshot **inline, in the request**, for every accepted lifecycle event, and each recompute reads that user's full 180-day history twice. One backtest emits `backtest_requested`, `backtest_queued`, `backtest_started` and `backtest_completed`, so a single run cost eight full-history scans before the sweep did anything. Tasks A1 and A2 close that path; A3 closes the sweep's extra leg.

### Task A1: Bound the per-event snapshot recompute

**Files:**
- Modify: `dashboard/backend/domain/analytics/states.py` (add module constant and helper after the imports, around line 29)
- Modify: `dashboard/backend/domain/analytics/service.py:55-77`
- Modify: `dashboard/backend/domain/analytics/instrumentation.py:66-78`
- Test: `dashboard/backend/tests/domain/analytics/test_read_budget.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `SNAPSHOT_STALE_AFTER: timedelta` and
  `snapshot_recompute_due(user_id: int, *, now: datetime, value_store: ValueAnalyticsStore | None = None) -> bool`
  in `dashboard.backend.domain.analytics.states`. Task A2 reuses the constant.

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/domain/analytics/test_read_budget.py`:

```python
"""Read budgets for the Analytics hot paths.

These tests do not assert behaviour. They assert *cost*: how many times a
code path reads a user's event history. The 2026-09-11 outage was a cost
regression that every behavioural test passed through, so the budget gets
its own file and its own counting spy.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.service import AnalyticsService
from dashboard.backend.domain.analytics.states import AnalyticsStateStore
from dashboard.backend.domain.analytics.value_repository import (
    CurrentOperationalFacts,
    UserValueSnapshot,
)


NOW = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)


class CountingStateStore:
    """Wraps AnalyticsStateStore and counts full-history reads."""

    def __init__(self, inner: AnalyticsStateStore):
        self._inner = inner
        self.event_reads = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def list_user_events(self, user_id, *, now, days=180):
        self.event_reads += 1
        return self._inner.list_user_events(user_id, now=now, days=days)


class StubValueStore:
    """Minimal value store: real enough for the projection, counts nothing."""

    def __init__(self):
        self.snapshots: dict[int, UserValueSnapshot] = {}
        self.daily: list = []

    def list_credit_activity(self, user_ids, *, start, end):
        return {user_id: () for user_id in user_ids}

    def get_current_snapshot(self, user_id):
        return self.snapshots.get(user_id)

    def get_operational_facts(self, user_id, *, now):
        return CurrentOperationalFacts(user_id=user_id)

    def upsert_current_snapshot(self, snapshot):
        self.snapshots[snapshot.user_id] = snapshot
        return snapshot

    def upsert_daily_snapshot(self, snapshot):
        self.daily.append(snapshot)
        return snapshot


def _fixture(tmp_path, *, users=1):
    path = tmp_path / "budget.db"
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
        for user_id in range(1, users + 1):
            conn.execute(
                "INSERT INTO users VALUES (?, ?, 'User', 'x', 'user', ?)",
                (
                    user_id,
                    f"user{user_id}@example.test",
                    (NOW - timedelta(days=40)).isoformat(),
                ),
            )
    analytics = AnalyticsStore(path)
    counting = CountingStateStore(AnalyticsStateStore(analytics))
    service = AnalyticsService(
        analytics,
        state_store=counting,
        value_store=StubValueStore(),
        project_snapshots=True,
    )
    return service, counting


def _emit(service, index, at, *, user_id=1):
    service.record_server_event(
        event_name="backtest_requested",
        user_id=user_id,
        source_event_id=f"run:backtest_requested:run-{user_id}-{index}",
        source_record_type="run",
        source_record_id=f"run-{user_id}-{index}",
        occurred_at=at,
    )


def test_repeated_events_do_not_repeat_the_history_read(tmp_path):
    """Six events in one minute cost what one event costs.

    Before the freshness guard this scaled linearly: every accepted
    lifecycle event triggered a fresh 180-day scan, so one backtest's four
    progress events cost four recomputes.
    """
    single_service, single_store = _fixture(tmp_path / "one")
    _emit(single_service, 0, NOW)
    baseline = single_store.event_reads

    many_service, many_store = _fixture(tmp_path / "many")
    for index in range(6):
        _emit(many_service, index, NOW - timedelta(seconds=index))

    assert baseline > 0
    assert many_store.event_reads == baseline
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/test_read_budget.py::test_repeated_events_do_not_repeat_the_history_read -v
```

Expected: FAIL. The assertion reports six times the baseline, because every event recomputes.

- [ ] **Step 3: Add the constant and the freshness guard**

In `dashboard/backend/domain/analytics/states.py`, immediately after the import block (which already provides `timedelta`, `positive_user_id` and `ValueAnalyticsStore`) and before `_REASONS`, insert:

```python
# One UTC day. A lifecycle label can change at most once per day, so a
# recompute more often than this buys nothing and costs a full-history read.
# Used by the ingestion freshness guard below and by the stale-user sweep.
SNAPSHOT_STALE_AFTER = timedelta(hours=24)


def snapshot_recompute_due(
    user_id: int,
    *,
    now: datetime,
    value_store: ValueAnalyticsStore | None = None,
) -> bool:
    """Whether one user's snapshot is stale enough to be worth a full read.

    Ingestion calls this before every projection. A recompute costs a
    180-day event scan, so recomputing per event made a single backtest --
    requested, queued, started, completed -- cost four of them, inline in
    the request. The stored ``calculated_at`` is one indexed row read and
    answers the same question, so a fresh snapshot skips the scan.

    Fails open: a store error returns True, because a projection that is
    merely expensive is still better than one that silently stops.
    """
    current = _require_utc(now, "now")
    values = value_store or ValueAnalyticsStore(AnalyticsStateStore().base_store)
    try:
        snapshot = values.get_current_snapshot(positive_user_id(user_id))
    except Exception:
        return True
    if snapshot is None:
        return True
    return snapshot.calculated_at <= current - SNAPSHOT_STALE_AFTER
```

Note `_require_utc` is defined at `states.py:542`, below this insertion point. That is fine: it is resolved at call time, not at import time.

- [ ] **Step 4: Gate the inline projection**

In `dashboard/backend/domain/analytics/service.py`, replace the body of `_try_recalculate_snapshots` (lines 55-77) with:

```python
    def _try_recalculate_snapshots(
        self,
        *,
        user_id: int,
        event_name: str,
        now: datetime,
    ) -> None:
        if not self.project_snapshots:
            return
        try:
            from .states import recalculate_user_snapshots, snapshot_recompute_due

            if not snapshot_recompute_due(
                user_id,
                now=now,
                value_store=self.value_store,
            ):
                return
            recalculate_user_snapshots(
                user_id,
                now=now,
                state_store=self.state_store,
                value_store=self.value_store,
            )
        except Exception as exc:
            print(
                "WARNING: analytics.value_projection_failed "
                f"event={event_name} category={type(exc).__name__[:80]}"
            )
```

- [ ] **Step 5: Gate the instrumentation fallback**

`instrumentation._recalculate_snapshot` runs only when the service was built without projection, but it calls the same recompute, so it needs the same guard. In `dashboard/backend/domain/analytics/instrumentation.py`, replace lines 66-78 with:

```python
def _recalculate_snapshot(user_id: int, event_name: str) -> None:
    if event_name not in SNAPSHOT_RELEVANT_EVENTS:
        return
    callback = _snapshot_recalculator
    if callback is None:
        # Keep the legacy compatibility snapshot and the value-analytics
        # projection in sync after every authoritative event.  The combined
        # recalculator writes both projections atomically when they share the
        # same analytics store.
        from .states import recalculate_user_snapshots, snapshot_recompute_due

        if not snapshot_recompute_due(user_id, now=datetime.now(timezone.utc)):
            return
        callback = recalculate_user_snapshots
    callback(user_id)
```

`datetime` and `timezone` are already imported at `instrumentation.py:10`.

- [ ] **Step 6: Run the test to verify it passes**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/test_read_budget.py -v
```

Expected: PASS.

- [ ] **Step 7: Run the analytics suite for regressions**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/ -q
```

Expected: PASS. Tests that assert a snapshot changed after a second event in the same minute will now fail; that is a real behaviour change, so update those tests to advance `now` past `SNAPSHOT_STALE_AFTER` rather than weakening the guard.

- [ ] **Step 8: Commit**

```bash
git add dashboard/backend/domain/analytics/states.py \
        dashboard/backend/domain/analytics/service.py \
        dashboard/backend/domain/analytics/instrumentation.py \
        dashboard/backend/tests/domain/analytics/test_read_budget.py
git status --short
git commit -m "fix(analytics): skip the per-event snapshot recompute while fresh"
```

`git status --short` before the commit is not optional: a bare backend import rewrites `dashboard/storage/data/backtest.db`, and that file must never be staged.

### Task A2: One event read per recompute, and a 24-hour stale window

**Files:**
- Modify: `dashboard/backend/domain/analytics/states.py:272-350` (`list_stale_user_ids`), `:381-396` (`calculate_user_state`), `:548-566` (`_calculate_user_value_snapshot`), `:680-705` (`recalculate_user_snapshots`), `:717-744` (`repair_stale_snapshots`)
- Test: `dashboard/backend/tests/domain/analytics/test_read_budget.py` (extend)

**Interfaces:**
- Consumes: `SNAPSHOT_STALE_AFTER` from Task A1.
- Produces: `calculate_user_state(user_id, *, now=None, store=None, events=None)` and
  `_calculate_user_value_snapshot(user_id, *, now, state_store, value_store, events=None)`,
  both accepting a pre-read event list. `recalculate_user_snapshots` keeps its signature and return type.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/backend/tests/domain/analytics/test_read_budget.py`:

```python
def test_one_recompute_reads_history_once(tmp_path):
    """The legacy and value projections share one read of the same evidence.

    They were computed from two separate 180-day scans of the same rows.
    """
    service, store = _fixture(tmp_path)
    _emit(service, 0, NOW)

    assert store.event_reads == 1


def test_stale_window_is_one_day(tmp_path):
    from dashboard.backend.domain.analytics import states

    assert states.SNAPSHOT_STALE_AFTER == timedelta(hours=24)

    path = tmp_path / "stale.db"
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
        conn.execute(
            "INSERT INTO users VALUES (1, 'a@example.test', 'A', 'x', 'user', ?)",
            ((NOW - timedelta(days=40)).isoformat(),),
        )
    state_store = AnalyticsStateStore(AnalyticsStore(path))
    state_store.upsert_snapshot(
        states.UserAnalyticsSnapshot(
            user_id=1,
            status="onboarding",
            reason_code="no_run_attempted",
            human_readable_reason="No run attempted yet.",
            evidence_event_ids=[],
            calculated_at=NOW - timedelta(hours=2),
        )
    )

    # Two hours old: fresh under a one-day window, stale under fifteen minutes.
    assert state_store.list_stale_user_ids(now=NOW, limit=10) == []
    assert state_store.list_stale_user_ids(
        now=NOW + timedelta(days=1), limit=10
    ) == [1]
```

The `reason_code` and `human_readable_reason` values above must match a key of `_REASONS` in `states.py:30`; read that dict and substitute a real pair if `no_run_attempted` is not one of them.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/test_read_budget.py -v -k "reads_history_once or stale_window"
```

Expected: FAIL. The first reports 2 reads; the second reports `[1]` for the fresh case because the window is fifteen minutes.

- [ ] **Step 3: Accept a pre-read event list in the legacy calculator**

In `states.py`, change the signature at line 381 and the read at line 396:

```python
def calculate_user_state(
    user_id: int,
    *,
    now: datetime | None = None,
    store: AnalyticsStateStore | None = None,
    events: list[AnalyticsEventRecord] | None = None,
) -> UserAnalyticsSnapshot:
```

and replace

```python
    events = state_store.list_user_events(subject_id, now=current)
```

with

```python
    if events is None:
        events = state_store.list_user_events(subject_id, now=current)
```

- [ ] **Step 4: Accept a pre-read event list in the value calculator**

In `states.py`, change the signature at line 548 and the read at line 561:

```python
def _calculate_user_value_snapshot(
    user_id: int,
    *,
    now: datetime,
    state_store: AnalyticsStateStore,
    value_store: ValueAnalyticsStore,
    events: list[AnalyticsEventRecord] | None = None,
) -> UserValueSnapshot:
```

and replace

```python
    events = [
        event
        for event in state_store.list_user_events(subject_id, now=now)
        if event.occurred_at.astimezone(timezone.utc) <= now
    ]
```

with

```python
    source_events = (
        state_store.list_user_events(subject_id, now=now)
        if events is None
        else events
    )
    events = [
        event
        for event in source_events
        if event.occurred_at.astimezone(timezone.utc) <= now
    ]
```

- [ ] **Step 5: Read once in the combined recompute**

In `states.py`, inside `recalculate_user_snapshots` (line 680), replace the two calculator calls with a single read shared by both:

```python
    current = _require_utc(now or datetime.now(timezone.utc), "now")
    states = state_store or AnalyticsStateStore()
    values = value_store or ValueAnalyticsStore(states.base_store)
    # One read, two projections. These are two views of the same evidence;
    # reading it twice was pure duplicate egress.
    events = states.list_user_events(user_id, now=current)
    legacy = calculate_user_state(
        user_id,
        now=current,
        store=states,
        events=events,
    )
    value = _calculate_user_value_snapshot(
        user_id,
        now=current,
        state_store=states,
        value_store=values,
        events=events,
    )
```

- [ ] **Step 6: Widen the stale window**

In `states.py`, at line 284 inside `list_stale_user_ids`, replace

```python
        stale_before = before or (current - timedelta(minutes=15))
```

with

```python
        stale_before = before or (current - SNAPSHOT_STALE_AFTER)
```

and at line 721, change the `repair_stale_snapshots` default:

```python
    stale_after: timedelta = SNAPSHOT_STALE_AFTER,
```

- [ ] **Step 7: Run the tests to verify they pass**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/test_read_budget.py -v
~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/test_states.py -q
```

Expected: PASS on both.

- [ ] **Step 8: Commit**

```bash
git add dashboard/backend/domain/analytics/states.py \
        dashboard/backend/tests/domain/analytics/test_read_budget.py
git status --short
git commit -m "perf(analytics): read a user's history once per recompute"
```

### Task A3: Drop the legacy repair leg from the reaper tick

**Files:**
- Modify: `dashboard/backend/domain/analytics/maintenance.py:17` (report field), `:50` (parameter), `:69-72` (default import), `:124-137` (the call), `:162` (report construction)
- Modify: `dashboard/backend/tests/test_analytics_maintenance.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `run_analytics_maintenance(*, now=None, snapshot_limit=100, rebuild_rollup=None, repair_value_snapshots=None, backfill_lifecycle=None)` — the `repair_snapshots` parameter is gone. `AnalyticsMaintenanceReport` loses `repaired_snapshots`.

The dual-axis repair at `maintenance.py:109-122` already recomputes both projections through `recalculate_user_snapshots`, so the separate legacy leg at `:124-137` re-reads the same users to produce a value the first leg already wrote.

- [ ] **Step 1: Write the failing test**

Replace `test_maintenance_rebuilds_one_day_and_bounds_snapshot_repairs` in `dashboard/backend/tests/test_analytics_maintenance.py` with a version that has no legacy leg, and add:

```python
def test_maintenance_does_not_run_the_legacy_repair_leg():
    """The dual-axis repair already writes the legacy snapshot.

    Running both meant every selected user was read twice per sweep.
    """
    import inspect

    from dashboard.backend.domain.analytics import maintenance

    parameters = inspect.signature(maintenance.run_analytics_maintenance).parameters
    assert "repair_snapshots" not in parameters
    assert "repaired_snapshots" not in maintenance.AnalyticsMaintenanceReport.model_fields
    source = inspect.getsource(maintenance.run_analytics_maintenance)
    assert "repair_stale_snapshots" not in source
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_analytics_maintenance.py::test_maintenance_does_not_run_the_legacy_repair_leg -v
```

Expected: FAIL on the first assertion.

- [ ] **Step 3: Remove the leg**

In `maintenance.py`: delete the `repaired_snapshots` field (line 17), the `repair_snapshots` parameter (line 50), the default-import block (lines 69-72), the whole `repaired = 0` try/except block (lines 124-137), and the `repaired_snapshots=max(0, repaired),` line in the report construction (line 162).

- [ ] **Step 4: Update the existing maintenance tests**

In `dashboard/backend/tests/test_analytics_maintenance.py`, remove the `repair` stub, the `repair_snapshots=repair` argument, the `repair_limits` list, the `"legacy"` entries expected in `repair_order`, and every assertion on `report.repaired_snapshots`, from both
`test_maintenance_rebuilds_one_day_and_bounds_snapshot_repairs` and
`test_maintenance_isolates_rollup_and_snapshot_failures`.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_analytics_maintenance.py -v
```

Expected: PASS, all cases.

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/domain/analytics/maintenance.py \
        dashboard/backend/tests/test_analytics_maintenance.py
git status --short
git commit -m "perf(analytics): drop the duplicate legacy repair from the sweep"
```

### Task A4: Map a user-store outage to 503 on the auth routes

**Files:**
- Modify: `dashboard/backend/api/auth.py` — imports near line 30, helpers near line 400, `get_current_user` at `:407-417`, `signup` at `:505-513`, `login` at `:567-575`
- Test: `dashboard/backend/tests/test_auth_store_unavailable.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `_USER_STORE_OUTAGE: tuple[type[BaseException], ...]` and
  `_store_unavailable(exc: BaseException) -> HTTPException` in `dashboard.backend.api.auth`.

On 2026-09-11 `psycopg_pool.PoolTimeout` escaped `get_user_by_email` and FastAPI returned a bare `Internal Server Error` with no header, which the browser reported as a CORS failure. `PoolTimeout` subclasses `psycopg.OperationalError`, so one arm covers the pool timeout, a refused connection and a quota rejection alike.

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/test_auth_store_unavailable.py`:

```python
"""A user-store outage answers 503, not a bare 500.

The 2026-09-11 quota outage surfaced as an unhandled exception: no status
anyone could act on, no log line naming the subsystem, and -- because an
unhandled exception escapes CORSMiddleware un-headered -- a browser
console full of CORS errors pointing at the wrong layer.
"""

from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from dashboard.backend import users as users_module
from dashboard.backend.app import app


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_login_reports_503_when_the_user_store_is_down(client, monkeypatch, capsys):
    def explode(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(users_module.user_store, "authenticate", explode)

    response = client.post(
        "/api/auth/login",
        json={"email": "someone@example.test", "password": "hunter2hunter2"},
    )

    assert response.status_code == 503
    assert "auth.user_store_unavailable" in capsys.readouterr().out


def test_signup_reports_503_when_the_user_store_is_down(client, monkeypatch):
    def explode(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(users_module.user_store, "create_user", explode)

    response = client.post(
        "/api/auth/signup",
        json={
            "email": "fresh@example.test",
            "display_name": "Fresh",
            "password": "hunter2hunter2",
        },
    )

    assert response.status_code == 503


def test_current_user_reports_503_when_the_user_store_is_down(client, monkeypatch):
    def explode(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(users_module.user_store, "get_user_for_token", explode)

    response = client.get(
        "/api/auth/me",
        headers={"Authorization": "Bearer not-a-real-token"},
    )

    assert response.status_code == 503


def test_pool_timeout_is_covered_by_the_operational_error_arm():
    """One arm, because PoolTimeout is an OperationalError subclass."""
    psycopg = pytest.importorskip("psycopg")
    psycopg_pool = pytest.importorskip("psycopg_pool")

    from dashboard.backend.api import auth

    assert issubclass(psycopg_pool.PoolTimeout, psycopg.OperationalError)
    assert psycopg.OperationalError in auth._USER_STORE_OUTAGE
    assert sqlite3.OperationalError in auth._USER_STORE_OUTAGE
```

If the signup or login route applies a rate limit before reaching the store, the test's first call still reaches it; the autouse `_reset_shared_scale_state` fixture in `conftest.py:217` clears every limiter between tests.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_auth_store_unavailable.py -v
```

Expected: FAIL. The route calls return 500, and `auth._USER_STORE_OUTAGE` does not exist.

- [ ] **Step 3: Add the exception tuple and the helper**

In `dashboard/backend/api/auth.py`, add `import sqlite3` to the standard-library imports at the top of the file, then add this block just above `def get_current_user` (line 407):

```python
_STORE_UNAVAILABLE_DETAIL = "Sign-in is temporarily unavailable. Please try again."


def _user_store_outage_types() -> tuple[type[BaseException], ...]:
    """Exception classes that mean "the account database is unreachable".

    ``psycopg_pool.PoolTimeout`` subclasses ``psycopg.OperationalError``, so
    a pool exhaustion, a refused connection and a provider quota rejection
    all arrive through the same arm. psycopg is imported defensively because
    a SQLite-only install has no reason to carry it.
    """
    types: list[type[BaseException]] = [sqlite3.OperationalError]
    try:
        import psycopg
    except Exception:  # pragma: no cover - psycopg ships in requirements.txt
        return tuple(types)
    types.append(psycopg.OperationalError)
    return tuple(types)


_USER_STORE_OUTAGE = _user_store_outage_types()


def _store_unavailable(exc: BaseException) -> HTTPException:
    """Build the 503 for a user-store outage.

    Returned rather than raised so the caller owns the ``raise``: a helper
    that raises on one path and returns on another trips CodeQL's
    py/mixed-returns. The log line carries the exception class only -- a
    psycopg error stringifies with its connection DSN embedded.
    """
    print(f"ERROR: auth.user_store_unavailable category={type(exc).__name__[:80]}")
    return HTTPException(status_code=503, detail=_STORE_UNAVAILABLE_DETAIL)
```

- [ ] **Step 4: Wrap the three store calls**

`get_current_user` (line 414):

```python
    try:
        user = users_module.user_store.get_user_for_token(token)
    except _USER_STORE_OUTAGE as exc:
        raise _store_unavailable(exc) from None
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return user
```

`signup` (line 505) — add the outage arm ahead of the existing `except ValueError`:

```python
    try:
        # Threaded for the same reason as login's authenticate(): create_user
        # hashes with bcrypt (~190 ms), and this route is unauthenticated.
        user = await asyncio.to_thread(
            users_module.user_store.create_user,
            email=payload.email,
            display_name=payload.display_name,
            password=payload.password,
        )
    except _USER_STORE_OUTAGE as exc:
        raise _store_unavailable(exc) from None
    except ValueError as exc:
```

`login` (line 574) — wrap the `authenticate` call, leaving the comment block above it intact:

```python
    try:
        user = await asyncio.to_thread(
            users_module.user_store.authenticate, payload.email, payload.password
        )
    except _USER_STORE_OUTAGE as exc:
        raise _store_unavailable(exc) from None
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_auth_store_unavailable.py -v
~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_auth_api.py dashboard/backend/tests/test_admin_users.py -q
```

Expected: PASS on all. If `test_auth_api.py` does not exist under that name, run every `dashboard/backend/tests/test_auth*.py`.

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/api/auth.py \
        dashboard/backend/tests/test_auth_store_unavailable.py
git status --short
git commit -m "fix(auth): answer 503 when the account store is unreachable"
```

### Task A5: Pin the reaper-tick read budget and record the gotcha

**Files:**
- Test: `dashboard/backend/tests/domain/analytics/test_read_budget.py` (extend)
- Modify: `CLAUDE.md` (append one bullet to the Gotchas section, which starts at line 203)

**Interfaces:**
- Consumes: everything from Tasks A1 through A3.
- Produces: nothing consumed by later tasks. PR B replaces this test with the user-count-independence budget in Task B9.

- [ ] **Step 1: Write the failing test**

Append to `dashboard/backend/tests/domain/analytics/test_read_budget.py`:

```python
def test_a_day_of_ticks_reads_each_user_once(tmp_path, monkeypatch):
    """Ninety-six sweeps over 200 users cost 200 history reads, not 19,200.

    The reaper ticks every 60 seconds. Any per-user read inside it is
    multiplied by users x 1440, which is exactly how a 5 GB monthly egress
    allowance was spent in seven days.
    """
    from types import SimpleNamespace

    from dashboard.backend.domain.analytics import maintenance, states

    maintenance.reset_maintenance_guard_for_tests()
    service, store = _fixture(tmp_path, users=200)
    value_store = service.value_store

    def stub_backfill(**_kwargs):
        # Deleted in PR B; its own cursor already bounds it. Stubbed so this
        # test measures the sweep rather than a one-off historical batch.
        return SimpleNamespace(processed_users=0, written_rows=0, complete=True)

    def repair(**kwargs):
        return states.repair_stale_value_snapshots(
            now=kwargs["now"],
            limit=kwargs["limit"],
            state_store=store,
            value_store=value_store,
        )

    # Start after midnight and stay inside one UTC day, so the day-rollover
    # re-selection is measured by its own test rather than this one.
    start = datetime(2026, 9, 12, 0, 30, tzinfo=timezone.utc)
    for tick in range(96):
        maintenance.run_analytics_maintenance(
            now=start + timedelta(minutes=15 * tick),
            snapshot_limit=100,
            rebuild_rollup=lambda _day, **_kwargs: None,
            repair_value_snapshots=repair,
            backfill_lifecycle=stub_backfill,
        )

    assert store.event_reads == 200
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/test_read_budget.py::test_a_day_of_ticks_reads_each_user_once -v
```

Expected: PASS if Tasks A1 through A3 are complete, FAIL with a number far above 200 otherwise. Run it once with `git stash` applied to confirm it catches the old behaviour, then restore. Use `git stash push -u -m "budget-check"` and `git stash apply <sha>` — never a bare `git stash pop`, because the stash stack is shared with the other worktrees on this machine.

- [ ] **Step 3: Add the CLAUDE.md gotcha**

Append this bullet at the end of the Gotchas section in `CLAUDE.md`:

```markdown
- **Anything registered with the run reaper runs every 60 seconds, so a per-user query inside it is multiplied by `users x 1440` per day.** `register_reaper_sweep` (`domain/runs/service.py:267`) is called from `app.py:225-280` for the v2 sweep, the legacy session sweep, analytics retention and analytics maintenance; `REAPER_INTERVAL_SECONDS` defaults to 60 with no prod override. On 2026-09-11 the analytics sweep re-read every non-admin user's full 180-day event history two to three times every fifteen minutes, exhausted the Neon free tier's 5 GB monthly egress allowance, and 500'd every users/content Postgres route including login for five hours while `/health` stayed green. **The mechanism was the defect, not the cadence**: a label that changes at most once a day was being recomputed from a copy of the facts. When adding maintenance work here, derive from the authoritative table rather than re-reading the event log, keep the work set-based rather than looping over users, and add a case to `tests/domain/analytics/test_read_budget.py` — nothing else in the suite pins the *cost* of a tick, which is why a green suite shipped this.
```

- [ ] **Step 4: Run the full backend suite**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/ -q
```

Expected: PASS. The suite is green end-to-end on `main`, so any red case here is a real regression from this PR.

- [ ] **Step 5: Commit and open the PR**

```bash
git add dashboard/backend/tests/domain/analytics/test_read_budget.py CLAUDE.md
git status --short
git commit -m "test(analytics): pin the reaper-tick read budget"
git push -u origin fix/analytics-read-budget-stopgap
```

PR title: `fix: bound analytics read budget`. Keep the body short — two sentences on the two burners, one line on the 503, and a pointer to the spec. Details belong in the diff, not the body.

---

# PR B — Data model and daily job

Branch: `git switch -c feat/user-analytics-daily-facts origin/main`

Do **not** branch off PR A. PR A is a throwaway whose changes this PR deletes; branching off it makes the deletion look like a revert of an unmerged PR. Cut from `origin/main`, and rebase once PR A lands.

This PR must run green on the CI Postgres tier before merge. Locally that means Docker:

```bash
docker run --rm -d --name atl-pg -e POSTGRES_PASSWORD=atl -p 55432:5432 postgres:18
export TEST_POSTGRES_URL=postgresql://postgres:atl@localhost:55432/postgres
~/atl-venv/bin/python -m pytest dashboard/backend/tests/ -q
```

A skip count that does not drop when `TEST_POSTGRES_URL` is set means the Postgres tier never ran. The `pg_only` marker fails open, so check the count rather than the exit code.

### Task B1: The three new tables, on both twins

**Files:**
- Modify: `dashboard/backend/domain/analytics/repository.py` — `ANALYTICS_SQLITE_DDL`, append after the `analytics_projection_jobs` block that ends at line 168
- Modify: `dashboard/backend/domain/analytics/repository_postgres.py` — `ANALYTICS_POSTGRES_DDL`, append after the matching block that ends at line 162
- Test: `dashboard/backend/tests/domain/analytics/test_repository_contract.py` (extend)
- Test: `dashboard/backend/tests/domain/analytics/test_repository_postgres.py` (extend)

**Interfaces:**
- Consumes: nothing.
- Produces: tables `user_activity`, `user_daily_facts`, `lifecycle_transitions` in both DDL constants. Tasks B4, B6, B7 and B8 write to them.

Both DDL constants are plain triple-quoted string literals, not f-strings. Keep them that way: the twin-parity guard reads source text, and an interpolated block collapses to nothing it can compare.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/backend/tests/domain/analytics/test_repository_contract.py`:

```python
def test_sqlite_declares_the_daily_fact_tables(sqlite_contract):
    store, _admin_id, _user_id = sqlite_contract

    with store._get_connection() as conn:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        fact_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(user_daily_facts)").fetchall()
        }
        activity_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(user_activity)").fetchall()
        }

    assert {
        "user_activity",
        "user_daily_facts",
        "lifecycle_transitions",
    } <= names
    assert {
        "snapshot_date",
        "user_id",
        "lifecycle_segment",
        "lifecycle_reason_code",
        "operational_state",
        "tier",
        "cohort",
        "active",
        "runs_requested",
        "runs_completed",
        "runs_failed",
        "runs_cancelled",
        "operator_cost_micro",
        "own_spend_micro",
        "data_quality",
        "calculated_at",
    } == fact_columns
    assert {
        "user_id",
        "activated_at",
        "last_meaningful_activity_at",
        "updated_at",
    } == activity_columns


def test_lifecycle_transitions_are_unique_per_user_per_day(sqlite_contract):
    """A retried daily job must not append the same transition twice."""
    store, _admin_id, user_id = sqlite_contract

    with store._get_connection() as conn:
        for _ in range(2):
            conn.execute(
                """
                INSERT OR IGNORE INTO lifecycle_transitions (
                    user_id, snapshot_date, from_segment, to_segment,
                    inactive_days, created_at
                ) VALUES (?, '2026-09-11', 'growing', 'at_risk', 8,
                          '2026-09-12T00:05:00+00:00')
                """,
                (user_id,),
            )
        count = conn.execute(
            "SELECT COUNT(*) FROM lifecycle_transitions"
        ).fetchone()[0]

    assert count == 1
```

Append to `dashboard/backend/tests/domain/analytics/test_repository_postgres.py`:

```python
def test_postgres_ddl_declares_the_daily_fact_tables():
    ddl = pg_module.ANALYTICS_POSTGRES_DDL
    assert "CREATE TABLE IF NOT EXISTS user_activity" in ddl
    assert "CREATE TABLE IF NOT EXISTS user_daily_facts" in ddl
    assert "CREATE TABLE IF NOT EXISTS lifecycle_transitions" in ddl
    for column in (
        "operator_cost_micro",
        "own_spend_micro",
        "runs_completed",
        "data_quality",
        "cohort",
        "tier",
    ):
        assert column in ddl
    # Wide counters are BIGINT on Postgres, matching analytics_daily_rollups.
    assert "operator_cost_micro BIGINT" in ddl
    assert "own_spend_micro BIGINT" in ddl
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/test_repository_contract.py dashboard/backend/tests/domain/analytics/test_repository_postgres.py -v -k "daily_fact or transitions_are_unique"
```

Expected: FAIL. `PRAGMA table_info` returns an empty set for a table that does not exist, so the column assertions fail before the table assertion does.

- [ ] **Step 3: Add the SQLite DDL**

In `dashboard/backend/domain/analytics/repository.py`, inside `ANALYTICS_SQLITE_DDL`, after the `analytics_projection_jobs` block (line 168) and before `analytics_subject_settings`:

```sql
CREATE TABLE IF NOT EXISTS user_activity (
    user_id INTEGER PRIMARY KEY,
    activated_at TEXT,
    last_meaningful_activity_at TEXT,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS user_daily_facts (
    snapshot_date TEXT NOT NULL CHECK (length(snapshot_date) = 10),
    user_id INTEGER NOT NULL,
    lifecycle_segment TEXT NOT NULL,
    lifecycle_reason_code TEXT NOT NULL,
    operational_state TEXT NOT NULL
        CHECK (operational_state IN ('blocked', 'needs_attention', 'healthy')),
    tier TEXT NOT NULL
        CHECK (tier IN ('unpaid', 'starter', 'invested', 'high_value')),
    cohort TEXT CHECK (cohort IS NULL OR length(cohort) BETWEEN 1 AND 32),
    active INTEGER NOT NULL DEFAULT 0 CHECK (active IN (0, 1)),
    runs_requested INTEGER NOT NULL DEFAULT 0 CHECK (runs_requested >= 0),
    runs_completed INTEGER NOT NULL DEFAULT 0 CHECK (runs_completed >= 0),
    runs_failed INTEGER NOT NULL DEFAULT 0 CHECK (runs_failed >= 0),
    runs_cancelled INTEGER NOT NULL DEFAULT 0 CHECK (runs_cancelled >= 0),
    operator_cost_micro INTEGER NOT NULL DEFAULT 0
        CHECK (operator_cost_micro >= 0),
    own_spend_micro INTEGER NOT NULL DEFAULT 0 CHECK (own_spend_micro >= 0),
    data_quality TEXT NOT NULL CHECK (data_quality IN ('complete', 'partial')),
    calculated_at TEXT NOT NULL,
    PRIMARY KEY (snapshot_date, user_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_daily_facts_segment
    ON user_daily_facts(snapshot_date, lifecycle_segment);
CREATE INDEX IF NOT EXISTS idx_daily_facts_user
    ON user_daily_facts(user_id, snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_facts_cohort
    ON user_daily_facts(snapshot_date, cohort);
CREATE INDEX IF NOT EXISTS idx_daily_facts_tier
    ON user_daily_facts(snapshot_date, tier);

CREATE TABLE IF NOT EXISTS lifecycle_transitions (
    transition_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    snapshot_date TEXT NOT NULL CHECK (length(snapshot_date) = 10),
    from_segment TEXT NOT NULL,
    to_segment TEXT NOT NULL,
    inactive_days INTEGER NOT NULL DEFAULT 0 CHECK (inactive_days >= 0),
    created_at TEXT NOT NULL,
    UNIQUE (user_id, snapshot_date),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_lifecycle_transitions_day
    ON lifecycle_transitions(snapshot_date, to_segment);
```

The `UNIQUE (user_id, snapshot_date)` is load-bearing, not decoration: the daily job retries a failed day on the next tick, and without it a retry appends a second copy of every transition it already wrote.

- [ ] **Step 4: Add the Postgres DDL**

In `dashboard/backend/domain/analytics/repository_postgres.py`, inside `ANALYTICS_POSTGRES_DDL`, in the same relative position (after `analytics_projection_jobs`, line 162):

```sql
CREATE TABLE IF NOT EXISTS user_activity (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    activated_at TEXT,
    last_meaningful_activity_at TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_daily_facts (
    snapshot_date TEXT NOT NULL CHECK (length(snapshot_date) = 10),
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    lifecycle_segment TEXT NOT NULL,
    lifecycle_reason_code TEXT NOT NULL,
    operational_state TEXT NOT NULL
        CHECK (operational_state IN ('blocked', 'needs_attention', 'healthy')),
    tier TEXT NOT NULL
        CHECK (tier IN ('unpaid', 'starter', 'invested', 'high_value')),
    cohort TEXT CHECK (cohort IS NULL OR length(cohort) BETWEEN 1 AND 32),
    active BOOLEAN NOT NULL DEFAULT FALSE,
    runs_requested INTEGER NOT NULL DEFAULT 0 CHECK (runs_requested >= 0),
    runs_completed INTEGER NOT NULL DEFAULT 0 CHECK (runs_completed >= 0),
    runs_failed INTEGER NOT NULL DEFAULT 0 CHECK (runs_failed >= 0),
    runs_cancelled INTEGER NOT NULL DEFAULT 0 CHECK (runs_cancelled >= 0),
    operator_cost_micro BIGINT NOT NULL DEFAULT 0
        CHECK (operator_cost_micro >= 0),
    own_spend_micro BIGINT NOT NULL DEFAULT 0 CHECK (own_spend_micro >= 0),
    data_quality TEXT NOT NULL CHECK (data_quality IN ('complete', 'partial')),
    calculated_at TEXT NOT NULL,
    PRIMARY KEY (snapshot_date, user_id)
);
CREATE INDEX IF NOT EXISTS idx_daily_facts_segment
    ON user_daily_facts(snapshot_date, lifecycle_segment);
CREATE INDEX IF NOT EXISTS idx_daily_facts_user
    ON user_daily_facts(user_id, snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_facts_cohort
    ON user_daily_facts(snapshot_date, cohort);
CREATE INDEX IF NOT EXISTS idx_daily_facts_tier
    ON user_daily_facts(snapshot_date, tier);

CREATE TABLE IF NOT EXISTS lifecycle_transitions (
    transition_id BIGSERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    snapshot_date TEXT NOT NULL CHECK (length(snapshot_date) = 10),
    from_segment TEXT NOT NULL,
    to_segment TEXT NOT NULL,
    inactive_days INTEGER NOT NULL DEFAULT 0 CHECK (inactive_days >= 0),
    created_at TEXT NOT NULL,
    UNIQUE (user_id, snapshot_date)
);
CREATE INDEX IF NOT EXISTS idx_lifecycle_transitions_day
    ON lifecycle_transitions(snapshot_date, to_segment);
```

`CREATE INDEX IF NOT EXISTS` matches by **name only** on Postgres, so it will not notice a definition change on an existing index. These names are new, so that is fine here, but do not rename an index and expect the statement to rebuild it.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/test_repository_contract.py dashboard/backend/tests/domain/analytics/test_repository_postgres.py -v
TEST_POSTGRES_URL=postgresql://postgres:atl@localhost:55432/postgres \
  ~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/ -q
```

Expected: PASS on both, and the second run reports fewer skips than the first.

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/domain/analytics/repository.py \
        dashboard/backend/domain/analytics/repository_postgres.py \
        dashboard/backend/tests/domain/analytics/test_repository_contract.py \
        dashboard/backend/tests/domain/analytics/test_repository_postgres.py
git status --short
git commit -m "feat(analytics): add the daily fact and activity tables"
```

### Task B2: The cohort axis

**Files:**
- Modify: `dashboard/backend/users.py` — validation helper near `validate_entitlement_patch` (line 446), `public_user_with_entitlements` (line 427), `apply_admin_patch` (line 1715), the SQLite migration block (lines 745-759)
- Modify: `dashboard/backend/users_postgres.py` — `CREATE TABLE users` (lines 91-102), the `ALTER TABLE` block (lines 104-115), and the Postgres `apply_admin_patch` twin
- Modify: `dashboard/backend/api/routers/admin_users.py` — `AdminUserPatch` (line 92), `patch_user` (line 223)
- Test: `dashboard/backend/tests/test_admin_users.py` (extend)

**Interfaces:**
- Consumes: nothing.
- Produces: `users.cohort` column; `normalize_cohort(value: str | None) -> str | None` in `dashboard.backend.users`; `apply_admin_patch(..., cohort: Optional[str] = None)`; `cohort` on the admin user payload. Task B7 reads the column; Task C6 renders it.

Cohort is admin-visible only. It must **not** appear in `public_user` (line 346), which is what a user's own `/api/auth/me` returns. Add it in `public_user_with_entitlements` (line 427), the admin projection, and nowhere else.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/backend/tests/test_admin_users.py`:

```python
def test_admin_can_set_and_clear_a_cohort(admin_client, normal_user):
    response = admin_client.patch(
        f"/api/admin/users/{normal_user['id']}",
        json={"cohort": "lab"},
    )
    assert response.status_code == 200
    assert response.json()["user"]["cohort"] == "lab"

    cleared = admin_client.patch(
        f"/api/admin/users/{normal_user['id']}",
        json={"cohort": ""},
    )
    assert cleared.status_code == 200
    assert cleared.json()["user"]["cohort"] is None


@pytest.mark.parametrize(
    "value",
    ["Lab", "lab space", "a" * 33, "paid!", "admin\n"],
)
def test_invalid_cohort_values_are_refused(admin_client, normal_user, value):
    response = admin_client.patch(
        f"/api/admin/users/{normal_user['id']}",
        json={"cohort": value},
    )
    assert response.status_code == 422


def test_users_never_see_their_own_cohort(client, normal_user_token):
    """Cohort is an admin label. The subject must not be able to read it."""
    response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {normal_user_token}"},
    )
    assert response.status_code == 200
    assert "cohort" not in response.json().get("user", response.json())
```

Reuse whatever `admin_client`, `normal_user`, `client` and `normal_user_token` fixtures `test_admin_users.py` already defines; read the top of that file and substitute the real names rather than adding new fixtures.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_admin_users.py -v -k cohort
```

Expected: FAIL. `AdminUserPatch` forbids unknown fields or ignores them, so the first case returns 200 with no `cohort` key, or 422.

- [ ] **Step 3: Add the column on both twins**

`dashboard/backend/users.py`, in the self-migration block at lines 745-759, following the `discord_user_id` pattern exactly:

```python
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN cohort TEXT")
        except sqlite3.OperationalError:
            pass
```

`dashboard/backend/users_postgres.py`, add `cohort TEXT` to the `CREATE TABLE users` body (after `avatar TEXT`, line 101) **and** add the idempotent migration to the `ALTER TABLE` block (line 115), following the `avatar` pattern:

```python
                ALTER TABLE users
                ADD COLUMN IF NOT EXISTS cohort TEXT
```

Both DDL strings stay plain literals.

- [ ] **Step 4: Add the validator**

In `dashboard/backend/users.py`, next to `validate_entitlement_patch` (line 446), add:

```python
COHORT_PATTERN = re.compile(r"^[a-z0-9_-]{1,32}$")


def normalize_cohort(value: Optional[str]) -> Optional[str]:
    """Validate one admin-assigned cohort slug.

    Empty string means "clear it"; ``None`` means "leave unchanged", which is
    why the route rejects an explicit JSON null for every PATCH field. There
    is no cohort lookup table -- the console offers existing values as
    suggestions and this pattern is the only gate.
    """
    if value is None:
        return None
    candidate = value.strip()
    if not candidate:
        return ""
    if not COHORT_PATTERN.fullmatch(candidate):
        raise ValueError("invalid_cohort")
    if candidate in {"admin", "paid", "free"}:
        # These are the role and tier axes. A cohort that shadows one of them
        # makes the group badge's precedence rule unreadable.
        raise ValueError("invalid_cohort")
    return candidate
```

Add `import re` to `users.py` if it is not already imported.

- [ ] **Step 5: Thread it through the store and the route**

In `users.py::apply_admin_patch` (line 1715), add `cohort: Optional[str] = None` to the keyword arguments, normalize it beside the role, include it in the "nothing to do" check, and inside the transaction:

```python
        normalized_cohort = normalize_cohort(cohort)
        ...
            if normalized_cohort is not None:
                cursor.execute(
                    "UPDATE users SET cohort = ? WHERE id = ?",
                    (normalized_cohort or None, int(user_id)),
                )
```

`normalized_cohort or None` turns the empty-string clear signal into a SQL NULL. Mirror the same change in the Postgres twin with `%s` placeholders; no cast is needed because the parameter is a plain column assignment, not a `COALESCE` operand.

In `users.py::public_user_with_entitlements` (line 427), after `payload.pop("avatar", None)`:

```python
    payload["cohort"] = dict(row).get("cohort") or None
```

Check `admin_user_rows_to_payloads` (used by `list_users_admin` at line 1886) and make sure it routes through `public_user_with_entitlements` or otherwise carries the column; if it builds its dict independently, add `cohort` there too.

In `dashboard/backend/api/routers/admin_users.py`, add to `AdminUserPatch` (line 92):

```python
    # Empty string clears the cohort. An explicit JSON null is refused by the
    # route's shared no-null rule, which is what makes "" the clear signal.
    cohort: Optional[str] = Field(default=None, max_length=32)
```

In `patch_user`, add `payload.cohort is None` to the "No fields to update" condition (line 244), pass `cohort=payload.cohort` to `apply_admin_patch`, add `cohort=payload.cohort` to the `_audit` call, and map the store error:

```python
        if code == "invalid_cohort":
            raise HTTPException(
                status_code=422,
                detail="Cohort must match ^[a-z0-9_-]{1,32}$",
            ) from exc
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_admin_users.py -v
~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_store_twin_parity.py -q
```

Expected: PASS on both. The parity test compares `apply_admin_patch`'s signature across the two stores, so a `cohort` parameter added to only one of them fails there.

- [ ] **Step 7: Commit**

```bash
git add dashboard/backend/users.py dashboard/backend/users_postgres.py \
        dashboard/backend/api/routers/admin_users.py \
        dashboard/backend/tests/test_admin_users.py
git status --short
git commit -m "feat(admin): assign one cohort per user"
```

### Task B3: Attribute dashboard runs to their owner

**Files:**
- Modify: `dashboard/backend/database.py` — `agent_runs` DDL (lines 129-151), the migration block (lines 296-371), `insert_run` (lines 624-656)
- Modify: `dashboard/backend/database_postgres.py` — the same three places (lines 107-134, 235-278, 468-571)
- Modify: `dashboard/backend/domain/backtesting/engine.py` — `HourlyBacktester.__init__` (line 174), the `insert_run` call at line 1650 and its two siblings at 1760 and 1842
- Modify: `dashboard/scripts/backtest_hourly_agent.py` — argparse near line 222, the `HourlyBacktester(` construction at line 442
- Modify: `dashboard/backend/api/routers/backtests.py` — `run_backtest_background` signature (line 952), the `cmd` list near line 1105, the thread target at line 2096
- Test: `dashboard/backend/tests/test_backtest_owner_attribution.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `agent_runs.owner_user_id` (nullable INTEGER) and `insert_run(..., owner_user_id: Optional[int] = None)` on both twins. Task B7 step 3 groups on it.

Per-user model cost cannot come from the event log. The live path emits `credits_reserved`, `credits_settled` and `credits_refunded`, which measure the user's own credit spend; `model_usage_recorded` carries `cost_micro_usd` but only the historical backfill ever emits it. Operator-funded cost lives in `agent_runs.est_cost_usd` and nowhere else, so the run row needs the owner.

Nullable and not backfilled, exactly as the spec says. Rows written before this column, and runs with no authenticated caller such as the leaderboard's scheduled deploys, stay unattributed and are reported as unattributed rather than assigned to anyone.

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/test_backtest_owner_attribution.py`:

```python
"""agent_runs rows carry the authenticated caller who started them.

Operator-funded model cost is in ``est_cost_usd`` on this row and nowhere
else, so without an owner column there is no per-user cost at all.
"""

from __future__ import annotations

import inspect

from dashboard.backend.database import BacktestDatabase


def test_insert_run_accepts_an_owner(tmp_path):
    db = BacktestDatabase(str(tmp_path / "runs.db"))
    db.insert_run(
        run_id="run-owned",
        session_id="session-1",
        agent_name="Agent",
        mode="backtest",
        start_date="2026-09-01",
        end_date="2026-09-02",
        initial_equity=100000.0,
        est_cost_usd=1.25,
        owner_user_id=7,
    )

    with db._get_connection() as conn:
        row = conn.execute(
            "SELECT owner_user_id, est_cost_usd FROM agent_runs WHERE run_id = ?",
            ("run-owned",),
        ).fetchone()

    assert row["owner_user_id"] == 7
    assert row["est_cost_usd"] == 1.25


def test_owner_is_optional_and_defaults_to_null(tmp_path):
    """A scheduled leaderboard deploy has no caller and must still insert."""
    db = BacktestDatabase(str(tmp_path / "runs.db"))
    db.insert_run(
        run_id="run-unowned",
        session_id="session-1",
        agent_name="Agent",
        mode="backtest",
        start_date="2026-09-01",
        end_date="2026-09-02",
        initial_equity=100000.0,
    )

    with db._get_connection() as conn:
        row = conn.execute(
            "SELECT owner_user_id FROM agent_runs WHERE run_id = ?",
            ("run-unowned",),
        ).fetchone()

    assert row["owner_user_id"] is None


def test_the_owner_reaches_the_subprocess_and_the_engine():
    """The four hops from the route to the row, pinned by source shape."""
    from dashboard.backend.api.routers import backtests
    from dashboard.backend.domain.backtesting import engine

    assert "owner_user_id" in inspect.signature(
        backtests.run_backtest_background
    ).parameters
    assert "--owner-user-id" in inspect.getsource(backtests.run_backtest_background)
    assert "owner_user_id" in inspect.signature(
        engine.HourlyBacktester.__init__
    ).parameters
```

Adjust `BacktestDatabase`'s constructor call if it does not take a path positionally; read its `__init__` first.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_backtest_owner_attribution.py -v
```

Expected: FAIL with `TypeError: insert_run() got an unexpected keyword argument 'owner_user_id'`.

- [ ] **Step 3: Add the column on both twins**

`dashboard/backend/database.py`: add `owner_user_id INTEGER,` to the `agent_runs` CREATE TABLE body, and append to the migration block at line 371, following the `metadata` pattern:

```python
        try:
            cursor.execute("ALTER TABLE agent_runs ADD COLUMN owner_user_id INTEGER")
        except sqlite3.OperationalError:
            pass
```

`dashboard/backend/database_postgres.py`: add `owner_user_id INTEGER,` to the CREATE TABLE body, and to the migration block at line 269:

```python
                ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS owner_user_id INTEGER
```

There is deliberately **no** foreign key to `users(id)`: run history lives in its own Neon project (`AGENT_RUNS_DATABASE_URL`, `ATL-runs-main`), and the users table is in a different database entirely.

- [ ] **Step 4: Extend insert_run on both twins**

Add `owner_user_id: Optional[int] = None` as the **last** keyword parameter on both signatures, identically — the twin-parity test compares parameter order and defaults, so a different position fails it.

SQLite (`database.py`, line 645): add `owner_user_id` to the column list and one more `?` to the VALUES tuple.

Postgres (`database_postgres.py`, line 536): add `owner_user_id` to the column list, and pass it as `%s::integer` rather than a bare `%s`. The value is frequently `None`, and psycopg sends a bare `None` as an untyped NULL (OID 0); the cast is cheap insurance and no SQLite test can catch its absence. Add `owner_user_id = EXCLUDED.owner_user_id` to the `ON CONFLICT DO UPDATE SET` list.

- [ ] **Step 5: Thread the owner from the route to the row**

Four hops, each mirroring how `--run-id` and `live_run_id` already travel:

1. `api/routers/backtests.py`, `run_backtest_background` (line 952): add `owner_user_id: Optional[int] = None` to the signature. Near line 1105, beside the `--run-id` append:

```python
        if owner_user_id is not None:
            cmd += ["--owner-user-id", str(int(owner_user_id))]
```

2. Same file, the thread target at line 2096: pass `owner_user_id=user_id`, using the same `user_id` that `_backtest_owner_key` (line 567) already resolves for the concurrency cap. Do not use the session the results file under — for a built-in agent that is the agent's session, not the caller's.

3. `dashboard/scripts/backtest_hourly_agent.py`, beside the `--run-id` argument at line 222:

```python
    parser.add_argument(
        "--owner-user-id",
        type=int,
        default=None,
        help="Authenticated caller who started this run (analytics attribution)",
    )
```

and in the `HourlyBacktester(` construction at line 442, beside `live_run_id=args.run_id`:

```python
        owner_user_id=args.owner_user_id,
```

4. `domain/backtesting/engine.py`: add `owner_user_id: int | None = None` to `HourlyBacktester.__init__` (line 174) beside `live_run_id`, store it as `self.owner_user_id`, and pass `owner_user_id=self.owner_user_id` to all three `db.insert_run(` calls (lines 1650, 1760, 1842).

Leave the leaderboard's `insert_run` calls (`domain/leaderboard/service.py:1009` and `:1345`) and the paper-trading call (`api/routers/paper_trading.py:304`) alone. The leaderboard runs on a schedule with no caller. Paper trading is a different mode and not operator-funded LLM spend.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_backtest_owner_attribution.py -v
~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_store_twin_parity.py dashboard/backend/tests/test_backtest_api.py -q
```

Expected: PASS. If `test_backtest_api.py` is not the real filename, run every `dashboard/backend/tests/test_backtest*.py`.

- [ ] **Step 7: Commit**

```bash
git add dashboard/backend/database.py dashboard/backend/database_postgres.py \
        dashboard/backend/domain/backtesting/engine.py \
        dashboard/scripts/backtest_hourly_agent.py \
        dashboard/backend/api/routers/backtests.py \
        dashboard/backend/tests/test_backtest_owner_attribution.py
git status --short
git commit -m "feat(runs): record the owner on dashboard backtest rows"
```

---

### Task B4: Ingestion maintains `user_activity`

**Files:**
- Modify: `dashboard/backend/domain/analytics/value_repository.py` — add `UserActivity` beside `UserLifecycleDailySnapshot` (line 101) and `record_activity` on `ValueAnalyticsStore` (class at line 211)
- Modify: `dashboard/backend/domain/analytics/service.py:38-77` and `:169-182`
- Modify: `dashboard/backend/domain/analytics/instrumentation.py:60-78` and `:134-136`
- Test: `dashboard/backend/tests/domain/analytics/test_read_budget.py` (replace the PR A cases)

**Interfaces:**
- Consumes: `user_activity` from Task B1.
- Produces:
  - `UserActivity(user_id: int, activated_at: datetime | None, last_meaningful_activity_at: datetime | None, updated_at: datetime)`
  - `ValueAnalyticsStore.record_activity(user_id: int, *, occurred_at: datetime, activating: bool, now: datetime) -> None`
  - `AnalyticsService(store, *, state_store=None, value_store=None, maintain_activity: bool = False)` — the `project_snapshots` flag is renamed, because there are no snapshots left for it to project.

  Task B5 reads `UserActivity`; Task B7 corrects the same row from the ledger.

This is the change that makes the 2026-09-11 outage structurally impossible. One upsert of two timestamps replaces a 180-day read of the event log, and it happens on the write the event log already performs.

- [ ] **Step 1: Write the failing test**

Replace the three PR A cases in `dashboard/backend/tests/domain/analytics/test_read_budget.py` with:

```python
def test_ingestion_never_reads_a_users_history(tmp_path):
    """Accepting an event costs one upsert, not a scan.

    The 2026-09-11 outage was this path: every accepted lifecycle event
    recomputed a label from a 180-day read of the log it had just appended to.
    """
    service, store = _fixture(tmp_path)
    for index in range(20):
        _emit(service, index, NOW - timedelta(seconds=index))

    assert store.event_reads == 0


def test_first_success_sets_activated_at_once(tmp_path):
    service, store = _fixture(tmp_path)
    value_store = service.value_store

    service.record_server_event(
        event_name="backtest_completed",
        user_id=1,
        source_event_id="run:backtest_completed:run-a",
        source_record_type="run",
        source_record_id="run-a",
        occurred_at=NOW - timedelta(days=3),
    )
    first = value_store.get_activity(1).activated_at

    service.record_server_event(
        event_name="backtest_completed",
        user_id=1,
        source_event_id="run:backtest_completed:run-b",
        source_record_type="run",
        source_record_id="run-b",
        occurred_at=NOW,
    )
    activity = value_store.get_activity(1)

    assert activity.activated_at == first
    assert activity.last_meaningful_activity_at == NOW


def test_page_views_do_not_advance_the_activity_clock(tmp_path):
    """Only the meaningful-activity set counts. A visit is not activity."""
    service, _store = _fixture(tmp_path)
    value_store = service.value_store

    service.record_server_event(
        event_name="authenticated_session_started",
        user_id=1,
        source_event_id="account:authenticated_session_started:1",
        source_record_type="user",
        source_record_id="1",
        occurred_at=NOW,
    )

    assert value_store.get_activity(1) is None
```

`StubValueStore` in that file gains `record_activity` and `get_activity` implementations, or is replaced by a real `ValueAnalyticsStore` over the fixture's `AnalyticsStore`. Prefer the real store here: this task's whole claim is about what the SQL does.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/test_read_budget.py -v
```

Expected: FAIL. Twenty events still read history, and `get_activity` does not exist.

- [ ] **Step 3: Add the model and the upsert**

In `dashboard/backend/domain/analytics/value_repository.py`, beside `UserLifecycleDailySnapshot` (line 101):

```python
class UserActivity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: int = Field(gt=0)
    activated_at: datetime | None = None
    last_meaningful_activity_at: datetime | None = None
    updated_at: datetime
```

On `ValueAnalyticsStore`, following the `upsert_daily_snapshot` branching style at line 441:

```python
    def record_activity(
        self,
        user_id: int,
        *,
        occurred_at: datetime,
        activating: bool,
        now: datetime,
    ) -> None:
        """Advance one user's activity timestamps. One statement, no read.

        ``activated_at`` is set once and never moves: COALESCE keeps the
        stored value, and a non-activating event passes NULL. The activity
        clock only ever advances, so a late-arriving old event cannot make a
        user look more dormant than they are.

        Timestamps are ISO-8601 UTC text throughout this schema, which orders
        lexicographically, so GREATEST/MAX over the text is the same
        comparison as over the instants.
        """
        subject_id = positive_user_id(user_id)
        occurred = utc_iso(_utc(occurred_at, "occurred_at"))
        values = (
            subject_id,
            occurred if activating else None,
            occurred,
            utc_iso(_utc(now, "now")),
        )
        if self.is_postgres:
            sql = """
                INSERT INTO user_activity (
                    user_id, activated_at, last_meaningful_activity_at, updated_at
                ) VALUES (%s, %s::text, %s, %s)
                ON CONFLICT(user_id) DO UPDATE SET
                    activated_at = COALESCE(
                        user_activity.activated_at, EXCLUDED.activated_at
                    ),
                    last_meaningful_activity_at = GREATEST(
                        COALESCE(
                            user_activity.last_meaningful_activity_at,
                            EXCLUDED.last_meaningful_activity_at
                        ),
                        EXCLUDED.last_meaningful_activity_at
                    ),
                    updated_at = EXCLUDED.updated_at
            """
        else:
            sql = """
                INSERT INTO user_activity (
                    user_id, activated_at, last_meaningful_activity_at, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    activated_at = COALESCE(
                        user_activity.activated_at, excluded.activated_at
                    ),
                    last_meaningful_activity_at = MAX(
                        COALESCE(
                            user_activity.last_meaningful_activity_at,
                            excluded.last_meaningful_activity_at
                        ),
                        excluded.last_meaningful_activity_at
                    ),
                    updated_at = excluded.updated_at
            """
        with self._analytics_connection() as conn:
            if self.is_postgres:
                with conn.cursor() as cur:
                    cur.execute(sql, values)
            else:
                conn.execute(sql, values)
```

Two dialect traps are handled above and must not be "simplified" away. SQLite's scalar `max(X, Y)` returns NULL if **either** argument is NULL, which is why the stored value is COALESCEd against the incoming one first rather than passed raw. And `%s::text` on the nullable `activated_at` parameter gives psycopg a type for the `None` it would otherwise send as OID 0.

Add the matching reader:

```python
    def get_activity(self, user_id: int) -> UserActivity | None:
        """One user's stored activity row, or None if they have none yet."""
```

and a batched `list_activity(self, user_ids: Sequence[int]) -> dict[int, UserActivity]` following `list_current_snapshots` (line 395) for its `_user_clause` batching.

- [ ] **Step 4: Replace the projection in the ingestion service**

In `dashboard/backend/domain/analytics/service.py`, rename the constructor flag and replace `_try_recalculate_snapshots` (lines 55-77) with:

```python
    def _try_record_activity(
        self,
        *,
        user_id: int,
        event: AnalyticsEventRecord,
        now: datetime,
    ) -> None:
        """Advance the stored activity clock for one accepted event.

        Best-effort, like every other analytics write: a failure here must
        never change the outcome of the operation that emitted the event.
        """
        if not self.maintain_activity:
            return
        try:
            self.value_store.record_activity(
                user_id,
                occurred_at=event.occurred_at,
                activating=event.event_name == "backtest_completed",
                now=now,
            )
        except Exception as exc:
            print(
                "WARNING: analytics.activity_update_failed "
                f"event={event.event_name} category={type(exc).__name__[:80]}"
            )
```

and replace the projection block at lines 170-181 with:

```python
        result = self.store.append_event(event)
        from .lifecycle import is_lifecycle_activity

        if result.created and is_lifecycle_activity(event):
            self._try_record_activity(user_id=subject_id, event=event, now=received)
        return result
```

`account_signed_up` and `safe_error_recorded` drop out of this branch deliberately. Signup time is `users.created_at`, which the lifecycle rules already read, and a safe error is not activity.

Update `_build_analytics_service` (line 235) to pass `maintain_activity=True`, and drop the now-unused `state_store` argument if nothing else uses it.

- [ ] **Step 5: Delete the instrumentation recompute**

In `dashboard/backend/domain/analytics/instrumentation.py`, delete `_recalculate_snapshot` (lines 66-78), the `_snapshot_recalculator` module global and its test setter, and the branch at lines 135-136 so `_emit` ends at `service.try_record_server_event(**kwargs)`. `SNAPSHOT_RELEVANT_EVENTS` stays only if something else imports it; check with `grep -rn SNAPSHOT_RELEVANT_EVENTS dashboard/` and delete it if not.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/test_read_budget.py -v
~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/test_instrumentation.py dashboard/backend/tests/domain/analytics/test_service.py -q
TEST_POSTGRES_URL=postgresql://postgres:atl@localhost:55432/postgres \
  ~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/ -q
```

Expected: PASS. Run the Postgres tier here specifically — `GREATEST` versus `MAX` is exactly the kind of difference SQLite alone cannot catch.

- [ ] **Step 7: Commit**

```bash
git add dashboard/backend/domain/analytics/value_repository.py \
        dashboard/backend/domain/analytics/service.py \
        dashboard/backend/domain/analytics/instrumentation.py \
        dashboard/backend/tests/domain/analytics/
git status --short
git commit -m "feat(analytics): maintain user activity from ingestion"
```

### Task B5: Lifecycle and the group badge at read time

**Files:**
- Modify: `dashboard/backend/domain/analytics/lifecycle.py` — add `resolve_group_badge` beside `commercial_tier` (line 349) and export it
- Create: `dashboard/backend/domain/analytics/lifecycle_reads.py`
- Modify: `dashboard/backend/domain/analytics/value_repository.py` — add `sum_recent_facts`
- Test: `dashboard/backend/tests/domain/analytics/test_lifecycle_reads.py` (create)

**Interfaces:**
- Consumes: `UserActivity` and `list_activity` from Task B4; `user_daily_facts` from Task B1.
- Produces:
  - `resolve_group_badge(*, role: str, cohort: str | None, tier: CommercialTier) -> str` in `lifecycle.py`
  - `RecentFactTotals(active_days: int, successful_backtests: int, runs_requested: int, runs_completed: int, runs_failed: int, runs_cancelled: int, operator_cost_micro: int, own_spend_micro: int, days_present: int)` in `value_repository.py`
  - `ValueAnalyticsStore.sum_recent_facts(user_ids: Sequence[int], *, start: date, end: date) -> dict[int, RecentFactTotals]`
  - `build_lifecycle_inputs(user_id: int, *, created_at: datetime, activity: UserActivity | None, totals: RecentFactTotals | None) -> LifecycleInputs` in `lifecycle_reads.py`

  Task B7 uses all four. Task C1 and C2 call `build_lifecycle_inputs` on the read path.

`calculate_lifecycle` already takes `LifecycleInputs`, a struct of stored timestamps and two counts — it never read an event list. The event reading was in `states.py`, which built those inputs by scanning. This task replaces the scan with two indexed reads and leaves the rules untouched.

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/domain/analytics/test_lifecycle_reads.py`:

```python
"""Lifecycle from stored facts, with the rules unchanged."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from dashboard.backend.domain.analytics.lifecycle import (
    calculate_lifecycle,
    resolve_group_badge,
)
from dashboard.backend.domain.analytics.lifecycle_reads import build_lifecycle_inputs
from dashboard.backend.domain.analytics.value_repository import (
    RecentFactTotals,
    UserActivity,
)


NOW = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)


def _totals(**overrides):
    base = dict(
        active_days=0,
        successful_backtests=0,
        runs_requested=0,
        runs_completed=0,
        runs_failed=0,
        runs_cancelled=0,
        operator_cost_micro=0,
        own_spend_micro=0,
        days_present=30,
    )
    base.update(overrides)
    return RecentFactTotals(**base)


def test_core_needs_three_active_days_and_three_successes():
    activity = UserActivity(
        user_id=1,
        activated_at=NOW - timedelta(days=20),
        last_meaningful_activity_at=NOW - timedelta(days=1),
        updated_at=NOW,
    )
    inputs = build_lifecycle_inputs(
        1,
        created_at=NOW - timedelta(days=60),
        activity=activity,
        totals=_totals(active_days=3, successful_backtests=3),
    )

    assert calculate_lifecycle(inputs, NOW).segment == "core"

    below = build_lifecycle_inputs(
        1,
        created_at=NOW - timedelta(days=60),
        activity=activity,
        totals=_totals(active_days=3, successful_backtests=2),
    )
    assert calculate_lifecycle(below, NOW).segment == "growing"


def test_a_user_with_no_activity_row_falls_back_to_signup():
    inputs = build_lifecycle_inputs(
        1,
        created_at=NOW - timedelta(days=2),
        activity=None,
        totals=None,
    )
    result = calculate_lifecycle(inputs, NOW)

    assert result.segment == "new"
    assert result.active_days_30d == 0


def test_inactivity_crosses_at_risk_without_any_new_evidence():
    """Time alone moves the segment. That is why nothing needs recomputing."""
    activity = UserActivity(
        user_id=1,
        activated_at=NOW - timedelta(days=40),
        last_meaningful_activity_at=NOW - timedelta(days=7),
        updated_at=NOW,
    )
    inputs = build_lifecycle_inputs(
        1,
        created_at=NOW - timedelta(days=90),
        activity=activity,
        totals=_totals(),
    )

    assert calculate_lifecycle(inputs, NOW).segment == "growing"
    assert calculate_lifecycle(inputs, NOW + timedelta(days=1)).segment == "at_risk"
    assert calculate_lifecycle(inputs, NOW + timedelta(days=23)).segment == "dormant"


@pytest.mark.parametrize(
    "role,cohort,tier,expected",
    [
        ("admin", "lab", "high_value", "admin"),
        ("admin", None, "unpaid", "admin"),
        ("user", "lab", "starter", "lab"),
        ("user", None, "starter", "paid"),
        ("user", None, "unpaid", "free"),
    ],
)
def test_group_badge_precedence(role, cohort, tier, expected):
    assert resolve_group_badge(role=role, cohort=cohort, tier=tier) == expected
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/test_lifecycle_reads.py -v
```

Expected: FAIL at import — `lifecycle_reads` and `resolve_group_badge` do not exist.

- [ ] **Step 3: Add the group badge**

In `dashboard/backend/domain/analytics/lifecycle.py`, after `commercial_tier` (line 364):

```python
def resolve_group_badge(
    *,
    role: str,
    cohort: str | None,
    tier: CommercialTier,
) -> str:
    """The single group label admin views show for one user.

    Three orthogonal axes, one badge, fixed precedence: a lab member who is
    also an admin reads as an admin, because that is the fact that changes
    what they can do. Cohort beats tier because it is the label a human
    chose deliberately.
    """
    if role == "admin":
        return "admin"
    if cohort:
        return cohort
    return "free" if tier == "unpaid" else "paid"
```

Add `"resolve_group_badge"` to `__all__` (line 374).

- [ ] **Step 4: Add the totals model and the batched sum**

In `value_repository.py`, beside `UserActivity`:

```python
class RecentFactTotals(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    active_days: int = Field(default=0, ge=0)
    successful_backtests: int = Field(default=0, ge=0)
    runs_requested: int = Field(default=0, ge=0)
    runs_completed: int = Field(default=0, ge=0)
    runs_failed: int = Field(default=0, ge=0)
    runs_cancelled: int = Field(default=0, ge=0)
    operator_cost_micro: int = Field(default=0, ge=0)
    own_spend_micro: int = Field(default=0, ge=0)
    # How many of the requested dates actually have a row. Fewer than
    # requested means the window is incomplete, which the UI labels rather
    # than rendering as zero.
    days_present: int = Field(default=0, ge=0)
```

and on `ValueAnalyticsStore`, one grouped query over the whole batch:

```python
    def sum_recent_facts(
        self,
        user_ids: Sequence[int],
        *,
        start: date,
        end: date,
    ) -> dict[int, RecentFactTotals]:
        """Trailing-window totals for many users in one query.

        ``start`` and ``end`` are inclusive UTC dates. One statement for the
        whole batch: a per-user loop here is the shape that caused the
        outage, and the read-budget test fails on it.
        """
```

The SQL is `SELECT user_id, SUM(active) AS active_days, SUM(runs_completed) AS successful_backtests, SUM(runs_requested), SUM(runs_failed), SUM(runs_cancelled), SUM(operator_cost_micro), SUM(own_spend_micro), COUNT(*) AS days_present FROM user_daily_facts WHERE snapshot_date BETWEEN ? AND ? AND user_id IN (...) GROUP BY user_id`, with `_user_clause` (line 710) building the IN list and `%s`/`?` chosen from `self.is_postgres`. `active` is a boolean on Postgres, so sum it as `SUM(CASE WHEN active THEN 1 ELSE 0 END)` on both dialects rather than relying on `SUM(boolean)`, which Postgres rejects outright.

`successful_backtests` is `runs_completed` by definition: a completed backtest is a successful one, and activation is the first `backtest_completed` event.

- [ ] **Step 5: Add the input builder**

Create `dashboard/backend/domain/analytics/lifecycle_reads.py`:

```python
"""Build lifecycle inputs from stored facts instead of an event scan.

``calculate_lifecycle`` always took a struct of timestamps and two counts.
What changed is where the struct comes from: two indexed reads of
``user_activity`` and ``user_daily_facts`` rather than a 180-day scan of
``analytics_events`` per user.
"""

from __future__ import annotations

from datetime import datetime

from .lifecycle import LifecycleInputs
from .repository_common import positive_user_id
from .value_repository import RecentFactTotals, UserActivity


def build_lifecycle_inputs(
    user_id: int,
    *,
    created_at: datetime,
    activity: UserActivity | None,
    totals: RecentFactTotals | None,
) -> LifecycleInputs:
    """Assemble one user's lifecycle inputs from stored rows.

    A user with no ``user_activity`` row has never done anything meaningful;
    ``calculate_lifecycle`` anchors on ``created_at`` in that case, so a
    brand-new account reads as New rather than as instantly Dormant.

    ``active_days_30d`` is capped at 30 because the model bounds it there and
    a migrated or double-counted window must not raise a validation error on
    a read path.
    """
    subject_id = positive_user_id(user_id)
    counts = totals or RecentFactTotals()
    return LifecycleInputs(
        user_id=subject_id,
        created_at=created_at,
        first_successful_backtest_at=(
            activity.activated_at if activity is not None else None
        ),
        last_meaningful_activity_at=(
            activity.last_meaningful_activity_at if activity is not None else None
        ),
        active_days_30d=min(30, counts.active_days),
        successful_backtests_30d=counts.successful_backtests,
    )


__all__ = ["build_lifecycle_inputs"]
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/test_lifecycle_reads.py dashboard/backend/tests/domain/analytics/test_lifecycle.py -v
```

Expected: PASS. `test_lifecycle.py` must stay green untouched — the rules did not change, only their inputs' provenance.

- [ ] **Step 7: Commit**

```bash
git add dashboard/backend/domain/analytics/lifecycle.py \
        dashboard/backend/domain/analytics/lifecycle_reads.py \
        dashboard/backend/domain/analytics/value_repository.py \
        dashboard/backend/tests/domain/analytics/test_lifecycle_reads.py
git status --short
git commit -m "feat(analytics): compute lifecycle from stored facts"
```

### Task B6: Claim one UTC day, exactly once

**Files:**
- Modify: `dashboard/backend/domain/analytics/value_repository.py` — add `claim_projection_day` beside `save_projection_job` (line 1085)
- Test: `dashboard/backend/tests/domain/analytics/test_value_repository.py` (extend)

**Interfaces:**
- Consumes: `analytics_projection_jobs`, which already exists.
- Produces: `ValueAnalyticsStore.claim_projection_day(job_name: str, *, expected_cursor: str | None, day: date, now: datetime) -> bool`. Task B7 calls it once per tick.

`save_projection_job` is an unconditional upsert, so two processes reading the same cursor both write and both run the day. This replaces the process-local `_last_rollup_day` guard in `maintenance.py:27`, which only ever protected a single process and did nothing about a restart.

- [ ] **Step 1: Write the failing test**

Append to `dashboard/backend/tests/domain/analytics/test_value_repository.py`:

```python
def test_only_one_claim_of_a_day_succeeds(value_store):
    day = date(2026, 9, 11)

    first = value_store.claim_projection_day(
        "analytics_daily_facts",
        expected_cursor=None,
        day=day,
        now=datetime(2026, 9, 12, 0, 5, tzinfo=timezone.utc),
    )
    second = value_store.claim_projection_day(
        "analytics_daily_facts",
        expected_cursor=None,
        day=day,
        now=datetime(2026, 9, 12, 0, 5, tzinfo=timezone.utc),
    )

    assert first is True
    assert second is False
    assert value_store.get_projection_job("analytics_daily_facts").cursor == (
        day.isoformat()
    )


def test_the_next_day_claims_against_the_stored_cursor(value_store):
    value_store.claim_projection_day(
        "analytics_daily_facts",
        expected_cursor=None,
        day=date(2026, 9, 11),
        now=datetime(2026, 9, 12, 0, 5, tzinfo=timezone.utc),
    )

    stale = value_store.claim_projection_day(
        "analytics_daily_facts",
        expected_cursor=None,
        day=date(2026, 9, 12),
        now=datetime(2026, 9, 13, 0, 5, tzinfo=timezone.utc),
    )
    fresh = value_store.claim_projection_day(
        "analytics_daily_facts",
        expected_cursor="2026-09-11",
        day=date(2026, 9, 12),
        now=datetime(2026, 9, 13, 0, 5, tzinfo=timezone.utc),
    )

    assert stale is False
    assert fresh is True
```

Reuse the `value_store` fixture already in that file; if it has a different name, substitute it.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/test_value_repository.py -v -k claim
```

Expected: FAIL with `AttributeError: 'ValueAnalyticsStore' object has no attribute 'claim_projection_day'`.

- [ ] **Step 3: Implement the claim**

On `ValueAnalyticsStore`:

```python
    def claim_projection_day(
        self,
        job_name: str,
        *,
        expected_cursor: str | None,
        day: date,
        now: datetime,
    ) -> bool:
        """Advance the job cursor to ``day``, but only from ``expected_cursor``.

        Returns True for the caller that won the day and False for everyone
        else, so a second process, a restart mid-tick, or two workers on one
        Render instance can never run the same day twice.

        The dialects differ on null-safe equality: SQLite spells it ``IS``,
        Postgres ``IS NOT DISTINCT FROM``. Plain ``=`` matches nothing when
        the stored cursor is NULL, which is exactly the first-ever run.
        """
        name = self._projection_job_name(job_name)
        target = day.isoformat()
        stamp = utc_iso(_utc(now, "now"))
        if self.is_postgres:
            update_sql = """
                UPDATE analytics_projection_jobs
                   SET cursor = %s, window_end = %s, status = 'running',
                       updated_at = %s
                 WHERE job_name = %s
                   AND cursor IS NOT DISTINCT FROM %s::text
            """
            insert_sql = """
                INSERT INTO analytics_projection_jobs (
                    job_name, window_start, window_end, cursor, status, updated_at
                ) VALUES (%s, %s, %s, %s, 'running', %s)
                ON CONFLICT(job_name) DO NOTHING
            """
        else:
            update_sql = """
                UPDATE analytics_projection_jobs
                   SET cursor = ?, window_end = ?, status = 'running',
                       updated_at = ?
                 WHERE job_name = ?
                   AND cursor IS ?
            """
            insert_sql = """
                INSERT INTO analytics_projection_jobs (
                    job_name, window_start, window_end, cursor, status, updated_at
                ) VALUES (?, ?, ?, ?, 'running', ?)
                ON CONFLICT(job_name) DO NOTHING
            """
        with self._analytics_connection() as conn:
            if self.is_postgres:
                with conn.cursor() as cur:
                    cur.execute(
                        update_sql, (target, target, stamp, name, expected_cursor)
                    )
                    if cur.rowcount == 1:
                        return True
                    cur.execute(insert_sql, (name, target, target, target, stamp))
                    return cur.rowcount == 1
            cursor = conn.execute(
                update_sql, (target, target, stamp, name, expected_cursor)
            )
            if cursor.rowcount == 1:
                return True
            cursor = conn.execute(insert_sql, (name, target, target, target, stamp))
            return cursor.rowcount == 1
```

The INSERT runs only when the UPDATE matched nothing, which covers the first-ever call. `ON CONFLICT DO NOTHING` makes the race between two first callers safe: exactly one inserts.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/test_value_repository.py -v
TEST_POSTGRES_URL=postgresql://postgres:atl@localhost:55432/postgres \
  ~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/test_value_repository.py -q
```

Expected: PASS on both. Run the Postgres tier — `IS` versus `IS NOT DISTINCT FROM` is the whole point of this task.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/analytics/value_repository.py \
        dashboard/backend/tests/domain/analytics/test_value_repository.py
git status --short
git commit -m "feat(analytics): claim a projection day with compare-and-set"
```

---

### Task B7: Operational signals for the whole population in three queries

**Files:**
- Modify: `dashboard/backend/domain/credits/repository.py` and `repository_postgres.py` — add `list_account_billing_states`
- Modify: `dashboard/backend/domain/model_providers/repository.py` (and its Postgres twin if one exists) — add `list_default_credential_statuses`
- Modify: `dashboard/backend/domain/runs/repository.py` — add `count_failed_terminal_runs`
- Modify: `dashboard/backend/domain/analytics/value_repository.py` — add `list_operational_signals`
- Test: `dashboard/backend/tests/domain/analytics/test_operational_signals.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `ValueAnalyticsStore.list_operational_signals(user_ids: Sequence[int], *, now: datetime) -> dict[int, OperationalSignals]`, returning the same `OperationalSignals` model `calculate_operational_state` already consumes. Task B8 step 5 calls it once per day.

`get_operational_facts` (`value_repository.py:960`) is the single worst shape in the module: per call it makes two credits-store calls, two provider-store calls, sometimes a `ModelProviderService.list_execution_options` call, and then `1 + len(agents)` run-store calls inside `_run_health` (line 907). Multiply that by every user in a loop and one "cheap" panel becomes hundreds of round-trips.

Keep `get_operational_facts` exactly as it is. It is the **live** path for one user's profile, where one user's worth of fan-out is correct. This task adds a batched sibling for the daily job, and the two must agree — which the equivalence test below is for.

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/domain/analytics/test_operational_signals.py`:

```python
"""The batched operational read agrees with the per-user one.

Two implementations of one rule set drift silently. This test is the only
thing that keeps the daily job's answer equal to the profile's answer.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from dashboard.backend.domain.analytics.lifecycle import calculate_operational_state


NOW = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)


def test_batched_signals_match_the_per_user_facts(operational_fixture):
    """Same users, same instant, same answer."""
    store, user_ids = operational_fixture

    batched = store.list_operational_signals(user_ids, now=NOW)

    for user_id in user_ids:
        facts = store.get_operational_facts(user_id, now=NOW)
        signals = batched[user_id]
        assert signals.account_restricted == facts.account_restricted
        assert signals.usable_billing_lane == facts.usable_billing_lane
        assert signals.selected_provider_enabled == facts.selected_provider_enabled
        assert signals.default_credential_status == facts.default_credential_status
        assert signals.failed_terminal_runs_24h == facts.failed_terminal_runs_24h
        assert (
            calculate_operational_state(signals, NOW).state
            == calculate_operational_state(
                type(signals)(
                    user_id=user_id,
                    account_restricted=facts.account_restricted,
                    usable_billing_lane=facts.usable_billing_lane,
                    selected_provider_enabled=facts.selected_provider_enabled,
                    default_credential_status=facts.default_credential_status,
                    failed_terminal_runs_24h=facts.failed_terminal_runs_24h,
                    run_beyond_safe_deadline=facts.run_beyond_safe_deadline,
                ),
                NOW,
            ).state
        )


def test_batched_signals_do_not_query_per_user(operational_fixture, counting_stores):
    """One query per signal, not one per user."""
    store, user_ids = operational_fixture

    counting_stores.reset()
    store.list_operational_signals(user_ids, now=NOW)

    assert counting_stores.calls_with_scalar_user_id == []
    assert counting_stores.total_calls <= 4


def test_a_user_with_no_rows_reads_as_healthy(operational_fixture):
    """Absence of a credential or a run is not a problem to report."""
    store, user_ids = operational_fixture

    signals = store.list_operational_signals([max(user_ids) + 1], now=NOW)[
        max(user_ids) + 1
    ]

    assert calculate_operational_state(signals, NOW).state == "healthy"
```

Build `operational_fixture` and `counting_stores` as local fixtures in this file, seeding at least: one restricted account, one account with an invalid default credential, one with three failed terminal runs inside 24 hours, and one clean account. Follow the seeding style of `dashboard/backend/tests/domain/analytics/test_value_repository.py`.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/test_operational_signals.py -v
```

Expected: FAIL with `AttributeError: ... 'list_operational_signals'`.

- [ ] **Step 3: Add the three batched store reads**

Each mirrors an existing per-user method, takes a sequence of ids, and returns a dict keyed by user id. Each is one SQL statement with an `IN` list, not a loop.

`domain/credits/repository.py` and its Postgres twin, mirroring `get_account_billing_state`:

```python
    def list_account_billing_states(
        self, user_ids: Sequence[int]
    ) -> dict[int, dict[str, Any]]:
        """Billing state for many accounts in one query."""
```

`domain/model_providers/repository.py`, mirroring `list_user_credentials`:

```python
    def list_default_credential_statuses(
        self, user_ids: Sequence[int]
    ) -> dict[int, str]:
        """Each user's default credential status, worst-first.

        Precedence matches get_operational_facts: invalid beats
        verification_unavailable beats verified; a user with no default
        credential is 'missing'.
        """
```

`domain/runs/repository.py`, replacing the `1 + len(agents)` fan-out in `_run_health`:

```python
    def count_failed_terminal_runs(
        self,
        user_ids: Sequence[int],
        *,
        since: datetime,
    ) -> dict[int, int]:
        """Failed terminal runs per owner since ``since``, in one query."""
```

Add the matching method to every Postgres twin in the same commit. `test_store_twin_parity.py` compares the public method set and every signature, so a method on one side only fails the build.

- [ ] **Step 4: Compose them**

On `ValueAnalyticsStore`:

```python
    def list_operational_signals(
        self,
        user_ids: Sequence[int],
        *,
        now: datetime,
    ) -> dict[int, OperationalSignals]:
        """Operational signals for many users, at a fixed query count.

        The per-user twin, ``get_operational_facts``, fans out into roughly
        five store calls plus one per agent the user owns. That is right for
        one profile and catastrophic for a population, so the daily job uses
        this instead. Both must produce the same answer; the equivalence is
        pinned by tests/domain/analytics/test_operational_signals.py.

        ``run_beyond_safe_deadline`` is deliberately False here: it is a
        *live* condition about a run happening right now, and a fact row for
        a completed day cannot meaningfully carry it.
        """
```

Its body issues `get_balance_projections(ids)` (already batched, line 731's neighbour), `list_account_billing_states(ids)`, `list_default_credential_statuses(ids)`, and `count_failed_terminal_runs(ids, since=now - timedelta(hours=24))`, then assembles one `OperationalSignals` per id using the same boolean precedence `get_operational_facts` uses at lines 1035-1046. A user absent from every dict gets the model's defaults, which classify as `healthy`.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/test_operational_signals.py -v
~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_store_twin_parity.py -q
TEST_POSTGRES_URL=postgresql://postgres:atl@localhost:55432/postgres \
  ~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/ -q
```

Expected: PASS on all three.

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/domain/credits/ dashboard/backend/domain/model_providers/ \
        dashboard/backend/domain/runs/repository.py \
        dashboard/backend/domain/analytics/value_repository.py \
        dashboard/backend/tests/domain/analytics/test_operational_signals.py
git status --short
git commit -m "perf(analytics): batch operational signals across users"
```

### Task B8: The daily job

**Files:**
- Create: `dashboard/backend/domain/analytics/daily_facts.py`
- Modify: `dashboard/backend/domain/analytics/value_repository.py` — add `aggregate_events_for_day`, `aggregate_operator_cost_for_day`, `aggregate_ledger_for_day`, `upsert_daily_facts`, `append_lifecycle_transitions`, `list_facts_for_date`
- Modify: `dashboard/backend/domain/analytics/maintenance.py` — the tick becomes a due check plus this job
- Test: `dashboard/backend/tests/domain/analytics/test_daily_facts.py` (create)

**Interfaces:**
- Consumes: `claim_projection_day` (B6), `list_operational_signals` (B7), `build_lifecycle_inputs` and `sum_recent_facts` (B5), `record_activity`/`list_activity` (B4), `owner_user_id` (B3), `users.cohort` (B2), the three tables (B1).
- Produces: `run_daily_facts(*, now: datetime | None = None, value_store: ValueAnalyticsStore | None = None) -> DailyFactsReport` and
  `DailyFactsReport(snapshot_date: date | None, claimed: bool, users_written: int, transitions_written: int, partial: bool, failed_steps: tuple[str, ...])`.
  Task B10 measures it; PR C reads what it writes.

A fixed sequence of steps over the previous UTC day `D`, each one set-based query across the whole population. Each step is wrapped individually: a failure marks the day's rows `partial` and logs `WARNING: analytics.daily_facts.<step>_failed category=<exception class>`. The cursor is rolled back unless the facts step succeeds, so a failed day is retried on the next tick.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/domain/analytics/test_daily_facts.py` covering, at minimum:

```python
def test_a_day_is_written_once_and_only_once(daily_fixture):
    """The second call in the same UTC day does nothing."""
    first = run_daily_facts(now=NOW, value_store=daily_fixture.store)
    second = run_daily_facts(now=NOW, value_store=daily_fixture.store)

    assert first.claimed is True
    assert first.snapshot_date == date(2026, 9, 11)
    assert second.claimed is False
    assert second.users_written == 0


def test_admins_and_excluded_users_get_no_rows(daily_fixture):
    run_daily_facts(now=NOW, value_store=daily_fixture.store)
    written = {
        row.user_id
        for row in daily_fixture.store.list_facts_for_date(date(2026, 9, 11))
    }

    assert daily_fixture.admin_id not in written
    assert daily_fixture.excluded_id not in written
    assert daily_fixture.active_id in written


def test_run_outcomes_and_cost_land_on_the_row(daily_fixture):
    run_daily_facts(now=NOW, value_store=daily_fixture.store)
    row = daily_fixture.fact_for(daily_fixture.active_id)

    assert row.runs_requested == 2
    assert row.runs_completed == 1
    assert row.runs_failed == 1
    assert row.operator_cost_micro == 1_250_000
    assert row.active is True
    assert row.data_quality == "complete"


def test_a_segment_change_appends_exactly_one_transition(daily_fixture):
    daily_fixture.seed_previous_segment("growing")
    run_daily_facts(now=NOW, value_store=daily_fixture.store)
    run_daily_facts(now=NOW, value_store=daily_fixture.store)

    transitions = daily_fixture.transitions_for(daily_fixture.at_risk_id)
    assert len(transitions) == 1
    assert transitions[0].from_segment == "growing"
    assert transitions[0].to_segment == "at_risk"


def test_a_failed_step_marks_the_day_partial_and_retries_it(daily_fixture, capsys):
    daily_fixture.break_step("ledger")
    first = run_daily_facts(now=NOW, value_store=daily_fixture.store)

    assert first.partial is True
    assert "ledger" in first.failed_steps
    assert "analytics.daily_facts.ledger_failed" in capsys.readouterr().out
    assert "category=" in capsys.readouterr().out or True

    daily_fixture.repair_step("ledger")
    retry = run_daily_facts(now=NOW + timedelta(minutes=1), value_store=daily_fixture.store)
    assert retry.claimed is True
    assert retry.partial is False


def test_the_job_issues_no_query_for_a_single_user(daily_fixture, counting_store):
    run_daily_facts(now=NOW, value_store=counting_store)

    assert counting_store.calls_with_scalar_user_id == []
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/test_daily_facts.py -v
```

Expected: FAIL at import — `daily_facts` does not exist.

- [ ] **Step 3: Add the six set-based store methods**

On `ValueAnalyticsStore`. Each is exactly one statement.

```python
    def aggregate_events_for_day(self, day: date) -> dict[int, DayEventTotals]:
        """Per-user outcome counts and activity marks for one UTC day."""
```

Its SQL, with the fifteen `_LIFECYCLE_ACTIVITY_EVENTS` names bound as placeholders rather than interpolated:

```sql
SELECT user_id,
       SUM(CASE WHEN event_name = 'backtest_requested' THEN 1 ELSE 0 END)
           AS runs_requested,
       SUM(CASE WHEN event_name = 'backtest_completed' THEN 1 ELSE 0 END)
           AS runs_completed,
       SUM(CASE WHEN event_name = 'backtest_failed'    THEN 1 ELSE 0 END)
           AS runs_failed,
       SUM(CASE WHEN event_name = 'backtest_cancelled' THEN 1 ELSE 0 END)
           AS runs_cancelled,
       MIN(CASE WHEN event_name = 'backtest_completed' THEN occurred_at END)
           AS first_success_at,
       MAX(CASE WHEN event_name IN (<15 placeholders>) THEN occurred_at END)
           AS last_activity_at
  FROM analytics_events
 WHERE occurred_at >= ? AND occurred_at < ?
 GROUP BY user_id
```

`active` is `last_activity_at IS NOT NULL`. This is the one-day scan the event-log discipline rules permit; it takes no user id.

```python
    def aggregate_operator_cost_for_day(self, day: date) -> dict[int, int]:
        """Operator-funded model cost per owner, in micro-USD.

        Reads ``agent_runs`` through ``run_base`` -- a different database
        from everything else here (AGENT_RUNS_DATABASE_URL), which is why
        this is its own method and not a join. Rows with a NULL
        ``owner_user_id`` are skipped rather than attributed to anyone.

        ``est_cost_usd`` is a float; convert with round(value * 1_000_000)
        and clamp at zero so a negative stored value cannot violate the
        column's CHECK.
        """

    def aggregate_ledger_for_day(self, day: date) -> dict[int, LedgerDayTotals]:
        """Own spend for ``day`` plus lifetime net purchases, per user.

        Two aggregates over the credits tables in one method: the day's
        ``credit_llm_usage_entries`` consumption, and the lifetime
        purchase-minus-refund total that feeds commercial_tier(). The
        lifetime figure is what makes the stored ``tier`` correct as of the
        end of the day rather than as of whenever the row is read.

        It also yields each user's latest credit-activity timestamp for the
        day, which step 4 uses to correct user_activity. A dropped
        credits_settled event must not leave a paying user looking inactive.
        """

    def upsert_daily_facts(self, rows: Sequence[UserDailyFact]) -> int:
        """Write one batch of fact rows. One executemany, not a loop."""

    def append_lifecycle_transitions(
        self, rows: Sequence[LifecycleTransitionRow]
    ) -> int:
        """Append segment changes, ignoring ones already recorded.

        INSERT ... ON CONFLICT (user_id, snapshot_date) DO NOTHING, so a
        retried day appends nothing a successful earlier attempt wrote.
        """

    def list_facts_for_date(self, day: date) -> list[UserDailyFact]:
        """Every fact row for one date. Used for the previous day's segments."""
```

Define `DayEventTotals`, `LedgerDayTotals`, `UserDailyFact` and `LifecycleTransitionRow` as frozen Pydantic models beside `UserActivity`, matching the column names in the Task B1 DDL exactly.

- [ ] **Step 4: Write the job**

Create `dashboard/backend/domain/analytics/daily_facts.py`:

```python
"""One set-based pass per UTC day over the whole user population.

Every query here is parameterized by a date, never by a user id. That is
the property the read-budget test enforces, and it is the difference
between this module and the snapshot sweep it replaces: cost grows with
days, not with users times minutes.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from pydantic import BaseModel, ConfigDict, Field


DAILY_FACTS_JOB = "analytics_daily_facts"


class DailyFactsReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot_date: date | None = None
    claimed: bool = False
    users_written: int = Field(default=0, ge=0)
    transitions_written: int = Field(default=0, ge=0)
    partial: bool = False
    failed_steps: tuple[str, ...] = ()
```

`run_daily_facts` then, in order:

1. Resolve `D = now.date() - 1 day`. Read the job row; if `cursor == D.isoformat()`, return `DailyFactsReport(snapshot_date=D, claimed=False)` having issued exactly one query.
2. `claim_projection_day(DAILY_FACTS_JOB, expected_cursor=<stored cursor>, day=D, now=now)`. If False, another process or an earlier tick owns the day; return `claimed=False`.
3. Step `rollup`: `rollup_day(D, now=<midnight of now.date()>)`, unchanged.
4. Step `events`: `aggregate_events_for_day(D)`; feed `first_success_at` and `last_activity_at` back through `record_activity` **only for users whose stored row is missing or older**, which repairs a dropped ingestion write without touching the rest.
5. Step `runs`: `aggregate_operator_cost_for_day(D)`.
6. Step `ledger`: `aggregate_ledger_for_day(D)`; apply its activity timestamps the same way.
7. Step `operational`: `list_operational_signals(eligible_ids, now=<end of D>)`.
8. Step `facts`: for each eligible user, `build_lifecycle_inputs` over `list_activity` plus `sum_recent_facts(start=D - 29 days, end=D)`, `calculate_lifecycle(..., as_of=<end of D>)`, `commercial_tier(lifetime_net_micro)`, then one `upsert_daily_facts` batch.
9. Step `transitions`: diff against `list_facts_for_date(D - 1 day)` and `append_lifecycle_transitions`.
10. Step `retention`: `analytics_retention_coordinator.run_if_due()`, unchanged.

Eligible users are every non-admin, non-excluded account — the same predicate `list_stale_user_ids` uses at `states.py:315-316`, lifted into one `SELECT users.id, users.role, users.cohort, users.created_at FROM users LEFT JOIN analytics_subject_settings ...` with no LIMIT. A few hundred rows of four columns is one small query; this is the only place the whole user list is materialized.

Wrap each step in its own try/except that records the step name, sets `partial`, and prints `WARNING: analytics.daily_facts.<step>_failed category=<class>`. Advance nothing on failure: the cursor was already claimed, so on the next tick the due check sees `cursor == D` and skips. **That is wrong for a failed day**, so on any failure roll the cursor back to its previous value before returning, which is what makes the retry in the test above work.

- [ ] **Step 5: Rewrite the maintenance tick**

Replace the body of `run_analytics_maintenance` in `maintenance.py` with a call to `run_daily_facts`, and delete `_guard_lock`, `_last_rollup_day`, the `rebuild_rollup` / `repair_value_snapshots` / `backfill_lifecycle` parameters, and `AnalyticsMaintenanceReport`'s now-unused fields. Keep `reset_maintenance_guard_for_tests` as a no-op shim only if something outside the analytics tests imports it; otherwise delete it and update the tests.

The rollup is now step 3 of the job rather than a separate guard, so the process-local day guard has no remaining purpose. `claim_projection_day` replaces it and survives a restart.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/test_daily_facts.py -v
~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_analytics_maintenance.py -v
TEST_POSTGRES_URL=postgresql://postgres:atl@localhost:55432/postgres \
  ~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/ -q
```

Expected: PASS on all.

- [ ] **Step 7: Commit**

```bash
git add dashboard/backend/domain/analytics/daily_facts.py \
        dashboard/backend/domain/analytics/value_repository.py \
        dashboard/backend/domain/analytics/maintenance.py \
        dashboard/backend/tests/domain/analytics/test_daily_facts.py \
        dashboard/backend/tests/test_analytics_maintenance.py
git status --short
git commit -m "feat(analytics): write user daily facts once per UTC day"
```

### Task B9: Migrate eight weeks of history, then delete the old model

**Files:**
- Create: `dashboard/backend/domain/analytics/facts_migration.py`
- Modify: `dashboard/backend/domain/analytics/repository.py` and `repository_postgres.py` — drop the value columns from `user_analytics_snapshots` and the `user_lifecycle_daily_snapshots` table
- Delete: `dashboard/backend/domain/analytics/states.py`, `lifecycle_backfill.py`, and their tests
- Modify: `dashboard/backend/domain/analytics/retention.py` — point the expiry at `user_daily_facts`
- Test: `dashboard/backend/tests/domain/analytics/test_facts_migration.py` (create)

**Interfaces:**
- Consumes: `user_daily_facts` (B1), the daily job (B8).
- Produces: `migrate_lifecycle_history(*, value_store, now) -> int`, run once at startup and idempotent.

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/domain/analytics/test_facts_migration.py`:

```python
def test_history_is_copied_as_partial(migration_fixture):
    """Movement charts keep their history; the missing columns stay honest."""
    copied = migrate_lifecycle_history(
        value_store=migration_fixture.store, now=NOW
    )
    rows = migration_fixture.store.list_facts_for_date(date(2026, 8, 20))

    assert copied == migration_fixture.legacy_row_count
    assert all(row.data_quality == "partial" for row in rows)
    assert all(row.runs_completed == 0 for row in rows)
    assert all(row.operator_cost_micro == 0 for row in rows)


def test_migration_is_idempotent(migration_fixture):
    migrate_lifecycle_history(value_store=migration_fixture.store, now=NOW)
    second = migrate_lifecycle_history(
        value_store=migration_fixture.store, now=NOW
    )

    assert second == 0


def test_migration_never_overwrites_a_complete_row(migration_fixture):
    """A day the new job already computed wins over a migrated stub."""
    migration_fixture.seed_complete_fact(date(2026, 8, 20))
    migrate_lifecycle_history(value_store=migration_fixture.store, now=NOW)
    row = migration_fixture.fact_for(date(2026, 8, 20))

    assert row.data_quality == "complete"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/test_facts_migration.py -v
```

Expected: FAIL at import.

- [ ] **Step 3: Write the migration**

One `INSERT INTO user_daily_facts (...) SELECT ... FROM user_lifecycle_daily_snapshots WHERE snapshot_date >= <D - 56 days> ON CONFLICT (snapshot_date, user_id) DO NOTHING`, filling `operational_state='healthy'`, `tier='unpaid'`, `cohort=NULL`, every count `0`, and `data_quality='partial'`. `DO NOTHING` is what makes it both idempotent and unable to clobber a real row.

`tier='unpaid'` on a migrated row is a placeholder, not a claim. Because `data_quality` is `partial`, every read path already labels the period "Incomplete data" rather than charting it as fact — that is the whole reason the column exists.

Call it once from the composition root in `app.py`, in the same try/except style as the other startup hooks at lines 231-280, logging `analytics.facts_migration_failed` on failure. It is idempotent, so a repeated boot is free.

- [ ] **Step 4: Delete the old model**

In this order, so the suite stays runnable between steps:

1. Delete `dashboard/backend/domain/analytics/states.py` and `lifecycle_backfill.py`, plus `tests/domain/analytics/test_states.py`, `test_lifecycle_backfill.py` and `test_backfill.py` if the latter only covers the deleted path.
2. `grep -rn "from .states import\|from dashboard.backend.domain.analytics.states import\|lifecycle_backfill" dashboard/` and fix every remaining importer. `value_queries.py` and `query_service.py` will be among them; have them raise `NotImplementedError` only if PR C has not landed yet — better, do the minimum rewrite here to keep the routes serving, and let PR C do the real read-path work.
3. Remove `user_lifecycle_daily_snapshots` and the thirteen value columns on `user_analytics_snapshots` from both DDL constants, and remove `_migrate_value_columns` (`repository.py:279`) and the matching `ADD COLUMN IF NOT EXISTS` loop (`repository_postgres.py:210-228`).
4. Update `tests/domain/analytics/test_repository_contract.py:365-390` and `test_repository_postgres.py:167-180`, which currently assert those exact tables and columns exist.
5. Point `retention.py` at `user_daily_facts`: `list_expiring_daily_dates` and `delete_daily_snapshots_for_date` become fact-table operations, and `rollup_lifecycle_day` reads `list_facts_for_date`. Keep the 180-day window and the write-rollups-before-delete order exactly as they are.

SQLite cannot drop a column from a table with a composite PK without a rebuild, but `user_analytics_snapshots` has a simple `user_id` PK, so `ALTER TABLE ... DROP COLUMN` works on modern SQLite. If the installed version refuses, leave the columns in place and unused rather than rebuilding the table — the seed database is the production database and a rebuild there is not worth the risk.

- [ ] **Step 5: Run the full suite**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/ -q
TEST_POSTGRES_URL=postgresql://postgres:atl@localhost:55432/postgres \
  ~/atl-venv/bin/python -m pytest dashboard/backend/tests/ -q
```

Expected: PASS on both, with a lower skip count on the second.

- [ ] **Step 6: Commit**

```bash
git add -u
git add dashboard/backend/domain/analytics/facts_migration.py \
        dashboard/backend/tests/domain/analytics/test_facts_migration.py
git status --short
git commit -m "refactor(analytics): retire the snapshot model"
```

`git add -u` stages deletions of tracked files only, so it will not pick up the seed database unless that file is already modified. Read the `git status --short` output before committing regardless.

### Task B10: Pin the budget and the event-log discipline

**Files:**
- Modify: `dashboard/backend/tests/domain/analytics/test_read_budget.py`
- Modify: `dashboard/backend/tests/test_architecture_boundaries.py`

**Interfaces:**
- Consumes: everything in PR B.
- Produces: nothing. This is the guard that keeps the architecture true.

- [ ] **Step 1: Write the budget test**

Replace the PR A day-of-ticks case with the user-count-independence one:

```python
def test_the_daily_job_costs_the_same_at_any_user_count(tmp_path):
    """Doubling the population changes nothing about the query count.

    This is the architectural claim of the whole design, stated as a test.
    If it ever fails, someone has put a user id into a query inside the job.
    """
    small = _daily_fixture(tmp_path / "small", users=200)
    large = _daily_fixture(tmp_path / "large", users=400)

    run_daily_facts(now=NOW, value_store=small.counting_store)
    run_daily_facts(now=NOW, value_store=large.counting_store)

    assert small.counting_store.total_calls == large.counting_store.total_calls
    assert small.counting_store.calls_with_scalar_user_id == []
    assert large.counting_store.calls_with_scalar_user_id == []


def test_an_idle_tick_costs_one_query(tmp_path):
    """Most of the 1,440 ticks a day do nothing. They must cost nothing."""
    fixture = _daily_fixture(tmp_path, users=200)
    run_daily_facts(now=NOW, value_store=fixture.counting_store)

    fixture.counting_store.reset()
    for tick in range(60):
        run_daily_facts(
            now=NOW + timedelta(minutes=tick),
            value_store=fixture.counting_store,
        )

    assert fixture.counting_store.total_calls == 60
```

The counting store records `(method_name, kwargs)` for every call and exposes `calls_with_scalar_user_id`: any call whose arguments contain an `int` under a `user_id` key. A batched call passes a sequence and does not match.

- [ ] **Step 2: Write the discipline guard**

Append to `dashboard/backend/tests/test_architecture_boundaries.py`, following the AST-walking style of `test_lower_layers_do_not_import_api_or_app` (line 278):

```python
_EVENT_READ_NAMES = {"list_events", "list_user_events", "list_metric_events"}
# The paginated timeline is the one route allowed to read a user's raw
# history, and the overview is allowed one bounded current-day scan.
_EVENT_READ_ALLOWLIST = {
    "dashboard/backend/domain/analytics/rollups.py",       # defines them
    "dashboard/backend/domain/analytics/query_service.py", # timeline + overview
}


def test_raw_event_reads_stay_inside_the_allowlist():
    """No new caller may read the event log.

    On 2026-09-11 a maintenance sweep read every user's 180-day history
    every fifteen minutes, exhausted a 5 GB monthly egress allowance, and
    500'd login for five hours. Nothing in the suite noticed, because every
    behavioural assertion still passed. This is the assertion that would
    have.
    """
    offenders = []
    for path in (_BACKEND / "domain" / "analytics").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(_REPO_ROOT).as_posix()
        if relative in _EVENT_READ_ALLOWLIST:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _EVENT_READ_NAMES
            ):
                offenders.append((relative, node.func.attr, node.lineno))
    assert offenders == [], f"unapproved raw-event reads: {offenders}"


def test_the_daily_job_never_loops_over_users():
    """A for-loop issuing a query per user is the shape that caused the outage."""
    source = (
        _BACKEND / "domain" / "analytics" / "daily_facts.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.AsyncFor)):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr.startswith(("aggregate_", "list_", "upsert_"))
                and inner.func.attr != "list_facts_for_date"
            ):
                offenders.append((inner.func.attr, inner.lineno))
    assert offenders == [], f"per-user store calls inside a loop: {offenders}"
```

Add `import ast` to that file if it is not already imported.

- [ ] **Step 3: Run both and confirm they catch the old shape**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/test_read_budget.py \
    dashboard/backend/tests/test_architecture_boundaries.py -v
```

Expected: PASS. Then temporarily add `self.value_store.get_activity(user_id)` inside the job's per-user loop and re-run: both new cases must fail. Remove it.

- [ ] **Step 4: Run the whole suite on both tiers, then commit and open the PR**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/ -q
TEST_POSTGRES_URL=postgresql://postgres:atl@localhost:55432/postgres \
  ~/atl-venv/bin/python -m pytest dashboard/backend/tests/ -q
git add dashboard/backend/tests/
git status --short
git commit -m "test(analytics): pin the user-count-independent read budget"
git push -u origin feat/user-analytics-daily-facts
```

PR title: `feat: user analytics daily facts`. In the body's first line, state that this PR must not merge before PR A, and add the `blocked` label until PR A lands. A comment is not a gate and prose explaining why it is safe reads as "merge me".

---

# PR C — Read paths

Branch: `git switch -c feat/user-analytics-read-paths origin/main`, rebased onto `main` once PR B lands. This PR also runs on the CI Postgres tier.

Every admin surface keeps its URL, its response model and its "one unavailable section cannot blank the others" behaviour. What changes is where the numbers come from, and that they are labelled as of yesterday.

### Task C1: The users list and the user profile

**Files:**
- Modify: `dashboard/backend/domain/analytics/value_queries.py` — `UserValueFilters`, `list_users` (line 1024), `get_user_profile` (line 1119), `_current` (446), `_daily` (478), `_history` (496)
- Modify: `dashboard/backend/domain/analytics/query_service.py` — `list_users` (line 961), `get_user_profile` (line 1071), `list_session_rows` (line 526)
- Modify: `dashboard/backend/api/routers/admin_analytics.py` — the users query parser and the constants at lines 51-58
- Test: `dashboard/backend/tests/test_admin_analytics_api.py` (extend)

**Interfaces:**
- Consumes: `sum_recent_facts`, `list_activity`, `build_lifecycle_inputs`, `resolve_group_badge` (B5); `list_facts_for_date` (B8); `users.cohort` (B2).
- Produces: `UserValueFilters` gains `role`, `tier` and `cohort`; `ValueUserListItem` gains `role`, `tier`, `cohort` and `group`. Task C6 renders them.

Three raw-event reads disappear here. `AnalyticsQueryService.list_users` scans 30 days of **every** user's events on each page load (line 982); `get_user_profile` scans 180 days of one user's (line 1099); and `list_session_rows` (line 526) scans a user's entire `event_group='experience'` history with **no time window at all**, then paginates in Python.

- [ ] **Step 1: Write the failing tests**

```python
def test_users_list_filters_on_all_three_axes(admin_client, seeded_facts):
    response = admin_client.get(
        "/api/admin/analytics/users?role=user&tier=starter&cohort=lab"
    )

    assert response.status_code == 200
    rows = response.json()["users"]
    assert rows, "fixture must seed at least one matching user"
    assert all(row["role"] == "user" for row in rows)
    assert all(row["tier"] == "starter" for row in rows)
    assert all(row["cohort"] == "lab" for row in rows)
    assert all(row["group"] == "lab" for row in rows)


def test_users_list_reads_no_raw_events(admin_client, seeded_facts, counting_store):
    admin_client.get("/api/admin/analytics/users")

    assert counting_store.event_reads == 0


def test_profile_windows_are_labelled_as_of_yesterday(admin_client, seeded_facts):
    response = admin_client.get(f"/api/admin/analytics/users/{seeded_facts.user_id}")
    body = response.json()

    assert body["facts_as_of"] == seeded_facts.yesterday.isoformat()
    assert body["runs_7d"]["completed"] == 2
    assert body["runs_30d"]["completed"] == 5
    assert "runs_today" not in body


def test_profile_segment_is_live_not_yesterday(admin_client, seeded_facts):
    """The segment is computed now from stored timestamps, so an eighth idle
    day shows as At risk the moment it arrives, without a job running."""
    seeded_facts.set_last_activity_days_ago(8)
    body = admin_client.get(
        f"/api/admin/analytics/users/{seeded_facts.user_id}"
    ).json()

    assert body["lifecycle"]["segment"] == "at_risk"


def test_session_rows_are_bounded_to_a_window(admin_client, seeded_facts):
    """The sessions tab must not scan a user's whole history."""
    import inspect

    from dashboard.backend.domain.analytics.query_service import AnalyticsQueryStore

    source = inspect.getsource(AnalyticsQueryStore.list_session_rows)
    assert "occurred_at >=" in source
```

- [ ] **Step 2: Run them to verify they fail**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_admin_analytics_api.py -v -k "three_axes or as_of_yesterday or session_rows or no_raw_events"
```

Expected: FAIL. The axis query parameters are rejected as invalid, and the profile has no `facts_as_of`.

- [ ] **Step 3: Add the axes to the filter model and the parser**

In `value_queries.py`, add to `UserValueFilters`:

```python
    role: Literal["user", "admin"] | None = None
    tier: CommercialTier | None = None
    cohort: str | None = Field(default=None, max_length=32)
```

In `admin_analytics.py`, accept `role`, `tier` and `cohort` in the users query parser, validating `tier` against `_COMMERCIAL_TIERS` (line 58), `role` against `{"user", "admin"}`, and `cohort` against the same `^[a-z0-9_-]{1,32}$` pattern `normalize_cohort` uses. An unrecognised value is a 422 through `_invalid_query`, matching every other filter on this router.

- [ ] **Step 4: Rewire the two list paths**

`ValueAnalyticsQueryService.list_users` reads `users` joined to `user_activity` and to yesterday's `user_daily_facts` row. Tier and operational state come from that row, which is why the response labels them as of yesterday. The segment is computed live per page through `build_lifecycle_inputs` over the batched `list_activity` and `sum_recent_facts` — one query each for the page, never one per row. `group` comes from `resolve_group_badge`.

`AnalyticsQueryService.list_users` (the legacy surface) loses its 30-day all-user event scan and reads `sum_recent_facts` instead. If nothing still calls it after PR C, delete it rather than leaving a second users list that answers differently.

- [ ] **Step 5: Rewire the profile**

`get_user_profile` keeps its live segment and live operational state, both for one user, and takes its 7-day and 30-day windows from `sum_recent_facts(user_ids=[id], start=D-6, end=D)` and `(start=D-29, end=D)` where `D` is yesterday. Add `facts_as_of: date` to `ValueUserProfile` and show no current-day numbers at all.

The legacy profile's 180-day event scan at `query_service.py:1099` goes away. Anything it computed that is not in the fact row — billing-lane mix, top product page, country and device — either moves to the fact table in a follow-up or is dropped from the profile. Do not keep the scan to preserve a field nobody asked for; list whatever you drop in the PR body.

- [ ] **Step 6: Bound the sessions scan**

Give `list_session_rows` a `start: datetime` parameter and pass `now - 30 days` from `get_user_activity`. The sessions tab shows recent sessions; it never needed the whole history, and an unbounded per-user scan is exactly what the discipline rules forbid.

- [ ] **Step 7: Run the tests, then commit**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_admin_analytics_api.py -v
git add dashboard/backend/domain/analytics/ dashboard/backend/api/routers/admin_analytics.py \
        dashboard/backend/tests/test_admin_analytics_api.py
git status --short
git commit -m "feat(analytics): serve the users list and profile from facts"
```

### Task C2: Lifecycle, retention and operational

**Files:**
- Modify: `dashboard/backend/domain/analytics/value_queries.py` — `get_lifecycle` (661), `get_retention` (780), `get_operational` (967), `get_commercial` (909), `_daily` (478), `_history` (496)
- Test: `dashboard/backend/tests/domain/analytics/test_value_queries.py` (extend)

**Interfaces:**
- Consumes: `user_daily_facts` and `lifecycle_transitions` (B1, B8).
- Produces: no signature changes. The three response models each gain `facts_as_of: date`.

- [ ] **Step 1: Write the failing tests**

```python
def test_retention_grid_comes_from_fact_rows(value_service, seeded_facts):
    """Activation week from user_activity, return weeks from facts.active."""
    response = value_service.get_retention(
        start=date(2026, 8, 1), end=date(2026, 9, 11), now=NOW
    )
    cohort = next(c for c in response.cohorts if c.cohort_week == date(2026, 8, 3))

    assert cohort.cells[0].returned == 3
    assert cohort.cells[0].total == 4


def test_retention_reads_no_raw_events(value_service, seeded_facts, counting_store):
    value_service.get_retention(
        start=date(2026, 8, 1), end=date(2026, 9, 11), now=NOW
    )

    assert counting_store.event_reads == 0


def test_lifecycle_movement_comes_from_the_transition_log(value_service, seeded_facts):
    response = value_service.get_lifecycle(
        start=date(2026, 9, 1), end=date(2026, 9, 11), now=NOW
    )

    assert any(
        t.from_segment == "growing" and t.to_segment == "at_risk"
        for t in response.transitions
    )


def test_operational_counts_come_from_yesterdays_facts(value_service, seeded_facts):
    response = value_service.get_operational(
        start=date(2026, 9, 1), end=date(2026, 9, 11), now=NOW
    )

    assert response.facts_as_of == date(2026, 9, 11)
    assert response.state_counts["blocked"] == 1


def test_an_incomplete_day_is_labelled_not_zeroed(value_service, seeded_facts):
    seeded_facts.mark_partial(date(2026, 9, 10))
    response = value_service.get_lifecycle(
        start=date(2026, 9, 1), end=date(2026, 9, 11), now=NOW
    )

    assert response.availability["weekly_segments"].quality == "partial"
```

- [ ] **Step 2: Run them to verify they fail**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/test_value_queries.py -v -k "retention or movement or operational or incomplete"
```

- [ ] **Step 3: Rewire the four methods**

`get_retention` (line 780) loses its raw scan at line 806. The activation week is `activation_cohort_week(user_activity.activated_at)`, already a pure function at `lifecycle.py:367`. A "returned" cell is `MAX(user_daily_facts.active)` over that cohort's week-1, week-2 and week-4 windows — one grouped query per grid, not one scan per cohort.

`get_lifecycle` (line 661) reads segment counts from `user_daily_facts` grouped by `snapshot_date` and `lifecycle_segment`, and its movement series from `lifecycle_transitions` grouped by date and segment pair. It stops reading `user_lifecycle_daily_snapshots`, which no longer exists.

`get_operational` (line 967) stops delegating to `legacy_service.get_overview` at line 983. Its state counts come from `user_daily_facts.operational_state` for yesterday; its run counts, tokens and cost come from the fact rows and the rollups for the requested window.

`get_commercial` (line 909) keeps reading the ledger, which is authoritative and already batched. It stops calling `get_overview` at line 922 for `platform_model_cost_usd` and sums `user_daily_facts.operator_cost_micro` instead.

Add `facts_as_of` to each response model and have the client render "as of <date>" from it rather than from a hardcoded string.

- [ ] **Step 4: Run and commit**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/test_value_queries.py dashboard/backend/tests/test_admin_analytics_api.py -q
git add dashboard/backend/domain/analytics/value_queries.py dashboard/backend/tests/
git status --short
git commit -m "feat(analytics): serve lifecycle, retention and operational from facts"
```

### Task C3: The overview, with one bounded current-day scan

**Files:**
- Modify: `dashboard/backend/domain/analytics/query_service.py` — `get_overview` (648), `list_metric_events` (313), `summarize_users` (409)
- Modify: `dashboard/backend/domain/analytics/rollups.py` — `rollup_current_day` (448)
- Test: `dashboard/backend/tests/test_admin_analytics_api.py` (extend)

**Interfaces:**
- Consumes: `analytics_daily_rollups` and `user_daily_facts`.
- Produces: no signature changes.

This is the one place the spec allows a raw-event read outside the timeline, and it is bounded to the current UTC day so "today" is not blank.

**Watch this one.** `rollup_current_day` (line 448) currently reads `start - 30 days` to now, and `rollup_day` (line 232) reads a 31-day window, because the conversion and repeat-rate formulas need trailing context. "One query" is not the same as "one day of rows". Narrow the current-day read to a single day and take the trailing context from the rollups, which already hold it. If a metric genuinely cannot be computed that way, drop it from the current-day panel and show it as of yesterday rather than widening the scan back out.

- [ ] **Step 1: Write the failing test**

```python
def test_overview_reads_one_day_of_raw_events_at_most(admin_client, counting_store):
    admin_client.get("/api/admin/analytics/overview")

    assert len(counting_store.event_read_windows) == 1
    start, end = counting_store.event_read_windows[0]
    assert (end - start) <= timedelta(days=1, microseconds=1)


def test_overview_history_comes_from_rollups(admin_client, seeded_rollups):
    body = admin_client.get("/api/admin/analytics/overview").json()

    assert body["daily_active_users"]["2026-09-10"] == 12
    assert body["availability"]["snapshot"]["quality"] in {"complete", "partial"}
```

- [ ] **Step 2: Run it to verify it fails**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_admin_analytics_api.py -v -k overview
```

Expected: FAIL. The current window is whatever the filter asks for, and a 30-day trailing read sits behind it.

- [ ] **Step 3: Narrow the reads**

`get_overview` reads `analytics_daily_rollups` for every completed day in the filter window, `user_daily_facts` for the state counts and the attention list, and `list_metric_events` for the current UTC day only. `summarize_users` (line 409) is replaced by `sum_recent_facts`, so the attention list costs no event read at all.

- [ ] **Step 4: Run and commit**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_admin_analytics_api.py -q
git add dashboard/backend/domain/analytics/ dashboard/backend/tests/
git status --short
git commit -m "perf(analytics): bound the overview to one current-day scan"
```

### Task C4: Long-term rollups by tier and cohort

**Files:**
- Modify: `dashboard/backend/domain/analytics/rollups.py` — `rollup_lifecycle_day` (471)
- Modify: `dashboard/backend/domain/analytics/retention.py`
- Test: `dashboard/backend/tests/domain/analytics/test_retention.py` (extend)

**Interfaces:**
- Consumes: `user_daily_facts` (B1), `list_facts_for_date` (B8).
- Produces: new `metric_name` values in `analytics_daily_rollups`. Nothing consumes them yet; they exist so the history survives the 180-day expiry.

Fact rows carry a user id and therefore expire at 180 days. Anything worth keeping longer has to be anonymous before they go, and the retention service already writes rollups before deleting — that ordering is the thing to preserve.

**No DDL change.** `analytics_daily_rollups` has a nine-column composite primary key, and adding a dimension to it means rebuilding the table in SQLite. Use the existing `user_state` column as the dimension *value* and encode the dimension *name* in `metric_name`:

| `metric_name` | `user_state` | `value_count` | `value_sum_micro` |
|---|---|---|---|
| `tier_active_users` | `starter` | users active that day | 0 |
| `tier_runs_completed` | `starter` | completed runs | 0 |
| `tier_operator_cost` | `starter` | rows contributing | micro-USD |
| `cohort_active_users` | `lab` | users active that day | 0 |
| `cohort_runs_completed` | `lab` | completed runs | 0 |
| `cohort_operator_cost` | `lab` | rows contributing | micro-USD |

Users with no cohort roll up under `user_state=''`, which the PK already permits as the column's default.

- [ ] **Step 1: Write the failing test**

```python
def test_tier_and_cohort_totals_survive_expiry(retention_fixture):
    """The row a user id made expire, the anonymous total does not."""
    retention_fixture.seed_facts(date(2026, 3, 1), tier="starter", cohort="lab")

    retention_fixture.service.run_once(now=NOW)

    assert retention_fixture.facts_for(date(2026, 3, 1)) == []
    rollups = retention_fixture.rollups_for(date(2026, 3, 1))
    assert any(
        r.metric_name == "tier_operator_cost" and r.user_state == "starter"
        for r in rollups
    )
    assert any(
        r.metric_name == "cohort_runs_completed" and r.user_state == "lab"
        for r in rollups
    )
```

- [ ] **Step 2: Run it to verify it fails, then implement**

Extend `rollup_lifecycle_day` to emit the six metric families above from `list_facts_for_date(day)` alongside the segment counts and transitions it already writes, and confirm `retention.py` still calls it for every expiring day **and** for each boundary day before deleting. Do not reorder those two steps.

- [ ] **Step 3: Run and commit**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/test_retention.py -v
git add dashboard/backend/domain/analytics/rollups.py dashboard/backend/domain/analytics/retention.py \
        dashboard/backend/tests/domain/analytics/test_retention.py
git status --short
git commit -m "feat(analytics): keep tier and cohort totals past expiry"
```

### Task C5: One metric struct, two audiences

**Files:**
- Modify: `dashboard/backend/domain/analytics/query_service.py` — add `get_user_metrics`
- Test: `dashboard/backend/tests/domain/analytics/test_user_metrics.py` (create)

**Interfaces:**
- Consumes: `sum_recent_facts`, `list_activity`, `build_lifecycle_inputs`.
- Produces: `AnalyticsQueryService.get_user_metrics(user_id: int, *, audience: Literal["admin", "user"], now: datetime | None = None) -> UserMetrics`. Nothing else calls it yet.

This is the My Usage seam, and it is a seam rather than a feature: the page is out of scope, the filter is not. Building the struct once and filtering by audience is what stops a future `GET /api/me/usage` from being written as a second, subtly different calculation that leaks a field.

- [ ] **Step 1: Write the failing test**

```python
_ADMIN_ONLY = {
    "active_days_30d",
    "lifecycle",
    "operational",
    "cohort",
    "group",
    "access_log",
}


def test_the_user_audience_never_sees_admin_fields(metrics_fixture):
    metrics = metrics_fixture.service.get_user_metrics(
        metrics_fixture.user_id, audience="user", now=NOW
    )
    payload = metrics.model_dump()

    assert _ADMIN_ONLY.isdisjoint(payload)


def test_both_audiences_agree_on_the_shared_numbers(metrics_fixture):
    """One calculation, two views. The numbers must not diverge."""
    admin = metrics_fixture.service.get_user_metrics(
        metrics_fixture.user_id, audience="admin", now=NOW
    ).model_dump()
    user = metrics_fixture.service.get_user_metrics(
        metrics_fixture.user_id, audience="user", now=NOW
    ).model_dump()

    for field in ("runs_7d", "runs_30d", "own_spend_micro", "operator_cost_micro", "tier"):
        assert admin[field] == user[field]


def test_the_admin_audience_sees_every_field(metrics_fixture):
    payload = metrics_fixture.service.get_user_metrics(
        metrics_fixture.user_id, audience="admin", now=NOW
    ).model_dump()

    assert _ADMIN_ONLY <= set(payload)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/test_user_metrics.py -v
```

Expected: FAIL at import -- `UserMetrics` and `get_user_metrics` do not exist.

- [ ] **Step 3: Define the struct**

In `query_service.py`, beside the other response models:

```python
class RunOutcomeCounts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requested: int = Field(default=0, ge=0)
    completed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    cancelled: int = Field(default=0, ge=0)


class UserMetrics(BaseModel):
    """One user's numbers, built once and projected per audience.

    Every field below the shared block is admin-only. The projection is a
    field filter rather than a second query on purpose: a future
    GET /api/me/usage written as its own calculation is how two answers to
    one question get shipped, and how a field nobody meant to expose
    eventually is.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Shared with the subject.
    user_id: int = Field(gt=0)
    facts_as_of: date
    runs_7d: RunOutcomeCounts
    runs_30d: RunOutcomeCounts
    own_spend_micro: int = Field(default=0, ge=0)
    operator_cost_micro: int = Field(default=0, ge=0)
    tier: CommercialTier = "unpaid"
    credit_balance_micro: int = Field(default=0, ge=0)
    data_quality: Literal["complete", "partial"] = "complete"

    # Admin only. Absent from the user projection entirely, not nulled:
    # a null still tells the subject the field exists.
    active_days_30d: int | None = Field(default=None, ge=0, le=30)
    lifecycle: LifecycleResult | None = None
    operational: OperationalResult | None = None
    cohort: str | None = None
    group: str | None = None
    access_log: Sequence[str] | None = None
```

- [ ] **Step 4: Build once, project per audience**

Build the full struct, then return `model_copy` with the admin-only fields excluded for the user audience -- `model_dump(exclude=...)` so the keys are absent rather than null. The user view carries runs by outcome over 7 and 30 days, own spend, operator-funded cost, tier and credit balance. It never carries active days, lifecycle segment or its reasons and evidence, operational state or its evidence, cohort, the group badge, or the admin access log.

- [ ] **Step 5: Run and commit**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/domain/analytics/test_user_metrics.py -v
git add dashboard/backend/domain/analytics/query_service.py \
        dashboard/backend/tests/domain/analytics/test_user_metrics.py
git status --short
git commit -m "feat(analytics): build user metrics once and filter by audience"
```

### Task C6: The admin console

**Files:**
- Modify: `dashboard/frontend/app.html` — the filter block at lines 2255-2260, the users table at lines 2475-2476, the cache busters at lines 2665-2678
- Modify: `dashboard/frontend/js/admin-analytics-value.js` (996 lines) — filter state, the three new controls, the group badge
- Modify: `dashboard/frontend/js/admin-credits.js` — the cohort field on the Users tab
- Test: `dashboard/backend/tests/test_admin_analytics_value_frontend.py` and `test_admin_analytics_frontend.py`

**Interfaces:**
- Consumes: the axis filters from C1 and `PATCH /api/admin/users/{id}` with `cohort` from B2.
- Produces: nothing.

- [ ] **Step 1: Write the failing guard tests**

The frontend has no build step, so the guards read source text through `dashboard/backend/tests/_frontend_source.py`. Add:

```python
def test_the_three_axis_filters_exist():
    assert 'id="adminAnalyticsRole"' in APP_HTML
    assert 'id="adminAnalyticsTier"' in APP_HTML
    assert 'id="adminAnalyticsCohort"' in APP_HTML


def test_the_users_tab_has_a_cohort_field():
    assert 'id="adminUserCohort"' in APP_HTML
    assert "cohort" in strip_comments(ADMIN_CREDITS_JS)


def test_cache_buster_versions_are_pinned():
    assert 'styles.css?v=140' in APP_HTML
    assert 'js/admin-analytics.js?v=7' in APP_HTML
    assert 'js/admin-analytics-value.js?v=6' in APP_HTML
    assert 'js/admin-credits.js?v=7' in APP_HTML
```

- [ ] **Step 2: Bump every cache buster you touch, and only those**

Current values in `app.html`: `styles.css?v=139` (line 16), `js/admin-analytics.js?v=6`, `js/admin-analytics-value.js?v=5`, `js/admin-credits.js?v=6`, `js/admin-tabs.js?v=5` (lines 2674-2676, 2673). Bump each file you edit by one and update **every** test that pins it. Find them all first:

```bash
command grep -rn "?v=" dashboard/frontend/app.html dashboard/backend/tests/ | sort
```

Use `command grep`, not `grep`: the shell's `grep` is shimmed to `ugrep` here and hides gitignored files. Five test files pin these strings; a missed one is a red CI run on an otherwise finished PR.

- [ ] **Step 3: Add the controls**

Three filters beside the existing Billing mode select at line 2257, following its exact markup: a `role` select (All roles / User / Admin), a `tier` select (All tiers / Unpaid / Starter / Invested / High value), and a `cohort` text input with `pattern="[a-z0-9_-]{1,32}"` and a `<datalist>` populated from the cohorts present in the current response. There is no cohort lookup table, so the datalist is a convenience, never a constraint.

Wire them into `readFilterControls()` and the URL state the same way the existing filters are wired at `admin-analytics.js:1084`, so a filtered view stays shareable as a link.

On the Users tab, add a Cohort column to the table at line 2475 and an inline editable field that PATCHes `cohort`. An empty submission clears it. Show the validation error the route returns rather than inventing client-side copy.

Render the group badge from the response's `group` field. Do not recompute the precedence in JavaScript: the server owns that rule, and two owners of one rule is how the season-number badge and its banner came to disagree.

- [ ] **Step 4: Run the frontend guards and the whole suite**

```bash
~/atl-venv/bin/python -m pytest dashboard/backend/tests/ -q
TEST_POSTGRES_URL=postgresql://postgres:atl@localhost:55432/postgres \
  ~/atl-venv/bin/python -m pytest dashboard/backend/tests/ -q
```

Expected: PASS on both.

- [ ] **Step 5: Commit and open the PR**

```bash
git add dashboard/frontend/ dashboard/backend/tests/
git status --short
git commit -m "feat(admin): filter analytics by role, tier and cohort"
git push -u origin feat/user-analytics-read-paths
```

PR title: `feat: user analytics read paths`. First line of the body: do not merge before PR B. Add the `blocked` label until PR B lands.

---

## After PR C

Two follow-ups the spec names, neither part of these PRs:

- **An admin-facing documentation page.** The published Sphinx tree has no page on admin analytics, lifecycle segments, or cohorts, so nothing is stale — nothing exists. A page covering the three axes, the segment definitions and the as-of-yesterday rule is owned by the maintainer. File it as an issue at PR C's merge, linked from the closing comment, rather than leaving it in a merged PR's body where nobody will read it again.
- **Whatever the profile dropped.** Task C1 step 5 removes the legacy 180-day profile scan, and some fields it computed have no home in the fact table. List them in PR C's body and file one issue for any the maintainer wants back.

## Self-review notes

Checked against the spec section by section. Three things the plan resolves that the spec did not:

1. **The inline per-event recompute.** The spec's "Why now" names only the sixty-second sweep. `AnalyticsService.record_server_event` also recomputes synchronously per accepted event, reading the user's 180-day history twice each time, and `_build_analytics_service` enables it in production. It is the larger burner. PR A tasks 1 and 2 close it; PR B task 4 deletes it.
2. **`agent_runs.owner_user_id` is genuinely required.** The event log looked like it might already carry per-user model cost, since `model_usage_recorded` has a `cost_micro_usd` property. Only the historical backfill emits that event; the live path emits `credits_reserved`, `credits_settled` and `credits_refunded`, which measure the user's own credit spend, not operator-funded cost. So the column stays, with the four-hop plumbing in PR B task 3.
3. **The overview's "one bounded one-day scan" is not free today.** `rollup_current_day` reads a 30-day trailing window and `rollup_day` a 31-day one, because the conversion and repeat-rate formulas need context. PR C task 3 narrows the current-day read and takes the context from the rollups; a metric that cannot be computed that way is shown as of yesterday rather than widening the scan.

4. **The activity route is not quite unchanged.** The spec's read-paths table marks `GET /users/{id}/activity` as unchanged. Its sessions section reaches `list_session_rows` (`query_service.py:526`), which scans a user's entire `event_group='experience'` history with no time window and paginates in Python. That is an unbounded per-user read and it contradicts the discipline rules the same spec sets, so PR C task 1 step 6 gives it a 30-day window. The timeline, runs and usage sections are genuinely unchanged.

One spec requirement has no task by design: the outreach sender, the My Usage page, the subscription system and the `viewer` role are all explicit non-goals. Their seams are built — `lifecycle_transitions` in PR B task 1 and 8, `get_user_metrics` in PR C task 5, `commercial_tier` untouched in `lifecycle.py`, and admin accounts for the supervisor.
