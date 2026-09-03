# Admin User Value Analytics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Admin Analytics operational-first overview with an explainable user-value workspace covering lifecycle, activation retention, commercial value, operational health, priority users, and evidence-backed user profiles.

**Architecture:** Add a pure lifecycle classifier and focused value-projection/query modules beside the existing Analytics foundation, while preserving the legacy five-state snapshot and `/overview` contract during migration. Persist current lifecycle and operational axes plus bounded daily lifecycle history in equivalent SQLite/PostgreSQL schemas, join commercial facts from the authoritative Credits ledger through read-only batched queries, and expose independent Admin endpoints so one unavailable section cannot blank the others. Keep the existing profile/activity controller focused on detailed user inspection and introduce `admin-analytics-value.js` for the new overview, disclosures, URL filters, charts, and evidence dialogs.

**Tech Stack:** Python 3, FastAPI, Pydantic v2, SQLite, PostgreSQL/psycopg, pytest, vanilla JavaScript, semantic HTML, CSS, Chart.js 4.4.0 already loaded by `app.html`.

**Spec:** `docs/superpowers/specs/2026-09-03-admin-user-value-analytics-design.md`

## Global Constraints

- Activation is the first server-authoritative `backtest_completed` event; queued, started, failed, and cancelled runs never activate a user.
- Lifecycle activity includes intentional Agent configuration, credential configuration, backtest progress including terminal failure, ATL Credits purchase, and ATL Credits consumption; page views, sign-in, refresh, polling, heartbeat, automatic Grants, and Admin Grants are excluded.
- `inactive_days` is calculated from UTC calendar dates. Values `0..7` are recent, `8..29` are At risk, and `30+` are Dormant.
- Core requires at least 3 distinct active UTC days and 3 successful backtests in the current UTC date plus preceding 29 UTC dates, with `inactive_days <= 7`.
- Lifecycle and operational state are independent axes. Operational precedence remains `Blocked`, then `Needs attention`, then `Healthy`.
- Commercial tier uses lifetime settled purchase Credits minus settled refund Credits: `Unpaid = 0`, `Starter = 1..4_999_999` microcredits, `Invested = 5_000_000..19_999_999`, and `High value >= 20_000_000`.
- Admin Grants and model consumption never count as revenue. Current Grant, Purchased, and total available balances remain independent measures.
- Retention cohorts start on the UTC Monday of the first successful backtest and expose mature Week 1, Week 2, and Week 4 meaningful-return windows only.
- Current lifecycle and commercial tier ignore the page date range; the date range affects lifecycle history, retention cohorts, and selected-period commercial/operational measures only.
- Default queries exclude Admin and `analytics_excluded` accounts. `Include internal accounts` is the only top-level inclusion override.
- Preserve the legacy five-state snapshot columns, `/api/admin/analytics/overview`, and legacy `status` input during migration; new UI code uses `lifecycle_segment` and `operational_state` and never derives them from `status`.
- All Analytics APIs are Admin-only, read-only, display-safe, and independently available. They never return SQL, raw events, provider bodies, secrets, prompt or strategy content, full IP addresses, raw User-Agent values, or credential material.
- User profile reads continue recording Admin access. Analytics failures never change authentication, Agent, run, provider, Credits, or payment behavior.
- User-level daily history is retained for 180 days only after anonymous lifecycle counts and bounded segment-transition rollups are written.
- Backfill covers the previous eight UTC weeks, is batched, resumable, idempotent, never uses future evidence for an earlier date, and labels untrustworthy evidence `partial` rather than fabricating zero.
- The Admin workspace uses a persistent vertical rail in `Analytics / Users / Providers / Activity` order. Narrow screens keep an icon rail. Credits and Billing navigation is unchanged in this pull request.
- Use synthetic fixtures and fake repositories. Tests require no real API key, Stripe call, provider call, production database, or copied production identity.
- Never stage or commit `dashboard/storage/data/backtest.db`, `.superpowers/`, `work/`, secrets, or generated mockup artifacts.

## Locked File Structure

- `dashboard/backend/domain/analytics/lifecycle.py`: pure meaningful-activity, lifecycle, operational, commercial-tier, and cohort-date rules with no I/O.
- `dashboard/backend/domain/analytics/states.py`: dual-axis current snapshot calculation and compatibility projection to the legacy five-state fields.
- `dashboard/backend/domain/analytics/value_repository.py`: SQLite/PostgreSQL-neutral daily lifecycle history and batched read-only Credits/user facts.
- `dashboard/backend/domain/analytics/lifecycle_backfill.py`: bounded historical reconstruction, quality propagation, anonymous aggregation, and resumable cursor.
- `dashboard/backend/domain/analytics/value_queries.py`: lifecycle, retention, commercial, operational, priority-user, and enriched-profile response models and query composition.
- `dashboard/backend/domain/analytics/query_service.py`: retain legacy overview/activity implementation; delegate new user filters/profile enrichment to `value_queries.py` without adding new aggregation logic.
- `dashboard/frontend/js/admin-analytics-value.js`: new overview state, fetch boundaries, charts, disclosures, filters, evidence dialogs, and drill-down.
- `dashboard/frontend/js/admin-analytics.js`: retain the dedicated profile/activity controller; accept enriched profile fields and expose `openProfile()` to the value controller.

---

### Task 1: Pure Lifecycle and Value Rules

**Files:**
- Create: `dashboard/backend/domain/analytics/lifecycle.py`
- Create: `dashboard/backend/tests/domain/analytics/test_lifecycle.py`

**Interfaces:**
- Consumes: `AnalyticsEventRecord` from `dashboard.backend.domain.analytics.models`; the legacy `metrics.is_meaningful_event()` remains unchanged for `/overview` and five-state compatibility.
- Produces: `LifecycleInputs`, `LifecycleResult`, `OperationalResult`, `CommercialTier`, `is_lifecycle_activity(event)`, `calculate_lifecycle(inputs, as_of)`, `calculate_operational_state(signals, as_of)`, `commercial_tier(net_purchased_micro)`, and `activation_cohort_week(activated_at)`.

- [ ] **Step 1: Write failing lifecycle boundary and activity-allowlist tests**

```python
def test_lifecycle_boundaries_use_utc_dates():
    assert calculate_lifecycle(inputs(account_age_days=6), NOW).segment == "new"
    assert calculate_lifecycle(inputs(account_age_days=7), NOW).segment == "onboarding"
    assert calculate_lifecycle(inputs(inactive_days=8), NOW).segment == "at_risk"
    assert calculate_lifecycle(inputs(inactive_days=29), NOW).segment == "at_risk"
    assert calculate_lifecycle(inputs(inactive_days=30), NOW).segment == "dormant"


def test_core_requires_both_trailing_thresholds():
    assert calculate_lifecycle(inputs(active_days_30d=3, successful_backtests_30d=3), NOW).segment == "core"
    assert calculate_lifecycle(inputs(active_days_30d=2, successful_backtests_30d=3), NOW).segment == "growing"
    assert calculate_lifecycle(inputs(active_days_30d=3, successful_backtests_30d=2), NOW).segment == "growing"


@pytest.mark.parametrize("event_name", ["page_viewed", "account_signed_in", "session_heartbeat", "admin_grant_assigned"])
def test_passive_or_admin_events_are_not_lifecycle_activity(event_name):
    assert is_lifecycle_activity(event(event_name)) is False
```

- [ ] **Step 2: Run the focused tests and verify missing imports fail**

Run: `pytest -q dashboard/backend/tests/domain/analytics/test_lifecycle.py`

Expected: FAIL during collection because `dashboard.backend.domain.analytics.lifecycle` does not exist.

- [ ] **Step 3: Implement frozen rule inputs/results and deterministic precedence**

```python
LifecycleSegment = Literal["new", "onboarding", "growing", "core", "at_risk", "dormant"]
OperationalState = Literal["blocked", "needs_attention", "healthy"]
CommercialTier = Literal["unpaid", "starter", "invested", "high_value"]


class LifecycleInputs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    user_id: int = Field(gt=0)
    created_at: datetime
    first_successful_backtest_at: datetime | None = None
    last_meaningful_activity_at: datetime | None = None
    active_days_30d: int = Field(ge=0, le=30)
    successful_backtests_30d: int = Field(ge=0)


class LifecycleResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    segment: LifecycleSegment
    reason_code: str
    reason: str
    evidence: Sequence[str]
    activated_at: datetime | None
    last_meaningful_activity_at: datetime | None
    inactive_days: int = Field(ge=0)
    active_days_30d: int = Field(ge=0, le=30)
    successful_backtests_30d: int = Field(ge=0)
    calculated_at: datetime


class OperationalSignals(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    user_id: int = Field(gt=0)
    account_restricted: bool = False
    usable_billing_lane: bool = True
    selected_provider_enabled: bool = True
    default_credential_status: Literal["verified", "invalid", "verification_unavailable", "missing"] = "verified"
    failed_terminal_runs_24h: int = Field(default=0, ge=0)
    run_beyond_safe_deadline: bool = False


class OperationalResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    state: OperationalState
    reason_code: str
    reason: str
    evidence: Sequence[str]
    calculated_at: datetime


def calculate_lifecycle(inputs: LifecycleInputs, as_of: datetime) -> LifecycleResult:
    calculation_time = require_utc(as_of, "as_of")
    anchor = inputs.last_meaningful_activity_at or inputs.created_at
    inactive_days = (calculation_time.date() - require_utc(anchor, "activity anchor").date()).days
    account_age_days = (calculation_time.date() - require_utc(inputs.created_at, "created_at").date()).days
    if inactive_days >= 30:
        return lifecycle_result("dormant", inputs, inactive_days, calculation_time)
    if inactive_days >= 8:
        return lifecycle_result("at_risk", inputs, inactive_days, calculation_time)
    if inputs.first_successful_backtest_at is None:
        return lifecycle_result("new" if account_age_days <= 6 else "onboarding", inputs, inactive_days, calculation_time)
    segment = "core" if inputs.active_days_30d >= 3 and inputs.successful_backtests_30d >= 3 else "growing"
    return lifecycle_result(segment, inputs, inactive_days, calculation_time)


def commercial_tier(net_purchased_micro: int) -> CommercialTier:
    value = max(0, net_purchased_micro)
    if value == 0:
        return "unpaid"
    if value < 5_000_000:
        return "starter"
    if value < 20_000_000:
        return "invested"
    return "high_value"
```

Use fixed allowlists in `is_lifecycle_activity()`. Count Agent create/configure, credential save/verify/reverify/default selection, every run request/progress/terminal event, and model usage/credit consumption; explicitly return false for frontend page/session events and all automatic/Admin Grant events. Purchase timestamps come from the authoritative Credits ledger in Task 2 because the accepted Analytics event names are unchanged. Return evidence as allowlisted display strings, never raw `properties_json`.

- [ ] **Step 4: Add independent operational precedence and reason tests**

```python
def test_operational_state_is_independent_from_lifecycle():
    operational = calculate_operational_state(
        OperationalSignals(user_id=1, account_restricted=True, failed_terminal_runs_24h=3),
        NOW,
    )
    assert operational.state == "blocked"
    assert operational.reason_code == "account_restricted"


def test_inactive_reason_distinguishes_activation_history():
    never = calculate_lifecycle(inputs(inactive_days=30, activated=False), NOW)
    previous = calculate_lifecycle(inputs(inactive_days=30, activated=True), NOW)
    assert never.reason_code == "dormant_never_activated"
    assert previous.reason_code == "dormant_previously_activated"
```

- [ ] **Step 5: Run domain tests and commit**

Run: `pytest -q dashboard/backend/tests/domain/analytics/test_lifecycle.py dashboard/backend/tests/domain/analytics/test_metrics.py dashboard/backend/tests/domain/analytics/test_states.py`

Expected: PASS.

```bash
git add dashboard/backend/domain/analytics/lifecycle.py dashboard/backend/tests/domain/analytics/test_lifecycle.py
git commit -m "feat: define user value analytics rules"
```

### Task 2: Additive Dual-Axis and Daily-History Storage

**Files:**
- Modify: `dashboard/backend/domain/analytics/repository.py`
- Modify: `dashboard/backend/domain/analytics/repository_postgres.py`
- Create: `dashboard/backend/domain/analytics/value_repository.py`
- Modify: `dashboard/backend/tests/domain/analytics/test_repository_contract.py`
- Modify: `dashboard/backend/tests/domain/analytics/test_repository_postgres.py`
- Create: `dashboard/backend/tests/domain/analytics/test_value_repository.py`

**Interfaces:**
- Consumes: `LifecycleSegment`, `LifecycleResult`, `OperationalResult`, and existing `analytics_store`.
- Produces: `UserValueSnapshot`, `UserLifecycleDailySnapshot`, `CommercialValueFact`, `CurrentOperationalFacts`, `ProjectionJob`, `ValueAnalyticsStore.upsert_current_snapshot(snapshot)`, `ValueAnalyticsStore.upsert_daily_snapshot(snapshot)`, `ValueAnalyticsStore.list_daily_snapshots(start, end, user_ids=None)`, `ValueAnalyticsStore.list_credit_activity(user_ids, start, end)`, `ValueAnalyticsStore.list_commercial_values(user_ids, start, end)`, `ValueAnalyticsStore.get_operational_facts(user_id, now)`, `ValueAnalyticsStore.get_projection_job(job_name)`, and `ValueAnalyticsStore.save_projection_job(job)`.

- [ ] **Step 1: Write SQLite migration, idempotency, and Credits-netting contract tests**

```python
def test_existing_snapshot_table_is_migrated_additively(store):
    columns = table_columns(store, "user_analytics_snapshots")
    assert {"status", "lifecycle_segment", "operational_state", "activated_at", "active_days_30d"} <= columns


def test_daily_snapshot_is_unique_per_user_and_utc_date(value_store):
    value_store.upsert_daily_snapshot(daily(user_id=7, segment="growing"))
    value_store.upsert_daily_snapshot(daily(user_id=7, segment="core"))
    rows = value_store.list_daily_snapshots(start=DAY, end=DAY + timedelta(days=1), user_ids=[7])
    assert [(row.user_id, row.lifecycle_segment) for row in rows] == [(7, "core")]


def test_commercial_value_nets_only_purchase_and_refund(value_store, credit_entries):
    facts = value_store.list_commercial_values([7], start=START, end=END)
    assert facts[7].lifetime_net_purchased_micro == 5_000_000
    assert facts[7].commercial_tier == "invested"
    assert facts[7].admin_grant_activity_micro == 1_500_000
    assert facts[7].consumed_micro == 400_000


def test_credit_activity_counts_purchase_and_consumption_but_not_grants(value_store, credit_entries):
    days = value_store.list_credit_activity([7], start=START, end=END)[7]
    assert days == (PURCHASED_AT, CONSUMED_AT)
```

- [ ] **Step 2: Run repository tests and verify schema assertions fail**

Run: `pytest -q dashboard/backend/tests/domain/analytics/test_repository_contract.py dashboard/backend/tests/domain/analytics/test_value_repository.py`

Expected: FAIL because dual-axis columns, daily history, and `ValueAnalyticsStore` do not exist.

- [ ] **Step 3: Add equivalent SQLite and PostgreSQL schema changes**

Add nullable compatibility-migration columns first, backfill safe defaults, then enforce values in repository validation:

```sql
ALTER TABLE user_analytics_snapshots ADD COLUMN lifecycle_segment TEXT;
ALTER TABLE user_analytics_snapshots ADD COLUMN lifecycle_reason_code TEXT;
ALTER TABLE user_analytics_snapshots ADD COLUMN lifecycle_reason TEXT;
ALTER TABLE user_analytics_snapshots ADD COLUMN lifecycle_evidence_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE user_analytics_snapshots ADD COLUMN operational_state TEXT;
ALTER TABLE user_analytics_snapshots ADD COLUMN operational_reason_code TEXT;
ALTER TABLE user_analytics_snapshots ADD COLUMN operational_reason TEXT;
ALTER TABLE user_analytics_snapshots ADD COLUMN operational_evidence_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE user_analytics_snapshots ADD COLUMN activated_at TEXT;
ALTER TABLE user_analytics_snapshots ADD COLUMN last_meaningful_activity_at TEXT;
ALTER TABLE user_analytics_snapshots ADD COLUMN active_days_30d INTEGER NOT NULL DEFAULT 0;
ALTER TABLE user_analytics_snapshots ADD COLUMN successful_backtests_30d INTEGER NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS user_lifecycle_daily_snapshots (
    snapshot_date TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    lifecycle_segment TEXT NOT NULL,
    lifecycle_reason_code TEXT NOT NULL,
    data_quality TEXT NOT NULL CHECK (data_quality IN ('complete', 'partial')),
    calculated_at TEXT NOT NULL,
    PRIMARY KEY (snapshot_date, user_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS analytics_projection_jobs (
    job_name TEXT PRIMARY KEY,
    window_start TEXT NOT NULL,
    window_end TEXT NOT NULL,
    cursor TEXT,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'complete')),
    updated_at TEXT NOT NULL
);
```

Use `ADD COLUMN IF NOT EXISTS` in PostgreSQL and an inspected-column loop in SQLite so reopening either repository is idempotent. Add indexes on `(snapshot_date, lifecycle_segment)` and `(user_id, snapshot_date DESC)` in both twins.

- [ ] **Step 4: Implement focused frozen models and batched repository methods**

```python
class UserValueSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    user_id: int = Field(gt=0)
    lifecycle_segment: LifecycleSegment
    lifecycle_reason_code: str
    lifecycle_reason: str
    lifecycle_evidence: Sequence[str]
    operational_state: OperationalState
    operational_reason_code: str
    operational_reason: str
    operational_evidence: Sequence[str]
    activated_at: datetime | None
    last_meaningful_activity_at: datetime | None
    inactive_days: int = Field(ge=0)
    active_days_30d: int = Field(ge=0, le=30)
    successful_backtests_30d: int = Field(ge=0)
    calculated_at: datetime


class UserLifecycleDailySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    snapshot_date: date
    user_id: int = Field(gt=0)
    lifecycle_segment: LifecycleSegment
    lifecycle_reason_code: str
    data_quality: Literal["complete", "partial"]
    calculated_at: datetime


class CommercialValueFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    user_id: int = Field(gt=0)
    lifetime_net_purchased_micro: int = Field(ge=0)
    commercial_tier: CommercialTier
    purchased_micro: int
    refunded_micro: int
    consumed_micro: int
    admin_grant_activity_micro: int
    grant_available_micro: int
    purchased_available_micro: int
    total_available_micro: int


class CurrentOperationalFacts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    user_id: int = Field(gt=0)
    account_restricted: bool
    usable_billing_lane: bool
    selected_provider_enabled: bool
    default_credential_status: Literal["verified", "invalid", "verification_unavailable", "missing"]
    failed_terminal_runs_24h: int = Field(ge=0)
    run_beyond_safe_deadline: bool


class ProjectionJob(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    job_name: str
    window_start: date
    window_end: date
    cursor: str | None = None
    status: Literal["pending", "running", "complete"]
    updated_at: datetime


class ValueAnalyticsStore:
    def __init__(self, analytics_base=analytics_store, credits_base=credits_store, provider_base=model_provider_store, agent_base=agent_store, run_base=run_store):
        self.analytics_base = analytics_base
        self.credits_base = credits_base
        self.provider_base = provider_base
        self.agent_base = agent_base
        self.run_base = run_base

    def list_commercial_values(
        self,
        user_ids: Sequence[int],
        *,
        start: datetime,
        end: datetime,
    ) -> dict[int, CommercialValueFact]:
        identities = validated_user_ids(user_ids)
        lifetime = self._purchase_refund_totals(identities)
        period = self._period_commercial_totals(identities, start=start, end=end)
        balances = self.credits_base.get_balance_projections(list(identities))
        return build_commercial_facts(identities, lifetime, period, balances)

    def list_credit_activity(
        self,
        user_ids: Sequence[int],
        *,
        start: datetime,
        end: datetime,
    ) -> dict[int, Sequence[datetime]]:
        return self._purchase_and_consumption_timestamps(validated_user_ids(user_ids), start=start, end=end)

    def get_operational_facts(self, user_id: int, *, now: datetime) -> CurrentOperationalFacts:
        billing = self.credits_base.get_account_billing_state(user_id)
        credentials = self.provider_base.list_user_credentials(user_id)
        providers = {row["provider_id"]: row for row in self.provider_base.list_all_providers()}
        runs = self._list_user_run_health(user_id, now=now)
        return build_current_operational_facts(user_id, billing, credentials, providers, runs, now)
```

Issue parameterized bounded `IN`/`ANY(%s)` queries rather than one query per user. Read `credit_ledger_entries.entry_type IN ('purchase', 'refund', 'admin_grant_assign', 'admin_grant_reclaim')`, `credit_llm_usage_entries`, and existing `get_balance_projections()` without copying balances into Analytics tables. `list_credit_activity()` returns UTC timestamps for settled purchases and consumption only, excluding refunds, automatic promotions, Admin assign, and Admin reclaim. Derive current blockers from `CreditsStore.get_account_billing_state()`, public credential status, current provider enablement, and run heartbeat/deadline records; resolve user-owned Agent IDs through `AgentStore.list_agents(owner_user_id=user_id)` before reading run health. Never decrypt a credential or infer a cleared blocker from an old error event. Persist only the backfill window/cursor/status in `analytics_projection_jobs`; never persist response data or commercial balances there.

- [ ] **Step 5: Run SQLite/PostgreSQL parity tests and commit**

Run: `pytest -q dashboard/backend/tests/domain/analytics/test_repository_contract.py dashboard/backend/tests/domain/analytics/test_repository_postgres.py dashboard/backend/tests/domain/analytics/test_value_repository.py`

Expected: PASS, including reopening migrations twice and dict-row PostgreSQL behavior.

```bash
git add dashboard/backend/domain/analytics/repository.py dashboard/backend/domain/analytics/repository_postgres.py dashboard/backend/domain/analytics/value_repository.py dashboard/backend/tests/domain/analytics/test_repository_contract.py dashboard/backend/tests/domain/analytics/test_repository_postgres.py dashboard/backend/tests/domain/analytics/test_value_repository.py
git commit -m "feat: store user value analytics projections"
```

### Task 3: Calculate Independent Current Snapshots

**Files:**
- Modify: `dashboard/backend/domain/analytics/states.py`
- Modify: `dashboard/backend/domain/analytics/service.py`
- Modify: `dashboard/backend/domain/analytics/maintenance.py`
- Modify: `dashboard/backend/tests/domain/analytics/test_states.py`
- Modify: `dashboard/backend/tests/test_analytics_maintenance.py`

**Interfaces:**
- Consumes: Task 1 classifiers and Task 2 `ValueAnalyticsStore`.
- Produces: `calculate_user_value_snapshot(user_id: int, *, now: datetime | None, state_store: AnalyticsStateStore | None, value_store: ValueAnalyticsStore | None) -> UserValueSnapshot`, time-aware stale repair, and unchanged `calculate_user_state(user_id: int, *, now: datetime | None, store: AnalyticsStateStore | None) -> UserAnalyticsSnapshot` compatibility behavior.

- [ ] **Step 1: Write failing current-snapshot compatibility and time-transition tests**

```python
def test_value_snapshot_keeps_lifecycle_and_operational_axes_separate(tmp_path):
    service, state_store, value_store = fixture(tmp_path, created_at=NOW - timedelta(days=20))
    completed_run(service, at=NOW - timedelta(days=9))
    restrict_credits(service, at=NOW - timedelta(hours=1))
    snapshot = calculate_user_value_snapshot(1, now=NOW, state_store=state_store, value_store=value_store)
    assert snapshot.lifecycle_segment == "at_risk"
    assert snapshot.operational_state == "blocked"
    assert snapshot.lifecycle_reason_code == "at_risk_previously_activated"
    assert snapshot.operational_reason_code == "account_restricted"


def test_legacy_snapshot_fields_remain_unchanged(tmp_path):
    snapshot = recalculate_user_snapshot(1, now=NOW, store=state_store)
    assert snapshot.status in {"blocked", "needs_attention", "dormant", "onboarding", "active"}


def test_relevant_event_recalculates_value_projection_without_breaking_acceptance(service, value_store):
    result = service.record_server_event(
        event_name="backtest_completed",
        user_id=1,
        source_event_id="run:completed:synthetic-1",
        occurred_at=NOW,
    )
    assert result.created is True
    assert value_store.get_current_snapshot(1).lifecycle_segment == "growing"


def test_projection_failure_does_not_reject_an_accepted_event(service, failing_value_store):
    result = service.record_server_event(
        event_name="backtest_requested",
        user_id=1,
        source_event_id="run:requested:synthetic-2",
        occurred_at=NOW,
    )
    assert result.created is True


def test_cleared_current_account_is_not_blocked_by_an_old_error(state_store, value_store):
    record_credits_unavailable(state_store, at=NOW - timedelta(hours=2))
    value_store.credits_base.set_active_account_for_test(1)
    snapshot = calculate_user_value_snapshot(1, now=NOW, state_store=state_store, value_store=value_store)
    assert snapshot.operational_state == "healthy"


def test_activation_timestamp_survives_raw_event_retention(state_store, value_store):
    value_store.upsert_current_snapshot(value_snapshot(activated_at=FIRST_SUCCESS))
    snapshot = calculate_user_value_snapshot(1, now=NOW, state_store=state_store, value_store=value_store)
    assert snapshot.activated_at == FIRST_SUCCESS
```

- [ ] **Step 2: Run state tests and verify the new calculator is missing**

Run: `pytest -q dashboard/backend/tests/domain/analytics/test_states.py dashboard/backend/tests/test_analytics_maintenance.py`

Expected: FAIL importing `calculate_user_value_snapshot`.

- [ ] **Step 3: Gather inputs once, calculate both axes, and persist one current projection**

```python
def calculate_user_value_snapshot(
    user_id: int,
    *,
    now: datetime | None = None,
    state_store: AnalyticsStateStore | None = None,
    value_store: ValueAnalyticsStore | None = None,
) -> UserValueSnapshot:
    current = require_utc(now or datetime.now(timezone.utc), "now")
    states = state_store or AnalyticsStateStore()
    values = value_store or ValueAnalyticsStore(states.base_store)
    user = require_user(states.get_user(user_id))
    events = states.list_user_events(user_id, now=current)
    credit_activity = values.list_credit_activity(
        [user_id],
        start=current - timedelta(days=30),
        end=current + timedelta(microseconds=1),
    )[user_id]
    previous = values.get_current_snapshot(user_id)
    lifecycle = calculate_lifecycle(lifecycle_inputs(user, events, credit_activity, previous, current), current)
    facts = values.get_operational_facts(user_id, now=current)
    operational = calculate_operational_state(operational_signals(facts, events, current), current)
    snapshot = UserValueSnapshot.from_results(user_id, lifecycle, operational)
    values.upsert_current_snapshot(snapshot)
    return snapshot
```

Compute first success, last meaningful activity, distinct active days, and successful runs from Analytics events plus authoritative purchase/consumption timestamps at or before `current`. Preserve a non-null previous `activated_at` after its raw activation event ages out. Keep the old `calculate_user_state()` path and fields until a later compatibility removal.

After `append_event()` succeeds, `AnalyticsService.record_server_event()` invokes value recalculation only for lifecycle/operationally relevant accepted events. Wrap projection work in the existing best-effort Analytics failure boundary: log only `event_name` and exception class, then return the successful `AppendEventResult`. Idempotent event replays do not create duplicate daily rows.

- [ ] **Step 4: Extend stale repair for UTC day-only transitions and daily writes**

```python
def repair_stale_value_snapshots(*, now=None, limit=100, state_store=None, value_store=None) -> int:
    current = require_utc(now or datetime.now(timezone.utc), "now")
    stores = state_store or AnalyticsStateStore()
    values = value_store or ValueAnalyticsStore(stores.base_store)
    user_ids = stores.list_stale_user_ids(now=current, limit=limit, include_time_transitions=True)
    for user_id in user_ids:
        snapshot = calculate_user_value_snapshot(user_id, now=current, state_store=stores, value_store=values)
        values.upsert_daily_snapshot(UserLifecycleDailySnapshot.from_current(snapshot, current.date(), "complete"))
    return len(user_ids)
```

Call this bounded repair from `run_analytics_maintenance()` in a separate `try` block so failure increments the safe failure counter without affecting legacy rollup/snapshot maintenance.

- [ ] **Step 5: Run state and maintenance tests and commit**

Run: `pytest -q dashboard/backend/tests/domain/analytics/test_states.py dashboard/backend/tests/test_analytics_maintenance.py dashboard/backend/tests/domain/analytics/test_lifecycle.py`

Expected: PASS.

```bash
git add dashboard/backend/domain/analytics/states.py dashboard/backend/domain/analytics/service.py dashboard/backend/domain/analytics/maintenance.py dashboard/backend/tests/domain/analytics/test_states.py dashboard/backend/tests/test_analytics_maintenance.py
git commit -m "feat: calculate dual-axis analytics snapshots"
```

### Task 4: Reconstruct Eight Weeks and Preserve Long-Term Aggregates

**Files:**
- Create: `dashboard/backend/domain/analytics/lifecycle_backfill.py`
- Modify: `dashboard/backend/domain/analytics/rollups.py`
- Modify: `dashboard/backend/domain/analytics/retention.py`
- Modify: `dashboard/backend/domain/analytics/maintenance.py`
- Create: `dashboard/backend/tests/domain/analytics/test_lifecycle_backfill.py`
- Modify: `dashboard/backend/tests/domain/analytics/test_rollups.py`
- Modify: `dashboard/backend/tests/domain/analytics/test_retention.py`
- Modify: `dashboard/backend/tests/test_analytics_maintenance.py`

**Interfaces:**
- Consumes: Task 1 point-in-time classifier and Task 2 daily-history repository.
- Produces: `LifecycleBackfillCursor`, `LifecycleBackfillReport`, `LifecycleBackfillSource`, `backfill_lifecycle_history(start, end, batch_size, cursor, now, source, store)`, `run_lifecycle_backfill_batch(now, batch_size, source, store)`, `rollup_lifecycle_day(day, store)`, and retention deletion that aggregates before removing expired user rows.

- [ ] **Step 1: Write failing no-future-leakage, resumability, and quality tests**

```python
def test_backfill_does_not_use_a_later_success_for_an_earlier_day(source, store):
    source.add_success(user_id=1, occurred_at=datetime(2026, 8, 20, tzinfo=UTC))
    backfill_lifecycle_history(start=date(2026, 8, 18), end=date(2026, 8, 22), batch_size=50, source=source, store=store, now=NOW)
    snapshots = {row.snapshot_date: row for row in store.list_daily_snapshots(start=date(2026, 8, 18), end=date(2026, 8, 22), user_ids=[1])}
    assert snapshots[date(2026, 8, 19)].lifecycle_segment == "onboarding"
    assert snapshots[date(2026, 8, 20)].lifecycle_segment == "growing"


def test_backfill_resumes_without_duplicate_daily_rows(source, store):
    first = backfill_lifecycle_history(start=START_DAY, end=END_DAY, batch_size=1, source=source, store=store, now=NOW)
    second = backfill_lifecycle_history(start=START_DAY, end=END_DAY, batch_size=50, cursor=first.next_cursor, source=source, store=store, now=NOW)
    assert second.complete is True
    assert unique_user_date_count(store) == total_row_count(store)


def test_untrustworthy_source_horizon_stays_partial(source, store):
    source.complete_from = date(2026, 8, 10)
    report = backfill_lifecycle_history(start=date(2026, 8, 1), end=date(2026, 8, 12), batch_size=100, source=source, store=store, now=NOW)
    assert report.partial_days == 9
```

- [ ] **Step 2: Run focused history tests and verify imports fail**

Run: `pytest -q dashboard/backend/tests/domain/analytics/test_lifecycle_backfill.py dashboard/backend/tests/domain/analytics/test_rollups.py dashboard/backend/tests/domain/analytics/test_retention.py`

Expected: FAIL because `lifecycle_backfill.py` and lifecycle rollups are missing.

- [ ] **Step 3: Implement point-in-time bounded reconstruction**

```python
class LifecycleBackfillReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    processed_users: int = Field(ge=0)
    written_rows: int = Field(ge=0)
    partial_days: int = Field(ge=0)
    next_cursor: str | None
    complete: bool


def backfill_lifecycle_history(
    *,
    start: date,
    end: date,
    batch_size: int = 100,
    cursor: str | None = None,
    now: datetime | None = None,
    source: LifecycleBackfillSource,
    store: ValueAnalyticsStore,
) -> LifecycleBackfillReport:
    window = validate_backfill_window(start, end, maximum_days=56)
    user_page = source.list_users(limit=bounded_batch_size(batch_size), cursor=cursor)
    for user in user_page.items:
        evidence = source.list_evidence(user.id, start=window.evidence_start, end=window.end_exclusive)
        for day in window.days:
            cutoff = datetime.combine(day + timedelta(days=1), time.min, tzinfo=UTC)
            inputs = lifecycle_inputs_at(user, evidence, cutoff=cutoff)
            quality = source.quality_for(day, required_from=day - timedelta(days=29))
            store.upsert_daily_snapshot(daily_from_result(calculate_lifecycle(inputs, cutoff), day, quality))
    return report_for_page(user_page, store)
```

Use only evidence with `occurred_at < cutoff`; never synthesize missing page/session/Agent/credential history. Record a deterministic encoded `(last_user_id, start, end)` cursor and refuse a cursor for a different window.

- [ ] **Step 4: Aggregate bounded segment counts and transitions before 180-day deletion**

```python
def rollup_lifecycle_day(day: date, *, store: ValueAnalyticsStore) -> Sequence[DailyRollup]:
    current = store.list_daily_snapshots(start=day, end=day + timedelta(days=1))
    previous = store.list_daily_snapshots(start=day - timedelta(days=1), end=day)
    rows = lifecycle_count_rollups(day, current)
    rows += lifecycle_transition_rollups(day, previous, current)
    store.replace_lifecycle_rollups(day, rows)
    return tuple(rows)


def delete_expired_lifecycle_history(*, before: date, batch_size: int) -> RetentionResult:
    candidates = value_store.list_expiring_daily_dates(before=before, limit=batch_size)
    for day in candidates:
        rollup_lifecycle_day(day, store=value_store)
        value_store.delete_daily_snapshots_for_date(day)
    return RetentionResult(lifecycle_rows_deleted=value_store.last_deleted_count, has_more_lifecycle_rows=value_store.has_daily_before(before))
```

Transition dimensions must be allowlisted `from_segment`/`to_segment` enum pairs encoded into fixed rollup fields, with no user ID in `analytics_daily_rollups`.

- [ ] **Step 5: Wire resumable deployment backfill and safe observability**

```python
def run_lifecycle_backfill_batch(
    *,
    now: datetime,
    batch_size: int,
    source: LifecycleBackfillSource,
    store: ValueAnalyticsStore,
) -> LifecycleBackfillReport:
    job = store.get_projection_job("lifecycle_previous_8_weeks")
    if job is None:
        end = now.astimezone(UTC).date() + timedelta(days=1)
        job = ProjectionJob(
            job_name="lifecycle_previous_8_weeks",
            window_start=end - timedelta(days=56),
            window_end=end,
            status="pending",
            updated_at=now,
        )
    if job.status == "complete":
        return LifecycleBackfillReport(
            processed_users=0,
            written_rows=0,
            partial_days=0,
            next_cursor=None,
            complete=True,
        )
    report = backfill_lifecycle_history(
        start=job.window_start,
        end=job.window_end,
        batch_size=batch_size,
        cursor=job.cursor,
        now=now,
        source=source,
        store=store,
    )
    store.save_projection_job(
        ProjectionJob(
            job_name=job.job_name,
            window_start=job.window_start,
            window_end=job.window_end,
            cursor=report.next_cursor,
            status="complete" if report.complete else "running",
            updated_at=now,
        )
    )
    return report
```

Call `run_lifecycle_backfill_batch()` from the already registered `run_analytics_maintenance()` reaper path in its own `try` block. Current snapshot repair runs regardless of backfill state; backfill failure increments a safe `lifecycle_backfill_failures` counter and retries the persisted cursor on the next sweep.

- [ ] **Step 6: Run history, retention, and maintenance tests and commit**

Run: `pytest -q dashboard/backend/tests/domain/analytics/test_lifecycle_backfill.py dashboard/backend/tests/domain/analytics/test_rollups.py dashboard/backend/tests/domain/analytics/test_retention.py dashboard/backend/tests/test_analytics_maintenance.py`

Expected: PASS, including safe failure counters and a maximum 56-day deployment window.

```bash
git add dashboard/backend/domain/analytics/lifecycle_backfill.py dashboard/backend/domain/analytics/rollups.py dashboard/backend/domain/analytics/retention.py dashboard/backend/domain/analytics/maintenance.py dashboard/backend/tests/domain/analytics/test_lifecycle_backfill.py dashboard/backend/tests/domain/analytics/test_rollups.py dashboard/backend/tests/domain/analytics/test_retention.py dashboard/backend/tests/test_analytics_maintenance.py
git commit -m "feat: backfill lifecycle analytics history"
```

### Task 5: Compose Lifecycle, Retention, Commercial, and Priority Queries

**Files:**
- Create: `dashboard/backend/domain/analytics/value_queries.py`
- Modify: `dashboard/backend/domain/analytics/query_service.py`
- Create: `dashboard/backend/tests/domain/analytics/test_value_queries.py`

**Interfaces:**
- Consumes: current/daily projections, anonymous rollups, batched commercial facts, existing operational overview metrics, exclusion settings, and existing profile/activity data.
- Produces: `LifecycleAnalyticsResponse`, `RetentionAnalyticsResponse`, `CommercialAnalyticsResponse`, `OperationalAnalyticsResponse`, `UserValueFilters`, `ValueUserListItem`, `ValueUserProfile`, and `ValueAnalyticsQueryService` methods matching Task 6 routes.

- [ ] **Step 1: Write failing fixed-window, retention-maturity, and priority-order tests**

```python
def test_date_filter_changes_history_not_current_identity(service):
    short = service.get_lifecycle(start=date(2026, 8, 25), end=date(2026, 8, 27), include_internal=False)
    long = service.get_lifecycle(start=date(2026, 7, 1), end=date(2026, 8, 27), include_internal=False)
    assert short.headline == long.headline
    assert short.segment_counts == long.segment_counts
    assert short.weekly_segments != long.weekly_segments


def test_retention_excludes_immature_target_week(service):
    response = service.get_retention(start=START_DAY, end=END_DAY, include_internal=False, now=NOW)
    immature = response.cohorts[-1].week_4
    assert immature.mature is False
    assert immature.eligible_users is None
    assert immature.rate is None


def test_priority_order_is_group_then_value_then_inactivity_then_id(service):
    page = service.list_users(filters=UserValueFilters(priority=True), limit=25, offset=0)
    assert [item.user_id for item in page.items] == [BLOCKED_HIGH, BLOCKED_LOW, ATTENTION, HEALTHY_AT_RISK, HEALTHY_ONBOARDING]
```

- [ ] **Step 2: Run query tests and verify response models are missing**

Run: `pytest -q dashboard/backend/tests/domain/analytics/test_value_queries.py`

Expected: FAIL importing `ValueAnalyticsQueryService`.

- [ ] **Step 3: Define exact display-safe response contracts**

```python
class SectionAvailability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    available: bool
    stale: bool = False
    status: Literal["ready", "building", "partial", "unavailable"]
    coverage_start: date | None = None
    coverage_end: date | None = None


class LifecycleHeadline(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    activated_users: int = Field(ge=0)
    core_users: int = Field(ge=0)
    at_risk_users: int = Field(ge=0)
    paid_users: int = Field(ge=0)


class WeeklyLifecycleCount(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    week_start: date
    segment_counts: dict[LifecycleSegment, int]
    data_quality: Literal["complete", "partial"]


class LifecycleTransition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    from_segment: LifecycleSegment
    to_segment: LifecycleSegment
    users: int = Field(ge=0)
    period_start: date
    period_end: date
    data_quality: Literal["complete", "partial"]


class LifecycleAnalyticsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    as_of: datetime
    headline: LifecycleHeadline
    segment_counts: dict[LifecycleSegment, int]
    weekly_segments: Sequence[WeeklyLifecycleCount]
    transitions: Sequence[LifecycleTransition]
    availability: dict[str, SectionAvailability]


class RetentionCell(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    week: Literal[1, 2, 4]
    mature: bool
    retained_users: int | None
    eligible_users: int | None
    rate: float | None = Field(default=None, ge=0, le=1)
    data_quality: Literal["complete", "partial"]


class RetentionCohort(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    cohort_week: date
    activated_users: int = Field(ge=0)
    week_1: RetentionCell
    week_2: RetentionCell
    week_4: RetentionCell


class RetentionAnalyticsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    as_of: datetime
    cohorts: Sequence[RetentionCohort]
    summary_week_1: RetentionCell
    summary_week_2: RetentionCell
    summary_week_4: RetentionCell
    availability: SectionAvailability


class CommercialPeriodSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    purchased_micro: int = Field(ge=0)
    refunded_micro: int = Field(ge=0)
    consumed_micro: int = Field(ge=0)
    admin_grant_activity_micro: int = Field(ge=0)
    platform_model_cost_micro_usd: int = Field(ge=0)


class BalanceTotals(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    grant_available_micro: int
    purchased_available_micro: int
    total_available_micro: int


class CommercialAnalyticsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    as_of: datetime
    tier_counts: dict[CommercialTier, int]
    lifetime_net_purchased_micro: int = Field(ge=0)
    selected_period: CommercialPeriodSummary
    current_balances: BalanceTotals
    availability: SectionAvailability


class OperationalAnalyticsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    as_of: datetime
    operational_state_counts: dict[OperationalState, int]
    backtest_success_rate: float | None = Field(default=None, ge=0, le=1)
    completed_runs: int = Field(ge=0)
    failed_runs: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    platform_model_cost_micro_usd: int = Field(ge=0)
    top_failure_categories: Sequence[FailureCategoryCount]
    availability: SectionAvailability


class UserValueFilters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    q: str | None = None
    lifecycle_segment: LifecycleSegment | None = None
    operational_state: OperationalState | None = None
    commercial_tier: CommercialTier | None = None
    activated: bool | None = None
    last_meaningful_activity_from: datetime | None = None
    last_meaningful_activity_to: datetime | None = None
    priority: bool = False
    legacy_status: str | None = None
    include_internal: bool = False


class ValueUserListItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    user_id: int = Field(gt=0)
    display_name: str
    email: str
    lifecycle: LifecycleResult
    operational: OperationalResult
    commercial_tier: CommercialTier
    lifetime_net_purchased_micro: int = Field(ge=0)
    priority_group: Literal["blocked", "needs_attention", "healthy_at_risk", "healthy_onboarding", "none"]
    profile_path: str


class PaginatedValueUsers(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    items: Sequence[ValueUserListItem]
    total: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)


class ValueUserProfile(AnalyticsUserProfile):
    lifecycle: LifecycleResult
    operational: OperationalResult
    commercial: CommercialValueFact
    selected_period_start: date
    selected_period_end: date
    recent_lifecycle_transitions: Sequence[LifecycleTransition]
```

`ValueUserListItem` returns identity, both reason/evidence objects, commercial tier and exact net purchase, activation and inactivity facts, `priority_group`, and stable `profile_path`. `ValueUserProfile` inherits the existing Overview/Timeline/Runs/Usage/Sessions summary fields and adds lifecycle, operational, commercial, period dates, and recent transitions.

- [ ] **Step 4: Implement query composition with independent availability**

```python
class ValueAnalyticsQueryService:
    def get_lifecycle(self, *, start: date, end: date, include_internal: bool) -> LifecycleAnalyticsResponse:
        users = self._eligible_users(include_internal=include_internal)
        current = self.value_store.list_current_snapshots([user.id for user in users])
        commercial = self.value_store.list_commercial_values([user.id for user in users], start=day_start(start), end=day_start(end))
        return build_lifecycle_response(users, current, commercial, self._history(start, end))

    def list_users(self, *, filters: UserValueFilters, limit: int, offset: int) -> PaginatedValueUsers:
        rows = join_user_value_facts(self._eligible_users(filters.include_internal), self.value_store)
        selected = filter_value_users(rows, filters)
        ordered = sorted(selected, key=priority_sort_key if filters.priority else user_sort_key(filters))
        return paginate_value_users(ordered, limit=limit, offset=offset)
```

Priority keys are `(group_rank, -tier_rank, -lifetime_net_purchased_micro, -inactive_days, user_id)`. Retention denominators include a cohort only after the full target week elapsed. Summary rates sum numerators and denominators across mature cohorts. Any partial source date propagates `partial` and never becomes a numeric zero.

- [ ] **Step 5: Run query tests and commit**

Run: `pytest -q dashboard/backend/tests/domain/analytics/test_value_queries.py dashboard/backend/tests/domain/analytics/test_retention.py`

Expected: PASS.

```bash
git add dashboard/backend/domain/analytics/value_queries.py dashboard/backend/domain/analytics/query_service.py dashboard/backend/tests/domain/analytics/test_value_queries.py
git commit -m "feat: query admin user value analytics"
```

### Task 6: Publish Independent Admin API Contracts and Safe Fixtures

**Files:**
- Modify: `dashboard/backend/api/routers/admin_analytics.py`
- Modify: `dashboard/backend/tests/test_admin_analytics_api.py`
- Create: `dashboard/backend/tests/fixtures/admin_analytics/lifecycle.json`
- Create: `dashboard/backend/tests/fixtures/admin_analytics/retention.json`
- Create: `dashboard/backend/tests/fixtures/admin_analytics/commercial.json`
- Create: `dashboard/backend/tests/fixtures/admin_analytics/operational.json`
- Modify: `dashboard/backend/tests/fixtures/admin_analytics/users.json`
- Modify: `dashboard/backend/tests/fixtures/admin_analytics/user_detail.json`
- Modify: `dashboard/backend/tests/fixtures/admin_analytics/overview_partial_error.json`

**Interfaces:**
- Consumes: Task 5 `ValueAnalyticsQueryService` response models.
- Produces: `GET /api/admin/analytics/lifecycle`, `/retention`, `/commercial`, `/operational`, `/users`, `/users/{user_id}`, and `/users/{user_id}/activity`; retains legacy `/overview`.

- [ ] **Step 1: Write failing route contract, authorization, audit, and safe-error tests**

```python
@pytest.mark.parametrize("path", ["lifecycle", "retention", "commercial", "operational", "users"])
def test_value_routes_require_admin(client, path):
    assert client.get(f"/api/admin/analytics/{path}").status_code in {401, 403}


def test_section_failure_returns_safe_503(client, admin_headers, failing_service):
    response = client.get("/api/admin/analytics/commercial", headers=admin_headers)
    assert response.status_code == 503
    assert response.json() == {"detail": "Analytics is temporarily unavailable."}
    assert "SELECT" not in response.text


def test_user_profile_records_one_admin_access(client, admin_headers, access_store):
    assert client.get("/api/admin/analytics/users/101?from=2026-08-01&to=2026-08-31", headers=admin_headers).status_code == 200
    assert [(row["subject_user_id"], row["section"]) for row in access_store.list_admin_access(101)] == [(101, "overview")]
```

- [ ] **Step 2: Run API tests and verify new routes return 404**

Run: `pytest -q dashboard/backend/tests/test_admin_analytics_api.py`

Expected: FAIL with `404 Not Found` for the four new aggregate routes.

- [ ] **Step 3: Add strict query parsers and independent routes**

```python
@router.get("/lifecycle", response_model=LifecycleAnalyticsResponse)
def get_lifecycle(request: Request, service: ValueAnalyticsQueryService = Depends(get_value_analytics_query_service)):
    start, end, include_internal = _value_range(request)
    return _call_or_http_error(service.get_lifecycle, start=start, end=end, include_internal=include_internal)


@router.get("/retention", response_model=RetentionAnalyticsResponse)
def get_retention(request: Request, service: ValueAnalyticsQueryService = Depends(get_value_analytics_query_service)):
    start, end, include_internal = _value_range(request)
    return _call_or_http_error(service.get_retention, start=start, end=end, include_internal=include_internal)


@router.get("/commercial", response_model=CommercialAnalyticsResponse)
def get_commercial(request: Request, service: ValueAnalyticsQueryService = Depends(get_value_analytics_query_service)):
    start, end, include_internal = _value_range(request)
    return _call_or_http_error(service.get_commercial, start=start, end=end, include_internal=include_internal)


@router.get("/operational", response_model=OperationalAnalyticsResponse)
def get_operational(request: Request, service: ValueAnalyticsQueryService = Depends(get_value_analytics_query_service)):
    filters = _operational_filters(request)
    return _call_or_http_error(service.get_operational, filters=filters)


@router.get("/users", response_model=PaginatedValueUsers)
def list_users(request: Request, service: ValueAnalyticsQueryService = Depends(get_value_analytics_query_service)):
    filters, limit, offset = _value_user_filters(request)
    return _call_or_http_error(service.list_users, filters=filters, limit=limit, offset=offset)


@router.get("/users/{user_id}", response_model=ValueUserProfile)
def get_user_profile(
    user_id: str,
    request: Request,
    admin: dict = Depends(require_admin),
    query_service: ValueAnalyticsQueryService = Depends(get_value_analytics_query_service),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
):
    start, end = _profile_range(request)
    subject_user_id = _parse_user_id(user_id)
    profile = _call_or_http_error(query_service.get_user_profile, user_id=subject_user_id, start=start, end=end)
    _record_access(analytics_service, admin=admin, subject_user_id=subject_user_id, section="overview")
    return profile
```

Allow only `from`, `to`, and `include_internal` on lifecycle/retention/commercial. Allow existing `billing_mode`, `provider`, and `model` only on operational. Extend users with exact `lifecycle_segment`, `operational_state`, `commercial_tier`, `activated`, `last_meaningful_activity_from`, `last_meaningful_activity_to`, and `priority`; keep `status` as `legacy_status` during migration. The base user profile accepts only `from` and `to`, defaulting to the trailing 30 UTC days, so its consumption and platform-cost facts match the overview range without changing lifecycle or tier. Bound trend, cohort, operational, commercial, and profile ranges to 180 inclusive UTC dates. Reject duplicates and unknown keys with the existing safe `422` response.

- [ ] **Step 4: Capture synthetic frontend contracts and validate them through Pydantic**

```python
@pytest.mark.parametrize(
    ("fixture_name", "model"),
    [
        ("lifecycle.json", LifecycleAnalyticsResponse),
        ("retention.json", RetentionAnalyticsResponse),
        ("commercial.json", CommercialAnalyticsResponse),
        ("operational.json", OperationalAnalyticsResponse),
        ("users.json", PaginatedValueUsers),
        ("user_detail.json", ValueUserProfile),
    ],
)
def test_frontend_contract_fixture_matches_response_model(fixture_name, model):
    payload = json.loads((FIXTURE_DIR / fixture_name).read_text())
    assert model.model_validate(payload).model_dump(mode="json") == payload
```

Use only `*.example.test`, synthetic IDs, invented provider/model identifiers, and small fake monetary values. Fixtures must include complete, partial/building, empty, and unavailable section examples.

- [ ] **Step 5: Run API and contract tests and commit**

Run: `pytest -q dashboard/backend/tests/test_admin_analytics_api.py dashboard/backend/tests/domain/analytics/test_value_queries.py`

Expected: PASS.

```bash
git add dashboard/backend/api/routers/admin_analytics.py dashboard/backend/tests/test_admin_analytics_api.py dashboard/backend/tests/fixtures/admin_analytics
git commit -m "feat: expose admin user value analytics APIs"
```

### Task 7: Convert Admin Sections to a Persistent Vertical Rail

**Files:**
- Modify: `dashboard/frontend/app.html`
- Modify: `dashboard/frontend/js/admin-tabs.js`
- Modify: `dashboard/frontend/styles.css`
- Modify: `dashboard/backend/tests/test_admin_analytics_frontend.py`

**Interfaces:**
- Consumes: existing `adminTab` URL parameter and `AdminTabs.openAccountManagement(userId)` bridge.
- Produces: vertical roving tab navigation with `aria-orientation="vertical"`, Up/Down/Home/End keys, desktop labels, narrow icon tooltips, and unchanged Credits tabs.

- [ ] **Step 1: Write failing markup, keyboard, and Credits non-regression assertions**

```python
def test_admin_navigation_is_vertical_and_credits_tabs_are_unchanged():
    assert 'id="adminTabs"' in APP_HTML
    assert 'aria-orientation="vertical"' in APP_HTML
    assert 'class="admin-workspace"' in APP_HTML
    assert 'id="creditsTabs"' in APP_HTML
    assert 'aria-orientation="vertical"' not in credits_tabs_fragment(APP_HTML)


def test_admin_rail_uses_vertical_roving_keys():
    source = admin_tabs_source()
    for key in ("ArrowUp", "ArrowDown", "Home", "End"):
        assert key in source
    assert "ArrowLeft" not in admin_rail_key_handler(source)
    assert "ArrowRight" not in admin_rail_key_handler(source)
```

- [ ] **Step 2: Run frontend contract tests and verify vertical assertions fail**

Run: `pytest -q dashboard/backend/tests/test_admin_analytics_frontend.py -k "navigation or rail"`

Expected: FAIL because the Admin tabs are horizontal and use Left/Right.

- [ ] **Step 3: Add icon-plus-label rail markup and workspace shell**

```html
<div class="admin-workspace">
  <nav id="adminTabs" class="admin-rail" aria-label="Admin sections" role="tablist" aria-orientation="vertical">
    <button id="adminTabAnalytics" class="admin-tab is-active" type="button" role="tab" aria-selected="true" aria-controls="adminPanelAnalytics" tabindex="0" data-admin-tab="analytics" aria-label="Analytics" title="Analytics">
      <svg aria-hidden="true"><use href="#icon-chart"/></svg><span>Analytics</span>
    </button>
    <button id="adminTabUsers" class="admin-tab" type="button" role="tab" aria-selected="false" aria-controls="adminPanelUsers" tabindex="-1" data-admin-tab="users" aria-label="Users" title="Users">
      <svg aria-hidden="true"><use href="#icon-users"/></svg><span>Users</span>
    </button>
  </nav>
  <div class="admin-workspace-content">
    <section id="adminPanelAnalytics" role="tabpanel" aria-labelledby="adminTabAnalytics" data-admin-panel="analytics"></section>
    <section id="adminPanelUsers" role="tabpanel" aria-labelledby="adminTabUsers" data-admin-panel="users" hidden></section>
    <section id="adminPanelProviders" role="tabpanel" aria-labelledby="adminTabProviders" data-admin-panel="providers" hidden></section>
    <section id="adminPanelActivity" role="tabpanel" aria-labelledby="adminTabActivity" data-admin-panel="activity" hidden></section>
  </div>
</div>
```

Move each existing tabpanel with all of its current descendants between the matching opening and closing tags shown above. Include Providers and Activity buttons in the locked order using existing sprite symbols. Do not edit the Credits/Billing `API Keys / Credits / Activity` tablist.

- [ ] **Step 4: Implement vertical keyboard behavior and responsive rail CSS**

```javascript
const keyOffsets = { ArrowDown: 1, ArrowUp: -1 };
if (event.key in keyOffsets) {
  event.preventDefault();
  const next = buttons[(index + keyOffsets[event.key] + buttons.length) % buttons.length];
  setTab(next.dataset.adminTab, { focus: true });
} else if (event.key === 'Home' || event.key === 'End') {
  event.preventDefault();
  const next = event.key === 'Home' ? buttons[0] : buttons[buttons.length - 1];
  setTab(next.dataset.adminTab, { focus: true });
}
```

```css
.admin-workspace { display: grid; grid-template-columns: minmax(10rem, 12rem) minmax(0, 1fr); align-items: start; }
.admin-rail { position: sticky; top: var(--header-offset); display: flex; flex-direction: column; }
@media (max-width: 700px) {
  .admin-workspace { grid-template-columns: 3.5rem minmax(0, 1fr); }
  .admin-rail .admin-tab span { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); }
}
```

Keep stable icon-button dimensions, visible focus, non-overlapping long labels, and existing `admin:tabchange`/Back/Forward synchronization.

- [ ] **Step 5: Run rail tests and commit**

Run: `pytest -q dashboard/backend/tests/test_admin_analytics_frontend.py -k "navigation or rail or responsive"`

Expected: PASS.

```bash
git add dashboard/frontend/app.html dashboard/frontend/js/admin-tabs.js dashboard/frontend/styles.css dashboard/backend/tests/test_admin_analytics_frontend.py
git commit -m "feat: add vertical admin navigation rail"
```

### Task 8: Build the User-Value Overview and Lazy Analysis Sections

**Files:**
- Create: `dashboard/frontend/js/admin-analytics-value.js`
- Modify: `dashboard/frontend/js/admin-analytics.js`
- Modify: `dashboard/frontend/app.html`
- Modify: `dashboard/frontend/styles.css`
- Modify: `dashboard/backend/tests/test_admin_analytics_frontend.py`
- Create: `dashboard/backend/tests/test_admin_analytics_value_frontend.py`

**Interfaces:**
- Consumes: Task 6 fixture-locked APIs and `AdminAnalytics.openProfile(userId)`.
- Produces: `window.AdminAnalyticsValue = { onEnter, refresh, syncAuth, applyUserFilters }`, always-visible lifecycle overview/Priority Users, and first-open lazy Retention/Commercial/Operational sections.

- [ ] **Step 1: Write failing endpoint, lazy-loading, URL-state, and partial-error tests**

```python
def test_value_client_uses_independent_endpoints():
    source = value_source()
    for endpoint in ("/lifecycle", "/retention", "/commercial", "/operational", "/users"):
        assert endpoint in source
    assert "Promise.allSettled" in source


def test_deep_sections_fetch_only_on_first_open():
    source = value_source()
    assert "loaded: false" in source
    assert "aria-expanded" in source
    assert "ensureDisclosureLoaded" in source
    assert "This section is temporarily unavailable." in source


def test_value_filters_are_deep_linkable():
    source = value_source()
    for key in ("analyticsLifecycle", "analyticsOperational", "analyticsCommercial", "analyticsUser", "analyticsProfile"):
        assert key in source
```

- [ ] **Step 2: Run value frontend tests and verify the module is absent**

Run: `pytest -q dashboard/backend/tests/test_admin_analytics_value_frontend.py dashboard/backend/tests/test_admin_analytics_frontend.py`

Expected: FAIL because `admin-analytics-value.js` and the new markup do not exist.

- [ ] **Step 3: Replace the overview markup with the approved information hierarchy**

```html
<section id="adminAnalyticsValueOverview" aria-labelledby="adminAnalyticsValueTitle">
  <header class="admin-value-header">
    <div><p class="credits-section-kicker">User value</p><h3 id="adminAnalyticsValueTitle">Analytics</h3></div>
    <button id="adminAnalyticsRulesOpen" class="credits-icon-btn" type="button" aria-haspopup="dialog" aria-controls="adminAnalyticsRulesDialog" title="How segments work" aria-label="How segments work"><svg aria-hidden="true"><use href="#icon-info"/></svg></button>
  </header>
  <div id="adminAnalyticsHeadline" class="admin-value-headline" aria-busy="true"></div>
  <section aria-labelledby="adminLifecycleDistributionTitle"><h4 id="adminLifecycleDistributionTitle">Lifecycle distribution</h4><div id="adminLifecycleDistribution"></div></section>
  <section aria-labelledby="adminLifecycleMovementTitle"><h4 id="adminLifecycleMovementTitle">Eight-week movement</h4><canvas id="adminLifecycleMovementChart" role="img" aria-describedby="adminLifecycleMovementTable"></canvas><table id="adminLifecycleMovementTable" class="sr-only"></table></section>
  <section aria-labelledby="adminPriorityUsersTitle"><h4 id="adminPriorityUsersTitle">Priority users</h4><div id="adminPriorityUsers"></div></section>
</section>
```

Add three native disclosure buttons with `aria-expanded="false"` and panels hidden by default. Add stable skeleton boxes, precise empty copy, section-local status/error/retry regions, coverage text, and `Incomplete data` labels.

- [ ] **Step 4: Implement independent fetch state, current-vs-range filters, drill-down, and stale retention**

```javascript
const state = {
  range: defaultUtcRange(),
  includeInternal: false,
  userFilters: { lifecycle: '', operational: '', commercial: '', query: '', profile: '' },
  sections: {
    lifecycle: { loaded: false, data: null, error: null, stale: false },
    users: { loaded: false, data: null, error: null, stale: false },
    retention: { loaded: false, data: null, error: null, stale: false },
    commercial: { loaded: false, data: null, error: null, stale: false },
    operational: { loaded: false, data: null, error: null, stale: false },
  },
};

async function refreshPrimary() {
  const results = await Promise.allSettled([fetchLifecycle(), fetchPriorityUsers()]);
  applySettledSection('lifecycle', results[0]);
  applySettledSection('users', results[1]);
}


async function handleAccessLost(error) {
  if (error?.status !== 401 && error?.status !== 403) return false;
  if (typeof window.refreshAuthUser === 'function') await window.refreshAuthUser();
  if (typeof window.navigateToPage === 'function') window.navigateToPage('home');
  return true;
}


async function ensureDisclosureLoaded(name) {
  const section = state.sections[name];
  if (section.loaded && !section.error) return renderSection(name);
  return loadSection(name, { keepStaleData: true });
}
```

Headline/segment requests send only range and inclusion. Operational sends provider/model/billing filters stored inside its disclosure. Segment count controls are buttons with `aria-pressed`; they and chart point controls call `applyUserFilters()` without changing current identity. Route any `401`/`403` through `handleAccessLost()` before rendering a section error. Render response text with `textContent`/DOM nodes only. Keep a stale successful response visible when refresh fails and show a stale indicator plus local retry.

- [ ] **Step 5: Render charts with equivalent hidden tables and stable responsive dimensions**

```javascript
function renderLifecycleMovement(series) {
  replaceHiddenMovementRows(series);
  const canvas = document.getElementById('adminLifecycleMovementChart');
  canvas.setAttribute('aria-label', describeLifecycleMovement(series));
  movementChart?.destroy();
  movementChart = new Chart(canvas, lifecycleChartConfig(series));
}
```

Use text labels/pattern-independent legend keys so color is not the only signal. Constrain the chart wrapper with a fixed responsive aspect ratio and never let loading/error copy resize it.

- [ ] **Step 6: Run value frontend tests and commit**

Run: `pytest -q dashboard/backend/tests/test_admin_analytics_value_frontend.py dashboard/backend/tests/test_admin_analytics_frontend.py`

Expected: PASS.

```bash
git add dashboard/frontend/js/admin-analytics-value.js dashboard/frontend/js/admin-analytics.js dashboard/frontend/app.html dashboard/frontend/styles.css dashboard/backend/tests/test_admin_analytics_value_frontend.py dashboard/backend/tests/test_admin_analytics_frontend.py
git commit -m "feat: build admin user value analytics overview"
```

### Task 9: Add Explainable Priority Users and Enriched User Profile

**Files:**
- Modify: `dashboard/frontend/js/admin-analytics-value.js`
- Modify: `dashboard/frontend/js/admin-analytics.js`
- Modify: `dashboard/frontend/app.html`
- Modify: `dashboard/frontend/styles.css`
- Modify: `dashboard/backend/tests/test_admin_analytics_value_frontend.py`
- Modify: `dashboard/backend/tests/test_admin_analytics_frontend.py`

**Interfaces:**
- Consumes: Task 6 enriched users/profile fixtures and existing `AdminTabs.openAccountManagement(userId)`.
- Produces: quick evidence dialog, complete rules dialog, lifecycle/operational tooltips, enriched Overview, unchanged independent Timeline/Runs/Usage/Sessions pagination, and working account-management link.

- [ ] **Step 1: Write failing evidence, focus-management, profile, and account-link tests**

```python
def test_rules_and_evidence_dialogs_are_named_and_focus_safe():
    assert 'id="adminAnalyticsRulesDialog"' in APP_HTML
    assert 'id="adminAnalyticsEvidenceDialog"' in APP_HTML
    source = value_source()
    for contract in ("showModal()", "Escape", "event.target === dialog", "returnFocus", "focus()"):
        assert contract in source


def test_profile_keeps_full_sections_and_account_management():
    source = profile_source()
    for section in ("overview", "timeline", "runs", "usage", "sessions"):
        assert section in source
    assert "openAccountManagement" in source
    assert "lifecycle_segment" in source
    assert "operational_state" in source
    assert "lifetime_net_purchased_micro" in source
```

- [ ] **Step 2: Run focused interaction tests and verify dialogs/profile fields fail**

Run: `pytest -q dashboard/backend/tests/test_admin_analytics_value_frontend.py dashboard/backend/tests/test_admin_analytics_frontend.py -k "dialog or profile or evidence or account"`

Expected: FAIL because the evidence dialogs and enriched facts are not wired.

- [ ] **Step 3: Render priority rows and lifecycle help from fixed display maps**

```javascript
const LIFECYCLE_RULES = Object.freeze({
  new: 'Account is 0–6 UTC days old and has no successful backtest.',
  onboarding: 'No successful backtest yet; account is no longer New and is not inactive.',
  growing: 'Activated and active in the last 7 UTC days, below the Core repeat-value threshold.',
  core: 'At least 3 active days and 3 successful backtests in 30 UTC days, active in the last 7 days.',
  at_risk: 'Last meaningful activity was 8–29 UTC days ago.',
  dormant: 'Last meaningful activity was at least 30 UTC days ago.',
});

function priorityRow(user) {
  const row = node('button', 'admin-priority-user');
  row.type = 'button';
  row.dataset.userId = String(user.user_id);
  row.setAttribute('aria-haspopup', 'dialog');
  row.append(identityNode(user), lifecycleBadge(user), operationalBadge(user), commercialBadge(user), reasonNode(user));
  return row;
}
```

Every lifecycle badge has the fixed concise rule in an accessible tooltip. Selecting a priority row opens user-specific display-safe evidence; `Open full analytics profile` calls `AdminAnalytics.openProfile(userId)` and the separate `Open account management` control calls the existing Admin bridge.

- [ ] **Step 4: Implement native-dialog focus return, Escape, and backdrop close**

```javascript
function openDialog(dialog, opener) {
  dialogState.set(dialog.id, opener);
  dialog.showModal();
  dialog.querySelector('[data-dialog-initial-focus]')?.focus();
}

function closeDialog(dialog) {
  dialog.close();
  const opener = dialogState.get(dialog.id);
  if (opener?.isConnected) opener.focus();
}

dialog.addEventListener('click', (event) => {
  if (event.target === dialog) closeDialog(dialog);
});
```

Use native `<dialog>` focus containment. Give each dialog `aria-labelledby`, a close icon with an accessible name, and no response-derived HTML.

- [ ] **Step 5: Enrich profile Overview while preserving independent activity state**

```javascript
function renderValueProfile(profile) {
  renderLifecycleEvidence(profile.lifecycle);
  renderOperationalEvidence(profile.operational);
  renderCommercialFacts(profile.commercial, profile.balances);
  renderActivationFacts(profile.activated_at, profile.active_days_30d, profile.successful_backtests_30d);
  renderLifecycleTransitions(profile.recent_lifecycle_transitions);
}

function openAccountManagement(userId) {
  window.AdminTabs.openAccountManagement(userId);
}
```

Do not merge Timeline/Runs/Usage/Sessions cursors or error state. A failed section retains its already-rendered items and shows only `More activity is temporarily unavailable.` in that section.

- [ ] **Step 6: Run interaction/profile tests and commit**

Run: `pytest -q dashboard/backend/tests/test_admin_analytics_value_frontend.py dashboard/backend/tests/test_admin_analytics_frontend.py`

Expected: PASS.

```bash
git add dashboard/frontend/js/admin-analytics-value.js dashboard/frontend/js/admin-analytics.js dashboard/frontend/app.html dashboard/frontend/styles.css dashboard/backend/tests/test_admin_analytics_value_frontend.py dashboard/backend/tests/test_admin_analytics_frontend.py
git commit -m "feat: explain admin user value signals"
```

### Task 10: Integration, Privacy, Accessibility, and Visual Regression

**Files:**
- Modify: `dashboard/frontend/app.html`
- Modify: `dashboard/frontend/styles.css`
- Modify: `dashboard/backend/tests/test_admin_analytics_api.py`
- Modify: `dashboard/backend/tests/test_admin_analytics_frontend.py`
- Modify: `dashboard/backend/tests/test_admin_analytics_value_frontend.py`
- Modify: `dashboard/backend/tests/domain/analytics/test_privacy.py`

**Interfaces:**
- Consumes: all previous tasks.
- Produces: cache-busted production wiring, verified desktop/mobile layout, complete regression coverage, and a clean PR-ready branch.

- [ ] **Step 1: Add final privacy, partial-availability, and semantic regression tests**

```python
def test_value_fixtures_and_client_exclude_sensitive_fields():
    combined = value_source() + "\n" + "\n".join(path.read_text() for path in FIXTURE_DIR.glob("*.json"))
    for prohibited in ("api_key", "password", "network_hash", "raw_user_agent", "provider_response_body", "credential_ciphertext", "strategy_content", "prompt_text"):
        assert prohibited not in combined


def test_one_failed_section_does_not_blank_other_sections():
    source = value_source()
    assert "Promise.allSettled" in source
    assert "applySettledSection" in source
    assert "keepStaleData: true" in source


def test_charts_have_equivalent_tables_and_disclosures_have_state():
    assert 'aria-describedby="adminLifecycleMovementTable"' in APP_HTML
    assert 'class="sr-only"' in lifecycle_table_fragment(APP_HTML)
    assert 'aria-expanded="false"' in APP_HTML
```

- [ ] **Step 2: Run the full Analytics suite before visual inspection**

Run: `pytest -q dashboard/backend/tests/domain/analytics dashboard/backend/tests/test_admin_analytics_api.py dashboard/backend/tests/test_admin_analytics_frontend.py dashboard/backend/tests/test_admin_analytics_value_frontend.py dashboard/backend/tests/test_analytics_maintenance.py`

Expected: PASS.

- [ ] **Step 3: Bump only touched static assets and run the local app**

```html
<link rel="stylesheet" href="styles.css?v=131">
<script src="js/admin-analytics.js?v=3" defer></script>
<script src="js/admin-analytics-value.js?v=1" defer></script>
<script src="js/admin-tabs.js?v=4" defer></script>
```

Run: `uvicorn dashboard.backend.app:app --host 127.0.0.1 --port 8000`

Expected: the app serves on `http://127.0.0.1:8000`; use a synthetic local Admin and synthetic fixture-backed repositories only.

- [ ] **Step 4: Verify keyboard and screenshots at desktop and narrow widths**

Use Playwright at `1440x900`, `1024x768`, and `390x844`. Verify:

```text
Admin rail: Up, Down, Home, End; selected tab and focus stay synchronized.
Rules/evidence dialogs: opener focus -> dialog -> Escape/backdrop/close -> opener focus.
Disclosures: first open fetches once; later open reuses success; retry is local.
Profile: Overview, Timeline, Runs, Usage, Sessions, Back, and Open account management work.
Layout: no overlap, no clipped longest rule/reason, stable chart/skeleton height, narrow icon rail stays left.
```

Save screenshots outside the repository or under `.superpowers/`; do not stage them. Inspect the browser console and network panel for uncaught errors, duplicate disclosure requests, and unexpected mutation methods.

- [ ] **Step 5: Run the complete backend test suite and inspect the diff**

Run: `pytest -q dashboard/backend/tests`

Expected: PASS.

Run: `git diff --check && git status --short && git diff --stat origin/main..HEAD`

Expected: no whitespace errors; `dashboard/storage/data/backtest.db`, `.superpowers/`, `work/`, secrets, and screenshots are absent from staged/tracked changes.

- [ ] **Step 6: Commit final integration fixes**

```bash
git add dashboard/frontend/app.html dashboard/frontend/styles.css dashboard/backend/tests/test_admin_analytics_api.py dashboard/backend/tests/test_admin_analytics_frontend.py dashboard/backend/tests/test_admin_analytics_value_frontend.py dashboard/backend/tests/domain/analytics/test_privacy.py
git commit -m "test: verify admin user value analytics"
```

## Final Verification Checklist

- [ ] `Activated`, `Core`, `At risk`, and `Paid` headline counts use current fixed-window identity and respect internal-account inclusion.
- [ ] All six lifecycle segments, eight-week movement, transition callout, and Priority Users remain visible before opening a disclosure.
- [ ] Retention, Commercial Value, and Operational Health are collapsed by default, load independently on first open, retain stale success, and retry locally.
- [ ] Week 1/2/4 retention uses first-success UTC cohorts, mature denominators, and meaningful return activity.
- [ ] Commercial revenue is purchase minus refund only; Grants, balances, consumption, and platform cost remain separate.
- [ ] Blocked, Needs attention, healthy At risk, and healthy Onboarding priority order is stable and commercial-value aware.
- [ ] Lifecycle tooltips, full rules dialog, quick user evidence, and full profile all explain their result with display-safe evidence.
- [ ] Current lifecycle and operational axes never overwrite the legacy status contract during migration.
- [ ] Date filters never change current lifecycle or commercial tier.
- [ ] Partial/backfill evidence displays coverage and `Incomplete data`; it never renders missing data as zero.
- [ ] SQLite and PostgreSQL migrations/repositories are equivalent and idempotent.
- [ ] User-level daily history is aggregated anonymously before 180-day deletion.
- [ ] Admin profile access auditing and `Open account management` still work.
- [ ] Vertical rail, dialogs, disclosures, charts, tables, focus, responsive layout, and profile tabs meet the documented ARIA and keyboard contracts.
- [ ] Credits/Billing navigation is unchanged.
- [ ] No prohibited secret, production data, database, `.superpowers/`, `work/`, or screenshot artifact is committed.
