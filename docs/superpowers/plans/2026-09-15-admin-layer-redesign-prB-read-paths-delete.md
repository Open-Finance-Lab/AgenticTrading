# Admin Layer PR B: Read Paths, Then Delete the Old Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move every one of the nine `/api/admin/analytics/*` read paths off `user_analytics_snapshots`, `user_lifecycle_daily_snapshots` and the raw `analytics_events` scans and onto the tables PR A created (`user_daily_facts`, `user_activity`, `lifecycle_transitions`) plus the anonymous `analytics_daily_rollups`, holding every response shape byte-identical except the two D20 items — then, in the same PR and only after every read has moved, delete the five-state model: the two snapshot tables on both twins, `states.py`, `lifecycle_backfill.py`, `maintenance.py`'s reaper sweep, `rollup_day`'s `user_state_count` rows, `user_state_counts` on `/overview` and the `status` filter on `/users` with its one dead frontend sender.

**Architecture:** Cross-user numbers come from set-based reads over `user_daily_facts` (one row per user per UTC day, complete through yesterday) and `analytics_daily_rollups`; the overview's only raw-event read is one scan bounded to the current UTC day; per-user live facts (segment, operational state) are computed at read time from `user_activity` and trailing sums over the facts table for exactly one user or one page of users; the users list is filtered, sorted and paginated in SQL on both store twins. New read methods land on `ValueAnalyticsStore` and `PostgresValueAnalyticsStore` together, and the ledger aggregate lands on `CreditsStore`/`PostgresCreditsStore`, because the analytics package never opens another domain's connection (design §6.11 rule 7, §6.14). The deletions are the last four tasks so that no intermediate commit reads a dropped table.

**Tech Stack:** Python 3, FastAPI, Pydantic v2, SQLite, PostgreSQL/psycopg 3, pytest (SQLite tier always; `@pg_only` tier on `TEST_POSTGRES_URL`), vanilla JS with no build step (one dead query line removed; tests via `node -e` are untouched by this PR).

**Spec:** `docs/superpowers/specs/2026-09-15-admin-layer-redesign-design.md` — implements §6.10 ("Read paths", the "After PR B" column), §6.7 (the long-term rollup encoding: `metric_name` carries the dimension name, `user_state` the dimension value), §6.15 items 2 (the overview's current-day scan narrowed to one UTC day) and 3 (the sessions section's 30-day window), §9 (the "Kept, re-sourced (PR B)" line, and the "Added" table's **service methods only** — nothing lands on a response model), §10.2 (the PR B contract rule: byte-identical except D20), §11 (re-seed `user_activity`, then drop the snapshot tables; retention expires `user_daily_facts` and `lifecycle_transitions`, never `user_activity`), §12 items 3 (the backend half: PR B does not build on the in-app panel), 6 (the seed re-run) and 7 (drop only after the reads have moved), §13 row **B** including its "Must not" column, D13 (`top_failure_categories` from rollups `error_category`), D14 (billing-lane mix from rollups `billing_mode`) and D20 (the one permitted contract break). §4.4 is the baseline this plan moves away from.

## Global Constraints

- Never commit `dashboard/storage/data/backtest.db` — a bare backend import runs `CREATE TABLE IF NOT EXISTS` (and, in this PR, `DROP TABLE IF EXISTS`) against `DATABASE_PATH` and rewrites that tracked seed file; stage files by name (`git add <path> <path>`), never `git add -A`, never a bare `git add -u`. Task 14 drops tables and runs the full suite immediately before committing, which is exactly the moment the seed DB is dirty — check `git status --short` before every `git add` there.
- Run every `pytest` invocation from the repo root.
- Node-driven frontend tests are `pytest.mark.skipif(shutil.which("node") is None, ...)`; this PR edits no JavaScript. PR C (merged before this PR since the 2026-09-16 re-ordering) deleted `admin-analytics.js`, `admin-analytics-value.js` and `admin-analytics.html`, and the `/admin` page's four modules are untouched here. The node tier must still run green: the page's renderer tests (`test_admin_overview_frontend.py`, `test_admin_users_frontend.py`) feed the committed fixtures, which Task 16 edits.
- No `?v=` bump: this PR changes no frontend file. (Before the re-ordering Task 16 bumped `admin-analytics.js?v`; that module no longer exists.)
- Both store twins change together: every read method added to `ValueAnalyticsStore` (`value_repository.py`) is added to `PostgresValueAnalyticsStore` (`value_repository_postgres.py`, PR T) in the same task with an identical signature, and `CreditsStore.sum_ledger_by_day` lands on `PostgresCreditsStore` in the same commit. `tests/test_store_twin_parity.py` compares public method sets and signatures; CI's Postgres tier (`.github/workflows/ci.yml`, `TEST_POSTGRES_URL`) must be green before merge (design §13 row B; PR T's absence-direction check also fails on a stale allowlist entry, see Task 17).
- Contract: byte-identical responses on all nine routes except `user_state_counts` leaving `GET /overview` and `status` leaving `GET /users` (design §10.2, D20). No new field on any response model, no renamed query parameter. `tests/test_admin_analytics_api.py`, `test_admin_analytics_frontend.py` (the fixture-shape tests as PR C rewrote them), the `/admin` page's node tests (`test_admin_shell_frontend.py`, `test_admin_live_frontend.py`, `test_admin_overview_frontend.py`, `test_admin_users_frontend.py`) and the JSON fixtures under `tests/fixtures/admin_analytics/` — committed and the `target/` copies PR C added — are the conformance oracle; the "Acceptance" section lists every assertion this plan changes and why.
- Frontend: none. Design §13 row B's one permitted edit — the dead `status` query in `admin-analytics.js::attentionQuery` — is already done: PR C deleted the module. Task 16's frontend steps are therefore gone; if `rg -n "params.set\('status'" dashboard/frontend/js` prints anything, PR C has not merged and this branch was cut in the wrong order.
- No `cohort`, anywhere (D9). Where the superseded 2026-09-12 plan (`git show c3bbf2ed:docs/superpowers/plans/2026-09-12-user-analytics-architecture.md`) wrote `cohort`, this plan writes `user_group`; the six long-term rollup metric names are the design's (`*_by_tier`, `*_by_user_group`), not the old plan's `tier_*`/`cohort_*`.
- Log lines use `print()`, never `logger`; a failure line carries an exception class only (`category={type(exc).__name__[:80]}`), never a message body.
- Ordering (§13, re-ordered 2026-09-16): PR 0, PR C, PR T and PR A are merged before this branch is cut; this PR must not merge until the daily job has written at least eight days of `user_daily_facts` in prod (see "Merge preconditions").
- Deletion order inside this PR: Tasks 1–13 leave every table and module in place; Tasks 14–17 delete. Each commit in between must serve all nine routes.

## Merge preconditions

Run these against the prod users database (the Neon project behind `USERS_DATABASE_URL`; the analytics tables live there, design §6.4) before pressing merge. The PR stays a **draft** until both pass — publish the gate where GitHub enforces it, not in a comment.

```sql
-- 1. At least eight complete days of facts written by the daily job (not the PR A history copy,
--    which is stamped data_quality = 'partial'). Expect days >= 8 and max_day = yesterday (UTC).
SELECT COUNT(DISTINCT snapshot_date) AS days,
       MIN(snapshot_date) AS min_day,
       MAX(snapshot_date) AS max_day
FROM user_daily_facts
WHERE data_quality = 'complete'
  AND snapshot_date >= to_char((now() AT TIME ZONE 'UTC')::date - 9, 'YYYY-MM-DD');

-- 2. The day claim is caught up: cursor is yesterday and the job is not stuck mid-run.
SELECT job_name, cursor, status, updated_at
FROM analytics_projection_jobs
WHERE job_name = 'analytics_daily_facts';

-- 3. The activity table PR A seeded is populated (otherwise Task 11's live segments have nothing to read).
SELECT COUNT(*) AS rows_total,
       COUNT(activated_at) AS activated
FROM user_activity;
```

Expected: (1) `days >= 8`, `max_day` equals yesterday's UTC date; (2) `status = 'complete'` and `cursor` equals the same date; (3) `rows_total > 0` and `activated > 0`. If (1) shows fewer than eight days, wait; if (2) shows `running` with an `updated_at` more than two hours old, the job is wedged and PR A's owner must look before this PR lands on top of it.

## Interfaces this plan takes from PR T and PR A

PR A's plan (`2026-09-15-admin-layer-redesign-prA-rewrite-create.md`) was being written in parallel with this one and was not present in the worktree when this plan was drafted, so the names below are derived from the design doc §6 and the superseded plan's tasks B1, B4, B5, B8 (`git show c3bbf2ed:docs/superpowers/plans/2026-09-12-user-analytics-architecture.md`). **Before starting Task 1, run `rg -n "class UserDailyFact|class UserActivity|class LifecycleTransitionRow|class RecentFactTotals|def sum_recent_facts|def list_activity|def get_activity|def list_facts_for_date|def upsert_daily_facts|def append_lifecycle_transitions|def build_lifecycle_inputs|def build_value_analytics_store|def seed_user_activity" dashboard/backend/domain/analytics/` and, where PR A landed a different name, substitute it throughout — the behaviour each name stands for is what the tasks depend on, and every one is exercised by a test in this plan, so a wrong name fails loudly at Step 2 of the task that uses it.**

| Name | Module | Shape assumed here |
|---|---|---|
| `ValueAnalyticsStore` / `PostgresValueAnalyticsStore` | `value_repository.py` / `value_repository_postgres.py` (PR T) | `__init__(self, analytics_base=None, credits_base=None, provider_base=None, agent_base=None, run_base=None)`; module-level `_user_clause(ids, postgres) -> tuple[str, list]` and `_fetchall(conn, postgres, sql, params)` (PR T moved them out of the class; today they are staticmethods at `value_repository.py:710-723`) |
| `build_value_analytics_store(analytics_base=None, credits_base=None, provider_base=None, agent_base=None, run_base=None)` | `value_repository.py` (PR T) | picks the twin by `hasattr(analytics_base, "database_url")` |
| `UserActivity` | `value_repository.py` (PR A, B4) | `user_id, activated_at: datetime \| None, last_meaningful_activity_at: datetime \| None, updated_at` |
| `UserDailyFact` | `value_repository.py` (PR A, B1/B8) | fields = the §6.7 columns: `snapshot_date, user_id, lifecycle_segment, lifecycle_reason_code, operational_state, operational_reason_code, tier, user_group, active: bool, runs_requested, runs_completed, runs_failed, runs_cancelled, operator_cost_micro, own_spend_micro, data_quality, calculated_at` |
| `LifecycleTransitionRow` | `value_repository.py` (PR A, B8) | `user_id, snapshot_date, from_segment, to_segment, inactive_days, data_quality, created_at` |
| `RecentFactTotals` | `value_repository.py` (PR A, B5) | `active_days, successful_backtests, runs_requested, runs_completed, runs_failed, runs_cancelled, operator_cost_micro, own_spend_micro, days_present, last_active_date: date \| None` |
| `store.get_activity(user_id) -> UserActivity \| None`, `store.list_activity(user_ids) -> dict[int, UserActivity]` | both twins (PR A, B4) | batched by `_user_clause` |
| `store.sum_recent_facts(user_ids, *, start: date, end: date) -> dict[int, RecentFactTotals]` | both twins (PR A, B5) | `start`/`end` inclusive UTC dates |
| `store.list_facts_for_date(day) -> list[UserDailyFact]`, `store.upsert_daily_facts(rows) -> int`, `store.append_lifecycle_transitions(rows) -> int` | both twins (PR A, B8) | used by Task 13 and by every test fixture below |
| `build_lifecycle_inputs(user_id, *, created_at, activity, totals, as_of, day_activity_at=None) -> LifecycleInputs` | `lifecycle_reads.py` (PR A, B5) | the live read path passes `as_of=now` |
| `seed_user_activity_from_snapshots(*, value_store) -> int` | `facts_migration.py` (PR A, §11) | idempotent: `MIN` on `activated_at`, `MAX` on `last_meaningful_activity_at` |
| the daily job calls `rollup_day(D, now=…)` once per day | `daily_job.py` (PR A) | so Task 5's change to `rollup_day` reaches prod the night after deploy |

---

### Task 1: `resolve_group_badge` and two operational text helpers in `lifecycle.py`

**Files:**
- Modify: `dashboard/backend/domain/analytics/lifecycle.py` — add three functions after `activation_cohort_week` (lines 367-371) and extend `__all__` (lines 374-387)
- Test: `dashboard/backend/tests/domain/analytics/test_lifecycle.py` (extend)

**Interfaces:**
- Consumes: `CommercialTier`, `_OPERATIONAL_REASONS`, `_OPERATIONAL_EVIDENCE`, `OperationalResult` (all existing in `lifecycle.py`).
- Produces:
  - `resolve_group_badge(*, role: str, user_group: str, tier: CommercialTier) -> str` (D11, §6.2 precedence: `admin` → the `user_group` when it is not `unknown` → `paid` when tier is not `unpaid` → `free`). Task 11's `get_user_metrics` consumes it; PR D puts it on the payload.
  - `operational_reason(reason_code: str) -> str` — the display text for a stored `operational_reason_code`. Tasks 3, 5, 8 and 10 consume it.
  - `operational_result_from_code(reason_code: str, *, as_of: datetime) -> OperationalResult` — rebuilds a display-safe `OperationalResult` from a stored code (yesterday's fact row), with the static evidence line where one exists and the reason text otherwise. Tasks 3 and 10 consume it.

The 09-12 plan's `resolve_group_badge` took `cohort: str | None`; the design struck the cohort axis (D9) and `user_group` is `NOT NULL DEFAULT 'unknown'`, so the "no label" case is the string `"unknown"`, not `None`. The 09-12 plan's sentence for PR C is kept verbatim in the docstring: *Do not recompute the precedence in JavaScript: the server owns that rule.*

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/backend/tests/domain/analytics/test_lifecycle.py` (the file already imports `pytest` and `from dashboard.backend.domain.analytics.lifecycle import ...`; extend that import with `operational_reason`, `operational_result_from_code`, `resolve_group_badge`):

```python
@pytest.mark.parametrize(
    "role,user_group,tier,expected",
    [
        ("admin", "internal", "high_value", "admin"),
        ("admin", "unknown", "unpaid", "admin"),
        ("user", "partner", "unpaid", "partner"),
        ("user", "competition", "invested", "competition"),
        ("user", "unknown", "starter", "paid"),
        ("user", "unknown", "high_value", "paid"),
        ("user", "unknown", "unpaid", "free"),
    ],
)
def test_group_badge_precedence(role, user_group, tier, expected):
    assert resolve_group_badge(role=role, user_group=user_group, tier=tier) == expected


def test_operational_reason_text_is_the_rule_table_copy():
    assert operational_reason("account_restricted") == (
        "The Credits account is restricted from model spending."
    )
    assert operational_reason("no_supported_issue") == (
        "No supported current operational issue was detected."
    )


def test_unknown_operational_reason_code_is_display_safe():
    """A code the rule table does not know must not raise on a read path."""
    text = operational_reason("synthetic_unknown_code")
    assert text == "An operational issue was recorded."
    assert "synthetic" not in text


def test_operational_result_from_code_carries_static_evidence():
    as_of = datetime(2026, 9, 14, 0, 0, tzinfo=timezone.utc)
    blocked = operational_result_from_code("billing_lane_unavailable", as_of=as_of)
    attention = operational_result_from_code(
        "three_consecutive_failed_runs", as_of=as_of
    )
    healthy = operational_result_from_code("no_supported_issue", as_of=as_of)

    assert blocked.state == "blocked"
    assert blocked.evidence == ("Usable billing lanes: 0.",)
    assert attention.state == "needs_attention"
    # No static evidence line exists for this code; the reason text stands in.
    assert attention.evidence == (
        "At least three consecutive terminal runs failed within 24 hours.",
    )
    assert healthy.state == "healthy"
    assert healthy.calculated_at == as_of
```

`datetime`/`timezone` are already imported at the top of `test_lifecycle.py`; if not, add `from datetime import datetime, timezone`.

- [ ] **Step 2: Run them and confirm the expected failure**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_lifecycle.py -v -k "group_badge or operational_reason or from_code"
```

Expected: **FAIL** at import — `ImportError: cannot import name 'resolve_group_badge'`.

- [ ] **Step 3: Implement**

In `dashboard/backend/domain/analytics/lifecycle.py`, after `activation_cohort_week` (line 371) add:

```python
_OPERATIONAL_STATE_BY_REASON: dict[str, OperationalState] = {
    "account_restricted": "blocked",
    "billing_lane_unavailable": "blocked",
    "provider_disabled": "blocked",
    "invalid_default_credential": "needs_attention",
    "three_consecutive_failed_runs": "needs_attention",
    "run_deadline_exceeded": "needs_attention",
    "no_supported_issue": "healthy",
}
_UNKNOWN_OPERATIONAL_REASON = "An operational issue was recorded."


def resolve_group_badge(
    *,
    role: str,
    user_group: str,
    tier: CommercialTier,
) -> str:
    """The single group label admin views show for one user.

    Three orthogonal axes, one badge, fixed precedence (design §6.2, D11): an
    admin reads as an admin, because that is the fact that changes what they
    can do; a named ``user_group`` beats tier because a human assigned it;
    ``unknown`` is the column's default, not a label, so it falls through to
    the tier. Do not recompute the precedence in JavaScript: the server owns
    that rule, and two owners of one rule is how the season-number badge and
    its banner came to disagree.
    """
    if role == "admin":
        return "admin"
    if user_group and user_group != "unknown":
        return user_group
    return "free" if tier == "unpaid" else "paid"


def operational_reason(reason_code: str) -> str:
    """Display text for a stored operational reason code, never the code itself."""

    return _OPERATIONAL_REASONS.get(reason_code, _UNKNOWN_OPERATIONAL_REASON)


def operational_result_from_code(
    reason_code: str,
    *,
    as_of: datetime,
) -> OperationalResult:
    """Rebuild a display-safe result from yesterday's stored reason code.

    ``user_daily_facts`` stores the code, not the evidence: the one dynamic
    evidence line (the consecutive-failure count) is not recoverable from a
    code, so it falls back to the reason text rather than inventing a number.
    """

    calculation_time = _require_utc(as_of, "as_of")
    state = _OPERATIONAL_STATE_BY_REASON.get(reason_code, "needs_attention")
    reason = operational_reason(reason_code)
    evidence = _OPERATIONAL_EVIDENCE.get(reason_code, reason)
    return OperationalResult(
        state=state,
        reason_code=reason_code,
        reason=reason,
        evidence=(evidence,),
        calculated_at=calculation_time,
    )
```

Extend `__all__` (lines 374-387) with `"operational_reason"`, `"operational_result_from_code"`, `"resolve_group_badge"` in alphabetical position.

- [ ] **Step 4: Run and confirm the pass**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_lifecycle.py -v
```

Expected: **PASS**, all cases including the pre-existing ones (nothing above changes `calculate_lifecycle` or `calculate_operational_state`).

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/analytics/lifecycle.py dashboard/backend/tests/domain/analytics/test_lifecycle.py
git commit -m "$(cat <<'EOF'
feat: add resolve_group_badge and operational reason helpers

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Set-based read methods over the fact tables, on both twins

**Files:**
- Modify: `dashboard/backend/domain/analytics/value_repository.py` — models after `CurrentOperationalFacts` (line 143), SQL templates at module scope, methods on `ValueAnalyticsStore` after `sum_recent_facts` (PR A); extend `__all__` (line 1118)
- Modify: `dashboard/backend/domain/analytics/value_repository_postgres.py` — the same ten methods on `PostgresValueAnalyticsStore`
- Test: `dashboard/backend/tests/domain/analytics/test_fact_reads.py` (create)
- Test: `dashboard/backend/tests/domain/analytics/test_repository_postgres.py` (extend with one `@pg_only` case that runs the shared contract)

**Interfaces:**
- Consumes: `user_daily_facts`, `lifecycle_transitions`, `user_activity` (PR A tables); `users` and `analytics_subject_settings` (the analytics store already reads both in `list_excluded_user_ids`, `repository.py:529-547`); `UserDailyFact`, `LifecycleTransitionRow`, `upsert_daily_facts`, `append_lifecycle_transitions`, `record_activity` (PR A) in the tests.
- Produces, on both twins with identical signatures:
  - `count_states_for_date(day: date) -> list[FactStateCount]` — one `GROUP BY operational_state, operational_reason_code, lifecycle_segment` over yesterday's rows. Serves `/operational.operational_state_counts`, `top_operational_reasons`, `/lifecycle.segment_counts`, and (until Task 16) the `user_state_counts` bridge.
  - `count_segments_by_date(*, start: date, end: date) -> list[SegmentDayCount]` — `/lifecycle` weekly and movement series.
  - `count_transitions(*, start: date, end: date) -> list[TransitionCount]` — `/lifecycle.transitions`.
  - `list_attention_candidates(*, day: date, window_start: date, limit: int) -> list[AttentionCandidate]` — `/overview.users_needing_attention` (D20: sourced from `operational_state IN ('blocked','needs_attention')`).
  - `sum_facts_by_user_group(*, start: date, end: date) -> dict[str, GroupFactTotals]` — `/groups` run counts and operator cost.
  - `list_active_dates(user_ids: Sequence[int], *, start: date, end: date) -> dict[int, list[date]]` — `/retention` cells.
  - `list_activations(*, start: datetime, end: datetime, include_internal: bool) -> dict[int, datetime]` — `/retention` cohort membership from `user_activity.activated_at`.
  - `count_activated_users(*, include_internal: bool) -> int` — `/lifecycle.headline.activated_users`.
  - `list_user_transitions(user_id: int, *, start: date, end: date) -> list[LifecycleTransitionRow]` — the profile's `recent_lifecycle_transitions`.
  - `list_transitions_for_date(day: date) -> list[LifecycleTransitionRow]` — Task 13's long-term rollup.
  - Models: `FactStateCount`, `SegmentDayCount`, `TransitionCount`, `AttentionCandidate`, `GroupFactTotals`.

Every method is exactly one statement (two for `sum_facts_by_user_group`, whose repeat-user count needs a `HAVING`), takes no per-user loop, and the ones keyed on ids batch through `_user_clause`. The SQL is dialect-neutral except for the placeholder and the id clause, so each template lives once at module scope in `value_repository.py` with a `{ph}` token and both twins render it — the Postgres module imports the templates, which is the same relationship PR T gave it to `_user_clause`. Boolean `active` is `INTEGER` on SQLite and `BOOLEAN` on Postgres, so the SQL says `WHERE active` and `CASE WHEN active THEN 1 ELSE 0 END` (both dialects accept an integer or boolean there) and never `active = 1`. `analytics_subject_settings.excluded` is likewise `WHERE excluded`, never `= 1`. The facts table already excludes admins and excluded users at write time (§6.7), so `include_internal` cannot re-include them from facts; the two methods that read `user_activity` (which carries every user) take the flag and apply the same exclusion `list_excluded_user_ids` applies today.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/domain/analytics/test_fact_reads.py`:

```python
"""Set-based reads over user_daily_facts, lifecycle_transitions and user_activity.

Every read here is one statement over the whole population (or one page of
it). The SQLite cases run always; ``assert_fact_read_contract`` is also driven
by the Postgres twin from test_repository_postgres.py under @pg_only.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.value_repository import (
    LifecycleTransitionRow,
    UserDailyFact,
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


def seed_population(store, users, *, admin_id: int) -> dict[str, int]:
    """Five users plus an admin. Returns the ids by role in the scenario."""

    ids = {}
    for key, email, group in (
        ("blocked", "blocked@example.test", "partner"),
        ("attention", "attention@example.test", "partner"),
        ("core", "core@example.test", "organic"),
        ("dormant", "dormant@example.test", "unknown"),
        ("fresh", "fresh@example.test", "organic"),
    ):
        user = users.create_user(email, key.title(), "SecurePass1!")
        users.apply_admin_patch(int(user["id"]), user_group=group)
        ids[key] = int(user["id"])
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
                 runs_requested=0, runs_completed=0, data_quality="partial"),
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


def test_sqlite_fact_reads_follow_the_contract(tmp_path):
    store, users, admin_id = _sqlite_fixture(tmp_path)
    ids = seed_population(store, users, admin_id=admin_id)
    assert_fact_read_contract(store, ids)


def test_fact_reads_are_one_statement_per_call(tmp_path):
    """No read here may issue a per-user statement (design §6.11 rule 6)."""

    store, users, admin_id = _sqlite_fixture(tmp_path)
    ids = seed_population(store, users, admin_id=admin_id)
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
```

Then add to `dashboard/backend/tests/domain/analytics/test_repository_postgres.py`, after `test_postgres_user_value_projection_round_trip` (line 214 today; PR T moved its body onto `PostgresValueAnalyticsStore`):

```python
@pg_only
def test_postgres_fact_reads_follow_the_contract(postgres_contract_store):
    from dashboard.backend.domain.analytics.value_repository_postgres import (
        PostgresValueAnalyticsStore,
    )
    from dashboard.backend.tests.domain.analytics.test_fact_reads import (
        assert_fact_read_contract,
        seed_population,
    )
    from dashboard.backend.users_postgres import PostgresUserStore

    store, admin_id, _user_id = postgres_contract_store
    value_store = PostgresValueAnalyticsStore(
        analytics_base=store,
        credits_base=object(),
        provider_base=object(),
        agent_base=object(),
        run_base=object(),
    )
    users = PostgresUserStore(store.database_url)
    ids = seed_population(value_store, users, admin_id=admin_id)
    assert_fact_read_contract(value_store, ids)
```

- [ ] **Step 2: Run them and confirm the expected failure**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_fact_reads.py -v
```

Expected: **FAIL** with `AttributeError: 'ValueAnalyticsStore' object has no attribute 'count_states_for_date'` (the seed helpers succeed because PR A's writers exist).

- [ ] **Step 3: Add the models and SQL templates**

In `dashboard/backend/domain/analytics/value_repository.py`, after `class CurrentOperationalFacts` (ends line 143) add:

```python
class FactStateCount(BaseModel):
    """One (operational state, reason, segment) cell of yesterday's population."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operational_state: OperationalState
    operational_reason_code: str
    lifecycle_segment: LifecycleSegment
    users: int = Field(ge=0)
    partial_rows: int = Field(default=0, ge=0)


class SegmentDayCount(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot_date: date
    lifecycle_segment: LifecycleSegment
    users: int = Field(ge=0)
    partial_rows: int = Field(default=0, ge=0)


class TransitionCount(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    from_segment: LifecycleSegment
    to_segment: LifecycleSegment
    users: int = Field(ge=0)
    partial_rows: int = Field(default=0, ge=0)


class AttentionCandidate(BaseModel):
    """A blocked or needs-attention user with their trailing-window run counts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: int = Field(gt=0)
    display_name: str
    email: str
    created_at: datetime
    operational_state: OperationalState
    operational_reason_code: str
    recent_runs: int = Field(default=0, ge=0)
    recent_failures: int = Field(default=0, ge=0)


class GroupFactTotals(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    total_runs: int = Field(default=0, ge=0)
    operator_cost_micro: int = Field(default=0, ge=0)
    successful_run_users: int = Field(default=0, ge=0)
    repeat_users: int = Field(default=0, ge=0)
```

Then, at module scope directly above `class ValueAnalyticsStore` (line 211), the templates. `{ph}` is the placeholder, `{internal}` the optional exclusion clause, `{users}` the id clause:

```python
_INTERNAL_EXCLUSION_SQL = """
      AND u.role <> 'admin'
      AND u.id NOT IN (
          SELECT user_id FROM analytics_subject_settings WHERE excluded
      )
"""

_COUNT_STATES_SQL = """
    SELECT operational_state, operational_reason_code, lifecycle_segment,
           COUNT(*) AS users,
           SUM(CASE WHEN data_quality = 'partial' THEN 1 ELSE 0 END) AS partial_rows
    FROM user_daily_facts
    WHERE snapshot_date = {ph}
    GROUP BY operational_state, operational_reason_code, lifecycle_segment
    ORDER BY operational_state, operational_reason_code, lifecycle_segment
"""

_COUNT_SEGMENTS_SQL = """
    SELECT snapshot_date, lifecycle_segment,
           COUNT(*) AS users,
           SUM(CASE WHEN data_quality = 'partial' THEN 1 ELSE 0 END) AS partial_rows
    FROM user_daily_facts
    WHERE snapshot_date >= {ph} AND snapshot_date < {ph}
    GROUP BY snapshot_date, lifecycle_segment
    ORDER BY snapshot_date, lifecycle_segment
"""

_COUNT_TRANSITIONS_SQL = """
    SELECT from_segment, to_segment,
           COUNT(*) AS users,
           SUM(CASE WHEN data_quality = 'partial' THEN 1 ELSE 0 END) AS partial_rows
    FROM lifecycle_transitions
    WHERE snapshot_date >= {ph} AND snapshot_date < {ph}
    GROUP BY from_segment, to_segment
    ORDER BY from_segment, to_segment
"""

_ATTENTION_SQL = """
    SELECT f.user_id, f.operational_state, f.operational_reason_code,
           u.display_name, u.email, u.created_at,
           COALESCE(SUM(w.runs_requested), 0) AS recent_runs,
           COALESCE(SUM(w.runs_failed), 0) AS recent_failures
    FROM user_daily_facts AS f
    JOIN users AS u ON u.id = f.user_id
    LEFT JOIN user_daily_facts AS w
      ON w.user_id = f.user_id
     AND w.snapshot_date >= {ph} AND w.snapshot_date <= {ph}
    WHERE f.snapshot_date = {ph}
      AND f.operational_state IN ('blocked', 'needs_attention')
    GROUP BY f.user_id, f.operational_state, f.operational_reason_code,
             u.display_name, u.email, u.created_at
    ORDER BY recent_failures DESC, f.user_id DESC
    LIMIT {ph}
"""

_GROUP_TOTALS_SQL = """
    SELECT user_group,
           COALESCE(SUM(runs_requested), 0) AS total_runs,
           COALESCE(SUM(operator_cost_micro), 0) AS operator_cost_micro,
           COUNT(DISTINCT CASE WHEN runs_completed > 0 THEN user_id END)
               AS successful_run_users
    FROM user_daily_facts
    WHERE snapshot_date >= {ph} AND snapshot_date < {ph}
    GROUP BY user_group
"""

_GROUP_REPEAT_SQL = """
    SELECT user_group, COUNT(*) AS repeat_users
    FROM (
        SELECT user_group, user_id
        FROM user_daily_facts
        WHERE snapshot_date >= {ph} AND snapshot_date < {ph}
          AND runs_completed > 0
        GROUP BY user_group, user_id
        HAVING COUNT(*) >= 2
    ) AS repeaters
    GROUP BY user_group
"""

_ACTIVE_DATES_SQL = """
    SELECT user_id, snapshot_date
    FROM user_daily_facts
    WHERE snapshot_date >= {ph} AND snapshot_date < {ph}
      AND active
      AND {users}
    ORDER BY user_id, snapshot_date
"""

_ACTIVATIONS_SQL = """
    SELECT a.user_id, a.activated_at
    FROM user_activity AS a
    JOIN users AS u ON u.id = a.user_id
    WHERE a.activated_at IS NOT NULL
      AND a.activated_at >= {ph} AND a.activated_at < {ph}
      {internal}
    ORDER BY a.user_id
"""

_COUNT_ACTIVATED_SQL = """
    SELECT COUNT(*) AS activated
    FROM user_activity AS a
    JOIN users AS u ON u.id = a.user_id
    WHERE a.activated_at IS NOT NULL
      {internal}
"""

_USER_TRANSITIONS_SQL = """
    SELECT * FROM lifecycle_transitions
    WHERE user_id = {ph}
      AND snapshot_date >= {ph} AND snapshot_date < {ph}
    ORDER BY snapshot_date, transition_id
"""

_TRANSITIONS_FOR_DATE_SQL = """
    SELECT * FROM lifecycle_transitions
    WHERE snapshot_date = {ph}
    ORDER BY user_id
"""


def _transition_row(row: Any) -> LifecycleTransitionRow:
    return LifecycleTransitionRow(
        user_id=int(_row_value(row, "user_id")),
        snapshot_date=date.fromisoformat(str(_row_value(row, "snapshot_date"))),
        from_segment=_row_value(row, "from_segment"),
        to_segment=_row_value(row, "to_segment"),
        inactive_days=int(_row_value(row, "inactive_days", 0)),
        data_quality=_row_value(row, "data_quality"),
        created_at=_timestamp(_row_value(row, "created_at")),
    )


def _validate_date_window(start: date, end: date) -> tuple[str, str]:
    if not isinstance(start, date) or isinstance(start, datetime):
        raise ValueError("start must be a date")
    if not isinstance(end, date) or isinstance(end, datetime):
        raise ValueError("end must be a date")
    if end <= start:
        raise ValueError("end must be later than start")
    return start.isoformat(), end.isoformat()
```

`LifecycleTransitionRow` is PR A's model in this same module; if PR A placed it below this point, move these helpers below it (module order matters only for the class reference at call time, which is fine either way, but keep the file readable).

- [ ] **Step 4: Add the SQLite methods**

On `ValueAnalyticsStore`, after `sum_recent_facts` (PR A):

```python
    # -- Fact-table reads (design §6.10). One statement each, never per user. --

    def count_states_for_date(self, day: date) -> list[FactStateCount]:
        if not isinstance(day, date) or isinstance(day, datetime):
            raise ValueError("day must be a date")
        with self._analytics_connection() as conn:
            rows = conn.execute(
                _COUNT_STATES_SQL.format(ph="?"), (day.isoformat(),)
            ).fetchall()
        return [
            FactStateCount(
                operational_state=_row_value(row, "operational_state"),
                operational_reason_code=_row_value(row, "operational_reason_code"),
                lifecycle_segment=_row_value(row, "lifecycle_segment"),
                users=int(_row_value(row, "users", 0)),
                partial_rows=int(_row_value(row, "partial_rows", 0) or 0),
            )
            for row in rows
        ]

    def count_segments_by_date(
        self, *, start: date, end: date
    ) -> list[SegmentDayCount]:
        params = _validate_date_window(start, end)
        with self._analytics_connection() as conn:
            rows = conn.execute(_COUNT_SEGMENTS_SQL.format(ph="?"), params).fetchall()
        return [
            SegmentDayCount(
                snapshot_date=date.fromisoformat(str(_row_value(row, "snapshot_date"))),
                lifecycle_segment=_row_value(row, "lifecycle_segment"),
                users=int(_row_value(row, "users", 0)),
                partial_rows=int(_row_value(row, "partial_rows", 0) or 0),
            )
            for row in rows
        ]

    def count_transitions(self, *, start: date, end: date) -> list[TransitionCount]:
        params = _validate_date_window(start, end)
        with self._analytics_connection() as conn:
            rows = conn.execute(
                _COUNT_TRANSITIONS_SQL.format(ph="?"), params
            ).fetchall()
        return [
            TransitionCount(
                from_segment=_row_value(row, "from_segment"),
                to_segment=_row_value(row, "to_segment"),
                users=int(_row_value(row, "users", 0)),
                partial_rows=int(_row_value(row, "partial_rows", 0) or 0),
            )
            for row in rows
        ]

    def list_attention_candidates(
        self, *, day: date, window_start: date, limit: int
    ) -> list[AttentionCandidate]:
        if window_start > day:
            raise ValueError("window_start must not be after day")
        page_size = positive_limit(limit)
        params = (window_start.isoformat(), day.isoformat(), day.isoformat(), page_size)
        with self._analytics_connection() as conn:
            rows = conn.execute(_ATTENTION_SQL.format(ph="?"), params).fetchall()
        return [_attention_candidate(row) for row in rows]

    def sum_facts_by_user_group(
        self, *, start: date, end: date
    ) -> dict[str, GroupFactTotals]:
        params = _validate_date_window(start, end)
        with self._analytics_connection() as conn:
            totals = conn.execute(_GROUP_TOTALS_SQL.format(ph="?"), params).fetchall()
            repeats = conn.execute(_GROUP_REPEAT_SQL.format(ph="?"), params).fetchall()
        return _group_totals(totals, repeats)

    def list_active_dates(
        self, user_ids: Sequence[int], *, start: date, end: date
    ) -> dict[int, list[date]]:
        ids = _ids(user_ids)
        window = _validate_date_window(start, end)
        result: dict[int, list[date]] = {user_id: [] for user_id in ids}
        if not ids:
            return result
        clause, id_params = _user_clause(ids, False)
        with self._analytics_connection() as conn:
            rows = conn.execute(
                _ACTIVE_DATES_SQL.format(ph="?", users=clause),
                [*window, *id_params],
            ).fetchall()
        for row in rows:
            result[int(_row_value(row, "user_id"))].append(
                date.fromisoformat(str(_row_value(row, "snapshot_date")))
            )
        return result

    def list_activations(
        self, *, start: datetime, end: datetime, include_internal: bool
    ) -> dict[int, datetime]:
        window_start, window_end = _validate_window(start, end)
        sql = _ACTIVATIONS_SQL.format(
            ph="?", internal="" if include_internal else _INTERNAL_EXCLUSION_SQL
        )
        with self._analytics_connection() as conn:
            rows = conn.execute(
                sql, (utc_iso(window_start), utc_iso(window_end))
            ).fetchall()
        return {
            int(_row_value(row, "user_id")): _timestamp(_row_value(row, "activated_at"))
            for row in rows
        }

    def count_activated_users(self, *, include_internal: bool) -> int:
        sql = _COUNT_ACTIVATED_SQL.format(
            internal="" if include_internal else _INTERNAL_EXCLUSION_SQL
        )
        with self._analytics_connection() as conn:
            row = conn.execute(sql).fetchone()
        return int(_row_value(row, "activated", 0) or 0)

    def list_user_transitions(
        self, user_id: int, *, start: date, end: date
    ) -> list[LifecycleTransitionRow]:
        subject = positive_user_id(user_id)
        window = _validate_date_window(start, end)
        with self._analytics_connection() as conn:
            rows = conn.execute(
                _USER_TRANSITIONS_SQL.format(ph="?"), (subject, *window)
            ).fetchall()
        return [_transition_row(row) for row in rows]

    def list_transitions_for_date(self, day: date) -> list[LifecycleTransitionRow]:
        if not isinstance(day, date) or isinstance(day, datetime):
            raise ValueError("day must be a date")
        with self._analytics_connection() as conn:
            rows = conn.execute(
                _TRANSITIONS_FOR_DATE_SQL.format(ph="?"), (day.isoformat(),)
            ).fetchall()
        return [_transition_row(row) for row in rows]
```

and two module-level row helpers beside `_transition_row`:

```python
def _attention_candidate(row: Any) -> AttentionCandidate:
    return AttentionCandidate(
        user_id=int(_row_value(row, "user_id")),
        display_name=str(_row_value(row, "display_name") or ""),
        email=str(_row_value(row, "email") or ""),
        created_at=_timestamp(_row_value(row, "created_at")),
        operational_state=_row_value(row, "operational_state"),
        operational_reason_code=_row_value(row, "operational_reason_code"),
        recent_runs=int(_row_value(row, "recent_runs", 0) or 0),
        recent_failures=int(_row_value(row, "recent_failures", 0) or 0),
    )


def _group_totals(totals: Sequence[Any], repeats: Sequence[Any]) -> dict[str, GroupFactTotals]:
    repeat_by_group = {
        str(_row_value(row, "user_group")): int(_row_value(row, "repeat_users", 0) or 0)
        for row in repeats
    }
    return {
        str(_row_value(row, "user_group")): GroupFactTotals(
            total_runs=int(_row_value(row, "total_runs", 0) or 0),
            operator_cost_micro=max(int(_row_value(row, "operator_cost_micro", 0) or 0), 0),
            successful_run_users=int(_row_value(row, "successful_run_users", 0) or 0),
            repeat_users=repeat_by_group.get(str(_row_value(row, "user_group")), 0),
        )
        for row in totals
    }
```

`users.created_at` on SQLite is `TIMESTAMP DEFAULT CURRENT_TIMESTAMP` (`'YYYY-MM-DD HH:MM:SS'`, no zone); `_timestamp` already treats a naive value as UTC, which is what `query_service._parse_timestamp` does for the same column today. Extend `__all__` (line 1118) with `AttentionCandidate`, `FactStateCount`, `GroupFactTotals`, `SegmentDayCount`, `TransitionCount`.

- [ ] **Step 5: Add the Postgres twin methods**

In `dashboard/backend/domain/analytics/value_repository_postgres.py`, extend the import from `.value_repository` with `_ACTIVATIONS_SQL, _ACTIVE_DATES_SQL, _ATTENTION_SQL, _COUNT_ACTIVATED_SQL, _COUNT_SEGMENTS_SQL, _COUNT_STATES_SQL, _COUNT_TRANSITIONS_SQL, _GROUP_REPEAT_SQL, _GROUP_TOTALS_SQL, _INTERNAL_EXCLUSION_SQL, _TRANSITIONS_FOR_DATE_SQL, _USER_TRANSITIONS_SQL, _attention_candidate, _group_totals, _transition_row, _validate_date_window, AttentionCandidate, FactStateCount, GroupFactTotals, LifecycleTransitionRow, SegmentDayCount, TransitionCount` (PR T already imports `_ids`, `_user_clause`, `_fetchall`, `_validate_window`, `_row_value`, `_timestamp`, `positive_limit`, `positive_user_id`, `utc_iso` there — add whichever of those are missing), and add to `PostgresValueAnalyticsStore` the same ten methods, each identical to the SQLite one except: `ph="%s"`, `_user_clause(ids, True)`, and execution through `_fetchall(conn, True, sql, params)` (or `cur.execute`/`cur.fetchone` for `count_activated_users`). Written out, the first and the id-batched one — the other eight follow the same two substitutions exactly:

```python
    def count_states_for_date(self, day: date) -> list[FactStateCount]:
        if not isinstance(day, date) or isinstance(day, datetime):
            raise ValueError("day must be a date")
        with self._analytics_connection() as conn:
            rows = _fetchall(conn, True, _COUNT_STATES_SQL.format(ph="%s"), (day.isoformat(),))
        return [
            FactStateCount(
                operational_state=_row_value(row, "operational_state"),
                operational_reason_code=_row_value(row, "operational_reason_code"),
                lifecycle_segment=_row_value(row, "lifecycle_segment"),
                users=int(_row_value(row, "users", 0)),
                partial_rows=int(_row_value(row, "partial_rows", 0) or 0),
            )
            for row in rows
        ]

    def list_active_dates(
        self, user_ids: Sequence[int], *, start: date, end: date
    ) -> dict[int, list[date]]:
        ids = _ids(user_ids)
        window = _validate_date_window(start, end)
        result: dict[int, list[date]] = {user_id: [] for user_id in ids}
        if not ids:
            return result
        clause, id_params = _user_clause(ids, True)
        with self._analytics_connection() as conn:
            rows = _fetchall(
                conn, True,
                _ACTIVE_DATES_SQL.format(ph="%s", users=clause),
                [*window, *id_params],
            )
        for row in rows:
            result[int(_row_value(row, "user_id"))].append(
                date.fromisoformat(str(_row_value(row, "snapshot_date")))
            )
        return result
```

`count_activated_users` on Postgres:

```python
    def count_activated_users(self, *, include_internal: bool) -> int:
        sql = _COUNT_ACTIVATED_SQL.format(
            internal="" if include_internal else _INTERNAL_EXCLUSION_SQL
        )
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                row = cur.fetchone()
        return int(_row_value(row, "activated", 0) or 0)
```

Postgres `GROUP BY` requires every non-aggregated select column, which `_ATTENTION_SQL` already lists; `ORDER BY recent_failures` uses the output alias, valid on both. `users.created_at` is `TIMESTAMP` on Postgres and psycopg returns a naive `datetime`; `_timestamp(str(value))` parses `'2026-09-14 12:00:00'` and assumes UTC, matching what `query_service._parse_timestamp` does for the same column today.

- [ ] **Step 6: Run and confirm the pass**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_fact_reads.py dashboard/backend/tests/test_store_twin_parity.py -v
TEST_POSTGRES_URL=postgresql://postgres:test@localhost:5432/atl_test python -m pytest dashboard/backend/tests/domain/analytics/test_repository_postgres.py -v -k fact_reads
```

Expected: **PASS**. The parity test's method-set and signature checks cover the new methods because the pair is registered in `_TWINS` (PR T Task 3). The Postgres run is skipped without `TEST_POSTGRES_URL`; CI runs it.

- [ ] **Step 7: Commit**

```bash
git add dashboard/backend/domain/analytics/value_repository.py \
        dashboard/backend/domain/analytics/value_repository_postgres.py \
        dashboard/backend/tests/domain/analytics/test_fact_reads.py \
        dashboard/backend/tests/domain/analytics/test_repository_postgres.py
git status --short
git commit -m "$(cat <<'EOF'
feat: add set-based fact-table reads on both analytics store twins

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: The users page in SQL, on both twins

**Files:**
- Modify: `dashboard/backend/domain/analytics/value_repository.py` — `UserPageQuery`, `UserPageRow`, module-level `_users_page_sql`, `ValueAnalyticsStore.list_users_page`
- Modify: `dashboard/backend/domain/analytics/value_repository_postgres.py` — `PostgresValueAnalyticsStore.list_users_page`
- Test: `dashboard/backend/tests/domain/analytics/test_fact_reads.py` (extend)
- Test: `dashboard/backend/tests/domain/analytics/test_repository_postgres.py` (extend the `@pg_only` case from Task 2)

**Interfaces:**
- Consumes: `users`, `user_activity`, `user_daily_facts`, `analytics_subject_settings`; `admin_user_search_pattern` from `dashboard.backend.users` (line 675 — the same escaped `LIKE` pattern `list_users_admin` uses on both user-store twins).
- Produces, on both twins:
  - `UserPageQuery` (frozen dataclass; the filter fields of `UserValueFilters`, so `value_queries.py` can convert without importing in the other direction and creating a cycle): `q, lifecycle_segment, operational_state, commercial_tier, user_group, activated, last_meaningful_activity_from, last_meaningful_activity_to, priority, legacy_status, include_internal`.
  - `UserPageRow`: `user_id, email, display_name, role, user_group, created_at, activated_at, last_meaningful_activity_at, lifecycle_segment (yesterday's, or None), operational_state, operational_reason_code, tier, data_quality (or None)`.
  - `list_users_page(*, query: UserPageQuery, day: date, limit: int, offset: int) -> tuple[list[UserPageRow], int]` — the page and the total under the same predicate, in two statements. Task 10's `ValueAnalyticsQueryService.list_users` consumes it.

Design §6.10: *filtering, sorting and pagination happen in SQL, not in Python over the whole population*, with tier and operational state taken from yesterday's facts row. The predicates are exactly the ones `list_users` applies in Python today (`value_queries.py:1277-1317`); the sort is today's `_PRIORITY_RANK`/`_TIER_RANK`/inactivity/`user_id` order (`:1332-1343`) with one documented change: the `-lifetime_net_purchased_micro` tiebreak between tier and inactivity is dropped, because the ledger is another domain's table and cannot be joined here (§6.14) — tier is the ledger-derived rank already on the fact row. `legacy_status` is translated into fact predicates by the same mapping `_legacy_seed` (`value_repository.py:185-208`) applies at snapshot-write time today, so the `status` filter answers identically until Task 16 removes it.

Two behaviour notes the row shape makes visible: users with no `user_activity` row are simply not activated (LEFT JOIN, NULL), and users with no fact row for `day` (signed up today, or excluded) still appear with `tier='unpaid'`, `operational_state='healthy'` and no yesterday segment — a `lifecycle_segment` filter excludes them, since the filter is "as of yesterday" and yesterday they had no segment. Today's Python loop drops any user without a snapshot outright (`:1281-1282`); after this task a brand-new signup is listed, which is the more honest answer and the one PR C's users list wants.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/backend/tests/domain/analytics/test_fact_reads.py` (extend the `value_repository` import with `UserPageQuery`):

```python
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
        legacy_status=None,
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

    rows, total = store.list_users_page(
        query=_page_query(include_internal=True), day=YESTERDAY, limit=100, offset=0
    )
    assert total == 6
    assert ids["admin"] in {row.user_id for row in rows}

    rows, total = store.list_users_page(
        query=_page_query(), day=YESTERDAY, limit=2, offset=2
    )
    assert total == 5
    assert len(rows) == 2

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

    for legacy, expected in (
        ("blocked", {ids["blocked"]}),
        ("needs_attention", {ids["attention"]}),
        ("dormant", {ids["dormant"]}),
        ("active", {ids["core"]}),
        ("onboarding", set()),
    ):
        rows, _total = store.list_users_page(
            query=_page_query(legacy_status=legacy), day=YESTERDAY, limit=100, offset=0
        )
        assert {row.user_id for row in rows} == expected, legacy


def test_sqlite_users_page_follows_the_contract(tmp_path):
    store, users, admin_id = _sqlite_fixture(tmp_path)
    ids = seed_population(store, users, admin_id=admin_id)
    assert_users_page_contract(store, ids)
```

The `"onboarding"` legacy case is empty on purpose: the mapping puts `new`/`onboarding` segments there and the fresh user has no fact row yet, so no row satisfies a yesterday-segment predicate. Add `assert_users_page_contract(value_store, ids)` as the last line of Task 2's `test_postgres_fact_reads_follow_the_contract` (import it beside `assert_fact_read_contract`).

- [ ] **Step 2: Run them and confirm the expected failure**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_fact_reads.py -v -k users_page
```

Expected: **FAIL** at import — `ImportError: cannot import name 'UserPageQuery'`.

- [ ] **Step 3: Add the query/row models and the SQL builder**

In `value_repository.py`, beside the Task 2 models (add `from dataclasses import dataclass` to the module imports):

```python
@dataclass(frozen=True)
class UserPageQuery:
    """The users-list filters, dialect-free, so the SQL builder can live here
    without importing value_queries (which imports this module)."""

    q: str | None = None
    lifecycle_segment: str | None = None
    operational_state: str | None = None
    commercial_tier: str | None = None
    user_group: str | None = None
    activated: bool | None = None
    last_meaningful_activity_from: datetime | None = None
    last_meaningful_activity_to: datetime | None = None
    priority: bool = False
    legacy_status: str | None = None
    include_internal: bool = False


class UserPageRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: int = Field(gt=0)
    email: str
    display_name: str
    role: str
    user_group: str
    created_at: datetime
    activated_at: datetime | None = None
    last_meaningful_activity_at: datetime | None = None
    lifecycle_segment: LifecycleSegment | None = None
    operational_state: OperationalState = "healthy"
    operational_reason_code: str = "no_supported_issue"
    tier: CommercialTier = "unpaid"
    data_quality: Literal["complete", "partial"] | None = None
```

and, at module scope beside the Task 2 templates:

```python
_LEGACY_STATUS_PREDICATES = {
    # The write-time mapping _legacy_seed applied, as fact-row predicates.
    # Removed with the status filter in the D20 task (Task 16).
    "blocked": "COALESCE(f.operational_state, 'healthy') = 'blocked'",
    "needs_attention": "COALESCE(f.operational_state, 'healthy') = 'needs_attention'",
    "dormant": (
        "COALESCE(f.operational_state, 'healthy') = 'healthy'"
        " AND f.lifecycle_segment = 'dormant'"
    ),
    "onboarding": (
        "COALESCE(f.operational_state, 'healthy') = 'healthy'"
        " AND f.lifecycle_segment IN ('new', 'onboarding')"
    ),
    "active": (
        "COALESCE(f.operational_state, 'healthy') = 'healthy'"
        " AND f.lifecycle_segment IN ('growing', 'core', 'at_risk')"
    ),
}

_USERS_PAGE_SELECT = """
    SELECT u.id AS user_id, u.email, u.display_name, u.role, u.user_group,
           u.created_at, a.activated_at, a.last_meaningful_activity_at,
           f.lifecycle_segment, f.operational_state, f.operational_reason_code,
           f.tier, f.data_quality
"""
_USERS_PAGE_FROM = """
    FROM users AS u
    LEFT JOIN user_activity AS a ON a.user_id = u.id
    LEFT JOIN user_daily_facts AS f
      ON f.user_id = u.id AND f.snapshot_date = {ph}
"""
_USERS_PAGE_PRIORITY_ORDER = """
    ORDER BY CASE COALESCE(f.operational_state, 'healthy')
                 WHEN 'blocked' THEN 0
                 WHEN 'needs_attention' THEN 1
                 ELSE CASE f.lifecycle_segment
                          WHEN 'at_risk' THEN 2
                          WHEN 'onboarding' THEN 3
                          ELSE 4
                      END
             END ASC,
             CASE COALESCE(f.tier, 'unpaid')
                 WHEN 'high_value' THEN 0
                 WHEN 'invested' THEN 1
                 WHEN 'starter' THEN 2
                 ELSE 3
             END ASC,
             CASE WHEN a.last_meaningful_activity_at IS NULL THEN 0 ELSE 1 END ASC,
             a.last_meaningful_activity_at ASC,
             u.id ASC
"""


def _users_page_sql(
    query: UserPageQuery, *, day: date, limit: int, offset: int, ph: str
) -> tuple[str, str, list[Any], list[Any]]:
    """Render (page_sql, count_sql, page_params, count_params) for one dialect.

    Every predicate is parameterised; the only interpolated text is the fixed
    predicate for a legacy status, chosen from a closed dict. The LIKE pattern
    comes from users.admin_user_search_pattern so '%' and '_' in a search are
    literal, exactly as the account console's own search treats them.
    """
    from dashboard.backend.users import admin_user_search_pattern

    where: list[str] = []
    params: list[Any] = [day.isoformat()]
    if not query.include_internal:
        where.append(
            "u.role <> 'admin' AND u.id NOT IN "
            "(SELECT user_id FROM analytics_subject_settings WHERE excluded)"
        )
    pattern = admin_user_search_pattern(query.q)
    if pattern is not None:
        where.append(
            f"(lower(u.email) LIKE lower({ph}) ESCAPE '\\'"
            f" OR lower(u.display_name) LIKE lower({ph}) ESCAPE '\\')"
        )
        params.extend([pattern, pattern])
    if query.user_group is not None:
        where.append(f"u.user_group = {ph}")
        params.append(query.user_group)
    if query.commercial_tier is not None:
        where.append(f"COALESCE(f.tier, 'unpaid') = {ph}")
        params.append(query.commercial_tier)
    if query.operational_state is not None:
        where.append(f"COALESCE(f.operational_state, 'healthy') = {ph}")
        params.append(query.operational_state)
    if query.lifecycle_segment is not None:
        where.append(f"f.lifecycle_segment = {ph}")
        params.append(query.lifecycle_segment)
    if query.activated is True:
        where.append("a.activated_at IS NOT NULL")
    elif query.activated is False:
        where.append("a.activated_at IS NULL")
    if query.last_meaningful_activity_from is not None:
        where.append(f"a.last_meaningful_activity_at >= {ph}")
        params.append(utc_iso(_utc(query.last_meaningful_activity_from)))
    if query.last_meaningful_activity_to is not None:
        where.append(f"a.last_meaningful_activity_at <= {ph}")
        params.append(utc_iso(_utc(query.last_meaningful_activity_to)))
    if query.priority:
        where.append(
            "(COALESCE(f.operational_state, 'healthy') IN ('blocked', 'needs_attention')"
            " OR f.lifecycle_segment IN ('at_risk', 'onboarding'))"
        )
    if query.legacy_status is not None:
        if query.legacy_status not in _LEGACY_STATUS_PREDICATES:
            raise ValueError("legacy_status is unsupported")
        where.append(f"({_LEGACY_STATUS_PREDICATES[query.legacy_status]})")
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    from_sql = _USERS_PAGE_FROM.format(ph=ph)
    order_sql = _USERS_PAGE_PRIORITY_ORDER if query.priority else " ORDER BY u.id ASC"
    page_sql = (
        f"{_USERS_PAGE_SELECT}{from_sql}{where_sql}{order_sql} LIMIT {ph} OFFSET {ph}"
    )
    count_sql = f"SELECT COUNT(*) AS total{from_sql}{where_sql}"
    return page_sql, count_sql, [*params, limit, offset], list(params)


def _user_page_row(row: Any) -> UserPageRow:
    return UserPageRow(
        user_id=int(_row_value(row, "user_id")),
        email=str(_row_value(row, "email") or ""),
        display_name=str(_row_value(row, "display_name") or ""),
        role=str(_row_value(row, "role") or "user"),
        user_group=str(_row_value(row, "user_group") or "unknown"),
        created_at=_timestamp(_row_value(row, "created_at")),
        activated_at=_optional_timestamp(_row_value(row, "activated_at")),
        last_meaningful_activity_at=_optional_timestamp(
            _row_value(row, "last_meaningful_activity_at")
        ),
        lifecycle_segment=_row_value(row, "lifecycle_segment"),
        operational_state=_row_value(row, "operational_state") or "healthy",
        operational_reason_code=(
            _row_value(row, "operational_reason_code") or "no_supported_issue"
        ),
        tier=_row_value(row, "tier") or "unpaid",
        data_quality=_row_value(row, "data_quality"),
    )
```

`u.created_at` on SQLite is `'YYYY-MM-DD HH:MM:SS'`; `_timestamp` parses it as UTC. On Postgres psycopg hands back a naive `datetime`, and `_timestamp(str(value))` parses `'2026-09-14 12:00:00'` the same way.

- [ ] **Step 4: Add the methods to both twins**

`ValueAnalyticsStore`:

```python
    def list_users_page(
        self,
        *,
        query: UserPageQuery,
        day: date,
        limit: int,
        offset: int,
    ) -> tuple[list[UserPageRow], int]:
        """One filtered, sorted, paginated page of users plus the total.

        Two statements, both in SQL, neither parameterised by a user id: the
        Python loop this replaces loaded every eligible user and every
        snapshot on every page load (value_queries.py:1267-1349 before PR B).
        """
        page_size = positive_limit(limit)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if not isinstance(day, date) or isinstance(day, datetime):
            raise ValueError("day must be a date")
        page_sql, count_sql, page_params, count_params = _users_page_sql(
            query, day=day, limit=page_size, offset=offset, ph="?"
        )
        with self._analytics_connection() as conn:
            rows = conn.execute(page_sql, page_params).fetchall()
            total_row = conn.execute(count_sql, count_params).fetchone()
        return [_user_page_row(row) for row in rows], int(
            _row_value(total_row, "total", 0) or 0
        )
```

`PostgresValueAnalyticsStore` — identical but `ph="%s"` and:

```python
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(page_sql, page_params)
                rows = cur.fetchall()
                cur.execute(count_sql, count_params)
                total_row = cur.fetchone()
```

Import `UserPageQuery`, `UserPageRow`, `_users_page_sql`, `_user_page_row` into the Postgres module. Extend `__all__` in `value_repository.py` with `UserPageQuery`, `UserPageRow`.

- [ ] **Step 5: Run and confirm the pass**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_fact_reads.py dashboard/backend/tests/test_store_twin_parity.py -v
TEST_POSTGRES_URL=postgresql://postgres:test@localhost:5432/atl_test python -m pytest dashboard/backend/tests/domain/analytics/test_repository_postgres.py -v -k fact_reads
```

Expected: **PASS** on the SQLite tier; the Postgres case is skipped locally without the URL and runs in CI. If the Postgres `LIKE ... ESCAPE '\'` case fails with a syntax error, the escape literal has been double-escaped by the f-string: it must reach the driver as the two characters `'\'`, exactly as `users_postgres.py:1079-1080` sends it.

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/domain/analytics/value_repository.py \
        dashboard/backend/domain/analytics/value_repository_postgres.py \
        dashboard/backend/tests/domain/analytics/test_fact_reads.py \
        dashboard/backend/tests/domain/analytics/test_repository_postgres.py
git status --short
git commit -m "$(cat <<'EOF'
feat: filter, sort and paginate the analytics users list in SQL

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `CreditsStore.sum_ledger_by_day` on both credits twins, and `purchased_by_day` / `consumed_by_day`

**Files:**
- Modify: `dashboard/backend/domain/credits/repository_common.py` — `LedgerDayTotal`, the two SQL templates and two helpers, after `_positive_limit` (line 93-97); both twins already import this module
- Modify: `dashboard/backend/domain/credits/repository.py` — add `sum_ledger_by_day` to `CreditsStore` directly after `get_balance_projections` (line 898-996)
- Modify: `dashboard/backend/domain/credits/repository_postgres.py` — the same method on `PostgresCreditsStore` after its `get_balance_projections` (line 753)
- Modify: `dashboard/backend/domain/analytics/value_queries.py` — `ValueAnalyticsQueryService.purchased_by_day` and `consumed_by_day` after `get_commercial` (line 1031)
- Test: `dashboard/backend/tests/domain/credits/test_ledger_by_day.py` (create; `dashboard/backend/tests/domain/credits/` already exists — confirm with `ls`, and if it does not, create it with an empty `__init__.py` like `tests/domain/analytics/__init__.py`)
- Test: `dashboard/backend/tests/domain/analytics/test_value_queries.py` (extend)

**Interfaces:**
- Consumes: `credit_ledger_entries` (`entry_type`, `amount_micro`, `created_at`) and `credit_llm_usage_entries` (`amount_micro`, `created_at`) — the credits domain's own tables, read by the credits store (§6.14).
- Produces:
  - `CreditsStore.sum_ledger_by_day(*, start: datetime, end: datetime) -> list[LedgerDayTotal]` on both twins, where `LedgerDayTotal` is a frozen Pydantic model `(day: date, purchased_micro: int, refunded_micro: int, consumed_micro: int)` defined in `dashboard/backend/domain/credits/repository_common.py` (the module both twins already import their shared helpers from — `repository_postgres.py` does not import `repository.py`). Two grouped statements over `[start, end)`; no user id anywhere.
  - `ValueAnalyticsQueryService.purchased_by_day(*, start: date, end: date) -> dict[str, int]` and `consumed_by_day(*, start: date, end: date) -> dict[str, int]` — ISO date → micro-credits, over the same window. **Not** added to `CommercialAnalyticsResponse`; PR D puts them on the payload (§9).

The ledger's timestamps are ISO-8601 UTC text (`created_at TEXT NOT NULL`), so the day bucket is `substr(created_at, 1, 10)` on both dialects — no `date_trunc`, no SQLite `date()` function, one expression that means the same thing on both. Refunds are stored with negative `amount_micro`; the sum negates them so the response carries a positive refunded total, matching `list_commercial_values` (`value_repository.py:757-758`). Consumption rows are negative too, for the same reason (`:792`).

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/domain/credits/test_ledger_by_day.py`:

```python
"""CreditsStore.sum_ledger_by_day: anonymous per-day ledger totals."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from dashboard.backend.domain.credits.repository import CreditsStore
from dashboard.backend.domain.credits.repository_common import LedgerDayTotal
from dashboard.backend.users import UserStore


UTC = timezone.utc
NOW = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)


def _store(tmp_path):
    db_path = tmp_path / "ledger.db"
    users = UserStore(db_path=db_path)
    first = int(users.create_user("one@example.test", "One", "SecurePass1!")["id"])
    second = int(users.create_user("two@example.test", "Two", "SecurePass1!")["id"])
    store = CreditsStore(db_path)
    return store, first, second


def _insert_ledger(store, *, user_id, entry_type, amount_micro, created_at):
    """Insert a minimal ledger row satisfying the shape CHECK for its type.

    The CHECK in ``domain/credits/repository.py::_CREDIT_LEDGER_DDL`` is
    per-type: ``purchase``/``refund`` rows must carry the Stripe trio and
    NULL grant columns, while ``admin_grant_*`` rows must carry
    ``request_digest``, ``actor_user_id``, ``reference_type = 'grant_pool'``
    and ``reference_id`` and NULL Stripe columns. ``PRAGMA foreign_keys = OFF``
    only silences the FK lookups, not the CHECK, so every column is a
    parameter here — a literal NULL for the grant trio fails the
    ``admin_grant_assign`` insert with ``IntegrityError``.
    """

    is_grant = entry_type.startswith("admin_grant")
    bucket = "grant" if is_grant else "purchased"
    stamp = created_at.timestamp()
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            """
            INSERT INTO credit_ledger_entries (
                user_id, bucket, entry_type, amount_micro, payment_order_id,
                refund_request_id, stripe_event_id, operation_key, operation_id,
                idempotency_key, request_digest, actor_user_id, source, reason,
                reference_type, reference_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'test', 'test', ?, ?, ?)
            """,
            (
                user_id,
                bucket,
                entry_type,
                amount_micro,
                f"order-{stamp}" if entry_type in {"purchase", "refund"} else None,
                f"refund-{stamp}" if entry_type == "refund" else None,
                f"evt-{stamp}" if not is_grant else None,
                f"op-{stamp}-{entry_type}",
                f"opid-{stamp}-{entry_type}",
                f"idem-{stamp}-{entry_type}",
                f"digest-{stamp}" if is_grant else None,
                1 if is_grant else None,
                "grant_pool" if is_grant else None,
                f"pool-{stamp}" if is_grant else None,
                created_at.isoformat(),
            ),
        )


def _insert_usage(store, *, user_id, amount_micro, created_at):
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        columns = [
            row[1]
            for row in conn.execute("PRAGMA table_info(credit_llm_usage_entries)").fetchall()
        ]
        values = {
            "user_id": user_id,
            "bucket": "grant",
            "amount_micro": amount_micro,
            "operation_key": f"usage-{created_at.timestamp()}",
            "created_at": created_at.isoformat(),
        }
        for column in columns:
            if column in values or column == "id":
                continue
            values[column] = (
                f"{column}-{created_at.timestamp()}"
                if column in {"reservation_id", "run_id", "operation_id", "idempotency_key"}
                else 0 if column in {"call_index", "input_tokens", "output_tokens"}
                else "{}" if column.endswith("_json")
                else "test"
            )
        names = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)
        conn.execute(
            f"INSERT INTO credit_llm_usage_entries ({names}) VALUES ({placeholders})",
            list(values.values()),
        )


def test_sum_ledger_by_day_groups_by_utc_day_and_signs_totals(tmp_path):
    store, first, second = _store(tmp_path)
    day_one = datetime(2026, 9, 10, 8, 0, tzinfo=UTC)
    day_two = datetime(2026, 9, 11, 23, 59, tzinfo=UTC)
    _insert_ledger(store, user_id=first, entry_type="purchase", amount_micro=5_000_000, created_at=day_one)
    _insert_ledger(store, user_id=second, entry_type="purchase", amount_micro=1_000_000, created_at=day_one + timedelta(hours=1))
    _insert_ledger(store, user_id=first, entry_type="refund", amount_micro=-2_000_000, created_at=day_two)
    _insert_ledger(store, user_id=first, entry_type="admin_grant_assign", amount_micro=3_000_000, created_at=day_two)
    _insert_usage(store, user_id=first, amount_micro=-250_000, created_at=day_one)
    _insert_usage(store, user_id=second, amount_micro=-750_000, created_at=day_two)
    _insert_ledger(store, user_id=first, entry_type="purchase", amount_micro=9_000_000, created_at=datetime(2026, 9, 12, 0, 0, tzinfo=UTC))

    rows = store.sum_ledger_by_day(
        start=datetime(2026, 9, 10, tzinfo=UTC),
        end=datetime(2026, 9, 12, tzinfo=UTC),
    )

    assert rows == [
        LedgerDayTotal(day=day_one.date(), purchased_micro=6_000_000, refunded_micro=0, consumed_micro=250_000),
        LedgerDayTotal(day=day_two.date(), purchased_micro=0, refunded_micro=2_000_000, consumed_micro=750_000),
    ]


def test_sum_ledger_by_day_rejects_a_reversed_window(tmp_path):
    store, _first, _second = _store(tmp_path)
    import pytest

    with pytest.raises(ValueError):
        store.sum_ledger_by_day(start=NOW, end=NOW - timedelta(days=1))
```

If the `_insert_usage` helper trips a `CHECK` on a column this test does not know about, read `_CREDIT_LLM_USAGE_DDL` (`repository.py:181-260`) and give that column the value the constraint wants — the helper is deliberately schema-driven so a future column does not break it, but a new `CHECK` can. Grant rows (`admin_grant_assign`) are inserted to prove they are *excluded* from purchased and refunded, as §15.4 requires.

Then extend `dashboard/backend/tests/domain/analytics/test_value_queries.py` with a fake credits reader and two tests (the file's `_service` helper builds a `FakeValueStore`; add `ledger_days=` to it per the snippet):

```python
class FakeCreditsReader:
    def __init__(self, rows):
        self.rows = list(rows)
        self.windows = []

    def sum_ledger_by_day(self, *, start, end):
        self.windows.append((start, end))
        return [row for row in self.rows if start <= _day_start(row.day) < end]


def test_purchased_and_consumed_by_day_read_the_ledger_once_each():
    from dashboard.backend.domain.credits.repository_common import LedgerDayTotal

    reader = FakeCreditsReader(
        [
            LedgerDayTotal(day=date(2026, 9, 1), purchased_micro=5_000_000, refunded_micro=0, consumed_micro=100_000),
            LedgerDayTotal(day=date(2026, 9, 2), purchased_micro=0, refunded_micro=1_000_000, consumed_micro=0),
            LedgerDayTotal(day=date(2026, 9, 9), purchased_micro=7_000_000, refunded_micro=0, consumed_micro=50_000),
        ]
    )
    service = _service(snapshots={1: _snapshot(1)}, credits_reader=reader)

    purchased = service.purchased_by_day(start=date(2026, 9, 1), end=date(2026, 9, 4))
    consumed = service.consumed_by_day(start=date(2026, 9, 1), end=date(2026, 9, 4))

    assert purchased == {"2026-09-01": 5_000_000, "2026-09-02": -1_000_000, "2026-09-03": 0}
    assert consumed == {"2026-09-01": 100_000, "2026-09-02": 0, "2026-09-03": 0}
    assert len(reader.windows) == 2
    assert all(end - start == timedelta(days=3) for start, end in reader.windows)


def test_by_day_series_are_not_on_the_commercial_response():
    """PR D puts them on the payload; PR B holds the shape (design §10.2)."""

    from dashboard.backend.domain.analytics.value_queries import CommercialAnalyticsResponse

    assert "purchased_by_day" not in CommercialAnalyticsResponse.model_fields
    assert "consumed_by_day" not in CommercialAnalyticsResponse.model_fields
```

`_day_start` is already imported in `test_value_queries.py`'s namespace via `value_queries`? It is not — add `from dashboard.backend.domain.analytics.value_queries import _day_start` to the test's imports. In `_service(...)`, add a `credits_reader=None` keyword and pass it through as `ValueAnalyticsQueryService(..., credits_reader=credits_reader)`.

- [ ] **Step 2: Run them and confirm the expected failure**

```bash
python -m pytest dashboard/backend/tests/domain/credits/test_ledger_by_day.py dashboard/backend/tests/domain/analytics/test_value_queries.py -v -k "ledger_by_day or by_day"
```

Expected: **FAIL** — `ImportError: cannot import name 'LedgerDayTotal'` for the credits file; `TypeError: __init__() got an unexpected keyword argument 'credits_reader'` for the value-queries file.

- [ ] **Step 3: Add the credits store method to both twins**

In `dashboard/backend/domain/credits/repository_common.py`, after `_positive_limit` (line 93-97), adding `from datetime import date` to its `datetime` import and `from pydantic import BaseModel, ConfigDict, Field` to its imports:

```python
class LedgerDayTotal(BaseModel):
    """Anonymous ledger movement for one UTC day. Purchases and refunds are
    positive magnitudes; grants are excluded because they are not revenue
    (design §15.4)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    day: date
    purchased_micro: int = Field(default=0, ge=0)
    refunded_micro: int = Field(default=0, ge=0)
    consumed_micro: int = Field(default=0, ge=0)


_LEDGER_BY_DAY_SQL = """
    SELECT substr(created_at, 1, 10) AS day,
           COALESCE(SUM(CASE WHEN entry_type = 'purchase' THEN amount_micro ELSE 0 END), 0)
               AS purchased_micro,
           COALESCE(SUM(CASE WHEN entry_type = 'refund' THEN -amount_micro ELSE 0 END), 0)
               AS refunded_micro
    FROM credit_ledger_entries
    WHERE created_at >= {ph} AND created_at < {ph}
      AND entry_type IN ('purchase', 'refund')
    GROUP BY substr(created_at, 1, 10)
"""
_USAGE_BY_DAY_SQL = """
    SELECT substr(created_at, 1, 10) AS day,
           COALESCE(SUM(-amount_micro), 0) AS consumed_micro
    FROM credit_llm_usage_entries
    WHERE created_at >= {ph} AND created_at < {ph}
    GROUP BY substr(created_at, 1, 10)
"""


def _ledger_window(start: datetime, end: datetime) -> tuple[str, str]:
    for name, value in (("start", start), ("end", end)):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{name} must include a timezone")
    if end <= start:
        raise ValueError("end must be later than start")
    return (
        start.astimezone(timezone.utc).isoformat(),
        end.astimezone(timezone.utc).isoformat(),
    )


def _merge_ledger_days(ledger_rows, usage_rows) -> list[LedgerDayTotal]:
    totals: dict[date, dict[str, int]] = {}
    for row in ledger_rows:
        bucket = totals.setdefault(date.fromisoformat(str(row["day"])), {})
        bucket["purchased_micro"] = max(int(row["purchased_micro"] or 0), 0)
        bucket["refunded_micro"] = max(int(row["refunded_micro"] or 0), 0)
    for row in usage_rows:
        bucket = totals.setdefault(date.fromisoformat(str(row["day"])), {})
        bucket["consumed_micro"] = max(int(row["consumed_micro"] or 0), 0)
    return [LedgerDayTotal(day=day, **values) for day, values in sorted(totals.items())]
```

In `dashboard/backend/domain/credits/repository.py`, extend the existing `from dashboard.backend.domain.credits.repository_common import (...)` block (line 16) with `LedgerDayTotal, _LEDGER_BY_DAY_SQL, _USAGE_BY_DAY_SQL, _ledger_window, _merge_ledger_days`. Then on `CreditsStore`, after `get_balance_projections`:

```python
    def sum_ledger_by_day(
        self, *, start: datetime, end: datetime
    ) -> list[LedgerDayTotal]:
        """Purchases, refunds and consumption per UTC day over [start, end).

        Two grouped statements, no user id. This is the credits domain's own
        read of its own tables (design §6.14); the analytics services call it
        rather than opening a credits connection themselves.
        """
        params = _ledger_window(start, end)
        with self._get_connection() as conn:
            ledger_rows = conn.execute(_LEDGER_BY_DAY_SQL.format(ph="?"), params).fetchall()
            usage_rows = conn.execute(_USAGE_BY_DAY_SQL.format(ph="?"), params).fetchall()
        return _merge_ledger_days(ledger_rows, usage_rows)
```

On `PostgresCreditsStore` (`repository_postgres.py`, after `get_balance_projections` at line 753), extending its existing `from dashboard.backend.domain.credits.repository_common import (...)` block (line 12) with the same five names:

```python
    def sum_ledger_by_day(
        self, *, start: datetime, end: datetime
    ) -> list[LedgerDayTotal]:
        params = _ledger_window(start, end)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(_LEDGER_BY_DAY_SQL.format(ph="%s"), params)
                ledger_rows = cur.fetchall()
                cur.execute(_USAGE_BY_DAY_SQL.format(ph="%s"), params)
                usage_rows = cur.fetchall()
        return _merge_ledger_days(ledger_rows, usage_rows)
```

Neither credits module declares `__all__`, so nothing else changes there.

- [ ] **Step 4: Add the two service methods**

In `dashboard/backend/domain/analytics/value_queries.py`, extend `ValueAnalyticsQueryService.__init__` (lines 479-499) with a `credits_reader: Any | None = None` keyword stored as `self.credits_reader`, defaulting to the credits store singleton:

```python
        if credits_reader is None:
            from dashboard.backend.domain.credits.repository import credits_store

            credits_reader = credits_store
        self.credits_reader = credits_reader
```

Then after `get_commercial` (line 1031):

```python
    def _ledger_days(
        self, *, start: date, end: date
    ) -> dict[str, "LedgerDayTotal"]:
        start, end = _validate_dates(start, end)
        rows = self.credits_reader.sum_ledger_by_day(
            start=_day_start(start), end=_day_start(end)
        )
        return {row.day.isoformat(): row for row in rows}

    def purchased_by_day(self, *, start: date, end: date) -> dict[str, int]:
        """Net purchases (purchases minus refunds) per UTC day, zero-filled.

        Service method only in PR B; PR D adds `purchased_by_day` to
        CommercialAnalyticsResponse (design §9). Net can be negative on a day
        of refunds, which is the honest revenue line.
        """
        by_day = self._ledger_days(start=start, end=end)
        return {
            (start + timedelta(days=offset)).isoformat(): (
                by_day[key].purchased_micro - by_day[key].refunded_micro
                if (key := (start + timedelta(days=offset)).isoformat()) in by_day
                else 0
            )
            for offset in range((end - start).days)
        }

    def consumed_by_day(self, *, start: date, end: date) -> dict[str, int]:
        """Credits consumed through model execution per UTC day, zero-filled."""
        by_day = self._ledger_days(start=start, end=end)
        return {
            (start + timedelta(days=offset)).isoformat(): (
                by_day[key].consumed_micro
                if (key := (start + timedelta(days=offset)).isoformat()) in by_day
                else 0
            )
            for offset in range((end - start).days)
        }
```

Add `from dashboard.backend.domain.credits.repository_common import LedgerDayTotal` to `value_queries.py`'s imports (that module holds no singleton, so importing it at module scope is free) and drop the quotes from the `dict[str, LedgerDayTotal]` annotation.

- [ ] **Step 5: Run and confirm the pass**

```bash
python -m pytest dashboard/backend/tests/domain/credits/test_ledger_by_day.py dashboard/backend/tests/domain/analytics/test_value_queries.py dashboard/backend/tests/test_store_twin_parity.py -v
```

Expected: **PASS**. The parity test covers `sum_ledger_by_day` because `CreditsStore`/`PostgresCreditsStore` are a registered pair. The Postgres behavioural check runs through the existing `@pg_only` credits tier in CI; no new `@pg_only` case is needed because both twins execute the same two template strings.

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/domain/credits/repository_common.py \
        dashboard/backend/domain/credits/repository.py \
        dashboard/backend/domain/credits/repository_postgres.py \
        dashboard/backend/domain/analytics/value_queries.py \
        dashboard/backend/tests/domain/credits/test_ledger_by_day.py \
        dashboard/backend/tests/domain/analytics/test_value_queries.py
git status --short
git commit -m "$(cat <<'EOF'
feat: add per-day ledger totals on the credits store and value service

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: The overview — rollups plus one one-day scan; `billing_lane_mix`; attention from facts

**Files:**
- Modify: `dashboard/backend/domain/analytics/metrics.py` — add `ACTIVATION_EVENTS` beside `_TERMINAL_SUCCESS` (line 15)
- Modify: `dashboard/backend/domain/analytics/rollups.py` — `rollup_day` (lines 232-445) writes `activation_users` rows; `rollup_current_day` (lines 448-462) reads one day
- Modify: `dashboard/backend/domain/analytics/query_service.py` — `_ACTIVATION_EVENTS` (line 37), `AnalyticsQueryService.__init__` (633-646), `get_overview` (648-959), new `billing_lane_mix`, new `_legacy_state`
- Test: `dashboard/backend/tests/test_admin_analytics_api.py` (the two query-service tests at lines 230-278 and 409-437, plus new cases)
- Test: `dashboard/backend/tests/domain/analytics/test_rollups.py` (extend)

**Interfaces:**
- Consumes: `AnalyticsRollupStore.list_rollups`, `AnalyticsQueryStore.list_metric_events` (bounded to today), `ValueAnalyticsStore.count_states_for_date`, `list_attention_candidates`, `list_activity` (Task 2 / PR A), `operational_reason` (Task 1), `build_value_analytics_store` (PR T).
- Produces:
  - `AnalyticsQueryService.__init__(*, store=analytics_store, user_store=None, value_store=None)` — `value_store` defaults to `build_value_analytics_store(analytics_base=store)`.
  - `AnalyticsQueryService.get_overview(...)` — same signature, same `AnalyticsOverview` shape, new sources (below).
  - `AnalyticsQueryService.billing_lane_mix(*, start: date, end: date) -> dict[str, dict[str, int]]` — per ISO day, `{"platform_credits": n, "byok": n}` model-call counts from the rollups' `event_count` rows for `model_usage_recorded` (D14). Service method only; PR D exposes it. Run events (`backtest_*`) carry no `billing_mode` (`instrumentation.emit_run_event` has no such parameter; only `emit_resource_event`, line 242-272, does), so "lane mix" here means what `AnalyticsUserProfile.billing_lane_mix` has always meant in this codebase: `model_usage_recorded` events per lane.
  - `rollup_day` additionally writes one `activation_users` row per activation event name per day (`event_name` dimension, `value_count` = distinct users that day). No DDL change.
  - `rollup_current_day(*, store=None, now=None, include_internal=False)` — same signature; reads `[start of today, now)` only and returns the same `AnalyticsOverviewMetrics` type, whose trailing-context fields (`active_users_7d`, `first_success_conversion`, `repeat_run_rate`) are therefore today-only numbers; nothing in the backend calls this function (`rg rollup_current_day` finds only its definition and `__all__`), so narrowing it is the design's §6.15 item 2 stated as code, not a behaviour change anyone reads.
  - `_legacy_state(operational_state, lifecycle_segment) -> str` in `query_service.py` — the `_legacy_seed` mapping (`value_repository.py:185-208`) as a pure function; feeds the `user_state_counts` bridge until Task 16 and the profile's `state` summary (Task 11).

How each `AnalyticsOverview` field is sourced after this task (D18: same field, one suspect per moved number):

| Field | Before | After |
|---|---|---|
| `active_users_7d` | 7-day distinct users from the window scan | `rolling_active_users_7d` on the latest completed day in the window ("as of yesterday"); `None` when no rollup row |
| `first_success_conversion` | mature signups vs. conversions from the scan | `Σ first_success_within_7d_users / Σ mature_signup_cohort_users` over completed days |
| `repeat_run_rate` | from the scan | `users_with_repeat_success_24h_30d / users_with_first_success` on the latest completed day |
| `activation_funnel` | distinct users per event from the scan | `Σ activation_users` rows per event + today's distinct users from the one-day scan (a user who repeats a step on two days counts twice — the daily tier's trade, §6.3) |
| `top_failure_categories` | distinct users per category from the scan | `affected_users` rows keyed by `error_category` summed over completed days + today; with a `billing_mode`/`provider`/`model` filter, `event_count` rows (which carry those dimensions) instead — a count of events, since `affected_users` has no dimension |
| `completed_runs`, `failed_runs`, `backtest_success_rate`, `platform_model_cost_usd`, `daily_*` | rollups + today's slice of the window scan | unchanged logic; today's slice is now the whole scan |
| `input_tokens`, `output_tokens` | summed over the whole window scan (`:830-841`) | rollups `input_tokens`/`output_tokens` rows (total row, or the per-dimension rows when filtered) + today |
| `user_state_counts` | `user_analytics_snapshots` full scan | bridge: yesterday's `count_states_for_date` mapped through `_legacy_state` — removed in Task 16 |
| `users_needing_attention` | snapshot states + 30-day event scan + `users` full scan | `list_attention_candidates(day=D, window_start=D-29)`; `last_meaningful_activity` from `user_activity` (D20) |

- [ ] **Step 1: Write the failing tests**

In `dashboard/backend/tests/test_admin_analytics_api.py`, replace `test_overview_marks_only_failed_rollup_panel_unavailable` (lines 409-437) — its `active_users_7d == 1` assertion described the scan-sourced number — with:

```python
def test_overview_marks_rollup_backed_panels_unavailable_when_rollups_fail(tmp_path, monkeypatch):
    analytics, events, _rollups, _states, users = _fixture(tmp_path)
    _event(
        events,
        "backtest_completed",
        NOW - timedelta(minutes=1),
        "run:backtest_completed:current",
        outcome="succeeded",
    )
    service = AnalyticsQueryService(store=analytics, user_store=users)
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
    # Panels that never touch rollups stay up.
    assert overview.availability["friction"].available is True
    assert overview.availability["attention"].available is True
```

and add, after it:

```python
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

    analytics, _events, _rollups, _states, users = _fixture(tmp_path)
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
    analytics, _events, _rollups, _states, users = _fixture(tmp_path)
    service = AnalyticsQueryService(store=analytics, user_store=users)
    recorder = _WindowRecordingQueryStore(service.query_store)
    service.query_store = recorder

    service.get_overview(
        now=NOW,
        filters=AnalyticsMetricFilters(
            start=NOW - timedelta(days=10),
            end=datetime.combine(NOW.date(), datetime.min.time(), tzinfo=timezone.utc),
        ),
    )

    assert recorder.windows == []


def test_overview_trailing_metrics_and_funnel_come_from_rollups(tmp_path):
    analytics, events, rollups, _states, users = _fixture(tmp_path)
    yesterday = NOW.date() - timedelta(days=1)
    stamp = datetime.combine(NOW.date(), datetime.min.time(), tzinfo=timezone.utc)
    rollups.replace_day(
        yesterday,
        [
            DailyRollup(rollup_date=yesterday, metric_name="rolling_active_users_7d", value_count=9, updated_at=stamp),
            DailyRollup(rollup_date=yesterday, metric_name="mature_signup_cohort_users", value_count=4, updated_at=stamp),
            DailyRollup(rollup_date=yesterday, metric_name="first_success_within_7d_users", value_count=1, updated_at=stamp),
            DailyRollup(rollup_date=yesterday, metric_name="users_with_first_success", value_count=8, updated_at=stamp),
            DailyRollup(rollup_date=yesterday, metric_name="users_with_repeat_success_24h_30d", value_count=2, updated_at=stamp),
            DailyRollup(rollup_date=yesterday, metric_name="activation_users", event_name="account_signed_up", value_count=5, updated_at=stamp),
            DailyRollup(rollup_date=yesterday, metric_name="activation_users", event_name="backtest_completed", value_count=2, updated_at=stamp),
            DailyRollup(rollup_date=yesterday, metric_name="affected_users", error_category="provider_timeout", value_count=3, updated_at=stamp),
            DailyRollup(rollup_date=yesterday, metric_name="input_tokens", value_count=1000, updated_at=stamp),
            DailyRollup(rollup_date=yesterday, metric_name="output_tokens", value_count=250, updated_at=stamp),
        ],
    )
    _event(events, "backtest_completed", NOW - timedelta(minutes=2), "run:backtest_completed:today", outcome="succeeded")
    _event(
        events, "backtest_failed", NOW - timedelta(minutes=1), "run:backtest_failed:today-fail",
        source_record_id="today-fail", outcome="failed", error_category="provider_timeout",
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
    assert [(c.error_category, c.affected_users) for c in overview.top_failure_categories] == [
        ("provider_timeout", 4)
    ]
    assert overview.input_tokens == 1000
    assert overview.output_tokens == 250


def test_billing_lane_mix_counts_model_calls_per_lane_per_day(tmp_path):
    analytics, _events, rollups, _states, users = _fixture(tmp_path)
    day = NOW.date() - timedelta(days=2)
    stamp = datetime.combine(NOW.date(), datetime.min.time(), tzinfo=timezone.utc)
    rollups.replace_day(
        day,
        [
            DailyRollup(rollup_date=day, metric_name="event_count", event_name="model_usage_recorded",
                        billing_mode="platform_credits", provider_id="openrouter", model_id="a", value_count=3, updated_at=stamp),
            DailyRollup(rollup_date=day, metric_name="event_count", event_name="model_usage_recorded",
                        billing_mode="platform_credits", provider_id="openrouter", model_id="b", value_count=1, updated_at=stamp),
            DailyRollup(rollup_date=day, metric_name="event_count", event_name="model_usage_recorded",
                        billing_mode="byok", provider_id="openai", model_id="c", value_count=2, updated_at=stamp),
            DailyRollup(rollup_date=day, metric_name="event_count", event_name="backtest_completed", value_count=7, updated_at=stamp),
        ],
    )
    service = AnalyticsQueryService(store=analytics, user_store=users)

    mix = service.billing_lane_mix(start=day, end=day + timedelta(days=2))

    assert mix == {
        day.isoformat(): {"platform_credits": 4, "byok": 2},
        (day + timedelta(days=1)).isoformat(): {"platform_credits": 0, "byok": 0},
    }
    assert "billing_lane_mix" not in AnalyticsOverview.model_fields
```

Add `from dashboard.backend.domain.analytics.query_service import AnalyticsOverview` to the test module's imports. Then extend `dashboard/backend/tests/domain/analytics/test_rollups.py`:

```python
def test_rollup_day_writes_distinct_activation_users_per_event(tmp_path):
    analytics, rollups = _store(tmp_path)
    service = AnalyticsService(analytics)
    day = date(2026, 8, 25)
    for index, (user_id, name, record_type) in enumerate(
        (
            (1, "account_signed_up", "user"),
            (2, "account_signed_up", "user"),
            (1, "agent_created", "agent"),
            (1, "agent_created", "agent"),
            (1, "backtest_completed", "run"),
        )
    ):
        service.record_server_event(
            event_name=name,
            user_id=user_id,
            source_event_id=f"{record_type}:{name}:{index}",
            source_record_type=record_type,
            source_record_id=str(index),
            occurred_at=datetime(2026, 8, 25, 10 + index, 0, tzinfo=timezone.utc),
        )

    rollup_day(day, store=rollups)
    stored = rollups.list_rollups(start=day, end=day + timedelta(days=1))

    assert {
        (row.event_name, row.value_count)
        for row in stored
        if row.metric_name == "activation_users"
    } == {("account_signed_up", 2), ("agent_created", 1), ("backtest_completed", 1)}


def test_rollup_current_day_reads_only_the_current_day(tmp_path):
    analytics, rollups = _store(tmp_path)
    windows = []
    original = rollups.list_events

    def recording(**kwargs):
        windows.append((kwargs["start"], kwargs["end"]))
        return original(**kwargs)

    rollups.list_events = recording
    now = datetime(2026, 8, 25, 15, 30, tzinfo=timezone.utc)

    rollup_current_day(store=rollups, now=now)

    assert windows == [(datetime(2026, 8, 25, tzinfo=timezone.utc), now)]
```

Add `rollup_current_day` to the test module's `rollups` import.

- [ ] **Step 2: Run them and confirm the expected failure**

```bash
python -m pytest dashboard/backend/tests/test_admin_analytics_api.py -v -k "overview or billing_lane"
python -m pytest dashboard/backend/tests/domain/analytics/test_rollups.py -v -k "activation_users or current_day"
```

Expected: **FAIL** — the window recorder sees a 30-day window (`start == NOW - 30 days`), `active_users_7d == 1` where `None`/`9` is asserted, `activation_users` rows are absent, `billing_lane_mix` does not exist, and `rollup_current_day` reads `start - 30 days`.

- [ ] **Step 3: `metrics.py` and `rollups.py`**

In `metrics.py`, after `_TERMINAL_FAILURE` (line 16):

```python
ACTIVATION_EVENTS: tuple[str, ...] = (
    "account_signed_up",
    "credential_verified",
    "agent_created",
    "backtest_completed",
)
```

and add `"ACTIVATION_EVENTS"` to `__all__`. In `rollups.py`, import it (`from .metrics import ACTIVATION_EVENTS, calculate_overview_metrics`) and in `rollup_day`, directly after the `affected_users`-by-category loop (line 420) and before the `state_counts` loop (line 421), add:

```python
    activation_users: dict[str, set[int]] = defaultdict(set)
    for event in day_events:
        if event.event_name in ACTIVATION_EVENTS:
            activation_users[event.event_name].add(event.user_id)
    for event_name in ACTIVATION_EVENTS:
        if activation_users[event_name]:
            rows.append(
                _row(
                    day,
                    "activation_users",
                    count=len(activation_users[event_name]),
                    event_name=event_name,
                    updated_at=current,
                )
            )
```

Replace `rollup_current_day` (lines 448-462) with:

```python
def rollup_current_day(
    *,
    store: AnalyticsRollupStore | None = None,
    now: datetime | None = None,
    include_internal: bool = False,
):
    """Metrics for the current UTC day from one bounded one-day scan.

    Trailing-context fields (``active_users_7d``, ``first_success_conversion``,
    ``repeat_run_rate``) are computed over today's events only. The overview
    takes that context from the rollups instead (design §6.15 item 2); this
    function no longer widens the read to supply it.
    """
    current = now or datetime.now(timezone.utc)
    start = _day_start(current.astimezone(timezone.utc).date())
    aggregate_store = store or AnalyticsRollupStore()
    events = aggregate_store.list_events(
        start=start,
        end=current,
        include_internal=include_internal,
    )
    return calculate_overview_metrics(events, start=start, end=current)
```

- [ ] **Step 4: `query_service.py`**

Replace the private tuple at lines 37-42 with an import: extend `from .metrics import (...)` with `ACTIVATION_EVENTS` and set `_ACTIVATION_EVENTS = ACTIVATION_EVENTS`. Add after `_availability` (line 616):

```python
def _legacy_state(operational_state: str, lifecycle_segment: str) -> str:
    """The five-state label the snapshot writer derived from the two axes
    (value_repository._legacy_seed). Kept only while user_state_counts and
    the profile's state summary exist; both go in the D20 task."""

    if operational_state in {"blocked", "needs_attention"}:
        return operational_state
    if lifecycle_segment == "dormant":
        return "dormant"
    if lifecycle_segment in {"new", "onboarding"}:
        return "onboarding"
    return "active"


def _latest_rollup_value(rows: Sequence[DailyRollup], metric_name: str) -> int | None:
    latest = None
    for row in rows:
        if (
            row.metric_name == metric_name
            and not row.event_name
            and not row.provider_id
            and not row.model_id
            and not row.billing_mode
        ):
            if latest is None or row.rollup_date > latest.rollup_date:
                latest = row
    return None if latest is None else latest.value_count


def _sum_rollup(rows: Sequence[DailyRollup], metric_name: str) -> int:
    return sum(
        row.value_count
        for row in rows
        if row.metric_name == metric_name
        and not row.event_name
        and not row.provider_id
        and not row.model_id
        and not row.billing_mode
        and not row.error_category
    )
```

(`Sequence` joins the `typing` import.) Change `__init__` (633-646):

```python
    def __init__(
        self,
        *,
        store: Any = analytics_store,
        user_store: Any | None = None,
        value_store: Any | None = None,
    ):
        if user_store is None:
            from dashboard.backend.users import user_store as default_user_store

            user_store = default_user_store
        if value_store is None:
            from .value_repository import build_value_analytics_store

            value_store = build_value_analytics_store(analytics_base=store)
        self.store = store
        self.user_store = user_store
        self.value_store = value_store
        self.query_store = AnalyticsQueryStore(store)
```

Replace `get_overview` (648-959) in full:

```python
    def get_overview(
        self,
        *,
        filters: AnalyticsMetricFilters,
        now: datetime | None = None,
    ) -> AnalyticsOverview:
        if not isinstance(filters, AnalyticsMetricFilters):
            filters = AnalyticsMetricFilters.model_validate(filters)
        current = _utc(now or datetime.now(timezone.utc), "now")
        effective_end = min(filters.end, current)
        yesterday = current.date() - timedelta(days=1)
        today_start = datetime.combine(
            current.date(), datetime.min.time(), tzinfo=timezone.utc
        )
        availability = {
            "snapshot": _availability(),
            "growth": _availability(),
            "funnel": _availability(),
            "friction": _availability(),
            "attention": _availability(),
        }
        dimensional = any(
            value is not None
            for value in (filters.billing_mode, filters.provider_id, filters.model_id)
        )

        # One bounded raw read: the current UTC day, and only if the window reaches it.
        current_events: list[_MetricEvent] = []
        scan_available = True
        scan_start = max(filters.start, today_start)
        if effective_end > scan_start:
            try:
                current_events = [
                    event
                    for event in self.query_store.list_metric_events(
                        start=scan_start,
                        end=effective_end,
                        include_internal=filters.include_internal,
                    )
                    if _event_matches_filters(event, filters)
                ]
            except Exception:
                scan_available = False

        rollups: list[DailyRollup] = []
        rollups_available = True
        try:
            historical_end = min(effective_end.date(), current.date())
            if historical_end > filters.start.date():
                rollups = self.query_store.rollups.list_rollups(
                    start=filters.start.date(),
                    end=historical_end,
                )
        except Exception:
            rollups_available = False

        active_users: int | None = None
        conversion: float | None = None
        repeat_rate: float | None = None
        if rollups_available:
            active_users = _latest_rollup_value(rollups, "rolling_active_users_7d")
            mature = _sum_rollup(rollups, "mature_signup_cohort_users")
            converted = _sum_rollup(rollups, "first_success_within_7d_users")
            conversion = None if mature == 0 else converted / mature
            first_success = _latest_rollup_value(rollups, "users_with_first_success")
            repeat = _latest_rollup_value(rollups, "users_with_repeat_success_24h_30d")
            if first_success:
                repeat_rate = (repeat or 0) / first_success
        else:
            availability["snapshot"] = _availability(False)

        funnel: dict[str, int] = {}
        failures: list[FailureCategoryCount] = []
        if rollups_available and scan_available:
            for event_name in _ACTIVATION_EVENTS:
                funnel[event_name] = sum(
                    row.value_count
                    for row in rollups
                    if row.metric_name == "activation_users"
                    and row.event_name == event_name
                ) + len(
                    {
                        event.user_id
                        for event in current_events
                        if event.event_name == event_name
                    }
                )
            failure_counts: dict[str, int] = defaultdict(int)
            for row in rollups:
                if not row.error_category:
                    continue
                if dimensional:
                    if row.metric_name == "event_count" and _matches_rollup_dimensions(
                        row, filters
                    ):
                        failure_counts[row.error_category] += row.value_count
                elif row.metric_name == "affected_users":
                    failure_counts[row.error_category] += row.value_count
            today_failures: dict[str, set[int]] = defaultdict(set)
            for event in current_events:
                if event.error_category:
                    today_failures[event.error_category].add(event.user_id)
            for category, users in today_failures.items():
                failure_counts[category] += len(users)
            failures = [
                FailureCategoryCount(error_category=category, affected_users=count)
                for category, count in sorted(
                    failure_counts.items(), key=lambda item: (-item[1], item[0])
                )[:10]
            ]
        else:
            availability["funnel"] = _availability(False)

        completed: int | None = None
        failed: int | None = None
        success_rate: float | None = None
        platform_cost: float | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None
        daily_active: dict[str, int] = {}
        daily_completed: dict[str, int] = {}
        if rollups_available and scan_available:
            if dimensional:
                historical_completed = sum(
                    row.value_count
                    for row in rollups
                    if row.metric_name == "event_count"
                    and row.event_name == "backtest_completed"
                    and _matches_rollup_dimensions(row, filters)
                )
                historical_failed = sum(
                    row.value_count
                    for row in rollups
                    if row.metric_name == "event_count"
                    and row.event_name == "backtest_failed"
                    and _matches_rollup_dimensions(row, filters)
                )
            else:
                historical_completed = _sum_rollup(rollups, "terminal_completed")
                historical_failed = _sum_rollup(rollups, "terminal_failed")
            current_completed = sum(
                event.event_name == "backtest_completed" for event in current_events
            )
            current_failed = sum(
                event.event_name == "backtest_failed" for event in current_events
            )
            completed = historical_completed + current_completed
            failed = historical_failed + current_failed
            denominator = completed + failed
            success_rate = None if denominator == 0 else completed / denominator

            if filters.billing_mode == "byok":
                platform_micro = 0
            elif filters.provider_id is not None or filters.model_id is not None:
                platform_micro = sum(
                    row.value_sum_micro
                    for row in rollups
                    if row.metric_name == "platform_model_cost_usd"
                    and row.billing_mode == "platform_credits"
                    and bool(row.provider_id)
                    and bool(row.model_id)
                    and (filters.provider_id is None or row.provider_id == filters.provider_id)
                    and (filters.model_id is None or row.model_id == filters.model_id)
                )
            else:
                platform_micro = sum(
                    row.value_sum_micro
                    for row in rollups
                    if row.metric_name == "platform_model_cost_usd"
                    and row.billing_mode == "platform_credits"
                    and not row.provider_id
                    and not row.model_id
                )
            platform_micro += sum(
                int(event.properties.get("cost_micro_usd", 0))
                for event in current_events
                if event.event_name == "model_usage_recorded"
                and event.billing_mode == "platform_credits"
            )
            platform_cost = platform_micro / 1_000_000

            def token_total(metric_name: str) -> int:
                if dimensional:
                    historical = sum(
                        row.value_count
                        for row in rollups
                        if row.metric_name == metric_name
                        and (row.billing_mode or row.provider_id or row.model_id)
                        and _matches_rollup_dimensions(row, filters)
                    )
                else:
                    historical = _sum_rollup(rollups, metric_name)
                return historical + sum(
                    int(event.properties.get(metric_name, 0))
                    for event in current_events
                    if event.event_name == "model_usage_recorded"
                )

            input_tokens = token_total("input_tokens")
            output_tokens = token_total("output_tokens")
            for row in rollups:
                key = row.rollup_date.isoformat()
                if row.metric_name == "daily_active_users" and not dimensional:
                    daily_active[key] = row.value_count
                if row.metric_name in {"completed_runs", "terminal_completed"} and not dimensional:
                    daily_completed[key] = row.value_count
                if (
                    dimensional
                    and row.metric_name == "event_count"
                    and row.event_name == "backtest_completed"
                    and _matches_rollup_dimensions(row, filters)
                ):
                    daily_completed[key] = daily_completed.get(key, 0) + row.value_count
            if current_events:
                day_key = current.date().isoformat()
                daily_active[day_key] = len(
                    {event.user_id for event in current_events if is_meaningful_event(event)}
                )
                daily_completed[day_key] = current_completed
        else:
            availability["growth"] = _availability(False)

        state_counts: dict[str, int] = {}
        try:
            for cell in self.value_store.count_states_for_date(yesterday):
                key = _legacy_state(cell.operational_state, cell.lifecycle_segment)
                state_counts[key] = state_counts.get(key, 0) + cell.users
        except Exception:
            availability["friction"] = _availability(False)

        attention: list[AnalyticsUserListItem] = []
        try:
            candidates = self.value_store.list_attention_candidates(
                day=yesterday,
                window_start=yesterday - timedelta(days=29),
                limit=10,
            )
            activity = (
                self.value_store.list_activity([c.user_id for c in candidates])
                if candidates
                else {}
            )
            attention = [
                AnalyticsUserListItem(
                    user_id=candidate.user_id,
                    display_name=candidate.display_name,
                    email=candidate.email,
                    joined_at=candidate.created_at,
                    status=candidate.operational_state,
                    reason_code=candidate.operational_reason_code,
                    human_readable_reason=operational_reason(
                        candidate.operational_reason_code
                    ),
                    last_meaningful_activity=(
                        activity[candidate.user_id].last_meaningful_activity_at
                        if candidate.user_id in activity
                        else None
                    ),
                    recent_runs=candidate.recent_runs,
                    recent_failures=candidate.recent_failures,
                    profile_path=f"/admin/analytics/users/{candidate.user_id}",
                )
                for candidate in candidates
            ]
        except Exception:
            availability["attention"] = _availability(False)

        return AnalyticsOverview(
            active_users_7d=active_users,
            first_success_conversion=conversion,
            backtest_success_rate=success_rate,
            repeat_run_rate=repeat_rate,
            platform_model_cost_usd=platform_cost,
            completed_runs=completed,
            failed_runs=failed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            daily_active_users=daily_active,
            daily_completed_runs=daily_completed,
            activation_funnel=funnel,
            user_state_counts=state_counts,
            top_failure_categories=failures,
            users_needing_attention=attention,
            last_updated=current,
            filters=filters,
            availability=availability,
        )

    def billing_lane_mix(self, *, start: date, end: date) -> dict[str, dict[str, int]]:
        """Model calls per billing lane per UTC day, from the rollups (D14).

        Service method only in PR B; PR D adds it to AnalyticsOverview.
        Zero-filled for every day in [start, end) so a chart never has holes.
        """
        if end <= start:
            raise ValueError("end must be later than start")
        rows = self.query_store.rollups.list_rollups(start=start, end=end)
        mix = {
            (start + timedelta(days=offset)).isoformat(): {"platform_credits": 0, "byok": 0}
            for offset in range((end - start).days)
        }
        for row in rows:
            if (
                row.metric_name == "event_count"
                and row.event_name == "model_usage_recorded"
                and row.billing_mode in mix.get(row.rollup_date.isoformat(), {})
            ):
                mix[row.rollup_date.isoformat()][row.billing_mode] += row.value_count
        return mix
```

Add `from .lifecycle import operational_reason` and `from datetime import date, datetime, timedelta, timezone` (extend the existing import) to `query_service.py`. `_ATTENTION_STATES` (line 36), `list_snapshots`, `summarize_users` and `_all_users` are now unused by `get_overview`; leave them until Task 15 (PR A may have left `_all_users` with no caller already; removing dead helpers is that task's job).

- [ ] **Step 5: Run and confirm the pass**

```bash
python -m pytest dashboard/backend/tests/test_admin_analytics_api.py dashboard/backend/tests/domain/analytics/test_rollups.py dashboard/backend/tests/domain/analytics/test_repository_contract.py -v
```

Expected: **PASS** after one seed change in `test_repository_contract.py`. `test_query_service_merges_completed_rollups_with_current_raw_day` (line 230) passes unchanged — its `terminal_completed` rollup and today's event still sum to 5. `assert_pr2_query_contract` (`test_repository_contract.py:218`) asserts `overview.completed_runs == 1` and `failed_runs == 1` from events recorded at `NOW - 1h`, which the one-day scan covers; but its attention case seeds the state through `state_store.upsert_snapshot(UserAnalyticsSnapshot(status="needs_attention", ...))` (lines 308-317), which the overview no longer reads. Replace those ten lines with a fact-row seed (the contract runs on the Postgres tier too, so this is the shared helper, not a SQLite-only test):

```python
    value_store = build_value_analytics_store(
        analytics_base=store,
        credits_base=object(),
        provider_base=object(),
        agent_base=object(),
        run_base=object(),
    )
    value_store.upsert_daily_facts(
        [
            UserDailyFact(
                snapshot_date=NOW.date() - timedelta(days=1),
                user_id=user_id,
                lifecycle_segment="growing",
                lifecycle_reason_code="growing_activated_below_core_threshold",
                operational_state="needs_attention",
                operational_reason_code="invalid_default_credential",
                tier="unpaid",
                user_group="unknown",
                active=True,
                runs_requested=2,
                runs_completed=1,
                runs_failed=1,
                runs_cancelled=0,
                operator_cost_micro=0,
                own_spend_micro=0,
                data_quality="complete",
                calculated_at=NOW,
            )
        ]
    )
```

with `from dashboard.backend.domain.analytics.value_repository import UserDailyFact, build_value_analytics_store` added to that file's imports. The two assertions at lines 340-341 (`users_needing_attention[0].user_id == user_id`, `.recent_failures == 1`) then hold from the fact row. The `states` import at lines 27-30 stays until Task 15 (line 281's `recalculate_user_snapshot` still feeds `profile.state` until Task 11 re-sources it).

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/domain/analytics/metrics.py \
        dashboard/backend/domain/analytics/rollups.py \
        dashboard/backend/domain/analytics/query_service.py \
        dashboard/backend/tests/test_admin_analytics_api.py \
        dashboard/backend/tests/domain/analytics/test_rollups.py \
        dashboard/backend/tests/domain/analytics/test_repository_contract.py
git status --short
git commit -m "$(cat <<'EOF'
perf: bound the analytics overview to rollups plus one current-day scan

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `/lifecycle` from yesterday's facts, the transition log and `user_activity`

**Files:**
- Modify: `dashboard/backend/domain/analytics/value_queries.py` — `_history` (lines 562-725), `get_lifecycle` (727-802); delete `_daily` (544-560)
- Test: `dashboard/backend/tests/domain/analytics/test_value_queries.py` — replace `FakeValueStore` (lines 132-176) and `FakeQueryStore` (194-203) with the fact-backed fakes below; rewrite the five lifecycle tests named in Step 1

**Interfaces:**
- Consumes: `count_states_for_date`, `count_segments_by_date`, `count_transitions`, `count_activated_users` (Task 2); `list_commercial_values` (unchanged, the ledger is the source of `paid_users`); rollups `lifecycle_segment_count` / `lifecycle_transition` rows for dates older than the 180-day facts horizon (unchanged fallback).
- Produces: `get_lifecycle` with the same signature and the same `LifecycleAnalyticsResponse`. `_history(*, start, end, use_anonymous_rollups, movement_start=None, movement_range="5d")` loses its `user_ids` parameter — the aggregate is grouped in SQL, so there is nothing to pass.

§6.10: *segment counts and movement from yesterday's `user_daily_facts` and `lifecycle_transitions`; headline `activated_users` from `user_activity`; read-time segments are for the one-user profile only.* Two value changes to record: `segment_counts` and `core_users`/`at_risk_users` are as of yesterday's fact rows rather than the live snapshot (the design's daily tier, §6.3), and `include_internal=True` no longer adds admins to those counts because admins have no fact rows (§6.7) — `activated_users` and `paid_users`, which read `user_activity` and the ledger, still honour the flag. `weekly_segments`/`movement_segments` keep the rollup fallback for days the facts table no longer holds (`use_anonymous_rollups = not include_internal`, unchanged).

- [ ] **Step 1: Replace the fakes and write the failing tests**

In `test_value_queries.py`, delete `FakeValueStore` (132-176) and `FakeQueryStore` (194-203) and put in their place (the `_daily`, `_snapshot` helpers stay until Tasks 10-11 remove their last users; add `FactStateCount, SegmentDayCount, TransitionCount, GroupFactTotals, AttentionCandidate, UserActivity, UserDailyFact, UserPageRow, RecentFactTotals, LifecycleTransitionRow` to the `value_repository` import):

```python
class FakeValueStore:
    """Fact-backed fake: answers every Task 2/3 read from in-memory rows."""

    def __init__(
        self,
        *,
        facts=(),
        transitions=(),
        activity=None,
        commercial=None,
        page_rows=(),
        operational_facts=None,
    ):
        self.facts = list(facts)
        self.transitions = list(transitions)
        self.activity = dict(activity or {})
        self.commercial = dict(commercial or {})
        self.page_rows = list(page_rows)
        self.operational_facts = dict(operational_facts or {})
        self.commercial_windows = []
        self.page_calls = []
        self.recent_windows = []

    # -- Task 2 reads --
    def count_states_for_date(self, day):
        cells = {}
        for row in self.facts:
            if row.snapshot_date != day:
                continue
            key = (row.operational_state, row.operational_reason_code, row.lifecycle_segment)
            users, partial = cells.get(key, (0, 0))
            cells[key] = (users + 1, partial + (row.data_quality == "partial"))
        return [
            FactStateCount(
                operational_state=key[0], operational_reason_code=key[1],
                lifecycle_segment=key[2], users=users, partial_rows=partial,
            )
            for key, (users, partial) in sorted(cells.items())
        ]

    def count_segments_by_date(self, *, start, end):
        cells = {}
        for row in self.facts:
            if not start <= row.snapshot_date < end:
                continue
            key = (row.snapshot_date, row.lifecycle_segment)
            users, partial = cells.get(key, (0, 0))
            cells[key] = (users + 1, partial + (row.data_quality == "partial"))
        return [
            SegmentDayCount(snapshot_date=key[0], lifecycle_segment=key[1], users=users, partial_rows=partial)
            for key, (users, partial) in sorted(cells.items())
        ]

    def count_transitions(self, *, start, end):
        cells = {}
        for row in self.transitions:
            if not start <= row.snapshot_date < end:
                continue
            key = (row.from_segment, row.to_segment)
            users, partial = cells.get(key, (0, 0))
            cells[key] = (users + 1, partial + (row.data_quality == "partial"))
        return [
            TransitionCount(from_segment=key[0], to_segment=key[1], users=users, partial_rows=partial)
            for key, (users, partial) in sorted(cells.items())
        ]

    def list_attention_candidates(self, *, day, window_start, limit):
        return []

    def sum_facts_by_user_group(self, *, start, end):
        totals = {}
        completion_days = {}
        for row in self.facts:
            if not start <= row.snapshot_date < end:
                continue
            group = totals.setdefault(row.user_group, {"total_runs": 0, "operator_cost_micro": 0, "successful": set()})
            group["total_runs"] += row.runs_requested
            group["operator_cost_micro"] += row.operator_cost_micro
            if row.runs_completed > 0:
                group["successful"].add(row.user_id)
                completion_days.setdefault((row.user_group, row.user_id), set()).add(row.snapshot_date)
        return {
            name: GroupFactTotals(
                total_runs=values["total_runs"],
                operator_cost_micro=values["operator_cost_micro"],
                successful_run_users=len(values["successful"]),
                repeat_users=sum(
                    1 for (group, _user), days in completion_days.items()
                    if group == name and len(days) >= 2
                ),
            )
            for name, values in totals.items()
        }

    def list_active_dates(self, user_ids, *, start, end):
        result = {user_id: [] for user_id in user_ids}
        for row in sorted(self.facts, key=lambda r: (r.user_id, r.snapshot_date)):
            if row.user_id in result and row.active and start <= row.snapshot_date < end:
                result[row.user_id].append(row.snapshot_date)
        return result

    def list_activations(self, *, start, end, include_internal):
        return {
            user_id: activity.activated_at
            for user_id, activity in self.activity.items()
            if activity.activated_at is not None and start <= activity.activated_at < end
            and (include_internal or not getattr(activity, "internal", False))
        }

    def count_activated_users(self, *, include_internal):
        return sum(
            1 for activity in self.activity.values()
            if activity.activated_at is not None
            and (include_internal or not getattr(activity, "internal", False))
        )

    def list_user_transitions(self, user_id, *, start, end):
        return [
            row for row in self.transitions
            if row.user_id == user_id and start <= row.snapshot_date < end
        ]

    def list_transitions_for_date(self, day):
        return [row for row in self.transitions if row.snapshot_date == day]

    # -- Task 3 --
    def list_users_page(self, *, query, day, limit, offset):
        self.page_calls.append((query, day, limit, offset))
        rows = list(self.page_rows)
        return rows[offset : offset + limit], len(rows)

    # -- PR A --
    def get_activity(self, user_id):
        return self.activity.get(user_id)

    def list_activity(self, user_ids):
        return {user_id: self.activity[user_id] for user_id in user_ids if user_id in self.activity}

    def sum_recent_facts(self, user_ids, *, start, end):
        self.recent_windows.append((start, end))
        result = {}
        for user_id in user_ids:
            rows = [r for r in self.facts if r.user_id == user_id and start <= r.snapshot_date <= end]
            if not rows:
                continue
            active_dates = [r.snapshot_date for r in rows if r.active]
            result[user_id] = RecentFactTotals(
                active_days=len(active_dates),
                successful_backtests=sum(r.runs_completed for r in rows),
                runs_requested=sum(r.runs_requested for r in rows),
                runs_completed=sum(r.runs_completed for r in rows),
                runs_failed=sum(r.runs_failed for r in rows),
                runs_cancelled=sum(r.runs_cancelled for r in rows),
                operator_cost_micro=sum(r.operator_cost_micro for r in rows),
                own_spend_micro=sum(r.own_spend_micro for r in rows),
                days_present=len(rows),
                last_active_date=max(active_dates) if active_dates else None,
            )
        return result

    def get_operational_facts(self, user_id, *, now):
        return self.operational_facts[user_id]

    # -- existing --
    def list_commercial_values(self, user_ids, *, start, end):
        self.commercial_windows.append((start, end))
        return {
            user_id: self.commercial[user_id] for user_id in user_ids if user_id in self.commercial
        }


class FakeRollups:
    def __init__(self, events=(), rollups=()):
        self.events = list(events)
        self.rollups = list(rollups)

    def list_events(self, *, start, end, include_internal, user_id=None):
        raise AssertionError("value queries must not read raw events after PR B")

    def list_rollups(self, *, start, end):
        return [row for row in self.rollups if start <= row.rollup_date < end]


class FakeQueryStore:
    def __init__(self, *, events=(), rollups=()):
        self.rollups = FakeRollups(events, rollups)
```

Add two builders beside `_daily`:

```python
def _fact(user_id: int, day: date, **overrides) -> UserDailyFact:
    values = dict(
        snapshot_date=day, user_id=user_id, lifecycle_segment="core",
        lifecycle_reason_code="core_repeated_value", operational_state="healthy",
        operational_reason_code="no_supported_issue", tier="unpaid", user_group="unknown",
        active=True, runs_requested=1, runs_completed=1, runs_failed=0, runs_cancelled=0,
        operator_cost_micro=0, own_spend_micro=0, data_quality="complete", calculated_at=NOW,
    )
    values.update(overrides)
    return UserDailyFact(**values)


def _activity(user_id: int, *, activated_days_ago=60, active_days_ago=2) -> UserActivity:
    return UserActivity(
        user_id=user_id,
        activated_at=None if activated_days_ago is None else NOW - timedelta(days=activated_days_ago),
        last_meaningful_activity_at=NOW - timedelta(days=active_days_ago),
        updated_at=NOW,
    )
```

`UserActivity` is frozen and carries no role, so the fake models "internal" as a set of ids: give `FakeValueStore.__init__` an `internal_ids=()` keyword stored as `self.internal_ids = set(internal_ids)`, and in `list_activations` / `count_activated_users` replace `(include_internal or not getattr(activity, "internal", False))` with `(include_internal or user_id not in self.internal_ids)` (iterate `self.activity.items()` in both so `user_id` is in scope).

Rewrite `_service` (lines 258-290 today) to build the fake from these keywords: `_service(*, facts=(), transitions=(), activity=None, commercial=None, page_rows=(), operational_facts=None, internal_ids=(), users=(), rollups=(), excluded=(), legacy_availability=None, credits_reader=None)`, passing `FakeUserStore([_user(user_id, group) for user_id, group in users])`, `FakeQueryStore(rollups=rollups)`, `FakeLegacyService(legacy_availability)` and `credits_reader` through to `ValueAnalyticsQueryService(...)`; every existing caller of `_service` that passes `snapshots=`/`daily=`/`events=`/`user_groups=` is rewritten in this task or the one that owns that route.

Delete these five tests, whose fixtures are the snapshot model — `test_date_filter_changes_history_not_current_lifecycle_identity` (308), `test_lifecycle_movement_returns_selected_range_and_granularity` (341), `test_lifecycle_transitions_ignore_rollups_outside_the_requested_window` (753), `test_lifecycle_coverage_reports_the_requested_window_not_the_movement_history` (788), `test_lifecycle_movement_buckets_never_start_before_the_selected_window` (814), `test_lifecycle_daily_scan_stays_within_the_derived_history_bound` (834) — and add:

```python
YESTERDAY = NOW.date() - timedelta(days=1)


def _lifecycle_service(**overrides):
    facts = [
        _fact(1, YESTERDAY, lifecycle_segment="core"),
        _fact(2, YESTERDAY, lifecycle_segment="at_risk", active=False),
        _fact(3, YESTERDAY, lifecycle_segment="onboarding", runs_completed=0),
        _fact(1, YESTERDAY - timedelta(days=1), lifecycle_segment="growing"),
        _fact(2, YESTERDAY - timedelta(days=1), lifecycle_segment="growing", data_quality="partial"),
        _fact(3, YESTERDAY - timedelta(days=1), lifecycle_segment="new", runs_completed=0),
    ]
    transitions = [
        LifecycleTransitionRow(user_id=1, snapshot_date=YESTERDAY, from_segment="growing", to_segment="core", inactive_days=0, data_quality="complete", created_at=NOW),
        LifecycleTransitionRow(user_id=2, snapshot_date=YESTERDAY, from_segment="growing", to_segment="at_risk", inactive_days=8, data_quality="partial", created_at=NOW),
    ]
    activity = {1: _activity(1), 2: _activity(2, active_days_ago=8), 3: _activity(3, activated_days_ago=None), 9: _activity(9)}
    values = dict(
        facts=facts, transitions=transitions, activity=activity, internal_ids={9},
        users=((1, "organic"), (2, "partner"), (3, "unknown"), (9, "internal")),
        commercial={1: _commercial(1, 6_000_000), 2: _commercial(2), 3: _commercial(3), 9: _commercial(9)},
    )
    values.update(overrides)
    return _service(**values)


def test_lifecycle_counts_are_yesterdays_facts_and_headline_reads_activity():
    service = _lifecycle_service()

    response = service.get_lifecycle(
        start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1), now=NOW
    )

    assert response.segment_counts == {
        "new": 0, "onboarding": 1, "growing": 0, "core": 1, "at_risk": 1, "dormant": 0
    }
    assert response.headline.core_users == 1
    assert response.headline.at_risk_users == 1
    assert response.headline.activated_users == 2  # user 9 is internal
    assert response.headline.paid_users == 1
    assert response.availability["current"].status == "ready"


def test_lifecycle_include_internal_counts_internal_activations():
    service = _lifecycle_service()

    response = service.get_lifecycle(
        start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1),
        include_internal=True, now=NOW,
    )

    assert response.headline.activated_users == 3


def test_lifecycle_movement_and_transitions_come_from_facts_and_the_transition_log():
    service = _lifecycle_service()

    response = service.get_lifecycle(
        start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1),
        movement_range="5d", now=NOW,
    )

    assert response.movement_granularity == "day"
    assert [point.period_start for point in response.movement_segments] == [
        YESTERDAY - timedelta(days=1), YESTERDAY
    ]
    assert response.movement_segments[0].segment_counts["growing"] == 2
    assert response.movement_segments[0].data_quality == "partial"
    assert response.movement_segments[1].segment_counts["core"] == 1
    assert {(t.from_segment, t.to_segment, t.users, t.data_quality) for t in response.transitions} == {
        ("growing", "core", 1, "complete"),
        ("growing", "at_risk", 1, "partial"),
    }
    assert response.availability["history"].status == "partial"
    assert response.availability["history"].coverage_start == YESTERDAY - timedelta(days=1)


def test_lifecycle_history_falls_back_to_rollups_for_days_without_facts():
    old_day = YESTERDAY - timedelta(days=3)
    rollups = [
        DailyRollup(rollup_date=old_day, metric_name="lifecycle_segment_count", user_state="core", outcome="complete", value_count=7, updated_at=NOW),
        DailyRollup(rollup_date=old_day, metric_name="lifecycle_transition", event_name="onboarding", user_state="growing", outcome="complete", value_count=2, updated_at=NOW),
    ]
    service = _lifecycle_service(rollups=rollups)

    response = service.get_lifecycle(
        start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1), now=NOW
    )

    assert any(
        point.period_start == old_day and point.segment_counts["core"] == 7
        for point in response.movement_segments
    )
    assert any(t.from_segment == "onboarding" and t.to_segment == "growing" and t.users == 2 for t in response.transitions)

    internal = service.get_lifecycle(
        start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1),
        include_internal=True, now=NOW,
    )
    assert not any(point.period_start == old_day for point in internal.movement_segments)


def test_lifecycle_reads_no_raw_events():
    service = _lifecycle_service()
    # FakeRollups.list_events raises; a passing call proves no raw read happened.
    service.get_lifecycle(start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1), now=NOW)
```

Add `from dashboard.backend.domain.analytics.rollups import DailyRollup` to the test imports.

- [ ] **Step 2: Run them and confirm the expected failure**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_value_queries.py -v -k lifecycle
```

Expected: **FAIL** — `AttributeError: 'FakeValueStore' object has no attribute 'list_current_snapshots'` from the old `_current` path.

- [ ] **Step 3: Rewrite `_history` and `get_lifecycle`; delete `_daily`**

Replace `_history` (562-725) with:

```python
    def _history(
        self,
        *,
        start: date,
        end: date,
        use_anonymous_rollups: bool,
        movement_start: date | None = None,
        movement_range: str = "5d",
    ) -> tuple[
        list[WeeklyLifecycleCount],
        list[LifecycleMovementPoint],
        list[LifecycleTransition],
        SectionAvailability,
    ]:
        if movement_range not in _MOVEMENT_WINDOWS:
            raise ValueError("unsupported lifecycle movement range")
        window_days, granularity = _MOVEMENT_WINDOWS[movement_range]
        selected_movement_start = movement_start or end - timedelta(days=window_days)
        if selected_movement_start >= end:
            raise ValueError("lifecycle movement range is empty")
        history_start = max(
            min(start, selected_movement_start),
            end - timedelta(days=_MAX_HISTORY_SCAN_DAYS),
        )
        segment_rows = self.value_store.count_segments_by_date(start=history_start, end=end)
        daily_counts: dict[date, dict[LifecycleSegment, int]] = {}
        daily_quality: dict[date, str] = {}
        for row in segment_rows:
            counts = daily_counts.setdefault(
                row.snapshot_date, {segment: 0 for segment in _LIFECYCLE_SEGMENTS}
            )
            counts[row.lifecycle_segment] = row.users
            if row.partial_rows:
                daily_quality[row.snapshot_date] = "partial"
            else:
                daily_quality.setdefault(row.snapshot_date, "complete")
        direct_dates = set(daily_counts)

        rollups = (
            self.query_store.rollups.list_rollups(start=history_start, end=end)
            if use_anonymous_rollups
            else []
        )
        rollup_counts: dict[date, dict[LifecycleSegment, int]] = defaultdict(dict)
        rollup_quality: dict[date, str] = {}
        for row in rollups:
            if row.metric_name != "lifecycle_segment_count":
                continue
            rollup_counts[row.rollup_date][row.user_state] = row.value_count
            if row.outcome == "partial":
                rollup_quality[row.rollup_date] = "partial"
            else:
                rollup_quality.setdefault(row.rollup_date, "complete")
        for day, counts in rollup_counts.items():
            if day not in direct_dates:
                daily_counts[day] = {
                    segment: int(counts.get(segment, 0)) for segment in _LIFECYCLE_SEGMENTS
                }
                daily_quality[day] = rollup_quality.get(day, "partial")

        weekly: list[WeeklyLifecycleCount] = []
        by_week: dict[date, list[date]] = defaultdict(list)
        for day in daily_counts:
            if start <= day < end:
                by_week[_week_start(day)].append(day)
        for week, dates in sorted(by_week.items()):
            latest = max(dates)
            weekly.append(
                WeeklyLifecycleCount(
                    week_start=week,
                    segment_counts=daily_counts[latest],
                    data_quality=daily_quality[latest],
                )
            )

        movement: list[LifecycleMovementPoint] = []
        by_period: dict[date, list[date]] = defaultdict(list)
        for day in daily_counts:
            if selected_movement_start <= day < end:
                by_period[_period_start(day, granularity)].append(day)
        for period, dates in sorted(by_period.items()):
            latest = max(dates)
            movement.append(
                LifecycleMovementPoint(
                    period_start=max(period, selected_movement_start),
                    segment_counts=daily_counts[latest],
                    data_quality=daily_quality[latest],
                )
            )

        transition_counts: Counter[tuple[str, str]] = Counter()
        transition_partial: set[tuple[str, str]] = set()
        for row in self.value_store.count_transitions(start=start, end=end):
            key = (row.from_segment, row.to_segment)
            transition_counts[key] += row.users
            if row.partial_rows:
                transition_partial.add(key)
        for row in rollups:
            if (
                row.metric_name != "lifecycle_transition"
                or not start <= row.rollup_date < end
                or row.rollup_date in direct_dates
            ):
                continue
            key = (row.event_name, row.user_state)
            transition_counts[key] += row.value_count
            if row.outcome == "partial":
                transition_partial.add(key)
        transitions = [
            LifecycleTransition(
                from_segment=from_segment,
                to_segment=to_segment,
                users=count,
                period_start=start,
                period_end=end,
                data_quality=(
                    "partial" if (from_segment, to_segment) in transition_partial else "complete"
                ),
            )
            for (from_segment, to_segment), count in sorted(
                transition_counts.items(), key=lambda item: (-item[1], item[0])
            )
        ]
        coverage = sorted(day for day in daily_counts if start <= day < end)
        if not coverage:
            availability = SectionAvailability(available=False, status="building")
        else:
            partial = (
                any(daily_quality[day] == "partial" for day in coverage)
                or coverage[0] > start
                or coverage[-1] < end - timedelta(days=1)
            )
            availability = SectionAvailability(
                available=True,
                status="partial" if partial else "ready",
                coverage_start=coverage[0],
                coverage_end=coverage[-1],
            )
        return weekly, movement, transitions, availability
```

Replace `get_lifecycle` (727-802) with:

```python
    def get_lifecycle(
        self,
        *,
        start: date,
        end: date,
        include_internal: bool = False,
        movement_range: str = "5d",
        now: datetime | None = None,
    ) -> LifecycleAnalyticsResponse:
        start, end = _validate_dates(start, end)
        if movement_range not in _MOVEMENT_WINDOWS:
            raise ValueError("unsupported lifecycle movement range")
        window_days, _granularity = _MOVEMENT_WINDOWS[movement_range]
        movement_start = end - timedelta(days=window_days)
        current_time = _utc(now or datetime.now(UTC), "now")
        yesterday = current_time.date() - timedelta(days=1)
        cells = self.value_store.count_states_for_date(yesterday)
        segment_counts = {
            segment: sum(cell.users for cell in cells if cell.lifecycle_segment == segment)
            for segment in _LIFECYCLE_SEGMENTS
        }
        availability: dict[str, SectionAvailability] = {
            "current": SectionAvailability(
                available=True,
                status="ready" if cells else "building",
                coverage_start=yesterday if cells else None,
                coverage_end=yesterday if cells else None,
            )
        }
        try:
            activated_users = self.value_store.count_activated_users(
                include_internal=include_internal
            )
        except Exception:
            activated_users = 0
            availability["current"] = SectionAvailability(available=bool(cells), status="partial")
        try:
            users = self._eligible_users(include_internal=include_internal)
            commercial = self._commercial(users, start=start, end=end)
            paid = sum(fact.lifetime_net_purchased_micro > 0 for fact in commercial.values())
            availability["commercial"] = SectionAvailability(available=True, status="ready")
        except Exception:
            paid = 0
            availability["commercial"] = SectionAvailability(available=False, status="unavailable")
        try:
            weekly, movement, transitions, history_availability = self._history(
                start=start,
                end=end,
                use_anonymous_rollups=not include_internal,
                movement_start=movement_start,
                movement_range=movement_range,
            )
        except Exception:
            weekly, movement, transitions = [], [], []
            history_availability = SectionAvailability(available=False, status="unavailable")
        availability["history"] = history_availability
        return LifecycleAnalyticsResponse(
            as_of=current_time,
            headline=LifecycleHeadline(
                activated_users=activated_users,
                core_users=segment_counts["core"],
                at_risk_users=segment_counts["at_risk"],
                paid_users=paid,
            ),
            segment_counts=segment_counts,
            weekly_segments=weekly,
            movement_range=movement_range,
            movement_granularity=_MOVEMENT_WINDOWS[movement_range][1],
            movement_segments=movement,
            transitions=transitions,
            availability=availability,
        )
```

Delete `_daily` (544-560). `get_retention` still calls it until Task 7 — do Task 7 immediately after, or leave `_daily` in place until Task 7 removes it; the plan's commit boundary is per task, so **keep `_daily` here and delete it in Task 7**. Remove `UserLifecycleDailySnapshot` from the `value_repository` import once nothing in the module references it (Task 7).

- [ ] **Step 4: Run and confirm the pass**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_value_queries.py -v -k lifecycle
python -m pytest dashboard/backend/tests/test_admin_analytics_api.py -q
```

Expected: **PASS** for the lifecycle cases; the API tests pass because the router's value service is the `FixtureValueQueryService` fake. The remaining `test_value_queries.py` cases that still build `FakeValueStore(snapshots=...)` fail until their own tasks (7-11) — run only `-k lifecycle` here, and the whole file at the end of Task 11.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/analytics/value_queries.py dashboard/backend/tests/domain/analytics/test_value_queries.py
git status --short
git commit -m "$(cat <<'EOF'
feat: serve the lifecycle board from daily facts and the transition log

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: `/retention` from `user_activity` cohorts and `user_daily_facts.active`

**Files:**
- Modify: `dashboard/backend/domain/analytics/value_queries.py` — `get_retention` (lines 846-973); delete `_daily` (544-560) and `_retention_quality` (804-823)
- Test: `dashboard/backend/tests/domain/analytics/test_value_queries.py` — replace `test_retention_uses_nulls_for_immature_cells_and_weighted_mature_summary` (375) and `test_missing_or_partial_retention_history_propagates_partial_quality` (423)

**Interfaces:**
- Consumes: `list_activations`, `list_active_dates`, `count_segments_by_date` (Task 2); `activation_cohort_week` (`lifecycle.py:367`).
- Produces: `get_retention` with the same signature and `RetentionAnalyticsResponse`. The raw-event scan at lines 872-880 and the `list_credit_activity` calls at 881-889 are gone: credit activity already reaches `user_daily_facts.active` through the daily job's ledger step (§6.5, §6.9 step 4), so a purchase day is an active day without a second source.

§6.10: *`user_daily_facts.active` grouped by activation week.* Cohort membership is `activation_cohort_week(user_activity.activated_at)` — the same Monday `_week_start(snapshot.activated_at.date())` computed today (`:863`), now read from the activity row instead of the snapshot. A cell is retained when the member has any active fact date inside the target week. Cell quality no longer inspects one row per member per day (`_retention_quality`, which needed the daily snapshot table): a target week is `partial` when any of its seven days has no fact rows at all or carries a `partial` row — population-level, from the one `count_segments_by_date` read the grid already needs. An immature cell stays `None`, never zero (§15.5).

- [ ] **Step 1: Write the failing tests**

Delete the two retention tests named above and add:

```python
def _retention_service():
    # Cohort week Monday 2026-08-03: users 1 and 2 activated that week; user 3 in the next.
    activity = {
        1: _activity(1, activated_days_ago=(NOW.date() - date(2026, 8, 4)).days),
        2: _activity(2, activated_days_ago=(NOW.date() - date(2026, 8, 7)).days),
        3: _activity(3, activated_days_ago=(NOW.date() - date(2026, 8, 12)).days),
        9: _activity(9, activated_days_ago=(NOW.date() - date(2026, 8, 5)).days),
    }
    facts = []
    # Every day from the cohort start through yesterday has at least one row, so no week is partial for lack of data.
    day = date(2026, 8, 3)
    while day <= YESTERDAY:
        facts.append(_fact(3, day, active=False, runs_completed=0))
        day += timedelta(days=1)
    # Week 1 (2026-08-10..16): user 1 active, user 2 not. Week 2 (08-17..23): both. Week 4 (08-31..09-06): user 2 only.
    facts += [
        _fact(1, date(2026, 8, 11)),
        _fact(1, date(2026, 8, 18)),
        _fact(2, date(2026, 8, 20)),
        _fact(2, date(2026, 9, 1)),
    ]
    return _service(
        facts=facts, activity=activity, internal_ids={9},
        users=((1, "organic"), (2, "organic"), (3, "organic"), (9, "internal")),
    )


def test_retention_cells_come_from_activation_weeks_and_active_fact_days():
    service = _retention_service()

    response = service.get_retention(start=date(2026, 8, 1), end=date(2026, 8, 20), now=NOW)

    first = next(c for c in response.cohorts if c.cohort_week == date(2026, 8, 3))
    assert first.activated_users == 2  # user 9 is internal
    assert (first.week_1.retained_users, first.week_1.eligible_users, first.week_1.rate) == (1, 2, 0.5)
    assert (first.week_2.retained_users, first.week_2.eligible_users) == (2, 2)
    assert (first.week_4.retained_users, first.week_4.eligible_users, first.week_4.rate) == (1, 2, 0.5)
    second = next(c for c in response.cohorts if c.cohort_week == date(2026, 8, 10))
    assert second.activated_users == 1
    assert second.week_1.retained_users == 0
    assert response.summary_week_1.rate == 1 / 3
    assert response.availability.status == "ready"


def test_retention_marks_immature_weeks_unavailable_not_zero():
    service = _retention_service()
    now = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)

    response = service.get_retention(start=date(2026, 8, 1), end=date(2026, 8, 20), now=now)

    first = next(c for c in response.cohorts if c.cohort_week == date(2026, 8, 3))
    assert first.week_2.mature is True
    assert first.week_4.mature is False
    assert first.week_4.retained_users is None and first.week_4.rate is None
    assert response.summary_week_4.mature is False


def test_retention_week_with_a_missing_or_partial_fact_day_is_partial():
    service = _retention_service()
    service.value_store.facts = [
        row for row in service.value_store.facts if row.snapshot_date != date(2026, 8, 12)
    ]
    service.value_store.facts.append(_fact(3, date(2026, 8, 19), active=False, data_quality="partial"))

    response = service.get_retention(start=date(2026, 8, 1), end=date(2026, 8, 20), now=NOW)

    first = next(c for c in response.cohorts if c.cohort_week == date(2026, 8, 3))
    assert first.week_1.data_quality == "partial"   # 08-12 has no rows
    assert first.week_2.data_quality == "partial"   # 08-19 carries a partial row
    assert first.week_4.data_quality == "complete"
    assert response.availability.status == "partial"


def test_retention_reads_no_raw_events():
    _retention_service().get_retention(start=date(2026, 8, 1), end=date(2026, 8, 20), now=NOW)
```

- [ ] **Step 2: Run them and confirm the expected failure**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_value_queries.py -v -k retention
```

Expected: **FAIL** — `AttributeError: 'FakeValueStore' object has no attribute 'list_current_snapshots'`.

- [ ] **Step 3: Rewrite `get_retention`**

Replace lines 846-973 with:

```python
    def get_retention(
        self,
        *,
        start: date,
        end: date,
        include_internal: bool = False,
        now: datetime | None = None,
    ) -> RetentionAnalyticsResponse:
        start, end = _validate_dates(start, end)
        current_time = _utc(now or datetime.now(UTC), "now")
        cohort_start = _week_start(start)
        activations = self.value_store.list_activations(
            start=_day_start(cohort_start),
            end=_day_start(end),
            include_internal=include_internal,
        )
        cohort_users: dict[date, list[int]] = defaultdict(list)
        for user_id, activated_at in activations.items():
            week = activation_cohort_week(activated_at)
            if cohort_start <= week < end:
                cohort_users[week].append(user_id)
        ids = sorted({user_id for members in cohort_users.values() for user_id in members})
        grid_end = min(end + timedelta(days=35), current_time.date() + timedelta(days=1))
        active_dates: dict[int, list[date]] = {}
        for offset in range(0, len(ids), 500):
            active_dates.update(
                self.value_store.list_active_dates(
                    ids[offset : offset + 500], start=cohort_start, end=grid_end
                )
            )
        day_quality: dict[date, str] = {}
        if ids:
            for row in self.value_store.count_segments_by_date(start=cohort_start, end=grid_end):
                if row.partial_rows:
                    day_quality[row.snapshot_date] = "partial"
                else:
                    day_quality.setdefault(row.snapshot_date, "complete")

        def week_quality(target_start: date, target_end: date) -> Literal["complete", "partial"]:
            day = target_start
            while day < target_end:
                if day_quality.get(day) != "complete":
                    return "partial"
                day += timedelta(days=1)
            return "complete"

        cohorts: list[RetentionCohort] = []
        all_cells: dict[int, list[RetentionCell]] = {1: [], 2: [], 4: []}
        for cohort_week, members in sorted(cohort_users.items()):
            cells: dict[int, RetentionCell] = {}
            for week in (1, 2, 4):
                target_start = cohort_week + timedelta(days=week * 7)
                target_end = target_start + timedelta(days=7)
                mature = current_time >= _day_start(target_end)
                retained = eligible = rate = None
                if mature:
                    eligible = len(members)
                    retained = sum(
                        any(target_start <= day < target_end for day in active_dates.get(user_id, ()))
                        for user_id in members
                    )
                    rate = retained / eligible if eligible else None
                cell = RetentionCell(
                    week=week,
                    mature=mature,
                    retained_users=retained,
                    eligible_users=eligible,
                    rate=rate,
                    data_quality=week_quality(target_start, target_end),
                )
                cells[week] = cell
                all_cells[week].append(cell)
            cohorts.append(
                RetentionCohort(
                    cohort_week=cohort_week,
                    activated_users=len(members),
                    week_1=cells[1],
                    week_2=cells[2],
                    week_4=cells[4],
                )
            )
        partial = any(
            cell.data_quality == "partial"
            for cells in all_cells.values()
            for cell in cells
            if cell.mature
        )
        return RetentionAnalyticsResponse(
            as_of=current_time,
            cohorts=cohorts,
            summary_week_1=self._summary_cell(1, all_cells[1]),
            summary_week_2=self._summary_cell(2, all_cells[2]),
            summary_week_4=self._summary_cell(4, all_cells[4]),
            availability=SectionAvailability(
                available=True,
                status="partial" if partial else "ready",
                coverage_start=cohort_start if cohorts else None,
                coverage_end=end - timedelta(days=1) if cohorts else None,
            ),
        )
```

Add `activation_cohort_week` to the `.lifecycle` import (line 11-18) and drop `is_lifecycle_activity` from it (no longer used). Delete `_daily` (544-560) and `_retention_quality` (804-823); remove `UserLifecycleDailySnapshot` from the `value_repository` import (line 28-33) — after this task nothing in `value_queries.py` names it.

- [ ] **Step 4: Run and confirm the pass**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_value_queries.py -v -k "retention or lifecycle"
```

Expected: **PASS**.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/analytics/value_queries.py dashboard/backend/tests/domain/analytics/test_value_queries.py
git status --short
git commit -m "$(cat <<'EOF'
feat: build the retention grid from activation weeks and daily facts

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: `/operational` from yesterday's facts, and `top_operational_reasons`

**Files:**
- Modify: `dashboard/backend/domain/analytics/value_queries.py` — `get_operational` (lines 1033-1088); add `OperationalReasonCount` beside `OperationalAnalyticsResponse` (264-276) and `top_operational_reasons` after `get_operational`
- Test: `dashboard/backend/tests/domain/analytics/test_value_queries.py` — replace `test_missing_operational_subsection_is_reported_as_partial` (647)

**Interfaces:**
- Consumes: `count_states_for_date` (Task 2); `legacy_service.get_overview` (Task 5 — now rollups + one-day scan, so delegating to it for run counts, tokens, cost and `top_failure_categories` is what D13 asks for: `top_failure_categories` from rollups `error_category`); `operational_reason` (Task 1).
- Produces:
  - `get_operational` — same signature, same `OperationalAnalyticsResponse`; `operational_state_counts` from yesterday's fact rows.
  - `OperationalReasonCount(reason_code: str, reason: str, users: int)` — a frozen model in `value_queries.py`, **not** referenced by any response model.
  - `ValueAnalyticsQueryService.top_operational_reasons(*, day: date, limit: int = 10) -> list[OperationalReasonCount]` — reasons behind `blocked`/`needs_attention` on `day`, most users first (§9 "Added"; PR D exposes it).

- [ ] **Step 1: Write the failing tests**

Delete `test_missing_operational_subsection_is_reported_as_partial` and add:

```python
def _operational_service(**overrides):
    facts = [
        _fact(1, YESTERDAY, operational_state="blocked", operational_reason_code="billing_lane_unavailable"),
        _fact(2, YESTERDAY, operational_state="blocked", operational_reason_code="billing_lane_unavailable"),
        _fact(3, YESTERDAY, operational_state="needs_attention", operational_reason_code="invalid_default_credential"),
        _fact(4, YESTERDAY),
        _fact(5, YESTERDAY - timedelta(days=1), operational_state="blocked", operational_reason_code="account_restricted"),
    ]
    values = dict(facts=facts, users=((1, "unknown"), (2, "unknown"), (3, "unknown"), (4, "unknown")))
    values.update(overrides)
    return _service(**values)


def test_operational_state_counts_are_yesterdays_facts():
    service = _operational_service()

    response = service.get_operational(
        start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1), now=NOW
    )

    assert response.operational_state_counts == {"blocked": 2, "needs_attention": 1, "healthy": 1}
    assert response.completed_runs == 9  # FakeLegacyService
    assert response.availability.status == "ready"


def test_operational_is_partial_when_the_overview_growth_panel_is_down():
    service = _operational_service(legacy_availability={"growth": False, "friction": True})

    response = service.get_operational(
        start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1), now=NOW
    )

    assert response.availability.available is True
    assert response.availability.status == "partial"


def test_operational_is_building_before_the_first_fact_day():
    service = _operational_service(facts=[])

    response = service.get_operational(
        start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1), now=NOW
    )

    assert response.operational_state_counts == {"blocked": 0, "needs_attention": 0, "healthy": 0}
    assert response.availability.status == "building"


def test_top_operational_reasons_rank_blocking_reasons_by_users():
    service = _operational_service()

    reasons = service.top_operational_reasons(day=YESTERDAY)

    assert [(r.reason_code, r.users) for r in reasons] == [
        ("billing_lane_unavailable", 2),
        ("invalid_default_credential", 1),
    ]
    assert reasons[0].reason == "No usable model billing lane is available."
    assert service.top_operational_reasons(day=YESTERDAY, limit=1)[0].reason_code == "billing_lane_unavailable"
    assert "top_operational_reasons" not in OperationalAnalyticsResponse.model_fields
```

Add `OperationalAnalyticsResponse` to the test's `value_queries` import.

- [ ] **Step 2: Run them and confirm the expected failure**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_value_queries.py -v -k operational
```

Expected: **FAIL** — `AttributeError: ... 'list_current_snapshots'` for the first three; `AttributeError: ... 'top_operational_reasons'` for the last.

- [ ] **Step 3: Implement**

Add after `OperationalAnalyticsResponse` (line 276):

```python
class OperationalReasonCount(BaseModel):
    """One blocking or attention reason and how many users carried it.

    Service-level only in PR B (design §9: PR D puts `top_operational_reasons`
    on OperationalAnalyticsResponse in the contract re-cut).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason_code: str
    reason: str
    users: int = Field(ge=0)
```

Replace `get_operational` (1033-1088) with:

```python
    def get_operational(
        self,
        *,
        start: date,
        end: date,
        include_internal: bool = False,
        provider_id: str | None = None,
        model_id: str | None = None,
        billing_mode: str | None = None,
        now: datetime | None = None,
    ) -> OperationalAnalyticsResponse:
        start, end = _validate_dates(start, end)
        current_time = _utc(now or datetime.now(UTC), "now")
        yesterday = current_time.date() - timedelta(days=1)
        cells = self.value_store.count_states_for_date(yesterday)
        counts = {
            state: sum(cell.users for cell in cells if cell.operational_state == state)
            for state in _OPERATIONAL_STATES
        }
        overview = self.legacy_service.get_overview(
            filters=AnalyticsMetricFilters(
                start=_day_start(start),
                end=_day_start(end),
                include_internal=include_internal,
                provider_id=provider_id,
                model_id=model_id,
                billing_mode=billing_mode,
            ),
            now=current_time,
        )
        growth_available = overview.availability["growth"].available
        friction_available = overview.availability["friction"].available
        available = bool(cells) or growth_available or friction_available
        if not cells:
            status = "building"
        elif growth_available and friction_available:
            status = "ready"
        else:
            status = "partial" if available else "unavailable"
        return OperationalAnalyticsResponse(
            as_of=current_time,
            operational_state_counts=counts,
            backtest_success_rate=overview.backtest_success_rate,
            completed_runs=overview.completed_runs or 0,
            failed_runs=overview.failed_runs or 0,
            input_tokens=overview.input_tokens or 0,
            output_tokens=overview.output_tokens or 0,
            platform_model_cost_micro_usd=round(
                (overview.platform_model_cost_usd or 0) * 1_000_000
            ),
            top_failure_categories=overview.top_failure_categories,
            availability=SectionAvailability(available=available, status=status),
        )

    def top_operational_reasons(
        self, *, day: date, limit: int = 10
    ) -> list[OperationalReasonCount]:
        """Why users were blocked or needed attention on ``day`` (design §9).

        One grouped read over the fact rows; the reason text is the rule
        table's copy, never the stored code alone.
        """
        page_size = positive_limit(limit)
        totals: Counter[str] = Counter()
        for cell in self.value_store.count_states_for_date(day):
            if cell.operational_state in {"blocked", "needs_attention"}:
                totals[cell.operational_reason_code] += cell.users
        return [
            OperationalReasonCount(
                reason_code=code, reason=operational_reason(code), users=users
            )
            for code, users in sorted(totals.items(), key=lambda item: (-item[1], item[0]))[
                :page_size
            ]
        ]
```

Add `operational_reason` to the `.lifecycle` import and `OperationalReasonCount` to `__all__`.

- [ ] **Step 4: Run and confirm the pass**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_value_queries.py -v -k "operational or lifecycle or retention"
```

Expected: **PASS**.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/analytics/value_queries.py dashboard/backend/tests/domain/analytics/test_value_queries.py
git status --short
git commit -m "$(cat <<'EOF'
feat: serve operational state counts and top reasons from daily facts

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: `/groups` — run counts and operator cost from facts grouped by `user_group`

**Files:**
- Modify: `dashboard/backend/domain/analytics/value_queries.py` — `get_groups` (lines 1090-1251); delete `_safe_cost_micro_usd` (456-475) and `_event_utc` (446-453)
- Test: `dashboard/backend/tests/domain/analytics/test_value_queries.py` — rewrite `test_group_summary_has_six_rows_zeroes_and_fixed_order` (502) and `test_group_summary_cost_excludes_byok_and_paid_uses_lifetime_credits` (546)

**Interfaces:**
- Consumes: `sum_facts_by_user_group` (Task 2); `_eligible_users` + `_commercial` (unchanged: user counts from `users`, `paid_users` from the ledger).
- Produces: `get_groups` — same signature, same `GroupAnalyticsResponse`. `total_runs` = Σ `runs_requested`; `atl_cost_micro_usd` = Σ `operator_cost_micro` (§6.7: operator-funded `agent_runs.est_cost_usd` in micro-USD — the same concept the `model_usage_recorded` cost summed today, from its authoritative source); `successful_run_users` and `repeat_users` (completed on ≥ 2 distinct days) from the same grouped read. The raw scan at lines 1155-1198 is gone.

`user_group` on a fact row is the value at write time (§6.7), so a user moved between groups mid-window contributes each day's row to the group they were in that day; the `users` column keeps coming from the account row, as today. With `user_group` selected, every other group's fact totals are zeroed, mirroring today's `user_rows` filter (`:1134-1140`), which restricted the event loop to that group's members.

- [ ] **Step 1: Write the failing tests**

Replace the two group tests with:

```python
def _groups_service(**overrides):
    facts = [
        _fact(1, YESTERDAY, user_group="organic", runs_requested=3, runs_completed=2, operator_cost_micro=200_000),
        _fact(1, YESTERDAY - timedelta(days=1), user_group="organic", runs_requested=1, runs_completed=1, operator_cost_micro=50_000),
        _fact(2, YESTERDAY, user_group="organic", runs_requested=2, runs_completed=0, runs_failed=2),
        _fact(3, YESTERDAY, user_group="partner", runs_requested=1, runs_completed=1),
        _fact(4, YESTERDAY - timedelta(days=20), user_group="partner", runs_requested=9, runs_completed=9),
    ]
    values = dict(
        facts=facts,
        users=((1, "organic"), (2, "organic"), (3, "partner"), (4, "partner"), (5, "unknown")),
        commercial={1: _commercial(1, 6_000_000), 2: _commercial(2), 3: _commercial(3, 1), 4: _commercial(4), 5: _commercial(5)},
    )
    values.update(overrides)
    return _service(**values)


def test_group_summary_has_six_rows_in_fixed_order_with_facts_totals():
    service = _groups_service()

    response = service.get_groups(
        start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1), now=NOW
    )

    assert [row.group for row in response.groups] == list(USER_GROUPS)
    by_group = {row.group: row for row in response.groups}
    assert (by_group["organic"].users, by_group["organic"].total_runs) == (2, 6)
    assert by_group["organic"].atl_cost_micro_usd == 250_000
    assert by_group["organic"].successful_run_users == 1
    assert by_group["organic"].repeat_users == 1
    assert by_group["organic"].paid_users == 1
    assert (by_group["partner"].users, by_group["partner"].total_runs) == (2, 1)  # user 4's row is outside the window
    assert by_group["partner"].paid_users == 1
    assert by_group["unknown"].users == 1
    assert by_group["internal"] == UserGroupSummary(
        group="internal", label="Internal", users=0, successful_run_users=0,
        repeat_users=0, total_runs=0, atl_cost_micro_usd=0, paid_users=0,
    )
    assert response.availability.status == "ready"
    assert response.availability.coverage_end == YESTERDAY


def test_group_filter_zeroes_every_other_group():
    service = _groups_service()

    response = service.get_groups(
        start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1),
        user_group="partner", now=NOW,
    )

    by_group = {row.group: row for row in response.groups}
    assert response.selected_user_group == "partner"
    assert by_group["partner"].total_runs == 1
    assert by_group["organic"].total_runs == 0
    assert by_group["organic"].users == 0
    assert by_group["organic"].paid_users == 0


def test_group_summary_is_partial_when_the_fact_read_fails():
    service = _groups_service()

    def explode(**_kwargs):
        raise RuntimeError("private detail")

    service.value_store.sum_facts_by_user_group = explode

    response = service.get_groups(
        start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1), now=NOW
    )

    assert response.availability.available is True
    assert response.availability.status == "partial"
    assert all(row.total_runs == 0 for row in response.groups)
```

Add `from dashboard.backend.domain.analytics.value_queries import UserGroupSummary` and `from dashboard.backend.domain.user_groups import USER_GROUPS` to the test imports if absent.

- [ ] **Step 2: Run them and confirm the expected failure**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_value_queries.py -v -k group
```

Expected: **FAIL** — `AssertionError` from `FakeRollups.list_events` ("value queries must not read raw events after PR B"), because `get_groups` falls through to `self.query_store.rollups.list_events` when `list_metric_events` is absent on the fake.

- [ ] **Step 3: Rewrite `get_groups`**

Replace lines 1145-1198 (from `requested_by_group: Counter[UserGroup] = Counter()` through the end of the `for event in events:` loop) with:

```python
        facts_available = True
        totals: dict[str, GroupFactTotals] = {}
        try:
            totals = self.value_store.sum_facts_by_user_group(start=start, end=end)
        except Exception:
            facts_available = False
        if selected_group is not None:
            totals = {group: value for group, value in totals.items() if group == selected_group}
```

and in the row-building loop (`for group in USER_GROUPS:`, lines 1216-1234) replace the body with:

```python
        for group in USER_GROUPS:
            group_totals = totals.get(group, GroupFactTotals())
            rows.append(
                UserGroupSummary(
                    group=group,
                    label=USER_GROUP_LABELS[group],
                    users=int(users_by_group[group]),
                    successful_run_users=group_totals.successful_run_users,
                    repeat_users=group_totals.repeat_users,
                    total_runs=group_totals.total_runs,
                    atl_cost_micro_usd=group_totals.operator_cost_micro,
                    paid_users=int(paid_by_group[group]),
                )
            )
```

and change `complete = events_available and commercial_available` (line 1239) to `complete = facts_available and commercial_available`. Delete the now-unused `_event_utc` (446-453) and `_safe_cost_micro_usd` (456-475), the `period_start`/`period_end` locals, and import `GroupFactTotals` from `.value_repository`. Everything above the deleted block (`users_available`, `group_by_user`, `user_rows`, `users_by_group`) and the `_commercial`/`paid_by_group` block after it stay exactly as they are.

- [ ] **Step 4: Run and confirm the pass**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_value_queries.py -v -k "group or operational or lifecycle or retention"
python -m pytest dashboard/backend/tests/test_admin_analytics_api.py -q -k group
```

Expected: **PASS**.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/analytics/value_queries.py dashboard/backend/tests/domain/analytics/test_value_queries.py
git status --short
git commit -m "$(cat <<'EOF'
feat: take group run counts and operator cost from daily facts

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: `/users` — the service reads the SQL page and computes live segments for one page

**Files:**
- Modify: `dashboard/backend/domain/analytics/value_queries.py` — `list_users` (lines 1253-1349); `_lifecycle`, `_operational`, `_priority_group` (398-432) are rewritten to take the new inputs
- Test: `dashboard/backend/tests/domain/analytics/test_value_queries.py` — rewrite `test_priority_order_is_group_value_inactivity_then_user_id` (448), `test_user_list_uses_injected_utc_day_for_commercial_window` (484), `test_user_group_filter_applies_to_priority_users` (584), `test_internal_accounts_are_excluded_unless_explicitly_included` (718)

**Interfaces:**
- Consumes: `list_users_page` (Task 3), `sum_recent_facts` + `build_lifecycle_inputs` + `calculate_lifecycle` (PR A; live segment per page row, `as_of=now`), `operational_result_from_code` (Task 1), `list_commercial_values` (ledger, for `lifetime_net_purchased_micro` on the page's ids only).
- Produces: `list_users` — same signature, same `PaginatedValueUsers`. Per page: one SQL page read, one count, one `sum_recent_facts` batch for the page's ids, one `list_commercial_values` batch (≤100 ids), zero raw-event reads. `_priority_group(operational_state, lifecycle_segment)` becomes a pure function of the two yesterday's-fact values (the same rule as `_users_page_sql`'s priority predicate and sort, so a row's `priority_group` agrees with where SQL placed it).

Value notes (D18: one suspect per moved number): `lifecycle` is live (today) while `operational`, `commercial_tier` and `priority_group` are yesterday's fact values — the profile shows the live operational state; a list of hundreds cannot make the per-user operational calls (§6.3, §6.10). `lifetime_net_purchased_micro` still comes from the ledger for the page, so `commercial_tier` (facts, yesterday) and that number (ledger, now) can disagree for a user who paid today; the design accepts a one-day lag on every daily figure.

- [ ] **Step 1: Write the failing tests**

Replace the four tests named above with:

```python
def _page_row(user_id: int, **overrides) -> UserPageRow:
    values = dict(
        user_id=user_id, email=f"value-{user_id}@example.test", display_name=f"Value User {user_id}",
        role="user", user_group="unknown", created_at=NOW - timedelta(days=90),
        activated_at=NOW - timedelta(days=60), last_meaningful_activity_at=NOW - timedelta(days=2),
        lifecycle_segment="core", operational_state="healthy",
        operational_reason_code="no_supported_issue", tier="unpaid", data_quality="complete",
    )
    values.update(overrides)
    return UserPageRow(**values)


def test_users_list_reads_one_sql_page_and_computes_live_segments():
    rows = [
        _page_row(1, operational_state="blocked", operational_reason_code="account_restricted", tier="invested"),
        _page_row(2, lifecycle_segment="growing", last_meaningful_activity_at=NOW - timedelta(days=9)),
    ]
    facts = [_fact(1, YESTERDAY - timedelta(days=offset)) for offset in range(3)]
    activity = {1: _activity(1), 2: _activity(2, active_days_ago=9)}
    service = _service(
        page_rows=rows, facts=facts, activity=activity,
        commercial={1: _commercial(1, 6_000_000), 2: _commercial(2)},
    )

    response = service.list_users(filters=UserValueFilters(priority=True), limit=25, offset=0, now=NOW)

    assert response.total == 2
    first, second = response.items
    assert first.user_id == 1
    assert first.lifecycle.segment == "core"           # live: 3 active days, 3 successes
    assert first.operational.state == "blocked"        # yesterday's fact row
    assert first.operational.reason == "The Credits account is restricted from model spending."
    assert first.commercial_tier == "invested"
    assert first.lifetime_net_purchased_micro == 6_000_000
    assert first.priority_group == "blocked"
    assert second.lifecycle.segment == "at_risk"       # live: 9 idle days, no fact rows
    assert second.priority_group == "none"             # yesterday's segment was growing
    query, day, limit, offset = service.value_store.page_calls[0]
    assert (day, limit, offset) == (YESTERDAY, 25, 0)
    assert query.priority is True
    assert service.value_store.recent_windows == [(YESTERDAY - timedelta(days=29), YESTERDAY)]
    assert service.value_store.commercial_windows == [(_day_start(YESTERDAY), _day_start(YESTERDAY + timedelta(days=1)))]


def test_users_list_passes_every_filter_to_sql_unchanged():
    service = _service(page_rows=[], activity={})
    filters = UserValueFilters(
        q="Ada", lifecycle_segment="core", operational_state="healthy", commercial_tier="starter",
        user_group="partner", activated=True, last_meaningful_activity_from=NOW - timedelta(days=3),
        last_meaningful_activity_to=NOW, priority=False, legacy_status="active", include_internal=True,
    )

    response = service.list_users(filters=filters, limit=10, offset=20, now=NOW)

    assert response.total == 0 and response.items == []
    query, _day, limit, offset = service.value_store.page_calls[0]
    assert (limit, offset) == (10, 20)
    assert query == UserPageQuery(
        q="Ada", lifecycle_segment="core", operational_state="healthy", commercial_tier="starter",
        user_group="partner", activated=True, last_meaningful_activity_from=NOW - timedelta(days=3),
        last_meaningful_activity_to=NOW, priority=False, legacy_status="active", include_internal=True,
    )


def test_users_list_row_without_a_fact_row_reads_as_healthy_unpaid():
    service = _service(
        page_rows=[_page_row(7, lifecycle_segment=None, data_quality=None, activated_at=None)],
        activity={}, commercial={},
    )

    response = service.list_users(filters=UserValueFilters(), limit=25, offset=0, now=NOW)

    item = response.items[0]
    assert item.operational.state == "healthy"
    assert item.commercial_tier == "unpaid"
    assert item.lifetime_net_purchased_micro == 0
    assert item.lifecycle.segment == "dormant"  # created 90 days ago, no activity row
    assert item.priority_group == "none"
```

Add `UserPageQuery` and `UserPageRow` to the test's `value_repository` import.

- [ ] **Step 2: Run them and confirm the expected failure**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_value_queries.py -v -k users_list
```

Expected: **FAIL** — `AttributeError: 'FakeValueStore' object has no attribute 'list_current_snapshots'`.

- [ ] **Step 3: Rewrite the helpers and `list_users`**

Replace `_lifecycle`, `_operational`, `_priority_group` (398-432) with:

```python
def _priority_group(operational_state: str, lifecycle_segment: str | None) -> PriorityGroup:
    if operational_state == "blocked":
        return "blocked"
    if operational_state == "needs_attention":
        return "needs_attention"
    if lifecycle_segment == "at_risk":
        return "healthy_at_risk"
    if lifecycle_segment == "onboarding":
        return "healthy_onboarding"
    return "none"
```

(`PriorityGroup` is defined at line 356; move this function below it.) Replace `list_users` (1253-1349) with:

```python
    def list_users(
        self,
        *,
        filters: UserValueFilters,
        limit: int,
        offset: int,
        now: datetime | None = None,
    ) -> PaginatedValueUsers:
        if not isinstance(filters, UserValueFilters):
            filters = UserValueFilters.model_validate(filters)
        page_size = positive_limit(limit)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        current_time = _utc(now or datetime.now(UTC), "now")
        yesterday = current_time.date() - timedelta(days=1)
        query = UserPageQuery(
            q=filters.q,
            lifecycle_segment=filters.lifecycle_segment,
            operational_state=filters.operational_state,
            commercial_tier=filters.commercial_tier,
            user_group=filters.user_group,
            activated=filters.activated,
            last_meaningful_activity_from=filters.last_meaningful_activity_from,
            last_meaningful_activity_to=filters.last_meaningful_activity_to,
            priority=filters.priority,
            legacy_status=filters.legacy_status,
            include_internal=filters.include_internal,
        )
        rows, total = self.value_store.list_users_page(
            query=query, day=yesterday, limit=page_size, offset=offset
        )
        ids = [row.user_id for row in rows]
        totals = (
            self.value_store.sum_recent_facts(
                ids, start=yesterday - timedelta(days=29), end=yesterday
            )
            if ids
            else {}
        )
        commercial = (
            self.value_store.list_commercial_values(
                ids,
                start=_day_start(yesterday),
                end=_day_start(yesterday + timedelta(days=1)),
            )
            if ids
            else {}
        )
        as_of = _day_start(yesterday + timedelta(days=1))
        items: list[ValueUserListItem] = []
        for row in rows:
            activity = (
                UserActivity(
                    user_id=row.user_id,
                    activated_at=row.activated_at,
                    last_meaningful_activity_at=row.last_meaningful_activity_at,
                    updated_at=current_time,
                )
                if row.activated_at is not None or row.last_meaningful_activity_at is not None
                else None
            )
            lifecycle = calculate_lifecycle(
                build_lifecycle_inputs(
                    row.user_id,
                    created_at=row.created_at,
                    activity=activity,
                    totals=totals.get(row.user_id),
                    as_of=current_time,
                ),
                current_time,
            )
            fact = commercial.get(row.user_id)
            items.append(
                ValueUserListItem(
                    user_id=row.user_id,
                    display_name=row.display_name,
                    email=row.email,
                    joined_at=row.created_at,
                    lifecycle=lifecycle,
                    operational=operational_result_from_code(
                        row.operational_reason_code, as_of=as_of
                    ),
                    commercial_tier=row.tier,
                    lifetime_net_purchased_micro=(
                        fact.lifetime_net_purchased_micro if fact is not None else 0
                    ),
                    priority_group=_priority_group(row.operational_state, row.lifecycle_segment),
                    profile_path=f"/admin/analytics/users/{row.user_id}",
                )
            )
        return PaginatedValueUsers(
            items=items, total=total, limit=page_size, offset=offset
        )
```

Imports: `from .lifecycle import calculate_lifecycle, operational_result_from_code` (extend the existing block), `from .lifecycle_reads import build_lifecycle_inputs`, and `UserActivity, UserPageQuery` from `.value_repository`. Delete `_current` (512-522) if nothing else references it — `get_user_profile` still does until Task 11, so leave it for that task. `_TIER_RANK` and `_PRIORITY_RANK` (lines 63, 89-95) become unused; delete both.

- [ ] **Step 4: Run and confirm the pass**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_value_queries.py -v -k "users_list or group or operational or lifecycle or retention"
python -m pytest dashboard/backend/tests/test_admin_analytics_api.py -q -k user_list
```

Expected: **PASS**. `test_admin_user_list_accepts_documented_filters` still passes (`filters.legacy_status == "active"` holds until Task 16).

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/analytics/value_queries.py dashboard/backend/tests/domain/analytics/test_value_queries.py
git status --short
git commit -m "$(cat <<'EOF'
feat: page the analytics users list from SQL with live segments per page

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: The profile — live axes, fact-window sums, one timeline page; `get_user_metrics`

**Files:**
- Modify: `dashboard/backend/domain/analytics/query_service.py` — `get_user_profile` (lines 1071-1200); add `get_user_axes`, `get_user_metrics`, `RunOutcomeCounts`, `UserMetrics`
- Modify: `dashboard/backend/domain/analytics/value_queries.py` — `get_user_profile` (1351-1390); delete `_current` (512-522) and `_lifecycle`/`_operational` if still present
- Test: `dashboard/backend/tests/domain/analytics/test_user_metrics.py` (create)
- Test: `dashboard/backend/tests/domain/analytics/test_value_queries.py` — rewrite `test_profile_includes_selected_period_and_lifecycle_transition` (663), `test_profile_never_uses_anonymous_transition_rollups` (693)
- Test: `dashboard/backend/tests/test_admin_analytics_api.py` — rewrite `test_user_list_and_profile_are_display_safe` (280-345)

**Interfaces:**
- Consumes: `get_activity`, `sum_recent_facts`, `get_operational_facts` (existing per-user method, `value_repository.py:960`), `list_user_transitions` (Task 2), `list_activity_rows(section="timeline", limit=100)` (existing), `build_lifecycle_inputs`, `calculate_lifecycle`, `calculate_operational_state`, `resolve_group_badge` (Task 1), `store.list_admin_access` (`repository.py:580`).
- Produces:
  - `AnalyticsQueryService.get_user_axes(user_id: int, *, now: datetime) -> tuple[LifecycleResult, OperationalResult]` — live segment (inputs from `user_activity` + trailing-30 facts, `as_of=now`) and live operational state (`get_operational_facts` → `calculate_operational_state`).
  - `AnalyticsQueryService.get_user_profile(*, user_id, now=None, start=None, end=None, axes=None) -> AnalyticsUserProfile` — same field set; sources below.
  - `AnalyticsQueryService.get_user_metrics(user_id: int, *, audience: Literal["admin", "user"], now: datetime | None = None) -> UserMetrics` (C5, §6.13); `RunOutcomeCounts` and `UserMetrics` models. Nothing else calls it; not on any response model.
  - `ValueAnalyticsQueryService.get_user_profile` — same signature, same `ValueUserProfile`.

§6.10: *live segment and operational state; 7-day and 30-day sums from facts.* §6.12: *admin user profile load — at most 1 raw-event query (timeline page).* The 180-day per-user scan at `query_service.py:1099-1104` goes; the one raw read the profile keeps is the first **timeline page** (`list_activity_rows`, `LIMIT 101`, indexed on `(user_id, occurred_at)`), and the fields that were computed from the scan are re-sourced:

| Field | Source after this task |
|---|---|
| `state` | `_legacy_state(operational.state, lifecycle.segment)` with the matching live reason; `evidence_event_ids=[]` (the five-state evidence list has no source once `calculate_user_state` goes; the list type stays) |
| `last_meaningful_activity` | `user_activity.last_meaningful_activity_at` |
| `run_summary`, `platform_model_cost_usd`, `credits_debited_micro` | `sum_recent_facts([id], start=window_start, end=min(window_end, yesterday))` — `runs_*`, `operator_cost_micro / 1e6`, `own_spend_micro`; "as of yesterday" (§6.3) |
| `activation_milestones` | `account_signed_up` = `users.created_at`; `backtest_completed` = `user_activity.activated_at`; `credential_verified` / `agent_created` = earliest occurrence on the timeline page, absent if not on the page |
| `recent_footprint` | the page's first 20 meaningful events (unchanged rule, `is_meaningful_event`) |
| `primary_billing_lane`, `default_provider`, `country_code`, `device_category`, `browser_family`, `billing_lane_mix`, `top_product_page`, `input_tokens`, `output_tokens` | computed over the page instead of the window — page-bounded; D15 drops the display of the middle four in PR C, and the token totals are per-event on the usage tab |

The page-bounded token totals are the one number here that gets *smaller* rather than moving by a day; that is recorded in the PR body and in the Acceptance section, and PR D's re-cut decides whether the profile keeps them at all (§9 lists them nowhere among the kept fields).

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/domain/analytics/test_user_metrics.py`:

```python
"""One metric struct, two audiences (design §6.13)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from dashboard.backend.domain.analytics.query_service import AnalyticsQueryService
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
    analytics = AnalyticsStore(db_path=db_path)
    credits = SyntheticCreditsStore(tmp_path / "credits.db")
    value_store = ValueAnalyticsStore(
        analytics,
        credits,
        provider_base=SyntheticProviderStore({}, []),
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

    payload = service.get_user_metrics(user_id, audience="user", now=NOW).model_dump()

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


def test_the_admin_audience_sees_every_field(metrics_fixture):
    service, user_id = metrics_fixture

    payload = service.get_user_metrics(user_id, audience="admin", now=NOW).model_dump()

    assert _ADMIN_ONLY <= set(payload)
    assert payload["active_days_30d"] == 10
    assert payload["lifecycle"]["segment"] == "core"
    assert payload["operational"]["state"] in {"blocked", "needs_attention", "healthy"}
    assert payload["user_group"] == "partner"
    assert payload["group_badge"] == "partner"
    assert payload["access_log"] == ["overview"] or len(payload["access_log"]) == 1


def test_get_user_metrics_rejects_an_unknown_audience(metrics_fixture):
    service, user_id = metrics_fixture
    with pytest.raises(ValueError):
        service.get_user_metrics(user_id, audience="public", now=NOW)
```

`analytics.record_admin_access(admin_id, subject_id, section)` is the existing store method (`repository.py:549`). Replace `test_user_list_and_profile_are_display_safe` (test_admin_analytics_api.py:280-345; PR A already removed its `list_users` half with the dead legacy list) with:

```python
def test_profile_is_display_safe_and_reads_facts_not_the_event_window(tmp_path):
    analytics, events, _rollups, _states, users = _fixture(tmp_path)
    from dashboard.backend.domain.analytics.value_repository import ValueAnalyticsStore
    from dashboard.backend.tests.domain.analytics.test_fact_reads import fact
    from dashboard.backend.tests.domain.analytics.test_value_repository import (
        SyntheticAgentStore, SyntheticCreditsStore, SyntheticProviderStore, SyntheticRunStore,
    )

    value_store = ValueAnalyticsStore(
        analytics, SyntheticCreditsStore(tmp_path / "credits.db"),
        provider_base=SyntheticProviderStore({}, []), agent_base=SyntheticAgentStore(), run_base=SyntheticRunStore({}),
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
    recorder = _WindowRecordingQueryStore(service.query_store)
    service.query_store = recorder

    profile = service.get_user_profile(user_id=1, now=NOW, start=NOW - timedelta(days=30), end=NOW)
    serialized = profile.model_dump(mode="json")

    assert profile.state.status == "active"
    assert profile.run_summary == {"requested": 3, "completed": 2, "failed": 1, "cancelled": 0}
    assert profile.platform_model_cost_usd == 0.25
    assert profile.credits_debited_micro == 100
    assert profile.input_tokens == 120
    assert profile.default_provider == "openrouter"
    assert profile.activation_milestones["backtest_completed"] == NOW - timedelta(hours=1)
    assert recorder.windows == []  # no list_metric_events call: the only raw read is one timeline page
    assert "properties" not in str(serialized)
    assert "session_id" not in str(serialized)
```

and in `test_value_queries.py` replace the two profile tests with:

```python
def test_value_profile_combines_live_axes_facts_and_user_transitions():
    transitions = [
        LifecycleTransitionRow(user_id=1, snapshot_date=YESTERDAY, from_segment="growing", to_segment="core",
                               inactive_days=0, data_quality="complete", created_at=NOW),
        LifecycleTransitionRow(user_id=2, snapshot_date=YESTERDAY, from_segment="growing", to_segment="at_risk",
                               inactive_days=8, data_quality="complete", created_at=NOW),
    ]
    service = _service(
        facts=[_fact(1, YESTERDAY - timedelta(days=offset)) for offset in range(3)],
        transitions=transitions, activity={1: _activity(1)}, commercial={1: _commercial(1, 6_000_000)},
        operational_facts={1: CurrentOperationalFacts(user_id=1, account_restricted=True)},
    )

    profile = service.get_user_profile(user_id=1, start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1), now=NOW)

    assert profile.lifecycle.segment == "core"
    assert profile.operational.state == "blocked"
    assert profile.operational.reason_code == "account_restricted"
    assert profile.commercial.commercial_tier == "invested"
    assert (profile.selected_period_start, profile.selected_period_end) == (YESTERDAY - timedelta(days=6), YESTERDAY + timedelta(days=1))
    assert [(t.from_segment, t.to_segment, t.users) for t in profile.recent_lifecycle_transitions] == [("growing", "core", 1)]
    assert service.legacy_service.profile_windows == [(_day_start(YESTERDAY - timedelta(days=6)), _day_start(YESTERDAY + timedelta(days=1)))]


def test_value_profile_raises_lookup_error_for_an_unknown_user():
    service = _service(activity={}, commercial={})
    service.legacy_service.raise_lookup = True

    with pytest.raises(LookupError):
        service.get_user_profile(user_id=404, start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1), now=NOW)
```

`FakeLegacyService.get_user_profile` gains `axes=None` in its signature and a `raise_lookup` attribute (default `False`) that raises `LookupError` when set; import `CurrentOperationalFacts` from `value_repository` in the test module.

- [ ] **Step 2: Run them and confirm the expected failure**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_user_metrics.py dashboard/backend/tests/domain/analytics/test_value_queries.py -v -k "metrics or profile"
python -m pytest dashboard/backend/tests/test_admin_analytics_api.py -v -k profile_is_display_safe
```

Expected: **FAIL** — `AttributeError: 'AnalyticsQueryService' object has no attribute 'get_user_metrics'`; the profile test fails because `recorder.windows` is empty but `list_events` on `rollups` is still called (the assertion order puts `state.status` first — with no snapshot it is computed via `calculate_user_state`, so the first failure is `recorder`-independent: `run_summary == {...}` mismatches, since today's numbers come from the scan).

- [ ] **Step 3: `query_service.py`**

Add the two models after `AnalyticsOverview` (line 264):

```python
class RunOutcomeCounts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requested: int = Field(default=0, ge=0)
    completed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    cancelled: int = Field(default=0, ge=0)


class UserMetrics(BaseModel):
    """One user's numbers, built once and projected per audience (design §6.13).

    Every field below the shared block is admin-only and is *absent* from the
    user projection, not nulled: a null still tells the subject the field
    exists. A future GET /api/me/usage is this call with audience="user".
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: int = Field(gt=0)
    facts_as_of: date
    runs_7d: RunOutcomeCounts
    runs_30d: RunOutcomeCounts
    own_spend_micro: int = Field(default=0, ge=0)
    operator_cost_micro: int = Field(default=0, ge=0)
    tier: CommercialTier = "unpaid"
    credit_balance_micro: int = Field(default=0, ge=0)
    data_quality: Literal["complete", "partial"] = "complete"

    active_days_30d: int | None = Field(default=None, ge=0, le=30)
    lifecycle: LifecycleResult | None = None
    operational: OperationalResult | None = None
    user_group: str | None = None
    group_badge: str | None = None
    access_log: Sequence[str] | None = None


_ADMIN_ONLY_METRIC_FIELDS = frozenset(
    {"active_days_30d", "lifecycle", "operational", "user_group", "group_badge", "access_log"}
)
```

with `from .lifecycle import (CommercialTier, LifecycleResult, OperationalResult, OperationalSignals, calculate_lifecycle, calculate_operational_state, operational_reason, resolve_group_badge)` and `from .lifecycle_reads import build_lifecycle_inputs` added to the imports. Then replace `get_user_profile` (1071-1200) with:

```python
    def get_user_axes(
        self, user_id: int, *, now: datetime
    ) -> tuple[LifecycleResult, OperationalResult]:
        """Live segment and live operational state for exactly one user."""
        subject_id = positive_user_id(user_id)
        current = _utc(now, "now")
        user = self.user_store.get_user_admin(subject_id)
        if user is None:
            raise LookupError("Analytics user was not found")
        yesterday = current.date() - timedelta(days=1)
        activity = self.value_store.get_activity(subject_id)
        totals = self.value_store.sum_recent_facts(
            [subject_id], start=yesterday - timedelta(days=29), end=yesterday
        ).get(subject_id)
        lifecycle = calculate_lifecycle(
            build_lifecycle_inputs(
                subject_id,
                created_at=_parse_timestamp(user["created_at"]),
                activity=activity,
                totals=totals,
                as_of=current,
            ),
            current,
        )
        facts = self.value_store.get_operational_facts(subject_id, now=current)
        operational = calculate_operational_state(
            OperationalSignals(**facts.model_dump()), current
        )
        return lifecycle, operational

    def get_user_profile(
        self,
        *,
        user_id: int,
        now: datetime | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        axes: tuple[LifecycleResult, OperationalResult] | None = None,
    ) -> AnalyticsUserProfile:
        subject_id = positive_user_id(user_id)
        current = _utc(now or datetime.now(timezone.utc), "now")
        range_start = _utc(start, "start") if start is not None else current - timedelta(days=180)
        range_end = _utc(end, "end") if end is not None else current + timedelta(microseconds=1)
        if range_end <= range_start:
            raise ValueError("end must be after start")
        user = self.user_store.get_user_admin(subject_id)
        if user is None:
            raise LookupError("Analytics user was not found")
        lifecycle, operational = axes or self.get_user_axes(subject_id, now=current)
        yesterday = current.date() - timedelta(days=1)
        window_start = range_start.date()
        window_end = min((range_end - timedelta(microseconds=1)).date(), yesterday)
        totals = (
            self.value_store.sum_recent_facts([subject_id], start=window_start, end=window_end).get(subject_id)
            if window_end >= window_start
            else None
        )
        activity = self.value_store.get_activity(subject_id)

        # The profile's one raw-event read: the first timeline page (design §6.12).
        rows, _next = self.query_store.list_activity_rows(
            user_id=subject_id, section="timeline", limit=100, cursor=None
        )
        events = [event for _sequence, event in rows]  # newest first
        meaningful = [event for event in events if is_meaningful_event(event)]
        milestones: dict[str, datetime] = {"account_signed_up": _parse_timestamp(user["created_at"])}
        if activity is not None and activity.activated_at is not None:
            milestones["backtest_completed"] = activity.activated_at
        for event_name in ("credential_verified", "agent_created"):
            matches = [event.occurred_at for event in events if event.event_name == event_name]
            if matches:
                milestones[event_name] = min(matches)
        lane_counts = Counter(
            event.billing_mode
            for event in events
            if event.event_name == "model_usage_recorded" and event.billing_mode
        )
        page_counts = Counter(
            event.page_view for event in events if event.event_name == "page_viewed" and event.page_view
        )
        provider_event = next(
            (
                event for event in events
                if event.provider_id
                and event.event_name in {"credential_defaulted", "credential_verified", "model_usage_recorded"}
            ),
            None,
        )
        client_event = next(
            (event for event in events if event.country_code or event.device_category or event.browser_family),
            None,
        )
        state_status = _legacy_state(operational.state, lifecycle.segment)
        state_reason = operational if state_status in {"blocked", "needs_attention"} else lifecycle
        return AnalyticsUserProfile(
            user_id=subject_id,
            display_name=str(user.get("display_name") or ""),
            email=str(user.get("email") or ""),
            joined_at=_parse_timestamp(user["created_at"]),
            last_meaningful_activity=(
                activity.last_meaningful_activity_at if activity is not None else None
            ),
            state=AnalyticsStateSummary(
                status=state_status,
                reason_code=state_reason.reason_code,
                human_readable_reason=state_reason.reason,
                evidence_event_ids=[],
                calculated_at=current,
            ),
            primary_billing_lane=(lane_counts.most_common(1)[0][0] if lane_counts else None),
            default_provider=(provider_event.provider_id if provider_event else None),
            country_code=(client_event.country_code if client_event else None),
            device_category=(client_event.device_category if client_event else None),
            browser_family=(client_event.browser_family if client_event else None),
            activation_milestones=milestones,
            recent_footprint=[
                AnalyticsFootprintItem(
                    event_id=event.event_id,
                    event_name=event.event_name,
                    occurred_at=event.occurred_at,
                    page_view=event.page_view,
                    provider_id=event.provider_id,
                    model_id=event.model_id,
                    billing_mode=event.billing_mode,
                    outcome=event.outcome,
                    error_category=event.error_category,
                )
                for event in meaningful[:20]
            ],
            run_summary={
                "requested": totals.runs_requested if totals else 0,
                "completed": totals.runs_completed if totals else 0,
                "failed": totals.runs_failed if totals else 0,
                "cancelled": totals.runs_cancelled if totals else 0,
            },
            billing_lane_mix=dict(lane_counts),
            input_tokens=sum(
                int(event.properties.get("input_tokens", 0))
                for event in events
                if event.event_name == "model_usage_recorded"
            ),
            output_tokens=sum(
                int(event.properties.get("output_tokens", 0))
                for event in events
                if event.event_name == "model_usage_recorded"
            ),
            platform_model_cost_usd=(totals.operator_cost_micro if totals else 0) / 1_000_000,
            credits_debited_micro=totals.own_spend_micro if totals else 0,
            top_product_page=(page_counts.most_common(1)[0][0] if page_counts else None),
        )

    def get_user_metrics(
        self,
        user_id: int,
        *,
        audience: Literal["admin", "user"],
        now: datetime | None = None,
    ) -> UserMetrics:
        """Build one user's metric struct once; project it for the audience."""
        if audience not in {"admin", "user"}:
            raise ValueError("audience must be admin or user")
        subject_id = positive_user_id(user_id)
        current = _utc(now or datetime.now(timezone.utc), "now")
        user = self.user_store.get_user_admin(subject_id)
        if user is None:
            raise LookupError("Analytics user was not found")
        yesterday = current.date() - timedelta(days=1)
        lifecycle, operational = self.get_user_axes(subject_id, now=current)
        week = self.value_store.sum_recent_facts(
            [subject_id], start=yesterday - timedelta(days=6), end=yesterday
        ).get(subject_id)
        month = self.value_store.sum_recent_facts(
            [subject_id], start=yesterday - timedelta(days=29), end=yesterday
        ).get(subject_id)
        commercial = self.value_store.list_commercial_values(
            [subject_id],
            start=datetime.combine(yesterday, datetime.min.time(), tzinfo=timezone.utc),
            end=datetime.combine(yesterday + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc),
        ).get(subject_id)
        tier = commercial.commercial_tier if commercial is not None else "unpaid"
        user_group = str(user.get("user_group") or "unknown")

        def counts(totals) -> RunOutcomeCounts:
            if totals is None:
                return RunOutcomeCounts()
            return RunOutcomeCounts(
                requested=totals.runs_requested,
                completed=totals.runs_completed,
                failed=totals.runs_failed,
                cancelled=totals.runs_cancelled,
            )

        metrics = UserMetrics(
            user_id=subject_id,
            facts_as_of=yesterday,
            runs_7d=counts(week),
            runs_30d=counts(month),
            own_spend_micro=month.own_spend_micro if month else 0,
            operator_cost_micro=month.operator_cost_micro if month else 0,
            tier=tier,
            credit_balance_micro=commercial.total_available_micro if commercial is not None else 0,
            data_quality="complete" if month is not None and month.days_present >= 30 else "partial",
            active_days_30d=min(30, month.active_days) if month else 0,
            lifecycle=lifecycle,
            operational=operational,
            user_group=user_group,
            group_badge=resolve_group_badge(
                role=str(user.get("role") or "user"), user_group=user_group, tier=tier
            ),
            access_log=[
                str(row["section"]) for row in self.store.list_admin_access(subject_id, limit=10)
            ],
        )
        if audience == "admin":
            return metrics
        return UserMetrics.model_validate(
            metrics.model_dump(exclude=_ADMIN_ONLY_METRIC_FIELDS)
        )
```

`UserMetrics.model_validate(model_dump(exclude=...))` re-creates the struct with the admin-only fields at their `None` defaults; the **user projection is what the caller serialises with `model_dump(exclude_none=True)`** — but the test above asserts absence on a plain `model_dump()`. To make absence structural rather than a serialisation option, add to `UserMetrics`:

```python
    def model_dump(self, **kwargs):  # type: ignore[override]
        data = super().model_dump(**kwargs)
        if self.lifecycle is None and self.operational is None and self.access_log is None:
            for name in _ADMIN_ONLY_METRIC_FIELDS:
                data.pop(name, None)
        return data
```

That override is the audience filter's teeth: a user-audience struct has every admin field at `None`, and dumping it never emits those keys. Extend `__all__` with `RunOutcomeCounts`, `UserMetrics`.

- [ ] **Step 4: `value_queries.py`**

Replace `get_user_profile` (1351-1390) with:

```python
    def get_user_profile(
        self,
        *,
        user_id: int,
        start: date,
        end: date,
        now: datetime | None = None,
    ) -> ValueUserProfile:
        start, end = _validate_dates(start, end)
        subject_id = positive_user_id(user_id)
        current_time = _utc(now or datetime.now(UTC), "now")
        axes = self.legacy_service.get_user_axes(subject_id, now=current_time)
        legacy = self.legacy_service.get_user_profile(
            user_id=subject_id,
            now=current_time,
            start=_day_start(start),
            end=_day_start(end),
            axes=axes,
        )
        commercial = self.value_store.list_commercial_values(
            [subject_id], start=_day_start(start), end=_day_start(end)
        )[subject_id]
        transitions = [
            LifecycleTransition(
                from_segment=row.from_segment,
                to_segment=row.to_segment,
                users=1,
                period_start=row.snapshot_date,
                period_end=row.snapshot_date + timedelta(days=1),
                data_quality=row.data_quality,
            )
            for row in self.value_store.list_user_transitions(subject_id, start=start, end=end)
        ]
        lifecycle, operational = axes
        return ValueUserProfile(
            **legacy.model_dump(),
            lifecycle=lifecycle,
            operational=operational,
            commercial=commercial,
            selected_period_start=start,
            selected_period_end=end,
            recent_lifecycle_transitions=transitions,
        )
```

Per-user transitions now carry `period_start`/`period_end` = the transition's own day; today every transition row in the profile carried the whole selected window (`_history` returned window-level aggregates). Same shape, more precise value. Delete `_current` (512-522), `_lifecycle` and `_operational` (398-420) and the `UserValueSnapshot` import. `FakeLegacyService.get_user_axes` in the test module returns `(LifecycleResult(...core...), OperationalResult(...))` built from `CurrentOperationalFacts` via `calculate_operational_state` when `operational_facts` were given, else healthy — add it to the fake:

```python
    def get_user_axes(self, user_id, *, now):
        from dashboard.backend.domain.analytics.lifecycle import (
            LifecycleInputs, OperationalSignals, calculate_lifecycle, calculate_operational_state,
        )
        facts = self.operational_facts.get(user_id)
        operational = calculate_operational_state(
            OperationalSignals(**facts.model_dump()) if facts else OperationalSignals(user_id=user_id), now
        )
        lifecycle = calculate_lifecycle(
            LifecycleInputs(user_id=user_id, created_at=now - timedelta(days=90),
                            first_successful_backtest_at=now - timedelta(days=60),
                            last_meaningful_activity_at=now - timedelta(days=2),
                            active_days_30d=3, successful_backtests_30d=3),
            now,
        )
        return lifecycle, operational
```

with `_service` passing `operational_facts` into `FakeLegacyService(legacy_availability, operational_facts=operational_facts)`.

- [ ] **Step 5: Run and confirm the pass**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_user_metrics.py dashboard/backend/tests/domain/analytics/test_value_queries.py dashboard/backend/tests/test_admin_analytics_api.py -v
```

Expected: **PASS**. This is the first point at which the whole of `test_value_queries.py` is expected green.

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/domain/analytics/query_service.py \
        dashboard/backend/domain/analytics/value_queries.py \
        dashboard/backend/tests/domain/analytics/test_user_metrics.py \
        dashboard/backend/tests/domain/analytics/test_value_queries.py \
        dashboard/backend/tests/test_admin_analytics_api.py
git status --short
git commit -m "$(cat <<'EOF'
feat: build the user profile from live axes and facts; add get_user_metrics

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: The activity route's sessions section gets a 30-day window

**Files:**
- Modify: `dashboard/backend/domain/analytics/query_service.py` — `AnalyticsQueryStore.list_session_rows` (lines 526-603), `AnalyticsQueryService.get_user_activity` (1202-1287)
- Test: `dashboard/backend/tests/test_admin_analytics_api.py` (extend)

**Interfaces:**
- Consumes: nothing new.
- Produces: `list_session_rows(*, user_id, limit, cursor, start: datetime)` — `start` is required; `get_user_activity(..., now: datetime | None = None)` passes `now - 30 days`. Same `AnalyticsActivityPage` shape.

§6.15 item 3: *the sessions section scans a user's entire experience-event history with no time window. PR B gives it a 30-day window.* The other three sections keep their keyset-paginated `LIMIT` reads.

- [ ] **Step 1: Write the failing test**

Append to `test_admin_analytics_api.py`:

```python
def test_sessions_section_is_bounded_to_thirty_days(tmp_path):
    analytics, events, _rollups, _states, users = _fixture(tmp_path)
    context = RequestAnalyticsContext(country_code="US", device_category="desktop", browser_family="Chrome")
    for days_ago in (2, 45):
        events.accept_frontend_event(
            user={"id": 1},
            payload=FrontendAnalyticsEvent(
                event_id=str(uuid4()), schema_version=1, event_name="page_viewed",
                session_id=str(uuid4()), occurred_at=NOW - timedelta(days=days_ago),
                page_view="agents", properties={},
            ),
            context=context,
            received_at=NOW,
        )
    service = AnalyticsQueryService(store=analytics, user_store=users)

    sessions = service.get_user_activity(user_id=1, section="sessions", limit=10, cursor=None, now=NOW)

    assert len(sessions.items) == 1
    assert sessions.items[0].occurred_at == NOW - timedelta(days=2)
    import inspect
    from dashboard.backend.domain.analytics.query_service import AnalyticsQueryStore
    assert "occurred_at >=" in inspect.getsource(AnalyticsQueryStore.list_session_rows)
```

- [ ] **Step 2: Run it and confirm the expected failure**

```bash
python -m pytest dashboard/backend/tests/test_admin_analytics_api.py -v -k thirty_days
```

Expected: **FAIL** — `TypeError: get_user_activity() got an unexpected keyword argument 'now'`.

- [ ] **Step 3: Implement**

In `list_session_rows` (526-603) add `start: datetime` to the keyword parameters and replace the two SQL statements' `WHERE user_id = ? AND event_group = 'experience'` (and the `%s` twin) with `WHERE user_id = ? AND occurred_at >= ? AND event_group = 'experience'` / `... %s ...`, passing `(subject_id, utc_iso(_utc(start, "start")))` as the parameters in both branches. In `get_user_activity` (1202) add `now: datetime | None = None` after `cursor`, and change the sessions call to:

```python
        if section == "sessions":
            current = _utc(now or datetime.now(timezone.utc), "now")
            rows, next_cursor = self.query_store.list_session_rows(
                user_id=user_id,
                limit=page_size,
                cursor=cursor,
                start=current - timedelta(days=30),
            )
```

The router (`admin_analytics.py:579-584`) does not pass `now`; the default is fine there.

- [ ] **Step 4: Run and confirm the pass**

```bash
python -m pytest dashboard/backend/tests/test_admin_analytics_api.py -v -k "thirty_days or activity_sections"
```

Expected: **PASS** — `test_activity_sections_page_independently_and_hide_session_ids` still passes (its events are minutes old).

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/analytics/query_service.py dashboard/backend/tests/test_admin_analytics_api.py
git status --short
git commit -m "$(cat <<'EOF'
perf: bound the analytics sessions section to a 30-day window

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: Long-term rollups by tier and `user_group`; retention expires facts and transitions

**Files:**
- Modify: `dashboard/backend/domain/analytics/value_repository.py` — `LIFECYCLE_ROLLUP_METRICS` (line 30-32), `replace_lifecycle_rollups` (524-637); add `list_expiring_fact_dates`, `delete_facts_for_date`, `delete_transitions_for_date`, `has_facts_before` after `list_transitions_for_date` (Task 2)
- Modify: `dashboard/backend/domain/analytics/value_repository_postgres.py` — the same four methods and the same validator on `PostgresValueAnalyticsStore`
- Modify: `dashboard/backend/domain/analytics/rollups.py` — `replace_day` (97-164), `rollup_lifecycle_day` (471-537)
- Modify: `dashboard/backend/domain/analytics/retention.py` — `run_once` (54-118)
- Test: `dashboard/backend/tests/domain/analytics/test_retention.py` — rewrite `test_lifecycle_history_is_aggregated_before_user_rows_are_deleted` (272-357)
- Test: `dashboard/backend/tests/domain/analytics/test_rollups.py` — rewrite `test_lifecycle_rollup_is_bounded_and_preserves_other_metrics` (109-174)

**Interfaces:**
- Consumes: `list_facts_for_date` (PR A), `list_transitions_for_date` (Task 2).
- Produces:
  - Six new `metric_name` values in `analytics_daily_rollups`, encoded per §6.7: `runs_by_tier`, `runs_by_user_group`, `operator_cost_by_tier`, `operator_cost_by_user_group`, `own_spend_by_tier`, `own_spend_by_user_group`; `user_state` carries the tier or group value, `value_count` the completed-run count (runs families) or the number of contributing users (cost families), `value_sum_micro` the micro-USD sum (cost families), `outcome` the day's quality. Nothing reads them yet; they exist so history survives the 180-day expiry (§11).
  - `LIFECYCLE_ROLLUP_METRICS` grows from two names to eight; `replace_day`'s `NOT IN (?, ?)` becomes a rendered list of that length.
  - `rollup_lifecycle_day(day, *, store)` reads `list_facts_for_date(day)` for segment counts and `list_transitions_for_date(day)` for transitions (the transition log is now the source, not a diff of two days), and writes the six families.
  - On both twins: `list_expiring_fact_dates(*, before: date, limit: int) -> list[date]` (distinct dates from **both** user-keyed tables), `delete_facts_for_date(day) -> int`, `delete_transitions_for_date(day) -> int`, `has_facts_before(before: date) -> bool` (either table).
  - `AnalyticsRetentionService.run_once` rolls up every expiring day, then deletes that day's facts **and** transitions; `RetentionResult.lifecycle_rows_deleted` counts both.

The 09-12 plan's C4 named the families `tier_*`/`cohort_*`; the design (§6.7, D9) renames them and this plan follows the design. `lifecycle_transitions` is the table the 09-12 plan called "the worst one to miss" — it carries a user id, is appended nightly, and nothing else deletes from it; the retention test below pins its expiry alongside the fact rows.

- [ ] **Step 1: Write the failing tests**

Replace `test_lifecycle_rollup_is_bounded_and_preserves_other_metrics` (`test_rollups.py:109-174`) with:

```python
def test_lifecycle_rollup_reads_facts_and_transitions_and_writes_tier_and_group_families(tmp_path):
    from dashboard.backend.tests.domain.analytics.test_fact_reads import fact, transition

    analytics, rollups = _store(tmp_path)
    values = ValueAnalyticsStore(
        analytics, credits_base=object(), provider_base=object(), agent_base=object(), run_base=object()
    )
    day = date(2026, 8, 25)
    updated_at = datetime(2026, 8, 26, tzinfo=timezone.utc)
    rollups.replace_day(
        day, [DailyRollup(rollup_date=day, metric_name="completed_runs", value_count=7, updated_at=updated_at)]
    )
    values.upsert_daily_facts(
        [
            fact(1, day, lifecycle_segment="growing", tier="starter", user_group="organic",
                 runs_completed=2, operator_cost_micro=100_000, own_spend_micro=5_000),
            fact(2, day, lifecycle_segment="onboarding", tier="unpaid", user_group="organic",
                 runs_completed=0, operator_cost_micro=0, own_spend_micro=0, data_quality="partial"),
        ]
    )
    values.append_lifecycle_transitions([transition(1, day, "new", "growing")])

    first = rollup_lifecycle_day(day, store=values)
    second = rollup_lifecycle_day(day, store=values)
    stored = rollups.list_rollups(start=day, end=day + timedelta(days=1))

    assert first == second
    assert any(row.metric_name == "completed_runs" and row.value_count == 7 for row in stored)
    assert {(r.user_state, r.value_count, r.outcome) for r in stored if r.metric_name == "lifecycle_segment_count"} == {
        ("growing", 1, "complete"), ("onboarding", 1, "partial")
    }
    assert {(r.event_name, r.user_state, r.value_count) for r in stored if r.metric_name == "lifecycle_transition"} == {
        ("new", "growing", 1)
    }
    assert {(r.user_state, r.value_count) for r in stored if r.metric_name == "runs_by_tier"} == {("starter", 2), ("unpaid", 0)}
    assert {(r.user_state, r.value_count) for r in stored if r.metric_name == "runs_by_user_group"} == {("organic", 2)}
    assert {(r.user_state, r.value_count, r.value_sum_micro) for r in stored if r.metric_name == "operator_cost_by_tier"} == {
        ("starter", 1, 100_000), ("unpaid", 1, 0)
    }
    assert {(r.user_state, r.value_sum_micro) for r in stored if r.metric_name == "own_spend_by_user_group"} == {("organic", 5_000)}
    assert all("user_id" not in row.model_dump() for row in first)

    rollup_day(day, store=rollups)
    rebuilt = rollups.list_rollups(start=day, end=day + timedelta(days=1))
    for name in ("lifecycle_transition", "lifecycle_segment_count", "runs_by_tier", "own_spend_by_user_group"):
        assert any(row.metric_name == name for row in rebuilt), name
```

Drop `UserLifecycleDailySnapshot` from that module's imports. Replace `test_lifecycle_history_is_aggregated_before_user_rows_are_deleted` (`test_retention.py:272-357`) with:

```python
def test_facts_and_transitions_are_aggregated_before_user_rows_are_deleted(tmp_path):
    from dashboard.backend.tests.domain.analytics.test_fact_reads import fact, transition

    db_path = tmp_path / "lifecycle-retention.db"
    users = UserStore(db_path=db_path)
    subject = users.create_user("lifecycle-retention@example.test", "Lifecycle Retention", "SecurePass1!")
    user_id = int(subject["id"])
    analytics = AnalyticsStore(db_path=db_path)
    values = ValueAnalyticsStore(
        analytics, credits_base=object(), provider_base=object(), agent_base=object(), run_base=object()
    )
    cutoff = NOW.date() - timedelta(days=RAW_EVENT_RETENTION_DAYS)
    values.upsert_daily_facts(
        [
            fact(user_id, cutoff - timedelta(days=2), lifecycle_segment="onboarding", tier="starter", user_group="partner", runs_completed=0),
            fact(user_id, cutoff - timedelta(days=1), lifecycle_segment="growing", tier="starter", user_group="partner", runs_completed=3, operator_cost_micro=90_000),
            fact(user_id, cutoff, lifecycle_segment="growing", tier="starter", user_group="partner"),
        ]
    )
    values.append_lifecycle_transitions(
        [
            transition(user_id, cutoff - timedelta(days=1), "onboarding", "growing"),
            transition(user_id, cutoff + timedelta(days=3), "growing", "at_risk", inactive_days=8),
        ]
    )
    with analytics._get_connection() as conn:
        conn.execute(
            "INSERT INTO analytics_daily_rollups (rollup_date, metric_name, value_count, updated_at) VALUES (?, 'completed_runs', 3, ?)",
            ((cutoff - timedelta(days=1)).isoformat(), utc_iso(NOW)),
        )

    result = AnalyticsRetentionService(store=analytics, value_store=values, batch_size=10, max_batches=10).run_once(NOW)

    assert result.lifecycle_rows_deleted == 3  # two fact rows + one transition
    assert result.has_more_lifecycle_rows is False
    assert [row.snapshot_date for row in values.list_facts_for_date(cutoff)] == [cutoff]
    assert values.list_facts_for_date(cutoff - timedelta(days=1)) == []
    assert values.list_transitions_for_date(cutoff - timedelta(days=1)) == []
    assert len(values.list_transitions_for_date(cutoff + timedelta(days=3))) == 1  # inside the window, untouched
    with analytics._get_connection() as conn:
        rows = {
            tuple(row)
            for row in conn.execute(
                "SELECT metric_name, event_name, user_state, value_count, value_sum_micro FROM analytics_daily_rollups WHERE rollup_date = ?",
                ((cutoff - timedelta(days=1)).isoformat(),),
            ).fetchall()
        }
    assert ("completed_runs", "", "", 3, 0) in rows
    assert ("lifecycle_transition", "onboarding", "growing", 1, 0) in rows
    assert ("runs_by_tier", "", "starter", 3, 0) in rows
    assert ("operator_cost_by_user_group", "", "partner", 1, 90_000) in rows


def test_user_activity_is_never_swept(tmp_path):
    """Design §11: user_activity is current state, not history."""

    db_path = tmp_path / "activity-retention.db"
    users = UserStore(db_path=db_path)
    subject = users.create_user("activity-retention@example.test", "Activity", "SecurePass1!")
    analytics = AnalyticsStore(db_path=db_path)
    values = ValueAnalyticsStore(
        analytics, credits_base=object(), provider_base=object(), agent_base=object(), run_base=object()
    )
    values.record_activity(int(subject["id"]), occurred_at=NOW - timedelta(days=400), activating=True, now=NOW)

    AnalyticsRetentionService(store=analytics, value_store=values, batch_size=10, max_batches=10).run_once(NOW)

    assert values.get_activity(int(subject["id"])).activated_at == NOW - timedelta(days=400)
```

Drop `UserLifecycleDailySnapshot` from `test_retention.py`'s imports; the `user_analytics_snapshots` insert and assertion inside `test_sqlite_retention_is_bounded_and_preserves_aggregates` (lines 231-238 and 266-268) are removed in Task 14 with the table — leave them for now.

- [ ] **Step 2: Run them and confirm the expected failure**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_rollups.py dashboard/backend/tests/domain/analytics/test_retention.py -v -k "tier_and_group or aggregated_before or never_swept"
```

Expected: **FAIL** — `rollup_lifecycle_day` still calls `store.list_daily_snapshots` and writes no `runs_by_tier` rows; `run_once` still calls `list_expiring_daily_dates`.

- [ ] **Step 3: `value_repository.py` (and the Postgres twin)**

Replace `LIFECYCLE_ROLLUP_METRICS` (30-32):

```python
LIFECYCLE_ROLLUP_METRICS = frozenset(
    {
        "lifecycle_segment_count",
        "lifecycle_transition",
        # Long-term families written before user-keyed rows expire (design §6.7):
        # metric_name names the dimension, user_state carries its value.
        "runs_by_tier",
        "runs_by_user_group",
        "operator_cost_by_tier",
        "operator_cost_by_user_group",
        "own_spend_by_tier",
        "own_spend_by_user_group",
    }
)
_DIMENSION_FAMILIES = frozenset(
    {"runs_by_tier", "runs_by_user_group", "operator_cost_by_tier",
     "operator_cost_by_user_group", "own_spend_by_tier", "own_spend_by_user_group"}
)
```

In `replace_lifecycle_rollups`, replace the validation block (lines 555-578) with:

```python
            if row.metric_name == "lifecycle_segment_count":
                valid_dimensions = not row.event_name and row.user_state in LIFECYCLE_SEGMENTS
                valid_sum = row.value_sum_micro == 0
            elif row.metric_name == "lifecycle_transition":
                valid_dimensions = (
                    replace_transitions
                    and row.event_name in LIFECYCLE_SEGMENTS
                    and row.user_state in LIFECYCLE_SEGMENTS
                    and row.event_name != row.user_state
                )
                valid_sum = row.value_sum_micro == 0
            else:
                # A long-term family: user_state carries the tier or group value.
                valid_dimensions = not row.event_name and bool(row.user_state)
                if row.metric_name.startswith("runs_by_"):
                    valid_sum = row.value_sum_micro == 0
                else:
                    valid_sum = row.value_sum_micro >= 0
            unused_dimensions = (row.billing_mode, row.provider_id, row.model_id, row.error_category)
            if (
                not valid_dimensions
                or any(unused_dimensions)
                or row.outcome not in {"complete", "partial"}
                or not valid_sum
            ):
                raise ValueError("invalid lifecycle rollup dimensions")
```

and replace `metrics = ["lifecycle_segment_count"]` / `if replace_transitions: metrics.append("lifecycle_transition")` (596-598) with:

```python
        metrics = sorted(LIFECYCLE_ROLLUP_METRICS - ({"lifecycle_transition"} if not replace_transitions else set()))
```

Add after `list_transitions_for_date` on `ValueAnalyticsStore`:

```python
    def list_expiring_fact_dates(self, *, before: date, limit: int) -> list[date]:
        """Distinct dates older than ``before`` in either user-keyed table."""
        if not isinstance(before, date) or isinstance(before, datetime):
            raise ValueError("before must be a date")
        page_size = positive_limit(limit, maximum=1000)
        sql = """
            SELECT snapshot_date FROM (
                SELECT DISTINCT snapshot_date FROM user_daily_facts WHERE snapshot_date < ?
                UNION
                SELECT DISTINCT snapshot_date FROM lifecycle_transitions WHERE snapshot_date < ?
            ) AS expiring
            ORDER BY snapshot_date
            LIMIT ?
        """
        with self._analytics_connection() as conn:
            rows = conn.execute(sql, (before.isoformat(), before.isoformat(), page_size)).fetchall()
        return [date.fromisoformat(str(_row_value(row, "snapshot_date"))) for row in rows]

    def delete_facts_for_date(self, day: date) -> int:
        if not isinstance(day, date) or isinstance(day, datetime):
            raise ValueError("day must be a date")
        with self._analytics_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM user_daily_facts WHERE snapshot_date = ?", (day.isoformat(),)
            )
            return max(0, int(cursor.rowcount))

    def delete_transitions_for_date(self, day: date) -> int:
        if not isinstance(day, date) or isinstance(day, datetime):
            raise ValueError("day must be a date")
        with self._analytics_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM lifecycle_transitions WHERE snapshot_date = ?", (day.isoformat(),)
            )
            return max(0, int(cursor.rowcount))

    def has_facts_before(self, before: date) -> bool:
        if not isinstance(before, date) or isinstance(before, datetime):
            raise ValueError("before must be a date")
        sql = """
            SELECT 1 FROM user_daily_facts WHERE snapshot_date < ?
            UNION ALL
            SELECT 1 FROM lifecycle_transitions WHERE snapshot_date < ?
            LIMIT 1
        """
        with self._analytics_connection() as conn:
            row = conn.execute(sql, (before.isoformat(), before.isoformat())).fetchone()
        return row is not None
```

Postgres twin: the same four methods with `%s` and `cur.execute`; the two `DELETE`s use `RETURNING user_id` and `len(cur.fetchall())` exactly as `delete_daily_snapshots_for_date` does today (`value_repository.py:671-681`), because psycopg's `rowcount` is reliable but the existing twin convention is `RETURNING`, and PR T's parity guard compares signatures, not bodies. Apply the same `replace_lifecycle_rollups` validator change to the twin (PR T carries the Postgres copy of that method).

- [ ] **Step 4: `rollups.py`**

In `replace_day` (97-164), replace both `AND metric_name NOT IN (%s, %s)` / `(?, ?)` with a rendered placeholder list and pass `sorted(LIFECYCLE_ROLLUP_METRICS)`:

```python
        preserved = sorted(LIFECYCLE_ROLLUP_METRICS)
        ...
                    cur.execute(
                        f"""
                        DELETE FROM analytics_daily_rollups
                        WHERE rollup_date = %s
                          AND metric_name NOT IN ({", ".join("%s" for _ in preserved)})
                        """,
                        (day.isoformat(), *preserved),
                    )
        ...
                conn.execute(
                    f"""
                    DELETE FROM analytics_daily_rollups
                    WHERE rollup_date = ?
                      AND metric_name NOT IN ({", ".join("?" for _ in preserved)})
                    """,
                    (day.isoformat(), *preserved),
                )
```

Replace `_snapshot_quality` (465-468) and `rollup_lifecycle_day` (471-537) with:

```python
def _quality(rows: Sequence[Any]) -> str:
    return "partial" if any(row.data_quality == "partial" for row in rows) else "complete"


def rollup_lifecycle_day(day: date, *, store: Any) -> tuple[DailyRollup, ...]:
    """Persist anonymous lifecycle counts, transitions and the long-term
    tier / user_group families for one day (design §6.7, §11).

    Reads the fact rows and the transition log for ``day``; writes nothing
    keyed by a user. Called by the retention sweep for every expiring day
    before that day's user-keyed rows are deleted.
    """
    facts = store.list_facts_for_date(day)
    transitions = store.list_transitions_for_date(day)
    updated_at = _day_start(day + timedelta(days=1))
    rows: list[DailyRollup] = []

    by_segment: dict[str, list[Any]] = defaultdict(list)
    for fact in facts:
        by_segment[fact.lifecycle_segment].append(fact)
    for segment, group in sorted(by_segment.items()):
        rows.append(
            _row(day, "lifecycle_segment_count", count=len(group), user_state=segment,
                 outcome=_quality(group), updated_at=updated_at)
        )

    by_transition: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for row in transitions:
        by_transition[(row.from_segment, row.to_segment)].append(row)
    for (from_segment, to_segment), group in sorted(by_transition.items()):
        rows.append(
            _row(day, "lifecycle_transition", count=len(group), event_name=from_segment,
                 user_state=to_segment, outcome=_quality(group), updated_at=updated_at)
        )

    for dimension, attribute in (("tier", "tier"), ("user_group", "user_group")):
        by_value: dict[str, list[Any]] = defaultdict(list)
        for fact in facts:
            by_value[getattr(fact, attribute)].append(fact)
        for value, group in sorted(by_value.items()):
            quality = _quality(group)
            rows.append(
                _row(day, f"runs_by_{dimension}", count=sum(f.runs_completed for f in group),
                     user_state=value, outcome=quality, updated_at=updated_at)
            )
            rows.append(
                _row(day, f"operator_cost_by_{dimension}", count=len(group),
                     sum_micro=sum(f.operator_cost_micro for f in group),
                     user_state=value, outcome=quality, updated_at=updated_at)
            )
            rows.append(
                _row(day, f"own_spend_by_{dimension}", count=len(group),
                     sum_micro=sum(f.own_spend_micro for f in group),
                     user_state=value, outcome=quality, updated_at=updated_at)
            )

    rows.sort(key=lambda row: (row.metric_name, row.event_name, row.user_state))
    store.replace_lifecycle_rollups(day, rows, replace_transitions=True)
    return tuple(rows)
```

Drop `UserLifecycleDailySnapshot` and `ValueAnalyticsStore` from the `rollups.py` import (line 19-23) — `store` is typed `Any` because either twin is passed. `replace_transitions=True` always: transitions now come from their own table for `day`, so there is no "previous day missing" case to protect against.

- [ ] **Step 5: `retention.py`**

Replace the `if self.value_store is not None:` block (81-109) with:

```python
        if self.value_store is not None:
            lifecycle_before = current.date() - timedelta(days=RAW_EVENT_RETENTION_DAYS)
            expiring_days = self.value_store.list_expiring_fact_dates(
                before=lifecycle_before,
                limit=min(self.batch_size, self.max_batches),
            )
            # Aggregate every expiring day before deleting any of it (design §11:
            # the anonymous rollups are what outlives the 180-day horizon).
            for day in expiring_days:
                rollup_lifecycle_day(day, store=self.value_store)
            for day in expiring_days:
                lifecycle_deleted += self.value_store.delete_facts_for_date(day)
                lifecycle_deleted += self.value_store.delete_transitions_for_date(day)
            has_more_lifecycle = self.value_store.has_facts_before(lifecycle_before)
```

Change the `value_store` annotation in `__init__` (line 33) to `Any | None` (add `from typing import Any`). `user_activity` is never touched here — the new test pins it.

- [ ] **Step 6: Run and confirm the pass**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_rollups.py dashboard/backend/tests/domain/analytics/test_retention.py dashboard/backend/tests/test_store_twin_parity.py -v
```

Expected: **PASS**. `test_sqlite_retention_is_bounded_and_preserves_aggregates` still passes (it constructs the service without a `value_store`).

- [ ] **Step 7: Commit**

```bash
git add dashboard/backend/domain/analytics/value_repository.py \
        dashboard/backend/domain/analytics/value_repository_postgres.py \
        dashboard/backend/domain/analytics/rollups.py \
        dashboard/backend/domain/analytics/retention.py \
        dashboard/backend/tests/domain/analytics/test_rollups.py \
        dashboard/backend/tests/domain/analytics/test_retention.py
git status --short
git commit -m "$(cat <<'EOF'
feat: keep tier and user-group totals past expiry; expire facts and transitions

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: Re-seed `user_activity`, then drop the two snapshot tables on both twins

**Files:**
- Modify: `dashboard/backend/domain/analytics/repository.py` — `ANALYTICS_SQLITE_DDL`: delete the `user_analytics_snapshots` and `user_lifecycle_daily_snapshots` blocks (between `analytics_daily_rollups` and `analytics_projection_jobs`); `_init_schema` (line 262-266) and `_migrate_value_columns` (279-306)
- Modify: `dashboard/backend/domain/analytics/repository_postgres.py` — `ANALYTICS_POSTGRES_DDL`: the same two blocks; `_init_schema` (lines 201-228, the `ADD COLUMN IF NOT EXISTS` loop)
- Modify: `dashboard/backend/domain/analytics/value_repository.py` — delete `UserValueSnapshot`, `UserLifecycleDailySnapshot`, `_legacy_seed`, `upsert_current_snapshot`, `get_current_snapshot`, `_current_snapshot_from_row`, `list_current_snapshots`, `upsert_daily_snapshot`, `list_daily_snapshots`, `list_expiring_daily_dates`, `delete_daily_snapshots_for_date`, `has_daily_before`
- Modify: `dashboard/backend/domain/analytics/value_repository_postgres.py` — the same method set
- Create: `dashboard/backend/domain/analytics/facts_migration.py` gains `drop_legacy_snapshot_tables(*, store) -> None` (PR A created this module for the history copy and the seed; if PR A named it differently, add the function beside the seed wherever it lives)
- Modify: `dashboard/backend/app.py` — the startup block that runs PR A's history copy/seed gains the re-seed-then-drop call
- Test: `dashboard/backend/tests/domain/analytics/test_facts_migration.py` (extend; PR A created it), `test_repository_contract.py` (`test_sqlite_schema_contains_all_foundation_tables` 344, `test_sqlite_schema_adds_user_value_projection_storage` 362, `test_sqlite_user_value_migration_is_idempotent` 394, `test_sqlite_migrates_existing_legacy_snapshot_table` 410), `test_repository_postgres.py` (`test_postgres_ddl_declares_user_value_projection_storage` 168, `test_postgres_user_value_projection_round_trip` 214), `test_value_repository.py` (168-265), `test_retention.py` (231-238, 266-268), `test_value_repository_postgres.py` (PR T; snapshot round-trip cases)

**Interfaces:**
- Consumes: `seed_user_activity_from_snapshots(*, value_store) -> int` (PR A, idempotent: `MIN` on `activated_at`, `MAX` on `last_meaningful_activity_at`).
- Produces: `drop_legacy_snapshot_tables(*, store) -> None` — `DROP TABLE IF EXISTS user_lifecycle_daily_snapshots; DROP TABLE IF EXISTS user_analytics_snapshots;` on the analytics base store's own connection (either twin). Idempotent. Runs at startup **after** the seed, once per boot.

§11 / §12 item 6: *PR A seeds on creation, PR B re-seeds before the drop, with a test.* The legacy snapshot row kept being maintained by the reaper's throttled repair between PR A and PR B, so the seed must run once more against the final row before the table goes. Both statements are `IF EXISTS`, so the second boot after this deploy is a no-op — and so is a fresh install whose DDL never created the tables. **Running the suite (or any bare backend import) after this task drops those two tables from `dashboard/storage/data/backtest.db`; do not stage that file.**

- [ ] **Step 1: Write the failing tests**

Add to `dashboard/backend/tests/domain/analytics/test_facts_migration.py`:

```python
def test_reseed_then_drop_carries_activation_across_and_removes_the_tables(tmp_path):
    """Design §11: the seed runs again immediately before the drop, because the
    legacy row was still being maintained between PR A and PR B."""
    import sqlite3

    from dashboard.backend.domain.analytics.facts_migration import (
        drop_legacy_snapshot_tables,
        seed_user_activity_from_snapshots,
    )
    from dashboard.backend.domain.analytics.repository import AnalyticsStore
    from dashboard.backend.domain.analytics.value_repository import ValueAnalyticsStore
    from dashboard.backend.users import UserStore

    db_path = tmp_path / "reseed.db"
    users = UserStore(db_path=db_path)
    user_id = int(users.create_user("seed@example.test", "Seed", "SecurePass1!")["id"])
    store = AnalyticsStore(db_path=db_path)
    value_store = ValueAnalyticsStore(
        store, credits_base=object(), provider_base=object(), agent_base=object(), run_base=object()
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_analytics_snapshots (
                user_id INTEGER PRIMARY KEY, status TEXT NOT NULL, reason_code TEXT NOT NULL,
                human_readable_reason TEXT NOT NULL, evidence_event_ids_json TEXT NOT NULL DEFAULT '[]',
                calculated_at TEXT NOT NULL, activated_at TEXT, last_meaningful_activity_at TEXT
            )
            """
        )
        conn.execute("CREATE TABLE IF NOT EXISTS user_lifecycle_daily_snapshots (snapshot_date TEXT, user_id INTEGER)")
        conn.execute(
            "INSERT INTO user_analytics_snapshots (user_id, status, reason_code, human_readable_reason, calculated_at, activated_at, last_meaningful_activity_at) VALUES (?, 'active', 'x', 'x', ?, ?, ?)",
            (user_id, "2026-09-14T00:00:00+00:00", "2026-03-01T09:00:00+00:00", "2026-09-13T18:00:00+00:00"),
        )

    seeded = seed_user_activity_from_snapshots(value_store=value_store)
    drop_legacy_snapshot_tables(store=store)
    drop_legacy_snapshot_tables(store=store)  # idempotent

    assert seeded >= 1
    activity = value_store.get_activity(user_id)
    assert activity.activated_at.isoformat() == "2026-03-01T09:00:00+00:00"
    assert activity.last_meaningful_activity_at.isoformat() == "2026-09-13T18:00:00+00:00"
    with sqlite3.connect(db_path) as conn:
        names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert "user_analytics_snapshots" not in names
    assert "user_lifecycle_daily_snapshots" not in names


def test_fresh_schema_declares_no_snapshot_tables():
    from dashboard.backend.domain.analytics import repository, repository_postgres

    for ddl in (repository.ANALYTICS_SQLITE_DDL, repository_postgres.ANALYTICS_POSTGRES_DDL):
        assert "user_analytics_snapshots" not in ddl
        assert "user_lifecycle_daily_snapshots" not in ddl
```

The first test creates the legacy table by hand because after this task the DDL no longer does — that is the point of the second test. `seed_user_activity_from_snapshots` must tolerate the legacy table being absent (return 0) so a fresh install boots; if PR A's version raises on a missing table, wrap its `SELECT` in the `sqlite3.OperationalError` / `psycopg.errors.UndefinedTable` guard shown in Step 3.

Then edit the existing tests:
- `test_repository_contract.py:344-360` — remove `"user_analytics_snapshots"` from the expected set. Delete `test_sqlite_schema_adds_user_value_projection_storage` (362-392) and replace with `assert {"user_daily_facts", "user_activity", "lifecycle_transitions", "analytics_projection_jobs"} <= names` in a test named `test_sqlite_schema_declares_the_fact_tables` (PR A's contract test may already assert this; if so, delete rather than duplicate). Delete `test_sqlite_user_value_migration_is_idempotent` (394-408) and `test_sqlite_migrates_existing_legacy_snapshot_table` (410-447).
- `test_repository_postgres.py:168-185` — replace the body of `test_postgres_ddl_declares_user_value_projection_storage` with assertions that `user_daily_facts`, `user_activity`, `lifecycle_transitions` and `analytics_projection_jobs` are declared and that `ADD COLUMN IF NOT EXISTS user_analytics_snapshots` no longer appears in `inspect.getsource(PostgresAnalyticsStore._init_schema)`; delete `test_postgres_user_value_projection_round_trip` (214-…) and the `UserLifecycleDailySnapshot`/`_value_snapshot` imports it needed.
- `test_value_repository.py` — delete `test_current_projection_round_trips_without_overwriting_legacy_state` (168), `test_current_projections_are_loaded_in_one_batched_query` (193), `test_daily_snapshot_and_projection_job_upserts_are_idempotent` (217; re-add its `ProjectionJob` half as `test_projection_job_upsert_is_idempotent` keeping only the `save_projection_job`/`get_projection_job` assertions), `_value_snapshot` (126) and the `states`/`UserLifecycleDailySnapshot`/`UserValueSnapshot` imports.
- `test_retention.py:231-238` — delete the `INSERT INTO user_analytics_snapshots` statement; `:266-268` — delete the `SELECT status FROM user_analytics_snapshots` assertion.
- `test_value_repository_postgres.py` (PR T) — delete every case that upserts or reads a current or daily snapshot; keep the projection-job and commercial-value cases.

- [ ] **Step 2: Run and confirm the expected failure**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_facts_migration.py -v -k "reseed or fresh_schema"
```

Expected: **FAIL** — `ImportError: cannot import name 'drop_legacy_snapshot_tables'`; `fresh_schema` fails on the SQLite DDL string.

- [ ] **Step 3: Implement**

In `facts_migration.py` add:

```python
_LEGACY_SNAPSHOT_TABLES = ("user_lifecycle_daily_snapshots", "user_analytics_snapshots")


def drop_legacy_snapshot_tables(*, store: Any) -> None:
    """Drop the five-state snapshot tables (design §11, §13 row B).

    IF EXISTS on both, so a second boot and a fresh install are no-ops. Run
    only after seed_user_activity_from_snapshots() has read the final row.
    """
    postgres = hasattr(store, "database_url")
    with store._get_connection() as conn:
        for table in _LEGACY_SNAPSHOT_TABLES:
            statement = f"DROP TABLE IF EXISTS {table}"
            if postgres:
                with conn.cursor() as cur:
                    cur.execute(statement)
            else:
                conn.execute(statement)
```

and, if PR A's `seed_user_activity_from_snapshots` does not already guard the missing table, wrap its `SELECT ... FROM user_analytics_snapshots` so that `sqlite3.OperationalError` (message containing `no such table`) and psycopg's `UndefinedTable` return `0`:

```python
    try:
        rows = ...  # the existing SELECT
    except Exception as exc:  # the table is gone after PR B's drop; nothing to seed
        if "no such table" in str(exc).lower() or type(exc).__name__ == "UndefinedTable":
            return 0
        raise
```

(`store` here is the analytics base store — `hasattr(store, "database_url")` is the same dialect probe `repository.py::_build_analytics_store` relies on; this module is registered in PR T's `_DIALECT_BRANCH_ALLOWLIST` in Task 17 if the absence-direction check flags it.)

In `repository.py`: delete the two `CREATE TABLE` blocks and the two `CREATE INDEX idx_lifecycle_daily_*` statements from `ANALYTICS_SQLITE_DDL`; delete `_migrate_value_columns` (279-306) and its call in `_init_schema` (line 264). In `repository_postgres.py`: delete the same two blocks and indexes from `ANALYTICS_POSTGRES_DDL`, and the entire `for column, definition in {...}.items(): cur.execute("ALTER TABLE user_analytics_snapshots ADD COLUMN IF NOT EXISTS ...")` loop (lines 210-228) from `_init_schema`. `tests/test_store_twin_parity.py::test_postgres_twin_repeats_every_sqlite_lazy_migration` stays green because both sides lose the same migration.

In `value_repository.py` and `value_repository_postgres.py`: delete the models and methods listed in **Files** above, `_legacy_seed`, and their `__all__` entries. `ProjectionJob`, `get_projection_job`, `save_projection_job`, `claim_projection_day` (PR A), `list_commercial_values`, `list_credit_activity`, `get_operational_facts`, `_run_health` and every Task 2/3/13 method stay.

In `app.py`, in the startup block PR A added for the history copy and seed (the `try:` beside the analytics retention registration at lines 262-274), add after the seed call:

```python
        from dashboard.backend.domain.analytics.facts_migration import (
            drop_legacy_snapshot_tables,
        )
        drop_legacy_snapshot_tables(store=analytics_store)
        print("🧹 Analytics legacy snapshot tables dropped (if present)")
```

inside the same `try`, so a failure prints the block's existing `WARNING: analytics.facts_migration_failed category=...` line and never blocks boot.

- [ ] **Step 4: Run and confirm the pass**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/ dashboard/backend/tests/test_store_twin_parity.py dashboard/backend/tests/test_admin_analytics_api.py -q
git status --short
```

Expected: **PASS**; `git status` shows `M dashboard/storage/data/backtest.db` — **do not stage it**. `test_states.py`, `test_lifecycle_backfill.py` and `test_analytics_maintenance.py` fail at this point because they write to the dropped tables; Task 15 deletes them, so run this task's check with `--deselect` on those three files or accept the red until Task 15 (they are the next commit).

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/analytics/repository.py \
        dashboard/backend/domain/analytics/repository_postgres.py \
        dashboard/backend/domain/analytics/value_repository.py \
        dashboard/backend/domain/analytics/value_repository_postgres.py \
        dashboard/backend/domain/analytics/facts_migration.py \
        dashboard/backend/app.py \
        dashboard/backend/tests/domain/analytics/test_facts_migration.py \
        dashboard/backend/tests/domain/analytics/test_repository_contract.py \
        dashboard/backend/tests/domain/analytics/test_repository_postgres.py \
        dashboard/backend/tests/domain/analytics/test_value_repository.py \
        dashboard/backend/tests/domain/analytics/test_value_repository_postgres.py \
        dashboard/backend/tests/domain/analytics/test_retention.py
git status --short
git commit -m "$(cat <<'EOF'
refactor: re-seed user_activity and drop the five-state snapshot tables

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 15: Delete `states.py`, `lifecycle_backfill.py`, `maintenance.py`, their tests, the reaper registration and every remaining importer

**Files:**
- Delete: `dashboard/backend/domain/analytics/states.py`, `dashboard/backend/domain/analytics/lifecycle_backfill.py`, `dashboard/backend/domain/analytics/maintenance.py`
- Delete: `dashboard/backend/tests/domain/analytics/test_states.py`, `dashboard/backend/tests/domain/analytics/test_lifecycle_backfill.py`, `dashboard/backend/tests/test_analytics_maintenance.py`
- Modify: `dashboard/backend/app.py` — lines 276-288 (the `run_analytics_maintenance` registration block)
- Modify: `dashboard/backend/domain/analytics/backfill.py` — `backfill_analytics` (565-717): the `recalculate_snapshot` parameter and leg; `BackfillReport.repaired_snapshots` (line 110)
- Modify: `dashboard/backend/domain/analytics/rollups.py` — `rollup_day` (232-445): the `state_counts` parameter and the `user_state_count` rows
- Modify: `dashboard/backend/domain/analytics/query_service.py` — the `.states` import (28-32), `_USER_STATES` (35), `_ATTENTION_STATES` (36), `AnalyticsQueryStore.list_snapshots` (287-311), `summarize_users` (409-462), `AnalyticsQueryStore.states` (284), `_snapshot_summary` (606-613), `_all_users` (267-275) if unreferenced
- Modify: `dashboard/backend/tests/test_admin_analytics_api.py` (35-38, 213, 325, 477-478), `test_repository_contract.py` (27-30, 280-281), `test_backfill.py` (the six `recalculate_snapshot=` call sites and lines 234-285), `test_analytics_integration.py` (27-31, 214-244) — whatever PR A left of the `states` imports

**Interfaces:**
- Consumes: nothing.
- Produces: `backfill_analytics(*, now=None, days=180, before=None, dry_run=False, source=None, store=analytics_store, rebuild_rollup=None)` — `recalculate_snapshot` is gone; `BackfillReport` loses `repaired_snapshots`. `rollup_day(day, *, store=None, include_internal=False, now=None)` — `state_counts` is gone. `AnalyticsQueryStore` loses `states`, `list_snapshots`, `summarize_users`.

Design §13 row B lists exactly these deletions; §4.3 names `repair_stale_snapshots` / `repair_stale_value_snapshots` (in `states.py`, called from `maintenance.py`) as one of the two outage burners — this task removes the code, not just the throttle PR 0 applied. `rollup_day`'s five-state rows (`rollups.py:421-430`) go because their only producer was `state_counts` from the snapshot table, which Task 14 dropped; `rollup_day` itself stays (PR A's daily job calls it).

- [ ] **Step 1: Write the failing test**

Add to `dashboard/backend/tests/test_architecture_boundaries.py`, beside `_DELETED_SHIMS`:

```python
_DELETED_ANALYTICS_MODULES = (
    "dashboard.backend.domain.analytics.states",
    "dashboard.backend.domain.analytics.lifecycle_backfill",
    "dashboard.backend.domain.analytics.maintenance",
)


@pytest.mark.parametrize("module_name", _DELETED_ANALYTICS_MODULES)
def test_deleted_five_state_module_is_not_importable(module_name):
    """Design §13 row B: the five-state model leaves in PR B. A stale
    __pycache__ resolves a deleted package dir as a namespace package, which
    is why _DELETED_SHIMS exists; these are plain modules, so a stale .pyc is
    the only way this can pass by accident -- see the CLAUDE.md gotcha."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_app_no_longer_registers_analytics_maintenance_with_the_reaper():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "run_analytics_maintenance" not in source
    assert "analytics.maintenance_registration_failed" not in source


def test_rollup_day_writes_no_five_state_rows():
    import inspect

    from dashboard.backend.domain.analytics import rollups

    assert "user_state_count" not in inspect.getsource(rollups.rollup_day)
    assert "state_counts" not in inspect.signature(rollups.rollup_day).parameters
```

(`importlib`, `pytest`, `Path` are already imported in that module for `_DELETED_SHIMS`.)

- [ ] **Step 2: Run and confirm the expected failure**

```bash
python -m pytest dashboard/backend/tests/test_architecture_boundaries.py -v -k "five_state or maintenance_with_the_reaper or no_five_state_rows"
```

Expected: **FAIL** — the three modules import, `app.py` still contains the registration, `rollup_day` still writes `user_state_count`.

- [ ] **Step 3: Delete, in this order, keeping the suite importable between steps**

1. `git rm dashboard/backend/domain/analytics/states.py dashboard/backend/domain/analytics/lifecycle_backfill.py dashboard/backend/domain/analytics/maintenance.py dashboard/backend/tests/domain/analytics/test_states.py dashboard/backend/tests/domain/analytics/test_lifecycle_backfill.py dashboard/backend/tests/test_analytics_maintenance.py`
2. `app.py:276-288` — delete the whole `try:` block that imports and registers `run_analytics_maintenance`.
3. `rollups.py::rollup_day` — remove `state_counts: Mapping[str, int] | None = None` from the signature (line 236) and the `for status, count in sorted((state_counts or {}).items()): rows.append(_row(day, "user_state_count", ...))` loop (421-430); drop `Mapping` from the `typing` import if unused.
4. `backfill.py` — remove `recalculate_snapshot: Any | None = None` (573), the default-builder block (657-663: `if recalculate_snapshot is None: from .states import ...`), the `repaired_snapshots` counter and the `for user_id in sorted(affected_users): recalculate_snapshot(...)` loop (677-684), `repaired_snapshots=repaired_snapshots` in the report (708), and `repaired_snapshots: int = Field(default=0, ge=0)` from `BackfillReport` (110). Ingestion maintains `user_activity` and the daily job's late-arrival recompute (§6.9 step 7) picks up backfilled days; there is no per-user repair to run.
5. `query_service.py` — delete the `from .states import (...)` block (28-32), `_USER_STATES` (35), `_ATTENTION_STATES` (36), `self.states = AnalyticsStateStore(base_store)` (284), `list_snapshots` (287-311), `summarize_users` (409-462), `_snapshot_summary` (606-613); delete `_all_users` (267-275) and `AnalyticsStateSummary`'s only remaining constructor is Task 11's profile — keep the model. `_legacy_state` stays until Task 16.
6. Run `rg -n "analytics\.states|from \.states|lifecycle_backfill|run_analytics_maintenance|recalculate_snapshot|repaired_snapshots|state_counts=" dashboard/` and fix every hit:
   - `test_admin_analytics_api.py:35-38` — delete the `states` import; `:213` (`AnalyticsStateStore(analytics)` in `_fixture`'s return tuple) — return `None` in that slot so the five-tuple unpacking in every test stays valid; `:325` (`recalculate_user_snapshot(1, ...)`) was removed with the test in Task 11; `:477-478` (`state_store = AnalyticsStateStore(analytics)` / `recalculate_user_snapshot(subject["id"], ...)` in the `admin_analytics_api` fixture) — delete both lines.
   - `test_repository_contract.py:27-30, 280-281` — delete the import and the two lines; `profile.state.status == "active"` (line 335) now comes from Task 11's `_legacy_state` over live axes: with a success at `NOW - 1h` and no fact rows, the live segment is `growing` and operational `healthy` → `"active"`, so the assertion holds; `completed.event_id in profile.state.evidence_event_ids` (336) is deleted (the list is empty by design, see Task 11).
   - `test_backfill.py` — delete `recalculate_snapshot=lambda ...` from the six calls (147, 174, 182, 223, 269, 276) and the `snapshots` list and `assert snapshots == [1, 2]` (234, 285).
   - `test_analytics_integration.py:27-31, 214, 236-244` — if PR A left them (its B4 removed `instrumentation._snapshot_recalculator`, which this test monkeypatches), delete the import, `state_store = AnalyticsStateStore(store)` and the `monkeypatch.setattr(instrumentation, "_snapshot_recalculator", ...)` call; the scenario's later assertions on the overview and profile responses hold from facts written by `value_store.record_activity` via the service.
7. `rm -rf dashboard/backend/domain/analytics/__pycache__` before running, so a stale `states.cpython-*.pyc` cannot make the import test pass by accident.

- [ ] **Step 4: Run and confirm the pass**

```bash
python -m pytest dashboard/backend/tests/ -q
```

Expected: **PASS** across the whole backend suite (SQLite tier) — this is the first full-suite run since Task 14. If `tests/test_event_loop_threadpool.py` or `test_app_composition.py` complain, it is an import path left over from step 6; `rg` again.

- [ ] **Step 5: Commit**

```bash
git status --short
git add dashboard/backend/app.py \
        dashboard/backend/domain/analytics/backfill.py \
        dashboard/backend/domain/analytics/rollups.py \
        dashboard/backend/domain/analytics/query_service.py \
        dashboard/backend/tests/test_architecture_boundaries.py \
        dashboard/backend/tests/test_admin_analytics_api.py \
        dashboard/backend/tests/test_analytics_integration.py \
        dashboard/backend/tests/domain/analytics/test_repository_contract.py \
        dashboard/backend/tests/domain/analytics/test_backfill.py
git add -u -- dashboard/backend/domain/analytics dashboard/backend/tests
git status --short
git commit -m "$(cat <<'EOF'
refactor: retire the five-state snapshot model and its maintenance sweep

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

`git add -u -- <paths>` is scoped to the two directories so the deletions from step 1 are staged without reaching `dashboard/storage/`; the second `git status --short` must not list `backtest.db` as staged (`M ` in the first column). If it does, `git restore --staged dashboard/storage/data/backtest.db`.

---

### Task 16: D20 — `user_state_counts` leaves `/overview`, `status` leaves `/users`, end to end

**Files:**
- Modify: `dashboard/backend/domain/analytics/query_service.py` — `AnalyticsOverview.user_state_counts` (line 259), the friction bridge in `get_overview` (Task 5), `_legacy_state` (kept: Task 11's profile `state` still uses it)
- Modify: `dashboard/backend/domain/analytics/value_queries.py` — `UserValueFilters.legacy_status` (317) and its validator arm (344-352); the `legacy_status=` line in `list_users` (Task 10)
- Modify: `dashboard/backend/domain/analytics/value_repository.py` — `UserPageQuery.legacy_status`, `_LEGACY_STATUS_PREDICATES`, the `legacy_status` arm of `_users_page_sql` (Task 3)
- Modify: `dashboard/backend/api/routers/admin_analytics.py` — `_USER_STATES` (53), `"status"` in `_value_user_filters`'s allowed set (215), lines 247-249, `legacy_status=legacy_status` (284)
- Modify: `dashboard/backend/tests/fixtures/admin_analytics/overview.json` (the `user_state_counts` block, lines 27-33) and `overview_partial_error.json` (lines 17-23), **and their copies under `fixtures/admin_analytics/target/`** (PR C's `test_admin_target_fixtures.py` asserts the two sets differ only by the §9 fields)
- Modify: `dashboard/backend/tests/test_admin_analytics_api.py` (`"status": "active"` at 710; `filters.legacy_status == "active"` at 738), `tests/domain/analytics/test_fact_reads.py` (the legacy loop in `assert_users_page_contract`), `test_value_queries.py` (`legacy_status="active"` in `test_users_list_passes_every_filter_to_sql_unchanged`)

**Interfaces:**
- Consumes: nothing.
- Produces: `AnalyticsOverview` without `user_state_counts`; `UserValueFilters` and `UserPageQuery` without `legacy_status`; `GET /users?status=…` answers **422** like any other unknown parameter (`_query_values` rejects unlisted keys, `admin_analytics.py:67-73`). The friction panel's availability keeps its key (`availability["friction"]`) — it now reports the `count_states_for_date` read that `/operational` and `/lifecycle` share, so the availability map's five keys are unchanged.

D20: *A shim faking a dead vocabulary from a live one is worse than the break, because it makes the field look maintained.* Tasks 5 and 3 carried the bridge and the SQL translation only so that every intermediate commit served the full contract; this task removes both. It makes no frontend edit: the dead sender of `status` (`admin-analytics.js::attentionQuery`) went with the whole module when PR C deleted it (§7.5), before this branch was cut.

- [ ] **Step 1: Write the failing tests**

In `test_admin_analytics_api.py`, append:

```python
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
```

(add `UserValueFilters` to that module's `value_queries` import). In `test_admin_user_list_accepts_documented_filters` delete `"status": "active",` (710) and `assert filters.legacy_status == "active"` (738). In `test_fact_reads.py::assert_users_page_contract` delete the `for legacy, expected in (...)` loop and change `_page_query`'s defaults to drop `legacy_status`. In `test_value_queries.py::test_users_list_passes_every_filter_to_sql_unchanged` drop `legacy_status="active"` from both the `UserValueFilters(...)` and the expected `UserPageQuery(...)`. In `overview.json`, `overview_partial_error.json` **and** `target/overview.json`, `target/overview_partial_error.json` delete the `"user_state_counts": {...},` block (the target copies must lose it too, or `test_admin_target_fixtures.py::test_target_fixtures_add_exactly_the_section_9_fields` fails with `target differs from committed beyond the §9 fields`).

- [ ] **Step 2: Run and confirm the expected failure**

```bash
python -m pytest dashboard/backend/tests/test_admin_analytics_api.py dashboard/backend/tests/test_admin_analytics_frontend.py dashboard/backend/tests/test_admin_target_fixtures.py -q
```

Expected: **FAIL** — `overview.json()` still carries `user_state_counts`; `/users?status=active` answers 200. `test_admin_target_fixtures.py` passes (both sets lost the block together); `test_fixtures_validate_against_committed_analytics_models` passes because `user_state_counts` is still an optional-by-omission field on the fixture side — it fails only if the model gains a required field.

- [ ] **Step 3: Remove the vocabulary**

`query_service.py`: delete `user_state_counts: dict[str, int]` (259) and the `user_state_counts=state_counts,` line in `AnalyticsOverview(...)`; in `get_overview` replace the friction block

```python
        state_counts: dict[str, int] = {}
        try:
            for cell in self.value_store.count_states_for_date(yesterday):
                key = _legacy_state(cell.operational_state, cell.lifecycle_segment)
                state_counts[key] = state_counts.get(key, 0) + cell.users
        except Exception:
            availability["friction"] = _availability(False)
```

with

```python
        try:
            # The friction panel's availability now reports the fact read the
            # operational board is served from; the five-state counts are gone (D20).
            self.value_store.count_states_for_date(yesterday)
        except Exception:
            availability["friction"] = _availability(False)
```

`value_queries.py`: delete `legacy_status: str | None = None` (317) and the `legacy = self.legacy_status ... raise ValueError("legacy_status is unsupported")` arm (344-352) from `UserValueFilters`; delete `legacy_status=filters.legacy_status,` from `list_users`. `value_repository.py`: delete `legacy_status: str | None = None` from `UserPageQuery`, `_LEGACY_STATUS_PREDICATES`, and the `if query.legacy_status is not None:` arm of `_users_page_sql`. `admin_analytics.py`: delete `_USER_STATES` (53), `"status",` from the allowed set (215), lines 247-249, and `legacy_status=legacy_status,` (284) — `_USER_STATES`'s other reader, `_user_filters` (317), was deleted by PR A with the dead legacy list. No frontend edit: PR C already deleted `admin-analytics.js`, the only sender of `status`.

- [ ] **Step 4: Run and confirm the pass**

```bash
python -m pytest dashboard/backend/tests/ -q
rg -n "user_state_counts|legacy_status|_USER_STATES|user_state_count\b" dashboard/
```

Expected: **PASS**, and the `rg` prints nothing (`admin-analytics.js`, the one file that used to render `user_state_counts`, was deleted by PR C; no `/admin` module reads the field — §8.2 cuts it).

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/analytics/query_service.py \
        dashboard/backend/domain/analytics/value_queries.py \
        dashboard/backend/domain/analytics/value_repository.py \
        dashboard/backend/api/routers/admin_analytics.py \
        dashboard/backend/tests/fixtures/admin_analytics/overview.json \
        dashboard/backend/tests/fixtures/admin_analytics/overview_partial_error.json \
        dashboard/backend/tests/fixtures/admin_analytics/target/overview.json \
        dashboard/backend/tests/fixtures/admin_analytics/target/overview_partial_error.json \
        dashboard/backend/tests/test_admin_analytics_api.py \
        dashboard/backend/tests/domain/analytics/test_fact_reads.py \
        dashboard/backend/tests/domain/analytics/test_value_queries.py
git status --short
git commit -m "$(cat <<'EOF'
refactor: drop the five-state vocabulary from the analytics contract

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 17: Allowlists left by PR T and PR A, `CLAUDE.md`, and the Postgres-tier run

**Files:**
- Modify: `dashboard/backend/tests/test_store_twin_parity.py` — `_DIALECT_BRANCH_ALLOWLIST` (PR T Task 4)
- Modify: `dashboard/backend/tests/test_architecture_boundaries.py` — PR A's event-log-discipline allowlist, if it names the deleted modules
- Modify: `CLAUDE.md` — the "Admin layer" pointer the docs PR added (and the Persistence bullet PR T extended) lose their mentions of the snapshot tables and `maintenance.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a green `test_store_twin_parity.py::test_dialect_branches_outside_a_registered_twin_are_allowlisted` — its second assertion fails on a **stale** entry ("names file(s) with no dialect-branch idiom left in them"), which is exactly the state Task 15 leaves `states.py` and `lifecycle_backfill.py` in.

- [ ] **Step 1: Run the guards and read the failure**

```bash
python -m pytest dashboard/backend/tests/test_store_twin_parity.py dashboard/backend/tests/test_architecture_boundaries.py -v
```

Expected: **FAIL** in `test_dialect_branches_outside_a_registered_twin_are_allowlisted` naming `dashboard/backend/domain/analytics/states.py` and `dashboard/backend/domain/analytics/lifecycle_backfill.py` as stale; possibly also an `unlisted` hit for `dashboard/backend/domain/analytics/facts_migration.py` if Task 14's `hasattr(store, "database_url")` probe is the first dialect branch in that module (PR A's history copy may already have put it on the list). If PR A's discipline test in `test_architecture_boundaries.py` allowlists `lifecycle_backfill.py` or `states.py` as legacy `list_events` callers, it fails the same way.

- [ ] **Step 2: Fix the allowlists**

In `_DIALECT_BRANCH_ALLOWLIST`, delete the `"dashboard/backend/domain/analytics/states.py"` and `"dashboard/backend/domain/analytics/lifecycle_backfill.py"` entries. Update the `query_service.py` entry's reason to the present tense: `"AnalyticsQueryStore dialect-branches over the already-twinned AnalyticsStore/PostgresAnalyticsStore base_store for list_metric_events (the overview's one-day scan), list_activity_rows and list_session_rows. Not a store of its own."` If `facts_migration.py` is reported unlisted, add:

```python
    "dashboard/backend/domain/analytics/facts_migration.py": (
        "drop_legacy_snapshot_tables (and PR A's seed) probe the already-twinned "
        "analytics base store's dialect to run one DROP TABLE IF EXISTS / one "
        "SELECT on each side. Not a store of its own; the tables it touches are "
        "gone after PR B, so this branch is startup-only and idempotent."
    ),
```

In `test_architecture_boundaries.py`, remove any allowlist entry naming the three deleted modules. In `CLAUDE.md`, under the Persistence bullet, replace the mention of `user_analytics_snapshots` / `user_lifecycle_daily_snapshots` (added by PR T's Task 5 or the docs PR) with one sentence: *Analytics keeps `user_activity` (never swept), `user_daily_facts` and `lifecycle_transitions` (180 days, rolled up before expiry) and the anonymous `analytics_daily_rollups` (indefinite); the five-state snapshot tables and `domain/analytics/maintenance.py` were deleted in PR B (design doc §11, §13).*

- [ ] **Step 3: Run everything, both tiers**

```bash
rm -rf dashboard/backend/domain/analytics/__pycache__
python -m pytest dashboard/backend/tests/ -q
TEST_POSTGRES_URL=postgresql://postgres:test@localhost:5432/atl_test python -m pytest dashboard/backend/tests/ -q
git status --short
```

Expected: **PASS** on both, with a lower skip count on the second; `git status` shows `M dashboard/storage/data/backtest.db` (unstaged, left alone) and the three files above.

- [ ] **Step 4: Commit and open the PR as a draft**

```bash
git add dashboard/backend/tests/test_store_twin_parity.py dashboard/backend/tests/test_architecture_boundaries.py CLAUDE.md
git status --short
git commit -m "$(cat <<'EOF'
test: retire the allowlist entries for the deleted analytics modules

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

Open the PR as a **draft** titled `refactor: move admin analytics onto daily facts and drop the snapshot model`, with the first line of the body: **DO NOT MERGE until the "Merge preconditions" queries in `docs/superpowers/plans/2026-09-15-admin-layer-redesign-prB-read-paths-delete.md` pass against prod (≥ 8 complete days of `user_daily_facts`).** The body then lists the D20 items and the "Acceptance" table's value-change rows in one line each — reviewers read the diff, not an essay.

---

## Not in this PR

From the design doc's §13 row B "Must not" column and the decisions behind it:

- **Re-cut the contract.** No field is added to any response model (`billing_lane_mix`, `top_operational_reasons`, `purchased_by_day`, `consumed_by_day`, `group_badge`, `user_group`, `role`, `last_meaningful_activity_at`, `facts_as_of` all wait for PR D — D19 as amended 2026-09-16: the page (PR C) already renders them from target-shape fixtures and shows "Awaiting data source" until this PR's service methods reach the payload in PR D); no query parameter is renamed or declared as a FastAPI parameter (`_query_values()` stays until PR D). `max_active_dashboard_backtests` on `GET /api/admin/stats` is PR C's, already merged.
- **Touch the frontend.** No change to any file under `dashboard/frontend/`. The `/admin` page's four modules read the nine routes under today's names and never read `user_state_counts` (§8.2 cuts it), so D20 reaches the page as a fixture edit only.
- **Build on the in-app panel.** §12 item 3: the 09-12 plan's task C6 targeted a surface #467 made unreachable; PR C was written from scratch against §7-§8 and has shipped. Nothing here adds axis filters, a group badge or a cohort field anywhere.
- **`cohort`** in any form (D9): no column, no validator, no filter, no `cohort_*` rollup names.
- **Delete `rollup_day`, `analytics_daily_rollups`, or `AnalyticsRollupStore`.** The rollups are the permanent table (§11); this PR adds rows to them.
- **Sweep `user_activity`** (§11): the retention service never touches it, and Task 13 pins that.
- **Change the daily job, the day claim, `owner_user_id`, or ingestion** — PR A's surface. This PR only reads what PR A writes, plus the two rollup families it asks `rollup_day` to add.
- **A read-budget or wall-clock test for the daily job** — PR A owns `test_read_budget.py`; the reads added here are page-bounded route reads, covered by the one-statement-per-call test in Task 2.
- **Port Users / Providers / Activity into `/admin`**, compute the badge client-side, add inline scripts or a `#live` route — PR C's "Must not" column, listed so a reader of this plan does not go looking for them here.

## Acceptance

Mirrors design §6.10 ("After PR B"), §9, §10.2, §11, §13 row B.

- [ ] `GET /overview`: rollups for completed days plus exactly one raw-event read bounded to the current UTC day (Task 5's `_WindowRecordingQueryStore` tests); `top_failure_categories` from rollups `error_category` (D13); `users_needing_attention` shape unchanged, sourced from `operational_state IN ('blocked','needs_attention')` (D20); `user_state_counts` absent (D20).
- [ ] `GET /lifecycle`: `segment_counts`/`core_users`/`at_risk_users` from yesterday's facts; movement and transitions from facts + `lifecycle_transitions` (rollup fallback for older days); `activated_users` from `user_activity`; `paid_users` from the ledger; no `list_events` call.
- [ ] `GET /retention`: cohorts from `user_activity.activated_at`, cells from `user_daily_facts.active`; immature cells `None`; no `list_events` call.
- [ ] `GET /commercial`: unchanged (ledger); `purchased_by_day` / `consumed_by_day` exist as service methods only.
- [ ] `GET /operational`: `operational_state_counts` from yesterday's facts; `top_operational_reasons` exists as a service method only.
- [ ] `GET /groups`: users from `users`, runs and operator cost from facts grouped by `user_group`, `paid_users` from the ledger; no `list_events` call.
- [ ] `GET /users`: filtered, sorted and paginated in SQL on both twins (`list_users_page`); `status` answers 422 (D20); live segment per page row; `operational`, `commercial_tier`, `priority_group` as of yesterday.
- [ ] `GET /users/{id}`: live segment and operational state; 7-/30-day sums from facts; one raw-event read (the first timeline page); `state` re-sourced from the live axes.
- [ ] `GET /users/{id}/activity`: `sessions` bounded to 30 days; other sections unchanged.
- [ ] Service methods, unit-tested, on no response model: `billing_lane_mix`, `top_operational_reasons`, `purchased_by_day`, `consumed_by_day`, `resolve_group_badge`, `get_user_metrics` with the §6.13 audience filter.
- [ ] Long-term rollups: `runs_by_tier`, `runs_by_user_group`, `operator_cost_by_tier`, `operator_cost_by_user_group`, `own_spend_by_tier`, `own_spend_by_user_group` written by `rollup_lifecycle_day` before facts and transitions expire; `user_state_count` rows no longer written; no DDL change to `analytics_daily_rollups`.
- [ ] Deletions: `user_activity` re-seeded then `user_lifecycle_daily_snapshots` and `user_analytics_snapshots` dropped on both twins; `states.py`, `lifecycle_backfill.py`, `maintenance.py` and their tests deleted; the reaper registration removed from `app.py`; `_USER_STATES` gone from the router and the service; allowlists updated.
- [ ] Both twins green: `test_store_twin_parity.py` and the `@pg_only` tier on `TEST_POSTGRES_URL` pass in CI before merge.
- [ ] Merge preconditions verified against prod and the draft flag lifted only then.

**Assertions in the conformance oracle that this plan changes, and why** (design §10.2 says the three test files are edited only for the two D20 items; the list below is longer because the fixtures those files built on — snapshot rows and full-window event scans — no longer exist, so their *seeding* had to move even where the *assertion* did not):

| File : test | Change | Why |
|---|---|---|
| `test_admin_analytics_api.py` : `test_overview_marks_only_failed_rollup_panel_unavailable` | renamed; `active_users_7d == 1` → `is None`; `funnel` now unavailable too | `active_users_7d`, conversion, funnel are rollup-sourced (§6.10) |
| same : `test_user_list_and_profile_are_display_safe` | replaced by `test_profile_is_display_safe_and_reads_facts_not_the_event_window`; drops `evidence_event_ids` membership | the legacy `list_users` went in PR A (D24); `evidence_event_ids` has no source without `calculate_user_state` |
| same : `test_admin_user_list_accepts_documented_filters` | `status` param and `legacy_status` assertion removed | D20 |
| same : fixture `admin_analytics_api` and `_fixture` | no `AnalyticsStateStore` / `recalculate_user_snapshot` | `states.py` deleted |
| same : new tests | one-day scan window; rollup-sourced trailing metrics; `billing_lane_mix`; 30-day sessions; D20 shape check | §6.15 items 2-3, D13, D14, D20 |
| `test_admin_analytics_frontend.py` (as PR C rewrote it) | **no change** | its shape test asserts today's keys, none of them `user_state_counts`; PR C removed the old `test_client_uses_exact_pr2_endpoints_and_query_names` with the module it pinned |
| `fixtures/admin_analytics/overview.json`, `overview_partial_error.json`, and their `target/` copies | `user_state_counts` block removed | D20; the target copies follow so `test_admin_target_fixtures.py` keeps the two sets in step |
| the `/admin` node tests (`test_admin_*_frontend.py`, `test_admin_page_modules.py`) | **no change** | no `/admin` module reads `user_state_counts` (§8.2 cut); the renderers are fed the fixtures above and paint the same |
| `test_repository_contract.py` : `assert_pr2_query_contract` | attention seeded via `upsert_daily_facts`; `evidence_event_ids` assertion removed; `states` import gone | snapshot table dropped |

Value changes with no shape change, for the PR body (one suspect each, D18): trailing overview metrics are as of yesterday; funnel and failure counts sum daily distinct users; `include_internal` no longer adds admins to fact-sourced counts; profile tokens, footprint, provider and client-context fields are computed over the latest 100-event timeline page; per-user `recent_lifecycle_transitions` carry their own day as the period; the users list's priority sort drops the lifetime-purchase tiebreak and lists users without a fact row as `healthy`/`unpaid`.

## Self-review notes

- **Spec coverage.** Each "After PR B" row of §6.10 has a task (5, 6, 7, 8-via-5, 8, 9, 10, 11, 12); §6.7's encoding and the six metric names are in Task 13 verbatim; §6.15 items 2 and 3 are Tasks 5 and 12; §9's "Added" table maps to Tasks 1, 4, 5, 8, 11 as service methods with a test in each asserting the field is *not* on the response model; §11's re-seed-then-drop, facts/transitions expiry and `user_activity` exception are Tasks 13-14 with tests; §12 items 3, 6, 7 are honoured by the task order (13 read-path tasks, then 4 deletion tasks); §13 row B's deletion list is Tasks 14-16 item by item; D13, D14, D20 each have a named test. The design's `activation_funnel`-from-rollups line needed a new `activation_users` rollup row (Task 5), which the design implies but does not spell out — the funnel's prod history is therefore short for 30 days after deploy, and that is recorded in Task 5's table rather than hidden.
- **Placeholder scan.** `rg -n "TBD|TODO|similar to Task|as in the old plan|appropriate handling|write tests for" <this file>` returns no hits. Every code step shows the code; the two places that say "the other eight follow the same two substitutions" (Task 2 Step 5) and "identical but `%s`" (Tasks 3, 13) name the exact substitutions and show one full instance each, which is the same convention PR T's plan uses for the twin.
- **Name consistency.** Store methods introduced in Task 2 (`count_states_for_date`, `count_segments_by_date`, `count_transitions`, `list_attention_candidates`, `sum_facts_by_user_group`, `list_active_dates`, `list_activations`, `count_activated_users`, `list_user_transitions`, `list_transitions_for_date`) are the names Tasks 5-13 call and the `FakeValueStore` in Task 6 implements; Task 3's `UserPageQuery`/`UserPageRow`/`list_users_page` are what Task 10 uses and Task 16 trims; Task 1's `operational_reason`/`operational_result_from_code`/`resolve_group_badge` appear in Tasks 5, 8, 10, 11; Task 4's `LedgerDayTotal`/`sum_ledger_by_day`/`credits_reader` are used only within Task 4; Task 13's `list_expiring_fact_dates`/`delete_facts_for_date`/`delete_transitions_for_date`/`has_facts_before` replace the four snapshot-table methods Task 14 deletes. PR A / PR T names are consolidated in one table at the top with the `rg` command that verifies them.
- **Ordering.** No task before 14 removes a table or module; Task 14 drops tables only after Tasks 5-13 have moved every reader; Task 15 deletes modules only after Task 14 removed the tables they wrote; Task 16 removes the D20 vocabulary only after Tasks 3 and 5 carried it through the switch; Task 17 fixes the guards that Tasks 14-15 made stale.
- **User-facing docs.** `rg -l "/api/admin/analytics|admin analytics" docs/ --glob '!docs/superpowers/**'` returns nothing: the published tree has no admin analytics page (design §14), so this PR leaves no user-facing doc stale. The one documentation touch is `CLAUDE.md` (Task 17).
- **Open items carried to the StructuredOutput for the orchestrator**: PR A names assumed here (see the table), `facts_migration.py`'s seed function name, whether PR A's `test_analytics_integration.py` edit already removed the `states` import, and the page-bounded token semantics on the profile.
