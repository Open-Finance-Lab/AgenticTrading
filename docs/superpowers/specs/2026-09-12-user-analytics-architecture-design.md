# User Analytics Architecture Design

**Date:** 2026-09-12
**Status:** Design approved in discussion; written spec pending review
**Delivery:** Three sequential pull requests (A, B, C)

## Summary

Admin user analytics keeps its product surface and changes its data
architecture. Per-user facts are derived from the authoritative tables that
already hold them, the event log is read only in bounded one-day or paginated
windows, lifecycle labels are computed at read time from stored timestamps, and
all cross-user aggregates come from one daily fact table written once per UTC
day. The periodic per-user recompute sweep is removed.

Users are described on three orthogonal axes: **role** (permission), **tier**
(derived from the credit ledger), and **cohort** (an admin-assigned label). The
design leaves defined hooks for automatic outreach, a user-facing usage page,
a subscription system, and a read-only role, without building any of them.

This design extends the collection, privacy, and retention contracts in
`docs/superpowers/specs/2026-08-26-admin-user-analytics-design.md` and the
lifecycle definitions in
`docs/superpowers/specs/2026-09-03-admin-user-value-analytics-design.md`. It
supersedes both where they define snapshot storage, freshness, the repair pass,
and the automatic-outreach non-goal. The Supersession section lists each
overridden clause.

## Why now

On 2026-09-11 every route on the users/content Postgres project, including
login, returned 500 for about five hours. The Neon free-tier egress quota had
been exhausted by `run_analytics_maintenance`, which runs on the sixty-second
run-reaper tick and, for every non-admin user whose snapshot was older than
fifteen minutes, re-read that user's full 180-day event history two or three
times to recompute a label that changes at most once per day. Nothing in the
test suite pinned the cost of a tick, and `/health` stayed green throughout.

The mechanism, not the cadence, was the defect: labels were materialized
snapshots recomputed from a copy of the facts. This document replaces that
mechanism. A separate stopgap (PR A) throttles the existing sweep so prod is
safe while the replacement lands.

Prod now runs on Neon Launch, which bills compute hours rather than capping
egress. A sixty-second sweep keeps the compute awake permanently, so "touch the
database once a day for maintenance" remains the design target.

## Goals

1. Answer, for a user base of a few hundred accounts, which users are getting
   value, which are burning operator-funded model spend, which cohorts behave
   differently, and where the activation funnel leaks.
2. Make every per-user number agree with the page that owns it: runs with run
   history, spend with the Credits ledger, sessions with the auth store.
3. Bound the cost of analytics by construction and pin that bound with tests.
4. Give administrators three orthogonal axes to cut every chart and list:
   role, tier, cohort.
5. Leave a defined seam for automatic outreach, a self-service usage page, a
   subscription system, and a read-only role.

## Non-goals

- Building the outreach sender, templates, or opt-out surface. Only the
  transition log it will consume is in scope.
- The user-facing My Usage page. Only the field-visibility rule is in scope.
- A subscription system. Only the tier resolver seam is in scope.
- A `viewer` role. The supervisor uses an admin account.
- Durable storage for protocol runs (`protocol_runs` stays ephemeral SQLite).
- Changing the event allowlist, the 1 KB property cap, or 180-day raw
  retention.
- Sub-day cross-user aggregates.

## Consumers and decisions

Readers are administrators and the project supervisor, all on admin accounts.
The analytics must support, in priority order:

1. **Cost control.** Which users consume operator-funded model spend, so
   entitlements and grants can be adjusted.
2. **Product prioritization.** Which surfaces get used and where the
   signup-to-first-successful-backtest funnel leaks.
3. **Cohort management.** How lab members, paid users, and free users differ.
4. **Retention outreach (future).** An email after fourteen inactive days is
   the intended first rule. The sender is out of scope; the data it needs is
   not.

## Three axes

| Axis | Values today | Source | Writable by |
|---|---|---|---|
| role | `user`, `admin` | `users.role` | admin (existing) |
| tier | `unpaid`, `starter`, `invested`, `high_value` | `commercial_tier()` over the credit ledger | nobody; derived |
| cohort | free text slug, or none (`lab` first) | new `users.cohort` column | admin, from the Users tab |

Rules:

- A user has at most one cohort. Cohorts do not overlap with role or tier;
  "paid" and "admin" are never cohort values.
- The tier resolver is a pure function in the credits domain. When a
  subscription system exists it changes the resolver's input, not analytics.
- **Display precedence** for the single "group" badge in admin views: `admin`
  if the role is admin, else the cohort if set, else `paid` when the tier is
  anything but `unpaid`, else `free`.
- Every analytics filter and chart accepts any combination of the three axes.
- Cohort values are validated to `^[a-z0-9_-]{1,32}$`. There is no lookup
  table; the admin console offers existing values as suggestions.

## Freshness contract

Two tiers replace the one-minute promise of the 08-26 design.

**Live** (read at request time, always scoped to one user or one ledger):

- Credit balance and entitlements. Read from the ledger, not analytics.
- One user's event timeline, paginated.
- One user's current lifecycle segment, computed from stored timestamps,
  the trailing-30-day sums from the daily fact table, and today's date.
- One user's operational state (`blocked`, `needs_attention`, `healthy`),
  three small queries.

**Daily** (from `user_daily_facts` and `analytics_daily_rollups`, complete
through the previous UTC day):

- Every cross-user number: overview counts, activation funnel, daily active
  users, segment distribution and movement, retention grid, cost trends,
  cohort and tier comparisons, priority list.
- Per-user trailing windows on the profile: runs and spend over the past 7 and
  30 days, labelled "as of yesterday".

The overview may additionally read the **current** UTC day from raw events as
one bounded one-day scan, so "today" is not blank. That is the only raw-event
read outside the paginated timeline. No per-user "today" numbers are shown.

Consequence to accept: a user's third successful backtest today promotes them
to Core tomorrow. Segments are day-granular by definition, so this is
consistent rather than late.

## Data model

### Sources of truth

| Fact | Authoritative table | Database |
|---|---|---|
| account, role, cohort, created_at | `users` | users |
| sessions, last seen | `auth_sessions` | users |
| dashboard runs, model cost, tokens | `agent_runs` (+ new `owner_user_id`) | runs |
| purchases, refunds, consumption | `credit_ledger_entries`, `credit_llm_usage_entries` | users |
| discrete product actions, page views | `analytics_events` | users |
| protocol and v2 run outcomes | `analytics_events` (`backtest_*`) until protocol runs are durable | users |

Analytics tables stay in the users database. Every read path joins `users`
for role and exclusion, the data is small, and the 2026-09-11 blast radius was
an unbounded loop, not co-location. The auth surface maps a user-store
`OperationalError` (including `PoolTimeout`) to 503 with an `ERROR:
auth.user_store_unavailable` log line so the next outage is visible instead of
a bare 500.

### `agent_runs.owner_user_id`

Nullable, written at run creation on both twins from the authenticated caller.
Not backfilled: rows before the column stay unattributed. It is the source for
per-user model cost (`est_cost_usd`, `input_tokens`, `output_tokens`). Run
*counts* come from events (below) so the three run surfaces share one column
set.

### `user_activity` (replaces `user_analytics_snapshots`)

One row per user, maintained by event ingestion itself. No recompute path
exists.

```text
user_id                       PK, FK users.id
activated_at                  first accepted backtest_completed; set once
last_meaningful_activity_at   GREATEST(existing, event.occurred_at) for every
                              accepted event in the lifecycle activity set
updated_at
```

Credit activity already reaches this row through the `model_usage_recorded`
and `credits_*` events the credits service emits, which are in the
meaningful-activity set. The daily job's ledger step corrects the timestamps
from the ledger itself, so a dropped event cannot leave a paying user looking
inactive. The meaningful-activity set is `_LIFECYCLE_ACTIVITY_EVENTS` in
`domain/analytics/lifecycle.py`, unchanged.

The legacy 08-26 five-state columns (`status`, `reason_code`,
`human_readable_reason`, `evidence_event_ids_json`), their calculator
(`calculate_user_state`), and their repair pass are deleted. The 09-03 value
columns (`lifecycle_*`, `operational_*`, `active_days_30d`,
`successful_backtests_30d`, `calculated_at`) are dropped: segment and
operational state are computed at read time or read from the daily table, and
the 30-day counts are sums over it.

### Lifecycle at read time

`calculate_lifecycle` keeps its rules and reason codes but takes stored inputs
instead of an event list:

```text
calculate_lifecycle(
    created_at, activated_at, last_meaningful_activity_at,
    active_days_30d, successful_backtests_30d, today
) -> LifecycleResult
```

`active_days_30d` and `successful_backtests_30d` are `SUM` over the user's
`user_daily_facts` rows for the 30 UTC dates ending yesterday. Evidence in the
explanation cites the timestamps and sums, not event IDs.

### `user_daily_facts` (replaces `user_lifecycle_daily_snapshots`)

At most one row per user per UTC day, written for the previous day.

```text
snapshot_date         PK part
user_id               PK part, FK users.id
lifecycle_segment     as of the end of snapshot_date
lifecycle_reason_code
operational_state
tier                  commercial_tier() as of the end of snapshot_date
cohort                users.cohort as of the write; NULL when none
active                1 when the user had any accepted event that day
runs_requested
runs_completed
runs_failed
runs_cancelled
operator_cost_micro   SUM(agent_runs.est_cost_usd) for operator-funded runs,
                      converted to micro-USD
own_spend_micro       SUM(credit_llm_usage_entries) consumed that day
data_quality          'complete' | 'partial'
calculated_at
```

Rules:

- Admins and users excluded via `analytics_subject_settings` get no rows,
  matching today's aggregate exclusion. Their profiles still work from live
  data.
- `partial` marks a row for which at least one source was unavailable, or a
  migrated row without run and cost columns. The UI keeps labelling such
  periods "Incomplete data" rather than zero.
- Retention is 180 days because the row carries a user id. Before rows expire
  the retention coordinator writes the anonymous long-term rollups (segment
  counts, transitions, and now run and cost totals keyed by tier and cohort)
  into `analytics_daily_rollups`, which is retained indefinitely.

### `lifecycle_transitions` (outreach hook)

Appended by the daily job when a user's segment on `snapshot_date` differs from
the previous stored day.

```text
transition_id      PK
user_id            FK users.id
snapshot_date
from_segment
to_segment
inactive_days      at the transition
created_at
```

No consumer exists in this design. The future outreach rule "activated user
reached fourteen inactive days" is one `SELECT` over this table plus an
`outreach_log` of its own, and fires at most once per inactivity episode
because each episode produces exactly one `growing → at_risk` row. Retained
180 days.

### `users.cohort`

Nullable text on both twins. Set through the existing admin Users tab
(`set_entitlements` gains the field, or a sibling call with the same guards).
Every analytics response that returns a user carries `role`, `tier`, and
`cohort`, plus the derived `group` badge.

## The daily job

One run per UTC day, scheduled from the existing run-reaper tick.

**Due check.** Each tick reads the `analytics_daily_facts` row in
`analytics_projection_jobs`. If `cursor` already equals yesterday's date the
tick does nothing else. Otherwise it claims the day by updating the cursor
first (compare-and-set on the previous cursor), so a restart or a second
process never runs the same day twice. This replaces the process-local
`_last_rollup_day` guard.

**Steps, for the previous UTC day `D`, each one set-based query over all
users:**

1. Roll up raw events for `D` into `analytics_daily_rollups` (existing
   `rollup_day`, unchanged).
2. One one-day event scan for `D`: per user, `active`, run outcome counts,
   and `activated_at` / `last_meaningful_activity_at` corrections if ingestion
   missed any (defensive, idempotent).
3. One `agent_runs` query grouped by `owner_user_id` for runs ending on `D`:
   operator cost.
4. One ledger query grouped by user for `D`: own spend and the lifetime net
   purchase that feeds `commercial_tier()`.
5. Three grouped queries for operational state (credits billing state,
   provider credential status, failed terminal runs), the same predicates the
   current `get_operational_facts` uses, written once per user rather than
   three queries per user in a loop.
6. Compute `lifecycle_segment` for `D` from `user_activity` plus the 30-day
   sums over existing facts, upsert `user_daily_facts`, and append
   `lifecycle_transitions` where the segment changed.
7. Run the retention coordinator (existing) if due.

Every step is wrapped individually; a failed step marks that day's rows
`partial` and logs `WARNING: analytics.daily_facts.<step>_failed
category=<exception class>`, never a message body. A failed day is retried on
the next tick because the cursor is only advanced past `D` after step 6
succeeds.

**Migration of existing history.** Eight weeks of `user_lifecycle_daily_snapshots`
are copied into `user_daily_facts` with run, cost, and tier columns NULL and
`data_quality='partial'`, then the old table is dropped. Movement charts keep
their history.

## Read paths

| Route | Today | After |
|---|---|---|
| `GET /overview` | live scan of raw events for the filter window | rollups for completed days plus one one-day raw scan for today |
| `GET /lifecycle` | snapshot table plus daily lifecycle table | read-time segment over `user_activity` plus `user_daily_facts` |
| `GET /retention` | live scan of raw events per cohort | `user_daily_facts.active` grouped by activation week |
| `GET /commercial` | live ledger aggregation | unchanged; ledger is the source |
| `GET /operational` | snapshot columns | yesterday's `operational_state` from `user_daily_facts` for lists; live for one profile |
| `GET /users` | snapshot listing | `user_activity` joined to `users` and yesterday's facts; filters on role, tier, cohort, segment, operational state, with tier and operational state taken from yesterday's facts row |
| `GET /users/{id}` | snapshot plus per-section queries | live segment and operational state; 7-day and 30-day sums from facts |
| `GET /users/{id}/activity` | paginated raw events | unchanged |

All routes keep the central admin dependency, the access log, and display-safe
responses.

## Event log discipline

These rules govern `analytics_events` from PR B onward and are enforced by a
test in `tests/test_architecture_boundaries.py` that scans the analytics
package source:

1. The allowlist stays closed and the 1 KB property cap stays.
2. Raw rows are retained 180 days; aggregates live in rollups.
3. No code path reads one user's full event history except the paginated
   timeline route.
4. Every other `list_events` call passes a window of at most one UTC day.
5. Cross-user aggregates read `analytics_daily_rollups` or `user_daily_facts`,
   never raw events, except the overview's current-day scan.
6. No maintenance step loops over users issuing per-user queries.

## Read budget

Pinned by a test that wraps the analytics store in a counting spy and drives
the maintenance path through 24 simulated hours with 200 synthetic users:

| Path | Budget |
|---|---|
| reaper tick when the day is not due | 1 query (`analytics_projection_jobs`) |
| daily job, per UTC day | a constant number of queries, independent of user count, none parameterized by a single user id |
| admin overview page load | at most 1 raw-event query, windowed to the current UTC day |
| admin user profile load | at most 1 raw-event query (timeline page) |

The implementation plan fixes the daily-job constant from the final step list
and the test pins that exact number. The test fails on the first user-id
parameterized query inside the daily job regardless of count, and it fails if
doubling the synthetic user count changes the query count at all.

## Field visibility (My Usage hook)

The per-user metric struct is built once by
`AnalyticsQueryService.get_user_metrics(user_id)` and filtered by audience:

| Field | Admin | User (future) |
|---|---|---|
| runs by outcome, 7d and 30d | yes | yes |
| own spend, operator-funded cost | yes | yes |
| tier, credit balance | yes | yes |
| active days | yes | no |
| lifecycle segment, reasons, evidence | yes | no |
| operational state and evidence | yes | no |
| cohort, group badge | yes | no |
| access log | yes | no |

A future `GET /api/me/usage` is the same call under the session dependency
with the user column of this table. It is not built here.

## Delivery

**PR A: stopgap.** Off `origin/main`, no schema change, deliberately
throwaway.

- `SNAPSHOT_STALE_AFTER = timedelta(hours=24)` shared by `list_stale_user_ids`
  and the repairs; the sweep re-selects a user at most once per day.
- `recalculate_user_snapshots` reads events once and passes them to both
  calculators.
- The legacy `repair_snapshots` call is dropped from the tick.
- `api/auth.py` maps `psycopg.OperationalError` and
  `sqlite3.OperationalError` from authenticate, signup, and current-user to
  503 with the `ERROR: auth.user_store_unavailable` line.
- Tests: reads-per-tick budget with a spy store, re-select cadence, one read
  per user per recompute, auth 503.
- `CLAUDE.md` gotcha: the analytics sweep is on the 60-second reaper; any
  per-user read inside it multiplies by users × 1440.

**PR B: data model and daily job.**

- `agent_runs.owner_user_id`, `users.cohort`, `user_activity`,
  `user_daily_facts`, `lifecycle_transitions`; both twins; migration of the
  eight-week history; drop of the legacy and value snapshot columns and the
  daily lifecycle table.
- Ingestion-maintained `user_activity`; read-time `calculate_lifecycle`.
- The daily job with the compare-and-set due check; PR A's throttled repair
  path deleted.
- Users tab cohort field; `role`, `tier`, `cohort`, `group` on every user
  payload.
- Tests: twin parity for the new DDL, migration on a seeded copy, day-rollover
  and double-claim of the due check, the read-budget test, the event-log
  discipline architecture test.

**PR C: read paths.**

- Overview, lifecycle, retention, operational, users and profile routes onto
  `user_daily_facts` and rollups per the Read paths table; 7-day and 30-day
  profile sums; axis filters in the API and the admin UI.
- Long-term rollup of run and cost totals by tier and cohort before
  `user_daily_facts` rows expire.
- `get_user_metrics` with the audience filter.
- Tests: route contracts, the overview budget, the retention grid against a
  fixture of facts, and the frontend source-shape guards for the filter
  controls.

Each PR leaves prod working on its own. PR B changes both Postgres twins and
must run green on the CI Postgres tier before merge.

## Supersession

- 08-26 "update within one minute" and "raw event visibility: under one
  minute" → live tier applies to the timeline only; cross-user numbers are
  daily.
- 08-26 "A periodic repair pass recalculates stale snapshots" → removed; no
  snapshot recompute exists.
- 08-26 "the current incomplete day reads raw events" → kept for the overview
  only, bounded to one UTC day.
- 09-03 "Current snapshot" section → replaced by `user_activity` and
  read-time lifecycle.
- 09-03 "User-level daily history" → replaced by `user_daily_facts`; retention
  and long-term rollup rules carried forward.
- 09-03 "Legacy status columns remain during compatibility migration" →
  migration is over; they are deleted.
- 09-03 non-goal "Automatically contacting ... based on a segment" → the
  sender remains out of scope; the transition log is in scope as its input.
- Both specs' "five explainable user states" → the 09-03 two-axis model is
  the only model.

## User-facing documentation

The published Sphinx tree has no page on admin analytics, lifecycle segments,
or cohorts. Nothing is stale because nothing exists. A page for administrators
describing the three axes, the segment definitions, and the "as of yesterday"
freshness rule is a follow-up owned by the maintainer, not part of these PRs.
