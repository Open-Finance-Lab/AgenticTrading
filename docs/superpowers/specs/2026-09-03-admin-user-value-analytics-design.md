# Admin User Value Analytics Design

**Date:** 2026-09-03
**Status:** Design approved; implementation planning pending written-spec review
**Target branch:** `feature/admin-user-value-analytics`

## Summary

The Admin Analytics workspace will shift from a primarily operational overview
to an explainable user-value product. It will show where each user is in the
product lifecycle, whether activated users return, which users have paid, and
which accounts require operator attention.

The primary model has two independent axes:

- **Lifecycle:** whether the user has reached and repeatedly received product
  value, measured through meaningful product behavior and successful
  backtests.
- **Commercial value:** the user's lifetime net purchase of ATL Credits,
  excluding Admin Grants.

Operational conditions such as `Blocked` and `Needs attention` remain visible,
but no longer compete with lifecycle stages. An account can therefore be both
`At risk` and `Blocked`, which answers two different questions without forcing
one label to hide the other.

The Admin workspace will also replace its horizontal section tabs with a
persistent left navigation rail. The Credits and Billing workspace will not be
changed in this delivery; its matching left-navigation redesign belongs in a
separate pull request.

This design extends the collection, privacy, retention, and read-only profile
contracts in
`docs/superpowers/specs/2026-08-26-admin-user-analytics-design.md`. It supersedes
that document only where it defines the Admin Analytics information hierarchy
and the former five-state user classification.

## Goals

1. Make activation, retention, lifecycle movement, and commercial value useful
   for day-to-day user analysis.
2. Distinguish user maturity from current operational problems.
3. Give every lifecycle and operational label a visible, evidence-backed
   explanation.
4. Let an administrator move from an aggregate signal to the exact affected
   users in one action.
5. Preserve partial availability, privacy, and source-ledger authority.

## Non-goals

- A single opaque user score, predictive churn model, or machine-learning
  segmentation system.
- Marketing automation, messaging, suspension, refunds, Grants, or any other
  automatic action based on a segment.
- Changing event collection allowlists, raw-event privacy rules, or the
  180-day raw-event retention contract.
- Treating Admin Grants as revenue or copying the Credits ledger into
  Analytics.
- Changing the Credits and Billing page navigation in this pull request.
- Exporting user lists or Analytics data.
- Replacing the detailed Timeline, Runs, Usage, or Sessions profile sections.

## Product Questions

The first screen must answer these questions in order:

1. How many users have activated, become Core, become At risk, or paid?
2. Where are users distributed across the lifecycle now?
3. Are users moving toward repeated value or toward inactivity?
4. Which exact users should an administrator inspect first?
5. What do retention, commercial value, and operational health reveal when
   deeper analysis is needed?

## Definitions

### Activation

A user activates at the first server-authoritative `backtest_completed` event.
This timestamp is stable after it is first observed. A failed, cancelled,
queued, or merely started backtest does not activate a user.

### Meaningful behavior

Lifecycle activity counts only intentional product behavior that represents
setup, execution, or commercial use:

- creating or materially configuring an Agent;
- saving, verifying, re-verifying, or selecting a model API credential;
- requesting or progressing a backtest, including a terminal failure;
- purchasing ATL Credits; and
- consuming ATL Credits through model execution.

Passive page visits, sign-in, token refresh, polling, browser refresh, and
session heartbeat events do not count. An automatic or administrator-assigned
Grant also does not count because it is not an action by the user.

An **active day** is a distinct UTC calendar day containing at least one
meaningful behavior. Multiple events on the same day count once.

### Inactivity clock

The inactivity clock starts from the most recent meaningful behavior. If none
exists, it starts from the account creation timestamp. `inactive_days` is the
difference between that timestamp's UTC date and the calculation UTC date.
Values from 0 through 7 are recent, 8 through 29 are At risk, and 30 or more
are Dormant. This removes sub-day gaps and makes the result independent of a
browser timezone.

## Lifecycle Segments

Each non-excluded user receives exactly one current lifecycle segment.

| Segment | Deterministic rule |
| --- | --- |
| `New` | UTC account age is 0 through 6 days and no successful backtest exists. |
| `Onboarding` | No successful backtest exists, and the user is not New, At risk, or Dormant. |
| `Growing` | A successful backtest exists, `inactive_days` is at most 7, and Core criteria are not met. |
| `Core` | In the current UTC date plus preceding 29 UTC dates, the user has at least 3 active days and at least 3 successful backtests, and `inactive_days` is at most 7. |
| `At risk` | `inactive_days` is 8 through 29. |
| `Dormant` | `inactive_days` is at least 30. |

Evaluation is deterministic:

1. `Dormant` and `At risk` are evaluated from the inactivity clock first.
2. A non-inactive, unactivated user is `New` while account age is less than 7
   days, then `Onboarding`.
3. A non-inactive, activated user is `Core` when the Core thresholds are met,
   otherwise `Growing`.

At risk and Dormant reasons include one of these reason qualifiers:

- `never_activated`
- `previously_activated`

The current identity always uses these fixed rolling windows. Changing the page
date filter never changes a user's current segment.

### Explainability

Every lifecycle result includes:

```text
lifecycle_segment
lifecycle_reason_code
lifecycle_reason
lifecycle_evidence
calculated_at
```

`lifecycle_evidence` contains display-safe facts rather than a private event
payload. Examples include:

```text
4 active days in the trailing 30 days
6 successful backtests in the trailing 30 days
last meaningful activity 2 days ago
first successful backtest on 2026-08-21
```

The interface exposes the rules in three places:

- a concise tooltip on each lifecycle label;
- a keyboard-accessible `How segments work` side panel containing all rules;
- user-specific evidence in the priority-user panel and User Analytics Profile.

## Operational State

Operational state is calculated separately from lifecycle and has this
precedence:

1. `Blocked`
2. `Needs attention`
3. `Healthy`

`Blocked` represents an unresolved condition that currently prevents a core
action, including an account Credits restriction, missing usable billing lane,
or disabled selected provider. `Needs attention` represents an actionable but
not necessarily blocking condition, including an invalid default credential,
three consecutive failed terminal runs within 24 hours, or a run beyond its
safe deadline. `Healthy` means no supported current blocker or attention rule
matched; it is not a promise that every external provider is available.

Every operational state returns its own reason code, human-readable reason,
evidence, and calculation time. Lifecycle and operational badges are shown
side by side where both are relevant.

The existing mixed `status` field and filters remain temporarily available to
old consumers. New Admin Analytics UI code must use `lifecycle_segment` and
`operational_state`; it must not infer one from legacy `status`.

## Commercial Value

Commercial value is based on lifetime net purchased ATL Credits:

```text
net_purchased_micro = settled purchase entries - settled refund entries
```

The Credits ledger is authoritative. Model consumption does not reduce
lifetime net purchases, and Admin Grant assignment or reclaim never contributes
to it.

| Tier | Lifetime net purchase |
| --- | ---: |
| `Unpaid` | exactly $0 |
| `Starter` | greater than $0 and less than $5 |
| `Invested` | at least $5 and less than $20 |
| `High value` | at least $20 |

Refunds can move a user to a lower tier. The displayed value uses the existing
one-dollar-to-one-ATL-Credit accounting contract.

The Commercial Value section also shows these independent measures without
combining them into revenue:

- ATL Credits consumed in the selected period;
- current Grant, Purchased, and total available balance;
- platform model cost in the selected period; and
- Admin Grant activity, clearly labelled as non-revenue.

## Retention Cohorts

Retention starts at the first successful backtest rather than signup.

- Cohort membership uses the UTC Monday-to-Sunday week containing activation.
- That activation week is Week 0.
- Week 1, Week 2, and Week 4 retention require at least one meaningful behavior
  in the corresponding later UTC calendar week.
- A target week enters the denominator only after the full target week has
  elapsed.
- An immature cell is unavailable, never zero.
- Summary percentages are `retained eligible users / all eligible users` across
  mature cohorts; the cohort table also shows each weekly cohort independently.

The date filter selects activation cohort weeks and trend coverage. It does not
reassign users to a historical current lifecycle segment.

## Information Architecture

### Admin workspace navigation

The Admin workspace uses a left rail in this order:

1. `Analytics`
2. `Users`
3. `Providers`
4. `Activity`

`Analytics` remains the default route. The existing `adminTab` URL parameter is
preserved. On a wide screen the rail shows icons and labels. On a narrow screen
it becomes a stable icon rail with accessible names and tooltips; it does not
move back to a horizontal strip above the content.

The control remains a tab interface with `role=tablist`,
`aria-orientation=vertical`, roving `tabindex`, Up/Down arrow navigation, Home,
End, and synchronized `aria-selected` and `aria-controls` values.

The Credits and Billing `API Keys / Credits / Activity` tabs remain unchanged
in this delivery. A later pull request may apply the same shell pattern there.

### User lifecycle overview

The right workspace uses a restrained, dense information hierarchy rather than
a set of decorative cards.

The always-visible area contains:

1. current Activated, Core, At risk, and Paid user counts;
2. current distribution across all six lifecycle segments;
3. an eight-week lifecycle movement chart and key transition callout; and
4. a short Priority Users list.

The four current headline metrics are calculated independently of the selected
date range:

- `Activated users`: users with a first successful backtest at any time;
- `Core users`: users whose current lifecycle segment is Core;
- `At risk`: users whose current lifecycle segment is At risk; and
- `Paid users`: users whose lifetime net purchase is greater than zero.

All four respect the current internal-account inclusion setting.

Selecting a segment count or chart point opens or filters the corresponding
user list without changing the current lifecycle calculation. URL state keeps
the selected lifecycle, operational state, commercial tier, and user profile
deep-linkable.

The following sections are collapsed by default and fetched only when opened:

- Retention Cohorts
- Commercial Value
- Operational Health

Operational Health contains the existing backtest success, run, model usage,
provider, cost, and failure-category analysis. Provider, model, and billing-mode
filters live inside this section because they do not apply coherently to current
lifecycle identity. The top-level controls contain only the trend/cohort date
range and the existing internal-account inclusion control.

### Priority Users

The default Priority Users query uses this group order:

1. `Blocked`
2. `Needs attention`
3. otherwise-healthy `At risk`
4. otherwise-healthy `Onboarding`

Within a group, users sort by commercial tier and exact lifetime net purchase
descending, then inactivity descending, then user ID ascending for stable
pagination. New, Growing, and Core users appear only when selected explicitly
or when an operational problem puts them in a higher-priority group.

Each row shows account identity, lifecycle, operational state, commercial tier,
last meaningful activity, and the concise user-specific reason. Selecting a row
opens a quick evidence side panel. `Open full analytics profile` continues to
the dedicated profile, while `Open account management` continues to the
existing Users workspace. Analytics remains read-only.

### User Analytics Profile

The priority-user list and other user tables provide a direct, display-safe
link to a dedicated User Analytics Profile route. The profile is a separate
workspace surface rather than an inline expansion so the list remains scannable
while Timeline, Runs, Usage, and Sessions can grow independently. A breadcrumb
and an explicit back action return to the exact Analytics list state, including
filters, date range, pagination, and scroll position. Browser back/forward and
deep links follow the same URL state. Opening a profile records a history entry;
switching profile sections replaces only the current entry, and a direct link
falls back to the Analytics overview when no parent history entry exists.

### Lifecycle Movement Ranges

The Direction of travel chart defaults to the most recent five UTC calendar
days. A compact range control in the card header offers `5D`, `1W`, `1M`, and
`1Y`. Five-day and one-week views use daily snapshots; one-month uses weekly
snapshots; and one-year uses monthly snapshots. The API returns the selected
range, granularity, and display-safe period points so the client never relabels
weekly data as daily data. Missing historical snapshots remain partial or empty
states; the system never fabricates zero-valued history. The selected movement
range is URL-backed independently from the broader Analytics date filters.

The existing dedicated User Analytics Profile remains the full inspection
surface with Overview, Timeline, Runs, Usage, and Sessions. Its Overview adds:

- lifecycle segment and user-specific evidence;
- operational state and independent evidence;
- activation date and recent active-day/success counts;
- commercial tier and lifetime net purchase;
- current balances, selected-period consumption, and platform cost; and
- recent lifecycle transitions when history is available.

Timeline, Runs, Usage, and Sessions retain independent cursor pagination and
partial error handling. The profile retains its Admin access audit record and
`Open account management` link.

## Storage and Calculation Design

SQLite and PostgreSQL implementations must expose equivalent behavior.

### Current snapshot

`user_analytics_snapshots` remains the current-state projection. It gains
explicit lifecycle and operational fields rather than overloading one status:

```text
lifecycle_segment
lifecycle_reason_code
lifecycle_reason
lifecycle_evidence_json
operational_state
operational_reason_code
operational_reason
operational_evidence_json
activated_at
last_meaningful_activity_at
active_days_30d
successful_backtests_30d
calculated_at
```

Legacy status columns remain during compatibility migration and retain their
existing calculation; they are not silently redefined as either new axis.
Relevant accepted events recalculate the user's snapshot. The existing bounded
stale-snapshot repair also handles time-only transitions, such as Growing to At
risk, even when no new event arrives.

Commercial value is joined from a bounded aggregate query over authoritative
purchase and refund entries. It is not persisted as a second mutable balance.

### User-level daily history

A new `user_lifecycle_daily_snapshots` projection stores at most one row per
user per UTC day:

```text
snapshot_date
user_id
lifecycle_segment
lifecycle_reason_code
data_quality
calculated_at
```

`data_quality` is `complete` or `partial`. This table supports user transitions
and the eight-week chart and is retained for 180 days because it contains a
user identifier.

Before user-level rows expire, bounded aggregate rows are written to
`analytics_daily_rollups` for long-term preservation:

- lifecycle segment counts; and
- transitions between the fixed, allowlisted lifecycle segments.

Transition keys are bounded enum combinations, not arbitrary strings. The
long-term rows contain no user identifier.

### Eight-week historical backfill

Deployment performs an idempotent, batched reconstruction of the previous eight
weeks. For each UTC day end, it evaluates lifecycle using only evidence that
would have been available by that time. It may read the existing 180-day
Analytics event set and authoritative user/run data; it must not use a later
event to classify an earlier day.

The backfill writes deterministic daily keys, may resume safely, and never
invents page visits, sessions, Agent edit history, credential history, or other
missing evidence. A reconstructed date is marked `partial` when its required
source horizon is not trustworthy. APIs return coverage and quality metadata,
and the UI labels affected periods `Incomplete data` instead of rendering them
as zero.

Current lifecycle becomes available independently of historical backfill. The
UI may show current counts while movement history is still building.

## API Design

All routes remain under `/api/admin/analytics`, require the central Admin
dependency, and return display-safe data only.

```text
GET /api/admin/analytics/lifecycle
GET /api/admin/analytics/retention
GET /api/admin/analytics/commercial
GET /api/admin/analytics/operational
GET /api/admin/analytics/users
GET /api/admin/analytics/users/{user_id}
GET /api/admin/analytics/users/{user_id}/activity
```

### Lifecycle

`/lifecycle` returns current headline metrics, current segment counts, weekly
segment counts, transition callouts, `as_of`, coverage, data quality, and
per-subsection availability. Its date range affects historical series only.

### Retention

`/retention` returns eligible activation cohorts, Week 1/2/4 numerators and
denominators, maturity, and quality status. It is requested only when its
section opens.

### Commercial

`/commercial` returns current tier counts and lifetime net purchase totals,
plus selected-period purchase, refund, consumption, balance, Grant, and
platform-cost breakdowns. It is requested only when its section opens.

### Operational

`/operational` preserves existing operational overview metrics and accepts the
provider, model, and billing-mode filters moved into its UI section.

### Users

The users endpoint adds independent filters for:

```text
lifecycle_segment
operational_state
commercial_tier
activated
last_meaningful_activity
priority
```

It retains bounded pagination, identity search, deterministic sorting, default
internal-account exclusion, and existing legacy status input during the
compatibility period.

The user profile adds lifecycle, operational, commercial, and recent-transition
objects. Detail activity contracts remain independently pageable.

## Loading, Empty, and Error States

Lifecycle, Retention, Commercial, Operational, and Users are independent query
boundaries. A failure in one never blanks another.

- Stable-size skeletons prevent the layout from moving during loading.
- Empty data says what is empty; it never resembles a loading or error state.
- A failed section shows `This section is temporarily unavailable` and a local
  retry action.
- Incomplete historical evidence shows coverage dates and `Incomplete data`.
- A stale successful response may remain visible with a stale indicator while
  its refresh fails.
- Authentication or authorization loss follows the existing Admin exit path.
- Server and browser errors remain display-safe and never include SQL, raw
  events, provider response bodies, secrets, or stack traces.

Analytics calculation, snapshot, daily history, aggregate rollup, or backfill
failure never changes authentication, Agent, backtest, provider, Credits, or
payment outcomes.

## Accessibility and Responsive Behavior

- The Admin rail uses a vertical ARIA tablist and complete keyboard navigation.
- Narrow layouts retain a left icon rail with accessible names and tooltips.
- Segment controls are buttons with selected state and visible focus.
- Collapsed analysis sections use buttons or native disclosure semantics with
  correct expanded state.
- The rules and quick-evidence side panels have dialog names, focus containment,
  Escape-to-close, backdrop close, and focus return to the opener.
- Charts provide descriptive labels and equivalent hidden data tables.
- Color is never the only signal for a segment, trend, quality state, or error.
- Tables keep column headers and become horizontally scrollable only when a
  vertical mobile representation would lose meaning.
- Text, controls, and badges must not overlap at supported desktop and mobile
  widths.

## Privacy and Security

The original Admin Analytics privacy contract remains unchanged. In particular:

- no API keys, prompts, strategy content, raw upstream bodies, passwords,
  tokens, full IP addresses, or raw User-Agent values enter these projections;
- lifecycle evidence contains allowlisted facts and event references only;
- Admin accounts and analytics-excluded accounts remain excluded by default;
- opening quick user evidence or the full profile records Admin profile access;
- APIs are read-only and cannot mutate lifecycle, operational, or commercial
  classifications; and
- test fixtures use synthetic identities and values only.

## Testing Strategy

### Domain tests

- Every lifecycle boundary at UTC account/inactivity day 7, 8, 29, and 30.
- New, Onboarding, Growing, and Core rules at exact active-day and successful-run
  thresholds.
- `never_activated` and `previously_activated` At risk/Dormant reasons.
- Meaningful-behavior allowlist and explicit page-view, login, heartbeat,
  polling, automatic Grant, and Admin Grant exclusions.
- Orthogonality and precedence of lifecycle and operational state.
- Commercial tier boundaries, purchase/refund netting, and Grant/consumption
  exclusion from revenue.
- Week 1/2/4 cohort maturity, UTC week boundaries, and meaningful-return rules.

### Storage and backfill tests

- SQLite/PostgreSQL migration and repository contract parity.
- One daily row per user/date and deterministic transition aggregation.
- Idempotent, resumable eight-week backfill with no future-event leakage.
- Partial source coverage remains partial rather than becoming zero.
- User history expires after 180 days only after anonymous aggregate creation.
- Existing legacy snapshot readers remain compatible during migration.

### API tests

- Admin authorization and Admin profile-access auditing.
- Independent lifecycle, retention, commercial, operational, and users
  responses.
- Date filters alter trends/cohorts but not current segment identity or tier.
- User filters, stable priority ordering, pagination, and URL-safe values.
- Partial availability and safe `503` responses without sensitive detail.
- Contract fixtures validate every frontend-consumed response.

### Frontend tests

- Left-rail selection, URL synchronization, vertical roving focus, and
  narrow-width icon rail.
- Segment and chart drill-down to the expected user filters.
- Priority rows show both lifecycle and operational state plus commercial tier.
- Rules and evidence panels manage focus, Escape, backdrop close, and focus
  return.
- Retention, Commercial, and Operational sections fetch only on first open,
  cache successful data, and retry locally after failure.
- One failed or incomplete section leaves all other sections usable.
- Chart hidden tables, ARIA states, visible focus, long-text containment, and
  desktop/mobile screenshot checks.
- Existing full profile Timeline, Runs, Usage, Sessions, and Open account
  management navigation remain functional.

Tests use frontend contract fixtures and fake Analytics/Credits repositories.
They do not require real API keys, Stripe, provider calls, production databases,
or copied production user data.

## Rollout and Observability

1. Apply additive SQLite and PostgreSQL schema migrations.
2. Start writing independent current and daily lifecycle projections while
   preserving legacy status.
3. Run the bounded eight-week backfill and publish quality/coverage status.
4. Enable the new Admin UI when current snapshot queries are available; allow
   historical sections to report building or partial status independently.
5. Observe safe counters for calculation, rollup, backfill, retention, and
   query failures before removing any legacy status consumer in a later change.

No automated user or billing action is introduced during rollout.

## Acceptance Criteria

- Admin navigation is a persistent left rail with Analytics first and selected
  by default; Credits navigation is unchanged.
- Every eligible user has one explainable lifecycle segment and one independent
  operational state.
- The overview always exposes Activated, Core, At risk, and Paid counts,
  lifecycle distribution, eight-week movement, and Priority Users.
- Retention uses first successful backtest cohorts and mature Week 1/2/4
  meaningful-return windows.
- Commercial tiers use lifetime purchases minus refunds; Admin Grants are never
  revenue.
- Aggregate interactions lead to correctly filtered user lists, and user rows
  expose display-safe evidence and links to full Analytics and account
  management.
- Date filters affect trends and cohorts, never current lifecycle identity.
- Eight weeks are backfilled idempotently, with incomplete evidence labelled
  rather than fabricated.
- User-level daily history expires after 180 days and anonymous aggregates
  remain available.
- Partial query failures do not blank unrelated analysis.
- Keyboard, ARIA, hidden-table, focus, and narrow-screen behavior meet the
  documented contracts.
- No prohibited secret, raw provider content, database, `.superpowers/`, or
  synthetic mockup artifact is committed.
