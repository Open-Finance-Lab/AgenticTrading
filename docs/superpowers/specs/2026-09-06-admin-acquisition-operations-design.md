# Admin Acquisition and ATL Operations Analytics

**Date:** 2026-09-06  
**Status:** Design addendum pending review  
**Extends:** `docs/superpowers/specs/2026-08-26-admin-user-analytics-design.md`

## Purpose

The existing Analytics experience explains platform health and individual user
activity. This addendum adds the smallest useful operating view for ATL:
which acquisition sources bring users, whether those users complete the core
task, whether they return, how many ATL Credits they consume, and whether they
show a real purchase signal.

This is a cross-cutting dimension of the existing Analytics modules. It does
not add another top-level Admin tab or a second user-profile system.

## Product decisions

### Acquisition dimensions

Every non-excluded account has one primary acquisition source and may have one
operating cohort.

```text
acquisition_source: student | community | friend | competition | unknown
acquisition_cohort: nullable short slug, for example 2026-fall-course
```

The database and API use English enum values. The UI labels them in the
administrator's locale (for example, Student, Community, Friend, Competition
team, Unknown). Missing attribution defaults to `unknown`.

Source means how the person entered ATL. Cohort means the course, campaign,
community event, or competition batch they belong to. Lifecycle (New, Active,
At risk, Dormant) describes current product use. Paid status describes a
verified purchase. These values remain separate.

The first source is captured from a controlled invite or campaign link when
the account is created. The system never infers it from an email address, IP
address, page views, or other guesses. An administrator may correct the source
or cohort only from the User Analytics Profile. Each correction records the
actor, timestamp, previous value, new value, and a short reason in the existing
admin audit path. Raw referral URLs and private referral text are not stored.

### Core task and activity definitions

These definitions reuse the approved Analytics rules:

- **Active:** a distinct user with at least one accepted
  `backtest_requested` in the selected date range, regardless of terminal
  outcome.
- **Task completed:** a user with at least one persisted, viewable successful
  `backtest_completed` result. The first such result completes the user's
  activation milestone.
- **Repeat:** a user with at least two successful backtests on different UTC
  calendar dates. This is a lifetime user attribute in the first version.
- **Runs / active user:** accepted backtest requests in the selected range
  divided by active users in that range; zero-active groups show `—`.
- **ATL Credits:** the sum of settled platform-credit debits in the selected
  range. Grants, reservations, refunds, and BYOK provider spend are not
  counted as settled consumption.

Progress and task completion use full history. Activity, runs, Credits, and
purchase-funnel events use the global `1D`, `1W`, `1M`, or `1Y` date filter.
The UI labels lifetime and selected-range values wherever they appear together.

## Information architecture

The Admin navigation remains:

```text
Analytics → Users → Providers → Activity
```

The existing Overview receives a collapsible **Acquisition groups** card. The
same Source and Cohort filters are available in Overview, Users, Retention,
Usage & Cost, and Operations. User Profile shows the two dimensions in its
header.

### Acquisition groups card

The collapsed card shows one row per source or cohort and these core columns:

| Column | Meaning | User question answered |
|---|---|---|
| Group | Source or operating cohort label | Where did this group come from? |
| Users | Total users in the group | How large is the group? |
| Active | Users with an accepted request in range | Who actually started using ATL? |
| Task completed | Users with a successful result | Who reached the core value? |
| Repeat | Users with two successful runs on different dates | Who came back and used it again? |
| Runs / user | Requests per active user in range | How deeply is the group using ATL? |
| ATL Credits | Settled platform-credit consumption in range | How much platform resource did it consume? |
| Paid intent | Unique users who started checkout | Who showed a strong purchase signal? |
| Paid | Unique users with a settled purchase | Who actually paid? |

Input, output, and total token columns are deliberately absent from this
first-version table. Token evidence remains available to future usage work and
is not removed from authoritative records.

The expanded area may show rates and supporting detail: Agent creation rate,
first-request rate, first-success rate, terminal run success rate, checkout to
purchase conversion, purchase/refund/net amounts, and common failure reasons.
These details do not add columns to the default table.

Clicking a group applies its filters and opens Users. Clicking a metric count
opens Users with the corresponding derived filter (for example, Task completed
or Paid intent). Clicking a user opens the dedicated User Analytics Profile;
Back returns to the prior filtered list.

### Shared filters

| Filter | Values | Effect |
|---|---|---|
| Date range | `1D`, `1W`, `1M`, `1Y` | Range-scoped activity, usage, Credits, and purchase events |
| Source | All or one of five enum values | Slices every supported module by acquisition source |
| Operating cohort | All or one cohort slug | Slices every supported module by campaign/batch |
| Lifecycle | All, New, Active, At risk, Dormant | Filters user lists and operational slices |
| Blocked | All, Blocked, Not blocked | Filters operationally blocked accounts |
| Paid | All, Paid, Unpaid | Filters verified purchase state |
| Include internal accounts | Off by default | Includes admin/test accounts only when explicitly enabled |

The current date range does not change historical progress milestones. Internal
and explicitly excluded accounts follow the base Analytics specification.

### Users and User Profile

Users adds read-only Source and Operating cohort columns beside the existing
identity and progress fields. The list supports filtering and group navigation
but does not edit attribution.

The User Profile header displays Source, Operating cohort, attribution method,
attributed time, and last correction time when present. Profile editing is
admin-only and writes an audit record. The existing four-step historical
progress remains the primary onboarding signal:

```text
Account created → Agent created → Backtest attempted → First successful result
```

## Purchase signal

Purchase intent is represented as an observable funnel, not a prediction score:

```text
Credits page viewed → Checkout started → Purchase settled
```

`page_viewed(page="credits")` is an existing weak signal. Add a
server-authoritative `checkout_started` event when a checkout session/order is
successfully created. A verified purchase or existing settled purchased-credit
ledger entry is the purchase fact. Admin grants never count as intent or
revenue.

The first version exposes `Paid intent` and `Paid` in the core group table. The
expanded detail can show funnel conversion. Checkout-to-purchase conversion is
attributed within seven days of checkout creation and excludes checkout rows
whose seven-day observation window is not complete; the UI marks a partial
window when necessary. A separate `checkout_cancelled` event is deferred.

## API contract

The existing Admin Analytics endpoints retain their paths and gain optional
query parameters:

```text
GET /api/admin/analytics/overview
  ?date_range=1d|1w|1m|1y
  &acquisition_source=student|community|friend|competition|unknown
  &acquisition_cohort=<slug>
  &lifecycle=new|active|at_risk|dormant
  &blocked=true|false
  &paid=true|false
  &include_internal=false

GET /api/admin/analytics/users
  ...same filters plus existing query, sort, and cursor parameters

GET /api/admin/analytics/users/{user_id}
  ...returns source/cohort attribution and audit-safe profile fields
```

Overview adds a stable `acquisition_groups` response member. Each group row
contains display-safe values with explicit count fields:

```json
{
  "group": {"kind": "source", "value": "community", "label": "Community"},
  "users": 42,
  "active": 18,
  "task_completed": 11,
  "repeat": 6,
  "runs": 31,
  "runs_per_active_user": 1.72,
  "atl_credits_settled": 3.84,
  "paid_intent": 4,
  "paid": 2
}
```

The API must not expose prompts, strategy text, API keys, raw referral URLs,
payment details, or raw provider responses. Unknown or unavailable aggregate
values use the existing partial-error contract and do not become zero silently.

## Storage and instrumentation

The authoritative account record gains nullable acquisition fields through the
existing user/profile storage path, or an equivalent one-to-one attribution
table if migrations require isolation. The analytics query layer reads a
display-safe projection and never exposes arbitrary profile JSON.

Add a server event for `checkout_started` with the checkout/order identifier as
an idempotent source reference. Do not store the payment provider payload.
Source corrections use the existing Admin audit mechanism and are not modeled
as user activity events.

Group metrics reuse existing server-authoritative run, usage, Credits, and
purchase records. A group query must exclude Admin and analytics-excluded
accounts by default and must preserve SQLite/PostgreSQL repository parity.

## Failure behavior and privacy

Source or cohort data unavailable for a user is shown as Unknown. A failed
acquisition-group query leaves the rest of Overview usable and shows the
standard panel-level unavailable state. It must not block account creation,
backtests, Credits settlement, or checkout creation.

Only display-safe source labels and audit metadata are returned. The system
does not store raw invite URLs, private referral content, prompts, credentials,
or payment-card data. Attribution changes are observable to administrators
through the audit record.

## Testing and acceptance

Contract tests must cover:

1. enum validation, default `unknown`, cohort slug bounds, and audit metadata;
2. all five source values and cohort filters across SQLite and PostgreSQL;
3. Active, Task completed, Repeat, Runs / user, settled Credits, Paid intent,
   and Paid formulas with deterministic UTC fixtures;
4. no Token columns in the default group response or rendered table;
5. `checkout_started` idempotency and seven-day conversion-window handling;
6. default exclusion of internal/excluded accounts;
7. group-row navigation preserving source/cohort filters and User Profile back
   navigation;
8. partial aggregate failures preserving other Overview panels;
9. no raw referral URLs, secrets, prompts, payment payloads, or provider bodies
   in storage, API responses, logs, fixtures, or rendered HTML.

Acceptance is met when an administrator can answer, for each source or cohort:

```text
How many came? How many started? How many completed and returned?
How many ATL Credits did they consume? How many showed intent and paid?
```

The first version does not include predictive payment scoring, automatic
marketing actions, machine-learned segments, or a new top-level navigation
module.
