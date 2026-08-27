# Admin Footprint and User Analytics Design

**Date:** 2026-08-26  
**Status:** Approved design  
**Delivery:** Three sequential pull requests

## Summary

Agentic Trading Lab will add a first-party Admin Analytics system that makes user activation, product usage, model execution, cost, and friction visible without exposing secrets or turning analytics into a second financial ledger.

The Admin console will gain a first-position `Analytics` tab. It will contain a read-only platform overview and link to a dedicated read-only User Analytics Profile for each account. Existing `Users`, `Providers`, and `Activity` tabs will retain mutation controls such as role changes, Grant Credits operations, and provider configuration.

Analytics will collect authenticated business events and major page visits, update within one minute, retain user-level event detail for 180 days, and preserve non-identifying daily aggregates for long-term trend analysis. Raw API keys, passwords, prompts, strategy text, full IP addresses, and upstream provider response bodies are prohibited from analytics storage and responses.

## Goals

1. Show how users progress from signup to first value and repeat usage.
2. Let administrators understand one user's product footprint, run outcomes, billing lane, and resource usage.
3. Identify users who are onboarding, active, dormant, blocked, or in need of attention with explainable evidence.
4. Surface common failure categories and cost trends without exposing sensitive data.
5. Keep analytics failures isolated from authentication, backtests, model execution, and Credits settlement.

## Non-goals

- Session replay, DOM recording, keystroke capture, or arbitrary click tracking.
- Recording prompts, strategy content, portfolio contents, API keys, passwords, or complete provider errors.
- Replacing the Credits ledger, model usage evidence, run records, or existing admin audit trails.
- Automatically contacting, suspending, refunding, or changing a user based on analytics status.
- Introducing a third-party analytics platform in the first version.
- Creating an opaque health score or machine-learning risk model.

## Product Structure

### Admin navigation

The Admin tab order becomes:

1. `Analytics`
2. `Users`
3. `Providers`
4. `Activity`

`Analytics` is the default Admin tab. It is read-only. Any account mutation links back to the existing `Users` tab through an `Open account management` action.

The Admin URL continues to use `adminTab`, with `adminTab=analytics` for the new page. The tab must retain the existing ARIA roles, selected state, keyboard navigation, and URL synchronization behavior.

### Analytics Overview

The overview follows a vertical reading order instead of a dense dashboard grid:

1. **Platform snapshot**
   - Active users, rolling seven days
   - First successful run conversion
   - Backtest success rate
   - Platform model cost
2. **Growth and engagement**
   - Daily active-user and completed-run trend
   - Activation funnel
3. **User health and friction**
   - Counts for the five explainable user states
   - Top actionable failure categories and affected-user counts
4. **Users needing attention**
   - Account identity
   - Current state and reason
   - Last meaningful activity
   - Recent run count and failures
   - Link to the User Analytics Profile

Global filters support date range, billing mode, provider, and model. Internal accounts are excluded by default. The page shows `Last updated` and allows a manual refresh.

### User Analytics Profile

Selecting a user opens a dedicated read-only page rather than expanding the account-management table.

The profile uses a stable two-column layout:

- **Left account summary**
  - Display name, email, and user ID
  - Join date and last meaningful activity
  - Current explainable state and reason
  - Primary billing lane and default provider, when available
  - Coarse region, device category, and browser family, when available
  - `Open account management` link
- **Right analytics content**
  - `Overview`
  - `Timeline`
  - `Runs`
  - `Usage`
  - `Sessions`

The Overview shows activation milestones, recent footprint events, run summary, billing-lane mix, model tokens, ATL model cost, and top product page. Timeline, Runs, Usage, and Sessions are independently paginated so a profile never loads 180 days of detail in one response.

## Collection Scope

### Server-authoritative events

The backend emits events only after the source operation reaches its authoritative outcome.

Event groups include:

- **Account:** account signup and authenticated session start
- **Credential:** credential saved, verified, defaulted, reverified, or revoked
- **Agent:** agent created, updated, or deleted
- **Run:** requested, queued, started, completed, failed, or cancelled
- **Resource:** model usage recorded, Credits reserved, settled, or refunded, and safe error classification

The frontend cannot claim server-authoritative outcomes such as `backtest_completed`, `credential_verified`, or `credits_settled`.

### Frontend experience events

The authenticated frontend may submit only an allowlisted set of events, initially:

- `page_viewed`
- `page_hidden`
- `session_heartbeat`

Allowed page identifiers are stable product views such as `home`, `agents`, `agent_editor`, `credits`, and `account`. The endpoint rejects unknown event names, unknown page identifiers, unknown property keys, oversized payloads, and unauthenticated requests.

The client never submits email, display name, role, API key metadata, prompt text, strategy text, error bodies, or arbitrary form values. Server authentication supplies `user_id`.

### Sessions and page duration

The browser creates a random analytics session identifier that is unrelated to the authentication token. A session ends after 30 minutes without accepted page activity. Page duration is estimated from accepted visibility and heartbeat events; background polling and authentication refreshes do not extend meaningful activity.

## Event Contract

`analytics_events` stores a narrow, versioned envelope:

```text
event_id
schema_version
event_name
event_group
user_id
session_id
occurred_at
received_at
event_source
source_event_id
source_record_type
source_record_id
correlation_id
page_view
provider_id
model_id
billing_mode
outcome
error_category
country_code
device_category
browser_family
network_hash
properties_json
```

`source_event_id` is unique when present and provides idempotency for server events and backfill. `correlation_id` links related lifecycle events for a run or another multi-step operation. `properties_json` accepts only event-specific allowlisted keys and has a strict serialized-size limit.

Analytics stores safe error categories, not raw exceptions or upstream bodies. Approved categories include stable values such as `credential_invalid`, `credential_missing`, `provider_timeout`, `provider_unavailable`, `credits_unavailable`, `model_not_allowed`, and `internal_error`.

## Storage Model

SQLite and PostgreSQL implementations must expose the same repository contract.

### `analytics_events`

Stores raw event detail for 180 days. Indexes support:

- user and occurrence time
- event name and occurrence time
- session and occurrence time
- outcome or error category and occurrence time
- source-event idempotency

### `analytics_daily_rollups`

Stores non-identifying daily aggregates by supported dimensions. Historical completed days read primarily from this table. Rollup dimensions are intentionally bounded to avoid uncontrolled cardinality.

Supported dimensions include event name, billing mode, provider, model, outcome, error category, and user-state count. The system does not aggregate by email, raw session identifier, or network hash.

### `user_analytics_snapshots`

Stores the current state, stable reason code, human-readable reason, evidence event IDs, and `calculated_at` for each user. A relevant new event recalculates that user's snapshot. A periodic repair pass recalculates stale snapshots.

### `analytics_subject_settings`

Stores analytics-specific user settings without expanding the authentication record. It contains `user_id`, exclusion state, actor, reason, and timestamps. Admin accounts and accounts marked `analytics_excluded` are excluded from default analytics queries.

### `admin_analytics_access_log`

Records which administrator opened which user's analytics profile, the requested section, and the access time. It does not store response bodies.

## Privacy and Security

### Prohibited data

The following values must never be persisted, returned, or logged by Analytics:

- Full API keys or authentication tokens
- Passwords or verification codes
- Prompt, instruction, strategy, portfolio, or form-input text
- Raw upstream provider response bodies
- Full raw IP addresses
- Raw User-Agent headers
- Encrypted credential ciphertext

Safe credential references may include provider ID, credential lifecycle outcome, and last four characters only when the existing public credential contract already permits them.

### Network pseudonymization

Analytics uses a dedicated deployment secret:

```text
ANALYTICS_PSEUDONYMIZATION_KEY
```

The request IP may exist transiently in process memory while an HMAC network identifier is calculated. The HMAC input includes the UTC calendar month so the identifier cannot link the same network indefinitely across retention periods. The raw value is immediately discarded and is never written to Analytics storage. Raw User-Agent headers are reduced to an allowlisted browser family and device category before storage.

If the key is absent or invalid, Analytics omits `network_hash`. It must never downgrade to plaintext storage. Country or region is optional and may only come from a trusted deployment-platform header; otherwise it is `Unknown`. The first version does not call an external IP geolocation provider.

### Authorization

All `/api/admin/analytics/*` endpoints use the centralized Admin dependency. Non-admin users receive `403`. Opening a user profile creates an Admin analytics access record.

The frontend ingestion endpoint requires an authenticated user, CSRF protection where required by the existing API pattern, rate limiting, and a strict event/property allowlist.

## Metrics

Default metrics exclude Admin accounts and accounts explicitly marked `analytics_excluded`. Admins may temporarily enable `Include internal accounts` for diagnosis.

### Active users, rolling seven days

Distinct users with at least one meaningful accepted page visit or server-authoritative business event in the preceding rolling seven days. Token refresh, background polling, and unattended heartbeat traffic are excluded.

### First successful run conversion

The percentage of mature signup-cohort users who complete a first successful backtest within seven days of signup. A cohort is mature only after its full seven-day observation window has elapsed, preventing recent signups from depressing the metric prematurely.

### Backtest success rate

```text
completed / (completed + failed)
```

User-cancelled runs do not enter the denominator. Queued or running records are not terminal and do not enter the calculation.

### Repeat run rate

The percentage of users with a first successful run who complete another successful run at least 24 hours later and no more than 30 days after the first success.

### Platform model cost

The sum of actual model-cost evidence for `platform_credits` execution only. BYOK usage appears separately as token and run usage and never contributes to ATL platform cost or ATL Credits debits.

## Explainable User States

Each user receives one state. Rules are evaluated in this precedence order:

1. **Blocked**
   - A recent attempted core action produced an unresolved condition that still prevents another run, such as no available billing lane, an administratively disabled provider, or an explicit account restriction. A new user who has not attempted a run remains `Onboarding` rather than `Blocked`.
2. **Needs Attention**
   - Three consecutive failed runs within 24 hours, an invalid default credential, or a run beyond its safe execution deadline.
3. **Dormant**
   - No meaningful accepted activity for 30 consecutive days.
4. **Onboarding**
   - No first successful backtest, while the account remains recently active and is not blocked or in need of attention.
5. **Active**
   - A first successful backtest exists and meaningful activity occurred within the previous 30 days.

Every snapshot returns:

```text
status
reason_code
human_readable_reason
evidence_event_ids
calculated_at
```

Status labels do not trigger automatic user mutations. They are explainable administrative support signals only.

## APIs

### Frontend ingestion

```text
POST /api/analytics/events
```

Accepts an allowlisted page event envelope. The server supplies user identity and received time, sanitizes every property, and returns a generic accepted response without echoing sensitive input.

### Admin queries

```text
GET /api/admin/analytics/overview
GET /api/admin/analytics/users
GET /api/admin/analytics/users/{user_id}
GET /api/admin/analytics/users/{user_id}/activity
```

Overview supports date range, billing mode, provider, model, and internal-account inclusion filters. User listing supports query, status, last-activity range, pagination, and sort. The base user-detail endpoint returns the profile header, current state, milestones, and summary cards. The activity endpoint accepts `section=timeline|runs|usage|sessions` and a cursor so detailed sections remain independently pageable.

Responses contain display-safe identities and Analytics fields only. They never expose secrets, ciphertext, raw provider bodies, full IP addresses, or arbitrary event properties.

## Freshness and Query Strategy

Raw accepted events become visible within one minute. Completed historical days read from `analytics_daily_rollups`; the current incomplete day reads raw events. `AnalyticsQueryService` merges both sources into one response.

Target response times are:

- Analytics Overview: under one second
- Initial User Analytics Profile: under 500 milliseconds
- Raw event visibility: under one minute

The UI displays `Last updated`. Sections load independently and use cursors for detailed history.

## Failure Isolation and Recovery

Analytics is observational, not part of a core transaction's success criteria. Server-authoritative events are emitted after the source operation commits. A failed Analytics append:

- does not fail login, credential operations, Agent mutations, backtests, or Credits settlement;
- records only a safe internal operational error;
- can be repaired from authoritative source tables when reconstruction is possible.

Stable `source_event_id` values prevent duplicates during retries and backfill.

If one Admin query fails, only that panel shows `This metric is temporarily unavailable.` Other panels remain usable. The page must not fail as a single all-or-nothing request.

## Historical Backfill

The instrumentation delivery includes an idempotent backfill for the previous 180 days using existing authoritative data:

- users
- agents
- run records
- model usage evidence
- Credits ledger entries

Backfilled records use `event_source = backfill` and deterministic source event IDs. The backfill does not invent historical page views, browser sessions, device metadata, region, or network hashes.

## Retention

A scheduled retention task deletes raw `analytics_events` older than 180 days in bounded batches. Daily non-identifying rollups remain available for long-term trends. Current user snapshots are recalculated rather than treated as immutable history. Admin profile-access records are retained for 365 days because they are a security audit trail, then deleted in bounded batches.

Retention failures are observable but do not block the application. Repeated failures must surface an operator-facing warning because unbounded event retention would violate the approved privacy contract.

## Delivery Plan

### Pull Request 1: Analytics Foundation

- SQLite and PostgreSQL schema and repository parity
- Event model, validation, and idempotency
- Authenticated frontend ingestion allowlist
- Session handling and network pseudonymization
- Subject exclusions, Admin access log, and retention service

No Admin Analytics tab ships in this pull request.

### Pull Request 2: Instrumentation and Metrics

- Account, credential, Agent, run, resource, and safe error instrumentation
- Metric calculations and daily rollups
- Explainable state snapshots
- Idempotent 180-day authoritative-data backfill
- Admin query services and APIs

No Admin Analytics tab ships in this pull request.

### Pull Request 3: Admin Analytics UI

- First-position `Analytics` Admin tab
- Overview filters, metrics, trends, health, friction, and attention table
- Dedicated User Analytics Profile
- Independent section pagination and partial-error states
- Link to existing account management

## Testing Strategy

### Foundation tests

- SQLite and PostgreSQL repository contract parity
- Event and property allowlists
- Idempotency and cursor behavior
- 180-day retention and permanent aggregate preservation
- Missing pseudonymization key omits network hash without plaintext fallback
- Secret canaries never appear in storage, responses, logs, or error messages

### Instrumentation and metric tests

- Server outcomes emit exactly one correct event after successful source commits
- Analytics append failure does not change source-operation outcomes
- Metric formulas use deterministic UTC boundary fixtures
- Internal accounts are excluded by default
- User-state precedence and evidence are deterministic
- Backfill is idempotent and never fabricates page or Session events

### API and frontend tests

- Non-admin Analytics access returns `403`
- User-profile access produces an Admin access record
- Filters, pagination, URL state, ARIA tabs, and keyboard navigation work
- One failed panel does not blank the page
- HTML and JavaScript responses contain no secret-bearing fields
- The Analytics page remains read-only

### End-to-end acceptance scenario

Use synthetic credentials and test data only:

1. Create a non-excluded user and record signup and major page visits.
2. Create an Agent and verify a fake BYOK credential through a fake adapter.
3. Record a failed run followed by a successful run.
4. Record a Platform Credits run with actual usage evidence and settlement.
5. Verify Overview, state reason, timeline, run history, BYOK/Platform split, cost, and Credits values agree with their authoritative sources.

The scenario must prove that BYOK never debits ATL Credits and that no real API key is required or displayed.

## Acceptance Criteria

- Admin opens `Analytics` as the first and default Admin tab.
- Overview presents a clear vertical hierarchy: snapshot, engagement, health/friction, and attention queue.
- Admin can open a dedicated read-only User Analytics Profile.
- New events appear within one minute.
- Metrics follow the documented formulas and default exclusions.
- User states are mutually exclusive, explainable, and supported by evidence.
- Raw event detail is deleted after 180 days while daily aggregates remain.
- Analytics failure never changes a core user-operation result.
- SQLite and PostgreSQL behavior is equivalent.
- No prohibited sensitive data appears in Analytics storage, APIs, logs, tests, commits, or UI.
