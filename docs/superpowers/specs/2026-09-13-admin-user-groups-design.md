# Admin User Groups and Source Analysis

## Status

Approved direction: independent `user_group` dimension, six visible values.

## Goal

Give ATL administrators a plain, reliable answer to “where did these users
come from?” and let them correct that classification without changing the
existing lifecycle, billing, or legacy acquisition semantics.

The first release covers:

1. A canonical `user_group` value on every account.
2. An admin-only editor in Account Management.
3. A Group filter and six-row Group Summary in Admin Analytics.

It does not add event tracking, machine-learning segmentation, or a second
analytics architecture.

## Product semantics

The only allowed values are:

| Stored value | Display label | Meaning |
| --- | --- | --- |
| `internal` | Internal | ATL developers, SecureFinAI members, and RCOS/course/research students |
| `invited` | Invited | Friends or peripheral students explicitly invited to try ATL |
| `organic` | Organic | People who arrived through GitHub, Discord, the website, or another self-directed path |
| `competition` | Competition | An official competition team |
| `partner` | Partner | A partner organisation such as 同花顺, AWS, or NVIDIA |
| `unknown` | Unknown | The source has not been confirmed yet |

New and historical accounts default to `unknown`. An administrator can change
the value later. No automatic inference is performed from the old
`student/community/friend/competition/unknown` acquisition source because those
values are not one-to-one with the new taxonomy.

`user_group` is deliberately separate from:

- Paid status, which is calculated from Purchased Credits.
- Lifecycle status (`new`, `onboarding`, `growing`, `core`, `at_risk`, `dormant`).
- Operational status (`blocked`, `needs_attention`, `healthy`).
- Existing acquisition `source` and `cohort` data, which remains untouched.

The existing `include_internal` control keeps its current meaning (the
analytics exclusion list); selecting Group `Internal` does not silently change
that control.

## Architecture

### Persistence

Add `user_group TEXT NOT NULL DEFAULT 'unknown'` to the account `users` table.

- SQLite and Postgres schemas receive the same logical column.
- Existing deployments use a lazy `ALTER TABLE ... ADD COLUMN` migration.
- The value is validated at the store boundary, not only in the HTTP layer.
- No database file or fixture is edited as part of the change.

The account store remains the source of truth because both Account Management
and Analytics already read user rows from it. This avoids a separate join table
and prevents an N+1 lookup for every analytics row.

### Admin API

Extend `PATCH /api/admin/users/{user_id}` with:

```json
{ "user_group": "organic" }
```

The existing atomic role/entitlement patch remains atomic; a request that
changes several supported fields commits them together. Explicit `null` and
unknown group values are rejected. The response includes the saved group in
the admin projection. `GET /api/admin/users` and `GET /api/admin/users/{id}`
also include it. Public account/session payloads do not expose this admin-only
field.

The mutation writes an operator audit line containing actor, target, old group,
and new group. It does not create a new analytics event stream.

### Analytics API and query layer

Add `user_group` to the value analytics filter model and its URL-backed query
state. The filter applies consistently to existing value/priority-user
queries.

Add a dedicated admin endpoint:

`GET /api/admin/analytics/groups?from=YYYY-MM-DD&to=YYYY-MM-DD&user_group=...`

The response always contains six rows in the fixed taxonomy order, including
zero rows, plus availability metadata. Each row contains:

- `group` and display `label`
- `users`
- `successful_run_users`
- `repeat_users`
- `total_runs`
- `atl_cost_micro_usd`
- `paid_users`

Definitions:

- Successful run user: at least one successful backtest in the selected period.
- Repeat user: successful backtests on at least two different UTC dates in the
  selected period.
- Paid user: Purchased Credits greater than zero, using the existing credits
  ledger.
- ATL cost: settled platform model cost; BYOK usage is excluded.

The summary uses the existing batched value/credits/run stores and does not
expose prompts, token payloads, or provider responses.

### Frontend

#### Account Management

Add a `Group` column to the canonical Account Management table. Render a native
`select` with the six labels, preserve the current value while a save is in
flight, and restore it with an inline error if the request fails. On success,
update the in-memory row from the server response. The hidden legacy Users
table is not given a second editor, so there is one obvious source of truth.

#### Analytics

Add a `User group` select beside the existing value filters. Changes update the
URL-backed filter state and refresh the affected data without changing the
existing date-range behavior.

Place `Group summary` near the top of the value overview, before the deeper
retention/commercial disclosures. Use a compact table with one visual bar per
row (user count as the bar scale) and aligned numeric columns, rather than six
detached number cards. The fixed rows make zero-valued categories visible and
keep comparisons easy at a glance. The table remains readable on narrow
screens via horizontal scrolling, with an accessible caption and a text table
fallback.

All copy remains English to match the ATL shell and existing analytics UI.

## Data flow

1. Account store bootstraps or lazily migrates `users.user_group`.
2. Admin list/get projections include the validated group.
3. An admin changes a row’s native select; the frontend sends the single PATCH.
4. The store validates and atomically commits the group, then returns the row.
5. Analytics loads eligible user rows in batches, groups them by `user_group`,
   and combines existing run, success, repeat, credits, and cost facts.
6. The frontend renders all six rows and preserves partial availability/error
   behavior already used by Analytics.

## Error and compatibility behavior

- Missing or malformed group values read as `unknown` at the migration boundary;
  invalid writes return HTTP 422.
- A missing user returns the existing HTTP 404 contract.
- Analytics failures show the existing section-level unavailable state while
  leaving other overview sections usable.
- Old acquisition source/cohort records remain readable and unchanged.
- SQLite and Postgres stores must return the same admin payload shape.

## Verification

Focused tests will cover:

1. SQLite schema migration, defaulting, validation, and atomic patch behavior.
2. Postgres twin DDL/query parity and admin projection shape.
3. Admin API success, 404, explicit-null, and invalid-enum responses.
4. Group summary definitions, fixed six-row ordering, zero groups, and filters.
5. Frontend source/DOM contracts for the Group select, URL filter, table, and
   error recovery.
6. A local browser smoke test with a temporary `DATABASE_PATH` and a screenshot
   of Account Management plus the Analytics summary.

## Deployment and acceptance

The repository has no Render preview service. Before merge, use local API and
admin-browser smoke tests plus the protected Vercel Preview for static UI
checks. Do not claim an end-to-end preview because Vercel Preview proxies API
requests to the current production backend.

After CI passes and the new branch is reviewed, merge `main`; the existing
workflow triggers the Render deploy hook. Wait for the Render build, then use a
real administrator session to:

1. Read a user row and confirm `Unknown` is shown.
2. Change it to `Organic`, save, refresh, and confirm persistence.
3. Open Analytics, filter `Organic`, and confirm the six-row summary and counts
   load without changing unrelated filters.

Only after those checks pass should the GitHub PR be reported as complete.

