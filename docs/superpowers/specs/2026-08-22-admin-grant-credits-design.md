# Admin Grant Credits Design

Date: 2026-08-22

Status: Approved for implementation planning

Target branch: `feature/admin-grant-credits`

Target base: `origin/main@abc4bc2`

## 1. Decision

PR2 adds administrator-managed sponsored Grant Credits to the existing
Credits domain. It extends the current purchased-Credits ledger with explicit
buckets and adds one audited Grant Pool ledger. The operation is deliberately
independent from Stripe purchases and from model execution.

The PR is bounded to accounting, authorization, presentation, and audit
history:

1. An administrator manually records sponsored budget in the Grant Pool.
2. An administrator assigns Grant Credits from that pool to a user.
3. An administrator reclaims only the user's unused Grant Credits.
4. Users can see their Grant and Purchased balances and activity.
5. No Credits are spent on model runs in this PR.

The existing integer `credits` entitlement used by the legacy run-count
metering path remains in place, but is relabeled `Legacy run Credits` in the
Admin UI. It is neither migrated into nor merged with Grant or Purchased
Credits. A later spending PR may retire it after the replacement path is live.

## 2. Product Boundaries

### Included

- one `grant` and one `purchased` bucket in the user Credits ledger;
- one manually funded, named Grant Pool;
- pool fund and reduction operations;
- user Grant assignment and unused-Grant reclaim;
- immutable integer micro-Credit entries with actor, source, reason, operation,
  idempotency, and timestamp evidence;
- authenticated user balance and activity APIs;
- separately authorized administrator pool, user-balance, mutation, and audit
  APIs;
- a top-level Admin page with `Users`, `Grant Credits`, and `Model Providers`
  sections, with Grant operations in the `Grant Credits` section;
- Grant/Purchased breakdown on the user Credits & Billing Overview and
  Activity surfaces;
- behaviorally equivalent SQLite and PostgreSQL repositories and contract
  tests; and
- browser verification with fake data only.

### Excluded

- model generation, model quotes, reservations, usage, or settlement;
- use of either Grant or Purchased Credits by a backtest;
- Stripe checkout, webhook, or refund behavior changes;
- automatic signup, referral, expiry, or promotional grants;
- provider API keys or model-provider execution;
- migration of the legacy run-count `credits` entitlement; and
- changing simulated portfolio cash, positions, orders, or performance.

## 3. Accounting Model

All authoritative amounts are signed integers in micro-Credits:

```text
1 Credit = 1,000,000 micro-Credits
```

The user ledger is append-only. Its balance projection is derived by bucket:

```text
grant_committed_micro = SUM(grant ledger amounts)
purchased_committed_micro = SUM(purchased ledger amounts)
grant_available_micro = grant_committed_micro
purchased_available_micro = purchased_committed_micro
total_available_micro = grant_available_micro + purchased_available_micro
```

PR2 has no model reservations, so committed and available values are equal in
this release. The schema and response names preserve the accounting boundary
needed by the future spending PR.

The Grant Pool projection is derived from its immutable ledger:

```text
pool_unallocated_micro = SUM(pool ledger amounts)
```

Pool entry signs are fixed:

| Operation | Pool change | User Grant change |
|---|---:|---:|
| `fund` | positive | none |
| `reduce` | negative | none |
| `assign` | negative | positive |
| `reclaim` | positive | negative |

`assign` and `reclaim` create a paired pool entry and user entry in one
database transaction. They share an operation ID, while each ledger line has
its own deterministic line-level idempotency key. A partial write is never
visible.

The Purchased bucket is immutable from the Grant API. Existing Stripe
purchase and refund rows remain in the Purchased bucket with their existing
payment, refund, and webhook references.

### Storage evolution

`credit_ledger_entries` gains an explicit bucket and the audit columns needed
by non-Stripe operations. Existing purchase and refund rows are backfilled as
`purchased` without changing their primary keys, signed amounts, operation
keys, payment references, refund references, webhook references, or creation
times. Their source remains Stripe/system evidence; the migration does not
invent an administrator actor for historical webhook mutations.

Payment, refund, and webhook references become nullable only so Grant rows can
exist. Database constraints continue to require the correct references for
purchase/refund entry types and forbid those references on Grant entry types.
New Grant entries require the administrator actor and all Grant audit fields.

SQLite performs the required table rebuild in one migration transaction and
copies the complete existing journal before replacing the old table.
PostgreSQL uses an idempotent add/backfill/constraint sequence in one migration
transaction. Migration contract tests start from the current production
schema, insert representative purchases and refunds, upgrade it, and compare
the historical rows field-for-field.

`credit_grant_pools` owns pool identity and status.
`credit_grant_pool_ledger_entries` owns pool mutations, their canonical request
digest, and the link to any paired user-ledger line. Both repositories expose
the same public methods and projections.

## 4. Invariants

- Pool unallocated balance never becomes negative.
- A user's committed Grant balance never becomes negative.
- A reclaim cannot exceed the user's unused Grant balance.
- Grant operations never change Purchased balance.
- A restricted account cannot receive a new Grant assignment; an administrator
  may reclaim its remaining unused Grant.
- Every mutation requires a positive integer amount in micro-Credits, a
  non-empty source, a non-empty reason, an actor, an operation ID, and an
  idempotency key.
- Repeating an identical idempotent mutation returns the original result and
  creates no new ledger lines.
- Reusing an idempotency key with any different user, amount, operation,
  source, reason, or target returns a conflict.
- A database error rolls back every line belonging to the logical operation.
- No Grant operation changes portfolio cash, holdings, orders, or performance.
- The legacy run-count entitlement remains independent from all Credits
  ledger projections.

## 5. Domain Interfaces

The Credits repository exposes projections and mutation methods behind the
existing SQLite/PostgreSQL store boundary. The service owns validation,
operation IDs, canonical request digests, and safe domain errors. Repository
methods enforce transactionality and balance constraints again so a route or
future caller cannot bypass the accounting rules.

UserStore remains the identity owner for email, display name, role, and account
pagination/search. The Admin user-balance query composes those identities with
a batch Credits projection; CreditsStore does not become a second user
directory.

The service returns typed results with:

- pool ID, pool name, pool status, and unallocated micro-Credits;
- user ID and Grant/Purchased/total balance projections;
- operation ID and ledger entry IDs;
- normalized operation type, amount, source, reason, actor, and timestamps; and
- a fixed `spending_enabled: false` indicator on user balance responses.

The current `balance_micro` and `display_credits` response fields remain as
backward-compatible aliases for the total balance. New explicit Grant,
Purchased, and total fields are additive, so the existing Top up and Activity
clients do not break during the migration.

No complete secret, payment credential, database exception, SQL text, or
internal stack detail appears in these results or errors.

## 6. HTTP API

### User routes

```text
GET /api/credits/balance
GET /api/credits/ledger
```

User routes are authenticated and scoped to the current account. They expose
only that account's bucket projections and ledger entries. The balance response
contains `grant`, `purchased`, `total`, and `spending_enabled: false`.

### Administrator routes

```text
GET  /api/admin/credits/grant-pool
GET  /api/admin/credits/grant-pool/activity
POST /api/admin/credits/grant-pool/fund
POST /api/admin/credits/grant-pool/reduce
GET  /api/admin/credits/users
POST /api/admin/credits/grants/assign
POST /api/admin/credits/grants/reclaim
GET  /api/admin/credits/activity
```

All administrator routes require an authenticated administrator and use the
existing CSRF protection for cookie-authenticated mutations. The UI's hidden
or visible state is never treated as authorization.

Mutation payloads use strict integer `amount_micro` values, a UUID
`client_request_id`, the target user or pool, and required `source` and
`reason` strings. Validation failures use the existing fixed HTTP error style;
domain conflicts return `409`, insufficient pool/user balances return `422`,
and authorization failures do not reveal target-account existence.

The browser amount control accepts a decimal Credit string with at most six
fractional digits, converts it to micro-Credits with string/decimal arithmetic,
and sends only the integer `amount_micro`. JavaScript floating-point arithmetic
is not used for authoritative amounts. The API rejects booleans, floats,
numeric strings, zero, and negative integers.

## 7. Frontend Structure

The existing top-level Admin entry remains the administrative boundary. Its
sections are:

- `Users`: existing account and quota controls;
- `Grant Credits`: the new pool, user balances, assign/reclaim drawer, and
  append-only activity; and
- `Model Providers`: the existing independent provider policy surface.

The Grant Credits section uses the approved prototype layout:

1. Pool summary with available, allocated, assigned-this-month, and
   reclaimed-this-month metrics;
2. searchable and filterable user balance table showing Grant and Purchased
   separately;
3. a user management drawer with `Assign`/`Reclaim`, amount, source, reason,
   constraints, and a confirmation step; and
4. a Grant activity table with operation, target, amount, actor, source,
   reason, and time.

The user Credits & Billing page remains the personal wallet surface. Overview
and Activity gain bucket labels and the explicit copy that model spending is
not enabled yet. Top up, API Keys, and Stripe flow behavior are unchanged.

## 8. Failure Handling

- Insufficient pool or user Grant balance refuses the mutation without writes.
- Empty or malformed source/reason is rejected before the repository call.
- The API rejects zero, negative, float, string, and boolean amounts; the UI
  also rejects Credit strings with more than six fractional digits before
  conversion.
- Duplicate identical requests return the stored result.
- Idempotency conflicts return `409` and do not alter either ledger.
- Any exception during a paired mutation rolls back both sides.
- Purchased balance is checked before and after Grant mutation tests and must
  be byte-for-byte unchanged in the projection.
- User and administrator responses remain safe and deterministic under store
  failures.

## 9. Testing and Acceptance Gates

### Domain and repository

- model tests cover strict amounts, required audit fields, and safe errors;
- service tests cover pool funding/reduction, assign/reclaim, insufficient
  balances, restricted accounts, idempotent replay, and conflict digests;
- repository contract tests run against SQLite and a live PostgreSQL service;
- concurrent assignment and reclaim tests prove no overdraft or duplicate
  operation; and
- transaction-failure tests prove paired writes roll back.

### API and frontend

- authentication, admin authorization, CSRF, ownership isolation, and fixed
  error responses are covered;
- user responses never expose another account or the Grant Pool;
- frontend tests pin the top-level Admin section and Legacy label;
- browser tests use fake accounts and verify desktop/mobile pool, table,
  drawer, confirmation, refresh, and activity behavior; and
- no real API key, payment method, or production account is used.

### Final PR gate

The PR is not merge-ready until the focused PR2 suite, full backend suite,
Packaging tests, CodeQL, and an independent read-only review all pass. The
modified local `dashboard/storage/data/backtest.db`, `.superpowers/`,
`AGENTS.md`, and `work/` remain outside the commit.

## 10. Future Boundary

The next spending PR may consume these projections with Grant-first ordering,
reservations, usage evidence, and settlement. It must not change the meaning
of the PR2 ledger entries or reintroduce a mutable balance field. It may retire
the legacy run-count entitlement only after the replacement spending path has
its own production acceptance gate.
