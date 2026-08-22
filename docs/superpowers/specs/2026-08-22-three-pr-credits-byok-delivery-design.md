# Credits and BYOK Three-PR Delivery Design

Date: 2026-08-22

Status: Approved for implementation planning

Current audit snapshot: `design/admin-sponsored-credits@880f2d3`

Target base: latest `origin/main`

## 1. Decision

The current mixed branch will not be delivered as one pull request. Its work
will be rebuilt as three bounded pull requests in this product-priority order:

1. secure BYOK Key Vault;
2. administrator-assigned Grant Credits; and
3. user-funded ATL model execution using purchased Credits.

PR 1 and PR 2 will be independent branches based on the latest `origin/main`.
PR 3 will begin only after PR 1 and PR 2 are merged because it depends on both
the approved provider registry and the grant/purchased accounting model.

BYOK in PR 1 means credential management only. Running a model with a user's
BYOK credential is explicitly outside these three pull requests.

## 2. Why the Existing Branch Must Be Split

The current branch grew from a credential-vault task into provider execution,
backtest admission, and Credits settlement. The zero-trust audit found useful
Vault foundations, but also found that the enlarged branch does not pass its
full test suite and contains confirmed verification and accounting defects.

The split restores a loop-engineering workflow for each deliverable:

```text
hypothesis -> smallest change -> falsifiable tests -> browser proof -> PR
```

No PR advances while its own lower-level gate is failing.

## 3. Branch and Review Topology

```text
latest origin/main
    |-- PR 1: feature/byok-key-vault
    |-- PR 2: feature/admin-grant-credits
    |
    `-- after PR 1 and PR 2 merge
          `-- PR 3: feature/purchased-credits-spending
```

Before rebuilding the branches, preserve `880f2d3` with a dedicated archive
branch so the experimental execution work remains recoverable. Use new
branches rather than rewriting or force-pushing the current branch.

The tracked local database currently modified by the audit must never enter a
commit. Restoring it from `HEAD` is a separate destructive action and requires
explicit user approval immediately before it is performed.

## 4. PR 1: Secure BYOK Key Vault

### 4.1 Goal

Let a signed-in user securely manage named API keys inside the API Keys tab of
Credits & Billing. Let an administrator manage the approved provider registry
and encrypted platform credentials on a separate admin surface.

### 4.2 Included scope

- seeded providers for OpenRouter, OpenAI, Anthropic, and Gemini;
- administrator-approved OpenAI-compatible providers;
- SQLite and PostgreSQL provider and credential repositories;
- encrypted user and platform credentials using
  `BROKER_TOKEN_ENCRYPTION_KEY`;
- multiple named user credentials per provider;
- no more than one verified default credential per user and provider;
- save, verify, reverify, set-default, and revoke operations;
- authenticated user APIs and separately authorized admin APIs;
- the fourth Credits & Billing tab, named API Keys;
- public responses containing credential metadata and the last four
  characters only; and
- complete user and admin tests for the credential lifecycle.

### 4.3 Security invariants

- Saving a credential fails closed when the encryption key is absent or
  invalid. Plaintext fallback is forbidden.
- A full key never appears in a response, URL, log, browser storage, validation
  error, or persisted plaintext column.
- Verification succeeds only when the provider returns the expected JSON
  envelope with at least one usable model identifier. A generic HTTP 2xx,
  malformed JSON, HTML, or an empty/unknown response shape is not verified.
- Redirects remain disabled. Provider requests may connect only to public IP
  addresses validated at connection time, including after DNS resolution, so
  DNS rebinding cannot reach loopback, link-local, private, reserved, or cloud
  metadata addresses.
- Provider mutations and their admin audit records commit atomically.
  Idempotency records bind the operation key to a canonical request digest;
  replaying the key with different input is a conflict.
- Seed initialization inserts missing official providers but does not overwrite
  administrator-managed configuration on application startup.
- Revocation destroys the encrypted secret while retaining a safe tombstone
  containing ownership, provider, label, last four characters, timestamps, and
  audit status. A revoked label may be reused for a new credential.
- Provider availability and credential status are rechecked before any future
  consumer can resolve a secret.

### 4.4 Excluded scope

- model generation with a user BYOK credential;
- backtest billing-mode selection;
- ATL Credits quotes, reservations, usage, or settlement;
- automatic provider discovery outside the approved registry; and
- storing model-list responses as an execution allowlist.

### 4.5 Acceptance gates

- SQLite credential repository contract passes.
- Live PostgreSQL credential repository contract passes without skips.
- Verification contract covers malformed JSON, HTML 2xx, empty lists,
  redirects, timeouts, authentication failure, and public/private DNS results.
- API contract proves ownership isolation, safe errors, request-size limits,
  rate limits, and last-four-only responses.
- Frontend static tests prove that secrets are not placed in URLs,
  `localStorage`, or reusable application state.
- Browser testing uses fake keys and a local controlled provider; no real key is
  entered or displayed.
- The complete backend suite and frozen route contract pass with zero failures.

## 5. PR 2: Administrator-Assigned Grant Credits

### 5.1 Goal

Let an administrator fund an audited Grant Pool, assign Grant Credits to a
user, and reclaim only the user's unused Grant Credits. Purchased Credits
remain separate and cannot be reclaimed by an ATL administrator.

### 5.2 Included scope

- separate `grant` and `purchased` ledger buckets;
- one audited Admin Grant Pool;
- pool fund, pool reduction, user assignment, and unused-grant reclaim;
- immutable integer micro-Credit ledger records;
- required actor, source, reason, operation, and idempotency evidence;
- authenticated user balance and activity views;
- separate Admin Credits controls and activity; and
- behaviorally equivalent SQLite and PostgreSQL implementations.

### 5.3 Accounting invariants

- One Credit equals 1,000,000 micro-Credits.
- Grant Pool and user bucket balances are derived from append-only ledgers.
- Pool and user buckets cannot become negative.
- Assign and reclaim update both sides atomically.
- Admin reclaim can reduce only available Grant Credits.
- Purchased Credits are never moved by a grant operation.
- An idempotency key is bound to the full canonical mutation request.
- Credits never change simulated portfolio cash, positions, or performance.

### 5.4 Excluded scope

- model price snapshots;
- run quotes, reservations, usage events, and settlement;
- BYOK or platform model execution; and
- removal of legacy run charging before the replacement spending path is
  ready.

### 5.5 Acceptance gates

- Repository and service contract tests pass against SQLite and live
  PostgreSQL.
- Concurrent assignment and reclaim tests prove no overdraft or duplicate
  ledger entry.
- User and admin API authorization and idempotency tests pass.
- Browser testing proves pool funding, assignment, reclaim, and user activity
  without touching purchased balances.
- The complete backend suite and frozen route contract pass with zero failures.

## 6. PR 3: Purchased Credits Spending

### 6.1 Goal

Let a signed-in user spend grant or Stripe-purchased Credits on eligible
ATL-funded model backtests with a visible estimate, bounded reservation,
per-call usage evidence, exact settlement, and reliable recovery.

### 6.2 Dependencies

- PR 1 supplies the approved provider registry and verified encrypted platform
  credentials.
- PR 2 supplies the separate grant/purchased balances and immutable ledgers.

### 6.3 Included scope

- server-managed model price snapshots;
- ATL-funded execution options and backtest quotes;
- concurrency admission before any Credit reservation;
- grant-first reservation and settlement;
- normalized per-call token and monetary-cost evidence;
- reservation top-up before the next billable call;
- partial settlement and zero-call release;
- persistent stale-reservation reconciliation;
- retirement of the legacy fixed-run Credit quota; and
- complete backtest UI selection, estimate, execution status, and activity.

### 6.4 Execution and accounting invariants

- A concurrency refusal creates no reservation and changes no balance.
- Every post-reservation failure path releases or settles the reservation, or
  writes a durable reconciliation incident.
- A quote identifier is either durably bound to the admitted provider, model,
  workload, and price snapshot or removed. Decorative, unconsumed quote IDs are
  forbidden.
- Malformed, negative, non-integer, or incomplete token usage is never marked
  reliable and enters review-required handling.
- Cached-input, reasoning-token, and provider-reported monetary-cost evidence
  is normalized according to each provider's documented semantics.
- A generation request is not retried after it may have reached a provider
  unless that provider supplies a verified idempotency mechanism. Attempts and
  request identifiers remain auditable.
- Settlement uses persisted usage evidence rather than an in-memory call
  counter.
- A production-invoked reconciliation path handles abandoned active
  reservations and exposes unresolved incidents to administrators.
- BYOK model execution is not enabled by this PR.

### 6.5 Acceptance gates

- Unit tests cover quote/admission binding, capacity refusal, every unwind
  branch, malformed usage, retry ambiguity, and reconciliation.
- SQLite and live PostgreSQL accounting contracts pass.
- The complete backend suite and frozen route contract pass with zero
  failures.
- A controlled fake provider proves successful, rejected, timed-out,
  malformed-usage, and missing-usage calls without real credentials or cost.
- Browser E2E proves top-up balance, quote, reservation visibility, completed
  settlement, and failure release.

## 7. Delivery Rules for Every PR

- Rebase or rebuild from the declared base; do not copy the current mixed HEAD
  wholesale.
- Keep commits in English and scoped so each commit has one falsifiable test
  target.
- Run `git diff --check`, focused tests, the full backend suite, live
  PostgreSQL contracts, and the relevant browser flow before opening the PR.
- Update frozen API route contracts in the same commit that intentionally
  changes routes.
- Never commit real API keys, local databases, `.superpowers/`, or `work/`.
- Include exact test counts and skipped-test reasons in the PR description.
- Stop the loop on the first failed lower-level gate; do not continue to
  browser polish while repository, service, or API tests are red.

## 8. Rollback and Recovery

- Preserve the current experimental execution branch before branch surgery.
- PR 1 and PR 2 must be independently revertible.
- PR 3 must use additive migrations and idempotent accounting transitions so a
  deployment rollback cannot duplicate or erase ledger evidence.
- Revoking a credential and disabling a provider remain immediate operational
  kill switches.
- Any settlement ambiguity fails closed into a visible incident rather than
  silently charging zero or releasing evidence-bearing reservations.
