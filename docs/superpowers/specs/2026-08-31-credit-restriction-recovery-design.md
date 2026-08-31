# Credit Restriction Recovery and Error Transparency

## Goal

Make Platform Credits restrictions recoverable without weakening the refund
audit guard. A model-usage overage must be payable through a later purchase or
administrator Grant, while an out-of-band Stripe refund must remain an
administrator-reviewed condition. Users and administrators must see the
reason and amount behind a restriction instead of an opaque billing error.

## Current problem

`credit_accounts.status` currently has only `active` and `restricted`. Both an
LLM settlement overage and a Stripe refund reconciliation failure write the
same status, while checkout and Grant assignment reject every restricted
account. This creates a deadlock for an LLM overage: the account needs Credits
to repay the outstanding amount, but the restriction prevents adding them.
The existing `reinstate_account` operation is appropriate for refund review,
but it is not a recovery path for normal model metering.

## Design

### Restriction reasons

Persist a safe reason on the account:

- `llm_overage`: a settled model reservation has an unpaid amount because the
  provider's billable usage exceeded the reserved ceiling.
- `refund_reconciliation`: Stripe reported a refund that could not be safely
  correlated with a local refund request or refundable purchase lot.

Historical restricted rows without a reason are treated as
`refund_reconciliation` for safety. The public balance and admin-user
responses include `account_status`, `restriction_reason`, and the aggregate
unrecovered overage in Credits. No provider secrets, raw Stripe payloads, or
internal stack traces are exposed.

### Automatic recovery for model overage

When a purchase webhook or administrator Grant successfully adds Credits to a
restricted account whose reason is `llm_overage`, the same database
transaction:

1. records the new Credit entry;
2. applies available Credits to the oldest unrecovered reservation overages;
3. records idempotent recovery usage entries in the existing LLM usage ledger;
4. clears the restriction only when all overage is recovered.

The recovery consumes the same Grant-first bucket order used for model
reservations. If the new amount is insufficient, the account remains
restricted and reports the remaining amount. Checkout and Grant assignment
are allowed for `llm_overage` accounts so this state cannot deadlock. A
`refund_reconciliation` account remains blocked from both operations until an
administrator explicitly reinstates it.

Recovery is idempotent and atomic with the source purchase/Grant operation.
Retries cannot double-charge the account or clear a restriction prematurely.

### Error and UI behavior

Add a dedicated safe execution category for restricted Credits. A failed
Platform Credits run reports a human-readable message that the account is
paused, identifies whether the cause is an unpaid model-use overage or refund
review, and includes the remaining amount when available. The frontend keeps
the detailed provider-safe message and does not display Python exception
names or stack traces.

The Credits view explains the current state and, for an LLM overage, keeps the
purchase controls enabled with the amount still required. For a refund review
it shows that only an administrator can restore purchases. Admin Users shows
the account status, reason, outstanding amount, and a Reinstate action only
for refund-review accounts. Reinstate remains an explicit administrator
override and does not erase or rewrite ledger history.

## Compatibility and storage

Apply backward-compatible migrations to SQLite and PostgreSQL. Add the account
restriction reason and reservation overage-recovery amount with safe defaults.
Existing balance calculations continue to subtract every LLM usage entry, so
recovered overage is reflected in the authoritative balance projection.

The SQLite and PostgreSQL stores expose equivalent behavior. Existing numeric
ledger cursors, historical reservations, and public balance aliases remain
valid.

## Testing

Cover both stores and the API/UI contracts for:

- overage restriction and reason reporting;
- partial and complete recovery from a purchase;
- partial and complete recovery from an administrator Grant;
- replay-safe recovery and no double debit;
- refund-review accounts remaining blocked;
- administrator Reinstate behavior;
- safe, human-readable restricted-account execution errors;
- public responses never containing secrets or raw exception text.

Run the focused Credits, execution, router, and frontend contract suites, then
the complete backend test suite before opening the PR.
