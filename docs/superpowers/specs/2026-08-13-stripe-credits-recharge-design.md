# Stripe Credits Recharge Design

Date: 2026-08-13

Status: Approved

Target branch: `feat/credits-recharge-design`

Base: `origin/main@c131f8c`

## 1. Context

Agentic Trading Lab (ATL) records estimated model costs but does not currently
have a user credit balance, a payment provider integration, or a billing
ledger. This design defines the first, deliberately narrow billing loop: a
signed-in user buys dollar-denominated ATL Credits through Stripe Test Mode,
sees the resulting balance and ledger history, and an ATL administrator can
issue a partial or full refund of unused purchased Credits.

Credits pay for ATL platform services in later loops. They are completely
separate from simulated portfolio cash. Adding 10 Credits must never add money
to a backtest or paper-trading portfolio.

## 2. Goals

The first release will:

- provide a dedicated Credits / Billing page for signed-in users;
- sell fixed packages of $5, $10, $20, and $50;
- accept a custom amount from $5.00 through $200.00, in one-cent increments;
- use Stripe-hosted Checkout in Test Mode without charging real money;
- grant one dollar-equivalent Credit for each dollar successfully paid;
- calculate and store values as integer micro-credits;
- maintain an append-only ledger instead of a mutable balance field;
- process Stripe events idempotently after verifying their signatures;
- support administrator-initiated partial and full refunds of unused purchased
  Credits; and
- provide SQLite and Postgres storage implementations with matching behavior.

## 3. Non-goals

The first release will not:

- enable Stripe Live Mode or collect real money;
- charge Credits for backtests, model calls, or other ATL activity;
- provide free models, signup grants, referral rewards, promotional Credits, or
  expiring Credits;
- support subscriptions, automatic top-ups, coupons, taxes, invoices, ACH,
  PayPal, Venmo, or cryptocurrency payments;
- allow user-initiated self-service refunds;
- treat Credits as transferable, withdrawable, or redeemable for cash; or
- add signup-email verification.

Backtest charging, free-model policy, signup and referral grants, and behavior
when a running task exhausts its Credits each require a separate design loop.

## 4. Confirmed Product Rules

| Rule | First-release decision |
|---|---|
| Payment provider | Stripe Hosted Checkout |
| Payment environment | Stripe Test Mode only |
| Access | Signed-in ATL users |
| Signup email verification | Deferred; mandatory before Live Mode |
| Fixed packages | $5, $10, $20, and $50 |
| Custom amount | $5.00 through $200.00, inclusive |
| Exchange rate | $1 paid = 1 ATL Credit |
| Stripe processing fee | Absorbed by ATL; the user receives the full purchased amount |
| Storage unit | 1 Credit = 1,000,000 micro-credits |
| Purchased-Credit expiry | Never expires |
| Balance source | Sum of immutable ledger entries |
| Refund initiator | An authenticated ATL administrator |
| Refund limit | Only the unused, unrefunded portion of the original purchase |
| Refund destination | The original Stripe payment method |

One cent is 10,000 micro-credits. For example, a $10.00 purchase creates a
`+10,000,000` ledger entry, and a later $4.00 refund creates a `-4,000,000`
entry. Floating-point values must not be used for authoritative payment or
Credit calculations.

## 5. User Flows

### 5.1 Purchase

```text
Sign in to ATL
  -> open Credits / Billing
  -> select $5/$10/$20/$50 or enter $5.00-$200.00
  -> ATL validates the selection and creates a pending local order
  -> ATL creates a Stripe Test Checkout Session
  -> browser opens Stripe-hosted Checkout
  -> user completes a test-card payment
  -> Stripe sends a signed webhook to ATL
  -> ATL confirms payment_status=paid and posts one purchase ledger entry
  -> Billing page displays the completed order, balance, and ledger entry
```

The Checkout success redirect is informational. It never grants Credits. The
returned Billing page polls the local order status until the verified webhook
has posted the purchase or the page reaches a bounded timeout.

### 5.2 Abandoned or failed payment

```text
User closes or cancels Checkout
  -> local order remains unpaid, then becomes expired or failed
  -> no Credit ledger entry is created
  -> balance remains unchanged
```

### 5.3 Administrator refund

```text
Administrator selects a paid order and refund amount
  -> ATL verifies the admin role and CSRF token
  -> ATL checks the order's unused, unrefunded purchased Credits
  -> ATL creates an idempotent refund request and reserves that amount
  -> ATL requests a refund from Stripe
  -> Stripe sends a signed successful-refund webhook
  -> ATL posts one negative refund ledger entry
  -> the order becomes partially_refunded or refunded
```

Users do not receive a self-service refund button. ATL administrators initiate
refunds from a protected administration surface so the platform can validate
the refundable Credit amount before asking Stripe to return money.

## 6. Architecture

```text
Credits / Billing UI
        |
        v
Authenticated Billing API -----> CreditsStore -----> USERS_DATABASE_URL
        |                               |
        |                               +-- credit accounts
        |                               +-- payment operations
        |                               +-- webhook receipts
        |                               +-- append-only ledger
        v
Stripe Checkout API
        |
        v
Signed Stripe Webhook ----------> Billing service transaction
```

### 6.1 Billing API

The API exposes these first-release operations:

- `GET /api/credits/balance` returns the current user's ledger-derived balance;
- `GET /api/credits/ledger` returns cursor-paginated ledger history;
- `POST /api/credits/checkout-sessions` accepts either a fixed package ID or a
  custom amount in cents and returns a Stripe Checkout URL;
- `GET /api/credits/orders/{order_id}` returns an authenticated user's local
  order status for the return-page poll;
- `GET /api/admin/credits/orders` returns cursor-paginated paid and refunded
  orders, including their currently refundable amounts, to administrators;
- `POST /api/admin/credits/refunds` creates an administrator refund request;
  and
- `POST /api/webhooks/stripe` receives signed Stripe events.

Browser-facing mutation endpoints use ATL's existing cookie authentication,
same-origin policy, and CSRF protection. The webhook endpoint is intentionally
unauthenticated by ATL sessions and instead requires a valid Stripe signature.
It must receive the raw request body needed by Stripe's verifier.

The checkout endpoint never accepts an authoritative number of Credits or a
Stripe price from the browser. For a fixed package, the server resolves the
allowlisted package ID. For a custom purchase, the server validates the integer
cent amount against the $5.00-$200.00 range and calculates Credits as:

```text
credits_micro = amount_usd_cents * 10,000
```

### 6.2 Billing service

The billing service owns checkout creation, payment state transitions, webhook
handling, balance queries, and refunds. Stripe-specific payload parsing stays
behind a small gateway interface so tests can use a fake provider without
network access.

Checkout Sessions include opaque local order and user identifiers as metadata,
but metadata is only a correlation hint. A webhook handler must resolve the
stored Stripe Checkout Session or PaymentIntent ID and verify that it matches
the local order before posting Credits.

Every Stripe API mutation uses a stable idempotency key derived from the local
operation ID. A timeout after Stripe accepted a request can therefore be
retried without creating a second Checkout Session or refund.

### 6.3 Credits store

Credits and payments are account entitlements, so their production store is
selected exclusively by `USERS_DATABASE_URL`, alongside ATL user identity. The
SQLite implementation uses the corresponding local users database. It must not
fall back to `CONTENT_DATABASE_URL` or `AGENT_RUNS_DATABASE_URL`.

The feature uses a focused `CreditsStore` rather than adding payment behavior
to the existing `UserStore`. SQLite and Postgres twins expose the same service
contract, constraints, transactions, and integer units.

## 7. Data Model

The schema contains five tables. Refund requests are payment operations and do
not form a second Credit ledger.

### 7.1 `credit_accounts`

| Field | Purpose |
|---|---|
| `user_id` | Immutable ATL user identity and primary key |
| `status` | `active` or an operational restriction state |
| `created_at` | Audit timestamp |

This table does not store a balance. An account is created lazily and
idempotently when its first balance or checkout operation is requested.

### 7.2 `credit_payment_orders`

| Field | Purpose |
|---|---|
| `id` | Opaque local order ID |
| `user_id` | Owner of the order |
| `stripe_mode` | `test` or `live`; first release accepts only `test` |
| `currency` | `usd` |
| `amount_usd_cents` | Authoritative amount charged |
| `credits_micro` | Authoritative Credits to grant |
| `status` | `pending`, `paid`, `expired`, `failed`, `partially_refunded`, or `refunded` |
| `stripe_checkout_session_id` | Unique Stripe Checkout reference |
| `stripe_payment_intent_id` | Unique Stripe payment reference when available |
| `created_at`, `updated_at`, `paid_at` | Audit timestamps |

The monetary fields are fixed after the pending order is created. Status is a
workflow projection; authoritative balance changes exist only in the ledger.

### 7.3 `credit_refund_requests`

| Field | Purpose |
|---|---|
| `id` | Local refund operation and Stripe idempotency key source |
| `payment_order_id` | Original purchase |
| `user_id` | Owner of the Credits |
| `requested_by_user_id` | Administrator who initiated the refund |
| `amount_usd_cents` | Requested Stripe refund |
| `credits_micro` | Credits reserved and later reversed |
| `status` | `pending`, `submitted`, `succeeded`, `failed`, or `cancelled` |
| `stripe_refund_id` | Unique Stripe refund reference |
| `created_at`, `updated_at`, `succeeded_at` | Audit timestamps |

Pending and submitted requests reserve their amount when calculating how much
can still be refunded. This prevents two simultaneous administrator requests
from refunding the same Credits.

### 7.4 `stripe_webhook_events`

| Field | Purpose |
|---|---|
| `stripe_event_id` | Primary idempotency key |
| `event_type` | Stripe event type |
| `livemode` | Must match the configured environment |
| `object_id` | Relevant Checkout, PaymentIntent, or Refund ID |
| `payload_sha256` | Audit fingerprint without duplicating the full Stripe payload |
| `outcome` | `processed`, `ignored`, or `rejected` |
| `created_at` | Processing timestamp |

Raw webhook payloads and payment-method details are not retained in the ATL
database. Application logs must not include secrets, full event payloads, or
customer payment details.

### 7.5 `credit_ledger_entries`

| Field | Purpose |
|---|---|
| `id` | Immutable entry ID |
| `user_id` | Account owner |
| `entry_type` | `purchase` or `refund` in Loop 1 |
| `amount_micro` | Signed, non-zero integer micro-credits |
| `payment_order_id` | Original purchase lot |
| `refund_request_id` | Present for a refund entry |
| `stripe_event_id` | Webhook that authorized the entry |
| `operation_key` | Unique semantic idempotency key |
| `created_at` | Posting timestamp |

The available balance is:

```sql
SUM(credit_ledger_entries.amount_micro) WHERE user_id = current_user_id
```

Application repository methods provide no update or delete operation for
ledger entries. Database constraints reject zero-value entries and duplicate
operation keys. Corrections are new compensating entries, never edits to
history.

Each purchase entry creates a purchase lot tied to its payment order. During
Loop 1, there are no spending debits, so the refundable amount is the purchase
entry minus successful refund entries and pending refund reservations. Before
any Credit-consuming feature is enabled, Loop 2 must define and implement debit
allocation against purchase and promotional lots. That allocation is required
to preserve the rule that only the unused portion of the original purchase can
be refunded.

## 8. Atomicity and Event Handling

For a signed `checkout.session.completed` event whose
`payment_status` is `paid`, one database transaction:

1. inserts the unique Stripe event receipt;
2. locks and validates the matching local order;
3. inserts the unique positive purchase ledger entry; and
4. changes the order status to `paid`.

If any step fails, the transaction rolls back so Stripe can retry. A duplicate
event ID or purchase operation key becomes a successful no-op and never grants
Credits twice. An event for an unknown order, the wrong mode, wrong currency,
wrong amount, or an unpaid session creates no ledger entry and is recorded as
ignored or rejected with a sanitized reason.

The equivalent successful-refund transaction is driven by a signed Stripe
refund event for the expected Refund object. It records the event, validates
the local refund request and Stripe refund ID, confirms that Stripe reports the
refund as succeeded, inserts one negative ledger entry, changes the refund
request to `succeeded`, and projects the order to `partially_refunded` or
`refunded`.

Stripe does not guarantee webhook delivery order. Handlers therefore use local
operation IDs, object IDs, and current states instead of assuming that one
event type always arrives before another. Events that cannot yet be correlated
return a retryable failure or enter a bounded reconciliation path; they do not
guess or post Credits.

## 9. Refund Rules

An ATL administrator may request a refund only when all of these are true:

- the original order is paid and belongs to the target user;
- the amount is a positive number of whole cents;
- the dollar refund and Credit reversal use the original 1:1 exchange rate;
- successful prior refunds plus pending refund reservations plus the new
  request do not exceed the original purchase;
- the original purchase lot contains enough unused purchased Credits; and
- the payment is still refundable through Stripe.

Promotional Credits introduced in later loops are not refundable. Stripe fees
are absorbed by ATL and are not removed from the Credit amount granted to the
user.

The normal operational rule is that refunds are created only through ATL, not
directly in the Stripe Dashboard. A Stripe-side refund performed outside ATL
is a reconciliation exception: the webhook is retained, the account is placed
in a restricted state if the corresponding Credits are unavailable, and an
administrator alert is raised. The exact negative-balance recovery policy must
be approved before Live Mode; Loop 1 can exercise this path only with test
payments.

## 10. Credits / Billing Experience

The dedicated page contains:

- the current available Credit balance, displayed to two decimal places;
- a visible Test Mode label;
- four fixed package controls;
- a custom dollar amount input with server-mirrored range validation;
- a clear action that opens Stripe Checkout;
- recent purchase/refund ledger history; and
- order states for pending, completed, failed, expired, and refunded payments.

The UI may display `$10.00` or `10.00 Credits`, while APIs return both an exact
integer micro-credit value and a formatted display string. The browser must not
perform authoritative arithmetic from formatted values.

The Checkout return state says that payment confirmation is pending until the
webhook is processed. It must not show Credits as available merely because the
URL is the success URL.

## 11. Security and Privacy

- `STRIPE_SECRET_KEY` and `STRIPE_WEBHOOK_SECRET` are server-only environment
  variables and must never enter source control, frontend bundles, logs, or API
  responses.
- Stripe publishable configuration is exposed only if the selected Checkout
  integration requires it; hosted redirect Checkout does not expose the secret
  key.
- The first release refuses Live Mode keys and rejects webhook events with
  `livemode=true`.
- Every browser mutation requires a valid ATL session and CSRF token.
- Every balance, order, and ledger query is scoped by immutable `users.id`, not
  mutable email.
- Refund endpoints require the existing ATL administrator role and record the
  initiating administrator.
- Checkout package and amount validation is repeated on the server.
- Webhook signatures are verified against the raw body before parsing or any
  database mutation.
- Responses and logs use sanitized error messages and never expose Stripe
  secrets or payment details.
- Rate limits apply to Checkout creation, order polling, and administrator
  refund creation.

## 12. Failure Behavior

| Failure | Required outcome |
|---|---|
| User cancels Checkout | Order becomes or remains unpaid; no Credits post |
| Payment fails | Order becomes failed; no Credits post |
| Success redirect arrives before webhook | Page shows pending and polls local status |
| Duplicate webhook | Existing operation is returned; no duplicate ledger entry |
| Invalid webhook signature | Request is rejected before parsing or mutation |
| Amount, currency, user, or mode mismatch | Event is rejected; no Credits post |
| Stripe API times out while creating Checkout | Retry uses the same idempotency key |
| Stripe API times out while refunding | Refund remains pending; retry uses the same idempotency key |
| Refund fails | Reservation is released; no negative ledger entry posts |
| Postgres or SQLite transaction fails | Entire local posting rolls back |
| Unknown or out-of-order Stripe object | No balance mutation; retry or reconcile |

## 13. Testing and Acceptance

### 13.1 Unit and store-contract tests

- cents convert to micro-credits exactly, including $5.00 and $200.00 bounds;
- values below $5.00, above $200.00, non-cent amounts, and floats are rejected;
- fixed package IDs resolve only through the server allowlist;
- balances equal the signed ledger sum;
- ledger entries cannot be updated or deleted through the store;
- duplicate event IDs and operation keys cannot create duplicate entries;
- refund reservations prevent concurrent over-refunds;
- non-admin users cannot create refunds; and
- SQLite and Postgres twins pass the same behavioral contract.

### 13.2 API and security tests

- unauthenticated balance, ledger, checkout, and order requests are rejected;
- users cannot read another user's order or ledger;
- missing or invalid CSRF tokens reject browser mutations;
- forged Stripe signatures and Live Mode events are rejected;
- client-side price and Credit tampering cannot alter the server amount;
- secrets and payment details do not appear in responses or sanitized logs; and
- checkout and refund retries reuse stable idempotency keys.

### 13.3 Stripe Test Mode flow

The release is accepted when this end-to-end sequence passes:

```text
Sign in
  -> buy $10.00 with a Stripe test card
  -> receive exactly 10.000000 Credits once
  -> replay the payment webhook and observe no balance change
  -> issue a $4.00 administrator refund
  -> receive exactly -4.000000 Credits once
  -> observe a 6.000000-Credit authoritative balance
  -> retain both immutable ledger entries
```

Additional end-to-end cases cover cancelled Checkout, failed payment, forged
webhook, duplicate refund, and custom purchases at both allowed boundaries.

## 14. Deployment and Operations

The first deployment uses a dedicated Stripe Test Mode account configuration,
test webhook endpoint, and test secrets. The application performs a startup
configuration check and exposes a health signal that says whether billing is
configured without revealing any secret value.

Before enabling Live Mode, ATL must complete a separate review and explicitly
approve all of the following:

- the receiving US merchant or legal entity and its Stripe account ownership;
- signup-email verification;
- terms of service, privacy policy, refund policy, tax treatment, and customer
  support process;
- production webhook monitoring, event replay, daily reconciliation, backups,
  and administrator alerting;
- dispute, chargeback, negative-balance, and account-restriction policy; and
- a security review of production secrets and administrator permissions.

Changing a Stripe key or environment variable alone must not activate real
payments. Live Mode requires a separate explicit application feature gate and
a reviewed deployment.

## 15. Later Loops

This design creates the funding foundation only. Follow-up designs proceed in
this order:

1. Credit consumption: preauthorization, metering, settlement, cancellation,
   and debit allocation for backtests and model calls.
2. Free access: free-model policy and signup grants.
3. Referrals: abuse-resistant invitation rewards for both users.
4. Runtime exhaustion: behavior when a running task reaches its Credit limit,
   including grace limits and user notifications.
5. Live billing readiness: legal, operational, email-verification, dispute, and
   production reconciliation controls.
