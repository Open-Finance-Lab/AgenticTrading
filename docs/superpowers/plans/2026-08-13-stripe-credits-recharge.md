# Stripe Credits Recharge Implementation Plan

> Implement one task at a time with test-driven development. Do not enable
> Stripe Live Mode or add Credit consumption in this plan.

Date: 2026-08-13

Status: Awaiting implementation-plan review

**Goal:** Let a signed-in ATL user buy non-expiring dollar-equivalent Credits
through Stripe Test Mode, see an immutable balance history, and let an ATL
administrator refund the unused portion of a purchase.

**Architecture:** A focused Credits domain owns an append-only ledger, local
payment operations, webhook receipts, and SQLite/Postgres store twins selected
only by `USERS_DATABASE_URL`. A small Stripe gateway wraps the pinned official
Python SDK. Authenticated API routes call a billing service; the unauthenticated
webhook route authenticates Stripe by signature. A standalone frontend module
renders a dedicated Credits / Billing page without adding payment logic to the
existing trading code.

**Tech stack:** Python 3.12, FastAPI, Pydantic v2, Stripe Python SDK 15.4.0,
SQLite, Postgres/psycopg, vanilla JavaScript, CSS, pytest, and Stripe CLI for
manual Test Mode acceptance.

**Approved specification:**
`docs/superpowers/specs/2026-08-13-stripe-credits-recharge-design.md`

## Global Constraints

- Test Mode only. `sk_live_*` keys and `livemode=true` events must be rejected.
- No real-money launch, model charging, signup grant, referral, subscription,
  automatic top-up, or self-service refund.
- One US dollar paid grants exactly 1 Credit; authoritative values are integer
  cents and integer micro-credits (`1 Credit = 1,000,000 micro-credits`).
- Purchased Credits never expire. Simulated portfolio cash never reads or
  writes the Credits store.
- Balance is derived from immutable ledger entries. Repository code exposes no
  update or delete method for ledger history.
- Stripe Checkout redirects never grant Credits. Only a verified, paid webhook
  may post a purchase.
- User ownership always keys on immutable `users.id`, never email.
- Payment and ledger data use `USERS_DATABASE_URL` only. They must not fall back
  to `CONTENT_DATABASE_URL` or `AGENT_RUNS_DATABASE_URL`.
- Browser mutations use the existing cookie, same-origin, and CSRF controls.
  Stripe webhooks use signature authentication and raw request bytes.
- The application must continue to start when Stripe is unconfigured. Billing
  endpoints report a sanitized unavailable state; unrelated ATL features work.
- Automated tests never call Stripe's network and never read a developer's
  real environment keys.
- Keep all repository content, UI text, commits, and PR material in English.
- Never stage `.env`, databases, webhook payload dumps, or payment details.
- Run commands from the dedicated `feat/credits-recharge-design` worktree.

## Task 1: Add Billing Configuration and Stripe Gateway

**Files:**

- Modify: `requirements.txt`
- Modify: `.env.example`
- Modify: `dashboard/backend/tests/conftest.py`
- Create: `dashboard/backend/domain/credits/__init__.py`
- Create: `dashboard/backend/domain/credits/config.py`
- Create: `dashboard/backend/domain/credits/stripe_gateway.py`
- Create: `dashboard/backend/tests/domain/credits/test_config.py`
- Create: `dashboard/backend/tests/domain/credits/test_stripe_gateway.py`

### Steps

1. Write failing configuration tests for these environment variables:
   `ATL_STRIPE_TEST_BILLING_ENABLED`, `STRIPE_SECRET_KEY`,
   `STRIPE_WEBHOOK_SECRET`, and `PUBLIC_APP_URL`.
2. Require the explicit enable flag plus an `sk_test_*` key and `whsec_*`
   webhook secret before billing is ready. Reject `sk_live_*`, missing public
   URLs, non-HTTP(S) return URLs, embedded credentials, fragments, and any
   configuration that could activate Live Mode.
3. Ensure an unconfigured gateway does not fail module import or application
   startup; only a billing operation raises a typed `BillingUnavailableError`.
4. Add `stripe==15.4.0` to `requirements.txt`. Use an instance of
   `stripe.StripeClient`; do not set process-global `stripe.api_key`.
5. Define a provider-neutral gateway interface with methods for creating a
   one-time Checkout Session, creating a PaymentIntent refund, and verifying a
   signed webhook. Return small ATL-owned data objects instead of leaking
   Stripe SDK objects into the service.
6. Create Checkout in `payment` mode with USD `price_data`, quantity 1, card
   payment methods, a trusted `PUBLIC_APP_URL` success URL, a cancel URL, the
   local order ID as correlation metadata, and the local operation key as the
   Stripe idempotency key.
7. Create refunds against the stored PaymentIntent, in integer cents, with the
   local refund ID in metadata and as the Stripe idempotency key.
8. Verify webhooks with Stripe's official signature verifier over raw bytes and
   the configured endpoint secret. Map invalid signatures to a typed error.
9. Strip Stripe secrets from the test environment in `conftest.py`. Document
   placeholders in `.env.example`; never add a real key.
10. Run:

```bash
python -m pytest -q \
  dashboard/backend/tests/domain/credits/test_config.py \
  dashboard/backend/tests/domain/credits/test_stripe_gateway.py
```

11. Commit the task files with `feat(billing): add Stripe test gateway`.

## Task 2: Build the Append-Only SQLite Credits Store

**Files:**

- Create: `dashboard/backend/domain/credits/repository.py`
- Create: `dashboard/backend/tests/domain/credits/test_repository.py`

### Required store contract

The repository creates and operates these five tables from the approved spec:

- `credit_accounts`
- `credit_payment_orders`
- `credit_refund_requests`
- `stripe_webhook_events`
- `credit_ledger_entries`

Expose focused methods for:

- lazily ensuring an account;
- creating or returning an order by `(user_id, client_request_id)`;
- attaching the Stripe Checkout Session and PaymentIntent with compare-and-set
  semantics;
- reading one user-owned order and cursor-paginating a user's ledger;
- calculating a user's signed ledger sum;
- atomically settling a paid Checkout event;
- listing paid/refunded orders and refundable amounts for an administrator;
- atomically reserving a refund against one purchase lot;
- attaching a Stripe Refund ID;
- atomically settling or failing a refund event; and
- recording ignored/rejected Stripe events without posting Credits.

### Steps

1. Write failing tests for schema creation, a zero balance, and exact conversion
   of `$5.00`, `$10.00`, and `$200.00` to micro-credits.
2. Add literal SQLite DDL with status checks, non-zero ledger amounts, unique
   Stripe object IDs, unique event IDs, and unique semantic operation keys.
3. Add a unique `(user_id, client_request_id)` constraint. A retry of one
   browser purchase request must return the original local order.
4. Implement paid-Checkout settlement in one `BEGIN IMMEDIATE` transaction:
   insert the webhook receipt, validate order/user/currency/amount/mode, insert
   one positive ledger entry, and move the order to `paid`.
5. Make duplicate event IDs and purchase operation keys successful no-ops.
   Unknown, unpaid, mismatched, or Live Mode sessions must not change balance.
6. Implement cursor-paginated ledger reads and calculate the authoritative
   balance with SQL `SUM`; do not add a cached balance column.
7. Implement refund reservation in one transaction. Successful plus pending
   refunds must never exceed the original purchase or currently unused lot.
8. Implement successful refund settlement as one negative ledger entry and
   project the order to `partially_refunded` or `refunded`. Failed refunds
   release their reservation and post no ledger entry.
9. Test concurrent checkout retries, duplicate webhooks, concurrent refund
   attempts, partial refunds, full refunds, over-refunds, and ledger immutability.
10. Run:

```bash
python -m pytest -q dashboard/backend/tests/domain/credits/test_repository.py
```

11. Commit with `feat(billing): add immutable Credits ledger`.

## Task 3: Add the Postgres Store Twin and Parity Guards

**Files:**

- Create: `dashboard/backend/domain/credits/repository_postgres.py`
- Modify: `dashboard/backend/domain/credits/repository.py`
- Modify: `dashboard/backend/tests/test_store_twin_parity.py`
- Create: `dashboard/backend/tests/domain/credits/test_repository_postgres.py`

### Steps

1. Register `CreditsStore` and `PostgresCreditsStore` in the static twin parity
   registry before creating the Postgres implementation. Run the parity tests
   and confirm that they fail for the missing twin.
2. Implement literal Postgres DDL with the same table and column contract. Use
   enforced foreign keys to `users(id)`, `BIGINT` for integer micro-credits,
   row locking for settlement/refund transactions, and pooled psycopg
   connections.
3. Make the factory select Postgres exclusively from `USERS_DATABASE_URL`.
   Invalid or unreachable configured URLs fail loudly and sanitize credentials;
   `CONTENT_DATABASE_URL` and `AGENT_RUNS_DATABASE_URL` must have no effect.
4. Match every public SQLite method's name, arguments, defaults, return shape,
   transaction behavior, and error semantics.
5. Run static parity tests without Postgres:

```bash
python -m pytest -q dashboard/backend/tests/test_store_twin_parity.py
```

6. When `TEST_POSTGRES_URL` is available, run behavioral tests that create an
   isolated schema, exercise purchase/refund idempotency and concurrency, and
   clean up only that schema:

```bash
python -m pytest -q \
  dashboard/backend/tests/domain/credits/test_repository_postgres.py
```

7. Commit with `feat(billing): add Postgres Credits store`.

## Task 4: Implement Purchase and Refund Services

**Files:**

- Create: `dashboard/backend/domain/credits/models.py`
- Create: `dashboard/backend/domain/credits/service.py`
- Create: `dashboard/backend/tests/domain/credits/test_service.py`

### Steps

1. Define typed request/result models and domain errors. Accept either one
   server allowlisted package ID (`usd_5`, `usd_10`, `usd_20`, `usd_50`) or one
   integer custom amount in cents from 500 through 20,000, never both.
2. Require a UUID client request ID for checkout creation. Resolve the amount
   server-side, create/get the local order, then call the gateway using a stable
   order-derived Stripe idempotency key. A network retry must return the same
   order and Checkout URL.
3. Implement paid webhook handling for signed
   `checkout.session.completed` events only when `payment_status=paid`. Validate
   local order ID, Checkout Session ID, PaymentIntent ID, USD currency, integer
   amount, Test Mode, and user ownership before calling the atomic store method.
4. Ignore unsupported signed events with a recorded, sanitized outcome. Invalid
   signatures never reach business parsing or storage.
5. Implement administrator refund creation: verify the order and integer-cent
   amount, reserve the refundable purchase lot, call Stripe with a stable local
   refund idempotency key, and attach the returned Refund ID.
6. Handle `refund.created`, `refund.updated`, and `refund.failed`. Post the
   negative ledger entry only for the expected Refund object with
   `status=succeeded`; release the reservation for a confirmed failure.
7. For a Stripe-side refund without a local request, correlate it by
   PaymentIntent. If the purchase lot is still available, create a reconciliation
   refund record and reverse the Credits once. Otherwise mark the account
   restricted and surface the exception in the administrator order list; never
   silently leave refunded money as spendable Credits.
8. Keep Stripe SDK objects, raw webhook payloads, card details, emails, and
   secrets out of the repository and logs. Persist only approved identifiers,
   amounts, hashes, states, and timestamps.
9. Use a fake gateway to test success, network timeout/retry, tampered metadata,
   wrong amount/currency/mode, duplicate and out-of-order events, refund failure,
   and the out-of-band reconciliation path without calling Stripe.
10. Run:

```bash
python -m pytest -q dashboard/backend/tests/domain/credits/test_service.py
```

11. Commit with `feat(billing): orchestrate Credit purchases and refunds`.

## Task 5: Expose Authenticated Billing and Signed Webhook APIs

**Files:**

- Create: `dashboard/backend/api/routers/credits.py`
- Modify: `dashboard/backend/api/router.py`
- Modify: `dashboard/backend/tests/test_app_composition.py`
- Create: `dashboard/backend/tests/test_credits_api.py`
- Modify: `dashboard/backend/tests/test_csrf.py`
- Modify: `dashboard/backend/tests/test_object_authz.py`

### Route contract

```text
GET  /api/credits/balance
GET  /api/credits/ledger
POST /api/credits/checkout-sessions
GET  /api/credits/orders/{order_id}
GET  /api/admin/credits/orders
POST /api/admin/credits/refunds
POST /api/webhooks/stripe
```

### Steps

1. Write failing route-contract tests and add the seven approved routes to the
   application's frozen route list.
2. Require `get_current_user` for balance, ledger, checkout, and user order
   reads. Scope every query by `current_user["id"]`; another user's order must
   look nonexistent.
3. Require `current_user["role"] == "admin"` for administrator order listing
   and refund creation. Return 403 to ordinary users.
4. Keep browser route functions synchronous so blocking SQLite/Postgres and
   Stripe work stays in FastAPI's thread pool. The webhook may be async only to
   read raw body bytes; offload blocking service work to a worker thread.
5. Apply bounded per-user rate limits to checkout creation, order polling, and
   administrator refunds. Return 429 with `Retry-After` without extending a
   rejected request's window.
6. Let the existing middleware enforce CSRF on cookie-authenticated checkout
   and refund requests. Confirm bearer-script behavior remains compatible.
7. Read the webhook body exactly once as bytes, read `Stripe-Signature`, verify
   it before parsing business data, and return a fast 2xx for a processed,
   duplicate, or intentionally ignored signed event.
8. Map typed domain errors to sanitized 400/401/403/404/409/422/429/503
   responses. Never return provider exceptions or secret-bearing config text.
9. Add API tests for login requirements, cross-user isolation, admin gates,
   CSRF, amount tampering, duplicate requests, forged signatures, missing
   configuration, and unrelated-app startup without Stripe.
10. Run:

```bash
python -m pytest -q \
  dashboard/backend/tests/test_credits_api.py \
  dashboard/backend/tests/test_csrf.py \
  dashboard/backend/tests/test_object_authz.py \
  dashboard/backend/tests/test_app_composition.py
```

11. Commit with `feat(billing): expose Credits API`.

## Task 6: Build the Dedicated Credits / Billing Page

**Files:**

- Create: `dashboard/frontend/js/credits.js`
- Modify: `dashboard/frontend/app.html`
- Modify: `dashboard/frontend/app.js`
- Modify: `dashboard/frontend/styles.css`
- Create: `dashboard/backend/tests/test_credits_frontend.py`
- Modify: `dashboard/backend/tests/test_frontend_fast_boot.py`

### Page behavior

- Add `credits` to the shared boot-navigation map and valid page states.
- Add `Credits & Billing` to the signed-in account menu. Do not consume a
  primary navigation slot.
- Add a separate `creditsView` with a visible `Test Mode` label.
- Show the ledger-derived balance to two decimal places while retaining the
  exact integer value in application state.
- Render fixed $5/$10/$20/$50 controls, a custom $5.00-$200.00 input, purchase
  history, refunds, pending states, and errors.
- Generate one UUID client request ID per deliberate purchase click and reuse it
  for that click's retries. Redirect only to an HTTPS Stripe Checkout URL
  returned by the backend.
- On `?view=credits&order_id=...`, poll the local order with bounded backoff.
  Display `Payment confirmation pending` until the webhook posts the purchase;
  never increase balance from the success URL.
- For an administrator, show a compact orders/refund panel. Ordinary users must
  not receive or see its controls.

### Steps

1. Write failing source-contract tests for navigation, signed-in menu entry,
   Test Mode label, package values, custom bounds, external checkout redirect,
   bounded polling, text-safe rendering, and admin-only controls.
2. Put Credits-specific API calls, state, rendering, event handling, and polling
   in `js/credits.js`. Expose only a small `window.CreditsPage` interface that
   `navigateToPage` calls on page entry.
3. Use the existing shared `API` wrapper and CSRF helper. Never render API data
   with unsanitized `innerHTML`; use DOM creation and `textContent`.
4. Keep the layout operational and compact: balance header, purchase controls,
   and ledger table are unframed page sections; repeated ledger rows and the
   administrator refund dialog may use restrained cards/modal treatment.
5. Add loading, empty, pending, failed, expired, partially refunded, fully
   refunded, unconfigured, signed-out, and restricted-account states.
6. Use existing CSS variables and responsive breakpoints. Verify that custom
   amount controls, tables, and buttons do not overflow a 375px viewport.
7. Bump `app.js?v=`, `styles.css?v=`, and the new `credits.js?v=` cache busters
   from the values on current main at implementation time. Update exact pins in
   frontend tests.
8. Run:

```bash
python -m pytest -q \
  dashboard/backend/tests/test_credits_frontend.py \
  dashboard/backend/tests/test_frontend_fast_boot.py \
  dashboard/backend/tests/test_frontend_xss_guards.py
```

9. Commit with `feat(billing): add Credits purchase page`.

## Task 7: Add Automated End-to-End Coverage

**Files:**

- Create: `dashboard/backend/tests/integration/test_credits_checkout_flow.py`
- Modify: `dashboard/backend/tests/conftest.py`

### Steps

1. Build an in-process integration fixture with a temporary SQLite users/Credits
   database and fake Stripe gateway. Do not patch internal service methods; fake
   only the external provider boundary.
2. Test the complete purchase sequence: signup/login, create a $10 Checkout,
   follow the fake provider result, deliver a signed paid event, observe exactly
   `+10,000,000` micro-credits, replay the event, and observe no change.
3. Promote the test user to the existing `admin` role in the isolated database,
   create a $4 refund, deliver a successful refund event, and observe exactly
   `-4,000,000` plus a `6,000,000` balance.
4. Test cancelled and failed Checkout, forged webhook, tampered amount, cross-user
   order access, duplicate refund, over-refund, and provider timeout/retry.
5. Assert that simulated portfolio cash is unchanged through the whole flow.
6. Run:

```bash
python -m pytest -q dashboard/backend/tests/integration/test_credits_checkout_flow.py
```

7. Commit with `test(billing): cover Credits checkout lifecycle`.

## Task 8: Run Full Regression and Browser QA

### Automated regression

1. Run all backend tests:

```bash
python -m pytest -q dashboard/backend/tests
```

2. Run packaging tests:

```bash
python -m pytest -q packaging/agentictrading/tests
```

3. Run repository checks:

```bash
git diff --check
git status --short
git diff --stat origin/main...
git grep -nE 'sk_(test|live)_[A-Za-z0-9]|whsec_[A-Za-z0-9]' -- . ':!.env.example'
```

4. Confirm no `.env`, SQLite database, raw Stripe event, card data, screenshots,
   local log, or unrelated artifact is staged.

### Local browser QA with the fake gateway

1. Start ATL on an unused local port with temporary SQLite data and the fake
   gateway enabled for development tests.
2. Test signed-out access, signed-in balance, all fixed packages, custom amount
   bounds, pending return state, ledger history, admin refund, and error states.
3. Inspect at desktop and 375px mobile widths. Check for clipping, overlap,
   layout shift, broken focus order, console errors, and failed network calls.
4. Stop every local process started for QA.

## Task 9: Perform Real Stripe Test Mode Acceptance

This task uses only Stripe Test Mode. It must not run until the automated suite
and fake-gateway browser QA pass.

1. Put Test Mode values in the local untracked `.env` or process environment:

```text
ATL_STRIPE_TEST_BILLING_ENABLED=1
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
PUBLIC_APP_URL=http://127.0.0.1:<port>
```

2. Start ATL on that port with a temporary local database.
3. Start Stripe CLI webhook forwarding to:

```text
http://127.0.0.1:<port>/api/webhooks/stripe
```

4. Sign in, open `Credits & Billing`, buy $10 with Stripe's documented Test
   Mode card `4242 4242 4242 4242`, any future expiry, and any CVC.
5. Verify the browser first shows pending if the redirect wins the race, then
   shows exactly 10 Credits after the signed webhook.
6. Replay the same Stripe event and verify the balance remains 10 Credits.
7. Promote the local test account to `admin`, refund $4 through ATL, and verify
   the ledger retains `+10` and `-4` entries with a 6-Credit balance.
8. Exercise cancelled Checkout, a declined Stripe test card, invalid webhook
   signature, and a duplicate refund.
9. Inspect sanitized server output. Confirm no secret, full event payload, card
   data, or customer payment details appear.
10. Stop the server and Stripe CLI. Delete the temporary database and local test
    customer/payment data as appropriate; never commit any of it.

## Task 10: Final Documentation and Delivery

**Files:**

- Modify: `README.md` only if its current setup section is the established home
  for optional backend integrations
- Modify: `.env.example`
- Create: `docs/integrations/stripe-credits-test-mode.md`

### Steps

1. Document the Test Mode setup, Stripe CLI forwarding, test purchase/refund
   flow, configuration health behavior, and troubleshooting without embedding
   any account-specific value.
2. State explicitly that Credits are platform-service units, not simulated
   trading capital, securities, stored cash, or withdrawable value.
3. Include the gate that Live Mode requires a separate approved design covering
   merchant identity, verified signup email, terms/privacy/refunds/tax, disputes,
   chargebacks, reconciliation, backups, alerts, and production secret review.
4. Re-run the full tests and repository secret scan.
5. Commit with `docs: explain Stripe Credits test flow`.
6. Review `git diff origin/main...` file by file. Push and create a PR only after
   the user separately authorizes implementation, remote push, and PR creation.

## Completion Criteria

- A signed-in user can buy $5/$10/$20/$50 or a custom $5.00-$200.00 amount in
  Stripe Test Mode and receives exactly one immutable purchase ledger entry.
- The success redirect alone never changes balance.
- Duplicate checkout requests, Stripe API retries, and webhook replays never
  double-credit or double-refund.
- A protected ATL administrator can partially or fully refund only the unused,
  unrefunded purchased lot; the refund produces one immutable negative entry.
- Balance is an integer ledger sum and purchased Credits never expire.
- Cross-user reads, ordinary-user refunds, CSRF failures, forged signatures,
  amount tampering, wrong mode/currency, and over-refunds are rejected.
- SQLite and Postgres stores pass the same contract and behavioral tests.
- ATL starts and all non-billing features work when Stripe is unconfigured.
- Simulated portfolio cash remains unchanged by all billing operations.
- Automated tests, real Stripe Test Mode acceptance, desktop/mobile browser QA,
  and secret scanning pass.
- No Live Mode payment, Credit consumption, promotional grant, referral, or
  self-service refund is introduced.
