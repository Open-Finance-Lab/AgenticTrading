# Stripe Credits Test Mode

ATL can sell platform Credits through Stripe Checkout in Test Mode. This guide
is for local development and acceptance testing only. The application rejects
Stripe Live Mode secret keys.

Credits are units for ATL platform services such as future model runs and
backtests. They are not simulated portfolio capital, securities, stored cash,
or withdrawable value. Buying Credits never changes a paper-trading or backtest
account balance.

## What This Release Supports

- signed-in users can buy fixed $5, $10, $20, or $50 packages;
- signed-in users can buy a custom amount from $5.00 through $200.00;
- $1 buys exactly 1 Credit, with six decimal places stored as integers;
- only a verified Stripe webhook can add Credits;
- duplicate checkout requests and webhook replays are idempotent;
- ATL administrators can issue partial or full refunds against unused Credits;
- successful refunds create immutable negative ledger entries; and
- SQLite is used locally while `USERS_DATABASE_URL` selects Postgres in a
  durable deployment.

This release does not consume Credits, grant promotional Credits, provide
referral rewards, expire balances, or allow users to request their own refunds.

## Prerequisites

1. Create or use a Stripe account in a sandbox.
2. Install the [Stripe CLI](https://docs.stripe.com/stripe-cli) and run:

   ```bash
   stripe login
   ```

3. Obtain a Stripe Test Mode secret key from the Stripe Dashboard. Never put a
   real key in a tracked file, command output, issue, screenshot, or commit.
4. Install ATL's Python dependencies.

## Local Configuration

Run the Stripe webhook listener first:

```bash
stripe listen \
  --events checkout.session.completed,checkout.session.expired,checkout.session.async_payment_failed,refund.created,refund.updated,refund.failed \
  --forward-to http://127.0.0.1:8000/api/webhooks/stripe
```

The command prints a temporary webhook endpoint secret beginning with
`whsec_`. In a second terminal, set Test Mode values in the process environment
or an untracked local `.env` file:

```bash
export ATL_STRIPE_TEST_BILLING_ENABLED=1
export STRIPE_SECRET_KEY=sk_test_replace_with_your_test_key
export STRIPE_WEBHOOK_SECRET=whsec_replace_with_the_listener_secret
export PUBLIC_APP_URL=http://127.0.0.1:8000
export DATABASE_PATH=/tmp/atl-credits-test.db
```

Use a disposable database for local acceptance testing. Do not run billing
tests against `dashboard/storage/data/backtest.db` or a production database.

Start ATL from the repository root:

```bash
python -m uvicorn dashboard.backend.app:app --host 127.0.0.1 --port 8000
```

When billing is disabled, ATL still starts and all non-billing features remain
available. When billing is enabled but a key, webhook secret, or
`PUBLIC_APP_URL` is missing, the Credits page reports that billing is
unavailable. Unsafe key formats and `sk_live_` keys are rejected.

## Test a Purchase

1. Open `http://127.0.0.1:8000/app?view=credits` and sign in.
2. Select the $10 package and continue to Stripe Checkout.
3. Use Stripe's interactive test card `4242 4242 4242 4242`, any future expiry,
   any three-digit CVC, and any postal code. Never use a real card for testing.
4. Complete Checkout and return to ATL.
5. Confirm the page first shows a pending state if the redirect arrives before
   the webhook, then shows exactly `10.00 Credits`.
6. Confirm Recent activity contains one `Credit purchase` entry for `+10.00`.

The Checkout return URL is informational. Reloading it or changing its query
parameters cannot grant Credits. The signed `checkout.session.completed`
webhook is the source of truth.

## Test an Administrator Refund

The refund endpoint requires `users.role = 'admin'`. For a disposable local
SQLite account only, promote the test user in the temporary database:

```bash
sqlite3 /tmp/atl-credits-test.db \
  "UPDATE users SET role = 'admin' WHERE email = 'credits-test@example.com';"
```

Sign out and back in if the administrator section does not appear. Then:

1. Open Credits & Billing and find the paid order under Admin refunds.
2. Request a $4.00 refund.
3. Wait for Stripe CLI to forward the signed refund event.
4. Refresh Credits and confirm the balance is `6.00 Credits`.
5. Confirm the immutable ledger retains `+10.00` and adds `-4.00`.
6. Confirm the order is Partially Refunded and still has $6.00 refundable.

A submitted refund reserves the corresponding purchased Credits. The balance
changes only after Stripe reports a successful refund. Refund failures release
the reservation. ATL rejects over-refunds, duplicate refunds, ordinary-user
refund attempts, and refunds that cannot be matched to the original purchase.

## Security Checks

- Send a request without a valid `Stripe-Signature` header and confirm the
  webhook returns HTTP 400 without changing the balance.
- Replay the same Stripe event and confirm the ledger and balance do not change.
- Confirm one user cannot read another user's payment order.
- Inspect application logs and confirm they contain no secret, full webhook
  payload, card data, or customer payment details.
- Confirm the Stripe Checkout host is exactly `checkout.stripe.com` before the
  browser leaves ATL.

## Troubleshooting

### Stripe Test Mode billing is unavailable

Confirm all four settings are present in the same process that starts ATL:

```text
ATL_STRIPE_TEST_BILLING_ENABLED=1
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
PUBLIC_APP_URL=http://127.0.0.1:8000
```

Restart ATL after changing environment variables.

### Checkout succeeds but Credits stay pending

Keep `stripe listen` running and confirm it forwards events to
`/api/webhooks/stripe` with HTTP 200. The listener's `whsec_` value must match
`STRIPE_WEBHOOK_SECRET`. A webhook secret copied from a different endpoint does
not validate local CLI events.

### A refund stays submitted

Confirm the listener includes `refund.created`, `refund.updated`, and
`refund.failed`. Refresh the Credits page after Stripe reports the final event.

## Live Mode Gate

Do not enable real payments by replacing Test Mode keys. A separate approved
design is required before Live Mode work begins. That design must cover the
merchant identity, verified signup email, terms and privacy notices, refund and
tax policy, disputes and chargebacks, financial reconciliation, database
backups, monitoring and alerts, production secret management, and operational
ownership.

For current Stripe commands and test payment details, use the official
[webhook testing](https://docs.stripe.com/webhooks) and
[test cards](https://docs.stripe.com/testing) documentation.
