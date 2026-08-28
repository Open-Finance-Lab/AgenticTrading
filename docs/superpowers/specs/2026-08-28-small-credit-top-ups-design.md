# Small Credit Top-Ups Design

**Date:** 2026-08-28
**Status:** Approved for planning
**PR:** #410 (`feature/default-user-credits`)

## Goal

Limit new ATL Credit purchases to small test amounts from **$0.50 through
$5.00**, inclusive. The Credits page must offer useful small presets while the
backend remains the authoritative enforcement boundary.

## Context

The current Credits page offers $5, $10, $20, and $50 packages and accepts
custom purchases from $5 through $200. Those amounts are unnecessarily large
for product testing. The application uses a fixed one-to-one rate: one US
dollar purchases one ATL Credit.

Stripe Checkout cannot create a useful zero-dollar top-up. The requested
"$0-$5" range is therefore represented by the smallest payable amount,
$0.50, through $5.00. A zero or empty amount is an invalid selection and must
not create a Checkout Session.

## Decisions

### Preset Packages

Replace the existing packages with four server-allowlisted packages:

| Package ID | Charge | Credits |
| --- | ---: | ---: |
| `usd_0_50` | $0.50 | 0.5 |
| `usd_1` | $1.00 | 1 |
| `usd_2` | $2.00 | 2 |
| `usd_5` | $5.00 | 5 |

The $1 package is selected by default. Package IDs remain symbolic API values;
the backend resolves each ID to cents and ignores any client-provided credit
quantity.

### Custom Amount

The custom input accepts decimal US-dollar values that resolve to an integer
number of cents from 50 through 500, inclusive. The UI copy, placeholder,
input attributes, client validation, and server validation use the same range:
`$0.50-$5.00`.

Values below $0.50, above $5.00, malformed decimals, fractional cents, booleans,
and floating-point JSON values are rejected. The browser must not be treated as
the security boundary; a direct API request is subject to the same backend
limit.

## Architecture

### Backend Contract

`dashboard/backend/domain/credits/models.py` remains the authoritative price
allowlist. It will:

- replace the large package IDs and their cent values with the four small
  packages;
- set the custom amount boundaries to 50 and 500 cents;
- keep `StrictInt` validation for custom cent amounts;
- continue resolving the purchased Credits from the server-side cent amount.

No database schema or ledger representation changes. Existing payment orders,
refunds, completed Checkout Sessions, webhook settlement, and displayed
activity remain valid because persisted orders store their resolved cent and
microcredit amounts rather than a package ID.

### Frontend Contract

`dashboard/frontend/app.html` will render the four small presets, mark `$1` as
selected, and describe the custom range as `$0.50-$5.00`. The custom input will
expose matching `min`, `max`, and `step` attributes as browser affordances.

`dashboard/frontend/js/credits.js` will default to `usd_1`, validate custom
amounts from 50 through 500 cents, and show a matching display-safe validation
message. Its cache-buster will be incremented so deployed browsers load the new
contract.

### Error Handling

Invalid frontend input produces an inline status message and does not call the
checkout endpoint. Invalid or retired package IDs and out-of-range direct API
requests receive the existing Pydantic `422` response. Stripe, webhook, retry,
and idempotency error handling are unchanged.

### Accessibility

The package buttons remain a keyboard-accessible radio group with exactly one
selected package. The custom input remains labelled and connected to visible
range guidance with `aria-describedby`. The change does not introduce a new
interaction pattern.

## Testing

Tests will verify:

- the exact preset allowlist and server-side cent resolution;
- acceptance of the 50-cent and 500-cent custom boundaries;
- rejection of 49 cents, 501 cents, non-integers, and retired large packages;
- frontend preset markup, default selection, input attributes, range copy, and
  client validation;
- checkout idempotency and the existing one-dollar-to-one-Credit conversion;
- the Credits script cache-buster.

Run the focused Credits model, API, frontend, integration, and cache-buster
tests, followed by the established Credits/Auth/store-parity regression suite.

## Out of Scope

- Changing the `$1 = 1 Credit` conversion rate.
- Creating free or zero-dollar Checkout Sessions.
- Modifying welcome Credits, admin Grant Pool behavior, refunds, or historical
  orders.
- Changing Stripe test/live mode selection or deployment configuration.
