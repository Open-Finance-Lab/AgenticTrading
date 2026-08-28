# Small Credit Top-Ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Limit every new ATL Credit purchase to $0.50-$5.00 while offering $0.50, $1, $2, and $5 presets.

**Architecture:** Keep `CheckoutRequest` as the server-authoritative package allowlist and custom-amount boundary. Mirror that contract in the no-build Credits frontend, then update API/integration fixtures so all purchase, webhook, refund, and idempotency paths exercise valid small packages without changing persisted order or ledger schemas.

**Tech Stack:** Python 3.13, FastAPI, Pydantic v2, pytest, vanilla HTML/JavaScript, Stripe test-mode gateway.

**Spec:** `docs/superpowers/specs/2026-08-28-small-credit-top-ups-design.md`

## Global Constraints

- New Checkout Sessions must charge from 50 through 500 US cents, inclusive.
- Fixed packages must be exactly `usd_0_50`, `usd_1`, `usd_2`, and `usd_5`.
- The one-dollar-to-one-ATL-Credit conversion remains unchanged.
- The backend remains authoritative; direct API requests cannot bypass the range.
- Existing orders, refunds, ledger entries, welcome Credits, and Grant Pool behavior remain unchanged.
- Zero-dollar Checkout Sessions are not supported.
- User-visible copy uses `$0.50-$5.00`; code, filenames, commits, and PR text remain English.
- Do not add dependencies, database migrations, secrets, or deployment configuration.

## File Structure

- `dashboard/backend/domain/credits/models.py`: owns fixed package IDs, cent values, custom amount limits, and typed checkout validation.
- `dashboard/backend/tests/domain/credits/test_service.py`: proves package resolution, conversion, and custom amount boundaries at the domain boundary.
- `dashboard/backend/tests/test_credits_api.py`: proves API requests cannot submit retired packages or out-of-range custom amounts and keeps webhook expectations aligned with a valid package.
- `dashboard/backend/tests/test_credits_api_review_fixes.py`: keeps error-mapping and account-reinstatement requests on a valid package.
- `dashboard/backend/tests/integration/test_credits_checkout_flow.py`: proves purchase, signed webhook, refund, replay, and isolation behavior with the new maximum package.
- `dashboard/frontend/app.html`: renders presets, range guidance, numeric input attributes, default selection, and the Credits script cache-buster.
- `dashboard/frontend/js/credits.js`: owns the initial package selection and client-side custom amount guard.
- `dashboard/backend/tests/test_credits_frontend.py`: statically verifies the no-build UI and client validation contract.
- `dashboard/backend/tests/test_frontend_fast_boot.py`: remains the single exact cache-buster owner.

---

### Task 1: Enforce the Small Purchase Contract Across Backend Consumers

**Files:**
- Modify: `dashboard/backend/tests/domain/credits/test_service.py:181-225`
- Modify: `dashboard/backend/domain/credits/models.py:18-73`
- Modify: `dashboard/backend/tests/test_credits_api.py:113-138,175-365`
- Modify: `dashboard/backend/tests/test_credits_api_review_fixes.py:50-65,190-218`
- Modify: `dashboard/backend/tests/integration/test_credits_checkout_flow.py:171-376`

**Interfaces:**
- Consumes: `CheckoutRequest(client_request_id: UUID, package_id: CreditPackageId | None, custom_amount_usd_cents: StrictInt | None)`.
- Produces: `CreditPackageId = Literal["usd_0_50", "usd_1", "usd_2", "usd_5"]`, `FIXED_PACKAGES_USD_CENTS`, `MIN_CUSTOM_USD_CENTS = 50`, `MAX_CUSTOM_USD_CENTS = 500`, and API/integration fixtures that use `usd_5` with a 500-cent charge and 5,000,000 purchased microcredits.

- [ ] **Step 1: Replace the domain expectations with the new allowlist and boundaries**

```python
@pytest.mark.parametrize(
    ("package_id", "cents"),
    [("usd_0_50", 50), ("usd_1", 100), ("usd_2", 200), ("usd_5", 500)],
)
def test_fixed_packages_are_resolved_server_side(package_id, cents):
    request = CheckoutRequest(
        client_request_id=CLIENT_REQUEST_ID,
        package_id=package_id,
    )
    assert request.amount_usd_cents == cents
    assert credits_micro_for_cents(cents) == cents * 10_000


@pytest.mark.parametrize("cents", [50, 51, 500])
def test_custom_amount_accepts_integer_cent_boundaries(cents):
    request = CheckoutRequest(
        client_request_id=CLIENT_REQUEST_ID,
        custom_amount_usd_cents=cents,
    )
    assert request.amount_usd_cents == cents


@pytest.mark.parametrize("cents", [49, 501, 50.0, True])
def test_custom_amount_rejects_out_of_range_or_non_integer_values(cents):
    with pytest.raises(ValidationError):
        CheckoutRequest(
            client_request_id=CLIENT_REQUEST_ID,
            custom_amount_usd_cents=cents,
        )


@pytest.mark.parametrize("package_id", ["usd_10", "usd_20", "usd_50"])
def test_retired_large_packages_are_rejected(package_id):
    with pytest.raises(ValidationError):
        CheckoutRequest(
            client_request_id=CLIENT_REQUEST_ID,
            package_id=package_id,
        )
```

- [ ] **Step 2: Run the domain tests to verify the new contract fails**

Run:

```bash
pytest -q dashboard/backend/tests/domain/credits/test_service.py \
  -k 'fixed_packages or custom_amount or retired_large_packages'
```

Expected: failures show `usd_0_50`, `usd_1`, and `usd_2` are not accepted, while 50-cent custom amounts are below the old minimum.

- [ ] **Step 3: Replace the backend allowlist and limits**

```python
CreditPackageId = Literal["usd_0_50", "usd_1", "usd_2", "usd_5"]

FIXED_PACKAGES_USD_CENTS: dict[str, int] = {
    "usd_0_50": 50,
    "usd_1": 100,
    "usd_2": 200,
    "usd_5": 500,
}
MIN_CUSTOM_USD_CENTS = 50
MAX_CUSTOM_USD_CENTS = 500
```

Change the validation error to:

```python
raise ValueError("custom amount must be from 50 through 500 cents")
```

- [ ] **Step 4: Run the complete domain Credits service test**

Run: `pytest -q dashboard/backend/tests/domain/credits/test_service.py`

Expected: PASS. Package resolution produces 500,000 microcredits for `$0.50` and 5,000,000 microcredits for `$5.00`.

- [ ] **Step 5: Run API and integration tests against the new allowlist**

Run:

```bash
pytest -q \
  dashboard/backend/tests/test_credits_api.py \
  dashboard/backend/tests/test_credits_api_review_fixes.py \
  dashboard/backend/tests/integration/test_credits_checkout_flow.py
```

Expected: FAIL because helpers still submit the retired `usd_10` package and receive `422`.

- [ ] **Step 6: Add API coverage for both custom boundaries and retired packages**

Extend `test_checkout_input_is_server_allowlisted_and_idempotent` with direct requests using unique UUIDs:

```python
too_small = billing_api.client.post(
    "/api/credits/checkout-sessions",
    headers=_auth(token),
    json={
        "client_request_id": "44444444-4444-4444-8444-444444444444",
        "custom_amount_usd_cents": 49,
    },
)
too_large = billing_api.client.post(
    "/api/credits/checkout-sessions",
    headers=_auth(token),
    json={
        "client_request_id": "55555555-5555-4555-8555-555555555555",
        "custom_amount_usd_cents": 501,
    },
)
retired = billing_api.client.post(
    "/api/credits/checkout-sessions",
    headers=_auth(token),
    json={
        "client_request_id": "66666666-6666-4666-8666-666666666666",
        "package_id": "usd_10",
    },
)
assert too_small.status_code == too_large.status_code == retired.status_code == 422
```

- [ ] **Step 7: Migrate API helpers and expectations to the `$5` package**

In `test_credits_api.py`, make `_checkout` submit `usd_5`. Set `_paid_checkout_event`'s default `amount` to `500`, set `atl_credits_micro` to `"5000000"`, and update purchase-connected assertions from 10,000,000 to 5,000,000 and refundable cents from 1,000 to 500. Keep explicit refund amounts and unrelated ledger setup unchanged.

```python
json={"client_request_id": request_id, "package_id": "usd_5"}
def _paid_checkout_event(checkout: dict, *, amount=500, event_id="evt_paid"):
"atl_credits_micro": "5000000"
```

In `test_credits_api_review_fixes.py`, replace valid `usd_10` requests with `usd_5`; these tests assert error mapping and authorization rather than a price.

- [ ] **Step 8: Migrate the end-to-end checkout fixture to `$5`**

In `test_credits_checkout_flow.py`, use:

```python
json={"client_request_id": request_id, "package_id": "usd_5"}
def _checkout_object(checkout, user_id, *, amount=500, payment_status="paid"):
"atl_credits_micro": "5000000"
```

Update only assertions derived from that purchase:

```python
assert flow.credits.get_balance_micro(buyer["id"]) == 5_000_000
assert settled.json()["result"]["balance_micro"] == 1_000_000
assert [entry["amount_micro"] for entry in ledger] == [-4_000_000, 5_000_000]
```

The existing 400-cent refund remains valid against a 500-cent purchase, while the 700-cent over-refund remains rejected.

- [ ] **Step 9: Run the API and integration tests**

Run:

```bash
pytest -q \
  dashboard/backend/tests/test_credits_api.py \
  dashboard/backend/tests/test_credits_api_review_fixes.py \
  dashboard/backend/tests/integration/test_credits_checkout_flow.py
```

Expected: PASS, including signed webhook settlement, duplicate delivery, refund, restriction, and checkout idempotency cases.

- [ ] **Step 10: Commit the complete backend contract and its consumers**

```bash
git add dashboard/backend/domain/credits/models.py \
  dashboard/backend/tests/domain/credits/test_service.py \
  dashboard/backend/tests/test_credits_api.py \
  dashboard/backend/tests/test_credits_api_review_fixes.py \
  dashboard/backend/tests/integration/test_credits_checkout_flow.py
git commit -m "feat: limit credit purchase amounts"
```

---

### Task 2: Replace the Credits Page with Small Presets

**Files:**
- Modify: `dashboard/backend/tests/test_credits_frontend.py:19-29`
- Modify: `dashboard/backend/tests/test_frontend_fast_boot.py:185-198`
- Modify: `dashboard/frontend/app.html:1895-1910,2405`
- Modify: `dashboard/frontend/js/credits.js:14-20,520-529`

**Interfaces:**
- Consumes: Task 1 package IDs and 50-500-cent custom range.
- Produces: an accessible four-option radio group, `usd_1` default selection, numeric custom input, matching client validation, and `credits.js?v=6`.

- [ ] **Step 1: Strengthen the static frontend contract**

Replace the purchase-control assertions with exact small-package and range checks:

```python
def test_credits_page_ships_small_purchase_controls():
    expected_packages = {
        'data-credit-package="usd_0_50" data-credit-cents="50"',
        'data-credit-package="usd_1" data-credit-cents="100"',
        'data-credit-package="usd_2" data-credit-cents="200"',
        'data-credit-package="usd_5" data-credit-cents="500"',
    }
    assert all(package in APP_HTML for package in expected_packages)
    assert 'data-credit-package="usd_1"' in APP_HTML
    assert 'data-credit-package="usd_10"' not in APP_HTML
    assert 'data-credit-package="usd_20"' not in APP_HTML
    assert 'data-credit-package="usd_50"' not in APP_HTML
    package_grid_start = APP_HTML.index('<div id="creditsPackageGrid"')
    package_grid_end = APP_HTML.index('</div>', package_grid_start)
    package_grid = APP_HTML[package_grid_start:package_grid_end]
    assert package_grid.count('aria-checked="true"') == 1
    assert 'aria-checked="true" data-credit-package="usd_1"' in package_grid
    assert 'id="creditsCustomAmount" type="number"' in APP_HTML
    assert 'min="0.50" max="5.00" step="0.01"' in APP_HTML
    assert 'Minimum $0.50, maximum $5.00.' in APP_HTML
```

Add client assertions:

```python
def test_credits_client_enforces_the_small_custom_range():
    source = CREDITS_JS_PATH.read_text(encoding="utf-8")
    assert "value: 'usd_1'" in source
    assert "cents < 50 || cents > 500" in source
    assert "Enter a custom amount from $0.50 through $5.00." in source
```

Change the cache-buster expectation in `test_frontend_fast_boot.py` to `js/credits.js?v=6`.

- [ ] **Step 2: Run frontend tests to verify the old controls fail**

Run:

```bash
pytest -q \
  dashboard/backend/tests/test_credits_frontend.py \
  dashboard/backend/tests/test_frontend_fast_boot.py::test_cache_busters_bumped
```

Expected: FAIL because the page still contains large presets, the old `$5-$200` range, and `credits.js?v=5`.

- [ ] **Step 3: Render the four small presets and numeric custom input**

Use this package markup in `app.html`, retaining the existing radiogroup and button classes:

```html
<button class="credits-package-btn" type="button" role="radio" aria-checked="false" data-credit-package="usd_0_50" data-credit-cents="50"><strong>$0.50</strong><span>0.5 Credits</span></button>
<button class="credits-package-btn is-selected" type="button" role="radio" aria-checked="true" data-credit-package="usd_1" data-credit-cents="100"><strong>$1</strong><span>1 Credit</span></button>
<button class="credits-package-btn" type="button" role="radio" aria-checked="false" data-credit-package="usd_2" data-credit-cents="200"><strong>$2</strong><span>2 Credits</span></button>
<button class="credits-package-btn" type="button" role="radio" aria-checked="false" data-credit-package="usd_5" data-credit-cents="500"><strong>$5</strong><span>5 Credits</span></button>
```

Use this labelled custom input and visible hint:

```html
<input id="creditsCustomAmount" type="number" inputmode="decimal" autocomplete="off" min="0.50" max="5.00" step="0.01" placeholder="0.50-5.00" aria-describedby="creditsCustomHint">
<p id="creditsCustomHint">Minimum $0.50, maximum $5.00.</p>
```

- [ ] **Step 4: Match JavaScript selection and validation to the backend**

```javascript
selection: { kind: 'package', value: 'usd_1' },
```

```javascript
const cents = parseUsdCents(element('creditsCustomAmount')?.value);
if (cents === null || cents < 50 || cents > 500) {
  throw new Error('Enter a custom amount from $0.50 through $5.00.');
}
return { custom_amount_usd_cents: cents };
```

Change the script tag to `js/credits.js?v=6`.

- [ ] **Step 5: Run frontend contracts**

Run:

```bash
pytest -q \
  dashboard/backend/tests/test_credits_frontend.py \
  dashboard/backend/tests/test_frontend_fast_boot.py::test_cache_busters_bumped
```

Expected: PASS. The radio group has one selected preset, the input remains labelled through `for="creditsCustomAmount"`, and the hint remains connected through `aria-describedby="creditsCustomHint"`.

- [ ] **Step 6: Commit the Credits UI**

```bash
git add dashboard/frontend/app.html dashboard/frontend/js/credits.js \
  dashboard/backend/tests/test_credits_frontend.py \
  dashboard/backend/tests/test_frontend_fast_boot.py
git commit -m "feat: offer small credit top-ups"
```

---

### Task 3: Run Regression Checks and Update PR #410

**Files:**
- Verify only; no additional source file is expected.
- Update: GitHub PR #410 description and branch.

**Interfaces:**
- Consumes: all Task 1-2 commits.
- Produces: a clean branch with test evidence and an updated remote PR.

- [ ] **Step 1: Confirm retired packages are absent from production code**

Run:

```bash
rg -n 'usd_10|usd_20|usd_50|200\.00|20_000|cents > 20000' \
  dashboard/backend/domain/credits dashboard/frontend
```

Expected: no matches. Tests and design documents may mention retired values only to assert rejection or describe the old state.

- [ ] **Step 2: Run the focused checkout suite**

Run:

```bash
pytest -q \
  dashboard/backend/tests/domain/credits/test_service.py \
  dashboard/backend/tests/test_credits_api.py \
  dashboard/backend/tests/test_credits_api_review_fixes.py \
  dashboard/backend/tests/integration/test_credits_checkout_flow.py \
  dashboard/backend/tests/test_credits_frontend.py \
  dashboard/backend/tests/test_frontend_fast_boot.py::test_cache_busters_bumped
```

Expected: PASS with no checkout, webhook, refund, frontend, or cache-buster failures.

- [ ] **Step 3: Run the broader Credits/Auth/store-parity regression suite**

Run:

```bash
pytest -q \
  dashboard/backend/tests/domain/credits \
  dashboard/backend/tests/test_auth.py \
  dashboard/backend/tests/test_credits_frontend.py \
  dashboard/backend/tests/test_frontend_fast_boot.py \
  dashboard/backend/tests/test_store_twin_parity.py
```

Expected: PASS, apart from environment-gated Postgres tests that are explicitly skipped when their test database is unavailable.

- [ ] **Step 4: Check formatting, scope, and secrets**

Run:

```bash
git diff main...HEAD --check
git status --short --branch
git diff main...HEAD --name-only
```

Expected: no whitespace errors and changes limited to the approved Credits implementation, its tests, spec, and plan. Leave every unrelated tracked or untracked path untouched and unstaged.

- [ ] **Step 5: Push the branch and update the PR description**

```bash
git push origin feature/default-user-credits
gh pr edit 410 --repo Open-Finance-Lab/AgenticTrading --body '## Summary

- Grant every existing and new account 1.500000 welcome Credits exactly once.
- Limit new Stripe top-ups to $0.50-$5.00 on both the backend and frontend.
- Offer $0.50, $1, $2, and $5 presets with $1 selected by default.
- Keep welcome promotions separate from the admin Grant Pool.

## Verification

- Focused checkout, webhook, refund, frontend, and cache-buster suite: passed.
- Credits, auth, frontend, and store-parity regression suite: passed.
- `git diff main...HEAD --check`: passed.'
gh pr checks 410 --repo Open-Finance-Lab/AgenticTrading --watch --interval 10
```

Expected: PR #410 contains the welcome-credit feature plus the approved small top-up contract. Report any unrelated baseline CI failure separately rather than changing out-of-scope marketplace code.
