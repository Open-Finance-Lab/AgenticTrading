# Backtest Credit Activity Precision and Aggregation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Display every ATL Credit amount with exactly six decimal places and project all settled model calls for one backtest as one exact run-level Activity debit.

**Architecture:** Keep the append-only per-call micro-Credit ledger unchanged. Aggregate usage by `user_id` and `run_id` in each persistence backend before pagination, fetch safe evidence for only the selected page in one additional query, normalize provider/model context in shared Python, and expose one public `backtest_usage` item. A small no-build frontend formatter renders decimal strings and integer micro-Credits without binary floating point conversion across Credits, Admin Grant Credits, and Admin Analytics.

**Tech Stack:** Python 3.12, FastAPI, SQLite, PostgreSQL/psycopg, pytest, vanilla JavaScript, Node.js behavioral frontend tests

**Spec:** `docs/superpowers/specs/2026-08-28-backtest-credit-activity-design.md`

## Global Constraints

- One Credit equals exactly `1,000,000` integer micro-Credits.
- Every displayed ATL Credit amount has exactly six fractional digits; USD remains at two digits.
- Activity aggregates settled usage by both `user_id` and `run_id`; equal run ids from different users never combine.
- Aggregation occurs before ordering, cursor filtering, and page limiting.
- Charged failed runs remain visible; released or failed calls without a usage row remain absent.
- The existing per-call reservation and usage tables, Grant-first allocation, Stripe behavior, BYOK behavior, and pricing calculations do not change.
- Raw evidence, provider response bodies, credentials, API keys, local databases, `.superpowers/`, and `work/` never enter public responses or commits.
- Preserve SQLite/PostgreSQL behavior parity and existing legacy decimal cursor support.
- Use synthetic run ids, amounts, and pricing evidence in all tests.
- The unrelated `test_frontend_model_facets.py::test_closed_card_differs_from_open_card_by_exactly_the_badge` baseline failure is outside this branch; do not modify marketplace-card code to hide it.

---

### Task 1: Shared Evidence Normalization and SQLite Run Projection

**Files:**
- Modify: `dashboard/backend/domain/credits/repository_common.py:9,162-195`
- Modify: `dashboard/backend/domain/credits/repository.py:1613-1712`
- Test: `dashboard/backend/tests/domain/credits/test_repository.py:415-492`

**Interfaces:**
- Produces: `summarize_activity_evidence(values: Iterable[object]) -> dict[str, object]`
- Produces: `normalize_activity_item(value: Mapping[str, object], *, evidence_json_values: Iterable[object] = ()) -> dict[str, object]`
- Produces: `CreditsStore.list_ledger_entries(...)` items with `entry_type="backtest_usage"`, exact summed `amount_micro`, `model_call_count`, safe provider/model fields, and no call-level evidence.
- Preserves: `source_kind="llm_usage"` and the existing opaque cursor encoder/decoder.

- [ ] **Step 1: Add failing SQLite run-level Activity fixtures and assertions**

Add this helper near the existing `_pay_order` helper in `test_repository.py`:

```python
def _settle_activity_call(
    store: CreditsStore,
    *,
    user_id: int = 1,
    run_id: str,
    call_index: int,
    actual_micro: int,
    provider_id: str = "openrouter",
    model_id: str = "anthropic/claude-haiku-4-5",
    evidence: object | None = None,
):
    reservation_id = f"activity:{user_id}:{run_id}:{call_index}"
    reservation = store.reserve_llm_credits(
        reservation_id=reservation_id,
        user_id=user_id,
        run_id=run_id,
        call_index=call_index,
        reserved_micro=actual_micro,
        operation_key=f"reserve:{reservation_id}",
        request_digest=f"{user_id}{call_index}".ljust(64, "a")[:64],
    )
    safe_evidence = evidence if evidence is not None else {
        "billing_source": "platform_credits",
        "pricing_snapshot": {
            "provider_id": provider_id,
            "model_id": model_id,
        },
    }
    return store.settle_llm_credits(
        reservation["reservation_id"],
        actual_micro=actual_micro,
        evidence=safe_evidence,
    )
```

Replace the call-level Activity expectation with run-level tests covering two
calls, a split Grant/Purchased debit, evidence handling, pagination, and account
isolation:

```python
def test_activity_aggregates_settled_calls_and_buckets_by_run(tmp_path):
    store = _store(tmp_path)
    store.fund_grant_pool(
        pool_id="default",
        amount_micro=500,
        operation_id="activity-fund",
        idempotency_key="activity-fund-key",
        request_digest="activity-fund-digest",
        actor_user_id=2,
        source="test",
        reason="Fund run aggregation test.",
    )
    store.assign_grant(
        pool_id="default",
        user_id=1,
        amount_micro=500,
        operation_id="activity-assign",
        idempotency_key="activity-assign-key",
        request_digest="activity-assign-digest",
        actor_user_id=2,
        source="test",
        reason="Assign run aggregation test Credits.",
    )
    _pending_order(store)
    _pay_order(store)
    _settle_activity_call(
        store, run_id="run-aggregate", call_index=0, actual_micro=600
    )
    _settle_activity_call(
        store, run_id="run-aggregate", call_index=1, actual_micro=684
    )

    items = store.list_ledger_entries(1, limit=50)["items"]
    usage = [item for item in items if item["entry_type"] == "backtest_usage"]

    assert len(usage) == 1
    assert usage[0]["amount_micro"] == -1_284
    assert usage[0]["model_call_count"] == 2
    assert usage[0]["run_id"] == "run-aggregate"
    assert usage[0]["provider_id"] == "openrouter"
    assert usage[0]["model_id"] == "anthropic/claude-haiku-4-5"
    assert usage[0]["provider_mixed"] is False
    assert usage[0]["model_mixed"] is False
    assert usage[0]["billing_source"] == "platform_credits"
    assert "reservation_id" not in usage[0]
    assert "call_index" not in usage[0]
    assert "evidence_json" not in usage[0]
    with store._get_connection() as conn:
        row_count = conn.execute(
            "SELECT COUNT(*) FROM credit_llm_usage_entries WHERE run_id = ?",
            ("run-aggregate",),
        ).fetchone()[0]
    assert row_count == 3


def test_activity_summarizes_mixed_and_malformed_evidence(tmp_path):
    store = _store(tmp_path)
    _pending_order(store)
    _pay_order(store)
    _settle_activity_call(
        store,
        run_id="run-mixed",
        call_index=0,
        actual_micro=100,
        provider_id="openrouter",
        model_id="model-a",
    )
    _settle_activity_call(
        store,
        run_id="run-mixed",
        call_index=1,
        actual_micro=200,
        provider_id="openai",
        model_id="model-b",
    )
    _settle_activity_call(
        store,
        run_id="run-unknown",
        call_index=0,
        actual_micro=300,
    )
    with store._get_connection() as conn:
        conn.execute(
            "UPDATE credit_llm_usage_entries SET evidence_json = ? WHERE run_id = ?",
            ("not-json", "run-unknown"),
        )

    items = store.list_ledger_entries(1, limit=50)["items"]
    mixed = next(item for item in items if item.get("run_id") == "run-mixed")
    unknown = next(item for item in items if item.get("run_id") == "run-unknown")

    assert mixed["provider_id"] is None
    assert mixed["model_id"] is None
    assert mixed["provider_mixed"] is True
    assert mixed["model_mixed"] is True
    assert unknown["amount_micro"] == -300
    assert unknown["provider_id"] is None
    assert unknown["model_id"] is None
    assert unknown["provider_mixed"] is False
    assert unknown["model_mixed"] is False


def test_activity_paginates_whole_runs_and_isolates_equal_run_ids(tmp_path):
    store = _store(tmp_path)
    _pending_order(store)
    _pay_order(store)
    _pending_order(
        store,
        order_id="ord_other",
        user_id=3,
        client_request_id="33333333-3333-4333-8333-333333333333",
    )
    _pay_order(store, order_id="ord_other", event_id="evt_paid_other")
    for call_index, amount in enumerate((101, 102, 103)):
        _settle_activity_call(
            store,
            user_id=1,
            run_id="run-shared",
            call_index=call_index,
            actual_micro=amount,
        )
    _settle_activity_call(
        store,
        user_id=3,
        run_id="run-shared",
        call_index=0,
        actual_micro=900,
    )

    pages = []
    cursor = None
    while True:
        page = store.list_ledger_entries(1, limit=1, cursor=cursor)
        pages.extend(page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            break

    user_one = [item for item in pages if item.get("run_id") == "run-shared"]
    user_three = store.list_ledger_entries(3, limit=50)["items"]
    user_three_usage = next(
        item for item in user_three if item.get("run_id") == "run-shared"
    )
    assert len(user_one) == 1
    assert user_one[0]["amount_micro"] == -306
    assert user_one[0]["model_call_count"] == 3
    assert user_three_usage["amount_micro"] == -900
    assert user_three_usage["model_call_count"] == 1


def test_activity_omits_released_run_without_settled_usage(tmp_path):
    store = _store(tmp_path)
    _pending_order(store)
    _pay_order(store)
    reservation = store.reserve_llm_credits(
        reservation_id="activity-released",
        user_id=1,
        run_id="run-released-without-charge",
        call_index=0,
        reserved_micro=1_000,
        operation_key="activity-released-reserve",
        request_digest="r" * 64,
    )
    store.release_llm_credits(
        reservation["reservation_id"],
        reason="synthetic provider failure",
    )

    items = store.list_ledger_entries(1, limit=50)["items"]
    assert all(
        item.get("run_id") != "run-released-without-charge" for item in items
    )
```

- [ ] **Step 2: Run the SQLite tests and verify the existing call-level query fails them**

Run:

```bash
pytest dashboard/backend/tests/domain/credits/test_repository.py -q -k "activity"
```

Expected: FAIL because multiple calls still produce multiple `llm_usage` rows,
`model_call_count` and mixed flags are absent, and malformed evidence is tied to
one call-level row.

- [ ] **Step 3: Add shared tolerant evidence summarization**

Import `Iterable` beside `Mapping` in `repository_common.py`, then add this
helper and extend `normalize_activity_item`:

```python
def summarize_activity_evidence(values: Iterable[object]) -> dict[str, object]:
    providers: set[str] = set()
    models: set[str] = set()
    billing_sources: set[str] = set()
    provider_unknown = False
    model_unknown = False
    billing_unknown = False
    for raw in values:
        try:
            evidence = json.loads(raw) if isinstance(raw, str) else {}
        except json.JSONDecodeError:
            evidence = {}
        if not isinstance(evidence, dict):
            evidence = {}
        snapshot = evidence.get("pricing_snapshot")
        if not isinstance(snapshot, dict):
            snapshot = {}
        provider = snapshot.get("provider_id")
        model = snapshot.get("model_id")
        billing = evidence.get("billing_source")
        if isinstance(provider, str) and provider.strip():
            providers.add(provider)
        else:
            provider_unknown = True
        if isinstance(model, str) and model.strip():
            models.add(model)
        else:
            model_unknown = True
        if isinstance(billing, str) and billing.strip():
            billing_sources.add(billing)
        else:
            billing_unknown = True

    return {
        "provider_id": (
            next(iter(providers))
            if len(providers) == 1 and not provider_unknown
            else None
        ),
        "model_id": (
            next(iter(models)) if len(models) == 1 and not model_unknown else None
        ),
        "billing_source": (
            next(iter(billing_sources))
            if len(billing_sources) == 1 and not billing_unknown
            else None
        ),
        "provider_mixed": len(providers) > 1,
        "model_mixed": len(models) > 1,
    }


def normalize_activity_item(
    value: Mapping[str, object],
    *,
    evidence_json_values: Iterable[object] = (),
) -> dict[str, object]:
    item = dict(value)
    item.pop("evidence_json", None)
    item["id"] = int(item.pop("source_id"))
    item["amount_micro"] = int(item["amount_micro"])
    if item.get("source_kind") != "llm_usage":
        item.pop("model_call_count", None)
        return item
    item.update(
        {
            "entry_type": "backtest_usage",
            "source": "llm_execution",
            "reason": "Backtest usage.",
            "model_call_count": int(item["model_call_count"]),
            **summarize_activity_evidence(evidence_json_values),
        }
    )
    item.pop("reservation_id", None)
    item.pop("call_index", None)
    return item
```

- [ ] **Step 4: Aggregate SQLite usage before cursor filtering and page limiting**

In `CreditsStore.list_ledger_entries`, add `NULL AS model_call_count` to the
historical projection and replace the usage CTE with:

```sql
llm_activity AS (
    SELECT MAX(id) AS source_id, 'llm_usage' AS source_kind,
           user_id, NULL AS bucket,
           'backtest_usage' AS entry_type,
           SUM(amount_micro) AS amount_micro,
           NULL AS payment_order_id,
           NULL AS refund_request_id, NULL AS stripe_event_id,
           NULL AS operation_key, NULL AS operation_id,
           NULL AS idempotency_key, NULL AS request_digest,
           NULL AS actor_user_id, 'llm_execution' AS source,
           'Backtest usage.' AS reason,
           NULL AS reference_type, NULL AS reference_id,
           MAX(created_at) AS created_at,
           NULL AS reservation_id, run_id, NULL AS call_index,
           COUNT(DISTINCT call_index) AS model_call_count,
           NULL AS evidence_json
    FROM credit_llm_usage_entries
    WHERE user_id = ?
    GROUP BY user_id, run_id
)
```

Before the connection closes, load evidence for only the selected run summaries
with one bounded query:

```python
selected_rows = rows[:page_size]
run_ids = list(
    dict.fromkeys(
        str(row["run_id"])
        for row in selected_rows
        if row["source_kind"] == "llm_usage" and row["run_id"] is not None
    )
)
evidence_by_run: dict[str, list[object]] = {run_id: [] for run_id in run_ids}
if run_ids:
    placeholders = ", ".join("?" for _ in run_ids)
    evidence_rows = conn.execute(
        f"""
        SELECT DISTINCT run_id, evidence_json
        FROM credit_llm_usage_entries
        WHERE user_id = ? AND run_id IN ({placeholders})
        """,
        [user_id, *run_ids],
    ).fetchall()
    for evidence_row in evidence_rows:
        evidence_by_run[str(evidence_row["run_id"])].append(
            evidence_row["evidence_json"]
        )
```

Normalize the selected rows with their page-local evidence sets:

```python
items = [
    normalize_activity_item(
        dict(row),
        evidence_json_values=evidence_by_run.get(str(row["run_id"]), ()),
    )
    for row in selected_rows
]
```

- [ ] **Step 5: Run SQLite repository tests**

Run:

```bash
pytest dashboard/backend/tests/domain/credits/test_repository.py -q
```

Expected: PASS. The original purchase/refund and legacy-cursor tests remain
green, and each run appears exactly once.

- [ ] **Step 6: Commit the SQLite projection**

```bash
git add dashboard/backend/domain/credits/repository_common.py dashboard/backend/domain/credits/repository.py dashboard/backend/tests/domain/credits/test_repository.py
git commit -m "fix(credits): aggregate SQLite activity by backtest"
```

---

### Task 2: PostgreSQL Projection Parity

**Files:**
- Modify: `dashboard/backend/domain/credits/repository_postgres.py:1410-1513`
- Test: `dashboard/backend/tests/domain/credits/test_repository_postgres.py:467-492`

**Interfaces:**
- Consumes: `normalize_activity_item(..., evidence_json_values=...)` from Task 1.
- Produces: `PostgresCreditsStore.list_ledger_entries(...)` with the same run-level item shape, ordering, and cursor behavior as SQLite.

- [ ] **Step 1: Add a failing live-PostgreSQL run aggregation test**

Add this test beside the current Postgres projection/pagination test:

```python
@pg_only
def test_postgres_activity_aggregates_calls_before_pagination(pg_credits_store):
    store = pg_credits_store
    _pending_order(store)
    _pay_order(store)
    for call_index, amount in enumerate((137, 1_147)):
        reservation_id = f"pg-activity:{call_index}"
        reservation = store.reserve_llm_credits(
            reservation_id=reservation_id,
            user_id=1,
            run_id="run-pg-activity",
            call_index=call_index,
            reserved_micro=amount,
            operation_key=f"reserve:{reservation_id}",
            request_digest=str(call_index).ljust(64, "b"),
        )
        store.settle_llm_credits(
            reservation["reservation_id"],
            actual_micro=amount,
            evidence={
                "billing_source": "platform_credits",
                "pricing_snapshot": {
                    "provider_id": "openrouter",
                    "model_id": "anthropic/claude-haiku-4-5",
                },
            },
        )

    first = store.list_ledger_entries(1, limit=1)
    second = store.list_ledger_entries(1, limit=1, cursor=first["next_cursor"])
    usage = next(
        item
        for item in [*first["items"], *second["items"]]
        if item["entry_type"] == "backtest_usage"
    )

    assert usage["amount_micro"] == -1_284
    assert usage["model_call_count"] == 2
    assert usage["provider_id"] == "openrouter"
    assert usage["model_id"] == "anthropic/claude-haiku-4-5"
    assert first["items"][0]["id"] != second["items"][0]["id"]
```

- [ ] **Step 2: Run the Postgres test and verify it fails when Postgres is configured**

Run:

```bash
pytest dashboard/backend/tests/domain/credits/test_repository_postgres.py -q -k "activity_aggregates"
```

Expected with `TEST_POSTGRES_URL` configured: FAIL because each call is still a
separate item. Expected without it: one documented SKIP; CI performs the live
contract run.

- [ ] **Step 3: Mirror the SQLite aggregation in PostgreSQL**

Add `NULL::BIGINT AS model_call_count` to the historical projection. Replace
the Postgres usage CTE with the typed equivalent:

```sql
llm_activity AS (
    SELECT MAX(id) AS source_id, 'llm_usage' AS source_kind,
           user_id, NULL::TEXT AS bucket,
           'backtest_usage' AS entry_type,
           SUM(amount_micro) AS amount_micro,
           NULL::TEXT AS payment_order_id,
           NULL::TEXT AS refund_request_id,
           NULL::TEXT AS stripe_event_id,
           NULL::TEXT AS operation_key,
           NULL::TEXT AS operation_id,
           NULL::TEXT AS idempotency_key,
           NULL::TEXT AS request_digest,
           NULL::INTEGER AS actor_user_id,
           'llm_execution' AS source, 'Backtest usage.' AS reason,
           NULL::TEXT AS reference_type,
           NULL::TEXT AS reference_id,
           MAX(created_at) AS created_at,
           NULL::TEXT AS reservation_id, run_id,
           NULL::BIGINT AS call_index,
           COUNT(DISTINCT call_index) AS model_call_count,
           NULL::TEXT AS evidence_json
    FROM credit_llm_usage_entries
    WHERE user_id = %s
    GROUP BY user_id, run_id
)
```

Fetch selected-page evidence inside the same connection and transaction scope:

```python
selected_rows = rows[:page_size]
run_ids = list(
    dict.fromkeys(
        str(row["run_id"])
        for row in selected_rows
        if row["source_kind"] == "llm_usage" and row["run_id"] is not None
    )
)
evidence_by_run: dict[str, list[object]] = {run_id: [] for run_id in run_ids}
if run_ids:
    cur.execute(
        """
        SELECT DISTINCT run_id, evidence_json
        FROM credit_llm_usage_entries
        WHERE user_id = %s AND run_id = ANY(%s)
        """,
        (user_id, run_ids),
    )
    for evidence_row in cur.fetchall():
        evidence_by_run[str(evidence_row["run_id"])].append(
            evidence_row["evidence_json"]
        )
```

Normalize the selected Postgres rows with the shared function:

```python
items = [
    normalize_activity_item(
        dict(row),
        evidence_json_values=evidence_by_run.get(str(row["run_id"]), ()),
    )
    for row in selected_rows
]
```

- [ ] **Step 4: Run Postgres and store-parity tests**

Run:

```bash
pytest dashboard/backend/tests/domain/credits/test_repository_postgres.py dashboard/backend/tests/test_store_twin_parity.py -q
```

Expected: PASS when Postgres is available; otherwise only the tests explicitly
marked `pg_only` skip and the structural twin checks pass.

- [ ] **Step 5: Commit PostgreSQL parity**

```bash
git add dashboard/backend/domain/credits/repository_postgres.py dashboard/backend/tests/domain/credits/test_repository_postgres.py
git commit -m "fix(credits): match Postgres backtest activity"
```

---

### Task 3: Public Run-Level Credits API Contract

**Files:**
- Modify: `dashboard/backend/api/routers/credits.py:82-106`
- Test: `dashboard/backend/tests/test_credits_api.py:209-257`

**Interfaces:**
- Consumes: repository items with `entry_type="backtest_usage"` from Tasks 1 and 2.
- Produces: authenticated `GET /api/credits/ledger` items containing `run_id`, `model_call_count`, safe provider/model summary fields, `billing_source`, exact `display_credits`, and no reservation/call/evidence fields.

- [ ] **Step 1: Replace the API test with a failing two-call run contract**

Update `test_credit_activity_exposes_safe_aggregated_llm_usage` so it settles
two calls for one run and asserts one public item:

```python
def test_credit_activity_exposes_safe_aggregated_backtest_usage(billing_api):
    token = _signup(billing_api.client, "usage-api@example.com")
    checkout = _checkout(billing_api.client, token).json()["checkout"]
    billing_api.gateway.event = _paid_checkout_event(checkout)
    assert _deliver_webhook(billing_api).status_code == 200
    for call_index, amount in enumerate((137, 1_147)):
        reservation_id = f"api-usage-reservation-{call_index}"
        reservation = billing_api.store.reserve_llm_credits(
            reservation_id=reservation_id,
            user_id=1,
            run_id="run-api-usage",
            call_index=call_index,
            reserved_micro=amount,
            operation_key=f"api-usage-reserve-{call_index}",
            request_digest=str(call_index).ljust(64, "f"),
        )
        billing_api.store.settle_llm_credits(
            reservation["reservation_id"],
            actual_micro=amount,
            evidence={
                "billing_source": "platform_credits",
                "pricing_snapshot": {
                    "provider_id": "openrouter",
                    "model_id": "anthropic/claude-haiku-4-5",
                },
                "api_key": "synthetic-secret-must-not-leak",
            },
        )

    response = billing_api.client.get(
        "/api/credits/ledger?limit=1",
        headers=_auth(token),
    )
    body = response.json()
    usage = body["items"][0]

    assert response.status_code == 200
    assert usage["entry_type"] == "backtest_usage"
    assert usage["amount_micro"] == -1_284
    assert usage["display_credits"] == "-0.001284"
    assert usage["run_id"] == "run-api-usage"
    assert usage["model_call_count"] == 2
    assert usage["provider_id"] == "openrouter"
    assert usage["model_id"] == "anthropic/claude-haiku-4-5"
    assert usage["provider_mixed"] is False
    assert usage["model_mixed"] is False
    assert usage["billing_source"] == "platform_credits"
    assert "reservation_id" not in usage
    assert "call_index" not in usage
    assert isinstance(body["next_cursor"], str)
    assert "evidence_json" not in str(body)
    assert "synthetic-secret-must-not-leak" not in str(body)

    second = billing_api.client.get(
        "/api/credits/ledger",
        params={"limit": 1, "cursor": body["next_cursor"]},
        headers=_auth(token),
    )
    assert second.status_code == 200
    assert second.json()["items"][0]["entry_type"] == "purchase"
```

- [ ] **Step 2: Run the API test and verify the call-level serializer fails**

Run:

```bash
pytest dashboard/backend/tests/test_credits_api.py -q -k "aggregated_backtest_usage"
```

Expected: FAIL until `_public_ledger_entry` recognizes `backtest_usage` and
allowlists the new summary fields.

- [ ] **Step 3: Update the public serializer allowlist**

Replace the usage-specific branch in `_public_ledger_entry` with:

```python
if entry.get("entry_type") == "backtest_usage":
    result.update(
        {
            "run_id": entry.get("run_id"),
            "model_call_count": entry.get("model_call_count"),
            "provider_id": entry.get("provider_id"),
            "model_id": entry.get("model_id"),
            "provider_mixed": bool(entry.get("provider_mixed")),
            "model_mixed": bool(entry.get("model_mixed")),
            "billing_source": entry.get("billing_source"),
        }
    )
```

Do not serialize `reservation_id`, `call_index`, `evidence_json`, or any raw
evidence keys.

- [ ] **Step 4: Run Credits API and service contracts**

Run:

```bash
pytest dashboard/backend/tests/test_credits_api.py dashboard/backend/tests/test_credits_api_review_fixes.py dashboard/backend/tests/domain/credits/test_service.py -q
```

Expected: PASS with exact six-decimal API strings and unchanged authentication,
purchase, refund, and error boundaries.

- [ ] **Step 5: Commit the API contract**

```bash
git add dashboard/backend/api/routers/credits.py dashboard/backend/tests/test_credits_api.py
git commit -m "fix(credits): expose backtest usage summaries"
```

---

### Task 4: Exact Six-Decimal Frontend and Run-Level Activity UI

**Files:**
- Create: `dashboard/frontend/js/credit-format.js`
- Create: `dashboard/backend/tests/test_credit_format_frontend.py`
- Modify: `dashboard/frontend/app.html:1893-1899,2401-2409`
- Modify: `dashboard/frontend/js/credits.js:45-52,443-479`
- Modify: `dashboard/frontend/js/admin-credits.js:77-81,130-153,406`
- Modify: `dashboard/frontend/js/admin-analytics.js:218-223`
- Modify: `dashboard/backend/tests/test_credits_frontend.py:20-86`
- Modify: `dashboard/backend/tests/test_admin_credits_frontend.py:107-115`
- Modify: `dashboard/backend/tests/test_admin_analytics_frontend.py:90-95,190-196`
- Modify: `dashboard/backend/tests/test_frontend_fast_boot.py:188-199`

**Interfaces:**
- Produces: `window.CreditFormat.formatCredits(value: unknown) -> string`
- Produces: `window.CreditFormat.formatCreditsMicro(value: unknown) -> string`
- Consumes: Task 3 `backtest_usage` fields and exact `display_credits`.
- Preserves: safe `textContent` rendering, no-build deferred script order, tab behavior, and existing ARIA live regions.

- [ ] **Step 1: Add behavioral tests for the exact formatter and static UI contract**

Create `test_credit_format_frontend.py`:

```python
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
FORMAT_JS = FRONTEND / "js" / "credit-format.js"
APP_HTML = (FRONTEND / "app.html").read_text(encoding="utf-8")


requires_node = pytest.mark.skipif(
    shutil.which("node") is None,
    reason="node is required for exact Credits formatter tests",
)


def _run_formatter(expressions: list[str]) -> list[str]:
    source = FORMAT_JS.read_text(encoding="utf-8")
    script = "\n".join(
        [
            "const window = globalThis;",
            source,
            f"console.log(JSON.stringify([{', '.join(expressions)}]));",
        ]
    )
    result = subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return json.loads(result.stdout)


@requires_node
def test_credit_formatter_keeps_exact_fixed_six_decimal_values():
    assert _run_formatter(
        [
            'CreditFormat.formatCredits("4.79")',
            'CreditFormat.formatCredits("-0.000137")',
            'CreditFormat.formatCredits("1234")',
            "CreditFormat.formatCreditsMicro(4790000)",
            "CreditFormat.formatCreditsMicro(-137)",
            'CreditFormat.formatCreditsMicro("9007199254740993")',
        ]
    ) == [
        "4.790000",
        "-0.000137",
        "1,234.000000",
        "4.790000",
        "-0.000137",
        "9,007,199,254.740993",
    ]


@requires_node
def test_credit_formatter_rejects_missing_float_and_overprecision_values():
    assert _run_formatter(
        [
            "CreditFormat.formatCredits(null)",
            'CreditFormat.formatCredits("1.0000001")',
            "CreditFormat.formatCreditsMicro(0.5)",
            "CreditFormat.formatCreditsMicro(Number.NaN)",
        ]
    ) == ["—", "—", "—", "—"]


def test_credit_formatter_loads_before_every_consumer():
    formatter_at = APP_HTML.index('src="js/credit-format.js?v=1"')
    for asset in (
        'src="js/credits.js?v=5"',
        'src="js/admin-credits.js?v=6"',
        'src="js/admin-analytics.js?v=2"',
    ):
        assert formatter_at < APP_HTML.index(asset)


def test_static_credit_package_amounts_use_six_decimals():
    for amount in ("5.000000", "10.000000", "20.000000", "50.000000"):
        assert f">{amount} Credits</span>" in APP_HTML
    assert "$1 = 1.000000 Credits" in APP_HTML
```

Extend the existing frontend contracts to require:

```python
def test_activity_renders_one_backtest_usage_summary_with_safe_context():
    source = CREDITS_JS_PATH.read_text(encoding="utf-8")
    assert "entry.entry_type === 'backtest_usage'" in source
    assert "'Backtest usage'" in source
    assert "entry.model_call_count" in source
    assert "'Multiple providers'" in source
    assert "'Multiple models'" in source
    assert "String(entry.run_id).slice(0, 12)" in source
    assert "entry.call_index" not in source
```

- [ ] **Step 2: Run frontend tests and verify the formatter and run title are absent**

Run:

```bash
pytest dashboard/backend/tests/test_credit_format_frontend.py dashboard/backend/tests/test_credits_frontend.py dashboard/backend/tests/test_admin_credits_frontend.py dashboard/backend/tests/test_admin_analytics_frontend.py dashboard/backend/tests/test_frontend_fast_boot.py -q
```

Expected: FAIL because the shared formatter does not exist, package values use
integer copy, and Credits Activity recognizes only `llm_usage`.

- [ ] **Step 3: Implement the shared string/integer formatter**

Create `dashboard/frontend/js/credit-format.js`:

```javascript
/** Exact ATL Credit formatting. One Credit is 1,000,000 micro-Credits. */
(function () {
  'use strict';

  const UNAVAILABLE = '—';

  function groupWhole(value) {
    return value.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  }

  function formatCredits(value) {
    const text = value == null ? '' : String(value);
    const match = /^(-?)(\d+)(?:\.(\d{1,6}))?$/.exec(text);
    if (!match) return UNAVAILABLE;
    const fraction = (match[3] || '').padEnd(6, '0');
    const isZero = /^0+$/.test(match[2]) && !/[1-9]/.test(fraction);
    const sign = match[1] && !isZero ? '-' : '';
    return `${sign}${groupWhole(match[2])}.${fraction}`;
  }

  function formatCreditsMicro(value) {
    const text = typeof value === 'number' && Number.isSafeInteger(value)
      ? String(value)
      : (typeof value === 'string' && /^-?\d+$/.test(value) ? value : '');
    if (!text) return UNAVAILABLE;
    const micro = BigInt(text);
    const sign = micro < 0n ? '-' : '';
    const absolute = micro < 0n ? -micro : micro;
    const whole = groupWhole(String(absolute / 1000000n));
    const fraction = String(absolute % 1000000n).padStart(6, '0');
    return `${sign}${whole}.${fraction}`;
  }

  window.CreditFormat = Object.freeze({ formatCredits, formatCreditsMicro });
})();
```

Load `js/credit-format.js?v=1` before `credits.js`, `admin-credits.js`, and
`admin-analytics.js`. Change the static package copy to:

```html
<span class="credits-rate">$1 = 1.000000 Credits</span>
```

and use `5.000000`, `10.000000`, `20.000000`, and `50.000000 Credits` in the
four package buttons.

- [ ] **Step 4: Use the formatter in all billing and analytics consumers**

In `credits.js`, bind the helper once and remove the local truncating
`formatCreditDisplay`:

```javascript
const { formatCredits } = window.CreditFormat;
```

Use `formatCredits(balance.display_credits)` for the balance and
`formatCredits(entry.display_credits)` for Activity.

In `admin-credits.js`, bind both functions:

```javascript
const { formatCredits, formatCreditsMicro } = window.CreditFormat;
```

Use authoritative integer fields for the Grant Pool total and ring inputs:

```javascript
const availableMicro = pool.pool_available_micro;
const allocatedMicro = pool.allocated_to_users_micro;
const validAvailable = Number.isSafeInteger(availableMicro);
const validAllocated = Number.isSafeInteger(allocatedMicro);
const totalMicro = validAvailable && validAllocated
  && Number.isSafeInteger(availableMicro + allocatedMicro)
  ? availableMicro + allocatedMicro
  : null;
const availableText = formatCreditsMicro(availableMicro);
const allocatedText = formatCreditsMicro(allocatedMicro);
const totalText = formatCreditsMicro(totalMicro);
const chartAvailable = validAvailable && availableMicro > 0 ? availableMicro : 0;
const chartAllocated = validAllocated && allocatedMicro > 0 ? allocatedMicro : 0;
const chartTotal = chartAvailable + chartAllocated;
const availableRatio = chartTotal > 0 ? chartAvailable / chartTotal : 0;
const allocatedRatio = chartTotal > 0 ? chartAllocated / chartTotal : 0;
```

Render the assign confirmation with
`formatCreditsMicro(amountMicro)` instead of echoing the raw input. Keep API
`display_credits` values formatted through `formatCredits` for user balance and
Grant Activity tables.

In `admin-analytics.js`, replace the floating point division formatter with:

```javascript
function formatCreditsMicro(value) {
  return window.CreditFormat.formatCreditsMicro(value) === '—'
    ? '—'
    : `${window.CreditFormat.formatCreditsMicro(value)} Credits`;
}
```

- [ ] **Step 5: Render one Backtest usage row with call count and mixed context**

Replace the usage branch in `credits.js` with:

```javascript
const isUsage = entry.entry_type === 'backtest_usage';
const isNegative = isUsage || entry.entry_type === 'refund';
const title = isUsage
  ? 'Backtest usage'
  : (entry.entry_type === 'refund' ? 'Refund' : 'Credit purchase');
const callCount = Number.isSafeInteger(entry.model_call_count)
  && entry.model_call_count > 0
  ? `${entry.model_call_count} model call${entry.model_call_count === 1 ? '' : 's'}`
  : null;
const usageDetail = isUsage
  ? [
      entry.provider_mixed ? 'Multiple providers' : entry.provider_id,
      entry.model_mixed ? 'Multiple models' : entry.model_id,
      callCount,
      entry.run_id ? `run ${String(entry.run_id).slice(0, 12)}` : null,
    ].filter(Boolean).join(' · ')
  : null;
```

Continue building nodes with `textContent`; do not introduce HTML interpolation
for provider, model, or run identifiers.

- [ ] **Step 6: Bump changed asset versions and update the single-owner cache contract**

Set the asset references in `app.html` to:

```html
<script src="js/credit-format.js?v=1" defer></script>
<script src="js/credits.js?v=5" defer></script>
<script src="js/admin-credits.js?v=6" defer></script>
<script src="js/admin-analytics.js?v=2" defer></script>
```

Update existing exact-version assertions that reference the changed assets and
add `credit-format.js?v=1` to
`test_frontend_fast_boot.py::test_cache_busters_bumped`.

- [ ] **Step 7: Run the behavioral and static frontend contracts**

Run:

```bash
pytest dashboard/backend/tests/test_credit_format_frontend.py dashboard/backend/tests/test_credits_frontend.py dashboard/backend/tests/test_admin_credits_frontend.py dashboard/backend/tests/test_admin_analytics_frontend.py dashboard/backend/tests/test_frontend_fast_boot.py -q
```

Expected: PASS. Node verifies exact large/small/negative formatting; static
contracts verify deferred load order, cache versions, and run-level copy.

- [ ] **Step 8: Perform safe browser visual verification**

Start the dashboard against a new temporary SQLite path:

```bash
ui_check_dir=$(mktemp -d /tmp/atl-credit-ui.XXXXXX)
DATABASE_PATH="$ui_check_dir/ui-check.db" python -m uvicorn dashboard.backend.app:app --host 127.0.0.1 --port 8770
```

Use browser request interception for `/api/auth/me`, `/api/credits/balance`,
and `/api/credits/ledger`. Return these synthetic bodies:

```json
{"user":{"id":1,"email":"ui-check@example.test","display_name":"UI Check","role":"admin"}}
```

```json
{"balance":{"balance_micro":4790000,"display_credits":"4.790000","account_status":"active","billing_available":true},"test_mode":true}
```

```json
{"items":[{"id":42,"source_kind":"llm_usage","bucket":null,"entry_type":"backtest_usage","amount_micro":-1284,"display_credits":"-0.001284","source":"llm_execution","reason":"Backtest usage.","payment_order_id":null,"created_at":"2026-08-27T15:44:00+00:00","run_id":"run_visual_check_001","model_call_count":12,"provider_id":"openrouter","model_id":"anthropic/claude-haiku-4-5","provider_mixed":false,"model_mixed":false,"billing_source":"platform_credits"}],"next_cursor":null}
```

Mock `/api/credits/model-providers` as `{"providers":[]}`,
`/api/credits/api-keys` as `{"items":[]}`, and
`/api/credits/execution-options` as `{"providers":[]}`. Then open
`http://127.0.0.1:8770/app?view=credits`, select Credits and Activity, and
verify at desktop and 390px viewport widths:

1. Balance reads `4.790000 Credits`.
2. Package cards and `$1` rate use six decimals without overflow.
3. One run with twelve calls shows one `Backtest usage` row.
4. The row reads `12 model calls` and `-0.001284`.
5. Mixed provider/model copy wraps without covering the amount.
6. Keyboard tab navigation and existing ARIA live regions still work.

Do not use a production database, real provider key, or real Stripe credential.

- [ ] **Step 9: Commit the frontend behavior**

```bash
git add dashboard/frontend/js/credit-format.js dashboard/frontend/app.html dashboard/frontend/js/credits.js dashboard/frontend/js/admin-credits.js dashboard/frontend/js/admin-analytics.js dashboard/backend/tests/test_credit_format_frontend.py dashboard/backend/tests/test_credits_frontend.py dashboard/backend/tests/test_admin_credits_frontend.py dashboard/backend/tests/test_admin_analytics_frontend.py dashboard/backend/tests/test_frontend_fast_boot.py
git commit -m "fix(ui): show exact run-level Credit activity"
```

---

### Task 5: Full Regression Verification and Pull Request

**Files:**
- Verify only: all files changed in Tasks 1-4
- Verify only: `docs/superpowers/specs/2026-08-28-backtest-credit-activity-design.md`
- Verify only: `docs/superpowers/plans/2026-08-28-backtest-credit-activity.md`

**Interfaces:**
- Consumes: all repository, API, and frontend contracts from Tasks 1-4.
- Produces: a reviewable GitHub pull request targeting `main` with no unrelated files or secrets.

- [ ] **Step 1: Run the complete focused regression set**

Run:

```bash
pytest dashboard/backend/tests/domain/credits/test_repository.py dashboard/backend/tests/domain/credits/test_repository_postgres.py dashboard/backend/tests/test_store_twin_parity.py dashboard/backend/tests/domain/credits/test_service.py dashboard/backend/tests/test_credits_api.py dashboard/backend/tests/test_credits_api_review_fixes.py dashboard/backend/tests/test_credit_format_frontend.py dashboard/backend/tests/test_credits_frontend.py dashboard/backend/tests/test_admin_credits_frontend.py dashboard/backend/tests/test_admin_analytics_frontend.py dashboard/backend/tests/test_frontend_fast_boot.py -q
```

Expected: all non-Postgres tests PASS; live Postgres tests PASS when
`TEST_POSTGRES_URL` is set and otherwise report only their declared skips.

- [ ] **Step 2: Run the full backend suite**

Run:

```bash
pytest dashboard/backend/tests/ --timeout=180 -p no:cacheprovider
```

Expected: no new failures. If the known marketplace-card baseline test remains
red, run the same test against `origin/main` and record both identical results
in the PR; do not alter unrelated marketplace code in this branch.

- [ ] **Step 3: Audit the diff and repository contents**

Run:

```bash
git diff --check origin/main...HEAD
git status --short
git diff --stat origin/main...HEAD
git log --oneline origin/main..HEAD
git diff --name-only origin/main...HEAD
```

Expected: only the approved spec, plan, Credits repositories/API/tests, exact
Credit frontend files/tests, and cache-version HTML are present. No database,
secret, `.superpowers/`, or nested `work/` path appears.

- [ ] **Step 4: Push the branch and create the pull request**

```bash
git push -u origin fix/backtest-credit-activity
gh pr create --base main --head fix/backtest-credit-activity --title "Fix backtest Credit activity precision" --body "## Summary
- display ATL Credit values with exact fixed six-decimal precision
- aggregate settled model-call debits into one Activity row per backtest run
- keep SQLite and PostgreSQL pagination, evidence safety, and account isolation aligned

## Tests
- focused Credits repository, API, frontend, and parity tests
- full dashboard backend test suite
- safe desktop and mobile browser verification"
```

Expected: GitHub returns a PR URL whose base is `main` and whose file list
matches Step 3.

- [ ] **Step 5: Inspect the remote PR checks and final diff**

Run:

```bash
gh pr view --json number,url,baseRefName,headRefName,files,statusCheckRollup
```

Expected: `baseRefName` is `main`, `headRefName` is
`fix/backtest-credit-activity`, checks are passing or in progress, and no
unrelated file is present.
