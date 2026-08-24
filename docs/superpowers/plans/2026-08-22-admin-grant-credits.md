# Admin Grant Credits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an audited administrator-funded Grant Pool, atomic Grant assignment/reclaim, separate Grant/Purchased user balances, and the approved Admin/User interfaces without enabling model spending.

**Architecture:** Extend the existing `dashboard.backend.domain.credits` accounting boundary instead of introducing a second Credits system. SQLite and PostgreSQL repositories own append-only ledgers, migrations, transactionality, and balance constraints; the Credits service owns canonical commands and safe domain errors; a dedicated Admin Credits router and frontend module keep privileged workflow code out of the existing Stripe and user-page modules.

**Tech Stack:** Python 3.13, FastAPI, Pydantic v2, SQLite, PostgreSQL/psycopg, pytest, static HTML/CSS/JavaScript, existing ATL API/CSRF helpers.

**Spec:** `docs/superpowers/specs/2026-08-22-admin-grant-credits-design.md`

## Global Constraints

- Implement PR2 only; model execution, quotes, reservations, usage, and settlement are excluded.
- One Credit equals exactly `1_000_000` micro-Credits; authoritative API and repository amounts are strict integers.
- Stripe purchase/refund entries remain in the `purchased` bucket with original IDs, amounts, references, operation keys, and timestamps.
- Grant mutations never alter Purchased balances or simulated portfolio state.
- Identical idempotent replays return the original operation; changed payloads using the same key return `409`.
- `assign` and `reclaim` write the pool and user lines in one transaction or write nothing.
- The old `user_entitlements.credits` field remains operational and is labeled `Legacy run Credits`; no migration is permitted.
- The user balance response keeps `balance_micro` and `display_credits` as total-balance aliases.
- Admin Credits UI lives under the existing top-level Admin entry, not inside Credits & Billing.
- Do not cherry-pick the mixed `design/admin-sponsored-credits` implementation commits; use them only as read-only reference.
- Never stage `dashboard/storage/data/backtest.db`, `.superpowers/`, `AGENTS.md`, or `work/`.

---

## File Structure

### Domain and persistence

- Modify `dashboard/backend/domain/credits/models.py`: strict Grant commands and typed balance/pool/mutation results.
- Modify `dashboard/backend/domain/credits/repository_common.py`: Grant-specific store errors and shared strict validators/digests.
- Modify `dashboard/backend/domain/credits/repository.py`: SQLite ledger migration, balance projections, Grant Pool tables, atomic mutations, summary/activity reads.
- Modify `dashboard/backend/domain/credits/repository_postgres.py`: PostgreSQL-equivalent migration, locking, mutations, and reads.
- Modify `dashboard/backend/domain/credits/service.py`: deterministic operation construction, safe Grant methods, additive balance projection.

### API and identity composition

- Create `dashboard/backend/api/routers/admin_credits.py`: router-level Admin authorization, mutation rate limit, pool/user/activity APIs, safe error mapping.
- Modify `dashboard/backend/api/routers/credits.py`: additive public bucket fields only; Stripe route behavior remains unchanged.
- Modify `dashboard/backend/api/router.py`: register the Admin Credits router exactly once.
- Modify `dashboard/backend/users.py` and `dashboard/backend/users_postgres.py`: optional server-side Admin user search while retaining existing pagination and identity ownership.

### Frontend

- Modify `dashboard/frontend/app.html`: Admin tabs/panels, approved Grant Credits markup, user balance breakdown, Legacy label.
- Create `dashboard/frontend/js/admin-credits.js`: Admin tab state, exact amount parsing, pool/user/activity loading, drawer mutations, refresh and cleanup.
- Modify `dashboard/frontend/js/credits.js`: render additive Grant/Purchased/total balance and bucket activity without changing Top up/API Keys.
- Modify `dashboard/frontend/app.js`: Admin tab orchestration and refresh hooks.
- Modify `dashboard/frontend/styles.css`: compact Admin workbench, stable tables/drawer, desktop/mobile constraints.

### Tests

- Create `dashboard/backend/tests/domain/credits/test_grant_models.py`.
- Create `dashboard/backend/tests/domain/credits/test_grant_repository_contract.py`.
- Create `dashboard/backend/tests/domain/credits/test_grant_service.py`.
- Create `dashboard/backend/tests/test_admin_credits_api.py`.
- Create `dashboard/backend/tests/test_admin_credits_frontend.py`.
- Modify existing Credits, Postgres, CSRF, route-composition, store-parity, Admin-user, and frontend tests listed in the tasks below.

---

### Task 1: Freeze Grant Command and Result Contracts

**Files:**
- Modify: `dashboard/backend/domain/credits/models.py`
- Modify: `dashboard/backend/domain/credits/repository_common.py`
- Create: `dashboard/backend/tests/domain/credits/test_grant_models.py`

**Interfaces:**
- Consumes: existing `format_credits(credits_micro: int) -> str`.
- Produces: `FundGrantPoolRequest`, `ReduceGrantPoolRequest`, `AssignGrantRequest`, `ReclaimGrantRequest`, `BalanceProjection`, `GrantPoolSummary`, `GrantMutationResult`.
- Produces: `IdempotencyConflictError`, `GrantPoolInsufficientError`, `GrantReclaimExceedsAvailableError`, `CreditAccountRestrictedStoreError`, `_required_text`, `_nonzero_integer`, `_canonical_digest`.

- [ ] **Step 1: Write strict model tests**

Create tests that prove booleans, floats, strings, zero, negative amounts,
untrimmed audit fields, blank audit fields, and extra JSON keys fail before the
service runs:

```python
@pytest.mark.parametrize("amount", [True, 1.0, "1000000", 0, -1])
def test_grant_commands_require_positive_strict_micro_credit_integer(amount):
    with pytest.raises(ValidationError):
        FundGrantPoolRequest(
            client_request_id=UUID("11111111-1111-4111-8111-111111111111"),
            amount_micro=amount,
            source="operations_budget",
            reason="Research allocation.",
        )


@pytest.mark.parametrize("field,value", [("source", ""), ("source", " x"), ("reason", " ")])
def test_grant_commands_require_trimmed_audit_text(field, value):
    payload = {
        "client_request_id": UUID("11111111-1111-4111-8111-111111111111"),
        "amount_micro": 1_000_000,
        "source": "operations_budget",
        "reason": "Research allocation.",
    }
    payload[field] = value
    with pytest.raises(ValidationError):
        AssignGrantRequest(**payload)
```

Also pin frozen result serialization and the backward-compatible total aliases
on `BalanceResult`.

- [ ] **Step 2: Run the tests and verify the red state**

Run:

```bash
python -m pytest dashboard/backend/tests/domain/credits/test_grant_models.py -q
```

Expected: collection/import failure because the Grant types do not exist.

- [ ] **Step 3: Add exact types and shared errors**

Add a common strict command base and typed projections:

```python
class _GrantCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_request_id: UUID
    amount_micro: StrictInt = Field(gt=0)
    source: str = Field(min_length=1, max_length=120)
    reason: str = Field(min_length=1, max_length=500)

    @field_validator("source", "reason")
    @classmethod
    def validate_trimmed_audit_text(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("audit text must be trimmed")
        return value


class FundGrantPoolRequest(_GrantCommand):
    pass


class ReduceGrantPoolRequest(_GrantCommand):
    pass


class AssignGrantRequest(_GrantCommand):
    pool_id: str = "default"


class ReclaimGrantRequest(_GrantCommand):
    pool_id: str = "default"
```

Add `BalanceProjection` with committed/available Grant and Purchased integer
fields plus formatted fields. Extend the existing `BalanceResult` additively:

```python
class BalanceResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    balance_micro: int
    display_credits: str
    grant_committed_micro: int
    purchased_committed_micro: int
    grant_available_micro: int
    purchased_available_micro: int
    total_available_micro: int
    display_grant_credits: str
    display_purchased_credits: str
    display_total_credits: str
    spending_enabled: bool = False
    account_status: str
    billing_available: bool
```

Add a frozen `GrantPoolSummary` with pool identity/status plus these exact
integer metrics: `pool_available_micro`, `allocated_to_users_micro`,
`assigned_this_month_micro`, and `reclaimed_this_month_micro`. Add their
formatted Credit strings and the UTC `month_start_iso` used for the query.

Implement `_canonical_digest(parts: Mapping[str, object]) -> str` using sorted,
compact JSON and SHA-256. The digest input must include operation, actor, pool,
target user, amount, source, and reason.

- [ ] **Step 4: Run model and existing Credits service tests**

Run:

```bash
python -m pytest \
  dashboard/backend/tests/domain/credits/test_grant_models.py \
  dashboard/backend/tests/domain/credits/test_service.py -q
```

Expected: all pass; existing `BalanceResult` construction is updated to provide
the additive fields.

- [ ] **Step 5: Commit the contract layer**

```bash
git add \
  dashboard/backend/domain/credits/models.py \
  dashboard/backend/domain/credits/repository_common.py \
  dashboard/backend/tests/domain/credits/test_grant_models.py \
  dashboard/backend/tests/domain/credits/test_service.py
git commit -m "feat(credits): define grant accounting contracts"
```

---

### Task 2: Migrate SQLite to Bucketed Ledgers and Add Atomic Grant Accounting

**Files:**
- Modify: `dashboard/backend/domain/credits/repository.py`
- Create: `dashboard/backend/tests/domain/credits/test_grant_repository_contract.py`
- Modify: `dashboard/backend/tests/domain/credits/test_repository.py`

**Interfaces:**
- Consumes: Task 1 validation helpers and store errors.
- Produces: `get_balance_projection(user_id)`, `get_balance_projections(user_ids)`, `get_grant_pool_summary(pool_id, month_start_iso)`, `list_grant_pool_activity(...)`, `fund_grant_pool(...)`, `reduce_grant_pool(...)`, `assign_grant(...)`, `reclaim_grant(...)`.
- Preserves: all existing order, webhook, refund, and pagination methods.

- [ ] **Step 1: Write the migration preservation test**

Build a database using the current production ledger schema, insert one paid
purchase and one refund, instantiate the upgraded store, and compare immutable
evidence:

```python
def test_sqlite_migration_preserves_existing_stripe_journal(tmp_path):
    path = tmp_path / "legacy.db"
    seed_current_credits_schema_with_purchase_and_refund(path)

    store = CreditsStore(path)
    page = store.list_ledger_entries(USER_ID, limit=50)

    assert [(row["id"], row["bucket"], row["amount_micro"]) for row in page["items"]] == [
        (2, "purchased", -2_000_000),
        (1, "purchased", 10_000_000),
    ]
    assert page["items"][0]["refund_request_id"] == "rfnd_legacy"
    assert page["items"][1]["payment_order_id"] == "ord_legacy"
    assert all(row["actor_user_id"] is None for row in page["items"])
```

The fixture must create the actual pre-PR2 columns, not call the new store's
schema helper.

- [ ] **Step 2: Write the SQLite accounting contract tests**

Cover all four operations, projections, page cursors, and Purchased isolation:

```python
def test_assign_and_reclaim_are_paired_and_leave_purchased_unchanged(store):
    fund(store, amount_micro=10_000_000, request_id="fund-1")
    purchased_before = purchased_balance(store, USER_ID)

    assigned = store.assign_grant(
        pool_id="default",
        user_id=USER_ID,
        amount_micro=3_000_000,
        operation_id="grant_assign_1",
        idempotency_key="assign-1",
        request_digest="digest-assign-1",
        actor_user_id=ADMIN_ID,
        source="research_budget",
        reason="Initial sponsored allocation.",
    )

    assert assigned["pool"]["balance_micro"] == 7_000_000
    assert assigned["user_balance"]["grant_available_micro"] == 3_000_000
    assert purchased_balance(store, USER_ID) == purchased_before
    assert assigned["entry"]["operation_id"] == assigned["user_entry"]["operation_id"]
```

Add tests for:

- pool reduction cannot overdraft;
- reclaim cannot exceed Grant;
- assignment rejects restricted account but reclaim succeeds;
- identical replay returns the same entry IDs;
- changed digest with the same idempotency key raises
  `IdempotencyConflictError`;
- injected failure after the user insert rolls back both lines;
- two SQLite threads assigning the last pool amount yield one success and one
  `GrantPoolInsufficientError`; and
- `get_balance_projections([])` returns `{}` and a mixed list returns zeros for
  accounts without a ledger;
- `get_grant_pool_summary("default", month_start_iso)` reports pool available,
  current Grant allocated to users, and absolute assigned/reclaimed totals for
  that UTC month; and
- reopening the store preserves an Admin-renamed pool and its ledger-derived
  balance instead of overwriting either during startup.

- [ ] **Step 3: Run the new tests and verify the red state**

Run:

```bash
python -m pytest \
  dashboard/backend/tests/domain/credits/test_grant_repository_contract.py \
  dashboard/backend/tests/domain/credits/test_repository.py -q
```

Expected: failures for missing bucket columns, pool tables, and methods.

- [ ] **Step 4: Implement the idempotent SQLite ledger migration**

Define one canonical ledger DDL with:

```sql
bucket TEXT NOT NULL CHECK (bucket IN ('grant', 'purchased')),
entry_type TEXT NOT NULL CHECK (entry_type IN (
  'purchase', 'refund', 'admin_grant_assign', 'admin_grant_reclaim'
)),
payment_order_id TEXT,
refund_request_id TEXT,
stripe_event_id TEXT,
operation_key TEXT NOT NULL UNIQUE,
operation_id TEXT NOT NULL,
idempotency_key TEXT NOT NULL UNIQUE,
request_digest TEXT,
actor_user_id INTEGER,
source TEXT NOT NULL CHECK (length(trim(source)) > 0),
reason TEXT NOT NULL CHECK (length(trim(reason)) > 0),
reference_type TEXT,
reference_id TEXT,
created_at TEXT NOT NULL
```

Add a table constraint requiring `request_digest IS NOT NULL` only for
`admin_grant_assign` and `admin_grant_reclaim`. When `bucket` is absent, rebuild
the table in the active migration transaction, copy the old rows with
`bucket='purchased'`, preserve the old evidence columns, set only
`operation_id/idempotency_key` from the existing unique operation key, leave
`request_digest=NULL`, set `actor_user_id=NULL`, and use deterministic
Stripe/system source/reason labels. Do not invent a digest for historical
Stripe/system rows. Drop the legacy table only after the copy count and signed
amount sum match.

- [ ] **Step 5: Add Grant Pool DDL and projections**

Create `credit_grant_pools` and `credit_grant_pool_ledger_entries`. Seed the
pool identity exactly once with ID `default` and name
`Platform Research Grants` using `INSERT ... ON CONFLICT DO NOTHING`; this seed
creates no ledger entry and adds no funding. Startup must never update an
existing pool name, status, or ledger-derived balance. The pool ledger stores
the canonical request digest and a nullable paired user entry ID. Add indexes
for `(pool_id, id DESC)`, `(user_id, id DESC)`, and unique operation and
idempotency keys.

Implement balance SQL with conditional sums:

```sql
SELECT
  COALESCE(SUM(CASE WHEN bucket = 'grant' THEN amount_micro ELSE 0 END), 0)
    AS grant_committed_micro,
  COALESCE(SUM(CASE WHEN bucket = 'purchased' THEN amount_micro ELSE 0 END), 0)
    AS purchased_committed_micro
FROM credit_ledger_entries
WHERE user_id = ?
```

Keep `get_balance_micro(user_id)` as the total alias so current callers do not
break. Implement `get_grant_pool_summary(pool_id, month_start_iso)` with one
stable snapshot containing:

```text
pool_available_micro = signed sum of all pool entries
allocated_to_users_micro = signed sum of all current user Grant entries
assigned_this_month_micro = absolute sum of assign pool entries since month_start_iso
reclaimed_this_month_micro = sum of reclaim pool entries since month_start_iso
```

Treat `month_start_iso` as an inclusive UTC boundary and use the same boundary
for SQLite and PostgreSQL.

- [ ] **Step 6: Implement the four transactional mutations**

Use `BEGIN IMMEDIATE`. For each operation:

1. ensure the named default pool exists with `ON CONFLICT DO NOTHING`, without
   creating funding or overwriting Admin-managed state;
2. read any existing idempotency row and compare `request_digest`;
3. verify pool/account/user constraints inside the transaction;
4. insert the paired user line first for assign/reclaim;
5. insert the pool line referencing that user line;
6. return the entry IDs and fresh projections before commit.

Do not call a public repository method from inside the transaction; use private
`_..._in_transaction` helpers so the operation stays on one connection.

- [ ] **Step 7: Run SQLite Credits regression tests**

Run:

```bash
python -m pytest \
  dashboard/backend/tests/domain/credits/test_grant_repository_contract.py \
  dashboard/backend/tests/domain/credits/test_repository.py \
  dashboard/backend/tests/integration/test_credits_checkout_flow.py -q
```

Expected: all pass, including existing purchase/refund behavior.

- [ ] **Step 8: Commit SQLite accounting**

```bash
git add \
  dashboard/backend/domain/credits/repository.py \
  dashboard/backend/tests/domain/credits/test_grant_repository_contract.py \
  dashboard/backend/tests/domain/credits/test_repository.py
git commit -m "feat(credits): add atomic grant pool accounting"
```

---

### Task 3: Build the PostgreSQL Twin and Prove Concurrency

**Files:**
- Modify: `dashboard/backend/domain/credits/repository_postgres.py`
- Modify: `dashboard/backend/tests/domain/credits/test_repository_postgres.py`
- Modify: `dashboard/backend/tests/test_store_twin_parity.py`

**Interfaces:**
- Consumes: Task 2 public repository signatures and shared contract helpers.
- Produces: behaviorally equivalent PostgreSQL methods with row-level locks.

- [ ] **Step 1: Parameterize the shared repository contract**

Expose assertions in `test_grant_repository_contract.py` that accept a store
fixture. Run them for SQLite directly and from the live PostgreSQL module. Do
not mock psycopg for transactional or concurrency behavior.

- [ ] **Step 2: Add PostgreSQL migration and concurrency tests**

Start from the current pre-PR2 Postgres tables, insert a purchase/refund pair,
run `_init_schema()`, then prove preserved IDs/references/amounts and nullable
historical actors. Add real concurrent operations:

```python
def test_postgres_concurrent_assign_cannot_overdraw_pool(pg_credits_store):
    fund(pg_credits_store, amount_micro=1_000_000, request_id="fund-one")
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(assign_one_credit, ("a", "b")))
    assert sum(result == "ok" for result in results) == 1
    assert sum(result == "insufficient" for result in results) == 1
    summary = pg_credits_store.get_grant_pool_summary("default", MONTH_START_ISO)
    assert summary["pool_available_micro"] == 0
```

Repeat for reclaiming the final user Grant amount.

- [ ] **Step 3: Run PostgreSQL tests and verify the red state**

Run:

```bash
python -m pytest \
  dashboard/backend/tests/domain/credits/test_repository_postgres.py \
  dashboard/backend/tests/test_store_twin_parity.py -q
```

Expected: live-store tests fail for missing schema/method parity. If the local
PostgreSQL service is absent, record the skip locally but require the CI service
run before PR readiness.

- [ ] **Step 4: Implement transactional Postgres migration**

Use an idempotent add/backfill/constraint sequence in one connection
transaction. Historical rows receive Purchased/system values without invented
Admin actors or request digests. Keep `request_digest` nullable for historical
Stripe/system rows and require it with a shape constraint for new Grant rows.
Seed the `default` / `Platform Research Grants` pool identity only when absent,
without a funding entry and without updating existing Admin-managed state.
Add `NOT NULL` only to columns whose historical rows have truthful backfills.
Keep all Stripe foreign-key constraints and add Grant shape constraints.

- [ ] **Step 5: Implement locked Postgres operations**

Match Task 2 signatures exactly. Acquire locks in a fixed order:

```sql
SELECT * FROM credit_grant_pools WHERE id = %s FOR UPDATE;
SELECT status FROM credit_accounts WHERE user_id = %s FOR UPDATE;
```

For assign/reclaim, lock pool first and account second on every path. Read the
idempotency row under the same transaction, compare the stored digest, insert
both ledger lines, then return fresh projections. Never catch a database error
and fall back to SQLite.

- [ ] **Step 6: Extend parity guards**

Add the Grant methods to the public store parity assertion and the new tables
to DDL column parity. Pin SQLite's lazy/rebuild migration to an explicit
PostgreSQL migration counterpart so future schema edits cannot update only one
backend.

- [ ] **Step 7: Run both repository backends together**

Run:

```bash
python -m pytest \
  dashboard/backend/tests/domain/credits/test_grant_repository_contract.py \
  dashboard/backend/tests/domain/credits/test_repository.py \
  dashboard/backend/tests/domain/credits/test_repository_postgres.py \
  dashboard/backend/tests/test_store_twin_parity.py -q
```

Expected: all available tests pass; no live Postgres test may skip in GitHub CI.

- [ ] **Step 8: Commit PostgreSQL parity**

```bash
git add \
  dashboard/backend/domain/credits/repository_postgres.py \
  dashboard/backend/tests/domain/credits/test_repository_postgres.py \
  dashboard/backend/tests/test_store_twin_parity.py
git commit -m "feat(credits): mirror grant accounting in postgres"
```

---

### Task 4: Add Service Orchestration and Admin Credits APIs

**Files:**
- Modify: `dashboard/backend/domain/credits/service.py`
- Modify: `dashboard/backend/users.py`
- Modify: `dashboard/backend/users_postgres.py`
- Create: `dashboard/backend/api/routers/admin_credits.py`
- Modify: `dashboard/backend/api/routers/credits.py`
- Modify: `dashboard/backend/api/router.py`
- Create: `dashboard/backend/tests/domain/credits/test_grant_service.py`
- Create: `dashboard/backend/tests/test_admin_credits_api.py`
- Modify: `dashboard/backend/tests/test_credits_api.py`
- Modify: `dashboard/backend/tests/test_admin_users.py`
- Modify: `dashboard/backend/tests/test_users_postgres.py`
- Modify: `dashboard/backend/tests/test_csrf.py`
- Modify: `dashboard/backend/tests/test_app_composition.py`

**Interfaces:**
- Consumes: Tasks 1-3 models and repositories.
- Produces: `CreditsService.fund_grant_pool`, `reduce_grant_pool`, `assign_grant`, `reclaim_grant`, `get_grant_pool_summary`, `list_grant_pool_activity`, `get_balance_projections`.
- Produces: the exact user/Admin routes from the approved spec.

- [ ] **Step 1: Write service orchestration tests**

Use a recording fake store to prove the service derives deterministic IDs and
binds replay to the full command:

```python
def test_assign_grant_binds_actor_target_and_audit_text_to_digest():
    request = AssignGrantRequest(
        client_request_id=UUID("11111111-1111-4111-8111-111111111111"),
        amount_micro=2_000_000,
        source="research_budget",
        reason="Approved pilot.",
    )
    result = service.assign_grant(admin_id=7, user_id=9, request=request)
    call = store.calls.single()
    assert call["operation_id"].startswith("grant_assign_")
    assert call["idempotency_key"] == "admin-grant:11111111-1111-4111-8111-111111111111"
    assert call["request_digest"] == canonical_digest(
        operation="assign", actor=7, pool="default", user=9,
        amount_micro=2_000_000, source="research_budget", reason="Approved pilot."
    )
```

Pin safe service errors for insufficient pool, excessive reclaim, restricted
account, and idempotency conflict. Pin
`get_grant_pool_summary(pool_id="default", month_start_iso=None)` so an omitted
boundary is computed as the first instant of the current UTC month and the
exact ISO boundary is passed to the repository.

- [ ] **Step 2: Write Admin/user API tests first**

Cover:

- every new Admin route returns `401` signed out and `403` for a normal user;
- router-level `Depends(require_admin)` guards future handlers;
- cookie mutations require CSRF and allowed Origin;
- strict amount Pydantic validation rejects float/string/bool;
- pool/user insufficient errors are sanitized `422` responses;
- idempotency conflicts are `409`;
- Admin user search returns only safe identity fields plus bucket projections;
- ordinary users cannot pass a user ID to read another balance;
- balance retains legacy total aliases and adds `spending_enabled: false`;
- ledger public fields include bucket/source/reason but omit actor internals not
  intended for the personal wallet; and
- route registration appears once in frozen composition tests.

- [ ] **Step 3: Run service/API tests and verify the red state**

Run:

```bash
python -m pytest \
  dashboard/backend/tests/domain/credits/test_grant_service.py \
  dashboard/backend/tests/test_admin_credits_api.py \
  dashboard/backend/tests/test_credits_api.py \
  dashboard/backend/tests/test_csrf.py \
  dashboard/backend/tests/test_app_composition.py -q
```

Expected: missing methods/router and old balance shape failures.

- [ ] **Step 4: Implement service methods and additive balance output**

Add one private command builder:

```python
def _grant_command(self, *, operation, admin_id, user_id, request):
    parts = {
        "operation": operation,
        "actor_user_id": admin_id,
        "pool_id": getattr(request, "pool_id", "default"),
        "user_id": user_id,
        "amount_micro": request.amount_micro,
        "source": request.source,
        "reason": request.reason,
    }
    return {
        **parts,
        "operation_id": _operation_id(f"grant_{operation}", request.client_request_id),
        "idempotency_key": f"admin-grant:{request.client_request_id}",
        "request_digest": _canonical_digest(parts),
    }
```

Each public mutation delegates exactly once to its repository counterpart and
maps the fresh projections into `GrantMutationResult`. `get_balance()` uses the
bucket projection and sets the legacy aliases equal to total.
`get_grant_pool_summary(...)` maps all four integer metrics and formatted
values into `GrantPoolSummary`; `list_grant_pool_activity(...)` is the only
pool-activity method name used by the service, router, and both repositories.

- [ ] **Step 5: Add server-side Admin user search without moving identity ownership**

Extend both UserStore twins with optional `query: str | None = None` on
`list_users_admin` and `count_users`. Normalize with `strip()` and use escaped,
case-insensitive email/display-name matching. Existing callers that omit query
must produce byte-identical results.

The Admin Credits route obtains identity rows from UserStore, then calls one
batch Credits projection for those user IDs and merges safe fields in memory.
CreditsStore must not query or render user emails/display names.

- [ ] **Step 6: Create the separately authorized Admin Credits router**

Use:

```python
router = APIRouter(
    prefix="/admin/credits",
    tags=["admin-credits"],
    dependencies=[Depends(require_admin)],
)
```

Implement:

```text
GET  /grant-pool
GET  /grant-pool/activity
POST /grant-pool/fund
POST /grant-pool/reduce
GET  /users
POST /grants/assign
POST /grants/reclaim
GET  /activity
```

The mutation limiter keys on the authenticated Admin ID. Require the named
Admin dependency again on handlers that need the actor row; FastAPI caches it.
Use a private safe error mapper dedicated to Grant domain errors. Do not route
Grant errors through Stripe's `_raise_billing_http_error`.

- [ ] **Step 7: Register once and update user public projections**

Include `admin_credits_router` once in `api/router.py`. Update only the public
balance/ledger serializers in `credits.py`; keep checkout, order, webhook, and
refund handlers unchanged.

- [ ] **Step 8: Run focused backend API tests**

Run:

```bash
python -m pytest \
  dashboard/backend/tests/domain/credits/test_grant_service.py \
  dashboard/backend/tests/test_admin_credits_api.py \
  dashboard/backend/tests/test_credits_api.py \
  dashboard/backend/tests/test_admin_users.py \
  dashboard/backend/tests/test_users_postgres.py \
  dashboard/backend/tests/test_csrf.py \
  dashboard/backend/tests/test_app_composition.py \
  dashboard/backend/tests/test_architecture_boundaries.py -q
```

Expected: all pass.

- [ ] **Step 9: Commit service and API**

```bash
git add \
  dashboard/backend/domain/credits/service.py \
  dashboard/backend/users.py \
  dashboard/backend/users_postgres.py \
  dashboard/backend/api/routers/admin_credits.py \
  dashboard/backend/api/routers/credits.py \
  dashboard/backend/api/router.py \
  dashboard/backend/tests/domain/credits/test_grant_service.py \
  dashboard/backend/tests/test_admin_credits_api.py \
  dashboard/backend/tests/test_credits_api.py \
  dashboard/backend/tests/test_admin_users.py \
  dashboard/backend/tests/test_users_postgres.py \
  dashboard/backend/tests/test_csrf.py \
  dashboard/backend/tests/test_app_composition.py
git commit -m "feat(credits): expose audited admin grants"
```

---

### Task 5: Build the Top-Level Admin Grant Credits Workspace

**Files:**
- Modify: `dashboard/frontend/app.html`
- Create: `dashboard/frontend/js/admin-credits.js`
- Modify: `dashboard/frontend/app.js`
- Modify: `dashboard/frontend/styles.css`
- Create: `dashboard/backend/tests/test_admin_credits_frontend.py`
- Modify: `dashboard/backend/tests/test_admin_console_frontend.py`
- Modify: `dashboard/backend/tests/test_frontend_fast_boot.py`

**Interfaces:**
- Consumes: Task 4 `/api/admin/credits/*` payloads and shared `apiRequest`.
- Produces: `window.AdminCredits = { onEnter, refresh, syncAuth, selectTab }`.
- Preserves: existing Users and Model Providers functionality.

- [ ] **Step 1: Write static frontend contract tests**

Pin the structural boundary:

```python
def test_admin_has_users_grants_and_provider_tabs():
    admin = admin_markup()
    assert 'data-admin-tab="users"' in admin
    assert 'data-admin-tab="grant-credits"' in admin
    assert 'data-admin-tab="model-providers"' in admin
    assert 'id="adminGrantCreditsPanel"' in admin
    assert 'id="creditsAdminSection"' not in credits_activity_markup()


def test_legacy_run_credits_are_not_presented_as_grant_balance():
    assert "Legacy run Credits" in APP_HTML
    assert "Grant available" in APP_HTML
    assert "Purchased" in APP_HTML
```

Also assert `admin-credits.js` loads after the shared API wrapper, does not use
`innerHTML`, does not place mutations in URLs/localStorage, and converts decimal
Credit strings without `parseFloat` or multiplication by a floating value.
Update `test_frontend_fast_boot.py` to require the exact changed-resource
cachebusters: `styles.css?v=119`, `app.js?v=116`, `js/credits.js?v=4`, and the
new `js/admin-credits.js?v=1`. Preserve the existing script-order assertions.

- [ ] **Step 2: Run static tests and verify the red state**

Run:

```bash
python -m pytest \
  dashboard/backend/tests/test_admin_credits_frontend.py \
  dashboard/backend/tests/test_admin_console_frontend.py \
  dashboard/backend/tests/test_frontend_fast_boot.py -q
```

Expected: missing tab/panel/script failures.

- [ ] **Step 3: Restructure Admin markup into three sibling panels**

Add an Admin tablist after the header. Wrap the existing user table in the
Users panel and existing provider markup in the Model Providers panel without
nesting tool cards. Add the Grant Credits panel from the approved prototype:

- stable four-column pool summary band;
- search/filter toolbar and user table;
- right-side `dialog`/drawer with Assign/Reclaim segmented control;
- amount, source, reason fields and explicit Purchased-is-unchanged text;
- confirmation modal for fund/reduce/assign/reclaim; and
- append-only activity table.

Use the repository's existing icon sprite in `app.html`; do not add inline SVG
paths or text-only icon substitutes.

- [ ] **Step 4: Implement exact decimal parsing and safe rendering**

In `admin-credits.js`:

```javascript
function parseCreditsToMicro(value) {
  const match = /^([0-9]+)(?:\.([0-9]{1,6}))?$/.exec(String(value || '').trim());
  if (!match) throw new Error('Enter a positive amount with up to 6 decimals.');
  const whole = BigInt(match[1]);
  const fraction = BigInt((match[2] || '').padEnd(6, '0'));
  const micro = whole * 1000000n + fraction;
  if (micro <= 0n || micro > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error('Credit amount is outside the supported range.');
  }
  return Number(micro);
}
```

Render API text via `textContent`/text nodes only. Keep the active mutation's
UUID in memory only while its amount/user/source/reason remain unchanged so an
ambiguous retry reuses the key; changing any field generates a fresh UUID.

- [ ] **Step 5: Implement Admin state and interaction flow**

`onEnter()` wires controls once. `selectTab()` updates `aria-selected`, panel
visibility, and loads only the selected section. `refresh()` reloads the active
section. Grant loading fetches pool summary, paginated users, and activity in
parallel, then renders a coherent success/error state.

The drawer must:

- populate from the selected row without storing user data in localStorage;
- show Grant and Purchased separately;
- disable Reclaim above the current Grant available amount;
- clear amount/source/reason and pending UUID on close, logout, or demotion;
- require a confirmation step naming operation, target, and amount; and
- after success, refresh pool, selected user, user table, and activity.

- [ ] **Step 6: Integrate existing Admin Users and Providers**

In `app.js`, Admin entry selects the last in-memory tab or Users by default,
calls `AdminCredits.onEnter()`, and retains existing stats/users/provider
loading. Rename the old column label to `Legacy run Credits` and keep its live
metering status note. The top-level Refresh button delegates to the active
panel instead of loading hidden panels unconditionally. In `app.html`, bump
`app.js?v=115` to `app.js?v=116`, `styles.css?v=118` to `styles.css?v=119`, and
`js/credits.js?v=3` to `js/credits.js?v=4`; add
`js/admin-credits.js?v=1` after `app.js` so the shared API wrapper already
exists. Pin all four exact references in `test_frontend_fast_boot.py`.

- [ ] **Step 7: Add stable responsive styling**

Use explicit grid tracks and drawer dimensions matching the approved prototype.
At `<=980px`, collapse summary metrics to two columns; at `<=700px`, stack
metrics, allow tab scrolling, keep tables in their own horizontal scroll area,
and make the drawer full viewport width. Buttons and counters must not resize
their parent tracks when values change.

- [ ] **Step 8: Run Admin frontend tests**

Run:

```bash
python -m pytest \
  dashboard/backend/tests/test_admin_credits_frontend.py \
  dashboard/backend/tests/test_admin_console_frontend.py \
  dashboard/backend/tests/test_admin_model_providers_frontend.py \
  dashboard/backend/tests/test_frontend_fast_boot.py -q
```

Expected: all pass.

- [ ] **Step 9: Commit the Admin workspace**

```bash
git add \
  dashboard/frontend/app.html \
  dashboard/frontend/js/admin-credits.js \
  dashboard/frontend/app.js \
  dashboard/frontend/styles.css \
  dashboard/backend/tests/test_admin_credits_frontend.py \
  dashboard/backend/tests/test_admin_console_frontend.py \
  dashboard/backend/tests/test_frontend_fast_boot.py
git commit -m "feat(admin): add grant credits workspace"
```

---

### Task 6: Add User Grant/Purchased Balance and Activity Presentation

**Files:**
- Modify: `dashboard/frontend/app.html`
- Modify: `dashboard/frontend/js/credits.js`
- Modify: `dashboard/frontend/styles.css`
- Modify: `dashboard/backend/tests/test_credits_frontend.py`

**Interfaces:**
- Consumes: Task 4 additive `/api/credits/balance` and ledger fields.
- Produces: personal Grant/Purchased/total breakdown and bucket-labeled activity.
- Preserves: Overview, Top up, API Keys, Activity tab order and all Stripe/API-key controls.

- [ ] **Step 1: Write user-surface compatibility tests**

Pin:

- exactly four Credits tabs remain `Overview`, `Top up`, `API Keys`, `Activity`;
- Overview contains Grant, Purchased, and Total values;
- the status copy is exactly `Model spending is not enabled yet.`;
- activity renders bucket and reason through text nodes;
- old `balance_micro/display_credits` fallback remains;
- no Admin Grant mutation controls or endpoints occur in `credits.js`; and
- Top up and API Keys IDs/scripts remain present.

- [ ] **Step 2: Run user frontend tests and verify the red state**

Run:

```bash
python -m pytest dashboard/backend/tests/test_credits_frontend.py -q
```

Expected: missing balance-breakdown/bucket rendering failures.

- [ ] **Step 3: Add unframed balance breakdown markup**

Under the existing Overview balance band, add a stable three-column breakdown
for Grant, Purchased, and Total. Do not add a nested card. Keep the spending
status visible in the first viewport and preserve the signed-webhook note for
Purchased Credits.

- [ ] **Step 4: Render additive values with backward fallback**

Use total aliases first and old fields only as fallback:

```javascript
const totalMicro = Number.isSafeInteger(balance.total_available_micro)
  ? balance.total_available_micro
  : Number(balance.balance_micro || 0);
const totalDisplay = balance.display_total_credits || balance.display_credits;
```

Render ledger labels from an explicit allowlist for purchase, refund,
admin-grant assign, and admin-grant reclaim. Unknown values receive a safe
generic label; raw HTML is never interpreted.

- [ ] **Step 5: Run Credits frontend and API regression tests**

Run:

```bash
python -m pytest \
  dashboard/backend/tests/test_credits_frontend.py \
  dashboard/backend/tests/test_credits_api.py \
  dashboard/backend/tests/test_model_credentials_api.py \
  dashboard/backend/tests/test_admin_model_providers_frontend.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit the personal wallet presentation**

```bash
git add \
  dashboard/frontend/app.html \
  dashboard/frontend/js/credits.js \
  dashboard/frontend/styles.css \
  dashboard/backend/tests/test_credits_frontend.py
git commit -m "feat(credits): show grant and purchased balances"
```

---

### Task 7: Exercise the Complete PR2 Flow in Browser

**Files:**
- Modify only if browser verification reveals a defect in a file owned by Tasks 1-6.
- Do not modify: `dashboard/storage/data/backtest.db`.

**Interfaces:**
- Consumes: complete PR2 API and frontend.
- Produces: desktop/mobile evidence for pool fund, assign, reclaim, user balance, and activity.

- [ ] **Step 1: Start an isolated local server**

Use a temporary database and fake accounts. Do not load the committed seed DB:

```bash
tmp_dir=$(mktemp -d /tmp/atl-admin-grants.XXXXXX)
DATABASE_PATH="$tmp_dir/backtest.db" \
BROKER_TOKEN_ENCRYPTION_KEY="$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')" \
python -m uvicorn dashboard.backend.app:app --host 127.0.0.1 --port 8770
```

If port 8770 is occupied, select the next free localhost port and record it.

- [ ] **Step 2: Seed only fake local identities and pool data**

Create an Admin and two ordinary fake accounts through the local API/test
helper. Use fake emails under `example.com`. Fund a fake pool through the Admin
API with `source="browser_test"`; do not enter a payment method or any API key.

- [ ] **Step 3: Verify desktop workflow at 1440x900**

Using the in-app browser:

1. open the top-level Admin entry;
2. switch among Users, Grant Credits, and Model Providers;
3. fund and reduce the pool;
4. search a user;
5. assign then reclaim part of the Grant;
6. verify Purchased never changes;
7. verify the activity row includes actor/source/reason/time; and
8. open that user's Credits & Billing Overview and Activity to verify the
   personal projection and disabled-spending copy.

Capture screenshots after the Grant table and open drawer states.

- [ ] **Step 4: Verify mobile workflow at 390x844**

Confirm no body-level horizontal overflow, tabs remain reachable, metrics stack,
the user table scrolls inside its container, the drawer fills the viewport,
text fits controls, and no fixed element covers submit/cancel.

- [ ] **Step 5: Inspect browser console and secret/storage surfaces**

Require zero console errors. Confirm no Grant form content enters URLs or
localStorage. Confirm the page contains no real API key/payment data.

- [ ] **Step 6: Fix any browser defect with a failing static/API test first**

For each defect, add the smallest reproducing test, verify it fails, implement
the fix, rerun the owning focused suite, and create one English fix commit.

- [ ] **Step 7: Stop the isolated server and record evidence**

Stop every server process started by this task. Preserve screenshots outside
the repository or in `.superpowers/`; do not stage them.

---

### Task 8: Run Final Gates, Review the Diff, and Prepare PR2

**Files:**
- Modify only for failures attributable to PR2.
- Do not stage local/generated files.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: a reviewable branch with reproducible test evidence.

- [ ] **Step 1: Run formatting and scope guards**

Run:

```bash
git diff --check origin/main...HEAD
git status --short
git diff --name-status origin/main...HEAD
```

Verify every changed file belongs to the spec. Verify
`dashboard/storage/data/backtest.db`, `.superpowers/`, `AGENTS.md`, and `work/`
are absent from the diff.

- [ ] **Step 2: Run the complete focused PR2 suite**

Run:

```bash
python -m pytest \
  dashboard/backend/tests/domain/credits \
  dashboard/backend/tests/test_admin_credits_api.py \
  dashboard/backend/tests/test_admin_credits_frontend.py \
  dashboard/backend/tests/test_credits_api.py \
  dashboard/backend/tests/test_credits_frontend.py \
  dashboard/backend/tests/test_admin_users.py \
  dashboard/backend/tests/test_users_postgres.py \
  dashboard/backend/tests/test_csrf.py \
  dashboard/backend/tests/test_app_composition.py \
  dashboard/backend/tests/test_store_twin_parity.py \
  dashboard/backend/tests/test_architecture_boundaries.py -q
```

Expected: zero failures; GitHub CI must exercise live PostgreSQL tests without
skips.

- [ ] **Step 3: Run full backend and Packaging suites**

Run:

```bash
python -m pytest dashboard/backend/tests -q
python -m pytest packaging/agentictrading/tests -q
```

If the full backend suite touches the default local database, stop and correct
the test environment before continuing; never restore or stage the user's DB.

- [ ] **Step 4: Run a zero-trust diff review**

Review `origin/main...HEAD` for:

- mutable balance fields or update/delete ledger paths;
- any Grant method that can touch Purchased rows;
- missing transaction boundaries or inconsistent lock order;
- SQLite/Postgres signature or schema drift;
- idempotency digest omissions;
- Admin authorization that relies on hidden UI;
- unsafe exception text in HTTP responses;
- floats in authoritative amount conversion;
- localStorage/URL leakage of form data; and
- accidental model-spending or legacy-entitlement changes.

- [ ] **Step 5: Run an independent read-only Review task**

Give the reviewer only the PR diff and approved spec. Require findings ordered
by severity with exact file/line references and focused review of accounting,
authorization, migrations, concurrency, idempotency, and frontend isolation.
Do not merge while any confirmed finding remains.

- [ ] **Step 6: Push and open PR2 only after local gates pass**

Use an English PR title such as:

```text
feat(credits): add audited admin grant accounting
```

The PR body must state that model spending is excluded, list SQLite/PostgreSQL
evidence, and include the browser verification matrix. Wait for Backend,
Packaging, CodeQL, and independent review to finish before asking to merge.
