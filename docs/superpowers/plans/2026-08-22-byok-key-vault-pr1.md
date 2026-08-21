# BYOK Key Vault PR 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a secure, independently reviewable BYOK Key Vault with approved provider administration, encrypted user and platform credentials, strict no-cost verification, and user/admin UI surfaces, without enabling model execution.

**Architecture:** Rebuild only the audited Vault and Provider Registry bounded context on the latest `origin/main`; do not carry over the mixed execution commits. Keep policy in `domain/model_providers`, SQL in SQLite/PostgreSQL twin repositories, pinned outbound verification in `infrastructure/llm/adapters`, thin FastAPI routers, and separate user/admin frontend modules. Every security rule is introduced by a failing test and every lower-level gate must pass before browser verification.

**Tech Stack:** Python 3.13 in the current workspace (code remains compatible with the repository's supported Python floor), FastAPI, Pydantic v2, SQLite, PostgreSQL/psycopg, `cryptography.fernet`, `httpx` 0.28.1, pytest, vanilla HTML/CSS/JavaScript.

**Spec:** `docs/superpowers/specs/2026-08-22-three-pr-credits-byok-delivery-design.md`

## Global Constraints

- This plan implements PR 1 only. Admin Grant Credits and purchased-Credits spending receive separate plans.
- PR 1 supports `openrouter`, `openai`, `anthropic`, `gemini`, and administrator-approved `openai_compatible` providers.
- PR 1 never starts a model generation request and contains no `model_execution` package, backtest billing mode, quote, reservation, usage, or settlement code.
- Credits & Billing has exactly four tabs in this order: Overview, Top up, API Keys, Activity.
- User credentials belong to user accounts, never Agents.
- A user may keep multiple active named credentials per provider and no more than one verified default.
- User and platform credentials are encrypted with `BROKER_TOKEN_ENCRYPTION_KEY`; missing or invalid encryption configuration fails closed.
- A full secret never appears in an API response, URL, log, exception payload, `localStorage`, or reusable frontend state.
- Verification uses model-list GET requests only and intentionally creates no model-generation cost.
- Redirects are disabled and outbound validation connections are pinned to public DNS results.
- Revocation destroys ciphertext and retains only safe tombstone metadata.
- SQLite and PostgreSQL implement the same public repository contract.
- Source, tests, UI copy, commits, and PR descriptions are written in English.
- Never stage real credentials, `dashboard/storage/data/backtest.db`, `.superpowers/`, or `work/`.

---

## File Map

- `dashboard/backend/domain/model_providers/models.py`: strict public request/result types and safe credential projections.
- `dashboard/backend/domain/model_providers/repository_common.py`: provider-origin syntax rules, canonical request digests, exceptions, seed records, and shared public store contract helpers.
- `dashboard/backend/domain/model_providers/repository.py`: SQLite provider registry, encrypted credential vault, tombstones, defaults, and atomic admin mutations.
- `dashboard/backend/domain/model_providers/repository_postgres.py`: PostgreSQL twin with transaction and locking parity.
- `dashboard/backend/domain/model_providers/service.py`: ownership, verification orchestration, lifecycle policy, and admin idempotency policy.
- `dashboard/backend/infrastructure/llm/adapters/safe_http.py`: public-address resolution and IP-pinned HTTPS targets with original Host/SNI identity.
- `dashboard/backend/infrastructure/llm/adapters/base.py`: bounded no-cost validation flow and strict discovery-response parsing.
- `dashboard/backend/infrastructure/llm/adapters/{openai,openrouter,anthropic,gemini,registry}.py`: approved provider-specific headers, paths, and model parsing.
- `dashboard/backend/api/routers/model_credentials.py`: authenticated user credential APIs with bounded secret-bearing request parsing.
- `dashboard/backend/api/routers/admin_model_providers.py`: separately authorized admin registry and platform credential APIs.
- `dashboard/backend/api/router.py`: route registration only.
- `dashboard/frontend/js/credits.js`: user API Keys tab behavior without persistent secret state.
- `dashboard/frontend/js/admin-model-providers.js`: separate admin provider and platform credential controls.
- `dashboard/frontend/app.html`: user tab and admin provider markup plus scripts.
- `dashboard/frontend/styles.css`: scoped user and admin Vault presentation.
- `dashboard/backend/tests/domain/model_providers/test_repository_contract.py`: backend-neutral active credential, tombstone, default, seed, and admin transaction rules.
- `dashboard/backend/tests/domain/model_providers/test_service.py`: user ownership and verification lifecycle.
- `dashboard/backend/tests/domain/model_providers/test_admin_service.py`: admin request-digest and mutation policy.
- `dashboard/backend/tests/infrastructure/llm/adapters/test_safe_http.py`: public-address and pinned-target contract.
- `dashboard/backend/tests/infrastructure/llm/adapters/test_validation_contract.py`: provider verification matrix.
- `dashboard/backend/tests/test_model_credentials_api.py`: user API security contract.
- `dashboard/backend/tests/test_admin_model_providers_api.py`: admin authorization and safe-body contract.
- `dashboard/backend/tests/test_model_credentials_frontend.py`: user frontend secret-handling contract.
- `dashboard/backend/tests/test_admin_model_providers_frontend.py`: separate admin frontend contract.
- `dashboard/backend/tests/e2e/model_credentials_harness.py`: controlled browser-only ASGI harness with fake DNS and `httpx.MockTransport`.

### Task 0: Preserve the Mixed Work and Create a Clean PR Branch

**Files:**
- Preserve without staging: `dashboard/storage/data/backtest.db`
- Preserve without staging: `.superpowers/`
- Preserve without staging: `work/`
- Carry into the clean branch: `docs/superpowers/specs/2026-08-22-three-pr-credits-byok-delivery-design.md`
- Carry into the clean branch: `docs/superpowers/plans/2026-08-22-byok-key-vault-pr1.md`

**Interfaces:**
- Consumes: the current planned snapshot, which contains design commit `846f9ad`, this implementation plan, and mixed business-code parent `880f2d3`.
- Produces: clean branch `feature/byok-key-vault` based on the latest fetched `origin/main`, plus immutable archive branch `archive/credits-byok-mixed-planned`.

- [ ] **Step 1: Verify the current safety boundary**

Run:

```bash
git status --short --branch
git rev-parse HEAD
git merge-base --is-ancestor 846f9ad HEAD
test -f docs/superpowers/specs/2026-08-22-three-pr-credits-byok-delivery-design.md
test -f docs/superpowers/plans/2026-08-22-byok-key-vault-pr1.md
```

Expected: the two documentation files exist, `846f9ad` is an ancestor of HEAD, and only `dashboard/storage/data/backtest.db`, `.superpowers/`, and `work/` appear outside committed history.

- [ ] **Step 2: Preserve the mixed implementation**

Run:

```bash
git branch archive/credits-byok-mixed-planned HEAD
git show-ref --verify refs/heads/archive/credits-byok-mixed-planned
```

Expected: the archive ref resolves to the current planned snapshot and therefore preserves both documentation files plus all audited mixed commits.

- [ ] **Step 3: Obtain explicit approval and restore only the tracked database**

Do not run the restore until the user explicitly authorizes overwriting the audit-modified local database.

Run after approval:

```bash
git restore --source=HEAD -- dashboard/storage/data/backtest.db
git status --short
```

Expected: `dashboard/storage/data/backtest.db` disappears from status; `.superpowers/` and `work/` remain untracked and untouched.

- [ ] **Step 4: Refresh the target base and create the PR branch**

Run:

```bash
git fetch origin main
git switch -c feature/byok-key-vault origin/main
git cherry-pick 846f9ad archive/credits-byok-mixed-planned
```

Expected: the new branch contains the approved three-PR spec and PR 1 implementation plan commits, and none of the mixed business-code commits.

- [ ] **Step 5: Verify no excluded work crossed the branch boundary**

Run:

```bash
git log --oneline origin/main..HEAD
git diff --name-only origin/main..HEAD
git status --short
```

Expected: only the approved design/plan documents are committed; no `model_execution`, backtest billing, Credits reservation, local database, `.superpowers/`, or `work/` path appears.

### Task 1: Rebuild the Vault and Registry Foundation Without Execution Types

**Files:**
- Create: `dashboard/backend/domain/model_providers/__init__.py`
- Create: `dashboard/backend/domain/model_providers/models.py`
- Create: `dashboard/backend/domain/model_providers/repository_common.py`
- Create: `dashboard/backend/domain/model_providers/repository.py`
- Create: `dashboard/backend/domain/model_providers/repository_postgres.py`
- Create: `dashboard/backend/domain/model_providers/service.py`
- Create: `dashboard/backend/infrastructure/llm/adapters/__init__.py`
- Create: `dashboard/backend/infrastructure/llm/adapters/base.py`
- Create: `dashboard/backend/infrastructure/llm/adapters/openai.py`
- Create: `dashboard/backend/infrastructure/llm/adapters/openrouter.py`
- Create: `dashboard/backend/infrastructure/llm/adapters/anthropic.py`
- Create: `dashboard/backend/infrastructure/llm/adapters/gemini.py`
- Create: `dashboard/backend/infrastructure/llm/adapters/registry.py`
- Create: `dashboard/backend/api/routers/model_credentials.py`
- Create: `dashboard/backend/api/routers/admin_model_providers.py`
- Modify: `dashboard/backend/api/router.py`
- Test: `dashboard/backend/tests/domain/model_providers/test_service.py`
- Test: `dashboard/backend/tests/domain/model_providers/test_admin_service.py`
- Test: `dashboard/backend/tests/infrastructure/llm/adapters/test_validation_contract.py`
- Test: `dashboard/backend/tests/test_model_credentials_api.py`
- Test: `dashboard/backend/tests/test_admin_model_providers_api.py`

**Interfaces:**
- Consumes: existing authentication, rate limiting, SQLite database path, PostgreSQL pool, and broker Fernet helpers.
- Produces: `ModelProviderService`, `ModelProviderStore`, `PostgresModelProviderStore`, `get_adapter(adapter_type)`, user router, and admin router.
- Produces: `get_model_provider_service() -> ModelProviderService`, used as the single FastAPI dependency-injection seam for both production routing and the controlled browser harness.

- [ ] **Step 1: Restore only the audited PR 1 source and focused tests**

Run this exact source extraction; do not restore shared frontend files or execution files:

```bash
git restore --source=45527ca -- \
  dashboard/backend/domain/model_providers \
  dashboard/backend/infrastructure/llm/adapters \
  dashboard/backend/api/routers/model_credentials.py \
  dashboard/backend/api/routers/admin_model_providers.py \
  dashboard/backend/tests/domain/model_providers \
  dashboard/backend/tests/infrastructure/llm/adapters/test_validation_contract.py \
  dashboard/backend/tests/test_model_credentials_api.py \
  dashboard/backend/tests/test_admin_model_providers_api.py \
  dashboard/backend/tests/test_model_provider_store_postgres.py
```

Expected: no file under `dashboard/backend/domain/model_execution`, `provider_executor.py`, `routed_client.py`, or `backtests.py` changes.

- [ ] **Step 2: Write a failing scope test**

Add to `dashboard/backend/tests/test_architecture_boundaries.py`:

```python
def test_key_vault_pr_has_no_model_execution_runtime():
    forbidden = [
        REPO_ROOT / "dashboard/backend/domain/model_execution",
        REPO_ROOT / "dashboard/backend/infrastructure/llm/provider_executor.py",
        REPO_ROOT / "dashboard/backend/infrastructure/llm/routed_client.py",
    ]
    assert not any(path.exists() for path in forbidden)
```

- [ ] **Step 3: Run the scope test and observe the intended result**

Run:

```bash
python3 -m pytest dashboard/backend/tests/test_architecture_boundaries.py::test_key_vault_pr_has_no_model_execution_runtime -v
```

Expected: PASS on the clean base. If the name collides with a runtime file newly added to `main`, replace the existence assertion with a diff-scoped assertion against `origin/main` before continuing.

- [ ] **Step 4: Remove execution-only contracts from the restored domain model**

Keep these PR 1 types in `models.py`:

```python
AdapterType = Literal[
    "openrouter", "openai", "anthropic", "gemini", "openai_compatible"
]
CredentialStatus = Literal[
    "verified", "invalid", "verification_unavailable", "revoked"
]
```

Keep `ProviderCapabilities`, `ProviderRecord`, admin requests, public credential projections, `UserCredentialCreate`, and `CredentialValidation`. Remove `NormalizedUsage`, `NormalizedModelRequest`, `NormalizedModelResponse`, and `NormalizedProviderError`; PR 1 has no execution consumer for them.

- [ ] **Step 5: Register only the two PR 1 routers**

Add to `dashboard/backend/api/router.py` following the existing import and include ordering:

```python
from dashboard.backend.api.routers.admin_model_providers import (
    router as admin_model_providers_router,
)
from dashboard.backend.api.routers.model_credentials import (
    router as model_credentials_router,
)

api_router.include_router(admin_model_providers_router)
api_router.include_router(model_credentials_router)
```

Do not import `model_execution_router`.

Expose the service through a dependency rather than importing a mutable singleton directly inside endpoint bodies:

```python
def get_model_provider_service() -> ModelProviderService:
    return model_provider_service
```

Every user and admin endpoint accepts `service: ModelProviderService = Depends(get_model_provider_service)` and calls that instance. Tests and the browser harness override the dependency through `app.dependency_overrides`; production still receives the singleton.

- [ ] **Step 6: Run the extracted focused suite**

Run:

```bash
python3 -m pytest \
  dashboard/backend/tests/domain/model_providers \
  dashboard/backend/tests/infrastructure/llm/adapters/test_validation_contract.py \
  dashboard/backend/tests/test_model_credentials_api.py \
  dashboard/backend/tests/test_admin_model_providers_api.py -q
```

Expected: the restored baseline tests pass before security-hardening tests are added.

- [ ] **Step 7: Commit the bounded foundation**

Run:

```bash
git add dashboard/backend/domain/model_providers \
  dashboard/backend/infrastructure/llm/adapters \
  dashboard/backend/api/router.py \
  dashboard/backend/api/routers/model_credentials.py \
  dashboard/backend/api/routers/admin_model_providers.py \
  dashboard/backend/tests/domain/model_providers \
  dashboard/backend/tests/infrastructure/llm/adapters/test_validation_contract.py \
  dashboard/backend/tests/test_model_credentials_api.py \
  dashboard/backend/tests/test_admin_model_providers_api.py \
  dashboard/backend/tests/test_model_provider_store_postgres.py \
  dashboard/backend/tests/test_architecture_boundaries.py
git commit -m "feat(credentials): add bounded BYOK key vault"
```

### Task 2: Make Revocation, Defaults, and Seeds Durable and Safe

**Files:**
- Modify: `dashboard/backend/domain/model_providers/repository.py`
- Modify: `dashboard/backend/domain/model_providers/repository_postgres.py`
- Modify: `dashboard/backend/domain/model_providers/repository_common.py`
- Create: `dashboard/backend/tests/domain/model_providers/test_repository_contract.py`
- Modify: `dashboard/backend/tests/test_model_provider_store_postgres.py`

**Interfaces:**
- Consumes: `_encrypt(secret: str) -> str` and `_decrypt(token: str) -> str` from the broker repository.
- Produces: active credentials with encrypted secrets, revoked tombstones with `api_key_enc = NULL`, reusable revoked labels, stable seed configuration, and one verified default per user/provider.

- [ ] **Step 1: Write failing SQLite lifecycle tests**

Add repository-contract tests with fake secrets only:

```python
from cryptography.fernet import Fernet
import pytest

from dashboard.backend.domain.brokers import repository as broker_repository
from dashboard.backend.domain.model_providers.repository import ModelProviderStore
from dashboard.backend.domain.model_providers.repository_common import CredentialConflictError


@pytest.fixture(autouse=True)
def encryption_key(monkeypatch):
    monkeypatch.setenv("BROKER_TOKEN_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setattr(broker_repository, "_fernet_instance", None)


@pytest.fixture
def store(tmp_path):
    return ModelProviderStore(tmp_path / "provider-contract.db")


@pytest.fixture
def store_factory(tmp_path):
    database_path = tmp_path / "provider-reopen.db"
    return lambda: ModelProviderStore(database_path)


def test_revoke_crypto_shreds_secret_and_allows_label_reuse(store):
    created = store.create_user_credential(
        user_id=7,
        provider_id="openai",
        label="Research",
        secret="sk-fake-research-abcd",
        status="verified",
        set_default=True,
    )
    revoked = store.revoke_user_credential(7, created["credential_id"])
    assert revoked["status"] == "revoked"
    assert revoked["key_last_four"] == "abcd"
    with pytest.raises(CredentialConflictError):
        store.get_user_credential_secret(7, created["credential_id"])

    replacement = store.create_user_credential(
        user_id=7,
        provider_id="openai",
        label="Research",
        secret="sk-fake-replacement-wxyz",
    )
    assert replacement["credential_id"] != created["credential_id"]
```

```python
def test_seed_initialization_does_not_overwrite_admin_configuration(store_factory):
    store = store_factory()
    provider = store.get_provider("openai")
    store.upsert_provider(
        provider_id="openai",
        display_name="Approved OpenAI",
        adapter_type="openai",
        approved_base_url=provider["approved_base_url"],
        capabilities=provider["capabilities"],
        byok_enabled=False,
        platform_enabled=False,
        status="disabled",
    )
    reopened = store_factory()
    assert reopened.get_provider("openai")["display_name"] == "Approved OpenAI"
    assert reopened.get_provider("openai")["status"] == "disabled"
```

- [ ] **Step 2: Run the tests and verify the current defects**

Run:

```bash
python3 -m pytest \
  dashboard/backend/tests/domain/model_providers/test_repository_contract.py \
  -k "crypto_shreds or seed_initialization" -v
```

Expected: FAIL because ciphertext survives revocation, the label remains globally unique, and seed startup overwrites administrator-managed fields.

- [ ] **Step 3: Define tombstone-capable schema and partial uniqueness**

Use these invariants in both repository DDLs:

```sql
api_key_enc TEXT,
status TEXT NOT NULL CHECK (
    status IN ('verified', 'invalid', 'verification_unavailable', 'revoked')
),
CHECK (
    (status = 'revoked' AND api_key_enc IS NULL)
    OR (status <> 'revoked' AND api_key_enc IS NOT NULL)
)
```

Replace the all-row label unique constraint with a partial unique index:

```sql
CREATE UNIQUE INDEX IF NOT EXISTS idx_user_model_credentials_active_label
ON user_model_credentials(user_id, provider_id, label)
WHERE status <> 'revoked';
```

Keep the verified-default partial unique index scoped to `status = 'verified' AND is_default`.

Do not rely on `CREATE TABLE IF NOT EXISTS` to change an existing table. In SQLite, inspect `PRAGMA table_info(user_model_credentials)` and the table SQL; when `api_key_enc` is still `NOT NULL` or the all-row `UNIQUE(user_id, provider_id, label)` remains, rebuild the table inside one `BEGIN IMMEDIATE` transaction, copy active ciphertext unchanged, copy revoked rows with `api_key_enc = NULL`, recreate both partial indexes, and drop the legacy table. In PostgreSQL, run idempotent `ALTER TABLE ... ALTER COLUMN api_key_enc DROP NOT NULL`, drop the legacy generated label constraint, add the named tombstone check, and create the two partial indexes. Add reopen/migration tests that start from the exact legacy DDL and prove row counts, active decryption, revoked shredding, and default selection survive.

- [ ] **Step 4: Crypto-shred secrets on revoke**

Implement the state transition in both stores:

```sql
UPDATE user_model_credentials
SET api_key_enc = NULL,
    status = 'revoked',
    is_default = FALSE,
    updated_at = :updated_at
WHERE user_id = :user_id AND credential_id = :credential_id
```

Make `get_user_credential_secret` reject missing ciphertext or non-active status before calling `_decrypt`.

- [ ] **Step 5: Preserve seed configuration**

Change both seed loops to insert-only semantics:

```sql
INSERT INTO provider_registry (...)
VALUES (...)
ON CONFLICT(provider_id) DO NOTHING
```

- [ ] **Step 6: Run SQLite and PostgreSQL-shaped tests**

Run:

```bash
python3 -m pytest \
  dashboard/backend/tests/domain/model_providers/test_repository_contract.py \
  dashboard/backend/tests/test_model_provider_store_postgres.py::test_postgres_store_rejects_non_postgres_url_before_connecting -q
```

Expected: PASS; live PostgreSQL cases remain for Task 8 and must not be counted as verified here.

- [ ] **Step 7: Commit the lifecycle invariants**

Run:

```bash
git add dashboard/backend/domain/model_providers/repository.py \
  dashboard/backend/domain/model_providers/repository_postgres.py \
  dashboard/backend/domain/model_providers/repository_common.py \
  dashboard/backend/tests/domain/model_providers/test_repository_contract.py \
  dashboard/backend/tests/test_model_provider_store_postgres.py
git commit -m "fix(credentials): crypto-shred revoked API keys"
```

### Task 3: Pin Verification Requests to Public HTTPS Targets

**Files:**
- Create: `dashboard/backend/infrastructure/llm/adapters/safe_http.py`
- Modify: `dashboard/backend/infrastructure/llm/adapters/base.py`
- Modify: `dashboard/backend/infrastructure/llm/adapters/anthropic.py`
- Modify: `dashboard/backend/infrastructure/llm/adapters/gemini.py`
- Modify: `dashboard/backend/domain/model_providers/service.py`
- Create: `dashboard/backend/tests/infrastructure/llm/adapters/test_safe_http.py`
- Modify: `dashboard/backend/tests/infrastructure/llm/adapters/test_validation_contract.py`

**Interfaces:**
- Consumes: an approved HTTPS discovery URL and a resolver compatible with `socket.getaddrinfo`.
- Produces: `resolve_pinned_https_targets(url, resolver=socket.getaddrinfo) -> tuple[PinnedHttpsTarget, ...]`, where each target contains `request_url`, `host_header`, and `sni_hostname`.
- Extends: `ProviderAdapter.validate(..., resolver=socket.getaddrinfo)` and `ModelProviderService(..., resolver=socket.getaddrinfo)` so tests can inject deterministic DNS without weakening production validation.

- [ ] **Step 1: Write failing public-address tests**

Add this matrix to `test_safe_http.py`:

```python
@pytest.mark.parametrize(
    "address",
    ["127.0.0.1", "10.0.0.8", "169.254.169.254", "::1", "fc00::1"],
)
def test_private_or_metadata_addresses_are_rejected(address):
    def resolver(*_args, **_kwargs):
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (address, 443))]

    with pytest.raises(UnsafeProviderTargetError):
        resolve_pinned_https_targets(
            "https://models.example.test/v1/models",
            resolver=resolver,
        )
```

```python
def test_public_dns_result_is_pinned_with_original_tls_identity():
    def resolver(*_args, **_kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))]

    target = resolve_pinned_https_targets(
        "https://models.example.test/v1/models",
        resolver=resolver,
    )[0]
    assert target.request_url == "https://8.8.8.8/v1/models"
    assert target.host_header == "models.example.test"
    assert target.sni_hostname == "models.example.test"
```

- [ ] **Step 2: Run the target tests and verify failure**

Run:

```bash
python3 -m pytest dashboard/backend/tests/infrastructure/llm/adapters/test_safe_http.py -v
```

Expected: FAIL because `safe_http.py` and its public target types do not exist.

- [ ] **Step 3: Implement IP-pinned HTTPS targets**

Create the core immutable type and reject any non-public result:

```python
@dataclass(frozen=True)
class PinnedHttpsTarget:
    request_url: str
    host_header: str
    sni_hostname: str


def _is_public_address(value: str) -> bool:
    address = ipaddress.ip_address(value)
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )
```

Resolve once per validation request, reject the complete request when any DNS answer is unsafe, deduplicate public addresses, format IPv6 with brackets, and preserve the original hostname in TLS SNI. The `Host` value is the original hostname plus the explicit port when it is not 443.

- [ ] **Step 4: Send validation to the pinned IP**

Update `ProviderAdapter.validate` to call the pinned URL:

```python
response = client.get(
    target.request_url,
    headers={**headers, "Host": target.host_header},
    extensions={"sni_hostname": target.sni_hostname},
)
```

Owned clients must use `follow_redirects=False`, bounded timeouts, and `trust_env=False`. Try the next already-approved public target only after a connect/network failure; never re-resolve inside the attempt loop.

- [ ] **Step 5: Write strict discovery-response failures**

Add to `test_validation_contract.py`:

```python
def public_test_resolver(*_args, **_kwargs):
    return [
        (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("8.8.8.8", 443))
    ]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="<html>ok</html>"),
        httpx.Response(200, json={}),
        httpx.Response(200, json={"data": []}),
        httpx.Response(200, json={"data": [{"name": "missing-id"}]}),
    ],
)
def test_success_status_without_usable_models_is_not_verified(response):
    adapter = OpenAIAdapter()
    with httpx.Client(
        transport=httpx.MockTransport(lambda _request: response), trust_env=False
    ) as client:
        result = adapter.validate(
            "https://api.openai.test/v1",
            "sk-fake-invalid-shape",
            client=client,
            resolver=public_test_resolver,
        )
    assert result.status != "verified"
```

- [ ] **Step 6: Make parsing shape-aware**

`parse_models` must raise `InvalidDiscoveryResponseError` for a missing/wrong envelope and return a non-empty bounded list for success. `validate` maps malformed or empty successful responses to `verification_unavailable` with a fixed safe message; 401/403 remains `invalid`, 429/5xx remains `verification_unavailable`, and redirects remain `invalid`.

- [ ] **Step 7: Run adapter security tests**

Run:

```bash
python3 -m pytest \
  dashboard/backend/tests/infrastructure/llm/adapters/test_safe_http.py \
  dashboard/backend/tests/infrastructure/llm/adapters/test_validation_contract.py -q
```

Expected: PASS with no outbound network access.

- [ ] **Step 8: Commit the pinned verifier**

Run:

```bash
git add dashboard/backend/infrastructure/llm/adapters \
  dashboard/backend/domain/model_providers/service.py \
  dashboard/backend/tests/infrastructure/llm/adapters
git commit -m "fix(credentials): pin provider verification to public targets"
```

### Task 4: Make Admin Mutations and Audit Records Atomic

**Files:**
- Modify: `dashboard/backend/domain/model_providers/models.py`
- Modify: `dashboard/backend/domain/model_providers/repository_common.py`
- Modify: `dashboard/backend/domain/model_providers/repository.py`
- Modify: `dashboard/backend/domain/model_providers/repository_postgres.py`
- Modify: `dashboard/backend/domain/model_providers/service.py`
- Modify: `dashboard/backend/tests/domain/model_providers/test_admin_service.py`
- Modify: `dashboard/backend/tests/domain/model_providers/test_repository_contract.py`
- Modify: `dashboard/backend/tests/test_model_provider_store_postgres.py`

**Interfaces:**
- Consumes: canonical admin request fields and, for secret-bearing requests, an HMAC fingerprint derived from the configured Fernet key.
- Produces: atomic `upsert_provider_admin`, `upsert_platform_credential_admin`, `set_platform_credential_status_admin`, and `revoke_platform_credential_admin` repository operations returning `(public_record, replayed)`; exact replays return the original safe public result snapshot.

- [ ] **Step 1: Write failing transaction and replay tests**

```python
def test_admin_provider_mutation_rolls_back_when_audit_insert_fails(store, monkeypatch):
    before = store.get_provider("openai")
    monkeypatch.setattr(store, "_insert_admin_operation", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("audit failed")))
    with pytest.raises(RuntimeError, match="audit failed"):
        store.upsert_provider_admin(
            actor_user_id=9,
            provider_id="openai",
            display_name="Changed",
            adapter_type="openai",
            approved_base_url=before["approved_base_url"],
            capabilities=before["capabilities"],
            byok_enabled=True,
            platform_enabled=False,
            status="enabled",
            source="admin_console",
            reason="Atomicity test",
            idempotency_key="provider-atomic-001",
            request_digest="digest-a",
        )
    assert store.get_provider("openai")["display_name"] == before["display_name"]
```

```python
def test_admin_idempotency_key_rejects_changed_payload():
    store = FakeAdminStore()
    service = _service(store, FakeAdapter())
    service.upsert_provider(
        9,
        "approved_vendor",
        _provider_request(
            display_name="Vendor A", idempotency_key="provider-replay-001"
        ),
    )
    with pytest.raises(CredentialConflictError):
        service.upsert_provider(
            9,
            "approved_vendor",
            _provider_request(
                display_name="Vendor B", idempotency_key="provider-replay-001"
            ),
        )
```

Refactor `FakeAdminStore` in the same test file to expose the new atomic repository methods and store `request_digest` in each fake audit row. Keep `_provider_request`, `_service`, and `FakeAdapter` as the concrete helpers shown earlier in that file.

- [ ] **Step 2: Run the tests and verify non-atomic behavior**

Run:

```bash
python3 -m pytest \
  dashboard/backend/tests/domain/model_providers/test_admin_service.py \
  dashboard/backend/tests/domain/model_providers/test_repository_contract.py \
  -k "atomic or idempotency" -v
```

Expected: FAIL because service mutations and `record_admin_operation` currently use separate transactions and replay does not bind request content.

- [ ] **Step 3: Add request digests and safe result snapshots to audit storage**

Add non-secret `request_digest TEXT NOT NULL` and `result_json TEXT NOT NULL` columns to `model_provider_admin_operations`. `result_json` contains only the same safe public projection returned by the API, never ciphertext or full secret. Build canonical request JSON with sorted keys and compact separators, then hash it with SHA-256. For a platform secret, include only this keyed fingerprint in the canonical payload:

```python
def secret_fingerprint(secret: str, encryption_key: bytes) -> str:
    return hmac.new(encryption_key, secret.encode("utf-8"), hashlib.sha256).hexdigest()
```

Decode the URL-safe Fernet key before using it as the HMAC key. Never persist or log the canonical secret-bearing input.

Migrate existing audit rows explicitly: add both columns as nullable, fill `request_digest` with a deterministic `legacy:` digest of the existing non-secret audit fields and `result_json` with `{}`, then make both columns non-null. A new request that meets a legacy idempotency row is a conflict because its canonical digest cannot equal the `legacy:` digest.

- [ ] **Step 4: Move each admin mutation and audit insert into one transaction**

For SQLite, execute the resource mutation and audit insert inside one `with connection:` block. For PostgreSQL, use one transaction, lock the idempotency row/key before mutation, and let a unique-key conflict roll back the resource mutation.

Replay behavior is exact:

```python
if existing_operation:
    if existing_operation["request_digest"] != request_digest:
        raise CredentialConflictError("idempotency key already used")
    return deserialize_public_result(existing_operation["result_json"]), True
```

- [ ] **Step 5: Verify platform credential sequencing**

`set_platform_credential` validates first, then atomically stores encrypted secret, final validation status, and audit record. `reverify_platform_credential` reads/decrypts the active secret, validates, then atomically updates status plus audit. `revoke_platform_credential` atomically nulls ciphertext, marks the tombstone revoked, and writes audit evidence.

- [ ] **Step 6: Run service and store tests**

Run:

```bash
python3 -m pytest \
  dashboard/backend/tests/domain/model_providers/test_admin_service.py \
  dashboard/backend/tests/domain/model_providers/test_repository_contract.py \
  dashboard/backend/tests/test_model_provider_store_postgres.py::test_postgres_store_rejects_non_postgres_url_before_connecting -q
```

Expected: PASS without a live database skip being interpreted as PostgreSQL verification.

- [ ] **Step 7: Commit atomic administration**

Run:

```bash
git add dashboard/backend/domain/model_providers \
  dashboard/backend/tests/domain/model_providers \
  dashboard/backend/tests/test_model_provider_store_postgres.py
git commit -m "fix(providers): make admin mutations auditable and atomic"
```

### Task 5: Harden User and Admin API Boundaries

**Files:**
- Modify: `dashboard/backend/api/routers/model_credentials.py`
- Modify: `dashboard/backend/api/routers/admin_model_providers.py`
- Modify: `dashboard/backend/api/router.py`
- Modify: `dashboard/backend/tests/test_model_credentials_api.py`
- Modify: `dashboard/backend/tests/test_admin_model_providers_api.py`
- Modify: `dashboard/backend/tests/test_app_composition.py`
- Modify: `dashboard/backend/tests/test_csrf.py`
- Modify: `dashboard/backend/tests/test_store_twin_parity.py`

**Interfaces:**
- Consumes: `ModelProviderService` safe public results and existing `get_current_user`/`require_admin` dependencies.
- Produces: `/api/credits/model-providers`, `/api/credits/api-keys`, and `/api/admin/model-providers` route families with fixed safe errors.

- [ ] **Step 1: Write failing secret-canary API tests**

```python
def test_rejected_create_body_never_echoes_the_key(credential_api):
    token = _signup(credential_api.client, "canary-api@example.com")
    canary = "sk-fake-never-echo-this-value"
    response = credential_api.client.post(
        "/api/credits/api-keys",
        headers=_auth(token),
        json={"provider_id": "openai", "label": "", "api_key": canary},
    )
    assert response.status_code == 422
    assert canary not in response.text
```

```python
def test_user_cannot_mutate_another_users_credential(credential_api):
    owner_token = _signup(credential_api.client, "owner-api@example.com")
    other_token = _signup(credential_api.client, "other-api@example.com")
    credential_api.adapter.queue("verified")
    credential_id = _create(credential_api, owner_token).json()["credential"][
        "credential_id"
    ]

    verify = credential_api.client.post(
        f"/api/credits/api-keys/{credential_id}/verify",
        headers=_auth(other_token),
    )
    revoke = credential_api.client.delete(
        f"/api/credits/api-keys/{credential_id}",
        headers=_auth(other_token),
    )
    assert verify.status_code == 404
    assert revoke.status_code == 404
```

- [ ] **Step 2: Run API tests and record failures**

Run:

```bash
python3 -m pytest \
  dashboard/backend/tests/test_model_credentials_api.py \
  dashboard/backend/tests/test_admin_model_providers_api.py -v
```

Expected: new canary, tombstone, atomic replay, and cross-owner cases fail until the routers and service mapping are complete.

- [ ] **Step 3: Keep secret-bearing Pydantic errors inside the boundary**

Continue manual bounded body parsing for user and platform credential creation. Keep the body limit at 16 KiB. Collapse JSON, Unicode, and `ValidationError` failures to fixed messages and never serialize `ValidationError.errors()`.

User public responses contain exactly:

```python
{
    "credential_id",
    "provider_id",
    "label",
    "key_last_four",
    "status",
    "is_default",
    "created_at",
    "updated_at",
    "last_verified_at",
}
```

- [ ] **Step 4: Keep admin authorization separate and singular**

Use one router-level `Depends(require_admin)` authorization boundary and obtain the cached admin identity in mutation endpoints without performing a second independent authorization lookup. Keep user APIs under `get_current_user`; never return user credentials through admin provider routes.

- [ ] **Step 5: Update the frozen route and CSRF contracts**

Add only these intentional route families to `EXPECTED_FULL_CONTRACT` and CSRF mutation coverage:

```text
GET    /api/credits/model-providers
GET    /api/credits/api-keys
POST   /api/credits/api-keys
POST   /api/credits/api-keys/{credential_id}/verify
POST   /api/credits/api-keys/{credential_id}/default
DELETE /api/credits/api-keys/{credential_id}
GET    /api/admin/model-providers
PUT    /api/admin/model-providers/{provider_id}
PUT    /api/admin/model-providers/{provider_id}/platform-credential
POST   /api/admin/model-providers/{provider_id}/platform-credential/verify
DELETE /api/admin/model-providers/{provider_id}/platform-credential
```

Assert that no `/api/model-execution` route exists in the PR.

- [ ] **Step 6: Run the API and architecture gates**

Run:

```bash
python3 -m pytest \
  dashboard/backend/tests/test_model_credentials_api.py \
  dashboard/backend/tests/test_admin_model_providers_api.py \
  dashboard/backend/tests/test_app_composition.py \
  dashboard/backend/tests/test_csrf.py \
  dashboard/backend/tests/test_store_twin_parity.py \
  dashboard/backend/tests/test_architecture_boundaries.py -q
```

Expected: PASS with no extra route-contract entries.

- [ ] **Step 7: Commit the API boundary**

Run:

```bash
git add dashboard/backend/api \
  dashboard/backend/tests/test_model_credentials_api.py \
  dashboard/backend/tests/test_admin_model_providers_api.py \
  dashboard/backend/tests/test_app_composition.py \
  dashboard/backend/tests/test_csrf.py \
  dashboard/backend/tests/test_store_twin_parity.py \
  dashboard/backend/tests/test_architecture_boundaries.py
git commit -m "feat(credentials): expose safe BYOK vault APIs"
```

### Task 6: Add the User API Keys Tab Without Secret Retention

**Files:**
- Modify: `dashboard/frontend/app.html`
- Modify: `dashboard/frontend/js/credits.js`
- Modify: `dashboard/frontend/styles.css`
- Modify: `dashboard/backend/tests/test_credits_frontend.py`
- Create: `dashboard/backend/tests/test_model_credentials_frontend.py`
- Modify: `dashboard/backend/tests/test_frontend_fast_boot.py`

**Interfaces:**
- Consumes: the user provider and credential APIs from Task 5 and `window.API.request`.
- Produces: the complete four-tab Credits & Billing set and user actions for create, verify, default, and revoke.

- [ ] **Step 1: Write failing tab and secret-lifetime tests**

```python
def test_credits_billing_has_four_tabs_in_required_order():
    tabs = re.findall(r'data-credits-tab="([^"]+)"', APP_HTML)
    assert tabs == ["overview", "topup", "api-keys", "activity"]
```

```python
def test_api_key_secret_is_not_put_in_url_or_browser_storage():
    start = CREDITS_JS.index("async function submitApiKey")
    end = CREDITS_JS.index("async function mutateApiKey")
    submit_source = CREDITS_JS[start:end]
    assert "localStorage" not in submit_source
    assert "sessionStorage" not in submit_source
    assert "URLSearchParams" not in submit_source
    assert "api_key: secret" in submit_source
    assert "secretInput.value = ''" in submit_source
```

Add `import re` to `test_model_credentials_frontend.py`; reuse its existing `APP_HTML` and `CREDITS_JS` constants.

- [ ] **Step 2: Run frontend contracts and verify failure**

Run:

```bash
python3 -m pytest \
  dashboard/backend/tests/test_credits_frontend.py \
  dashboard/backend/tests/test_model_credentials_frontend.py \
  dashboard/backend/tests/test_frontend_fast_boot.py -v
```

Expected: FAIL because the clean main branch does not yet contain the complete four-tab set or Vault controls.

- [ ] **Step 3: Add only the user Vault markup and behavior**

The tab order is fixed:

```html
<button type="button" data-credits-tab="overview">Overview</button>
<button type="button" data-credits-tab="topup">Top up</button>
<button type="button" data-credits-tab="api-keys">API Keys</button>
<button type="button" data-credits-tab="activity">Activity</button>
```

The create form contains provider, label, password-type secret, and set-default controls. The list renders provider display name, label, status, default badge, last verification time, and `.... ${key_last_four}` only.

- [ ] **Step 4: Bound the secret to one submit operation**

Use a local variable and clear every reference immediately after serialization and again in `finally`:

```javascript
let secret = secretInput.value;
let requestBody = JSON.stringify({ provider_id: providerId, label, api_key: secret, set_default });
secret = '';
secretInput.value = '';
try {
  await apiRequest('/api/credits/api-keys', { method: 'POST', body: requestBody });
} finally {
  requestBody = '';
  secretInput.value = '';
}
```

Do not add the secret to module `state`, URLs, logs, analytics, browser storage, or DOM text.

- [ ] **Step 5: Preserve truthful Credits spending copy**

PR 1 does not enable model spending. Preserve the latest-main copy stating that spending is not enabled; do not use `Ready for eligible ATL-funded model runs` in this PR.

- [ ] **Step 6: Run frontend contracts**

Run:

```bash
python3 -m pytest \
  dashboard/backend/tests/test_credits_frontend.py \
  dashboard/backend/tests/test_model_credentials_frontend.py \
  dashboard/backend/tests/test_frontend_fast_boot.py -q
```

Expected: PASS, including fast-boot behavior when the Credits page module is unavailable.

- [ ] **Step 7: Commit the user UI**

Run:

```bash
git add dashboard/frontend/app.html \
  dashboard/frontend/js/credits.js \
  dashboard/frontend/styles.css \
  dashboard/backend/tests/test_credits_frontend.py \
  dashboard/backend/tests/test_model_credentials_frontend.py \
  dashboard/backend/tests/test_frontend_fast_boot.py
git commit -m "feat(frontend): add API Keys billing tab"
```

### Task 7: Add a Separate Admin Provider Surface

**Files:**
- Create: `dashboard/frontend/js/admin-model-providers.js`
- Modify: `dashboard/frontend/app.html`
- Modify: `dashboard/frontend/app.js`
- Modify: `dashboard/frontend/styles.css`
- Create: `dashboard/backend/tests/test_admin_model_providers_frontend.py`
- Modify: `dashboard/backend/tests/test_frontend_fast_boot.py`

**Interfaces:**
- Consumes: the admin provider APIs from Task 5 and the existing Admin view lifecycle.
- Produces: `window.AdminModelProviders.syncAuth(user)` and `window.AdminModelProviders.onEnter(user)` without sharing state with `CreditsPage`.

- [ ] **Step 1: Write failing admin separation tests**

```python
from pathlib import Path


FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
APP_HTML = (FRONTEND / "app.html").read_text(encoding="utf-8")
ADMIN_PROVIDER_JS = (FRONTEND / "js" / "admin-model-providers.js").read_text(
    encoding="utf-8"
)


def api_keys_panel_html() -> str:
    start = APP_HTML.index('data-credits-panel="api-keys"')
    end = APP_HTML.index('data-credits-panel="activity"', start)
    return APP_HTML[start:end]


def test_admin_provider_controls_are_not_inside_user_api_keys_panel():
    api_keys_panel = api_keys_panel_html()
    assert "platform credential" not in api_keys_panel.lower()
    assert 'id="adminModelProviders"' in APP_HTML
```

```python
def test_admin_module_never_persists_platform_secret():
    assert "localStorage" not in ADMIN_PROVIDER_JS
    assert "sessionStorage" not in ADMIN_PROVIDER_JS
    assert "platformSecretInput.value = ''" in ADMIN_PROVIDER_JS
```

- [ ] **Step 2: Run the admin frontend tests and verify failure**

Run:

```bash
python3 -m pytest dashboard/backend/tests/test_admin_model_providers_frontend.py -v
```

Expected: FAIL because the separate admin provider module and markup do not exist.

- [ ] **Step 3: Define the admin module boundary**

Create a private module state containing public provider metadata only:

```javascript
const state = { user: null, providers: [], initialized: false };

function syncAuth(user) {
  state.user = user?.role === 'admin' ? user : null;
  if (!state.user) state.providers = [];
}

window.AdminModelProviders = { syncAuth, onEnter };
```

The Admin view supports provider create/update, enable/disable, platform credential set, reverify, and revoke. It displays last four and status only. Every mutation requires source, reason, and a fresh `crypto.randomUUID()` idempotency key.

- [ ] **Step 4: Clear platform secret input before awaiting network**

Use the same bounded local-variable pattern as the user form. Do not pass platform secrets through `app.js`; `admin-model-providers.js` owns the form and request.

- [ ] **Step 5: Integrate without making app boot depend on the module**

Call the module defensively from auth and Admin navigation paths:

```javascript
window.AdminModelProviders?.syncAuth(currentUser);
window.AdminModelProviders?.onEnter(currentUser);
```

Load the script from `app.html` as a normal local asset. `app.js` must continue booting when the module is absent in the fast-boot test.

- [ ] **Step 6: Run admin and fast-boot frontend contracts**

Run:

```bash
python3 -m pytest \
  dashboard/backend/tests/test_admin_model_providers_frontend.py \
  dashboard/backend/tests/test_frontend_fast_boot.py -q
```

Expected: PASS with the user API Keys panel free of admin controls.

- [ ] **Step 7: Commit the separate admin UI**

Run:

```bash
git add dashboard/frontend/js/admin-model-providers.js \
  dashboard/frontend/app.html \
  dashboard/frontend/app.js \
  dashboard/frontend/styles.css \
  dashboard/backend/tests/test_admin_model_providers_frontend.py \
  dashboard/backend/tests/test_frontend_fast_boot.py
git commit -m "feat(admin): add model provider management surface"
```

### Task 8: Prove PostgreSQL, Browser, and Full-Suite Delivery Gates

**Files:**
- Modify: `dashboard/backend/tests/test_model_provider_store_postgres.py`
- Create: `dashboard/backend/tests/e2e/model_credentials_harness.py`
- Modify only if a real defect is found: PR 1 files listed above
- Create at PR preparation time: no local artifact; PR description remains in GitHub, not the repository

**Interfaces:**
- Consumes: completed PR 1 implementation, a dedicated disposable PostgreSQL database, and the in-app browser.
- Produces: zero-failure test evidence, browser proof using fake secrets, and an English PR description.

- [ ] **Step 1: Run formatting and focused tests**

Run:

```bash
git diff --check
git diff --cached --check
git diff --check origin/main..HEAD
python3 -m pytest \
  dashboard/backend/tests/domain/model_providers \
  dashboard/backend/tests/infrastructure/llm/adapters \
  dashboard/backend/tests/test_model_credentials_api.py \
  dashboard/backend/tests/test_admin_model_providers_api.py \
  dashboard/backend/tests/test_model_credentials_frontend.py \
  dashboard/backend/tests/test_admin_model_providers_frontend.py -q -ra
```

Expected: zero failures and no skipped SQLite, service, API, adapter, or frontend Vault test.

- [ ] **Step 2: Run the live PostgreSQL contract on a disposable database**

The database must contain no non-test data because the fixture drops the four model-provider tables.

Run with the dedicated URL supplied by the local PostgreSQL service or CI service container:

```bash
TEST_POSTGRES_URL=postgresql://postgres:postgres@127.0.0.1:5432/atl_test \
python3 -m pytest dashboard/backend/tests/test_model_provider_store_postgres.py -q -ra
```

Expected: every PostgreSQL Vault test passes; none is skipped. If no disposable PostgreSQL service is available, stop this gate and obtain one rather than reporting the PR ready.

- [ ] **Step 3: Create the controlled browser harness**

The harness imports the production app with a temporary `DATABASE_PATH`, then overrides the canonical service dependency created in Task 1:

```python
import httpx

from dashboard.backend.app import app
from dashboard.backend.domain.model_providers.repository import ModelProviderStore
from dashboard.backend.domain.model_providers.service import (
    ModelProviderService,
    get_model_provider_service,
)


def public_test_resolver(*_args, **_kwargs):
    return [
        (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("8.8.8.8", 443))
    ]


validation_client = httpx.Client(
    transport=httpx.MockTransport(provider_handler),
    follow_redirects=False,
    trust_env=False,
)
browser_service = ModelProviderService(
    store=ModelProviderStore(),
    http_client=validation_client,
    resolver=public_test_resolver,
)
app.dependency_overrides[get_model_provider_service] = lambda: browser_service
```

Define `provider_handler` above `validation_client`; register a shutdown handler that closes `validation_client`. The fake provider returns:

```python
def provider_handler(request: httpx.Request) -> httpx.Response:
    if request.headers.get("authorization") != "Bearer sk-fake-browser-valid-abcd":
        return httpx.Response(401, json={"error": "invalid credential"})
    return httpx.Response(200, json={"data": [{"id": "fake/model-v1"}]})
```

The harness is test-only and does not add a production environment flag that permits private origins.

- [ ] **Step 4: Run the local app and browser workflow**

Start the controlled server on the existing local port contract and a new disposable database:

```bash
atl_byok_browser_dir=$(mktemp -d /tmp/atl-byok-browser.XXXXXX)
DATABASE_PATH="$atl_byok_browser_dir/backtest.db" \
BROKER_TOKEN_ENCRYPTION_KEY=MDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA= \
SESSION_HASH_SECRET=test-session-hash-secret-for-byok-browser \
ADMIN_BOOTSTRAP_SECRET=test-admin-bootstrap-secret \
python3 -m uvicorn dashboard.backend.tests.e2e.model_credentials_harness:app \
  --host 127.0.0.1 --port 8766
```

In the in-app browser, verify this exact user flow with fake secrets only:

1. sign up the disposable harness user, then promote it once through `POST /api/admin/bootstrap` using the configured fake bootstrap secret;
2. open Credits & Billing and confirm the four-tab order;
3. create two named keys for one provider;
4. observe only the last four characters in the list and network response;
5. set the second verified key as default;
6. reverify the first key;
7. revoke the default key and query the disposable SQLite row to confirm `api_key_enc IS NULL`;
8. recreate the revoked label successfully;
9. confirm the full fake key is absent from the URL, console, local/session storage, and subsequent API responses; and
10. confirm admin provider controls appear only in Admin.

- [ ] **Step 5: Run the complete backend suite with isolated local storage**

Run:

```bash
audit_db_dir=$(mktemp -d /tmp/atl-byok-full.XXXXXX)
audit_pycache_dir=$(mktemp -d /tmp/atl-byok-pycache.XXXXXX)
DATABASE_PATH="$audit_db_dir/backtest.db" \
PYTHONPYCACHEPREFIX="$audit_pycache_dir" \
SESSION_HASH_SECRET=test-session-hash-secret-for-byok-full-suite \
BROKER_TOKEN_ENCRYPTION_KEY=MDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA= \
python3 -m pytest dashboard/backend/tests -q -ra
```

Expected: zero failures. Record the exact passed/skipped counts and every skip reason. The three Model Provider PostgreSQL tests must already have passed in Step 2 and must not be represented as verified by this SQLite run.

- [ ] **Step 6: Verify the final diff and forbidden paths**

Run:

```bash
git status --short
git diff --name-only origin/main..HEAD | rg 'model_execution|provider_executor|routed_client|backtests.py' && exit 1 || true
git diff --name-only --cached
git log --oneline origin/main..HEAD
```

Expected: no local database, `.superpowers/`, `work/`, execution runtime, or staged leftover appears.

- [ ] **Step 7: Commit the browser harness if it is not already committed**

Run:

```bash
git add dashboard/backend/tests/e2e/model_credentials_harness.py \
  dashboard/backend/tests/test_model_provider_store_postgres.py
git commit -m "test(credentials): prove Vault delivery contracts"
```

- [ ] **Step 8: Prepare the English PR description**

Use this structure. After each test label, copy the exact pytest summary produced by the named step; do not estimate or reuse an older run:

```markdown
## Summary
- add encrypted user and platform credential vaults for approved model providers
- add strict no-cost key verification with public-IP-pinned HTTPS requests
- add separate user API Keys and admin provider management surfaces

## Security invariants
- full credentials never leave secret-bearing request boundaries
- missing encryption configuration fails closed
- revocation destroys ciphertext and preserves safe audit tombstones
- admin mutations and audit evidence commit atomically

## Tests
- focused Vault suite: exact Task 8 Step 1 pytest summary
- live PostgreSQL Vault contract: exact Task 8 Step 2 pytest summary, with 0 skipped
- full backend suite: exact Task 8 Step 5 pytest summary, with 0 failed
- controlled browser workflow: passed with fake credentials only

## Out of scope
- BYOK model execution
- Admin Grant Credits
- purchased-Credits model spending
```

Do not open the PR while any gate is red or while the working tree contains the tracked local database modification.
