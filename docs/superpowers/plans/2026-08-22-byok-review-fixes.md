# BYOK Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the two independent security/consistency findings on PR #396 and repair its shallow-checkout CI failure.

**Architecture:** Keep proxy enforcement inside the outbound adapter boundary and make proxy eligibility an explicit property of each native adapter. Validate credentials before persistence, but preflight the existing Fernet configuration first; then pass the final verification state into the repository's existing single create transaction. Replace the remote-ref-dependent architecture test with a checkout-depth-independent local Git inventory plus ignored-runtime filesystem assertion.

**Tech Stack:** Python 3.13, FastAPI domain services, httpx/httpcore, SQLite, PostgreSQL/psycopg, cryptography/Fernet, pytest.

**Spec:** `docs/superpowers/specs/2026-08-22-byok-review-fixes-design.md`

## Global Constraints

- Never print, log, return, or commit a real API key or encryption key.
- `BROKER_TOKEN_ENCRYPTION_KEY` must remain fail-closed.
- Only OpenRouter, OpenAI, Anthropic, and Gemini official HTTPS origins may use `BROKER_CREDENTIAL_VERIFICATION_PROXY`.
- Every custom origin, including OpenAI-compatible providers, must use DNS validation and IP pinning.
- Provider validation must not hold a database transaction open.
- SQLite and Postgres must preserve one verified default credential per user and provider.
- Do not add model execution to this PR.
- Do not stage `dashboard/storage/data/backtest.db`, `.superpowers/`, `AGENTS.md`, or `work/`.

---

### Task 1: Restrict Explicit Proxy Use to Official Native Origins

**Files:**
- Modify: `dashboard/backend/infrastructure/llm/adapters/base.py`
- Modify: `dashboard/backend/infrastructure/llm/adapters/openrouter.py`
- Modify: `dashboard/backend/infrastructure/llm/adapters/openai.py`
- Modify: `dashboard/backend/infrastructure/llm/adapters/anthropic.py`
- Modify: `dashboard/backend/infrastructure/llm/adapters/gemini.py`
- Test: `dashboard/backend/tests/infrastructure/llm/adapters/test_validation_contract.py`

**Interfaces:**
- Consumes: `build_explicit_proxy_transport(proxy_url)` and `build_pinned_transport(url)`.
- Produces: `ProviderAdapter.proxy_origin: str | None` and an origin-equality gate inside `ProviderAdapter._transport(url)`.

- [ ] **Step 1: Write failing regression tests for all proxy decisions**

Add parameterized tests that patch both transport builders and assert all four native adapters proxy only their official origins:

```python
@pytest.mark.parametrize(
    ("adapter_type", "official_url"),
    [
        ("openrouter", "https://openrouter.ai/api/v1/key"),
        ("openai", "https://api.openai.com/v1/models"),
        ("anthropic", "https://api.anthropic.com/v1/models"),
        ("gemini", "https://generativelanguage.googleapis.com/v1beta/models"),
    ],
)
def test_explicit_proxy_is_limited_to_native_official_origins(
    monkeypatch, adapter_type, official_url
):
    monkeypatch.setenv(
        "BROKER_CREDENTIAL_VERIFICATION_PROXY", "http://127.0.0.1:7897"
    )
    monkeypatch.setattr(
        adapter_base,
        "build_explicit_proxy_transport",
        lambda proxy: ("proxy", proxy),
    )
    monkeypatch.setattr(
        adapter_base,
        "build_pinned_transport",
        lambda url: ("pinned", url),
    )

    assert get_adapter(adapter_type)._transport(official_url)[0] == "proxy"
    assert get_adapter(adapter_type)._transport(
        "https://custom-provider.example/v1/models"
    )[0] == "pinned"


def test_openai_compatible_never_uses_explicit_proxy(monkeypatch):
    monkeypatch.setenv(
        "BROKER_CREDENTIAL_VERIFICATION_PROXY", "http://127.0.0.1:7897"
    )
    monkeypatch.setattr(
        adapter_base,
        "build_explicit_proxy_transport",
        lambda proxy: ("proxy", proxy),
    )
    monkeypatch.setattr(
        adapter_base,
        "build_pinned_transport",
        lambda url: ("pinned", url),
    )

    transport = get_adapter("openai_compatible")._transport(
        "https://models.example.com/v1/models"
    )
    assert transport[0] == "pinned"
```

- [ ] **Step 2: Run the proxy regression tests and verify the custom-native case fails**

Run:

```bash
pytest dashboard/backend/tests/infrastructure/llm/adapters/test_validation_contract.py -q
```

Expected: the new custom-origin assertion fails because the current code proxies every non-`openai_compatible` adapter.

- [ ] **Step 3: Add explicit official-origin metadata and exact origin comparison**

Extend `ProviderAdapter` with `proxy_origin: str | None = None`. Add a private helper in `base.py` that uses `urllib.parse.urlsplit`, requires HTTPS, rejects userinfo/query/fragment, normalizes the hostname with IDNA, and returns `(host, effective_port)` or `None`.

Implement the transport decision as:

```python
proxy = (os.getenv("BROKER_CREDENTIAL_VERIFICATION_PROXY") or "").strip()
if (
    proxy
    and self.proxy_origin
    and _https_origin(url) == _https_origin(self.proxy_origin)
):
    return build_explicit_proxy_transport(proxy)
return build_pinned_transport(url)
```

Pass these official origins from the four native adapters:

```python
"https://openrouter.ai"
"https://api.openai.com"
"https://api.anthropic.com"
"https://generativelanguage.googleapis.com"
```

Leave the registry's `openai_compatible` adapter without `proxy_origin`.

- [ ] **Step 4: Run adapter safety tests**

Run:

```bash
pytest \
  dashboard/backend/tests/infrastructure/llm/adapters/test_validation_contract.py \
  dashboard/backend/tests/infrastructure/llm/adapters/test_safe_http.py -q
```

Expected: all tests pass without making a real provider request.

- [ ] **Step 5: Commit the proxy hardening**

```bash
git add \
  dashboard/backend/infrastructure/llm/adapters/base.py \
  dashboard/backend/infrastructure/llm/adapters/openrouter.py \
  dashboard/backend/infrastructure/llm/adapters/openai.py \
  dashboard/backend/infrastructure/llm/adapters/anthropic.py \
  dashboard/backend/infrastructure/llm/adapters/gemini.py \
  dashboard/backend/tests/infrastructure/llm/adapters/test_validation_contract.py
git commit -m "fix(byok): restrict verification proxy origins"
```

---

### Task 2: Persist New Credentials Once with Their Final State

**Files:**
- Modify: `dashboard/backend/domain/model_providers/repository.py`
- Modify: `dashboard/backend/domain/model_providers/service.py`
- Test: `dashboard/backend/tests/domain/model_providers/test_service.py`
- Verify: `dashboard/backend/tests/domain/model_providers/test_repository_contract.py`
- Verify: `dashboard/backend/tests/test_model_provider_store_postgres.py`

**Interfaces:**
- Consumes: the existing broker Fernet factory and `create_user_credential(..., status, verification_message, set_default, last_verified_at)` transaction contract.
- Produces: `ensure_credential_encryption_ready() -> None`; `ModelProviderService.create_credential` performs exactly one repository write after validation.

- [ ] **Step 1: Write failing service tests for single-write creation**

Strengthen the missing-key test with `assert adapter.calls == []`. Add a test that replaces the two post-create mutation methods with failures and captures the final create arguments:

```python
def test_create_persists_final_state_without_follow_up_mutations(tmp_path, monkeypatch):
    adapter = FakeAdapter(_validation("verified"))
    service, store = _service(tmp_path, adapter)
    original_create = store.create_user_credential
    create_calls = []

    def capture_create(**kwargs):
        create_calls.append(kwargs)
        return original_create(**kwargs)

    def unexpected_mutation(*_args, **_kwargs):
        raise AssertionError("credential creation must not perform a second write")

    monkeypatch.setattr(store, "create_user_credential", capture_create)
    monkeypatch.setattr(store, "set_user_credential_status", unexpected_mutation)
    monkeypatch.setattr(store, "set_default_user_credential", unexpected_mutation)

    created = service.create_credential(7, _request(set_default=True))

    assert created.status == "verified"
    assert created.is_default is True
    assert len(create_calls) == 1
    assert create_calls[0]["status"] == "verified"
    assert create_calls[0]["set_default"] is True
    assert create_calls[0]["last_verified_at"] is not None
```

Also parameterize the existing non-verified outcome test to confirm `set_default=True` is passed safely but the repository returns `is_default=False` when status is not verified.

- [ ] **Step 2: Run the service tests and verify the second-write assertion fails**

Run:

```bash
pytest dashboard/backend/tests/domain/model_providers/test_service.py -q
```

Expected: `test_create_persists_final_state_without_follow_up_mutations` fails because current creation calls `set_user_credential_status` after inserting the placeholder row.

- [ ] **Step 3: Add encryption preflight and final-state creation**

In `repository.py`, expose a model-provider-scoped wrapper around the existing Fernet factory:

```python
def ensure_credential_encryption_ready() -> None:
    _get_fernet()
```

Import `_get_fernet` beside the existing `_encrypt` and `_decrypt` imports. In `service.py`, call the wrapper after resolving the provider and before adapter validation.

Extract or reuse one adapter-validation helper that collapses exceptions into the fixed `verification_unavailable` result. Then replace placeholder creation plus `_verify_and_update` with one call:

```python
validation = self._validate_credential(provider, secret)
created = self.store.create_user_credential(
    user_id=user_id,
    provider_id=provider["provider_id"],
    label=request.label,
    secret=secret,
    status=validation.status,
    verification_message=_safe_verification_message(validation),
    set_default=request.set_default,
    last_verified_at=_utcnow_iso() if validation.status == "verified" else None,
)
return UserCredentialPublic.model_validate(created)
```

Keep `_verify_and_update` for reverification, which intentionally mutates an existing credential.

- [ ] **Step 4: Run SQLite service and repository contracts**

Run:

```bash
pytest \
  dashboard/backend/tests/domain/model_providers/test_service.py \
  dashboard/backend/tests/domain/model_providers/test_repository_contract.py \
  dashboard/backend/tests/test_model_credentials_api.py -q
```

Expected: all tests pass; no real key or network request is used.

- [ ] **Step 5: Run the Postgres twin contract when available**

Run:

```bash
pytest dashboard/backend/tests/test_model_provider_store_postgres.py -q
```

Expected locally: tests pass when `TEST_POSTGRES_URL` is configured; otherwise Postgres cases are skipped and will execute in CI.

- [ ] **Step 6: Commit atomic creation**

```bash
git add \
  dashboard/backend/domain/model_providers/repository.py \
  dashboard/backend/domain/model_providers/service.py \
  dashboard/backend/tests/domain/model_providers/test_service.py
git commit -m "fix(byok): persist verified credentials atomically"
```

---

### Task 3: Make the Architecture Guard Independent of Git Fetch Depth

**Files:**
- Modify: `dashboard/backend/tests/test_architecture_boundaries.py`

**Interfaces:**
- Consumes: `_REPO_ROOT: pathlib.Path`.
- Produces: a deterministic assertion that forbidden model-execution files are absent from both local Git inventory and ignored importable runtime artifacts.

- [ ] **Step 1: Replace the remote-ref diff with a local Git inventory assertion**

Implement:

```python
def test_key_vault_pr_has_no_model_execution_runtime():
    forbidden = (
        "dashboard/backend/domain/model_execution",
        "dashboard/backend/infrastructure/llm/provider_executor.py",
        "dashboard/backend/infrastructure/llm/routed_client.py",
    )
    files = set(
        subprocess.check_output(
            ["git", "ls-files"], cwd=_REPO_ROOT, text=True
        ).splitlines()
    )
    files.update(
        subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=_REPO_ROOT,
            text=True,
        ).splitlines()
    )
    violations = {
        path
        for path in files
        if any(path == prefix or path.startswith(f"{prefix}/") for prefix in forbidden)
    }
    assert violations == set(), f"key vault scope includes model execution: {violations}"
```

Add an actual-filesystem helper using `importlib.machinery` suffixes so ignored `.pyc`, `__pycache__`, and native extension artifacts are also rejected. Cover the helper with temporary sourceless-bytecode fixtures. Do not remove `subprocess` from the module because later import-isolation tests also use it.

- [ ] **Step 2: Run the exact CI failure and architecture suite**

Run:

```bash
pytest \
  dashboard/backend/tests/test_architecture_boundaries.py::test_key_vault_pr_has_no_model_execution_runtime \
  dashboard/backend/tests/test_architecture_boundaries.py -q
```

Expected: both invocations pass without requiring `origin/main`.

- [ ] **Step 3: Commit the CI repair**

```bash
git add dashboard/backend/tests/test_architecture_boundaries.py
git commit -m "test(byok): remove remote ref dependency"
```

---

### Task 4: Full Verification, Push, and Independent Re-review

**Files:**
- Verify only; do not add unrelated local files.

**Interfaces:**
- Consumes: Tasks 1-3 commits.
- Produces: a green PR #396 head and a fresh independent review report.

- [ ] **Step 1: Run the focused BYOK suite**

```bash
pytest \
  dashboard/backend/tests/domain/model_providers/ \
  dashboard/backend/tests/infrastructure/llm/adapters/ \
  dashboard/backend/tests/test_model_credentials_api.py \
  dashboard/backend/tests/test_admin_model_providers_api.py \
  dashboard/backend/tests/test_model_provider_store_postgres.py \
  dashboard/backend/tests/test_architecture_boundaries.py -q
```

Expected: all runnable tests pass; Postgres-only cases may skip locally.

- [ ] **Step 2: Run the complete backend and packaging suites**

```bash
pytest dashboard/backend/tests/ --timeout=180 -p no:cacheprovider
pytest packaging/agentictrading/tests/ -p no:cacheprovider
```

Expected: no failures. If the seed-database account-data test fails because the user's local `dashboard/storage/data/backtest.db` contains account state, report it separately and do not alter that database.

- [ ] **Step 3: Verify the exact staged scope and secret hygiene**

```bash
git diff --check origin/feature/byok-key-vault...HEAD
git status --short
git diff --name-only origin/feature/byok-key-vault...HEAD
```

Expected: only the design/plan and BYOK review-fix files are committed; local database and excluded directories remain unstaged.

- [ ] **Step 4: Push the existing PR branch**

```bash
git push origin feature/byok-key-vault
```

Expected: PR #396 updates to the new HEAD.

- [ ] **Step 5: Wait for GitHub checks and start a fresh review task**

Use a new independent Worktree at the pushed HEAD. Instruct it to review only, not modify, push, comment, approve, or merge. It must verify the two original findings first, then repeat the complete BYOK risk checklist.

- [ ] **Step 6: Merge only after evidence is green**

Require focused tests, complete backend tests, packaging tests, CodeQL, GitHub CI, and the fresh review to have no unresolved findings. Merging remains a separate user-confirmed action because it deploys `main` to production.
