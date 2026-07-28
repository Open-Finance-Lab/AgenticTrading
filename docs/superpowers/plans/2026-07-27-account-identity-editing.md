# Account Identity Editing Implementation Plan

> **✅ EXECUTED — this is now a historical record, not a work list.** Rate limiting was
> revised after execution: the single 60-second cooldown described throughout this plan
> was joined by a 7-day minimum interval between *completed* changes and a rolling
> 3-per-24h request cap, and `email_change_requests` became append-only to support them.
> The code blocks below predate that. **Read the spec's "Rate limiting (revised
> 2026-07-28)" section for the current design**, not this file.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a signed-in user change their display name and (behind their current password plus two 6-character emailed codes) their email address, and move the account-page log out button to the bottom in destructive red.

**Architecture:** Two twin stores (`users.py` SQLite / `users_postgres.py` Postgres) gain identity-mutation methods and a new `email_change_requests` table. A new provider-neutral `infrastructure/email/sender.py` talks to Brevo over `httpx.AsyncClient`. Five new routes hang off the existing `/api/auth` router; the email change is modeled as a singleton pending-resource with one stage-driven verify endpoint. Frontend is vanilla JS in the existing `app.js` / `app.html` / `styles.css`, no build step.

**Tech Stack:** Python 3 / FastAPI / Pydantic v2 / pytest / sqlite3 / psycopg (dict rows) / httpx 0.28.1 / vanilla JS + node-under-pytest for frontend assertions.

**Spec:** `docs/superpowers/specs/2026-07-27-account-identity-editing-design.md`

## Global Constraints

- **Every store change lands in BOTH twins** — `dashboard/backend/users.py` (SQLite) and `dashboard/backend/users_postgres.py` (Postgres). A method in one and not the other is a prod-only crash.
- **Normalize every email with `.strip().lower()` before writing.** SQLite's `users.email` is `UNIQUE COLLATE NOCASE`; the Postgres twin is plain `UNIQUE`. Case-insensitivity in prod is purely an artifact of callers lowercasing. A path that forgets lets `A@x.com` and `a@x.com` coexist in prod while being rejected locally.
- **`print()`, never `logging`.** Logger output from `dashboard.backend.*` is invisible under the deployed uvicorn config. Tests assert on `capsys`, never `caplog`.
- **`ERROR:` prefix = the user got nothing** (mail failures). **`WARNING:` prefix = the durable write already succeeded**, only best-effort cleanup failed (session revocation, pending-change cancellation). Do not level-shift either.
- **Send before persist.** Both mail steps attempt delivery first and write state only on success.
- **Code alphabet:** `23456789ABCDEFGHJKMNPQRSTUVWXYZ` (31 symbols; `0`,`O`,`1`,`I`,`L` removed — note: original text listed 32 but that was unsatisfiable against the test). Length 6. TTL 15 minutes. Max 5 attempts. Cooldown 60 seconds.
- **Route-contract freeze:** every new route tuple goes into `EXPECTED_FULL_CONTRACT` (`dashboard/backend/tests/test_app_composition.py`) **in the same commit that adds the route**. Otherwise `test_full_route_contract_unchanged` turns red on every open PR, not just this one.
- **Seed-DB caution:** running the suite or the app locally in SQLite mode lazily creates `email_change_requests` inside the committed `dashboard/storage/data/backtest.db`, and the write can hide in the untracked `-wal` sidecar. Run `git status --short dashboard/storage/data/backtest.db` before **every** commit; it must show nothing.
- **`@pg_only` fails open.** `TEST_POSTGRES_URL` is unset locally, so the Postgres tier silently skips. A green local run proves nothing about Task 7 — verify by grepping the CI job log.
- **Worktree:** `/mnt/d/github/atl-wt-account`, branch `feat/account-identity-editing`, cut from `origin/main` @ `116874e`. Run all commands from that directory. Test command: `python -m pytest dashboard/backend/tests/ -q`.
- **Base-commit drift:** `origin/main` has moved twice since this plan's `116874e` cut (PR #227, #234), and PR #227 alone edits `api/auth.py`, `test_app_composition.py`, `app.html`, `app.js`, and `styles.css` — the same files Tasks 2/5/8/9/10/11 edit. Every bare "currently line N" citation below is anchored to `116874e` and will be off by the time you read it; the named-anchor and literal-snippet text around each citation is what actually locates the edit — trust that over the number. Task 12, Step 1 rebases onto current `origin/main` before the suite runs and the PR opens.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `dashboard/backend/infrastructure/email/__init__.py` | New subpackage docstring | 1 |
| `dashboard/backend/infrastructure/email/sender.py` | Brevo transactional-mail adapter; `send_email() -> bool`, never raises | 1 |
| `dashboard/backend/tests/test_email_sender.py` | Sender unit tests | 1 |
| `dashboard/backend/tests/conftest.py` | Strip ambient mail env vars | 1 |
| `dashboard/backend/api/auth.py` | Call-time store reads (#185); 5 new routes; D7 | 2, 5, 8, 9 |
| `dashboard/backend/api/routers/discord.py` | Call-time store read (#185) | 2 |
| `dashboard/backend/tests/test_portfolio_api.py` | Drop the now-redundant `auth_module` patch | 2 |
| `dashboard/backend/tests/test_portfolio_allocate.py` | Drop the now-redundant `auth_module` patch | 2 |
| `dashboard/backend/tests/test_users_postgres.py` | Drop the patch; add store-twin coverage | 2, 4, 7 |
| `dashboard/backend/verification_codes.py` | `generate_code()` / `hash_code()` | 3 |
| `dashboard/backend/tests/test_verification_codes.py` | Code helper unit tests | 3 |
| `dashboard/backend/users.py` | SQLite twin: timestamp helpers, constants, identity + request methods | 4, 6 |
| `dashboard/backend/users_postgres.py` | Postgres twin: same surface | 4, 7 |
| `dashboard/backend/tests/test_users_store.py` | New: SQLite store-level tests | 4, 6 |
| `dashboard/backend/tests/test_auth.py` | Route tests for every new endpoint | 5, 8, 9 |
| `dashboard/backend/tests/test_app_composition.py` | Route-contract freeze entries | 5, 8 |
| `dashboard/frontend/app.html` | New sections; logout moved; cache-bust bumps | 10, 11 |
| `dashboard/frontend/styles.css` | `.auth-btn-danger`, `.account-hint`, `.account-email-actions` | 10, 11 |
| `dashboard/frontend/app.js` | `AuthAPI` methods, two init functions, state machine | 11 |
| `dashboard/backend/tests/test_frontend_account_page.py` | New: markup order + CSS cascade guards | 10 |

---

## Task 1: Brevo mail sender

**Files:**
- Create: `dashboard/backend/infrastructure/email/__init__.py`
- Create: `dashboard/backend/infrastructure/email/sender.py`
- Create: `dashboard/backend/tests/test_email_sender.py`
- Modify: `dashboard/backend/tests/conftest.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `async def send_email(to: str, subject: str, text_body: str) -> bool` and `def email_configured() -> bool` in `dashboard.backend.infrastructure.email.sender`. Tasks 8 and 9 call `send_email` **through the module object** (`email_sender.send_email(...)`), never a name bound at import — so there is exactly one place to patch and patching it works.

`infrastructure/` is covered by `test_lower_layers_do_not_import_api_or_app` — this module must not import `api/` or `app.py`. It imports only `os` and `httpx`.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/test_email_sender.py`:

```python
"""Brevo transactional-mail adapter.

Drives the coroutine with asyncio.run() rather than adding a pytest-asyncio
dependency, and monkeypatches httpx.AsyncClient inside the sender module so no
socket is ever opened.
"""

import asyncio

import pytest

from dashboard.backend.infrastructure.email import sender


class _FakeResponse:
    def __init__(self, status_code=201, text="{}"):
        self.status_code = status_code
        self.text = text


def _install_fake_client(monkeypatch, response):
    """Replace httpx.AsyncClient with a recorder; return the captured calls."""
    calls = []

    class _FakeAsyncClient:
        def __init__(self, timeout=None):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def post(self, url, headers=None, json=None):
            calls.append(
                {"url": url, "headers": headers, "json": json, "timeout": self.timeout}
            )
            if isinstance(response, Exception):
                raise response
            return response

    monkeypatch.setattr(sender.httpx, "AsyncClient", _FakeAsyncClient)
    return calls


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("BREVO_API_KEY", "test-key")
    monkeypatch.setenv("ACCOUNT_EMAIL_FROM", "noreply@example.com")
    monkeypatch.delenv("ACCOUNT_EMAIL_FROM_NAME", raising=False)


def test_email_configured_requires_both_vars(monkeypatch):
    monkeypatch.delenv("BREVO_API_KEY", raising=False)
    monkeypatch.delenv("ACCOUNT_EMAIL_FROM", raising=False)
    assert sender.email_configured() is False
    monkeypatch.setenv("BREVO_API_KEY", "k")
    assert sender.email_configured() is False
    monkeypatch.setenv("ACCOUNT_EMAIL_FROM", "a@b.com")
    assert sender.email_configured() is True


def test_send_email_unconfigured_returns_false_and_prints_error(monkeypatch, capsys):
    monkeypatch.delenv("BREVO_API_KEY", raising=False)
    monkeypatch.delenv("ACCOUNT_EMAIL_FROM", raising=False)

    assert asyncio.run(sender.send_email("u@example.com", "Subj", "Body")) is False

    out = capsys.readouterr().out
    assert "ERROR" in out
    assert "BREVO_API_KEY" in out


def test_send_email_posts_the_brevo_payload(monkeypatch, configured):
    calls = _install_fake_client(monkeypatch, _FakeResponse(201))

    assert asyncio.run(sender.send_email("u@example.com", "Subj", "Body")) is True

    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == "https://api.brevo.com/v3/smtp/email"
    assert call["headers"]["api-key"] == "test-key"
    assert call["timeout"] == sender.SEND_TIMEOUT_SECONDS
    assert call["json"]["sender"] == {
        "name": sender.DEFAULT_FROM_NAME,
        "email": "noreply@example.com",
    }
    assert call["json"]["to"] == [{"email": "u@example.com"}]
    assert call["json"]["subject"] == "Subj"
    assert call["json"]["textContent"] == "Body"


def test_send_email_uses_configured_from_name(monkeypatch, configured):
    monkeypatch.setenv("ACCOUNT_EMAIL_FROM_NAME", "Custom Name")
    calls = _install_fake_client(monkeypatch, _FakeResponse(201))

    assert asyncio.run(sender.send_email("u@example.com", "S", "B")) is True
    assert calls[0]["json"]["sender"]["name"] == "Custom Name"


def test_send_email_non_2xx_returns_false_and_prints_error(monkeypatch, configured, capsys):
    _install_fake_client(monkeypatch, _FakeResponse(401, '{"message":"bad key"}'))

    assert asyncio.run(sender.send_email("u@example.com", "S", "B")) is False

    out = capsys.readouterr().out
    assert "ERROR" in out
    assert "401" in out


def test_send_email_strips_newlines_from_the_provider_body(monkeypatch, configured, capsys):
    # A provider body is attacker-influencable (it can echo the submitted
    # address), so it must never inject a forged second log line.
    _install_fake_client(monkeypatch, _FakeResponse(400, "line one\nERROR: forged"))

    assert asyncio.run(sender.send_email("u@example.com", "S", "B")) is False

    out = capsys.readouterr().out.strip()
    assert len(out.splitlines()) == 1


def test_send_email_transport_failure_returns_false(monkeypatch, configured, capsys):
    _install_fake_client(monkeypatch, RuntimeError("connection reset"))

    assert asyncio.run(sender.send_email("u@example.com", "S", "B")) is False

    out = capsys.readouterr().out
    assert "ERROR" in out
    assert "connection reset" in out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest dashboard/backend/tests/test_email_sender.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'dashboard.backend.infrastructure.email'`

- [ ] **Step 3: Create the package**

Create `dashboard/backend/infrastructure/email/__init__.py`:

```python
"""Email infrastructure: transactional-mail provider adapters.

Naming this package ``email`` does not shadow the standard library. Python 3
uses absolute imports, so a bare ``import email`` anywhere in the codebase
still resolves to stdlib; only ``dashboard.backend.infrastructure.email``
reaches this package.
"""
```

- [ ] **Step 4: Write the sender**

Create `dashboard/backend/infrastructure/email/sender.py`:

```python
"""Transactional email delivery via Brevo.

Provider-neutral at the call site: callers get a bool and never an exception --
a provider outage must not turn a route into a 500. But it is loudly
fail-VISIBLE rather than fail-silent: every failure path print()s an ERROR line
before returning False, and the caller is expected to surface a 503 rather than
a cheerful {"status": "ok"} for a message nobody will receive.

print(), not logging: dashboard.backend.* loggers sit at WARNING in every real
deployment (nothing configures logging, and uvicorn's LOGGING_CONFIG has no
'root' key), so logger.error() here would be invisible exactly where it matters.

Config is read at call time, not import time, so tests can set and unset freely.
"""

import os

import httpx

BREVO_ENDPOINT = "https://api.brevo.com/v3/smtp/email"
SEND_TIMEOUT_SECONDS = 10.0
DEFAULT_FROM_NAME = "Agentic Trading Lab"
_PROVIDER_ERROR_SNIPPET_CHARS = 200


def email_configured() -> bool:
    """True when the provider credentials needed to send anything are present.

    Public and called from send_email() below (not just tested standalone) so
    the "are we configured" check exists in exactly one place.
    """
    return bool(os.getenv("BREVO_API_KEY") and os.getenv("ACCOUNT_EMAIL_FROM"))


def _one_line(value: str) -> str:
    """Collapse a provider-supplied string to a single log line.

    The response body can echo submitted content, so an embedded newline would
    let it forge a second log entry. Log-injection defence only -- nothing here
    is a security boundary.
    """
    return value.replace("\r", " ").replace("\n", " ")


async def send_email(to: str, subject: str, text_body: str) -> bool:
    """Send one plain-text transactional email. True on a 2xx, False otherwise."""
    if not email_configured():
        print(
            "ERROR: transactional email requested but BREVO_API_KEY / "
            "ACCOUNT_EMAIL_FROM are not set -- no message was sent"
        )
        return False
    api_key = os.getenv("BREVO_API_KEY")
    sender_email = os.getenv("ACCOUNT_EMAIL_FROM")

    payload = {
        "sender": {
            "name": os.getenv("ACCOUNT_EMAIL_FROM_NAME") or DEFAULT_FROM_NAME,
            "email": sender_email,
        },
        "to": [{"email": to}],
        "subject": subject,
        "textContent": text_body,
    }

    try:
        async with httpx.AsyncClient(timeout=SEND_TIMEOUT_SECONDS) as client:
            response = await client.post(
                BREVO_ENDPOINT,
                headers={"api-key": api_key, "accept": "application/json"},
                json=payload,
            )
    except Exception as exc:  # noqa: BLE001 -- a provider outage must not 500 the route
        print(f"ERROR: transactional email POST failed: {exc!r}")
        return False

    if response.status_code // 100 != 2:
        snippet = _one_line(response.text[:_PROVIDER_ERROR_SNIPPET_CHARS])
        print(
            f"ERROR: transactional email rejected by provider "
            f"(HTTP {response.status_code}): {snippet}"
        )
        return False
    return True
```

- [ ] **Step 5: Strip ambient mail credentials in conftest**

In `dashboard/backend/tests/conftest.py`, find the block of `os.environ.pop(...)` lines and append immediately after `os.environ.pop("AGENT_AUTH_CACHE_TTL_SECONDS", None)`:

```python
# Mail credentials: a developer with a real BREVO_API_KEY exported would
# otherwise have the suite send live email, and would see the
# unconfigured-provider tests fail for a reason that has nothing to do with
# their change. Individual tests set these back via monkeypatch.
os.environ.pop("BREVO_API_KEY", None)
os.environ.pop("ACCOUNT_EMAIL_FROM", None)
os.environ.pop("ACCOUNT_EMAIL_FROM_NAME", None)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest dashboard/backend/tests/test_email_sender.py -q`
Expected: 7 passed

- [ ] **Step 7: Confirm the architecture guard still passes**

Run: `python -m pytest dashboard/backend/tests/test_architecture_boundaries.py -q`
Expected: all pass (the new `infrastructure/` module imports no `api/` or `app`)

- [ ] **Step 8: Commit**

```bash
git status --short dashboard/storage/data/backtest.db   # must print nothing
git add dashboard/backend/infrastructure/email dashboard/backend/tests/test_email_sender.py dashboard/backend/tests/conftest.py
git commit -m "feat(email): add fail-visible Brevo transactional mail sender"
```

---

## Task 2: Fix issue #185 — call-time store resolution

**Files:**
- Modify: `dashboard/backend/api/auth.py` (line 12 import + 10 call sites)
- Modify: `dashboard/backend/api/routers/discord.py` (line 11 import + 1 call site)
- Modify: `dashboard/backend/tests/test_portfolio_api.py`
- Modify: `dashboard/backend/tests/test_portfolio_allocate.py`
- Modify: `dashboard/backend/tests/test_users_postgres.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `dashboard.backend.api.auth` no longer has a module-level `user_store` attribute. Every later task reads the store as `users_module.user_store` inside the function body.

**Why this is bigger than the spec said.** The spec named `pg_client` as the only affected fixture. It is not. **Three** test files patch `auth_module.user_store`, and `monkeypatch.setattr` raises `AttributeError` when the attribute is gone — so leaving any of them in place turns the fixture into an error, not a failure:

| File | Line | Note |
|---|---|---|
| `test_portfolio_api.py` | 39 | guarded by `test_signup_lands_in_the_fixture_user_store` (`:68`) |
| `test_portfolio_allocate.py` | 42 | |
| `test_users_postgres.py` | 82 | `@pg_only` — **fails open locally**, so a green local run will not catch a mistake here |

- [ ] **Step 1: Write the failing test**

Append to `dashboard/backend/tests/test_auth.py`:

```python
def test_auth_routes_resolve_the_store_at_call_time(temp_user_store, monkeypatch):
    """Issue #185: api/auth.py must not bind the user_store singleton at import.

    Patching only dashboard.backend.users must be enough to redirect every auth
    route. When auth.py holds its own import-time binding, this signup lands in
    the process-wide store and the temp store below stays empty -- silently, with
    the test still green, which is exactly how #185 survived this long.
    """
    from dashboard.backend import users as users_module

    monkeypatch.setattr(users_module, "user_store", temp_user_store)
    client = TestClient(app)

    response = client.post(
        "/api/auth/signup",
        json={
            "email": "callsite@example.com",
            "display_name": "Callsite",
            "password": "securepass1",
        },
    )
    assert response.status_code == 200
    assert temp_user_store.get_user_by_email("callsite@example.com") is not None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest dashboard/backend/tests/test_auth.py::test_auth_routes_resolve_the_store_at_call_time -q`
Expected: FAIL — `assert None is not None` (the account went to the process-wide store)

- [ ] **Step 3: Fix `api/auth.py`**

Replace line 12:

```python
from dashboard.backend.users import public_user, user_store, verify_password
```

with:

```python
from dashboard.backend import users as users_module
from dashboard.backend.users import public_user, verify_password
```

`public_user` and `verify_password` are pure functions and stay bound at import;
only the mutable singleton must be read late.

Then replace every bare `user_store.` with `users_module.user_store.` — **10 call sites**, at lines 121, 134, 144, 150, 154, 167, 182, 192, 214 and 279 of the current file:

```python
user = users_module.user_store.get_user_for_token(token)          # get_current_user
user = users_module.user_store.create_user(...)                   # signup
token = users_module.user_store.create_session(user["id"])        # signup
user = users_module.user_store.authenticate(...)                  # login
token = users_module.user_store.create_session(user["id"])        # login
users_module.user_store.delete_session(token)                     # logout
users_module.user_store.update_password(...)                      # change_password
users_module.user_store.delete_other_sessions(...)                # change_password
return users_module.user_store.set_avatar(user_id, value)         # _store_avatar
users_module.user_store.link_discord_user, user_id, ...           # discord callback
```

Verify none were missed:

```bash
grep -n '[^.]user_store' dashboard/backend/api/auth.py
```

Expected: only the `import` line matches.

- [ ] **Step 4: Fix `api/routers/discord.py`**

Replace line 11:

```python
from dashboard.backend.users import user_store
```

with:

```python
from dashboard.backend import users as users_module
```

and line 25:

```python
    user = users_module.user_store.get_user_by_discord_id(discord_user_id)
```

- [ ] **Step 5: Drop the now-redundant patch in `test_portfolio_api.py`**

Delete the `auth_module` import at line 11 (`import dashboard.backend.api.auth as auth_module`) and replace lines 34-39 with:

```python
        monkeypatch.setattr(users_module, "user_store", user_store)
```

Then update the guard test's docstring at `:68` so it documents what it now proves:

```python
def test_signup_lands_in_the_fixture_user_store(client, temp_stores):
    """Guards call-time store resolution in api/auth.py (issue #185).

    If any auth route goes back to binding the singleton at import time, the
    temp users.db stays empty and every account these tests create leaks into
    the session-wide DB -- making the fixed email above an ordering hazard for
    any other test that signs up with it. The failure is invisible otherwise:
    the tests still pass, against the wrong store.
    """
```

- [ ] **Step 6: Drop the redundant patch in `test_portfolio_allocate.py`**

Delete the `auth_module` import at line 16 and replace lines 40-42 with:

```python
        monkeypatch.setattr(users_module, "user_store", user_store)
```

- [ ] **Step 7: Drop the redundant patch in `test_users_postgres.py`**

Replace the `pg_client` fixture (lines 70-83) with:

```python
@pytest.fixture
def pg_client(temp_postgres_store, monkeypatch):
    import dashboard.backend.users as users_module

    # api/auth.py resolves users_module.user_store at call time (issue #185),
    # so this single patch redirects every auth route. Before that fix it also
    # needed dashboard.backend.api.auth patched, and without it this "postgres"
    # test silently exercised SQLite -- caught only when CI first ran the live
    # tier and it collided with test_auth.py's alice@example.com.
    monkeypatch.setattr(users_module, "user_store", temp_postgres_store)
    return TestClient(app)
```

- [ ] **Step 8: Confirm no `auth_module` patches survive**

```bash
grep -rn 'auth_module' dashboard/backend/tests/
```

Expected: no output.

- [ ] **Step 9: Run the affected suites**

Run: `python -m pytest dashboard/backend/tests/test_auth.py dashboard/backend/tests/test_portfolio_api.py dashboard/backend/tests/test_portfolio_allocate.py dashboard/backend/tests/test_users_postgres.py -q`
Expected: all pass, including the new call-time test.

- [ ] **Step 10: Re-read the revocation test deliberately**

Open `dashboard/backend/tests/test_auth.py::test_change_password_revocation_failure_still_succeeds` (around line 231). Its in-file comment says the `client` fixture "only reassigns users_module.user_store; api/auth.py may still hold the original singleton binding" — **that premise is the bug just fixed.** The test patches `UserStore` at the *class* level, which still works and is still the right call (it fails whichever instance the route resolves). Update only the stale half of the comment:

```python
def test_change_password_revocation_failure_still_succeeds(client, monkeypatch, capsys):
    # The password write and the other-session revocation are two separate
    # transactions. If revocation raises, the (already-durable) password change
    # must still report success rather than a misleading 500. Patch at the CLASS
    # level so it fails for any UserStore instance, including the fixture's.
    # `UserStore` is already imported.
```

Run: `python -m pytest dashboard/backend/tests/test_auth.py::test_change_password_revocation_failure_still_succeeds -q`
Expected: PASS

- [ ] **Step 11: Commit**

```bash
git status --short dashboard/storage/data/backtest.db   # must print nothing
git add dashboard/backend/api/auth.py dashboard/backend/api/routers/discord.py dashboard/backend/tests/
git commit -m "fix(auth): resolve user_store at call time (#185)"
```

---

## Task 3: Verification code helpers

**Files:**
- Create: `dashboard/backend/verification_codes.py`
- Create: `dashboard/backend/tests/test_verification_codes.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `generate_code() -> str` and `hash_code(code: str) -> str` in `dashboard.backend.verification_codes`, plus the constants `CODE_ALPHABET` and `CODE_LENGTH`. Tasks 8 and 9 import both functions.

Backend-root module, matching the existing `password_policy.py` sibling convention.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/test_verification_codes.py`:

```python
"""Short-lived confirmation codes for the email-change flow."""

from dashboard.backend.verification_codes import (
    CODE_ALPHABET,
    CODE_LENGTH,
    generate_code,
    hash_code,
)


def test_generated_code_has_the_expected_shape():
    for _ in range(50):
        code = generate_code()
        assert len(code) == CODE_LENGTH == 6
        assert set(code) <= set(CODE_ALPHABET)


def test_alphabet_excludes_the_characters_users_misread():
    # 0/O and 1/I/L are the pairs users transcribe wrong off a phone screen.
    for ambiguous in "0O1IL":
        assert ambiguous not in CODE_ALPHABET
    assert len(CODE_ALPHABET) == 31
    assert len(set(CODE_ALPHABET)) == 31


def test_generated_codes_are_not_all_identical():
    assert len({generate_code() for _ in range(50)}) > 1


def test_hash_is_case_insensitive_and_whitespace_tolerant():
    assert hash_code("abc234") == hash_code("ABC234")
    assert hash_code("  ABC234  ") == hash_code("ABC234")


def test_hash_is_stable_and_distinguishes_codes():
    assert hash_code("ABC234") == hash_code("ABC234")
    assert hash_code("ABC234") != hash_code("ABC235")
    assert len(hash_code("ABC234")) == 64  # sha256 hex
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest dashboard/backend/tests/test_verification_codes.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'dashboard.backend.verification_codes'`

- [ ] **Step 3: Write the module**

Create `dashboard/backend/verification_codes.py`:

```python
"""Short-lived alphanumeric confirmation codes.

Used by the email-change flow; the password-reset flow (#187) is the intended
second consumer, which is why this is a standalone module rather than private
helpers inside api/auth.py.
"""

import hashlib
import secrets

# 31 symbols: digits and uppercase letters, minus 0/O and 1/I/L -- the pairs a
# user misreads off a phone screen and types back wrong. 31**6 is about
# 8.9e8 combinations, which a 5-attempt cap and a 15-minute expiry make
# comfortably unguessable.
CODE_ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"
CODE_LENGTH = 6


def generate_code() -> str:
    """Return a fresh code drawn from a CSPRNG."""
    return "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))


def hash_code(code: str) -> str:
    """SHA-256 hex of the normalized (stripped, uppercased) code.

    Normalizing here is what makes comparison case-insensitive, so a user typing
    lowercase still matches.

    SHA-256 rather than bcrypt: this is a short-lived high-entropy secret, and
    what stops guessing is the attempt cap and the expiry, not hash cost --
    bcrypt would add work to every attempt for no gain. The hash is not an
    offline-attack defence (1e9 candidates is trivially searchable, and anyone
    with database write access could simply rewrite users.email). It stops a
    *casual* read -- a log line, a backup, a support query -- from yielding a
    live code.
    """
    return hashlib.sha256(code.strip().upper().encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest dashboard/backend/tests/test_verification_codes.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git status --short dashboard/storage/data/backtest.db   # must print nothing
git add dashboard/backend/verification_codes.py dashboard/backend/tests/test_verification_codes.py
git commit -m "feat(auth): add short-lived verification code helpers"
```

---

## Task 4: Display name — both store twins

**Files:**
- Modify: `dashboard/backend/users.py` (add method after `set_avatar`, currently ending line 326)
- Modify: `dashboard/backend/users_postgres.py` (add method after `set_avatar`, currently ending line 220)
- Create: `dashboard/backend/tests/test_users_store.py`
- Modify: `dashboard/backend/tests/test_users_postgres.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `update_display_name(user_id: int, display_name: str) -> dict` on **both** stores. Returns the `public_user(...)` dict; raises `ValueError("user_not_found")` when the row is gone. Task 5 calls it.

- [ ] **Step 1: Write the failing SQLite tests**

Create `dashboard/backend/tests/test_users_store.py`:

```python
"""UserStore (SQLite twin) behaviour.

The Postgres twin mirrors every case here under @pg_only in
test_users_postgres.py -- a method that exists in one twin and not the other is
a prod-only crash.
"""

import tempfile
from pathlib import Path

import pytest

from dashboard.backend.users import UserStore


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield UserStore(db_path=Path(tmpdir) / "users.db")


@pytest.fixture
def user(store):
    return store.create_user("owner@example.com", "Owner", "securepass1")


def test_update_display_name_persists_and_returns_public_user(store, user):
    updated = store.update_display_name(user["id"], "Renamed")

    assert updated["display_name"] == "Renamed"
    assert "password_hash" not in updated
    assert store.get_user_by_id(user["id"])["display_name"] == "Renamed"


def test_update_display_name_strips_surrounding_whitespace(store, user):
    updated = store.update_display_name(user["id"], "  Padded  ")
    assert updated["display_name"] == "Padded"


def test_update_display_name_rejects_a_missing_user(store):
    with pytest.raises(ValueError, match="user_not_found"):
        store.update_display_name(999_999, "Ghost")
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest dashboard/backend/tests/test_users_store.py -q`
Expected: 3 failed — `AttributeError: 'UserStore' object has no attribute 'update_display_name'`

- [ ] **Step 3: Implement the SQLite twin**

In `dashboard/backend/users.py`, insert immediately after `set_avatar` (after its `return public_user(row)`, currently line 326):

```python
    def update_display_name(self, user_id: int, display_name: str) -> Dict[str, Any]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET display_name = ? WHERE id = ?",
            (display_name.strip(), user_id),
        )
        conn.commit()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            raise ValueError("user_not_found")
        return public_user(row)
```

- [ ] **Step 4: Implement the Postgres twin**

In `dashboard/backend/users_postgres.py`, insert immediately after `set_avatar` (after its `return public_user(row)`, currently line 220):

```python
    def update_display_name(self, user_id: int, display_name: str) -> Dict[str, Any]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET display_name = %s WHERE id = %s RETURNING *",
                    (display_name.strip(), user_id),
                )
                row = cur.fetchone()
        if not row:
            raise ValueError("user_not_found")
        return public_user(row)
```

- [ ] **Step 5: Mirror the coverage in the Postgres tier**

Append to `dashboard/backend/tests/test_users_postgres.py`:

```python
@pg_only
def test_update_display_name_postgres(temp_postgres_store):
    user = temp_postgres_store.create_user("pgname@example.com", "PG Name", "securepass1")

    updated = temp_postgres_store.update_display_name(user["id"], "  PG Renamed  ")

    assert updated["display_name"] == "PG Renamed"
    assert temp_postgres_store.get_user_by_id(user["id"])["display_name"] == "PG Renamed"


@pg_only
def test_update_display_name_missing_user_postgres(temp_postgres_store):
    with pytest.raises(ValueError, match="user_not_found"):
        temp_postgres_store.update_display_name(999_999, "Ghost")
```

- [ ] **Step 6: Run to verify they pass**

Run: `python -m pytest dashboard/backend/tests/test_users_store.py dashboard/backend/tests/test_users_postgres.py -q`
Expected: SQLite cases pass; the `@pg_only` cases report as skipped locally (`TEST_POSTGRES_URL not set`). That skip is **not** evidence they work — see the Global Constraints.

- [ ] **Step 7: Commit**

```bash
git status --short dashboard/storage/data/backtest.db   # must print nothing
git add dashboard/backend/users.py dashboard/backend/users_postgres.py dashboard/backend/tests/test_users_store.py dashboard/backend/tests/test_users_postgres.py
git commit -m "feat(auth): add update_display_name to both user store twins"
```

---

## Task 5: Display name — route

**Files:**
- Modify: `dashboard/backend/api/auth.py`
- Modify: `dashboard/backend/tests/test_auth.py`
- Modify: `dashboard/backend/tests/test_app_composition.py`

**Interfaces:**
- Consumes: `users_module.user_store.update_display_name` (Task 4); call-time store resolution (Task 2).
- Produces: `PUT /api/auth/display-name` returning `{"user": {...}}`.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/backend/tests/test_auth.py`:

```python
def test_update_display_name_happy_path(client):
    token = _signup_and_token(client, email="name@example.com")

    response = client.put(
        "/api/auth/display-name",
        headers={"Authorization": f"Bearer {token}"},
        json={"display_name": "New Name"},
    )

    assert response.status_code == 200
    assert response.json()["user"]["display_name"] == "New Name"
    # And it is durable, not just echoed back.
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.json()["user"]["display_name"] == "New Name"


def test_update_display_name_strips_whitespace(client):
    token = _signup_and_token(client, email="trim@example.com")

    response = client.put(
        "/api/auth/display-name",
        headers={"Authorization": f"Bearer {token}"},
        json={"display_name": "  Trimmed  "},
    )

    assert response.status_code == 200
    assert response.json()["user"]["display_name"] == "Trimmed"


def test_update_display_name_rejects_whitespace_only(client):
    # Field(min_length=1) passes on "   " because pydantic measures the raw
    # string. Storing it would repeat issue #167 on a second surface.
    token = _signup_and_token(client, email="blank@example.com")

    response = client.put(
        "/api/auth/display-name",
        headers={"Authorization": f"Bearer {token}"},
        json={"display_name": "     "},
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_update_display_name_requires_auth(client):
    response = client.put("/api/auth/display-name", json={"display_name": "Nope"})
    assert response.status_code == 401


def test_update_display_name_rejects_overlong_value(client):
    token = _signup_and_token(client, email="long@example.com")

    response = client.put(
        "/api/auth/display-name",
        headers={"Authorization": f"Bearer {token}"},
        json={"display_name": "x" * 101},
    )

    assert response.status_code == 422
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest dashboard/backend/tests/test_auth.py -q -k display_name`
Expected: FAIL — 405/404 rather than 200 (route does not exist)

- [ ] **Step 3: Add the request model**

In `dashboard/backend/api/auth.py`, insert after `ChangePasswordRequest` (currently ending line 59):

```python
class DisplayNameRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=100)
```

- [ ] **Step 4: Add the route**

Insert after the `change_password` route. `return {"status": "ok"}` is not a unique anchor
— it also closes `logout` (currently line 173) — so anchor on the tail of
`change_password` specifically, currently ending at line 205:

```python
    except Exception as exc:  # noqa: BLE001 -- password change already committed
        print(
            f"WARNING: change-password committed for user {current_user['id']} but "
            f"other-session revocation failed: {exc!r}"
        )
    return {"status": "ok"}
```

```python
@router.put("/display-name")
async def update_display_name(
    payload: DisplayNameRequest,
    current_user: dict = Depends(get_current_user),
):
    display_name = payload.display_name.strip()
    if not display_name:
        # Field(min_length=1) measures the raw string, so "   " reaches here.
        # Storing it would repeat issue #167 (a whitespace-only name persisted
        # as an empty label with no way to tell it from a missing one).
        raise HTTPException(status_code=400, detail="Display name cannot be empty.")
    # No password required: a display name is not an authentication factor, and
    # gating it behind one is not what any comparable platform does.
    try:
        user = users_module.user_store.update_display_name(
            current_user["id"], display_name
        )
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Session is no longer valid.") from exc
    return {"user": user}
```

- [ ] **Step 5: Update the route-contract freeze**

In `dashboard/backend/tests/test_app_composition.py`, add to `EXPECTED_FULL_CONTRACT` beside the other `/api/auth/*` entries (currently 12 entries at lines 68-79 — nine at this plan's `116874e` base, plus three `/api/auth/robinhood/*` tuples PR #227 added afterward; re-grep before editing, `main` may have moved further by execution time):

```python
    ("PUT", "/api/auth/display-name"),
```

- [ ] **Step 6: Run to verify they pass**

Run: `python -m pytest dashboard/backend/tests/test_auth.py dashboard/backend/tests/test_app_composition.py -q`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git status --short dashboard/storage/data/backtest.db   # must print nothing
git add dashboard/backend/api/auth.py dashboard/backend/tests/
git commit -m "feat(auth): add PUT /api/auth/display-name"
```

---

## Task 6: Email-change store — SQLite twin

**Files:**
- Modify: `dashboard/backend/users.py`
- Modify: `dashboard/backend/tests/test_users_store.py`

**Interfaces:**
- Consumes: nothing.
- Produces, at module level in `dashboard.backend.users`:
  - `EMAIL_CHANGE_TTL_MINUTES = 15`, `EMAIL_CHANGE_MAX_ATTEMPTS = 5`, `EMAIL_CHANGE_COOLDOWN_SECONDS = 60`
  - `parse_stored_timestamp(value: str) -> datetime` (public — Task 8 uses it for the cooldown)
  - `is_expired(expires_at: str) -> bool`
- Produces, on `UserStore`:
  - `create_email_change_request(user_id: int, new_email: str, code_hash: str) -> dict`
  - `get_active_email_change(user_id: int) -> dict | None`
  - `advance_email_change(request_id: int, code_hash: str) -> dict`
  - `record_email_change_attempt(request_id: int) -> int`
  - `mark_email_change_used(request_id: int) -> None`
  - `cancel_email_change(user_id: int) -> None`
  - `update_email(user_id: int, new_email: str) -> dict`
  - `last_email_change_request_at(user_id: int) -> str | None`

**Three deliberate deviations from the spec:**

1. **`mark_email_change_used` is an eighth method the spec's list omits.** The spec gives `used_at` the job "set on commit; also what the cooldown reads" but lists no method that ever writes it, and has the cooldown read `created_at` instead. Deleting the row on commit would work, but then a user could start a second change one second after finishing the first. Marking it used keeps `get_active_email_change` returning `None` (the `used_at IS NULL` filter) *and* keeps the row visible to `last_email_change_request_at`, so the cooldown survives a completed change.

2. **`created_at` is written explicitly rather than defaulted.** A bare SQLite `TIMESTAMP` column with no application-side write reads back as whatever the driver's default formatting produces — `'2026-07-27 10:00:00'`, space-separated, no offset — while the Postgres twin stores `_utcnow_iso()` (`'2026-07-27T10:00:00+00:00'`). One parser has to read both. Writing `_utcnow_iso()` explicitly in both twins makes the stored format identical, so `parse_stored_timestamp` has one job instead of two.
3. **`cancel_email_change` deactivates rather than deletes.** The spec's store-methods list has it "delete[] any row for the user" (Part B). Deleting also erases the row `last_email_change_request_at` reads for the 60-second cooldown, and `DELETE /api/auth/email-change` (Task 8) needs only a valid session, not the password — so a caller who already knows the account's password could loop request (send, password-gated) → cancel (wipe the cooldown clock, session-only) → request again, with the cooldown never enforced: mail-bombing the original address and burning the platform's shared 300/day Brevo quota in well under a minute. `mark_email_change_used` already solves exactly this for the success path by setting `used_at` instead of deleting; `cancel_email_change` gets a matching `cancelled_at` column instead of a `DELETE`, so the cooldown survives a cancel exactly as it survives a completed change. The Task 9 five-wrong-attempts path calls this same method, so one change closes both trigger paths.

- [ ] **Step 1: Write the failing tests**

First extend the **import block at the top** of `dashboard/backend/tests/test_users_store.py` (do not leave these mid-file):

```python
from datetime import timedelta

from dashboard.backend.users import (
    EMAIL_CHANGE_TTL_MINUTES,
    UserStore,
    _utcnow,
    parse_stored_timestamp,
)
from dashboard.backend.verification_codes import hash_code
```

Then append the cases:

```python


def test_create_email_change_request_starts_at_stage_old(store, user):
    row = store.create_email_change_request(user["id"], "next@example.com", hash_code("ABC234"))

    assert row["stage"] == "old"
    assert row["new_email"] == "next@example.com"
    assert row["attempts"] == 0
    assert row["used_at"] is None
    expires = parse_stored_timestamp(row["expires_at"])
    assert timedelta(minutes=EMAIL_CHANGE_TTL_MINUTES - 1) < expires - _utcnow()


def test_create_email_change_request_normalizes_the_new_email(store, user):
    row = store.create_email_change_request(user["id"], "  MiXeD@Example.COM ", hash_code("A"))
    assert row["new_email"] == "mixed@example.com"


def test_create_email_change_request_replaces_any_prior_request(store, user):
    store.create_email_change_request(user["id"], "first@example.com", hash_code("A"))
    store.create_email_change_request(user["id"], "second@example.com", hash_code("B"))

    active = store.get_active_email_change(user["id"])
    assert active["new_email"] == "second@example.com"


def test_get_active_email_change_is_none_without_a_request(store, user):
    assert store.get_active_email_change(user["id"]) is None


def test_get_active_email_change_ignores_an_expired_request(store, user):
    row = store.create_email_change_request(user["id"], "next@example.com", hash_code("A"))
    stale = (_utcnow() - timedelta(minutes=1)).replace(microsecond=0).isoformat()
    conn = store._get_connection()
    conn.execute(
        "UPDATE email_change_requests SET expires_at = ? WHERE id = ?", (stale, row["id"])
    )
    conn.commit()
    conn.close()

    assert store.get_active_email_change(user["id"]) is None


def test_advance_email_change_moves_to_stage_new_and_resets_attempts(store, user):
    row = store.create_email_change_request(user["id"], "next@example.com", hash_code("A"))
    store.record_email_change_attempt(row["id"])

    advanced = store.advance_email_change(row["id"], hash_code("Z9Y8X7"))

    assert advanced["stage"] == "new"
    assert advanced["code_hash"] == hash_code("Z9Y8X7")
    assert advanced["attempts"] == 0
    assert advanced["new_email"] == "next@example.com"


def test_record_email_change_attempt_increments_and_returns_the_count(store, user):
    row = store.create_email_change_request(user["id"], "next@example.com", hash_code("A"))

    assert store.record_email_change_attempt(row["id"]) == 1
    assert store.record_email_change_attempt(row["id"]) == 2


def test_mark_email_change_used_deactivates_but_keeps_the_row(store, user):
    row = store.create_email_change_request(user["id"], "next@example.com", hash_code("A"))

    store.mark_email_change_used(row["id"])

    assert store.get_active_email_change(user["id"]) is None
    # Still visible to the cooldown, so a completed change cannot be immediately
    # followed by another.
    assert store.last_email_change_request_at(user["id"]) is not None


def test_cancel_email_change_deactivates_but_preserves_the_cooldown(store, user):
    row = store.create_email_change_request(user["id"], "next@example.com", hash_code("A"))

    store.cancel_email_change(user["id"])

    assert store.get_active_email_change(user["id"]) is None
    # The cooldown clock must survive a cancel: otherwise an authenticated caller
    # who knows the password could loop request/cancel/request with the cooldown
    # never enforced, mail-bombing the account and burning the shared quota.
    assert store.last_email_change_request_at(user["id"]) == row["created_at"]


def test_last_email_change_request_at_is_none_without_a_request(store, user):
    assert store.last_email_change_request_at(user["id"]) is None


def test_update_email_persists_lowercased(store, user):
    updated = store.update_email(user["id"], "  NEW@Example.COM  ")

    assert updated["email"] == "new@example.com"
    assert store.get_user_by_email("new@example.com") is not None


def test_update_email_rejects_an_address_another_account_owns(store, user):
    store.create_user("taken@example.com", "Taken", "securepass1")

    with pytest.raises(ValueError, match="email_already_registered"):
        store.update_email(user["id"], "taken@example.com")

    # The original address is untouched.
    assert store.get_user_by_id(user["id"])["email"] == "owner@example.com"


def test_update_email_rejects_a_missing_user(store):
    with pytest.raises(ValueError, match="user_not_found"):
        store.update_email(999_999, "ghost@example.com")
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest dashboard/backend/tests/test_users_store.py -q`
Expected: collection error — `ImportError: cannot import name 'EMAIL_CHANGE_TTL_MINUTES'`

- [ ] **Step 3: Add module-level constants and timestamp helpers**

In `dashboard/backend/users.py`, after `BCRYPT_MAX_BYTES = 72` (line 22):

```python
EMAIL_CHANGE_TTL_MINUTES = 15
EMAIL_CHANGE_MAX_ATTEMPTS = 5
EMAIL_CHANGE_COOLDOWN_SECONDS = 60
```

And after `_utcnow_iso` (line 30):

```python
def parse_stored_timestamp(value: str) -> datetime:
    """Read a timestamp written by either twin.

    Both stores write _utcnow_iso() (offset-aware ISO-8601), but rows predating
    that convention -- or written by SQLite's CURRENT_TIMESTAMP default -- come
    back naive. Treat naive as UTC, which is what every writer here means.
    """
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def is_expired(expires_at: str) -> bool:
    return parse_stored_timestamp(expires_at) < _utcnow()
```

- [ ] **Step 4: Add the table to `_init_schema`**

In `UserStore._init_schema`, insert before `conn.commit()` (currently line 180):

```python
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS email_change_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                new_email TEXT NOT NULL,
                stage TEXT NOT NULL,
                code_hash TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                used_at TIMESTAMP,
                cancelled_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_email_change_requests_user_id
            ON email_change_requests(user_id)
            """
        )
```

- [ ] **Step 5: Add the store methods**

In `dashboard/backend/users.py`, insert after `update_display_name` (added in Task 4):

```python
    def _email_change_expiry(self) -> str:
        return (
            (_utcnow() + timedelta(minutes=EMAIL_CHANGE_TTL_MINUTES))
            .replace(microsecond=0)
            .isoformat()
        )

    def create_email_change_request(
        self, user_id: int, new_email: str, code_hash: str
    ) -> Dict[str, Any]:
        """Replace any in-flight request for this user with a fresh stage-'old' one."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM email_change_requests WHERE user_id = ?", (user_id,))
        cursor.execute(
            """
            INSERT INTO email_change_requests
                (user_id, new_email, stage, code_hash, created_at, expires_at)
            VALUES (?, ?, 'old', ?, ?, ?)
            """,
            (
                user_id,
                new_email.strip().lower(),
                code_hash,
                _utcnow_iso(),
                self._email_change_expiry(),
            ),
        )
        conn.commit()
        request_id = cursor.lastrowid
        cursor.execute(
            "SELECT * FROM email_change_requests WHERE id = ?", (request_id,)
        )
        row = cursor.fetchone()
        conn.close()
        return dict(row)

    def get_active_email_change(self, user_id: int) -> Optional[Dict[str, Any]]:
        """The user's in-flight request, or None if absent, used, cancelled, or expired."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM email_change_requests
            WHERE user_id = ? AND used_at IS NULL AND cancelled_at IS NULL
            ORDER BY id DESC LIMIT 1
            """,
            (user_id,),
        )
        row = cursor.fetchone()
        conn.close()
        if not row or is_expired(row["expires_at"]):
            return None
        return dict(row)

    def advance_email_change(self, request_id: int, code_hash: str) -> Dict[str, Any]:
        """Move a verified stage-'old' request to stage 'new' with a fresh code."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE email_change_requests
            SET stage = 'new', code_hash = ?, attempts = 0, expires_at = ?
            WHERE id = ?
            """,
            (code_hash, self._email_change_expiry(), request_id),
        )
        conn.commit()
        cursor.execute(
            "SELECT * FROM email_change_requests WHERE id = ?", (request_id,)
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            raise ValueError("email_change_request_not_found")
        return dict(row)

    def record_email_change_attempt(self, request_id: int) -> int:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE email_change_requests SET attempts = attempts + 1 WHERE id = ?",
            (request_id,),
        )
        conn.commit()
        cursor.execute(
            "SELECT attempts FROM email_change_requests WHERE id = ?", (request_id,)
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            raise ValueError("email_change_request_not_found")
        return int(row["attempts"])

    def mark_email_change_used(self, request_id: int) -> None:
        """Retire a completed request without deleting it.

        used_at makes get_active_email_change skip the row while
        last_email_change_request_at still sees it, so the cooldown applies to a
        change that just succeeded as well as one still in flight.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE email_change_requests SET used_at = ? WHERE id = ?",
            (_utcnow_iso(), request_id),
        )
        conn.commit()
        conn.close()

    def cancel_email_change(self, user_id: int) -> None:
        """Deactivate the user's request without deleting it.

        Mirrors mark_email_change_used: cancelled_at makes get_active_email_change
        skip the row while last_email_change_request_at still sees it. Deleting
        instead would let an authenticated caller who knows the password loop
        request/cancel/request with the 60-second cooldown never enforced --
        mail-bombing the account and burning the shared Brevo quota.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE email_change_requests SET cancelled_at = ? WHERE user_id = ?",
            (_utcnow_iso(), user_id),
        )
        conn.commit()
        conn.close()

    def last_email_change_request_at(self, user_id: int) -> Optional[str]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT created_at FROM email_change_requests
            WHERE user_id = ? ORDER BY id DESC LIMIT 1
            """,
            (user_id,),
        )
        row = cursor.fetchone()
        conn.close()
        return str(row["created_at"]) if row else None

    def update_email(self, user_id: int, new_email: str) -> Dict[str, Any]:
        normalized_email = new_email.strip().lower()
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "UPDATE users SET email = ? WHERE id = ?",
                (normalized_email, user_id),
            )
            conn.commit()
        except sqlite3.IntegrityError as exc:
            conn.close()
            raise ValueError("email_already_registered") from exc
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            raise ValueError("user_not_found")
        return public_user(row)
```

- [ ] **Step 6: Run to verify they pass**

Run: `python -m pytest dashboard/backend/tests/test_users_store.py -q`
Expected: 16 passed

- [ ] **Step 7: Commit**

```bash
git status --short dashboard/storage/data/backtest.db   # must print nothing
git add dashboard/backend/users.py dashboard/backend/tests/test_users_store.py
git commit -m "feat(auth): add email-change request store (SQLite twin)"
```

---

## Task 7: Email-change store — Postgres twin

**Files:**
- Modify: `dashboard/backend/users_postgres.py`
- Modify: `dashboard/backend/tests/test_users_postgres.py`

**Interfaces:**
- Consumes: the constants and helpers from Task 6 (`EMAIL_CHANGE_TTL_MINUTES`, `is_expired`, `_utcnow_iso`), imported from `dashboard.backend.users` exactly as the twin already imports `hash_password`/`public_user`.
- Produces: the same eight methods as Task 6, with identical signatures and identical `ValueError` strings.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/backend/tests/test_users_postgres.py`:

```python
@pg_only
def test_email_change_request_lifecycle_postgres(temp_postgres_store):
    from dashboard.backend.verification_codes import hash_code

    store = temp_postgres_store
    user = store.create_user("pgmail@example.com", "PG Mail", "securepass1")

    row = store.create_email_change_request(user["id"], "  NEXT@Example.COM ", hash_code("A"))
    assert row["stage"] == "old"
    assert row["new_email"] == "next@example.com"
    assert row["attempts"] == 0

    assert store.record_email_change_attempt(row["id"]) == 1

    advanced = store.advance_email_change(row["id"], hash_code("Z9Y8X7"))
    assert advanced["stage"] == "new"
    assert advanced["attempts"] == 0
    assert advanced["code_hash"] == hash_code("Z9Y8X7")

    store.mark_email_change_used(row["id"])
    assert store.get_active_email_change(user["id"]) is None
    assert store.last_email_change_request_at(user["id"]) is not None

    store.cancel_email_change(user["id"])
    assert store.get_active_email_change(user["id"]) is None
    # Deactivated, not deleted -- the cooldown clock must survive a cancel.
    assert store.last_email_change_request_at(user["id"]) is not None


@pg_only
def test_update_email_postgres(temp_postgres_store):
    store = temp_postgres_store
    user = store.create_user("pgold@example.com", "PG Old", "securepass1")

    updated = store.update_email(user["id"], "  PGNEW@Example.COM  ")

    # Lowercasing is mandatory here: this twin's UNIQUE is NOT case-insensitive,
    # so an un-normalized write would let two casings of one address coexist in
    # prod while SQLite rejects them locally.
    assert updated["email"] == "pgnew@example.com"
    assert store.get_user_by_email("pgnew@example.com") is not None


@pg_only
def test_update_email_conflict_postgres(temp_postgres_store):
    store = temp_postgres_store
    user = store.create_user("pgmine@example.com", "PG Mine", "securepass1")
    store.create_user("pgtaken@example.com", "PG Taken", "securepass1")

    with pytest.raises(ValueError, match="email_already_registered"):
        store.update_email(user["id"], "pgtaken@example.com")
```

Also extend the `temp_postgres_store` fixture's cleanup (currently lines 63-66) so a leftover request never leaks between cases:

```python
    with store._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM email_change_requests")
            cur.execute("DELETE FROM auth_sessions")
            cur.execute("DELETE FROM users")
```

- [ ] **Step 2: Run to verify they skip locally**

Run: `python -m pytest dashboard/backend/tests/test_users_postgres.py -q`
Expected: the new cases report **skipped** (`TEST_POSTGRES_URL not set`). This tier cannot be verified locally — see Step 6.

- [ ] **Step 3: Update the twin's imports and schema**

In `dashboard/backend/users_postgres.py`, extend the import on line 18:

```python
from dashboard.backend.users import (
    EMAIL_CHANGE_TTL_MINUTES,
    _utcnow,
    _utcnow_iso,
    hash_password,
    is_expired,
    public_user,
    verify_password,
)
```

`timedelta` is already imported on line 12 (`from datetime import datetime, timedelta, timezone`) — no change needed there.

In `_init_schema`, after the `idx_auth_sessions_user_id` index (currently ending line 88):

```python
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS email_change_requests (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        new_email TEXT NOT NULL,
                        stage TEXT NOT NULL,
                        code_hash TEXT NOT NULL,
                        attempts INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        used_at TEXT,
                        cancelled_at TEXT
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_email_change_requests_user_id
                    ON email_change_requests(user_id)
                    """
                )
```

- [ ] **Step 4: Add the store methods**

In `dashboard/backend/users_postgres.py`, insert after `update_display_name` (added in Task 4):

```python
    def _email_change_expiry(self) -> str:
        return (
            (_utcnow() + timedelta(minutes=EMAIL_CHANGE_TTL_MINUTES))
            .replace(microsecond=0)
            .isoformat()
        )

    def create_email_change_request(
        self, user_id: int, new_email: str, code_hash: str
    ) -> Dict[str, Any]:
        """Replace any in-flight request for this user with a fresh stage-'old' one."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM email_change_requests WHERE user_id = %s", (user_id,)
                )
                cur.execute(
                    """
                    INSERT INTO email_change_requests
                        (user_id, new_email, stage, code_hash, created_at, expires_at)
                    VALUES (%s, %s, 'old', %s, %s, %s)
                    RETURNING *
                    """,
                    (
                        user_id,
                        new_email.strip().lower(),
                        code_hash,
                        _utcnow_iso(),
                        self._email_change_expiry(),
                    ),
                )
                row = cur.fetchone()
        return dict(row)

    def get_active_email_change(self, user_id: int) -> Optional[Dict[str, Any]]:
        """The user's in-flight request, or None if absent, used, cancelled, or expired."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM email_change_requests
                    WHERE user_id = %s AND used_at IS NULL AND cancelled_at IS NULL
                    ORDER BY id DESC LIMIT 1
                    """,
                    (user_id,),
                )
                row = cur.fetchone()
        if not row or is_expired(row["expires_at"]):
            return None
        return dict(row)

    def advance_email_change(self, request_id: int, code_hash: str) -> Dict[str, Any]:
        """Move a verified stage-'old' request to stage 'new' with a fresh code."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE email_change_requests
                    SET stage = 'new', code_hash = %s, attempts = 0, expires_at = %s
                    WHERE id = %s
                    RETURNING *
                    """,
                    (code_hash, self._email_change_expiry(), request_id),
                )
                row = cur.fetchone()
        if not row:
            raise ValueError("email_change_request_not_found")
        return dict(row)

    def record_email_change_attempt(self, request_id: int) -> int:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE email_change_requests
                    SET attempts = attempts + 1
                    WHERE id = %s
                    RETURNING attempts
                    """,
                    (request_id,),
                )
                row = cur.fetchone()
        if not row:
            raise ValueError("email_change_request_not_found")
        return int(row["attempts"])

    def mark_email_change_used(self, request_id: int) -> None:
        """Retire a completed request without deleting it (see the SQLite twin)."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE email_change_requests SET used_at = %s WHERE id = %s",
                    (_utcnow_iso(), request_id),
                )

    def cancel_email_change(self, user_id: int) -> None:
        """Deactivate without deleting (see the SQLite twin's cancel_email_change)."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE email_change_requests SET cancelled_at = %s WHERE user_id = %s",
                    (_utcnow_iso(), user_id),
                )

    def last_email_change_request_at(self, user_id: int) -> Optional[str]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT created_at FROM email_change_requests
                    WHERE user_id = %s ORDER BY id DESC LIMIT 1
                    """,
                    (user_id,),
                )
                row = cur.fetchone()
        return str(row["created_at"]) if row else None

    def update_email(self, user_id: int, new_email: str) -> Dict[str, Any]:
        # .lower() is mandatory, not stylistic: this twin's users.email UNIQUE is
        # case-SENSITIVE (the SQLite twin's is COLLATE NOCASE), so skipping it
        # would let two casings of one address coexist in prod while being
        # rejected locally -- twin drift no SQLite-only test run can see.
        normalized_email = new_email.strip().lower()
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE users SET email = %s WHERE id = %s RETURNING *",
                        (normalized_email, user_id),
                    )
                    row = cur.fetchone()
        except psycopg.errors.UniqueViolation as exc:
            raise ValueError("email_already_registered") from exc
        if not row:
            raise ValueError("user_not_found")
        return public_user(row)
```

- [ ] **Step 5: Confirm the twins expose the same surface**

Append to `dashboard/backend/tests/test_users_postgres.py` — this one runs **without** Postgres, so it is the real local guard against twin drift:

```python
def test_store_twins_expose_the_same_public_surface():
    """A method in one twin and not the other is a prod-only crash.

    This compares classes, not instances, so it needs no live Postgres -- which
    matters because every @pg_only case above fails open when TEST_POSTGRES_URL
    is unset.
    """
    import dashboard.backend.users as users_module
    import dashboard.backend.users_postgres as users_postgres_module

    def public_methods(cls):
        return {
            name
            for name in dir(cls)
            if not name.startswith("_") and callable(getattr(cls, name))
        }

    sqlite_methods = public_methods(users_module.UserStore)
    postgres_methods = public_methods(users_postgres_module.PostgresUserStore)
    assert sqlite_methods == postgres_methods
```

- [ ] **Step 6: Run to verify**

Run: `python -m pytest dashboard/backend/tests/test_users_postgres.py -q`
Expected: `test_store_twins_expose_the_same_public_surface` **passes**; the `@pg_only` cases skip.

The Postgres tier is verified on CI, not here. After pushing, confirm it actually ran:

```bash
gh run list --branch feat/account-identity-editing --limit 1
gh run view <run-id> --log | grep -c "test_email_change_request_lifecycle_postgres"
```

- [ ] **Step 7: Commit**

```bash
git status --short dashboard/storage/data/backtest.db   # must print nothing
git add dashboard/backend/users_postgres.py dashboard/backend/tests/test_users_postgres.py
git commit -m "feat(auth): mirror email-change store in the Postgres twin"
```

---

## Task 8: Email-change routes — request, status, cancel

**Files:**
- Modify: `dashboard/backend/api/auth.py`
- Modify: `dashboard/backend/tests/test_auth.py`
- Modify: `dashboard/backend/tests/test_app_composition.py`

**Interfaces:**
- Consumes: Task 1's `send_email`, Task 3's `generate_code`/`hash_code`, Task 6/7's store methods.
- Produces: `POST /api/auth/email-change` → `{"stage": "old", "new_email": str}`; `GET /api/auth/email-change` → `{"pending": bool, "stage": str|None, "new_email": str|None, "expires_at": str|None}`; `DELETE /api/auth/email-change` → `{"status": "ok"}`. Also `_email_change_body(code, new_email) -> str`, reused by Task 9.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/backend/tests/test_auth.py`. Add this shared fixture first — Task 9 reuses it:

```python
class _Outbox(list):
    """Captured messages, plus a switch to make sending start failing.

    Subclasses list rather than pairing a bare list with a flag: a plain list
    rejects attribute assignment (`outbox.ok = False` raises AttributeError),
    and the tests read better asserting on the outbox directly.
    """

    ok = True

    def fail_sends(self):
        self.ok = False

    def resume_sends(self):
        self.ok = True


@pytest.fixture
def sent_emails(monkeypatch):
    """Capture outbound mail instead of sending it; control success/failure.

    Patches the attribute on the sender module, which is exactly how
    api/auth.py reaches it (`from ...email import sender as email_sender`,
    then `email_sender.send_email(...)`) -- one place to patch, and patching
    it works.
    """
    from dashboard.backend.infrastructure.email import sender as email_sender

    outbox = _Outbox()

    async def _fake_send(to, subject, text_body):
        outbox.append({"to": to, "subject": subject, "body": text_body})
        return outbox.ok

    monkeypatch.setattr(email_sender, "send_email", _fake_send)
    return outbox


def _code_from(email_body):
    """Pull the 6-character code out of a captured message body."""
    import re

    from dashboard.backend.verification_codes import CODE_ALPHABET

    match = re.search(rf"code is: ([{CODE_ALPHABET}]{{6}})", email_body)
    assert match, f"no code found in: {email_body!r}"
    return match.group(1)


def test_email_change_request_mails_the_original_address(client, sent_emails):
    token = _signup_and_token(client, email="orig@example.com")

    response = client.post(
        "/api/auth/email-change",
        headers={"Authorization": f"Bearer {token}"},
        json={"current_password": "orig-sturdy-pw-1", "new_email": "fresh@example.com"},
    )

    assert response.status_code == 200
    assert response.json() == {"stage": "old", "new_email": "fresh@example.com"}
    assert len(sent_emails) == 1
    # The authorizing code goes to the address the user already controls.
    assert sent_emails[0]["to"] == "orig@example.com"
    assert "fresh@example.com" in sent_emails[0]["body"]


def test_email_change_request_rejects_a_wrong_password(client, sent_emails):
    token = _signup_and_token(client, email="wrongpw@example.com")

    response = client.post(
        "/api/auth/email-change",
        headers={"Authorization": f"Bearer {token}"},
        json={"current_password": "not-the-password", "new_email": "fresh@example.com"},
    )

    assert response.status_code == 400
    assert "Current password is incorrect" in response.json()["detail"]
    assert sent_emails == []


def test_email_change_request_rejects_the_current_address(client, sent_emails):
    token = _signup_and_token(client, email="same@example.com")

    response = client.post(
        "/api/auth/email-change",
        headers={"Authorization": f"Bearer {token}"},
        json={"current_password": "orig-sturdy-pw-1", "new_email": "SAME@example.com"},
    )

    assert response.status_code == 400
    assert sent_emails == []


def test_email_change_request_rejects_a_registered_address(client, sent_emails):
    _signup_and_token(client, email="taken@example.com")
    token = _signup_and_token(client, email="mover@example.com")

    response = client.post(
        "/api/auth/email-change",
        headers={"Authorization": f"Bearer {token}"},
        json={"current_password": "orig-sturdy-pw-1", "new_email": "taken@example.com"},
    )

    assert response.status_code == 409
    assert sent_emails == []


def test_email_change_request_is_cooldown_limited(client, sent_emails):
    token = _signup_and_token(client, email="fast@example.com")
    body = {"current_password": "orig-sturdy-pw-1", "new_email": "fresh@example.com"}
    headers = {"Authorization": f"Bearer {token}"}

    assert client.post("/api/auth/email-change", headers=headers, json=body).status_code == 200
    second = client.post("/api/auth/email-change", headers=headers, json=body)

    assert second.status_code == 429
    assert second.headers["Retry-After"] == "60"
    assert len(sent_emails) == 1


def test_email_change_cooldown_survives_cancel_and_resend(client, sent_emails):
    # The bug this guards against: DELETE needs only a session, not the
    # password, so without a fix a caller who knows the password could loop
    # request -> cancel -> request with the cooldown never enforced --
    # mail-bombing the account and burning the shared Brevo daily quota.
    token = _signup_and_token(client, email="bounce@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    body = {"current_password": "orig-sturdy-pw-1", "new_email": "fresh@example.com"}

    assert client.post("/api/auth/email-change", headers=headers, json=body).status_code == 200
    assert client.delete("/api/auth/email-change", headers=headers).status_code == 200

    second = client.post("/api/auth/email-change", headers=headers, json=body)

    assert second.status_code == 429
    assert len(sent_emails) == 1


def test_email_change_request_checks_password_before_cooldown(client, sent_emails):
    # A mistyped password must not burn the one-per-minute allowance.
    token = _signup_and_token(client, email="order@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    client.post(
        "/api/auth/email-change",
        headers=headers,
        json={"current_password": "orig-sturdy-pw-1", "new_email": "fresh@example.com"},
    )
    response = client.post(
        "/api/auth/email-change",
        headers=headers,
        json={"current_password": "wrong", "new_email": "other@example.com"},
    )

    assert response.status_code == 400  # not 429


def test_email_change_request_503s_when_mail_fails_and_persists_nothing(
    client, sent_emails
):
    # Send before persist: a failed send must not burn the cooldown for a code
    # that does not exist.
    token = _signup_and_token(client, email="nomail@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    sent_emails.fail_sends()

    response = client.post(
        "/api/auth/email-change",
        headers=headers,
        json={"current_password": "orig-sturdy-pw-1", "new_email": "fresh@example.com"},
    )

    assert response.status_code == 503
    assert client.get("/api/auth/email-change", headers=headers).json()["pending"] is False


def test_email_change_request_503s_when_the_provider_is_unconfigured(client, capsys):
    # No sent_emails fixture here: exercise the real sender with no credentials.
    token = _signup_and_token(client, email="unconfigured@example.com")

    response = client.post(
        "/api/auth/email-change",
        headers={"Authorization": f"Bearer {token}"},
        json={"current_password": "orig-sturdy-pw-1", "new_email": "fresh@example.com"},
    )

    assert response.status_code == 503
    # Fail-VISIBLE: an operator can tell "not configured" from "provider down".
    # capsys, not caplog -- logger output is invisible in the deployment.
    assert "ERROR" in capsys.readouterr().out


def test_email_change_status_reports_the_pending_request(client, sent_emails):
    token = _signup_and_token(client, email="status@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    assert client.get("/api/auth/email-change", headers=headers).json() == {
        "pending": False,
        "stage": None,
        "new_email": None,
        "expires_at": None,
    }

    client.post(
        "/api/auth/email-change",
        headers=headers,
        json={"current_password": "orig-sturdy-pw-1", "new_email": "fresh@example.com"},
    )
    pending = client.get("/api/auth/email-change", headers=headers).json()

    assert pending["pending"] is True
    assert pending["stage"] == "old"
    assert pending["new_email"] == "fresh@example.com"
    assert pending["expires_at"]


def test_email_change_cancel_clears_the_request(client, sent_emails):
    token = _signup_and_token(client, email="cancel@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    client.post(
        "/api/auth/email-change",
        headers=headers,
        json={"current_password": "orig-sturdy-pw-1", "new_email": "fresh@example.com"},
    )

    assert client.delete("/api/auth/email-change", headers=headers).status_code == 200
    assert client.get("/api/auth/email-change", headers=headers).json()["pending"] is False


@pytest.mark.parametrize(
    "method,path",
    [
        ("post", "/api/auth/email-change"),
        ("get", "/api/auth/email-change"),
        ("delete", "/api/auth/email-change"),
    ],
)
def test_email_change_routes_require_auth(client, method, path):
    response = getattr(client, method)(path, json={})
    assert response.status_code == 401
```

The verify route is deliberately absent from that list — it does not exist until Task 9, and an unrouted path answers 404, not 401. Task 9 adds the row back.

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest dashboard/backend/tests/test_auth.py -q -k email_change`
Expected: FAIL — 404/405 (routes do not exist)

- [ ] **Step 3: Add imports and request models**

In `dashboard/backend/api/auth.py`, extend the import block at the top:

```python
from datetime import datetime, timezone
```

and beside the existing first-party imports:

```python
from dashboard.backend.infrastructure.email import sender as email_sender
from dashboard.backend.users import parse_stored_timestamp, public_user, verify_password
from dashboard.backend.verification_codes import generate_code, hash_code
```

(the `public_user, verify_password` import from Task 2 is being extended, not duplicated).

After `DisplayNameRequest` (Task 5), add:

```python
class EmailChangeRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_email: str = Field(min_length=3, max_length=254)

    @field_validator("new_email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        return _normalize_email(value)


class EmailChangeVerifyRequest(BaseModel):
    code: str = Field(min_length=1, max_length=32)
```

- [ ] **Step 4: Add the shared helpers**

After the models, add:

```python
def _seconds_since(timestamp: str) -> float:
    return (
        datetime.now(timezone.utc) - parse_stored_timestamp(timestamp)
    ).total_seconds()


def _email_change_body(code: str, new_email: str) -> str:
    return (
        "Someone asked to change the email address on your Agentic Trading Lab "
        f"account to {new_email}.\n\n"
        f"Your confirmation code is: {code}\n\n"
        f"It expires in {users_module.EMAIL_CHANGE_TTL_MINUTES} minutes. If this "
        "was not you, ignore this message and change your password."
    )
```

- [ ] **Step 5: Add the three routes**

Append to `dashboard/backend/api/auth.py`, after the display-name route:

```python
@router.post("/email-change")
async def request_email_change(
    payload: EmailChangeRequest,
    current_user: dict = Depends(get_current_user),
):
    store = users_module.user_store
    if not verify_password(payload.current_password, current_user["password_hash"]):
        raise HTTPException(status_code=400, detail="Current password is incorrect.")
    if payload.new_email == str(current_user["email"]).strip().lower():
        raise HTTPException(status_code=400, detail="That is already your email address.")
    # This 409 is an account-enumeration oracle, and that is accepted: POST
    # /signup already answers the same question unauthenticated and unlimited.
    # It runs BEFORE the cooldown check below, so cooldown does not bound it --
    # what bounds it is that this path additionally requires a valid session
    # and the account's own password, unlike signup. Failing here beats walking
    # someone through two codes only to 409 at commit -- the commit-time check
    # stays as the TOCTOU backstop.
    if store.get_user_by_email(payload.new_email):
        raise HTTPException(status_code=409, detail="Email is already registered")

    # Cooldown AFTER the password check, so a typo does not burn the allowance.
    last_at = store.last_email_change_request_at(current_user["id"])
    cooldown = users_module.EMAIL_CHANGE_COOLDOWN_SECONDS
    if last_at and _seconds_since(last_at) < cooldown:
        raise HTTPException(
            status_code=429,
            detail="Please wait a minute before requesting another code.",
            headers={"Retry-After": str(cooldown)},
        )

    code = generate_code()
    # Send BEFORE persisting. Persisting first and then failing to send would
    # burn the cooldown on a code that does not exist.
    sent = await email_sender.send_email(
        to=str(current_user["email"]),
        subject="Confirm your Agentic Trading Lab email change",
        text_body=_email_change_body(code, payload.new_email),
    )
    if not sent:
        raise HTTPException(
            status_code=503,
            detail="Could not send the confirmation email. Please try again later.",
        )
    store.create_email_change_request(
        current_user["id"], payload.new_email, hash_code(code)
    )
    return {"stage": "old", "new_email": payload.new_email}


@router.get("/email-change")
async def get_email_change(current_user: dict = Depends(get_current_user)):
    """Let a reloaded page pick the flow back up instead of stranding the user."""
    row = users_module.user_store.get_active_email_change(current_user["id"])
    if not row:
        return {"pending": False, "stage": None, "new_email": None, "expires_at": None}
    return {
        "pending": True,
        "stage": row["stage"],
        "new_email": row["new_email"],
        "expires_at": str(row["expires_at"]),
    }


@router.delete("/email-change")
async def cancel_email_change(current_user: dict = Depends(get_current_user)):
    """Cancel a pending change. Also the resend path: cancel, then start again,
    which re-verifies the password.

    Store-level cancel deactivates rather than deletes, so a caller cannot use
    this (session-only, no password) to reset the 60-second request cooldown.
    """
    users_module.user_store.cancel_email_change(current_user["id"])
    return {"status": "ok"}
```

- [ ] **Step 6: Update the route-contract freeze**

Add exactly these three tuples to `EXPECTED_FULL_CONTRACT` in `dashboard/backend/tests/test_app_composition.py`, beside the display-name entry from Task 5:

```python
    ("POST", "/api/auth/email-change"),
    ("GET", "/api/auth/email-change"),
    ("DELETE", "/api/auth/email-change"),
```

**Do not add the `/email-change/verify` tuple yet.** The frozen set is compared for equality against the app's *actual* routes, so an entry for a route that does not exist turns `test_full_route_contract_unchanged` red just as surely as a missing one. Task 9 adds the route and its tuple together.

For the same reason, the parametrized auth-required test in Step 1 must omit its verify row in this commit — the route would 404, not 401. Ship it with three rows here:

```python
@pytest.mark.parametrize(
    "method,path",
    [
        ("post", "/api/auth/email-change"),
        ("get", "/api/auth/email-change"),
        ("delete", "/api/auth/email-change"),
    ],
)
```

Task 9 restores the fourth row.

- [ ] **Step 7: Run to verify they pass**

Run: `python -m pytest dashboard/backend/tests/test_auth.py dashboard/backend/tests/test_app_composition.py -q`
Expected: all pass (verify-route cases are not present yet)

- [ ] **Step 8: Commit**

```bash
git status --short dashboard/storage/data/backtest.db   # must print nothing
git add dashboard/backend/api/auth.py dashboard/backend/tests/
git commit -m "feat(auth): add email-change request, status and cancel routes"
```

---

## Task 9: Email-change verify route + password-change cancellation (D7)

**Files:**
- Modify: `dashboard/backend/api/auth.py`
- Modify: `dashboard/backend/tests/test_auth.py`
- Modify: `dashboard/backend/tests/test_app_composition.py`

**Interfaces:**
- Consumes: everything from Tasks 1, 3, 6, 7, 8.
- Produces: `POST /api/auth/email-change/verify` → at stage `old`, `{"stage": "new", "new_email": str}`; at stage `new`, `{"status": "ok", "user": {...}}`.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/backend/tests/test_auth.py`:

```python
def _start_email_change(client, token, new_email="fresh@example.com"):
    response = client.post(
        "/api/auth/email-change",
        headers={"Authorization": f"Bearer {token}"},
        json={"current_password": "orig-sturdy-pw-1", "new_email": new_email},
    )
    assert response.status_code == 200, response.text
    return response


def test_email_change_full_two_stage_happy_path(client, sent_emails):
    token = _signup_and_token(client, email="two@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    _start_email_change(client, token)

    first_code = _code_from(sent_emails[0]["body"])
    stage_two = client.post(
        "/api/auth/email-change/verify", headers=headers, json={"code": first_code}
    )

    assert stage_two.status_code == 200
    assert stage_two.json() == {"stage": "new", "new_email": "fresh@example.com"}
    # The second code goes to the NEW address -- that is the reachability proof.
    assert len(sent_emails) == 2
    assert sent_emails[1]["to"] == "fresh@example.com"

    second_code = _code_from(sent_emails[1]["body"])
    done = client.post(
        "/api/auth/email-change/verify", headers=headers, json={"code": second_code}
    )

    assert done.status_code == 200
    assert done.json()["status"] == "ok"
    assert done.json()["user"]["email"] == "fresh@example.com"
    # Durable, and the old address no longer signs in.
    assert client.post(
        "/api/auth/login",
        json={"email": "fresh@example.com", "password": "orig-sturdy-pw-1"},
    ).status_code == 200
    assert client.post(
        "/api/auth/login",
        json={"email": "two@example.com", "password": "orig-sturdy-pw-1"},
    ).status_code == 401


def test_email_change_verify_accepts_a_lowercase_code(client, sent_emails):
    token = _signup_and_token(client, email="lower@example.com")
    _start_email_change(client, token)

    code = _code_from(sent_emails[0]["body"]).lower()
    response = client.post(
        "/api/auth/email-change/verify",
        headers={"Authorization": f"Bearer {token}"},
        json={"code": code},
    )

    assert response.status_code == 200


def test_email_change_verify_rejects_a_wrong_code(client, sent_emails):
    token = _signup_and_token(client, email="badcode@example.com")
    _start_email_change(client, token)

    response = client.post(
        "/api/auth/email-change/verify",
        headers={"Authorization": f"Bearer {token}"},
        json={"code": "ZZZZZZ"},
    )

    assert response.status_code == 400
    assert "not correct" in response.json()["detail"]


def test_email_change_verify_gives_up_after_five_attempts(client, sent_emails):
    token = _signup_and_token(client, email="attempts@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    _start_email_change(client, token)
    real_code = _code_from(sent_emails[0]["body"])
    wrong = "ZZZZZZ" if real_code != "ZZZZZZ" else "YYYYYY"

    for _ in range(4):
        assert client.post(
            "/api/auth/email-change/verify", headers=headers, json={"code": wrong}
        ).status_code == 400

    fifth = client.post(
        "/api/auth/email-change/verify", headers=headers, json={"code": wrong}
    )
    assert fifth.status_code == 400
    assert "start the email change again" in fifth.json()["detail"].lower()

    # The request is gone -- even the correct code is dead now.
    assert client.get("/api/auth/email-change", headers=headers).json()["pending"] is False
    assert client.post(
        "/api/auth/email-change/verify", headers=headers, json={"code": real_code}
    ).status_code == 400


def test_email_change_verify_rejects_an_expired_request(client, sent_emails):
    from dashboard.backend.users import _utcnow

    token = _signup_and_token(client, email="expired@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    _start_email_change(client, token)
    code = _code_from(sent_emails[0]["body"])

    # The `client` fixture patched users_module.user_store to the temp store,
    # so this reaches exactly the database the route just wrote to.
    from dashboard.backend import users as users_module

    stale = (_utcnow() - timedelta(minutes=1)).replace(microsecond=0).isoformat()
    conn = users_module.user_store._get_connection()
    conn.execute("UPDATE email_change_requests SET expires_at = ?", (stale,))
    conn.commit()
    conn.close()

    response = client.post(
        "/api/auth/email-change/verify", headers=headers, json={"code": code}
    )
    assert response.status_code == 400
    assert "no email change" in response.json()["detail"].lower()


def test_email_change_verify_without_a_request_400s(client):
    token = _signup_and_token(client, email="norequest@example.com")

    response = client.post(
        "/api/auth/email-change/verify",
        headers={"Authorization": f"Bearer {token}"},
        json={"code": "ABC234"},
    )

    assert response.status_code == 400


def test_email_change_stage_two_mail_failure_leaves_stage_old_intact(
    client, sent_emails
):
    # Send before persist, the case that matters most: if stage 'new' were
    # written first and the send then failed, the user would be waiting on a
    # code that never went out while the code they DO hold stopped working.
    token = _signup_and_token(client, email="stuck@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    _start_email_change(client, token)
    code = _code_from(sent_emails[0]["body"])
    sent_emails.fail_sends()

    response = client.post(
        "/api/auth/email-change/verify", headers=headers, json={"code": code}
    )

    assert response.status_code == 503
    # Still stage 'old' -- nothing was persisted ahead of the failed send.
    assert client.get("/api/auth/email-change", headers=headers).json()["stage"] == "old"

    # And this is the point of the ordering: the code the user already holds is
    # still valid, so once mail recovers they simply resubmit it. No dead end.
    sent_emails.resume_sends()
    retry = client.post(
        "/api/auth/email-change/verify", headers=headers, json={"code": code}
    )
    assert retry.status_code == 200
    assert retry.json()["stage"] == "new"


def test_email_change_commit_conflicts_when_the_address_was_taken_meanwhile(
    client, sent_emails
):
    # TOCTOU backstop: the request-time 409 cannot cover a signup that lands
    # between the two stages.
    token = _signup_and_token(client, email="race@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    _start_email_change(client, token, new_email="contested@example.com")

    first_code = _code_from(sent_emails[0]["body"])
    client.post("/api/auth/email-change/verify", headers=headers, json={"code": first_code})
    second_code = _code_from(sent_emails[1]["body"])

    _signup_and_token(client, email="contested@example.com")

    response = client.post(
        "/api/auth/email-change/verify", headers=headers, json={"code": second_code}
    )
    assert response.status_code == 409


def test_email_change_commit_revokes_other_sessions_but_keeps_the_caller(
    client, sent_emails
):
    token_a = _signup_and_token(client, email="sessions@example.com")
    token_b = client.post(
        "/api/auth/login",
        json={"email": "sessions@example.com", "password": "orig-sturdy-pw-1"},
    ).json()["token"]
    headers = {"Authorization": f"Bearer {token_a}"}
    _start_email_change(client, token_a)

    client.post(
        "/api/auth/email-change/verify",
        headers=headers,
        json={"code": _code_from(sent_emails[0]["body"])},
    )
    client.post(
        "/api/auth/email-change/verify",
        headers=headers,
        json={"code": _code_from(sent_emails[1]["body"])},
    )

    assert client.get("/api/auth/me", headers=headers).status_code == 200
    assert client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {token_b}"}
    ).status_code == 401


def test_changing_the_password_cancels_a_pending_email_change(client, sent_emails):
    # D7: a user who suspects compromise changes their password; an attacker's
    # in-flight email change must die with it.
    token = _signup_and_token(client, email="d7@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    _start_email_change(client, token)
    assert client.get("/api/auth/email-change", headers=headers).json()["pending"] is True

    assert client.post(
        "/api/auth/change-password",
        headers=headers,
        json={
            "current_password": "orig-sturdy-pw-1",
            "new_password": "new-sturdy-pw-2",
        },
    ).status_code == 200

    assert client.get("/api/auth/email-change", headers=headers).json()["pending"] is False
```

Also add `from datetime import timedelta` to the test module's imports, and restore the verify entry in the parametrized auth-required test from Task 8:

```python
        ("post", "/api/auth/email-change/verify"),
```

Both new helpers used above (`_start_email_change`, and `_Outbox.resume_sends`) come from Task 8's fixture block — no separate setup needed.

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest dashboard/backend/tests/test_auth.py -q -k "verify or two_stage or d7 or cancels"`
Expected: FAIL — 404/405 on the verify route

- [ ] **Step 3: Add the verify route**

Append to `dashboard/backend/api/auth.py`, after the cancel route:

```python
@router.post("/email-change/verify")
async def verify_email_change(
    payload: EmailChangeVerifyRequest,
    current_user: dict = Depends(get_current_user),
    authorization: Optional[str] = Header(default=None),
):
    """One stage-driven endpoint, not two.

    The server already knows which stage is outstanding; separate
    verify-current and confirm endpoints would only give the client a way to
    call the wrong one.
    """
    store = users_module.user_store
    request_row = store.get_active_email_change(current_user["id"])
    if not request_row:
        raise HTTPException(
            status_code=400, detail="No email change is in progress. Start again."
        )

    if hash_code(payload.code) != request_row["code_hash"]:
        attempts = store.record_email_change_attempt(request_row["id"])
        if attempts >= users_module.EMAIL_CHANGE_MAX_ATTEMPTS:
            store.cancel_email_change(current_user["id"])
            raise HTTPException(
                status_code=400,
                detail="Too many incorrect codes. Start the email change again.",
            )
        raise HTTPException(status_code=400, detail="That code is not correct.")

    new_email = str(request_row["new_email"])

    if request_row["stage"] == "old":
        code = generate_code()
        # Send BEFORE persisting stage 'new'. The other order strands the user:
        # waiting on a code that was never delivered, while the code they do
        # hold is no longer accepted, with Cancel the only exit and nothing on
        # screen to explain it. Failing here leaves stage 'old' untouched, so
        # they can simply resubmit the code they already have.
        sent = await email_sender.send_email(
            to=new_email,
            subject="Confirm your new Agentic Trading Lab email address",
            text_body=_email_change_body(code, new_email),
        )
        if not sent:
            raise HTTPException(
                status_code=503,
                detail="Could not send the confirmation email. Please try again.",
            )
        store.advance_email_change(request_row["id"], hash_code(code))
        return {"stage": "new", "new_email": new_email}

    try:
        user = store.update_email(current_user["id"], new_email)
    except ValueError as exc:
        if str(exc) == "email_already_registered":
            store.cancel_email_change(current_user["id"])
            raise HTTPException(
                status_code=409, detail="Email is already registered"
            ) from exc
        raise HTTPException(
            status_code=401, detail="Session is no longer valid."
        ) from exc

    store.mark_email_change_used(request_row["id"])
    # Best-effort, exactly as in change-password: an email change is an identity
    # change, so other sessions end -- but the durable write already landed, so a
    # revocation failure is a WARNING, not a 500. ERROR is reserved for the mail
    # failures above, where the user genuinely gets nothing.
    try:
        store.delete_other_sessions(
            current_user["id"], keep_token=_extract_bearer_token(authorization)
        )
    except Exception as exc:  # noqa: BLE001 -- email change already committed
        print(
            f"WARNING: email change committed for user {current_user['id']} but "
            f"other-session revocation failed: {exc!r}"
        )
    return {"status": "ok", "user": user}
```

- [ ] **Step 4: Wire D7 into change-password**

In the `change_password` route, insert immediately before `return {"status": "ok"}`:

```python
    # D7: a user changing their password may be reacting to a compromise, so an
    # attacker's in-flight email change dies with it. Best-effort and next to
    # the session revocation above, so the whole "invalidate what the old
    # password could reach" policy sits in one place.
    try:
        users_module.user_store.cancel_email_change(current_user["id"])
    except Exception as exc:  # noqa: BLE001 -- password change already committed
        print(
            f"WARNING: change-password committed for user {current_user['id']} but "
            f"cancelling the pending email change failed: {exc!r}"
        )
```

- [ ] **Step 5: Add the last contract entry**

In `dashboard/backend/tests/test_app_composition.py`:

```python
    ("POST", "/api/auth/email-change/verify"),
```

- [ ] **Step 6: Run to verify they pass**

Run: `python -m pytest dashboard/backend/tests/test_auth.py dashboard/backend/tests/test_app_composition.py -q`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git status --short dashboard/storage/data/backtest.db   # must print nothing
git add dashboard/backend/api/auth.py dashboard/backend/tests/
git commit -m "feat(auth): add stage-driven email-change verification"
```

---

## Task 10: Part C — move and recolour the logout button

**Files:**
- Modify: `dashboard/frontend/app.html` (remove lines 1505-1507; re-insert before line 1541)
- Modify: `dashboard/frontend/styles.css` (insert after line 391)
- Create: `dashboard/backend/tests/test_frontend_account_page.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `.auth-btn-danger` / `.auth-btn-danger:hover` in `styles.css`. `id="authLogoutBtn"` is preserved verbatim — `app.js:2517` reads it by id and `:2556-2558` binds the handler, both inside `initAuthUI()`. No JS change.

**Re-read these before editing** — cache-bust values and account-card line numbers are a standing merge-conflict magnet:

```bash
grep -n 'styles.css?v=\|app.js?v=' dashboard/frontend/app.html
grep -n 'id="accountSignedIn"\|id="authLogoutBtn"' dashboard/frontend/app.html
```

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/test_frontend_account_page.py`:

```python
"""Account-page markup and cascade guards.

The frontend has no JS test harness, and these two contracts are structural
rather than behavioural -- an ordering and a CSS source-order requirement -- so
they are asserted against the shipped source directly.
"""

import re
from pathlib import Path

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_APP_HTML = _FRONTEND / "app.html"
_STYLES_CSS = _FRONTEND / "styles.css"


def _account_card() -> str:
    html = _APP_HTML.read_text(encoding="utf-8")
    start = html.index('<div id="accountSignedIn"')
    end = html.index('<div id="accountSignedOut"')
    return html[start:end]


def test_logout_button_is_last_in_the_account_card():
    card = _account_card()
    logout_at = card.index('id="authLogoutBtn"')

    for marker in ("id=\"avatarUploadBtn\"", "id=\"changePasswordForm\""):
        assert card.index(marker) < logout_at, f"{marker} must come before Log out"

    # "after those two" is not "last" -- a section appended later would keep the
    # assertions above green. Nothing else in the card may carry an id.
    tail = card[logout_at + len('id="authLogoutBtn"'):]
    assert 'id="' not in tail, f"something with an id follows Log out: {tail!r}"


def test_logout_button_carries_the_danger_class():
    card = _account_card()
    match = re.search(r'<button[^>]*id="authLogoutBtn"[^>]*>', card)
    assert match, "logout button not found in the account card"
    tag = match.group(0)
    # Assert on the button's OWN tag. A substring search over the whole card
    # would pass if "auth-btn-danger" appeared anywhere else, and a fixed-width
    # window before the id cannot see the class at all -- this file's markup
    # puts id= before class=.
    assert "auth-btn-danger" in tag
    assert "auth-btn-secondary" not in tag


def test_header_dropdown_logout_is_untouched():
    # The brief targeted the account-page button only. Removing the dropdown
    # item would also make docs/source/lab/accounts.rst factually wrong.
    html = _APP_HTML.read_text(encoding="utf-8")
    assert 'id="accountMenuLogoutBtn"' in html


def test_auth_btn_danger_is_declared_after_the_generic_hover():
    """.auth-btn:hover and .auth-btn-danger:hover both score (0,2,0).

    With identical specificity, source order alone decides. Declared earlier,
    the logout button reverts to info-blue on hover -- red at rest, wrong on
    mouseover, which a screenshot taken at rest would not catch.
    """
    css = _STYLES_CSS.read_text(encoding="utf-8")
    assert css.index(".auth-btn-danger:hover") > css.index(".auth-btn:hover")
    assert css.index(".auth-btn-danger {") > css.index(".auth-btn {")
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest dashboard/backend/tests/test_frontend_account_page.py -q`
Expected: FAIL — `auth-btn-danger` is not in the card, and `.index()` raises `ValueError` for the CSS selectors

- [ ] **Step 3: Move the button in `app.html`**

Delete lines 1505-1507:

```html
            <div class="account-actions">
                <button id="authLogoutBtn" class="auth-btn auth-btn-secondary" type="button">Log out</button>
            </div>
```

and re-insert, as the last child of `#accountSignedIn` (immediately before the `</div>` that closes it, currently line 1541):

```html
            <div class="account-section account-actions">
                <button id="authLogoutBtn" class="auth-btn auth-btn-danger" type="button">Log out</button>
            </div>
```

Both classes are kept deliberately: `.account-section` supplies the `border-top` divider that separates it from Change password, `.account-actions` keeps the flex container for any future second control. They both set `margin-top`; `.account-section` (line ~9101) wins over `.account-actions` (line ~4694) on source order, giving 16px. That is intended.

- [ ] **Step 4: Add the CSS**

In `dashboard/frontend/styles.css`, insert after the `.auth-btn-secondary` block (currently ending line 391), before `.auth-modal[hidden]`:

```css
/* Destructive-control red. MUST stay after .auth-btn:hover above: the two
   hover rules have identical (0,2,0) specificity, so source order is the only
   thing making this one win. A copy of .agent-delete-btn's palette rather than
   a reuse of it -- that class is agent-scoped, and the codebase already makes
   this same "same look, own class" split between .account-menu-item--danger
   and .agent-menu-item--danger. Note the repo carries two reds: #f87171 for
   destructive controls, --danger-color #ff4141 for negative P&L. */
.auth-btn-danger {
    border-color: rgba(248, 113, 113, 0.55);
    background: rgba(248, 113, 113, 0.1);
    color: #f87171;
}

.auth-btn-danger:hover {
    border-color: #f87171;
    background: rgba(248, 113, 113, 0.2);
    color: #fca5a5;
}
```

- [ ] **Step 5: Run to verify they pass**

Run: `python -m pytest dashboard/backend/tests/test_frontend_account_page.py -q`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git status --short dashboard/storage/data/backtest.db   # must print nothing
git add dashboard/frontend/app.html dashboard/frontend/styles.css dashboard/backend/tests/test_frontend_account_page.py
git commit -m "feat(account): move log out to the bottom in destructive red"
```

---

## Task 11: Frontend editors for display name and email

**Files:**
- Modify: `dashboard/frontend/app.html`
- Modify: `dashboard/frontend/app.js`
- Modify: `dashboard/frontend/styles.css`
- Modify: `dashboard/backend/tests/test_frontend_account_page.py`

**Interfaces:**
- Consumes: the five routes from Tasks 5, 8, 9.
- Produces: `initDisplayNameForm()` and `initEmailChangeForm()`, registered in the initializer block at the tail of `initAuthUI()`.

**`applyUpdatedUser(user)` already calls `updateAuthUI()`, which calls `updateAccountPage()` at its own tail** — confirmed at source (`app.js`, `updateAuthUI`'s last statement before the `refreshHomeModules` hook). `initAvatarControls()`'s existing success handler already relies on exactly this cascade without an explicit follow-up call. So neither new success path below needs to call `updateAccountPage()` again after `applyUpdatedUser(...)` — doing so only double-renders harmlessly, and the existing convention in this file is not to.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/backend/tests/test_frontend_account_page.py`:

```python
def test_account_card_section_order():
    card = _account_card()
    order = [
        'id="accountDisplayName"',      # read-only summary row
        'id="accountEmail"',            # read-only summary row
        'id="accountDisplayNameForm"',  # editor
        'id="accountEmailForm"',        # editor
        'id="avatarUploadBtn"',
        'id="changePasswordForm"',
        'id="authLogoutBtn"',
    ]
    positions = [card.index(marker) for marker in order]
    assert positions == sorted(positions), "account card sections are out of order"


def test_email_change_copy_mentions_the_spam_folder():
    """An unauthenticated single sender has materially degraded inbox placement,
    and a code silently in spam is indistinguishable from one never sent."""
    js = (_FRONTEND / "app.js").read_text(encoding="utf-8")
    assert js.lower().count("spam folder") >= 2  # one line per stage


def test_cache_bust_versions_were_bumped():
    html = _APP_HTML.read_text(encoding="utf-8")
    assert "styles.css?v=65" in html
    assert "app.js?v=48" in html
```

**Before running:** re-read the two cache-bust lines and set the assertions to *one above whatever `main` actually carries* — they moved from 63/44 to 64/47 between the spec being drafted and PR #227 landing on `main` afterward, so `app.js` is **already** at `?v=47`; the bump target below is `48`, not `47`. Do not assume the numbers in this plan are still one below current `main`.

```bash
grep -n 'styles.css?v=\|app.js?v=' dashboard/frontend/app.html
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest dashboard/backend/tests/test_frontend_account_page.py -q`
Expected: FAIL — `accountDisplayNameForm` not found

- [ ] **Step 3: Add the two sections to `app.html`**

Insert immediately after the read-only Email `.account-row` (currently ending line 1504) and before the Profile photo section:

```html
            <div class="account-section">
                <h3 class="account-section-title">Display name</h3>
                <form id="accountDisplayNameForm" class="auth-form account-password-form">
                    <label class="auth-field">
                        <span>Display name</span>
                        <input id="displayNameInput" type="text" maxlength="100" autocomplete="nickname" required>
                    </label>
                    <p id="displayNameError" class="auth-error" hidden></p>
                    <p id="displayNameSuccess" class="account-success" hidden>Display name updated.</p>
                    <button class="auth-btn auth-btn-primary" type="submit">Save name</button>
                </form>
            </div>
            <div class="account-section">
                <h3 class="account-section-title">Email address</h3>
                <form id="accountEmailForm" class="auth-form account-password-form">
                    <div id="emailChangeIdle">
                        <label class="auth-field">
                            <span>New email address</span>
                            <input id="newEmailInput" type="email" maxlength="254" autocomplete="email">
                        </label>
                        <label class="auth-field">
                            <span>Current password</span>
                            <input id="emailChangePasswordInput" type="password" autocomplete="current-password">
                        </label>
                    </div>
                    <div id="emailChangeCodeStep" hidden>
                        <p id="emailChangeStepCopy" class="account-hint"></p>
                        <label class="auth-field">
                            <span>Confirmation code</span>
                            <input id="emailChangeCodeInput" type="text" maxlength="6" autocomplete="one-time-code">
                        </label>
                    </div>
                    <p id="emailChangeError" class="auth-error" hidden></p>
                    <p id="emailChangeSuccess" class="account-success" hidden>Email address updated. Other devices were signed out.</p>
                    <div class="account-email-actions">
                        <button id="emailChangeSubmitBtn" class="auth-btn auth-btn-primary" type="submit">Send code</button>
                        <button id="emailChangeCancelBtn" class="auth-btn auth-btn-secondary" type="button" hidden>Cancel</button>
                    </div>
                </form>
            </div>
```

Then bump both cache-bust versions (line 12 and the last `<script>`):

```html
    <link rel="stylesheet" href="styles.css?v=65">
    <script src="app.js?v=48"></script>
```

- [ ] **Step 4: Add the supporting CSS**

In `dashboard/frontend/styles.css`, after the `.account-success` block (currently ending near line 9128):

```css
.account-hint {
    margin: 0 0 10px;
    font-size: 12px;
    color: var(--text-secondary);
}

.account-email-actions {
    display: flex;
    gap: 8px;
    align-items: center;
}
```

- [ ] **Step 5: Add the `AuthAPI` methods**

In `dashboard/frontend/app.js`, inside the `AuthAPI` object, after `removeAvatar()` (currently line 2016-2018):

```js
  updateDisplayName(displayName) {
    return this.request('/api/auth/display-name', {
      method: 'PUT',
      body: JSON.stringify({ display_name: displayName }),
    });
  },

  requestEmailChange(currentPassword, newEmail) {
    return this.request('/api/auth/email-change', {
      method: 'POST',
      body: JSON.stringify({ current_password: currentPassword, new_email: newEmail }),
    });
  },

  verifyEmailChange(code) {
    return this.request('/api/auth/email-change/verify', {
      method: 'POST',
      body: JSON.stringify({ code }),
    });
  },

  emailChangeStatus() {
    return this.request('/api/auth/email-change', { method: 'GET' });
  },

  cancelEmailChange() {
    return this.request('/api/auth/email-change', { method: 'DELETE' });
  },
```

- [ ] **Step 6: Seed the display-name input**

In `updateAccountPage()` (currently line 2062), inside the `if (user) {` branch after the `emailEl` line:

```js
    const nameInput = document.getElementById('displayNameInput');
    // Skip while focused so a re-render mid-edit does not stomp what is typed.
    if (nameInput && document.activeElement !== nameInput) {
      nameInput.value = user.display_name || '';
    }
```

- [ ] **Step 7: Add the two init functions**

In `dashboard/frontend/app.js`, after `initChangePasswordForm()` (currently ending line 2257):

```js
function initDisplayNameForm() {
  const form = document.getElementById('accountDisplayNameForm');
  if (!form) return;
  const input = document.getElementById('displayNameInput');
  const errorEl = document.getElementById('displayNameError');
  const successEl = document.getElementById('displayNameSuccess');

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const submitBtn = form.querySelector('button[type="submit"]');
    if (errorEl) errorEl.hidden = true;
    if (successEl) successEl.hidden = true;

    const value = (input?.value || '').trim();
    if (!value) {
      if (errorEl) {
        errorEl.textContent = 'Display name cannot be empty.';
        errorEl.hidden = false;
      }
      return;
    }

    if (submitBtn) submitBtn.disabled = true;
    try {
      const data = await AuthAPI.updateDisplayName(value);
      applyUpdatedUser(data.user);   // cascades into updateAuthUI() -> updateAccountPage()
      if (successEl) successEl.hidden = false;
    } catch (error) {
      if (errorEl) {
        errorEl.textContent = error.message;
        errorEl.hidden = false;
      }
    } finally {
      if (submitBtn) submitBtn.disabled = false;
    }
  });
}

function renderEmailChangeState(state) {
  const idle = document.getElementById('emailChangeIdle');
  const codeStep = document.getElementById('emailChangeCodeStep');
  const copy = document.getElementById('emailChangeStepCopy');
  const submitBtn = document.getElementById('emailChangeSubmitBtn');
  const cancelBtn = document.getElementById('emailChangeCancelBtn');
  if (!idle || !codeStep) return;

  const pending = Boolean(state && state.pending);
  idle.hidden = pending;
  codeStep.hidden = !pending;
  if (cancelBtn) cancelBtn.hidden = !pending;

  if (!pending) {
    if (submitBtn) submitBtn.textContent = 'Send code';
    return;
  }

  const user = getStoredAuthUser();
  if (copy) {
    // textContent, never innerHTML: new_email is user-supplied.
    copy.textContent = state.stage === 'new'
      ? `Code sent to ${state.new_email}. Enter it to finish — check your spam folder if it doesn't arrive.`
      : `We sent a 6-character code to ${user?.email || 'your current address'}. Check your spam folder if it doesn't arrive.`;
  }
  if (submitBtn) submitBtn.textContent = state.stage === 'new' ? 'Confirm' : 'Verify';
}

function initEmailChangeForm() {
  const form = document.getElementById('accountEmailForm');
  if (!form) return;
  const errorEl = document.getElementById('emailChangeError');
  const successEl = document.getElementById('emailChangeSuccess');
  const codeInput = document.getElementById('emailChangeCodeInput');
  const cancelBtn = document.getElementById('emailChangeCancelBtn');
  let stage = null;

  const showError = (message) => {
    if (errorEl) {
      errorEl.textContent = message;
      errorEl.hidden = false;
    }
  };

  const reset = () => {
    stage = null;
    form.reset();
    renderEmailChangeState({ pending: false });
  };

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const submitBtn = document.getElementById('emailChangeSubmitBtn');
    if (errorEl) errorEl.hidden = true;
    if (successEl) successEl.hidden = true;
    if (submitBtn) submitBtn.disabled = true;
    try {
      if (!stage) {
        const newEmail = (document.getElementById('newEmailInput')?.value || '').trim();
        const password = document.getElementById('emailChangePasswordInput')?.value;
        const state = await AuthAPI.requestEmailChange(password, newEmail);
        stage = state.stage;
        renderEmailChangeState({ pending: true, ...state });
        const pwInput = document.getElementById('emailChangePasswordInput');
        if (pwInput) pwInput.value = '';
      } else {
        const data = await AuthAPI.verifyEmailChange(codeInput?.value || '');
        if (data.status === 'ok') {
          applyUpdatedUser(data.user);   // cascades into updateAuthUI() -> updateAccountPage()
          reset();
          if (successEl) successEl.hidden = false;
        } else {
          // Stage advanced: a fresh code just went to the new address.
          stage = data.stage;
          if (codeInput) codeInput.value = '';
          renderEmailChangeState({ pending: true, ...data });
        }
      }
    } catch (error) {
      showError(error.message);
    } finally {
      if (submitBtn) submitBtn.disabled = false;
    }
  });

  cancelBtn?.addEventListener('click', async () => {
    if (errorEl) errorEl.hidden = true;
    try {
      await AuthAPI.cancelEmailChange();
    } catch (error) {
      showError(error.message);
      return;
    }
    reset();
  });

  // Re-entering the page mid-flow must not strand the user on the idle form.
  if (getStoredAuthUser()) {
    AuthAPI.emailChangeStatus()
      .then((state) => {
        stage = state.pending ? state.stage : null;
        renderEmailChangeState(state);
      })
      .catch(() => renderEmailChangeState({ pending: false }));
  }
}
```

- [ ] **Step 8: Register both initializers**

In the tail of `initAuthUI()` (currently lines 2644-2646), change:

```js
  initChangePasswordForm();
  initDisplayNameForm();
  initEmailChangeForm();
  initAvatarControls();
  refreshAuthUser();
```

- [ ] **Step 9: Run to verify they pass**

Run: `python -m pytest dashboard/backend/tests/test_frontend_account_page.py -q`
Expected: 7 passed

- [ ] **Step 10: Verify the page actually renders**

The suite cannot catch a JS syntax error in `app.js`. Parse it:

```bash
node --check dashboard/frontend/app.js
```

Expected: no output (exit 0)

- [ ] **Step 11: Commit**

```bash
git status --short dashboard/storage/data/backtest.db   # must print nothing
git add dashboard/frontend/ dashboard/backend/tests/test_frontend_account_page.py
git commit -m "feat(account): add display name and email editors to the account page"
```

---

## Task 12: Full verification and draft PR

**Files:** none modified (verification only), plus the PR itself.

- [ ] **Step 1: Rebase onto current `origin/main`, then run the entire suite**

`origin/main` has moved since this plan's stated base (`116874e`) — PR #227 and #234 landed
during drafting, touching `api/auth.py`, `test_app_composition.py`, `app.html`, `app.js`,
and `styles.css`, and more may land before Tasks 1-11 finish. Rebase first so "no
regressions" is measured against what this will actually merge onto, not a stale citation:

```bash
git fetch origin main
git rebase origin/main
```

Resolve any conflicts (they will cluster in the five files above) before continuing. Then:

Run: `python -m pytest dashboard/backend/tests/ -q`
Expected: **zero failures.** The suite has been green end-to-end since PR #71, so a red test is a real regression, not a known-flaky.

Record the pass count rather than checking it against a number from memory. To confirm this branch only *added* coverage, compare against a disposable checkout of the (now-current, post-rebase) `main` tip — not a number from memory, and not just the commit hash printed without ever being run:

```bash
git worktree add /mnt/d/github/atl-main-check origin/main
python -m pytest /mnt/d/github/atl-main-check/dashboard/backend/tests/ -q 2>&1 | tail -3
git worktree remove /mnt/d/github/atl-main-check
```

If `test_deleted_shim_is_not_importable` fails with `DID NOT RAISE ModuleNotFoundError`, that is stale bytecode, not a regression: `rm -rf dashboard/backend/engines dashboard/backend/services`.

- [ ] **Step 2: Confirm the seed DB is untouched**

```bash
git status --short dashboard/storage/data/backtest.db
ls dashboard/storage/data/
```

Expected: no output from `git status`. If a `backtest.db-wal` sidecar exists, the lazy `CREATE TABLE email_change_requests` may not have been folded into the tracked binary yet — do **not** checkpoint it; confirm the tracked file is still identical to `main`:

```bash
git diff --stat main -- dashboard/storage/data/backtest.db
```

Expected: no output.

- [ ] **Step 3: Confirm the route contract matches reality**

Run: `python -m pytest dashboard/backend/tests/test_app_composition.py -q`
Expected: pass. All five new tuples present, no extras.

- [ ] **Step 4: Push and open the PR as a draft**

```bash
git push -u origin feat/account-identity-editing
```

Open it **as a draft**, with the gate as the **first line of the body** — `main` has no branch protection and the observed norm is that any collaborator merges any open PR at any moment. A comment is not a gate.

```bash
gh pr create --draft \
  --title "feat: account identity editing" \
  --body "$(cat <<'EOF'
DO NOT MERGE until BREVO_API_KEY / ACCOUNT_EMAIL_FROM are set in the Render dashboard.

Display name editing, email change (current password + a code to the old address, then a code to the new one), and the account-page log out button moved to the bottom in destructive red.

Unset mail credentials mean email change returns 503; display name and logout work regardless.

Closes #185 (auth routes now resolve `user_store` at call time, at both import-time binding sites).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01RAQHik1DjVT9emU2PNXynb
EOF
)"
```

- [ ] **Step 5: Verify the Postgres tier actually ran on CI**

`@pg_only` fails open, so a green local run proves nothing about Task 7.

```bash
gh run list --branch feat/account-identity-editing --limit 3
gh run view <run-id> --log | grep -E "test_email_change_request_lifecycle_postgres|test_update_email_postgres"
```

If the tier skipped on CI too, say so plainly rather than reporting Task 7 as verified.

- [ ] **Step 6: Operator handoff (blocking, and not yours to do)**

Brevo account + sender chosen: `ACCOUNT_EMAIL_FROM=flymiss.privateserver@gmail.com`, a
personal Gmail address (accepting D1's free-mail-sender caveats — Brevo rewrites the
visible From, worse inbox placement than an authenticated domain). `BREVO_API_KEY` is set
locally in the gitignored `dashboard/.env` for dev/internal testing only — it was never
written into this doc or any other git-tracked file. Report to the user that the PR is a
draft pending **their** remaining action:

1. ~~Create a Brevo account, verify a single sender address.~~ Done.
2. Set `BREVO_API_KEY`, `ACCOUNT_EMAIL_FROM=flymiss.privateserver@gmail.com`, and optionally `ACCOUNT_EMAIL_FROM_NAME` in the **Render dashboard** — `render.yaml` is documentation, not the mechanism.
3. Then the draft can be marked ready.

- [ ] **Step 7: Issue follow-ups at merge (ask before filing)**

Filing on a shared repo implicitly assigns work to others — **ask first**.

- **#185** closes with this PR (already in the body).
- **#187** does **not** close, but its stated blocker dissolves. Comment there noting the mail module now exists and that **Brevo** was chosen over the Resend that #187 and the Phase 2 spec both name — so its env-var list (`RESEND_API_KEY`, `RESET_EMAIL_FROM`) needs updating before anyone builds against it.
- **#167** and **#202** are referenced as precedent only; neither closes.

- [ ] **Step 8: Surface the user-facing docs follow-ups**

Do **not** edit these — the user coordinates hosted-doc updates explicitly. Report them as a session-end follow-up:

- `docs/source/lab/accounts.rst:35-46` — "Manage your profile" documents only password change and profile photo. Now **incomplete**: needs display-name and email-change paragraphs, including where each code is sent.
- `docs/source/lab/accounts.rst:20` — describes the display name as set at signup, never mentioning it is editable.
- `docs/source/lab/accounts.rst:31-32` — stays accurate (the dropdown logout is unchanged). Noted so a later reader does not "fix" it.
- Pre-existing and unrelated: the ReadTheDocs protocol page and the PyPI README still state a 30-second decision deadline; it is 60 seconds.

---

## Spec Coverage Check

| Spec section | Task |
|---|---|
| D1 Brevo provider | 1 |
| D2 two codes | 8, 9 |
| D3 one stage-driven verify endpoint | 9 |
| D4 non-blocking HTTP (`httpx.AsyncClient`) | 1 |
| D5 fail-visible mail → 503 | 1, 8, 9 |
| D6 fix #185 at both binding sites | 2 |
| D7 password change cancels pending change | 9 |
| D8 no "email changed" notice | — (deliberately nothing to build) |
| Part A store methods | 4 |
| Part A route | 5 |
| Part B schema, both twins | 6, 7 |
| Part B code format | 3 |
| Part B store methods | 6, 7 |
| Part B routes (4) | 8, 9 |
| Send before persist | 8, 9 |
| On commit: revoke other sessions | 9 |
| Mail module | 1 |
| Part C logout move + CSS | 10 |
| Frontend section order + states | 11 |
| Cache-bust bumps | 11 |
| Route-contract freeze | 5, 8, 9 |
| Testing: store twins, API, mail, frontend | 4, 6, 7, 8, 9, 10, 11 |
| Delivery: draft PR, operator step, seed-DB, `@pg_only` | 12 |
| Issues: #185 closes, #187 comment | 12 |
| User-facing docs follow-ups | 12 |
