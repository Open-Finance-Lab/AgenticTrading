# Admin User Analytics Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the privacy-safe Analytics foundation: durable event storage, strict validation and idempotency, authenticated page-event ingestion, pseudonymous sessions, subject exclusions, Admin profile-access auditing, and bounded retention.

**Architecture:** Analytics lives in a new `dashboard.backend.domain.analytics` package and uses the account database selected by `USERS_DATABASE_URL`, with equivalent SQLite and PostgreSQL stores. The browser sends only allowlisted experience events; the service adds authenticated identity, server timestamps, coarse device/browser data, and an optional monthly HMAC network identifier before persistence. Retention rides the existing run-reaper sweep, so it adds no second scheduler and can fail without changing authentication, Credits, credential, Agent, or backtest outcomes.

**Tech Stack:** Python 3.13, FastAPI, Pydantic v2, SQLite, PostgreSQL/psycopg, vanilla JavaScript, pytest.

**Spec:** `docs/superpowers/specs/2026-08-26-admin-user-analytics-design.md`

## Global Constraints

- No Admin Analytics UI or metrics are included in this PR; PR 1 also ships no Admin Analytics query endpoints.
- Raw `analytics_events` are retained for 180 days; `admin_analytics_access_log` rows are retained for 365 days.
- `analytics_daily_rollups` and current `user_analytics_snapshots` are not deleted by raw-event retention.
- Missing or invalid `ANALYTICS_PSEUDONYMIZATION_KEY` omits `network_hash`; plaintext IP fallback is forbidden.
- Never persist, return, or log API keys, passwords, tokens, prompts, strategy text, portfolio/form input, raw provider bodies, encrypted credential ciphertext, full IP addresses, or raw User-Agent values.
- Frontend ingestion accepts only `page_viewed`, `page_hidden`, and `session_heartbeat` for an explicit page allowlist.
- Authentication supplies `user_id`; the client cannot submit identity, role, email, display name, credential metadata, or billing evidence.
- Analytics failures must not change login, credential, Agent, backtest, model-execution, or Credits results.
- SQLite and PostgreSQL public repository methods and behavior must remain equivalent.
- Tests use synthetic values only and never require or display a real API key.
- Do not commit `.superpowers/`, `work/`, `dashboard/storage/data/backtest.db`, or any local database file.
- Use TDD and stop at the first failing layer; do not continue stacking changes on a failed test.

---

### Task 1: Define the event contract and privacy reducers

**Files:**
- Create: `dashboard/backend/domain/analytics/__init__.py`
- Create: `dashboard/backend/domain/analytics/models.py`
- Create: `dashboard/backend/domain/analytics/privacy.py`
- Create: `dashboard/backend/tests/domain/analytics/__init__.py`
- Create: `dashboard/backend/tests/domain/analytics/test_models.py`
- Create: `dashboard/backend/tests/domain/analytics/test_privacy.py`

**Interfaces:**
- Consumes: Pydantic v2 `BaseModel`/`ConfigDict`/`Field`/`field_validator` and `dashboard.backend.api.rate_limit.client_ip(request)`.
- Produces: `FrontendAnalyticsEvent`, `AnalyticsEventDraft`, `AnalyticsEventRecord`, `AppendEventResult`, `RetentionResult`, `RequestAnalyticsContext`, `sanitize_frontend_properties(event_name, properties)`, `sanitize_server_properties(event_name, properties)`, `monthly_network_hash(ip_address, received_at)`, and `request_analytics_context(request, received_at)`.

- [ ] **Step 1: Write failing model-contract tests**

Create `dashboard/backend/tests/domain/analytics/test_models.py` with exact allowlist and rejection coverage:

```python
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from dashboard.backend.domain.analytics.models import (
    ALLOWED_BILLING_MODES,
    ALLOWED_ERROR_CATEGORIES,
    ALLOWED_EVENT_NAMES,
    ALLOWED_FRONTEND_EVENT_NAMES,
    ALLOWED_PAGE_VIEWS,
    FrontendAnalyticsEvent,
)


def _payload(**overrides):
    value = {
        "event_id": str(uuid4()),
        "schema_version": 1,
        "event_name": "page_viewed",
        "session_id": str(uuid4()),
        "occurred_at": datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc),
        "page_view": "home",
        "properties": {},
    }
    value.update(overrides)
    return value


def test_allowlists_are_closed_and_versioned():
    assert ALLOWED_FRONTEND_EVENT_NAMES == {
        "page_viewed",
        "page_hidden",
        "session_heartbeat",
    }
    assert ALLOWED_PAGE_VIEWS == {
        "home",
        "agents",
        "agent_editor",
        "backtest",
        "paper_trading",
        "competition",
        "community",
        "credits",
        "account",
    }
    assert ALLOWED_BILLING_MODES == {"byok", "platform_credits"}
    assert {
        "credential_invalid",
        "credential_missing",
        "provider_timeout",
        "provider_unavailable",
        "credits_unavailable",
        "model_not_allowed",
        "internal_error",
    } <= ALLOWED_ERROR_CATEGORIES
    assert ALLOWED_FRONTEND_EVENT_NAMES < ALLOWED_EVENT_NAMES


@pytest.mark.parametrize(
    "patch",
    [
        {"schema_version": 2},
        {"event_name": "backtest_completed"},
        {"event_name": "clicked_anything"},
        {"page_view": "admin"},
        {"session_id": "auth-token-shaped-value"},
        {"email": "not-accepted@example.test"},
        {"properties": {"prompt": "must not cross"}},
        {"properties": {"api_key": "synthetic-secret-canary"}},
    ],
)
def test_frontend_event_rejects_unknown_or_sensitive_shape(patch):
    with pytest.raises(ValidationError):
        FrontendAnalyticsEvent.model_validate(_payload(**patch))


def test_frontend_event_accepts_only_bounded_duration_metadata():
    event = FrontendAnalyticsEvent.model_validate(
        _payload(event_name="page_hidden", properties={"visible_ms": 12_500})
    )
    assert event.properties == {"visible_ms": 12_500}
    with pytest.raises(ValidationError):
        FrontendAnalyticsEvent.model_validate(
            _payload(event_name="page_hidden", properties={"visible_ms": 1_800_001})
        )
```

- [ ] **Step 2: Run model tests and verify the import fails**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q dashboard/backend/tests/domain/analytics/test_models.py
```

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'dashboard.backend.domain.analytics'`.

- [ ] **Step 3: Implement the closed event models**

Create `models.py` with these public constants and models:

```python
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ANALYTICS_SCHEMA_VERSION = 1
MAX_PROPERTIES_BYTES = 1024

ALLOWED_FRONTEND_EVENT_NAMES = {
    "page_viewed", "page_hidden", "session_heartbeat",
}
ALLOWED_SERVER_EVENT_NAMES = {
    "account_signed_up", "authenticated_session_started",
    "credential_saved", "credential_verified", "credential_defaulted",
    "credential_reverified", "credential_revoked",
    "agent_created", "agent_updated", "agent_deleted",
    "backtest_requested", "backtest_queued", "backtest_started",
    "backtest_completed", "backtest_failed", "backtest_cancelled",
    "model_usage_recorded", "credits_reserved", "credits_settled",
    "credits_refunded", "safe_error_recorded",
}
ALLOWED_EVENT_NAMES = ALLOWED_FRONTEND_EVENT_NAMES | ALLOWED_SERVER_EVENT_NAMES
ALLOWED_PAGE_VIEWS = {
    "home", "agents", "agent_editor", "backtest", "paper_trading",
    "competition", "community", "credits", "account",
}
ALLOWED_BILLING_MODES = {"byok", "platform_credits"}
ALLOWED_OUTCOMES = {"succeeded", "failed", "cancelled"}
ALLOWED_ERROR_CATEGORIES = {
    "credential_invalid", "credential_missing", "provider_timeout",
    "provider_unavailable", "credits_unavailable", "model_not_allowed",
    "internal_error",
}
SERVER_EVENT_PROPERTY_RULES = {
    "account_signed_up": {},
    "authenticated_session_started": {},
    "credential_saved": {},
    "credential_verified": {},
    "credential_defaulted": {},
    "credential_reverified": {},
    "credential_revoked": {},
    "agent_created": {},
    "agent_updated": {},
    "agent_deleted": {},
    "backtest_requested": {},
    "backtest_queued": {},
    "backtest_started": {},
    "backtest_completed": {},
    "backtest_failed": {},
    "backtest_cancelled": {},
    "model_usage_recorded": {
        "input_tokens": (int, 0, 2_000_000_000),
        "output_tokens": (int, 0, 2_000_000_000),
        "cost_micro_usd": (int, 0, 10_000_000_000),
    },
    "credits_reserved": {
        "amount_micro": (int, 1, 10_000_000_000),
        "bucket": (str, {"grant", "purchased"}),
    },
    "credits_settled": {
        "amount_micro": (int, 1, 10_000_000_000),
        "bucket": (str, {"grant", "purchased"}),
    },
    "credits_refunded": {
        "amount_micro": (int, 1, 10_000_000_000),
        "bucket": (str, {"grant", "purchased"}),
    },
    "safe_error_recorded": {},
}
EVENT_GROUP_BY_NAME = {
    "page_viewed": "experience",
    "page_hidden": "experience",
    "session_heartbeat": "experience",
    "account_signed_up": "account",
    "authenticated_session_started": "account",
    "credential_saved": "credential",
    "credential_verified": "credential",
    "credential_defaulted": "credential",
    "credential_reverified": "credential",
    "credential_revoked": "credential",
    "agent_created": "agent",
    "agent_updated": "agent",
    "agent_deleted": "agent",
    "backtest_requested": "run",
    "backtest_queued": "run",
    "backtest_started": "run",
    "backtest_completed": "run",
    "backtest_failed": "run",
    "backtest_cancelled": "run",
    "model_usage_recorded": "resource",
    "credits_reserved": "resource",
    "credits_settled": "resource",
    "credits_refunded": "resource",
    "safe_error_recorded": "resource",
}


class FrontendAnalyticsEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=36, max_length=36)
    schema_version: Literal[1]
    event_name: Literal["page_viewed", "page_hidden", "session_heartbeat"]
    session_id: str = Field(min_length=36, max_length=36)
    occurred_at: datetime
    page_view: str
    properties: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id", "session_id")
    @classmethod
    def valid_uuid(cls, value: str) -> str:
        from uuid import UUID
        parsed = UUID(value)
        if str(parsed) != value.lower():
            raise ValueError("identifier must be a canonical UUID")
        return value.lower()

    @field_validator("page_view")
    @classmethod
    def allowed_page(cls, value: str) -> str:
        if value not in ALLOWED_PAGE_VIEWS:
            raise ValueError("unknown page view")
        return value

    @model_validator(mode="after")
    def allowed_properties(self):
        self.properties = sanitize_frontend_properties(
            self.event_name, self.properties
        )
        return self


class AnalyticsEventDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str = Field(min_length=36, max_length=36)
    schema_version: Literal[1] = 1
    event_name: str = Field(min_length=1, max_length=64)
    user_id: int = Field(gt=0)
    session_id: str | None = Field(default=None, min_length=36, max_length=36)
    occurred_at: datetime
    event_source: Literal["frontend", "server", "backfill"]
    source_event_id: str | None = Field(default=None, min_length=1, max_length=200)
    source_record_type: str | None = Field(default=None, min_length=1, max_length=64)
    source_record_id: str | None = Field(default=None, min_length=1, max_length=200)
    correlation_id: str | None = Field(default=None, min_length=1, max_length=200)
    page_view: str | None = Field(default=None, min_length=1, max_length=64)
    provider_id: str | None = Field(default=None, min_length=1, max_length=128)
    model_id: str | None = Field(default=None, min_length=1, max_length=256)
    billing_mode: str | None = Field(default=None, min_length=1, max_length=32)
    outcome: str | None = Field(default=None, min_length=1, max_length=32)
    error_category: str | None = Field(default=None, min_length=1, max_length=64)
    properties: dict[str, Any] = Field(default_factory=dict)


class AnalyticsEventRecord(AnalyticsEventDraft):
    event_group: str
    received_at: datetime
    country_code: str | None = None
    device_category: str | None = None
    browser_family: str | None = None
    network_hash: str | None = None


class AppendEventResult(BaseModel):
    event: AnalyticsEventRecord
    created: bool


class RetentionResult(BaseModel):
    raw_events_deleted: int = 0
    access_rows_deleted: int = 0
    has_more_raw_events: bool = False
    has_more_access_rows: bool = False
```

Implement `sanitize_frontend_properties` before the model class. It must accept no keys for `page_viewed`, accept only `visible_ms` for `page_hidden` and `session_heartbeat`, require a non-boolean integer from 0 through 1,800,000, serialize with compact sorted JSON, and reject a serialized payload over 1,024 bytes.

Implement `sanitize_server_properties` from `SERVER_EVENT_PROPERTY_RULES`. It must reject unknown keys, booleans masquerading as integers, out-of-range token/cost/amount values, unknown bucket strings, and compact JSON over 1,024 bytes. This gives PR 2 a safe instrumentation interface without permitting prompt text or provider response content.

Add `AnalyticsEventDraft`/`AnalyticsEventRecord` validators that reject:

```python
@field_validator("event_name")
@classmethod
def allowed_event_name(cls, value: str) -> str:
    if value not in ALLOWED_EVENT_NAMES:
        raise ValueError("unknown analytics event")
    return value

@model_validator(mode="after")
def coherent_event(self):
    if self.billing_mode is not None and self.billing_mode not in ALLOWED_BILLING_MODES:
        raise ValueError("unknown billing mode")
    if self.error_category is not None and self.error_category not in ALLOWED_ERROR_CATEGORIES:
        raise ValueError("unknown error category")
    if self.outcome is not None and self.outcome not in ALLOWED_OUTCOMES:
        raise ValueError("unknown outcome")
    if self.page_view is not None and self.page_view not in ALLOWED_PAGE_VIEWS:
        raise ValueError("unknown page view")
    if isinstance(self, AnalyticsEventRecord):
        expected_group = EVENT_GROUP_BY_NAME[self.event_name]
        if self.event_group != expected_group:
            raise ValueError("event group does not match event name")
    return self
```

Extend `test_models.py` with a valid `AnalyticsEventRecord` fixture and assert that an unknown event name, mismatched event group, unknown billing mode, unknown error category, and oversized identifier each raise `ValidationError`.

- [ ] **Step 4: Run the model tests**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q dashboard/backend/tests/domain/analytics/test_models.py
```

Expected: PASS.

- [ ] **Step 5: Write failing privacy-reducer tests**

Create `test_privacy.py`. Use `synthetic-secret-canary` only as a negative assertion:

```python
from datetime import datetime, timezone

from starlette.requests import Request

from dashboard.backend.domain.analytics.privacy import (
    monthly_network_hash,
    request_analytics_context,
)


def _request(headers=None, client=("203.0.113.19", 443)):
    raw = [(key.lower().encode(), value.encode()) for key, value in (headers or {}).items()]
    return Request({
        "type": "http",
        "method": "POST",
        "path": "/api/analytics/events",
        "headers": raw,
        "client": client,
        "scheme": "https",
        "server": ("testserver", 443),
        "query_string": b"",
    })


def test_missing_or_short_key_omits_network_hash(monkeypatch):
    occurred_at = datetime(2026, 8, 26, tzinfo=timezone.utc)
    monkeypatch.delenv("ANALYTICS_PSEUDONYMIZATION_KEY", raising=False)
    assert monthly_network_hash("203.0.113.19", occurred_at) is None
    monkeypatch.setenv("ANALYTICS_PSEUDONYMIZATION_KEY", "too-short")
    assert monthly_network_hash("203.0.113.19", occurred_at) is None


def test_network_hash_is_month_scoped_and_never_contains_plain_ip(monkeypatch):
    monkeypatch.setenv(
        "ANALYTICS_PSEUDONYMIZATION_KEY",
        "synthetic-analytics-hmac-key-at-least-32-bytes",
    )
    august = monthly_network_hash(
        "203.0.113.19", datetime(2026, 8, 26, tzinfo=timezone.utc)
    )
    september = monthly_network_hash(
        "203.0.113.19", datetime(2026, 9, 1, tzinfo=timezone.utc)
    )
    assert august and september and august != september
    assert "203.0.113.19" not in august
    assert len(august) == 64


def test_request_context_reduces_user_agent_and_country(monkeypatch):
    monkeypatch.setenv(
        "ANALYTICS_PSEUDONYMIZATION_KEY",
        "synthetic-analytics-hmac-key-at-least-32-bytes",
    )
    monkeypatch.setenv("RENDER", "true")
    raw_agent = (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) "
        "AppleWebKit/605.1.15 Version/18.0 Mobile/15E148 Safari/604.1 "
        "synthetic-secret-canary"
    )
    context = request_analytics_context(
        _request({"user-agent": raw_agent, "cf-ipcountry": "US"}),
        datetime(2026, 8, 26, tzinfo=timezone.utc),
    )
    assert context.browser_family == "Safari"
    assert context.device_category == "mobile"
    assert context.country_code == "US"
    assert "synthetic-secret-canary" not in repr(context)
    assert "Mozilla" not in repr(context)
    assert "203.0.113.19" not in repr(context)
```

- [ ] **Step 6: Run privacy tests and verify they fail**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q dashboard/backend/tests/domain/analytics/test_privacy.py
```

Expected: FAIL because `privacy.py` and `RequestAnalyticsContext` do not exist.

- [ ] **Step 7: Implement privacy reduction without plaintext fallback**

Add this model to `models.py`:

```python
class RequestAnalyticsContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    country_code: str | None = None
    device_category: str
    browser_family: str
    network_hash: str | None = None
```

Create `privacy.py` with these exact rules:

```python
def monthly_network_hash(ip_address: str | None, received_at: datetime) -> str | None:
    secret = (os.getenv("ANALYTICS_PSEUDONYMIZATION_KEY") or "").strip()
    if not ip_address or len(secret.encode("utf-8")) < 32:
        return None
    month = received_at.astimezone(timezone.utc).strftime("%Y-%m")
    message = f"{month}\n{ip_address.strip()}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
```

`request_analytics_context` must:

1. Call `client_ip(request)` once, pass the result directly into `monthly_network_hash`, and never return the raw value.
2. Reduce User-Agent to browser family `Edge`, `Chrome`, `Firefox`, `Safari`, or `Other`.
3. Reduce device to `mobile`, `tablet`, `desktop`, or `unknown`.
4. Accept `CF-IPCountry` only when `RENDER` is truthy and `X-Vercel-IP-Country` only when `VERCEL` is truthy.
5. Accept only two ASCII letters as country code; otherwise return `None`.
6. Never include headers, raw User-Agent, IP, or the HMAC secret in the returned model or exception text.

- [ ] **Step 8: Run Task 1 tests and commit**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/domain/analytics/test_models.py \
  dashboard/backend/tests/domain/analytics/test_privacy.py
```

Expected: PASS.

Commit:

```bash
git add \
  dashboard/backend/domain/analytics/__init__.py \
  dashboard/backend/domain/analytics/models.py \
  dashboard/backend/domain/analytics/privacy.py \
  dashboard/backend/tests/domain/analytics/__init__.py \
  dashboard/backend/tests/domain/analytics/test_models.py \
  dashboard/backend/tests/domain/analytics/test_privacy.py
git commit -m "feat: define analytics event privacy contract"
```

---

### Task 2: Build the SQLite Analytics repository

**Files:**
- Create: `dashboard/backend/domain/analytics/repository_common.py`
- Create: `dashboard/backend/domain/analytics/repository.py`
- Create: `dashboard/backend/tests/domain/analytics/test_repository_contract.py`

**Interfaces:**
- Consumes: `AnalyticsEventRecord`, `AppendEventResult`, `RetentionResult`, `DB_PATH`, and the existing account `users` table.
- Produces: `AnalyticsStoreError`, `AnalyticsIdempotencyConflictError`, `encode_event_cursor(occurred_at, sequence)`, `decode_event_cursor(cursor)`, `AnalyticsStore`, and the repository methods listed below.

Repository public methods:

```python
append_event(event: AnalyticsEventRecord) -> AppendEventResult
get_event(event_id: str) -> AnalyticsEventRecord | None
list_user_events(user_id: int, *, limit: int = 50, cursor: str | None = None) -> dict
set_subject_exclusion(user_id: int, *, excluded: bool, actor_user_id: int, reason: str) -> dict
get_subject_setting(user_id: int) -> dict | None
list_excluded_user_ids(*, include_admin_accounts: bool = True) -> set[int]
record_admin_access(admin_user_id: int, subject_user_id: int, section: str) -> dict
list_admin_access(subject_user_id: int, *, limit: int = 50) -> list[dict]
delete_expired(*, raw_before: datetime, access_before: datetime, batch_size: int) -> RetentionResult
```

- [ ] **Step 1: Write the shared SQLite repository contract**

Create `test_repository_contract.py` with a fixture that creates users through `UserStore` and passes the same database path to `AnalyticsStore`. Define reusable assertion functions so PostgreSQL can run the identical behavior later:

```python
def assert_event_idempotency_contract(store, user_id):
    event = event_record(user_id=user_id, event_id="10000000-0000-4000-8000-000000000001")
    first = store.append_event(event)
    replay = store.append_event(event)
    assert first.created is True
    assert replay.created is False
    assert replay.event == first.event

    changed = event.model_copy(update={"page_view": "credits"})
    with pytest.raises(AnalyticsIdempotencyConflictError):
        store.append_event(changed)


def assert_source_event_idempotency_contract(store, user_id):
    event = event_record(
        user_id=user_id,
        event_id="10000000-0000-4000-8000-000000000002",
        event_source="server",
        source_event_id="run:run_123:completed",
        event_name="backtest_completed",
        event_group="run",
        page_view=None,
        session_id=None,
    )
    assert store.append_event(event).created is True
    replay = event.model_copy(
        update={"event_id": "10000000-0000-4000-8000-000000000003"}
    )
    assert store.append_event(replay).created is False


def assert_cursor_contract(store, user_id):
    for index in range(3):
        store.append_event(event_record(
            user_id=user_id,
            event_id=f"20000000-0000-4000-8000-00000000000{index}",
            occurred_at=datetime(2026, 8, 26, 12, index, tzinfo=timezone.utc),
        ))
    first = store.list_user_events(user_id, limit=2)
    second = store.list_user_events(user_id, limit=2, cursor=first["next_cursor"])
    assert len(first["items"]) == 2
    assert len(second["items"]) == 1
    assert not ({item.event_id for item in first["items"]} & {item.event_id for item in second["items"]})


def assert_subject_and_access_contract(store, admin_id, user_id):
    setting = store.set_subject_exclusion(
        user_id,
        excluded=True,
        actor_user_id=admin_id,
        reason="Synthetic QA account.",
    )
    assert setting["excluded"] is True
    assert user_id in store.list_excluded_user_ids()
    access = store.record_admin_access(admin_id, user_id, "overview")
    assert access["admin_user_id"] == admin_id
    assert access["subject_user_id"] == user_id
    assert "response" not in access
    assert store.list_admin_access(user_id)[0]["section"] == "overview"
```

Add tests that assert:

- all five tables exist;
- an invalid cursor raises `ValueError("invalid analytics cursor")`;
- `limit` accepts 1 through 100 only;
- access sections accept `overview`, `timeline`, `runs`, `usage`, and `sessions` only;
- subject reason is trimmed and 1 through 500 characters;
- Admin users are included by `list_excluded_user_ids(include_admin_accounts=True)` even without a settings row;
- disabling a settings row removes a normal user from the exclusion set;
- foreign keys reject nonexistent users.

- [ ] **Step 2: Run the SQLite contract and verify it fails**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q dashboard/backend/tests/domain/analytics/test_repository_contract.py
```

Expected: FAIL because the repository modules do not exist.

- [ ] **Step 3: Implement shared errors, cursor encoding, and canonical comparison**

In `repository_common.py` define:

```python
class AnalyticsStoreError(RuntimeError):
    pass


class AnalyticsIdempotencyConflictError(AnalyticsStoreError):
    pass


def encode_event_cursor(occurred_at: str, sequence: int) -> str:
    payload = json.dumps(
        [occurred_at, sequence], ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def decode_event_cursor(cursor: str) -> tuple[str, int]:
    if not isinstance(cursor, str) or not cursor or len(cursor) > 256:
        raise ValueError("invalid analytics cursor")
    try:
        raw = base64.b64decode(
            cursor + "=" * (-len(cursor) % 4),
            altchars=b"-_",
            validate=True,
        )
        value = json.loads(raw.decode("utf-8"))
    except (ValueError, binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid analytics cursor") from exc
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("invalid analytics cursor")
    occurred_at, sequence = value
    if (
        not isinstance(occurred_at, str)
        or not occurred_at
        or len(occurred_at) > 64
        or isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence <= 0
    ):
        raise ValueError("invalid analytics cursor")
    return occurred_at, sequence


def canonical_event_payload(
    event: AnalyticsEventRecord,
    *,
    ignore_event_id: bool = False,
) -> dict[str, object]:
    value = event.model_dump(mode="json")
    value.pop("received_at", None)
    if ignore_event_id:
        value.pop("event_id", None)
    value["properties"] = json.loads(
        json.dumps(value["properties"], sort_keys=True, separators=(",", ":"))
    )
    return value
```

- [ ] **Step 4: Create all five SQLite tables and indexes**

In `repository.py` create `AnalyticsStore`. Use a connection context manager that enables foreign keys, commits or rolls back, and always closes.

The schema must contain:

1. `analytics_events` with an integer `sequence` primary key, unique `event_id`, nullable unique `source_event_id`, every field in the approved event envelope, compact `properties_json`, and indexes on user/time, event/time, session/time, outcome/time, error/time, and source event.
2. `analytics_daily_rollups` keyed by `rollup_date`, `metric_name`, and non-null bounded dimension strings for event, billing mode, provider, model, outcome, error category, and user state. Store `value_count` and `value_sum_micro`.
3. `user_analytics_snapshots` keyed by `user_id` with status, reason code, human-readable reason, `evidence_event_ids_json`, and `calculated_at`.
4. `analytics_subject_settings` keyed by `user_id` with `excluded`, `actor_user_id`, reason, created time, and updated time.
5. `admin_analytics_access_log` with integer primary key, Admin user, subject user, section, and access time.

Use `ON DELETE CASCADE` only for event/snapshot/settings rows owned by the subject. Use `ON DELETE RESTRICT` for the Admin actor in subject settings and for both identities in access logs.

Use this concrete SQLite DDL:

```python
ANALYTICS_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS analytics_events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE CHECK (length(event_id) = 36),
    schema_version INTEGER NOT NULL CHECK (schema_version = 1),
    event_name TEXT NOT NULL CHECK (length(event_name) BETWEEN 1 AND 64),
    event_group TEXT NOT NULL
        CHECK (event_group IN ('experience', 'account', 'credential', 'agent', 'run', 'resource')),
    user_id INTEGER NOT NULL,
    session_id TEXT CHECK (session_id IS NULL OR length(session_id) = 36),
    occurred_at TEXT NOT NULL,
    received_at TEXT NOT NULL,
    event_source TEXT NOT NULL
        CHECK (event_source IN ('frontend', 'server', 'backfill')),
    source_event_id TEXT UNIQUE
        CHECK (source_event_id IS NULL OR length(source_event_id) BETWEEN 1 AND 200),
    source_record_type TEXT
        CHECK (source_record_type IS NULL OR length(source_record_type) BETWEEN 1 AND 64),
    source_record_id TEXT
        CHECK (source_record_id IS NULL OR length(source_record_id) BETWEEN 1 AND 200),
    correlation_id TEXT
        CHECK (correlation_id IS NULL OR length(correlation_id) BETWEEN 1 AND 200),
    page_view TEXT CHECK (page_view IS NULL OR length(page_view) BETWEEN 1 AND 64),
    provider_id TEXT CHECK (provider_id IS NULL OR length(provider_id) BETWEEN 1 AND 128),
    model_id TEXT CHECK (model_id IS NULL OR length(model_id) BETWEEN 1 AND 256),
    billing_mode TEXT
        CHECK (billing_mode IS NULL OR billing_mode IN ('byok', 'platform_credits')),
    outcome TEXT
        CHECK (outcome IS NULL OR outcome IN ('succeeded', 'failed', 'cancelled')),
    error_category TEXT CHECK (
        error_category IS NULL OR error_category IN (
            'credential_invalid', 'credential_missing', 'provider_timeout',
            'provider_unavailable', 'credits_unavailable',
            'model_not_allowed', 'internal_error'
        )
    ),
    country_code TEXT
        CHECK (country_code IS NULL OR length(country_code) = 2),
    device_category TEXT CHECK (
        device_category IS NULL OR device_category IN (
            'mobile', 'tablet', 'desktop', 'unknown'
        )
    ),
    browser_family TEXT CHECK (
        browser_family IS NULL OR browser_family IN (
            'Edge', 'Chrome', 'Firefox', 'Safari', 'Other'
        )
    ),
    network_hash TEXT
        CHECK (network_hash IS NULL OR length(network_hash) = 64),
    properties_json TEXT NOT NULL DEFAULT '{}'
        CHECK (length(properties_json) <= 1024),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_analytics_events_user_time
    ON analytics_events(user_id, occurred_at DESC, sequence DESC);
CREATE INDEX IF NOT EXISTS idx_analytics_events_name_time
    ON analytics_events(event_name, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_analytics_events_session_time
    ON analytics_events(session_id, occurred_at DESC)
    WHERE session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_analytics_events_outcome_time
    ON analytics_events(outcome, occurred_at DESC)
    WHERE outcome IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_analytics_events_error_time
    ON analytics_events(error_category, occurred_at DESC)
    WHERE error_category IS NOT NULL;

CREATE TABLE IF NOT EXISTS analytics_daily_rollups (
    rollup_date TEXT NOT NULL CHECK (length(rollup_date) = 10),
    metric_name TEXT NOT NULL CHECK (length(metric_name) BETWEEN 1 AND 64),
    event_name TEXT NOT NULL DEFAULT '' CHECK (length(event_name) <= 64),
    billing_mode TEXT NOT NULL DEFAULT '' CHECK (length(billing_mode) <= 32),
    provider_id TEXT NOT NULL DEFAULT '' CHECK (length(provider_id) <= 128),
    model_id TEXT NOT NULL DEFAULT '' CHECK (length(model_id) <= 256),
    outcome TEXT NOT NULL DEFAULT '' CHECK (length(outcome) <= 32),
    error_category TEXT NOT NULL DEFAULT '' CHECK (length(error_category) <= 64),
    user_state TEXT NOT NULL DEFAULT '' CHECK (length(user_state) <= 32),
    value_count INTEGER NOT NULL DEFAULT 0 CHECK (value_count >= 0),
    value_sum_micro INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (
        rollup_date, metric_name, event_name, billing_mode, provider_id,
        model_id, outcome, error_category, user_state
    )
);

CREATE TABLE IF NOT EXISTS user_analytics_snapshots (
    user_id INTEGER PRIMARY KEY,
    status TEXT NOT NULL CHECK (
        status IN ('blocked', 'needs_attention', 'dormant', 'onboarding', 'active')
    ),
    reason_code TEXT NOT NULL CHECK (length(reason_code) BETWEEN 1 AND 100),
    human_readable_reason TEXT NOT NULL
        CHECK (length(human_readable_reason) BETWEEN 1 AND 500),
    evidence_event_ids_json TEXT NOT NULL DEFAULT '[]'
        CHECK (length(evidence_event_ids_json) <= 4096),
    calculated_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS analytics_subject_settings (
    user_id INTEGER PRIMARY KEY,
    excluded INTEGER NOT NULL CHECK (excluded IN (0, 1)),
    actor_user_id INTEGER NOT NULL,
    reason TEXT NOT NULL CHECK (length(reason) BETWEEN 1 AND 500),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (actor_user_id) REFERENCES users(id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS admin_analytics_access_log (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    admin_user_id INTEGER NOT NULL,
    subject_user_id INTEGER NOT NULL,
    section TEXT NOT NULL CHECK (
        section IN ('overview', 'timeline', 'runs', 'usage', 'sessions')
    ),
    accessed_at TEXT NOT NULL,
    FOREIGN KEY (admin_user_id) REFERENCES users(id) ON DELETE RESTRICT,
    FOREIGN KEY (subject_user_id) REFERENCES users(id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_admin_analytics_access_subject_time
    ON admin_analytics_access_log(subject_user_id, accessed_at DESC, sequence DESC);
CREATE INDEX IF NOT EXISTS idx_admin_analytics_access_admin_time
    ON admin_analytics_access_log(admin_user_id, accessed_at DESC, sequence DESC);
"""
```

- [ ] **Step 5: Implement idempotent append and opaque event pagination**

`append_event` must execute inside one transaction:

1. Attempt the insert.
2. On uniqueness conflict, look up both `event_id` and a non-null `source_event_id`. If both keys resolve to different stored sequences, raise a conflict immediately. Otherwise use the one resolved row.
3. Compare `canonical_event_payload` after decoding stored `properties_json`. The helper always ignores server-assigned `received_at`. Pass `ignore_event_id=True` when replaying a non-null `source_event_id` so a retry may carry a fresh transport event UUID while the authoritative source identity remains stable.
4. Return `created=False` for an exact replay.
5. Raise `AnalyticsIdempotencyConflictError("analytics event idempotency conflict")` for changed data.

`list_user_events` must order by `occurred_at DESC, sequence DESC`, fetch `limit + 1`, and encode the last returned row into `next_cursor` only when another row exists.

- [ ] **Step 6: Implement exclusions, access audit, and bounded deletion**

`delete_expired` must:

- validate `batch_size` from 1 through 10,000;
- delete at most one batch from `analytics_events` where `received_at < raw_before`;
- delete at most one batch from `admin_analytics_access_log` where `accessed_at < access_before`;
- return exact deletion counts;
- set each `has_more_*` flag by a bounded `SELECT 1 ... LIMIT 1`;
- never delete from `analytics_daily_rollups` or `user_analytics_snapshots`.

- [ ] **Step 7: Run the SQLite contract and commit**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q dashboard/backend/tests/domain/analytics/test_repository_contract.py
```

Expected: PASS.

Commit:

```bash
git add \
  dashboard/backend/domain/analytics/repository_common.py \
  dashboard/backend/domain/analytics/repository.py \
  dashboard/backend/tests/domain/analytics/test_repository_contract.py
git commit -m "feat: add sqlite analytics repository"
```

---

### Task 3: Add PostgreSQL parity and account-database dispatch

**Files:**
- Create: `dashboard/backend/domain/analytics/repository_postgres.py`
- Create: `dashboard/backend/tests/domain/analytics/test_repository_postgres.py`
- Modify: `dashboard/backend/domain/analytics/repository.py`
- Modify: `dashboard/backend/tests/test_ci_postgres_wired.py`

**Interfaces:**
- Consumes: Task 2 contract assertions, `require_postgres_url`, `describe_database_url`, `db_pool.connection`, and `USERS_DATABASE_URL`.
- Produces: `PostgresAnalyticsStore` and `_build_analytics_store()`; exports singleton `analytics_store`.

- [ ] **Step 1: Write dispatch and live-PostgreSQL tests**

Create `test_repository_postgres.py` with:

```python
TEST_POSTGRES_URL = require_local_postgres_url(os.getenv("TEST_POSTGRES_URL"))
pg_only = pytest.mark.skipif(
    not TEST_POSTGRES_URL,
    reason="TEST_POSTGRES_URL is not configured",
)


def test_build_analytics_store_defaults_to_sqlite(monkeypatch, capsys):
    monkeypatch.delenv("USERS_DATABASE_URL", raising=False)
    store = repo_module._build_analytics_store()
    assert isinstance(store, repo_module.AnalyticsStore)
    assert "analytics_store backend: sqlite" in capsys.readouterr().out


def test_build_analytics_store_uses_only_users_database_url(monkeypatch):
    created = {}

    class FakePostgresAnalyticsStore:
        def __init__(self, database_url):
            created["database_url"] = database_url

    monkeypatch.setattr(pg_module, "PostgresAnalyticsStore", FakePostgresAnalyticsStore)
    monkeypatch.setenv("USERS_DATABASE_URL", "postgresql://fake/accounts")
    monkeypatch.setenv("CONTENT_DATABASE_URL", "postgresql://ignored/content")
    monkeypatch.setenv("AGENT_RUNS_DATABASE_URL", "postgresql://ignored/runs")
    assert isinstance(repo_module._build_analytics_store(), FakePostgresAnalyticsStore)
    assert created["database_url"] == "postgresql://fake/accounts"


def test_dispatch_log_never_contains_database_password(monkeypatch, capsys):
    monkeypatch.setattr(pg_module, "PostgresAnalyticsStore", lambda database_url: object())
    monkeypatch.setenv(
        "USERS_DATABASE_URL",
        "postgresql://admin:synthetic-password@host/accounts",
    )
    repo_module._build_analytics_store()
    output = capsys.readouterr().out
    assert "synthetic-password" not in output
    assert "host/accounts" in output
```

For the live fixture, create a random schema on local `TEST_POSTGRES_URL`, create a minimal `users(id INTEGER PRIMARY KEY, role TEXT NOT NULL)` table inside it, instantiate `PostgresAnalyticsStore` with a URL whose `search_path` is that schema, run every shared contract assertion from Task 2, and drop only that random schema during teardown.

- [ ] **Step 2: Run dispatch tests and verify they fail**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/domain/analytics/test_repository_postgres.py \
  -k "build_analytics_store or dispatch_log"
```

Expected: FAIL because `PostgresAnalyticsStore` and `_build_analytics_store` do not exist.

- [ ] **Step 3: Implement the PostgreSQL schema and methods**

Create `ANALYTICS_POSTGRES_DDL` as a literal string so parity tests can inspect it without a live database. Mirror every SQLite table, column, check, foreign key, uniqueness rule, and index, using `BIGSERIAL` for sequences and `BOOLEAN` for exclusion state.

Use this PostgreSQL DDL:

```python
ANALYTICS_POSTGRES_DDL = """
CREATE TABLE IF NOT EXISTS analytics_events (
    sequence BIGSERIAL PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE CHECK (length(event_id) = 36),
    schema_version INTEGER NOT NULL CHECK (schema_version = 1),
    event_name TEXT NOT NULL CHECK (length(event_name) BETWEEN 1 AND 64),
    event_group TEXT NOT NULL
        CHECK (event_group IN ('experience', 'account', 'credential', 'agent', 'run', 'resource')),
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id TEXT CHECK (session_id IS NULL OR length(session_id) = 36),
    occurred_at TEXT NOT NULL,
    received_at TEXT NOT NULL,
    event_source TEXT NOT NULL
        CHECK (event_source IN ('frontend', 'server', 'backfill')),
    source_event_id TEXT UNIQUE
        CHECK (source_event_id IS NULL OR length(source_event_id) BETWEEN 1 AND 200),
    source_record_type TEXT
        CHECK (source_record_type IS NULL OR length(source_record_type) BETWEEN 1 AND 64),
    source_record_id TEXT
        CHECK (source_record_id IS NULL OR length(source_record_id) BETWEEN 1 AND 200),
    correlation_id TEXT
        CHECK (correlation_id IS NULL OR length(correlation_id) BETWEEN 1 AND 200),
    page_view TEXT CHECK (page_view IS NULL OR length(page_view) BETWEEN 1 AND 64),
    provider_id TEXT CHECK (provider_id IS NULL OR length(provider_id) BETWEEN 1 AND 128),
    model_id TEXT CHECK (model_id IS NULL OR length(model_id) BETWEEN 1 AND 256),
    billing_mode TEXT
        CHECK (billing_mode IS NULL OR billing_mode IN ('byok', 'platform_credits')),
    outcome TEXT
        CHECK (outcome IS NULL OR outcome IN ('succeeded', 'failed', 'cancelled')),
    error_category TEXT CHECK (
        error_category IS NULL OR error_category IN (
            'credential_invalid', 'credential_missing', 'provider_timeout',
            'provider_unavailable', 'credits_unavailable',
            'model_not_allowed', 'internal_error'
        )
    ),
    country_code TEXT
        CHECK (country_code IS NULL OR length(country_code) = 2),
    device_category TEXT CHECK (
        device_category IS NULL OR device_category IN (
            'mobile', 'tablet', 'desktop', 'unknown'
        )
    ),
    browser_family TEXT CHECK (
        browser_family IS NULL OR browser_family IN (
            'Edge', 'Chrome', 'Firefox', 'Safari', 'Other'
        )
    ),
    network_hash TEXT
        CHECK (network_hash IS NULL OR length(network_hash) = 64),
    properties_json TEXT NOT NULL DEFAULT '{}'
        CHECK (length(properties_json) <= 1024)
);

CREATE INDEX IF NOT EXISTS idx_analytics_events_user_time
    ON analytics_events(user_id, occurred_at DESC, sequence DESC);
CREATE INDEX IF NOT EXISTS idx_analytics_events_name_time
    ON analytics_events(event_name, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_analytics_events_session_time
    ON analytics_events(session_id, occurred_at DESC)
    WHERE session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_analytics_events_outcome_time
    ON analytics_events(outcome, occurred_at DESC)
    WHERE outcome IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_analytics_events_error_time
    ON analytics_events(error_category, occurred_at DESC)
    WHERE error_category IS NOT NULL;

CREATE TABLE IF NOT EXISTS analytics_daily_rollups (
    rollup_date TEXT NOT NULL CHECK (length(rollup_date) = 10),
    metric_name TEXT NOT NULL CHECK (length(metric_name) BETWEEN 1 AND 64),
    event_name TEXT NOT NULL DEFAULT '' CHECK (length(event_name) <= 64),
    billing_mode TEXT NOT NULL DEFAULT '' CHECK (length(billing_mode) <= 32),
    provider_id TEXT NOT NULL DEFAULT '' CHECK (length(provider_id) <= 128),
    model_id TEXT NOT NULL DEFAULT '' CHECK (length(model_id) <= 256),
    outcome TEXT NOT NULL DEFAULT '' CHECK (length(outcome) <= 32),
    error_category TEXT NOT NULL DEFAULT '' CHECK (length(error_category) <= 64),
    user_state TEXT NOT NULL DEFAULT '' CHECK (length(user_state) <= 32),
    value_count BIGINT NOT NULL DEFAULT 0 CHECK (value_count >= 0),
    value_sum_micro BIGINT NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (
        rollup_date, metric_name, event_name, billing_mode, provider_id,
        model_id, outcome, error_category, user_state
    )
);

CREATE TABLE IF NOT EXISTS user_analytics_snapshots (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK (
        status IN ('blocked', 'needs_attention', 'dormant', 'onboarding', 'active')
    ),
    reason_code TEXT NOT NULL CHECK (length(reason_code) BETWEEN 1 AND 100),
    human_readable_reason TEXT NOT NULL
        CHECK (length(human_readable_reason) BETWEEN 1 AND 500),
    evidence_event_ids_json TEXT NOT NULL DEFAULT '[]'
        CHECK (length(evidence_event_ids_json) <= 4096),
    calculated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analytics_subject_settings (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    excluded BOOLEAN NOT NULL,
    actor_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    reason TEXT NOT NULL CHECK (length(reason) BETWEEN 1 AND 500),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS admin_analytics_access_log (
    sequence BIGSERIAL PRIMARY KEY,
    admin_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    subject_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    section TEXT NOT NULL CHECK (
        section IN ('overview', 'timeline', 'runs', 'usage', 'sessions')
    ),
    accessed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_admin_analytics_access_subject_time
    ON admin_analytics_access_log(subject_user_id, accessed_at DESC, sequence DESC);
CREATE INDEX IF NOT EXISTS idx_admin_analytics_access_admin_time
    ON admin_analytics_access_log(admin_user_id, accessed_at DESC, sequence DESC);
"""
```

Implement `PostgresAnalyticsStore` with the same public method names and return shapes as `AnalyticsStore`. Use:

- `require_postgres_url` before connecting;
- the shared pooled connection helper already used by current PostgreSQL stores;
- `psycopg.rows.dict_row`;
- `ON CONFLICT DO NOTHING` followed by a canonical replay comparison for idempotency;
- tuple comparison `(occurred_at, sequence) < (%s, %s)` for cursor pagination;
- CTE-limited deletes so each retention call removes at most `batch_size` rows.

- [ ] **Step 4: Implement account-database dispatch**

At the bottom of `repository.py` add:

```python
def _build_analytics_store():
    database_url = (os.getenv("USERS_DATABASE_URL") or "").strip()
    if database_url:
        from .repository_postgres import PostgresAnalyticsStore
        print(
            "analytics_store backend: postgres "
            f"({describe_database_url(database_url)})"
        )
        return PostgresAnalyticsStore(database_url)
    print("analytics_store backend: sqlite (ephemeral on Render)")
    return AnalyticsStore()


analytics_store = _build_analytics_store()
```

Do not read `CONTENT_DATABASE_URL` or `AGENT_RUNS_DATABASE_URL`.

- [ ] **Step 5: Add CI wiring guards**

Extend `test_ci_postgres_wired.py` so the PostgreSQL CI command includes `dashboard/backend/tests/domain/analytics/test_repository_postgres.py` and so accidental removal of that path fails the static guard.

- [ ] **Step 6: Run parity tests and commit**

Run the non-live tier first:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/domain/analytics/test_repository_contract.py \
  dashboard/backend/tests/domain/analytics/test_repository_postgres.py
```

Expected: PASS with live PostgreSQL cases skipped when `TEST_POSTGRES_URL` is unset.

When a local disposable PostgreSQL URL is available, run:

```bash
TEST_POSTGRES_URL=postgresql://postgres:test@127.0.0.1:5433/atl_test \
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/domain/analytics/test_repository_postgres.py
```

Expected: PASS; the fixture must refuse a non-local PostgreSQL host before any destructive SQL.

Commit:

```bash
git add \
  dashboard/backend/domain/analytics/repository.py \
  dashboard/backend/domain/analytics/repository_postgres.py \
  dashboard/backend/tests/domain/analytics/test_repository_postgres.py \
  dashboard/backend/tests/test_ci_postgres_wired.py
git commit -m "feat: add postgres analytics repository"
```

---

### Task 4: Add the Analytics service, subject controls, and access audit

**Files:**
- Create: `dashboard/backend/domain/analytics/service.py`
- Create: `dashboard/backend/tests/domain/analytics/test_service.py`

**Interfaces:**
- Consumes: `AnalyticsStore` contract, `FrontendAnalyticsEvent`, `RequestAnalyticsContext`, `EVENT_GROUP_BY_NAME`.
- Produces: `AnalyticsService`, `get_analytics_service()`, singleton `analytics_service`, `accept_frontend_event`, `record_server_event`, `try_record_server_event`, `set_subject_exclusion`, and `record_admin_profile_access`.

- [ ] **Step 1: Write failing service tests**

Cover:

```python
def test_frontend_event_uses_authenticated_identity_and_server_received_time():
    result = service.accept_frontend_event(
        user={"id": 42, "role": "user", "email": "must-not-persist@example.test"},
        payload=frontend_event(),
        context=RequestAnalyticsContext(
            country_code="US",
            device_category="desktop",
            browser_family="Chrome",
            network_hash="a" * 64,
        ),
        received_at=datetime(2026, 8, 26, 12, 0, 1, tzinfo=timezone.utc),
    )
    assert result.event.user_id == 42
    assert result.event.event_group == "experience"
    assert result.event.event_source == "frontend"
    assert result.event.received_at.isoformat() == "2026-08-26T12:00:01+00:00"
    assert "email" not in result.event.model_dump()


def test_frontend_event_rejects_large_clock_skew():
    with pytest.raises(ValueError, match="occurred_at"):
        service.accept_frontend_event(
            user={"id": 42, "role": "user"},
            payload=frontend_event(occurred_at="2026-08-24T12:00:00Z"),
            context=safe_context(),
            received_at=datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc),
        )


def test_best_effort_server_event_never_raises_or_prints_sensitive_fields(capsys):
    failing = AnalyticsService(store=FailingStore())
    result = failing.try_record_server_event(
        event_name="credential_verified",
        user_id=42,
        source_event_id="credential:synthetic-id:verified",
        source_record_type="credential",
        source_record_id="synthetic-id",
        properties={},
    )
    assert result is None
    output = capsys.readouterr().out
    assert "synthetic-secret-canary" not in output
    assert "analytics.append_failed" in output
```

Also assert:

- `record_server_event` rejects a frontend event name;
- frontend `occurred_at` may be at most 24 hours old and 5 minutes in the future;
- `source_event_id` is required for server and backfill events;
- server events cannot carry properties outside an event-specific allowlist;
- `set_subject_exclusion` requires an Admin actor and never changes the user role;
- `record_admin_profile_access` requires an Admin actor and accepts only the five profile sections;
- no service result or exception contains the synthetic email, raw IP, raw User-Agent, token, or canary secret.

- [ ] **Step 2: Run service tests and verify they fail**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q dashboard/backend/tests/domain/analytics/test_service.py
```

Expected: FAIL because `service.py` does not exist.

- [ ] **Step 3: Implement normalized frontend acceptance**

Implement:

```python
class AnalyticsService:
    def __init__(self, store):
        self.store = store

    def accept_frontend_event(
        self,
        *,
        user: dict,
        payload: FrontendAnalyticsEvent,
        context: RequestAnalyticsContext,
        received_at: datetime | None = None,
    ) -> AppendEventResult:
        received = received_at or datetime.now(timezone.utc)
        if payload.occurred_at < received - timedelta(hours=24):
            raise ValueError("occurred_at is too old")
        if payload.occurred_at > received + timedelta(minutes=5):
            raise ValueError("occurred_at is in the future")
        event = AnalyticsEventRecord(
            event_id=payload.event_id,
            schema_version=payload.schema_version,
            event_name=payload.event_name,
            event_group=EVENT_GROUP_BY_NAME[payload.event_name],
            user_id=int(user["id"]),
            session_id=payload.session_id,
            occurred_at=payload.occurred_at,
            received_at=received,
            event_source="frontend",
            page_view=payload.page_view,
            country_code=context.country_code,
            device_category=context.device_category,
            browser_family=context.browser_family,
            network_hash=context.network_hash,
            properties=payload.properties,
        )
        return self.store.append_event(event)
```

Do not pass the complete `user` dictionary to the repository.

- [ ] **Step 4: Implement future instrumentation and failure isolation interfaces**

`record_server_event` must require a deterministic `source_event_id`, generate a fresh `event_id`, force `event_source="server"`, validate event/group/field combinations, and pass only allowlisted normalized values to `append_event`.

`try_record_server_event` must wrap `record_server_event` in `try/except Exception`, print only:

```text
WARNING: analytics.append_failed event=<event_name> category=<exception-class-name>
```

and return `None`. It must not print `properties`, source IDs, user IDs, exception text, or stack traces because upstream exceptions can contain provider bodies.

- [ ] **Step 5: Implement Admin-only subject and access methods**

`set_subject_exclusion` and `record_admin_profile_access` must check `actor.get("role") == "admin"` before calling the repository. Raise `PermissionError("admin required")` otherwise. The service methods return display-safe repository rows only and never load an Analytics profile response body into the access log.

- [ ] **Step 6: Run service tests and commit**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/domain/analytics/test_models.py \
  dashboard/backend/tests/domain/analytics/test_privacy.py \
  dashboard/backend/tests/domain/analytics/test_repository_contract.py \
  dashboard/backend/tests/domain/analytics/test_service.py
```

Expected: PASS.

Commit:

```bash
git add \
  dashboard/backend/domain/analytics/service.py \
  dashboard/backend/tests/domain/analytics/test_service.py
git commit -m "feat: add analytics foundation service"
```

---

### Task 5: Add safe authenticated frontend ingestion

**Files:**
- Create: `dashboard/backend/api/routers/analytics.py`
- Create: `dashboard/backend/tests/test_analytics_api.py`
- Modify: `dashboard/backend/api/router.py`
- Modify: `dashboard/backend/tests/test_csrf.py`

**Interfaces:**
- Consumes: `get_current_user`, `FixedWindowRateLimiter`, `FrontendAnalyticsEvent`, `request_analytics_context`, and `get_analytics_service`.
- Produces: `POST /api/analytics/events` and `reset_analytics_ingestion_limiter()`.

- [ ] **Step 1: Write failing authentication, validation, rate-limit, and secret-canary API tests**

Create `test_analytics_api.py` using an isolated `UserStore` and `AnalyticsStore` on the same temporary database. Patch `get_analytics_service` through FastAPI dependency overrides.

Required assertions:

```python
def test_analytics_ingestion_requires_authentication(client):
    response = client.post("/api/analytics/events", json=frontend_payload())
    assert response.status_code == 401


def test_analytics_ingestion_returns_generic_acceptance(client, signed_in_user, store):
    response = client.post("/api/analytics/events", json=frontend_payload())
    assert response.status_code == 202
    assert response.json() == {"accepted": True}
    saved = store.list_user_events(signed_in_user["id"])["items"]
    assert len(saved) == 1
    assert saved[0].user_id == signed_in_user["id"]


def test_analytics_validation_never_echoes_secret_canary(client, signed_in_user, capsys):
    payload = frontend_payload()
    payload["api_key"] = "synthetic-secret-canary"
    response = client.post("/api/analytics/events", json=payload)
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid analytics event."}
    assert "synthetic-secret-canary" not in response.text
    assert "synthetic-secret-canary" not in capsys.readouterr().out
```

Also test:

- a declared `Content-Length` or actual body over 8 KiB returns generic `413`;
- malformed JSON and non-object JSON return generic `422`;
- the 121st accepted request in five minutes for one user returns `429` and a `Retry-After` header;
- a second user has an independent limiter bucket;
- a store failure returns generic `503 {"detail":"Analytics is temporarily unavailable."}` without echoing exception text;
- the route is registered exactly once beneath `/api`.

- [ ] **Step 2: Run API tests and verify they fail**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q dashboard/backend/tests/test_analytics_api.py
```

Expected: FAIL with `404` or missing module.

- [ ] **Step 3: Implement manual bounded parsing**

Create a router with `prefix="/analytics"` and `tags=["analytics"]`. Do not declare `FrontendAnalyticsEvent` as the body parameter because FastAPI's default validation response can include rejected input values.

Use this parsing shape:

```python
MAX_ANALYTICS_BODY_BYTES = 8 * 1024


async def _parse_event(request: Request) -> FrontendAnalyticsEvent:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_ANALYTICS_BODY_BYTES:
                raise HTTPException(413, "Analytics event is too large.")
        except ValueError:
            raise HTTPException(400, "Invalid request.") from None
    body = await request.body()
    if len(body) > MAX_ANALYTICS_BODY_BYTES:
        raise HTTPException(413, "Analytics event is too large.")
    try:
        value = json.loads(body)
        if not isinstance(value, dict):
            raise ValueError("body must be an object")
        return FrontendAnalyticsEvent.model_validate(value)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError):
        raise HTTPException(422, "Invalid analytics event.") from None
```

- [ ] **Step 4: Implement authenticated ingestion and rate limiting**

Use `FixedWindowRateLimiter(max_events=120, window_seconds=300)` keyed only by authenticated user ID. The endpoint must:

1. Resolve `current_user=Depends(get_current_user)`.
2. Spend the limiter slot before parsing/persistence.
3. Capture one UTC `received_at` server timestamp.
4. Build `RequestAnalyticsContext` with that `received_at` timestamp, without logging the request.
5. Pass the same `received_at` into `accept_frontend_event` and run synchronous service persistence in Starlette's threadpool.
6. Return status `202` and exactly `{"accepted": True}` for both a newly created event and an exact idempotent replay.
7. Convert `AnalyticsIdempotencyConflictError` to generic `409`, validation errors to generic `422`, and store errors to generic `503`.

- [ ] **Step 5: Register the router and pin CSRF behavior**

Import and include `analytics_router` in `dashboard/backend/api/router.py`.

Add a `test_csrf.py` case that signs in with `ATL_CSRF=1`, proves the endpoint rejects a cookie-authenticated POST without `X-CSRF-Token`, and accepts the same synthetic payload with `_csrf_headers(client)`.

- [ ] **Step 6: Run API and CSRF tests and commit**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/test_analytics_api.py \
  dashboard/backend/tests/test_csrf.py
```

Expected: PASS.

Commit:

```bash
git add \
  dashboard/backend/api/routers/analytics.py \
  dashboard/backend/api/router.py \
  dashboard/backend/tests/test_analytics_api.py \
  dashboard/backend/tests/test_csrf.py
git commit -m "feat: add authenticated analytics ingestion"
```

---

### Task 6: Add browser analytics sessions and major-page lifecycle events

**Files:**
- Create: `dashboard/frontend/js/analytics.js`
- Create: `dashboard/backend/tests/test_analytics_frontend.py`
- Modify: `dashboard/frontend/app.html`
- Modify: `dashboard/frontend/app.js`
- Modify: `dashboard/frontend/js/agent-editor.js`
- Modify: `dashboard/backend/tests/test_frontend_fast_boot.py`

**Interfaces:**
- Consumes: `window.getStoredAuthUser`, `window.csrfHeaders`, navigation state from `navigateToPage`, and `POST /api/analytics/events`.
- Produces: `window.ATLAnalytics.recordNavigation(page, options)`, `window.ATLAnalytics.enterTransientView(pageView)`, and `window.ATLAnalytics.leaveTransientView()`.

- [ ] **Step 1: Write failing static frontend safety and lifecycle tests**

Create `test_analytics_frontend.py` and read `app.html`, `app.js`, `js/agent-editor.js`, and `js/analytics.js` as text.

Pin these contracts:

- `analytics.js` is deferred after `app.js` and before page-specific scripts that may navigate.
- The session storage key is `atl-analytics-session-v1`.
- Session timeout is exactly `30 * 60 * 1000`.
- Heartbeat interval is exactly `30 * 1000`.
- Only the nine approved page identifiers appear in `PAGE_VIEW_MAP`.
- The module sends only `event_id`, `schema_version`, `event_name`, `session_id`, `occurred_at`, `page_view`, and `properties`.
- It uses `credentials: 'include'`, `keepalive: true`, and `window.csrfHeaders()`.
- It does not contain payload fields named `email`, `display_name`, `role`, `api_key`, `token`, `prompt`, `strategy`, `provider_response`, or `user_agent`.
- `navigateToPage` calls `window.ATLAnalytics.recordNavigation(page, options)` only after the final page/subtab normalization.
- Agent editor `open` calls `enterTransientView('agent_editor')` and `close` calls `leaveTransientView()`.
- Visibility-hidden sends `page_hidden`; visible restoration sends `page_viewed`.
- Heartbeats are skipped when signed out or `document.visibilityState !== 'visible'`.
- The module never throws an ingestion failure into navigation; rejected fetches are swallowed after an optional fixed, data-free warning.

- [ ] **Step 2: Run the frontend contract and verify it fails**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q dashboard/backend/tests/test_analytics_frontend.py
```

Expected: FAIL because `analytics.js` is absent.

- [ ] **Step 3: Implement the tab-scoped anonymous Analytics session**

In `analytics.js`:

1. Store `{"id": crypto.randomUUID(), "last_activity_at": <epoch-ms>}` in `sessionStorage`.
2. Reuse the ID while accepted page activity is less than 30 minutes old.
3. Rotate after 30 minutes, on malformed state, or when the ID is not a canonical UUID.
4. Never reuse `trading-session-id`, `browser-owner-id`, an auth token, an Agent ID, or a run ID.
5. Update `last_activity_at` only when queueing one of the three Analytics events, not during auth refresh or unrelated polling.

- [ ] **Step 4: Implement safe event delivery and view mapping**

Use `fetch(API_BASE + '/api/analytics/events', {...})` with a JSON body containing only the seven approved client fields. Resolve navigation as:

```javascript
function resolvePageView(page, options) {
  if (page === 'home') return 'home';
  if (page === 'credits') return 'credits';
  if (page === 'account') return 'account';
  if (page === 'community') return 'community';
  if (page === 'competition') return 'competition';
  if (page !== 'playground') return null;
  const tab = options.playgroundTab || document.documentElement.dataset.navPlaygroundTab;
  if (tab === 'agents') return 'agents';
  if (tab === 'backtest') return 'backtest';
  if (tab === 'paper') return 'paper_trading';
  return null;
}
```

`recordNavigation` must send `page_hidden` for the previous different view, then `page_viewed` for the new view. Re-entering the same view without a visibility transition must not duplicate the event.

Use `visible_ms` only on `page_hidden` and `session_heartbeat`, clamp it to 0 through 1,800,000, and never read form values or DOM text.

- [ ] **Step 5: Integrate navigation, Agent editor, visibility, and heartbeat**

- Call `window.ATLAnalytics?.recordNavigation(page, { playgroundTab, competitionTab })` near the end of `navigateToPage` after the page is visible and navigation state has been normalized.
- Call `enterTransientView('agent_editor')` after `agentEditorView.hidden = false`.
- Call `leaveTransientView()` after the editor is hidden.
- On `DOMContentLoaded`, resolve the already-rendered page and send one initial `page_viewed` only when `getStoredAuthUser()` returns a user.
- On `visibilitychange`, send hidden/visible lifecycle events.
- Every 30 seconds, send `session_heartbeat` only while signed in, visible, and holding a current page.
- On `pagehide`, queue a final `page_hidden` using `keepalive`; do not use `sendBeacon` because it cannot carry the existing CSRF header.

- [ ] **Step 6: Add the deferred script and cache versions**

Add:

```html
<script src="js/analytics.js?v=1" defer></script>
```

immediately after `app.js`. Increment `app.js` and `agent-editor.js` cache-buster versions exactly once, and update the exact floors in `test_frontend_fast_boot.py`.

- [ ] **Step 7: Run frontend checks and commit**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/test_analytics_frontend.py \
  dashboard/backend/tests/test_frontend_fast_boot.py
node --check dashboard/frontend/js/analytics.js
node --check dashboard/frontend/app.js
node --check dashboard/frontend/js/agent-editor.js
```

Expected: PASS.

Commit:

```bash
git add \
  dashboard/frontend/js/analytics.js \
  dashboard/frontend/app.html \
  dashboard/frontend/app.js \
  dashboard/frontend/js/agent-editor.js \
  dashboard/backend/tests/test_analytics_frontend.py \
  dashboard/backend/tests/test_frontend_fast_boot.py
git commit -m "feat: collect safe analytics page events"
```

---

### Task 7: Add bounded retention and operator-visible failure reporting

**Files:**
- Create: `dashboard/backend/domain/analytics/retention.py`
- Create: `dashboard/backend/tests/domain/analytics/test_retention.py`
- Modify: `dashboard/backend/app.py`
- Modify: `dashboard/backend/tests/test_run_lifecycle_unification.py`
- Modify: `dashboard/backend/tests/conftest.py`
- Modify: `render.yaml`

**Interfaces:**
- Consumes: repository `delete_expired`, existing `register_reaper_sweep`, and UTC time.
- Produces: `AnalyticsRetentionService.run_once(now=None)`, `AnalyticsRetentionCoordinator.run_if_due()`, singleton `analytics_retention_coordinator`, and startup registration with the existing reaper.

- [ ] **Step 1: Write failing retention tests**

Create `test_retention.py` with a fake store and deterministic clock. Assert:

```python
def test_run_once_uses_180_and_365_day_cutoffs():
    store = RecordingStore()
    service = AnalyticsRetentionService(store=store, batch_size=500)
    result = service.run_once(datetime(2026, 8, 26, tzinfo=timezone.utc))
    assert store.calls == [{
        "raw_before": datetime(2026, 2, 27, tzinfo=timezone.utc),
        "access_before": datetime(2025, 8, 26, tzinfo=timezone.utc),
        "batch_size": 500,
    }]
    assert result.raw_events_deleted == 0


def test_coordinator_runs_at_most_once_per_24_hours():
    clock = FakeClock()
    service = RecordingRetentionService()
    coordinator = AnalyticsRetentionCoordinator(service=service, clock=clock)
    coordinator.run_if_due()
    coordinator.run_if_due()
    assert service.calls == 1
    clock.advance(24 * 60 * 60)
    coordinator.run_if_due()
    assert service.calls == 2


def test_retention_failure_is_swallowed_and_reports_only_safe_metadata(capsys):
    coordinator = AnalyticsRetentionCoordinator(
        service=FailingRetentionService("synthetic-secret-canary"),
        clock=lambda: 0,
    )
    assert coordinator.run_if_due() is None
    output = capsys.readouterr().out
    assert "analytics.retention_failed" in output
    assert "consecutive_failures=1" in output
    assert "synthetic-secret-canary" not in output
```

Add a real SQLite retention test that inserts rows on each side of both cutoffs, runs with `batch_size=1` until both `has_more` flags are false, and proves:

- old raw events are deleted;
- recent raw events remain;
- old access rows are deleted;
- recent access rows remain;
- a seeded daily rollup remains;
- a seeded current snapshot remains;
- each repository call deletes at most one row from each retained table.

- [ ] **Step 2: Run retention tests and verify they fail**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q dashboard/backend/tests/domain/analytics/test_retention.py
```

Expected: FAIL because `retention.py` does not exist.

- [ ] **Step 3: Implement the retention service and coordinator**

Use constants:

```python
RAW_EVENT_RETENTION_DAYS = 180
ADMIN_ACCESS_RETENTION_DAYS = 365
RETENTION_BATCH_SIZE = 1000
RETENTION_INTERVAL_SECONDS = 24 * 60 * 60
MAX_BATCHES_PER_RUN = 20
```

`run_once` must call repository deletion repeatedly while either `has_more` flag is true, stop after 20 batches, sum deletion counts, and return a `RetentionResult`. The coordinator must:

- use a lock so concurrent reaper passes cannot overlap;
- skip until the 24-hour monotonic deadline;
- catch every exception;
- increment `consecutive_failures`;
- print only `WARNING: analytics.retention_failed consecutive_failures=N category=RuntimeError`-shaped metadata, with the category produced by `type(exc).__name__`;
- reset the failure count after success;
- return without raising.

- [ ] **Step 4: Register retention with the existing run reaper**

In `startup_event`, import `analytics_retention_coordinator.run_if_due` and register it with `register_reaper_sweep` in its own `try/except` block. Do not create a new thread or timer.

Extend `test_run_lifecycle_unification.py` to assert startup registration includes the Analytics retention sweep exactly once and that a raising retention sweep cannot prevent the next registered sweep.

- [ ] **Step 5: Document deployment pseudonymization configuration**

Add to `render.yaml`:

```yaml
      # Monthly HMAC key for non-reversible Analytics network grouping.
      # Missing/invalid values omit network_hash; Analytics never stores plaintext IP.
      - key: ANALYTICS_PSEUDONYMIZATION_KEY
        sync: false
```

Add `os.environ.pop("ANALYTICS_PSEUDONYMIZATION_KEY", None)` to `tests/conftest.py` before backend imports so a developer's real deployment secret cannot affect tests.

- [ ] **Step 6: Run retention and lifecycle tests and commit**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/domain/analytics/test_retention.py \
  dashboard/backend/tests/test_run_lifecycle_unification.py
```

Expected: PASS.

Commit:

```bash
git add \
  dashboard/backend/domain/analytics/retention.py \
  dashboard/backend/app.py \
  dashboard/backend/tests/domain/analytics/test_retention.py \
  dashboard/backend/tests/test_run_lifecycle_unification.py \
  dashboard/backend/tests/conftest.py \
  render.yaml
git commit -m "feat: enforce analytics data retention"
```

---

### Task 8: Run the PR 1 security and parity acceptance gate

**Files:**
- Modify only if a failing test identifies a PR 1 defect.

**Interfaces:**
- Consumes: all Task 1 through Task 7 deliverables.
- Produces: a clean, reviewable PR 1 foundation branch with no local data or secret material.

- [ ] **Step 1: Run the focused Analytics suite**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/domain/analytics \
  dashboard/backend/tests/test_analytics_api.py \
  dashboard/backend/tests/test_analytics_frontend.py \
  dashboard/backend/tests/test_csrf.py \
  dashboard/backend/tests/test_frontend_fast_boot.py \
  dashboard/backend/tests/test_run_lifecycle_unification.py
```

Expected: PASS. If any test fails, stop here, fix only the first failing layer, rerun this command, and do not proceed until it is green.

- [ ] **Step 2: Run adjacent account/database regression tests**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/test_auth.py \
  dashboard/backend/tests/test_admin_users.py \
  dashboard/backend/tests/test_users_postgres.py \
  dashboard/backend/tests/test_model_credentials_api.py \
  dashboard/backend/tests/domain/credits/test_repository_postgres.py
```

Expected: PASS with live PostgreSQL cases skipped when `TEST_POSTGRES_URL` is unset.

- [ ] **Step 3: Run JavaScript syntax and whitespace checks**

Run:

```bash
node --check dashboard/frontend/js/analytics.js
node --check dashboard/frontend/app.js
node --check dashboard/frontend/js/agent-editor.js
git diff --check
```

Expected: all commands exit 0 with no output from `git diff --check`.

- [ ] **Step 4: Scan for prohibited data and local artifacts**

Run:

```bash
git diff --cached --name-only
git status --short
rg -n \
  "synthetic-secret-canary|BEGIN PRIVATE KEY|sk-[A-Za-z0-9]|api_key_enc|provider_response_body" \
  dashboard/backend/domain/analytics \
  dashboard/backend/api/routers/analytics.py \
  dashboard/frontend/js/analytics.js \
  dashboard/backend/tests/domain/analytics \
  dashboard/backend/tests/test_analytics_api.py \
  dashboard/backend/tests/test_analytics_frontend.py
```

Expected:

- only explicit synthetic negative-test canaries appear;
- no real-looking key or private-key material appears;
- no `.superpowers/`, `work/`, `*.db`, `*.db-wal`, or `*.db-shm` path is staged.

- [ ] **Step 5: Inspect the complete branch diff**

Run:

```bash
git diff --stat 1aecb635edcd8792d883cf4b091ccbdf5046ee4d..HEAD
git diff --name-only 1aecb635edcd8792d883cf4b091ccbdf5046ee4d..HEAD
```

Expected: only PR 1 foundation, tests, `render.yaml`, and this plan/spec documentation are present; no Admin Analytics UI, metrics, rollups computation, state calculation, backfill, or Admin query API implementation is included.

- [ ] **Step 6: Commit any final test-only correction**

Only if Step 1 through Step 5 required a correction:

```bash
git add \
  dashboard/backend/domain/analytics \
  dashboard/backend/api/routers/analytics.py \
  dashboard/backend/api/router.py \
  dashboard/backend/app.py \
  dashboard/frontend/js/analytics.js \
  dashboard/frontend/js/agent-editor.js \
  dashboard/frontend/app.js \
  dashboard/frontend/app.html \
  dashboard/backend/tests/domain/analytics \
  dashboard/backend/tests/test_analytics_api.py \
  dashboard/backend/tests/test_analytics_frontend.py \
  dashboard/backend/tests/test_csrf.py \
  dashboard/backend/tests/test_frontend_fast_boot.py \
  dashboard/backend/tests/test_run_lifecycle_unification.py \
  dashboard/backend/tests/test_ci_postgres_wired.py \
  dashboard/backend/tests/conftest.py \
  render.yaml
git commit -m "test: harden analytics foundation contracts"
```

If no correction was required, do not create an empty commit.

- [ ] **Step 7: Prepare the pull request**

Use:

```text
Title: Add privacy-safe Admin Analytics foundation

Summary:
- add equivalent SQLite and PostgreSQL Analytics event repositories
- enforce authenticated allowlisted page-event ingestion and pseudonymous sessions
- add subject exclusions, Admin profile-access auditing, and bounded retention

Testing:
- focused Analytics, API, CSRF, frontend, and lifecycle suites
- adjacent auth, Admin user, credential, Credits, and PostgreSQL regression suites
- JavaScript syntax checks and git diff validation

Safety:
- no real API keys or raw sensitive Analytics fields
- missing pseudonymization configuration omits network hashes
- no Admin Analytics UI or metrics are included in this PR
```

Push the current branch and open the PR only after every required test above is green.
