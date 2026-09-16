# Admin Layer PR T: ValueAnalyticsStore Postgres Twin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract a real `PostgresValueAnalyticsStore` twin for `ValueAnalyticsStore` (today 21 `is_postgres` branches and 3 `hasattr(..., "database_url")` checks, no `*_postgres.py` file), register the pair in `tests/test_store_twin_parity.py::_TWINS`, remove that registry's duplicate `AnalyticsStore` entry, and add an absence-direction check so a store that inlines its own dialect branching can never again go unnoticed the way this one did through the whole PR A/B build-out.

**Architecture:** `ValueAnalyticsStore` keeps every SQLite code path and drops every `is_postgres` branch that chose between SQLite and Postgres SQL against its own tables (`user_analytics_snapshots`, `user_lifecycle_daily_snapshots`, `analytics_projection_jobs`); the new `PostgresValueAnalyticsStore` (`value_repository_postgres.py`) carries the Postgres half of those same branches, unchanged. Both classes keep `list_commercial_values`/`list_credit_activity` verbatim and identical, because those two branch on a *different* store's dialect (`self.credits_base`), never on this class's own — a caller can pair either `analytics_base` with either `credits_base`, and both twins must handle it. A new `build_value_analytics_store()` factory in `value_repository.py` picks the twin the same way `repository.py::_build_analytics_store()` already picks `AnalyticsStore`/`PostgresAnalyticsStore` (by inspecting the resolved `analytics_base`), and every production construction site switches to it so a Postgres deployment keeps working exactly as it does today.

**Tech Stack:** Python 3, SQLite, PostgreSQL/psycopg 3, pytest.

**Spec:** docs/superpowers/specs/2026-09-15-admin-layer-redesign-design.md — implements §4.5 (the `ValueAnalyticsStore` inventory: 21 `is_postgres` branches, no twin file, `_TWINS` registers 11 distinct pairs because the `AnalyticsStore` tuple is listed twice), §6.14 (the ownership-boundaries row "Every store has a Postgres twin, and the parity guard can see a store with no twin", fixed in "PR T"), §12 items 2 and 8 (the corrections this PR exists to make: about twelve dual-dialect methods would land on the untwinned class if PR A shipped first; `_TWINS` lists the `AnalyticsStore` pair twice), and §13 row **T** ("Store twin" — extract the postgres module, register it, remove the duplicate, add the absence-direction check; behaviour-preserving; green on the CI Postgres tier before merge), plus D22 ("The `ValueAnalyticsStore` Postgres twin is extracted before PR A, as its own PR... Extracting now is a day; after PR A it is several").

## Global Constraints

- Behaviour-preserving only (§13 row T, "Must not: Change any query, response or table"). No SQL, no response shape, no table changes — only where the dialect branch that produces that SQL lives.
- Never commit `dashboard/storage/data/backtest.db` — a bare backend import rewrites it; stage files by name (`git add <path> <path>`), never `-A` and never a bare `git add -u`.
- Run pytest from the repo root.
- Node-driven frontend tests are `skipif(shutil.which("node") is None)` — not applicable to this plan (no frontend files touch), noted because a reviewer checking "did this PR touch the frontend" should find the answer is no.
- The `?v=` pins in `test_admin_analytics_frontend.py::test_app_lifecycle_and_cache_versions_are_wired` must be updated in lockstep with any bump — not applicable here either, for the same reason.
- Every store must have a Postgres twin, and the parity guard must be able to see a store with none (design §6.14) — this is the rule PR T exists to satisfy for `ValueAnalyticsStore`; Task 4 generalizes the check so the next untwinned store cannot repeat this silently.
- CI's Postgres tier (`@pg_only`, gated on `TEST_POSTGRES_URL`) must be green before merge (§13 row T).

## File Map

- `dashboard/backend/domain/analytics/value_repository.py`: reduced to SQLite-only code paths; four private helpers move to module scope; gains `build_value_analytics_store()`.
- `dashboard/backend/domain/analytics/value_repository_postgres.py`: new. `PostgresValueAnalyticsStore`.
- `dashboard/backend/domain/analytics/service.py`, `states.py`, `lifecycle_backfill.py`, `retention.py`, `value_queries.py`: switch construction from `ValueAnalyticsStore(...)` to `build_value_analytics_store(...)`.
- `dashboard/backend/tests/test_store_twin_parity.py`: dedupe `_TWINS`, register the new pair, add `_NO_OWN_DDL_TWINS` tolerance, add the absence-direction check and its allowlist.
- `dashboard/backend/tests/domain/analytics/test_value_repository_postgres.py`: new.
- `dashboard/backend/tests/domain/analytics/test_repository_postgres.py`: `test_postgres_user_value_projection_round_trip` moves onto `PostgresValueAnalyticsStore`.
- `CLAUDE.md`: Persistence bullet gains one clause.

---

### Task 1: Registry hygiene — remove the duplicate `_TWINS` tuple

**Files:**
- Modify: `dashboard/backend/tests/test_store_twin_parity.py:54-127` (`_TWINS`)
- Test: `dashboard/backend/tests/test_store_twin_parity.py` (extend, same file)

**Interfaces:**
- Consumes: nothing.
- Produces: `_TWINS` with 11 distinct tuples instead of 12 (the `AnalyticsStore` pair once, not twice). No public function signatures change.

- [ ] **Step 1: Write the failing test.**

Add to `dashboard/backend/tests/test_store_twin_parity.py`, directly after `_TWIN_IDS = [pg_cls for _, _, _, pg_cls in _TWINS]` (line 129):

```python
def test_twins_registry_has_no_duplicate_pairs():
    """`_TWINS` listed the AnalyticsStore pair twice (lines 55-60 and 109-114).

    Harmless today -- both instances of a duplicate tuple pass or fail
    together -- but it is the exact list PR T's absence-direction check and
    every future twin extends, so a reader counting entries gets 12 when
    there are 11 distinct stores. `_TWIN_IDS` uses the postgres class name
    as the parametrize id, so a duplicate also means two test instances
    sharing one id in every parametrized case above.
    """
    assert len(_TWINS) == len(set(_TWINS)), (
        f"_TWINS has {len(_TWINS) - len(set(_TWINS))} duplicate tuple(s); "
        "each store pair belongs in the registry exactly once."
    )
```

- [ ] **Step 2: Run the test and confirm it fails.**

```bash
python -m pytest dashboard/backend/tests/test_store_twin_parity.py::test_twins_registry_has_no_duplicate_pairs -v
```

Expected: FAIL. `_TWINS` has 12 entries; `set(_TWINS)` collapses to 11 because the `AnalyticsStore` tuple at lines 55-60 and 109-114 is byte-identical, so the length comparison reports a 1-entry gap.

- [ ] **Step 3: Remove the duplicate tuple.**

In `dashboard/backend/tests/test_store_twin_parity.py`, delete this entry (currently lines 109-114, the tuple immediately after the `StrategyStore` entry and immediately before the `dashboard.backend.users` / `UserStore` entry; match by content, not by line number, if the file has moved):

```python
    (
        "dashboard.backend.domain.analytics.repository",
        "AnalyticsStore",
        "dashboard.backend.domain.analytics.repository_postgres",
        "PostgresAnalyticsStore",
    ),
```

After deletion `_TWINS` has 11 entries: `AnalyticsStore` (once, first position), `ModelProviderStore`, `CreditsStore`, `AgentCredentialStore`, `AgentStore`, `AgentVersionStore`, `BrokerConnectionStore`, `PortfolioStore`, `StrategyStore`, `UserStore`, `BacktestDatabase`.

- [ ] **Step 4: Run the test, then the full parametrized suite, for regressions.**

```bash
python -m pytest dashboard/backend/tests/test_store_twin_parity.py -v
```

Expected: PASS on every case. Every test parametrized over `_TWINS` now runs 11 instances instead of 12; none of them asserts a count, so removing the duplicate changes no other test's outcome.

- [ ] **Step 5: Commit.**

```bash
git add dashboard/backend/tests/test_store_twin_parity.py
git status --short
git commit -m "test: remove the duplicate AnalyticsStore entry from _TWINS"
```

---

### Task 2: Extract `PostgresValueAnalyticsStore`; reduce `ValueAnalyticsStore` to SQLite-only

**Files:**
- Create: `dashboard/backend/domain/analytics/value_repository_postgres.py`
- Modify: `dashboard/backend/domain/analytics/value_repository.py` (full-file rewrite; source read in full, 1125 lines)
- Modify: `dashboard/backend/domain/analytics/service.py:237,243`
- Modify: `dashboard/backend/domain/analytics/states.py:23-27,670,691,754`
- Modify: `dashboard/backend/domain/analytics/lifecycle_backfill.py:16-20,146,406`
- Modify: `dashboard/backend/domain/analytics/retention.py:12,172-179`
- Modify: `dashboard/backend/domain/analytics/value_queries.py:28-33,494`
- Test: `dashboard/backend/tests/domain/analytics/test_value_repository_postgres.py` (create)
- Test: `dashboard/backend/tests/domain/analytics/test_repository_postgres.py` (modify `test_postgres_user_value_projection_round_trip` and its imports)

**Interfaces:**
- Consumes: nothing from Task 1 (independent files).
- Produces: `PostgresValueAnalyticsStore` in `dashboard.backend.domain.analytics.value_repository_postgres`, with the same 14 public methods and constructor signature as `ValueAnalyticsStore`: `__init__(self, analytics_base=None, credits_base=None, provider_base=None, agent_base=None, run_base=None)`. `build_value_analytics_store(analytics_base=None, credits_base=None, provider_base=None, agent_base=None, run_base=None)` in `dashboard.backend.domain.analytics.value_repository`, returning `PostgresValueAnalyticsStore` when the resolved `analytics_base` exposes `database_url`, else `ValueAnalyticsStore`. Task 3 registers the pair in `_TWINS`; Task 4's allowlist names both new file paths.

- [ ] **Step 1: Write the failing tests.**

Create `dashboard/backend/tests/domain/analytics/test_value_repository_postgres.py`:

```python
"""Structural parity checks for PostgresValueAnalyticsStore.

Neither test here needs TEST_POSTGRES_URL: both check the class shape and
the factory's dispatch logic with plain Python objects, never a live
connection. The behavioural round-trip against a real database lives in
tests/domain/analytics/test_repository_postgres.py
(test_postgres_user_value_projection_round_trip, @pg_only), which already
had a live-Postgres tier before this store had a twin to put on it -- this
file only has to point that existing test at the new class.
"""

from __future__ import annotations

import inspect

from dashboard.backend.domain.analytics.value_repository import (
    ValueAnalyticsStore,
    build_value_analytics_store,
)


def _public_methods(cls) -> set[str]:
    return {
        name
        for name in dir(cls)
        if not name.startswith("_") and callable(getattr(cls, name, None))
    }


def test_postgres_value_analytics_store_matches_sqlite_public_surface():
    from dashboard.backend.domain.analytics.value_repository_postgres import (
        PostgresValueAnalyticsStore,
    )

    sqlite_methods = _public_methods(ValueAnalyticsStore)
    postgres_methods = _public_methods(PostgresValueAnalyticsStore)
    assert sqlite_methods == postgres_methods, (
        f"sqlite-only={sorted(sqlite_methods - postgres_methods)} "
        f"postgres-only={sorted(postgres_methods - sqlite_methods)}"
    )
    for name in sqlite_methods:
        assert inspect.signature(
            getattr(ValueAnalyticsStore, name)
        ) == inspect.signature(getattr(PostgresValueAnalyticsStore, name)), (
            f"{name} signature diverges between the twins"
        )


class _FakePostgresBase:
    """A stand-in analytics_base with no real connection.

    build_value_analytics_store() dispatches on ``hasattr(base,
    "database_url")`` alone, so a plain attribute exercises the branch
    without a live database -- exactly like the ``object()`` sentinels
    domain/analytics/retention.py already passes for credits_base/
    provider_base/agent_base/run_base when it never calls those stores.
    """

    database_url = "postgresql://example/test"


def test_build_value_analytics_store_selects_postgres_twin_for_a_postgres_base():
    from dashboard.backend.domain.analytics.value_repository_postgres import (
        PostgresValueAnalyticsStore,
    )

    store = build_value_analytics_store(
        _FakePostgresBase(),
        credits_base=object(),
        provider_base=object(),
        agent_base=object(),
        run_base=object(),
    )
    assert isinstance(store, PostgresValueAnalyticsStore)


def test_build_value_analytics_store_selects_sqlite_twin_by_default():
    class _FakeSqliteBase:
        pass

    store = build_value_analytics_store(
        _FakeSqliteBase(),
        credits_base=object(),
        provider_base=object(),
        agent_base=object(),
        run_base=object(),
    )
    assert isinstance(store, ValueAnalyticsStore)
```

Then edit `dashboard/backend/tests/domain/analytics/test_repository_postgres.py`. Replace the import block (lines 22-26):

```python
from dashboard.backend.domain.analytics.value_repository import (
    ProjectionJob,
    UserLifecycleDailySnapshot,
    ValueAnalyticsStore,
)
```

with:

```python
from dashboard.backend.domain.analytics.value_repository import (
    ProjectionJob,
    UserLifecycleDailySnapshot,
)
from dashboard.backend.domain.analytics.value_repository_postgres import (
    PostgresValueAnalyticsStore,
)
```

and inside `test_postgres_user_value_projection_round_trip` replace:

```python
    analytics, _admin_id, user_id = postgres_contract_store
    value_store = ValueAnalyticsStore(
        analytics,
        SyntheticCreditsStore(tmp_path / "postgres-value-credits.db"),
        provider_base=SyntheticProviderStore({}, []),
        agent_base=SyntheticAgentStore(),
        run_base=SyntheticRunStore({}),
    )
```

with:

```python
    analytics, _admin_id, user_id = postgres_contract_store
    value_store = PostgresValueAnalyticsStore(
        analytics,
        SyntheticCreditsStore(tmp_path / "postgres-value-credits.db"),
        provider_base=SyntheticProviderStore({}, []),
        agent_base=SyntheticAgentStore(),
        run_base=SyntheticRunStore({}),
    )
```

`ValueAnalyticsStore` is not referenced anywhere else in this file, so removing it from the import is correct rather than incidental.

- [ ] **Step 2: Run the new and edited tests, and confirm they fail.**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_value_repository_postgres.py dashboard/backend/tests/domain/analytics/test_repository_postgres.py::test_postgres_user_value_projection_round_trip -v
```

Expected: collection/FAIL. `test_value_repository_postgres.py` fails to import `build_value_analytics_store` from `value_repository` (`ImportError`, does not exist yet), which fails collection for the whole file. `test_repository_postgres.py` fails to import `dashboard.backend.domain.analytics.value_repository_postgres` (`ModuleNotFoundError`, the file does not exist yet).

- [ ] **Step 3: Replace `value_repository.py` in full** with the SQLite-only version. Four private helpers (`_current_snapshot_from_row`, `_fetchall`, `_user_clause`, `_projection_job_name`) move from `@staticmethod`s on the class to bare module functions — nothing outside this module referenced them as staticmethods (verified: `rg -rn "_fetchall\|_user_clause\|_current_snapshot_from_row\|_projection_job_name" dashboard/backend --include="*.py"` outside this file returns nothing), so this is not a behaviour change; it lets `value_repository_postgres.py` import them the same way `repository_postgres.py` already imports bare functions (`_row_to_event`, `_event_values`) from `repository.py`. Every `if self.is_postgres: ... else: ...` branch collapses to its SQLite arm; `self.is_postgres` is deleted from `__init__` (nothing outside the class read it). `list_commercial_values`, `list_credit_activity`, `get_operational_facts` and `_run_health` are unchanged except for calling the now-bare `_fetchall`/`_user_clause` without `self.`:

```python
"""Persistence primitives for the user-value Analytics projection.

This module deliberately keeps the Credits ledger authoritative.  Analytics
stores only calculated lifecycle history and reads commercial facts in batches.

SQLite twin. The PostgreSQL twin is ``value_repository_postgres.py``
(``PostgresValueAnalyticsStore``); ``build_value_analytics_store()`` below
selects between them the same way ``repository.py``'s ``_build_analytics_store()``
selects between ``AnalyticsStore`` and ``PostgresAnalyticsStore``. Every method
that reads or writes this store's own tables (``user_analytics_snapshots``,
``user_lifecycle_daily_snapshots``, ``analytics_projection_jobs``) has exactly
one code path here; the two methods that branch on a *different* store's
dialect (``list_commercial_values``, ``list_credit_activity``, both reading
``self.credits_base``) are unchanged and identical on both twins, because that
branch was never about which twin this class is -- a caller can pair either
``analytics_base`` with either ``credits_base``, and both twins must handle it.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .lifecycle import (
    CommercialTier,
    LifecycleSegment,
    OperationalState,
    commercial_tier,
)
from .repository import analytics_store
from .repository_common import positive_limit, positive_user_id, utc_iso


MAX_USER_BATCH = 500
RUN_SAFE_DEADLINE = timedelta(minutes=60)
LIFECYCLE_SEGMENTS = frozenset(
    {"new", "onboarding", "growing", "core", "at_risk", "dormant"}
)
LIFECYCLE_ROLLUP_METRICS = frozenset(
    {"lifecycle_segment_count", "lifecycle_transition"}
)
_ACTIVE_RUN_STATUSES = frozenset({"created", "loading", "running"})
_TERMINAL_RUN_STATUSES = frozenset(
    {"completed", "failed", "cancelled", "closed", "timed_out"}
)


def _utc(value: datetime, name: str = "timestamp") -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return value.astimezone(timezone.utc)


def _timestamp(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _row_value(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    try:
        return row[name]
    except (IndexError, KeyError, TypeError):
        return default


def _object_value(value: object, name: str, default: Any = 0) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _optional_timestamp(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        return _timestamp(value)
    except (TypeError, ValueError):
        return None


class UserValueSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: int = Field(gt=0)
    lifecycle_segment: LifecycleSegment
    lifecycle_reason_code: str = Field(min_length=1, max_length=100)
    lifecycle_reason: str = Field(min_length=1, max_length=500)
    lifecycle_evidence: Sequence[str] = Field(default_factory=tuple, max_length=10)
    operational_state: OperationalState
    operational_reason_code: str = Field(min_length=1, max_length=100)
    operational_reason: str = Field(min_length=1, max_length=500)
    operational_evidence: Sequence[str] = Field(default_factory=tuple, max_length=10)
    activated_at: datetime | None = None
    last_meaningful_activity_at: datetime | None = None
    inactive_days: int = Field(ge=0)
    active_days_30d: int = Field(ge=0, le=30)
    successful_backtests_30d: int = Field(ge=0)
    calculated_at: datetime

    @field_validator("activated_at", "last_meaningful_activity_at", "calculated_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        return _utc(value) if value is not None else None


class UserLifecycleDailySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot_date: date
    user_id: int = Field(gt=0)
    lifecycle_segment: LifecycleSegment
    lifecycle_reason_code: str = Field(min_length=1, max_length=100)
    data_quality: Literal["complete", "partial"]
    calculated_at: datetime

    @field_validator("calculated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return _utc(value)


class CommercialValueFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: int = Field(gt=0)
    lifetime_net_purchased_micro: int = Field(ge=0)
    commercial_tier: CommercialTier
    purchased_micro: int = Field(ge=0)
    refunded_micro: int = Field(ge=0)
    consumed_micro: int = Field(ge=0)
    admin_grant_activity_micro: int = Field(ge=0)
    grant_available_micro: int = Field(ge=0)
    purchased_available_micro: int = Field(ge=0)
    total_available_micro: int = Field(ge=0)


class CurrentOperationalFacts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: int = Field(gt=0)
    account_restricted: bool = False
    usable_billing_lane: bool = True
    selected_provider_enabled: bool = True
    default_credential_status: Literal[
        "verified", "invalid", "verification_unavailable", "missing"
    ] = "verified"
    failed_terminal_runs_24h: int = Field(default=0, ge=0)
    run_beyond_safe_deadline: bool = False


class ProjectionJob(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    job_name: str = Field(min_length=1, max_length=100)
    window_start: date
    window_end: date
    cursor: str | None = None
    status: Literal["pending", "running", "complete"]
    updated_at: datetime

    @field_validator("updated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return _utc(value)

    @model_validator(mode="after")
    def validate_window(self) -> "ProjectionJob":
        if self.window_end < self.window_start:
            raise ValueError("window_end must not precede window_start")
        return self


def _ids(user_ids: Sequence[int]) -> list[int]:
    if not isinstance(user_ids, (list, tuple)):
        raise ValueError("user_ids must be a list or tuple")
    values = list(dict.fromkeys(positive_user_id(item) for item in user_ids))
    if len(values) > MAX_USER_BATCH:
        raise ValueError(f"user_ids must contain at most {MAX_USER_BATCH} users")
    return values


def _validate_window(start: datetime, end: datetime) -> tuple[datetime, datetime]:
    window_start = _utc(start, "start")
    window_end = _utc(end, "end")
    if window_end <= window_start:
        raise ValueError("end must be later than start")
    return window_start, window_end


def _legacy_seed(snapshot: UserValueSnapshot) -> tuple[str, str, str]:
    """Seed compatibility fields only when no legacy projection exists yet."""

    if snapshot.operational_state == "blocked":
        status = "blocked"
        reason_code = snapshot.operational_reason_code
        reason = snapshot.operational_reason
    elif snapshot.operational_state == "needs_attention":
        status = "needs_attention"
        reason_code = snapshot.operational_reason_code
        reason = snapshot.operational_reason
    elif snapshot.lifecycle_segment == "dormant":
        status = "dormant"
        reason_code = snapshot.lifecycle_reason_code
        reason = snapshot.lifecycle_reason
    elif snapshot.lifecycle_segment in {"new", "onboarding"}:
        status = "onboarding"
        reason_code = snapshot.lifecycle_reason_code
        reason = snapshot.lifecycle_reason
    else:
        status = "active"
        reason_code = snapshot.lifecycle_reason_code
        reason = snapshot.lifecycle_reason
    return status, reason_code, reason


def _current_snapshot_from_row(row: Any) -> UserValueSnapshot | None:
    """Shared by both twins.

    Moved out of the class (was ``ValueAnalyticsStore._current_snapshot_from_row``,
    a ``@staticmethod``) so ``value_repository_postgres.py`` can import it
    directly, mirroring how ``repository_postgres.py`` imports bare functions
    (``_row_to_event``) from ``repository.py``. No caller outside this module
    referenced the staticmethod, so this is not a behaviour change.
    """
    if row is None or _row_value(row, "lifecycle_segment") is None:
        return None

    def seq(name: str) -> tuple[str, ...]:
        try:
            value = json.loads(_row_value(row, name, "[]"))
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                return ()
            return tuple(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return ()

    return UserValueSnapshot(
        user_id=int(_row_value(row, "user_id")),
        lifecycle_segment=_row_value(row, "lifecycle_segment"),
        lifecycle_reason_code=_row_value(row, "lifecycle_reason_code"),
        lifecycle_reason=_row_value(row, "lifecycle_reason"),
        lifecycle_evidence=seq("lifecycle_evidence_json"),
        operational_state=_row_value(row, "operational_state") or "healthy",
        operational_reason_code=(
            _row_value(row, "operational_reason_code") or "no_supported_issue"
        ),
        operational_reason=(
            _row_value(row, "operational_reason")
            or "No supported current operational issue was detected."
        ),
        operational_evidence=seq("operational_evidence_json"),
        activated_at=_optional_timestamp(_row_value(row, "activated_at")),
        last_meaningful_activity_at=_optional_timestamp(
            _row_value(row, "last_meaningful_activity_at")
        ),
        inactive_days=int(_row_value(row, "inactive_days", 0)),
        active_days_30d=int(_row_value(row, "active_days_30d", 0)),
        successful_backtests_30d=int(
            _row_value(row, "successful_backtests_30d", 0)
        ),
        calculated_at=_timestamp(_row_value(row, "calculated_at")),
    )


def _fetchall(conn: Any, postgres: bool, sql: str, params: Sequence[Any]):
    """Shared by both twins.

    Still dialect-parameterised: it serves ``list_commercial_values``/
    ``list_credit_activity``, which branch on the *credits* store's dialect,
    never on this class's own -- see the module docstring.
    """
    if postgres:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    return conn.execute(sql, params).fetchall()


def _user_clause(ids: list[int], postgres: bool) -> tuple[str, list[Any]]:
    """Shared by both twins; see ``_fetchall`` above."""
    if postgres:
        return "user_id = ANY(%s)", [ids]
    placeholders = ", ".join("?" for _ in ids)
    return f"user_id IN ({placeholders})", list(ids)


def _projection_job_name(value: object) -> str:
    """Shared by both twins; see ``_current_snapshot_from_row`` above."""
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 100
    ):
        raise ValueError("job_name must be a trimmed non-empty string")
    return value


class ValueAnalyticsStore:
    """SQLite value projection storage.

    See ``value_repository_postgres.py`` for the PostgreSQL twin;
    ``build_value_analytics_store()`` below picks between them.

    ``credits_base`` and the optional operational stores are injectable to keep
    contract tests synthetic and to avoid importing production singletons.
    """

    def __init__(
        self,
        analytics_base: Any | None = None,
        credits_base: Any | None = None,
        provider_base: Any | None = None,
        agent_base: Any | None = None,
        run_base: Any | None = None,
    ) -> None:
        self.analytics_base = analytics_base or analytics_store
        if credits_base is None:
            from dashboard.backend.domain.credits.repository import credits_store

            credits_base = credits_store
        self.credits_base = credits_base
        if provider_base is None:
            from dashboard.backend.domain.model_providers.repository import (
                model_provider_store,
            )

            provider_base = model_provider_store
        self.provider_base = provider_base
        if agent_base is None:
            from dashboard.backend.domain.agents.repository import agent_store

            agent_base = agent_store
        self.agent_base = agent_base
        if run_base is None:
            from dashboard.backend.domain.runs.repository import run_store

            run_base = run_store
        self.run_base = run_base

    def _analytics_connection(self):
        return self.analytics_base._get_connection()

    def upsert_current_snapshot(self, snapshot: UserValueSnapshot) -> UserValueSnapshot:
        def evidence(value: Sequence[str]) -> str:
            return json.dumps(
                list(value),
                separators=(",", ":"),
                ensure_ascii=True,
            )

        legacy_status, legacy_reason_code, legacy_reason = _legacy_seed(snapshot)
        values = (
            snapshot.user_id,
            legacy_status,
            legacy_reason_code,
            legacy_reason,
            "[]",
            snapshot.lifecycle_segment,
            snapshot.lifecycle_reason_code,
            snapshot.lifecycle_reason,
            evidence(snapshot.lifecycle_evidence),
            snapshot.operational_state,
            snapshot.operational_reason_code,
            snapshot.operational_reason,
            evidence(snapshot.operational_evidence),
            utc_iso(snapshot.activated_at) if snapshot.activated_at else None,
            (
                utc_iso(snapshot.last_meaningful_activity_at)
                if snapshot.last_meaningful_activity_at
                else None
            ),
            snapshot.inactive_days,
            snapshot.active_days_30d,
            snapshot.successful_backtests_30d,
            utc_iso(snapshot.calculated_at),
        )
        columns = """
            user_id, status, reason_code, human_readable_reason,
            evidence_event_ids_json, lifecycle_segment, lifecycle_reason_code,
            lifecycle_reason, lifecycle_evidence_json, operational_state,
            operational_reason_code, operational_reason,
            operational_evidence_json, activated_at,
            last_meaningful_activity_at, inactive_days, active_days_30d,
            successful_backtests_30d, calculated_at
        """
        updates = """
            lifecycle_segment=excluded.lifecycle_segment,
            lifecycle_reason_code=excluded.lifecycle_reason_code,
            lifecycle_reason=excluded.lifecycle_reason,
            lifecycle_evidence_json=excluded.lifecycle_evidence_json,
            operational_state=excluded.operational_state,
            operational_reason_code=excluded.operational_reason_code,
            operational_reason=excluded.operational_reason,
            operational_evidence_json=excluded.operational_evidence_json,
            activated_at=excluded.activated_at,
            last_meaningful_activity_at=excluded.last_meaningful_activity_at,
            inactive_days=excluded.inactive_days,
            active_days_30d=excluded.active_days_30d,
            successful_backtests_30d=excluded.successful_backtests_30d,
            calculated_at=excluded.calculated_at
        """
        with self._analytics_connection() as conn:
            conn.execute(
                f"""
                INSERT INTO user_analytics_snapshots ({columns})
                VALUES ({", ".join(["?"] * len(values))})
                ON CONFLICT(user_id) DO UPDATE SET {updates}
                """,
                values,
            )
        return snapshot

    def get_current_snapshot(self, user_id: int) -> UserValueSnapshot | None:
        subject = positive_user_id(user_id)
        with self._analytics_connection() as conn:
            row = conn.execute(
                "SELECT * FROM user_analytics_snapshots WHERE user_id=?", (subject,)
            ).fetchone()
        return _current_snapshot_from_row(row)

    def list_current_snapshots(
        self,
        user_ids: Sequence[int],
    ) -> dict[int, UserValueSnapshot]:
        ids = _ids(user_ids)
        if not ids:
            return {}
        result: dict[int, UserValueSnapshot] = {}
        for offset in range(0, len(ids), MAX_USER_BATCH):
            chunk = ids[offset : offset + MAX_USER_BATCH]
            clause = f"user_id IN ({', '.join('?' for _ in chunk)})"
            with self._analytics_connection() as conn:
                rows = conn.execute(
                    f"""
                    SELECT *
                    FROM user_analytics_snapshots
                    WHERE {clause} AND lifecycle_segment IS NOT NULL
                    ORDER BY user_id
                    """,
                    chunk,
                ).fetchall()
            for row in rows:
                user_id = int(_row_value(row, "user_id"))
                snapshot = _current_snapshot_from_row(row)
                if snapshot is not None:
                    result[user_id] = snapshot
        return result

    def upsert_daily_snapshot(
        self,
        snapshot: UserLifecycleDailySnapshot,
    ) -> UserLifecycleDailySnapshot:
        values = (
            snapshot.snapshot_date.isoformat(),
            snapshot.user_id,
            snapshot.lifecycle_segment,
            snapshot.lifecycle_reason_code,
            snapshot.data_quality,
            utc_iso(snapshot.calculated_at),
        )
        sql = """
            INSERT INTO user_lifecycle_daily_snapshots (
                snapshot_date, user_id, lifecycle_segment,
                lifecycle_reason_code, data_quality, calculated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(snapshot_date, user_id) DO UPDATE SET
                lifecycle_segment=excluded.lifecycle_segment,
                lifecycle_reason_code=excluded.lifecycle_reason_code,
                data_quality=excluded.data_quality,
                calculated_at=excluded.calculated_at
        """
        with self._analytics_connection() as conn:
            conn.execute(sql, values)
        return snapshot

    def list_daily_snapshots(
        self,
        *,
        start: date,
        end: date,
        user_ids: Sequence[int] | None = None,
    ) -> list[UserLifecycleDailySnapshot]:
        if end <= start:
            raise ValueError("end must be later than start")
        ids = _ids(user_ids) if user_ids is not None else None
        if ids == []:
            return []
        params: list[Any] = [start.isoformat(), end.isoformat()]
        clause = ""
        if ids:
            clause = f" AND user_id IN ({','.join('?' for _ in ids)})"
            params.extend(ids)
        sql = f"""
            SELECT *
            FROM user_lifecycle_daily_snapshots
            WHERE snapshot_date >= ?
              AND snapshot_date < ?
              {clause}
            ORDER BY snapshot_date, user_id
        """
        with self._analytics_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            UserLifecycleDailySnapshot(
                snapshot_date=date.fromisoformat(str(_row_value(row, "snapshot_date"))),
                user_id=int(_row_value(row, "user_id")),
                lifecycle_segment=_row_value(row, "lifecycle_segment"),
                lifecycle_reason_code=_row_value(row, "lifecycle_reason_code"),
                data_quality=_row_value(row, "data_quality"),
                calculated_at=_timestamp(_row_value(row, "calculated_at")),
            )
            for row in rows
        ]

    def replace_lifecycle_rollups(
        self,
        day: date,
        rows: Sequence[Any],
        *,
        replace_transitions: bool = True,
    ) -> None:
        """Replace only lifecycle aggregates, preserving other daily metrics."""

        values = list(rows)
        columns = (
            "rollup_date",
            "metric_name",
            "event_name",
            "billing_mode",
            "provider_id",
            "model_id",
            "outcome",
            "error_category",
            "user_state",
            "value_count",
            "value_sum_micro",
            "updated_at",
        )
        payloads = []
        for row in values:
            if (
                row.rollup_date != day
                or row.metric_name not in LIFECYCLE_ROLLUP_METRICS
            ):
                raise ValueError("invalid lifecycle rollup row")
            if row.metric_name == "lifecycle_segment_count":
                valid_dimensions = (
                    not row.event_name and row.user_state in LIFECYCLE_SEGMENTS
                )
            else:
                valid_dimensions = (
                    replace_transitions
                    and row.event_name in LIFECYCLE_SEGMENTS
                    and row.user_state in LIFECYCLE_SEGMENTS
                    and row.event_name != row.user_state
                )
            unused_dimensions = (
                row.billing_mode,
                row.provider_id,
                row.model_id,
                row.error_category,
            )
            if (
                not valid_dimensions
                or any(unused_dimensions)
                or row.outcome not in {"complete", "partial"}
                or row.value_sum_micro != 0
            ):
                raise ValueError("invalid lifecycle rollup dimensions")
            payloads.append(
                (
                    row.rollup_date.isoformat(),
                    row.metric_name,
                    row.event_name,
                    row.billing_mode,
                    row.provider_id,
                    row.model_id,
                    row.outcome,
                    row.error_category,
                    row.user_state,
                    row.value_count,
                    row.value_sum_micro,
                    utc_iso(row.updated_at),
                )
            )
        placeholders = ", ".join(["?"] * len(columns))
        metrics = ["lifecycle_segment_count"]
        if replace_transitions:
            metrics.append("lifecycle_transition")
        metric_placeholders = ", ".join(["?"] * len(metrics))
        with self._analytics_connection() as conn:
            conn.execute(
                f"""
                DELETE FROM analytics_daily_rollups
                WHERE rollup_date = ?
                  AND metric_name IN ({metric_placeholders})
                """,
                (day.isoformat(), *metrics),
            )
            if payloads:
                conn.executemany(
                    f"""
                    INSERT INTO analytics_daily_rollups ({', '.join(columns)})
                    VALUES ({placeholders})
                    """,
                    payloads,
                )

    def list_expiring_daily_dates(
        self,
        *,
        before: date,
        limit: int,
    ) -> list[date]:
        if not isinstance(before, date) or isinstance(before, datetime):
            raise ValueError("before must be a date")
        page_size = positive_limit(limit, maximum=1000)
        sql = """
            SELECT DISTINCT snapshot_date
            FROM user_lifecycle_daily_snapshots
            WHERE snapshot_date < ?
            ORDER BY snapshot_date
            LIMIT ?
        """
        with self._analytics_connection() as conn:
            rows = conn.execute(sql, (before.isoformat(), page_size)).fetchall()
        return [
            date.fromisoformat(str(_row_value(row, "snapshot_date"))) for row in rows
        ]

    def delete_daily_snapshots_for_date(self, day: date) -> int:
        if not isinstance(day, date) or isinstance(day, datetime):
            raise ValueError("day must be a date")
        with self._analytics_connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM user_lifecycle_daily_snapshots
                WHERE snapshot_date = ?
                """,
                (day.isoformat(),),
            )
            return max(0, int(cursor.rowcount))

    def has_daily_before(self, before: date) -> bool:
        if not isinstance(before, date) or isinstance(before, datetime):
            raise ValueError("before must be a date")
        sql = """
            SELECT 1
            FROM user_lifecycle_daily_snapshots
            WHERE snapshot_date < ?
            LIMIT 1
        """
        with self._analytics_connection() as conn:
            row = conn.execute(sql, (before.isoformat(),)).fetchone()
        return row is not None

    def list_commercial_values(
        self,
        user_ids: Sequence[int],
        *,
        start: datetime,
        end: datetime,
    ) -> dict[int, CommercialValueFact]:
        ids = _ids(user_ids)
        window_start, window_end = _validate_window(start, end)
        if not ids:
            return {}

        lifetime_by_user: dict[int, tuple[int, int]] = {}
        period_by_user: dict[int, tuple[int, int, int]] = {}
        usage_by_user: dict[int, int] = {}
        if hasattr(self.credits_base, "_get_connection"):
            postgres = hasattr(self.credits_base, "database_url")
            user_clause, user_params = _user_clause(ids, postgres)
            placeholder = "%s" if postgres else "?"
            window_params = [
                *user_params,
                utc_iso(window_start),
                utc_iso(window_end),
            ]
            with self.credits_base._get_connection() as conn:
                lifetime_rows = _fetchall(
                    conn,
                    postgres,
                    f"""
                    SELECT user_id,
                           COALESCE(SUM(CASE WHEN entry_type = 'purchase'
                               THEN amount_micro ELSE 0 END), 0) AS purchased_micro,
                           COALESCE(SUM(CASE WHEN entry_type = 'refund'
                               THEN -amount_micro ELSE 0 END), 0) AS refunded_micro
                    FROM credit_ledger_entries
                    WHERE {user_clause}
                      AND entry_type IN ('purchase', 'refund')
                    GROUP BY user_id
                    """,
                    user_params,
                )
                period_rows = _fetchall(
                    conn,
                    postgres,
                    f"""
                    SELECT user_id,
                           COALESCE(SUM(CASE WHEN entry_type = 'purchase'
                               THEN amount_micro ELSE 0 END), 0) AS purchased_micro,
                           COALESCE(SUM(CASE WHEN entry_type = 'refund'
                               THEN -amount_micro ELSE 0 END), 0) AS refunded_micro,
                           COALESCE(SUM(CASE
                               WHEN entry_type = 'admin_grant_assign' THEN amount_micro
                               WHEN entry_type = 'admin_grant_reclaim' THEN -amount_micro
                               ELSE 0 END), 0) AS grant_activity_micro
                    FROM credit_ledger_entries
                    WHERE {user_clause}
                      AND created_at >= {placeholder}
                      AND created_at < {placeholder}
                    GROUP BY user_id
                    """,
                    window_params,
                )
                usage_rows = _fetchall(
                    conn,
                    postgres,
                    f"""
                    SELECT user_id,
                           COALESCE(SUM(-amount_micro), 0) AS consumed_micro
                    FROM credit_llm_usage_entries
                    WHERE {user_clause}
                      AND created_at >= {placeholder}
                      AND created_at < {placeholder}
                    GROUP BY user_id
                    """,
                    window_params,
                )
            lifetime_by_user = {
                int(_row_value(row, "user_id")): (
                    max(int(_row_value(row, "purchased_micro", 0)), 0),
                    max(int(_row_value(row, "refunded_micro", 0)), 0),
                )
                for row in lifetime_rows
            }
            period_by_user = {
                int(_row_value(row, "user_id")): (
                    max(int(_row_value(row, "purchased_micro", 0)), 0),
                    max(int(_row_value(row, "refunded_micro", 0)), 0),
                    max(int(_row_value(row, "grant_activity_micro", 0)), 0),
                )
                for row in period_rows
            }
            usage_by_user = {
                int(_row_value(row, "user_id")): max(
                    int(_row_value(row, "consumed_micro", 0)), 0
                )
                for row in usage_rows
            }

        balances = (
            self.credits_base.get_balance_projections(ids)
            if hasattr(self.credits_base, "get_balance_projections")
            else {}
        )
        result: dict[int, CommercialValueFact] = {}
        for user_id in ids:
            lifetime_purchased, lifetime_refunded = lifetime_by_user.get(
                user_id, (0, 0)
            )
            purchased, refunded, grant_activity = period_by_user.get(user_id, (0, 0, 0))
            net_purchased = max(lifetime_purchased - lifetime_refunded, 0)
            balance = balances.get(user_id, {})
            result[user_id] = CommercialValueFact(
                user_id=user_id,
                lifetime_net_purchased_micro=net_purchased,
                commercial_tier=commercial_tier(net_purchased),
                purchased_micro=purchased,
                refunded_micro=refunded,
                consumed_micro=usage_by_user.get(user_id, 0),
                admin_grant_activity_micro=grant_activity,
                grant_available_micro=max(
                    int(_object_value(balance, "grant_available_micro")), 0
                ),
                purchased_available_micro=max(
                    int(_object_value(balance, "purchased_available_micro")), 0
                ),
                total_available_micro=max(
                    int(_object_value(balance, "total_available_micro")), 0
                ),
            )
        return result

    def list_credit_activity(
        self,
        user_ids: Sequence[int],
        *,
        start: datetime,
        end: datetime,
    ) -> dict[int, Sequence[datetime]]:
        ids = _ids(user_ids)
        window_start, window_end = _validate_window(start, end)
        result: dict[int, list[datetime]] = {user_id: [] for user_id in ids}
        if not ids or not hasattr(self.credits_base, "_get_connection"):
            return {user_id: () for user_id in ids}

        postgres = hasattr(self.credits_base, "database_url")
        user_clause, user_params = _user_clause(ids, postgres)
        placeholder = "%s" if postgres else "?"
        params = [*user_params, utc_iso(window_start), utc_iso(window_end)]
        with self.credits_base._get_connection() as conn:
            purchase_rows = _fetchall(
                conn,
                postgres,
                f"""
                SELECT user_id, created_at
                FROM credit_ledger_entries
                WHERE {user_clause}
                  AND entry_type = 'purchase'
                  AND created_at >= {placeholder}
                  AND created_at < {placeholder}
                """,
                params,
            )
            usage_rows = _fetchall(
                conn,
                postgres,
                f"""
                SELECT user_id, created_at
                FROM credit_llm_usage_entries
                WHERE {user_clause}
                  AND created_at >= {placeholder}
                  AND created_at < {placeholder}
                """,
                params,
            )
        for row in (*purchase_rows, *usage_rows):
            user_id = int(_row_value(row, "user_id"))
            if user_id in result:
                result[user_id].append(_timestamp(_row_value(row, "created_at")))
        return {
            user_id: tuple(sorted(timestamps)) for user_id, timestamps in result.items()
        }

    def _run_health(self, user_id: int, now: datetime) -> tuple[int, bool]:
        if self.agent_base is None or self.run_base is None:
            return 0, False
        agents = self.agent_base.list_agents(owner_user_id=user_id)
        agent_ids = [str(row.get("agent_id") or "") for row in agents]
        runs = [
            run
            for agent_id in agent_ids
            if agent_id
            for run in self.run_base.list_runs(agent_id)
        ]
        ordered = sorted(
            runs,
            key=lambda run: (
                _optional_timestamp(run.get("updated_at"))
                or _optional_timestamp(run.get("created_at"))
                or datetime.min.replace(tzinfo=timezone.utc)
            ),
            reverse=True,
        )
        terminal_24h = [
            run
            for run in ordered
            if str(run.get("status")) in _TERMINAL_RUN_STATUSES
            and (
                timestamp := (
                    _optional_timestamp(run.get("updated_at"))
                    or _optional_timestamp(run.get("created_at"))
                )
            )
            and now - timedelta(hours=24) <= timestamp <= now
        ]
        consecutive_failures = 0
        for run in terminal_24h:
            if str(run.get("status")) not in {"failed", "timed_out"}:
                break
            consecutive_failures += 1

        beyond_deadline = False
        for run in ordered:
            if str(run.get("status")) not in _ACTIVE_RUN_STATUSES:
                continue
            explicit_deadline = _optional_timestamp(run.get("deadline_at"))
            created_at = _optional_timestamp(run.get("created_at"))
            if (explicit_deadline is not None and explicit_deadline < now) or (
                explicit_deadline is None
                and created_at is not None
                and created_at + RUN_SAFE_DEADLINE < now
            ):
                beyond_deadline = True
                break
        return consecutive_failures, beyond_deadline

    def get_operational_facts(
        self,
        user_id: int,
        *,
        now: datetime,
    ) -> CurrentOperationalFacts:
        user_id = positive_user_id(user_id)
        current = _utc(now, "now")
        billing = (
            self.credits_base.get_account_billing_state(user_id)
            if hasattr(self.credits_base, "get_account_billing_state")
            else {}
        )
        balances = (
            self.credits_base.get_balance_projections([user_id])
            if hasattr(self.credits_base, "get_balance_projections")
            else {}
        )
        total_available = int(
            _object_value(balances.get(user_id, {}), "total_available_micro")
        )

        credential_status: Literal[
            "verified", "invalid", "verification_unavailable", "missing"
        ] = "missing"
        selected_provider_enabled = True
        verified_byok_lane = False
        platform_lane = total_available > 0
        if self.provider_base is not None:
            credentials = self.provider_base.list_user_credentials(user_id)
            providers = {
                str(row.get("provider_id")): row
                for row in self.provider_base.list_all_providers()
            }
            defaults = [row for row in credentials if row.get("is_default")]
            default_statuses = {str(row.get("status")) for row in defaults}
            if "invalid" in default_statuses:
                credential_status = "invalid"
            elif "verification_unavailable" in default_statuses:
                credential_status = "verification_unavailable"
            elif "verified" in default_statuses:
                credential_status = "verified"

            def provider_supports(row: Mapping[str, Any], mode: str) -> bool:
                provider = providers.get(str(row.get("provider_id")), {})
                return provider.get("status") == "enabled" and bool(
                    provider.get(f"{mode}_enabled")
                )

            selected_provider_enabled = all(
                providers.get(str(row.get("provider_id")), {}).get("status")
                == "enabled"
                for row in defaults
            )
            verified_byok_lane = any(
                row.get("status") == "verified"
                and row.get("is_default")
                and provider_supports(row, "byok")
                for row in credentials
            )
            platform_lane = total_available > 0 and any(
                row.get("status") == "enabled" and row.get("platform_enabled")
                for row in providers.values()
            )
            if hasattr(self.provider_base, "get_platform_credential_public"):
                from dashboard.backend.domain.model_providers.service import (
                    ModelProviderService,
                )

                options = ModelProviderService(
                    store=self.provider_base
                ).list_execution_options(user_id)
                verified_byok_lane = any(option.byok_available for option in options)
                platform_lane = total_available > 0 and any(
                    option.platform_credits_available for option in options
                )

        failures, beyond_deadline = self._run_health(user_id, current)
        return CurrentOperationalFacts(
            user_id=user_id,
            account_restricted=billing.get("account_status") == "restricted",
            usable_billing_lane=platform_lane or verified_byok_lane,
            selected_provider_enabled=selected_provider_enabled,
            default_credential_status=credential_status,
            failed_terminal_runs_24h=failures,
            run_beyond_safe_deadline=beyond_deadline,
        )

    def get_projection_job(self, job_name: str) -> ProjectionJob | None:
        name = _projection_job_name(job_name)
        with self._analytics_connection() as conn:
            row = conn.execute(
                "SELECT * FROM analytics_projection_jobs WHERE job_name=?",
                (name,),
            ).fetchone()
        if row is None:
            return None
        return ProjectionJob(
            job_name=_row_value(row, "job_name"),
            window_start=date.fromisoformat(str(_row_value(row, "window_start"))),
            window_end=date.fromisoformat(str(_row_value(row, "window_end"))),
            cursor=_row_value(row, "cursor"),
            status=_row_value(row, "status"),
            updated_at=_timestamp(_row_value(row, "updated_at")),
        )

    def save_projection_job(self, job: ProjectionJob) -> ProjectionJob:
        _projection_job_name(job.job_name)
        values = (
            job.job_name,
            job.window_start.isoformat(),
            job.window_end.isoformat(),
            job.cursor,
            job.status,
            utc_iso(job.updated_at),
        )
        sql = """
            INSERT INTO analytics_projection_jobs (
                job_name, window_start, window_end, cursor, status, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_name) DO UPDATE SET
                window_start=excluded.window_start,
                window_end=excluded.window_end,
                cursor=excluded.cursor,
                status=excluded.status,
                updated_at=excluded.updated_at
        """
        with self._analytics_connection() as conn:
            conn.execute(sql, values)
        return job


def build_value_analytics_store(
    analytics_base: Any | None = None,
    credits_base: Any | None = None,
    provider_base: Any | None = None,
    agent_base: Any | None = None,
    run_base: Any | None = None,
):
    """Pick the SQLite or PostgreSQL value-analytics twin.

    Mirrors ``repository.py``'s ``_build_analytics_store()``: the decision is
    made from the resolved ``analytics_base`` alone -- the same object every
    caller already passes, or, if none, the same ``analytics_store`` singleton
    that decision is based on. ``credits_base``/``provider_base``/
    ``agent_base``/``run_base`` are forwarded unexamined; their own dialect,
    if any, is handled inside ``list_commercial_values``/``list_credit_activity``
    on either twin, independently of this choice.
    """
    resolved_analytics_base = analytics_base or analytics_store
    if hasattr(resolved_analytics_base, "database_url"):
        from .value_repository_postgres import PostgresValueAnalyticsStore

        return PostgresValueAnalyticsStore(
            resolved_analytics_base,
            credits_base=credits_base,
            provider_base=provider_base,
            agent_base=agent_base,
            run_base=run_base,
        )
    return ValueAnalyticsStore(
        resolved_analytics_base,
        credits_base=credits_base,
        provider_base=provider_base,
        agent_base=agent_base,
        run_base=run_base,
    )


__all__ = [
    "CommercialValueFact",
    "CurrentOperationalFacts",
    "ProjectionJob",
    "UserLifecycleDailySnapshot",
    "UserValueSnapshot",
    "ValueAnalyticsStore",
    "build_value_analytics_store",
]
```

- [ ] **Step 4: Create `value_repository_postgres.py` in full.** Every method that touched `user_analytics_snapshots`, `user_lifecycle_daily_snapshots` or `analytics_projection_jobs` keeps only its `is_postgres=True` arm (`%s` placeholders, `conn.cursor()`); `list_commercial_values`, `list_credit_activity`, `get_operational_facts` and `_run_health` are duplicated verbatim from `value_repository.py` because none of the four contains a branch on this class's own dialect:

```python
"""PostgreSQL twin of the user-value Analytics projection store.

See ``value_repository.py`` for the SQLite twin and its module docstring,
which explains which methods branch on this class's own dialect (all
rewritten below, one path each) versus a *different* store's dialect
(``list_commercial_values``, ``list_credit_activity`` -- unchanged here,
and identical to the SQLite twin, because that branch was never about
which twin this class is).
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal, Mapping, Sequence

from .repository_common import positive_limit, positive_user_id, utc_iso
from .value_repository import (
    _ACTIVE_RUN_STATUSES,
    _TERMINAL_RUN_STATUSES,
    LIFECYCLE_ROLLUP_METRICS,
    LIFECYCLE_SEGMENTS,
    MAX_USER_BATCH,
    RUN_SAFE_DEADLINE,
    CommercialValueFact,
    CurrentOperationalFacts,
    ProjectionJob,
    UserLifecycleDailySnapshot,
    UserValueSnapshot,
    _current_snapshot_from_row,
    _fetchall,
    _ids,
    _legacy_seed,
    _object_value,
    _optional_timestamp,
    _projection_job_name,
    _row_value,
    _timestamp,
    _user_clause,
    _utc,
    _validate_window,
    analytics_store,
    commercial_tier,
)


class PostgresValueAnalyticsStore:
    """PostgreSQL value projection storage.

    See ``value_repository.py`` for the SQLite twin;
    ``build_value_analytics_store()`` there picks between them.
    """

    def __init__(
        self,
        analytics_base: Any | None = None,
        credits_base: Any | None = None,
        provider_base: Any | None = None,
        agent_base: Any | None = None,
        run_base: Any | None = None,
    ) -> None:
        self.analytics_base = analytics_base or analytics_store
        if credits_base is None:
            from dashboard.backend.domain.credits.repository import credits_store

            credits_base = credits_store
        self.credits_base = credits_base
        if provider_base is None:
            from dashboard.backend.domain.model_providers.repository import (
                model_provider_store,
            )

            provider_base = model_provider_store
        self.provider_base = provider_base
        if agent_base is None:
            from dashboard.backend.domain.agents.repository import agent_store

            agent_base = agent_store
        self.agent_base = agent_base
        if run_base is None:
            from dashboard.backend.domain.runs.repository import run_store

            run_base = run_store
        self.run_base = run_base

    def _analytics_connection(self):
        return self.analytics_base._get_connection()

    def upsert_current_snapshot(self, snapshot: UserValueSnapshot) -> UserValueSnapshot:
        def evidence(value: Sequence[str]) -> str:
            return json.dumps(
                list(value),
                separators=(",", ":"),
                ensure_ascii=True,
            )

        legacy_status, legacy_reason_code, legacy_reason = _legacy_seed(snapshot)
        values = (
            snapshot.user_id,
            legacy_status,
            legacy_reason_code,
            legacy_reason,
            "[]",
            snapshot.lifecycle_segment,
            snapshot.lifecycle_reason_code,
            snapshot.lifecycle_reason,
            evidence(snapshot.lifecycle_evidence),
            snapshot.operational_state,
            snapshot.operational_reason_code,
            snapshot.operational_reason,
            evidence(snapshot.operational_evidence),
            utc_iso(snapshot.activated_at) if snapshot.activated_at else None,
            (
                utc_iso(snapshot.last_meaningful_activity_at)
                if snapshot.last_meaningful_activity_at
                else None
            ),
            snapshot.inactive_days,
            snapshot.active_days_30d,
            snapshot.successful_backtests_30d,
            utc_iso(snapshot.calculated_at),
        )
        columns = """
            user_id, status, reason_code, human_readable_reason,
            evidence_event_ids_json, lifecycle_segment, lifecycle_reason_code,
            lifecycle_reason, lifecycle_evidence_json, operational_state,
            operational_reason_code, operational_reason,
            operational_evidence_json, activated_at,
            last_meaningful_activity_at, inactive_days, active_days_30d,
            successful_backtests_30d, calculated_at
        """
        updates = """
            lifecycle_segment=excluded.lifecycle_segment,
            lifecycle_reason_code=excluded.lifecycle_reason_code,
            lifecycle_reason=excluded.lifecycle_reason,
            lifecycle_evidence_json=excluded.lifecycle_evidence_json,
            operational_state=excluded.operational_state,
            operational_reason_code=excluded.operational_reason_code,
            operational_reason=excluded.operational_reason,
            operational_evidence_json=excluded.operational_evidence_json,
            activated_at=excluded.activated_at,
            last_meaningful_activity_at=excluded.last_meaningful_activity_at,
            inactive_days=excluded.inactive_days,
            active_days_30d=excluded.active_days_30d,
            successful_backtests_30d=excluded.successful_backtests_30d,
            calculated_at=excluded.calculated_at
        """
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO user_analytics_snapshots ({columns})
                    VALUES ({", ".join(["%s"] * len(values))})
                    ON CONFLICT(user_id) DO UPDATE SET {updates}
                    """,
                    values,
                )
        return snapshot

    def get_current_snapshot(self, user_id: int) -> UserValueSnapshot | None:
        subject = positive_user_id(user_id)
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM user_analytics_snapshots WHERE user_id=%s",
                    (subject,),
                )
                row = cur.fetchone()
        return _current_snapshot_from_row(row)

    def list_current_snapshots(
        self,
        user_ids: Sequence[int],
    ) -> dict[int, UserValueSnapshot]:
        ids = _ids(user_ids)
        if not ids:
            return {}
        result: dict[int, UserValueSnapshot] = {}
        for offset in range(0, len(ids), MAX_USER_BATCH):
            chunk = ids[offset : offset + MAX_USER_BATCH]
            with self._analytics_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT *
                        FROM user_analytics_snapshots
                        WHERE user_id = ANY(%s) AND lifecycle_segment IS NOT NULL
                        ORDER BY user_id
                        """,
                        [chunk],
                    )
                    rows = cur.fetchall()
            for row in rows:
                user_id = int(_row_value(row, "user_id"))
                snapshot = _current_snapshot_from_row(row)
                if snapshot is not None:
                    result[user_id] = snapshot
        return result

    def upsert_daily_snapshot(
        self,
        snapshot: UserLifecycleDailySnapshot,
    ) -> UserLifecycleDailySnapshot:
        values = (
            snapshot.snapshot_date.isoformat(),
            snapshot.user_id,
            snapshot.lifecycle_segment,
            snapshot.lifecycle_reason_code,
            snapshot.data_quality,
            utc_iso(snapshot.calculated_at),
        )
        sql = """
            INSERT INTO user_lifecycle_daily_snapshots (
                snapshot_date, user_id, lifecycle_segment,
                lifecycle_reason_code, data_quality, calculated_at
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT(snapshot_date, user_id) DO UPDATE SET
                lifecycle_segment=excluded.lifecycle_segment,
                lifecycle_reason_code=excluded.lifecycle_reason_code,
                data_quality=excluded.data_quality,
                calculated_at=excluded.calculated_at
        """
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, values)
        return snapshot

    def list_daily_snapshots(
        self,
        *,
        start: date,
        end: date,
        user_ids: Sequence[int] | None = None,
    ) -> list[UserLifecycleDailySnapshot]:
        if end <= start:
            raise ValueError("end must be later than start")
        ids = _ids(user_ids) if user_ids is not None else None
        if ids == []:
            return []
        params: list[Any] = [start.isoformat(), end.isoformat()]
        clause = ""
        if ids:
            clause = " AND user_id = ANY(%s)"
            params.append(ids)
        sql = f"""
            SELECT *
            FROM user_lifecycle_daily_snapshots
            WHERE snapshot_date >= %s
              AND snapshot_date < %s
              {clause}
            ORDER BY snapshot_date, user_id
        """
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [
            UserLifecycleDailySnapshot(
                snapshot_date=date.fromisoformat(str(_row_value(row, "snapshot_date"))),
                user_id=int(_row_value(row, "user_id")),
                lifecycle_segment=_row_value(row, "lifecycle_segment"),
                lifecycle_reason_code=_row_value(row, "lifecycle_reason_code"),
                data_quality=_row_value(row, "data_quality"),
                calculated_at=_timestamp(_row_value(row, "calculated_at")),
            )
            for row in rows
        ]

    def replace_lifecycle_rollups(
        self,
        day: date,
        rows: Sequence[Any],
        *,
        replace_transitions: bool = True,
    ) -> None:
        """Replace only lifecycle aggregates, preserving other daily metrics."""

        values = list(rows)
        columns = (
            "rollup_date",
            "metric_name",
            "event_name",
            "billing_mode",
            "provider_id",
            "model_id",
            "outcome",
            "error_category",
            "user_state",
            "value_count",
            "value_sum_micro",
            "updated_at",
        )
        payloads = []
        for row in values:
            if (
                row.rollup_date != day
                or row.metric_name not in LIFECYCLE_ROLLUP_METRICS
            ):
                raise ValueError("invalid lifecycle rollup row")
            if row.metric_name == "lifecycle_segment_count":
                valid_dimensions = (
                    not row.event_name and row.user_state in LIFECYCLE_SEGMENTS
                )
            else:
                valid_dimensions = (
                    replace_transitions
                    and row.event_name in LIFECYCLE_SEGMENTS
                    and row.user_state in LIFECYCLE_SEGMENTS
                    and row.event_name != row.user_state
                )
            unused_dimensions = (
                row.billing_mode,
                row.provider_id,
                row.model_id,
                row.error_category,
            )
            if (
                not valid_dimensions
                or any(unused_dimensions)
                or row.outcome not in {"complete", "partial"}
                or row.value_sum_micro != 0
            ):
                raise ValueError("invalid lifecycle rollup dimensions")
            payloads.append(
                (
                    row.rollup_date.isoformat(),
                    row.metric_name,
                    row.event_name,
                    row.billing_mode,
                    row.provider_id,
                    row.model_id,
                    row.outcome,
                    row.error_category,
                    row.user_state,
                    row.value_count,
                    row.value_sum_micro,
                    utc_iso(row.updated_at),
                )
            )
        placeholders = ", ".join(["%s"] * len(columns))
        metrics = ["lifecycle_segment_count"]
        if replace_transitions:
            metrics.append("lifecycle_transition")
        metric_placeholders = ", ".join(["%s"] * len(metrics))
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    DELETE FROM analytics_daily_rollups
                    WHERE rollup_date = %s
                      AND metric_name IN ({metric_placeholders})
                    """,
                    (day.isoformat(), *metrics),
                )
                if payloads:
                    cur.executemany(
                        f"""
                        INSERT INTO analytics_daily_rollups ({', '.join(columns)})
                        VALUES ({placeholders})
                        """,
                        payloads,
                    )

    def list_expiring_daily_dates(
        self,
        *,
        before: date,
        limit: int,
    ) -> list[date]:
        if not isinstance(before, date) or isinstance(before, datetime):
            raise ValueError("before must be a date")
        page_size = positive_limit(limit, maximum=1000)
        sql = """
            SELECT DISTINCT snapshot_date
            FROM user_lifecycle_daily_snapshots
            WHERE snapshot_date < %s
            ORDER BY snapshot_date
            LIMIT %s
        """
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (before.isoformat(), page_size))
                rows = cur.fetchall()
        return [
            date.fromisoformat(str(_row_value(row, "snapshot_date"))) for row in rows
        ]

    def delete_daily_snapshots_for_date(self, day: date) -> int:
        if not isinstance(day, date) or isinstance(day, datetime):
            raise ValueError("day must be a date")
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM user_lifecycle_daily_snapshots
                    WHERE snapshot_date = %s
                    RETURNING user_id
                    """,
                    (day.isoformat(),),
                )
                return len(cur.fetchall())

    def has_daily_before(self, before: date) -> bool:
        if not isinstance(before, date) or isinstance(before, datetime):
            raise ValueError("before must be a date")
        sql = """
            SELECT 1
            FROM user_lifecycle_daily_snapshots
            WHERE snapshot_date < %s
            LIMIT 1
        """
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (before.isoformat(),))
                row = cur.fetchone()
        return row is not None

    def list_commercial_values(
        self,
        user_ids: Sequence[int],
        *,
        start: datetime,
        end: datetime,
    ) -> dict[int, CommercialValueFact]:
        ids = _ids(user_ids)
        window_start, window_end = _validate_window(start, end)
        if not ids:
            return {}

        lifetime_by_user: dict[int, tuple[int, int]] = {}
        period_by_user: dict[int, tuple[int, int, int]] = {}
        usage_by_user: dict[int, int] = {}
        if hasattr(self.credits_base, "_get_connection"):
            postgres = hasattr(self.credits_base, "database_url")
            user_clause, user_params = _user_clause(ids, postgres)
            placeholder = "%s" if postgres else "?"
            window_params = [
                *user_params,
                utc_iso(window_start),
                utc_iso(window_end),
            ]
            with self.credits_base._get_connection() as conn:
                lifetime_rows = _fetchall(
                    conn,
                    postgres,
                    f"""
                    SELECT user_id,
                           COALESCE(SUM(CASE WHEN entry_type = 'purchase'
                               THEN amount_micro ELSE 0 END), 0) AS purchased_micro,
                           COALESCE(SUM(CASE WHEN entry_type = 'refund'
                               THEN -amount_micro ELSE 0 END), 0) AS refunded_micro
                    FROM credit_ledger_entries
                    WHERE {user_clause}
                      AND entry_type IN ('purchase', 'refund')
                    GROUP BY user_id
                    """,
                    user_params,
                )
                period_rows = _fetchall(
                    conn,
                    postgres,
                    f"""
                    SELECT user_id,
                           COALESCE(SUM(CASE WHEN entry_type = 'purchase'
                               THEN amount_micro ELSE 0 END), 0) AS purchased_micro,
                           COALESCE(SUM(CASE WHEN entry_type = 'refund'
                               THEN -amount_micro ELSE 0 END), 0) AS refunded_micro,
                           COALESCE(SUM(CASE
                               WHEN entry_type = 'admin_grant_assign' THEN amount_micro
                               WHEN entry_type = 'admin_grant_reclaim' THEN -amount_micro
                               ELSE 0 END), 0) AS grant_activity_micro
                    FROM credit_ledger_entries
                    WHERE {user_clause}
                      AND created_at >= {placeholder}
                      AND created_at < {placeholder}
                    GROUP BY user_id
                    """,
                    window_params,
                )
                usage_rows = _fetchall(
                    conn,
                    postgres,
                    f"""
                    SELECT user_id,
                           COALESCE(SUM(-amount_micro), 0) AS consumed_micro
                    FROM credit_llm_usage_entries
                    WHERE {user_clause}
                      AND created_at >= {placeholder}
                      AND created_at < {placeholder}
                    GROUP BY user_id
                    """,
                    window_params,
                )
            lifetime_by_user = {
                int(_row_value(row, "user_id")): (
                    max(int(_row_value(row, "purchased_micro", 0)), 0),
                    max(int(_row_value(row, "refunded_micro", 0)), 0),
                )
                for row in lifetime_rows
            }
            period_by_user = {
                int(_row_value(row, "user_id")): (
                    max(int(_row_value(row, "purchased_micro", 0)), 0),
                    max(int(_row_value(row, "refunded_micro", 0)), 0),
                    max(int(_row_value(row, "grant_activity_micro", 0)), 0),
                )
                for row in period_rows
            }
            usage_by_user = {
                int(_row_value(row, "user_id")): max(
                    int(_row_value(row, "consumed_micro", 0)), 0
                )
                for row in usage_rows
            }

        balances = (
            self.credits_base.get_balance_projections(ids)
            if hasattr(self.credits_base, "get_balance_projections")
            else {}
        )
        result: dict[int, CommercialValueFact] = {}
        for user_id in ids:
            lifetime_purchased, lifetime_refunded = lifetime_by_user.get(
                user_id, (0, 0)
            )
            purchased, refunded, grant_activity = period_by_user.get(user_id, (0, 0, 0))
            net_purchased = max(lifetime_purchased - lifetime_refunded, 0)
            balance = balances.get(user_id, {})
            result[user_id] = CommercialValueFact(
                user_id=user_id,
                lifetime_net_purchased_micro=net_purchased,
                commercial_tier=commercial_tier(net_purchased),
                purchased_micro=purchased,
                refunded_micro=refunded,
                consumed_micro=usage_by_user.get(user_id, 0),
                admin_grant_activity_micro=grant_activity,
                grant_available_micro=max(
                    int(_object_value(balance, "grant_available_micro")), 0
                ),
                purchased_available_micro=max(
                    int(_object_value(balance, "purchased_available_micro")), 0
                ),
                total_available_micro=max(
                    int(_object_value(balance, "total_available_micro")), 0
                ),
            )
        return result

    def list_credit_activity(
        self,
        user_ids: Sequence[int],
        *,
        start: datetime,
        end: datetime,
    ) -> dict[int, Sequence[datetime]]:
        ids = _ids(user_ids)
        window_start, window_end = _validate_window(start, end)
        result: dict[int, list[datetime]] = {user_id: [] for user_id in ids}
        if not ids or not hasattr(self.credits_base, "_get_connection"):
            return {user_id: () for user_id in ids}

        postgres = hasattr(self.credits_base, "database_url")
        user_clause, user_params = _user_clause(ids, postgres)
        placeholder = "%s" if postgres else "?"
        params = [*user_params, utc_iso(window_start), utc_iso(window_end)]
        with self.credits_base._get_connection() as conn:
            purchase_rows = _fetchall(
                conn,
                postgres,
                f"""
                SELECT user_id, created_at
                FROM credit_ledger_entries
                WHERE {user_clause}
                  AND entry_type = 'purchase'
                  AND created_at >= {placeholder}
                  AND created_at < {placeholder}
                """,
                params,
            )
            usage_rows = _fetchall(
                conn,
                postgres,
                f"""
                SELECT user_id, created_at
                FROM credit_llm_usage_entries
                WHERE {user_clause}
                  AND created_at >= {placeholder}
                  AND created_at < {placeholder}
                """,
                params,
            )
        for row in (*purchase_rows, *usage_rows):
            user_id = int(_row_value(row, "user_id"))
            if user_id in result:
                result[user_id].append(_timestamp(_row_value(row, "created_at")))
        return {
            user_id: tuple(sorted(timestamps)) for user_id, timestamps in result.items()
        }

    def _run_health(self, user_id: int, now: datetime) -> tuple[int, bool]:
        if self.agent_base is None or self.run_base is None:
            return 0, False
        agents = self.agent_base.list_agents(owner_user_id=user_id)
        agent_ids = [str(row.get("agent_id") or "") for row in agents]
        runs = [
            run
            for agent_id in agent_ids
            if agent_id
            for run in self.run_base.list_runs(agent_id)
        ]
        ordered = sorted(
            runs,
            key=lambda run: (
                _optional_timestamp(run.get("updated_at"))
                or _optional_timestamp(run.get("created_at"))
                or datetime.min.replace(tzinfo=timezone.utc)
            ),
            reverse=True,
        )
        terminal_24h = [
            run
            for run in ordered
            if str(run.get("status")) in _TERMINAL_RUN_STATUSES
            and (
                timestamp := (
                    _optional_timestamp(run.get("updated_at"))
                    or _optional_timestamp(run.get("created_at"))
                )
            )
            and now - timedelta(hours=24) <= timestamp <= now
        ]
        consecutive_failures = 0
        for run in terminal_24h:
            if str(run.get("status")) not in {"failed", "timed_out"}:
                break
            consecutive_failures += 1

        beyond_deadline = False
        for run in ordered:
            if str(run.get("status")) not in _ACTIVE_RUN_STATUSES:
                continue
            explicit_deadline = _optional_timestamp(run.get("deadline_at"))
            created_at = _optional_timestamp(run.get("created_at"))
            if (explicit_deadline is not None and explicit_deadline < now) or (
                explicit_deadline is None
                and created_at is not None
                and created_at + RUN_SAFE_DEADLINE < now
            ):
                beyond_deadline = True
                break
        return consecutive_failures, beyond_deadline

    def get_operational_facts(
        self,
        user_id: int,
        *,
        now: datetime,
    ) -> CurrentOperationalFacts:
        user_id = positive_user_id(user_id)
        current = _utc(now, "now")
        billing = (
            self.credits_base.get_account_billing_state(user_id)
            if hasattr(self.credits_base, "get_account_billing_state")
            else {}
        )
        balances = (
            self.credits_base.get_balance_projections([user_id])
            if hasattr(self.credits_base, "get_balance_projections")
            else {}
        )
        total_available = int(
            _object_value(balances.get(user_id, {}), "total_available_micro")
        )

        credential_status: Literal[
            "verified", "invalid", "verification_unavailable", "missing"
        ] = "missing"
        selected_provider_enabled = True
        verified_byok_lane = False
        platform_lane = total_available > 0
        if self.provider_base is not None:
            credentials = self.provider_base.list_user_credentials(user_id)
            providers = {
                str(row.get("provider_id")): row
                for row in self.provider_base.list_all_providers()
            }
            defaults = [row for row in credentials if row.get("is_default")]
            default_statuses = {str(row.get("status")) for row in defaults}
            if "invalid" in default_statuses:
                credential_status = "invalid"
            elif "verification_unavailable" in default_statuses:
                credential_status = "verification_unavailable"
            elif "verified" in default_statuses:
                credential_status = "verified"

            def provider_supports(row: Mapping[str, Any], mode: str) -> bool:
                provider = providers.get(str(row.get("provider_id")), {})
                return provider.get("status") == "enabled" and bool(
                    provider.get(f"{mode}_enabled")
                )

            selected_provider_enabled = all(
                providers.get(str(row.get("provider_id")), {}).get("status")
                == "enabled"
                for row in defaults
            )
            verified_byok_lane = any(
                row.get("status") == "verified"
                and row.get("is_default")
                and provider_supports(row, "byok")
                for row in credentials
            )
            platform_lane = total_available > 0 and any(
                row.get("status") == "enabled" and row.get("platform_enabled")
                for row in providers.values()
            )
            if hasattr(self.provider_base, "get_platform_credential_public"):
                from dashboard.backend.domain.model_providers.service import (
                    ModelProviderService,
                )

                options = ModelProviderService(
                    store=self.provider_base
                ).list_execution_options(user_id)
                verified_byok_lane = any(option.byok_available for option in options)
                platform_lane = total_available > 0 and any(
                    option.platform_credits_available for option in options
                )

        failures, beyond_deadline = self._run_health(user_id, current)
        return CurrentOperationalFacts(
            user_id=user_id,
            account_restricted=billing.get("account_status") == "restricted",
            usable_billing_lane=platform_lane or verified_byok_lane,
            selected_provider_enabled=selected_provider_enabled,
            default_credential_status=credential_status,
            failed_terminal_runs_24h=failures,
            run_beyond_safe_deadline=beyond_deadline,
        )

    def get_projection_job(self, job_name: str) -> ProjectionJob | None:
        name = _projection_job_name(job_name)
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM analytics_projection_jobs WHERE job_name=%s",
                    (name,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return ProjectionJob(
            job_name=_row_value(row, "job_name"),
            window_start=date.fromisoformat(str(_row_value(row, "window_start"))),
            window_end=date.fromisoformat(str(_row_value(row, "window_end"))),
            cursor=_row_value(row, "cursor"),
            status=_row_value(row, "status"),
            updated_at=_timestamp(_row_value(row, "updated_at")),
        )

    def save_projection_job(self, job: ProjectionJob) -> ProjectionJob:
        _projection_job_name(job.job_name)
        values = (
            job.job_name,
            job.window_start.isoformat(),
            job.window_end.isoformat(),
            job.cursor,
            job.status,
            utc_iso(job.updated_at),
        )
        sql = """
            INSERT INTO analytics_projection_jobs (
                job_name, window_start, window_end, cursor, status, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT(job_name) DO UPDATE SET
                window_start=excluded.window_start,
                window_end=excluded.window_end,
                cursor=excluded.cursor,
                status=excluded.status,
                updated_at=excluded.updated_at
        """
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, values)
        return job
```

- [ ] **Step 5: Run the affected tests and confirm they pass.**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_value_repository_postgres.py -v
python -m pytest dashboard/backend/tests/domain/analytics/test_value_repository.py -v
python -m pytest dashboard/backend/tests/domain/analytics/test_repository_postgres.py -v
```

Expected: PASS on the first two files. In the third, `test_postgres_user_value_projection_round_trip` and the other four `@pg_only` cases SKIP unless `TEST_POSTGRES_URL` is set in this environment (matching every other `@pg_only` case already in the file); the non-`@pg_only` cases (`test_build_analytics_store_defaults_to_sqlite`, `test_public_repository_methods_match`, `test_postgres_ddl_declares_user_value_projection_storage`, etc.) PASS unconditionally since they are unaffected by this change. If `TEST_POSTGRES_URL` is set locally, `test_postgres_user_value_projection_round_trip` must PASS against `PostgresValueAnalyticsStore` — this is the live behavioural proof the extraction did not change any query.

- [ ] **Step 6: Switch every production construction site to `build_value_analytics_store()`.**

In `dashboard/backend/domain/analytics/service.py`, replace (lines 237, 243):

```python
    from .value_repository import ValueAnalyticsStore
```

with:

```python
    from .value_repository import build_value_analytics_store
```

and:

```python
        value_store=ValueAnalyticsStore(analytics_store),
```

with:

```python
        value_store=build_value_analytics_store(analytics_store),
```

In `dashboard/backend/domain/analytics/states.py`, replace the import block (lines 23-27):

```python
from .value_repository import (
    UserLifecycleDailySnapshot,
    UserValueSnapshot,
    ValueAnalyticsStore,
)
```

with:

```python
from .value_repository import (
    UserLifecycleDailySnapshot,
    UserValueSnapshot,
    ValueAnalyticsStore,
    build_value_analytics_store,
)
```

(`ValueAnalyticsStore` stays imported: it is still used as a type annotation on `value_store: ValueAnalyticsStore | None = None` at all three call sites below — only the construction call changes.) Then replace, at each of the three call sites (lines 670, 691, 754):

```python
    values = value_store or ValueAnalyticsStore(states.base_store)
```

with:

```python
    values = value_store or build_value_analytics_store(states.base_store)
```

In `dashboard/backend/domain/analytics/lifecycle_backfill.py`, replace the import block (lines 16-20):

```python
from .value_repository import (
    ProjectionJob,
    UserLifecycleDailySnapshot,
    ValueAnalyticsStore,
)
```

with:

```python
from .value_repository import (
    ProjectionJob,
    UserLifecycleDailySnapshot,
    ValueAnalyticsStore,
    build_value_analytics_store,
)
```

Then, inside `LifecycleBackfillSource.__init__` (line 146):

```python
        self.value_store = value_store or ValueAnalyticsStore(self.analytics_base)
```

becomes:

```python
        self.value_store = value_store or build_value_analytics_store(self.analytics_base)
```

and inside `run_lifecycle_backfill_batch` (line 406):

```python
    values = store or ValueAnalyticsStore()
```

becomes:

```python
    values = store or build_value_analytics_store()
```

In `dashboard/backend/domain/analytics/retention.py`, replace the import (line 12):

```python
from .value_repository import ValueAnalyticsStore
```

with:

```python
from .value_repository import build_value_analytics_store
```

and the module-level construction (lines 172-179):

```python
analytics_retention_service = AnalyticsRetentionService(
    store=analytics_store,
    value_store=ValueAnalyticsStore(
        analytics_store,
        credits_base=object(),
        provider_base=object(),
        agent_base=object(),
        run_base=object(),
    ),
)
```

with:

```python
analytics_retention_service = AnalyticsRetentionService(
    store=analytics_store,
    value_store=build_value_analytics_store(
        analytics_store,
        credits_base=object(),
        provider_base=object(),
        agent_base=object(),
        run_base=object(),
    ),
)
```

In `dashboard/backend/domain/analytics/value_queries.py`, replace the import block (lines 28-33):

```python
from .value_repository import (
    CommercialValueFact,
    UserLifecycleDailySnapshot,
    UserValueSnapshot,
    ValueAnalyticsStore,
)
```

with:

```python
from .value_repository import (
    CommercialValueFact,
    UserLifecycleDailySnapshot,
    UserValueSnapshot,
    ValueAnalyticsStore,
    build_value_analytics_store,
)
```

(kept for its type annotation there too) and, at the call site (line 494):

```python
        self.value_store = value_store or ValueAnalyticsStore(store)
```

becomes:

```python
        self.value_store = value_store or build_value_analytics_store(store)
```

- [ ] **Step 7: Run the full analytics test directory and `test_store_twin_parity.py` for regressions.**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/ dashboard/backend/tests/test_store_twin_parity.py -v
```

Expected: every case PASSES **except** `test_store_twin_parity.py::test_every_postgres_twin_module_is_registered`, which now FAILS because `value_repository_postgres.py` exists on disk but is not yet in `_TWINS` — this is the expected, anticipated state Task 3 resolves next, not a regression from this task. No other test in `dashboard/backend/tests/domain/analytics/` should fail: `build_value_analytics_store` preserves every prior call site's runtime behaviour (SQLite base in, SQLite twin out — unchanged), and Postgres bases still route to a class exposing the identical 14 methods and SQL.

- [ ] **Step 8: Commit.**

```bash
git add dashboard/backend/domain/analytics/value_repository.py \
        dashboard/backend/domain/analytics/value_repository_postgres.py \
        dashboard/backend/domain/analytics/service.py \
        dashboard/backend/domain/analytics/states.py \
        dashboard/backend/domain/analytics/lifecycle_backfill.py \
        dashboard/backend/domain/analytics/retention.py \
        dashboard/backend/domain/analytics/value_queries.py \
        dashboard/backend/tests/domain/analytics/test_value_repository_postgres.py \
        dashboard/backend/tests/domain/analytics/test_repository_postgres.py
git status --short
git commit -m "refactor: extract PostgresValueAnalyticsStore from ValueAnalyticsStore"
```

`git status --short` first is not optional: a bare backend import rewrites `dashboard/storage/data/backtest.db`, which must never be staged.

---

### Task 3: Register the pair in `_TWINS`; tolerate a store with no DDL of its own

**Files:**
- Modify: `dashboard/backend/tests/test_store_twin_parity.py` (`_TWINS`, plus a new `_NO_OWN_DDL_TWINS` registry and one guard inside `test_postgres_twin_schema_columns_match_sqlite`)

**Interfaces:**
- Consumes: `PostgresValueAnalyticsStore` and `ValueAnalyticsStore` from Task 2; the deduplicated `_TWINS` from Task 1.
- Produces: `_TWINS` with 12 distinct tuples (11 from Task 1 plus the new pair); `_NO_OWN_DDL_TWINS: dict[str, str]`, consulted only by `test_postgres_twin_schema_columns_match_sqlite`.

Before writing code: registering the pair makes every test parametrized over `_TWINS` run one more instance. Three of those tests pass on the new pair with **no code change**, because they are structurally vacuous for a module with zero `CREATE TABLE`/`CREATE INDEX` statements — worth stating so nobody "fixes" them unnecessarily:

- `test_postgres_twin_exposes_every_sqlite_method` / `test_postgres_twin_signatures_match_sqlite`: pass because Task 2 already made the method sets and signatures identical.
- `test_ddl_parser_sees_every_create_index`: `parsed == keyword_hits` is `0 == 0`.
- `test_postgres_twin_indexes_a_migrated_column_only_after_adding_it`: the `too_early` list starts and stays empty.
- `test_postgres_twin_repeats_every_sqlite_lazy_migration`: `sqlite_migrated` is `{}`, so the `for table, columns in sorted(sqlite_migrated.items())` loop never runs and `gaps` stays empty.

Only `test_postgres_twin_schema_columns_match_sqlite` fails, on its own non-vacuity guard (`assert sqlite_schema, f"no CREATE TABLE parsed from {sqlite_path}"` — `sqlite_schema` is `{}` because `value_repository.py` declares no table). That guard exists to catch a parser that silently extracted nothing from a store that *does* own tables; `ValueAnalyticsStore` genuinely owns none, so this is the one test that needs an explicit, documented exemption rather than a vacuous pass.

- [ ] **Step 1: Write the failing test.**

Add to `dashboard/backend/tests/test_store_twin_parity.py`, directly after the `test_twins_registry_has_no_duplicate_pairs` test added in Task 1:

```python
def test_value_analytics_store_pair_is_registered():
    """PR T's whole point: a store with no *_postgres.py file is invisible
    to test_every_postgres_twin_module_is_registered, which starts from
    files on disk. This asserts the registry side directly.
    """
    assert (
        "dashboard.backend.domain.analytics.value_repository",
        "ValueAnalyticsStore",
        "dashboard.backend.domain.analytics.value_repository_postgres",
        "PostgresValueAnalyticsStore",
    ) in _TWINS
```

- [ ] **Step 2: Run the test and confirm it fails; run the full parametrized suite and confirm the one expected pre-existing failure.**

```bash
python -m pytest dashboard/backend/tests/test_store_twin_parity.py::test_value_analytics_store_pair_is_registered -v
python -m pytest dashboard/backend/tests/test_store_twin_parity.py -v
```

Expected: the new test FAILS (tuple not in `_TWINS` yet). The full run shows the same single failure Task 2 Step 7 already surfaced (`test_every_postgres_twin_module_is_registered`), and no others — confirming nothing else regressed between tasks.

- [ ] **Step 3: Register the pair and add the DDL tolerance.**

In `dashboard/backend/tests/test_store_twin_parity.py`, append to `_TWINS` (after the `BacktestDatabase` tuple, before the closing `]`):

```python
    (
        "dashboard.backend.domain.analytics.value_repository",
        "ValueAnalyticsStore",
        "dashboard.backend.domain.analytics.value_repository_postgres",
        "PostgresValueAnalyticsStore",
    ),
```

Then, directly above `_DELIBERATELY_POSTGRES_ABSENT_TABLES` (the existing table-level exemption registry), add the module-level exemption:

```python
# Twins whose class owns no table of its own: it reads and writes tables
# declared by a *different*, already-registered twin through an injected
# connection, rather than declaring CREATE TABLE/CREATE INDEX itself. For
# such a pair, test_postgres_twin_schema_columns_match_sqlite's non-vacuity
# assert ("no CREATE TABLE parsed from ...") would fail on every entry, not
# because a column drifted but because there is nothing to parse -- exactly
# the false positive that assert exists to prevent for a store that *does*
# own tables. Each entry names which already-registered pair actually owns
# the schema, so this cannot silently swallow a future twin that adds its
# own DDL. The other three DDL-adjacent tests in this file
# (test_ddl_parser_sees_every_create_index,
# test_postgres_twin_indexes_a_migrated_column_only_after_adding_it,
# test_postgres_twin_repeats_every_sqlite_lazy_migration) need no matching
# entry: each is vacuously true for zero DDL statements by construction, and
# that vacuity is not a blind spot the way the non-vacuity assert would be.
_NO_OWN_DDL_TWINS: dict[str, str] = {
    "PostgresValueAnalyticsStore": (
        "ValueAnalyticsStore/PostgresValueAnalyticsStore own no CREATE TABLE "
        "or CREATE INDEX: both read and write user_analytics_snapshots, "
        "user_lifecycle_daily_snapshots and analytics_projection_jobs "
        "through an injected analytics_base connection, and that base is "
        "AnalyticsStore/PostgresAnalyticsStore -- already a registered pair "
        "above, whose own schema-column test covers those tables."
    ),
}
```

Finally, in `test_postgres_twin_schema_columns_match_sqlite`, insert a skip immediately after the docstring:

```python
def test_postgres_twin_schema_columns_match_sqlite(
    sqlite_mod, sqlite_cls, postgres_mod, postgres_cls
):
    """The column half of the #227 bug, which signature parity cannot see.

    Compares column *names* only: types legitimately differ per dialect
    (REAL/DOUBLE PRECISION, INTEGER/BOOLEAN, TIMESTAMP/TEXT).
    """
    if postgres_cls in _NO_OWN_DDL_TWINS:
        pytest.skip(_NO_OWN_DDL_TWINS[postgres_cls])
    sqlite_path = _module_source_path(sqlite_mod)
```

(The next line, `assert sqlite_path.is_file(), ...`, is unchanged; only the two new lines are inserted above it.)

- [ ] **Step 4: Run the new test, then the full parametrized suite, and confirm everything passes.**

```bash
python -m pytest dashboard/backend/tests/test_store_twin_parity.py -v
```

Expected: PASS on every case, including `test_every_postgres_twin_module_is_registered` (now sees `value_repository_postgres.py` registered) and `test_postgres_twin_schema_columns_match_sqlite[PostgresValueAnalyticsStore]` (now SKIPPED with the documented reason, not failing).

- [ ] **Step 5: Run the full backend suite once as a broader regression check.**

```bash
python -m pytest dashboard/backend/tests/ -q
```

Expected: PASS (aside from any `@pg_only`/live-service cases already conditioned on infrastructure not present in this environment, unrelated to this change).

- [ ] **Step 6: Commit.**

```bash
git add dashboard/backend/tests/test_store_twin_parity.py
git status --short
git commit -m "test: register the ValueAnalyticsStore Postgres twin in _TWINS"
```

---

### Task 4: Absence-direction check — a dialect branch outside a registered twin must be named

**Files:**
- Modify: `dashboard/backend/tests/test_store_twin_parity.py` (new regex, allowlist, and test, placed after `test_every_postgres_twin_module_is_registered`)

**Interfaces:**
- Consumes: `_TWINS` (12 entries, from Task 3) and `_module_source_path` (existing helper).
- Produces: `_DIALECT_BRANCH_ALLOWLIST: dict[str, str]` and `test_dialect_branches_outside_a_registered_twin_are_allowlisted()`. Nothing else consumes these.

Design doc §13 row T words this as "assert zero". That means zero *unexplained* hits, not zero hits: a literal zero would fail on `states.py`, `rollups.py`, `query_service.py`, `lifecycle_backfill.py` and `backfill.py`, all of which branch over an injected, already-twinned base store and are out of PR T's scope (Not in this PR, below), and rewriting them here would require the query changes the row's "Must not" column forbids. So the test carries `_DIALECT_BRANCH_ALLOWLIST`, one reasoned entry per file, and pairs it with a `stale` assertion so an entry cannot outlive the branch it excuses — the list can only shrink. §13 row T names the same reading.

The scan is over the whole backend, not just `domain/analytics`. Verified now, against this worktree, with the pattern the test below uses:

```bash
python3 -c "
import re
from pathlib import Path
pattern = re.compile(r'is_postgres|hasattr\([^)]*[\"\x27]database_url[\"\x27]')
backend = Path('dashboard/backend')
for path in sorted(backend.rglob('*.py')):
    if 'tests' in path.parts:
        continue
    if pattern.search(path.read_text(encoding='utf-8')):
        print(path)
"
```

Run against the pre-Task-2 tree, every hit is inside `dashboard/backend/domain/analytics/`: `backfill.py`, `lifecycle_backfill.py`, `query_service.py`, `rollups.py`, `states.py`, `value_repository.py`. No hit exists anywhere else in the backend — `database.py`, `db_url.py` and every other store's `_build_*_store()` factory select on `os.getenv("...DATABASE_URL")` directly, never on `hasattr(x, "database_url")` or an `is_postgres` name, so none of them needs an allowlist entry. After Task 2, `value_repository.py`'s hits are reduced from 24 to exactly 2 (`hasattr(self.credits_base, "database_url")` inside `list_commercial_values` and `list_credit_activity` — the cross-store branch this plan's architecture section explains is deliberately unchanged), and `value_repository_postgres.py` is excluded from the scan because it is now a registered `postgres_mod`. The allowlist below reflects that post-Task-2 state.

- [ ] **Step 1: Write the failing test.**

Add to `dashboard/backend/tests/test_store_twin_parity.py`, directly after `test_every_postgres_twin_module_is_registered` (after line 169, before the "Axis 1: call signatures" section comment):

```python
# --------------------------------------------------------------------------
# Absence-direction check: a dialect branch outside a registered twin
# --------------------------------------------------------------------------
#
# The registry above and the rglob discovery test are both existence-first:
# they start from a *_postgres.py file or a _TWINS entry and check it is
# complete. Neither can catch a class that branches SQLite-vs-Postgres
# inline -- exactly what ValueAnalyticsStore did until PR T -- because such
# a class owns no *_postgres.py file to discover. This test starts from the
# other end: every dialect-branch idiom in the backend, asking "is this
# inside a registered twin?" A hit outside one is either a twin extraction
# that has not happened yet, or a non-twin helper reading a table a
# registered twin already owns. Either way it needs a name below with a
# reason -- silence here is exactly how ValueAnalyticsStore's 21 branches
# went unnoticed for as long as they did.
_DIALECT_BRANCH_PATTERN = re.compile(
    r"is_postgres|hasattr\([^)]*[\"']database_url[\"']"
)

_DIALECT_BRANCH_ALLOWLIST: dict[str, str] = {
    "dashboard/backend/domain/analytics/value_repository.py": (
        "list_commercial_values and list_credit_activity branch on the "
        "injected credits_base's own dialect (hasattr(self.credits_base, "
        "'database_url')), not on ValueAnalyticsStore's -- a caller can (and "
        "in tests does) pair either analytics_base with either credits_base. "
        "PostgresValueAnalyticsStore carries the identical branch for the "
        "same reason (see value_repository_postgres.py's module docstring); "
        "both are correct, not a missing extraction."
    ),
    "dashboard/backend/domain/analytics/states.py": (
        "AnalyticsStateStore dialect-branches over the already-twinned "
        "AnalyticsStore/PostgresAnalyticsStore base_store for the legacy "
        "user_analytics_snapshots table, not a store of its own. Admin "
        "layer redesign PR A (docs/superpowers/specs/"
        "2026-09-15-admin-layer-redesign-design.md §6.5) deletes the "
        "five-state columns and repair path this class serves, which "
        "removes or shrinks this branch -- tracked there, not in PR T."
    ),
    "dashboard/backend/domain/analytics/rollups.py": (
        "AnalyticsRollupStore dialect-branches over the already-twinned "
        "AnalyticsStore/PostgresAnalyticsStore base_store for "
        "analytics_daily_rollups, a table that pair already declares and "
        "parity-checks (repository.py:100-117 / repository_postgres.py:96-113). "
        "Not a store of its own; out of scope for PR T."
    ),
    "dashboard/backend/domain/analytics/query_service.py": (
        "AnalyticsQueryService dialect-branches over the already-twinned "
        "AnalyticsStore/PostgresAnalyticsStore base_store for the legacy "
        "overview/users query surface. Not a store of its own; that surface "
        "is rewritten in admin layer redesign PR A/PR B (design doc §6.10), "
        "which is where this branch is next touched."
    ),
    "dashboard/backend/domain/analytics/lifecycle_backfill.py": (
        "LifecycleBackfillSource dialect-branches over the already-twinned "
        "analytics_base for the historical 8-week lifecycle reconstruction "
        "job. Not a store of its own; admin layer redesign PR A's daily job "
        "and migration (design doc §6.9) replace this reconstruction path."
    ),
    "dashboard/backend/domain/analytics/backfill.py": (
        "_query_all dialect-branches over injected credits/agent stores for "
        "the authoritative-history backfill; admin layer redesign PR A "
        "moves those two ledger and run reads onto CreditsStore/"
        "PostgresCreditsStore and the run-history store (design doc §12 "
        "item 1), which removes this branch. _existing_source_event_ids "
        "separately dialect-branches over the already-twinned analytics_store "
        "to deduplicate against analytics_events, the analytics domain's own "
        "table -- a different case, out of scope for PR T."
    ),
}


def test_dialect_branches_outside_a_registered_twin_are_allowlisted():
    """A store that inlines SQLite/Postgres branching owns no *_postgres.py
    file, so test_every_postgres_twin_module_is_registered above cannot see
    it -- that test starts from files on disk, and there is no second file
    for a branch like this. This test starts from the branch instead, across
    every non-test module, and requires each hit to be named here.

    ValueAnalyticsStore had 21 such branches until PR T split it into a real
    twin. Nothing before this test would have caught it going in, and
    nothing would catch the next one either.
    """
    backend = _REPO_ROOT / "dashboard" / "backend"
    registered_postgres_paths = {
        _module_source_path(postgres_mod) for _, _, postgres_mod, _ in _TWINS
    }
    hits: set[str] = set()
    for path in backend.rglob("*.py"):
        if "tests" in path.parts:
            continue
        if path in registered_postgres_paths:
            continue
        source = path.read_text(encoding="utf-8")
        if _DIALECT_BRANCH_PATTERN.search(source):
            hits.add(str(path.relative_to(_REPO_ROOT)))

    unlisted = sorted(hits - set(_DIALECT_BRANCH_ALLOWLIST))
    assert not unlisted, (
        "Dialect-branch idiom (is_postgres / hasattr(*, 'database_url')) "
        "found outside a registered twin, with no allowlist entry. Either "
        "extract a real Postgres twin and add it to _TWINS, or add this "
        f"file to _DIALECT_BRANCH_ALLOWLIST with a reason: {unlisted}"
    )

    stale = sorted(set(_DIALECT_BRANCH_ALLOWLIST) - hits)
    assert not stale, (
        "_DIALECT_BRANCH_ALLOWLIST names file(s) with no dialect-branch "
        "idiom left in them -- the exemption is stale and silently widens "
        f"this guard for whatever gets added there next: {stale}"
    )
```

Add `import re` to the top-level imports of `dashboard/backend/tests/test_store_twin_parity.py` if it is not already imported (the file currently imports `ast, importlib, inspect, re, pathlib.Path, typing.NamedTuple` — `re` is already there via the `_SQL_IDENT`/`_CREATE_TABLE` patterns, so no import change is needed; confirm with `grep -n "^import re" dashboard/backend/tests/test_store_twin_parity.py` before assuming otherwise).

- [ ] **Step 2: Run the test and confirm it fails.**

```bash
python -m pytest dashboard/backend/tests/test_store_twin_parity.py::test_dialect_branches_outside_a_registered_twin_are_allowlisted -v
```

Expected: this test as written (with the allowlist already filled in from verified hits) should PASS immediately, because Step 1 already contains the correct, verified allowlist rather than an empty one — there is no separate "implementation" step for this task. To see it fail as a genuine regression check, temporarily comment out one allowlist entry (for example, delete the `"dashboard/backend/domain/analytics/rollups.py"` entry), rerun, confirm the `unlisted` assertion fails naming `rollups.py`, then restore the entry. This demonstrates the guard has teeth before trusting it.

- [ ] **Step 3: Run the full parity suite and confirm no other regression.**

```bash
python -m pytest dashboard/backend/tests/test_store_twin_parity.py -v
```

Expected: PASS on every case.

- [ ] **Step 4: Commit.**

```bash
git add dashboard/backend/tests/test_store_twin_parity.py
git status --short
git commit -m "test: add an absence-direction check for dialect branches outside a twin"
```

---

### Task 5: Document the twin in `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md:89` (Architecture → Persistence bullet)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing consumed elsewhere; this is a doc-only change.

- [ ] **Step 1: Add the clause.**

In `CLAUDE.md`, the Persistence bullet currently reads (line 89):

```markdown
- **Persistence** (`database.py` + per-store repositories like `domain/runs/repository.py`, `domain/strategies/repository.py`): thin SQLite wrappers over `DATABASE_PATH` in **WAL journal mode** (readers aren't blocked by finalize's heavy writes); schema is created lazily and self-migrates. `agent_runs` carries a JSON `metadata` column recording the effective `LLM_MAX_OUTPUT_TOKENS` per run.
```

Replace it with:

```markdown
- **Persistence** (`database.py` + per-store repositories like `domain/runs/repository.py`, `domain/strategies/repository.py`): thin SQLite wrappers over `DATABASE_PATH` in **WAL journal mode** (readers aren't blocked by finalize's heavy writes); schema is created lazily and self-migrates. `agent_runs` carries a JSON `metadata` column recording the effective `LLM_MAX_OUTPUT_TOKENS` per run. `domain/analytics/value_repository.py`'s `ValueAnalyticsStore` had 21 `is_postgres` branches and no Postgres twin until the 2026-09-15 admin layer redesign's PR T split it into `value_repository_postgres.py`'s `PostgresValueAnalyticsStore`, registered in `tests/test_store_twin_parity.py::_TWINS` like every other store pair.
```

This is a single-line insertion at the end of the existing bullet; nothing else in the file changes.

- [ ] **Step 2: Verify the file is still valid Markdown by inspection and commit.**

```bash
git diff CLAUDE.md
git add CLAUDE.md
git status --short
git commit -m "docs: note the ValueAnalyticsStore twin in the Persistence bullet"
```

---

## Verification

Run from the repo root.

```bash
# The parity file alone (fast; the primary target of this PR)
python -m pytest dashboard/backend/tests/test_store_twin_parity.py -v

# The analytics domain tests (behaviour-preserving check)
python -m pytest dashboard/backend/tests/domain/analytics/ -v

# The Postgres-tier round-trip specifically, when TEST_POSTGRES_URL is set
TEST_POSTGRES_URL=postgresql://... python -m pytest \
    dashboard/backend/tests/domain/analytics/test_repository_postgres.py -v

# Full backend suite
python -m pytest dashboard/backend/tests/ -q
```

CI's Postgres tier must be green before merge (§13 row T: "Green on the CI Postgres tier before merge"). That tier is the workflow that sets `TEST_POSTGRES_URL` and runs the `@pg_only`-marked cases against a real service container — confirm which `.github/workflows/*.yml` provisions a `postgres:` service and sets `TEST_POSTGRES_URL` before merging (`grep -rl "TEST_POSTGRES_URL" .github/workflows/`), and check that workflow's latest run on this branch is green, in addition to the local commands above. `dashboard/backend/tests/test_ci_postgres_wired.py` (referenced in this file's own module docstring: "test_ci_postgres_wired.py asserts CI itself never lands in that state") independently pins that CI can never silently drop this tier — if that test passes locally with `TEST_POSTGRES_URL` unset, it is asserting CI's own configuration file, not this environment, so its pass here does not substitute for checking the workflow run itself.

## Not in this PR

Per design doc §13 (PR T's "Must not" column: "Change any query, response or table") and D22, the following are explicitly **not** done here:

- **Any query, response-shape or table change.** Every SQL string in `value_repository.py` and `value_repository_postgres.py` is transcribed from the pre-split source; `PostgresValueAnalyticsStore` carries the Postgres arm of each former branch byte-for-byte. A reviewer who finds a rewritten query, a renamed column or a new index in this diff has found a scope violation, not an improvement.
- **PR A's ~12 new dual-dialect methods on the twins** (design doc §12 item 2). PR T only makes those methods land on a *twinned* class; adding them (`user_daily_facts`, `user_activity`, `lifecycle_transitions`, the day claim, the population operational signals) is PR A's content, and the parity guard will cover them there because of this PR.
- **Frontend work.** No file under `dashboard/frontend/` is touched, so no `?v=` pin moves and no node-driven test applies (see Global Constraints).
- **Extracting the five other files on `_DIALECT_BRANCH_ALLOWLIST`.** `states.py`, `rollups.py`, `query_service.py`, `lifecycle_backfill.py` and `backfill.py` keep dialect-branching over an *injected, already-twinned* base store (or, for `backfill.py::_query_all`, over injected credits/agent stores). None is a store of its own with a table to twin; each allowlist entry names the PR that next touches its branch (PR A for `states.py`, `lifecycle_backfill.py`, `backfill.py`; PR A/PR B for `query_service.py`; `rollups.py` is a legitimate non-twin helper over a table `AnalyticsStore`/`PostgresAnalyticsStore` already declares). Rewriting them here would require the query changes the "Must not" column forbids. PR A's row in §13 removes `backfill.py`'s entry when it moves those reads onto `CreditsStore`/`PostgresCreditsStore`; the `stale` assertion in Task 4 is what forces that entry out at the same time.
- **The two `hasattr(self.credits_base, "database_url")` branches left in `value_repository.py`.** They select on the *credits* store's dialect, not this class's, and are correct on both twins (Architecture, above); they are allowlisted with that reason, not extracted.

## Acceptance

- [ ] `ValueAnalyticsStore` (`value_repository.py`) contains zero `is_postgres` branches and zero `hasattr(self.analytics_base, "database_url")` checks; its two `hasattr(self.credits_base, "database_url")` checks (in `list_commercial_values`/`list_credit_activity`) remain, unchanged, and are allowlisted with a reason.
- [ ] `PostgresValueAnalyticsStore` (`value_repository_postgres.py`) exposes the same 14 public methods and constructor signature as `ValueAnalyticsStore`.
- [ ] `build_value_analytics_store()` exists in `value_repository.py` and is the sole construction path at every production call site (`service.py`, `states.py` ×3, `lifecycle_backfill.py` ×2, `retention.py`, `value_queries.py`).
- [ ] `_TWINS` in `test_store_twin_parity.py` has no duplicate tuples and registers the `ValueAnalyticsStore`/`PostgresValueAnalyticsStore` pair (12 distinct entries total).
- [ ] `test_postgres_twin_schema_columns_match_sqlite` tolerates the new pair via a documented `_NO_OWN_DDL_TWINS` exemption, not a weakened assertion.
- [ ] A new absence-direction test scans the backend for `is_postgres`/`hasattr(*, "database_url")` outside a registered twin and requires every hit to be named in `_DIALECT_BRANCH_ALLOWLIST` with a reason; the allowlist has no stale entries.
- [ ] No query, response shape, or table changed anywhere in this PR.
- [ ] `CLAUDE.md`'s Persistence bullet names the new twin.
- [ ] CI's Postgres tier is green on this branch before merge.

## Self-review notes

- **Spec coverage:** §4.5 (method count, branch count, no twin file, `_TWINS` duplicate) is the basis for Tasks 1-3 and was re-verified against the current source (14 public methods listed and confirmed by direct read of `value_repository.py`; 21 `is_postgres` + 3 `hasattr` confirmed by `rg -c`; duplicate `_TWINS` tuple confirmed by direct read of `test_store_twin_parity.py:54-127`). §6.14's twin-ownership row is Task 2-4 collectively. §12 items 2 and 8 map directly to Task 2 (extraction before the dual-dialect methods land on the untwinned class) and Task 1 (the duplicate). §13 row T's "Must not: Change any query, response or table" is why `list_commercial_values`/`list_credit_activity`/`get_operational_facts`/`_run_health` are duplicated verbatim rather than "cleaned up," and why every SQL string in both new files is transcribed from the read source rather than rewritten. D22's reasoning (extract before PR A, when the untwinned class would instead gain ~12 more branches) is the plan's stated motivation, not just a citation.
- **Placeholder scan:** every code block in Tasks 1-5 is complete, runnable Python (or exact CLAUDE.md prose) transcribed from source already read in this session (`value_repository.py`, `test_store_twin_parity.py`, `test_repository_postgres.py`, `retention.py`, `states.py`, `lifecycle_backfill.py`, `value_queries.py`, `service.py`, `CLAUDE.md`) or newly authored against that source; no step says "similar to Task N" or leaves a TODO. The Task 4 allowlist reasons were derived from an actual `rg`/regex scan run against this worktree during planning (recorded in Task 4's preamble), not guessed — the "check database.py / db_url.py ... factory selection" possibility named in the brief was checked and found to have zero matching hits, so no such entry was invented to fill a slot that would have failed the "no stale entries" assertion.
- **Name consistency:** `build_value_analytics_store`, `ValueAnalyticsStore`, `PostgresValueAnalyticsStore`, `_current_snapshot_from_row`, `_fetchall`, `_user_clause`, `_projection_job_name`, `_NO_OWN_DDL_TWINS`, `_DIALECT_BRANCH_PATTERN`, `_DIALECT_BRANCH_ALLOWLIST` are spelled identically everywhere they appear across Tasks 1-4. Every call-site edit in Task 2 Step 6 was matched against a fresh `rg -n "ValueAnalyticsStore("` run over the whole backend (excluding tests) so no production construction site is missed or invented.
- **Verified, not assumed:** the exact line numbers, current file contents, and the "14 public methods" and "3 `hasattr`" counts were confirmed by directly reading `value_repository.py` in full (1125 lines) rather than relying on the fact-base summaries alone; the fact-base was used to navigate, and every claim it made that this plan depends on was re-checked at source (the `_run_health` DB-attribution correction in the fact-base, for instance, is why `_run_health`/`get_operational_facts` are described as containing no dialect branch at all, rather than being mistaken for cross-Postgres-project code).
