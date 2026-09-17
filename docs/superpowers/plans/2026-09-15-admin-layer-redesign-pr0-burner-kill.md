# Admin Layer PR 0: Burner Kill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop both outage burners named in the design doc's §2.2 ("Both outage burners are still live") and its §4.3 ("The burners") without building any other part of the superseded 2026-09-12 stopgap plan, add the client-side guard that stops the phantom analytics fetch on every admin visit, add the auth-store 503 mapping the 09-12 plan's Task A4 never landed, and record the reaper-cadence gotcha so the next reader does not reintroduce a burner while working on PR A.

**Architecture:** Two independent backend throttles (a 24-hour snapshot-repair window in `states.py`, and a non-projecting `AnalyticsService` singleton in `service.py` whose own instrumentation-layer safety net must also be disarmed — see Task 3) remove per-request and per-tick full-history recomputes without touching the tables or the daily-job architecture PR A replaces them with. A one-line frontend guard stops `app.js` from firing three analytics GETs into a page that `admin-tabs.js` has already scheduled a navigation away from. A new decorated-free helper in `api/auth.py` maps a user-store connectivity failure to a logged 503 instead of a bare, unmapped 500. None of this touches a database schema, a response shape, or the frontend beyond `admin-tabs.js`/`app.js`.

**Tech Stack:** FastAPI + Pydantic (backend), vanilla-JS IIFE modules with no build step (frontend), pytest (backend tests, run from the repo root), Node.js via `dashboard/backend/tests/_frontend_source.py` (frontend tests, `skipif` when `node` is absent).

**Spec:** `docs/superpowers/specs/2026-09-15-admin-layer-redesign-design.md` — implements §2.2 ("Both outage burners are still live"), §4.2 ("The redirect and the phantom fetches"), §4.3 ("The burners"), §4.6 ("Guards that exist, and what they do not cover" — the `api/auth.py` bullet), and the **PR 0** row of §13 ("Delivery"), including its "Must not" column and D21 ("PR letters are renumbered: PR A is the rewrite").

## Global Constraints

- Never commit `dashboard/storage/data/backtest.db` — importing a backend store module runs `CREATE TABLE IF NOT EXISTS` against `DATABASE_PATH` and can rewrite that seed file; stage files by name (`git add <path> <path>`), never `git add -A` or `git add .`.
- Run every `pytest` invocation from the repo root.
- Node-driven frontend tests (anything reading `dashboard/backend/tests/_frontend_source.py` or shelling out to `node -e`) are `pytest.mark.skipif(shutil.which("node") is None, ...)` — a skip on a machine with no `node` on `PATH`, not a pass; install a current LTS Node to actually exercise Task 4's frontend tests locally.
- The `?v=` pins asserted by `test_admin_analytics_frontend.py::test_app_lifecycle_and_cache_versions_are_wired` must be updated in lockstep with any cache-buster bump — Task 4 bumps `app.js` and `js/admin-tabs.js` and edits this exact test in the same step.
- PR 0 must not build any other part of the superseded 09-12 stopgap: no shared stale-window constant, no one-read recompute, no spy-store sweep-budget test (design doc §13, PR 0's "Must not" column; D21). Task 3 below implements a clause the design doc's PR 0 row states explicitly — "`instrumentation.py::_emit`'s fallback recompute disarmed (it fires exactly when the service does not project, so the flag alone would move the burner, not remove it)" — it is not scope this plan adds. The 09-12 plan gated that same fallback through the shared stale-window guard the "Must not" column now forbids, which is why Task 3 uses a different mechanism (a no-op recalculator registered at boot) and writes the reasoning out in full.
- PR 0 lands first and alone; nothing in this plan depends on PR T, PR A, PR B, or PR C, and nothing here should be blocked waiting on them (design doc §13, "Ordering constraints").
- `AnalyticsService.record_server_event` and everything it touches (`states.py`, `value_repository.py`) is deleted by PR A — this plan changes exactly two literals in `states.py`/`service.py` and does not restructure either module.

## Task 1: `states.py` — snapshot-repair staleness window, 15 minutes → 24 hours

**Files:**
- Modify: `dashboard/backend/domain/analytics/states.py` — `repair_stale_snapshots`, lines 717-742
- Test: `dashboard/backend/tests/domain/analytics/test_states.py` — extend

**Interfaces:**
- Consumes: nothing new.
- Produces: no signature change. `repair_stale_snapshots(*, now=None, limit=100, stale_after=timedelta(hours=24), store=None) -> int` — only the `stale_after` default changes.

`repair_stale_snapshots` is called exactly once in production, by `maintenance.py::run_analytics_maintenance` (imported and aliased to `repair_snapshots` at lines 69-72, invoked as `repair_snapshots(now=current, limit=page_size)` at lines 127-130), with no `stale_after` override — so the module-level default is the only knob, and the 60-second reaper tick (`RUN_HEARTBEAT_STALE_SECONDS`/`RUN_REAPER_INTERVAL_SECONDS` in `domain/runs/service.py`, default 60s) calls it every tick regardless. Today's 15-minute default means any user who touches a lifecycle-relevant event has their snapshot recomputed from a full 180-day history scan roughly every 15 minutes forever, on top of whatever Task 2/3 stop happening synchronously in-request. `repair_stale_snapshots` currently has **no test anywhere in the repo** — confirmed by `rg -n "repair_stale_snapshots\b" dashboard/backend/` returning only its own definition and the one production call site; the sibling `repair_stale_value_snapshots` has tests, this function does not.

- [ ] **Step 1: Write the failing test**

Add to `dashboard/backend/tests/domain/analytics/test_states.py`. First, extend the existing import from `.states` (currently `from dashboard.backend.domain.analytics.states import (AnalyticsStateStore, UserAnalyticsSnapshot, calculate_user_value_snapshot, calculate_user_state, recalculate_user_snapshot, repair_stale_value_snapshots)`) to also import `repair_stale_snapshots`:

```python
from dashboard.backend.domain.analytics.states import (
    AnalyticsStateStore,
    UserAnalyticsSnapshot,
    calculate_user_value_snapshot,
    calculate_user_state,
    recalculate_user_snapshot,
    repair_stale_snapshots,
    repair_stale_value_snapshots,
)
```

Then append this test function (anywhere after `_fixture` and `NOW` are defined, e.g. immediately before `test_utc_day_transition_repairs_current_and_daily_value_snapshots`):

```python
def test_repair_stale_snapshots_uses_a_24_hour_default_window(tmp_path):
    _service, state_store = _fixture(tmp_path)
    with state_store.base_store._get_connection() as conn:
        conn.execute(
            "INSERT INTO users VALUES (2, 'second@example.test', 'Second', 'x', 'user', ?)",
            ((NOW - timedelta(days=2)).isoformat(),),
        )
    state_store.upsert_snapshot(
        UserAnalyticsSnapshot(
            user_id=1,
            status="onboarding",
            reason_code="no_successful_run",
            human_readable_reason="The user has not completed a successful backtest yet.",
            calculated_at=NOW - timedelta(hours=23),
        )
    )
    state_store.upsert_snapshot(
        UserAnalyticsSnapshot(
            user_id=2,
            status="onboarding",
            reason_code="no_successful_run",
            human_readable_reason="The user has not completed a successful backtest yet.",
            calculated_at=NOW - timedelta(hours=25),
        )
    )

    repaired = repair_stale_snapshots(now=NOW, limit=10, store=state_store)

    assert repaired == 1
    assert state_store.get_snapshot(1).calculated_at == NOW - timedelta(hours=23)
    assert state_store.get_snapshot(2).calculated_at == NOW
```

`stale_after` is deliberately **not** passed: the test exercises the default, which is exactly what this task changes. `_fixture(tmp_path)` (no `created_at` override) seeds user 1 with `created_at=NOW - timedelta(days=1)` — matching the pattern every other test in this file already uses.

- [ ] **Step 2: Run it and confirm the expected failure**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_states.py::test_repair_stale_snapshots_uses_a_24_hour_default_window -v
```

Expected: **FAIL**. With the current 15-minute default, both the 23-hour-old and the 25-hour-old snapshot are far past staleness, so `list_stale_user_ids` selects both users: `repaired == 2` (not the asserted `1`), and `state_store.get_snapshot(1).calculated_at` becomes `NOW` (recalculated) instead of staying at `NOW - timedelta(hours=23)`. The first failing assertion will be `assert repaired == 1` with `repaired == 2`.

- [ ] **Step 3: Change the default**

In `dashboard/backend/domain/analytics/states.py`, change:

```python
def repair_stale_snapshots(
    *,
    now: datetime | None = None,
    limit: int = 100,
    stale_after: timedelta = timedelta(minutes=15),
    store: AnalyticsStateStore | None = None,
) -> int:
```

to:

```python
def repair_stale_snapshots(
    *,
    now: datetime | None = None,
    limit: int = 100,
    stale_after: timedelta = timedelta(hours=24),
    store: AnalyticsStateStore | None = None,
) -> int:
```

- [ ] **Step 4: Run it and confirm the pass**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_states.py -v
```

Expected: **PASS**, including every pre-existing test in the file (this is a default-value change only; no other test in `test_states.py` passes a custom `stale_after` or otherwise depends on the 15-minute value).

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/analytics/states.py dashboard/backend/tests/domain/analytics/test_states.py
git commit -m "$(cat <<'EOF'
fix: throttle the analytics snapshot-repair sweep to a 24-hour window

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

## Task 2: `service.py` — stop the live singleton's synchronous, in-request recompute

**Files:**
- Modify: `dashboard/backend/domain/analytics/service.py` — `_build_analytics_service`, lines 235-245
- Test: `dashboard/backend/tests/domain/analytics/test_service.py` — extend

**Interfaces:**
- Consumes: nothing new.
- Produces: `analytics_service` (the module-level singleton) now has `project_snapshots=False`. `record_server_event` is unchanged; its existing `if not self.project_snapshots: return` guard (line 62 of `_try_recalculate_snapshots`) is what now takes effect on the singleton.

`_build_analytics_service` (`service.py:235-245`) is the sole factory for the module-level `analytics_service` singleton every production caller reaches through `get_analytics_service()`. It currently passes `project_snapshots=True`, so `record_server_event` (`service.py:120-174`) calls `self._try_recalculate_snapshots(...)` synchronously, in the request, for every accepted event whose name is lifecycle-relevant (or is `account_signed_up`/`safe_error_recorded`) — and `_try_recalculate_snapshots` (`service.py:55-77`) then calls `states.recalculate_user_snapshots`, which reads the user's full event history twice (once for the legacy 5-state calculation, once for the value snapshot). One backtest emits four lifecycle events, so a single run pays this cost four times before the reaper sweep in Task 1 ever runs.

- [ ] **Step 1: Write the failing test**

Add to `dashboard/backend/tests/domain/analytics/test_service.py` (it already defines `RecordingStore` and imports `AnalyticsService` from `dashboard.backend.domain.analytics.service`; `NOW` is already defined at module scope). Append:

```python
def test_built_singleton_does_not_synchronously_project_snapshots(monkeypatch):
    from dashboard.backend.domain.analytics import service as service_module
    from dashboard.backend.domain.analytics import states as states_module

    store = RecordingStore()
    monkeypatch.setattr(service_module, "analytics_store", store)
    calls = []
    monkeypatch.setattr(
        states_module,
        "recalculate_user_snapshots",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    built = service_module._build_analytics_service()

    assert built.project_snapshots is False

    result = built.record_server_event(
        event_name="backtest_completed",
        user_id=42,
        source_event_id="run:backtest_completed:run-1",
        source_record_type="run",
        source_record_id="run-1",
        occurred_at=NOW,
        received_at=NOW,
    )

    assert result.created is True
    assert calls == []
```

Monkeypatching `service_module.analytics_store` before calling `_build_analytics_service()` is safe here: `AnalyticsStateStore.__init__`/`AnalyticsRollupStore.__init__`/`ValueAnalyticsStore.__init__` only store the reference they are given (no query runs at construction time), so substituting `RecordingStore()` for the real `AnalyticsStore` does not touch any database. Monkeypatching `states_module.recalculate_user_snapshots` works because `_try_recalculate_snapshots` imports it with a **local** `from .states import recalculate_user_snapshots` inside the method body (`service.py:65`), which resolves the name against the `states` module's current attribute at call time, not at `service.py`'s import time.

- [ ] **Step 2: Run it and confirm the expected failure**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_service.py::test_built_singleton_does_not_synchronously_project_snapshots -v
```

Expected: **FAIL** at `assert built.project_snapshots is False`, with `built.project_snapshots` equal to `True`.

- [ ] **Step 3: Flip the flag**

In `dashboard/backend/domain/analytics/service.py`, change:

```python
def _build_analytics_service() -> AnalyticsService:
    from .states import AnalyticsStateStore
    from .value_repository import ValueAnalyticsStore

    state_store = AnalyticsStateStore(analytics_store)
    return AnalyticsService(
        store=analytics_store,
        state_store=state_store,
        value_store=ValueAnalyticsStore(analytics_store),
        project_snapshots=True,
    )
```

to:

```python
def _build_analytics_service() -> AnalyticsService:
    from .states import AnalyticsStateStore
    from .value_repository import ValueAnalyticsStore

    state_store = AnalyticsStateStore(analytics_store)
    return AnalyticsService(
        store=analytics_store,
        state_store=state_store,
        value_store=ValueAnalyticsStore(analytics_store),
        # PR 0 (burner kill): the live singleton no longer recomputes a
        # snapshot synchronously inside record_server_event. Task 3 of the
        # same plan disarms instrumentation.py's own fallback recalculator,
        # which otherwise steps in for exactly this state (see that task --
        # flipping this flag alone is not sufficient).
        project_snapshots=False,
    )
```

- [ ] **Step 4: Run it and confirm the pass**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_service.py -v
```

Expected: **PASS**, including every pre-existing test in the file — every other test in `test_service.py` constructs its own `AnalyticsService(store=...)` directly (defaulting `project_snapshots=False` already) rather than going through `_build_analytics_service()`, so none of them observe this change.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/analytics/service.py dashboard/backend/tests/domain/analytics/test_service.py
git commit -m "$(cat <<'EOF'
fix: stop the analytics singleton recomputing a snapshot synchronously

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

## Task 3: `instrumentation.py` + `app.py` — disarm the fallback that would otherwise resurrect Task 2's burner

**This task is the second half of the design doc's PR 0 row, quoted here verbatim (§13, "Delivery"):** "`service.py::_build_analytics_service`: `project_snapshots=False`, **and** `instrumentation.py::_emit`'s fallback recompute disarmed (it fires exactly when the service does not project, so the flag alone would move the burner, not remove it)." §2.2 ("Both halves are switched off together"), §4.3 ("so both must be disarmed") and §6.15 item 1 ("PR 0 switches it off, including the `instrumentation.py` fallback; PR B deletes it") say the same thing. It is a separate task from Task 2 because it touches different files and needs its own test, and because the superseded 09-12 plan handled this same fallback by routing it through the shared stale-window guard that PR 0's "Must not" column now forbids — so the mechanism below (a no-op recalculator registered at boot) is new even though the requirement is not. The reasoning is written out in full rather than left implicit, per this repo's "fail-closed is not fail-visible" convention (`CLAUDE.md`, "Fail-closed is not fail-visible" section): Task 2 alone looks complete and silently isn't, which is exactly the failure mode that section warns about.

**Files:**
- Modify: `dashboard/backend/domain/analytics/instrumentation.py` — new function `disable_synchronous_projection`, inserted after `register_snapshot_recalculator` (current lines 59-63), before `_recalculate_snapshot` (current line 66)
- Modify: `dashboard/backend/app.py` — new registration block inside `startup_event`, inserted after the "Analytics maintenance sweep registered" block (current lines 276-288), before the `start_reaper` block (current line 290)
- Test: `dashboard/backend/tests/domain/analytics/test_instrumentation.py` — extend
- Test: `dashboard/backend/tests/test_analytics_maintenance.py` — extend

**Interfaces:**
- Consumes: `instrumentation.register_snapshot_recalculator` (existing).
- Produces: `instrumentation.disable_synchronous_projection() -> None`, called once from `app.py::startup_event`.

**Why this task exists.** `instrumentation.py::_emit` (line 118-137) is the **only** production caller of `AnalyticsService.record_server_event` — confirmed by `rg -n "\.record_server_event\(" dashboard/backend/ ` (excluding tests) returning exactly one hit, `service.py:186`, inside `try_record_server_event`, which is itself called only from `instrumentation.py:134`. `_emit`'s own guard reads:

```python
result = service.try_record_server_event(**kwargs)
if result is not None and not getattr(service, "project_snapshots", False):
    _recalculate_snapshot(int(kwargs["user_id"]), str(event_name))
```

This guard exists as a safety net for a caller-supplied `AnalyticsService` that does not itself project — proven by `test_stored_event_recalculates_snapshot_best_effort` in `test_instrumentation.py`, which constructs a `RecordingService` with **no** `project_snapshots` attribute at all (so `getattr(service, "project_snapshots", False)` is `False`) and asserts that `_recalculate_snapshot` fires. `_recalculate_snapshot` (line 66-77), when no callback has been registered via `register_snapshot_recalculator`, falls back to `states.recalculate_user_snapshots(user_id)` — **the exact same expensive recompute** Task 2 just stopped `record_server_event` from calling itself, over an even wider set of trigger event names (`SNAPSHOT_RELEVANT_EVENTS`, line 48-55, is a strict superset of the `_LIFECYCLE_ACTIVITY_EVENTS` set `record_server_event` checks). `register_snapshot_recalculator` is defined but never called anywhere in production code today (`rg -n "register_snapshot_recalculator" dashboard/backend/` outside tests returns only its own definition), so `_snapshot_recalculator` stays `None` in prod and this fallback is live.

Before Task 2, `analytics_service.project_snapshots` was `True` in production, so `not getattr(service, "project_snapshots", False)` was `False` and this fallback never fired — `record_server_event`'s own path did the (single) recompute. After Task 2 alone, `analytics_service.project_snapshots` is `False`, so the fallback's condition becomes `True` and **the same recompute reappears here instead**, for every event `instrumentation.py` emits (100% of production traffic). Task 2's one-line change, by itself, does not reduce production cost at all — it relocates the call site. This task closes that gap by registering a no-op recalculator at boot, which keeps the fallback mechanism itself intact (the test above continues to pin it against its own stub service) while retiring it for the singleton every real event actually flows through.

- [ ] **Step 1: Write the failing test for the new function**

Append to `dashboard/backend/tests/domain/analytics/test_instrumentation.py` (it already imports `instrumentation`, `AppendEventResult`, and defines `NOW`):

```python
def test_disable_synchronous_projection_makes_the_fallback_a_no_op(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "dashboard.backend.domain.analytics.states.recalculate_user_snapshots",
        lambda *a, **k: calls.append((a, k)),
    )
    monkeypatch.setattr(instrumentation, "_snapshot_recalculator", None)

    instrumentation.disable_synchronous_projection()

    class NonProjectingService:
        project_snapshots = False

        def try_record_server_event(self, **kwargs):
            return AppendEventResult.model_construct(event=None, created=True)

    monkeypatch.setattr(instrumentation, "get_analytics_service", lambda: NonProjectingService())

    instrumentation.emit_agent_event(
        event_name="agent_created",
        user_id=7,
        agent_id="agent-1",
        occurred_at=NOW,
    )

    assert calls == []
```

- [ ] **Step 2: Run it and confirm the expected failure**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_instrumentation.py::test_disable_synchronous_projection_makes_the_fallback_a_no_op -v
```

Expected: **FAIL** with `AttributeError: module 'dashboard.backend.domain.analytics.instrumentation' has no attribute 'disable_synchronous_projection'`.

- [ ] **Step 3: Add the function**

In `dashboard/backend/domain/analytics/instrumentation.py`, immediately after:

```python
def register_snapshot_recalculator(callback: Any) -> None:
    global _snapshot_recalculator
    if callback is not None and not callable(callback):
        raise TypeError("snapshot recalculator must be callable")
    _snapshot_recalculator = callback
```

insert:

```python
def disable_synchronous_projection() -> None:
    """Make this module's own snapshot-recompute fallback inert.

    ``_emit``'s guard below fires whenever the service handling an event does
    not itself project a snapshot (``not getattr(service, "project_snapshots",
    False)``) -- a safety net for a caller-supplied service that never
    learned to project. That is exactly the state PR 0 puts the live
    singleton into (``service.py::_build_analytics_service`` now builds with
    ``project_snapshots=False``), so without this call the fallback --
    calling ``states.recalculate_user_snapshots`` per accepted event -- fires
    on every real request, reproducing the exact synchronous recompute PR 0
    exists to kill, just one module over. Registering a no-op keeps the
    mechanism itself intact (``test_stored_event_recalculates_snapshot_
    best_effort`` pins it directly, against its own stub service, and does
    not go through this default) while retiring it for the singleton every
    production event actually flows through.
    """
    register_snapshot_recalculator(lambda user_id: None)
```

- [ ] **Step 4: Run it and confirm the pass**

```bash
python -m pytest dashboard/backend/tests/domain/analytics/test_instrumentation.py -v
```

Expected: **PASS**, including the pre-existing `test_stored_event_recalculates_snapshot_best_effort` and `test_projecting_service_is_not_recalculated_by_instrumentation` (neither calls `disable_synchronous_projection`, so neither observes it).

- [ ] **Step 5: Write the failing test for the `app.py` wiring**

Add `import inspect` and `import dashboard.backend.app as app_module` to `dashboard/backend/tests/test_analytics_maintenance.py`'s imports (currently `from datetime import date, datetime, timezone`, `from pathlib import Path`, `from types import SimpleNamespace`, `import dashboard.backend.domain.analytics.maintenance as maintenance`), then append:

```python
def test_startup_disables_synchronous_snapshot_projection():
    source = inspect.getsource(app_module.startup_event)
    assert source.count("disable_synchronous_projection()") == 1
```

This follows the exact precedent already in the repo for pinning a startup-event registration by source text rather than by executing the ASGI lifespan: `test_run_lifecycle_unification.py::test_startup_registers_analytics_retention_sweep_once` does the same thing for `register_reaper_sweep(analytics_retention_coordinator.run_if_due)`.

- [ ] **Step 6: Run it and confirm the expected failure**

```bash
python -m pytest dashboard/backend/tests/test_analytics_maintenance.py::test_startup_disables_synchronous_snapshot_projection -v
```

Expected: **FAIL** at `assert source.count("disable_synchronous_projection()") == 1`, with the count equal to `0`.

- [ ] **Step 7: Wire it into `app.py`**

In `dashboard/backend/app.py`, immediately after:

```python
    try:
        from dashboard.backend.domain.analytics.maintenance import (
            run_analytics_maintenance,
        )
        from dashboard.backend.domain.runs.service import register_reaper_sweep

        register_reaper_sweep(run_analytics_maintenance)
        print("🧹 Analytics maintenance sweep registered with the reaper")
    except Exception as e:
        print(
            "WARNING: analytics.maintenance_registration_failed "
            f"category={type(e).__name__}"
        )
```

insert:

```python

    try:
        from dashboard.backend.domain.analytics.instrumentation import (
            disable_synchronous_projection,
        )

        # PR 0: analytics_service now builds with project_snapshots=False, so
        # record_server_event no longer recomputes synchronously. Without
        # this call, instrumentation.py's own fallback guard -- written for a
        # caller-supplied service that never learned to project -- would
        # still call recalculate_user_snapshots per accepted event, moving
        # the burner one module over instead of killing it.
        disable_synchronous_projection()
        print("🧹 Analytics snapshot projection left off the request path")
    except Exception as e:
        print(
            "WARNING: analytics.snapshot_projection_disable_failed "
            f"category={type(e).__name__}"
        )
```

- [ ] **Step 8: Run it and confirm the pass**

```bash
python -m pytest dashboard/backend/tests/test_analytics_maintenance.py -v
python -m pytest dashboard/backend/tests/domain/analytics/test_instrumentation.py -v
```

Expected: **PASS** on both files.

- [ ] **Step 9: Commit**

```bash
git add dashboard/backend/domain/analytics/instrumentation.py \
        dashboard/backend/app.py \
        dashboard/backend/tests/domain/analytics/test_instrumentation.py \
        dashboard/backend/tests/test_analytics_maintenance.py
git commit -m "$(cat <<'EOF'
fix: disarm instrumentation's own snapshot-recompute fallback at boot

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

## Task 4: Frontend guard — stop the phantom analytics fetch on every admin visit

**Files:**
- Modify: `dashboard/frontend/js/admin-tabs.js` — `setTab` (lines 19-30), `onEnter` (lines 79-83)
- Modify: `dashboard/frontend/app.js` — `navigateToPage`'s `page === 'admin'` branch (lines 10839-10861)
- Modify: `dashboard/frontend/app.html` — `app.js?v=131` (line 2701) → `?v=132`; `js/admin-tabs.js?v=7` (line 2710) → `?v=8`
- Modify: `dashboard/backend/tests/test_admin_analytics_frontend.py` — `test_app_lifecycle_and_cache_versions_are_wired` (lines 240-251); new static-shape test
- Modify: `dashboard/backend/tests/test_admin_tabs_redirect.py` — new helper + two tests

**Interfaces:**
- Consumes: nothing new.
- Produces: `window.AdminTabs.onEnter()` now returns `true` when it scheduled the `/admin-analytics` redirect, `false` otherwise (today it returns nothing/`undefined`). `window.AdminTabs.setTab(...)`'s redirect branch now returns the boolean `true` instead of the tab-name string every other branch still returns (nothing outside `admin-tabs.js` consumes `setTab`'s return value today — confirmed by `rg -n "\.setTab\(" dashboard/frontend/ dashboard/backend/tests/` finding no external caller that reads it).

Per design doc §4.2: `window.location.replace()` only *schedules* a navigation; the synchronous script tick that called it keeps running. `app.js::navigateToPage`'s `page === 'admin'` branch calls `loadAdminStats()`, `loadAdminUsers()`, `window.AdminTabs.onEnter()` (which is what schedules the redirect), then unconditionally `window.AdminAnalytics.onEnter()` and `window.AdminAnalyticsValue.onEnter()` — the latter fires three real analytics GETs (`Promise.allSettled([fetchLifecycle(), fetchPriorityUsers(), fetchGroups()])`) into a page that is already leaving. This task makes `onEnter()` report whether it scheduled that departure, and makes `app.js` stop doing the rest of the admin-branch work when it did.

- [ ] **Step 1: Write the failing frontend tests**

In `dashboard/backend/tests/test_admin_tabs_redirect.py`, add a helper that also captures `onEnter()`'s return value (the existing `_run` helper only returns `nav`; leave it untouched since five existing tests rely on its exact shape). Append after the module-level `_run` function:

```python
def _run_on_enter(url: str) -> tuple[list, bool]:
    scenario = (
        "fire(docListeners, 'DOMContentLoaded');"
        f"setHref('{url}');"
        "const entered = window.AdminTabs.onEnter();"
        "console.log(JSON.stringify({nav, entered}));"
    )
    script = "\n".join([_STUB, ADMIN_TABS_JS, scenario])
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    return payload["nav"], payload["entered"]
```

Then append two new tests:

```python
def test_on_enter_returns_true_when_it_schedules_the_redirect():
    nav, entered = _run_on_enter("https://atl.example/app?view=admin")
    assert nav == [["replace", "/admin-analytics"]]
    assert entered is True


def test_on_enter_returns_false_on_the_providers_tab():
    nav, entered = _run_on_enter("https://atl.example/app?view=admin&adminTab=providers")
    assert nav == []
    assert entered is False
```

In `dashboard/backend/tests/test_admin_analytics_frontend.py`, change the two pinned versions inside `test_app_lifecycle_and_cache_versions_are_wired`:

```python
    assert 'app.js?v=131' in APP_HTML
```
→
```python
    assert 'app.js?v=132' in APP_HTML
```

and

```python
    assert 'js/admin-tabs.js?v=7' in APP_HTML
```
→
```python
    assert 'js/admin-tabs.js?v=8' in APP_HTML
```

Then append a new static-shape test to the same file (it already imports `fn_body` from `_frontend_source`, per its existing `from dashboard.backend.tests._frontend_source import ...` block):

```python
def test_admin_branch_checks_admin_tabs_on_enter_before_loading_stats():
    body = fn_body("function navigateToPage(")
    admin_branch_start = body.index("page === 'admin'")
    on_enter_index = body.index("window.AdminTabs.onEnter()", admin_branch_start)
    load_stats_index = body.index("loadAdminStats()", admin_branch_start)
    assert on_enter_index < load_stats_index
    assert "return" in body[admin_branch_start:load_stats_index]
```

- [ ] **Step 2: Run them and confirm the expected failures**

```bash
python -m pytest dashboard/backend/tests/test_admin_tabs_redirect.py -v
python -m pytest dashboard/backend/tests/test_admin_analytics_frontend.py::test_app_lifecycle_and_cache_versions_are_wired dashboard/backend/tests/test_admin_analytics_frontend.py::test_admin_branch_checks_admin_tabs_on_enter_before_loading_stats -v
```

Expected: **FAIL**.
- `test_on_enter_returns_true_when_it_schedules_the_redirect`: `entered` is `None` (JS `undefined` serializes to `null`), not `True` — `onEnter()` currently returns nothing.
- `test_on_enter_returns_false_on_the_providers_tab`: same — `entered` is `None`, not `False`.
- `test_app_lifecycle_and_cache_versions_are_wired`: fails at the (not-yet-edited) `assert 'app.js?v=132' in APP_HTML`, since `app.html` still says `?v=131`.
- `test_admin_branch_checks_admin_tabs_on_enter_before_loading_stats`: fails at `assert "return" in body[...]` — today's admin branch has no `return` between the `onEnter()` call and `loadAdminStats()`.

- [ ] **Step 3: Edit `admin-tabs.js`**

Change:

```js
  function setTab(value, { updateUrl = true, leave = true } = {}) {
    const tab = normalizeTab(value);
    if (tab === 'analytics' && leave) {
      // The Analytics tab now lives on the standalone /admin-analytics page
      // (preview with synthetic data). Leave the in-app panel in place but
      // never show it; providers/users/activity still render here.
      // `replace`, not `assign`: `/app?view=admin` must not stay in history,
      // or Back reloads it, app.js routes to admin, and we redirect forward
      // again — a loop the user cannot escape with the Back button.
      window.location.replace('/admin-analytics');
      return tab;
    }
```

to:

```js
  function setTab(value, { updateUrl = true, leave = true } = {}) {
    const tab = normalizeTab(value);
    if (tab === 'analytics' && leave) {
      // The Analytics tab now lives on the standalone /admin-analytics page
      // (preview with synthetic data). Leave the in-app panel in place but
      // never show it; providers/users/activity still render here.
      // `replace`, not `assign`: `/app?view=admin` must not stay in history,
      // or Back reloads it, app.js routes to admin, and we redirect forward
      // again — a loop the user cannot escape with the Back button.
      // Returns `true` (not the tab string every other path returns) so
      // `onEnter` can report a navigation was scheduled and its caller can
      // stop running the loaders that page is already leaving (PR 0).
      window.location.replace('/admin-analytics');
      return true;
    }
```

Change:

```js
  function onEnter() {
    bind();
    const requested = new URL(window.location.href).searchParams.get('adminTab');
    setTab(requested || DEFAULT_TAB);
  }
```

to:

```js
  function onEnter() {
    bind();
    const requested = new URL(window.location.href).searchParams.get('adminTab');
    return setTab(requested || DEFAULT_TAB) === true;
  }
```

- [ ] **Step 4: Edit `app.js`**

Change:

```js
        } else if (page === 'admin') {
            currentMode = 'admin';
            if (adminView) adminView.style.display = 'block';
            // Stats load on entry and on explicit refresh — not on every
            // pager click, which only changes the user page.
            loadAdminStats();
            loadAdminUsers();
            if (window.AdminTabs) {
                window.AdminTabs.onEnter();
            }
            if (window.AdminAnalytics) {
                window.AdminAnalytics.onEnter();
            }
            if (window.AdminAnalyticsValue) {
                window.AdminAnalyticsValue.onEnter();
            }
            if (window.AdminModelProviders) {
                window.AdminModelProviders.onEnter();
            }
            if (window.AdminCredits) {
                window.AdminCredits.onEnter();
            }
        }
```

to:

```js
        } else if (page === 'admin') {
            currentMode = 'admin';
            if (adminView) adminView.style.display = 'block';
            // AdminTabs.onEnter() runs first: window.location.replace() only
            // *schedules* the /admin-analytics redirect, so without this
            // early return every onEnter() below (plus the stats/user-list
            // loads) still ran, in the same tick, into a page that was
            // already leaving (PR 0 -- kills the "phantom fetch").
            if (window.AdminTabs && window.AdminTabs.onEnter()) {
                return;
            }
            // Stats load on entry and on explicit refresh — not on every
            // pager click, which only changes the user page.
            loadAdminStats();
            loadAdminUsers();
            if (window.AdminAnalytics) {
                window.AdminAnalytics.onEnter();
            }
            if (window.AdminAnalyticsValue) {
                window.AdminAnalyticsValue.onEnter();
            }
            if (window.AdminModelProviders) {
                window.AdminModelProviders.onEnter();
            }
            if (window.AdminCredits) {
                window.AdminCredits.onEnter();
            }
        }
```

The early `return` exits `navigateToPage` entirely, skipping the trailing history/analytics bookkeeping below the whole `if/else if` chain (`clearNavBootState(); persistNavigation(); window.ATLAnalytics?.recordNavigation(...); syncNavigationHistory(...)`). That is deliberate, not incidental: `window.location.replace()` is about to tear down this document for a different HTML file, not perform an SPA-internal transition, so recording SPA nav state for a page that is already gone is exactly the kind of work into-a-departing-page this task removes.

- [ ] **Step 5: Bump the cache-busters in `app.html`**

Change line 2701 from `<script src="app.js?v=131" defer></script>` to `<script src="app.js?v=132" defer></script>`.

Change line 2710 from `<script src="js/admin-tabs.js?v=7" defer></script>` to `<script src="js/admin-tabs.js?v=8" defer></script>`.

- [ ] **Step 6: Run everything and confirm the pass**

```bash
python -m pytest dashboard/backend/tests/test_admin_tabs_redirect.py -v
python -m pytest dashboard/backend/tests/test_admin_analytics_frontend.py -v
```

Expected: **PASS** on every test in both files, including the pre-existing `test_page_load_on_a_non_admin_view_does_not_redirect`, `test_popstate_does_not_redirect`, `test_entering_admin_view_redirects_to_the_analytics_page`, `test_entering_admin_view_on_another_tab_stays_in_the_console`, and `test_redirect_replaces_history_so_back_does_not_loop` (none of them inspect `setTab`'s or `onEnter`'s return value, only `nav`, so the boolean-return change does not affect them), and the full set of assertions inside `test_app_lifecycle_and_cache_versions_are_wired`.

- [ ] **Step 7: Commit**

```bash
git add dashboard/frontend/js/admin-tabs.js \
        dashboard/frontend/app.js \
        dashboard/frontend/app.html \
        dashboard/backend/tests/test_admin_tabs_redirect.py \
        dashboard/backend/tests/test_admin_analytics_frontend.py
git commit -m "$(cat <<'EOF'
fix: stop the admin view's analytics fetches from firing into a departing page

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

## Task 5: `api/auth.py` — map a user-store outage to a logged 503

**Files:**
- Modify: `dashboard/backend/api/auth.py` — imports (lines 1-15), new helper block before `get_current_user` (before line 407), `get_current_user` (lines 407-417), `signup` (lines 505-514), `login` (lines 567-570), `me` (lines 607-611)
- Test: `dashboard/backend/tests/test_auth_store_unavailable.py` (create)

**Interfaces:**
- Consumes: nothing new.
- Produces: `auth._USER_STORE_OUTAGE: tuple[type[BaseException], ...]` and `auth._store_unavailable(exc: BaseException, *, route: str) -> HTTPException` in `dashboard.backend.api.auth`.

**Design choice, and why.** A small helper invoked from an inline `try/except` at each of the four call sites, not a single app-level exception handler. FastAPI/Starlette exception handlers attach only to the top-level ASGI `app` (`app.add_exception_handler`), never to an `APIRouter` — registering one would mean the mapping's actual logic lives in `app.py`, while the design doc names only `api/auth.py` (§4.6, §13). The four call sites are also not uniform: `get_current_user` is a dependency resolved before every gated route runs, `signup`/`login` run their store calls inside `asyncio.to_thread`, and `me` has two sequential store reads with a fallback between them — a single generic handler cannot name which call failed, but wrapping each site can, which is what makes `route=<name>` in the log line an actual fact rather than a guess. The helper **returns** the `HTTPException` rather than raising it, and the caller does the `raise` — this repository already has this exact convention for exactly this reason: `reset_password`'s `_failure()` helper (`auth.py`, around line 1352) carries the comment "Built and returned, never raised here (a raise-helper trips py/mixed-returns)", a live CodeQL check in this repo (`.github/workflows/codeql.yml`). This carries forward the superseded 2026-09-12 stopgap plan's Task A4 (`_USER_STORE_OUTAGE` tuple, the same returns-not-raises helper shape) at this file's current line numbers, and extends it to `me`, which that task never named — the design doc's PR 0 row explicitly lists `get_current_user`, `login`, `signup`, **and** `me`.

`psycopg_pool.PoolTimeout` is not listed as a third tuple member: it subclasses `psycopg.OperationalError` (verified directly: `psycopg_pool.PoolTimeout.__mro__` includes `psycopg.OperationalError`), so the two-member tuple already covers it, exactly as the 09-12 plan's own `test_pool_timeout_is_covered_by_the_operational_error_arm` proved. `psycopg` is imported unconditionally (no `try/except ImportError` guard, unlike the 09-12 draft): `requirements.txt` pins `psycopg[binary,pool]==3.3.4` as a hard, unconditional dependency, and `dashboard/backend/db_pool.py`/`users_postgres.py` already import it the same way at module scope.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/test_auth_store_unavailable.py`:

```python
"""api/auth.py maps a user-store connectivity failure to 503, not a bare 500.

The September 2026 outage's failure mode: the users/content Postgres project
returned errors for ~5 hours (Neon free-tier egress exhausted by analytics
maintenance -- see the burner-kill plan's Tasks 1-3), and every auth call --
signup, login, the /me boot probe, and get_current_user's own dependency
lookup -- propagated a raw psycopg.OperationalError (or, on the SQLite twin,
sqlite3.OperationalError) as an unmapped 500 with no server-side log line.
This pins the fix at all four call sites: each maps the failure to a 503 the
caller can retry, and prints one line naming the route and the exception
category.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg_pool import PoolTimeout

from dashboard.backend.app import app
from dashboard.backend.users import UserStore


@pytest.fixture
def temp_user_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = UserStore(db_path=Path(tmpdir) / "auth_unavailable_test.db")
        yield store


@pytest.fixture
def client(temp_user_store, monkeypatch):
    from dashboard.backend import users
    from dashboard.backend.api import auth
    from dashboard.backend.domain.credits.repository import CreditsStore
    from dashboard.backend.domain.credits.service import CreditsService

    monkeypatch.setattr(users, "user_store", temp_user_store)
    monkeypatch.setattr(
        auth,
        "credits_service",
        CreditsService(store=CreditsStore(temp_user_store.db_path)),
    )
    return TestClient(app)


def _raiser(exc):
    def _raise(*args, **kwargs):
        raise exc

    return _raise


def test_pool_timeout_is_covered_by_the_operational_error_arm():
    """One arm, because PoolTimeout is an OperationalError subclass."""
    from dashboard.backend.api import auth

    assert issubclass(PoolTimeout, psycopg.OperationalError)
    assert psycopg.OperationalError in auth._USER_STORE_OUTAGE
    assert sqlite3.OperationalError in auth._USER_STORE_OUTAGE


def test_get_current_user_maps_a_sqlite_operational_error(client, temp_user_store, monkeypatch, capsys):
    signup = client.post(
        "/api/auth/signup",
        json={"email": "gcu@example.com", "display_name": "GCU", "password": "securepass1"},
    )
    assert signup.status_code == 200

    monkeypatch.setattr(
        temp_user_store,
        "get_user_for_token",
        _raiser(sqlite3.OperationalError("database is locked")),
    )

    response = client.get("/api/auth/me")

    assert response.status_code == 503
    assert response.json()["detail"] == "Service temporarily unavailable"
    output = capsys.readouterr().out
    assert (
        "ERROR: auth.user_store_unavailable route=get_current_user category=OperationalError"
        in output
    )


def test_login_maps_a_psycopg_operational_error(client, temp_user_store, monkeypatch, capsys):
    monkeypatch.setattr(
        temp_user_store,
        "authenticate",
        _raiser(psycopg.OperationalError("connection refused")),
    )

    response = client.post(
        "/api/auth/login",
        json={"email": "nobody@example.com", "password": "whatever1"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Service temporarily unavailable"
    output = capsys.readouterr().out
    assert "ERROR: auth.user_store_unavailable route=login category=OperationalError" in output


def test_signup_maps_a_pool_timeout(client, temp_user_store, monkeypatch, capsys):
    monkeypatch.setattr(
        temp_user_store,
        "create_user",
        _raiser(PoolTimeout("couldn't get a connection")),
    )

    response = client.post(
        "/api/auth/signup",
        json={
            "email": "signup-pool@example.com",
            "display_name": "Signup Pool",
            "password": "securepass1",
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Service temporarily unavailable"
    output = capsys.readouterr().out
    assert "ERROR: auth.user_store_unavailable route=signup category=PoolTimeout" in output


def test_me_maps_a_sqlite_operational_error_from_get_entitlements(client, temp_user_store, monkeypatch, capsys):
    from dashboard.backend import users as users_module

    signup = client.post(
        "/api/auth/signup",
        json={
            "email": "me-entitlements@example.com",
            "display_name": "Me Entitlements",
            "password": "securepass1",
        },
    )
    assert signup.status_code == 200

    monkeypatch.setattr(users_module, "entitlements_from_session_row", lambda *a, **k: None)
    monkeypatch.setattr(
        temp_user_store,
        "get_entitlements",
        _raiser(sqlite3.OperationalError("database is locked")),
    )

    response = client.get("/api/auth/me")

    assert response.status_code == 503
    assert response.json()["detail"] == "Service temporarily unavailable"
    output = capsys.readouterr().out
    assert "ERROR: auth.user_store_unavailable route=me category=OperationalError" in output


def test_health_is_unaffected_by_a_user_store_outage(client, temp_user_store, monkeypatch):
    monkeypatch.setattr(
        temp_user_store,
        "get_user_for_token",
        _raiser(sqlite3.OperationalError("database is locked")),
    )

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 2: Run them and confirm the expected failure**

```bash
python -m pytest dashboard/backend/tests/test_auth_store_unavailable.py -v
```

Expected: **FAIL/ERROR** on every case except `test_health_is_unaffected_by_a_user_store_outage` (which passes today, since `/api/health` never touches `user_store`).
- `test_pool_timeout_is_covered_by_the_operational_error_arm`: `AttributeError: module 'dashboard.backend.api.auth' has no attribute '_USER_STORE_OUTAGE'`.
- The other four: FastAPI's `TestClient` defaults to `raise_server_exceptions=True`, so an exception that escapes a route handler unhandled is **re-raised in the test process**, not turned into a 500 response — confirmed directly (`TestClient(app).get(...)` on a handler that raises `RuntimeError` re-raises `RuntimeError` to the caller). So these four report as pytest **errors** (an uncaught `sqlite3.OperationalError` / `psycopg.OperationalError` / `PoolTimeout` propagating out of `client.get(...)`/`client.post(...)`), not as a clean `assert response.status_code == 503` failure.

- [ ] **Step 3: Add the imports**

In `dashboard/backend/api/auth.py`, change:

```python
import asyncio
import base64
import hmac
import ipaddress
import logging
import math
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urlencode

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request
```

to:

```python
import asyncio
import base64
import hmac
import ipaddress
import logging
import math
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urlencode

import psycopg
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request
```

- [ ] **Step 4: Add the helper, and wrap `get_current_user`**

Change:

```python
def get_current_user(
    request: Request,
    authorization: Optional[str] = Header(default=None),
) -> dict:
    token = _session_token(request, authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = users_module.user_store.get_user_for_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return user
```

to:

```python
# Exception classes that mean "the account database is unreachable" -- a down
# Postgres primary, an exhausted connection pool, or a locked local SQLite
# file. psycopg_pool.PoolTimeout is not listed separately: it subclasses
# psycopg.OperationalError (pinned by test_pool_timeout_is_covered_by_the_
# operational_error_arm in test_auth_store_unavailable.py), so one arm
# already covers a pool checkout that timed out, a refused connection, and a
# quota rejection alike.
_USER_STORE_OUTAGE: tuple[type[BaseException], ...] = (
    sqlite3.OperationalError,
    psycopg.OperationalError,
)
_STORE_UNAVAILABLE_DETAIL = "Service temporarily unavailable"


def _store_unavailable(exc: BaseException, *, route: str) -> HTTPException:
    """Build the 503 for a user-store outage; the caller owns the ``raise``.

    Built and returned, never raised here: a helper that raises trips
    CodeQL's py/mixed-returns the same way reset_password's ``_failure``
    below does, and that guard is why this repository already avoids the
    pattern. The log line carries only the exception's class -- a psycopg
    error's string form embeds the connection DSN.
    """
    print(
        "ERROR: auth.user_store_unavailable "
        f"route={route} category={type(exc).__name__[:80]}"
    )
    return HTTPException(status_code=503, detail=_STORE_UNAVAILABLE_DETAIL)


def get_current_user(
    request: Request,
    authorization: Optional[str] = Header(default=None),
) -> dict:
    token = _session_token(request, authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        user = users_module.user_store.get_user_for_token(token)
    except _USER_STORE_OUTAGE as exc:
        raise _store_unavailable(exc, route="get_current_user") from None
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return user
```

- [ ] **Step 5: Wrap `signup`**

Change:

```python
    try:
        # Threaded for the same reason as login's authenticate(): create_user
        # hashes with bcrypt (~190 ms), and this route is unauthenticated.
        user = await asyncio.to_thread(
            users_module.user_store.create_user,
            email=payload.email,
            display_name=payload.display_name,
            password=payload.password,
        )
    except ValueError as exc:
        if str(exc) == "email_already_registered":
```

to:

```python
    try:
        # Threaded for the same reason as login's authenticate(): create_user
        # hashes with bcrypt (~190 ms), and this route is unauthenticated.
        user = await asyncio.to_thread(
            users_module.user_store.create_user,
            email=payload.email,
            display_name=payload.display_name,
            password=payload.password,
        )
    except _USER_STORE_OUTAGE as exc:
        raise _store_unavailable(exc, route="signup") from None
    except ValueError as exc:
        if str(exc) == "email_already_registered":
```

- [ ] **Step 6: Wrap `login`** (today it has no `try` around `authenticate` at all)

Change:

```python
    user = await asyncio.to_thread(
        users_module.user_store.authenticate, payload.email, payload.password
    )
    if not user:
```

to:

```python
    try:
        user = await asyncio.to_thread(
            users_module.user_store.authenticate, payload.email, payload.password
        )
    except _USER_STORE_OUTAGE as exc:
        raise _store_unavailable(exc, route="login") from None
    if not user:
```

- [ ] **Step 7: Wrap `me`**

Change:

```python
    entitlements = users_module.entitlements_from_session_row(
        current_user, current_user["id"]
    )
    if entitlements is None:
        entitlements = users_module.user_store.get_entitlements(current_user["id"])
```

to:

```python
    try:
        entitlements = users_module.entitlements_from_session_row(
            current_user, current_user["id"]
        )
        if entitlements is None:
            entitlements = users_module.user_store.get_entitlements(current_user["id"])
    except _USER_STORE_OUTAGE as exc:
        raise _store_unavailable(exc, route="me") from None
```

- [ ] **Step 8: Run everything and confirm the pass**

```bash
python -m pytest dashboard/backend/tests/test_auth_store_unavailable.py -v
python -m pytest dashboard/backend/tests/test_auth.py dashboard/backend/tests/test_auth_cache.py -v
```

Expected: **PASS** on every case in the new file, and no regression in `test_auth.py`/`test_auth_cache.py` (neither monkeypatches a store method to raise, so the new `try/except` arms are simply never taken on the happy paths those files already cover).

- [ ] **Step 9: Commit**

```bash
git add dashboard/backend/api/auth.py dashboard/backend/tests/test_auth_store_unavailable.py
git commit -m "$(cat <<'EOF'
fix: map a user-store outage to 503 on the auth routes

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

## Task 6: `CLAUDE.md` — record the reaper-cadence gotcha

**Files:**
- Modify: `CLAUDE.md` — append one bullet to the end of the `## Gotchas` section (currently ends at line 303, the last line of the file)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing consumed by later tasks; this is documentation only.

- [ ] **Step 1: Append the bullet**

At the end of `CLAUDE.md` (after the final existing bullet, "**User accounts were silently lost on every prod redeploy until 2026-07...**"), append:

```markdown
- **Anything registered with `register_reaper_sweep` runs on the same 60-second tick** (`RUN_REAPER_INTERVAL_SECONDS`, `domain/runs/service.py`) — there is no per-sweep interval, so a sweep that does per-user work multiplies its cost by every tick, not by however slow that work actually is. The analytics snapshot-repair sweep registered there (`domain/analytics/maintenance.py::run_analytics_maintenance` → `states.py::repair_stale_snapshots`) was one of two September 2026 outage burners for exactly this reason; it is throttled to a 24-hour staleness window by PR 0 and deleted outright by PR A once `user_daily_facts` replaces `user_analytics_snapshots`. The admin layer's full architecture — the analytics data model, the daily job, the nine `/api/admin/analytics` endpoints, and the `/admin` page — is designed at `docs/superpowers/specs/2026-09-15-admin-layer-redesign-design.md`; read it before touching anything under `domain/analytics/` or the admin frontend.
```

This is documentation with no test to write against it; there is no Step 2/3 TDD cycle for this task.

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs: record the reaper-cadence gotcha and point to the admin layer design

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

## Not in this PR

Per design doc §13 (PR 0's "Must not" column) and D21, the following pieces of the superseded 2026-09-12 stopgap plan are explicitly **not** built here:

- **A shared stale-window constant.** The 09-12 plan's Task A1/A2 shape would have introduced one module-level `timedelta` constant reused by both the legacy and value snapshot repair paths. This plan changes only `repair_stale_snapshots`'s own default (Task 1); `repair_stale_value_snapshots` (a different function, with its own `include_time_transitions` staleness logic) is untouched, because both functions and the tables they repair are deleted wholesale by PR A — sharing a constant between two code paths PR A deletes buys nothing.
- **A one-read recompute.** The 09-12 plan's Task A2 area described narrowing `_calculate_user_value_snapshot`'s per-recompute event read from two full-history scans to one. PR A replaces the recompute mechanism entirely (read-time lifecycle over `user_daily_facts`, §6.6 of the design doc) rather than shrinking it, so optimizing the doomed version is design-doc-forbidden scaffolding for a code path PR A deletes.
- **A spy-store sweep-read-budget test.** The 09-12 plan's Task A5 pinned the reaper tick's *query count* against a synthetic population, as a regression guard for the exact mechanism this plan throttles. PR A's own `test_read_budget.py` budget (design doc §6.12) supersedes it with a stronger, user-count-independence guarantee once the daily job replaces per-tick per-user repair; a budget test against a mechanism about to be deleted is that same forbidden scaffolding.
- **The 09-12 plan's Task A4, unmodified.** Task 5 above carries A4's shape forward (the `_USER_STORE_OUTAGE` tuple, the returns-not-raises helper) but extends it to `me`, which A4 never named, and drops A4's defensive `try/except ImportError` around importing `psycopg` (confirmed a hard, unconditional dependency in `requirements.txt`).

## Acceptance

- [ ] `repair_stale_snapshots`'s default staleness window is 24 hours, pinned by a test proving a 23-hour-old snapshot is skipped and a 25-hour-old one is repaired (Task 1).
- [ ] The live `analytics_service` singleton builds with `project_snapshots=False`, and a lifecycle event recorded through it does not call `states.recalculate_user_snapshots` (Task 2).
- [ ] `instrumentation.py`'s own fallback recalculator is disarmed at boot for the production singleton, so Task 2's flag flip is not merely relocating the same per-request recompute (Task 3).
- [ ] Entering the admin view no longer fires `loadAdminStats`, `loadAdminUsers`, `AdminAnalytics.onEnter`, `AdminAnalyticsValue.onEnter`, `AdminModelProviders.onEnter`, or `AdminCredits.onEnter` when `AdminTabs.onEnter()` has scheduled the `/admin-analytics` redirect; entering on a non-`analytics` tab is unaffected (Task 4).
- [ ] `app.js`/`admin-tabs.js`'s cache-busters are bumped and every pin in `test_app_lifecycle_and_cache_versions_are_wired` matches (Task 4).
- [ ] `get_current_user`, `signup`, `login`, and `me` each map a `sqlite3.OperationalError`/`psycopg.OperationalError`/`psycopg_pool.PoolTimeout` from the user store to a 503 with the exact log line `ERROR: auth.user_store_unavailable route=<name> category=<ExceptionClass>`, and `/api/health` is unaffected (Task 5).
- [ ] `CLAUDE.md` documents the 60-second reaper-tick cadence rule and points to the admin layer design doc (Task 6).
- [ ] No table, response model, or frontend surface outside `admin-tabs.js`/`app.js` changed shape.
- [ ] No item from "Not in this PR" was built.

## Self-review notes

- **Spec coverage.** Every named section (§2.2, §4.2, §4.3, §4.6, §13's PR 0 row, D21) is implemented by at least one task above; the PR 0 row's `service.py` + `instrumentation.py` clause spans Tasks 2 and 3, and Task 3's header quotes that clause verbatim so a reader can check it is spec coverage rather than an addition. The "Not in this PR" section maps each forbidden 09-12 item to the design-doc reason it was cut (D21, §6.9, §6.12).
- **Placeholder scan.** Every code block in every step is the literal file content before or after the change, read directly from the worktree at `c3bbf2ed` (the commit `git log -1` reported while writing this plan) — no `TBD`, no "similar to Task N", no elided function body. Every function referenced (`repair_stale_snapshots`, `_build_analytics_service`, `_try_recalculate_snapshots`, `_emit`, `_recalculate_snapshot`, `register_snapshot_recalculator`, `get_current_user`, `signup`, `login`, `me`, `setTab`, `onEnter`, `navigateToPage`) exists in the current source or is defined earlier in this same plan.
- **Name consistency.** `disable_synchronous_projection` (Task 3), `_USER_STORE_OUTAGE`/`_store_unavailable` (Task 5), and the `entered`/boolean-return contract on `AdminTabs.onEnter()` (Task 4) are each introduced once and referenced identically everywhere else they appear, including across tasks (Task 2's comment references Task 3 by name; Task 5's design-choice paragraph references the exact `reset_password`/`_failure` line this repo already uses for the same CodeQL constraint).
- **Verified, not assumed:** the `TestClient(app)` re-raise behavior on an unhandled exception (Task 5, Step 2) was confirmed by running a minimal FastAPI app under the installed `fastapi`/`starlette` versions in this environment, not inferred from the 09-12 plan's (differently-worded) expectation. The `psycopg_pool.PoolTimeout` subclass relationship (Task 5) was confirmed the same way (`PoolTimeout.__mro__`). The absence of any existing test for `repair_stale_snapshots` (Task 1) and of any production caller of `register_snapshot_recalculator` (Task 3) were both confirmed by exhaustive `rg` searches, not inferred from the design doc's prose.
- **Task 3's mechanism is this plan's choice; its requirement is the design doc's.** The PR 0 row names the outcome ("fallback recompute disarmed") without naming how. The 09-12 plan's how — gating `_recalculate_snapshot` behind the shared stale-window guard — is forbidden by PR 0's "Must not" column, so this plan registers a no-op recalculator at boot instead. That keeps `_emit`'s guard, `_recalculate_snapshot`, and `register_snapshot_recalculator` textually intact for PR B to delete (design doc §6.15 item 1: "PR 0 switches it off, including the `instrumentation.py` fallback; PR B deletes it"), which is why the change is confined to one new function and one `app.py` call. No design-doc edit is needed; this plan does not touch the spec file.
