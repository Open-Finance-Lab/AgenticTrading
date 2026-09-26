# LLM Backtest Step Latency — S1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the two pure-waste costs in every ATL Credits model call: the ~31s synchronous analytics rebuild and the ~3s dead OpenRouter attempt. Also spend CommonStack's prepaid balance first, and make its exhaustion visible.

**Architecture:** Two PRs, shipped in order so one prod run can attribute each saving.
- **PR 1 (S1a)** makes an unregistered analytics snapshot recalculator inert. Today it falls back to a full `states.recalculate_user_snapshots` per event, and the backtest child never registers the web process's no-op.
- **PR 2 (S1b)** is the existing draft **#535** (`feat/commonstack-first`), which already ships Haiku 4.5 on CommonStack. This plan completes it:
  - an env-driven platform provider order, CommonStack first by default;
  - a once-per-process operator ERROR line on platform quota exhaustion;
  - neutral billing-hint copy;
  - docs.

**Tech Stack:** Python 3 / FastAPI backend (`dashboard.backend` package), pytest, SQLite locally and in tests, a Postgres twin exercised in CI through `TEST_POSTGRES_URL`, and a vanilla-JS frontend (`dashboard/frontend/app.js`, no build step).

**Spec:** `docs/superpowers/specs/2026-09-23-llm-backtest-step-latency-design.md`, amended 2026-09-26. Read §4 (S1a), §5 (S1b), §6 (failure behaviour) and §8 (rollout) before starting.

> **S1b is superseded by #535's `ab8919b6` (2026-09-26). Do not re-execute its tasks from this plan.** That commit changed four things this plan's S1b tasks spell out. The spec's §5 and §7 now describe the shipped behaviour; where a task's code block below disagrees with them, the spec wins.
> - An explicit `provider_id` is honoured only when it is in the configured order.
> - Only platform-only lanes move in `list_execution_options`.
> - The legacy `("openrouter",)` expansion is deleted.
> - The billing hint is `ATL Credits cover the model calls. ATL picks an available provider automatically.`

## Global Constraints

- Run everything from the repo root of the relevant worktree. Import by full package path (`dashboard.backend...`), never by file path.
- Test command: `python3 -m pytest <path> -q -p no:cacheprovider`. The whole suite is `python3 -m pytest dashboard/backend/tests/ -q -p no:cacheprovider --timeout=180`. Install `pytest-timeout` first if it is missing, or drop the flag.
- `node` must be on `PATH` for the whole-suite run. Without it the frontend-harness tests skip instead of running.
- Never run ad-hoc `python3 -c` imports of backend modules. They mutate the tracked seed DB `dashboard/storage/data/backtest.db`. pytest is safe, because `tests/conftest.py` redirects `DATABASE_PATH`.
- Never print a secret. The Haiku probe (Task 6) reads `COMMONSTACK_API_KEY` from the main checkout's `dashboard/.env` into a shell variable and must never echo it.
- Operator log lines use the repo's `print("ERROR: …")` / `print("WARNING: …")` convention. They carry no exception text and no identifiers beyond provider ids.
- The quota ERROR line is exactly `ERROR: llm.platform_quota_exhausted provider=<id> fallback=<next id|none>`.
- The env var is exactly `ATL_PLATFORM_PROVIDER_ORDER`, default `commonstack,openrouter`.
- The billing-hint copy is exactly `ATL Credits cover the model calls. ATL picks an available provider automatically.` (was the CommonStack/OpenRouter "switch between" copy until `ab8919b6`).
- The legacy `("openrouter",)` expansion in `infrastructure/llm/execution/service.py` is deleted (`ab8919b6`). The route's candidate tuple is authoritative.
- Do not touch `PIPELINE_SECONDS_PER_LLM_CALL`. Recalibrating it is S4.
- Do not edit user-facing docs (`app.html` copy other than the cache-buster, `strategy.html`, `docs/source/**`). They are listed as follow-ups in spec §9.
- Commits end with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Commit locally as each task completes. **Pushing, opening PRs, editing PR #535's body and marking anything ready all need the user's explicit go-ahead.**
- Before every commit, run `git branch --show-current`. Parallel sessions share this checkout family and have switched branches mid-task before.

## Review Focus

1. **BYOK provider order must not move.** Re-sorting `list_execution_options` for the platform lanes could reorder BYOK-only providers too. `app.js` picks `providers[0]` as the default BYOK provider, so a user's preselected provider would silently change. Expected: the relative order of every provider outside `ATL_PLATFORM_PROVIDER_ORDER` is unchanged. Pinned in Task 2 (`test_execution_options_follow_platform_order_and_keep_byok_order`).
2. **A typo'd `ATL_PLATFORM_PROVIDER_ORDER` must not half-apply.** Values like `commonstack;openrouter`, `Common Stack` or `commonstack,open-router` would otherwise drop a lane. Expected: the whole value is rejected, the default is used, and exactly one `WARNING` is printed per distinct bad value. Pinned in Task 2.
3. **An order that leaves nothing routable must fail visibly.** Example: `ATL_PLATFORM_PROVIDER_ORDER=anthropic`. Expected: no candidates, so the route's existing 422 fires. It must not quietly pick some provider. Pinned in Task 2 (`resolve_platform_execution_candidates` returns `()`).
4. **Concurrent calls in one process must not race the once-per-provider guard.** Expected: exactly one ERROR line even when several threads hit a drained provider at once. Pinned in Task 3 (thread test on the helper).
5. **A killed child must still have printed the ERROR line.** The parent kills the child at the 3600s timeout, and piped stdout is block-buffered, so an unflushed line dies with the process. Expected: the line is flushed as it is printed. Pinned in Task 3 (source check for `flush=True`).

---

## PR 1 — S1a: the unregistered recalculator is inert

**Worktree:** `/mnt/c/Users/27740/OneDrive/Documents/Github/AgenticTrading/.claude/worktrees/522-llm-step-latency`. It is on branch `docs/522-llm-step-latency` at `origin/main` (`e1871bae`), and already holds the spec and this plan as a docs commit. The PR carries the spec, this plan and the S1a fix.

### Task 1: Make `_recalculate_snapshot` inert when nothing is registered

**Files:**
- Modify: `dashboard/backend/domain/analytics/instrumentation.py:56-98` (`_snapshot_recalculator` comment, `disable_synchronous_projection` docstring, `_recalculate_snapshot`)
- Modify: `dashboard/backend/app.py:337-342` (comment above `disable_synchronous_projection()`)
- Modify: `CLAUDE.md` (Admin layer section, one appended paragraph)
- Test: `dashboard/backend/tests/domain/analytics/test_instrumentation.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `instrumentation._recalculate_snapshot(user_id: int, event_name: str) -> None` now returns without effect when `_snapshot_recalculator is None`. The signatures of `register_snapshot_recalculator(callback)` and `disable_synchronous_projection()` are unchanged.

- [ ] **Step 1: Rename the branch for the PR**

```bash
cd /mnt/c/Users/27740/OneDrive/Documents/Github/AgenticTrading/.claude/worktrees/522-llm-step-latency
git branch --show-current   # expect: docs/522-llm-step-latency
git branch -m fix/522-inert-snapshot-recalculator
```

- [ ] **Step 2: Write the failing test**

Append to `dashboard/backend/tests/domain/analytics/test_instrumentation.py`:

```python
def test_unregistered_recalculator_never_rebuilds_snapshots(monkeypatch):
    """The dashboard backtest child's exact state (#522).

    The child never runs app.py's startup, so nothing registers a
    recalculator, and the live analytics singleton does not project. Before
    #522 that combination fell back to states.recalculate_user_snapshots on
    every snapshot-relevant event -- ~5s each, ~31s of every model call.
    """
    calls = []
    monkeypatch.setattr(
        "dashboard.backend.domain.analytics.states.recalculate_user_snapshots",
        lambda *a, **k: calls.append((a, k)),
    )
    monkeypatch.setattr(instrumentation, "_snapshot_recalculator", None)

    class NonProjectingService:
        project_snapshots = False

        def try_record_server_event(self, **kwargs):
            return AppendEventResult.model_construct(event=None, created=True)

    monkeypatch.setattr(
        instrumentation, "get_analytics_service", lambda: NonProjectingService()
    )

    for event_name in ("credits_reserved", "credits_settled", "model_usage_recorded"):
        instrumentation.emit_resource_event(
            event_name=event_name,
            user_id=7,
            source_record_type="llm_reservation",
            source_record_id=f"reservation-{event_name}",
            properties={},
            occurred_at=NOW,
        )
    instrumentation.emit_agent_event(
        event_name="agent_created",
        user_id=7,
        agent_id="agent-1",
        occurred_at=NOW,
    )

    assert calls == []
```

- [ ] **Step 3: Run it and confirm it fails**

Run: `python3 -m pytest dashboard/backend/tests/domain/analytics/test_instrumentation.py::test_unregistered_recalculator_never_rebuilds_snapshots -q -p no:cacheprovider`

Expected: FAIL at `assert calls == []`, with four recorded calls. If it fails on an `emit_resource_event` validation error first, the fallback was never reached, so the test would be vacuous. Check the `properties` and `source_record_type` values against `emit_resource_event` (`instrumentation.py:262`), fix the arguments, and re-run until the failure is the `calls` assertion.

- [ ] **Step 4: Implement**

In `dashboard/backend/domain/analytics/instrumentation.py`, replace the line `_snapshot_recalculator: Any = None` with:

```python
# None means "no synchronous projection". That is the default every process
# gets, including a dashboard backtest child, which never runs app.py's
# startup hook. Before #522, None fell back to
# states.recalculate_user_snapshots: ~5s per event in the child, ~31s of every
# model call. The reaper's throttled repair and the daily job keep snapshots
# fresh instead, for worker-emitted events exactly as for web-emitted ones.
_snapshot_recalculator: Any = None
```

Replace the whole `disable_synchronous_projection` function with:

```python
def disable_synchronous_projection() -> None:
    """Register an explicit no-op snapshot recalculator.

    Since #522 an unregistered recalculator is already inert, so this call
    no longer carries the fix. PR 0 added it because ``_emit``'s guard below
    fires whenever the handling service does not project
    (``not getattr(service, "project_snapshots", False)``). The live
    singleton builds with ``project_snapshots=False``, and the old fallback
    then recomputed per accepted event. That default was fixed only in the
    process that ran this call, which left the backtest child paying for it.
    ``app.py`` still calls this so the web process states its intent
    explicitly rather than depending on the default.
    """
    register_snapshot_recalculator(lambda user_id: None)
```

Replace the whole `_recalculate_snapshot` function with:

```python
def _recalculate_snapshot(user_id: int, event_name: str) -> None:
    if event_name not in SNAPSHOT_RELEVANT_EVENTS:
        return
    callback = _snapshot_recalculator
    if callback is None:
        # Inert by default (#522). A default that is expensive until something
        # disables it is inherited by every entry point that skips app.py's
        # startup: the backtest child, CLI scripts, a future worker service.
        return
    callback(user_id)
```

- [ ] **Step 5: Update the stale test name and docstring**

In the same test file, rename `test_disable_synchronous_projection_makes_the_fallback_a_no_op` to `test_disable_synchronous_projection_registers_a_no_op`. Add this docstring as its first line:

```python
    """app.py's explicit opt-out still registers a callable no-op (#522 made
    the unregistered default inert, so this pins the call, not the fix)."""
```

Then add these two lines just before its final `assert calls == []`:

```python
    assert instrumentation._snapshot_recalculator is not None
    assert instrumentation._snapshot_recalculator(7) is None
```

- [ ] **Step 6: Update the `app.py` comment**

In `dashboard/backend/app.py`, replace the six-line comment that starts `# PR 0: analytics_service now builds with project_snapshots=False, so` (just above `disable_synchronous_projection()`) with:

```python
        # Since #522 an unregistered recalculator is already inert, so this
        # call no longer carries the fix: PR 0 put it here, which fixed the
        # web process and left the backtest child (which never runs this
        # hook) rebuilding snapshots on every event. It stays so this process
        # states its intent explicitly instead of relying on the default.
```

- [ ] **Step 7: Run the analytics tests**

Run: `python3 -m pytest dashboard/backend/tests/domain/analytics/ dashboard/backend/tests/test_analytics_integration.py -q -p no:cacheprovider`

Expected: all pass, including the new test.
- `test_stored_event_recalculates_snapshot_best_effort` and `test_analytics_integration.py` register their own callbacks, so they are unaffected.
- A failure here means some test relied on the implicit fallback. Register an explicit callback in that test with `monkeypatch.setattr(instrumentation, "_snapshot_recalculator", ...)`, as those two do. Never restore the fallback.

- [ ] **Step 8: CLAUDE.md**

In `CLAUDE.md`'s `### Admin layer (analytics, /admin)` section, find the paragraph that ends with `See the Persistence bullet above for the rule that replaced the third.` Append this paragraph right after it, separated by a blank line:

```markdown
PR 0 killed the synchronous recompute in the web process only. `instrumentation._recalculate_snapshot` fell back to `states.recalculate_user_snapshots` whenever no recalculator was registered, and only `app.py`'s startup registers one. So every dashboard backtest child kept rebuilding snapshots per event, at ~5 s each and ~31 s of every model call, until #522 made the unregistered default inert. **Do not restore that fallback.** A default that is expensive until something disables it is inherited by every entry point that skips `app.py`'s startup. `disable_synchronous_projection()` still runs at boot, but only as an explicit statement of intent. Design: `docs/superpowers/specs/2026-09-23-llm-backtest-step-latency-design.md` §4.
```

- [ ] **Step 9: Full suite**

Run: `python3 -m pytest dashboard/backend/tests/ -q -p no:cacheprovider --timeout=180`

Expected: 0 failed. Record the passed/skipped counts for the PR body. Then run `git status --short` and confirm `dashboard/storage/data/backtest.db` is **not** modified.

- [ ] **Step 10: Commit**

```bash
git branch --show-current   # expect: fix/522-inert-snapshot-recalculator
git add dashboard/backend/domain/analytics/instrumentation.py dashboard/backend/app.py \
        dashboard/backend/tests/domain/analytics/test_instrumentation.py CLAUDE.md
git commit -m "fix(analytics): make the unregistered snapshot recalculator inert

The backtest child never runs app.py's startup, so it never registered
PR 0's no-op and rebuilt analytics snapshots synchronously on every
event: ~5s each, ~31s of every model call (#522).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 11: Stop for the user**

Report the suite counts. Ask before pushing and opening the PR. Proposed title: `fix(analytics): make the unregistered snapshot recalculator inert`. Proposed body: 3–5 lines, `Refs #522` (no closing keyword, because #522 stays open until S4), plus the spec path and the prod verification query from spec §8.

---

## PR 2 — S1b: complete draft #535 (`feat/commonstack-first`)

**Worktree:** create one on the existing branch. Do not cut a new branch: #535 already carries the Haiku allowlist backfill (`commonstack-allowlist-v1`) with tests in both twins, and CI is green on it.

```bash
cd /mnt/c/Users/27740/OneDrive/Documents/Github/AgenticTrading
git fetch origin feat/commonstack-first
gh pr view 535 --json state,isDraft --jq '"\(.state) draft=\(.isDraft)"'   # expect: OPEN draft=true
git worktree add .claude/worktrees/commonstack-first -B feat/commonstack-first origin/feat/commonstack-first
cd .claude/worktrees/commonstack-first
git log --oneline -1   # expect: b589505c feat(llm): route Claude Haiku 4.5 through CommonStack (or later)
```

If `gh` says MERGED, stop and cut `feat/commonstack-first-2` from `origin/main` instead. Commits behind a merged PR orphan silently.

### Task 2: Env-driven platform provider order, CommonStack first

**Files:**
- Modify: `dashboard/backend/domain/model_providers/service.py`:
  - the `.repository_common` import block (~`:43-51`);
  - new module-level helper after `_environment_platform_secret` (~`:91-99`);
  - the `list_execution_options` sort (~`:213-215`);
  - the `resolve_platform_execution_candidates` loop (~`:234`).
- Modify: `dashboard/backend/tests/conftest.py` (strip the new env var)
- Test: `dashboard/backend/tests/domain/model_providers/test_execution_catalog.py:107-122`
- Test: `dashboard/backend/tests/domain/model_providers/test_service.py:474-489`

**Interfaces:**
- Consumes: #535's `COMMONSTACK_MODEL_ALLOWLIST`, which includes `anthropic/claude-haiku-4-5`.
- Produces:
  - `DEFAULT_PLATFORM_PROVIDER_ORDER: tuple[str, ...] = ("commonstack", "openrouter")`
  - `platform_provider_order() -> tuple[str, ...]`
  - `_warned_platform_provider_orders: set[str]`, which tests reset.

  All three live in `dashboard.backend.domain.model_providers.service`.

- [ ] **Step 1: Write the failing tests**

In `dashboard/backend/tests/domain/model_providers/test_execution_catalog.py`, replace the whole `test_platform_candidates_prefer_openrouter_and_support_commonstack_only` function with:

```python
def test_platform_candidates_prefer_commonstack_and_follow_the_env_order(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("ATL_PLATFORM_PROVIDER_ORDER", raising=False)
    store = ModelProviderStore(tmp_path / "providers.db")
    service = ModelProviderService(store=store)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-openrouter-test-abcd")
    monkeypatch.setenv("COMMONSTACK_API_KEY", "cs-commonstack-test-abcd")

    assert service.resolve_platform_execution_candidates(
        "qwen/qwen3.7-plus"
    ) == ("commonstack", "openrouter")
    # Haiku reaches CommonStack only through #535's allowlist backfill.
    assert service.resolve_platform_execution_candidates(
        "anthropic/claude-haiku-4-5"
    ) == ("commonstack", "openrouter")

    monkeypatch.setenv("ATL_PLATFORM_PROVIDER_ORDER", "openrouter,commonstack")
    assert service.resolve_platform_execution_candidates(
        "qwen/qwen3.7-plus"
    ) == ("openrouter", "commonstack")

    # Leaving a lane out takes it out of automatic routing...
    monkeypatch.setenv("ATL_PLATFORM_PROVIDER_ORDER", "openrouter")
    assert service.resolve_platform_execution_candidates(
        "qwen/qwen3.7-plus"
    ) == ("openrouter",)
    # ...but an explicit provider_id from the caller is still honoured first.
    assert service.resolve_platform_execution_candidates(
        "qwen/qwen3.7-plus", preferred_provider_id="commonstack"
    ) == ("commonstack", "openrouter")

    # An order naming nothing routable yields no candidates; the route 422s.
    monkeypatch.setenv("ATL_PLATFORM_PROVIDER_ORDER", "anthropic")
    assert service.resolve_platform_execution_candidates("qwen/qwen3.7-plus") == ()

    monkeypatch.delenv("ATL_PLATFORM_PROVIDER_ORDER")
    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert service.resolve_platform_execution_candidates(
        "qwen/qwen3.7-plus"
    ) == ("commonstack",)

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-openrouter-test-abcd")
    monkeypatch.delenv("COMMONSTACK_API_KEY")
    assert service.resolve_platform_execution_candidates(
        "qwen/qwen3.7-plus"
    ) == ("openrouter",)


def test_platform_provider_order_parses_and_rejects_junk_whole(monkeypatch, capsys):
    from dashboard.backend.domain.model_providers import service as service_module

    monkeypatch.setattr(service_module, "_warned_platform_provider_orders", set())

    monkeypatch.delenv("ATL_PLATFORM_PROVIDER_ORDER", raising=False)
    assert service_module.platform_provider_order() == ("commonstack", "openrouter")
    monkeypatch.setenv("ATL_PLATFORM_PROVIDER_ORDER", "  ")
    assert service_module.platform_provider_order() == ("commonstack", "openrouter")

    monkeypatch.setenv(
        "ATL_PLATFORM_PROVIDER_ORDER", " OpenRouter , commonstack,,openrouter "
    )
    assert service_module.platform_provider_order() == ("openrouter", "commonstack")
    assert capsys.readouterr().out == ""

    for junk in ("commonstack;openrouter", "Common Stack", "commonstack,open-router"):
        monkeypatch.setenv("ATL_PLATFORM_PROVIDER_ORDER", junk)
        assert service_module.platform_provider_order() == (
            "commonstack",
            "openrouter",
        )
        assert service_module.platform_provider_order() == (
            "commonstack",
            "openrouter",
        )
        out = capsys.readouterr().out
        assert out.count("WARNING: ATL_PLATFORM_PROVIDER_ORDER") == 1
        assert junk not in out
```

In `dashboard/backend/tests/domain/model_providers/test_service.py`, replace the whole `test_execution_options_keep_openrouter_ahead_of_commonstack` function with:

```python
def test_execution_options_follow_platform_order_and_keep_byok_order(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("ATL_PLATFORM_PROVIDER_ORDER", raising=False)
    service, _store = _service(tmp_path, FakeAdapter())
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-fake-options-abcd")
    monkeypatch.setenv("COMMONSTACK_API_KEY", "cs-fake-options-wxyz")

    # Providers outside the platform order keep repository (display-name)
    # order ahead of the platform lanes, so app.js's providers[0] BYOK
    # default is exactly what it was before CommonStack moved first.
    assert [
        option.provider_id for option in service.list_execution_options(7)
    ] == ["anthropic", "gemini", "openai", "commonstack", "openrouter"]

    monkeypatch.setenv("ATL_PLATFORM_PROVIDER_ORDER", "openrouter,commonstack")
    assert [
        option.provider_id for option in service.list_execution_options(7)
    ] == ["anthropic", "gemini", "openai", "openrouter", "commonstack"]
    assert "cs-fake-options-wxyz" not in repr(service.list_execution_options(7))
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `python3 -m pytest dashboard/backend/tests/domain/model_providers/test_execution_catalog.py dashboard/backend/tests/domain/model_providers/test_service.py -q -p no:cacheprovider -k "platform_candidates or platform_provider_order or follow_platform_order"`

Expected: 3 FAIL.
- The candidates test fails with `('openrouter', 'commonstack') != ('commonstack', 'openrouter')`.
- The parse test fails with `AttributeError: … has no attribute '_warned_platform_provider_orders'`.
- The options test fails with a list-order mismatch that shows `openrouter` before `commonstack`.

- [ ] **Step 3: Implement the helper**

In `dashboard/backend/domain/model_providers/service.py`, add `ModelProviderStoreError,` to the `from .repository_common import (` block, keeping it alphabetical after `CredentialConflictError,`. Then add this directly after the `_environment_platform_secret` function:

```python
DEFAULT_PLATFORM_PROVIDER_ORDER: tuple[str, ...] = ("commonstack", "openrouter")
_PLATFORM_PROVIDER_ORDER_ENV = "ATL_PLATFORM_PROVIDER_ORDER"
_warned_platform_provider_orders: set[str] = set()


def platform_provider_order() -> tuple[str, ...]:
    """Return the ATL Credits provider preference, read per call.

    CommonStack comes first by default so its prepaid balance is spent
    before OpenRouter's (#522). The order used to be hard-coded
    OpenRouter-first, and CommonStack carried the traffic only because
    OpenRouter's key was quota-dead (#523).

    The value is read per call, never at import, so a bad value cannot kill
    boot. Any invalid token rejects the whole value, because a typo'd order
    must not half-apply. A provider left out is never an automatic
    candidate, which is the operator's no-deploy way to pull a drained lane.
    """

    raw = os.getenv(_PLATFORM_PROVIDER_ORDER_ENV, "")
    tokens = [token.strip().lower() for token in raw.split(",") if token.strip()]
    if not tokens:
        return DEFAULT_PLATFORM_PROVIDER_ORDER
    ordered: list[str] = []
    try:
        for token in tokens:
            provider_id = validate_provider_id(token)
            if provider_id not in ordered:
                ordered.append(provider_id)
    except ModelProviderStoreError:
        if raw not in _warned_platform_provider_orders:
            _warned_platform_provider_orders.add(raw)
            print(
                f"WARNING: {_PLATFORM_PROVIDER_ORDER_ENV} is not a comma-separated "
                "list of provider ids; using "
                f"{','.join(DEFAULT_PLATFORM_PROVIDER_ORDER)}"
            )
        return DEFAULT_PLATFORM_PROVIDER_ORDER
    return tuple(ordered)
```

- [ ] **Step 4: Apply it to both call sites**

In `list_execution_options`, replace:

```python
        # Preserve the repository's existing order while keeping OpenRouter as
        # the preferred platform lane before the CommonStack fallback.
        options.sort(key=lambda option: option.provider_id == "commonstack")
        return options
```

with:

```python
        # Providers outside the platform order keep repository order, ahead
        # of the platform lanes, so a BYOK user's default pick (app.js takes
        # providers[0]) does not move. The platform lanes follow in
        # ATL_PLATFORM_PROVIDER_ORDER order, CommonStack first by default.
        rank = {
            provider_id: index
            for index, provider_id in enumerate(platform_provider_order())
        }
        options.sort(
            key=lambda option: (
                option.provider_id in rank,
                rank.get(option.provider_id, 0),
            )
        )
        return options
```

In `resolve_platform_execution_candidates`, replace:

```python
        for provider_id in (preferred, "openrouter", "commonstack"):
```

with:

```python
        for provider_id in (preferred, *platform_provider_order()):
```

- [ ] **Step 5: Strip the env var in the test harness**

In `dashboard/backend/tests/conftest.py`, directly after the line `os.environ.pop("PIPELINE_SECONDS_PER_LLM_CALL", None)`, add:

```python

# Decides which platform provider every ATL Credits call tries first. A
# developer configured like prod would otherwise see the routing and
# failover tests' order assertions fail as unexplained mismatches.
os.environ.pop("ATL_PLATFORM_PROVIDER_ORDER", None)
```

- [ ] **Step 6: Run the provider tests**

Run: `python3 -m pytest dashboard/backend/tests/domain/model_providers/ dashboard/backend/tests/infrastructure/llm/ dashboard/backend/tests/test_backtests_router.py -q -p no:cacheprovider`

Expected: all pass.
- `test_platform_credits_env_fallback.py` builds requests with `provider_id="openrouter"` alone, so it still reaches the legacy expansion and is unaffected.
- If any other test asserts the old `("openrouter", "commonstack")` candidate order or the old options order, invert its expectation to match the spec. Record which tests you changed.
- Never change the new code to satisfy an old-order assertion.

- [ ] **Step 7: Commit**

```bash
git branch --show-current   # expect: feat/commonstack-first
git add dashboard/backend/domain/model_providers/service.py dashboard/backend/tests/conftest.py \
        dashboard/backend/tests/domain/model_providers/test_execution_catalog.py \
        dashboard/backend/tests/domain/model_providers/test_service.py
git commit -m "feat(llm): try CommonStack first for ATL Credits, ordered by ATL_PLATFORM_PROVIDER_ORDER

CommonStack carried all traffic only because OpenRouter's key is
quota-dead (#523); topping it up would have stranded CommonStack's
prepaid balance. BYOK provider order is unchanged.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 3: Print one operator ERROR line when a platform provider runs out

**Files:**
- Modify: `dashboard/backend/infrastructure/llm/execution/service.py`:
  - imports (`:3-6`);
  - new module-level state and helper after `_PLATFORM_FAILOVER_CATEGORIES` (`:55-63`);
  - the `except` arm of `_execute_with_platform_failover` (`:399-402`).
- Test: `dashboard/backend/tests/infrastructure/llm/test_platform_credits_env_fallback.py`

**Interfaces:**
- Consumes: `LLMExecutionRequest(provider_id=..., provider_ids=...)` from `infrastructure/llm/execution/models.py`. `provider_ids` must start with `provider_id` and be unique.
- Produces, in `dashboard.backend.infrastructure.llm.execution.service`:
  - `_quota_exhausted_reported: set[str]`, which tests reset;
  - `_quota_exhausted_lock: threading.Lock`;
  - `_report_platform_quota_exhausted(provider_id: str, fallback: str | None) -> None`.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/backend/tests/infrastructure/llm/test_platform_credits_env_fallback.py`:

```python
def _platform_request(run_id: str, provider_ids: tuple[str, ...]) -> LLMExecutionRequest:
    return LLMExecutionRequest(
        user_id=USER_ID,
        run_id=run_id,
        call_index=0,
        billing_mode=BillingMode.PLATFORM_CREDITS,
        provider_id=provider_ids[0],
        provider_ids=provider_ids,
        model_id=MODEL_ID,
        system_message="Return one trading decision.",
        messages=(LLMMessage(role="user", content="Analyze the market."),),
        usage_policy=UsagePolicy(max_output_tokens=100),
    )


def _quota_error() -> ProviderExecutionError:
    return ProviderExecutionError(ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED)


def _ok_response() -> AdapterResponse:
    return AdapterResponse(
        text="BUY",
        model_id=MODEL_ID,
        usage=LLMUsage(input_tokens=40, output_tokens=20),
        finish_reason="stop",
    )


@pytest.fixture
def fresh_quota_reports(monkeypatch):
    monkeypatch.setattr(execution_service_module, "_quota_exhausted_reported", set())


def test_drained_commonstack_prints_one_operator_error_per_process(
    tmp_path, monkeypatch, capsys, fresh_quota_reports
):
    monkeypatch.setenv("COMMONSTACK_API_KEY", "cs-fake-drained-abcd")
    commonstack = ScriptedExecutionAdapter([_quota_error(), _quota_error()])
    openrouter = ScriptedExecutionAdapter([_ok_response(), _ok_response()])
    service, _store = _execution_service(
        tmp_path,
        monkeypatch,
        openrouter,
        adapters={"commonstack": commonstack, "openrouter": openrouter},
    )

    for run_id in ("drained-1", "drained-2"):
        result = service.execute(
            _platform_request(run_id, ("commonstack", "openrouter"))
        )
        assert result.provider_id == "openrouter"

    lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if "llm.platform_quota_exhausted" in line
    ]
    assert lines == [
        "ERROR: llm.platform_quota_exhausted provider=commonstack fallback=openrouter"
    ]


def test_quota_exhaustion_with_no_next_candidate_names_none(
    tmp_path, monkeypatch, capsys, fresh_quota_reports
):
    monkeypatch.setenv("COMMONSTACK_API_KEY", "cs-fake-drained-only-abcd")
    commonstack = ScriptedExecutionAdapter([_quota_error()])
    service, _store = _execution_service(
        tmp_path,
        monkeypatch,
        commonstack,
        adapters={"commonstack": commonstack},
    )

    with pytest.raises(LLMExecutionError) as exc_info:
        service.execute(_platform_request("drained-only", ("commonstack",)))

    assert exc_info.value.category is ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED
    assert (
        "ERROR: llm.platform_quota_exhausted provider=commonstack fallback=none"
        in capsys.readouterr().out
    )


def test_byok_quota_exhaustion_prints_no_operator_error(
    tmp_path, monkeypatch, capsys, fresh_quota_reports
):
    service, _store = _execution_service(
        tmp_path,
        monkeypatch,
        ScriptedExecutionAdapter([]),
    )

    def fail_byok_once(*_args, **_kwargs):
        raise LLMExecutionError(ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED)

    monkeypatch.setattr(service, "_execute_once", fail_byok_once)
    request = _request("byok-drained").model_copy(
        update={"billing_mode": BillingMode.BYOK}
    )

    with pytest.raises(LLMExecutionError):
        service.execute(request)

    assert "llm.platform_quota_exhausted" not in capsys.readouterr().out


def test_quota_report_is_once_under_concurrency_and_flushed(
    capsys, fresh_quota_reports
):
    import inspect
    import threading

    barrier = threading.Barrier(8)

    def report():
        barrier.wait()
        execution_service_module._report_platform_quota_exhausted(
            "commonstack", "openrouter"
        )

    threads = [threading.Thread(target=report) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert capsys.readouterr().out.count("llm.platform_quota_exhausted") == 1
    # A killed child's block-buffered stdout dies with it, and the 3600s
    # timeout kill is exactly when this line matters.
    assert "flush=True" in inspect.getsource(
        execution_service_module._report_platform_quota_exhausted
    )
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `python3 -m pytest dashboard/backend/tests/infrastructure/llm/test_platform_credits_env_fallback.py -q -p no:cacheprovider -k "operator_error or names_none or once_under"`

Expected:
- The drained, none, and concurrency tests error in the fixture with `AttributeError: … has no attribute '_quota_exhausted_reported'`.
- The BYOK test also errors in the fixture. That is fine: it goes green once the attribute exists. It guards against a future regression, not the current code.

If the BYOK test later fails because `execute` wraps the error differently, check that the `except` path in `execute` (`service.py:123-135`) still re-raises `LLMExecutionError`. Adjust the test's `pytest.raises` target to the category it actually raises, never the implementation.

- [ ] **Step 3: Implement**

In `dashboard/backend/infrastructure/llm/execution/service.py`, add `import threading` after `import json`. Then add this directly after the `_PLATFORM_FAILOVER_CATEGORIES = frozenset(...)` block:

```python
# CommonStack has no balance endpoint (every candidate path 404s as of
# 2026-09-23), so a drained platform lane is only observable as a failed
# call. Once per provider per process: a backtest child is one run, so a
# drained lane costs one line per run, not one per call.
_quota_exhausted_reported: set[str] = set()
_quota_exhausted_lock = threading.Lock()


def _report_platform_quota_exhausted(provider_id: str, fallback: str | None) -> None:
    with _quota_exhausted_lock:
        if provider_id in _quota_exhausted_reported:
            return
        _quota_exhausted_reported.add(provider_id)
    print(
        "ERROR: llm.platform_quota_exhausted "
        f"provider={provider_id} fallback={fallback or 'none'}",
        flush=True,
    )
```

In `_execute_with_platform_failover`, replace:

```python
            except LLMExecutionError as exc:
                last_error = exc
                if exc.category not in _PLATFORM_FAILOVER_CATEGORIES:
                    raise
```

with:

```python
            except LLMExecutionError as exc:
                last_error = exc
                if exc.category is ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED:
                    _report_platform_quota_exhausted(
                        provider_id,
                        candidates[attempt_index + 1]
                        if attempt_index + 1 < len(candidates)
                        else None,
                    )
                if exc.category not in _PLATFORM_FAILOVER_CATEGORIES:
                    raise
```

`ExecutionErrorCategory` is already imported from `...execution.errors`; confirm with `grep -n "ExecutionErrorCategory" dashboard/backend/infrastructure/llm/execution/service.py | head -3`.

- [ ] **Step 4: Run the LLM execution tests**

Run: `python3 -m pytest dashboard/backend/tests/infrastructure/llm/ -q -p no:cacheprovider`

Expected: all pass. The pre-existing OpenRouter-quota tests now also print a line. None of them asserts on stdout, and the fixture isolates only the new tests.

- [ ] **Step 5: Commit**

```bash
git branch --show-current   # expect: feat/commonstack-first
git add dashboard/backend/infrastructure/llm/execution/service.py \
        dashboard/backend/tests/infrastructure/llm/test_platform_credits_env_fallback.py
git commit -m "feat(llm): log once when a platform provider's quota runs out

CommonStack exposes no balance endpoint, so a failed call is the only
signal that it drained. One flushed ERROR line per provider per process;
BYOK never prints it.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 4: Neutral billing-hint copy and cache-buster

**Files:**
- Modify: `dashboard/frontend/app.js` (the `'ATL Credits automatically use OpenRouter first, then CommonStack if needed.'` literal, ~`:9885`)
- Modify: `dashboard/frontend/app.html` (`<script src="app.js?v=N" defer>`, ~`:2414`)
- Modify: every test that pins `app.js?v=N`, found by grep
- Test: `dashboard/backend/tests/test_byok_backtest_frontend.py:60-63`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing code-facing; copy only.

- [ ] **Step 1: Update the pinned copy test first**

In `dashboard/backend/tests/test_byok_backtest_frontend.py`, in `test_atl_credits_hides_provider_and_omits_provider_payload`, replace:

```python
    _assert_contains(
        APP_JS,
        "ATL Credits automatically use OpenRouter first, then CommonStack if needed.",
    )
```

with:

```python
    # Neutral on order: ATL_PLATFORM_PROVIDER_ORDER can flip it with no
    # deploy, so copy naming an order would go stale silently.
    _assert_contains(
        APP_JS,
        "ATL Credits automatically switch between CommonStack and OpenRouter if one is unavailable.",
    )
    assert "use OpenRouter first" not in APP_JS
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `python3 -m pytest dashboard/backend/tests/test_byok_backtest_frontend.py -q -p no:cacheprovider -k hides_provider`

Expected: FAIL. The new string is not in `app.js`.

- [ ] **Step 3: Change the copy**

In `dashboard/frontend/app.js`, replace:

```javascript
                ? 'ATL Credits automatically use OpenRouter first, then CommonStack if needed.'
```

with:

```javascript
                ? 'ATL Credits automatically switch between CommonStack and OpenRouter if one is unavailable.'
```

- [ ] **Step 4: Bump the cache-buster everywhere it is pinned**

Derive the current value and every pin. Do not trust a remembered count.

```bash
CUR=$(grep -o 'app\.js?v=[0-9]*' dashboard/frontend/app.html | head -1 | cut -d= -f2); NEW=$((CUR+1)); echo "$CUR -> $NEW"
grep -rln "app\.js?v=$CUR" dashboard/frontend dashboard/backend/tests
```

The expected pins (as of `e1871bae`, v=143) are `app.html` plus four tests:
- `test_admin_analytics_frontend.py`
- `test_analytics_frontend.py`
- `test_backtest_comparison_frontend.py`
- `test_frontend_fast_boot.py`

Replace in exactly the files the grep printed:

```bash
grep -rln "app\.js?v=$CUR" dashboard/frontend dashboard/backend/tests | xargs sed -i "s/app\.js?v=$CUR\b/app.js?v=$NEW/g"
grep -rn "app\.js?v=$CUR" dashboard/frontend dashboard/backend/tests   # expect: no output
```

- [ ] **Step 5: Verify**

Run:

```bash
node --check dashboard/frontend/app.js
python3 -m pytest dashboard/backend/tests/test_byok_backtest_frontend.py dashboard/backend/tests/test_admin_analytics_frontend.py dashboard/backend/tests/test_analytics_frontend.py dashboard/backend/tests/test_backtest_comparison_frontend.py dashboard/backend/tests/test_frontend_fast_boot.py -q -p no:cacheprovider
```

Expected: `node --check` prints nothing, and all tests pass.

- [ ] **Step 6: Commit**

```bash
git branch --show-current   # expect: feat/commonstack-first
git add dashboard/frontend/app.js dashboard/frontend/app.html dashboard/backend/tests/
git status --short          # expect only the files above; nothing under dashboard/storage/
git commit -m "fix(app): describe ATL Credits failover without naming an order

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 5: Pin Haiku pricing and document the order and the ERROR line

**Files:**
- Test: `dashboard/backend/tests/infrastructure/llm/test_pricing.py`
- Modify: `CLAUDE.md` (the LLM backtest billing bullet, plus a new env bullet)
- Modify: `docs/superpowers/specs/2026-09-01-platform-provider-auto-routing-design.md` (one amendment line)

**Interfaces:**
- Consumes: Task 2's `ATL_PLATFORM_PROVIDER_ORDER` and Task 3's ERROR-line format, as documented facts.
- Produces: nothing code-facing.

- [ ] **Step 1: Pin the Haiku price**

Append to `dashboard/backend/tests/infrastructure/llm/test_pricing.py`:

```python
def test_haiku_on_commonstack_prices_at_the_listed_rate():
    # CommonStack lists anthropic/claude-haiku-4-5 at $1 / $5 per million
    # (checked 2026-09-23). The claude-haiku-4 entry already matches, so the
    # #535 allowlist addition needed no pricing-table change.
    assert pricing.price_for_model("anthropic/claude-haiku-4-5") == (1.0, 5.0)
```

- [ ] **Step 2: Run it**

Run: `python3 -m pytest dashboard/backend/tests/infrastructure/llm/test_pricing.py -q -p no:cacheprovider -k haiku_on_commonstack`

Expected: PASS immediately. This pins an existing fact rather than driving new code.
- If it FAILS, the pricing table does not match CommonStack's rate. Stop and report the returned tuple to the user.
- Do not edit `pricing.py`: a wrong price there mis-bills every Haiku call, and the fix is a product decision.

- [ ] **Step 3: CLAUDE.md, billing bullet**

In `CLAUDE.md`, find this text in the `**LLM backtest billing — the live path…**` bullet:

`Design: \`docs/superpowers/plans/2026-08-24-unified-llm-execution-layer.md\`. A run killed by that timeout`

Insert this text between `unified-llm-execution-layer.md\`.` and ` A run killed by that timeout`:

```markdown
 ATL Credits try platform providers in `ATL_PLATFORM_PROVIDER_ORDER` order (below; CommonStack first by default) and fail over on `_PLATFORM_FAILOVER_CATEGORIES`. The first `PROVIDER_QUOTA_EXHAUSTED` a platform provider raises in a process prints `ERROR: llm.platform_quota_exhausted provider=<id> fallback=<next|none>`, once and flushed. It exists because CommonStack has no balance endpoint, so a failed call is the only way to see it drain. BYOK never prints it.
```

- [ ] **Step 4: CLAUDE.md, new env bullet**

In `CLAUDE.md`, insert this new bullet on its own line directly before the line starting `` - `ATL_STRIPE_TEST_BILLING_ENABLED` ``:

```markdown
- `ATL_PLATFORM_PROVIDER_ORDER` (optional, default **`commonstack,openrouter`**): the order in which ATL Credits (`billing_mode=platform_credits`) try platform providers. `platform_provider_order()` in `domain/model_providers/service.py` reads it per call, and both `resolve_platform_execution_candidates` and `list_execution_options` use it.
  - **Why CommonStack is first:** its prepaid balance should be spent before OpenRouter's (#522). The order was hard-coded OpenRouter-first until 2026-09, and CommonStack carried all the traffic only because OpenRouter's key was quota-dead (#523). Topping that key up would have silently stranded CommonStack's balance.
  - **Leaving a provider out** removes it from automatic routing. That is the no-deploy way to pull a drained lane. An explicit `provider_id` in the request body is still honoured first. If nothing routable is left, the route answers 422 rather than guessing.
  - **An invalid token** rejects the whole value, prints one `WARNING`, and falls back to the default. A typo'd order must not half-apply.
  - **Options order:** in `list_execution_options`, providers outside the list keep repository order *ahead* of the platform lanes. `app.js` takes `providers[0]` as the BYOK default, and that pick must not move.
  - **Scope:** BYOK ignores it. The legacy `("openrouter",)` expansion in `infrastructure/llm/execution/service.py` deliberately does not read it; only a direct caller naming OpenRouter alone reaches that expansion.
  - `tests/conftest.py` strips it. Design: `docs/superpowers/specs/2026-09-23-llm-backtest-step-latency-design.md` §5.
```

- [ ] **Step 5: Amend the superseded routing spec**

In `docs/superpowers/specs/2026-09-01-platform-provider-auto-routing-design.md`, insert this line directly after the document's first heading line (the `# …` title), followed by a blank line:

```markdown
> **Amended 2026-09-26:** the OpenRouter-first order this document specifies is superseded. ATL Credits now follow `ATL_PLATFORM_PROVIDER_ORDER`, CommonStack first by default. See `2026-09-23-llm-backtest-step-latency-design.md` §5.
```

- [ ] **Step 6: Commit**

```bash
git branch --show-current   # expect: feat/commonstack-first
git add CLAUDE.md docs/superpowers/specs/2026-09-01-platform-provider-auto-routing-design.md \
        dashboard/backend/tests/infrastructure/llm/test_pricing.py
git commit -m "docs: record the CommonStack-first order and the quota ERROR line

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 6: Pre-merge verification of #535 (controller-owned, not delegated)

**Files:** none modified.

This task makes one real paid call (about $0.001) and touches a shared PR. The controller runs it, not a subagent.

- [ ] **Step 1: Haiku-through-CommonStack probe**

The key lives in the main checkout's `.env`, not in a worktree. Never echo it.

```bash
ENV=/mnt/c/Users/27740/OneDrive/Documents/Github/AgenticTrading/dashboard/.env
KEY=$(grep -E '^COMMONSTACK_API_KEY=' "$ENV" | head -1 | cut -d= -f2- | tr -d '"'"'"' \r')
[ -n "$KEY" ] || { echo "COMMONSTACK_API_KEY missing"; exit 1; }
curl -sS https://api.commonstack.ai/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"anthropic/claude-haiku-4-5","max_tokens":80,"messages":[{"role":"system","content":"Reply with one JSON object only."},{"role":"user","content":"Return {\"action\": \"HOLD\", \"reason\": \"probe\"} exactly."}]}' \
  | jq '{model, finish: .choices[0].finish_reason, content: .choices[0].message.content, usage, error}'
unset KEY
```

Expected: `content` is the JSON object, not a greeting, and `error` is `null`.
- If CommonStack returns a canned greeting, as it once did (`domain/chat/service.py:77`), stop and report to the user. Haiku should not ship on CommonStack.

- [ ] **Step 2: Whole suite on the branch**

Run: `python3 -m pytest dashboard/backend/tests/ -q -p no:cacheprovider --timeout=180`

Expected: 0 failed.
- The Postgres twin tests skip locally unless `TEST_POSTGRES_URL` is set. CI runs them.
- Run `git status --short` and confirm `dashboard/storage/data/backtest.db` is unmodified.

- [ ] **Step 3: Stop for the user**

Report the probe result and the suite counts. Then ask before each of:
1. **Pushing** `feat/commonstack-first`. Re-check `gh pr view 535 --json state` first.
2. **Replacing #535's body.** The first line must stay the imperative `DO NOT MERGE until PR 1 (S1a) has merged and one prod run has verified it.` The rest is a short done-list plus `Refs #522 #523`, with no closing keywords.
3. **Marking #535 ready.** Only after PR 1 is merged and its §8 prod check passed.

After S1b deploys, verify with spec §8's read-only query: every `credit_llm_reservations` row with `attempt_index = 0` has `provider_id = 'commonstack'`.

---

## Follow-ups (not in this plan; surface at the end)

- **User-facing docs, stale** (spec §9, not edited here):
  - `app.html:1305` ("several minutes");
  - `strategy.html:170`;
  - `docs/source/lab/operating_modes.rst`;
  - `docs/source/lab/key_features.rst`.
- **S2–S5** each need their own spec. The default-trading-instruction workstream (central DB `knowledge/atl-default-trading-prompt.md`) feeds S3/S4: roughly halved seconds per bar, and CommonStack `response_invalid` on about 19% of calls.
- **Still waiting on the user from 2026-09-23:**
  - filing the worker-rebuild and SDK-regeneration issues;
  - correcting #522 from 47 to 38 bars.
