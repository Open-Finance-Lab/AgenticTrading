# Dashboard Backtest Timeout Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give dashboard backtests a consistent, bounded 60-minute observation and execution window without changing provider request timeouts or backtest semantics.

**Architecture:** Keep the backend parent-process budget and frontend polling ceiling as two explicit 3600-second contract values because the shipped frontend has no build-time configuration channel. Reuse the existing status-slot, subprocess-finally, and shared frontend polling paths; only update their constants, copy, and focused tests.

**Tech Stack:** Python 3, FastAPI background-thread backtest runner, `subprocess.run`, pytest, vanilla JavaScript, Node-based frontend source harnesses, static HTML.

**Spec:** `docs/superpowers/specs/2026-08-30-backtest-timeout-consistency-design.md`

## Global Constraints

- Do not change provider HTTP request timeouts, database timeouts, email timeouts, or other unrelated network/runtime limits to 60 minutes.
- Do not change OpenRouter/Qwen response handling, Gemini behavior, or model reasoning policy in this PR.
- Do not make the backtest faster. LLM call count, hourly decision cadence, market-data loading, and pipeline semantics remain unchanged.
- Do not remove the hard timeout or allow an unbounded background process.
- Do not change the 31-day date-range validation in this PR. A 31-day run may still exceed the fixed 60-minute budget and must report a normal timeout.
- Do not change the Render plan or the server-wide concurrent-run default in code. Free-tier rollout capacity is an operational setting, not a timeout contract.
- Preserve sanitized errors, platform-credit settlement, backtest-slot release, and browser running-entry cleanup.
- Do not include real API keys, database credentials, `.superpowers/`, or `work/` in changes.

## File Map

- Modify `dashboard/backend/api/routers/backtests.py`: set and document the fixed pipeline parent budget; preserve hosted-runtime dynamic sizing and timeout cleanup.
- Modify `dashboard/frontend/app.js`: set the shared polling/lifetime ceiling to 3600 seconds and update the terminal timeout copy.
- Modify `dashboard/frontend/app.html`: update the progress-card duration hint and remove the false exact 3-10 minute promise.
- Modify `dashboard/backend/tests/test_backtests_router.py`: assert the 60-minute pipeline budget and preserve hosted-runtime sizing behavior.
- Modify `dashboard/backend/tests/test_backtest_progress_card.py`: exercise 60-minute retention/sweep and shared fallback-bar behavior through the existing Node harness.
- Modify `dashboard/backend/tests/test_app_copy_register.py`: assert that shipped HTML has the 60-minute hint and no stale ten-minute limit.

## Implementation Tasks

### Task 1: Lock the backend timeout contract with tests

**Files:**
- Modify: `dashboard/backend/tests/test_backtests_router.py` near `test_hosted_backtest_timeout_covers_every_decision_step`

**Interfaces:**
- Consumes: `backtests_router._backtest_subprocess_timeout(runtime_type, start_date, end_date)` and `PIPELINE_SUBPROCESS_TIMEOUT_SECONDS`.
- Produces: deterministic assertions that the pipeline runtime uses exactly 3600 seconds while hosted runtime sizing remains dynamic and capped.

- [ ] **Step 1: Add the failing pipeline-budget assertion**

Update the test's pipeline branch to assert the explicit target and the fixed-range behavior:

```python
assert bt.PIPELINE_SUBPROCESS_TIMEOUT_SECONDS == 3600
assert bt._backtest_subprocess_timeout(
    "pipeline", "2026-01-01", "2026-01-31"
) == 3600
```

Keep the hosted assertions checking `hosted >= step_seconds * decision_days` and `hosted > bt.PIPELINE_SUBPROCESS_TIMEOUT_SECONDS`; this proves hosted sizing still adapts instead of silently becoming a fixed 60-minute value.

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
cd dashboard
pytest backend/tests/test_backtests_router.py::test_hosted_backtest_timeout_covers_every_decision_step -q
```

Expected: FAIL because the current pipeline constant is 1800 seconds.

- [ ] **Step 3: Commit the test-only checkpoint**

```bash
git add dashboard/backend/tests/test_backtests_router.py
git commit -m "test: lock the 60 minute pipeline timeout"
```

### Task 2: Raise the backend pipeline parent budget and verify cleanup ownership

**Files:**
- Modify: `dashboard/backend/api/routers/backtests.py` at the pipeline timeout constants and `run_backtest_background()` timeout path
- Modify: `dashboard/backend/tests/test_backtests_router.py` for timeout cleanup coverage

**Interfaces:**
- Consumes: `PIPELINE_SUBPROCESS_TIMEOUT_SECONDS`, `_backtest_subprocess_timeout()`, `run_backtest_background()`, `_finalize_slot()`, and `LLMExecutionService.finalize_run()`.
- Produces: a 3600-second pipeline parent budget; a timeout path that remains bounded and finalizes each resource once.

- [ ] **Step 1: Add a deterministic timeout-cleanup test harness**

Use the existing test fixtures/mocks around `run_backtest_background()` and replace the local `subprocess.run` dependency with a `TimeoutExpired` fake. Record calls to the slot finalizer and execution finalizer, then assert:

```python
assert subprocess_call.kwargs["timeout"] == 3600
assert finalized_slots == [(run_id, 0)]
assert finalized_execution_runs == [run_id]
```

The fake must not expose command arguments or credentials in assertions. Keep the test focused on the already-established exception/finally ownership; do not add a second cleanup mechanism.

- [ ] **Step 2: Run the cleanup test and verify it fails or exposes the old budget**

Run the new test by its node and run the existing slot/credit cleanup tests:

```bash
cd dashboard
pytest backend/tests/test_backtests_router.py -k "timeout or finalize or slot" -q
```

Expected: the budget assertion fails at 1800 seconds before the implementation change; any cleanup regression must identify a duplicate or missing finalizer.

- [ ] **Step 3: Change only the pipeline constant and its comments**

Set:

```python
PIPELINE_SUBPROCESS_TIMEOUT_SECONDS = 3600
```

Rewrite the nearby comments to say "60-minute dashboard pipeline parent budget." Leave `SUBPROCESS_TIMEOUT_OVERHEAD_SECONDS`, `MAX_SUBPROCESS_TIMEOUT_SECONDS`, `resolve_step_timeout_seconds()`, provider HTTP timeouts, and hosted-runtime arithmetic unchanged.

- [ ] **Step 4: Make the timeout test pass without broadening exception behavior**

Run:

```bash
cd dashboard
pytest backend/tests/test_backtests_router.py::test_hosted_backtest_timeout_covers_every_decision_step -q
pytest backend/tests/test_backtests_router.py -k "timeout or finalize or slot" -q
```

Expected: PASS. A `subprocess.TimeoutExpired` must still enter the existing sanitized exception path, invoke execution finalization in `finally`, and release the slot once; no provider or credential details may be asserted or emitted.

- [ ] **Step 5: Commit the backend change**

```bash
git add dashboard/backend/api/routers/backtests.py dashboard/backend/tests/test_backtests_router.py
git commit -m "fix: extend dashboard pipeline timeout to 60 minutes"
```

### Task 3: Lock the frontend 60-minute observation contract

**Files:**
- Modify: `dashboard/backend/tests/test_backtest_progress_card.py` near the existing `js_const("BACKTEST_POLL_MAX_SECONDS")` harnesses
- Modify: `dashboard/backend/tests/test_app_copy_register.py` near `test_backtest_hint_uses_strategy_and_limit`

**Interfaces:**
- Consumes: `BACKTEST_POLL_MAX_SECONDS`, `listRunningBacktests()`, `getAgentBacktestRunning()`, `updateBacktestRunProgress()`, and the shipped HTML source.
- Produces: failing tests that pin 3600-second retention, 3600-second elapsed-bar fallback, and 60-minute copy.

- [ ] **Step 1: Add the constant and copy assertions**

Use the existing `js_const()`/Node harness to assert:

```python
assert _node(js_const("BACKTEST_POLL_MAX_SECONDS")) == 3600
```

Update the HTML copy assertion to require:

```python
"Multi-step strategies can take several minutes (limit: 60 minutes)."
```

and reject the stale 10-minute copy. Add a source assertion that the timeout branch says `Timed out after 60 minutes.`.

- [ ] **Step 2: Add boundary tests for localStorage running entries**

Exercise the real `getAgentBacktestRunning()`/`listRunningBacktests()` helpers with mocked `Date.now()` values. Assert an entry at 3599 seconds is retained and an entry older than 3600 seconds is removed. Keep the strict comparison aligned with the existing `elapsed > BACKTEST_POLL_MAX_SECONDS` behavior.

- [ ] **Step 3: Add the elapsed-bar fallback assertion for the new denominator**

Keep the existing panel test but change its expected width from `10% // 60 / 600` to approximately `2% // 60 / 3600`. This proves `updateBacktestRunProgress()` inherits the same shared constant instead of retaining a hidden 600-second denominator.

- [ ] **Step 4: Run the frontend tests and verify they fail**

Run:

```bash
cd dashboard
pytest backend/tests/test_backtest_progress_card.py backend/tests/test_app_copy_register.py -q
```

Expected: FAIL on the 600-second constant, old 10-minute copy, and old elapsed-bar percentage.

### Task 4: Apply the frontend timeout and copy changes

**Files:**
- Modify: `dashboard/frontend/app.js:20` and the timeout branch around line 7027
- Modify: `dashboard/frontend/app.html:1246` and the general backtest description around line 1392

**Interfaces:**
- Consumes: the failing contract tests from Task 3 and every existing consumer of `BACKTEST_POLL_MAX_SECONDS`.
- Produces: a 3600-second poll/lifetime ceiling shared by the poller, registry cleanup, progress fallback, and legacy wait helper.

- [ ] **Step 1: Set the shared frontend ceiling**

Change only the constant and comment:

```javascript
const BACKTEST_POLL_MAX_SECONDS = 3600; // 60 minutes at 1-second polling intervals
```

Do not introduce a second frontend timeout constant. Existing consumers must continue reading this value.

- [ ] **Step 2: Update terminal and progress-card copy**

Change the timeout branch to:

```javascript
message: 'Timed out after 60 minutes. The backtest may still be running in the background.',
```

Change the progress hint to:

```html
<p class="backtest-run-progress-hint">Multi-step strategies can take several minutes (limit: 60 minutes).</p>
```

Change the general product sentence to avoid a false exact promise while retaining the meaning that the workflow can be long:

```html
<p>Chat with Claude; backtests use Alpaca historical data and multi-step runs can take several minutes.</p>
```

Keep the existing HTML structure and typography unchanged.

- [ ] **Step 3: Run the frontend contract tests**

Run:

```bash
cd dashboard
pytest backend/tests/test_backtest_progress_card.py backend/tests/test_app_copy_register.py -q
```

Expected: PASS, including 3599-second retention, post-ceiling cleanup, 2% elapsed fallback at 60 seconds, and the absence of stale ten-minute copy.

- [ ] **Step 4: Commit the frontend change**

```bash
git add dashboard/frontend/app.js dashboard/frontend/app.html dashboard/backend/tests/test_backtest_progress_card.py dashboard/backend/tests/test_app_copy_register.py
git commit -m "fix: align dashboard backtest polling with 60 minute budget"
```

### Task 5: Run the complete focused verification suite

**Files:**
- No new files; verify all files changed in Tasks 1-4.

**Interfaces:**
- Consumes: backend and frontend contract changes from the previous tasks.
- Produces: a clean, reviewable branch ready for the pull request.

- [ ] **Step 1: Run all focused tests**

Run:

```bash
cd dashboard
pytest backend/tests/test_backtests_router.py backend/tests/test_backtest_progress_card.py backend/tests/test_app_copy_register.py -q
```

Expected: PASS with no unrelated test skips beyond the existing Node availability marker.

- [ ] **Step 2: Run Python syntax checks for touched backend modules**

Run:

```bash
python -m py_compile dashboard/backend/api/routers/backtests.py
```

Expected: exit code 0 and no generated bytecode tracked by git.

- [ ] **Step 3: Check for stale timeout copy and unrelated constant changes**

Run:

```bash
rg -n "10 minutes|3.?10 minutes|BACKTEST_POLL_MAX_SECONDS = 600|PIPELINE_SUBPROCESS_TIMEOUT_SECONDS = 1800" dashboard/frontend dashboard/backend
git diff --check
git diff --stat origin/main...HEAD
```

Expected: no stale dashboard backtest-limit copy or old constants; `git diff --check` is clean; the diff contains only the specified backend/frontend files and tests.

- [ ] **Step 4: Inspect cleanup-sensitive diff and status**

Run:

```bash
git diff origin/main...HEAD -- dashboard/backend/api/routers/backtests.py dashboard/frontend/app.js dashboard/frontend/app.html
git status --short --branch
```

Confirm no provider timeout, database timeout, Render config, credential, `.superpowers/`, or `work/` changes are present.

- [ ] **Step 5: Commit any final test-only adjustments**

If the verification step required a test correction, commit only that correction:

```bash
git add dashboard/backend/tests/test_backtests_router.py dashboard/backend/tests/test_backtest_progress_card.py dashboard/backend/tests/test_app_copy_register.py
git commit -m "test: verify the 60 minute backtest timeout contract"
```

Otherwise leave the existing task commits intact and report the commit list in the PR description.

## Pull Request Acceptance Checklist

- [ ] Pipeline dashboard parent timeout is exactly 3600 seconds.
- [ ] Hosted runtime timeout calculation and 14400-second ceiling are unchanged.
- [ ] Frontend polling, running-entry cleanup, legacy wait helper, and elapsed fallback all use 3600 seconds.
- [ ] User-visible dashboard copy consistently says 60 minutes or avoids an exact duration promise.
- [ ] Timeout cleanup releases subprocess/slot/credits resources without double finalization.
- [ ] Focused tests, syntax check, and `git diff --check` pass.
- [ ] No real credentials, `.superpowers/`, or `work/` files are included.
- [ ] Render deployment is separately verified against the merged commit because the service uses manual deployment.
