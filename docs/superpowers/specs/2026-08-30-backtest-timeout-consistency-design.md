# Dashboard Backtest Timeout Consistency

**Status:** Draft for user review

**Date:** 2026-08-30

## Context

Dashboard backtests currently expose three different time boundaries. The
browser stops polling after 600 seconds, the pipeline worker parent is allowed
1800 seconds, and some hosted runtimes calculate a separate budget from their
per-decision timeout. A backtest can therefore still be running on Render after
the browser says it timed out, or the browser can stop showing progress before
the backend has reached its own limit.

The dashboard backtest loop performs one decision for each hourly market bar.
Longer ranges and reasoning-enabled models can legitimately exceed ten minutes.
The requested behavior is a fixed 60-minute wall-clock budget for the normal
dashboard pipeline run, with matching browser status visibility.

## Goals

1. Raise the dashboard pipeline subprocess hard limit from 30 minutes to 60
   minutes (3600 seconds).
2. Raise the browser's dashboard backtest polling and local running-entry
   lifetime from 10 minutes to 60 minutes (3600 seconds).
3. Update every user-visible dashboard backtest duration message that currently
   promises or reports ten minutes.
4. Preserve terminal cleanup: a timeout must terminate the child process,
   release platform-credit reservations, release the backtest slot, and expose a
   terminal status instead of leaving a permanent running state.
5. Cover the backend/frontend timeout contract with deterministic tests.

## Non-goals

- Do not change provider HTTP request timeouts, database timeouts, email
  timeouts, or other unrelated network/runtime limits to 60 minutes.
- Do not change OpenRouter/Qwen response handling, Gemini behavior, or model
  reasoning policy in this PR.
- Do not make the backtest faster. LLM call count, hourly decision cadence,
  market-data loading, and pipeline semantics remain unchanged.
- Do not remove the hard timeout or allow an unbounded background process.
- Do not change the 31-day date-range validation in this PR. A 31-day run may
  still exceed the fixed 60-minute budget and must report a normal timeout.
- Do not change the Render plan or the server-wide concurrent-run default in
  code. Free-tier rollout capacity is an operational setting, not a timeout
  contract.

## Design

### 1. Backend wall-clock budget

Set `PIPELINE_SUBPROCESS_TIMEOUT_SECONDS` in
`dashboard/backend/api/routers/backtests.py` to `3600`. The existing
`_backtest_subprocess_timeout()` behavior remains otherwise unchanged:

- `runtime_type == "pipeline"` receives the fixed 3600-second parent budget;
- hosted runtime types retain their existing per-decision calculation and
  14400-second ceiling;
- the timeout passed to each provider HTTP client remains its existing short
  request timeout, so one stalled provider call does not pin a worker for an
  hour by itself.

When `subprocess.run(..., timeout=3600)` expires, the current background-worker
exception/finally path remains the owner of cleanup. The implementation must
verify that the child is terminated and that both `finalize_run()` and slot
finalization still run exactly once. The user-facing error remains sanitized;
it must not include provider prompts, responses, or credentials.

The constant name can remain stable to minimize downstream churn. Comments and
tests must describe the value as a 60-minute pipeline budget rather than a
generic provider timeout.

### 2. Frontend observation window

Set `BACKTEST_POLL_MAX_SECONDS` in `dashboard/frontend/app.js` to `3600`.
Every existing consumer of this constant must inherit the same value:

- the shared status poller;
- the localStorage running-backtest registry cleanup;
- per-agent running-card cleanup;
- the legacy `pollBacktestStatus()` wait helper;
- progress-bar fallback calculations.

The poller continues to treat a failed status request as inconclusive and keeps
watching until the existing failure budget is exhausted. At the 60-minute
client ceiling it stops polling and shows a clear message that the run may
still be running in the background. It clears only the browser's local mirror;
it must not claim a successful result or silently replace the run with another
one. The backend remains authoritative for final status and cleanup.

Update `dashboard/frontend/app.html` and the timeout message in `app.js` so the
visible copy consistently says 60 minutes. General product copy should avoid a
false exact promise such as "3-10 minutes" and instead say that longer
multi-step backtests can take several minutes.

### 3. Contract and resource boundaries

The two 3600-second values are intentionally duplicated because this dashboard
has no frontend build-time configuration channel. A focused contract test will
assert that both values and the visible timeout copy agree. No provider or
database timeout is allowed to inherit this constant.

The longer budget increases the time each run can hold a loaded bar window and
an active slot. The existing server-wide concurrency guard remains in place.
Before deploying to the 512 MB free Render instance, operators should use a
conservative `MAX_ACTIVE_DASHBOARD_BACKTESTS` value (one or two) if live memory
measurements show contention. That operational decision is deliberately outside
this code PR.

### 4. Failure and status sequence

The expected lifecycle is:

1. The API accepts the run and returns its `live_run_id` immediately.
2. The browser polls the run status for up to 3600 seconds and continues to
   render progress/staleness notices from the server payload.
3. A completed or failed worker publishes its terminal status; the browser
   loads or displays that status and removes the local running entry.
4. If the parent subprocess reaches 3600 seconds first, it terminates the
   worker, finalizes the execution run, releases any open reservation and slot,
   and records a sanitized timeout error.
5. If the browser reaches 3600 seconds first because of a network or process
   discrepancy, it displays the background-running message and does not invent
   a result. Reloading can discover the server's terminal status.

The fixed limit is a protection against orphaned work, not a performance
guarantee. It prevents the current 10-minute/30-minute mismatch while keeping a
bounded failure mode for unusually long ranges.

## Testing strategy

### Backend

Extend `dashboard/backend/tests/test_backtests_router.py` to verify:

1. A pipeline run resolves to 3600 seconds.
2. Hosted runtime sizing remains dynamic and still respects the existing
   14400-second ceiling.
3. A timeout path releases the worker execution run and backtest slot without
   double-finalizing them.
4. Existing date-range and concurrency validations are unchanged.

### Frontend

Extend `dashboard/backend/tests/test_backtest_progress_card.py` to verify:

1. `BACKTEST_POLL_MAX_SECONDS` is 3600.
2. Running entries are retained through 3599 seconds and swept after the
   3600-second ceiling.
3. The poller, legacy wait helper, and progress-bar fallback use the same
   ceiling.
4. The terminal copy reports 60 minutes, and no visible copy still reports a
   ten-minute backtest limit.
5. A server terminal error/result still wins over the client timeout branch.

Run the focused backend/frontend tests, the existing backtest router tests,
`git diff --check`, and the relevant Python syntax checks before opening the PR.

## Rollout and acceptance

The PR is accepted when:

- all targeted tests pass;
- no unrelated timeout constants change;
- `git diff --check` is clean;
- a deployed test run that lasts longer than ten minutes remains visible in the
  browser and can still reach a terminal result before 60 minutes;
- a forced timeout leaves no active slot, open platform-credit reservation, or
  stale browser running card.

The deployment must be verified against the new commit. Render's current
service uses manual deployment, so merging or pushing alone is not evidence that
the 60-minute backend budget is live.
