# Backtest run visibility — design

Date: 2026-08-04
Status: approved (requester sign-off in session; approach and scope chosen explicitly)
Scope: failed-run records + failure visibility, LLM decision-coverage disclosure,
editor close-on-run, cache-header fix. Cancel is **out** (issue #273).

## Origin

The recurring tester pain point: minutes into a backtest the page shows only
"Backtesting…" and a timer — no percent, no ETA, no stuck-or-alive signal, no
cancel, and a failed run vanishes without a trace. Most of the *progress* half
shipped in PR #277 (merged 2026-08-02: determinate card bar, step/percent, ETA,
staleness notice). The requester re-tested on stale JS and confirmed the
remaining, still-real gaps are about **failure visibility** and **where the user
is standing when the run starts**.

## What the code actually says (verified 2026-08-04, main @ 8fb868f)

### Finding 1 — a failed backtest leaves zero persistent trace

The frontend-launched backtest is a daemon thread running
`subprocess.run(backtest_hourly_agent.py)` (`backtests.py:384-604`). The engine
calls `db.insert_run(...)` only **after** the full hourly loop completes
(`engine.py:~1018`), so a failure anywhere in the loop writes no `agent_runs`,
`equity_timeseries`, or `trades` rows. The only failure signal is the transient
process-global `backtest_status["error"]`, wiped by the next `/backtest/run`
(`backtests.py:1240`) or a restart. Consequently no history surface can show a
failed run: it is unrepresentable, not merely unrendered.

### Finding 2 — mid-run failure is invisible on the page the app itself navigates to

`runBacktest()` navigates to My Agents at launch (`app.js:6009`). But the poll
loop's terminal-error branch (`app.js:5391-5407`) clears every running badge,
repaints the cards to idle, and writes the error text only into the Backtest-tab
panel — and only `if (viewingLive)`. Standing on My Agents (the landing page
after launch), a failed first run reverts the card to "No backtests yet" as if
nothing happened. Launch-time failures are fine (`showBacktestLaunchFailure`
alerts on My Agents).

### Finding 3 — the quota case usually doesn't fail the run at all

The default pipeline runtime silently falls back to rule-based per step on LLM
failure. A quota-dead key typically produces a **successful** backtest that the
model barely drove. The engine persists only `llm_calls` (a billing counter that
also ticks on failed calls); `PortfolioManager.llm_decisions`
(`portfolio_manager.py:100`, the H6 coverage counter with exactly two write
sites) is never persisted per run, so the condition is invisible after the fact.
Hosted runtimes differ: they carry a bounded failure budget and abort
(`engine.py:917-944`), landing in Finding 1's bucket.

### Finding 4 — launching from the agent editor hides the launch

The agent editor is a fullscreen overlay (`position:fixed; inset:0;
z-index:1200`, `styles.css:8625`). The run-backtest modal sits above it
(`z-index:1300`) and works, but nothing in the launch path closes the editor —
`navigateToPage('playground', {playgroundTab:'agents'})` happens underneath the
opaque overlay. The user keeps staring at the settings page with no sign the run
started. (`AgentEditor.close(true)` is already called for the
`agent-editor-open-run` event, `app.js:3818-3826` — the pattern exists.)

### Finding 5 — the HTML that carries the cache-busters is the one thing cached for an hour

`dashboard/frontend/vercel.json` gives `/(.*)` `max-age=3600` and overrides `/`,
`/app.js`, `/styles.css` to `must-revalidate` — but not `/app` (nor `/app.html`),
which serves the buster-carrying markup. For up to an hour after any deploy a
plain reload keeps old busters → old JS. This is why the requester's re-test
looked broken, and it degrades every frontend deploy.

### Finding 6 — the join key already exists

`POST /backtest/run` mints `live_run_id` (`backtests.py:1229`), passes it to the
subprocess as `--run-id` (`:507`), and the CLI uses it for the DB row
(`backtest_hourly_agent.py:220`), returning `run_id == live_run_id` to the
client (`:1283-1284`). A launch-time journal row therefore joins cleanly to the
eventual `agent_runs` row. `run_backtest_background` already holds `agent_id`,
`session_id`, and the per-run computed `subprocess_timeout` (`:518-528`).

## Decisions

- **D1 — storage**: additive `backtest_attempts` journal, not a status column on
  `agent_runs`. `agent_runs` has two writers in two processes and success-only
  semantics assumed by every reader; the journal keeps one writer per table and
  makes failure-awareness opt-in per surface.
- **D2 — no new routes.** All data rides existing responses (agents payload,
  `agent.runs`, `GET /api/backtest/runs`), so the three route-contract freeze
  guards stay untouched.
- **D3 — interrupted classification is read-side, not write-side.** No startup
  sweep: a `running` row older than its own stored `timeout_seconds` + 10 min
  margin is *presented* as interrupted. A sweep that writes would let a local
  dev process pointed at the shared `AGENT_RUNS_DATABASE_URL` clobber a genuine
  prod run.
- **D4 — coverage disclosure keys on the H6 constant** (`< 0.95`,
  `MIN_LLM_DECISION_COVERAGE`) and lives in the existing `metadata` JSON column
  — no schema change, no leaderboard/H6 behavior change.
- **D5 — deploy-skew tolerance** (issue #304): every new field is additive and
  the frontend treats its absence as "no information" — a new frontend against
  the old backend renders exactly today's UI.
- **D6 — two PRs.** PR-1: cache headers + editor close (frontend-only, zero
  backend dependency, ships immediately). PR-2: journal + surfaces + coverage
  chip.
- **D7 — dismissal is client-local** (localStorage keyed by `run_id`), not an
  API concern. A dismissed failure reappearing on another device is acceptable;
  an acknowledgement API is not worth a route + auth surface.

## Non-goals

- Cancel — tracked in issue #273; touches process handling and the PR #163
  completion-detection race, deliberately excluded here.
- Changing the silent per-step LLM fallback itself (failure budgets for the
  pipeline runtime, hard-fail modes) — disclosure only.
- Any RTD/user-facing docs edit — `getting_started.rst` (describes the launch
  flow) and `architecture.rst` (names both endpoints and the subprocess) will
  both be stale after this round; surfaced as an explicit end-of-round
  follow-up for the requester to coordinate.
- Multi-run concurrency, run queueing, or per-user run isolation — the
  one-run-per-process global stays as is.
- `backtest_status` refactoring beyond what the journal touches.

## Workstream 1 — `backtest_attempts` journal (backend)

### 1.1 Schema (both twins, literal DDL)

```sql
CREATE TABLE IF NOT EXISTS backtest_attempts (
    run_id TEXT PRIMARY KEY,          -- == live_run_id == agent_runs.run_id on success
    agent_id TEXT,
    session_id TEXT NOT NULL,
    agent_name TEXT,
    start_date TEXT,
    end_date TEXT,
    params_json TEXT,                 -- small launch-config snapshot
    status TEXT NOT NULL,             -- 'running' | 'completed' | 'failed'
    error TEXT,                       -- sanitized, capped (500 chars)
    timeout_seconds INTEGER,          -- the per-run computed subprocess timeout
    created_at TEXT NOT NULL,
    finished_at TEXT
)
```

Index on `(agent_id, created_at)` and `(session_id, created_at)`. Methods (same
signatures in `BacktestDatabase` and `PostgresBacktestDatabase`, new pair in
`test_store_twin_parity.py`): `insert_attempt(...)`,
`finalize_attempt(run_id, status, error=None)`,
`get_attempts_for_session(session_id, limit)`,
`get_latest_attempt_for_agents(agent_ids)` (batched — one query, no N+1; the
My Agents list is the hot path).

### 1.2 Lifecycle

- `POST /backtest/run` inserts the `running` row immediately after minting
  `live_run_id` and passing validation, before the thread starts.
- `run_backtest_background` finalizes: returncode 0 → `completed`; non-zero or
  exception (incl. timeout) → `failed` with the **same sanitized string** it
  already computes via `_sanitize_backtest_error` (secrets already stripped —
  reuse, don't re-derive). Truncate to 500 chars at write.
- Journal writes are best-effort: wrapped, failures `print()`ed
  (`logger.*` is invisible under deployed uvicorn), never abort a launch or a
  finalize.
- Read-side classification (D3): helper `present_attempt(row)` maps stale
  `running` → `interrupted` with a fixed human message; used by every reader so
  the rule lives in one place.

## Workstream 2 — API payload additions (backend)

- `agent_with_stats` (`domain/agents/service.py`): each agent gains
  `latest_backtest_attempt: {run_id, status, error, created_at, finished_at} | null`,
  resolved via the batched query for the list endpoint.
- `agent.runs` (editor history) and `GET /api/backtest/runs` (Backtest-tab
  selector): merge in attempts with presented status `failed`/`interrupted`,
  newest-first, each entry carrying `status` and `error`. Successful attempts
  are **not** merged (their `agent_runs` row already appears; `run_id` equality
  is the dedup key). Existing entries gain `status: "completed"` so consumers
  can switch on one field.

## Workstream 3 — frontend failure states

### 3.1 Card failed state

- Display rule: `latest_backtest_attempt.status ∈ {failed, interrupted}` AND
  (no successful run newer than it) AND `run_id` not in the localStorage
  dismissal set → card renders the Failed state: red accent, "Backtest failed ·
  <relative time>", first line of `error`, Dismiss control. `interrupted`
  wording: "Backtest interrupted (server restarted)".
- Derivation lives in **one** function consumed by both the full render
  (`renderAgentCards`) and any patch path — the PR #277 two-renderers lesson.
- Absent field (old backend) → today's card exactly (D5).

### 3.2 The poll loop stops being the only failure messenger

The terminal-error branch keeps its current shape (Backtest-panel error when
`viewingLive`; `loadAgents()` when on the agents tab). What changes is the
*source of truth*: the failed card derives from `latest_backtest_attempt` in
the agents payload, so failure visibility no longer depends on which tab the
poll loop happened to find the user on — any later visit to My Agents, from any
device, shows it. No new alert: the card itself is the signal.

### 3.3 History surfaces

- Editor run-history list: failed/interrupted entries get a status badge and no
  return figure (there is none).
- Backtest-tab selector: failed attempts are selectable; selection renders the
  config panel through the existing `statusLabel: 'Failed'` path with the stored
  error, and **skips chart-data fetches** (no 404s); chart area shows a single
  empty-state line.

## Workstream 4 — LLM coverage disclosure

- Engine: at the `insert_run` call site (manager in scope), add to `metadata`:
  `llm_decisions`, `llm_total_steps`, `llm_decision_coverage` (float 0-1).
  Written whenever the run was **configured with a model** (regardless of
  whether any call succeeded — a quota-dead key from step 1 must still yield
  coverage 0); absent for rule-based configs, and absence must not render as
  0% coverage.
- Frontend: on a completed run's config panel and its editor-history entry, when
  a model was configured and coverage `< 0.95`: warning chip
  "Model drove N% of steps — rule-based fallback filled the rest; check your
  API key/quota." Coverage `0` sharpens to "Model drove 0 steps".
- Old runs without the metadata keys show nothing (no retroactive claims).

## Workstream 5 — editor close-on-run (PR-1)

`runBacktest()` closes the editor overlay (`AgentEditor.close(true)` guarded by
existence, the `agent-editor-open-run` pattern) at the same optimistic point it
calls `navigateToPage`. Launch-failure alert behavior unchanged — the user is
then on My Agents, which is where the alert convention already fires.

## Workstream 6 — cache headers (PR-1)

`dashboard/frontend/vercel.json`: add `source: "/app"` and `source: "/app.html"`
rules with `Cache-Control: public, max-age=0, must-revalidate`. Nothing else
changes. Verification is external-probe only (curl response headers on the
deployed host — the local file proves nothing about prod).

## Error handling summary

- Journal write failure: printed, swallowed; launch/finalize proceed.
- Attempt row missing at finalize (e.g. insert failed earlier): finalize
  inserts a minimal terminal row from the fields the runner holds (upsert by
  `run_id`) — the failure record survives; never raises.
- Frontend: all new payload fields optional; `typeof` checks (not truthiness —
  the `Number(null) === 0` trap from #277 applies to coverage floats).
- Error text is already sanitized upstream; the cap is a second bound, not the
  sanitizer.

## Testing

- Twin parity: new store pair registered; both twins carry identical method
  signatures and column names (mind the f-string-DDL invisibility trap — literal
  DDL only).
- Backend: lifecycle (insert → finalize success/failure/timeout), read-side
  interrupted classification (age math against the row's own
  `timeout_seconds`), payload merge + dedup-by-run_id, batched latest-attempt
  query shape (no N+1), coverage metadata written/absent correctly, error-cap.
- Frontend static guards (house rules from `tests/_frontend_source.py`): strip
  comments before `in`/`not in`; scope `not in` to the narrowest branch;
  derive-once assertion for the failed-card state (render → patch → re-render
  field equality); buster floors are parsed `>=`, never literal pins. Each PR
  bumps the busters of the files it touches (at time of writing: `app.js` v=62,
  `styles.css` v=79).
- Payload-driven failure guard: the failed-card state must render from the
  agents payload alone (no live poll required) — assert `renderAgentCards`
  consumes `latest_backtest_attempt` without touching poll-loop globals.
- PR-2 must not regress PR #277's invariants: `deriveRunningProgress` gating on
  `entry.runId === liveBacktestRunId`, server-side `progress_age_seconds`, ETA
  anchored at first observed step.
