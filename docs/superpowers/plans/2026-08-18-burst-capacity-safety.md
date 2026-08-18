# Burst Capacity & Safety — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans` to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** make a 100-agent burst safe and observable, with no per-run CPU optimization.

**Spec:** `docs/superpowers/specs/2026-08-18-burst-capacity-safety-design.md`
**Predecessor:** `docs/superpowers/plans/2026-07-24-agent-scale-sustainability.md`
(four tiers merged; its Task 12 acceptance run executed 2026-08-18 — close it out in T0).

**Architecture:** four independent one-file changes plus a validation run. Each is its
own branch and PR; none depends on another, so they can land in any order.

**Tech stack:** Python 3.13 / FastAPI / threading / requests / pytest.

## Global constraints

- **Wire contract frozen.** No new run/step status literals. New payload *keys* are fine.
- **No new HTTP routes.** The three route-contract freeze tests must pass untouched.
- **`print()`, not `logger`** — `dashboard.backend.*` logger output is invisible under
  deployed uvicorn. Assert with `capsys`, never `caplog`.
- **Env vars read once at import** (mirroring `MAX_ACTIVE_RUNS_PER_AGENT`); tests
  monkeypatch the module constant; every new var is stripped in
  `dashboard/backend/tests/conftest.py`.
- **New defaults, copy verbatim:** `ALPACA_HTTP_TIMEOUT_SECONDS` = `"60"`;
  `ALPACA_HTTP_CONNECT_TIMEOUT_SECONDS` = `"10"`;
  `LEGACY_SESSION_RETENTION_SECONDS` = `"300"`.
- `domain/` must not import `api/` or `app.py` (`test_architecture_boundaries.py`).
- **Never `git add -A`** — a bare backend import runs lazy `ALTER`s against the committed
  seed `dashboard/storage/data/backtest.db`. `git status` before every commit; if the
  seed DB or its `-wal`/`-shm` sidecars are dirty, `git checkout --` them.
- Run from the repo root: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/ -q`.
- One branch + PR per task, cut from up-to-date `origin/main`. Short PR titles.
  **Merging to `main` auto-deploys prod.** Never push to a branch whose PR merged.
- Commit messages use the repo's `feat:`/`fix:`/`test:`/`docs:` convention with the
  session's standard `Co-Authored-By:` / `Claude-Session:` trailers.

---

## T0 — Close out the predecessor plan (docs only)

Branch: `docs/close-scale-acceptance`

The predecessor's status header still says the acceptance run was never executed, and
its Tier-1 rationale cites a dataset size that measurement refuted. Both mislead the
next reader.

**Files:** `docs/superpowers/plans/2026-07-24-agent-scale-sustainability.md`

- [ ] **Step 1: Update the status header (line 3).** Replace "**Still pending:** Task 12
      acceptance …" with the executed result: 100 agents / 35.6 s / 100 completed /
      0 failures / `timeout_holds` 0 in every rung, dated 2026-08-18, linking this plan.
      Note that the run was performed with the harness bug of T3 below still present,
      so its CPU figures are a floor.
- [ ] **Step 2: Correct the `~50 MB` dataset claim** in the Tier-1 comment block
      (around line 615). Measured ~1.7 MB. Keep `MARKET_DATA_CACHE_MAX_ENTRIES=4` —
      only the stated justification changes.
- [ ] **Step 3:** `git status`, commit, PR.

---

## T1 — Default HTTP timeout on the Alpaca client

Branch: `fix/alpaca-http-timeout`

**Why:** `alpaca_bars.py:220` builds `StockHistoricalDataClient` with no timeout;
alpaca-py 0.43.2 calls `self._session.request(method, url, **opts)` with none in `opts`.
`requests` then blocks forever and permanently leaks a threadpool thread. Binds at
concurrency ≥ 1.

**Files:** modify `dashboard/backend/infrastructure/market_data/alpaca_bars.py`,
`dashboard/backend/tests/conftest.py`; create
`dashboard/backend/tests/test_alpaca_http_timeout.py`.

- [ ] **Step 1: Write the failing tests.**
  - A fake client object exposing a `_session` whose `request` records its kwargs →
    after `_apply_default_timeout`, a call with no `timeout` receives
    `(connect, read)` from the module constants.
  - A caller-supplied `timeout=` is **not** overridden.
  - Applying twice does not double-wrap (assert via the guard attribute, and that one
    call still records exactly one timeout kwarg).
  - A client object with **no** `_session` attribute → returns without raising **and
    prints a warning** containing `_session`. This is the F2 lesson: an upstream rename
    must not silently restore unbounded behaviour.
- [ ] **Step 2: Run them; verify they fail.**
- [ ] **Step 3: Implement.** Module constants read once at import:

  ```python
  ALPACA_HTTP_TIMEOUT_SECONDS = float(os.getenv("ALPACA_HTTP_TIMEOUT_SECONDS", "60"))
  ALPACA_HTTP_CONNECT_TIMEOUT_SECONDS = float(
      os.getenv("ALPACA_HTTP_CONNECT_TIMEOUT_SECONDS", "10"))
  ```

  `_apply_default_timeout(client)` fetches `getattr(client, "_session", None)`, warns
  and returns if absent or lacking `request`, returns early if already wrapped, then
  installs a wrapper doing `kwargs.setdefault("timeout", (connect, read))`. Call it
  immediately after the `StockHistoricalDataClient(...)` construction at line 220.
- [ ] **Step 4: Strip both vars in `tests/conftest.py`** alongside the existing scale knobs.
- [ ] **Step 5:** Run the new file, then `test_market_data*`, then the full suite.
- [ ] **Step 6:** `git status`, commit, PR — `fix: bound Alpaca HTTP requests with a timeout`.

---

## T2 — Make decision-deadline auto-holds visible

Branch: `fix/log-decision-deadline-holds`

**Why:** `_maybe_apply_timeout` (`external_run_service.py:421`) reattributes a step to
`decision_source="timeout_hold"` with no output whatsoever. A published curve containing
auto-held steps is not the agent's curve.

**Files:** modify `dashboard/backend/domain/backtesting/external_run_service.py`;
create `dashboard/backend/tests/test_deadline_hold_visibility.py`.

- [ ] **Step 1: Write the failing tests.**
  - Drive a session past its deadline (monkeypatch `DECISION_TIMEOUT_SECONDS` or the
    clock), poll `get_current_step`, assert `capsys` output contains the backtest id
    and the hold count.
  - **One line per poll, not per step:** force ≥ 3 steps to expire in a single drain
    and assert exactly one `decision deadline` line is printed, carrying `3`.
  - A run with no expired step prints nothing.
  - `timeout_holds` still increments exactly as before (guard against the counter
    being disturbed).
- [ ] **Step 2: Run them; verify they fail.**
- [ ] **Step 3: Implement.** `_maybe_apply_timeout` keeps its current signature and
      stays silent. The `while self._maybe_apply_timeout():` loop in
      `get_current_step` counts iterations and, after the loop, prints once when the
      count is non-zero:

  ```
  ⚠️ decision deadline: auto-held {n} step(s) for {backtest_id}
     (agent={agent_name}, step_index={i}, total_holds={t})
     — these steps are NOT the agent's decisions
  ```

  Check `drain_expired()` (line 734) for the same loop shape and give it the same
  treatment if it drains independently.
- [ ] **Step 4:** Run the new file, then `test_run_lifecycle_unification.py`,
      `test_protocol_api.py`, then the full suite.
- [ ] **Step 5:** `git status`, commit, PR — `fix: log decision-deadline auto-holds`.

---

## T3 — Sweep terminal legacy sessions

Branch: `fix/evict-terminal-legacy-sessions`

**Why:** `reap_runs()` evicts terminal engine sessions by walking `_runs`, which the
legacy `/api/v1/backtest/*` surface never populates. `MAX_LEGACY_ACTIVE_GLOBAL` cannot
bound them either — `_count_active_locked` (`:919`) skips terminal sessions. Unbounded.

**Files:** modify `dashboard/backend/domain/backtesting/external_run_service.py`,
`dashboard/backend/app.py`, `dashboard/backend/tests/conftest.py`; create
`dashboard/backend/tests/test_legacy_session_sweep.py`.

- [ ] **Step 1: Write the failing tests.**
  - A terminal session is **not** evicted on the first sweep (TTL clock starts) and
    **is** evicted on a sweep after `LEGACY_SESSION_RETENTION_SECONDS` (monkeypatch the
    constant to `0` or advance the clock).
  - A non-terminal session is never evicted regardless of age.
  - The sweep is idempotent and returns the number dropped.
  - Reading a run whose session was evicted still works via the persisted row
    (this is the safety claim in `evict_session`'s docstring — pin it).
  - **Regression guard:** a terminal session left in `_sessions` does not count toward
    `_count_active_locked`, so capacity is unaffected either way — assert the cap
    behaves identically before and after a sweep.
- [ ] **Step 2: Run them; verify they fail.**
- [ ] **Step 3: Implement `sweep_terminal_sessions()`** in `external_run_service.py`.
      Under `_lock`, iterate `list(_sessions.items())`; skip
      `s.status not in TERMINAL_STATUSES`; stamp `s.terminal_seen_at = _utcnow()` on
      first sighting and `continue`; pop once
      `(now - terminal_seen_at) >= LEGACY_SESSION_RETENTION_SECONDS`. Return the count.
      First-sighting rather than stamping in `_finalize`/`cancel` deliberately: it
      cannot be missed by a terminal path that forgets to stamp.
- [ ] **Step 4: Register it** in `app.py` beside the existing
      `register_reaper_sweep(reap_v2_runs)` (`app.py:219-220`). No new thread; it rides
      the 60 s reaper pass.
- [ ] **Step 5: Strip `LEGACY_SESSION_RETENTION_SECONDS`** in `tests/conftest.py`.
- [ ] **Step 6:** Run the new file, then `test_run_lifecycle_unification.py`,
      `test_architecture_boundaries.py`, then the full suite.
- [ ] **Step 7:** `git status`, commit, PR — `fix: evict terminal legacy backtest sessions`.

---

## T4 — Fix the load-test harness

Branch: `fix/loadtest-harness-baselines`

**Why:** `stress_serve.py:60` patches `create_market_data_provider` with a one-argument
lambda against a two-argument signature, so every baseline through every rung failed
and was swallowed as a warning. The harness is the instrument T5 depends on.

**Files:** modify `dashboard/scripts/loadtest/stress_serve.py`,
`dashboard/scripts/loadtest/drive_agents.py`, `dashboard/scripts/loadtest/README.md`.

- [ ] **Step 1:** `stress_serve.py:60` → `lambda *a, **k: FakeAlpacaLoader()`.
- [ ] **Step 2: Make a swallowed baseline failure loud in the harness.** The bug
      survived a full ladder because the failure printed a warning nobody read. Have
      `stress_serve.py` count `Baseline generation failed` occurrences and print a
      summary banner at shutdown, so a future harness break cannot be mistaken for a
      clean run.
- [ ] **Step 3: Add `--windows shared|distinct` to `drive_agents.py`** (default
      `shared`). `shared` gives every agent one date range; `distinct` gives each its
      own. This is the switch F5's diagnosis needs and the difference between one
      queued baseline backtest and N serialized ones.
- [ ] **Step 4: Update the README** — document both flags and state plainly that
      figures produced before this fix are a floor.
- [ ] **Step 5: Smoke-run at 10 agents both ways**, confirming baselines now complete
      (no warning banner) and that `distinct` is visibly slower than `shared`.
- [ ] **Step 6:** `git status`, commit, PR — `fix: repair baseline patching in the load-test harness`.

---

## T5 — Validation against a real Render instance

**Not a code change. Do not start it until T1–T4 have merged.**

Every number in the spec comes from a 12-core dev box plus arithmetic. Nothing in this
workstream has ever run on Render.

- [ ] **Step 1:** Deploy `main` (with T1–T4) to a Render **Standard** instance.
- [ ] **Step 2:** Run the fixed harness at **25 agents** first, `--windows shared`.
      Record wall time, failures, `timeout_holds`, peak RSS.
- [ ] **Step 3:** If 25 is clean, run **100 agents**, `--windows shared`.
      **Acceptance:** zero failures, `timeout_holds == 0`, RSS below the instance
      ceiling, wall time within ~2× the ~47 s prediction.
- [ ] **Step 4:** Run **100 agents `--windows distinct`** once, purely to measure the
      baseline-worker serialization from F5. Expected to be materially slower; that is
      the datum, not a failure.
- [ ] **Step 5: Record results** in this plan's status header. **If they contradict the
      spec's §5 table, the spec is wrong** — correct §5 rather than explaining the
      measurement away.
- [ ] **Step 6:** Only then decide whether the deferred §4 optimization is needed.

---

## Deliberately not in this plan

- **Per-run CPU optimization** (memo + dict rows, 1.51× measured). Deferred by decision;
  numbers are in spec §4 so nobody re-derives them.
- **Raising the anyio threadpool.** Refuted by A/B on the same binary: 40 → 35.6 s /
  0 failures; 160 → 30.3 s / **2 failures**. Do not retry.
- **Any protocol-surface leak fix.** Refuted by measurement — 200 sequential runs, flat
  CPU, plateaued RSS.
- **A fix for F5.** Mechanism unidentified; T5 Step 4 gathers the evidence first.
- **Horizontal scaling / extra workers.** Architecturally closed while `_sessions` and
  `_runs` are module-level and the heartbeat path fails orphaned runs.
