# Burst Capacity & Safety — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans` to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** make a 100-agent burst safe and observable, with no per-run CPU optimization.

**Spec:** `docs/superpowers/specs/2026-08-18-burst-capacity-safety-design.md`
**Predecessor:** `docs/superpowers/plans/2026-07-24-agent-scale-sustainability.md`
(four tiers merged; its Task 12 acceptance run **partially** executed 2026-08-18 — 3 of
6 Step 2 criteria captured, Step 3 prod smoke not run. T0 records that honestly; T5
closes the rest).

**Architecture:** four small changes plus a validation run, one branch and PR each.
They are independent in *content* but **not in files, so they are not order-free**:

| | Files touched | Shares with |
|---|---|---|
| T0 | predecessor plan, `market_data_store.py` (comment) | — |
| T1 | `alpaca_bars.py`, `conftest.py`, `.env.example`, +1 test | T3 (`conftest.py`, `.env.example`) |
| T2 | `external_run_service.py`, +1 test | T3 (`external_run_service.py`) |
| T3 | `external_run_service.py`, `app.py`, `conftest.py`, `.env.example`, +1 test | T1, T2 |
| T4 | `stress_serve.py`, `drive_agents.py`, `README.md`, `baseline_worker.py`, +1 test | — |

**Land T2 before T3**, and rebase T3 on merged `main` before pushing. `main` has no
branch protection and the observed norm is that open PRs get merged promptly, so two
branches editing `external_run_service.py` and `conftest.py` will collide. A careless
conflict resolution on `conftest.py` silently drops an env-strip line — which is the
exact failure the strips exist to prevent, and it fails *open* rather than red.

**Tech stack:** Python 3.13 / FastAPI / threading / requests / pytest.

## Global constraints

- **Wire contract frozen.** No new run/step status literals. New payload *keys* are fine.
- **No new HTTP routes.** The three route-contract freeze tests must pass untouched.
- **`print()`, not `logger`** — `dashboard.backend.*` logger output is invisible under
  deployed uvicorn. Assert with `capsys`, never `caplog`.
- **Env vars read once at import** (mirroring `MAX_ACTIVE_RUNS_PER_AGENT`); tests
  monkeypatch the module constant; every new var is stripped in
  `dashboard/backend/tests/conftest.py` (append to the scale-knob block at
  `conftest.py:80-93`, same `os.environ.pop("VAR", None)` shape).
- **Every new var is documented in `.env.example` in the same commit** — a module
  constant plus a conftest strip is not a documented default. Follow the existing
  convention there (comment block explaining purpose + default, then a commented-out
  `# VAR=value`, then a blank line); `MARKET_DATA_CACHE_MAX_ENTRIES` at
  `.env.example:206-208` is the model, and the predecessor plan carried this same step
  explicitly (`plans/2026-07-24-…md:1131-1136`). `.env.example` is the canonical home for
  these; only promote to the repo `CLAUDE.md` "Environment & credentials" section
  (`CLAUDE.md:53`) if an operator would need to reason about the knob during an incident.
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

## T0 — Record the predecessor's acceptance result (docs + one source comment)

Branch: `docs/close-scale-acceptance`

The predecessor's status header still says the acceptance run was never executed, and
its Tier-1 rationale cites a dataset size that measurement refuted. Both mislead the
next reader. **This task does not close Task 12** — half its criteria were never
captured, and a gate recorded as closed teaches every later reader that the gate works.

**Files:** `docs/superpowers/plans/2026-07-24-agent-scale-sustainability.md`,
`dashboard/backend/domain/backtesting/market_data_store.py` (comment only).

- [ ] **Step 1: Update the status header (line 3) to a *partial* result.** Replace
      "**Still pending:** Task 12 acceptance …" with what actually happened, dated
      2026-08-18 and linking this plan:
  - **Executed and met** (Step 2): 100 agents / 35.6 s wall / 100 completed / 0 failures
    / `timeout_holds` 0 in every rung.
  - **Still pending** (Step 2): `create p95`, `decision p95` and **RSS growth** were
    never captured — peak RSS was recorded instead, which is a different quantity.
    Carried into this plan's T5.
  - **Still pending** (Step 3): the post-deploy prod smoke was not run at all.
  - Caveat, stated precisely — **two different executions sit behind these numbers, and
    only one is a floor.** The **ladder sweep** (1→100 agents), which is where
    `timeout_holds 0 in every rung` comes from, ran with the **T4** harness bug present,
    so its CPU figures (0.406–0.440 CPU-s) *and* its RSS are floors. The wall-time and
    completion numbers above come from a separate **fresh 100-agent run** taken after an
    ad-hoc local repair of that bug; its CPU (0.522 CPU-s) and RSS (311 MB) are **not**
    floors. Do not write a blanket "all figures are a floor" caveat — conflating the two
    is exactly what produced the 0.47 error (spec §2).
  - Keep the wording "Still pending" for the three uncaptured criteria and the smoke.
    Task 12 says "do not relax the criteria"; three unmeasured criteria are not three met
    ones, and this header is the only place a later reader looks.
- [ ] **Step 2: Correct the `~50 MB` dataset claim in the plan** — the fenced Tier-1
      comment block at lines 614-618. Measured ~1.7 MB (a floor: the source print counts
      `all_data` frames only, on synthetic harness bars — see spec §2). Keep
      `MARKET_DATA_CACHE_MAX_ENTRIES=4`; only the stated justification changes.
- [ ] **Step 3: Correct the same claim in the shipped source.** Lines 614-618 of the plan
      are a *verbatim quote of a live source comment* at
      `dashboard/backend/domain/backtesting/market_data_store.py:34-37`. Fixing only the
      plan leaves the wrong number — and its "~200 MB worst case against the 512 MB free
      tier" headroom claim, off by ~30× — in the file a capacity reviewer actually opens.
      Comment text only; no behaviour change, no test change.
- [ ] **Step 4:** `git status` (confirm the seed DB and its `-wal`/`-shm` sidecars are
      clean — this task imports no backend module, so they should be), commit, PR.

---

## T1 — Default HTTP timeout on the Alpaca client

Branch: `fix/alpaca-http-timeout`

**Why:** `alpaca_bars.py:220` builds `StockHistoricalDataClient` with no timeout;
alpaca-py 0.43.2 calls `self._session.request(method, url, **opts)` with none in `opts`.
`requests` then blocks forever and permanently leaks a threadpool thread. Binds at
concurrency ≥ 1.

**Files:** modify `dashboard/backend/infrastructure/market_data/alpaca_bars.py`,
`dashboard/backend/tests/conftest.py`, `.env.example`; create
`dashboard/backend/tests/test_alpaca_http_timeout.py`.
⚠ `conftest.py` and `.env.example` are also touched by T3 — see the Architecture note.

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
- [ ] **Step 4: Strip both vars in `tests/conftest.py`** alongside the existing scale
      knobs (the block at `conftest.py:80-93`).
- [ ] **Step 5: Document both vars in `.env.example`**, in the same commit, following the
      convention there:

  ```
  # Bound every Alpaca market-data HTTP request. alpaca-py issues requests with no
  # timeout, so one stalled socket leaks a threadpool thread for the life of the
  # process. Connect/read seconds. Defaults 10 / 60.
  # ALPACA_HTTP_CONNECT_TIMEOUT_SECONDS=10
  # ALPACA_HTTP_TIMEOUT_SECONDS=60
  ```
- [ ] **Step 6:** Run the new file, then `test_market_data*`, then the full suite.
- [ ] **Step 7:** `git status`, commit, PR — `fix: bound Alpaca HTTP requests with a timeout`.

---

## T2 — Make decision-deadline auto-holds visible

Branch: `fix/log-decision-deadline-holds`

**Why:** `_maybe_apply_timeout` (`external_run_service.py:421`) reattributes a step to
`decision_source="timeout_hold"` with no output whatsoever. A published curve containing
auto-held steps is not the agent's curve.

**Files:** modify `dashboard/backend/domain/backtesting/external_run_service.py`;
create `dashboard/backend/tests/test_deadline_hold_visibility.py`.
⚠ `external_run_service.py` is also touched by T3 — land this one first.

- [ ] **Step 1: Write the failing tests.**
  - Drive a session past its deadline (monkeypatch `DECISION_TIMEOUT_SECONDS` or the
    clock), poll `get_current_step`, assert `capsys` output contains the backtest id
    and the hold count.
  - **One line per poll, not per step:** force ≥ 3 steps to expire in a single drain
    and assert exactly one `decision deadline` line is printed, carrying `3`.
  - **The same for `drain_expired()`** — call it directly on a session with ≥ 3 expired
    steps and assert it prints one equivalent line. This is the reaper's path and the
    one with no agent watching; a test that only drives `get_current_step` would pass
    against an implementation that leaves it silent.
  - **The same for `get_status()`** — expire one step, call `session.get_status()`
    directly, assert one line. This is the site an implementer will miss and the one the
    live protocol path actually uses (`runs/service.py:384`, `:749`, `:781`).
  - **Through the router, not just the session.** Drive a real
    `run_service.get_step(run_id, step_id)` past a deadline and assert a line is printed.
    This is the assertion that catches the pre-emption bug: `get_step` calls
    `session.get_status()` first, which applies the hold and advances `step_index`, so
    its `session.step_index == seq` guard then fails and `get_current_step()` — the loop
    everyone thinks of instrumenting — is never reached for that poll. A session-level
    test suite passes happily against an implementation that is silent here.
  - **The printed step range names the steps that were actually held.** Open at step
    index *i*, expire 3 steps, assert the line carries `i..i+2` — **not** the
    post-drain index. `_advance_step` increments `self.step_index` (`:601`) inside the
    loop, so an index read after the loop names a step that was never held. This is the
    only assertion that distinguishes a correct implementation from the obvious wrong one.
  - A run with no expired step prints nothing (both call sites).
  - `timeout_holds` still increments exactly as before (guard against the counter
    being disturbed).
- [ ] **Step 2: Run them; verify they fail.**
- [ ] **Step 3: Implement at all three sites.** `_maybe_apply_timeout` keeps its current
      signature and stays silent — it is called from inside the two loops, so printing
      there would produce one line per step. Factor the emit into a small helper and call
      it from **all three**: `get_current_step` (`:430-434`), `drain_expired`
      (`:734-742`), and `get_status` (`:710-712`). The first two run the identical
      `while` loop; `get_status` calls `_maybe_apply_timeout()` **once, unlooped**, so
      there it emits at most one line per hold with no volume concern. The helper must
      **not** acquire `_step_lock` — all three callers already hold it, and re-acquiring
      a non-reentrant `threading.Lock` deadlocks. Capture the starting `self.step_index`
      **before** the loop (or before the single call), count holds, and print once
      afterwards when the count is non-zero:

  ```
  ⚠️ decision deadline: auto-held {n} step(s) for {backtest_id}
     (agent={agent_name}, steps={first}..{last}, total_holds={t})
     — these steps are NOT the agent's decisions
  ```

  One request can legitimately produce two lines — `get_step` calls `get_status()` and
  then possibly `get_current_step()`, and each may hold different steps. That is correct
  and the tests should not forbid it; what they must forbid is *zero* lines.
- [ ] **Step 4:** Run the new file, then `test_deadline_and_holds.py` (the existing
      coverage for this path — it already exercises `drain_expired` directly),
      `test_run_lifecycle_unification.py`, `test_protocol_api.py`, then the full suite.
- [ ] **Step 5:** `git status`, commit, PR — `fix: log decision-deadline auto-holds`.

---

## T3 — Sweep terminal legacy sessions

Branch: `fix/evict-terminal-legacy-sessions`

**Why:** `reap_runs()` evicts terminal engine sessions by walking `_runs`, which the
legacy `/api/v1/backtest/*` surface never populates. `MAX_LEGACY_ACTIVE_GLOBAL` cannot
bound them either — `_count_active_locked` (`:919`) skips terminal sessions. Unbounded.

**Files:** modify `dashboard/backend/domain/backtesting/external_run_service.py`,
`dashboard/backend/app.py`, `dashboard/backend/tests/conftest.py`, `.env.example`;
create `dashboard/backend/tests/test_legacy_session_sweep.py`.
⚠ Shares `external_run_service.py` with T2 and `conftest.py`/`.env.example` with T1 —
land this **last** and rebase on merged `main` first (see the Architecture note).

**Scope note — `_sessions` is not legacy-only.** Its single write site is
`_sessions[backtest_id] = session` (`:990`) in `start_backtest` (`:947`), which serves
the legacy route *and* the protocol surface via `run_service.create_run`. The sweep
walks the whole dict anyway, which is correct because `reap_runs` evicts terminal
protocol sessions at `runs/service.py:434-436` **before** invoking registered sweeps at
`:466-470` — so the sweep only ever sees one if that eviction failed, where the TTL is a
backstop. Do not reorder those blocks, and do not add a legacy-only filter expecting it
to be a no-op. v2 is unaffected either way: `execution/backtest_backend.py:74-78`
constructs its session directly and never registers it, so this sweep cannot bound v2.

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
  - **The sweep does not raise through the reaper.** Register it, run a real
    `reap_runs()` pass with a terminal session present, and assert `capsys` contains **no**
    `registered sweep failed` line. Without this, the `AttributeError` of Step 3 is
    swallowed into a warning and every other test in this file still passes.
- [ ] **Step 2: Run them; verify they fail.**
- [ ] **Step 3: Add `self.terminal_seen_at: Optional[datetime] = None` to
      `ExternalBacktestSession.__init__`** (`:184-253`, alongside `self.step_opened_at`
      at `:238`). **The attribute does not exist today** — a sweep that reads
      `s.terminal_seen_at` before anything writes it raises `AttributeError`, and because
      registered sweeps run inside `reap_runs`' `try/except` (`runs/service.py:466-470`)
      that surfaces only as `⚠️ reap_runs: registered sweep failed: …` while the reaper
      keeps reporting healthy. The sweep would silently never work. (A
      `getattr(s, "terminal_seen_at", None)` read is the alternative; prefer the explicit
      `__init__` default so the attribute is discoverable.)
- [ ] **Step 4: Implement `sweep_terminal_sessions()`** in `external_run_service.py`.
      Under `_lock`, iterate `list(_sessions.items())`; skip
      `s.status not in TERMINAL_STATUSES`; stamp `s.terminal_seen_at = _utcnow()`
      (`:117-118`) on first sighting and `continue`; pop once
      `(now - terminal_seen_at) >= LEGACY_SESSION_RETENTION_SECONDS`. Return the count.
      First-sighting rather than stamping in `_finalize`/`cancel` deliberately: it
      cannot be missed by a terminal path that forgets to stamp.
- [ ] **Step 5: Register it** in `app.py` beside the existing
      `register_reaper_sweep(reap_v2_runs)` (the call is at `app.py:220`). No new thread;
      it rides the 60 s reaper pass.
- [ ] **Step 6: Strip `LEGACY_SESSION_RETENTION_SECONDS`** in `tests/conftest.py`
      (the scale-knob block at `:80-93`) **and document it in `.env.example`**:

  ```
  # How long a terminal legacy /api/v1/backtest/* session is kept in memory before
  # the reaper sweep drops it. Reads for an evicted run fall back to the persisted
  # row, so this only buys an in-flight reader some slack. Seconds. Default 300.
  # LEGACY_SESSION_RETENTION_SECONDS=300
  ```
- [ ] **Step 7:** Run the new file, then `test_run_lifecycle_unification.py`,
      `test_architecture_boundaries.py`, then the full suite.
- [ ] **Step 8:** `git status`, commit, PR — `fix: evict terminal legacy backtest sessions`.

---

## T4 — Fix the load-test harness, and make a wholesale baseline failure loud

Branch: `fix/loadtest-harness-baselines`

**Why:** `stress_serve.py:60` patches `create_market_data_provider` with a one-argument
lambda against a two-argument signature, so every baseline through every rung failed
and was swallowed as a warning. The harness is the instrument T5 depends on.

**Files:** modify `dashboard/scripts/loadtest/stress_serve.py`,
`dashboard/scripts/loadtest/drive_agents.py`, `dashboard/scripts/loadtest/README.md`,
`dashboard/backend/domain/backtesting/baseline_worker.py`; create
`dashboard/backend/tests/test_baseline_failure_visibility.py`.

- [ ] **Step 1:** `stress_serve.py:60` → `lambda *a, **k: FakeAlpacaLoader()`. The real
      `create_market_data_provider(data_source=ALPACA, universe=None)` takes two
      arguments; the current `lambda ds=None:` takes one.
- [ ] **Step 2: Make a wholesale baseline failure loud *in production*, not just in the
      harness.** `_drain_forever` prints `⚠️ Baseline generation failed (run saved): {exc}`
      per job (`baseline_worker.py:100`) inside `except (Exception, SystemExit)` and
      continues; the module keeps **no failure counter at all** — `_completed`
      (`:56-61`) records only successes. That per-item warning is exactly why F4 survived
      a whole ladder unnoticed, and fixing it only in `stress_serve.py` leaves prod as
      blind as the harness was: an upstream break failing *every* baseline would still be
      a stream of warnings nobody reads. Repo rule: **a per-item warning cannot report a
      total contract break.**
  - Add a consecutive-failure counter to `baseline_worker`, reset on any success. On
    crossing a small threshold, print one unmissable escalation line naming the count and
    the last exception. `print()`, not `logger` — logger output is invisible under
    deployed uvicorn.
  - Test it: monkeypatch `_run_job` to always raise, submit N jobs, assert via `capsys`
    that the escalation line appears exactly once and that a subsequent success resets
    the counter.
- [ ] **Step 3: Then** have `stress_serve.py` count `Baseline generation failed`
      occurrences and print a summary banner at shutdown, so a future *harness* break
      cannot be mistaken for a clean run.
- [ ] **Step 4: Add `--windows shared|distinct` to `drive_agents.py`** (default
      `shared`). `shared` gives every agent one date range; `distinct` gives each its
      own. This is the switch F5's diagnosis needs and the difference between one
      queued baseline backtest and N serialized ones.
- [ ] **Step 5: Update the README** — document both flags and state plainly which figures
      are floors: anything produced by the **unrepaired** harness understates both CPU and
      RSS, because the baseline worker's `HourlyBacktester` never allocated. The
      2026-08-18 ladder rungs are in that category; the fresh 100-agent run (0.522 CPU-s,
      311 MB) was taken after an ad-hoc local repair and is not. Say so explicitly, so a
      later reader does not average the two.
- [ ] **Step 6: Smoke-run at 10 agents both ways**, confirming baselines now complete
      (no warning banner) and that `distinct` is visibly slower than `shared`.
- [ ] **Step 7:** `git status`, commit, PR — `fix: repair baseline patching in the load-test harness`.

---

## T5 — Validation against a real Render instance

> ## ✅ EXECUTED 2026-08-18 — but **not against Render**, and the answers moved
>
> The capacity question is **answered**; the steps below are kept for the record and
> annotated with what actually happened. Results live in **spec §5**.
>
> **Why not Render.** The harness cannot be pointed at a deployed instance. Every
> hermetic property is an in-process monkeypatch inside `stress_serve.py`, not client
> configuration: `:66` swaps `create_market_data_provider` for `FakeAlpacaLoader`;
> `:17-20` redirects `DATABASE_PATH` and pops the Postgres URLs; `:70-79` seeds the
> agents and writes the keys `drive_agents.py:45` requires. `--allow-remote`
> (`drive_agents.py:41-42`) only relaxes a hostname guard — it supplies none of that.
> Aimed at Render the run would register 100 junk agents in the prod content DB, hit
> **real Alpaca** (the create payload carries only dates, so the server default feed
> applies; the sole synthetic provider, `vnpy_simulation`, is absent from the image
> because `Dockerfile:9` installs `requirements.txt` only), and write into the real
> run-history store — while loading **prod**, the only Render service there is. That
> measures a different system than §5 describes.
>
> **What was run instead.** The same harness, unmodified, with the *server* confined to
> each tier's CPU budget on the dev box (`taskset -c 0` ≈ Standard 1.0 CPU;
> `systemd-run --user --scope -p CPUQuota=10%` ≈ Free 0.1 CPU) and the driver pinned to
> other cores. That reproduces the one variable that matters while keeping every
> hermetic property intact. Ladders at N = 12/16/20/25/50/100.
>
> **The premise also changed.** T5 was scoped to a 100-agent *burst*. The operator's
> actual plan is ~12 hosted agents **sustained**, growing with users — see spec §1.
> §5 now answers that question.
>
> **Three things this found that the plan did not anticipate:**
> 1. `timeout_holds == 0` — a Step 3 acceptance criterion — was **structurally
>    unfalsifiable**. Fixed in `drive_agents.py` before any verdict was recorded.
> 2. Free fails by **SQLite write-lock timeouts**, not by the predicted deadline breach,
>    and it fails between 16 and 25 concurrent runs rather than at 100.
> 3. Per-run CPU is a property *of the tier*, not of the code (0.31 CPU-s dedicated vs
>    0.57 throttled), which is why Step 3's "~52 s prediction" was wrong.

**Not a code change. Do not start it until T1–T4 have merged.**

Every number in the spec comes from a 12-core dev box plus arithmetic. Nothing in this
workstream has ever run on Render.

- [~] **Step 1:** ~~Deploy `main` (with T1–T4) to a Render **Standard** instance.~~
      **Not done — and should not be.** Superseded by CPU-limited emulation; see the
      EXECUTED block above. The only Render service is prod.
- [ ] **Step 2:** Run the fixed harness at **25 agents** first, `--windows shared`.
      Record wall time, failures, `timeout_holds`, **create p95, decision p95, RSS at
      start and at end** (growth, not just peak — see Step 3).
- [ ] **Step 3:** If 25 is clean, run **100 agents**, `--windows shared`.
      **Acceptance:** zero failures, `timeout_holds == 0`, RSS below the instance
      ceiling, wall time within ~2× the ~52 s prediction — **plus the three predecessor
      criteria that were never captured** (`plans/2026-07-24-…md:2729`): `create p95 <
      1000 ms`, `decision p95 < 1000 ms`, `RSS growth < 100 MB`. This is the first
      environment where they mean anything and the first run whose baselines actually
      execute, so this is what finally closes Task 12 Step 2. Task 12's own rule applies:
      if a criterion misses, diagnose — do not relax it.
      **→ Run under CPU-limited emulation instead (spec §5). Standard: 100/100, zero
      failures, zero locks, RSS +89 MB. Two corrections to this step's own text: the
      `timeout_holds == 0` criterion could not fail as written and had to be fixed first,
      and the "~52 s prediction" it measures against was itself wrong (per-run CPU is a
      property of the tier). Task 12's rule was followed — the `create p95` miss was
      diagnosed as burst-start queueing, not relaxed.**
- [ ] **Step 4:** Run **100 agents `--windows distinct`** once, purely to measure the
      baseline-worker serialization from F5. Expected to be materially slower; that is
      the datum, not a failure.
- [ ] **Step 5: Run the predecessor's outstanding Step 3 prod smoke** — one 100-agent
      smoke against **prod** with a throwaway config window, watching the startup lines
      (`market-data dataset built`, `pg pool created`) and `timeout_holds` in results
      (`plans/2026-07-24-…md:2731-2733`). Manual/observational, no repo change. This is
      the last outstanding piece of Task 12.
- [ ] **Step 6: Record results** in this plan's status header **and update the
      predecessor's line-3 header** from partial to closed — only once Steps 3 and 5 have
      both passed. **If they contradict the spec's §5 table, the spec is wrong** —
      correct §5 rather than explaining the measurement away.
- [ ] **Step 7:** Only then decide whether the deferred §4 optimization is needed.

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
