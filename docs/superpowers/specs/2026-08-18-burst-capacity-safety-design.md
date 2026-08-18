# Burst Capacity & Safety — Design

**Date:** 2026-08-18
**Predecessor:** `docs/superpowers/specs/2026-07-24-agent-scale-sustainability-design.md`
(plan: `docs/superpowers/plans/2026-07-24-agent-scale-sustainability.md` — four tiers
merged as #208/#209/#211/#212; its **Task 12 acceptance run was executed 2026-08-18**,
results below).

---

## 1. What this is

The predecessor made 100 concurrent protocol agents *survivable*. This spec covers the
residue that the acceptance run surfaced: four verified defects, and the hosting decision
for a **100-agent burst** (a demo/launch moment, not sustained load).

**Explicitly out of scope, by decision:** per-run CPU optimization. It was measured
(§4) and is real, but the burst target does not need it. Recorded here so a later
reader does not re-derive it.

## 2. Measured baseline (2026-08-18)

The predecessor's acceptance criteria were never run until now. They pass.

| Measurement | Result |
|---|---|
| 100 agents, fresh process | **35.6 s wall, 100/100 completed, 0 failures**, worst request 5.6 s |
| `timeout_holds` | **0 in every rung** (1 → 100 agents) — the #208–#212 deadline fix is validated |
| Peak RSS | 311 MB (never above 360 MB in any configuration) |
| CPU per run (ladder, over HTTP) | 0.406 / 0.412 / 0.432 / 0.420 CPU-s across a 100× concurrency range — **linear** |
| CPU per run, working figure | **~0.47 CPU-s** — the ladder ran with baselines silently disabled (F4), so its numbers are a floor; 0.47 is the value used for every tier calculation below |

**Layer decomposition** (same 21-step run, three ways):

| Layer | CPU/run |
|---|---|
| Engine only (direct calls, no ASGI) | 32.5 ms |
| Full ASGI stack (routing, deps, validation, JSON) | 293 ms |
| Real HTTP under concurrency | ~470 ms |

The engine is ~7% of the bill. Attribution of the ASGI run: pandas 42%,
FastAPI + Starlette + pydantic + json **~8% combined**. The cost is not the web
framework; it is work the request path does over pandas objects.

### Correction to prior records

Two things previously recorded in session memory are wrong and are corrected here:

- **`market_data_store.py:158` prints a misleading per-dataset size.** A scan read it
  as "~50 MB per month-long DJIA-30 dataset", and the predecessor plan repeats that
  figure at `plans/2026-07-24-...md:615` to justify `MARKET_DATA_CACHE_MAX_ENTRIES=4`.
  Measured: **~1.7 MB**. The cache sizing is still fine; the stated reason is not.
- **There is no process-age leak on the protocol surface.** 200 sequential protocol
  runs: CPU/run flat (251 → 248 ms), RSS plateaus at 292 MB, `_sessions`/`_runs`
  sawtooth as the reaper reclaims them, gc object count flat. The 181 s aged-process
  result from the acceptance run is real but its **mechanism remains unidentified** —
  see F5.

## 3. Findings — verified at source

**F1. No HTTP timeout anywhere in the Alpaca fetch chain.**
`alpaca_bars.py:220` constructs `StockHistoricalDataClient(api_key, secret_key)`.
That constructor exposes no timeout parameter, and alpaca-py 0.43.2's
`RESTClient._one_request` calls `self._session.request(method, url, **opts)` with no
timeout in `opts`. `requests` with no timeout blocks forever. One stalled socket
permanently leaks a threadpool thread. **Binds at concurrency ≥ 1**, so it is not a
scale issue at all — it is a standing availability bug.

**F2. Decision-deadline auto-holds are completely silent.**
`external_run_service.py:421 _maybe_apply_timeout` calls
`_advance_step(executable=[], decision_source="timeout_hold")` and returns. No print,
no log, no metric. The step is attributed to a decision the agent never made. The
`timeout_holds` counter exists but is only visible to someone who fetches
`GET /api/v1/runs/{id}` and reads `engine_status`. This is the exact pattern
`CLAUDE.md`'s "Fail-closed is not fail-visible" section exists to prevent.

**F3. Terminal legacy sessions accumulate without bound.**
`reap_runs()` (`domain/runs/service.py:417`) evicts terminal engine sessions — but it
iterates `_runs`, the *protocol* registry. The legacy `/api/v1/backtest/*` surface
writes no `protocol_runs` row and no `_runs` entry, so its sessions are never reached.
`MAX_LEGACY_ACTIVE_GLOBAL=50` does not bound them either: `_count_active_locked`
(`external_run_service.py:919`) counts only `s.status not in TERMINAL_STATUSES`, so a
terminal session is invisible to the very cap that would have limited it. Each retained
session pins its `all_data` reference past LRU eviction.

**F4. The repo's own load-test harness silently disables baselines.**
`dashboard/scripts/loadtest/stress_serve.py:60` patches
`create_market_data_provider` with a **1-argument lambda** while the real signature
takes two. Every baseline generation through the whole ladder failed with
`<lambda>() takes from 0 to 1 positional arguments but 2 were given`, printed as a
warning and swallowed. Any CPU figure produced by the unpatched harness is an
**underestimate**, including the predecessor's acceptance numbers.

**F5. The aged-process tail is real but unexplained.**
The acceptance run measured 35.6 s on a fresh process vs 181 s (5 failures, one 134.9 s
request) on one that had already served 186 runs. §2 refutes the leak hypothesis. The
leading remaining hypothesis is the **baseline worker**: `baseline_worker.py` drains a
queue with a *single* thread and dedups by `(start_date, end_date, mode)`. The
sequential probe used one window, so 199 of 200 jobs hit the dedup cache; the aged
process served runs across many windows, where each distinct window is a full
`HourlyBacktester` run serialized behind that one thread. **Untested.** No fix is
specified for it — diagnosis first.

## 4. Optimization headroom — measured, then deferred

Recorded so it is not re-litigated. `_market_data_at` is called **147 times per run**
(7 per step) and returns rows derived from the shared read-only dataset, so its result
is identical for every run on that window.

| Variant | CPU/run (ASGI) | vs shipped |
|---|---|---|
| As shipped | 293 ms | 1.00× |
| + memoise `_market_data_at` per `(dataset, timestamp)` | 204 ms | 1.44× |
| + memoised rows as plain dicts rather than pandas Series | 194 ms | **1.51×** |

Both variants produced **bit-identical step observations and final metrics** across 15
runs. A further candidate exists (`sqlite3.connect()` per operation — 104 connections
per run, `database.py:62` and `:119`) and is unmeasured.

**Why it is deferred:** 1.51× moves free tier from ~7.6 to ~11.5 sustained agents.
It does not reach 100 by any path. For the burst target it is unnecessary (§5).

## 5. Hosting decision

Per-run cost is ~0.47 CPU-s over HTTP. For a **burst** of 100 agents running once:

| Tier | CPU | Burst wall time | Worst request (extrapolated) | Verdict |
|---|---|---|---|---|
| Free | 0.1 | ~470 s | **~82 s** | **Breaches the 60 s deadline** → silent auto-holds |
| Standard | 1.0 | ~47 s | ~8 s | **Adequate as-shipped** |

The deadline, not throughput, is the failure boundary: a step served later than
`EXTERNAL_AGENT_DECISION_TIMEOUT_SECONDS` (60) is auto-held and attributed to the agent
anyway (F2). Free tier is expected to breach it under a 100-agent burst; Standard is not,
**without any CPU work**. This is what makes deferring §4 safe.

Sustained 100 agents is a different question and is not answered here.

**Horizontal scaling stays closed.** Run state is module-level (`_sessions`,
`external_run_service.py:67`; `_runs`, `runs/service.py:49`) and the heartbeat path
*fails* orphaned runs rather than migrating them. Every tier above Standard buys
nothing until run state leaves process memory.

## 6. Design

Five work items. T1–T3 are behaviour changes; T4–T5 restore trust in the instrument
and then use it.

### T1 — Default HTTP timeout on the Alpaca client

Wrap the client's `requests.Session.request` immediately after construction, injecting
`timeout` when the caller did not supply one.

```python
ALPACA_HTTP_TIMEOUT_SECONDS = float(os.getenv("ALPACA_HTTP_TIMEOUT_SECONDS", "60"))
ALPACA_HTTP_CONNECT_TIMEOUT_SECONDS = float(
    os.getenv("ALPACA_HTTP_CONNECT_TIMEOUT_SECONDS", "10"))
```

The wrapper is idempotent (guarded by an attribute flag) and **prints a warning if it
cannot find `_session`** — an upstream rename must not silently restore the unbounded
behaviour, which is the same failure class F2 describes.

### T2 — Make deadline auto-holds visible

`get_current_step` drains expired steps in a `while` loop, so one poll can apply many
holds. Count them in the loop and emit **one** line after it, never one per step
(21 steps × 100 agents = 2,100 lines otherwise):

```
⚠️ decision deadline: auto-held N step(s) for <backtest_id> (agent=<name>,
   step_index=<i>, total_holds=<t>) — these steps are NOT the agent's decisions
```

Rationale is integrity, not diagnostics: a published curve containing auto-held steps
is not the agent's curve, which is the same concern the H6 guard exists for.

### T3 — Sweep terminal legacy sessions

Add `sweep_terminal_sessions()` to `external_run_service.py`, registered in `app.py`
next to `register_reaper_sweep(reap_v2_runs)` (`app.py:219`) so it runs on the existing
60 s reaper pass. No new thread.

Retention TTL (`LEGACY_SESSION_RETENTION_SECONDS`, default `300`) rather than immediate
eviction: a client may still be reading a completed run. The clock starts at the first
sweep that observes the session terminal, so no `_finalize`/`cancel` path can miss it.
Reads for an evicted run already fall back to the persisted row — `evict_session`'s own
docstring carries that safety argument.

### T4 — Fix the load-test harness

`stress_serve.py:60` → `lambda *a, **k: FakeAlpacaLoader()`. Add a `--windows
shared|distinct` flag to `drive_agents.py`: shared collapses baseline work to one job
via the dedup cache, distinct forces N jobs through the single worker thread. That
switch is what F5's diagnosis needs, and it is the difference between a cheap and an
expensive demo (§7).

### T5 — Validation against a real Render instance

Nothing in this workstream has ever run on Render. Everything in §2 and §5 is a
12-core dev box plus arithmetic. Deploy to Standard, run the fixed harness at 100
agents, and assert: zero failures, `timeout_holds == 0`, RSS below the instance
ceiling. **If T5 disagrees with §5, §5 is wrong**, not T5.

## 7. Operational note — share the window

`baseline_worker` dedups by `(start_date, end_date, mode)` behind one thread. 100 demo
agents on **one** window queue a single baseline backtest; 100 agents on **distinct**
windows queue 100 serialized ones. Give the demo agents a shared date range.

## 8. Architecture review

| | | |
|---|---|---|
| Q0 principle | Additive safety on a working design; no new subsystems, no new routes | **PASS** |
| Q1 scalability | Bounded by CPU, linear, measured; no new bottleneck introduced | **PASS** |
| Q2 customizability | Every new knob is an env var with a documented default | **PASS** |
| Q3 failure story | T1 bounds a hung socket; T2 makes a corrupt-provenance event audible; T3 bounds retention | **PASS** |
| Q4 observability | T2 is the fix; T5 is the only real-environment evidence and does not exist yet | **CONCERN** — accepted, T5 closes it |
| Q5 cost | $25/mo Standard, at demo time only. §4 deferred with measurements recorded | **PASS** |
| Q6 trust boundaries | Unchanged. No new routes, no auth changes, no new unauthenticated surface | **PASS** |
| Q7 reality check | T1–T4 each touch one file; T3 reuses the existing sweep hook rather than adding a thread | **PASS** |

**Planning call:** the walking skeleton is specable — every box, contract and failure
mode is named. Build it. F5 stays a diagnosis task, not an implementation task,
precisely because its mechanism is unidentified.

## 9. Constraints inherited from the predecessor

- **Wire contract frozen.** No new status literals; new payload *keys* are fine.
- **No new HTTP routes** — three route-contract freeze tests must pass untouched.
- **`print()`, not `logger`** — logger output is invisible under deployed uvicorn.
  Assert with `capsys`, never `caplog`.
- **Env vars read once at import**; every new one stripped in `tests/conftest.py`.
- `domain/` must not import `api/` or `app.py`.
- Do not modify the committed seed `dashboard/storage/data/backtest.db`; `git status`
  after any session that imports backend modules.
- Branch per item, cut from up-to-date `origin/main`. Merging to `main` auto-deploys
  prod.
