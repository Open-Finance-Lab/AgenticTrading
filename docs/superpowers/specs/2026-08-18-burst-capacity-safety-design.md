# Burst Capacity & Safety — Design

**Date:** 2026-08-18
**Predecessor:** `docs/superpowers/specs/2026-07-24-agent-scale-sustainability-design.md`
(plan: `docs/superpowers/plans/2026-07-24-agent-scale-sustainability.md` — four tiers
merged as #208/#209/#211/#212; its **Task 12 acceptance run was partially executed
2026-08-18** — 3 of 6 Step 2 criteria captured, Step 3 prod smoke not run — results
below).

---

## 1. What this is

The predecessor made 100 concurrent protocol agents *survivable*. This spec covers the
residue that the acceptance run surfaced: four verified defects, and the hosting decision
for a **100-agent burst** (a demo/launch moment, not sustained load).

⚠ **The burst framing was wrong about the product, and §5 has been re-scoped.** The
operator's actual plan is to **keep roughly a dozen agents hosted continuously and grow
that number as users arrive** — sustained load, not a launch moment. That is a different
question from the one this spec originally asked, and it has a different answer: §5 now
carries a measured Free-tier ceiling for *sustained concurrency* rather than an
arithmetic burst estimate. Read §5, not this paragraph, for the hosting decision.

**Explicitly out of scope, by decision:** per-run CPU optimization. It was measured
(§4) and is real, but the burst target does not need it. Recorded here so a later
reader does not re-derive it.

## 2. Measured baseline (2026-08-18)

The predecessor's Task 12 acceptance was executed for the first time on 2026-08-18.
**Three of its six Step 2 criteria are met, three were never captured, and its Step 3
prod smoke was not run at all.** The gate is partially closed, not closed.

| Task 12 Step 2 criterion (`plans/2026-07-24-…md:2729`) | Status |
|---|---|
| 0 server `timeout_holds` | **Met** — 0 in every rung |
| 0 FAILURES | **Met** — 100/100 completed on a fresh process |
| total wall < 60 s | **Met** — 35.6 s |
| create p95 < 1000 ms | **Not captured** — only per-*run* p95 and worst-request were recorded |
| decision p95 < 1000 ms | **Not captured** — same |
| RSS growth < 100 MB | **Not captured** — *peak* RSS was recorded, which is a different quantity |

Task 12 states: "If any criterion misses, diagnose before merging (systematic-debugging)
— do not relax the criteria." Three unmeasured criteria are not three met criteria, so
they are carried into T5 (§6) rather than written off here. Step 3 (post-deploy prod
smoke) is likewise still outstanding — nothing in this workstream has run on Render.

| Measurement | Result |
|---|---|
| 100 agents, fresh process | **35.6 s wall, 100/100 completed, 0 failures**, worst request 5.6 s |
| `timeout_holds` | **0 in every rung** (1 → 100 agents) — ⚠ **not evidence**, see below |
| Peak RSS | 311 MB on the fresh 100-agent run (never above 360 MB in any configuration) |
| CPU per run (ladder, over HTTP) | 0.440 / 0.406 / 0.412 / 0.432 / 0.420 CPU-s at 1 / 10 / 25 / 50 / 100 agents — **linear** across a 100× concurrency range |
| CPU per run, working figure | **0.522 CPU-s** — the *only* run whose baselines actually executed (the ladder rungs ran with baselines silently disabled by F4, so 0.406–0.440 is a floor). 0.522 is the value used for every tier calculation below |

⚠ **Which rows carry the F4 caveat, precisely.** The harness bug was repaired before the
fresh 100-agent run — that repair is *why* its CPU/run is 0.522 rather than the ladder's
~0.42 — so the 0.522 and 311 MB figures are the ones taken with baselines actually
executing (`baseline_worker._run_job` building an `HourlyBacktester` and running two
baselines, `baseline_worker.py:105-129`). The **ladder rungs** are the floors: their CPU
(0.406–0.440) *and* their RSS were measured with baselines silently failing. Do not
extend the F4 discount to 0.522/311 MB, and do not treat the ladder's numbers as
comparable to them.

⚠ **The `timeout_holds` row was never evidence, and this was found the hard way.** Until
2026-08-18 the driver folded three distinct *unknowns* into the same `0` that means "no
holds" — a non-200 on the final GET, a swept live session (`engine_status` is `None` once
it is gone), and an absent field. That read has been there since the harness's first
commit (`11a102b`), so **every rung above, and the predecessor's whole ladder, reported a
number that could not fail.** It reads zero hardest under the load that produces holds: a
Free-tier run printed `0` while the server log carried **7**. Fixed in
`drive_agents.py` (unknown now prints as `unknown`, never `0`).

This does not mean the #208–#212 deadline fix is broken — on an unsaturated 12-core box
those zeros were most likely true. It means these runs could not tell a working fix from
an unreadable counter, so they never validated it.

**Auto-holds have three instruments, and none of them is complete.** Rank them and take
the largest:

1. **Client-observed deadline losses** (`drive_agents.py:152-155`) — counted only when
   the server answers a decision with **409 + "deadline"/"finalized"**, i.e. the server
   itself reporting the auto-hold. Authoritative; trust first.
2. **T2's `decision deadline` log lines** — a **lower bound**. It misses the path where
   `get_status` applies the hold *before* the instrumented loop in `get_current_step`
   runs, leaving that loop no delta to report. Measured in §5: a 25-agent Standard run
   logged **0** while the client observed **1**. Issue #375 is a separate second blind
   spot (the fourth deadline branch in `submit_decisions`).
3. **The `timeout_holds` counter** — weakest, and at load it is `unknown` for essentially
   every run.

⚠ An earlier draft of this section said "when they disagree, believe the log." That was
written before the Standard 25-agent run showed the log *under*-counting, and is wrong.
The log is a floor, not a truth.

RSS nonetheless stays unsettled, for different reasons: 311 MB is **one run**, on a
12-core dev box, against `FakeAlpacaLoader`/`synth_bars` synthetic frames rather than real
Alpaca DJIA-30 bars, on a **single shared window**. Many distinct windows and a 512 MB
instance are both untested. That is what §8 Q1 and T5 treat as open — not an F4 discount.

**Layer decomposition** (same 21-step run, three ways):

| Layer | Figure |
|---|---|
| Engine only (direct calls, no ASGI) | 32.5 ms CPU/run |
| Full ASGI stack (routing, deps, validation, JSON) | 293 ms CPU/run |
| Real HTTP under concurrency | ~470 ms — **per-request latency, not CPU/run** |

⚠ That last row is the trap that produced the 0.47 error corrected below. It is a
*latency*, and the per-run CPU figure it was mistaken for is **0.522**. Do not read down
this column as if all three rows were the same quantity.

The engine is **~11%** of the bill — 32.5 ÷ 293, the only two rows that are the same
quantity. ⚠ This was previously recorded as "~7%", which is 32.5 ÷ 470: the *same*
latency-as-CPU conflation the warning above describes, surviving one sentence further
down. Do not restore it, and do not divide 32.5 by 0.522 either — the 0.522 run had
baselines executing and the 293 ms profile did not, so they are not the same scope.
Attribution of the ASGI run: pandas 42%,
FastAPI + Starlette + pydantic + json **~8% combined**. The cost is not the web
framework; it is work the request path does over pandas objects.

### Correction to prior records

Three things previously recorded are wrong and are corrected here:

- **The "~50 MB per month-long DJIA-30 dataset" figure is wrong.** It originates in a
  *source comment*, `dashboard/backend/domain/backtesting/market_data_store.py:34-37`,
  which justifies `MARKET_DATA_CACHE_MAX_ENTRIES=4` by "4 entries caps the worst case at
  ~200 MB". The predecessor plan reproduces that comment verbatim inside a fenced block
  at `plans/2026-07-24-…md:614-618`, so **the doc and the shipped code carry the same
  wrong number** — correcting only the plan leaves it live in the file a future capacity
  reviewer opens (T0 Step 3 fixes the source comment for that reason).
  Measured: **~1.7 MB**. The cache sizing is still fine; the stated reason is not.
  Two limits on that measurement, both of which make it a floor:
  its only source is the per-build print at `market_data_store.py:157-159`, whose size
  expression (`:157`) is `sum(df.memory_usage(deep=True) …) for df in all_data.values()`
  — it counts the
  `all_data` frames and **not** the `timestamps` list or the `price_cache`
  (`Dict[symbol, Dict[timestamp, float]]`, `:192-205`), both of which the cached
  `MarketDataset` also retains (`:46-57`); and it was taken under the load harness, which
  substitutes `FakeAlpacaLoader`/`synth_bars` (`stress_serve.py:27,48`) for real bars, so
  it is a *synthetic* window rather than a real Alpaca DJIA-30 one. It refutes ~50 MB
  comfortably; it is not a settled per-dataset budget.
- **The working CPU/run figure was 0.47; it should be 0.522.** 0.47 corresponds to no
  per-run measurement: it is above the ladder floor (0.406–0.440) and below the only run
  whose baselines executed (0.522). It appears to have been lifted from the
  layer-decomposition row below — "Real HTTP under concurrency ~470 ms" — which is a
  per-request *latency*, not a per-run *CPU-second*. The error propagated into §5's
  burst-wall column while that table's worst-request column was computed from 0.522,
  leaving the two columns mutually inconsistent. §5 now uses 0.522 throughout.
  It propagated a **second** time, into the engine's share of the bill above
  ("~7%" = 32.5 ÷ 470, corrected to ~11% = 32.5 ÷ 293). Both consumers of 470 have now
  been found and fixed; the lesson is that one conflated figure is rarely quoted once,
  so correcting the number is not the same as correcting the document.
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

**F2. Decision-deadline auto-holds emit nothing at the moment they happen.**
`external_run_service.py:421 _maybe_apply_timeout` calls
`_advance_step(executable=[], decision_source="timeout_hold")` and returns. No print,
no log, no metric. The step is attributed to a decision the agent never made.

Be precise about what *is* already there, so the fix does not rebuild it: the
`timeout_holds` **count** is durable and well surfaced — `_finalize` writes it into
`agent_runs.metadata` (`:646`) and it is read back in `build_final_metrics` (`:176`) and
`get_run_result` (`:1112`), and served live from `get_current_step` (`:450`) and
`get_status` (`:721`). The gap is narrower and worse than "invisible":

- **Nothing is emitted at hold time**, so a hold is only discoverable by someone who
  already suspects one and goes looking for the number.
- **The count is an aggregate.** It says three steps were auto-held; it never says
  *which* three, so a published curve cannot be reconciled against the log after the fact.
- **The reaper path is silent too** — `drain_expired` (`:734-742`) runs the identical
  `while self._maybe_apply_timeout():` loop, and that is the path with no agent watching
  (see T2).

This is the pattern `CLAUDE.md`'s "Fail-closed is not fail-visible" section exists to
prevent: the state is recorded, but nothing announces the event that produced it.

**F3. Terminal legacy sessions accumulate without bound.**
`reap_runs()` (`domain/runs/service.py:417`) evicts terminal engine sessions — but it
iterates `_runs`, the *protocol* registry. The legacy `/api/v1/backtest/*` surface
writes no `protocol_runs` row and no `_runs` entry, so its sessions are never reached.
`MAX_LEGACY_ACTIVE_GLOBAL=50` does not bound them either: `_count_active_locked`
(`external_run_service.py:919`) counts only `s.status not in TERMINAL_STATUSES`, so a
terminal session is invisible to the very cap that would have limited it.

**Size the harm honestly.** `adopt_dataset` (`:268`) assigns *references* into the shared
cached dataset — `self.all_data = dataset.all_data` (`:279-281`, likewise `timestamps`
and `price_cache`) — and its own docstring says so outright: *"SHARED + READ-ONLY:
all_data/timestamps/price_cache belong to the store and other sessions."* Both entry
paths reach it: `load_market_data` (`:259`) and `start_backtest`'s peek fast path
(`:999`). So N terminal sessions on one window pin **one** dataset, not N.
The market-data bound is therefore (distinct windows × dataset size), and the correction
above revises that size from ~50 MB to ~1.7 MB. The genuinely unbounded growth is the
**per-session** state: `decision_log`, `last_executed`, `context_ref_by_step`, the
`PortfolioManager` and its trades (`:216-241`). That is small per session and never
reclaimed, which makes this a slow leak on a long-lived instance rather than the
fast one a per-session dataset copy would imply. It is worth fixing for exactly that
reason — an instance that is never restarted has no other floor — but it should not be
prioritised above T1/T2 on a memory argument the same document has just refuted.

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
per run, `database.py:119` in `_get_connection`) and is unmeasured. The *other*
`sqlite3.connect()` in that file, `:62`, is **not** a candidate: it is inside
`enable_wal()`, runs once per `BacktestDatabase.__init__` (`:112`) and closes in its own
`finally` (`:66`). Its docstring calls it the single definition that keeps the two call
sites from drifting — do not fold it into any connection-pooling change.

**Why it is deferred:** 1.51× moves free tier from ~7.6 to ~11.5 sustained agents.
It does not reach 100 by any path. For the burst target it is unnecessary (§5).

⚠ **Both of those agent counts are wrong** — they are `0.522 ÷ 0.1` arithmetic, the
method §5 falsified. **Measured**, Free clears **20** concurrent active runs before the
first failure, not 7.6. The deferral still holds, but not for the reason given: the
optimization was sized as the thing that would lift Free from "under the operator's
target" to "just over it", and Free was already over it. What actually bounds Free is
SQLite write-lock contention under CPU throttling (§5), which shaving 33% off per-run
CPU does not address — the lock is held across a *freeze*, not across compute. Do not
revive this optimization as a way to raise Free's agent ceiling.

## 5. Hosting decision — MEASURED 2026-08-18

> **This section was arithmetic and it was wrong in both directions.** It predicted
> Standard at ~52 s (measured 26–49 s), Free at ~520 s (measured 702 s), and — more
> importantly — the **wrong failure mode at the wrong load**. Replaced with measurement.
> The method it used, `CPU-seconds ÷ CPU budget`, must not be reused; see *Why the
> arithmetic failed*.

### What is being sized

**N is the number of concurrently *active runs*, not registered agents.** A dozen hosted
agents that each step hourly are a negligible load; a dozen agents all backtesting at the
same moment is the worst case, and that is what N means below.

The operator's target is **~12 sustained, growing with users** (§1) — not the 100-agent
burst this spec was originally scoped to.

### Method

The hermetic harness (`dashboard/scripts/loadtest/`), unmodified, on merged `main`
(`e8473fe`), `--windows shared`, with the **server** confined to each tier's CPU budget
and the driver pinned to other cores:

- **Standard** ≈ `taskset -c 0` (1.0 CPU)
- **Free** ≈ `systemd-run --user --scope -p CPUQuota=10%` (0.1 CPU); quota enforcement
  verified first with a busy-loop probe (0.52 CPU-s over 5 s wall = 10.4%)

Server CPU read directly from `/proc/<pid>/stat` (process-wide, all threads). Not Render
itself — the harness cannot be pointed at a deployed instance; see the plan's T5 block.

### Free tier (0.1 CPU) — the ceiling is 20

| N | wall | server CPU | CPU/run | util of quota | completed | failures | `database is locked` | holds (log / client) |
|---|---|---|---|---|---|---|---|---|
| 12 | 68.9 s | 6.9 | 0.575 | 100% | **12/12** | **0** | 0 | 0 / 0 |
| 16 | 89.1 s | 8.9 | 0.556 | 100% | **16/16** | **0** | 0 | 0 / 0 |
| 20 | 141.0 s | 11.2 | 0.560 | 79% | **20/20** | **0** | 0 | 0 / 0 |
| 25 | 141.3 s | 14.0 | 0.560 | 99% | 22/25 | **3** | **6** | 0 / 0 |
| 50 | 313.6 s | 28.3 | 0.566 | 90% | 36/50 | **14** | **26** | 2 / 1 |
| 100 | 702.4 s | ~70 † | ~0.70 † | ~100% † | 81/100 | **19** | **34** | 7 / 7 |

† The N=100 row's CPU is **inferred** from wall × quota; that run predates the direct
`/proc` instrumentation. Every other row is measured.

**Clean through 20 concurrent; first failures at 25.** But "clean" is not "comfortable":
the server runs **at or near its full 0.1 CPU quota across the whole range**, so there is
no headroom for dashboard traffic, the leaderboard refresh, or the baseline worker.
Latency is already seconds — `decision` p95 2.0 s at N=12, 3.7 s at N=20.

The two rungs below 100% are the informative ones. N=50 (90%) and N=20 (79%) are *not*
spare capacity: both lost wall-clock to blocking rather than compute — N=50 to lock waits
(26 of them), N=20 to one long stall (below). CPU going idle on this stack is a symptom,
not headroom.

RSS stayed 276–303 MB across the whole range, so the 512 MB ceiling is **not** the
constraint at any point. This failure is purely CPU.

### Standard tier (1.0 CPU)

| N | wall | server CPU | CPU/run | util | completed | failures | locks | create p95 | decision p95 |
|---|---|---|---|---|---|---|---|---|---|
| 12 | 77.5 s | 3.7 | 0.308 | 4.8% | **12/12** | **0** | 0 | 529 ms | 120 ms |
| 25 | 73.9 s | 6.8 | 0.272 | 9.2% | **25/25** | **0** | 0 | 259 ms | 269 ms |
| 100 | 48.8 s | 31.0 | 0.310 | 63.5% | **100/100** | **0** | 0 | 1494 ms | 1014 ms |

**Zero failures and zero lock errors at every rung, including 100.** At N=100 the server
used 63.5% of one CPU, so even the burst case leaves headroom. An earlier, less
instrumented run of the same configuration finished in 26.6 s with `create` p95 1157 ms
and `decision` p95 884 ms; the difference is contention for core 0 with other work on the
dev box, which makes this emulation a **floor** for Standard rather than a ceiling.

Wall time here is dominated by an outlier stall, not by CPU — see *The ~68 s stall*.

### Why the arithmetic failed

**Per-run CPU is a property of the tier, not of the code.** Measured directly:

| | CPU-s per run |
|---|---|
| 1.0 CPU, dedicated (`taskset`) | **0.31** |
| 0.1 CPU, throttled (`CPUQuota`) | **0.57** (≈1.9×) |
| 12-core dev box (the figure §2 used) | 0.522 |

A cgroup quota does not slow a process down smoothly — it **freezes** it for ~90 ms of
every 100 ms period, including mid-transaction and mid-critical-section. Throttled CPU
therefore does less useful work per CPU-second than dedicated CPU. Dividing one
per-run constant by each tier's budget assumes exactly the opposite, so it overestimated
the fast tier (0.522 vs the true 0.31) and underestimated the throttled one (0.522 vs the
true 0.57). The 12-core figure happened to sit between them, which is why the result
looked plausible.

Use each tier's own measured per-run cost, or measure the tier.

### The failure mode is not the one that was predicted

§5 predicted Free would fail by **breaching the 60 s decision deadline → silent
auto-holds**. It does not. It fails by **SQLite write-lock timeouts**, at a quarter of
the predicted load, with *zero* auto-holds at the threshold:

```
sqlite3.OperationalError: database is locked
  → database.py:645 insert_run          (via external_run_service _finalize → _advance_step → submit_decisions)
  → domain/runs/repository.py:433 finalize_step   (via runs/service.py:971 submit_decision)
```

Both sit on the agent's decision path (`api/routers/runs.py:157`) and both carry a ~5 s
busy timeout (`repository.py:79` explicitly; `database.py:119` via Python's default). A
writer frozen by the quota still holds the lock, so every other writer exhausts its
timeout and the agent gets a bare **500**.

**This is throttling contention, not concurrency contention** — 100 concurrent writers on
a full CPU produce **zero** lock errors, while 25 on a tenth of a CPU produce six.

⚠ **On prod the two sites diverge, and the hotter one cannot be relieved.** `insert_run`
moves to Postgres when `AGENT_RUNS_DATABASE_URL` is set. `finalize_step` writes
`protocol_steps`, which has **no Postgres twin** (`domain/runs/` has no `*_postgres.py`)
and always lives on local SQLite — and it runs once per *step*, ~21× more often than
`insert_run` runs per *run*. Configuration cannot move the hot site.

### The ~68 s stall — reproduced, unexplained

Three runs across both tiers showed a single request stalling for a fixed ~68–70 s while
the server was **idle**:

| run | stalled endpoint | duration | server util |
|---|---|---|---|
| Standard N=12 | `steps/next` | 68.6 s | 4.8% |
| Standard N=25 | `decision` | 69.6 s | 9.2% |
| Free N=50 | `steps/next` | 68.2 s | 9.0% |

At Standard N=12, eleven of twelve runs finished in ~2.9 s and one took 77.4 s, with only
3.7 CPU-s consumed in total — so this is a **wait, not work**, and the near-identical
duration across unrelated configurations argues against a scheduling artifact.
`get_current_step` does not long-poll (`external_run_service.py:491`), so a 68 s response
is a genuine server-side stall. It exceeds the 60 s decision deadline, and at Standard
N=25 it produced a real client-observed auto-hold.

**Not diagnosed.** Recorded here so it is not rediscovered as a "flake"; it deserves its
own investigation.

### Verdict

**Standard, standing — not a per-event flip.** The operator's plan is sustained hosting,
so the relevant comparison is headroom, not survival:

- Free clears a dozen concurrent runs, but **at or near its full CPU quota throughout**,
  with second-scale latencies and a hard wall at 25 where agents start receiving 500s.
  Nothing is left for the dashboard the agents are hosted behind.
- Standard clears 100 concurrent at 63.5% utilisation with no failures and no lock
  errors — roughly **5× the target load, with headroom to grow into**.

The upgrade trigger is not "a dozen agents"; it is **any expectation of more than ~20
concurrent active runs**, or of Free's CPU being shared with anything else.

Sustained 100 agents remains a different question and is still not answered here.

**Horizontal scaling stays closed.** Run state is module-level (`_sessions`,
`external_run_service.py:67`; `_runs`, `runs/service.py:49`) and the heartbeat path
*fails* orphaned runs rather than migrating them. Every tier above Standard buys
nothing until run state leaves process memory.

## 6. Design

Five work items. T1–T3 are behaviour changes; T4 restores trust in the instrument (and
carries one production change of its own — the wholesale-failure detector below); T5
uses it. **They are not one file each and not order-free** — see §8 Q7 and the plan's
Architecture note.

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

**Three call sites, not one.** `_maybe_apply_timeout` is reached from three places, all
of which already hold `self._step_lock`:

| Site | Shape | Who drives it |
|---|---|---|
| `get_current_step` (`:430-434`) | `while` loop — many holds per call | a polling agent |
| `drain_expired` (`:734-742`) | identical `while` loop | the reaper (`runs/service.py:432`) and `execution/backtest_backend.py:267` every pass |
| `get_status` (`:710-712`) | **one unlooped call** | the protocol router, on essentially every request |

**All three must emit**, and the third is the one an implementer will miss.

- **The reaper path carries the worst case.** An agent that crashes at step 3 of 21 has
  its remaining 18 steps auto-held by a background thread with nobody polling and nobody
  watching, and the finalized curve is then published as that agent's.
- **The `get_status` path carries the *most traffic*, and it silently pre-empts the
  instrumented one.** `runs/service.py:384` calls it inside `_sync_status` — its own
  comment reads `# applies timeout side-effects` — reached from ~10 router-driven sites
  (an eleventh `_sync_status` call, `runs/service.py:433`, is inside `reap_runs` and
  belongs to the reaper row above, not to router traffic), plus `get_step` (`:749`) and
  `submit_decision` (`:781`, `# apply timeout side effects`).
  Worse, `get_step` calls `session.get_status()` **first** and only then guards
  `if session.status == "waiting_decision" and session.step_index == seq` before calling
  `get_current_step()`. A hold applied inside that `get_status()` has already advanced
  `step_index` (`:601`), so the guard fails and the instrumented loop is **never reached
  for the very poll that produced the hold**. Instrumenting only the two `while` loops
  would therefore leave the normal live-agent path exactly as silent as it is today —
  which is the whole of F2.

Because `get_status` applies at most one hold per call, emitting there costs at most one
line per genuine hold; there is no volume argument against it. The shared helper must
**not** re-acquire `_step_lock` — every one of the three callers already holds it.

Count holds in the loop and emit **one** line after it, never one per step
(21 steps × 100 agents = 2,100 lines otherwise). **Capture the step index before the
loop** — `_maybe_apply_timeout` → `_advance_step` increments `self.step_index` (`:601`),
so an index read afterwards names a step that was *not* held:

```
⚠️ decision deadline: auto-held N step(s) for <backtest_id> (agent=<name>,
   steps=<first>..<last>, total_holds=<t>) — these steps are NOT the agent's decisions
```

Rationale is integrity, not diagnostics: a published curve containing auto-held steps
is not the agent's curve, which is the same concern the H6 guard exists for. A range is
what makes the line answer that question — an auditor reconciling a curve against the log
needs to know *which* steps were not the agent's, and a single post-drain index cannot
say. The count is already durable (F2); the range is the part that is new.

### T3 — Sweep terminal legacy sessions

Add `sweep_terminal_sessions()` to `external_run_service.py`, registered in `app.py`
next to `register_reaper_sweep(reap_v2_runs)` (`app.py:220`) so it runs on the existing
60 s reaper pass. No new thread.

Retention TTL (`LEGACY_SESSION_RETENTION_SECONDS`, default `300`) rather than immediate
eviction: a client may still be reading a completed run. The clock starts at the first
sweep that observes the session terminal, so no `_finalize`/`cancel` path can miss it.
Reads for an evicted run already fall back to the persisted row — `evict_session`'s own
docstring (`:1040-1043`) carries that safety argument.

**`_sessions` is not a legacy-only registry — say so before implementing.** It has a
single write site, `_sessions[backtest_id] = session` (`:990`) inside `start_backtest`
(`:947`), which serves the legacy `/api/v1/backtest/*` route *and* the protocol surface
via `run_service.create_run`. A sweep that walks all of `_sessions` therefore sees
protocol sessions too. It is still correct to walk the whole dict, for a reason worth
recording because it is an ordering dependency:

- **Within one reaper pass, protocol sessions are already gone.** `reap_runs` evicts
  terminal engine sessions from its `_runs` walk at `runs/service.py:434-436` and only
  *then* invokes registered sweeps at `:466-470`. The sweep sees a protocol session only
  if that eviction failed — where the TTL is a backstop, not a regression. Do not reorder
  those two blocks.
- **v2 is out of scope either way, and cannot be brought in.** `BacktestBackend`
  constructs its `ExternalBacktestSession` directly (`execution/backtest_backend.py:74-78`)
  and never registers it in `_sessions`, so this sweep can never bound v2 sessions. v2's
  bound is the archive-on-terminal path, which already exists.

### T4 — Fix the load-test harness, and make a wholesale baseline failure loud

`stress_serve.py:60` → `lambda *a, **k: FakeAlpacaLoader()`. Add a `--windows
shared|distinct` flag to `drive_agents.py`: shared collapses baseline work to one job
via the dedup cache, distinct forces N jobs through the single worker thread. That
switch is what F5's diagnosis needs, and it is the difference between a cheap and an
expensive demo (§7).

**The wholesale-failure detector belongs in `baseline_worker`, not only in the harness.**
F4 survived an entire ladder because `_drain_forever` prints
`⚠️ Baseline generation failed (run saved): {exc}` per job (`baseline_worker.py:100`) and
continues; the module keeps no failure counter of any kind — `_completed` (`:56-61`)
records only successes. Fixing this in `stress_serve.py` alone would leave *production*
exactly as blind as the harness was: an upstream break in which every baseline fails
while runs still finalize would remain a stream of per-item warnings nobody reads. This
is the repo's own rule that **a per-item warning cannot report a total contract break**,
and it applies to the finding F4 was derived from, not just to F2. So: a consecutive- or
aggregate-failure counter in `baseline_worker` that escalates to one unmissable line, and
*separately* a shutdown banner in the harness.

### T5 — Validation against a real Render instance

✅ **Executed 2026-08-18, under CPU-limited emulation rather than on Render** (the
harness cannot target a deployed instance — see the plan's T5 block). §5 carries the
results and has been rewritten around them. It did disagree with §5, and §5 was wrong,
exactly as the rule below required.

Nothing in this workstream has ever run on Render. Everything in §2 and §5 is a
12-core dev box plus arithmetic. Deploy to Standard, run the fixed harness at 100
agents, and assert: zero failures, `timeout_holds == 0`, RSS below the instance
ceiling. **If T5 disagrees with §5, §5 is wrong**, not T5.

⚠ Two of the assertions in that sentence were themselves defective: `timeout_holds == 0`
could not fail as written (§2), and "RSS below the instance ceiling" was never the
binding constraint on either tier (§5).

T5 also carries the three predecessor criteria §2 could not close, because this is the
first environment on which they mean anything: **create p95 < 1000 ms, decision p95 <
1000 ms, and RSS *growth* < 100 MB** (growth across the run, not peak). Capturing them
here — with baselines actually executing — is what finally closes Task 12 Step 2. Step 3
(post-deploy prod smoke) is the same call against prod and stays outstanding until run.

## 7. Operational note — share the window

`baseline_worker` dedups by `(start_date, end_date, mode)` behind one thread. 100 demo
agents on **one** window queue a single baseline backtest; 100 agents on **distinct**
windows queue 100 serialized ones. Give the demo agents a shared date range.

## 8. Architecture review

| | | |
|---|---|---|
| Q0 principle | Additive safety on a working design; no new subsystems, no new routes | **PASS** |
| Q1 scalability | CPU is bounded, linear and measured, and no new bottleneck is introduced — but **memory is not settled**: 311 MB is a single dev-box run on synthetic bars and one shared window, and the predecessor's `RSS growth` criterion was never captured at all | **CONCERN** — T5 measures RSS growth on the target instance |
| Q2 customizability | Every new knob is an env var with a documented default — documented in **`.env.example`**, which is where this repo's operational knobs live and where the predecessor plan put its own (`MARKET_DATA_CACHE_MAX_ENTRIES` at `.env.example:206-208`). T1 and T3 each carry that step; a constant that exists only in a source file and a conftest strip is not a documented default | **PASS** — conditional on those steps landing |
| Q3 failure story | T1 bounds a hung socket; T2 makes a corrupt-provenance event audible; T3 bounds retention | **PASS** — conditional on T2 covering **all three** `_maybe_apply_timeout` sites; two out of three leaves the live protocol path silent |
| Q4 observability | T2 is the fix; T5 is the only real-environment evidence and does not exist yet | **CONCERN** — accepted, T5 closes it |
| Q5 cost | $25/mo Standard, at demo time only. §4 deferred with measurements recorded | **PASS** |
| Q6 trust boundaries | Unchanged. No new routes, no auth changes, no new unauthenticated surface | **PASS** |
| Q7 reality check | T3 reuses the existing sweep hook rather than adding a thread. But **the items are not one file each and not order-free**: T2 and T3 both modify `external_run_service.py`; T1 and T3 both modify `tests/conftest.py`; T3 also touches `app.py`; T4 touches five (four modified — `stress_serve.py`, `drive_agents.py`, `README.md`, `baseline_worker.py` — plus one new test) | **CONCERN** — land T2 before T3, and rebase T3 on merged `main`; see the plan's Architecture note |

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
