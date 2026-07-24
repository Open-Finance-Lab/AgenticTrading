# Agent-API scale sustainability (100 concurrent agents, 1000-ready seams)

**Date:** 2026-07-24
**Status:** Draft for review
**Related:** `2026-07-20-run-history-persistence-design.md` (approved, unshipped — this spec
builds on it and reworks two of its stated assumptions, see §Alignment), issue #140 (hot-half
persistence tracker), #145, #195.

## Problem

A teammate's 100 concurrent protocol agents saw minutes-long response times in prod. A local
hermetic load harness (synthetic market data, 3-day ≈ 21-step runs) reproduced and quantified
it on 2026-07-24:

| Concurrent agents | Wall | Throughput | `create_run` median | Worst request |
|---|---|---|---|---|
| 10 | 4.1 s | 108 req/s | 114 ms | 0.8 s |
| 25 | 10.4 s | 103 req/s | 1.4 s | 3.0 s |
| 50 | 20.8 s | 104 req/s | 3.2 s | 7.3 s |
| 100 | **161.7 s** | **26.5 req/s** | 4.6 s (max 12 s) | **133.5 s** + 1 conn timeout |

Throughput is pinned (~104 req/s) regardless of concurrency — a single Python worker with
GIL-serialized pandas work gains nothing from more clients — and collapses 4× at 100 agents.
On Render's free tier (0.1 shared vCPU) the same work is 10–30× slower: that is the reported
"minutes". Root mechanisms, all verified at source 2026-07-24:

1. **Silent result corruption (the integrity failure).** At 100 agents, 2 steps were
   auto-held because the *server* couldn't process decisions inside the 30 s deadline
   (`external_run_service.py:46`, applied at `:381`). The run completes green while recording
   `timeout_hold` steps the agent never chose. Nothing surfaces the count anywhere.
2. **Per-run private market data.** Every `ExternalBacktestSession` fetches its own DJIA-30
   bars from Alpaca and computes its own indicators (`external_run_service.py:179–204`),
   holding a private `all_data`/`timestamps`/`price_cache` copy (~1–5 MB per 3-day run,
   ~10× for month runs). 100 simultaneous creates = 100 concurrent pandas indicator passes
   fighting for one core — this, not the lock itself, is most of the 4.6 s create latency —
   and 1000 in-memory sessions cannot fit in 512 MB.
3. **Finalize runs two baseline backtests in-request, under the step lock.** The final
   decision's request thread runs `_finalize()` — 4 DB writes plus buy-hold and DJIA baseline
   backtests ("seconds-to-minutes", `external_run_service.py:597–604`) — while holding the
   session `_step_lock` (`submit_decisions`, `:443` → `_advance_step` → `_finalize`). Every
   `steps/next` poll for that run blocks a threadpool thread for the duration (AnyIO cap 40 —
   the observed accept starvation). That's the 133.5 s tail. 100 runs finishing together =
   200 baseline backtests on one core.
4. **Per-request auth + connection cost.** `resolve_api_key` does an unconditional
   `last_used_at` UPDATE on every authenticated request (`repository.py:354–358`,
   `repository_postgres.py:321–324`), and the Postgres twin opens a **fresh
   `psycopg.connect()` (TLS to Neon) per call** (`repository_postgres.py:39–40`); no pooling
   exists anywhere in the repo. A typical request costs 1–3 fresh connections.

**Goal:** 100 concurrent protocol agents complete runs on the current free tier with **zero
silent corruption**; every tier leaves an explicit seam toward the 1000-agent goal instead of
a rewrite.

## Decisions (settled with Felix, 2026-07-24)

1. **Target = 100 now; 1000-ready seams, not 1000-scale machinery.** Multi-worker execution
   and durable per-step state stay out of scope (tracked by #140; see §Alignment).
2. **Decision deadline default 30 s → 60 s.** Already env-tunable
   (`EXTERNAL_AGENT_DECISION_TIMEOUT_SECONDS`); only the hardcoded default changes.
3. **Overload policy: global backstop cap + 429.** New runs beyond a global active-run cap
   are rejected with 429 + `Retry-After`. An admission queue (runs parked in a `pending`
   state, which the shipped SDK tolerates) is the 1000-tier evolution, not built now.
4. **Dashboard-path fixes ship separately.** The My Agents N+1, `async def`-blocking, and
   runs-payload pagination fixes are their own small PRs (the N+1 is already a named
   follow-up of the 2026-07-20 spec) — this spec covers only the agent-API path. Filed
   2026-07-24 as #201 (N+1), #202 (async-def blocking), #203 (`GET /runs` pagination).
5. **The wire contract is frozen: no new step/run statuses.** The shipped `AgentRunner`
   raises `ATLAPIError` on any `steps/next` status outside its closed allow-list
   (`packaging/agentictrading/src/agentictrading/runner.py:143–185` — the if/elif chain,
   with the unexpected-status raise at `:180–185`; locked by
   `test_atl_runner.py:183–193`). Every change below is designed so `"completed"` keeps
   meaning "results persisted" and no polling endpoint ever emits a new literal.

## Architecture

Four tiers, ordered by leverage, each independently shippable and observable.

### T1 — Shared market-data store (biggest single win)

New module `dashboard/backend/domain/backtesting/market_data_store.py`:

- `get_dataset(symbols, start_date, end_date) -> MarketDataset`, where `MarketDataset` is an
  immutable bundle of what today lives per-session: indicator-enriched `all_data` frames,
  `timestamps`, `price_cache`, `total_steps`. Keyed by `(tuple(symbols), start, end)`
  (symbols are in the key so future non-DJIA universes just work).
- **Single-flight (blocking — a new mechanism, not `cache.py`'s):** the first requester
  builds (one Alpaca fetch + one indicator pass); concurrent requesters for the same key
  wait on a `threading.Event` and receive the same dataset. This deliberately diverges from
  `cache.py`'s coordinator, whose followers *never block* and return `default` instead
  (`cache.py:56–61`) — correct for dashboard reads, wrong for a dataset a run cannot start
  without. A build failure propagates to all waiters and is negative-cached for ~30 s so a
  dead Alpaca doesn't trigger a retry stampede.
- **Blocking waits never happen under `_create_lock`.** Both create surfaces hold the
  global `_create_lock` across session creation (`domain/runs/service.py:421–461`,
  `api/v2/runs.py:346–388`), so the possibly-blocking `get_dataset()` is called only from
  the loader threads (today's `load_market_data` call site). Under the lock, creation does a
  non-blocking `peek(key)`: **resident hit** → skip the loader thread and open the first
  step synchronously; **miss or build-in-flight** → spawn the loader thread exactly as today
  and let *it* build or wait. A follower's create must never serialize every other agent's
  create (any config, either surface) behind someone else's Alpaca fetch.
- **Eviction:** LRU capped by `MARKET_DATA_CACHE_MAX_ENTRIES` (default 4). The cap counts
  entries, not bytes, and a month-long dataset is ~10× a 3-day one (up to ~50 MB), so the
  default is sized for the worst case against the 512 MB free tier (4 × 50 MB ≈ 200 MB
  ceiling, vs 8 entries ≈ 400 MB which alone would crowd the instance); the store `print()`s
  each dataset's approximate size at build so a pathological mix is visible in logs.
  Byte-based accounting is a 1000-tier refinement. Sessions hold a direct reference to
  their dataset, so eviction only stops *future* sharing — it can never yank data from a
  live run (Python GC keeps the referenced bundle alive).
- **Safety:** shared frames are read-only by verified convention — every downstream access
  is a `.loc` read (`external_run_service.py:246–254,233–244`; `engine.py:380`; baselines
  receive `all_data` by reference at `:615` and never mutate). A regression test asserts two
  sessions with the same config share the same object identity, and the module docstring
  states the read-only contract.
- `ExternalBacktestSession.load_market_data()` (`:179`) delegates to the store; the session
  keeps only per-run mutable state (`PortfolioManager`, decision log, token counters —
  KB, not MB). The peek fast path applies to **both** loader call sites — v1
  `start_backtest` (`:792`) and v2's own `BacktestBackend.start_background_load`
  (`execution/backtest_backend.py:86–126`), which unconditionally spawns its own thread
  today — so 100 same-config creates on either surface spawn at most one loader thread.
  v2's create response keeps its hardcoded `"status": "loading"` literal
  (`api/v2/runs.py:397–399`) even when the fast path has already opened step 0 — harmless
  (the next poll reports the real state) and avoids any wire change.

Effect: N× Alpaca fetches → 1; N× indicator passes → 1 (removing the GIL storm behind the
4.6 s create median); per-run memory drops ~99%, which is the change that makes 1000 runs
*arithmetically possible* at all. `_create_lock` (`domain/runs/service.py:74`) is untouched —
with the storm gone its hold time is milliseconds, and single-lock correctness is worth
keeping until the 1000-tier.

### T2 — Finalize split + baseline dedup

`_finalize()` (`external_run_service.py:556–644`) splits along its existing seam (the status
flip at `:605` already separates the halves):

- **Synchronous half (stays in-request, under `_step_lock` as today):** equity conversion,
  run-ID mint, the four persistence writes (`:574–595`), agent auto-register (`:637–643`),
  and the `status = "completed"` flip. The four writes are four *independent*
  single-transaction connections (each `database.py` method opens/commits/closes its own,
  `database.py:56–59`) — deliberately kept that way, matching the run-history spec's idiom 5
  (no cross-method transaction, unchanged failure semantics). WAL single-writer contention
  at 100 concurrent persist halves is real but small: each write is milliseconds, the
  measured 100-agent harness logged zero "database is locked" errors, and the acceptance
  criteria (0 errors) re-verify it after the change. Fast and bounded, so `"completed"`
  still means "results persisted" — the decision response's `metrics`/`run_id` (`:503–513`)
  are available exactly as today (the `compare_url`/baseline caveat is under *Observable
  consequences* below). **Zero wire-status changes** (Decision 5).
- **Background half:** baseline generation moves to a new
  `dashboard/backend/domain/backtesting/baseline_worker.py` — a bounded `queue.Queue`
  (default 500) drained by one lazily-started daemon thread (same pattern as `start_reaper`,
  `domain/runs/service.py:331–348`). The job carries `(run_id, start, end, mode, …)` **plus
  a direct reference to the T1 dataset** — capturing the reference at enqueue keeps the
  bundle alive (the same GC guarantee sessions rely on), so LRU eviction between enqueue
  and drain can never force the worker into a refetch/rebuild storm; same-config jobs share
  one object, so pinned memory is bounded by the distinct configs in the queue. No session
  reference is required: the worker runs the two baselines, writes
  `db.update_run_baselines(run_id, …)`, and — only if the session is still registered —
  publishes the ids by building a **new dict and assigning it in one statement**
  (`session.baseline_run_ids = {…}`, a GIL-atomic reference swap), never mutating the
  existing dict in place (today's `:619–621` pattern). That matters because
  `get_status()`/`get_current_step()` alias the dict out from under `_step_lock`
  (`:670`, `:403`) and serialize it *after* releasing the lock — an in-place write from the
  worker thread could tear a poll response mid-serialization. A lost session just means the
  DB row is the source of truth, which it already is for the v2 tombstone path
  (`ArchivedBacktestBackend.status()` rebuilds `baseline_run_ids` from `db.get_run()` on
  every call, `execution/backtest_backend.py:394–420`).
- **Baseline dedup:** buy-hold and DJIA baselines depend only on the run *config*, not the
  agent (`HourlyBacktester(start, end, session_id, use_llm=False, mode)`, `:608–614`, fed the
  shared dataset). The worker keeps a completed-baseline cache keyed by the exact parameter
  set the baselines consume (enumerated at implementation time — at minimum
  `(start, end, mode)`, plus initial capital if the baseline honors it); 100 identical-config
  finalizes produce **2** baseline backtests, not 200, with later runs pointing at the shared
  baseline rows. Baseline rows are independent `agent_runs` rows, so sharing survives
  `delete_run` of any parent (verify at implementation that no deletion path cascades into
  pointed-to baselines; if one does, dedup falls back to per-run baselines rather than
  refcounting).
- **Degradation semantics unchanged:** baselines are already best-effort
  (exceptions swallowed and printed, `:627–634`). A full queue drops the job with a printed
  error; the run stays completed without baselines — exactly today's baseline-failure
  behavior. Jobs are lost on process restart, which matches today's durability (a restart
  mid-finalize loses baselines now too).
- **Observable consequences (accepted, stated precisely):** *polled* surfaces self-heal —
  v1 status/result and completed step views read the session/DB, and the v2 tombstone
  rebuilds from the DB per call, so `baseline_run_ids` appears there as soon as the worker
  lands. Two caveats are permanent, not "brief": (1) the **one-shot legacy decision
  response** (`submit_decisions`' completed payload, `external_run_service.py:502–513`,
  returned verbatim by `api/routers/external_backtest.py:181–202` and treated as the
  *final* result by the legacy `AgenticTradingClient.run_backtest` with no re-poll) builds
  its `compare_url` and baseline fields before baselines exist — that snapshot permanently
  lacks them. Accepted: the dashboard compare view already resolves absent baseline ids by
  date-range fallback (`app.js:4547–4640`, verified), so the URL still renders; the
  protocol docs gain a note that baseline ids arrive asynchronously. (2) Under
  *heterogeneous* configs, dedup collapses nothing across differing `(start, end, mode)` —
  a 100-agent wave of distinct configs queues up to 200 baseline backtests behind one
  worker on 0.1 vCPU, which can lag minutes to hours. Accepted deliberately: the runs
  themselves are complete and correct the moment they finalize, baselines are comparison
  overlays with the same availability semantics as today's swallowed baseline-failure path,
  worker parallelism > 1 buys nothing on a 0.1 vCPU box, and the worker `print()`s queue
  depth whenever it exceeds 25 so a backed-up queue is visible in logs. Neither SDK client
  reads any of these fields for control flow.

Side effect worth naming: this *fixes a shipped latent bug* — `ATLClient` uses a flat 30 s
HTTP timeout with no per-call override (`atl_client.py:69,103`), so today's in-request
baselines can already trip a client-side `ATLTimeoutError` on the final decision submit.
After T2 the final submit returns in milliseconds.

### T3 — Deadline integrity: 60 s default, visible holds, global backstop

- **Default 60 s:** `external_run_service.py:46` changes `"30"` → `"60"`. Everything
  downstream reads the constant (deadline stamping `:262`, timeout check `:381`, the
  `decision_timeout_seconds` fields at `:435,843` and v2's `runs.py:399` /
  `backtest_backend.py:158–159`) — it auto-follows. Hardcoded duplicates that must move:
  test fixtures `tests/_v2_fakes.py:41,90`; living docs listed in §Config & docs surface.
- **`timeout_holds` counter:** the engine already stamps `decision_source="timeout_hold"`
  per held step (`:383,458`); it just never aggregates it. Add `self.timeout_holds`,
  incremented on both hold paths, surfaced in: `get_status()` / completed `get_current_step()`
  payloads; the v1 result and v2 `ResultEnvelope` inside the existing `metrics` dict
  (`metrics["timeout_holds"]` — a dict key, so no Pydantic schema change); and persisted into
  `agent_runs.metadata` JSON along with the effective `decision_timeout_seconds` (this also
  closes the known "protocol runs never populate `metadata`" inconsistency — follow-up 4 of
  the 2026-07-20 spec). A latency-corrupted run is thereby *visible* at every surface where
  its results appear. Whether leaderboards should *gate* on it (an H6-style integrity floor
  for protocol runs) is a follow-up decision for the leaderboard owners — the counter makes
  it possible.
- **Global backstop cap:** `MAX_ACTIVE_RUNS_GLOBAL` (default **100** — the tested number
  and the stated goal; `0` disables; raising it requires a measured smoke run at the higher
  value first, since nothing above 100 has ever been exercised). Enforced in the same
  `_create_lock` critical section as the per-agent cap on **both** surfaces (v1
  `service.py:421–431`; v2 `api/v2/runs.py:346–356`, which already imports the shared lock)
  via a new `run_store.count_active_runs_total()` (sibling of `repository.py:271–283`, same
  `_ACTIVE_STATUSES`). Per-agent check runs first (existing behavior preserved), then
  global. The count is deliberately reconciliation-free: both surfaces flip the
  `protocol_runs` row to terminal synchronously inside the finalizing request itself
  (`domain/runs/service.py:240–246`, `execution/backtest_backend.py:227–236`), so a raw
  COUNT tracks reality; the residual case (a run that went terminal with no subsequent
  poll) is bounded by the 60 s reaper and errs toward a spurious 429 — the safe direction,
  and `Retry-After` tells the client when to try again. Rejection: 429, code
  `too_many_active_runs_global` — registered in v2's `ERROR_CODES`
  (`api/v2/errors.py:26–29`) so the self-describing `GET /api/v2/schema` reports it — with
  a `Retry-After: 30` header. The header plumbing is explicit in two places, not free: v2
  passes `headers={"Retry-After": "30"}` via `ApiError`'s generic headers param (the
  existing auto-injection at `api/v2/errors.py:61–62` fires only for code `rate_limited`);
  v1's `ProtocolError` (`domain/runs/protocol.py:30`) gains an optional `headers` field
  that `_handle_protocol_error` (`api/routers/runs.py:41–42`) forwards to `HTTPException`.
  The shipped SDK surfaces any 429 immediately to the caller with no retry (verified — no
  retry/backoff/header reading anywhere in the package); that is the accepted contract
  (Decision 3), and the SDK README gains a "handle 429 by retrying with backoff" note.

### T4 — Per-request cost: auth cache, write debounce, connection pool

- **Auth TTL cache:** new `domain/agents/auth_cache.py` used by both call sites (v1
  `resolve_agent_by_key`, v2 `auth_scopes.resolve_agent` — both funnel into
  `agent_store.resolve_api_key` today): an in-process, thread-safe dict keyed by the API-key
  hash, TTL `AGENT_AUTH_CACHE_TTL_SECONDS` (default 10 with ±20% per-entry jitter so 100
  agents' entries never expire in lockstep and stampede the pool — a worst-case miss wave
  is ~100 cheap SELECTs; `0` disables). Revocation/rotation propagates within ≤ TTL
  (documented); for immediate invalidation the cache keeps an agent-id → hash reverse index
  and the `delete`/`rotate` paths call `auth_cache.invalidate_agent(agent_id)` — required
  because `rotate_api_key` (`repository.py:433–455`) blind-UPDATEs by agent id and never
  has the old hash in scope. The cache lives above the repository so the SQLite/Postgres
  twins stay dumb.
- **`last_used_at` debounce:** on cache miss the resolve still hits the DB, but the UPDATE
  fires only if the cached last-write for that agent is older than 60 s — per-agent write
  rate drops from per-request to ≤ 1/min (at 100 polling agents: from ~100 writes/s to
  ~1.7/s against Neon).
- **Postgres connection pool:** extend the existing pin to `psycopg[binary,pool]==3.3.4`
  (repo convention is exact pins, `requirements.txt:52`) and add a shared
  `get_pool(database_url)` helper returning one cached `ConnectionPool` per URL (small: max
  ~5, `max_idle` ≈ 5 min so Neon scale-to-zero doesn't hand out dead sockets). The pool
  must be constructed with `kwargs={"row_factory": dict_row}` — all five twins
  (`domain/agents/repository_postgres.py:40`, `version_repository_postgres.py:36`,
  `domain/portfolios/repository_postgres.py:29`,
  `domain/strategies/repository_postgres.py:39`, `users_postgres.py:32`) pass it
  per-connection today, and dropping it silently turns their dict-style row access into
  tuples. With that in place the twins swap `psycopg.connect(...)` for `pool.connection()`
  with no method-body changes (same context-manager transaction semantics). SQLite paths
  keep per-call connections (cheap, WAL-mode; no change).

## Alignment with the 2026-07-20 run-history spec

That design is approved but unshipped; this spec composes with it rather than contradicting:

- Its accepted trade-off "finalize's Neon writes sit inside the final decision-submit
  request, the 30 s deadline gives headroom" (`:199`) gets **strictly better** under T2/T3:
  the synchronous half shrinks to the four batched writes, and the deadline doubles.
- Its Decision 5(a) (single-worker assumption; in-process locks) **stands**. T1–T4 change
  nothing about worker topology. The 1000-tier (below) is where that assumption is revisited,
  via #140's hot half — exactly the seam that spec reserved.
- Its follow-up 4 (`metadata` never populated by protocol finalize) is closed by T3.
- Baseline dedup changes nothing it ports: `update_run_baselines` and the baseline rows keep
  their shapes; only *who computes them and when* moves.

## The 1000-agent seams (design-only here)

- **T1's store interface** is the later mount point for a disk/shared-store backend when
  multiple workers exist (each worker warms from one build, or a sidecar materializes
  datasets).
- **T2's queue** is the later mount point for an external job runner (pgmq/DBOS per the
  tech-stack blueprint) once finalize must survive restarts.
- **Overload evolution:** the global 429 becomes an admission queue by parking created runs
  in `pending` — already in the SDK's tolerated status set (`runner.py:176–178`) — and
  starting them as slots free. Fairness/starvation design belongs to that phase.
- **The hard gate to 1000** stays what #140 tracks: hot per-step state
  (`protocol_runs`/`protocol_steps`/`idempotency_keys` + in-memory sessions) must leave
  single-process memory before a second worker is safe. This spec deliberately makes every
  run *cheaper* rather than *distributable*, which is what the free tier can actually use.

## Performance expectations & acceptance

From the measured baseline (161.7 s wall, 26.5 req/s, 2 corrupted steps at 100 agents):

- T1 removes ~100× redundant fetch/indicator work and ~99% of per-run memory.
- T2 removes the 133 s tails, the step-lock convoy during finalize, and 99% of baseline
  compute under concurrent same-config load.
- T4 cuts 1–3 fresh connections/request to warm pooled checkouts and ~60× fewer auth writes.

**Acceptance harness:** the investigation's hermetic load scripts get promoted (cleaned) to
`dashboard/scripts/loadtest/` (`stress_serve.py`, `drive_agents.py` + a short README) so the
numbers are reproducible by anyone. Acceptance at 100 concurrent agents, 21-step runs,
local dev hardware: **0 `timeout_holds`, 0 errors, create p95 < 1 s, decision p95 < 1 s,
total wall < 60 s, server RSS growth < 100 MB**. Prod validation after deploy: one 100-agent
smoke run against a throwaway config window, watching `timeout_holds` and the startup lines.

## Testing (failing-test-first, per tier)

- **T1:** single-flight under `threading.Barrier` (pattern: `test_cache_get_or_fetch.py:57–78`);
  key isolation; build-failure propagation + recovery after negative-TTL; identity-sharing
  (two sessions, same object); eviction never affects a live session; counting-fake loader
  proves one fetch for N concurrent same-config creates (loader-patch pattern already used
  throughout `test_protocol_api.py`).
- **T2:** final decision returns `completed` + metrics with baselines still pending; worker
  fills `baseline_run_ids` and `update_run_baselines`; dedup — N same-config runs → exactly
  2 baseline backtests (counting fake); queue-overflow drop path; session-evicted-before-job
  path; the v2 lifecycle pins (`test_run_lifecycle_unification.py:199–246,454–466`) and
  `SubmitAck`/`ResultEnvelope` parity tests must pass **unchanged** — they pin exactly the
  contract Decision 5 freezes.
- **T3:** deadline default (fixtures `_v2_fakes.py:41,90` move to 60); `timeout_holds`
  increments on both hold paths (reuse the 0.01 s-timeout monkeypatch pattern,
  `test_protocol_api.py:585,1030`); counter lands in status/result/metadata; global cap 429
  + `Retry-After` on both surfaces; `0` disables; per-agent cap behavior unchanged
  (`test_protocol_api.py:732–754`).
- **T4:** auth-cache TTL/invalidation and debounce with a monkeypatched clock; pool dispatch
  tests (no live DB) + `@pg_only` pool round-trips following the established fixture rules
  (`require_local_postgres_url` guard; never point `TEST_POSTGRES_URL` at prod).
- **Env hygiene:** the four new env vars are read once at import (mirroring
  `MAX_ACTIVE_RUNS_PER_AGENT`, `service.py:52`); `tests/conftest.py` strips them at import
  time alongside the DB-URL vars (`conftest.py:44–54`, same ambient-shell rationale) so a
  stray shell value can't skew a test run, and tests monkeypatch the module constants
  (existing pattern). The promoted loadtest scripts write all artifacts (DBs, agent dumps,
  pid files) to a temp dir — never the repo tree — contain no credentials, and refuse
  non-localhost targets by default.
- **No new routes anywhere** — the global cap and counters ride existing endpoints, so the
  three route-contract freeze tests stay untouched (avoiding the #88–#91 trap).

## Config & docs surface

New env vars (all defaults safe; nothing must be set in Render):
`MAX_ACTIVE_RUNS_GLOBAL` (100), `MARKET_DATA_CACHE_MAX_ENTRIES` (4),
`AGENT_AUTH_CACHE_TTL_SECONDS` (10), `BASELINE_QUEUE_MAX` (500). Add to `.env.example` with
one-line comments; extend the CLAUDE.md env-var bullet and the deadline sentence
(`CLAUDE.md:86`: 30 s → 60 s).

Living docs to update in the same PRs:
- `docs/api/agent-environment-protocol-v1.md:293` (deadline default), `:347` (document the
  new global-cap 429 code + `Retry-After` beside the per-agent one), `:323–324` (results
  note: baselines may lag completion briefly).
- `packaging/agentictrading/README.md:47–50` (60 s; 429 guidance) and the `runner.py:62,117`
  comment/docstring — docs-only SDK changes, shipped with the next SDK release; no code
  change required by this spec.

Historical specs/plans/reviews that say "30 s" (`2026-06-23-agent-api-foundation-*`,
`2026-07-15`/`2026-07-20` specs, `docs/reviews/PR67-FIX-CHECKLIST.md`) are **not edited** —
same historical-record convention the 2026-07-20 spec applied.

User-facing docs follow-ups (surfaced, not edited mid-session): the ReadTheDocs protocol
page and PyPI README above. (The sweep also flagged `dashboard/frontend/app.html:943`'s
"10 min" pipeline-timeout copy as possibly stale; **re-verified at source 2026-07-24 and
found accurate** — it matches the enforced client-side timeout `BACKTEST_POLL_MAX_SECONDS =
600` [`app.js:18`], applied by `pollBacktestStatus()` [`app.js:3767-3857`]. No change needed;
not filed.)

## Out of scope (follow-ups)

1. **Dashboard-path fixes** — My Agents N+1 batch (helper already exists:
   `database.py:521` / `service.py:228`), `async def`-blocking handlers, runs-payload
   pagination. Separate small PRs (Decision 4). **Filed 2026-07-24:** #201, #202, #203.
2. **Admission queue** (`pending`-parked runs) — the 1000-tier overload evolution.
3. **SDK 0.2.0** — 429 retry/backoff, v2 migration (existing gate for the 0.2.0 release).
4. **Hot-half persistence / multi-worker safety** — #140 remains the tracker.
5. **Leaderboard integrity gating on `timeout_holds`** (H6-style floor for protocol runs) —
   leaderboard owners' call once the counter exists.
6. **`app.html:943` "10 min" copy** — re-verified at source 2026-07-24: **not stale**. The
   copy matches a real, enforced 10-minute client-side poll timeout
   (`BACKTEST_POLL_MAX_SECONDS = 600`, `app.js:18` / `pollBacktestStatus`). No issue filed;
   the earlier "matches nothing" note was a false positive.
7. **#195** (trades-schema migration re-runs every startup, widening write-lock windows) —
   adjacent contention defect, already filed, not fixed here.
