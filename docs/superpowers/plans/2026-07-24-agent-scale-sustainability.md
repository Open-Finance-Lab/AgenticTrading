# Agent-Scale Sustainability Implementation Plan

> **✅ STATUS 2026-07-24 — IMPLEMENTATION COMPLETE.** All four tiers merged to `origin/main` and
> auto-deployed to prod: **T1** #208 (`e333f69`), **T2** #209 (`a178c01`), **T3** #211
> (`17bf012`), **T4** #212 (`9eef9cc`). Shipped unattended via `afk-loop-runner.sh`, 0 deferrals.
> **Update 2026-08-18** (`docs/superpowers/plans/2026-08-18-burst-capacity-safety.md`): Task 12
> acceptance ran, partially. **Executed and met** — 100 agents, 35.6 s wall, 100/100 completed, 0
> failures, `timeout_holds` 0 in every rung. **Still pending** — `create p95`, `decision p95` and
> RSS growth were never captured (peak RSS was recorded instead, a different quantity); carried
> into `docs/superpowers/plans/2026-08-18-burst-capacity-safety.md`'s T5. **Still pending** — the
> Step 3 post-deploy prod smoke was not run at all. Caveat: two different executions sit behind
> these numbers, and only one is a floor. The **ladder sweep** (1→100 agents) — where
> `timeout_holds` 0 in every rung comes from — ran with the **T4** harness bug present, so its CPU
> figures (0.406–0.440 CPU-s) and its RSS are floors. The wall-time and completion numbers above
> come from a separate **fresh 100-agent run**, taken after an ad-hoc local repair of that bug;
> its CPU (0.522 CPU-s) and RSS (311 MB) are **not** floors.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make 100 concurrent protocol agents complete backtest runs on the current free tier with zero silent corruption, leaving 1000-agent seams (spec: `docs/superpowers/specs/2026-07-24-agent-scale-sustainability-design.md`).

**Architecture:** Four independently-shippable tiers, one PR each: **T1** a shared, immutable market-data store with blocking single-flight (kills the N× Alpaca-fetch/indicator storm and ~99% of per-run memory); **T2** finalize split — persistence stays in-request, baselines move to a deduping background worker; **T3** deadline 60 s + a `timeout_holds` integrity counter + a global active-run backstop cap (429); **T4** auth TTL cache, `last_used_at` debounce, and a shared psycopg3 connection pool for the Postgres twins.

**Tech Stack:** Python 3.11 / FastAPI / threading (no asyncio changes) / SQLite WAL / psycopg 3 + psycopg_pool / pytest.

## Global Constraints

- **Wire contract is frozen.** No new step/run status literals anywhere (`"loading"`, `"waiting_decision"`, `"completed"`, `"closed"`, `"failed"` only); the shipped SDK raises on unknown statuses. New *keys* in payloads are fine (SDK models use `.get()`).
- **No new HTTP routes.** The three route-contract freeze tests must pass untouched.
- **Exact defaults (copy verbatim):** `EXTERNAL_AGENT_DECISION_TIMEOUT_SECONDS` default `"60"`; `MAX_ACTIVE_RUNS_GLOBAL` default `"100"` (`0` disables); `MARKET_DATA_CACHE_MAX_ENTRIES` default `"4"`; `AGENT_AUTH_CACHE_TTL_SECONDS` default `"10"` (±20 % per-entry jitter, `0` disables); `BASELINE_QUEUE_MAX` default `"500"`; last-used debounce `60.0` s; negative-cache TTL `30.0` s; pool `max_size=5`, `max_idle=300`; `Retry-After: 30`.
- **Env vars are read once at import** (mirroring `MAX_ACTIVE_RUNS_PER_AGENT`); tests monkeypatch the module constants; every new env var is stripped at import time in `dashboard/backend/tests/conftest.py`.
- **`print()`, not `logger`** — `dashboard.backend.*` logger output is invisible under the deployed config. Assert on output with `capsys`, never `caplog`.
- **Layering:** `domain/` must not import `api/` or `app.py` (`test_architecture_boundaries.py` enforces).
- **Run everything from the repo root** with `~/atl-venv/bin/python -m pytest dashboard/backend/tests/ ...`. Never point `TEST_POSTGRES_URL` at a prod URL. Do not modify the committed seed `dashboard/storage/data/backtest.db`.
- **Branch/PR discipline:** the main checkout is shared with another active session — do all work in an isolated worktree (superpowers:using-git-worktrees). One branch + PR per tier, cut from up-to-date `origin/main`: `feat/scale-t1-market-data-store`, `feat/scale-t2-baseline-worker`, `feat/scale-t3-deadline-cap`, `feat/scale-t4-auth-pool`. Short PR titles (`feat: shared market-data store for backtest runs`, etc.). Never push to a branch whose PR merged. Run the full backend suite before opening each PR (merging to main auto-deploys prod).
- Commit messages follow the repo's `feat:`/`fix:`/`test:` convention and end with the session's standard `Co-Authored-By: Claude …` / `Claude-Session: …` trailer.

---

# Tier 1 — Shared market-data store (branch `feat/scale-t1-market-data-store`)

### Task 1: Promote the load-test harness to `dashboard/scripts/loadtest/`

The acceptance instrument lands first so every later tier's PR can cite before/after numbers. Source material: the investigation scripts (session scratchpad `agents_stress_serve.py` / `agents_load.py`); they are rewritten here with the required hygiene: **all artifacts in a temp dir, no credentials, localhost-only by default**.

**Files:**
- Create: `dashboard/scripts/loadtest/stress_serve.py`
- Create: `dashboard/scripts/loadtest/drive_agents.py`
- Create: `dashboard/scripts/loadtest/README.md`

**Interfaces:**
- Produces: `stress_serve.py` serves the real app on `127.0.0.1:8402` with a synthetic Alpaca loader and `N_AGENTS` pre-seeded agents; writes `agents.json` + `server.pid` into a printed temp dir. `drive_agents.py <M> --artifacts <dir>` drives M concurrent full runs and prints the latency/wall/RSS/timeout-holds report.

- [ ] **Step 1: Write `dashboard/scripts/loadtest/stress_serve.py`**

```python
"""Serve the real app with synthetic market data + N pre-seeded protocol agents.

Patches the Alpaca loader with a deterministic in-process generator so the
load test measures OUR stack (locks, threadpool, SQLite, pandas, finalize),
not Alpaca's API. Binds to localhost only. All artifacts (DB, agent keys,
pid file) go to a fresh temp dir, printed at startup — never the repo tree.

Usage (from the repo root):
    N_AGENTS=100 python dashboard/scripts/loadtest/stress_serve.py
"""
import os
import sys
import json
import tempfile

ARTIFACTS = tempfile.mkdtemp(prefix="atl_loadtest_")
os.environ["DATABASE_PATH"] = os.path.join(ARTIFACTS, "stress.db")
# Ambient prod/dev URLs must never leak into a load test.
os.environ.pop("CONTENT_DATABASE_URL", None)
os.environ.pop("USERS_DATABASE_URL", None)
sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def synth_bars(symbols, start, end):
    idx = pd.date_range(start=start, end=str(end) + " 23:59", freq="1h", tz="UTC")
    et = idx.tz_convert("US/Eastern")
    mask = (et.dayofweek < 5) & (
        ((et.hour > 9) & (et.hour < 16)) | ((et.hour == 16) & (et.minute == 0))
    )
    idx = idx[mask]
    data = {}
    for si, sym in enumerate(sorted(symbols)):
        n = len(idx)
        close = 100.0 + si * 5 + np.linspace(0, 2.0, n) + np.sin(np.arange(n) * 0.3) * 1.5
        df = pd.DataFrame(
            {"open": close - 0.2, "high": close + 0.5, "low": close - 0.5,
             "close": close, "volume": 10000.0},
            index=idx,
        )
        df.index.name = "timestamp"
        data[sym] = df
    return data


class FakeAlpacaLoader:
    def __init__(self, *a, **k):
        pass

    def fetch_bars(self, symbols, start, end):
        return synth_bars(symbols, start, end)


import dashboard.backend.domain.backtesting.external_run_service as ebs  # noqa: E402
import dashboard.backend.domain.backtesting.engine as engine_mod  # noqa: E402

ebs.AlpacaDataLoader = FakeAlpacaLoader
engine_mod.create_market_data_provider = lambda ds=None: FakeAlpacaLoader()

from dashboard.backend.domain.agents.repository import agent_store  # noqa: E402

N = int(os.environ.get("N_AGENTS", "100"))
agents = []
for i in range(N):
    a = agent_store.create_agent(
        name=f"load-agent-{i}",
        model_name="external/load-test",
        agent_type="external",
        description="concurrency load test",
    )
    agents.append({"agent_id": a["agent_id"], "api_key": a["api_key"]})

with open(os.path.join(ARTIFACTS, "agents.json"), "w") as f:
    json.dump(agents, f)
with open(os.path.join(ARTIFACTS, "server.pid"), "w") as f:
    f.write(str(os.getpid()))
print(f"artifacts dir: {ARTIFACTS}", flush=True)
print(f"seeded {N} agents; serving on 127.0.0.1:8402", flush=True)

import uvicorn  # noqa: E402

uvicorn.run("dashboard.backend.app:app", host="127.0.0.1", port=8402, log_level="warning")
```

- [ ] **Step 2: Write `dashboard/scripts/loadtest/drive_agents.py`**

```python
"""Simulate M concurrent protocol agents driving full backtests.

Each agent: POST /api/v1/runs (3 trading days ~= 21 hourly steps), then loop
GET steps/next -> POST decision until completed. Records per-endpoint latency,
end-to-end run wall time, errors, steps lost to the decision deadline, and the
server-reported timeout_holds counter (present once T3 ships; 0 before).

Usage:
    python dashboard/scripts/loadtest/drive_agents.py 100 --artifacts /tmp/atl_loadtest_xxx
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

parser = argparse.ArgumentParser()
parser.add_argument("agents", type=int, nargs="?", default=10)
parser.add_argument("--artifacts", required=True,
                    help="artifacts dir printed by stress_serve.py")
parser.add_argument("--base", default="http://127.0.0.1:8402")
parser.add_argument("--allow-remote", action="store_true",
                    help="required to target anything but localhost")
args = parser.parse_args()

host = urllib.parse.urlparse(args.base).hostname or ""
if host not in ("127.0.0.1", "localhost", "::1") and not args.allow_remote:
    sys.exit(f"refusing non-localhost target {args.base!r} (pass --allow-remote)")

BASE = args.base
M = args.agents
AGENTS = json.load(open(os.path.join(args.artifacts, "agents.json")))
PID_FILE = os.path.join(args.artifacts, "server.pid")

samples = []          # (kind, ms, http_status)
lock = threading.Lock()
deadline_losses = []  # steps auto-held because our decision arrived too late
server_holds = []     # server-reported timeout_holds per completed run (T3+)
run_walls = []        # end-to-end seconds per completed run
failures = []


def req(method, path, key, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"X-API-Key": key, "Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(r, timeout=180) as resp:
            payload = json.loads(resp.read())
            return resp.status, payload, (time.perf_counter() - t0) * 1000
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read())
        except Exception:
            payload = {}
        return e.code, payload, (time.perf_counter() - t0) * 1000
    except Exception as e:
        return -1, {"error": str(e)}, (time.perf_counter() - t0) * 1000


def record(kind, ms, status):
    with lock:
        samples.append((kind, ms, status))


def drive(agent):
    key = agent["api_key"]
    t_run = time.perf_counter()
    status, body, ms = req("POST", "/api/v1/runs", key, {
        "config": {"start_date": "2026-06-01", "end_date": "2026-06-03"},
    })
    record("create_run", ms, status)
    if status != 200:
        with lock:
            failures.append(("create", status, str(body)[:150]))
        return
    run_id = body["run_id"]
    lost = 0
    first = True
    for _ in range(400):  # hard cap: a run is ~21 steps
        status, step, ms = req("GET", f"/api/v1/runs/{run_id}/steps/next", key)
        record("steps_next", ms, status)
        if status != 200:
            with lock:
                failures.append(("steps_next", status, str(step)[:150]))
            return
        st = step.get("status")
        if st == "loading":
            time.sleep(0.25)
            continue
        if st == "completed":
            break
        if st != "awaiting_decision":
            with lock:
                failures.append(("unexpected_step", 200, st or "?"))
            return
        orders = []
        if first:
            orders = [{"symbol": "AAPL", "side": "buy", "quantity": 5}]
            first = False
        status, res, ms = req(
            "POST", f"/api/v1/runs/{run_id}/steps/{step['step_id']}/decision", key,
            {"idempotency_key": uuid.uuid4().hex, "orders": orders,
             "rationale": "load test decision"},
        )
        record("decision", ms, status)
        if status == 409:
            if "deadline" in str(res) or "finalized" in str(res):
                lost += 1
                continue  # step was auto-held under us; keep going
            with lock:
                failures.append(("decision409", 409, str(res)[:150]))
            return
        if status != 200:
            with lock:
                failures.append(("decision", status, str(res)[:150]))
            return
        if res.get("run_status") == "completed":
            break
    status, view, _ = req("GET", f"/api/v1/runs/{run_id}", key)
    holds = 0
    if status == 200:
        holds = (view.get("engine_status") or {}).get("timeout_holds") or 0
    with lock:
        run_walls.append(time.perf_counter() - t_run)
        deadline_losses.append(lost)
        server_holds.append(holds)


def server_stats():
    try:
        pid = int(open(PID_FILE).read())
        st = open(f"/proc/{pid}/status").read()
        rss = next(l for l in st.splitlines() if l.startswith("VmRSS")).split()[1]
        thr = next(l for l in st.splitlines() if l.startswith("Threads")).split()[1]
        return int(rss) // 1024, int(thr)
    except Exception:
        return -1, -1


def dist(label, vals):
    if not vals:
        print(f"  {label:14s} (none)")
        return
    s = sorted(vals)
    print(f"  {label:14s} n={len(s):5d}  med={statistics.median(s):8.1f}  "
          f"p95={s[max(0, int(len(s)*0.95)-1)]:8.1f}  max={s[-1]:8.1f}")


rss0, thr0 = server_stats()
print(f"\n===== {M} concurrent agents =====  (server before: {rss0} MB RSS, {thr0} threads)")
t0 = time.perf_counter()
threads = [threading.Thread(target=drive, args=(a,)) for a in AGENTS[:M]]
for t in threads:
    t.start()
for t in threads:
    t.join()
wall = time.perf_counter() - t0
rss1, thr1 = server_stats()

by_kind = {}
for kind, ms, status in samples:
    by_kind.setdefault(kind, []).append(ms)
print(f"total wall: {wall:.1f}s  |  requests: {len(samples)}  |  "
      f"throughput: {len(samples)/wall:.1f} req/s")
print(f"server after: {rss1} MB RSS ({rss1-rss0:+d}), {thr1} threads ({thr1-thr0:+d})")
print("per-request latency (ms):")
for kind in ("create_run", "steps_next", "decision"):
    dist(kind, by_kind.get(kind, []))
print("end-to-end run wall time (s):")
dist("full_run", run_walls)
total_lost = sum(deadline_losses)
runs_hit = sum(1 for x in deadline_losses if x)
print(f"completed runs: {len(run_walls)}/{M}  |  client-observed deadline losses: "
      f"{total_lost} (across {runs_hit} runs)  |  server timeout_holds: {sum(server_holds)}")
if failures:
    print(f"FAILURES: {len(failures)}")
    for f in failures[:8]:
        print("  ", f)
```

- [ ] **Step 3: Write `dashboard/scripts/loadtest/README.md`**

```markdown
# Protocol-agent load test

Reproduces the 2026-07-24 concurrency measurements (spec:
`docs/superpowers/specs/2026-07-24-agent-scale-sustainability-design.md`).

Hermetic: synthetic market data (no Alpaca), fresh temp-dir SQLite, localhost
only, no credentials. Never writes into the repo tree.

## Run

Terminal 1 (from the repo root):

    N_AGENTS=100 python dashboard/scripts/loadtest/stress_serve.py
    # prints:  artifacts dir: /tmp/atl_loadtest_XXXX

Terminal 2:

    python dashboard/scripts/loadtest/drive_agents.py 100 --artifacts /tmp/atl_loadtest_XXXX

## Acceptance target (100 agents, 21-step runs, local dev hardware)

0 timeout_holds, 0 failures, create p95 < 1 s, decision p95 < 1 s,
total wall < 60 s, server RSS growth < 100 MB.
```

- [ ] **Step 4: Smoke-run the harness at 10 agents**

Run (two terminals or backgrounded):
`N_AGENTS=10 ~/atl-venv/bin/python dashboard/scripts/loadtest/stress_serve.py` then
`~/atl-venv/bin/python dashboard/scripts/loadtest/drive_agents.py 10 --artifacts <printed dir>`
Expected: `completed runs: 10/10`, `FAILURES` absent, artifacts under `/tmp/atl_loadtest_*`, nothing new under `git status`. Kill the server afterwards.

- [ ] **Step 5: Commit**

```bash
git add dashboard/scripts/loadtest/
git commit -m "test: add hermetic protocol-agent load-test harness"
```

---

### Task 2: `market_data_store` module (blocking single-flight, LRU, negative cache)

**Files:**
- Create: `dashboard/backend/domain/backtesting/market_data_store.py`
- Test: `dashboard/backend/tests/test_market_data_store.py`

**Interfaces:**
- Produces (later tasks depend on these exact names):
  - `class MarketDataset` with attributes `key`, `all_data: Dict[str, pd.DataFrame]`, `timestamps: List`, `price_cache: Dict[str, Dict]`, `total_steps: int`
  - `get_dataset(symbols, start_date, end_date, loader_factory=None) -> MarketDataset` — **blocking** single-flight; must never be called while holding the create lock
  - `peek(symbols, start_date, end_date) -> Optional[MarketDataset]` — non-blocking; safe under the create lock
  - `MARKET_DATA_CACHE_MAX_ENTRIES` module constant (env `MARKET_DATA_CACHE_MAX_ENTRIES`, default 4)
  - `_reset_for_tests()` — clears the cache
- Consumes: `AlpacaDataLoader` (default factory), `TechnicalIndicators.calculate_indicators`, `DJIA_30`.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/test_market_data_store.py`:

```python
"""Shared market-data store: blocking single-flight, key isolation, negative
cache, LRU eviction (T1 of the 2026-07-24 agent-scale spec)."""

import threading
import time

import numpy as np
import pandas as pd
import pytest

from dashboard.backend.domain.backtesting import market_data_store as mds


def _synth_bars(symbols=("AAPL", "MSFT"), start="2026-04-15", end="2026-04-16"):
    idx = pd.date_range(start=start, end=str(end) + " 23:59", freq="1h", tz="UTC")
    et = idx.tz_convert("US/Eastern")
    mask = (et.dayofweek < 5) & (
        ((et.hour > 9) & (et.hour < 16)) | ((et.hour == 16) & (et.minute == 0))
    )
    idx = idx[mask]
    data = {}
    for si, sym in enumerate(sorted(symbols)):
        n = len(idx)
        close = 100.0 + si * 5 + np.linspace(0, 1.0, n)
        df = pd.DataFrame(
            {"open": close, "high": close + 0.5, "low": close - 0.5,
             "close": close, "volume": 1000.0},
            index=idx,
        )
        data[sym] = df
    return data


class _CountingLoader:
    calls = 0

    def fetch_bars(self, symbols, start, end):
        type(self).calls += 1
        return _synth_bars(symbols, start, end)


@pytest.fixture(autouse=True)
def _fresh_store():
    mds._reset_for_tests()
    _CountingLoader.calls = 0
    yield
    mds._reset_for_tests()


SYMS = ["AAPL", "MSFT"]


def test_build_returns_complete_bundle():
    ds = mds.get_dataset(SYMS, "2026-04-15", "2026-04-16",
                         loader_factory=_CountingLoader)
    assert set(ds.all_data) == {"AAPL", "MSFT"}
    assert ds.total_steps == len(ds.timestamps) > 0
    assert "AAPL" in ds.price_cache
    assert _CountingLoader.calls == 1


def test_single_flight_one_build_for_concurrent_requesters():
    n = 8
    barrier = threading.Barrier(n)
    out = [None] * n

    def go(i):
        barrier.wait()
        out[i] = mds.get_dataset(SYMS, "2026-04-15", "2026-04-16",
                                 loader_factory=_CountingLoader)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert _CountingLoader.calls == 1
    assert all(o is out[0] for o in out)  # identical object, not copies


def test_key_isolation_by_date_range():
    a = mds.get_dataset(SYMS, "2026-04-15", "2026-04-16", loader_factory=_CountingLoader)
    b = mds.get_dataset(SYMS, "2026-04-13", "2026-04-14", loader_factory=_CountingLoader)
    assert a is not b
    assert _CountingLoader.calls == 2


def test_peek_is_nonblocking_and_only_returns_resident():
    assert mds.peek(SYMS, "2026-04-15", "2026-04-16") is None  # cold miss

    started, release = threading.Event(), threading.Event()

    class _SlowLoader:
        def fetch_bars(self, symbols, start, end):
            started.set()
            release.wait(5)
            return _synth_bars(symbols, start, end)

    t = threading.Thread(
        target=lambda: mds.get_dataset(SYMS, "2026-04-15", "2026-04-16",
                                       loader_factory=_SlowLoader))
    t.start()
    assert started.wait(5)
    assert mds.peek(SYMS, "2026-04-15", "2026-04-16") is None  # in-flight: still None
    release.set()
    t.join(10)
    assert mds.peek(SYMS, "2026-04-15", "2026-04-16") is not None  # resident hit


def test_build_failure_propagates_and_negative_caches(monkeypatch):
    class _Boom:
        calls = 0

        def fetch_bars(self, symbols, start, end):
            type(self).calls += 1
            raise RuntimeError("alpaca down")

    with pytest.raises(RuntimeError, match="alpaca down"):
        mds.get_dataset(SYMS, "2026-04-15", "2026-04-16", loader_factory=_Boom)
    # Within the negative TTL: same error, NO second fetch (no retry stampede).
    with pytest.raises(RuntimeError, match="alpaca down"):
        mds.get_dataset(SYMS, "2026-04-15", "2026-04-16", loader_factory=_Boom)
    assert _Boom.calls == 1
    # After the negative TTL the build is retried (and can now succeed).
    real_now = time.monotonic()
    monkeypatch.setattr(mds, "_now", lambda: real_now + 31.0)
    ds = mds.get_dataset(SYMS, "2026-04-15", "2026-04-16",
                         loader_factory=_CountingLoader)
    assert ds.total_steps > 0
    assert _CountingLoader.calls == 1


def test_failure_propagates_to_concurrent_waiters():
    started, release = threading.Event(), threading.Event()
    errors = []

    class _SlowBoom:
        def fetch_bars(self, symbols, start, end):
            started.set()
            release.wait(5)
            raise RuntimeError("alpaca down")

    def leader():
        try:
            mds.get_dataset(SYMS, "2026-04-15", "2026-04-16", loader_factory=_SlowBoom)
        except RuntimeError as e:
            errors.append(("leader", str(e)))

    def waiter():
        started.wait(5)
        try:
            mds.get_dataset(SYMS, "2026-04-15", "2026-04-16", loader_factory=_SlowBoom)
        except RuntimeError as e:
            errors.append(("waiter", str(e)))

    tl, tw = threading.Thread(target=leader), threading.Thread(target=waiter)
    tl.start(); tw.start()
    started.wait(5)
    time.sleep(0.05)  # let the waiter reach event.wait()
    release.set()
    tl.join(10); tw.join(10)
    assert sorted(who for who, _ in errors) == ["leader", "waiter"]
    assert all("alpaca down" in msg for _, msg in errors)


def test_lru_eviction_bounds_entries_and_never_breaks_holders(monkeypatch):
    monkeypatch.setattr(mds, "MARKET_DATA_CACHE_MAX_ENTRIES", 2)
    d1 = mds.get_dataset(SYMS, "2026-04-13", "2026-04-14", loader_factory=_CountingLoader)
    mds.get_dataset(SYMS, "2026-04-15", "2026-04-16", loader_factory=_CountingLoader)
    mds.get_dataset(SYMS, "2026-04-17", "2026-04-18", loader_factory=_CountingLoader)
    assert mds.peek(SYMS, "2026-04-13", "2026-04-14") is None      # LRU-evicted
    assert mds.peek(SYMS, "2026-04-15", "2026-04-16") is not None
    assert mds.peek(SYMS, "2026-04-17", "2026-04-18") is not None
    # The evicted dataset stays fully usable via the held reference (GC contract).
    assert d1.total_steps > 0 and "AAPL" in d1.all_data


def test_empty_fetch_raises_runtime_error():
    class _Empty:
        def fetch_bars(self, symbols, start, end):
            return {}

    with pytest.raises(RuntimeError, match="No market data"):
        mds.get_dataset(SYMS, "2026-04-15", "2026-04-16", loader_factory=_Empty)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_market_data_store.py -v`
Expected: FAIL at import — `ModuleNotFoundError: No module named 'dashboard.backend.domain.backtesting.market_data_store'`.

- [ ] **Step 3: Write `dashboard/backend/domain/backtesting/market_data_store.py`**

```python
"""Shared, immutable market-data datasets for backtest sessions (T1).

One dataset (indicator-enriched bars + trading timestamps + price cache) per
``(symbols, start_date, end_date)`` key, shared by every session with that
config. READ-ONLY CONTRACT: every consumer treats ``all_data`` frames,
``timestamps`` and ``price_cache`` as immutable — verified convention across
the engine, baselines, and PortfolioManager. Never mutate a dataset.

Concurrency model (deliberately NOT cache.py's coordinator, whose followers
never block): the first requester for a key builds; concurrent requesters
block on a ``threading.Event`` and receive the same object. A build failure
propagates to every waiter and is negative-cached for ``NEGATIVE_TTL_SECONDS``
so a dead upstream doesn't trigger a retry stampede.

LOCK RULE: ``get_dataset`` may block for a full Alpaca fetch — it must only be
called from loader threads, NEVER while holding the run-creation lock.
``peek`` is non-blocking and is the only entry point allowed under that lock.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import pytz

from dashboard.backend.domain.backtesting.features import TechnicalIndicators
from dashboard.backend.infrastructure.market_data.alpaca_bars import AlpacaDataLoader

# Read once at import (tests monkeypatch the module constant). Entry count, not
# bytes: measured ~1.7 MB for a month-long dataset (was cited as ~50 MB), but
# that is a floor — the size print below counts only the all_data frames (not
# timestamps or price_cache) and was taken on synthetic harness bars, not real
# Alpaca DJIA-30 data. It no longer supports the old ~200 MB worst-case claim
# against the 512 MB free tier; there is no settled byte budget, so the
# 4-entry cap rests on entry count alone. Byte-aware accounting is a
# 1000-tier refinement; the size print below keeps a pathological mix visible.
MARKET_DATA_CACHE_MAX_ENTRIES = int(os.getenv("MARKET_DATA_CACHE_MAX_ENTRIES", "4"))
NEGATIVE_TTL_SECONDS = 30.0

_ET_TZ = pytz.timezone("US/Eastern")

_now = time.monotonic  # indirection so tests can advance the clock


class MarketDataset:
    """Immutable bundle of everything a session needs from market data."""

    __slots__ = ("key", "all_data", "timestamps", "price_cache", "total_steps")

    def __init__(self, key: Tuple, all_data: Dict[str, pd.DataFrame],
                 timestamps: List[Any], price_cache: Dict[str, Dict[Any, float]]):
        self.key = key
        self.all_data = all_data
        self.timestamps = timestamps
        self.price_cache = price_cache
        self.total_steps = len(timestamps)


class _Entry:
    __slots__ = ("event", "dataset", "error", "negative_until")

    def __init__(self):
        self.event = threading.Event()
        self.dataset: Optional[MarketDataset] = None
        self.error: Optional[BaseException] = None
        self.negative_until: float = 0.0


_cache_lock = threading.Lock()
_cache: "OrderedDict[Tuple, _Entry]" = OrderedDict()


def _dataset_key(symbols, start_date, end_date) -> Tuple:
    return (tuple(symbols), str(start_date), str(end_date))


def peek(symbols, start_date, end_date) -> Optional[MarketDataset]:
    """Non-blocking: the resident dataset, or None (miss / build in flight /
    negative-cached failure). The only store call allowed under _create_lock."""
    with _cache_lock:
        entry = _cache.get(_dataset_key(symbols, start_date, end_date))
        if entry is None or entry.dataset is None:
            return None
        _cache.move_to_end(entry.dataset.key)
        return entry.dataset


def get_dataset(symbols, start_date, end_date,
                loader_factory: Optional[Callable[[], Any]] = None) -> MarketDataset:
    """Blocking single-flight build-or-wait. NEVER call under _create_lock."""
    key = _dataset_key(symbols, start_date, end_date)
    factory = loader_factory or AlpacaDataLoader
    while True:
        with _cache_lock:
            entry = _cache.get(key)
            if (entry is not None and entry.error is not None
                    and _now() >= entry.negative_until):
                del _cache[key]  # negative entry expired: retry the build
                entry = None
            if entry is None:
                entry = _Entry()
                _cache[key] = entry
                is_leader = True
            else:
                _cache.move_to_end(key)
                is_leader = False

        if is_leader:
            try:
                dataset = _build_dataset(key, symbols, start_date, end_date, factory)
            except BaseException as exc:
                with _cache_lock:
                    entry.error = exc
                    entry.negative_until = _now() + NEGATIVE_TTL_SECONDS
                entry.event.set()
                raise
            with _cache_lock:
                entry.dataset = dataset
                _evict_lru_locked()
            entry.event.set()
            return dataset

        entry.event.wait()
        if entry.error is not None:
            raise entry.error
        if entry.dataset is not None:
            with _cache_lock:
                if _cache.get(key) is entry:
                    _cache.move_to_end(key)
            return entry.dataset
        # Entry was reset underneath us (tests); retry from scratch.


def _build_dataset(key, symbols, start_date, end_date, factory) -> MarketDataset:
    loader = factory()
    all_data = loader.fetch_bars(list(symbols), start_date, end_date)
    if not all_data:
        raise RuntimeError("No market data returned from Alpaca")
    for symbol, df in all_data.items():
        all_data[symbol] = TechnicalIndicators.calculate_indicators(df)
    timestamps = _build_trading_timestamps(all_data)
    if not timestamps:
        raise RuntimeError("No trading hours in the selected date range")
    price_cache = _build_price_cache(all_data, timestamps)
    dataset = MarketDataset(key, all_data, timestamps, price_cache)
    mb = sum(float(df.memory_usage(deep=True).sum()) for df in all_data.values()) / 1e6
    print(f"📊 market-data dataset built: {key[1]}→{key[2]} "
          f"({len(key[0])} syms, {dataset.total_steps} steps, ~{mb:.1f} MB)")
    return dataset


def _build_trading_timestamps(all_data: Dict[str, pd.DataFrame]) -> List[Any]:
    """Moved verbatim from ExternalBacktestSession._build_trading_timestamps."""
    all_timestamps: set = set()
    for df in all_data.values():
        all_timestamps.update(df.index)
    ordered = sorted(all_timestamps)

    min_required = int(len(all_data) * 0.8)
    filtered = []
    for ts in ordered:
        real_count = sum(1 for df in all_data.values() if ts in df.index)
        if real_count >= min_required:
            filtered.append(ts)
    ordered = filtered if filtered else ordered

    market_hours = []
    for ts in ordered:
        ts_et = ts.astimezone(_ET_TZ)
        hour, minute = ts_et.hour, ts_et.minute
        is_market = (
            (hour > 9 and hour < 16)
            or (hour == 9 and minute >= 30)
            or (hour == 16 and minute == 0)
        )
        if is_market:
            market_hours.append(ts)
    return market_hours


def _build_price_cache(all_data: Dict[str, pd.DataFrame],
                       timestamps: List[Any]) -> Dict[str, Dict[Any, float]]:
    """Moved verbatim from ExternalBacktestSession._build_price_cache."""
    cache: Dict[str, Dict[Any, float]] = {}
    for symbol, df in all_data.items():
        cache[symbol] = {}
        last_price = None
        for timestamp in timestamps:
            if timestamp in df.index:
                last_price = df.loc[timestamp, "close"]
                cache[symbol][timestamp] = float(last_price)
            elif last_price is not None:
                cache[symbol][timestamp] = float(last_price)
    return cache


def _evict_lru_locked() -> None:
    """Drop least-recently-used COMPLETED entries beyond the cap. In-flight
    builds are never evicted. Sessions hold direct references, so eviction
    only stops future sharing — it cannot break a live run."""
    done = [k for k, e in _cache.items() if e.dataset is not None or e.error is not None]
    excess = len(done) - MARKET_DATA_CACHE_MAX_ENTRIES
    for k in done[:max(0, excess)]:
        del _cache[k]


def _reset_for_tests() -> None:
    with _cache_lock:
        for entry in _cache.values():
            entry.event.set()  # release any stranded waiter
        _cache.clear()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_market_data_store.py -v`
Expected: all 8 PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/backtesting/market_data_store.py dashboard/backend/tests/test_market_data_store.py
git commit -m "feat: shared market-data store with blocking single-flight"
```

---

### Task 3: Sessions delegate to the store; test isolation + env hygiene

**Files:**
- Modify: `dashboard/backend/domain/backtesting/external_run_service.py:179-244` (replace `load_market_data`; delete `_build_trading_timestamps` / `_build_price_cache`; add `adopt_dataset`)
- Modify: `dashboard/backend/tests/conftest.py` (env strip + autouse store reset)
- Test: `dashboard/backend/tests/test_market_data_sharing.py`

**Interfaces:**
- Produces: `ExternalBacktestSession.adopt_dataset(dataset)` — attaches a built `MarketDataset` and opens step 0 under `_step_lock` (Task 4's fast paths call it).
- Consumes: `market_data_store.get_dataset(...)` from Task 2.
- Preserved: every existing test monkeypatch of `ebs.AlpacaDataLoader` keeps working — the session passes its module-global loader class down as `loader_factory` (the name is resolved at call time).

- [ ] **Step 1: Check for direct references to the methods being moved**

Run: `grep -rn "_build_trading_timestamps\|_build_price_cache\|No market data returned" dashboard/backend --include=*.py | grep -v market_data_store`
Expected: only `external_run_service.py` definitions/uses. If a test references them, update it to the store's module functions in Step 2.

- [ ] **Step 2: Write the failing test**

Create `dashboard/backend/tests/test_market_data_sharing.py`:

```python
"""Two same-config sessions share one dataset object; the loader runs once."""

import threading

import numpy as np
import pandas as pd
import pytest

import dashboard.backend.domain.backtesting.external_run_service as ebs
from dashboard.backend.domain.backtesting import market_data_store as mds


def _synth_bars(symbols, start, end):
    idx = pd.date_range(start=start, end=str(end) + " 23:59", freq="1h", tz="UTC")
    et = idx.tz_convert("US/Eastern")
    mask = (et.dayofweek < 5) & (
        ((et.hour > 9) & (et.hour < 16)) | ((et.hour == 16) & (et.minute == 0))
    )
    idx = idx[mask]
    data = {}
    for si, sym in enumerate(sorted(symbols)):
        n = len(idx)
        close = 100.0 + si + np.linspace(0, 1.0, n)
        data[sym] = pd.DataFrame(
            {"open": close, "high": close + 0.5, "low": close - 0.5,
             "close": close, "volume": 1000.0}, index=idx)
    return data


class _CountingLoader:
    calls = 0

    def fetch_bars(self, symbols, start, end):
        type(self).calls += 1
        return _synth_bars(symbols, start, end)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(ebs, "AlpacaDataLoader", _CountingLoader)
    monkeypatch.setattr(ebs, "_sessions", {})  # keep the global registry test-local
    _CountingLoader.calls = 0
    yield


def _session(i):
    return ebs.ExternalBacktestSession(
        backtest_id=f"bt_share_{i}", session_id=f"sess_{i}", agent_name=f"a{i}",
        model_name="m", start_date="2026-04-15", end_date="2026-04-16",
    )


def test_same_config_sessions_share_one_dataset_and_one_fetch():
    s1, s2 = _session(1), _session(2)
    s1.load_market_data()
    s2.load_market_data()
    assert _CountingLoader.calls == 1
    assert s1.all_data is s2.all_data          # shared object identity
    assert s1.price_cache is s2.price_cache
    assert s1.timestamps is s2.timestamps
    assert s1.status == s2.status == "waiting_decision"
    assert s1.total_steps == s2.total_steps > 0


def test_adopt_dataset_respects_terminal_status():
    ds = mds.get_dataset(["AAPL", "MSFT"], "2026-04-15", "2026-04-16",
                         loader_factory=_CountingLoader)
    s = _session(3)
    s.status = "closed"  # cancelled while loading
    s.adopt_dataset(ds)
    assert s.status == "closed"  # never resurrected
    assert s.total_steps > 0     # data attached is fine; status is not touched
```

- [ ] **Step 3: Run to verify it fails**

Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_market_data_sharing.py -v`
Expected: FAIL — `calls == 2` (each session fetches privately today) and `AttributeError: ... has no attribute 'adopt_dataset'`.

- [ ] **Step 4: Rewire the session**

In `external_run_service.py`, add the import near the other backtesting imports (after line 41):

```python
from dashboard.backend.domain.backtesting import market_data_store
```

Replace `load_market_data` (lines 179–204) and delete `_build_trading_timestamps` (206–231) and `_build_price_cache` (233–244) entirely, putting in their place:

```python
    def load_market_data(self) -> None:
        # loader_factory resolves the module-global name at call time, so every
        # existing monkeypatch of ebs.AlpacaDataLoader still controls the fetch.
        dataset = market_data_store.get_dataset(
            DJIA_30, self.start_date, self.end_date,
            loader_factory=AlpacaDataLoader,
        )
        self.adopt_dataset(dataset)

    def adopt_dataset(self, dataset: "market_data_store.MarketDataset") -> None:
        """Attach a built shared dataset and open step 0.

        SHARED + READ-ONLY: all_data/timestamps/price_cache belong to the
        store and other sessions — never mutate them (see the store docstring).

        Publish the loaded state under the step lock, and only if a concurrent
        cancel() hasn't already moved the run to a terminal state (cancel()
        writes "closed" under this same lock; an unlocked write here used to
        resurrect a cancelled run back to waiting_decision).
        """
        self.all_data = dataset.all_data
        self.timestamps = dataset.timestamps
        self.price_cache = dataset.price_cache
        self.total_steps = dataset.total_steps

        with self._step_lock:
            if self.status in TERMINAL_STATUSES:
                return
            self.status = "waiting_decision"
            self._open_current_step()
```

- [ ] **Step 5: Add store hygiene to `tests/conftest.py`**

Append to `dashboard/backend/tests/conftest.py` (after the `CONTENT_DATABASE_URL` pop):

```python
# T1+ scale knobs are read once at import (like MAX_ACTIVE_RUNS_PER_AGENT); a
# stray shell value would silently skew the whole run. Same rationale as the
# DB-URL strips above. Later tiers append their vars here.
os.environ.pop("MARKET_DATA_CACHE_MAX_ENTRIES", None)

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_shared_scale_state():
    """The market-data store is a module-level cache shared across the test
    process; without a per-test reset, one test's synthetic bars would be
    served to every later test with the same (symbols, dates) key."""
    from dashboard.backend.domain.backtesting import market_data_store
    market_data_store._reset_for_tests()
    yield
    market_data_store._reset_for_tests()
```

- [ ] **Step 6: Run the new tests, then the affected suites**

Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_market_data_sharing.py dashboard/backend/tests/test_market_data_store.py -v`
Expected: PASS.
Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_protocol_api.py dashboard/backend/tests/test_execution_backends.py dashboard/backend/tests/test_v2_merge_hardening.py dashboard/backend/tests/test_pr71_review_fixes.py -v`
Expected: PASS unchanged — their `AlpacaDataLoader` patches ride through `loader_factory`; the autouse reset keeps their synthetic datasets test-local.

- [ ] **Step 7: Commit**

```bash
git add dashboard/backend/domain/backtesting/external_run_service.py dashboard/backend/tests/conftest.py dashboard/backend/tests/test_market_data_sharing.py
git commit -m "feat: backtest sessions share market data via the store"
```

---

### Task 4: `peek` fast paths on both create surfaces

**Files:**
- Modify: `dashboard/backend/domain/backtesting/external_run_service.py:815-833` (v1 `start_backtest`)
- Modify: `dashboard/backend/execution/backtest_backend.py:86-126` (v2 `start_background_load`)
- Test: append to `dashboard/backend/tests/test_market_data_sharing.py`

**Interfaces:**
- Consumes: `market_data_store.peek(...)` (non-blocking — the only store call legal under `_create_lock`), `session.adopt_dataset(...)`.
- Wire contract: both create responses keep their hardcoded `"status": "loading"` literal even when the fast path already opened step 0 — the next poll reports the real state; no wire change.

- [ ] **Step 1: Write the failing tests** (append to `test_market_data_sharing.py`)

```python
def test_v1_start_backtest_fast_path_skips_loader_thread():
    # Warm the cache through a normal session load.
    _session(0).load_market_data()
    assert _CountingLoader.calls == 1

    res = ebs.start_backtest(
        session_id="sess_fast", agent_name="fast", model_name="m",
        start_date="2026-04-15", end_date="2026-04-16",
    )
    assert res["status"] == "loading"  # wire literal is frozen (Decision 5)
    session = ebs.get_session(res["backtest_id"])
    # Resident hit: step 0 opened synchronously — no loader thread, no fetch.
    assert session.status == "waiting_decision"
    assert _CountingLoader.calls == 1


def test_v2_start_background_load_fast_path(monkeypatch):
    import dashboard.backend.execution.backtest_backend as bb_mod

    _session(0).load_market_data()  # warm
    assert _CountingLoader.calls == 1

    row_updates = []

    class _FakeRunStore:
        def update_run(self, run_id, **kw):
            row_updates.append((run_id, kw))

    monkeypatch.setattr(bb_mod.run_repo, "run_store", _FakeRunStore())
    backend = bb_mod.BacktestBackend(
        run_id="run_fast", session_id="sess_v2", agent_name="a", model_name="m",
        start_date="2026-04-15", end_date="2026-04-16",
    )
    backend.start_background_load()
    assert backend.session.status == "waiting_decision"  # no thread needed
    assert _CountingLoader.calls == 1
    assert ("run_fast", {"status": "running"}) in row_updates
```

- [ ] **Step 2: Run to verify they fail**

Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_market_data_sharing.py -v -k fast_path`
Expected: FAIL — both surfaces spawn a loader thread today, so `session.status` is `"loading"` at the assert (and a second fetch may be counted).

- [ ] **Step 3: v1 fast path**

In `external_run_service.py` `start_backtest`, replace the block from `with _lock:` through `threading.Thread(...).start()` (currently lines 817–833) with:

```python
    with _lock:
        _sessions[backtest_id] = session

    # Fast path: creation runs under the caller's _create_lock, where blocking
    # is forbidden — peek() is non-blocking. A resident dataset means no loader
    # thread at all: attach it and open step 0 synchronously. Miss or
    # build-in-flight falls through to the loader thread exactly as before
    # (get_dataset inside the thread blocks/dedupes there).
    dataset = market_data_store.peek(DJIA_30, start_date, end_date)
    if dataset is not None:
        session.adopt_dataset(dataset)
    else:
        def _load_in_background() -> None:
            # load_market_data() constructs AlpacaDataLoader; missing credentials
            # now raise MarketDataUnavailableError (a plain Exception, B0 deep
            # fix). The SystemExit catch stays as defense-in-depth: on this daemon
            # thread the default threading.excepthook silently swallows
            # SystemExit, so a regression back to sys.exit() would strand the run
            # in "loading" forever. (Mirrors the _finalize() catch above.)
            try:
                session.load_market_data()
            except (Exception, SystemExit) as exc:
                session.status = "failed"
                session.error = str(exc)

        threading.Thread(target=_load_in_background, daemon=True).start()
```

The returned dict keeps its hardcoded `"status": "loading"`.

- [ ] **Step 4: v2 fast path**

In `execution/backtest_backend.py` `start_background_load`, insert at the top of the method (before the `def _load()` definition):

```python
        # Fast path (runs under the shared create lock — peek only, never
        # get_dataset): a resident dataset skips the loader thread entirely.
        dataset = ext.market_data_store.peek(
            DJIA_30, self.session.start_date, self.session.end_date)
        if dataset is not None:
            self.session.adopt_dataset(dataset)
            # Mirror the background loader's post-load row transition, with the
            # same don't-resurrect-a-terminal-run guard.
            with self.session._step_lock:
                status_now = self.session.status
            if status_now not in TERMINAL_STATUSES:
                try:
                    run_repo.run_store.update_run(self.run_id, status="running")
                except Exception:
                    pass  # best-effort; both statuses count as active anyway
            return
```

(`ext.market_data_store` resolves because `external_run_service` imports the store; no new import line is needed, but adding `from dashboard.backend.domain.backtesting import market_data_store` and calling it directly is equally fine — pick the explicit import.) The create response in `api/v2/runs.py:397-399` stays byte-identical (`"status": "loading"` literal).

- [ ] **Step 5: Run the tests**

Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_market_data_sharing.py -v`
Expected: all PASS.

- [ ] **Step 6: Full suite + measurement**

Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/ -q`
Expected: green (phantom `test_deleted_shim_is_not_importable` failures = stale `__pycache__`, see CLAUDE.md).
Then re-run the Task 1 harness at 100 agents and record the numbers in the PR body (expect create median to collapse from ~4.6 s and RSS growth to drop sharply; deadline losses may still be non-zero until T2/T3).

- [ ] **Step 7: Commit, push, open PR 1**

```bash
git add -A
git commit -m "feat: peek fast path on both create surfaces"
git push -u origin feat/scale-t1-market-data-store
gh pr create --title "feat: shared market-data store for backtest runs" --body "T1 of docs/superpowers/specs/2026-07-24-agent-scale-sustainability-design.md. New env var: MARKET_DATA_CACHE_MAX_ENTRIES (default 4). Paste the Step 6 harness report (before/after) here before requesting review."
```

Also add `MARKET_DATA_CACHE_MAX_ENTRIES` to `.env.example` in this commit:

```bash
# Shared market-data cache: max distinct (symbols, start, end) datasets held
# in memory for reuse across backtest runs (LRU beyond this). Default 4.
# MARKET_DATA_CACHE_MAX_ENTRIES=4
```

---

# Tier 2 — Finalize split + baseline dedup (branch `feat/scale-t2-baseline-worker`)

### Task 5: `baseline_worker` module

**Files:**
- Create: `dashboard/backend/domain/backtesting/baseline_worker.py`
- Test: `dashboard/backend/tests/test_baseline_worker.py`

**Interfaces:**
- Produces (Task 6 depends on these exact names):
  - `class BaselineJob(*, run_id, session_id, start_date, end_date, mode, all_data, publish)` — `publish: Callable[[Dict[str, str]], None]` receives the baseline-id dict
  - `submit(job) -> bool` — non-blocking enqueue; `False` + printed warning when full
  - `wait_idle(timeout=30.0) -> bool` — test helper, True once the queue is drained
  - `_reset_for_tests(maxsize=None)`
  - `BASELINE_QUEUE_MAX` module constant (env, default 500)
- Consumes: `HourlyBacktester` (imported into this module — the class object is shared, so existing test monkeypatches of its methods apply here too), `database` module's `db` **late-bound** (`db_module.db`) so per-test DB swaps take effect.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/test_baseline_worker.py`:

```python
"""Baseline worker: dedup by config, atomic publish, overflow drop (T2)."""

import threading

import pytest

import dashboard.backend.database as db_module
from dashboard.backend.domain.backtesting import baseline_worker as bw


class _FakeBacktester:
    instances = 0

    def __init__(self, start, end, session_id, use_llm=False, mode="safe_trading"):
        type(self).instances += 1
        self.all_data = None

    def run_buyhold_baseline(self):
        return f"buyhold_{type(self).instances}", []

    def run_djia_baseline(self):
        return f"djia_{type(self).instances}", []


class _RecordingDb:
    def __init__(self):
        self.baseline_writes = []

    def update_run_baselines(self, run_id, *, djia_run_id=None, buyhold_run_id=None):
        self.baseline_writes.append((run_id, djia_run_id, buyhold_run_id))


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    bw._reset_for_tests()
    _FakeBacktester.instances = 0
    fake_db = _RecordingDb()
    monkeypatch.setattr(db_module, "db", fake_db)
    monkeypatch.setattr(bw, "HourlyBacktester", _FakeBacktester)
    yield fake_db
    bw._reset_for_tests()


def _job(run_id, start="2026-04-15", end="2026-04-16", mode="safe_trading",
         publish=None):
    return bw.BaselineJob(
        run_id=run_id, session_id=f"sess_{run_id}", start_date=start,
        end_date=end, mode=mode, all_data={"AAPL": object()},
        publish=publish or (lambda ids: None),
    )


def test_same_config_jobs_run_baselines_once(_isolate):
    published = {}
    for i in range(3):
        assert bw.submit(_job(f"r{i}", publish=lambda ids, i=i: published.__setitem__(i, ids)))
    assert bw.wait_idle(10)
    assert _FakeBacktester.instances == 1                      # dedup: 1, not 3
    assert len(_isolate.baseline_writes) == 3                  # every run row linked
    assert published[0] == published[1] == published[2]
    assert published[0] == {"buy_and_hold": "buyhold_1", "djia": "djia_1"}
    assert published[0] is not published[1]                    # fresh dict per job


def test_distinct_configs_run_their_own_baselines():
    assert bw.submit(_job("a", start="2026-04-13", end="2026-04-14"))
    assert bw.submit(_job("b", start="2026-04-15", end="2026-04-16"))
    assert bw.wait_idle(10)
    assert _FakeBacktester.instances == 2


def test_job_failure_is_swallowed_and_worker_continues(capsys, monkeypatch):
    class _Boom(_FakeBacktester):
        def run_buyhold_baseline(self):
            raise RuntimeError("baseline blew up")

    monkeypatch.setattr(bw, "HourlyBacktester", _Boom)
    done = threading.Event()
    bw.submit(_job("bad"))
    assert bw.wait_idle(10)
    monkeypatch.setattr(bw, "HourlyBacktester", _FakeBacktester)
    bw.submit(_job("good", start="2026-05-01", end="2026-05-02",
                   publish=lambda ids: done.set()))
    assert bw.wait_idle(10) and done.is_set()   # worker survived the failure
    assert "Baseline generation failed" in capsys.readouterr().out


def test_full_queue_drops_job_with_print(capsys, monkeypatch):
    bw._reset_for_tests(maxsize=1)
    entered, release = threading.Event(), threading.Event()

    class _Blocking(_FakeBacktester):
        def run_buyhold_baseline(self):
            entered.set()
            release.wait(10)
            return "buyhold_x", []

    monkeypatch.setattr(bw, "HourlyBacktester", _Blocking)
    assert bw.submit(_job("one", start="2026-06-01", end="2026-06-02"))
    assert entered.wait(10)                     # worker busy on job one
    assert bw.submit(_job("two", start="2026-06-03", end="2026-06-04"))  # fills slot
    assert bw.submit(_job("three", start="2026-06-05", end="2026-06-06")) is False
    assert "Baseline queue full" in capsys.readouterr().out
    release.set()
    assert bw.wait_idle(10)
```

- [ ] **Step 2: Run to verify failure**

Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_baseline_worker.py -v`
Expected: FAIL at import — module not found.

- [ ] **Step 3: Write `dashboard/backend/domain/backtesting/baseline_worker.py`**

```python
"""Background baseline generation with config-level dedup (T2).

Baselines (buy-hold + DJIA) depend only on the run *config*, not the agent, so
a wave of same-config finalizes needs exactly one baseline pair. Finalize
enqueues a job here and returns; one lazily-started daemon thread drains the
queue (single consumer — the dedup cache therefore needs no lock: job N for a
config fully completes, DB write included, before job N+1 dequeues).

Degradation semantics match today's best-effort baselines: a full queue drops
the job with a printed error (the run stays completed without baselines), a
failure is printed and swallowed, and jobs are lost on process restart —
identical durability to the old in-request path, which lost baselines on a
mid-finalize restart too.

Each job carries a direct reference to its dataset's ``all_data`` (the T1
bundle), so LRU eviction between enqueue and drain can never force a
refetch/rebuild storm; same-config jobs share one object, bounding pinned
memory by the distinct configs in the queue.

``publish`` callbacks must swap in a NEW dict (never mutate one the session
already exposed) — see ExternalBacktestSession._publish_baselines.
"""

from __future__ import annotations

import os
import queue
import threading
from typing import Any, Callable, Dict, Optional, Tuple

from dashboard.backend import database as db_module
from dashboard.backend.domain.backtesting.engine import HourlyBacktester

BASELINE_QUEUE_MAX = int(os.getenv("BASELINE_QUEUE_MAX", "500"))
QUEUE_DEPTH_WARN = 25

_STOP = object()


class BaselineJob:
    __slots__ = ("run_id", "session_id", "start_date", "end_date", "mode",
                 "all_data", "publish")

    def __init__(self, *, run_id: str, session_id: str, start_date: str,
                 end_date: str, mode: str, all_data: Dict[str, Any],
                 publish: Callable[[Dict[str, str]], None]):
        self.run_id = run_id
        self.session_id = session_id
        self.start_date = start_date
        self.end_date = end_date
        self.mode = mode
        self.all_data = all_data
        self.publish = publish


_queue: "queue.Queue" = queue.Queue(maxsize=BASELINE_QUEUE_MAX)
_worker_thread: Optional[threading.Thread] = None
_worker_lock = threading.Lock()
# (start, end, mode) -> {"buy_and_hold": id, "djia": id}. Single-consumer, so
# unlocked access is safe; grows one tiny entry per distinct config.
_completed: Dict[Tuple[str, str, str], Dict[str, str]] = {}


def submit(job: BaselineJob) -> bool:
    """Enqueue baseline generation for a finalized run. Never blocks or raises."""
    _ensure_worker()
    try:
        _queue.put_nowait(job)
    except queue.Full:
        print(f"⚠️ Baseline queue full ({_queue.maxsize}); dropping baselines "
              f"for {job.run_id} (run saved)")
        return False
    depth = _queue.qsize()
    if depth > QUEUE_DEPTH_WARN:
        print(f"⚠️ Baseline queue depth {depth} — worker backlog")
    return True


def _ensure_worker() -> None:
    global _worker_thread
    with _worker_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return
        _worker_thread = threading.Thread(
            target=_drain_forever, args=(_queue,), daemon=True,
            name="baseline-worker")
        _worker_thread.start()


def _drain_forever(q: "queue.Queue") -> None:
    while True:
        job = q.get()
        try:
            if job is _STOP:
                return
            _run_job(job)
        # SystemExit guard mirrors the old in-finalize catch: a daemon thread
        # swallows SystemExit silently, which would kill the worker forever.
        except (Exception, SystemExit) as exc:
            print(f"⚠️ Baseline generation failed (run saved): {exc}")
        finally:
            q.task_done()


def _run_job(job: BaselineJob) -> None:
    key = (job.start_date, job.end_date, job.mode)
    ids = _completed.get(key)
    if ids is None:
        backtester = HourlyBacktester(
            job.start_date, job.end_date, job.session_id,
            use_llm=False, mode=job.mode,
        )
        backtester.all_data = job.all_data
        buyhold_id, _ = backtester.run_buyhold_baseline()
        djia_id, _ = backtester.run_djia_baseline()
        ids = {}
        if buyhold_id:
            ids["buy_and_hold"] = buyhold_id
        if djia_id:
            ids["djia"] = djia_id
        _completed[key] = ids
    # db is late-bound through the module attribute so per-test DB swaps
    # (monkeypatch.setattr(db_module, "db", ...)) reach the worker thread.
    db_module.db.update_run_baselines(
        job.run_id,
        djia_run_id=ids.get("djia"),
        buyhold_run_id=ids.get("buy_and_hold"),
    )
    job.publish(dict(ids))


def wait_idle(timeout: float = 30.0) -> bool:
    """Test helper: True once every enqueued job (publish included) finished."""
    import time
    deadline = time.monotonic() + timeout
    with _queue.all_tasks_done:
        while _queue.unfinished_tasks:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            _queue.all_tasks_done.wait(remaining)
    return True


def _reset_for_tests(maxsize: Optional[int] = None) -> None:
    """Fresh queue + dedup cache; wakes a worker blocked on the old queue."""
    global _queue, _worker_thread
    with _worker_lock:
        old_q = _queue
        _completed.clear()
        _queue = queue.Queue(
            maxsize=maxsize if maxsize is not None else BASELINE_QUEUE_MAX)
        if _worker_thread is not None and _worker_thread.is_alive():
            try:
                old_q.put_nowait(_STOP)
            except queue.Full:
                try:
                    old_q.get_nowait()
                    old_q.task_done()
                except queue.Empty:
                    pass
                old_q.put_nowait(_STOP)
        _worker_thread = None
```

- [ ] **Step 4: Run the tests**

Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_baseline_worker.py -v`
Expected: all 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/backtesting/baseline_worker.py dashboard/backend/tests/test_baseline_worker.py
git commit -m "feat: deduping background baseline worker"
```

---

### Task 6: Split `_finalize()`; wire the worker; fixture migration

**Files:**
- Modify: `dashboard/backend/domain/backtesting/external_run_service.py:556-644` (`_finalize` + new `_publish_baselines`; drop the now-unused `HourlyBacktester` import at line 44)
- Modify: `dashboard/backend/tests/conftest.py` (extend the autouse fixture)
- Modify: every test fixture that patches baselines via `ebs.HourlyBacktester` (found in Step 1)
- Modify: `docs/api/agent-environment-protocol-v1.md` (results note ~line 323)
- Modify: `.env.example` (add `BASELINE_QUEUE_MAX`)
- Test: `dashboard/backend/tests/test_finalize_split.py`

**Interfaces:**
- Consumes: `baseline_worker.BaselineJob` / `submit` from Task 5.
- Produces: `ExternalBacktestSession._publish_baselines(ids)` — GIL-atomic new-dict swap of `self.baseline_run_ids`.
- Contract pins that must NOT change: `test_run_lifecycle_unification.py` and the `SubmitAck`/`ResultEnvelope` parity tests pass unchanged; `"completed"` still means "results persisted".

- [ ] **Step 1: Find every baseline-patching fixture**

Run: `grep -rn "run_buyhold_baseline\|run_djia_baseline\|HourlyBacktester" dashboard/backend/tests/*.py`
Expected hits include `test_protocol_api.py:94-95` (`monkeypatch.setattr(ebs.HourlyBacktester, "run_buyhold_baseline", ...)`). List them all — Step 5 updates each to patch via `baseline_worker` instead.

- [ ] **Step 2: Write the failing tests**

Create `dashboard/backend/tests/test_finalize_split.py`:

```python
"""Finalize split (T2): the final submit persists + completes in-request;
baselines arrive asynchronously via the worker; polled surfaces self-heal."""

import threading

import numpy as np
import pandas as pd
import pytest

import dashboard.backend.database as db_module
import dashboard.backend.domain.backtesting.external_run_service as ebs
from dashboard.backend.domain.backtesting import baseline_worker as bw


def _synth_bars(symbols, start, end):
    idx = pd.date_range(start=start, end=str(end) + " 23:59", freq="1h", tz="UTC")
    et = idx.tz_convert("US/Eastern")
    mask = (et.dayofweek < 5) & (
        ((et.hour > 9) & (et.hour < 16)) | ((et.hour == 16) & (et.minute == 0))
    )
    idx = idx[mask]
    data = {}
    for si, sym in enumerate(sorted(symbols)):
        n = len(idx)
        close = 100.0 + si + np.linspace(0, 1.0, n)
        data[sym] = pd.DataFrame(
            {"open": close, "high": close + 0.5, "low": close - 0.5,
             "close": close, "volume": 1000.0}, index=idx)
    return data


class _Loader:
    def fetch_bars(self, symbols, start, end):
        return _synth_bars(symbols, start, end)


class _FakeBacktester:
    instances = 0
    gate = None  # when set (threading.Event), baselines wait on it

    def __init__(self, start, end, session_id, use_llm=False, mode="safe_trading"):
        type(self).instances += 1
        self.all_data = None

    def run_buyhold_baseline(self):
        if type(self).gate is not None:
            type(self).gate.wait(10)
        return "buyhold_shared", []

    def run_djia_baseline(self):
        return "djia_shared", []


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    test_db = db_module.BacktestDatabase(db_path=tmp_path / "finalize_split.db")
    monkeypatch.setattr(db_module, "db", test_db)
    monkeypatch.setattr(ebs, "db", test_db)
    monkeypatch.setattr(ebs, "AlpacaDataLoader", _Loader)
    monkeypatch.setattr(ebs, "_sessions", {})
    monkeypatch.setattr(bw, "HourlyBacktester", _FakeBacktester)
    bw._reset_for_tests()
    _FakeBacktester.instances = 0
    _FakeBacktester.gate = None
    yield test_db
    bw._reset_for_tests()


def _run_to_completion(i):
    s = ebs.ExternalBacktestSession(
        backtest_id=f"bt_fin_{i}", session_id=f"sess_{i}", agent_name=f"a{i}",
        model_name="m", start_date="2026-04-15", end_date="2026-04-16",
    )
    s.load_market_data()
    last = None
    for _ in range(s.total_steps):
        last = s.submit_decisions({"actions": []})
    return s, last


def test_final_submit_completes_before_baselines(_isolate):
    _FakeBacktester.gate = threading.Event()  # hold the worker mid-baseline
    s, last = _run_to_completion(0)
    # The one-shot decision response: completed + persisted, baselines pending.
    assert last["status"] == "completed"
    assert last["run_id"] == s.run_id
    assert last["metrics"]["final_equity"] is not None       # results persisted
    assert last["compare_url"] == f"/compare?run_ids={s.run_id}"  # no baseline ids
    assert s.baseline_run_ids == {}
    assert _isolate.get_run(s.run_id) is not None
    _FakeBacktester.gate.set()
    assert bw.wait_idle(10)
    # Polled surfaces self-heal once the worker lands.
    status = s.get_status()
    assert status["baseline_run_ids"] == {"buy_and_hold": "buyhold_shared",
                                          "djia": "djia_shared"}
    assert "buyhold_shared" in status["compare_url"]
    row = _isolate.get_run(s.run_id)
    assert row["baseline_buyhold_run_id"] == "buyhold_shared"
    assert row["baseline_djia_run_id"] == "djia_shared"


def test_same_config_finalizes_share_one_baseline_pair(_isolate):
    s1, _ = _run_to_completion(1)
    s2, _ = _run_to_completion(2)
    assert bw.wait_idle(10)
    assert _FakeBacktester.instances == 1  # 2 runs, 1 baseline backtester
    assert s1.baseline_run_ids == s2.baseline_run_ids
    assert s1.baseline_run_ids is not s2.baseline_run_ids  # swapped-in copies
    assert _isolate.get_run(s1.run_id)["baseline_djia_run_id"] == "djia_shared"
    assert _isolate.get_run(s2.run_id)["baseline_djia_run_id"] == "djia_shared"


def test_evicted_session_still_gets_db_baselines(_isolate):
    s, _ = _run_to_completion(3)
    ebs.evict_session(s.backtest_id)  # reaper freed the session before the worker ran
    assert bw.wait_idle(10)
    row = _isolate.get_run(s.run_id)
    assert row["baseline_buyhold_run_id"] == "buyhold_shared"  # DB is source of truth
```

- [ ] **Step 3: Run to verify failure**

Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_finalize_split.py -v`
Expected: FAIL — today `_finalize` runs baselines inline via `ebs.HourlyBacktester` (the un-patched real class → `_FakeBacktester.instances == 0`, and with the gate set the submit would block, tripping asserts).

- [ ] **Step 4: Split `_finalize`**

In `external_run_service.py`: delete `from dashboard.backend.domain.backtesting.engine import HourlyBacktester` (line 44) and add `from dashboard.backend.domain.backtesting import baseline_worker` next to the `market_data_store` import. Then replace the baseline try-block in `_finalize` (lines 607–634, from `try:` through the `except (Exception, SystemExit)` handler) with:

```python
        # Background half (T2): baselines depend only on the run config, so
        # they move to the deduping worker. The job pins all_data (the shared
        # T1 bundle) so cache eviction can't force a rebuild, and publishes
        # back via _publish_baselines. Queue-full/failure degrade exactly like
        # the old best-effort inline path: run saved, baselines absent.
        baseline_worker.submit(baseline_worker.BaselineJob(
            run_id=self.run_id,
            session_id=self.session_id,
            start_date=self.start_date,
            end_date=self.end_date,
            mode=self.mode,
            all_data=self.all_data,
            publish=self._publish_baselines,
        ))
```

Keep the existing comment above `self.status = "completed"` (it is still accurate) and the agent auto-register block unchanged. Add the publish method after `_finalize`:

```python
    def _publish_baselines(self, ids: Dict[str, str]) -> None:
        """Worker-thread callback: swap in a NEW dict in one statement.

        get_status()/get_current_step() alias baseline_run_ids out from under
        _step_lock and serialize it after releasing the lock — an in-place
        write from the worker could tear a poll response mid-iteration. A
        single reference assignment is GIL-atomic: readers see the old (empty)
        dict or the new (complete) one, never a hybrid.
        """
        self.baseline_run_ids = dict(ids)
```

- [ ] **Step 5: Migrate the baseline-patching fixtures**

For every hit from Step 1 (e.g. `test_protocol_api.py:94-95`): the patched class must now be the worker's. Replace

```python
    monkeypatch.setattr(ebs.HourlyBacktester, "run_buyhold_baseline", lambda self: (None, None))
    monkeypatch.setattr(ebs.HourlyBacktester, "run_djia_baseline", lambda self: (None, None))
```

with

```python
    from dashboard.backend.domain.backtesting import baseline_worker as bw
    monkeypatch.setattr(bw.HourlyBacktester, "run_buyhold_baseline", lambda self: (None, None))
    monkeypatch.setattr(bw.HourlyBacktester, "run_djia_baseline", lambda self: (None, None))
```

(Method patches land on the shared class object, so the worker thread sees them; the module reference just has to exist — `ebs.HourlyBacktester` no longer does.)

- [ ] **Step 6: Extend the conftest autouse fixture**

In `dashboard/backend/tests/conftest.py`, extend `_reset_shared_scale_state` and the env strips:

```python
os.environ.pop("BASELINE_QUEUE_MAX", None)
```

```python
@pytest.fixture(autouse=True)
def _reset_shared_scale_state():
    from dashboard.backend.domain.backtesting import baseline_worker, market_data_store
    market_data_store._reset_for_tests()
    baseline_worker._reset_for_tests()
    yield
    # Best-effort drain so a job enqueued in this test doesn't leak into the
    # next. Note pytest tears fixtures down LIFO, so a test's own monkeypatches
    # may already be reverted here — in practice patched baseline fakes return
    # instantly, so the queue is empty long before teardown. Any T2 test that
    # gates or slows the worker must call baseline_worker.wait_idle() itself
    # before returning (every test in this plan does).
    baseline_worker.wait_idle(timeout=5)
    baseline_worker._reset_for_tests()
    market_data_store._reset_for_tests()
```

- [ ] **Step 7: Verify baseline-row sharing survives run deletion (spec requirement)**

Dedup makes several runs point at ONE pair of baseline `agent_runs` rows, so deleting a parent run must not cascade into a baseline row another run still references. Run:
`grep -n "def delete_run\|DELETE FROM agent_runs\|baseline_djia_run_id\|baseline_buyhold_run_id" dashboard/backend/database.py`
and read each deletion path. Expected: `delete_run` deletes only the given `run_id`'s rows (agent_runs/equity/trades/decisions) and never touches the runs named by its `baseline_*_run_id` columns. If any path DOES cascade into pointed-to baselines, do not ship dedup as-is — per the spec, fall back to per-run baselines (drop the `_completed` cache lookup so every job generates its own pair) and note it in the PR. Also sanity-check consumers: the dashboard compare view reads baseline runs through public (non-session-gated) endpoints, so a baseline row recorded under the first finalizer's `session_id` stays readable for later runs — confirm with `grep -n "get_run_with_session" dashboard/backend/api/routers/external_backtest.py` that only the *parent* run's session-gated reads use the session check.

- [ ] **Step 8: Run the new tests, then the contract pins, then the full suite**

Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_finalize_split.py dashboard/backend/tests/test_baseline_worker.py -v`
Expected: PASS.
Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_run_lifecycle_unification.py dashboard/backend/tests/test_protocol_api.py dashboard/backend/tests/test_execution_backends.py -v`
Expected: PASS **unchanged** — these pin the frozen contract.
Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/ -q`
Expected: green.

- [ ] **Step 9: Docs + env example**

- `docs/api/agent-environment-protocol-v1.md` (results section, ~line 323): add after the `result` availability sentence:

```markdown
Baseline comparison runs (`baseline_run_ids`, the baseline ids inside
`compare_url`) are generated asynchronously after completion and normally land
within seconds; poll `GET /api/v1/runs/{run_id}` to pick them up. The decision
response that completes the run is a snapshot from before they exist.
```

- `.env.example`: add under the T1 entry:

```bash
# Max queued background baseline-generation jobs (drops beyond this; the run
# itself is unaffected). Default 500.
# BASELINE_QUEUE_MAX=500
```

- [ ] **Step 10: Commit, push, open PR 2**

```bash
git add -A
git commit -m "feat: move baseline generation off the finalize request"
git push -u origin feat/scale-t2-baseline-worker
gh pr create --title "feat: background baseline worker + finalize split" --body "T2 of docs/superpowers/specs/2026-07-24-agent-scale-sustainability-design.md. Final decision submit now returns in ms (was seconds-to-minutes under _step_lock); same-config waves run 2 baseline backtests instead of 2N. New env var: BASELINE_QUEUE_MAX (500)."
```

---

# Tier 3 — Deadline 60 s, `timeout_holds`, global cap (branch `feat/scale-t3-deadline-cap`)

### Task 7: Deadline default 30 → 60 s (+ fixtures + docs)

**Files:**
- Modify: `dashboard/backend/domain/backtesting/external_run_service.py:46`
- Modify: `dashboard/backend/tests/_v2_fakes.py:40-41,90`
- Modify: `dashboard/backend/tests/conftest.py` (strip the timeout var)
- Modify: `CLAUDE.md:86`, `docs/api/agent-environment-protocol-v1.md:293`, `packaging/agentictrading/README.md:47-50`, `packaging/agentictrading/src/agentictrading/runner.py` (docstring/comment mentions of "30s")
- Test: `dashboard/backend/tests/test_deadline_and_holds.py` (new file, shared with Task 8)

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/test_deadline_and_holds.py`:

```python
"""T3: 60 s deadline default + the timeout_holds integrity counter."""

import dashboard.backend.domain.backtesting.external_run_service as ebs


def test_default_decision_timeout_is_60():
    # conftest strips EXTERNAL_AGENT_DECISION_TIMEOUT_SECONDS at import, so
    # the module constant IS the default.
    assert ebs.DECISION_TIMEOUT_SECONDS == 60
```

- [ ] **Step 2: Run to verify failure**

Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_deadline_and_holds.py -v`
Expected: FAIL — `assert 30 == 60`.

- [ ] **Step 3: Flip the default and its hardcoded shadows**

- `external_run_service.py:46`: `os.getenv("EXTERNAL_AGENT_DECISION_TIMEOUT_SECONDS", "30")` → `"60"`.
- `tests/conftest.py`, with the other strips: `os.environ.pop("EXTERNAL_AGENT_DECISION_TIMEOUT_SECONDS", None)` (a test now asserts the default).
- `tests/_v2_fakes.py:40-41`: `"decision_deadline_at": "2026-04-15T13:30:30+00:00"` → `"2026-04-15T13:31:00+00:00"`; `"decision_timeout_seconds": 30` → `60`. Line 90: `"decision_timeout_seconds": 30` → `60`.
- Run `grep -rn "decision_timeout_seconds" dashboard/backend/tests/ | grep 30` — fix any remaining assertion pinning 30.

- [ ] **Step 4: Docs sweep (same commit)**

- `CLAUDE.md:86`: "(default 30s)" → "(default 60s)".
- `docs/api/agent-environment-protocol-v1.md:293`: "`EXTERNAL_AGENT_DECISION_TIMEOUT_SECONDS` (default 30s)" → "(default 60s)". Also `grep -n "30s\|30 s" docs/api/agent-environment-protocol-v1.md` and update any other deadline mention (the auto-hold bullet).
- `packaging/agentictrading/README.md:47`: "default **30s**" → "default **60s**"; append to that callout: "If the backend returns **429** (server at capacity), wait and retry with backoff — the client does not retry for you."
- `packaging/agentictrading/src/agentictrading/runner.py`: `grep -n "30s" packaging/agentictrading/src/agentictrading/` → update each comment/docstring ("default 30s" → "default 60s"). Docs-only; ships with the next SDK release, no code change.
- Historical specs/plans that say 30 s are **not** edited (historical-record convention).

- [ ] **Step 5: Run the test + fake-consuming suites**

Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_deadline_and_holds.py dashboard/backend/tests/test_v2_api.py dashboard/backend/tests/test_run_lifecycle_unification.py -v` (if `test_v2_api.py` doesn't exist, run whichever suites import `_v2_fakes` — `grep -rln "_v2_fakes" dashboard/backend/tests/`).
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: raise decision-deadline default to 60s"
```

---

### Task 8: `timeout_holds` counter, surfaced + persisted

**Files:**
- Modify: `dashboard/backend/domain/backtesting/external_run_service.py` (`__init__`, `_advance_step`, `get_status`, `get_current_step` completed branch, `_finalize` insert_run, `build_final_metrics`, `get_run_result`)
- Test: append to `dashboard/backend/tests/test_deadline_and_holds.py`

**Interfaces:**
- Produces: `session.timeout_holds: int`; `metrics["timeout_holds"]` in v1 result / v2 `ResultEnvelope`/`SubmitAck` metrics (plain dict keys — no Pydantic change); `agent_runs.metadata` JSON gains `{"decision_timeout_seconds": ..., "timeout_holds": ...}` (closes the "protocol runs never populate metadata" follow-up).

- [ ] **Step 1: Verify metadata decoding**

Run: `grep -n "metadata" dashboard/backend/database.py | head` and read the decode helper around line 484 — confirm `get_run()` returns `metadata` as a decoded dict (it does via the JSON-decode helper). If any read path returns raw text, decode defensively in `build_final_metrics` as shown below (it already guards).

- [ ] **Step 2: Write the failing tests** (append to `test_deadline_and_holds.py`)

```python
import numpy as np
import pandas as pd
import pytest

import dashboard.backend.database as db_module
from dashboard.backend.domain.backtesting import baseline_worker as bw


def _synth_bars(symbols, start, end):
    idx = pd.date_range(start=start, end=str(end) + " 23:59", freq="1h", tz="UTC")
    et = idx.tz_convert("US/Eastern")
    mask = (et.dayofweek < 5) & (
        ((et.hour > 9) & (et.hour < 16)) | ((et.hour == 16) & (et.minute == 0))
    )
    idx = idx[mask]
    data = {}
    for si, sym in enumerate(sorted(symbols)):
        n = len(idx)
        close = 100.0 + si + np.linspace(0, 1.0, n)
        data[sym] = pd.DataFrame(
            {"open": close, "high": close + 0.5, "low": close - 0.5,
             "close": close, "volume": 1000.0}, index=idx)
    return data


class _Loader:
    def fetch_bars(self, symbols, start, end):
        return _synth_bars(symbols, start, end)


@pytest.fixture
def session(monkeypatch, tmp_path):
    test_db = db_module.BacktestDatabase(db_path=tmp_path / "holds.db")
    monkeypatch.setattr(db_module, "db", test_db)
    monkeypatch.setattr(ebs, "db", test_db)
    monkeypatch.setattr(ebs, "AlpacaDataLoader", _Loader)
    monkeypatch.setattr(bw.HourlyBacktester, "run_buyhold_baseline",
                        lambda self: (None, None))
    monkeypatch.setattr(bw.HourlyBacktester, "run_djia_baseline",
                        lambda self: (None, None))
    s = ebs.ExternalBacktestSession(
        backtest_id="bt_holds", session_id="sess_h", agent_name="a",
        model_name="m", start_date="2026-04-15", end_date="2026-04-16",
    )
    s.load_market_data()
    return s, test_db


def test_expired_poll_increments_timeout_holds(session, monkeypatch):
    s, test_db = session
    monkeypatch.setattr(ebs, "DECISION_TIMEOUT_SECONDS", 0.0)
    s.drain_expired()  # every step auto-holds
    assert s.status == "completed"
    assert s.timeout_holds == s.total_steps
    assert s.get_status()["timeout_holds"] == s.total_steps
    step_view = s.get_current_step()
    assert step_view["status"] == "completed"
    assert step_view["timeout_holds"] == s.total_steps
    row = test_db.get_run(s.run_id)
    assert row["metadata"]["timeout_holds"] == s.total_steps
    assert row["metadata"]["decision_timeout_seconds"] == 0.0
    metrics = ebs.build_final_metrics(row)
    assert metrics["timeout_holds"] == s.total_steps


def test_late_submit_increments_timeout_holds(session, monkeypatch):
    s, _ = session
    assert s.timeout_holds == 0
    monkeypatch.setattr(ebs, "DECISION_TIMEOUT_SECONDS", 0.0)  # deadline now past
    res = s.submit_decisions({"actions": []})
    assert res["accepted"] is False and res["outcome"] == "timeout_hold"
    assert s.timeout_holds == 1


def test_clean_run_reports_zero_holds(session):
    s, test_db = session
    for _ in range(s.total_steps):
        s.submit_decisions({"actions": []})
    assert s.status == "completed"
    assert s.timeout_holds == 0
    assert test_db.get_run(s.run_id)["metadata"]["timeout_holds"] == 0
    assert ebs.get_run_result(s.run_id, "sess_h")["metrics"]["timeout_holds"] == 0
```

- [ ] **Step 3: Run to verify failure**

Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_deadline_and_holds.py -v`
Expected: the three new tests FAIL (`AttributeError: ... 'timeout_holds'`).

- [ ] **Step 4: Implement the counter**

In `external_run_service.py`:

1. `__init__` (after `self.llm_calls = 0` block): `self.timeout_holds = 0`.
2. `_advance_step`, first line of the body — the single choke point both hold paths flow through:

```python
        if decision_source == "timeout_hold":
            # Integrity counter: steps the server auto-held past the deadline.
            # A latency-corrupted run stays green but is now VISIBLE at every
            # surface where its results appear (status, result metrics,
            # agent_runs.metadata).
            self.timeout_holds += 1
```

3. `get_status` `base` dict: add `"timeout_holds": self.timeout_holds,` after `"run_id"`.
4. `get_current_step` completed branch: add `"timeout_holds": self.timeout_holds,` after `"total_steps"`.
5. `_finalize`'s `db.insert_run(...)` call: add

```python
            metadata={
                "decision_timeout_seconds": DECISION_TIMEOUT_SECONDS,
                "timeout_holds": self.timeout_holds,
            },
```

6. `build_final_metrics` return dict: add

```python
        "timeout_holds": (run.get("metadata") or {}).get("timeout_holds")
        if isinstance(run.get("metadata"), dict) else None,
```

7. `get_run_result`'s inline `"metrics"` dict: add the same `"timeout_holds"` expression. (v2's `ResultEnvelope.metrics`/`SubmitAck.metrics` are `Dict[str, Any]` — the key rides through with zero model changes.)

- [ ] **Step 5: Run the tests, then the suite**

Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_deadline_and_holds.py -v` → PASS.
Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/ -q` → green (watch `test_protocol_api.py:579` `test_timeout_generates_hold` and the v2 parity tests — new keys are additive, they must not break).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: surface timeout_holds at every result surface"
```

---

### Task 9: Global backstop cap — 429 on both surfaces

**Files:**
- Modify: `dashboard/backend/domain/runs/repository.py` (add `count_active_runs_total` after `count_active_runs`, line 283)
- Modify: `dashboard/backend/domain/runs/service.py` (new constant near line 52; check in `create_run` after line 431)
- Modify: `dashboard/backend/domain/runs/protocol.py:27-39` (`ProtocolError.headers`)
- Modify: `dashboard/backend/api/routers/runs.py:41-42` (`_handle_protocol_error` forwards headers)
- Modify: `dashboard/backend/api/v2/errors.py:26-30` (`ERROR_CODES`)
- Modify: `dashboard/backend/api/v2/runs.py:32-35,346-356` (import + check)
- Modify: `dashboard/backend/tests/conftest.py` (strip), `.env.example`, `docs/api/agent-environment-protocol-v1.md:347` (error table), `CLAUDE.md` env bullet
- Test: `dashboard/backend/tests/test_global_run_cap.py`

**Interfaces:**
- Produces: `run_store.count_active_runs_total() -> int`; `MAX_ACTIVE_RUNS_GLOBAL` in `domain/runs/service.py` (imported by value into `api/v2/runs.py`, matching how `MAX_ACTIVE_RUNS_PER_AGENT` is shared — tests patch each module's copy, exactly like `test_run_lifecycle_unification.py:116,137` does today); error code `too_many_active_runs_global` (429, `Retry-After: 30`).
- The count is deliberately reconciliation-free: both surfaces flip rows terminal synchronously in the finalizing request; the residual is bounded by the 60 s reaper and errs toward a safe 429.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/test_global_run_cap.py`:

```python
"""Global active-run backstop cap (T3): 429 too_many_active_runs_global with
Retry-After on BOTH create surfaces; 0 disables; per-agent cap unchanged."""

import pytest
from fastapi.testclient import TestClient

import dashboard.backend.api.v2.runs as runs_mod
import dashboard.backend.domain.runs.service as run_service
from dashboard.backend.app import app
from dashboard.backend.domain.runs.repository import run_store

client = TestClient(app)


def _agent(name):
    r = client.post("/api/v2/agents", json={"name": name}).json()
    return r["api_key"], r["session_id"], r["agent_id"]


class _StubBackend:
    loop = "lockstep"
    news_sentiment_source = None

    def __init__(self, **kwargs):
        self._active = True

    def start_background_load(self):
        pass

    def is_active(self):
        return self._active

    def status(self):
        return {"status": "waiting_decision"}

    def advance(self):
        pass

    def cancel(self):
        self._active = False


def _seed_active_run(agent_id, sid, n=1):
    for i in range(n):
        run_store.create_run(
            agent_id=agent_id, agent_version_id=None, session_id=sid,
            environment_id="us-equity-hourly-v1", environment_type="backtest",
            config={}, backtest_id=f"bt_cap_{agent_id}_{i}", status="running",
        )


def test_count_active_runs_total_spans_agents():
    _, sid_a, aid_a = _agent("cap-count-a")
    _, sid_b, aid_b = _agent("cap-count-b")
    before = run_store.count_active_runs_total()
    _seed_active_run(aid_a, sid_a)
    _seed_active_run(aid_b, sid_b)
    assert run_store.count_active_runs_total() == before + 2


def test_v2_global_cap_rejects_with_retry_after(monkeypatch):
    key_b, _, _ = _agent("cap-v2-victim")
    _, sid_a, aid_a = _agent("cap-v2-filler")
    _seed_active_run(aid_a, sid_a)
    monkeypatch.setattr(runs_mod, "MAX_ACTIVE_RUNS_GLOBAL",
                        run_store.count_active_runs_total())
    monkeypatch.setattr(runs_mod, "BacktestBackend", _StubBackend)
    resp = client.post(
        "/api/v2/runs",
        json={"start_date": "2026-04-15", "end_date": "2026-04-16"},
        headers={"X-API-Key": key_b},
    )
    assert resp.status_code == 429, resp.text
    assert resp.json()["error"]["code"] == "too_many_active_runs_global"
    assert resp.json()["error"]["retryable"] is True
    assert resp.headers.get("Retry-After") == "30"


def test_v1_global_cap_rejects_with_retry_after(monkeypatch):
    _, sid_a, aid_a = _agent("cap-v1-filler")
    _, sid_b, aid_b = _agent("cap-v1-victim")
    _seed_active_run(aid_a, sid_a)
    monkeypatch.setattr(run_service, "MAX_ACTIVE_RUNS_GLOBAL",
                        run_store.count_active_runs_total())
    with pytest.raises(run_service.ProtocolError) as ei:
        run_service.create_run(
            agent={"agent_id": aid_b, "session_id": sid_b, "name": "x"},
            agent_version=None,
            environment_id="us-equity-hourly-v1",
            config={"start_date": "2026-04-15", "end_date": "2026-04-16"},
        )
    assert ei.value.code == "too_many_active_runs_global"
    assert ei.value.status_code == 429
    assert ei.value.headers == {"Retry-After": "30"}


def test_zero_disables_the_global_cap(monkeypatch):
    key, _, _ = _agent("cap-disabled")
    _, sid_a, aid_a = _agent("cap-disabled-filler")
    _seed_active_run(aid_a, sid_a, n=3)
    monkeypatch.setattr(runs_mod, "MAX_ACTIVE_RUNS_GLOBAL", 0)
    monkeypatch.setattr(runs_mod, "BacktestBackend", _StubBackend)
    resp = client.post(
        "/api/v2/runs",
        json={"start_date": "2026-04-15", "end_date": "2026-04-16"},
        headers={"X-API-Key": key},
    )
    assert resp.status_code == 200, resp.text


def test_schema_endpoint_reports_the_new_code():
    resp = client.get("/api/v2/schema")
    assert "too_many_active_runs_global" in resp.json()["error_codes"]


def test_v1_global_cap_over_http_carries_retry_after(monkeypatch):
    """End-to-end v1: _handle_protocol_error must forward ProtocolError.headers
    onto the HTTPException so the 429 actually carries Retry-After."""
    key, _, _ = _agent("cap-v1-http")          # v2-registered key works on v1 too
    _, sid_f, aid_f = _agent("cap-v1-http-filler")
    _seed_active_run(aid_f, sid_f)
    monkeypatch.setattr(run_service, "MAX_ACTIVE_RUNS_GLOBAL",
                        run_store.count_active_runs_total())
    resp = client.post(
        "/api/v1/runs",
        json={"environment": {"type": "backtest",
                              "environment_id": "us-equity-hourly-v1"},
              "config": {"start_date": "2026-04-15", "end_date": "2026-04-16"}},
        headers={"X-API-Key": key},
    )
    assert resp.status_code == 429, resp.text
    assert resp.json()["detail"]["error"]["code"] == "too_many_active_runs_global"
    assert resp.headers.get("Retry-After") == "30"
```

(The global-cap raise fires inside `_create_lock` before `ebs.start_backtest`, so this test never touches the engine or a market-data loader.)

- [ ] **Step 2: Run to verify failure**

Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_global_run_cap.py -v`
Expected: FAIL — `AttributeError: 'RunStore' ... 'count_active_runs_total'`, `module ... has no attribute 'MAX_ACTIVE_RUNS_GLOBAL'`, `ProtocolError.__init__() got an unexpected keyword argument 'headers'`.

- [ ] **Step 3: Implement**

1. `domain/runs/repository.py`, after `count_active_runs` (line 283):

```python
    def count_active_runs_total(self) -> int:
        """Active runs across ALL agents — the global backstop cap's count.

        Deliberately reconciliation-free: both surfaces flip their row to a
        terminal status synchronously inside the finalizing request, so a raw
        COUNT tracks reality; the residual (terminal run, no subsequent poll)
        is bounded by the reaper interval and errs toward a spurious 429 —
        the safe direction, and Retry-After tells the client when to retry."""
        placeholders = ",".join("?" for _ in self._ACTIVE_STATUSES)
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT COUNT(*) FROM protocol_runs WHERE status IN ({placeholders})",
            self._ACTIVE_STATUSES,
        )
        (count,) = cursor.fetchone()
        conn.close()
        return int(count)
```

2. `domain/runs/service.py`, after `MAX_ACTIVE_RUNS_PER_AGENT` (line 52):

```python
# Global backstop across ALL agents (Decision 3 of the 2026-07-24 scale spec):
# beyond this, creates 429 with Retry-After instead of degrading everyone.
# Default 100 = the highest load ever exercised by the loadtest harness; raise
# it only after a measured smoke run at the higher value. 0 disables.
MAX_ACTIVE_RUNS_GLOBAL = int(os.getenv("MAX_ACTIVE_RUNS_GLOBAL", "100"))
```

3. `create_run`, inside `with _create_lock:` immediately after the per-agent block (after line 431):

```python
        if MAX_ACTIVE_RUNS_GLOBAL > 0:
            total = run_store.count_active_runs_total()
            if total >= MAX_ACTIVE_RUNS_GLOBAL:
                raise ProtocolError(
                    "too_many_active_runs_global",
                    f"The server is at its active-run capacity "
                    f"({total} of {MAX_ACTIVE_RUNS_GLOBAL}); retry shortly",
                    429,
                    details={"active_runs_total": total,
                             "limit": MAX_ACTIVE_RUNS_GLOBAL},
                    headers={"Retry-After": "30"},
                )
```

4. `domain/runs/protocol.py` `ProtocolError.__init__`:

```python
    def __init__(self, code: str, message: str, status_code: int = 400,
                 details: Any = None, headers: Optional[Dict[str, str]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details
        # Response headers (e.g. Retry-After on 429) — forwarded to the
        # HTTPException by the router's _handle_protocol_error.
        self.headers = headers or {}
```

5. `api/routers/runs.py:41-42`:

```python
def _handle_protocol_error(exc: ProtocolError):
    raise HTTPException(status_code=exc.status_code, detail=exc.to_body(),
                        headers=exc.headers or None)
```

6. `api/v2/errors.py` `ERROR_CODES`: append `"too_many_active_runs_global",` to the list.

7. `api/v2/runs.py`: extend the existing service import (line 32-35) to also bring `MAX_ACTIVE_RUNS_GLOBAL`, then insert after the per-agent 429 block (after line 356):

```python
        if MAX_ACTIVE_RUNS_GLOBAL > 0:
            total = run_repo.run_store.count_active_runs_total()
            if total >= MAX_ACTIVE_RUNS_GLOBAL:
                raise ApiError(
                    "too_many_active_runs_global",
                    f"The server is at its active-run capacity "
                    f"({total} of {MAX_ACTIVE_RUNS_GLOBAL}); retry shortly",
                    status=429, retryable=True,
                    details={"active_runs_total": total,
                             "limit": MAX_ACTIVE_RUNS_GLOBAL},
                    # Explicit: ApiError's auto Retry-After injection only
                    # fires for code "rate_limited".
                    headers={"Retry-After": "30"},
                )
```

Note: this file references the module-level name (like `MAX_ACTIVE_RUNS_PER_AGENT`), so tests patch `runs_mod.MAX_ACTIVE_RUNS_GLOBAL` — the established convention.

8. `tests/conftest.py`: `os.environ.pop("MAX_ACTIVE_RUNS_GLOBAL", None)`.

- [ ] **Step 4: Run the tests, then the suite**

Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_global_run_cap.py -v` → PASS.
Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/ -q` → green — in particular `test_concurrent_run_cap` (per-agent unchanged) and the three route-freeze tests (no routes added). If ambient runs from earlier tests push the global count near 100 and cause flakes, the seeded-count pattern above (cap = current count) keeps tests order-independent — do not hardcode absolute counts.

- [ ] **Step 5: Docs**

- `docs/api/agent-environment-protocol-v1.md:347` error table, after the per-agent 429 row:

```markdown
| 429  | `too_many_active_runs_global` | server-wide active-run capacity reached — retry after the `Retry-After` header (seconds) |
```

- `.env.example`:

```bash
# Global backstop: max active (non-terminal) runs across ALL agents; creates
# beyond this 429 with Retry-After. 0 disables. Default 100 — the tested
# number; raise only after a measured smoke run at the higher value.
# MAX_ACTIVE_RUNS_GLOBAL=100
```

- `CLAUDE.md` protocol bullet (line 86 area): mention the global cap alongside the deadline sentence, one line.

- [ ] **Step 6: Commit, push, open PR 3**

```bash
git add -A
git commit -m "feat: global active-run backstop cap (429 + Retry-After)"
git push -u origin feat/scale-t3-deadline-cap
gh pr create --title "feat: 60s decision deadline, timeout_holds, global run cap" --body "T3 of docs/superpowers/specs/2026-07-24-agent-scale-sustainability-design.md. Deadline default 30->60s; timeout_holds surfaced in status/result/metadata (closes the protocol-metadata gap); MAX_ACTIVE_RUNS_GLOBAL=100 backstop with 429+Retry-After on both surfaces, code registered in /api/v2/schema."
```

---

# Tier 4 — Auth cache, debounce, connection pool (branch `feat/scale-t4-auth-pool`)

### Task 10: Auth TTL cache + `last_used_at` debounce + invalidation

**Files:**
- Create: `dashboard/backend/domain/agents/auth_cache.py`
- Modify: `dashboard/backend/domain/agents/repository.py:342-360` (`resolve_api_key` gains `touch: bool = True`)
- Modify: `dashboard/backend/domain/agents/repository_postgres.py:309-325` (same)
- Modify: `dashboard/backend/api/protocol_auth.py:16-21`, `dashboard/backend/api/v2/auth_scopes.py:22-27` (route through the cache)
- Modify: `dashboard/backend/domain/agents/service.py:298-304` (invalidate on delete/rotate), `dashboard/backend/api/v2/agents.py:74` (invalidate on v2 rotate)
- Modify: `dashboard/backend/tests/conftest.py` (strip + reset), `.env.example`
- Test: `dashboard/backend/tests/test_auth_cache.py`

**Interfaces:**
- Produces: `auth_cache.resolve_api_key(api_key) -> Optional[dict]` (the hot-path entry); `auth_cache.invalidate_agent(agent_id)`; `AGENT_AUTH_CACHE_TTL_SECONDS` constant; `_reset_for_tests()`.
- Consumes: `repository.agent_store` **via module attribute at call time** (`_repo.agent_store`) so the per-test store swaps in `test_protocol_api.py`-style fixtures keep working; `_hash_api_key` from the repository module.
- Scope: only the two per-request hot paths go through the cache. Management endpoints (`dependencies.py`, `routers/agents.py`, `domain/agents/service.py` resolves) stay direct — always-fresh auth for security-sensitive operations.

- [ ] **Step 1: Check what patches the auth modules**

Run: `grep -rn "protocol_auth\|auth_scopes" dashboard/backend/tests/*.py | grep -i "setattr\|agent_store"`
Expected: `test_protocol_api.py:80` patches `protocol_auth.agent_store`. After rewiring, that attribute no longer exists — Step 5 removes those now-dead patch lines (the store swap at `agent_store_module.agent_store` covers the cache's late-bound lookup).

- [ ] **Step 2: Write the failing tests**

Create `dashboard/backend/tests/test_auth_cache.py`:

```python
"""Auth TTL cache + last_used_at debounce + invalidation (T4)."""

import pytest

import dashboard.backend.domain.agents.repository as repo_module
from dashboard.backend.domain.agents import auth_cache


class _RecordingStore:
    def __init__(self):
        self.resolves = []          # (api_key, touch)
        self.agents = {}            # key -> agent dict

    def add(self, api_key, agent_id):
        self.agents[api_key] = {"agent_id": agent_id, "session_id": f"s_{agent_id}",
                                "scopes": ["runs:read"]}

    def resolve_api_key(self, api_key, touch=True):
        self.resolves.append((api_key, touch))
        return dict(self.agents[api_key]) if api_key in self.agents else None


@pytest.fixture
def store(monkeypatch):
    s = _RecordingStore()
    s.add("key-A", "agent-A")
    monkeypatch.setattr(repo_module, "agent_store", s)
    auth_cache._reset_for_tests()
    monkeypatch.setattr(auth_cache, "_jitter", lambda: 1.0)  # deterministic TTL
    yield s
    auth_cache._reset_for_tests()


def test_hit_within_ttl_skips_the_db(store, monkeypatch):
    t = [1000.0]
    monkeypatch.setattr(auth_cache, "_now", lambda: t[0])
    a1 = auth_cache.resolve_api_key("key-A")
    a2 = auth_cache.resolve_api_key("key-A")
    assert a1["agent_id"] == a2["agent_id"] == "agent-A"
    assert len(store.resolves) == 1                    # second call: cache hit
    t[0] += auth_cache.AGENT_AUTH_CACHE_TTL_SECONDS + 1
    auth_cache.resolve_api_key("key-A")
    assert len(store.resolves) == 2                    # expired: DB again


def test_last_used_write_debounced_to_60s(store, monkeypatch):
    t = [1000.0]
    monkeypatch.setattr(auth_cache, "_now", lambda: t[0])
    auth_cache.resolve_api_key("key-A")
    assert store.resolves[-1] == ("key-A", True)       # first resolve touches
    t[0] += auth_cache.AGENT_AUTH_CACHE_TTL_SECONDS + 1
    auth_cache.resolve_api_key("key-A")
    assert store.resolves[-1] == ("key-A", False)      # <60s since last write
    t[0] += 61.0
    auth_cache.resolve_api_key("key-A")
    assert store.resolves[-1] == ("key-A", True)       # debounce window passed


def test_invalidate_agent_evicts_by_reverse_index(store, monkeypatch):
    monkeypatch.setattr(auth_cache, "_now", lambda: 1000.0)
    auth_cache.resolve_api_key("key-A")
    auth_cache.invalidate_agent("agent-A")
    auth_cache.resolve_api_key("key-A")
    assert len(store.resolves) == 2                    # cache was evicted


def test_zero_ttl_disables_caching(store, monkeypatch):
    monkeypatch.setattr(auth_cache, "AGENT_AUTH_CACHE_TTL_SECONDS", 0.0)
    auth_cache.resolve_api_key("key-A")
    auth_cache.resolve_api_key("key-A")
    assert len(store.resolves) == 2
    assert all(touch for _, touch in store.resolves)   # passthrough always touches


def test_misses_are_not_cached(store):
    assert auth_cache.resolve_api_key("nope") is None
    assert auth_cache.resolve_api_key("nope") is None
    assert len(store.resolves) == 2


def test_rotate_endpoint_invalidates_old_key():
    """End-to-end: after key rotation the OLD key must fail immediately, not
    after the TTL — exactly the behavior the cache would break without
    invalidate_agent wired into the rotate paths. Route verified:
    POST /api/v2/agents/{agent_id}/rotate-key (api/v2/agents.py:69)."""
    from fastapi.testclient import TestClient
    from dashboard.backend.app import app

    client = TestClient(app)
    r = client.post("/api/v2/agents", json={"name": "rotate-me"}).json()
    old_key, agent_id = r["api_key"], r["agent_id"]
    # Warm the auth cache with the old key: an authenticated read of a missing
    # run answers 404; an unauthenticated one answers 401. No run is created.
    probe = client.get("/api/v2/runs/does-not-exist", headers={"X-API-Key": old_key})
    assert probe.status_code == 404
    rot = client.post(f"/api/v2/agents/{agent_id}/rotate-key",
                      headers={"X-API-Key": old_key})
    assert rot.status_code == 200, rot.text
    denied = client.get("/api/v2/runs/does-not-exist",
                        headers={"X-API-Key": old_key})
    assert denied.status_code == 401                   # immediately, not <=TTL
```

- [ ] **Step 3: Run to verify failure**

Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_auth_cache.py -v`
Expected: FAIL at import — `auth_cache` doesn't exist.

- [ ] **Step 4: Write `dashboard/backend/domain/agents/auth_cache.py`**

```python
"""In-process TTL cache for API-key auth + last_used_at debounce (T4).

Sits ABOVE the repository so the SQLite/Postgres twins stay dumb. Only the
per-request hot paths (v1 resolve_agent_by_key, v2 auth_scopes.resolve_agent)
route through here; management endpoints resolve directly for always-fresh
auth. Revocation/rotation propagates within <=TTL for entries nobody
invalidates; the delete/rotate paths call invalidate_agent() for immediate
effect (required: rotate_api_key blind-UPDATEs by agent id, so the OLD hash —
the cache key — is never in scope there; hence the reverse index).
"""

from __future__ import annotations

import os
import random
import threading
import time
from typing import Any, Dict, Optional, Set, Tuple

from dashboard.backend.domain.agents import repository as _repo
from dashboard.backend.domain.agents.repository import _hash_api_key

# Read once at import; 0 disables. +-20% per-entry jitter keeps 100 agents'
# entries from expiring in lockstep and stampeding the DB/pool.
AGENT_AUTH_CACHE_TTL_SECONDS = float(os.getenv("AGENT_AUTH_CACHE_TTL_SECONDS", "10"))
LAST_USED_WRITE_INTERVAL_SECONDS = 60.0

_now = time.monotonic  # test seam


def _jitter() -> float:
    return random.uniform(0.8, 1.2)


_lock = threading.Lock()
_by_hash: Dict[str, Tuple[float, Dict[str, Any]]] = {}   # hash -> (expires_at, agent)
_hashes_by_agent: Dict[str, Set[str]] = {}               # agent_id -> {hashes}
_last_write: Dict[str, float] = {}                       # hash -> last touch time


def resolve_api_key(api_key: str) -> Optional[Dict[str, Any]]:
    """Cached resolve. Falls through to the store on miss/expiry; debounces
    the store's last_used_at write to once per LAST_USED_WRITE_INTERVAL."""
    if not api_key or not api_key.strip():
        return None
    ttl = AGENT_AUTH_CACHE_TTL_SECONDS
    if ttl <= 0:
        # Late-bound module attribute so per-test store swaps apply.
        return _repo.agent_store.resolve_api_key(api_key)

    key_hash = _hash_api_key(api_key.strip())
    now = _now()
    with _lock:
        hit = _by_hash.get(key_hash)
        if hit is not None and hit[0] > now:
            return dict(hit[1])
        should_touch = (now - _last_write.get(key_hash, float("-inf"))
                        >= LAST_USED_WRITE_INTERVAL_SECONDS)

    agent = _repo.agent_store.resolve_api_key(api_key, touch=should_touch)
    if agent is None:
        return None  # misses are not cached: an invalid key always re-checks

    with _lock:
        _by_hash[key_hash] = (now + ttl * _jitter(), dict(agent))
        _hashes_by_agent.setdefault(agent["agent_id"], set()).add(key_hash)
        if should_touch:
            _last_write[key_hash] = now
    return agent


def invalidate_agent(agent_id: str) -> None:
    """Immediate eviction for delete/rotate (old hash found via reverse index)."""
    with _lock:
        for key_hash in _hashes_by_agent.pop(agent_id, set()):
            _by_hash.pop(key_hash, None)
            _last_write.pop(key_hash, None)


def _reset_for_tests() -> None:
    with _lock:
        _by_hash.clear()
        _hashes_by_agent.clear()
        _last_write.clear()
```

- [ ] **Step 5: Wire it in**

1. `domain/agents/repository.py` `resolve_api_key` (line 342): signature → `def resolve_api_key(self, api_key: str, touch: bool = True) -> Optional[Dict[str, Any]]:`; wrap the UPDATE (lines 353–358) in `if row and touch:` instead of `if row:`.
2. `domain/agents/repository_postgres.py` `resolve_api_key` (line 309): same — `touch: bool = True`, `if row and touch:` around the UPDATE.
3. `api/protocol_auth.py`: replace the store import/use —

```python
from dashboard.backend.domain.agents import auth_cache


def resolve_agent_by_key(x_api_key: Optional[str]) -> Dict[str, Any]:
    """Resolve an Agent API key to the agent, or raise 401."""
    agent = auth_cache.resolve_api_key(x_api_key or "")
    if not agent:
        raise HTTPException(status_code=401, detail="Invalid or missing API key (X-API-Key)")
    return agent
```

Keep the module's existing `from ...repository import agent_store` **only if** something else in the file uses it; otherwise delete it and remove the now-dead `monkeypatch.setattr(protocol_auth, "agent_store", ...)` lines found in Step 1.
4. `api/v2/auth_scopes.py` `resolve_agent`: same swap (`auth_cache.resolve_api_key((x_api_key or "").strip())`).
5. `domain/agents/service.py` (lines ~298–304): in the service's `delete_agent` and `rotate_api_key` wrappers, add `from dashboard.backend.domain.agents import auth_cache` (module top) and call `auth_cache.invalidate_agent(agent_id)` immediately after the repo call succeeds.
6. `api/v2/agents.py:74` (v2 rotate): after `new_key = agent_store.rotate_api_key(agent_id)`, add `auth_cache.invalidate_agent(agent_id)`.
7. `tests/conftest.py`: `os.environ.pop("AGENT_AUTH_CACHE_TTL_SECONDS", None)`, and add to the autouse fixture (both sides of the yield):

```python
    from dashboard.backend.domain.agents import auth_cache
    auth_cache._reset_for_tests()
```

8. `.env.example`:

```bash
# Auth cache TTL for agent API keys (seconds; +-20% jitter; 0 disables).
# Rotation/deletion invalidates immediately; other agent-record edits
# propagate within the TTL. Default 10.
# AGENT_AUTH_CACHE_TTL_SECONDS=10
```

- [ ] **Step 6: Run the tests, then the auth-heavy suites, then everything**

Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_auth_cache.py -v` → PASS.
Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_protocol_api.py dashboard/backend/tests/test_run_lifecycle_unification.py -v` → PASS (per-test store swaps still reach auth through the late-bound `_repo.agent_store`).
Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/ -q` → green.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: auth TTL cache with debounced last_used_at"
```

---

### Task 11: Shared psycopg3 connection pool for the Postgres twins

**Files:**
- Modify: `requirements.txt:52` (`psycopg[binary]==3.3.4` → `psycopg[binary,pool]==3.3.4`)
- Create: `dashboard/backend/db_pool.py`
- Modify: `_get_connection` in the five twins: `dashboard/backend/domain/agents/repository_postgres.py:39-40`, `dashboard/backend/domain/agents/version_repository_postgres.py:~36`, `dashboard/backend/domain/portfolios/repository_postgres.py:~29`, `dashboard/backend/domain/strategies/repository_postgres.py:~39`, `dashboard/backend/users_postgres.py:~32`
- Test: `dashboard/backend/tests/test_db_pool.py`

**Interfaces:**
- Produces: `db_pool.get_pool(database_url) -> ConnectionPool` — one cached pool per URL, `max_size=5`, `max_idle=300`, `kwargs={"row_factory": dict_row}` (all five twins rely on dict-style rows — dropping this silently turns them into tuples).
- Twins' `_get_connection()` returns `get_pool(...).connection()` — a context manager with the same commit-on-exit transaction semantics as `psycopg.connect(...)`, so **no method-body changes** (verified: all five use `with self._get_connection() as conn:` exclusively — re-verify in Step 1).

- [ ] **Step 1: Verify the context-manager-only claim**

Run: `grep -rn "self._get_connection()" dashboard/backend/domain/agents/repository_postgres.py dashboard/backend/domain/agents/version_repository_postgres.py dashboard/backend/domain/portfolios/repository_postgres.py dashboard/backend/domain/strategies/repository_postgres.py dashboard/backend/users_postgres.py | grep -v "with self._get_connection() as conn"`
Expected: only the `def _get_connection` lines. Any bare `conn = self._get_connection()` caller must be converted to the `with` form in this task.

- [ ] **Step 2: Write the failing tests**

Create `dashboard/backend/tests/test_db_pool.py`:

```python
"""Shared per-URL psycopg pool (T4). Unit tests need no live Postgres; the
@pg_only round-trip follows the established local-postgres fixture rules."""

import os

import pytest

pytest.importorskip("psycopg_pool")

from dashboard.backend import db_pool


class _FakePool:
    instances = []

    def __init__(self, url, **kwargs):
        self.url = url
        self.kwargs = kwargs
        type(self).instances.append(self)

    def connection(self):
        raise AssertionError("not used in dispatch tests")


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(db_pool, "ConnectionPool", _FakePool)
    db_pool._reset_for_tests()
    _FakePool.instances = []
    yield
    db_pool._reset_for_tests()


def test_one_pool_per_url_cached():
    p1 = db_pool.get_pool("postgresql://u@h/db1")
    p2 = db_pool.get_pool("postgresql://u@h/db1")
    p3 = db_pool.get_pool("postgresql://u@h/db2")
    assert p1 is p2
    assert p1 is not p3
    assert len(_FakePool.instances) == 2


def test_pool_configured_for_neon_and_dict_rows():
    from psycopg.rows import dict_row

    db_pool.get_pool("postgresql://u@h/db")
    kwargs = _FakePool.instances[0].kwargs
    assert kwargs["max_size"] == 5
    assert kwargs["max_idle"] == 300          # < Neon scale-to-zero idle window
    assert kwargs["kwargs"] == {"row_factory": dict_row}


TEST_PG = os.getenv("TEST_POSTGRES_URL")
pg_only = pytest.mark.skipif(not TEST_PG, reason="TEST_POSTGRES_URL not set")


@pg_only
def test_pooled_agent_store_round_trip(monkeypatch):
    """A twin resolves through the real pool. Guard: never a prod URL."""
    from psycopg_pool import ConnectionPool as RealPool

    from dashboard.backend.tests._postgres_testing import require_local_postgres_url

    require_local_postgres_url(TEST_POSTGRES_URL)
    monkeypatch.setattr(db_pool, "ConnectionPool", RealPool)  # replace the fake
    db_pool._reset_for_tests()
    from dashboard.backend.domain.agents.repository_postgres import PostgresAgentStore

    store = PostgresAgentStore(TEST_PG)
    created = store.create_agent(name="pool-probe", model_name="m",
                                 agent_type="external", description="")
    resolved = store.resolve_api_key(created["api_key"])
    assert resolved and resolved["agent_id"] == created["agent_id"]
    store.delete_agent(created["agent_id"])
    db_pool._reset_for_tests()  # close the real pool before the fake returns
```

Match `create_agent`'s real signature against `test_agent_store_postgres.py` (same file provides the `require_local_postgres_url` localhost-only guard pattern) and adjust the kwargs if they differ.

- [ ] **Step 3: Run to verify failure**

Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_db_pool.py -v`
Expected: FAIL — `dashboard.backend.db_pool` doesn't exist (or `psycopg_pool` missing → first `pip install "psycopg[binary,pool]==3.3.4"` into `~/atl-venv`).

- [ ] **Step 4: Implement**

1. `requirements.txt:52`: `psycopg[binary]==3.3.4` → `psycopg[binary,pool]==3.3.4`, then `~/atl-venv/bin/pip install "psycopg[binary,pool]==3.3.4"`.
2. Create `dashboard/backend/db_pool.py`:

```python
"""Shared psycopg3 connection pools, one per database URL (T4).

Replaces the fresh psycopg.connect() (a full TLS handshake to Neon) every
store call used to pay. Small and short-lived by design: max_size 5 fits the
single-worker deployment, and max_idle 300s closes idle sockets before Neon's
scale-to-zero suspend can hand back a dead one. row_factory is configured at
pool construction because every twin relies on dict-style row access.
"""

from __future__ import annotations

import threading
from typing import Dict

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from dashboard.backend.db_url import describe_database_url

_pools: Dict[str, ConnectionPool] = {}
_lock = threading.Lock()


def get_pool(database_url: str) -> ConnectionPool:
    """One cached pool per URL; construction is lazy and logged."""
    with _lock:
        pool = _pools.get(database_url)
        if pool is None:
            pool = ConnectionPool(
                database_url,
                min_size=0,
                max_size=5,
                max_idle=300,
                kwargs={"row_factory": dict_row},
                open=True,
            )
            _pools[database_url] = pool
            print(f"🏊 pg pool created for {describe_database_url(database_url)}")
    return pool


def _reset_for_tests() -> None:
    with _lock:
        for pool in _pools.values():
            try:
                pool.close()
            except Exception:
                pass
        _pools.clear()
```

Version notes: if the pinned `psycopg_pool` rejects `min_size=0`, use `min_size=1` (the 300 s `max_idle` still lets Neon suspend). If it warns about `open=` in the constructor, construct with `open=False` and call `pool.open(wait=False)` before returning. Verify `describe_database_url` exists in `db_url.py` (`grep -n describe_database_url dashboard/backend/db_url.py`); it never emits credentials.

3. Swap each twin's `_get_connection` (e.g. `repository_postgres.py:39-40`):

```python
    def _get_connection(self):
        # Pooled checkout: same context-manager transaction semantics as
        # psycopg.connect (commit on clean exit), returned to the pool on close.
        from dashboard.backend.db_pool import get_pool
        return get_pool(self.database_url).connection()
```

Apply the identical change in all five twins (the import is function-local to keep `psycopg_pool` an import-time no-op for SQLite-only deployments). SQLite stores are untouched.

- [ ] **Step 5: Run the tests**

Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/test_db_pool.py -v` → dispatch tests PASS, `@pg_only` skips locally (no docker/sudo on this machine — it runs on CI; after pushing, grep the CI job log to confirm the pg tier actually executed, per the `no-local-postgres-for-pg-only-tier` rule).
Run: `~/atl-venv/bin/python -m pytest dashboard/backend/tests/ -q` → green.

- [ ] **Step 6: Commit, push, open PR 4**

```bash
git add -A
git commit -m "perf: pooled pg connections for the store twins"
git push -u origin feat/scale-t4-auth-pool
gh pr create --title "perf: auth cache + last_used_at debounce + pg connection pool" --body "T4 of docs/superpowers/specs/2026-07-24-agent-scale-sustainability-design.md. Hot-path auth: TTL cache (10s, jittered, rotation-invalidated), last_used_at writes debounced to 1/min/key, and one shared psycopg pool per URL (max 5, max_idle 300s) replacing a fresh Neon TLS connect per store call. New env var: AGENT_AUTH_CACHE_TTL_SECONDS (10). Verify the @pg_only pool test ran in CI before merging."
```

---

### Task 12: Acceptance run + wrap-up

**Files:** none (measurement + memory/docs bookkeeping)

- [ ] **Step 1: Full-suite gate**

On the final branch (all tiers merged or stacked): `~/atl-venv/bin/python -m pytest dashboard/backend/tests/ -q` and `~/atl-venv/bin/python -m pytest packaging/agentictrading/tests/ -q` — both green.

- [ ] **Step 2: 100-agent acceptance run**

Run the Task 1 harness at `N_AGENTS=100` / `drive_agents.py 100`. Acceptance (local dev hardware, 21-step runs): **0 server timeout_holds, 0 FAILURES, create p95 < 1000 ms, decision p95 < 1000 ms, total wall < 60 s, RSS growth < 100 MB.** Paste the report into the last PR. If any criterion misses, diagnose before merging (systematic-debugging) — do not relax the criteria.

- [ ] **Step 3: Post-deploy prod smoke (after the PRs merge and auto-deploy)**

One 100-agent smoke against prod with a throwaway config window, watching the startup lines (`market-data dataset built`, `pg pool created`) and `timeout_holds` in results. Manual/observational — no repo change.

- [ ] **Step 4: Surface the user-facing-docs follow-ups**

Report (do not edit mid-session): ReadTheDocs protocol page + PyPI README changes ship with the next SDK release. (`app.html:943`'s "10 min" copy was re-verified 2026-07-24 and is accurate — it matches `BACKTEST_POLL_MAX_SECONDS = 600` in `app.js:18` / `pollBacktestStatus`; no follow-up needed. The three dashboard-path fixes are filed as #201/#202/#203.)
