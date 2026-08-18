# Protocol-agent load test

Reproduces the 2026-07-24 concurrency measurements (spec:
`docs/superpowers/specs/2026-07-24-agent-scale-sustainability-design.md`).

Hermetic: synthetic market data (no Alpaca), fresh temp-dir SQLite, localhost
only, no credentials. Never writes into the repo tree.

## Run

Terminal 1 (from the repo root):

    N_AGENTS=100 python dashboard/scripts/loadtest/stress_serve.py
    # prints:  artifacts dir: /tmp/atl_loadtest_XXXX
    # on shutdown (Ctrl-C), prints a SHUTDOWN SUMMARY line naming how many
    # baseline jobs failed during the run (0 on a clean run).

Terminal 2:

    python dashboard/scripts/loadtest/drive_agents.py 100 --artifacts /tmp/atl_loadtest_XXXX

## Flags

- `--windows shared|distinct` (default `shared`) on `drive_agents.py`.
  `shared` gives every agent the same date range, so background baseline
  generation dedups to a single queued job for the whole run. `distinct`
  offsets each agent's start date by its index in **whole weeks** (same
  span, same trading-day count for every agent), so every agent gets its own
  baseline config — N serialized baseline backtests instead of one. Use
  `distinct` to see baseline-worker cost scale with agent count instead of
  being hidden by dedup. (A per-day offset instead of per-week was tried
  first and silently starved windows that crossed a weekend down to 1-2
  trading days instead of 3 — see the comment above `date_window()` in
  `drive_agents.py`.)
- `N_AGENTS` (env var, default `100`) on `stress_serve.py` — how many
  protocol agents to pre-seed.

Baseline generation is asynchronous: finalize enqueues the job and returns
immediately, so the run is already `completed` before the worker even
dequeues it, and `drive_agents.py`'s reported wall time never waits on
baselines. `distinct`'s extra cost is real (N serialized `HourlyBacktester`
builds instead of one deduped build) but structurally invisible to that
timer. The signal that `distinct` is doing more work is the server-side
count of baselines the worker actually **executed** (1 unique config under
`shared` vs. N under `distinct`), not the wall time `drive_agents.py` prints.

Count executions, and count them with this exact grep:

    grep -c "Running Buy & Hold baseline" <the stress_serve.py log>

⚠ **Do not read that off the queue-depth line, which is the loud number and
says the opposite.** Dedup lives at the *consumer* — `_run_job` keys
`_completed` on `(start_date, end_date, mode)` and skips the work
(`baseline_worker.py`) — while `submit()` enqueues unconditionally. So under
`shared` the queue genuinely does grow to ~N and stdout fills with
`⚠️ Baseline queue depth N — worker backlog` warnings (a 100-agent run
printed 24 of them, peaking at 37) while exactly **one** baseline executed.
Those are cheap cache-hit jobs draining, not a capacity problem — but read
as "the server-side baseline-job count" they look exactly like broken dedup.

## ⚠ Figures produced before 2026-08-18 are a floor, not a measurement

Until 2026-08-18, `stress_serve.py` patched `create_market_data_provider`
with a **one-argument** lambda against the real **two-argument**
`create_market_data_provider(data_source, universe)` signature —
`HourlyBacktester.__init__` always calls it positionally with both args, so
every call raised `TypeError`. Background baseline generation
(`baseline_worker.py`) swallows job failures as a per-item printed warning
and keeps draining, so this broke **every baseline job, every run, on every
rung of the ladder**, silently: no baseline `HourlyBacktester` was ever
actually allocated. The 2026-08-18 ladder rungs (and everything before them)
are in this category — their CPU and RSS numbers understate real load,
because a real workload's baseline generation does allocate.

The patch is now `lambda *a, **k: FakeAlpacaLoader()`, matching the real
signature, and `baseline_worker.py` now escalates loudly (a printed line
naming the count and last exception) if 3 baseline jobs fail consecutively,
so a regression like this can't go unnoticed again. `stress_serve.py` also
prints a shutdown summary of total baseline failures.

The fresh 100-agent run taken after an **ad-hoc local repair** of this same
bug (0.522 CPU-s, 311 MB RSS) is **not** in the understated category — it
was measured with baselines actually running. Do not average it with the
earlier rungs; they are not measuring the same thing.

## Emulating a hosting tier

The dev box has 12 cores; every tier this project can afford has one or a
fraction of one. Running unconstrained measures hardware nobody deploys on,
so confine the **server** and pin the driver to other cores:

    # Render Standard (1.0 CPU)
    N_AGENTS=100 taskset -c 0 python dashboard/scripts/loadtest/stress_serve.py

    # Render Free (0.1 CPU)
    systemd-run --user --scope -p CPUQuota=10% \
        env N_AGENTS=100 python dashboard/scripts/loadtest/stress_serve.py

    # driver, in both cases
    taskset -c 2-11 python dashboard/scripts/loadtest/drive_agents.py 100 --artifacts /tmp/atl_loadtest_XXXX

Verify the quota actually binds before trusting a run (a busy-loop probe
should burn ~0.1 CPU-s per wall-second). Results: spec §5.

This is *not* a substitute for running against Render, but it is the closest
thing available, because **the harness cannot be pointed at a deployed
instance**: every hermetic property above is an in-process monkeypatch inside
`stress_serve.py` (the market-data provider swap, the `DATABASE_PATH`
redirect, the seeded agents and their keys). `--allow-remote` only relaxes a
hostname guard — it supplies none of them. Against a real deployment the
driver would find no agents, and the runs it created would hit real Alpaca
and write to real stores.

⚠ **Do not extrapolate one tier's numbers to another by dividing CPU-seconds
by the tier's CPU budget.** That method was used to size this stack and was
wrong in *both* directions — 2× pessimistic on Standard, 35% optimistic on
Free. A cgroup quota does not slow work down smoothly; it freezes the process
for ~90 ms of every 100 ms, including mid-transaction, so throttled CPU does
less useful work per CPU-second than dedicated CPU. Measure each tier.

## Acceptance target (100 agents, 21-step runs, local dev hardware)

0 timeout_holds, 0 failures, create p95 < 1 s, decision p95 < 1 s,
total wall < 60 s, server RSS growth < 100 MB.

⚠ **Auto-holds have three instruments and none of them is complete.** Rank
them in this order and take the largest:

1. **Client-observed deadline losses** — `drive_agents.py` counts a loss only
   when the server answers a decision with **409 + "deadline"/"finalized"**,
   i.e. the server itself reporting that it auto-held the step under the
   agent. Authoritative and cannot be fabricated. Trust this one first.
2. **The server log** — `grep -c "decision deadline"`. A **lower bound**: it
   misses the path where `get_status` applies the hold *before* the
   instrumented loop in `get_current_step` runs, so the loop sees no delta
   and logs nothing. Measured: a 25-agent Standard run logged **0** while the
   client observed **1**. Issue #375 is a second, separate blind spot (the
   fourth deadline branch in `submit_decisions`, silent on the legacy and v2
   surfaces).
3. **The `timeout_holds` counter** — weakest. Until 2026-08-18 an unreadable
   count printed as `0`, so this criterion could not fail; it now prints
   `unknown`. At load it is unknown for essentially every run (a 100-agent
   Standard run: `unknown [100/100 runs unreadable]`), because the live
   session it reads from has usually been swept by the time the run is
   polled.

A run is only clean when **all three** are zero. Earlier guidance in this
repo said "when they disagree, believe the log" — that is wrong, and was
corrected once the client-observed count was seen to exceed it.
