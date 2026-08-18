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
baseline-job count (1 unique config under `shared` vs. N under `distinct`,
visible in `stress_serve.py`'s stdout), not the wall time `drive_agents.py`
prints.

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

## Acceptance target (100 agents, 21-step runs, local dev hardware)

0 timeout_holds, 0 failures, create p95 < 1 s, decision p95 < 1 s,
total wall < 60 s, server RSS growth < 100 MB.
