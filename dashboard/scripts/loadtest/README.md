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
