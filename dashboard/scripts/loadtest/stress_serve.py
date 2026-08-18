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
import dashboard.backend.domain.backtesting.baseline_worker as baseline_worker  # noqa: E402

ebs.AlpacaDataLoader = FakeAlpacaLoader
# HourlyBacktester.__init__ always calls create_market_data_provider(data_source,
# universe) positionally with BOTH args (engine.py), even though both are
# defaulted in the real signature. A one-arg lambda here raised TypeError on
# every call — silently, because baseline generation swallows job failures
# (baseline_worker.py) and nothing else in the harness ever hits this path.
engine_mod.create_market_data_provider = lambda *a, **k: FakeAlpacaLoader()

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

try:
    uvicorn.run("dashboard.backend.app:app", host="127.0.0.1", port=8402, log_level="warning")
finally:
    # Read the counter, don't scrape stdout — a harness break must be as loud
    # as the production one baseline_worker now raises on its own.
    failed = baseline_worker._total_failures
    if failed:
        print(f"⚠️  SHUTDOWN SUMMARY: {failed} baseline job(s) failed during this run "
              f"— see warnings above", flush=True)
    else:
        print("SHUTDOWN SUMMARY: 0 baseline job failures", flush=True)
