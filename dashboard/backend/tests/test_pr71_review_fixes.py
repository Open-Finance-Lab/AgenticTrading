"""Regression tests for the PR #71 review findings.

Each test pins a defect surfaced by the review and is red before its fix:

1. cancel-during-loading race — load_market_data / background _load must not
   resurrect a run a concurrent cancel() moved to "closed".
2. archived status() must keep baseline_run_ids + compare_url for a completed
   run (the live get_status returns them; the tombstone dropped them).
3. a rehydrated failed/closed run must report its real step counts, not 0/0.
4. finalize must publish "completed" before the slow best-effort baseline
   generation, so cancel()/is_active() (lock-free reads) don't block on it.
5. the v2 _runs registry must be bounded by ACTIVE runs — the reaper evicts
   archived tombstones (a later read rehydrates from the terminal row).
"""

import threading
import uuid

from fastapi.testclient import TestClient

import dashboard.backend.api.v2.runs as runs_mod
import dashboard.backend.domain.backtesting.external_run_service as svc
import dashboard.backend.domain.runs.repository as run_store_module
from dashboard.backend.app import app
from dashboard.backend.database import BacktestDatabase
from dashboard.backend.execution.backtest_backend import (
    ArchivedBacktestBackend,
    BacktestBackend,
)
from dashboard.backend.tests._v2_fakes import FakeBackend

client = TestClient(app)
run_store = run_store_module.run_store


def _agent(name):
    r = client.post("/api/v2/agents", json={"name": name}).json()
    return r["api_key"], r["session_id"], r["agent_id"]


# ---------------------------------------------------------------------------
# 1. cancel-during-loading race
# ---------------------------------------------------------------------------


def test_load_market_data_does_not_resurrect_a_cancelled_run(monkeypatch):
    """A cancel() that lands while the background fetch is in flight writes
    'closed' under _step_lock. load_market_data's publish must respect that
    terminal state instead of overwriting it back to waiting_decision."""
    session = svc.ExternalBacktestSession(
        backtest_id="bt_cancel_load", session_id="s", agent_name="a",
        model_name="m", start_date="2026-04-15", end_date="2026-04-16",
    )

    # load_market_data now delegates to the shared store; stub get_dataset so
    # the terminal-status guard in adopt_dataset is exercised without a fetch.
    class _FakeDataset:
        all_data = {"AAA": object()}
        timestamps = ["ts0"]
        price_cache = {"AAA": {"ts0": 1.0}}
        total_steps = 1

    monkeypatch.setattr(
        svc.market_data_store, "get_dataset", lambda *a, **k: _FakeDataset(),
    )

    # The agent cancelled while the fetch was running.
    session.status = "closed"
    session.load_market_data()

    assert session.status == "closed", "load must not resurrect a cancelled run"


def test_background_load_does_not_revive_a_cancelled_row(monkeypatch):
    """The backend's _load success path must not stamp the row 'running' when
    a cancel() won during loading (the session's fixed guard left it terminal).
    Without the guard, _load unconditionally revived the row to 'running'."""
    run_id = f"run_loadrace_{uuid.uuid4().hex[:6]}"
    run_store.create_run(
        run_id=run_id, agent_id="ag_lr", agent_version_id=None,
        session_id="sess_lr", environment_id=None, environment_type="backtest",
        config={}, backtest_id=None, status="loading",
    )

    done = threading.Event()

    class _CancelledDuringLoad:
        def __init__(self):
            self._step_lock = threading.Lock()
            self.status = "loading"
            self.error = None
            # start_background_load's peek fast path reads these; the store is
            # reset per test, so peek cold-misses and _load runs as intended.
            self.start_date = "2026-04-15"
            self.end_date = "2026-04-16"

        def load_market_data(self):
            # A cancel() landed mid-load; load_market_data's terminal guard
            # (the real fix) leaves the run "closed" instead of publishing
            # waiting_decision.
            self.status = "closed"
            done.set()

    backend = BacktestBackend.__new__(BacktestBackend)
    backend.run_id = run_id
    backend.session = _CancelledDuringLoad()
    backend.start_background_load()

    assert done.wait(timeout=3)
    threading.Event().wait(0.2)  # let _load finish its post-load row write

    assert run_store.get_run(run_id)["status"] != "running", (
        "a cancelled run's row must not be revived to 'running' by the loader"
    )


# ---------------------------------------------------------------------------
# 2. archived status() keeps baseline_run_ids + compare_url
# ---------------------------------------------------------------------------


def test_archived_status_preserves_baseline_run_ids_and_compare_url(tmp_path, monkeypatch):
    """Once archived, a completed run's status() must still carry the same
    baseline_run_ids + compare_url the live get_status() returns."""
    tdb = BacktestDatabase(tmp_path / "arch.db")
    monkeypatch.setattr("dashboard.backend.execution.backtest_backend.db", tdb)

    result_run_id = "run_arch_complete"
    tdb.insert_run(
        run_id=result_run_id, session_id="s", agent_name="a", mode="backtest",
        start_date="2026-04-15", end_date="2026-04-16", initial_equity=100000.0,
        final_equity=101000.0, total_return=0.01,
    )
    tdb.update_run_baselines(
        result_run_id, djia_run_id="run_djia_x", buyhold_run_id="run_bh_x",
    )

    archived = ArchivedBacktestBackend(
        run_id=result_run_id, session_id="s", status="completed",
        result_run_id=result_run_id, step_index=3, total_steps=3,
    )
    body = archived.status()

    assert body["baseline_run_ids"] == {
        "buy_and_hold": "run_bh_x", "djia": "run_djia_x",
    }
    # Same ordering as the live _compare_url: run_id, djia, buy_and_hold.
    assert body["compare_url"] == (
        f"/compare?run_ids={result_run_id},run_djia_x,run_bh_x"
    )


# ---------------------------------------------------------------------------
# 3. rehydrated failed/closed run reports real step counts
# ---------------------------------------------------------------------------


def test_archive_persists_step_counts_for_failed_run(monkeypatch):
    """_archive_run must persist step_index/total_steps so a post-restart
    from_record recovers real progress for a FAILED run (not a bogus 0/0 —
    a failed run has no decision-log rows to recover from)."""
    run_id = f"run_failsteps_{uuid.uuid4().hex[:6]}"
    run_store.create_run(
        run_id=run_id, agent_id="ag_fs", agent_version_id=None,
        session_id="sess_fs", environment_id=None, environment_type="backtest",
        config={}, backtest_id=None, status="running",
    )
    fake = FakeBackend(run_id=run_id, total_steps=5, session_id="sess_fs")
    fake.step_index = 2
    fake._status = "failed"
    runs_mod.register_run(run_id, fake, "sess_fs", "ag_fs")

    runs_mod._archive_run(run_id, runs_mod._runs[run_id], fake)

    record = run_store.get_run(run_id)
    assert record["step_index"] == 2
    assert record["total_steps"] == 5
    # Post-restart rehydration path: no live session, DB row only.
    rehydrated = ArchivedBacktestBackend.from_record(record)
    assert rehydrated.step_index == 2
    assert rehydrated.total_steps == 5


# ---------------------------------------------------------------------------
# 4. finalize publishes "completed" before slow baseline generation
# ---------------------------------------------------------------------------


def test_finalize_marks_completed_before_baselines(monkeypatch, tmp_path):
    """cancel()/is_active() read session.status without _step_lock. If status
    only flips to 'completed' AFTER baseline generation (seconds-to-minutes
    under the lock), those reads are wrong/blocked. Status must be set first."""
    tdb = BacktestDatabase(tmp_path / "fin.db")
    monkeypatch.setattr(svc, "db", tdb)

    started = threading.Event()
    release = threading.Event()

    class _BlockingBacktester:
        def __init__(self, *a, **k):
            self.all_data = {}

        def run_buyhold_baseline(self):
            started.set()
            release.wait(timeout=5)
            return (None, None)

        def run_djia_baseline(self):
            return (None, None)

    from dashboard.backend.domain.backtesting import baseline_worker as bw
    monkeypatch.setattr(bw, "HourlyBacktester", _BlockingBacktester)

    session = svc.ExternalBacktestSession(
        backtest_id="bt_fin", session_id="s", agent_name="a", model_name="m",
        start_date="2026-04-15", end_date="2026-04-16",
    )
    session.run_id = f"run_fin_{uuid.uuid4().hex[:6]}"
    session.total_steps = 1
    session.step_index = 1

    t = threading.Thread(target=session._finalize, daemon=True)
    t.start()
    try:
        assert started.wait(timeout=3), "baseline generation never started"
        assert session.status == "completed", (
            "status must be 'completed' before the slow baseline block runs"
        )
    finally:
        release.set()
        t.join(timeout=5)
    assert session.status == "completed"


# ---------------------------------------------------------------------------
# 5. the v2 _runs registry is bounded (tombstones are evicted)
# ---------------------------------------------------------------------------


def test_reaper_evicts_archived_tombstones(monkeypatch):
    """A tombstone archived in a prior pass must be evicted from _runs on the
    next pass (a later read rehydrates it from the terminal row), so _runs is
    bounded by ACTIVE runs, not total historical runs."""
    key, sid, agent_id = _agent("prune-owner")
    run_id = f"run_prune_{uuid.uuid4().hex[:6]}"
    fake = FakeBackend(run_id=run_id, total_steps=1, session_id=sid)
    run_store.create_run(
        run_id=run_id, agent_id=agent_id, agent_version_id=None, session_id=sid,
        environment_id=None, environment_type="backtest", config={},
        backtest_id=None, status="running",
    )
    runs_mod.register_run(run_id, fake, sid, agent_id)
    fake.apply_decisions([])  # completes the fake

    runs_mod.reap_v2_runs()  # pass 1: archives -> tombstone kept in _runs
    with runs_mod._lock:
        assert isinstance(runs_mod._runs[run_id]["backend"], ArchivedBacktestBackend)

    runs_mod.reap_v2_runs()  # pass 2: evicts the prior-pass tombstone
    with runs_mod._lock:
        assert run_id not in runs_mod._runs, "tombstone must be evicted"

    # Reads still work — the terminal row rehydrates on demand.
    resp = client.get(f"/api/v2/runs/{run_id}", headers={"X-API-Key": key})
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"
