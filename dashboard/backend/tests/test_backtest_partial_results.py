"""Partial-result recovery for interrupted backtests.

Covers the three pieces of the design:
1. The engine's throttled durable flush (_flush_live_progress_durable) —
   exercised here against a real sqlite store via the public store methods,
   including the throttle cadence and the swallowed-failure contract.
2. The startup reclaimer (reclaim_interrupted_backtests) — snapshot without
   a terminal run becomes an interrupted run with a partial curve; snapshot
   WITH a terminal run is stale and dropped.
3. The metadata contract the frontend reads (interrupted / step markers).
"""

import json
from pathlib import Path

import pytest

from dashboard.backend.database import BacktestDatabase
from dashboard.backend.domain.backtesting.partial_results import (
    reclaim_interrupted_backtests,
)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_RUNS_DATABASE_URL", "")
    db = BacktestDatabase(db_path=tmp_path / "runs.db")
    return db


def _snapshot(run_id="run_a", step=14, total=49, equity=10500.0):
    return {
        "run_id": run_id,
        "session_id": "sess-1",
        "agent_name": "TestAgent",
        "payload": {
            "run_id": run_id,
            "step": step,
            "total_steps": total,
            "equity_curve": [
                {"timestamp": "2026-08-01T14:00:00", "equity": equity - 500,
                 "cash": 9000.0, "positions_value": 1000.0},
                {"timestamp": "2026-08-01T15:00:00", "equity": equity,
                 "cash": 9200.0, "positions_value": 1300.0},
            ],
            "trades": [
                {"symbol": "AAPL", "side": "buy", "qty": 10, "price": 100.0,
                 "timestamp": "2026-08-01T15:00:00"},
            ],
            "run_metadata": {
                "start_date": "2026-08-01", "end_date": "2026-08-31",
                "mode": "llm", "llm_model": "gpt-test",
            },
        },
    }


def test_reclaim_turns_snapshot_into_interrupted_run(store, monkeypatch):
    snap = _snapshot()
    store.upsert_live_progress(
        snap["run_id"], snap["session_id"], snap["agent_name"], snap["payload"]
    )
    monkeypatch.setattr("dashboard.backend.database.db", store, raising=False)

    counts = reclaim_interrupted_backtests()

    assert counts["reclaimed"] == 1
    run = store.get_run("run_a")
    assert run is not None
    meta = json.loads(run["metadata"]) if isinstance(run.get("metadata"), str) else run.get("metadata")
    assert meta["interrupted"] is True
    assert meta["interrupted_step"] == 14
    assert meta["interrupted_total_steps"] == 49
    # Partial curve landed in the regular table: every existing reader works.
    curve = store.get_equity_curve("run_a")
    assert len(curve) == 2
    assert curve[-1]["equity"] == pytest.approx(10500.0)
    trades = store.get_trades("run_a")
    assert len(trades) == 1
    # Snapshot consumed
    assert store.get_live_progress("run_a") is None


def test_reclaim_drops_stale_snapshot_of_completed_run(store, monkeypatch):
    # A terminal row already exists → the snapshot is a leftover from the
    # completion path, not an interrupted run.
    store.insert_run(
        "run_b", "sess-1", "Agent", "llm", "2026-08-01", "2026-08-31",
        10000.0, final_equity=11000.0,
    )
    snap = _snapshot(run_id="run_b")
    store.upsert_live_progress(
        "run_b", snap["session_id"], snap["agent_name"], snap["payload"]
    )
    monkeypatch.setattr("dashboard.backend.database.db", store, raising=False)

    counts = reclaim_interrupted_backtests()

    assert counts["reclaimed"] == 0
    assert counts["dropped_stale"] == 1
    # The completed run's own numbers were NOT overwritten by the snapshot.
    run = store.get_run("run_b")
    assert run["final_equity"] == pytest.approx(11000.0)
    assert store.get_live_progress("run_b") is None


def test_reclaim_is_idempotent(store, monkeypatch):
    snap = _snapshot()
    store.upsert_live_progress(
        snap["run_id"], snap["session_id"], snap["agent_name"], snap["payload"]
    )
    monkeypatch.setattr("dashboard.backend.database.db", store, raising=False)
    first = reclaim_interrupted_backtests()
    second = reclaim_interrupted_backtests()
    assert first["reclaimed"] == 1
    assert second["reclaimed"] == 0
    assert len(store.get_equity_curve("run_a")) == 2


def test_engine_flush_cadence_and_tombstone(store, monkeypatch):
    """The engine-side throttle: step 1 always flushes (a run that dies young
    still leaves a tombstone), then every 10th step; other steps are no-ops."""
    from dashboard.backend.domain.backtesting.engine import HourlyBacktester

    monkeypatch.setattr("dashboard.backend.database.db", store, raising=False)
    calls = []

    def fake_upsert(run_id, session_id, agent_name, payload):
        calls.append(step_holder["step"])

    step_holder = {"step": 0}
    engine = object.__new__(HourlyBacktester)
    engine.live_run_id = "run_x"
    engine.session_id = "sess-x"
    engine.agent_name = "X"
    original = store.upsert_live_progress
    monkeypatch.setattr(store, "upsert_live_progress", fake_upsert)

    for step in range(1, 31):
        step_holder["step"] = step
        engine._flush_live_progress_durable(step, {"step": step})

    assert calls == [1, 10, 20, 30]


def test_engine_flush_swallows_db_failure():
    from dashboard.backend.domain.backtesting.engine import HourlyBacktester

    engine = object.__new__(HourlyBacktester)
    engine.live_run_id = "run_y"
    engine.session_id = "s"
    engine.agent_name = "Y"

    class Boom:
        def upsert_live_progress(self, *a, **k):
            raise RuntimeError("db down")

    import dashboard.backend.database as database_module
    original_db = database_module.db
    database_module.db = Boom()
    try:
        # Must not raise: the bar loop continues regardless of DB health.
        engine._flush_live_progress_durable(10, {"step": 10})
    finally:
        database_module.db = original_db
