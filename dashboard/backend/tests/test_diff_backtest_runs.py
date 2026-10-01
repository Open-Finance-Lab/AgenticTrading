"""Where do two backtests of one configuration first disagree?

The proof for pinned sampling is a number, not a promise: the first bar at
which two runs of one configuration diverge, how many bars diverge, and how
far the final equity moves. This script reads both off the tables the
dashboard already writes.
"""
import importlib.util
import sys
from pathlib import Path

from dashboard.backend.database import BacktestDatabase

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load_script():
    path = _SCRIPTS_DIR / "diff_backtest_runs.py"
    spec = importlib.util.spec_from_file_location("diff_backtest_runs_script", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(_SCRIPTS_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(_SCRIPTS_DIR))
    return module


def _decision(step, actions):
    return {
        "step_index": step,
        "timestamp": f"2026-09-0{1 + step // 7}T1{step % 7}:00:00",
        "decision_source": "llm",
        "actions_submitted": actions,
        "actions_executed": len(actions),
    }


def _point(step, equity):
    return {
        "timestamp": f"2026-09-0{1 + step // 7}T1{step % 7}:00:00",
        "equity": equity,
        "cash": equity,
        "positions_value": 0.0,
    }


def _seed(db, run_id, decisions, final_equity, equity=()):
    db.insert_run(
        run_id=run_id,
        session_id="diff-session",
        agent_name="diff-agent",
        mode="backtest",
        start_date="2026-09-01",
        end_date="2026-09-08",
        initial_equity=100000.0,
        final_equity=final_equity,
        metadata={"llm_sampling": {"temperature": 0.0, "reasoning_effort": None}},
    )
    if decisions:
        db.insert_decisions(run_id, decisions)
    if equity:
        db.insert_equity_points(
            run_id, [_point(i, value) for i, value in enumerate(equity)]
        )


def test_compare_decisions_finds_the_first_divergent_bar():
    module = _load_script()
    a = [_decision(0, []), _decision(1, [{"symbol": "AAPL", "side": "buy"}]), _decision(2, [])]
    b = [_decision(0, []), _decision(1, []), _decision(2, [{"symbol": "MSFT", "side": "buy"}])]

    report = module.compare_decisions(a, b)

    assert report["steps_compared"] == 3
    assert report["divergent_steps"] == 2
    assert report["first_divergence"] == {"step_index": 1, "timestamp": a[1]["timestamp"]}


def test_compare_decisions_ignores_key_order_inside_an_action():
    module = _load_script()
    a = [_decision(0, [{"symbol": "AAPL", "side": "buy"}])]
    b = [_decision(0, [{"side": "buy", "symbol": "AAPL"}])]
    assert module.compare_decisions(a, b)["divergent_steps"] == 0


def test_compare_equity_finds_the_first_divergent_bar():
    module = _load_script()
    a = [_point(0, 100000.0), _point(1, 100500.0), _point(2, 101000.0)]
    b = [_point(0, 100000.0), _point(1, 100500.0), _point(2, 100900.0)]

    report = module.compare_equity(a, b)

    assert report["equity_points_compared"] == 3
    assert report["divergent_equity_points"] == 1
    assert report["first_equity_divergence"] == {
        "index": 2,
        "timestamp": a[2]["timestamp"],
    }


def test_compare_runs_reads_both_rows(tmp_path, monkeypatch):
    module = _load_script()
    db = BacktestDatabase(tmp_path / "diff.db")
    monkeypatch.setattr(module, "db", db)
    _seed(
        db,
        "run_a",
        [_decision(0, []), _decision(1, [{"symbol": "AAPL", "side": "buy"}])],
        101000.0,
        equity=[100000.0, 101000.0],
    )
    _seed(
        db,
        "run_b",
        [_decision(0, []), _decision(1, [])],
        99990.0,
        equity=[100000.0, 99990.0],
    )

    report = module.compare_runs("run_a", "run_b")

    assert report["run_a"] == "run_a"
    assert report["decisions_recorded"] is True
    assert report["basis"] == "decisions"
    assert report["divergent_steps"] == 1
    assert report["first_divergence"]["step_index"] == 1
    assert report["divergent_equity_points"] == 1
    assert report["final_equity_a"] == 101000.0
    assert report["final_equity_b"] == 99990.0
    assert round(report["final_equity_gap_pct"], 4) == -1.0
    assert report["sampling_a"] == {"temperature": 0.0, "reasoning_effort": None}


def test_an_absent_decision_log_is_unmeasured_not_agreement(tmp_path, monkeypatch):
    """A pipeline-runtime backtest writes no backtest_decisions rows at all.

    `run_agent_backtest` calls `db.insert_decisions` only for the AI Hedge
    Fund runtime (the `insert_decisions` guard in `HourlyBacktester` keyed on
    `AI_HEDGE_FUND_RUNTIME_TYPE`); the other writer in the backend is the
    external-agent surface. Every run Task 8 measures is a pipeline run, so
    both logs come back empty -- and `divergent_steps: 0` out of an empty log
    is this script announcing that two runs agreed on every bar because it
    had no bars to look at. That number would then be copied into the Final
    verification table as the headline result of the whole track.
    """
    module = _load_script()
    db = BacktestDatabase(tmp_path / "diff.db")
    monkeypatch.setattr(module, "db", db)
    _seed(db, "run_a", [], 101000.0, equity=[100000.0, 100500.0, 101000.0])
    _seed(db, "run_b", [], 99990.0, equity=[100000.0, 100500.0, 99990.0])

    report = module.compare_runs("run_a", "run_b")

    assert report["decisions_recorded"] is False
    assert report["basis"] == "equity"
    assert report["steps_compared"] is None
    assert report["divergent_steps"] is None
    assert report["first_divergence"] is None
    assert report["divergent_equity_points"] == 1
    assert report["first_equity_divergence"]["index"] == 2


def test_compare_runs_refuses_an_unknown_run(tmp_path, monkeypatch):
    import pytest

    module = _load_script()
    db = BacktestDatabase(tmp_path / "diff.db")
    monkeypatch.setattr(module, "db", db)
    _seed(db, "run_a", [_decision(0, [])], 100000.0)

    with pytest.raises(SystemExit, match="run_zzz"):
        module.compare_runs("run_a", "run_zzz")
