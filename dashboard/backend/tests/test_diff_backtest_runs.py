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
    external-agent surface. An ordinary pipeline-runtime run therefore has no
    decision rows, so both logs come back empty -- and `divergent_steps: 0`
    out of an empty log is this script announcing that two runs agreed on
    every bar because it had no bars to look at.
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


def test_equity_missing_a_middle_bar_is_not_divergence():
    module = _load_script()
    a = [_point(0, 100000.0), _point(1, 100500.0), _point(2, 101000.0), _point(3, 101500.0)]
    b = [_point(0, 100000.0), _point(2, 101000.0), _point(3, 101500.0)]

    report = module.compare_equity(a, b)

    assert report["equity_points_compared"] == 3
    assert report["equity_only_in_a"] == 1
    assert report["equity_only_in_b"] == 0
    assert report["divergent_equity_points"] == 0
    assert report["first_equity_divergence"] is None


def test_equity_divergence_after_a_missing_bar_names_the_shared_timestamp():
    module = _load_script()
    a = [_point(0, 100000.0), _point(1, 100500.0), _point(2, 101000.0), _point(3, 101500.0)]
    b = [_point(0, 100000.0), _point(2, 101000.0), _point(3, 99000.0)]

    report = module.compare_equity(a, b)

    assert report["divergent_equity_points"] == 1
    assert report["first_equity_divergence"]["timestamp"] == a[3]["timestamp"]


def test_equity_truncated_run_reports_a_tail_not_divergence():
    module = _load_script()
    a = [_point(i, 100000.0 + i) for i in range(5)]
    b = a[:3]  # a timed-out run: same values, fewer bars

    report = module.compare_equity(a, b)

    assert report["equity_points_compared"] == 3
    assert report["equity_only_in_a"] == 2
    assert report["equity_only_in_b"] == 0
    assert report["divergent_equity_points"] == 0
    assert report["first_equity_divergence"] is None


def test_decisions_join_on_step_index_not_position():
    module = _load_script()
    buy = [{"symbol": "AAPL", "side": "buy"}]
    a = [_decision(0, []), _decision(1, []), _decision(2, buy), _decision(3, [])]
    b = [_decision(0, []), _decision(2, buy)]  # step 1 missing, step 3 cut off

    report = module.compare_decisions(a, b)

    assert report["steps_compared"] == 2
    assert report["decisions_only_in_a"] == 2
    assert report["divergent_steps"] == 0
    assert report["first_divergence"] is None


def test_no_shared_bars_is_unmeasured_not_agreement():
    module = _load_script()
    report = module.compare_equity([_point(0, 1.0)], [_point(1, 1.0)])
    assert report["equity_points_compared"] == 0
    assert report["divergent_equity_points"] is None


def _seed_pair(db, **overrides):
    _seed(db, "run_a", [], 101000.0, equity=[100000.0, 101000.0])
    _seed(db, "run_b", [], 101000.0, equity=[100000.0, 101000.0])
    for key, value in overrides.items():
        conn = db._get_connection()
        conn.execute(f"UPDATE agent_runs SET {key} = ? WHERE run_id = 'run_b'", (value,))
        conn.commit()
        conn.close()


def test_matching_runs_are_comparable(tmp_path, monkeypatch):
    module = _load_script()
    db = BacktestDatabase(tmp_path / "diff.db")
    monkeypatch.setattr(module, "db", db)
    _seed_pair(db)

    report = module.compare_runs("run_a", "run_b")

    assert report["comparable"] is True
    assert report["mismatches"] == []
    assert report["start_date_a"] == "2026-09-01"
    assert report["llm_model_b"] == "rule-based"


def test_a_different_window_or_model_is_named_in_mismatches(tmp_path, monkeypatch):
    module = _load_script()
    db = BacktestDatabase(tmp_path / "diff.db")
    monkeypatch.setattr(module, "db", db)
    _seed_pair(db, end_date="2026-09-09", llm_model="other-model")

    report = module.compare_runs("run_a", "run_b")

    assert report["comparable"] is False
    assert report["mismatches"] == ["end_date", "llm_model"]


def test_cli_exits_nonzero_on_mismatch_unless_allowed(tmp_path, monkeypatch, capsys):
    module = _load_script()
    db = BacktestDatabase(tmp_path / "diff.db")
    monkeypatch.setattr(module, "db", db)
    _seed_pair(db, end_date="2026-09-09")

    assert module.main(["run_a", "run_b"]) == 2
    captured = capsys.readouterr()
    assert '"mismatches"' in captured.out
    assert "end_date" in captured.err
    assert f"sqlite ({db.db_path})" in captured.err

    assert module.main(["run_a", "run_b", "--allow-mismatch"]) == 0


def test_cli_names_a_postgres_backend_without_credentials(monkeypatch, capsys):
    module = _load_script()

    class _FakePostgres:
        database_url = "postgresql://user:hunter2@ep-x.neon.tech/runs"

        def get_run(self, run_id):
            return None

    monkeypatch.setattr(module, "db", _FakePostgres())

    try:
        module.main(["a", "b"])
    except SystemExit:
        pass
    err = capsys.readouterr().err
    assert "postgres (ep-x.neon.tech/runs)" in err
    assert "hunter2" not in err


def test_missing_final_equity_gets_a_gap_reason(tmp_path, monkeypatch):
    module = _load_script()
    db = BacktestDatabase(tmp_path / "diff.db")
    monkeypatch.setattr(module, "db", db)
    _seed(db, "run_a", [], None, equity=[100000.0])
    _seed(db, "run_b", [], 99990.0, equity=[100000.0])

    report = module.compare_runs("run_a", "run_b")

    assert report["final_equity_gap_pct"] is None
    assert "run_a" in report["final_equity_gap_reason"]


def test_a_different_universe_is_a_mismatch_and_a_missing_one_is_not(tmp_path, monkeypatch):
    module = _load_script()
    db = BacktestDatabase(tmp_path / "diff.db")
    monkeypatch.setattr(module, "db", db)
    _seed_pair(db)

    def set_symbols(run_id, symbols):
        db.insert_run(
            run_id=run_id, session_id="diff-session", agent_name="diff-agent",
            mode="backtest", start_date="2026-09-01", end_date="2026-09-08",
            initial_equity=100000.0, final_equity=101000.0,
            metadata={"symbols": symbols},
        )

    set_symbols("run_a", ["AAPL", "MSFT"])
    assert module.compare_runs("run_a", "run_b")["mismatches"] == []  # run_b never recorded one

    set_symbols("run_b", ["MSFT", "NVDA"])
    assert module.compare_runs("run_a", "run_b")["mismatches"] == ["symbols"]

    set_symbols("run_b", ["MSFT", "AAPL"])  # order is not a difference
    assert module.compare_runs("run_a", "run_b")["comparable"] is True


def _set_metadata(db, run_id, metadata):
    db.insert_run(
        run_id=run_id, session_id="diff-session", agent_name="diff-agent",
        mode="backtest", start_date="2026-09-01", end_date="2026-09-08",
        initial_equity=100000.0, final_equity=101000.0, metadata=metadata,
    )


def test_a_different_pipeline_is_a_mismatch(tmp_path, monkeypatch):
    """A different strategy prompt or step count diverges for reasons that
    have nothing to do with sampling, and would otherwise be attributed to it."""
    module = _load_script()
    db = BacktestDatabase(tmp_path / "diff.db")
    monkeypatch.setattr(module, "db", db)
    _seed_pair(db)

    _set_metadata(db, "run_a", {"initial_pipeline": [{"presetKey": "momentum"}]})
    _set_metadata(db, "run_b", {"initial_pipeline": [{"presetKey": "value"}]})

    report = module.compare_runs("run_a", "run_b")

    assert report["comparable"] is False
    assert report["mismatches"] == ["initial_pipeline"]


def test_pipeline_key_order_is_not_a_difference(tmp_path, monkeypatch):
    module = _load_script()
    db = BacktestDatabase(tmp_path / "diff.db")
    monkeypatch.setattr(module, "db", db)
    _seed_pair(db)

    _set_metadata(db, "run_a", {"initial_pipeline": [{"presetKey": "m", "prompt": "p"}]})
    _set_metadata(db, "run_b", {"initial_pipeline": [{"prompt": "p", "presetKey": "m"}]})

    assert module.compare_runs("run_a", "run_b")["comparable"] is True


def test_a_different_data_source_is_a_mismatch(tmp_path, monkeypatch):
    module = _load_script()
    db = BacktestDatabase(tmp_path / "diff.db")
    monkeypatch.setattr(module, "db", db)
    _seed_pair(db)

    _set_metadata(db, "run_a", {"data_source": "alpaca"})
    _set_metadata(db, "run_b", {"data_source": "ifind_ashare"})

    assert module.compare_runs("run_a", "run_b")["mismatches"] == ["data_source"]


def test_a_run_that_never_recorded_a_pipeline_or_source_is_not_a_mismatch(
    tmp_path, monkeypatch
):
    module = _load_script()
    db = BacktestDatabase(tmp_path / "diff.db")
    monkeypatch.setattr(module, "db", db)
    _seed_pair(db)

    _set_metadata(
        db, "run_a", {"initial_pipeline": [{"presetKey": "m"}], "data_source": "alpaca"}
    )
    _set_metadata(db, "run_b", {})

    assert module.compare_runs("run_a", "run_b")["mismatches"] == []
