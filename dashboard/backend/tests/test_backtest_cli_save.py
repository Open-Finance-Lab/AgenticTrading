"""Tests for the backtest CLI save path (dashboard/scripts/backtest.py).

Regression for the "insert_run missing session_id" TypeError: the CLI's
save_backtest_to_database() called db.insert_run() without the required
session_id positional arg, so every `python dashboard/scripts/backtest.py`
save raised. Each standalone CLI run has no session concept, so it now passes
session_id=run_id — the same pattern the /paper/start-session fix uses.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load_backtest_script():
    """Import dashboard/scripts/backtest.py (not a package) in isolation."""
    path = _SCRIPTS_DIR / "backtest.py"
    spec = importlib.util.spec_from_file_location("backtest_script", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # The script imports sibling modules (backtest_engine, _bootstrap) by bare
    # name, so the scripts dir must be on sys.path for those to resolve.
    sys.path.insert(0, str(_SCRIPTS_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(_SCRIPTS_DIR))
    return module


def test_save_backtest_to_database_passes_session_id(monkeypatch):
    """insert_run must receive session_id (= run_id) so the CLI save succeeds."""
    module = _load_backtest_script()

    calls = {}

    class FakeDb:
        def insert_run(self, **kwargs):
            calls["insert_run"] = kwargs

        def insert_equity_points(self, run_id, curve):
            calls["insert_equity_points"] = (run_id, curve)

    monkeypatch.setattr(module, "db", FakeDb())

    results = {
        "agent_a": {
            "metrics": {
                "total_return": 0.05,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.03,
                "num_trades": 5,
            },
            "equity_curve": [{"timestamp": "t", "equity": 100000}],
        },
    }

    module.save_backtest_to_database(results, "2026-01-01", "2026-02-01")

    kwargs = calls["insert_run"]
    assert "session_id" in kwargs, "insert_run missing session_id"
    assert kwargs["session_id"] == kwargs["run_id"]
    assert kwargs["agent_name"] == "agent_a"
    assert calls["insert_equity_points"][0] == kwargs["run_id"]
