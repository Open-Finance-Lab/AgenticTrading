"""Tests for the backtest CLI save path (dashboard/scripts/backtest.py).

Regression for the "insert_run missing session_id" TypeError: the CLI's
save_backtest_to_database() called db.insert_run() without the required
session_id positional arg, so every `python dashboard/scripts/backtest.py`
save raised.

The stub deliberately binds each call against the *real*
``BacktestDatabase.insert_run`` signature instead of swallowing ``**kwargs``.
A permissive stub cannot fail the way production fails: it would keep passing
if ``insert_run`` grew another required parameter, which is precisely the
defect class this file exists to catch.
"""

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

from dashboard.backend.database import BacktestDatabase

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"

_INSERT_RUN_SIGNATURE = inspect.signature(BacktestDatabase.insert_run)


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


class _ContractCheckingDb:
    """Records insert_run calls, rejecting any that the real DB would reject."""

    def __init__(self):
        self.calls = {}

    def insert_run(self, *args, **kwargs):
        # Raises TypeError for a missing/unknown parameter, exactly as the real
        # BacktestDatabase.insert_run would. `self` stands in for the bound
        # instance the real (unbound) signature still expects.
        bound = _INSERT_RUN_SIGNATURE.bind(self, *args, **kwargs)
        bound.apply_defaults()
        self.calls["insert_run"] = dict(bound.arguments)

    def insert_equity_points(self, run_id, curve):
        self.calls["insert_equity_points"] = (run_id, curve)


def test_save_backtest_to_database_satisfies_insert_run_contract(monkeypatch):
    """The CLI's insert_run call must satisfy the real DB signature."""
    module = _load_backtest_script()
    db = _ContractCheckingDb()
    monkeypatch.setattr(module, "db", db)

    module.save_backtest_to_database(
        {
            "agent_a": {
                # BacktestMetrics reports percent (see the scale test below).
                "metrics": {
                    "total_return": 10.0,
                    "sharpe_ratio": 1.2,
                    "max_drawdown": -3.0,
                    "num_trades": 5,
                },
                "equity_curve": [{"timestamp": "t", "equity": 110000}],
            },
        },
        "2026-01-01",
        "2026-02-01",
    )

    kwargs = db.calls["insert_run"]
    assert kwargs["session_id"], "insert_run missing session_id"
    assert kwargs["agent_name"] == "agent_a"
    assert kwargs["mode"] == "backtest"
    assert db.calls["insert_equity_points"][0] == kwargs["run_id"]


def test_save_backtest_to_database_uses_the_shared_sessionless_key(monkeypatch):
    """Sessionless writers share one grouping key; a per-run id orphans the row.

    ``agent_runs.session_id`` is an ownership/grouping key — every
    session-scoped query (``get_runs_by_session``, ``get_runs_by_sessions``,
    ``get_run_with_session``) and every stale-run sweep reaches rows *through*
    it. Minting a fresh session id per CLI invocation makes each run its own
    unreachable island. ``legacy-demo-session`` is what BacktestEngine defaults
    to, what backtest_hourly_agent's --session-id defaults to, and what
    database.py backfills onto pre-session rows.
    """
    module = _load_backtest_script()
    db = _ContractCheckingDb()
    monkeypatch.setattr(module, "db", db)

    module.save_backtest_to_database(
        {
            "agent_a": {
                "metrics": {
                    "total_return": 10.0,
                    "sharpe_ratio": 1.2,
                    "max_drawdown": -3.0,
                    "num_trades": 5,
                },
                "equity_curve": [{"timestamp": "t", "equity": 110000}],
            },
        },
        "2026-01-01",
        "2026-02-01",
    )

    kwargs = db.calls["insert_run"]
    assert kwargs["session_id"] == "legacy-demo-session"
    assert kwargs["session_id"] != kwargs["run_id"]


def test_backtest_metrics_are_reported_in_percent():
    """Pins the producer side of the scale contract.

    ``BacktestMetrics.total_return`` / ``max_drawdown`` are percentages
    (backtest_engine multiplies by 100). If that ever changes to fractions, the
    CLI's /100 conversion becomes a double-divide — this test reds first so the
    two halves can't drift apart silently.
    """
    sys.path.insert(0, str(_SCRIPTS_DIR))
    try:
        from backtest_engine import BacktestEngine
    finally:
        sys.path.remove(str(_SCRIPTS_DIR))

    engine = BacktestEngine(initial_capital=100000)
    engine.register_agent("agent_a")
    portfolio = engine.portfolios["agent_a"]
    portfolio.record_daily_equity("2026-01-01", {})
    portfolio.cash = 110000  # all-cash +10%
    portfolio.record_daily_equity("2026-01-02", {})

    metrics = engine.calculate_metrics("agent_a")
    assert metrics.total_return == pytest.approx(10.0)  # percent, not 0.10


def test_save_backtest_to_database_stores_fractions_not_percent(monkeypatch):
    """agent_runs stores fractions; the CLI must convert down from percent.

    Every other insert_run writer stores ``(final - initial) / initial``, and
    the frontend disambiguates by magnitude (``Math.abs(x) <= 1 ? x * 100 : x``
    in app.js). Storing percent would make a genuine +0.4% run render as
    "+40.00%" on the public, unfiltered /runs list.
    """
    module = _load_backtest_script()
    db = _ContractCheckingDb()
    monkeypatch.setattr(module, "db", db)

    module.save_backtest_to_database(
        {
            "agent_a": {
                "metrics": {
                    "total_return": 0.4,  # +0.4 percent
                    "sharpe_ratio": 1.2,
                    "max_drawdown": -12.5,  # -12.5 percent
                    "num_trades": 5,
                },
                "equity_curve": [{"timestamp": "t", "equity": 100400}],
            },
        },
        "2026-01-01",
        "2026-02-01",
    )

    kwargs = db.calls["insert_run"]
    assert kwargs["total_return"] == pytest.approx(0.004)
    assert kwargs["max_drawdown"] == pytest.approx(-0.125)
    # The magnitude heuristic must read these as fractions, not as percentages.
    assert abs(kwargs["total_return"]) <= 1
    assert abs(kwargs["max_drawdown"]) <= 1
