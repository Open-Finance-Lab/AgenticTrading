"""agent_runs rows carry the authenticated caller who started them.

Operator-funded model cost is in ``est_cost_usd`` on this row and nowhere
else, so without an owner column there is no per-user cost at all.
"""

from __future__ import annotations

import inspect

from dashboard.backend.database import BacktestDatabase


def test_insert_run_accepts_an_owner(tmp_path):
    db = BacktestDatabase(tmp_path / "runs.db")
    db.insert_run(
        run_id="run-owned",
        session_id="session-1",
        agent_name="Agent",
        mode="backtest",
        start_date="2026-09-01",
        end_date="2026-09-02",
        initial_equity=100000.0,
        est_cost_usd=1.25,
        owner_user_id=7,
    )

    conn = db._get_connection()
    try:
        row = conn.execute(
            "SELECT owner_user_id, est_cost_usd FROM agent_runs WHERE run_id = ?",
            ("run-owned",),
        ).fetchone()
    finally:
        conn.close()

    assert row["owner_user_id"] == 7
    assert row["est_cost_usd"] == 1.25


def test_owner_is_optional_and_defaults_to_null(tmp_path):
    """A scheduled leaderboard deploy has no caller and must still insert."""
    db = BacktestDatabase(tmp_path / "runs.db")
    db.insert_run(
        run_id="run-unowned",
        session_id="session-1",
        agent_name="Agent",
        mode="backtest",
        start_date="2026-09-01",
        end_date="2026-09-02",
        initial_equity=100000.0,
    )

    conn = db._get_connection()
    try:
        row = conn.execute(
            "SELECT owner_user_id FROM agent_runs WHERE run_id = ?",
            ("run-unowned",),
        ).fetchone()
    finally:
        conn.close()

    assert row["owner_user_id"] is None


def test_the_owner_reaches_the_subprocess_and_the_engine():
    """The four hops from the route to the row, pinned by source shape."""
    from dashboard.backend.api.routers import backtests
    from dashboard.backend.domain.backtesting import engine

    assert "owner_user_id" in inspect.signature(
        backtests.run_backtest_background
    ).parameters
    assert "--owner-user-id" in inspect.getsource(backtests.run_backtest_background)
    assert "owner_user_id" in inspect.signature(
        engine.HourlyBacktester.__init__
    ).parameters
