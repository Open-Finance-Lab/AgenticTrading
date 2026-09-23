"""Operator-funded model cost per owner comes from the run-history store."""

from __future__ import annotations

from datetime import date

from dashboard.backend.database import BacktestDatabase


def _seed(db, run_id, *, owner, cost, day):
    db.insert_run(
        run_id=run_id,
        session_id="session-1",
        agent_name="Agent",
        mode="backtest",
        start_date="2026-09-01",
        end_date="2026-09-02",
        initial_equity=100000.0,
        est_cost_usd=cost,
        owner_user_id=owner,
    )
    conn = db._get_connection()
    try:
        # created_at/updated_at are CURRENT_TIMESTAMP text ("YYYY-MM-DD HH:MM:SS");
        # pin the row onto the day under test.
        conn.execute(
            "UPDATE agent_runs SET updated_at = ? WHERE run_id = ?",
            (f"{day} 15:30:00", run_id),
        )
        conn.commit()
    finally:
        conn.close()


def test_operator_cost_is_grouped_by_owner_for_one_day(tmp_path):
    db = BacktestDatabase(tmp_path / "runs.db")
    _seed(db, "a", owner=7, cost=1.25, day="2026-09-11")
    _seed(db, "b", owner=7, cost=0.5, day="2026-09-11")
    _seed(db, "c", owner=9, cost=0.1, day="2026-09-11")
    _seed(db, "d", owner=7, cost=3.0, day="2026-09-10")  # another day
    _seed(db, "e", owner=None, cost=2.0, day="2026-09-11")  # unattributed

    totals = db.aggregate_operator_cost_for_day(date(2026, 9, 11))

    assert totals == {7: 1_750_000, 9: 100_000}
    assert db.aggregate_operator_cost_for_day(date(2026, 9, 12)) == {}


def test_a_negative_stored_cost_cannot_violate_the_fact_check(tmp_path):
    db = BacktestDatabase(tmp_path / "runs.db")
    _seed(db, "neg", owner=3, cost=-0.75, day="2026-09-11")

    assert db.aggregate_operator_cost_for_day(date(2026, 9, 11)) == {3: 0}
