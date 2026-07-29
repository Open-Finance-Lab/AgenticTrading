"""Cold-half schema and delete semantics for the SQLite BacktestDatabase."""

from dashboard.backend.database import BacktestDatabase


def _insert_run(db: BacktestDatabase, run_id: str) -> None:
    db.insert_run(
        run_id=run_id, session_id="cold-half", agent_name="Agent", mode="backtest",
        start_date="2026-01-01", end_date="2026-01-02", initial_equity=1_000,
    )


def test_backtest_decisions_has_actions_trace_ref(tmp_path):
    db = BacktestDatabase(tmp_path / "cold.db")
    conn = db._get_connection()
    cols = {row[1] for row in conn.execute("PRAGMA table_info(backtest_decisions)")}
    assert "actions_trace_ref" in cols


def test_delete_run_removes_the_manifest(tmp_path):
    db = BacktestDatabase(tmp_path / "cold.db")
    _insert_run(db, "r1")
    db.insert_run_manifest("r1", {"any": "thing"})
    db.delete_run("r1")
    assert db.get_run_manifest("r1") is None


def test_clear_all_removes_manifests(tmp_path):
    db = BacktestDatabase(tmp_path / "cold.db")
    _insert_run(db, "r1")
    db.insert_run_manifest("r1", {"any": "thing"})
    db.clear_all()
    assert db.get_run_manifest("r1") is None
