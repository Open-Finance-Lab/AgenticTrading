"""backtest_attempts journal — store-level lifecycle (SQLite twin).

The journal is the only durable record of a failed backtest: agent_runs rows
are written only after a run completes (spec Finding 1), so these rows must
exist from launch and survive an error path that never reaches insert_run.
"""

import uuid

from dashboard.backend.database import BacktestDatabase


def _db(tmp_path):
    return BacktestDatabase(db_path=tmp_path / "attempts.db")


def _insert(db, run_id, session_id="sess-1", agent_id="agent-1", **kw):
    db.insert_attempt(
        run_id,
        session_id,
        agent_id=agent_id,
        agent_name=kw.get("agent_name", "My Agent"),
        start_date=kw.get("start_date", "2026-05-01"),
        end_date=kw.get("end_date", "2026-05-07"),
        params=kw.get("params", {"data_source": "alpaca", "initial_capital": 10000}),
        timeout_seconds=kw.get("timeout_seconds", 1800),
    )


def test_insert_then_finalize_failed_round_trips(tmp_path):
    db = _db(tmp_path)
    run_id = f"agent_test_{uuid.uuid4().hex[:8]}"
    _insert(db, run_id)

    rows = db.get_attempts_for_session("sess-1")
    assert [r["run_id"] for r in rows] == [run_id]
    row = rows[0]
    assert row["status"] == "running"
    assert row["error"] is None
    assert row["timeout_seconds"] == 1800
    assert row["created_at"]
    assert row["finished_at"] is None

    db.finalize_attempt(run_id, "failed", error="Backtest failed with return code 1. quota exceeded")
    row = db.get_attempts_for_session("sess-1")[0]
    assert row["status"] == "failed"
    assert "quota exceeded" in row["error"]
    assert row["finished_at"]


def test_finalize_completed_clears_error(tmp_path):
    db = _db(tmp_path)
    _insert(db, "run-ok")
    db.finalize_attempt("run-ok", "completed")
    row = db.get_attempts_for_session("sess-1")[0]
    assert row["status"] == "completed"
    assert row["error"] is None


def test_finalize_without_prior_insert_upserts_terminal_row(tmp_path):
    """The failure record must survive even when the launch-time insert failed."""
    db = _db(tmp_path)
    db.finalize_attempt("run-orphan", "failed", error="boom", session_id="sess-2")
    rows = db.get_attempts_for_session("sess-2")
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["error"] == "boom"
    assert rows[0]["created_at"] and rows[0]["finished_at"]


def test_finalize_without_insert_and_without_session_is_a_noop(tmp_path):
    db = _db(tmp_path)
    db.finalize_attempt("run-unknown", "failed", error="boom")  # must not raise
    assert db.get_attempts_for_session("sess-1") == []


def test_error_capped_at_500_chars(tmp_path):
    db = _db(tmp_path)
    _insert(db, "run-long")
    db.finalize_attempt("run-long", "failed", error="x" * 2000)
    assert len(db.get_attempts_for_session("sess-1")[0]["error"]) == 500


def test_get_attempts_for_session_orders_newest_first_and_limits(tmp_path):
    db = _db(tmp_path)
    for i in range(5):
        _insert(db, f"run-{i}")
    rows = db.get_attempts_for_session("sess-1", limit=3)
    assert len(rows) == 3
    created = [r["created_at"] for r in rows]
    assert created == sorted(created, reverse=True)


def test_get_latest_attempt_for_agents_is_batched_by_agent(tmp_path):
    db = _db(tmp_path)
    _insert(db, "run-a1-old", agent_id="agent-a")
    _insert(db, "run-a1-new", agent_id="agent-a")
    _insert(db, "run-b1", agent_id="agent-b", session_id="sess-b")
    db.finalize_attempt("run-a1-new", "failed", error="boom")

    latest = db.get_latest_attempt_for_agents(["agent-a", "agent-b", "agent-missing"])
    assert set(latest) == {"agent-a", "agent-b"}
    # CURRENT_TIMESTAMP has 1-second resolution, so the two agent-a rows tie on
    # created_at; the store must break ties by insertion order (rowid DESC) for
    # this to hold deterministically.
    assert latest["agent-a"]["run_id"] == "run-a1-new"
    assert latest["agent-b"]["run_id"] == "run-b1"
    assert db.get_latest_attempt_for_agents([]) == {}
