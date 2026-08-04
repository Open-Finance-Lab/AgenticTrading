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


def test_insert_attempt_duplicate_run_id_resets_terminal_fields(tmp_path):
    """A second insert_attempt for the same run_id is SQLite's ``INSERT OR
    REPLACE``, i.e. a DELETE+INSERT: every column the statement's column list
    omits (``error``, ``finished_at``, ``created_at``) reverts to its schema
    default, it does not survive from the row being replaced.

    Pinned here (2026-08-04 fix round) because the Postgres twin's
    ``ON CONFLICT (run_id) DO UPDATE`` must reproduce this exact reset
    column-by-column rather than silently preserving the old terminal state
    -- see database_postgres.py's insert_attempt docstring. A relaunch under
    a reused run_id must read as a fresh 'running' attempt, not one still
    carrying the previous attempt's error/finished_at.
    """
    db = _db(tmp_path)
    _insert(db, "run-relaunch")
    db.finalize_attempt("run-relaunch", "failed", error="first attempt died")

    row = db.get_attempts_for_session("sess-1")[0]
    assert row["status"] == "failed"
    assert row["error"] == "first attempt died"
    assert row["finished_at"]

    # Relaunch under the same run_id (e.g. a client retry of the launch call).
    _insert(db, "run-relaunch", agent_name="Retried Agent")

    row = db.get_attempts_for_session("sess-1")[0]
    assert row["status"] == "running"
    assert row["error"] is None
    assert row["finished_at"] is None
    assert row["agent_name"] == "Retried Agent"
    # created_at is also reset by REPLACE (schema DEFAULT CURRENT_TIMESTAMP);
    # not asserted against the pre-relaunch value since CURRENT_TIMESTAMP's
    # 1-second resolution would make an equality/inequality check flaky --
    # its presence after the relaunch is enough to prove it wasn't dropped.
    assert row["created_at"]


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
