"""Cold-half schema and delete semantics for the SQLite BacktestDatabase."""

import datetime
import warnings

from dashboard.backend.database import BacktestDatabase, as_timestamp_text


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


# ---------------------------------------------------------------------------
# Timestamp normalisation (as_timestamp_text)
#
# The Postgres half of this behaviour needs a live server
# (test_backtest_db_postgres.py, @pg_only). These cases pin the SQLite half and
# the helper itself, which is what makes the two stores land the same text --
# so a regression here is catchable without Postgres.
# ---------------------------------------------------------------------------

def test_as_timestamp_text_uses_the_t_separator_not_sqlites_adapter():
    """Not merely "some ISO string": the *separator* is the whole point.

    ``sqlite3``'s deprecated default datetime adapter -- the code path these
    writers used to fall through to -- stores ``isoformat(" ")``, with a space.
    ``insert_trades`` on both stores, and external_run_service's decision log,
    have always written ``isoformat()`` with a ``T``. Pinning ``T`` here is
    what keeps one format across every timestamp column; asserting only
    "startswith 2026-01-02" would pass for either and let them drift apart.
    """
    assert as_timestamp_text(datetime.datetime(2026, 1, 2, 9, 30, 15)) == "2026-01-02T09:30:15"
    assert as_timestamp_text(datetime.date(2026, 1, 2)) == "2026-01-02"


def test_as_timestamp_text_passes_through_strings_and_none():
    """``None`` must survive as ``None``, not become the string "None".

    Both ``equity_timeseries.timestamp`` and ``backtest_decisions.timestamp``
    are NOT NULL, so a caller that forgets the field should hit an integrity
    error. ``insert_trades`` wraps its already-converted value in ``str()`` and
    therefore stores "None" -- that quirk is matched on both stores and is not
    this helper's job to spread further.
    """
    assert as_timestamp_text("2026-01-02T09:30:15") == "2026-01-02T09:30:15"
    assert as_timestamp_text(None) is None


def test_equity_and_decision_writers_accept_datetime_without_the_deprecated_adapter(tmp_path):
    """A ``datetime``-passing caller is stored as ``T``-separated text, and the
    deprecated ``sqlite3`` adapter is never reached.

    ``filterwarnings("error")`` is the real assertion: before the conversion,
    these three writers handed the raw ``datetime`` to ``sqlite3`` and the
    default adapter did the work while emitting a DeprecationWarning. Turning
    that warning into an error proves the value is converted *before* it
    reaches the driver, so this keeps working when Python removes the adapter.
    """
    db = BacktestDatabase(tmp_path / "cold.db")
    _insert_run(db, "r1")
    ts = datetime.datetime(2026, 1, 2, 9, 30, 15)

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=DeprecationWarning)
        db.insert_equity_point("r1", ts, equity=1_000.0, cash=1_000.0, positions_value=0.0)
        db.insert_equity_points(
            "r2-append",
            [{"timestamp": ts, "equity": 1.0, "cash": 1.0, "positions_value": 0.0}],
        )
        db.insert_decisions(
            "r1", [{"step_index": 0, "timestamp": ts, "decision_source": "llm"}]
        )

    assert [p["timestamp"] for p in db.get_equity_curve("r1")] == ["2026-01-02T09:30:15"]
    assert [d["timestamp"] for d in db.get_decisions("r1")] == ["2026-01-02T09:30:15"]
