"""Regression tests for legacy equity-timeseries uniqueness migration."""

import sqlite3

import pytest

from dashboard.backend.database import BacktestDatabase


RUN_ID = "legacy-equity-run"
TIMESTAMP = "2026-04-15T14:00:00"


def _create_legacy_database(path, *, conflicting_index=False):
    """Create the pre-constraint table shape and two versions of one point."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE equity_timeseries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            equity REAL NOT NULL,
            cash REAL,
            positions_value REAL,
            daily_return REAL,
            FOREIGN KEY (run_id) REFERENCES agent_runs(run_id)
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO equity_timeseries
            (run_id, timestamp, equity, cash, positions_value, daily_return)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (RUN_ID, TIMESTAMP, 100_000.0, 50_000.0, 50_000.0, 0.0),
            (RUN_ID, TIMESTAMP, 10_123.0, 4_000.0, 6_123.0, 0.01),
        ],
    )
    if conflicting_index:
        conn.execute(
            """
            CREATE INDEX uq_equity_timeseries_run_timestamp
            ON equity_timeseries(run_id, timestamp)
            """
        )
    conn.commit()
    conn.close()


def _raw_points(path):
    conn = sqlite3.connect(str(path))
    rows = conn.execute(
        """
        SELECT id, equity, cash, positions_value, daily_return
        FROM equity_timeseries
        WHERE run_id = ? AND timestamp = ?
        ORDER BY id
        """,
        (RUN_ID, TIMESTAMP),
    ).fetchall()
    conn.close()
    return rows


def test_legacy_equity_duplicates_are_deduplicated_and_protected(tmp_path):
    path = tmp_path / "legacy-equity.db"
    _create_legacy_database(path)

    db = BacktestDatabase(path)

    assert _raw_points(path) == [(2, 10_123.0, 4_000.0, 6_123.0, 0.01)]

    conn = sqlite3.connect(str(path))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO equity_timeseries
                (run_id, timestamp, equity, cash, positions_value, daily_return)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (RUN_ID, TIMESTAMP, 9_999.0, 9_999.0, 0.0, -0.01),
        )
    conn.rollback()
    conn.close()

    db.insert_equity_point(
        RUN_ID,
        TIMESTAMP,
        equity=10_456.0,
        cash=4_100.0,
        positions_value=6_356.0,
        daily_return=0.02,
    )
    assert len(_raw_points(path)) == 1
    assert db.get_equity_curve(RUN_ID) == [
        {
            "timestamp": TIMESTAMP,
            "equity": 10_456.0,
            "cash": 4_100.0,
            "positions_value": 6_356.0,
            "daily_return": 0.02,
        }
    ]

    BacktestDatabase(path)
    assert len(_raw_points(path)) == 1


def test_failed_uniqueness_migration_rolls_back_and_stops_startup(tmp_path):
    path = tmp_path / "conflicting-index.db"
    _create_legacy_database(path, conflicting_index=True)

    with pytest.raises(RuntimeError, match="equity_timeseries.*unique"):
        BacktestDatabase(path)

    assert len(_raw_points(path)) == 2


def test_transient_lock_is_retried_then_deferred(tmp_path, monkeypatch, capsys):
    """A busy database must not take the app down.

    ``database.py`` ends with a module-level ``db = BacktestDatabase()``, so a
    hard failure here aborts *import* of the module. Lock contention is
    transient (a backtest CLI or the Discord bot holding the write lock), and
    the migrations above it degrade on exactly the same condition.
    """
    path = tmp_path / "locked-equity.db"
    _create_legacy_database(path)
    attempts = []

    def _locked(self, conn, cursor):
        attempts.append(1)
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(
        BacktestDatabase, "_apply_equity_timeseries_uniqueness", _locked
    )

    BacktestDatabase(path)  # must not raise

    assert len(attempts) == 2, "a locked write should be retried once"
    assert "equity_timeseries uniqueness migration deferred" in capsys.readouterr().out
    # Deferred, not half-applied: the duplicates are still there for next boot.
    assert len(_raw_points(path)) == 2


def test_data_error_still_fails_startup(tmp_path, monkeypatch):
    """Only *transient* errors degrade — a genuine data problem still aborts."""
    path = tmp_path / "integrity-equity.db"
    _create_legacy_database(path)

    def _integrity(self, conn, cursor):
        raise sqlite3.IntegrityError("UNIQUE constraint failed")

    monkeypatch.setattr(
        BacktestDatabase, "_apply_equity_timeseries_uniqueness", _integrity
    )

    with pytest.raises(RuntimeError, match="equity_timeseries.*unique"):
        BacktestDatabase(path)


def _point(timestamp, equity):
    return {
        "timestamp": timestamp,
        "equity": equity,
        "cash": equity / 2,
        "positions_value": equity / 2,
        "daily_return": 0.0,
    }


def test_rerun_replaces_curve_and_drops_stale_points(tmp_path):
    """The unique key collapses *repeated* timestamps; it cannot remove points
    the new curve no longer produces. A rerun must replace, not merge."""
    db = BacktestDatabase(tmp_path / "rerun.db")
    db.insert_equity_points(
        RUN_ID,
        [
            _point("2026-04-15T14:00:00", 100.0),
            _point("2026-04-15T15:00:00", 101.0),
            _point("2026-04-15T16:00:00", 102.0),
        ],
    )

    db.insert_equity_points(
        RUN_ID,
        [
            _point("2026-04-15T14:00:00", 200.0),
            _point("2026-04-15T15:00:00", 201.0),
        ],
    )

    curve = db.get_equity_curve(RUN_ID)
    assert [p["timestamp"] for p in curve] == [
        "2026-04-15T14:00:00",
        "2026-04-15T15:00:00",
    ]
    assert [p["equity"] for p in curve] == [200.0, 201.0]


def test_rerun_leaves_other_runs_untouched(tmp_path):
    db = BacktestDatabase(tmp_path / "rerun-scope.db")
    db.insert_equity_points("other-run", [_point("2026-04-15T14:00:00", 500.0)])
    db.insert_equity_points(RUN_ID, [_point("2026-04-15T14:00:00", 100.0)])

    db.insert_equity_points(RUN_ID, [_point("2026-04-15T15:00:00", 300.0)])

    assert len(db.get_equity_curve("other-run")) == 1
    assert [p["equity"] for p in db.get_equity_curve(RUN_ID)] == [300.0]


def test_empty_curve_is_a_noop_not_a_wipe(tmp_path):
    db = BacktestDatabase(tmp_path / "empty-curve.db")
    db.insert_equity_points(RUN_ID, [_point("2026-04-15T14:00:00", 100.0)])

    db.insert_equity_points(RUN_ID, [])

    assert len(db.get_equity_curve(RUN_ID)) == 1


def test_replace_false_appends_to_an_existing_curve(tmp_path):
    db = BacktestDatabase(tmp_path / "append.db")
    db.insert_equity_points(RUN_ID, [_point("2026-04-15T14:00:00", 100.0)])

    db.insert_equity_points(
        RUN_ID, [_point("2026-04-15T15:00:00", 101.0)], replace=False
    )

    assert [p["equity"] for p in db.get_equity_curve(RUN_ID)] == [100.0, 101.0]
