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
