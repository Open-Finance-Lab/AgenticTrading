"""End-to-end cover for the three data-driven ALTER loops in ``database.py``.

A ``BacktestDatabase(tmp_path / "x.db")`` gets every column straight from
``CREATE TABLE``, so it never executes a single ``ALTER``. That leaves the
lazy-migration path -- the one that runs against the *deployed* database on
every boot -- covered only in narrow slices: ``test_agent_runs_metadata.py``
checks ``agent_runs.metadata`` alone, and ``test_currency_audit_database.py``
checks the native-currency fields on an otherwise-current schema. Neither one
would notice a loop that dropped a column, changed a type, or stopped running.

This file opens a deliberately pre-migration database and asserts the full
column set each of the three loops owns:

* ``_migrate_schema``'s ``token_columns`` loop  -> ``agent_runs``
* ``_migrate_trades_schema``'s ``additions`` loop -> ``trades``
* ``_migrate_currency_audit_schema``            -> ``equity_timeseries``, ``trades``
"""

import sqlite3

from dashboard.backend.database import BacktestDatabase

# Columns each loop is responsible for adding to an existing table.
_TOKEN_COLUMNS = {"llm_calls", "input_tokens", "output_tokens", "est_cost_usd", "metadata"}
_TRADE_COLUMNS = {"quantity", "side", "value", "reason"}
_CURRENCY_EQUITY_COLUMNS = {
    "native_equity",
    "native_cash",
    "native_positions_value",
    "fx_rate",
}
_CURRENCY_TRADE_COLUMNS = {"native_price", "native_value", "fx_rate"}


def _build_pre_migration_database(path):
    """Write the old table shapes with raw sqlite3.

    Deliberately not through ``BacktestDatabase``, which would migrate them
    for us and leave the ALTERs unexercised. ``agent_runs`` keeps
    ``session_id``/``llm_model`` so only the ``token_columns`` loop has work
    to do; ``trades`` is the legacy shares/action/total_value shape.
    """
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE agent_runs (
            run_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            mode TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            initial_equity REAL NOT NULL,
            final_equity REAL,
            total_return REAL,
            sharpe_ratio REAL,
            max_drawdown REAL,
            num_trades INTEGER DEFAULT 0,
            llm_model TEXT DEFAULT 'rule-based',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE equity_timeseries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            equity REAL NOT NULL,
            cash REAL NOT NULL,
            positions_value REAL NOT NULL,
            daily_return REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(run_id, timestamp)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            shares INTEGER NOT NULL,
            action TEXT NOT NULL,
            price REAL NOT NULL,
            total_value REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    # A row so the ALTERs run against a non-empty table, as they do on prod,
    # and the legacy->new backfill has something to copy.
    conn.execute(
        """
        INSERT INTO trades (run_id, timestamp, symbol, shares, action, price, total_value)
        VALUES ('legacy-run', '2026-04-01T10:30:00', 'AAPL', 3, 'buy', 100.0, 300.0)
        """
    )
    conn.commit()
    conn.close()


def _columns(path, table) -> set[str]:
    conn = sqlite3.connect(str(path))
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()


def test_opening_a_pre_migration_database_adds_every_lazily_migrated_column(tmp_path):
    path = tmp_path / "pre-migration.db"
    _build_pre_migration_database(path)

    # Precondition: the columns really are absent, so a green assertion below
    # cannot be the CREATE TABLE path quietly supplying them.
    assert not (_TOKEN_COLUMNS & _columns(path, "agent_runs"))
    assert not (_CURRENCY_EQUITY_COLUMNS & _columns(path, "equity_timeseries"))
    assert not ((_TRADE_COLUMNS | _CURRENCY_TRADE_COLUMNS) & _columns(path, "trades"))

    BacktestDatabase(path)

    assert _TOKEN_COLUMNS <= _columns(path, "agent_runs")
    assert _CURRENCY_EQUITY_COLUMNS <= _columns(path, "equity_timeseries")
    assert (_TRADE_COLUMNS | _CURRENCY_TRADE_COLUMNS) <= _columns(path, "trades")


def test_legacy_trade_columns_are_backfilled_into_the_new_ones(tmp_path):
    path = tmp_path / "backfill.db"
    _build_pre_migration_database(path)

    BacktestDatabase(path)

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM trades WHERE run_id = 'legacy-run'").fetchone()
    finally:
        conn.close()
    assert row["quantity"] == 3
    assert row["side"] == "BUY"
    assert row["value"] == 300.0


def test_lazy_migrations_are_idempotent_across_reopens(tmp_path):
    path = tmp_path / "reopen.db"
    _build_pre_migration_database(path)

    BacktestDatabase(path)
    after_first = {
        table: _columns(path, table)
        for table in ("agent_runs", "equity_timeseries", "trades")
    }

    BacktestDatabase(path)
    after_second = {
        table: _columns(path, table)
        for table in ("agent_runs", "equity_timeseries", "trades")
    }

    assert after_second == after_first
