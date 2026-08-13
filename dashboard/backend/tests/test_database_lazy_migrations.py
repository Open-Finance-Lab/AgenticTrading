"""End-to-end cover for the three data-driven ALTER loops in ``database.py``.

A ``BacktestDatabase(tmp_path / "x.db")`` gets every column straight from
``CREATE TABLE``, so it never executes a single ``ALTER``. That leaves the
lazy-migration path -- the one that runs against the *deployed* database on
every boot -- covered only in narrow slices: ``test_agent_runs_metadata.py``
checks ``agent_runs.metadata`` alone, and ``test_currency_audit_database.py``
checks the native-currency fields on an otherwise-current schema. Neither one
would notice a loop that dropped a column, changed a type, or stopped running.

This file opens a deliberately pre-migration database and asserts the full
column set each loop owns:

* ``_migrate_schema``'s ``token_columns`` loop  -> ``agent_runs``
* ``_migrate_schema``'s baseline-link ALTERs    -> ``agent_runs``
* ``_migrate_trades_schema``'s ``additions`` loop -> ``trades``
* ``_migrate_currency_audit_schema``            -> ``equity_timeseries``, ``trades``

Declared **type and default, not just name.** ``PRAGMA table_info`` hands back
``(cid, name, type, notnull, dflt_value, pk)``; an earlier version of this file
kept only ``name`` and discarded the rest, which made it strictly weaker than
the behaviour-preservation proof it was cited as. Name-only comparison cannot
see a column that comes back with the right name and the wrong type -- and
``notnull`` least of all, even though SQLite *rejects* ``ADD COLUMN ... NOT
NULL`` without a default, so a migration that flipped nullability would abort
the boot of the deployed database rather than degrade quietly.
"""

import sqlite3

from dashboard.backend.database import BacktestDatabase

# Columns each loop is responsible for adding to an existing table, as
# ``(declared type, notnull flag, default)`` exactly as PRAGMA table_info
# reports them -- note defaults come back as SQL *text*, so 0 is "'0'" and a
# string default keeps its quotes.
_TOKEN_COLUMNS = {
    "llm_calls": ("INTEGER", 0, "0"),
    "input_tokens": ("INTEGER", 0, "0"),
    "output_tokens": ("INTEGER", 0, "0"),
    "est_cost_usd": ("REAL", 0, "0"),
    "metadata": ("TEXT", 0, None),
}
_BASELINE_COLUMNS = {
    "baseline_djia_run_id": ("TEXT", 0, None),
    "baseline_buyhold_run_id": ("TEXT", 0, None),
}
_TRADE_COLUMNS = {
    "quantity": ("INTEGER", 0, None),
    "side": ("TEXT", 0, None),
    "value": ("REAL", 0, None),
    "reason": ("TEXT", 0, None),
}
_CURRENCY_EQUITY_COLUMNS = {
    "native_equity": ("REAL", 0, None),
    "native_cash": ("REAL", 0, None),
    "native_positions_value": ("REAL", 0, None),
    "fx_rate": ("REAL", 0, None),
}
_CURRENCY_TRADE_COLUMNS = {
    "reference_price": ("REAL", 0, None),
    "gross_value": ("REAL", 0, None),
    "slippage_amount": ("REAL", 0, None),
    "commission": ("REAL", 0, None),
    "stamp_duty": ("REAL", 0, None),
    "transfer_fee": ("REAL", 0, None),
    "total_fees": ("REAL", 0, None),
    "net_cash_impact": ("REAL", 0, None),
    "native_price": ("REAL", 0, None),
    "native_value": ("REAL", 0, None),
    "native_reference_price": ("REAL", 0, None),
    "native_gross_value": ("REAL", 0, None),
    "native_slippage_amount": ("REAL", 0, None),
    "native_commission": ("REAL", 0, None),
    "native_stamp_duty": ("REAL", 0, None),
    "native_transfer_fee": ("REAL", 0, None),
    "native_total_fees": ("REAL", 0, None),
    "native_net_cash_impact": ("REAL", 0, None),
    "fx_rate": ("REAL", 0, None),
    "market_rule_date": ("TEXT", 0, None),
    "market_rule_suspended": ("INTEGER", 0, None),
    "market_rule_closing_limit_state": ("TEXT", 0, None),
    "market_rule_official_close": ("REAL", 0, None),
    "market_rule_closing_gate_effective": ("INTEGER", 0, None),
}

# The one place the ALTER path cannot reproduce CREATE TABLE, and why.
#
# ``CREATE TABLE trades`` declares quantity/side/value NOT NULL. SQLite rejects
# ``ALTER TABLE ... ADD COLUMN <x> NOT NULL`` unless a non-null DEFAULT comes
# with it (there is no value it could put in the existing rows), and these three
# are backfilled from the legacy shares/action/total_value columns *after* being
# added -- so no default would be right. A database migrated from the legacy
# shape therefore holds them nullable forever, and only a full table rebuild
# would change that.
#
# Pre-existing and latent, not introduced by the run-history work: every writer
# goes through ``insert_trades``, which coerces (``int(... or 0)``,
# ``str(...).upper()``) before the value reaches SQLite, so nothing ever tries
# to store NULL. Listed rather than tolerated silently -- the cross-check below
# asserts this set is matched *exactly*, so a fourth divergence fails the test
# and a fixed one does too instead of leaving a stale allowance behind.
_KNOWN_ALTER_PATH_DIVERGENCES = {
    ("trades", "quantity"),
    ("trades", "side"),
    ("trades", "value"),
}


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


def _column_specs(path, table) -> dict[str, tuple[str, int, object]]:
    """``{name: (type, notnull, default)}`` for one table.

    PRAGMA table_info rows are ``(cid, name, type, notnull, dflt_value, pk)``.
    ``cid`` is dropped deliberately -- it is positional, and ALTER TABLE ADD
    COLUMN always appends, so asserting on it would pin the *order the
    migrations happen to run in* rather than the schema they produce.
    """
    conn = sqlite3.connect(str(path))
    try:
        return {
            name: (col_type, notnull, default)
            for _cid, name, col_type, notnull, default, _pk in conn.execute(
                f"PRAGMA table_info({table})"
            )
        }
    finally:
        conn.close()


def _assert_columns_match(path, table, expected):
    """Every expected column present *with its declared type and default*."""
    actual = _column_specs(path, table)
    missing = sorted(set(expected) - set(actual))
    assert not missing, f"{table} is missing lazily-migrated columns: {missing}"
    wrong = {
        name: {"expected": spec, "actual": actual[name]}
        for name, spec in expected.items()
        if actual[name] != spec
    }
    assert not wrong, f"{table} columns migrated with the wrong type/default: {wrong}"


def test_opening_a_pre_migration_database_adds_every_lazily_migrated_column(tmp_path):
    path = tmp_path / "pre-migration.db"
    _build_pre_migration_database(path)

    # Precondition: the columns really are absent, so a green assertion below
    # cannot be the CREATE TABLE path quietly supplying them.
    assert not (set(_TOKEN_COLUMNS) & set(_column_specs(path, "agent_runs")))
    assert not (set(_BASELINE_COLUMNS) & set(_column_specs(path, "agent_runs")))
    assert not (
        set(_CURRENCY_EQUITY_COLUMNS) & set(_column_specs(path, "equity_timeseries"))
    )
    assert not (
        (set(_TRADE_COLUMNS) | set(_CURRENCY_TRADE_COLUMNS))
        & set(_column_specs(path, "trades"))
    )

    BacktestDatabase(path)

    _assert_columns_match(path, "agent_runs", {**_TOKEN_COLUMNS, **_BASELINE_COLUMNS})
    _assert_columns_match(path, "equity_timeseries", _CURRENCY_EQUITY_COLUMNS)
    _assert_columns_match(
        path, "trades", {**_TRADE_COLUMNS, **_CURRENCY_TRADE_COLUMNS}
    )


def test_lazily_migrated_columns_match_a_freshly_created_database(tmp_path):
    """The migration path and the CREATE TABLE path must agree.

    This is the check that actually earns the phrase "behaviour-preserving": the
    ALTER statements and the CREATE TABLE DDL are two independent spellings of
    one schema, maintained by hand in the same file, and nothing else compares
    them. A deployed database gets its columns from the ALTERs while every test
    fixture and every fresh install gets them from CREATE TABLE -- so a drift
    between the two is invisible in both directions until prod does something a
    test database cannot reproduce.

    It also means the hard-coded expectations above cannot silently rot: if a
    column's declared type changes in one place only, this fails even when the
    literal in this file was updated to match.

    ``_KNOWN_ALTER_PATH_DIVERGENCES`` carries the only exception, with the
    reason. Compared as an exact set, so this test still fails if a *new*
    divergence appears or a listed one is resolved.
    """
    migrated_path = tmp_path / "migrated.db"
    _build_pre_migration_database(migrated_path)
    BacktestDatabase(migrated_path)

    fresh_path = tmp_path / "fresh.db"
    BacktestDatabase(fresh_path)

    divergent = {}
    for table, columns in (
        ("agent_runs", {**_TOKEN_COLUMNS, **_BASELINE_COLUMNS}),
        ("equity_timeseries", _CURRENCY_EQUITY_COLUMNS),
        ("trades", {**_TRADE_COLUMNS, **_CURRENCY_TRADE_COLUMNS}),
    ):
        migrated = _column_specs(migrated_path, table)
        fresh = _column_specs(fresh_path, table)
        for name in columns:
            if migrated[name] != fresh[name]:
                divergent[(table, name)] = {
                    "migrated": migrated[name],
                    "fresh": fresh[name],
                }

    unexpected = {k: v for k, v in divergent.items() if k not in _KNOWN_ALTER_PATH_DIVERGENCES}
    assert not unexpected, (
        "the ALTER path and CREATE TABLE disagree on columns not on the known list: "
        f"{unexpected}"
    )

    resolved = _KNOWN_ALTER_PATH_DIVERGENCES - set(divergent)
    assert not resolved, (
        "these divergences no longer exist -- drop them from "
        f"_KNOWN_ALTER_PATH_DIVERGENCES: {sorted(resolved)}"
    )

    # Pin the *shape* of the allowed exception too: nullability only. A listed
    # column coming back with a different declared type would otherwise ride
    # through on its name being on the list.
    for table, name in _KNOWN_ALTER_PATH_DIVERGENCES:
        migrated_spec = _column_specs(migrated_path, table)[name]
        fresh_spec = _column_specs(fresh_path, table)[name]
        assert migrated_spec[0] == fresh_spec[0], f"{table}.{name} type drifted"
        assert (migrated_spec[1], fresh_spec[1]) == (0, 1), (
            f"{table}.{name} should be nullable on the ALTER path and NOT NULL "
            f"on the CREATE path; got {migrated_spec} vs {fresh_spec}"
        )


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
        table: _column_specs(path, table)
        for table in ("agent_runs", "equity_timeseries", "trades")
    }

    BacktestDatabase(path)
    after_second = {
        table: _column_specs(path, table)
        for table in ("agent_runs", "equity_timeseries", "trades")
    }

    # Full specs, not just names: a reopen that re-declared a column with a
    # different type or default would leave the name set identical.
    assert after_second == after_first
