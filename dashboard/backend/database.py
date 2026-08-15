"""
SQLite database layer for agentic trading backtesting.
Handles schema initialization and CRUD operations.

Session isolation: session_id added to agent_runs table only.
Equity timeseries and trades can be verified through agent_runs ownership.
"""

import sqlite3
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Any

from dashboard.backend.paths import DEFAULT_DB_PATH
from dashboard.backend.db_url import describe_database_url

# Use persistent disk path if set (Render), otherwise local dashboard storage path
DB_PATH = Path(os.getenv("DATABASE_PATH", str(DEFAULT_DB_PATH)))

TRADE_OPTIONAL_AUDIT_FIELDS = (
    "reference_price",
    "gross_value",
    "slippage_amount",
    "commission",
    "stamp_duty",
    "transfer_fee",
    "total_fees",
    "net_cash_impact",
    "native_price",
    "native_value",
    "native_reference_price",
    "native_gross_value",
    "native_slippage_amount",
    "native_commission",
    "native_stamp_duty",
    "native_transfer_fee",
    "native_total_fees",
    "native_net_cash_impact",
    "fx_rate",
    "market_rule_date",
    "market_rule_suspended",
    "market_rule_closing_limit_state",
    "market_rule_official_close",
    "market_rule_closing_gate_effective",
)


def enable_wal(db_path) -> None:
    """Switch a SQLite file to WAL journal mode (best-effort, idempotent).

    Both DB layers (BacktestDatabase and the protocol RunStore) share one
    file; WAL lets request-thread reads proceed while a backtest finalize
    commits its heavy equity/trade writes. journal_mode is persisted in the
    file, so one switch covers every later connection from either layer.
    Filesystems without shared-memory support (some network mounts) can
    refuse WAL — keep the default rollback journal there rather than failing
    startup. Single definition so a future refinement can't drift between the
    two call sites.
    """
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        finally:
            conn.close()
    except sqlite3.Error:
        pass


def as_timestamp_text(value: Any) -> Any:
    """Normalise a timestamp argument to the text both stores keep.

    Every ``timestamp`` column in this schema is TEXT, and the two stores used
    to react very differently to a caller handing over a ``datetime`` instead
    of a string:

    * SQLite accepted it silently, through ``sqlite3``'s *default datetime
      adapter* -- which stores ``isoformat(" ")``, a **space** separator, and
      which Python deprecated in 3.12 and intends to remove.
    * Postgres refused it: psycopg sends the parameter typed as
      ``timestamp``/``timestamptz``, and Postgres has had no implicit
      assignment cast to ``text`` since 8.3, so the INSERT raised and the
      request 500'd.

    Converting at the boundary fixes both halves at once: the same call now
    works on either backend *and* lands byte-identical text, and the stored
    format no longer depends on a deprecated stdlib adapter that would
    otherwise flip the divergence around when it is removed (SQLite starting
    to raise while Postgres kept working).

    ``isoformat()`` -- a ``T`` separator -- is the format the rest of the
    backend already writes: ``insert_trades`` on both stores does exactly this,
    and ``external_run_service`` applies it to every decision timestamp before
    ``insert_decisions`` ever sees it.

    Non-datetime values pass through untouched, ``None`` included, so a missing
    NOT NULL timestamp still surfaces as an integrity error instead of being
    quietly stored as the string ``"None"``.
    """
    return value.isoformat() if hasattr(value, "isoformat") else value


class BacktestDatabase:
    """Minimal SQLite wrapper for equity curve storage."""

    def __init__(self, db_path: Path = None):
        if db_path is None:
            db_path = DB_PATH
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        enable_wal(self.db_path)
        self._init_schema()
        self._migrate_schema()  # Handle existing DBs
        self._ensure_equity_timeseries_uniqueness()

    def _get_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Return rows as dicts
        return conn
    
    def _init_schema(self):
        """Create tables if they don't exist (new installations)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # agent_runs: metadata about each backtest
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_runs (
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
                llm_calls INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                est_cost_usd REAL DEFAULT 0,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Index for session-scoped queries (only if table has session_id)
        try:
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_runs_session 
                ON agent_runs(session_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_runs_session_mode 
                ON agent_runs(session_id, mode)
            """)
        except Exception:
            # Indexes may fail if table exists without session_id; will be handled by migration
            pass
        
        # equity_timeseries: daily snapshots
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS equity_timeseries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                equity REAL NOT NULL,
                cash REAL NOT NULL,
                positions_value REAL NOT NULL,
                daily_return REAL,
                native_equity REAL,
                native_cash REAL,
                native_positions_value REAL,
                fx_rate REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES agent_runs(run_id),
                UNIQUE(run_id, timestamp)
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_run_timestamp 
            ON equity_timeseries(run_id, timestamp)
        """)
        
        # trades: detailed trade log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                value REAL NOT NULL,
                reason TEXT,
                reference_price REAL,
                gross_value REAL,
                slippage_amount REAL,
                commission REAL,
                stamp_duty REAL,
                transfer_fee REAL,
                total_fees REAL,
                net_cash_impact REAL,
                native_price REAL,
                native_value REAL,
                native_reference_price REAL,
                native_gross_value REAL,
                native_slippage_amount REAL,
                native_commission REAL,
                native_stamp_duty REAL,
                native_transfer_fee REAL,
                native_total_fees REAL,
                native_net_cash_impact REAL,
                fx_rate REAL,
                market_rule_date TEXT,
                market_rule_suspended INTEGER,
                market_rule_closing_limit_state TEXT,
                market_rule_official_close REAL,
                market_rule_closing_gate_effective INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES agent_runs(run_id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_run
            ON trades(run_id, timestamp)
        """)

        # backtest_decisions: hourly agent decisions (external + internal)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                decision_source TEXT NOT NULL,
                actions_submitted TEXT,
                actions_executed INTEGER DEFAULT 0,
                context_ref TEXT,
                actions_trace_ref TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES agent_runs(run_id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_decisions_run
            ON backtest_decisions(run_id, step_index)
        """)

        # idempotency_keys: replay-safe decision submissions (v2)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS idempotency_keys (
                run_id TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                idem_key TEXT NOT NULL,
                ack_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (run_id, idem_key)
            )
        """)

        # run_manifest: reproducibility manifest per v2 run (written at creation)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS run_manifest (
                run_id TEXT PRIMARY KEY,
                manifest_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()
    
    def _migrate_schema(self):
        """Migrate existing databases: add session_id and llm_model columns if missing."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Check what columns exist
            cursor.execute("PRAGMA table_info(agent_runs)")
            columns = [row[1] for row in cursor.fetchall()]
            
            # Add session_id if missing
            if 'session_id' not in columns:
                print("🔄 Migrating: Adding session_id to agent_runs...")
                cursor.execute("""
                    ALTER TABLE agent_runs 
                    ADD COLUMN session_id TEXT DEFAULT 'legacy-demo-session'
                """)
                cursor.execute("UPDATE agent_runs SET session_id = 'legacy-demo-session' WHERE session_id IS NULL")
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_agent_runs_session 
                    ON agent_runs(session_id)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_agent_runs_session_mode 
                    ON agent_runs(session_id, mode)
                """)
                conn.commit()
                print("✅ Added session_id to agent_runs")
            
            # Add llm_model if missing (tracks which LLM was used)
            if 'llm_model' not in columns:
                print("🔄 Migrating: Adding llm_model to agent_runs...")
                cursor.execute("""
                    ALTER TABLE agent_runs 
                    ADD COLUMN llm_model TEXT DEFAULT 'rule-based'
                """)
                cursor.execute("UPDATE agent_runs SET llm_model = 'rule-based' WHERE llm_model IS NULL")
                conn.commit()
                print("✅ Added llm_model to agent_runs")

            cursor.execute("PRAGMA table_info(agent_runs)")
            columns = [row[1] for row in cursor.fetchall()]

            if 'baseline_djia_run_id' not in columns:
                print("🔄 Migrating: Adding baseline_djia_run_id to agent_runs...")
                cursor.execute("""
                    ALTER TABLE agent_runs
                    ADD COLUMN baseline_djia_run_id TEXT
                """)
                conn.commit()
                print("✅ Added baseline_djia_run_id to agent_runs")

            if 'baseline_buyhold_run_id' not in columns:
                print("🔄 Migrating: Adding baseline_buyhold_run_id to agent_runs...")
                cursor.execute("""
                    ALTER TABLE agent_runs
                    ADD COLUMN baseline_buyhold_run_id TEXT
                """)
                conn.commit()
                print("✅ Added baseline_buyhold_run_id to agent_runs")

            # Token usage / cost tracking columns + the JSON config snapshot
            # (metadata records env-dependent knobs like the effective
            # LLM_MAX_OUTPUT_TOKENS that shaped the run).
            # Each ALTER is spelled out in full rather than assembled from the
            # column name and type: the Postgres-twin parity guard
            # (tests/test_store_twin_parity.py) reads this module's DDL out of
            # its *source text*, and an f-string collapses to a placeholder it
            # cannot see a column name through. Literal SQL is also greppable.
            token_columns = [
                ("llm_calls",
                 "ALTER TABLE agent_runs ADD COLUMN llm_calls INTEGER DEFAULT 0"),
                ("input_tokens",
                 "ALTER TABLE agent_runs ADD COLUMN input_tokens INTEGER DEFAULT 0"),
                ("output_tokens",
                 "ALTER TABLE agent_runs ADD COLUMN output_tokens INTEGER DEFAULT 0"),
                ("est_cost_usd",
                 "ALTER TABLE agent_runs ADD COLUMN est_cost_usd REAL DEFAULT 0"),
                ("metadata",
                 "ALTER TABLE agent_runs ADD COLUMN metadata TEXT"),
            ]
            for col_name, add_column_sql in token_columns:
                if col_name not in columns:
                    print(f"🔄 Migrating: Adding {col_name} to agent_runs...")
                    cursor.execute(add_column_sql)
                    conn.commit()
                    print(f"✅ Added {col_name} to agent_runs")
            
            if 'session_id' in columns and 'llm_model' in columns:
                print("✅ Schema up-to-date (session_id, llm_model exist)")

            cursor.execute("PRAGMA table_info(backtest_decisions)")
            dec_cols = {row[1] for row in cursor.fetchall()}
            if dec_cols and "context_ref" not in dec_cols:
                print("🔄 Migrating: Adding context_ref to backtest_decisions...")
                cursor.execute("ALTER TABLE backtest_decisions ADD COLUMN context_ref TEXT")
                conn.commit()
                print("✅ Added context_ref to backtest_decisions")

            if dec_cols and "actions_trace_ref" not in dec_cols:
                print("🔄 Migrating: Adding actions_trace_ref to backtest_decisions...")
                cursor.execute("ALTER TABLE backtest_decisions ADD COLUMN actions_trace_ref TEXT")
                conn.commit()
                print("✅ Added actions_trace_ref to backtest_decisions")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS idempotency_keys (
                    run_id TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    idem_key TEXT NOT NULL,
                    ack_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (run_id, idem_key)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS run_manifest (
                    run_id TEXT PRIMARY KEY,
                    manifest_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            self._ensure_decisions_table(cursor)
            self._migrate_trades_schema(cursor)
            self._migrate_currency_audit_schema(cursor)
            conn.commit()
        
        except Exception as e:
            print(f"⚠️ Migration warning: {e}")
        
        finally:
            conn.close()

        # Trades migration is critical for external backtest finalize; retry isolated.
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            self._migrate_trades_schema(cursor)
            self._migrate_currency_audit_schema(cursor)
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"⚠️ Trades migration retry warning: {e}")

    @staticmethod
    def _has_equity_timeseries_unique_index(cursor) -> bool:
        """Return whether all equity points are protected by the natural key."""
        cursor.execute("PRAGMA index_list(equity_timeseries)")
        for index in cursor.fetchall():
            is_unique = bool(index[2])
            is_partial = bool(index[4]) if len(index) > 4 else False
            if not is_unique or is_partial:
                continue

            index_name = str(index[1]).replace('"', '""')
            cursor.execute(f'PRAGMA index_info("{index_name}")')
            columns = [row[2] for row in cursor.fetchall()]
            if columns == ["run_id", "timestamp"]:
                return True
        return False

    # A locked database is retried once with a fresh connection (each attempt
    # gets its own busy timeout) before the migration is deferred.
    _UNIQUENESS_MIGRATION_ATTEMPTS = 2

    def _ensure_equity_timeseries_uniqueness(self) -> None:
        """Upgrade legacy equity tables so reruns replace existing points.

        Fails hard on a *data* problem (leftover duplicates, an index name
        already taken) — that needs a human. Lock contention is different: it
        is transient, and this module ends with a module-level singleton, so
        raising here aborts ``import dashboard.backend.database`` and the app
        never boots. The migrations above degrade on the same condition, and
        every equity writer already uses INSERT OR REPLACE, so deferring to the
        next startup is no worse than the state before this migration existed.
        """
        last_error: Optional[sqlite3.Error] = None
        for attempt in range(1, self._UNIQUENESS_MIGRATION_ATTEMPTS + 1):
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                if self._has_equity_timeseries_unique_index(cursor):
                    return
                self._apply_equity_timeseries_uniqueness(conn, cursor)
                return
            except sqlite3.OperationalError as exc:
                last_error = exc
                if attempt < self._UNIQUENESS_MIGRATION_ATTEMPTS:
                    print(f"⚠️ equity_timeseries uniqueness migration retry: {exc}")
            except sqlite3.Error as exc:
                raise RuntimeError(
                    "equity_timeseries unique constraint migration failed"
                ) from exc
            finally:
                conn.close()

        print(f"⚠️ equity_timeseries uniqueness migration deferred: {last_error}")

    def _apply_equity_timeseries_uniqueness(self, conn, cursor) -> None:
        """Dedup to the newest point per (run_id, timestamp), then protect it.

        One transaction, so a failure never leaves rows deleted without the
        constraint that justified deleting them. The index is re-read after
        creation because ``IF NOT EXISTS`` silently succeeds when a *non-unique*
        index already owns the name.
        """
        with conn:
            cursor.execute(
                """
                DELETE FROM equity_timeseries
                WHERE id NOT IN (
                    SELECT MAX(id)
                    FROM equity_timeseries
                    GROUP BY run_id, timestamp
                )
                """
            )
            cursor.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                    uq_equity_timeseries_run_timestamp
                ON equity_timeseries(run_id, timestamp)
                """
            )
            if not self._has_equity_timeseries_unique_index(cursor):
                raise RuntimeError(
                    "equity_timeseries unique constraint migration failed"
                )

    def _migrate_trades_schema(self, cursor) -> None:
        """Upgrade legacy trades table (shares/action/total_value) to new columns."""
        cursor.execute("PRAGMA table_info(trades)")
        columns = {row[1] for row in cursor.fetchall()}
        if not columns:
            return

        needed = {"quantity", "side", "value", "reason"}
        if needed.issubset(columns) and "shares" not in columns:
            return

        print("🔄 Migrating: upgrading trades table schema...")
        # Literal SQL, not f-string assembly -- see the note on token_columns.
        additions = [
            ("quantity", "ALTER TABLE trades ADD COLUMN quantity INTEGER"),
            ("side", "ALTER TABLE trades ADD COLUMN side TEXT"),
            ("value", "ALTER TABLE trades ADD COLUMN value REAL"),
            ("reason", "ALTER TABLE trades ADD COLUMN reason TEXT"),
        ]
        for name, add_column_sql in additions:
            if name not in columns:
                cursor.execute(add_column_sql)

        if "shares" in columns:
            cursor.execute("UPDATE trades SET quantity = shares WHERE quantity IS NULL")
        if "action" in columns:
            cursor.execute("UPDATE trades SET side = UPPER(action) WHERE side IS NULL")
        if "total_value" in columns:
            cursor.execute("UPDATE trades SET value = total_value WHERE value IS NULL")

        print("✅ trades table migrated (quantity, side, value, reason)")

    @staticmethod
    def _migrate_currency_audit_schema(cursor) -> None:
        """Add nullable currency and execution audit fields to stored history."""
        # Literal SQL, not f-string assembly -- see the note on token_columns.
        # This loop interpolated the *table* name too, which made the parity
        # guard's parser see a whole phantom table.
        additions = {
            "equity_timeseries": (
                ("native_equity",
                 "ALTER TABLE equity_timeseries ADD COLUMN native_equity REAL"),
                ("native_cash",
                 "ALTER TABLE equity_timeseries ADD COLUMN native_cash REAL"),
                ("native_positions_value",
                 "ALTER TABLE equity_timeseries ADD COLUMN native_positions_value REAL"),
                ("fx_rate",
                 "ALTER TABLE equity_timeseries ADD COLUMN fx_rate REAL"),
            ),
            "trades": (
                ("reference_price", "ALTER TABLE trades ADD COLUMN reference_price REAL"),
                ("gross_value", "ALTER TABLE trades ADD COLUMN gross_value REAL"),
                ("slippage_amount", "ALTER TABLE trades ADD COLUMN slippage_amount REAL"),
                ("commission", "ALTER TABLE trades ADD COLUMN commission REAL"),
                ("stamp_duty", "ALTER TABLE trades ADD COLUMN stamp_duty REAL"),
                ("transfer_fee", "ALTER TABLE trades ADD COLUMN transfer_fee REAL"),
                ("total_fees", "ALTER TABLE trades ADD COLUMN total_fees REAL"),
                ("net_cash_impact", "ALTER TABLE trades ADD COLUMN net_cash_impact REAL"),
                ("native_price", "ALTER TABLE trades ADD COLUMN native_price REAL"),
                ("native_value", "ALTER TABLE trades ADD COLUMN native_value REAL"),
                ("native_reference_price", "ALTER TABLE trades ADD COLUMN native_reference_price REAL"),
                ("native_gross_value", "ALTER TABLE trades ADD COLUMN native_gross_value REAL"),
                ("native_slippage_amount", "ALTER TABLE trades ADD COLUMN native_slippage_amount REAL"),
                ("native_commission", "ALTER TABLE trades ADD COLUMN native_commission REAL"),
                ("native_stamp_duty", "ALTER TABLE trades ADD COLUMN native_stamp_duty REAL"),
                ("native_transfer_fee", "ALTER TABLE trades ADD COLUMN native_transfer_fee REAL"),
                ("native_total_fees", "ALTER TABLE trades ADD COLUMN native_total_fees REAL"),
                ("native_net_cash_impact", "ALTER TABLE trades ADD COLUMN native_net_cash_impact REAL"),
                ("fx_rate", "ALTER TABLE trades ADD COLUMN fx_rate REAL"),
                ("market_rule_date", "ALTER TABLE trades ADD COLUMN market_rule_date TEXT"),
                ("market_rule_suspended", "ALTER TABLE trades ADD COLUMN market_rule_suspended INTEGER"),
                ("market_rule_closing_limit_state", "ALTER TABLE trades ADD COLUMN market_rule_closing_limit_state TEXT"),
                ("market_rule_official_close", "ALTER TABLE trades ADD COLUMN market_rule_official_close REAL"),
                ("market_rule_closing_gate_effective", "ALTER TABLE trades ADD COLUMN market_rule_closing_gate_effective INTEGER"),
            ),
        }
        for table, fields in additions.items():
            cursor.execute(f"PRAGMA table_info({table})")
            columns = {row[1] for row in cursor.fetchall()}
            if not columns:
                continue
            for field, add_column_sql in fields:
                if field not in columns:
                    cursor.execute(add_column_sql)

    def _ensure_decisions_table(self, cursor) -> None:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                decision_source TEXT NOT NULL,
                actions_submitted TEXT,
                actions_executed INTEGER DEFAULT 0,
                context_ref TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES agent_runs(run_id)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_decisions_run
            ON backtest_decisions(run_id, step_index)
        """)

    def _trades_column_set(self, cursor) -> set:
        cursor.execute("PRAGMA table_info(trades)")
        return {row[1] for row in cursor.fetchall()}
    
    def insert_run(self, run_id: str, session_id: str, agent_name: str, mode: str, 
                   start_date: str, end_date: str, 
                   initial_equity: float,
                   final_equity: Optional[float] = None,
                   total_return: Optional[float] = None,
                   sharpe_ratio: Optional[float] = None,
                   max_drawdown: Optional[float] = None,
                   num_trades: int = 0,
                   llm_model: str = "rule-based",
                   llm_calls: int = 0,
                   input_tokens: int = 0,
                   output_tokens: int = 0,
                   est_cost_usd: float = 0.0,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """Insert a new backtest run with session_id, LLM model and token-cost tracking.

        ``metadata`` is an optional JSON config snapshot (e.g. the effective
        LLM_MAX_OUTPUT_TOKENS in force during the run)."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO agent_runs
            (run_id, session_id, agent_name, mode, start_date, end_date,
             initial_equity, final_equity, total_return, sharpe_ratio,
             max_drawdown, num_trades, llm_model,
             llm_calls, input_tokens, output_tokens, est_cost_usd, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (run_id, session_id, agent_name, mode, start_date, end_date,
              initial_equity, final_equity, total_return, sharpe_ratio,
              max_drawdown, num_trades, llm_model,
              llm_calls, input_tokens, output_tokens, est_cost_usd,
              json.dumps(metadata) if metadata is not None else None))

        conn.commit()
        conn.close()

    def update_run_baselines(
        self,
        run_id: str,
        *,
        djia_run_id: Optional[str] = None,
        buyhold_run_id: Optional[str] = None,
    ) -> None:
        """Link an external backtest run to its paired baseline runs."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE agent_runs
            SET baseline_djia_run_id = COALESCE(?, baseline_djia_run_id),
                baseline_buyhold_run_id = COALESCE(?, baseline_buyhold_run_id),
                updated_at = CURRENT_TIMESTAMP
            WHERE run_id = ?
            """,
            (djia_run_id, buyhold_run_id, run_id),
        )
        conn.commit()
        conn.close()
    
    def insert_equity_point(self, run_id: str, timestamp: str, 
                          equity: float, cash: float, 
                          positions_value: float,
                          daily_return: Optional[float] = None,
                          native_equity: Optional[float] = None,
                          native_cash: Optional[float] = None,
                          native_positions_value: Optional[float] = None,
                          fx_rate: Optional[float] = None) -> None:
        """Insert a single equity data point."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO equity_timeseries 
            (run_id, timestamp, equity, cash, positions_value, daily_return,
             native_equity, native_cash, native_positions_value, fx_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, as_timestamp_text(timestamp), equity, cash, positions_value,
            daily_return,
            native_equity, native_cash, native_positions_value, fx_rate,
        ))

        conn.commit()
        conn.close()
    
    def insert_equity_points(self, run_id: str,
                           points: List[Dict[str, Any]],
                           replace: bool = True) -> None:
        """Replace this run's equity curve with ``points``, atomically.

        Each point should have: timestamp, equity, cash, positions_value, [daily_return]

        Every production caller hands over a *whole* curve, and a rerun can
        legitimately produce a different set of timestamps (fewer bars, a
        partial run, a changed symbol list). The (run_id, timestamp) unique key
        only collapses timestamps that repeat, so without the delete the
        leftovers of the previous curve stay spliced into the new one — which
        is exactly the force-refresh case. Pass ``replace=False`` to append to
        an existing curve instead. An empty ``points`` list is a no-op rather
        than a wipe: a failed rerun must not erase the curve on the board.
        """
        if not points:
            return

        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                if replace:
                    cursor.execute(
                        "DELETE FROM equity_timeseries WHERE run_id = ?", (run_id,)
                    )
                cursor.executemany("""
                    INSERT OR REPLACE INTO equity_timeseries
                    (run_id, timestamp, equity, cash, positions_value, daily_return,
                     native_equity, native_cash, native_positions_value, fx_rate)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    (
                        run_id,
                        as_timestamp_text(point['timestamp']),
                        point['equity'],
                        point['cash'],
                        point['positions_value'],
                        point.get('daily_return'),
                        point.get('native_equity'),
                        point.get('native_cash'),
                        point.get('native_positions_value'),
                        point.get('fx_rate'),
                    )
                    for point in points
                ])
        finally:
            conn.close()
    
    @staticmethod
    def _parse_run_row(run: Dict) -> Dict:
        """Decode the JSON metadata column (SELECT * returns raw text) so
        every agent_runs reader hands out the same parsed shape."""
        raw = run.get("metadata")
        if raw is not None:
            try:
                run["metadata"] = json.loads(raw)
            except (TypeError, ValueError):
                run["metadata"] = None
        return run

    def get_all_runs(self) -> List[Dict]:
        """Get metadata for all runs."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM agent_runs ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        
        return [self._parse_run_row(dict(row)) for row in rows]
    
    def get_runs_by_session(self, session_id: str) -> List[Dict]:
        """Get all runs for a specific session."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM agent_runs
            WHERE session_id = ?
            ORDER BY created_at DESC
        """, (session_id,))

        rows = cursor.fetchall()
        conn.close()

        return [self._parse_run_row(dict(row)) for row in rows]

    def get_runs_by_sessions(self, session_ids: List[str]) -> Dict[str, List[Dict]]:
        """Get all runs for several sessions in one query, grouped by session.

        Batch companion to ``get_runs_by_session`` so listings that enrich many
        agents don't issue one query per agent. Every requested session id is
        present in the result (empty list when it has no runs); per-session
        ordering matches ``get_runs_by_session`` (created_at DESC).
        """
        grouped: Dict[str, List[Dict]] = {sid: [] for sid in session_ids if sid}
        if not grouped:
            return grouped
        conn = self._get_connection()
        cursor = conn.cursor()

        placeholders = ",".join("?" * len(grouped))
        cursor.execute(f"""
            SELECT * FROM agent_runs
            WHERE session_id IN ({placeholders})
            ORDER BY created_at DESC
        """, list(grouped))

        rows = cursor.fetchall()
        conn.close()

        for row in rows:
            run = self._parse_run_row(dict(row))
            grouped[run["session_id"]].append(run)
        return grouped
    
    def get_run(self, run_id: str) -> Optional[Dict]:
        """Get metadata for a specific run (no session check)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM agent_runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        conn.close()
        
        return self._parse_run_row(dict(row)) if row else None
    
    def get_run_with_session(self, run_id: str, session_id: str) -> Optional[Dict]:
        """Get a run, verifying it belongs to the session."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM agent_runs 
            WHERE run_id = ? AND session_id = ?
        """, (run_id, session_id))
        row = cursor.fetchone()
        conn.close()
        
        return self._parse_run_row(dict(row)) if row else None
    
    def get_equity_curve(self, run_id: str) -> List[Dict]:
        """Get full equity curve for a run."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, equity, cash, positions_value, daily_return,
                   native_equity, native_cash, native_positions_value, fx_rate
            FROM equity_timeseries 
            WHERE run_id = ?
            ORDER BY timestamp ASC
        """, (run_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        result = []
        audit_fields = {
            "native_equity",
            "native_cash",
            "native_positions_value",
            "fx_rate",
        }
        for row in rows:
            item = dict(row)
            for field in audit_fields:
                if item.get(field) is None:
                    item.pop(field, None)
            result.append(item)
        return result
    
    def get_equity_curves(self, run_ids: List[str]) -> Dict[str, List[Dict]]:
        """Get equity curves for multiple runs."""
        result = {}
        for run_id in run_ids:
            result[run_id] = self.get_equity_curve(run_id)
        return result
    
    def get_runs_by_mode(self, mode: str) -> List[Dict]:
        """Get all runs for a specific mode ('backtest' or 'paper')."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM agent_runs
            WHERE mode = ?
            ORDER BY created_at DESC
        """, (mode,))

        rows = cursor.fetchall()
        conn.close()

        return [self._parse_run_row(dict(row)) for row in rows]

    def insert_trades(self, run_id: str, trades: List[Dict[str, Any]]) -> None:
        """Batch insert trade records for a backtest run."""
        if not trades:
            return
        conn = self._get_connection()
        cursor = conn.cursor()
        self._migrate_trades_schema(cursor)
        conn.commit()
        columns = self._trades_column_set(cursor)
        for trade in trades:
            ts = trade.get("timestamp")
            if hasattr(ts, "isoformat"):
                ts = ts.isoformat()
            side = str(trade.get("side", "")).upper()
            qty = int(trade.get("shares") or trade.get("quantity") or 0)
            price = float(trade.get("price") or 0)
            value = float(trade.get("cost") or trade.get("proceeds") or trade.get("value") or qty * price)

            if "quantity" in columns:
                col_names = ["run_id", "timestamp", "symbol", "quantity", "side", "price", "value"]
                col_values = [run_id, str(ts), trade.get("symbol"), qty, side, price, value]
                if "reason" in columns:
                    col_names.append("reason")
                    col_values.append(trade.get("reason"))
                for field in TRADE_OPTIONAL_AUDIT_FIELDS:
                    if field in columns:
                        col_names.append(field)
                        col_values.append(trade.get(field))
                # Legacy columns may still be NOT NULL after migration
                if "action" in columns:
                    col_names.extend(["action", "shares", "total_value"])
                    col_values.extend([side, qty, value])
                placeholders = ", ".join("?" for _ in col_names)
                cursor.execute(
                    f"INSERT INTO trades ({', '.join(col_names)}) VALUES ({placeholders})",
                    col_values,
                )
            elif "shares" in columns:
                cursor.execute("""
                    INSERT INTO trades
                    (run_id, timestamp, symbol, action, shares, price, total_value)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    str(ts),
                    trade.get("symbol"),
                    side,
                    qty,
                    price,
                    value,
                ))
            else:
                raise RuntimeError("trades table has unsupported schema")
        conn.commit()
        conn.close()

    def insert_decisions(self, run_id: str, decisions: List[Dict[str, Any]]) -> None:
        """Batch insert hourly decision log entries."""
        if not decisions:
            return
        conn = self._get_connection()
        cursor = conn.cursor()
        self._ensure_decisions_table(cursor)
        conn.commit()
        for entry in decisions:
            cursor.execute("""
                INSERT INTO backtest_decisions
                (run_id, step_index, timestamp, decision_source, actions_submitted, actions_executed, context_ref)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                entry.get("step_index", 0),
                as_timestamp_text(entry.get("timestamp")),
                entry.get("decision_source"),
                json.dumps(entry.get("actions_submitted") or []),
                entry.get("actions_executed", 0),
                entry.get("context_ref"),
            ))
        conn.commit()
        conn.close()

    def get_trades(self, run_id: str) -> List[Dict]:
        conn = self._get_connection()
        cursor = conn.cursor()
        columns = self._trades_column_set(cursor)
        if "quantity" in columns:
            selected = [
                "timestamp", "symbol", "quantity", "side", "price", "value", "reason"
            ]
            selected.extend(
                field
                for field in TRADE_OPTIONAL_AUDIT_FIELDS
                if field in columns
            )
            cursor.execute(
                f"SELECT {', '.join(selected)} FROM trades WHERE run_id = ? "
                "ORDER BY timestamp ASC, id ASC",
                (run_id,),
            )
        else:
            cursor.execute("""
                SELECT timestamp, symbol, shares AS quantity, action AS side,
                       price, total_value AS value
                FROM trades WHERE run_id = ?
                ORDER BY timestamp ASC, id ASC
            """, (run_id,))
        rows = cursor.fetchall()
        conn.close()
        result = []
        for row in rows:
            item = dict(row)
            for field in TRADE_OPTIONAL_AUDIT_FIELDS:
                if item.get(field) is None:
                    item.pop(field, None)
            result.append(item)
        return result

    def get_decisions(self, run_id: str) -> List[Dict]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT step_index, timestamp, decision_source, actions_submitted,
                   actions_executed, context_ref
            FROM backtest_decisions WHERE run_id = ?
            ORDER BY step_index ASC
        """, (run_id,))
        rows = cursor.fetchall()
        conn.close()
        result = []
        for row in rows:
            item = dict(row)
            try:
                item["actions_submitted"] = json.loads(item.get("actions_submitted") or "[]")
            except json.JSONDecodeError:
                item["actions_submitted"] = []
            result.append(item)
        return result

    def put_idempotency(self, run_id: str, step_index: int,
                        idem_key: str, ack: Dict[str, Any]) -> None:
        """Store the ack for an idempotency key. INSERT OR IGNORE keeps the first write."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO idempotency_keys
            (run_id, step_index, idem_key, ack_json)
            VALUES (?, ?, ?, ?)
        """, (run_id, step_index, idem_key, json.dumps(ack)))
        conn.commit()
        conn.close()

    def get_idempotency(self, run_id: str, step_index: int,
                        idem_key: str) -> Optional[Dict[str, Any]]:
        """Look up a stored ack by (run_id, idem_key).

        The key scope is (run_id, idem_key), NOT the step. This realizes the spec
        §5.2 *intent* — the idempotency_key enables "safe retries past the
        step_already_closed race" — which per-step keying cannot provide: this
        engine advances the step synchronously inside submit_decisions, so a retry
        arriving after the advance would look up a later step and miss the record,
        re-executing the decision. (The spec's §5.2 mechanism wording lists
        step_index in the key; that contradicts its own stated intent for a
        synchronous-advance engine.) step_index is accepted for call symmetry with
        put_idempotency, where it is stored as audit metadata, but is not part of
        the lookup key.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT ack_json FROM idempotency_keys
            WHERE run_id = ? AND idem_key = ?
        """, (run_id, idem_key))
        row = cursor.fetchone()
        conn.close()
        return json.loads(row["ack_json"]) if row else None

    def insert_run_manifest(self, run_id: str, manifest: Dict[str, Any]) -> None:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO run_manifest (run_id, manifest_json)
            VALUES (?, ?)
        """, (run_id, json.dumps(manifest)))
        conn.commit()
        conn.close()

    def get_run_manifest(self, run_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT manifest_json FROM run_manifest WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        conn.close()
        return json.loads(row["manifest_json"]) if row else None
    
    def delete_run(self, run_id: str) -> None:
        """Delete a run and all its data."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM equity_timeseries WHERE run_id = ?", (run_id,))
        cursor.execute("DELETE FROM trades WHERE run_id = ?", (run_id,))
        cursor.execute("DELETE FROM backtest_decisions WHERE run_id = ?", (run_id,))
        cursor.execute("DELETE FROM run_manifest WHERE run_id = ?", (run_id,))
        cursor.execute("DELETE FROM agent_runs WHERE run_id = ?", (run_id,))

        conn.commit()
        conn.close()

    def clear_all(self) -> None:
        """Clear all data (useful for testing)."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM equity_timeseries")
        cursor.execute("DELETE FROM trades")
        cursor.execute("DELETE FROM backtest_decisions")
        cursor.execute("DELETE FROM run_manifest")
        cursor.execute("DELETE FROM agent_runs")

        conn.commit()
        conn.close()


def _build_backtest_db():
    # AGENT_RUNS_DATABASE_URL only, deliberately: CONTENT_DATABASE_URL is scoped to
    # agents/versions/strategies and USERS_DATABASE_URL to accounts; neither may
    # select the run-history database (spec, Decision 3). Do not "simplify" this
    # into a fallback chain.
    database_url = os.getenv("AGENT_RUNS_DATABASE_URL")
    if database_url:
        from dashboard.backend.database_postgres import PostgresBacktestDatabase

        # print(), not logger.info() -- info is invisible under the prod logging
        # config. See users.py's _build_user_store for the full rationale.
        print(f"run history backend: postgres ({describe_database_url(database_url)})")
        return PostgresBacktestDatabase(database_url)
    print("run history backend: sqlite (ephemeral on Render)")
    return BacktestDatabase()


# Singleton instance
db = _build_backtest_db()
