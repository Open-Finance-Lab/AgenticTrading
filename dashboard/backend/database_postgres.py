"""Postgres-backed BacktestDatabase implementation (run history, the "cold half").

Selected instead of the default SQLite BacktestDatabase when AGENT_RUNS_DATABASE_URL is
set (see database.py's _build_backtest_db). Exists because the SQLite store lives
in DATABASE_PATH, which resets to the committed seed database on every deploy of
the disk-less Render free-tier host -- silently deleting every backtest run,
equity curve, trade log and decision log, which is why issue #145 (the leaderboard
refresh cron) is blocked. Method surface and return schemas are identical to
BacktestDatabase, and for the overwhelming majority of calls only the SQL
dialect differs.

Behavioral divergences, in full
-------------------------------
Every known way this store behaves differently from the SQLite one. Keep this
list exhaustive: a divergence that is real but undocumented is how a twin
quietly stops being a twin, and an earlier revision of this docstring claimed
"exactly two" while three more existed.

1. ``insert_run`` does not reset ``created_at`` on re-insert (first-seen
   semantics), where SQLite's ``INSERT OR REPLACE`` does. This is
   *user-visible* -- see ``insert_run``.
2. ``insert_run`` explicitly refreshes ``updated_at`` on re-insert, which
   SQLite's REPLACE also does, but for the opposite reason -- see
   ``insert_run``.
3. ``insert_run`` leaves ``baseline_djia_run_id``/``baseline_buyhold_run_id``
   intact on re-insert, where SQLite's REPLACE nulls them -- see ``insert_run``.
4. ``equity_timeseries``/``trades``/``backtest_decisions`` carry a *live* FK to
   ``agent_runs`` here. The SQLite schema declares the same FKs but never
   issues ``PRAGMA foreign_keys=ON``, so they are inert there -- see
   ``_init_schema``.
5. ``get_equity_curves`` is a single batched query; the SQLite twin loops.
   Same return shape either way.
6. ``trades`` never carried the legacy ``shares``/``action``/``total_value``
   columns, so ``insert_trades``/``get_trades`` have no introspection branch --
   see those methods.

The hot half stays local: get_idempotency/put_idempotency are delegated to an
embedded plain BacktestDatabase so the per-step agent request never gains a
network round-trip. protocol_runs/protocol_steps are not ours -- they belong to
domain/runs/repository.py and are untouched.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from dashboard.backend.database import (
    BacktestDatabase,
    TRADE_OPTIONAL_AUDIT_FIELDS,
    as_timestamp_text,
)
from dashboard.backend.db_url import require_postgres_url


class PostgresBacktestDatabase:
    def __init__(self, database_url: str):
        self.database_url = require_postgres_url(database_url)
        self._sqlite = BacktestDatabase()   # hot half: idempotency_keys stays local
        self._init_schema()

    def _get_connection(self):
        # Pooled checkout: same context-manager transaction semantics as
        # psycopg.connect (commit on clean exit), returned to the pool on close.
        from dashboard.backend.db_pool import get_pool

        return get_pool(self.database_url).connection()

    def _init_schema(self) -> None:
        # DIVERGENCE 4 (module docstring): the ``REFERENCES agent_runs(run_id)
        # ON DELETE CASCADE`` clauses below are *live*. database.py declares the
        # same three FKs but never issues ``PRAGMA foreign_keys=ON`` -- SQLite
        # defaults that off per connection, and no connection there turns it on
        # -- so on SQLite they are decorative and enforce nothing.
        #
        # Two consequences worth knowing before touching either schema:
        #   * Deleting an ``agent_runs`` row cascades here. SQLite relies on
        #     delete_run/clear_all issuing the child DELETEs by hand, which is
        #     why both still do (and must keep doing -- it is the only thing
        #     removing those rows there).
        #   * A child insert for a run_id that does not exist *fails* here and
        #     succeeds on SQLite. That is what makes the backfill's
        #     "skipped_orphan" accounting load-bearing rather than cosmetic: an
        #     orphaned equity/trade/decision row in the source, which SQLite
        #     was happy to hold, cannot be written here at all.
        #
        # The parity guard cannot see any of this: test_store_twin_parity
        # compares column *names*, so constraints and enforcement mode are
        # invisible to it. Only the @pg_only mirror suite exercises it.
        #
        # created_at/updated_at are TEXT with a Postgres DEFAULT that produces
        # SQLite's exact CURRENT_TIMESTAMP string ("YYYY-MM-DD HH:MM:SS", UTC),
        # so read shapes are identical across both stores and the *database*
        # stamps the row exactly like SQLite does -- not clock-dependent app code.
        created_at_default = (
            "DEFAULT to_char(now() AT TIME ZONE 'utc', 'YYYY-MM-DD HH24:MI:SS')"
        )

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # --------------------------------------------------------
                # CREATE: full post-migration column set for all five
                # "cold" tables (agent_runs, equity_timeseries, trades,
                # backtest_decisions, run_manifest). Mirrors database.py
                # after every lazy migration has run, post Task 2's
                # actions_trace_ref addition.
                # --------------------------------------------------------
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS agent_runs (
                        run_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        agent_name TEXT NOT NULL,
                        mode TEXT NOT NULL,
                        start_date TEXT NOT NULL,
                        end_date TEXT NOT NULL,
                        initial_equity DOUBLE PRECISION NOT NULL,
                        final_equity DOUBLE PRECISION,
                        total_return DOUBLE PRECISION,
                        sharpe_ratio DOUBLE PRECISION,
                        max_drawdown DOUBLE PRECISION,
                        num_trades INTEGER DEFAULT 0,
                        llm_model TEXT DEFAULT 'rule-based',
                        llm_calls INTEGER DEFAULT 0,
                        input_tokens INTEGER DEFAULT 0,
                        output_tokens INTEGER DEFAULT 0,
                        est_cost_usd DOUBLE PRECISION DEFAULT 0,
                        metadata TEXT,
                        created_at TEXT NOT NULL {created_at_default},
                        updated_at TEXT NOT NULL {created_at_default},
                        baseline_djia_run_id TEXT,
                        baseline_buyhold_run_id TEXT
                    )
                    """
                )

                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS equity_timeseries (
                        id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                        run_id TEXT NOT NULL
                            REFERENCES agent_runs(run_id) ON DELETE CASCADE,
                        timestamp TEXT NOT NULL,
                        equity DOUBLE PRECISION NOT NULL,
                        cash DOUBLE PRECISION NOT NULL,
                        positions_value DOUBLE PRECISION NOT NULL,
                        daily_return DOUBLE PRECISION,
                        native_equity DOUBLE PRECISION,
                        native_cash DOUBLE PRECISION,
                        native_positions_value DOUBLE PRECISION,
                        fx_rate DOUBLE PRECISION,
                        created_at TEXT NOT NULL {created_at_default},
                        UNIQUE (run_id, timestamp)
                    )
                    """
                )

                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS trades (
                        id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                        run_id TEXT NOT NULL
                            REFERENCES agent_runs(run_id) ON DELETE CASCADE,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        side TEXT NOT NULL,
                        price DOUBLE PRECISION NOT NULL,
                        value DOUBLE PRECISION NOT NULL,
                        reason TEXT,
                        reference_price DOUBLE PRECISION,
                        gross_value DOUBLE PRECISION,
                        slippage_amount DOUBLE PRECISION,
                        commission DOUBLE PRECISION,
                        stamp_duty DOUBLE PRECISION,
                        transfer_fee DOUBLE PRECISION,
                        total_fees DOUBLE PRECISION,
                        net_cash_impact DOUBLE PRECISION,
                        native_price DOUBLE PRECISION,
                        native_value DOUBLE PRECISION,
                        native_reference_price DOUBLE PRECISION,
                        native_gross_value DOUBLE PRECISION,
                        native_slippage_amount DOUBLE PRECISION,
                        native_commission DOUBLE PRECISION,
                        native_stamp_duty DOUBLE PRECISION,
                        native_transfer_fee DOUBLE PRECISION,
                        native_total_fees DOUBLE PRECISION,
                        native_net_cash_impact DOUBLE PRECISION,
                        fx_rate DOUBLE PRECISION,
                        market_rule_date TEXT,
                        market_rule_suspended BOOLEAN,
                        market_rule_closing_limit_state TEXT,
                        market_rule_official_close DOUBLE PRECISION,
                        market_rule_closing_gate_effective BOOLEAN,
                        created_at TEXT NOT NULL {created_at_default}
                    )
                    """
                )

                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS backtest_decisions (
                        id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                        run_id TEXT NOT NULL
                            REFERENCES agent_runs(run_id) ON DELETE CASCADE,
                        step_index INTEGER NOT NULL,
                        timestamp TEXT NOT NULL,
                        decision_source TEXT NOT NULL,
                        actions_submitted TEXT,
                        actions_executed INTEGER DEFAULT 0,
                        context_ref TEXT,
                        actions_trace_ref TEXT,
                        created_at TEXT NOT NULL {created_at_default}
                    )
                    """
                )

                # No FK: deliberate. The SQLite side has none either, and its
                # rows are deleted explicitly (see delete_run/clear_all).
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS run_manifest (
                        run_id TEXT PRIMARY KEY,
                        manifest_json TEXT NOT NULL,
                        created_at TEXT NOT NULL {created_at_default}
                    )
                    """
                )

                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_runs_session "
                    "ON agent_runs(session_id)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_runs_session_mode "
                    "ON agent_runs(session_id, mode)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_run_timestamp "
                    "ON equity_timeseries(run_id, timestamp)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_trades_run "
                    "ON trades(run_id, timestamp)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_decisions_run "
                    "ON backtest_decisions(run_id, step_index)"
                )

                # ADDING A COLUMN LATER? It must go in an `ALTER TABLE ... ADD COLUMN IF
                # NOT EXISTS` below, *not* only in the CREATE above. CREATE TABLE IF NOT
                # EXISTS silently no-ops once the table exists, so an existing deployment
                # would never gain the column, and every query naming it would raise
                # UndefinedColumn -- 500ing this whole surface while /health stays green.
                # See domain/agents/repository_postgres.py for the full worked example.
                cur.execute(
                    "ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS "
                    "session_id TEXT DEFAULT 'legacy-demo-session'"
                )
                cur.execute(
                    "ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS "
                    "llm_model TEXT DEFAULT 'rule-based'"
                )
                cur.execute(
                    "ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS "
                    "baseline_djia_run_id TEXT"
                )
                cur.execute(
                    "ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS "
                    "baseline_buyhold_run_id TEXT"
                )
                cur.execute(
                    "ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS "
                    "llm_calls INTEGER DEFAULT 0"
                )
                cur.execute(
                    "ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS "
                    "input_tokens INTEGER DEFAULT 0"
                )
                cur.execute(
                    "ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS "
                    "output_tokens INTEGER DEFAULT 0"
                )
                cur.execute(
                    "ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS "
                    "est_cost_usd DOUBLE PRECISION DEFAULT 0"
                )
                cur.execute(
                    "ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS metadata TEXT"
                )

                cur.execute(
                    "ALTER TABLE backtest_decisions ADD COLUMN IF NOT EXISTS "
                    "context_ref TEXT"
                )
                cur.execute(
                    "ALTER TABLE backtest_decisions ADD COLUMN IF NOT EXISTS "
                    "actions_trace_ref TEXT"
                )

                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS quantity INTEGER"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS side TEXT"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS "
                    "value DOUBLE PRECISION"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS reason TEXT"
                )

                cur.execute(
                    "ALTER TABLE equity_timeseries ADD COLUMN IF NOT EXISTS "
                    "native_equity DOUBLE PRECISION"
                )
                cur.execute(
                    "ALTER TABLE equity_timeseries ADD COLUMN IF NOT EXISTS "
                    "native_cash DOUBLE PRECISION"
                )
                cur.execute(
                    "ALTER TABLE equity_timeseries ADD COLUMN IF NOT EXISTS "
                    "native_positions_value DOUBLE PRECISION"
                )
                cur.execute(
                    "ALTER TABLE equity_timeseries ADD COLUMN IF NOT EXISTS "
                    "fx_rate DOUBLE PRECISION"
                )

                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS "
                    "reference_price DOUBLE PRECISION"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS "
                    "gross_value DOUBLE PRECISION"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS "
                    "slippage_amount DOUBLE PRECISION"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS "
                    "commission DOUBLE PRECISION"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS "
                    "stamp_duty DOUBLE PRECISION"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS "
                    "transfer_fee DOUBLE PRECISION"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS "
                    "total_fees DOUBLE PRECISION"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS "
                    "net_cash_impact DOUBLE PRECISION"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS "
                    "native_price DOUBLE PRECISION"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS "
                    "native_value DOUBLE PRECISION"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS "
                    "native_reference_price DOUBLE PRECISION"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS "
                    "native_gross_value DOUBLE PRECISION"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS "
                    "native_slippage_amount DOUBLE PRECISION"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS "
                    "native_commission DOUBLE PRECISION"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS "
                    "native_stamp_duty DOUBLE PRECISION"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS "
                    "native_transfer_fee DOUBLE PRECISION"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS "
                    "native_total_fees DOUBLE PRECISION"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS "
                    "native_net_cash_impact DOUBLE PRECISION"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS "
                    "fx_rate DOUBLE PRECISION"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS "
                    "market_rule_date TEXT"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS "
                    "market_rule_suspended BOOLEAN"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS "
                    "market_rule_closing_limit_state TEXT"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS "
                    "market_rule_official_close DOUBLE PRECISION"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS "
                    "market_rule_closing_gate_effective BOOLEAN"
                )

                # Postgres counterpart of SQLite's
                # _ensure_equity_timeseries_uniqueness/_apply_equity_timeseries_uniqueness:
                # the natural key that makes a rerun replace rather than duplicate, and
                # what ON CONFLICT (run_id, timestamp) needs to exist at all (Task 4/5).
                #
                # NOTHING STATIC GUARDS THIS LINE. The lazy-migration guard
                # (test_postgres_twin_repeats_every_sqlite_lazy_migration) regex-matches
                # only `ALTER TABLE ... ADD COLUMN` strings, and the SQLite original
                # protects this key with a unique *index*, not a column -- so its
                # omission here would be invisible to every static check. Only a live
                # Postgres upsert test (Task 9, @pg_only) would catch a missing/dropped
                # copy of this statement.
                cur.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS "
                    "uq_equity_timeseries_run_timestamp "
                    "ON equity_timeseries (run_id, timestamp)"
                )

    def get_idempotency(self, run_id: str, step_index: int, idem_key: str) -> Optional[Dict[str, Any]]:
        return self._sqlite.get_idempotency(run_id, step_index, idem_key)

    def put_idempotency(self, run_id: str, step_index: int, idem_key: str, ack: Dict[str, Any]) -> None:
        self._sqlite.put_idempotency(run_id, step_index, idem_key, ack)

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------

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
        """Insert or refresh a backtest run.

        Carries divergences 1-3 from the module docstring, all of them
        consequences of the same thing: SQLite's ``INSERT OR REPLACE`` is a
        DELETE+INSERT, so every column its VALUES list omits goes back to its
        default, while ``ON CONFLICT DO UPDATE`` only touches the columns named
        in ``DO UPDATE SET``. Three columns fall in that gap.

        ``created_at`` is deliberately left out of ``DO UPDATE SET``, so a
        re-insert preserves it (first-seen semantics) where SQLite resets it to
        the moment of the rerun. **This one is user-visible, not cosmetic.**
        Eight backend call sites read it for ordering: the four listing queries
        below (``get_all_runs``, ``get_runs_by_session``,
        ``get_runs_by_sessions``, ``get_runs_by_mode``) order by
        ``created_at DESC``, and ``domain/agents/service.py`` sorts by it in
        four more to pick "the latest run for this agent".

        The frontend depends on it *more strongly than ordering*, which is the
        real reason first-seen is the right choice here rather than a matter of
        taste: ``dashboard/frontend/app.js`` pairs an external run with its
        baseline runs by a ``created_at`` **range** --
        ``r.created_at >= extRun.created_at && r.created_at < nextExtCreated``.
        Resetting the column therefore does not merely reorder a list, it can
        move a run across that boundary and mis-associate whole baseline
        curves. First-seen is also what the column *name* means, and what the
        backfill's ``_restore_created_at`` relies on to stay idempotent; a
        rerun recomputes an old run, it does not create a new one.

        (An earlier revision of this docstring asserted "nothing reads that
        reset as meaningful". That was simply wrong in both directions -- eight
        backend sites order by it and the frontend range-matches on it.)

        ``updated_at`` is the opposite case: it *is* named in
        ``DO UPDATE SET``, so a ``force_refresh`` still shows as "just
        updated", matching what SQLite's REPLACE incidentally achieves.
        Omitting it would freeze ``updated_at`` at the original insert --
        diverging in the other direction, and lying about this row's write
        history.

        ``baseline_djia_run_id``/``baseline_buyhold_run_id`` appear in neither
        the ``INSERT`` column list nor ``DO UPDATE SET``, so a re-insert leaves
        the existing links intact where SQLite's REPLACE nulls them. Kept
        deliberately rather than "fixed" to match: the only production writers
        (``domain/backtesting/baseline_worker.py`` and
        ``scripts/backtest_hourly_agent.py``) call ``update_run_baselines``
        immediately after ``insert_run``, so in the happy path both stores end
        up identical. They differ only when baseline regeneration *fails* --
        and ``baseline_worker`` swallows its own exception ("⚠️ Baseline
        generation failed (run saved)"), so on SQLite that permanently orphans
        the run's comparison chart while here the previous links survive.
        Nulling them here would be importing a defect into the durable store.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO agent_runs
                    (run_id, session_id, agent_name, mode, start_date, end_date,
                     initial_equity, final_equity, total_return, sharpe_ratio,
                     max_drawdown, num_trades, llm_model,
                     llm_calls, input_tokens, output_tokens, est_cost_usd, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id) DO UPDATE SET
                        session_id = EXCLUDED.session_id,
                        agent_name = EXCLUDED.agent_name,
                        mode = EXCLUDED.mode,
                        start_date = EXCLUDED.start_date,
                        end_date = EXCLUDED.end_date,
                        initial_equity = EXCLUDED.initial_equity,
                        final_equity = EXCLUDED.final_equity,
                        total_return = EXCLUDED.total_return,
                        sharpe_ratio = EXCLUDED.sharpe_ratio,
                        max_drawdown = EXCLUDED.max_drawdown,
                        num_trades = EXCLUDED.num_trades,
                        llm_model = EXCLUDED.llm_model,
                        llm_calls = EXCLUDED.llm_calls,
                        input_tokens = EXCLUDED.input_tokens,
                        output_tokens = EXCLUDED.output_tokens,
                        est_cost_usd = EXCLUDED.est_cost_usd,
                        metadata = EXCLUDED.metadata,
                        updated_at = to_char(now() AT TIME ZONE 'utc', 'YYYY-MM-DD HH24:MI:SS')
                    """,
                    (
                        run_id, session_id, agent_name, mode, start_date, end_date,
                        initial_equity, final_equity, total_return, sharpe_ratio,
                        max_drawdown, num_trades, llm_model,
                        llm_calls, input_tokens, output_tokens, est_cost_usd,
                        json.dumps(metadata) if metadata is not None else None,
                    ),
                )

    def update_run_baselines(
        self,
        run_id: str,
        *,
        djia_run_id: Optional[str] = None,
        buyhold_run_id: Optional[str] = None,
    ) -> None:
        """Link an external backtest run to its paired baseline runs."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE agent_runs
                    SET baseline_djia_run_id = COALESCE(%s, baseline_djia_run_id),
                        baseline_buyhold_run_id = COALESCE(%s, baseline_buyhold_run_id),
                        updated_at = to_char(now() AT TIME ZONE 'utc', 'YYYY-MM-DD HH24:MI:SS')
                    WHERE run_id = %s
                    """,
                    (djia_run_id, buyhold_run_id, run_id),
                )

    def insert_equity_point(self, run_id: str, timestamp: str,
                          equity: float, cash: float,
                          positions_value: float,
                          daily_return: Optional[float] = None,
                          native_equity: Optional[float] = None,
                          native_cash: Optional[float] = None,
                          native_positions_value: Optional[float] = None,
                          fx_rate: Optional[float] = None) -> None:
        """Insert a single equity data point."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO equity_timeseries
                    (run_id, timestamp, equity, cash, positions_value, daily_return,
                     native_equity, native_cash, native_positions_value, fx_rate)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id, timestamp) DO UPDATE SET
                        equity = EXCLUDED.equity,
                        cash = EXCLUDED.cash,
                        positions_value = EXCLUDED.positions_value,
                        daily_return = EXCLUDED.daily_return,
                        native_equity = EXCLUDED.native_equity,
                        native_cash = EXCLUDED.native_cash,
                        native_positions_value = EXCLUDED.native_positions_value,
                        fx_rate = EXCLUDED.fx_rate
                    """,
                    (
                        run_id, as_timestamp_text(timestamp), equity, cash,
                        positions_value, daily_return,
                        native_equity, native_cash, native_positions_value, fx_rate,
                    ),
                )

    def insert_equity_points(self, run_id: str,
                           points: List[Dict[str, Any]],
                           replace: bool = True) -> None:
        """Replace this run's equity curve with ``points``, atomically.

        Each point should have: timestamp, equity, cash, positions_value, [daily_return]

        Every production caller hands over a *whole* curve, and a rerun can
        legitimately produce a different set of timestamps (fewer bars, a
        partial run, a changed symbol list). The (run_id, timestamp) unique key
        only collapses timestamps that repeat, so without the delete the
        leftovers of the previous curve stay spliced into the new one -- which
        is exactly the force-refresh case. Pass ``replace=False`` to append to
        an existing curve instead. An empty ``points`` list is a no-op rather
        than a wipe: a failed rerun must not erase the curve on the board --
        and, since it short-circuits before opening a connection, it never
        pays a pool checkout either.
        """
        if not points:
            return

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if replace:
                    cur.execute(
                        "DELETE FROM equity_timeseries WHERE run_id = %s", (run_id,)
                    )
                cur.executemany(
                    """
                    INSERT INTO equity_timeseries
                    (run_id, timestamp, equity, cash, positions_value, daily_return,
                     native_equity, native_cash, native_positions_value, fx_rate)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id, timestamp) DO UPDATE SET
                        equity = EXCLUDED.equity,
                        cash = EXCLUDED.cash,
                        positions_value = EXCLUDED.positions_value,
                        daily_return = EXCLUDED.daily_return,
                        native_equity = EXCLUDED.native_equity,
                        native_cash = EXCLUDED.native_cash,
                        native_positions_value = EXCLUDED.native_positions_value,
                        fx_rate = EXCLUDED.fx_rate
                    """,
                    [
                        (
                            run_id,
                            as_timestamp_text(point["timestamp"]),
                            point["equity"],
                            point["cash"],
                            point["positions_value"],
                            point.get("daily_return"),
                            point.get("native_equity"),
                            point.get("native_cash"),
                            point.get("native_positions_value"),
                            point.get("fx_rate"),
                        )
                        for point in points
                    ],
                )

    def insert_trades(self, run_id: str, trades: List[Dict[str, Any]]) -> None:
        """Batch insert trade records for a backtest run.

        Unlike the SQLite side, the Postgres ``trades`` table never carried
        the legacy ``shares``/``action``/``total_value`` columns, so there is
        no introspection branch here -- the modern column set is written
        directly.
        """
        if not trades:
            return

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for trade in trades:
                    ts = trade.get("timestamp")
                    if hasattr(ts, "isoformat"):
                        ts = ts.isoformat()
                    side = str(trade.get("side", "")).upper()
                    qty = int(trade.get("shares") or trade.get("quantity") or 0)
                    price = float(trade.get("price") or 0)
                    value = float(
                        trade.get("cost") or trade.get("proceeds")
                        or trade.get("value") or qty * price
                    )
                    cur.execute(
                        """
                        INSERT INTO trades
                        (run_id, timestamp, symbol, quantity, side, price, value,
                         reason, reference_price, gross_value, slippage_amount,
                         commission, stamp_duty, transfer_fee, total_fees,
                         net_cash_impact, native_price, native_value,
                         native_reference_price, native_gross_value,
                         native_slippage_amount, native_commission,
                         native_stamp_duty, native_transfer_fee,
                         native_total_fees, native_net_cash_impact, fx_rate,
                         market_rule_date, market_rule_suspended,
                         market_rule_closing_limit_state,
                         market_rule_official_close,
                         market_rule_closing_gate_effective)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s)
                        """,
                        (
                            run_id,
                            str(ts),
                            trade.get("symbol"),
                            qty,
                            side,
                            price,
                            value,
                            trade.get("reason"),
                            trade.get("reference_price"),
                            trade.get("gross_value"),
                            trade.get("slippage_amount"),
                            trade.get("commission"),
                            trade.get("stamp_duty"),
                            trade.get("transfer_fee"),
                            trade.get("total_fees"),
                            trade.get("net_cash_impact"),
                            trade.get("native_price"),
                            trade.get("native_value"),
                            trade.get("native_reference_price"),
                            trade.get("native_gross_value"),
                            trade.get("native_slippage_amount"),
                            trade.get("native_commission"),
                            trade.get("native_stamp_duty"),
                            trade.get("native_transfer_fee"),
                            trade.get("native_total_fees"),
                            trade.get("native_net_cash_impact"),
                            trade.get("fx_rate"),
                            trade.get("market_rule_date"),
                            trade.get("market_rule_suspended"),
                            trade.get("market_rule_closing_limit_state"),
                            trade.get("market_rule_official_close"),
                            trade.get("market_rule_closing_gate_effective"),
                        ),
                    )

    def insert_decisions(self, run_id: str, decisions: List[Dict[str, Any]]) -> None:
        """Batch insert hourly decision log entries."""
        if not decisions:
            return

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO backtest_decisions
                    (run_id, step_index, timestamp, decision_source, actions_submitted,
                     actions_executed, context_ref)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            run_id,
                            entry.get("step_index", 0),
                            as_timestamp_text(entry.get("timestamp")),
                            entry.get("decision_source"),
                            json.dumps(entry.get("actions_submitted") or []),
                            entry.get("actions_executed", 0),
                            entry.get("context_ref"),
                        )
                        for entry in decisions
                    ],
                )

    def insert_run_manifest(self, run_id: str, manifest: Dict[str, Any]) -> None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO run_manifest (run_id, manifest_json)
                    VALUES (%s, %s)
                    ON CONFLICT (run_id) DO UPDATE SET
                        manifest_json = EXCLUDED.manifest_json
                    """,
                    (run_id, json.dumps(manifest)),
                )

    def delete_run(self, run_id: str) -> None:
        """Delete a run and all its data.

        ``equity_timeseries``, ``trades`` and ``backtest_decisions`` carry
        ``ON DELETE CASCADE`` back to ``agent_runs`` (see ``_init_schema``),
        so only ``run_manifest`` -- which has no FK -- needs an explicit
        delete alongside the ``agent_runs`` row itself.

        Deliberately does NOT touch ``idempotency_keys``: SQLite's
        ``delete_run`` leaves those rows orphaned too, and the mirror suite
        (Task 9) asserts both backends behave identically. Fixing that
        orphan here would be a divergence, not a parity port.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM run_manifest WHERE run_id = %s", (run_id,))
                cur.execute("DELETE FROM agent_runs WHERE run_id = %s", (run_id,))

    def clear_all(self) -> None:
        """Clear all data (useful for testing).

        Truncates all five Postgres tables and nothing else. In particular it
        does **not** delegate to ``self._sqlite.clear_all()``, which an earlier
        cut of this file did: that call could only ever do harm. The embedded
        ``BacktestDatabase`` exists for exactly one purpose -- the
        ``idempotency_keys`` "hot half" -- and ``clear_all`` deliberately
        leaves that table alone (see below), so every DELETE the delegation
        issued landed on cold tables this object never reads. What it *could*
        reach, since ``self._sqlite`` defaults to ``DATABASE_PATH`` and that
        defaults to the committed seed, is the seed's own ``agent_runs`` --
        **all 17 rows**, i.e. every run prod has. That is the 12 ``lb_*`` rows
        behind the leaderboard's 7 ``auto_compute: false`` entries, which
        nothing regenerates, *and* the 3 runs ``config/defaults.json`` names as
        the dashboard's default comparison (see
        ``tests/test_seed_database_integrity.py``). A local
        ``backtest_hourly_agent.py --clear`` with ``AGENT_RUNS_DATABASE_URL``
        set would have silently blanked the board and the default view both.

        Deliberately does NOT touch ``idempotency_keys``, matching SQLite's
        ``clear_all`` (same parity reasoning as ``delete_run`` above).
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM equity_timeseries")
                cur.execute("DELETE FROM trades")
                cur.execute("DELETE FROM backtest_decisions")
                cur.execute("DELETE FROM run_manifest")
                cur.execute("DELETE FROM agent_runs")

    # ------------------------------------------------------------------
    # Readers
    # ------------------------------------------------------------------

    def get_all_runs(self) -> List[Dict]:
        """Get metadata for all runs."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM agent_runs ORDER BY created_at DESC")
                rows = cur.fetchall()
        return [BacktestDatabase._parse_run_row(row) for row in rows]

    def get_runs_by_session(self, session_id: str) -> List[Dict]:
        """Get all runs for a specific session."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM agent_runs
                    WHERE session_id = %s
                    ORDER BY created_at DESC
                    """,
                    (session_id,),
                )
                rows = cur.fetchall()
        return [BacktestDatabase._parse_run_row(row) for row in rows]

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
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM agent_runs
                    WHERE session_id = ANY(%s)
                    ORDER BY created_at DESC
                    """,
                    (list(grouped),),
                )
                rows = cur.fetchall()
        for row in rows:
            run = BacktestDatabase._parse_run_row(row)
            grouped[run["session_id"]].append(run)
        return grouped

    def get_run(self, run_id: str) -> Optional[Dict]:
        """Get metadata for a specific run (no session check)."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM agent_runs WHERE run_id = %s", (run_id,))
                row = cur.fetchone()
        return BacktestDatabase._parse_run_row(row) if row else None

    def get_run_with_session(self, run_id: str, session_id: str) -> Optional[Dict]:
        """Get a run, verifying it belongs to the session."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM agent_runs
                    WHERE run_id = %s AND session_id = %s
                    """,
                    (run_id, session_id),
                )
                row = cur.fetchone()
        return BacktestDatabase._parse_run_row(row) if row else None

    def get_equity_curve(self, run_id: str) -> List[Dict]:
        """Get full equity curve for a run."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT timestamp, equity, cash, positions_value, daily_return,
                           native_equity, native_cash, native_positions_value, fx_rate
                    FROM equity_timeseries
                    WHERE run_id = %s
                    ORDER BY timestamp ASC
                    """,
                    (run_id,),
                )
                rows = cur.fetchall()
        audit_fields = (
            "native_equity",
            "native_cash",
            "native_positions_value",
            "fx_rate",
        )
        for row in rows:
            for field in audit_fields:
                if row.get(field) is None:
                    row.pop(field, None)
        return rows

    def get_equity_curves(self, run_ids: List[str]) -> Dict[str, List[Dict]]:
        """Batched: one query for all runs (the SQLite twin loops; over the
        network that would be one round-trip per agent on the My Agents page).
        """
        result: Dict[str, List[Dict]] = {run_id: [] for run_id in run_ids}
        if not run_ids:
            return result
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT run_id, timestamp, equity, cash, positions_value, daily_return,
                           native_equity, native_cash, native_positions_value, fx_rate
                    FROM equity_timeseries
                    WHERE run_id = ANY(%s)
                    ORDER BY run_id, timestamp ASC
                    """,
                    (list(run_ids),),
                )
                rows = cur.fetchall()
        audit_fields = (
            "native_equity",
            "native_cash",
            "native_positions_value",
            "fx_rate",
        )
        for row in rows:
            run_id = row.pop("run_id")
            for field in audit_fields:
                if row.get(field) is None:
                    row.pop(field, None)
            result[run_id].append(row)
        return result

    def get_runs_by_mode(self, mode: str) -> List[Dict]:
        """Get all runs for a specific mode ('backtest' or 'paper')."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM agent_runs
                    WHERE mode = %s
                    ORDER BY created_at DESC
                    """,
                    (mode,),
                )
                rows = cur.fetchall()
        return [BacktestDatabase._parse_run_row(row) for row in rows]

    def get_trades(self, run_id: str) -> List[Dict]:
        """Get all trades for a run.

        Unlike the SQLite side, the Postgres ``trades`` table never carried the
        legacy ``shares``/``action``/``total_value`` columns (see
        ``insert_trades``), so there is no column-introspection branch here --
        the modern column set is selected directly.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT timestamp, symbol, quantity, side, price, value, reason,
                           reference_price, gross_value, slippage_amount,
                           commission, stamp_duty, transfer_fee, total_fees,
                           net_cash_impact, native_price, native_value,
                           native_reference_price, native_gross_value,
                           native_slippage_amount, native_commission,
                           native_stamp_duty, native_transfer_fee,
                           native_total_fees, native_net_cash_impact, fx_rate,
                           market_rule_date, market_rule_suspended,
                           market_rule_closing_limit_state,
                           market_rule_official_close,
                           market_rule_closing_gate_effective
                    FROM trades WHERE run_id = %s
                    ORDER BY timestamp ASC, id ASC
                    """,
                    (run_id,),
                )
                rows = cur.fetchall()
        for row in rows:
            for field in TRADE_OPTIONAL_AUDIT_FIELDS:
                if row.get(field) is None:
                    row.pop(field, None)
        return rows

    def get_decisions(self, run_id: str) -> List[Dict]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT step_index, timestamp, decision_source, actions_submitted,
                           actions_executed, context_ref
                    FROM backtest_decisions WHERE run_id = %s
                    ORDER BY step_index ASC
                    """,
                    (run_id,),
                )
                rows = cur.fetchall()
        for row in rows:
            try:
                row["actions_submitted"] = json.loads(row.get("actions_submitted") or "[]")
            except json.JSONDecodeError:
                row["actions_submitted"] = []
        return rows

    def get_run_manifest(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT manifest_json FROM run_manifest WHERE run_id = %s",
                    (run_id,),
                )
                row = cur.fetchone()
        return json.loads(row["manifest_json"]) if row else None
