"""Tests for the run-history backfill script (Task 11,
``dashboard/scripts/backfill_runs_to_postgres.py``).

Two tiers, mirroring test_backtest_db_postgres.py:
1. ``--dry-run`` / CLI-error tests -- no live Postgres needed, run for real in
   any sandbox.
2. The live-Postgres backfill + idempotency test -- @pg_only, skipped unless
   TEST_POSTGRES_URL is set (never runs in this sandbox; see
   GLOBAL-CONSTRAINTS.md). Verified in CI.

Three of the script's five tables (trades, backtest_decisions, run_manifest)
are EMPTY in the real committed seed database, so a real backfill run against
prod never exercises those code paths at all. The synthetic source built by
``_build_source_db`` below is therefore the only thing that ever does --
it deliberately puts rows in all five tables, not just the two prod happens
to have.
"""

import os
import sqlite3
import sys

import pytest

from dashboard.backend.database import BacktestDatabase
from dashboard.backend.tests._postgres_testing import require_local_postgres_url

TEST_POSTGRES_URL = os.getenv("TEST_POSTGRES_URL")

pg_only = pytest.mark.skipif(
    not TEST_POSTGRES_URL,
    reason="TEST_POSTGRES_URL not set; skipping live-Postgres tests",
)


def _build_source_db(tmp_path):
    """A small SQLite BacktestDatabase, built against a tmp_path file (never
    the committed seed), with rows in all five tables: two runs -- "run-full"
    carries equity/trades/decisions/manifest and a baseline link to
    "run-baseline", which carries only a minimal equity point (a plausible
    baseline-run shape, same as the real DJIA/buy-hold pairing).
    """
    source_path = tmp_path / "source.db"
    db = BacktestDatabase(db_path=source_path)

    db.insert_run(
        run_id="run-baseline", session_id="s1", agent_name="DJIA baseline", mode="backtest",
        start_date="2024-01-01", end_date="2024-01-02", initial_equity=1000.0,
        final_equity=1010.0,
    )
    db.insert_equity_points(
        "run-baseline",
        [{"timestamp": "2024-01-01T00:00:00", "equity": 1000.0, "cash": 1000.0, "positions_value": 0.0}],
    )

    db.insert_run(
        run_id="run-full", session_id="s1", agent_name="Full Agent", mode="backtest",
        start_date="2024-01-01", end_date="2024-01-02", initial_equity=1000.0,
        final_equity=1050.0, total_return=0.05, sharpe_ratio=1.1, max_drawdown=-0.02,
        num_trades=2, llm_model="claude-sonnet", llm_calls=4,
        input_tokens=500, output_tokens=100, est_cost_usd=0.12,
        metadata={"llm_max_output_tokens": 4096},
    )
    db.update_run_baselines("run-full", djia_run_id="run-baseline")
    db.insert_equity_points(
        "run-full",
        [
            {"timestamp": "2024-01-01T00:00:00", "equity": 1000.0, "cash": 1000.0, "positions_value": 0.0},
            {"timestamp": "2024-01-01T01:00:00", "equity": 1050.0, "cash": 950.0, "positions_value": 100.0},
        ],
    )
    db.insert_trades(
        "run-full",
        [
            {
                "timestamp": "2024-01-01T00:30:00", "symbol": "AAPL", "quantity": 1,
                "side": "buy", "price": 100.0, "value": 100.0, "reason": "signal",
            },
            {
                # legacy shares/cost alias shape -- BacktestDatabase.insert_trades
                # normalizes it to quantity/value at SOURCE insert time, same as
                # every other trades writer in the app; the backfill script's own
                # read/write never sees the alias keys at all.
                "timestamp": "2024-01-01T01:00:00", "symbol": "AAPL", "shares": 1,
                "side": "sell", "price": 105.0, "cost": 105.0,
            },
        ],
    )
    db.insert_decisions(
        "run-full",
        [
            {
                "step_index": 0, "timestamp": "2024-01-01T00:00:00", "decision_source": "llm",
                "actions_submitted": [{"action": "buy", "symbol": "AAPL"}],
                "actions_executed": 1, "context_ref": "ctx-1",
            },
            {
                "step_index": 1, "timestamp": "2024-01-01T01:00:00", "decision_source": "llm",
                "actions_submitted": [{"action": "sell", "symbol": "AAPL"}],
                "actions_executed": 1,
            },
        ],
    )
    db.insert_run_manifest("run-full", {"symbols": ["AAPL"], "version": 1})

    return source_path


# --- CLI-level tests: no live Postgres needed, run for real ------------------

def test_dry_run_reports_source_counts_without_writing(tmp_path, monkeypatch, capsys):
    """Exercises the script's non-Postgres path for real: --dry-run must read
    the source and report every table's count without requiring
    AGENT_RUNS_DATABASE_URL, and without attempting any write.
    """
    source_path = _build_source_db(tmp_path)

    from dashboard.scripts import backfill_runs_to_postgres

    monkeypatch.delenv("AGENT_RUNS_DATABASE_URL", raising=False)
    monkeypatch.setattr(
        sys, "argv",
        ["backfill_runs_to_postgres.py", "--source", str(source_path), "--dry-run"],
    )

    exit_code = backfill_runs_to_postgres.main()
    assert exit_code == 0

    out = capsys.readouterr().out
    assert "agent_runs: 2" in out
    assert "equity_timeseries: 3" in out
    assert "trades: 2" in out
    assert "backtest_decisions: 2" in out
    assert "run_manifest: 1" in out
    assert "AGENT_RUNS_DATABASE_URL is not set" in out
    assert "Dry run: no writes performed." in out


def test_dry_run_with_target_set_previews_but_does_not_connect(tmp_path, monkeypatch, capsys):
    """A dry run with AGENT_RUNS_DATABASE_URL set must still short-circuit
    before ever constructing PostgresBacktestDatabase (an unreachable fake URL
    here would raise on connect if the script tried) -- and must never print
    the credential embedded in it.
    """
    source_path = _build_source_db(tmp_path)

    from dashboard.scripts import backfill_runs_to_postgres

    monkeypatch.setenv(
        "AGENT_RUNS_DATABASE_URL", "postgresql://admin:sup3r-s3cret@example.invalid/atl"
    )
    monkeypatch.setattr(
        sys, "argv",
        ["backfill_runs_to_postgres.py", "--source", str(source_path), "--dry-run"],
    )

    exit_code = backfill_runs_to_postgres.main()
    assert exit_code == 0

    out = capsys.readouterr().out
    assert "sup3r-s3cret" not in out
    assert "example.invalid/atl" in out
    assert "Dry run: no writes performed." in out


def test_missing_source_file_fails_loudly(tmp_path, monkeypatch, capsys):
    from dashboard.scripts import backfill_runs_to_postgres

    missing = tmp_path / "does-not-exist.db"
    monkeypatch.setattr(
        sys, "argv",
        ["backfill_runs_to_postgres.py", "--source", str(missing)],
    )

    exit_code = backfill_runs_to_postgres.main()
    assert exit_code == 1
    assert "source database not found" in capsys.readouterr().err


# --- live-Postgres backfill + idempotency -------------------------------------

@pytest.fixture
def pg_backtest_db():
    require_local_postgres_url(TEST_POSTGRES_URL)
    from dashboard.backend.database_postgres import PostgresBacktestDatabase

    store = PostgresBacktestDatabase(TEST_POSTGRES_URL)
    with store._get_connection() as conn:
        with conn.cursor() as cur:
            # children first, then parents -- the FKs are enforced here
            cur.execute("DELETE FROM equity_timeseries")
            cur.execute("DELETE FROM trades")
            cur.execute("DELETE FROM backtest_decisions")
            cur.execute("DELETE FROM run_manifest")
            cur.execute("DELETE FROM agent_runs")
    yield store


# Distinctive, obviously-not-"now" values so a false pass (the test happening
# to run at a matching wall-clock time) is not physically possible for either.
BACKDATED_CREATED_AT = "2019-06-15 12:00:00"
BACKDATED_UPDATED_AT = "2019-06-20 08:00:00"


@pg_only
def test_backfill_migrates_all_five_tables_and_is_idempotent_on_rerun(
    tmp_path, monkeypatch, capsys, pg_backtest_db
):
    source_path = _build_source_db(tmp_path)

    # Back-date run-full's created_at/updated_at in the SOURCE (raw UPDATE,
    # since insert_run has no timestamp params) to two distinct old values,
    # before the backfill ever reads/copies this file. Proves two things at
    # once: created_at survives the migration (restored from source), and
    # updated_at deliberately does NOT (it must NOT come back as
    # BACKDATED_UPDATED_AT -- see backfill_runs_to_postgres.py's module
    # docstring for why only created_at is restored).
    source_conn = sqlite3.connect(str(source_path))
    try:
        source_conn.execute(
            "UPDATE agent_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            (BACKDATED_CREATED_AT, BACKDATED_UPDATED_AT, "run-full"),
        )
        source_conn.commit()
    finally:
        source_conn.close()

    from dashboard.scripts import backfill_runs_to_postgres

    monkeypatch.setenv("AGENT_RUNS_DATABASE_URL", TEST_POSTGRES_URL)
    monkeypatch.setattr(
        sys, "argv",
        ["backfill_runs_to_postgres.py", "--source", str(source_path)],
    )

    # Real source counts, re-derived here via a plain independent connection
    # (not the script's own _copy_source/_table_count) rather than hardcoded
    # -- so this test still catches a divergence if _build_source_db above
    # ever changes, without just re-running the code under test on itself.
    source_conn = sqlite3.connect(str(source_path))
    try:
        source_counts = {
            table: source_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in backfill_runs_to_postgres.TABLES_IN_FK_ORDER
        }
    finally:
        source_conn.close()
    assert source_counts == {
        "agent_runs": 2, "equity_timeseries": 3, "trades": 2,
        "backtest_decisions": 2, "run_manifest": 1,
    }

    def _live_counts():
        counts = {"agent_runs": len(pg_backtest_db.get_all_runs())}
        with pg_backtest_db._get_connection() as conn:
            with conn.cursor() as cur:
                for table in ("equity_timeseries", "trades", "backtest_decisions", "run_manifest"):
                    cur.execute(f"SELECT COUNT(*) AS n FROM {table}")
                    counts[table] = cur.fetchone()["n"]
        return counts

    exit_code = backfill_runs_to_postgres.main()
    assert exit_code == 0
    assert "Backfill complete." in capsys.readouterr().out
    assert _live_counts() == source_counts

    # --- spot-check one full run: its row, its curve, its trades -----------
    run = pg_backtest_db.get_run("run-full")
    assert run["agent_name"] == "Full Agent"
    assert run["mode"] == "backtest"
    assert run["final_equity"] == 1050.0
    assert run["baseline_djia_run_id"] == "run-baseline"
    assert run["metadata"] == {"llm_max_output_tokens": 4096}
    # created_at restored from source; updated_at deliberately left as the
    # twin's own backfill-time stamp, not copied from the source's value.
    assert run["created_at"] == BACKDATED_CREATED_AT
    assert run["updated_at"] != BACKDATED_UPDATED_AT

    curve = pg_backtest_db.get_equity_curve("run-full")
    assert [c["timestamp"] for c in curve] == ["2024-01-01T00:00:00", "2024-01-01T01:00:00"]
    assert curve[1]["equity"] == 1050.0
    assert curve[1]["cash"] == 950.0

    trades = pg_backtest_db.get_trades("run-full")
    assert len(trades) == 2
    assert trades[0]["side"] == "BUY"
    assert trades[0]["reason"] == "signal"
    assert trades[1]["side"] == "SELL"
    assert trades[1]["quantity"] == 1
    assert trades[1]["value"] == 105.0

    decisions = pg_backtest_db.get_decisions("run-full")
    assert [d["step_index"] for d in decisions] == [0, 1]
    assert decisions[0]["actions_submitted"] == [{"action": "buy", "symbol": "AAPL"}]
    assert decisions[0]["context_ref"] == "ctx-1"

    manifest = pg_backtest_db.get_run_manifest("run-full")
    assert manifest == {"symbols": ["AAPL"], "version": 1}

    assert pg_backtest_db.get_run("run-baseline")["final_equity"] == 1010.0

    # --- idempotency: this is the point of the test -------------------------
    # Re-run against the same source and target; every count must be
    # unchanged (agent_runs/equity_timeseries/run_manifest upsert in place,
    # trades/backtest_decisions are skipped once the target already has them
    # -- see backfill_runs_to_postgres.py's module docstring).
    exit_code_2 = backfill_runs_to_postgres.main()
    assert exit_code_2 == 0
    assert _live_counts() == source_counts

    trades_after = pg_backtest_db.get_trades("run-full")
    assert len(trades_after) == 2  # not duplicated
    decisions_after = pg_backtest_db.get_decisions("run-full")
    assert len(decisions_after) == 2  # not duplicated

    # The point of this test: created_at must still match after a second run
    # -- proving the restore converges rather than drifting (it always writes
    # the same source-captured value, never "now()").
    run_after = pg_backtest_db.get_run("run-full")
    assert run_after["created_at"] == BACKDATED_CREATED_AT
    assert run_after["updated_at"] != BACKDATED_UPDATED_AT
