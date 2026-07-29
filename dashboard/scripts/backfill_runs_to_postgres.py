#!/usr/bin/env python3
"""One-time backfill: copy backtest run history from the local SQLite store into
the Postgres run-history twin selected by ``AGENT_RUNS_DATABASE_URL``.

Why this exists
----------------
``PostgresBacktestDatabase`` (``database_postgres.py``) starts with five empty
tables the day ``AGENT_RUNS_DATABASE_URL`` is first set in prod -- switching the
backend does not carry over history that was already sitting in the committed
SQLite seed (``dashboard/storage/data/backtest.db``). Until this script runs,
prod's ``/runs`` listing is empty even though the seed file it replaced has real
history. Run it once, immediately after the first green deploy with
``AGENT_RUNS_DATABASE_URL`` set:

    python dashboard/scripts/backfill_runs_to_postgres.py --dry-run   # preview
    python dashboard/scripts/backfill_runs_to_postgres.py             # for real

Idempotent by design, so a second run (after a partial failure, or by mistake)
is safe:
  * ``agent_runs`` / ``equity_timeseries`` / ``run_manifest`` are written through
    the twin's own upsert methods (``insert_run``, ``insert_equity_points``,
    ``insert_run_manifest``), which already dedupe on every call -- see
    ``database_postgres.py``'s ``ON CONFLICT`` clauses.
  * ``trades`` / ``backtest_decisions`` do **not** dedupe on the twin side:
    ``insert_trades`` / ``insert_decisions`` are plain appends -- there is no
    natural unique key for a trade or a decision row, so re-running them
    verbatim would duplicate every row. This script adds its own coarse
    idempotency for those two tables only: before writing a run's trades (or
    decisions) it reads them back through the twin's own ``get_trades`` /
    ``get_decisions`` and skips the run entirely if anything is already there.
    That is sufficient for a one-time backfill of closed, historical runs
    nothing else writes to concurrently -- it is not a general dedup
    mechanism, and does not attempt to reconcile a partially-written run one
    row at a time.

Two traps this script is built around
--------------------------------------
1. ``DATABASE_PATH`` (and, for the same reason, ``AGENT_RUNS_DATABASE_URL``)
   must be neutralised *before* the first ``dashboard.backend`` import.
   Importing ``dashboard.backend.database`` builds its module-level singleton
   (``db = _build_backtest_db()``) as an unavoidable side effect, and
   ``PostgresBacktestDatabase.__init__`` embeds a plain ``BacktestDatabase()``
   for its ``idempotency_keys`` "hot half" -- both default to ``DATABASE_PATH``,
   which itself defaults to the committed seed file. Without neutralising both
   env vars first, merely importing this script would run lazy-migration DDL
   against the real seed, and/or open a real Postgres connection, before
   ``main()`` -- let alone ``--dry-run`` -- ever runs. Exactly the mechanism
   ``dashboard/backend/tests/conftest.py`` uses, extended to the run-history
   var it also strips. ``AGENT_RUNS_DATABASE_URL`` is restored right after the
   import (see the comment at the import site) so ``main()`` can still read it
   normally, and so a test's ``monkeypatch.setenv`` -- applied after this
   module is already imported -- is what wins.
2. The ``--source`` file is never opened as a live, writable connection, and
   never read with a hand-rolled query against its raw columns either -- see
   ``_copy_source`` and ``SourceData`` for why (a first cut of this script did
   read raw columns directly and crashed on the real committed seed, which
   predates the currency-audit migration; see task-11-report.md).
"""

from __future__ import annotations

import argparse
import atexit
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

# --- Trap 1a: DATABASE_PATH, forced before any dashboard.backend import ----
# A throwaway per-process path: nothing durable is ever written here. It only
# exists so database.py's module-level singleton and the Postgres twin's
# embedded BacktestDatabase() have somewhere harmless to point instead of the
# committed seed. Force (not setdefault) so an ambient DATABASE_PATH in the
# operator's shell can't leak through either -- this script never reads the
# real DATABASE_PATH for anything; --source is read via its own copy (below).
_SCRATCH_DIR = tempfile.mkdtemp(prefix="atl_backfill_scratch_")
os.environ["DATABASE_PATH"] = os.path.join(_SCRATCH_DIR, "scratch.db")
atexit.register(lambda: shutil.rmtree(_SCRATCH_DIR, ignore_errors=True))

# Bootstrap for direct-file execution (``python dashboard/scripts/backfill_runs_to_postgres.py``).
# When imported as ``dashboard.scripts.backfill_runs_to_postgres`` (e.g. by
# test_backfill_runs.py under pytest) the repo root is already importable and
# __package__ is truthy, so this is skipped -- see backtest_hourly_agent.py for
# the same pattern and its docstring for the full rationale.
if not __package__:
    from _bootstrap import ensure_repo_root

    ensure_repo_root()

from dotenv import load_dotenv  # noqa: E402

DASHBOARD_DIR = Path(__file__).resolve().parent.parent
load_dotenv(DASHBOARD_DIR / ".env")
load_dotenv(DASHBOARD_DIR.parent / ".env")

# --- Trap 1b: AGENT_RUNS_DATABASE_URL, hidden across the backend import -----
# Same reasoning as DATABASE_PATH above: dashboard.backend.database reads this
# var at import time to build its singleton, and PostgresBacktestDatabase's
# embedded BacktestDatabase() would too. Pop it, import, then restore it --
# main() always reads it fresh via os.environ.get(), never from a captured
# value, so restoring costs nothing and keeps this script testable (a test's
# monkeypatch.setenv, applied after this module is already imported, wins).
_agent_runs_url_snapshot = os.environ.pop("AGENT_RUNS_DATABASE_URL", None)
from dashboard.backend.database import BacktestDatabase  # noqa: E402
from dashboard.backend.db_url import describe_database_url  # noqa: E402
from dashboard.backend.paths import DEFAULT_DB_PATH  # noqa: E402

if _agent_runs_url_snapshot is not None:
    os.environ["AGENT_RUNS_DATABASE_URL"] = _agent_runs_url_snapshot


TABLES_IN_FK_ORDER = (
    "agent_runs",
    "equity_timeseries",
    "trades",
    "backtest_decisions",
    "run_manifest",
)


# ---------------------------------------------------------------------------
# Source: copied, migrated, then read only through BacktestDatabase's own
# public methods -- never a hand-rolled query against --source's raw columns.
# ---------------------------------------------------------------------------

def _copy_source(source_path: Path, dest_path: Path) -> None:
    """Copy ``source_path`` into ``dest_path`` using SQLite's own backup API,
    never a raw byte copy and never a live connection to ``source_path``.

    Two things this avoids:
    * A raw ``shutil.copy2`` of a SQLite file can tear mid-copy if the source
      is ever a live file being written concurrently (not the committed seed
      in the common case, but ``--source`` accepts any path). ``backup()``
      reads through SQLite's own pager layer, so the copy is always a
      consistent snapshot.
    * Opening ``source_path`` directly -- even read-only -- risks leaving
      ``-wal``/``-shm`` sidecars next to a *committed* file just from the
      open. Confirmed empirically (not assumed from the docs): opening a
      fresh copy of the committed seed with ``mode=ro`` creates both sidecar
      files as a side effect; ``immutable=1`` creates neither. That is
      exactly why ``test_seed_database_integrity.py`` -- which reads this
      same committed file -- already uses ``immutable=1``; the source
      connection here uses it for the same reason, and only as the backup
      API's source, never for reading rows directly.

    ``immutable=1`` trusts the main file as static truth and does not look at
    the WAL. That is not a risk in practice: every write path in
    ``BacktestDatabase`` (both the real one behind the committed seed and the
    one the @pg_only test builds against a ``tmp_path``) opens a connection,
    writes, commits and closes it per call rather than holding one open --
    confirmed empirically that this leaves no live WAL data behind (SQLite
    auto-checkpoints on last-connection close) -- so ``immutable=1`` sees the
    same data a normal connection would.
    """
    src_conn = sqlite3.connect(f"file:{source_path}?immutable=1", uri=True)
    dest_conn = sqlite3.connect(str(dest_path))
    try:
        src_conn.backup(dest_conn)
    finally:
        dest_conn.close()
        src_conn.close()


def _table_count(conn: sqlite3.Connection, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _child_counts_by_run(conn: sqlite3.Connection, table: str) -> Dict[str, int]:
    """Row count per ``run_id`` for a child table, using only the ``run_id``
    column -- present in every schema version this script has ever seen,
    unlike the optional/legacy columns ``get_equity_curve``/``get_trades``/
    ``get_decisions`` already handle internally (see ``SourceData``). Used
    only to size the "orphan" bucket in the report; the actual row data comes
    from those public methods, not from this query.
    """
    if table == "run_manifest":
        rows = conn.execute("SELECT run_id FROM run_manifest").fetchall()
        return {row[0]: 1 for row in rows}
    rows = conn.execute(f"SELECT run_id, COUNT(*) AS n FROM {table} GROUP BY run_id").fetchall()
    return {row[0]: row[1] for row in rows}


def _coalesce(row: Dict[str, Any], key: str, default: Any) -> Any:
    value = row.get(key)
    return default if value is None else value


class SourceData:
    """Everything needed from ``--source``, read through a migrated private
    copy's own ``BacktestDatabase`` public methods -- the same methods the
    app itself uses, so this can never drift from what "the current schema"
    means the way a hand-rolled column list can (and, on the real committed
    seed, did: see the module docstring).

    ``raw_counts`` and the per-table orphan counts are computed straight off
    the copy afterward (schema-agnostic ``COUNT`` queries), so the report can
    show source vs. migrated vs. skipped without a second read of anything.
    """

    def __init__(self, copy_path: Path):
        source_db = BacktestDatabase(db_path=copy_path)

        self.runs = source_db.get_all_runs()
        # Oldest first: agent_runs is written in this order below, so ties in
        # the twin's second-granularity `created_at` DEFAULT (very likely for
        # a batch backfill that completes in under a second) at least sort by
        # insertion order rather than an arbitrary source order. Exact
        # original timestamps are not preserved either way -- see the module
        # docstring's note on writing only through the twin's public methods.
        self.runs.sort(key=lambda r: r.get("created_at") or "")
        self.run_ids = {r["run_id"] for r in self.runs}

        self.equity_by_run = {rid: source_db.get_equity_curve(rid) for rid in self.run_ids}
        self.trades_by_run = {rid: source_db.get_trades(rid) for rid in self.run_ids}
        self.decisions_by_run = {rid: source_db.get_decisions(rid) for rid in self.run_ids}
        self.manifests_by_run: Dict[str, Dict[str, Any]] = {}
        for rid in self.run_ids:
            manifest = source_db.get_run_manifest(rid)
            if manifest is not None:
                self.manifests_by_run[rid] = manifest

        raw_conn = sqlite3.connect(str(copy_path))
        try:
            self.raw_counts = {table: _table_count(raw_conn, table) for table in TABLES_IN_FK_ORDER}
            self.orphan_counts = {
                table: sum(
                    n for run_id, n in _child_counts_by_run(raw_conn, table).items()
                    if run_id not in self.run_ids
                )
                for table in TABLES_IN_FK_ORDER
                if table != "agent_runs"
            }
        finally:
            raw_conn.close()


# ---------------------------------------------------------------------------
# Target: write through public methods only
# ---------------------------------------------------------------------------

def _migrate_agent_runs(target: "PostgresBacktestDatabase", runs: List[Dict[str, Any]]) -> int:
    """Naturally idempotent: insert_run upserts on (run_id), and
    update_run_baselines COALESCEs -- a rerun just re-writes the same values.
    """
    for run in runs:
        target.insert_run(
            run_id=run["run_id"],
            session_id=run["session_id"],
            agent_name=run["agent_name"],
            mode=run["mode"],
            start_date=run["start_date"],
            end_date=run["end_date"],
            initial_equity=run["initial_equity"],
            final_equity=run.get("final_equity"),
            total_return=run.get("total_return"),
            sharpe_ratio=run.get("sharpe_ratio"),
            max_drawdown=run.get("max_drawdown"),
            num_trades=_coalesce(run, "num_trades", 0),
            llm_model=_coalesce(run, "llm_model", "rule-based"),
            llm_calls=_coalesce(run, "llm_calls", 0),
            input_tokens=_coalesce(run, "input_tokens", 0),
            output_tokens=_coalesce(run, "output_tokens", 0),
            est_cost_usd=_coalesce(run, "est_cost_usd", 0.0),
            metadata=run.get("metadata"),
        )
        djia = run.get("baseline_djia_run_id")
        buyhold = run.get("baseline_buyhold_run_id")
        if djia or buyhold:
            # insert_run has no baseline-link params -- those are set via this
            # separate call, same as every other production writer of them.
            target.update_run_baselines(run["run_id"], djia_run_id=djia, buyhold_run_id=buyhold)
    return len(runs)


def _migrate_equity(
    target: "PostgresBacktestDatabase", equity_by_run: Dict[str, List[Dict[str, Any]]]
) -> int:
    """Naturally idempotent: insert_equity_points(replace=True) deletes then
    re-inserts this run's curve every call, landing on the same final rows.
    """
    moved = 0
    for run_id, points in equity_by_run.items():
        if not points:
            continue
        target.insert_equity_points(run_id, points)
        moved += len(points)
    return moved


def _migrate_trades(
    target: "PostgresBacktestDatabase", trades_by_run: Dict[str, List[Dict[str, Any]]]
) -> Tuple[int, int]:
    """insert_trades is a plain append on the twin (no natural unique key for
    a trade row), so this script supplies its own idempotency: skip a run
    entirely if the target already has any trades for it.
    """
    moved = 0
    skipped_present = 0
    for run_id, trades in trades_by_run.items():
        if not trades:
            continue
        if target.get_trades(run_id):
            skipped_present += len(trades)
            continue
        target.insert_trades(run_id, trades)
        moved += len(trades)
    return moved, skipped_present


def _migrate_decisions(
    target: "PostgresBacktestDatabase", decisions_by_run: Dict[str, List[Dict[str, Any]]]
) -> Tuple[int, int]:
    """Same append-only shape as trades -- see _migrate_trades."""
    moved = 0
    skipped_present = 0
    for run_id, decisions in decisions_by_run.items():
        if not decisions:
            continue
        if target.get_decisions(run_id):
            skipped_present += len(decisions)
            continue
        target.insert_decisions(run_id, decisions)
        moved += len(decisions)
    return moved, skipped_present


def _migrate_manifests(
    target: "PostgresBacktestDatabase", manifests_by_run: Dict[str, Dict[str, Any]]
) -> int:
    """Naturally idempotent: insert_run_manifest upserts on (run_id)."""
    for run_id, manifest in manifests_by_run.items():
        target.insert_run_manifest(run_id, manifest)
    return len(manifests_by_run)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill backtest run history from a SQLite BacktestDatabase file "
            "into the Postgres run-history twin selected by AGENT_RUNS_DATABASE_URL."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"source SQLite file to read (default: the committed seed, {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report source counts and exit without writing anything or requiring "
             "AGENT_RUNS_DATABASE_URL to be set",
    )
    args = parser.parse_args()

    source_path: Path = args.source
    if not source_path.exists():
        print(f"ERROR: source database not found: {source_path}", file=sys.stderr)
        return 1

    print(f"Source: {source_path}")
    copy_path = Path(_SCRATCH_DIR) / "source_copy.db"
    _copy_source(source_path, copy_path)
    data = SourceData(copy_path)

    print("Source table counts:")
    for table in TABLES_IN_FK_ORDER:
        print(f"  {table}: {data.raw_counts[table]}")

    if args.dry_run:
        preview_url = os.environ.get("AGENT_RUNS_DATABASE_URL")
        if preview_url:
            print(f"Target (not connected -- dry run): postgres ({describe_database_url(preview_url)})")
        else:
            print("Target: AGENT_RUNS_DATABASE_URL is not set (fine for a dry run).")
        print("\nDry run: no writes performed.")
        return 0

    database_url = os.environ.get("AGENT_RUNS_DATABASE_URL")
    if not database_url:
        print(
            "ERROR: AGENT_RUNS_DATABASE_URL is not set. This script writes through "
            "the Postgres run-history twin and needs a target database.",
            file=sys.stderr,
        )
        return 1

    from dashboard.backend.database_postgres import PostgresBacktestDatabase

    print(f"Target: postgres ({describe_database_url(database_url)})")
    target = PostgresBacktestDatabase(database_url)

    results: Dict[str, Dict[str, int]] = {}

    moved = _migrate_agent_runs(target, data.runs)
    results["agent_runs"] = {"migrated": moved}

    moved = _migrate_equity(target, data.equity_by_run)
    results["equity_timeseries"] = {"migrated": moved, "skipped_orphan": data.orphan_counts["equity_timeseries"]}

    moved, skipped_present = _migrate_trades(target, data.trades_by_run)
    results["trades"] = {
        "migrated": moved,
        "skipped_orphan": data.orphan_counts["trades"],
        "skipped_already_present": skipped_present,
    }

    moved, skipped_present = _migrate_decisions(target, data.decisions_by_run)
    results["backtest_decisions"] = {
        "migrated": moved,
        "skipped_orphan": data.orphan_counts["backtest_decisions"],
        "skipped_already_present": skipped_present,
    }

    moved = _migrate_manifests(target, data.manifests_by_run)
    results["run_manifest"] = {"migrated": moved, "skipped_orphan": data.orphan_counts["run_manifest"]}

    print("\nPer-table results (source vs. migrated):")
    accounting_ok = True
    for table in TABLES_IN_FK_ORDER:
        source_count = data.raw_counts[table]
        r = results[table]
        accounted = r["migrated"] + r.get("skipped_orphan", 0) + r.get("skipped_already_present", 0)
        line = f"  {table}: source={source_count} migrated={r['migrated']}"
        if r.get("skipped_orphan"):
            line += f" skipped_orphan(no matching agent_runs)={r['skipped_orphan']}"
        if "skipped_already_present" in r:
            line += f" skipped_already_present={r['skipped_already_present']}"
        print(line)
        if accounted != source_count:
            accounting_ok = False
            print(
                f"  WARNING: {table} accounting mismatch -- source={source_count} "
                f"but migrated+skipped={accounted}",
                file=sys.stderr,
            )

    if not accounting_ok:
        print("\nBackfill finished with accounting mismatches -- see warnings above.", file=sys.stderr)
        return 1

    print("\nBackfill complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
