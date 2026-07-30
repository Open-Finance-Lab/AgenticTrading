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

Re-runnable *to completion*, which is a separate property from idempotency and
was not true of the first cut. Every per-run write is now wrapped individually
(``_Failures``), so a legacy row this script cannot migrate is reported and
skipped instead of aborting the pass. Before that, one bad row stalled the
backfill permanently: it propagated out of ``main()``, and because the source is
read in a fixed order every rerun re-failed on the same row, so every run
ordered after it never migrated at all -- while the summary the operator saw was
a traceback rather than "these 3 runs need attention".

Only one copy may run at a time, enforced rather than assumed
(``_exclusive_backfill_lock``). The trades/decisions skip-logic described above
is a read-then-write spanning two *separate* pooled checkouts, so two concurrent
invocations both observe "nothing there yet" and both append -- doubling every
row, with no natural unique key on either table to collapse the duplicates
afterwards. A Postgres advisory lock is held for the whole write phase, and a
second invocation refuses to start rather than queueing behind the first (a
queued run would be pointless: by the time it acquired the lock the work would
already be done).

``created_at`` is restored from the source after the copy (``_restore_created_at``);
``updated_at`` deliberately is not. ``insert_run``'s own upsert has no
``created_at`` column in its ``INSERT``, so a freshly-migrated run gets
Postgres's insert-time default (i.e. backfill time) instead of when the run
actually happened -- and ``created_at DESC`` genuinely drives run ordering in
``database.py`` and "latest run for this agent" in
``domain/agents/service.py``, so getting it wrong is user-visible, not
cosmetic. ``updated_at`` is different: grep across the backend turns up no
code that reads ``agent_runs.updated_at`` for anything (the API's
``RunMetadata`` schema does not even expose it) -- and the twin's own upsert
docstring already establishes what the column means, "last time this row was
written here," refreshed on every ``insert_run`` call on purpose. The backfill
genuinely does write the row at backfill time, so leaving ``updated_at`` at
that value is the truthful choice; overwriting it with the SQLite source's
historical value would misrepresent this database's own write history for a
column nothing currently depends on.

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
import contextlib
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

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

# Arbitrary but fixed: pg_advisory_lock takes a bigint, and any two invocations
# of this script must pick the *same* number or the lock protects nothing.
# Never derive it from anything environmental (pid, url, timestamp).
BACKFILL_ADVISORY_LOCK_KEY = 4_020_260_729


class BackfillLockUnavailable(RuntimeError):
    """Another invocation holds the advisory lock; this one must not proceed.

    Its own class rather than a bare RuntimeError so ``main()`` can catch
    exactly this around the whole write phase without also swallowing a genuine
    RuntimeError from somewhere inside it.
    """


class _Failures:
    """Per-run migration failures, collected instead of raised.

    The unit of failure is one ``(table, run_id)`` pair, because that is the
    unit this script can actually skip and re-attempt later. Recording the
    exception as text rather than keeping the object is deliberate: the report
    is printed after every connection involved has been returned to the pool,
    and a psycopg error's ``__str__`` is already the useful part.
    """

    def __init__(self) -> None:
        # Named "entries", not "items": a plain list attribute called ``items``
        # reads like the dict method and invites ``failures.items()``.
        self.entries: List[Tuple[str, str, str]] = []

    def record(self, table: str, run_id: str, exc: BaseException) -> None:
        self.entries.append((table, run_id, f"{type(exc).__name__}: {exc}"))

    def run_ids_for(self, table: str) -> set:
        return {run_id for tbl, run_id, _ in self.entries if tbl == table}

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return bool(self.entries)


@contextlib.contextmanager
def _exclusive_backfill_lock(target: "PostgresBacktestDatabase") -> Iterator[None]:
    """Hold a Postgres advisory lock for the whole write phase.

    ``pg_try_advisory_lock`` rather than ``pg_advisory_lock``: blocking would
    make a second invocation sit indefinitely and then do nothing useful (the
    first one will have finished the work), so failing immediately with a clear
    message is strictly more informative than a hang.

    The lock is *session*-scoped, i.e. bound to the connection that took it, so
    this deliberately holds one pooled checkout open across the entire phase
    instead of taking and returning one per table. The explicit unlock in
    ``finally`` matters for the same reason: pooled connections are reused, and
    a lock left behind would travel into whatever checked that connection out
    next. (Process exit would release it too, but only eventually, and only if
    the process actually dies.)
    """
    with target._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_try_advisory_lock(%s) AS acquired",
                (BACKFILL_ADVISORY_LOCK_KEY,),
            )
            acquired = cur.fetchone()["acquired"]
        # Close the implicit transaction the SELECT opened. Without this the
        # lock connection sits `idle in transaction` for the whole write phase,
        # and managed providers (Neon included) commonly enforce
        # idle_in_transaction_session_timeout -- which would kill the session,
        # silently release the lock mid-phase, and then make the unlock raise.
        # Safe because pg_try_advisory_lock takes a *session*-scoped lock:
        # committing does not release it (only pg_advisory_xact_lock is
        # transaction-scoped).
        conn.commit()
        if not acquired:
            # Raised before the try/finally on purpose: no lock was taken, so
            # there is nothing to unlock.
            raise BackfillLockUnavailable(
                "another backfill_runs_to_postgres.py run holds the advisory "
                f"lock ({BACKFILL_ADVISORY_LOCK_KEY}) on this database. "
                "Two concurrent runs would double every trade and decision "
                "row -- neither table has a unique key to collapse them. "
                "Wait for the other run to finish, then re-run this one."
            )
        try:
            yield
        finally:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT pg_advisory_unlock(%s)", (BACKFILL_ADVISORY_LOCK_KEY,)
                )
            conn.commit()


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
        # Per-run guard on the manifest read specifically. Every other reader
        # tolerates junk already -- _parse_run_row catches TypeError/ValueError,
        # get_decisions catches JSONDecodeError -- but get_run_manifest
        # (database.py) does a bare json.loads, so one corrupt manifest_json
        # would raise here, in the READ phase, before _Failures exists and
        # before a single row is written. That would make the module docstring's
        # "re-runnable to completion" false in the one place no rerun can help.
        # Only reachable from a hand-edited or legacy source (api/v2/runs.py is
        # the sole writer and always passes a model_dump()), which is exactly
        # what --source accepts.
        self.manifests_by_run: Dict[str, Dict[str, Any]] = {}
        self.unreadable_manifests: List[Tuple[str, str]] = []
        for rid in self.run_ids:
            try:
                manifest = source_db.get_run_manifest(rid)
            except Exception as exc:
                self.unreadable_manifests.append((rid, f"{type(exc).__name__}: {exc}"))
                continue
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
            # The second way a manifest row can fail to migrate, and the reason
            # the try/except above is not sufficient on its own: a
            # ``manifest_json`` holding the JSON literal ``null`` decodes to
            # Python ``None`` *without raising*, so ``get_run_manifest`` returns
            # None exactly as it does for "no row at all" and the public API
            # cannot tell the two apart. Left unaccounted, such a row is counted
            # in raw_counts but lands in no bucket -- i.e. it reports as an
            # *unexplained* shortfall, which is the one signal the accounting
            # check exists to raise. Resolved with the row set from the raw
            # connection, the only place that distinction is visible.
            manifest_row_ids = set(_child_counts_by_run(raw_conn, "run_manifest"))
            accounted_ids = set(self.manifests_by_run) | {
                rid for rid, _ in self.unreadable_manifests
            }
            self.null_manifests = sorted(
                (manifest_row_ids & self.run_ids) - accounted_ids
            )
        finally:
            raw_conn.close()


# ---------------------------------------------------------------------------
# Target: write through public methods only
# ---------------------------------------------------------------------------

def _migrate_agent_runs(
    target: "PostgresBacktestDatabase",
    runs: List[Dict[str, Any]],
    failures: _Failures,
) -> Dict[str, int]:
    """Naturally idempotent: insert_run upserts on (run_id), and
    update_run_baselines COALESCEs -- a rerun just re-writes the same values.

    Per-run ``try``/``except`` so one unmigratable legacy row costs that row
    rather than the whole pass. Safe as a plain handler here because every
    writer method opens its own ``with self._get_connection()`` block, i.e. its
    own transaction, which the pool rolls back before handing the connection to
    the next call -- unlike ``_restore_created_at``, which shares one connection
    and therefore needs an explicit per-run ``conn.transaction()`` block.

    A run that fails here is also excluded from every child table below: its
    ``agent_runs`` parent may not exist, and the FK would reject the children
    anyway (with the failure attributed to the child rather than the cause).
    """
    migrated = 0
    failed = 0
    for run in runs:
        try:
            _insert_one_run(target, run)
        except Exception as exc:
            failures.record("agent_runs", run.get("run_id", "<unknown>"), exc)
            failed += 1
            continue
        migrated += 1
    return {"migrated": migrated, "failed": failed}


def _insert_one_run(target: "PostgresBacktestDatabase", run: Dict[str, Any]) -> None:
    """One run's ``agent_runs`` row plus its baseline links.

    Split out of the loop above purely so the loop's ``try`` wraps a single
    call and cannot accidentally grow to cover the bookkeeping around it.
    """
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


def _restore_created_at(
    target: "PostgresBacktestDatabase",
    runs: List[Dict[str, Any]],
    failures: _Failures,
) -> int:
    """Restore each run's original ``created_at`` from the source, after
    ``_migrate_agent_runs`` has upserted the row.

    Why this is needed: ``insert_run``'s ``INSERT`` (the first-ever write of a
    run_id into this Postgres table) has no ``created_at`` in its column list,
    so Postgres stamps it from the column ``DEFAULT`` -- i.e. the moment this
    backfill ran, not when the run actually happened. `database.py` orders
    run listings by ``created_at DESC`` in four places, and
    ``domain/agents/service.py`` sorts by it in four more specifically to pick
    the *latest* run for an agent; leaving every migrated run stamped with
    the same backfill-time timestamp would make that ordering non-deterministic
    among ties (all 17 production runs land within the same ~second).

    Deliberately does not restore ``updated_at`` -- see the module docstring
    for the reasoning. Deliberately not done as an extra ``insert_run``
    parameter: that would change the twin's method signature, which
    ``test_postgres_twin_signatures_match_sqlite`` compares against
    ``BacktestDatabase`` and would redden CI. A raw ``UPDATE`` here is
    acceptable -- the "write through public methods" rule was about the copy
    itself (so idempotency comes free from the twin's own upserts), a
    property this restoration pass does not touch: ``insert_run``'s own
    ``ON CONFLICT`` deliberately never writes ``created_at`` on a re-insert
    (first-seen semantics -- see its docstring), so this pass only needs to
    fix the very first insert's wrong default, and is naturally idempotent
    regardless -- every call sets the same source-captured value, never
    "now()", so a rerun converges rather than drifts.

    Skips a run (or just its ``created_at``) whose source value is missing --
    the column is ``NOT NULL`` on both sides, and every real row has always
    had a DEFAULT-populated value (the column existed with
    ``DEFAULT CURRENT_TIMESTAMP`` since the very first schema version, so it
    is never NULL in practice) -- but this must not become a NULL constraint
    violation on some future source this script has never seen.

    Unlike the ``_migrate_*`` loops, this one shares a single connection across
    every run, so a bare ``try``/``except`` would *not* isolate a failure: on
    Postgres a failed statement aborts the whole transaction, and every
    subsequent UPDATE on that connection would then raise
    ``InFailedSqlTransaction`` -- turning one bad row into a total loss while
    looking, in the report, like every row was individually bad. Each UPDATE
    therefore runs inside its own ``conn.transaction()`` block.

    What that block actually emits, since the name is easy to guess wrong
    (checked against psycopg 3.3.4's ``transaction.py``, not assumed): it
    branches on ``pgconn.transaction_status == IDLE``. A pooled checkout does
    not ``BEGIN`` until something executes, and each block here COMMITs and
    returns the connection to IDLE -- so every iteration is an *independent
    transaction* (``BEGIN``/``COMMIT``, rolled back with plain ``ROLLBACK``),
    not a savepoint. ``SAVEPOINT``/``ROLLBACK TO`` would only appear if a
    transaction were already open on this connection when the block was
    entered. Either form gives the per-run isolation this needs; the
    distinction matters only if someone later executes something on ``conn``
    before the loop, which silently switches the mechanism (and is fine).

    A consequence worth naming: because each run commits separately, a run that
    fails leaves the successful restores durable rather than discarding them.
    That is the intent -- this pass is idempotent and converges on a rerun (it
    always writes the same source-captured value, never ``now()``), so partial
    progress is strictly better than all-or-nothing here.
    """
    restored = 0
    # A run whose insert failed has no row to update. The UPDATE would match
    # zero rows and raise nothing, so without this the reported "restored for
    # N/M" count would silently include runs that are not in the target at all.
    blocked = failures.run_ids_for("agent_runs")
    with target._get_connection() as conn:
        for run in runs:
            created_at = run.get("created_at")
            if created_at is None or run["run_id"] in blocked:
                continue
            try:
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE agent_runs SET created_at = %s WHERE run_id = %s",
                            (created_at, run["run_id"]),
                        )
            except Exception as exc:
                failures.record("agent_runs.created_at", run["run_id"], exc)
                continue
            restored += 1
    return restored


def _migrate_equity(
    target: "PostgresBacktestDatabase",
    equity_by_run: Dict[str, List[Dict[str, Any]]],
    failures: _Failures,
) -> Dict[str, int]:
    """Naturally idempotent: insert_equity_points(replace=True) deletes then
    re-inserts this run's curve every call, landing on the same final rows.

    The ``skipped_parent_failed`` and ``failed`` buckets exist so ``main()``'s
    accounting check still balances when something went wrong above: those rows
    were neither migrated nor orphaned in the source, and without their own
    buckets they would show up as an *unexplained* shortfall -- which is the one
    signal that check exists to raise.
    """
    moved = 0
    failed = 0
    skipped_parent_failed = 0
    blocked = failures.run_ids_for("agent_runs")
    for run_id, points in equity_by_run.items():
        if not points:
            continue
        if run_id in blocked:
            skipped_parent_failed += len(points)
            continue
        try:
            target.insert_equity_points(run_id, points)
        except Exception as exc:
            failures.record("equity_timeseries", run_id, exc)
            failed += len(points)
            continue
        moved += len(points)
    return {
        "migrated": moved,
        "failed": failed,
        "skipped_parent_failed": skipped_parent_failed,
    }


def _migrate_trades(
    target: "PostgresBacktestDatabase",
    trades_by_run: Dict[str, List[Dict[str, Any]]],
    failures: _Failures,
) -> Dict[str, int]:
    """insert_trades is a plain append on the twin (no natural unique key for
    a trade row), so this script supplies its own idempotency: skip a run
    entirely if the target already has any trades for it.

    The read-back that provides that idempotency is also why the whole write
    phase runs under an advisory lock -- see ``_exclusive_backfill_lock``.
    """
    moved = 0
    failed = 0
    skipped_present = 0
    skipped_parent_failed = 0
    blocked = failures.run_ids_for("agent_runs")
    for run_id, trades in trades_by_run.items():
        if not trades:
            continue
        if run_id in blocked:
            skipped_parent_failed += len(trades)
            continue
        try:
            if target.get_trades(run_id):
                skipped_present += len(trades)
                continue
            target.insert_trades(run_id, trades)
        except Exception as exc:
            failures.record("trades", run_id, exc)
            failed += len(trades)
            continue
        moved += len(trades)
    return {
        "migrated": moved,
        "failed": failed,
        "skipped_already_present": skipped_present,
        "skipped_parent_failed": skipped_parent_failed,
    }


def _migrate_decisions(
    target: "PostgresBacktestDatabase",
    decisions_by_run: Dict[str, List[Dict[str, Any]]],
    failures: _Failures,
) -> Dict[str, int]:
    """Same append-only shape as trades -- see _migrate_trades."""
    moved = 0
    failed = 0
    skipped_present = 0
    skipped_parent_failed = 0
    blocked = failures.run_ids_for("agent_runs")
    for run_id, decisions in decisions_by_run.items():
        if not decisions:
            continue
        if run_id in blocked:
            skipped_parent_failed += len(decisions)
            continue
        try:
            if target.get_decisions(run_id):
                skipped_present += len(decisions)
                continue
            target.insert_decisions(run_id, decisions)
        except Exception as exc:
            failures.record("backtest_decisions", run_id, exc)
            failed += len(decisions)
            continue
        moved += len(decisions)
    return {
        "migrated": moved,
        "failed": failed,
        "skipped_already_present": skipped_present,
        "skipped_parent_failed": skipped_parent_failed,
    }


def _migrate_manifests(
    target: "PostgresBacktestDatabase",
    manifests_by_run: Dict[str, Dict[str, Any]],
    failures: _Failures,
) -> Dict[str, int]:
    """Naturally idempotent: insert_run_manifest upserts on (run_id).

    ``run_manifest`` has no FK to ``agent_runs`` (deliberately -- see
    database_postgres.py's ``_init_schema``), so a manifest *could* be written
    for a run whose parent failed. It is still skipped: a manifest with no run
    to describe is not useful, and writing it would leave the target holding
    rows the operator was told had failed.
    """
    migrated = 0
    failed = 0
    skipped_parent_failed = 0
    blocked = failures.run_ids_for("agent_runs")
    for run_id, manifest in manifests_by_run.items():
        if run_id in blocked:
            skipped_parent_failed += 1
            continue
        try:
            target.insert_run_manifest(run_id, manifest)
        except Exception as exc:
            failures.record("run_manifest", run_id, exc)
            failed += 1
            continue
        migrated += 1
    return {
        "migrated": migrated,
        "failed": failed,
        "skipped_parent_failed": skipped_parent_failed,
    }


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

    source_is_empty = not any(data.raw_counts.values())
    if source_is_empty:
        # Fail-visible: an existing-but-wrong --source (a typo that happens to
        # land on some other SQLite file, a fresh scratch DB, a path pointing
        # into a container that never had the seed) otherwise produces exactly
        # the same output as a legitimate no-op, down to the exit code. The
        # existence check above cannot tell them apart -- the file *is* there.
        print(
            "\n"
            "WARNING: every source table is empty -- 0 runs, 0 equity points,\n"
            f"         0 trades, 0 decisions, 0 manifests, read from:\n"
            f"           {source_path.resolve()}\n"
            "         Either there is genuinely nothing to migrate, or this is\n"
            "         the wrong file. It exists and opened cleanly, so nothing\n"
            "         else in this run will flag it.",
            file=sys.stderr,
        )
        if source_path != DEFAULT_DB_PATH:
            print(
                "         This is not the default seed path, which makes a\n"
                f"         mistyped --source the more likely of the two.\n"
                f"         The default would have been: {DEFAULT_DB_PATH}",
                file=sys.stderr,
            )

    if args.dry_run:
        preview_url = os.environ.get("AGENT_RUNS_DATABASE_URL")
        if preview_url:
            print(f"Target (not connected -- dry run): postgres ({describe_database_url(preview_url)})")
        else:
            print("Target: AGENT_RUNS_DATABASE_URL is not set (fine for a dry run).")
        if source_is_empty:
            print("\nDry run: no writes performed (and nothing to write -- see warning above).")
        else:
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
    failures = _Failures()

    try:
        with _exclusive_backfill_lock(target):
            results["agent_runs"] = _migrate_agent_runs(target, data.runs, failures)
            restored = _restore_created_at(target, data.runs, failures)
            print(
                f"  agent_runs.created_at restored from source for "
                f"{restored}/{len(data.runs)} run(s)"
            )

            results["equity_timeseries"] = _migrate_equity(
                target, data.equity_by_run, failures
            )
            results["trades"] = _migrate_trades(target, data.trades_by_run, failures)
            results["backtest_decisions"] = _migrate_decisions(
                target, data.decisions_by_run, failures
            )
            results["run_manifest"] = _migrate_manifests(
                target, data.manifests_by_run, failures
            )
    except BackfillLockUnavailable as exc:
        # Distinct from a per-run failure: nothing was attempted, so there is no
        # partial state and no per-table report worth printing.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for table in TABLES_IN_FK_ORDER:
        results[table]["skipped_orphan"] = data.orphan_counts.get(table, 0)
    # Its own bucket, so an unreadable manifest reads as a *named* reason rather
    # than an unexplained shortfall -- which is the only thing the accounting
    # check should ever fail on.
    results["run_manifest"]["skipped_unreadable"] = (
        len(data.unreadable_manifests) + len(data.null_manifests)
    )

    print("\nPer-table results (source vs. migrated):")
    accounting_ok = True
    for table in TABLES_IN_FK_ORDER:
        source_count = data.raw_counts[table]
        r = results[table]
        # Every bucket except "migrated" is a *reason* a source row is not in
        # the target. They must sum to the source count, or something happened
        # this script cannot explain -- which is the only thing worth failing on.
        accounted = (
            r["migrated"]
            + r.get("skipped_orphan", 0)
            + r.get("skipped_already_present", 0)
            + r.get("skipped_parent_failed", 0)
            + r.get("skipped_unreadable", 0)
            + r.get("failed", 0)
        )
        line = f"  {table}: source={source_count} migrated={r['migrated']}"
        if r.get("skipped_orphan"):
            line += f" skipped_orphan(no matching agent_runs)={r['skipped_orphan']}"
        if "skipped_already_present" in r:
            line += f" skipped_already_present={r['skipped_already_present']}"
        if r.get("skipped_parent_failed"):
            line += f" skipped_parent_failed={r['skipped_parent_failed']}"
        if r.get("skipped_unreadable"):
            line += f" skipped_unreadable={r['skipped_unreadable']}"
        if r.get("failed"):
            line += f" FAILED={r['failed']}"
        print(line)
        if accounted != source_count:
            accounting_ok = False
            print(
                f"  WARNING: {table} accounting mismatch -- source={source_count} "
                f"but migrated+skipped={accounted}",
                file=sys.stderr,
            )

    if failures:
        # Deliberately not "these rows are NOT in the target": _insert_one_run
        # writes agent_runs and then update_run_baselines in two separate
        # transactions, so a run listed under agent_runs may in fact have its
        # row present with its baseline links missing. Every other run was
        # migrated, and a rerun converges either way -- but the report should
        # not assert something stronger than it can know.
        print(
            f"\n{len(failures)} per-run failure(s) -- these runs did not migrate "
            "cleanly and are incomplete or absent in the target. Every other run "
            "was migrated; re-run this script after fixing them:",
            file=sys.stderr,
        )
        for table, run_id, error in failures.entries:
            print(f"  {table} / {run_id}: {error}", file=sys.stderr)

    if data.unreadable_manifests or data.null_manifests:
        total = len(data.unreadable_manifests) + len(data.null_manifests)
        print(
            f"\n{total} manifest(s) could not be read from the source and were "
            "skipped (see SourceData):",
            file=sys.stderr,
        )
        for run_id, error in data.unreadable_manifests:
            print(f"  run_manifest / {run_id}: {error}", file=sys.stderr)
        for run_id in data.null_manifests:
            print(
                f"  run_manifest / {run_id}: manifest_json decodes to null",
                file=sys.stderr,
            )

    if not accounting_ok:
        print("\nBackfill finished with accounting mismatches -- see warnings above.", file=sys.stderr)
        return 1

    if failures or data.unreadable_manifests or data.null_manifests:
        print("\nBackfill finished with failures -- see the list above.", file=sys.stderr)
        return 1

    if source_is_empty:
        print("\nBackfill complete: nothing to migrate (see warning above).")
        return 0

    print("\nBackfill complete.")
    return 0


def _close_pools() -> None:
    """Return pooled sockets before the interpreter tears itself down.

    psycopg_pool's own exit path cannot stop its background workers in time
    and prints ``couldn't stop thread 'pool-1-worker-N' within 5.0 seconds``
    once per worker -- *after* this script's own "Backfill complete.", so a
    fully successful run ends looking like a partial failure.

    Keyed off ``sys.modules`` rather than importing: ``--dry-run`` returns
    before the Postgres stack is ever imported, and it has to keep working on
    a machine with no psycopg installed at all. Nothing was pooled in that
    case, so there is nothing to close.
    """
    pool_module = sys.modules.get("dashboard.backend.db_pool")
    if pool_module is not None:
        pool_module.close_all_pools()


if __name__ == "__main__":
    # finally, not "after main()": a crash mid-backfill should still hand the
    # sockets back rather than bury the traceback under teardown warnings.
    try:
        _exit_code = main()
    finally:
        _close_pools()
    raise SystemExit(_exit_code)
