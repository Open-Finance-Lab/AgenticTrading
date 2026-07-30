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

Tier 1 is bigger than "the CLI errors", because the branches that matter most
are the ones that only fire when something goes *wrong*: a row that will not
migrate, a second invocation racing the first, an empty source. Leaving those to
the @pg_only tier would mean they were never executed outside CI. ``_FakeTarget``
below stands in for ``PostgresBacktestDatabase`` -- ``main()`` imports that class
*inside* the function, so patching the attribute on its source module is enough
to drive the entire script, failure paths included, with no server anywhere.
That is a supplement to tier 2, not a replacement: a fake cannot prove a
transaction block really isolates a failed statement on Postgres, only that the
code opens one per run.
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


def test_close_pools_returns_pooled_sockets(monkeypatch):
    """The entry point hands the pooled sockets back, so psycopg_pool's own
    teardown does not print "couldn't stop thread 'pool-1-worker-N'" *after*
    "Backfill complete." and make a clean run read as a partial failure.

    Covers the helper, not the ``__main__`` block that calls it: every test
    here drives ``main()`` in-process, so that wiring is only exercised by
    actually running the script.
    """
    from dashboard.backend import db_pool
    from dashboard.scripts import backfill_runs_to_postgres

    calls = []
    monkeypatch.setattr(db_pool, "close_all_pools", lambda: calls.append(1))

    backfill_runs_to_postgres._close_pools()

    assert calls == [1]


def test_close_pools_is_a_noop_when_postgres_was_never_imported(monkeypatch):
    """``--dry-run`` returns before the Postgres stack is imported and has to
    keep working where psycopg is not installed at all. Nothing was pooled, so
    the helper must not reach for the module -- importing it there would both
    risk an ImportError and build a *second* pool registry."""
    from dashboard.scripts import backfill_runs_to_postgres

    monkeypatch.delitem(sys.modules, "dashboard.backend.db_pool", raising=False)

    backfill_runs_to_postgres._close_pools()

    assert "dashboard.backend.db_pool" not in sys.modules


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


# --- empty-but-valid source: the case the existence check cannot catch --------

def test_empty_source_warns_loudly_instead_of_reporting_plain_success(
    tmp_path, monkeypatch, capsys
):
    """A --source that exists, opens cleanly, and holds nothing.

    Before this warning, the run was byte-identical to a legitimate no-op:
    five ``: 0`` lines, "Backfill complete", exit 0. That is the repo's
    fail-closed-is-not-fail-visible trap -- a mistyped path that happens to land
    on some other SQLite file reports success. The existence check above cannot
    help; the file is right there.

    Asserted on stderr, and on the *completion* line changing too: a warning
    that scrolls past above an unchanged "Backfill complete." is still a
    misleading final answer.
    """
    empty_source = tmp_path / "empty.db"
    BacktestDatabase(db_path=empty_source)  # creates the schema, inserts nothing

    from dashboard.scripts import backfill_runs_to_postgres

    monkeypatch.delenv("AGENT_RUNS_DATABASE_URL", raising=False)
    monkeypatch.setattr(
        sys, "argv",
        ["backfill_runs_to_postgres.py", "--source", str(empty_source), "--dry-run"],
    )

    assert backfill_runs_to_postgres.main() == 0

    captured = capsys.readouterr()
    assert "WARNING: every source table is empty" in captured.err
    assert str(empty_source.resolve()) in captured.err
    # Not the default seed path, so the likely-typo hint must appear as well.
    assert "not the default seed path" in captured.err
    assert "nothing to write" in captured.out


def test_non_empty_source_does_not_warn(tmp_path, monkeypatch, capsys):
    """The other half of the pair: the warning must not cry wolf on a real
    source, or operators will learn to ignore it."""
    source_path = _build_source_db(tmp_path)

    from dashboard.scripts import backfill_runs_to_postgres

    monkeypatch.delenv("AGENT_RUNS_DATABASE_URL", raising=False)
    monkeypatch.setattr(
        sys, "argv",
        ["backfill_runs_to_postgres.py", "--source", str(source_path), "--dry-run"],
    )

    assert backfill_runs_to_postgres.main() == 0

    captured = capsys.readouterr()
    assert "every source table is empty" not in captured.err
    assert "Dry run: no writes performed." in captured.out


# --- a fake target: drives the whole script, failure paths included -----------

class _FakeCursor:
    """Routes the handful of raw statements the script issues itself.

    Everything else reaches the target through its public writer methods, so
    this only needs the advisory-lock calls and _restore_created_at's UPDATE.
    Unknown SQL raises rather than silently succeeding -- otherwise a future
    statement added to the script would be "covered" by a fake that ignored it.
    """

    def __init__(self, target):
        self._target = target
        self._row = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def execute(self, sql, params=()):
        statement = " ".join(sql.split())
        if statement.startswith("SELECT pg_try_advisory_lock"):
            self._target.lock_acquire_calls.append(params[0])
            self._row = {"acquired": self._target.lock_available}
        elif statement.startswith("SELECT pg_advisory_unlock"):
            self._target.lock_release_calls.append(params[0])
            self._row = {"pg_advisory_unlock": True}
        elif statement.startswith("UPDATE agent_runs SET created_at"):
            created_at, run_id = params
            if run_id in self._target.raise_on_created_at:
                raise RuntimeError(f"simulated created_at failure for {run_id}")
            self._target.runs[run_id]["created_at"] = created_at
            self._row = None
        else:
            raise AssertionError(f"_FakeCursor got unexpected SQL: {statement}")

    def fetchone(self):
        return self._row


class _FakeConnection:
    def __init__(self, target):
        self._target = target

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def cursor(self):
        return _FakeCursor(self._target)

    def commit(self):
        """Counted so the lock's commits are assertable.

        The advisory-lock helper commits right after acquiring, specifically so
        the lock connection does not sit ``idle in transaction`` for the whole
        write phase (managed providers enforce
        idle_in_transaction_session_timeout, which would kill the session and
        silently drop the lock). That commit is load-bearing, so it is recorded
        rather than swallowed by a no-op stub.
        """
        self._target.commits += 1

    def transaction(self):
        """Stand-in for psycopg's ``conn.transaction()`` block.

        Counted, not simulated: on a real connection this is what keeps one
        failed UPDATE from poisoning the rest of the batch, which no fake can
        demonstrate. What it *can* pin is that the script opens one per run
        rather than one for the whole loop -- the actual shape of the fix.

        (On a real pooled connection each of these is an independent
        ``BEGIN``/``COMMIT``, not a ``SAVEPOINT`` -- psycopg branches on whether
        a transaction is already open, and here none is. See
        ``_restore_created_at``'s docstring.)
        """
        # Bound before the class so its methods can keep the conventional
        # ``self`` name (CodeQL py/not-named-self) without shadowing the
        # connection's own ``self``.
        target = self._target

        class _Txn:
            def __enter__(self):
                target.transaction_entries += 1
                return self

            def __exit__(self, *exc_info):
                return False

        return _Txn()


class _FakeTarget:
    """In-memory stand-in for PostgresBacktestDatabase.

    Deliberately not a MagicMock: the accounting assertions need real row
    counts, and the whole point of these tests is what the script reports about
    rows that did and did not land.
    """

    def __init__(self, *, fail_runs=(), lock_available=True, raise_on_created_at=()):
        self.fail_runs = set(fail_runs)
        self.lock_available = lock_available
        self.raise_on_created_at = set(raise_on_created_at)

        self.runs = {}
        self.equity = {}
        self.trades = {}
        self.decisions = {}
        self.manifests = {}
        self.baselines = {}

        self.lock_acquire_calls = []
        self.lock_release_calls = []
        self.transaction_entries = 0
        self.commits = 0

    def _get_connection(self):
        return _FakeConnection(self)

    def insert_run(self, *, run_id, **kwargs):
        if run_id in self.fail_runs:
            raise RuntimeError(f"simulated insert_run failure for {run_id}")
        self.runs[run_id] = {"run_id": run_id, **kwargs}

    def update_run_baselines(self, run_id, *, djia_run_id=None, buyhold_run_id=None):
        self.baselines[run_id] = (djia_run_id, buyhold_run_id)

    def insert_equity_points(self, run_id, points, replace=True):
        self.equity[run_id] = list(points)

    def insert_trades(self, run_id, trades):
        self.trades.setdefault(run_id, []).extend(trades)

    def insert_decisions(self, run_id, decisions):
        self.decisions.setdefault(run_id, []).extend(decisions)

    def insert_run_manifest(self, run_id, manifest):
        self.manifests[run_id] = manifest

    def get_trades(self, run_id):
        return list(self.trades.get(run_id, []))

    def get_decisions(self, run_id):
        return list(self.decisions.get(run_id, []))


def _run_main_against(monkeypatch, source_path, target):
    """Point ``main()`` at ``target`` and run it, with no Postgres involved.

    ``main()`` does its ``from dashboard.backend.database_postgres import
    PostgresBacktestDatabase`` *inside* the function body, so the name is looked
    up on that module at call time -- patching it there is what makes this work
    (patching the script's own namespace would be a no-op, since the name is
    never bound there).
    """
    import dashboard.backend.database_postgres as database_pg_module
    from dashboard.scripts import backfill_runs_to_postgres

    monkeypatch.setattr(
        database_pg_module, "PostgresBacktestDatabase", lambda url: target
    )
    monkeypatch.setenv(
        "AGENT_RUNS_DATABASE_URL", "postgresql://u:p@example.invalid/atl"
    )
    monkeypatch.setattr(
        sys, "argv",
        ["backfill_runs_to_postgres.py", "--source", str(source_path)],
    )
    return backfill_runs_to_postgres.main()


def test_one_poison_row_does_not_stall_the_whole_backfill(tmp_path, monkeypatch, capsys):
    """The finding this exists for: before per-run isolation, one bad legacy row
    propagated out of ``main()``, so every run *after* it in source order never
    migrated -- and because the order is fixed, every rerun re-failed on the
    same row. "Idempotent" held; "re-runnable to completion" did not.

    ``run-baseline`` sorts first (older created_at), so failing it also proves
    the loop continues past a failure rather than merely tolerating one at the
    end.
    """
    source_path = _build_source_db(tmp_path)
    target = _FakeTarget(fail_runs={"run-baseline"})

    exit_code = _run_main_against(monkeypatch, source_path, target)

    # The healthy run migrated in full, despite the failure ahead of it.
    assert "run-full" in target.runs
    assert len(target.equity["run-full"]) == 2
    assert len(target.trades["run-full"]) == 2
    assert len(target.decisions["run-full"]) == 2
    assert target.manifests["run-full"] == {"symbols": ["AAPL"], "version": 1}

    # The poison row landed nowhere.
    assert "run-baseline" not in target.runs

    captured = capsys.readouterr()
    # created_at is not "restored" for a run that has no row to update. The
    # UPDATE would match zero rows and raise nothing, so an un-skipped loop
    # would report 2/2 restored while only one run exists in the target.
    assert target.transaction_entries == 1
    assert "restored from source for 1/2 run(s)" in captured.out

    err = captured.err
    assert "per-run failure(s)" in err
    assert "agent_runs / run-baseline" in err
    assert "simulated insert_run failure" in err
    # Non-zero, so a wrapper script or CI step cannot mistake this for success.
    assert exit_code == 1


def test_children_of_a_failed_parent_are_skipped_and_accounted_not_attempted(
    tmp_path, monkeypatch, capsys
):
    """A failed parent must not produce a second, misleading failure per child.

    The FK would reject the children anyway, so attempting them would fill the
    report with consequences instead of the cause. They are counted in
    ``skipped_parent_failed`` rather than dropped, so the per-table accounting
    still balances -- an unexplained shortfall is the one thing that check is
    for, and burying real ones under expected ones would blind it.
    """
    source_path = _build_source_db(tmp_path)
    target = _FakeTarget(fail_runs={"run-full"})  # the run that owns children

    exit_code = _run_main_against(monkeypatch, source_path, target)

    assert "run-full" not in target.runs
    assert "run-full" not in target.equity
    assert "run-full" not in target.trades
    assert "run-full" not in target.decisions
    assert "run-full" not in target.manifests

    captured = capsys.readouterr()
    # Exactly one failure -- the cause -- not one per child table.
    assert captured.err.count("per-run failure(s)") == 1
    assert "agent_runs / run-full" in captured.err
    assert "trades / run-full" not in captured.err
    assert "backtest_decisions / run-full" not in captured.err

    assert "skipped_parent_failed" in captured.out
    # Accounting must still balance: no unexplained-shortfall warning.
    assert "accounting mismatch" not in captured.err
    assert exit_code == 1


def test_restore_created_at_uses_one_transaction_per_run_and_isolates_a_failure(
    tmp_path, monkeypatch, capsys
):
    """``_restore_created_at`` shares one connection across every run, so a bare
    try/except would not isolate anything: on Postgres the first failed
    statement aborts the transaction and every later UPDATE raises
    ``InFailedSqlTransaction``. One bad row would look like every row being bad.

    Two assertions, because they fail for different reasons: the block count
    catches someone hoisting ``conn.transaction()`` out of the loop, and the
    surviving restore catches the isolation being lost some other way.
    """
    source_path = _build_source_db(tmp_path)
    target = _FakeTarget(raise_on_created_at={"run-baseline"})

    exit_code = _run_main_against(monkeypatch, source_path, target)

    assert target.transaction_entries == 2, "one transaction block per run, not one per batch"
    # run-baseline's restore failed; run-full's still happened.
    assert "created_at" in target.runs["run-full"]

    err = capsys.readouterr().err
    assert "agent_runs.created_at / run-baseline" in err
    assert exit_code == 1


def test_second_concurrent_invocation_refuses_instead_of_doubling_rows(
    tmp_path, monkeypatch, capsys
):
    """The trades/decisions skip-logic is a read-then-write across two separate
    pooled checkouts, so two concurrent runs both see "nothing there yet" and
    both append -- and neither table has a unique key to collapse the result.

    Asserting nothing was written matters as much as the exit code: refusing
    *after* a partial write would be worse than not refusing at all.
    """
    source_path = _build_source_db(tmp_path)
    target = _FakeTarget(lock_available=False)

    exit_code = _run_main_against(monkeypatch, source_path, target)

    assert exit_code == 1
    assert target.runs == {}
    assert target.trades == {}
    assert "another backfill_runs_to_postgres.py run holds the advisory lock" in (
        capsys.readouterr().err
    )


def test_advisory_lock_is_released_even_when_runs_fail(tmp_path, monkeypatch, capsys):
    """Pooled connections are reused, so a lock left behind travels into
    whatever checks that connection out next -- and would then block every
    later backfill on a database that is not actually busy. The release has to
    survive the failure path, not just the happy one.
    """
    source_path = _build_source_db(tmp_path)
    target = _FakeTarget(fail_runs={"run-full", "run-baseline"})

    _run_main_against(monkeypatch, source_path, target)

    from dashboard.scripts import backfill_runs_to_postgres

    key = backfill_runs_to_postgres.BACKFILL_ADVISORY_LOCK_KEY
    assert target.lock_acquire_calls == [key]
    assert target.lock_release_calls == [key], "the lock must be released on the failure path"
    # Acquire and release are each followed by a commit, so the lock connection
    # never sits `idle in transaction` across the write phase.
    assert target.commits >= 2


def test_corrupt_manifest_json_is_reported_not_fatal(tmp_path, monkeypatch, capsys):
    """A read-phase failure, which ``_Failures`` cannot reach.

    ``SourceData.__init__`` reads everything up front, before any write and
    before the failure collector exists. Every other reader tolerates junk
    already (``_parse_run_row`` catches TypeError/ValueError, ``get_decisions``
    catches JSONDecodeError) but ``BacktestDatabase.get_run_manifest`` does a
    bare ``json.loads``, so one corrupt row aborted the entire backfill before
    a single write -- the one place where "re-runnable to completion" could not
    possibly be true, since no rerun gets past it either.

    The corruption is written with raw sqlite3 because no public method can
    produce it (``insert_run_manifest`` always ``json.dumps``), which is also
    why this only happens with a hand-edited or legacy ``--source``.
    """
    source_path = _build_source_db(tmp_path)
    conn = sqlite3.connect(str(source_path))
    try:
        conn.execute(
            "UPDATE run_manifest SET manifest_json = ? WHERE run_id = ?",
            ("{not valid json", "run-full"),
        )
        conn.commit()
    finally:
        conn.close()

    target = _FakeTarget()
    exit_code = _run_main_against(monkeypatch, source_path, target)

    captured = capsys.readouterr()
    # Everything else still migrated -- the read did not abort the run.
    assert "run-full" in target.runs
    assert len(target.trades["run-full"]) == 2
    assert target.manifests == {}, "the unreadable manifest was skipped, not guessed at"

    assert "manifest(s) could not be read" in captured.err
    assert "run_manifest / run-full" in captured.err
    # Counted in its own bucket, so it is a named reason rather than an
    # unexplained shortfall.
    assert "skipped_unreadable=1" in captured.out
    assert "accounting mismatch" not in captured.err
    assert exit_code == 1


def test_null_manifest_json_is_accounted_not_an_unexplained_shortfall(
    tmp_path, monkeypatch, capsys
):
    """The sibling of the corrupt-manifest case, and a genuinely separate branch.

    ``manifest_json`` holding the JSON literal ``null`` decodes to Python
    ``None`` *without raising*, so it never reaches the try/except above --
    ``get_run_manifest`` returns None exactly as it does for "no row at all",
    and the public API cannot distinguish them. Left unhandled, the row is
    counted in ``raw_counts``, lands in no bucket, and reports as an
    *unexplained* accounting mismatch, which is the single signal that check
    exists to raise. Asserting the absence of "accounting mismatch" is therefore
    the real assertion here, not the presence of the message.
    """
    source_path = _build_source_db(tmp_path)
    conn = sqlite3.connect(str(source_path))
    try:
        conn.execute(
            "UPDATE run_manifest SET manifest_json = ? WHERE run_id = ?",
            ("null", "run-full"),
        )
        conn.commit()
    finally:
        conn.close()

    target = _FakeTarget()
    exit_code = _run_main_against(monkeypatch, source_path, target)

    captured = capsys.readouterr()
    assert "run-full" in target.runs, "everything else still migrated"
    assert target.manifests == {}

    assert "decodes to null" in captured.err
    assert "skipped_unreadable=1" in captured.out
    assert "accounting mismatch" not in captured.err
    assert exit_code == 1


def test_clean_run_reports_success_and_balances_the_accounting(
    tmp_path, monkeypatch, capsys
):
    """The control case for every test above: with nothing wrong, none of the
    new failure machinery may fire. Without this, a bug that reported failures
    unconditionally would still pass all of them.
    """
    source_path = _build_source_db(tmp_path)
    target = _FakeTarget()

    exit_code = _run_main_against(monkeypatch, source_path, target)

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Backfill complete." in captured.out
    assert "accounting mismatch" not in captured.err
    assert "per-run failure(s)" not in captured.err
    assert "FAILED=" not in captured.out
    assert target.baselines["run-full"] == ("run-baseline", None)


# --- live-Postgres backfill + idempotency -------------------------------------

@pytest.fixture
def pg_backtest_db():
    require_local_postgres_url(TEST_POSTGRES_URL)
    # ``import ... as`` + attribute access, not ``from ... import``: the
    # monkeypatch seam in _run_main_against needs the module object, and mixing
    # both forms for one module trips CodeQL py/import-and-import-from.
    import dashboard.backend.database_postgres as database_pg_module

    store = database_pg_module.PostgresBacktestDatabase(TEST_POSTGRES_URL)
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
