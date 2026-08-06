"""PostgresBacktestDatabase dispatch tests (run-history backend factory).

Two tiers, mirroring test_agent_store_postgres.py:
1. Dispatch-logic tests (no live Postgres needed) -- verify _build_backtest_db
   picks the right database class based on AGENT_RUNS_DATABASE_URL.
2. Behavioral tests against a real Postgres -- skipped unless TEST_POSTGRES_URL
   is set. Point it at a throwaway database, e.g.:
     docker run --rm -e POSTGRES_PASSWORD=test -e POSTGRES_DB=atl_test \
       -p 5433:5432 postgres:18-alpine
     export TEST_POSTGRES_URL=postgresql://postgres:test@localhost:5433/atl_test

Task 9 appends the @pg_only behavioral half below the dispatch tests in this
same file.
"""

import os

import pytest

from dashboard.backend.tests._postgres_testing import require_local_postgres_url

TEST_POSTGRES_URL = os.getenv("TEST_POSTGRES_URL")

pg_only = pytest.mark.skipif(
    not TEST_POSTGRES_URL,
    reason="TEST_POSTGRES_URL not set; skipping live-Postgres tests",
)


# --- dispatch tests (backtest db) --------------------------------------------

def test_build_backtest_db_defaults_to_sqlite(monkeypatch, capsys):
    import dashboard.backend.database as database_module

    monkeypatch.delenv("AGENT_RUNS_DATABASE_URL", raising=False)
    store = database_module._build_backtest_db()

    assert isinstance(store, database_module.BacktestDatabase)
    assert (
        "run history backend: sqlite (ephemeral on Render)"
        in capsys.readouterr().out
    )


def test_build_backtest_db_picks_postgres_when_url_set(monkeypatch, capsys):
    import dashboard.backend.database as database_module
    import dashboard.backend.database_postgres as database_pg_module

    created = {}

    class FakePostgresBacktestDatabase:
        def __init__(self, database_url):
            created["database_url"] = database_url

    # _build_backtest_db imports PostgresBacktestDatabase *inside the function*
    # from database_postgres, so the name is never bound on database_module --
    # patching it there would be a no-op. Patch the source module instead.
    monkeypatch.setattr(
        database_pg_module, "PostgresBacktestDatabase", FakePostgresBacktestDatabase
    )
    monkeypatch.setenv("AGENT_RUNS_DATABASE_URL", "postgresql://fake/db")

    store = database_module._build_backtest_db()

    assert isinstance(store, FakePostgresBacktestDatabase)
    assert created["database_url"] == "postgresql://fake/db"
    assert "run history backend: postgres (fake/db)" in capsys.readouterr().out


def test_build_backtest_db_ignores_content_and_users_database_url(monkeypatch, capsys):
    """AGENT_RUNS_DATABASE_URL is the ONLY var allowed to select the Postgres
    run-history backend. CONTENT_DATABASE_URL (agents/versions/strategies) and
    USERS_DATABASE_URL (accounts) are scoped to their own stores and must never
    leak into this decision (spec, Decision 3) -- neither may substitute for
    AGENT_RUNS_DATABASE_URL, not even when it is unset.

    This is the no-fallback-chain guarantee: it is what would catch someone
    later "simplifying" the factory into falling back to a sibling database's
    URL, a one-line change that reads like an improvement and keeps the rest
    of the suite green while silently binding run history to the wrong
    database. Both siblings are set here and SQLite must still be chosen.
    """
    import dashboard.backend.database as database_module

    monkeypatch.delenv("AGENT_RUNS_DATABASE_URL", raising=False)
    monkeypatch.setenv("CONTENT_DATABASE_URL", "postgresql://fake/content")
    monkeypatch.setenv("USERS_DATABASE_URL", "postgresql://fake/users")

    store = database_module._build_backtest_db()

    assert isinstance(store, database_module.BacktestDatabase)
    assert (
        "run history backend: sqlite (ephemeral on Render)"
        in capsys.readouterr().out
    )


def test_build_backtest_db_never_prints_the_credentials(monkeypatch, capsys):
    """The printed line is the design's only misconfiguration tripwire (see
    describe_database_url in db_url.py) -- assert BOTH halves: the secret is
    absent, AND the exact host/db line is present. Asserting only absence
    would keep passing even if the whole line silently disappeared, which
    would delete the tripwire without any test noticing.
    """
    import dashboard.backend.database as database_module
    import dashboard.backend.database_postgres as database_pg_module

    class FakePostgresBacktestDatabase:
        def __init__(self, database_url):
            pass

    monkeypatch.setattr(
        database_pg_module, "PostgresBacktestDatabase", FakePostgresBacktestDatabase
    )
    monkeypatch.setenv(
        "AGENT_RUNS_DATABASE_URL", "postgresql://admin:sup3r-s3cret@host/db"
    )

    database_module._build_backtest_db()

    out = capsys.readouterr().out
    assert "sup3r-s3cret" not in out
    assert "run history backend: postgres (host/db)" in out


def test_unreachable_postgres_raises_instead_of_falling_back():
    """Fail loud: a set-but-unreachable AGENT_RUNS_DATABASE_URL must not
    silently degrade to ephemeral SQLite. A silent fallback here would be
    exactly the "absent vs. broken" failure shape CLAUDE.md's "Fail-closed is
    not fail-visible" section warns about -- run history would just look
    empty, with nothing in the logs saying why.

    What this proves is "it fails loudly", not "psycopg raises one particular
    class": the pool checkout (db_pool.py) actually raises
    psycopg_pool.PoolTimeout, a subclass of psycopg.OperationalError, so a
    future psycopg upgrade that changes the concrete subclass should not read
    as a product regression here -- only a change in fail-loud-vs-silent
    behavior should.

    Timing: db_pool.get_pool() retries the connection for
    POOL_TIMEOUT_SECONDS (prod default 10s) before giving up.
    conftest.py's autouse `_reset_shared_scale_state` fixture already
    monkeypatches that constant down to 1.0s for every test in this suite, so
    this raises in ~1s here rather than paying the full prod timeout -- no
    additional patching needed in this test.
    """
    import psycopg

    # Module-alias form, not `from ... import PostgresBacktestDatabase`, in every
    # test below: the dispatch tests above *must* have the module object to
    # monkeypatch the attribute, and importing one module both ways in a single
    # file trips CodeQL's py/import-and-import-from. Keep this file uniform.
    import dashboard.backend.database_postgres as database_pg_module

    with pytest.raises(psycopg.OperationalError):
        database_pg_module.PostgresBacktestDatabase(
            "postgresql://u:p@127.0.0.1:1/nope?connect_timeout=2"
        )


def test_malformed_url_is_rejected_before_psycopg_can_echo_it():
    """A typo'd AGENT_RUNS_DATABASE_URL must not put the password in the log.

    psycopg parses anything not starting with postgresql:// as a keyword DSN
    and quotes the whole input back ('missing "=" after "<the entire URL>"').
    This runs at import time with no try/except, so that message is the boot
    failure and it lands in Render's log. require_postgres_url must therefore
    run before psycopg ever sees the value.
    """
    import dashboard.backend.database_postgres as database_pg_module

    with pytest.raises(ValueError) as excinfo:
        database_pg_module.PostgresBacktestDatabase(
            '"postgresql://u:sup3r-s3cret@ep-x.neon.tech/atl"'
        )
    assert "sup3r-s3cret" not in str(excinfo.value)


# --- live-Postgres behavioral tests (backtest db) ----------------------------
#
# Round-trip through public PostgresBacktestDatabase methods only -- never raw
# SQL with sqlite3-style `?` placeholders copied from the SQLite-era test
# files. Every behavior asserted below was confirmed against the current
# dashboard/backend/database.py source (not just the plan's reference doc,
# which predates Task 2's run_manifest cleanup and is stale on delete_run /
# clear_all's table list -- see task-9-report.md).

@pytest.fixture
def pg_backtest_db():
    require_local_postgres_url(TEST_POSTGRES_URL)
    import dashboard.backend.database_postgres as database_pg_module

    store = database_pg_module.PostgresBacktestDatabase(TEST_POSTGRES_URL)
    with store._get_connection() as conn:
        with conn.cursor() as cur:
            # equity_timeseries/trades/backtest_decisions carry a real FK to
            # agent_runs (ON DELETE CASCADE) and must go first; run_manifest
            # has no FK at all (see database_postgres.py) so its position
            # here doesn't matter for integrity, it's just grouped with the
            # other per-run tables before the agent_runs parent row
            cur.execute("DELETE FROM equity_timeseries")
            cur.execute("DELETE FROM trades")
            cur.execute("DELETE FROM backtest_decisions")
            cur.execute("DELETE FROM run_manifest")
            cur.execute("DELETE FROM agent_runs")
    yield store


@pg_only
def test_insert_run_round_trips_through_every_reader_postgres(pg_backtest_db):
    """Insert once, read back through six different accessor methods -- proves
    a real round trip through Postgres (persistence), not an echo of the
    insert call's own arguments. insert_run itself returns nothing, so every
    assertion here necessarily goes through a distinct reader."""
    metadata = {"llm_max_output_tokens": 4096}
    pg_backtest_db.insert_run(
        run_id="run-1",
        session_id="session-1",
        agent_name="Agent One",
        mode="backtest",
        start_date="2024-01-01",
        end_date="2024-01-31",
        initial_equity=10000.0,
        final_equity=10500.0,
        total_return=0.05,
        sharpe_ratio=1.2,
        max_drawdown=-0.03,
        num_trades=4,
        llm_model="claude-sonnet",
        llm_calls=8,
        input_tokens=1000,
        output_tokens=200,
        est_cost_usd=0.42,
        metadata=metadata,
    )

    by_id = pg_backtest_db.get_run("run-1")
    assert by_id["agent_name"] == "Agent One"
    assert by_id["metadata"] == metadata  # parsed back to a dict, not raw JSON text

    all_runs = pg_backtest_db.get_all_runs()
    assert [r["run_id"] for r in all_runs] == ["run-1"]

    by_session = pg_backtest_db.get_runs_by_session("session-1")
    assert [r["run_id"] for r in by_session] == ["run-1"]

    by_sessions = pg_backtest_db.get_runs_by_sessions(["session-1", "no-such-session"])
    assert [r["run_id"] for r in by_sessions["session-1"]] == ["run-1"]
    assert by_sessions["no-such-session"] == []

    with_session = pg_backtest_db.get_run_with_session("run-1", "session-1")
    assert with_session["run_id"] == "run-1"
    assert pg_backtest_db.get_run_with_session("run-1", "wrong-session") is None

    by_mode = pg_backtest_db.get_runs_by_mode("backtest")
    assert [r["run_id"] for r in by_mode] == ["run-1"]
    assert pg_backtest_db.get_runs_by_mode("paper") == []


@pg_only
def test_insert_run_upsert_preserves_created_at_advances_updated_at_postgres(pg_backtest_db):
    """The copy-paste trap the plan calls out at database_postgres.py's
    insert_run docstring: a naive 1:1 translation of SQLite's INSERT OR
    REPLACE (a DELETE+INSERT that names neither timestamp) resets BOTH
    columns on every re-insert. This twin deliberately diverges in opposite
    directions per column -- created_at survives (first-seen semantics),
    updated_at is always refreshed -- so both halves are asserted
    independently rather than as one combined check.

    Backdates both columns via a raw UPDATE to a fixed, far-past instant
    instead of comparing two insert_run calls' wall-clock timestamps: the DB
    stamps these to *second* granularity (to_char(now()...)), so two calls
    issued within the same wall-clock second would make the "updated_at
    advanced" half flaky. Backdating first makes the comparison deterministic
    regardless of timing.
    """
    pg_backtest_db.insert_run(
        run_id="run-ts", session_id="s1", agent_name="Agent", mode="backtest",
        start_date="2024-01-01", end_date="2024-01-02", initial_equity=1000.0,
    )

    backdated = "2020-01-01 00:00:00"
    with pg_backtest_db._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE agent_runs SET created_at = %s, updated_at = %s "
                "WHERE run_id = %s",
                (backdated, backdated, "run-ts"),
            )

    before = pg_backtest_db.get_run("run-ts")
    assert before["created_at"] == backdated
    assert before["updated_at"] == backdated

    # Re-insert (the ON CONFLICT (run_id) DO UPDATE path) with a changed field.
    pg_backtest_db.insert_run(
        run_id="run-ts", session_id="s1", agent_name="Agent", mode="backtest",
        start_date="2024-01-01", end_date="2024-01-02", initial_equity=1000.0,
        final_equity=1234.0,
    )

    after = pg_backtest_db.get_run("run-ts")
    assert after["final_equity"] == 1234.0
    assert after["created_at"] == backdated, "created_at must survive a re-insert"
    assert after["updated_at"] != backdated, "updated_at must advance on a re-insert"
    assert after["updated_at"] > backdated  # fixed-width YYYY-MM-DD HH24:MI:SS sorts lexically


@pg_only
def test_insert_run_upsert_preserves_baseline_links_postgres(pg_backtest_db):
    """Divergence 3 (module docstring): a re-insert must NOT drop the baseline
    links, where SQLite's INSERT OR REPLACE nulls them.

    Same root cause as created_at above -- ON CONFLICT DO UPDATE only touches
    the columns it names, and these two appear in neither the INSERT list nor
    DO UPDATE SET -- but the consequence is the opposite sign, so it gets its
    own case rather than an extra assert on that one.

    Pinned rather than left implicit because "make the twin behave exactly like
    SQLite" is a plausible-sounding future change that would silently break the
    failure path this preserves: baseline_worker catches its own exceptions, so
    on SQLite a rerun whose baseline regeneration fails leaves the run's
    comparison chart permanently unlinked. Here the previous links survive.
    """
    pg_backtest_db.insert_run(
        run_id="run-bl", session_id="s1", agent_name="Agent", mode="backtest",
        start_date="2024-01-01", end_date="2024-01-02", initial_equity=1000.0,
    )
    pg_backtest_db.update_run_baselines(
        "run-bl", djia_run_id="djia-1", buyhold_run_id="bah-1"
    )
    assert pg_backtest_db.get_run("run-bl")["baseline_djia_run_id"] == "djia-1"

    # The rerun: same run_id, no baseline params (insert_run has none), and
    # crucially no update_run_baselines call afterwards -- i.e. the case where
    # baseline regeneration failed.
    pg_backtest_db.insert_run(
        run_id="run-bl", session_id="s1", agent_name="Agent", mode="backtest",
        start_date="2024-01-01", end_date="2024-01-02", initial_equity=1000.0,
        final_equity=1111.0,
    )

    after = pg_backtest_db.get_run("run-bl")
    assert after["final_equity"] == 1111.0, "the re-insert really did take effect"
    assert after["baseline_djia_run_id"] == "djia-1"
    assert after["baseline_buyhold_run_id"] == "bah-1"


@pg_only
def test_insert_run_reinsert_does_not_cascade_delete_equity_children_postgres(pg_backtest_db):
    """The FK-sensitive upsert case the brief calls mandatory: an external
    run's finalize step (and the leaderboard refresh path) can call
    insert_run again for a run_id that already has equity rows. Because the
    upsert is ON CONFLICT DO UPDATE -- an actual UPDATE, not SQLite's
    DELETE+INSERT -- the equity_timeseries rows (ON DELETE CASCADE back to
    agent_runs) must survive the re-insert untouched.
    """
    pg_backtest_db.insert_run(
        run_id="run-cascade", session_id="s1", agent_name="Agent", mode="backtest",
        start_date="2024-01-01", end_date="2024-01-02", initial_equity=1000.0,
    )
    pg_backtest_db.insert_equity_points(
        "run-cascade",
        [
            {"timestamp": "2024-01-01T00:00:00", "equity": 1000.0, "cash": 1000.0, "positions_value": 0.0},
            {"timestamp": "2024-01-01T01:00:00", "equity": 1010.0, "cash": 990.0, "positions_value": 20.0},
        ],
    )

    # Re-insert: this must NOT delete-then-recreate the agent_runs row.
    pg_backtest_db.insert_run(
        run_id="run-cascade", session_id="s1", agent_name="Agent", mode="backtest",
        start_date="2024-01-01", end_date="2024-01-02", initial_equity=1000.0,
        final_equity=1010.0,
    )

    curve = pg_backtest_db.get_equity_curve("run-cascade")
    assert len(curve) == 2
    assert pg_backtest_db.get_run("run-cascade")["final_equity"] == 1010.0


@pg_only
def test_insert_run_manifest_upsert_postgres(pg_backtest_db):
    """The third upsert path: ON CONFLICT (run_id) DO UPDATE SET
    manifest_json = EXCLUDED.manifest_json. Without it, a re-insert would
    raise a duplicate-key error on the run_manifest PK instead of replacing
    the stored manifest."""
    pg_backtest_db.insert_run(
        run_id="run-manifest", session_id="s1", agent_name="Agent", mode="backtest",
        start_date="2024-01-01", end_date="2024-01-02", initial_equity=1000.0,
    )
    pg_backtest_db.insert_run_manifest("run-manifest", {"symbols": ["AAPL"], "version": 1})
    assert pg_backtest_db.get_run_manifest("run-manifest") == {"symbols": ["AAPL"], "version": 1}

    pg_backtest_db.insert_run_manifest("run-manifest", {"symbols": ["AAPL", "MSFT"], "version": 2})
    assert pg_backtest_db.get_run_manifest("run-manifest") == {
        "symbols": ["AAPL", "MSFT"], "version": 2
    }


@pg_only
def test_insert_equity_point_duplicate_timestamp_updates_not_duplicates_postgres(pg_backtest_db):
    """The mandatory uniqueness-constraint test: the only thing standing in
    for the missing static guard on uq_equity_timeseries_run_timestamp (see
    the "NOTHING STATIC GUARDS THIS LINE" comment in database_postgres.py's
    _init_schema). A dropped copy of that CREATE UNIQUE INDEX statement is
    invisible to every static parity check and would only surface here, as a
    duplicate row instead of an update.
    """
    pg_backtest_db.insert_run(
        run_id="run-dup", session_id="s1", agent_name="Agent", mode="backtest",
        start_date="2024-01-01", end_date="2024-01-02", initial_equity=1000.0,
    )
    ts = "2024-01-01T00:00:00"
    pg_backtest_db.insert_equity_point("run-dup", ts, equity=1000.0, cash=1000.0, positions_value=0.0)
    pg_backtest_db.insert_equity_point("run-dup", ts, equity=2000.0, cash=1500.0, positions_value=500.0)

    curve = pg_backtest_db.get_equity_curve("run-dup")
    assert len(curve) == 1
    assert curve[0]["equity"] == 2000.0
    assert curve[0]["cash"] == 1500.0


@pg_only
def test_insert_equity_points_rerun_with_shorter_curve_drops_leftovers_postgres(pg_backtest_db):
    """insert_equity_points' delete-then-insert (replace=True, the default):
    a rerun that produces fewer bars than the previous curve must not leave
    the old curve's extra timestamps spliced in."""
    pg_backtest_db.insert_run(
        run_id="run-shrink", session_id="s1", agent_name="Agent", mode="backtest",
        start_date="2024-01-01", end_date="2024-01-02", initial_equity=1000.0,
    )
    long_curve = [
        {
            "timestamp": f"2024-01-01T0{i}:00:00",
            "equity": 1000.0 + i,
            "cash": 1000.0,
            "positions_value": 0.0,
        }
        for i in range(3)
    ]
    pg_backtest_db.insert_equity_points("run-shrink", long_curve)
    assert len(pg_backtest_db.get_equity_curve("run-shrink")) == 3

    short_curve = [long_curve[0]]
    pg_backtest_db.insert_equity_points("run-shrink", short_curve)  # replace=True default

    curve = pg_backtest_db.get_equity_curve("run-shrink")
    assert [c["timestamp"] for c in curve] == [long_curve[0]["timestamp"]]


@pg_only
def test_insert_equity_points_replace_false_appends_postgres(pg_backtest_db):
    pg_backtest_db.insert_run(
        run_id="run-append", session_id="s1", agent_name="Agent", mode="backtest",
        start_date="2024-01-01", end_date="2024-01-02", initial_equity=1000.0,
    )
    first = [{"timestamp": "2024-01-01T00:00:00", "equity": 1000.0, "cash": 1000.0, "positions_value": 0.0}]
    second = [{"timestamp": "2024-01-01T01:00:00", "equity": 1010.0, "cash": 990.0, "positions_value": 20.0}]

    pg_backtest_db.insert_equity_points("run-append", first)
    pg_backtest_db.insert_equity_points("run-append", second, replace=False)

    curve = pg_backtest_db.get_equity_curve("run-append")
    assert [c["timestamp"] for c in curve] == [first[0]["timestamp"], second[0]["timestamp"]]


@pg_only
def test_insert_equity_points_empty_list_is_noop_not_a_wipe_postgres(pg_backtest_db):
    """A failed/empty rerun must not erase the curve already on the board."""
    pg_backtest_db.insert_run(
        run_id="run-noop", session_id="s1", agent_name="Agent", mode="backtest",
        start_date="2024-01-01", end_date="2024-01-02", initial_equity=1000.0,
    )
    points = [{"timestamp": "2024-01-01T00:00:00", "equity": 1000.0, "cash": 1000.0, "positions_value": 0.0}]
    pg_backtest_db.insert_equity_points("run-noop", points)
    assert len(pg_backtest_db.get_equity_curve("run-noop")) == 1

    pg_backtest_db.insert_equity_points("run-noop", [])

    assert len(pg_backtest_db.get_equity_curve("run-noop")) == 1


@pg_only
def test_get_equity_curve_strips_null_audit_fields_postgres(pg_backtest_db):
    """A row where e.g. fx_rate IS NULL comes back without an fx_rate key at
    all (not fx_rate: None) -- downstream code keys off presence, not value,
    to decide whether a run has currency-audit data."""
    pg_backtest_db.insert_run(
        run_id="run-audit", session_id="s1", agent_name="Agent", mode="backtest",
        start_date="2024-01-01", end_date="2024-01-02", initial_equity=1000.0,
    )
    pg_backtest_db.insert_equity_point(
        "run-audit", "2024-01-01T00:00:00", equity=1000.0, cash=1000.0, positions_value=0.0,
        native_equity=7200.0,  # only native_equity provided; the rest stay NULL
    )

    curve = pg_backtest_db.get_equity_curve("run-audit")
    assert len(curve) == 1
    point = curve[0]
    assert point["native_equity"] == 7200.0
    assert "native_cash" not in point
    assert "native_positions_value" not in point
    assert "fx_rate" not in point


@pg_only
def test_delete_run_cascades_children_and_leaves_idempotency_orphaned_postgres(pg_backtest_db):
    """FK cascade covers equity_timeseries/trades/backtest_decisions;
    run_manifest has no FK so delete_run deletes it explicitly (confirmed
    against the current database.py source, which now deletes run_manifest
    too -- see the module-level comment on this file's stale reference doc).
    idempotency_keys is untouched by design in both backends -- parity, not a
    bug -- so the row must still be readable afterward.
    """
    pg_backtest_db.insert_run(
        run_id="run-del", session_id="s1", agent_name="Agent", mode="backtest",
        start_date="2024-01-01", end_date="2024-01-02", initial_equity=1000.0,
    )
    pg_backtest_db.insert_equity_points(
        "run-del",
        [{"timestamp": "2024-01-01T00:00:00", "equity": 1000.0, "cash": 1000.0, "positions_value": 0.0}],
    )
    pg_backtest_db.insert_trades(
        "run-del",
        [
            {
                "timestamp": "2024-01-01T00:00:00",
                "symbol": "AAPL",
                "quantity": 1,
                "side": "buy",
                "price": 100.0,
                "value": 100.0,
            }
        ],
    )
    pg_backtest_db.insert_decisions(
        "run-del",
        [{"step_index": 0, "timestamp": "2024-01-01T00:00:00", "decision_source": "llm"}],
    )
    pg_backtest_db.insert_run_manifest("run-del", {"symbols": ["AAPL"]})
    pg_backtest_db.put_idempotency("run-del", 0, "idem-1", {"ok": True})

    pg_backtest_db.delete_run("run-del")

    assert pg_backtest_db.get_run("run-del") is None
    assert pg_backtest_db.get_equity_curve("run-del") == []  # FK cascade
    assert pg_backtest_db.get_trades("run-del") == []  # FK cascade
    assert pg_backtest_db.get_decisions("run-del") == []  # FK cascade
    assert pg_backtest_db.get_run_manifest("run-del") is None  # explicit delete, no FK

    assert pg_backtest_db.get_idempotency("run-del", 0, "idem-1") == {"ok": True}


@pg_only
def test_clear_all_empties_all_five_tables_postgres(pg_backtest_db):
    for run_id in ("run-clear-1", "run-clear-2"):
        pg_backtest_db.insert_run(
            run_id=run_id, session_id="s1", agent_name="Agent", mode="backtest",
            start_date="2024-01-01", end_date="2024-01-02", initial_equity=1000.0,
        )
        pg_backtest_db.insert_equity_points(
            run_id,
            [{"timestamp": "2024-01-01T00:00:00", "equity": 1000.0, "cash": 1000.0, "positions_value": 0.0}],
        )
        pg_backtest_db.insert_trades(
            run_id,
            [
                {
                    "timestamp": "2024-01-01T00:00:00",
                    "symbol": "AAPL",
                    "quantity": 1,
                    "side": "buy",
                    "price": 100.0,
                    "value": 100.0,
                }
            ],
        )
        pg_backtest_db.insert_decisions(
            run_id,
            [{"step_index": 0, "timestamp": "2024-01-01T00:00:00", "decision_source": "llm"}],
        )
        pg_backtest_db.insert_run_manifest(run_id, {"symbols": ["AAPL"]})

    pg_backtest_db.clear_all()

    assert pg_backtest_db.get_all_runs() == []
    for run_id in ("run-clear-1", "run-clear-2"):
        assert pg_backtest_db.get_equity_curve(run_id) == []
        assert pg_backtest_db.get_trades(run_id) == []
        assert pg_backtest_db.get_decisions(run_id) == []
        assert pg_backtest_db.get_run_manifest(run_id) is None


@pg_only
def test_datetime_timestamps_are_converted_before_reaching_postgres(pg_backtest_db):
    """The half of ``as_timestamp_text`` only a live server can prove.

    Every ``timestamp`` column here is TEXT, and psycopg types a ``datetime``
    parameter as ``timestamp``/``timestamptz`` -- for which Postgres has had no
    implicit assignment cast to ``text`` since 8.3. So before the conversion
    these three writers raised where SQLite silently stored a value, i.e. the
    same call 500'd on Postgres and worked on SQLite. Converting at the
    boundary makes both accept it *and* store identical text; the SQLite half
    is pinned in test_database_cold_half.py.

    ``insert_trades`` already did this on its own and is included to keep all
    four timestamp writers asserted in one place.
    """
    import datetime

    ts = datetime.datetime(2026, 1, 2, 9, 30, 15)
    expected = "2026-01-02T09:30:15"

    pg_backtest_db.insert_run(
        run_id="run-dt", session_id="s1", agent_name="Agent", mode="backtest",
        start_date="2026-01-01", end_date="2026-01-03", initial_equity=1000.0,
    )
    pg_backtest_db.insert_equity_point(
        "run-dt", ts, equity=1000.0, cash=1000.0, positions_value=0.0
    )
    pg_backtest_db.insert_decisions(
        "run-dt", [{"step_index": 0, "timestamp": ts, "decision_source": "llm"}]
    )
    pg_backtest_db.insert_trades(
        "run-dt",
        [{"timestamp": ts, "symbol": "AAPL", "quantity": 1, "side": "buy",
          "price": 100.0, "value": 100.0}],
    )

    assert [p["timestamp"] for p in pg_backtest_db.get_equity_curve("run-dt")] == [expected]
    assert [d["timestamp"] for d in pg_backtest_db.get_decisions("run-dt")] == [expected]
    assert [t["timestamp"] for t in pg_backtest_db.get_trades("run-dt")] == [expected]

    # insert_equity_points takes the batch path (executemany), so it is asserted
    # separately rather than assumed to share insert_equity_point's code path.
    # replace=True wipes the row written above first, which is why the curve is
    # still one point -- and the equity value proves it is the NEW row.
    pg_backtest_db.insert_equity_points(
        "run-dt",
        [{"timestamp": ts, "equity": 2000.0, "cash": 2000.0, "positions_value": 0.0}],
    )
    batch_curve = pg_backtest_db.get_equity_curve("run-dt")
    assert [p["timestamp"] for p in batch_curve] == [expected]
    assert [p["equity"] for p in batch_curve] == [2000.0]


@pg_only
def test_clear_all_leaves_the_embedded_sqlite_cold_tables_alone_postgres(pg_backtest_db):
    """``clear_all`` must confine itself to Postgres.

    An earlier cut delegated to ``self._sqlite.clear_all()`` as well. That
    embedded store exists for the ``idempotency_keys`` hot half *only*, and
    ``clear_all`` deliberately never touches that table -- so the delegation's
    DELETEs could only hit cold tables this object never reads, while reaching
    a file that defaults to ``DATABASE_PATH`` i.e. the committed seed. In prod
    terms: ``backtest_hourly_agent.py --clear`` with AGENT_RUNS_DATABASE_URL
    set would have blanked the seed's ``lb_*`` rows, and with them the seven
    ``auto_compute: false`` leaderboard entries that exist nowhere else (see
    tests/test_seed_database_integrity.py).

    Asserting on the embedded store rather than on the absence of a call is
    deliberate: a mock-the-call test would still pass if someone reached the
    same file by another route.
    """
    pg_backtest_db._sqlite.insert_run(
        run_id="lb_local_only", session_id="seed", agent_name="Seeded", mode="backtest",
        start_date="2024-01-01", end_date="2024-01-02", initial_equity=1000.0,
    )
    # A Postgres row too, so the control assertion below is not vacuous: the
    # fixture already empties agent_runs at setup, so asserting "Postgres is
    # empty" without writing to it first would hold before clear_all() is even
    # called -- and would still hold if clear_all() were `pass`.
    pg_backtest_db.insert_run(
        run_id="run-pg-side", session_id="s1", agent_name="Agent", mode="backtest",
        start_date="2024-01-01", end_date="2024-01-02", initial_equity=1000.0,
    )
    assert pg_backtest_db._sqlite.get_run("lb_local_only") is not None
    assert [r["run_id"] for r in pg_backtest_db.get_all_runs()] == ["run-pg-side"]

    pg_backtest_db.clear_all()

    assert pg_backtest_db._sqlite.get_run("lb_local_only") is not None
    # The wipe really happened -- this is what makes the assertion above mean
    # "clear_all spared the local store", not "clear_all did nothing".
    assert pg_backtest_db.get_all_runs() == []

    # Housekeeping: the embedded store is the session-shared temp DB from
    # conftest, and clear_all deliberately no longer touches it, so this row
    # would otherwise outlive the test. No test asserts an exact global list
    # today, but leaving it behind is the sort of cross-test coupling that
    # conftest's DATABASE_PATH redirect exists to prevent.
    pg_backtest_db._sqlite.delete_run("lb_local_only")


@pg_only
def test_idempotency_delegates_to_embedded_sqlite_postgres(pg_backtest_db):
    """The hot/cold split's only behavioral proof: get_idempotency/
    put_idempotency must never touch the Postgres side at all -- there is no
    idempotency_keys table anywhere in _init_schema's DDL, by design (see the
    module docstring: "The hot half stays local").
    """
    ack = {"status": "ok", "step": 3}
    pg_backtest_db.put_idempotency("run-idem", 3, "key-abc", ack)

    # Round trip through the public method.
    assert pg_backtest_db.get_idempotency("run-idem", 3, "key-abc") == ack

    # Proves it actually landed in the embedded SQLite store.
    assert pg_backtest_db._sqlite.get_idempotency("run-idem", 3, "key-abc") == ack

    # Proves the Postgres side has no idea this ever happened.
    with pg_backtest_db._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.idempotency_keys') AS tbl")
            assert cur.fetchone()["tbl"] is None


@pg_only
def test_get_equity_curves_batched_multiple_unknown_and_empty_postgres(pg_backtest_db):
    """get_equity_curves is a single batched query (unlike the SQLite twin's
    N-query loop): multiple ids in one call, an unknown id still gets a key
    mapped to [], and an empty input list short-circuits to {}."""
    pg_backtest_db.insert_run(
        run_id="run-a", session_id="s1", agent_name="Agent", mode="backtest",
        start_date="2024-01-01", end_date="2024-01-02", initial_equity=1000.0,
    )
    pg_backtest_db.insert_run(
        run_id="run-b", session_id="s1", agent_name="Agent", mode="backtest",
        start_date="2024-01-01", end_date="2024-01-02", initial_equity=1000.0,
    )
    pg_backtest_db.insert_equity_points(
        "run-a",
        [
            {"timestamp": "2024-01-01T00:00:00", "equity": 1000.0, "cash": 1000.0, "positions_value": 0.0},
            {"timestamp": "2024-01-01T01:00:00", "equity": 1010.0, "cash": 990.0, "positions_value": 20.0},
        ],
    )
    pg_backtest_db.insert_equity_points(
        "run-b",
        [{"timestamp": "2024-01-01T00:00:00", "equity": 500.0, "cash": 500.0, "positions_value": 0.0}],
    )

    curves = pg_backtest_db.get_equity_curves(["run-a", "run-b", "run-unknown"])
    assert set(curves.keys()) == {"run-a", "run-b", "run-unknown"}
    assert len(curves["run-a"]) == 2
    assert len(curves["run-b"]) == 1
    assert curves["run-unknown"] == []  # key present, empty list -- not a missing key

    assert pg_backtest_db.get_equity_curves([]) == {}


@pg_only
def test_insert_trades_and_get_trades_round_trip_postgres(pg_backtest_db):
    """Round-trips both trade-dict input shapes insert_trades accepts
    (quantity/value vs. the legacy shares/cost aliases) and confirms
    get_trades strips NULL audit fields the same way get_equity_curve does.
    """
    pg_backtest_db.insert_run(
        run_id="run-trades", session_id="s1", agent_name="Agent", mode="backtest",
        start_date="2024-01-01", end_date="2024-01-02", initial_equity=1000.0,
    )
    pg_backtest_db.insert_trades(
        "run-trades",
        [
            {
                "timestamp": "2024-01-01T10:00:00",
                "symbol": "AAPL",
                "quantity": 10,
                "side": "buy",
                "price": 150.0,
                "value": 1500.0,
                "reason": "signal",
                "reference_price": 149.9,
                "gross_value": 1500.0,
                "slippage_amount": 1.0,
                "commission": 0.4,
                "stamp_duty": 0.0,
                "transfer_fee": 0.02,
                "total_fees": 0.42,
                "net_cash_impact": -1500.42,
                "native_price": 1080.0,
                "native_value": 10800.0,
                "native_reference_price": 1079.28,
                "native_gross_value": 10800.0,
                "native_slippage_amount": 7.2,
                "native_commission": 2.88,
                "native_stamp_duty": 0.0,
                "native_transfer_fee": 0.144,
                "native_total_fees": 3.024,
                "native_net_cash_impact": -10803.024,
                "fx_rate": 7.2,
            },
            {
                # legacy aliases: shares -> quantity, cost -> value; no
                # native_* fields -> audit fields must be stripped on read.
                # symbol has no alias (unlike shares/cost) and is NOT NULL
                # on both backends, so it must still be supplied here.
                "timestamp": "2024-01-01T11:00:00",
                "symbol": "MSFT",
                "shares": 5,
                "side": "sell",
                "price": 152.0,
                "cost": 760.0,
            },
        ],
    )

    trades = pg_backtest_db.get_trades("run-trades")
    assert len(trades) == 2
    first, second = trades

    assert first["symbol"] == "AAPL"
    assert first["side"] == "BUY"
    assert first["quantity"] == 10
    assert first["fx_rate"] == 7.2
    assert first["total_fees"] == 0.42
    assert first["native_total_fees"] == 3.024
    assert first["native_net_cash_impact"] == -10803.024

    assert second["symbol"] == "MSFT"
    assert second["quantity"] == 5
    assert second["side"] == "SELL"
    assert second["value"] == 760.0
    assert "fx_rate" not in second
    assert "native_price" not in second
    assert "native_value" not in second
    assert "commission" not in second
    assert "total_fees" not in second
    assert "native_total_fees" not in second


@pg_only
def test_insert_decisions_and_get_decisions_round_trip_postgres(pg_backtest_db):
    pg_backtest_db.insert_run(
        run_id="run-dec", session_id="s1", agent_name="Agent", mode="backtest",
        start_date="2024-01-01", end_date="2024-01-02", initial_equity=1000.0,
    )
    pg_backtest_db.insert_decisions(
        "run-dec",
        [
            {
                "step_index": 0,
                "timestamp": "2024-01-01T00:00:00",
                "decision_source": "llm",
                "actions_submitted": [{"action": "buy", "symbol": "AAPL"}],
                "actions_executed": 1,
                "context_ref": "ctx-1",
            },
            {
                "step_index": 1,
                "timestamp": "2024-01-01T01:00:00",
                "decision_source": "rule-based",
                # actions_submitted omitted -> writer defaults it to []
            },
        ],
    )

    decisions = pg_backtest_db.get_decisions("run-dec")
    assert [d["step_index"] for d in decisions] == [0, 1]
    assert decisions[0]["actions_submitted"] == [{"action": "buy", "symbol": "AAPL"}]
    assert decisions[0]["context_ref"] == "ctx-1"
    assert decisions[1]["actions_submitted"] == []
    assert decisions[1]["decision_source"] == "rule-based"


@pg_only
def test_update_run_baselines_coalesce_preserves_unset_field_postgres(pg_backtest_db):
    """COALESCE(%s, existing_value): passing None for one baseline kwarg must
    leave the other's already-set value untouched, not null it out."""
    pg_backtest_db.insert_run(
        run_id="run-base", session_id="s1", agent_name="Agent", mode="backtest",
        start_date="2024-01-01", end_date="2024-01-02", initial_equity=1000.0,
    )

    pg_backtest_db.update_run_baselines("run-base", djia_run_id="djia-1")
    run = pg_backtest_db.get_run("run-base")
    assert run["baseline_djia_run_id"] == "djia-1"
    assert run["baseline_buyhold_run_id"] is None

    pg_backtest_db.update_run_baselines("run-base", buyhold_run_id="bh-1")
    run = pg_backtest_db.get_run("run-base")
    assert run["baseline_djia_run_id"] == "djia-1"  # untouched by the second call
    assert run["baseline_buyhold_run_id"] == "bh-1"
