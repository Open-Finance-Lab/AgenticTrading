"""Partial-result recovery for interrupted backtests.

Covers the pieces of the design:
1. The engine's throttled durable flush (_flush_live_progress_durable) and the
   snapshot it writes -- built here by the engine itself, so the reclaimer is
   tested against the producer's real shape rather than a hand-written one.
2. The store contract both twins share: listing reports each snapshot's age on
   the database clock without loading payloads, a heartbeat refreshes it, and
   deleting a run deletes its snapshot.
3. The reclaimer: a snapshot is only reclaimed once nothing has stamped it for
   RECLAIM_STALE_AFTER_SECONDS (a fresher one may be a live run on another
   instance), and only by the one reclaimer that claims it; discarded past
   RECLAIM_MAX_AGE_SECONDS or when no heartbeat backs its age; dropped when its
   run finished.
4. The boot watcher, which re-sweeps every snapshot for RECLAIM_WATCH_SECONDS
   after boot, and the child-side heartbeat that keeps a live run fresh.
"""

import json
import threading
import time
from pathlib import Path

import pytest

from dashboard.backend.database import BacktestDatabase
from dashboard.backend.domain.backtesting import partial_results
from dashboard.backend.domain.backtesting.partial_results import (
    LIVE_PROGRESS_HEARTBEAT_SECONDS,
    RECLAIM_MAX_AGE_SECONDS,
    RECLAIM_STALE_AFTER_SECONDS,
    LiveProgressHeartbeat,
    reclaim_interrupted_backtests,
    watch_interrupted_backtests,
)

STALE = RECLAIM_STALE_AFTER_SECONDS + 60
ANCIENT = RECLAIM_MAX_AGE_SECONDS + 60


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_RUNS_DATABASE_URL", "")
    db = BacktestDatabase(db_path=tmp_path / "runs.db")
    monkeypatch.setattr("dashboard.backend.database.db", db, raising=False)
    return db


class _FakeManager:
    """The slice of PortfolioManager _publish_live_progress reads."""

    def __init__(self, curve, trades):
        self._curve = curve
        self.trades = trades
        self.rejected_orders = []
        self.order_events = []

    def get_equity_curve(self):
        return list(self._curve)


def engine_snapshot(tmp_path, *, run_id="run_a", step=10, total=49,
                    equity=10500.0, owner_user_id=7, llm_used=True):
    """Have the real engine flush one durable snapshot into ``database.db``.

    Step 10 is on the flush cadence. Returns the engine for further calls.
    """
    from dashboard.backend.domain.backtesting.currency import CurrencyContext
    from dashboard.backend.domain.backtesting.engine import HourlyBacktester
    from dashboard.backend.domain.agents.runtime import PIPELINE_RUNTIME_TYPE

    engine = object.__new__(HourlyBacktester)
    engine.live_run_id = run_id
    engine.session_id = "sess-1"
    engine.start_date = "2026-08-01"
    engine.end_date = "2026-08-31"
    engine.model = "gpt-test"
    engine.runtime_type = PIPELINE_RUNTIME_TYPE
    engine.mode = "safe_trading"
    engine.initial_capital = 10000.0
    engine.owner_user_id = owner_user_id
    engine.progress_file = str(tmp_path / f"progress_{run_id}.json")
    engine.currency_context = CurrencyContext.identity("USD", "America/New_York")
    manager = _FakeManager(
        curve=[
            {"timestamp": "2026-08-01T14:00:00", "equity": 10000.0,
             "cash": 9000.0, "positions_value": 1000.0},
            {"timestamp": "2026-08-01T15:00:00", "equity": equity,
             "cash": 9200.0, "positions_value": equity - 9200.0},
        ],
        trades=[
            {"symbol": "AAPL", "side": "buy", "shares": 10, "price": 100.0,
             "cost": 1000.0, "timestamp": "2026-08-01T15:00:00"},
        ],
    )
    engine._publish_live_progress(step, total, manager, llm_used=llm_used)
    return engine


def backdate(store, run_id, seconds):
    """Make a snapshot look ``seconds`` old on the database's own clock."""
    conn = store._get_connection()
    try:
        conn.execute(
            "UPDATE run_live_progress SET updated_at = datetime('now', ?)"
            " WHERE run_id = ?",
            (f"-{int(seconds)} seconds", run_id),
        )
        conn.commit()
    finally:
        conn.close()


def _metadata(run):
    meta = run.get("metadata")
    return json.loads(meta) if isinstance(meta, str) else meta


# ------------------------------------------------------------ store contract


def test_listing_reports_age_on_the_db_clock_without_payloads(store, tmp_path):
    engine_snapshot(tmp_path)
    backdate(store, "run_a", 600)

    [row] = store.list_live_progress()

    assert set(row) == {"run_id", "session_id", "agent_name", "age_seconds"}
    assert (row["run_id"], row["agent_name"]) == ("run_a", "Agent")
    assert row["age_seconds"] == pytest.approx(600, abs=5)


def test_touch_refreshes_age_and_never_creates_a_row(store, tmp_path):
    engine_snapshot(tmp_path)
    backdate(store, "run_a", 600)

    store.touch_live_progress("run_a")
    store.touch_live_progress("no_such_run")

    rows = store.list_live_progress()
    assert [r["run_id"] for r in rows] == ["run_a"]
    assert rows[0]["age_seconds"] < 5


def test_delete_run_deletes_the_runs_snapshot(store, tmp_path):
    """A run an admin deleted must not come back from its snapshot."""
    engine_snapshot(tmp_path)
    store.insert_run("run_a", "sess-1", "Agent", "backtest",
                     "2026-08-01", "2026-08-31", 10000.0)

    store.delete_run("run_a")

    assert store.get_live_progress("run_a") is None
    assert store.list_live_progress() == []


def test_clear_all_clears_snapshots(store, tmp_path):
    engine_snapshot(tmp_path)

    store.clear_all()

    assert store.list_live_progress() == []


def test_corrupt_payload_is_listed_but_reads_as_none(store):
    conn = store._get_connection()
    try:
        conn.execute(
            "INSERT INTO run_live_progress (run_id, session_id, agent_name, payload)"
            " VALUES ('run_bad', 's', 'Agent', '{not json')"
        )
        conn.commit()
    finally:
        conn.close()

    assert [r["run_id"] for r in store.list_live_progress()] == ["run_bad"]
    assert store.get_live_progress("run_bad") is None


def test_claim_restamps_a_stale_snapshot_exactly_once(store, tmp_path):
    """The claim is a compare-and-set on staleness: the first caller re-stamps
    the row, which makes it fresh, so every later caller is refused."""
    engine_snapshot(tmp_path, run_id="run_a")
    engine_snapshot(tmp_path, run_id="run_fresh")
    backdate(store, "run_a", STALE)

    assert store.claim_live_progress("run_a", RECLAIM_STALE_AFTER_SECONDS) is True
    assert store.claim_live_progress("run_a", RECLAIM_STALE_AFTER_SECONDS) is False
    assert store.claim_live_progress("run_fresh", RECLAIM_STALE_AFTER_SECONDS) is False
    assert store.claim_live_progress("no_such_run", RECLAIM_STALE_AFTER_SECONDS) is False
    ages = {r["run_id"]: r["age_seconds"] for r in store.list_live_progress()}
    assert ages["run_a"] < 5


# ------------------------------------------------------------ reclaimer


def test_stale_snapshot_becomes_an_interrupted_run_shaped_like_a_completed_one(
    store, tmp_path
):
    engine_snapshot(tmp_path, step=10, total=49, equity=10500.0, owner_user_id=7)
    backdate(store, "run_a", STALE)

    counts = reclaim_interrupted_backtests()

    assert counts["reclaimed"] == 1
    run = store.get_run("run_a")
    # The same name and mode run_agent_backtest gives the completed row, so the
    # card labels it and get_runs_by_mode("backtest") lists it.
    assert run["agent_name"] == "Agent"
    assert run["mode"] == "backtest"
    assert run["owner_user_id"] == 7
    assert run["llm_model"] == "gpt-test"
    assert run["total_return"] == pytest.approx(0.05)
    assert run["num_trades"] == 1
    meta = _metadata(run)
    assert meta["interrupted"] is True
    assert meta["interrupted_step"] == 10
    assert meta["interrupted_total_steps"] == 49
    curve = store.get_equity_curve("run_a")
    assert len(curve) == 2
    assert curve[-1]["equity"] == pytest.approx(10500.0)
    [trade] = store.get_trades("run_a")
    assert trade["quantity"] == 10
    assert trade["side"] == "BUY"
    assert store.get_live_progress("run_a") is None


def test_fresh_snapshot_is_left_alone_and_reported_pending(store, tmp_path):
    """Fresher than the stale window, it may be a live run on another instance
    (a zero-downtime deploy's old instance keeps running while this one boots)."""
    engine_snapshot(tmp_path)

    counts = reclaim_interrupted_backtests()

    assert counts["reclaimed"] == 0
    assert counts["pending"] == ["run_a"]
    assert store.get_run("run_a") is None
    assert store.get_live_progress("run_a") is not None


def test_snapshot_past_max_age_is_discarded_not_published(store, tmp_path):
    """Nothing this design writes stays unreclaimed that long while any instance
    is up, so such a row predates it -- cancelled and failed runs used to leave
    one -- and how its run ended is unknown."""
    engine_snapshot(tmp_path)
    backdate(store, "run_a", ANCIENT)

    counts = reclaim_interrupted_backtests()

    assert counts["discarded"] == 1
    assert store.get_run("run_a") is None
    assert store.list_live_progress() == []


def test_completed_runs_snapshot_is_dropped_even_when_fresh(store, tmp_path):
    store.insert_run(
        "run_b", "sess-1", "Agent", "backtest", "2026-08-01", "2026-08-31",
        10000.0, final_equity=11000.0,
    )
    engine_snapshot(tmp_path, run_id="run_b")

    counts = reclaim_interrupted_backtests()

    assert counts["dropped_stale"] == 1
    assert counts["pending"] == []
    assert store.get_run("run_b")["final_equity"] == pytest.approx(11000.0)
    assert store.get_live_progress("run_b") is None


def test_unreadable_stale_snapshot_is_discarded(store):
    conn = store._get_connection()
    try:
        conn.execute(
            "INSERT INTO run_live_progress (run_id, session_id, agent_name, payload)"
            " VALUES ('run_bad', 's', 'Agent', '{not json')"
        )
        conn.commit()
    finally:
        conn.close()
    backdate(store, "run_bad", STALE)

    counts = reclaim_interrupted_backtests()

    assert counts["discarded"] == 1
    assert counts["failed"] == 0
    assert store.list_live_progress() == []


def test_reclaim_is_idempotent(store, tmp_path):
    engine_snapshot(tmp_path)
    backdate(store, "run_a", STALE)

    first = reclaim_interrupted_backtests()
    second = reclaim_interrupted_backtests()

    assert first["reclaimed"] == 1
    assert second["reclaimed"] == 0
    assert len(store.get_equity_curve("run_a")) == 2


def test_rule_based_run_is_reclaimed_under_the_label_its_completed_row_would_carry(
    store, tmp_path
):
    """The engine's model id is set even when no model is called; the completed
    row says rule-based then, and a reclaimed one must not name the model."""
    engine_snapshot(tmp_path, llm_used=False)
    backdate(store, "run_a", STALE)

    reclaim_interrupted_backtests()

    assert store.get_run("run_a")["llm_model"] == "rule-based"


def test_snapshot_without_a_heartbeat_is_discarded_not_published(store, tmp_path):
    """A writer from before the heartbeat existed lets a live run go unstamped
    between its ten-bar flushes, so a stale age does not show its run ended --
    nor that it was not cancelled, since its parent never deleted the row."""
    engine_snapshot(tmp_path)
    payload = store.get_live_progress("run_a")
    del payload["run_metadata"]["heartbeat_seconds"]
    store.upsert_live_progress("run_a", "sess-1", "Agent", payload)
    backdate(store, "run_a", STALE)

    counts = reclaim_interrupted_backtests()

    assert counts["discarded"] == 1
    assert store.get_run("run_a") is None
    assert store.list_live_progress() == []


def test_engine_snapshot_declares_its_heartbeat(store, tmp_path):
    engine_snapshot(tmp_path)

    meta = store.get_live_progress("run_a")["run_metadata"]

    assert meta["heartbeat_seconds"] == LIVE_PROGRESS_HEARTBEAT_SECONDS


def test_snapshot_stamped_after_the_listing_is_left_pending(store, tmp_path):
    """The listing can be stale by the time a row is judged: a heartbeat that
    lands in between means the run is alive."""
    engine_snapshot(tmp_path)
    backdate(store, "run_a", STALE)
    listed = store.list_live_progress()
    store.touch_live_progress("run_a")
    store.list_live_progress = lambda: listed

    counts = reclaim_interrupted_backtests()

    assert counts["pending"] == ["run_a"]
    assert store.get_run("run_a") is None


def test_two_reclaimers_racing_one_snapshot_publish_it_once(store, tmp_path):
    """During a deploy two instances sweep the same table. Trades are appended,
    so if both published one run its trades would be written twice."""
    engine_snapshot(tmp_path)
    backdate(store, "run_a", STALE)
    real_read = store.get_live_progress
    raced = {"done": False}

    def read_then_lose_the_race(run_id):
        payload = real_read(run_id)
        if not raced["done"]:
            raced["done"] = True
            # The other instance judges the same row before this one publishes.
            raced["other"] = reclaim_interrupted_backtests()
        return payload

    store.get_live_progress = read_then_lose_the_race

    counts = reclaim_interrupted_backtests()

    assert counts["reclaimed"] + raced["other"]["reclaimed"] == 1
    assert len(store.get_trades("run_a")) == 1


def test_listing_failure_is_reported_not_read_as_empty(monkeypatch):
    class Down:
        def list_live_progress(self):
            raise RuntimeError("db down")

    monkeypatch.setattr("dashboard.backend.database.db", Down(), raising=False)

    counts = reclaim_interrupted_backtests()

    assert counts["listing_failed"] is True
    assert counts["pending"] == []


# ------------------------------------------------------------ boot watcher


class _FakeTime:
    """A clock the watcher's sleeps advance, with a hook run after each sleep."""

    def __init__(self, on_sleep=None):
        self.now = 0.0
        self.sleeps = 0
        self.on_sleep = on_sleep

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        self.sleeps += 1
        if self.on_sleep is not None:
            self.on_sleep(self.sleeps)


def _watch(fake, limit_seconds=300):
    return watch_interrupted_backtests(
        sleep=fake.sleep, clock=fake.clock, recheck_seconds=60,
        limit_seconds=limit_seconds,
    )


def test_watcher_sweeps_for_its_whole_window_even_when_boot_finds_nothing(
    store, monkeypatch
):
    sweeps = []
    real_sweep = partial_results.reclaim_interrupted_backtests

    def counted_sweep():
        sweeps.append(1)
        return real_sweep()

    monkeypatch.setattr(partial_results, "reclaim_interrupted_backtests", counted_sweep)
    fake = _FakeTime()

    _watch(fake, limit_seconds=300)

    assert fake.now == pytest.approx(300)
    assert len(sweeps) == 6  # t = 0, 60, ..., 300


def test_watcher_reclaims_a_snapshot_first_written_after_boot(store, tmp_path):
    """The old instance keeps running -- and starting -- runs until Render stops
    it, so a run that dies with it can write its first snapshot after this
    instance's boot listing."""

    def old_instance(sleeps):
        if sleeps == 1:
            engine_snapshot(tmp_path, run_id="run_late")
        if sleeps == 2:
            backdate(store, "run_late", STALE)  # it died with its instance

    totals = _watch(_FakeTime(old_instance))

    assert totals["reclaimed"] == 1
    assert store.get_run("run_late") is not None


def test_watcher_leaves_a_run_that_is_still_stamped(store, tmp_path):
    engine_snapshot(tmp_path)
    backdate(store, "run_a", STALE)
    store.touch_live_progress("run_a")

    totals = _watch(_FakeTime(lambda _n: store.touch_live_progress("run_a")))

    assert totals["reclaimed"] == 0
    assert store.get_live_progress("run_a") is not None


def test_watcher_leaves_a_snapshot_its_parent_cleaned_up(store, tmp_path):
    engine_snapshot(tmp_path)

    totals = _watch(_FakeTime(lambda _n: store.delete_live_progress("run_a")))

    assert totals["reclaimed"] == 0
    assert store.get_run("run_a") is None


def test_watcher_retries_a_snapshot_whose_judgment_failed(store, tmp_path):
    engine_snapshot(tmp_path)
    backdate(store, "run_a", STALE)
    real_get_run = store.get_run
    calls = {"n": 0}

    def flaky_get_run(run_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("pool timeout")
        return real_get_run(run_id)

    store.get_run = flaky_get_run

    totals = _watch(_FakeTime())

    assert totals["failed"] == 1
    assert totals["reclaimed"] == 1


def test_watcher_survives_a_listing_failure(store, tmp_path):
    engine_snapshot(tmp_path)
    backdate(store, "run_a", STALE)
    real_list = store.list_live_progress
    calls = {"n": 0}

    def flaky_list():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return real_list()

    store.list_live_progress = flaky_list

    totals = _watch(_FakeTime())

    assert totals["reclaimed"] == 1


def test_watcher_reports_a_sweep_when_it_happens(store, tmp_path, capsys):
    """Not once the window closes, half an hour later."""
    engine_snapshot(tmp_path)
    backdate(store, "run_a", STALE)
    printed = []

    _watch(_FakeTime(lambda _n: printed.append(capsys.readouterr().out)))

    assert "Reclaimed 1 interrupted run" in printed[0]


def test_watch_window_outlasts_the_previous_instance():
    """Render stops the old instance 60s after this one takes traffic and gives
    it at most 300s to exit; its runs go stale a stale window after that."""
    assert partial_results.RECLAIM_WATCH_SECONDS >= (
        60 + 300 + RECLAIM_STALE_AFTER_SECONDS + partial_results.RECLAIM_RECHECK_SECONDS
    )


def test_reclaimer_thread_runs_off_the_boot_path(store, tmp_path, monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def slow_watch(**_kwargs):
        started.set()
        release.wait(5)
        return {}

    monkeypatch.setattr(partial_results, "watch_interrupted_backtests", slow_watch)

    thread = partial_results.start_interrupted_run_reclaimer()

    # Returned while the sweep is still running: boot does not wait on it.
    assert started.wait(5)
    assert thread.daemon is True
    assert thread.is_alive()
    release.set()
    thread.join(5)
    assert not thread.is_alive()


# ------------------------------------------------------------ heartbeat


def test_heartbeat_keeps_a_live_runs_snapshot_fresh(store, tmp_path):
    engine_snapshot(tmp_path)
    backdate(store, "run_a", STALE)

    heartbeat = LiveProgressHeartbeat("run_a", interval_seconds=0.01).start()
    try:
        deadline = time.monotonic() + 5
        while store.list_live_progress()[0]["age_seconds"] > 5:
            assert time.monotonic() < deadline, "heartbeat never touched the row"
            time.sleep(0.01)
    finally:
        heartbeat.stop()

    assert not heartbeat.is_alive()
    assert reclaim_interrupted_backtests()["pending"] == ["run_a"]


def test_heartbeat_survives_db_failures(monkeypatch):
    calls = []

    class Flaky:
        def touch_live_progress(self, run_id):
            calls.append(run_id)
            raise RuntimeError("db down")

    monkeypatch.setattr("dashboard.backend.database.db", Flaky(), raising=False)

    heartbeat = LiveProgressHeartbeat("run_y", interval_seconds=0.01).start()
    deadline = time.monotonic() + 5
    while len(calls) < 3:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    heartbeat.stop()

    assert not heartbeat.is_alive()


# ------------------------------------------------------------ engine flush


def test_engine_flush_cadence_and_tombstone(store, monkeypatch):
    """The engine-side throttle: step 1 always flushes (a run that dies young
    still leaves a tombstone), then every 10th step; other steps are no-ops."""
    from dashboard.backend.domain.backtesting.engine import HourlyBacktester

    calls = []
    step_holder = {"step": 0}

    def fake_upsert(run_id, session_id, agent_name, payload):
        calls.append(step_holder["step"])

    engine = object.__new__(HourlyBacktester)
    engine.live_run_id = "run_x"
    engine.session_id = "sess-x"
    monkeypatch.setattr(store, "upsert_live_progress", fake_upsert)

    for step in range(1, 31):
        step_holder["step"] = step
        engine._flush_live_progress_durable(step, {"step": step})

    assert calls == [1, 10, 20, 30]


def test_engine_flush_swallows_db_failure(monkeypatch):
    from dashboard.backend.domain.backtesting.engine import HourlyBacktester

    engine = object.__new__(HourlyBacktester)
    engine.live_run_id = "run_y"
    engine.session_id = "s"

    class Boom:
        def upsert_live_progress(self, *a, **k):
            raise RuntimeError("db down")

    monkeypatch.setattr("dashboard.backend.database.db", Boom(), raising=False)

    # Must not raise: the bar loop continues regardless of DB health.
    engine._flush_live_progress_durable(10, {"step": 10})


@pytest.mark.parametrize(
    "runtime, llm_used, dispatcher_calls, expected",
    [
        ("pipeline", True, 0, "gpt-test"),
        ("pipeline", False, 0, "rule-based"),
        ("hosted", False, 3, "hosted-model"),
        ("hosted", False, 0, "rule-based"),
    ],
)
def test_recorded_llm_model(runtime, llm_used, dispatcher_calls, expected):
    """One rule for the completed row and the snapshot: a model is named only
    once it drove a step."""
    from types import SimpleNamespace

    from dashboard.backend.domain.agents.runtime import (
        AI_HEDGE_FUND_RUNTIME_TYPE,
        PIPELINE_RUNTIME_TYPE,
    )
    from dashboard.backend.domain.backtesting.engine import HourlyBacktester

    engine = object.__new__(HourlyBacktester)
    engine.model = "gpt-test"
    engine.runtime_type = (
        PIPELINE_RUNTIME_TYPE if runtime == "pipeline" else AI_HEDGE_FUND_RUNTIME_TYPE
    )
    engine.runtime_dispatcher = SimpleNamespace(
        calls=dispatcher_calls, model_name="hosted-model"
    )

    assert engine._recorded_llm_model(llm_used) == expected


def test_bar_loop_and_completed_row_share_the_llm_model_rule():
    import ast
    import inspect
    import textwrap

    from dashboard.backend.domain.backtesting.engine import HourlyBacktester

    tree = ast.parse(textwrap.dedent(inspect.getsource(HourlyBacktester.run_agent_backtest)))
    publishes = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "_publish_live_progress"
    ]
    assert publishes and all(
        any(kw.arg == "llm_used" for kw in call.keywords) for call in publishes
    )
    assigned = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "llm_model" for t in node.targets)
    ]
    assert assigned and all(
        isinstance(value, ast.Call)
        and getattr(value.func, "attr", None) == "_recorded_llm_model"
        for value in assigned
    )


# ------------------------------------------------------------ wiring
# Both seams are a single statement each in modules whose behaviour cannot be
# driven here without launching a backtest child or booting the app, so their
# shape is pinned instead.

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_the_dashboard_child_heartbeats_for_the_whole_agent_loop():
    import ast

    script = REPO_ROOT / "dashboard" / "scripts" / "backtest_hourly_agent.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))

    def calls(nodes, attr):
        return [
            n for node in nodes for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == attr
        ]

    [loop_try] = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Try) and calls(n.body, "run_agent_backtest")
    ]
    [stop] = calls(loop_try.finalbody, "stop")
    stopped = ast.unparse(stop.func.value)
    starts = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and [ast.unparse(t) for t in n.targets] == [stopped]
        and "LiveProgressHeartbeat(" in ast.unparse(n.value)
    ]
    assert starts, f"{stopped} is stopped but never started from LiveProgressHeartbeat"
    assert all(s.lineno < loop_try.lineno for s in starts)


def test_app_startup_reclaims_on_a_thread_not_inline():
    source = (REPO_ROOT / "dashboard" / "backend" / "app.py").read_text(encoding="utf-8")
    assert "start_interrupted_run_reclaimer()" in source
    assert "reclaim_interrupted_backtests(" not in source
