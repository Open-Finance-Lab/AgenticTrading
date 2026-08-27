"""Run-lifecycle unification across /api/v1 and /api/v2 (PR #67 H4 follow-ups).

Three related contracts, one mechanism — v2 runs persist protocol_runs rows
exactly like v1 runs do:

1. Unified per-agent active-run cap: run_store.count_active_runs() covers both
   surfaces, and both create paths serialize on the same lock, so an agent can
   no longer hold 2× MAX_ACTIVE_RUNS_PER_AGENT by splitting across v1 and v2.
2. Reaper/orphan-recovery registration for v2: the v1 reaper drives a
   registered v2 sweep (drain deadlines, archive terminal backends); startup
   recovery marks crashed v2 rows failed, and the v2 API rehydrates terminal
   rows so reads survive both archival and restart.
3. Multi-worker recovery hardening: rows carry owner_instance + heartbeat_at;
   live runs are heartbeated by the reaper, and only stale rows are failed —
   a second worker's live runs are no longer collateral damage.
"""

import inspect
import sqlite3
import uuid

import pytest
from fastapi.testclient import TestClient

import dashboard.backend.api.v2.runs as runs_mod
import dashboard.backend.app as app_module
import dashboard.backend.domain.runs.repository as run_store_module
import dashboard.backend.domain.runs.service as run_service
from dashboard.backend.app import app
from dashboard.backend.database import db
from dashboard.backend.execution.backtest_backend import ArchivedBacktestBackend
from dashboard.backend.tests._v2_fakes import FakeBackend

client = TestClient(app)

run_store = run_store_module.run_store


def _agent(name):
    r = client.post("/api/v2/agents", json={"name": name}).json()
    return r["api_key"], r["session_id"], r["agent_id"]


class _StubBackend:
    """Create-path stub: no Alpaca, stays active until told otherwise."""

    loop = "lockstep"
    news_sentiment_source = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._active = True
        self._final = "completed"

    def start_background_load(self):
        pass

    def is_active(self):
        return self._active

    def status(self):
        return {"status": "waiting_decision" if self._active else self._final}

    def advance(self):
        pass

    def cancel(self):
        self._active = False
        self._final = "closed"


def _create_v2_run(key, monkeypatch, backend_cls=_StubBackend):
    monkeypatch.setattr(runs_mod, "BacktestBackend", backend_cls)
    resp = client.post(
        "/api/v2/runs",
        json={"start_date": "2026-04-15", "end_date": "2026-04-16"},
        headers={"X-API-Key": key},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["run_id"]


# ---------------------------------------------------------------------------
# 1. v2 runs persist protocol_runs rows through their lifecycle
# ---------------------------------------------------------------------------


def test_v2_create_persists_protocol_run_row(monkeypatch):
    key, sid, agent_id = _agent("row-writer")
    run_id = _create_v2_run(key, monkeypatch)

    record = run_store.get_run(run_id)
    assert record is not None, "v2 create must write a protocol_runs row"
    assert record["agent_id"] == agent_id
    assert record["session_id"] == sid
    assert record["status"] in ("created", "loading", "running")


def test_v2_cancel_marks_row_closed(monkeypatch):
    key, _, _ = _agent("row-canceller")
    run_id = _create_v2_run(key, monkeypatch)

    resp = client.post(f"/api/v2/runs/{run_id}/cancel", headers={"X-API-Key": key})
    assert resp.status_code == 200
    assert run_store.get_run(run_id)["status"] == "closed"


# ---------------------------------------------------------------------------
# 2. Unified active-run cap across surfaces
# ---------------------------------------------------------------------------


def test_v1_active_runs_count_against_v2_creates(monkeypatch):
    """An agent at the cap on the v1 surface must be refused on v2."""
    key, sid, agent_id = _agent("cross-cap-v1v2")
    monkeypatch.setattr(runs_mod, "MAX_ACTIVE_RUNS_PER_AGENT", 1)
    # A v1-created run: row in protocol_runs, no v2 registry entry.
    run_store.create_run(
        agent_id=agent_id, agent_version_id=None, session_id=sid,
        environment_id="us-equity-hourly-v1", environment_type="backtest",
        config={}, backtest_id="bt_cross_v1", status="running",
    )

    monkeypatch.setattr(runs_mod, "BacktestBackend", _StubBackend)
    resp = client.post(
        "/api/v2/runs",
        json={"start_date": "2026-04-15", "end_date": "2026-04-16"},
        headers={"X-API-Key": key},
    )
    assert resp.status_code == 429, resp.text
    assert resp.json()["error"]["code"] == "too_many_active_runs"


def test_v2_active_runs_count_against_v1_creates(monkeypatch):
    """The reverse hole: a v2-held run must count when v1 checks the cap."""
    key, sid, agent_id = _agent("cross-cap-v2v1")
    monkeypatch.setattr(run_service, "MAX_ACTIVE_RUNS_PER_AGENT", 1)
    _create_v2_run(key, monkeypatch)  # leaves an active v2 run + its row

    with pytest.raises(run_service.ProtocolError) as ei:
        run_service.create_run(
            agent={"agent_id": agent_id, "session_id": sid, "name": "x"},
            agent_version=None,
            environment_id="us-equity-hourly-v1",
            config={"start_date": "2026-04-15", "end_date": "2026-04-16"},
        )
    assert ei.value.code == "too_many_active_runs"


def test_v2_cap_frees_terminal_runs_without_status_side_effects(monkeypatch):
    """A v2 run that finished must stop counting even before the reaper's next
    sweep — the create path reconciles its own registry's inactive backends.
    Preserves the merge-hardening invariant (cap ≠ status() on live runs)."""
    key, _, _ = _agent("cap-reconcile")
    monkeypatch.setattr(runs_mod, "MAX_ACTIVE_RUNS_PER_AGENT", 1)

    run_id = _create_v2_run(key, monkeypatch)
    # The run finishes: backend goes inactive, but nothing updated the row yet.
    with runs_mod._lock:
        runs_mod._runs[run_id]["backend"]._active = False

    # Second create must succeed (the terminal run no longer holds a slot).
    resp = client.post(
        "/api/v2/runs",
        json={"start_date": "2026-04-15", "end_date": "2026-04-16"},
        headers={"X-API-Key": key},
    )
    assert resp.status_code == 200, resp.text
    # And the reconciled row is terminal now.
    assert run_store.get_run(run_id)["status"] in ("completed", "failed", "closed")


# ---------------------------------------------------------------------------
# 3. Reaper sweep registration + v2 archival
# ---------------------------------------------------------------------------


def test_reap_runs_invokes_registered_sweeps(monkeypatch):
    calls = []
    monkeypatch.setattr(run_service, "_extra_reaper_sweeps", [])
    run_service.register_reaper_sweep(lambda: calls.append("swept"))
    run_service.reap_runs()
    assert calls == ["swept"]


def test_startup_registers_analytics_retention_sweep_once():
    source = inspect.getsource(app_module.startup_event)
    call = "register_reaper_sweep(analytics_retention_coordinator.run_if_due)"
    assert source.count(call) == 1


def test_reap_runs_survives_a_raising_analytics_retention_sweep(monkeypatch):
    calls = []

    def analytics_retention_sweep():
        raise RuntimeError("synthetic retention failure")

    monkeypatch.setattr(run_service, "_extra_reaper_sweeps", [])
    run_service.register_reaper_sweep(analytics_retention_sweep)
    run_service.register_reaper_sweep(lambda: calls.append("still swept"))
    run_service.reap_runs()  # must not raise
    assert calls == ["still swept"]


def test_v2_sweep_archives_terminal_backends(monkeypatch):
    """After the sweep, a finished v2 run: frees its backend (tombstone), has a
    terminal row, and still answers status/decisions/replay over HTTP."""
    key, sid, agent_id = _agent("sweeper")
    run_id = f"run_sweep_{uuid.uuid4().hex[:6]}"
    fake = FakeBackend(run_id=run_id, total_steps=1, session_id=sid)
    run_store.create_run(
        run_id=run_id, agent_id=agent_id, agent_version_id=None, session_id=sid,
        environment_id=None, environment_type="backtest", config={},
        backtest_id=None, status="running",
    )
    runs_mod.register_run(run_id, fake, sid, agent_id)
    # Drive to completion, with an idempotent ack recorded like the live path.
    ack = {"accepted": True, "executed": [], "rejected": [],
           "decision_source": "external_agent", "next_step": 1,
           "status": "completed", "run_id": run_id, "metrics": None}
    db.put_idempotency(run_id, 0, "key-before-archive", ack)
    fake.apply_decisions([])
    assert fake._status == "completed"

    runs_mod.reap_v2_runs()

    with runs_mod._lock:
        archived = runs_mod._runs[run_id]["backend"]
    assert isinstance(archived, ArchivedBacktestBackend)
    assert run_store.get_run(run_id)["status"] == "completed"

    # Reads still answer.
    status = client.get(f"/api/v2/runs/{run_id}", headers={"X-API-Key": key})
    assert status.status_code == 200
    assert status.json()["status"] == "completed"
    # A new decision is refused with the live path's terminal error shape.
    submit = client.post(
        f"/api/v2/runs/{run_id}/decisions",
        json={"idempotency_key": "fresh-key", "actions": []},
        headers={"X-API-Key": key},
    )
    assert submit.status_code == 409
    assert submit.json()["error"]["code"] == "invalid_status"
    # An idempotent replay still returns the recorded ack (DB-backed).
    replay = client.post(
        f"/api/v2/runs/{run_id}/decisions",
        json={"idempotency_key": "key-before-archive", "actions": []},
        headers={"X-API-Key": key},
    )
    assert replay.status_code == 200
    assert replay.json()["status"] == "completed"


def test_v2_sweep_drives_elapsed_deadlines(monkeypatch):
    """Abandoned lockstep runs must be drained (advance()) by the sweep, not
    sit waiting_decision forever."""
    key, sid, agent_id = _agent("drainer")
    run_id = f"run_drain_{uuid.uuid4().hex[:6]}"

    class _Abandoned(FakeBackend):
        def __init__(self):
            super().__init__(run_id=run_id, total_steps=1, session_id=sid)
            self.advanced = 0

        def advance(self):
            # Deadline elapsed → engine auto-holds to completion.
            self.advanced += 1
            self._status = "completed"

        def is_active(self):
            return self._status not in ("completed", "failed", "closed")

    backend = _Abandoned()
    run_store.create_run(
        run_id=run_id, agent_id=agent_id, agent_version_id=None, session_id=sid,
        environment_id=None, environment_type="backtest", config={},
        backtest_id=None, status="running",
    )
    runs_mod.register_run(run_id, backend, sid, agent_id)

    runs_mod.reap_v2_runs()

    assert backend.advanced >= 1
    assert run_store.get_run(run_id)["status"] == "completed"


# ---------------------------------------------------------------------------
# 4. Restart visibility (orphan recovery + rehydration)
# ---------------------------------------------------------------------------


def test_v2_run_visible_after_restart_as_failed(monkeypatch):
    """Crash-orphaned v2 run: recovery fails the row, and the v2 surface
    rehydrates it as a terminal tombstone instead of 404ing the owner."""
    key, sid, agent_id = _agent("restart-owner")
    run_id = _create_v2_run(key, monkeypatch)

    # Simulate a process restart: in-memory registry gone.
    with runs_mod._lock:
        runs_mod._runs.pop(run_id, None)
    monkeypatch.setattr(run_service, "RUN_RECOVERY_ON_STARTUP", True)
    run_service.recover_orphaned_runs()
    assert run_store.get_run(run_id)["status"] == "failed"

    resp = client.get(f"/api/v2/runs/{run_id}", headers={"X-API-Key": key})
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "failed"


def test_v2_rehydration_is_not_an_existence_oracle(monkeypatch):
    """Terminal-row rehydration must answer a foreign prober exactly like a
    missing run — same code, same message template."""
    key_owner, _, _ = _agent("rehydrate-owner")
    key_probe, _, _ = _agent("rehydrate-prober")
    run_id = _create_v2_run(key_owner, monkeypatch)
    with runs_mod._lock:
        runs_mod._runs.pop(run_id, None)
    run_store.update_run(run_id, status="failed")

    denied = client.get(f"/api/v2/runs/{run_id}", headers={"X-API-Key": key_probe})
    missing = client.get(
        f"/api/v2/runs/run_missing_{uuid.uuid4().hex[:6]}",
        headers={"X-API-Key": key_probe},
    )
    assert denied.status_code == missing.status_code == 404
    assert denied.json()["error"]["code"] == missing.json()["error"]["code"] == "run_not_found"


# ---------------------------------------------------------------------------
# 5. Instance-id / heartbeat hardening (multi-worker recovery)
# ---------------------------------------------------------------------------


def test_create_run_stamps_instance_and_heartbeat(tmp_path):
    store = run_store_module.RunStore(tmp_path / "hb.db")
    rec = store.create_run(
        agent_id="ag_hb", agent_version_id=None, session_id="sess",
        environment_id=None, environment_type="backtest", config={},
        backtest_id=None, status="running",
    )
    conn = sqlite3.connect(str(store.db_path))
    row = conn.execute(
        "SELECT owner_instance, heartbeat_at FROM protocol_runs WHERE run_id = ?",
        (rec["run_id"],),
    ).fetchone()
    conn.close()
    assert row[0] == run_store_module.INSTANCE_ID
    assert row[1]  # heartbeat stamped at creation


def test_fail_stale_runs_spares_fresh_heartbeats(tmp_path):
    store = run_store_module.RunStore(tmp_path / "stale.db")
    fresh = store.create_run(
        agent_id="ag_live", agent_version_id=None, session_id="s",
        environment_id=None, environment_type="backtest", config={},
        backtest_id=None, status="running",
    )
    stale = store.create_run(
        agent_id="ag_dead", agent_version_id=None, session_id="s",
        environment_id=None, environment_type="backtest", config={},
        backtest_id=None, status="running",
    )
    conn = sqlite3.connect(str(store.db_path))
    conn.execute(
        "UPDATE protocol_runs SET heartbeat_at = '2020-01-01T00:00:00+00:00' "
        "WHERE run_id = ?", (stale["run_id"],),
    )
    conn.commit()
    conn.close()

    failed = store.fail_stale_runs(stale_seconds=300)

    assert failed == 1
    assert store.get_run(stale["run_id"])["status"] == "failed"
    assert store.get_run(fresh["run_id"])["status"] == "running"


def test_fail_stale_runs_treats_legacy_null_heartbeat_as_stale(tmp_path):
    """Rows written before the heartbeat column existed must still be
    recoverable — fall back to updated_at for staleness."""
    store = run_store_module.RunStore(tmp_path / "legacy.db")
    rec = store.create_run(
        agent_id="ag_legacy", agent_version_id=None, session_id="s",
        environment_id=None, environment_type="backtest", config={},
        backtest_id=None, status="running",
    )
    conn = sqlite3.connect(str(store.db_path))
    conn.execute(
        "UPDATE protocol_runs SET heartbeat_at = NULL, "
        "updated_at = '2020-01-01T00:00:00+00:00' WHERE run_id = ?",
        (rec["run_id"],),
    )
    conn.commit()
    conn.close()

    assert store.fail_stale_runs(stale_seconds=300) == 1
    assert store.get_run(rec["run_id"])["status"] == "failed"


def test_runstore_migrates_legacy_schema(tmp_path):
    """A DB created before the heartbeat columns must gain them on open."""
    path = tmp_path / "old.db"
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE protocol_runs (
            run_id TEXT PRIMARY KEY,
            agent_id TEXT,
            agent_version_id TEXT,
            session_id TEXT NOT NULL,
            environment_id TEXT,
            environment_type TEXT,
            config TEXT NOT NULL DEFAULT '{}',
            backtest_id TEXT,
            result_run_id TEXT,
            status TEXT NOT NULL DEFAULT 'created',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        "INSERT INTO protocol_runs (run_id, session_id, status) "
        "VALUES ('run_old', 'sess', 'running')"
    )
    conn.commit()
    conn.close()

    store = run_store_module.RunStore(path)  # must migrate, not crash
    conn = sqlite3.connect(str(path))
    cols = {row[1] for row in conn.execute("PRAGMA table_info(protocol_runs)")}
    conn.close()
    assert {"owner_instance", "heartbeat_at"} <= cols
    assert store.get_run("run_old")["status"] == "running"


def _make_completed_v2_run(monkeypatch_none, name):
    """A finished-but-not-yet-archived v2 run: terminal backend still live in
    the registry, protocol_runs row already 'completed' (the state between a
    final decision and the reaper's next sweep)."""
    key, sid, agent_id = _agent(name)
    run_id = f"run_{name}_{uuid.uuid4().hex[:6]}"
    fake = FakeBackend(run_id=run_id, total_steps=1, session_id=sid)
    run_store.create_run(
        run_id=run_id, agent_id=agent_id, agent_version_id=None, session_id=sid,
        environment_id=None, environment_type="backtest", config={},
        backtest_id=None, status="running",
    )
    runs_mod.register_run(run_id, fake, sid, agent_id)
    fake.apply_decisions([])  # completes the fake
    run_store.update_run(run_id, status="completed", result_run_id=run_id)
    return key, run_id, fake


# ---------------------------------------------------------------------------
# 6. Adversarial-review round (confirmed findings)
# ---------------------------------------------------------------------------


def test_cancel_after_completion_does_not_clobber_the_ledger(monkeypatch):
    """CRITICAL finding: a cancel arriving after the final decision (but
    before the sweep archives the backend) must not downgrade the row's
    'completed' to 'closed' — the run genuinely finished; report the truth."""
    key, run_id, fake = _make_completed_v2_run(monkeypatch, "clobber")

    resp = client.post(f"/api/v2/runs/{run_id}/cancel", headers={"X-API-Key": key})

    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"  # the truth, not "closed"
    assert run_store.get_run(run_id)["status"] == "completed"
    assert fake._status == "completed"  # in-memory state not clobbered either


def test_cancel_of_archived_run_reports_true_terminal_state(monkeypatch):
    key, run_id, fake = _make_completed_v2_run(monkeypatch, "arccancel")
    runs_mod.reap_v2_runs()  # archive it

    resp = client.post(f"/api/v2/runs/{run_id}/cancel", headers={"X-API-Key": key})

    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"
    assert run_store.get_run(run_id)["status"] == "completed"


def test_backtest_backend_cancel_never_clobbers_a_terminal_session():
    """The in-memory guard: cancel() on an already-completed session must not
    flip it to 'closed' (a race between a finalizing submit and a cancel
    would otherwise corrupt what every later read derives its state from)."""
    import threading

    from dashboard.backend.execution.backtest_backend import BacktestBackend

    class _Session:
        _step_lock = threading.Lock()
        status = "completed"

    backend = BacktestBackend.__new__(BacktestBackend)
    backend.run_id = "run_cancel_guard"
    backend.session = _Session()
    backend.cancel()
    assert backend.session.status == "completed"

    backend.session.status = "waiting_decision"
    backend.cancel()
    assert backend.session.status == "closed"


def test_archive_run_keeps_backend_live_when_row_write_fails(monkeypatch):
    """If the terminal row update fails (locked DB), swapping in the
    tombstone anyway would freeze the row in an active status with nothing
    left to retry — the swap must be gated on the write landing."""
    key, sid, agent_id = _agent("swap-order")
    run_id = f"run_swap_{uuid.uuid4().hex[:6]}"
    fake = FakeBackend(run_id=run_id, total_steps=1, session_id=sid)
    run_store.create_run(
        run_id=run_id, agent_id=agent_id, agent_version_id=None, session_id=sid,
        environment_id=None, environment_type="backtest", config={},
        backtest_id=None, status="running",
    )
    runs_mod.register_run(run_id, fake, sid, agent_id)
    fake.apply_decisions([])

    def _boom(*a, **k):
        raise RuntimeError("db locked")

    monkeypatch.setattr(run_store, "update_run", _boom)
    runs_mod.reap_v2_runs()
    with runs_mod._lock:
        still_live = runs_mod._runs[run_id]["backend"]
    assert not isinstance(still_live, ArchivedBacktestBackend), (
        "backend must stay live so the next sweep can retry the row write"
    )

    monkeypatch.undo()
    runs_mod.reap_v2_runs()
    with runs_mod._lock:
        archived = runs_mod._runs[run_id]["backend"]
    assert isinstance(archived, ArchivedBacktestBackend)
    assert run_store.get_run(run_id)["status"] == "completed"


def test_v2_row_transitions_to_running_after_load(monkeypatch):
    """Rows must not sit in 'loading' for the whole active life of the run —
    a successful market-data load flips them to 'running'."""
    import time as _time

    from dashboard.backend.execution.backtest_backend import BacktestBackend

    import threading

    class _InstantLoadSession:
        def __init__(self):
            self._step_lock = threading.Lock()
            self.status = "loading"
            # start_background_load's peek fast path reads these; the store is
            # reset per test, so peek cold-misses and _load runs as intended.
            self.start_date = "2026-04-15"
            self.end_date = "2026-04-16"

        def load_market_data(self):
            self.status = "waiting_decision"

    run_id = f"run_load_{uuid.uuid4().hex[:6]}"
    run_store.create_run(
        run_id=run_id, agent_id="ag_load", agent_version_id=None,
        session_id="sess_load", environment_id=None, environment_type="backtest",
        config={}, backtest_id=None, status="loading",
    )
    backend = BacktestBackend.__new__(BacktestBackend)
    backend.run_id = run_id
    backend.session = _InstantLoadSession()
    backend.start_background_load()
    deadline = _time.time() + 5
    while _time.time() < deadline:
        if run_store.get_run(run_id)["status"] == "running":
            break
        _time.sleep(0.02)
    assert run_store.get_run(run_id)["status"] == "running"


def test_runstore_survives_partially_migrated_schema(tmp_path):
    """A concurrently-started sibling may have added one heartbeat column
    already — startup must add only the missing one, without crashing."""
    path = tmp_path / "partial.db"
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE protocol_runs (
            run_id TEXT PRIMARY KEY,
            agent_id TEXT,
            agent_version_id TEXT,
            session_id TEXT NOT NULL,
            environment_id TEXT,
            environment_type TEXT,
            config TEXT NOT NULL DEFAULT '{}',
            backtest_id TEXT,
            result_run_id TEXT,
            status TEXT NOT NULL DEFAULT 'created',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            owner_instance TEXT
        )
        """
    )
    conn.commit()
    conn.close()

    run_store_module.RunStore(path)  # must add only heartbeat_at, without crashing
    conn = sqlite3.connect(str(path))
    cols = {row[1] for row in conn.execute("PRAGMA table_info(protocol_runs)")}
    conn.close()
    assert {"owner_instance", "heartbeat_at"} <= cols


def test_runstore_migration_tolerates_losing_the_alter_race(tmp_path):
    """Two workers can both probe (columns missing) and both ALTER; the loser
    gets 'duplicate column name'. That means the column exists — the goal —
    so it must be swallowed, not crash the process at startup."""
    store = run_store_module.RunStore(tmp_path / "race.db")  # columns already exist
    conn = sqlite3.connect(str(store.db_path))
    try:
        # Simulate the raced probe: a stale 'nothing exists yet' snapshot.
        store._add_recovery_columns(conn.cursor(), existing_columns=set())
    finally:
        conn.close()


def test_rehydrated_completed_run_reports_step_counts(monkeypatch):
    """from_record used to lose step_index/total_steps (0/0) for
    restart-rehydrated runs; for completed runs the decision log has one row
    per executed step, so the counts are recoverable."""
    key, sid, agent_id = _agent("rehydrate-steps")
    run_id = f"run_steps_{uuid.uuid4().hex[:6]}"
    run_store.create_run(
        run_id=run_id, agent_id=agent_id, agent_version_id=None, session_id=sid,
        environment_id=None, environment_type="backtest", config={},
        backtest_id=None, status="running",
    )
    run_store.update_run(run_id, status="completed", result_run_id=run_id)
    db.insert_decisions(run_id, [
        {"step_index": i, "timestamp": f"2026-04-15T1{i}:30:00+00:00",
         "decision_source": "external_agent", "actions_submitted": [],
         "actions_executed": 0}
        for i in range(3)
    ])

    resp = client.get(f"/api/v2/runs/{run_id}", headers={"X-API-Key": key})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["step_index"] == 3
    assert body["total_steps"] == 3


def test_reap_pass_heartbeats_live_v1_runs(monkeypatch):
    """The reaper keeps live runs' heartbeats fresh so a sibling worker's
    stale-recovery never fails them."""
    import dashboard.backend.domain.backtesting.external_run_service as ebs

    rec = run_store.create_run(
        agent_id="ag_hb_live", agent_version_id=None, session_id="sess_hb",
        environment_id="us-equity-hourly-v1", environment_type="backtest",
        config={}, backtest_id="bt_hb_live", status="running",
    )
    conn = sqlite3.connect(str(run_store.db_path))
    conn.execute(
        "UPDATE protocol_runs SET heartbeat_at = '2020-01-01T00:00:00+00:00' "
        "WHERE run_id = ?", (rec["run_id"],),
    )
    conn.commit()
    conn.close()

    class _LiveSession:
        def drain_expired(self):
            return "waiting_decision"

        def get_status(self):
            return {"status": "waiting_decision"}

    run = run_service.ProtocolRun(
        record=rec, environment={"type": "backtest", "constraints": {}},
    )
    monkeypatch.setitem(ebs._sessions, "bt_hb_live", _LiveSession())
    with run_service._registry_lock:
        run_service._runs[rec["run_id"]] = run
    try:
        run_service.reap_runs()
    finally:
        with run_service._registry_lock:
            run_service._runs.pop(rec["run_id"], None)

    conn = sqlite3.connect(str(run_store.db_path))
    (hb,) = conn.execute(
        "SELECT heartbeat_at FROM protocol_runs WHERE run_id = ?",
        (rec["run_id"],),
    ).fetchone()
    conn.close()
    assert hb and hb > "2020-01-02", "reaper must refresh live runs' heartbeats"
