"""T2: decision-deadline auto-holds must be printed, never silent.

``_maybe_apply_timeout`` reattributes a step to ``decision_source ==
"timeout_hold"`` with no output whatsoever. A published equity curve
containing auto-held steps is not the agent's curve. This file pins the
visibility contract at all four sites that can trigger a hold
(``get_current_step``, ``drain_expired``, ``get_status``,
``submit_decisions``), which all go through ``_apply_expired_holds``, plus
the routes that reach them. That includes the live
protocol path (``run_service.get_step``), which calls ``get_status()``
first — applying the hold and advancing ``step_index`` before
``get_current_step``'s own ``step_index == seq`` guard is ever reached, so an
implementation instrumented only inside ``get_current_step`` stays silent on
the path production traffic actually takes.
"""

import time
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

import dashboard.backend.database as db_module
import dashboard.backend.domain.backtesting.external_run_service as ebs
from dashboard.backend.domain.backtesting import baseline_worker as bw


def _synth_bars(symbols, start, end):
    idx = pd.date_range(start=start, end=str(end) + " 23:59", freq="1h", tz="UTC")
    et = idx.tz_convert("US/Eastern")
    mask = (et.dayofweek < 5) & (
        ((et.hour > 9) & (et.hour < 16)) | ((et.hour == 16) & (et.minute == 0))
    )
    idx = idx[mask]
    data = {}
    for si, sym in enumerate(sorted(symbols)):
        n = len(idx)
        close = 100.0 + si + np.linspace(0, 1.0, n)
        data[sym] = pd.DataFrame(
            {"open": close, "high": close + 0.5, "low": close - 0.5,
             "close": close, "volume": 1000.0}, index=idx)
    return data


class _Loader:
    def fetch_bars(self, symbols, start, end):
        return _synth_bars(symbols, start, end)


@pytest.fixture
def session(monkeypatch, tmp_path):
    test_db = db_module.BacktestDatabase(db_path=tmp_path / "hold_visibility.db")
    monkeypatch.setattr(db_module, "db", test_db)
    monkeypatch.setattr(ebs, "db", test_db)
    monkeypatch.setattr(ebs, "AlpacaDataLoader", _Loader)
    monkeypatch.setattr(bw.HourlyBacktester, "run_buyhold_baseline",
                        lambda self: (None, None))
    monkeypatch.setattr(bw.HourlyBacktester, "run_djia_baseline",
                        lambda self: (None, None))
    # Two trading days of synthetic hourly bars -> ~13 steps, comfortably
    # more than the handful any single test expires, so a partial drain
    # never accidentally completes the run out from under an assertion.
    s = ebs.ExternalBacktestSession(
        backtest_id="bt_visible", session_id="sess_v", agent_name="agent-v",
        model_name="m", start_date="2026-04-15", end_date="2026-04-17",
    )
    s.load_market_data()
    return s


class _JumpyClock:
    """Fake ``ebs._utcnow`` that advances by ``jump`` on each of its first
    ``n_jumps`` calls, then holds steady.

    ``_maybe_apply_timeout``'s while loop calls ``_utcnow()`` exactly twice
    per hold it applies: once to compare "now" against the current step's
    deadline, and once inside ``_advance_step`` -> ``_open_current_step`` to
    stamp the *next* step's ``opened_at``. Installing ``jump = timeout + 1``
    guarantees every such comparison sees an already-expired deadline;
    capping the advance at ``2 * n_holds`` calls means the freshly-stamped
    deadline of the step *after* the last intended hold is never exceeded,
    so the loop (or the single unlooped call in ``get_status``) stops there
    deterministically — no real-clock races, no sleeps.
    """

    def __init__(self, start, jump_seconds, n_jumps):
        self.now = start
        self.jump = timedelta(seconds=jump_seconds)
        self.n_jumps = n_jumps
        self.calls = 0

    def __call__(self):
        if self.calls < self.n_jumps:
            self.now = self.now + self.jump
            self.calls += 1
        return self.now


def _expire_next_n_steps(monkeypatch, session, n, timeout_seconds=60):
    """Force exactly the next ``n`` steps opened from ``session``'s current
    step to auto-hold in one loop/call, then stop expiring further ones."""
    monkeypatch.setattr(ebs, "DECISION_TIMEOUT_SECONDS", timeout_seconds)
    clock = _JumpyClock(
        start=session.step_opened_at,
        jump_seconds=timeout_seconds + 1,
        n_jumps=2 * n,
    )
    monkeypatch.setattr(ebs, "_utcnow", clock)
    return clock


# ---------------------------------------------------------------------------
# get_current_step: one line per poll, carrying the range actually held
# ---------------------------------------------------------------------------


def test_current_step_multiple_holds_single_line_with_actual_range(
    session, monkeypatch, capsys
):
    s = session
    assert s.total_steps > 4  # headroom so this poll can't run to completion
    capsys.readouterr()
    _expire_next_n_steps(monkeypatch, s, n=3)

    step = s.get_current_step()

    out = capsys.readouterr().out
    assert out.count("decision deadline") == 1
    assert "auto-held 3 step(s) for bt_visible" in out
    assert "agent=agent-v" in out
    # The range that was ACTUALLY held (0..2), not self.step_index (3) read
    # after the loop — the one assertion that catches the post-drain-index
    # bug (_advance_step increments step_index inside the loop).
    assert "steps=0..2" in out
    assert "total_holds=3" in out
    assert "— these steps are NOT the agent's decisions" in out
    assert s.timeout_holds == 3
    assert s.step_index == 3
    assert s.status == "waiting_decision"
    assert step["status"] == "waiting_decision"


def test_current_step_no_expired_step_prints_nothing(session, capsys):
    s = session
    capsys.readouterr()
    step = s.get_current_step()
    out = capsys.readouterr().out
    assert "decision deadline" not in out
    assert s.timeout_holds == 0
    assert step["status"] == "waiting_decision"


# ---------------------------------------------------------------------------
# drain_expired: the reaper's path — no agent polling at all
# ---------------------------------------------------------------------------


def test_drain_expired_multiple_holds_single_line_with_actual_range(
    session, monkeypatch, capsys
):
    s = session
    assert s.total_steps > 4
    capsys.readouterr()
    _expire_next_n_steps(monkeypatch, s, n=3)

    status = s.drain_expired()

    out = capsys.readouterr().out
    assert out.count("decision deadline") == 1
    assert "auto-held 3 step(s) for bt_visible" in out
    assert "steps=0..2" in out
    assert "total_holds=3" in out
    assert s.timeout_holds == 3
    assert s.step_index == 3
    assert status == "waiting_decision"


def test_drain_expired_no_expired_step_prints_nothing(session, capsys):
    s = session
    capsys.readouterr()
    status = s.drain_expired()
    out = capsys.readouterr().out
    assert "decision deadline" not in out
    assert s.timeout_holds == 0
    assert status == "waiting_decision"


# ---------------------------------------------------------------------------
# get_status: single unlooped _maybe_apply_timeout call — the site the live
# protocol path actually reaches first (runs/service.py get_step / _sync_status)
# ---------------------------------------------------------------------------


def test_get_status_single_hold_single_line(session, monkeypatch, capsys):
    s = session
    capsys.readouterr()
    _expire_next_n_steps(monkeypatch, s, n=1)

    result = s.get_status()

    out = capsys.readouterr().out
    assert out.count("decision deadline") == 1
    assert "auto-held 1 step(s) for bt_visible" in out
    assert "steps=0..0" in out
    assert "total_holds=1" in out
    assert s.timeout_holds == 1
    assert s.step_index == 1
    assert result["status"] == "waiting_decision"


def test_get_status_no_expired_step_prints_nothing(session, capsys):
    s = session
    capsys.readouterr()
    result = s.get_status()
    out = capsys.readouterr().out
    assert "decision deadline" not in out
    assert s.timeout_holds == 0
    assert result["status"] == "waiting_decision"


# ---------------------------------------------------------------------------
# submit_decisions: the fourth auto-hold site (#375). When nothing polls
# between the agent's GET and its late POST, the POST is what holds the step.
# When something else holds it first (the v2 reaper, a legacy /status poll),
# the late POST must be refused, not applied to the next step.
# ---------------------------------------------------------------------------


def test_late_submit_single_hold_single_line(session, monkeypatch, capsys):
    s = session
    capsys.readouterr()
    _expire_next_n_steps(monkeypatch, s, n=1)

    result = s.submit_decisions({"actions": []})

    out = capsys.readouterr().out
    assert out.count("decision deadline") == 1
    assert "auto-held 1 step(s) for bt_visible" in out
    assert "steps=0..0" in out
    assert "total_holds=1" in out
    assert s.timeout_holds == 1
    assert s.step_index == 1
    # The fields both routes map to 409.
    assert result["accepted"] is False
    assert result["error"] == "step_already_closed"
    assert result["outcome"] == "timeout_hold"
    assert result["step_index"] == 0
    assert result["next_step"] == 1
    assert result["run_id"] is None


def test_on_time_submit_prints_nothing(session, capsys):
    s = session
    capsys.readouterr()
    result = s.submit_decisions({"actions": []})
    out = capsys.readouterr().out
    assert "decision deadline" not in out
    assert s.timeout_holds == 0
    # Without these the test also passes when submit returns early.
    assert result["accepted"] is True
    assert result["step_index"] == 0
    assert s.step_index == 1
    # _open_current_step() used to reset this before the ack read it.
    assert result["decision_source"] == "external_agent"


def _submit_through_to_last_step(s):
    while s.step_index < s.total_steps - 1:
        result = s.submit_decisions({"actions": []})
        assert result["accepted"] is True


def test_late_submit_on_final_step_returns_the_finished_run(session, monkeypatch, capsys):
    s = session
    _submit_through_to_last_step(s)
    capsys.readouterr()
    _expire_next_n_steps(monkeypatch, s, n=1)

    result = s.submit_decisions({"actions": []})

    assert capsys.readouterr().out.count("decision deadline") == 1
    assert s.status == "completed"
    assert result["error"] == "step_already_closed"
    assert result["status"] == "completed"
    # A 409 for a finished run must still say where the result is.
    assert result["run_id"] == s.run_id and s.run_id
    assert result["metrics"] is not None
    assert "compare_url" in result


@pytest.mark.parametrize("holder", ["drain_expired", "get_status"])
def test_submit_after_another_path_held_the_served_step_is_refused(
    session, monkeypatch, capsys, holder
):
    s = session
    s.get_current_step()  # the agent is served step 0
    capsys.readouterr()
    _expire_next_n_steps(monkeypatch, s, n=1)
    getattr(s, holder)()  # the v2 reaper / a legacy status poll wins the race
    assert capsys.readouterr().out.count("decision deadline") == 1
    assert s.step_index == 1

    result = s.submit_decisions({"actions": [
        {"action": "buy", "symbol": "AAPL", "confidence": 0.9,
         "reasoning": "built from step 0", "position_size": 1},
    ]})

    assert result["accepted"] is False
    assert result["error"] == "step_already_closed"
    assert result["outcome"] == "timeout_hold"
    assert result["step_index"] == 0
    assert result["next_step"] == 1
    # Step 1 is untouched: not executed, not held, not printed twice.
    assert s.step_index == 1
    assert s.timeout_holds == 1
    assert "decision deadline" not in capsys.readouterr().out
    assert [e["decision_source"] for e in s.decision_log] == ["timeout_hold"]

    # Fetching step 1 first is how the agent recovers.
    s.get_current_step()
    recovered = s.submit_decisions({"actions": []})
    assert recovered["accepted"] is True


def test_submit_without_fetching_context_still_works(session):
    """The stale check fires only when the served step was HELD. A client
    that POSTs without a GET after deciding a step itself is not refused."""
    s = session
    s.get_current_step()
    first = s.submit_decisions({"actions": []})
    assert first["accepted"] is True
    result = s.submit_decisions({"actions": []})
    assert result["accepted"] is True
    assert result["step_index"] == 1


def test_expected_step_that_moved_on_is_refused(session):
    s = session
    first = s.submit_decisions({"actions": []}, expected_step=0)
    assert first["accepted"] is True

    result = s.submit_decisions({"actions": []}, expected_step=0)

    assert result["error"] == "step_already_closed"
    assert result["outcome"] == "external_agent"
    assert result["step_index"] == 0
    assert s.step_index == 1
    assert s.timeout_holds == 0


# ---------------------------------------------------------------------------
# Through the routes that reach submit_decisions with no poll in between:
# the legacy /api/v1/backtest decisions route and v2's _submit_for.
# ---------------------------------------------------------------------------


def test_legacy_route_late_submit_prints_and_409s(session, monkeypatch, capsys):
    from fastapi.testclient import TestClient

    from dashboard.backend.app import app

    s = session
    monkeypatch.setitem(ebs._sessions, s.backtest_id, s)
    client = TestClient(app)
    headers = {"X-Session-Id": s.session_id}
    url = f"/api/v1/backtest/{s.backtest_id}/steps/current"
    served = client.get(url, headers=headers).json()
    assert served["step_index"] == 0
    capsys.readouterr()
    _expire_next_n_steps(monkeypatch, s, n=1)

    resp = client.post(f"{url}/decisions", json={"actions": []}, headers=headers)

    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["outcome"] == "timeout_hold"
    out = capsys.readouterr().out
    assert out.count("decision deadline") == 1
    assert "steps=0..0" in out


@pytest.fixture
def v2_run(monkeypatch, tmp_path):
    import dashboard.backend.api.v2.runs as runs_mod
    import dashboard.backend.domain.runs.repository as run_store_module
    from dashboard.backend.execution import backtest_backend as bb_mod

    path = tmp_path / "hold_visibility_v2.db"
    test_db = db_module.BacktestDatabase(db_path=path)
    test_runs = run_store_module.RunStore(db_path=path)
    monkeypatch.setattr(db_module, "db", test_db)
    monkeypatch.setattr(ebs, "db", test_db)
    monkeypatch.setattr(runs_mod, "db", test_db)
    monkeypatch.setattr(run_store_module, "run_store", test_runs)
    monkeypatch.setattr(ebs, "AlpacaDataLoader", _Loader)
    monkeypatch.setattr(bb_mod, "load_news_sentiment", lambda *_: ({}, None))
    monkeypatch.setattr(bw.HourlyBacktester, "run_buyhold_baseline",
                        lambda self: (None, None))
    monkeypatch.setattr(bw.HourlyBacktester, "run_djia_baseline",
                        lambda self: (None, None))

    run_id = "run_hold_v2"
    test_runs.create_run(
        run_id=run_id, agent_id=None, agent_version_id=None,
        session_id="sess_v2", environment_id=None, environment_type="backtest",
        config={}, backtest_id=run_id, status="running",
    )
    backend = bb_mod.BacktestBackend(
        run_id=run_id, session_id="sess_v2", agent_name="agent-v2",
        model_name="m", start_date="2026-04-15", end_date="2026-04-17",
    )
    backend.load_blocking()
    monkeypatch.setitem(runs_mod._runs, run_id,
                        {"backend": backend, "session_id": "sess_v2", "agent_id": None})
    return runs_mod, backend, test_db, test_runs


def _audit_steps(test_db, run_id):
    conn = test_db._get_connection()
    try:
        rows = conn.execute(
            "SELECT step_index FROM idempotency_keys WHERE run_id = ?", (run_id,)
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def test_v2_late_submit_prints_and_409s(v2_run, monkeypatch, capsys):
    from dashboard.backend.api.v2.errors import ApiError

    runs_mod, backend, test_db, _ = v2_run
    backend.build_context()
    capsys.readouterr()
    _expire_next_n_steps(monkeypatch, backend.session, n=1)

    with pytest.raises(ApiError) as ei:
        runs_mod._submit_for(backend.run_id, "sess_v2", "k-late", [])

    assert ei.value.code == "step_already_closed" and ei.value.status == 409
    assert ei.value.details["step_index"] == 0
    assert capsys.readouterr().out.count("decision deadline") == 1
    assert _audit_steps(test_db, backend.run_id) == [0]


def test_v2_reaper_hold_between_read_and_apply_refuses_the_decision(
    v2_run, monkeypatch, capsys
):
    """_submit_for reads the step without the session lock. If the reaper
    holds it before apply_decisions runs, the decision must be refused and
    audited under the step it was meant for, not executed on the next one."""
    runs_mod, backend, test_db, _ = v2_run
    from dashboard.backend.api.v2.errors import ApiError

    real_read = backend.current_step_index

    def read_then_reaper_wins():
        step = real_read()
        _expire_next_n_steps(monkeypatch, backend.session, n=1)
        backend.advance()  # v2 reaper -> drain_expired()
        return step

    monkeypatch.setattr(backend, "current_step_index", read_then_reaper_wins)
    capsys.readouterr()

    with pytest.raises(ApiError) as ei:
        runs_mod._submit_for(backend.run_id, "sess_v2", "k-race", [
            {"action": "buy", "symbol": "AAPL", "confidence": 0.9,
             "reasoning": "built from step 0", "position_size": 1},
        ])

    assert ei.value.code == "step_already_closed"
    assert ei.value.details["outcome"] == "timeout_hold"
    assert ei.value.details["step_index"] == 0
    assert backend.session.step_index == 1  # step 1 not consumed
    assert [e["decision_source"] for e in backend.session.decision_log] == ["timeout_hold"]
    assert _audit_steps(test_db, backend.run_id) == [0]
    # One hold, one line: printed by the reaper, not again by the refusal.
    assert capsys.readouterr().out.count("decision deadline") == 1


def test_v2_late_submit_on_final_step_marks_the_run_completed(v2_run, monkeypatch):
    from dashboard.backend.api.v2.errors import ApiError

    runs_mod, backend, _, test_runs = v2_run
    _submit_through_to_last_step(backend.session)
    assert test_runs.get_run(backend.run_id)["status"] == "running"
    _expire_next_n_steps(monkeypatch, backend.session, n=1)

    with pytest.raises(ApiError) as ei:
        runs_mod._submit_for(backend.run_id, "sess_v2", "k-final", [])

    assert ei.value.code == "step_already_closed"
    assert ei.value.details["status"] == "completed"
    # Not left 'running' (and counted against the caps) until the reaper.
    assert test_runs.get_run(backend.run_id)["status"] == "completed"


# ---------------------------------------------------------------------------
# Through the router: run_service.get_step calls session.get_status() FIRST,
# which applies the hold and advances step_index, so its own
# `session.step_index == seq` guard then fails and get_current_step() is
# never reached for that poll. A test that only drives get_current_step
# would pass against an implementation silent on this path.
# ---------------------------------------------------------------------------


@pytest.fixture
def router_run(monkeypatch, tmp_path):
    import dashboard.backend.domain.runs.repository as run_store_module
    import dashboard.backend.domain.runs.service as run_service

    test_db = db_module.BacktestDatabase(db_path=tmp_path / "hold_visibility_router.db")
    test_runs = run_store_module.RunStore(db_path=tmp_path / "hold_visibility_router.db")

    monkeypatch.setattr(db_module, "db", test_db)
    monkeypatch.setattr(ebs, "db", test_db)
    monkeypatch.setattr(run_service, "db", test_db)
    monkeypatch.setattr(run_store_module, "run_store", test_runs)
    monkeypatch.setattr(run_service, "run_store", test_runs)
    monkeypatch.setattr(ebs, "AlpacaDataLoader", _Loader)
    monkeypatch.setattr(bw.HourlyBacktester, "run_buyhold_baseline",
                        lambda self: (None, None))
    monkeypatch.setattr(bw.HourlyBacktester, "run_djia_baseline",
                        lambda self: (None, None))
    monkeypatch.setattr(ebs, "DECISION_TIMEOUT_SECONDS", 300)
    # Isolate the in-memory run/session registries from any other test.
    monkeypatch.setattr(run_service, "_runs", {})
    monkeypatch.setattr(ebs, "_sessions", {})

    # No agent_id -> resolve_owner_cap_context returns None immediately
    # (no account, no Postgres round-trip needed for this test).
    agent = {"session_id": "sess_router", "name": "router-agent", "model_name": "m"}
    view = run_service.create_run(
        agent=agent,
        agent_version=None,
        environment_id="us-equity-hourly-v1",
        config={"start_date": "2026-04-15", "end_date": "2026-04-17"},
    )
    run_id = view["run_id"]

    # start_backtest loads market data on a background thread; poll until
    # step 0 is open. This also mints run.step_seq's entry for it, which
    # get_step() needs to resolve step_id -> sequence.
    first_step = None
    for _ in range(50):
        first_step = run_service.get_next_step(run_id)
        if first_step.get("status") != "loading":
            break
        time.sleep(0.05)
    assert first_step is not None and "step_id" in first_step, first_step
    step_id = first_step["step_id"]

    return run_service, run_id, step_id


def test_router_get_step_prints_on_deadline(router_run, monkeypatch, capsys):
    run_service, run_id, step_id = router_run
    backtest_id = run_service.run_store.get_run(run_id)["backtest_id"]

    capsys.readouterr()
    # Expire the currently-open step regardless of when it was really
    # opened, deterministically, the same way test_deadline_and_holds.py
    # forces expiry (monkeypatching the module constant).
    monkeypatch.setattr(ebs, "DECISION_TIMEOUT_SECONDS", -1)

    result = run_service.get_step(run_id, step_id)

    out = capsys.readouterr().out
    assert "decision deadline" in out
    assert f"for {backtest_id}" in out
    # get_status() applied the hold and advanced step_index past seq, so
    # get_step falls back to the historical view for the now-superseded step.
    assert result["status"] == "timed_out"


def test_router_submit_refused_when_reaper_holds_after_the_step_check(
    router_run, monkeypatch, capsys
):
    """submit_decision checks seq under run.lock, but the reaper's
    drain_expired() takes only the session lock. A hold landing between the
    check and the engine call must refuse the decision, not apply it to
    seq + 1."""
    from dashboard.backend.domain.runs.protocol import DecisionIn, ProtocolError

    run_service, run_id, step_id = router_run
    session = run_service._get_run(run_id).session()
    # A v1 step_id names its step explicitly; isolate that pin from the
    # served-step inference the legacy route relies on.
    session.served_step_index = None
    real_trade_count = session.trade_count

    def reaper_wins():  # called just before session.submit_decisions
        _expire_next_n_steps(monkeypatch, session, n=1)
        session.drain_expired()
        return real_trade_count()

    monkeypatch.setattr(session, "trade_count", reaper_wins)
    capsys.readouterr()

    with pytest.raises(ProtocolError) as ei:
        run_service.submit_decision(
            run_id, step_id, DecisionIn(idempotency_key="k-v1-race")
        )

    assert ei.value.code == "decision_deadline_exceeded"
    assert session.step_index == 1
    assert [e["decision_source"] for e in session.decision_log] == ["timeout_hold"]
    assert capsys.readouterr().out.count("decision deadline") == 1


def test_router_get_step_no_deadline_prints_nothing(router_run, capsys):
    run_service, run_id, step_id = router_run
    capsys.readouterr()
    run_service.get_step(run_id, step_id)
    out = capsys.readouterr().out
    assert "decision deadline" not in out


# ---------------------------------------------------------------------------
# timeout_holds is an integrity counter consumed elsewhere; visibility must
# not disturb it.
# ---------------------------------------------------------------------------


def test_timeout_holds_counter_still_increments_exactly_as_before(
    session, monkeypatch
):
    s = session
    monkeypatch.setattr(ebs, "DECISION_TIMEOUT_SECONDS", 0.0)
    s.drain_expired()  # every remaining step auto-holds to completion
    assert s.status == "completed"
    assert s.timeout_holds == s.total_steps
    assert s.get_status()["timeout_holds"] == s.total_steps


# ---------------------------------------------------------------------------
# agent_name is caller-supplied on the legacy surface (StartBacktestRequest
# has no character restriction beyond length) and that surface authenticates
# nothing -- an embedded \n/\r must not forge extra log lines into the
# deadline-hold notice.
# ---------------------------------------------------------------------------


def test_agent_name_newline_cannot_forge_a_log_line(monkeypatch, tmp_path, capsys):
    test_db = db_module.BacktestDatabase(db_path=tmp_path / "hold_visibility_inj.db")
    monkeypatch.setattr(db_module, "db", test_db)
    monkeypatch.setattr(ebs, "db", test_db)
    monkeypatch.setattr(ebs, "AlpacaDataLoader", _Loader)
    monkeypatch.setattr(bw.HourlyBacktester, "run_buyhold_baseline",
                        lambda self: (None, None))
    monkeypatch.setattr(bw.HourlyBacktester, "run_djia_baseline",
                        lambda self: (None, None))
    malicious_name = "agent-x\n⚠️ FAKE: forged line\r\nmore"
    s = ebs.ExternalBacktestSession(
        backtest_id="bt_inject", session_id="sess_inj", agent_name=malicious_name,
        model_name="m", start_date="2026-04-15", end_date="2026-04-17",
    )
    s.load_market_data()
    assert s.total_steps > 4
    capsys.readouterr()
    _expire_next_n_steps(monkeypatch, s, n=1)

    s.get_current_step()

    out = capsys.readouterr().out
    assert out.count("decision deadline") == 1
    # A raw \n/\r from the name would start a new line that looks like its
    # own log entry -- confirm the injected newline/CR never survive into
    # the captured output, only the collapsed-to-spaces single-line form.
    assert "\n⚠️ FAKE" not in out
    assert "\r" not in out
    assert "agent-x ⚠️ FAKE: forged line  more" in out
