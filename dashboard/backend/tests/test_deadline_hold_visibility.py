"""T2: decision-deadline auto-holds must be printed, never silent.

``_maybe_apply_timeout`` reattributes a step to ``decision_source ==
"timeout_hold"`` with no output whatsoever. A published equity curve
containing auto-held steps is not the agent's curve. This file pins the
visibility contract at all three sites that can trigger a hold
(``get_current_step``, ``drain_expired``, ``get_status``) plus the live
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
