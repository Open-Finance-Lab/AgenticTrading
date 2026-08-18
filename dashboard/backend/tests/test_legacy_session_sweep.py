"""T3: TTL sweep for terminal legacy /api/v1/backtest/* sessions.

``_sessions`` is written by ``start_backtest`` for both the legacy surface
and the protocol surface, and unlike the protocol run registry (``_runs``,
reaped by walking it directly) nothing else ever drops a legacy entry once
it goes terminal -- each one pins a loaded bar window forever.
``sweep_terminal_sessions`` rides the existing 60s reaper pass, stamping
``terminal_seen_at`` on first sighting and evicting once
``LEGACY_SESSION_RETENTION_SECONDS`` has elapsed since then.

Uses a lightweight fake session (only the attributes the sweep touches)
rather than a real ``ExternalBacktestSession`` for the pure sweep-logic
cases -- no market data or DB needed to exercise eviction timing. The
persisted-row test below is the one that needs a real DB, since it pins the
safety claim in ``evict_session``'s docstring: a completed run's data
survives its in-memory session being dropped.
"""

from datetime import datetime, timedelta, timezone

import pytest

import dashboard.backend.database as db_module
import dashboard.backend.domain.backtesting.external_run_service as ebs
import dashboard.backend.domain.runs.service as run_service


class _FakeSession:
    """Stand-in for ExternalBacktestSession carrying only what the sweep
    reads/writes: ``status`` and ``terminal_seen_at``."""

    def __init__(self, status):
        self.status = status
        self.terminal_seen_at = None


@pytest.fixture(autouse=True)
def isolated_sessions(monkeypatch):
    """Every test gets its own ``_sessions`` dict so it can't see leftovers
    from -- or leak state into -- any other test in the suite."""
    monkeypatch.setattr(ebs, "_sessions", {})


@pytest.fixture
def clock(monkeypatch):
    """A controllable stand-in for ``_utcnow`` -- a mutable single-element
    list so tests can advance it after monkeypatch has captured the lambda."""
    t = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
    monkeypatch.setattr(ebs, "_utcnow", lambda: t[0])
    return t


def test_terminal_session_not_evicted_on_first_sweep(monkeypatch, clock):
    # TTL=0 deliberately: with no continue after the first-sighting stamp,
    # age would be (now - now) == 0 >= 0 and evict immediately even though
    # the clock never advanced. A TTL of 300 wouldn't catch that -- 0 < 300
    # regardless -- so this has to use 0 to actually pin "never evict on the
    # same pass a session is first seen terminal".
    monkeypatch.setattr(ebs, "LEGACY_SESSION_RETENTION_SECONDS", 0)
    ebs._sessions["bt_1"] = _FakeSession("completed")

    dropped = ebs.sweep_terminal_sessions()

    assert dropped == 0
    assert "bt_1" in ebs._sessions
    assert ebs._sessions["bt_1"].terminal_seen_at == clock[0]


def test_terminal_session_evicted_after_ttl_elapses(monkeypatch, clock):
    monkeypatch.setattr(ebs, "LEGACY_SESSION_RETENTION_SECONDS", 300)
    ebs._sessions["bt_1"] = _FakeSession("completed")

    ebs.sweep_terminal_sessions()  # first sighting -- stamps, does not evict

    clock[0] += timedelta(seconds=299)
    assert ebs.sweep_terminal_sessions() == 0
    assert "bt_1" in ebs._sessions

    clock[0] += timedelta(seconds=2)  # now 301s since the stamp
    assert ebs.sweep_terminal_sessions() == 1
    assert "bt_1" not in ebs._sessions


def test_non_terminal_session_never_evicted(monkeypatch, clock):
    monkeypatch.setattr(ebs, "LEGACY_SESSION_RETENTION_SECONDS", 0)
    ebs._sessions["bt_running"] = _FakeSession("running")

    for _ in range(5):
        assert ebs.sweep_terminal_sessions() == 0
        clock[0] += timedelta(seconds=1000)

    assert "bt_running" in ebs._sessions
    assert ebs._sessions["bt_running"].terminal_seen_at is None


def test_sweep_is_idempotent_and_returns_drop_count(monkeypatch, clock):
    monkeypatch.setattr(ebs, "LEGACY_SESSION_RETENTION_SECONDS", 0)
    ebs._sessions["bt_a"] = _FakeSession("completed")
    ebs._sessions["bt_b"] = _FakeSession("running")

    assert ebs.sweep_terminal_sessions() == 0  # first sighting of bt_a
    clock[0] += timedelta(seconds=1)
    assert ebs.sweep_terminal_sessions() == 1  # bt_a evicted, TTL=0
    assert ebs.sweep_terminal_sessions() == 0  # idempotent: nothing left
    assert set(ebs._sessions) == {"bt_b"}


def test_persisted_row_readable_after_session_evicted(monkeypatch, tmp_path, clock):
    """Pins the safety claim in ``evict_session``'s docstring: a completed
    run's DB row (and everything built from it) stays readable after the
    sweep drops its in-memory session."""
    monkeypatch.setattr(ebs, "LEGACY_SESSION_RETENTION_SECONDS", 0)
    test_db = db_module.BacktestDatabase(db_path=tmp_path / "sweep_persist.db")
    monkeypatch.setattr(ebs, "db", test_db)
    test_db.insert_run(
        run_id="run_sweep_persist", session_id="sess_persist", agent_name="a",
        mode="safe_trading", start_date="2026-04-15", end_date="2026-04-16",
        initial_equity=10000.0, final_equity=10500.0,
    )
    ebs._sessions["bt_persist"] = _FakeSession("completed")

    before = ebs.get_run_result("run_sweep_persist", "sess_persist")
    assert before is not None
    assert before["run"]["run_id"] == "run_sweep_persist"

    ebs.sweep_terminal_sessions()  # first sighting
    clock[0] += timedelta(seconds=1)
    dropped = ebs.sweep_terminal_sessions()
    assert dropped == 1
    assert "bt_persist" not in ebs._sessions

    after = ebs.get_run_result("run_sweep_persist", "sess_persist")
    assert after is not None
    assert after["run"]["run_id"] == "run_sweep_persist"
    assert after["metrics"]["final_equity"] == 10500.0


def test_cap_count_unaffected_by_sweep(monkeypatch, clock):
    """A terminal session already doesn't count toward
    ``_count_active_locked`` (it filters on TERMINAL_STATUSES), so capacity
    accounting must be identical before and after a sweep pass evicts it."""
    monkeypatch.setattr(ebs, "LEGACY_SESSION_RETENTION_SECONDS", 0)
    ebs._sessions["bt_a"] = _FakeSession("completed")
    ebs._sessions["bt_b"] = _FakeSession("running")

    before = ebs.count_active_sessions()
    assert before == 1  # only bt_b is non-terminal

    ebs.sweep_terminal_sessions()  # first sighting, stamps bt_a
    clock[0] += timedelta(seconds=1)
    ebs.sweep_terminal_sessions()  # evicts bt_a

    after = ebs.count_active_sessions()
    assert after == before == 1


def test_sweep_does_not_raise_through_a_real_reaper_pass(monkeypatch, clock, capsys):
    """Registered sweeps run inside reap_runs' try/except (runs/service.py)
    which swallows any exception into a printed warning while the reaper
    keeps reporting healthy. A raise here would be a silent no-op, not a
    loud failure -- this pins that a genuine terminal session never trips it."""
    monkeypatch.setattr(ebs, "LEGACY_SESSION_RETENTION_SECONDS", 300)
    monkeypatch.setattr(run_service, "_extra_reaper_sweeps", [])
    run_service.register_reaper_sweep(ebs.sweep_terminal_sessions)
    ebs._sessions["bt_reap"] = _FakeSession("completed")

    run_service.reap_runs()

    captured = capsys.readouterr()
    assert "registered sweep failed" not in captured.out
    assert "bt_reap" in ebs._sessions  # first sighting this pass, not evicted


def test_foreign_session_without_status_is_skipped_not_raised(monkeypatch, clock, capsys):
    """A ``_sessions`` entry with no ``status`` attribute must be skipped.

    ``_sessions`` is shared with the protocol surface and is monkeypatched
    directly by other suites: ``test_run_lifecycle_unification.py`` injects a
    fake carrying only ``drain_expired``/``get_status`` -- no ``status`` at
    all -- and then runs a real ``reap_runs()`` pass. A bare ``s.status`` read
    in the sweep raises ``AttributeError`` there, which reap_runs swallows
    into one ``registered sweep failed`` line while continuing to report
    healthy -- the sweep would silently never work. Treat a missing status as
    "not terminal" and leave the entry alone.
    """

    class _NoStatusSession:
        def drain_expired(self):
            return "waiting_decision"

        def get_status(self):
            return {"status": "waiting_decision"}

    monkeypatch.setattr(ebs, "LEGACY_SESSION_RETENTION_SECONDS", 0)
    monkeypatch.setattr(run_service, "_extra_reaper_sweeps", [])
    run_service.register_reaper_sweep(ebs.sweep_terminal_sessions)
    ebs._sessions["bt_foreign"] = _NoStatusSession()
    ebs._sessions["bt_done"] = _FakeSession("completed")

    run_service.reap_runs()

    assert "registered sweep failed" not in capsys.readouterr().out

    clock[0] += timedelta(seconds=1)
    assert ebs.sweep_terminal_sessions() == 1  # bt_done only
    assert "bt_foreign" in ebs._sessions
    assert "bt_done" not in ebs._sessions


# ===========================================================================
# LEGACY_SESSION_RETENTION_SECONDS env-var parsing
# ===========================================================================
#
# Read once at import via a bare ``int(os.getenv(...))``-shaped call -- a
# typo'd operator value must not raise at import, matching
# ``_max_active_dashboard_backtests`` in ``api/routers/backtests.py``.

def test_bad_retention_value_falls_back_instead_of_raising(monkeypatch, capsys):
    monkeypatch.setenv("LEGACY_SESSION_RETENTION_SECONDS", "5m")
    assert (
        ebs._legacy_session_retention_seconds()
        == ebs._DEFAULT_LEGACY_SESSION_RETENTION_SECONDS
    )
    assert "LEGACY_SESSION_RETENTION_SECONDS" in capsys.readouterr().out


def test_unset_retention_value_uses_the_default(monkeypatch):
    monkeypatch.delenv("LEGACY_SESSION_RETENTION_SECONDS", raising=False)
    assert (
        ebs._legacy_session_retention_seconds()
        == ebs._DEFAULT_LEGACY_SESSION_RETENTION_SECONDS
    )


def test_valid_retention_value_is_parsed(monkeypatch):
    monkeypatch.setenv("LEGACY_SESSION_RETENTION_SECONDS", "600")
    assert ebs._legacy_session_retention_seconds() == 600
