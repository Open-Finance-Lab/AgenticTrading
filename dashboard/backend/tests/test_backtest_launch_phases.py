"""The parent tells the child when it was launched.

The child's first progress write can only account for the gap before it
(imports, store startup) if it knows when the parent spawned it. That clock
rides argv beside --run-id, so a run's `starting` phase is measured rather
than missing.
"""
import math
import os
import subprocess
import sys
import time
import uuid

import pytest

import dashboard.backend.api.routers.backtests as backtests
from dashboard.backend.tests._fake_child import FakeChild

REAL_RUN_BACKTEST_BACKGROUND = backtests.run_backtest_background


@pytest.fixture(autouse=True)
def _clean_slots():
    # Same idiom as test_backtest_cancel.py's _clean_slots: this module now
    # pre-registers a real slot (see _acquire below), so it has to be torn
    # down between cases the same way, or a leftover slot trips the
    # process-wide concurrency cap on a later test.
    backtests._reset_slots_for_tests()
    yield
    backtests._reset_slots_for_tests()


def _acquire(run_id: str, session_id: str) -> None:
    assert (
        backtests._try_acquire_backtest_slot(
            live_run_id=run_id, session_id=session_id, user_id=None
        )
        is None
    )


def _launch(monkeypatch, live_run_id: str):
    captured = {}

    # A plain "call time.time() twice and compare" spy is unreliable here:
    # two real calls microseconds apart round to the SAME millisecond often
    # enough (~20% of trials, measured against a deliberately-reintroduced
    # two-call regression) that a bare equality check on the raw values would
    # only catch the regression it exists for some of the time. Instead, the
    # FIRST time.time() call inside run_backtest_background (verified by
    # instrumentation to be exactly the `launched_at = time.time()` line --
    # slot acquisition above already consumed the real clock once, before
    # this patch is even installed) is diverted to a fixed, distinguishable
    # stand-in: the same integer second as the real clock (so the window
    # assertion below still holds), but a fractional part a second, separate
    # time.time() call would reproduce only by a one-in-a-million coincidence.
    # Every call after the first passes through to the real clock unchanged,
    # so nothing downstream of `launched_at` (finalize, elapsed-time math) is
    # affected.
    real_time = time.time
    call_count = {"n": 0}
    launched_at_sentinel = None

    def fake_time():
        nonlocal launched_at_sentinel
        now = real_time()
        call_count["n"] += 1
        if call_count["n"] == 1:
            launched_at_sentinel = math.floor(now) + 0.123
            return launched_at_sentinel
        return now

    monkeypatch.setattr(backtests.time, "time", fake_time)

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs.get("env")
        # Snapshot the slot HERE, not after run_backtest_background returns:
        # with a FakeChild the "subprocess" finishes instantly, so by the time
        # the function returns, finalize has already popped the slot out of
        # _active_slots (see test_backtest_cancel.py's "started_at is cleared
        # at finalize"). _update_slot(started_at=launched_at, ...) runs
        # earlier in run_backtest_background, well before Popen is reached, so
        # the slot already carries the real value by this point.
        captured["slot_started_at"] = backtests._active_slots[live_run_id]["started_at"]
        return FakeChild()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(backtests.db, "get_runs_by_mode", lambda mode: [])
    before = real_time()
    REAL_RUN_BACKTEST_BACKGROUND(
        "2026-04-01",
        "2026-04-08",
        "session-id",
        decision_source="rule_based",
        live_run_id=live_run_id,
    )
    captured["before"] = before
    captured["after"] = real_time()
    captured["launched_at_sentinel"] = launched_at_sentinel
    return captured


def test_child_argv_carries_the_launch_time(monkeypatch):
    # A pre-registered slot, not an auto-minted id: _update_slot only merges
    # into an *existing* _active_slots entry (see its docstring) -- with no
    # slot registered first, it silently falls back to the legacy status
    # mirror instead, and the identity check below would be inspecting the
    # wrong dict. Acquiring the slot first, the same way test_backtest_cancel.
    # py's _acquire does, is what makes _active_slots the right place to look.
    run_id = f"agent_launch_phase_test_{uuid.uuid4().hex[:8]}"
    _acquire(run_id, "session-id")

    captured = _launch(monkeypatch, run_id)
    command = captured["command"]
    launched_at = float(command[command.index("--launched-at") + 1])
    assert captured["before"] - 1 <= launched_at <= captured["after"] + 1

    # The argv value must equal our injected sentinel exactly (formatted to
    # the same 3 decimals backtests.py itself uses, f"{launched_at:.3f}") --
    # if it doesn't, `launched_at` was NOT read from the one time.time() call
    # our fake_time diverts, meaning some other call produced it.
    expected = float(f"{captured['launched_at_sentinel']:.3f}")
    assert launched_at == expected, (
        f"child --launched-at ({launched_at}) does not match the sentinel "
        f"reading ({expected}) -- launched_at was not read from the single "
        f"diverted time.time() call"
    )

    # Not just "a recent-looking number" -- the *same* reading the run slot's
    # started_at carries. A regression to two separate time.time() calls
    # would still pass the window assertion above (microseconds apart both
    # land inside +/-1s), which is exactly the mistake backtests.py:1568-1577
    # warns against; only an equality check on the shared reading catches it.
    # Because fake_time only diverts the FIRST call and passes every later
    # call through to the real clock, a regressed second time.time() call for
    # the argv line would return the real "now" -- astronomically unlikely to
    # equal our fixed .123 fractional sentinel -- so this is deterministic,
    # not a coin flip on which millisecond two real calls land in.
    slot_started_at = captured["slot_started_at"]
    slot_started_at_rounded = float(f"{slot_started_at:.3f}")
    assert slot_started_at_rounded == launched_at, (
        f"slot started_at ({slot_started_at}, rounded {slot_started_at_rounded}) "
        f"and child --launched-at ({launched_at}) must derive from the same "
        f"time.time() reading spent twice, not two separate calls"
    )


def test_script_accepts_launched_at(tmp_path):
    result = subprocess.run(
        [sys.executable, "dashboard/scripts/backtest_hourly_agent.py", "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "DATABASE_PATH": str(tmp_path / "backtest.db")},
    )
    assert result.returncode == 0, result.stderr
    assert "--launched-at" in result.stdout
