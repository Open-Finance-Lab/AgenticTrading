"""Baseline worker: wholesale-failure visibility (T4).

A per-item warning ("Baseline generation failed") cannot report a total
contract break -- it survived an entire measurement ladder unnoticed because
nobody reads a stream of per-job warnings. This asserts the escalation line
that fires once a small consecutive-failure threshold is crossed, that any
success resets it, and that the plain total-failure counter a caller like
stress_serve.py reads for a shutdown summary keeps accumulating regardless.
"""

import pytest

from dashboard.backend.domain.backtesting import baseline_worker as bw


@pytest.fixture(autouse=True)
def _isolate():
    bw._reset_for_tests()
    yield
    bw._reset_for_tests()


def _job(run_id):
    return bw.BaselineJob(
        run_id=run_id, session_id=f"sess_{run_id}", start_date="2026-04-15",
        end_date="2026-04-16", mode="safe_trading", all_data={"AAPL": object()},
        publish=lambda ids: None,
    )


def test_escalation_fires_once_at_threshold_and_last_exception_is_named(capsys, monkeypatch):
    def _boom(job):
        raise RuntimeError(f"boom for {job.run_id}")

    monkeypatch.setattr(bw, "_run_job", _boom)

    for i in range(bw._ESCALATION_THRESHOLD):
        assert bw.submit(_job(f"bad{i}"))
    assert bw.wait_idle(10)

    out = capsys.readouterr().out
    # The existing per-job line is untouched -- it still fires every time.
    assert out.count("Baseline generation failed") == bw._ESCALATION_THRESHOLD
    # The escalation line fires exactly once, at the threshold crossing.
    assert out.count("consecutive failures") == 1
    assert f"{bw._ESCALATION_THRESHOLD} consecutive failures" in out
    assert "boom for bad2" in out  # names the LAST exception, not the first
    assert bw._consecutive_failures == bw._ESCALATION_THRESHOLD
    assert bw._total_failures == bw._ESCALATION_THRESHOLD


def test_success_resets_consecutive_counter_but_not_total(capsys, monkeypatch):
    def _boom(job):
        raise RuntimeError(f"boom for {job.run_id}")

    monkeypatch.setattr(bw, "_run_job", _boom)
    for i in range(bw._ESCALATION_THRESHOLD):
        assert bw.submit(_job(f"bad{i}"))
    assert bw.wait_idle(10)
    assert bw._consecutive_failures == bw._ESCALATION_THRESHOLD
    capsys.readouterr()  # drain

    def _ok(job):
        return None

    monkeypatch.setattr(bw, "_run_job", _ok)
    assert bw.submit(_job("good"))
    assert bw.wait_idle(10)
    assert bw._consecutive_failures == 0            # reset by the success
    assert bw._total_failures == bw._ESCALATION_THRESHOLD  # NOT reset by the success

    # A fresh run of ESCALATION_THRESHOLD consecutive failures escalates
    # again -- the reset genuinely re-arms the check rather than latching.
    monkeypatch.setattr(bw, "_run_job", _boom)
    for i in range(bw._ESCALATION_THRESHOLD):
        assert bw.submit(_job(f"bad2-{i}"))
    assert bw.wait_idle(10)
    out2 = capsys.readouterr().out
    assert out2.count("consecutive failures") == 1
    assert bw._total_failures == 2 * bw._ESCALATION_THRESHOLD


def test_failures_past_threshold_do_not_reprint_escalation(capsys, monkeypatch):
    """Escalation must fire on the threshold CROSSING only, not stay armed.

    A `>=` comparison at the crossing check would reprint the escalation line
    on every failure once _consecutive_failures reaches the threshold -- the
    exact log-flood the brief warns a per-item warning must not become. Drive
    two more failures past the threshold and count occurrences in stdout
    (not just presence) to catch that regression.
    """
    def _boom(job):
        raise RuntimeError(f"boom for {job.run_id}")

    monkeypatch.setattr(bw, "_run_job", _boom)

    overrun = bw._ESCALATION_THRESHOLD + 2
    for i in range(overrun):
        assert bw.submit(_job(f"bad{i}"))
    assert bw.wait_idle(10)

    out = capsys.readouterr().out
    assert out.count("Baseline generation failed") == overrun
    assert out.count("consecutive failures") == 1   # still exactly once, past the threshold
    assert bw._consecutive_failures == overrun
    assert bw._total_failures == overrun


def test_fewer_than_threshold_failures_do_not_escalate(capsys, monkeypatch):
    def _boom(job):
        raise RuntimeError(f"boom for {job.run_id}")

    monkeypatch.setattr(bw, "_run_job", _boom)
    for i in range(bw._ESCALATION_THRESHOLD - 1):
        assert bw.submit(_job(f"bad{i}"))
    assert bw.wait_idle(10)

    out = capsys.readouterr().out
    assert out.count("Baseline generation failed") == bw._ESCALATION_THRESHOLD - 1
    assert "consecutive failures" not in out
    assert bw._consecutive_failures == bw._ESCALATION_THRESHOLD - 1
