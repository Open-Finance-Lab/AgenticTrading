"""The daily-facts worker owns its schedule (design D23, SS6.9)."""

from __future__ import annotations

import inspect
import threading

import pytest

import dashboard.backend.app as app_module
from dashboard.backend.domain.analytics import daily_facts as daily_facts_module
from dashboard.backend.domain.analytics import daily_job


@pytest.fixture(autouse=True)
def _stop_worker():
    yield
    daily_job.stop_daily_facts_worker()


def test_interval_defaults_to_five_minutes(monkeypatch):
    monkeypatch.delenv("ANALYTICS_DAILY_JOB_INTERVAL_SECONDS", raising=False)
    assert daily_job.daily_job_interval_seconds() == 300.0


@pytest.mark.parametrize("raw", ["junk", "", "  ", "4", "3601", "-1"])
def test_bad_intervals_fall_back_with_a_line_instead_of_raising(monkeypatch, capsys, raw):
    monkeypatch.setenv("ANALYTICS_DAILY_JOB_INTERVAL_SECONDS", raw)
    assert daily_job.daily_job_interval_seconds() == 300.0
    if raw.strip():
        assert "ANALYTICS_DAILY_JOB_INTERVAL_SECONDS" in capsys.readouterr().out


def test_a_valid_interval_is_honoured(monkeypatch):
    monkeypatch.setenv("ANALYTICS_DAILY_JOB_INTERVAL_SECONDS", "60")
    assert daily_job.daily_job_interval_seconds() == 60.0


def test_the_worker_prepares_once_then_ticks_until_stopped():
    stop = threading.Event()
    calls: list[str] = []

    def prepare():
        calls.append("prepare")

    def tick():
        calls.append("tick")
        if calls.count("tick") >= 3:
            stop.set()

    thread = daily_job.start_daily_facts_worker(
        0.01, stop, tick=tick, prepare=prepare
    )
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert calls[0] == "prepare"
    assert calls.count("prepare") == 1
    assert calls.count("tick") >= 3
    assert thread.daemon is True
    assert thread.name == "analytics-daily-facts"


def test_a_failing_tick_is_logged_and_the_loop_continues(capsys):
    stop = threading.Event()
    ticks: list[int] = []

    def tick():
        ticks.append(len(ticks))
        if len(ticks) == 1:
            raise RuntimeError("private tick detail")
        if len(ticks) >= 2:
            stop.set()

    daily_job.start_daily_facts_worker(0.01, stop, tick=tick, prepare=lambda: None).join(5)
    printed = capsys.readouterr().out

    assert len(ticks) >= 2
    assert "WARNING: analytics.daily_facts.tick_failed category=RuntimeError" in printed
    assert "private tick detail" not in printed


def test_a_failing_prepare_is_logged_and_the_worker_still_ticks(capsys):
    stop = threading.Event()

    def prepare():
        raise RuntimeError("private migration detail")

    daily_job.start_daily_facts_worker(0.01, stop, tick=stop.set, prepare=prepare).join(5)
    printed = capsys.readouterr().out

    assert "WARNING: analytics.facts_migration_failed category=RuntimeError" in printed
    assert "private migration detail" not in printed


def test_starting_twice_returns_the_live_thread():
    stop = threading.Event()
    first = daily_job.start_daily_facts_worker(60, stop, tick=lambda: None, prepare=lambda: None)
    second = daily_job.start_daily_facts_worker(60, stop, tick=lambda: None, prepare=lambda: None)

    assert first is second
    daily_job.stop_daily_facts_worker()
    assert not first.is_alive()


def test_startup_starts_the_worker_and_leaves_the_reaper_to_heartbeats():
    source = inspect.getsource(app_module.startup_event)

    assert source.count("start_daily_facts_worker()") == 1
    assert "register_reaper_sweep(analytics_retention_coordinator.run_if_due)" not in source
    assert "register_reaper_sweep(run_daily_facts" not in source
    # The throttled snapshot repairs stay on the reaper until PR B deletes them.
    assert "register_reaper_sweep(run_analytics_maintenance)" in source


def test_the_job_owns_the_retention_coordinator_now():
    source = inspect.getsource(daily_facts_module)
    assert source.count("analytics_retention_coordinator") == 1
