"""Behavioral tests for the Discord backtest watcher.

These exercise ``watch_and_deliver_backtest`` end-to-end against a fake API
layer (no Discord connection, no HTTP). The API seams (``api_get``,
``api_get_bytes``, ``_post_channel_result``) are monkeypatched; the real
SQLite job store is used via a temp DB so status transitions are asserted.

Requires the optional ``discord`` dependency; skipped when absent so the
suite stays green on minimal interpreters (CI installs core requirements only).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

import pytest

discord = pytest.importorskip("discord")

import dashboard.backend.integrations.discord_bot as bot
from dashboard.backend.integrations.discord_jobs import (
    STATUS_NOTIFIED,
    get_job_store,
    reset_job_store_for_tests,
)


@pytest.fixture
def job_store(tmp_path: Path, monkeypatch):
    """Point the singleton job store at a temp DB for this test."""
    monkeypatch.setenv("DISCORD_JOBS_DB", str(tmp_path / "jobs.db"))
    reset_job_store_for_tests()
    store = get_job_store()
    yield store
    reset_job_store_for_tests()


class _PostRecorder:
    """Stand-in for ``_post_channel_result`` that records what was posted."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        *,
        channel_id: int,
        discord_user_id: str,
        content: str,
        chart: Optional[Any] = None,
    ) -> None:
        self.calls.append(
            {
                "channel_id": channel_id,
                "discord_user_id": discord_user_id,
                "content": content,
                "chart": chart,
            }
        )


def _metrics(run_id: str, *, total_return: float) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "llm_model": "claude-haiku-4-5",
        "total_return": total_return,
        "sharpe_ratio": 1.2,
        "max_drawdown": -0.05,
        "num_trades": 4,
        "final_equity": 110000.0,
    }


def _install_common(monkeypatch, poster: _PostRecorder) -> None:
    monkeypatch.setattr(bot, "_POLL_INTERVAL_SEC", 0)  # no real sleeps
    monkeypatch.setattr(bot, "_post_channel_result", poster)

    async def _fake_bytes(path: str, *, headers=None, timeout: int = 60) -> bytes:
        return b"\x89PNG-fake"

    monkeypatch.setattr(bot, "api_get_bytes", _fake_bytes)


def test_watcher_happy_path_posts_summary_and_marks_notified(job_store, monkeypatch):
    live_run_id = "agent_20260722_new00001"
    job = job_store.create_job(
        discord_user_id="42",
        channel_id=99,
        session_id="sess-1",
        label="e23badad",
        live_run_id=live_run_id,
    )
    poster = _PostRecorder()
    _install_common(monkeypatch, poster)

    status_calls = {"n": 0}

    async def fake_api_get(path: str, *, headers=None, timeout: int = 30):
        if path == "/backtest/status":
            status_calls["n"] += 1
            # First poll: still running; second poll: completed. Exercises the loop.
            if status_calls["n"] == 1:
                return {"running": True}
            return {"running": False, "success": True, "runs_count": 1}
        if path == f"/runs/{live_run_id}":
            return _metrics(live_run_id, total_return=0.10)
        raise AssertionError(f"unexpected api_get path: {path}")

    monkeypatch.setattr(bot, "api_get", fake_api_get)

    asyncio.run(bot.watch_and_deliver_backtest(job.job_id))

    assert len(poster.calls) == 1
    posted = poster.calls[0]
    assert "Backtest complete" in posted["content"]
    assert "10.00%" in posted["content"]  # this run's return
    assert posted["chart"] is not None  # plot.png attached

    done = job_store.get(job.job_id)
    assert done.status == STATUS_NOTIFIED
    assert done.run_id == live_run_id


def test_watcher_does_not_post_a_different_runs_metrics(job_store, monkeypatch):
    """Regression: the ``latest/metrics`` fallback must be identity-gated.

    Discord sessions are stable per user, so ``/runs/latest/metrics`` can return
    a PRIOR run. If the exact-run fetch fails, the watcher must NOT post that
    stale run's numbers under this fresh backtest — it soft-fails instead.
    """
    live_run_id = "agent_20260722_fresh0002"
    stale_run_id = "agent_20260101_old00001"
    job = job_store.create_job(
        discord_user_id="42",
        channel_id=99,
        session_id="sess-1",
        label="fresh-label",
        live_run_id=live_run_id,
    )
    poster = _PostRecorder()
    _install_common(monkeypatch, poster)

    async def fake_api_get(path: str, *, headers=None, timeout: int = 30):
        if path == "/backtest/status":
            return {"running": False, "success": True, "runs_count": 1}
        if path == f"/runs/{live_run_id}":
            raise RuntimeError("run not queryable yet")  # transient exact-run failure
        if path == "/runs/latest/metrics":
            # A DIFFERENT (older) run — must never be posted as this job's result.
            return _metrics(stale_run_id, total_return=0.99)
        raise AssertionError(f"unexpected api_get path: {path}")

    monkeypatch.setattr(bot, "api_get", fake_api_get)

    asyncio.run(bot.watch_and_deliver_backtest(job.job_id))

    assert len(poster.calls) == 1
    posted = poster.calls[0]
    assert "could not be read" in posted["content"]
    assert "99.00%" not in posted["content"]  # stale return must NOT leak through

    done = job_store.get(job.job_id)
    # Delivered (a soft-failure message), so the job is closed, not left open.
    assert done.status == STATUS_NOTIFIED


def test_watcher_accepts_latest_when_it_is_this_run(job_store, monkeypatch):
    """The fallback is allowed when ``latest`` IS this run (ids match)."""
    live_run_id = "agent_20260722_match0003"
    job = job_store.create_job(
        discord_user_id="42",
        channel_id=99,
        session_id="sess-1",
        label="match-label",
        live_run_id=live_run_id,
    )
    poster = _PostRecorder()
    _install_common(monkeypatch, poster)

    async def fake_api_get(path: str, *, headers=None, timeout: int = 30):
        if path == "/backtest/status":
            return {"running": False, "success": True, "runs_count": 1}
        if path == f"/runs/{live_run_id}":
            raise RuntimeError("exact-run endpoint hiccup")
        if path == "/runs/latest/metrics":
            return _metrics(live_run_id, total_return=0.07)  # same id → acceptable
        raise AssertionError(f"unexpected api_get path: {path}")

    monkeypatch.setattr(bot, "api_get", fake_api_get)

    asyncio.run(bot.watch_and_deliver_backtest(job.job_id))

    assert len(poster.calls) == 1
    posted = poster.calls[0]
    assert "Backtest complete" in posted["content"]
    assert "7.00%" in posted["content"]

    done = job_store.get(job.job_id)
    assert done.status == STATUS_NOTIFIED
    assert done.run_id == live_run_id


def test_watcher_reports_api_error_status(job_store, monkeypatch):
    """A terminal error from /backtest/status is delivered as a failure post."""
    job = job_store.create_job(
        discord_user_id="42",
        channel_id=99,
        session_id="sess-1",
        label="err-label",
        live_run_id="agent_20260722_err00004",
    )
    poster = _PostRecorder()
    _install_common(monkeypatch, poster)

    async def fake_api_get(path: str, *, headers=None, timeout: int = 30):
        if path == "/backtest/status":
            return {"running": False, "error": "boom in worker"}
        raise AssertionError(f"unexpected api_get path: {path}")

    monkeypatch.setattr(bot, "api_get", fake_api_get)

    asyncio.run(bot.watch_and_deliver_backtest(job.job_id))

    assert len(poster.calls) == 1
    assert "Backtest failed" in poster.calls[0]["content"]
    assert "boom in worker" in poster.calls[0]["content"]

    done = job_store.get(job.job_id)
    assert done.status == STATUS_NOTIFIED


def test_watcher_reports_a_timed_out_run_instead_of_waiting_out_the_budget(
    job_store, monkeypatch
):
    """The loop knew only running/error/success, so a `timed_out` payload fell
    through with neither continue nor break -- 360 polls later the for/else
    delivered "still running after 30 minutes" for an outcome the server had
    already reported on the second poll."""
    live_run_id = "agent_20260914_timeout01"
    job = job_store.create_job(
        discord_user_id="42",
        channel_id=99,
        session_id="sess-timeout",
        label="t1me0ut",
        live_run_id=live_run_id,
    )
    poster = _PostRecorder()
    _install_common(monkeypatch, poster)

    polls = {"n": 0}

    async def fake_api_get(path: str, *, headers=None, timeout: int = 30):
        assert path == "/backtest/status"
        polls["n"] += 1
        if polls["n"] == 1:
            return {"running": True}
        return {
            "running": False,
            "timed_out": True,
            "elapsed_seconds": 3600,
            "live_run_id": live_run_id,
            "message": "Backtest stopped at the time limit.",
            "timeout": {
                "limit_seconds": 3600,
                "billing_mode": "platform_credits",
                "spent_micro": 42_318,
                "model_calls": 2,
            },
        }

    monkeypatch.setattr(bot, "api_get", fake_api_get)

    asyncio.run(bot.watch_and_deliver_backtest(job.job_id))

    # Broke out on the second poll, not after 360 of them.
    assert polls["n"] == 2
    assert len(poster.calls) == 1
    content = poster.calls[0]["content"]
    assert "60-minute limit" in content
    assert "still running after 30 minutes" not in content


def test_watcher_reports_a_cancelled_run_instead_of_waiting_out_the_budget(
    job_store, monkeypatch
):
    """The same gap, which this surface has had since the cancel route shipped:
    a user who cancels a Discord-launched backtest gets the identical
    thirty-minute non-answer."""
    live_run_id = "agent_20260914_cancel01"
    job = job_store.create_job(
        discord_user_id="42",
        channel_id=99,
        session_id="sess-cancel",
        label="cance11ed",
        live_run_id=live_run_id,
    )
    poster = _PostRecorder()
    _install_common(monkeypatch, poster)

    polls = {"n": 0}

    async def fake_api_get(path: str, *, headers=None, timeout: int = 30):
        assert path == "/backtest/status"
        polls["n"] += 1
        return {
            "running": False,
            "cancelled": True,
            "elapsed_seconds": 120,
            "live_run_id": live_run_id,
            "message": "Backtest cancelled.",
        }

    monkeypatch.setattr(bot, "api_get", fake_api_get)

    asyncio.run(bot.watch_and_deliver_backtest(job.job_id))

    assert polls["n"] == 1
    assert len(poster.calls) == 1
    content = poster.calls[0]["content"]
    assert "cancelled" in content.lower()
    assert "still running after 30 minutes" not in content
