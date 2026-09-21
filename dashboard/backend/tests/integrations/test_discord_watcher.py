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
    through with neither continue nor break -- 840 polls later the for/else
    delivered "still running after 70 minutes" for an outcome the server had
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
    assert "still running after 70 minutes" not in content


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
    assert "still running after 70 minutes" not in content


def test_the_watcher_outlives_the_server_budget_it_watches():
    """Two constants, two different jobs -- and the reason they must not
    reconverge (the same failure class as issue #474 item 5, but for the
    Discord watcher rather than the browser poller).

    The server's own backtest budget is ``PIPELINE_SUBPROCESS_TIMEOUT_SECONDS``
    plus ``SUBPROCESS_TIMEOUT_OVERHEAD_SECONDS``; no pipeline backtest can time
    out before that. ``_MAX_POLLS * _POLL_INTERVAL_SEC`` is how long the Discord
    watcher keeps polling before giving up and printing its own "still running"
    guess. If the watcher's window is not longer than the server's budget, the
    watcher always exits through the for/else *before* the server ever reaches
    its own ``timed_out`` verdict -- which means the ``timed_out`` branch in
    ``watch_and_deliver_backtest`` is unreachable in production. It is reachable
    only via ``resume_open_backtest_jobs()`` after a bot restart mid-run.

    The other watcher tests in this module inject a ``timed_out`` payload
    directly on the second fake poll, which proves the branch's *logic* is
    correct but cannot catch this: they never let real wall-clock timing decide
    whether the branch is ever reached at all. Only comparing the two
    constants -- imported live from the server module, not copied -- can catch
    that regression, the same way ``test_the_client_outlives_the_server_budget_
    it_draws`` in ``test_ifind_ashare_frontend.py`` pins it for the browser
    poller. If this test fails, the ``timed_out`` branch this module tests
    elsewhere has gone back to being dead code.
    """
    from dashboard.backend.api.routers.backtests import (
        PIPELINE_SUBPROCESS_TIMEOUT_SECONDS,
        SUBPROCESS_TIMEOUT_OVERHEAD_SECONDS,
    )

    watch_seconds = bot._MAX_POLLS * bot._POLL_INTERVAL_SEC

    assert watch_seconds > PIPELINE_SUBPROCESS_TIMEOUT_SECONDS, (
        "the Discord watcher must keep polling past the server's own budget, "
        "or it can never receive the server's terminal verdict"
    )
    assert watch_seconds == (
        PIPELINE_SUBPROCESS_TIMEOUT_SECONDS + SUBPROCESS_TIMEOUT_OVERHEAD_SECONDS
    )


def test_a_stopped_run_is_not_posted_as_a_failure(job_store, monkeypatch):
    """A cancel and a timeout are not failures, and this was the one surface
    saying they were.

    `_finalize_slot`'s docstring, the status route's own branch, app.js's
    four-state panel and `.is-timed-out` in styles.css all make the same
    argument -- a cancel is the owner's deliberate action and a timeout is the
    product running out of the budget it set itself. Routing both through
    `terminal_error` posted "**Backtest failed**" and filed the job as
    STATUS_FAILED, which is exactly the lie the rest of the feature exists to
    stop.
    """
    live_run_id = "agent_20260914_stopped01"
    job = job_store.create_job(
        discord_user_id="42",
        channel_id=99,
        session_id="sess-stopped",
        label="st0pped",
        live_run_id=live_run_id,
    )
    poster = _PostRecorder()
    _install_common(monkeypatch, poster)

    async def fake_api_get(path: str, *, headers=None, timeout: int = 30):
        return {
            "running": False,
            "timed_out": True,
            "live_run_id": live_run_id,
            "timeout": {
                "limit_seconds": 3600,
                "billing_mode": "platform_credits",
                "spent_micro": 42_318,
                "model_calls": 2,
            },
        }

    monkeypatch.setattr(bot, "api_get", fake_api_get)

    asyncio.run(bot.watch_and_deliver_backtest(job.job_id))

    content = poster.calls[0]["content"]
    assert content.startswith("**Backtest stopped**")
    assert "Backtest failed" not in content
    # The job is terminal and absent from _OPEN_STATUSES exactly like
    # STATUS_FAILED, so a bot restart does not resume it -- but it is not
    # counted as a failure either.
    assert job_store.get(job.job_id).status == STATUS_NOTIFIED
    assert not job_store.list_open()


def test_a_real_error_is_still_posted_as_a_failure(job_store, monkeypatch):
    """The stopped/failed split must not swallow genuine failures."""
    job = job_store.create_job(
        discord_user_id="42",
        channel_id=99,
        session_id="sess-err",
        label="brok3n",
        live_run_id="agent_20260914_err01",
    )
    poster = _PostRecorder()
    _install_common(monkeypatch, poster)

    async def fake_api_get(path: str, *, headers=None, timeout: int = 30):
        return {"running": False, "error": "Backtest failed (code 1)"}

    monkeypatch.setattr(bot, "api_get", fake_api_get)

    asyncio.run(bot.watch_and_deliver_backtest(job.job_id))

    assert poster.calls[0]["content"].startswith("**Backtest failed**")


def test_the_watcher_prefers_cancelled_over_timed_out(job_store, monkeypatch):
    """Three surfaces, one branch order.

    `test_status_prefers_cancelled_over_timed_out_when_a_slot_carries_both` and
    the poll dispatch in app.js both resolve a slot carrying both flags as the
    cancel -- the owner's own action outranks the budget that would have
    stopped the run anyway. This watcher tested `timed_out` first, so the one
    future state the router test exists to catch would have been reported
    differently here than in the browser.
    """
    job = job_store.create_job(
        discord_user_id="42",
        channel_id=99,
        session_id="sess-both",
        label="b0th",
        live_run_id="agent_20260914_both01",
    )
    poster = _PostRecorder()
    _install_common(monkeypatch, poster)

    async def fake_api_get(path: str, *, headers=None, timeout: int = 30):
        return {
            "running": False,
            "cancelled": True,
            "timed_out": True,
            "timeout": {"limit_seconds": 3600},
        }

    monkeypatch.setattr(bot, "api_get", fake_api_get)

    asyncio.run(bot.watch_and_deliver_backtest(job.job_id))

    assert "Backtest cancelled." in poster.calls[0]["content"]
    assert "limit" not in poster.calls[0]["content"]


def test_the_discord_timeout_notice_does_not_reimplement_credit_formatting():
    """The amount goes through `format_credits`, the module that already owns
    exact micro-Credit rendering for every other backend surface.

    The `spent_micro / 1_000_000:.6f` this replaced was float division on an
    integer ledger: inexact in general, and simply wrong above 2**53
    micro-Credits, where the float cannot represent the integer at all.
    """
    huge = 2**53 + 1
    notice = bot._format_timeout_notice(
        {"limit_seconds": 3600, "spent_micro": huge, "model_calls": 3}
    )

    assert "9007199254.740993 Credits" in notice
    assert "3 model calls completed" in notice


def test_the_discord_timeout_notice_rounds_minutes_the_way_javascript_does():
    """Python's `round` is banker's rounding and JavaScript's `Math.round` is
    half-up, so a 150-second budget printed "2-minute limit" in Discord and
    "3-minute limit" on the card for the same run."""
    notice = bot._format_timeout_notice({"limit_seconds": 150})
    assert notice.startswith("Stopped at the 3-minute limit.")


def test_the_discord_timeout_notice_omits_the_cost_when_nothing_settled():
    """`spent_micro: 0, model_calls: 0` is a reachable platform-credits payload
    -- a run that timed out before its first call settled. Rendering it claims
    calls that did not happen."""
    notice = bot._format_timeout_notice(
        {"limit_seconds": 3600, "billing_mode": "platform_credits",
         "spent_micro": 0, "model_calls": 0}
    )
    assert "Credits" not in notice


def test_the_discord_timeout_notice_never_formats_a_boolean_as_money():
    """`bool` is an `int` subclass, and these fields arrive from JSON."""
    notice = bot._format_timeout_notice(
        {"limit_seconds": True, "spent_micro": True, "model_calls": True}
    )
    assert notice.startswith("Stopped at the time limit.")
    assert "Credits" not in notice
