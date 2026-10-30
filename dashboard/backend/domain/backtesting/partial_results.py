"""Boot reclaimer for interrupted backtests (design: partial results).

A dashboard backtest is a subprocess whose results only reach the database at
completion. If its parent dies mid-run (a deploy, a crash), the run leaves no
row and no curve -- the user's wait simply vanishes. While it runs, the engine
mirrors its progress payload into ``run_live_progress`` every ten bars
(``_flush_live_progress_durable``) and ``LiveProgressHeartbeat`` re-stamps the
row every minute. This module turns a snapshot nothing stamps any more into an
honest *interrupted* run: an ``agent_runs`` row carrying ``interrupted``
metadata and the step it reached, plus the partial equity curve and trades in
the regular tables, so every existing reader serves the partial result with
zero changes.

Who deletes a snapshot decides what a surviving one means. The parent
(``run_backtest_background``'s ``finally``) deletes it on every outcome it
sees -- success, failure, timeout, cancel -- so a row survives only when the
parent never reached that ``finally``: its process died. That is what makes
the "server restart" label the frontend shows true.

The age gate exists because the run database is shared across instances. The
live service has no disk, so Render deploys with zero downtime: this instance
boots while the old one's children are still running. Reclaiming those would
publish a live run as interrupted, then duplicate its trades when it finished.
So a row is reclaimed only once nothing has stamped it for
``RECLAIM_STALE_AFTER_SECONDS``, and only by the reclaimer whose
``claim_live_progress`` re-stamped it -- both instances sweep during a deploy,
and trades are appended, so two publishers would write them twice. The boot
sweep cannot finish the job alone: the old instance's runs die *after* this one
boots, and some write their first snapshot after it. So
``watch_interrupted_backtests`` re-sweeps the whole table for
``RECLAIM_WATCH_SECONDS``, long enough to outlast the old instance. After that
every row belongs to this process's own runs, whose parent cleans up; if this
process dies, the next boot's sweep finds them.

Two kinds of row are discarded, not published, because nothing says how their
run ended. One unreclaimed for ``RECLAIM_MAX_AGE_SECONDS``: while any instance
is up, a dead run's row is reclaimed within minutes. And one with no
``heartbeat_seconds`` in its metadata: its writer predates the heartbeat, so a
live run's row went unstamped between ten-bar flushes, and its parent never
deleted the row of a run that was cancelled or failed.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Optional

from dashboard.backend.domain.backtesting.constants import fractional_return

#: How often a running child re-stamps its snapshot.
LIVE_PROGRESS_HEARTBEAT_SECONDS = 60
#: Unstamped this long, a snapshot's run is dead. Five heartbeats, so one slow
#: or failed write cannot get a live run reclaimed.
RECLAIM_STALE_AFTER_SECONDS = 5 * LIVE_PROGRESS_HEARTBEAT_SECONDS
#: Unreclaimed this long, a snapshot is discarded unpublished (see above).
RECLAIM_MAX_AGE_SECONDS = 3600
#: How often the boot watcher re-sweeps the table.
RECLAIM_RECHECK_SECONDS = 60
#: How long after boot the watcher sweeps. Render stops the old instance 60s
#: after this one takes traffic and gives it at most 300s (its maximum
#: ``maxShutdownDelaySeconds``) to exit, so its last runs are stale about eleven
#: minutes after boot. Thirty leaves room for a slow health check.
RECLAIM_WATCH_SECONDS = 30 * 60

# The completed row's identity (run_agent_backtest's insert_run), so a reclaimed
# run is labelled, listed and filtered exactly like a finished one.
_AGENT_NAME = "Agent"
_MODE = "backtest"

_COUNTS = ("reclaimed", "dropped_stale", "discarded", "failed")


def reclaim_interrupted_backtests() -> Dict[str, Any]:
    """One sweep over ``run_live_progress``.

    Returns the per-outcome counts, ``pending`` -- the run ids still too fresh
    to judge -- and ``listing_failed``, which is not the same answer as an
    empty table and must not be read as one.
    """
    from dashboard.backend.database import db

    result: Dict[str, Any] = {name: 0 for name in _COUNTS}
    result.update(pending=[], listing_failed=False)
    try:
        rows = db.list_live_progress()
    except Exception as exc:  # noqa: BLE001 - reclaim must never break boot
        print(f"⚠️  interrupted-run reclaim: snapshot listing failed: {exc}")
        result["listing_failed"] = True
        return result

    for row in rows:
        run_id = str(row.get("run_id") or "").strip()
        if not run_id:
            continue
        try:
            outcome = _judge(db, run_id, row)
        except Exception as exc:  # noqa: BLE001 - one bad row starves the rest
            result["failed"] += 1
            print(f"⚠️  interrupted-run reclaim: {run_id} failed: {exc}")
            continue
        if outcome == "pending":
            result["pending"].append(run_id)
        else:
            result[outcome] += 1
    return result


def _judge(db: Any, run_id: str, row: Dict[str, Any]) -> str:
    if db.get_run(run_id) is not None:
        # The run finished and its result is already in agent_runs: the parent
        # died after the child completed, or the child is still writing its
        # baselines. Either way the snapshot is redundant.
        db.delete_live_progress(run_id)
        return "dropped_stale"
    age = row.get("age_seconds")
    age = float(age) if age is not None else float("inf")
    if age < RECLAIM_STALE_AFTER_SECONDS:
        return "pending"
    if age >= RECLAIM_MAX_AGE_SECONDS:
        db.delete_live_progress(run_id)
        print(
            f"🧹 interrupted-run reclaim: discarded {run_id}, unstamped for"
            f" {age:.0f}s -- older than this design can produce, so how it"
            " ended is unknown"
        )
        return "discarded"
    if not db.claim_live_progress(run_id, RECLAIM_STALE_AFTER_SECONDS):
        # Stamped since the listing: the run is alive after all, or another
        # instance's reclaimer claimed it first.
        return "pending"
    payload = db.get_live_progress(run_id)
    if not isinstance(payload, dict):
        db.delete_live_progress(run_id)
        print(f"⚠️  interrupted-run reclaim: discarded {run_id}, payload unreadable")
        return "discarded"
    if not (payload.get("run_metadata") or {}).get("heartbeat_seconds"):
        db.delete_live_progress(run_id)
        print(
            f"🧹 interrupted-run reclaim: discarded {run_id}, written before runs"
            " heartbeated -- its age does not show the run ended"
        )
        return "discarded"
    _publish_interrupted_run(db, run_id, row, payload)
    db.delete_live_progress(run_id)
    return "reclaimed"


def _optional_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _publish_interrupted_run(
    db: Any, run_id: str, row: Dict[str, Any], payload: Dict[str, Any]
) -> None:
    meta = payload.get("run_metadata") or {}
    curve = payload.get("equity_curve") or []
    trades = payload.get("trades") or []
    first_equity = float(curve[0]["equity"]) if curve else None
    last_equity = float(curve[-1]["equity"]) if curve else None
    # The completed row divides by the run's configured capital, not by the
    # first bar's equity; a snapshot from before initial_capital was recorded
    # falls back to the first bar.
    initial_capital = _optional_float(meta.get("initial_capital"))
    if initial_capital is None:
        initial_capital = first_equity or 0.0
    owner = meta.get("owner_user_id")

    db.insert_run(
        run_id,
        str(row.get("session_id") or ""),
        _AGENT_NAME,
        _MODE,
        str(meta.get("start_date") or ""),
        str(meta.get("end_date") or ""),
        first_equity if first_equity is not None else initial_capital,
        final_equity=last_equity,
        total_return=(
            fractional_return(last_equity, initial_capital)
            if last_equity is not None
            else None
        ),
        num_trades=len(trades),
        llm_model=str(meta.get("llm_model") or "rule-based"),
        metadata={
            "interrupted": True,
            "interrupted_reason": "server restart",
            "interrupted_step": int(payload.get("step") or 0),
            "interrupted_total_steps": int(payload.get("total_steps") or 0),
            "partial": True,
        },
        owner_user_id=int(owner) if owner is not None else None,
    )
    # Curve points already carry timestamp/equity/cash/positions_value in the
    # shape insert_equity_points expects; trades are _serialize_trades output.
    db.insert_equity_points(run_id, curve, replace=True)
    if trades:
        db.insert_trades(run_id, trades)


def watch_interrupted_backtests(
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    recheck_seconds: float = RECLAIM_RECHECK_SECONDS,
    limit_seconds: float = RECLAIM_WATCH_SECONDS,
) -> Dict[str, int]:
    """Sweep the whole table now and every ``recheck_seconds`` until
    ``limit_seconds`` have passed, reporting each sweep that changed anything.
    Returns the totals across sweeps.

    Every sweep re-lists, so a row that failed or was still fresh, and a row
    first written after boot, are all judged again on the next one.
    """
    totals = {name: 0 for name in _COUNTS}
    deadline = clock() + limit_seconds
    while True:
        result = reclaim_interrupted_backtests()
        for name in _COUNTS:
            totals[name] += result[name]
        if any(result[name] for name in _COUNTS):
            print(
                f"🧹 Reclaimed {result['reclaimed']} interrupted run(s) with"
                f" partial results (dropped {result['dropped_stale']} finished,"
                f" discarded {result['discarded']}, {result['failed']} failed)"
            )
        if clock() >= deadline:
            return totals
        sleep(recheck_seconds)


def start_interrupted_run_reclaimer() -> threading.Thread:
    """Run the boot sweep and its watch on a daemon thread. Boot never waits:
    the sweep reaches the run database once per row, and the watch lasts until
    the previous instance's runs have ended."""
    thread = threading.Thread(
        target=_run_reclaimer, name="interrupted-run-reclaimer", daemon=True
    )
    thread.start()
    return thread


def _run_reclaimer() -> None:
    try:
        watch_interrupted_backtests()
    except Exception as exc:  # noqa: BLE001 - reclaim must never take the process down
        print(f"⚠️  interrupted-run reclaim error: {exc}")


class LiveProgressHeartbeat:
    """Re-stamps one run's snapshot for as long as the run is alive.

    Started by the dashboard child (``backtest_hourly_agent.py``) around the
    agent loop. The child is the only process whose life is the run's: the bar
    loop's own flush comes every ten bars, which on an LLM run can be many
    minutes apart, and a parent on another instance cannot be asked.
    """

    def __init__(
        self, run_id: str, *, interval_seconds: float = LIVE_PROGRESS_HEARTBEAT_SECONDS
    ) -> None:
        self.run_id = run_id
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._beat, name="live-progress-heartbeat", daemon=True
        )

    def start(self) -> "LiveProgressHeartbeat":
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _beat(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                from dashboard.backend.database import db

                db.touch_live_progress(self.run_id)
            except Exception as exc:  # noqa: BLE001 - the stale window spans five missed stamps
                print(f"   ⚠️  live-progress heartbeat skipped: {exc}", flush=True)
