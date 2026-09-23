"""The daily-facts worker: its own thread, its own interval, its own stop event.

Design D23 / SS6.9: the run reaper exists to keep ``heartbeat_at`` fresh against
``RUN_HEARTBEAT_STALE_SECONDS`` (300 s). A whole-population batch across three
databases on that thread would let a slow analytics night mark live runs as
orphaned, so the job is *not* a ``register_reaper_sweep`` step. It is still
inside the single web process (design SS2.3): no external scheduler, no second
service.

Every tick is ``daily_facts.run_daily_facts``; the first thing the thread does,
once, is ``facts_migration.run_startup_migrations`` (seed ``user_activity``,
copy the legacy history). Both are idempotent, so a restart costs nothing.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Callable


_DEFAULT_INTERVAL_SECONDS = 300
_MIN_INTERVAL_SECONDS = 5
_MAX_INTERVAL_SECONDS = 3600
_ENV_NAME = "ANALYTICS_DAILY_JOB_INTERVAL_SECONDS"


def daily_job_interval_seconds() -> float:
    """Seconds between worker ticks. Junk falls back to the default with a line.

    Same shape as ``MAX_ACTIVE_DASHBOARD_BACKTESTS`` in
    ``api/routers/backtests.py``: an unparseable value read with a bare
    ``int()`` at module scope once killed app boot, so a bad value here logs
    and uses 300 rather than raising.
    """
    raw = os.getenv(_ENV_NAME)
    if raw is None or not str(raw).strip():
        return float(_DEFAULT_INTERVAL_SECONDS)
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        print(
            f"{_ENV_NAME} is not an integer ({raw!r}); using {_DEFAULT_INTERVAL_SECONDS}",
            flush=True,
        )
        return float(_DEFAULT_INTERVAL_SECONDS)
    if value < _MIN_INTERVAL_SECONDS or value > _MAX_INTERVAL_SECONDS:
        print(
            f"{_ENV_NAME} is out of range ({value}; allowed "
            f"{_MIN_INTERVAL_SECONDS}-{_MAX_INTERVAL_SECONDS}); "
            f"using {_DEFAULT_INTERVAL_SECONDS}",
            flush=True,
        )
        return float(_DEFAULT_INTERVAL_SECONDS)
    return float(value)


_worker_lock = threading.Lock()
_worker_thread: threading.Thread | None = None
_worker_stop: threading.Event | None = None


def _default_tick() -> Any:
    from .daily_facts import run_daily_facts

    return run_daily_facts()


def _default_prepare() -> Any:
    from .facts_migration import run_startup_migrations

    return run_startup_migrations()


def start_daily_facts_worker(
    interval_seconds: float | None = None,
    stop_event: threading.Event | None = None,
    *,
    tick: Callable[[], Any] | None = None,
    prepare: Callable[[], Any] | None = None,
) -> threading.Thread:
    """Start the worker (idempotent -- a second call while it is alive no-ops).

    ``tick`` and ``prepare`` are injectable for tests; production leaves both
    at their defaults. The loop waits ``interval_seconds`` *before* the first
    tick, so a boot does not run the day's job on top of everything else the
    startup hook is doing; the claim makes the first tick cheap when the day
    is already done.
    """
    global _worker_thread, _worker_stop
    interval = (
        float(interval_seconds) if interval_seconds is not None else daily_job_interval_seconds()
    )
    with _worker_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return _worker_thread
        stop = stop_event if stop_event is not None else threading.Event()
        tick_fn = tick if tick is not None else _default_tick
        prepare_fn = prepare if prepare is not None else _default_prepare

        def _loop() -> None:
            try:
                prepare_fn()
            except Exception as exc:
                print(
                    "WARNING: analytics.facts_migration_failed "
                    f"category={type(exc).__name__[:80]}"
                )
            while not stop.wait(interval):
                try:
                    tick_fn()
                except Exception as exc:
                    print(
                        "WARNING: analytics.daily_facts.tick_failed "
                        f"category={type(exc).__name__[:80]}"
                    )

        thread = threading.Thread(target=_loop, daemon=True, name="analytics-daily-facts")
        thread.start()
        _worker_thread = thread
        _worker_stop = stop
        return thread


def stop_daily_facts_worker(timeout: float = 5.0) -> None:
    """Stop the worker and wait for it; a no-op when none is running (tests)."""
    global _worker_thread, _worker_stop
    with _worker_lock:
        thread, stop = _worker_thread, _worker_stop
        _worker_thread = None
        _worker_stop = None
    if stop is not None:
        stop.set()
    if thread is not None and thread.is_alive():
        thread.join(timeout=timeout)


__all__ = [
    "daily_job_interval_seconds",
    "start_daily_facts_worker",
    "stop_daily_facts_worker",
]
