"""Background baseline generation with config-level dedup (T2).

Baselines (buy-hold + DJIA) depend only on the run *config*, not the agent, so
a wave of same-config finalizes needs exactly one baseline pair. Finalize
enqueues a job here and returns; one lazily-started daemon thread drains the
queue (single consumer — the dedup cache therefore needs no lock: job N for a
config fully completes, DB write included, before job N+1 dequeues).

Degradation semantics match today's best-effort baselines: a full queue drops
the job with a printed error (the run stays completed without baselines), a
failure is printed and swallowed, and jobs are lost on process restart —
identical durability to the old in-request path, which lost baselines on a
mid-finalize restart too.

Each job carries a direct reference to its dataset's ``all_data`` (the T1
bundle), so LRU eviction between enqueue and drain can never force a
refetch/rebuild storm; same-config jobs share one object, bounding pinned
memory by the distinct configs in the queue.

``publish`` callbacks must swap in a NEW dict (never mutate one the session
already exposed) — see ExternalBacktestSession._publish_baselines.
"""

from __future__ import annotations

import os
import queue
import threading
from typing import Any, Callable, Dict, Optional, Tuple

from dashboard.backend import database as db_module
from dashboard.backend.domain.backtesting.engine import HourlyBacktester

BASELINE_QUEUE_MAX = int(os.getenv("BASELINE_QUEUE_MAX", "500"))
QUEUE_DEPTH_WARN = 25
# A per-item warning cannot report a total contract break (repo rule): this is
# what makes a wholesale failure (e.g. an upstream signature change breaking
# every job) loud instead of just a stream of per-job warnings nobody reads.
_ESCALATION_THRESHOLD = 3

_STOP = object()


class BaselineJob:
    __slots__ = ("run_id", "session_id", "start_date", "end_date", "mode",
                 "all_data", "publish")

    def __init__(self, *, run_id: str, session_id: str, start_date: str,
                 end_date: str, mode: str, all_data: Dict[str, Any],
                 publish: Callable[[Dict[str, str]], None]):
        self.run_id = run_id
        self.session_id = session_id
        self.start_date = start_date
        self.end_date = end_date
        self.mode = mode
        self.all_data = all_data
        self.publish = publish


_queue: "queue.Queue" = queue.Queue(maxsize=BASELINE_QUEUE_MAX)
_worker_thread: Optional[threading.Thread] = None
_worker_lock = threading.Lock()
# (start, end, mode) -> {"buy_and_hold": id, "djia": id}. Single-consumer, so
# unlocked access is safe; grows one tiny entry per distinct config.
_completed: Dict[Tuple[str, str, str], Dict[str, str]] = {}
# Both counters are written only by the single drain thread (same
# single-consumer reasoning as _completed above), so they need no lock either.
# _consecutive_failures resets to 0 on any success and drives the escalation
# print below; _total_failures never resets — it's the plain integer a caller
# like stress_serve.py reads for a shutdown summary.
_consecutive_failures = 0
_total_failures = 0


def submit(job: BaselineJob) -> bool:
    """Enqueue baseline generation for a finalized run. Never blocks or raises."""
    _ensure_worker()
    try:
        _queue.put_nowait(job)
    except queue.Full:
        print(f"⚠️ Baseline queue full ({_queue.maxsize}); dropping baselines "
              f"for {job.run_id} (run saved)")
        return False
    depth = _queue.qsize()
    if depth > QUEUE_DEPTH_WARN:
        print(f"⚠️ Baseline queue depth {depth} — worker backlog")
    return True


def _ensure_worker() -> None:
    global _worker_thread
    with _worker_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return
        _worker_thread = threading.Thread(
            target=_drain_forever, args=(_queue,), daemon=True,
            name="baseline-worker")
        _worker_thread.start()


def _drain_forever(q: "queue.Queue") -> None:
    global _consecutive_failures, _total_failures
    while True:
        job = q.get()
        try:
            if job is _STOP:
                return
            _run_job(job)
            _consecutive_failures = 0
        # SystemExit guard mirrors the old in-finalize catch: a daemon thread
        # swallows SystemExit silently, which would kill the worker forever.
        except (Exception, SystemExit) as exc:
            print(f"⚠️ Baseline generation failed (run saved): {exc}")
            _consecutive_failures += 1
            _total_failures += 1
            if _consecutive_failures == _ESCALATION_THRESHOLD:
                print(
                    f"🔥 Baseline worker: {_consecutive_failures} consecutive "
                    f"failures — last error: {exc}"
                )
        finally:
            q.task_done()


def _run_job(job: BaselineJob) -> None:
    key = (job.start_date, job.end_date, job.mode)
    ids = _completed.get(key)
    if ids is None:
        backtester = HourlyBacktester(
            job.start_date, job.end_date, job.session_id,
            use_llm=False, mode=job.mode,
        )
        backtester.all_data = job.all_data
        buyhold_id, _ = backtester.run_buyhold_baseline()
        djia_id, _ = backtester.run_djia_baseline()
        ids = {}
        if buyhold_id:
            ids["buy_and_hold"] = buyhold_id
        if djia_id:
            ids["djia"] = djia_id
        _completed[key] = ids
    # db is late-bound through the module attribute so per-test DB swaps
    # (monkeypatch.setattr(db_module, "db", ...)) reach the worker thread.
    db_module.db.update_run_baselines(
        job.run_id,
        djia_run_id=ids.get("djia"),
        buyhold_run_id=ids.get("buy_and_hold"),
    )
    job.publish(dict(ids))


def wait_idle(timeout: float = 30.0) -> bool:
    """Test helper: True once every enqueued job (publish included) finished."""
    import time
    deadline = time.monotonic() + timeout
    with _queue.all_tasks_done:
        while _queue.unfinished_tasks:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            _queue.all_tasks_done.wait(remaining)
    return True


def _reset_for_tests(maxsize: Optional[int] = None) -> None:
    """Fresh queue + dedup cache; wakes a worker blocked on the old queue."""
    global _queue, _worker_thread, _consecutive_failures, _total_failures
    with _worker_lock:
        old_q = _queue
        _completed.clear()
        _consecutive_failures = 0
        _total_failures = 0
        _queue = queue.Queue(
            maxsize=maxsize if maxsize is not None else BASELINE_QUEUE_MAX)
        if _worker_thread is not None and _worker_thread.is_alive():
            try:
                old_q.put_nowait(_STOP)
            except queue.Full:
                try:
                    old_q.get_nowait()
                    old_q.task_done()
                except queue.Empty:
                    pass  # queue drained concurrently; nothing to evict before enqueuing _STOP
                old_q.put_nowait(_STOP)
        _worker_thread = None
