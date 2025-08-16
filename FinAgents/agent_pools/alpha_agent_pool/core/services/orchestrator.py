from __future__ import annotations

import queue
import threading
from typing import Optional

from ..domain.models import AlphaTask, Ack, AlphaResult
from ..ports.orchestrator import OrchestratorPort
from .planner import Planner
from .executor import Executor


class Orchestrator(OrchestratorPort):
    """Synchronous intake with background worker executing tasks.

    Thread-safe queue-based intake to keep core sync semantics while execution is async-capable.
    """

    def __init__(self, planner: Planner, executor: Executor):
        self._planner = planner
        self._executor = executor
        self._queue: "queue.Queue[AlphaTask]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._ensure_worker()

    def _ensure_worker(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._run_loop, name="alpha-orchestrator", daemon=True)
        self._worker.start()

    def _run_loop(self) -> None:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        while not self._stop_event.is_set():
            try:
                task = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                plan = self._planner.plan(task)
                result: AlphaResult = loop.run_until_complete(self._executor.run(plan))
                # TODO: push to ResultPort/outbox in adapter layer
            except Exception:
                pass
            finally:
                self._queue.task_done()

        loop.close()

    def submit(self, task: AlphaTask) -> Ack:
        self._queue.put(task)
        return Ack(status="ACK", task_id=task.task_id, idempotency_key=task.idempotency_key)

    def subscribe(self, topic: str) -> None:
        # no-op: real implementation is in adapter layer
        return None

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker:
            self._worker.join(timeout=1.0)

