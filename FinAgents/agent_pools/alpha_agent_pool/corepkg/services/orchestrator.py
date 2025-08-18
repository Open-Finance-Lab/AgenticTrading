from __future__ import annotations

import queue
import threading
import time
from typing import Optional, Dict, Any, List
from enum import Enum

from ..domain.models import AlphaTask, Ack, AlphaResult
from ..ports.orchestrator import OrchestratorPort
from .planner import Planner
from .executor import Executor


class TaskStatus(Enum):
    """Task execution status."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Orchestrator(OrchestratorPort):
    """Thread-safe orchestrator with complete task lifecycle management."""

    def __init__(self, planner: Planner, executor: Executor):
        self._planner = planner
        self._executor = executor
        self._queue: "queue.Queue[AlphaTask]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Task tracking
        self._task_status: Dict[str, Dict[str, Any]] = {}
        self._active_tasks: Dict[str, AlphaTask] = {}
        self._cancelled_tasks: set = set()
        self._lock = threading.RLock()
        
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
            
            # Check if task was cancelled
            if task.task_id in self._cancelled_tasks:
                self._update_task_status(task.task_id, TaskStatus.CANCELLED)
                self._queue.task_done()
                continue
            
            try:
                # Update status to running
                self._update_task_status(task.task_id, TaskStatus.RUNNING)
                
                # Execute task
                plan = self._planner.plan(task)
                result: AlphaResult = loop.run_until_complete(self._executor.run(plan))
                
                # Update status to completed
                self._update_task_status(task.task_id, TaskStatus.COMPLETED, result=result)
                
            except Exception as e:
                # Update status to failed
                self._update_task_status(task.task_id, TaskStatus.FAILED, error=str(e))
            finally:
                # Remove from active tasks
                with self._lock:
                    self._active_tasks.pop(task.task_id, None)
                self._queue.task_done()

        loop.close()

    def _update_task_status(self, task_id: str, status: TaskStatus, **kwargs):
        """Update task status with thread safety."""
        with self._lock:
            if task_id not in self._task_status:
                self._task_status[task_id] = {}
            
            self._task_status[task_id].update({
                "status": status.value,
                "updated_at": time.time(),
                **kwargs
            })

    def submit(self, task: AlphaTask) -> Ack:
        """Submit task for processing."""
        with self._lock:
            # Check for duplicate task
            if task.task_id in self._task_status:
                existing_status = self._task_status[task.task_id]["status"]
                if existing_status in [TaskStatus.QUEUED.value, TaskStatus.RUNNING.value]:
                    return Ack(status="DUPLICATE", task_id=task.task_id, idempotency_key=task.idempotency_key)
            
            # Add to tracking
            self._active_tasks[task.task_id] = task
            self._update_task_status(task.task_id, TaskStatus.QUEUED)
        
        # Queue for execution
        self._queue.put(task)
        return Ack(status="ACK", task_id=task.task_id, idempotency_key=task.idempotency_key)

    def subscribe(self, topic: str) -> None:
        """Subscribe to result notifications (placeholder for future implementation)."""
        # TODO: Implement pub/sub mechanism for result streaming
        pass

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a submitted task."""
        with self._lock:
            return self._task_status.get(task_id, None)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        with self._lock:
            if task_id not in self._task_status:
                return False
            
            current_status = self._task_status[task_id]["status"]
            
            # Can only cancel queued or running tasks
            if current_status == TaskStatus.QUEUED.value:
                # Remove from queue if possible (best effort)
                self._cancelled_tasks.add(task_id)
                self._update_task_status(task_id, TaskStatus.CANCELLED)
                return True
            elif current_status == TaskStatus.RUNNING.value:
                # Mark for cancellation (actual cancellation depends on execution)
                self._cancelled_tasks.add(task_id)
                return True
            
            return False

    def list_active_tasks(self) -> List[str]:
        """List all currently active task IDs."""
        with self._lock:
            return [
                task_id for task_id, status_info in self._task_status.items()
                if status_info["status"] in [TaskStatus.QUEUED.value, TaskStatus.RUNNING.value]
            ]

    def get_task_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics."""
        with self._lock:
            status_counts = {}
            for status_info in self._task_status.values():
                status = status_info["status"]
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                "queue_size": self._queue.qsize(),
                "active_tasks": len(self._active_tasks),
                "total_tasks": len(self._task_status),
                "status_distribution": status_counts,
                "worker_alive": self._worker.is_alive() if self._worker else False
            }

    def stop(self) -> None:
        """Stop orchestrator gracefully."""
        self._stop_event.set()
        if self._worker:
            self._worker.join(timeout=1.0)

