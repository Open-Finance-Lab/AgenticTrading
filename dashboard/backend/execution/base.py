"""ExecutionBackend interface. Schema parity is universal; lifecycle parity is per-loop."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# A run's terminal statuses — no longer active, cannot advance. Single source
# of truth shared by every liveness/cap/archival check so a new status (e.g. a
# distinct "cancelled") can't silently drift between the surfaces that gate on
# it (BacktestBackend.is_active, ArchivedBacktestBackend, the v2 reaper).
TERMINAL_STATUSES = ("completed", "failed", "closed")


class ExecutionBackend(ABC):
    """One run's execution. `loop` advertises lifecycle parity: lockstep | realtime."""

    loop: str = "lockstep"

    @abstractmethod
    def build_context(self) -> Dict[str, Any]:
        """Return a ContextEnvelope-shaped dict for the current step."""

    @abstractmethod
    def apply_decisions(self, actions: List[Dict[str, Any]],
                        expected_step: Optional[int] = None) -> Dict[str, Any]:
        """Validate + execute actions; return a SubmitAck-shaped dict.

        ``expected_step`` is the step the caller read as current; a backend
        refuses the decision (step_already_closed) if the run has moved on."""

    @abstractmethod
    def status(self) -> Dict[str, Any]:
        """Return a run-status dict."""

    @abstractmethod
    def result(self) -> Optional[Dict[str, Any]]:
        """Return a ResultEnvelope-shaped dict, or None if not finalized."""

    def decisions(self) -> List[Dict[str, Any]]:
        """Per-decision log for this run (empty if the backend keeps none)."""
        return []

    def is_active(self) -> bool:
        """Cheap, side-effect-free liveness peek (used by the active-run cap).

        Must not take engine locks or advance state — status() is NOT a
        substitute: on a live session it can cascade into deadline handling
        and finalization."""
        return True

    def advance(self) -> None:
        """Lockstep stepping hook. Realtime backends are wall-clock driven (no-op)."""
        return None

    def cancel(self) -> None:
        """Best-effort cancel → closed."""
        return None
