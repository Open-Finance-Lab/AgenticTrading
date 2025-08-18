from __future__ import annotations

from typing import Protocol

from ..domain.models import AlphaTask, Ack


class OrchestratorPort(Protocol):
    """Port for task intake and result subscription."""

    def submit(self, task: AlphaTask) -> Ack:  # sync semantic; adapter may persist/queue
        ...

    def subscribe(self, topic: str) -> None:
        ...

