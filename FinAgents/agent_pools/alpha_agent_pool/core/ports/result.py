from __future__ import annotations

from typing import Protocol, Dict, Any


class ResultPort(Protocol):
    """Port for result publishing and auditing."""

    def publish(self, result: Dict[str, Any]) -> None:
        ...

    def store_artifact(self, blob_ref: str, metadata: Dict[str, Any]) -> None:
        ...

    def emit_event(self, event: Dict[str, Any]) -> None:
        ...

