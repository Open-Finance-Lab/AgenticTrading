from __future__ import annotations

import json
from typing import Any, Dict, Callable

from ...corepkg.domain.models import AlphaTask, Ack
from ...corepkg.ports.orchestrator import OrchestratorPort


class McpServerAdapter(OrchestratorPort):
    """Adapter exposing orchestrator via MCP server tool handlers.

    This is a façade; actual server binding remains in existing core.py to avoid breaking external semantics.
    """

    def __init__(self, submit_fn: Callable[[AlphaTask], Ack], subscribe_fn: Callable[[str], None] | None = None):
        self._submit = submit_fn
        self._subscribe = subscribe_fn or (lambda _topic: None)

    def submit(self, task: AlphaTask) -> Ack:
        return self._submit(task)

    def subscribe(self, topic: str) -> None:
        self._subscribe(topic)

    # Helper to parse DTO and call core
    def handle_submit_dto(self, dto: Dict[str, Any]) -> Dict[str, Any]:
        task = AlphaTask(
            task_id=dto["task_id"],
            strategy_id=dto["strategy_id"],
            market_ctx=dto.get("market_ctx", {}),
            time_window=dto.get("time_window", {}),
            features_req=dto.get("features_req", []),
            risk_hint=dto.get("risk_hint"),
            idempotency_key=dto.get("idempotency_key"),
        )
        ack = self.submit(task)
        return {"status": ack.status, "task_id": ack.task_id, "idempotency_key": ack.idempotency_key}

