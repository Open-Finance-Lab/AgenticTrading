from __future__ import annotations

from typing import Protocol, Dict, Any


class StrategyPort(Protocol):
    """Port for strategy plugins (local or via MCP)."""

    def probe(self) -> Dict[str, Any]:
        ...

    async def run(self, node_ctx: Dict[str, Any]) -> Dict[str, Any]:
        ...

