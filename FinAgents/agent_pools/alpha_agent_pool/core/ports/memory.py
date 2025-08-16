from __future__ import annotations

from typing import Protocol, Dict, Any


class MemoryPort(Protocol):
    """Port for A2A memory coordinator and peer knowledge exchange."""

    async def retrieve(self, query: Dict[str, Any], scope: str = "global") -> Dict[str, Any]:
        ...

    async def append(self, event: Dict[str, Any]) -> Dict[str, Any]:
        ...

    async def lockless_suggest(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ...

