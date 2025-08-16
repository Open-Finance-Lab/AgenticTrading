from __future__ import annotations

from typing import Protocol, Dict, Any


class FeaturePort(Protocol):
    """Port for feature/data retrieval or computation."""

    async def fetch(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        ...

    async def compute(self, node_ctx: Dict[str, Any]) -> Dict[str, Any]:
        ...

