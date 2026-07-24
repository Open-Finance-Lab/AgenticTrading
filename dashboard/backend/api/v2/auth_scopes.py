"""Scope constants and the require_scope() FastAPI dependency (spec §6)."""

from __future__ import annotations

from typing import List, Optional

from fastapi import Header

from dashboard.backend.domain.agents import auth_cache
from dashboard.backend.api.v2.errors import ApiError

SCOPES: List[str] = [
    "agents:register", "runs:write", "context:read",
    "decisions:write", "runs:read",
]


def parse_scopes(csv: str) -> List[str]:
    return [s.strip() for s in (csv or "").split(",") if s.strip()]


def resolve_agent(x_api_key: Optional[str]) -> dict:
    """Resolve an X-API-Key to an agent record, or raise unauthorized."""
    agent = auth_cache.resolve_api_key((x_api_key or "").strip())
    if not agent:
        raise ApiError("unauthorized", "Invalid or missing API key", status=401)
    return agent


def require_scope(scope: str):
    """FastAPI dependency factory: resolve caller and assert it holds `scope`."""

    def dependency(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> dict:
        agent = resolve_agent(x_api_key)
        if scope not in agent.get("scopes", []):
            raise ApiError(
                "forbidden_scope", f"Missing required scope: {scope}",
                status=403, details={"required": scope, "held": agent.get("scopes", [])},
            )
        return agent

    return dependency
