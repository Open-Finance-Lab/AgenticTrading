"""v2 agents: register / me / rotate-key (spec §4.1, §6)."""

from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel, Field

from dashboard.backend.api.rate_limit import FixedWindowRateLimiter, client_key
from dashboard.backend.domain.agents.repository import agent_store
from dashboard.backend.domain.agents import auth_cache
from dashboard.backend.api.v2.errors import ApiError
from dashboard.backend.api.v2.auth_scopes import resolve_agent, require_scope

router = APIRouter(prefix="/v2/agents", tags=["v2-agents"])

# Registration is unauthenticated by design (it's how you GET a key), so it
# needs a per-client budget: each registration writes a DB row and mints a
# permanent rate-limit bucket. Best-effort (see api/rate_limit) — the header
# key is rotatable — but it bounds naive flooding.
_register_rate_limiter = FixedWindowRateLimiter(
    max_events=int(os.getenv("V2_REGISTRATIONS_PER_HOUR", "120")),
    window_seconds=3600,
)


class RegisterAgentBody(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    model_name: str = Field(default="local-model", max_length=100)


def _public(agent: dict) -> dict:
    return {
        "agent_id": agent["agent_id"],
        "name": agent["name"],
        "session_id": agent["session_id"],
        "model_name": agent["model_name"],
        "scopes": agent.get("scopes", []),
    }


@router.post("")
def register(body: RegisterAgentBody, request: Request):
    """register → create an agent, return api_key (shown once), session_id, scopes."""
    if not _register_rate_limiter.allow(client_key(request)):
        raise ApiError(
            "rate_limited", "Too many agent registrations from this client",
            status=429, retryable=True, details={"retry_after": 3600},
        )
    browser = request.headers.get("x-session-id") or request.headers.get("X-Session-Id")
    agent = agent_store.create_agent(
        name=body.name.strip(),
        model_name=body.model_name.strip() or "local-model",
        owner_browser_session=browser.strip() if browser else None,
    )
    out = _public(agent)
    out["api_key"] = agent["api_key"]
    return out


@router.get("/me")
def me(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    """Resolve the caller from X-API-Key → identity + scopes."""
    agent = resolve_agent(x_api_key)
    return _public(agent)


@router.post("/{agent_id}/rotate-key")
def rotate_key(agent_id: str, agent: dict = Depends(require_scope("agents:register"))):
    """Rotate the caller's own key. The previous key stops working immediately."""
    if agent["agent_id"] != agent_id:
        raise ApiError("forbidden_scope", "Can only rotate your own key", status=403)
    new_key = agent_store.rotate_api_key(agent_id)
    if not new_key:
        raise ApiError("agent_not_found", "Agent not found", status=404)
    auth_cache.invalidate_agent(agent_id)
    return {"agent_id": agent_id, "api_key": new_key}
