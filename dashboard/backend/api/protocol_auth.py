"""Shared auth helpers for the Agent-Environment Protocol endpoints.

External agents authenticate directly with their Agent API key
(``X-API-Key``). For agent-scoped management endpoints we also accept the
existing owner context (logged-in user or browser session) for backward
compatibility with the dashboard.
"""

from typing import Any, Dict, Optional

from fastapi import HTTPException, Request

from dashboard.backend.domain.agents import auth_cache


def resolve_agent_by_key(x_api_key: Optional[str]) -> Dict[str, Any]:
    """Resolve an Agent API key to the agent, or raise 401."""
    agent = auth_cache.resolve_api_key(x_api_key or "")
    if not agent:
        raise HTTPException(status_code=401, detail="Invalid or missing API key (X-API-Key)")
    return agent


def require_agent_access(
    agent_id: str,
    *,
    request: Request,
    x_api_key: Optional[str],
    authorization: Optional[str],
) -> Dict[str, Any]:
    """Authorize access to a specific agent via API key or owner context."""
    if x_api_key:
        agent = resolve_agent_by_key(x_api_key)
        if agent["agent_id"] != agent_id:
            raise HTTPException(status_code=403, detail="API key does not match this agent")
        return agent

    # Fall back to owner context (dashboard / browser session / logged-in user).
    from dashboard.backend.api.dependencies import _owner_context, _require_agent_access

    ctx = _owner_context(request, authorization)
    if not ctx["user_id"] and not ctx["browser_session"] and not ctx["trading_session"]:
        raise HTTPException(
            status_code=401,
            detail="Provide X-API-Key, log in, or send X-Session-Id",
        )
    return _require_agent_access(agent_id, ctx)
