"""Shared FastAPI auth/ownership dependency helpers.

Canonical home (Phase 3A4) for the owner-context and agent-access helpers used by
both the agents router (``api/routers/agents.py``) and the protocol auth fallback
(``api/protocol_auth.py``). Moved verbatim from ``api/routers/agents.py`` so the
two consumers no longer reach through the legacy ``api/agents.py`` shim.

Authentication, ownership, browser-session fallback, exception types, and status
codes are unchanged; only the definition location moved.
"""

from typing import Any, Dict, Optional

from fastapi import HTTPException, Request

from dashboard.backend.domain.agents.service import (
    AgentAccessDeniedError,
    AgentNotFoundError,
    agent_service,
)


def _optional_user(request: Request, authorization: Optional[str] = None) -> Optional[dict]:
    from dashboard.backend.api.auth import _session_token
    from dashboard.backend.users import user_store

    token = _session_token(request, authorization)
    if not token:
        return None
    return user_store.get_user_for_token(token)


def _owner_context(request: Request, authorization: Optional[str]) -> Dict[str, Any]:
    trading_session = request.headers.get("x-session-id") or request.headers.get("X-Session-Id")
    browser_owner = request.headers.get("x-browser-id") or request.headers.get("X-Browser-Id")
    if not browser_owner:
        # Fallback: with no X-Browser-Id, the X-Session-Id doubles as the owner
        # identity. For agents created via import_session (which stores
        # owner_browser_session = session_id) the session id therefore IS an
        # ownership credential — "session_id is never a credential" only holds
        # for built-in agents. Clients that can send X-Browser-Id should; this
        # branch exists for API-only importers with no browser identity.
        browser_owner = trading_session
    user = _optional_user(request, authorization)
    return {
        "user_id": user["id"] if user else None,
        "browser_session": browser_owner.strip() if browser_owner else None,
        "trading_session": trading_session.strip() if trading_session else None,
    }


def _require_owner_context(request: Request, authorization: Optional[str]) -> Dict[str, Any]:
    ctx = _owner_context(request, authorization)
    if not ctx["user_id"] and not ctx["browser_session"]:
        raise HTTPException(
            status_code=400,
            detail="Missing owner context (log in or send X-Session-Id header)",
        )
    return ctx


def _require_agent_access(
    agent_id: str,
    ctx: Dict[str, Any],
    *,
    api_key: Optional[str] = None,
    reclaim_on_session_match: bool = False,
) -> Dict[str, Any]:
    agent, _owned = _resolve_agent_access(
        agent_id,
        ctx,
        api_key=api_key,
        reclaim_on_session_match=reclaim_on_session_match,
    )
    return agent


def _resolve_agent_access(
    agent_id: str,
    ctx: Dict[str, Any],
    *,
    api_key: Optional[str] = None,
    reclaim_on_session_match: bool = False,
) -> tuple[Dict[str, Any], bool]:
    """``(agent, owned)``. See ``AgentService.resolve_access``.

    Routes that only *read* or edit the agent can use ``_require_agent_access``
    and ignore the flag. Routes that write ownership must not: a grant that came
    from a bare ``session_id`` match is not proof of ownership, and binding an
    account on it cannot be undone by the real owner.
    """
    # The agent's own API key is a valid credential for its own agent — this is
    # how API-only clients (which have no logged-in user or browser session)
    # authorize state-changing operations on the agent they own.
    if api_key:
        keyed = agent_service.resolve_api_key(api_key)
        if keyed and keyed.get("agent_id") == agent_id:
            return keyed, True
    try:
        return agent_service.resolve_access(
            agent_id,
            user_id=ctx.get("user_id"),
            browser_session=ctx.get("browser_session"),
            trading_session=ctx.get("trading_session"),
            reclaim_on_session_match=reclaim_on_session_match,
        )
    except AgentNotFoundError:
        raise HTTPException(status_code=404, detail="Agent not found")
    except AgentAccessDeniedError:
        raise HTTPException(status_code=403, detail="Not your agent")
