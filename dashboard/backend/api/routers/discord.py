"""Discord bot-facing endpoints (service-secret auth)."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from dashboard.backend.api import discord_oauth
from dashboard.backend.domain.agents.service import agent_service
from dashboard.backend import users as users_module

router = APIRouter(prefix="/v1/discord", tags=["discord"])


def _require_bot_auth(
    x_discord_bot_secret: Optional[str],
    x_discord_user_id: Optional[str],
) -> tuple[dict, str]:
    if not discord_oauth.verify_bot_secret(x_discord_bot_secret):
        raise HTTPException(status_code=401, detail="Invalid Discord bot secret")
    discord_user_id = (x_discord_user_id or "").strip()
    if not discord_user_id:
        raise HTTPException(status_code=400, detail="X-Discord-User-Id is required")
    user = users_module.user_store.get_user_by_discord_id(discord_user_id)
    if not user:
        raise HTTPException(
            status_code=404,
            detail={"code": "discord_not_linked", "message": "Discord account is not linked"},
        )
    return user, discord_user_id


@router.get("/agents")
def list_discord_user_agents(
    x_discord_bot_secret: Optional[str] = Header(default=None, alias="X-Discord-Bot-Secret"),
    x_discord_user_id: Optional[str] = Header(default=None, alias="X-Discord-User-Id"),
):
    """List agents owned by the website user linked to this Discord id.

    Called by the Discord bot only (shared ``DISCORD_BOT_API_SECRET``).
    Built-in agents are sorted first so ``/agent`` can prefer selectable cards.
    """
    user, _discord_user_id = _require_bot_auth(x_discord_bot_secret, x_discord_user_id)
    agents = agent_service.list_agents_with_stats(
        owner_user_id=int(user["id"]),
        owner_browser_session=None,
        trading_session_id=None,
    )

    public = []
    for agent in agents:
        latest = agent.get("latest_run") or {}
        public.append(
            {
                "agent_id": agent["agent_id"],
                "name": agent["name"],
                "agent_type": agent.get("agent_type") or "external",
                "model_name": agent.get("model_name") or "local-model",
                "description": agent.get("description"),
                "run_count": agent.get("run_count", 0),
                "latest_return": latest.get("total_return"),
                "latest_sharpe": latest.get("sharpe_ratio"),
            }
        )

    # Prefer builtins for Discord select (backtest agent_id path requires builtin).
    public.sort(key=lambda a: (0 if a["agent_type"] == "builtin" else 1, a["name"].lower()))
    return {
        "user": {
            "id": user["id"],
            "display_name": user.get("display_name"),
        },
        "agents": public,
    }
