"""Robinhood live trading + connection status API."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from dashboard.backend.api.auth import get_current_user
from dashboard.backend.domain.agents.service import agent_service
from dashboard.backend.execution import robinhood_live_service as live_service
from dashboard.backend.domain.brokers.repository import broker_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/robinhood", tags=["robinhood"])


class LiveRunBody(BaseModel):
    dry_run: bool = Field(default=True)
    idempotency_key: Optional[str] = Field(default=None, max_length=128)


@router.get("/status")
async def robinhood_status(
    portfolio: bool = False,
    current_user: dict = Depends(get_current_user),
):
    try:
        status = await live_service.get_connection_status(
            int(current_user["id"]),
            include_portfolio=portfolio,
        )
    except Exception as exc:
        logger.exception("robinhood_status_failed")
        raise HTTPException(status_code=502, detail="Failed to fetch Robinhood status") from exc
    return status


@router.delete("/disconnect")
async def robinhood_disconnect(current_user: dict = Depends(get_current_user)):
    user_id = int(current_user["id"])
    revoked_upstream = False
    revoke = getattr(live_service, "revoke_connection", None)
    if callable(revoke):
        try:
            await revoke(user_id)
            revoked_upstream = True
        except Exception:
            logger.warning("robinhood_revoke_failed for user_id=%s", user_id, exc_info=True)
    broker_store.delete(user_id)
    return {"status": "ok", "connected": False, "revoked_upstream": revoked_upstream}


@router.post("/agents/{agent_id}/live-run")
async def robinhood_live_run(
    agent_id: str,
    body: LiveRunBody,
    current_user: dict = Depends(get_current_user),
):
    agent = agent_service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    if int(agent.get("owner_user_id") or 0) != int(current_user["id"]):
        raise HTTPException(status_code=403, detail="Agent not owned by this user")

    try:
        result = await live_service.run_live_for_agent(
            user_id=int(current_user["id"]),
            agent=agent,
            dry_run=body.dry_run,
            idempotency_key=body.idempotency_key,
        )
    except ValueError as exc:
        code = str(exc)
        if code == "robinhood_not_connected":
            raise HTTPException(status_code=409, detail="Connect Robinhood first") from exc
        if code == "live_trading_not_enabled":
            raise HTTPException(status_code=409, detail="Enable live trading for this agent") from exc
        if code == "live_run_in_progress":
            raise HTTPException(
                status_code=409, detail="A live run is already in progress for this account."
            ) from exc
        if code == "llm_unavailable":
            raise HTTPException(
                status_code=503, detail="No LLM provider is configured, so no decision was made."
            ) from exc
        raise HTTPException(status_code=400, detail=code) from exc
    except Exception as exc:
        logger.exception("live_run_failed")
        raise HTTPException(status_code=502, detail="Live run failed") from exc
    return result
