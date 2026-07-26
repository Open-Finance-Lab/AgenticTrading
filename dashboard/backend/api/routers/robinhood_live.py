"""Robinhood live trading + connection status API."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from dashboard.backend.api.auth import get_current_user
from dashboard.backend.domain.agents.service import agent_service
from dashboard.backend.domain.brokers import live_service
from dashboard.backend.domain.brokers.repository import broker_store

router = APIRouter(prefix="/v1/robinhood", tags=["robinhood"])


class LiveRunBody(BaseModel):
    dry_run: bool = Field(default=True)


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
    except ValueError as exc:
        if str(exc) == "robinhood_not_connected":
            return {
                "connected": False,
                "broker": "robinhood",
                "execute_enabled": live_service.execute_enabled(),
            }
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"robinhood_status_failed:{exc}") from exc
    return status


@router.delete("/disconnect")
async def robinhood_disconnect(current_user: dict = Depends(get_current_user)):
    broker_store.delete(int(current_user["id"]))
    return {"status": "ok", "connected": False}


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
        )
    except ValueError as exc:
        code = str(exc)
        if code == "robinhood_not_connected":
            raise HTTPException(status_code=409, detail="Connect Robinhood first") from exc
        if code == "live_trading_not_enabled":
            raise HTTPException(status_code=409, detail="Enable live trading for this agent") from exc
        raise HTTPException(status_code=400, detail=code) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"live_run_failed:{exc}") from exc
    return result
