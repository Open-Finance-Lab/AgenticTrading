"""Account-bound portfolio API (signed-in users only)."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from dashboard.backend.api.auth import get_current_user
from dashboard.backend.domain.agents.service import agent_service
from dashboard.backend.domain.backtesting.constants import MAX_AGENT_CASH_ALLOCATION
from dashboard.backend.domain.portfolios.service import portfolio_service

router = APIRouter(prefix="/v1/portfolio", tags=["portfolio"])


class TransferBody(BaseModel):
    agent_id: str = Field(min_length=1, max_length=100)
    amount: float = Field(gt=0, le=MAX_AGENT_CASH_ALLOCATION)


def _owned_agent(agent_id: str, user_id: int) -> dict:
    agent = agent_service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    if agent.get("owner_user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not your agent")
    return agent


@router.get("")
def get_portfolio(current_user: dict = Depends(get_current_user)):
    """Return the caller's portfolio, bootstrapping at $10k if missing."""
    portfolio = portfolio_service.get_or_create_portfolio(current_user["id"])
    return {"portfolio": portfolio}


@router.post("/allocate")
def allocate_cash(
    body: TransferBody,
    current_user: dict = Depends(get_current_user),
):
    """Move unallocated cash → agent sleeve."""
    agent = _owned_agent(body.agent_id, current_user["id"])
    try:
        result = portfolio_service.allocate_to_agent(
            owner_user_id=current_user["id"],
            agent=agent,
            amount=body.amount,
        )
    except ValueError as exc:
        # InsufficientCashError subclasses ValueError; both are caller errors.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result["agent"] = agent_service.agent_with_stats(result["agent"])
    return result


@router.post("/reclaim")
def reclaim_cash(
    body: TransferBody,
    current_user: dict = Depends(get_current_user),
):
    """Move agent sleeve → unallocated cash."""
    agent = _owned_agent(body.agent_id, current_user["id"])
    try:
        result = portfolio_service.reclaim_from_agent(
            owner_user_id=current_user["id"],
            agent=agent,
            amount=body.amount,
        )
    except ValueError as exc:
        # InsufficientSleeveError subclasses ValueError; both are caller errors.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result["agent"] = agent_service.agent_with_stats(result["agent"])
    return result
