"""Account-bound portfolio API (signed-in users only)."""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from dashboard.backend.api.auth import get_current_user
from dashboard.backend.api.dependencies import _browser_context
from dashboard.backend.domain.agents.service import agent_service
from dashboard.backend.domain.backtesting.constants import MAX_AGENT_CASH_ALLOCATION
from dashboard.backend.domain.portfolios.service import AgentScope, portfolio_service

router = APIRouter(prefix="/v1/portfolio", tags=["portfolio"])


class TransferBody(BaseModel):
    agent_id: str = Field(min_length=1, max_length=100)
    amount: float = Field(gt=0, le=MAX_AGENT_CASH_ALLOCATION)


def _scope(request: Request) -> AgentScope:
    """Scope every figure to the agent set this caller's My Agents list shows.

    The Capital Allocation pie draws one slice per agent in that list, so a
    total summed over a narrower set publishes a number the panel beside it
    contradicts -- and offers the difference back as cash the account can spend
    a second time.
    """
    return AgentScope.from_owner_context(_browser_context(request))


def _owned_agent(agent_id: str, user_id: int) -> dict:
    agent = agent_service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    if agent.get("owner_user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not your agent")
    return agent


@router.get("")
def get_portfolio(request: Request, current_user: dict = Depends(get_current_user)):
    """Return the caller's portfolio, bootstrapping at $10k if missing."""
    portfolio = portfolio_service.get_or_create_portfolio(
        current_user["id"], _scope(request)
    )
    return {"portfolio": portfolio}


@router.post("/allocate")
def allocate_cash(
    request: Request,
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
            scope=_scope(request),
        )
    except ValueError as exc:
        # InsufficientCashError subclasses ValueError; both are caller errors.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result["agent"] = agent_service.agent_with_stats(result["agent"])
    return result


@router.post("/reclaim")
def reclaim_cash(
    request: Request,
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
            scope=_scope(request),
        )
    except ValueError as exc:
        # InsufficientSleeveError subclasses ValueError; both are caller errors.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result["agent"] = agent_service.agent_with_stats(result["agent"])
    return result
