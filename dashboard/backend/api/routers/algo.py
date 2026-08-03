"""My Trading Algo API: real LLM chat + async Alpaca backtest.

Canonical location (Phase 3C2). Moved verbatim from
``dashboard/backend/api/algo.py``, which is now a thin compatibility re-export
shim. Endpoint paths, methods, names, prefix, tags, request/response models,
status codes, exception messages, chat/backtest behavior, async/background
behavior, and service calls are unchanged; only the module location moved.
"""

from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from dashboard.backend.domain.backtesting.algo_service import (
    execute_algo,
    get_algo_status,
    get_all_submissions,
    get_default_blocks,
    get_submissions_for_session,
    process_chat,
)

router = APIRouter(prefix="/algo", tags=["algo"])


class AlgoBlocks(BaseModel):
    info_retrieval: str = ""
    signal_transfer: str = ""
    trading_algorithm: str = ""
    stop_loss_take_profit: str = ""


class ChatRequest(BaseModel):
    message: str
    blocks: Optional[AlgoBlocks] = None


class ExecuteRequest(BaseModel):
    blocks: AlgoBlocks
    team_name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


def _blocks_to_dict(blocks: AlgoBlocks | dict | None) -> dict[str, str]:
    if blocks is None:
        return get_default_blocks()
    if isinstance(blocks, dict):
        return blocks
    return blocks.model_dump()


def _require_session(session_id: Optional[str]) -> str:
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing X-Session-Id header")
    return session_id


@router.get("/setup")
def algo_setup_status():
    """Tell frontend which credentials / routes are ready."""
    from dashboard.backend.domain.backtesting.algo_service import _has_alpaca_credentials
    import os
    return {
        "anthropic_configured": bool(os.getenv("ANTHROPIC_API_KEY")),
        "alpaca_configured": _has_alpaca_credentials(),
        "ready": bool(os.getenv("ANTHROPIC_API_KEY")) and _has_alpaca_credentials(),
    }


@router.get("/defaults")
def algo_defaults():
    from dashboard.backend.domain.backtesting.algo_service import _default_backtest_dates
    start, end = _default_backtest_dates()
    return {
        "blocks": get_default_blocks(),
        "backtest_window": {"start_date": start, "end_date": end},
    }


@router.post("/chat")
def algo_chat(body: ChatRequest, x_session_id: Optional[str] = Header(None)):
    _require_session(x_session_id)
    blocks = _blocks_to_dict(body.blocks)
    return process_chat(body.message, blocks)


@router.post("/execute")
def algo_execute(body: ExecuteRequest, x_session_id: Optional[str] = Header(None)):
    session_id = _require_session(x_session_id)
    blocks = _blocks_to_dict(body.blocks)
    if not any(v.strip() for v in blocks.values()):
        raise HTTPException(status_code=400, detail="Strategy blocks cannot be empty")
    try:
        return execute_algo(
            blocks,
            session_id,
            body.team_name,
            body.start_date,
            body.end_date,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/status")
def algo_execution_status(x_session_id: Optional[str] = Header(None)):
    session_id = _require_session(x_session_id)
    return get_algo_status(session_id)


@router.get("/submissions")
def list_submissions(
    x_session_id: Optional[str] = Header(None),
    mine_only: bool = False,
):
    _require_session(x_session_id)
    if mine_only:
        return {"submissions": get_submissions_for_session(x_session_id)}
    return {"submissions": get_all_submissions()}
