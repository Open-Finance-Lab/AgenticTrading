"""External agent backtest API — hourly step loop with configurable decision timeout.

Canonical location (Phase 3C2). Moved verbatim from
``dashboard/backend/api/external_backtest.py``, which is now a thin compatibility
re-export shim. Endpoint paths, methods, names, prefix, tags, request/response
models, status codes, exception messages, session behavior, and service calls are
unchanged; only the module location moved.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from dashboard.backend.domain.backtesting.external_run_service import (
    DECISION_TIMEOUT_SECONDS,
    BacktestCapacityError,
    get_backtest_decisions,
    get_current_step,
    get_decision_format,
    get_run_decisions,
    get_run_result,
    get_run_trades,
    get_session,
    get_status,
    start_backtest,
    submit_decisions,
    verify_session,
)
from dashboard.backend.infrastructure.llm.validator import DJIA_30

router = APIRouter(prefix="/v1/backtest", tags=["external-backtest"])


class StartBacktestRequest(BaseModel):
    start_date: str = Field(examples=["2026-04-15"])
    end_date: str = Field(examples=["2026-04-16"])
    agent_name: str = Field(default="external-agent", min_length=1, max_length=100)
    model_name: str = Field(default="local-model", min_length=1, max_length=100)
    mode: str = Field(default="safe_trading", pattern="^(safe_trading|buy_and_hold)$")


class TradingActionItem(BaseModel):
    action: str = Field(description="buy, sell, or hold")
    symbol: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=5, max_length=500)
    position_size: int = Field(ge=0)
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None


class SubmitDecisionsRequest(BaseModel):
    actions: List[TradingActionItem]


def _require_session(session_id: Optional[str]) -> str:
    if not session_id or not session_id.strip():
        raise HTTPException(status_code=400, detail="Missing X-Session-Id header")
    return session_id.strip()


def _require_backtest(backtest_id: str, session_id: str):
    session = get_session(backtest_id)
    if not session:
        raise HTTPException(status_code=404, detail="Backtest not found")
    if not verify_session(session, session_id):
        raise HTTPException(status_code=403, detail="Backtest belongs to a different session")
    return session


@router.get("/schema")
def api_decision_schema():
    """
    Decision JSON schema for external agents.

    Submit this shape to POST .../steps/current/decisions each trading hour.
    """
    return {
        "decision_timeout_seconds": DECISION_TIMEOUT_SECONDS,
        "valid_symbols": DJIA_30,
        "format": get_decision_format(),
        "workflow": [
            "POST /api/v1/backtest/start",
            "GET  /api/v1/backtest/{backtest_id}/steps/current",
            "POST /api/v1/backtest/{backtest_id}/steps/current/decisions",
            "GET  /api/v1/backtest/runs/{run_id}/result",
        ],
    }


@router.post("/start")
def api_start_backtest(
    body: StartBacktestRequest,
    x_session_id: Optional[str] = Header(None),
):
    """
    Start an external-agent backtest.

    Loads Alpaca data, then waits for decisions at each trading hour.
    Use the same X-Session-Id as the dashboard to see results on the website.
    """
    session_id = _require_session(x_session_id)
    try:
        result = start_backtest(
            session_id=session_id,
            agent_name=body.agent_name,
            model_name=body.model_name,
            start_date=body.start_date,
            end_date=body.end_date,
            mode=body.mode,
            # The only caller that opts in. Nothing else gates this route —
            # any non-empty X-Session-Id reaches it — and its runs never enter
            # protocol_runs, so the entitlement/per-agent/global caps on the
            # /v1/runs and /v2 surfaces are all blind to them.
            enforce_session_cap=True,
            emit_analytics=True,
        )
    except BacktestCapacityError as exc:
        raise HTTPException(
            status_code=429,
            detail=(
                f"This {exc.scope} is at its concurrent-backtest limit "
                f"({exc.active} of {exc.limit} active); wait for one to finish"
            ),
            headers={"Retry-After": "30"},
        ) from exc
    if result.get("status") == "failed":
        raise HTTPException(status_code=500, detail=result.get("error", "Backtest failed to start"))
    return result


@router.get("/runs/{run_id}/result")
def api_run_result(run_id: str, x_session_id: Optional[str] = Header(None)):
    """Full result for a completed run: metadata, equity curve, trades, decisions."""
    session_id = _require_session(x_session_id)
    result = get_run_result(run_id, session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Run not found or not in your session")
    return result


@router.get("/runs/{run_id}/trades")
def api_run_trades(run_id: str, x_session_id: Optional[str] = Header(None)):
    """Trade log for a completed external-agent backtest run."""
    session_id = _require_session(x_session_id)
    trades = get_run_trades(run_id, session_id)
    if trades is None:
        raise HTTPException(status_code=404, detail="Run not found or not in your session")
    return {"run_id": run_id, "trades": trades, "count": len(trades)}


@router.get("/runs/{run_id}/decisions")
def api_run_decisions(run_id: str, x_session_id: Optional[str] = Header(None)):
    """Hourly decision log for a completed external-agent backtest run."""
    session_id = _require_session(x_session_id)
    decisions = get_run_decisions(run_id, session_id)
    if decisions is None:
        raise HTTPException(status_code=404, detail="Run not found or not in your session")
    return {"run_id": run_id, "decisions": decisions, "count": len(decisions)}


@router.get("/{backtest_id}/status")
def api_backtest_status(
    backtest_id: str,
    x_session_id: Optional[str] = Header(None),
):
    """Poll backtest progress."""
    session_id = _require_session(x_session_id)
    _require_backtest(backtest_id, session_id)
    return get_status(backtest_id)


@router.get("/{backtest_id}/decisions")
def api_backtest_decisions(
    backtest_id: str,
    x_session_id: Optional[str] = Header(None),
):
    """In-progress or completed decision log for an active backtest session."""
    session_id = _require_session(x_session_id)
    _require_backtest(backtest_id, session_id)
    decisions = get_backtest_decisions(backtest_id)
    return {"backtest_id": backtest_id, "decisions": decisions or [], "count": len(decisions or [])}


@router.get("/{backtest_id}/steps/current")
def api_get_current_step(
    backtest_id: str,
    x_session_id: Optional[str] = Header(None),
):
    """Get market context for the current hour. Auto-holds after timeout without a decision."""
    session_id = _require_session(x_session_id)
    _require_backtest(backtest_id, session_id)
    return get_current_step(backtest_id)


@router.post("/{backtest_id}/steps/current/decisions")
def api_submit_decisions(
    backtest_id: str,
    body: SubmitDecisionsRequest,
    x_session_id: Optional[str] = Header(None),
):
    """
    Submit trading decisions for the current hour.

    Body format:
    {"actions": [{"action":"buy|sell|hold","symbol":"AAPL","confidence":0.8,...}]}
    """
    session_id = _require_session(x_session_id)
    _require_backtest(backtest_id, session_id)
    payload: Dict[str, Any] = {"actions": [a.model_dump() for a in body.actions]}
    try:
        result = submit_decisions(backtest_id, payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Backtest not found")
    if not result.get("accepted") and result.get("error") == "step_already_closed":
        raise HTTPException(status_code=409, detail=result)
    return result
