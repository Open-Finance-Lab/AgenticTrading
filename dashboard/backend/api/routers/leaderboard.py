"""Leaderboard API — contest baselines and (later) agent teams.

Canonical location (Phase 3C4). Moved verbatim from
``dashboard/backend/api/leaderboard.py``, which is now a thin compatibility
re-export shim. Endpoint path, method, name, prefix, tags, query parameters,
status codes, exception messages, and service calls are unchanged; only the
module location moved.
"""

from fastapi import APIRouter, HTTPException, Query

from dashboard.backend.domain.leaderboard.service import get_leaderboard

router = APIRouter(prefix="/v1/leaderboard", tags=["leaderboard"])


@router.get("")
def api_get_leaderboard(
    refresh: bool = Query(default=False),
    period: str = Query(
        default="contest",
        description="Leaderboard period: 'contest' (fixed preseason window) or 'daily' (last completed weekday).",
    ),
):
    """
    Official competition / daily leaderboard for the requested period.

    Baselines are computed from Alpaca hourly backtest data and cached in SQLite.
    Pass ?refresh=true to recompute (e.g. after config change).
    Pass ?period=daily for the rolling one-day board.
    """
    try:
        return get_leaderboard(force_refresh=refresh, period=period)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
