"""Leaderboard API — contest baselines and daily rolling board.

Canonical location (Phase 3C4). Moved verbatim from
``dashboard/backend/api/leaderboard.py``, which is now a thin compatibility
re-export shim. Endpoint path, method, name, prefix, tags, query parameters,
status codes, exception messages, and service calls are unchanged; only the
module location moved.
"""

from fastapi import APIRouter, Header, HTTPException, Query

from dashboard.backend.domain.leaderboard.service import (
    LeaderboardFallbackError,
    get_leaderboard,
    refresh_daily_leaderboard,
    verify_daily_refresh_secret,
)

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
    Pass ?period=daily for the rolling one-day board (weekends show Friday).
    """
    try:
        return get_leaderboard(force_refresh=refresh, period=period)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/daily/refresh")
def api_refresh_daily_leaderboard(
    deploy_models: bool = Query(
        default=True,
        description="Also run every competition LLM model for the daily window (expensive).",
    ),
    force: bool = Query(default=False, description="Ignore the per-window refresh cache."),
    allow_fallback: bool = Query(
        default=False,
        description="Allow publishing LLM entries that fell back to rule-based trading.",
    ),
    x_leaderboard_refresh_secret: str | None = Header(default=None, alias="X-Leaderboard-Refresh-Secret"),
):
    """Cron/admin hook: refresh the Daily Leaderboard for the last completed weekday.

    Requires ``LEADERBOARD_DAILY_REFRESH_SECRET`` and the matching request header.
    Used by ``dashboard/scripts/refresh_daily_leaderboard.py`` and the GitHub
    Actions nightly workflow.
    """
    try:
        verify_daily_refresh_secret(x_leaderboard_refresh_secret)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    try:
        return refresh_daily_leaderboard(
            deploy_models=deploy_models,
            force_refresh=force,
            allow_fallback=allow_fallback,
        )
    except LeaderboardFallbackError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
