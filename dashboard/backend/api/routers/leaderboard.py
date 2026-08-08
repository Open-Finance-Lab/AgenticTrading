"""Leaderboard API — contest baselines and daily rolling board.

Canonical location (Phase 3C4). Moved verbatim from
``dashboard/backend/api/leaderboard.py``, which is now a thin compatibility
re-export shim. Endpoint path, method, name, prefix, tags, query parameters,
status codes, exception messages, and service calls are unchanged; only the
module location moved.
"""

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from dashboard.backend.api.rate_limit import FixedWindowRateLimiter, client_key
from dashboard.backend.domain.leaderboard.service import (
    enqueue_daily_leaderboard_refresh,
    get_leaderboard,
    verify_daily_refresh_secret,
)

router = APIRouter(prefix="/v1/leaderboard", tags=["leaderboard"])

# The refresh hook is unauthenticated apart from one shared secret, and a hit
# schedules real LLM spend. Best-effort budget (see api/rate_limit): it bounds
# naive abuse and secret-guessing volume, it is not the security boundary.
_daily_refresh_rate_limiter = FixedWindowRateLimiter(max_events=20, window_seconds=3600)


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
    request: Request,
    deploy_models: bool = Query(
        default=True,
        description="Also run every competition LLM model for the daily window (expensive).",
    ),
    force: bool = Query(default=False, description="Ignore the per-window refresh cache."),
    x_leaderboard_refresh_secret: str | None = Header(default=None, alias="X-Leaderboard-Refresh-Secret"),
):
    """Cron/admin hook: enqueue a Daily Leaderboard refresh (non-blocking).

    Requires ``LEADERBOARD_DAILY_REFRESH_SECRET`` and the matching request header.
    Returns **202 Accepted** immediately; model deploys run in a background
    thread so Render/GitHub Actions HTTP timeouts cannot abort a multi-hour job.
    Poll ``GET /api/v1/leaderboard?period=daily`` for ``daily_status``.

    There is deliberately no ``allow_fallback`` parameter: that flag bypasses
    the H6 leaderboard-integrity guard (publishing a rule-based curve under an
    LLM's name) and stays a local operator decision on
    ``scripts/refresh_daily_leaderboard.py``, not something reachable over HTTP
    behind a single shared secret.
    """
    if not _daily_refresh_rate_limiter.allow(client_key(request)):
        raise HTTPException(
            status_code=429,
            detail="Too many daily leaderboard refresh requests. Try again later.",
        )

    try:
        verify_daily_refresh_secret(x_leaderboard_refresh_secret)
    except ValueError:
        # Unconfigured and wrong-secret both answer 401: a public endpoint
        # should not tell an anonymous caller whether it is armed. The operator
        # signal goes to the server log instead.
        print("⚠️ Daily leaderboard refresh rejected: LEADERBOARD_DAILY_REFRESH_SECRET is not configured")
        raise HTTPException(
            status_code=401, detail="Invalid daily leaderboard refresh secret"
        ) from None
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    try:
        payload = enqueue_daily_leaderboard_refresh(
            deploy_models=deploy_models,
            force_refresh=force,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to enqueue daily leaderboard refresh",
        ) from None

    return JSONResponse(status_code=202, content=payload)
