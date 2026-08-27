"""Authenticated ingestion for privacy-reduced frontend Analytics events."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from dashboard.backend.api.auth import get_current_user
from dashboard.backend.api.rate_limit import FixedWindowRateLimiter
from dashboard.backend.domain.analytics.models import FrontendAnalyticsEvent
from dashboard.backend.domain.analytics.privacy import request_analytics_context
from dashboard.backend.domain.analytics.repository_common import (
    AnalyticsIdempotencyConflictError,
)
from dashboard.backend.domain.analytics.service import (
    AnalyticsService,
    get_analytics_service,
)


router = APIRouter(prefix="/analytics", tags=["analytics"])

MAX_ANALYTICS_BODY_BYTES = 8 * 1024
_ANALYTICS_INGESTION_LIMITER = FixedWindowRateLimiter(
    max_events=120,
    window_seconds=300,
)


def reset_analytics_ingestion_limiter() -> None:
    _ANALYTICS_INGESTION_LIMITER.reset()


def _limit_ingestion(user_id: int) -> None:
    key = f"analytics-ingestion:{user_id}"
    if _ANALYTICS_INGESTION_LIMITER.allow(key):
        return
    raise HTTPException(
        status_code=429,
        detail="Too many analytics events; please try again later.",
        headers={
            "Retry-After": str(
                _ANALYTICS_INGESTION_LIMITER.retry_after_seconds(key)
            )
        },
    )


async def _parse_event(request: Request) -> FrontendAnalyticsEvent:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_ANALYTICS_BODY_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="Analytics event is too large.",
                )
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid request.",
            ) from None

    chunks: list[bytes] = []
    received = 0
    async for chunk in request.stream():
        received += len(chunk)
        if received > MAX_ANALYTICS_BODY_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Analytics event is too large.",
            )
        chunks.append(chunk)

    try:
        payload = json.loads(b"".join(chunks))
        if not isinstance(payload, dict):
            raise ValueError("request body must be an object")
        return FrontendAnalyticsEvent.model_validate(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError):
        raise HTTPException(
            status_code=422,
            detail="Invalid analytics event.",
        ) from None


@router.post("/events", status_code=status.HTTP_202_ACCEPTED)
async def ingest_analytics_event(
    request: Request,
    current_user: dict = Depends(get_current_user),
    service: AnalyticsService = Depends(get_analytics_service),
):
    user_id = int(current_user["id"])
    _limit_ingestion(user_id)
    payload = await _parse_event(request)
    received_at = datetime.now(timezone.utc)
    try:
        context = request_analytics_context(request, received_at)
        await run_in_threadpool(
            service.accept_frontend_event,
            user=current_user,
            payload=payload,
            context=context,
            received_at=received_at,
        )
    except AnalyticsIdempotencyConflictError:
        raise HTTPException(
            status_code=409,
            detail="Analytics event conflicts with a replay.",
        ) from None
    except (ValidationError, ValueError):
        raise HTTPException(
            status_code=422,
            detail="Invalid analytics event.",
        ) from None
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Analytics is temporarily unavailable.",
        ) from None
    return {"accepted": True}
