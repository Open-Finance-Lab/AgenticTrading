"""Admin-only, display-safe Analytics query endpoints."""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Never

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ValidationError

from dashboard.backend.api.auth import require_admin
from dashboard.backend.domain.analytics.metrics import AnalyticsMetricFilters
from dashboard.backend.domain.analytics.query_service import (
    AnalyticsActivityPage,
    AnalyticsOverview,
    AnalyticsQueryService,
    AnalyticsUserFilters,
    AnalyticsUserProfile,
    PaginatedUsers,
    get_analytics_query_service,
)
from dashboard.backend.domain.analytics.service import (
    AnalyticsService,
    get_analytics_service,
)


router = APIRouter(
    prefix="/admin/analytics",
    tags=["admin-analytics"],
    dependencies=[Depends(require_admin)],
)

_INVALID_QUERY_DETAIL = "Invalid Analytics query."
_NOT_FOUND_DETAIL = "Analytics user was not found."
_UNAVAILABLE_DETAIL = "Analytics is temporarily unavailable."
_PROVIDER_ID_PATTERN = re.compile(r"^[a-z0-9_]{2,64}$")
_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/\-:]{0,255}$")
_POSITIVE_INTEGER_PATTERN = re.compile(r"^[0-9]+$")
_USER_STATES = {"blocked", "needs_attention", "dormant", "onboarding", "active"}
_USER_SORTS = {"last_activity", "joined_at", "recent_runs", "recent_failures"}
_SORT_ORDERS = {"asc", "desc"}
_ACTIVITY_SECTIONS = {"timeline", "runs", "usage", "sessions"}


def _invalid_query() -> Never:
    raise HTTPException(status_code=422, detail=_INVALID_QUERY_DETAIL)


def _query_values(request: Request, allowed: set[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for key, value in request.query_params.multi_items():
        if key not in allowed or key in values:
            _invalid_query()
        values[key] = value
    return values


def _parse_date(value: str) -> date:
    if len(value) != 10:
        _invalid_query()
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        _invalid_query()
    if parsed.isoformat() != value:
        _invalid_query()
    return parsed


def _utc_midnight(value: date) -> datetime:
    return datetime.combine(value, time.min, tzinfo=timezone.utc)


def _exclusive_date_end(value: date) -> datetime:
    try:
        return _utc_midnight(value + timedelta(days=1))
    except OverflowError:
        _invalid_query()


def _parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    _invalid_query()


def _parse_integer(
    value: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if len(value) > 20 or not _POSITIVE_INTEGER_PATTERN.fullmatch(value):
        _invalid_query()
    parsed = int(value)
    if parsed < minimum or (maximum is not None and parsed > maximum):
        _invalid_query()
    return parsed


def _parse_user_id(value: str) -> int:
    return _parse_integer(value, minimum=1)


def _overview_filters(request: Request) -> AnalyticsMetricFilters:
    values = _query_values(
        request,
        {"from", "to", "billing_mode", "provider", "model", "include_internal"},
    )
    now = datetime.now(timezone.utc)
    from_date = _parse_date(values["from"]) if "from" in values else None
    to_date = _parse_date(values["to"]) if "to" in values else None
    if from_date is not None and to_date is not None and to_date < from_date:
        _invalid_query()

    end = _exclusive_date_end(to_date) if to_date else now
    try:
        start = _utc_midnight(from_date) if from_date else end - timedelta(days=30)
    except OverflowError:
        _invalid_query()

    billing_mode = values.get("billing_mode")
    if billing_mode not in {None, "byok", "platform_credits"}:
        _invalid_query()
    provider_id = values.get("provider")
    if provider_id is not None and not _PROVIDER_ID_PATTERN.fullmatch(provider_id):
        _invalid_query()
    model_id = values.get("model")
    if model_id is not None and not _MODEL_ID_PATTERN.fullmatch(model_id):
        _invalid_query()

    try:
        return AnalyticsMetricFilters(
            start=start,
            end=end,
            billing_mode=billing_mode,
            provider_id=provider_id,
            model_id=model_id,
            include_internal=(
                _parse_bool(values["include_internal"])
                if "include_internal" in values
                else False
            ),
        )
    except (ValidationError, ValueError):
        _invalid_query()


def _user_filters(request: Request) -> tuple[AnalyticsUserFilters, int, int]:
    values = _query_values(
        request,
        {
            "q",
            "status",
            "last_activity_from",
            "last_activity_to",
            "sort",
            "order",
            "limit",
            "offset",
            "include_internal",
        },
    )
    query = values.get("q")
    if query is not None and len(query) > 100:
        _invalid_query()
    status = values.get("status")
    if status is not None and status not in _USER_STATES:
        _invalid_query()
    sort = values.get("sort", "last_activity")
    if sort not in _USER_SORTS:
        _invalid_query()
    order = values.get("order", "desc")
    if order not in _SORT_ORDERS:
        _invalid_query()

    from_date = (
        _parse_date(values["last_activity_from"])
        if "last_activity_from" in values
        else None
    )
    to_date = (
        _parse_date(values["last_activity_to"])
        if "last_activity_to" in values
        else None
    )
    if from_date is not None and to_date is not None and to_date < from_date:
        _invalid_query()
    activity_start = _utc_midnight(from_date) if from_date else None
    activity_end = (
        _exclusive_date_end(to_date) - timedelta(microseconds=1)
        if to_date
        else None
    )

    try:
        filters = AnalyticsUserFilters(
            q=query,
            status=status,
            last_activity_from=activity_start,
            last_activity_to=activity_end,
            sort=sort,
            order=order,
            include_internal=(
                _parse_bool(values["include_internal"])
                if "include_internal" in values
                else False
            ),
        )
    except (ValidationError, ValueError):
        _invalid_query()
    limit = _parse_integer(values.get("limit", "50"), minimum=1, maximum=100)
    offset = _parse_integer(values.get("offset", "0"), minimum=0)
    return filters, limit, offset


def _activity_query(request: Request) -> tuple[str, int, str | None]:
    values = _query_values(request, {"section", "limit", "cursor"})
    section = values.get("section")
    if section not in _ACTIVITY_SECTIONS:
        _invalid_query()
    limit = _parse_integer(values.get("limit", "50"), minimum=1, maximum=100)
    cursor = values.get("cursor")
    if cursor is not None and (not cursor or len(cursor) > 256):
        _invalid_query()
    return section, limit, cursor


def _raise_service_error(exc: Exception) -> Never:
    if isinstance(exc, LookupError):
        raise HTTPException(status_code=404, detail=_NOT_FOUND_DETAIL) from None
    if isinstance(exc, (ValidationError, ValueError)):
        raise HTTPException(status_code=422, detail=_INVALID_QUERY_DETAIL) from None
    raise HTTPException(status_code=503, detail=_UNAVAILABLE_DETAIL) from None


def _record_access(
    service: AnalyticsService,
    *,
    admin: dict,
    subject_user_id: int,
    section: str,
) -> None:
    try:
        service.record_admin_profile_access(
            actor=admin,
            subject_user_id=subject_user_id,
            section=section,
        )
    except Exception:
        raise HTTPException(status_code=503, detail=_UNAVAILABLE_DETAIL) from None


@router.get("/overview", response_model=AnalyticsOverview)
def get_overview(
    request: Request,
    service: AnalyticsQueryService = Depends(get_analytics_query_service),
):
    filters = _overview_filters(request)
    try:
        return service.get_overview(filters=filters)
    except Exception as exc:
        _raise_service_error(exc)


@router.get("/users", response_model=PaginatedUsers)
def list_users(
    request: Request,
    service: AnalyticsQueryService = Depends(get_analytics_query_service),
):
    filters, limit, offset = _user_filters(request)
    try:
        return service.list_users(filters=filters, limit=limit, offset=offset)
    except Exception as exc:
        _raise_service_error(exc)


@router.get("/users/{user_id}", response_model=AnalyticsUserProfile)
def get_user_profile(
    user_id: str,
    request: Request,
    admin: dict = Depends(require_admin),
    query_service: AnalyticsQueryService = Depends(get_analytics_query_service),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
):
    _query_values(request, set())
    subject_user_id = _parse_user_id(user_id)
    try:
        profile = query_service.get_user_profile(user_id=subject_user_id)
    except Exception as exc:
        _raise_service_error(exc)
    _record_access(
        analytics_service,
        admin=admin,
        subject_user_id=subject_user_id,
        section="overview",
    )
    return profile


@router.get("/users/{user_id}/activity", response_model=AnalyticsActivityPage)
def get_user_activity(
    user_id: str,
    request: Request,
    admin: dict = Depends(require_admin),
    query_service: AnalyticsQueryService = Depends(get_analytics_query_service),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
):
    subject_user_id = _parse_user_id(user_id)
    section, limit, cursor = _activity_query(request)
    try:
        activity = query_service.get_user_activity(
            user_id=subject_user_id,
            section=section,
            limit=limit,
            cursor=cursor,
        )
    except Exception as exc:
        _raise_service_error(exc)
    _record_access(
        analytics_service,
        admin=admin,
        subject_user_id=subject_user_id,
        section=section,
    )
    return activity


__all__ = ["router"]
