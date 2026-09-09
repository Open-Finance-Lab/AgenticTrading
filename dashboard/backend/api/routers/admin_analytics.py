"""Admin-only, display-safe Analytics query endpoints."""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Never

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, ValidationError

from dashboard.backend.api.auth import require_admin
from dashboard.backend.domain.analytics.metrics import AnalyticsMetricFilters
from dashboard.backend.domain.analytics.acquisition import (
    AcquisitionAttribution,
    AcquisitionFilters,
    normalize_cohort_slug,
)
from dashboard.backend.domain.analytics.query_service import (
    AnalyticsActivityPage,
    AnalyticsOverview,
    AnalyticsQueryService,
    AnalyticsUserFilters,
    get_analytics_query_service,
    get_value_analytics_query_service,
)
from dashboard.backend.domain.analytics.service import (
    AnalyticsService,
    get_analytics_service,
)
from dashboard.backend.domain.analytics.value_queries import (
    AcquisitionAnalyticsResponse,
    CommercialAnalyticsResponse,
    LifecycleAnalyticsResponse,
    MAX_VALUE_RANGE_DAYS,
    OperationalAnalyticsResponse,
    PaginatedValueUsers,
    RetentionAnalyticsResponse,
    UserValueFilters,
    ValueAnalyticsQueryService,
    ValueUserProfile,
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
_LIFECYCLE_SEGMENTS = {"new", "onboarding", "growing", "core", "at_risk", "dormant"}
_OPERATIONAL_STATES = {"blocked", "needs_attention", "healthy"}
_COMMERCIAL_TIERS = {"unpaid", "starter", "invested", "high_value"}
_LIFECYCLE_MOVEMENT_RANGES = {"5d", "1w", "1m", "1y"}
_ACQUISITION_SOURCES = {"student", "community", "friend", "competition", "unknown"}
_ACQUISITION_LIFECYCLES = {"new", "active", "at_risk", "dormant"}
_DATE_RANGES = {"1d": 1, "1w": 7, "1m": 30, "1y": 365}


class AttributionUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    cohort: str | None = None


def _acquisition_filters(
    values: dict[str, str], *, include_internal: bool
) -> AcquisitionFilters:
    source = values.get("acquisition_source")
    if source is not None and source not in _ACQUISITION_SOURCES:
        _invalid_query()
    lifecycle = values.get("lifecycle")
    if lifecycle is not None and lifecycle not in _ACQUISITION_LIFECYCLES:
        _invalid_query()
    cohort = values.get("acquisition_cohort")
    try:
        return AcquisitionFilters(
            source=source,
            cohort=normalize_cohort_slug(cohort) if cohort is not None else None,
            lifecycle=lifecycle,
            blocked=_parse_bool(values["blocked"]) if "blocked" in values else None,
            paid=_parse_bool(values["paid"]) if "paid" in values else None,
            active=(
                _parse_bool(values["acquisition_active"])
                if "acquisition_active" in values
                else None
            ),
            task_completed=(
                _parse_bool(values["acquisition_task_completed"])
                if "acquisition_task_completed" in values
                else None
            ),
            repeat=(
                _parse_bool(values["acquisition_repeat"])
                if "acquisition_repeat" in values
                else None
            ),
            paid_intent=(
                _parse_bool(values["acquisition_paid_intent"])
                if "acquisition_paid_intent" in values
                else None
            ),
            include_internal=include_internal,
        )
    except (ValidationError, ValueError):
        _invalid_query()


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
        {
            "from",
            "to",
            "date_range",
            "billing_mode",
            "provider",
            "model",
            "include_internal",
            "acquisition_source",
            "acquisition_cohort",
            "lifecycle",
            "blocked",
            "paid",
            "acquisition_active",
            "acquisition_task_completed",
            "acquisition_repeat",
            "acquisition_paid_intent",
        },
    )
    now = datetime.now(timezone.utc)
    from_date = _parse_date(values["from"]) if "from" in values else None
    to_date = _parse_date(values["to"]) if "to" in values else None
    if from_date is not None and to_date is not None and to_date < from_date:
        _invalid_query()

    date_range = values.get("date_range")
    if date_range is not None and date_range not in _DATE_RANGES:
        _invalid_query()
    if date_range is not None and (from_date is not None or to_date is not None):
        _invalid_query()
    end = _exclusive_date_end(to_date) if to_date else now
    try:
        start = (
            _utc_midnight(from_date)
            if from_date
            else end - timedelta(days=_DATE_RANGES.get(date_range or "1m", 30))
        )
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
        include_internal = (
            _parse_bool(values["include_internal"])
            if "include_internal" in values
            else False
        )
        return AnalyticsMetricFilters(
            start=start,
            end=end,
            billing_mode=billing_mode,
            provider_id=provider_id,
            model_id=model_id,
            include_internal=include_internal,
            acquisition=_acquisition_filters(values, include_internal=include_internal),
        )
    except (ValidationError, ValueError):
        _invalid_query()


def _value_range_from_values(values: dict[str, str]) -> tuple[date, date, bool]:
    today = datetime.now(timezone.utc).date()
    date_range = values.get("date_range")
    if date_range is not None:
        if date_range not in _DATE_RANGES or "from" in values or "to" in values:
            _invalid_query()
        end = today + timedelta(days=1)
        start = end - timedelta(days=_DATE_RANGES[date_range])
        include_internal = (
            _parse_bool(values["include_internal"])
            if "include_internal" in values
            else False
        )
        return start, end, include_internal
    from_date = _parse_date(values["from"]) if "from" in values else None
    to_date = _parse_date(values["to"]) if "to" in values else None
    if from_date is not None and to_date is not None and to_date < from_date:
        _invalid_query()
    try:
        end = (to_date or today) + timedelta(days=1)
        start = from_date or end - timedelta(days=30)
    except OverflowError:
        _invalid_query()
    if end <= start or (end - start).days > MAX_VALUE_RANGE_DAYS:
        _invalid_query()
    include_internal = (
        _parse_bool(values["include_internal"])
        if "include_internal" in values
        else False
    )
    return start, end, include_internal


def _value_range(
    request: Request,
    *,
    additional: set[str] | None = None,
) -> tuple[date, date, bool, dict[str, str]]:
    values = _query_values(
        request,
        {"from", "to", "date_range", "include_internal"} | (additional or set()),
    )
    start, end, include_internal = _value_range_from_values(values)
    return start, end, include_internal, values


def _profile_range(request: Request) -> tuple[date, date]:
    values = _query_values(request, {"from", "to", "date_range"})
    start, end, _include_internal = _value_range_from_values(values)
    return start, end


def _value_user_filters(request: Request) -> tuple[UserValueFilters, int, int]:
    values = _query_values(
        request,
        {
            "q",
            "status",
            "lifecycle_segment",
            "operational_state",
            "commercial_tier",
            "activated",
            "last_meaningful_activity_from",
            "last_meaningful_activity_to",
            "priority",
            "limit",
            "offset",
            "include_internal",
            "acquisition_source",
            "acquisition_cohort",
            "lifecycle",
            "blocked",
            "paid",
            "acquisition_active",
            "acquisition_task_completed",
            "acquisition_repeat",
            "acquisition_paid_intent",
            "from",
            "to",
            "date_range",
        },
    )
    query = values.get("q")
    if query is not None and len(query) > 100:
        _invalid_query()
    lifecycle_segment = values.get("lifecycle_segment")
    if lifecycle_segment is not None and lifecycle_segment not in _LIFECYCLE_SEGMENTS:
        _invalid_query()
    operational_state = values.get("operational_state")
    if operational_state is not None and operational_state not in _OPERATIONAL_STATES:
        _invalid_query()
    tier = values.get("commercial_tier")
    if tier is not None and tier not in _COMMERCIAL_TIERS:
        _invalid_query()
    legacy_status = values.get("status")
    if legacy_status is not None and legacy_status not in _USER_STATES:
        _invalid_query()

    acquisition_range_values = {
        key: values[key]
        for key in ("from", "to", "date_range", "include_internal")
        if key in values
    }
    acquisition_start, acquisition_end, _ = _value_range_from_values(
        acquisition_range_values
    )

    from_date = (
        _parse_date(values["last_meaningful_activity_from"])
        if "last_meaningful_activity_from" in values
        else None
    )
    to_date = (
        _parse_date(values["last_meaningful_activity_to"])
        if "last_meaningful_activity_to" in values
        else None
    )
    if from_date is not None and to_date is not None and to_date < from_date:
        _invalid_query()
    try:
        include_internal = (
            _parse_bool(values["include_internal"])
            if "include_internal" in values
            else False
        )
        filters = UserValueFilters(
            q=query,
            lifecycle_segment=lifecycle_segment,
            operational_state=operational_state,
            commercial_tier=tier,
            activated=(
                _parse_bool(values["activated"]) if "activated" in values else None
            ),
            last_meaningful_activity_from=(
                _utc_midnight(from_date) if from_date is not None else None
            ),
            last_meaningful_activity_to=(
                _exclusive_date_end(to_date) - timedelta(microseconds=1)
                if to_date is not None
                else None
            ),
            priority=(
                _parse_bool(values["priority"]) if "priority" in values else False
            ),
            legacy_status=legacy_status,
            include_internal=include_internal,
            acquisition=_acquisition_filters(values, include_internal=include_internal),
            acquisition_start=_utc_midnight(acquisition_start),
            acquisition_end=_utc_midnight(acquisition_end),
        )
    except (ValidationError, ValueError):
        _invalid_query()
    limit = _parse_integer(values.get("limit", "50"), minimum=1, maximum=100)
    offset = _parse_integer(values.get("offset", "0"), minimum=0)
    return filters, limit, offset


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
        _exclusive_date_end(to_date) - timedelta(microseconds=1) if to_date else None
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
    if isinstance(exc, HTTPException):
        raise exc
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


@router.get("/acquisition", response_model=AcquisitionAnalyticsResponse)
def get_acquisition(
    request: Request,
    service: ValueAnalyticsQueryService = Depends(get_value_analytics_query_service),
):
    start, end, include_internal, values = _value_range(
        request,
        additional={
            "group_by",
            "acquisition_source",
            "acquisition_cohort",
            "lifecycle",
            "blocked",
            "paid",
            "acquisition_active",
            "acquisition_task_completed",
            "acquisition_repeat",
            "acquisition_paid_intent",
        },
    )
    group_by = values.get("group_by", "source")
    if group_by not in {"source", "cohort"}:
        _invalid_query()
    try:
        return service.get_acquisition_groups(
            start=start,
            end=end,
            group_by=group_by,
            filters=_acquisition_filters(values, include_internal=include_internal),
        )
    except Exception as exc:
        _raise_service_error(exc)


@router.get("/lifecycle", response_model=LifecycleAnalyticsResponse)
def get_lifecycle(
    request: Request,
    service: ValueAnalyticsQueryService = Depends(get_value_analytics_query_service),
):
    start, end, include_internal, values = _value_range(
        request,
        additional={"movement_range"},
    )
    movement_range = values.get("movement_range", "5d")
    if movement_range not in _LIFECYCLE_MOVEMENT_RANGES:
        _invalid_query()
    try:
        return service.get_lifecycle(
            start=start,
            end=end,
            include_internal=include_internal,
            movement_range=movement_range,
        )
    except Exception as exc:
        _raise_service_error(exc)


@router.get("/retention", response_model=RetentionAnalyticsResponse)
def get_retention(
    request: Request,
    service: ValueAnalyticsQueryService = Depends(get_value_analytics_query_service),
):
    start, end, include_internal, _values = _value_range(request)
    try:
        return service.get_retention(
            start=start,
            end=end,
            include_internal=include_internal,
        )
    except Exception as exc:
        _raise_service_error(exc)


@router.get("/commercial", response_model=CommercialAnalyticsResponse)
def get_commercial(
    request: Request,
    service: ValueAnalyticsQueryService = Depends(get_value_analytics_query_service),
):
    start, end, include_internal, _values = _value_range(request)
    try:
        return service.get_commercial(
            start=start,
            end=end,
            include_internal=include_internal,
        )
    except Exception as exc:
        _raise_service_error(exc)


@router.get("/operational", response_model=OperationalAnalyticsResponse)
def get_operational(
    request: Request,
    service: ValueAnalyticsQueryService = Depends(get_value_analytics_query_service),
):
    start, end, include_internal, values = _value_range(
        request,
        additional={"billing_mode", "provider", "model"},
    )
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
        return service.get_operational(
            start=start,
            end=end,
            include_internal=include_internal,
            billing_mode=billing_mode,
            provider_id=provider_id,
            model_id=model_id,
        )
    except Exception as exc:
        _raise_service_error(exc)


@router.get("/users", response_model=PaginatedValueUsers)
def list_users(
    request: Request,
    service: ValueAnalyticsQueryService = Depends(get_value_analytics_query_service),
):
    filters, limit, offset = _value_user_filters(request)
    try:
        return service.list_users(filters=filters, limit=limit, offset=offset)
    except Exception as exc:
        _raise_service_error(exc)


@router.get("/users/{user_id}", response_model=ValueUserProfile)
def get_user_profile(
    user_id: str,
    request: Request,
    admin: dict = Depends(require_admin),
    query_service: ValueAnalyticsQueryService = Depends(
        get_value_analytics_query_service
    ),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
):
    start, end = _profile_range(request)
    subject_user_id = _parse_user_id(user_id)
    try:
        profile = query_service.get_user_profile(
            user_id=subject_user_id,
            start=start,
            end=end,
        )
    except Exception as exc:
        _raise_service_error(exc)
    _record_access(
        analytics_service,
        admin=admin,
        subject_user_id=subject_user_id,
        section="overview",
    )
    return profile


@router.patch("/users/{user_id}/attribution", response_model=AcquisitionAttribution)
def update_user_attribution(
    user_id: str,
    payload: AttributionUpdateRequest,
    admin: dict = Depends(require_admin),
    service: ValueAnalyticsQueryService = Depends(get_value_analytics_query_service),
):
    subject_user_id = _parse_user_id(user_id)
    if payload.source not in _ACQUISITION_SOURCES:
        _invalid_query()
    try:
        return service.store.update_user_attribution(
            subject_user_id,
            source=payload.source,
            cohort=payload.cohort,
            actor_user_id=int(admin["id"]),
        )
    except Exception as exc:
        _raise_service_error(exc)


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
