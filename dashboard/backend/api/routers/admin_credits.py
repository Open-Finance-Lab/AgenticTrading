"""Administrator Grant Credits workspace API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from dashboard.backend import users as users_module
from dashboard.backend.api.auth import require_admin
from dashboard.backend.api.rate_limit import FixedWindowRateLimiter
from dashboard.backend.domain.credits.models import (
    AssignGrantRequest,
    FundGrantPoolRequest,
    ReclaimGrantRequest,
    ReduceGrantPoolRequest,
    format_credits,
)
from dashboard.backend.domain.credits.repository_common import (
    CreditAccountRestrictedStoreError,
    CreditsStoreError,
    GrantPoolInsufficientError,
    GrantReclaimExceedsAvailableError,
    IdempotencyConflictError,
)
from dashboard.backend.domain.credits.service import credits_service


router = APIRouter(
    prefix="/admin/credits",
    tags=["admin-credits"],
    dependencies=[Depends(require_admin)],
)

_MUTATION_LIMITER = FixedWindowRateLimiter(max_events=30, window_seconds=300)


class _TargetGrantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: StrictInt = Field(gt=0)


class AssignGrantPayload(_TargetGrantRequest, AssignGrantRequest):
    pass


class ReclaimGrantPayload(_TargetGrantRequest, ReclaimGrantRequest):
    pass


def reset_admin_credits_limiter() -> None:
    _MUTATION_LIMITER.reset()


def _rate_limit(admin_id: int) -> None:
    key = f"admin-grants:{int(admin_id)}"
    if _MUTATION_LIMITER.allow(key):
        return
    raise HTTPException(
        status_code=429,
        detail="Too many Grant Credits administration requests; please try again later.",
        headers={"Retry-After": str(_MUTATION_LIMITER.retry_after_seconds(key))},
    )


def _raise_grant_http_error(exc: Exception) -> None:
    if isinstance(exc, IdempotencyConflictError):
        raise HTTPException(
            status_code=409,
            detail="Grant operation conflicts with an existing idempotency key.",
        ) from exc
    if isinstance(exc, GrantPoolInsufficientError):
        raise HTTPException(
            status_code=422,
            detail="Grant Pool does not have enough available Credits.",
        ) from exc
    if isinstance(exc, GrantReclaimExceedsAvailableError):
        raise HTTPException(
            status_code=422,
            detail="The user does not have enough available Grant Credits.",
        ) from exc
    if isinstance(exc, CreditAccountRestrictedStoreError):
        raise HTTPException(
            status_code=422,
            detail=(
                "The target account is paused for payment refund review; "
                "an administrator must reinstate it before assigning Credits."
            ),
        ) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=422, detail="Invalid Grant Credits request.") from exc
    if isinstance(exc, CreditsStoreError):
        raise HTTPException(
            status_code=503,
            detail="Grant Credits service is temporarily unavailable.",
        ) from exc
    raise exc


def _public_activity(entry: dict[str, Any]) -> dict[str, Any]:
    amount_micro = abs(int(entry["amount_micro"]))
    return {
        "id": int(entry["id"]),
        "pool_id": entry["pool_id"],
        "pool_name": entry.get("pool_name_snapshot"),
        "pool_status": entry.get("pool_status_snapshot"),
        "entry_type": entry["entry_type"],
        "operation_id": entry["operation_id"],
        "amount_micro": amount_micro,
        "display_credits": format_credits(amount_micro),
        "actor_user_id": int(entry["actor_user_id"]),
        "user_id": int(entry["user_id"]) if entry.get("user_id") is not None else None,
        "source": entry["source"],
        "reason": entry["reason"],
        "created_at": entry["created_at"],
    }


@router.get("/grant-pool")
def get_grant_pool(
    pool_id: str = Query(default="default", min_length=1, max_length=120),
    month_start_iso: str | None = Query(default=None),
    _admin: dict = Depends(require_admin),
):
    try:
        summary = credits_service.get_grant_pool_summary(pool_id, month_start_iso)
    except Exception as exc:
        _raise_grant_http_error(exc)
    return {"pool": summary.model_dump()}


@router.get("/grant-pool/activity")
def get_grant_pool_activity(
    pool_id: str = Query(default="default", min_length=1, max_length=120),
    limit: int = Query(default=50, ge=1, le=100),
    cursor: int | None = Query(default=None, ge=1),
    _admin: dict = Depends(require_admin),
):
    try:
        page = credits_service.list_grant_pool_activity(
            pool_id, limit=limit, cursor=cursor
        )
    except Exception as exc:
        _raise_grant_http_error(exc)
    return {
        "items": [_public_activity(item) for item in page["items"]],
        "next_cursor": page["next_cursor"],
    }


@router.post("/grant-pool/fund")
def fund_grant_pool(
    payload: FundGrantPoolRequest,
    admin: dict = Depends(require_admin),
):
    admin_id = int(admin["id"])
    _rate_limit(admin_id)
    try:
        result = credits_service.fund_grant_pool(admin_id=admin_id, request=payload)
    except Exception as exc:
        _raise_grant_http_error(exc)
    return {"grant": result.model_dump()}


@router.post("/grant-pool/reduce")
def reduce_grant_pool(
    payload: ReduceGrantPoolRequest,
    admin: dict = Depends(require_admin),
):
    admin_id = int(admin["id"])
    _rate_limit(admin_id)
    try:
        result = credits_service.reduce_grant_pool(admin_id=admin_id, request=payload)
    except Exception as exc:
        _raise_grant_http_error(exc)
    return {"grant": result.model_dump()}


@router.get("/users")
def list_grant_users(
    query: str | None = Query(default=None, max_length=120),
    limit: int = Query(default=25, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    _admin: dict = Depends(require_admin),
):
    identities = users_module.user_store.list_users_admin(
        limit=limit, offset=offset, query=query
    )
    user_ids = [int(user["id"]) for user in identities]
    projections = credits_service.get_balance_projections(user_ids)
    users = []
    state_reader = getattr(credits_service.store, "get_account_billing_state", None)
    for identity in identities:
        user_id = int(identity["id"])
        balance = projections[user_id].model_dump()
        account = (
            state_reader(user_id)
            if callable(state_reader)
            else credits_service.get_balance(user_id)
        )
        balance.update(
            account_status=(
                account.get("account_status", "active")
                if isinstance(account, dict)
                else account.account_status
            ),
            restriction_reason=(
                account.get("restriction_reason")
                if isinstance(account, dict)
                else account.restriction_reason
            ),
            outstanding_credits_micro=(
                int(account.get("outstanding_credits_micro", 0) or 0)
                if isinstance(account, dict)
                else account.outstanding_credits_micro
            ),
        )
        users.append(
            {
                "id": user_id,
                "email": identity["email"],
                "display_name": identity["display_name"],
                "role": identity["role"],
                "balance": balance,
            }
        )
    return {
        "users": users,
        "total": users_module.user_store.count_users(query=query),
        "limit": limit,
        "offset": offset,
    }


@router.post("/grants/assign")
def assign_grant(
    payload: AssignGrantPayload = Body(...),
    admin: dict = Depends(require_admin),
):
    admin_id = int(admin["id"])
    _rate_limit(admin_id)
    request = AssignGrantRequest.model_validate(payload.model_dump(exclude={"user_id"}))
    try:
        result = credits_service.assign_grant(
            admin_id=admin_id, user_id=payload.user_id, request=request
        )
    except Exception as exc:
        _raise_grant_http_error(exc)
    return {"grant": result.model_dump()}


@router.post("/grants/reclaim")
def reclaim_grant(
    payload: ReclaimGrantPayload = Body(...),
    admin: dict = Depends(require_admin),
):
    admin_id = int(admin["id"])
    _rate_limit(admin_id)
    request = ReclaimGrantRequest.model_validate(
        payload.model_dump(exclude={"user_id"})
    )
    try:
        result = credits_service.reclaim_grant(
            admin_id=admin_id, user_id=payload.user_id, request=request
        )
    except Exception as exc:
        _raise_grant_http_error(exc)
    return {"grant": result.model_dump()}


@router.get("/activity")
def get_grant_activity(
    pool_id: str = Query(default="default", min_length=1, max_length=120),
    limit: int = Query(default=50, ge=1, le=100),
    cursor: int | None = Query(default=None, ge=1),
    _admin: dict = Depends(require_admin),
):
    return get_grant_pool_activity(pool_id=pool_id, limit=limit, cursor=cursor)
