"""Admin account management: list users, set roles / entitlements.

Mounted under ``/api/admin``. Separate from the legacy root-level
``/admin/runs/{run_id}`` debug delete route — that path stays where it is for
external callers; this surface is the product admin console.
"""

import hashlib
import os
import secrets
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from dashboard.backend import users as users_module
from dashboard.backend.api.auth import get_current_user
from dashboard.backend.api.rate_limit import FixedWindowRateLimiter
from dashboard.backend.users import (
    MAX_CONCURRENT_BACKTESTS_CAP,
    MAX_CREDITS_CAP,
    VALID_ROLES,
)

router = APIRouter(prefix="/admin", tags=["admin"])

# Failed bootstrap guesses per signed-in user. Success does not consume a slot.
# 5 / 15 min matches AUTH_LOGIN_EMAIL_MAX — enough for a typo, not a brute force.
_BOOTSTRAP_LIMITER = FixedWindowRateLimiter(max_events=5, window_seconds=900)
_BOOTSTRAP_RATE_DETAIL = "Too many bootstrap attempts; please try again later."


def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return current_user


class AdminUserPatch(BaseModel):
    role: Optional[Literal["user", "admin"]] = None
    max_concurrent_backtests: Optional[int] = Field(
        default=None, ge=1, le=MAX_CONCURRENT_BACKTESTS_CAP
    )
    credits: Optional[int] = Field(default=None, ge=0, le=MAX_CREDITS_CAP)


class AdminBootstrapRequest(BaseModel):
    """Promote the caller to admin when the shared bootstrap secret matches.

    One-shot on a fresh deploy (or local box) so the first operator does not
    need raw SQL. Refuses when ``ADMIN_BOOTSTRAP_SECRET`` is unset, and refuses
    once any admin account already exists.
    """

    secret: str = Field(min_length=8, max_length=256)


def secrets_equal(provided: str, expected: str) -> bool:
    """Constant-time compare that never 500s on length mismatch.

    ``hmac.compare_digest`` / ``secrets.compare_digest`` raise ValueError when
    the two buffers differ in length, which would turn a wrong guess into a 500
    and leak that the secret is a different length. Hash both sides to a fixed
    size first; a SHA-256 collision is not a practical oracle here.
    """
    try:
        left = hashlib.sha256(
            (provided or "").encode("utf-8", "surrogateescape")
        ).digest()
        right = hashlib.sha256(
            (expected or "").encode("utf-8", "surrogateescape")
        ).digest()
    except Exception:
        return False
    return secrets.compare_digest(left, right)


def _bootstrap_key(user_id: int) -> str:
    return f"bootstrap:user:{int(user_id)}"


@router.get("/stats")
def admin_stats(_admin: dict = Depends(require_admin)):
    """Site-wide counters for the admin console header."""
    from dashboard.backend.api.routers.backtests import count_active_dashboard_backtests
    from dashboard.backend.domain.agents.repository import agent_store

    return {
        "users": users_module.user_store.count_users(),
        "admins": users_module.user_store.count_admins(),
        "agents": agent_store.count_agents(),
        "active_dashboard_backtests": count_active_dashboard_backtests(),
    }


@router.get("/users")
def list_users(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    _admin: dict = Depends(require_admin),
):
    return {"users": users_module.user_store.list_users_admin(limit=limit, offset=offset)}


@router.get("/users/{user_id}")
def get_user(user_id: int, _admin: dict = Depends(require_admin)):
    payload = users_module.user_store.get_user_admin(user_id)
    if not payload:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user": payload}


@router.patch("/users/{user_id}")
def patch_user(
    user_id: int,
    payload: AdminUserPatch,
    admin: dict = Depends(require_admin),
):
    if (
        payload.role is None
        and payload.max_concurrent_backtests is None
        and payload.credits is None
    ):
        raise HTTPException(status_code=400, detail="No fields to update")

    if payload.role is not None:
        if payload.role not in VALID_ROLES:
            raise HTTPException(status_code=400, detail="Invalid role")
        # Self-demotion is a lockout footgun: once you drop your own admin bit
        # you cannot open this page to undo it. Another admin (or SQL) must
        # demote you. Last-admin is a separate store-level guard.
        if payload.role != "admin" and int(user_id) == int(admin["id"]):
            raise HTTPException(
                status_code=400,
                detail="Cannot demote yourself; ask another admin",
            )
        try:
            users_module.user_store.set_user_role(user_id, payload.role)
        except ValueError as exc:
            code = str(exc)
            if code == "user_not_found":
                raise HTTPException(status_code=404, detail="User not found") from exc
            if code == "last_admin":
                raise HTTPException(
                    status_code=400,
                    detail="Cannot demote the last admin account",
                ) from exc
            if code == "invalid_role":
                raise HTTPException(status_code=400, detail="Invalid role") from exc
            raise

    if payload.max_concurrent_backtests is not None or payload.credits is not None:
        try:
            users_module.user_store.set_entitlements(
                user_id,
                max_concurrent_backtests=payload.max_concurrent_backtests,
                credits=payload.credits,
                updated_by_admin_id=admin["id"],
            )
        except ValueError as exc:
            code = str(exc)
            if code == "user_not_found":
                raise HTTPException(status_code=404, detail="User not found") from exc
            if code == "invalid_max_concurrent_backtests":
                raise HTTPException(
                    status_code=400, detail="Invalid max_concurrent_backtests"
                ) from exc
            if code == "invalid_credits":
                raise HTTPException(status_code=400, detail="Invalid credits") from exc
            raise

    updated = users_module.user_store.get_user_admin(user_id)
    if not updated:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user": updated}


@router.post("/bootstrap")
def bootstrap_admin(
    payload: AdminBootstrapRequest,
    current_user: dict = Depends(get_current_user),
):
    """Promote the signed-in caller to admin using ``ADMIN_BOOTSTRAP_SECRET``.

    One-shot: refuses once any admin exists. Break-glass after that is SQL.
    """
    key = _bootstrap_key(current_user["id"])
    if not _BOOTSTRAP_LIMITER.check(key):
        raise HTTPException(
            status_code=429,
            detail=_BOOTSTRAP_RATE_DETAIL,
            headers={"Retry-After": str(_BOOTSTRAP_LIMITER.retry_after_seconds(key))},
        )

    expected = (os.getenv("ADMIN_BOOTSTRAP_SECRET") or "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Admin bootstrap is not configured",
        )
    if not secrets_equal(payload.secret, expected):
        _BOOTSTRAP_LIMITER.record(key)
        raise HTTPException(status_code=403, detail="Invalid bootstrap secret")

    try:
        users_module.user_store.promote_first_admin(current_user["id"])
    except ValueError as exc:
        code = str(exc)
        if code == "admin_exists":
            raise HTTPException(
                status_code=403,
                detail="Bootstrap is only available when no admin exists",
            ) from exc
        if code == "user_not_found":
            raise HTTPException(status_code=404, detail="User not found") from exc
        raise

    # First admin gets a usable concurrent slot budget out of the box.
    users_module.user_store.set_entitlements(
        current_user["id"],
        max_concurrent_backtests=max(
            users_module.user_store.get_entitlements(current_user["id"])[
                "max_concurrent_backtests"
            ],
            5,
        ),
        updated_by_admin_id=current_user["id"],
    )
    return {"user": users_module.user_store.get_user_admin(current_user["id"])}
