"""Admin account management: list users, set roles / entitlements.

Mounted under ``/api/admin``. Separate from the legacy root-level
``/admin/runs/{run_id}`` debug delete route — that path stays where it is for
external callers; this surface is the product admin console.
"""

import hashlib
import os
import secrets
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from dashboard.backend import users as users_module
from dashboard.backend.api.auth import get_current_user
from dashboard.backend.api.rate_limit import FixedWindowRateLimiter, client_key
from dashboard.backend.users import (
    MAX_CONCURRENT_BACKTESTS_CAP,
    MAX_CREDITS_CAP,
    VALID_ROLES,
)

router = APIRouter(prefix="/admin", tags=["admin"])

# Failed bootstrap guesses. Success does not consume a slot.
#
# Counting per signed-in user alone was not a bound at all: signup is open, so
# an attacker who exhausts one account's five guesses just registers another
# and gets a fresh budget, for as many accounts as they care to create. The
# per-user counter stays (it is the friendliest 429 for an operator with a
# typo), but the two that actually cap guessing are the per-client key and the
# server-wide one below.
_BOOTSTRAP_LIMITER = FixedWindowRateLimiter(max_events=5, window_seconds=900)
# Server-wide ceiling across every account and address. Bootstrap is a one-shot
# action a single operator performs once on a fresh deploy, so 20 wrong guesses
# per 15 minutes is far more than legitimate use needs. It can be used to lock
# the operator out for a window, which is the right trade: the route is inert
# the moment one admin exists, and SQL is always available as break-glass.
_BOOTSTRAP_GLOBAL_LIMITER = FixedWindowRateLimiter(max_events=20, window_seconds=900)
_BOOTSTRAP_GLOBAL_KEY = "bootstrap:global"
_BOOTSTRAP_RATE_DETAIL = "Too many bootstrap attempts; please try again later."
# Slots the first admin gets seeded with, so the operator can actually run
# something without editing their own row first.
BOOTSTRAP_ADMIN_MIN_CONCURRENT_BACKTESTS = 5


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
    """Constant-time compare that cannot 500 on a hostile guess.

    ``secrets.compare_digest`` does **not** raise on a length mismatch —
    buffers of different lengths simply compare unequal — but it does raise
    ``TypeError`` when either side is a ``str`` holding a non-ASCII character,
    and a JSON body can contain any character the caller likes. It also runs in
    time proportional to the shorter operand, so comparing raw secrets leaks
    the expected length. Hashing both sides to a fixed 32 bytes removes both:
    every comparison is over the same length and over pure bytes. A SHA-256
    collision is not a practical oracle here.
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


def _bootstrap_client_key(request: Request) -> str:
    return f"bootstrap:client:{client_key(request)}"


def _bootstrap_rate_limited(
    limiter: FixedWindowRateLimiter, key: str
) -> HTTPException:
    return HTTPException(
        status_code=429,
        detail=_BOOTSTRAP_RATE_DETAIL,
        headers={"Retry-After": str(limiter.retry_after_seconds(key))},
    )


def reset_bootstrap_limiters() -> None:
    """Clear every bootstrap budget. Test helper — no route calls this."""
    _BOOTSTRAP_LIMITER.reset()
    _BOOTSTRAP_GLOBAL_LIMITER.reset()


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
    # ``total`` is what makes the page window legible: without it the console
    # cannot tell "these are all the users" from "these are the first 100 of
    # 400", and the rest of the list is simply invisible.
    return {
        "users": users_module.user_store.list_users_admin(limit=limit, offset=offset),
        "total": users_module.user_store.count_users(),
        "limit": limit,
        "offset": offset,
    }


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

    # One store call, one transaction. Applying the role and the entitlements
    # as two separate writes meant a failure on the second left the first
    # committed behind a 500, so the console kept showing a row the database no
    # longer agreed with.
    try:
        updated = users_module.user_store.apply_admin_patch(
            user_id,
            role=payload.role,
            max_concurrent_backtests=payload.max_concurrent_backtests,
            credits=payload.credits,
            updated_by_admin_id=admin["id"],
        )
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
        if code == "invalid_max_concurrent_backtests":
            raise HTTPException(
                status_code=400, detail="Invalid max_concurrent_backtests"
            ) from exc
        if code == "invalid_credits":
            raise HTTPException(status_code=400, detail="Invalid credits") from exc
        raise

    if not updated:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user": updated}


@router.post("/bootstrap")
def bootstrap_admin(
    request: Request,
    payload: AdminBootstrapRequest,
    current_user: dict = Depends(get_current_user),
):
    """Promote the signed-in caller to admin using ``ADMIN_BOOTSTRAP_SECRET``.

    One-shot: refuses once any admin exists. Break-glass after that is SQL.
    """
    keys = (_bootstrap_key(current_user["id"]), _bootstrap_client_key(request))
    for key in keys:
        if not _BOOTSTRAP_LIMITER.check(key):
            raise _bootstrap_rate_limited(_BOOTSTRAP_LIMITER, key)
    if not _BOOTSTRAP_GLOBAL_LIMITER.check(_BOOTSTRAP_GLOBAL_KEY):
        raise _bootstrap_rate_limited(_BOOTSTRAP_GLOBAL_LIMITER, _BOOTSTRAP_GLOBAL_KEY)

    expected = (os.getenv("ADMIN_BOOTSTRAP_SECRET") or "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Admin bootstrap is not configured",
        )
    if not secrets_equal(payload.secret, expected):
        for key in keys:
            _BOOTSTRAP_LIMITER.record(key)
        _BOOTSTRAP_GLOBAL_LIMITER.record(_BOOTSTRAP_GLOBAL_KEY)
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

    # First admin gets a usable concurrent slot budget out of the box. This is
    # a convenience, not part of the promotion: the role change above is
    # already committed and bootstrap is one-shot, so letting an error here
    # escape as a 500 would tell the operator it failed while leaving them an
    # admin whose retry now 403s on ``admin_exists``. Report and carry on --
    # the quota is one PATCH away in the console they can now open.
    try:
        users_module.user_store.set_entitlements(
            current_user["id"],
            max_concurrent_backtests=max(
                users_module.user_store.get_entitlements(current_user["id"])[
                    "max_concurrent_backtests"
                ],
                BOOTSTRAP_ADMIN_MIN_CONCURRENT_BACKTESTS,
            ),
            updated_by_admin_id=current_user["id"],
        )
    except Exception as exc:  # noqa: BLE001 - promotion already succeeded
        # print, not logging: logger output is invisible under deployed uvicorn.
        print(
            "admin bootstrap: promoted user "
            f"{current_user['id']} but could not seed entitlements: {exc!r}"
        )
    return {"user": users_module.user_store.get_user_admin(current_user["id"])}
