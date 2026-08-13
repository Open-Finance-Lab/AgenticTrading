"""Admin account management: list users, set roles / entitlements.

Mounted under ``/api/admin``. Separate from the legacy root-level
``/admin/runs/{run_id}`` debug delete route — that path stays where it is for
external callers; this surface is the product admin console.
"""

import os
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from dashboard.backend.api.auth import get_current_user
from dashboard.backend.users import (
    MAX_CONCURRENT_BACKTESTS_CAP,
    MAX_CREDITS_CAP,
    VALID_ROLES,
    user_store,
)

router = APIRouter(prefix="/admin", tags=["admin"])


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

    Used once on a fresh deploy (or local box) so the first operator does not
    need raw SQL. Refuses when ``ADMIN_BOOTSTRAP_SECRET`` is unset.
    """

    secret: str = Field(min_length=8, max_length=256)


@router.get("/stats")
def admin_stats(_admin: dict = Depends(require_admin)):
    """Site-wide counters for the admin console header."""
    from dashboard.backend.api.routers import backtests as backtests_mod
    from dashboard.backend.domain.agents.repository import agent_store

    # Compatible with single-flight ``backtest_status`` and multi-slot ledger.
    slots = getattr(backtests_mod, "_active_slots", None)
    lock = getattr(backtests_mod, "_backtest_slots_lock", None)
    if isinstance(slots, dict) and lock is not None:
        with lock:
            active_backtests = sum(1 for slot in slots.values() if slot.get("running"))
    else:
        status = getattr(backtests_mod, "backtest_status", None) or {}
        active_backtests = 1 if status.get("running") else 0

    return {
        "users": user_store.count_users(),
        "admins": user_store.count_admins(),
        "agents": agent_store.count_agents(),
        "active_dashboard_backtests": active_backtests,
    }


@router.get("/users")
def list_users(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    _admin: dict = Depends(require_admin),
):
    return {"users": user_store.list_users_admin(limit=limit, offset=offset)}


@router.get("/users/{user_id}")
def get_user(user_id: int, _admin: dict = Depends(require_admin)):
    payload = user_store.get_user_admin(user_id)
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
        # you cannot open this page to undo it. Another admin (or bootstrap /
        # DB) must demote you. Last-admin is a separate store-level guard.
        if payload.role != "admin" and int(user_id) == int(admin["id"]):
            raise HTTPException(
                status_code=400,
                detail="Cannot demote yourself; ask another admin",
            )
        try:
            user_store.set_user_role(user_id, payload.role)
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
            user_store.set_entitlements(
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

    updated = user_store.get_user_admin(user_id)
    if not updated:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user": updated}


@router.post("/bootstrap")
def bootstrap_admin(
    payload: AdminBootstrapRequest,
    current_user: dict = Depends(get_current_user),
):
    """Promote the signed-in caller to admin using ``ADMIN_BOOTSTRAP_SECRET``."""
    expected = (os.getenv("ADMIN_BOOTSTRAP_SECRET") or "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Admin bootstrap is not configured",
        )
    if not secrets_equal(payload.secret, expected):
        raise HTTPException(status_code=403, detail="Invalid bootstrap secret")

    user_store.set_user_role(current_user["id"], "admin")
    # First admin gets a usable concurrent slot budget out of the box.
    user_store.set_entitlements(
        current_user["id"],
        max_concurrent_backtests=max(
            user_store.get_entitlements(current_user["id"])["max_concurrent_backtests"],
            5,
        ),
        updated_by_admin_id=current_user["id"],
    )
    return {"user": user_store.get_user_admin(current_user["id"])}


def secrets_equal(provided: str, expected: str) -> bool:
    import hmac

    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))
