"""Admin account management: list users, set roles / entitlements.

Mounted under ``/api/admin``. Separate from the legacy root-level
``/admin/runs/{run_id}`` debug delete route — that path stays where it is for
external callers; this surface is the product admin console.
"""

import os
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from dashboard.backend import users as users_module
from dashboard.backend.api.auth import get_current_user, require_admin
from dashboard.backend.api.rate_limit import (
    FixedWindowRateLimiter,
    client_ip,
    rate_limited_error,
)
from dashboard.backend.session_tokens import secrets_equal
from dashboard.backend.users import (
    MAX_CONCURRENT_BACKTESTS_CAP,
    MAX_CREDITS_CAP,
)

router = APIRouter(prefix="/admin", tags=["admin"])

# Every route on this sub-router is admin-only, enforced once here rather than
# per handler. The per-route ``Depends(require_admin)`` below stays — a handler
# that needs the admin's own row still has to name it, and FastAPI caches the
# dependency so it resolves once per request either way. What this adds is that
# a route added later *without* naming it is still gated: the guard was
# previously per-route opt-in spelled ``_admin:``, a leading-underscore
# parameter that reads as deletable to anyone tidying unused arguments.
#
# POST /bootstrap is deliberately NOT on this router: it is the one route that
# must work when no admin exists yet.
admin_router = APIRouter(dependencies=[Depends(require_admin)])

# Failed bootstrap guesses. Success does not consume a slot.
#
# Three budgets, in honesty order:
#   per-user   keyed on the authenticated caller's id, which is the one key
#              here that is NOT attacker-chosen — the route requires a session,
#              so spending someone else's budget means owning their account.
#              Signup is open, so fresh accounts are cheap but not free (the
#              signup limiters bound minting them), which makes this the budget
#              that actually costs a guesser something.
#   per-IP     bounds naive abuse only. It deliberately keys on client_ip(),
#              not client_key(): the x-browser-id/x-session-id headers
#              client_key() prefers cost nothing to rotate per request, which
#              made that budget a no-op, where forging a fresh X-Forwarded-For
#              at least takes deliberate header surgery.
#   global     the only budget a header-rotating caller cannot dodge — and for
#              that same reason the only one they can turn on the operator. It
#              is checked AFTER the secret compare (see bootstrap_admin), so
#              exhausting it sheds wrong guesses without ever refusing a right
#              one. Guessing is bounded by the secret's entropy, which
#              _BOOTSTRAP_MIN_LENGTH now makes mandatory rather than
#              hoped-for; per process, see #349.
_BOOTSTRAP_LIMITER = FixedWindowRateLimiter(max_events=5, window_seconds=900)
_BOOTSTRAP_GLOBAL_LIMITER = FixedWindowRateLimiter(max_events=20, window_seconds=900)
_BOOTSTRAP_GLOBAL_KEY = "bootstrap:global"
_BOOTSTRAP_RATE_DETAIL = "Too many bootstrap attempts; please try again later."
# Slots the first admin gets seeded with, so the operator can actually run
# something without editing their own row first.
BOOTSTRAP_ADMIN_MIN_CONCURRENT_BACKTESTS = 5
# Shortest ADMIN_BOOTSTRAP_SECRET this route will honour. A shared secret that
# promotes its bearer to admin is a password with no account behind it and no
# lockout to hide behind, so "the operator picked something strong" cannot be
# an assumption — anything shorter is refused as if unset, loudly, server-side.
# 32 is what ``secrets.token_urlsafe(24)`` produces and comfortably below the
# 43 chars the deployed value already has, so this tightens the floor without
# stranding the live deployment.
#
# Named ``..._MIN_LENGTH``, not ``..._SECRET_LEN``: CodeQL's
# py/clear-text-logging-sensitive-data classifies a source by the *name* of the
# thing flowing to the sink, so a plain int constant whose name contains SECRET
# made the length-only warning below read as "logs a secret in clear text"
# (alert #1249, source pinned at the literal 32 itself). The value never was
# sensitive; the name was.
_BOOTSTRAP_MIN_LENGTH = 32
# One refusal for every reason bootstrap can decline a secret. Telling "not
# configured" (503) apart from "wrong" (403) hands an unauthenticated prober a
# free answer to "is this deployment bootstrappable?" — the same question the
# repo already declines to answer for LEADERBOARD_DAILY_REFRESH_SECRET, which
# 401s whether or not it is armed and sends the operator's signal to the log.
_BOOTSTRAP_REFUSAL = "Invalid bootstrap secret"


class AdminUserPatch(BaseModel):
    role: Optional[Literal["user", "admin"]] = None
    # ge=0, not ge=1: a floor equal to the default quota is not a control. An
    # admin watching a fresh account burn LLM budget could only lower it to the
    # value that account already had; 0 is "suspended" and needs no new field
    # or column to mean it (check_owner_active_run_cap refuses at
    # active >= limit, so a zero budget refuses the first run).
    max_concurrent_backtests: Optional[int] = Field(
        default=None, ge=0, le=MAX_CONCURRENT_BACKTESTS_CAP
    )
    credits: Optional[int] = Field(default=None, ge=0, le=MAX_CREDITS_CAP)


class AdminBootstrapRequest(BaseModel):
    """Promote the caller to admin when the shared bootstrap secret matches.

    One-shot on a fresh deploy (or local box) so the first operator does not
    need raw SQL. Refuses when ``ADMIN_BOOTSTRAP_SECRET`` is unset, and refuses
    once any admin account already exists.
    """

    secret: str = Field(min_length=8, max_length=256)


def _bootstrap_key(user_id: int) -> str:
    return f"bootstrap:user:{int(user_id)}"


def _bootstrap_client_key(request: Request) -> str:
    # client_ip, not client_key: see the budget notes on _BOOTSTRAP_LIMITER.
    return f"bootstrap:client:{client_ip(request)}"


def reset_bootstrap_limiters() -> None:
    """Clear every bootstrap budget. Test helper — no route calls this."""
    _BOOTSTRAP_LIMITER.reset()
    _BOOTSTRAP_GLOBAL_LIMITER.reset()


def _bootstrap_secret() -> Optional[str]:
    """The configured bootstrap secret, or ``None`` when the route must refuse.

    Unset and too-weak collapse to the same answer on purpose: from the
    caller's side both are "this deployment will not bootstrap", and the route
    above turns both into the same 403. The distinction the *operator* needs
    goes to the log, where reading it already implies server access.

    print, not logging: logger output is invisible under deployed uvicorn.
    """
    expected = (os.getenv("ADMIN_BOOTSTRAP_SECRET") or "").strip()
    if not expected:
        return None
    if len(expected) < _BOOTSTRAP_MIN_LENGTH:
        # Length only — never the value, and never a slice of it.
        print(
            "admin bootstrap: ADMIN_BOOTSTRAP_SECRET is shorter than "
            f"{_BOOTSTRAP_MIN_LENGTH} characters; refusing every attempt. "
            "Set a value from `python -c \"import secrets;"
            'print(secrets.token_urlsafe(32))"`.'
        )
        return None
    return expected


def _audit(event: str, **fields: object) -> None:
    """One line per privileged mutation, on the operator's only real channel.

    Role and quota changes are bare UPDATEs in both twins: no actor column, no
    timestamp, no history. That is a real gap for a privilege system, and a
    proper audit table is a schema change this PR should not grow — but "who
    promoted whom" being *nowhere* is worse than it being in the log, so this
    at least leaves a trail an operator can grep.

    Values are ints and role literals only. Nothing here interpolates an
    email, display name, or store exception: those are attacker-influenced
    strings, and the log-injection guard in ``api/auth.py::_email_domain``
    exists because one of them already reached a print sink.
    """
    parts = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
    print(f"admin.{event} {parts}".rstrip())


@admin_router.get("/stats")
def admin_stats(_admin: dict = Depends(require_admin)):
    """Site-wide counters for the admin console header."""
    from dashboard.backend.api.routers.backtests import count_active_dashboard_backtests
    from dashboard.backend.domain.agents.repository import agent_store
    from dashboard.backend.domain.entitlements import credits

    counts = users_module.user_store.count_users_and_admins()
    return {
        "users": counts["users"],
        "admins": counts["admins"],
        "agents": agent_store.count_agents(),
        "active_dashboard_backtests": count_active_dashboard_backtests(),
        # Whether the Credits column an admin is about to edit does anything.
        # Shipped from the server rather than assumed by the console: metering
        # is an env var on the backend, so the frontend has no other way to
        # tell an enforced balance from a stored number, and an admin setting a
        # quota that silently binds nothing is exactly what issue #351 was.
        "credits_metering_enabled": credits.metering_enabled(),
        "default_credits": users_module.DEFAULT_CREDITS,
    }


@admin_router.get("/users")
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


@admin_router.get("/users/{user_id}")
def get_user(user_id: int, _admin: dict = Depends(require_admin)):
    payload = users_module.user_store.get_user_admin(user_id)
    if not payload:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user": payload}


@admin_router.patch("/users/{user_id}")
def patch_user(
    user_id: int,
    payload: AdminUserPatch,
    admin: dict = Depends(require_admin),
):
    # Explicit JSON null is not "leave unchanged": every field here defaults to
    # None, so ``{"credits": null}`` was indistinguishable from omitting the key
    # and PATCHed nothing behind a 200 — a silent no-op at the API contract.
    # model_fields_set tells the two apart; reject the null outright.
    explicit_nulls = sorted(
        name
        for name in payload.model_fields_set
        if getattr(payload, name) is None
    )
    if explicit_nulls:
        raise HTTPException(
            status_code=400,
            detail=f"Fields cannot be null: {', '.join(explicit_nulls)}",
        )
    if (
        payload.role is None
        and payload.max_concurrent_backtests is None
        and payload.credits is None
    ):
        raise HTTPException(status_code=400, detail="No fields to update")

    # No VALID_ROLES re-check here: the model's Literal["user", "admin"] IS the
    # route's validation (anything else 422s before this body runs), and the
    # store re-validates for its non-HTTP callers.
    if payload.role is not None:
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
        # invalid_role / invalid_* range errors are unreachable from this
        # route — the Pydantic model's Literal and Field bounds reject those
        # requests as 422 before the store runs — so they re-raise as the
        # 500 an impossible state deserves rather than masquerading as
        # handled input validation.
        raise

    if not updated:
        raise HTTPException(status_code=404, detail="User not found")
    _audit(
        "user_patched",
        actor=int(admin["id"]),
        target=int(user_id),
        role=payload.role,
        max_concurrent_backtests=payload.max_concurrent_backtests,
        credits=payload.credits,
    )
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
            raise rate_limited_error(_BOOTSTRAP_LIMITER, key, _BOOTSTRAP_RATE_DETAIL)

    # The compare runs before the global budget is consulted, so a *correct*
    # secret is never refused by other people's wrong guesses. That ordering is
    # the whole point: the global budget is the one key an attacker cannot
    # dodge, which also made it the one they could aim at the operator —
    # 20 wrong guesses per window, re-spent every window, previously locked the
    # real operator out for as long as anyone cared to keep going, and the
    # window it blocks is exactly the fresh-deploy window bootstrap exists for.
    # Nothing is leaked by comparing first: secrets_equal is constant-time, and
    # a wrong guess still leaves with 403/429 either way.
    expected = _bootstrap_secret()
    if expected is None or not secrets_equal(payload.secret, expected):
        for key in keys:
            _BOOTSTRAP_LIMITER.record(key)
        # allow(), not record()-then-check(): the budget is spent and tested in
        # one atomic step, and a rejected attempt does not extend the window.
        if not _BOOTSTRAP_GLOBAL_LIMITER.allow(_BOOTSTRAP_GLOBAL_KEY):
            raise rate_limited_error(
                _BOOTSTRAP_GLOBAL_LIMITER, _BOOTSTRAP_GLOBAL_KEY, _BOOTSTRAP_RATE_DETAIL
            )
        _audit("bootstrap_rejected", user=int(current_user["id"]))
        raise HTTPException(status_code=403, detail=_BOOTSTRAP_REFUSAL)

    try:
        users_module.user_store.promote_first_admin(current_user["id"])
        _audit("bootstrap_promoted", user=int(current_user["id"]))
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
    except Exception:  # noqa: BLE001 - promotion already succeeded
        # No ``{exc!r}``: a psycopg error stringifies with its connection DSN
        # embedded, and this line lands in the same log an operator pastes into
        # a ticket. The failing store already reports its own details; all this
        # line has to carry is which user needs a quota PATCH.
        _audit("bootstrap_entitlements_failed", user=int(current_user["id"]))
    return {"user": users_module.user_store.get_user_admin(current_user["id"])}


# Mounted last so the module reads top-down: the admin-gated routes are defined
# above, then attached under the same ``/admin`` prefix as /bootstrap.
router.include_router(admin_router)
