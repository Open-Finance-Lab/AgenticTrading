import asyncio
import base64
import hmac
import ipaddress
import logging
import math
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urlencode

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, field_validator

from dashboard.backend.api import discord_oauth
from dashboard.backend.auth_cookies import (
    clear_session_cookie,
    read_session_token,
    set_session_cookie,
)
from dashboard.backend.api.rate_limit import (
    FixedWindowRateLimiter,
    client_ip,
    client_key,
    rate_limited_error,
)
from dashboard.backend.domain.brokers.repository import broker_store
from dashboard.backend.domain.analytics import instrumentation as analytics_instrumentation
from dashboard.backend.domain.agents.service import agent_service
from dashboard.backend.domain.credits.service import credits_service
from dashboard.backend.infrastructure.brokers import pending_links, robinhood_oauth
from dashboard.backend.infrastructure.email import sender as email_sender
from dashboard.backend import users as users_module
from dashboard.backend.users import (
    format_stored_timestamp,
    parse_stored_timestamp,
    public_user,
    verify_password,
)
from dashboard.backend.password_policy import validate_new_password
from dashboard.backend.verification_codes import generate_code, hash_code

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# Generic copy for failed logins — never reveal whether the email is registered.
LOGIN_FAILURE_DETAIL = "Invalid email or password."
RATE_LIMIT_DETAIL = "Too many login attempts; please try again later."
# One constant for every reset-password failure branch (unknown email, no
# active request, wrong code, lost CAS), mirroring LOGIN_FAILURE_DETAIL's
# role: distinct copy per branch would be an account/state oracle.
RESET_FAILURE_DETAIL = "Invalid or expired code."
FORGOT_RATE_LIMIT_DETAIL = "Too many reset requests. Please wait before trying again."
RESET_RATE_LIMIT_DETAIL = "Too many attempts. Please wait before trying again."


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    """Read an int override from the environment, or return ``default``.

    ``minimum=0`` for counts, because 0 is a real value meaning "switch this
    budget off" — the convention ``MAX_ACTIVE_RUNS_GLOBAL`` already uses — so an
    operator can disable a limit from the Render dashboard without a deploy.
    Windows pass ``minimum=1`` (a zero-width window is not a setting anyone
    means).

    Anything unparseable or out of range falls back to the default *and says so
    on stdout*. Silently honouring a typo is how the limit that is running stops
    being the limit everyone believes is running.
    """
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        print(f"WARNING: {name}={raw!r} is not an integer; using default {default}")
        return default
    if value < minimum:
        print(f"WARNING: {name}={value} is below the minimum {minimum}; using default {default}")
        return default
    return value


def _build_limiter(prefix: str, max_default: int, window_default: int) -> FixedWindowRateLimiter:
    """Best-effort in-process limit (see rate_limit.py), tunable via env.

    Reports the effective setting at startup, following the same rule as the
    ``*_DATABASE_URL`` backends: a limit nobody can read is a limit nobody can
    verify, and these are exactly the knobs an operator reaches for while
    something is already going wrong.
    """
    max_events = _env_int(f"{prefix}_MAX", max_default)
    window = _env_int(f"{prefix}_WINDOW_SECONDS", window_default, minimum=1)
    effective = "disabled" if max_events == 0 else f"{max_events} per {window}s"
    print(f"auth rate limit {prefix}: {effective}")
    return FixedWindowRateLimiter(max_events=max_events, window_seconds=float(window))


# Per-client budgets are deliberately loose: they are keyed on headers the
# caller controls, so they bound naive abuse and cap queued bcrypt work rather
# than stopping an attacker, and a tight value costs a shared address (one
# office, one classroom, one NAT) far more than it costs anyone hostile. The
# per-email failure budget is the control that actually holds; see the
# rate_limit module docstring.
_LOGIN_IP_LIMITER = _build_limiter("AUTH_LOGIN_IP", 60, 900)
_LOGIN_EMAIL_LIMITER = _build_limiter("AUTH_LOGIN_EMAIL", 10, 900)
_SIGNUP_IP_LIMITER = _build_limiter("AUTH_SIGNUP_IP", 60, 3600)
_SIGNUP_EMAIL_LIMITER = _build_limiter("AUTH_SIGNUP_EMAIL", 5, 3600)
# forgot-password: both keys are existence-blind (the email key is the *typed*
# normalized address), so a 429 reveals nothing about accounts. The global
# limiter is the shared-Brevo-quota bound: an anonymous caller iterating known
# addresses is the one genuinely new drain surface this flow opens, and
# 10/hour caps the worst case under the provider's free-tier daily allowance.
_FORGOT_IP_LIMITER = _build_limiter("AUTH_FORGOT_IP", 30, 3600)
_FORGOT_EMAIL_LIMITER = _build_limiter("AUTH_FORGOT_EMAIL", 5, 3600)
_FORGOT_GLOBAL_LIMITER = _build_limiter("AUTH_FORGOT_GLOBAL", 10, 3600)
# reset-password mirrors login's check/record/allow split: the per-email
# budget is the one keyed on the account under attack, so it is the control
# that actually bounds code guessing.
_RESET_IP_LIMITER = _build_limiter("AUTH_RESET_IP", 30, 900)
_RESET_EMAIL_LIMITER = _build_limiter("AUTH_RESET_EMAIL", 10, 900)


# Leading run of characters a hostname may contain. Anchored and non-greedy
# about nothing: it stops at the first byte a domain cannot hold, so injected
# text is cut off rather than folded into the value.
_DOMAIN_PREFIX = re.compile(r"[a-z0-9.\-]*")


def _email_domain(email: str) -> str:
    """Domain only: enough to spot a scripted sweep, never the address itself.

    Filtered, not just split. ``_normalize_email`` strips the *ends* of the
    address, so ``"victim@example.com\\nauth.login_failed domain=attacker.test"``
    validates -- it has an ``@`` and a dotted right-hand side -- and would let an
    unauthenticated caller write whole forged lines into the log an operator
    reads while deciding whether they are under attack. CodeQL caught this as
    py/log-injection while these were ``logger`` calls and stops reporting it at
    a ``print`` sink it does not model, so the guard has to be the code, not the
    alert. Truncating at the first character a domain cannot hold, rather than
    filtering them out: dropping them would splice the injected text onto the
    real domain instead of discarding it, and nothing downstream needs to
    reverse this.
    """
    domain = email.rsplit("@", 1)[-1].lower()
    return _DOMAIN_PREFIX.match(domain).group()[:64] or "?"


# The 429 builder now lives in rate_limit.rate_limited_error, shared with every
# other limited route; this name stays for the module's existing call sites.
_auth_rate_limited = rate_limited_error


def _login_ip_key(request: Request) -> str:
    return f"login:{client_key(request)}"


def _signup_ip_key(request: Request) -> str:
    return f"signup:{client_key(request)}"


async def _ensure_welcome_credits(user_id: int, *, category: str) -> None:
    """Best-effort grant; login and startup provide deterministic repair."""
    try:
        await asyncio.to_thread(credits_service.grant_default_signup_credits, user_id)
    except Exception:  # noqa: BLE001 - never expose a store/DSN error here
        print(f"WARNING: credits.welcome_grant_failed category={category}")


def _app_redirect(query: dict[str, str]) -> RedirectResponse:
    """Send the browser back to the dashboard after Discord OAuth."""
    base = (os.getenv("PUBLIC_APP_URL") or "").rstrip("/")
    if base:
        if not base.endswith("/app"):
            base = f"{base}/app"
    else:
        base = "/app"
    return RedirectResponse(url=f"{base}?{urlencode(query)}", status_code=302)


def _normalize_email(value: str) -> str:
    email = value.strip().lower()
    # Strip() only touches the ends, so an address whose *interior* holds a
    # newline validated fine and was stored verbatim. Everything that later
    # renders an email as plain text then inherits it: the admin console builds
    # its role-change confirm() as
    # ``Promote {email} to admin?\n\nThey will see Admin…``, and an address of
    # the form ``a@b.com\n\nThis account is verified by SecureFinAI Lab.`` puts
    # attacker-chosen lines into the dialog an admin reads while deciding
    # whether to grant admin. Escaping cannot help there — a native dialog has
    # no markup to escape — so the address must never contain the character.
    # Same reasoning covers the log sinks; ``_email_domain`` above truncates
    # for its own output, but that is a second guard, not this one.
    if any(ch.isspace() or ord(ch) < 0x20 or ord(ch) == 0x7F for ch in email):
        raise ValueError("invalid email address")
    if "@" not in email or "." not in email.split("@", 1)[-1]:
        raise ValueError("invalid email address")
    return email


class SignupRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    display_name: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        return _normalize_email(value)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        return _normalize_email(value)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=1, max_length=128)


class DisplayNameRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=100)


class EmailChangeRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_email: str = Field(min_length=3, max_length=254)

    @field_validator("new_email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        return _normalize_email(value)


class EmailChangeVerifyRequest(BaseModel):
    code: str = Field(min_length=1, max_length=32)


class ForgotPasswordRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        # A malformed address 422s here on both reset routes. Deliberate,
        # accepted exception to response uniformity: the 422 is shape-keyed,
        # not account-keyed -- a malformed address cannot have an account.
        return _normalize_email(value)


class ResetPasswordRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    code: str = Field(min_length=1, max_length=16)
    new_password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        return _normalize_email(value)


AVATAR_MAX_DECODED_BYTES = 100 * 1024

# Declared mime -> required leading bytes. WebP is RIFF-framed, checked separately.
_AVATAR_MAGIC = {
    "image/jpeg": b"\xff\xd8\xff",
    "image/png": b"\x89PNG\r\n\x1a\n",
}


class AvatarRequest(BaseModel):
    avatar: str = Field(min_length=1, max_length=200_000)


def _validate_avatar_data_uri(value: str) -> str:
    """Server-side avatar gate: allowlisted mime, valid base64, magic-number
    match, <= 100 KB decoded. Never trust the client's canvas pipeline."""
    mime = None
    payload = None
    for candidate in ("image/jpeg", "image/png", "image/webp"):
        prefix = f"data:{candidate};base64,"
        if value.startswith(prefix):
            mime = candidate
            payload = value[len(prefix):]
            break
    if mime is None:
        raise ValueError("Avatar must be a base64 data URI (JPEG, PNG, or WebP).")
    try:
        decoded = base64.b64decode(payload, validate=True)
    except ValueError as exc:  # binascii.Error subclasses ValueError
        raise ValueError("Avatar data is not valid base64.") from exc
    if len(decoded) > AVATAR_MAX_DECODED_BYTES:
        raise ValueError("Avatar image must be 100 KB or smaller.")
    if mime == "image/webp":
        ok = len(decoded) >= 12 and decoded[:4] == b"RIFF" and decoded[8:12] == b"WEBP"
    else:
        ok = decoded.startswith(_AVATAR_MAGIC[mime])
    if not ok:
        raise ValueError("Avatar image bytes do not match the declared format.")
    return value


class AuthResponse(BaseModel):
    user: dict


_MAX_STORED_USER_AGENT = 200


def _session_client_context(request: Request) -> dict:
    """What auth_sessions records about the client a session was issued to.

    ``ip_prefix``, deliberately, not the address: this row outlives the request
    and is meant to answer "was that me?" on a signed-in-devices list. Keeping
    the exact address would turn the session table into a location log for
    every user, which is a different product with different obligations. /24
    and /48 are coarse enough to be that, and no coarser.

    Both values come from headers the caller controls, so neither is a security
    control -- see rate_limit's module docstring. They are display data.
    """
    raw_ip = client_ip(request)
    try:
        address = ipaddress.ip_address(raw_ip)
    except ValueError:
        prefix = None
    else:
        prefix = str(
            ipaddress.ip_network(
                f"{address}/{24 if address.version == 4 else 48}", strict=False
            )
        )
    agent = (request.headers.get("user-agent") or "").strip()
    return {
        "user_agent": agent[:_MAX_STORED_USER_AGENT] or None,
        "ip_prefix": prefix,
    }


def _extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token.strip()


def _session_token(
    request: Request,
    authorization: Optional[str] = None,
) -> Optional[str]:
    """Resolve the raw session token for this request.

    Explicit ``Authorization: Bearer`` wins when present (scripts / TestClient);
    browsers after the HttpOnly migration send only the cookie.
    """
    return _extract_bearer_token(authorization) or read_session_token(request)


def get_current_user(
    request: Request,
    authorization: Optional[str] = Header(default=None),
) -> dict:
    token = _session_token(request, authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = users_module.user_store.get_user_for_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return user


def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    """Dependency: the signed-in caller, who must hold the admin role.

    The one admin gate — routers depend on this rather than re-checking
    ``role`` inline, so "who counts as an admin" cannot drift per route.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return current_user


def _issue_session(user: dict, request: Request) -> tuple[str, Optional[dict]]:
    """Mint the session row and read the caller's entitlements — one hop.

    Both of these are synchronous store calls, i.e. a network round trip to
    pooled Postgres in a durable deployment, and the only two callers are the
    ``async def`` signup/login handlers. Bundled into a single function so they
    cross into the threadpool together: two ``asyncio.to_thread`` awaits would
    pay two hops for work that has to happen in sequence anyway, and — worse —
    leaving *either* of them inline puts blocking I/O back on the event loop,
    where one slow query stalls every concurrent request server-wide.

    ``test_event_loop_threadpool`` cannot catch that regression here: it pins
    plain-``def`` handlers, and these two are exempt for already awaiting.
    ``test_auth_async_handlers_offload_store_io`` is the guard that can.
    """
    token = users_module.user_store.create_session(
        user["id"], **_session_client_context(request)
    )
    entitlements = (
        users_module.user_store.get_entitlements(user["id"])
        if user.get("id") is not None
        else None
    )
    return token, entitlements


def _auth_json(
    user: dict, raw_token: str, entitlements: Optional[dict] = None
) -> JSONResponse:
    from dashboard.backend.csrf import set_csrf_cookie

    payload = dict(user)
    if "entitlements" not in payload and payload.get("id") is not None:
        # Prefetched by the caller when it had a threadpool hop to spend; the
        # query below is the fallback for a caller that did not.
        payload["entitlements"] = (
            entitlements
            if entitlements is not None
            else users_module.user_store.get_entitlements(payload["id"])
        )
    response = JSONResponse({"user": payload})
    set_session_cookie(response, raw_token)
    set_csrf_cookie(response)
    return response


@router.post("/signup", response_model=AuthResponse)
async def signup(payload: SignupRequest, request: Request):
    # Password policy first, budgets second. A rejected password creates
    # nothing and costs nothing, so charging for it would let someone spend
    # their whole signup allowance discovering the strength rules and end up
    # rate-limited out of the account they were being careful about.
    violations = validate_new_password(payload.password, payload.email)
    if violations:
        raise HTTPException(status_code=400, detail=" ".join(violations))

    # allow(), not the check/record split login uses: here both outcomes are
    # worth metering. A success spends a bcrypt hash and a durable row; a 409 is
    # an account-existence probe (see below).
    ip_key = _signup_ip_key(request)
    if not _SIGNUP_IP_LIMITER.allow(ip_key):
        raise _auth_rate_limited(
            _SIGNUP_IP_LIMITER,
            ip_key,
            "Too many signup attempts from this network; please try again later.",
        )
    email_key = f"signup:email:{payload.email}"
    if not _SIGNUP_EMAIL_LIMITER.allow(email_key):
        raise _auth_rate_limited(
            _SIGNUP_EMAIL_LIMITER,
            email_key,
            "Too many signup attempts for this email; please try again later.",
        )

    try:
        # Threaded for the same reason as login's authenticate(): create_user
        # hashes with bcrypt (~190 ms), and this route is unauthenticated.
        user = await asyncio.to_thread(
            users_module.user_store.create_user,
            email=payload.email,
            display_name=payload.display_name,
            password=payload.password,
        )
    except ValueError as exc:
        if str(exc) == "email_already_registered":
            # This 409 tells an anonymous caller that an address has an account,
            # which /login deliberately no longer does. Kept anyway, and kept
            # honestly: the alternative that actually closes it is confirm-by-
            # email signup (accept every attempt, reveal nothing, mail the
            # address), which is a product change, not a copy change. Nothing
            # here bounds it either -- one request per address answers the
            # question, and the budgets above are keyed on headers the caller
            # rotates for free. Documented as a known gap rather than papered
            # over, so nobody reads the /login fix as covering both routes.
            print(f"auth.signup_conflict domain={_email_domain(payload.email)}")
            raise HTTPException(status_code=409, detail="Email is already registered") from exc
        raise

    await _ensure_welcome_credits(user["id"], category="signup")

    token, entitlements = await asyncio.to_thread(_issue_session, user, request)
    browser_session = (request.headers.get("x-browser-id") or "").strip() or None
    await asyncio.to_thread(
        agent_service.provision_starter_agents,
        owner_user_id=user["id"],
        owner_browser_session=browser_session,
    )
    analytics_instrumentation.emit_account_event(
        event_name="account_signed_up",
        user_id=user["id"],
        source_record_id=user["id"],
        version=None,
    )
    return _auth_json(user, token, entitlements=entitlements)


@router.post("/login", response_model=AuthResponse)
async def login(payload: LoginRequest, request: Request):
    ip_key = _login_ip_key(request)
    # check(), not allow(): both budgets here meter *failures* only. Charging a
    # correct password would make a busy shared address lock its own users out
    # of an endpoint they are using correctly -- and every landing-page visitor
    # shares one key whenever no forwarded address is available at all, which is
    # what client_ip() in rate_limit.py exists to avoid. The check still runs
    # before authenticate(), so an over-budget caller is refused without
    # spending a bcrypt round on them.
    if not _LOGIN_IP_LIMITER.check(ip_key):
        print("auth.login_rate_limited scope=ip")
        raise _auth_rate_limited(_LOGIN_IP_LIMITER, ip_key, RATE_LIMIT_DETAIL)

    # Off the event loop: authenticate() now runs one bcrypt compare on *both*
    # branches (see users.verify_password_for_account), so an unknown email
    # costs ~190 ms of CPU where it used to cost ~0. Inline, that is a
    # single-threaded stall any unauthenticated caller can drive, and the
    # limiters above cannot close it -- rotating X-Browser-Id buys a fresh
    # budget. Matches how the OAuth calls further down already offload.
    user = await asyncio.to_thread(
        users_module.user_store.authenticate, payload.email, payload.password
    )
    if not user:
        _LOGIN_IP_LIMITER.record(ip_key)
        email_key = f"login:email:{payload.email}"
        # Count only failures against the per-email budget so a successful
        # login is not blocked by an attacker's prior guesses — but repeated
        # failures still earn a temporary cooldown (not a permanent lockout).
        # This is the budget that actually bounds guessing: it is keyed on the
        # account under attack, not on anything the attacker chooses.
        if not _LOGIN_EMAIL_LIMITER.allow(email_key):
            print(f"auth.login_rate_limited scope=email domain={_email_domain(payload.email)}")
            raise _auth_rate_limited(_LOGIN_EMAIL_LIMITER, email_key, RATE_LIMIT_DETAIL)
        print(f"auth.login_failed domain={_email_domain(payload.email)}")
        raise HTTPException(status_code=401, detail=LOGIN_FAILURE_DETAIL)

    await _ensure_welcome_credits(user["id"], category="login")
    token, entitlements = await asyncio.to_thread(_issue_session, user, request)
    analytics_instrumentation.emit_account_event(
        event_name="authenticated_session_started",
        user_id=user["id"],
        source_record_id=user["id"],
    )
    return _auth_json(public_user(user), token, entitlements=entitlements)


@router.get("/me")
def me(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    current_user: dict = Depends(get_current_user),
):
    user_payload = public_user(current_user)
    # /me is on the page-boot critical path, where every query is a round-trip
    # to pooled Postgres. get_user_for_token already LEFT JOINs the
    # entitlements row, so read it off the session row instead of paying a
    # second one. The fallback covers a store whose session query has not been
    # taught the join -- it returns None rather than silently reporting
    # defaults for a user who has real quotas.
    entitlements = users_module.entitlements_from_session_row(
        current_user, current_user["id"]
    )
    if entitlements is None:
        entitlements = users_module.user_store.get_entitlements(current_user["id"])
    user_payload["entitlements"] = entitlements
    response = JSONResponse({"user": user_payload})
    # Migration bridge: a browser signed in before the HttpOnly-cookie change
    # holds a valid session only in localStorage. app.js sends it once as
    # Bearer on the boot /me probe; upgrading it to a cookie here keeps that
    # session alive instead of force-logging every user out on deploy.
    if not read_session_token(request):
        token = _extract_bearer_token(authorization)
        if token:
            set_session_cookie(response, token)
    return response


@router.post("/logout")
def logout(
    request: Request,
    authorization: Optional[str] = Header(default=None),
):
    token = _session_token(request, authorization)
    if token:
        users_module.user_store.delete_session(token)
    response = JSONResponse({"status": "ok"})
    clear_session_cookie(response)
    from dashboard.backend.csrf import clear_csrf_cookie

    clear_csrf_cookie(response)
    return response


@router.post("/logout-all")
def logout_all(request: Request, current_user: dict = Depends(get_current_user)):
    users_module.user_store.delete_other_sessions(current_user["id"], keep_token=None)
    response = JSONResponse({"status": "ok"})
    clear_session_cookie(response)
    from dashboard.backend.csrf import clear_csrf_cookie

    clear_csrf_cookie(response)
    return response


@router.post("/change-password")
def change_password(
    payload: ChangePasswordRequest,
    request: Request,
    current_user: dict = Depends(get_current_user),
    authorization: Optional[str] = Header(default=None),
):
    # Plain def, not async: nothing in this handler awaits, and *everything* in
    # it blocks -- one bcrypt verify (~190 ms), a second bcrypt hash of the same
    # cost inside update_password(), and three store round trips (network calls
    # against Postgres in prod). Starlette runs a sync handler in the threadpool,
    # so the whole lot costs one worker thread instead of freezing the event
    # loop for every concurrent request. This is #292's fix applied to the
    # handler #297 singled out for it; offloading only the verify would have
    # left an identically slow hash on the loop. Pinned by BLOCKING_IO_HANDLERS
    # in tests/test_event_loop_threadpool.py -- adding an await here silently
    # takes it back out of that guard.
    #
    # Policy first, bcrypt second, for the same reason signup() checks in that
    # order: a rejected new password changes nothing, so it should not cost a
    # hash. The trade is that a request wrong on both counts now reports the
    # policy violation instead of "Current password is incorrect"; neither is a
    # secret, since signup enforces the same policy unauthenticated.
    violations = validate_new_password(payload.new_password, current_user["email"])
    if violations:
        raise HTTPException(status_code=400, detail=" ".join(violations))
    if not verify_password(payload.current_password, current_user["password_hash"]):
        raise HTTPException(status_code=400, detail="Current password is incorrect.")
    users_module.user_store.update_password(current_user["id"], payload.new_password)
    # Best-effort: revoke every other session so a stolen token dies with the old
    # password. Deliberately NOT atomic with the update above -- the two are separate
    # transactions/connections in both twin stores. The password change is already
    # durable here; if revocation raises (e.g. a transient Postgres blip on the prod
    # pool), turning it into a 500 would wrongly tell the client the change failed and
    # make a retry hit "Current password is incorrect". So swallow + surface via
    # print() (logger output is invisible under the deployed config) and still
    # return ok. Revocation is defence-in-depth, not a hard guarantee.
    _d7_cleanup(
        "change-password",
        current_user["id"],
        "other-session revocation",
        lambda: users_module.user_store.delete_other_sessions(
            current_user["id"], keep_token=_session_token(request, authorization)
        ),
    )
    # D7: a user changing their password may be reacting to a compromise, so an
    # attacker's in-flight email change dies with it. Best-effort and next to
    # the session revocation above, so the whole "invalidate what the old
    # password could reach" policy sits in one place.
    _d7_cleanup(
        "change-password",
        current_user["id"],
        "cancelling the pending email change",
        lambda: users_module.user_store.cancel_email_change(current_user["id"]),
    )
    # Same D7 symmetry for the reset flow: a code already mailed out must not
    # survive the password it would replace being changed by its real owner.
    _d7_cleanup(
        "change-password",
        current_user["id"],
        "cancelling the pending password reset",
        lambda: users_module.user_store.cancel_password_reset(current_user["id"]),
    )
    return {"status": "ok"}


@router.put("/display-name")
def update_display_name(
    payload: DisplayNameRequest,
    current_user: dict = Depends(get_current_user),
):
    display_name = payload.display_name.strip()
    if not display_name:
        # Field(min_length=1) measures the raw string, so "   " reaches here.
        # Storing it would repeat issue #167 (a whitespace-only name persisted
        # as an empty label with no way to tell it from a missing one).
        raise HTTPException(status_code=400, detail="Display name cannot be empty.")
    # No password required: a display name is not an authentication factor, and
    # gating it behind one is not what any comparable platform does.
    try:
        user = users_module.user_store.update_display_name(
            current_user["id"], display_name
        )
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Session is no longer valid.") from exc
    return {"user": user}


def _seconds_since(timestamp: str) -> float:
    return (
        datetime.now(timezone.utc) - parse_stored_timestamp(timestamp)
    ).total_seconds()


def _too_soon(detail: str, seconds_remaining: float) -> HTTPException:
    """A 429 whose Retry-After is the wait that is actually left.

    Not the window's width: with a seven-day window those differ by up to seven
    days, and a client that honours the header would sit out the whole period
    over a wait that had minutes to run.
    """
    return HTTPException(
        status_code=429,
        detail=detail,
        headers={"Retry-After": str(max(1, math.ceil(seconds_remaining)))},
    )


def _humanize_wait(seconds: float) -> str:
    """Round the remaining wait UP to a whole unit, for user-facing copy.

    Up, so the message never invites a retry that would 429 again: "in 1 day"
    with 4 hours left is a lie the user discovers by being refused twice.
    """
    if seconds >= 86400:
        days = math.ceil(seconds / 86400)
        return f"{days} day{'s' if days != 1 else ''}"
    if seconds >= 3600:
        hours = math.ceil(seconds / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''}"
    minutes = max(1, math.ceil(seconds / 60))
    return f"{minutes} minute{'s' if minutes != 1 else ''}"


def _email_change_old_body(code: str, new_email: str) -> str:
    """Stage 'old', to the CURRENT address: the account owner."""
    return (
        "Someone asked to change the email address on your Agentic Trading Lab "
        f"account to {new_email}.\n\n"
        f"Your confirmation code is: {code}\n\n"
        f"It expires in {users_module.EMAIL_CHANGE_TTL_MINUTES} minutes. If this "
        "was not you, ignore this message and change your password."
    )


def _email_change_new_body(code: str) -> str:
    """Stage 'new', to the address being adopted -- possibly a bystander's.

    Its owner may have no Agentic Trading Lab account, so no "your account"
    and no advice to change a password they do not have: instructions that
    cannot apply to the reader are exactly what phishing looks like.
    """
    return (
        "Someone asked to make this address the contact email for their "
        "Agentic Trading Lab account.\n\n"
        f"Your confirmation code is: {code}\n\n"
        f"It expires in {users_module.EMAIL_CHANGE_TTL_MINUTES} minutes. If "
        "this was you, enter it on the account page to finish. If not, ignore "
        "this message and do not share the code -- without it, nothing gets "
        "linked to this address."
    )


def _authorize_email_change(store, current_user: dict, payload: EmailChangeRequest) -> None:
    """Password check + policy + the three rate limits, or raise.

    Extracted so request_email_change -- which must stay ``async def`` for the
    awaited send below -- can run this whole blocking half in one worker-thread
    hop. Every step here blocks: one bcrypt verify (~190 ms) plus four store
    round trips, which are network calls against Postgres in prod. Offloading
    only the bcrypt would leave the four round trips stalling the event loop,
    which on a cold pooled connection costs more than the hash does.

    Raises HTTPException directly; those propagate out of asyncio.to_thread
    unchanged, so FastAPI turns them into the same responses as before.
    """
    if not verify_password(payload.current_password, current_user["password_hash"]):
        raise HTTPException(status_code=400, detail="Current password is incorrect.")
    if payload.new_email == str(current_user["email"]).strip().lower():
        raise HTTPException(status_code=400, detail="That is already your email address.")
    # This 409 is an account-enumeration oracle, and that is accepted: POST
    # /signup already answers the same question unauthenticated and unlimited.
    # It runs BEFORE the cooldown check below, so cooldown does not bound it --
    # what bounds it is that this path additionally requires a valid session
    # and the account's own password, unlike signup. Failing here beats walking
    # someone through two codes only to 409 at commit -- the commit-time check
    # stays as the TOCTOU backstop.
    if store.get_user_by_email(payload.new_email):
        raise HTTPException(status_code=409, detail="Email is already registered")

    # Three rate limits, all AFTER the password check so a typo does not burn
    # any allowance, and ordered widest window first: whichever fires, the user
    # is told the longest wait that actually applies rather than the shortest.
    user_id = current_user["id"]

    # 1. Product policy: an email address is a contact attribute, not something
    #    to churn. Keyed on the last *completed* change -- keying it on the last
    #    request would cost a week for a single mistyped address.
    #    It also protects anything later bound to an account (entitlements,
    #    paid plans) from being shed or inherited by hopping addresses. Those
    #    must key on users.id regardless; see the note atop users.py.
    min_interval = users_module.EMAIL_CHANGE_MIN_INTERVAL_DAYS * 86400
    completed_at = store.last_email_change_completed_at(user_id)
    if completed_at:
        remaining = min_interval - _seconds_since(completed_at)
        if remaining > 0:
            raise _too_soon(
                "You can change your email address once every "
                f"{users_module.EMAIL_CHANGE_MIN_INTERVAL_DAYS} days. "
                f"Try again in {_humanize_wait(remaining)}.",
                remaining,
            )

    # 2. Rolling daily cap. The 60-second cooldown below bounds a *cycle*, but
    #    each cycle can send two messages -- one to this account, one to an
    #    address the requester picked -- so on its own it still lets one account
    #    drain the platform's shared daily provider quota in a couple of hours,
    #    half of it aimed at a third party from our sending domain. This is what
    #    closes that; the window frees as the oldest request in it ages out.
    recent = store.email_change_request_times_since(user_id, _daily_window_start())
    if len(recent) >= users_module.EMAIL_CHANGE_MAX_REQUESTS_PER_DAY:
        remaining = _DAILY_CAP_WINDOW.total_seconds() - _seconds_since(recent[0])
        raise _too_soon(
            "Too many email-change requests today. "
            f"Try again in {_humanize_wait(remaining)}.",
            remaining,
        )

    # 3. Per-request throttle.
    last_at = store.last_email_change_request_at(user_id)
    cooldown = users_module.EMAIL_CHANGE_COOLDOWN_SECONDS
    if last_at:
        remaining = cooldown - _seconds_since(last_at)
        if remaining > 0:
            raise _too_soon(
                "Please wait a minute before requesting another code.", remaining
            )


@router.post("/email-change")
async def request_email_change(
    payload: EmailChangeRequest,
    current_user: dict = Depends(get_current_user),
):
    # Stays async: send_email() is a real coroutine. Everything either side of
    # it blocks, so each half goes to a worker thread rather than running on the
    # event loop (#297). Two hops, not five inline store calls plus a bcrypt.
    store = users_module.user_store
    await asyncio.to_thread(_authorize_email_change, store, current_user, payload)

    code = generate_code()
    # Send BEFORE persisting. Persisting first and then failing to send would
    # burn the cooldown on a code that does not exist.
    sent = await email_sender.send_email(
        to=str(current_user["email"]),
        subject="Confirm your Agentic Trading Lab email change",
        text_body=_email_change_old_body(code, payload.new_email),
    )
    if not sent:
        raise HTTPException(
            status_code=503,
            detail="Could not send the confirmation email. Please try again later.",
        )
    await asyncio.to_thread(
        store.create_email_change_request,
        current_user["id"],
        payload.new_email,
        hash_code(code),
    )
    return {"stage": "old", "new_email": payload.new_email}


@router.get("/email-change")
def get_email_change(current_user: dict = Depends(get_current_user)):
    """Let a reloaded page pick the flow back up instead of stranding the user."""
    row = users_module.user_store.get_active_email_change(current_user["id"])
    if not row:
        return {"pending": False, "stage": None, "new_email": None, "expires_at": None}
    return {
        "pending": True,
        "stage": row["stage"],
        "new_email": row["new_email"],
        "expires_at": str(row["expires_at"]),
    }


@router.delete("/email-change")
def cancel_email_change(current_user: dict = Depends(get_current_user)):
    """Cancel a pending change. Also the resend path: cancel, then start again,
    which re-verifies the password.

    Store-level cancel deactivates rather than deletes, so a caller cannot use
    this (session-only, no password) to reset the 60-second request cooldown.
    """
    users_module.user_store.cancel_email_change(current_user["id"])
    return {"status": "ok"}


@router.post("/email-change/verify")
async def verify_email_change(
    payload: EmailChangeVerifyRequest,
    request: Request,
    current_user: dict = Depends(get_current_user),
    authorization: Optional[str] = Header(default=None),
):
    """One stage-driven endpoint, not two.

    The server already knows which stage is outstanding; separate
    verify-current and confirm endpoints would only give the client a way to
    call the wrong one.
    """
    store = users_module.user_store
    request_row = store.get_active_email_change(current_user["id"])
    if not request_row:
        raise HTTPException(
            status_code=400, detail="No email change is in progress. Start again."
        )

    # Constant-time out of idiom rather than necessity -- comparing two
    # fixed-length digests already denies a byte-by-byte timing oracle.
    if not hmac.compare_digest(hash_code(payload.code), str(request_row["code_hash"])):
        attempts = store.record_email_change_attempt(request_row["id"])
        if attempts >= users_module.EMAIL_CHANGE_MAX_ATTEMPTS:
            store.cancel_email_change(current_user["id"])
            raise HTTPException(
                status_code=400,
                detail="Too many incorrect codes. Start the email change again.",
            )
        raise HTTPException(status_code=400, detail="That code is not correct.")

    new_email = str(request_row["new_email"])

    if request_row["stage"] == "old":
        code = generate_code()
        # Send BEFORE persisting stage 'new'. The other order strands the user:
        # waiting on a code that was never delivered, while the code they do
        # hold is no longer accepted, with Cancel the only exit and nothing on
        # screen to explain it. Failing here leaves stage 'old' untouched, so
        # they can simply resubmit the code they already have.
        sent = await email_sender.send_email(
            to=new_email,
            subject="Confirm your new Agentic Trading Lab email address",
            text_body=_email_change_new_body(code),
        )
        if not sent:
            raise HTTPException(
                status_code=503,
                detail="Could not send the confirmation email. Please try again.",
            )
        store.advance_email_change(request_row["id"], hash_code(code))
        return {"stage": "new", "new_email": new_email}

    try:
        user = store.update_email(current_user["id"], new_email)
    except ValueError as exc:
        if str(exc) == "email_already_registered":
            store.cancel_email_change(current_user["id"])
            raise HTTPException(
                status_code=409, detail="Email is already registered"
            ) from exc
        raise HTTPException(
            status_code=401, detail="Session is no longer valid."
        ) from exc

    store.mark_email_change_used(request_row["id"])
    # Best-effort, exactly as in change-password: an email change is an identity
    # change, so other sessions end -- but the durable write already landed, so a
    # revocation failure is a WARNING, not a 500. ERROR is reserved for the mail
    # failures above, where the user genuinely gets nothing.
    _d7_cleanup(
        "email change",
        current_user["id"],
        "other-session revocation",
        lambda: store.delete_other_sessions(
            current_user["id"], keep_token=_session_token(request, authorization)
        ),
    )
    # A reset code was mailed to the *old* address; it must not survive the
    # address changing (D7 symmetry with change-password above).
    _d7_cleanup(
        "email change",
        current_user["id"],
        "cancelling the pending password reset",
        lambda: store.cancel_password_reset(current_user["id"]),
    )
    return {"status": "ok", "user": user}


def _d7_cleanup(action: str, user_id, step: str, fn) -> None:
    """One D7 best-effort invalidation, after a committed credential change.

    The durable write already landed, so a failure here surfaces as a WARNING
    print (logger output is invisible under the deployed config) and never as
    a 500 that would misreport the committed change to the client.
    """
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 -- the credential change already committed
        print(f"WARNING: {action} committed for user {user_id} but {step} failed: {exc!r}")


_DAILY_CAP_WINDOW = timedelta(days=1)


def _daily_window_start() -> str:
    """Stored-format start of the rolling daily request-cap window."""
    return format_stored_timestamp(datetime.now(timezone.utc) - _DAILY_CAP_WINDOW)


def _password_reset_body(code: str) -> str:
    """Sent to the account's own address; the requester may be an attacker,
    so the closing line has to be safe for the innocent-recipient case."""
    return (
        "Someone requested a password reset for the Agentic Trading Lab "
        "account belonging to this address.\n\n"
        f"Your reset code: {code}\n\n"
        "Enter it on the password reset screen along with your new password. "
        f"The code expires in {users_module.PASSWORD_RESET_TTL_MINUTES} "
        "minutes and can be used once.\n\n"
        "If you didn't request this, you can ignore this email -- your "
        "password has not been changed."
    )


async def _deliver_password_reset_code(email: str) -> None:
    """Everything account-shaped about forgot-password, after the response.

    Runs as a BackgroundTasks task so the route's status *and latency* are
    uniform for every caller -- the verify_password_for_account lesson applied
    to the email send. Store calls hop to a worker thread; the send is a real
    coroutine and is awaited directly. Every skip prints a reason, because a
    silent 200 is the only thing the caller ever sees.
    """
    store = users_module.user_store
    domain = _email_domain(email)
    user = await asyncio.to_thread(store.get_user_by_email, email)
    if not user:
        print(f"auth.reset_skipped reason=unknown domain={domain}")
        return
    user_id = int(user["id"])

    # Durable cooldown + daily cap, enforced silently (an inline 429 keyed on
    # account state would be an enumeration oracle). Both reads are
    # status-blind, so cancelling a request never resets the clock. The
    # check-then-act here is accepted as racy: the inline per-email limiter is
    # the hard bound on issuance cadence, these are the backstop that survives
    # a redeploy.
    last_at = await asyncio.to_thread(store.last_password_reset_request_at, user_id)
    if last_at and _seconds_since(last_at) < users_module.PASSWORD_RESET_COOLDOWN_SECONDS:
        print(f"auth.reset_skipped reason=cooldown domain={domain}")
        return
    recent = await asyncio.to_thread(
        store.password_reset_request_times_since, user_id, _daily_window_start()
    )
    if len(recent) >= users_module.PASSWORD_RESET_MAX_REQUESTS_PER_DAY:
        print(f"auth.reset_skipped reason=daily_cap domain={domain}")
        return

    # Server-wide send budget, charged immediately before the send: bounds an
    # anonymous caller draining the shared Brevo quota through known accounts.
    if not _FORGOT_GLOBAL_LIMITER.allow("global"):
        print(f"WARNING: auth.reset_skipped reason=global_cap domain={domain}")
        return

    code = generate_code()
    # Send BEFORE persisting (the email-change invariant): a failed send
    # persists nothing, so the user's retry is not cooldown-blocked.
    sent = await email_sender.send_email(
        to=str(user["email"]),
        subject="Your Agentic Trading Lab password reset code",
        text_body=_password_reset_body(code),
    )
    if not sent:
        print(
            f"ERROR: password reset code send failed domain={domain}; "
            "nothing was persisted"
        )
        return
    await asyncio.to_thread(
        store.create_password_reset_request, user_id, hash_code(code)
    )
    print(f"auth.reset_requested domain={domain}")


async def _deliver_password_reset_code_task(email: str) -> None:
    """Failure boundary for the background task.

    It runs after the 200 already went out, so an uncaught store/send error
    would otherwise vanish with no log line at all (the daily-leaderboard
    background refresh wraps its body the same way). ERROR, not a skip line:
    the caller was told ok and got nothing.
    """
    try:
        await _deliver_password_reset_code(email)
    except Exception as exc:  # noqa: BLE001 -- post-response; print is the only sink
        print(
            "ERROR: password reset delivery failed "
            f"domain={_email_domain(email)}: {exc!r}"
        )


@router.post("/forgot-password")
async def forgot_password(
    payload: ForgotPasswordRequest, request: Request, background_tasks: BackgroundTasks
):
    # allow() on both keys for every accepted request -- there is no
    # success/failure split to exempt, and both keys are existence-blind.
    ip_key = f"forgot:{client_key(request)}"
    if not _FORGOT_IP_LIMITER.allow(ip_key):
        raise _auth_rate_limited(_FORGOT_IP_LIMITER, ip_key, FORGOT_RATE_LIMIT_DETAIL)
    email_key = f"forgot:email:{payload.email}"
    if not _FORGOT_EMAIL_LIMITER.allow(email_key):
        raise _auth_rate_limited(
            _FORGOT_EMAIL_LIMITER, email_key, FORGOT_RATE_LIMIT_DETAIL
        )

    # Before any account lookup, identically for every caller: this is
    # config-shaped information, not account-shaped, and it keeps a
    # Brevo-unconfigured deploy fail-visible instead of silently 200ing.
    # email_configured() has no side effects, so the route prints its own line.
    if not email_sender.email_configured():
        print(
            "ERROR: password reset requested but BREVO_API_KEY / "
            "ACCOUNT_EMAIL_FROM are not set -- returning 503"
        )
        raise HTTPException(
            status_code=503,
            detail="Could not send the confirmation email. Please try again later.",
        )

    # Immediate generic 200 for everyone else; all real work happens after the
    # response so neither the body nor the latency says whether an account
    # exists. (TestClient runs background tasks before returning, so tests
    # stay deterministic.)
    background_tasks.add_task(_deliver_password_reset_code_task, payload.email)
    return {"status": "ok"}


@router.post("/reset-password")
def reset_password(payload: ResetPasswordRequest, request: Request):
    # Plain def, like change_password: store I/O plus one bcrypt hash inside
    # update_password, no await. Pinned in BLOCKING_IO_HANDLERS.
    ip_key = f"reset:{client_key(request)}"
    if not _RESET_IP_LIMITER.check(ip_key):
        raise _auth_rate_limited(_RESET_IP_LIMITER, ip_key, RESET_RATE_LIMIT_DETAIL)

    store = users_module.user_store
    email_key = f"reset:email:{payload.email}"

    def _failure() -> HTTPException:
        # Built and returned, never raised here (a raise-helper trips
        # py/mixed-returns). Charges both budgets on every failure outcome:
        # record() on the IP limiter, allow() on the per-email one -- allow()
        # is what makes the per-email budget actually enforce, so a hammered
        # address starts answering 429 on subsequent attempts.
        _RESET_IP_LIMITER.record(ip_key)
        if not _RESET_EMAIL_LIMITER.allow(email_key):
            return _auth_rate_limited(
                _RESET_EMAIL_LIMITER, email_key, RESET_RATE_LIMIT_DETAIL
            )
        return HTTPException(status_code=400, detail=RESET_FAILURE_DETAIL)

    # Unknown email and "no active row" are one uniform failure; expiry is not
    # a separate branch because the store folds it into "no active row".
    user = store.get_user_by_email(payload.email)
    if not user:
        raise _failure()
    row = store.get_active_password_reset(int(user["id"]))
    if not row:
        raise _failure()

    if not hmac.compare_digest(hash_code(payload.code), str(row["code_hash"])):
        # The store cancels the request in SQL when this attempt reaches the
        # cap; the response is the same generic 400 either way.
        store.record_password_reset_attempt(int(row["id"]))
        raise _failure()

    # Policy only after the code passes: the request row is untouched, so the
    # user resubmits the same still-valid code with a better password. Only a
    # caller who already presented the correct code can tell this 400 from the
    # generic one -- i.e. someone who has already won; it leaks nothing.
    violations = validate_new_password(payload.new_password, str(user["email"]))
    if violations:
        raise HTTPException(status_code=400, detail=" ".join(violations))

    # Consume the code FIRST (atomic CAS; a losing concurrent redeem gets the
    # generic 400), then write the password. A crash between the two burns a
    # code -- the user re-requests -- instead of leaving a live code after a
    # state change.
    if not store.mark_password_reset_used(int(row["id"])):
        raise _failure()
    store.update_password(int(user["id"]), payload.new_password)

    # Best-effort compromise response (D7), after the durable write: ALL
    # sessions die (there is no session to keep -- the caller proved an inbox,
    # not a login), and a pending email change dies with them. Failures are
    # WARNINGs, never a 500 that would misreport a committed change.
    _d7_cleanup(
        "password reset",
        user["id"],
        "session revocation",
        lambda: store.delete_other_sessions(int(user["id"]), keep_token=None),
    )
    _d7_cleanup(
        "password reset",
        user["id"],
        "cancelling the pending email change",
        lambda: store.cancel_email_change(int(user["id"])),
    )
    print(f"auth.reset_completed domain={_email_domain(payload.email)}")
    # No auto-login: minting a session for an email-only proof would skip the
    # fresh-password check login performs.
    return {"status": "ok"}


def _store_avatar(user_id: int, value: Optional[str]) -> dict:
    """
    Write the avatar, mapping a vanished account to 401 instead of 500.

    Both twin stores raise ValueError("user_not_found") when the row is gone between
    the session lookup in get_current_user and this write. That is a session that
    outlived its account -- an auth failure the client can act on (sign in again),
    not a server fault. Unreachable today (nothing deletes users), which is exactly
    why it is worth pinning down before account deletion lands in a later phase.
    """
    try:
        return users_module.user_store.set_avatar(user_id, value)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Session is no longer valid.") from exc


@router.put("/avatar")
def set_avatar(payload: AvatarRequest, current_user: dict = Depends(get_current_user)):
    try:
        value = _validate_avatar_data_uri(payload.avatar)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"user": _store_avatar(current_user["id"], value)}


@router.delete("/avatar")
def delete_avatar(current_user: dict = Depends(get_current_user)):
    return {"user": _store_avatar(current_user["id"], None)}


class RobinhoodStartBody(BaseModel):
    agent_id: Optional[str] = Field(default=None, max_length=64)


@router.post("/robinhood/start")
async def robinhood_oauth_start(
    body: RobinhoodStartBody | None = None,
    current_user: dict = Depends(get_current_user),
):
    """Begin Robinhood Agentic OAuth for the logged-in user."""
    if not robinhood_oauth.oauth_configured():
        raise HTTPException(status_code=503, detail="Robinhood OAuth is not configured")
    user_id = int(current_user["id"])
    existing = broker_store.get_public(user_id)
    if existing:
        return {
            "already_linked": True,
            "authorize_url": None,
            "agent_id": (body.agent_id if body else None),
            "user": public_user(current_user),
        }

    # register_client() and build_authorize_url() both make synchronous httpx
    # calls with a 40s timeout, so running them inline would stall the whole
    # event loop for up to ~80s per click. Push both onto worker threads.
    try:
        client_id = await asyncio.to_thread(robinhood_oauth.register_client)
    except Exception as exc:  # noqa: BLE001 -- upstream/network failure, not a client error
        logger.exception("Robinhood dynamic client registration failed")
        raise HTTPException(status_code=502, detail="Could not reach Robinhood") from exc

    code_verifier, code_challenge = robinhood_oauth.generate_pkce_pair()
    agent_id = body.agent_id if body else None
    state = robinhood_oauth.mint_oauth_state(
        user_id,
        agent_id=agent_id,
        code_verifier=code_verifier,
        client_id=client_id,
    )
    try:
        authorize_url = await asyncio.to_thread(
            robinhood_oauth.build_authorize_url,
            state=state,
            client_id=client_id,
            code_challenge=code_challenge,
        )
    except Exception as exc:  # noqa: BLE001 -- metadata fetch failure
        logger.exception("Robinhood authorize URL construction failed")
        raise HTTPException(status_code=502, detail="Could not reach Robinhood") from exc

    return {
        "already_linked": False,
        "authorize_url": authorize_url,
        "agent_id": agent_id,
        "user": public_user(current_user),
    }


class RobinhoodCompleteBody(BaseModel):
    link_code: str = Field(min_length=8, max_length=128)


@router.get("/robinhood/callback")
async def robinhood_oauth_callback(code: Optional[str] = None, state: Optional[str] = None):
    """OAuth redirect: exchange the code, park the tokens, return to /app.

    Deliberately unauthenticated -- it is a browser redirect from Robinhood, and
    this app authenticates with an ``Authorization: Bearer`` header only (there
    are no cookies), so no session can be proven here. That makes the ``uid``
    inside the signed state a *hint about who started the flow*, never proof of
    who finished it: an attacker can start the flow on their own account and
    hand the resulting authorize_url to a victim.

    So this endpoint never writes to ``broker_store``. It parks the exchanged
    tokens in a single-use, short-lived slot and returns only the opaque code
    for it; ``POST /auth/robinhood/complete`` redeems that code against a real
    session and refuses to bind the tokens to a different account.
    """
    if not code or not state:
        return _app_redirect({"robinhood": "error", "reason": "missing_params"})
    try:
        payload = robinhood_oauth.parse_oauth_state(state)
    except ValueError as exc:
        reason = str(exc) if str(exc) in {"invalid_state", "state_expired"} else "invalid_state"
        return _app_redirect({"robinhood": "error", "reason": reason})

    try:
        started_by_user_id = int(payload["uid"])
    except (KeyError, TypeError, ValueError):
        return _app_redirect({"robinhood": "error", "reason": "invalid_state"})

    client_id = str(payload["cid"])
    agent_id = payload.get("aid")
    try:
        token_data = await asyncio.to_thread(
            robinhood_oauth.exchange_code_for_tokens,
            code=code,
            client_id=client_id,
            code_verifier=str(payload["cv"]),
        )
    except Exception:  # noqa: BLE001 -- any exchange failure is one user-facing outcome
        logger.exception("Robinhood token exchange failed")
        return _app_redirect({"robinhood": "error", "reason": "oauth_failed"})

    if not isinstance(token_data, dict) or not token_data.get("access_token"):
        logger.warning("Robinhood token exchange returned no access_token")
        return _app_redirect({"robinhood": "error", "reason": "oauth_failed"})

    link_code = pending_links.put(
        user_id=started_by_user_id,
        agent_id=str(agent_id) if agent_id else None,
        tokens=token_data,
        client_id=client_id,
    )

    query: dict[str, str] = {"robinhood": "pending", "link_code": link_code}
    if agent_id:
        query["agent_id"] = str(agent_id)
    return _app_redirect(query)


@router.post("/robinhood/complete")
async def robinhood_oauth_complete(
    body: RobinhoodCompleteBody,
    current_user: dict = Depends(get_current_user),
):
    """Second leg of the link: bind parked Robinhood tokens to the caller's account.

    This is the only place broker tokens are persisted, and it runs under a real
    session -- so the account that receives live-trading credentials is always
    the account that redeemed them.
    """
    record = pending_links.pop(body.link_code)
    if record is None:
        raise HTTPException(status_code=400, detail="Link expired - please connect again.")

    user_id = int(current_user["id"])
    if record["user_id"] != user_id:
        # The flow was started by one account and finished by another. That is the
        # account-linking CSRF this two-legged handshake exists to stop, so drop the
        # record on the floor rather than re-storing it -- discarding it is the point.
        logger.warning(
            "Robinhood link rejected: pending record was started by user %s "
            "but redeemed by user %s",
            record["user_id"],
            user_id,
        )
        raise HTTPException(
            status_code=403,
            detail="This Robinhood link was started from a different account.",
        )

    tokens = record["tokens"]
    await asyncio.to_thread(
        broker_store.upsert_tokens,
        user_id,
        access_token=str(tokens["access_token"]),
        refresh_token=tokens.get("refresh_token"),
        client_id=record["client_id"],
        token_expires_at=robinhood_oauth.token_expires_at_iso(tokens.get("expires_in")),
    )
    return {"status": "ok", "connected": True, "agent_id": record.get("agent_id")}


@router.post("/discord/start")
def discord_oauth_start(current_user: dict = Depends(get_current_user)):
    """Begin Discord OAuth linking for the logged-in website user."""
    if not discord_oauth.oauth_configured():
        raise HTTPException(
            status_code=503,
            detail="Discord OAuth is not configured on this server",
        )
    # Already linked → client can skip OAuth and open Discord directly.
    if current_user.get("discord_user_id"):
        return {
            "already_linked": True,
            "authorize_url": None,
            "discord_url": discord_oauth.discord_guild_channel_url(),
            "user": public_user(current_user),
        }

    state = discord_oauth.mint_oauth_state(int(current_user["id"]))
    return {
        "already_linked": False,
        "authorize_url": discord_oauth.build_authorize_url(state),
        "discord_url": discord_oauth.discord_guild_channel_url(),
        "user": public_user(current_user),
    }


@router.get("/discord/callback")
async def discord_oauth_callback(code: Optional[str] = None, state: Optional[str] = None):
    """OAuth redirect target: exchange code, persist discord_user_id, return to /app."""
    if not code or not state:
        return _app_redirect({"discord": "error", "reason": "missing_params"})
    try:
        user_id = discord_oauth.parse_oauth_state(state)
    except ValueError:
        return _app_redirect({"discord": "error", "reason": "invalid_state"})

    try:
        # These make blocking HTTP/DB calls; run them off the event loop so a slow
        # Discord token exchange (up to ~40s) doesn't stall every other request.
        access_token = await asyncio.to_thread(
            discord_oauth.exchange_code_for_access_token, code
        )
        discord_user = await asyncio.to_thread(
            discord_oauth.fetch_discord_user, access_token
        )
        await asyncio.to_thread(
            users_module.user_store.link_discord_user, user_id, str(discord_user["id"])
        )
    except ValueError as exc:
        reason = str(exc) if str(exc) in {"discord_already_linked", "user_not_found"} else "link_failed"
        return _app_redirect({"discord": "error", "reason": reason})
    except Exception:
        return _app_redirect({"discord": "error", "reason": "oauth_failed"})

    return _app_redirect({"discord": "linked"})
