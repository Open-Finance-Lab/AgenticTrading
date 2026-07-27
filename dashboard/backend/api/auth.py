import asyncio
import base64
import logging
import os
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, field_validator

from dashboard.backend.api import discord_oauth
from dashboard.backend.domain.brokers.repository import broker_store
from dashboard.backend.infrastructure.brokers import pending_links, robinhood_oauth
from dashboard.backend.infrastructure.email import sender as email_sender
from dashboard.backend import users as users_module
from dashboard.backend.users import parse_stored_timestamp, public_user, verify_password
from dashboard.backend.password_policy import validate_new_password
from dashboard.backend.verification_codes import generate_code, hash_code

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


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
    token: str


def _extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token.strip()


def get_current_user(authorization: Optional[str] = Header(default=None)) -> dict:
    token = _extract_bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = users_module.user_store.get_user_for_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return user


@router.post("/signup", response_model=AuthResponse)
async def signup(payload: SignupRequest):
    violations = validate_new_password(payload.password, payload.email)
    if violations:
        raise HTTPException(status_code=400, detail=" ".join(violations))

    try:
        user = users_module.user_store.create_user(
            email=payload.email,
            display_name=payload.display_name,
            password=payload.password,
        )
    except ValueError as exc:
        if str(exc) == "email_already_registered":
            raise HTTPException(status_code=409, detail="Email is already registered") from exc
        raise

    token = users_module.user_store.create_session(user["id"])
    return {"user": user, "token": token}


@router.post("/login", response_model=AuthResponse)
async def login(payload: LoginRequest):
    user = users_module.user_store.authenticate(payload.email, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = users_module.user_store.create_session(user["id"])
    return {"user": public_user(user), "token": token}


@router.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    return {"user": public_user(current_user)}


@router.post("/logout")
async def logout(authorization: Optional[str] = Header(default=None)):
    token = _extract_bearer_token(authorization)
    if token:
        users_module.user_store.delete_session(token)
    return {"status": "ok"}


@router.post("/change-password")
async def change_password(
    payload: ChangePasswordRequest,
    current_user: dict = Depends(get_current_user),
    authorization: Optional[str] = Header(default=None),
):
    if not verify_password(payload.current_password, current_user["password_hash"]):
        raise HTTPException(status_code=400, detail="Current password is incorrect.")
    violations = validate_new_password(payload.new_password, current_user["email"])
    if violations:
        raise HTTPException(status_code=400, detail=" ".join(violations))
    users_module.user_store.update_password(current_user["id"], payload.new_password)
    # Best-effort: revoke every other session so a stolen token dies with the old
    # password. Deliberately NOT atomic with the update above -- the two are separate
    # transactions/connections in both twin stores. The password change is already
    # durable here; if revocation raises (e.g. a transient Postgres blip on the prod
    # pool), turning it into a 500 would wrongly tell the client the change failed and
    # make a retry hit "Current password is incorrect". So swallow + surface via
    # print() (logger output is invisible under the deployed config) and still
    # return ok. Revocation is defence-in-depth, not a hard guarantee.
    try:
        users_module.user_store.delete_other_sessions(
            current_user["id"], keep_token=_extract_bearer_token(authorization)
        )
    except Exception as exc:  # noqa: BLE001 -- password change already committed
        print(
            f"WARNING: change-password committed for user {current_user['id']} but "
            f"other-session revocation failed: {exc!r}"
        )
    # D7: a user changing their password may be reacting to a compromise, so an
    # attacker's in-flight email change dies with it. Best-effort and next to
    # the session revocation above, so the whole "invalidate what the old
    # password could reach" policy sits in one place.
    try:
        users_module.user_store.cancel_email_change(current_user["id"])
    except Exception as exc:  # noqa: BLE001 -- password change already committed
        print(
            f"WARNING: change-password committed for user {current_user['id']} but "
            f"cancelling the pending email change failed: {exc!r}"
        )
    return {"status": "ok"}


@router.put("/display-name")
async def update_display_name(
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


def _email_change_body(code: str, new_email: str) -> str:
    return (
        "Someone asked to change the email address on your Agentic Trading Lab "
        f"account to {new_email}.\n\n"
        f"Your confirmation code is: {code}\n\n"
        f"It expires in {users_module.EMAIL_CHANGE_TTL_MINUTES} minutes. If this "
        "was not you, ignore this message and change your password."
    )


@router.post("/email-change")
async def request_email_change(
    payload: EmailChangeRequest,
    current_user: dict = Depends(get_current_user),
):
    store = users_module.user_store
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

    # Cooldown AFTER the password check, so a typo does not burn the allowance.
    last_at = store.last_email_change_request_at(current_user["id"])
    cooldown = users_module.EMAIL_CHANGE_COOLDOWN_SECONDS
    if last_at and _seconds_since(last_at) < cooldown:
        raise HTTPException(
            status_code=429,
            detail="Please wait a minute before requesting another code.",
            headers={"Retry-After": str(cooldown)},
        )

    code = generate_code()
    # Send BEFORE persisting. Persisting first and then failing to send would
    # burn the cooldown on a code that does not exist.
    sent = await email_sender.send_email(
        to=str(current_user["email"]),
        subject="Confirm your Agentic Trading Lab email change",
        text_body=_email_change_body(code, payload.new_email),
    )
    if not sent:
        raise HTTPException(
            status_code=503,
            detail="Could not send the confirmation email. Please try again later.",
        )
    store.create_email_change_request(
        current_user["id"], payload.new_email, hash_code(code)
    )
    return {"stage": "old", "new_email": payload.new_email}


@router.get("/email-change")
async def get_email_change(current_user: dict = Depends(get_current_user)):
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
async def cancel_email_change(current_user: dict = Depends(get_current_user)):
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

    if hash_code(payload.code) != request_row["code_hash"]:
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
            text_body=_email_change_body(code, new_email),
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
    try:
        store.delete_other_sessions(
            current_user["id"], keep_token=_extract_bearer_token(authorization)
        )
    except Exception as exc:  # noqa: BLE001 -- email change already committed
        print(
            f"WARNING: email change committed for user {current_user['id']} but "
            f"other-session revocation failed: {exc!r}"
        )
    return {"status": "ok", "user": user}


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
async def set_avatar(payload: AvatarRequest, current_user: dict = Depends(get_current_user)):
    try:
        value = _validate_avatar_data_uri(payload.avatar)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"user": _store_avatar(current_user["id"], value)}


@router.delete("/avatar")
async def delete_avatar(current_user: dict = Depends(get_current_user)):
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
async def discord_oauth_start(current_user: dict = Depends(get_current_user)):
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
