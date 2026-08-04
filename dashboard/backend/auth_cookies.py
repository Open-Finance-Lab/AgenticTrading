"""HttpOnly session-cookie helpers for account auth.

Raw session tokens must never be readable by frontend JavaScript. The cookie
carries the opaque token; the DB stores only its HMAC digest (session_tokens).
"""

from __future__ import annotations

import os
import re
from typing import Optional

from fastapi import Request, Response

from dashboard.backend.session_tokens import session_ttl_days

_HOST_COOKIE = "__Host-atl_session"
_DEV_COOKIE = "atl_session"

# Server tokens are secrets.token_urlsafe(32) — 43 chars of [A-Za-z0-9_-].
# The bounds stay loose so a future length change doesn't lock users out.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{20,128}")


def cookie_secure() -> bool:
    raw = (os.getenv("ATL_COOKIE_SECURE") or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool((os.getenv("RENDER") or "").strip()) or (
        (os.getenv("ATL_ENV") or "").strip().lower() in {"production", "prod"}
    )


def session_cookie_name() -> str:
    # __Host- requires Secure + Path=/ + no Domain; unavailable on plain HTTP.
    return _HOST_COOKIE if cookie_secure() else _DEV_COOKIE


def read_session_token(request: Request) -> Optional[str]:
    for name in (_HOST_COOKIE, _DEV_COOKIE):
        value = request.cookies.get(name)
        if value and value.strip():
            return value.strip()
    return None


def set_session_cookie(response: Response, raw_token: str) -> None:
    # The /me migration bridge feeds this a client-supplied Bearer value, so
    # gate on the exact server-token alphabet before it can reach a Set-Cookie
    # header. Server-generated tokens always match; anything else is garbage a
    # cookie could never authenticate with, so silently skipping it is safe.
    if not _TOKEN_RE.fullmatch(raw_token):
        return
    response.set_cookie(
        key=session_cookie_name(),
        value=raw_token,
        httponly=True,
        secure=cookie_secure(),
        samesite="lax",
        path="/",
        max_age=session_ttl_days() * 24 * 60 * 60,
    )


def clear_session_cookie(response: Response) -> None:
    # Clear both names so a device that flipped Secure mid-flight cannot keep
    # a stale cookie under the other name.
    #
    # The __Host- deletion must itself carry Secure: RFC 6265bis §4.1.3.1 makes
    # a UA reject any Set-Cookie for a __Host-* name that lacks it, and
    # delete_cookie defaults secure=False — so without this flag the prod
    # cookie survives logout in the browser (revoked server-side, but still
    # sent). httponly/samesite match the set path so the delete is the exact
    # attribute-for-attribute counterpart of set_session_cookie.
    for name in {_HOST_COOKIE, _DEV_COOKIE, session_cookie_name()}:
        response.delete_cookie(
            key=name,
            path="/",
            secure=name.startswith("__Host-") or cookie_secure(),
            httponly=True,
            samesite="lax",
        )
