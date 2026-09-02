"""CSRF defenses for cookie-authenticated browser requests.

Mutating requests that carry the account session cookie must present:
1. A matching double-submit ``atl_csrf`` / ``__Host-atl_csrf`` cookie +
   ``X-CSRF-Token`` header, and
2. An ``Origin`` (preferred) or ``Referer`` host on the frontend allowlist
   when either header is present.

``X-API-Key`` agent traffic and cookie-less Bearer scripts skip this gate.
Login/signup (no session cookie yet) still reject disallowed Origin/Referer.
"""

from __future__ import annotations

import os
import secrets
from typing import Optional
from urllib.parse import urlparse

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from dashboard.backend.auth_cookies import cookie_secure, read_session_token

_HOST_CSRF = "__Host-atl_csrf"
_DEV_CSRF = "atl_csrf"
CSRF_HEADER = "x-csrf-token"
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})

# Login/signup must stay reachable for a browser holding a *stale* session
# cookie: sessions idle out at 24h while the cookie lives 7 days, so the
# ordinary return visit carries a dead session cookie (and possibly no CSRF
# cookie). Gating these on double-submit would 403 the only endpoints that can
# mint fresh cookies — an unrecoverable lockout. The Origin allowlist still
# applies, and none of the four trusts the session: login, signup, and
# reset-password authenticate by caller-supplied credentials, while
# forgot-password takes only a (public) email address and mutates no account
# state inline. The two password-reset routes are the same class: unauthenticated
# browser POSTs a user with a dead session cookie must still be able to make.
_CSRF_EXEMPT_PATHS = frozenset(
    {
        "/api/auth/login",
        "/api/auth/signup",
        "/api/auth/forgot-password",
        "/api/auth/reset-password",
    }
)


def csrf_cookie_name() -> str:
    return _HOST_CSRF if cookie_secure() else _DEV_CSRF


def new_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def set_csrf_cookie(response: Response, token: Optional[str] = None) -> str:
    value = token or new_csrf_token()
    response.set_cookie(
        key=csrf_cookie_name(),
        value=value,
        httponly=False,
        secure=cookie_secure(),
        samesite="lax",
        path="/",
        max_age=7 * 24 * 60 * 60,
    )
    return value


def clear_csrf_cookie(response: Response) -> None:
    for name in {_HOST_CSRF, _DEV_CSRF, csrf_cookie_name()}:
        response.delete_cookie(key=name, path="/")


def read_csrf_cookie(request: Request) -> Optional[str]:
    for name in (_HOST_CSRF, _DEV_CSRF):
        value = request.cookies.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _local_origins() -> list[str]:
    return [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


def allowed_browser_origins() -> set[str]:
    """Origins trusted for cookie/CSRF browser calls.

    Mirrors ``ATL_FRONTEND_ORIGINS`` (+ local dev hosts). When unset, only
    local hosts are listed — production must set the env var (same as CORS).
    """
    origins = set(_local_origins())
    raw = (os.getenv("ATL_FRONTEND_ORIGINS") or "").strip()
    if raw:
        for part in raw.split(","):
            origin = part.strip().rstrip("/")
            if origin:
                origins.add(origin)
    return origins


def _origin_from_headers(request: Request) -> Optional[str]:
    origin = (request.headers.get("origin") or "").strip()
    if origin:
        return origin.rstrip("/")
    referer = (request.headers.get("referer") or "").strip()
    if not referer:
        return None
    parsed = urlparse(referer)
    if not parsed.scheme or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def _request_self_origin(request: Request) -> str:
    # TestClient uses http://testserver
    return str(request.base_url).rstrip("/")


def origin_allowed(request: Request) -> bool:
    """True when Origin/Referer is absent (non-browser) or on the allowlist."""
    presented = _origin_from_headers(request)
    if presented is None:
        return True
    allowed = allowed_browser_origins()
    allowed.add(_request_self_origin(request))
    return presented in allowed


def _has_api_key(request: Request) -> bool:
    return bool((request.headers.get("x-api-key") or "").strip())


def _csrf_header_token(request: Request) -> Optional[str]:
    return (request.headers.get(CSRF_HEADER) or request.headers.get("X-CSRF-Token") or "").strip() or None


def csrf_tokens_match(request: Request) -> bool:
    cookie = read_csrf_cookie(request)
    header = _csrf_header_token(request)
    if not cookie or not header:
        return False
    # Compare bytes: compare_digest raises TypeError on non-ASCII str (headers
    # decode as latin-1), which would surface as an unauthenticated bare 500.
    return secrets.compare_digest(cookie.encode("utf-8"), header.encode("utf-8"))


def csrf_enforced() -> bool:
    """Kill-switch for the TestClient suite; production leaves ATL_CSRF unset (=on)."""
    raw = (os.getenv("ATL_CSRF") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


class CsrfMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not csrf_enforced():
            return await call_next(request)

        if request.method in _SAFE_METHODS:
            response = await call_next(request)
            # Ensure browsers that already have a session also have a CSRF cookie
            # (e.g. after a deploy that introduced CSRF mid-session).
            if read_session_token(request) and not read_csrf_cookie(request):
                set_csrf_cookie(response)
            return response

        if not origin_allowed(request):
            return JSONResponse(
                status_code=403,
                content={"detail": "Cross-origin request blocked"},
            )

        if request.url.path in _CSRF_EXEMPT_PATHS:
            return await call_next(request)

        # Session cookie present ⇒ double-submit CSRF required. A forged
        # X-API-Key header must not bypass this when the browser also sends
        # the session cookie. Cookie-less agent calls (API key only) skip it.
        if read_session_token(request):
            if not csrf_tokens_match(request):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "CSRF token missing or invalid"},
                )
            return await call_next(request)

        if _has_api_key(request):
            return await call_next(request)

        return await call_next(request)
