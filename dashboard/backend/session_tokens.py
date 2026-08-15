"""Opaque auth-session token hashing and lifetime policy.

Raw session tokens are shown only to the client (today in JSON; later in an
HttpOnly cookie). The database stores HMAC-SHA256 digests so a DB leak does not
yield usable credentials.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Set, Tuple

_DEFAULT_TTL_DAYS = 7
_DEFAULT_IDLE_HOURS = 24
_DEFAULT_LAST_SEEN_THROTTLE_SECONDS = 600
_DEV_FALLBACK_SECRET = b"atl-dev-only-session-hash-secret"

# Every announcement in this module goes through print(), never logging: nothing
# configures logging for dashboard.backend.*, and uvicorn's LOGGING_CONFIG has no
# 'root' key, so a logger call emits nothing in a real deployment -- see the same
# note on users._build_user_store(). Deduped because the lifetime getters are
# read per call (so an operator can retune them without a restart), and one line
# per request would bury the deploy log.
_reported_env_problems: Set[Tuple[str, str]] = set()


def _report_once(name: str, raw: str, message: str) -> None:
    key = (name, raw)
    if key in _reported_env_problems:
        return
    _reported_env_problems.add(key)
    print(message)


def _env_int(name: str, default: int) -> int:
    """Read a positive-integer env override, announcing anything it rejects.

    Falling back to the default on garbage is right -- a mistyped lifetime must
    not open an unbounded session. Doing it silently is not: ``SESSION_TTL_DAYS=7d``
    looks like it took effect, and the difference is an auth policy.
    """
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        _report_once(
            name,
            raw,
            f"{name}={raw!r} is not an integer; falling back to {default}",
        )
        return default
    if value < 1:
        _report_once(
            name,
            raw,
            f"{name}={raw!r} must be >= 1; falling back to {default}",
        )
        return default
    return value


def session_ttl_days() -> int:
    return _env_int("SESSION_TTL_DAYS", _DEFAULT_TTL_DAYS)


def session_idle_hours() -> int:
    return _env_int("SESSION_IDLE_HOURS", _DEFAULT_IDLE_HOURS)


def session_last_seen_throttle_seconds() -> int:
    return _env_int(
        "SESSION_LAST_SEEN_THROTTLE_SECONDS",
        _DEFAULT_LAST_SEEN_THROTTLE_SECONDS,
    )


def _is_production() -> bool:
    return bool((os.getenv("RENDER") or "").strip()) or (
        os.getenv("ATL_ENV") or ""
    ).strip().lower() in {"production", "prod"}


def session_hash_secret() -> bytes:
    """HMAC key for token digests -- the silent, per-request accessor.

    ``SESSION_HASH_SECRET`` must be set in real deployments. Local/tests may
    omit it and get a fixed fallback so the suite stays hermetic; the fallback
    is intentionally not suitable for production.

    The raise is kept here as well as in ``require_session_hash_secret`` so a
    misconfiguration can never quietly downgrade to the dev key, even if some
    future entrypoint skips the startup check. Announcing is the startup
    check's job -- this runs on every authenticated request.
    """
    secret = (os.getenv("SESSION_HASH_SECRET") or "").strip()
    if secret:
        return secret.encode("utf-8")
    if _is_production():
        raise RuntimeError(
            "SESSION_HASH_SECRET must be set when running in production"
        )
    return _DEV_FALLBACK_SECRET


def require_session_hash_secret() -> None:
    """Startup gate: resolve the HMAC key once, where a human is watching.

    Without this, a prod deploy missing ``SESSION_HASH_SECRET`` boots clean and
    serves ``/health`` 200 -- the deploy hook reports success -- and then every
    login and every authenticated request 500s, un-CORS-headered, because the
    raise happens at *call* time. Render keeps the previous version live when a
    boot fails, so failing here is strictly the safer half of that trade.
    """
    session_hash_secret()
    if not (os.getenv("SESSION_HASH_SECRET") or "").strip():
        _report_once(
            "SESSION_HASH_SECRET",
            "",
            "SESSION_HASH_SECRET unset; using the development-only fallback key "
            "(sessions are not protected against a database leak)",
        )


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def hash_session_token(raw_token: str) -> str:
    return hmac.new(
        session_hash_secret(),
        raw_token.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def secrets_equal(provided: str, expected: str) -> bool:
    """Constant-time compare of an untrusted string against a server secret.

    The canonical helper for shared-secret gates (admin bootstrap today);
    routes must import this rather than growing another ad-hoc compare.

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


def absolute_expiry(now: Optional[datetime] = None) -> datetime:
    current = now or datetime.now(timezone.utc)
    return (current + timedelta(days=session_ttl_days())).replace(microsecond=0)


def idle_deadline(last_activity: datetime) -> datetime:
    return (last_activity + timedelta(hours=session_idle_hours())).replace(
        microsecond=0
    )


def should_touch_last_seen(last_seen_at: datetime, now: Optional[datetime] = None) -> bool:
    current = now or datetime.now(timezone.utc)
    age = (current - last_seen_at).total_seconds()
    return age >= session_last_seen_throttle_seconds()
