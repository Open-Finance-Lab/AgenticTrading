"""Opaque auth-session token hashing and lifetime policy.

Raw session tokens are shown only to the client (today in JSON; later in an
HttpOnly cookie). The database stores HMAC-SHA256 digests so a DB leak does not
yield usable credentials.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_TTL_DAYS = 7
_DEFAULT_IDLE_HOURS = 24
_DEFAULT_LAST_SEEN_THROTTLE_SECONDS = 600
_DEV_FALLBACK_SECRET = b"atl-dev-only-session-hash-secret"


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 1 else default


def session_ttl_days() -> int:
    return _env_int("SESSION_TTL_DAYS", _DEFAULT_TTL_DAYS)


def session_idle_hours() -> int:
    return _env_int("SESSION_IDLE_HOURS", _DEFAULT_IDLE_HOURS)


def session_last_seen_throttle_seconds() -> int:
    return _env_int(
        "SESSION_LAST_SEEN_THROTTLE_SECONDS",
        _DEFAULT_LAST_SEEN_THROTTLE_SECONDS,
    )


def session_hash_secret() -> bytes:
    """HMAC key for token digests.

    ``SESSION_HASH_SECRET`` must be set in real deployments. Local/tests may
    omit it and get a fixed fallback so the suite stays hermetic; the fallback
    is intentionally not suitable for production.
    """
    secret = (os.getenv("SESSION_HASH_SECRET") or "").strip()
    if secret:
        return secret.encode("utf-8")
    if (os.getenv("RENDER") or "").strip() or (
        os.getenv("ATL_ENV") or ""
    ).strip().lower() in {"production", "prod"}:
        raise RuntimeError(
            "SESSION_HASH_SECRET must be set when running in production"
        )
    logger.warning(
        "SESSION_HASH_SECRET unset; using a development-only fallback"
    )
    return _DEV_FALLBACK_SECRET


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def hash_session_token(raw_token: str) -> str:
    return hmac.new(
        session_hash_secret(),
        raw_token.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


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
