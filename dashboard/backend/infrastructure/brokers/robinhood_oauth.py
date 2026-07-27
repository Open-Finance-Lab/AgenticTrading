"""Robinhood Agentic Trading MCP OAuth (PKCE + dynamic client registration)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)

MCP_URL = (os.getenv("ROBINHOOD_MCP_URL") or "https://agent.robinhood.com/mcp/trading").strip()
METADATA_URL = f"{MCP_URL.rsplit('/mcp/', 1)[0]}/.well-known/oauth-authorization-server/mcp/trading"
STATE_TTL_SECONDS = 900
_HTTP_TIMEOUT = 40.0
_METADATA_TTL_SECONDS = 3600

_metadata_lock = threading.Lock()
_metadata_cache: Optional[Dict[str, Any]] = None
_metadata_fetched_at: float = 0.0

_client_lock = threading.Lock()
_client_id_cache: Dict[str, str] = {}

_state_key_lock = threading.Lock()
_ephemeral_state_key: Optional[bytes] = None


def redirect_uri() -> str:
    return (
        os.getenv("ROBINHOOD_REDIRECT_URI")
        or "http://localhost:8000/api/auth/robinhood/callback"
    ).strip()


def oauth_configured() -> bool:
    return bool(redirect_uri())


def _configured_state_secret() -> Optional[str]:
    """Return the operator-supplied state signing secret, if any."""
    raw = (
        os.getenv("ROBINHOOD_OAUTH_STATE_SECRET")
        or os.getenv("DISCORD_CLIENT_SECRET")
        or ""
    ).strip()
    return raw or None


def state_secret_is_ephemeral() -> bool:
    """True when OAuth state is signed with a random per-process key.

    Callers (and tests) can use this to tell a properly configured deployment
    apart from one that will invalidate in-flight OAuth links on restart.
    """
    return _configured_state_secret() is None


def _state_signing_key() -> bytes:
    """Return the HMAC key used to sign/verify OAuth ``state``.

    Prefers ``ROBINHOOD_OAUTH_STATE_SECRET`` (then ``DISCORD_CLIENT_SECRET``).
    With neither set we generate a random per-process key instead of falling
    back to a constant: a constant committed to a public repository lets anyone
    forge OAuth state, whereas an ephemeral key only breaks links that outlive
    the process.
    """
    global _ephemeral_state_key

    configured = _configured_state_secret()
    if configured is not None:
        return configured.encode("utf-8")

    with _state_key_lock:
        if _ephemeral_state_key is None:
            _ephemeral_state_key = secrets.token_bytes(32)
            logger.warning(
                "Robinhood OAuth state is signed with an ephemeral per-process key: "
                "authorize links break across restarts and are not shared between "
                "workers. Set ROBINHOOD_OAUTH_STATE_SECRET to a stable random secret."
            )
        return _ephemeral_state_key


def _pkce_pair() -> Tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")
    return verifier, challenge


def generate_pkce_pair() -> Tuple[str, str]:
    """Return (code_verifier, code_challenge) for PKCE OAuth."""
    return _pkce_pair()


def _metadata_is_fresh(now: float) -> bool:
    return _metadata_cache is not None and (now - _metadata_fetched_at) < _METADATA_TTL_SECONDS


def fetch_metadata() -> Dict[str, Any]:
    """Return the authorization-server metadata document, cached with a TTL.

    The cache expires after ``_METADATA_TTL_SECONDS`` so a rotated endpoint
    list is eventually picked up without a restart. The refresh happens under a
    lock, so concurrent callers issue one request rather than a thundering herd.
    """
    global _metadata_cache, _metadata_fetched_at

    cached = _metadata_cache
    if cached is not None and _metadata_is_fresh(time.monotonic()):
        return cached

    with _metadata_lock:
        cached = _metadata_cache
        if cached is not None and _metadata_is_fresh(time.monotonic()):
            return cached
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            resp = client.get(METADATA_URL)
            resp.raise_for_status()
            data = resp.json()
        _metadata_cache = data
        _metadata_fetched_at = time.monotonic()
        return data


def register_client(*, force: bool = False) -> str:
    """Return this process's RFC 7591 client_id for the current redirect URI.

    The registration is cached per redirect URI, so repeated "Connect" clicks
    reuse the already-registered client instead of burning a fresh dynamic
    registration on every call. Pass ``force=True`` to re-register and replace
    the cached value (e.g. after the provider revokes the client).
    """
    uri = redirect_uri()
    if not force:
        cached = _client_id_cache.get(uri)
        if cached:
            return cached

    meta = fetch_metadata()
    registration_endpoint = meta["registration_endpoint"]
    payload = {
        "redirect_uris": [uri],
        "client_name": "Agentic Trading Lab",
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }

    with _client_lock:
        if not force:
            cached = _client_id_cache.get(uri)
            if cached:
                return cached
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            resp = client.post(registration_endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json()
        client_id = str(data.get("client_id") or "")
        if not client_id:
            raise ValueError("registration_missing_client_id")
        _client_id_cache[uri] = client_id
        return client_id


def mint_oauth_state(
    user_id: int,
    *,
    agent_id: Optional[str] = None,
    code_verifier: str,
    client_id: str,
) -> str:
    payload = {
        "uid": int(user_id),
        "aid": agent_id,
        "cv": code_verifier,
        "cid": client_id,
        "exp": int(time.time()) + STATE_TTL_SECONDS,
        "n": secrets.token_hex(8),
    }
    raw = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()
    raw = raw.rstrip("=")
    sig = hmac.new(_state_signing_key(), raw.encode(), hashlib.sha256).hexdigest()
    return f"{raw}.{sig}"


def parse_oauth_state(state: str) -> Dict[str, Any]:
    try:
        raw, sig = state.rsplit(".", 1)
    except ValueError as exc:
        raise ValueError("invalid_state") from exc
    expected = hmac.new(_state_signing_key(), raw.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        raise ValueError("invalid_state")
    padded = raw + "=" * (-len(raw) % 4)
    payload = json.loads(base64.urlsafe_b64decode(padded.encode()))
    if int(payload.get("exp") or 0) < int(time.time()):
        raise ValueError("state_expired")
    if not payload.get("cv") or not payload.get("cid"):
        raise ValueError("invalid_state")
    return payload


def build_authorize_url(*, state: str, client_id: str, code_challenge: str) -> str:
    meta = fetch_metadata()
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri(),
        "response_type": "code",
        "scope": "internal",
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "resource": MCP_URL,
    }
    return f"{meta['authorization_endpoint']}?{urlencode(params)}"


def exchange_code_for_tokens(
    *,
    code: str,
    client_id: str,
    code_verifier: str,
) -> Dict[str, Any]:
    meta = fetch_metadata()
    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri(),
        "client_id": client_id,
        "code_verifier": code_verifier,
        "resource": MCP_URL,
    }
    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        resp = client.post(meta["token_endpoint"], data=payload)
        if resp.status_code >= 400:
            raise ValueError(f"token_exchange_failed:{resp.status_code}:{resp.text[:200]}")
        return resp.json()


def refresh_access_token(*, refresh_token: str, client_id: str) -> Dict[str, Any]:
    meta = fetch_metadata()
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "resource": MCP_URL,
    }
    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        resp = client.post(meta["token_endpoint"], data=payload)
        if resp.status_code >= 400:
            raise ValueError(f"token_refresh_failed:{resp.status_code}")
        return resp.json()


def token_expires_at_iso(expires_in: Optional[int]) -> Optional[str]:
    if not expires_in:
        return None
    from datetime import datetime, timedelta, timezone

    return (
        datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))
    ).replace(microsecond=0).isoformat()
