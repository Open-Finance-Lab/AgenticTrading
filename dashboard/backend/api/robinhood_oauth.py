"""Robinhood Agentic Trading MCP OAuth (PKCE + dynamic client registration)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode

import httpx

MCP_URL = (os.getenv("ROBINHOOD_MCP_URL") or "https://agent.robinhood.com/mcp/trading").strip()
METADATA_URL = f"{MCP_URL.rsplit('/mcp/', 1)[0]}/.well-known/oauth-authorization-server/mcp/trading"
STATE_TTL_SECONDS = 900
_HTTP_TIMEOUT = 40.0

_metadata_cache: Optional[Dict[str, Any]] = None


def redirect_uri() -> str:
    return (
        os.getenv("ROBINHOOD_REDIRECT_URI")
        or "http://localhost:8000/api/auth/robinhood/callback"
    ).strip()


def oauth_configured() -> bool:
    return bool(redirect_uri())


def _state_signing_key() -> bytes:
    raw = (
        os.getenv("ROBINHOOD_OAUTH_STATE_SECRET")
        or os.getenv("DISCORD_CLIENT_SECRET")
        or "dev-robinhood-oauth-state"
    )
    return raw.encode("utf-8")


def _pkce_pair() -> Tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")
    return verifier, challenge


def generate_pkce_pair() -> Tuple[str, str]:
    """Return (code_verifier, code_challenge) for PKCE OAuth."""
    return _pkce_pair()


def fetch_metadata() -> Dict[str, Any]:
    global _metadata_cache
    if _metadata_cache is not None:
        return _metadata_cache
    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        resp = client.get(METADATA_URL)
        resp.raise_for_status()
        _metadata_cache = resp.json()
        return _metadata_cache


def register_client() -> str:
    meta = fetch_metadata()
    registration_endpoint = meta["registration_endpoint"]
    payload = {
        "redirect_uris": [redirect_uri()],
        "client_name": "Agentic Trading Lab",
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }
    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        resp = client.post(registration_endpoint, json=payload)
        resp.raise_for_status()
        data = resp.json()
    client_id = data.get("client_id")
    if not client_id:
        raise ValueError("registration_missing_client_id")
    return str(client_id)


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
