"""Reduce request metadata to coarse, non-reversible Analytics dimensions."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import os
import re
from datetime import datetime, timezone

from fastapi import Request

from dashboard.backend.infrastructure.request_metadata import client_ip

from .models import RequestAnalyticsContext


_FALSE_VALUES = {"", "0", "false", "no", "off"}


def _env_truthy(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() not in _FALSE_VALUES


def _normalized_ip(value: str | None) -> str | None:
    if not value or value == "unknown":
        return None
    try:
        return str(ipaddress.ip_address(value.strip()))
    except ValueError:
        return None


def monthly_network_hash(
    ip_address: str | None,
    received_at: datetime,
) -> str | None:
    secret = (os.getenv("ANALYTICS_PSEUDONYMIZATION_KEY") or "").strip()
    normalized_ip = _normalized_ip(ip_address)
    if normalized_ip is None or len(secret.encode("utf-8")) < 32:
        return None
    month = received_at.astimezone(timezone.utc).strftime("%Y-%m")
    message = f"{month}\n{normalized_ip}".encode("utf-8")
    return hmac.new(
        secret.encode("utf-8"),
        message,
        hashlib.sha256,
    ).hexdigest()


def _browser_family(user_agent: str) -> str:
    if re.search(r"(?:Edg|Edge)/", user_agent, re.IGNORECASE):
        return "Edge"
    if re.search(r"(?:Firefox|FxiOS)/", user_agent, re.IGNORECASE):
        return "Firefox"
    if re.search(r"(?:Chrome|CriOS)/", user_agent, re.IGNORECASE):
        return "Chrome"
    if "Safari/" in user_agent and "Version/" in user_agent:
        return "Safari"
    return "Other"


def _device_category(user_agent: str) -> str:
    lowered = user_agent.lower()
    if "ipad" in lowered or "tablet" in lowered:
        return "tablet"
    if any(token in lowered for token in ("iphone", "android", "mobile")):
        return "mobile"
    if any(
        token in lowered
        for token in (
            "mozilla/",
            "windows",
            "macintosh",
            "x11",
            "linux",
            "edg/",
            "chrome/",
            "firefox/",
            "safari/",
        )
    ):
        return "desktop"
    return "unknown"


def _trusted_country_code(request: Request) -> str | None:
    raw: str | None = None
    if _env_truthy("RENDER"):
        raw = request.headers.get("cf-ipcountry")
    elif _env_truthy("VERCEL"):
        raw = request.headers.get("x-vercel-ip-country")
    value = (raw or "").strip().upper()
    return value if re.fullmatch(r"[A-Z]{2}", value) else None


def request_analytics_context(
    request: Request,
    received_at: datetime,
) -> RequestAnalyticsContext:
    user_agent = (request.headers.get("user-agent") or "").strip()
    return RequestAnalyticsContext(
        country_code=_trusted_country_code(request),
        device_category=_device_category(user_agent),
        browser_family=_browser_family(user_agent),
        network_hash=monthly_network_hash(client_ip(request), received_at),
    )
