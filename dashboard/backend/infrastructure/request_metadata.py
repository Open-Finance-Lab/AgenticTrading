"""Request metadata helpers shared by API adapters and domain privacy code."""

from __future__ import annotations

from fastapi import Request


# Longest textual IPv6 form ("ffff:...:255.255.255.255%eth0" territory). The
# forwarded header is attacker-controlled and unbounded, and an untruncated
# value would let one client mint arbitrarily large dict keys.
_MAX_IP_KEY_LEN = 64


def client_ip(request: Request) -> str:
    """Best-effort originating client IP.

    Reads the left-most ``X-Forwarded-For`` entry before falling back to the
    socket peer. The header is caller-suppliable, so this is only a granularity
    helper for accidental-abuse budgets, never a security boundary.
    """
    for part in request.headers.get("x-forwarded-for", "").split(","):
        candidate = part.strip()
        if candidate:
            return candidate[:_MAX_IP_KEY_LEN]
    return request.client.host if request.client else "unknown"
