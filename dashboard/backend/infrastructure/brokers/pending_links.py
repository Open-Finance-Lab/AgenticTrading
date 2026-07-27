"""Short-lived, single-use handoff slots for broker account linking.

An OAuth callback arrives as a plain browser redirect, so it cannot carry this
app's ``Authorization: Bearer`` header and therefore cannot prove *who* is
sitting in front of the browser. Rather than trust the user id embedded in the
signed OAuth state (which the person who *started* the flow chose, not the
person who *finished* it), the callback parks the freshly exchanged broker
tokens here under an opaque code and hands that code back to the SPA. The SPA
then redeems it on an authenticated endpoint, where the session token proves
the caller's identity and the two user ids can be compared.

**This is per-process, in-memory state.** It is not shared between workers and
does not survive a restart: if the redeem call lands on a different process
than the callback did, the code simply will not be found. That is why the TTL
is short (:data:`PENDING_TTL_SECONDS`) and why a miss surfaces as a visible
"link expired, please connect again" error instead of anything silent -- a
failure the user can retry out of is strictly better than a mis-link that
binds one person's brokerage tokens to another person's account.

Infrastructure layer: this module must not import from ``dashboard.backend.api``.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

#: How long a pending link stays redeemable. Short on purpose -- the SPA
#: redeems it on the very next request after the OAuth redirect lands.
PENDING_TTL_SECONDS = 600

#: Hard ceiling on stored records so an attacker replaying the callback cannot
#: grow the store without bound. Oldest entries are evicted first.
MAX_PENDING_LINKS = 256

_lock = threading.Lock()
_pending: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()


def _purge_expired_locked(now: float) -> None:
    """Drop every timed-out record. Caller must hold ``_lock``."""
    expired = [code for code, record in _pending.items() if record["expires_at"] <= now]
    for code in expired:
        _pending.pop(code, None)


def put(
    *,
    user_id: int,
    agent_id: Optional[str] = None,
    tokens: Dict[str, Any],
    client_id: str,
) -> str:
    """Park exchanged broker tokens and return the opaque code that redeems them.

    ``user_id`` is only the *hint* recorded by whoever started the OAuth flow;
    it carries no authority on its own and exists so :func:`pop`'s caller can
    compare it against the authenticated session.
    """
    code = secrets.token_urlsafe(32)
    now = time.monotonic()
    record: Dict[str, Any] = {
        "user_id": int(user_id),
        "agent_id": agent_id,
        "tokens": dict(tokens),
        "client_id": str(client_id),
        "expires_at": now + PENDING_TTL_SECONDS,
    }
    with _lock:
        _purge_expired_locked(now)
        while len(_pending) >= MAX_PENDING_LINKS:
            _pending.popitem(last=False)
        _pending[code] = record
    return code


def pop(code: str) -> Optional[Dict[str, Any]]:
    """Remove and return the record for ``code``, or ``None`` if it is gone.

    Single use: a code that is redeemed, expired, or never existed all return
    ``None``, and a successful redemption cannot be repeated.
    """
    if not code:
        return None
    now = time.monotonic()
    with _lock:
        _purge_expired_locked(now)
        return _pending.pop(code, None)


def clear() -> None:
    """Drop every pending record. Test-support hook; not used in request paths."""
    with _lock:
        _pending.clear()
