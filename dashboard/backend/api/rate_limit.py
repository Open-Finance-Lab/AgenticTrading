"""Lightweight in-process rate limiting for public, unauthenticated routes.

This is a **best-effort abuse control, not a security boundary**:

* State is per-process — it resets on restart and is not shared across multiple
  workers/replicas. On a single-instance deployment (the current Render setup)
  that is adequate; a multi-replica deployment would need a shared store.
* Keys are derived from client-supplied headers (session/browser id), which a
  determined attacker can rotate, falling back to the peer host.

It exists to bound *accidental / naive* abuse of public endpoints that spend
real resources (operator LLM credits, unbounded DB writes) without requiring
auth. Endpoints that need real protection must add authentication.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Callable, Deque, Dict

from fastapi import Request


class FixedWindowRateLimiter:
    """Sliding-window counter: at most ``max_events`` per ``window_seconds`` per key.

    ``clock`` is injectable so tests are deterministic (no wall-clock sleeps).
    """

    def __init__(
        self,
        max_events: int,
        window_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ) -> None:
        if max_events < 1:
            raise ValueError("max_events must be >= 1")
        self.max_events = max_events
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        self._clock = clock
        self._events: Dict[str, Deque[float]] = {}

    def allow(self, key: str) -> bool:
        """Record an attempt for ``key``; return True iff it is within the limit.

        A rejected attempt does NOT extend the window (we don't append its
        timestamp), so a client hammering the endpoint recovers exactly one
        window after its *allowed* burst, not after it stops trying.
        """
        now = self._clock()
        cutoff = now - self.window_seconds
        q = self._events.get(key)
        if q is None:
            # New key. Opportunistically reclaim fully-expired buckets before
            # growing so total key-cardinality stays bounded over the process
            # lifetime. (The previous ``del`` guard here was dead code: it needed
            # both ``len(q) >= max_events`` and ``q`` empty, impossible when
            # max_events >= 1, so empty buckets were never reclaimed.)
            if len(self._events) >= self.max_keys:
                self._sweep(cutoff)
            q = deque()
            self._events[key] = q
        else:
            while q and q[0] <= cutoff:
                q.popleft()
        if len(q) >= self.max_events:
            return False
        q.append(now)
        return True

    def _sweep(self, cutoff: float) -> None:
        """Drop keys whose entire window has expired (newest event older than the
        window) or that hold no events — bounds memory regardless of how many
        distinct keys are ever seen."""
        stale = [k for k, dq in self._events.items() if not dq or dq[-1] <= cutoff]
        for k in stale:
            del self._events[k]

    def reset(self) -> None:
        """Clear all state (used by tests and between logical sessions)."""
        self._events.clear()

    def retry_after_seconds(self, key: str) -> int:
        """Seconds until ``key`` can pass ``allow`` again (at least 1).

        Uses the oldest recorded event: once it ages out of the window the
        client recovers one slot. Empty / unknown keys fall back to the full
        window width.
        """
        q = self._events.get(key)
        if not q:
            return max(1, int(math.ceil(self.window_seconds)))
        remaining = q[0] + self.window_seconds - self._clock()
        return max(1, int(math.ceil(remaining)))


def client_key(request: Request) -> str:
    """Best-effort stable key for an anonymous client.

    Prefers the browser/session id the rest of the anonymous app already uses,
    else falls back to the peer host. Prefixed so an id can never collide with
    an ip.
    """
    hdr = request.headers
    ident = hdr.get("x-browser-id") or hdr.get("x-session-id")
    if ident and ident.strip():
        return f"id:{ident.strip()}"
    host = request.client.host if request.client else "unknown"
    return f"ip:{host}"
