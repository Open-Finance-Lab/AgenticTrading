"""Lightweight in-process rate limiting for public, unauthenticated routes.

This is a **best-effort abuse control, not a security boundary**:

* State is per-process — it resets on restart and is not shared across multiple
  workers/replicas. On a single-instance deployment (the current Render setup)
  that is adequate; a multi-replica deployment would need a shared store.
* Keys are derived from client-supplied headers (session/browser id, then
  ``X-Forwarded-For``), which a determined attacker can rotate, falling back to
  the peer host.

It exists to bound *accidental / naive* abuse of public endpoints that spend
real resources (operator LLM credits, unbounded DB writes) without requiring
auth. Endpoints that need real protection must add authentication.

**The auth routes are the one exception to that last sentence**, because
``POST /api/auth/login`` *is* the authentication endpoint — it has no stronger
gate to defer to. So do not read the per-client budgets there as brute-force
protection: every key this module can compute comes from a header the caller
controls, and an attacker who rotates it gets a fresh budget for free. What
actually bounds password guessing is the **per-email failure counter** in
``api/auth.py``, which is keyed on the account being attacked rather than on
anything the attacker supplies. The per-client budgets bound accidental abuse
and cap the bcrypt work one naive client can queue; nothing more.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Callable, Deque, Dict

from fastapi import HTTPException, Request

from dashboard.backend.infrastructure.request_metadata import (
    _MAX_IP_KEY_LEN,
    client_ip,
)


class FixedWindowRateLimiter:
    """Sliding-window counter: at most ``max_events`` per ``window_seconds`` per key.

    ``clock`` is injectable so tests are deterministic (no wall-clock sleeps).

    ``max_events=0`` disables the limiter entirely (every call is allowed and
    nothing is recorded), so an operator can switch a budget off through config
    without a deploy. Negative values are still a configuration error.

    **Thread-safe.** Every route that limits here is a plain ``def`` handler,
    which FastAPI runs on the threadpool — so concurrent requests hit one
    limiter instance from different threads, and the buckets are mutated on
    every call (``_pruned`` popleft()s from inside ``check``, not only inside
    ``record``). Two threads interleaving there can drop the same entry twice
    or lose an append. ``allow()`` is atomic for the same reason: as separate
    ``check`` then ``record`` calls it was a check-then-act race that let N
    concurrent callers all observe the last free slot.
    """

    def __init__(
        self,
        max_events: int,
        window_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ) -> None:
        if max_events < 0:
            raise ValueError("max_events must be >= 0 (0 disables the limiter)")
        self.max_events = max_events
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        self._clock = clock
        self._events: Dict[str, Deque[float]] = {}
        # Plain Lock, not RLock: the public methods below acquire it exactly
        # once and delegate to ``_*_locked`` helpers that never re-acquire.
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.max_events > 0

    def _pruned(self, key: str) -> Deque[float] | None:
        """``key``'s bucket with expired events dropped, or None if unknown.

        Pruning only removes entries the window has already released, so calling
        this from a read-only method does not change any answer it could give.
        """
        q = self._events.get(key)
        if q is None:
            return None
        cutoff = self._clock() - self.window_seconds
        while q and q[0] <= cutoff:
            q.popleft()
        return q

    def check(self, key: str) -> bool:
        """True iff ``key`` is currently under its limit. Consumes nothing.

        Pair this with :meth:`record` when the budget should be spent on some
        outcomes but not others — guarding an expensive call without charging
        for its *successful* result. :meth:`allow` is the combined form.
        """
        if not self.enabled:
            return True
        with self._lock:
            return self._check_locked(key)

    def _check_locked(self, key: str) -> bool:
        q = self._pruned(key)
        return q is None or len(q) < self.max_events

    def record(self, key: str) -> None:
        """Spend one unit of ``key``'s budget.

        Never grows a bucket past ``max_events``: the cap is what bounds memory
        per key, and a caller that records without checking must not defeat it.
        """
        if not self.enabled:
            return
        with self._lock:
            self._record_locked(key)

    def _record_locked(self, key: str) -> None:
        now = self._clock()
        q = self._pruned(key)
        if q is None:
            # New key. Opportunistically reclaim fully-expired buckets before
            # growing so total key-cardinality stays bounded over the process
            # lifetime. (The previous ``del`` guard here was dead code: it needed
            # both ``len(q) >= max_events`` and ``q`` empty, impossible when
            # max_events >= 1, so empty buckets were never reclaimed.)
            if len(self._events) >= self.max_keys:
                self._sweep(now - self.window_seconds)
            q = deque()
            self._events[key] = q
        if len(q) >= self.max_events:
            return
        q.append(now)

    def allow(self, key: str) -> bool:
        """Record an attempt for ``key``; return True iff it is within the limit.

        A rejected attempt does NOT extend the window (we don't append its
        timestamp), so a client hammering the endpoint recovers exactly one
        window after its *allowed* burst, not after it stops trying.

        Check and record happen under one lock acquisition, so the last free
        slot goes to exactly one of N concurrent callers.
        """
        if not self.enabled:
            return True
        with self._lock:
            if not self._check_locked(key):
                return False
            self._record_locked(key)
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
        with self._lock:
            self._events.clear()

    def retry_after_seconds(self, key: str) -> int:
        """Seconds until ``key`` can pass ``allow`` again (at least 1).

        Uses the oldest *live* recorded event: once it ages out of the window
        the client recovers one slot. Prunes first, so this is correct when
        called on its own and not only straight after a rejecting ``allow``
        (which pruned as a side effect). Empty / unknown keys fall back to the
        full window width. Never exceeds the window.
        """
        if not self.enabled:
            return 1
        with self._lock:
            q = self._pruned(key)
            if not q:
                return max(1, math.ceil(self.window_seconds))
            remaining = q[0] + self.window_seconds - self._clock()
        return max(1, math.ceil(remaining))


def rate_limited_error(
    limiter: FixedWindowRateLimiter, key: str, detail: str
) -> HTTPException:
    """A 429 whose ``Retry-After`` reports when ``key``'s budget frees up.

    The one shared way to refuse on a limiter, so every limited route reports
    the same recovery signal instead of each router growing its own copy.
    """
    return HTTPException(
        status_code=429,
        detail=detail,
        headers={"Retry-After": str(limiter.retry_after_seconds(key))},
    )


def client_key(request: Request) -> str:
    """Best-effort stable key for an anonymous client.

    Prefers the browser/session id the rest of the anonymous app already uses,
    else falls back to the client IP. Prefixed so an id can never collide with
    an ip.
    """
    hdr = request.headers
    ident = hdr.get("x-browser-id") or hdr.get("x-session-id")
    if ident and ident.strip():
        return f"id:{ident.strip()}"
    return f"ip:{client_ip(request)}"
