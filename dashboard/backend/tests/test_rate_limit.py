"""FixedWindowRateLimiter: budget arithmetic and thread safety.

Every route that limits here is a plain ``def`` handler, which FastAPI runs on
the Starlette threadpool — so one limiter instance is hit from several threads
at once. The module carried no ``threading`` import at all, and its buckets are
mutated on *every* call (``_pruned`` popleft()s from inside ``check``, not only
inside ``record``), so the data structure itself was racing, not just the
check-then-act sequence above it.
"""

import threading

from dashboard.backend.api.rate_limit import FixedWindowRateLimiter


def test_allow_hands_the_last_slot_to_exactly_one_caller():
    """The check-then-act race, at the one budget that actually bounds abuse.

    ``allow()`` was ``check()`` then ``record()`` as two separate acquisitions,
    so N threads could all read the same under-limit count and all proceed. A
    budget of 3 handed out by 40 racing callers must still be 3.
    """
    limiter = FixedWindowRateLimiter(max_events=3, window_seconds=60)
    granted = []
    lock = threading.Lock()
    start = threading.Barrier(40)

    def contend():
        start.wait()
        if limiter.allow("shared"):
            with lock:
                granted.append(1)

    threads = [threading.Thread(target=contend) for _ in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(granted) == 3


def test_concurrent_check_and_record_never_exceed_the_budget():
    """Hammer one key from many threads; the bucket must stay bounded.

    ``record`` prunes, appends and (on a new key) sweeps the whole dict —
    interleaved with a ``check`` that is also popleft()ing from the same deque.
    Without a lock this corrupts silently: the failure mode is a bucket that
    over- or under-counts, not an exception, so it would surface in production
    as a budget that quietly stopped bounding anything.
    """
    limiter = FixedWindowRateLimiter(max_events=5, window_seconds=60)
    start = threading.Barrier(24)

    def hammer(i):
        start.wait()
        for _ in range(50):
            limiter.check(f"key-{i % 4}")
            limiter.record(f"key-{i % 4}")

    threads = [threading.Thread(target=hammer, args=(i,)) for i in range(24)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for i in range(4):
        assert len(limiter._events[f"key-{i}"]) <= limiter.max_events
        assert not limiter.check(f"key-{i}")


def test_rejected_attempts_do_not_extend_the_window():
    """A client hammering the endpoint recovers one window after its allowed
    burst, not after it gives up."""
    now = [1000.0]
    limiter = FixedWindowRateLimiter(
        max_events=2, window_seconds=60, clock=lambda: now[0]
    )
    assert limiter.allow("k") is True
    assert limiter.allow("k") is True
    for _ in range(10):
        now[0] += 1
        assert limiter.allow("k") is False

    now[0] = 1000.0 + 61
    assert limiter.allow("k") is True


def test_disabled_limiter_allows_everything():
    """0 disables — the MAX_ACTIVE_RUNS_GLOBAL convention, so an operator can
    switch a budget off through config without a deploy."""
    limiter = FixedWindowRateLimiter(max_events=0, window_seconds=60)
    assert limiter.enabled is False
    for _ in range(100):
        assert limiter.allow("k") is True
        assert limiter.check("k") is True
    assert limiter._events == {}
    assert limiter.retry_after_seconds("k") == 1
