"""Event-loop protection: blocking-I/O routes must run in the threadpool.

FastAPI runs ``async def`` handlers on the event loop itself; a synchronous
call inside one (``requests``, ``yfinance``, ``sqlite3``, sync ``psycopg``)
freezes every concurrent request server-wide. Plain ``def`` handlers run in
the Starlette threadpool, which contains the blocking to one worker thread.

Measured in prod before the fix: a /ticker quote-provider cache miss (~5-7s of
yfinance calls) inflated a concurrent /health from ~0.9s to ~6.3s.

The routers pinned here perform sync I/O in every handler and contain no
``await``, so ``def`` is both safe and required. New handlers added to these
modules must stay ``def`` (or move their blocking work off the loop first).
"""

import asyncio
import inspect
import time

import httpx
from fastapi.routing import APIRoute

from dashboard.backend.app import app

# Every module here does synchronous I/O (sqlite/psycopg/requests/yfinance)
# directly in its handler bodies. None of them contains an ``await``.
BLOCKING_IO_ROUTER_MODULES = {
    "dashboard.backend.api.routers.market",
    "dashboard.backend.api.routers.config",
    "dashboard.backend.api.routers.paper_trading",
    "dashboard.backend.api.routers.agents",
    "dashboard.backend.api.routers.agent_versions",
    "dashboard.backend.api.routers.portfolio",
    "dashboard.backend.api.routers.leaderboard",
    "dashboard.backend.api.routers.backtests",
    "dashboard.backend.api.routers.admin",
    "dashboard.backend.api.routers.admin_users",
    "dashboard.backend.api.routers.discord",
    "dashboard.backend.api.routers.external_backtest",
    "dashboard.backend.api.v2.leaderboard",
    # algo/* is the most expensive offender of the lot: /api/algo/chat calls
    # anthropic's synchronous client.messages.create(), which parks the loop for
    # as long as the model takes to answer. In the threadpool it costs one
    # worker thread instead of the whole server.
    "dashboard.backend.api.routers.algo",
}

# Handlers in modules that legitimately mix awaited and blocking work, so the
# whole module cannot be pinned. auth.py's OAuth callbacks and the two mail
# routes genuinely await; everything below does sync store I/O -- a network
# round trip against Postgres in prod, and for change_password two bcrypt
# rounds -- with no await in sight, so `def` is both safe and required.
#
# This set is also what stops the fix from being undone by accident: a handler
# only stays covered while it has no ``await``, so adding one (e.g. wrapping a
# single blocking call in ``asyncio.to_thread`` and leaving the rest inline)
# silently drops it out of the guard. That is exactly what #332 first did to
# change_password before #297's plain-``def`` fix was applied instead.
BLOCKING_IO_HANDLERS = {
    ("dashboard.backend.api.auth", "me"),
    ("dashboard.backend.api.auth", "logout"),
    ("dashboard.backend.api.auth", "logout_all"),
    ("dashboard.backend.api.auth", "change_password"),
    ("dashboard.backend.api.auth", "reset_password"),
    ("dashboard.backend.api.auth", "update_display_name"),
    ("dashboard.backend.api.auth", "get_email_change"),
    ("dashboard.backend.api.auth", "cancel_email_change"),
    ("dashboard.backend.api.auth", "set_avatar"),
    ("dashboard.backend.api.auth", "delete_avatar"),
    ("dashboard.backend.api.auth", "discord_oauth_start"),
}


def test_blocking_io_routers_have_no_async_handlers():
    offenders = sorted(
        f"{sorted(route.methods)} {route.path}"
        for route in app.routes
        if isinstance(route, APIRoute)
        and route.endpoint.__module__ in BLOCKING_IO_ROUTER_MODULES
        and inspect.iscoroutinefunction(route.endpoint)
    )
    assert offenders == [], (
        "async def handlers doing blocking I/O on the event loop "
        f"(declare them as plain def so they run in the threadpool): {offenders}"
    )


def test_covered_modules_actually_serve_routes():
    # Guards the invariant above against silently going vacuous if a router
    # module is renamed: every pinned module must still own at least one route.
    served = {
        route.endpoint.__module__
        for route in app.routes
        if isinstance(route, APIRoute)
    }
    missing = sorted(BLOCKING_IO_ROUTER_MODULES - served)
    assert missing == [], f"pinned modules no longer serve any route: {missing}"


def test_individually_pinned_handlers_are_sync():
    by_name = {
        (route.endpoint.__module__, route.endpoint.__name__): route.endpoint
        for route in app.routes
        if isinstance(route, APIRoute)
    }
    missing = sorted(k for k in BLOCKING_IO_HANDLERS if k not in by_name)
    assert missing == [], f"pinned handlers no longer serve a route: {missing}"

    offenders = sorted(
        f"{mod}.{name}"
        for (mod, name) in BLOCKING_IO_HANDLERS
        if inspect.iscoroutinefunction(by_name[(mod, name)])
    )
    assert offenders == [], (
        "these handlers do blocking store I/O with no await and must stay "
        f"plain def: {offenders}"
    )


def test_slow_ticker_fetch_does_not_stall_concurrent_requests(monkeypatch):
    """A slow quote fetch inside /ticker must not delay other requests.

    The clock starts BEFORE the ticker task gets its first scheduler slice: on
    a blocked event loop, any ``await`` between task creation and measurement
    would absorb the freeze and make the test pass vacuously.
    """
    import threading

    from dashboard.backend.api.routers import market

    handler_entered = threading.Event()

    def slow_quotes(symbols):
        handler_entered.set()
        # 1.0s blocked vs a 0.5s assertion: a blocked loop overshoots by 2x
        # while a healthy one answers in ~10ms, so the gap absorbs a loaded CI
        # runner without letting a real regression squeak through.
        time.sleep(1.0)
        return [
            {"symbol": s, "price": 1.0, "changePercent": 0.0, "timestamp": "t"}
            for s in symbols
        ]

    monkeypatch.setattr(market, "get_market_quotes", slow_quotes)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            # The ticker task is created first, so it takes the loop's next
            # slice; no await may sit between here and the health await.
            started = time.perf_counter()
            ticker_task = asyncio.create_task(client.get("/ticker?symbols=AAPL"))
            health = await client.get("/health")
            health_elapsed = time.perf_counter() - started
            ticker = await ticker_task
        assert ticker.status_code == 200
        assert ticker.json()["quotes"], "monkeypatched quotes should round-trip"
        assert health.status_code == 200
        assert handler_entered.is_set(), "ticker never reached its handler"
        return health_elapsed

    health_elapsed = asyncio.run(scenario())
    assert health_elapsed < 0.5, (
        f"/health took {health_elapsed:.3f}s while /ticker was fetching quotes "
        "-- the ticker handler is blocking the event loop"
    )


def test_auth_async_handlers_offload_store_io():
    """The two ``async def`` auth routes must not call the store inline.

    ``signup`` and ``login`` cannot be pinned by the guards above: they are
    exempt from BLOCKING_IO_HANDLERS for already containing an ``await``, and
    ``api.auth`` is not in BLOCKING_IO_ROUTER_MODULES because its OAuth
    callbacks genuinely await. That exemption is a hole a new blocking call
    walks straight into — a sync ``get_entitlements()`` was added to
    ``_auth_json`` (called from both) and no guard here saw it, even though
    both handlers already take deliberate care to keep bcrypt off the loop.

    A store call passed to ``asyncio.to_thread`` appears as a bare attribute
    reference, so it does not match; only an actual ``(``-suffixed call does.
    """
    import inspect
    import re

    from dashboard.backend.api import auth as auth_module

    offenders = {}
    for name in ("signup", "login"):
        handler = inspect.unwrap(getattr(auth_module, name))
        assert inspect.iscoroutinefunction(handler), (
            f"{name} is no longer async -- move it into BLOCKING_IO_HANDLERS "
            "instead, which is the stronger guarantee"
        )
        source = inspect.getsource(handler)
        found = re.findall(r"user_store\.\w+\s*\(", source)
        if found:
            offenders[name] = sorted(set(found))

    assert not offenders, (
        f"blocking store calls on the event loop: {offenders}. "
        "Wrap them in asyncio.to_thread (see _issue_session)."
    )
