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
    "dashboard.backend.api.routers.discord",
    "dashboard.backend.api.routers.external_backtest",
    "dashboard.backend.api.v2.leaderboard",
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
        time.sleep(0.6)
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
    assert health_elapsed < 0.4, (
        f"/health took {health_elapsed:.3f}s while /ticker was fetching quotes "
        "-- the ticker handler is blocking the event loop"
    )
