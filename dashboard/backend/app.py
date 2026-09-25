"""
FastAPI backend for agentic trading dashboard.
Serves equity curves, run metadata, and comparison data.

This module is the application composition root: it creates the FastAPI app,
configures middleware, registers routers, wires startup hooks, and serves the
frontend. Backend API route bodies live in ``dashboard.backend.api.routers.*``.
"""

from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from pathlib import Path
import asyncio
import os

import dashboard.backend.database as _database
from dashboard.backend.paths import FRONTEND_DIR
from dashboard.backend.middleware import SessionMiddleware, CSPHeaderMiddleware
from dashboard.backend.csrf import CsrfMiddleware
from dashboard.backend.api.router import api_router
from dashboard.backend.api.routers.paper_trading import router as paper_trading_router
from dashboard.backend.api.routers.health import router as health_router
from dashboard.backend.api.routers.backtests import router as backtests_router
from dashboard.backend.api.routers.config import router as config_router
from dashboard.backend.api.routers.market import router as market_router
from dashboard.backend.api.routers.admin import router as admin_router
from dashboard.backend.api.v2.errors import ApiError, api_error_handler, validation_error_handler
from dashboard.backend.domain.backtesting.baselines.paper import create_paper_baselines_if_not_exists

# Re-exported from database.py. ``db`` is not referenced anywhere else in
# this module -- it exists so tests can swap the shared connection with
# `monkeypatch.setattr(app_module, "db", temp_db)`
# (test_external_backtest_api.py, test_backtest_isolation.py). That reference
# is a *string*, so neither grep nor AST analysis sees it, and dropping the
# name as an "unused import" errors 9 tests. Bound explicitly so the
# re-export reads as deliberate rather than as a stale import.
db = _database.db

# Load .env from project root (ANTHROPIC_API_KEY, ALPACA_*)
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        pass

# Initialize FastAPI app
app = FastAPI(
    title="Agentic Trading Dashboard API",
    description="Backend API for backtesting and paper trading equity curves",
    version="1.0.0"
)

# Uniform error envelope for the typed /api/v2 surface (spec §5.4). The same
# handler keeps legacy routes on FastAPI's default {"detail": ...} shape; both
# branches sanitize non-finite floats so an ``Infinity`` payload stays a 422.
app.add_exception_handler(ApiError, api_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)


def _cors_allow_origins() -> list[str]:
    """Browser origins allowed to call this API cross-origin.

    Same-origin Vercel→rewrite traffic does not need CORS. External agent
    browsers and legacy split-origin frontends still do. When
    ``ATL_FRONTEND_ORIGINS`` is unset we keep ``*`` (credentials remain
    disabled). When set, only that allowlist (+ local dev hosts) is used —
    never combine ``*`` with ``allow_credentials=True``.
    """
    raw = (os.getenv("ATL_FRONTEND_ORIGINS") or "").strip()
    locals_ = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    if not raw:
        return ["*"]
    origins = [part.strip().rstrip("/") for part in raw.split(",") if part.strip()]
    for host in locals_:
        if host not in origins:
            origins.append(host)
    return origins


_CORS_ORIGINS = _cors_allow_origins()
# Browsers refuse credentials with Access-Control-Allow-Origin: *. Only enable
# credentialed CORS when ATL_FRONTEND_ORIGINS pins an explicit allowlist
# (same-origin Vercel rewrites do not need CORS at all).
_CORS_ALLOW_CREDENTIALS = _CORS_ORIGINS != ["*"]

# Compress JSON responses when the client accepts it. Equity curves and agent
# lists run to hundreds of KB uncompressed; minimum_size skips tiny payloads
# where the gzip header would cost more than it saves.
#
# Added FIRST on purpose, which makes it the INNERMOST layer. add_middleware
# prepends, so the last one added wraps everything. GZip must sit below
# SessionMiddleware because that is a BaseHTTPMiddleware: it re-emits every
# response as a stream, and GZip only honours minimum_size on the non-streaming
# branch. Stacked above Session it therefore compressed *every* response
# including a 15-byte {"status": "ok"} -- inflating it and burning event-loop
# CPU -- with minimum_size silently inert (test_small_response_is_not_gzipped).
#
# compresslevel=6, not Starlette's default of 9: Starlette compresses inline on
# the event loop (no threadpool hop), so the cost is paid by every concurrent
# request -- the loop is single-threaded no matter how many cores the instance
# has, so a stall here delays every other in-flight response. Measured on a
# 462 KB equity curve, level 9 costs 25.0 ms for 20.1% of original while
# level 6 costs 6.4 ms for 20.8% -- 4x the event-loop stall to save 0.7
# percentage points.
#
# ⚠ The original argument leaned on the prod CPU being *slower* than wherever
# those numbers were taken, which amplified the stall. Prod moved to a full CPU
# on 2026-09-11, so that amplification is gone and the case for 6 over 9 is
# weaker than it was, not stronger. The ratio should survive the move; the
# absolute gap wants re-measuring on the current plan before anyone leans on it
# again.
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=_CORS_ALLOW_CREDENTIALS,
    # PATCH backs the agent Configure screen's Save. Cross-origin callers
    # (and pre-proxy split-origin frontends) need the method listed here or
    # the browser fails at preflight even though the route exists
    # (test_cors_preflight_allows_every_routed_method).
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["content-type", "authorization", "x-session-id", "x-browser-id", "x-api-key", "x-csrf-token", "accept"],
    # x-ratelimit-*/retry-after: the v2 spec promises these to agent clients;
    # browsers strip headers absent from Access-Control-Expose-Headers.
    # x-ratelimit-scope: a forgot-password 429 says whether the address or the
    # client was refused, and the login modal's code step reads it.
    expose_headers=["content-type", "cache-control", "etag", "x-session-id",
                    "x-ratelimit-limit", "x-ratelimit-remaining",
                    "x-ratelimit-reset", "x-ratelimit-scope", "retry-after"],
    max_age=3600,
)

# Add session middleware (selective: backtest routes only)
app.add_middleware(SessionMiddleware)

# Cookie-session CSRF (session cookie ⇒ double-submit; API-key-only skips)
app.add_middleware(CsrfMiddleware)

# Versioned REST API (auth, future teams/contest/config)
app.include_router(api_router)

# Paper Trading routes (unprefixed: external paths remain /paper/...)
app.include_router(paper_trading_router)

# Backend API routes (unprefixed: external paths unchanged — /health, /ticker,
# /backtest/*, /api/backtest/*, /runs*, /compare, /config/defaults, /admin/*)
app.include_router(health_router)
app.include_router(backtests_router)
app.include_router(config_router)
app.include_router(market_router)
app.include_router(admin_router)

# CSP Middleware: Permit Chart.js and inline scripts (for development)
app.add_middleware(CSPHeaderMiddleware)

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize API server."""
    import os

    print("🚀 Starting API server...")

    print("📊 Backtesting: LLM-powered agent via dashboard/scripts/backtest_hourly_agent.py")
    if os.getenv("ANTHROPIC_API_KEY"):
        print("✅ ANTHROPIC_API_KEY detected - LLM trading enabled")
    else:
        print("⚠️ ANTHROPIC_API_KEY not set - LLM trading disabled")
    print("📊 Paper Trading: Baselines initialized on startup...")

    try:
        from dashboard.backend.domain.credits.service import credits_service

        report = await asyncio.to_thread(
            credits_service.backfill_default_signup_credits
        )
        print(
            "[credits] welcome campaign "
            f"total={report['total']} granted={report['granted']} "
            f"existing={report['existing']} failed={report['failed']}"
        )
    except Exception:  # noqa: BLE001 - startup must keep serving for a retry
        print("WARNING: credits.welcome_backfill_failed")
    
    # Initialize paper trading baselines (non-blocking)
    import threading
    
    # Initialize paper trading baselines (non-blocking)
    import threading
    
    def init_paper_baselines():
        """Background initialization - create paper trading baselines only."""
        try:
            create_paper_baselines_if_not_exists()
        except Exception as e:
            print(f"⚠️ Paper baseline initialization error: {e}")
    
    thread = threading.Thread(target=init_paper_baselines, daemon=True)
    thread.start()
    # Don't wait - server starts immediately

    def init_daily_leaderboard():
        """Background: baselines + optional LLM deploy for the daily board."""
        try:
            from dashboard.backend.domain.leaderboard.service import (
                maybe_schedule_daily_leaderboard_refresh,
            )

            if maybe_schedule_daily_leaderboard_refresh():
                print("📊 Daily Leaderboard: background refresh started")
        except Exception as e:
            print(f"⚠️ Daily leaderboard initialization error: {e}")

    threading.Thread(target=init_daily_leaderboard, daemon=True).start()

    # On-disk bar cache: name the state at boot, matching the
    # `<store> backend: …` convention, then warm the default windows on a
    # daemon thread so a cold instance does not charge the first visitor the
    # full bar fetch. Non-blocking by construction: it must never delay boot
    # or fail the health check.
    # Wrapped like every other block in this hook, and for the same reason:
    # `startup_event` has no handler of its own, so anything raising here
    # skips everything BELOW it -- `recover_orphaned_runs` (protocol runs
    # orphaned by the previous process stay `running` forever) and
    # `register_reaper_sweep(reap_v2_runs)` (abandoned v2 runs keep holding
    # their concurrency slots). This was the one unguarded statement in the
    # hook, and the import is not inert: it pulls in pandas and `paths`, and
    # `alpaca_bars` already imports `bar_cache`, so the cycle edge is live.
    try:
        from dashboard.backend.infrastructure.market_data import bar_cache

        print(bar_cache.describe())
    except Exception as e:  # noqa: BLE001 - a cold cache is the status quo
        print(f"⚠️ bar cache: error: {e}")

    def bar_cache_background():
        """Background: reclaim last process's strays, then warm if armed."""
        try:
            from dashboard.backend.infrastructure.market_data import bar_cache

            # The stray sweep and the LRU pass otherwise run ONLY from inside
            # `write_many`, so a deployment whose writes all fail stops
            # reclaiming the `*.tmp` files its killed writers leave behind --
            # the state where reclaiming matters most. Boot is the one moment
            # guaranteed to arrive without a successful write in front of it,
            # and on a mounted `ATL_BAR_CACHE_DIR` it is where the previous
            # process's leftovers are. Cheap: one `scandir` over a directory
            # holding tens of entries.
            if bar_cache.enabled():
                bar_cache.enforce_size_cap()
        except Exception as e:  # noqa: BLE001 - eviction must never fail boot
            print(f"⚠️ bar cache: sweep error: {e}")
        try:
            from dashboard.backend.infrastructure.market_data.bar_cache_warm import (
                warm_bar_cache,
            )

            warm_bar_cache()
        except Exception as e:  # noqa: BLE001 - a cold cache is the status quo
            # "bar cache warm:", lowercase, like every other line this
            # feature prints: the live-call detector greps that exact string
            # to prove the suite makes no billable Alpaca calls, and a line
            # it does not match is a proof this handler can blind.
            print(f"⚠️ bar cache warm: error: {e}")

    threading.Thread(target=bar_cache_background, daemon=True).start()

    # Protocol run lifecycle: fail runs orphaned by the previous process (their
    # in-memory engine sessions did not survive the restart) and start the
    # background reaper that drains/evicts abandoned runs. Kept in separate
    # try/except blocks so a recovery failure can't prevent the reaper starting.
    try:
        from dashboard.backend.domain.runs.service import recover_orphaned_runs
        recovered = recover_orphaned_runs()
        if recovered:
            print(f"🧹 Recovered {recovered} orphaned run(s) → failed")
    except Exception as e:
        print(f"⚠️ Orphaned-run recovery error: {e}")

    try:
        # Composition-root wiring (the domain reaper must not import api/*):
        # each reaper pass also sweeps the v2 registry — drains abandoned v2
        # runs, heartbeats live ones, archives terminal backends.
        from dashboard.backend.api.v2.runs import reap_v2_runs
        from dashboard.backend.domain.runs.service import register_reaper_sweep
        register_reaper_sweep(reap_v2_runs)
        print("🧹 v2 run sweep registered with the reaper")
    except Exception as e:
        print(f"⚠️ v2 sweep registration error: {e}")

    try:
        # Same reaper pass also TTL-evicts terminal legacy /api/v1/backtest/*
        # sessions, which have no registry of their own to be walked by.
        from dashboard.backend.domain.backtesting.external_run_service import (
            sweep_terminal_sessions,
        )
        from dashboard.backend.domain.runs.service import register_reaper_sweep
        register_reaper_sweep(sweep_terminal_sessions)
        print("🧹 legacy session sweep registered with the reaper")
    except Exception as e:
        print(f"⚠️ legacy session sweep registration error: {e}")


    try:
        from dashboard.backend.domain.analytics.maintenance import (
            run_analytics_maintenance,
        )
        from dashboard.backend.domain.runs.service import register_reaper_sweep

        register_reaper_sweep(run_analytics_maintenance)
        print("🧹 Analytics maintenance sweep registered with the reaper")
    except Exception as e:
        print(
            "WARNING: analytics.maintenance_registration_failed "
            f"category={type(e).__name__}"
        )

    try:
        from dashboard.backend.domain.analytics.instrumentation import (
            disable_synchronous_projection,
        )

        # PR 0: analytics_service now builds with project_snapshots=False, so
        # record_server_event no longer recomputes synchronously. Without
        # this call, instrumentation.py's own fallback guard -- written for a
        # caller-supplied service that never learned to project -- would
        # still call recalculate_user_snapshots per accepted event, moving
        # the burner one module over instead of killing it.
        disable_synchronous_projection()
        print("🧹 Analytics snapshot projection left off the request path")
    except Exception as e:
        print(
            "WARNING: analytics.snapshot_projection_disable_failed "
            f"category={type(e).__name__}"
        )

    try:
        # Admin layer redesign PR A (design D23, SS6.9): the daily-facts job
        # runs on its own thread, not as a reaper sweep -- a whole-population
        # batch across three databases on the heartbeat thread would let a
        # slow analytics night mark live runs as orphaned. The worker also
        # owns rollup_day and the retention coordinator now, and runs the
        # idempotent user_activity seed + history copy once before ticking.
        from dashboard.backend.domain.analytics.daily_job import (
            start_daily_facts_worker,
        )
        from dashboard.backend.domain.analytics.facts_migration import (
            run_startup_migrations,
        )
        run_startup_migrations()
        start_daily_facts_worker()
        print("📊 Analytics daily-facts worker started")
    except Exception as e:
        print(f"⚠️ Analytics daily-facts worker start error: {e}")

    try:
        from dashboard.backend.domain.runs.service import start_reaper
        start_reaper()
        print("🧹 Run reaper started")
    except Exception as e:
        print(f"⚠️ Run reaper start error: {e}")


# ============================================================================
# Static Frontend Routes (must come AFTER API routes to not intercept them)
# ============================================================================

frontend_path = FRONTEND_DIR

# HTML/JS/CSS always revalidate: the local console bumps ?v= per change, and
# browsers heuristic-cache FileResponses that carry no Cache-Control header,
# which is how Chrome kept serving a pre-autologin app.js (design N2/PR2).
NO_CACHE = {"Cache-Control": "no-cache"}

@app.get("/", include_in_schema=False)
async def serve_root():
    """Serve marketing landing page."""
    return FileResponse(frontend_path / "index.html")


@app.get("/app", include_in_schema=False)
async def serve_app(request: Request):
    """Serve the main dashboard application.

    The admin console moved to /admin (design D5), so the old ?view=admin deep
    links hand off to the absorbed section instead of rendering the legacy
    console. 307 rather than 308: the query string is not preserved verbatim —
    adminTab becomes a hash route and adminUserQuery becomes the #account
    ?user= hand-off.
    """
    if request.query_params.get("view") == "admin":
        tab = request.query_params.get("adminTab") or "users"
        route = {"users": "account", "providers": "providers", "activity": "activity"}.get(tab, "account")
        target = f"/admin#{route}"
        user_query = request.query_params.get("adminUserQuery")
        if user_query:
            target += f"?user={quote(str(user_query), safe='')}"
        return RedirectResponse(url=target, status_code=307)
    return FileResponse(frontend_path / "app.html", headers=NO_CACHE)


@app.get("/app/", include_in_schema=False)
async def serve_app_trailing_slash(request: Request):
    """Redirect /app/ → /app (method-preserving 308).

    app.html references its assets with relative paths (``styles.css``,
    ``app.js``, ``images/...``). Served from ``/app/`` a browser resolves those
    against the ``/app/`` base (``/app/styles.css`` → 404), so the dashboard
    renders unstyled. Redirecting to ``/app`` makes relative assets resolve
    against root.

    Preserve the query string: the frontend deep-links via query params on this
    route (``?auth=login``, ``?view=paper``, ``?mode=…`` from generateShareURL /
    openAuthFromUrl), which a bare ``/app`` redirect would drop.
    """
    target = "/app"
    if request.url.query:
        target = f"{target}?{request.url.query}"
    return RedirectResponse(url=target, status_code=308)

@app.get("/favicon.svg", include_in_schema=False)
async def serve_favicon_svg():
    """Serve the real SVG favicon (frontend/favicon.svg)."""
    favicon_path = frontend_path / "favicon.svg"
    if not favicon_path.exists():
        raise HTTPException(status_code=404, detail="Favicon not found")
    return FileResponse(favicon_path, media_type="image/svg+xml")


@app.get("/favicon.ico", include_in_schema=False)
async def serve_favicon():
    """Serve the PNG logo for legacy /favicon.ico requests."""
    favicon_path = frontend_path / "images" / "atltransparent.png"
    if not favicon_path.exists():
        raise HTTPException(status_code=404, detail="Favicon not found")
    return FileResponse(favicon_path, media_type="image/png")


@app.get("/assets/{file_name}", include_in_schema=False)
async def serve_landing_assets(file_name: str):
    """Serve landing page Vite build assets."""
    if "/" in file_name or "\\" in file_name:
        raise HTTPException(status_code=404, detail="Asset not found")

    asset_path = (frontend_path / "assets" / file_name).resolve()
    assets_dir = (frontend_path / "assets").resolve()
    if not asset_path.is_file() or assets_dir not in asset_path.parents:
        raise HTTPException(status_code=404, detail="Asset not found")

    ext = asset_path.suffix.lower()
    media_types = {
        ".js": "text/javascript",
        ".css": "text/css",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml",
        ".woff": "font/woff",
        ".woff2": "font/woff2",
    }
    return FileResponse(asset_path, media_type=media_types.get(ext, "application/octet-stream"))


@app.get("/strategy", include_in_schema=False)
async def serve_strategy_viewer():
    """Serve the standalone strategy viewer (reads ?code=... client-side)."""
    return FileResponse(frontend_path / "strategy.html")


@app.get("/admin", include_in_schema=False)
async def serve_admin():
    """Serve the admin console shell (dashboard/frontend/admin.html).

    The HTML carries no data: every number arrives from a require_admin-gated
    /api/admin/* route after js/admin-shell.js has probed /api/auth/me. That
    probe is a courtesy redirect, not the gate -- Vercel serves this same file
    as a static asset with no session access, so a server-side gate here would
    exist on one host only (design D7). This route is for the Render origin;
    Vercel serves the file through ``cleanUrls``.
    """
    return FileResponse(frontend_path / "admin.html")


@app.get("/admin.css", include_in_schema=False)
async def serve_admin_css():
    """Serve admin.css beside /styles.css (every static file is an explicit route)."""
    return FileResponse(frontend_path / "admin.css", media_type="text/css")


@app.get("/admin-console.css", include_in_schema=False)
async def serve_admin_console_css():
    """Serve the absorbed old-console component styles (design D5)."""
    return FileResponse(frontend_path / "admin-console.css", media_type="text/css")


@app.get("/admin-analytics", include_in_schema=False)
async def redirect_admin_analytics(request: Request):
    """308 /admin-analytics → /admin for one release, then this route goes.

    Preserve the query string the way /app/ → /app does: the page keeps its
    range and filters in the query (``?range=1M&group=organic``), and a bare
    redirect would drop them. The hash (``#users/42``) never reaches the
    server; browsers carry it across a redirect whose Location has none.
    """
    target = "/admin"
    if request.url.query:
        target = f"{target}?{request.url.query}"
    return RedirectResponse(url=target, status_code=308)

@app.get("/styles.css", include_in_schema=False)
async def serve_styles():
    """Serve styles.css."""
    return FileResponse(frontend_path / "styles.css", media_type="text/css", headers=NO_CACHE)

@app.get("/app.js", include_in_schema=False)
async def serve_app_js():
    """Serve app.js."""
    return FileResponse(frontend_path / "app.js", media_type="text/javascript", headers=NO_CACHE)

@app.get("/home-page.js", include_in_schema=False)
async def serve_home_page_js():
    """Serve home-page.js for the Home tab mock live UI."""
    return FileResponse(frontend_path / "home-page.js", media_type="text/javascript")

@app.get("/home-news-signals.js", include_in_schema=False)
async def serve_home_news_signals_js():
    """Serve home-news-signals.js for the Home tab news & signals panel."""
    return FileResponse(frontend_path / "home-news-signals.js", media_type="text/javascript")

@app.get("/js/{file_name}", include_in_schema=False)
async def serve_js_module(file_name: str):
    """Serve js/*.js modules (e.g. leaderboard.js)."""
    if not file_name.endswith(".js") or "/" in file_name or "\\" in file_name:
        raise HTTPException(status_code=404, detail="Script not found")

    script_path = (frontend_path / "js" / file_name).resolve()
    js_dir = (frontend_path / "js").resolve()
    if not script_path.is_file() or js_dir not in script_path.parents:
        raise HTTPException(status_code=404, detail="Script not found")

    return FileResponse(script_path, media_type="text/javascript")


@app.get("/data/{file_name}", include_in_schema=False)
async def serve_frontend_data(file_name: str):
    """Serve allowlisted static JSON used by the Home Get Started demo."""
    if file_name != "home-demo-run.json" or "/" in file_name or "\\" in file_name:
        raise HTTPException(status_code=404, detail="Data file not found")
    data_path = (frontend_path / "data" / file_name).resolve()
    data_dir = (frontend_path / "data").resolve()
    if not data_path.is_file() or data_dir not in data_path.parents:
        raise HTTPException(status_code=404, detail="Data file not found")
    return FileResponse(data_path, media_type="application/json")


@app.get("/market-events/{file_name}", include_in_schema=False)
async def serve_market_events_js(file_name: str):
    """Serve market-events/*.js modules for the Live Market Events panel."""
    if not file_name.endswith(".js") or "/" in file_name or "\\" in file_name:
        raise HTTPException(status_code=404, detail="Script not found")

    script_path = (frontend_path / "market-events" / file_name).resolve()
    market_events_dir = (frontend_path / "market-events").resolve()
    if not script_path.is_file() or market_events_dir not in script_path.parents:
        raise HTTPException(status_code=404, detail="Script not found")

    return FileResponse(script_path, media_type="text/javascript")

@app.get("/images/{file_name}", include_in_schema=False)
async def serve_image(file_name: str):
    """Serve image files from the images directory."""
    if "/" in file_name or "\\" in file_name:
        raise HTTPException(status_code=404, detail="Image not found")

    image_path = (frontend_path / "images" / file_name).resolve()
    images_dir = (frontend_path / "images").resolve()
    if not image_path.is_file() or not image_path.is_relative_to(images_dir):
        raise HTTPException(status_code=404, detail="Image not found")

    # Determine media type based on file extension
    ext = image_path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
    }
    media_type = media_types.get(ext, "application/octet-stream")
    
    return FileResponse(image_path, media_type=media_type)


# ============================================================================
# Run the app
# ============================================================================

if __name__ == "__main__":
    # Canonical startup is ``uvicorn dashboard.backend.app:app``. This direct
    # invocation is a deprecated compatibility path; reference the app by its
    # canonical import string so the reloader resolves the same module identity.
    import os
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("dashboard.backend.app:app", host="0.0.0.0", port=port, reload=True)
