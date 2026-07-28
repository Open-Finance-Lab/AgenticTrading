"""
FastAPI backend for agentic trading dashboard.
Serves equity curves, run metadata, and comparison data.

This module is the application composition root: it creates the FastAPI app,
configures middleware, registers routers, wires startup hooks, and serves the
frontend. Backend API route bodies live in ``dashboard.backend.api.routers.*``.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from pathlib import Path

import dashboard.backend.database as _database
from dashboard.backend.paths import FRONTEND_DIR
from dashboard.backend.middleware import SessionMiddleware, CSPHeaderMiddleware
from dashboard.backend.api.router import api_router
from dashboard.backend.api.routers.paper_trading import router as paper_trading_router
from dashboard.backend.api.routers.health import router as health_router
from dashboard.backend.api.routers.backtests import router as backtests_router
from dashboard.backend.api.routers.config import router as config_router
from dashboard.backend.api.routers.market import router as market_router
from dashboard.backend.api.routers.admin import router as admin_router
from dashboard.backend.api.v2.errors import ApiError, api_error_handler, validation_error_handler
from dashboard.backend.domain.backtesting.baselines.paper import create_paper_baselines_if_not_exists

# Re-exported from database.py. ``DB_PATH`` is used below; ``db`` is not
# referenced in this module at all -- it exists so tests can swap the shared
# connection with `monkeypatch.setattr(app_module, "db", temp_db)`
# (test_external_backtest_api.py, test_backtest_isolation.py). That reference
# is a *string*, so neither grep nor AST analysis sees it, and dropping the
# name as an "unused import" errors 9 tests. Bound explicitly so the
# re-export reads as deliberate rather than as a stale import.
db = _database.db
DB_PATH = _database.DB_PATH

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

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    # PATCH backs the agent Configure screen's Save. Frontend and API are
    # separate origins in prod, so a method absent here fails at the preflight
    # even though the route exists (test_cors_preflight_allows_every_routed_method).
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["content-type", "authorization", "x-session-id", "x-browser-id", "x-api-key", "accept"],
    # x-ratelimit-*/retry-after: the v2 spec promises these to agent clients;
    # browsers strip headers absent from Access-Control-Expose-Headers.
    expose_headers=["content-type", "cache-control", "etag", "x-session-id",
                    "x-ratelimit-limit", "x-ratelimit-remaining",
                    "x-ratelimit-reset", "retry-after"],
    max_age=3600,
)

# Add session middleware (selective: backtest routes only)
app.add_middleware(SessionMiddleware)

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
    from pathlib import Path
    import sqlite3
    
    print("🚀 Starting API server...")
    
    # DEBUG: Database location
    print("\n=== 📂 DATABASE DEBUG ===")
    print(f"CWD: {os.getcwd()}")
    print(f"Database path: {DB_PATH}")
    print(f"Database exists: {DB_PATH.exists()}")
    
    # Check database content
    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM agent_runs")
            count = cursor.fetchone()[0]
            print(f"✅ Database has {count} runs")
            
            if count > 0:
                cursor.execute("SELECT run_id, agent_name FROM agent_runs LIMIT 3")
                print("Sample runs:")
                for row in cursor.fetchall():
                    print(f"  • {row[0]}: {row[1]}")
            conn.close()
        except Exception as e:
            print(f"❌ Database error: {e}")
    else:
        print("❌ Database NOT FOUND")
    
    print("=== END DATABASE DEBUG ===")
    
    # Also check database directly
    print("\nDirect database check at startup:")
    try:
        import sqlite3
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM agent_runs")
        count = cursor.fetchone()[0]
        cursor.execute("SELECT run_id, session_id FROM agent_runs LIMIT 3")
        rows = cursor.fetchall()
        print(f"Total runs: {count}")
        for run_id, session_id in rows:
            print(f"  - {run_id}: session={session_id}")
        conn.close()
    except Exception as e:
        print(f"Error: {e}")
    print()
    
    print("📊 Backtesting: LLM-powered agent via dashboard/scripts/backtest_hourly_agent.py")
    if os.getenv("ANTHROPIC_API_KEY"):
        print("✅ ANTHROPIC_API_KEY detected - LLM trading enabled")
    else:
        print("⚠️ ANTHROPIC_API_KEY not set - LLM trading disabled")
    print("📊 Paper Trading: Baselines initialized on startup...")
    
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
        from dashboard.backend.domain.runs.service import start_reaper
        start_reaper()
        print("🧹 Run reaper started")
    except Exception as e:
        print(f"⚠️ Run reaper start error: {e}")


# ============================================================================
# Static Frontend Routes (must come AFTER API routes to not intercept them)
# ============================================================================

frontend_path = FRONTEND_DIR

@app.get("/", include_in_schema=False)
async def serve_root():
    """Serve marketing landing page."""
    return FileResponse(frontend_path / "index.html")


@app.get("/app", include_in_schema=False)
async def serve_app():
    """Serve the main dashboard application."""
    return FileResponse(frontend_path / "app.html")


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

@app.get("/styles.css", include_in_schema=False)
async def serve_styles():
    """Serve styles.css."""
    return FileResponse(frontend_path / "styles.css", media_type="text/css")

@app.get("/app.js", include_in_schema=False)
async def serve_app_js():
    """Serve app.js."""
    return FileResponse(frontend_path / "app.js", media_type="text/javascript")

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
