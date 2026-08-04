"""
Session Middleware: Selective session ID enforcement.
- Backtest routes: REQUIRE X-Session-Id header
- Paper trading routes (/paper/*): EXEMPT (global/shared)
- Static/health routes: EXEMPT
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import uuid

EXEMPT_PATHS = {
    '/',
    '/index.html',
    '/app',
    '/app/',
    '/app.html',
    '/favicon.ico',
    '/favicon.svg',
    '/health',
    '/api/health',
    '/ticker',
    '/config/defaults',  # Default configuration (public, no session needed)
    '/config/features',  # Optional frontend capabilities (public, read-only)
    '/compare',  # Public equity comparison (browser-friendly links)
    '/runs',  # Public backtest run listing
    '/strategy',  # Public strategy viewer page (shared links, no session needed)
}

EXEMPT_EXTENSIONS = {'.js', '.css', '.png', '.jpg', '.gif', '.svg', '.woff', '.woff2', '.ttf'}

def is_exempt(path: str) -> bool:
    """Check if path is exempt from session middleware."""
    if path in EXEMPT_PATHS:
        return True
    if any(path.endswith(ext) for ext in EXEMPT_EXTENSIONS):
        return True
    if path.startswith(('/static/', '/public/', '/assets/', '/js/', '/market-events/', '/images/')):
        return True
    return False

def is_paper_trading_route(path: str) -> bool:
    """Check if this is a paper trading route (allowed to skip session_id)."""
    return path.startswith('/paper/')

def is_api_route(path: str) -> bool:
    """REST API routes use their own auth; skip anonymous session middleware."""
    return path.startswith('/api/')


def get_session_id_from_request(request: Request) -> str:
    """Read session id from header (API routes) or middleware state (backtest routes)."""
    header_val = request.headers.get('x-session-id') or request.headers.get('X-Session-Id')
    if header_val:
        return header_val
    state_val = getattr(request.state, 'session_id', None)
    if state_val:
        return state_val
    raise HTTPException(status_code=400, detail="Missing X-Session-Id header")

class SessionMiddleware(BaseHTTPMiddleware):
    """
    Middleware: require X-Session-Id for backtest routes, allow paper trading to be global.
    
    Uses JSONResponse for error handling (not HTTPException).
    """
    
    async def dispatch(self, request: Request, call_next):
        # Skip exempted paths
        path = request.url.path
        
        if is_exempt(path) or is_api_route(path):
            return await call_next(request)
        
        # Allow OPTIONS (CORS preflight)
        if request.method == 'OPTIONS':
            return await call_next(request)
        
        # Paper trading routes don't need session_id (shared/global)
        if is_paper_trading_route(path):
            return await call_next(request)
        
        # For backtest routes: require X-Session-Id
        session_id = request.headers.get('X-Session-Id')
        
        if not session_id:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Missing X-Session-Id header (required for backtest routes)",
                    "hint": "Include X-Session-Id header in request"
                }
            )
        
        # Validate UUID format
        try:
            uuid.UUID(session_id)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Invalid X-Session-Id format (must be valid UUID)",
                    "received": session_id
                }
            )
        
        # Store in request state for use in routes
        request.state.session_id = session_id
        print(f"✅ Session: {session_id[:8]}... | Route: {path}")
        
        return await call_next(request)


# CSP Middleware: Permit Chart.js and inline scripts (for development)
class CSPHeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        # Allow Chart.js and scripts from same origin, plus unsafe-inline for development
        # No 'unsafe-eval': Chart.js 4 from jsDelivr does not need eval, and
        # nothing in dashboard/frontend calls eval()/new Function(). Keep
        # 'unsafe-inline' until the inline scripts are nonced or externalized.
        response.headers["Content-Security-Policy"] = (
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com data:; "
            "connect-src *; img-src * data:;"
        )
        # DO NOT override CORS headers here — let CORSMiddleware handle them
        return response
