"""Logging out has to hand the user back to the landing page.

index.html states the routing rule in its own words -- "Landing is for
first-time / logged-out visitors only. Logged-in users go straight to the
dashboard." -- and enforces the signed-in half itself: a visit to `/` carrying a
cached auth-user probes /api/auth/me and `location.replace`s to /app on 200.
Only that half was ever built. `logoutUser` cleared the session and left the
user standing on the signed-in shell, where /app's home re-renders as the "Guest
Account" demo portfolio -- byte-identical to what a never-signed-in visitor
sees, reached by the one action whose entire point was to leave.
`syncHeaderBrand` already repoints the brand at `/` on sign-out, so the app knew
home had moved; it just never took the user there.

The asymmetry matters more than the missing hop: /app is reachable by guests on
purpose (backtests work signed-out), so nothing *breaks*, and that is exactly why
it survived. A logout with no visible consequence beyond the header swapping to
"Sign in" reads as a logout that failed.

Source-text guards because /app has no build step and CI has no browser (the
convention set by test_ai_hedge_fund_frontend.py). No cache-buster assertion
lives here on purpose -- test_frontend_fast_boot.py owns that invariant alone.
"""

from pathlib import Path

from dashboard.backend.tests._frontend_source import fn_body

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_LANDING_HTML = (_FRONTEND / "index.html").read_text(encoding="utf-8")


def test_logout_returns_the_user_to_the_landing_page():
    """The missing half of the contract. Without it the user lands on /app's
    guest home -- the same screen a first-time visitor gets -- so the only
    feedback that logout worked is the header swapping to "Sign in"."""
    body = fn_body("async function logoutUser")

    assert "location.replace('/')" in body, (
        "logoutUser must send the signed-out user back to the landing page"
    )


def test_logout_uses_replace_so_back_cannot_restore_the_signed_in_shell():
    """`href = '/'` would leave /app in history: one Back press re-renders the
    shell the user just left. `replace` drops it, and mirrors the verb the
    landing's own redirect uses."""
    body = fn_body("async function logoutUser")

    assert "location.href" not in body, "use location.replace, not location.href"


def test_logout_clears_local_session_state_before_redirecting():
    """Ordering is load-bearing, not style. The landing bounces any visit that
    still has a cached auth-user straight back to /app, so redirecting first and
    clearing second is a round trip to the page the user was trying to leave.
    clearActiveAgentSession is in the same bind: it is pure localStorage
    cleanup, and the redirect tears the page down before a later call runs."""
    body = fn_body("async function logoutUser")
    redirect_at = body.index("location.replace('/')")

    assert body.index("clearAuthState()") < redirect_at, (
        "clearAuthState() must run before the redirect or the landing bounces "
        "the user straight back to /app"
    )
    assert body.index("clearActiveAgentSession()") < redirect_at, (
        "the active-agent keys must be cleared before the page is torn down"
    )


def test_logout_awaits_the_server_before_leaving_the_page():
    """The redirect tears down the tab. Fire-and-forget would race the POST that
    invalidates the cookie against the unload, leaving a live server-side
    session behind a UI that says signed out."""
    body = fn_body("async function logoutUser")

    assert body.index("await AuthAPI.logout()") < body.index("location.replace('/')")


def test_session_expiry_does_not_evict_guests_from_the_app():
    """clearAuthState is the choke point *every* sign-out path funnels through,
    including refreshAuthUser's 401 branch -- which fires on every guest page
    load. A redirect there would bounce each first-time visitor off /app before
    they saw it. The redirect belongs to the deliberate action, not the state
    change."""
    body = fn_body("function clearAuthState")

    assert "location.replace" not in body
    assert "location.href" not in body


def test_landing_still_redirects_signed_in_visitors():
    """The other half of the pair. If this ever goes away, the guard above is
    enforcing a round trip to nowhere."""
    assert "window.location.replace(APP_URL)" in _LANDING_HTML
    assert "var APP_URL = '/app';" in _LANDING_HTML
