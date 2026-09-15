"""Logging out has to hand the user back to the landing page.

`logoutUser` cleared the session and left the user standing on the signed-in
shell, where /app's home re-renders as the "Guest Account" demo portfolio --
byte-identical to what a never-signed-in visitor sees, reached by the one action
whose entire point was to leave. `syncHeaderBrand` already repoints the brand at
`/` on sign-out, so the app knew home had moved; it just never took the user
there.

THIS MODULE USED TO GUARD A PAIR, AND THE OTHER HALF IS GONE. index.html stated
a routing rule in its own words -- "Landing is for first-time / logged-out
visitors only. Logged-in users go straight to the dashboard." -- and enforced it:
a visit to `/` carrying a cached auth-user probed /api/auth/me and
`location.replace`d to /app on 200. That rule is what made the two homepages
diverge into two products, because no signed-in visitor could reach `/` to
notice. It was removed when `/` became the one homepage for both audiences.

What that changes here is smaller than it looks. The logout redirect is
unaffected -- `/` is still where a signed-out user belongs, and it is now also
where a signed-in one belongs, which if anything makes the hop more obviously
right. What changed is the CONSEQUENCE of getting the ordering wrong, and the
case below says so in its own docstring: clearing after redirecting used to mean
a round trip back to /app, and now means the landing renders the signed-in CTAs
to someone who just signed out.

The asymmetry matters more than the missing hop: /app is reachable by guests on
purpose (backtests work signed-out), so nothing *breaks*, and that is exactly why
it survived. A logout with no visible consequence beyond the header swapping to
"Sign in" reads as a logout that failed.

Source-text guards because /app has no build step and CI has no browser (the
convention set by test_ai_hedge_fund_frontend.py). No cache-buster assertion
lives here on purpose -- test_frontend_fast_boot.py owns that invariant alone.
"""

import re
from pathlib import Path

from dashboard.backend.tests._frontend_source import fn_body

# Whole-line `//` only, matching the convention in test_landing_chart_first.py:
# an inline `//` would eat the tail of any line holding a URL.
_JS_LINE_COMMENT = re.compile(r"(?m)^\s*//.*$")

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
    """Ordering is load-bearing, not style, and it outlived the reason it was
    written for.

    It used to be a round trip: the landing bounced any visit that still had a
    cached auth-user straight back to /app, so redirecting before clearing
    returned the user to the page they were trying to leave. The landing no
    longer bounces anyone. What it does instead is READ that cached auth-user to
    decide which CTAs to render -- so redirecting first now lands the
    just-signed-out user on a homepage offering "Test a trading idea" and a link
    into the app they just left. Quieter than the round trip and harder to
    notice; the same fix.

    The window is real rather than theoretical: /api/auth/me has to answer
    before the landing can correct itself, and on a cold free-tier backend that
    is tens of seconds.

    clearActiveAgentSession is in the same bind for the original reason: it is
    pure localStorage cleanup, and the redirect tears the page down before a
    later call runs."""
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


def test_the_landing_does_not_bounce_a_signed_in_visitor_back_to_the_app():
    """THE INVERSE OF THE CASE THIS REPLACES, which asserted the redirect was
    still there.

    One homepage serves both audiences now. The redirect is what previously made
    that impossible -- not by hiding a feature, but by ensuring nobody who could
    compare the two pages ever saw the first one -- so its absence is the
    load-bearing fact and it is pinned here rather than left to a code review.
    Restoring it would make every signed-in-CTA guard in the suite green and
    unreachable at once.

    BANS THE OPERATION, NOT A SPELLING, and the first draft of this case did the
    opposite and failed open. It banned the literal strings
    ``window.location.replace(APP_URL)`` and ``redirectToDashboard`` -- the two
    forms the redirect had when it was deleted. But the same change deleted
    ``var APP_URL = '/app';`` from that IIFE, so a restored redirect CANNOT be
    written either way; the shortest way back is
    ``if (res.ok) { window.location.replace('/app'); return; }``, which matches
    neither banned string. That exact line was pasted into the gate during review
    and all five changed modules stayed green. A negative assertion anchored on
    one spelling of the thing it bans is worth very little; this one bans every
    way of leaving the page, from inside the block that must not leave it.

    SCOPED TO THE GATE SCRIPT, because navigation elsewhere in the file is
    correct: the modal's ``goToDashboardLoggedIn`` navigates to
    ``/app?view=agents`` after a successful signup, which is the whole point of
    signing up. Only the gate -- the code that runs on arrival, before anything
    is clicked -- must leave the visitor where they are.

    The POSITIVE assertions are the other half. A guard made only of bans passes
    on a file that lost the revalidation entirely, and losing it is the easy
    mistake: the fetch was the redirect's only consumer, so deleting the redirect
    makes the fetch look like dead code. It is not dead. A cached auth-user is a
    claim, not a session, and this fetch is what turns an expired one back into
    the signed-out CTAs -- and the clearing branch has to TELL somebody, because
    localStorage's storage event does not fire in the tab that wrote.

    Comments are stripped first: the removal is explained in prose directly above
    the code that replaced it, and a raw substring scan reads its own explanation
    as the thing it bans."""
    code = _JS_LINE_COMMENT.sub("", _LANDING_HTML)

    anchor = code.find("document.documentElement.classList.add('landing-auth-pending')")
    assert anchor != -1, (
        "the pre-hydration auth gate is gone from index.html — it cannot be "
        "regenerated by `vite build` (see dashboard/landing/README.md)"
    )
    gate_end = code.find("</script>", anchor)
    assert gate_end != -1, "the auth-gate <script> is unterminated"
    gate = code[anchor:gate_end]

    for leaving in ("location.replace(", "location.assign(", "location.href"):
        assert leaving not in gate, (
            f"the auth gate navigates away from / ({leaving!r}). One homepage "
            f"cannot serve both audiences if half of them never reach it."
        )
    assert "redirectToDashboard" not in gate, "the redirect helper is back"

    assert "/api/auth/me" in gate, (
        "the gate must still revalidate a cached auth-user, or an expired "
        "session keeps being offered the signed-in CTAs"
    )
    assert "credentials: 'include'" in gate, (
        "the revalidation must send the session cookie or it always 401s"
    )
    assert gate.count("showLanding();") >= 3, (
        "every branch of the gate must end on the landing page — the "
        "short-circuit, the revalidated one, and the network failure"
    )
    assert "clearAuth();" in gate, "a 401 must still clear the cached profile"
    assert "landing-auth-change" in code, (
        "clearing the cached profile must notify the mounted bundle — the "
        "storage event does not fire in the tab that wrote"
    )
    # BOTH SIDES OF THE SPLIT, and this half is why the assertion above was not
    # enough on its own. `/app` has no build step, so this event name is a bare
    # string literal here and a `const` in the bundle's source; nothing in the
    # toolchain relates the two. A rename on either side leaves the other
    # compiling, passing, and deaf -- the gate would clear an expired profile
    # and every CTA would keep offering the dashboard until a reload. Reading
    # the React side by its own name is what turns a one-sided ban into a
    # contract.
    session_ts = (
        Path(__file__).resolve().parents[2] / "landing" / "src" / "lib" / "session.ts"
    ).read_text(encoding="utf-8")
    assert 'LANDING_AUTH_EVENT = "landing-auth-change"' in session_ts, (
        "index.html dispatches 'landing-auth-change' as a literal; session.ts "
        "must still listen for that exact name — there is no build step to "
        "share the constant across the split"
    )

    # THE REVEAL HAS A DEADLINE. `html.landing-auth-pending body` is
    # `visibility: hidden`, so the page is BLANK until something removes that
    # class -- and the only things that removed it were the fetch's own `.then`
    # and `.catch`. A request that never settles reaches neither. On the free
    # tier that is not hypothetical: the instance cold-starts in 30-60 seconds,
    # and a visitor holding a cached profile got a white screen for all of it.
    #
    # The timeout bounds the BLANKNESS, not the revalidation: `clearAuth()`
    # still runs when a slow 401 lands, so the CTAs still correct themselves --
    # visibly, which is strictly better than correcting them behind a blank
    # page.
    assert "setTimeout(showLanding" in gate, (
        "the reveal must have a timeout — otherwise a fetch that never settles "
        "leaves the page blank under `visibility: hidden` forever"
    )
    assert "clearTimeout(" in gate, (
        "the timer must be cleared when the fetch wins, or it fires into an "
        "already-revealed page"
    )
    assert re.search(r"if\s*\(revealed\)\s*return;", gate), (
        "showLanding() is now called from up to two places for one page load; "
        "the second must be a no-op rather than a second class flip"
    )
