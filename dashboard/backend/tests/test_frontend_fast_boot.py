"""Boot-order contracts for /app: interactive before network, feedback while slow.

The boot regression this pins: the DOMContentLoaded handler used to serially
await ~12 network round trips (auth refresh, agent claim + list, deep link,
defaults, feature flags) before initNavigation() wired a single nav button and
before the ticker's first fetch was even issued. On a Render cold start that
left the page looking dead for 30+ seconds -- nav clicks silently dropped, no
loading state anywhere, ticker (the de-facto "app is ready" signal) dead last.

Guarded as source text per the convention in _frontend_source.py.
"""

from dashboard.backend.tests._frontend_source import (
    APP_HTML,
    APP_JS,
    STYLES,
    _match_brace,
    fn_body,
)


def _boot_handler() -> str:
    """The DOMContentLoaded handler's source, brace-matched."""
    start = APP_JS.index("document.addEventListener('DOMContentLoaded'")
    open_brace = APP_JS.index("{", start)
    return APP_JS[start : _match_brace(APP_JS, open_brace) + 1]


# ---------------------------------------------------------------------------
# 1. Interactivity and the ticker must not queue behind the auth waterfall.
# ---------------------------------------------------------------------------

def test_nav_wiring_precedes_first_network_await():
    boot = _boot_handler()
    assert "initNavigation()" in boot
    assert boot.index("initNavigation()") < boot.index(
        "await restoreActiveAgentSession()"
    ), "nav buttons must be wired before the first network await, not after it"


def test_ticker_starts_before_auth_waterfall():
    boot = _boot_handler()
    assert boot.index("loadMarketTicker()") < boot.index(
        "await restoreActiveAgentSession()"
    ), "the ticker is the page's liveness signal; it must not wait on auth"


def test_config_fetches_are_not_serial_awaits():
    boot = _boot_handler()
    assert "await loadDefaults()" not in boot, (
        "loadDefaults must be kicked off in parallel (awaited later as a "
        "promise), not serially awaited on the boot critical path"
    )
    assert "await loadMarketDataFeatures()" not in boot


# ---------------------------------------------------------------------------
# 2. Early nav clicks must not race the account-claim ordering invariant:
#    refreshAuthUser -> claimAgentsForUser must land before the first real
#    agents fetch, or a landing-signup handoff misses the guest Foundation
#    agent and provisions a duplicate starter.
# ---------------------------------------------------------------------------

def test_load_agents_waits_for_auth_boot_gate():
    body = fn_body("async function loadAgents(")
    assert "authBootGate" in body, (
        "loadAgents must await the auth boot gate so an early nav click "
        "cannot fetch agents before the account claim settles"
    )
    assert "agentsLoadInFlight" in body, (
        "concurrent early clicks must coalesce into one fetch, not stack "
        "identical requests behind the gate"
    )


def test_claim_calls_the_ungated_loader():
    body = fn_body("async function claimAgentsForUser(")
    assert "loadAgentsNow(" in body, (
        "claimAgentsForUser runs inside the gated section of boot; calling "
        "the gated loadAgents() from there would deadlock on the gate"
    )


def test_boot_opens_the_gate_after_auth_refresh():
    boot = _boot_handler()
    assert "openAuthBootGate()" in boot
    assert boot.index("refreshAuthUser()") < boot.index("openAuthBootGate()"), (
        "the gate must open only after the refresh/claim phase has settled"
    )


# ---------------------------------------------------------------------------
# 3. A slow boot must not yank the page from under an explicit user click.
# ---------------------------------------------------------------------------

def test_initial_navigation_respects_user_click():
    body = fn_body("function applyInitialNavigation(")
    assert "userHasNavigated" in body, (
        "with nav live during boot, restoring the saved page must be skipped "
        "once the user has explicitly navigated"
    )


# ---------------------------------------------------------------------------
# 4. The network head start: preconnect + warmup so the Render cold start
#    begins before the JS pipeline finishes downloading.
# ---------------------------------------------------------------------------

def test_app_html_preconnects_to_api_origin():
    assert (
        '<link rel="preconnect" href="https://agentictrading.onrender.com" crossorigin>'
        in APP_HTML
    )


def test_boot_script_warms_the_backend():
    head = APP_HTML[: APP_HTML.index("</head>")]
    assert "/health" in head and "API_WARMUP" in head, (
        "the inline boot script must fire a /health warmup fetch so a Render "
        "cold start begins during JS download, not after it"
    )


def test_scripts_are_deferred():
    # Every external script must carry defer: the Chart.js CDN tag in <head>
    # was render-blocking, and defer preserves execution order document-wide
    # (head tag first, body tags after, all before DOMContentLoaded).
    import re

    tags = re.findall(r"<script [^>]*src=[^>]*>", APP_HTML)
    assert tags, "expected external script tags in app.html"
    offenders = [t for t in tags if "defer" not in t]
    assert offenders == [], f"script tags missing defer: {offenders}"


# ---------------------------------------------------------------------------
# 5. Visible loading feedback.
# ---------------------------------------------------------------------------

def test_agents_grid_ships_loading_skeleton():
    # Pre-JS, My Agents must show placeholder cards instead of a blank panel,
    # in every shelf that renders agents. Prompted Models, Open Agents, and
    # External all have grids. The Crypto/Futures rows are locked and have no
    # grid at all, so there is nothing to skeleton there.
    grid_ids = (
        "agentsGridPrompted",
        "agentsGridOpen",
        "agentsGridExternal",
    )
    skeletons = [
        m for m in range(len(APP_HTML)) if APP_HTML.startswith("agent-card--skeleton", m)
    ]
    for grid_id in grid_ids:
        grid_at = APP_HTML.index(f'id="{grid_id}"')
        assert any(grid_at < at < grid_at + 2000 for at in skeletons), grid_id
    assert ".agent-card--skeleton" in STYLES


def test_skeletons_self_clear_on_every_render():
    # The skeletons need no removal code IF every render path overwrites the
    # grid. renderAgentCards clears unconditionally; renderAgentsError clears
    # explicitly. This pins the clear so a future early-return above it
    # cannot strand skeletons behind real content.
    body = fn_body("function renderAgentCards(")
    assert body.index("grid.innerHTML = ''") < body.index("agents.length")
    assert "grid.innerHTML = ''" in fn_body("function renderAgentsError(")


def test_slow_boot_notice_is_wired():
    boot = _boot_handler()
    assert "API_WARMUP" in boot and "showAppToast(" in boot, (
        "when the warmup ping is still pending after the notice delay, the "
        "user must be told the free-tier server is waking up"
    )


# ---------------------------------------------------------------------------
# 6. Cache busters: shipped HTML must reference the new revisions.
# ---------------------------------------------------------------------------

def test_cache_busters_bumped():
    # Floor advances whenever one of these files changes and its ?v= must ship.
    # The floor must clear whatever main already ships, not just whatever this
    # branch started from. Two branches cut from different bases can both
    # "bump" to the same number, merge without a conflict, and leave prod
    # serving new app.js content under a ?v= browsers have already cached.
    #
    # THE SINGLE OWNER of this invariant, for every busted asset — do not add a
    # second cache-buster guard in a feature suite. A `>=` floor and an exact
    # match are different rules for the same fact: the floor stays green through
    # the next bump, so the exact one looks like the broken guard and gets
    # "fixed" by loosening it. That collision has already cost this repo one
    # round of follow-ups (#347/#348).
    assert "app.js?v=125" in APP_HTML
    assert "js/agent-editor.js?v=30" in APP_HTML
    assert "styles.css?v=130" in APP_HTML
    assert "js/leaderboard.js?v=32" in APP_HTML
    assert "home-page.js?v=50" in APP_HTML
    assert "js/credit-format.js?v=1" in APP_HTML
    assert "js/credits.js?v=8" in APP_HTML
    assert "js/admin-credits.js?v=6" in APP_HTML
    assert "js/admin-analytics.js?v=2" in APP_HTML
