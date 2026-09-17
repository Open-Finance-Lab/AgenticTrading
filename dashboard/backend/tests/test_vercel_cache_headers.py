"""Cache-header contract for the deployed frontend (vercel.json).

/app serves the buster-carrying HTML; a cached copy pins users to old JS for
up to an hour after every deploy (Finding 5, 2026-08-04 backtest-visibility
spec). Vercel applies all matching header rules with the last match winning
per key, so the catch-all CSP survives these overrides — assert both halves
so neither regresses silently.
"""

import json
from pathlib import Path

VERCEL = json.loads(
    (Path(__file__).resolve().parents[2] / "frontend" / "vercel.json")
    .read_text(encoding="utf-8")
)


def _cache_control(source: str):
    values = [
        h["value"]
        for entry in VERCEL["headers"]
        if entry["source"] == source
        for h in entry["headers"]
        if h["key"] == "Cache-Control"
    ]
    return values[-1] if values else None


def test_app_html_routes_must_revalidate():
    # /app.html is belt-and-braces: `cleanUrls: true` turns it into a 308 to
    # /app that already carries must-revalidate, so the rule only becomes
    # load-bearing if cleanUrls is ever switched off. /app is the real fix.
    for source in ("/app", "/app.html"):
        assert _cache_control(source) == "public, max-age=0, must-revalidate", source


def test_catch_all_precedes_the_cache_control_overrides():
    """Order is the whole mechanism — assert it, don't assume it.

    Every rule whose source matches is applied and the *last* one wins per
    key, so an override only overrides while it sits after the catch-all.
    Moving /(.*) to the end of the array would silently restore max-age=3600
    on every one of these routes while each _cache_control() assertion above
    still passed, because those match on the source string alone.
    """
    order = [entry["source"] for entry in VERCEL["headers"]]
    catch_all = order.index("/(.*)")
    for source in ("/", "/app", "/app.html", "/app.js", "/styles.css", "/assets/(.*)"):
        assert catch_all < order.index(source), source


def test_existing_overrides_unchanged():
    assert _cache_control("/") == "public, max-age=0, must-revalidate"
    assert _cache_control("/app.js") == "public, max-age=0, must-revalidate"
    assert _cache_control("/styles.css") == "public, max-age=0, must-revalidate"
    assert _cache_control("/assets/(.*)") == "public, max-age=31536000, immutable"


def test_catch_all_keeps_csp():
    catch_all = next(e for e in VERCEL["headers"] if e["source"] == "/(.*)")
    assert any(h["key"] == "Content-Security-Policy" for h in catch_all["headers"])


def test_admin_shell_must_revalidate_like_every_other_shell():
    for source in ("/admin", "/admin.html"):
        assert _cache_control(source) == "public, max-age=0, must-revalidate", source


def test_admin_overrides_follow_the_api_no_store_rule():
    """The API no-store rule's optional group matches the bare /admin.

    `/(api|...|admin|...)(/.*)?` was written for `/admin/runs/{id}`; with the
    trailing group optional it also matches `/admin` itself. Last match wins per
    key, so the shell's override only overrides while it sits *after* that
    rule. Moving it earlier silently restores no-store with every
    `_cache_control` assertion above still green.
    """
    order = [entry["source"] for entry in VERCEL["headers"]]
    api_no_store = order.index(
        "/(api|paper|backtest|runs|config|admin|ticker|health|compare)(/.*)?"
    )
    assert order.index("/(.*)") < order.index("/admin")
    assert api_no_store < order.index("/admin")
    assert api_no_store < order.index("/admin.html")


ADMIN_API_REWRITE = "/admin/runs/:path*"


def test_no_rewrite_can_claim_the_admin_page():
    """`/admin/:path*` → Render is gone (design §7.1).

    Whether `:path*` also matches the bare `/admin` differs between
    path-to-regexp versions; with that rewrite present, the page's reachability
    on Vercel depended on a router version nobody controls.

    The invariant is "nothing may claim the page", not "no source may start with
    /admin". The blunt prefix ban was the same rule written one notch too wide:
    it also unproxied `DELETE /admin/runs/{run_id}`, which api/routers/admin.py
    registers directly on the app outside `/api` and which the API no-store
    header rule above was written for. A required literal segment after /admin
    is what makes the narrow rewrite safe -- no matcher version can fold the
    bare `/admin` into a pattern that demands `/runs/` next.
    """
    admin_sources = [e["source"] for e in VERCEL["rewrites"] if e["source"].startswith("/admin")]
    assert admin_sources == [ADMIN_API_REWRITE]
    assert ADMIN_API_REWRITE.split("/")[2] == "runs"


def test_the_admin_runs_debug_route_stays_proxied_to_render():
    entry = next(e for e in VERCEL["rewrites"] if e["source"] == ADMIN_API_REWRITE)
    assert entry["destination"] == "https://agentictrading.onrender.com/admin/runs/:path*"


def test_admin_analytics_redirects_permanently_to_admin():
    redirects = VERCEL["redirects"]
    entry = next(e for e in redirects if e["source"] == "/admin-analytics")
    assert entry["destination"] == "/admin"
    assert entry["permanent"] is True
    assert not any(e["source"] == "/admin-analytics" for e in VERCEL["headers"])
