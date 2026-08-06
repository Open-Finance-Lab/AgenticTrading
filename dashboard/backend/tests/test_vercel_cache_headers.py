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
