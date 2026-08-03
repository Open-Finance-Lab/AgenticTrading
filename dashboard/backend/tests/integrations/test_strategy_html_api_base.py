"""MEDIUM #6 — strategy.html must resolve the API base the way app.js does.

strategy.html is a standalone shared-link page (no app.js include). It used
``const API = location.origin`` unconditionally, which is correct locally
(frontend + backend share ``localhost:8000``) but wrong on Vercel, where the
static frontend and the API are on different origins — every API call would hit
the frontend host and 404. It also hardcoded fixed default dates. These source
checks run without a browser (the page has no JS test harness).
"""

from pathlib import Path

_STRATEGY_HTML = (
    Path(__file__).resolve().parents[3] / "frontend" / "strategy.html"
)


def _source() -> str:
    return _STRATEGY_HTML.read_text(encoding="utf-8")


def test_strategy_html_uses_hosted_api_base_off_origin():
    src = _source()
    # The broken bare-origin assignment is gone.
    assert "const API = location.origin" not in src
    # Same localhost-vs-hosted resolution as app.js.
    assert "https://agentictrading.onrender.com" in src
    assert 'window.location.hostname === "localhost"' in src
    assert 'window.location.hostname === "127.0.0.1"' in src


def test_strategy_html_no_hardcoded_default_dates():
    src = _source()
    # The old fixed defaults are replaced by a runtime past-7-days initializer.
    assert 'value="2026-05-01"' not in src
    assert 'value="2026-05-07"' not in src
    assert "function initDateDefaults(" in src
    # Dates are formatted from LOCAL parts, not UTC toISOString (off-by-one near
    # local midnight in non-UTC timezones).
    assert "getFullYear()" in src
    assert "toISOString" not in src
