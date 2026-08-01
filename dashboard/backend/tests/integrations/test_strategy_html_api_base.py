"""strategy.html must resolve the API base the way app.js does.

strategy.html is a standalone shared-link page (no app.js include). Same-origin
locally; empty base on Vercel so backend paths hit vercel.json rewrites.
"""

from pathlib import Path

_STRATEGY_HTML = (
    Path(__file__).resolve().parents[3] / "frontend" / "strategy.html"
)


def _source() -> str:
    return _STRATEGY_HTML.read_text(encoding="utf-8")


def test_strategy_html_uses_same_origin_api_base_off_localhost():
    src = _source()
    assert "const API = location.origin" not in src
    assert "https://agentictrading.onrender.com" not in src
    assert 'window.location.hostname === "localhost"' in src
    assert 'window.location.hostname === "127.0.0.1"' in src
    assert ': ""' in src or ": ''" in src


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
