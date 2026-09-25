"""Source guards for the Live Trading Leaderboard tab.

/app has no JS test toolchain, so these assert against shipped HTML/JS as text.
"""

import re
from pathlib import Path

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_APP_HTML = (_FRONTEND / "app.html").read_text(encoding="utf-8")
_LIVE_JS = (_FRONTEND / "js" / "live-leaderboard.js").read_text(encoding="utf-8")
_APP_JS = (_FRONTEND / "app.js").read_text(encoding="utf-8")


def _strip_js_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"^\s*//.*$", "", source, flags=re.MULTILINE)


def test_live_subtab_is_visible_and_daily_stays_parked():
    assert re.search(
        r'data-competition-tab="live"[^>]*>\s*Live Trading Leaderboard',
        _APP_HTML,
    )
    assert 'data-competition-tab="live"' in _APP_HTML
    assert not re.search(
        r'<button[^>]*data-competition-tab="live"[^>]*\bhidden\b',
        _APP_HTML,
    )
    # Daily tab was removed from the subtab bar on main; keep it from returning
    # as a visible sibling of Live.
    daily = re.search(r'data-competition-tab="daily"[^>]*>', _APP_HTML)
    if daily:
        assert "hidden" in daily.group(0)


def test_live_panel_is_a_separate_view():
    assert 'id="liveLeaderboardView"' in _APP_HTML
    assert 'id="liveEquityCurvesChart"' in _APP_HTML
    assert 'id="equityCurvesChart"' in _APP_HTML
    assert _APP_HTML.index('id="liveEquityCurvesChart"') != _APP_HTML.index(
        'id="equityCurvesChart"'
    )


def test_live_script_is_cache_busted():
    match = re.search(r"js/live-leaderboard\.js\?v=(\d+)", _APP_HTML)
    assert match, "live-leaderboard.js must load with a ?v= cache buster"
    assert int(match.group(1)) >= 1


def test_live_fetch_uses_period_live():
    source = _strip_js_comments(_LIVE_JS)
    assert "period=live" in source or "period=${" in source
    assert "/api/v1/leaderboard" in source
    assert "chart_axis" in source


def test_live_chart_does_not_span_empty_future_with_fake_values():
    source = _strip_js_comments(_LIVE_JS)
    assert "has_prints" in source
    # Future nodes stay null; we must not invent contest-window timestamps.
    assert "2026-04-15" not in source


def test_live_nav_round_trips():
    assert "live: { page: 'competition', competitionTab: 'live' }" in _APP_HTML
    body_start = _APP_JS.index("function viewParamForNavState")
    body = _APP_JS[body_start : _APP_JS.index("function buildNavigationUrl")]
    assert "return 'live'" in body
    assert "competitionTab === 'live'" in body or 'competitionTab === "live"' in body
