"""Source guards for the Live Trading Leaderboard tab.

/app has no JS test toolchain, so these assert against shipped HTML/JS as text.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

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


def _y_bounds_source() -> str:
    start = _LIVE_JS.index("const LIVE_Y_PAD_RATIO")
    end = _LIVE_JS.index("function renderLiveChart")
    return _LIVE_JS[start:end]


def _y_bounds(datasets, *, is_money=True, capital=100_000):
    script = _y_bounds_source() + (
        f"\nprocess.stdout.write(JSON.stringify(liveYAxisBounds("
        f"{json.dumps(datasets)}, {{isMoney: {str(is_money).lower()}, capital: {capital}}})));"
    )
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


_needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


@_needs_node
def test_y_axis_fits_curves_huddled_near_capital():
    b = _y_bounds([{"data": [100_000, 100_300, None, 100_550]}, {"data": [99_900, 100_100]}])
    assert b["min"] <= 99_900 - 0.2 * 650 and b["max"] >= 100_550 + 0.2 * 650
    assert b["max"] - b["min"] < 2_000  # was a fixed $20k window
    assert b["min"] % b["stepSize"] == 0 and b["max"] % b["stepSize"] == 0


@_needs_node
def test_y_axis_keeps_a_minimum_span_and_the_start_value():
    flat = _y_bounds([{"data": [100_000, 100_000]}])
    assert flat["min"] <= 99_500 and flat["max"] >= 100_500
    up_only = _y_bounds([{"data": [104_000, 106_000]}])
    assert up_only["min"] <= 100_000


@_needs_node
def test_y_axis_ignores_hidden_series_and_supports_percent():
    b = _y_bounds([
        {"data": [100_200]},
        {"data": [150_000], "hidden": True},
    ])
    assert b["max"] < 101_000
    pct = _y_bounds([{"data": [-0.01, 0.02]}], is_money=False)
    assert pct["min"] <= -0.016 and pct["max"] >= 0.026


def test_live_chart_uses_fitted_y_bounds():
    source = _strip_js_comments(_LIVE_JS)
    assert "liveYAxisBounds(datasets" in source
    assert "capital * 0.9" not in source


def test_live_nav_round_trips():
    assert "live: { page: 'competition', competitionTab: 'live' }" in _APP_HTML
    body_start = _APP_JS.index("function viewParamForNavState")
    body = _APP_JS[body_start : _APP_JS.index("function buildNavigationUrl")]
    assert "return 'live'" in body
    assert "competitionTab === 'live'" in body or 'competitionTab === "live"' in body
