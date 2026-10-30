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


def test_live_chart_hint_names_the_early_close():
    """The axis ends at 13:00 on an NYSE early close; the caption under the
    chart must not promise a 16:00 node that day."""
    view = _APP_HTML.split('id="liveLeaderboardView"')[1].split('id="leaderboardView"')[0]
    assert "10:00–13:00 on early closes" in view


def test_live_board_copy_names_the_early_close():
    """The header cadence and the About text sit on the same view as the chart
    hint; neither may say the session ends at 16:00 on a 13:00 day."""
    leaderboard_js = (_FRONTEND / "js" / "leaderboard.js").read_text(encoding="utf-8")
    assert "after the 16:00 ET cash session" not in leaderboard_js
    assert "13:00 on early closes" in leaderboard_js
    assert "Each night after the 16:00 ET cash close" not in _APP_HTML
    assert "13:00 on an NYSE early close" in _APP_HTML


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


def test_live_table_is_performance_not_rank():
    source = _strip_js_comments(_LIVE_JS)
    assert 'id="liveStandingsTitle">Performance' in _APP_HTML
    assert 'data-sort="rank"' not in _APP_HTML.split('id="liveLeaderboardView"')[1].split('id="leaderboardView"')[0]
    assert 'data-sort="return"' in _APP_HTML
    assert 'data-sort="trades"' in _APP_HTML
    assert 'data-sort="hold"' in _APP_HTML
    assert "liveSortKey = 'return'" in source
    assert "liveFormatHold" in source
    assert "liveVsSpy" in source
    assert "data-sort=\"rank\"" not in source
    assert "entry.rank" not in source


def test_live_nav_round_trips():
    assert "live: { page: 'competition', competitionTab: 'live' }" in _APP_HTML
    body_start = _APP_JS.index("function viewParamForNavState")
    body = _APP_JS[body_start : _APP_JS.index("function buildNavigationUrl")]
    assert "return 'live'" in body
    assert "competitionTab === 'live'" in body or 'competitionTab === "live"' in body


# --- Sharpe: displayed, never recomputed (D20), always with its interval -----

from dashboard.backend.domain.backtesting.metrics import (  # noqa: E402
    SHARPE_MIN_DAYS,
    daily_sharpe_basis,
)
from dashboard.backend.tests._frontend_source import fn_body, run_node  # noqa: E402

_SHARPE_BASIS = daily_sharpe_basis()
_SHARPE_OK = {
    "sharpe_status": "ok",
    "sharpe_ratio": 2.611715,
    "sharpe_ci_low": -11.39649,
    "sharpe_ci_high": 16.619919,
    "sharpe_days": 5,
}
_DECISION_5_TITLE = (
    "Return above what cash would have earned, divided by how much the daily "
    "returns swung. Based on 5 trading days; cash rate 0%/yr; scaled to a yearly figure."
)


def _format_sharpe(entry, basis=_SHARPE_BASIS):
    script = fn_body("function liveFormatSharpe(", _LIVE_JS) + (
        f"\nconsole.log(JSON.stringify(liveFormatSharpe({json.dumps(entry)}, {json.dumps(basis)})));"
    )
    return run_node(script)


def _sharpe_row(entry_id, **fields):
    return {
        "entry_id": entry_id, "model": entry_id, "team_badge": "Model", "is_model": True,
        "status": "frozen", "portfolio_value": 100000, "cumulative_return": 0.01,
        "max_drawdown": -0.01, "num_trades": 1, "equity_curve": [], **fields,
    }


def _render_live(entries, *, selected=None):
    """Load the whole shipped file under node, then paint the table and detail.

    Whole-file rather than lifted functions, so a helper another change adds
    to populateLiveTable cannot turn this into a ReferenceError. If node
    reports one for a helper defined in app.js, add its `fn_body(...)` here.
    """
    script = "\n".join([
        "const window = {};",
        fn_body("function escapeHtml("),
        _LIVE_JS,
        "const tbody = { innerHTML: '' }; const detail = { innerHTML: '' };",
        "const document = {",
        "  getElementById: (id) => (id === 'liveSelectedDetail' ? detail : tbody),",
        "  querySelectorAll: () => [],",
        "};",
        f"livePayload = {{ entries: {json.dumps(entries)}, live_status: {{ sharpe: {json.dumps(_SHARPE_BASIS)} }} }};",
        "liveSortKey = 'sharpe'; liveSortDir = 'desc';",
        "populateLiveTable();",
        f"renderLiveDetail(livePayload.entries.find((e) => e.entry_id === {json.dumps(selected)}) || null);",
        "console.log(JSON.stringify({ table: tbody.innerHTML, detail: detail.innerHTML,"
        " order: liveSortedEntries().map((e) => e.entry_id) }));",
    ])
    return run_node(script)


def test_live_js_never_recomputes_sharpe():
    source = _strip_js_comments(_LIVE_JS)
    assert "Math.sqrt" not in source
    assert "stdev" not in source.lower()
    assert "Number(entry.sharpe_ratio).toFixed" not in source


@_needs_node
def test_format_sharpe_shows_the_value_with_its_interval():
    out = _format_sharpe(_SHARPE_OK)
    assert out == {
        "main": "2.61",
        "sub": "95% interval -11.40 to 16.62",
        "title": _DECISION_5_TITLE,
    }


@_needs_node
def test_format_sharpe_tooltip_reads_the_payloads_cash_rate():
    out = _format_sharpe(_SHARPE_OK, {**_SHARPE_BASIS, "rf_annual": 0.0252})
    assert "cash rate 2.52%/yr" in out["title"]


@_needs_node
def test_format_sharpe_explains_every_dash():
    short = _format_sharpe({"sharpe_status": "insufficient_days", "sharpe_ratio": None, "sharpe_days": 3})
    assert short == {"main": "—", "sub": f"Needs {SHARPE_MIN_DAYS} trading days (have 3)", "title": ""}
    flat = _format_sharpe({"sharpe_status": "no_variation", "sharpe_ratio": None, "sharpe_days": 6})
    assert flat == {"main": "—", "sub": "Returns never moved", "title": ""}
    pending = _format_sharpe({"sharpe_status": "pending", "sharpe_ratio": None, "sharpe_days": 0})
    assert pending == {"main": "—", "sub": "", "title": ""}
    # An ok status with a missing bound is malformed: no number without its interval.
    torn = _format_sharpe({**_SHARPE_OK, "sharpe_ci_high": None})
    assert torn["main"] == "—"
    # A payload from before the basis existed still says what is missing.
    no_basis = _format_sharpe({"sharpe_status": "insufficient_days", "sharpe_days": 2}, None)
    assert no_basis["sub"] == "Needs more trading days (have 2)"


@_needs_node
def test_live_table_and_detail_render_the_interval_and_sort_null_last():
    out = _render_live(
        [
            _sharpe_row("up", **_SHARPE_OK),
            _sharpe_row("young", sharpe_status="insufficient_days", sharpe_ratio=None, sharpe_days=3),
            _sharpe_row("down", sharpe_status="ok", sharpe_ratio=-0.5, sharpe_ci_low=-14.2,
                        sharpe_ci_high=13.2, sharpe_days=5),
        ],
        selected="up",
    )
    assert out["order"] == ["up", "down", "young"]
    assert "95% interval -11.40 to 16.62" in out["table"]
    assert f'title="{_DECISION_5_TITLE}"' in out["table"]
    assert "Needs 5 trading days (have 3)" in out["table"]
    assert "Sharpe ratio" in out["detail"]
    assert "2.61 · 95% interval -11.40 to 16.62" in out["detail"]


@_needs_node
def test_null_sharpe_sorts_last_in_both_directions():
    rows = [
        _sharpe_row("none", sharpe_status="insufficient_days", sharpe_ratio=None, sharpe_days=3),
        _sharpe_row("neg", sharpe_status="ok", sharpe_ratio=-0.5, sharpe_ci_low=-14.0,
                    sharpe_ci_high=13.0, sharpe_days=5),
        _sharpe_row("pos", **_SHARPE_OK),
    ]
    script = "\n".join([
        "const window = {};",
        _LIVE_JS,
        f"livePayload = {{ entries: {json.dumps(rows)} }}; liveSortKey = 'sharpe';",
        "liveSortDir = 'desc'; const desc = liveSortedEntries().map((e) => e.entry_id);",
        "liveSortDir = 'asc'; const asc = liveSortedEntries().map((e) => e.entry_id);",
        "console.log(JSON.stringify({ desc, asc }));",
    ])
    out = run_node(script)
    assert out == {"desc": ["pos", "neg", "none"], "asc": ["neg", "pos", "none"]}


def test_live_sharpe_header_states_its_basis_in_plain_words():
    live_html = _APP_HTML.split('id="liveLeaderboardView"')[1].split('id="leaderboardView"')[0]
    match = re.search(r'<th[^>]*data-sort="sharpe"[^>]*title="([^"]*)"', live_html)
    assert match, "the Live Sharpe header lost its tooltip"
    title = match.group(1)
    assert "Return above what cash would have earned" in title
    assert "divided by how much the daily returns swung" in title
    assert "scaled to a yearly figure" in title
    assert f"Shown from {SHARPE_MIN_DAYS} trading days" in title
    assert "95% interval" in title
    for jargon in ("T-bill", "excess", "√252", "sqrt", "sample SD"):
        assert jargon not in title
    assert "noisy until a few weeks" not in _APP_HTML


@_needs_node
def test_an_old_payloads_hourly_number_is_neither_shown_nor_sorted():
    """Vercel can serve this file before Render serves the new payload: a row
    with a stored hourly ``sharpe_ratio`` and no status must read "—" and sort
    with the absent rows, not by the hidden value."""
    assert _format_sharpe({"sharpe_ratio": 1.7}) == {"main": "—", "sub": "", "title": ""}
    out = _render_live(
        [
            _sharpe_row("stale", sharpe_ratio=9.7),
            _sharpe_row("up", **_SHARPE_OK),
            _sharpe_row("young", sharpe_status="insufficient_days", sharpe_ratio=None, sharpe_days=3),
        ],
        selected="young",
    )
    assert out["order"][0] == "up"
    assert set(out["order"][1:]) == {"stale", "young"}
    assert "9.70" not in out["table"]
    assert "Needs 5 trading days (have 3)" in out["detail"]


@_needs_node
def test_format_sharpe_never_prints_minus_zero_or_an_invented_day_count():
    tiny = _format_sharpe({**_SHARPE_OK, "sharpe_ratio": -0.004, "sharpe_ci_low": -0.004,
                           "sharpe_ci_high": 0.003})
    assert tiny["main"] == "0.00"
    assert tiny["sub"] == "95% interval 0.00 to 0.00"
    no_days = _format_sharpe({k: v for k, v in _SHARPE_OK.items() if k != "sharpe_days"})
    assert no_days["main"] == "2.61"
    assert "trading days" not in no_days["title"]
    assert "cash rate 0%/yr" in no_days["title"]
