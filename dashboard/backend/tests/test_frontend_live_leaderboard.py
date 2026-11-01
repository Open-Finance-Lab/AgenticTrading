"""Source guards for the Live Leaderboard tab.

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
# Loaded before live-leaderboard.js in app.html; the Live Sharpe cell borrows
# its formatter (formatBoardSharpe) and sort predicate (boardSharpeValue).
_LEADERBOARD_JS = (_FRONTEND / "js" / "leaderboard.js").read_text(encoding="utf-8")
_APP_JS = (_FRONTEND / "app.js").read_text(encoding="utf-8")


def _strip_js_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"^\s*//.*$", "", source, flags=re.MULTILINE)


def test_live_subtab_is_visible_and_daily_stays_parked():
    assert re.search(
        r'data-competition-tab="live"[^>]*>\s*Live Leaderboard<',
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


def _board_sharpe_js():
    """leaderboard.js's Sharpe helpers and their exports, lifted: the whole file
    plus live-leaderboard.js overruns `node -e`'s 128 KB argument limit."""
    return "\n".join([
        fn_body("function finiteNumber(", _LEADERBOARD_JS),
        fn_body("function boardSharpeValue(", _LEADERBOARD_JS),
        fn_body("function formatBoardSharpe(", _LEADERBOARD_JS),
        "window.formatBoardSharpe = formatBoardSharpe;",
        "window.boardSharpeValue = boardSharpeValue;",
    ])


def _format_sharpe(entry, basis=_SHARPE_BASIS):
    script = "\n".join([
        "const window = {};",
        _board_sharpe_js(),
        fn_body("function liveFormatSharpe(", _LIVE_JS),
        f"console.log(JSON.stringify(liveFormatSharpe({json.dumps(entry)}, {json.dumps(basis)})));",
    ])
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
        _board_sharpe_js(),
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
        "console.log(JSON.stringify({ table: tbody.innerHTML, detail: detail.innerHTML, "
        + "order: liveSortedEntries().map((e) => e.entry_id) }));",
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
    undefined = _format_sharpe({"sharpe_status": "undefined", "sharpe_ratio": None, "sharpe_days": 6})
    assert undefined == {"main": "—", "sub": "Cannot be computed", "title": ""}
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
        _board_sharpe_js(),
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


@_needs_node
def test_an_unprinted_row_carries_no_sharpe_tooltip_or_rank():
    """A row with no stored final value renders "—" everywhere; its Sharpe
    tooltip must not describe a number the cell does not show."""
    out = _render_live(
        [
            _sharpe_row("unprinted", portfolio_value=None, **_SHARPE_OK),
            _sharpe_row("down", sharpe_status="ok", sharpe_ratio=-0.5, sharpe_ci_low=-14.2,
                        sharpe_ci_high=13.2, sharpe_days=5),
        ],
        selected="unprinted",
    )
    assert out["table"].count("Based on 5 trading days") == 1  # "down" only
    assert "Based on 5 trading days" not in out["detail"]
    assert out["order"] == ["down", "unprinted"]


@_needs_node
def test_live_sharpe_is_the_competition_formatter_not_a_copy():
    """One wording for both boards: liveFormatSharpe delegates rather than
    re-implementing (a parity test over two copies misses any case it does not
    enumerate). Without leaderboard.js it shows a dash, never its own words."""
    code = _strip_js_comments(fn_body("function liveFormatSharpe(", _LIVE_JS))
    assert "window.formatBoardSharpe(entry, basis)" in code
    assert "toFixed" not in code and "interval" not in code
    alone = run_node("\n".join([
        "const window = {};",
        fn_body("function liveFormatSharpe(", _LIVE_JS),
        fn_body("function liveSortValue(", _LIVE_JS),
        f"console.log(JSON.stringify([liveFormatSharpe({json.dumps(_SHARPE_OK)}, null), "
        f"liveSortValue({json.dumps(_SHARPE_OK)}, 'sharpe')]));",
    ]))
    assert alone == [{"main": "—", "sub": "", "title": ""}, None]


# ── Naming and header copy (direction spec §6) ──────────────────────────────


def _live_view_markup() -> str:
    """The #liveLeaderboardView markup, up to the Competition view."""
    return _APP_HTML.split('id="liveLeaderboardView"')[1].split('id="leaderboardView"')[0]


def _static_text(html: str, element_id: str) -> str:
    """The text inside the element carrying `id="<element_id>"` (no child tags)."""
    match = re.search(rf'id="{element_id}"[^>]*>([^<]*)<', html)
    assert match, f"#{element_id} is missing from app.html"
    return match.group(1)


def test_the_board_is_named_live_leaderboard_on_every_surface():
    """'Trading' read as real orders and collided with the Robinhood
    live-trading docs. The static title, the JS fallback and the payload all
    resolve to service.LIVE_BOARD_TITLE."""
    from dashboard.backend.domain.leaderboard import service

    assert _static_text(_APP_HTML, "liveBoardTitle") == service.LIVE_BOARD_TITLE
    assert f"payload.board_title || '{service.LIVE_BOARD_TITLE}'" in _LIVE_JS
    visible = _strip_js_comments(re.sub(r"<!--.*?-->", "", _APP_HTML, flags=re.DOTALL))
    assert "Live Trading Leaderboard" not in visible
    assert "Live Trading Leaderboard" not in _strip_js_comments(_LIVE_JS)


def test_the_subtitle_is_static_simulated_forward_only_copy():
    assert (
        _static_text(_APP_HTML, "liveBoardSubtitle")
        == "Simulated fills · forward-only · no real money"
    )
    source = _strip_js_comments(_LIVE_JS)
    assert "liveBoardSubtitle" not in source, (
        "the subtitle is static copy in app.html; rewriting it from JS put "
        "'paper trading' back on screen"
    )
    assert "paper trading" not in source.lower()


def test_the_updated_stat_states_the_cadence_not_the_request_clock():
    """payload.updated_at is the wall clock of the GET, so rendering it always
    read 'just now' on a board that last advanced days ago."""
    assert re.search(
        r'<span class="stat-label">Updated</span>\s*'
        r'<span class="stat-value" id="liveUpdatedLabel">Nightly after the close</span>',
        _live_view_markup(),
    )
    source = _strip_js_comments(_LIVE_JS)
    assert "liveUpdatedLabel" not in source
    assert "updated_at" not in source


def test_the_season_stat_reads_season_0_preview_from_one_owner():
    from dashboard.backend.domain.leaderboard import service

    assert service.LIVE_PHASE_LABEL == "Season 0 · preview"
    assert _static_text(_APP_HTML, "livePhaseLabel") == service.LIVE_PHASE_LABEL
    assert f"payload.phase_label || '{service.LIVE_PHASE_LABEL}'" in _LIVE_JS


def test_neither_board_renders_a_stop_or_take_profit_price():
    """Both boards disclaim stops as not executed. A future trade-log or
    reasoning view is the first place a model's stop_loss_price could appear
    next to its decisions; it must label the value 'not executed' before this
    guard is lifted, or the page contradicts its own Rules card."""
    leaderboard_js = (_FRONTEND / "js" / "leaderboard.js").read_text(encoding="utf-8")
    for name, source in (
        ("js/live-leaderboard.js", _LIVE_JS),
        ("js/leaderboard.js", leaderboard_js),
    ):
        stripped = _strip_js_comments(source)
        assert "stop_loss" not in stripped, name
        assert "take_profit" not in stripped, name


# ── #586: catching-up rows are printed, flagged, and not reported missing ───
#
# These evaluate the whole of js/live-leaderboard.js under node against a fake
# DOM instead of lifting single functions, so a helper another change adds to
# the render path is exercised too rather than raising a ReferenceError here.

_LIVE_FAKE_DOM = """
const els = {};
function el(id) {
  if (!els[id]) {
    els[id] = {
      id, textContent: '', innerHTML: '', hidden: false, dataset: {}, style: {},
      classList: { toggle() {}, add() {}, remove() {}, contains() { return false; } },
      setAttribute() {}, querySelector() { return null; },
    };
  }
  return els[id];
}
globalThis.window = globalThis;
globalThis.document = {
  getElementById: el,
  querySelector: () => null,
  querySelectorAll: () => [],
  addEventListener() {},
};
globalThis.API_BASE = '';
"""


def _render_catching_up_board(payload, select_id):
    """Render `payload` the way loadLiveLeaderboardData does; return the banner,
    the table body and the detail panel for the row `select_id`."""
    from dashboard.backend.tests._frontend_source import APP_JS, fn_body, run_node

    script = (
        _LIVE_FAKE_DOM
        + fn_body("function escapeHtml(", APP_JS)
        + "\n"
        + _LIVE_JS
        + f"\nconst payload = {json.dumps(payload)};\n"
        + "livePayload = payload;\n"
        + "updateLiveHeader(payload);\n"
        + "populateLiveTable();\n"
        + f"renderLiveDetail(payload.entries.find((e) => e.entry_id === {json.dumps(select_id)}));\n"
        + "process.stdout.write(JSON.stringify({\n"
        + "  banner: els.livePhaseBanner.textContent,\n"
        + "  bannerHidden: els.livePhaseBanner.hidden,\n"
        + "  table: els.liveLeaderboardTableBody.innerHTML,\n"
        + "  detail: els.liveSelectedDetail.innerHTML,\n"
        + "}));\n"
    )
    return run_node(script)


def _catching_up_entry(entry_id, name, status, *, value, snapshot_end, is_model=True):
    """One board row in the shape live._entry_from_strategy serves."""
    printed = value is not None
    return {
        "entry_id": entry_id,
        "model": name,
        "team_name": name,
        "team_badge": "Model" if is_model else "Index",
        "is_model": is_model,
        "status": status,
        "rank": None,
        "portfolio_value": value,
        "cumulative_return": 0.05 if printed else None,
        "sharpe_ratio": None,
        "max_drawdown": None,
        "num_trades": 3 if printed else None,
        "snapshot_end": snapshot_end,
        "equity_curve": (
            [{"timestamp": "2026-08-03T10:00:00-04:00", "equity": 100000}] if printed else []
        ),
    }


def _catching_up_payload(*entries):
    return {
        "period": "live",
        "board_title": "Live Leaderboard",
        "phase_label": "Season 0 · preview",
        "window": {"label": "2026-08-01 — 2026-08-31"},
        "entries": list(entries),
        "live_status": {
            "has_prints": True,
            "snapshot_stale": False,
            "frozen_through": "2026-08-27",
            "snapshot_end": "2026-08-27",
            "session_state": "closed",
            # The server counts only ranked (frozen) rows here, so a model
            # mid-replay reads 0. Held at 0 in every case: the banner must not
            # depend on it.
            "models_cached": 0,
            "catching_up": [e["entry_id"] for e in entries if e["status"] == "catching_up"],
            "trading_days_elapsed": 19,
            "trading_days_total": 21,
        },
    }


_SPY_ROW = _catching_up_entry(
    "spy_index", "SPY", "frozen", value=101000.0, snapshot_end="2026-08-27", is_model=False
)


def _deepseek(status, *, value=105000.0, snapshot_end="2026-08-20"):
    return _catching_up_entry(
        "deepseek_v4_pro", "DeepSeek V4 Pro", status, value=value, snapshot_end=snapshot_end
    )


@_needs_node
def test_a_catching_up_model_is_printed_and_flagged_with_its_own_date():
    out = _render_catching_up_board(
        _catching_up_payload(_SPY_ROW, _deepseek("catching_up")), "deepseek_v4_pro"
    )
    row = out["table"].split('data-live-entry="deepseek_v4_pro"')[1].split("</tr>")[0]
    assert (
        '<span class="team-badge" title="Catching up · through 2026-08-20">Catching up</span>'
    ) in row
    assert "5.00%" in row, "a catching-up row prints its figures"
    assert (
        '<span class="team-detail-label">Status</span>'
        '<span class="team-detail-value">Catching up · through 2026-08-20</span>'
    ) in out["detail"]
    assert "$105,000.00" in out["detail"]
    assert "Awaiting freeze" not in out["detail"]


@_needs_node
def test_the_banner_names_the_catching_up_model_instead_of_calling_it_missing():
    out = _render_catching_up_board(
        _catching_up_payload(_SPY_ROW, _deepseek("catching_up")), "deepseek_v4_pro"
    )
    assert out["bannerHidden"] is False
    assert "Model curves appear" not in out["banner"]
    assert out["banner"] == (
        "DeepSeek V4 Pro is catching up (through 2026-08-20): it is shown but "
        "not ranked until it reaches the latest session."
    )


@_needs_node
def test_the_banner_lists_every_catching_up_entry():
    qwen = _catching_up_entry(
        "qwen3_7_plus", "Qwen3.7 Plus", "catching_up", value=99000.0, snapshot_end="2026-08-22"
    )
    out = _render_catching_up_board(
        _catching_up_payload(_SPY_ROW, _deepseek("catching_up"), qwen), "qwen3_7_plus"
    )
    assert out["banner"] == (
        "2 entries are catching up (DeepSeek V4 Pro through 2026-08-20, "
        "Qwen3.7 Plus through 2026-08-22): they are shown but not ranked "
        "until they reach the latest session."
    )


@_needs_node
def test_a_model_with_no_curve_still_gets_the_models_pending_banner():
    out = _render_catching_up_board(
        _catching_up_payload(_SPY_ROW, _deepseek("pending", value=None, snapshot_end=None)),
        "deepseek_v4_pro",
    )
    assert out["banner"] == (
        "Baselines frozen through 2026-08-27. Model curves appear once the nightly "
        "refresh deploys them; they are never computed on page load."
    )
    assert "Catching up" not in out["table"]
    assert "Awaiting freeze" in out["detail"]
    assert "Status</span>" not in out["detail"]


@_needs_node
def test_a_catching_up_row_with_no_value_reads_as_pending_not_catching_up():
    """A row the server marks catching_up but sends no value for cannot print:
    it is pending, and a 'Catching up' badge beside 'Awaiting freeze' would say
    both at once."""
    out = _render_catching_up_board(
        _catching_up_payload(_SPY_ROW, _deepseek("catching_up", value=None)),
        "deepseek_v4_pro",
    )
    assert "Catching up" not in out["table"]
    assert "Status</span>" not in out["detail"]
    assert "Awaiting freeze" in out["detail"]
    assert out["banner"].startswith("Baselines frozen through 2026-08-27.")


@_needs_node
def test_a_stale_snapshot_banner_outranks_the_catching_up_line_and_the_badge_stays():
    payload = _catching_up_payload(_SPY_ROW, _deepseek("catching_up"))
    payload["live_status"]["snapshot_stale"] = True
    payload["live_status"]["frozen_through"] = "2026-08-28"
    out = _render_catching_up_board(payload, "deepseek_v4_pro")
    assert out["banner"].startswith("Showing the snapshot through 2026-08-27.")
    assert ">Catching up</span>" in out["table"]


@_needs_node
def test_a_current_board_shows_no_banner_and_no_badge():
    out = _render_catching_up_board(
        _catching_up_payload(_SPY_ROW, _deepseek("frozen", snapshot_end="2026-08-27")),
        "deepseek_v4_pro",
    )
    assert out["bannerHidden"] is True
    assert out["banner"] == ""
    assert "Catching up" not in out["table"]
    assert "Catching up" not in out["detail"]


def _ok_sharpe(value):
    """The Sharpe fields boardSharpeValue accepts: status 'ok' plus a finite interval."""
    return {
        "sharpe_status": "ok",
        "sharpe_ratio": value,
        "sharpe_ci_low": value - 1,
        "sharpe_ci_high": value + 1,
        "sharpe_days": 5,
    }


@_needs_node
def test_catching_up_rows_sort_after_current_rows_and_before_pending_ones():
    """A catching-up row is shown but not ranked, and its figures are from an
    earlier session than the rows beside it. Sorting a column must not slot it
    in among the current rows, in either direction: DeepSeek's Sharpe would
    sort first descending and Qwen's first ascending."""
    from dashboard.backend.tests._frontend_source import run_node

    rows = [
        _catching_up_entry(
            "claude_haiku_4_5", "Claude Haiku 4.5", "pending", value=None, snapshot_end=None
        ),
        {**_deepseek("catching_up"), **_ok_sharpe(9.0)},
        {**_SPY_ROW, **_ok_sharpe(1.0)},
        {
            **_catching_up_entry(
                "qwen3_7_plus", "Qwen3.7 Plus", "catching_up",
                value=99000.0, snapshot_end="2026-08-22",
            ),
            **_ok_sharpe(-5.0),
        },
        {
            **_catching_up_entry(
                "equal_weight_djia", "Equal Weight DJIA", "frozen",
                value=100800.0, snapshot_end="2026-08-27", is_model=False,
            ),
            **_ok_sharpe(0.5),
        },
    ]
    script = (
        _LIVE_FAKE_DOM
        + "\n"
        + _board_sharpe_js()
        + "\n"
        + _LIVE_JS
        + f"\nlivePayload = {{ entries: {json.dumps(rows)} }};\n"
        + "liveSortKey = 'sharpe';\n"
        + "liveSortDir = 'desc'; const desc = liveSortedEntries().map((e) => e.entry_id);\n"
        + "liveSortDir = 'asc'; const asc = liveSortedEntries().map((e) => e.entry_id);\n"
        + "process.stdout.write(JSON.stringify({ desc, asc }));\n"
    )
    assert run_node(script) == {
        "desc": [
            "spy_index", "equal_weight_djia",
            "deepseek_v4_pro", "qwen3_7_plus",
            "claude_haiku_4_5",
        ],
        "asc": [
            "equal_weight_djia", "spy_index",
            "qwen3_7_plus", "deepseek_v4_pro",
            "claude_haiku_4_5",
        ],
    }


def test_the_banner_no_longer_reads_models_cached():
    assert "models_cached" not in _strip_js_comments(_LIVE_JS)
