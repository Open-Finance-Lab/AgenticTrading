"""Competition board + home rank list: Sharpe is displayed, never recomputed.

The server publishes P19's daily Sharpe with its 95% interval
(``sharpe_ratio``/``sharpe_ci_low``/``sharpe_ci_high``/``sharpe_days``/
``sharpe_status`` per entry, ``sharpe_basis`` on the payload). These run the
shipped JS under node -- the whole file where it has no top-level side effects,
so a helper added later cannot turn the harness into a ReferenceError.
"""

import json
import re
import shutil

import pytest

from dashboard.backend.domain.backtesting.metrics import SHARPE_MIN_DAYS, daily_sharpe_basis
from dashboard.backend.tests._frontend_source import (
    APP_HTML,
    FRONTEND,
    fn_body,
    run_node,
    strip_comments,
)

_LEADERBOARD_JS = (FRONTEND / "js" / "leaderboard.js").read_text(encoding="utf-8")
_LIVE_JS = (FRONTEND / "js" / "live-leaderboard.js").read_text(encoding="utf-8")
_HOME_JS = (FRONTEND / "home-page.js").read_text(encoding="utf-8")

_needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

_BASIS = daily_sharpe_basis()
_OK = {
    "sharpe_status": "ok",
    "sharpe_ratio": 2.611715,
    "sharpe_ci_low": -11.39649,
    "sharpe_ci_high": 16.619919,
    "sharpe_days": 5,
}
_TITLE = (
    "Return above what cash would have earned, divided by how much the daily "
    "returns swung. Based on 5 trading days; cash rate 0%/yr; scaled to a yearly figure."
)
_HEADER_TITLE = (
    "Sharpe ratio. Return above what cash would have earned, divided by how much the "
    f"daily returns swung, scaled to a yearly figure. Shown from {SHARPE_MIN_DAYS} "
    "trading days, always with its 95% interval."
)
# Every shape the payload can send, plus the malformed ones a stale or broken
# payload could: the two boards must answer each the same way.
_CASES = [
    (_OK, _BASIS),
    ({**_OK, "sharpe_ratio": -0.5, "sharpe_ci_low": -14.5, "sharpe_ci_high": 13.5}, _BASIS),
    (_OK, {**_BASIS, "rf_annual": 0.0252}),
    (_OK, None),
    ({**_OK, "sharpe_ci_high": None}, _BASIS),
    ({"sharpe_status": "insufficient_days", "sharpe_ratio": None, "sharpe_days": 3}, _BASIS),
    ({"sharpe_status": "insufficient_days", "sharpe_days": 2}, None),
    ({"sharpe_status": "no_variation", "sharpe_ratio": None, "sharpe_days": 6}, _BASIS),
    ({"sharpe_status": "undefined", "sharpe_ratio": None, "sharpe_days": 6}, _BASIS),
    ({"sharpe_status": "pending", "sharpe_ratio": None, "sharpe_days": 0}, _BASIS),
    ({"sharpe_ratio": 1.5}, _BASIS),
    (None, _BASIS),
    ({**_OK, "sharpe_ratio": -0.004, "sharpe_ci_low": -0.004, "sharpe_ci_high": 0.003}, _BASIS),
    ({k: v for k, v in _OK.items() if k != "sharpe_days"}, _BASIS),
    # The two files coerce through different helpers; hold them together on
    # the off-type shapes a hand-edited or proxied payload could carry.
    ({**_OK, "sharpe_ratio": "2.6", "sharpe_days": "5"}, _BASIS),
    ({**_OK, "sharpe_ratio": "Infinity"}, _BASIS),
    ({**_OK, "sharpe_ci_low": ""}, _BASIS),
    ({"sharpe_status": "insufficient_days", "sharpe_days": 3}, {**_BASIS, "min_days": "7"}),
]


def _board_script(*lines):
    """app.js's escapeHtml, the whole of leaderboard.js, then ``lines``."""
    return "\n".join(["const window = {};", fn_body("function escapeHtml("), _LEADERBOARD_JS, *lines])


def _board_row(entry_id, **fields):
    return {
        "entry_id": entry_id, "team_name": entry_id, "model": entry_id, "team_badge": "Model",
        "is_model": True, "rank": 1, "portfolio_value": 100000, "cumulative_return": 0.01,
        "max_drawdown": -0.01, "equity_curve": [], **fields,
    }


def test_board_js_never_recomputes_sharpe():
    for name, source in (("leaderboard.js", _LEADERBOARD_JS), ("home-page.js", _HOME_JS)):
        code = strip_comments(source)
        assert "Math.sqrt" not in code, name
        assert "stdev" not in code.lower(), name
        assert "Number(entry.sharpe_ratio || 0)" not in code, name


@_needs_node
def test_competition_formatter_shows_the_value_with_its_interval():
    out = run_node(_board_script(
        f"console.log(JSON.stringify(formatBoardSharpe({json.dumps(_OK)}, {json.dumps(_BASIS)})));"
    ))
    assert out == {"main": "2.61", "sub": "95% interval -11.40 to 16.62", "title": _TITLE}


@_needs_node
def test_both_boards_word_sharpe_identically():
    """liveFormatSharpe delegates to formatBoardSharpe (one mapping, not two
    copies); this holds the delegation to one answer for every payload shape."""
    script = "\n".join([
        "const window = {};",
        _LEADERBOARD_JS,
        fn_body("function liveFormatSharpe(", _LIVE_JS),
        f"const cases = {json.dumps(_CASES)};",
        "console.log(JSON.stringify(cases.map(([e, b]) => [formatBoardSharpe(e, b), liveFormatSharpe(e, b)])));",
    ])
    pairs = run_node(script)
    for board, live in pairs:
        assert board == live
    mains = [board["main"] for board, _live in pairs]
    assert mains == ["2.61", "-0.50", "2.61", "2.61", "—", "—", "—", "—", "—", "—", "—", "—", "0.00", "2.61", "2.60", "—", "—", "—"]
    subs = {json.dumps(e, sort_keys=True): board["sub"] for (e, _b), (board, _l) in zip(_CASES, pairs)}
    assert subs[json.dumps(_CASES[7][0], sort_keys=True)] == "Returns never moved"
    assert subs[json.dumps(_CASES[8][0], sort_keys=True)] == "Cannot be computed"


@_needs_node
def test_competition_row_and_detail_render_the_interval_and_never_zero():
    ok = _board_row("up", **_OK)
    young = _board_row("young", sharpe_status="insufficient_days", sharpe_ratio=None, sharpe_days=2)
    out = run_node(_board_script(
        f"leaderboardPayload = {{ sharpe_basis: {json.dumps(_BASIS)}, entries: [] }};",
        "console.log(JSON.stringify({",
        f"  okRow: renderLeaderboardRowHtml({json.dumps(ok)}),",
        f"  youngRow: renderLeaderboardRowHtml({json.dumps(young)}),",
        f"  okDetail: renderLeaderboardDetailHtml({json.dumps(ok)}, 12),",
        f"  youngDetail: renderLeaderboardDetailHtml({json.dumps(young)}, 12),",
        "}));",
    ))
    assert "2.61" in out["okRow"] and "95% interval -11.40 to 16.62" in out["okRow"]
    assert f'title="{_TITLE}"' in out["okRow"]
    assert "Needs 5 trading days (have 2)" in out["youngRow"]
    assert '<span class="metric-value-text">—</span>' in out["youngRow"]
    assert '<span class="metric-value-text">0.00</span>' not in out["youngRow"]
    assert "Sharpe ratio" in out["okDetail"]
    assert "2.61 · 95% interval -11.40 to 16.62" in out["okDetail"]
    assert "— · Needs 5 trading days (have 2)" in out["youngDetail"]


@_needs_node
def test_competition_sort_puts_a_null_sharpe_last_both_ways():
    rows = [
        _board_row("none", sharpe_status="insufficient_days", sharpe_ratio=None, sharpe_days=2),
        _board_row("neg", sharpe_status="ok", sharpe_ratio=-0.5, sharpe_ci_low=-14.0, sharpe_ci_high=13.0, sharpe_days=5),
        _board_row("pos", **_OK),
        _board_row("stale", sharpe_ratio=9.7),
    ]
    out = run_node(_board_script(
        f"leaderboardPayload = {{ entries: {json.dumps(rows)} }}; currentLeaderboardSort = 'sharpe';",
        "currentLeaderboardSortDir = 'desc'; const desc = getFilteredLeaderboardEntries().map((e) => e.entry_id);",
        "currentLeaderboardSortDir = 'asc'; const asc = getFilteredLeaderboardEntries().map((e) => e.entry_id);",
        "console.log(JSON.stringify({ desc, asc }));",
    ))
    # "stale" is an old payload's hourly number: shown as "—", so absent too.
    assert out["desc"][:2] == ["pos", "neg"] and set(out["desc"][2:]) == {"none", "stale"}
    assert out["asc"][:2] == ["neg", "pos"] and set(out["asc"][2:]) == {"none", "stale"}


@_needs_node
def test_home_rank_row_borrows_the_boards_wording():
    lifted = fn_body("function homeFormatSharpe(", _HOME_JS)
    young = {"sharpe_status": "insufficient_days", "sharpe_ratio": None, "sharpe_days": 2}
    out = run_node(_board_script(
        lifted,
        "console.log(JSON.stringify({",
        f"  ok: homeFormatSharpe({json.dumps(_OK)}, {json.dumps(_BASIS)}),",
        f"  young: homeFormatSharpe({json.dumps(young)}, {json.dumps(_BASIS)}),",
        "}));",
    ))
    assert out["ok"] == {
        "text": "2.61",
        "sub": "95% interval -11.40 to 16.62",
        "title": f"95% interval -11.40 to 16.62 · {_TITLE}",
    }
    assert out["young"] == {
        "text": "—",
        "sub": "Needs 5 trading days (have 2)",
        "title": "Needs 5 trading days (have 2)",
    }
    alone = run_node("\n".join([
        "const window = {};", lifted,
        f"console.log(JSON.stringify(homeFormatSharpe({json.dumps(_OK)}, null)));",
    ]))
    assert alone == {"text": "—", "sub": "", "title": ""}


def test_home_rank_list_passes_the_basis_and_renders_the_cell():
    code = strip_comments(_HOME_JS)
    body = fn_body("function renderEntries(", code)
    assert "homeFormatSharpe(entry, sharpeBasis)" in body
    assert "sharpe.title" in body and "homeEscape(sharpe.text)" in body
    # The interval is in the cell's text for a screen reader and the reading
    # order, not only in a title a touch or keyboard user cannot reach.
    assert '<span class="sr-only">, ${homeEscape(sharpe.sub)}</span>' in body
    assert "renderEntries(models, { sharpeBasis: payload.sharpe_basis || null })" in code


def test_leaderboard_js_exports_the_formatter_for_home():
    assert "window.formatBoardSharpe = formatBoardSharpe;" in _LEADERBOARD_JS


def test_sharpe_headers_say_what_the_number_is():
    competition = APP_HTML.split('id="leaderboardTableBody"')[0]
    competition = competition[competition.rindex("<table") :]
    match = re.search(r'<th[^>]*data-sort="sharpe"[^>]*title="([^"]*)"', competition)
    assert match, "the Competition Sharpe header lost its tooltip"
    assert match.group(1) == _HEADER_TITLE
    live = APP_HTML.split('id="liveLeaderboardView"')[1].split('id="leaderboardView"')[0]
    live_title = re.search(r'<th[^>]*data-sort="sharpe"[^>]*title="([^"]*)"', live).group(1)
    assert live_title == _HEADER_TITLE, "one definition, worded once on both boards"
    assert "annualized from hourly results" not in APP_HTML


def test_every_sharpe_sink_is_escaped():
    """The formatters only emit numbers and fixed text today, so no fixture can
    show a missing escape; pin the sinks by source instead."""
    raw = re.compile(r"\$\{\s*sharpe(?:Text|\.(?:main|sub|text|title))\s*\}")
    for name, source in (
        ("leaderboard.js", _LEADERBOARD_JS),
        ("live-leaderboard.js", _LIVE_JS),
        ("home-page.js", _HOME_JS),
    ):
        code = strip_comments(source)
        assert not raw.search(code), name
    assert _LEADERBOARD_JS.count("escapeHtml(sharpe.main)") == 1
    assert _LEADERBOARD_JS.count("escapeHtml(sharpe.sub)") == 1
    assert _LEADERBOARD_JS.count("escapeHtml(sharpe.title)") == 2
    assert _LIVE_JS.count("escapeHtml(sharpe.title)") == 2
    assert "homeEscape(sharpe.text)" in _HOME_JS and "homeEscape(sharpe.title)" in _HOME_JS
    assert "homeEscape(sharpe.sub)" in _HOME_JS
