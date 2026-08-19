"""The landing hero draws the same live board the signed-in Home screen draws.

TWO TIERS, AND THE SECOND SKIPS IN CI. The static-shape guards below are source
regex/substring checks against `landing/src` -- nothing in CI builds or
type-checks the landing, so these are also the only layer that can compare the
landing's selection rule against screen 0's, since the two live in different
bundles and one of them ships minified. They run everywhere but exercise no
TypeScript at all.

The behavioural tests further down transpile `leaderboard.ts` with the esbuild
inside `dashboard/landing/node_modules` and run it under node, so they need an
`npm install` CI does not do -- they skip there. A green CI therefore says the
static shapes agree, NOT that `selectBoardEntries`/`buildBoardData` were ever
executed: run this suite locally before shipping a change to this module.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_LIB = _ROOT / "landing" / "src" / "lib"
_HOME_JS = (_ROOT / "frontend" / "home-page.js").read_text(encoding="utf-8")
_LEADERBOARD_JS = (_ROOT / "frontend" / "js" / "leaderboard.js").read_text(encoding="utf-8")
_LIB_TS_PATH = _LIB / "leaderboard.ts"
_LIB_TS = _LIB_TS_PATH.read_text(encoding="utf-8")
_ESBUILD = _ROOT / "landing" / "node_modules" / ".bin" / "esbuild"


def _js_array(source: str, name: str) -> list[str]:
    match = re.search(rf"{name}\s*=\s*\[(.*?)\]", source, re.S)
    assert match, f"{name} not found"
    return re.findall(r"['\"]([^'\"]+)['\"]", match.group(1))


def test_the_hero_and_screen_zero_pick_the_same_baselines():
    """Screen 0's own source states the reason: seven model curves with no
    baseline leave the reader nothing to judge them against. That is equally
    true on the acquisition page, and it is what "sync the two pages" means
    concretely rather than cosmetically."""
    assert _js_array(_HOME_JS, "HOME_CHART_BASELINE_IDS") == _js_array(
        _LIB_TS, "BOARD_BASELINE_IDS"
    )


def test_the_hero_uses_the_same_model_test_as_screen_zero():
    assert "is_model" in _LIB_TS and "team_badge" in _LIB_TS
    assert '"Model"' in _LIB_TS or "'Model'" in _LIB_TS


def test_baseline_colours_are_keyed_on_entry_id_not_on_a_display_label():
    """`LEADERBOARD_STYLES` on /app keys on the label for historical reasons, but
    the label is copy and can be renamed in dashboard/config/leaderboard.json
    with nothing failing. `id` is that file's primary key and reaches the client
    as `entry.entry_id`; screen 0's HOME_CHART_BASELINE_IDS already made this
    correction."""
    styles = re.search(r"BASELINE_STYLES[^=]*=\s*\{(.*?)\n\};", _LIB_TS, re.S)
    assert styles, "BASELINE_STYLES not found"
    body = styles.group(1)
    assert "buy_hold_djia" in body and "djia_index" in body
    assert "Buy & Hold" not in body and '"DJIA"' not in body


def test_the_model_palette_is_the_same_list_in_the_same_order():
    """The hero and /app must colour the same model the same way -- a visitor who
    signs up lands on a board whose curves they have already learned. The order
    matters as much as the members: /app assigns MODEL_COLOR_PALETTE[n] in
    first-seen order over the ranked payload, so the hero must index models in
    payload order too."""
    assert _js_array(_LEADERBOARD_JS, "MODEL_COLOR_PALETTE") == _js_array(
        _LIB_TS, "MODEL_COLOR_PALETTE"
    )


def test_the_fetch_is_root_relative_and_names_no_origin():
    """Vercel rewrites /api/:path* to Render (dashboard/frontend/vercel.json), and
    test_frontend_api_base.py requires an EMPTY production base for exactly that
    reason -- it calls a hardcoded Render origin a same-origin cookie auth
    regression. MarketTicker.tsx's apiBase() survives that guard only because it
    excludes minified assets/; do not copy it."""
    assert '"/api/v1/leaderboard' in _LIB_TS or "'/api/v1/leaderboard" in _LIB_TS
    assert "onrender.com" not in _LIB_TS
    assert "window.location.origin" not in _LIB_TS


def test_the_fetch_is_bounded_by_an_abort_signal():
    """Render's free tier cold-starts in 30-60s. A fetch with no ceiling leaves
    the card shimmering forever, which is the failure state this design most
    wants to be distinguishable."""
    assert "AbortSignal" in _LIB_TS or "signal" in _LIB_TS


def test_a_failed_request_throws_rather_than_returning_an_empty_board():
    """An empty board and a broken backend must not produce the same value. That
    is the fail-closed-is-not-fail-visible failure in miniature, and it is why
    the caller gets three states rather than two."""
    assert re.search(r"throw new Error", _LIB_TS), (
        "a non-ok response must raise, not resolve to an empty board"
    )
    assert "res.ok" in _LIB_TS or "response.ok" in _LIB_TS


# ---------------------------------------------------------------------------
# Behavioural tier -- see the module docstring for why this skips in CI.
# ---------------------------------------------------------------------------


def _run_ts(script: str):
    """Transpile leaderboard.ts to CJS and run `script` against it under node."""
    node = shutil.which("node")
    if not node or not _ESBUILD.is_file():
        pytest.skip("node and dashboard/landing/node_modules are required")
    bundled = subprocess.run(
        [str(_ESBUILD), str(_LIB_TS_PATH), "--bundle", "--format=cjs",
         "--platform=node", "--log-level=error"],
        capture_output=True, text=True, timeout=60,
    )
    assert bundled.returncode == 0, bundled.stderr
    proc = subprocess.run(
        [node, "-e", bundled.stdout + "\n" + script],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


# The real twelve-entry roster from dashboard/config/leaderboard.json, in
# payload (ranked) order -- deliberately NOT grouped by type, so a test that
# passes here proves selectBoardEntries reorders rather than merely preserving
# whatever order the fixture happened to already be in.
_TWELVE_ENTRY_FIXTURE_JS = """
const entries = [
  {entry_id: 'spy_index', team_name: 'Agentic Trading Lab', team_badge: 'Market Index', model: 'SPY', is_model: false, cumulative_return: 0.01, portfolio_value: 10100, initial_equity: 10000, equity_curve: []},
  {entry_id: 'claude_haiku_4_5', team_name: 'Claude Haiku 4.5', team_badge: 'Model', model: 'Claude Haiku 4.5', is_model: true, cumulative_return: 0.02, portfolio_value: 10200, initial_equity: 10000, equity_curve: []},
  {entry_id: 'mean_variance_djia', team_name: 'Agentic Trading Lab', team_badge: 'Baseline Strategy', model: 'Mean-Variance', is_model: false, cumulative_return: -0.01, portfolio_value: 9900, initial_equity: 10000, equity_curve: []},
  {entry_id: 'gpt_5_5', team_name: 'GPT-5.5', team_badge: 'Model', model: 'GPT-5.5', is_model: true, cumulative_return: 0.028, portfolio_value: 10280, initial_equity: 10000, equity_curve: []},
  {entry_id: 'djia_index', team_name: 'Agentic Trading Lab', team_badge: 'Market Index', model: 'DJIA', is_model: false, cumulative_return: 0.005, portfolio_value: 10050, initial_equity: 10000, equity_curve: []},
  {entry_id: 'gemini_3_1_pro_preview', team_name: 'Gemini 3.1 Pro Preview', team_badge: 'Model', model: 'Gemini 3.1 Pro Preview', is_model: true, cumulative_return: 0.0156, portfolio_value: 10156, initial_equity: 10000, equity_curve: []},
  {entry_id: 'buy_hold_djia', team_name: 'Agentic Trading Lab', team_badge: 'Baseline Strategy', model: 'Buy & Hold', is_model: false, cumulative_return: 0.008, portfolio_value: 10080, initial_equity: 10000, equity_curve: []},
  {entry_id: 'claude_sonnet_4_6', team_name: 'Claude Sonnet 4.6', team_badge: 'Model', model: 'Claude Sonnet 4.6', is_model: true, cumulative_return: 0.0312, portfolio_value: 10312, initial_equity: 10000, equity_curve: []},
  {entry_id: 'equal_weight_djia', team_name: 'Agentic Trading Lab', team_badge: 'Baseline Strategy', model: 'Equal-Weight', is_model: false, cumulative_return: 0.003, portfolio_value: 10030, initial_equity: 10000, equity_curve: []},
  {entry_id: 'deepseek_v4_pro', team_name: 'DeepSeek V4 Pro', team_badge: 'Model', model: 'DeepSeek V4 Pro', is_model: true, cumulative_return: 0.0749, portfolio_value: 10749, initial_equity: 10000, equity_curve: []},
  {entry_id: 'qwen3_7_plus', team_name: 'Qwen3.7 Plus', team_badge: 'Model', model: 'Qwen3.7 Plus', is_model: false, cumulative_return: 0.0249, portfolio_value: 10249, initial_equity: 10000, equity_curve: []},
  {entry_id: 'nemotron_3_nano_30b', team_name: 'Nemotron 3 Nano 30B', team_badge: 'Model', model: 'Nemotron 3 Nano 30B', is_model: true, cumulative_return: -0.004, portfolio_value: 9960, initial_equity: 10000, equity_curve: []},
];
"""


def test_select_board_entries_returns_nine_of_twelve_models_first_then_baselines():
    """Screen 0 draws every model plus exactly the two reference baselines --
    9 of the 12 entries dashboard/config/leaderboard.json currently carries.
    The fixture interleaves models and baselines by rank; the assertion on
    order is therefore a real check of the models.concat(baselines)
    regrouping, not an accident of input order."""
    result = _run_ts(
        _TWELVE_ENTRY_FIXTURE_JS
        + """
const selected = module.exports.selectBoardEntries(entries);
console.log(JSON.stringify(selected.map((e) => e.entry_id)));
"""
    )
    assert result == [
        "claude_haiku_4_5", "gpt_5_5", "gemini_3_1_pro_preview", "claude_sonnet_4_6",
        "deepseek_v4_pro", "qwen3_7_plus", "nemotron_3_nano_30b",
        "djia_index", "buy_hold_djia",
    ]


def test_select_board_entries_honours_the_team_badge_fallback_when_is_model_is_false():
    """`qwen3_7_plus` in the fixture above carries `is_model: false,
    team_badge: 'Model'` -- the OR-with-fallback branch home-page.js's
    `homeChartEntries` mirrors, which no live entry currently exercises but
    which this module must not silently narrow to a bare `is_model` check.
    Mutating `||` to `&&` in the shipped module drops this entry from the
    model bucket; the assertion below is what catches that (see the mutation
    check in the task report)."""
    result = _run_ts(
        _TWELVE_ENTRY_FIXTURE_JS
        + """
const selected = module.exports.selectBoardEntries(entries);
console.log(JSON.stringify(selected.map((e) => e.entry_id)));
"""
    )
    assert "qwen3_7_plus" in result


def test_select_board_entries_excludes_a_baseline_not_on_the_allowlist():
    """`mean_variance_djia`, `equal_weight_djia` and `spy_index` are real
    baseline/index rows in the config -- none is one of the two ids screen 0
    draws. They must not leak into the selection just for being
    `is_model: false` with SOME recognisable badge."""
    result = _run_ts(
        _TWELVE_ENTRY_FIXTURE_JS
        + """
const selected = module.exports.selectBoardEntries(entries);
console.log(JSON.stringify(selected.map((e) => e.entry_id)));
"""
    )
    assert "mean_variance_djia" not in result
    assert "equal_weight_djia" not in result
    assert "spy_index" not in result


# Five entries: two full three-point curves (A, E), one curve missing an
# INTERIOR point (B: t1 and t3 only), one curve of different length than the
# rest (C: t1 only), and one entry with NO curve at all (D) -- the Important-1
# case: a real entry with `equity_curve: []`, which the server never actually
# emits (chart_equity_curve always synthesises an opening point) but which
# this module must not treat the same as "broken". F is a fifth model placed
# AFTER D in payload order, to catch a skipped entry burning D's palette slot
# and shifting F's colour (Important 3).
_RAGGED_CURVE_FIXTURE_JS = """
const entries = [
  {entry_id: 'claude_haiku_4_5', team_name: 'Claude Haiku 4.5', team_badge: 'Model', model: 'Claude Haiku 4.5', is_model: true, cumulative_return: 0.02, portfolio_value: 10200, initial_equity: 10000,
   equity_curve: [{timestamp: '2026-04-15T14:00:00+00:00', equity: 10000}, {timestamp: '2026-04-15T15:00:00+00:00', equity: 10100}, {timestamp: '2026-04-15T16:00:00+00:00', equity: 10200}]},
  {entry_id: 'gpt_5_5', team_name: 'GPT-5.5', team_badge: 'Model', model: 'GPT-5.5', is_model: true, cumulative_return: -0.01, portfolio_value: 9900, initial_equity: 10000,
   equity_curve: [{timestamp: '2026-04-15T14:00:00+00:00', equity: 10000}, {timestamp: '2026-04-15T16:00:00+00:00', equity: 9900}]},
  {entry_id: 'deepseek_v4_pro', team_name: 'DeepSeek V4 Pro', team_badge: 'Model', model: 'DeepSeek V4 Pro', is_model: true, cumulative_return: 0.05, portfolio_value: 10500, initial_equity: 10000,
   equity_curve: []},
  {entry_id: 'buy_hold_djia', team_name: 'Agentic Trading Lab', team_badge: 'Baseline Strategy', model: 'Buy & Hold', is_model: false, cumulative_return: 0.0, portfolio_value: 10000, initial_equity: 10000,
   equity_curve: [{timestamp: '2026-04-15T14:00:00+00:00', equity: 10000}]},
  {entry_id: 'qwen3_7_plus', team_name: 'Qwen3.7 Plus', team_badge: 'Model', model: 'Qwen3.7 Plus', is_model: true, cumulative_return: 0.015, portfolio_value: 10150, initial_equity: 10000,
   equity_curve: [{timestamp: '2026-04-15T14:00:00+00:00', equity: 10000}, {timestamp: '2026-04-15T15:00:00+00:00', equity: 10075}, {timestamp: '2026-04-15T16:00:00+00:00', equity: 10150}]},
  {entry_id: 'djia_index', team_name: 'Agentic Trading Lab', team_badge: 'Market Index', model: 'DJIA', is_model: false, cumulative_return: 0.01, portfolio_value: 10100, initial_equity: 10000,
   equity_curve: [{timestamp: '2026-04-15T14:00:00+00:00', equity: 10000}, {timestamp: '2026-04-15T15:00:00+00:00', equity: 10050}, {timestamp: '2026-04-15T16:00:00+00:00', equity: 10100}]},
];
const board = module.exports.buildBoardData({entries, window: {label: 'test window'}});
"""


def test_a_curveless_entry_still_stands_in_standings_but_not_in_the_chart_series():
    """Important 1: DeepSeek's `equity_curve: []` must not drop it out of the
    standings the way it drops out of the chart series. Rank/return come from
    `cumulative_return`, present regardless of curve data -- mirroring /app's
    rank list, which shows a model independent of whether it has a drawable
    curve."""
    result = _run_ts(
        _RAGGED_CURVE_FIXTURE_JS
        + """
console.log(JSON.stringify({
  standingsKeys: board.standings.map((s) => s.key),
  seriesKeys: board.series.map((s) => s.key),
}));
"""
    )
    assert "deepseek_v4_pro" in result["standingsKeys"], (
        "a curve-less entry must still stand in the standings"
    )
    assert "deepseek_v4_pro" not in result["seriesKeys"], (
        "a curve-less entry has nothing to plot and must not enter series"
    )
    # Ragged curves (missing interior point on gpt_5_5, shorter buy_hold_djia)
    # still produce a series -- they are not "curve-less".
    assert "gpt_5_5" in result["seriesKeys"]
    assert "buy_hold_djia" in result["seriesKeys"]


def test_a_missing_interior_point_null_fills_rather_than_shifting_the_axis():
    """gpt_5_5's curve has t1 and t3 but not t2 -- the value at the shared t2
    tick must be `null` (a gap Recharts can skip), not silently reindexed onto
    t3's value, which would misalign the point against the shared time axis
    every other series is drawn against."""
    result = _run_ts(
        _RAGGED_CURVE_FIXTURE_JS
        + """
const gpt = board.series.find((s) => s.key === 'gpt_5_5');
console.log(JSON.stringify({times: board.times, values: gpt.values}));
"""
    )
    assert result["times"] == ["2026-04-15T14:00", "2026-04-15T15:00", "2026-04-15T16:00"]
    assert result["values"][1] is None, "the missing interior point must null-fill, not shift"
    assert result["values"][0] == pytest.approx(0.0)
    assert result["values"][2] == pytest.approx((9900 - 10000) / 10000)


def test_a_shorter_curve_null_fills_the_times_it_never_reported():
    """buy_hold_djia only reports t1 -- its series must be exactly 1 real value
    plus 2 nulls, aligned to the SAME times array every other series uses, not
    a 1-element array of its own."""
    result = _run_ts(
        _RAGGED_CURVE_FIXTURE_JS
        + """
const bh = board.series.find((s) => s.key === 'buy_hold_djia');
console.log(JSON.stringify(bh.values));
"""
    )
    assert result == [pytest.approx(0.0), None, None]


def test_a_skipped_entrys_colour_slot_is_not_reused_by_the_next_model():
    """Important 3. deepseek_v4_pro (curve-less) sits third in payload order,
    between gpt_5_5 and qwen3_7_plus. If a curve-less entry were skipped
    BEFORE colour assignment, qwen3_7_plus would be handed
    MODEL_COLOR_PALETTE[2] -- the slot deepseek_v4_pro's rank in the standings
    still claims -- rather than [3]. That is exactly the /app incident
    documented at home-page.js:1748: a skipped key desyncs every later
    model's colour from the page that does not skip it."""
    result = _run_ts(
        _RAGGED_CURVE_FIXTURE_JS
        + """
console.log(JSON.stringify({
  deepseek: board.standings.find((s) => s.key === 'deepseek_v4_pro').color,
  qwen: board.series.find((s) => s.key === 'qwen3_7_plus').color,
}));
"""
    )
    palette = [
        "#FBBF24", "#FB923C", "#F472B6", "#A78BFA", "#34D399",
        "#22D3EE", "#F87171", "#A3E635", "#E879F9", "#60A5FA",
    ]
    # Model order in payload is claude_haiku_4_5(0), gpt_5_5(1),
    # deepseek_v4_pro(2), qwen3_7_plus(3).
    assert result["deepseek"] == palette[2]
    assert result["qwen"] == palette[3]


def test_series_and_standings_agree_on_a_shared_entrys_colour():
    """An entry that appears in both collections -- every entry with a
    drawable curve does -- must be the same colour in each. Two independently
    computed styles for one entry_id is the shape of bug this whole module
    exists to rule out."""
    result = _run_ts(
        _RAGGED_CURVE_FIXTURE_JS
        + """
const seriesColor = Object.fromEntries(board.series.map((s) => [s.key, s.color]));
const standingsColor = Object.fromEntries(board.standings.map((s) => [s.key, s.color]));
console.log(JSON.stringify({seriesColor, standingsColor}));
"""
    )
    for key, color in result["seriesColor"].items():
        assert result["standingsColor"][key] == color, f"{key} disagrees between series and standings"
