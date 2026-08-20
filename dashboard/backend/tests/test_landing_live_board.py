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
import tempfile
from pathlib import Path

import pytest

from dashboard.backend.tests._frontend_source import call_args, strip_comments

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
    # THE QUERY IS PART OF THE URL. Pinning only the path prefix left the period
    # unguarded: `?period=daily` kept 98 tests green, and `daily` is the one
    # period reaching `maybe_schedule_daily_leaderboard_refresh()` -- from `/`,
    # the highest-traffic ANONYMOUS surface in the product. `?period=live`
    # likewise puts the Season-0 preview board in the hero with none of the
    # preview chrome that exists to say nothing has advanced.
    assert "/api/v1/leaderboard?period=contest" in _LIB_TS, (
        "the hero draws the Competition board; any other period either changes "
        "what the card claims or reaches a refresh path an anonymous GET must "
        "not touch"
    )


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


# ---------------------------------------------------------------------------
# The two things `buildBoardData` does to the standings that nothing else does:
# it formats the return, and it restores rank order after selectBoardEntries
# has destroyed it. Neither was pinned by anything.
# ---------------------------------------------------------------------------


def test_standings_returns_carry_two_decimals():
    """Source pin, and the tier that actually runs in CI (the behavioural cases
    need an `npm install` CI does not do).

    `2` is the whole claim of "the hero shows the same numbers /app shows":
    /app's rank rows and this card's own tooltip are both two decimals. Nothing
    pinned it. Changing it to `0` left 98 tests green and rendered `+7%` and --
    the one that matters -- `-0%` where /app renders `+7.49%` and `-0.43%`: a
    real loss displayed as no movement, on the acquisition page.
    `test_the_two_surfaces_agree_on_the_numbers_that_must_agree` is not this
    guard: it pins the one-decimal AXIS formatter, a different call."""
    # PINNED ON THE DECIMALS, NOT ON THE COERCION. This was an exact-string
    # match including `Number(...)`, so replacing that with `finiteNumber(...)`
    # -- a fix to how an ABSENT return is read, which has nothing to do with the
    # precision this case is about -- reddened it and read as a decimals
    # regression. The `2` is the whole claim; the argument expression is not.
    assert re.search(
        r"formatPercent\(\s*\w+\(entry\.cumulative_return\)\s*,\s*2\s*\)", _LIB_TS
    ), (
        "standings returns are two decimals, matching /app's rank rows and this "
        "card's tooltip"
    )


def test_a_small_loss_renders_as_a_loss_and_not_as_zero():
    """The behavioural half of the case above, and the reason `0` is not a
    cosmetic choice: at zero decimals a -0.43% return formats as `-0%`."""
    result = _run_ts(
        """
console.log(JSON.stringify({
  loss: module.exports.formatPercent(-0.0043, 2),
  gain: module.exports.formatPercent(0.0749, 2),
}));
"""
    )
    assert result == {"loss": "-0.43%", "gain": "+7.49%"}


# A fixture whose payload (ranked) order is deliberately DESTROYED by
# selectBoardEntries: it regroups into models-then-baselines, so the two
# baselines land at the end regardless of where they ranked. buy_hold_djia
# (+3%) outranks two of the three models here, so a board that skipped the
# re-sort would publish gpt_5_5 at -2% ABOVE buy_hold_djia at +3%.
_INTERLEAVED_RANK_FIXTURE_JS = """
const entries = [
  {entry_id: 'claude_haiku_4_5', team_name: 'Claude Haiku 4.5', team_badge: 'Model', model: 'Claude Haiku 4.5', is_model: true, cumulative_return: 0.05, portfolio_value: 10500, initial_equity: 10000, equity_curve: []},
  {entry_id: 'buy_hold_djia', team_name: 'Agentic Trading Lab', team_badge: 'Baseline Strategy', model: 'Buy & Hold', is_model: false, cumulative_return: 0.03, portfolio_value: 10300, initial_equity: 10000, equity_curve: []},
  {entry_id: 'qwen3_7_plus', team_name: 'Qwen3.7 Plus', team_badge: 'Model', model: 'Qwen3.7 Plus', is_model: true, cumulative_return: 0.02, portfolio_value: 10200, initial_equity: 10000, equity_curve: []},
  {entry_id: 'djia_index', team_name: 'Agentic Trading Lab', team_badge: 'Market Index', model: 'DJIA', is_model: false, cumulative_return: 0.01, portfolio_value: 10100, initial_equity: 10000, equity_curve: []},
  {entry_id: 'gpt_5_5', team_name: 'GPT-5.5', team_badge: 'Model', model: 'GPT-5.5', is_model: true, cumulative_return: -0.02, portfolio_value: 9800, initial_equity: 10000, equity_curve: []},
];
const board = module.exports.buildBoardData({entries, window: {label: 'test window'}});
"""


def test_standings_are_re_ranked_by_return_after_the_model_baseline_regroup():
    """`selectBoardEntries` returns models.concat(baselines) -- it DELIBERATELY
    destroys rank order, because the model colour palette is assigned by
    position. `standings.sort` is the only thing that puts rank back, and
    Race.tsx renders `#{index + 1}` straight off the array index, so without it
    the rank column is a lie about a board that mostly lost to buy-and-hold.

    Nothing pinned the sort. Deleting it left 80 tests green while publishing
    GPT-5.5 at -2% as rank 3, ahead of Buy & Hold at +3% at rank 4.

    EQUALITY, not containment: a containment check passes on any order, which is
    the entire failure being guarded against here."""
    result = _run_ts(
        _INTERLEAVED_RANK_FIXTURE_JS
        + """
console.log(JSON.stringify({
  standingsKeys: board.standings.map((s) => s.key),
  rets: board.standings.map((s) => s.ret),
}));
"""
    )
    assert result["standingsKeys"] == [
        "claude_haiku_4_5",   # +5%
        "buy_hold_djia",      # +3% -- a BASELINE, second, which is the point
        "qwen3_7_plus",       # +2%
        "djia_index",         # +1%
        "gpt_5_5",            # -2%
    ]
    assert result["rets"] == ["+5.00%", "+3.00%", "+2.00%", "+1.00%", "-2.00%"]


# ---------------------------------------------------------------------------
# Hero x-axis date formatting -- the dashboard prints "Apr 15" for the same
# `timeKey` output this module bins onto the shared axis; the hero must not
# print the raw ISO key.
# ---------------------------------------------------------------------------


def test_format_axis_date_formats_an_hourly_key_as_a_short_date():
    result = _run_ts(
        "console.log(JSON.stringify(module.exports.formatAxisDate('2026-04-15T14:00')));"
    )
    assert result == "Apr 15"


def test_format_axis_date_formats_a_date_only_key_as_a_short_date():
    """timeKey falls back to a 10-char date-only key (`s.length >= 10`) when
    the timestamp has no `T` at index 10 -- formatAxisDate must take that
    branch too, not only the hourly one."""
    result = _run_ts(
        "console.log(JSON.stringify(module.exports.formatAxisDate('2026-04-15')));"
    )
    assert result == "Apr 15"


def test_format_axis_date_on_the_empty_key_returns_the_empty_string():
    """timeKey returns '' for a missing/unparseable timestamp -- Recharts must
    get back '' for that tick, not 'Invalid Date' or the literal string
    'undefined'."""
    result = _run_ts(
        "console.log(JSON.stringify(module.exports.formatAxisDate('')));"
    )
    assert result == ""


def test_format_axis_date_passes_through_an_unparseable_key_unchanged():
    """Mirrors formatShortDate's `if (Number.isNaN(d.getTime())) return
    isoDay;` -- an input `new Date(...)` cannot parse comes back UNCHANGED,
    not as the string "Invalid Date". A mutant that drops this branch (or
    returns `String(d)` instead of `isoDay`) fails this assertion."""
    result = _run_ts(
        "console.log(JSON.stringify(module.exports.formatAxisDate('not-a-date')));"
    )
    assert result == "not-a-date"


_BOARD_PREVIEW_TSX = (_ROOT / "landing" / "src" / "components" / "home" / "BoardPreview.tsx").read_text(
    encoding="utf-8"
)


def test_the_hero_x_axis_is_wired_to_the_date_formatter():
    """Source guard: the <XAxis> must actually use formatAxisDate as its
    tickFormatter, not just have the function exist unused in the module.
    Deleting `tickFormatter={formatAxisDate}` from BoardPreview.tsx while
    leaving the export in leaderboard.ts turns this assertion red -- verified
    by hand (see the task report) -- while every other test in this file
    still passes, since none of them render BoardPreview.tsx."""
    xaxis_match = re.search(r"<XAxis\b.*?/>", _BOARD_PREVIEW_TSX, re.S)
    assert xaxis_match, "no <XAxis> found in BoardPreview.tsx"
    assert "tickFormatter={formatAxisDate}" in xaxis_match.group(0), (
        "the XAxis must format its ticks with formatAxisDate, or the hero "
        "prints raw ISO keys like '2026-04-15T14:00' instead of 'Apr 15'"
    )
    assert "formatAxisDate" in re.search(
        r"import\s*\{[^}]*\}\s*from\s*\"@/lib/leaderboard\";", _BOARD_PREVIEW_TSX
    ).group(0), "formatAxisDate must be imported from the shared lib, not redefined inline"


# ---------------------------------------------------------------------------
# The tooltip HEADER is a second formatter, and the axis fix did not reach it.
# ---------------------------------------------------------------------------


def test_format_tooltip_date_prints_the_hour_the_dashboard_prints():
    """`/app` renders `Apr 15, 2:00 PM` for this key
    (`formatChartTooltipLabel`, js/leaderboard.js:1103). The axis deliberately
    drops the hour -- an hourly series would otherwise repeat `Apr 15` seven
    times -- so the tooltip needs its OWN formatter rather than reusing
    `formatAxisDate`, which is exactly why the raw key survived the axis fix."""
    result = _run_ts(
        "console.log(JSON.stringify(module.exports.formatTooltipDate('2026-04-15T14:00')));"
    )
    assert result == "Apr 15, 2:00 PM"


def test_format_tooltip_date_keeps_the_date_only_branch_dateless():
    """`formatChartTooltipLabel` branches on the `T`: a date-only key has no
    hour to print, and appending `12:00 AM` to one would invent a time the
    payload never carried."""
    result = _run_ts(
        "console.log(JSON.stringify(module.exports.formatTooltipDate('2026-04-15')));"
    )
    assert result == "Apr 15"


def test_format_tooltip_date_on_the_empty_key_returns_the_empty_string():
    result = _run_ts(
        "console.log(JSON.stringify(module.exports.formatTooltipDate('')));"
    )
    assert result == ""


def test_format_tooltip_date_passes_through_an_unparseable_key_unchanged():
    """Same `if (Number.isNaN(d.getTime())) return raw;` branch the axis
    formatter mirrors -- the input comes back UNCHANGED, not as the string
    "Invalid Date"."""
    result = _run_ts(
        "console.log(JSON.stringify(module.exports.formatTooltipDate('not-a-date')));"
    )
    assert result == "not-a-date"


def test_the_hero_tooltip_header_is_wired_to_the_tooltip_formatter():
    """Source guard, scoped to `<Tooltip>` the way the case above is scoped to
    `<XAxis>` -- and that scoping is why the axis guard could not see this.

    In recharts 2.15.4 the tooltip header is `tooltipTicks[activeIndex].value`
    (generateCategoricalChart.js:232) -- the raw category value -- rendered
    verbatim by DefaultTooltipContent unless a `labelFormatter` is supplied.
    `XAxis.tickFormatter` never reaches it. The category here is the `t`
    column, i.e. `timeKey()` output, so with no `labelFormatter` the hero
    printed `2026-04-15T14:00` directly above an axis correctly reading
    `Apr 15`.

    Binding-scoped rather than a bare `"formatTooltipDate" in _BOARD`, which
    the import line alone satisfies (`noUnusedLocals` is off, so an unused
    import typechecks clean)."""
    tooltip = re.search(r"<Tooltip\b.*?/>", _BOARD_PREVIEW_TSX, re.S)
    assert tooltip, "no <Tooltip> found in BoardPreview.tsx"
    assert "labelFormatter={formatTooltipDate}" in tooltip.group(0), (
        "the tooltip header must format its label, or the hero prints the raw "
        "ISO key '2026-04-15T14:00' where /app prints 'Apr 15, 2:00 PM'"
    )


# ---------------------------------------------------------------------------
# Task 7 -- one fetch, shared by the hero and the Race standings.
# ---------------------------------------------------------------------------

_HOOK_TS = None


def _hook() -> str:
    global _HOOK_TS
    if _HOOK_TS is None:
        _HOOK_TS = (_LIB / "useLeaderboard.tsx").read_text(encoding="utf-8")
    return _HOOK_TS


def test_the_board_is_fetched_once_for_the_whole_page():
    """The hero and the Race standings are four screens apart and render the same
    board. Two fetches double the load on a backend that cold-starts in 30-60s
    and, worse, can disagree -- real numbers in the hero above different ones in
    the table is worse than either alone."""
    assert "createContext" in _hook()
    assert "LeaderboardProvider" in _hook()
    page = (_ROOT / "landing" / "src" / "pages" / "landing-page.tsx").read_text(
        encoding="utf-8"
    )
    # Real containment, not text order. `provider_at < hero_at < race_at` alone
    # is satisfied by a mutant where </LeaderboardProvider> closes right after
    # <Hero /> -- <Race /> then sits textually after the opening tag but
    # OUTSIDE the provider's actual JSX children, which is well-formed JSX
    # and typechecks clean (verified). Exactly one provider -- so a stray
    # second one can't satisfy this by itself -- and both consumers must fall
    # strictly between its one open tag and its one close tag.
    assert page.count("<LeaderboardProvider>") == 1, "exactly one provider expected"
    assert page.count("</LeaderboardProvider>") == 1, "exactly one closing tag expected"
    provider_at = page.index("<LeaderboardProvider>")
    provider_close_at = page.index("</LeaderboardProvider>")
    assert provider_at < provider_close_at, "the provider must actually close"
    hero_at = page.index("<Hero />")
    race_at = page.index("<Race />")
    assert provider_at < hero_at < provider_close_at, "<Hero /> must sit inside the provider"
    assert provider_at < race_at < provider_close_at, "<Race /> must sit inside the provider"
    assert hero_at < race_at, "Hero renders above Race on the page"


def test_the_three_states_are_distinguishable_in_the_type():
    """Loading, ready and failed are three states, not two plus a fallback. A
    silent fallback to sample curves would make "the backend is down" and "the
    backend is fine" render near-identically -- the exact failure shape
    CLAUDE.md's fail-closed-is-not-fail-visible section is about."""
    src = _hook()
    for status in ('"loading"', '"ready"', '"error"'):
        assert status in src, f"{status} is not one of the states"
    assert "SAMPLE_" not in src, "no fallback to invented curves, ever"


def test_a_failed_fetch_carries_a_message_rather_than_a_bare_flag():
    """The failed card names the failure. "Something went wrong" with no cause is
    the dead end this landing's auth modal already had to be corrected for."""
    assert re.search(r"message:\s*", _hook())


def test_the_fetch_is_cancelled_on_unmount():
    assert "AbortController" in _hook()
    assert ".abort()" in _hook()


def test_the_unmount_cleanup_itself_calls_abort_not_just_the_timeout_handler():
    """`.abort()` also appears inside the 45s timeout handler
    (`setTimeout(() => controller.abort(), 45_000)`), so a mutant that deletes
    ONLY the effect's own cleanup call -- leaving a fetch for an unmounted
    provider to keep running until the timeout, or forever if it resolves
    first -- leaves both substring checks above green (see the mutation check
    in the task report: this exact mutant passed `test_the_fetch_is_cancelled_
    on_unmount` unchanged). This targets `.abort()` specifically inside the
    effect's `return () => {...}` cleanup body."""
    match = re.search(r"return \(\) => \{(.*?)\};", _hook(), re.S)
    assert match, "no cleanup function found in the effect"
    assert ".abort()" in match.group(1), "the cleanup function itself must call .abort()"


# ---------------------------------------------------------------------------
# Behavioural coverage for useLeaderboard.tsx.
#
# The provider itself (createContext/useState/useEffect, the actual mount ->
# fetch -> unmount -> abort lifecycle, and React 18 StrictMode's dev-only
# double-invoke) needs a real React render to mean anything -- a mounted
# fiber, a committed effect, an unmount. That needs a DOM (jsdom) or a fake
# host config (react-test-renderer / @testing-library/react), and NONE of
# those are present in dashboard/landing/node_modules (checked: no jsdom, no
# react-test-renderer, no @testing-library/*). Installing a new devDependency
# was out of scope for this task, so the provider's own lifecycle is covered
# only by the source-shape tests above plus the structural nesting check in
# test_the_board_is_fetched_once_for_the_whole_page (one <LeaderboardProvider>
# wrapping both <Hero /> and <Race /> is what makes "one fetch" true: one
# component instance, one effect, one fetchLeaderboard call per real mount).
# That is an honest gap, not a papered-over one.
#
# What CAN run under plain node is the one piece of genuinely tricky,
# DOM-independent logic in the file: classifyFetchFailure, which decides
# whether a caught rejection becomes the "gave up waiting" message or the
# request's own error message. It is exported from useLeaderboard.tsx for
# exactly this reason. Bundled and run the same way _run_ts runs
# leaderboard.ts, with --jsx=automatic added since this file is .tsx.
# ---------------------------------------------------------------------------

_ESBUILD_HOOK = _ESBUILD


def _run_tsx(script: str):
    """Transpile useLeaderboard.tsx to CJS and run `script` against it under
    node. Same mechanism as _run_ts above (esbuild -> node), generalised with
    --jsx=automatic because this module (unlike leaderboard.ts) contains JSX.

    Bundling React itself in makes the output ~130KB, past Linux's per-argument
    MAX_ARG_STRLEN (128KiB) -- `node -e <bundle+script>` (the _run_ts approach)
    hits `OSError: [Errno 7] Argument list too long` on this file specifically.
    Writing the combined source to a temp file and running `node <file>`
    instead sidesteps the argv limit without changing the technique."""
    node = shutil.which("node")
    if not node or not _ESBUILD_HOOK.is_file():
        pytest.skip("node and dashboard/landing/node_modules are required")
    bundled = subprocess.run(
        [str(_ESBUILD_HOOK), str(_LIB / "useLeaderboard.tsx"), "--bundle", "--format=cjs",
         "--platform=node", "--jsx=automatic", "--log-level=error"],
        capture_output=True, text=True, timeout=60,
    )
    assert bundled.returncode == 0, bundled.stderr
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".js", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(bundled.stdout + "\n" + script)
        tmp_path = tmp.name
    try:
        proc = subprocess.run(
            [node, tmp_path], capture_output=True, text=True, timeout=30,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_an_aborted_request_that_rejects_with_an_abort_error_reports_a_timeout():
    """The routine case: the 45s ceiling fires, `controller.abort()` rejects the
    underlying fetch with a real AbortError, and the card should say the board
    timed out -- not surface a raw "AbortError" message a visitor can't act
    on."""
    result = _run_tsx(
        """
const err = Object.assign(new Error('aborted'), {name: 'AbortError'});
const state = module.exports.classifyFetchFailure(err, true);
console.log(JSON.stringify(state));
"""
    )
    assert result == {"status": "error", "message": "Timed out waiting for the board."}


def test_an_aborted_request_that_rejects_with_a_real_error_keeps_that_message():
    """If the timeout fires but the rejection is some OTHER real error (not an
    AbortError -- e.g. a network failure that happened to lose the race with
    the timeout), the card must show that error's own message, not blame it on
    the timeout it didn't actually hit. This is the branch a naive
    `if (aborted) show timeout` would get wrong."""
    result = _run_tsx(
        """
const state = module.exports.classifyFetchFailure(new Error('HTTP 503'), true);
console.log(JSON.stringify(state));
"""
    )
    assert result == {"status": "error", "message": "HTTP 503"}


def test_a_non_aborted_request_reports_its_own_error_message():
    """The ordinary failure path: no timeout involved, the request just failed.
    The message must be the real error, not the generic timeout copy."""
    result = _run_tsx(
        """
const state = module.exports.classifyFetchFailure(new Error('HTTP 500'), false);
console.log(JSON.stringify(state));
"""
    )
    assert result == {"status": "error", "message": "HTTP 500"}


def test_a_non_error_rejection_falls_back_to_unknown_error_when_not_aborted():
    """`fetchLeaderboard` can only ever reject with an Error today, but the
    catch handler is typed `unknown` and must not throw trying to read
    `.message` off something that isn't one."""
    result = _run_tsx(
        """
const state = module.exports.classifyFetchFailure('not an Error instance', false);
console.log(JSON.stringify(state));
"""
    )
    assert result == {"status": "error", "message": "Unknown error"}


_RAIL = None


def _rail() -> str:
    global _RAIL
    if _RAIL is None:
        _RAIL = (
            _ROOT / "landing" / "src" / "components" / "home" / "EndpointRail.tsx"
        ).read_text(encoding="utf-8")
    return _RAIL


def test_the_rail_degrades_to_nothing_when_recharts_internals_change():
    """`Customized` is cloned with the chart's props and state, which is internal
    shape rather than contract. When it is not what the rail expects, the rail
    renders nothing and the chip strip below keeps keying every curve -- a real
    fallback, not a silent one."""
    src = _rail()
    assert "Array.isArray(formattedGraphicalItems)" in src
    assert "return null" in src


def test_the_rail_draws_the_frame_and_not_a_second_geometry():
    """Every number comes from boardFrame.ts, which is pinned against
    js/leaderboard.js. A literal here would be a third copy nothing guards.

    THE CONSTANT ASSERTIONS READ THE BODY, NOT THE FILE. Against the whole
    source every one of them is satisfied by the IMPORT BLOCK ALONE, with
    nothing below it referring to any of them -- and `tsconfig.json` sets
    `noUnusedLocals: false`, so an import nothing uses type-checks clean.
    Verified by mutation: leaving the import byte-identical and replacing each
    USE below with a divergent literal (radius 6, stub 14, arrow head 16, gap
    40) put exactly the unguarded third copy this docstring forbids into the
    rail, with `npm run typecheck` clean and this file at 43 passed.

    The import line and `stackLabels(` stay scoped to the full source on
    purpose: the import is a claim about where the numbers come from, and
    `stackLabels` is only ever called, never imported-and-shadowed."""
    src = _rail()
    assert "from \"@/lib/boardFrame\"" in src or "from '@/lib/boardFrame'" in src
    assert "stackLabels(" in src
    body = src.split("type RailProps", 1)
    assert len(body) == 2, "could not split the import block from the rail body"
    body = body[1]
    assert "BOARD_DOT_RADIUS" in body and "BOARD_STUB_LENGTH" in body
    assert "BOARD_ARROW_HEAD_LENGTH" in body
    assert "BOARD_LABEL_GAP_MAX" in body, "even the fallback gap is the frame's"


def test_the_rail_never_sorts_by_declaration_order():
    """`formattedGraphicalItems` arrives in <Line> declaration order, not visual
    order. `stackLabels` sorts by y itself; anything that assumed the incoming
    order was meaningful would stagger the wrong labels."""
    src = _rail()
    assert "formattedGraphicalItems" in src
    assert ".sort(" not in src, "sorting is stackLabels' job and it does it by y"


# ---------------------------------------------------------------------------
# THE THREE DRIFTS THE CONSTANT MIRROR COULD NOT SEE.
#
# boardFrame.ts's own docstring says it: the guard in test_landing_board_frame.py
# diffs `BOARD_*` SOURCE TEXT, not function bodies, so it is green while the two
# copies compute different things from the same numbers. Every case below pins a
# USE against the shipped `js/leaderboard.js` rather than a declaration, and each
# names the mutant it exists to fail on.
# ---------------------------------------------------------------------------


def test_the_rail_stacks_into_the_canvas_band_and_not_the_plot_band():
    """`bottom` is the CANVAS inset by half a pill, never the plot's bottom edge.

    The shipped hook calls `boardStackLabels(labels, frame.gap, half,
    chart.height - half)` and its comment gives the reason: a gutter label sits
    beside the plot, so hanging below `chartArea.bottom` into the x-axis strip is
    legitimate, and clamping to it clipped the stack -- measured at 5px past the
    canvas on the tab and 10.4px on screen 0, sliced through the middle.

    The mirror shipped `top: offset.top, bottom: offset.top + offset.height`,
    which is that clamp restored AND the half-pill inset dropped: the head's 15px
    pill then centres on the band edge and its top 3.5px falls outside the SVG
    viewBox. It also unhooks `frameLayout`'s stack-fits-the-canvas guard, which
    is checked against the full `height` and so no longer bounds the band the
    rail actually walks.

    Both sides are asserted, because the failure is DISAGREEMENT: a change to
    either copy alone should redden this."""
    args = re.search(r"stackLabels\(rows,\s*\{(.*?)\}\)", _rail(), re.S)
    assert args, "the rail no longer calls stackLabels with an options object"
    band = args.group(1)
    assert "top: HALF_PILL" in band, "the band's top must be inset by half a pill"
    assert "bottom: height - HALF_PILL" in band, (
        "the band's bottom is the CANVAS inset by half a pill, not the plot's bottom"
    )
    assert "offset." not in band, (
        "offset.* is the PLOT area; using it here is the clipping bug the shipped "
        "boardStackLabels docstring records fixing"
    )
    assert re.search(r"HALF_PILL\s*=\s*BOARD_PILL_HEIGHT\s*/\s*2", _rail()), (
        "HALF_PILL must be derived from the frame's pill height, not typed again"
    )
    assert re.search(
        r"boardStackLabels\(labels,\s*frame\.gap,\s*half,\s*chart\.height - half\)",
        _LEADERBOARD_JS,
    ), "the shipped hook's band changed; the mirror above now disagrees with it"


def test_the_rail_spends_the_tick_clearance_the_gutter_reserves():
    """`labelBlockWidth` adds BOARD_TICK_CLEARANCE to every gutter it reserves;
    the rail has to spend it or the reserve is idle and the overlap it was bought
    to prevent is still there.

    The shipped hook computes `labelX + (lab.y + half > chartArea.bottom ?
    BOARD_TICK_CLEARANCE : 0)` -- indenting only the labels that actually descend
    into the axis strip, which is where recharts' last x tick overhangs into the
    gutter (`textAnchor="middle"` on the plot's right edge, ~22px at fontSize
    14). The mirror imported neither the constant nor the shift, so the lowest
    labels painted over the axis date.

    Pinned on the DRAWN coordinates, not just the constant's presence: importing
    it and computing `lx` while still drawing at `labelX` is the same defect with
    the substring check green."""
    body = _rail().split("type RailProps", 1)[1]
    assert "BOARD_TICK_CLEARANCE" in body, "the reserved clearance is never spent"
    assert re.search(r"\?\s*BOARD_TICK_CLEARANCE\s*:\s*0", body), (
        "the clearance is CONDITIONAL -- only labels below the axis line pay it"
    )
    for drawn in ("x={labelX", "cx={labelX", "x1={labelX", "x2={labelX"):
        assert drawn not in body, (
            f"{drawn!r} draws at the unshifted gutter x; every drawn coordinate "
            f"must use the tick-clearance-adjusted `lx`"
        )
    assert re.search(
        r"lab\.y \+ half > chartArea\.bottom \? BOARD_TICK_CLEARANCE : 0",
        _LEADERBOARD_JS,
    ), "the shipped hook's clearance shift changed; the mirror now disagrees"


def test_the_rail_lays_out_the_label_block_with_the_frames_named_gaps():
    """`BOARD_DOT_GAP` (4) and `BOARD_NAME_GAP` (6) were re-typed as bare literals
    in the two places that lay out the block, while `labelBlockWidth` reserved the
    gutter with the named constants -- so raising either widened the reserve and
    not the drawn block, or lowered it and overran the reserve, with the constant
    mirror green throughout because it only diffs declarations.

    This is the "a literal here would be a third copy nothing guards" line in
    EndpointRail.tsx's own docstring, which the file then did not honour for these
    two."""
    body = _rail().split("type RailProps", 1)[1]
    pill = re.search(r"const pillX =(.*?);", body, re.S)
    assert pill, "the pill's x is no longer computed in one expression"
    expr = pill.group(1)
    assert "BOARD_DOT_GAP" in expr and "BOARD_NAME_GAP" in expr, (
        "the label block's two gaps come from the frame, not from literals"
    )
    assert not re.search(r"(?<![\w.])[46](?![\w.])", expr), (
        "a bare 4 or 6 in the block layout is the third copy of BOARD_DOT_GAP / "
        "BOARD_NAME_GAP that nothing guards"
    )
    assert body.count("BOARD_DOT_GAP") >= 2, (
        "both the name's x and the pill's x are offset by the dot gap"
    )


def test_the_rail_does_not_rebuild_its_geometry_on_every_mousemove():
    """`Customized` is cloned with the chart's mutable STATE, and `chartX`/
    `chartY`/`activeTooltipIndex` are written into it by a throttled 60fps
    mousemove handler -- so an unmemoised rail rebuilds every row, re-walks the
    stack and re-runs 2N canvas `measureText` calls per pointer frame, for values
    that cannot change on hover. The shipped Chart.js hook documents removing
    exactly this.

    The memos must sit ABOVE the internals-changed guard: hook order may not
    depend on a prop, and a `useMemo` after an early `return null` is a
    conditional hook call that React throws on."""
    src = _rail()
    assert "useMemo" in src, "the rail's geometry is recomputed on every render"
    memo_at = src.index("useMemo")
    guard = re.search(r"if \(!Array\.isArray\(formattedGraphicalItems\)", src)
    assert guard, "the internals-changed guard is gone"
    assert memo_at < guard.start(), (
        "a useMemo below the early return is a conditional hook call; the memos "
        "must run unconditionally and yield empty in the cases the guard rejects"
    )


# ---------------------------------------------------------------------------
# The data layer: one bad point, and one bad comparator.
# ---------------------------------------------------------------------------

_BAD_EQUITY_FIXTURE_JS = """
const entries = [
  {entry_id: 'gpt_5_5', team_name: 'GPT-5.5', team_badge: 'Model', model: 'GPT-5.5', is_model: true, cumulative_return: 0.01, portfolio_value: 10100, initial_equity: 10000,
   equity_curve: [{timestamp: '2026-04-15T14:00:00+00:00', equity: 10000},
                  {timestamp: '2026-04-15T15:00:00+00:00', equity: null},
                  {timestamp: '2026-04-15T16:00:00+00:00', equity: 10100}]},
  {entry_id: 'buy_hold_djia', team_name: 'Agentic Trading Lab', team_badge: 'Baseline Strategy', model: 'Buy & Hold', is_model: false, cumulative_return: 0.02, portfolio_value: 10200, initial_equity: 10000,
   equity_curve: [{timestamp: '2026-04-15T14:00:00+00:00', equity: 10000},
                  {timestamp: '2026-04-15T15:00:00+00:00', equity: 10150},
                  {timestamp: '2026-04-15T16:00:00+00:00', equity: 10200}]},
];
const board = module.exports.buildBoardData({entries, window: {label: 'test window'}});
"""


def test_a_non_numeric_equity_point_is_a_gap_and_not_a_hundred_percent_loss():
    """`Number(pt.equity) || 0` read null/undefined/NaN/a string as a $0 account.

    `base` is the $10,000 display capital, so that point became
    `(0 - 10000) / 10000 = -100%` -- and the damage is not one bad marker.
    `percentDomain` then spans about [-1.12, hi], and the real board's
    -0.43%..+7.49% spread collapses into a sliver at the top of the axis with a
    vertical spike to the floor. Every curve on the card goes flat because of one
    point on one of them.

    A missing point already has a representation here -- `null`, which Recharts
    skips and `connectNulls` bridges. Reachable end to end: `get_leaderboard`
    normalises a stored NULL to `0` server-side (`float(pt.get("equity") or 0)`),
    and this client-side coercion turned every other malformed shape into the
    same thing."""
    result = _run_ts(
        _BAD_EQUITY_FIXTURE_JS
        + """
const gpt = board.series.find((s) => s.key === 'gpt_5_5');
console.log(JSON.stringify({values: gpt.values, times: board.times}));
"""
    )
    assert result["values"][1] is None, (
        "an unparseable equity is a GAP; zeroing it renders the account as -100%"
    )
    assert result["values"][0] == pytest.approx(0.0)
    assert result["values"][2] == pytest.approx(0.01)
    assert min(v for v in result["values"] if v is not None) > -0.5, (
        "no curve may be dragged to the -100% floor by a single malformed point"
    )


_NEAR_TIE_FIXTURE_JS = """
const entries = [
  {entry_id: 'a_lower', team_name: 'A', team_badge: 'Model', model: 'A', is_model: true, cumulative_return: 0.070001, portfolio_value: 10700, initial_equity: 10000, equity_curve: []},
  {entry_id: 'b_higher', team_name: 'B', team_badge: 'Model', model: 'B', is_model: true, cumulative_return: 0.070049, portfolio_value: 10700, initial_equity: 10000, equity_curve: []},
];
const board = module.exports.buildBoardData({entries, window: {label: 'test window'}});
"""

_NON_FINITE_RETURN_FIXTURE_JS = """
const entries = [
  {entry_id: 'broken', team_name: 'Broken', team_badge: 'Model', model: 'Broken', is_model: true, cumulative_return: null, portfolio_value: 0, initial_equity: 10000, equity_curve: []},
  {entry_id: 'low', team_name: 'Low', team_badge: 'Model', model: 'Low', is_model: true, cumulative_return: -0.02, portfolio_value: 9800, initial_equity: 10000, equity_curve: []},
  {entry_id: 'high', team_name: 'High', team_badge: 'Model', model: 'High', is_model: true, cumulative_return: 0.05, portfolio_value: 10500, initial_equity: 10000, equity_curve: []},
  {entry_id: 'mid', team_name: 'Mid', team_badge: 'Model', model: 'Mid', is_model: true, cumulative_return: 0.01, portfolio_value: 10100, initial_equity: 10000, equity_curve: []},
];
const board = module.exports.buildBoardData({entries, window: {label: 'test window'}});
"""


def test_two_returns_inside_the_display_precision_still_rank_correctly():
    """The standings were ordered by `parseFloat(b.ret) - parseFloat(a.ret)` --
    re-reading the two-decimal DISPLAY string. Both entries here format to
    "+7.00%", so that comparator returned 0 and left them in payload order while
    Race printed them under distinct `#1`/`#2` ranks that did not reflect the
    returns. The existing rank guard uses 5%/3%/2%/1%/-2%, which is far outside
    the precision the formatter throws away, so it never saw this."""
    result = _run_ts(
        _NEAR_TIE_FIXTURE_JS
        + """
console.log(JSON.stringify({
  keys: board.standings.map((s) => s.key),
  rets: board.standings.map((s) => s.ret),
}));
"""
    )
    assert result["rets"] == ["+7.00%", "+7.00%"], (
        "fixture no longer exercises the sub-display-precision case"
    )
    assert result["keys"] == ["b_higher", "a_lower"], (
        "rank must come from the number, not from the string it was formatted to"
    )


def test_one_non_finite_return_does_not_scramble_every_other_rank():
    """`formatPercent` returns the em-dash for a non-finite return and
    `parseFloat('\u2014')` is NaN. A comparator that returns NaN leaves V8's sort
    order IMPLEMENTATION-DEFINED for the whole array -- so one malformed entry
    could reorder every rank and every chip on the page, not just misplace
    itself.

    Subtracting numerics is not enough either: sinking non-finite values to
    -Infinity reintroduces NaN the moment two of them meet (-Infinity minus
    -Infinity). The comparator has to be a total order."""
    result = _run_ts(
        _NON_FINITE_RETURN_FIXTURE_JS
        + """
console.log(JSON.stringify({
  keys: board.standings.map((s) => s.key),
  rets: board.standings.map((s) => s.ret),
}));
"""
    )
    assert result["keys"][:3] == ["high", "mid", "low"], (
        "the finite returns must stay correctly ranked regardless of a bad entry"
    )
    assert result["keys"][3] == "broken", "a non-finite return ranks last"
    assert result["rets"][3] == "\u2014"


def test_board_headline_counts_report_the_field_and_who_beat_it():
    """Race's opening sentence hardcoded "Seven leading AI models ... Only one
    finished ahead of both" beside a table that is now live off the same payload.
    These are the numbers that replace both literals."""
    result = _run_ts(
        _INTERLEAVED_RANK_FIXTURE_JS
        + """
console.log(JSON.stringify(module.exports.boardHeadlineCounts(board.standings)));
"""
    )
    # claude_haiku +5%, qwen +2%, gpt -2% vs buy_hold +3% and djia +1%:
    # only claude clears the BEST baseline, which is what "ahead of both" means.
    assert result == {"models": 3, "baselines": 2, "ahead": 1}


def test_board_headline_counts_claim_nothing_without_a_baseline_to_beat():
    """"Ahead of both" is a claim about a comparison. With no baseline resolved
    there is nothing to be ahead OF, and reporting `ahead` off a -Infinity
    sentinel would make every model a winner."""
    result = _run_ts(
        """
const entries = [
  {entry_id: 'solo', team_name: 'Solo', team_badge: 'Model', model: 'Solo', is_model: true, cumulative_return: 0.05, portfolio_value: 10500, initial_equity: 10000, equity_curve: []},
];
const board = module.exports.buildBoardData({entries, window: {label: 'w'}});
console.log(JSON.stringify(module.exports.boardHeadlineCounts(board.standings)));
"""
    )
    assert result == {"models": 1, "baselines": 0, "ahead": 0}


# ---------------------------------------------------------------------------
# What a visitor is shown when the fetch fails.
# ---------------------------------------------------------------------------


def test_a_two_hundred_that_is_not_json_is_reported_as_unreadable():
    """`await res.json()` ran on ANY 2xx. The Vercel `/api/:path*` rewrite
    (dashboard/frontend/vercel.json) proxies to Render, and either end can answer
    with an HTML error page -- so the public failure message on the hero card was
    `Unexpected token '<', "<!DOCTYPE "... is not valid JSON`, rendered in
    font-mono, and mid-sentence in Race ("The standings didn't load (...)").

    Checked on the content type rather than by catching the parse error, so the
    HTML body is never fed to the parser in the first place."""
    result = _run_ts(
        """
globalThis.fetch = async () => ({
  ok: true,
  status: 200,
  headers: {get: () => 'text/html; charset=utf-8'},
  json: async () => { throw new SyntaxError('Unexpected token \\'<\\', "<!DOCTYPE "... is not valid JSON'); },
});
module.exports.fetchLeaderboard(undefined).then(
  () => console.log(JSON.stringify({message: null})),
  (err) => console.log(JSON.stringify({message: err.message})),
);
"""
    )
    assert result["message"] is not None, "an HTML 2xx must not resolve to a board"
    assert "DOCTYPE" not in result["message"] and "JSON" not in result["message"], (
        "the parser's own text is not a sentence to show a visitor"
    )


def test_a_dead_backend_is_reported_without_the_browsers_own_wording():
    """A dead backend rejects `fetch` with `TypeError: Failed to fetch`, which the
    card rendered verbatim. AbortError must still pass through untouched -- the
    provider's 45s ceiling aborts this same signal, and `classifyFetchFailure`
    reads exactly that to say "timed out" instead."""
    result = _run_ts(
        """
const calls = [];
globalThis.fetch = async () => { throw new TypeError('Failed to fetch'); };
module.exports.fetchLeaderboard(undefined).catch((err) => calls.push(err.message));
globalThis.fetch = async () => { throw Object.assign(new Error('aborted'), {name: 'AbortError'}); };
module.exports.fetchLeaderboard(undefined).catch((err) => calls.push(err.name));
setTimeout(() => console.log(JSON.stringify(calls)), 0);
"""
    )
    assert "Failed to fetch" not in result[0], (
        "the browser's transport wording is not visitor-facing copy"
    )
    assert result[1] == "AbortError", (
        "AbortError must reach classifyFetchFailure unchanged or the timeout "
        "branch can never fire"
    )


def test_an_unmount_is_not_reported_to_the_visitor_as_a_timeout():
    """`controller.signal.aborted` has THREE causes, and `classifyFetchFailure`
    knows two. The effect's own cleanup aborts as well, so an unmount landed in
    the timeout branch. Today that setState hits an unmounted provider and is a
    no-op -- but add `<StrictMode>` to main.tsx and React 18 double-invokes the
    effect, aborting the first fetch on a component whose state survives, so the
    first thing a visitor sees is "Timed out waiting for the board." before the
    second fetch resolves over it.

    Intent has to be tracked, not inferred from a flag that cannot tell the three
    apart."""
    src = _hook()
    cleanup = re.search(r"return \(\) => \{(.*?)\};", src, re.S)
    assert cleanup, "no cleanup function found in the effect"
    assert re.search(r"\bcancelled\s*=\s*true", cleanup.group(1)), (
        "the cleanup must record that it cancelled, so the catch can tell an "
        "unmount from the 45s ceiling"
    )
    assert re.search(r"if \(!cancelled\) setState", src), (
        "both settle paths must check the flag before reporting to the visitor"
    )


# ---------------------------------------------------------------------------
# A 200 is not a board. `get_leaderboard` skips any strategy with no cached run
# (service.py:1434, `if not run: continue`) and still answers 200, so
# `entries: []` and "only the baselines resolved" are ordinary SUCCESSFUL
# responses -- not errors, and not distinguishable from a full board by
# `status`. Coverage is the question `status: "ready"` cannot answer.
# ---------------------------------------------------------------------------

# Two baselines and no models: the reachable half. All seven LLM entries carry
# `auto_compute: false` in dashboard/config/leaderboard.json while the baselines
# auto-recompute, so editing the contest window misses `_find_cached_run` on all
# twelve, rebuilds the two baselines and never rebuilds the seven models.
_BASELINES_ONLY_FIXTURE_JS = """
const entries = [
  {entry_id: 'djia_index', team_name: 'Agentic Trading Lab', team_badge: 'Market Index', model: 'DJIA', is_model: false, cumulative_return: 0.0224, portfolio_value: 10224, initial_equity: 10000,
   equity_curve: [{timestamp: '2026-04-15T14:00:00+00:00', equity: 10000}, {timestamp: '2026-04-15T15:00:00+00:00', equity: 10224}]},
  {entry_id: 'buy_hold_djia', team_name: 'Agentic Trading Lab', team_badge: 'Baseline Strategy', model: 'Buy & Hold', is_model: false, cumulative_return: 0.0487, portfolio_value: 10487, initial_equity: 10000,
   equity_curve: [{timestamp: '2026-04-15T14:00:00+00:00', equity: 10000}, {timestamp: '2026-04-15T15:00:00+00:00', equity: 10487}]},
];
const board = module.exports.buildBoardData({entries, window: {label: '2026-04-15 → 2026-05-15'}});
"""

_EMPTY_FIXTURE_JS = """
const board = module.exports.buildBoardData({entries: [], window: {label: '2026-04-15 → 2026-05-15'}});
"""


def test_an_empty_two_hundred_is_reported_as_empty_by_both_coverage_rules():
    """`entries: []` is a 200. Without this the hero draws its whole frame over
    it -- a percent axis labelled -5.0%..5.0% off `percentDomain`'s hardcoded
    fallback, a scale no run produced, under the axis arrow, the title, the
    window chip and the caption "Return over the competition window, hour by
    hour" -- and Race renders its Rank/AI model/Return header over zero rows.
    Both are silent, confident, and wrong."""
    result = _run_ts(
        _EMPTY_FIXTURE_JS
        + """
console.log(JSON.stringify({
  chart: module.exports.chartCoverage(board.series),
  standings: module.exports.standingsCoverage(board.standings),
}));
"""
    )
    assert result == {"chart": "empty", "standings": "empty"}


def test_a_baselines_only_two_hundred_is_reported_as_carrying_no_models():
    """The more reachable half, and the one that looks more plausible: a card
    captioned "Each line is one AI model's return" drawing two dashed baselines
    and no models, above Race copy still reading "Seven leading AI models traded
    the same days." Not an error -- a 200 whose model entries all missed cache."""
    result = _run_ts(
        _BASELINES_ONLY_FIXTURE_JS
        + """
console.log(JSON.stringify({
  chart: module.exports.chartCoverage(board.series),
  standings: module.exports.standingsCoverage(board.standings),
  seriesKeys: board.series.map((s) => s.key),
  standingsKeys: board.standings.map((s) => s.key),
}));
"""
    )
    assert result["chart"] == "baselines-only"
    assert result["standings"] == "baselines-only"
    # Non-vacuity: the payload really did carry two drawable reference curves,
    # so "baselines-only" is distinguishing them from models rather than from
    # nothing at all -- which is what separates this state from "empty".
    assert sorted(result["seriesKeys"]) == ["buy_hold_djia", "djia_index"]
    assert sorted(result["standingsKeys"]) == ["buy_hold_djia", "djia_index"]


def test_a_full_board_is_reported_as_full_so_the_notice_is_not_permanent():
    """The other direction, and the one a mutant `return "baselines-only"` would
    fail: today's real roster must report `full`, or every visitor sees a notice
    saying the board is incomplete while looking at a complete board."""
    result = _run_ts(
        _RAGGED_CURVE_FIXTURE_JS
        + """
console.log(JSON.stringify({
  chart: module.exports.chartCoverage(board.series),
  standings: module.exports.standingsCoverage(board.standings),
}));
"""
    )
    assert result == {"chart": "full", "standings": "full"}


def test_a_model_with_no_drawable_curve_still_counts_as_a_model_in_the_standings():
    """The two rules answer for different collections on purpose: the hero draws
    `series` and Race lists `standings`, and a curve-less model reaches the
    second but not the first. deepseek_v4_pro in the ragged fixture is exactly
    that entry, and it must not make Race announce a board with no models."""
    result = _run_ts(
        _RAGGED_CURVE_FIXTURE_JS
        + """
const deepseek = board.standings.find((s) => s.key === 'deepseek_v4_pro');
console.log(JSON.stringify({
  isModel: deepseek.isModel,
  inSeries: board.series.some((s) => s.key === 'deepseek_v4_pro'),
  baselineIsNotAModel: board.standings.find((s) => s.key === 'buy_hold_djia').isModel,
}));
"""
    )
    assert result == {"isModel": True, "inSeries": False, "baselineIsNotAModel": False}


def test_the_gutter_is_measured_over_the_curves_the_rail_actually_draws():
    """`frameLayout` sizes the endpoint gutter, so it must be handed the curves
    the rail will paint -- `series` -- and not every ranked entry.

    The same asymmetry as the test above, one layer out. `buildBoardData` pushes
    every selected entry to `standings` unconditionally and only reaches
    `series.push` past `if (!values.some(v => v != null)) return`, so `series` is
    a strict subset. Measured over `standings`, the gutter reserved width for a
    pill the rail never paints -- and worse, `labelBlockWidth` could carry the
    floor past BOARD_GUTTER_MAX_FRACTION and drop the WHOLE rail to arrow-only,
    losing the labels of curves that would have fitted on account of one curve
    that does not exist. deepseek_v4_pro in the ragged fixture is that entry.

    Comments are stripped first, and that is load-bearing here: the call site's
    own note names BOTH collections, so an un-stripped scan is satisfied by the
    prose explaining the bug instead of by the code avoiding it.

    Both halves come from `_frontend_source` rather than being written here.
    The paren walk is the same one `fn_body` has always used, and the stripping
    is the part that was quietly wrong when this case first shipped its own: a
    `//[^\n]*` sub deletes the rest of any line carrying a URL, which shortens
    the region the assertions below run over without failing anything.
    """
    call = call_args(strip_comments(_BOARD_PREVIEW_TSX), "frameLayout")

    assert "series.map" in call, (
        "frameLayout must be measured over `series` -- the set the rail draws. "
        f"Its call site reads: {call.strip()!r}"
    )
    assert "standings" not in call, (
        "frameLayout is being measured over `standings`, which carries entries "
        "with no drawable curve: the gutter then reserves width for a pill the "
        "rail never paints, and can degrade the whole rail to arrow-only"
    )


def test_the_gutter_basis_source_and_shipped_bundle_agree():
    """The case above reads `landing/src`; prod reads `frontend/assets/index-*.js`.

    So it stays green against a bundle built before the fix, and that bundle
    still measures the gutter over `standings` on the live site. Nothing else
    catches it: test_frontend_bundle_integrity's staleness checks key on the CTA
    surface only, and its docstring says in as many words that a source change
    outside that surface needs its own agreement pin. This is that pin.

    Anchoring a STRING on both sides is the usual technique and is unavailable
    here -- the fix swapped one collection for another and introduced no copy.
    What survives minification instead is property NAMES: esbuild mangles
    locals, never object keys, so `.series`, `.standings` and `labels:` are all
    still in the bundle verbatim. Resolving the two mangled locals through the
    destructuring that reads those properties, then asking which one reaches
    `labels:`, tests the shipped artifact rather than a marker planted for the
    test.

    Deliberately loud rather than lenient: if the aliases cannot be resolved
    this fails instead of skipping, because "the bundle no longer looks like
    that" and "the fix is not in the bundle" must not have the same outcome --
    that equivalence is the whole failure mode the guard exists for.
    """
    bundles = sorted((_ROOT / "frontend" / "assets").glob("index-*.js"))
    assert len(bundles) == 1, (
        f"expected exactly one shipped entry bundle, found {[b.name for b in bundles]}"
    )
    shipped = bundles[0].read_text(encoding="utf-8")

    def alias(prop: str) -> str:
        # `a=(i==null?void 0:i.series)??[]` -- or `a=i?.series??[]` on a target
        # that keeps optional chaining. Neither form contains a comma or a
        # semicolon between the local and the property, which is what bounds
        # the search to one declaration.
        match = re.search(rf"([A-Za-z_$][\w$]*)=[^,;]*?\.{prop}\)?\?\?\[\]", shipped)
        assert match, (
            f"cannot find where the shipped bundle reads `.{prop}` off the board "
            "payload, so this guard can no longer tell which collection the "
            "gutter is measured over. Re-derive it from the current build "
            "output -- do not delete it."
        )
        return match.group(1)

    series, standings = alias("series"), alias("standings")
    assert series != standings

    assert f"labels:{series}.map" in shipped, (
        f"the shipped bundle does not measure `labels` over `series` (local "
        f"{series!r}). Either the fix was never rebuilt into frontend/assets/ "
        "-- run the refresh in dashboard/landing/README.md -- or the call was "
        "re-shaped and this guard needs re-deriving."
    )
    assert f"labels:{standings}.map" not in shipped, (
        f"the shipped bundle measures `labels` over `standings` (local "
        f"{standings!r}): frontend/assets/ predates the fix, so prod still "
        "reserves gutter for pills the rail never paints."
    )


_RACE_TSX_SRC = (_ROOT / "landing" / "src" / "components" / "home" / "Race.tsx").read_text(
    encoding="utf-8"
)


def test_both_board_consumers_branch_on_coverage_and_not_only_on_status():
    """The fourth render path, in both consumers.

    `BoardState` is exactly `loading | ready | error` and before this both
    components branched on exactly those three -- there was no length or
    emptiness check anywhere in landing/src. This is the repo's own documented
    fail-closed-is-not-fail-visible shape: "the upstream returned nothing" and
    "everything is fine" rendered byte-identically, at HTTP 200, with a green
    suite.

    Pinned as the CALL (parens), not the imported name: `noUnusedLocals` is off,
    so an import with no call site typechecks clean and would satisfy a bare
    substring check while both components went back to drawing a confident
    frame over nothing."""
    assert "chartCoverage(" in _BOARD_PREVIEW_TSX, (
        "the hero must ask what the 200 actually carried before drawing a frame"
    )
    assert "standingsCoverage(" in _RACE_TSX_SRC, (
        "the standings table must ask the same question before rendering rows"
    )
    for name, src in (("BoardPreview.tsx", _BOARD_PREVIEW_TSX), ("Race.tsx", _RACE_TSX_SRC)):
        assert '=== "empty"' in src, f"{name} must render the empty case differently"
        assert '=== "baselines-only"' in src, (
            f"{name} must say so when a 200 carried no model entries"
        )


def test_neither_consumer_invents_a_fallback_dataset_for_the_empty_case():
    """The fix for an empty board is to SAY it is empty. Substituting curves --
    under any name -- is the bug the whole change exists to remove, and a
    name-scoped ban on the three retired SAMPLE_* symbols cannot express that,
    so this one bans the shapes a new fallback would take."""
    for name, src in (("BoardPreview.tsx", _BOARD_PREVIEW_TSX), ("Race.tsx", _RACE_TSX_SRC)):
        assert "SAMPLE_" not in src, f"{name} carries a sample dataset again"
        assert not re.search(r"(FALLBACK|DEMO|PLACEHOLDER)_(CURVES|STANDINGS|ENTRIES)", src), (
            f"{name} carries a renamed fallback dataset"
        )
