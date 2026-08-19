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
    js/leaderboard.js. A literal here would be a third copy nothing guards."""
    src = _rail()
    assert "from \"@/lib/boardFrame\"" in src or "from '@/lib/boardFrame'" in src
    assert "stackLabels(" in src
    assert "BOARD_DOT_RADIUS" in src and "BOARD_STUB_LENGTH" in src
    assert "BOARD_ARROW_HEAD_LENGTH" in src
    assert "BOARD_LABEL_GAP_MAX" in src, "even the fallback gap is the frame's"


def test_the_rail_never_sorts_by_declaration_order():
    """`formattedGraphicalItems` arrives in <Line> declaration order, not visual
    order. `stackLabels` sorts by y itself; anything that assumed the incoming
    order was meaningful would stagger the wrong labels."""
    src = _rail()
    assert "formattedGraphicalItems" in src
    assert ".sort(" not in src, "sorting is stackLabels' job and it does it by y"
