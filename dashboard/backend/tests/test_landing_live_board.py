"""The landing hero draws the same live board the signed-in Home screen draws.

Source-shape guards. Nothing in CI builds or type-checks the landing, so these
read `landing/src` directly -- which is also the only layer that can compare the
landing's selection rule against screen 0's, since the two live in different
bundles and one of them ships minified.
"""

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_LIB = _ROOT / "landing" / "src" / "lib"
_HOME_JS = (_ROOT / "frontend" / "home-page.js").read_text(encoding="utf-8")
_LEADERBOARD_JS = (_ROOT / "frontend" / "js" / "leaderboard.js").read_text(encoding="utf-8")
_LIB_TS = (_LIB / "leaderboard.ts").read_text(encoding="utf-8")


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
