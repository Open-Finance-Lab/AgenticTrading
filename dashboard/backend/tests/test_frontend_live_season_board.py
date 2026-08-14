"""Source guards for the Live Season board (dashboard/frontend/js/leaderboard.js).

/app has no build step and no JS test toolchain, so these assert against the
shipped source as text -- the convention set by test_ai_hedge_fund_frontend.py.
This file replaces test_frontend_daily_leaderboard.py: the Daily Leaderboard was
retired in favour of a two-week season board that carries a portfolio across
trading days, and the poll/visibility machinery moved with it under new names.

Two things here are load-bearing for reasons the code alone does not show.

**The poll guard.** ``scheduleLiveBoardPoll`` is only ever re-entered from
``loadLeaderboardData``, which fires on a Competition subtab switch but *not*
when the user navigates away from Competition entirely. Without a visibility
re-check inside the timeout callback, a refresh that is in progress keeps the
30s poll re-fetching and re-rendering a hidden Chart.js canvas for the whole
(possibly multi-hour) model deploy.

**The preview banner.** The season engine is not deployed, and the server coerces
an unknown ``period`` back to 'contest' rather than 4xx-ing it, so asking for a
season returns HTTP 200 carrying the Competition board. Every other element on
the tab -- chart, table, curve picker, rankings -- renders identically either
way, because those shapes are shared between the two boards. The banner is the
*only* thing on screen that distinguishes "no season has ever run" from "these
are the live season standings", and it can only do that by comparing what was
requested against what came back. See the fail-closed-is-not-fail-visible
section of CLAUDE.md, and the FinSearch news adapter it was written about.
"""

import re
from pathlib import Path

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_LEADERBOARD_JS = (_FRONTEND / "js" / "leaderboard.js").read_text(encoding="utf-8")
_APP_HTML = (_FRONTEND / "app.html").read_text(encoding="utf-8")
_APP_JS = (_FRONTEND / "app.js").read_text(encoding="utf-8")


def _strip_js_comments(source: str) -> str:
    """Drop // and /* */ comments so a guard cannot be satisfied by prose.

    Every assertion below is about code that must (or must not) exist. A comment
    mentioning ``isLiveBoardVisible`` would otherwise pass the check while the
    call itself was deleted.
    """
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"^\s*//.*$", "", source, flags=re.MULTILINE)


_SOURCE = _strip_js_comments(_LEADERBOARD_JS)

# app.html with both comment syntaxes removed: HTML comments, and the JS
# comments inside its inline <script> blocks. Copy assertions run against this
# so that documenting *why* a retired label was retired cannot fail the guard
# that says the label is gone.
_APP_HTML_VISIBLE = _strip_js_comments(re.sub(r"<!--.*?-->", "", _APP_HTML, flags=re.DOTALL))


def _fn_body(name: str, source: str | None = None) -> str:
    """The named function's source, brace-matched to its closing brace."""
    text = _SOURCE if source is None else source
    start = text.index(f"function {name}(")
    index = text.index("{", text.index(")", start))
    depth = 0
    while True:
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
        index += 1


# ── Poll / visibility (carried over from the Daily board) ────────────────────


def test_live_poll_rechecks_visibility_when_it_fires():
    body = _fn_body("scheduleLiveBoardPoll")
    assert "setTimeout" in body
    callback = body[body.index("setTimeout") :]
    assert "isLiveBoardVisible()" in callback, (
        "the 30s season poll must re-check visibility when it fires; navigating "
        "away from Competition never calls loadLeaderboardData again, so "
        "without this it polls and re-renders a hidden chart forever"
    )
    # The guard has to bail, not merely observe.
    assert re.search(r"if\s*\(\s*!\s*isLiveBoardVisible\(\)\s*\)\s*return", callback)


def test_live_poll_only_runs_while_a_worker_is_running():
    body = _fn_body("scheduleLiveBoardPoll")
    assert "refresh_in_progress" in body, (
        "pending curves with no worker wait for the nightly advance; polling "
        "those just hammers the API"
    )


def test_visibility_helper_reads_the_dom_not_the_boot_attributes():
    body = _fn_body("isLiveBoardVisible")
    # html[data-nav-*] is written by navigateToPage but NOT by
    # showCompetitionPanel, so it goes stale on a plain subtab switch.
    assert "data-nav-competition-tab" not in body
    assert "leaderboardView" in body
    assert "competitionTab" in body
    assert "'season'" in body, "the poll must key on the season tab, not the retired daily one"


def test_live_board_subtitle_matches_the_cash_session_window():
    """Window math is the 16:00 ET close, not a calendar weekday -- say so."""
    body = _fn_body("formatLiveBoardSubtitle")
    assert "cash session" in body
    assert "weekday" not in body


# ── The preview banner: the one control that can report an absent engine ─────


def test_preview_is_decided_by_comparing_request_against_response():
    """A season the server cannot serve comes back as a valid contest payload.

    Checking only ``payload.season`` would be a check against the *response*,
    which is exactly what a coerced period leaves looking normal. The comparison
    has to involve what was asked for.
    """
    body = _fn_body("isSeasonPreview")
    assert "isSeasonBoard()" in body, (
        "preview detection must consult the requested board, not just the payload"
    )
    assert re.search(r"period\s*!==\s*'season'", body), (
        "preview detection must compare the returned period against 'season'"
    )


def test_requested_period_is_recorded_from_the_request_not_the_payload():
    """Non-vacuity for the test above: ``isSeasonBoard`` has to read a real value.

    If ``requestedBoardPeriod`` were ever assigned from the response, the two
    sides of the comparison would always agree and the banner could never fire.
    """
    load = _fn_body("loadLeaderboardData")
    assert re.search(r"requestedBoardPeriod\s*=\s*boardPeriod", load), (
        "loadLeaderboardData must record the period it asked for"
    )
    assignments = re.findall(r"requestedBoardPeriod\s*=\s*([^;\n]+)", _SOURCE)
    for rhs in assignments:
        assert "leaderboardPayload" not in rhs and "payload" not in rhs, (
            f"requestedBoardPeriod assigned from the response ({rhs.strip()!r}); "
            "that collapses the request-vs-response comparison the preview "
            "banner depends on"
        )


def test_preview_banner_is_reachable_from_the_header_render():
    """The banner must be wired into the path every board render takes."""
    assert "renderSeasonPreviewBanner(" in _fn_body("updateLeaderboardHeader")
    banner = _fn_body("renderSeasonPreviewBanner")
    assert "isSeasonPreview(" in banner
    assert "seasonPreviewBanner" in banner
    assert "hidden = false" in banner, "the banner must actually be shown, not only computed"


def test_hidden_season_containers_are_actually_hidden():
    """`display: flex` on a class outranks the UA stylesheet's [hidden] rule.

    Every season container ships with the `hidden` attribute and is un-hidden by
    JS only on the season tab. A `display` declaration without a matching
    `[hidden]` override renders it on the Competition board anyway -- and it is
    invisible to any test that checks `element.hidden`, because the attribute is
    set correctly; only computed style disagrees.
    """
    css = (_FRONTEND / "styles.css").read_text(encoding="utf-8")
    # The containers that both ship hidden and declare a `display`.
    for selector in (".season-strip", ".season-gaps"):
        block = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
        assert block, f"{selector} not found in styles.css"
        if "display:" not in block.group(1):
            continue
        assert re.search(re.escape(selector) + r"\[hidden\]", css), (
            f"{selector} sets `display` but has no `{selector}[hidden]` override, "
            "so the hidden attribute cannot hide it"
        )


def test_preview_banner_markup_exists_and_starts_hidden():
    assert 'id="seasonPreviewBanner"' in _APP_HTML
    match = re.search(r'<div id="seasonPreviewBanner"[^>]*>', _APP_HTML)
    assert match and "hidden" in match.group(0), (
        "the preview banner must ship hidden; a banner that flashes on the "
        "Competition board trains people to ignore it"
    )


def test_preview_banner_says_the_numbers_are_not_real():
    """Copy, not just presence. A banner that only says 'preview' is decoration."""
    banner = _fn_body("renderSeasonPreviewBanner")
    lowered = banner.lower()
    assert "not deployed" in lowered
    assert "no season has been run" in lowered, (
        "the banner must state that no season ran, not merely that this is a preview"
    )


# ── Gap markers: a missed night must not read like a flat market ─────────────


def test_gap_copy_distinguishes_failure_kinds():
    """One shared string for every failure_kind would defeat the whole list.

    CLAUDE.md's rule is that 'the market was flat' and 'our job died' must never
    render identically. Distinct copy per kind is how that holds on this board.
    """
    match = re.search(r"const SEASON_GAP_COPY = \{(.*?)\n\};", _SOURCE, re.DOTALL)
    assert match, "SEASON_GAP_COPY moved -- re-point this guard or it checks nothing"
    phrases = re.findall(r":\s*'([^']+)'", match.group(1))
    assert len(phrases) >= 3, f"expected a copy line per failure_kind, found {phrases}"
    assert len(set(phrases)) == len(phrases), f"duplicate gap copy: {phrases}"


def test_gap_renderer_states_that_positions_carried_forward():
    """The policy is carry-flat-and-mark, never backfill -- the UI has to say so."""
    body = _fn_body("renderSeasonGaps")
    assert "carried forward" in body


def test_gap_renderer_never_builds_html_from_server_text():
    """``detail`` is server-supplied prose; it goes in via textContent only."""
    body = _fn_body("renderSeasonGaps")
    assert "innerHTML" not in body
    assert "textContent" in body


# ── The retired Daily tab must alias, not vanish ─────────────────────────────


def test_daily_deep_links_resolve_to_the_season_board():
    """#daily is in Discord messages and the nightly-refresh runbook."""
    assert re.search(r"daily:\s*\{\s*page:\s*'competition',\s*competitionTab:\s*'season'\s*\}", _APP_HTML), (
        "the retired ?view=daily / #daily deep link must map to the season tab"
    )


def test_saved_daily_nav_state_is_migrated():
    """localStorage still holds 'daily' for anyone whose last visit was that tab.

    An unrecognised competitionTab matches no boot-CSS rule and no panel, so the
    Competition page paints empty -- a blank screen, not a wrong tab.
    """
    assert "migrateSavedNavState" in _APP_HTML
    migrate = _fn_body("migrateSavedNavState", _strip_js_comments(_APP_HTML))
    assert "'daily'" in migrate and "'season'" in migrate


def test_competition_panel_accepts_the_legacy_tab_key():
    panel = _fn_body("showCompetitionPanel", _strip_js_comments(_APP_JS))
    assert "'daily'" in panel, (
        "showCompetitionPanel is the direct target of the subtab click handler "
        "and of restored nav state; a stray 'daily' must not blank the page"
    )
    assert "tab === 'season'" in panel


def test_the_daily_leaderboard_tab_is_gone_from_the_ui():
    assert 'data-competition-tab="daily"' not in _APP_HTML
    # Comment-stripped: the alias in the nav map documents what it aliases, and
    # that explanation is the opposite of advertising the retired board.
    assert "Daily Leaderboard" not in _APP_HTML_VISIBLE, (
        "the Daily Leaderboard was retired; leaving the label in rendered copy "
        "advertises a board that no longer exists"
    )
    assert 'data-competition-tab="season"' in _APP_HTML
    assert "Live Season" in _APP_HTML


# ── Cache busters ────────────────────────────────────────────────────────────


def test_leaderboard_cache_buster_bumped():
    """leaderboard.js changed, so its ?v= must ship or browsers keep the old file."""
    match = re.search(r"js/leaderboard\.js\?v=(\d+)", _APP_HTML)
    assert match, "leaderboard.js must be loaded with a ?v= cache buster"
    assert int(match.group(1)) >= 26


def test_app_shell_cache_busters_bumped():
    """app.js owns the subtab routing and styles.css the season chrome.

    A stale app.js still routes the retired 'daily' key; a stale styles.css
    renders the season strip and the preview banner unstyled -- the banner in
    particular degrades to a plain paragraph that reads like body copy.
    """
    for asset, floor in (("styles\\.css", 111), ("app\\.js", 110)):
        match = re.search(rf'(?:href|src)="{asset}\?v=(\d+)"', _APP_HTML)
        assert match, f"{asset} must be loaded with a ?v= cache buster"
        assert int(match.group(1)) >= floor
