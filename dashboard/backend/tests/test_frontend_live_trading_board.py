"""Source guards for the Live Trading Leaderboard (frontend/js/leaderboard.js).

/app has no build step and no JS test toolchain, so these assert against the
shipped source as text -- the convention set by test_ai_hedge_fund_frontend.py.
This file replaces test_frontend_daily_leaderboard.py: the Daily Leaderboard was
retired in favour of a forward-running board that carries a portfolio across
trading days in two-week seasons, and the poll/visibility machinery moved with
it under new names.

Three things here are load-bearing for reasons the code alone does not show.

**The poll guard.** ``scheduleLiveBoardPoll`` is only ever re-entered from
``loadLeaderboardData``, which fires on a Competition subtab switch but *not*
when the user navigates away from Competition entirely. Without a visibility
re-check inside the timeout callback, a refresh that is in progress keeps the
30s poll re-fetching and re-rendering a hidden Chart.js canvas for the whole
(possibly multi-hour) model deploy.

**The preview banner.** The season engine is not deployed, and the server
coerces an unknown ``period`` back to 'contest' rather than 4xx-ing it, so
asking for the live board returns HTTP 200 carrying the Competition board.
Every other element on the tab -- chart, table, curve picker, rankings --
renders identically either way, because those shapes are shared between the two
boards. The banner is the *only* thing on screen that distinguishes "no season
has ever run" from "these are the live standings", and it can only do that by
comparing what was requested against what came back. See the
fail-closed-is-not-fail-visible section of CLAUDE.md, and the FinSearch news
adapter it was written about.

**Season 0.** The current season is numbered zero, which is falsy. Every
``season.number ? ... : '-'`` in this file's subject matter renders the live
season as *no season at all*, and it does so silently and only for season 0 --
the exact season shipping right now. The number is therefore read in one place,
through an explicit finite check.
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
        "the 30s poll must re-check visibility when it fires; navigating away "
        "from Competition never calls loadLeaderboardData again, so without "
        "this it polls and re-renders a hidden chart forever"
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
    assert "'live'" in body, "the poll must key on the live tab, not a retired one"


def test_live_board_subtitle_matches_the_cash_session_window():
    """Window math is the 16:00 ET close, not a calendar weekday -- say so."""
    body = _fn_body("formatLiveBoardSubtitle")
    assert "cash session" in body
    assert "weekday" not in body


# ── The preview banner: the one control that can report an absent engine ─────


def test_preview_is_decided_by_comparing_request_against_response():
    """A board the server cannot serve comes back as a valid contest payload.

    Checking only ``payload.season`` would be a check against the *response*,
    which is exactly what a coerced period leaves looking normal. The comparison
    has to involve what was asked for.
    """
    body = _fn_body("isLivePreview")
    assert "isLiveBoard()" in body, (
        "preview detection must consult the requested board, not just the payload"
    )
    assert re.search(r"period\s*!==\s*'live'", body), (
        "preview detection must compare the returned period against 'live'"
    )


def test_requested_period_is_recorded_from_the_request_not_the_payload():
    """Non-vacuity for the test above: ``isLiveBoard`` has to read a real value.

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
    assert "renderLivePreviewBanner(" in _fn_body("updateLeaderboardHeader")
    banner = _fn_body("renderLivePreviewBanner")
    assert "isLivePreview(" in banner
    assert "seasonPreviewBanner" in banner
    assert "hidden = false" in banner, "the banner must actually be shown, not only computed"


def test_hidden_season_containers_are_actually_hidden():
    """`display: flex` on a class outranks the UA stylesheet's [hidden] rule.

    Every season container ships with the `hidden` attribute and is un-hidden by
    JS only on the live tab. A `display` declaration without a matching
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
    banner = _fn_body("renderLivePreviewBanner")
    lowered = banner.lower()
    assert "not deployed" in lowered
    assert "has not been run" in lowered, (
        "the banner must state that the season has not run, not merely that "
        "this is a preview"
    )


def test_preview_never_claims_a_completed_advance():
    """The subtitle used to print the *board window's* first day as "last completed".

    In preview that rendered "last completed 2026-04-15" on a board that has
    never advanced once -- a specific, plausible, entirely invented date sitting
    directly under a banner saying no season has run. `window.start_date` is a
    display range, never evidence that a nightly job did anything.
    """
    body = _fn_body("formatLiveBoardSubtitle")
    assert "isLivePreview(" in body, (
        "the subtitle must suppress any last-completed claim in preview"
    )
    assert not re.search(r"last completed[^`\n]*window", body), (
        "last-completed must not be sourced from the board window"
    )
    # The fallback chain itself must not reach window.start_date.
    tail = body[body.index("isLivePreview(") :]
    assert "window?.start_date" not in tail and "window.start_date" not in tail


def test_preview_never_promises_a_scheduled_advance():
    """"Next advance: nightly after the close" is a promise no deployed job keeps."""
    body = _fn_body("renderSeasonStrip")
    assert "isLivePreview(" in body, (
        "the next-advance line must branch on preview; describing the cadence "
        "unconditionally advertises a nightly job that does not exist"
    )


# ── Season 0 is falsy, and that is the whole hazard ──────────────────────────


def test_season_zero_is_never_tested_for_truthiness():
    """`season?.number ? ... : '-'` renders Season 0 as "no season".

    This is the live value right now, so the bug would ship pointing at the
    only season that exists. It also cannot be caught by a test that passes a
    non-zero number, which is why the guard is on the source shape.
    """
    offenders = re.findall(r"season\s*\??\.\s*number\s*(?:\?[^?]|\|\|)", _SOURCE)
    assert not offenders, (
        f"season number tested for truthiness ({offenders}); season 0 is a real "
        "season and would render as absent. Use displayedSeasonNumber()."
    )


def test_season_number_is_resolved_through_one_finite_check():
    body = _fn_body("displayedSeasonNumber")
    assert "Number.isFinite" in body, (
        "the season number must be validated as a finite number, not by "
        "truthiness -- 0 is a season and NaN is not"
    )
    assert "PREVIEW_SEASON_NUMBER" in body, (
        "with no engine deployed the payload carries no season; the preview "
        "still has to name the season it is previewing"
    )


def test_preview_season_is_zero():
    assert re.search(r"const PREVIEW_SEASON_NUMBER\s*=\s*0\b", _SOURCE), (
        "the shakedown season is Season 0; Season 1 is the first that counts"
    )


def test_every_rendered_season_number_goes_through_the_resolver():
    """Two places print the number: the strip badge and the Phase stat."""
    for fn in ("renderSeasonStrip", "updateLeaderboardHeader"):
        body = _fn_body(fn)
        if "Season $" not in body and "Season ${" not in body:
            continue
        assert "displayedSeasonNumber(" in body, (
            f"{fn} formats a season number without the resolver, so it can "
            "reintroduce the season-0-is-falsy bug independently"
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


# ── Naming: the board is named, and the name does not over-claim ─────────────


def test_the_board_names_itself_on_screen():
    header = _fn_body("updateLeaderboardHeader")
    assert "'Live Trading Leaderboard'" in header, (
        "the board title must name the board; folding the season number in "
        "here instead leaves it unnamed whenever a season is running"
    )
    assert 'data-competition-tab="live">Live Trading Leaderboard<' in _APP_HTML


def test_the_live_name_is_disclaimed_as_simulated():
    """"Live Trading" is a claim. The About card is where it gets qualified.

    PR #328's spec puts brokered execution in non-goals and
    execution/paper_backend.py is still a stub, so a board named for live
    trading that never says "simulated" is the UI making a promise the system
    does not keep.
    """
    about = _APP_HTML_VISIBLE[_APP_HTML_VISIBLE.index("Live Trading Leaderboard</h3>") :]
    about = about[: about.index("</div>")]
    lowered = about.lower()
    assert "simulated" in lowered, "the About card must say the trading is simulated"
    assert "no real capital" in lowered or "no broker" in lowered


def test_about_card_names_the_current_season():
    about = _APP_HTML_VISIBLE[_APP_HTML_VISIBLE.index("Live Trading Leaderboard</h3>") :]
    about = about[: about.index("</div>")]
    assert "Season 0" in about, (
        "the board is in Season 0; the About card is where that is established"
    )


# ── The retired tab keys must alias, not vanish ──────────────────────────────


def test_retired_deep_links_resolve_to_the_live_board():
    """#daily is in Discord messages and the nightly-refresh runbook.

    'season' is the same problem one generation later: it was this tab's key
    through PR #352's review screenshots before the board was named.
    """
    for legacy in ("daily", "season"):
        assert re.search(
            rf"{legacy}:\s*\{{\s*page:\s*'competition',\s*competitionTab:\s*'live'\s*\}}",
            _APP_HTML,
        ), f"the retired ?view={legacy} deep link must map to the live tab"


def test_saved_nav_state_is_migrated_from_both_retired_keys():
    """localStorage still holds the old key for anyone whose last visit was it.

    An unrecognised competitionTab matches no boot-CSS rule and no panel, so the
    Competition page paints empty -- a blank screen, not a wrong tab.
    """
    assert "migrateSavedNavState" in _APP_HTML
    migrate = _fn_body("migrateSavedNavState", _strip_js_comments(_APP_HTML))
    assert "'daily'" in migrate and "'season'" in migrate
    assert "competitionTab: 'live'" in migrate


def test_competition_panel_accepts_the_legacy_tab_keys():
    panel = _fn_body("showCompetitionPanel", _strip_js_comments(_APP_JS))
    assert "'daily'" in panel and "'season'" in panel, (
        "showCompetitionPanel is the direct target of the subtab click handler "
        "and of restored nav state; a stray retired key must not blank the page"
    )
    assert "tab === 'live'" in panel


def test_the_daily_leaderboard_tab_is_gone_from_the_ui():
    assert 'data-competition-tab="daily"' not in _APP_HTML
    # Comment-stripped: the alias in the nav map documents what it aliases, and
    # that explanation is the opposite of advertising the retired board.
    assert "Daily Leaderboard" not in _APP_HTML_VISIBLE, (
        "the Daily Leaderboard was retired; leaving the label in rendered copy "
        "advertises a board that no longer exists"
    )
    assert 'data-competition-tab="live"' in _APP_HTML


def test_the_working_title_is_gone_from_rendered_copy():
    """'Live Season' was the placeholder name; only the key survives as an alias."""
    assert "Live Season" not in _APP_HTML_VISIBLE


# ── Cache busters ────────────────────────────────────────────────────────────


def test_leaderboard_cache_buster_bumped():
    """leaderboard.js changed, so its ?v= must ship or browsers keep the old file."""
    match = re.search(r"js/leaderboard\.js\?v=(\d+)", _APP_HTML)
    assert match, "leaderboard.js must be loaded with a ?v= cache buster"
    assert int(match.group(1)) >= 27


def test_app_shell_cache_busters_bumped():
    """app.js owns the subtab routing and styles.css the season chrome.

    A stale app.js still routes the retired keys; a stale styles.css renders the
    season strip and the preview banner unstyled -- the banner in particular
    degrades to a plain paragraph that reads like body copy.
    """
    for asset, floor in (("styles\\.css", 112), ("app\\.js", 111)):
        match = re.search(rf'(?:href|src)="{asset}\?v=(\d+)"', _APP_HTML)
        assert match, f"{asset} must be loaded with a ?v= cache buster"
        assert int(match.group(1)) >= floor


def test_home_module_link_targets_the_live_board():
    assert 'id="homeModuleLiveBtn"' in _APP_HTML
    home = (_FRONTEND / "home-page.js").read_text(encoding="utf-8")
    assert "homeModuleLiveBtn" in home, "the home-page handler must bind the shipped id"
    assert re.search(r"competitionTab:\s*'live'", home)
