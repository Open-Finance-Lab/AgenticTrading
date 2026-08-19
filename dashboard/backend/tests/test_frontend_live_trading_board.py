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

**The preview banner.** The season engine is not deployed. The server now
answers ``?period=live`` with ``period: 'live'`` and a real ``season`` block --
it no longer coerces the period back to 'contest' -- but that block is
hardcoded to the not-yet-advanced state, so the hazard is unchanged: every
other element on the tab (chart, table, curve picker, rankings) renders
identically whether or not a season ran, because those shapes are shared
between the two boards. The banner is the *only* thing on screen that
distinguishes "no season has ever run" from "these are the live standings", and
it can only do that by testing a field an advance had to write. See the
fail-closed-is-not-fail-visible section of CLAUDE.md, and the FinSearch news
adapter it was written about. The server half of this contract is pinned in
tests/test_leaderboard_season.py.

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


# ── The retired Daily board's payload shape must not survive as dead code ────


def test_no_daily_status_machinery_survives():
    """``daily_status`` was the Daily board's notice/poll contract.

    The server attaches it only for ``period == 'daily'`` (service.py), a period
    this UI can no longer request -- it asks for 'live', which is now its own
    period and carries a `season` block instead. So every branch keyed on that
    field became unreachable the moment the tab was renamed, and the season
    payload contract defines `season.*` with no `daily_status` at all, so it
    stays unreachable after the engine lands.

    Carried-over code that can never run is worse than absent code once tests
    guard it: three cases in this file used to exercise the notice and the poll,
    passing green, reading as live coverage of the live board.
    """
    assert "daily_status" not in _SOURCE, (
        "leaderboard.js still branches on the retired Daily payload shape"
    )
    assert "leaderboardBoardNotice" not in _APP_HTML, (
        "the notice host outlived the only code that wrote to it"
    )
    for gone in ("scheduleLiveBoardPoll", "isLiveBoardVisible", "renderLiveBoardNotice"):
        assert gone not in _SOURCE, f"{gone} is unreachable and must not survive"


# ── Board switching: the request and its response must move as one value ─────


def test_a_stale_response_can_never_render():
    """Two loads can be in flight at once, and only the newest may paint.

    Subtab clicks are user-paced and the API is not, so Competition -> Live ->
    Competition inside one round-trip resolves out of order. The two boards share
    the chart, the table and the curve picker wholesale -- only the chrome
    differs -- so the loser repainting the winner is silent by construction.
    """
    load = _fn_body("loadLeaderboardData")
    assert re.search(r"seq\s*=\s*\+\+\s*boardRequestSeq", load), (
        "each load must claim a request id before it awaits"
    )
    # Both exits, not only the happy one: an error path that renders regardless
    # drops a failed switch's error over the board the user has since moved to.
    assert len(re.findall(r"seq\s*!==\s*boardRequestSeq\s*\)\s*return", load)) >= 2, (
        "both the success and the failure path must drop a superseded response"
    )


def test_the_rendered_board_is_committed_together_with_its_payload():
    """`renderedBoardPeriod` is the only thing that tells the two boards apart.

    Assigned before the await it belongs to whichever request was issued most
    recently rather than to the response being rendered -- the same defect as
    rendering a stale response, reached from the other side. So it is written at
    the single entry point of the render path, from that render's own argument.
    """
    header = _fn_body("updateLeaderboardHeader")
    assert re.search(r"renderedBoardPeriod\s*=\s*board\b", header), (
        "the render entry point must commit the board its payload was fetched for"
    )
    assert not re.search(r"renderedBoardPeriod\s*=(?!=)", _fn_body("loadLeaderboardData")), (
        "assigning the rendered board inside the fetch re-creates the race: the "
        "only safe moment is the render itself"
    )
    for rhs in re.findall(r"renderedBoardPeriod\s*=(?!=)\s*([^;\n]+)", _SOURCE):
        assert "payload" not in rhs, (
            f"renderedBoardPeriod assigned from the response ({rhs.strip()!r}); "
            "that collapses the request-vs-response comparison every disclaimer "
            "on this tab depends on"
        )


def test_the_board_identity_is_painted_before_the_request():
    """Until the fetch resolves, the markup on screen is the Competition board's.

    The boot stylesheet covers only up to app.js executing -- navigateToPage
    clears data-nav-boot before the fetch is even issued. On the free tier the
    rest of that window is 30-60s, which for a shared ?view=live link is the
    entire first impression: contest title, contest dates, "Upcoming" badge, and
    the preview banner still hidden.
    """
    load = _fn_body("loadLeaderboardData")
    head = load[: load.index("try")]
    assert "updateLeaderboardHeader(" in head, (
        "the board's identity must be painted before awaiting the response"
    )
    # The table has to move with the header. Re-running the normal renderer over
    # an empty payload prints "No entries in this season yet", which for the
    # length of a cold start is a claim about the board nobody can make yet;
    # leaving the previous rows re-attributes them to the board being opened.
    assert "showLeaderboardTableLoading()" in head, (
        "the table must show a loading state across the switch, not an emptiness "
        "claim and not the previous board's rows"
    )
    loading = _fn_body("showLeaderboardTableLoading")
    assert "No entries" not in loading and "${" not in loading, (
        "the placeholder must be a static string: neither an emptiness claim nor "
        "an interpolation into innerHTML"
    )


def test_a_failed_load_repaints_the_board_rather_than_leaving_the_last_one():
    """A failed switch used to leave the previous board's chrome standing.

    Competition -> Live with a failing fetch showed the contest title over the
    contest curves with no preview banner anywhere, while the Live tab sat
    highlighted -- the disclaimer missing in precisely the direction that
    matters. Free-tier cold starts make that an ordinary path, not a corner case.
    """
    load = _fn_body("loadLeaderboardData")
    catch = load[load.index("catch") :]
    assert "updateLeaderboardHeader(" in catch, (
        "the error path must repaint the board identity, not only print an error"
    )
    assert "equityCurvesData = null" in catch, (
        "curves from the previous board must not stay attributed to this one"
    )


def test_live_board_subtitle_matches_the_cash_session_window():
    """Window math is the 16:00 ET close, not a calendar weekday -- say so."""
    body = _fn_body("formatLiveBoardSubtitle")
    assert "cash session" in body
    assert "weekday" not in body


# ── The preview banner: the one control that can report an absent engine ─────


def test_preview_is_anchored_on_evidence_that_a_season_ran():
    """Not on the period string, which is a fact about the backend's vocabulary.

    ``payload.period !== 'live'`` tests whether the server *recognises the word*.
    Adding "live" to VALID_PERIODS is the natural first commit of the season
    engine -- a one-line change that needs no season payload at all -- and it
    would clear the banner, flip the badge from Preview to Running, print "Next
    advance: nightly after the 16:00 ET close" and promise a completed session,
    with nothing else on the tab left to disagree. Nothing would have run.

    The banner's claim is that nothing here advanced, so that is what it tests:
    a field only a real advance can write.
    """
    body = _fn_body("isLivePreview")
    assert "isLiveBoard()" in body, (
        "preview detection must consult the requested board, not just the payload"
    )
    assert "seasonHasAdvanced(" in body, (
        "preview must be decided by whether a session was actually advanced"
    )
    evidence = _fn_body("seasonHasAdvanced")
    assert "last_advanced_date" in evidence and "trading_days_elapsed" in evidence
    assert not re.search(r"period\s*!==?\s*'live'", _SOURCE), (
        "no disclaimer on this tab may hang on the period string; a backend that "
        "learns the word has not run a season"
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


def test_preview_banner_has_a_boot_state():
    """The honesty control needs to exist before any script does.

    ?view=live is revealed by the boot stylesheet during parse, so without a boot
    rule the banner stays `hidden` through first paint and the contest identity
    below it is what the visitor reads. The identity is suppressed for the same
    window rather than shown wrong.
    """
    assert re.search(
        r'html\[data-nav-boot\][^{]*data-nav-competition-tab="live"[^{]*'
        r"#seasonPreviewBanner\s*\{[^}]*display:\s*block\s*!important",
        _APP_HTML,
    ), "the live tab must reveal the preview banner at boot"
    assert re.search(
        r'html\[data-nav-boot\][^{]*data-nav-competition-tab="live"[^{]*'
        r"\.contest-identity[^{]*\{[^}]*visibility:\s*hidden",
        _APP_HTML,
    ), "the Competition identity must not paint on the live tab before JS runs"


def test_preview_banner_ships_with_real_text_not_an_empty_div():
    """A revealed empty div is indistinguishable from no disclaimer at all."""
    match = re.search(r'<div id="seasonPreviewBanner"[^>]*>(.*?)</div>', _APP_HTML, re.DOTALL)
    assert match, "the preview banner element must exist"
    static = match.group(1)
    assert "not deployed" in static, (
        "the boot-visible banner must carry the disclaimer itself; JS replaces it "
        "with the fuller wording once the payload names the window"
    )


def test_competition_only_chrome_is_hidden_on_the_live_board():
    """The header rewrite covers the title, badge and first subtitle only.

    Everything else in that block describes the SecureFinAI contest specifically:
    an organizing body, a Rules modal written around a fixed window and a
    registration deadline, and a "Phase" stat label captioning a season number.
    Left standing they caption the live board as a contest event it is not part
    of, under a Rules button whose rules do not govern it.
    """
    header = _fn_body("updateLeaderboardHeader")
    # The assignment, not the id string. Looking one up and doing nothing with it
    # reads identically to a substring check, which is how a "toggle" guard stays
    # green over an element that never moves.
    for element_id, var in (("contestOrganizerLine", "organizerEl"), ("competitionRulesBtn", "rulesBtn")):
        assert f'id="{element_id}"' in _APP_HTML, f"{element_id} must exist to be toggled"
        assert f"getElementById('{element_id}')" in header, (
            f"{element_id} is Competition-only and must be resolved per board"
        )
        assert re.search(rf"{var}\.hidden\s*=\s*liveBoard\b", header), (
            f"{element_id} must actually be hidden on the live board, not merely looked up"
        )
    assert re.search(r"phaseLabelEl\.textContent\s*=\s*liveBoard", header), (
        "the stat label must move with its value; 'Phase: Season 0' captions one "
        "board's noun with the other's"
    )
    assert 'id="boardPhaseLabel"' in _APP_HTML


def test_relative_time_never_renders_a_dangling_label():
    """Both call sites interpolate the phrase straight into a caption.

    An empty string renders "Next advance " or "Entries open - 42 entered -
    closes " -- a label with its value silently deleted, which reads as a broken
    template rather than as missing data.
    """
    body = _fn_body("formatRelativeFromNow")
    assert re.search(r"return\s+null", body), (
        "an unparseable timestamp must return null, not an empty string"
    )
    assert "'due now'" not in body, (
        "the phrase is composed into 'closes ...' as well as 'Next advance ...', "
        "and an entry window cannot be 'closes due now'"
    )
    strip = _fn_body("renderSeasonStrip")
    # The formatted phrase, not the raw field, decides whether the clause ships.
    assert re.search(r"closesIn\s*\?", strip), (
        "the entry line must test the formatted phrase; testing the raw field "
        "renders a bare '- closes '"
    )
    assert not re.search(r"season\?\.next_advance_at\s*\)\s*\{", strip), (
        "the next-advance line must branch on the formatted phrase too, or an "
        "unparseable timestamp renders 'Next advance null'"
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
    assert "window?.start_date" not in body and "window.start_date" not in body
    # And it must read the field the season contract actually defines. The
    # carried-over `daily_status.trading_date` is the retired Daily board's
    # shape: never sent for this board, so the subtitle would fall through to
    # the bare cadence line forever while the backend was sending the date.
    assert "last_advanced_date" in body, (
        "the subtitle must read season.last_advanced_date -- the only field the "
        "season payload contract defines for a completed advance"
    )


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
    """Four places print the number: strip badge, Phase stat, banner, subtitle."""
    for fn in (
        "renderSeasonStrip",
        "updateLeaderboardHeader",
        "renderLivePreviewBanner",
        "formatLiveBoardSubtitle",
    ):
        body = _fn_body(fn)
        if "Season $" not in body and "Season ${" not in body:
            continue
        assert "displayedSeasonNumber(" in body or "displayedSeasonLabel(" in body, (
            f"{fn} formats a season number without the resolver, so it can "
            "reintroduce the season-0-is-falsy bug independently"
        )


def test_the_season_number_has_exactly_one_owner_on_screen():
    """The payload's `season.number` decides; this file's constant only fills in.

    The banner and the subtitle used to interpolate PREVIEW_SEASON_NUMBER
    directly while the badge read the payload, which meant the number had two
    owners that could disagree. The disagreement is not hypothetical: the first
    act of the advance engine is bumping the *server's* PREVIEW_SEASON_NUMBER,
    at which point the badge reads "Season 1" and the banner underneath it reads
    "Season 0 has not been run".

    Guarding the resolver alone cannot catch this -- the offending lines never
    called it -- so the guard is on where the constant may appear at all.
    """
    remainder = _SOURCE.replace(_fn_body("displayedSeasonNumber"), "")
    remainder = re.sub(r"const PREVIEW_SEASON_NUMBER\s*=\s*\d+\s*;", "", remainder)
    assert "PREVIEW_SEASON_NUMBER" not in remainder, (
        "PREVIEW_SEASON_NUMBER is read outside displayedSeasonNumber(); it is a "
        "fallback for a missing payload, not a second source of the season "
        "number. Render it through displayedSeasonLabel()/displayedSeasonNumber()."
    )


def test_the_season_label_never_prints_a_null_number():
    """`displayedSeasonNumber` returns null for "no season at all"."""
    body = _fn_body("displayedSeasonLabel")
    assert "=== null" in body or "== null" in body, (
        "displayedSeasonLabel must handle the null the resolver can return, or "
        "the banner renders the words 'Season null'"
    )


def test_the_season_constants_are_the_same_number_in_all_three_places():
    """10 and 0 are declared in Python, in JS, and in leaderboard.json.

    Nothing but a comment tied them together, and the comment is in the file a
    reader is already looking at -- so the drift it warns about is invisible
    from either of the other two. Season length in particular is the divisor
    behind the progress bar on one side and the window builder on the other.
    """
    import json

    from dashboard.backend.domain.leaderboard import service

    config = json.loads(
        (Path(__file__).resolve().parents[2] / "config" / "leaderboard.json")
        .read_text(encoding="utf-8")
    )
    js_days = re.search(r"const SEASON_TRADING_DAYS\s*=\s*(\d+)", _SOURCE)
    js_preview = re.search(r"const PREVIEW_SEASON_NUMBER\s*=\s*(\d+)", _SOURCE)
    assert js_days and js_preview, "the JS season constants moved -- re-point this guard"

    assert int(js_days.group(1)) == service.DEFAULT_SEASON_TRADING_DAYS, (
        "js/leaderboard.js and service.py disagree on the length of a season"
    )
    assert config["season"]["length_trading_days"] == service.DEFAULT_SEASON_TRADING_DAYS, (
        "leaderboard.json and service.py disagree on the length of a season"
    )
    assert int(js_preview.group(1)) == service.PREVIEW_SEASON_NUMBER, (
        "js/leaderboard.js and service.py disagree on which season is the preview"
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


def test_no_earlier_redirect_undoes_the_alias():
    """PR #335 parked the Daily tab by redirecting 'daily' -> 'leaderboard'.

    That redirect sat ~20 lines above this branch's 'daily' -> 'live' aliasing and
    ran first, so every path where `options.competitionTab` was absent and the
    module-level `competitionTab` still held 'daily' landed on Competition. Two
    rules for one key, the older one winning: the silent fall-through to the
    wrong board that the aliases exist to prevent.
    """
    nav = _fn_body("navigateToPage", _strip_js_comments(_APP_JS))
    assert "competitionTab: 'leaderboard'" not in nav, (
        "a surviving 'daily' -> 'leaderboard' redirect overrides the live alias"
    )
    # Normalising `options.competitionTab` alone misses the case the redirect
    # used to cover -- a caller with no option and a stale module value.
    assert re.search(
        r"if\s*\(\s*competitionTab\s*===\s*'daily'\s*\|\|\s*competitionTab\s*===\s*'season'\s*\)",
        nav,
    ), "the retired keys must be normalised on the resolved value, not the argument"


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
#
# Deliberately none here. This suite's four assets (app.js, styles.css,
# js/leaderboard.js, home-page.js) are all asserted, exactly, by
# test_frontend_fast_boot.py::test_cache_busters_bumped, which owns the
# invariant. A `>=` floor beside that exact match is not extra safety: the two
# disagree at the next bump, the floor stays green, and the exact one reads as
# the broken guard and gets loosened. One owner, one rule.


def test_home_module_link_targets_the_live_board():
    assert 'id="homeModuleLiveBtn"' in _APP_HTML
    home = (_FRONTEND / "home-page.js").read_text(encoding="utf-8")
    assert "homeModuleLiveBtn" in home, "the home-page handler must bind the shipped id"
    assert re.search(r"competitionTab:\s*'live'", home)


def test_the_home_blurb_does_not_claim_the_advance_is_running():
    """The home page never renders the preview banner.

    So a present-tense "advances nightly on bars nobody has seen yet" there is the
    board's over-claim reproduced two clicks upstream of its only correction --
    the same sentence this PR removed from the landing's Race section and then
    reintroduced in-app.
    """
    blurb = re.search(
        r'<p class="hm-rank-season">(.*?)</p>', _APP_HTML_VISIBLE, re.DOTALL
    )
    assert blurb, "the home ranking module must carry the live-board blurb"
    text = " ".join(blurb.group(1).split())
    assert "Season 0 preview" in text, (
        "the home blurb must carry the preview qualifier; nothing else on that "
        "page does"
    )


def test_the_home_sample_note_separates_absent_from_broken():
    """`markSample` covered both an unreachable API and an empty model roster.

    An empty roster is an ordinary live state -- baselines compute on first load
    and models deploy after -- so one shared message reporting "could not be
    loaded" diagnoses a healthy backend as broken. Fail-closed is not
    fail-visible; the note is the only visible surface either state has.
    """
    home = (_FRONTEND / "home-page.js").read_text(encoding="utf-8")
    assert "SAMPLE_NOTES" in home
    assert "unreachable" in home and "empty" in home
    assert re.search(r"sample:\s*'empty'", home), (
        "the zero-model-entries path must not report a transport failure"
    )
    assert re.search(r"sample:\s*'unreachable'", home)


def test_manual_refresh_does_not_default_to_billable_deploys():
    """The schedule is paused because the job's output has no surface.

    Leaving workflow_dispatch's deploy_models defaulted to true keeps the same
    seven-model spend one click away, for curves no route can display. Opting in
    has to be a deliberate tick.
    """
    workflow = (
        Path(__file__).resolve().parents[3] / ".github" / "workflows" / "daily-leaderboard.yml"
    ).read_text(encoding="utf-8")
    block = re.search(r"deploy_models:(.*?)(?=\n\S|\nconcurrency)", workflow, re.DOTALL)
    assert block, "the deploy_models input must exist"
    assert re.search(r"default:\s*false", block.group(1)), (
        "deploy_models must default to false while no board can show the result"
    )
