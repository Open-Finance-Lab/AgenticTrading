"""Source guards for the Daily Leaderboard tab (dashboard/frontend/js/leaderboard.js).

/app has no build step and no JS test toolchain, so these assert against the
shipped source as text -- the convention set by test_ai_hedge_fund_frontend.py.

The one that matters is the poll guard. ``scheduleDailyLeaderboardPoll`` is only
ever re-entered from ``loadLeaderboardData``, which fires on a Competition
subtab switch but *not* when the user navigates away from Competition entirely.
Without a visibility re-check inside the timeout callback, a refresh that is in
progress keeps the 30s poll re-fetching and re-rendering a hidden Chart.js
canvas for the whole (possibly multi-hour) model deploy.
"""

import re
from pathlib import Path

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_LEADERBOARD_JS = (_FRONTEND / "js" / "leaderboard.js").read_text(encoding="utf-8")
_APP_HTML = (_FRONTEND / "app.html").read_text(encoding="utf-8")


def _strip_js_comments(source: str) -> str:
    """Drop // and /* */ comments so a guard cannot be satisfied by prose.

    Every assertion below is about code that must (or must not) exist. A comment
    mentioning ``isDailyBoardVisible`` would otherwise pass the check while the
    call itself was deleted.
    """
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"^\s*//.*$", "", source, flags=re.MULTILINE)


_SOURCE = _strip_js_comments(_LEADERBOARD_JS)


def _fn_body(name: str) -> str:
    """The named function's source, brace-matched to its closing brace."""
    start = _SOURCE.index(f"function {name}(")
    index = _SOURCE.index("{", _SOURCE.index(")", start))
    depth = 0
    while True:
        if _SOURCE[index] == "{":
            depth += 1
        elif _SOURCE[index] == "}":
            depth -= 1
            if depth == 0:
                return _SOURCE[start : index + 1]
        index += 1


def test_daily_poll_rechecks_visibility_when_it_fires():
    body = _fn_body("scheduleDailyLeaderboardPoll")
    assert "setTimeout" in body
    callback = body[body.index("setTimeout") :]
    assert "isDailyBoardVisible()" in callback, (
        "the 30s daily poll must re-check visibility when it fires; navigating "
        "away from Competition never calls loadLeaderboardData again, so "
        "without this it polls and re-renders a hidden chart forever"
    )
    # The guard has to bail, not merely observe.
    assert re.search(r"if\s*\(\s*!\s*isDailyBoardVisible\(\)\s*\)\s*return", callback)


def test_daily_poll_only_runs_while_a_worker_is_running():
    body = _fn_body("scheduleDailyLeaderboardPoll")
    assert "refresh_in_progress" in body, (
        "pending curves with no worker wait for the nightly cron; polling those "
        "just hammers the API"
    )


def test_visibility_helper_reads_the_dom_not_the_boot_attributes():
    body = _fn_body("isDailyBoardVisible")
    # html[data-nav-*] is written by navigateToPage but NOT by
    # showCompetitionPanel, so it goes stale on a plain subtab switch.
    assert "data-nav-competition-tab" not in body
    assert "leaderboardView" in body
    assert "competitionTab" in body


def test_daily_subtitle_matches_the_cash_session_window():
    """Window math is 16:00 ET close, not a calendar weekday — say so."""
    body = _fn_body("formatDailyBoardSubtitle")
    assert "cash session" in body
    assert "weekday" not in body


def test_leaderboard_cache_buster_bumped():
    """leaderboard.js changed, so its ?v= must ship or browsers keep the old file."""
    match = re.search(r"js/leaderboard\.js\?v=(\d+)", _APP_HTML)
    assert match, "leaderboard.js must be loaded with a ?v= cache buster"
    assert int(match.group(1)) >= 19


def test_daily_leaderboard_subtab_is_parked():
    """UI is hidden until daily model deploys are reliable; backend stays.

    Re-enable by removing ``hidden`` from the subtab (and About card), restoring
    ``NAV_VIEW_MAP['daily']`` → ``competitionTab: 'daily'``, dropping the
    navigateToPage redirect in app.js, and turning the workflow schedule back on.
    """
    assert re.search(
        r'data-competition-tab="daily"[^>]*\bhidden\b',
        _APP_HTML,
    ), "Daily Leaderboard subtab must stay hidden while the board is parked"
    assert "competitionTab: 'leaderboard'" in _APP_HTML
    assert re.search(r"\bdaily\s*:\s*\{[^}]*competitionTab:\s*'leaderboard'", _APP_HTML)
