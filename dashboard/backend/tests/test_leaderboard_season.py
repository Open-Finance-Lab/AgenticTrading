"""The season contract: `live` is a real period, and Season 0 has not advanced.

The client half of this contract already ships in full and has never had a
server to talk to -- js/leaderboard.js reads eleven season fields. These pin the
shape it reads and, more importantly, the one thing that must NOT be true yet.
"""

import json
from datetime import date

import pytest

from dashboard.backend.domain.leaderboard import service


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """`get_leaderboard` calls `ensure_leaderboard_runs`, which fetches bars.

    Stubbed here for every case in this module, following the pattern in
    tests/domain/leaderboard/test_service_move.py. Nothing in this file is about
    run production -- with the suite's temp DATABASE_PATH `_find_cached_run`
    misses on all twelve entries and `entries` comes back empty, which is
    exactly the shape these assertions want: the season block must be attached
    by the PERIOD, not by anything the roster happens to contain.
    """
    monkeypatch.setattr(
        service,
        "ensure_leaderboard_runs",
        lambda force_refresh=False, period="contest", config=None: {
            "session_id": (config or {}).get("session_id", "leaderboard-contest"),
            "created": 0,
            "refreshed_at": "2026-08-19T00:00:00+00:00",
        },
    )


def test_live_is_a_real_period():
    """`_normalize_period` coerces anything unrecognised back to 'contest', so
    before this change `?period=live` returned a perfectly successful HTTP 200
    carrying the Competition board."""
    assert "live" in service.VALID_PERIODS
    assert service._normalize_period("live") == "live"
    assert service._normalize_period("LIVE") == "live"
    assert service._normalize_period("season") == "contest", (
        "coercion stays the behaviour for genuinely unknown periods"
    )


def test_the_live_board_reuses_the_contest_runs_and_window():
    """Nothing in this change may spend money. A live branch with its own window
    would miss `_find_cached_run` on all twelve entries and start recomputing
    baselines -- and, with LEADERBOARD_DAILY_AUTO_DEPLOY armed, LLM deploys --
    from a public, unauthenticated GET."""
    base = service.load_leaderboard_config()
    live = service.resolve_leaderboard_config("live")
    assert live["session_id"] == base["session_id"]
    assert live["start_date"] == base["start_date"]
    assert live["end_date"] == base["end_date"]
    assert live["period"] == "live"


def test_a_season_is_ten_trading_days_which_is_two_calendar_weeks():
    """Ten US cash sessions, Monday through Friday. Not a new number:
    js/leaderboard.js already declares `const SEASON_TRADING_DAYS = 10;` with
    exactly that comment."""
    start, end = service.season_window("2026-08-12", 10)
    assert start == "2026-08-12"
    assert end == "2026-08-25", "Wed 12 Aug through Tue 25 Aug is ten sessions"


def test_the_season_payload_says_nothing_has_advanced():
    """THE invariant. `seasonHasAdvanced()` tests `last_advanced_date` and
    `trading_days_elapsed` -- deliberately, rather than the period string --
    precisely so that adding "live" to VALID_PERIODS cannot clear the preview
    banner. A non-null date here flips the badge to "Running" and promises a
    nightly advance that nothing performs."""
    season = service.build_season_payload(service.resolve_leaderboard_config("live"))
    assert season["last_advanced_date"] is None
    assert season["trading_days_elapsed"] == 0
    assert season["next_advance_at"] is None
    assert season["entries_open"] is False
    assert season["status"] != "running"


def test_season_zero_is_numbered_zero_and_survives_json():
    """Season 0 is the shakedown season by convention: numbered, so the board has
    a real identity to show, but explicitly the one whose results nobody should
    read as a standing. It is also FALSY, which is the whole hazard on the client
    side -- `displayedSeasonNumber()` exists for it."""
    season = service.build_season_payload(service.resolve_leaderboard_config("live"))
    assert season["number"] == 0
    assert json.loads(json.dumps(season))["number"] == 0


def test_the_season_payload_carries_every_field_the_client_reads():
    """The client contract was written before the server existed. A missing key
    is not a crash there -- the render path uses optional chaining throughout --
    it is a silently blank strip."""
    season = service.build_season_payload(service.resolve_leaderboard_config("live"))
    for field in (
        "number", "status", "start_date", "end_date", "last_advanced_date",
        "trading_days_elapsed", "trading_days_total", "entries_open",
        "entry_closes_at", "entry_count", "next_advance_at", "gaps",
    ):
        assert field in season, f"the client reads season.{field}"


def test_the_config_declares_the_season_rather_than_the_code():
    cfg = service.load_leaderboard_config()
    assert cfg["season"]["length_trading_days"] == 10
    assert cfg["season"]["season_zero_start"] == "2026-08-12"


def test_only_the_live_board_carries_a_season():
    """The Competition board is one fixed historical window and is not a season;
    attaching one would make the season strip render on a board that has none."""
    assert "season" not in service.get_leaderboard(period="contest")
    assert "season" in service.get_leaderboard(period="live")


# ── The window is a claim, and a claim that nothing maintains goes stale ─────


def test_the_preview_window_disappears_once_it_has_elapsed():
    """Nothing advances Season 0, so its fortnight becomes a *past* fortnight.

    The config pins 2026-08-12 → 2026-08-25. On 2026-08-26 and every day after,
    the unfixed code kept publishing that window with `trading_days_elapsed: 0`,
    so the strip read "Aug 12 – Aug 25 · Day 0 of 10" indefinitely -- a board
    advertising a schedule it had already missed. No operator error is needed to
    reach this state, only time, which is why it cannot be left to a config bump.

    `(None, None)` lands on copy the client already ships for exactly this:
    "Dates set when the first season opens".
    """
    live = service.resolve_leaderboard_config("live")
    inside = service.build_season_payload(live, as_of=date(2026, 8, 20))
    assert inside["start_date"] == "2026-08-12"
    assert inside["end_date"] == "2026-08-25"

    after = service.build_season_payload(live, as_of=date(2026, 9, 30))
    assert after["start_date"] is None
    assert after["end_date"] is None
    # Present-but-null, never absent: the client reads these keys unguarded-ish.
    assert "start_date" in after and "end_date" in after
    assert after["status"] == "preview"
    assert after["trading_days_total"] == 10


def test_the_last_day_of_the_window_still_counts_as_inside_it():
    """A season that ends today has not elapsed. `<` not `<=`."""
    live = service.resolve_leaderboard_config("live")
    on_the_day = service.build_season_payload(live, as_of=date(2026, 8, 25))
    assert on_the_day["end_date"] == "2026-08-25"


def test_a_missing_season_start_reports_no_window_not_the_contest_one():
    """This fell back to `config["start_date"]` -- the *contest* window -- and
    published 2026-04-15 → 2026-04-28 as a season, `status: preview`, HTTP 200.

    "Nobody configured a season" and "the season is the April contest window"
    were byte-identical responses. That is the failure shape CLAUDE.md's
    fail-closed-is-not-fail-visible section is about.
    """
    season = service.build_season_payload(
        {"start_date": "2026-04-15", "season": {"length_trading_days": 10}}
    )
    assert season["start_date"] is None
    assert season["end_date"] is None


def test_an_unparseable_season_start_reports_no_window_rather_than_raising():
    """`date.fromisoformat` raises ValueError, on a public unauthenticated GET."""
    season = service.build_season_payload({"season": {"season_zero_start": "not-a-date"}})
    assert season["start_date"] is None


def test_a_weekend_start_rolls_forward_to_the_first_session():
    """The returned start is the season's first *session*.

    `season_window` counts weekdays, so a Saturday start was echoed back verbatim
    as a first day on which, by this function's own rule, nothing traded -- and
    the client prints it as the window's first day.
    """
    start, end = service.season_window("2026-08-15", 10)  # Saturday
    assert start == "2026-08-17", "Sat 15 Aug rolls forward to Mon 17 Aug"
    assert end == "2026-08-28"


# ── The length is config, so it is untrusted input on a public GET ──────────


def test_trading_days_total_is_the_number_the_window_was_built_from():
    """The clamp used to live inside `season_window` while the payload reported
    the raw config value, so the two consumers disagreed. The client computes
    `elapsed / total` for a CSS width: a negative total rendered a **100%-full**
    progress bar directly under the banner denying that anything had advanced.
    """
    season = service.build_season_payload(
        {"season": {"length_trading_days": -5, "season_zero_start": "2026-08-12"}}
    )
    assert season["trading_days_total"] == 1, "clamped, and reported as clamped"
    assert season["trading_days_total"] >= 1


def test_a_junk_season_length_falls_back_instead_of_raising():
    """A bare `int()` on a config string raised ValueError out of the GET."""
    season = service.build_season_payload(
        {"season": {"length_trading_days": "ten", "season_zero_start": "2026-08-12"}}
    )
    assert season["trading_days_total"] == service.DEFAULT_SEASON_TRADING_DAYS


def test_the_season_length_bounds_the_calendar_walk():
    """`season_window` steps one day at a time and runs inside a public GET, so
    the config number is the loop bound. Bounded in the function itself, not only
    in its caller."""
    start, end = service.season_window("2026-08-12", 10 ** 9)
    assert start == "2026-08-12"
    span = date.fromisoformat(end) - date.fromisoformat(start)
    assert span.days < 400, "a season length may not become an unbounded loop"


# ── The live board describes itself ─────────────────────────────────────────


def test_the_live_board_does_not_inherit_the_contest_rules_as_its_description():
    """`window.description` is served to every non-browser consumer of this
    payload; /app never renders it, so nothing on screen showed the mismatch.
    Inherited, it shipped the Competition board's *rules* to a tab that does not
    run under them."""
    contest = service.resolve_leaderboard_config("contest")
    live = service.resolve_leaderboard_config("live")
    assert live["description"] != contest["description"]
    assert "look-ahead" not in live["description"]
    assert "preview" in live["description"].lower()


def test_the_live_phase_label_is_derived_from_the_season_number():
    """A literal "Season 0" here is a second owner of the season number, and
    bumping PREVIEW_SEASON_NUMBER is the first act of the advance engine."""
    live = service.resolve_leaderboard_config("live")
    assert live["phase_label"] == f"Season {service.PREVIEW_SEASON_NUMBER}"


def test_the_route_documents_the_third_period():
    """The description is what /docs shows and what the next reader believes.

    Asserted against the `Query` object FastAPI actually serves, not against the
    router file as text: a whole-file substring search for "live" passes on the
    word appearing anywhere -- a comment, an unrelated identifier, the import
    block -- so it could not tell a documented period from an undocumented one.
    """
    import inspect

    from dashboard.backend.api.routers import leaderboard as router

    param = inspect.signature(router.api_get_leaderboard).parameters["period"]
    described = param.default.description
    assert "live" in described, "the period the route now accepts must be in its own description"
    for known in ("contest", "daily"):
        assert known in described, f"the {known} period lost its documentation"
