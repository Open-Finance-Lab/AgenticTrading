"""The season contract: `live` is a real period, and Season 0 has not advanced.

The client half of this contract already ships in full and has never had a
server to talk to -- js/leaderboard.js reads eleven season fields. These pin the
shape it reads and, more importantly, the one thing that must NOT be true yet.
"""

import json

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
