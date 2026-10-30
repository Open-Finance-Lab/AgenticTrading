"""Unit tests for the leaderboard refresh loops' per-entry failure bookkeeping."""

import pytest

from dashboard.backend.domain.leaderboard.roster_failures import (
    RosterFailures,
    describe_error,
)
from dashboard.backend.domain.leaderboard.service import LeaderboardFallbackError


def _raised(exc):
    """``exc`` with a real traceback, as the refresh loop's catch sees it."""
    try:
        raise exc
    except Exception as caught:
        return caught


class _PsycopgError(Exception):
    pass


_PsycopgError.__module__ = "psycopg"
_PsycopgError.__qualname__ = "OperationalError"


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (TimeoutError(), "TimeoutError"),
        (ConnectionResetError("reset"), "ConnectionResetError: reset"),
        (KeyError("cash"), "KeyError: 'cash'"),
        (
            _PsycopgError("server closed\n\tThis probably means"),
            "psycopg.OperationalError: server closed This probably means",
        ),
        (LeaderboardFallbackError("h6"), "LeaderboardFallbackError: h6"),
    ],
)
def test_describe_error_is_one_line_and_never_empty(exc, expected):
    assert describe_error(exc) == expected


def test_one_failure_or_two_different_ones_do_not_stop_the_roster():
    roster = RosterFailures("live")
    roster.failed("a", _raised(ConnectionError("reset")))
    assert not roster.stopped
    roster.failed("b", _raised(TimeoutError("slow")))
    assert not roster.stopped
    assert [f["entry_id"] for f in roster.failures] == ["a", "b"]


def test_the_same_error_twice_in_a_row_stops_it_even_when_it_names_the_entry():
    roster = RosterFailures("live")
    roster.failed("a", _raised(RuntimeError("Live 'a': no bars for 2026-08-27")))
    roster.failed("b", _raised(RuntimeError("Live 'b': no bars for 2026-08-27")))
    assert roster.stopped
    roster.skip("c")
    assert roster.failures[-1] == {
        "entry_id": "c",
        "error": "not attempted: a and b failed in a row with the same RuntimeError",
    }


def test_only_the_whole_entry_id_is_masked():
    """An id inside another token (a run id) is left alone, so the two errors
    differ and the roster carries on: masking errs toward isolation."""
    roster = RosterFailures("live")
    roster.failed("a", _raised(RuntimeError("row lb_a_20260801 is locked")))
    roster.failed("b", _raised(RuntimeError("row lb_b_20260801 is locked")))
    assert not roster.stopped


def test_an_entry_that_trades_in_between_resets_the_streak_and_a_cached_one_does_not():
    roster = RosterFailures("live")
    roster.failed("a", _raised(KeyError("cash")))
    roster.traded({"entry_id": "b", "cached": True})
    roster.failed("c", _raised(KeyError("cash")))
    assert roster.stopped

    roster = RosterFailures("live")
    roster.failed("a", _raised(KeyError("cash")))
    roster.traded({"entry_id": "b", "cached": False})
    roster.failed("c", _raised(KeyError("cash")))
    assert not roster.stopped


def test_only_types_the_catch_never_named_print_a_traceback(capsys):
    roster = RosterFailures("daily")
    roster.failed("a", _raised(LeaderboardFallbackError("coverage 40%")))
    expected_out = capsys.readouterr().out
    assert expected_out == (
        "ERROR: daily.model_deploy_failed entry=a "
        "error=LeaderboardFallbackError: coverage 40%\n"
    )

    roster.failed("b", _raised(AttributeError("'NoneType' object has no attribute 'cash'")))
    out = capsys.readouterr().out
    first, rest = out.split("\n", 1)
    assert first == (
        "ERROR: daily.model_deploy_failed entry=b "
        "error=AttributeError: 'NoneType' object has no attribute 'cash'"
    )
    assert rest.startswith("Traceback (most recent call last)")
