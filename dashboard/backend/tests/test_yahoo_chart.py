"""Contract tests for the Yahoo chart fetcher.

Two failure classes are covered here, and they are the two that were previously
invisible:

* a run window that isn't ``YYYY-MM-DD`` — raised ``ValueError`` *before* any
  HTTP call, so it could never be a ``RequestException`` and escaped every
  transport-level guard as an unhandled 500 on a public route;
* a failure Yahoo delivers *inside* a 200 response — classified as "this window
  simply had no data", which is the absent-vs-broken collapse CLAUDE.md warns
  about ("fail-closed is not fail-visible").
"""

import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
import requests

from dashboard.backend.domain.leaderboard.strategies import _yahoo
from dashboard.backend.domain.leaderboard.strategies._yahoo import (
    YahooChartError,
    _epoch,
    fetch_index_hourly,
    usable_window,
)


class _FakeResponse:
    def __init__(self, payload, status: int = 200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error")

    def json(self):
        return self._payload


def _stub_response(monkeypatch, payload, status: int = 200):
    # Rebind the *name* inside _yahoo, never `requests.get` itself: `_yahoo.requests`
    # is the one shared requests module, so patching an attribute on it swaps
    # HTTP for every module in the process for the duration of the test.
    stub = SimpleNamespace(
        get=lambda *_a, **_k: _FakeResponse(payload, status),
        RequestException=requests.RequestException,
    )
    monkeypatch.setattr(_yahoo, "requests", stub)


# --------------------------------------------------------------------------
# Window parsing
# --------------------------------------------------------------------------


def test_epoch_accepts_plain_dates():
    assert _epoch("2026-05-01") == int(
        dt.datetime(2026, 5, 1, tzinfo=dt.timezone.utc).timestamp()
    )


def test_epoch_accepts_iso_stamps_written_by_paper_baselines():
    # domain/backtesting/baselines/paper.py stores start_date.isoformat(), and
    # two such rows are live in the committed seed DB (which *is* the prod DB).
    assert _epoch("2026-06-02T20:00:00") == int(
        dt.datetime(2026, 6, 2, 20, 0, tzinfo=dt.timezone.utc).timestamp()
    )


def test_epoch_keeps_an_explicit_offset():
    assert _epoch("2026-06-02T20:00:00Z") == int(
        dt.datetime(2026, 6, 2, 20, 0, tzinfo=dt.timezone.utc).timestamp()
    )


@pytest.mark.parametrize("bad", ["", "   ", "not-a-date", "05/01/2026"])
def test_epoch_rejects_unparseable_windows(bad):
    with pytest.raises(ValueError):
        _epoch(bad)


def test_usable_window_screens_both_ends():
    assert usable_window("2026-05-01", "2026-05-07") is True
    assert usable_window("2026-06-02T20:00:00", "2026-07-02T20:00:00") is True
    # api/routers/paper_trading.py writes end_date="" — and the plot route
    # passes `run.get("end_date") or ""`, so "" is reachable by construction.
    assert usable_window("2026-05-01", "") is False
    assert usable_window("", "2026-05-01") is False


# --------------------------------------------------------------------------
# Response envelopes
# --------------------------------------------------------------------------


def test_fetch_returns_points_for_a_healthy_payload(monkeypatch):
    ts = int(dt.datetime(2026, 5, 1, 14, 30, tzinfo=dt.timezone.utc).timestamp())
    _stub_response(
        monkeypatch,
        {
            "chart": {
                "error": None,
                "result": [
                    {
                        "timestamp": [ts],
                        "indicators": {"quote": [{"close": [41_000.0]}]},
                    }
                ],
            }
        },
    )
    points = fetch_index_hourly("^DJI", "2026-05-01", "2026-05-01")
    assert [value for _stamp, value in points] == [41_000.0]


def test_fetch_raises_on_an_error_envelope_delivered_with_200(monkeypatch):
    # The failure mode the status code cannot see: HTTP 200, null result, and
    # the real problem sitting in chart.error.
    _stub_response(
        monkeypatch,
        {
            "chart": {
                "result": None,
                "error": {"code": "Unauthorized", "description": "Invalid Crumb"},
            }
        },
    )
    with pytest.raises(YahooChartError, match="Invalid Crumb"):
        fetch_index_hourly("^DJI", "2026-05-01", "2026-05-01")


def test_fetch_raises_when_the_body_has_no_chart_object(monkeypatch):
    # A consent/interstitial page. Previously `(… or {}).get("result") or []`
    # turned this into an empty window with no log line anywhere.
    _stub_response(monkeypatch, {"finance": {"result": None}})
    with pytest.raises(YahooChartError, match="no chart object"):
        fetch_index_hourly("^DJI", "2026-05-01", "2026-05-01")


def test_fetch_raises_on_a_null_result_without_an_error(monkeypatch):
    _stub_response(monkeypatch, {"chart": {"result": None, "error": None}})
    with pytest.raises(YahooChartError, match="null with no error"):
        fetch_index_hourly("^DJI", "2026-05-01", "2026-05-01")


def test_fetch_treats_a_well_formed_empty_result_as_absent(monkeypatch):
    # Well-formed and genuinely empty: NOT an error. This is the case that must
    # stay distinguishable from every raise above.
    _stub_response(monkeypatch, {"chart": {"result": [], "error": None}})
    assert fetch_index_hourly("^DJI", "2026-05-01", "2026-05-01") == []


def test_yahoo_chart_error_is_a_request_exception():
    # Load-bearing: equity_plot catches requests.RequestException, so this is
    # what routes a 200-delivered failure into the degrade-and-flag path.
    assert issubclass(YahooChartError, requests.RequestException)


# --------------------------------------------------------------------------
# NYSE early closes: Yahoo stamps the closing level at the close itself
# --------------------------------------------------------------------------
#
# The fixtures are real ^DJI and ^GSPC responses to the request
# fetch_index_hourly sends (period1 = the day's 00:00Z, period2 = +1 day,
# interval=1h), recorded 2026-10-15 and trimmed to ``meta``'s relevant fields
# and the closes (Yahoo serves 1h data for the last 730 days only, so the
# 2024 ones could not be recorded after late 2026). The trailing
# 2026-10-14T20:00Z point is Yahoo appending the latest close to every hourly
# response; the window filter drops it unless the window reaches the latest
# session. On 2025-11-28 (13:00 close) the 12:30 half bar comes back null and
# the closing level, 47,716.42, arrives stamped 13:00 (18:00Z); the full
# session of 2025-11-26 has no such point. Every NYSE early close from
# 2024-11-29 to 2025-12-24 has that shape on both indices (all five are
# pinned below), and Yahoo's own ``tradingPeriods`` gives 2025-11-28 a
# 14:30Z-18:00Z regular session.

_FIXTURES = Path(__file__).parent / "fixtures"
_NEW_YORK = ZoneInfo("America/New_York")


@pytest.fixture(autouse=True)
def _fresh_error_reports():
    """The module reports each condition once per process; each test starts clean."""
    from dashboard.backend.domain.leaderboard.strategies import _yahoo

    _yahoo._reported.clear()
    yield
    _yahoo._reported.clear()


def _recorded(name):
    return json.loads((_FIXTURES / name).read_text())


def _clock(points):
    return [(stamp.astimezone(_NEW_YORK).strftime("%H:%M"), value) for stamp, value in points]


def test_an_early_close_level_lands_in_the_last_half_bar(monkeypatch):
    _stub_response(monkeypatch, _recorded("yahoo-chart-dji-1h-2025-11-28.json"))
    points = fetch_index_hourly("^DJI", "2025-11-28", "2025-11-28")
    assert _clock(points) == [
        ("09:30", 47677.671875),
        ("10:30", 47699.78125),
        ("11:30", 47660.55859375),
        ("12:30", 47716.421875),
    ]


#: Each early close's published close, equal to Yahoo's own daily bar.
_OFFICIAL_CLOSES = {
    ("^DJI", "2024-11-29"): 44910.6484375,
    ("^GSPC", "2024-11-29"): 6032.3798828125,
    ("^DJI", "2024-12-24"): 43297.03125,
    ("^GSPC", "2024-12-24"): 6040.0400390625,
    ("^DJI", "2025-07-03"): 44828.53125,
    ("^GSPC", "2025-07-03"): 6279.35009765625,
    ("^DJI", "2025-11-28"): 47716.421875,
    ("^GSPC", "2025-11-28"): 6849.08984375,
    ("^DJI", "2025-12-24"): 48731.16015625,
    ("^GSPC", "2025-12-24"): 6932.0498046875,
}


@pytest.mark.parametrize(("symbol", "day"), sorted(_OFFICIAL_CLOSES))
def test_each_recorded_early_close_ends_on_the_official_close(
    monkeypatch, capsys, symbol, day
):
    fixture = f"yahoo-chart-{symbol[1:].lower()}-1h-{day}.json"
    official_close = _OFFICIAL_CLOSES[(symbol, day)]
    _stub_response(monkeypatch, _recorded(fixture))
    points = fetch_index_hourly(symbol, day, day)
    assert [clock for clock, _value in _clock(points)] == [
        "09:30", "10:30", "11:30", "12:30",
    ]
    assert points[-1][1] == official_close
    assert "ERROR" not in capsys.readouterr().out


def _without_the_close_point(payload):
    """The 2025-11-28 response as it would look if Yahoo dropped its 13:00 point."""
    result = payload["chart"]["result"][0]
    at = result["timestamp"].index(1764352800)  # 2025-11-28T18:00Z, 13:00 ET
    del result["timestamp"][at]
    del result["indicators"]["quote"][0]["close"][at]
    return payload


def test_a_missing_early_close_level_is_reported(monkeypatch, capsys):
    """Fail-visible: without the 13:00 point the day ends on its 11:30 bar and
    the board's 13:00 node repeats it, a plausible and wrong curve."""
    _stub_response(
        monkeypatch, _without_the_close_point(_recorded("yahoo-chart-dji-1h-2025-11-28.json"))
    )
    points = fetch_index_hourly("^DJI", "2025-11-28", "2025-11-28")
    assert [clock for clock, _value in _clock(points)] == ["09:30", "10:30", "11:30"]
    assert (
        "ERROR: yahoo.early_close_level_missing symbol=^DJI date=2025-11-28"
        in capsys.readouterr().out
    )


def test_a_missing_early_close_level_is_reported_once_per_process(monkeypatch, capsys):
    """Every plot render and refresh fetches again; one standing condition must
    not print one ERROR per call."""
    payload = _without_the_close_point(_recorded("yahoo-chart-dji-1h-2025-11-28.json"))
    _stub_response(monkeypatch, payload)
    fetch_index_hourly("^DJI", "2025-11-28", "2025-11-28")
    fetch_index_hourly("^DJI", "2025-11-28", "2025-11-28")
    assert capsys.readouterr().out.count("yahoo.early_close_level_missing") == 1


@pytest.mark.parametrize("meta", [{}, None, {"exchangeTimezoneName": ""}])
def test_a_missing_exchange_zone_is_reported_not_skipped_silently(monkeypatch, capsys, meta):
    """Absent is not "another exchange". Without the field the restamp and the
    missing-level check would both skip and every early close would end on its
    11:30 bar with nothing logged, so absence is an ERROR (once per symbol)."""
    payload = _recorded("yahoo-chart-dji-1h-2025-11-28.json")
    payload["chart"]["result"][0]["meta"] = meta
    _stub_response(monkeypatch, payload)
    points = fetch_index_hourly("^DJI", "2025-11-28", "2025-11-28")
    fetch_index_hourly("^DJI", "2025-11-28", "2025-11-28")
    assert _clock(points)[-1] == ("13:00", 47716.421875)
    out = capsys.readouterr().out
    assert out.count("ERROR: yahoo.exchange_zone_missing symbol=^DJI") == 1


def test_the_profiles_spelling_of_new_york_also_restamps(monkeypatch):
    payload = _recorded("yahoo-chart-dji-1h-2025-11-28.json")
    payload["chart"]["result"][0]["meta"]["exchangeTimezoneName"] = "US/Eastern"
    _stub_response(monkeypatch, payload)
    points = fetch_index_hourly("^DJI", "2025-11-28", "2025-11-28")
    assert _clock(points)[-1] == ("12:30", 47716.421875)


def test_an_early_close_within_its_settle_hour_is_not_reported(capsys):
    from dashboard.backend.domain.leaderboard.strategies._yahoo import (
        _report_missing_close_levels,
    )

    morning = [
        (dt.datetime(2025, 11, 28, 14 + i, 30, tzinfo=dt.timezone.utc), 1.0)
        for i in range(3)
    ]
    for hour, minute in ((13, 30), (13, 59)):
        _report_missing_close_levels(
            "^DJI", morning, "America/New_York",
            now=dt.datetime(2025, 11, 28, hour, minute, tzinfo=_NEW_YORK),
        )
    assert capsys.readouterr().out == ""
    _report_missing_close_levels(
        "^DJI", morning, "America/New_York",
        now=dt.datetime(2025, 11, 28, 14, 0, tzinfo=_NEW_YORK),
    )
    assert "date=2025-11-28" in capsys.readouterr().out


def test_the_early_close_level_is_the_live_boards_13_00_node(monkeypatch):
    """Read as an open-stamped bar, a point stamped 13:00 opens after the close
    and has no node, so the index curve's day would end on its 12:30 level."""
    from dashboard.backend.domain.leaderboard import live

    _stub_response(monkeypatch, _recorded("yahoo-chart-dji-1h-2025-11-28.json"))
    points = fetch_index_hourly("^DJI", "2025-11-28", "2025-11-28")
    nodes = {live._axis_node_key(stamp.astimezone(_NEW_YORK)): value for stamp, value in points}
    assert nodes == {
        "2025-11-28T10:00": 47677.671875,
        "2025-11-28T11:00": 47699.78125,
        "2025-11-28T12:00": 47660.55859375,
        "2025-11-28T13:00": 47716.421875,
    }


def test_a_full_session_is_left_as_yahoo_sent_it(monkeypatch):
    _stub_response(monkeypatch, _recorded("yahoo-chart-dji-1h-2025-11-26.json"))
    points = fetch_index_hourly("^DJI", "2025-11-26", "2025-11-26")
    assert [clock for clock, _value in _clock(points)] == [
        "09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30",
    ]
    assert points[-1][1] == 47419.671875


def test_yahoos_appended_close_is_not_restamped_on_a_full_session(monkeypatch):
    """Yahoo appends the latest close to every response, stamped at the close
    itself: 16:00 ET on a full session. The nightly refresh fetches through
    today, so the point is inside its window. Restamped, it would overwrite
    today's 15:30 half bar with the official close, and a later replay of the
    same day (the point gone by then) would mark it pre-auction again. Only an
    NYSE early close restamps; here the point stays at 16:00, where the board's
    node mapping drops it."""
    _stub_response(monkeypatch, _recorded("yahoo-chart-dji-1h-2025-11-26.json"))
    points = fetch_index_hourly("^DJI", "2025-11-26", "2026-10-14")
    assert points[-2][1] == 47419.671875  # 2025-11-26's 15:30 half bar
    assert points[-1] == (
        dt.datetime(2026, 10, 14, 20, tzinfo=dt.timezone.utc), 51621.48046875,
    )


def test_the_close_stamped_level_wins_over_a_half_bar_yahoo_also_filled(monkeypatch):
    payload = _recorded("yahoo-chart-dji-1h-2025-11-28.json")
    payload["chart"]["result"][0]["indicators"]["quote"][0]["close"][3] = 47700.0
    _stub_response(monkeypatch, payload)
    points = fetch_index_hourly("^DJI", "2025-11-28", "2025-11-28")
    assert _clock(points)[-1] == ("12:30", 47716.421875)
    assert len(points) == 4


def test_a_non_us_index_is_not_restamped_on_an_nyse_early_close(monkeypatch, capsys):
    payload = _recorded("yahoo-chart-dji-1h-2025-11-28.json")
    payload["chart"]["result"][0]["meta"]["exchangeTimezoneName"] = "Asia/Shanghai"
    _stub_response(monkeypatch, payload)
    points = fetch_index_hourly("^DJI", "2025-11-28", "2025-11-28")
    assert _clock(points)[-1] == ("13:00", 47716.421875)
    assert "ERROR" not in capsys.readouterr().out
