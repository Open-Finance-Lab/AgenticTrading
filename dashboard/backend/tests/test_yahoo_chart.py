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
from types import SimpleNamespace

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
