"""The Run Backtest modal defaults to the latest settled US trading week."""

from datetime import date, datetime, timedelta
from itertools import islice

import pytest
import pytz
from fastapi.testclient import TestClient

from dashboard.backend.infrastructure.market_data import bar_cache, provider
from dashboard.backend.infrastructure.market_data.us_market_calendar import (
    trading_weeks_back,
)

_NY = pytz.timezone("America/New_York")
_SHANGHAI = pytz.timezone("Asia/Shanghai")


def _at(zone, text: str) -> float:
    return zone.localize(datetime.fromisoformat(text)).timestamp()


@pytest.mark.parametrize(
    "day, expected",
    [
        # The week holding `day` comes first, still-trading or not; finishing
        # is the caller's judgement, not the calendar's.
        (date(2026, 10, 7), [(date(2026, 10, 5), date(2026, 10, 9)),
                             (date(2026, 9, 28), date(2026, 10, 2))]),
        # A weekend day belongs to the Mon-Fri before it.
        (date(2026, 10, 11), [(date(2026, 10, 5), date(2026, 10, 9)),
                              (date(2026, 9, 28), date(2026, 10, 2))]),
        # Holidays trim the ends: Good Friday 2026-04-03, Labor Day 2026-09-07.
        (date(2026, 4, 8), [(date(2026, 4, 6), date(2026, 4, 10)),
                            (date(2026, 3, 30), date(2026, 4, 2))]),
        (date(2026, 9, 16), [(date(2026, 9, 14), date(2026, 9, 18)),
                             (date(2026, 9, 8), date(2026, 9, 11))]),
    ],
)
def test_trading_weeks_back(day, expected):
    assert list(islice(trading_weeks_back(day), 2)) == expected


@pytest.mark.parametrize(
    "now, expected",
    [
        # Mid-week: this week is still trading.
        (_at(_NY, "2026-10-07T12:00"), ("2026-09-28", "2026-10-02")),
        # Friday after the close, and Saturday daytime: the week has closed but
        # the bar cache would refuse it (provider end Sat 00:00Z, under a day
        # old), so the default stays on the week it can store.
        (_at(_NY, "2026-10-09T17:00"), ("2026-09-28", "2026-10-02")),
        (_at(_NY, "2026-10-10T10:00"), ("2026-09-28", "2026-10-02")),
        (_at(_NY, "2026-10-10T19:59"), ("2026-09-28", "2026-10-02")),
        # Saturday 20:00 EDT = Sunday 00:00Z: settled, so it rolls.
        (_at(_NY, "2026-10-10T20:00"), ("2026-10-05", "2026-10-09")),
        (_at(_NY, "2026-10-12T09:00"), ("2026-10-05", "2026-10-09")),
        # In winter the same instant is 19:00 EST.
        (_at(_NY, "2026-12-05T18:59"), ("2026-11-23", "2026-11-27")),
        (_at(_NY, "2026-12-05T19:00"), ("2026-11-30", "2026-12-04")),
        # A week ending Thursday (Good Friday) rolls a day earlier.
        (_at(_NY, "2026-04-03T21:00"), ("2026-03-30", "2026-04-02")),
        # Saturday morning in Shanghai is still Friday in New York.
        (_at(_SHANGHAI, "2026-10-10T09:00"), ("2026-09-28", "2026-10-02")),
    ],
)
def test_default_backtest_window(now, expected):
    assert provider.default_backtest_window(now) == expected


def test_the_default_is_always_a_window_the_cache_will_store():
    """The invariant the rollover rule exists for, swept hourly over five
    weeks spanning both DST transitions' neighbourhoods and a holiday: the
    default is settled, and it is the NEWEST settled week -- the one after it
    is not yet storable."""
    start = _at(_NY, "2026-10-26T00:00")
    for hour in range(24 * 7 * 5):
        now = start + hour * 3600
        first, last = provider.default_backtest_window(now)
        assert bar_cache.window_is_settled(provider.exclusive_end(last), now=now)
        next_week = list(
            islice(trading_weeks_back(date.fromisoformat(first) + timedelta(days=7)), 1)
        )[0]
        assert not bar_cache.window_is_settled(
            provider.exclusive_end(next_week[1].isoformat()), now=now
        ), (hour, first, last)


def test_the_default_reads_the_wall_clock(monkeypatch):
    """Without `now` the window comes from `time.time()` -- an absolute
    instant -- never from a host-local or market-local date."""
    monkeypatch.setattr(provider.time, "time", lambda: _at(_NY, "2026-10-10T10:00"))
    assert provider.default_backtest_window() == ("2026-09-28", "2026-10-02")
    monkeypatch.setattr(provider.time, "time", lambda: _at(_NY, "2026-10-10T20:00"))
    assert provider.default_backtest_window() == ("2026-10-05", "2026-10-09")


def test_config_defaults_serves_the_rolling_window(monkeypatch):
    from dashboard.backend.app import app
    from dashboard.backend.api.routers import config

    monkeypatch.setattr(
        config, "default_backtest_window", lambda: ("2026-09-28", "2026-10-02")
    )
    settings = TestClient(app).get("/config/defaults").json()["defaultSettings"]
    assert (settings["startDate"], settings["endDate"]) == ("2026-09-28", "2026-10-02")
    # The rest of the file still reaches the modal, and the description keeps
    # the universe it describes alongside the new window.
    assert "AAPL" in settings["assetList"]
    assert settings["description"] == (
        "Magnificent 7 over the latest complete trading week, 2026-09-28 to 2026-10-02"
    )
