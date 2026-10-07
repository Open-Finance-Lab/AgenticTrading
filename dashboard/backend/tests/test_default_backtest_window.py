"""The Run Backtest modal defaults to the latest complete US trading week."""

from datetime import date

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.domain.leaderboard.us_market_calendar import latest_complete_week
from dashboard.backend.infrastructure.market_data import provider


@pytest.mark.parametrize(
    "today, expected",
    [
        # Mid-week: this week is still trading, so last Mon-Fri.
        (date(2026, 10, 7), (date(2026, 9, 28), date(2026, 10, 2))),
        # Monday and Friday are both mid-week too.
        (date(2026, 10, 5), (date(2026, 9, 28), date(2026, 10, 2))),
        (date(2026, 10, 9), (date(2026, 9, 28), date(2026, 10, 2))),
        # Weekend: the week that just closed.
        (date(2026, 10, 10), (date(2026, 10, 5), date(2026, 10, 9))),
        (date(2026, 10, 11), (date(2026, 10, 5), date(2026, 10, 9))),
        # Holidays trim the ends: Good Friday 2026-04-03, Labor Day 2026-09-07.
        (date(2026, 4, 8), (date(2026, 3, 30), date(2026, 4, 2))),
        (date(2026, 9, 16), (date(2026, 9, 8), date(2026, 9, 11))),
    ],
)
def test_latest_complete_week(today, expected):
    assert latest_complete_week(today) == expected


def test_the_window_reads_the_market_clock(monkeypatch):
    """Saturday morning in Shanghai is still Friday in New York: the week is
    not over, so the default must not jump to it."""
    monkeypatch.setattr(provider, "market_today", lambda market=None: date(2026, 10, 9))
    assert provider.default_backtest_window() == ("2026-09-28", "2026-10-02")


def test_config_defaults_serves_the_rolling_window(monkeypatch):
    from dashboard.backend.app import app
    from dashboard.backend.api.routers import config

    monkeypatch.setattr(
        config, "default_backtest_window", lambda: ("2026-09-28", "2026-10-02")
    )
    settings = TestClient(app).get("/config/defaults").json()["defaultSettings"]
    assert (settings["startDate"], settings["endDate"]) == ("2026-09-28", "2026-10-02")
    # The rest of the file still reaches the modal.
    assert "AAPL" in settings["assetList"]
