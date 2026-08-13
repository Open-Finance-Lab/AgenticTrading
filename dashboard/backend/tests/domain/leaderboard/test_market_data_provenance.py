"""A published leaderboard curve must record which tape priced it.

The loader stamps the feed on the dataframes it returns, but frames are
transient and ``agent_runs`` is not: without persisting the stamp, a curve
computed off an IEX fallback is byte-identical in the database to a full-tape
SIP one, and the board would rank them against each other. These tests pin the
stamp all the way into the stored run metadata.
"""

import pandas as pd
import pytest

import dashboard.backend.domain.leaderboard.service as service
from dashboard.backend.database import db
from dashboard.backend.infrastructure.market_data.alpaca_bars import (
    FRAME_ATTR_END_CLAMPED,
    FRAME_ATTR_FEED,
    FRAME_ATTR_SIP_FALLBACK,
)

SESSION = "provenance-test-session"
START = "2026-04-01"
END = "2026-04-02"


def _stamped_bars(feed="sip", fallback=False, clamped=False):
    frame = pd.DataFrame(
        {"open": [10.0], "high": [11.0], "low": [9.0], "close": [10.5], "volume": [100]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-04-01 14:00")], name="timestamp"),
    )
    frame.attrs[FRAME_ATTR_FEED] = feed
    frame.attrs[FRAME_ATTR_SIP_FALLBACK] = fallback
    frame.attrs[FRAME_ATTR_END_CLAMPED] = clamped
    return {"AAPL": frame}


class _StubStrategy:
    """Cheap stand-in for a baseline: one symbol, a two-point curve."""

    llm_calls = 0
    llm_decisions = 0
    decision_steps = 0
    model_id = None

    def required_symbols(self):
        return ["AAPL"]

    def run(self, bars, start_date, end_date, initial_capital):
        return [
            {
                "timestamp": f"{day}T14:00:00+00:00",
                "equity": initial_capital * factor,
                "cash": 0.0,
                "positions_value": initial_capital * factor,
            }
            for day, factor in ((start_date, 1.0), (end_date, 1.1))
        ]

    def num_trades(self):
        return 1


@pytest.fixture
def leaderboard_config():
    return {
        "session_id": SESSION,
        "start_date": START,
        "end_date": END,
        "initial_capital": 100_000,
        "period": "contest",
        "strategies": [
            {"id": "buy_hold", "name": "Buy & Hold", "strategy": "buy_hold"}
        ],
    }


@pytest.fixture(autouse=True)
def _stub_strategy(monkeypatch):
    monkeypatch.setattr(service, "get_strategy", lambda strategy: _StubStrategy())


def _stored_metadata(strategy_id="buy_hold"):
    run_id = service._run_id(strategy_id, START, END)
    for run in db.get_runs_by_session(SESSION) or []:
        if run.get("run_id") == run_id:
            return run.get("metadata")
    return None


def test_published_baseline_records_its_feed(monkeypatch, leaderboard_config):
    monkeypatch.setattr(
        service, "fetch_hourly_bars", lambda *a, **k: _stamped_bars(feed="sip")
    )

    service.ensure_leaderboard_runs(force_refresh=True, config=leaderboard_config)

    metadata = _stored_metadata()
    assert metadata is not None, "baseline runs must carry provenance, not None"
    assert metadata["market_data_feed"] == "sip"
    assert metadata["sip_fallback_to_iex"] is False


def test_iex_fallback_is_distinguishable_in_the_database(
    monkeypatch, leaderboard_config
):
    """The whole point: a fallback curve must not look like a SIP curve."""
    monkeypatch.setattr(
        service,
        "fetch_hourly_bars",
        lambda *a, **k: _stamped_bars(feed="iex", fallback=True, clamped=True),
    )

    service.ensure_leaderboard_runs(force_refresh=True, config=leaderboard_config)

    metadata = _stored_metadata()
    assert metadata["market_data_feed"] == "iex"
    assert metadata["sip_fallback_to_iex"] is True
    assert metadata["end_clamped"] is True


def test_provenance_merge_keeps_llm_config_snapshot():
    merged = service._with_market_data_provenance(
        {"entry_id": "claude", "model_id": "claude-x"},
        {"market_data_feed": "sip", "sip_fallback_to_iex": False, "end_clamped": False},
    )
    assert merged["entry_id"] == "claude"
    assert merged["market_data_feed"] == "sip"

    # No Alpaca involvement (e.g. a Yahoo index line) leaves metadata untouched.
    assert service._with_market_data_provenance(None, None) is None
    assert service._with_market_data_provenance({"entry_id": "x"}, None) == {
        "entry_id": "x"
    }


def test_feed_drift_warns_but_does_not_refetch(capsys):
    """A mismatch is reported, never auto-refreshed.

    ``ensure_leaderboard_runs`` runs on a public unauthenticated GET; treating
    a feed mismatch as "missing" would hit Alpaca on every page load for as
    long as the mismatch lasts.
    """
    runs = [
        {"metadata": {"market_data_feed": "iex"}},
        {"metadata": {"market_data_feed": "sip"}},
        {"metadata": None},
    ]
    service._warned_feed_drift.clear()
    service._warn_on_feed_drift(runs, "sip")
    out = capsys.readouterr().out
    assert "iex" in out
    assert "force-refresh" in out

    # Repeats stay quiet: this runs on the hot path of a public endpoint.
    service._warn_on_feed_drift(runs, "sip")
    assert capsys.readouterr().out == ""

    service._warned_feed_drift.clear()
    service._warn_on_feed_drift(runs, None)
    assert "iex" not in capsys.readouterr().out


def test_unusable_feed_env_does_not_500_a_cached_board(monkeypatch, capsys):
    monkeypatch.setenv("ALPACA_DATA_FEED", "IEXX")
    assert service._configured_feed_or_none() is None
    assert "WARNING" in capsys.readouterr().out
