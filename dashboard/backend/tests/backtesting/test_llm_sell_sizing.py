"""An LLM SELL sells what it asked for (#661).

The prompt asks the model for the number of shares to sell, and the sell
branch used to ignore it and close the whole position, so every trim became a
full exit. A missing or zero size still means "sell everything"; a pipeline
risk action's ``size_fraction`` sells that share of the position.
"""

import json

import pytest

from dashboard.backend.domain.backtesting.portfolio_manager import PortfolioManager
from dashboard.backend.tests.backtesting.test_portfolio_manager_move import (
    _FakeClient,
    _FakeResp,
    _FakeUsage,
    _llm_state,
)

_MISSING = object()


def _sell(pm, **fields):
    action = {"symbol": "AAPL", "action": "sell", "confidence": 0.9, "reasoning": "x"}
    action.update({k: v for k, v in fields.items() if v is not _MISSING})
    text = json.dumps({"actions": [action]})
    return pm.make_trading_decision_with_llm(
        _llm_state(), _FakeClient(_FakeResp(text, _FakeUsage(1, 1)))
    )["actions"]


def _holding(shares, **kwargs):
    pm = PortfolioManager(100000, **kwargs)
    pm.positions["AAPL"] = shares
    pm.entry_prices["AAPL"] = 100.0
    return pm


def test_a_sell_of_10_sells_10_of_500():
    actions = _sell(_holding(500), position_size=10)
    assert [(a["action"], a["shares"]) for a in actions] == [("sell", 10)]
    assert "requested_shares" not in actions[0]


@pytest.mark.parametrize("size", [_MISSING, 0, None])
def test_no_size_still_sells_the_whole_position(size):
    actions = _sell(_holding(500), position_size=size)
    assert [a["shares"] for a in actions] == [500]


def test_a_sell_larger_than_the_position_is_capped_and_says_so():
    actions = _sell(_holding(500), position_size=900)
    assert [a["shares"] for a in actions] == [500]
    assert actions[0]["requested_shares"] == 900


@pytest.mark.parametrize("size", ["ten", float("inf")])
def test_an_unparseable_size_skips_the_sell(size, capsys):
    assert _sell(_holding(500), position_size=size) == []
    assert "SELL AAPL: Skip (unparseable position_size" in capsys.readouterr().out


def test_a_negative_size_skips_the_sell_and_says_why(capsys):
    assert _sell(_holding(500), position_size=-5) == []
    assert "SELL AAPL: Skip (negative position_size -5)" in capsys.readouterr().out


@pytest.mark.parametrize("size", [0.5, 0.9])
def test_a_size_under_one_share_is_skipped_not_read_as_sell_all(size, capsys):
    """int() truncates 0.5 to 0, and 0 means the whole position: a model
    asking to sell half a share must not liquidate."""
    assert _sell(_holding(500), position_size=size) == []
    assert "rounds to 0 shares" in capsys.readouterr().out


@pytest.mark.parametrize(("fraction", "shares"), [(0.25, 125), (1.0, 500), (1.5, 500)])
def test_a_size_fraction_sells_that_share_of_the_position(fraction, shares):
    actions = _sell(_holding(500), position_size=0, size_fraction=fraction)
    assert [a["shares"] for a in actions] == [shares]


@pytest.mark.parametrize("fraction", [0, -0.5, "half", None])
def test_an_unusable_size_fraction_falls_back_to_the_whole_position(fraction):
    actions = _sell(_holding(500), position_size=0, size_fraction=fraction)
    assert [a["shares"] for a in actions] == [500]


def test_a_size_fraction_rounds_down_to_whole_lots_on_ashares():
    actions = _sell(_holding(500, lot_size=100), position_size=0, size_fraction=0.25)
    assert [a["shares"] for a in actions] == [100]


def test_a_trim_below_one_lot_is_skipped(capsys):
    assert _sell(_holding(500, lot_size=100), position_size=0, size_fraction=0.1) == []
    assert "SELL AAPL: Skip" in capsys.readouterr().out


def test_t_plus_one_caps_a_partial_sell_and_records_the_deferral():
    pm = _holding(500, t_plus_one_enabled=True)
    pm.available_positions["AAPL"] = 200
    pm.frozen_lots["AAPL"] = [{"buy_date": "2026-01-01", "shares": 300}]

    actions = _sell(pm, position_size=300)

    assert [a["shares"] for a in actions] == [200]
    assert actions[0]["requested_shares"] == 300
    deferral = pm.t1_deferrals[("AAPL", "2026-01-01")]
    assert deferral["requested_shares"] == 300
    assert deferral["deferred_shares"] == 100


def test_t_plus_one_leaves_a_sell_within_the_sellable_shares_alone():
    pm = _holding(500, t_plus_one_enabled=True)
    pm.available_positions["AAPL"] = 200
    pm.frozen_lots["AAPL"] = [{"buy_date": "2026-01-01", "shares": 300}]

    actions = _sell(pm, position_size=50)

    assert [a["shares"] for a in actions] == [50]
    assert pm.t1_deferrals == {}
