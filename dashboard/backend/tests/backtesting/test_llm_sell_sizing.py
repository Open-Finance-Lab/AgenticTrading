"""An LLM SELL sells what it asked for (#661).

The prompt asks the model for the number of shares to sell, and the sell
branch used to ignore it and close the whole position, so every trim became a
full exit. A missing, null or zero size still means "sell everything"; a
``size_fraction`` (what a pipeline risk action sends) sells that share of the
position. A size the gate cannot read is refused and recorded on the order
ledger, never read as "sell everything".
"""

import json

import pytest

from dashboard.backend.domain.backtesting.portfolio_manager import (
    SELL_SIZE_REFUSED_REASON,
    PortfolioManager,
)
from dashboard.backend.infrastructure.llm.validator import (
    LLMTradingDecision,
    actions_to_executable,
    requested_sell_shares,
)
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


def _assert_refused(pm, requested=0):
    """The skipped SELL left a rejection on the ledger, as an executor
    refusal does, and sold nothing."""
    [event] = pm.order_events
    assert (event["symbol"], event["side"], event["status"]) == ("AAPL", "SELL", "rejected")
    assert event["reason"] == SELL_SIZE_REFUSED_REASON
    assert event["executed_shares"] == 0
    assert event["requested_shares"] == requested
    assert event["strategy_reason"].startswith("[LLM] x (confidence")
    [rejection] = pm.rejected_orders
    assert (rejection["action"], rejection["reason"]) == ("sell", SELL_SIZE_REFUSED_REASON)
    assert pm.positions["AAPL"] == 500


def test_a_sell_of_10_sells_10_of_500():
    actions = _sell(_holding(500), position_size=10)
    assert [(a["action"], a["shares"]) for a in actions] == [("sell", 10)]
    assert "requested_shares" not in actions[0]


def test_a_numeric_string_size_is_read_as_its_number():
    assert [a["shares"] for a in _sell(_holding(500), position_size="10")] == [10]


# A blank string is a missing value everywhere a model's number is read
# (``model_number``): a pipeline risk action's ``size_pct: ""`` is a full exit
# too (test_pipeline_runner), and a ``confidence: ""`` takes the default.
@pytest.mark.parametrize("size", [_MISSING, 0, None, 0.0, "", "  "])
def test_no_size_still_sells_the_whole_position(size):
    pm = _holding(500)
    actions = _sell(pm, position_size=size)
    assert [a["shares"] for a in actions] == [500]
    assert pm.order_events == []


def test_a_sell_larger_than_the_position_is_capped_and_says_so():
    actions = _sell(_holding(500), position_size=900)
    assert [a["shares"] for a in actions] == [500]
    assert actions[0]["requested_shares"] == 900


@pytest.mark.parametrize(
    ("size", "recorded"),
    [
        ("ten", 0),
        ("all", 0),
        (float("inf"), 0),
        (float("nan"), 0),
        (-5, -5),
        # int(-0.5) is 0, and 0 means the whole position: the sign must be
        # read before anything truncates it.
        (-0.5, -0.5),
        (False, 0),
        (True, 0),
        ([], 0),
        ({}, 0),
        (10**400, 0),
    ],
)
def test_an_unusable_size_is_refused_and_recorded_never_sold_whole(size, recorded, capsys):
    pm = _holding(500)
    assert _sell(pm, position_size=size) == []
    assert "is not a share count" in capsys.readouterr().out
    _assert_refused(pm, requested=recorded)


@pytest.mark.parametrize("size", [0.5, 0.9])
def test_a_size_under_one_share_is_refused_not_read_as_sell_all(size, capsys):
    """int() truncates 0.5 to 0, and 0 means the whole position: a model
    asking to sell half a share must not liquidate."""
    pm = _holding(500)
    assert _sell(pm, position_size=size) == []
    assert "is less than one share" in capsys.readouterr().out
    _assert_refused(pm, requested=size)


def test_a_sell_of_an_unheld_symbol_is_skipped_as_before(capsys):
    pm = PortfolioManager(100000)
    assert _sell(pm, position_size="ten") == []
    assert "Skip (not in portfolio" in capsys.readouterr().out
    assert pm.order_events == []


@pytest.mark.parametrize(
    ("fraction", "shares"),
    [(0.25, 125), (1.0, 500), ("0.5", 250), (None, 500), (100, 500), (150, 500), (1e307, 500)],
)
def test_a_size_fraction_sells_that_share_of_the_position(fraction, shares):
    """1 is the whole position, and so is 100 or more: read as a fraction or
    as a percentage, each of those means all of it."""
    actions = _sell(_holding(500), position_size=0, size_fraction=fraction)
    assert [a["shares"] for a in actions] == [shares]


@pytest.mark.parametrize("fraction", [1.5, 25, 50, "25", 99.99])
def test_a_size_fraction_between_1_and_100_is_refused_not_sold_whole(fraction, capsys):
    """No prompt defined size_pct's unit, so a model writing 25 for 25% got a
    full exit (#690). Between 1 and 100 a fraction and a percentage disagree,
    so the gate refuses and records it rather than guessing; reading it as a
    percentage would silently change what 1.5 did."""
    pm = _holding(500)
    assert _sell(pm, position_size=0, size_fraction=fraction) == []
    assert "is between 1 and 100" in capsys.readouterr().out
    _assert_refused(pm)


def test_a_share_count_is_not_refused_for_a_size_fraction_beside_it():
    """The share count is the size; the fraction beside it is never read."""
    actions = _sell(_holding(500), position_size=10, size_fraction=25)
    assert [a["shares"] for a in actions] == [10]


@pytest.mark.parametrize(("fraction", "held", "shares"), [(0.57, 100, 57), (0.29, 100, 29)])
def test_a_size_fraction_is_not_shaved_by_float_error(fraction, held, shares):
    """0.57 * 100 is 56.99999999999999 in binary floating point."""
    assert fraction * held < shares  # the trap this guards against
    actions = _sell(_holding(held), position_size=0, size_fraction=fraction)
    assert [a["shares"] for a in actions] == [shares]


@pytest.mark.parametrize(
    "fraction", [0, -0.5, "half", True, float("nan"), float("-inf"), [], 10**400]
)
def test_an_unusable_size_fraction_is_refused_not_read_as_the_whole_position(fraction, capsys):
    """Neither a full exit nor an OverflowError that throws the whole bar into
    the fallback: a 400-digit fraction is one refused action."""
    pm = _holding(500)
    assert _sell(pm, position_size=0, size_fraction=fraction) == []
    assert "is not a share of the position" in capsys.readouterr().out
    _assert_refused(pm)
    assert pm.llm_decisions == 1  # the model still drove the bar


def test_a_size_fraction_rounds_down_to_whole_lots_on_ashares():
    actions = _sell(_holding(500, lot_size=100), position_size=0, size_fraction=0.25)
    assert [a["shares"] for a in actions] == [100]


def test_a_trim_below_one_lot_is_refused(capsys):
    """Recorded with the 50 shares it asked for: the risk branch sends no
    share count, so recording ``position_size`` logged a well-formed trim as
    a request for 0."""
    pm = _holding(500, lot_size=100)
    assert _sell(pm, position_size=0, size_fraction=0.1) == []
    assert "is less than one lot of 100" in capsys.readouterr().out
    _assert_refused(pm, requested=50)


def test_a_trim_below_one_share_records_the_part_it_asked_for(capsys):
    pm = _holding(3)
    assert _sell(pm, position_size=0, size_fraction=0.25) == []
    assert "is less than one share" in capsys.readouterr().out
    [event] = pm.order_events
    assert event["requested_shares"] == 0.75


def test_a_refusal_that_repeats_on_one_day_collapses_like_an_executor_one():
    pm = _holding(500)
    _sell(pm, position_size="all")
    _sell(pm, position_size="all")
    [event] = pm.order_events
    assert event["repeat_count"] == 2
    assert len(pm.rejected_orders) == 2


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


def test_a_lot_market_passes_a_share_count_on_as_written():
    """The executor's lot check owns an off-lot count, as it does for a BUY."""
    assert requested_sell_shares(150, None, 500, lot_size=100) == (150, None)
    assert requested_sell_shares(0.5, None, 500, lot_size=100) == (0.5, None)


@pytest.mark.parametrize(("size", "held", "sold"), [(10, 500, 10), (0, 500, 500), (900, 500, 500)])
def test_external_agents_size_a_sell_by_the_same_rule(size, held, sold):
    """``actions_to_executable`` reads a SELL through ``requested_sell_shares``
    too, so the protocol/v2 path and the gate cannot drift apart again."""
    decision = LLMTradingDecision(
        symbol="AAPL", action="sell", confidence=0.9,
        reasoning="trim the position", position_size=size,
    )
    [action] = actions_to_executable(
        [decision], cash=0, positions={"AAPL": held}, current_prices={"AAPL": 100.0}
    )
    assert action["shares"] == sold
    shares, refusal = requested_sell_shares(size, None, held)
    assert refusal is None and min(shares, held) == sold
