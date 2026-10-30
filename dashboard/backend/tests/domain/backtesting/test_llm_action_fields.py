"""The trading loop reads the model's action the way the strict check does (#658).

``"buy "`` passed the strict check, which strips, then matched neither the
buy nor the sell branch, which compared it raw: no trade, yet the step was
counted in ``llm_decisions``. Both now read it with ``normalize_action``, and
an action neither one recognises is refused or skipped, never counted as a
hold. The fields beside it must not abort the step either.
"""

import json
from datetime import datetime

import pytest

from dashboard.backend.domain.backtesting import portfolio_manager as pm
from dashboard.backend.domain.backtesting.portfolio_manager import PortfolioManager


class _Resp:
    def __init__(self, text):
        self.text = text


def _state():
    return {
        "timestamp": datetime(2026, 4, 1, 15, 30),
        "cash": 1000.0,
        "positions": [],
        "positions_value": 0.0,
        "total_equity": 1000.0,
        "market_signals": {
            "AAPL": {"price": 250.0, "rsi": 50.0, "macd": 0.0,
                     "macd_signal": 0.0, "sma20": 250.0, "sma50": 250.0,
                     "bb_upper": 260.0, "bb_lower": 240.0, "volume": 1000},
        },
    }


@pytest.mark.parametrize("strict", [True, False], ids=["strict", "lenient"])
@pytest.mark.parametrize("action", ["buy ", "  BUY\n"])
def test_a_padded_buy_is_placed(monkeypatch, strict, action):
    reply = json.dumps({"actions": [{
        "symbol": "AAPL", "action": action, "position_size": 1,
        "confidence": 0.8, "reasoning": "RSI recovering",
    }]})
    monkeypatch.setattr(pm, "_request_trading_decision", lambda *_a, **_k: _Resp(reply))
    monkeypatch.setattr(pm, "_extract_response_text", lambda response: response.text)
    monkeypatch.setattr(pm, "_extract_token_usage", lambda response: (100, 50))
    manager = PortfolioManager(initial_capital=1000.0, allowed_symbols=["AAPL"])

    decision = manager.make_trading_decision_with_llm(
        _state(), llm_client=object(), model="m", strict_llm=strict
    )

    assert [(a["symbol"], a["action"], a["shares"]) for a in decision["actions"]] == [
        ("AAPL", "buy", 1)
    ]
    assert manager.llm_decisions == 1


def _decide(monkeypatch, actions, *, strict, symbols=("AAPL",)):
    reply = json.dumps({"actions": actions})
    monkeypatch.setattr(pm, "_request_trading_decision", lambda *_a, **_k: _Resp(reply))
    monkeypatch.setattr(pm, "_extract_response_text", lambda response: response.text)
    monkeypatch.setattr(pm, "_extract_token_usage", lambda response: (100, 50))
    manager = PortfolioManager(initial_capital=1000.0, allowed_symbols=list(symbols))
    decision = manager.make_trading_decision_with_llm(
        _state(), llm_client=object(), model="m", strict_llm=strict
    )
    return manager, [(a["symbol"], a["action"], a["shares"]) for a in decision["actions"]]


@pytest.mark.parametrize("strict", [True, False], ids=["strict", "lenient"])
@pytest.mark.parametrize("reasoning", [None, 7, ["a", "b"]])
def test_a_reasoning_that_is_not_text_does_not_abort_the_step(monkeypatch, strict, reasoning):
    # None[:60] raised TypeError: a billed run re-raised it and aborted, a
    # lenient one swapped in the rule-based decision.
    manager, placed = _decide(monkeypatch, [{
        "symbol": "AAPL", "action": "buy", "position_size": 1, "reasoning": reasoning,
    }], strict=strict)
    assert placed == [("AAPL", "buy", 1)]
    assert manager.llm_decisions == 1


@pytest.mark.parametrize("bad", [None, 1, ["buy"]])
def test_lenient_bad_action_is_skipped_not_the_whole_batch(monkeypatch, bad):
    # null.strip() raised AttributeError, which threw away the valid AAPL buy
    # and traded the bar rule-based. A null action holds, as a missing one
    # does; a number or a list is an unrecognised action and is skipped.
    manager, placed = _decide(monkeypatch, [
        {"symbol": "AAPL", "action": "buy", "position_size": 1},
        {"symbol": "MSFT", "action": bad},
    ], strict=False, symbols=("AAPL", "MSFT"))
    assert placed == [("AAPL", "buy", 1)]
    assert manager.llm_decisions == 1


@pytest.mark.parametrize("bad", [None, "", "BUY_MORE", "Buy.", "long", "buy "])
def test_strict_refuses_an_unrecognised_action(monkeypatch, bad):
    # No strike budget is set here, so the first refusal is fatal.
    with pytest.raises(pm.LLMDecisionError, match="invalid action batch"):
        _decide(monkeypatch, [{"symbol": "AAPL", "action": bad}], strict=True)


@pytest.mark.parametrize("bad", ["BUY_MORE", "Buy.", "long", "buy "])
def test_lenient_batch_with_no_recognised_action_earns_no_h6_credit(monkeypatch, bad):
    # It placed nothing yet counted as a step the model drove, so a curve the
    # model never steered could clear the leaderboard's H6 coverage guard.
    manager, placed = _decide(monkeypatch, [{"symbol": "AAPL", "action": bad}], strict=False)
    assert placed == []
    assert manager.llm_decisions == 0


def test_lenient_unrecognised_action_beside_a_recognised_one_still_counts(monkeypatch):
    manager, placed = _decide(monkeypatch, [
        {"symbol": "AAPL", "action": "hold"},
        {"symbol": "MSFT", "action": "BUY_MORE"},
    ], strict=False, symbols=("AAPL", "MSFT"))
    assert placed == []
    assert manager.llm_decisions == 1
