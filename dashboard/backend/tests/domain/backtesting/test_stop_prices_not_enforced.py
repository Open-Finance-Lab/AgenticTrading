"""The legacy engine does not execute stop-loss or take-profit prices (D12).

The safe-trading prompt's JSON shape carries ``stop_loss_price`` and
``take_profit_price`` (infrastructure/llm/validator.py), so a model can write
them, but the single-prompt decision loop that both boards' LLM entries run
(``PortfolioManager.make_trading_decision_with_llm``, reached from
``LLMAgentStrategy``) copies only symbol, action, size, confidence and reasoning
out of each reply. Both boards say so in their Rules and About cards
(dashboard/frontend/app.html). These tests are the other half of that copy: the
day this path starts reading the fields, they fail, and the copy changes in the
same PR. The new core executes stops for real (direction spec 2026-10-14 §4);
when it lands, invert these tests and delete the four disclaimers together.
"""

import json
from datetime import datetime
from pathlib import Path

from dashboard.backend.domain.backtesting import portfolio_manager as pm
from dashboard.backend.domain.backtesting.portfolio_manager import PortfolioManager

_BACKEND = Path(__file__).resolve().parents[3]
_STOP_FIELDS = {"stop_loss_price": 240.0, "take_profit_price": 300.0}


class _Resp:
    def __init__(self, text):
        self._text = text
        self.usage = None
        self.content = []


def _script(monkeypatch, replies):
    """Answer each decision request with the next reply, in order."""
    queue = [json.dumps(reply) for reply in replies]

    def fake_request(client, **_kwargs):
        return _Resp(queue.pop(0))

    monkeypatch.setattr(pm, "_request_trading_decision", fake_request)
    monkeypatch.setattr(pm, "_extract_response_text", lambda response: response._text)
    monkeypatch.setattr(pm, "_extract_token_usage", lambda response: (100, 50))
    return queue


def _state(manager, price, timestamp):
    held = manager.positions.get("AAPL", 0)
    positions = []
    if held:
        entry = manager.entry_prices.get("AAPL", price)
        positions.append({
            "symbol": "AAPL",
            "shares": held,
            "entry_price": entry,
            "current_price": price,
            "position_value": held * price,
            "pnl_pct": (price / entry - 1.0) * 100.0,
        })
    return {
        "timestamp": timestamp,
        "cash": manager.cash,
        "positions": positions,
        "positions_value": held * price,
        "total_equity": manager.cash + held * price,
        "market_signals": {
            "AAPL": {"price": price, "rsi": 50.0, "macd": 0.0,
                     "macd_signal": 0.0, "sma_20": price, "sma_50": price,
                     "bb_upper": price * 1.04, "bb_lower": price * 0.96,
                     "volume": 1000},
        },
    }


def _reply(action, **fields):
    return {"actions": [{
        "symbol": "AAPL",
        "action": action,
        "confidence": 0.8,
        "reasoning": "scripted",
        **_STOP_FIELDS,
        **fields,
    }]}


def _decide(manager, price, timestamp):
    return manager.make_trading_decision_with_llm(
        _state(manager, price, timestamp), llm_client=object(), model="m"
    )


def test_stop_and_take_profit_fields_never_reach_the_executed_action(monkeypatch):
    _script(monkeypatch, [_reply("buy", position_size=2)])
    manager = PortfolioManager(initial_capital=1000.0, allowed_symbols=["AAPL"])

    decision = _decide(manager, 250.0, datetime(2026, 4, 1, 14, 30))

    assert manager.llm_decisions == 1, "the model drove this step"
    assert len(decision["actions"]) == 1
    action = decision["actions"][0]
    assert set(action) == {"symbol", "action", "shares", "reason", "confidence"}
    assert (action["symbol"], action["action"], action["shares"]) == ("AAPL", "buy", 2)


def test_a_held_position_is_not_sold_when_the_price_crosses_its_stop_or_target(monkeypatch):
    remaining = _script(monkeypatch, [
        _reply("buy", position_size=2),
        _reply("hold"),
        _reply("hold"),
    ])
    manager = PortfolioManager(initial_capital=1000.0, allowed_symbols=["AAPL"])
    opened = datetime(2026, 4, 1, 14, 30)
    entry = _decide(manager, 250.0, opened)
    manager.execute_actions(entry["actions"], {"AAPL": {"close": 250.0}}, opened)
    assert manager.positions == {"AAPL": 2}

    # 200 is below the 240 stop; 320 is above the 300 target. Neither sells.
    for hour, price in ((15, 200.0), (16, 320.0)):
        timestamp = datetime(2026, 4, 1, hour, 30)
        decision = _decide(manager, price, timestamp)
        manager.execute_actions(decision["actions"], {"AAPL": {"close": price}}, timestamp)
        assert decision["actions"] == []
        assert manager.positions == {"AAPL": 2}

    assert remaining == []
    assert [trade["side"] for trade in manager.trades] == ["BUY"]
    assert manager.llm_decisions == 3


def test_the_legacy_order_path_has_no_stop_handling():
    """A source guard, because a stop could be wired in a branch the two cases
    above never reach. Lifting it is a decision, not a refactor: update the
    "not executed" copy in app.html (Live and Competition Rules + About cards)
    in the same PR."""
    for rel in (
        "domain/backtesting/portfolio_manager.py",
        "domain/trading/execution.py",
        "domain/leaderboard/strategies/llm_agent.py",
    ):
        text = (_BACKEND / rel).read_text(encoding="utf-8")
        for field in ("stop_loss", "take_profit"):
            assert field not in text, (
                f"{rel} now mentions {field}: if stops execute on the legacy "
                "engine, the 'not executed' disclaimers in app.html are false"
            )
