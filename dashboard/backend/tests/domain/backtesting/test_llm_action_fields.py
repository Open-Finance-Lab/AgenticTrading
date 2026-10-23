"""The trading loop strips the model's action the way the strict check does (#658).

``"buy "`` passed the strict check, which strips, then matched neither the
buy nor the sell branch, which compared it raw: no trade, yet the step was
counted in ``llm_decisions``.
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
                     "macd_signal": 0.0, "sma_20": 250.0, "sma_50": 250.0,
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
