import json
from datetime import datetime

import pandas as pd
import pytz

from dashboard.backend.database import BacktestDatabase
from dashboard.backend.domain.backtesting.portfolio_manager import PortfolioManager
from dashboard.backend.domain.leaderboard.strategies import llm_agent as llm_agent_module
from dashboard.backend.domain.leaderboard.strategies.llm_agent import LLMAgentStrategy


class _Usage:
    input_tokens = 12
    output_tokens = 8


class _Block:
    def __init__(self, block_type, text=None):
        self.type = block_type
        if text is not None:
            self.text = text


class _Response:
    def __init__(self, blocks):
        self.content = blocks
        self.usage = _Usage()


class _Client:
    def __init__(self, response):
        self.response = response

        class _Messages:
            def __init__(inner_self, outer):
                inner_self.outer = outer

            def create(inner_self, **kwargs):
                return inner_self.outer.response

        self.messages = _Messages(self)


class _SequenceClient:
    def __init__(self, responses):
        self.responses = list(responses)

        class _Messages:
            def __init__(inner_self, outer):
                inner_self.outer = outer

            def create(inner_self, **kwargs):
                return inner_self.outer.responses.pop(0)

        self.messages = _Messages(self)


def _state():
    return {
        "timestamp": datetime(2026, 1, 1),
        "cash": 100000,
        "positions": [],
        "positions_value": 0,
        "total_equity": 100000,
        "market_signals": {
            "AAPL": {
                "price": 100.0,
                "rsi": 55.0,
                "macd": 1.0,
                "macd_signal": 0.5,
                "sma20": 99.0,
                "sma50": 98.0,
                "bb_upper": 110.0,
                "bb_lower": 90.0,
            }
        },
    }


def test_database_diagnostics_round_trip_is_idempotent(tmp_path):
    db = BacktestDatabase(tmp_path / "diagnostics.db")
    entry = {
        "step_index": 0,
        "timestamp": "2026-01-01T14:00:00+00:00",
        "model_id": "nvidia/nemotron-3-nano-30b-a3b",
        "integration": "openrouter",
        "reasoning_effort": "medium",
        "response_block_types": ["thinking", "text"],
        "text_present": True,
        "parse_success": True,
        "retry_count": 0,
        "llm_call_count": 1,
        "actions_proposed": [{"symbol": "AAPL", "action": "buy", "position_size": 10}],
        "actions_accepted": 1,
        "trades_executed": 1,
        "latency_ms": 250,
    }

    db.insert_llm_diagnostics("run-1", [entry])
    db.insert_llm_diagnostics("run-1", [{**entry, "trades_executed": 2}])

    rows = db.get_llm_diagnostics("run-1")
    assert len(rows) == 1
    assert rows[0]["response_block_types"] == ["thinking", "text"]
    assert rows[0]["actions_proposed"][0]["symbol"] == "AAPL"
    assert rows[0]["trades_executed"] == 2


def test_diagnostic_action_summary_excludes_reasoning():
    response = _Response([
        _Block("thinking"),
        _Block(
            "text",
            json.dumps({
                "actions": [{
                    "symbol": "AAPL",
                    "action": "buy",
                    "confidence": 0.9,
                    "position_size": 10,
                    "reasoning": "private model reasoning",
                }]
            }),
        ),
    ])
    manager = PortfolioManager(100000)
    result = manager.make_trading_decision_with_llm(
        _state(),
        _Client(response),
        model="nvidia/nemotron-3-nano-30b-a3b",
        integration="openrouter",
        step_index=3,
    )

    assert result["actions"][0]["symbol"] == "AAPL"
    diagnostic = manager.llm_diagnostics[0]
    assert diagnostic["response_block_types"] == ["thinking", "text"]
    assert diagnostic["text_present"] is True
    assert diagnostic["parse_success"] is True
    assert diagnostic["actions_accepted"] == 1
    assert diagnostic["actions_proposed"][0]["position_size"] == 10
    assert "reasoning" not in diagnostic["actions_proposed"][0]


def test_diagnostic_records_no_text_retries_and_fallback():
    response = _Response([_Block("thinking"), _Block("redacted_thinking")])
    manager = PortfolioManager(100000)
    result = manager.make_trading_decision_with_llm(
        _state(),
        _Client(response),
        model="nvidia/nemotron-3-nano-30b-a3b",
        integration="openrouter",
        step_index=1,
    )

    assert result == manager.make_trading_decision(_state())
    diagnostic = manager.llm_diagnostics[0]
    assert diagnostic["response_block_types"] == ["thinking", "redacted_thinking"]
    assert diagnostic["text_present"] is False
    assert diagnostic["parse_success"] is False
    assert diagnostic["retry_count"] == 4
    assert diagnostic["llm_call_count"] == 5
    assert diagnostic["fallback_reason"] == "no_text_after_retries"
    assert manager.llm_diagnostic_summary()["no_text_steps"] == 1


def test_summary_counts_recovered_no_text_step():
    valid = _Response([_Block("text", '{"actions": [{"symbol": "AAPL", "action": "hold"}]}')])
    empty = _Response([_Block("thinking"), _Block("redacted_thinking")])
    manager = PortfolioManager(100000)
    manager.make_trading_decision_with_llm(
        _state(),
        _SequenceClient([empty, valid]),
        model="nvidia/nemotron-3-nano-30b-a3b",
        integration="openrouter",
        step_index=2,
    )

    diagnostic = manager.llm_diagnostics[0]
    assert diagnostic["text_present"] is True
    assert diagnostic["retry_count"] == 1
    assert diagnostic["fallback_reason"] is None
    assert manager.llm_diagnostic_summary()["no_text_steps"] == 1


def test_leaderboard_strategy_produces_one_diagnostic_per_llm_step(monkeypatch):
    et = pytz.timezone("US/Eastern")
    index = pd.DatetimeIndex([
        et.localize(datetime(2026, 1, 5 + (i // 6), 10 + (i % 6), 0))
        for i in range(12)
    ])
    prices = [100.0 + i * 0.25 for i in range(len(index))]
    bars = {
        "AAPL": pd.DataFrame(
            {
                "open": prices,
                "high": [p + 1 for p in prices],
                "low": [p - 1 for p in prices],
                "close": prices,
                "volume": [1000] * len(prices),
            },
            index=index,
        )
    }
    hold_response = _Response([
        _Block("text", json.dumps({
            "actions": [{
                "symbol": "AAPL",
                "action": "hold",
                "confidence": 0.9,
                "reasoning": "do not persist this",
            }]
        }))
    ])
    monkeypatch.setattr(llm_agent_module, "HAS_ANTHROPIC", True)
    monkeypatch.setattr(
        llm_agent_module,
        "make_llm_client",
        lambda integration=None: _Client(hold_response),
    )

    strategy = LLMAgentStrategy({
        "id": "nemotron-test",
        "name": "Nemotron test",
        "model_id": "nvidia/nemotron-3-nano-30b-a3b",
        "integration": "openrouter",
        "symbols": ["AAPL"],
    })
    curve = strategy.run(bars, "2026-01-05", "2026-01-05", 100000)

    assert curve
    assert len(strategy.llm_diagnostics) == len(curve)
    assert all(item["parse_success"] for item in strategy.llm_diagnostics)
    assert strategy.run_metadata["decision_steps"] == len(curve)
