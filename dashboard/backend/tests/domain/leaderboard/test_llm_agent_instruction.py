"""C1: an Open Track entry's instruction must reach the shared prompt builder.

`make_trading_decision_with_llm` already accepts `strategy_prompt` and threads it
into `create_prompt(custom_prompt=...)`. The house path never passed it, so every
leaderboard entry ran the bare SAFE_TRADING_PROMPT. These guard that the wire is
connected and, just as importantly, that omitting the key preserves today's
behaviour for the seven published Model Track entries.
"""

import pytest

from dashboard.backend.domain.leaderboard.strategies.llm_agent import LLMAgentStrategy

BASE_CONFIG = {
    "strategy": "llm_agent",
    "model_id": "nvidia/nemotron-3-nano-30b-a3b",
    "integration": "openrouter",
    "temperature": 0,
    "reasoning_effort": "none",
    "mode": "safe_trading",
    "symbols": [],
}


class FakeManager:
    """Stands in for PortfolioManager so the decision loop runs without bars."""

    cash = 10_000.0
    trades = []
    equity_history = [{"equity": 10_000.0}]
    llm_calls = 0
    llm_decisions = 0
    input_tokens = 0
    output_tokens = 0

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs

    def get_portfolio_state(self, market_data, price_cache, ts):
        return {}

    def make_trading_decision_with_llm(self, state, client, **kwargs):
        FakeManager.seen = kwargs
        return {"actions": []}

    def execute_actions(self, actions, market_data, ts):
        pass

    def update_equity(self, market_data, price_cache, ts):
        pass

    def get_equity_curve(self):
        return [{"timestamp": "2026-04-15T14:00:00", "equity": 10_000.0}]


def _drive_one_step(monkeypatch, config):
    """Run a single decision step and return the kwargs the manager saw."""
    import dashboard.backend.domain.leaderboard.strategies.llm_agent as mod

    FakeManager.seen = {}
    monkeypatch.setattr(mod, "PortfolioManager", FakeManager)

    strategy = LLMAgentStrategy(config)
    strategy._run_decision_loop(
        client=object(),
        timestamps=["2026-04-15T14:00:00"],
        symbols=["AAPL"],
        data={},
        price_cache={},
        initial_capital=10_000.0,
    )
    return FakeManager.seen


def test_instruction_is_read_from_config():
    strategy = LLMAgentStrategy({**BASE_CONFIG, "strategy_prompt": "Buy the dip."})
    assert strategy.strategy_prompt == "Buy the dip."


def test_missing_instruction_is_none_not_empty_string():
    """The published Model Track entries carry no `strategy_prompt` key.

    `None` and `""` are NOT interchangeable downstream: `create_prompt` branches on
    truthiness, so an empty string would take the same branch as None today but
    silently diverge if that branch is ever tightened to `is not None`.
    """
    strategy = LLMAgentStrategy(dict(BASE_CONFIG))
    assert strategy.strategy_prompt is None


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
def test_blank_instruction_collapses_to_none(blank):
    strategy = LLMAgentStrategy({**BASE_CONFIG, "strategy_prompt": blank})
    assert strategy.strategy_prompt is None


def test_instruction_is_stripped():
    strategy = LLMAgentStrategy({**BASE_CONFIG, "strategy_prompt": "  Hold cash.  "})
    assert strategy.strategy_prompt == "Hold cash."


def test_instruction_is_passed_to_the_decision_call(monkeypatch):
    """The attribute existing is not the contract — reaching the call site is."""
    seen = _drive_one_step(
        monkeypatch, {**BASE_CONFIG, "strategy_prompt": "Rotate weekly."}
    )
    assert seen.get("strategy_prompt") == "Rotate weekly."


def test_extraction_preserved_the_other_decision_kwargs(monkeypatch):
    """Guards the *extraction*, not the feature.

    Pulling the loop out of run() is the risky half of this task: the seven
    published Model Track curves are produced by these exact kwargs, and a
    dropped one would change them silently while every instruction test above
    still passed.
    """
    seen = _drive_one_step(
        monkeypatch, {**BASE_CONFIG, "strategy_prompt": "Rotate weekly."}
    )
    assert seen.get("mode") == "safe_trading"
    assert seen.get("model") == "nvidia/nemotron-3-nano-30b-a3b"
    assert seen.get("temperature") == 0


def test_published_entries_send_no_instruction(monkeypatch):
    """A Model Track entry must reach the call with strategy_prompt=None.

    Not merely 'absent from config' — absent must still arrive as an explicit
    None, because that is what keeps create_prompt on the SAFE_TRADING_PROMPT
    branch that produced the published curves.
    """
    seen = _drive_one_step(monkeypatch, dict(BASE_CONFIG))
    assert seen.get("strategy_prompt") is None
