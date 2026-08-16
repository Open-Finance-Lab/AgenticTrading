"""C1: an Open Track entry's instruction must reach the shared prompt builder.

`make_trading_decision_with_llm` already accepts `strategy_prompt` and threads it
into `create_prompt(custom_prompt=...)`. The house path never passed it, so every
leaderboard entry ran the bare SAFE_TRADING_PROMPT. These guard that the wire is
connected and, just as importantly, that omitting the key preserves today's
behaviour for the seven published Model Track entries.
"""

import pytest

from dashboard.backend.domain.leaderboard.strategies import llm_agent
from dashboard.backend.infrastructure.market_data.profiles import ALPACA, get_market_profile

# Imported as a module because `_drive_one_step` monkeypatches PortfolioManager on
# it; the alias keeps the cases reading as plain constructor calls. One import form
# only — importing the module *and* `from`-importing the class trips
# CodeQL py/import-and-import-from.
LLMAgentStrategy = llm_agent.LLMAgentStrategy
MAX_STRATEGY_PROMPT_CHARS = llm_agent.MAX_STRATEGY_PROMPT_CHARS

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
        # On the class, not the instance: _run_decision_loop constructs the
        # manager itself, so a test can only see these through the fake.
        FakeManager.init_kwargs_seen = kwargs

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
    FakeManager.seen = {}
    FakeManager.init_kwargs_seen = {}
    monkeypatch.setattr(llm_agent, "PortfolioManager", FakeManager)

    strategy = LLMAgentStrategy(config)
    strategy._run_decision_loop(
        client=object(),
        timestamps=["2026-04-15T14:00:00"],
        symbols=["AAPL"],
        data={},
        price_cache={},
        initial_capital=10_000.0,
        model_id=config.get("model_id"),
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


@pytest.mark.parametrize("bad", [250, 12.5, ["a"], {"a": 1}, True])
def test_non_string_instruction_raises_a_typed_error(bad):
    """`get_strategy()` runs on every public GET /api/v1/leaderboard.

    `_symbols_for_config` and `_config_needs_alpaca` construct every entry to
    decide the symbol set, so an AttributeError out of `.strip()` here 500s the
    whole board anonymously — and per the prod notes a bare 500 reaches the
    browser as a CORS error, which is unreadable as a diagnosis.
    """
    with pytest.raises(ValueError, match="strategy_prompt must be a string"):
        LLMAgentStrategy({**BASE_CONFIG, "strategy_prompt": bad})


def test_overlong_instruction_is_refused():
    """The text rides every LLM call in the run, not one request.

    The probe write-up's finding #2 is the live case: an instruction that makes
    the model emit one action per DJIA symbol overran LLM_MAX_OUTPUT_TOKENS,
    truncated 18 steps into rule-based fallback, failed H6 at 89.4% coverage —
    and was the most expensive run in the leg, because truncated calls bill full.
    """
    with pytest.raises(ValueError, match="over the 4000 limit"):
        LLMAgentStrategy(
            {**BASE_CONFIG, "strategy_prompt": "x" * (MAX_STRATEGY_PROMPT_CHARS + 1)}
        )


def test_instruction_cap_matches_the_api_surface():
    """The two constants are copies — domain/ may not import api/.

    Without this, raising the cap on /backtest/run would silently leave the
    leaderboard path on the old number (or vice versa).
    """
    from dashboard.backend.api.routers.backtests import (
        MAX_STRATEGY_PROMPT_CHARS as API_CAP,
    )

    assert MAX_STRATEGY_PROMPT_CHARS == API_CAP


def test_explicit_non_default_mode_conflicts_with_an_instruction():
    """`create_prompt` ignores `mode` once a custom prompt is set.

    A buy_and_hold entry carrying an instruction would publish a curve labelled
    with a mode whose prompt body was never sent.
    """
    with pytest.raises(ValueError, match="cannot be combined with strategy_prompt"):
        LLMAgentStrategy(
            {**BASE_CONFIG, "mode": "buy_and_hold", "strategy_prompt": "Buy the dip."}
        )


def test_explicit_safe_trading_mode_is_not_a_conflict():
    """Replacing the safe_trading body is the point of the feature, not a bug.

    Guards the conflict check against over-firing: every Model Track entry names
    safe_trading explicitly, so rejecting that pair would refuse the only
    combination Phase 2 will ever ship.
    """
    strategy = LLMAgentStrategy(
        {**BASE_CONFIG, "mode": "safe_trading", "strategy_prompt": "Buy the dip."}
    )
    assert strategy.strategy_prompt == "Buy the dip."


def test_extraction_preserved_the_manager_construction(monkeypatch):
    """The extraction moved the PortfolioManager() call too, not just the loop.

    `t_plus_one_enabled` comes from the ALPACA market profile: drop it and every
    published curve silently re-runs under settlement semantics its market does
    not have, while all the kwarg assertions above still pass. Asserted against
    the profile rather than a literal so a profile change moves both together.
    """
    _drive_one_step(monkeypatch, dict(BASE_CONFIG))

    expected = get_market_profile(ALPACA).t_plus_one_enabled
    assert FakeManager.init_kwargs_seen.get("initial_capital") == 10_000.0
    assert FakeManager.init_kwargs_seen.get("t_plus_one_enabled") == expected


def test_published_entries_send_no_instruction(monkeypatch):
    """A Model Track entry must reach the call with strategy_prompt=None.

    Not merely 'absent from config' — absent must still arrive as an explicit
    None, because that is what keeps create_prompt on the SAFE_TRADING_PROMPT
    branch that produced the published curves.
    """
    seen = _drive_one_step(monkeypatch, dict(BASE_CONFIG))
    # Membership first. `seen.get("strategy_prompt") is None` is equally true when
    # the keyword was never passed at all, so on its own it passes with
    # `strategy_prompt=self.strategy_prompt` deleted from llm_agent.py — i.e. it
    # could not fail for the reason this docstring gives.
    assert "strategy_prompt" in seen
    assert seen["strategy_prompt"] is None
