"""Phase 3C3 — leaderboard strategies registration characterization.

Verifies the canonical ``dashboard.backend.domain.leaderboard.strategies``
package and characterizes registry lookup, aliases, and unknown-strategy
behavior.
"""

import json

import pandas as pd
import pytest

from dashboard.backend.paths import CONFIG_DIR
from dashboard.backend.domain.leaderboard import strategies as canon
from dashboard.backend.domain.leaderboard.strategies import (
    llm_agent as llm_agent_module,
)
from dashboard.backend.domain.leaderboard.strategies.buy_hold import BuyHoldStrategy
from dashboard.backend.domain.leaderboard.strategies.equal_weight_buyhold import (
    EqualWeightBuyHoldStrategy,
)
from dashboard.backend.domain.leaderboard.strategies.equal_weight_index import (
    EqualWeightIndexStrategy,
)
from dashboard.backend.domain.leaderboard.strategies.llm_agent import LLMAgentStrategy
from dashboard.backend.domain.leaderboard.strategies.market_index import MarketIndexStrategy
from dashboard.backend.domain.leaderboard.strategies.mean_variance import MeanVarianceStrategy


_EXPECTED_KEYS = {
    "buy_hold",
    "equal_weight_index",
    "equal_weight_buyhold",
    "market_index",
    "mean_variance",
    "llm_agent",
}


# ---------------------------------------------------------------------------
# Registry class identity
# ---------------------------------------------------------------------------

def test_registry_module_identity_single_class_objects():
    # The registry must resolve to the SAME class objects exposed by submodules.
    registry = canon.available_strategies()
    assert registry["buy_hold"] is BuyHoldStrategy
    assert registry["equal_weight_index"] is EqualWeightIndexStrategy
    assert registry["equal_weight_buyhold"] is EqualWeightBuyHoldStrategy
    assert registry["market_index"] is MarketIndexStrategy
    assert registry["mean_variance"] is MeanVarianceStrategy
    assert registry["llm_agent"] is LLMAgentStrategy


# ---------------------------------------------------------------------------
# Registration + lookup
# ---------------------------------------------------------------------------

def test_available_strategies_keys():
    assert set(canon.available_strategies().keys()) == _EXPECTED_KEYS


def test_get_strategy_resolves_by_strategy_key():
    strat = canon.get_strategy({"id": "x", "name": "X", "strategy": "mean_variance"})
    assert isinstance(strat, MeanVarianceStrategy)
    assert strat.id == "x"
    assert strat.name == "X"


def test_get_strategy_supports_type_aliases():
    # Legacy config ``type`` values still resolve via the alias table.
    assert isinstance(canon.get_strategy({"type": "index"}), EqualWeightIndexStrategy)
    assert isinstance(canon.get_strategy({"type": "buy_hold"}), BuyHoldStrategy)


def test_get_strategy_unknown_raises_value_error():
    with pytest.raises(ValueError, match="Unknown baseline strategy"):
        canon.get_strategy({"strategy": "does_not_exist"})


def test_required_symbols_defaults_and_overrides():
    # equal_weight_index defaults to DJIA 30, honors explicit symbols.
    default = EqualWeightIndexStrategy({}).required_symbols()
    assert len(default) == 30
    custom = EqualWeightIndexStrategy({"symbols": ["AAPL", "MSFT"]}).required_symbols()
    assert custom == ["AAPL", "MSFT"]
    # market_index excludes ^-prefixed index symbols from the Alpaca fetch.
    assert MarketIndexStrategy({"symbols": ["^DJI"]}).required_symbols() == []


def test_llm_agent_passes_model_reasoning_to_client_factory(monkeypatch):
    captured = {}
    expected_client = object()

    def _fake_factory(integration=None, *, reasoning_effort=None):
        captured["integration"] = integration
        captured["reasoning_effort"] = reasoning_effort
        return expected_client

    monkeypatch.setattr(llm_agent_module, "HAS_ANTHROPIC", True)
    monkeypatch.setattr(llm_agent_module, "make_llm_client", _fake_factory)

    strategy = LLMAgentStrategy(
        {
            "integration": "openrouter",
            "reasoning_effort": "none",
        }
    )

    assert strategy.reasoning_effort == "none"
    assert strategy._make_client() is expected_client
    assert captured == {
        "integration": "openrouter",
        "reasoning_effort": "none",
    }


def test_llm_agent_without_reasoning_keeps_legacy_factory_behavior(monkeypatch):
    captured = {}

    def _fake_factory(integration=None, *, reasoning_effort=None):
        captured["integration"] = integration
        captured["reasoning_effort"] = reasoning_effort
        return object()

    monkeypatch.setattr(llm_agent_module, "HAS_ANTHROPIC", True)
    monkeypatch.setattr(llm_agent_module, "make_llm_client", _fake_factory)

    strategy = LLMAgentStrategy({"integration": "commonstack"})
    strategy._make_client()

    assert strategy.reasoning_effort is None
    assert captured == {
        "integration": "commonstack",
        "reasoning_effort": None,
    }


def test_only_nemotron_leaderboard_entry_disables_reasoning():
    config = json.loads((CONFIG_DIR / "leaderboard.json").read_text(encoding="utf-8"))
    strategies = config["strategies"]
    nemotron = next(s for s in strategies if s["id"] == "nemotron_3_nano_30b")

    assert nemotron["reasoning_effort"] == "none"
    assert all(
        "reasoning_effort" not in strategy
        for strategy in strategies
        if strategy["id"] != "nemotron_3_nano_30b"
    )


@pytest.mark.parametrize("temperature", [0, 0.5, 1])
def test_llm_agent_accepts_valid_temperature(temperature):
    strategy = LLMAgentStrategy({"temperature": temperature})
    assert strategy.temperature == temperature


def test_llm_agent_without_temperature_keeps_provider_default():
    strategy = LLMAgentStrategy({})
    assert strategy.temperature is None


@pytest.mark.parametrize(
    "temperature",
    ["0", True, False, -0.1, 1.1, 2, float("nan"), float("inf"), -float("inf")],
)
def test_llm_agent_rejects_invalid_temperature(temperature):
    """Every gateway here speaks the Anthropic Messages shape, which caps
    ``temperature`` at 1.0 — accepting the OpenAI 0–2 range would only turn a
    typo into a 400 at request time, i.e. a silent rule-based fallback."""
    with pytest.raises(ValueError, match="temperature"):
        LLMAgentStrategy({"temperature": temperature})


@pytest.mark.parametrize("effort", ["low", "medium", "high", "max"])
def test_llm_agent_rejects_temperature_with_reasoning_enabled(effort):
    """Anthropic rejects ``temperature`` alongside extended thinking, and the
    OpenRouter wrapper injects ``thinking`` whenever the effort enables it.
    Catch the conflict in config, not as a per-request 400."""
    with pytest.raises(ValueError, match="reasoning_effort"):
        LLMAgentStrategy({"temperature": 0, "reasoning_effort": effort})


@pytest.mark.parametrize("effort", ["none", "off", "disabled", "auto", "default"])
def test_llm_agent_allows_temperature_when_reasoning_off_or_passthrough(effort):
    strategy = LLMAgentStrategy({"temperature": 0, "reasoning_effort": effort})
    assert strategy.temperature == 0


def test_only_nemotron_leaderboard_entry_pins_temperature():
    config = json.loads((CONFIG_DIR / "leaderboard.json").read_text(encoding="utf-8"))
    strategies = config["strategies"]
    nemotron = next(s for s in strategies if s["id"] == "nemotron_3_nano_30b")

    assert nemotron["temperature"] == 0
    assert all(
        "temperature" not in strategy
        for strategy in strategies
        if strategy["id"] != "nemotron_3_nano_30b"
    )


@pytest.mark.parametrize(
    ("config_temperature", "expected"),
    [(0, 0), (None, None)],
)
def test_llm_agent_passes_temperature_to_portfolio_manager(
    monkeypatch,
    config_temperature,
    expected,
):
    captured = []
    timestamp = pd.Timestamp("2026-04-15 14:00:00+00:00")
    bars = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1000],
            "rsi": [50.0],
            "macd": [0.0],
            "macd_signal": [0.0],
            "sma20": [100.0],
            "sma50": [100.0],
            "bb_upper": [110.0],
            "bb_lower": [90.0],
        },
        index=[timestamp],
    )

    def _fake_decision(self, state, client, **kwargs):
        captured.append(kwargs)
        return {"actions": []}

    monkeypatch.setattr(
        llm_agent_module.TechnicalIndicators,
        "calculate_indicators",
        staticmethod(lambda frame: frame),
    )
    monkeypatch.setattr(
        llm_agent_module.LLMAgentStrategy,
        "_make_client",
        lambda self: object(),
    )
    monkeypatch.setattr(
        llm_agent_module.PortfolioManager,
        "make_trading_decision_with_llm",
        _fake_decision,
    )

    strategy_config = {
        "symbols": ["AAPL"],
        "model_id": "test-model",
    }
    if config_temperature is not None:
        strategy_config["temperature"] = config_temperature
    strategy = LLMAgentStrategy(strategy_config)
    strategy.run(
        {"AAPL": bars},
        "2026-04-15",
        "2026-04-15",
        10000,
    )

    assert len(captured) == 1
    assert captured[0]["temperature"] == expected
