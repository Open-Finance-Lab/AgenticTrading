"""Hosted agent runtime dispatch and AI Hedge Fund adapter tests."""

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from dashboard.backend.domain.agents.runtime import (
    AI_HEDGE_FUND_RUNTIME_TYPE,
    PIPELINE_RUNTIME_TYPE,
    AgentRuntimeContext,
    RuntimeDispatcher,
    normalize_runtime_config,
)
from dashboard.backend.infrastructure.ai_hedge_fund.adapter import (
    AiHedgeFundOutputError,
    AiHedgeFundRuntime,
    AiHedgeFundRuntimeError,
    AiHedgeFundSubprocessRunner,
)
from dashboard.backend.infrastructure.ai_hedge_fund.bridge import (
    _disable_dotenv_loading,
)
from dashboard.backend.domain.backtesting.engine import (
    _prior_market_date_by_decision_date,
)


def _context(*, timestamp=None, latest_market_date_before_decision=date(2026, 4, 30)):
    return AgentRuntimeContext(
        timestamp=timestamp or datetime(2026, 5, 1, 14, tzinfo=timezone.utc),
        backtest_start_date="2026-05-01",
        symbols=["AAPL", "MSFT", "IBM", "JPM", "DIS"],
        cash=1_000.0,
        total_equity=1_100.0,
        positions={"MSFT": 5},
        entry_prices={"MSFT": 18.0},
        current_prices={
            "AAPL": 10.0,
            "MSFT": 20.0,
            "IBM": 30.0,
            "JPM": 40.0,
            "DIS": 50.0,
        },
        latest_market_date_before_decision=latest_market_date_before_decision,
        market={"market": "US", "timeframe": "1h"},
    )


def test_pipeline_dispatch_preserves_existing_handler_result():
    expected = {"actions": [{"symbol": "AAPL", "action": "buy", "shares": 1}]}
    calls = []
    dispatcher = RuntimeDispatcher(PIPELINE_RUNTIME_TYPE)

    result = dispatcher.dispatch(
        _context(), pipeline_handler=lambda: calls.append("pipeline") or expected
    )

    assert result is expected
    assert calls == ["pipeline"]
    assert dispatcher.calls == 0


def test_ai_hedge_fund_dispatch_skips_pipeline_handler():
    class FakeRuntime:
        calls = 1

        def decide(self, context):
            return {"actions": [{"symbol": context.symbols[0], "action": "hold"}]}

    dispatcher = RuntimeDispatcher(
        AI_HEDGE_FUND_RUNTIME_TYPE, runtime=FakeRuntime()
    )
    pipeline_calls = []

    result = dispatcher.dispatch(
        _context(), pipeline_handler=lambda: pipeline_calls.append("called")
    )

    assert result["actions"][0]["symbol"] == "AAPL"
    assert pipeline_calls == []
    assert dispatcher.calls == 1


def test_ai_hedge_fund_maps_buy_sell_and_long_only_holds_through_atl():
    output = {
        "decisions": {
            "AAPL": {
                "action": "buy",
                "quantity": 10,
                "confidence": 80,
                "reasoning": "Strong combined signal",
            },
            "MSFT": {
                "action": "sell",
                "quantity": 3,
                "confidence": 70,
                "reasoning": "Valuation is stretched",
            },
            "IBM": {
                "action": "hold",
                "quantity": 0,
                "confidence": 60,
                "reasoning": "No edge right now",
            },
            "JPM": {
                "action": "short",
                "quantity": 2,
                "confidence": 90,
                "reasoning": "Bearish but ATL is long only",
            },
            "DIS": {
                "action": "cover",
                "quantity": 1,
                "confidence": 90,
                "reasoning": "Cover is unavailable in ATL MVP",
            },
        }
    }

    actions = AiHedgeFundRuntime.output_to_atl_actions(output, _context())

    assert [(item["symbol"], item["action"], item["shares"]) for item in actions] == [
        ("AAPL", "buy", 10),
        ("MSFT", "sell", 3),
    ]
    assert all(item["reason"].startswith("[AI Hedge Fund]") for item in actions)


@pytest.mark.parametrize(
    "output",
    [
        {},
        {"decisions": []},
        {"decisions": {"TSLA": {"action": "hold", "quantity": 0, "confidence": 50, "reasoning": "Outside universe"}}},
        {"decisions": {"AAPL": {"action": "explode", "quantity": 1, "confidence": 50, "reasoning": "Invalid action"}}},
        {"decisions": {"AAPL": {"action": "buy", "quantity": "10", "confidence": 50, "reasoning": "Invalid quantity"}}},
    ],
)
def test_ai_hedge_fund_rejects_invalid_output(output):
    with pytest.raises(AiHedgeFundOutputError):
        AiHedgeFundRuntime.output_to_atl_actions(output, _context())


def test_ai_hedge_fund_builds_upstream_portfolio_and_runs_once_daily():
    class FakeRunner:
        def __init__(self):
            self.payloads = []

        def run(self, payload, *, timeout_seconds):
            self.payloads.append((payload, timeout_seconds))
            return {"decisions": {}}

    runner = FakeRunner()
    runtime = AiHedgeFundRuntime(
        {"analysts": ["technical_analyst"]},
        runner=runner,
        environment={
            "AI_HEDGE_FUND_LOOKBACK_DAYS": "30",
            "AI_HEDGE_FUND_TIMEOUT_SECONDS": "45",
            "AI_HEDGE_FUND_MODEL_NAME": "gpt-platform-model",
        },
    )

    first = runtime.decide(_context())
    same_day = runtime.decide(
        _context(timestamp=datetime(2026, 5, 1, 19, tzinfo=timezone.utc))
    )
    next_day = runtime.decide(
        _context(
            timestamp=datetime(2026, 5, 2, 14, tzinfo=timezone.utc),
            latest_market_date_before_decision=date(2026, 5, 1),
        )
    )

    assert first == same_day == next_day == {"actions": []}
    assert runtime.calls == 2
    assert len(runner.payloads) == 2
    payload, timeout = runner.payloads[0]
    assert timeout == 45
    assert payload["start_date"] == "2026-03-31"
    assert payload["end_date"] == "2026-04-30"
    assert payload["selected_analysts"] == ["technical_analyst"]
    assert payload["model_name"] == "gpt-platform-model"
    assert payload["model_provider"] == "OpenAI"
    assert payload["portfolio"]["positions"]["MSFT"]["long"] == 5
    assert payload["portfolio"]["positions"]["MSFT"]["short"] == 0


def test_ai_hedge_fund_cutoff_is_previous_atl_trading_date_not_calendar_day():
    timestamps = [
        datetime(2026, 5, 1, 14, tzinfo=timezone.utc),  # Friday
        datetime(2026, 5, 1, 15, tzinfo=timezone.utc),
        datetime(2026, 5, 4, 14, tzinfo=timezone.utc),  # Monday
    ]
    prior_dates = _prior_market_date_by_decision_date(timestamps)

    assert prior_dates[date(2026, 5, 1)] is None
    assert prior_dates[date(2026, 5, 4)] == date(2026, 5, 1)

    runtime = AiHedgeFundRuntime(
        {"analysts": ["technical_analyst"]},
        runner=object(),
        environment={"AI_HEDGE_FUND_LOOKBACK_DAYS": "30"},
    )
    payload = runtime._upstream_payload(
        _context(
            timestamp=datetime(2026, 5, 4, 14, tzinfo=timezone.utc),
            latest_market_date_before_decision=prior_dates[date(2026, 5, 4)],
        )
    )

    assert payload["end_date"] == "2026-05-01"
    assert payload["start_date"] == "2026-04-01"


def test_ai_hedge_fund_holds_when_atl_has_no_prior_market_date():
    class UnexpectedRunner:
        def run(self, *_args, **_kwargs):
            raise AssertionError("runtime must not run without a prior ATL market date")

    runtime = AiHedgeFundRuntime(
        {"analysts": ["technical_analyst"]}, runner=UnexpectedRunner()
    )

    assert runtime.decide(
        _context(latest_market_date_before_decision=None)
    ) == {"actions": []}
    assert runtime.calls == 0


def test_ai_hedge_fund_rejects_non_prior_cutoff():
    runtime = AiHedgeFundRuntime(
        {"analysts": ["technical_analyst"]}, runner=object()
    )

    with pytest.raises(AiHedgeFundRuntimeError, match="before the decision date"):
        runtime._upstream_payload(
            _context(
                latest_market_date_before_decision=date(2026, 5, 1)
            )
        )


def test_upstream_dependencies_are_not_in_main_requirements():
    repo_root = Path(__file__).resolve().parents[5]
    main_requirements = (repo_root / "requirements.txt").read_text(encoding="utf-8")
    isolated_requirements = (
        repo_root / "requirements-ai-hedge-fund.txt"
    ).read_text(encoding="utf-8")

    assert "ai-hedge-fund" not in main_requirements
    assert "langgraph" not in main_requirements
    assert "ai-hedge-fund" in isolated_requirements
    assert "9557e64273e212635a4a28cbd8128df22f166c07" in isolated_requirements


def test_isolated_runtime_does_not_inherit_unrelated_atl_secrets():
    runner = AiHedgeFundSubprocessRunner(
        {
            "PATH": "/usr/bin",
            "OPENAI_API_KEY": "allowed-model-key",
            "FINANCIAL_DATASETS_API_KEY": "allowed-data-key",
            "ANTHROPIC_API_KEY": "must-not-cross-boundary",
            "CONTENT_DATABASE_URL": "must-not-cross-boundary",
            "DISCORD_CLIENT_SECRET": "must-not-cross-boundary",
        }
    )

    environment = runner._subprocess_environment()

    assert environment["OPENAI_API_KEY"] == "allowed-model-key"
    assert environment["FINANCIAL_DATASETS_API_KEY"] == "allowed-data-key"
    assert "CONTENT_DATABASE_URL" not in environment
    assert "DISCORD_CLIENT_SECRET" not in environment
    assert "ANTHROPIC_API_KEY" not in environment
    assert environment["PYTHON_DOTENV_DISABLED"] == "1"


def test_bridge_blocks_dotenv_for_the_pinned_legacy_dependency():
    import dotenv
    from dotenv import main as dotenv_main

    public_loader = dotenv.load_dotenv
    module_loader = dotenv_main.load_dotenv
    try:
        _disable_dotenv_loading()

        assert dotenv.load_dotenv() is False
        assert dotenv_main.load_dotenv() is False
    finally:
        dotenv.load_dotenv = public_loader
        dotenv_main.load_dotenv = module_loader


def test_ai_hedge_fund_config_allows_analyst_composition_only():
    assert normalize_runtime_config(
        AI_HEDGE_FUND_RUNTIME_TYPE,
        {"analysts": ["warren_buffett", "technical_analyst"]},
    ) == {"analysts": ["warren_buffett", "technical_analyst"]}

    for protected in (
        "model_name",
        "model_provider",
        "decision_interval",
        "lookback_days",
        "timeout_seconds",
    ):
        with pytest.raises(ValueError, match="Unsupported"):
            normalize_runtime_config(
                AI_HEDGE_FUND_RUNTIME_TYPE,
                {"analysts": ["technical_analyst"], protected: "user-value"},
            )


@pytest.mark.parametrize(
    "analysts",
    [[], ["unknown_analyst"], ["technical_analyst", "technical_analyst"]],
)
def test_ai_hedge_fund_rejects_invalid_analyst_composition(analysts):
    with pytest.raises(ValueError):
        normalize_runtime_config(
            AI_HEDGE_FUND_RUNTIME_TYPE, {"analysts": analysts}
        )
