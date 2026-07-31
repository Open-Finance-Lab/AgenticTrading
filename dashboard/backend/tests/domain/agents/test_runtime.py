"""Hosted agent runtime dispatch and AI Hedge Fund adapter tests."""

import json
import sys
import types
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
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PROVIDER,
    AiHedgeFundOutputError,
    AiHedgeFundRuntime,
    AiHedgeFundRuntimeError,
    AiHedgeFundSubprocessRunner,
)
from dashboard.backend.infrastructure.ai_hedge_fund.bridge import (
    _disable_dotenv_loading,
    _managed_model_from_payload,
)
from dashboard.backend.infrastructure.ai_hedge_fund import bridge
from dashboard.backend.domain.backtesting.engine import (
    _prior_market_date_by_decision_date,
)
from dashboard.backend.infrastructure.llm.providers import openrouter


def _context(
    *,
    timestamp=None,
    latest_market_date_before_decision=date(2026, 4, 30),
    cash=1_000.0,
    positions=None,
    current_prices=None,
):
    return AgentRuntimeContext(
        timestamp=timestamp or datetime(2026, 5, 1, 14, tzinfo=timezone.utc),
        backtest_start_date="2026-05-01",
        symbols=["AAPL", "MSFT", "IBM", "JPM", "DIS"],
        cash=cash,
        total_equity=1_100.0,
        positions={"MSFT": 5} if positions is None else positions,
        entry_prices={"MSFT": 18.0},
        current_prices=current_prices
        if current_prices is not None
        else {
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

    actions, traces = AiHedgeFundRuntime.output_to_atl_actions_with_trace(
        output, _context()
    )

    assert [(item["symbol"], item["action"], item["shares"]) for item in actions] == [
        ("AAPL", "buy", 10),
        ("MSFT", "sell", 3),
    ]
    assert all(item["reason"].startswith("[AI Hedge Fund]") for item in actions)
    by_ticker = {trace["ticker"]: trace for trace in traces}
    assert by_ticker["AAPL"]["order_emitted"] is True
    assert by_ticker["AAPL"]["mapped_atl_action"] == "buy"
    assert by_ticker["MSFT"]["order_emitted"] is True
    assert by_ticker["MSFT"]["mapped_atl_action"] == "sell"
    assert by_ticker["IBM"]["filter_reason"] == "upstream_hold"
    assert by_ticker["JPM"]["filter_reason"] == "short_unsupported_long_only"
    assert by_ticker["DIS"]["filter_reason"] == "cover_unsupported_long_only"
    assert all("reasoning" not in trace for trace in traces)


@pytest.mark.parametrize(
    ("symbol", "decision", "context", "mapped_action", "filter_reason"),
    [
        (
            "AAPL",
            {"action": "hold", "quantity": 0, "confidence": 60, "reasoning": "No edge"},
            _context(),
            "hold",
            "upstream_hold",
        ),
        (
            "JPM",
            {
                "action": "short",
                "quantity": 2,
                "confidence": 90,
                "reasoning": "Bearish signal",
            },
            _context(),
            "hold",
            "short_unsupported_long_only",
        ),
        (
            "DIS",
            {
                "action": "cover",
                "quantity": 1,
                "confidence": 90,
                "reasoning": "Cover signal",
            },
            _context(),
            "hold",
            "cover_unsupported_long_only",
        ),
        (
            "AAPL",
            {
                "action": "buy",
                "quantity": 0,
                "confidence": 80,
                "reasoning": "Zero sized buy",
            },
            _context(),
            "buy",
            "zero_quantity",
        ),
        (
            "AAPL",
            {
                "action": "buy",
                "quantity": "10",
                "confidence": 80,
                "reasoning": "Malformed size",
            },
            _context(),
            None,
            "invalid_quantity",
        ),
        (
            "AAPL",
            {
                "action": "buy",
                "quantity": 2,
                "confidence": 80,
                "reasoning": "Too expensive",
            },
            _context(cash=10.0),
            "buy",
            "insufficient_cash",
        ),
        (
            "IBM",
            {
                "action": "buy",
                "quantity": 1,
                "confidence": 80,
                "reasoning": "No current price",
            },
            _context(current_prices={"AAPL": 10.0, "MSFT": 20.0}),
            "buy",
            "missing_or_invalid_price",
        ),
        (
            "AAPL",
            {
                "action": "sell",
                "quantity": 1,
                "confidence": 80,
                "reasoning": "Nothing held",
            },
            _context(positions={}),
            "sell",
            "sell_without_position",
        ),
    ],
)
def test_ai_hedge_fund_traces_every_required_no_order_reason(
    symbol, decision, context, mapped_action, filter_reason
):
    actions, traces = AiHedgeFundRuntime.output_to_atl_actions_with_trace(
        {"decisions": {symbol: decision}}, context
    )

    assert actions == []
    assert traces == [
        {
            "decision_date": "2026-05-01",
            "data_cutoff_date": "2026-04-30",
            "ticker": symbol,
            "upstream_action": decision["action"],
            "upstream_quantity": (
                decision["quantity"] if isinstance(decision["quantity"], int) else None
            ),
            "confidence": float(decision["confidence"]),
            "mapped_atl_action": mapped_action,
            "order_emitted": False,
            "filter_reason": filter_reason,
        }
    ]


def test_ai_hedge_fund_trace_covers_existing_low_confidence_filter():
    actions, traces = AiHedgeFundRuntime.output_to_atl_actions_with_trace(
        {
            "decisions": {
                "AAPL": {
                    "action": "buy",
                    "quantity": 1,
                    "confidence": 29,
                    "reasoning": "Below the ATL threshold",
                }
            }
        },
        _context(),
    )

    assert actions == []
    assert traces[0]["filter_reason"] == "below_minimum_confidence"


def test_ai_hedge_fund_persists_only_bounded_upstream_diagnostics():
    omitted_reasoning = "must-not-be-persisted " * 100
    output = {
        "decisions": {
            "AAPL": {
                "action": "hold",
                "quantity": 0,
                "confidence": 88,
                "reasoning": "  Portfolio Manager chose HOLD.  " + "x" * 400,
            }
        },
        "analyst_signals": {
            "technical_analyst_agent": {
                "AAPL": {
                    "signal": "bullish",
                    "confidence": 71.25,
                    "reasoning": omitted_reasoning,
                }
            },
            "fundamentals_analyst_agent": {
                "AAPL": {
                    "signal": "neutral",
                    "confidence": 62,
                    "reasoning": omitted_reasoning,
                }
            },
            "sentiment_analyst_agent": {
                "AAPL": {
                    "signal": "bearish",
                    "confidence": 55,
                    "reasoning": omitted_reasoning,
                }
            },
            "valuation_analyst_agent": {
                "AAPL": {
                    "signal": "bullish",
                    "confidence": 80,
                    "reasoning": omitted_reasoning,
                }
            },
            "risk_management_agent": {
                "AAPL": {
                    "remaining_position_limit": 187.5,
                    "current_price": 202.25,
                    "volatility_metrics": {
                        "daily_volatility": 0.0123456789,
                        "annualized_volatility": 0.1959,
                        "volatility_percentile": 44.5,
                        "data_points": 60,
                    },
                    "correlation_metrics": {
                        "avg_correlation_with_active": None,
                        "max_correlation_with_active": None,
                        "top_correlated_tickers": [],
                    },
                    "reasoning": {
                        "portfolio_value": 1000,
                        "current_position_value": 0,
                        "base_position_limit_pct": 0.1875,
                        "correlation_multiplier": 1,
                        "combined_position_limit_pct": 0.1875,
                        "position_limit": 187.5,
                        "remaining_limit": 187.5,
                        "available_cash": 1000,
                        "risk_adjustment": "Volatility x Correlation adjusted: 18.8%",
                        "unbounded_detail": omitted_reasoning,
                    },
                    "credential": "must-not-be-persisted",
                }
            },
            "unknown_agent": {"AAPL": {"signal": "bullish", "confidence": 100}},
        },
    }

    actions, traces = AiHedgeFundRuntime.output_to_atl_actions_with_trace(
        output, _context()
    )

    assert actions == []
    diagnostics = traces[0]["diagnostics"]
    assert diagnostics["analyst_signals"] == {
        "fundamentals_analyst_agent": {
            "signal": "neutral",
            "confidence": 62.0,
        },
        "sentiment_analyst_agent": {
            "signal": "bearish",
            "confidence": 55.0,
        },
        "technical_analyst_agent": {
            "signal": "bullish",
            "confidence": 71.25,
        },
        "valuation_analyst_agent": {
            "signal": "bullish",
            "confidence": 80.0,
        },
    }
    assert diagnostics["risk_management_agent"] == {
        "remaining_position_limit": 187.5,
        "current_price": 202.25,
        "volatility_metrics": {
            "daily_volatility": 0.012346,
            "annualized_volatility": 0.1959,
            "volatility_percentile": 44.5,
            "data_points": 60,
        },
        "correlation_metrics": {
            "avg_correlation_with_active": None,
            "max_correlation_with_active": None,
            "top_correlated_tickers": [],
        },
        "reasoning": {
            "portfolio_value": 1000.0,
            "current_position_value": 0.0,
            "base_position_limit_pct": 0.1875,
            "correlation_multiplier": 1.0,
            "combined_position_limit_pct": 0.1875,
            "position_limit": 187.5,
            "remaining_limit": 187.5,
            "available_cash": 1000.0,
            "risk_adjustment": "Volatility x Correlation adjusted: 18.8%",
        },
    }
    assert diagnostics["portfolio_manager"] == {
        "action": "hold",
        "quantity": 0,
        "confidence": 88.0,
        "reasoning_summary": ("Portfolio Manager chose HOLD. " + "x" * 400)[:240],
    }
    serialized = json.dumps(diagnostics)
    assert "must-not-be-persisted" not in serialized
    assert len(diagnostics["portfolio_manager"]["reasoning_summary"]) == 240


@pytest.mark.parametrize(
    "output",
    [
        {},
        {"decisions": []},
        {
            "decisions": {
                "TSLA": {
                    "action": "hold",
                    "quantity": 0,
                    "confidence": 50,
                    "reasoning": "Outside universe",
                }
            }
        },
        {
            "decisions": {
                "AAPL": {
                    "action": "explode",
                    "quantity": 1,
                    "confidence": 50,
                    "reasoning": "Invalid action",
                }
            }
        },
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
            "AI_HEDGE_FUND_MODEL_NAME": "must-be-ignored",
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
    assert len(runtime.decision_audit_rows) == 2
    assert runtime.decision_audit_rows[0] == {
        "step_index": 0,
        "timestamp": "2026-05-01T14:00:00+00:00",
        "decision_source": "ai_hedge_fund",
        "actions_submitted": [],
        "actions_executed": 0,
        "context_ref": "2026-04-30",
    }
    payload, timeout = runner.payloads[0]
    assert timeout == 45
    assert payload["start_date"] == "2026-03-31"
    assert payload["end_date"] == "2026-04-30"
    assert payload["selected_analysts"] == ["technical_analyst"]
    assert payload["model_name"] == openrouter.DEFAULT_MODEL
    assert payload["model_provider"] == "OpenRouter"
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
            "OPENROUTER_API_KEY": "allowed-model-key",
            "OPENROUTER_BASE_URL": "must-not-cross-boundary",
            "FINANCIAL_DATASETS_API_KEY": "allowed-data-key",
            "OPENAI_API_KEY": "must-not-cross-boundary",
            "OPENAI_API_BASE": "must-not-cross-boundary",
            "ANTHROPIC_API_KEY": "must-not-cross-boundary",
            "ALPACA_API_KEY": "must-not-cross-boundary",
            "ALPACA_SECRET_KEY": "must-not-cross-boundary",
            "BROKER_TOKEN_ENCRYPTION_KEY": "must-not-cross-boundary",
            "CONTENT_DATABASE_URL": "must-not-cross-boundary",
            "DISCORD_CLIENT_SECRET": "must-not-cross-boundary",
        }
    )

    environment = runner._subprocess_environment()

    assert environment["OPENROUTER_API_KEY"] == "allowed-model-key"
    assert "OPENROUTER_BASE_URL" not in environment
    assert environment["FINANCIAL_DATASETS_API_KEY"] == "allowed-data-key"
    assert "OPENAI_API_KEY" not in environment
    assert "OPENAI_API_BASE" not in environment
    assert "CONTENT_DATABASE_URL" not in environment
    assert "DISCORD_CLIENT_SECRET" not in environment
    assert "ANTHROPIC_API_KEY" not in environment
    assert "ALPACA_API_KEY" not in environment
    assert "ALPACA_SECRET_KEY" not in environment
    assert "BROKER_TOKEN_ENCRYPTION_KEY" not in environment
    assert environment["PYTHON_DOTENV_DISABLED"] == "1"


def test_ai_hedge_fund_reuses_atl_managed_openrouter_model():
    runtime = AiHedgeFundRuntime(
        {"analysts": ["technical_analyst"]},
        runner=object(),
        environment={"AI_HEDGE_FUND_MODEL_NAME": "user-override"},
    )

    assert DEFAULT_MODEL_NAME == openrouter.DEFAULT_MODEL
    assert DEFAULT_MODEL_PROVIDER == "OpenRouter"
    assert runtime.model_name == openrouter.DEFAULT_MODEL


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"model_name": DEFAULT_MODEL_NAME, "model_provider": "OpenAI"},
        {"model_name": "another/model", "model_provider": "OpenRouter"},
    ],
)
def test_bridge_cannot_fall_back_from_managed_openrouter(payload):
    with pytest.raises(ValueError, match="platform-managed OpenRouter"):
        _managed_model_from_payload(payload)


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


def test_bridge_returns_upstream_analyst_signals(monkeypatch, tmp_path):
    fake_src = types.ModuleType("src")
    fake_main = types.ModuleType("src.main")
    captured = {}
    analyst_signals = {
        "technical_analyst_agent": {"AAPL": {"signal": "bullish", "confidence": 75}}
    }

    def fake_run_hedge_fund(**kwargs):
        captured.update(kwargs)
        return {
            "decisions": {
                "AAPL": {
                    "action": "hold",
                    "quantity": 0,
                    "confidence": 75,
                    "reasoning": "No trade",
                }
            },
            "analyst_signals": analyst_signals,
        }

    fake_main.run_hedge_fund = fake_run_hedge_fund
    fake_src.main = fake_main
    monkeypatch.setitem(sys.modules, "src", fake_src)
    monkeypatch.setitem(sys.modules, "src.main", fake_main)
    monkeypatch.setattr(bridge, "_disable_dotenv_loading", lambda: None)

    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(
        json.dumps(
            {
                "tickers": ["AAPL"],
                "start_date": "2026-02-03",
                "end_date": "2026-05-04",
                "portfolio": {"cash": 1000},
                "selected_analysts": ["technical_analyst"],
                "model_name": DEFAULT_MODEL_NAME,
                "model_provider": DEFAULT_MODEL_PROVIDER,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bridge.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )

    bridge.main()

    assert captured["model_name"] == DEFAULT_MODEL_NAME
    assert captured["model_provider"] == DEFAULT_MODEL_PROVIDER
    assert (
        json.loads(output_path.read_text(encoding="utf-8"))["analyst_signals"]
        == analyst_signals
    )


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


def test_pipeline_runtime_config_is_size_bounded():
    """Permissive about *which* keys must not mean unbounded in size.

    The column is plain TEXT, so without a ceiling any authenticated caller can
    PATCH megabytes onto every agent they own. The sibling ``pipeline`` field is
    capped at 50 steps for exactly this reason.
    """
    from dashboard.backend.domain.agents.runtime import MAX_RUNTIME_CONFIG_BYTES

    # A realistic pipeline knob still passes.
    assert normalize_runtime_config(
        PIPELINE_RUNTIME_TYPE, {"some_future_knob": "value"}
    ) == {"some_future_knob": "value"}

    oversized = {"junk": "A" * (MAX_RUNTIME_CONFIG_BYTES + 1)}
    with pytest.raises(ValueError, match="bytes of JSON"):
        normalize_runtime_config(PIPELINE_RUNTIME_TYPE, oversized)


def test_runtime_config_must_be_json_serializable():
    with pytest.raises(ValueError, match="JSON-serializable"):
        normalize_runtime_config(PIPELINE_RUNTIME_TYPE, {"bad": {1, 2, 3}})


def test_no_order_reason_falls_back_to_unknown_instead_of_raising():
    """An unlabelled audit trace must not be able to abort a whole backtest.

    ``_no_order_reason`` hand-mirrors ``actions_to_executable``'s filters. The
    day a filter is added there, raising here would discard the run over a
    missing string.
    """
    context = AgentRuntimeContext(
        timestamp=datetime(2026, 5, 4, 15, 0, tzinfo=timezone.utc),
        backtest_start_date="2026-05-01",
        symbols=["AAPL"],
        cash=1_000_000.0,
        total_equity=1_000_000.0,
        positions={"AAPL": 100},
        entry_prices={"AAPL": 10.0},
        current_prices={"AAPL": 10.0},
        latest_market_date_before_decision=date(2026, 5, 1),
    )
    # Every mirrored filter passes, so no known reason applies.
    assert (
        AiHedgeFundRuntime._no_order_reason(
            action="buy",
            quantity=1,
            confidence=0.9,
            context=context,
            symbol="AAPL",
        )
        == "unknown"
    )


def test_failed_decision_consumes_its_trading_day():
    """A failure must not be retried on every remaining hour of the same day.

    Marking the day only on the success path meant one bad day cost
    bars-per-day full timeouts and API spend for no new information.
    """

    class _AlwaysFails:
        calls = 0

        def run(self, payload, *, timeout_seconds):
            _AlwaysFails.calls += 1
            raise AiHedgeFundRuntimeError("upstream exploded")

    runtime = AiHedgeFundRuntime(
        {"analysts": ["technical_analyst"]},
        runner=_AlwaysFails(),
        environment={},
    )
    base = datetime(2026, 5, 4, 14, 30, tzinfo=timezone.utc)
    contexts = [
        AgentRuntimeContext(
            timestamp=base.replace(hour=hour),
            backtest_start_date="2026-05-01",
            symbols=["AAPL"],
            cash=10_000.0,
            total_equity=10_000.0,
            positions={},
            entry_prices={},
            current_prices={"AAPL": 10.0},
            latest_market_date_before_decision=date(2026, 5, 1),
        )
        for hour in (14, 15, 16, 17)
    ]

    with pytest.raises(AiHedgeFundRuntimeError):
        runtime.decide(contexts[0])
    # Remaining bars of the same day hold without re-invoking upstream.
    for context in contexts[1:]:
        assert runtime.decide(context) == {"actions": []}
    assert _AlwaysFails.calls == 1


def test_missing_interpreter_is_a_configuration_error(tmp_path):
    """Deployment-level failures are distinguishable from transient ones."""
    from dashboard.backend.domain.agents.runtime import (
        AgentRuntimeConfigurationError,
    )
    from dashboard.backend.infrastructure.ai_hedge_fund.adapter import (
        AiHedgeFundConfigurationError,
        runtime_unavailable_reason,
    )

    environment = {"AI_HEDGE_FUND_PYTHON": str(tmp_path / "nope" / "python")}
    runner = AiHedgeFundSubprocessRunner(environment)
    with pytest.raises(AiHedgeFundConfigurationError):
        runner._python_executable()
    # The engine catches the runtime-agnostic base, so the hierarchy matters.
    assert issubclass(AiHedgeFundConfigurationError, AgentRuntimeConfigurationError)

    reason = runtime_unavailable_reason(environment)
    assert reason and "AI_HEDGE_FUND_PYTHON" in reason


def test_step_timeout_is_read_once_and_shared():
    """One reading of the env var, shared by the parent and the child.

    Two independent readings is how the outer backtest timeout ends up smaller
    than the inner per-step budget and kills the run mid-flight.
    """
    from dashboard.backend.infrastructure.ai_hedge_fund.adapter import (
        AiHedgeFundConfigurationError,
        DEFAULT_TIMEOUT_SECONDS,
        MAX_TIMEOUT_SECONDS,
        resolve_step_timeout_seconds,
    )

    assert resolve_step_timeout_seconds({}) == DEFAULT_TIMEOUT_SECONDS
    assert resolve_step_timeout_seconds({"AI_HEDGE_FUND_TIMEOUT_SECONDS": "45"}) == 45

    runtime = AiHedgeFundRuntime(
        {"analysts": ["technical_analyst"]},
        runner=object(),
        environment={"AI_HEDGE_FUND_TIMEOUT_SECONDS": "45"},
    )
    assert runtime.timeout_seconds == 45

    for bad in ("abc", "0", str(MAX_TIMEOUT_SECONDS + 1)):
        with pytest.raises(AiHedgeFundConfigurationError):
            resolve_step_timeout_seconds({"AI_HEDGE_FUND_TIMEOUT_SECONDS": bad})


def test_upstream_error_text_cannot_forge_log_lines():
    """Upstream-controlled text reaches print() and persisted run metadata."""
    from dashboard.backend.infrastructure.ai_hedge_fund.adapter import (
        _redact_runtime_error,
    )

    forged = "boom\n2026-07-30 ERROR   fake log line\r\ndone"
    cleaned = _redact_runtime_error(forged, {})
    assert "\n" not in cleaned
    assert "\r" not in cleaned
    assert "fake log line" in cleaned  # content kept, framing removed

    redacted = _redact_runtime_error(
        "failed with key sk-secret-value",
        {"FINANCIAL_DATASETS_API_KEY": "sk-secret-value"},
    )
    assert "sk-secret-value" not in redacted
    assert "[REDACTED]" in redacted


def test_domain_runtime_module_does_not_import_any_adapter():
    """Keep the domain -> infrastructure arrow one-way.

    A lazy import inside __init__ defers an import cycle rather than removing
    it. Callers supply the concrete runtime instead.
    """
    import ast
    import pathlib

    # Resolved through a name this module already imports, rather than
    # importing the module a second time under another form.
    source = pathlib.Path(sys.modules[RuntimeDispatcher.__module__].__file__)
    tree = ast.parse(source.read_text(encoding="utf-8"))
    # Check import *statements*, not the word: the explanatory comments in that
    # module legitimately mention infrastructure.
    targets = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            targets.append(node.module)
        elif isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
    assert not [t for t in targets if "infrastructure" in t], targets

    with pytest.raises(Exception) as excinfo:
        RuntimeDispatcher(AI_HEDGE_FUND_RUNTIME_TYPE, {"analysts": ["ben_graham"]})
    assert "requires its runtime instance" in str(excinfo.value)
