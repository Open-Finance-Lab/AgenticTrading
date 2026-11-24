"""Hosted TradingAgents runtime adapter tests.

TradingAgents returns a five-tier *rating* and nothing else -- no quantity, no
confidence -- so this adapter is the only place those two numbers come into
existence. Everything downstream (``actions_to_executable``, the portfolio
manager, the equity curve) trusts them, which is why sizing and confidence
carry the bulk of the cases below.
"""

import importlib.util
import json
import os
import sys
from datetime import date, datetime, timezone

import pytest

from dashboard.backend.domain.agents.runtime import (
    TRADINGAGENTS_RUNTIME_TYPE,
    AgentRuntimeConfigurationError,
    AgentRuntimeContext,
    AgentRuntimeError,
)
from dashboard.backend.infrastructure.tradingagents import adapter, bridge
from dashboard.backend.infrastructure.tradingagents.adapter import (
    DEFAULT_MAX_POSITION_WEIGHT,
    DEFAULT_TIMEOUT_SECONDS,
    TradingAgentsConfigurationError,
    TradingAgentsOutputError,
    TradingAgentsRuntime,
    TradingAgentsRuntimeError,
    TradingAgentsSubprocessRunner,
    resolve_max_position_weight,
    resolve_step_timeout_seconds,
    runtime_unavailable_reason,
)


def _context(
    *,
    timestamp=None,
    latest_market_date_before_decision=date(2026, 3, 12),
    symbols=("AAPL",),
    cash=10_000.0,
    total_equity=10_000.0,
    positions=None,
    current_prices=None,
):
    return AgentRuntimeContext(
        timestamp=timestamp or datetime(2026, 3, 13, 15, tzinfo=timezone.utc),
        backtest_start_date="2026-03-01",
        symbols=list(symbols),
        cash=cash,
        total_equity=total_equity,
        positions=dict(positions or {}),
        entry_prices={"AAPL": 40.0},
        current_prices=dict(current_prices or {"AAPL": 50.0}),
        latest_market_date_before_decision=latest_market_date_before_decision,
    )


def _bridge_output(*, rating="Buy", action="BUY", status="valid"):
    return {
        "schema_version": "tradingagents-atl-v1",
        "manifest": {
            "symbol": "AAPL",
            "tradingagents_version": "0.3.1",
            "selected_analysts": ["market", "news"],
            "safe_config_sha256": "0" * 64,
        },
        "decision": {
            "analysis_date": "2026-03-12",
            "rating": rating,
            "atl_action": action,
            "status": status,
            "attempts": 1,
            "raw_sha256": "a" * 64,
            "raw_excerpt": "FINAL TRANSACTION PROPOSAL",
        },
    }


class _FakeRunner:
    """Records what the runtime asked upstream for, without a subprocess."""

    def __init__(self, output=None, error=None):
        self.output = output if output is not None else _bridge_output()
        self.error = error
        self.payloads = []
        self.timeouts = []

    def run(self, payload, *, timeout_seconds):
        self.payloads.append(payload)
        self.timeouts.append(timeout_seconds)
        if self.error is not None:
            raise self.error
        return self.output


# The default roster is the only as-of-safe one (see
# ``test_tradingagents_lookahead.py``); the product refuses the others in a
# historical backtest, so a test that configured one would be exercising a
# roster that cannot reach production.
def _runtime(runner=None, *, environment=None, analysts=("market",)):
    return TradingAgentsRuntime(
        {"analysts": list(analysts)},
        runner=runner if runner is not None else _FakeRunner(),
        environment=environment if environment is not None else {},
    )


# ---------------------------------------------------------------------------
# When the runtime may spend money at all
# ---------------------------------------------------------------------------


def test_runtime_decides_once_per_trading_day():
    """Every hourly bar reaches decide(); without the guard a single day would
    buy a full multi-agent debate per bar."""
    runner = _FakeRunner()
    runtime = _runtime(runner)

    runtime.decide(_context(timestamp=datetime(2026, 3, 13, 15, tzinfo=timezone.utc)))
    second = runtime.decide(
        _context(timestamp=datetime(2026, 3, 13, 16, tzinfo=timezone.utc))
    )

    assert len(runner.payloads) == 1
    assert second == {"actions": []}


def test_runtime_consumes_the_day_even_when_upstream_fails():
    """Marking the day only on success makes one bad day cost bars-per-day full
    timeouts, because every later bar retries it."""
    runner = _FakeRunner(error=RuntimeError("upstream down"))
    runtime = _runtime(runner)

    with pytest.raises(RuntimeError):
        runtime.decide(
            _context(timestamp=datetime(2026, 3, 13, 15, tzinfo=timezone.utc))
        )
    later = runtime.decide(
        _context(timestamp=datetime(2026, 3, 13, 16, tzinfo=timezone.utc))
    )

    assert later == {"actions": []}
    assert len(runner.payloads) == 1


def test_runtime_holds_when_no_earlier_market_date_exists():
    """The first trading date of a run has no strictly earlier market date;
    inventing one would hand TradingAgents same-day data."""
    runner = _FakeRunner()
    runtime = _runtime(runner)

    decision = runtime.decide(_context(latest_market_date_before_decision=None))

    assert decision == {"actions": []}
    assert runner.payloads == []


def test_runtime_refuses_a_multi_symbol_context_without_calling_upstream():
    runner = _FakeRunner()
    runtime = _runtime(runner)

    with pytest.raises(AgentRuntimeConfigurationError):
        runtime.decide(_context(symbols=("AAPL", "MSFT")))
    assert runner.payloads == []


# ---------------------------------------------------------------------------
# What crosses to the bridge
# ---------------------------------------------------------------------------


def test_runtime_asks_for_one_date_and_it_is_the_data_cutoff():
    runner = _FakeRunner()
    runtime = _runtime(runner)

    runtime.decide(_context(latest_market_date_before_decision=date(2026, 3, 12)))

    payload = runner.payloads[0]
    assert payload["analysis_dates"] == ["2026-03-12"]
    assert payload["symbol"] == "AAPL"
    assert payload["selected_analysts"] == ["market"]


def test_runtime_pins_provider_and_models_rather_than_passing_agent_config():
    runner = _FakeRunner()
    runtime = _runtime(runner)

    runtime.decide(_context())

    config = runner.payloads[0]["config"]
    assert config["llm_provider"] == "openrouter"
    assert config["deep_think_llm"] == runtime.model_name
    assert config["quick_think_llm"] == runtime.model_name


def test_runtime_payload_is_json_serializable():
    """It is written to a file for the child process; a stray date object would
    only fail once a real subprocess ran."""
    runner = _FakeRunner()
    runtime = _runtime(runner)

    runtime.decide(_context())

    json.dumps(runner.payloads[0])


# ---------------------------------------------------------------------------
# Sizing: the numbers TradingAgents does not provide
# ---------------------------------------------------------------------------


def test_buy_sizes_the_order_to_the_platform_position_weight():
    """10_000 equity x 0.2 weight / 50.0 price = 40 shares."""
    runtime = _runtime(_FakeRunner(_bridge_output(rating="Buy", action="BUY")))

    actions = runtime.decide(_context())["actions"]

    assert [(a["symbol"], a["action"], a["shares"]) for a in actions] == [
        ("AAPL", "buy", 40)
    ]


def test_buy_tops_up_to_the_target_instead_of_re_buying_it():
    runtime = _runtime(_FakeRunner(_bridge_output(rating="Buy", action="BUY")))

    actions = runtime.decide(_context(positions={"AAPL": 15}))["actions"]

    assert [a["shares"] for a in actions] == [25]


def test_buy_emits_no_order_once_the_target_weight_is_held():
    runtime = _runtime(_FakeRunner(_bridge_output(rating="Buy", action="BUY")))

    actions = runtime.decide(_context(positions={"AAPL": 40}))["actions"]

    assert actions == []


def test_buy_emits_no_order_when_one_share_exceeds_the_position_budget():
    runtime = _runtime(_FakeRunner(_bridge_output(rating="Buy", action="BUY")))

    actions = runtime.decide(_context(current_prices={"AAPL": 2_500.0}))["actions"]

    assert actions == []


def test_sell_closes_the_whole_position():
    runtime = _runtime(_FakeRunner(_bridge_output(rating="Sell", action="SELL")))

    actions = runtime.decide(_context(positions={"AAPL": 12}))["actions"]

    assert [(a["action"], a["shares"]) for a in actions] == [("sell", 12)]


def test_sell_without_a_position_emits_no_order():
    runtime = _runtime(_FakeRunner(_bridge_output(rating="Sell", action="SELL")))

    actions = runtime.decide(_context(positions={}))["actions"]

    assert actions == []


def test_hold_emits_no_order():
    runtime = _runtime(_FakeRunner(_bridge_output(rating="Hold", action="HOLD")))

    actions = runtime.decide(_context(positions={"AAPL": 10}))["actions"]

    assert actions == []


def test_missing_price_emits_no_order():
    runtime = _runtime(_FakeRunner(_bridge_output(rating="Buy", action="BUY")))

    actions = runtime.decide(_context(current_prices={"AAPL": 0.0}))["actions"]

    assert actions == []


@pytest.mark.parametrize(
    "rating,expected_confidence",
    [
        ("Buy", 0.85),
        ("Sell", 0.85),
        ("Overweight", 0.55),
        ("Underweight", 0.55),
    ],
)
def test_conviction_tier_survives_the_collapse_to_three_directions(
    rating, expected_confidence
):
    """RATING_TO_ACTION folds five tiers into BUY/HOLD/SELL. The strength it
    discards is the only confidence signal TradingAgents emits, so a flat
    constant here would throw away real information."""
    action = "BUY" if rating in ("Buy", "Overweight") else "SELL"
    runtime = _runtime(_FakeRunner(_bridge_output(rating=rating, action=action)))

    runtime.decide(_context(positions={"AAPL": 5}))

    trace = runtime.decision_audit_rows[0]["actions_submitted"][0]
    assert trace["confidence"] == pytest.approx(expected_confidence * 100)


# ---------------------------------------------------------------------------
# Untrusted output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "output",
    [
        {},
        {"decision": {}},
        {"decision": {"atl_action": "SHORT", "rating": "Buy", "status": "valid"}},
        {"decision": {"atl_action": "BUY", "rating": "Maybe", "status": "valid"}},
        {"decision": "BUY"},
    ],
)
def test_output_that_is_not_a_tradingagents_decision_is_refused(output):
    runtime = _runtime(_FakeRunner(output))

    with pytest.raises(TradingAgentsOutputError):
        runtime.decide(_context())


def test_audit_row_keeps_the_digest_and_excerpt_but_never_a_transcript():
    runtime = _runtime(_FakeRunner())

    runtime.decide(_context())

    row = runtime.decision_audit_rows[0]
    assert row["decision_source"] == TRADINGAGENTS_RUNTIME_TYPE
    assert row["context_ref"] == "2026-03-12"
    diagnostics = row["actions_submitted"][0]["diagnostics"]
    assert diagnostics["raw_sha256"] == "a" * 64
    assert diagnostics["tradingagents_version"] == "0.3.1"
    assert "raw_final_trade_decision" not in json.dumps(row)


# ---------------------------------------------------------------------------
# Metering
# ---------------------------------------------------------------------------


def test_runtime_counts_the_calls_the_run_is_billed_for():
    """The engine folds ``runtime.calls`` into ``agent_runs.llm_calls``. A run
    that spent real money while reporting zero calls is exactly what the H6
    integrity guard reads as a rule-based curve wearing a model's name."""
    runtime = _runtime(_FakeRunner())

    runtime.decide(_context(timestamp=datetime(2026, 3, 13, 15, tzinfo=timezone.utc)))
    runtime.decide(_context(timestamp=datetime(2026, 3, 16, 15, tzinfo=timezone.utc)))

    assert runtime.calls == 2


def test_held_days_do_not_count_as_billed_calls():
    runtime = _runtime(_FakeRunner())

    runtime.decide(_context(latest_market_date_before_decision=None))

    assert runtime.calls == 0


def test_executed_count_is_attached_to_the_latest_call():
    runtime = _runtime(_FakeRunner())

    runtime.decide(_context())
    runtime.record_latest_execution(1)

    assert runtime.decision_audit_rows[0]["actions_executed"] == 1


# ---------------------------------------------------------------------------
# Platform-owned settings
# ---------------------------------------------------------------------------


def test_position_weight_is_platform_owned_with_a_conservative_default():
    assert DEFAULT_MAX_POSITION_WEIGHT == 0.2
    assert resolve_max_position_weight({}) == 0.2
    assert (
        resolve_max_position_weight({"TRADINGAGENTS_MAX_POSITION_WEIGHT": "0.5"}) == 0.5
    )


@pytest.mark.parametrize("raw", ["0", "-0.1", "1.5", "abc"])
def test_an_unusable_position_weight_stops_the_run_instead_of_guessing(raw):
    with pytest.raises(TradingAgentsConfigurationError):
        resolve_max_position_weight({"TRADINGAGENTS_MAX_POSITION_WEIGHT": raw})


def test_step_timeout_allows_a_full_multi_agent_debate_by_default():
    """A TradingAgents day is a multi-agent debate plus one retry, far longer
    than AI Hedge Fund's 300s single pass."""
    assert DEFAULT_TIMEOUT_SECONDS == 600
    assert resolve_step_timeout_seconds({}) == 600
    assert resolve_step_timeout_seconds({"TRADINGAGENTS_TIMEOUT_SECONDS": "900"}) == 900


@pytest.mark.parametrize("raw", ["0", "901", "soon"])
def test_an_unusable_step_timeout_stops_the_run_instead_of_guessing(raw):
    with pytest.raises(TradingAgentsConfigurationError):
        resolve_step_timeout_seconds({"TRADINGAGENTS_TIMEOUT_SECONDS": raw})


def test_runtime_passes_the_platform_timeout_to_the_subprocess():
    runner = _FakeRunner()
    runtime = TradingAgentsRuntime(
        {"analysts": ["market"]},
        runner=runner,
        environment={"TRADINGAGENTS_TIMEOUT_SECONDS": "750"},
    )

    runtime.decide(_context())

    assert runner.timeouts == [750]


# ---------------------------------------------------------------------------
# Subprocess isolation
# ---------------------------------------------------------------------------


def test_subprocess_environment_carries_the_platform_key_and_nothing_unrelated():
    runner = TradingAgentsSubprocessRunner(
        {
            "PATH": "/usr/bin",
            "OPENROUTER_API_KEY": "platform-key",
            "FRED_API_KEY": "fred-key",
            "ALPACA_API_KEY": "broker-key",
            "USERS_DATABASE_URL": "postgres://user:pw@host/db",
            "BROKER_TOKEN_ENCRYPTION_KEY": "encryption-key",
            "ANTHROPIC_API_KEY": "other-provider",
        }
    )

    environment = runner._subprocess_environment("/tmp/home")

    assert environment["OPENROUTER_API_KEY"] == "platform-key"
    assert environment["FRED_API_KEY"] == "fred-key"
    assert "ALPACA_API_KEY" not in environment
    assert "USERS_DATABASE_URL" not in environment
    assert "BROKER_TOKEN_ENCRYPTION_KEY" not in environment
    assert "ANTHROPIC_API_KEY" not in environment
    assert environment["HOME"] == "/tmp/home"
    assert environment["PYTHON_DOTENV_DISABLED"] == "1"


def test_subprocess_keeps_the_windows_socket_stack_reachable():
    """Windows resolves the socket service provider through SystemRoot. Drop it
    and every networked child dies with WinError 10106 before it can make a
    single request -- which on a developer machine looks like an upstream
    outage rather than a missing environment variable."""
    runner = TradingAgentsSubprocessRunner(
        {"PATH": "/usr/bin", "SystemRoot": "C:\\Windows", "WINDIR": "C:\\Windows"}
    )

    environment = runner._subprocess_environment("/tmp/home")

    assert environment["SystemRoot"] == "C:\\Windows"
    assert environment["WINDIR"] == "C:\\Windows"


def test_subprocess_can_import_the_sdk_the_bridge_reuses():
    """The bridge calls ``TradingAgentsDecisionGenerator`` from the SDK source
    tree, which is not installed into the isolated environment."""
    runner = TradingAgentsSubprocessRunner({"PATH": "/usr/bin"})

    environment = runner._subprocess_environment("/tmp/home")

    assert "agentictrading" in environment["PYTHONPATH"]


def test_missing_interpreter_is_a_configuration_error_not_a_step_failure():
    """It fails identically on every step, so retrying it once per trading day
    only turns one clear error into a long silent all-holds run."""
    runner = TradingAgentsSubprocessRunner(
        {"TRADINGAGENTS_PYTHON": "/nonexistent/python"}
    )

    with pytest.raises(TradingAgentsConfigurationError):
        runner._python_executable()


def test_runtime_failure_text_never_repeats_a_platform_credential():
    from dashboard.backend.infrastructure.tradingagents.adapter import (
        _redact_runtime_error,
    )

    redacted = _redact_runtime_error(
        "upstream said 401 for sk-or-v1-topsecret\nsecond line",
        {"OPENROUTER_API_KEY": "sk-or-v1-topsecret"},
    )

    assert "sk-or-v1-topsecret" not in redacted
    assert "\n" not in redacted


def test_hosted_model_matches_the_model_atl_has_registered():
    """The bridge pins the model it will accept and ATL registers the model it
    pays for. If those drift, every hosted run fails closed in the subprocess
    with a message no operator would connect to a model-catalog change."""
    from dashboard.backend.infrastructure.llm.providers import openrouter

    assert _runtime().model_name == openrouter.DEFAULT_MODEL


# ---------------------------------------------------------------------------
# The adapter/bridge seam, driven through a real subprocess
# ---------------------------------------------------------------------------


def _real_subprocess_runner():
    """A runner whose interpreter is this one, so the real bridge really runs.

    Everything between the two processes is exercised for real: the JSON files,
    the environment allowlist, the SDK path handed over on PYTHONPATH, the exit
    code, and the redaction applied to whatever the child printed.
    """
    return TradingAgentsSubprocessRunner(
        {
            "PATH": os.environ.get("PATH", ""),
            "TRADINGAGENTS_PYTHON": sys.executable,
        }
    )


def _real_payload(**overrides):
    from dashboard.backend.infrastructure.tradingagents import bridge

    payload = {
        "schema": "tradingagents-bridge-v1",
        "symbol": "AAPL",
        "analysis_dates": ["2026-03-12"],
        "selected_analysts": ["market"],
        "config": {
            "llm_provider": "openrouter",
            "deep_think_llm": bridge.MANAGED_DEEP_THINK_MODEL,
            "quick_think_llm": bridge.MANAGED_QUICK_THINK_MODEL,
        },
    }
    payload.update(overrides)
    return payload


def test_real_subprocess_refuses_an_unpinned_model_before_spending_anything():
    """The fail-closed check has to hold across the process boundary, not just
    when the bridge is called in-process by its unit tests."""
    runner = _real_subprocess_runner()
    payload = _real_payload(
        config={
            "llm_provider": "openrouter",
            "deep_think_llm": "someone-elses/model",
            "quick_think_llm": "someone-elses/model",
        }
    )

    with pytest.raises(Exception) as exc:
        runner.run(payload, timeout_seconds=120)

    message = str(exc.value)
    assert "platform-managed" in message
    assert "Traceback" not in message


@pytest.mark.skipif(
    importlib.util.find_spec("tradingagents") is not None,
    reason="a real TradingAgents install would attempt a billable LLM call",
)
def test_real_subprocess_reaches_the_sdk_and_reports_a_missing_upstream():
    """A valid payload must travel all the way into the SDK generator.

    Without ``tradingagents`` installed the generator stops at its own
    dependency guard, which is the furthest this can go without spending money
    -- and it proves the child could import ``agentictrading`` from the source
    tree the runner put on PYTHONPATH.
    """
    runner = _real_subprocess_runner()

    with pytest.raises(Exception) as exc:
        runner.run(_real_payload(), timeout_seconds=120)

    message = str(exc.value)
    assert "TradingAgents is required" in message
    assert "Traceback" not in message


# ---------------------------------------------------------------------------
# Deployment readiness
# ---------------------------------------------------------------------------


def test_availability_names_the_interpreter_the_deployment_is_missing():
    reason = runtime_unavailable_reason({"TRADINGAGENTS_PYTHON": "/nonexistent/python"})

    assert reason is not None
    assert "TRADINGAGENTS_PYTHON" in reason


def test_availability_reports_an_unreachable_sdk_source_tree(monkeypatch, tmp_path):
    """The bridge imports the decision generator from the repository's own SDK
    sources, which the adapter hands over on PYTHONPATH. A deployment that
    ships only ``dashboard/`` has an interpreter and a bridge but nothing for
    the child to import -- and without this check it would discover that one
    step at a time, inside a subprocess, minutes into a run.
    """
    from dashboard.backend.infrastructure.tradingagents import adapter

    monkeypatch.setattr(adapter, "_SDK_SOURCE_DIR", tmp_path / "absent")

    reason = runtime_unavailable_reason({"TRADINGAGENTS_PYTHON": sys.executable})

    assert reason is not None
    assert "agentictrading" in reason


def test_availability_is_silent_when_the_deployment_can_host_the_runtime():
    """The negative cases above only mean something if the healthy one passes;
    otherwise the probe could be refusing every deployment."""
    assert runtime_unavailable_reason({"TRADINGAGENTS_PYTHON": sys.executable}) is None


def test_error_classes_carry_the_contract_the_engine_dispatches_on():
    """The engine holds the step on ``AgentRuntimeError`` and aborts the run on
    ``AgentRuntimeConfigurationError``. Neither decision is observable from
    outside the engine -- an error that inherits from neither simply escapes
    ``run_agent_backtest``, which looks identical to an abort while silently
    discarding every completed step. So the hierarchy is asserted here, where
    breaking it is visible.
    """
    assert issubclass(TradingAgentsRuntimeError, AgentRuntimeError)
    assert issubclass(TradingAgentsOutputError, AgentRuntimeError)
    assert issubclass(TradingAgentsConfigurationError, AgentRuntimeConfigurationError)
    # And the configuration error must remain a runtime error, so a caller that
    # only knows the transient base still catches the fatal one.
    assert issubclass(TradingAgentsConfigurationError, TradingAgentsRuntimeError)


# ---------------------------------------------------------------------------
# Relocating the SDK sources
# ---------------------------------------------------------------------------


def test_sdk_sources_can_be_relocated_by_the_deployment(tmp_path):
    """``TRADINGAGENTS_ROOT`` is this runtime's ``AI_HEDGE_FUND_ROOT``: a
    directory the child gets on PYTHONPATH. A deployment that vendors the SDK
    somewhere other than the repository layout has no other way to say so, and
    the alternative -- hard-coding the repo path -- makes the image layout part
    of the adapter.
    """
    (tmp_path / "agentictrading").mkdir()
    runner = TradingAgentsSubprocessRunner(
        {"PATH": "/usr/bin", "TRADINGAGENTS_ROOT": str(tmp_path)}
    )

    environment = runner._subprocess_environment("/tmp/home")

    assert environment["PYTHONPATH"] == str(tmp_path)


def test_availability_probes_the_relocated_sdk_sources(tmp_path):
    """The probe has to follow the override, or a correctly relocated
    deployment is refused and a broken one is accepted."""
    populated = tmp_path / "vendored"
    (populated / "agentictrading").mkdir(parents=True)
    empty = tmp_path / "empty"
    empty.mkdir()

    assert (
        runtime_unavailable_reason(
            {
                "TRADINGAGENTS_PYTHON": sys.executable,
                "TRADINGAGENTS_ROOT": str(populated),
            }
        )
        is None
    )
    reason = runtime_unavailable_reason(
        {"TRADINGAGENTS_PYTHON": sys.executable, "TRADINGAGENTS_ROOT": str(empty)}
    )
    assert reason is not None
    assert "agentictrading" in reason


# ---------------------------------------------------------------------------
# Gateway selection and credential mapping
# ---------------------------------------------------------------------------

def test_commonstack_is_preferred_when_its_platform_key_is_present():
    """Mirrors the convention the rest of ATL already follows (chat and the
    leaderboard both prefer CommonStack when its key is set)."""
    assert (
        adapter.select_managed_gateway({"COMMONSTACK_API_KEY": "x"}) == "commonstack"
    )


def test_openrouter_is_used_when_only_its_key_is_present():
    assert (
        adapter.select_managed_gateway({"OPENROUTER_API_KEY": "x"}) == "openrouter"
    )


def test_an_explicit_gateway_override_wins_over_autodetect():
    chosen = adapter.select_managed_gateway(
        {"COMMONSTACK_API_KEY": "x", "TRADINGAGENTS_GATEWAY": "openrouter"}
    )
    assert chosen == "openrouter"


def test_an_unknown_gateway_override_is_a_configuration_error():
    with pytest.raises(adapter.TradingAgentsConfigurationError):
        adapter.select_managed_gateway({"TRADINGAGENTS_GATEWAY": "ollama"})


def test_the_payload_carries_the_selected_gateway_not_the_default():
    runtime = adapter.TradingAgentsRuntime(
        {"analysts": ["market"]},
        environment={"COMMONSTACK_API_KEY": "x"},
    )
    payload = runtime._bridge_payload(_context(), "AAPL")
    expected = bridge.MANAGED_GATEWAYS["commonstack"]
    assert payload["config"]["llm_provider"] == expected["llm_provider"]
    assert payload["config"]["deep_think_llm"] == expected["deep_think_llm"]


def test_commonstack_key_reaches_the_child_under_the_name_upstream_reads():
    """Upstream's generic endpoint reads OPENAI_COMPATIBLE_API_KEY; ATL stores
    the same secret as COMMONSTACK_API_KEY. Without the rename the child
    authenticates as nobody and the run dies on a 401."""
    runner = adapter.TradingAgentsSubprocessRunner(
        {"COMMONSTACK_API_KEY": "cs-secret"}
    )
    environment = runner._subprocess_environment()
    assert environment["OPENAI_COMPATIBLE_API_KEY"] == "cs-secret"


def test_openrouter_only_deployments_do_not_gain_a_compatible_key():
    """The rename is scoped to the CommonStack credential: an OpenRouter-only
    deployment must not start sending a key to the generic endpoint."""
    runner = adapter.TradingAgentsSubprocessRunner({"OPENROUTER_API_KEY": "or-secret"})
    environment = runner._subprocess_environment()
    assert "OPENAI_COMPATIBLE_API_KEY" not in environment
    assert environment["OPENROUTER_API_KEY"] == "or-secret"


# ---------------------------------------------------------------------------
# Experiment provenance and credential scoping
# ---------------------------------------------------------------------------

def test_audit_row_records_both_quick_and_deep_models():
    """A hosted run's record used to name only the deep model -- the model just
    2 of upstream's 11 LLM calls actually use. Comparing experiments across a
    quick/deep split needs both, and they are read from the manifest the bridge
    returned: what upstream really ran, not what ATL intended."""
    output = _bridge_output()
    output["manifest"]["quick_think_llm"] = "qwen/qwen3.7-plus"
    output["manifest"]["deep_think_llm"] = "deepseek/deepseek-v4-pro"
    runtime = _runtime(_FakeRunner(output))

    runtime.decide(_context())

    diagnostics = runtime.decision_audit_rows[0]["actions_submitted"][0][
        "diagnostics"
    ]
    assert diagnostics["quick_think_llm"] == "qwen/qwen3.7-plus"
    assert diagnostics["deep_think_llm"] == "deepseek/deepseek-v4-pro"


def test_audit_row_tolerates_a_manifest_without_model_fields():
    """The fields are allow-listed, not required: a bridge that does not echo
    them must still produce a writable row rather than raising."""
    runtime = _runtime(_FakeRunner(_bridge_output()))

    runtime.decide(_context())

    diagnostics = runtime.decision_audit_rows[0]["actions_submitted"][0][
        "diagnostics"
    ]
    assert diagnostics["quick_think_llm"] == ""
    assert diagnostics["deep_think_llm"] == ""


def test_only_the_selected_gateways_credential_reaches_the_child():
    """Both platform keys can sit in the service environment, but the child may
    receive only the one funding the selected gateway. Upstream maps its
    ``openrouter`` provider to OPENROUTER_API_KEY, so leaving that key in a
    CommonStack child's environment lets any openrouter code path spend the
    wrong account."""
    runner = adapter.TradingAgentsSubprocessRunner(
        {
            "COMMONSTACK_API_KEY": "cs-secret",
            "OPENROUTER_API_KEY": "or-secret",
            "TRADINGAGENTS_GATEWAY": "commonstack",
        }
    )

    environment = runner._subprocess_environment()

    assert environment["OPENAI_COMPATIBLE_API_KEY"] == "cs-secret"
    assert "OPENROUTER_API_KEY" not in environment
    # Upstream never reads this name; dropping it keeps the surface minimal.
    assert "COMMONSTACK_API_KEY" not in environment


def test_openrouter_child_does_not_receive_the_commonstack_key():
    runner = adapter.TradingAgentsSubprocessRunner(
        {
            "COMMONSTACK_API_KEY": "cs-secret",
            "OPENROUTER_API_KEY": "or-secret",
            "TRADINGAGENTS_GATEWAY": "openrouter",
        }
    )

    environment = runner._subprocess_environment()

    assert environment["OPENROUTER_API_KEY"] == "or-secret"
    assert "COMMONSTACK_API_KEY" not in environment
    assert "OPENAI_COMPATIBLE_API_KEY" not in environment


# ---------------------------------------------------------------------------
# Run parameters an experiment has to be able to read back
# ---------------------------------------------------------------------------

def test_payload_pins_the_debate_structure_instead_of_inheriting_it():
    """These two values decide how many LLM calls one decision costs (the graph
    exits on ``count >= 2 * debate`` and ``count >= 3 * risk``). Leaving them to
    upstream's DEFAULT_CONFIG means an upstream bump could double both runtime
    and spend with nothing in ATL's records to show it."""
    runtime = _runtime(runner := _FakeRunner())

    runtime.decide(_context())

    config = runner.payloads[0]["config"]
    assert config["max_debate_rounds"] == bridge.MANAGED_MAX_DEBATE_ROUNDS
    assert config["max_risk_discuss_rounds"] == (
        bridge.MANAGED_MAX_RISK_DISCUSS_ROUNDS
    )


def test_audit_row_records_the_run_parameters_not_just_their_hash():
    """``safe_config_sha256`` can only say two runs differed, never what
    differed. Comparing experiments needs the values themselves."""
    output = _bridge_output()
    output["manifest"].update(
        {
            "llm_provider": "openai_compatible",
            "selected_analysts": ["market", "news"],
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 1,
        }
    )
    runtime = _runtime(_FakeRunner(output))

    runtime.decide(_context())

    diagnostics = runtime.decision_audit_rows[0]["actions_submitted"][0][
        "diagnostics"
    ]
    assert diagnostics["llm_provider"] == "openai_compatible"
    assert diagnostics["selected_analysts"] == ["market", "news"]
    assert diagnostics["max_debate_rounds"] == 1
    assert diagnostics["max_risk_discuss_rounds"] == 1


def test_audit_row_run_parameters_tolerate_a_manifest_without_them():
    runtime = _runtime(_FakeRunner(_bridge_output()))

    runtime.decide(_context())

    diagnostics = runtime.decision_audit_rows[0]["actions_submitted"][0][
        "diagnostics"
    ]
    assert diagnostics["llm_provider"] == ""
    # The default fixture's manifest does carry this one; the point here is
    # that the fields absent from it degrade rather than raise.
    assert diagnostics["selected_analysts"] == ["market", "news"]
    assert diagnostics["max_debate_rounds"] is None
    assert diagnostics["max_risk_discuss_rounds"] is None


def test_audit_row_refuses_a_hostile_analyst_list():
    """The manifest is upstream-controlled text. A crafted list must not be
    able to grow the row without bound or smuggle non-strings into it."""
    output = _bridge_output()
    output["manifest"]["selected_analysts"] = ["market"] * 50 + [{"x": 1}]
    runtime = _runtime(_FakeRunner(output))

    runtime.decide(_context())

    analysts = runtime.decision_audit_rows[0]["actions_submitted"][0][
        "diagnostics"
    ]["selected_analysts"]
    assert len(analysts) <= 8
    assert all(isinstance(item, str) for item in analysts)


def test_the_deployment_can_redirect_where_upstream_writes_its_trajectory():
    """Upstream writes its full state log under ``results_dir``, which defaults
    to ``$HOME/.tradingagents/logs`` -- and the adapter points the child's HOME
    at the TemporaryDirectory that is deleted the moment the call returns, so
    the trajectory is lost by construction, not by accident.

    ``TRADINGAGENTS_RESULTS_DIR`` is upstream's own escape hatch. Forwarding it
    lets the *deployment* keep the trajectory; it does not let a payload
    redirect writes, because ``results_dir`` is still not an allowed config
    key (pinned separately in test_bridge.py)."""
    runner = adapter.TradingAgentsSubprocessRunner(
        {"TRADINGAGENTS_RESULTS_DIR": "/srv/atl/trajectories"}
    )

    environment = runner._subprocess_environment()

    assert environment["TRADINGAGENTS_RESULTS_DIR"] == "/srv/atl/trajectories"


def test_audit_row_records_the_exact_upstream_commit():
    """The version string cannot distinguish two builds of the same tag, so the
    commit is what makes a past run reproducible."""
    output = _bridge_output()
    output["upstream_commit"] = "a33fd4c0f134485a43553a2c23a63cb14adbd88f"
    runtime = _runtime(_FakeRunner(output))

    runtime.decide(_context())

    diagnostics = runtime.decision_audit_rows[0]["actions_submitted"][0][
        "diagnostics"
    ]
    assert diagnostics["upstream_commit"] == (
        "a33fd4c0f134485a43553a2c23a63cb14adbd88f"
    )


def test_audit_row_tolerates_a_bridge_that_reports_no_commit():
    runtime = _runtime(_FakeRunner(_bridge_output()))

    runtime.decide(_context())

    diagnostics = runtime.decision_audit_rows[0]["actions_submitted"][0][
        "diagnostics"
    ]
    assert diagnostics["upstream_commit"] == ""
