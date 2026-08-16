"""H6 — leaderboard must not publish a rule-based fallback under an LLM name.

``deploy_model_run`` refuses to persist an LLM entry that silently fell back to
rule-based trading (no client, or a model id the gateway rejected so every call
failed), unless ``allow_fallback=True``. Rule-based baselines (which expose no
``used_llm``) are unaffected.
"""

import pytest

from dashboard.backend.database import BacktestDatabase
from dashboard.backend.domain.leaderboard import service as canon_service
from dashboard.backend.infrastructure.llm import backtest_harness as llm_harness


_CONFIG = {
    "session_id": "lb-guard-test",
    "start_date": "2026-04-15",
    "end_date": "2026-05-15",
    "initial_capital": 100000,
    "strategies": [
        {"id": "claude_haiku_4_5", "name": "Haiku", "model": "Claude Haiku",
         "strategy": "llm_agent", "model_id": "test-model"},
        {"id": "djia_index", "name": "DJIA", "model": "DJIA", "strategy": "market_index"},
    ],
}


class FakeLLMStrategy:
    """Mimics LLMAgentStrategy's reporting surface (exposes ``used_llm``).

    ``decision_steps`` is the number of decision points in the run; when omitted
    it defaults to ``llm_calls`` (i.e. 100% LLM coverage), so existing tests that
    only care about the used_llm/llm_calls axis stay at full coverage.

    ``llm_decisions`` is how many steps produced a *usable* model decision (the
    H6 coverage numerator); when omitted it defaults to ``llm_calls`` — the
    common case where every billed call yielded a usable decision.
    """

    def __init__(self, *, used_llm, llm_calls, decision_steps=None,
                 llm_decisions=None, report_decisions=True, model_id="test-model",
                 strategy_prompt=None):
        self.strategy_prompt = strategy_prompt
        self.used_llm = used_llm
        self.llm_calls = llm_calls
        # An older strategy shape may not report llm_decisions at all; omit the
        # attribute entirely so getattr on the guard side has to default it.
        if report_decisions:
            self.llm_decisions = llm_calls if llm_decisions is None else llm_decisions
        self.decision_steps = llm_calls if decision_steps is None else decision_steps
        self.input_tokens = 10
        self.output_tokens = 5
        self.model_id = model_id

    def required_symbols(self):
        return ["AAPL"]

    def run(self, bars, start, end, capital):
        return [{"timestamp": "2026-04-15T14:00:00", "equity": capital,
                 "cash": 0, "positions_value": capital}]

    def num_trades(self):
        return 0


class FakeBaseline:
    """A rule-based baseline: intentionally exposes NO ``used_llm`` attribute."""

    def __init__(self):
        self.llm_calls = 0
        self.model_id = None
        self.input_tokens = 0
        self.output_tokens = 0

    def required_symbols(self):
        return ["AAPL"]

    def run(self, bars, start, end, capital):
        return [{"timestamp": "2026-04-15T14:00:00", "equity": capital,
                 "cash": 0, "positions_value": capital}]

    def num_trades(self):
        return 0


@pytest.fixture
def guard_env(tmp_path, monkeypatch):
    test_db = BacktestDatabase(db_path=tmp_path / "lb.db")
    monkeypatch.setattr(canon_service, "db", test_db)
    monkeypatch.setattr(canon_service, "load_leaderboard_config", lambda: dict(_CONFIG))
    monkeypatch.setattr(canon_service, "fetch_hourly_bars", lambda syms, s, e: {"AAPL": object()})
    monkeypatch.setattr(canon_service, "calc_metrics", lambda curve, cap: {
        "initial_equity": cap, "final_equity": cap, "total_return": 0.0,
        "sharpe_ratio": 0.0, "max_drawdown": 0.0,
    })
    monkeypatch.setattr(canon_service.token_cost, "estimate_cost_usd", lambda m, i, o: 0.0)
    return test_db


def _use(monkeypatch, impl):
    monkeypatch.setattr(canon_service, "get_strategy", lambda entry: impl)


def test_refuses_when_used_llm_false(guard_env, monkeypatch):
    _use(monkeypatch, FakeLLMStrategy(used_llm=False, llm_calls=0))
    with pytest.raises(canon_service.LeaderboardFallbackError):
        canon_service.deploy_model_run("claude_haiku_4_5", force_refresh=True)
    run_id = canon_service._run_id("claude_haiku_4_5", "2026-04-15", "2026-05-15")
    assert guard_env.get_run(run_id) is None  # nothing persisted


def test_refuses_when_llm_calls_zero(guard_env, monkeypatch):
    # Client existed (used_llm True) but every call failed → 0 real LLM calls.
    _use(monkeypatch, FakeLLMStrategy(used_llm=True, llm_calls=0))
    with pytest.raises(canon_service.LeaderboardFallbackError):
        canon_service.deploy_model_run("claude_haiku_4_5", force_refresh=True)


def test_refuses_partial_llm_fallback(guard_env, monkeypatch):
    # The client worked for one step then every other step fell back to
    # rule-based: 1 of 161 decisions came from the model. Publishing this curve
    # under the model's name would be ~99% rule-based. (This is the real Qwen
    # 1-of-161 run that silently topped the board.)
    _use(monkeypatch, FakeLLMStrategy(used_llm=True, llm_calls=1, decision_steps=161))
    with pytest.raises(canon_service.LeaderboardFallbackError):
        canon_service.deploy_model_run("claude_haiku_4_5", force_refresh=True)
    run_id = canon_service._run_id("claude_haiku_4_5", "2026-04-15", "2026-05-15")
    assert guard_env.get_run(run_id) is None  # nothing persisted


def test_refuses_just_below_coverage_threshold(guard_env, monkeypatch):
    # 94 of 100 steps LLM-decided = 94% < 95% threshold → refuse.
    _use(monkeypatch, FakeLLMStrategy(used_llm=True, llm_calls=94, decision_steps=100))
    with pytest.raises(canon_service.LeaderboardFallbackError):
        canon_service.deploy_model_run("claude_haiku_4_5", force_refresh=True)


def test_publishes_at_coverage_threshold(guard_env, monkeypatch):
    # 95 of 100 steps LLM-decided = exactly 95% → allowed (transient API blips
    # on a genuine LLM run must not be misread as a fallback curve).
    _use(monkeypatch, FakeLLMStrategy(used_llm=True, llm_calls=95, decision_steps=100))
    result = canon_service.deploy_model_run("claude_haiku_4_5", force_refresh=True)
    assert guard_env.get_run(result["run_id"]) is not None


def test_refuses_when_calls_succeed_but_no_usable_decisions(guard_env, monkeypatch):
    # Every API call "succeeded" (llm_calls == decision_steps == 161) but the
    # model's output was empty/unparseable almost every step, so only 1 step
    # produced a usable decision. The curve is ~99% rule-based despite 100% call
    # coverage — the guard must key off usable decisions, not billed calls.
    _use(monkeypatch, FakeLLMStrategy(
        used_llm=True, llm_calls=161, llm_decisions=1, decision_steps=161))
    with pytest.raises(canon_service.LeaderboardFallbackError):
        canon_service.deploy_model_run("claude_haiku_4_5", force_refresh=True)
    run_id = canon_service._run_id("claude_haiku_4_5", "2026-04-15", "2026-05-15")
    assert guard_env.get_run(run_id) is None  # nothing persisted


def test_allow_fallback_publishes(guard_env, monkeypatch):
    _use(monkeypatch, FakeLLMStrategy(used_llm=False, llm_calls=0))
    result = canon_service.deploy_model_run(
        "claude_haiku_4_5", force_refresh=True, allow_fallback=True
    )
    assert guard_env.get_run(result["run_id"]) is not None


def test_allow_fallback_publishes_partial_run(guard_env, monkeypatch):
    # allow_fallback must bypass the *partial*-coverage guard too, not only the
    # total-fallback case above — a deploy that explicitly opts in publishes a
    # low-coverage run instead of being rejected.
    _use(monkeypatch, FakeLLMStrategy(
        used_llm=True, llm_calls=10, llm_decisions=10, decision_steps=161))
    result = canon_service.deploy_model_run(
        "claude_haiku_4_5", force_refresh=True, allow_fallback=True
    )
    assert guard_env.get_run(result["run_id"]) is not None


def test_strategy_without_llm_decisions_defaults_to_llm_calls(guard_env, monkeypatch):
    # An older strategy that reports decision_steps but not llm_decisions must
    # fall back to llm_calls coverage (the documented default), not be wrongly
    # rejected as 0/decision_steps.
    _use(monkeypatch, FakeLLMStrategy(
        used_llm=True, llm_calls=100, decision_steps=100, report_decisions=False))
    result = canon_service.deploy_model_run("claude_haiku_4_5", force_refresh=True)
    assert guard_env.get_run(result["run_id"]) is not None


def test_publishes_real_llm_run(guard_env, monkeypatch):
    _use(monkeypatch, FakeLLMStrategy(used_llm=True, llm_calls=5))
    result = canon_service.deploy_model_run("claude_haiku_4_5", force_refresh=True)
    assert result["llm_calls"] == 5
    assert guard_env.get_run(result["run_id"]) is not None


def test_deploy_records_effective_llm_run_metadata(guard_env, monkeypatch):
    config = {
        "session_id": "lb-metadata-test",
        "start_date": "2026-04-15",
        "end_date": "2026-05-15",
        "initial_capital": 10000,
        "strategies": [
            {
                "id": "nemotron_3_nano_30b",
                "name": "Nemotron",
                "model": "Nemotron 3 Nano 30B",
                "strategy": "llm_agent",
                "integration": "openrouter",
                "model_id": "configured-model-id",
                "temperature": 0,
                "reasoning_effort": "none",
            },
        ],
    }
    monkeypatch.setattr(canon_service, "load_leaderboard_config", lambda: config)
    monkeypatch.setattr(llm_harness, "DEFAULT_MAX_OUTPUT_TOKENS", 1234)
    _use(
        monkeypatch,
        FakeLLMStrategy(
            used_llm=True,
            llm_calls=5,
            model_id="resolved-model-id",
        ),
    )

    result = canon_service.deploy_model_run(
        "nemotron_3_nano_30b",
        force_refresh=True,
        start_date="2026-04-16",
        end_date="2026-04-30",
    )

    assert guard_env.get_run(result["run_id"])["metadata"] == {
        "entry_id": "nemotron_3_nano_30b",
        "model_id": "resolved-model-id",
        "integration": "openrouter",
        "temperature": 0,
        "reasoning_effort": "none",
        # None, not absent: a Model Track entry carries no instruction, and the
        # stored row has to say so positively or a later Open Track curve is
        # indistinguishable from one written before the field existed.
        "strategy_prompt": None,
        "llm_max_output_tokens": 1234,
        "initial_capital": 10000.0,
        "start_date": "2026-04-16",
        "end_date": "2026-04-30",
    }


def test_deploy_records_the_instruction_that_produced_the_curve(
    guard_env, monkeypatch
):
    """An Open Track curve must carry its instruction, not just its model.

    `leaderboard.json` is editable and `_find_cached_run` does not key on this
    field (issue #365's omission, one field over), so the config is not a record
    of what actually ran — the stored row is the only place the pairing survives.
    """
    config = {
        "session_id": "lb-instruction-test",
        "start_date": "2026-04-15",
        "end_date": "2026-05-15",
        "initial_capital": 10000,
        "strategies": [
            {
                "id": "open_alice",
                "name": "Alice",
                "strategy": "llm_agent",
                "integration": "openrouter",
                "model_id": "configured-model-id",
                # Deliberately DIFFERENT from what the strategy actually ran with,
                # standing in for a leaderboard.json edited after the deploy.
                "strategy_prompt": "Edited after the fact.",
            },
        ],
    }
    monkeypatch.setattr(canon_service, "load_leaderboard_config", lambda: config)
    _use(
        monkeypatch,
        FakeLLMStrategy(
            used_llm=True, llm_calls=5, strategy_prompt="Buy the dip, sell the rip."
        ),
    )

    result = canon_service.deploy_model_run("open_alice", force_refresh=True)

    metadata = guard_env.get_run(result["run_id"])["metadata"]
    # The strategy instance, not the config dict: the run records what executed.
    assert metadata["strategy_prompt"] == "Buy the dip, sell the rip."


def test_deploy_records_null_for_provider_default_parameters(guard_env, monkeypatch):
    _use(monkeypatch, FakeLLMStrategy(used_llm=True, llm_calls=5))
    result = canon_service.deploy_model_run("claude_haiku_4_5", force_refresh=True)

    metadata = guard_env.get_run(result["run_id"])["metadata"]
    assert metadata["temperature"] is None
    assert metadata["reasoning_effort"] is None


def test_cached_historical_run_is_not_backfilled_or_recomputed(
    guard_env,
    monkeypatch,
):
    run_id = canon_service._run_id(
        "claude_haiku_4_5",
        "2026-04-15",
        "2026-05-15",
    )
    guard_env.insert_run(
        run_id=run_id,
        session_id="lb-guard-test",
        agent_name="Haiku",
        mode="leaderboard",
        start_date="2026-04-15",
        end_date="2026-05-15",
        initial_equity=100000,
        llm_model="claude_haiku_4_5",
    )
    monkeypatch.setattr(
        canon_service,
        "get_strategy",
        lambda entry: pytest.fail("cached run must not recompute"),
    )

    result = canon_service.deploy_model_run("claude_haiku_4_5")

    assert result["cached"] is True
    assert guard_env.get_run(run_id)["metadata"] is None


def test_ensure_leaderboard_runs_records_llm_metadata(tmp_path, monkeypatch):
    """Provenance must not depend on *which* path published the run. The
    auto-compute path already guards against a misconfigured LLM entry landing
    here, so it has to record that entry's config too."""
    cfg = {
        "session_id": "lb-auto-meta",
        "start_date": "2026-04-15",
        "end_date": "2026-05-15",
        "initial_capital": 10000,
        "strategies": [
            {"id": "auto_llm", "name": "Auto", "model": "Auto",
             "strategy": "llm_agent", "integration": "openrouter",
             "temperature": 0, "reasoning_effort": "none",
             "auto_compute": True},
            {"id": "djia_index", "name": "DJIA", "model": "DJIA",
             "strategy": "market_index"},
        ],
    }
    test_db = BacktestDatabase(db_path=tmp_path / "lb.db")
    monkeypatch.setattr(canon_service, "db", test_db)
    monkeypatch.setattr(canon_service, "load_leaderboard_config", lambda: dict(cfg))
    monkeypatch.setattr(llm_harness, "DEFAULT_MAX_OUTPUT_TOKENS", 1234)
    monkeypatch.setattr(canon_service, "fetch_hourly_bars", lambda syms, s, e: {"AAPL": object()})
    monkeypatch.setattr(canon_service, "calc_metrics", lambda curve, cap: {
        "initial_equity": cap, "final_equity": cap, "total_return": 0.0,
        "sharpe_ratio": 0.0, "max_drawdown": 0.0,
    })
    monkeypatch.setattr(
        canon_service,
        "get_strategy",
        lambda entry: FakeBaseline() if entry["id"] == "djia_index"
        else FakeLLMStrategy(used_llm=True, llm_calls=5, model_id="resolved-model-id"),
    )

    canon_service.ensure_leaderboard_runs(force_refresh=True)

    llm_run = test_db.get_run(canon_service._run_id("auto_llm", "2026-04-15", "2026-05-15"))
    assert llm_run["metadata"] == {
        "entry_id": "auto_llm",
        "model_id": "resolved-model-id",
        "integration": "openrouter",
        "temperature": 0,
        "reasoning_effort": "none",
        "strategy_prompt": None,
        "llm_max_output_tokens": 1234,
        "initial_capital": 10000.0,
        "start_date": "2026-04-15",
        "end_date": "2026-05-15",
    }
    # A rule-based baseline on the same path stays metadata-free.
    baseline_run = test_db.get_run(
        canon_service._run_id("djia_index", "2026-04-15", "2026-05-15")
    )
    assert baseline_run["metadata"] is None


def test_baseline_without_used_llm_publishes(guard_env, monkeypatch):
    # A rule-based baseline legitimately makes 0 LLM calls and must NOT be blocked.
    _use(monkeypatch, FakeBaseline())
    result = canon_service.deploy_model_run("djia_index", force_refresh=True)
    run = guard_env.get_run(result["run_id"])
    assert run is not None
    assert run["metadata"] is None


def test_ensure_leaderboard_runs_also_guards_llm_fallback(tmp_path, monkeypatch):
    """Belt-and-suspenders: a misconfigured LLM entry left on the auto-compute
    path (auto_compute true) is still refused, not silently published."""
    cfg = {
        "session_id": "lb-auto-test",
        "start_date": "2026-04-15",
        "end_date": "2026-05-15",
        "initial_capital": 100000,
        "strategies": [
            {"id": "sneaky_llm", "name": "Sneaky", "model": "Sneaky",
             "strategy": "llm_agent", "auto_compute": True},
        ],
    }
    test_db = BacktestDatabase(db_path=tmp_path / "lb.db")
    monkeypatch.setattr(canon_service, "db", test_db)
    monkeypatch.setattr(canon_service, "load_leaderboard_config", lambda: dict(cfg))
    monkeypatch.setattr(canon_service, "get_strategy", lambda entry: FakeLLMStrategy(used_llm=False, llm_calls=0))
    monkeypatch.setattr(canon_service, "fetch_hourly_bars", lambda syms, s, e: {"AAPL": object()})
    monkeypatch.setattr(canon_service, "calc_metrics", lambda curve, cap: {
        "initial_equity": cap, "final_equity": cap, "total_return": 0.0,
        "sharpe_ratio": 0.0, "max_drawdown": 0.0,
    })
    with pytest.raises(canon_service.LeaderboardFallbackError):
        canon_service.ensure_leaderboard_runs(force_refresh=True)


def test_ensure_leaderboard_runs_names_model_id_in_fallback(tmp_path, monkeypatch):
    """The auto-compute guard path must name the offending model id in its
    diagnostic (finding #3). Previously it omitted model_id and printed 'None',
    hiding which gateway id failed."""
    cfg = {
        "session_id": "lb-auto-test",
        "start_date": "2026-04-15",
        "end_date": "2026-05-15",
        "initial_capital": 100000,
        "strategies": [
            {"id": "sneaky_llm", "name": "Sneaky", "model": "Sneaky",
             "strategy": "llm_agent", "auto_compute": True},
        ],
    }
    test_db = BacktestDatabase(db_path=tmp_path / "lb.db")
    monkeypatch.setattr(canon_service, "db", test_db)
    monkeypatch.setattr(canon_service, "load_leaderboard_config", lambda: dict(cfg))
    monkeypatch.setattr(canon_service, "get_strategy", lambda entry: FakeLLMStrategy(
        used_llm=True, llm_calls=1, llm_decisions=1, decision_steps=161,
        model_id="sneaky-gateway-id"))
    monkeypatch.setattr(canon_service, "fetch_hourly_bars", lambda syms, s, e: {"AAPL": object()})
    monkeypatch.setattr(canon_service, "calc_metrics", lambda curve, cap: {
        "initial_equity": cap, "final_equity": cap, "total_return": 0.0,
        "sharpe_ratio": 0.0, "max_drawdown": 0.0,
    })
    with pytest.raises(canon_service.LeaderboardFallbackError) as exc:
        canon_service.ensure_leaderboard_runs(force_refresh=True)
    assert "sneaky-gateway-id" in str(exc.value)


def test_default_model_name_is_gateway_aware(monkeypatch):
    """The gateway-aware default llm_agent.py now uses: native id without a
    CommonStack key, the CommonStack slug with one; OpenRouter only when asked."""
    from dashboard.backend.infrastructure.llm import backtest_harness as bh

    monkeypatch.delenv("COMMONSTACK_API_KEY", raising=False)
    assert bh.default_model_name() == bh.LLM_MODEL_NAME

    monkeypatch.setenv("COMMONSTACK_API_KEY", "x")
    assert bh.default_model_name() == bh.COMMONSTACK_MODEL_NAME
    assert bh.default_model_name("openrouter") == bh.OPENROUTER_MODEL_NAME
