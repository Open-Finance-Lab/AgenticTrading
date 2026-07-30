"""Phase 2C5 — move HourlyBacktester to the canonical engine module.

Verifies the class identity/re-export, the backend->scripts import boundary (now
zero), and that constructor, baseline, and the deterministic rule-based backtest
behave exactly as before. No real Alpaca or Anthropic calls are made.
"""

import ast
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytz
import pytest

from dashboard.backend.domain.backtesting import engine as engine_mod
from dashboard.backend.domain.backtesting.engine import HourlyBacktester
from dashboard.backend.domain.backtesting.metrics import (
    calculate_max_drawdown,
    calculate_sharpe,
)
from dashboard.backend.domain.agents.runtime import (
    AI_HEDGE_FUND_RUNTIME_TYPE,
    RuntimeDispatcher,
)
from dashboard.backend.infrastructure.ai_hedge_fund.adapter import AiHedgeFundRuntime
from dashboard.scripts import backtest_hourly_agent as bha

_REPO_ROOT = Path(__file__).resolve().parents[4]
ENGINE_MODULE = "dashboard.backend.domain.backtesting.engine"
FLAT = "backtest_hourly_agent"


# ---------------------------------------------------------------------------
# Fakes / fixtures
# ---------------------------------------------------------------------------

class _FakeLoader:
    """Stand-in for AlpacaDataLoader; returns preset bars, no network/creds."""

    bars: dict = {}

    def __init__(self, *args, **kwargs):
        pass

    def fetch_bars(self, symbols, start_date, end_date):
        return {s: df for s, df in self.bars.items() if s in symbols}


class _FakeDB:
    def __init__(self):
        self.runs = []
        self.equity_points = []
        self.trades = []
        self.decisions = []

    def insert_run(self, **kwargs):
        self.runs.append(kwargs)

    def insert_equity_points(self, run_id, points):
        self.equity_points.append((run_id, list(points)))

    def insert_trades(self, run_id, trades):
        self.trades.append((run_id, list(trades)))

    def insert_decisions(self, run_id, decisions):
        self.decisions.append((run_id, list(decisions)))


def _fake_provider_factory(data_source="alpaca", universe=None):
    return _FakeLoader()


def _make_bars(symbols, n_hours=70):
    """Deterministic OHLCV bars on tz-aware ET market-hour timestamps."""
    et = pytz.timezone("US/Eastern")
    timestamps = []
    day = datetime(2026, 3, 2)  # Monday
    while len(timestamps) < n_hours:
        if day.weekday() < 5:  # weekdays only
            for hour in range(10, 16):  # 10:00-15:00 ET (all market hours)
                timestamps.append(et.localize(datetime(day.year, day.month, day.day, hour, 0)))
        day += timedelta(days=1)
    timestamps = timestamps[:n_hours]
    idx = pd.DatetimeIndex(timestamps)

    out = {}
    for s_i, sym in enumerate(symbols):
        base = 100.0 + s_i * 10.0
        # Smooth deterministic series with mild oscillation + drift.
        prices = [base + ((i % 7) - 3) * 0.5 + i * 0.1 for i in range(n_hours)]
        out[sym] = pd.DataFrame(
            {
                "open": prices,
                "high": [p + 1 for p in prices],
                "low": [p - 1 for p in prices],
                "close": prices,
                "volume": [1000] * n_hours,
            },
            index=idx,
        )
    return out


@pytest.fixture
def patched_engine(monkeypatch):
    """Patch the loader + db so the engine never touches network or the real DB."""
    _FakeLoader.bars = _make_bars(["AAPL", "MSFT", "JPM"], n_hours=70)
    monkeypatch.setattr(engine_mod, "create_market_data_provider", _fake_provider_factory)
    fake_db = _FakeDB()
    monkeypatch.setattr(engine_mod, "db", fake_db)
    return fake_db


# ---------------------------------------------------------------------------
# Identity / re-export
# ---------------------------------------------------------------------------

def test_class_identity_and_module():
    assert bha.HourlyBacktester is HourlyBacktester
    assert HourlyBacktester.__module__ == ENGINE_MODULE
    assert HourlyBacktester.__qualname__ == "HourlyBacktester"


def test_no_duplicate_compat_subclass():
    # The script must re-export the exact canonical object, not a wrapper.
    assert bha.HourlyBacktester.__mro__[0] is HourlyBacktester
    assert FLAT not in sys.modules  # flat module name never created by the suite


# ---------------------------------------------------------------------------
# Backend -> scripts boundary (zero after Phase 2C5)
# ---------------------------------------------------------------------------

def test_no_backend_source_imports_scripts():
    backend = _REPO_ROOT / "dashboard" / "backend"
    offenders = []
    for path in backend.rglob("*.py"):
        rel = str(path).replace(os.sep, "/")
        if "/tests/" in rel:
            continue  # boundary-test fixtures legitimately reference these names
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == FLAT or alias.name.startswith("dashboard.scripts"):
                        offenders.append((rel, alias.name))
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod == FLAT or mod.startswith("dashboard.scripts"):
                    offenders.append((rel, mod))
    assert offenders == [], f"backend->scripts imports found: {offenders}"


def test_external_backtest_service_uses_canonical_engine():
    # T2 moved baseline generation off the finalize request into the deduping
    # baseline_worker, so the worker (not the service) is now the backend's
    # HourlyBacktester consumer. It must still bind the canonical engine class,
    # never the legacy flat script.
    import dashboard.backend.domain.backtesting.external_run_service as ebs
    from dashboard.backend.domain.backtesting import baseline_worker as bw

    assert bw.HourlyBacktester is HourlyBacktester
    assert bw.HourlyBacktester.__module__ == ENGINE_MODULE
    assert not hasattr(ebs, "HourlyBacktester")  # moved out of the service (T2)


# ---------------------------------------------------------------------------
# Constructor + metric delegation
# ---------------------------------------------------------------------------

def test_constructor_attributes(monkeypatch):
    monkeypatch.setattr(engine_mod, "create_market_data_provider", _fake_provider_factory)
    bt = HourlyBacktester("2026-03-01", "2026-04-01", "sess-1", use_llm=False, mode="safe_trading")
    assert bt.start_date == "2026-03-01"
    assert bt.end_date == "2026-04-01"
    assert bt.session_id == "sess-1"
    assert bt.mode == "safe_trading"
    assert bt.data_source == "alpaca"
    assert bt.use_llm is False
    assert bt.llm_client is None
    assert bt.all_data == {}
    assert isinstance(bt.data_loader, _FakeLoader)


def test_constructor_swaps_backwards_dates(monkeypatch):
    monkeypatch.setattr(engine_mod, "create_market_data_provider", _fake_provider_factory)
    bt = HourlyBacktester("2026-04-01", "2026-03-01", use_llm=False)
    assert bt.start_date == "2026-03-01"
    assert bt.end_date == "2026-04-01"


def test_calc_metrics_delegate():
    curve = [{"equity": 100000}, {"equity": 101000}, {"equity": 99000}, {"equity": 102000}]
    assert HourlyBacktester._calc_sharpe(curve) == calculate_sharpe(curve)
    assert HourlyBacktester._calc_max_dd(curve) == calculate_max_drawdown(curve)
    # Legacy re-export path returns the same.
    assert bha.HourlyBacktester._calc_sharpe(curve) == calculate_sharpe(curve)


# ---------------------------------------------------------------------------
# load_data + calculate_indicators
# ---------------------------------------------------------------------------

def test_load_and_calculate_indicators(patched_engine):
    bt = HourlyBacktester("2026-03-01", "2026-04-01", use_llm=False)
    bt.load_data()
    assert set(bt.all_data.keys()) == {"AAPL", "MSFT", "JPM"}
    bt.calculate_indicators()
    for df in bt.all_data.values():
        for col in ["rsi_14", "macd", "macd_signal", "bb_upper", "bb_lower", "sma20", "sma50"]:
            assert col in df.columns


# ---------------------------------------------------------------------------
# Deterministic rule-based backtest smoke test
# ---------------------------------------------------------------------------

def _run_rule_based(fake_db):
    bt = HourlyBacktester("2026-03-01", "2026-04-01", "smoke", use_llm=False)
    bt.load_data()
    bt.calculate_indicators()
    return bt.run_agent_backtest()


def test_run_agent_backtest_smoke(patched_engine):
    run_id, equity_curve = _run_rule_based(patched_engine)

    assert run_id.startswith("agent_")
    assert isinstance(equity_curve, list) and len(equity_curve) == 70
    first = equity_curve[0]
    assert set(first.keys()) == {"timestamp", "equity", "cash", "positions_value"}
    assert isinstance(first["timestamp"], str)  # converted to isoformat
    assert first["equity"] > 0

    # Exactly one run + equity-points insert recorded; schema preserved.
    assert len(patched_engine.runs) == 1
    run = patched_engine.runs[0]
    for key in ("run_id", "session_id", "agent_name", "sharpe_ratio", "max_drawdown",
                "num_trades", "llm_model", "llm_calls", "input_tokens", "output_tokens"):
        assert key in run
    assert run["agent_name"] == "Agent"
    assert run["llm_model"] == "rule-based"  # no LLM used
    assert run["sharpe_ratio"] == HourlyBacktester._calc_sharpe(equity_curve)
    assert run["max_drawdown"] == HourlyBacktester._calc_max_dd(equity_curve)
    assert patched_engine.equity_points[0][0] == run_id


def test_ai_hedge_fund_run_persists_bounded_decision_audit(patched_engine):
    class HoldRunner:
        def run(self, payload, *, timeout_seconds):
            return {
                "decisions": {
                    "AAPL": {
                        "action": "hold",
                        "quantity": 0,
                        "confidence": 61.23456,
                        "reasoning": "Bounded summary " + "x" * 500,
                    }
                },
                "analyst_signals": {
                    "technical_analyst_agent": {
                        "AAPL": {
                            "signal": "neutral",
                            "confidence": 67,
                            "reasoning": "Unbounded analyst reasoning " + "x" * 500,
                        }
                    },
                    "risk_management_agent": {
                        "AAPL": {
                            "remaining_position_limit": 200,
                            "current_price": 100,
                            "reasoning": {
                                "portfolio_value": 1000,
                                "remaining_limit": 200,
                                "available_cash": 1000,
                            },
                        }
                    },
                },
            }

    bt = HourlyBacktester(
        "2026-03-01",
        "2026-04-01",
        "aihf-audit",
        runtime_type=AI_HEDGE_FUND_RUNTIME_TYPE,
        runtime_config={"analysts": ["technical_analyst"]},
        symbols=["AAPL", "MSFT", "JPM"],
        decision_source="llm",
    )
    runtime = AiHedgeFundRuntime(
        {"analysts": ["technical_analyst"]},
        runner=HoldRunner(),
        environment={},
    )
    bt.runtime_dispatcher = RuntimeDispatcher(
        AI_HEDGE_FUND_RUNTIME_TYPE,
        {"analysts": ["technical_analyst"]},
        runtime=runtime,
    )
    bt.load_data()
    bt.calculate_indicators()

    run_id, _curve = bt.run_agent_backtest()

    assert len(patched_engine.decisions) == 1
    persisted_run_id, rows = patched_engine.decisions[0]
    assert persisted_run_id == run_id
    assert len(rows) == runtime.calls
    assert rows[0]["decision_source"] == AI_HEDGE_FUND_RUNTIME_TYPE
    assert rows[0]["context_ref"] == "2026-03-02"
    assert rows[0]["actions_executed"] == 0
    assert rows[0]["actions_submitted"] == [
        {
            "decision_date": "2026-03-03",
            "data_cutoff_date": "2026-03-02",
            "ticker": "AAPL",
            "upstream_action": "hold",
            "upstream_quantity": 0,
            "confidence": 61.2346,
            "mapped_atl_action": "hold",
            "order_emitted": False,
            "filter_reason": "upstream_hold",
            "diagnostics": {
                "analyst_signals": {
                    "technical_analyst_agent": {
                        "signal": "neutral",
                        "confidence": 67.0,
                    }
                },
                "risk_management_agent": {
                    "remaining_position_limit": 200.0,
                    "current_price": 100.0,
                    "reasoning": {
                        "portfolio_value": 1000.0,
                        "remaining_limit": 200.0,
                        "available_cash": 1000.0,
                    },
                },
                "portfolio_manager": {
                    "action": "hold",
                    "quantity": 0,
                    "confidence": 61.2346,
                    "reasoning_summary": ("Bounded summary " + "x" * 500)[:240],
                },
            },
        }
    ]
    submitted = rows[0]["actions_submitted"][0]
    assert "reasoning" not in submitted
    assert (
        "reasoning"
        not in submitted["diagnostics"]["analyst_signals"]["technical_analyst_agent"]
    )
    assert (
        len(submitted["diagnostics"]["portfolio_manager"]["reasoning_summary"]) == 240
    )


def test_run_agent_backtest_deterministic(monkeypatch):
    bars = _make_bars(["AAPL", "MSFT", "JPM"], n_hours=70)

    def _run_once():
        _FakeLoader.bars = bars
        monkeypatch.setattr(engine_mod, "create_market_data_provider", _fake_provider_factory)
        monkeypatch.setattr(engine_mod, "db", _FakeDB())
        bt = HourlyBacktester("2026-03-01", "2026-04-01", use_llm=False)
        bt.load_data()
        bt.calculate_indicators()
        _, curve = bt.run_agent_backtest()
        return [round(e["equity"], 6) for e in curve]

    assert _run_once() == _run_once()


# ---------------------------------------------------------------------------
# Baselines (credential-free via patched generate_baselines)
# ---------------------------------------------------------------------------

def test_buyhold_baseline_schema(monkeypatch):
    monkeypatch.setattr(engine_mod, "create_market_data_provider", _fake_provider_factory)
    fake_db = _FakeDB()
    monkeypatch.setattr(engine_mod, "db", fake_db)
    fake_curve = [
        {"timestamp": "2026-03-02T10:00:00", "equity": 100000.0, "cash": 0.0, "positions_value": 100000.0},
        {"timestamp": "2026-03-02T11:00:00", "equity": 101000.0, "cash": 0.0, "positions_value": 101000.0},
    ]
    monkeypatch.setattr(engine_mod, "generate_baselines", lambda **kw: (fake_curve, []))

    bt = HourlyBacktester("2026-03-01", "2026-04-01", use_llm=False)
    bt.all_data = {"AAPL": pd.DataFrame()}  # non-empty so the method proceeds
    run_id, history = bt.run_buyhold_baseline()

    assert run_id.startswith("buyhold_")
    assert history == fake_curve
    assert fake_db.runs[0]["agent_name"] == "buy-and-hold"
    assert fake_db.runs[0]["num_trades"] == 1


def test_djia_baseline_schema(monkeypatch):
    monkeypatch.setattr(engine_mod, "create_market_data_provider", _fake_provider_factory)
    fake_db = _FakeDB()
    monkeypatch.setattr(engine_mod, "db", fake_db)
    fake_curve = [
        {"timestamp": "2026-03-02T10:00:00", "equity": 100000.0, "cash": 0, "positions_value": 100000.0},
    ]
    monkeypatch.setattr(engine_mod, "generate_baselines", lambda **kw: ([], fake_curve))

    bt = HourlyBacktester("2026-03-01", "2026-04-01", use_llm=False)
    bt.all_data = {"AAPL": pd.DataFrame()}
    run_id, history = bt.run_djia_baseline()

    assert run_id.startswith("djia_index_")
    assert history == fake_curve
    assert fake_db.runs[0]["agent_name"] == "DJIA"
    assert fake_db.runs[0]["num_trades"] == 0


def test_baselines_empty_data(monkeypatch):
    monkeypatch.setattr(engine_mod, "create_market_data_provider", _fake_provider_factory)
    monkeypatch.setattr(engine_mod, "db", _FakeDB())
    monkeypatch.setattr(engine_mod, "generate_baselines", lambda **kw: ([], []))

    bt = HourlyBacktester("2026-03-01", "2026-04-01", use_llm=False)
    bt.all_data = {}
    assert bt.run_buyhold_baseline() == (None, [])
    assert bt.run_djia_baseline() == (None, [])


# ---------------------------------------------------------------------------
# Custom-algo subclass compatibility (verified in its run context)
# ---------------------------------------------------------------------------

def test_custom_algo_subclasses_canonical_engine():
    code = (
        "import backtest_custom_algo as bca\n"
        "from dashboard.backend.domain.backtesting.engine import HourlyBacktester\n"
        "assert issubclass(bca.CustomAlgoBacktester, HourlyBacktester), 'not a subclass'\n"
        "assert HourlyBacktester in bca.CustomAlgoBacktester.__mro__\n"
        "print('OK')\n"
    )
    scripts_dir = _REPO_ROOT / "dashboard" / "scripts"
    with tempfile.TemporaryDirectory(prefix="atl_engine_") as tmp:
        env = {**os.environ, "DATABASE_PATH": os.path.join(tmp, "t.db")}
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(scripts_dir),
            env=env,
            capture_output=True,
            text=True,
        )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stderr}"
    assert "OK" in proc.stdout


def _hosted_backtester(runner, *, symbols=("AAPL", "MSFT", "JPM")):
    bt = HourlyBacktester(
        "2026-03-01",
        "2026-04-01",
        "aihf-resilience",
        runtime_type=AI_HEDGE_FUND_RUNTIME_TYPE,
        runtime_config={"analysts": ["technical_analyst"]},
        symbols=list(symbols),
        decision_source="llm",
    )
    bt.runtime_dispatcher = RuntimeDispatcher(
        AI_HEDGE_FUND_RUNTIME_TYPE,
        {"analysts": ["technical_analyst"]},
        runtime=AiHedgeFundRuntime(
            {"analysts": ["technical_analyst"]},
            runner=runner,
            environment={},
        ),
    )
    bt.load_data()
    bt.calculate_indicators()
    return bt


def test_hosted_runtime_holds_through_a_transient_step_failure(patched_engine):
    """One bad day must not discard a run's worth of completed steps.

    Every failure mode of a hosted decision -- step timeout, non-zero exit,
    malformed JSON, an upstream rate limit on day 14 of 21 -- used to raise
    straight out of run_agent_backtest with no partial results persisted.
    """
    from dashboard.backend.infrastructure.ai_hedge_fund.adapter import (
        AiHedgeFundRuntimeError,
    )

    class FlakyRunner:
        def __init__(self):
            self.calls = 0

        def run(self, payload, *, timeout_seconds):
            self.calls += 1
            if self.calls == 2:
                raise AiHedgeFundRuntimeError("timed out after 300 seconds")
            return {"decisions": {"AAPL": {"action": "hold", "quantity": 0,
                                           "confidence": 55.0, "reasoning": "ok"}}}

    bt = _hosted_backtester(FlakyRunner())
    run_id, curve = bt.run_agent_backtest()

    assert curve, "the run must still produce an equity curve"
    assert len(bt.runtime_step_failures) == 1
    assert "timed out" in bt.runtime_step_failures[0]

    run_row = patched_engine.runs[0]
    assert run_row["run_id"] == run_id
    # Metadata makes the degradation legible rather than silently clean.
    assert run_row["metadata"]["runtime_step_failures"] == 1
    assert run_row["metadata"]["runtime_step_failure_samples"]


def test_hosted_runtime_aborts_once_failures_exceed_the_budget(patched_engine):
    """Absorbing *every* failure would persist a flat curve as a hosted run."""
    from dashboard.backend.domain.agents.runtime import AgentRuntimeError
    from dashboard.backend.infrastructure.ai_hedge_fund.adapter import (
        AiHedgeFundRuntimeError,
    )

    class DeadRunner:
        def run(self, payload, *, timeout_seconds):
            raise AiHedgeFundRuntimeError("upstream is down")

    bt = _hosted_backtester(DeadRunner())
    with pytest.raises(AgentRuntimeError, match="over the"):
        bt.run_agent_backtest()
    assert patched_engine.runs == []


def test_hosted_runtime_configuration_error_aborts_on_the_first_step(patched_engine):
    """A missing interpreter fails identically every step; do not spend the budget."""
    from dashboard.backend.infrastructure.ai_hedge_fund.adapter import (
        AiHedgeFundConfigurationError,
    )

    class UninstalledRunner:
        calls = 0

        def run(self, payload, *, timeout_seconds):
            UninstalledRunner.calls += 1
            raise AiHedgeFundConfigurationError(
                "AI Hedge Fund runtime is not installed"
            )

    bt = _hosted_backtester(UninstalledRunner())
    with pytest.raises(AiHedgeFundConfigurationError):
        bt.run_agent_backtest()
    assert UninstalledRunner.calls == 1
    assert bt.runtime_step_failures == []


def test_hosted_run_row_reports_runtime_calls_not_zero(patched_engine):
    """A model name beside llm_calls=0 is the shape H6 reads as rule-based."""

    class HoldRunner:
        def run(self, payload, *, timeout_seconds):
            return {"decisions": {"AAPL": {"action": "hold", "quantity": 0,
                                           "confidence": 55.0, "reasoning": "ok"}}}

    bt = _hosted_backtester(HoldRunner())
    bt.run_agent_backtest()

    run_row = patched_engine.runs[0]
    assert bt.runtime_dispatcher.calls > 0
    assert run_row["llm_calls"] == bt.runtime_dispatcher.calls
    assert run_row["llm_model"] == bt.runtime_dispatcher.model_name
    assert "gpt-4.1" not in str(run_row["llm_model"])


def test_hosted_run_that_never_reached_the_model_is_not_attributed_to_it(
    patched_engine, monkeypatch
):
    """Zero successful calls must persist the honest label, not the model name."""

    class NeverRuns:
        def run(self, payload, *, timeout_seconds):  # pragma: no cover - unused
            raise AssertionError("should not be invoked")

    bt = _hosted_backtester(NeverRuns())
    # With no prior market date for any bar, every step holds before it can
    # dispatch -- the "first trading date of the run" case, generalized.
    monkeypatch.setattr(
        engine_mod, "_prior_market_date_by_decision_date", lambda timestamps: {}
    )
    bt.run_agent_backtest()

    run_row = patched_engine.runs[0]
    assert run_row["llm_calls"] == 0
    assert run_row["llm_model"] == "rule-based"
