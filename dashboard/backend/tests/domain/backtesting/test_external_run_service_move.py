"""Phase 3C1 — external backtest service move + characterization.

Verifies the external agent backtest service moved to the canonical
``dashboard.backend.domain.backtesting.external_run_service`` package while the
old root module remains a re-export shim, that it wires the canonical engine /
portfolio / metrics / market-data / agent-repository symbols, and that the
framework-free contract (registry, decision format, result formatting) behaves as
before. Engine-coupled, network-bound flows are exercised end-to-end by
``test_protocol_api`` and ``test_external_backtest_api``.
"""

import ast
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from dashboard.backend.database import BacktestDatabase
from dashboard.backend.domain.backtesting import external_run_service as svc

_PUBLIC = [
    "ExternalBacktestSession",
    "get_decision_format",
    "verify_session",
    "start_backtest",
    "get_session",
    "get_current_step",
    "submit_decisions",
    "get_status",
    "get_backtest_decisions",
    "get_run_trades",
    "get_run_decisions",
    "get_run_result",
]


# ---------------------------------------------------------------------------
# Canonical import + wiring
# ---------------------------------------------------------------------------

def test_canonical_module_imports():
    assert svc.ExternalBacktestSession.__module__ == (
        "dashboard.backend.domain.backtesting.external_run_service"
    )
    for name in _PUBLIC:
        assert hasattr(svc, name), name


def test_service_wires_canonical_symbols():
    from dashboard.backend.domain.backtesting.engine import HourlyBacktester
    from dashboard.backend.domain.backtesting.portfolio_manager import PortfolioManager
    from dashboard.backend.domain.backtesting.features import TechnicalIndicators
    from dashboard.backend.infrastructure.market_data.alpaca_bars import AlpacaDataLoader
    from dashboard.backend.domain.agents.repository import agent_store
    from dashboard.backend.domain.backtesting.constants import INITIAL_CAPITAL
    from dashboard.backend.domain.backtesting import baseline_worker as bw

    # T2: HourlyBacktester is consumed by the baseline_worker now, not the
    # service; the service no longer imports it.
    assert bw.HourlyBacktester is HourlyBacktester
    assert not hasattr(svc, "HourlyBacktester")
    assert svc.PortfolioManager is PortfolioManager
    assert svc.TechnicalIndicators is TechnicalIndicators
    assert svc.AlpacaDataLoader is AlpacaDataLoader
    assert svc.agent_store is agent_store
    assert svc.INITIAL_CAPITAL == INITIAL_CAPITAL


# ---------------------------------------------------------------------------
# Import boundary: no API routers, no scripts
# ---------------------------------------------------------------------------

def test_service_does_not_import_api_or_scripts():
    tree = ast.parse(Path(svc.__file__).read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    for m in mods:
        assert not m.startswith("dashboard.backend.api"), m
        assert not m.startswith("dashboard.scripts"), m
        assert m != "fastapi" and not m.startswith("fastapi."), m


# ---------------------------------------------------------------------------
# Registry + decision-format contract
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_registry(monkeypatch):
    monkeypatch.setattr(svc, "_sessions", {})
    return svc


def test_get_session_missing_returns_none(isolated_registry):
    assert svc.get_session("nope") is None
    assert svc.get_current_step("nope") is None
    assert svc.submit_decisions("nope", {}) is None
    assert svc.get_status("nope") is None
    assert svc.get_backtest_decisions("nope") is None


class _FakeLoader:
    def fetch_bars(self, symbols, start, end):
        return {}  # no data -> background load marks the session "failed"


def test_start_backtest_contract_and_registration(isolated_registry, monkeypatch):
    monkeypatch.setattr(svc, "AlpacaDataLoader", _FakeLoader)

    out = svc.start_backtest(
        session_id="sess-1",
        agent_name="agent-x",
        model_name="claude-test",
        start_date="2026-04-15",
        end_date="2026-04-16",
    )
    assert out["status"] == "loading"
    assert out["agent_name"] == "agent-x"
    assert out["model_name"] == "claude-test"
    assert out["session_id"] == "sess-1"
    assert out["total_steps"] == 0
    assert out["decision_timeout_seconds"] == svc.DECISION_TIMEOUT_SECONDS
    assert "decision_format" in out
    bt_id = out["backtest_id"]
    assert bt_id.startswith("bt_")

    session = svc.get_session(bt_id)
    assert session is not None
    assert session.session_id == "sess-1"
    assert svc.verify_session(session, "sess-1") is True
    assert svc.verify_session(session, "other") is False


def test_get_decision_format_schema():
    fmt = svc.get_decision_format()
    assert "actions" in fmt
    action = fmt["actions"][0]
    for key in ("action", "symbol", "confidence", "reasoning", "position_size"):
        assert key in action


def test_protocol_bars_serializes_current_allowed_symbols_only():
    timestamp = pd.Timestamp(2026, 4, 15, 10, tz=ZoneInfo("US/Eastern"))
    session = svc.ExternalBacktestSession(
        backtest_id="bt-bars",
        session_id="sess-bars",
        agent_name="agent-bars",
        model_name="local-model",
        start_date="2026-04-15",
        end_date="2026-04-16",
        symbols=["AAPL"],
    )
    session.all_data = {
        "AAPL": pd.DataFrame(
            {
                "open": [100.0],
                "high": [102.0],
                "low": [99.0],
                "close": [101.5],
                "volume": [1_000_000],
            },
            index=pd.DatetimeIndex([timestamp]),
        ),
        "MSFT": pd.DataFrame(
            {
                "open": [200.0],
                "high": [202.0],
                "low": [199.0],
                "close": [201.5],
                "volume": [2_000_000],
            },
            index=pd.DatetimeIndex([timestamp]),
        ),
    }

    assert session.protocol_bars(timestamp) == {
        "AAPL": {
            "timestamp": timestamp.isoformat(),
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": 101.5,
            "volume": 1_000_000.0,
        }
    }
    assert session.protocol_bars(timestamp + pd.Timedelta(hours=1)) == {}


def test_protocol_bars_drops_invalid_rows_instead_of_raising():
    """protocol_bars() is polled unconditionally on every /steps/next call; a
    malformed bar must be dropped, not raised, or it permanently 500s the run's
    step loop (the timestamp doesn't advance until a decision is submitted).
    """
    timestamp = pd.Timestamp(2026, 4, 15, 10, tz=ZoneInfo("US/Eastern"))
    session = svc.ExternalBacktestSession(
        backtest_id="bt-bad-bar",
        session_id="sess-bad-bar",
        agent_name="agent-bad-bar",
        model_name="local-model",
        start_date="2026-04-15",
        end_date="2026-04-16",
        symbols=["AAPL", "MSFT"],
    )
    session.all_data = {
        "AAPL": pd.DataFrame(
            {
                "open": [100.0],
                "high": [102.0],
                "low": [99.0],
                "close": [float("nan")],
                "volume": [1_000_000],
            },
            index=pd.DatetimeIndex([timestamp]),
        ),
        "MSFT": pd.DataFrame(
            {
                "open": [200.0],
                "high": [202.0],
                "low": [199.0],
                "close": [201.5],
                "volume": [2_000_000],
            },
            index=pd.DatetimeIndex([timestamp]),
        ),
    }

    bars = session.protocol_bars(timestamp)
    assert "AAPL" not in bars
    assert bars == {
        "MSFT": {
            "timestamp": timestamp.isoformat(),
            "open": 200.0,
            "high": 202.0,
            "low": 199.0,
            "close": 201.5,
            "volume": 2_000_000.0,
        }
    }


def test_protocol_bars_accepts_prefetched_market_data():
    timestamp = pd.Timestamp(2026, 4, 15, 10, tz=ZoneInfo("US/Eastern"))
    session = svc.ExternalBacktestSession(
        backtest_id="bt-prefetch",
        session_id="sess-prefetch",
        agent_name="agent-prefetch",
        model_name="local-model",
        start_date="2026-04-15",
        end_date="2026-04-16",
        symbols=["AAPL"],
    )
    session.all_data = {
        "AAPL": pd.DataFrame(
            {
                "open": [100.0],
                "high": [102.0],
                "low": [99.0],
                "close": [101.5],
                "volume": [1_000_000],
            },
            index=pd.DatetimeIndex([timestamp]),
        ),
    }

    market_data = session.protocol_market_data(timestamp)
    assert session.protocol_bars(timestamp, market_data=market_data) == {
        "AAPL": {
            "timestamp": timestamp.isoformat(),
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": 101.5,
            "volume": 1_000_000.0,
        }
    }


# ---------------------------------------------------------------------------
# Result formatting / serialization against an isolated database
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    test_db = BacktestDatabase(db_path=tmp_path / "ext_svc.db")
    monkeypatch.setattr(svc, "db", test_db)
    return test_db


def test_get_run_result_missing_returns_none(isolated_db):
    assert svc.get_run_result("missing", "sess-1") is None
    assert svc.get_run_trades("missing", "sess-1") is None
    assert svc.get_run_decisions("missing", "sess-1") is None


def test_get_run_result_serialization_schema(isolated_db):
    isolated_db.insert_run(
        run_id="ext_test_1",
        session_id="sess-1",
        agent_name="agent-x",
        mode="backtest",
        start_date="2026-04-15",
        end_date="2026-04-16",
        initial_equity=100000.0,
        final_equity=101000.0,
        total_return=0.01,
        sharpe_ratio=1.2,
        max_drawdown=-0.05,
        num_trades=3,
        llm_model="claude-test",
        llm_calls=4,
        input_tokens=10,
        output_tokens=20,
        est_cost_usd=0.5,
    )

    result = svc.get_run_result("ext_test_1", "sess-1")
    assert result is not None
    assert set(result.keys()) == {"run", "equity_curve", "trades", "decisions", "metrics"}
    metrics = result["metrics"]
    for key in (
        "total_return", "sharpe_ratio", "max_drawdown", "num_trades",
        "final_equity", "llm_calls", "input_tokens", "output_tokens", "est_cost_usd",
    ):
        assert key in metrics
    assert metrics["final_equity"] == 101000.0
    assert metrics["num_trades"] == 3

    # Session ownership is enforced: a different session cannot read the run.
    assert svc.get_run_result("ext_test_1", "other-session") is None
