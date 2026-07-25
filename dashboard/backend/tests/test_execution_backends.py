import sys

import pytest

from dashboard.backend.execution.base import ExecutionBackend
from dashboard.backend.execution.paper_backend import PaperBackend


def test_execution_backend_is_abstract():
    with pytest.raises(TypeError):
        ExecutionBackend()  # abstract — cannot instantiate


def test_paper_backend_is_designed_for_stub():
    backend = PaperBackend()
    assert backend.loop == "realtime"
    with pytest.raises(NotImplementedError):
        backend.build_context()


import pandas as pd

import dashboard.backend.execution.backtest_backend as bb_mod
from dashboard.backend.api.v2.models import ContextEnvelope, SubmitAck


def test_news_sentiment_fail_closed_when_plan1_absent(monkeypatch):
    """Module absent -> loader degrades to ({}, None). Simulated absence:
    a None entry in sys.modules makes the in-function import raise ImportError."""
    import sys
    monkeypatch.setitem(sys.modules,
                        "dashboard.backend.integrations.news_sentiment", None)
    sentiment, overview = bb_mod.load_news_sentiment(["AAPL"], "2026-04-15T10:30:00+00:00")
    assert sentiment == {}
    assert overview is None


def _synthetic_bars(symbols, periods=40):
    idx = pd.date_range("2026-04-15 13:30", periods=periods, freq="h", tz="UTC")
    data = {}
    for i, sym in enumerate(symbols):
        base = 100 + i * 10
        df = pd.DataFrame({
            "open": base, "high": base + 1, "low": base - 1,
            "close": [base + (j % 5) for j in range(periods)],
            "volume": 1000,
        }, index=idx)
        data[sym] = df
    return data


def test_backtest_backend_emits_typed_context(monkeypatch):
    symbols = ["AAPL", "MSFT", "JPM"]

    class _Loader:
        def fetch_bars(self, syms, start, end):
            return _synthetic_bars(symbols)

    monkeypatch.setattr(bb_mod.ext, "AlpacaDataLoader", lambda: _Loader())
    monkeypatch.setattr(bb_mod, "DJIA_30", symbols, raising=False)

    backend = bb_mod.BacktestBackend(
        run_id="run_test_1", session_id="sess_1", agent_name="a",
        model_name="m", start_date="2026-04-15", end_date="2026-04-16",
    )
    backend.load_blocking()  # synchronous load for the test
    assert backend.loop == "lockstep"

    ctx = backend.build_context()
    # Slot guaranteed and the whole envelope validates against the typed model.
    assert "news_sentiment" in ctx
    ContextEnvelope.model_validate(ctx)

    ack = backend.apply_decisions([
        {"action": "hold", "symbol": "AAPL", "confidence": 0.4,
         "reasoning": "hold steady", "position_size": 0},
    ])
    SubmitAck.model_validate(ack)
    assert ack["accepted"] is True


def _install_fake_adapter(monkeypatch, raises: Exception):
    """Install an importable news_sentiment adapter whose get_news_sentiment
    raises, i.e. the call-time failure branch rather than the import-time one."""
    import types

    fake = types.ModuleType("dashboard.backend.integrations.news_sentiment")

    def _boom(universe, timestamp):
        raise raises

    fake.get_news_sentiment = _boom
    monkeypatch.setitem(sys.modules, "dashboard.backend.integrations.news_sentiment", fake)


def test_news_sentiment_fail_closed_when_loader_raises(monkeypatch):
    # Plan 1 adapter exists but throws at call time → still fail-closed.
    _install_fake_adapter(monkeypatch, RuntimeError("news service down"))

    sentiment, overview = bb_mod.load_news_sentiment(["AAPL"], "2026-04-15T10:30:00+00:00")
    assert sentiment == {} and overview is None


def test_news_sentiment_loader_failure_is_logged(monkeypatch, caplog):
    """Fail-closed must not mean fail-silent: swallowed without a log, an adapter
    fault empties the sentiment slot with no exception and no red test — from
    out here, indistinguishable from a quiet news day.

    A producer rename no longer arrives this way. The adapter drops and alarms on
    its own wire-shape drift, so this is the backstop for what is left: a fault
    unexpected enough to escape it entirely, reported with whatever it carried
    (hence the assertion on the exception's own text, not on a hint this end
    guesses about the other end's internals)."""
    _install_fake_adapter(monkeypatch, KeyError("sentiment_score"))

    with caplog.at_level("ERROR"):
        sentiment, overview = bb_mod.load_news_sentiment(["AAPL"], "2026-04-15T10:30:00+00:00")

    assert sentiment == {} and overview is None
    assert any("sentiment_score" in r.getMessage() for r in caplog.records)


def test_news_sentiment_import_failure_is_logged(monkeypatch, caplog):
    """The adapter shipped long ago, so an ImportError here is a real packaging
    or deployment fault — not the pre-Plan-1 'not landed yet' state this
    fallback was originally written for. Say so rather than degrading mutely."""
    monkeypatch.setitem(sys.modules,
                        "dashboard.backend.integrations.news_sentiment", None)

    with caplog.at_level("ERROR"):
        sentiment, overview = bb_mod.load_news_sentiment(["AAPL"], "2026-04-15T10:30:00+00:00")

    assert sentiment == {} and overview is None
    assert any("news_sentiment" in r.getMessage() for r in caplog.records)


def test_apply_decisions_executes_pre_validated_actions():
    # Per-action validation now happens at the v2 boundary (validate_actions);
    # the backend receives only valid actions and reports execution results.
    backend = bb_mod.BacktestBackend.__new__(bb_mod.BacktestBackend)
    backend.run_id = "run_exec"

    class _StubSession:
        step_index = 1
        status = "waiting_decision"

        def submit_decisions(self, payload):
            executed = [{"action": a["action"], "symbol": a["symbol"],
                         "shares": a["position_size"]} for a in payload["actions"]]
            return {"accepted": True, "executed": executed,
                    "decision_source": "external_agent", "next_step": 1,
                    "status": "waiting_decision", "metrics": None}

    backend.session = _StubSession()
    ack = backend.apply_decisions([
        {"action": "buy", "symbol": "AAPL", "confidence": 0.7,
         "reasoning": "valid action here", "position_size": 3},
    ])
    SubmitAck.model_validate(ack)
    assert any(e["symbol"] == "AAPL" for e in ack["executed"])
    assert ack["rejected"] == []


def test_cancel_makes_context_report_closed(monkeypatch):
    symbols = ["AAPL", "MSFT", "JPM"]

    class _Loader:
        def fetch_bars(self, syms, start, end):
            return _synthetic_bars(symbols)

    monkeypatch.setattr(bb_mod.ext, "AlpacaDataLoader", lambda: _Loader())
    monkeypatch.setattr(bb_mod, "DJIA_30", symbols, raising=False)

    backend = bb_mod.BacktestBackend(
        run_id="run_closed_1", session_id="sess_c", agent_name="a",
        model_name="m", start_date="2026-04-15", end_date="2026-04-16",
    )
    backend.load_blocking()
    backend.cancel()

    ctx = backend.build_context()
    assert ctx["status"] == "closed"
    ContextEnvelope.model_validate(ctx)
