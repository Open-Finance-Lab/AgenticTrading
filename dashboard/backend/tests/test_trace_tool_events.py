"""Tool and retrieval trace adapters stay bounded and best effort."""

from pathlib import Path
import inspect

import pandas as pd

from dashboard.backend.domain.traces import service
from dashboard.backend.domain.traces.repository import TraceStore
from dashboard.backend.infrastructure.market_data.provider import TraceAwareMarketDataProvider


def test_trace_tool_events_record_bounded_metadata(tmp_path: Path, monkeypatch):
    store = TraceStore(tmp_path / "tool-events.db")
    monkeypatch.setattr(service, "trace_store", store)
    store.create_trace(run_id="run-tools", agent_id="agent-1")

    service.record_tool_call(
        run_id="run-tools", tool_name="market.search",
        input_summary={"symbol_count": 2}, idempotency_key="call-1",
    )
    service.record_tool_result(
        run_id="run-tools", tool_name="market.search", outcome="success",
        duration_ms=12.3456, result_summary={"row_count": 4}, idempotency_key="call-1",
    )
    events = store.list_events(store.get_trace_for_run("run-tools")["trace_id"])
    assert [event["event_type"] for event in events["items"]] == ["tool_call", "tool_result"]
    assert events["items"][1]["payload"]["duration_ms"] == 12.346


def test_market_data_wrapper_records_counts_and_does_not_break_provider(tmp_path: Path, monkeypatch):
    store = TraceStore(tmp_path / "retrieval.db")
    monkeypatch.setattr(service, "trace_store", store)
    trace = store.create_trace(run_id="run-data", agent_id="agent-1")

    class Provider:
        def fetch_bars(self, symbols, start, end):
            return {symbol: pd.DataFrame({"close": [1, 2]}) for symbol in symbols}

    wrapped = TraceAwareMarketDataProvider(Provider(), run_id="run-data", source="alpaca")
    frames = wrapped.fetch_bars(["AAPL", "MSFT"], "2026-10-01", "2026-10-02")
    assert sorted(frames) == ["AAPL", "MSFT"]
    events = store.list_events(trace["trace_id"])["items"]
    assert events[0]["event_type"] == "data_retrieval"
    assert events[0]["payload"]["result_summary"] == {
        "bar_counts": {"AAPL": 2, "MSFT": 2}, "symbol_count": 2,
    }


def test_market_data_wrapper_preserves_explicit_depth_start_signature():
    assert "depth_start" in inspect.signature(TraceAwareMarketDataProvider.fetch_bars).parameters


def test_trace_observability_failure_is_best_effort(tmp_path: Path, monkeypatch):
    store = TraceStore(tmp_path / "failure.db")
    monkeypatch.setattr(service, "trace_store", store)
    store.create_trace(run_id="run-failure")

    class Provider:
        def fetch_bars(self, symbols, start, end):
            raise RuntimeError("provider down")

    def broken(*_args, **_kwargs):
        raise RuntimeError("trace db down")

    monkeypatch.setattr(service, "record_data_retrieval", broken)
    wrapped = TraceAwareMarketDataProvider(Provider(), run_id="run-failure", source="alpaca")
    try:
        wrapped.fetch_bars(["AAPL"], "2026-10-01", "2026-10-02")
    except RuntimeError as exc:
        assert str(exc) == "provider down"
    else:
        raise AssertionError("provider failure was swallowed")


def test_trace_lookup_failure_does_not_escape_tool_adapters(monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("trace db down")

    monkeypatch.setattr(service, "trace_for_run", broken)
    assert service.record_tool_call(
        run_id="run-failure", tool_name="llm.call", input_summary={}, idempotency_key="call"
    ) is None
    assert service.record_tool_result(
        run_id="run-failure", tool_name="llm.call", outcome="failed",
        duration_ms=1, idempotency_key="result",
    ) is None
    assert service.record_data_retrieval(
        run_id="run-failure", source="alpaca", query_summary={}, result_summary={},
        duration_ms=1, idempotency_key="data",
    ) is None


def test_hourly_provider_path_ensures_trace(monkeypatch):
    from types import SimpleNamespace

    import dashboard.backend.domain.backtesting.engine as engine_module

    provider = object()
    monkeypatch.setattr(engine_module, "create_market_data_provider", lambda *_args, **_kwargs: provider)
    captured = {}

    def ensure(**kwargs):
        captured.update(kwargs)
        return {"trace_id": "trace-live"}

    monkeypatch.setattr(service, "ensure_trace_for_run", ensure)
    backtester = engine_module.HourlyBacktester.__new__(engine_module.HourlyBacktester)
    backtester.data_source = "alpaca"
    backtester.profile = SimpleNamespace(universe=None)
    backtester.requested_source_timeframe = "1h"
    backtester.live_run_id = "run-live"
    backtester.symbols = ["AAPL"]
    backtester.start_date = "2026-10-01"
    backtester.end_date = "2026-10-02"
    backtester.decision_source = "rule_based"

    wrapped = backtester._create_market_data_provider()
    assert isinstance(wrapped, TraceAwareMarketDataProvider)
    assert captured["run_id"] == "run-live"
    assert captured["initial_input"]["symbols"] == ["AAPL"]
