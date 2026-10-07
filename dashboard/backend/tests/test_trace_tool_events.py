"""Tool and retrieval trace adapters stay bounded and best effort."""

from pathlib import Path

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
