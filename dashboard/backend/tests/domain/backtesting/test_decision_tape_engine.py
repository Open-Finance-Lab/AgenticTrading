"""Decision tape through the real HourlyBacktester (spec 2026-10-08).

Harness: the conformance suite's in-memory 5m bars and recording DB, so no
network, plus a real SQLite TraceStore patched into the trace service.
"""

import dataclasses
import json
from datetime import timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from dashboard.backend.domain.backtesting import decision_tape as decision_tape_mod
from dashboard.backend.domain.backtesting import engine as engine_mod
from dashboard.backend.domain.backtesting.engine import HourlyBacktester
from dashboard.backend.domain.backtesting.portfolio_manager import PortfolioManager
from dashboard.backend.domain.traces import service as trace_service
from dashboard.backend.domain.traces.repository import TraceStore
from dashboard.backend.infrastructure.market_data.profiles import TransactionCostProfile
from dashboard.backend.tests.conformance.adapters.current_engine import (
    _CaseLoader,
    _RecordingDB,
    synthesize_source_bars,
)
from dashboard.backend.tests.conformance.cases import CASH, _flat_pair

BARS = _flat_pair(6)  # AAPL @100, MSFT @50, flat, seven hourly bars on one day

# The true originals, captured once at import. Each _run patches over them;
# wrapping whatever is currently installed would chain a second run's wrapper
# onto the first run's holder, and the replay test would then compare the
# second run's ledger with itself.
_ORIGINAL_INIT = PortfolioManager.__init__


@pytest.fixture
def store(tmp_path, monkeypatch):
    store = TraceStore(tmp_path / "traces.db")
    monkeypatch.setattr(trace_service, "trace_store", store)
    return store


def _run(monkeypatch, *, run_id="agent_tape_engine", decide=None, llm_client=None, pipeline=None, configure=None):
    """``configure(backtester)`` runs after construction and before
    ``load_data()`` -- the hook the perturbed-engine test uses to hand this
    run a cost profile without touching ``profiles.py``."""
    loader = _CaseLoader(synthesize_source_bars(SimpleNamespace(bars=BARS)))

    def factory(data_source="alpaca", universe=None, *, source_timeframe=None):
        loader.configure_source_timeframe(source_timeframe)
        return loader

    monkeypatch.setattr(engine_mod, "create_market_data_provider", factory)
    monkeypatch.setattr(engine_mod, "db", _RecordingDB())
    holder = SimpleNamespace(pm=None, calls=0)

    def init(pm, *args, **kwargs):
        _ORIGINAL_INIT(pm, *args, **kwargs)
        holder.pm = pm

    monkeypatch.setattr(PortfolioManager, "__init__", init)
    if decide is not None:
        def scripted(pm, state):
            index = holder.calls
            holder.calls += 1
            return {"actions": [dict(action) for action in decide(index, pm)]}

        monkeypatch.setattr(PortfolioManager, "make_trading_decision", scripted)

    day = min(bar.ts for bar in BARS).date()
    backtester = HourlyBacktester(
        day.isoformat(), day.isoformat(), use_llm=False,
        symbols=["AAPL", "MSFT"], initial_capital=float(CASH), live_run_id=run_id,
    )
    backtester.provider_end_date = (day + timedelta(days=1)).isoformat()
    if llm_client is not None:
        backtester.use_llm = True
        backtester.llm_client = llm_client
        backtester.pipeline = pipeline
        backtester.strict_llm = False
    if configure is not None:
        configure(backtester)
    backtester.load_data()
    backtester.calculate_indicators()
    _run_id, curve = backtester.run_agent_backtest()
    return backtester, holder, curve


def _first_divergent_bar(curve_a, curve_b):
    """Index of the first equity point that differs; ``None`` when none does.
    The same question ``scripts/diff_backtest_runs.py`` answers for two
    stored runs."""
    for index, (a, b) in enumerate(zip(curve_a, curve_b)):
        if a["equity"] != b["equity"]:
            return index
    return None if len(curve_a) == len(curve_b) else min(len(curve_a), len(curve_b))


def _decision_bar_at(tape, timestamp):
    """The decision bar whose valuation pass marked an equity point.

    The curve is marked per SOURCE bar (78 points for seven hourly bars in
    intraday mode), so a curve index is not a bar index. After executing bar
    ``i`` the engine marks every source bar up to and including
    ``fill_plan.bar`` (``engine.py``'s ``valuation_cursor`` loop), so a point
    belongs to the first bar whose fill bar is at or after it. Points past the
    last fill bar (the window's closing marks) belong to no bar: ``None``."""
    at = pd.Timestamp(timestamp)
    for bar in tape:
        if pd.Timestamp(bar["execution"]["fill_plan"]["bar"]) >= at:
            return bar["bar_index"]
    return None


def _late_script(index, pm):
    """First fill on bar 2, so curve index and bar index cannot agree at 0."""
    return {
        2: [{"symbol": "AAPL", "action": "buy", "shares": 3, "reason": "open AAPL"}],
        4: [{"symbol": "MSFT", "action": "buy", "shares": 2, "reason": "open MSFT"},
            {"symbol": "AAPL", "action": "sell", "shares": 1, "reason": "trim AAPL"}],
    }.get(index, [])


def _script(index, pm):
    return {
        0: [{"symbol": "AAPL", "action": "buy", "shares": 3, "reason": "open AAPL"}],
        1: [{"symbol": "MSFT", "action": "buy", "shares": 2, "reason": "open MSFT"},
            {"symbol": "AAPL", "action": "sell", "shares": 1, "reason": "trim AAPL"}],
        2: [{"symbol": "AAPL", "action": "buy", "shares": 10_000_000, "reason": "too big"}],
        3: [{"symbol": "AAPL", "action": "buy", "shares": 10_000_000, "reason": "too big"}],
    }.get(index, [])


def test_every_bar_records_one_decision_execution_pair_in_order(monkeypatch, store):
    backtester, holder, _curve = _run(monkeypatch, decide=_script)
    tape = trace_service.load_decision_tape("agent_tape_engine")
    assert len(tape) == holder.calls >= 5
    assert [bar["bar_index"] for bar in tape] == list(range(holder.calls))
    first = tape[0]
    assert first["decision"]["driver"] == "rule_based"
    assert first["decision"]["intent"] is None
    assert first["decision"]["state"] == {"cash": float(CASH), "equity": float(CASH), "positions": []}
    assert first["decision"]["actions"] == [{"symbol": "AAPL", "action": "buy", "shares": 3, "reason": "open AAPL"}]
    assert first["decision"]["gate_rewrote"] is False  # rule-based: no model order to rewrite
    assert [(f["symbol"], f["side"], f["shares"]) for f in first["execution"]["fills"]] == [("AAPL", "BUY", 3)]
    assert set(first["execution"]["fill_plan"]) == {"bar", "price_field", "filled_at"}
    assert tape[1]["decision"]["state"]["positions"] == [["AAPL", 3]]
    summary = backtester.decision_tape_summary()
    assert summary["bars_recorded"] == holder.calls and summary["write_failures"] == 0
    assert summary["gate_rewrites"] == 0


def test_collapsed_repeat_rejection_appears_on_its_own_bar(monkeypatch, store):
    _run(monkeypatch, decide=_script)
    tape = trace_service.load_decision_tape("agent_tape_engine")
    second, third = tape[2]["execution"]["rejected"], tape[3]["execution"]["rejected"]
    assert [(r["symbol"], r["reason"], r["collapsed"]) for r in second] == [("AAPL", "insufficient_cash", False)]
    assert [(r["symbol"], r["reason"], r["collapsed"], r["count"]) for r in third] == [
        ("AAPL", "insufficient_cash", True, 1)
    ]


def _comparable(trades):
    return json.dumps(
        [{k: str(v) for k, v in trade.items()} for trade in trades], sort_keys=True
    )


def test_recorded_actions_reproduce_the_run_on_the_same_engine(monkeypatch, store):
    """Reproducibility, not replay-grade: a second run of THIS engine driven
    by the recorded post-gate actions reproduces the first run's trades and
    curve. It proves the tape survives _pick + JSON and that the engine is
    deterministic on a fixed sequence -- nothing about another engine."""
    _bt, first, first_curve = _run(monkeypatch, run_id="agent_tape_original", decide=_script)
    tape = trace_service.load_decision_tape("agent_tape_original")

    def replay(index, pm):
        return tape[index]["decision"]["actions"]

    _bt, second, second_curve = _run(monkeypatch, run_id="agent_tape_replay", decide=replay)

    assert first.pm is not second.pm
    assert first.pm.trades, "the script must trade, or the replay proves nothing"
    assert _comparable(second.pm.trades) == _comparable(first.pm.trades)
    assert [p["equity"] for p in second_curve] == [p["equity"] for p in first_curve]


@pytest.mark.parametrize("script", [_script, _late_script], ids=["fill_on_bar_0", "fill_on_bar_2"])
def test_perturbed_engine_diverges_at_the_first_filled_bar(monkeypatch, store, script):
    """Review Focus 6: replaying recorded ACTIONS into an engine whose fills
    differ must diverge where the fill is, visibly, and never be absorbed by
    a later recorded size. The perturbation is one tick of slippage on the
    test's own MarketProfile copy (the mechanism A-share uses); profiles.py
    is untouched. This is also the shape diff_backtest_runs.py reports."""
    _bt, first, first_curve = _run(monkeypatch, run_id="agent_tape_clean", decide=script)
    tape = trace_service.load_decision_tape("agent_tape_clean")
    first_fill_bar = next(i for i, bar in enumerate(tape) if bar["execution"]["fills"])
    if script is _late_script:
        assert first_fill_bar == 2

    def replay(index, pm):
        return tape[index]["decision"]["actions"]

    def slip(backtester):
        backtester.profile = dataclasses.replace(
            backtester.profile,
            transaction_cost_profile=TransactionCostProfile(
                version="test-slip-1tick", currency="USD",
                commission_rate=0.0, minimum_commission=0.0,
                stamp_duty_sell_rate=0.0, transfer_fee_rate=0.0,
                buy_slippage_rate=0.001, sell_slippage_rate=0.001, price_tick=0.01,
            ),
        )

    _bt, second, second_curve = _run(monkeypatch, run_id="agent_tape_slipped", decide=replay, configure=slip)

    divergent = _first_divergent_bar(first_curve, second_curve)
    assert divergent is not None
    # Localized: the first equity point that moves belongs to the first bar
    # that filled, mapped through the fill plan -- not compared as raw indexes.
    assert _decision_bar_at(tape, first_curve[divergent]["timestamp"]) == first_fill_bar
    assert any(float(t.get("slippage_amount") or 0) > 0 for t in second.pm.trades)
    # The recorded SELL sizes are the clean engine's; the slipped engine held
    # the same shares (slippage moves price, not quantity), so the trade list
    # keeps its shape and only prices differ -- which is exactly why actions
    # cannot be the replay unit once quantities, not prices, diverge.
    assert [(t["symbol"], t["side"], t["shares"]) for t in second.pm.trades] == \
        [(t["symbol"], t["side"], t["shares"]) for t in first.pm.trades]
    assert _comparable(second.pm.trades) != _comparable(first.pm.trades)


class _PipelineStub:
    """Answers each pipeline call from a script (one decision step per bar)."""

    execution_service = "decision-tape-stub"

    def __init__(self, replies):
        self.replies = list(replies)
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **_request):
        text = self.replies.pop(0) if self.replies else json.dumps(
            {"actions": [{"symbol": "AAPL", "action": "hold", "confidence": 1.0, "reasoning": "hold"}]}
        )
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=0, output_tokens=0),
            stop_reason="end_turn",
        )


def test_llm_pipeline_bars_record_intent_and_fallback(monkeypatch, store):
    stub = _PipelineStub([
        json.dumps({"actions": [{
            "symbol": "AAPL", "action": "buy", "position_size": 5, "confidence": 0.9,
            "reasoning": "buy five", "api_key": "must-not-be-stored",
        }]}),
        "this is not json",
    ])
    _run(monkeypatch, llm_client=stub, pipeline=[{"label": "Decision", "prompt": "Decide."}])
    tape = trace_service.load_decision_tape("agent_tape_engine")

    assert tape[0]["decision"]["driver"] == "llm"
    assert tape[0]["decision"]["intent"] == {"orders": [{
        "symbol": "AAPL", "action": "buy", "position_size": 5,
        "confidence": 0.9, "reasoning": "buy five",
    }]}
    assert tape[0]["decision"]["actions"][0]["shares"] == 5
    assert tape[0]["decision"]["gate_rewrote"] is False  # asked 5, admitted 5

    assert tape[1]["decision"]["driver"] == "llm_fallback"
    assert tape[1]["decision"]["intent"] == {"unparsed": True, "completed_steps": 0}
    assert tape[1]["decision"]["gate_rewrote"] is False  # nothing parsed, nothing to rewrite


def test_llm_sell_widened_by_the_gate_is_counted_as_a_rewrite(monkeypatch, store):
    """Finding 1: every model SELL becomes 'sell all sellable'. The tape must
    say so per bar and per run, so the gate's rewrite rate is a number."""
    stub = _PipelineStub([
        json.dumps({"actions": [{"symbol": "AAPL", "action": "buy", "position_size": 4, "confidence": 0.9, "reasoning": "open"}]}),
        json.dumps({"actions": [{"symbol": "AAPL", "action": "sell", "position_size": 1, "confidence": 0.9, "reasoning": "trim one"}]}),
    ])
    backtester, _holder, _curve = _run(monkeypatch, llm_client=stub, pipeline=[{"label": "Decision", "prompt": "Decide."}])
    tape = trace_service.load_decision_tape("agent_tape_engine")
    assert tape[0]["decision"]["gate_rewrote"] is False
    assert tape[1]["decision"]["intent"]["orders"][0]["position_size"] == 1
    assert tape[1]["decision"]["actions"][0]["shares"] == 4  # the whole position
    assert tape[1]["decision"]["gate_rewrote"] is True
    assert backtester.decision_tape_summary()["gate_rewrites"] == 1


def test_intent_replay_through_the_gate_reproduces_an_llm_run(monkeypatch, store):
    """The replay-grade proof (Review Focus 6): a second run whose MODEL
    answers each bar with the first run's recorded intent -- so the orders go
    through the gate and the fill model again rather than around them --
    reproduces the trades and the curve. This is the path a candidate engine
    takes with a tape; the actions path above is not."""
    script = [
        json.dumps({"actions": [{"symbol": "AAPL", "action": "buy", "position_size": 3, "confidence": 0.9, "reasoning": "open"}]}),
        json.dumps({"actions": [
            {"symbol": "MSFT", "action": "buy", "position_size": 2, "confidence": 0.8, "reasoning": "add"},
            {"symbol": "AAPL", "action": "sell", "position_size": 1, "confidence": 0.7, "reasoning": "trim"},
        ]}),
    ]
    pipeline = [{"label": "Decision", "prompt": "Decide."}]
    _bt, first, first_curve = _run(monkeypatch, run_id="agent_intent_original", llm_client=_PipelineStub(script), pipeline=pipeline)
    tape = trace_service.load_decision_tape("agent_intent_original")
    assert all(bar["decision"]["driver"] == "llm" for bar in tape)
    assert all("orders" in bar["decision"]["intent"] for bar in tape), "every bar must carry parsed intent"

    replies = [json.dumps({"actions": decision_tape_mod.orders_for_replay(bar["decision"]["intent"])}) for bar in tape]
    _bt, second, second_curve = _run(monkeypatch, run_id="agent_intent_replay", llm_client=_PipelineStub(replies), pipeline=pipeline)
    replayed = trace_service.load_decision_tape("agent_intent_replay")

    assert first.pm.trades, "the script must trade, or the replay proves nothing"
    assert _comparable(second.pm.trades) == _comparable(first.pm.trades)
    assert [p["equity"] for p in second_curve] == [p["equity"] for p in first_curve]
    assert [b["decision"]["intent"] for b in replayed] == [b["decision"]["intent"] for b in tape]
    assert [b["decision"]["gate_rewrote"] for b in replayed] == [b["decision"]["gate_rewrote"] for b in tape]


def test_intent_replay_keeps_null_and_non_finite_fields(monkeypatch, store):
    """Finding 3: ``confidence: null`` makes the gate raise (the bar falls back
    to rule-based) while an absent confidence reads as 0.5 and trades; an
    ``Infinity`` position_size is skipped while ``null`` becomes a
    confidence-sized buy. The tape must hand the replay the same values."""
    script = [
        json.dumps({"actions": [{"symbol": "AAPL", "action": "buy", "position_size": 3, "confidence": None, "reasoning": "null conf"}]}),
        json.dumps({"actions": [{"symbol": "MSFT", "action": "buy", "position_size": float("inf"), "confidence": 0.9, "reasoning": "inf size"}]}),
        json.dumps({"actions": [{"symbol": "AAPL", "action": "buy", "position_size": 2, "confidence": 0.9, "reasoning": None}]}),
        json.dumps({"actions": [{"symbol": "AAPL", "action": "buy", "position_size": 1, "confidence": 0.9, "reasoning": "ok"}]}),
    ]
    pipeline = [{"label": "Decision", "prompt": "Decide."}]
    _bt, first, first_curve = _run(monkeypatch, run_id="agent_intent_edge", llm_client=_PipelineStub(script), pipeline=pipeline)
    tape = trace_service.load_decision_tape("agent_intent_edge")

    assert tape[0]["decision"]["driver"] == "llm_fallback"
    assert tape[0]["decision"]["intent"]["orders"][0]["confidence"] is None
    assert tape[0]["decision"]["gate_rewrote"] is False  # a fallback is not a rewrite
    assert tape[1]["decision"]["intent"]["orders"][0]["position_size"] is None
    assert tape[1]["decision"]["intent"]["orders"][0]["raw"] == {"position_size": "Infinity"}
    assert tape[2]["decision"]["driver"] == "llm_fallback"  # reasoning[:60] on None
    assert tape[3]["decision"]["driver"] == "llm"

    replies = [json.dumps({"actions": decision_tape_mod.orders_for_replay(bar["decision"]["intent"])}) for bar in tape]
    _bt, second, second_curve = _run(monkeypatch, run_id="agent_intent_edge_replay", llm_client=_PipelineStub(replies), pipeline=pipeline)
    replayed = trace_service.load_decision_tape("agent_intent_edge_replay")

    assert first.pm.trades, "the script must trade, or the replay proves nothing"
    assert [b["decision"]["driver"] for b in replayed] == [b["decision"]["driver"] for b in tape]
    assert [b["decision"]["actions"] for b in replayed] == [b["decision"]["actions"] for b in tape]
    assert _comparable(second.pm.trades) == _comparable(first.pm.trades)
    assert [p["equity"] for p in second_curve] == [p["equity"] for p in first_curve]


def test_portfolio_manager_reassigns_pipeline_outputs_per_call(monkeypatch, store):
    """Review Focus 9, pinned on the manager, not through the tape: the
    tape's freshness check is `outputs is outputs_before`. A refactor of
    portfolio_manager.py:650 to `.clear()` + `.extend()` would keep one list
    object alive across bars, null `intent` on every bar, and leave every
    tape unit test green. This is the test that must go red, with this name."""
    stub = _PipelineStub([
        json.dumps({"actions": [{"symbol": "AAPL", "action": "buy", "position_size": 1, "confidence": 0.9, "reasoning": "a"}]}),
        json.dumps({"actions": [{"symbol": "AAPL", "action": "hold", "confidence": 1.0, "reasoning": "b"}]}),
    ])
    seen = []
    original = PortfolioManager.make_trading_decision_with_llm

    def spy(pm, *args, **kwargs):
        before = pm.last_pipeline_step_outputs
        result = original(pm, *args, **kwargs)
        seen.append((before, pm.last_pipeline_step_outputs))
        return result

    monkeypatch.setattr(PortfolioManager, "make_trading_decision_with_llm", spy)
    _run(monkeypatch, llm_client=stub, pipeline=[{"label": "Decision", "prompt": "Decide."}])
    assert len(seen) >= 2
    for before, after in seen:
        assert after is not before, "PortfolioManager must assign a NEW list per pipeline call"


def test_run_without_live_run_id_writes_no_tape(monkeypatch, store):
    backtester, _holder, _curve = _run(monkeypatch, run_id=None, decide=_script)
    assert backtester.decision_tape_summary() == {}
    assert store.list_traces()["items"] == []


def test_broken_trace_writes_do_not_change_the_run(monkeypatch, store):
    _bt, healthy, healthy_curve = _run(monkeypatch, run_id="agent_tape_healthy", decide=_script)
    attempts = []

    def broken(**_kwargs):
        attempts.append(1)
        raise RuntimeError("trace store down")

    monkeypatch.setattr(trace_service, "record_tape_bar", broken)
    backtester, broken_holder, broken_curve = _run(monkeypatch, run_id="agent_tape_broken", decide=_script)
    assert [p["equity"] for p in broken_curve] == [p["equity"] for p in healthy_curve]
    assert len(broken_holder.pm.trades) == len(healthy.pm.trades)
    summary = backtester.decision_tape_summary()
    # The store is called at most MAX_CONSECUTIVE_STORE_FAILURES times: a
    # store that hangs rather than raising would otherwise cost every bar.
    limit = decision_tape_mod.MAX_CONSECUTIVE_STORE_FAILURES
    assert broken_holder.calls > limit
    assert len(attempts) == summary["write_failures"] == limit
    assert summary["suspended_skipped"] == broken_holder.calls - limit
    assert summary["bars_recorded"] == 0
