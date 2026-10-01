"""The market snapshot offers the whole universe up to ``SNAPSHOT_FULL_UNIVERSE_MAX``.

Issue #541: the model saw only ``trend_sorted[:12]`` of a 30-name universe and
was never told a cut happened, so an instruction to spread across the universe
quietly meant "across today's top 12 by trend", and the shortlist moved from bar
to bar -- a path-dependent confound for any before/after comparison. Up to 30
names there is now no cut; above it the shortlist survives and the pipeline
prompt announces it.

The single-prompt path is deliberately unchanged: it already lists every VALID
SYMBOL, its contract caps a reply at 10 actions, and the leaderboard's LLM
entries run it against cached curves that record nothing about the snapshot.
"""

from datetime import datetime

import pytest

from dashboard.backend.domain.backtesting import portfolio_manager as pm
from dashboard.backend.domain.backtesting.portfolio_manager import PortfolioManager
from dashboard.backend.infrastructure.llm.pipeline_runner import _build_step_prompt

_PIPELINE = [{"id": "decision", "label": "Decide", "prompt": "Pick trades."}]


def _signals(n):
    """``n`` symbols whose trend scores differ, so the ordering is observable."""
    return {
        f"S{i:02d}": {
            "price": 100.0 + i, "rsi": 55.0, "macd": float(i % 3),
            "macd_signal": 1.0, "sma20": 100.0, "sma50": 90.0 + (i % 7),
            "bb_upper": 130.0, "bb_lower": 80.0,
        }
        for i in range(n)
    }


def _state(n, held=()):
    signals = _signals(n)
    return {
        "timestamp": datetime(2026, 4, 1, 15, 30),
        "cash": 10000.0,
        "positions": [
            {"symbol": s, "shares": 1, "entry_price": 100.0,
             "current_price": 100.0, "position_value": 100.0, "pnl_pct": 0.0}
            for s in held
        ],
        "positions_value": 100.0 * len(held),
        "total_equity": 10000.0 + 100.0 * len(held),
        "market_signals": signals,
    }


@pytest.fixture
def seen(monkeypatch):
    """Capture the snapshot the pipeline receives and the single-prompt text."""
    got = {}

    def fake_pipeline(client, *, pipeline, market_snapshot, model=None):
        got["snapshot"] = market_snapshot
        return {"actions": []}, (0, 0), 1, []

    def fake_request(client, *, prompt, **kwargs):
        got["prompt"] = prompt

        class _R:
            content = []

        return _R()

    real_create_prompt = pm.create_prompt

    def recording_create_prompt(snapshot, **kwargs):
        got["prompt_snapshot"] = snapshot
        return real_create_prompt(snapshot, **kwargs)

    monkeypatch.setattr(pm, "create_prompt", recording_create_prompt)
    monkeypatch.setattr(pm, "run_pipeline_decision", fake_pipeline)
    monkeypatch.setattr(pm, "_request_trading_decision", fake_request)
    monkeypatch.setattr(pm, "_extract_response_text", lambda r: '{"actions": []}')
    monkeypatch.setattr(pm, "_extract_token_usage", lambda r: (0, 0))
    return got


def _manager(n):
    return PortfolioManager(
        initial_capital=10000.0, allowed_symbols=list(_signals(n))
    )


def _run(n, *, pipeline, held=(), market_context=None):
    manager = _manager(n)
    manager.make_trading_decision_with_llm(
        _state(n, held), llm_client=object(), model="m",
        pipeline=_PIPELINE if pipeline else None,
        market_context=market_context,
    )


@pytest.mark.parametrize("n", [7, 12, 30])
def test_pipeline_snapshot_carries_the_whole_universe_up_to_the_limit(seen, n):
    _run(n, pipeline=True)
    assert set(seen["snapshot"]["top_signals"]) == set(_signals(n))
    assert "universe_note" not in seen["snapshot"]


def test_full_universe_keeps_the_trend_ordering(seen):
    def sig(price, sma20, sma50, macd, macd_signal):
        return {"price": price, "rsi": 55.0, "macd": macd,
                "macd_signal": macd_signal, "sma20": sma20, "sma50": sma50,
                "bb_upper": 130.0, "bb_lower": 80.0}

    state = _state(0)
    # Registered weakest-first so dict order cannot explain the result.
    state["market_signals"] = {
        "WEAK": sig(80.0, 95.0, 100.0, -1.0, 0.0),
        "FLAT": sig(100.0, 100.0, 100.0, 0.0, 0.0),
        "STRONG": sig(120.0, 110.0, 100.0, 1.0, 0.0),
    }
    manager = PortfolioManager(
        initial_capital=10000.0, allowed_symbols=["WEAK", "FLAT", "STRONG"]
    )
    manager.make_trading_decision_with_llm(
        state, llm_client=object(), model="m", pipeline=_PIPELINE
    )
    assert list(seen["snapshot"]["top_signals"]) == ["STRONG", "FLAT", "WEAK"]


@pytest.mark.parametrize("n", [30, 50])
def test_single_prompt_path_keeps_its_shortlist_and_adds_no_note(seen, n):
    """Byte-for-byte the pre-#541 snapshot: the leaderboard caches its curves."""
    _run(n, pipeline=False)
    snapshot = seen["prompt_snapshot"]
    assert len(snapshot["top_signals"]) == pm.SNAPSHOT_SHORTLIST_SIZE
    assert "universe_note" not in snapshot
    assert "Snapshot shows" not in seen["prompt"]


def test_over_limit_universe_keeps_shortlist_plus_holdings_and_announces_it(seen):
    held = "S00"  # lowest-trend name: only the holdings rule can keep it
    _run(50, pipeline=True, held=[held])
    snapshot = seen["snapshot"]
    assert len(snapshot["top_signals"]) == pm.SNAPSHOT_SHORTLIST_SIZE + 1
    assert held in snapshot["top_signals"]
    note = (
        f"Snapshot shows 13 of 50 symbols: the top {pm.SNAPSHOT_SHORTLIST_SIZE} "
        "by trend score, plus 1 current holding outside them."
    )
    assert snapshot["universe_note"] == note
    prompt = _build_step_prompt(
        step_index=0, step=_PIPELINE[0], market_snapshot=snapshot,
        prior_outputs=[], is_last=True,
    )
    assert note in prompt


def test_constants_are_named():
    assert pm.SNAPSHOT_FULL_UNIVERSE_MAX == 30
    assert pm.SNAPSHOT_SHORTLIST_SIZE == 12


def test_a_missing_bar_does_not_flip_a_31_name_universe_to_the_full_view(seen):
    """31 configured names, one without a bar: still a cut, still N=31."""
    state = _state(31)
    del state["market_signals"]["S30"]  # 30 names have a bar this timestamp
    manager = _manager(31)
    manager.make_trading_decision_with_llm(
        state, llm_client=object(), model="m", pipeline=_PIPELINE
    )
    snapshot = seen["snapshot"]
    assert len(snapshot["top_signals"]) == pm.SNAPSHOT_SHORTLIST_SIZE
    assert snapshot["universe_note"] == (
        f"Snapshot shows 12 of 31 symbols: the top {pm.SNAPSHOT_SHORTLIST_SIZE} "
        "by trend score."
    )


def test_the_note_precedes_the_signals_in_the_snapshot(seen):
    _run(50, pipeline=True)
    keys = list(seen["snapshot"])
    assert keys.index("universe_note") < keys.index("top_signals")


def test_the_note_moves_no_other_key(seen):
    """Only ``universe_note`` is inserted; ``market`` keeps its place."""
    _run(50, pipeline=True, market_context={"regime": "x"})
    assert list(seen["snapshot"]) == [
        "timestamp", "portfolio", "current_holdings", "recent_trades",
        "universe_note", "top_signals", "market",
    ]


def test_duplicate_configured_symbols_do_not_inflate_the_universe(seen):
    """Upper-cased duplicates count once: 30 unique names stay whole."""
    names = list(_signals(30))
    manager = PortfolioManager(
        initial_capital=10000.0, allowed_symbols=names + [names[0].lower()]
    )
    manager.make_trading_decision_with_llm(
        _state(30), llm_client=object(), model="m", pipeline=_PIPELINE
    )
    assert set(seen["snapshot"]["top_signals"]) == set(names)
    assert "universe_note" not in seen["snapshot"]


def test_the_note_counts_what_the_snapshot_actually_holds(seen):
    """Eight names with a bar in a 40-name universe: the note says eight."""
    state = _state(40)
    state["market_signals"] = dict(list(state["market_signals"].items())[:8])
    _manager(40).make_trading_decision_with_llm(
        state, llm_client=object(), model="m", pipeline=_PIPELINE
    )
    snapshot = seen["snapshot"]
    assert len(snapshot["top_signals"]) == 8
    assert snapshot["universe_note"] == (
        "Snapshot shows 8 of 40 symbols: the top 8 by trend score."
    )


def test_a_holding_already_ranked_is_not_announced_as_extra(seen):
    _run(50, pipeline=True)
    top = next(iter(seen["snapshot"]["top_signals"]))  # ranked first
    _run(50, pipeline=True, held=[top])
    assert seen["snapshot"]["universe_note"] == (
        f"Snapshot shows 12 of 50 symbols: the top {pm.SNAPSHOT_SHORTLIST_SIZE} "
        "by trend score."
    )


def test_nan_scores_sort_last_and_leave_the_rest_in_order(seen):
    """A NaN key has no defined place in ``sorted``; it must not scramble."""
    def sig(price, sma50):
        return {"price": price, "rsi": 55.0, "macd": 0.0, "macd_signal": 0.0,
                "sma20": 0.0, "sma50": sma50, "bb_upper": 0.0, "bb_lower": 0.0}

    nan = float("nan")
    state = _state(0)
    # Distance above sma50 is the only thing that differs, so the expected
    # order is exactly C, B, A; the NaN name sits between them in dict order.
    state["market_signals"] = {
        "A": sig(101.0, 100.0), "NANBAR": sig(nan, 100.0),
        "B": sig(105.0, 100.0), "C": sig(110.0, 100.0),
    }
    PortfolioManager(
        initial_capital=10000.0, allowed_symbols=["A", "NANBAR", "B", "C"]
    ).make_trading_decision_with_llm(
        state, llm_client=object(), model="m", pipeline=_PIPELINE
    )
    assert list(seen["snapshot"]["top_signals"]) == ["C", "B", "A", "NANBAR"]
