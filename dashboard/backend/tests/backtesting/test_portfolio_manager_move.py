"""Characterization tests for the PortfolioManager move (Phase 2C3).

Verifies that ``PortfolioManager`` now lives canonically in
``dashboard.backend.domain.backtesting.portfolio_manager`` and that the legacy
script re-exports the exact same object, with constructor/state/method behavior
unchanged. No external services are called.
"""

import json
from datetime import datetime

import pandas as pd
import pytest

from dashboard.backend.domain.backtesting import portfolio_manager
from dashboard.backend.domain.backtesting.portfolio_manager import (
    LLMDecisionError,
    PortfolioManager as CanonicalPortfolioManager,
)
from dashboard.scripts import backtest_hourly_agent as bha


def _row(close, **kwargs):
    data = {"close": close}
    data.update(kwargs)
    return pd.Series(data)


# ---------------------------------------------------------------------------
# Identity / re-export
# ---------------------------------------------------------------------------

def test_legacy_reexports_canonical_class():
    assert bha.PortfolioManager is CanonicalPortfolioManager


def test_canonical_module_path():
    assert CanonicalPortfolioManager.__module__ == (
        "dashboard.backend.domain.backtesting.portfolio_manager"
    )


def test_no_separate_legacy_class_object():
    # The script must not define its own duplicate class.
    assert bha.PortfolioManager.__qualname__ == "PortfolioManager"
    assert bha.PortfolioManager is CanonicalPortfolioManager


def test_hourly_backtester_moved_to_engine_in_phase_2c5():
    # Phase 2C5 moved HourlyBacktester to the canonical engine module; the script
    # re-exports the same class object.
    from dashboard.backend.domain.backtesting.engine import HourlyBacktester as Canon

    assert bha.HourlyBacktester is Canon
    assert bha.HourlyBacktester.__module__ == (
        "dashboard.backend.domain.backtesting.engine"
    )


# ---------------------------------------------------------------------------
# Constructor / initial state
# ---------------------------------------------------------------------------

def test_constructor_defaults():
    pm = CanonicalPortfolioManager()
    assert pm.initial_capital == 1000
    assert pm.cash == 1000
    assert pm.positions == {}
    assert pm.entry_prices == {}
    assert pm.trades == []
    assert pm.order_events == []
    assert pm.equity_history == []
    assert pm.llm_calls == 0
    assert pm.input_tokens == 0
    assert pm.output_tokens == 0
    assert pm.lot_size == 1


def test_constructor_accepts_market_lot_size():
    pm = CanonicalPortfolioManager(1000, lot_size=100)

    assert pm.lot_size == 100


def test_constructor_custom_capital():
    pm = CanonicalPortfolioManager(50000)
    assert pm.initial_capital == 50000
    assert pm.cash == 50000


# ---------------------------------------------------------------------------
# Delegation / method behavior (golden)
# ---------------------------------------------------------------------------

def test_get_portfolio_state_delegates():
    pm = CanonicalPortfolioManager(100000)
    pm.positions = {"AAPL": 10}
    pm.entry_prices = {"AAPL": 200.0}
    state = pm.get_portfolio_state({"AAPL": _row(200.0)})
    assert state["positions_value"] == 2000.0
    assert state["total_equity"] == 102000.0


def test_make_trading_decision_delegates():
    pm = CanonicalPortfolioManager(100000)
    state = {
        "total_equity": 100000,
        "market_signals": {"AAPL": {"price": 100.0, "rsi": 25.0, "sma20": 110.0, "sma50": 120.0}},
    }
    out = pm.make_trading_decision(state)
    assert out["actions"][0]["action"] == "buy"


def test_execute_actions_delegates():
    pm = CanonicalPortfolioManager(100000)
    pm.execute_actions(
        [{"symbol": "AAPL", "action": "buy", "shares": 10}],
        {"AAPL": _row(200.0)},
        "t0",
    )
    assert pm.cash == 98000.0
    assert pm.positions == {"AAPL": 10}
    assert pm.trades[0]["side"] == "BUY"


def test_update_equity_and_get_equity_curve_delegate():
    pm = CanonicalPortfolioManager(100000)
    pm.update_equity({}, timestamp="t0")
    curve = pm.get_equity_curve()
    assert curve is pm.equity_history
    assert curve[0] == {"timestamp": "t0", "equity": 100000, "cash": 100000, "positions_value": 0}


# ---------------------------------------------------------------------------
# LLM workflow with a fake client (no network)
# ---------------------------------------------------------------------------

class _FakeUsage:
    def __init__(self, i, o):
        self.input_tokens = i
        self.output_tokens = o


class _FakeResp:
    def __init__(self, text, usage=None):
        self.content = [type("B", (), {"text": text})()]
        self.usage = usage


class _FakeClient:
    def __init__(self, resp):
        self._resp = resp

        class _M:
            @staticmethod
            def create(**kwargs):
                return resp
        self.messages = _M()


def _llm_state():
    return {
        "timestamp": datetime(2026, 1, 1),
        "cash": 100000,
        "positions": [],
        "positions_value": 0,
        "total_equity": 100000,
        "market_signals": {
            "AAPL": {"price": 100.0, "rsi": 25.0, "macd": 1.0, "macd_signal": 0.5,
                     "sma20": 110.0, "sma50": 120.0, "bb_upper": 130.0, "bb_lower": 90.0},
        },
    }


def test_make_trading_decision_with_llm_no_client_fallback():
    pm = CanonicalPortfolioManager(100000)
    out = pm.make_trading_decision_with_llm(_llm_state(), None)
    assert out == pm.make_trading_decision(_llm_state())
    assert pm.llm_calls == 0


def test_make_trading_decision_with_llm_buy_and_tokens():
    pm = CanonicalPortfolioManager(100000)
    resp_text = json.dumps({"actions": [
        {"symbol": "AAPL", "action": "buy", "confidence": 0.9,
         "reasoning": "x", "position_size": 5},
    ]})
    client = _FakeClient(_FakeResp(resp_text, _FakeUsage(12, 8)))
    out = pm.make_trading_decision_with_llm(_llm_state(), client)
    assert out["actions"][0]["action"] == "buy"
    assert pm.input_tokens == 12
    assert pm.output_tokens == 8
    assert pm.llm_calls == 1


def test_llm_omitting_position_size_yields_a_whole_lot_on_ashares():
    """The harness must not size an order the executor is certain to reject.

    With no `position_size` the manager derives one from the risk budget. That
    number is essentially never a 100-multiple, so submitting it raw minted a
    guaranteed `invalid_lot_size` rejection on every fallback -- punishing the
    agent for our own arithmetic. 2% of $100k at $100 is 20 shares, which is
    below one lot, so the correct outcome is no order at all.
    """
    pm = CanonicalPortfolioManager(100000, lot_size=100)
    resp_text = json.dumps({"actions": [
        {"symbol": "AAPL", "action": "buy", "confidence": 1.0,
         "reasoning": "x"},
    ]})
    client = _FakeClient(_FakeResp(resp_text, _FakeUsage(1, 1)))

    out = pm.make_trading_decision_with_llm(_llm_state(), client)

    assert out["actions"] == []


def test_llm_fallback_size_is_floored_to_whole_lots_not_truncated_to_shares():
    state = _llm_state()
    state["cash"] = 10_000_000
    state["total_equity"] = 10_000_000
    pm = CanonicalPortfolioManager(10_000_000, lot_size=100)
    resp_text = json.dumps({"actions": [
        {"symbol": "AAPL", "action": "buy", "confidence": 1.0,
         "reasoning": "x"},
    ]})
    client = _FakeClient(_FakeResp(resp_text, _FakeUsage(1, 1)))

    out = pm.make_trading_decision_with_llm(state, client)

    # 2% of $10M at $100 = 2000 shares, already a whole lot.
    assert [action["shares"] for action in out["actions"]] == [2000]
    assert out["actions"][0]["shares"] % 100 == 0


def test_llm_requested_size_is_still_passed_through_unrounded():
    """Only *our* derived size is rounded; a bad request stays visible."""
    pm = CanonicalPortfolioManager(100000, lot_size=100)
    resp_text = json.dumps({"actions": [
        {"symbol": "AAPL", "action": "buy", "confidence": 0.9,
         "reasoning": "x", "position_size": 150},
    ]})
    client = _FakeClient(_FakeResp(resp_text, _FakeUsage(1, 1)))

    out = pm.make_trading_decision_with_llm(_llm_state(), client)

    assert [action["shares"] for action in out["actions"]] == [150]


def test_strict_llm_rejects_missing_client_without_rule_fallback(monkeypatch):
    pm = CanonicalPortfolioManager(100000)
    fallback_calls = []
    monkeypatch.setattr(
        pm,
        "make_trading_decision",
        lambda _state: fallback_calls.append(True) or {"actions": []},
    )

    with pytest.raises(LLMDecisionError, match="client"):
        pm.make_trading_decision_with_llm(
            _llm_state(),
            None,
            strict_llm=True,
        )

    assert fallback_calls == []


def test_strict_llm_propagates_request_failure_without_rule_fallback(monkeypatch):
    class _BoomClient:
        class messages:
            @staticmethod
            def create(**_kwargs):
                raise RuntimeError("upstream-secret-detail")

    pm = CanonicalPortfolioManager(100000)
    fallback_calls = []
    monkeypatch.setattr(
        pm,
        "make_trading_decision",
        lambda _state: fallback_calls.append(True) or {"actions": []},
    )

    with pytest.raises(LLMDecisionError) as exc_info:
        pm.make_trading_decision_with_llm(
            _llm_state(),
            _BoomClient(),
            strict_llm=True,
        )

    assert "upstream-secret-detail" not in str(exc_info.value)
    assert fallback_calls == []


def test_strict_llm_rejects_unparseable_response():
    pm = CanonicalPortfolioManager(100000)
    client = _FakeClient(_FakeResp("not json", _FakeUsage(3, 2)))

    with pytest.raises(LLMDecisionError, match="parse"):
        pm.make_trading_decision_with_llm(
            _llm_state(),
            client,
            strict_llm=True,
        )

    assert pm.llm_calls == 1
    assert pm.llm_decisions == 0


def test_strict_llm_accepts_empty_actions_as_model_hold():
    pm = CanonicalPortfolioManager(100000)
    client = _FakeClient(_FakeResp('{"actions": []}', _FakeUsage(3, 2)))

    result = pm.make_trading_decision_with_llm(
        _llm_state(),
        client,
        strict_llm=True,
    )

    assert result == {"actions": []}
    assert pm.llm_calls == 1
    assert pm.llm_decisions == 1


def test_strict_pipeline_accepts_empty_orders_as_model_hold():
    pm = CanonicalPortfolioManager(100000)
    client = _FakeClient(_FakeResp('{"orders": []}', _FakeUsage(3, 2)))

    result = pm.make_trading_decision_with_llm(
        _llm_state(),
        client,
        pipeline=[
            {
                "label": "Trading instruction",
                "prompt": "Return the orders for this bar.",
                "outputFormat": '{"orders": []}',
            }
        ],
        strict_llm=True,
    )

    assert result == {"actions": []}
    assert pm.llm_calls == 1
    assert pm.llm_decisions == 1


@pytest.mark.parametrize(
    "action",
    [
        {
            "symbol": "NOT_ALLOWED.SH",
            "action": "buy",
            "confidence": 0.9,
            "position_size": 1,
        },
        {
            "symbol": "AAPL",
            "action": "dance",
            "confidence": 0.9,
            "position_size": 1,
        },
    ],
)
def test_strict_llm_rejects_invalid_action_batch(action):
    pm = CanonicalPortfolioManager(100000, allowed_symbols=["AAPL"])
    client = _FakeClient(
        _FakeResp(json.dumps({"actions": [action]}), _FakeUsage(3, 2))
    )

    with pytest.raises(LLMDecisionError, match="invalid action"):
        pm.make_trading_decision_with_llm(
            _llm_state(),
            client,
            strict_llm=True,
        )

    assert pm.llm_decisions == 0


def test_market_context_is_added_to_snapshot_and_llm_request(monkeypatch):
    from dashboard.backend.domain.backtesting import portfolio_manager as pm_mod

    captured = {}
    market_context = {
        "market": "CN",
        "timezone": "Asia/Shanghai",
        "timeframe": "60m",
        "symbols": ["AAPL"],
        "paper_backtest": True,
    }

    def fake_create_prompt(snapshot, **_kwargs):
        captured["snapshot_market"] = snapshot["market"]
        return "PROMPT"

    def fake_request(_client, **kwargs):
        captured["request_market"] = kwargs["market_context"]
        return _FakeResp(
            json.dumps(
                {
                    "actions": [
                        {
                            "symbol": "AAPL",
                            "action": "hold",
                            "confidence": 0.9,
                            "reasoning": "wait",
                        }
                    ]
                }
            ),
            _FakeUsage(3, 2),
        )

    monkeypatch.setattr(pm_mod, "create_prompt", fake_create_prompt)
    monkeypatch.setattr(pm_mod, "_request_trading_decision", fake_request)

    pm = CanonicalPortfolioManager(100000, allowed_symbols=["AAPL"])
    result = pm.make_trading_decision_with_llm(
        _llm_state(),
        object(),
        market_context=market_context,
        strict_llm=True,
    )

    assert result == {"actions": []}
    assert captured == {
        "snapshot_market": market_context,
        "request_market": market_context,
    }


# ---------------------------------------------------------------------------
# MEDIUM #7 — safe_trading candidate ranking is trend-based, NOT RSI-extremity
# (the module docstring previously claimed the class was "functionally
# identical" / "moved verbatim", which hid this deliberate strategy change).
# ---------------------------------------------------------------------------

def _trend_sig(price, rsi, sma20, sma50, macd=1.0, macd_signal=0.0):
    return {"price": price, "rsi": rsi, "macd": macd, "macd_signal": macd_signal,
            "sma20": sma20, "sma50": sma50, "bb_upper": 0.0, "bb_lower": 0.0}


class _StopAfterCapture(BaseException):
    """BaseException so it threads through make_trading_decision_with_llm's
    ``except Exception`` fallback and stops exactly after the ranking."""


def _capture_top_signals(monkeypatch):
    """Patch create_prompt to record the ranked ``top_signals`` and halt
    before the (unavailable) LLM call."""
    from dashboard.backend.domain.backtesting import portfolio_manager as pm_mod
    captured = {}

    def _fake_create_prompt(snapshot, **kwargs):
        captured["top"] = set(snapshot["top_signals"].keys())
        raise _StopAfterCapture()

    monkeypatch.setattr(pm_mod, "create_prompt", _fake_create_prompt)
    return captured


def test_safe_trading_ranks_by_trend_not_rsi_extremity(monkeypatch):
    captured = _capture_top_signals(monkeypatch)
    signals = {
        # Strong trend confluence + healthy mid RSI -> highest trend score.
        "TREND": _trend_sig(110.0, 55.0, 100.0, 95.0),
        # Deeply oversold, no trend confluence: the pre-refactor |RSI-50|
        # ranking would surface this FIRST; trend ranking ranks it last.
        "OVERSOLD": _trend_sig(80.0, 15.0, 100.0, 110.0, macd=-1.0),
    }
    # 12 solid-trend fillers to fill the top-12 and push OVERSOLD out.
    for i in range(12):
        signals[f"F{i:02d}"] = _trend_sig(105.0, 50.0, 100.0, 98.0)
    state = {
        "timestamp": datetime(2026, 1, 1), "cash": 100000, "positions": [],
        "positions_value": 0, "total_equity": 100000, "market_signals": signals,
    }
    pm = CanonicalPortfolioManager(100000)
    with pytest.raises(_StopAfterCapture):
        pm.make_trading_decision_with_llm(state, llm_client=object(), mode="safe_trading")

    top = captured["top"]
    assert "TREND" in top
    assert "OVERSOLD" not in top   # the old RSI-extremity ranking would include it
    assert len(top) == 12          # top-12 cut, nothing appended (no holdings)


def test_safe_trading_always_includes_current_holdings(monkeypatch):
    captured = _capture_top_signals(monkeypatch)
    signals = {f"F{i:02d}": _trend_sig(105.0, 50.0, 100.0, 98.0) for i in range(12)}
    # A held name that ranks LAST under BOTH schemes: neutral RSI (|50-50|=0, so
    # the old RSI-extremity ranking excludes it too) AND a terrible trend score
    # (price below both SMAs, negative MACD). So its appearance can only be the
    # holdings-append step, not either ranking.
    signals["HELD"] = _trend_sig(70.0, 50.0, 100.0, 120.0, macd=-1.0)
    state = {
        "timestamp": datetime(2026, 1, 1), "cash": 50000,
        "positions": [{"symbol": "HELD", "shares": 10, "entry_price": 90.0,
                       "current_price": 70.0, "position_value": 700.0, "pnl_pct": -22.2}],
        "positions_value": 700.0, "total_equity": 50700.0, "market_signals": signals,
    }
    pm = CanonicalPortfolioManager(50000)
    with pytest.raises(_StopAfterCapture):
        pm.make_trading_decision_with_llm(state, llm_client=object(), mode="safe_trading")

    # Force-included despite a bottom-tier trend score (so the model can exit it).
    assert "HELD" in captured["top"]


def test_safe_trading_ranking_survives_nan_indicator_bars(monkeypatch):
    """Early bars have NaN indicators (e.g. sma50 before 50 periods). The trend
    ranking must not crash and must rank such names out of the top-12 (a NaN
    trend score sorts below real scores) rather than surfacing them."""
    captured = _capture_top_signals(monkeypatch)
    signals = {"GOOD": _trend_sig(110.0, 55.0, 100.0, 95.0)}
    for i in range(12):
        signals[f"F{i:02d}"] = _trend_sig(105.0, 50.0, 100.0, 98.0)
    nan = float("nan")
    signals["NANBAR"] = _trend_sig(nan, nan, nan, nan, macd=nan, macd_signal=nan)
    state = {
        "timestamp": datetime(2026, 1, 1), "cash": 100000, "positions": [],
        "positions_value": 0, "total_equity": 100000, "market_signals": signals,
    }
    pm = CanonicalPortfolioManager(100000)
    with pytest.raises(_StopAfterCapture):
        pm.make_trading_decision_with_llm(state, llm_client=object(), mode="safe_trading")
    top = captured["top"]
    assert "GOOD" in top
    assert "NANBAR" not in top  # NaN score ranked out, not surfaced
    assert len(top) == 12


def test_safe_trading_threads_custom_strategy_prompt(monkeypatch):
    """A custom strategy_prompt is threaded through to create_prompt via
    custom_prompt= (the 'My Trading Algo' / strategy-share path)."""
    from dashboard.backend.domain.backtesting import portfolio_manager as pm_mod
    captured = {}

    def fake_create_prompt(snapshot, mode=None, custom_prompt=None, allowed_symbols=None):
        captured["custom_prompt"] = custom_prompt
        captured["mode"] = mode
        raise _StopAfterCapture()

    monkeypatch.setattr(pm_mod, "create_prompt", fake_create_prompt)
    signals = {f"F{i:02d}": _trend_sig(105.0, 50.0, 100.0, 98.0) for i in range(3)}
    state = {
        "timestamp": datetime(2026, 1, 1), "cash": 100000, "positions": [],
        "positions_value": 0, "total_equity": 100000, "market_signals": signals,
    }
    pm = CanonicalPortfolioManager(100000)
    with pytest.raises(_StopAfterCapture):
        pm.make_trading_decision_with_llm(
            state, llm_client=object(), mode="safe_trading",
            strategy_prompt="MY CUSTOM STRATEGY",
        )
    assert captured["custom_prompt"] == "MY CUSTOM STRATEGY"
    assert captured["mode"] == "safe_trading"


def test_module_docstring_no_longer_claims_verbatim_identity():
    doc = portfolio_manager.__doc__ or ""
    assert "functionally identical" not in doc
    assert "Moved verbatim" not in doc
    # It must instead disclose the safe_trading divergence.
    assert "safe_trading" in doc


# ---------------------------------------------------------------------------
# Subclass compatibility (a fresh subclass)
# ---------------------------------------------------------------------------

def test_simple_subclass_works():
    class MyPM(CanonicalPortfolioManager):
        def custom(self):
            return "ok"

    pm = MyPM(100000)
    pm.execute_actions([{"symbol": "AAPL", "action": "buy", "shares": 1}],
                       {"AAPL": _row(100.0)}, "t0")
    assert pm.cash == 99900.0
    assert pm.custom() == "ok"
    assert [c.__name__ for c in MyPM.__mro__] == ["MyPM", "PortfolioManager", "object"]


# ---------------------------------------------------------------------------
# Strict-LLM strike budget
#
# A single truncated response must not discard a multi-hour backtest, but the
# surviving curve still has to be honestly publishable — so the budget is a
# fraction of the run, not an unbounded retry.
# ---------------------------------------------------------------------------

def test_strict_llm_budget_is_zero_until_the_run_declares_its_length():
    pm = CanonicalPortfolioManager(100000)

    assert pm.strict_llm_total_steps is None
    assert pm.strict_llm_fallback_budget() == 0


def test_strict_llm_budget_scales_with_run_length():
    pm = CanonicalPortfolioManager(100000)

    pm.strict_llm_total_steps = 500
    assert pm.strict_llm_fallback_budget() == 10
    pm.strict_llm_total_steps = 10
    assert pm.strict_llm_fallback_budget() == 0


def test_strict_llm_budget_clears_h6():
    """Surviving the strike budget must still clear the leaderboard's H6 floor.

    The two constants live in modules that cannot import each other (the H6 one
    pulls in the db singleton), so this test is what keeps them consistent.
    """
    from dashboard.backend.domain.backtesting.portfolio_manager import (
        STRICT_LLM_MAX_FALLBACK_RATIO,
    )
    from dashboard.backend.domain.leaderboard.service import (
        MIN_LLM_DECISION_COVERAGE,
    )

    assert STRICT_LLM_MAX_FALLBACK_RATIO <= 1 - MIN_LLM_DECISION_COVERAGE


def test_strict_llm_absorbs_a_bad_response_within_budget(monkeypatch):
    pm = CanonicalPortfolioManager(100000)
    pm.strict_llm_total_steps = 200  # budget of 4
    monkeypatch.setattr(
        pm, "make_trading_decision", lambda _state: {"actions": ["rule-based"]}
    )
    client = _FakeClient(_FakeResp("not json", _FakeUsage(3, 2)))

    result = pm.make_trading_decision_with_llm(
        _llm_state(), client, strict_llm=True
    )

    assert result == {"actions": ["rule-based"]}
    assert pm.strict_llm_fallbacks == 1
    # The step was rule-based, so it must NOT count toward H6 coverage.
    assert pm.llm_decisions == 0
    assert pm.llm_calls == 1


def test_strict_llm_aborts_once_the_budget_is_spent(monkeypatch):
    pm = CanonicalPortfolioManager(100000)
    pm.strict_llm_total_steps = 100  # budget of 2
    monkeypatch.setattr(
        pm, "make_trading_decision", lambda _state: {"actions": []}
    )

    def bad_step():
        return pm.make_trading_decision_with_llm(
            _llm_state(),
            _FakeClient(_FakeResp("not json", _FakeUsage(3, 2))),
            strict_llm=True,
        )

    assert bad_step() == {"actions": []}
    assert bad_step() == {"actions": []}
    with pytest.raises(LLMDecisionError, match="parse"):
        bad_step()

    assert pm.strict_llm_fallbacks == 3
    assert pm.llm_decisions == 0


def test_strict_llm_missing_client_stays_fatal_regardless_of_budget(monkeypatch):
    """A missing client is not transient — no budget should absorb it."""
    pm = CanonicalPortfolioManager(100000)
    pm.strict_llm_total_steps = 10_000
    fallback_calls = []
    monkeypatch.setattr(
        pm,
        "make_trading_decision",
        lambda _state: fallback_calls.append(True) or {"actions": []},
    )

    with pytest.raises(LLMDecisionError, match="client"):
        pm.make_trading_decision_with_llm(_llm_state(), None, strict_llm=True)

    assert fallback_calls == []
    assert pm.strict_llm_fallbacks == 0


def test_strict_llm_absorbed_upstream_error_still_hides_provider_detail(
    monkeypatch, capsys
):
    class _BoomClient:
        class messages:
            @staticmethod
            def create(**_kwargs):
                raise RuntimeError("upstream-secret-detail")

    pm = CanonicalPortfolioManager(100000)
    pm.strict_llm_total_steps = 200
    monkeypatch.setattr(
        pm, "make_trading_decision", lambda _state: {"actions": ["rule-based"]}
    )

    result = pm.make_trading_decision_with_llm(
        _llm_state(), _BoomClient(), strict_llm=True
    )

    assert result == {"actions": ["rule-based"]}
    assert pm.strict_llm_fallbacks == 1
    # Absorbing a strike must not turn the provider's message into log output;
    # print() is the channel that actually reaches prod, so assert on capsys.
    assert "upstream-secret-detail" not in capsys.readouterr().out
