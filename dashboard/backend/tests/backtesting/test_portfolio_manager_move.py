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
    assert pm.equity_history == []
    assert pm.llm_calls == 0
    assert pm.input_tokens == 0
    assert pm.output_tokens == 0


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
