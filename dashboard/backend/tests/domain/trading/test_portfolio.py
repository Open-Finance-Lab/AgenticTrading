"""Characterization tests for extracted portfolio helpers (Phase 2B2).

Locks in the exact behavior of the portfolio state / valuation / equity-curve
helpers in ``dashboard.backend.domain.trading.portfolio``. Imports use the
canonical package path; no external services are touched.
"""

import pandas as pd

from dashboard.backend.domain.trading import portfolio
from dashboard.backend.domain.trading.portfolio import (
    append_equity_record,
    build_equity_record,
    build_market_signals,
    build_portfolio_state,
    build_position_list,
    calculate_positions_value,
    get_equity_curve,
    resolve_price,
)


def _row(close, **kwargs):
    """Build a market-data row (pd.Series) like the real backtester uses."""
    data = {"close": close}
    data.update(kwargs)
    return pd.Series(data)


# ---------------------------------------------------------------------------
# resolve_price
# ---------------------------------------------------------------------------

def test_resolve_price_prefers_market_data():
    md = {"AAPL": _row(200.0)}
    cache = {"AAPL": {"t0": 999.0}}
    assert resolve_price("AAPL", md, cache, "t0") == 200.0


def test_resolve_price_falls_back_to_cache():
    md = {}
    cache = {"AAPL": {"t0": 150.0}}
    assert resolve_price("AAPL", md, cache, "t0") == 150.0


def test_resolve_price_missing_returns_none():
    assert resolve_price("AAPL", {}, None, None) is None
    assert resolve_price("AAPL", {}, {"AAPL": {"t0": 1.0}}, "t1") is None


def test_resolve_price_zero_is_kept():
    md = {"AAPL": _row(0.0)}
    assert resolve_price("AAPL", md, None, None) == 0.0


# ---------------------------------------------------------------------------
# build_position_list / valuation
# ---------------------------------------------------------------------------

def test_position_list_single():
    positions = {"AAPL": 10}
    entry = {"AAPL": 100.0}
    md = {"AAPL": _row(200.0)}
    plist, pvalue = build_position_list(positions, entry, md)
    assert pvalue == 2000.0
    assert plist == [{
        "symbol": "AAPL",
        "shares": 10,
        "entry_price": 100.0,
        "current_price": 200.0,
        "position_value": 2000.0,
        "pnl_pct": 100.0,
    }]


def test_position_list_multiple():
    positions = {"AAPL": 10, "MSFT": 5}
    entry = {}
    md = {"AAPL": _row(200.0), "MSFT": _row(400.0)}
    plist, pvalue = build_position_list(positions, entry, md)
    assert pvalue == 4000.0
    assert len(plist) == 2
    # entry defaults to current price -> pnl 0
    assert all(p["pnl_pct"] == 0 for p in plist)


def test_position_list_zero_quantity_included():
    positions = {"AAPL": 0}
    md = {"AAPL": _row(200.0)}
    plist, pvalue = build_position_list(positions, {}, md)
    assert pvalue == 0
    assert plist[0]["shares"] == 0
    assert plist[0]["position_value"] == 0


def test_position_list_missing_price_skipped():
    positions = {"AAPL": 10, "MSFT": 5}
    md = {"AAPL": _row(200.0)}  # MSFT absent, no cache
    plist, pvalue = build_position_list(positions, {}, md)
    assert pvalue == 2000.0
    assert [p["symbol"] for p in plist] == ["AAPL"]


def test_position_list_zero_price_entry_zero_pnl():
    positions = {"AAPL": 10}
    md = {"AAPL": _row(0.0)}
    plist, pvalue = build_position_list(positions, {}, md)
    assert pvalue == 0
    assert plist[0]["current_price"] == 0.0
    # entry defaults to current (0) -> entry_price > 0 is False -> pnl 0
    assert plist[0]["pnl_pct"] == 0


def test_position_list_fractional_prices():
    positions = {"AAPL": 3}
    md = {"AAPL": _row(33.33)}
    plist, pvalue = build_position_list(positions, {}, md)
    assert pvalue == 3 * 33.33
    assert plist[0]["position_value"] == 3 * 33.33


def test_calculate_positions_value_skips_missing():
    positions = {"AAPL": 10, "MSFT": 5}
    md = {"AAPL": _row(200.0)}
    assert calculate_positions_value(positions, md) == 2000.0


# ---------------------------------------------------------------------------
# build_market_signals
# ---------------------------------------------------------------------------

def test_build_market_signals_schema():
    md = {"AAPL": _row(200.0, rsi_14=55.0, macd=1.0, macd_signal=0.5,
                        sma20=190.0, sma50=180.0, bb_upper=210.0, bb_lower=185.0)}
    sig = build_market_signals(md)
    assert sig == {"AAPL": {
        "price": 200.0,
        "rsi": 55.0,
        "macd": 1.0,
        "macd_signal": 0.5,
        "sma20": 190.0,
        "sma50": 180.0,
        "bb_upper": 210.0,
        "bb_lower": 185.0,
    }}


def test_build_market_signals_missing_keys_none():
    md = {"AAPL": _row(200.0)}
    sig = build_market_signals(md)["AAPL"]
    assert sig["price"] == 200.0
    assert sig["rsi"] is None
    assert sig["macd"] is None


# ---------------------------------------------------------------------------
# build_portfolio_state (golden fixture)
# ---------------------------------------------------------------------------

def test_build_portfolio_state_golden():
    cash = 100000
    positions = {"AAPL": 10, "MSFT": 5}
    entry = {"AAPL": 200.0, "MSFT": 400.0}
    md = {"AAPL": _row(200.0), "MSFT": _row(400.0)}
    state = build_portfolio_state(cash, positions, entry, md)

    assert set(state.keys()) == {
        "cash", "positions", "positions_value", "total_equity", "market_signals",
    }
    assert state["cash"] == 100000
    assert state["positions_value"] == 4000.0
    assert state["total_equity"] == 104000.0
    assert len(state["positions"]) == 2
    assert set(state["positions"][0].keys()) == {
        "symbol", "shares", "entry_price", "current_price", "position_value", "pnl_pct",
    }


def test_build_portfolio_state_empty():
    state = build_portfolio_state(100000, {}, {}, {})
    assert state["cash"] == 100000
    assert state["positions"] == []
    assert state["positions_value"] == 0
    assert state["total_equity"] == 100000
    assert state["market_signals"] == {}


# ---------------------------------------------------------------------------
# equity-curve helpers
# ---------------------------------------------------------------------------

def test_build_equity_record_schema():
    rec = build_equity_record(100000, {"AAPL": 10}, {"AAPL": _row(200.0)}, timestamp="t0")
    assert rec == {
        "timestamp": "t0",
        "equity": 102000.0,
        "cash": 100000,
        "positions_value": 2000.0,
    }


def test_append_equity_record_mutates_in_place():
    hist = []
    out = append_equity_record(hist, 100000, {}, {}, timestamp="t0")
    assert len(hist) == 1
    assert hist[0] is out
    assert hist[0]["equity"] == 100000


def test_append_equity_record_multiple_and_duplicate_timestamps():
    hist = []
    append_equity_record(hist, 100000, {}, {}, timestamp="t0")
    append_equity_record(hist, 100000, {}, {}, timestamp="t1")
    append_equity_record(hist, 100000, {}, {}, timestamp="t1")  # duplicate ts kept
    assert [r["timestamp"] for r in hist] == ["t0", "t1", "t1"]


def test_get_equity_curve_is_alias_not_copy():
    hist = [{"timestamp": "t0", "equity": 1}]
    out = get_equity_curve(hist)
    assert out is hist
    hist.append({"timestamp": "t1", "equity": 2})
    assert out[-1]["timestamp"] == "t1"


def test_module_importable():
    assert portfolio is not None


# ---------------------------------------------------------------------------
# T+1 sellable exposure (available_positions)
# ---------------------------------------------------------------------------

def test_portfolio_state_omits_sellable_shares_when_settlement_is_immediate():
    """Non-A-share snapshots keep the pre-T+1 schema exactly.

    The position dicts feed the LLM prompt, so an extra key here would change
    every DJIA prompt and make historical runs non-comparable.
    """
    state = build_portfolio_state(
        100000, {"AAPL": 10}, {"AAPL": 200.0}, {"AAPL": _row(200.0)}
    )
    assert "sellable_shares" not in state["positions"][0]


def test_portfolio_state_exposes_sellable_shares_under_t_plus_one():
    state = build_portfolio_state(
        100000,
        {"AAPL": 10, "MSFT": 5},
        {"AAPL": 200.0, "MSFT": 400.0},
        {"AAPL": _row(200.0), "MSFT": _row(400.0)},
        available_positions={"AAPL": 4},
    )
    by_symbol = {p["symbol"]: p for p in state["positions"]}
    assert by_symbol["AAPL"]["shares"] == 10
    assert by_symbol["AAPL"]["sellable_shares"] == 4
    # Absent from the balance means fully frozen, not "unconstrained".
    assert by_symbol["MSFT"]["sellable_shares"] == 0


def test_sellable_shares_never_exceeds_the_held_quantity():
    state = build_portfolio_state(
        100000, {"AAPL": 3}, {"AAPL": 200.0}, {"AAPL": _row(200.0)},
        available_positions={"AAPL": 99},
    )
    assert state["positions"][0]["sellable_shares"] == 3


def test_sellable_shares_does_not_disturb_valuation():
    """Valuation is on the total holding; freezing changes what can be sold."""
    args = (100000, {"AAPL": 10}, {"AAPL": 200.0}, {"AAPL": _row(200.0)})
    plain = build_portfolio_state(*args)
    frozen = build_portfolio_state(*args, available_positions={})
    assert plain["positions_value"] == frozen["positions_value"] == 2000.0
    assert plain["total_equity"] == frozen["total_equity"]


# ---------------------------------------------------------------------------
# sellable_shares must reach the model, not just the state dict
# ---------------------------------------------------------------------------

def _t1_manager_holding(shares, sellable):
    from dashboard.backend.domain.backtesting.portfolio_manager import (
        PortfolioManager,
    )

    manager = PortfolioManager(100000, t_plus_one_enabled=True)
    manager.positions = {"AAPL": shares}
    manager.entry_prices = {"AAPL": 200.0}
    manager.available_positions = {"AAPL": sellable} if sellable else {}
    return manager


def _capture_llm_snapshot(monkeypatch, manager):
    """Run one LLM decision and return the snapshot handed to create_prompt."""
    from dashboard.backend.domain.backtesting import portfolio_manager as pm_module

    captured = {}

    def fake_create_prompt(market_snapshot, **_kwargs):
        captured["snapshot"] = market_snapshot
        return "prompt"

    monkeypatch.setattr(pm_module, "create_prompt", fake_create_prompt)
    monkeypatch.setattr(
        pm_module, "_request_trading_decision",
        lambda *a, **k: object(),
    )
    monkeypatch.setattr(pm_module, "_extract_response_text", lambda _r: "{}")
    monkeypatch.setattr(pm_module, "_extract_token_usage", lambda _r: (0, 0))
    monkeypatch.setattr(pm_module, "_parse_llm_response", lambda _t: {"decisions": []})

    state = manager.get_portfolio_state({"AAPL": _row(210.0)})
    manager.make_trading_decision_with_llm(state, llm_client=object())
    return captured["snapshot"]


def test_sellable_shares_is_forwarded_into_the_llm_holdings(monkeypatch):
    """The state dict carrying the key is not enough — the prompt whitelists.

    Without this the field is write-only: the order still gets capped
    downstream, but the model reasons as though it can exit in full.
    """
    manager = _t1_manager_holding(shares=10, sellable=4)
    snapshot = _capture_llm_snapshot(monkeypatch, manager)

    holding = snapshot["current_holdings"]["AAPL"]
    assert holding["shares"] == 10
    assert holding["sellable_shares"] == 4


def test_llm_holdings_are_unchanged_without_t_plus_one(monkeypatch):
    from dashboard.backend.domain.backtesting.portfolio_manager import (
        PortfolioManager,
    )

    manager = PortfolioManager(100000)
    manager.positions = {"AAPL": 10}
    manager.entry_prices = {"AAPL": 200.0}
    snapshot = _capture_llm_snapshot(monkeypatch, manager)

    assert set(snapshot["current_holdings"]["AAPL"]) == {
        "shares", "entry_price", "current_price", "position_value", "pnl_pct",
    }
