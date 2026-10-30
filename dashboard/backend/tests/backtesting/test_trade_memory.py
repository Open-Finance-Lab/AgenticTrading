"""24h trade memory carried across a ``PortfolioManager`` snapshot.

The Live board runs each night in a fresh process with a fresh manager. The
book always crossed that boundary through ``snapshot_state``/``restore_state``;
the trades the prompt's ``recent_trades`` block reads did not, so the first
decisions of every night saw no trades at all and the "do not reverse a trade
from the last day" rule could not fire. These pin the carried memory: what is
kept, how it reads back, that it never joins ``trades``, which the Live board
counts and folds into the month's trade statistics, and the marker that says
from which bar a chain of snapshots has carried it.
"""

import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from dashboard.backend.domain.backtesting.portfolio_manager import (
    SNAPSHOT_TRADE_MEMORY_LIMIT,
    TRADE_MEMORY_VERSION,
    TRADE_MEMORY_WINDOW,
    PortfolioManager,
)


def _trade(ts, symbol="AAPL", side="BUY", shares=10, price=100.0, **extra):
    return {
        "timestamp": pd.Timestamp(ts),
        "symbol": symbol,
        "side": side,
        "shares": shares,
        "price": price,
        **extra,
    }


def _memory(ts, symbol="AAPL", side="BUY", shares=10.0, price=100.0):
    return {
        "symbol": symbol,
        "side": side,
        "shares": shares,
        "price": price,
        "timestamp": ts,
    }


class _Usage:
    input_tokens = 1
    output_tokens = 1


class _Reply:
    content = [type("Block", (), {"text": json.dumps({"actions": []})})()]
    usage = _Usage()


class _Client:
    class messages:
        @staticmethod
        def create(**_kwargs):
            return _Reply()


def _prompt_memory(manager, monkeypatch, decision_ts):
    """The ``recent_trades`` block one decision at ``decision_ts`` is shown."""
    seen = []

    def capture(snapshot, **_kwargs):
        seen.append(snapshot)
        return "prompt"

    monkeypatch.setattr(
        "dashboard.backend.domain.backtesting.portfolio_manager.create_prompt", capture
    )
    state = {
        "timestamp": pd.Timestamp(decision_ts),
        "cash": manager.cash,
        "positions": [],
        "positions_value": 0,
        "total_equity": manager.cash,
        "market_signals": {},
    }
    manager.make_trading_decision_with_llm(state, _Client())
    assert len(seen) == 1
    return seen[0]["recent_trades"]


def test_window_and_cap_constants():
    assert TRADE_MEMORY_WINDOW == timedelta(hours=24)
    assert SNAPSHOT_TRADE_MEMORY_LIMIT == 200
    assert TRADE_MEMORY_VERSION == 1


def test_snapshot_keeps_the_24h_tail_as_strict_json():
    manager = PortfolioManager(initial_capital=10_000)
    manager.trades = [
        _trade("2026-08-25T14:00:00Z", side="BUY"),
        _trade(
            "2026-08-26T14:00:00Z",
            shares=np.int64(10),
            price=np.float64(100.125),
            reason="[LLM] momentum (confidence: 90%)",
            cost=1001.25,
        ),
        _trade("2026-08-26T18:00:00Z", symbol="MSFT", side="SELL", shares=5.0, price=200.5),
    ]
    snap = manager.snapshot_state()
    # 08-25 14:00 is more than 24h before the newest trade; reason/cost are
    # not prompt fields and are not stored.
    assert snap["recent_trades"] == [
        _memory("2026-08-26T14:00:00+00:00", price=100.125),
        _memory("2026-08-26T18:00:00+00:00", symbol="MSFT", side="SELL", shares=5.0, price=200.5),
    ]
    assert json.loads(json.dumps(snap, allow_nan=False)) == snap


def test_snapshot_without_trades_has_empty_memory():
    assert PortfolioManager().snapshot_state()["recent_trades"] == []


def test_snapshot_memory_is_capped_to_the_newest_trades():
    manager = PortfolioManager(initial_capital=10_000)
    start = pd.Timestamp("2026-08-26T14:00:00Z")
    manager.trades = [
        _trade(start + pd.Timedelta(minutes=i), shares=i + 1) for i in range(250)
    ]
    memory = manager.snapshot_state()["recent_trades"]
    assert len(memory) == SNAPSHOT_TRADE_MEMORY_LIMIT
    assert memory[0]["shares"] == 51.0
    assert memory[-1]["shares"] == 250.0


def test_snapshot_skips_trade_rows_it_cannot_describe():
    manager = PortfolioManager(initial_capital=10_000)
    manager.trades = [
        _trade("2026-08-26T14:00:00Z"),
        {"symbol": "AAPL", "side": "BUY", "shares": 1, "price": 1.0},
        _trade("2026-08-26T15:00:00Z", price=float("nan")),
        _trade("2026-08-26T16:00:00Z", shares=None),
        {**_trade("2026-08-26T17:00:00Z"), "timestamp": "not-a-date"},
    ]
    assert manager.snapshot_state()["recent_trades"] == [
        _memory("2026-08-26T14:00:00+00:00"),
    ]


def test_restore_keeps_memory_out_of_the_trade_list():
    night1 = PortfolioManager(initial_capital=10_000)
    night1.trades = [
        _trade("2026-08-26T14:00:00Z"),
        _trade("2026-08-26T18:00:00Z", side="SELL", shares=4),
    ]
    stored = json.loads(json.dumps(night1.snapshot_state()))

    night2 = PortfolioManager(initial_capital=10_000)
    night2.restore_state(stored)
    assert night2.trades == []
    assert [t["timestamp"] for t in night2.restored_trades] == [
        pd.Timestamp("2026-08-26T14:00:00Z"),
        pd.Timestamp("2026-08-26T18:00:00Z"),
    ]
    assert all(t["timestamp"].tzinfo is not None for t in night2.restored_trades)
    assert [(t["side"], t["shares"]) for t in night2.restored_trades] == [
        ("BUY", 10.0),
        ("SELL", 4.0),
    ]


def test_an_old_snapshot_restores_no_memory():
    manager = PortfolioManager(initial_capital=10_000)
    manager.restore_state({"cash": 100.0, "positions": {"AAPL": 1}})
    assert manager.restored_trades == []
    assert manager.cash == 100.0


def test_restore_skips_corrupt_memory_rows():
    valid = _memory("2026-08-26T18:00:00+00:00")
    manager = PortfolioManager(initial_capital=10_000)
    manager.restore_state({
        "cash": 100.0,
        "positions": {},
        "recent_trades": [
            {k: v for k, v in valid.items() if k != "timestamp"},
            {**valid, "timestamp": "not-a-date"},
            {**valid, "timestamp": None},
            {**valid, "timestamp": 1756231200},
            {k: v for k, v in valid.items() if k != "symbol"},
            {**valid, "side": ""},
            {**valid, "price": "NaN"},
            {**valid, "shares": "ten"},
            "garbage",
            None,
            valid,
        ],
    })
    assert manager.restored_trades == [
        {**valid, "timestamp": pd.Timestamp("2026-08-26T18:00:00Z")},
    ]


@pytest.mark.parametrize("memory", ["not a list", {"AAPL": 1}, 7])
def test_restore_ignores_memory_that_is_not_a_list(memory):
    manager = PortfolioManager(initial_capital=10_000)
    manager.restore_state({"cash": 100.0, "positions": {}, "recent_trades": memory})
    assert manager.restored_trades == []


def test_restore_reads_a_naive_timestamp_as_utc():
    manager = PortfolioManager(initial_capital=10_000)
    manager.restore_state({
        "cash": 100.0,
        "recent_trades": [_memory("2026-08-26T18:00:00")],
    })
    assert manager.restored_trades[0]["timestamp"] == pd.Timestamp("2026-08-26T18:00:00Z")


def test_restore_caps_an_oversized_memory():
    start = pd.Timestamp("2026-08-26T14:00:00Z")
    rows = [
        _memory((start + pd.Timedelta(minutes=i)).isoformat(), shares=float(i + 1))
        for i in range(250)
    ]
    manager = PortfolioManager(initial_capital=10_000)
    manager.restore_state({"cash": 100.0, "recent_trades": rows})
    assert len(manager.restored_trades) == SNAPSHOT_TRADE_MEMORY_LIMIT
    assert manager.restored_trades[0]["shares"] == 51.0


def test_restore_puts_memory_in_time_order_before_capping_it():
    # The writer sorts, but a restore trusts no row: out-of-order memory is
    # still capped to the newest trades and read back oldest first.
    start = pd.Timestamp("2026-08-26T14:00:00Z")
    rows = [
        _memory((start + pd.Timedelta(minutes=i)).isoformat(), shares=float(i + 1))
        for i in range(250)
    ]
    manager = PortfolioManager(initial_capital=10_000)
    manager.restore_state({"cash": 100.0, "recent_trades": rows[::-1]})
    assert [t["shares"] for t in manager.restored_trades] == [
        float(n) for n in range(51, 251)
    ]


def test_prompt_shows_restored_trades_inside_24h_before_its_own(monkeypatch):
    manager = PortfolioManager(initial_capital=10_000)
    manager.restore_state({
        "cash": 10_000.0,
        "recent_trades": [
            _memory("2026-08-26T14:00:00+00:00", price=100.126),
            _memory("2026-08-26T18:00:00+00:00", price=100.126),
        ],
    })
    manager.trades = [_trade("2026-08-27T14:00:00Z", symbol="MSFT", side="SELL", shares=5)]

    memory = _prompt_memory(manager, monkeypatch, "2026-08-27T15:00:00Z")
    assert memory == [
        _memory("2026-08-26T18:00:00+00:00", price=100.13),
        {
            "symbol": "MSFT",
            "side": "SELL",
            "shares": 5,
            "price": 100.0,
            "timestamp": "2026-08-27T14:00:00+00:00",
        },
    ]
    assert manager.trades == [
        _trade("2026-08-27T14:00:00Z", symbol="MSFT", side="SELL", shares=5)
    ]


def test_a_weekend_empties_monday_mornings_memory(monkeypatch):
    friday = PortfolioManager(initial_capital=10_000)
    friday.trades = [_trade("2026-08-28T19:00:00Z")]
    monday = PortfolioManager(initial_capital=10_000)
    monday.restore_state(json.loads(json.dumps(friday.snapshot_state())))
    assert len(monday.restored_trades) == 1
    assert _prompt_memory(monday, monkeypatch, "2026-08-31T14:00:00Z") == []


def test_memory_chains_across_nights():
    night1 = PortfolioManager(initial_capital=10_000)
    night1.trades = [
        _trade("2026-08-26T14:00:00Z"),
        _trade("2026-08-26T18:00:00Z"),
    ]
    night2 = PortfolioManager(initial_capital=10_000)
    night2.restore_state(json.loads(json.dumps(night1.snapshot_state())))
    night2.trades = [_trade("2026-08-27T15:00:00Z", side="SELL", shares=20)]
    # Anchored on the new newest trade: 08-26 14:00 is now over 24h old.
    assert [
        (row["timestamp"], row["side"])
        for row in night2.snapshot_state()["recent_trades"]
    ] == [
        ("2026-08-26T18:00:00+00:00", "BUY"),
        ("2026-08-27T15:00:00+00:00", "SELL"),
    ]


def _decide_at(manager, ts):
    """Record one bar's equity point, as the decision loop does after each bar."""
    manager.update_equity({}, None, pd.Timestamp(ts))


def test_memory_expires_while_the_model_holds():
    manager = PortfolioManager(initial_capital=10_000)
    manager.trades = [_trade("2026-08-26T14:00:00Z")]
    _decide_at(manager, "2026-08-26T14:00:00Z")
    _decide_at(manager, "2026-08-27T13:00:00Z")
    # The next decision comes after the last bar, so a trade inside 24h of
    # that bar may still be shown and is kept.
    assert len(manager.snapshot_state()["recent_trades"]) == 1
    # Once the last bar is a full window past it, no later prompt can show
    # it, and a model that stopped trading does not carry it night after night.
    _decide_at(manager, "2026-08-27T14:00:00Z")
    assert manager.snapshot_state()["recent_trades"] == []


def test_snapshot_marks_the_memory_from_the_managers_first_bar():
    manager = PortfolioManager(initial_capital=10_000)
    snap = manager.snapshot_state()
    assert (snap["trade_memory"], snap["trade_memory_since"]) == (1, None)
    _decide_at(manager, "2026-08-26T14:00:00Z")
    _decide_at(manager, "2026-08-26T19:00:00Z")
    snap = manager.snapshot_state()
    assert (snap["trade_memory"], snap["trade_memory_since"]) == (
        1,
        "2026-08-26T14:00:00+00:00",
    )
    assert json.loads(json.dumps(snap, allow_nan=False)) == snap


def test_restore_carries_where_the_memory_started():
    night1 = PortfolioManager(initial_capital=10_000)
    _decide_at(night1, "2026-08-26T14:00:00Z")
    night2 = PortfolioManager(initial_capital=10_000)
    night2.restore_state(json.loads(json.dumps(night1.snapshot_state())))
    assert night2.trade_memory_since == pd.Timestamp("2026-08-26T14:00:00Z")
    _decide_at(night2, "2026-08-27T14:00:00Z")
    assert night2.snapshot_state()["trade_memory_since"] == "2026-08-26T14:00:00+00:00"


@pytest.mark.parametrize(
    "marker",
    [
        {},
        {"trade_memory_since": "2026-08-03T14:00:00+00:00"},
        {
            "trade_memory": TRADE_MEMORY_VERSION,
            "trade_memory_since": "not-a-date",
            "recent_trades": [],
        },
        {
            "trade_memory": TRADE_MEMORY_VERSION,
            "trade_memory_since": None,
            "recent_trades": [],
        },
    ],
    ids=["pre-marker", "stamp-without-marker", "unreadable-stamp", "no-stamp"],
)
def test_a_snapshot_without_a_readable_mark_restarts_it_here(marker, capsys):
    manager = PortfolioManager(initial_capital=10_000)
    manager.restore_state({"cash": 100.0, "positions": {}, **marker})
    assert manager.trade_memory_since is None
    _decide_at(manager, "2026-08-27T14:00:00Z")
    assert manager.snapshot_state()["trade_memory_since"] == "2026-08-27T14:00:00+00:00"
    assert "ERROR" not in capsys.readouterr().out


def test_an_empty_marked_memory_is_a_quiet_night_and_keeps_the_mark(capsys):
    manager = PortfolioManager(initial_capital=10_000)
    manager.restore_state({
        "cash": 100.0,
        "recent_trades": [],
        "trade_memory": TRADE_MEMORY_VERSION,
        "trade_memory_since": "2026-08-03T14:00:00+00:00",
    })
    assert manager.restored_trades == []
    assert manager.trade_memory_since == pd.Timestamp("2026-08-03T14:00:00Z")
    assert "ERROR" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "memory, found",
    [
        ({}, "missing"),
        ({"recent_trades": {"AAPL": 1}}, "dict"),
        ({"recent_trades": [{"symbol": "AAPL"}, "garbage"]}, "2"),
    ],
    ids=["key-missing", "not-a-list", "every-row-unreadable"],
)
def test_marked_memory_that_reads_back_as_nothing_is_an_error(memory, found, capsys):
    # Broken, not absent: the night opens with no memory, so the stamp must
    # restart here too or the row would claim this night had it.
    manager = PortfolioManager(initial_capital=10_000)
    manager.restore_state({
        "cash": 100.0,
        "trade_memory": TRADE_MEMORY_VERSION,
        "trade_memory_since": "2026-08-03T14:00:00+00:00",
        **memory,
    })
    assert manager.restored_trades == []
    assert manager.trade_memory_since is None
    assert (
        f"ERROR: portfolio.trade_memory_unreadable recent_trades={found} restored=0"
        in capsys.readouterr().out
    )
