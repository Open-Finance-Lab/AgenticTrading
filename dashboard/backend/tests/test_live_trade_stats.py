"""Unit tests for the Live board's folded trade statistics."""

from datetime import datetime, timezone

from dashboard.backend.domain.leaderboard.live_trade_stats import (
    fold_trade_stats,
    summarize_trade_stats,
)


def _t(hour, side, shares, price, symbol="AAPL"):
    return {
        "timestamp": datetime(2026, 10, 1, hour, 0, tzinfo=timezone.utc),
        "side": side,
        "shares": shares,
        "price": price,
        "symbol": symbol,
    }


def test_fold_measures_hold_time_and_win_rate():
    stats = fold_trade_stats(None, [
        _t(10, "BUY", 10, 100),
        _t(16, "SELL", 10, 110),
        _t(10, "BUY", 4, 50, "MSFT"),
        _t(12, "SELL", 4, 40, "MSFT"),
    ])
    figures = summarize_trade_stats(stats)
    assert figures["closed_trades"] == 2
    assert figures["win_rate"] == 0.5
    # 10 shares × 6h + 4 shares × 2h = 68 share-hours / 14 shares
    assert figures["avg_hold_hours"] == 68 / 14


def test_a_prior_row_without_the_fold_is_marked_incomplete():
    figures = summarize_trade_stats(fold_trade_stats(
        None, [_t(10, "BUY", 1, 10)], prior_has_history=True
    ))
    assert figures == {"win_rate": None, "avg_hold_hours": None, "closed_trades": None}


def test_resume_keeps_an_open_lot():
    first = fold_trade_stats(None, [_t(10, "BUY", 8, 20)])
    later = fold_trade_stats(first, [
        {"timestamp": datetime(2026, 10, 2, 10, 0, tzinfo=timezone.utc),
         "side": "SELL", "shares": 8, "price": 25, "symbol": "AAPL"},
    ])
    figures = summarize_trade_stats(later)
    assert figures["closed_trades"] == 1
    assert figures["avg_hold_hours"] == 24
    assert figures["win_rate"] == 1.0
