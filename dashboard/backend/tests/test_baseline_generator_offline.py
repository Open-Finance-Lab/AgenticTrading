"""Baseline calculations must remain offline when bars are already supplied."""

from __future__ import annotations

import pandas as pd
import pytest
from datetime import date
from decimal import Decimal
from zoneinfo import ZoneInfo

from dashboard.backend import baseline_generator as baseline_module
from dashboard.backend.baseline_generator import BaselineGenerator
from dashboard.backend.domain.backtesting.market_rules import (
    ClosingLimitState,
    DailyMarketRule,
    MarketRuleCalendar,
)
from dashboard.backend.infrastructure.market_data.alpaca_bars import (
    MarketDataUnavailableError,
)
from dashboard.backend.infrastructure.market_data.profiles import (
    ASHARE_TRANSACTION_COST_PROFILE,
)


def sample_bars() -> dict[str, pd.DataFrame]:
    index = pd.date_range(
        "2026-04-01 10:00",
        periods=8,
        freq="h",
        tz="US/Eastern",
        name="timestamp",
    )
    return {
        "AAPL": pd.DataFrame(
            {
                "open": range(100, 108),
                "high": range(102, 110),
                "low": range(99, 107),
                "close": range(101, 109),
                "volume": [1_000] * 8,
            },
            index=index,
        ),
        "MSFT": pd.DataFrame(
            {
                "open": range(200, 208),
                "high": range(202, 210),
                "low": range(199, 207),
                "close": range(201, 209),
                "volume": [2_000] * 8,
            },
            index=index,
        ),
    }


def sample_cn_bars() -> dict[str, pd.DataFrame]:
    index = pd.DatetimeIndex(
        [
            "2026-04-01 10:30:00",
            "2026-04-01 11:30:00",
            "2026-04-01 14:00:00",
            "2026-04-01 15:00:00",
            "2026-04-02 10:30:00",
            "2026-04-02 11:30:00",
            "2026-04-02 14:00:00",
            "2026-04-02 15:00:00",
        ],
        tz=ZoneInfo("Asia/Shanghai"),
        name="timestamp",
    )
    return {
        symbol: pd.DataFrame(
            {
                "open": [100] * len(index),
                "high": [101] * len(index),
                "low": [99] * len(index),
                "close": [100 + row for row in range(len(index))],
                "volume": [1_000] * len(index),
            },
            index=index,
        )
        for symbol in ("600519.SH", "601318.SH")
    }


def test_constructor_and_supplied_bar_calculations_do_not_load_credentials(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Alpaca credentials must not be loaded")

    monkeypatch.setattr(BaselineGenerator, "_load_credentials", fail_if_called)
    generator = BaselineGenerator()
    bars = sample_bars()

    buyhold = generator.generate_buyhold_baseline(
        bars, "2026-04-01", "2026-04-02", initial_capital=100_000
    )
    index = generator.generate_index_baseline(
        bars, "2026-04-01", "2026-04-02", initial_capital=100_000
    )

    assert buyhold
    assert index
    assert buyhold[0]["equity"] > 0
    assert index[0]["equity"] > 0


def test_real_alpaca_fetch_loads_credentials_lazily(monkeypatch, tmp_path):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.setattr(baseline_module, "CREDENTIALS_DIR", tmp_path)

    generator = BaselineGenerator()

    with pytest.raises(MarketDataUnavailableError, match="credentials"):
        generator._fetch_bars_for_symbol("AAPL", "2026-04-01", "2026-04-02")


def test_cn_baselines_keep_shanghai_session_timestamps():
    bars = sample_cn_bars()

    buyhold, index = baseline_module.generate_baselines(
        bars,
        "2026-04-01",
        "2026-04-02",
        initial_capital=100_000,
        symbols_list=list(bars),
        market_timezone="Asia/Shanghai",
    )

    assert buyhold
    assert index
    assert buyhold[0]["timestamp"].startswith("2026-04-01T10:30:00+08:00")
    assert index[0]["timestamp"].startswith("2026-04-01T10:30:00+08:00")


def test_a_share_buyhold_baseline_accounts_for_initial_buy_costs():
    bars = sample_cn_bars()
    totals = {}
    curve = BaselineGenerator().generate_buyhold_baseline(
        bars,
        "2026-04-01",
        "2026-04-02",
        initial_capital=100_000,
        symbols_to_buy=list(bars),
        market_timezone="Asia/Shanghai",
        transaction_cost_profile=ASHARE_TRANSACTION_COST_PROFILE,
        transaction_cost_totals=totals,
    )

    assert curve
    assert totals["commission"] > 0
    assert totals["transfer_fee"] > 0
    assert totals["total_fees"] == pytest.approx(
        totals["commission"] + totals["transfer_fee"]
    )
    assert curve[0]["cash"] < 100_000


def wide_price_cn_bars(prices: dict[str, float]) -> dict[str, pd.DataFrame]:
    """CN bars at a fixed price per symbol, spanning realistic A-share levels."""
    index = pd.DatetimeIndex(
        ["2026-04-01 10:30:00", "2026-04-01 11:30:00"],
        tz=ZoneInfo("Asia/Shanghai"),
        name="timestamp",
    )
    return {
        symbol: pd.DataFrame(
            {
                "open": [price] * len(index),
                "high": [price] * len(index),
                "low": [price] * len(index),
                "close": [price] * len(index),
                "volume": [1_000] * len(index),
            },
            index=index,
        )
        for symbol, price in prices.items()
    }


# CSI300-shaped: a 200x spread between the cheapest and dearest name, which is
# what makes an equal slice smaller than one 100-share lot on a small account.
_CSI300_SHAPED_PRICES = {
    f"{600000 + i}.SH": price
    for i, price in enumerate(
        [
            7.9, 12.4, 18.6, 24.1, 31.7, 38.2, 44.9, 55.3, 62.8, 71.5,
            88.0, 104.2, 133.6, 162.9, 198.4, 251.0, 318.7, 462.3, 940.5, 1680.0,
        ]
    )
}


@pytest.mark.parametrize("initial_capital", [1_000, 3_000])
def test_lot_constrained_baseline_stays_invested_at_real_capital(initial_capital):
    """The benchmark every A-share agent is scored against must hold stock.

    An equal slice of a $1,000 account across 20 names is ~¥350 — less than one
    100-share lot of anything here. Flooring each slice on its own therefore
    put the WHOLE sleeve in cash: a flat line that every agent beats for free,
    and that reads exactly like a legitimately-flat benchmark.
    """
    bars = wide_price_cn_bars(_CSI300_SHAPED_PRICES)
    summary: dict = {}
    totals: dict = {}
    curve = BaselineGenerator().generate_buyhold_baseline(
        bars,
        "2026-04-01",
        "2026-04-01",
        initial_capital=initial_capital,
        symbols_to_buy=list(bars),
        market_timezone="Asia/Shanghai",
        transaction_cost_profile=ASHARE_TRANSACTION_COST_PROFILE,
        transaction_cost_totals=totals,
        lot_size=100,
        allocation_summary=summary,
        currency_context=None,
    )

    assert curve
    assert summary["symbols_bought"] >= 1
    assert summary["invested_ratio"] > 0.5, "benchmark collapsed into cash"
    assert totals["total_fees"] > 0
    assert curve[0]["positions_value"] > 0


def test_baseline_allocation_summary_reports_a_partial_sleeve():
    """Absent must not look like broken.

    Without these counts a sleeve that placed 2 of 20 names is byte-identical
    to one that correctly placed all of them, which is how the collapse above
    stayed invisible.
    """
    bars = wide_price_cn_bars(_CSI300_SHAPED_PRICES)
    summary: dict = {}
    BaselineGenerator().generate_buyhold_baseline(
        bars,
        "2026-04-01",
        "2026-04-01",
        initial_capital=1_000,
        symbols_to_buy=list(bars),
        market_timezone="Asia/Shanghai",
        transaction_cost_profile=ASHARE_TRANSACTION_COST_PROFILE,
        lot_size=100,
        allocation_summary=summary,
    )

    assert summary["symbols_requested"] == len(_CSI300_SHAPED_PRICES)
    assert summary["symbols_priced"] == len(_CSI300_SHAPED_PRICES)
    assert summary["lot_size"] == 100
    assert (
        summary["symbols_bought"] + summary["symbols_skipped"]
        == summary["symbols_requested"]
    )
    assert 0 < summary["invested_ratio"] <= 1


def test_lot_size_comes_from_the_market_not_the_cost_profile():
    """Board lot and fee schedule are separate market rules.

    Inferring one from the other floors buys to 100 in any future market that
    charges fees but trades single shares.
    """
    bars = wide_price_cn_bars({"600000.SH": 7.9, "600001.SH": 12.4})
    single_share: dict = {}
    BaselineGenerator().generate_buyhold_baseline(
        bars,
        "2026-04-01",
        "2026-04-01",
        initial_capital=1_000,
        symbols_to_buy=list(bars),
        market_timezone="Asia/Shanghai",
        transaction_cost_profile=ASHARE_TRANSACTION_COST_PROFILE,
        lot_size=1,
        allocation_summary=single_share,
    )

    assert single_share["lot_size"] == 1
    # Costs are on, lots are off: both names fill, and not in multiples of 100.
    assert single_share["symbols_bought"] == 2


def test_baseline_composition_does_not_depend_on_symbol_order():
    """A symbol must spend its own slice, never its neighbours' cash.

    Comparing affordability against the running total let whichever symbol
    happened to iterate first eat the rest of the sleeve's budget.
    """
    prices = dict(list(_CSI300_SHAPED_PRICES.items())[:8])
    forward: dict = {}
    reverse: dict = {}
    for order, summary in ((list(prices), forward), (list(reversed(prices)), reverse)):
        BaselineGenerator().generate_buyhold_baseline(
            wide_price_cn_bars({s: prices[s] for s in order}),
            "2026-04-01",
            "2026-04-01",
            initial_capital=3_000,
            symbols_to_buy=order,
            market_timezone="Asia/Shanghai",
            transaction_cost_profile=ASHARE_TRANSACTION_COST_PROFILE,
            lot_size=100,
            allocation_summary=summary,
        )

    assert forward["symbols_bought"] == reverse["symbols_bought"]
    assert forward["invested_ratio"] == pytest.approx(reverse["invested_ratio"])


def test_minimum_commission_is_charged_once_per_symbol_not_per_lot():
    """The ¥5 floor is a per-ORDER minimum.

    The top-up sweep grows positions one lot at a time; costing each lot as its
    own order would charge the floor once per lot instead of once per symbol.
    """
    bars = wide_price_cn_bars({"600000.SH": 7.9})
    totals: dict = {}
    summary: dict = {}
    BaselineGenerator().generate_buyhold_baseline(
        bars,
        "2026-04-01",
        "2026-04-01",
        initial_capital=1_000,
        symbols_to_buy=list(bars),
        market_timezone="Asia/Shanghai",
        transaction_cost_profile=ASHARE_TRANSACTION_COST_PROFILE,
        transaction_cost_totals=totals,
        lot_size=100,
        allocation_summary=summary,
    )

    # ¥7,000 at ¥7.9 buys many lots; commission must still be a single ¥5 floor
    # (the percentage fee on this notional is well under it).
    assert summary["symbols_bought"] == 1
    assert totals["commission"] == pytest.approx(
        ASHARE_TRANSACTION_COST_PROFILE.minimum_commission
    )


def test_us_baseline_is_untouched_by_the_lot_and_cost_plumbing():
    """No profile and no board lot must reproduce the pre-existing curve."""
    bars = sample_bars()
    generator = BaselineGenerator()

    before = generator.generate_buyhold_baseline(
        bars, "2026-04-01", "2026-04-02", initial_capital=100_000
    )
    after = generator.generate_buyhold_baseline(
        bars,
        "2026-04-01",
        "2026-04-02",
        initial_capital=100_000,
        lot_size=1,
        allocation_summary={},
    )

    assert before == after
    # 100k over two names at 101/201 -> 495 and 248 whole shares, no top-up.
    assert before[0]["positions_value"] == pytest.approx(495 * 101 + 248 * 201)


def test_a_share_baseline_retries_market_blocked_initial_buy_on_later_bar():
    index = pd.DatetimeIndex(
        [
            "2026-04-01 15:00:00",
            "2026-04-02 10:30:00",
            "2026-04-02 15:00:00",
        ],
        tz=ZoneInfo("Asia/Shanghai"),
        name="timestamp",
    )
    bars = {
        "600519.SH": pd.DataFrame(
            {"close": [10.0, 12.0, 13.0]},
            index=index,
        )
    }
    calendar = MarketRuleCalendar([
        DailyMarketRule(
            symbol="600519.SH",
            trading_date=date(2026, 4, 1),
            suspended=True,
        ),
        DailyMarketRule(
            symbol="600519.SH",
            trading_date=date(2026, 4, 2),
            suspended=False,
            closing_limit_state=ClosingLimitState.NONE,
            official_close_price=Decimal("13.00"),
            final_bar_timestamp=index[-1].to_pydatetime(),
        ),
    ])
    summary = {}
    totals = {}

    curve = BaselineGenerator().generate_buyhold_baseline(
        bars,
        "2026-04-01",
        "2026-04-02",
        initial_capital=10_000,
        symbols_to_buy=["600519.SH"],
        market_timezone="Asia/Shanghai",
        transaction_cost_profile=ASHARE_TRANSACTION_COST_PROFILE,
        transaction_cost_totals=totals,
        lot_size=100,
        allocation_summary=summary,
        market_rule_calendar=calendar,
    )

    assert curve[0]["cash"] == 10_000
    assert curve[0]["positions_value"] == 0
    assert curve[1]["cash"] < 10_000
    assert curve[1]["positions_value"] > 0
    assert summary["symbols_delayed"] == 1
    assert summary["symbols_unfilled"] == 0
    assert summary["symbols_bought"] == 1
    assert totals["total_fees"] > 0


def test_a_share_baseline_holds_a_blocked_symbols_slice_for_its_retry():
    """A blocked name's slice must survive the top-up sweep of its neighbours.

    The sweep exists to place cash whole lots stranded, and it is blind to why
    a symbol is absent from the price map. Left alone it spends the suspended
    name's slice on the tradable one at the open, so the retry finds no cash,
    and the equal-weight benchmark every agent is scored against silently
    becomes a single-name position.
    """
    index = pd.DatetimeIndex(
        [
            "2026-04-01 15:00:00",
            "2026-04-02 10:30:00",
            "2026-04-02 15:00:00",
        ],
        tz=ZoneInfo("Asia/Shanghai"),
        name="timestamp",
    )
    bars = {
        "600519.SH": pd.DataFrame({"close": [10.0, 10.0, 10.0]}, index=index),
        "600520.SH": pd.DataFrame({"close": [10.0, 10.0, 10.0]}, index=index),
    }
    rules = [
        # 600519 is suspended for the whole opening day, so its slice can only
        # be placed on a later bar.
        DailyMarketRule(
            symbol="600519.SH",
            trading_date=date(2026, 4, 1),
            suspended=True,
        ),
    ]
    for symbol in ("600519.SH", "600520.SH"):
        rules.append(
            DailyMarketRule(
                symbol=symbol,
                trading_date=date(2026, 4, 2),
                suspended=False,
                closing_limit_state=ClosingLimitState.NONE,
                official_close_price=Decimal("10.00"),
                final_bar_timestamp=index[-1].to_pydatetime(),
            )
        )
    rules.append(
        DailyMarketRule(
            symbol="600520.SH",
            trading_date=date(2026, 4, 1),
            suspended=False,
            closing_limit_state=ClosingLimitState.NONE,
            official_close_price=Decimal("10.00"),
            final_bar_timestamp=index[0].to_pydatetime(),
        )
    )
    summary = {}

    curve = BaselineGenerator().generate_buyhold_baseline(
        bars,
        "2026-04-01",
        "2026-04-02",
        initial_capital=100_000,
        symbols_to_buy=["600519.SH", "600520.SH"],
        market_timezone="Asia/Shanghai",
        transaction_cost_profile=ASHARE_TRANSACTION_COST_PROFILE,
        lot_size=100,
        allocation_summary=summary,
        market_rule_calendar=MarketRuleCalendar(rules),
    )

    # The tradable name takes its own ¥50k slice and no more; the rest is held
    # for the suspended name rather than swept into a second sleeve.
    assert curve[0]["cash"] == pytest.approx(50_000, abs=1_500)
    assert curve[0]["positions_value"] == pytest.approx(49_000, abs=1_500)
    # Which is what lets the retry actually fill on the next day's bar.
    assert summary["symbols_delayed"] == 1
    assert summary["symbols_unfilled"] == 0
    assert summary["symbols_bought"] == 2
    assert curve[-1]["positions_value"] == pytest.approx(98_000, abs=3_000)
