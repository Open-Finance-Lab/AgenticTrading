"""Tests for the 14 external strategies added to the leaderboard baseline
registry from the "Strategy Lab" catalog (TradingAgents / QuantConnect /
freqtrade translations). Mirrors ``test_strategies_move.py``'s pattern:
registry identity, key resolution, required_symbols defaults, and a run()
smoke test against small synthetic hourly bars.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pandas as pd
import pytest

from dashboard.backend.domain.leaderboard import strategies as canon
from dashboard.backend.domain.leaderboard.strategies import _signal_engine
from dashboard.backend.domain.leaderboard.strategies._common import build_price_cache
from dashboard.backend.domain.leaderboard.strategies.almgren_chriss_twap import AlmgrenChrissTwapStrategy
from dashboard.backend.domain.leaderboard.strategies.bandtastic import BandtasticStrategy
from dashboard.backend.domain.leaderboard.strategies.capm_alpha_ranking import CAPMAlphaRankingStrategy
from dashboard.backend.domain.leaderboard.strategies.hlhb import HlhbStrategy
from dashboard.backend.domain.leaderboard.strategies.momentum_effect import MomentumEffectStrategy
from dashboard.backend.domain.leaderboard.strategies.overnight_anomaly import OvernightAnomalyStrategy
from dashboard.backend.domain.leaderboard.strategies.pattern_recognition import PatternRecognitionStrategy
from dashboard.backend.domain.leaderboard.strategies.short_term_reversal import ShortTermReversalStrategy
from dashboard.backend.domain.leaderboard.strategies.supertrend_triple import SupertrendTripleStrategy
from dashboard.backend.domain.leaderboard.strategies.trendrider import TrendRiderStrategy
from dashboard.backend.domain.leaderboard.strategies.turn_of_month import TurnOfMonthStrategy
from dashboard.backend.domain.leaderboard.strategies.universal_macd import UniversalMACDStrategy
from dashboard.backend.domain.leaderboard.strategies.volatility_effect import VolatilityEffectStrategy
from dashboard.backend.domain.leaderboard.strategies.tradingagents_composite import (
    TradingAgentsCompositeStrategy,
)

_EXTERNAL_KEYS_TO_CLASSES = {
    "tradingagents_composite": TradingAgentsCompositeStrategy,
    "capm_alpha_ranking": CAPMAlphaRankingStrategy,
    "momentum_effect": MomentumEffectStrategy,
    "volatility_effect": VolatilityEffectStrategy,
    "short_term_reversal": ShortTermReversalStrategy,
    "overnight_anomaly": OvernightAnomalyStrategy,
    "turn_of_month": TurnOfMonthStrategy,
    "bandtastic": BandtasticStrategy,
    "supertrend_triple": SupertrendTripleStrategy,
    "hlhb": HlhbStrategy,
    "trendrider": TrendRiderStrategy,
    "pattern_recognition": PatternRecognitionStrategy,
    "universal_macd": UniversalMACDStrategy,
    "almgren_chriss_twap": AlmgrenChrissTwapStrategy,
}

_SINGLE_SYMBOL_KEYS = {"overnight_anomaly", "turn_of_month"}
_MULTI_SYMBOL_ITEMS = [(k, c) for k, c in _EXTERNAL_KEYS_TO_CLASSES.items() if k not in _SINGLE_SYMBOL_KEYS]

# The synthetic fixtures are keyed SYM0..SYM29, not the DJIA-30 default
# universe, so every multi-symbol run() must be told which symbols to use --
# otherwise ``subset_bars`` finds nothing, run() returns [] before touching a
# single indicator, and the smoke test passes without exercising anything.
_SYNTHETIC_SYMBOLS = [f"SYM{i}" for i in range(30)]


# ---------------------------------------------------------------------------
# Registry identity + resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key,cls", _EXTERNAL_KEYS_TO_CLASSES.items())
def test_registry_identity(key, cls):
    registry = canon.available_strategies()
    assert registry[key] is cls


@pytest.mark.parametrize("key,cls", _EXTERNAL_KEYS_TO_CLASSES.items())
def test_get_strategy_resolves_by_key(key, cls):
    strat = canon.get_strategy({"id": "x", "name": "X", "strategy": key})
    assert isinstance(strat, cls)
    assert strat.id == "x"
    assert strat.name == "X"


@pytest.mark.parametrize("key,cls", _EXTERNAL_KEYS_TO_CLASSES.items())
def test_required_symbols_defaults_and_overrides(key, cls):
    strat = cls({})
    default = strat.required_symbols()
    if key in _SINGLE_SYMBOL_KEYS:
        assert default == ["SPY"]
        custom = cls({"symbols": ["AAPL"]}).required_symbols()
        assert custom == ["AAPL"]
    else:
        assert len(default) == 30
        custom = cls({"symbols": ["AAPL", "MSFT"]}).required_symbols()
        assert custom == ["AAPL", "MSFT"]


# ---------------------------------------------------------------------------
# run() smoke tests against small synthetic hourly bars
# ---------------------------------------------------------------------------

_ET_HOURS = ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30", "16:00"]


def _hourly_index(n_days: int, start: str = "2026-01-05") -> pd.DatetimeIndex:
    """n_days of market-hours hourly timestamps, tz-aware UTC (matching
    Alpaca's actual bar timestamps), skipping weekends."""
    timestamps = []
    day = pd.Timestamp(start, tz="US/Eastern")
    added = 0
    while added < n_days:
        if day.weekday() < 5:
            for hhmm in _ET_HOURS:
                h, m = map(int, hhmm.split(":"))
                timestamps.append(day.replace(hour=h, minute=m).tz_convert("UTC"))
            added += 1
        day = day + pd.Timedelta(days=1)
    return pd.DatetimeIndex(timestamps)


def _synthetic_bars(n_days: int, seed: int, base_price: float = 100.0) -> pd.DataFrame:
    idx = _hourly_index(n_days)
    rng = np.random.default_rng(seed)
    n = len(idx)
    rets = rng.normal(loc=0.0003, scale=0.004, size=n)
    close = base_price * np.cumprod(1 + rets)
    open_ = np.concatenate([[base_price], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.002, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.002, n))
    volume = rng.integers(1_000, 10_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture(scope="module")
def bars_30_symbols():
    symbols = [f"SYM{i}" for i in range(30)]
    return {sym: _synthetic_bars(40, seed=i) for i, sym in enumerate(symbols)}


@pytest.fixture(scope="module")
def bars_spy_only():
    return {"SPY": _synthetic_bars(40, seed=999)}


def _assert_valid_curve(curve, expected_len=None):
    assert isinstance(curve, list)
    if expected_len is not None:
        # One row per hourly bar in the window; an empty curve here means the
        # strategy bailed before running, which is exactly what this must catch.
        assert len(curve) == expected_len
    for row in curve:
        assert set(row) >= {"timestamp", "equity", "cash", "positions_value", "daily_return"}
        assert np.isfinite(row["equity"])
        assert row["equity"] >= 0


@pytest.mark.parametrize("key,cls", _MULTI_SYMBOL_ITEMS)
def test_run_produces_valid_curve_multi_symbol(key, cls, bars_30_symbols):
    strat = cls({"symbols": _SYNTHETIC_SYMBOLS})
    start = bars_30_symbols["SYM0"].index[0].date().isoformat()
    end = bars_30_symbols["SYM0"].index[-1].date().isoformat()
    curve = strat.run(bars_30_symbols, start, end, 100_000.0)
    _assert_valid_curve(curve, expected_len=len(bars_30_symbols["SYM0"]))
    assert strat.num_trades() >= 0


@pytest.mark.parametrize("key,cls", _MULTI_SYMBOL_ITEMS)
def test_run_with_default_universe_and_foreign_bars_returns_empty(key, cls, bars_30_symbols):
    """The default DJIA-30 universe shares no symbol with the synthetic bars, so
    run() has nothing to trade and says so with [] rather than a partial curve."""
    assert cls({}).run(bars_30_symbols, "2026-01-05", "2026-01-09", 100_000.0) == []


@pytest.mark.parametrize("key,cls", [(k, c) for k, c in _EXTERNAL_KEYS_TO_CLASSES.items() if k in _SINGLE_SYMBOL_KEYS])
def test_run_produces_valid_curve_single_symbol(key, cls, bars_spy_only):
    strat = cls({})
    start = bars_spy_only["SPY"].index[0].date().isoformat()
    end = bars_spy_only["SPY"].index[-1].date().isoformat()
    curve = strat.run(bars_spy_only, start, end, 100_000.0)
    _assert_valid_curve(curve)
    # Every hourly bar in the window should produce one curve row.
    assert len(curve) == len(bars_spy_only["SPY"])


def test_run_with_empty_bars_returns_empty_list():
    for cls in _EXTERNAL_KEYS_TO_CLASSES.values():
        strat = cls({})
        assert strat.run({}, "2026-01-05", "2026-01-06", 100_000.0) == []


def test_run_with_insufficient_history_does_not_crash(bars_30_symbols):
    """A contest window shorter than a strategy's desired lookback (the real
    leaderboard's ~1-month window vs. e.g. momentum_effect's 252-day
    default) must degrade gracefully, never raise or NaN out equity."""
    short_bars = {sym: df.iloc[:8] for sym, df in bars_30_symbols.items()}  # 1 trading day
    start = short_bars["SYM0"].index[0].date().isoformat()
    end = short_bars["SYM0"].index[-1].date().isoformat()
    for key, cls in _MULTI_SYMBOL_ITEMS:
        strat = cls({"symbols": _SYNTHETIC_SYMBOLS})
        curve = strat.run(short_bars, start, end, 100_000.0)
        _assert_valid_curve(curve, expected_len=8)
        # Nothing to go on yet: the whole book stays in cash.
        assert all(row["cash"] == pytest.approx(100_000.0) for row in curve), key


# ---------------------------------------------------------------------------
# Reference window: weight_fn on contest day 0 sees only pre-start history
# ---------------------------------------------------------------------------

def test_engine_feeds_reference_history_before_contest_start():
    """The leaderboard fetches bars from ``reference_start_date`` (a month
    before the contest) so a strategy has lookback on day 0. The engine must
    hand day 0's weight_fn exactly the daily rows before ``start_date`` -- the
    reference period -- and never the current day's own bars (no look-ahead)."""
    bars = {"SYM0": _synthetic_bars(30, seed=1), "SYM1": _synthetic_bars(30, seed=2)}
    all_days = sorted({ts.tz_convert("US/Eastern").date() for ts in bars["SYM0"].index})
    start_day = all_days[20]
    seen = []

    def weight_fn(history, cur_date, day_index):
        seen.append((day_index, cur_date, history.close.copy()))
        return {}

    curve, _ = _signal_engine.run_daily_signal_strategy(
        bars, start_day.isoformat(), all_days[-1].isoformat(), 100_000.0, weight_fn,
    )
    assert len(curve) == 10 * len(_ET_HOURS)
    day_index, cur_date, history_close = seen[0]
    assert day_index == 0 and cur_date == start_day
    assert len(history_close) == 20
    assert history_close.index.max() < pd.Timestamp(start_day)
    assert list(history_close.columns) == ["SYM0", "SYM1"]
    # The reference rows are the daily last-close of the pre-start bars.
    ref_bars = bars["SYM0"][bars["SYM0"].index.tz_convert("US/Eastern").date < start_day]
    expected_last_close = float(ref_bars["close"].iloc[-1])
    assert history_close["SYM0"].iloc[-1] == pytest.approx(expected_last_close)
    # Every later day still sees strictly-before-today history only.
    for day_index, cur_date, history_close in seen[1:]:
        assert history_close.index.max() < pd.Timestamp(cur_date)


# ---------------------------------------------------------------------------
# build_price_cache: per-symbol coverage
# ---------------------------------------------------------------------------

def test_build_price_cache_prices_a_late_symbol_from_its_own_first_bar():
    full = _synthetic_bars(3, seed=5)
    late = _synthetic_bars(3, seed=6).iloc[8:]  # no bars on the first day
    timestamps = list(full.index)
    cache = build_price_cache({"FULL": full, "LATE": late, "NONE": full.iloc[:0]}, timestamps)
    assert set(cache) == {"FULL", "LATE"}
    assert len(cache["FULL"]) == len(timestamps)
    # LATE is unpriced on day 1 (not tradable yet) and priced from its own
    # first bar onward -- it is not dropped for the whole run.
    assert all(ts not in cache["LATE"] for ts in timestamps[:8])
    assert all(ts in cache["LATE"] for ts in timestamps[8:])
    assert cache["LATE"][timestamps[8]] == pytest.approx(float(late["close"].iloc[0]))


def test_build_price_cache_forward_fills_and_takes_last_of_duplicate_stamp():
    full = _synthetic_bars(2, seed=7)
    gappy = full.iloc[[0, 1, 2, 2, 5]].copy()  # a duplicated stamp and gaps
    gappy.iloc[3, gappy.columns.get_loc("close")] = 123.0
    cache = build_price_cache({"G": gappy}, list(full.index))
    assert cache["G"][full.index[2]] == 123.0  # last bar of the duplicate wins
    assert cache["G"][full.index[3]] == 123.0  # forward-filled through the gap
    assert cache["G"][full.index[4]] == 123.0
    assert cache["G"][full.index[5]] == pytest.approx(float(gappy["close"].iloc[4]))


# ---------------------------------------------------------------------------
# Strategy-specific regressions
# ---------------------------------------------------------------------------

def _spy_bars_between(start: str, n_days: int, seed: int = 11) -> pd.DataFrame:
    return _synthetic_bars(n_days, seed=seed).set_index(_hourly_index(n_days, start=start))


def test_turn_of_month_does_not_treat_the_last_contest_day_as_a_month_end():
    # 2026-01-26 .. 2026-02-06 (10 trading days): one real month-end, Fri 01-30.
    df = _spy_bars_between("2026-01-26", 10)
    strat = TurnOfMonthStrategy({})
    curve = strat.run({"SPY": df}, "2026-01-26", "2026-02-06", 100_000.0)
    assert len(curve) == len(df)
    # Buy at the 01-30 open, sell at the close three trading days later (02-04):
    # exactly two trades, and the book is back in cash on the final day.
    assert strat.num_trades() == 2
    assert curve[-1]["positions_value"] == 0
    assert curve[-1]["cash"] > 0
    by_day = {}
    for row in curve:
        by_day.setdefault(row["timestamp"][:10], row)
    assert by_day["2026-01-30"]["cash"] == 0  # invested at the month-end open
    assert by_day["2026-02-05"]["positions_value"] == 0  # flat again after the hold


def test_turn_of_month_skips_a_zero_open_without_starting_a_hold():
    df = _spy_bars_between("2026-01-26", 10)
    first_bar_0130 = [ts for ts in df.index if ts.tz_convert("US/Eastern").date().isoformat() == "2026-01-30"][0]
    df.loc[first_bar_0130, "open"] = 0.0
    strat = TurnOfMonthStrategy({})
    curve = strat.run({"SPY": df}, "2026-01-26", "2026-02-06", 100_000.0)
    assert strat.num_trades() == 0
    assert all(row["cash"] == 100_000.0 and row["positions_value"] == 0 for row in curve)


def test_capm_alpha_ranking_gives_a_lone_qualifier_the_whole_book():
    """Only one symbol clears the regression's valid-day floor: it gets weight
    1.0, not the hardcoded 0.5 that parked half the capital in cash."""
    good = _synthetic_bars(45, seed=21)
    # Too few daily rows for the regression: excluded from the ranking.
    sparse = _synthetic_bars(45, seed=22).iloc[-5 * len(_ET_HOURS):]
    days = sorted({ts.tz_convert("US/Eastern").date() for ts in good.index})
    strat = CAPMAlphaRankingStrategy({"symbols": ["GOOD", "SPARSE"]})
    curve = strat.run({"GOOD": good, "SPARSE": sparse}, days[35].isoformat(), days[-1].isoformat(), 100_000.0)
    assert len(curve) == 10 * len(_ET_HOURS)
    assert strat.num_trades() >= 1
    first_day = curve[0]
    assert first_day["cash"] < 0.01 * first_day["equity"]
    assert first_day["positions_value"] > 0.99 * first_day["equity"]


def test_bandtastic_trades_within_the_leaderboard_sized_window():
    """The reference buffer + contest is ~44 trading days, below the strategy's
    nominal 50-day slow EMA; with the span capped to the available history the
    strategy must still be able to enter on a clear band dip."""
    n_days = 43
    idx = _hourly_index(n_days)
    per_day = len(_ET_HOURS)
    daily = np.full(n_days, 100.0)
    for i in range(1, n_days):
        daily[i] = daily[i - 1] * (1.006 if i < 30 else (0.96 if i < 33 else 1.002))
    close = np.repeat(daily, per_day)
    df = pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1000.0},
        index=idx,
    )
    days = sorted({ts.tz_convert("US/Eastern").date() for ts in idx})
    strat = BandtasticStrategy({"symbols": ["DIP"]})
    curve = strat.run({"DIP": df}, days[21].isoformat(), days[-1].isoformat(), 100_000.0)
    assert len(curve) == (n_days - 21) * per_day
    assert strat.num_trades() >= 1
    assert any(row["positions_value"] > 0 for row in curve)


def test_pct_change_never_pads_gaps_into_zero_returns():
    """pandas' ``pct_change`` default forward-fills a missing close, turning a
    data gap into a 0% return day; every return series the strategies rank on
    must opt out (``fill_method=None``)."""
    strategies_dir = pathlib.Path(_signal_engine.__file__).parent
    calls = []
    for path in sorted(strategies_dir.glob("*.py")):
        for match in re.finditer(r"pct_change\(([^)]*)\)", path.read_text()):
            calls.append((path.name, match.group(1)))
    assert calls, "expected at least one pct_change call in the strategies package"
    offenders = [(name, args) for name, args in calls if "fill_method=None" not in args]
    assert offenders == []
