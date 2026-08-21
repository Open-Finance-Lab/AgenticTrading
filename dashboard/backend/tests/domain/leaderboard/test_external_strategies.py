"""Tests for the 14 external strategies added to the leaderboard baseline
registry from the "Strategy Lab" catalog (TradingAgents / QuantConnect /
freqtrade translations). Mirrors ``test_strategies_move.py``'s pattern:
registry identity, key resolution, required_symbols defaults, and a run()
smoke test against small synthetic hourly bars.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from dashboard.backend.domain.leaderboard import strategies as canon
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


def _assert_valid_curve(curve):
    assert isinstance(curve, list)
    if not curve:
        return
    for row in curve:
        assert set(row) >= {"timestamp", "equity", "cash", "positions_value", "daily_return"}
        assert np.isfinite(row["equity"])
        assert row["equity"] >= 0


@pytest.mark.parametrize(
    "key,cls",
    [(k, c) for k, c in _EXTERNAL_KEYS_TO_CLASSES.items() if k not in _SINGLE_SYMBOL_KEYS],
)
def test_run_produces_valid_curve_multi_symbol(key, cls, bars_30_symbols):
    strat = cls({})
    start = bars_30_symbols["SYM0"].index[0].date().isoformat()
    end = bars_30_symbols["SYM0"].index[-1].date().isoformat()
    curve = strat.run(bars_30_symbols, start, end, 100_000.0)
    _assert_valid_curve(curve)
    assert strat.num_trades() >= 0


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
    for key, cls in _EXTERNAL_KEYS_TO_CLASSES.items():
        if key in _SINGLE_SYMBOL_KEYS:
            continue
        strat = cls({})
        curve = strat.run(short_bars, start, end, 100_000.0)
        _assert_valid_curve(curve)
