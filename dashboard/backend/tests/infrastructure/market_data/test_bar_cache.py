"""Unit tests for the cross-process on-disk bar cache.

No network: this module never touches Alpaca. Every test points
ATL_BAR_CACHE_DIR at tmp_path, so nothing is written under
dashboard/storage/data.
"""

import json
import os
import time

import pandas as pd
import pytest

from dashboard.backend.infrastructure.market_data import bar_cache


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Enable the cache against an isolated directory."""
    monkeypatch.setenv("ATL_BAR_CACHE", "1")
    monkeypatch.setenv("ATL_BAR_CACHE_DIR", str(tmp_path / "bar_cache"))
    monkeypatch.delenv("ATL_BAR_CACHE_MAX_MB", raising=False)
    monkeypatch.delenv("ATL_BAR_CACHE_TTL_DAYS", raising=False)
    return tmp_path / "bar_cache"


def _frame(rows=3, start="2026-05-04T13:30:00Z"):
    index = pd.date_range(start, periods=rows, freq="5min", tz="UTC")
    index.name = "timestamp"
    return pd.DataFrame(
        {
            "open": [10.0 + i for i in range(rows)],
            "high": [11.0 + i for i in range(rows)],
            "low": [9.0 + i for i in range(rows)],
            "close": [10.5 + i for i in range(rows)],
            "volume": [100 * (i + 1) for i in range(rows)],
        },
        index=index,
    )


KEY = dict(start="2026-05-04", end="2026-05-12", source_timeframe="5m", feed="sip")
LAST_FETCH = {
    "feed": "sip",
    "source_timeframe": "5m",
    "requested_end": "2026-05-12",
    "effective_end": "2026-05-12",
    "sip_fallback_to_iex": False,
    "end_clamped": False,
}


# --- configuration surface -------------------------------------------------


def test_cache_is_enabled_by_default(monkeypatch):
    """Production default is ON. Pinned here rather than by enabling the
    cache suite-wide, which would change every existing market-data test."""
    monkeypatch.delenv("ATL_BAR_CACHE", raising=False)
    assert bar_cache.enabled() is True


def test_warm_is_enabled_by_default(monkeypatch):
    monkeypatch.delenv("ATL_BAR_CACHE_WARM", raising=False)
    assert bar_cache.warm_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF", " 0 "])
def test_falsey_values_disable_the_cache(monkeypatch, value):
    monkeypatch.setenv("ATL_BAR_CACHE", value)
    assert bar_cache.enabled() is False


def test_a_junk_flag_value_reads_as_off_and_warns(monkeypatch, capsys):
    """The only reason to set a kill switch is to turn it off. A typo'd value
    that kept the cache ON would defeat the switch at exactly the moment
    someone reached for it; `allow_recent_sip` in alpaca_bars.py already reads
    anything outside its truthy set as off, and this matches it -- adding only
    the warning, so the typo is visible."""
    monkeypatch.setenv("ATL_BAR_CACHE", "maybe")
    assert bar_cache.enabled() is False
    assert "ATL_BAR_CACHE" in capsys.readouterr().out


def test_a_junk_warm_flag_reads_as_off_without_touching_the_cache_flag(monkeypatch):
    monkeypatch.delenv("ATL_BAR_CACHE", raising=False)
    monkeypatch.setenv("ATL_BAR_CACHE_WARM", "fasle")
    assert bar_cache.warm_enabled() is False
    assert bar_cache.enabled() is True


def test_defaults_match_the_spec(monkeypatch):
    monkeypatch.delenv("ATL_BAR_CACHE_MAX_MB", raising=False)
    monkeypatch.delenv("ATL_BAR_CACHE_TTL_DAYS", raising=False)
    assert bar_cache.max_bytes() == 256 * 1024 * 1024
    assert bar_cache.ttl_seconds() == 7 * 86400.0


@pytest.mark.parametrize("value", ["not-a-number", "0", "-5", "99999"])
def test_out_of_range_size_cap_falls_back_and_logs(monkeypatch, capsys, value):
    monkeypatch.setenv("ATL_BAR_CACHE_MAX_MB", value)
    assert bar_cache.max_bytes() == 256 * 1024 * 1024
    assert "ATL_BAR_CACHE_MAX_MB" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["nope", "0", "-1", "400"])
def test_out_of_range_ttl_falls_back_and_logs(monkeypatch, capsys, value):
    monkeypatch.setenv("ATL_BAR_CACHE_TTL_DAYS", value)
    assert bar_cache.ttl_seconds() == 7 * 86400.0
    assert "ATL_BAR_CACHE_TTL_DAYS" in capsys.readouterr().out


def test_cache_dir_defaults_to_the_paths_constant(monkeypatch):
    from dashboard.backend.paths import BAR_CACHE_DIR

    monkeypatch.delenv("ATL_BAR_CACHE_DIR", raising=False)
    assert bar_cache.cache_dir() == BAR_CACHE_DIR
    assert BAR_CACHE_DIR.name == "bar_cache"
    assert BAR_CACHE_DIR.parent.name == "data"


def test_describe_names_the_state(cache_dir, monkeypatch):
    assert bar_cache.describe().startswith("bar cache: enabled (")
    monkeypatch.setenv("ATL_BAR_CACHE", "0")
    assert bar_cache.describe() == "bar cache: disabled"
