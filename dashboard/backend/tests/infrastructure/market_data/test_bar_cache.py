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


# --- store and restore -----------------------------------------------------


def test_write_then_read_returns_an_equal_frame(cache_dir):
    frame = _frame()
    assert bar_cache.write_many({"AAPL": frame}, last_fetch=LAST_FETCH, **KEY) == 1
    hits, metas = bar_cache.read_many(["AAPL"], **KEY)
    assert set(hits) == {"AAPL"}
    # check_freq=False: parquet has no slot for DatetimeIndex.freq, so the
    # fixture's freq="5min" comes back as None. Nothing downstream reads freq
    # (aggregate_bars_by_symbol resamples from the timestamps), so the values,
    # dtypes, index name and tz are the contract -- and those are all checked.
    pd.testing.assert_frame_equal(hits["AAPL"], frame, check_freq=False)
    assert metas["AAPL"] == LAST_FETCH


def test_read_restores_the_three_attrs_stamps(cache_dir):
    """feed_provenance() reads these back and the engine persists the result
    into agent_runs.metadata. A hit that loses them is a silent behaviour
    change, not a speed-up."""
    frame = _frame()
    frame.attrs["alpaca_feed"] = "sip"
    frame.attrs["alpaca_sip_fallback"] = False
    frame.attrs["alpaca_end_clamped"] = False
    bar_cache.write_many({"AAPL": frame}, last_fetch=LAST_FETCH, **KEY)
    hits, _ = bar_cache.read_many(["AAPL"], **KEY)
    assert hits["AAPL"].attrs == {
        "alpaca_feed": "sip",
        "alpaca_sip_fallback": False,
        "alpaca_end_clamped": False,
    }


def test_read_preserves_the_index_name_and_timezone(cache_dir):
    frame = _frame()
    bar_cache.write_many({"AAPL": frame}, last_fetch=LAST_FETCH, **KEY)
    hits, _ = bar_cache.read_many(["AAPL"], **KEY)
    assert hits["AAPL"].index.name == "timestamp"
    assert str(hits["AAPL"].index.dtype) == "datetime64[ns, UTC]"


def test_a_miss_returns_nothing_for_that_symbol(cache_dir):
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY)
    hits, metas = bar_cache.read_many(["AAPL", "MSFT"], **KEY)
    assert set(hits) == {"AAPL"}
    assert set(metas) == {"AAPL"}


@pytest.mark.parametrize(
    "override",
    [
        {"feed": "iex"},
        {"source_timeframe": "60m"},
        {"start": "2026-05-05"},
        {"end": "2026-05-13"},
    ],
)
def test_every_key_dimension_is_load_bearing(cache_dir, override):
    """Changing any one of them must miss. The feed especially: curves priced
    off different tapes are not comparable, and omitting it is the exact
    defect market_data_store's own key has today."""
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY)
    hits, _ = bar_cache.read_many(["AAPL"], **{**KEY, **override})
    assert hits == {}


def test_symbols_are_keyed_individually_so_order_cannot_matter(cache_dir):
    """Sidesteps market_data_store._dataset_key's order-sensitive
    tuple(symbols): the same set in a different order is a hit here."""
    bar_cache.write_many(
        {"AAPL": _frame(), "MSFT": _frame(rows=4)}, last_fetch=LAST_FETCH, **KEY
    )
    hits, _ = bar_cache.read_many(["MSFT", "AAPL"], **KEY)
    assert set(hits) == {"AAPL", "MSFT"}


def test_a_schema_version_bump_invalidates_every_entry(cache_dir, monkeypatch):
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY)
    monkeypatch.setattr(bar_cache, "SCHEMA_VERSION", bar_cache.SCHEMA_VERSION + 1)
    hits, _ = bar_cache.read_many(["AAPL"], **KEY)
    assert hits == {}


def test_a_datetime_in_last_fetch_is_stored_as_an_iso_string(cache_dir):
    """`_effective_end` returns a datetime on the SIP path. The sidecar is
    JSON, so it comes back as a string -- and that is fine: the only readers
    are `baselines.py`, which prints it, and the two `source_timeframe`
    checks."""
    from datetime import datetime, timezone

    last_fetch = dict(
        LAST_FETCH, effective_end=datetime(2026, 5, 12, tzinfo=timezone.utc)
    )
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=last_fetch, **KEY)
    _, metas = bar_cache.read_many(["AAPL"], **KEY)
    assert metas["AAPL"]["effective_end"] == "2026-05-12T00:00:00+00:00"


def test_writes_are_atomic_and_leave_no_temp_files(cache_dir):
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY)
    assert sorted(p.suffix for p in cache_dir.iterdir()) == [".json", ".parquet"]


def test_a_disabled_cache_writes_and_reads_nothing(cache_dir, monkeypatch):
    monkeypatch.setenv("ATL_BAR_CACHE", "0")
    assert bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY) == 0
    assert bar_cache.read_many(["AAPL"], **KEY) == ({}, {})


# --- carried from Task 1 review ---------------------------------------------


def test_cache_dir_honours_the_env_override(tmp_path, monkeypatch):
    """Today the override branch is unasserted: replacing the whole function
    body with `return BAR_CACHE_DIR` keeps the suite green."""
    override = tmp_path / "custom_bar_cache_dir"
    monkeypatch.setenv("ATL_BAR_CACHE_DIR", str(override))
    assert bar_cache.cache_dir() == override


def test_a_valid_override_is_honoured_for_max_bytes_and_ttl(monkeypatch):
    """Today only the unset-default and the invalid/out-of-range paths are
    asserted, so a mutant that always returned the default would keep the
    suite green."""
    monkeypatch.setenv("ATL_BAR_CACHE_MAX_MB", "100")
    assert bar_cache.max_bytes() == 100 * 1024 * 1024
    monkeypatch.setenv("ATL_BAR_CACHE_TTL_DAYS", "3")
    assert bar_cache.ttl_seconds() == 3 * 86400.0
