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


def test_a_relative_override_is_anchored_at_the_repo_root(monkeypatch, tmp_path):
    """MUTATION TEST: `return Path(override)` and this fails. The parent
    uvicorn runs from the repo root and a backtest child is spawned with
    `cwd=DASHBOARD_DIR`, so an unanchored relative value gives the two
    processes different directories: a 100% miss rate for the one case the
    cache exists for, two directories each growing to the cap, and no signal
    -- both processes log the same unresolved string. `.resolve()` is not the
    fix either; it resolves per process and reproduces the split exactly."""
    from pathlib import Path

    from dashboard.backend.paths import DASHBOARD_DIR, REPO_ROOT

    monkeypatch.setenv("ATL_BAR_CACHE_DIR", "bar_cache")
    resolved = bar_cache.cache_dir()
    assert resolved.is_absolute()
    assert resolved == REPO_ROOT / "bar_cache"
    monkeypatch.chdir(DASHBOARD_DIR)
    assert bar_cache.cache_dir() == resolved
    monkeypatch.setenv("ATL_BAR_CACHE_DIR", "~/atl_bar_cache")
    assert bar_cache.cache_dir() == Path.home() / "atl_bar_cache"


def test_a_valid_override_is_honoured_for_max_bytes_and_ttl(monkeypatch):
    """Today only the unset-default and the invalid/out-of-range paths are
    asserted, so a mutant that always returned the default would keep the
    suite green."""
    monkeypatch.setenv("ATL_BAR_CACHE_MAX_MB", "100")
    assert bar_cache.max_bytes() == 100 * 1024 * 1024
    monkeypatch.setenv("ATL_BAR_CACHE_TTL_DAYS", "3")
    assert bar_cache.ttl_seconds() == 3 * 86400.0


def test_the_design_doc_does_not_call_this_unimplemented():
    """SOURCE-SHAPE GUARD. CLAUDE.md's ATL_BAR_CACHE bullet ends by pointing
    at this spec, so its status banner is the first thing every future agent
    sent there reads -- and it said "DESIGN ONLY -- nothing under dashboard/
    implements this" on the branch that implements it, inviting a reader to
    re-implement the cache or to treat the module as dead. A document is only
    as trustworthy as the thing that notices it went stale."""
    from dashboard.backend.paths import REPO_ROOT

    spec = REPO_ROOT / "docs" / "superpowers" / "specs"
    spec = spec / "2026-09-21-backtest-bar-cache-design.md"
    banner = spec.read_text(encoding="utf-8").split("\n\n", 2)[1]
    assert "Status:" in banner, "the spec's status banner moved"
    assert "DESIGN ONLY" not in banner, banner
    assert "no `bar_cache` module exists" not in banner, banner


# --- the refusal rules (design section 5) ----------------------------------


def test_a_clamped_window_is_never_stored(cache_dir, capsys):
    """MUTATION TEST: delete the `if end_clamped:` guard in write_many and
    this must fail. A clamped SIP window returns a SHORTER frame for the same
    requested key depending on the wall clock; caching it pins the truncation
    for the life of the instance."""
    written = bar_cache.write_many(
        {"AAPL": _frame()},
        last_fetch=dict(LAST_FETCH, end_clamped=True),
        end_clamped=True,
        **KEY,
    )
    assert written == 0
    assert not cache_dir.exists() or list(cache_dir.iterdir()) == []
    assert "clamped" in capsys.readouterr().out


def test_an_iex_fallback_window_is_never_stored(cache_dir, capsys):
    """MUTATION TEST: delete the `if sip_fallback_to_iex:` guard and this must
    fail. The fallback retry re-requests with the original unclamped end and
    never sets end_clamped, so the frame looks pristine while carrying ~2.5%
    of the volume the key claims."""
    written = bar_cache.write_many(
        {"AAPL": _frame()},
        last_fetch=dict(LAST_FETCH, sip_fallback_to_iex=True),
        sip_fallback_to_iex=True,
        **KEY,
    )
    assert written == 0
    assert not cache_dir.exists() or list(cache_dir.iterdir()) == []
    assert "IEX-fallback" in capsys.readouterr().out


def test_a_window_that_has_not_settled_is_never_stored(cache_dir, capsys):
    """MUTATION TEST: delete the `if not window_is_settled(end):` guard and
    this must fail. The clamp and fallback flags only cover SIP-on-Basic;
    under iex, ALPACA_ALLOW_RECENT_SIP=1 or a zero delay a window whose end is
    still ahead of the clock comes back partial with end_clamped=False."""
    from datetime import date, timedelta

    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    written = bar_cache.write_many(
        {"AAPL": _frame()}, last_fetch=LAST_FETCH, **{**KEY, "end": tomorrow}
    )
    assert written == 0
    assert not cache_dir.exists() or list(cache_dir.iterdir()) == []
    assert "not settled" in capsys.readouterr().out


@pytest.mark.parametrize(
    "end, now, expected",
    [
        # `now` is 2026-05-14T00:00:00Z throughout. A date-only end of D covers
        # bars opening before D 00:00 UTC, so D=05-13 settles at 05-14 00:00.
        ("2026-05-13", 1778716800.0, True),
        ("2026-05-13", 1778716799.0, False),
        ("2026-05-14", 1778716800.0, False),
        ("2026-05-13T00:00:00Z", 1778716800.0, True),
        ("2026-05-13T00:00:00+00:00", 1778716800.0, True),
        ("2026-05-12T20:00:00-04:00", 1778716800.0, True),
        ("not-a-date", 1778716800.0, False),
        (None, 1778716800.0, False),
    ],
)
def test_window_is_settled_needs_a_full_day_past_the_exclusive_end(end, now, expected):
    assert bar_cache.window_is_settled(end, now=now) is expected


def test_an_absent_symbol_is_not_cached_as_an_empty_frame(cache_dir):
    """A missing symbol is not a negative fact worth persisting."""
    empty = _frame().iloc[0:0]
    assert bar_cache.write_many({"AAPL": empty}, last_fetch=LAST_FETCH, **KEY) == 0
    assert bar_cache.read_many(["AAPL"], **KEY) == ({}, {})


def test_a_missing_last_fetch_stores_nothing(cache_dir):
    assert bar_cache.write_many({"AAPL": _frame()}, last_fetch=None, **KEY) == 0


# --- expiry, corruption, half-entries --------------------------------------


def test_an_entry_past_its_ttl_is_a_miss_and_is_removed(cache_dir):
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY)
    parquet_path, meta_path = bar_cache.entry_paths("AAPL", **KEY)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["fetched_at"] = time.time() - (8 * 86400)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    assert bar_cache.read_many(["AAPL"], **KEY) == ({}, {})
    assert not parquet_path.exists() and not meta_path.exists()


def test_a_read_touch_cannot_keep_a_stale_entry_alive(cache_dir):
    """mtime is the LRU clock; the TTL reads `fetched_at`. If the TTL were
    computed from mtime, touching on read would make it unreachable."""
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY)
    parquet_path, meta_path = bar_cache.entry_paths("AAPL", **KEY)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["fetched_at"] = time.time() - (8 * 86400)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    os.utime(parquet_path, None)  # fresh mtime, stale fetch
    assert bar_cache.read_many(["AAPL"], **KEY) == ({}, {})


def test_a_corrupt_parquet_is_a_miss_and_is_removed(cache_dir):
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY)
    parquet_path, meta_path = bar_cache.entry_paths("AAPL", **KEY)
    parquet_path.write_bytes(b"not a parquet file")
    assert bar_cache.read_many(["AAPL"], **KEY) == ({}, {})
    assert not parquet_path.exists() and not meta_path.exists()


def test_a_corrupt_sidecar_is_a_miss_and_is_removed(cache_dir):
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY)
    parquet_path, meta_path = bar_cache.entry_paths("AAPL", **KEY)
    meta_path.write_text("{not json", encoding="utf-8")
    assert bar_cache.read_many(["AAPL"], **KEY) == ({}, {})
    assert not parquet_path.exists() and not meta_path.exists()


def test_a_fresh_half_entry_is_a_miss_but_is_left_for_its_writer(cache_dir):
    """MUTATION TEST: replace `_discard_if_stale` in read_many with `_discard`
    and this must fail. A sidecar without its parquet is what a concurrent
    writer looks like between its two os.replace calls; a reader that unlinks
    it destroys that write, and under contention the key stays cold while
    every child pays the fetch."""
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY)
    parquet_path, meta_path = bar_cache.entry_paths("AAPL", **KEY)
    parquet_path.unlink()
    assert bar_cache.read_many(["AAPL"], **KEY) == ({}, {})
    assert meta_path.exists()


def test_a_stale_half_entry_is_a_miss_and_is_removed(cache_dir):
    """Older than the grace, the survivor belongs to a writer that died."""
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY)
    parquet_path, meta_path = bar_cache.entry_paths("AAPL", **KEY)
    meta_path.unlink()
    stamp = time.time() - 2 * bar_cache._STRAY_GRACE_SECONDS
    os.utime(parquet_path, (stamp, stamp))
    assert bar_cache.read_many(["AAPL"], **KEY) == ({}, {})
    assert not parquet_path.exists()


def test_a_sidecar_whose_key_fields_disagree_is_a_miss(cache_dir):
    """Turns a truncated-digest collision into a miss rather than wrong bars."""
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY)
    _, meta_path = bar_cache.entry_paths("AAPL", **KEY)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["feed"] = "iex"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    assert bar_cache.read_many(["AAPL"], **KEY) == ({}, {})


def test_two_writers_racing_one_key_leave_a_readable_entry(cache_dir):
    """os.replace is atomic on POSIX, so a reader sees the old entry or the
    complete new one -- never a partial parquet. Up to
    MAX_ACTIVE_DASHBOARD_BACKTESTS children race the same key on one instance."""
    import threading

    barrier = threading.Barrier(2)

    def writer(rows):
        barrier.wait()
        for _ in range(20):
            bar_cache.write_many(
                {"AAPL": _frame(rows=rows)}, last_fetch=LAST_FETCH, **KEY
            )

    threads = [threading.Thread(target=writer, args=(n,)) for n in (3, 5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    hits, _ = bar_cache.read_many(["AAPL"], **KEY)
    assert len(hits["AAPL"]) in (3, 5)


# --- eviction --------------------------------------------------------------


def _tiny_cap(monkeypatch, cache_dir, keep_entries=2):
    """Shrink the cap to `keep_entries` times the size of one entry on disk.

    Monkeypatching `max_bytes` rather than setting ATL_BAR_CACHE_MAX_MB, whose
    floor is 1MB: a parquet of a few hundred OHLCV rows is tens of kilobytes,
    so four of them never reach 1MB and an env-var version of this test would
    pass while evicting nothing. The env-var path is covered separately by
    test_defaults_match_the_spec and test_out_of_range_size_cap_falls_back.
    Sized from the LARGEST entry present, not the mean: every entry holds the
    same 200 rows, but the sidecars differ by a byte or two (the repr of
    `fetched_at` varies in length), and a mean-based cap can sit one byte
    under the two survivors it means to keep.
    """
    per_entry = max(
        path.stat().st_size + path.with_suffix(".json").stat().st_size
        for path in cache_dir.glob("*.parquet")
    )
    cap = per_entry * keep_entries
    monkeypatch.setattr(bar_cache, "max_bytes", lambda: cap)
    return cap


def _stamp(cache_dir, symbol, offset_seconds):
    """Give one entry a definite mtime, `offset_seconds` from now."""
    stamp = time.time() + offset_seconds
    for path in bar_cache.entry_paths(symbol, **KEY):
        os.utime(path, (stamp, stamp))


def test_the_size_cap_evicts_the_least_recently_used_entries(cache_dir, monkeypatch):
    """Write everything FIRST, then order the mtimes, then shrink the cap.

    Filesystem mtimes are coarse (jiffy granularity on Linux), so entries
    written within one tick tie and "least recently used" is undefined among
    them. Setting the stamps after all the writes, and only then running the
    pass, is what makes the expected survivors deterministic.
    """
    for index in range(4):
        bar_cache.write_many(
            {f"S{index}": _frame(rows=200)}, last_fetch=LAST_FETCH, **KEY
        )
    for index in range(4):
        _stamp(cache_dir, f"S{index}", -(10 - index))  # S0 oldest, S3 newest
    cap = _tiny_cap(monkeypatch, cache_dir, keep_entries=2)
    assert bar_cache.enforce_size_cap() == 2
    total = sum(path.stat().st_size for path in cache_dir.iterdir())
    assert total <= cap
    hits, _ = bar_cache.read_many(["S0", "S1", "S2", "S3"], **KEY)
    assert set(hits) == {"S2", "S3"}


def test_a_read_protects_an_entry_from_eviction(cache_dir, monkeypatch):
    """MUTATION TEST: drop the os.utime pair in read_many and this fails.

    mtime is the LRU clock and enforce_size_cap has no other hotness signal,
    so without the touch eviction degrades to FIFO by write time -- and the
    three warm default windows, read on every backtest and never rewritten,
    become the FIRST victims under cap pressure. That inverts the policy this
    module documents. The inverse case (a touch must not keep a stale entry
    alive) is asserted separately and passes either way."""
    bar_cache.write_many({"HOT": _frame(rows=200)}, last_fetch=LAST_FETCH, **KEY)
    time.sleep(0.01)  # HOT is written first, so only the read can make it newest
    bar_cache.write_many({"COLD": _frame(rows=200)}, last_fetch=LAST_FETCH, **KEY)
    bar_cache.read_many(["HOT"], **KEY)
    _tiny_cap(monkeypatch, cache_dir, keep_entries=1)
    assert bar_cache.enforce_size_cap() == 1
    hits, _ = bar_cache.read_many(["HOT", "COLD"], **KEY)
    assert set(hits) == {"HOT"}


def test_a_write_never_evicts_its_own_batch(cache_dir, monkeypatch):
    """MUTATION TEST: drop the `protect=` argument from write_many's eviction
    call and this must fail. Under cap pressure the trailing pass would
    otherwise discard the symbols just fetched -- a paid fetch that never
    becomes a hit -- and with tying mtimes it did exactly that."""
    bar_cache.write_many({"OLD": _frame(rows=200)}, last_fetch=LAST_FETCH, **KEY)
    # Make the resident entry look NEWER than anything written next, so pure
    # LRU order would pick the fresh batch as the victim.
    _stamp(cache_dir, "OLD", +100)
    _tiny_cap(monkeypatch, cache_dir, keep_entries=1)
    bar_cache.write_many({"NEW": _frame(rows=200)}, last_fetch=LAST_FETCH, **KEY)
    hits, _ = bar_cache.read_many(["OLD", "NEW"], **KEY)
    assert set(hits) == {"NEW"}


def test_eviction_leaves_no_half_entries(cache_dir, monkeypatch):
    for index in range(6):
        bar_cache.write_many(
            {f"S{index}": _frame(rows=200)}, last_fetch=LAST_FETCH, **KEY
        )
    _tiny_cap(monkeypatch, cache_dir, keep_entries=2)
    bar_cache.enforce_size_cap()
    parquets = list(cache_dir.glob("*.parquet"))
    assert parquets, "eviction removed everything"
    for path in parquets:
        assert path.with_suffix(".json").exists()
    for path in cache_dir.glob("*.json"):
        assert path.with_suffix(".parquet").exists()


def test_stale_temp_files_are_swept(cache_dir):
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY)
    orphan = cache_dir / "AAPL-deadbeefdeadbeef.parquet7xk.tmp"
    orphan.write_bytes(b"crashed writer")
    stamp = time.time() - 2 * bar_cache._STRAY_GRACE_SECONDS
    os.utime(orphan, (stamp, stamp))
    bar_cache.enforce_size_cap()
    assert not orphan.exists()


def test_a_stale_orphan_sidecar_is_swept(cache_dir):
    """A crashed writer's sidecar matches no *.parquet glob, so the LRU pass
    would never reclaim it; the stray sweep does, once it is past the grace."""
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY)
    parquet_path, meta_path = bar_cache.entry_paths("AAPL", **KEY)
    parquet_path.unlink()
    stamp = time.time() - 2 * bar_cache._STRAY_GRACE_SECONDS
    os.utime(meta_path, (stamp, stamp))
    bar_cache.enforce_size_cap()
    assert not meta_path.exists()


def test_a_fresh_orphan_sidecar_survives_the_sweep(cache_dir):
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY)
    parquet_path, meta_path = bar_cache.entry_paths("AAPL", **KEY)
    parquet_path.unlink()
    bar_cache.enforce_size_cap()
    assert meta_path.exists()


def test_a_fresh_temp_file_is_left_alone(cache_dir):
    """A concurrent writer's in-flight temp file must survive another
    process's eviction pass."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    in_flight = cache_dir / "AAPL-deadbeefdeadbeef.parquetab1.tmp"
    in_flight.write_bytes(b"in flight")
    bar_cache.enforce_size_cap()
    assert in_flight.exists()


def test_eviction_never_touches_a_file_this_module_did_not_write(cache_dir):
    """MUTATION TEST: drop the `_is_owned` filter in enforce_size_cap and this
    fails. The sweep classifies by SUFFIX and then unlinks, and `cache_dir`'s
    docstring invites an operator to point ATL_BAR_CACHE_DIR wherever they
    like. Point it at `dashboard/storage/data` -- the parent of the default,
    and populated today -- and the first backtest deletes live application
    state as soon as it is an hour old."""
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY)
    stale = time.time() - 2 * bar_cache._STRAY_GRACE_SECONDS
    foreign = []
    for name in (
        "algo_submissions.json",  # user-submitted trading-algo state
        "leaderboard_daily_refresh.json",
        "leaderboard_skip_cache.json",
        "half-written.tmp",
        "notes.parquet",
    ):
        path = cache_dir / name
        path.write_bytes(b"not ours")
        os.utime(path, (stale, stale))
        foreign.append(path)
    bar_cache.enforce_size_cap()
    assert [path for path in foreign if not path.exists()] == []
    # And the entry that IS ours still reads back.
    hits, _ = bar_cache.read_many(["AAPL"], **KEY)
    assert set(hits) == {"AAPL"}


def test_a_foreign_file_is_neither_counted_nor_evicted_under_cap_pressure(
    cache_dir, monkeypatch
):
    """The cap measures this cache's own footprint. Counting a neighbour's
    bytes would evict our own entries to make room for a file we must not
    touch -- and then unlink the neighbour too, once ours ran out."""
    for index in range(3):
        bar_cache.write_many(
            {f"S{index}": _frame(rows=200)}, last_fetch=LAST_FETCH, **KEY
        )
    cap = _tiny_cap(monkeypatch, cache_dir, keep_entries=3)
    foreign = cache_dir / "algo_submissions.parquet"  # sized past the cap
    foreign.write_bytes(b"x" * (cap * 2))
    assert bar_cache.enforce_size_cap() == 0
    assert foreign.exists()
    hits, _ = bar_cache.read_many(["S0", "S1", "S2"], **KEY)
    assert set(hits) == {"S0", "S1", "S2"}


def test_an_absent_directory_says_so_rather_than_returning_silently(
    cache_dir, capsys
):
    """A silent 0 and a clean sweep are the same empty stdout."""
    assert not cache_dir.exists()
    assert bar_cache.enforce_size_cap() == 0
    assert str(cache_dir) in capsys.readouterr().out


def test_a_scan_failure_is_logged_rather_than_silent(cache_dir, monkeypatch, capsys):
    """Every other failure in the module logs. A directory this process
    cannot read means the cap is not being enforced at all."""
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY)

    def _refuse(_directory):
        raise OSError("permission denied")

    monkeypatch.setattr(bar_cache.os, "scandir", _refuse)
    assert bar_cache.enforce_size_cap() == 0
    assert "cannot scan" in capsys.readouterr().out


def test_write_many_enforces_the_cap(cache_dir, monkeypatch):
    """Eviction is not a separate chore someone has to remember to run."""
    bar_cache.write_many({"S0": _frame(rows=200)}, last_fetch=LAST_FETCH, **KEY)
    cap = _tiny_cap(monkeypatch, cache_dir, keep_entries=2)
    for index in range(1, 6):
        bar_cache.write_many(
            {f"S{index}": _frame(rows=200)}, last_fetch=LAST_FETCH, **KEY
        )
    total = sum(path.stat().st_size for path in cache_dir.iterdir())
    assert total <= cap


def test_write_many_skips_the_scan_when_it_cannot_have_crossed_the_cap(
    cache_dir, monkeypatch
):
    """The full directory scan is O(entries) and the cap holds thousands of
    them; paying it on every fetch far under cap is the wrong trade. The first
    write scans (nothing is known yet); the second, with the running estimate
    well under cap and the scan fresh, must not."""
    scans = []
    real = bar_cache.enforce_size_cap
    monkeypatch.setattr(
        bar_cache, "enforce_size_cap", lambda **kw: scans.append(1) or real(**kw)
    )
    bar_cache.write_many({"S0": _frame(rows=200)}, last_fetch=LAST_FETCH, **KEY)
    assert len(scans) == 1
    bar_cache.write_many({"S1": _frame(rows=200)}, last_fetch=LAST_FETCH, **KEY)
    assert len(scans) == 1


def test_write_many_rescans_once_the_interval_has_elapsed(cache_dir, monkeypatch):
    """Other processes' writes are invisible to the estimate, so the scan
    also runs on a clock, not only on this process's bytes."""
    scans = []
    real = bar_cache.enforce_size_cap
    monkeypatch.setattr(
        bar_cache, "enforce_size_cap", lambda **kw: scans.append(1) or real(**kw)
    )
    bar_cache.write_many({"S0": _frame(rows=200)}, last_fetch=LAST_FETCH, **KEY)
    state = bar_cache._scan_state[str(cache_dir)]
    state[0] -= 2 * bar_cache._SWEEP_INTERVAL_SECONDS
    bar_cache.write_many({"S1": _frame(rows=200)}, last_fetch=LAST_FETCH, **KEY)
    assert len(scans) == 2
