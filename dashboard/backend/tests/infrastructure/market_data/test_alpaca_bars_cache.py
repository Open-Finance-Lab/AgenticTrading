"""The on-disk bar cache, driven through the real AlpacaDataLoader.fetch_bars.

No network: the Alpaca SDK client is a fake, exactly as in test_alpaca_bars.py.
Every test points ATL_BAR_CACHE_DIR at tmp_path.
"""

import pandas as pd
import pytest

from dashboard.backend.infrastructure.market_data import bar_cache
from dashboard.backend.infrastructure.market_data.alpaca_bars import (
    FRAME_ATTR_END_CLAMPED,
    FRAME_ATTR_FEED,
    FRAME_ATTR_SIP_FALLBACK,
    AlpacaDataLoader,
)

CLIENT_TARGET = "alpaca.data.historical.StockHistoricalDataClient"
CLAMP_TARGET = (
    "dashboard.backend.infrastructure.market_data.alpaca_bars.clamp_end_for_sip"
)


def _bars_df(symbols, rows=2):
    frames = []
    for symbol in symbols:
        index = pd.MultiIndex.from_tuples(
            [
                (symbol, pd.Timestamp("2026-05-04T13:30:00Z") + pd.Timedelta(minutes=5 * i))
                for i in range(rows)
            ],
            names=["symbol", "timestamp"],
        )
        frames.append(
            pd.DataFrame(
                [
                    {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 100}
                    for _ in range(rows)
                ],
                index=index,
            )
        )
    if not frames:
        # pd.concat([]) raises "No objects to concatenate", and the fixture's
        # initial state is exactly this.
        return pd.DataFrame(
            {"open": [], "high": [], "low": [], "close": [], "volume": []},
            index=pd.MultiIndex.from_arrays([[], []], names=["symbol", "timestamp"]),
        )
    return pd.concat(frames)


@pytest.fixture
def fake_alpaca(monkeypatch):
    state = {"df": _bars_df([]), "exc": None, "requests": []}

    class _FakeBars:
        def __init__(self, df):
            self.df = df

    class _FakeSession:
        def request(self, *args, **kwargs):
            raise NotImplementedError("not exercised by these tests")

    class _FakeClient:
        def __init__(self, api_key, secret_key):
            self._session = _FakeSession()

        def get_stock_bars(self, request):
            state["requests"].append(request)
            if state["exc"] is not None:
                raise state["exc"]
            # Answer only for what was ASKED, the way Alpaca does. A fake that
            # returns its whole frame regardless cannot tell "this symbol has
            # no bars" from "the response was empty" -- the two cases the
            # wholly-empty guard in `fetch_bars` exists to separate, since only
            # the second one means the cached symbols are untrustworthy too.
            df = state["df"]
            asked = request.symbol_or_symbols
            asked = [asked] if isinstance(asked, str) else list(asked)
            if len(df.index.names) == 2 and not df.empty:
                df = df[df.index.get_level_values("symbol").isin(asked)]
            return _FakeBars(df)

    monkeypatch.setattr(CLIENT_TARGET, _FakeClient)
    return state


@pytest.fixture
def cached_loader(fake_alpaca, tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    monkeypatch.setenv("ALPACA_DATA_FEED", "sip")
    monkeypatch.setenv("ATL_BAR_CACHE", "1")
    monkeypatch.setenv("ATL_BAR_CACHE_DIR", str(tmp_path / "bar_cache"))
    monkeypatch.delenv("ATL_BAR_CACHE_MAX_MB", raising=False)
    monkeypatch.delenv("ATL_BAR_CACHE_TTL_DAYS", raising=False)
    loader = AlpacaDataLoader()
    loader.configure_source_timeframe("5m")
    return loader


def _requested(state):
    return [list(request.symbol_or_symbols) for request in state["requests"]]


def test_a_second_identical_request_makes_no_alpaca_call(cached_loader, fake_alpaca):
    fake_alpaca["df"] = _bars_df(["AAPL", "MSFT"])
    first = cached_loader.fetch_bars(["AAPL", "MSFT"], "2026-05-04", "2026-05-12")
    assert len(fake_alpaca["requests"]) == 1
    second = cached_loader.fetch_bars(["AAPL", "MSFT"], "2026-05-04", "2026-05-12")
    assert len(fake_alpaca["requests"]) == 1  # served entirely from disk
    assert set(second) == set(first) == {"AAPL", "MSFT"}
    pd.testing.assert_frame_equal(second["AAPL"], first["AAPL"])


def test_a_mixed_request_fetches_only_the_missing_symbols(cached_loader, fake_alpaca):
    """This is what makes the DJIA_30 index baseline cheap after a Mag7 run:
    five of its thirty names are already on disk under the same window."""
    fake_alpaca["df"] = _bars_df(["AAPL"])
    cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    fake_alpaca["df"] = _bars_df(["MSFT", "NVDA"])
    result = cached_loader.fetch_bars(
        ["AAPL", "MSFT", "NVDA"], "2026-05-04", "2026-05-12"
    )
    assert _requested(fake_alpaca) == [["AAPL"], ["MSFT", "NVDA"]]
    assert set(result) == {"AAPL", "MSFT", "NVDA"}


def test_a_cache_hit_restores_last_fetch(cached_loader, fake_alpaca):
    """market_data_store._build_dataset and engine.load_data read last_fetch to
    verify the source timeframe with evidence="fetch". A hit that leaves it
    stale silently downgrades that to the weaker evidence="configured" path."""
    fake_alpaca["df"] = _bars_df(["AAPL"])
    cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    cached_loader.last_fetch = {"source_timeframe": "60m", "feed": "iex"}
    cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    assert cached_loader.last_fetch["source_timeframe"] == "5m"
    assert cached_loader.last_fetch["feed"] == "sip"
    assert cached_loader.last_fetch["sip_fallback_to_iex"] is False
    assert cached_loader.last_fetch["end_clamped"] is False


#: Every field ``AlpacaDataLoader._record_fetch`` puts in ``last_fetch``
#: (``alpaca_bars.py:414``), with the reason serving it from a sidecar is
#: sound. ``write_many`` stores this dict (``bar_cache.py``) and a TOTAL hit
#: restores it (``fetch_bars``'s ``if not misses:`` branch,
#: ``alpaca_bars.py:575``; a MIXED hit re-records it live from the fetch of
#: the misses instead), so a value written once is served to a DIFFERENT
#: PROCESS for the whole TTL. That is safe only while every field is:
#:
#:   KEYED   -- a component of the cache key ``{start, end, source_timeframe,
#:              feed}``, so the stored value is a pure function of the key and
#:              cannot describe a different request than the one being served.
#:   REFUSED -- a flag ``write_many`` refuses to store on, so an entry holding
#:              anything other than the benign value cannot exist on disk.
#:
#: A seventh field that is neither -- a fetch timestamp, a retry count, the
#: per-chunk symbol list -- would be served stale, silently, to the three call
#: sites that read ``last_fetch`` as fetch evidence: ``engine.py:load_data``,
#: ``market_data_store._build_dataset`` (both ``verify_source_timeframe(...,
#: evidence="fetch")``) and ``leaderboard/baselines.py`` (the IEX/clamp
#: warning). Add a field here only with its classification.
_LAST_FETCH_FIELDS = {
    # The key's `feed` is `configured_feed_name()` -- the tape REQUESTED. This
    # is the tape ANSWERED, and they diverge exactly when the IEX fallback
    # fired, which `write_many` refuses to store. So for any entry that can be
    # on disk, keyed.
    "feed": "KEYED",
    "source_timeframe": "KEYED",
    "requested_end": "KEYED",  # literally the key's `end`
    # Differs from `requested_end` only when the SIP clamp moved it, and a
    # clamped window is refused.
    "effective_end": "REFUSED",
    "sip_fallback_to_iex": "REFUSED",
    "end_clamped": "REFUSED",
}


def test_last_fetch_carries_only_keyed_or_refused_fields(cached_loader, fake_alpaca):
    """GUARD for the invariant that makes a cached `last_fetch` servable.

    Asserted against a real fetch rather than the source text, because what
    matters is the dict that reaches `write_many`, not the literal in
    `_record_fetch`.
    """
    fake_alpaca["df"] = _bars_df(["AAPL"])
    cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    assert set(cached_loader.last_fetch) == set(_LAST_FETCH_FIELDS), (
        "last_fetch gained or lost a field. Every field is served cross-process "
        "from a sidecar for the whole TTL, so a new one must be either a cache-key "
        "component or a flag write_many refuses to store -- classify it in "
        "_LAST_FETCH_FIELDS and say which."
    )


def test_a_cache_hit_reproduces_last_fetch_exactly(
    cached_loader, fake_alpaca, tmp_path, monkeypatch
):
    """The behavioural half of the guard above: a served copy must be
    indistinguishable from a live one for the same KEY.

    The symbol list is deliberately not in that key -- which is what makes the
    index baseline cheap -- so the entry read here is written by a DIFFERENT
    request than the one it serves: a two-symbol fetch warms it, a one-symbol
    fetch reads it, and the comparison is against a live loader on a cold
    cache.

    Two identical calls could not catch a request-dependent field at all.
    `write_many` stores `last_fetch` verbatim and a hit returns it verbatim,
    so the restored dict IS the stored dict: comparing it against the call
    that stored it compares a value with itself, and a `symbol_count`, a
    `fetched_at` or a per-chunk symbol list all match. What that shape does
    pin is the JSON round-trip through the sidecar (a `datetime` field comes
    back a `str`), which this comparison still covers -- the served half
    round-trips either way.
    """
    fake_alpaca["df"] = _bars_df(["AAPL", "MSFT"])
    cached_loader.fetch_bars(["AAPL", "MSFT"], "2026-05-04", "2026-05-12")
    cached_loader.last_fetch = None
    served = cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    assert len(fake_alpaca["requests"]) == 1  # a TOTAL hit: the restore branch
    assert set(served) == {"AAPL"}
    restored = dict(cached_loader.last_fetch)

    # The same one-symbol request, made live. A second cache dir rather than a
    # second window: moving the window would move `requested_end` too, and the
    # two dicts would then differ for a reason that is not the invariant.
    monkeypatch.setenv("ATL_BAR_CACHE_DIR", str(tmp_path / "cold_bar_cache"))
    live_loader = AlpacaDataLoader()
    live_loader.configure_source_timeframe("5m")
    live_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    assert len(fake_alpaca["requests"]) == 2

    assert restored == live_loader.last_fetch, (
        "a served last_fetch describes the request that WROTE the entry, not "
        "the one being served. Every field must be a cache-key component or a "
        "flag write_many refuses to store -- see _LAST_FETCH_FIELDS."
    )


def test_a_cache_hit_restores_the_attrs_stamps(cached_loader, fake_alpaca):
    fake_alpaca["df"] = _bars_df(["AAPL"])
    cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    hit = cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    assert hit["AAPL"].attrs[FRAME_ATTR_FEED] == "sip"
    assert hit["AAPL"].attrs[FRAME_ATTR_SIP_FALLBACK] is False
    assert hit["AAPL"].attrs[FRAME_ATTR_END_CLAMPED] is False


def test_changing_the_feed_misses(cached_loader, fake_alpaca, monkeypatch):
    fake_alpaca["df"] = _bars_df(["AAPL"])
    cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    monkeypatch.setenv("ALPACA_DATA_FEED", "iex")
    cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    assert len(fake_alpaca["requests"]) == 2


def test_changing_the_source_timeframe_misses(cached_loader, fake_alpaca):
    """source_timeframe is a mutable instance attribute set by
    configure_source_timeframe, not an argument -- it must be read at call
    time, and it materially changes the bars for an identical window."""
    fake_alpaca["df"] = _bars_df(["AAPL"])
    cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    cached_loader.configure_source_timeframe("60m")
    cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    assert len(fake_alpaca["requests"]) == 2


def test_a_clamped_response_is_never_written(cached_loader, fake_alpaca, monkeypatch):
    """MUTATION TEST, through the real path: remove the end_clamped guard in
    bar_cache.write_many and this must fail."""
    import datetime as _dt

    monkeypatch.setattr(
        CLAMP_TARGET,
        lambda end, **kwargs: _dt.datetime(2026, 5, 11, tzinfo=_dt.timezone.utc),
    )
    fake_alpaca["df"] = _bars_df(["AAPL"])
    result = cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    assert result["AAPL"].attrs[FRAME_ATTR_END_CLAMPED] is True
    hits, _ = bar_cache.read_many(
        ["AAPL"],
        start="2026-05-04",
        end="2026-05-12",
        source_timeframe="5m",
        feed="sip",
    )
    assert hits == {}


def test_an_iex_fallback_response_is_never_written(cached_loader, fake_alpaca):
    """MUTATION TEST, through the real path: remove the sip_fallback_to_iex
    guard in bar_cache.write_many and this must fail."""
    calls = {"n": 0}
    real_df = _bars_df(["AAPL"])

    class _FallbackClient:
        def __init__(self, *args, **kwargs):
            self._session = None

        def get_stock_bars(self, request):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("your subscription does not permit this")

            class _Bars:
                df = real_df

            return _Bars()

    cached_loader.client = _FallbackClient()
    result = cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    assert result["AAPL"].attrs[FRAME_ATTR_SIP_FALLBACK] is True
    hits, _ = bar_cache.read_many(
        ["AAPL"],
        start="2026-05-04",
        end="2026-05-12",
        source_timeframe="5m",
        feed="sip",
    )
    assert hits == {}


def test_a_window_ending_in_the_future_is_never_written_on_iex(
    cached_loader, fake_alpaca, monkeypatch
):
    """MUTATION TEST, through the real path: remove the window_is_settled
    guard in bar_cache.write_many and this must fail. `_effective_end` returns
    (end, False) for IEX -- no clamp, no flag -- so with the window still
    open nothing else stops a half-day frame being stored as complete."""
    from datetime import date, timedelta

    monkeypatch.setenv("ALPACA_DATA_FEED", "iex")
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    fake_alpaca["df"] = _bars_df(["AAPL"])
    result = cached_loader.fetch_bars(["AAPL"], "2026-05-04", tomorrow)
    assert result["AAPL"].attrs[FRAME_ATTR_END_CLAMPED] is False
    hits, _ = bar_cache.read_many(
        ["AAPL"],
        start="2026-05-04",
        end=tomorrow,
        source_timeframe="5m",
        feed="iex",
    )
    assert hits == {}


def test_a_failed_fetch_for_the_misses_fails_the_whole_request(
    cached_loader, fake_alpaca
):
    """Before the cache a request returned what Alpaca had or {}, and
    engine.load_data raises on {}. Five Dow names on disk plus Alpaca down for
    the other twenty-five must still be {} -- not a 5-symbol "Dow"."""
    fake_alpaca["df"] = _bars_df(["AAPL"])
    cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    fake_alpaca["exc"] = RuntimeError("alpaca is down")
    assert cached_loader.fetch_bars(["AAPL", "MSFT"], "2026-05-04", "2026-05-12") == {}
    assert cached_loader.last_fetch is None


def test_a_symbol_alpaca_had_no_bars_for_still_returns_the_hits(
    cached_loader, fake_alpaca
):
    """A genuinely dataless symbol must not cost its neighbours their bars:
    the old code asked for both and answered with what Alpaca had."""
    fake_alpaca["df"] = _bars_df(["AAPL"])
    cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    # AAPL still has bars; NOPE never did. The miss-fetch for NOPE alone comes
    # back empty, so `fetch_bars` re-asks for both -- and this time the answer
    # covers AAPL, which is what makes the partial universe legitimate.
    result = cached_loader.fetch_bars(["AAPL", "NOPE"], "2026-05-04", "2026-05-12")
    assert set(result) == {"AAPL"}
    assert cached_loader.last_fetch is not None


def test_an_empty_answer_for_the_misses_cannot_publish_a_partial_universe(
    cached_loader, fake_alpaca
):
    """REGRESSION. The guard used to be `not fetched and last_fetch is None`,
    which fires only on a HARD failure -- every failure exit clears that
    field, but a 200 answering with no rows leaves it set. So a transient
    empty answer returned the cached subset alone: five Dow names from an
    earlier Mag7 run, twenty-five answered with nothing, and `load_data`
    never raises because the dict is not empty.

    Distinguishing that from the dataless-symbol case above is impossible
    from the miss-fetch alone, so the re-request decides: here it comes back
    empty for AAPL too, and the answer is {}."""
    fake_alpaca["df"] = _bars_df(["AAPL"])
    cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    # The tape goes quiet for everything -- the shape a transient outage takes
    # when it answers 200 instead of raising.
    fake_alpaca["df"] = _bars_df([])
    result = cached_loader.fetch_bars(["AAPL", "MSFT"], "2026-05-04", "2026-05-12")
    assert result == {}, "a cached subset was published as the whole universe"


def test_a_wholly_empty_miss_fetch_is_not_re_requested_when_nothing_was_cached(
    cached_loader, fake_alpaca
):
    """The re-request exists to re-cover the CACHED symbols. With no hits the
    call just made already was the pre-cache call, so re-issuing it would only
    bill the same request twice."""
    fake_alpaca["df"] = _bars_df([])
    assert cached_loader.fetch_bars(["AAPL", "MSFT"], "2026-05-04", "2026-05-12") == {}
    assert len(fake_alpaca["requests"]) == 1


def test_an_iex_fallback_in_an_earlier_chunk_refuses_the_whole_batch(
    cached_loader, fake_alpaca
):
    """MUTATION TEST: derive the write_many flags from `self.last_fetch`
    instead of the frames and this must fail. `last_fetch` describes only the
    LAST 100-symbol chunk; here chunk one falls back to IEX and chunk two
    succeeds on SIP, so it reads sip_fallback_to_iex=False while 100 of the
    150 frames are IEX."""
    symbols = [f"S{i:03}" for i in range(150)]
    calls = {"n": 0}

    class _Client:
        def __init__(self, *args, **kwargs):
            self._session = None

        def get_stock_bars(self, request):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("your subscription does not permit this")

            class _Bars:
                df = _bars_df(list(request.symbol_or_symbols), rows=1)

            return _Bars()

    cached_loader.client = _Client()
    result = cached_loader.fetch_bars(symbols, "2026-05-04", "2026-05-12")
    assert len(result) == 150
    assert result["S000"].attrs[FRAME_ATTR_SIP_FALLBACK] is True
    assert result["S149"].attrs[FRAME_ATTR_SIP_FALLBACK] is False
    assert cached_loader.last_fetch["sip_fallback_to_iex"] is False  # the trap
    hits, _ = bar_cache.read_many(
        symbols,
        start="2026-05-04",
        end="2026-05-12",
        source_timeframe="5m",
        feed="sip",
    )
    assert hits == {}


def test_a_cached_sip_hit_is_never_merged_with_an_iex_fallback(
    cached_loader, fake_alpaca, capsys
):
    """MUTATION TEST: delete the tape re-request in fetch_bars and this fails.

    `write_many` refuses to STORE an IEX-fallback batch, but that rule governs
    the write only. AAPL is a genuine SIP entry on disk, MSFT comes back on
    IEX, and the merge would hand `engine.load_data` two tapes for one window
    -- a shape the pre-cache path could not produce, because one request meant
    one feed. The whole universe is re-requested instead: degraded, uniform,
    comparable.
    """
    fake_alpaca["df"] = _bars_df(["AAPL"])
    cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    assert cached_loader.last_fetch["feed"] == "sip"

    class _SipRefusingClient:
        """Alpaca after the key is rotated or the subscription lapses: SIP is
        refused, IEX answers. The cached SIP entry is still valid and still
        inside its TTL."""

        def __init__(self, *args, **kwargs):
            self._session = None

        def get_stock_bars(self, request):
            if getattr(request.feed, "value", request.feed) == "sip":
                raise RuntimeError("your subscription does not permit this")

            class _Bars:
                df = _bars_df(list(request.symbol_or_symbols))

            return _Bars()

    cached_loader.client = _SipRefusingClient()
    result = cached_loader.fetch_bars(["AAPL", "MSFT"], "2026-05-04", "2026-05-12")
    assert set(result) == {"AAPL", "MSFT"}
    assert {frame.attrs[FRAME_ATTR_FEED] for frame in result.values()} == {"iex"}
    assert all(
        frame.attrs[FRAME_ATTR_SIP_FALLBACK] is True for frame in result.values()
    )
    assert "re-requesting" in capsys.readouterr().out
    # The SIP entry is left on disk: it is a correct entry for its key, and
    # the subscription may come back before its TTL does.
    hits, _ = bar_cache.read_many(
        ["AAPL"],
        start="2026-05-04",
        end="2026-05-12",
        source_timeframe="5m",
        feed="sip",
    )
    assert set(hits) == {"AAPL"}


def test_an_all_hit_request_is_untouched_by_the_tape_guard(cached_loader, fake_alpaca):
    """The guard fires only when a live fetch changed tape. With nothing to
    fetch there is no second tape, and re-requesting would undo the cache."""
    fake_alpaca["df"] = _bars_df(["AAPL"])
    cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    fake_alpaca["requests"].clear()
    result = cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    assert set(result) == {"AAPL"}
    assert fake_alpaca["requests"] == []


def test_an_unconfigured_client_caches_nothing(cached_loader, tmp_path):
    cached_loader.client = None
    assert cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12") == {}
    assert cached_loader.last_fetch is None
    cache_dir = tmp_path / "bar_cache"
    assert not cache_dir.exists() or list(cache_dir.iterdir()) == []


def test_a_symbol_alpaca_did_not_return_is_not_cached(cached_loader, fake_alpaca):
    fake_alpaca["df"] = _bars_df(["AAPL"])
    cached_loader.fetch_bars(["AAPL", "NOPE"], "2026-05-04", "2026-05-12")
    fake_alpaca["df"] = _bars_df(["NOPE"])
    result = cached_loader.fetch_bars(["AAPL", "NOPE"], "2026-05-04", "2026-05-12")
    # AAPL served from disk, NOPE re-requested because nothing was stored.
    assert _requested(fake_alpaca) == [["AAPL", "NOPE"], ["NOPE"]]
    assert set(result) == {"AAPL", "NOPE"}


def test_the_cache_resolves_above_the_hundred_symbol_recursion(
    cached_loader, fake_alpaca
):
    """Below the recursion, the cache would run once per 100-symbol chunk and
    the chunking -- not the cache -- would decide what is fetched. Above it,
    the recursion simply sees a shorter list."""
    symbols = [f"S{i:03}" for i in range(235)]
    fake_alpaca["df"] = _bars_df(symbols, rows=1)
    cached_loader.fetch_bars(symbols, "2026-05-04", "2026-05-12")
    assert [len(batch) for batch in _requested(fake_alpaca)] == [100, 100, 35]
    fake_alpaca["requests"].clear()
    fake_alpaca["df"] = _bars_df(symbols[:5], rows=1)
    cached_loader.fetch_bars(symbols, "2026-05-04", "2026-05-12")
    assert _requested(fake_alpaca) == []  # all 235 served from disk


def test_a_partially_warm_large_request_rebatches_only_the_misses(
    cached_loader, fake_alpaca
):
    symbols = [f"S{i:03}" for i in range(235)]
    fake_alpaca["df"] = _bars_df(symbols[:120], rows=1)
    cached_loader.fetch_bars(symbols[:120], "2026-05-04", "2026-05-12")
    fake_alpaca["requests"].clear()
    fake_alpaca["df"] = _bars_df(symbols[120:], rows=1)
    result = cached_loader.fetch_bars(symbols, "2026-05-04", "2026-05-12")
    assert [len(batch) for batch in _requested(fake_alpaca)] == [100, 15]
    assert len(result) == 235


def test_an_empty_symbol_list_does_not_take_the_all_hit_shortcut(
    cached_loader, fake_alpaca, tmp_path
):
    """With no symbols there are no misses either, so a shortcut keyed only on
    `not misses` would fire and `next(symbol for symbol in symbols ...)` would
    raise StopIteration straight out of fetch_bars. The `not symbols` guard
    ahead of it is what prevents that."""
    assert cached_loader.fetch_bars([], "2026-05-04", "2026-05-12") == {}
    cache_dir = tmp_path / "bar_cache"
    assert not cache_dir.exists() or list(cache_dir.iterdir()) == []


def test_a_disabled_cache_restores_the_old_behaviour(
    cached_loader, fake_alpaca, monkeypatch
):
    monkeypatch.setenv("ATL_BAR_CACHE", "0")
    fake_alpaca["df"] = _bars_df(["AAPL"])
    cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    assert len(fake_alpaca["requests"]) == 2
