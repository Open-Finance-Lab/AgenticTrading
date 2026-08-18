"""Characterization tests for the extracted AlpacaDataLoader (Phase 2B1).

No real network/API calls: the Alpaca SDK client is replaced with a fake via
monkeypatch. Imports use the canonical package path.
"""

import json

import pandas as pd
import pytest

from alpaca.data.timeframe import TimeFrame

from dashboard.backend.infrastructure.market_data.alpaca_bars import (
    FRAME_ATTR_FEED,
    FRAME_ATTR_SIP_FALLBACK,
    AlpacaDataLoader,
    clamp_end_for_sip,
    feed_provenance,
)
from dashboard.scripts import backtest_hourly_agent as bha

CLIENT_TARGET = "alpaca.data.historical.StockHistoricalDataClient"
# Patch target for the clamp. A string keeps this module on ONE import form for
# alpaca_bars: adding `import ... as bars_mod` alongside the `from` import above
# is what py/import-and-import-from flags.
CLAMP_TARGET = (
    "dashboard.backend.infrastructure.market_data.alpaca_bars.clamp_end_for_sip"
)


def _bars_df(symbol_to_rows):
    """Build an Alpaca-style multi-index (symbol, timestamp) OHLCV dataframe."""
    frames = []
    for sym, rows in symbol_to_rows.items():
        idx = pd.MultiIndex.from_tuples(
            [(sym, pd.Timestamp(ts)) for ts, *_ in rows],
            names=["symbol", "timestamp"],
        )
        frame = pd.DataFrame(
            [
                {"open": o, "high": h, "low": l, "close": c, "volume": v}
                for _, o, h, l, c, v in rows
            ],
            index=idx,
        )
        frames.append(frame)
    return pd.concat(frames)


def _empty_bars_df():
    return pd.DataFrame(
        {"open": [], "high": [], "low": [], "close": [], "volume": []},
        index=pd.MultiIndex.from_arrays([[], []], names=["symbol", "timestamp"]),
    )


@pytest.fixture
def fake_alpaca(monkeypatch):
    """Patch the Alpaca client; returns a controllable state dict."""
    state = {"df": _empty_bars_df(), "exc": None, "requests": [], "ctor": []}

    class _FakeBars:
        def __init__(self, df):
            self.df = df

    class _FakeSession:
        """Just enough of ``requests.Session`` for ``_apply_default_timeout``
        to find a wrappable ``.request`` -- these tests call
        ``get_stock_bars`` directly, never through ``_session.request``, so
        it's never actually invoked. Without this, every ``AlpacaDataLoader``
        constructed in this file hits the no-``_session`` warning path and
        prints "timeout was NOT applied" on every passing test; that
        production warning is exercised deliberately (with no ``_session``
        at all) by ``test_alpaca_http_timeout.py``."""

        def request(self, *args, **kwargs):
            raise NotImplementedError("not exercised by these tests")

    class _FakeClient:
        def __init__(self, api_key, secret_key):
            state["ctor"].append((api_key, secret_key))
            self._session = _FakeSession()

        def get_stock_bars(self, request):
            state["requests"].append(request)
            if state["exc"] is not None:
                raise state["exc"]
            return _FakeBars(state["df"])

    monkeypatch.setattr(CLIENT_TARGET, _FakeClient)
    return state


# --- compatibility / identity ----------------------------------------------

def test_old_script_exports_class():
    assert hasattr(bha, "AlpacaDataLoader")


def test_class_identity_between_paths():
    assert bha.AlpacaDataLoader is AlpacaDataLoader


# --- constructor -----------------------------------------------------------

def test_constructor_with_explicit_keys(fake_alpaca):
    loader = AlpacaDataLoader(api_key="explicit-k", secret_key="explicit-s")
    assert loader.api_key == "explicit-k"
    assert loader.secret_key == "explicit-s"
    assert loader.base_url == "https://data.alpaca.markets"
    # explicit keys -> credentials loader not used
    assert fake_alpaca["ctor"] == [("explicit-k", "explicit-s")]


def test_credentials_from_environment(fake_alpaca, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "env-k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "env-s")
    loader = AlpacaDataLoader()
    assert loader.api_key == "env-k"
    assert loader.secret_key == "env-s"


def test_environment_takes_precedence_over_file(fake_alpaca, monkeypatch, tmp_path):
    monkeypatch.setenv("ALPACA_API_KEY", "env-k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "env-s")
    monkeypatch.setattr(
        "dashboard.backend.infrastructure.market_data.alpaca_bars.CREDENTIALS_DIR",
        tmp_path,
    )
    (tmp_path / "alpaca.json").write_text(
        json.dumps({"api_key": "file-k", "secret_key": "file-s"})
    )
    loader = AlpacaDataLoader()
    assert loader.api_key == "env-k"  # env wins over file


def test_credentials_from_file_fallback(fake_alpaca, monkeypatch, tmp_path):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.setattr(
        "dashboard.backend.infrastructure.market_data.alpaca_bars.CREDENTIALS_DIR",
        tmp_path,
    )
    (tmp_path / "alpaca.json").write_text(
        json.dumps({"api_key": "file-k", "secret_key": "file-s"})
    )
    loader = AlpacaDataLoader()
    assert loader.api_key == "file-k"
    assert loader.secret_key == "file-s"


def test_missing_credentials_raises(fake_alpaca, monkeypatch, tmp_path):
    """Missing credentials raise MarketDataUnavailableError — deliberately NOT
    SystemExit (B0 deep fix): a plain exception is catchable by the server's
    `except Exception` boundaries. See tests/test_market_data_errors.py."""
    from dashboard.backend.infrastructure.market_data.alpaca_bars import (
        MarketDataUnavailableError,
    )

    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.setattr(
        "dashboard.backend.infrastructure.market_data.alpaca_bars.CREDENTIALS_DIR",
        tmp_path,  # no alpaca.json here
    )
    with pytest.raises(MarketDataUnavailableError):
        AlpacaDataLoader()


# --- fetch_bars ------------------------------------------------------------

@pytest.fixture(autouse=True)
def _hermetic_alpaca_feed_env(monkeypatch):
    monkeypatch.delenv("ALPACA_DATA_FEED", raising=False)
    monkeypatch.delenv("ALPACA_ALLOW_RECENT_SIP", raising=False)
    monkeypatch.delenv("ALPACA_SIP_DELAY_MINUTES", raising=False)


def test_request_construction(fake_alpaca):
    from alpaca.data.enums import DataFeed

    loader = AlpacaDataLoader(api_key="k", secret_key="s")
    fake_alpaca["df"] = _bars_df({"AAPL": [("2026-01-02 10:00", 1, 2, 0.5, 1.5, 100)]})
    out = loader.fetch_bars(["AAPL"], "2026-01-01", "2026-01-03")
    req = fake_alpaca["requests"][0]
    assert req.symbol_or_symbols == ["AAPL"]
    assert req.timeframe.value == TimeFrame.Hour.value
    assert str(req.start).startswith("2026-01-01")
    # Historical exclusive end is already older than the 15m SIP window.
    assert str(req.end).startswith("2026-01-03")
    assert req.feed == DataFeed.SIP
    assert out["AAPL"].attrs[FRAME_ATTR_FEED] == "sip"
    assert out["AAPL"].attrs[FRAME_ATTR_SIP_FALLBACK] is False
    assert loader.last_fetch["sip_fallback_to_iex"] is False


def test_clamp_end_for_sip_helper():
    from datetime import datetime, timezone

    from dashboard.backend.infrastructure.market_data.alpaca_bars import clamp_end_for_sip

    now = datetime(2026, 8, 12, 23, 50, tzinfo=timezone.utc)
    clamped = clamp_end_for_sip("2026-08-13", now=now, delay_minutes=15)
    assert clamped == datetime(2026, 8, 12, 23, 35, tzinfo=timezone.utc)

    untouched = clamp_end_for_sip("2026-07-01", now=now, delay_minutes=15)
    assert untouched == datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc)


def test_clamp_never_moves_end_before_start():
    """A same-day run just after 00:00 UTC must not be clamped into yesterday.

    Without the ``start`` floor the cutoff (now−15m) precedes the requested
    start, Alpaca answers an inverted range with nothing, and the caller's
    negative cache pins that as a hard failure for the whole TTL.
    """
    from datetime import datetime, timezone

    from dashboard.backend.infrastructure.market_data.alpaca_bars import clamp_end_for_sip

    now = datetime(2026, 8, 13, 0, 5, tzinfo=timezone.utc)
    clamped = clamp_end_for_sip(
        "2026-08-14", start="2026-08-13", now=now, delay_minutes=15
    )
    assert clamped >= datetime(2026, 8, 13, 0, 0, tzinfo=timezone.utc)

    # Without a start floor the same call inverts the window.
    unfloored = clamp_end_for_sip("2026-08-14", now=now, delay_minutes=15)
    assert unfloored < datetime(2026, 8, 13, 0, 0, tzinfo=timezone.utc)


def test_clamp_keeps_final_rth_bar_after_close():
    """The 15-minute cutoff must stay later than the last hourly bar's open.

    Alpaca filters bars on their opening timestamp, so a cutoff of 15:50 ET
    still returns the complete 15:00–16:00 ET bar. The margin is one bar wide:
    a delay above ~65 minutes would drop the closing hour, and the daily board
    would cache that truncated curve for the rest of the session.
    """
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo

    from dashboard.backend.infrastructure.market_data.alpaca_bars import (
        DEFAULT_SIP_DELAY_MINUTES,
        clamp_end_for_sip,
    )

    eastern = ZoneInfo("America/New_York")
    just_after_close = datetime(2026, 8, 13, 16, 5, tzinfo=eastern)
    last_bar_open = datetime(2026, 8, 13, 15, 0, tzinfo=eastern)

    clamped = clamp_end_for_sip(
        "2026-08-14",
        start="2026-08-13",
        now=just_after_close.astimezone(timezone.utc),
        delay_minutes=DEFAULT_SIP_DELAY_MINUTES,
    )
    assert clamped > last_bar_open


def test_clamp_returns_unparseable_end_unchanged():
    """``end`` is unvalidated user input; the clamp must not become a validator.

    Raising here would surface a bare ValueError from a new place, ahead of the
    SDK's own (better) validation error.
    """
    from dashboard.backend.infrastructure.market_data.alpaca_bars import clamp_end_for_sip

    assert clamp_end_for_sip("08/13/2026", delay_minutes=15) == "08/13/2026"
    assert clamp_end_for_sip("not-a-date", delay_minutes=15) == "not-a-date"


def test_unknown_feed_raises(monkeypatch):
    """A typo must not silently price a published run off the other tape."""
    from alpaca.data.enums import DataFeed

    from dashboard.backend.infrastructure.market_data.alpaca_bars import (
        AlpacaFeedConfigError,
        configured_feed_name,
        resolve_alpaca_data_feed,
    )

    monkeypatch.setenv("ALPACA_DATA_FEED", "IEXX")
    with pytest.raises(AlpacaFeedConfigError):
        configured_feed_name()
    with pytest.raises(AlpacaFeedConfigError):
        resolve_alpaca_data_feed(DataFeed)


def test_configured_feed_name_defaults_and_normalizes(monkeypatch):
    from dashboard.backend.infrastructure.market_data.alpaca_bars import (
        configured_feed_name,
    )

    monkeypatch.delenv("ALPACA_DATA_FEED", raising=False)
    assert configured_feed_name() == "sip"

    monkeypatch.setenv("ALPACA_DATA_FEED", "  IEX ")
    assert configured_feed_name() == "iex"


def test_feed_provenance_reads_frame_stamps(fake_alpaca):
    """Provenance must survive as data, not only as a log line."""
    from dashboard.backend.infrastructure.market_data.alpaca_bars import feed_provenance

    loader = AlpacaDataLoader(api_key="k", secret_key="s")
    fake_alpaca["df"] = _bars_df({"AAPL": [("2026-01-02 10:00", 1, 2, 0.5, 1.5, 100)]})
    out = loader.fetch_bars(["AAPL"], "2026-01-01", "2026-01-03")

    assert feed_provenance(out) == {
        "market_data_feed": "sip",
        "sip_fallback_to_iex": False,
        "end_clamped": False,
    }
    # Nothing to attribute when no Alpaca frame was involved.
    assert feed_provenance({}) is None
    assert feed_provenance({"AAPL": pd.DataFrame()}) is None


def test_clamped_fetch_is_marked_in_provenance(fake_alpaca, monkeypatch):
    from datetime import datetime, timezone

    frozen = datetime(2026, 8, 12, 23, 50, tzinfo=timezone.utc)

    def _clamp(end, *, start=None, now=None, delay_minutes=None):
        return clamp_end_for_sip(
            end, start=start, now=now or frozen, delay_minutes=delay_minutes
        )

    monkeypatch.setattr(CLAMP_TARGET, _clamp)
    loader = AlpacaDataLoader(api_key="k", secret_key="s")
    fake_alpaca["df"] = _bars_df({"AAPL": [("2026-08-12 10:00", 1, 2, 0.5, 1.5, 100)]})
    out = loader.fetch_bars(["AAPL"], "2026-07-12", "2026-08-13")

    assert feed_provenance(out)["end_clamped"] is True
    assert loader.last_fetch["end_clamped"] is True


def test_sip_fetch_uses_clamped_end(fake_alpaca, monkeypatch):
    from datetime import datetime, timezone

    from alpaca.data.enums import DataFeed

    frozen = datetime(2026, 8, 12, 23, 50, tzinfo=timezone.utc)
    clamped = datetime(2026, 8, 12, 23, 35, tzinfo=timezone.utc)

    def _clamp(end, *, start=None, now=None, delay_minutes=None):
        return clamp_end_for_sip(
            end, start=start, now=now or frozen, delay_minutes=delay_minutes
        )

    monkeypatch.setenv("ALPACA_DATA_FEED", "sip")
    monkeypatch.setattr(CLAMP_TARGET, _clamp)
    loader = AlpacaDataLoader(api_key="k", secret_key="s")
    fake_alpaca["df"] = _bars_df({"AAPL": [("2026-08-12 10:00", 1, 2, 0.5, 1.5, 100)]})
    loader.fetch_bars(["AAPL"], "2026-07-12", "2026-08-13")
    req = fake_alpaca["requests"][0]
    assert req.feed == DataFeed.SIP
    # alpaca-py may drop tzinfo when storing the request field.
    assert req.end.replace(tzinfo=timezone.utc) == clamped
    assert loader.last_fetch["sip_fallback_to_iex"] is False


def test_subscription_error_retries_iex(fake_alpaca):
    from alpaca.data.enums import DataFeed

    loader = AlpacaDataLoader(api_key="k", secret_key="s")

    class _Flaky:
        def __init__(self):
            self.calls = 0

        def get_stock_bars(self, request):
            self.calls += 1
            fake_alpaca["requests"].append(request)
            if self.calls == 1:
                raise RuntimeError('{"message":"subscription does not permit querying recent SIP data"}')
            return type("Bars", (), {"df": _bars_df({"AAPL": [("2026-01-02 10:00", 1, 2, 0.5, 1.5, 100)]})})()

    loader.client = _Flaky()
    out = loader.fetch_bars(["AAPL"], "2026-01-01", "2026-01-03")
    assert set(out.keys()) == {"AAPL"}
    assert fake_alpaca["requests"][0].feed == DataFeed.SIP
    assert fake_alpaca["requests"][1].feed == DataFeed.IEX
    assert out["AAPL"].attrs[FRAME_ATTR_FEED] == "iex"
    assert out["AAPL"].attrs[FRAME_ATTR_SIP_FALLBACK] is True
    assert loader.last_fetch["feed"] == "iex"
    assert loader.last_fetch["sip_fallback_to_iex"] is True
    assert loader.last_fetch["requested_end"] == "2026-01-03"


def test_single_symbol_response_schema(fake_alpaca):
    loader = AlpacaDataLoader(api_key="k", secret_key="s")
    fake_alpaca["df"] = _bars_df(
        {
            "AAPL": [
                ("2026-01-02 10:00", 10, 11, 9, 10.5, 1000),
                ("2026-01-02 11:00", 10.5, 12, 10, 11.5, 1200),
            ]
        }
    )
    out = loader.fetch_bars(["AAPL"], "2026-01-01", "2026-01-03")
    assert set(out.keys()) == {"AAPL"}
    df = out["AAPL"]
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "timestamp"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df) == 2


def test_multi_symbol_response(fake_alpaca):
    loader = AlpacaDataLoader(api_key="k", secret_key="s")
    fake_alpaca["df"] = _bars_df(
        {
            "AAPL": [("2026-01-02 10:00", 10, 11, 9, 10.5, 1000)],
            "MSFT": [("2026-01-02 10:00", 20, 21, 19, 20.5, 2000)],
        }
    )
    out = loader.fetch_bars(["AAPL", "MSFT"], "2026-01-01", "2026-01-03")
    assert set(out.keys()) == {"AAPL", "MSFT"}
    assert out["MSFT"]["close"].iloc[0] == 20.5


def test_missing_symbol_skipped(fake_alpaca):
    loader = AlpacaDataLoader(api_key="k", secret_key="s")
    fake_alpaca["df"] = _bars_df({"AAPL": [("2026-01-02 10:00", 10, 11, 9, 10.5, 1000)]})
    out = loader.fetch_bars(["AAPL", "TSLA"], "2026-01-01", "2026-01-03")
    assert set(out.keys()) == {"AAPL"}  # TSLA absent -> skipped


def test_empty_response_returns_empty_dict(fake_alpaca):
    loader = AlpacaDataLoader(api_key="k", secret_key="s")
    fake_alpaca["df"] = _empty_bars_df()
    out = loader.fetch_bars(["AAPL"], "2026-01-01", "2026-01-03")
    assert out == {}


def test_results_sorted_by_timestamp(fake_alpaca):
    loader = AlpacaDataLoader(api_key="k", secret_key="s")
    fake_alpaca["df"] = _bars_df(
        {
            "AAPL": [
                ("2026-01-02 13:00", 13, 14, 12, 13.5, 1300),
                ("2026-01-02 10:00", 10, 11, 9, 10.5, 1000),
                ("2026-01-02 11:00", 11, 12, 10, 11.5, 1100),
            ]
        }
    )
    out = loader.fetch_bars(["AAPL"], "2026-01-01", "2026-01-03")
    idx = out["AAPL"].index
    assert list(idx) == sorted(idx)


def test_timezone_preserved(fake_alpaca):
    loader = AlpacaDataLoader(api_key="k", secret_key="s")
    fake_alpaca["df"] = _bars_df(
        {"AAPL": [(pd.Timestamp("2026-01-02 10:00", tz="UTC"), 10, 11, 9, 10.5, 1000)]}
    )
    out = loader.fetch_bars(["AAPL"], "2026-01-01", "2026-01-03")
    assert out["AAPL"].index.tz is not None
    assert str(out["AAPL"].index.tz) == "UTC"


def test_timezone_naive_preserved(fake_alpaca):
    loader = AlpacaDataLoader(api_key="k", secret_key="s")
    fake_alpaca["df"] = _bars_df({"AAPL": [("2026-01-02 10:00", 10, 11, 9, 10.5, 1000)]})
    out = loader.fetch_bars(["AAPL"], "2026-01-01", "2026-01-03")
    assert out["AAPL"].index.tz is None


def test_exception_is_caught_and_returns_empty(fake_alpaca):
    loader = AlpacaDataLoader(api_key="k", secret_key="s")
    fake_alpaca["exc"] = RuntimeError("boom")
    out = loader.fetch_bars(["AAPL"], "2026-01-01", "2026-01-03")
    assert out == {}
