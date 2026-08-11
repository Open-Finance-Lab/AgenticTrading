"""Tests for gapless equity plot rendering."""

from datetime import datetime

import pytest
import pytz
import requests

from dashboard.backend.domain.leaderboard.strategies._yahoo import YahooChartError
from dashboard.backend.equity_plot import (
    build_backtest_chart_data,
    compute_index_baseline_values,
    gapless_chart_x_labels,
    gapless_market_axis,
    market_index_baselines_with_status,
    render_backtest_equity_png,
)


def test_gapless_axis_advances_one_hour_per_market_bar():
    et = pytz.timezone("US/Eastern")
    fri = et.localize(datetime(2026, 5, 1, 10, 30))
    mon = et.localize(datetime(2026, 5, 4, 10, 30))
    x, ts_et = gapless_market_axis([fri, mon])
    assert len(x) == 2
    assert x[1] - x[0] == pytest.approx(1.0 / 24.0)
    assert ts_et[0].date().isoformat() == "2026-05-01"
    assert ts_et[1].date().isoformat() == "2026-05-04"


def test_compute_index_baseline_values_scales_to_initial_capital(monkeypatch):
    et = pytz.timezone("US/Eastern")
    t0 = et.localize(datetime(2026, 5, 1, 10, 30)).astimezone(pytz.UTC)
    t1 = et.localize(datetime(2026, 5, 1, 11, 30)).astimezone(pytz.UTC)

    monkeypatch.setattr(
        "dashboard.backend.equity_plot.fetch_index_hourly",
        lambda _sym, _start, _end: [(t0, 40_000.0), (t1, 41_000.0)],
    )

    values = compute_index_baseline_values(
        "^DJI",
        [t0, t1],
        "2026-05-01",
        "2026-05-01",
        100_000.0,
    )
    assert values == [100_000.0, pytest.approx(102_500.0)]


def _session_stamps():
    et = pytz.timezone("US/Eastern")
    return (
        et.localize(datetime(2026, 5, 1, 10, 30)).astimezone(pytz.UTC),
        et.localize(datetime(2026, 5, 1, 11, 30)).astimezone(pytz.UTC),
    )


def test_market_index_baselines_report_upstream_failure(monkeypatch, capsys):
    # Yahoo unreachable: drop the baselines, flag it, and say so on stdout —
    # never let the exception escape into a caller rendering a public chart.
    t0, t1 = _session_stamps()

    def boom(_sym, _start, _end):
        raise requests.ConnectionError("yahoo unreachable")

    monkeypatch.setattr("dashboard.backend.equity_plot.fetch_index_hourly", boom)

    baselines, upstream_ok = market_index_baselines_with_status(
        [t0, t1], "2026-05-01", "2026-05-01", 100_000.0
    )
    assert baselines == []
    assert upstream_ok is False
    out = capsys.readouterr().out
    assert "index baseline ^DJI unavailable: ConnectionError" in out
    assert "index baseline ^NDX unavailable: ConnectionError" in out


def test_market_index_baselines_distinguish_absent_from_broken(monkeypatch):
    # Yahoo answered and simply had nothing for the window. Same empty baselines
    # as an outage, but upstream_ok stays True — the flag is the only thing
    # separating "no data" from "upstream is down", and callers key retries and
    # cacheability off it.
    t0, t1 = _session_stamps()
    monkeypatch.setattr(
        "dashboard.backend.equity_plot.fetch_index_hourly", lambda _s, _a, _b: []
    )

    baselines, upstream_ok = market_index_baselines_with_status(
        [t0, t1], "2026-05-01", "2026-05-01", 100_000.0
    )
    assert baselines == []
    assert upstream_ok is True


def test_market_index_baselines_partial_outage_keeps_the_symbol_that_answered(monkeypatch):
    t0, t1 = _session_stamps()

    def only_djia(symbol, _start, _end):
        if symbol == "^NDX":
            raise requests.Timeout("read timeout")
        return [(t0, 40_000.0), (t1, 41_000.0)]

    monkeypatch.setattr("dashboard.backend.equity_plot.fetch_index_hourly", only_djia)

    baselines, upstream_ok = market_index_baselines_with_status(
        [t0, t1], "2026-05-01", "2026-05-01", 100_000.0
    )
    assert [label for label, _key, _values in baselines] == ["DJIA index"]
    assert upstream_ok is False  # partial is still incomplete: retry it


def test_market_index_baselines_skip_an_unusable_window_without_fetching(monkeypatch, capsys):
    # A run window that doesn't parse used to raise ValueError out of _epoch
    # *before* any HTTP call, so no transport-level guard could catch it and the
    # public plot.png route 500'd. paper_trading.py writes end_date="", and the
    # route passes `run.get("end_date") or ""`, so this window is reachable.
    t0, t1 = _session_stamps()

    def must_not_be_called(*_a, **_k):  # pragma: no cover - asserts absence
        raise AssertionError("Yahoo must not be asked for an unparseable window")

    monkeypatch.setattr(
        "dashboard.backend.equity_plot.fetch_index_hourly", must_not_be_called
    )

    baselines, upstream_ok = market_index_baselines_with_status(
        [t0, t1], "2026-05-01", "", 100_000.0, context="run_abc"
    )
    assert baselines == []
    # True, not False: the window is *permanently* baseline-free, so there is
    # nothing to retry and the render is safe to cache. The flag means
    # "transient and retryable", which is the only thing a caller can act on.
    assert upstream_ok is True
    out = capsys.readouterr().out
    assert "unusable run window" in out
    assert "run_abc" in out  # the log line names the request, not just a symbol


def test_log_context_cannot_forge_a_log_line(monkeypatch, capsys):
    # `context` is a run id from a URL path parameter. A newline in it would let
    # one request write what looks like a second, unrelated log entry.
    t0, t1 = _session_stamps()
    monkeypatch.setattr(
        "dashboard.backend.equity_plot.fetch_index_hourly", lambda *_a, **_k: []
    )

    market_index_baselines_with_status(
        [t0, t1], "2026-05-01", "", 100_000.0,
        context="run_1\n⚠️ index baselines skipped [forged]: unusable run window",
    )
    out = capsys.readouterr().out
    assert out.count("index baselines skipped") == 1
    assert "\n⚠️" not in out.rstrip("\n")


def test_market_index_baselines_treat_a_200_delivered_failure_as_broken(monkeypatch):
    # Yahoo answers 200 with an error envelope. The status code says "fine", so
    # only YahooChartError keeps this out of the "no data for this window" bucket.
    t0, t1 = _session_stamps()

    def error_envelope(_sym, _start, _end):
        raise YahooChartError("^DJI: Invalid Crumb")

    monkeypatch.setattr(
        "dashboard.backend.equity_plot.fetch_index_hourly", error_envelope
    )

    baselines, upstream_ok = market_index_baselines_with_status(
        [t0, t1], "2026-05-01", "2026-05-01", 100_000.0
    )
    assert baselines == []
    assert upstream_ok is False  # retryable, and the chart is incomplete


def test_render_note_changes_the_rendered_bytes():
    # The caption is the only thing marking a Discord-posted chart as degraded;
    # that artifact is permanent and cannot be re-fetched once Yahoo recovers.
    t0, t1 = _session_stamps()
    kwargs = dict(
        agent_label="Agent",
        agent_run_id="run_note",
        timestamps=[t0, t1],
        agent_values=[100_000.0, 101_000.0],
        baselines=[],
    )
    assert render_backtest_equity_png(**kwargs) != render_backtest_equity_png(
        **kwargs, note="⚠ Index benchmarks unavailable"
    )


def test_chart_data_reports_index_baseline_status(monkeypatch):
    # The JSON chart path must expose the flag too, or /chart-data 200s with the
    # benchmark silently missing and the Playground paints an unexplained
    # single-line chart.
    t0, t1 = _session_stamps()
    curve = [
        {"timestamp": t0.isoformat(), "equity": 100_000},
        {"timestamp": t1.isoformat(), "equity": 100_500},
    ]
    monkeypatch.setattr(
        "dashboard.backend.equity_plot.fetch_index_hourly",
        lambda *_a, **_k: (_ for _ in ()).throw(requests.ConnectionError("down")),
    )

    payload = build_backtest_chart_data(
        run_id="agent_test_outage",
        agent_name="Agent",
        llm_model=None,
        start_date="2026-05-01",
        end_date="2026-05-01",
        initial_capital=100_000,
        agent_curve=curve,
    )
    assert payload["index_baselines_ok"] is False
    assert [s["label"] for s in payload["series"][1:]] == []


def test_market_index_baselines_do_not_swallow_non_transport_errors(monkeypatch):
    # A delivered-but-malformed payload is a different bug. Swallowing it here
    # would turn a code defect into a permanently baseline-free chart.
    t0, t1 = _session_stamps()

    def garbage(_sym, _start, _end):
        raise TypeError("unorderable index payload")

    monkeypatch.setattr("dashboard.backend.equity_plot.fetch_index_hourly", garbage)

    with pytest.raises(TypeError):
        market_index_baselines_with_status([t0, t1], "2026-05-01", "2026-05-01", 100_000.0)


def test_gapless_chart_x_labels_anchor_at_day_start():
    et = pytz.timezone("US/Eastern")
    stamps = [
        et.localize(datetime(2026, 5, 1, 10, 30)),
        et.localize(datetime(2026, 5, 1, 11, 30)),
        et.localize(datetime(2026, 5, 4, 10, 30)),
    ]
    labels = gapless_chart_x_labels(stamps)
    assert labels == ["2026-05-01", "", "2026-05-04"]


def test_build_backtest_chart_data_uses_card_name(monkeypatch):
    et = pytz.timezone("US/Eastern")
    t0 = et.localize(datetime(2026, 5, 1, 10, 30))
    t1 = et.localize(datetime(2026, 5, 1, 11, 30))
    curve = [
        {"timestamp": t0.isoformat(), "equity": 100_000},
        {"timestamp": t1.isoformat(), "equity": 100_500},
    ]

    monkeypatch.setattr(
        "dashboard.backend.equity_plot.market_index_baselines_with_status",
        lambda *_a, **_k: (
            [
                ("DJIA index", "index:^DJI", [100_000, 99_800]),
                ("Nasdaq-100", "index:^NDX", [100_000, 99_700]),
            ],
            True,
        ),
    )

    payload = build_backtest_chart_data(
        run_id="agent_test_1",
        agent_name="Agent",
        llm_model="claude-haiku-4.5",
        start_date="2026-05-01",
        end_date="2026-05-01",
        initial_capital=100_000,
        agent_curve=curve,
        card_name="test agent 1",
    )
    assert payload["series"][0]["label"] == "test agent 1"
    assert [s["label"] for s in payload["series"][1:]] == ["DJIA index", "Nasdaq-100"]
    assert payload["x_labels"] == ["2026-05-01", ""]


def test_render_backtest_equity_png_bytes():
    et = pytz.timezone("US/Eastern")
    stamps = [
        et.localize(datetime(2026, 5, 1, 10, 30)),
        et.localize(datetime(2026, 5, 1, 11, 30)),
        et.localize(datetime(2026, 5, 4, 10, 30)),
    ]
    png = render_backtest_equity_png(
        agent_label="Agent",
        agent_run_id="agent_test_1",
        timestamps=stamps,
        agent_values=[100_000, 100_500, 101_000],
        baselines=[
            ("DJIA index", "index:^DJI", [100_000, 99_800, 99_600]),
            ("Nasdaq-100", "index:^NDX", [100_000, 99_700, 99_400]),
        ],
    )
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
