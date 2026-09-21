"""MEDIUM #2 / #3 — /backtest routes hardening.

#3: GET /runs/{run_id}/plot.png must not block the event loop (sync handler ->
    threadpool), must not re-import/re-configure matplotlib per request, and
    should cache the immutable rendered PNG per run_id.
#2: POST /backtest/run must not let an anonymous caller burn operator LLM
    credits — model allowlist, prompt size cap, date-range cap, write rate limit.
"""

import inspect
import json
import subprocess
import threading
import time
import uuid
from datetime import datetime

import pytest
import pytz
import requests
from fastapi import HTTPException
from fastapi.testclient import TestClient

from dashboard.backend.app import app
from dashboard.backend.api.rate_limit import FixedWindowRateLimiter
from dashboard.backend.domain.model_providers.execution_catalog import (
    ExecutionModelRoute,
    UnsupportedExecutionModel,
)
import dashboard.backend.api.routers.backtests as bt
from dashboard.backend.tests._fake_child import FakeChild


_REAL_RUN_BACKTEST_BACKGROUND = bt.run_backtest_background


# ===========================================================================
# #3 — plot.png: event loop + caching
# ===========================================================================

def test_plot_png_handler_is_sync_offloaded():
    # Sync def -> FastAPI runs the CPU-bound render in a threadpool, not on the
    # event loop. (Was `async def`, which blocked the loop for the whole render.)
    assert not inspect.iscoroutinefunction(bt.get_run_plot)


def test_plot_png_matplotlib_hoisted_to_module():
    # The renderer no longer imports/configures matplotlib per call.
    src = inspect.getsource(bt._render_run_plot_png)
    assert "import matplotlib" not in src
    assert 'matplotlib.use(' not in src
    # It's configured once at module import instead.
    assert bt.matplotlib.get_backend().lower() == "agg"


def test_plot_png_cached_per_run(monkeypatch):
    bt._render_run_plot_png.cache_clear()
    calls = {"get_run": 0, "equity": 0, "yahoo": 0}
    fake_run = {
        "session_id": None, "created_at": "2026-05-01T10:00:00", "agent_name": "Agent",
        "start_date": "2026-05-01", "end_date": "2026-05-07", "mode": "safe_trading",
        "baseline_buyhold_run_id": None, "baseline_djia_run_id": None,
    }

    # #139's traceback was lost, so the real flake mechanism is unconfirmed.
    # Defense in depth against cross-test interference: the render cache is a
    # module-level lru_cache and bt.db is a process-wide singleton (patched
    # for any thread while this test runs), so (a) a unique per-invocation
    # key guarantees no other code path can collide with this test's cache
    # entry, and (b) the counters only count calls for *this* key, so a stray
    # db.get_run from another test's leftover threadpool thread cannot skew
    # the assertions either way.
    run_id = f"run_x_{uuid.uuid4().hex}"

    # 10:30/11:30 ET (== 14:30/15:30 UTC). compute_index_baseline_values first
    # drops every index level failing is_market_hour, then aligns the survivors
    # onto the agent's own timeline with a 30-minute nearest-match tolerance —
    # so the equity curve *and* the stubbed index levels both have to sit inside
    # the session. Stamps outside it (e.g. a naive 10:00 "UTC", which is 06:00
    # ET) make the baselines come back empty with the test still passing.
    et = pytz.timezone("US/Eastern")
    t0 = et.localize(datetime(2026, 5, 1, 10, 30)).astimezone(pytz.UTC)
    t1 = et.localize(datetime(2026, 5, 1, 11, 30)).astimezone(pytz.UTC)

    def fake_get_run(rid):
        if rid == run_id:
            calls["get_run"] += 1
        return fake_run

    def fake_equity(rid):
        if rid == run_id:
            calls["equity"] += 1
        # Naive-UTC ISO strings, the shape the DB stores.
        return [{"timestamp": t0.replace(tzinfo=None).isoformat(), "equity": 100000},
                {"timestamp": t1.replace(tzinfo=None).isoformat(), "equity": 101000}]

    # fake_run carries no `metadata`, so _market_profile_for_run falls back to
    # the Alpaca/US profile with index_baseline_enabled true, which makes
    # _render_run_plot_png call market_index_baselines_with_status ->
    # fetch_index_hourly for ^DJI/^NDX. Without this patch that is a real
    # HTTPS call to query1.finance.yahoo.com on every run (issue #320: flakes
    # when Yahoo is slow/rate-limited). Stub it here, same pattern as
    # test_equity_plot.py; the network fetch is incidental to what this test
    # covers (lru_cache behaviour).
    def fake_fetch_index_hourly(_symbol, start, end):
        if (start, end) == (fake_run["start_date"], fake_run["end_date"]):
            calls["yahoo"] += 1  # same stray-thread gating as the counters above
        return [(t0, 40_000.0), (t1, 41_000.0)]

    monkeypatch.setattr(
        "dashboard.backend.equity_plot.fetch_index_hourly", fake_fetch_index_hourly
    )
    monkeypatch.setattr(bt.db, "get_run", fake_get_run)
    monkeypatch.setattr(bt.db, "get_equity_curve", fake_equity)
    monkeypatch.setattr(bt, "filter_market_hours", lambda pts: pts)  # isolate caching

    # Record what the renderer is handed: an index baseline that got filtered
    # away upstream arrives as [], which would leave this path uncovered while
    # every other assertion still passed.
    rendered = {}
    real_render = bt.render_backtest_equity_png

    def spy_render(**kwargs):
        rendered.setdefault("baselines", kwargs["baselines"])
        return real_render(**kwargs)

    monkeypatch.setattr(bt, "render_backtest_equity_png", spy_render)

    first = bt._render_run_plot_png(run_id)
    second = bt._render_run_plot_png(run_id)

    assert first == second
    assert first[:8] == b"\x89PNG\r\n\x1a\n"      # valid PNG
    assert calls["get_run"] == 1                  # 2nd call served from cache
    assert calls["yahoo"] == 2                    # stub answered ^DJI + ^NDX, no network
    assert [label for label, _key, _values in rendered["baselines"]] == [
        "DJIA index",
        "Nasdaq-100",
    ]
    bt._render_run_plot_png.cache_clear()


def test_plot_png_missing_run_not_cached(monkeypatch):
    # A 404 must not be cached: a run that appears later should still render.
    bt._render_run_plot_png.cache_clear()
    from fastapi import HTTPException
    monkeypatch.setattr(bt.db, "get_run", lambda rid: None)
    with pytest.raises(HTTPException):
        bt._render_run_plot_png("missing")
    # Nothing cached -> a second call re-queries (would render if data existed).
    hits = {"n": 0}

    def counting_get_run(rid):
        if rid == "missing":  # same stray-thread gating as the cache test above
            hits["n"] += 1
        return None

    monkeypatch.setattr(bt.db, "get_run", counting_get_run)
    with pytest.raises(HTTPException):
        bt._render_run_plot_png("missing")
    assert hits["n"] == 1  # re-evaluated, not served from a cached exception
    bt._render_run_plot_png.cache_clear()


def test_plot_png_survives_index_outage_and_retries_it(monkeypatch, capsys):
    """#320, second half: Yahoo going down must cost the chart its baselines, not 500.

    ``/runs/{run_id}/plot.png`` is public and unauthenticated, and app.py only
    registers handlers for ApiError/RequestValidationError — so before this fix a
    Yahoo 429 (the very flakiness #320 opened on) propagated out of the route as
    an unhandled 500. The degraded render must also stay out of the lru_cache, or
    one rate-limited minute would pin a baseline-free chart to that run until the
    process restarted.
    """
    bt._render_run_plot_png.cache_clear()
    run_id = f"run_outage_{uuid.uuid4().hex}"  # unique key, as in the cache test
    fake_run = {
        "session_id": None, "created_at": "2026-05-01T10:00:00", "agent_name": "Agent",
        "start_date": "2026-05-01", "end_date": "2026-05-07", "mode": "safe_trading",
        "baseline_buyhold_run_id": None, "baseline_djia_run_id": None,
    }
    et = pytz.timezone("US/Eastern")
    t0 = et.localize(datetime(2026, 5, 1, 10, 30)).astimezone(pytz.UTC)
    t1 = et.localize(datetime(2026, 5, 1, 11, 30)).astimezone(pytz.UTC)

    monkeypatch.setattr(bt.db, "get_run", lambda rid: fake_run if rid == run_id else None)
    monkeypatch.setattr(
        bt.db, "get_equity_curve",
        lambda rid: [{"timestamp": t0.replace(tzinfo=None).isoformat(), "equity": 100000},
                     {"timestamp": t1.replace(tzinfo=None).isoformat(), "equity": 101000}],
    )
    monkeypatch.setattr(bt, "filter_market_hours", lambda pts: pts)

    outage = {"on": True}
    fetches = {"n": 0}

    def flaky_fetch(_symbol, start, end):
        if (start, end) != (fake_run["start_date"], fake_run["end_date"]):
            return []  # not this test's run; don't touch the counter
        fetches["n"] += 1
        if outage["on"]:
            # What requests raises on 429/5xx via raise_for_status().
            raise requests.HTTPError("429 Client Error: Too Many Requests")
        return [(t0, 40_000.0), (t1, 41_000.0)]

    monkeypatch.setattr("dashboard.backend.equity_plot.fetch_index_hourly", flaky_fetch)
    client = TestClient(app)

    degraded = client.get(f"/runs/{run_id}/plot.png")
    assert degraded.status_code == 200                       # was an unhandled 500
    assert degraded.headers["content-type"] == "image/png"
    assert degraded.content[:8] == b"\x89PNG\r\n\x1a\n"      # a real chart, sans baselines
    assert fetches["n"] == 2

    # Inside the negative-cache TTL the degraded bytes are replayed without
    # re-rendering. This route is public and unauthenticated, so an unbounded
    # retry would let a *persistent* Yahoo block turn every hit into a full
    # matplotlib render — the exact cost the lru_cache exists to avoid.
    assert client.get(f"/runs/{run_id}/plot.png").content == degraded.content
    assert fetches["n"] == 2

    # Past the TTL, Yahoo is tried again: the degraded render was never allowed
    # into the lru_cache, so recovery is never more than one TTL away.
    bt._clear_degraded_plot_cache()
    assert client.get(f"/runs/{run_id}/plot.png").status_code == 200
    assert fetches["n"] == 4

    outage["on"] = False
    bt._clear_degraded_plot_cache()
    healthy = client.get(f"/runs/{run_id}/plot.png")
    assert healthy.status_code == 200
    assert healthy.content != degraded.content  # baselines are back on the chart
    assert fetches["n"] == 6
    again = client.get(f"/runs/{run_id}/plot.png")
    assert again.content == healthy.content
    assert fetches["n"] == 6  # ...and a complete render *is* cached, as before

    # The outage has to be legible in the logs: absent baselines and broken
    # upstream otherwise render identically (see CLAUDE.md, "fail-closed is not
    # fail-visible"). print(), because logging is invisible under prod uvicorn.
    out = capsys.readouterr().out
    assert "index baseline ^DJI unavailable" in out and "HTTPError" in out
    assert "index baseline ^NDX unavailable" in out and "HTTPError" in out
    assert run_id in out  # names the request, not just the symbol
    bt._render_run_plot_png.cache_clear()
    bt._clear_degraded_plot_cache()


def test_degraded_plot_negative_cache_expires_on_its_own(monkeypatch):
    """The negative cache must *lapse*, not just be clearable.

    Bounding the re-render storm is only half the requirement: a degraded chart
    that never expires is the failure the lru_cache escape (_UncachedPlotPng)
    was added to prevent in the first place, just on a slower clock. Exercised
    against the real expiry branch by shrinking the TTL rather than clearing.
    """
    bt._clear_degraded_plot_cache()
    run_id = f"run_ttl_{uuid.uuid4().hex}"
    bt._degraded_plot_store(run_id, b"degraded-bytes")
    assert bt._degraded_plot_cached(run_id) == b"degraded-bytes"

    monkeypatch.setattr(bt, "_DEGRADED_PLOT_TTL_SECONDS", 0.0)
    assert bt._degraded_plot_cached(run_id) is None  # lapsed, so Yahoo is retried
    bt._clear_degraded_plot_cache()


def test_degraded_plot_negative_cache_is_bounded():
    """Both bounds are load-bearing on a public, unauthenticated route.

    The TTL caps how stale a degraded chart can get; the entry cap stops one
    outage window from pinning a PNG per requested run in memory.
    """
    assert 0 < bt._DEGRADED_PLOT_TTL_SECONDS <= 300
    bt._clear_degraded_plot_cache()
    for i in range(bt._DEGRADED_PLOT_MAX_ENTRIES + 25):
        bt._degraded_plot_store(f"run_bound_{i}", b"x")
    assert len(bt._degraded_plot_cache) <= bt._DEGRADED_PLOT_MAX_ENTRIES
    bt._clear_degraded_plot_cache()


def test_plot_png_survives_a_run_window_that_does_not_parse(monkeypatch, capsys):
    """#320, third path: a non-``YYYY-MM-DD`` window must not 500 either.

    ``_epoch`` raises ValueError *before* any HTTP call, so this never reached
    the transport-level guard. It is reachable in prod two ways: paper-trading
    baselines store ``start_date.isoformat()``, and
    ``api/routers/paper_trading.py`` writes ``end_date=""``.
    """
    bt._render_run_plot_png.cache_clear()
    bt._clear_degraded_plot_cache()
    run_id = f"run_badwindow_{uuid.uuid4().hex}"
    fake_run = {
        "session_id": None, "created_at": "2026-05-01T10:00:00", "agent_name": "Agent",
        "start_date": "2026-06-02T20:00:00", "end_date": "", "mode": "safe_trading",
        "baseline_buyhold_run_id": None, "baseline_djia_run_id": None,
    }
    et = pytz.timezone("US/Eastern")
    t0 = et.localize(datetime(2026, 5, 1, 10, 30)).astimezone(pytz.UTC)
    t1 = et.localize(datetime(2026, 5, 1, 11, 30)).astimezone(pytz.UTC)

    monkeypatch.setattr(bt.db, "get_run", lambda rid: fake_run if rid == run_id else None)
    monkeypatch.setattr(
        bt.db, "get_equity_curve",
        lambda rid: [{"timestamp": t0.replace(tzinfo=None).isoformat(), "equity": 100000},
                     {"timestamp": t1.replace(tzinfo=None).isoformat(), "equity": 101000}],
    )
    monkeypatch.setattr(bt, "filter_market_hours", lambda pts: pts)

    def must_not_be_called(*_a, **_k):  # pragma: no cover - asserts absence
        raise AssertionError("Yahoo must not be asked for an unparseable window")

    monkeypatch.setattr(
        "dashboard.backend.equity_plot.fetch_index_hourly", must_not_be_called
    )

    resp = TestClient(app).get(f"/runs/{run_id}/plot.png")
    assert resp.status_code == 200  # was an unhandled 500
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert "unusable run window" in capsys.readouterr().out
    bt._render_run_plot_png.cache_clear()
    bt._clear_degraded_plot_cache()


# ===========================================================================
# #2 — /backtest/run: cost-abuse hardening
# ===========================================================================

class _Spy:
    def __init__(self):
        self.calls = 0
        self.last_args = None
        self.last_kwargs = None

    def __call__(self, *a, **k):
        self.calls += 1
        self.last_args = a
        self.last_kwargs = k


class _ExecutionPreflightService:
    def __init__(self):
        self.execution_calls = []
        self.credential_calls = []

    def preflight_execution_model(self, provider_id, catalog_model_id):
        self.execution_calls.append((provider_id, catalog_model_id))
        if provider_id != "openai" or catalog_model_id != "openai/gpt-5.5":
            raise UnsupportedExecutionModel(catalog_model_id)
        return ExecutionModelRoute(
            catalog_id="openai/gpt-5.5",
            label="GPT-5.5",
            provider_model_id="gpt-5.5",
        )

    def preflight_user_default_credential(self, user_id, provider_id):
        self.credential_calls.append((user_id, provider_id))

    def preflight_platform_credential(self, provider_id):
        self.credential_calls.append((None, provider_id))


_ATL_EXECUTION_MODEL_IDS = (
    "anthropic/claude-haiku-4-5",
    "anthropic/claude-sonnet-4-6",
    "openai/gpt-5.5",
    "google/gemini-3.1-pro-preview",
    "deepseek/deepseek-v4-pro",
    "qwen/qwen3.7-plus",
)


class _OpenRouterExecutionPreflightService:
    def __init__(self):
        self.execution_calls = []
        self.credential_calls = []

    def preflight_execution_model(self, provider_id, catalog_model_id):
        self.execution_calls.append((provider_id, catalog_model_id))
        if (
            provider_id != "openrouter"
            or catalog_model_id not in _ATL_EXECUTION_MODEL_IDS
        ):
            raise UnsupportedExecutionModel(catalog_model_id)
        return ExecutionModelRoute(
            catalog_id=catalog_model_id,
            label=catalog_model_id,
            provider_model_id=catalog_model_id,
        )

    def preflight_user_default_credential(self, user_id, provider_id):
        self.credential_calls.append((user_id, provider_id))

    def preflight_platform_credential(self, provider_id):
        self.credential_calls.append((None, provider_id))


class _AutoPlatformExecutionPreflightService(_OpenRouterExecutionPreflightService):
    def resolve_platform_execution_candidates(
        self, catalog_model_id, preferred_provider_id=None
    ):
        assert catalog_model_id in _ATL_EXECUTION_MODEL_IDS
        return ("openrouter", "commonstack")

    def preflight_execution_model(self, provider_id, catalog_model_id):
        self.execution_calls.append((provider_id, catalog_model_id))
        if provider_id not in {"openrouter", "commonstack"}:
            raise UnsupportedExecutionModel(catalog_model_id)
        return ExecutionModelRoute(
            catalog_id=catalog_model_id,
            label=catalog_model_id,
            provider_model_id=catalog_model_id,
        )


def _enable_authenticated_openrouter_byok(monkeypatch):
    service = _OpenRouterExecutionPreflightService()
    monkeypatch.setattr(bt, "get_model_provider_service", lambda: service)
    monkeypatch.setattr(
        "dashboard.backend.api.dependencies._optional_user",
        lambda *_args, **_kwargs: {"id": 7},
    )
    return service


def _run_record(metadata=None):
    return {
        "run_id": "run_source",
        "agent_name": "Agent",
        "mode": "backtest",
        "start_date": "2026-04-01",
        "end_date": "2026-04-23",
        "initial_equity": 100_000,
        "num_trades": 1,
        "created_at": "2026-04-23T16:00:00",
        "metadata": metadata,
    }


def test_run_metadata_response_exposes_simulation_source():
    response = bt._run_metadata_response(
        _run_record({"data_source": "vnpy_simulation"})
    )

    assert response.data_source == "vnpy_simulation"


def test_run_metadata_response_defaults_legacy_runs_to_alpaca():
    assert bt._run_metadata_response(_run_record()).data_source == "alpaca"


def test_run_metadata_response_exposes_complete_ifind_profile():
    response = bt._run_metadata_response(
        _run_record(
            {
                "data_source": "ifind_ashare",
                "market": "CN",
                "universe": "a_share_demo_6",
                "timeframe": "60m",
                "timezone": "Asia/Shanghai",
                "decision_source": "rule_based",
                "benchmark": "equal_weight_buyhold",
                "symbols": ["600519.SH", "601318.SH"],
                "native_currency": "CNY",
                "reporting_currency": "USD",
                "native_initial_capital": 7_000,
                "fx_pair": "USD/CNY",
                "fx_source": "ifind_history_currency_conversion",
                "fx_policy": "daily_implied_median_forward_fill",
                "fx_start_rate": 7.0,
                "fx_end_rate": 7.1,
                "t_plus_one_enabled": True,
                "lot_size": 100,
                "transaction_cost_profile": {
                    "commission_rate": 0.00025,
                    "minimum_commission": 5.0,
                    "stamp_duty_sell_rate": 0.0005,
                    "transfer_fee_rate": 0.00001,
                    "buy_slippage_rate": 0.0005,
                    "sell_slippage_rate": 0.0005,
                    "price_tick": 0.01,
                },
                "transaction_cost_totals": {
                    "gross_value": 10005.0,
                    "slippage_amount": 5.0,
                    "commission": 5.0,
                    "stamp_duty": 0.0,
                    "transfer_fee": 0.1,
                    "total_fees": 5.1,
                },
                "market_rule_profile": {
                    "enabled": True,
                    "source": "ifind_http",
                    "version": "ifind-ashare-closing-rules-v1",
                },
                "market_rule_rejections": {
                    "suspended": 2,
                    "limit_up_buy_blocked": 1,
                },
                # The records themselves are deliberately NOT projected onto
                # RunMetadata (it backs two list routes); only the scalars are.
                "rejected_orders": [{
                    "symbol": "600519.SH",
                    "reason": "t1_frozen",
                    "status": "rejected",
                }],
                "rejected_orders_count": 7200,
                "rejected_orders_truncated": 7000,
                "order_events": [{"status": "rejected"}],
                "order_events_count": 7300,
                "order_events_truncated": 7100,
            }
        )
    )

    assert response.data_source == "ifind_ashare"
    assert response.market == "CN"
    assert response.universe == "a_share_demo_6"
    assert response.timeframe == "60m"
    assert response.timezone == "Asia/Shanghai"
    assert response.decision_source == "rule_based"
    assert response.benchmark == "equal_weight_buyhold"
    assert response.symbols == ["600519.SH", "601318.SH"]
    assert response.native_currency == "CNY"
    assert response.reporting_currency == "USD"
    assert response.native_initial_capital == 7_000
    assert response.fx_pair == "USD/CNY"
    assert response.fx_source == "ifind_history_currency_conversion"
    assert response.fx_start_rate == 7.0
    assert response.fx_end_rate == 7.1
    assert response.t_plus_one_enabled is True
    assert response.lot_size == 100
    assert response.transaction_cost_profile["minimum_commission"] == 5.0
    assert response.transaction_cost_totals["total_fees"] == 5.1
    assert response.market_rule_profile["enabled"] is True
    assert response.market_rule_rejections["suspended"] == 2
    assert response.rejected_orders_count == 7200
    assert response.rejected_orders_truncated == 7000
    assert response.order_events_count == 7300
    assert response.order_events_truncated == 7100
    # The unbounded array must never reach a list-route payload.
    assert not hasattr(response, "rejected_orders")
    assert "rejected_orders" not in response.model_dump()
    assert "order_events" not in response.model_dump()


def test_run_metadata_response_keeps_new_fields_optional_for_legacy_runs():
    response = bt._run_metadata_response(_run_record())

    assert response.market is None
    assert response.universe is None
    assert response.timeframe is None
    assert response.timezone is None
    assert response.decision_source is None
    assert response.benchmark is None
    assert response.symbols is None
    assert response.native_currency is None
    assert response.reporting_currency is None
    assert response.native_initial_capital is None
    assert response.fx_pair is None
    assert response.fx_source is None
    assert response.fx_start_rate is None
    assert response.t_plus_one_enabled is None
    assert response.lot_size is None
    assert response.transaction_cost_profile is None
    assert response.transaction_cost_totals is None
    assert response.market_rule_profile is None
    assert response.market_rule_rejections is None
    # None, not 0: a legacy run predates the feature, it did not record zero
    # rejections. Same convention as t_plus_one_enabled above.
    assert response.rejected_orders_count is None
    assert response.rejected_orders_truncated is None
    assert response.order_events_count is None
    assert response.order_events_truncated is None
    assert response.llm_execution is None
    assert response.frequency_contract is None
    assert response.market_data_quality is None
    assert response.market_data_feed is None
    assert response.sip_fallback_to_iex is None
    assert response.end_clamped is None


def test_run_metadata_response_exposes_minute_data_contract_and_quality():
    response = bt._run_metadata_response(
        _run_record(
            {
                "data_source": "alpaca",
                "frequency_contract": {
                    "source_timeframe": "5m",
                    "decision_timeframe": "60m",
                    "decision_frequency": "1h",
                    "execution_timeframe": "5m",
                    "valuation_frequency": "5m",
                    "aggregation": "session_anchored_completed_bars",
                    "fill_policy": "next_source_bar_open",
                    "verification_status": "verified",
                },
                "market_data_quality": {
                    "policy": "drop_incomplete_decision_bars",
                    "decision_timestamp_min_symbol_coverage": 0.8,
                    "total_decision_bars": 210,
                    "usable_decision_bars": 207,
                    "dropped_decision_bars": 3,
                    "missing_source_bars": 2,
                    "duplicate_source_bars": 1,
                    "off_grid_source_bars": 0,
                    "invalid_source_bars": 0,
                    "symbols": {"AAPL": {"dropped_decision_bars": 3}},
                },
                "market_data_feed": "iex",
                "sip_fallback_to_iex": True,
                "end_clamped": True,
            }
        )
    )

    assert response.frequency_contract["source_timeframe"] == "5m"
    assert response.frequency_contract["decision_frequency"] == "1h"
    assert response.frequency_contract["fill_policy"] == "next_source_bar_open"
    assert response.frequency_contract["verification_status"] == "verified"
    assert response.market_data_quality["usable_decision_bars"] == 207
    assert response.market_data_quality["dropped_decision_bars"] == 3
    assert "symbols" not in response.market_data_quality
    assert response.market_data_feed == "iex"
    assert response.sip_fallback_to_iex is True
    assert response.end_clamped is True


def test_run_metadata_response_exposes_sanitized_llm_execution_evidence():
    response = bt._run_metadata_response(
        _run_record({
            "data_source": "alpaca",
            "llm_execution": {
                "billing_mode": "platform_credits",
                "requested_provider_id": "openrouter",
                "provider_id": "mixed",
                "provider_ids": ["openrouter", "commonstack"],
                "provider_mixed": True,
                "model_id": "openai/gpt-5.5",
                "credential_id": None,
                "credential_key_last_four": None,
                "call_count": 2,
                "input_tokens": 30,
                "output_tokens": 13,
                "usage_available": True,
                "provider_cost_usd": 0.03,
                "estimated_cost_usd": 0.028,
                "pricing_snapshot": None,
                "debited_credits_micro": 24_000,
                "outstanding_credits_micro": 5_000,
                "outcome": "settled_overage",
                "provider_api_key": "raw-secret-must-not-leak",
            },
        })
    )

    serialized = response.model_dump(mode="json")
    assert serialized["llm_execution"]["outcome"] == "settled_overage"
    assert serialized["llm_execution"]["credential_key_last_four"] is None
    assert serialized["llm_execution"]["requested_provider_id"] == "openrouter"
    assert serialized["llm_execution"]["provider_id"] == "mixed"
    assert serialized["llm_execution"]["provider_ids"] == [
        "openrouter",
        "commonstack",
    ]
    assert serialized["llm_execution"]["provider_mixed"] is True
    assert "raw-secret-must-not-leak" not in str(serialized)
    assert "provider_api_key" not in serialized["llm_execution"]


@pytest.fixture(autouse=True)
def _reset_backtest_guards(monkeypatch):
    bt._backtest_rate_limiter.reset()
    bt.backtest_status.update({
        "running": False,
        "error": None,
        "runs_count": 0,
        "started_at": None,
        "progress_file": None,
        "live_run_id": None,
    })
    # Safety net: no test in this file may launch a real backtest thread.
    monkeypatch.setattr(bt, "run_backtest_background", lambda *a, **k: None)
    yield
    bt._backtest_rate_limiter.reset()


def _sess():
    return {"X-Session-Id": str(uuid.uuid4())}


def test_backtest_run_valid_request_ok():
    resp = TestClient(app).post(
        "/backtest/run",
        json={
            "start_date": "2026-05-01",
            "end_date": "2026-05-07",
            "decision_source": "rule_based",
        },
        headers=_sess(),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert "session_id" in body


def test_backtest_run_targets_builtin_agent_session(client, monkeypatch):
    """Discord (and website) can pass agent_id so runs land on the agent card."""
    spy = _Spy()
    monkeypatch.setattr(bt, "run_backtest_background", spy)

    owner = str(uuid.uuid4())
    created = client.post(
        "/api/v1/agents",
        json={"name": "Discord Card Bot", "agent_type": "builtin"},
        headers={"X-Session-Id": owner},
    ).json()
    agent_session = created["session_id"]
    agent_id = created["agent"]["agent_id"]

    resp = client.post(
        "/backtest/run",
        json={
            "start_date": "2026-05-01",
            "end_date": "2026-05-02",
            "agent_id": agent_id,
            "decision_source": "rule_based",
        },
        headers={"X-Session-Id": str(uuid.uuid4())},
    )
    assert resp.status_code == 200
    assert resp.json()["session_id"] == agent_session
    assert spy.calls == 1
    assert spy.last_kwargs["session_id"] == agent_session
    assert spy.last_kwargs["runtime_type"] == "pipeline"
    assert spy.last_kwargs["runtime_config"] == {}


def _stub_hosted_runtime_installed(monkeypatch):
    """Pretend the isolated upstream venv exists on this deployment.

    CI installs core requirements only, so the real check always reports the
    runtime as missing. Tests about *other* preconditions have to say which
    deployment they are describing.
    """
    monkeypatch.setattr(bt, "runtime_unavailable_reason", lambda: None)


def test_backtest_run_dispatches_ai_hedge_fund_runtime(client, monkeypatch):
    spy = _Spy()
    monkeypatch.setattr(bt, "run_backtest_background", spy)
    monkeypatch.setenv("OPENROUTER_API_KEY", "platform-openrouter-test-key")
    _stub_hosted_runtime_installed(monkeypatch)
    monkeypatch.setattr(
        bt.agent_credential_store,
        "get_secret",
        lambda agent_id, credential_name: "user-financial-datasets-test-key",
    )
    owner = str(uuid.uuid4())
    headers = {"X-Session-Id": owner}
    cloned = client.post(
        "/api/v1/agents/marketplace/ai-hedge-fund/clone",
        json={},
        headers=headers,
    )
    assert cloned.status_code == 200
    agent = cloned.json()["agent"]

    response = client.post(
        "/backtest/run",
        json={
            "start_date": "2026-05-01",
            "end_date": "2026-05-02",
            "decision_source": "llm",
            "model": "claude-haiku-4.5",
            "pipeline": [{"label": "must be ignored"}],
            "agent_id": agent["agent_id"],
        },
        headers=headers,
    )

    assert response.status_code == 200, response.text
    assert response.json()["runtime_type"] == "ai_hedge_fund"
    assert set(response.json()["ignored_fields"]) == {"model", "pipeline"}
    assert spy.calls == 1
    assert spy.last_kwargs["runtime_type"] == "ai_hedge_fund"
    assert spy.last_kwargs["runtime_config"]["analysts"]
    assert (
        spy.last_kwargs["financial_datasets_api_key"]
        == "user-financial-datasets-test-key"
    )
    assert spy.last_kwargs["model"] is None
    assert spy.last_kwargs["pipeline"] is None


def test_ai_hedge_fund_backtest_requires_owned_agent_credential(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "platform-openrouter-test-key")
    _stub_hosted_runtime_installed(monkeypatch)
    owner = str(uuid.uuid4())
    headers = {"X-Session-Id": owner}
    agent = client.post(
        "/api/v1/agents/marketplace/ai-hedge-fund/clone",
        json={},
        headers=headers,
    ).json()["agent"]
    monkeypatch.setattr(
        bt.agent_credential_store,
        "get_secret",
        lambda agent_id, credential_name: None,
    )

    missing = client.post(
        "/backtest/run",
        json={
            "start_date": "2026-05-01",
            "end_date": "2026-05-02",
            "agent_id": agent["agent_id"],
        },
        headers=headers,
    )
    assert missing.status_code == 422
    assert "Financial Datasets API key" in missing.text

    unauthorized = client.post(
        "/backtest/run",
        json={
            "start_date": "2026-05-01",
            "end_date": "2026-05-02",
            "agent_id": agent["agent_id"],
        },
        headers={"X-Session-Id": str(uuid.uuid4())},
    )
    assert unauthorized.status_code == 403


def test_ai_hedge_fund_backtest_rejects_run_when_runtime_not_installed(
    client, monkeypatch
):
    """A missing isolated venv must fail at request time, not 30 minutes later.

    render.yaml is documentation rather than the deploy mechanism here, so a
    service without the venv is a live possibility. Without this the run is
    accepted, backgrounded, and dies on its first decision step.
    """
    spy = _Spy()
    monkeypatch.setattr(bt, "run_backtest_background", spy)
    monkeypatch.setenv("OPENROUTER_API_KEY", "platform-openrouter-test-key")
    monkeypatch.setattr(
        bt,
        "runtime_unavailable_reason",
        lambda: "AI Hedge Fund runtime is not installed; configure AI_HEDGE_FUND_PYTHON",
    )
    monkeypatch.setattr(
        bt.agent_credential_store,
        "get_secret",
        lambda agent_id, credential_name: "user-financial-datasets-test-key",
    )
    headers = {"X-Session-Id": str(uuid.uuid4())}
    agent = client.post(
        "/api/v1/agents/marketplace/ai-hedge-fund/clone",
        json={},
        headers=headers,
    ).json()["agent"]

    response = client.post(
        "/backtest/run",
        json={
            "start_date": "2026-05-01",
            "end_date": "2026-05-02",
            "agent_id": agent["agent_id"],
        },
        headers=headers,
    )

    assert response.status_code == 503, response.text
    assert "AI_HEDGE_FUND_PYTHON" in response.text
    assert spy.calls == 0


def test_hosted_backtest_timeout_covers_every_decision_step():
    """The parent timeout must not be the binding constraint on a hosted run.

    A fixed 3600s cap gives a month-long run enough room for its decision steps
    while keeping a bounded parent process lifetime.
    """
    step_seconds = bt.resolve_step_timeout_seconds()
    decision_days = bt._estimated_decision_days("2026-01-01", "2026-01-31")
    assert decision_days == 22

    hosted = bt._backtest_subprocess_timeout(
        "ai_hedge_fund", "2026-01-01", "2026-01-31"
    )
    assert hosted >= step_seconds * decision_days
    assert hosted > bt.PIPELINE_SUBPROCESS_TIMEOUT_SECONDS

    # Pipeline runs use the shared 60-minute budget exactly.
    assert bt.PIPELINE_SUBPROCESS_TIMEOUT_SECONDS == 3600
    assert (
        bt._backtest_subprocess_timeout("pipeline", "2026-01-01", "2026-01-31")
        == 3600
    )


def test_hosted_backtest_timeout_is_capped_and_never_below_pipeline():
    """A long range is capped rather than pinning a worker thread forever."""
    capped = bt._backtest_subprocess_timeout(
        "ai_hedge_fund", "2020-01-01", "2030-01-01"
    )
    assert capped == bt.MAX_SUBPROCESS_TIMEOUT_SECONDS

    # Unparseable dates must not collapse the budget to the overhead constant.
    assert (
        bt._backtest_subprocess_timeout("ai_hedge_fund", "not-a-date", "also-bad")
        == bt.PIPELINE_SUBPROCESS_TIMEOUT_SECONDS
    )


def test_pipeline_timeout_finalizes_execution_and_slot_once(monkeypatch):
    """A parent timeout keeps the existing single-owner cleanup path.

    The ``Popen`` rewrite that made cancel possible (issue #273) is the easiest
    place to lose this: ``subprocess.run(timeout=...)`` enforced the budget for
    free and ``Popen`` does not, so the wall-clock bound now lives in an
    explicit ``wait(timeout=...)``. The parent must still stop the child at
    ``_backtest_subprocess_timeout``'s number, still surface it as a
    ``TimeoutExpired``, and still finalize the slot and the execution
    reservation exactly once each.
    """
    run_id = "agent_timeout_cleanup"
    session_id = str(uuid.uuid4())
    assert (
        bt._try_acquire_backtest_slot(
            live_run_id=run_id,
            session_id=session_id,
            user_id=None,
        )
        is None
    )

    popen_calls = []
    finalized_slots = []
    finalized_execution_runs = []
    child = FakeChild(stdout="run header line\n", timeout_waits=1)

    def fake_popen(cmd, **kwargs):
        popen_calls.append({"cmd": cmd, **kwargs})
        return child

    class FakeExecutionService:
        def __init__(self, **_kwargs):
            pass

        def finalize_run(self, execution_run_id, *, billing_mode=None):
            finalized_execution_runs.append(execution_run_id)

    real_finalize_slot = bt._finalize_slot

    def spy_finalize_slot(
        live_run_id,
        *,
        error,
        runs_count,
        cancelled=False,
        timed_out=False,
        timeout_detail=None,
    ):
        finalized_slots.append((live_run_id, runs_count))
        return real_finalize_slot(
            live_run_id,
            error=error,
            runs_count=runs_count,
            cancelled=cancelled,
            timed_out=timed_out,
            timeout_detail=timeout_detail,
        )

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(bt, "LLMExecutionService", FakeExecutionService)
    monkeypatch.setattr(bt, "get_model_provider_service", lambda: object())
    monkeypatch.setattr(bt, "_finalize_slot", spy_finalize_slot)
    monkeypatch.setattr(bt, "run_backtest_background", _REAL_RUN_BACKTEST_BACKGROUND)

    bt.run_backtest_background(
        start_date="2026-01-01",
        end_date="2026-01-02",
        session_id=session_id,
        live_run_id=run_id,
        decision_source="rule_based",
        execution_handoff_payload="opaque-test-handoff",
    )

    assert len(popen_calls) == 1
    # The budget moved from a `run(timeout=)` kwarg to the parent's own wait.
    assert child.wait_timeouts[0] == 3600
    # Timed out => terminated. The child died inside the grace period, so no
    # SIGKILL was needed.
    assert (child.terminated, child.killed) == (1, 0)
    assert finalized_execution_runs == [run_id]
    assert run_id not in bt._active_slots
    assert bt._recent_slots[run_id]["running"] is False
    assert bt._recent_slots[run_id]["cancelled"] is False
    # The handle is dropped with the run; _recent_slots retains fifty of these.
    assert bt._recent_slots[run_id]["process"] is None
    # The outcome, not just the cleanup. Before this the TimeoutExpired fell
    # into the generic `except Exception` arm and the user received a tail
    # slice of the child's argv -- `_sanitize_backtest_error` truncates from
    # the END, which is right for a stack trace and wrong for a TimeoutExpired
    # whose informative clause trails a long command line.
    assert bt._recent_slots[run_id]["timed_out"] is True
    assert bt._recent_slots[run_id]["error"] is None
    # Exactly once. A second _finalize_slot from the `finally` would overwrite
    # the timeout with a zero-run completion.
    assert finalized_slots == [(run_id, 0)]


def _drive_pipeline_timeout(
    monkeypatch, *, run_id, billing_mode, user_id=None, spend_lookup=None
):
    """Run the real worker to a parent timeout and return the finalized slot.

    Shared by the two lane cases below rather than copied: the setup is eight
    monkeypatches and a FakeChild, and a copy that drifts in one of them stops
    testing the branch it names.
    """
    session_id = str(uuid.uuid4())
    assert (
        bt._try_acquire_backtest_slot(
            live_run_id=run_id, session_id=session_id, user_id=user_id
        )
        is None
    )

    child = FakeChild(stdout="run header line\n", timeout_waits=1)

    class FakeExecutionService:
        def __init__(self, **_kwargs):
            pass

        def finalize_run(self, execution_run_id, *, billing_mode=None):
            return []

    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: child)
    monkeypatch.setattr(bt, "LLMExecutionService", FakeExecutionService)
    monkeypatch.setattr(bt, "get_model_provider_service", object)
    monkeypatch.setattr(bt, "run_backtest_background", _REAL_RUN_BACKTEST_BACKGROUND)
    monkeypatch.setattr(
        bt.credits_service,
        "sum_run_llm_spend",
        spend_lookup or (lambda uid, rid: (42_318, 2)),
    )

    bt.run_backtest_background(
        start_date="2026-01-01",
        end_date="2026-01-02",
        session_id=session_id,
        live_run_id=run_id,
        decision_source="rule_based",
        execution_handoff_payload="opaque-test-handoff",
        billing_mode=billing_mode,
    )
    return bt._recent_slots[run_id]


def test_platform_credits_timeout_discloses_what_the_run_cost(monkeypatch):
    """The whole point: the user waited an hour, got nothing, and was billed.

    `finalize_run` in the `finally` only RELEASES open holds -- it never settles
    -- so every settled row was already written by the child as it went, and the
    sum read here is final.
    """
    slot = _drive_pipeline_timeout(
        monkeypatch,
        run_id="agent_timeout_platform",
        billing_mode="platform_credits",
        user_id=4242,
    )

    assert slot["timed_out"] is True
    assert slot["timeout_detail"] == {
        "limit_seconds": 3600,
        "billing_mode": "platform_credits",
        "spent_micro": 42_318,
        "model_calls": 2,
    }


def test_byok_timeout_reports_no_spend_rather_than_zero(monkeypatch):
    """None, not 0. BYOK never touches the ATL ledger, so a zero here would be
    indistinguishable from "platform lane, nothing settled yet" -- and the card
    would tell a BYOK user their run cost 0.000000 Credits, which is a claim
    about a ledger that has no row for them at all."""
    slot = _drive_pipeline_timeout(
        monkeypatch,
        run_id="agent_timeout_byok",
        billing_mode="byok",
        user_id=4242,
    )

    assert slot["timed_out"] is True
    assert slot["timeout_detail"] == {
        "limit_seconds": 3600,
        "billing_mode": "byok",
        "spent_micro": None,
        "model_calls": None,
    }


def test_ai_hedge_fund_requires_openrouter_not_direct_openai(client, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-authorize-hosted-runtime")
    owner = str(uuid.uuid4())
    headers = {"X-Session-Id": owner}
    agent = client.post(
        "/api/v1/agents/marketplace/ai-hedge-fund/clone",
        json={},
        headers=headers,
    ).json()["agent"]

    response = client.post(
        "/backtest/run",
        json={
            "start_date": "2026-05-01",
            "end_date": "2026-05-02",
            "agent_id": agent["agent_id"],
        },
        headers=headers,
    )

    assert response.status_code == 503
    assert "platform-managed OpenRouter provider" in response.text


def test_backtest_run_forwards_selected_assets(client, monkeypatch):
    """UI Asset Universe must reach the background worker (not stay mocked/DJIA-only)."""
    spy = _Spy()
    monkeypatch.setattr(bt, "run_backtest_background", spy)

    mag7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"]
    resp = client.post(
        "/backtest/run",
        json={
            "start_date": "2026-05-01",
            "end_date": "2026-05-02",
            "assets": mag7,
            "decision_source": "rule_based",
        },
        headers=_sess(),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["assets"] == mag7
    assert spy.calls == 1
    # By name, not position: universe/timeframe were inserted mid-signature
    # once, and an index-based assertion would have kept passing on the wrong
    # argument.
    assert spy.last_args == ()
    assert spy.last_kwargs["assets"] == mag7
    assert spy.last_kwargs["decision_source"] == "rule_based"


def test_backtest_run_rejects_bad_assets(monkeypatch):
    spy = _Spy()
    monkeypatch.setattr(bt, "run_backtest_background", spy)
    resp = TestClient(app).post(
        "/backtest/run",
        json={
            "start_date": "2026-05-01",
            "end_date": "2026-05-02",
            "assets": ["NOT A TICKER!!!"],
        },
        headers=_sess(),
    )
    assert resp.status_code == 422
    assert spy.calls == 0


def test_backtest_run_rejects_external_agent_id(client):
    owner = str(uuid.uuid4())
    created = client.post(
        "/api/v1/agents",
        json={"name": "External Only", "agent_type": "external"},
        headers={"X-Session-Id": owner},
    ).json()
    agent_id = created["agent"]["agent_id"]

    resp = client.post(
        "/backtest/run",
        json={"start_date": "2026-05-01", "end_date": "2026-05-02", "agent_id": agent_id},
        headers=_sess(),
    )
    assert resp.status_code == 422


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.parametrize("model", _ATL_EXECUTION_MODEL_IDS)
def test_openrouter_byok_accepts_every_catalog_model(monkeypatch, model):
    service = _enable_authenticated_openrouter_byok(monkeypatch)

    resp = TestClient(app).post(
        "/backtest/run",
        json={
            "start_date": "2026-05-01",
            "end_date": "2026-05-02",
            "decision_source": "llm",
            "billing_mode": "byok",
            "provider_id": "openrouter",
            "model": model,
        },
        headers=_sess(),
    )

    assert resp.status_code == 200, (model, resp.text)
    assert service.execution_calls == [("openrouter", model)]
    assert service.credential_calls == [(7, "openrouter")]


@pytest.mark.parametrize("model", _ATL_EXECUTION_MODEL_IDS)
def test_ifind_llm_accepts_every_openrouter_catalog_model(monkeypatch, model):
    monkeypatch.setenv("ENABLE_IFIND_ASHARE", "true")
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", "test-token-not-a-secret")
    service = _enable_authenticated_openrouter_byok(monkeypatch)
    monkeypatch.setattr(
        bt,
        "ensure_llm_client_available",
        object,
        raising=False,
    )

    resp = TestClient(app).post(
        "/backtest/run",
        json={
            "start_date": "2026-05-01",
            "end_date": "2026-05-02",
            "data_source": "ifind_ashare",
            "universe": "a_share_demo_6",
            "timeframe": "60m",
            "decision_source": "llm",
            "billing_mode": "byok",
            "provider_id": "openrouter",
            "model": model,
        },
        headers=_sess(),
    )

    assert resp.status_code == 200, (model, resp.text)
    assert resp.json()["decision_source"] == "llm"
    assert service.execution_calls == [("openrouter", model)]
    assert service.credential_calls == [(7, "openrouter")]


def test_explicit_llm_requires_model_before_scheduling(monkeypatch):
    monkeypatch.setenv("ENABLE_IFIND_ASHARE", "true")
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", "test-token-not-a-secret")
    spy = _Spy()
    monkeypatch.setattr(bt, "run_backtest_background", spy)
    monkeypatch.setattr(
        bt,
        "ensure_llm_client_available",
        object,
        raising=False,
    )

    resp = TestClient(app).post(
        "/backtest/run",
        json={
            "data_source": "ifind_ashare",
            "universe": "a_share_demo_6",
            "timeframe": "60m",
            "decision_source": "llm",
        },
        headers=_sess(),
    )

    assert resp.status_code == 422
    assert "model" in resp.text.lower()
    assert spy.calls == 0


def test_openai_byok_accepts_gpt_catalog_id_and_keeps_it_in_handoff(monkeypatch):
    spy = _Spy()
    service = _ExecutionPreflightService()
    captured_handoff = {}
    monkeypatch.setattr(bt, "run_backtest_background", spy)
    monkeypatch.setattr(bt, "get_model_provider_service", lambda: service)
    monkeypatch.setattr(
        "dashboard.backend.api.dependencies._optional_user",
        lambda *_args, **_kwargs: {"id": 7},
    )

    def capture_handoff(**kwargs):
        captured_handoff.update(kwargs)
        return "signed-test-handoff"

    monkeypatch.setattr(bt, "create_execution_handoff", capture_handoff)

    response = TestClient(app).post(
        "/backtest/run",
        json={
            "start_date": "2026-05-01",
            "end_date": "2026-05-02",
            "decision_source": "llm",
            "billing_mode": "byok",
            "provider_id": "openai",
            "model": "openai/gpt-5.5",
        },
        headers=_sess(),
    )

    assert response.status_code == 200, response.text
    assert captured_handoff["model_id"] == "openai/gpt-5.5"
    assert service.execution_calls == [("openai", "openai/gpt-5.5")]
    assert service.credential_calls == [(7, "openai")]
    assert spy.calls == 1


def test_platform_credits_resolves_candidates_without_provider_input(monkeypatch):
    spy = _Spy()
    service = _AutoPlatformExecutionPreflightService()
    captured_handoff = {}
    monkeypatch.setattr(bt, "run_backtest_background", spy)
    monkeypatch.setattr(bt, "get_model_provider_service", lambda: service)
    monkeypatch.setattr(
        "dashboard.backend.api.dependencies._optional_user",
        lambda *_args, **_kwargs: {"id": 7},
    )

    def capture_handoff(**kwargs):
        captured_handoff.update(kwargs)
        return "signed-platform-handoff"

    monkeypatch.setattr(bt, "create_execution_handoff", capture_handoff)

    response = TestClient(app).post(
        "/backtest/run",
        json={
            "start_date": "2026-05-01",
            "end_date": "2026-05-02",
            "decision_source": "llm",
            "billing_mode": "platform_credits",
            "model": "qwen/qwen3.7-plus",
        },
        headers=_sess(),
    )

    assert response.status_code == 200, response.text
    assert response.json()["provider_id"] == "openrouter"
    assert captured_handoff["provider_id"] == "openrouter"
    assert captured_handoff["provider_ids"] == ("openrouter", "commonstack")
    assert service.execution_calls == [("openrouter", "qwen/qwen3.7-plus")]
    assert service.credential_calls == []
    assert spy.calls == 1
    # The lane reaches the worker, not just the handoff envelope. This is the
    # only route-level wire in front of the timeout arm's spend disclosure, and
    # its failure is silent: drop the kwarg and every production timeout reports
    # `billing_mode: "byok", spent_micro: None` with the whole suite still green,
    # because the timeout tests call the worker directly with an explicit lane.
    assert spy.last_kwargs["billing_mode"] == "platform_credits"


def test_openai_byok_rejects_claude_before_worker_start(monkeypatch):
    spy = _Spy()
    service = _ExecutionPreflightService()
    monkeypatch.setattr(bt, "run_backtest_background", spy)
    monkeypatch.setattr(bt, "get_model_provider_service", lambda: service)
    monkeypatch.setattr(
        "dashboard.backend.api.dependencies._optional_user",
        lambda *_args, **_kwargs: {"id": 7},
    )

    response = TestClient(app).post(
        "/backtest/run",
        json={
            "start_date": "2026-05-01",
            "end_date": "2026-05-02",
            "decision_source": "llm",
            "billing_mode": "byok",
            "provider_id": "openai",
            "model": "anthropic/claude-sonnet-4-6",
        },
        headers=_sess(),
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "The selected model is not available from this provider."
    }
    assert service.execution_calls == [
        ("openai", "anthropic/claude-sonnet-4-6")
    ]
    assert service.credential_calls == []
    assert spy.calls == 0


@pytest.mark.parametrize("model", [
    "bad model with spaces", "x; rm -rf /", "a" * 100, "m\nnewline", "café",
])
def test_backtest_run_rejects_malformed_model(monkeypatch, model):
    spy = _Spy()
    monkeypatch.setattr(bt, "run_backtest_background", spy)
    resp = TestClient(app).post(
        "/backtest/run",
        json={"start_date": "2026-05-01", "end_date": "2026-05-02", "model": model},
        headers=_sess(),
    )
    assert resp.status_code == 422, (model, resp.text)
    assert spy.calls == 0  # nothing scheduled


def test_backtest_run_rejects_oversized_prompt(monkeypatch):
    spy = _Spy()
    monkeypatch.setattr(bt, "run_backtest_background", spy)
    resp = TestClient(app).post(
        "/backtest/run",
        json={"start_date": "2026-05-01", "end_date": "2026-05-02",
              "strategy_prompt": "x" * 5000},
        headers=_sess(),
    )
    assert resp.status_code == 422
    assert spy.calls == 0


def test_backtest_run_rejects_excessive_date_range(monkeypatch):
    spy = _Spy()
    monkeypatch.setattr(bt, "run_backtest_background", spy)
    resp = TestClient(app).post(
        "/backtest/run",
        json={"start_date": "2020-01-01", "end_date": "2026-01-01"},
        headers=_sess(),
    )
    assert resp.status_code == 422
    assert spy.calls == 0


def test_backtest_run_rejects_bad_date_format():
    resp = TestClient(app).post(
        "/backtest/run",
        json={"start_date": "05/01/2026", "end_date": "2026-05-02"},
        headers=_sess(),
    )
    assert resp.status_code == 422


def test_backtest_status_includes_live_progress(tmp_path):
    progress_file = tmp_path / "progress.json"
    progress_file.write_text(json.dumps({
        "run_id": "agent_test",
        "step": 5,
        "total_steps": 100,
        "equity_curve": [{"timestamp": "2026-05-01T10:00:00", "equity": 100500, "cash": 50000, "positions_value": 50500}],
        "trades": [{
            "timestamp": "2026-05-01T10:00:00",
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10,
            "price": 150.25,
            "value": 1502.5,
        }],
        "rejected_orders": [{
            "symbol": "600519.SH",
            "reason": "t1_frozen",
            "status": "rejected",
        }],
    }), encoding="utf-8")
    bt.backtest_status.update({
        "running": True,
        "error": None,
        "started_at": time.time(),
        "progress_file": str(progress_file),
        "live_run_id": "agent_test",
    })
    resp = TestClient(app).get("/backtest/status", headers=_sess())
    assert resp.status_code == 200
    body = resp.json()
    assert body["running"] is True
    assert body["progress"]["step"] == 5
    assert body["progress"]["total_steps"] == 100
    assert len(body["progress"]["equity_curve"]) == 1
    assert len(body["progress"]["trades"]) == 1
    assert body["progress"]["trades"][0]["symbol"] == "AAPL"
    assert body["progress"]["rejected_orders"][0]["reason"] == "t1_frozen"
    assert "step 5/100" in body["message"]
    # Pinned at the wire, not just at _read_backtest_progress: the payload is
    # assigned wholesale today, so a later whitelist would drop these fields with
    # a green helper test and a UI that silently stops reporting staleness.
    assert body["progress"]["progress_updated_at"] == progress_file.stat().st_mtime
    # The age is the field the browser actually reads -- deriving it client-side
    # from the mtime above would make a skewed clock look like a wedged run.
    assert 0 <= body["progress"]["progress_age_seconds"] < 30


def test_get_run_trades_endpoint(client, monkeypatch):
    session_id = str(uuid.uuid4())
    run_id = "agent_test_trades"

    def fake_get_run_with_session(rid, sid):
        if rid == run_id and sid == session_id:
            return {
                "run_id": run_id,
                "agent_name": "Agent",
                "mode": "backtest",
                "metadata": {
                    "order_events": [{
                        "symbol": "600519.SH",
                        "status": "rejected",
                        "reason": "insufficient_cash_for_lot",
                    }],
                    "order_events_count": 201,
                    "order_events_truncated": 200,
                },
            }
        return None

    def fake_get_trades(rid):
        if rid == run_id:
            return [{
                "timestamp": "2026-05-01T10:00:00",
                "symbol": "MSFT",
                "quantity": 5,
                "side": "BUY",
                "price": 380.5,
                "value": 1902.5,
            }]
        return []

    monkeypatch.setattr(bt.db, "get_run_with_session", fake_get_run_with_session)
    monkeypatch.setattr(bt.db, "get_trades", fake_get_trades)

    resp = client.get(f"/runs/{run_id}/trades", headers={"X-Session-Id": session_id})
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == run_id
    assert body["count"] == 1
    assert body["trades"][0]["symbol"] == "MSFT"
    assert body["order_events"][0]["status"] == "rejected"
    assert body["order_event_count"] == 201
    assert body["order_events_returned"] == 1
    assert body["order_events_truncated"] == 200


def test_get_run_trades_tolerates_legacy_order_event_metadata(client, monkeypatch):
    session_id = str(uuid.uuid4())
    run_id = "agent_legacy_order_events"

    monkeypatch.setattr(
        bt.db,
        "get_run_with_session",
        lambda rid, sid: {"run_id": rid, "metadata": None}
        if rid == run_id and sid == session_id
        else None,
    )
    monkeypatch.setattr(bt.db, "get_trades", lambda _rid: [])

    body = client.get(
        f"/runs/{run_id}/trades", headers={"X-Session-Id": session_id}
    ).json()

    assert body["trades"] == []
    assert body["count"] == 0
    assert body["order_events"] == []
    assert body["order_event_count"] == 0
    assert body["order_events_truncated"] == 0


def _rejected_orders_run(monkeypatch, run_id, session_id, metadata):
    def fake_get_run_with_session(rid, sid):
        if rid == run_id and sid == session_id:
            return {
                "run_id": run_id,
                "agent_name": "Agent",
                "mode": "backtest",
                "metadata": metadata,
            }
        return None

    monkeypatch.setattr(bt.db, "get_run_with_session", fake_get_run_with_session)


def test_get_run_rejected_orders_endpoint(client, monkeypatch):
    session_id = str(uuid.uuid4())
    run_id = "agent_test_rejections"
    _rejected_orders_run(monkeypatch, run_id, session_id, {
        "rejected_orders": [{"symbol": "600519.SH", "reason": "t1_frozen"}],
        "rejected_orders_count": 1,
    })

    resp = client.get(
        f"/runs/{run_id}/rejected-orders", headers={"X-Session-Id": session_id}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == run_id
    assert body["count"] == 1
    assert body["returned"] == 1
    assert body["truncated"] == 0
    assert body["rejected_orders"][0]["reason"] == "t1_frozen"


def test_get_run_rejected_orders_reports_the_cap_it_applied(client, monkeypatch):
    """A capped sample must never read as the run's full rejection history."""
    session_id = str(uuid.uuid4())
    run_id = "agent_test_rejections_capped"
    _rejected_orders_run(monkeypatch, run_id, session_id, {
        "rejected_orders": [{"symbol": "600519.SH", "reason": "t1_frozen"}] * 200,
        "rejected_orders_count": 7_200,
        "rejected_orders_truncated": 7_000,
    })

    body = client.get(
        f"/runs/{run_id}/rejected-orders", headers={"X-Session-Id": session_id}
    ).json()
    assert body["count"] == 7_200      # what the run actually produced
    assert body["returned"] == 200     # what this response carries
    assert body["truncated"] == 7_000  # the difference, stated outright


def test_get_run_rejected_orders_is_session_scoped(client, monkeypatch):
    session_id = str(uuid.uuid4())
    _rejected_orders_run(monkeypatch, "agent_owned", session_id, {})

    resp = client.get(
        "/runs/agent_owned/rejected-orders",
        headers={"X-Session-Id": str(uuid.uuid4())},
    )
    assert resp.status_code == 404


def test_get_run_rejected_orders_tolerates_legacy_runs(client, monkeypatch):
    """Runs predating the feature have no metadata dict at all."""
    session_id = str(uuid.uuid4())
    _rejected_orders_run(monkeypatch, "agent_legacy", session_id, None)

    body = client.get(
        "/runs/agent_legacy/rejected-orders", headers={"X-Session-Id": session_id}
    ).json()
    assert body["rejected_orders"] == []
    assert body["count"] == 0
    assert body["truncated"] == 0


def test_backtest_run_rate_limited_per_client(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(
        bt, "_backtest_rate_limiter",
        FixedWindowRateLimiter(max_events=2, window_seconds=3600, clock=lambda: now[0]),
    )
    client = TestClient(app)
    headers = _sess()  # same session -> same rate key across the three calls
    body = {
        "start_date": "2026-05-01",
        "end_date": "2026-05-02",
        "decision_source": "rule_based",
    }
    assert client.post("/backtest/run", json=body, headers=headers).status_code == 200
    assert client.post("/backtest/run", json=body, headers=headers).status_code == 200
    assert client.post("/backtest/run", json=body, headers=headers).status_code == 429


def test_rule_based_still_validates_llm_only_fields_before_dropping(monkeypatch):
    """Dropping them before validation answered 200 to a malformed model."""
    monkeypatch.setenv("ENABLE_IFIND_ASHARE", "true")
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", "test-token-not-a-secret")
    spy = _Spy()
    monkeypatch.setattr(bt, "run_backtest_background", spy)

    resp = TestClient(app).post(
        "/backtest/run",
        json={
            "data_source": "ifind_ashare",
            "universe": "a_share_demo_6",
            "timeframe": "60m",
            "decision_source": "rule_based",
            "model": "x; rm -rf /",
        },
        headers=_sess(),
    )

    assert resp.status_code == 422
    assert "Invalid model id" in resp.text
    assert spy.calls == 0


def test_rule_based_reports_the_llm_fields_it_dropped(monkeypatch):
    """Dropping them is right; doing it invisibly is what hid the bad input."""
    monkeypatch.setenv("ENABLE_IFIND_ASHARE", "true")
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", "test-token-not-a-secret")
    spy = _Spy()
    monkeypatch.setattr(bt, "run_backtest_background", spy)

    resp = TestClient(app).post(
        "/backtest/run",
        json={
            "data_source": "ifind_ashare",
            "universe": "a_share_demo_6",
            "timeframe": "60m",
            "decision_source": "rule_based",
            "model": "gpt-5.2",
            "strategy_prompt": "buy low",
        },
        headers=_sess(),
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["decision_source"] == "rule_based"
    assert sorted(resp.json()["ignored_fields"]) == ["model", "strategy_prompt"]


def test_llm_run_reports_no_ignored_fields(monkeypatch):
    _enable_authenticated_openrouter_byok(monkeypatch)

    resp = TestClient(app).post(
        "/backtest/run",
        json={
            "start_date": "2026-05-01",
            "end_date": "2026-05-02",
            "decision_source": "llm",
            "billing_mode": "byok",
            "provider_id": "openrouter",
            "model": "openai/gpt-5.5",
        },
        headers=_sess(),
    )

    assert resp.status_code == 200, resp.text
    assert "ignored_fields" not in resp.json()


def test_subprocess_log_dump_is_redacted_but_not_truncated(monkeypatch):
    """print() is the only prod log channel; trimming drops every run's head."""
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", "super-secret-token")
    head = "UNIVERSE-LINE-AT-THE-VERY-TOP"
    noise = "x" * 8000
    log = f"{head}\n{noise}\naccess_token=super-secret-token\n"

    redacted = bt._redact_credentials(log)

    assert head in redacted
    assert len(redacted) > 8000
    assert "super-secret-token" not in redacted
    assert "[REDACTED]" in redacted


def test_subprocess_log_redacts_ifind_refresh_token(monkeypatch):
    monkeypatch.setenv("IFIND_REFRESH_TOKEN", "refresh-secret-token")

    redacted = bt._redact_credentials(
        "refresh_token=refresh-secret-token"
    )

    assert "refresh-secret-token" not in redacted
    assert "[REDACTED]" in redacted


def test_subprocess_log_redacts_user_financial_datasets_credential():
    credential = "user-financial-datasets-plaintext-canary"

    redacted = bt._redact_credentials(
        f"upstream failed with key={credential}", credential
    )

    assert credential not in redacted
    assert "[REDACTED]" in redacted


def test_background_error_redacts_user_financial_datasets_credential():
    credential = "user-financial-datasets-test-key"
    summary = bt._sanitize_backtest_error(
        f"runtime rejected credential={credential}",
        extra_secret=credential,
    )

    assert credential not in summary
    assert "[REDACTED]" in summary


def test_error_summary_stays_bounded(monkeypatch):
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", "super-secret-token")

    summary = bt._sanitize_backtest_error("y" * 5000 + " super-secret-token", 500)

    assert len(summary) == 500
    assert "super-secret-token" not in summary


def test_rejected_orders_endpoint_reports_t1_deferrals(client, monkeypatch):
    """Built-in agents size down instead of over-asking, so deferrals -- not
    rejections -- are where T+1's effect on strategy shows up."""
    session_id = str(uuid.uuid4())
    run_id = "agent_test_deferrals"
    _rejected_orders_run(monkeypatch, run_id, session_id, {
        "t1_deferrals": [{
            "date": "2026-04-01", "symbol": "600519.SH",
            "requested_shares": 100, "sellable_shares": 40,
            "deferred_shares": 60,
        }],
        "t1_deferred_events": 12,
        "t1_deferred_shares": 640,
        "t1_deferrals_truncated": 11,
    })

    body = client.get(
        f"/runs/{run_id}/rejected-orders", headers={"X-Session-Id": session_id}
    ).json()
    assert body["t1_deferred_events"] == 12
    assert body["t1_deferred_shares"] == 640
    assert body["t1_deferrals_truncated"] == 11
    assert body["t1_deferrals"][0]["deferred_shares"] == 60
    # A clean run on the rejection side must not read as missing data.
    assert body["rejected_orders"] == []
    assert body["count"] == 0


def test_run_metadata_exposes_deferral_scalars(client, monkeypatch):
    response = bt._run_metadata_response(
        _run_record(metadata={
            "data_source": "ifind_ashare",
            "t_plus_one_enabled": True,
            "t1_deferred_events": 12,
            "t1_deferred_shares": 640,
            "t1_deferrals": [{"symbol": "600519.SH"}] * 200,
        })
    )
    assert response.t1_deferred_events == 12
    assert response.t1_deferred_shares == 640
    # Scalars only: the records stay off the list-route model.
    assert "t1_deferrals" not in response.model_dump()


def test_run_metadata_deferral_scalars_are_none_for_legacy_runs():
    response = bt._run_metadata_response(_run_record())
    assert response.t1_deferred_events is None
    assert response.t1_deferred_shares is None


def test_backtest_window_cap_is_two_weeks():
    """A dashboard backtest is capped at a fortnight (issue #474 item 1).

    31 days was set when the parent budget was the only bound and nothing
    counted LLM calls. A 31-day window is ~22 weekdays x ~7 hourly bars = ~154
    decision bars, and a multi-step pipeline multiplies that by its step count
    -- far past what the fixed 3600s budget can finish. Two weeks matches the
    fortnight the Live Trading Leaderboard's seasons already use, so the
    product has one window vocabulary rather than two.
    """
    assert bt.MAX_BACKTEST_DAYS == 14


def test_pipeline_call_estimate_counts_one_call_per_bar_with_no_pipeline():
    """A single-prompt LLM run calls the model once per hourly bar.

    ``max(1, len(decision_steps))`` is what makes the no-pipeline path cost
    anything at all -- ``split_pipeline(None)`` returns two empty lists, and a
    bare multiplication would estimate zero calls for the most common run
    there is.
    """
    # 2026-04-15 (Wed) -> 2026-04-23 (Thu) is 7 weekdays.
    assert bt._estimated_decision_days("2026-04-15", "2026-04-23") == 7
    assert (
        bt._estimated_pipeline_llm_calls("2026-04-15", "2026-04-23", None)
        == 7 * bt.PIPELINE_DECISION_BARS_PER_TRADING_DAY
    )


def test_pipeline_call_estimate_multiplies_by_decision_steps():
    """Each decision step is its own model call, per bar.

    ``run_pipeline_decision`` loops the decision steps inside the hourly loop,
    so a four-module pipeline costs 4x a single prompt over the same window --
    which is the whole reason window length alone cannot bound a run.
    """
    pipeline = [{"label": f"Step {i}"} for i in range(4)]
    assert (
        bt._estimated_pipeline_llm_calls("2026-04-15", "2026-04-23", pipeline)
        == 7 * bt.PIPELINE_DECISION_BARS_PER_TRADING_DAY * 4
    )


def test_pipeline_call_estimate_charges_post_trade_steps_per_day_not_per_bar():
    """Post-trade steps fire at a day boundary, not every bar.

    ``engine.py`` calls ``run_post_trade_analysis`` once per trading day, so
    counting them per bar would overstate a post-trade pipeline by ~7x and
    refuse windows that finish comfortably.
    """
    days = 7
    bars = days * bt.PIPELINE_DECISION_BARS_PER_TRADING_DAY
    decision_only = [{"label": "Signal"}]
    with_post_trade = decision_only + [{"presetKey": "post_trade_analysis"}]

    assert (
        bt._estimated_pipeline_llm_calls("2026-04-15", "2026-04-23", decision_only)
        == bars
    )
    assert (
        bt._estimated_pipeline_llm_calls("2026-04-15", "2026-04-23", with_post_trade)
        == bars + days
    )


def test_pipeline_call_estimate_is_zero_for_an_unusable_range():
    """Unparseable or inverted dates estimate zero rather than raising.

    ``_validate_backtest_params`` already 422s those before the guard runs, so
    this only has to avoid becoming a second, competing date validator with its
    own error surface.
    """
    assert bt._estimated_pipeline_llm_calls("not-a-date", "also-bad", None) == 0
    assert bt._estimated_pipeline_llm_calls("2026-04-23", "2026-04-15", None) == 0


def test_pipeline_seconds_per_call_default_and_derived_call_budget():
    """15s/call over the usable budget leaves 200 calls.

    Usable is the parent budget minus the overhead constant, because data load,
    baseline generation and persistence sit outside the decision loop and their
    share is already reserved.
    """
    assert bt._DEFAULT_PIPELINE_SECONDS_PER_LLM_CALL == 15
    assert bt.PIPELINE_SECONDS_PER_LLM_CALL == 15

    usable = (
        bt.PIPELINE_SUBPROCESS_TIMEOUT_SECONDS
        - bt.SUBPROCESS_TIMEOUT_OVERHEAD_SECONDS
    )
    assert usable == 3000
    assert bt._max_pipeline_llm_calls() == 200


@pytest.mark.parametrize("raw", ["", "   ", "abc", "12.5", "0", "-4", "301"])
def test_pipeline_seconds_per_call_falls_back_on_junk(monkeypatch, raw):
    """A mistyped env value logs and falls back; it never kills app boot.

    CLAUDE.md records that a bare int() at module scope in this very module once
    took the whole app down on a typo. 0 is refused specifically because it
    would make the call budget a ZeroDivisionError rather than a bound.
    """
    monkeypatch.setenv("PIPELINE_SECONDS_PER_LLM_CALL", raw)
    assert (
        bt._pipeline_seconds_per_llm_call()
        == bt._DEFAULT_PIPELINE_SECONDS_PER_LLM_CALL
    )


def test_pipeline_seconds_per_call_honours_a_valid_override(monkeypatch):
    """An operator who has measured real latency tunes this without a deploy."""
    monkeypatch.setenv("PIPELINE_SECONDS_PER_LLM_CALL", "30")
    assert bt._pipeline_seconds_per_llm_call() == 30

    monkeypatch.setenv("PIPELINE_SECONDS_PER_LLM_CALL", "1")
    assert bt._pipeline_seconds_per_llm_call() == 1

    monkeypatch.setenv("PIPELINE_SECONDS_PER_LLM_CALL", "300")
    assert bt._pipeline_seconds_per_llm_call() == 300


def test_pipeline_window_guard_allows_the_shipped_default_window():
    """The modal's own pre-filled window must not 422 on the onboarding path.

    2026-04-15 -> 2026-04-23 with the advertised four-module pipeline is 196 of
    200 calls. It passes by four -- deliberately pinned, so that anyone who
    shrinks the budget or raises the per-call estimate sees exactly which run
    they just made illegal.
    """
    pipeline = [{"label": f"Step {i}"} for i in range(4)]
    assert (
        bt._enforce_pipeline_llm_window(
            "pipeline", "llm", "2026-04-15", "2026-04-23", pipeline
        )
        is None
    )


def test_pipeline_window_guard_refuses_a_wide_pipeline_over_a_legal_window():
    """A window inside MAX_BACKTEST_DAYS can still be uncompletable.

    This is the case MAX_BACKTEST_DAYS cannot express: 14 calendar days is
    legal, but four decision steps over it is 280 calls against a 200-call
    budget. Window length alone does not say how much a run costs.
    """
    pipeline = [{"label": f"Step {i}"} for i in range(4)]
    with pytest.raises(HTTPException) as excinfo:
        bt._enforce_pipeline_llm_window(
            "pipeline", "llm", "2026-04-06", "2026-04-19", pipeline
        )
    assert excinfo.value.status_code == 422


def test_pipeline_window_guard_names_both_levers_and_both_numbers():
    """A refusal naming one lever is a dead end for a user blocked by the other.

    The bound is a product of window length and pipeline width, so the message
    has to offer both exits -- and name the estimate AND the budget, the way
    _enforce_ai_hedge_fund_window names the bound and the request.
    """
    pipeline = [{"label": f"Step {i}"} for i in range(4)]
    with pytest.raises(HTTPException) as excinfo:
        bt._enforce_pipeline_llm_window(
            "pipeline", "llm", "2026-04-06", "2026-04-19", pipeline
        )
    detail = excinfo.value.detail
    assert "280" in detail          # what this request needs
    assert "200" in detail          # what the deployment allows
    assert "shorten" in detail.lower()
    assert "step" in detail.lower()


def test_pipeline_window_guard_allows_a_single_prompt_over_the_full_window():
    """The no-pipeline path costs one call per bar and fits the full fortnight.

    70 calls of 200. Lowering MAX_BACKTEST_DAYS must not be read as having
    made the simple path marginal too.
    """
    assert (
        bt._enforce_pipeline_llm_window(
            "pipeline", "llm", "2026-04-06", "2026-04-19", None
        )
        is None
    )


def test_pipeline_window_guard_ignores_runs_it_does_not_govern():
    """Rule-based runs and the hosted runtime are out of scope.

    A rule-based run makes no model calls at all, and the hosted path has its
    own window bound sized from its own enforced per-step timeout. Applying a
    pipeline-shaped estimate to either would refuse runs on arithmetic that
    does not describe them.
    """
    wide = [{"label": f"Step {i}"} for i in range(4)]
    assert (
        bt._enforce_pipeline_llm_window(
            "pipeline", "rule_based", "2026-04-06", "2026-04-19", wide
        )
        is None
    )
    assert (
        bt._enforce_pipeline_llm_window(
            "ai_hedge_fund", "llm", "2026-04-06", "2026-04-19", wide
        )
        is None
    )


def test_pipeline_window_guard_tracks_the_operator_override(monkeypatch):
    """Raising the per-call estimate shrinks what is runnable, immediately."""
    pipeline = [{"label": f"Step {i}"} for i in range(4)]
    monkeypatch.setattr(bt, "PIPELINE_SECONDS_PER_LLM_CALL", 30)
    assert bt._max_pipeline_llm_calls() == 100
    with pytest.raises(HTTPException) as excinfo:
        bt._enforce_pipeline_llm_window(
            "pipeline", "llm", "2026-04-15", "2026-04-23", pipeline
        )
    assert excinfo.value.status_code == 422


def test_backtest_run_refuses_an_uncompletable_pipeline_window(client):
    """The 422 arrives before the rate limiter, the slot ledger and any spend.

    A refusal that costs a rate-limit token or a concurrency slot punishes the
    user for a request the server was always going to decline -- and one that
    arrives after the billing preflight has touched the credential store has
    already done work on a run that cannot happen.
    """
    response = client.post(
        "/backtest/run",
        json={
            "start_date": "2026-04-06",
            "end_date": "2026-04-19",
            "decision_source": "llm",
            "model": "claude-haiku-4-5-20251001",
            "billing_mode": "byok",
            "pipeline": [{"label": f"Step {i}"} for i in range(4)],
        },
        headers=_sess(),
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "280" in detail
    assert "shorten" in detail.lower()


def test_backtest_run_still_refuses_an_over_long_window_on_dates_alone(client):
    """MAX_BACKTEST_DAYS keeps its own distinct message.

    The two bounds answer different questions and a user who picked a
    three-week window should be told the window is too long, not handed call
    arithmetic about a pipeline they may not have configured.
    """
    response = client.post(
        "/backtest/run",
        json={
            "start_date": "2026-04-01",
            "end_date": "2026-04-30",
            "decision_source": "llm",
            "model": "claude-haiku-4-5-20251001",
            "billing_mode": "byok",
        },
        headers=_sess(),
    )
    assert response.status_code == 422
    assert "max 14 days" in response.json()["detail"]


def test_pipeline_refusal_breakdown_accounts_for_post_trade_steps():
    """The shown arithmetic must reconcile with the total printed beside it.

    The breakdown is a product of per-bar terms, but a post-trade step fires
    once a trading day. Without naming it the message multiplies out to LESS
    than the total it quotes, so a user who checks the sum concludes a correct
    refusal is a bug -- and the one lever the message does not mention is the
    one that would fix their run.
    """
    days = 10
    bars_per_day = bt.PIPELINE_DECISION_BARS_PER_TRADING_DAY
    pipeline = [{"label": f"Step {i}"} for i in range(4)] + [
        {"presetKey": "post_trade_analysis"}
    ]
    expected = days * bars_per_day * 4 + days

    with pytest.raises(HTTPException) as excinfo:
        bt._enforce_pipeline_llm_window(
            bt.PIPELINE_RUNTIME_TYPE,
            bt.LLM_DECISION_SOURCE,
            "2026-04-06",
            "2026-04-19",
            pipeline,
        )

    detail = excinfo.value.detail
    assert str(expected) in detail
    assert f"{days} trading days x {bars_per_day} hourly bars x 4 pipeline step(s)" in detail
    assert "plus 1 post-trade step(s) once per trading day" in detail
    # The breakdown now reconciles: the per-bar product plus the per-day term
    # equals the quoted total.
    assert days * bars_per_day * 4 + days == expected


def test_pipeline_refusal_breakdown_is_unchanged_without_post_trade_steps():
    """The common case keeps the exact wording it shipped with."""
    with pytest.raises(HTTPException) as excinfo:
        bt._enforce_pipeline_llm_window(
            bt.PIPELINE_RUNTIME_TYPE,
            bt.LLM_DECISION_SOURCE,
            "2026-04-06",
            "2026-04-19",
            [{"label": f"Step {i}"} for i in range(4)],
        )

    detail = excinfo.value.detail
    assert "280" in detail
    assert "post-trade" not in detail
    assert "plus" not in detail


# ===========================================================================
# The bar cadence is per market (the A-share false refusal)
# ===========================================================================
#
# PIPELINE_DECISION_BARS_PER_TRADING_DAY was one number for both markets,
# taken from the wider one. This guard computes a REFUSAL, so billing CN's
# four-bar session at seven overstated an A-share run by ~75% and refused
# windows that would have finished -- the direction the banner above the
# constants explicitly rules out.


def test_a_share_runs_are_billed_at_the_a_share_session_length():
    """CN trades four 60m bars a day, not the US session's seven."""
    from dashboard.backend.infrastructure.market_data.ifind_ashare import (
        ASHARE_SESSIONS_PER_TRADING_DAY,
    )

    assert bt._pipeline_decision_bars_per_trading_day(bt.IFIND_ASHARE) == (
        ASHARE_SESSIONS_PER_TRADING_DAY
    )
    assert ASHARE_SESSIONS_PER_TRADING_DAY < bt.PIPELINE_DECISION_BARS_PER_TRADING_DAY


@pytest.mark.parametrize("source", [None, "", "alpaca", "vnpy_simulation", "unknown"])
def test_unrecognised_markets_keep_the_wider_us_cadence(source):
    """Falling back to the wider market keeps an unknown source conservative."""
    assert bt._pipeline_decision_bars_per_trading_day(source) == (
        bt.PIPELINE_DECISION_BARS_PER_TRADING_DAY
    )


def test_the_shipped_a_share_prefill_is_not_refused_at_three_pipeline_steps():
    """The regression this fix exists for, on the A-share onboarding path.

    The prefill window with a three-step pipeline estimated 11 x 7 x 3 = 231
    against a 200 budget and 422'd. Its real cost is 11 x 4 x 3 = 132.
    """
    pipeline = [{"label": f"Step {index}"} for index in range(3)]

    assert (
        bt._estimated_pipeline_llm_calls(
            "2026-04-01", "2026-04-15", pipeline, bt.IFIND_ASHARE
        )
        <= bt._max_pipeline_llm_calls()
    )
    # Not a no-op guard: the same window on the US cadence is still refused,
    # so this passes because the cadence moved, not because the budget is roomy.
    assert (
        bt._estimated_pipeline_llm_calls("2026-04-01", "2026-04-15", pipeline, "alpaca")
        > bt._max_pipeline_llm_calls()
    )

    bt._enforce_pipeline_llm_window(
        bt.PIPELINE_RUNTIME_TYPE,
        bt.LLM_DECISION_SOURCE,
        "2026-04-01",
        "2026-04-15",
        pipeline,
        bt.IFIND_ASHARE,
    )


def test_the_refusal_breakdown_quotes_the_market_it_refused():
    """A CN refusal that prints 7 bars/day is arithmetic the user cannot check."""
    pipeline = [{"label": f"Step {index}"} for index in range(40)]

    with pytest.raises(HTTPException) as excinfo:
        bt._enforce_pipeline_llm_window(
            bt.PIPELINE_RUNTIME_TYPE,
            bt.LLM_DECISION_SOURCE,
            "2026-04-01",
            "2026-04-15",
            pipeline,
            bt.IFIND_ASHARE,
        )

    detail = excinfo.value.detail
    bars = bt._pipeline_decision_bars_per_trading_day(bt.IFIND_ASHARE)
    assert f"x {bars} hourly bars x" in detail
    assert f"x {bt.PIPELINE_DECISION_BARS_PER_TRADING_DAY} hourly bars x" not in detail


def test_omitting_the_data_source_preserves_the_previous_estimate():
    """Callers that predate the parameter must be unaffected."""
    pipeline = [{"label": "Step 1"}, {"presetKey": "post_trade_analysis"}]

    assert bt._estimated_pipeline_llm_calls(
        "2026-04-06", "2026-04-19", pipeline
    ) == bt._estimated_pipeline_llm_calls("2026-04-06", "2026-04-19", pipeline, "alpaca")


def test_finalize_slot_records_a_timeout_as_its_own_outcome():
    """Four outcomes, not three: succeeded, failed, cancelled, timed out.

    A timeout is not a cancel (the user did nothing) and not a crash (nothing
    threw -- the product ran out of the budget it set itself, and billed for
    what settled first). Asserted as separate fields rather than one tuple
    equality so a failure names which field moved.
    """
    run_id = "agent_timeout_outcome"
    session_id = str(uuid.uuid4())
    assert (
        bt._try_acquire_backtest_slot(
            live_run_id=run_id, session_id=session_id, user_id=None
        )
        is None
    )

    detail = {
        "limit_seconds": 3600,
        "billing_mode": "platform_credits",
        "spent_micro": 42_318,
        "model_calls": 2,
    }
    bt._finalize_slot(
        run_id, error=None, runs_count=0, timed_out=True, timeout_detail=detail
    )

    slot = bt._recent_slots[run_id]
    assert slot["timed_out"] is True
    assert slot["cancelled"] is False
    assert slot["error"] is None
    assert slot["running"] is False
    assert slot["timeout_detail"] == detail

    # A real cancel must never also read as a timeout: the two flags are set
    # by disjoint call sites (the cancel route always passes `cancelled=True`
    # with `timed_out` left at its default), and the status route trusts that
    # disjointness to place its two branches in either order safely. Pin it
    # at the source rather than only at the read path.
    cancel_run_id = "agent_timeout_outcome_cancel_check"
    assert (
        bt._try_acquire_backtest_slot(
            live_run_id=cancel_run_id, session_id=session_id, user_id=None
        )
        is None
    )
    bt._finalize_slot(cancel_run_id, error=None, runs_count=0, cancelled=True)
    cancel_slot = bt._recent_slots[cancel_run_id]
    assert cancel_slot["cancelled"] is True
    assert cancel_slot["timed_out"] is False


def test_status_prefers_cancelled_over_timed_out_when_a_slot_carries_both(client):
    """Pins `get_backtest_status`'s branch order directly, because nothing else
    can.

    In real traffic `cancelled` and `timed_out` are set by disjoint call sites
    -- the cancel route never passes `timed_out=True`, and the worker's timeout
    arm never passes `cancelled=True` -- so a slot never legitimately carries
    both today. That is exactly why the ordering is otherwise untestable: a
    mutation check that swapped the two `elif` branches in
    `get_backtest_status` left every other test in this suite green, because
    none of them can construct the one slot state where the order matters.

    This test manufactures that state directly -- artificial today, but
    exactly the shape a future bug that sets both flags would produce -- so
    the branch order (cancelled wins) stays pinned even though production
    code never manufactures it on its own.
    """
    session_id = str(uuid.uuid4())
    headers = {"X-Session-Id": session_id}
    run_id = "agent_timeout_and_cancel_both"
    assert (
        bt._try_acquire_backtest_slot(
            live_run_id=run_id, session_id=session_id, user_id=None
        )
        is None
    )
    bt._finalize_slot(
        run_id,
        error=None,
        runs_count=0,
        cancelled=True,
        timed_out=True,
        timeout_detail={
            "limit_seconds": 3600,
            "billing_mode": "platform_credits",
            "spent_micro": 42_318,
            "model_calls": 2,
        },
    )

    body = client.get(
        f"/backtest/status?live_run_id={run_id}", headers=headers
    ).json()

    assert body["cancelled"] is True
    assert "timed_out" not in body
    assert "timeout" not in body


def test_status_reports_timed_out_as_neither_an_error_nor_a_success(client):
    """Both frontend surfaces branch on these keys.

    A timeout routed through `error` paints the red "Backtest did not start"
    panel for something the user did not cause; one routed through `success`
    would claim results that do not exist. Neither key may appear.
    """
    session_id = str(uuid.uuid4())
    headers = {"X-Session-Id": session_id}
    run_id = "agent_timeout_status"
    assert (
        bt._try_acquire_backtest_slot(
            live_run_id=run_id, session_id=session_id, user_id=None
        )
        is None
    )
    bt._finalize_slot(
        run_id,
        error=None,
        runs_count=0,
        timed_out=True,
        timeout_detail={
            "limit_seconds": 3600,
            "billing_mode": "platform_credits",
            "spent_micro": 42_318,
            "model_calls": 2,
        },
    )

    body = client.get(
        f"/backtest/status?live_run_id={run_id}", headers=headers
    ).json()

    assert body["running"] is False
    assert body["timed_out"] is True
    assert "error" not in body
    assert "success" not in body
    assert body["message"] == "Backtest stopped at the time limit."
    assert body["timeout"] == {
        "limit_seconds": 3600,
        "billing_mode": "platform_credits",
        "spent_micro": 42_318,
        "model_calls": 2,
    }


def test_status_omits_the_timeout_block_when_no_detail_was_recorded(client):
    """The branch must still answer for a slot finalized without a detail dict
    -- the shape is conditional, so absence is a real case, not an error."""
    session_id = str(uuid.uuid4())
    headers = {"X-Session-Id": session_id}
    run_id = "agent_timeout_no_detail"
    assert (
        bt._try_acquire_backtest_slot(
            live_run_id=run_id, session_id=session_id, user_id=None
        )
        is None
    )
    bt._finalize_slot(run_id, error=None, runs_count=0, timed_out=True)

    body = client.get(
        f"/backtest/status?live_run_id={run_id}", headers=headers
    ).json()

    assert body["timed_out"] is True
    assert "timeout" not in body


def _timed_out_slot_with_cost(*, run_id, session_id, owner_session=None, user_id=None):
    """Finalize a run as timed out carrying a full, owner-only `timeout_detail`."""
    assert (
        bt._try_acquire_backtest_slot(
            live_run_id=run_id,
            session_id=session_id,
            owner_session=owner_session,
            user_id=user_id,
        )
        is None
    )
    bt._finalize_slot(
        run_id,
        error=None,
        runs_count=0,
        timed_out=True,
        timeout_detail={
            "limit_seconds": 3600,
            "billing_mode": "platform_credits",
            "spent_micro": 42_318,
            "model_calls": 2,
        },
    )


def test_status_withholds_the_run_cost_from_a_caller_who_did_not_start_it(client):
    """`_slot_visible_to` is deliberately weaker than ownership, and this route
    is what makes that weakness reachable.

    For a built-in agent the read rule admits anyone holding the slot's
    `session_id` -- the AGENT's session, which the running branch of this very
    route hands back to every visitor who runs that agent (see
    `_slot_cancellable_by`'s docstring, which documents the gap). That was a
    tolerable trade while the payload carried progress and elapsed time.
    `timeout_detail` carries what the run CHARGED, so an anonymous visitor who
    ran the same built-in agent once could poll with the published id and read
    a signed-in stranger's Credits spend out of `_recent_slots`.

    The outcome itself stays public -- every poller branches on `timed_out`,
    and blinding them would break the built-in-agent card on purpose -- so only
    the account-level facts are withheld.
    """
    agent_session = str(uuid.uuid4())
    _timed_out_slot_with_cost(
        run_id="agent_timeout_shared_session",
        session_id=agent_session,
        owner_session=str(uuid.uuid4()),
        user_id=4242,
    )

    stranger = client.get(
        "/backtest/status?live_run_id=agent_timeout_shared_session",
        headers={"X-Session-Id": agent_session},
    ).json()

    # Still told what happened, and against which budget.
    assert stranger["timed_out"] is True
    assert stranger["timeout"] == {"limit_seconds": 3600}
    assert "42318" not in json.dumps(stranger)


def test_status_gives_the_run_cost_to_the_session_that_started_the_run(client):
    """The redaction above must not blind the owner. An anonymous run stays
    bound to the browser session that started it, and that caller is exactly
    who the disclosure is for."""
    owner_session = str(uuid.uuid4())
    _timed_out_slot_with_cost(
        run_id="agent_timeout_owner_reads_cost", session_id=owner_session
    )

    body = client.get(
        "/backtest/status?live_run_id=agent_timeout_owner_reads_cost",
        headers={"X-Session-Id": owner_session},
    ).json()

    assert body["timeout"]["spent_micro"] == 42_318
    assert body["timeout"]["model_calls"] == 2
    assert body["timeout"]["billing_mode"] == "platform_credits"


def test_a_second_finalize_cannot_overwrite_a_timeout():
    """The already-finalized guard enumerated only `cancelled`.

    The timeout arm finalizes and only THEN clears `resolved_live_run_id`, so
    anything raising in between -- `_emit_slot_run_event` reaches the analytics
    store -- lands in the worker's `finally` with the id still set and
    finalizes the same run a second time with `error=None, runs_count=0`. With
    `timed_out` missing from the guard that call fell through into the branch
    that rewrites the legacy mirror unconditionally, stamping this run's
    outcome onto whichever unrelated run that mirror currently describes.
    """
    session_id = str(uuid.uuid4())
    run_id = "agent_timeout_double_finalize"
    _timed_out_slot_with_cost(run_id=run_id, session_id=session_id)

    # What the legacy mirror describes now is a DIFFERENT, still-live run.
    bt.backtest_status.update(
        {"running": True, "live_run_id": "agent_unrelated_live", "runs_count": 7}
    )

    bt._finalize_slot(run_id, error=None, runs_count=0)

    assert bt._recent_slots[run_id]["timed_out"] is True
    assert bt._recent_slots[run_id]["timeout_detail"]["spent_micro"] == 42_318
    assert bt.backtest_status["live_run_id"] == "agent_unrelated_live"
    assert bt.backtest_status["running"] is True
    assert bt.backtest_status["runs_count"] == 7


def test_a_hanging_spend_lookup_cannot_strand_the_slot(monkeypatch):
    """`except Exception` answers a query that raises; only a deadline answers
    one that never returns.

    `sum_run_llm_spend` reaches Postgres with no statement timeout beneath it
    (`db_pool`'s POOL_TIMEOUT_SECONDS bounds connection checkout, not the
    query). Run inline, a resuming Neon instance blocks the worker BEFORE
    `_finalize_slot`: the slot stays `running` forever, holding one of the five
    global backtest slots and one of the owner's quota for a child that is
    already dead, while the poller keeps reporting it as in flight.

    The outcome must not depend on the disclosure being fetchable, so the
    finalize happens either way and the amount is simply absent.
    """
    monkeypatch.setattr(bt, "TIMEOUT_SPEND_LOOKUP_SECONDS", 0.2)
    released = threading.Event()

    def _never_returns(_user_id, _run_id):
        released.wait(30)
        return (42_318, 2)

    try:
        slot = _drive_pipeline_timeout(
            monkeypatch,
            run_id="agent_timeout_slow_lookup",
            billing_mode="platform_credits",
            user_id=4242,
            spend_lookup=_never_returns,
        )

        assert slot["running"] is False
        assert slot["timed_out"] is True
        assert slot["timeout_detail"] == {
            "limit_seconds": 3600,
            "billing_mode": "platform_credits",
            "spent_micro": None,
            "model_calls": None,
        }
        assert "agent_timeout_slow_lookup" not in bt._active_slots
    finally:
        # Let the abandoned daemon thread finish rather than leaving it parked
        # for the rest of the session.
        released.set()


def test_billing_mode_is_not_threaded_for_a_run_the_llm_preflight_never_saw(
    client, monkeypatch
):
    """`billing_mode` is a request field that reaches the thread kwargs on
    EVERY run, but it is only validated inside the
    `decision_source='llm' and runtime_type=='pipeline'` block.

    Threaded unconditionally, `{"decision_source": "rule_based",
    "billing_mode": "platform_credits"}` made the worker's timeout arm run a
    Credits lookup -- and put a Credits sentence on the timeout card -- for a
    run that never touched a model.
    """
    spy = _Spy()
    monkeypatch.setattr(bt, "run_backtest_background", spy)

    resp = client.post(
        "/backtest/run",
        json={
            "start_date": "2026-05-01",
            "end_date": "2026-05-02",
            "decision_source": "rule_based",
            "billing_mode": "platform_credits",
        },
        headers=_sess(),
    )

    assert resp.status_code == 200
    assert spy.calls == 1
    assert spy.last_kwargs["execution_handoff_payload"] is None
    assert spy.last_kwargs["billing_mode"] is None


def test_timed_out_run_emits_backtest_failed_with_the_timeout_reason(monkeypatch):
    """A timeout IS a failure -- the user got nothing and was charged -- so it
    keeps `backtest_failed` and the success-rate KPI stays honest. What was
    missing is the reason, and `error_category` is the field that carries it."""
    emitted = []
    monkeypatch.setattr(
        bt.analytics_instrumentation,
        "emit_run_event",
        lambda **kwargs: emitted.append(kwargs),
    )

    run_id = "agent_timeout_analytics"
    session_id = str(uuid.uuid4())
    assert (
        bt._try_acquire_backtest_slot(
            live_run_id=run_id, session_id=session_id, user_id=4242
        )
        is None
    )
    bt._finalize_slot(run_id, error=None, runs_count=0, timed_out=True)

    assert emitted == [
        {
            "event_name": "backtest_failed",
            "user_id": 4242,
            "run_id": run_id,
            "error_category": "run_timeout",
        }
    ]
