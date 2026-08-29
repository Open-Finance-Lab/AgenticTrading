# Backtest Benchmark Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show exact Agent, DJIA, Nasdaq-100, and Buy & Hold performance metrics below the Backtest chart while moving the complete Trading Log into an independently scrolling right rail.

**Architecture:** Extend the existing chart-data route to include the stored Buy & Hold curve beside US index curves. Put benchmark classification, metric calculation, best-value selection, and percentage-point deltas in a small dependency-free browser helper; keep DOM rendering and Chart.js orchestration in `app.js`. Historical chart/comparison and Trading Log requests use one run-scoped request token but settle independently so either surface can fail without clearing the other.

**Tech Stack:** FastAPI/Pydantic, vanilla JavaScript, Chart.js 4.4, HTML/CSS, pytest, Node.js runtime contract tests

**Spec:** `docs/superpowers/specs/2026-08-28-backtest-benchmark-comparison-design.md`

## Global Constraints

- Supported US runs compare `Your Agent`, `DJIA`, `Nasdaq-100`, and `Buy & Hold` in that order.
- Metrics are exactly Final Value, Total Return, Max Drawdown, and hourly annualized Sharpe Ratio; do not add Annualized Return, Annualized Volatility, alpha, beta, or tracking error.
- Calculate every series from the aligned chart-data `values` arrays; do not mix stored Agent metrics with frontend-computed benchmark metrics.
- Sharpe uses population standard deviation, zero risk-free rate, and `sqrt(252 * 6.5)`.
- Non-US profiles show only applicable series; do not display DJIA or Nasdaq-100 for iFinD A-share runs.
- Missing, flat, insufficient, or non-finite inputs render unavailable and never participate in `Best` selection.
- Legend toggles affect chart-line visibility only; they never remove comparison-table columns.
- Preserve every existing Trading Log field, filter, audit detail, empty state, error state, and truncation notice.
- Keep the existing session/ownership boundary and escaping helpers. Do not add a database table, migration, dependency, or endpoint.
- Use only synthetic fixture values. Never read, display, or commit a real API key, local database, `.superpowers/`, or `work/` content.

## File Map

- Create `dashboard/frontend/js/backtest-comparison.js`: pure series classification, metric calculation, comparison model, and formatting constants; no DOM or API access.
- Create `dashboard/backend/tests/fixtures/backtest-comparison.json`: synthetic US run, four aligned series, and order records shared by Node/runtime and visual checks.
- Create `dashboard/backend/tests/test_backtest_chart_data.py`: chart-data route contract for combined stored and index baselines.
- Create `dashboard/backend/tests/test_backtest_comparison_frontend.py`: Node-executed pure helper tests plus shipped HTML/app.js contracts.
- Modify `dashboard/backend/api/routers/backtests.py`: always offer the paired stored Buy & Hold curve to chart-data.
- Modify `dashboard/backend/chart_style.py`: give Buy & Hold a color distinct from Nasdaq-100 once both appear together.
- Modify `dashboard/backend/tests/test_chart_style.py`: pin the distinct Buy & Hold color.
- Modify `dashboard/frontend/app.html`: semantic comparison markup, DOM legend host, right-rail Trading Log feed, and script/cache revisions.
- Modify `dashboard/frontend/app.js`: comparison rendering, DOM legend behavior, independent loading, stale-response guard, and vertical log rendering.
- Modify `dashboard/frontend/styles.css`: approved three-column layout, comparison table, log feed, focus, and responsive rules.
- Modify `dashboard/backend/tests/test_trading_log_order_events_ui.py`: keep current behavioral coverage while asserting the vertical feed contract.
- Modify `dashboard/backend/tests/test_chart_baseline_notice_frontend.py`, `dashboard/backend/tests/test_ifind_ashare_frontend.py`, `dashboard/backend/tests/test_frontend_fast_boot.py`, `dashboard/backend/tests/test_analytics_frontend.py`, and `dashboard/backend/tests/test_admin_analytics_frontend.py`: update integration, applicability, and asset-order/cache contracts.

---

### Task 1: Return Buy & Hold Beside US Index Baselines

**Files:**
- Create: `dashboard/backend/tests/test_backtest_chart_data.py`
- Modify: `dashboard/backend/api/routers/backtests.py:208-219,2168-2208`
- Modify: `dashboard/backend/chart_style.py:15-27`
- Modify: `dashboard/backend/tests/test_chart_style.py:14-27`

**Interfaces:**
- Consumes: existing `run["baseline_buyhold_run_id"]`, `db.get_run()`, `db.get_equity_curve()`, and `build_backtest_chart_data(..., stored_baselines, include_market_indexes)`.
- Produces: unchanged `GET /api/backtest/{run_id}/chart-data` schema whose US `series` contains Agent, stored Buy & Hold, `index:^DJI`, and `index:^NDX` when all sources are available.

- [ ] **Step 1: Write failing route and color tests**

Create `dashboard/backend/tests/test_backtest_chart_data.py` with a direct route-function test that avoids network and database state:

```python
import dashboard.backend.api.routers.backtests as bt
import pytest
from fastapi import HTTPException


def _curve():
    return [
        {"timestamp": "2026-05-04T14:30:00", "equity": 1_000.0},
        {"timestamp": "2026-05-04T15:30:00", "equity": 1_010.0},
    ]


def _install_route_fakes(monkeypatch, *, buyhold_curve, index_ok=True):
    run = {
        "run_id": "agent-1",
        "agent_name": "Agent",
        "start_date": "2026-05-04",
        "end_date": "2026-05-04",
        "initial_equity": 1_000.0,
        "baseline_buyhold_run_id": "buyhold-1",
        "metadata": {"data_source": "alpaca"},
    }
    monkeypatch.setattr(bt, "get_session_id_from_request", lambda _request: "session-1")
    monkeypatch.setattr(
        bt.db,
        "get_run_with_session",
        lambda run_id, session_id: run
        if (run_id, session_id) == ("agent-1", "session-1")
        else None,
    )
    monkeypatch.setattr(
        bt.db,
        "get_run",
        lambda run_id: {"run_id": run_id, "agent_name": "buy-and-hold"}
        if run_id == "buyhold-1"
        else None,
    )
    monkeypatch.setattr(
        bt.db,
        "get_equity_curve",
        lambda run_id: _curve() if run_id == "agent-1" else buyhold_curve,
    )
    monkeypatch.setattr(bt, "_filter_equity_for_run", lambda _run, curve: curve)
    monkeypatch.setattr(
        bt.agent_service.agents,
        "get_agent_by_session",
        lambda _session_id: None,
    )

    def fake_indexes(timestamps, *_args, **_kwargs):
        indexes = [
            ("DJIA index", "index:^DJI", [1_000.0, 1_005.0]),
            ("Nasdaq-100", "index:^NDX", [1_000.0, 1_015.0]),
        ]
        return (indexes if index_ok else [], index_ok)

    monkeypatch.setattr(
        "dashboard.backend.equity_plot.market_index_baselines_with_status",
        fake_indexes,
    )


def test_us_chart_data_includes_buyhold_and_market_indexes(monkeypatch):
    _install_route_fakes(monkeypatch, buyhold_curve=_curve())
    response = bt.get_backtest_chart_data("agent-1", object())
    assert [series.run_id for series in response.series] == [
        "agent-1",
        "buyhold-1",
        "index:^DJI",
        "index:^NDX",
    ]


def test_missing_buyhold_curve_does_not_substitute_another_run(monkeypatch):
    _install_route_fakes(monkeypatch, buyhold_curve=[])
    response = bt.get_backtest_chart_data("agent-1", object())
    assert "buyhold-1" not in [series.run_id for series in response.series]
    assert response.index_baselines_ok is True


def test_index_failure_retains_agent_and_buyhold(monkeypatch):
    _install_route_fakes(monkeypatch, buyhold_curve=_curve(), index_ok=False)
    response = bt.get_backtest_chart_data("agent-1", object())
    assert [series.run_id for series in response.series] == [
        "agent-1", "buyhold-1"
    ]
    assert response.index_baselines_ok is False


def test_chart_data_keeps_selected_run_session_scoped(monkeypatch):
    _install_route_fakes(monkeypatch, buyhold_curve=_curve())
    monkeypatch.setattr(
        bt, "get_session_id_from_request", lambda _request: "other-session"
    )
    with pytest.raises(HTTPException) as exc_info:
        bt.get_backtest_chart_data("agent-1", object())
    assert exc_info.value.status_code == 404
```

Extend `test_series_kind_and_colors_match_playground()`:

```python
assert series_color("idx2", "Nasdaq-100") == "#9AA4B2"
assert series_color("buyhold_1", "buy-and-hold") == "#34D399"
assert series_color("buyhold_1", "buy-and-hold") != series_color(
    "idx2", "Nasdaq-100"
)
```

- [ ] **Step 2: Run the tests and verify the missing combined-baseline contract**

Run:

```bash
python -m pytest dashboard/backend/tests/test_backtest_chart_data.py dashboard/backend/tests/test_chart_style.py -q
```

Expected: the US route test fails because `buyhold-1` is absent; the color test fails because Buy & Hold still uses `#9AA4B2`.

- [ ] **Step 3: Make stored and index baselines coexist**

Change the chart-data call to pass the stored curve for every profile. Keep `include_market_indexes` profile-controlled:

```python
payload = build_backtest_chart_data(
    run_id=run_id,
    agent_name=run.get("agent_name") or "Agent",
    llm_model=run.get("llm_model"),
    start_date=run.get("start_date") or "",
    end_date=run.get("end_date") or "",
    initial_capital=initial_capital,
    agent_curve=agent_curve,
    card_name=card_name,
    stored_baselines=_stored_buyhold_baseline(run),
    include_market_indexes=profile.index_baseline_enabled,
    market_timezone=profile.timezone,
)
```

Update the route docstring to say the dashboard response may include the paired stored Buy & Hold curve in addition to the Discord-compatible index baselines. Set the shared color map entry:

```python
"buy-and-hold": "#34D399",
```

- [ ] **Step 4: Run route, plot, and non-US regressions**

Run:

```bash
python -m pytest dashboard/backend/tests/test_backtest_chart_data.py dashboard/backend/tests/test_chart_style.py dashboard/backend/tests/test_equity_plot.py dashboard/backend/tests/integration/test_ifind_ashare_backtest.py -q
```

Expected: PASS. The provider-failure route case retains Agent + Buy & Hold with `index_baselines_ok=false`; the ownership case remains 404; and the iFinD assertions remain Agent + Buy & Hold with no DJIA/Nasdaq series.

- [ ] **Step 5: Commit the backend contract**

```bash
git add dashboard/backend/api/routers/backtests.py dashboard/backend/chart_style.py dashboard/backend/tests/test_backtest_chart_data.py dashboard/backend/tests/test_chart_style.py
git commit -m "feat(backtest): include buy-and-hold chart baseline"
```

---

### Task 2: Build the Pure Benchmark Comparison Model

**Files:**
- Create: `dashboard/frontend/js/backtest-comparison.js`
- Create: `dashboard/backend/tests/fixtures/backtest-comparison.json`
- Create: `dashboard/backend/tests/test_backtest_comparison_frontend.py`

**Interfaces:**
- Consumes: chart-data `{agent_run_id, series, index_baselines_ok}` and selected run `{data_source, baseline_buyhold_run_id, reporting_currency?}`.
- Produces: `window.BacktestComparison.buildModel(payload, run)` returning `{columns, bestByMetric, agentDeltas, indexBaselinesOk}`, plus `calculateMetrics(values)` and immutable `METRICS` descriptors.

- [ ] **Step 1: Add a synthetic shared fixture**

Create `dashboard/backend/tests/fixtures/backtest-comparison.json` with no credential or database material:

```json
{
  "run": {
    "run_id": "agent-fixture",
    "data_source": "alpaca",
    "baseline_buyhold_run_id": "buyhold-fixture",
    "reporting_currency": "USD"
  },
  "chart": {
    "agent_run_id": "agent-fixture",
    "timestamps": [
      "2026-05-04T14:30:00Z",
      "2026-05-04T15:30:00Z",
      "2026-05-04T16:30:00Z",
      "2026-05-04T17:30:00Z"
    ],
    "x_labels": ["2026-05-04", "", "", ""],
    "index_baselines_ok": true,
    "series": [
      {"run_id": "index:^NDX", "label": "Nasdaq-100", "values": [1000, 1020, 1010, 1060], "color": "#9AA4B2", "dashed": true},
      {"run_id": "buyhold-fixture", "label": "buy-and-hold", "values": [1000, 1008, 1014, 1024], "color": "#34D399", "dashed": true},
      {"run_id": "agent-fixture", "label": "Agent 33", "values": [1000, 1010, 995, 1017.4], "color": "#4FC3F7", "dashed": false},
      {"run_id": "index:^DJI", "label": "DJIA index", "values": [1000, 1005, 1002, 1011.1], "color": "#F5C04A", "dashed": true}
    ]
  },
  "trades": {
    "trades": [
      {"timestamp": "2026-05-04T15:30:00Z", "symbol": "AAPL", "side": "BUY", "quantity": 1, "price": 277.33, "value": 277.33, "reason": "Momentum entry"}
    ],
    "order_events": [],
    "order_events_truncated": 0
  }
}
```

- [ ] **Step 2: Write failing Node-executed model tests**

In `test_backtest_comparison_frontend.py`, load and execute the future helper as a browser global:

```python
import json
import math
import shutil
import statistics
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "frontend" / "js" / "backtest-comparison.js"
FIXTURE = Path(__file__).parent / "fixtures" / "backtest-comparison.json"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)


def run_helper(expression):
    fixture = FIXTURE.read_text(encoding="utf-8")
    source = SCRIPT.read_text(encoding="utf-8")
    result = subprocess.run(
        [
            "node",
            "-e",
            "\n".join(
                [
                    "global.window = global;",
                    source,
                    f"const fixture = {fixture};",
                    f"console.log(JSON.stringify({expression}));",
                ]
            ),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_model_canonicalizes_shuffled_series_and_calculates_deltas():
    model = run_helper("BacktestComparison.buildModel(fixture.chart, fixture.run)")
    assert [column["key"] for column in model["columns"]] == [
        "agent", "djia", "nasdaq", "buyhold"
    ]
    assert [column["label"] for column in model["columns"]] == [
        "Your Agent", "DJIA", "Nasdaq-100", "Buy & Hold"
    ]
    assert model["agentDeltas"]["djia"] == pytest.approx(0.63)
    assert model["agentDeltas"]["nasdaq"] == pytest.approx(-4.26)
    assert model["agentDeltas"]["buyhold"] == pytest.approx(-0.66)
    assert model["bestByMetric"]["totalReturn"] == ["nasdaq"]


def test_metrics_use_population_stddev_and_hourly_annualization():
    metrics = run_helper(
        "BacktestComparison.calculateMetrics([100, 110, 99, 120])"
    )
    returns = [0.1, -0.1, 120 / 99 - 1]
    expected_sharpe = (
        statistics.fmean(returns)
        / statistics.pstdev(returns)
        * math.sqrt(252 * 6.5)
    )
    assert metrics["finalValue"] == pytest.approx(120.0)
    assert metrics["totalReturn"] == pytest.approx(20.0)
    assert metrics["maxDrawdown"] == pytest.approx(-10.0)
    assert metrics["sharpe"] == pytest.approx(expected_sharpe)


def test_invalid_and_flat_series_are_unavailable_not_zero():
    result = run_helper(
        "({"
        "short: BacktestComparison.calculateMetrics([100]),"
        "flat: BacktestComparison.calculateMetrics([100, 100, 100]),"
        "zeroStart: BacktestComparison.calculateMetrics([0, 10, 12]),"
        "missing: BacktestComparison.calculateMetrics([null, undefined]),"
        "gapped: BacktestComparison.calculateMetrics([100, null, 110]),"
        "nonFinite: BacktestComparison.calculateMetrics([100, Number.NaN, 110])"
        "})"
    )
    assert result["short"] == {
        "finalValue": 100,
        "totalReturn": None,
        "maxDrawdown": None,
        "sharpe": None,
    }
    assert result["flat"]["sharpe"] is None
    assert result["zeroStart"]["totalReturn"] is None
    assert result["missing"] == {
        "finalValue": None,
        "totalReturn": None,
        "maxDrawdown": None,
        "sharpe": None,
    }
    assert result["gapped"]["totalReturn"] == pytest.approx(10.0)
    assert result["nonFinite"]["totalReturn"] == pytest.approx(10.0)


def test_ifind_model_omits_inapplicable_us_indexes():
    keys = run_helper(
        "BacktestComparison.buildModel(fixture.chart, "
        "{...fixture.run, data_source: 'ifind_ashare'}).columns.map(c => c.key)"
    )
    assert keys == ["agent", "buyhold"]


def test_degraded_indexes_and_missing_buyhold_stay_explicitly_unavailable():
    model = run_helper(
        "BacktestComparison.buildModel("
        "{...fixture.chart, index_baselines_ok: false, "
        "series: fixture.chart.series.filter(s => s.run_id !== 'buyhold-fixture')}, "
        "fixture.run)"
    )
    columns = {column["key"]: column for column in model["columns"]}
    assert columns["agent"]["available"] is True
    assert columns["djia"]["available"] is False
    assert columns["nasdaq"]["available"] is False
    assert columns["buyhold"]["available"] is False
    assert model["agentDeltas"] == {
        "djia": None, "nasdaq": None, "buyhold": None
    }


def test_exact_raw_ties_mark_every_tied_series_best():
    best = run_helper(
        "(() => { const agent = fixture.chart.series.find("
        "s => s.run_id === fixture.chart.agent_run_id); "
        "const chart = {...fixture.chart, series: fixture.chart.series"
        ".filter(s => ['agent-fixture', 'index:^DJI'].includes(s.run_id))"
        ".map(s => s.run_id === 'index:^DJI' ? {...s, values: agent.values} : s)}; "
        "return BacktestComparison.buildModel(chart, fixture.run).bestByMetric; })()"
    )
    assert best["totalReturn"] == ["agent", "djia"]
    assert best["maxDrawdown"] == ["agent", "djia"]
```

- [ ] **Step 3: Run the model tests and verify the helper is absent**

Run:

```bash
python -m pytest dashboard/backend/tests/test_backtest_comparison_frontend.py -q
```

Expected: FAIL because `dashboard/frontend/js/backtest-comparison.js` does not exist.

- [ ] **Step 4: Implement the dependency-free helper**

Create the helper as an IIFE. Use these stable public names and return shapes:

```javascript
(function () {
  'use strict';

  const SERIES = Object.freeze({
    agent: { label: 'Your Agent', color: '#4FC3F7' },
    djia: { label: 'DJIA', color: '#F5C04A' },
    nasdaq: { label: 'Nasdaq-100', color: '#9AA4B2' },
    buyhold: { label: 'Buy & Hold', color: '#34D399' },
  });
  const METRICS = Object.freeze([
    { key: 'finalValue', label: 'Final Value', kind: 'currency' },
    { key: 'totalReturn', label: 'Total Return', kind: 'percent' },
    { key: 'maxDrawdown', label: 'Max Drawdown', kind: 'percent' },
    { key: 'sharpe', label: 'Sharpe Ratio', kind: 'ratio' },
  ]);
  const INDEX_IDS = Object.freeze({ 'index:^DJI': 'djia', 'index:^NDX': 'nasdaq' });

  function finiteValues(values) {
    if (!Array.isArray(values)) return [];
    return values.reduce((clean, value) => {
      if (value === null || value === undefined || value === '') return clean;
      const numeric = Number(value);
      if (Number.isFinite(numeric)) clean.push(numeric);
      return clean;
    }, []);
  }

  function calculateMetrics(values) {
    const clean = finiteValues(values);
    const result = {
      finalValue: clean.length ? clean[clean.length - 1] : null,
      totalReturn: null,
      maxDrawdown: null,
      sharpe: null,
    };
    if (clean.length < 2 || clean[0] <= 0) return result;
    result.totalReturn = (clean[clean.length - 1] / clean[0] - 1) * 100;
    let peak = clean[0];
    let maxDrawdown = 0;
    for (const value of clean) {
      peak = Math.max(peak, value);
      maxDrawdown = Math.min(maxDrawdown, value / peak - 1);
    }
    result.maxDrawdown = maxDrawdown * 100;
    const returns = [];
    for (let index = 1; index < clean.length; index += 1) {
      if (clean[index - 1] !== 0) returns.push(clean[index] / clean[index - 1] - 1);
    }
    if (returns.length < 2) return result;
    const mean = returns.reduce((sum, value) => sum + value, 0) / returns.length;
    const variance = returns.reduce(
      (sum, value) => sum + (value - mean) ** 2,
      0,
    ) / returns.length;
    const deviation = Math.sqrt(variance);
    if (deviation > 0) result.sharpe = mean / deviation * Math.sqrt(252 * 6.5);
    return result;
  }

  function classify(entry, payload, run) {
    if (entry?.run_id === payload?.agent_run_id) return 'agent';
    if (INDEX_IDS[entry?.run_id]) return INDEX_IDS[entry.run_id];
    if (entry?.run_id === run?.baseline_buyhold_run_id) return 'buyhold';
    const label = String(entry?.label || '').toLowerCase();
    if (label === 'djia index' || label === 'djia') return 'djia';
    if (label === 'nasdaq-100' || label === 'nasdaq 100') return 'nasdaq';
    if (label === 'buy-and-hold' || label === 'buy & hold') return 'buyhold';
    return null;
  }

  function buildModel(payload, run) {
    const desired = run?.data_source === 'ifind_ashare'
      ? ['agent', 'buyhold']
      : ['agent', 'djia', 'nasdaq', 'buyhold'];
    const indexBaselinesOk = payload?.index_baselines_ok !== false;
    const found = new Map();
    for (const entry of payload?.series || []) {
      const key = classify(entry, payload, run);
      if (key && !found.has(key)) found.set(key, entry);
    }
    const columns = desired.map((key) => {
      const candidate = found.get(key) || null;
      const entry = !indexBaselinesOk && (key === 'djia' || key === 'nasdaq')
        ? null
        : candidate;
      return {
        key,
        label: SERIES[key].label,
        color: entry?.color || SERIES[key].color,
        available: Boolean(entry),
        runId: entry?.run_id || null,
        values: entry?.values || [],
        metrics: entry ? calculateMetrics(entry.values) : calculateMetrics([]),
        dashed: Boolean(entry?.dashed),
      };
    });
    const bestByMetric = Object.fromEntries(METRICS.map(({ key }) => {
      const finite = columns.filter((column) => Number.isFinite(column.metrics[key]));
      if (!finite.length) return [key, []];
      const best = Math.max(...finite.map((column) => column.metrics[key]));
      return [key, finite.filter((column) => column.metrics[key] === best).map((column) => column.key)];
    }));
    const agent = columns.find((column) => column.key === 'agent');
    const agentDeltas = Object.fromEntries(columns
      .filter((column) => column.key !== 'agent')
      .map((column) => [
        column.key,
        Number.isFinite(agent?.metrics.totalReturn)
          && Number.isFinite(column.metrics.totalReturn)
          ? agent.metrics.totalReturn - column.metrics.totalReturn
          : null,
      ]));
    return {
      columns,
      bestByMetric,
      agentDeltas,
      indexBaselinesOk,
    };
  }

  window.BacktestComparison = Object.freeze({ METRICS, buildModel, calculateMetrics });
})();
```

Max Drawdown uses the numerically largest raw value for `Best`, which correctly selects the value closest to zero. Keep labels fixed and do not pass payload labels into visible headings.

- [ ] **Step 5: Run the pure model tests**

```bash
python -m pytest dashboard/backend/tests/test_backtest_comparison_frontend.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the model and fixture**

```bash
git add dashboard/frontend/js/backtest-comparison.js dashboard/backend/tests/fixtures/backtest-comparison.json dashboard/backend/tests/test_backtest_comparison_frontend.py
git commit -m "feat(backtest): calculate benchmark comparison metrics"
```

---

### Task 3: Render the Comparison Table and Accessible Chart Legend

**Files:**
- Modify: `dashboard/frontend/app.html:1231-1261,1300-1327,2394-2403`
- Modify: `dashboard/frontend/app.js:5206-5335,6413-6434,6531-6575,9099-9116,9392-9531`
- Modify: `dashboard/frontend/styles.css:1753-1802`
- Modify: `dashboard/backend/tests/test_backtest_comparison_frontend.py`
- Modify: `dashboard/backend/tests/test_chart_baseline_notice_frontend.py`
- Modify: `dashboard/backend/tests/test_ifind_ashare_frontend.py`
- Modify: `dashboard/backend/tests/test_frontend_fast_boot.py`
- Modify: `dashboard/backend/tests/test_analytics_frontend.py`
- Modify: `dashboard/backend/tests/test_admin_analytics_frontend.py`

**Interfaces:**
- Consumes: `window.BacktestComparison.buildModel(backtestChartData, window.SELECTED_RUN)` from Task 2.
- Produces: `renderPerformanceComparison(payload, run)`, `setPerformanceComparisonState(state, message)`, `renderPerformanceLegend(model)`, and Chart.js datasets carrying `comparisonKey`.

- [ ] **Step 1: Add failing semantic markup and integration tests**

Extend `test_backtest_comparison_frontend.py` using `_frontend_source`:

```python
from dashboard.backend.tests._frontend_source import APP_HTML, STYLES, fn_body


def test_comparison_script_and_semantic_table_ship_before_app():
    helper = '<script src="js/backtest-comparison.js?v=1" defer></script>'
    app = '<script src="app.js?v=117" defer></script>'
    assert 'href="styles.css?v=126"' in APP_HTML
    assert APP_HTML.index(helper) < APP_HTML.index(app)
    for element_id in (
        "performanceLegend",
        "performanceComparison",
        "performanceComparisonHead",
        "performanceComparisonBody",
        "performanceComparisonStatus",
    ):
        assert f'id="{element_id}"' in APP_HTML
    assert 'aria-describedby="performanceComparisonCaption"' in APP_HTML
    assert 'aria-details="performanceComparison"' in APP_HTML
    assert 'role="img"' in APP_HTML


def test_chart_uses_canonical_model_and_dom_legend():
    body = fn_body("function initializeCharts()")
    assert "BacktestComparison.buildModel" in body
    assert "renderPerformanceComparison" in body
    assert "renderPerformanceLegend" in body
    assert "comparisonKey" in body
    assert "display: false" in body


def test_renderers_create_semantic_headers_and_native_legend_buttons():
    comparison = fn_body("function renderPerformanceComparison(")
    legend = fn_body("function renderPerformanceLegend(")
    assert ".scope = 'col'" in comparison
    assert ".scope = 'row'" in comparison
    assert "Best" in comparison
    assert "document.createElement('button')" in legend
    assert "button.type = 'button'" in legend
    assert "aria-pressed" in legend
    assert "setDatasetVisibility" in legend


def test_comparison_states_and_focus_styles_ship():
    assert ".performance-comparison" in STYLES
    assert ".performance-legend-button:focus-visible" in STYLES
    assert ".performance-comparison-scroll:focus-visible" in STYLES
    assert 'data-metric="final-value"' not in APP_HTML
```

Update the degraded-index test to assert that `initializeCharts()` still uses the exact older-payload-safe check and also calls the comparison renderer. Update the iFinD test to expect `BacktestComparison.buildModel` as the applicability boundary; retain structural `index:` filtering as defense in depth for chart datasets.

In the cache/order owners, update `test_frontend_fast_boot.py` to expect `styles.css?v=126`, the helper before `app.js?v=117`, and all page scripts after `app.js?v=117`. Update `test_analytics_frontend.py` to locate `app.js?v=117`, and update the Admin Analytics lifecycle assertions to expect `styles.css?v=126` and `app.js?v=117`.

- [ ] **Step 2: Run the tests and verify the UI contract is absent**

```bash
python -m pytest dashboard/backend/tests/test_backtest_comparison_frontend.py dashboard/backend/tests/test_chart_baseline_notice_frontend.py dashboard/backend/tests/test_ifind_ashare_frontend.py dashboard/backend/tests/test_frontend_fast_boot.py dashboard/backend/tests/test_analytics_frontend.py dashboard/backend/tests/test_admin_analytics_frontend.py -q
```

Expected: FAIL on the missing helper/script tag, markup, renderer, styles, and old `app.js?v=116` / `styles.css?v=125` references.

- [ ] **Step 3: Add semantic comparison and DOM legend hosts**

Change the stylesheet revision to `styles.css?v=126`, then load the pure helper immediately before the revised `app.js?v=117` tag:

```html
<link rel="stylesheet" href="styles.css?v=126">
...
<script src="js/backtest-comparison.js?v=1" defer></script>
<script src="app.js?v=117" defer></script>
<script src="js/analytics.js?v=1" defer></script>
```

Keep every page script after `app.js` in its existing relative order. Place these elements inside `.performance-card` after the benchmark notice and around the canvas/table:

```html
<div id="performanceLegend" class="performance-legend" role="group" aria-label="Chart series visibility"></div>
<div class="chart-container">
    <canvas id="performanceChart" role="img" aria-label="Portfolio value comparison chart" aria-describedby="performanceComparisonCaption" aria-details="performanceComparison"></canvas>
</div>
<section id="performanceComparison" class="performance-comparison" aria-labelledby="performanceComparisonTitle">
    <div class="performance-comparison-header">
        <h3 id="performanceComparisonTitle">Performance comparison</h3>
        <p id="performanceComparisonCaption">Same capital, date range, and sampling interval.</p>
    </div>
    <div class="performance-comparison-scroll" tabindex="0" aria-label="Performance comparison table">
        <table>
            <thead id="performanceComparisonHead"></thead>
            <tbody id="performanceComparisonBody"></tbody>
        </table>
    </div>
    <p id="performanceComparisonStatus" class="performance-comparison-status" role="status" aria-live="polite"></p>
</section>
```

Remove the Agent-only Trading Performance Summary markup. Do not move Trading Log yet; Task 5 changes its structure and location in one reviewable edit.

- [ ] **Step 4: Replace Agent-only metric rendering with the model renderer**

Remove `loadPerformanceMetrics()`, `displayPerformanceMetrics()`, and the selector-based `displayNoMetrics()` implementation. Replace every existing caller in the same change so no removed symbol survives:

- `onBacktestAgentSelectChange()`: remove the extra `loadPerformanceMetrics()` after `await loadData()` because `loadData()` owns the selected surface.
- `prepareLiveBacktestView()`, `attachToLiveBacktest()`, and `showBacktestLaunchFailure()`: call `clearPerformanceComparison(...)`; the first two then render the live-only `Your Agent` legend.
- completed-run polling: remove `await loadPerformanceMetrics()` after `await loadData()`.
- `showPlaygroundPanel('backtest')`: remove the second loader call after `loadData()`.
- the no-run branch in `loadData()`: call `clearPerformanceComparison('empty', ...)`; after `initializeCharts()` remove `displayPerformanceMetrics(selectedRun)` because `initializeCharts()` renders the model.

Add stable formatters and state helpers:

```javascript
function formatComparisonMetric(metric, value, currency = 'USD') {
    if (!Number.isFinite(value)) return '—';
    if (metric.kind === 'currency') {
        return new Intl.NumberFormat('en-US', {
            style: 'currency', currency, maximumFractionDigits: 0,
        }).format(value);
    }
    if (metric.kind === 'percent') return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
    return value.toFixed(2);
}

function formatComparisonDelta(value, label) {
    if (!Number.isFinite(value)) return '';
    const sign = value > 0 ? '+' : value < 0 ? '-' : '';
    const css = value > 0 ? 'is-positive' : value < 0 ? 'is-negative' : 'is-neutral';
    return `<span class="performance-delta ${css}">${escapeHtml(label)} ${sign}${Math.abs(value).toFixed(2)}pp</span>`;
}

function setPerformanceComparisonState(state, message = '') {
    const region = document.getElementById('performanceComparison');
    const status = document.getElementById('performanceComparisonStatus');
    if (!region || !status) return;
    region.dataset.state = state;
    status.textContent = message;
}

function clearPerformanceComparison(state = 'empty', message = '') {
    document.getElementById('performanceComparisonHead')?.replaceChildren();
    document.getElementById('performanceComparisonBody')?.replaceChildren();
    renderPerformanceLegend({ columns: [] });
    setPerformanceComparisonState(state, message);
}
```

Implement the renderer with DOM nodes so headings always come from the normalized model, never from `entry.label`:

```javascript
function renderPerformanceComparison(payload, run) {
    const head = document.getElementById('performanceComparisonHead');
    const body = document.getElementById('performanceComparisonBody');
    if (!head || !body) return;
    const model = window.BacktestComparison.buildModel(payload, run);
    const currency = run?.reporting_currency
        || run?.metadata?.reporting_currency
        || 'USD';

    const headerRow = document.createElement('tr');
    const metricHeader = document.createElement('th');
    metricHeader.scope = 'col';
    metricHeader.textContent = 'Metric';
    headerRow.appendChild(metricHeader);
    for (const column of model.columns) {
        const header = document.createElement('th');
        header.scope = 'col';
        header.dataset.seriesKey = column.key;
        const swatch = document.createElement('span');
        swatch.className = 'performance-series-swatch';
        swatch.style.backgroundColor = column.color;
        swatch.setAttribute('aria-hidden', 'true');
        header.append(swatch, document.createTextNode(column.label));
        headerRow.appendChild(header);
    }
    head.replaceChildren(headerRow);

    const rows = window.BacktestComparison.METRICS.map((metric) => {
        const row = document.createElement('tr');
        const rowHeader = document.createElement('th');
        rowHeader.scope = 'row';
        rowHeader.textContent = metric.label;
        row.appendChild(rowHeader);
        for (const column of model.columns) {
            const cell = document.createElement('td');
            const value = column.metrics[metric.key];
            cell.dataset.seriesKey = column.key;
            cell.textContent = formatComparisonMetric(metric, value, currency);
            if (model.bestByMetric[metric.key].includes(column.key)) {
                cell.classList.add('performance-best');
                const best = document.createElement('span');
                best.className = 'performance-best-label';
                best.textContent = 'Best';
                cell.appendChild(best);
            }
            if (metric.key === 'totalReturn' && column.key === 'agent') {
                const deltas = document.createElement('span');
                deltas.className = 'performance-deltas';
                for (const benchmark of model.columns.filter((item) => item.key !== 'agent')) {
                    const html = formatComparisonDelta(
                        model.agentDeltas[benchmark.key],
                        benchmark.label,
                    );
                    if (html) deltas.insertAdjacentHTML('beforeend', html);
                }
                cell.appendChild(deltas);
            }
            row.appendChild(cell);
        }
        return row;
    });
    body.replaceChildren(...rows);

    const missing = model.columns.filter((column) => !column.available);
    const message = missing.length
        ? `${missing.map((column) => column.label).join(', ')} unavailable for this run.`
        : '';
    setPerformanceComparisonState(missing.length ? 'partial' : 'ready', message);
}
```

- [ ] **Step 5: Build DOM legend buttons and canonical Chart.js datasets**

In `initializeCharts()`:

```javascript
const model = window.BacktestComparison.buildModel(backtestChartData, window.SELECTED_RUN);
const chartColumns = model.columns.filter((column) => column.available);
const datasets = chartColumns.map((column) => ({
    label: column.label,
    comparisonKey: column.key,
    data: column.values,
    borderColor: column.color,
    backgroundColor: 'transparent',
    borderWidth: 2.5,
    borderDash: column.dashed ? [6, 4] : [],
    tension: 0,
    fill: false,
    pointRadius: 0,
    pointHoverRadius: 5,
}));
renderPerformanceComparison(backtestChartData, window.SELECTED_RUN);
renderPerformanceLegend(model);
```

Disable the built-in Chart.js legend and add the native DOM legend:

```javascript
function renderPerformanceLegend(model, { disabled = false } = {}) {
    const host = document.getElementById('performanceLegend');
    if (!host) return;
    const available = (model?.columns || []).filter((column) => column.available);
    const buttons = available.map((column, index) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'performance-legend-button';
        button.dataset.datasetIndex = String(index);
        button.setAttribute('aria-pressed', 'true');
        button.disabled = disabled;
        const swatch = document.createElement('span');
        swatch.className = 'performance-series-swatch';
        swatch.style.backgroundColor = column.color;
        swatch.setAttribute('aria-hidden', 'true');
        button.append(swatch, document.createTextNode(column.label));
        button.addEventListener('click', () => {
            if (!chartInstance) return;
            const visible = button.getAttribute('aria-pressed') === 'true';
            chartInstance.setDatasetVisibility(index, !visible);
            button.setAttribute('aria-pressed', String(!visible));
            chartInstance.update();
        });
        return button;
    });
    host.replaceChildren(...buttons);
}
```

The comparison table is not mutated by this handler.

For live runs, render one disabled `Your Agent` legend button with `aria-pressed="true"` and set the comparison state text to `Benchmark metrics will appear when this run finishes.`. Clear stale historical rows in `prepareLiveBacktestView()` and `attachToLiveBacktest()` before the live chart paints.

- [ ] **Step 6: Add the comparison and legend styles**

Add scoped rules with the existing palette:

```css
.performance-legend { display: flex; justify-content: center; flex-wrap: wrap; gap: 6px; }
.performance-legend-button { display: inline-flex; align-items: center; gap: 6px; min-height: 32px; padding: 4px 8px; border: 1px solid transparent; border-radius: 4px; background: transparent; color: var(--text-primary); }
.performance-legend-button[aria-pressed="false"] { color: var(--text-muted); opacity: .7; }
.performance-legend-button:focus-visible,
.performance-comparison-scroll:focus-visible { outline: 2px solid var(--info-color); outline-offset: 2px; }
.performance-series-swatch { width: 14px; height: 3px; border-radius: 2px; }
.performance-comparison { border-top: 1px solid var(--border-color); background: rgba(8, 18, 40, .28); }
.performance-comparison-header { display: flex; justify-content: space-between; gap: 12px; padding: 12px 14px; }
.performance-comparison-scroll { overflow-x: auto; }
.performance-comparison table { width: 100%; min-width: 680px; border-collapse: collapse; font-variant-numeric: tabular-nums; }
.performance-comparison th,
.performance-comparison td { padding: 10px 12px; border-top: 1px solid var(--border-color); text-align: right; }
.performance-comparison th:first-child { text-align: left; }
.performance-comparison [data-series-key="agent"] { background: rgba(0, 191, 255, .06); }
.performance-best { color: var(--success-color); }
.performance-best-label { margin-left: 4px; font-size: 9px; text-transform: uppercase; }
.performance-deltas { display: flex; justify-content: flex-end; flex-wrap: wrap; gap: 4px; margin-top: 4px; }
.performance-delta.is-positive { color: var(--success-color); }
.performance-delta.is-negative { color: var(--danger-color); }
.performance-comparison-status:empty { display: none; }
```

- [ ] **Step 7: Run the comparison integration tests**

```bash
python -m pytest dashboard/backend/tests/test_backtest_comparison_frontend.py dashboard/backend/tests/test_chart_baseline_notice_frontend.py dashboard/backend/tests/test_ifind_ashare_frontend.py dashboard/backend/tests/test_frontend_fast_boot.py dashboard/backend/tests/test_analytics_frontend.py dashboard/backend/tests/test_admin_analytics_frontend.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit the comparison surface**

```bash
git add dashboard/frontend/app.html dashboard/frontend/app.js dashboard/frontend/styles.css dashboard/backend/tests/test_backtest_comparison_frontend.py dashboard/backend/tests/test_chart_baseline_notice_frontend.py dashboard/backend/tests/test_ifind_ashare_frontend.py dashboard/backend/tests/test_frontend_fast_boot.py dashboard/backend/tests/test_analytics_frontend.py dashboard/backend/tests/test_admin_analytics_frontend.py
git commit -m "feat(ui): compare backtest performance with benchmarks"
```

---

### Task 4: Isolate Historical Chart and Trading Log Loading

**Files:**
- Modify: `dashboard/frontend/app.js:4897-4916,7252-7277,9279-9385`
- Modify: `dashboard/backend/tests/test_backtest_comparison_frontend.py`
- Modify: `dashboard/backend/tests/test_trading_log_order_events_ui.py`

**Interfaces:**
- Consumes: selected run id, existing `API.get`, `initializeCharts()`, and `loadTradingLogForRun()`.
- Produces: `beginBacktestSurfaceRequest(runId)`, `isCurrentBacktestSurfaceRequest(token)`, and `loadHistoricalBacktestSurfaces(selectedRun)`; `loadTradingLogForRun(runId, {isCurrent})` paints only when current.

- [ ] **Step 1: Write failing stale-response and partial-failure tests**

Add these extraction/runtime helpers to `test_backtest_comparison_frontend.py` so the test executes the shipped functions instead of copies:

```python
def extract_app_function(source, name):
    for marker in (f"async function {name}(", f"function {name}("):
        start = source.find(marker)
        if start != -1:
            break
    else:
        raise AssertionError(f"{name} not found in app.js")
    depth = 0
    index = source.index("{", start)
    while index < len(source):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
        index += 1
    raise AssertionError(f"unterminated function {name}")


def run_app_functions(source, names, harness_lines):
    script = "\n".join(
        [*(extract_app_function(source, name) for name in names), *harness_lines]
    )
    result = subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)
```

Then add a Node test for the token helpers:

```python
def test_request_token_rejects_previous_run_after_selection_changes():
    source = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    result = run_app_functions(
        source,
        ["beginBacktestSurfaceRequest", "isCurrentBacktestSurfaceRequest"],
        [
            "global.window = { SELECTED_RUN: { run_id: 'run-a' } };",
            "let liveBacktestChartActive = false;",
            "let backtestSurfaceRequestSeq = 0;",
            "const first = beginBacktestSurfaceRequest('run-a');",
            "window.SELECTED_RUN = { run_id: 'run-b' };",
            "const second = beginBacktestSurfaceRequest('run-b');",
            "console.log(JSON.stringify({",
            "  first: isCurrentBacktestSurfaceRequest(first),",
            "  second: isCurrentBacktestSurfaceRequest(second),",
            "}));",
        ],
    )
    assert result == {"first": False, "second": True}
```

Add source-contract assertions:

```python
def test_historical_surfaces_settle_independently():
    body = fn_body("async function loadHistoricalBacktestSurfaces(")
    assert "Promise.allSettled" in body
    assert "setPerformanceComparisonState('error'" in body
    assert "loadTradingLogForRun" in body
    assert "isCurrentBacktestSurfaceRequest" in body
```

Extend the Trading Log test harness to prove `loadTradingLogForRun()` ignores a resolved response when `isCurrent()` returns false and does not overwrite the existing feed.

- [ ] **Step 2: Run the tests and verify sequential loading still fails the contract**

```bash
python -m pytest dashboard/backend/tests/test_backtest_comparison_frontend.py dashboard/backend/tests/test_trading_log_order_events_ui.py -q
```

Expected: FAIL because the token functions and `Promise.allSettled` coordinator do not exist.

- [ ] **Step 3: Add the request token and current-run guard**

Near `backtestChartData`, add:

```javascript
let backtestSurfaceRequestSeq = 0;

function beginBacktestSurfaceRequest(runId) {
    return { seq: ++backtestSurfaceRequestSeq, runId };
}

function isCurrentBacktestSurfaceRequest(token) {
    return token?.seq === backtestSurfaceRequestSeq
        && token.runId === window.SELECTED_RUN?.run_id
        && !liveBacktestChartActive;
}
```

Increment `backtestSurfaceRequestSeq` when entering or attaching to a live run so late historical promises cannot repaint the live surface.

- [ ] **Step 4: Coordinate both requests without coupling their failures**

Extract the historical fetch branch from `loadData()`:

```javascript
async function loadHistoricalBacktestSurfaces(selectedRun) {
    const token = beginBacktestSurfaceRequest(selectedRun.run_id);
    setPerformanceComparisonState('loading', 'Loading performance comparison…');
    clearTradingLog('Loading orders…');
    const chartUrl = `${API_BASE}/api/backtest/${encodeURIComponent(selectedRun.run_id)}/chart-data?t=${Date.now()}`;

    const chartRequest = API.get(chartUrl).then((payload) => {
        if (!isCurrentBacktestSurfaceRequest(token)) return;
        backtestChartData = payload;
        initializeCharts();
    }).catch((error) => {
        if (!isCurrentBacktestSurfaceRequest(token)) return;
        backtestChartData = null;
        if (chartInstance) {
            chartInstance.destroy();
            chartInstance = null;
        }
        renderPerformanceLegend({ columns: [] });
        setPerformanceComparisonState(
            'error',
            'Performance comparison is unavailable. Reload to retry.',
        );
        console.warn('Could not load performance comparison:', error.message);
    });

    const logRequest = loadTradingLogForRun(selectedRun.run_id, {
        isCurrent: () => isCurrentBacktestSurfaceRequest(token),
    });
    await Promise.allSettled([chartRequest, logRequest]);
}
```

Call this function after rendering Run config. Remove the sequential chart fetch, `displayPerformanceMetrics(selectedRun)`, and awaited log call from `loadData()`.

Change `loadTradingLogForRun` to accept `{isCurrent = () => true}` and guard both success and catch painting. It continues to own only the Trading Log error text.

- [ ] **Step 5: Run loading and existing launch/live tests**

```bash
python -m pytest dashboard/backend/tests/test_backtest_comparison_frontend.py dashboard/backend/tests/test_trading_log_order_events_ui.py dashboard/backend/tests/test_backtest_launch_visibility.py dashboard/backend/tests/test_backtest_progress_card.py dashboard/backend/tests/test_backtest_progress_status.py -q
```

Expected: PASS. A chart failure leaves the log result untouched; a log failure leaves the comparison untouched; an older run cannot repaint a newer or live run.

- [ ] **Step 6: Commit the independent loading flow**

```bash
git add dashboard/frontend/app.js dashboard/backend/tests/test_backtest_comparison_frontend.py dashboard/backend/tests/test_trading_log_order_events_ui.py
git commit -m "fix(backtest): isolate comparison and order loading"
```

---

### Task 5: Move Trading Log Right and Render a Vertical Activity Feed

**Files:**
- Modify: `dashboard/frontend/app.html:1263-1327`
- Modify: `dashboard/frontend/app.js:4898-4903,4991-5001,7173-7277`
- Modify: `dashboard/frontend/styles.css:1808-2030,2413-2431,2561-2597`
- Modify: `dashboard/backend/tests/test_trading_log_order_events_ui.py`
- Modify: `dashboard/backend/tests/test_backtest_comparison_frontend.py`

**Interfaces:**
- Consumes: existing normalized order records and audit renderers from `normalizeOrderRecord()`, `renderOrderCostAudit()`, and `renderMarketRuleAudit()`.
- Produces: `#tradingLogFeed[role=list]`, `#tradingLogCount`, `#tradingLogStatusSummary`, and `paintTradingLog()` output as complete `article[role=listitem]` events.

- [ ] **Step 1: Rewrite the failing markup contract and extend behavioral assertions**

Replace the eight-column assertion with:

```python
def test_trading_log_markup_is_an_accessible_right_rail_feed():
    html = _APP_HTML.read_text(encoding="utf-8")
    center_end = html.index('</section>', html.index('class="center-panel"'))
    log_at = html.index('id="tradingLogFeed"')
    right_at = html.index('class="right-panel"')
    assert right_at < log_at
    assert log_at > center_end
    assert 'id="tradingLogFeed"' in html
    assert 'role="list"' in html
    assert 'tabindex="0"' in html
    assert 'aria-label="Trading Log orders"' in html
    assert 'id="tradingLogCount"' in html
    assert 'id="tradingLogStatusSummary"' in html
    assert 'class="trading-log-table"' not in html
```

Update `_render_harness()` so `document.getElementById()` returns a feed plus count/summary text nodes. Keep every existing behavior test. Add assertions that a rendered event contains `role="listitem"`, action, asset, timestamp, quantity, price, value, status, reason, native currency, costs, market-rule audit, repeat count, and the truncation notice.

- [ ] **Step 2: Run the Trading Log tests and verify the wide table is still shipped**

```bash
python -m pytest dashboard/backend/tests/test_trading_log_order_events_ui.py -q
```

Expected: FAIL on the old table markup and `<tr>` renderer.

- [ ] **Step 3: Move the log card and replace the table host**

Inside `.right-panel`, replace the removed summary with:

```html
<section class="section-card trading-log-card" aria-labelledby="tradingLogTitle">
    <div class="section-header trading-log-header">
        <div>
            <h2 id="tradingLogTitle">Trading Log</h2>
            <p class="trading-log-summary"><span id="tradingLogCount">0 orders</span><span id="tradingLogStatusSummary"></span></p>
        </div>
        <div class="log-controls">
            <label class="sr-only" for="tradingLogFilter">Filter Trading Log orders</label>
            <select class="filter-select" id="tradingLogFilter">
                <option value="all">All Orders</option>
                <option value="buy">Buys Only</option>
                <option value="sell">Sells Only</option>
            </select>
        </div>
    </div>
    <div id="tradingLogFeed" class="trading-log-scroll" role="list" tabindex="0" aria-label="Trading Log orders">
        <p class="trading-log-empty">Run a backtest to see orders here.</p>
    </div>
</section>
```

Remove the former Trading Log section from `.center-panel`.

- [ ] **Step 4: Render complete vertical order events**

Change `paintTradingLog()` to target `tradingLogFeed`. Keep filtering before summary counts. Render each record with this hierarchy:

```javascript
return `<article class="trading-log-event" role="listitem">
    <div class="trading-log-event-head">
        <span class="trading-log-action ${actionClass}">${actionLabel}</span>
        <div class="trading-log-asset"><strong>${escapeHtml(order.symbol)}</strong>${assetName ? `<small>${escapeHtml(assetName)}</small>` : ''}<time>${escapeHtml(formatTradeTimestamp(order.timestamp))}</time></div>
        <span class="order-status order-status-${order.status}" aria-label="Order status: ${statusLabel}">${statusLabel}</span>
    </div>
    <div class="trading-log-event-meta">
        <span><small>Filled / requested</small>${escapeHtml(quantity)}</span>
        <span><small>Execution price</small>${formatTradingMoney(order.price, '$')}${priceAudit}</span>
        <span><small>Filled value</small>${order.executedShares > 0 ? `${formatTradingMoney(order.value, '$')}${valueAudit}` : '--'}</span>
    </div>
    ${costAudit}
    <p class="trading-log-reason">${escapeHtml(reason)}${marketRuleAudit}${repeatNote}</p>
</article>`;
```

Call `renderTradingLogSummary(filtered)` immediately after applying the All/Buy/Sell filter, before the empty branch, so zero-result filters also update the count. Use paragraph messages instead of table rows for empty and truncation states. The summary sets the pluralized visible count and a `filled / partial / rejected` breakdown. Continue storing normalized records once and reusing them on filter changes.

- [ ] **Step 5: Style the independent right-rail feed and responsive placement**

Scope the layout to Backtest so other `.main-container` pages do not change:

```css
@media (min-width: 1280px) {
    .playground-backtest-panel.main-container { grid-template-columns: minmax(220px, 250px) minmax(620px, 1fr) minmax(320px, 360px); }
}
.trading-log-card { display: grid; grid-template-rows: auto minmax(0, 1fr); max-height: 720px; min-width: 0; }
.trading-log-header { align-items: flex-start; }
.trading-log-summary { display: flex; flex-wrap: wrap; gap: 4px 10px; margin-top: 5px; color: var(--text-muted); font-size: 10px; }
.trading-log-scroll { min-height: 0; overflow-y: auto; overflow-x: hidden; border-top: 1px solid var(--border-color); }
.trading-log-scroll:focus-visible { outline: 2px solid var(--info-color); outline-offset: -2px; }
.trading-log-event { padding: 12px; border-bottom: 1px solid var(--border-color); }
.trading-log-event-head { display: grid; grid-template-columns: max-content minmax(0, 1fr) max-content; gap: 8px; align-items: start; }
.trading-log-event-meta { display: flex; flex-wrap: wrap; gap: 4px 10px; margin: 8px 0 0 46px; color: var(--text-secondary); font-variant-numeric: tabular-nums; }
.trading-log-event-meta > span { display: grid; gap: 2px; min-width: 86px; }
.trading-log-event-meta small { color: var(--text-muted); font-size: 9px; }
.trading-log-event .trading-log-reason { margin: 8px 0 0 46px; max-width: none; overflow-wrap: anywhere; }
@media (max-width: 1279px) {
    .playground-backtest-panel.main-container { grid-template-columns: 1fr; }
    .playground-backtest-panel .right-panel { display: flex; }
    .trading-log-card { max-height: 560px; }
}
@media (max-width: 600px) {
    .trading-log-card { max-height: none; }
    .trading-log-scroll { overflow-y: visible; }
    .trading-log-event-meta,
    .trading-log-event .trading-log-reason { margin-left: 0; }
}
```

Reconcile these scoped rules with the existing generic 1200/900 pixel rules so Backtest ordering is Run config, performance, Trading Log and no `.right-panel { display:none }` rule wins at mobile widths.

- [ ] **Step 6: Run Trading Log and layout contracts**

```bash
python -m pytest dashboard/backend/tests/test_trading_log_order_events_ui.py dashboard/backend/tests/test_backtest_comparison_frontend.py -q
```

Expected: PASS, including every pre-existing audit-detail test.

- [ ] **Step 7: Commit the approved layout**

```bash
git add dashboard/frontend/app.html dashboard/frontend/app.js dashboard/frontend/styles.css dashboard/backend/tests/test_trading_log_order_events_ui.py dashboard/backend/tests/test_backtest_comparison_frontend.py
git commit -m "feat(ui): move Trading Log beside backtest results"
```

---

### Task 6: Run Full Regression and Visual Verification

**Files:**
- Verify only: no source, fixture, screenshot, or test file is modified in this task.

**Interfaces:**
- Consumes: all prior task outputs, including the Task 3 asset revisions `styles.css?v=126`, `js/backtest-comparison.js?v=1`, and `app.js?v=117`.
- Produces: passing focused/full suites and recorded viewport/degraded-state evidence; no additional commit.

- [ ] **Step 1: Run the complete focused regression set**

```bash
python -m pytest dashboard/backend/tests/test_backtest_chart_data.py dashboard/backend/tests/test_chart_style.py dashboard/backend/tests/test_equity_plot.py dashboard/backend/tests/test_backtest_comparison_frontend.py dashboard/backend/tests/test_trading_log_order_events_ui.py dashboard/backend/tests/test_chart_baseline_notice_frontend.py dashboard/backend/tests/test_ifind_ashare_frontend.py dashboard/backend/tests/test_frontend_fast_boot.py dashboard/backend/tests/test_analytics_frontend.py dashboard/backend/tests/test_admin_analytics_frontend.py dashboard/backend/tests/test_backtest_launch_visibility.py dashboard/backend/tests/test_backtest_progress_card.py dashboard/backend/tests/test_backtest_progress_status.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the full backend/frontend contract suite**

```bash
python -m pytest dashboard/backend/tests/ --timeout=180 -p no:cacheprovider
```

Expected: PASS with no test reading or modifying `dashboard/storage/data/backtest.db`.

- [ ] **Step 3: Start the local app for visual verification**

Run on an unused local port:

```bash
python -m uvicorn dashboard.backend.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/app?view=backtest`. Use the committed synthetic fixture to mock only these authenticated reads in the browser session:

- `/api/backtest/runs` returns an array containing `fixture.run` plus `initial_equity`, `final_equity`, `total_return`, `max_drawdown`, `sharpe_ratio`, `start_date`, `end_date`, and `created_at` synthetic fields;
- `/api/backtest/agent-fixture/chart-data` returns `fixture.chart`;
- `/runs/agent-fixture/trades` returns `fixture.trades`; and
- `/backtest/status` returns `{"running": false}`.

Do not point the browser at a real database or copy a production response.

- [ ] **Step 4: Capture and inspect three viewport screenshots**

Using the browser/Playwright tooling available to the executing agent, capture:

1. 1440 x 1000: Run config left, chart/comparison center, Trading Log right; right rail scrolls without moving the page.
2. 1024 x 900: Trading Log is below performance; comparison has contained horizontal scrolling; no page-level horizontal overflow.
3. 390 x 844: single column, nonblank chart, readable controls, contained comparison scroll, natural-height log, no overlap.

For each viewport, also verify canvas pixels are nonblank, all four US legend labels render, Agent deltas show `pp`, keyboard focus is visible on legend/filter/scroll regions, toggling a legend button leaves its comparison column visible, and the full order reason is present.

- [ ] **Step 5: Inspect degraded states with the same safe fixture**

Repeat the desktop check with:

- `index_baselines_ok: false` and both index series removed: Agent and Buy & Hold remain, DJIA/Nasdaq cells say unavailable, and the index notice is visible;
- the Buy & Hold series removed: its column says unavailable with no fabricated Agent delta;
- chart-data returning 503: Trading Log remains populated while performance shows its error;
- trades returning 503: performance remains populated while only Trading Log shows its error; and
- two rapid run selections resolved out of order: only the final selection is visible.

- [ ] **Step 6: Verify repository hygiene**

```bash
git diff --check
git status --short --untracked-files=all
git status --short dashboard/storage/data/backtest.db .superpowers work
```

Expected: `git diff --check` passes; the database, `.superpowers/`, and `work/` status command prints nothing; no screenshot or browser-session artifact is present. Task 6 creates no source changes and no commit.
