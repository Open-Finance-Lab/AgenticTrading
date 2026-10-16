/** Read-only trace chart in the Backtest panel's shape: agent vs benchmarks, legend, range, comparison. */
(function () {
  'use strict';
  const svgNS = 'http://www.w3.org/2000/svg';
  // Ranges count trading days present in the series, not bars or calendar
  // time: a 5-minute run and an hourly one both show one session for 1D, and
  // a market with a different session length needs no constant of its own.
  const RANGE_DAYS = { '1D': 1, '1W': 5, ALL: Infinity };
  function tradingDay(ts) {
    return String(ts).slice(0, 10);
  }
  // Index of the first bar inside the last `days` trading days.
  function rangeStart(timestamps, range) {
    const days = RANGE_DAYS[range] ?? Infinity;
    if (!Number.isFinite(days)) return 0;
    let seen = 0;
    let current = null;
    for (let i = timestamps.length - 1; i >= 0; i -= 1) {
      const day = tradingDay(timestamps[i]);
      if (day !== current) {
        seen += 1;
        if (seen > days) return i + 1;
        current = day;
      }
    }
    return 0;
  }
  // Decision snapshots are the fallback curve when a run stored no equity, so
  // a failed or still-running trace keeps a chart.
  function series(events, performance) {
    const snapshots = (Array.isArray(events) ? events : []).filter(e => e.event_type === 'decision_recorded')
      .map(e => ({ timestamp: e.payload?.decision_at, equity: e.payload?.state?.equity, event: e }));
    const raw = performance?.points?.length ? performance.points : snapshots;
    const points = raw.filter(p => typeof p.equity === 'number' && Number.isFinite(p.equity) && Number.isFinite(Date.parse(p.timestamp)))
      .map(p => ({ ...p, time: Date.parse(p.timestamp) })).sort((a, b) => a.time - b.time);
    return { points, snapshots: raw === snapshots };
  }
  // Columns come from BacktestComparison (the panel's own model) when the
  // run's chart payload is available; otherwise one agent column from the
  // recorded points, so the legend and the table still describe the curve.
  function columns(chart, points) {
    if (chart?.available && window.BacktestComparison) {
      const model = window.BacktestComparison.buildModel(chart, chart.run || {});
      return { model, timestamps: chart.timestamps || [], cols: model.columns.filter(c => c.available) };
    }
    const values = points.map(p => p.equity);
    const metrics = window.BacktestComparison ? window.BacktestComparison.calculateMetrics(values) : {};
    return { model: null, timestamps: points.map(p => p.timestamp), cols: [{ key: 'agent', label: 'Your Agent', color: '#4FC3F7', available: true, values, metrics, dashed: false }] };
  }
  function money(value) {
    return Number.isFinite(value) ? new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(value) : window.AdminShell.DASH;
  }
  function percent(value) {
    return Number.isFinite(value) ? `${value >= 0 ? '+' : ''}${value.toFixed(2)}%` : window.AdminShell.DASH;
  }
  function renderChart(chart, data, view, events, onDecision) {
    while (chart.firstChild) chart.removeChild(chart.firstChild);
    const visible = data.cols.filter(c => !view.hidden.has(c.key));
    const total = data.timestamps.length;
    const from = rangeStart(data.timestamps, view.range);
    const values = visible.flatMap(c => c.values.slice(from).filter(Number.isFinite));
    if (!values.length || total - from < 2) return;
    const min = values.reduce((v, x) => Math.min(v, x), Infinity), max = values.reduce((v, x) => Math.max(v, x), -Infinity);
    const pad = (max - min) * 0.08 || Math.max(Math.abs(max) * .01, 1);
    const lo = min - pad, hi = max + pad;
    const n = total - from;
    const x = i => 70 + i / (n - 1) * 810;
    const y = v => 250 - (v - lo) / (hi - lo) * 225;
    function shape(tag, attrs, text) {
      const node = document.createElementNS(svgNS, tag);
      Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, String(v)));
      if (text !== undefined) node.textContent = text;
      chart.appendChild(node); return node;
    }
    for (let k = 0; k <= 4; k += 1) {
      const v = lo + (hi - lo) * k / 4;
      shape('line', { x1: 70, x2: 880, y1: y(v), y2: y(v), stroke: 'currentColor', opacity: '.1' });
      shape('text', { x: 62, y: y(v) + 4, 'text-anchor': 'end', fill: 'currentColor', 'font-size': 11 }, money(v));
    }
    // One label per trading day, at its first bar.
    let lastDay = '';
    data.timestamps.slice(from).forEach((ts, i) => {
      const day = tradingDay(ts);
      if (day !== lastDay) { lastDay = day; shape('text', { x: x(i), y: 285, 'text-anchor': i ? 'middle' : 'start', fill: 'currentColor', 'font-size': 11 }, day); }
    });
    // Benchmarks first, so the agent line draws on top of them.
    [...visible].reverse().forEach(c => {
      const pts = c.values.slice(from).map((v, i) => Number.isFinite(v) ? `${x(i)},${y(v)}` : null).filter(Boolean);
      shape('polyline', { points: pts.join(' '), fill: 'none', stroke: c.color, 'stroke-width': c.key === 'agent' ? 2.2 : 1.6, 'stroke-dasharray': c.dashed ? '6 4' : '' });
    });
    if (view.hidden.has('agent')) return;
    // A marker sits at the decision's bar on the agent curve: the decision
    // snapshot, not equity at fill time.
    const times = data.timestamps.map(t => Date.parse(t));
    const agent = data.cols.find(c => c.key === 'agent');
    const decisions = new Map(events.filter(e => e.decision_id && e.event_type === 'decision_recorded').map(e => [e.decision_id, e]));
    events.filter(e => e.event_type === 'execution_result' && (e.payload?.fills?.length || e.payload?.rejected?.length)).forEach(e => {
      const decision = decisions.get(e.decision_id);
      const at = Date.parse(decision?.payload?.decision_at);
      if (!decision || !Number.isFinite(at)) return;
      let index = -1;
      for (let i = from; i < total; i += 1) { if (times[i] <= at) index = i; else break; }
      if (index < from || !Number.isFinite(agent.values[index])) return;
      const fills = e.payload.fills || [];
      const label = fills.length ? fills.map(f => `${f.side} ${f.symbol}`).join(', ') : 'Rejected orders';
      const when = window.AdminShell.formatTimestamp(decision.payload.decision_at);
      const marker = shape('circle', { cx: x(index - from), cy: y(agent.values[index]), r: 5, fill: '#fbbf24', stroke: '#0a0e27', 'stroke-width': 1.5, tabindex: 0, role: 'button', class: 'trace-marker', 'aria-label': `Decision ${when}: ${label}` });
      const title = document.createElementNS(svgNS, 'title');
      title.textContent = `${when} · ${label}`;
      marker.appendChild(title);
      marker.addEventListener('click', () => onDecision(decision));
      marker.addEventListener('keydown', ev => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); onDecision(decision); } });
    });
  }
  function renderComparison(data) {
    const el = window.AdminShell.el;
    const section = el('section', 'trace-comparison');
    section.setAttribute('aria-label', 'Performance comparison');
    const head = el('div', 'trace-comparison-head');
    head.appendChild(el('h3', '', 'Performance comparison'));
    head.appendChild(el('p', '', data.model ? 'Same capital, date range, and sampling interval.' : 'Benchmarks unavailable for this run; recorded equity only.'));
    section.appendChild(head);
    const scroll = el('div', 'trace-comparison-scroll');
    scroll.setAttribute('tabindex', '0');
    scroll.setAttribute('aria-label', 'Performance comparison table');
    const table = el('table');
    const thead = el('thead');
    const hr = el('tr');
    const metricHead = el('th', '', 'Metric');
    metricHead.setAttribute('scope', 'col');
    hr.appendChild(metricHead);
    data.cols.forEach(c => {
      const th = el('th');
      th.setAttribute('scope', 'col');
      th.dataset.seriesKey = c.key;
      const sw = el('span', 'trace-swatch');
      sw.style.setProperty('background', c.color);
      sw.setAttribute('aria-hidden', 'true');
      th.appendChild(sw); th.appendChild(document.createTextNode(c.label));
      hr.appendChild(th);
    });
    thead.appendChild(hr);
    // Sharpe is left out on purpose: the 2026-10-14 direction has the engine
    // compute it once, with its interval, and never the frontend.
    const metrics = [['finalValue', 'Final value', money], ['totalReturn', 'Total return', percent], ['maxDrawdown', 'Max drawdown', percent]];
    const body = el('tbody');
    metrics.forEach(([key, label, format]) => {
      const tr = el('tr');
      const th = el('th', '', label);
      th.setAttribute('scope', 'row');
      tr.appendChild(th);
      const finite = data.cols.filter(c => Number.isFinite(c.metrics[key]));
      const best = data.cols.length > 1 && finite.length ? Math.max(...finite.map(c => c.metrics[key])) : null;
      data.cols.forEach(c => {
        const td = el('td');
        td.dataset.seriesKey = c.key;
        td.appendChild(el('span', '', format(c.metrics[key])));
        if (best !== null && c.metrics[key] === best) td.appendChild(el('span', 'trace-best', 'Best'));
        if (key === 'totalReturn' && c.key === 'agent' && data.model) {
          data.cols.filter(o => o.key !== 'agent').forEach(o => {
            const diff = data.model.agentDeltas[o.key];
            if (Number.isFinite(diff)) td.appendChild(el('span', `trace-delta ${diff >= 0 ? 'is-up' : 'is-down'}`, `${o.label} ${diff >= 0 ? '+' : ''}${diff.toFixed(2)}pp`));
          });
        }
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });
    table.appendChild(thead); table.appendChild(body);
    scroll.appendChild(table);
    section.appendChild(scroll);
    return section;
  }
  // `view` ({range, hidden}) is owned by the caller so a poll re-render keeps
  // the admin's range and hidden series.
  function render(trace, events, performance, chartData = null, options = {}) {
    const el = window.AdminShell.el;
    const items = Array.isArray(events) ? events : [];
    const view = options.view || { range: 'ALL', hidden: new Set() };
    const onDecision = options.onDecision || (decision => window.AdminTraceTimeline?.reveal(decision.sequence_no));
    const panel = el('section', 'trace-performance');
    const head = el('div', 'trace-perf-head');
    const titles = el('div');
    titles.appendChild(el('h2', '', 'Trading performance'));
    const { points, snapshots } = series(items, performance);
    titles.appendChild(el('p', 'muted', snapshots
      ? 'Decision equity snapshots · market time (UTC). Sampled observations, not the complete backtest result.'
      : 'Run equity · market time (UTC). One observed outcome, not expected return.'));
    head.appendChild(titles);
    const range = el('div', 'trace-range');
    range.setAttribute('role', 'group'); range.setAttribute('aria-label', 'Chart range');
    head.appendChild(range);
    panel.appendChild(head);
    if (trace?.status !== 'completed') panel.appendChild(el('p', 'trace-performance-notice',
      trace?.status === 'failed' ? 'Run failed. Recorded performance is incomplete; inspect the failure in the timeline.' : 'Run in progress. Recorded performance is incomplete.'));
    if (performance?.unavailable) panel.appendChild(el('p', 'muted', 'Performance refresh unavailable. Showing recorded snapshots or previously loaded data.'));
    if (chartData?.available && chartData.index_baselines_ok === false) panel.appendChild(el('p', 'trace-performance-notice', 'Index benchmarks unavailable: the market-data provider could not be reached. Reload to retry.'));
    const data = columns(chartData, points);
    if (data.timestamps.length < 2) {
      panel.appendChild(el('p', 'panel-empty', data.timestamps.length ? 'Only one equity observation is available; a curve requires two.' : 'No equity observations recorded for this run.'));
      return panel;
    }
    const legend = el('div', 'trace-legend');
    legend.setAttribute('role', 'group'); legend.setAttribute('aria-label', 'Chart series visibility');
    panel.appendChild(legend);
    const chart = document.createElementNS(svgNS, 'svg');
    chart.setAttribute('viewBox', '0 0 900 300'); chart.setAttribute('role', 'img');
    chart.setAttribute('class', 'trace-chart');
    chart.setAttribute('aria-label', 'Portfolio value over historical market time, agent and benchmarks');
    panel.appendChild(chart);
    const paint = () => renderChart(chart, data, view, items, onDecision);
    Object.keys(RANGE_DAYS).forEach(key => {
      const button = el('button', `time-btn${view.range === key ? ' active' : ''}`, key);
      button.setAttribute('type', 'button');
      button.setAttribute('aria-pressed', String(view.range === key));
      button.addEventListener('click', () => {
        view.range = key;
        Array.from(range.children).forEach(b => { const on = b.textContent === key; b.classList.toggle('active', on); b.setAttribute('aria-pressed', String(on)); });
        paint();
      });
      range.appendChild(button);
    });
    data.cols.forEach(c => {
      const button = el('button', 'trace-legend-btn');
      button.setAttribute('type', 'button');
      button.setAttribute('aria-pressed', String(!view.hidden.has(c.key)));
      const sw = el('span', `trace-swatch${c.dashed ? ' trace-swatch--dash' : ''}`);
      sw.style.setProperty('background', c.color);
      sw.setAttribute('aria-hidden', 'true');
      button.appendChild(sw); button.appendChild(document.createTextNode(c.label));
      button.addEventListener('click', () => {
        if (view.hidden.has(c.key)) view.hidden.delete(c.key); else view.hidden.add(c.key);
        button.setAttribute('aria-pressed', String(!view.hidden.has(c.key)));
        paint();
      });
      legend.appendChild(button);
    });
    paint();
    panel.appendChild(el('p', 'trace-chart-hint', 'Yellow markers are trade decisions. Select one, or a decision time in the Trading log, to open it in the Timeline.'));
    panel.appendChild(renderComparison(data));
    return panel;
  }
  window.AdminTracePerformance = { series, columns, render, rangeStart };
})();
