/** Read-only equity context for one trace; market time is not event wall time. */
(function () {
  'use strict';
  const svgNS = 'http://www.w3.org/2000/svg';
  function series(events, performance) {
    const snapshots = (events || []).filter(e => e.event_type === 'decision_recorded')
      .map(e => ({ timestamp: e.payload?.decision_at, equity: e.payload?.state?.equity, event: e }));
    const raw = performance?.points?.length ? performance.points : snapshots;
    const points = raw.filter(p => typeof p.equity === 'number' && Number.isFinite(p.equity) && Number.isFinite(Date.parse(p.timestamp)))
      .map(p => ({ ...p, time: Date.parse(p.timestamp) })).sort((a, b) => a.time - b.time);
    return { points, snapshots: raw === snapshots };
  }
  function render(trace, events, performance) {
    const el = window.AdminShell.el;
    const panel = el('section', 'trace-performance');
    panel.appendChild(el('h2', '', 'Trading performance'));
    const { points, snapshots } = series(events, performance);
    panel.appendChild(el('p', 'muted', snapshots
      ? 'Decision equity snapshots · market time (UTC). Sampled observations, not the complete backtest result.'
      : 'Run equity · market time (UTC). One observed outcome, not expected return.'));
    if (trace.status !== 'completed') panel.appendChild(el('p', 'trace-performance-notice',
      trace.status === 'failed' ? 'Run failed. Recorded performance is incomplete; inspect the failure in the timeline.' : 'Run in progress. Recorded performance is incomplete.'));
    if (performance?.unavailable) panel.appendChild(el('p', 'muted', 'Performance refresh unavailable. Showing recorded snapshots or previously loaded data.'));
    if (points.length < 2) {
      panel.appendChild(el('p', 'panel-empty', points.length ? 'Only one equity observation is available; a curve requires two.' : 'No equity observations recorded for this run.'));
      return panel;
    }
    let peak = points[0].equity, dd = 0;
    points.forEach(p => { peak = Math.max(peak, p.equity); if (peak > 0) dd = Math.max(dd, (peak - p.equity) / peak * 100); });
    const ret = snapshots ? (points[0].equity > 0 ? (points.at(-1).equity / points[0].equity - 1) * 100 : null) : performance.metrics?.return_pct;
    const drawdown = snapshots ? dd : performance.metrics?.max_drawdown_pct;
    const pct = v => typeof v === 'number' && Number.isFinite(v) ? `${v.toFixed(2)}%` : '—';
    panel.appendChild(el('p', '', `${snapshots ? 'Snapshot change' : 'Return'}: ${pct(ret)} · ${snapshots ? 'Observed' : 'Maximum'} drawdown: ${pct(drawdown)}`));
    const chart = document.createElementNS(svgNS, 'svg');
    chart.setAttribute('viewBox', '0 0 900 260'); chart.setAttribute('role', 'img');
    chart.setAttribute('aria-label', 'Equity over historical market time');
    const min = points.reduce((v, p) => Math.min(v, p.equity), Infinity), max = points.reduce((v, p) => Math.max(v, p.equity), -Infinity);
    const span = max - min || Math.max(Math.abs(max) * .01, 1);
    const x = t => 75 + (t - points[0].time) / (points.at(-1).time - points[0].time || 1) * 800;
    const y = v => 210 - (v - min) / span * 180;
    function shape(tag, attrs, text) {
      const n = document.createElementNS(svgNS, tag);
      Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, String(v)));
      if (text !== undefined) n.textContent = text;
      chart.appendChild(n); return n;
    }
    [min, min + span / 2, min + span].forEach(v => {
      shape('line', { x1: 75, x2: 875, y1: y(v), y2: y(v), stroke: 'currentColor', opacity: '.15' });
      shape('text', { x: 68, y: y(v) + 4, 'text-anchor': 'end', fill: 'currentColor', 'font-size': 12 }, v.toFixed(2));
    });
    const plotted = points.length > 1000 ? Array.from({length: 1000}, (_, i) => points[Math.round(i * (points.length - 1) / 999)]) : points;
    shape('polyline', { points: plotted.map(p => `${x(p.time)},${y(p.equity)}`).join(' '), fill: 'none', stroke: '#38bdf8', 'stroke-width': 2 });
    [points[0], points.at(-1)].forEach((p, i) => shape('text', { x: i ? 875 : 75, y: 245, 'text-anchor': i ? 'end' : 'start', fill: 'currentColor', 'font-size': 12 }, new Date(p.time).toISOString().slice(0, 16).replace('T', ' ')));
    const decisions = new Map((events || []).filter(e => e.decision_id && e.event_type === 'decision_recorded').map(e => [e.decision_id, e]));
    const links = el('div', 'trace-performance-links');
    (events || []).filter(e => e.event_type === 'execution_result' && e.payload?.fills?.length).forEach(e => {
      const decision = decisions.get(e.decision_id);
      if (!decision) return;
      const time = Date.parse(decision.payload?.decision_at);
      const equity = decision.payload?.state?.equity;
      const label = e.payload.fills.map(f => `${f.side} ${f.symbol}`).join(', ');
      const jump = target => { const n = document.getElementById(`trace-event-${target.sequence_no}`); n?.scrollIntoView?.({block: 'center', behavior: 'smooth'}); n?.focus?.({preventScroll: true}); };
      if (Number.isFinite(time) && typeof equity === 'number' && Number.isFinite(equity) && time >= points[0].time && time <= points.at(-1).time) {
        // A marker indicates the decision snapshot, not equity at fill time.
        const marker = shape('circle', { cx: x(time), cy: Math.max(30, Math.min(210, y(equity))), r: 5, fill: '#fbbf24', tabindex: 0, role: 'button', 'aria-label': `Decision: ${label}` });
        marker.addEventListener('click', () => jump(decision));
        marker.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); jump(decision); } });
      }
      const button = el('button', 'auth-btn auth-btn-secondary', `${label} · ${decision.payload?.decision_at || 'Unknown market time'}`);
      button.setAttribute('type', 'button'); button.addEventListener('click', () => jump(decision)); links.appendChild(button);
      const execution = el('button', 'auth-btn auth-btn-secondary', 'Execution');
      execution.setAttribute('type', 'button'); execution.addEventListener('click', () => jump(e)); links.appendChild(execution);
    });
    panel.appendChild(chart);
    panel.appendChild(el('p', 'muted', 'Trade markers identify decision snapshots. Select a trade to inspect its reasoning or execution.'));
    panel.appendChild(links);
    return panel;
  }
  window.AdminTracePerformance = { series, render };
})();
