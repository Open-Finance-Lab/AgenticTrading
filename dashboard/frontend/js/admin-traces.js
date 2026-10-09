/** /admin Agent execution traces: a read-only list and ordered run timeline. */
(function () {
  'use strict';

  const SURFACE = 'traces';
  const state = {
    traceId: null, trace: null, events: [], nextSequence: 0,
    pollTimer: null, stale: false, context: {}, listItems: [], listHasMore: false, listNextCursor: 0,
  };

  function shell() {
    return window.AdminShell;
  }

  function element(id) {
    return document.getElementById(id);
  }

  function node(tag, className, text) {
    return shell().el(tag, className, text);
  }

  function value(value) {
    return value === null || value === undefined || value === '' ? shell().DASH : String(value);
  }

  function duration(trace) {
    if (!trace?.started_at || !trace?.ended_at) return shell().DASH;
    const start = Date.parse(trace.started_at);
    const end = Date.parse(trace.ended_at);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return shell().DASH;
    const seconds = (end - start) / 1000;
    return seconds < 1 ? `${Math.round((end - start))} ms` : `${seconds.toFixed(1)} s`;
  }

  function statusBadge(status) {
    const tone = status === 'completed' ? 'good' : status === 'failed' ? 'bad' : 'warn';
    return node('span', `badge ${tone}`, shell().humanize(status || 'unknown'));
  }

  function renderTraceRows(payload) {
    const body = node('tbody');
    const items = Array.isArray(payload?.items) ? payload.items : [];
    if (!items.length) {
      const row = node('tr');
      row.appendChild(node('td', 'panel-empty', 'No traces yet.'));
      row.children[0].setAttribute('colspan', '6');
      body.appendChild(row);
      return body;
    }
    items.forEach((trace) => {
      const row = node('tr');
      const link = node('a', 'trace-link', value(trace.trace_id));
      link.setAttribute('href', `#traces/${encodeURIComponent(trace.trace_id)}`);
      const cells = [
        link,
        node('span', '', value(trace.agent_id)),
        node('span', '', value(trace.run_id)),
        statusBadge(trace.status),
        node('span', '', shell().formatTimestamp(trace.started_at)),
        node('span', '', duration(trace)),
      ];
      cells.forEach((content, index) => {
        const cell = node('td');
        cell.appendChild(content);
        row.appendChild(cell);
        if (index === 0) cell.classList.add('trace-id-cell');
      });
      body.appendChild(row);
    });
    return body;
  }

  function payloadText(payload) {
    if (payload === null || payload === undefined) return shell().DASH;
    try {
      return JSON.stringify(payload, null, 2);
    } catch (_error) {
      return shell().DASH;
    }
  }

  function renderEventTimeline(events) {
    return window.AdminTraceTimeline.render(events, state.openState || {});
  }

  function renderTraceList(payload) {
    const section = node('section', 'trace-list-panel');
    section.appendChild(node('div', 'page-head', null));
    const head = section.children[0];
    const copy = node('div');
    copy.appendChild(node('h1', '', 'Agent traces'));
    copy.appendChild(node('p', 'muted', 'Ordered records of agent runs, decisions, executions, and artifacts.'));
    head.appendChild(copy);
    const table = node('table', 'trace-table');
    const thead = node('thead');
    const header = node('tr');
    ['Trace', 'Agent', 'Run', 'Status', 'Started (UTC)', 'Duration'].forEach((label) => header.appendChild(node('th', '', label)));
    thead.appendChild(header);
    table.append(thead, renderTraceRows(payload));
    const wrap = node('div', 'table-wrap');
    wrap.appendChild(table);
    section.appendChild(wrap);
    if (payload?.has_more) {
      const controls = node('div', 'pager trace-list-pager');
      const loadMore = node('button', 'load-more', 'Load more traces');
      loadMore.setAttribute('type', 'button');
      loadMore.setAttribute('data-trace-list-action', 'load-more');
      controls.appendChild(loadMore);
      section.appendChild(controls);
    }
    return section;
  }

  function renderTraceDetail(trace, events, context = {}, performance = null) {
    const section = node('section', 'trace-detail-panel');
    const breadcrumb = node('nav', 'breadcrumb');
    breadcrumb.setAttribute('aria-label', 'Breadcrumb');
    const back = node('a', '', context.user_id ? 'Back to user analytics' : 'Agent traces');
    back.setAttribute('href', context.user_id ? `#users/${encodeURIComponent(String(context.user_id))}` : '#traces');
    breadcrumb.append(back, node('span', '', '/'), node('span', '', value(trace?.trace_id)));
    section.appendChild(breadcrumb);
    const head = node('div', 'page-head');
    const copy = node('div');
    copy.appendChild(node('h1', '', value(trace?.trace_id)));
    copy.appendChild(node('p', 'muted', `${value(trace?.agent_id)} · ${value(trace?.run_id)}`));
    if (trace?.parent_trace_id) {
      const parent = node('a', 'trace-parent-link', `Parent trace: ${trace.parent_trace_id}`);
      parent.setAttribute('href', `#traces/${encodeURIComponent(trace.parent_trace_id)}`);
      copy.appendChild(parent);
    }
    head.append(copy, statusBadge(trace?.status));
    section.appendChild(head);
    const metrics = node('div', 'trace-metrics');
    [['Status', shell().humanize(trace?.status || 'unknown')], ['Started', shell().formatTimestamp(trace?.started_at)], ['Ended', shell().formatTimestamp(trace?.ended_at)], ['Duration', duration(trace)]].forEach(([label, text]) => {
      const metric = node('div', 'detail-metric');
      metric.append(node('span', '', label), node('strong', '', text));
      metrics.appendChild(metric);
    });
    section.appendChild(metrics);
    const config = trace?.initial_input || events.find(e => e.event_type === 'run_started')?.payload?.config_summary || {};
    const overview = node('details', 'trace-run-overview');
    overview.setAttribute('data-trace-expand', 'run-configuration');
    overview.open = Boolean(state.openState?.['run-configuration']);
    overview.appendChild(node('summary', '', 'Run configuration'));
    const fields = [['Agent', trace?.agent_id], ['Run', trace?.run_id], ['Model', config.model || config.model_name], ['Market / data source', config.market || config.data_source], ['Start date', config.start_date], ['End date', config.end_date], ['Initial capital', config.initial_capital ?? config.initial_equity]];
    fields.forEach(([label, v]) => overview.appendChild(node('p', '', `${label}: ${v == null ? 'Not recorded' : String(v)}`)));
    const configRaw = node('details'); configRaw.appendChild(node('summary', '', 'Recorded configuration (JSON)'));
    configRaw.appendChild(node('pre', 'trace-event-payload', payloadText(config))); overview.appendChild(configRaw);
    section.appendChild(overview);
    const report = node('details', 'trace-backtest-report');
    report.setAttribute('data-trace-expand', 'backtest-report');
    report.appendChild(node('summary', '', 'View this backtest · admin read-only'));
    const reportBody = node('div'); report.appendChild(reportBody);
    let reportLoaded = false;
    report.addEventListener('toggle', async () => {
      if (!report.open || reportLoaded) return;
      reportLoaded = true; reportBody.textContent = 'Loading backtest…';
      try {
        const data = await shell().request(`/api/admin/traces/${encodeURIComponent(trace.trace_id)}/backtest`);
        shell().clear(reportBody);
        if (!data.available) { reportBody.textContent = 'No stored backtest result for this exact run. Trace events remain available.'; return; }
        Object.entries(data.run).forEach(([key, v]) => reportBody.appendChild(node('p', '', `${shell().humanize(key)}: ${v == null ? 'Not recorded' : String(v)}`)));
        const raw = node('details'); raw.appendChild(node('summary', '', 'Recorded backtest configuration'));
        raw.appendChild(node('pre', 'trace-event-payload', payloadText(data.configuration))); reportBody.appendChild(raw);
      } catch (error) {
        if (await shell().handleAccessLost(error)) return;
        reportLoaded = false; reportBody.textContent = 'Backtest details unavailable. Close and reopen to retry.';
      }
    });
    report.open = Boolean(state.openState?.['backtest-report']);
    section.appendChild(report);
    const downloads = node('div', 'trace-downloads');
    [['json', 'Export JSON'], ['markdown', 'Export Markdown']].forEach(([format, label]) => {
      const link = node('a', 'auth-btn auth-btn-secondary', label);
      link.setAttribute('href', `/api/admin/traces/${encodeURIComponent(trace.trace_id)}/export?format=${format}`);
      link.setAttribute('download', ''); downloads.appendChild(link);
    });
    downloads.appendChild(node('span', 'muted', 'All recorded events · running traces export a fixed snapshot'));
    section.appendChild(downloads);
    if (window.AdminTracePerformance) section.appendChild(window.AdminTracePerformance.render(trace, events, performance));
    const title = node('h2', 'trace-section-title', 'Timeline');
    section.appendChild(title);
    section.appendChild(renderEventTimeline(events));
    return section;
  }

  function renderStaleNotice(host, message) {
    const notice = node('p', 'trace-stale panel-status', message);
    host.appendChild(notice);
  }

  function setMessage(message, error = false) {
    const host = element('tracesView');
    if (!host) return;
    shell().clear(host);
    const item = node('p', error ? 'panel-error' : 'panel-empty', message);
    item.textContent = String(message);
    host.appendChild(item);
  }

  async function loadList({ append = false } = {}) {
    stopPolling();
    const seq = shell().nextSeq(SURFACE);
    try {
      const cursor = append ? state.listNextCursor : 0;
      const payload = await shell().request(`/api/admin/traces?cursor=${cursor}&limit=50`);
      if (!shell().isCurrent(SURFACE, seq)) return;
      const page = Array.isArray(payload?.items) ? payload.items : [];
      state.listItems = append ? state.listItems.concat(page) : page;
      state.listHasMore = Boolean(payload?.has_more);
      state.listNextCursor = Number(payload?.next_cursor) || 0;
      const host = element('tracesView');
      if (!host) return;
      shell().clear(host);
      host.appendChild(renderTraceList({
        items: state.listItems,
        has_more: state.listHasMore,
        next_cursor: state.listNextCursor,
      }));
      const loadMore = host.querySelector('[data-trace-list-action="load-more"]');
      if (loadMore) loadMore.addEventListener('click', () => loadList({ append: true }));
    } catch (error) {
      if (await shell().handleAccessLost(error)) return;
      if (!shell().isCurrent(SURFACE, seq)) return;
      setMessage(error.message || 'Agent traces could not be loaded.', true);
    }
  }

  async function loadAllEvents(traceId, afterSequence = 0) {
    const events = [];
    let cursor = Number(afterSequence) || 0;
    while (true) {
      const payload = await shell().request(
        `/api/admin/traces/${encodeURIComponent(traceId)}/events?after_sequence=${cursor}&limit=100`,
      );
      const page = Array.isArray(payload?.items) ? payload.items : [];
      events.push(...page);
      if (!payload?.has_more || !page.length) break;
      const next = Number(page[page.length - 1].sequence_no);
      if (!Number.isFinite(next) || next <= cursor) break;
      cursor = next;
    }
    return events;
  }

  async function refreshTraceEnvelope(traceId) {
    return shell().request(`/api/admin/traces/${encodeURIComponent(traceId)}`);
  }

  async function loadDetail(traceId, context = {}) {
    stopPolling();
    const seq = shell().nextSeq(SURFACE);
    try {
      let trace = await shell().request(`/api/admin/traces/${encodeURIComponent(traceId)}`);
      const events = await loadAllEvents(traceId);
      const performance = await loadPerformance(traceId);
      const terminal = [...events].reverse().find((event) => ['run_completed', 'run_failed'].includes(event.event_type));
      if (terminal) trace = await refreshTraceEnvelope(traceId);
      if (!shell().isCurrent(SURFACE, seq)) return;
      if (state.traceId !== traceId) state.openState = {};
      state.traceId = traceId;
      state.context = { ...context };
      state.trace = trace;
      state.performance = performance;
      state.events = events;
      state.nextSequence = state.events.length ? state.events[state.events.length - 1].sequence_no : 0;
      state.stale = false;
      const host = element('tracesView');
      if (!host) return;
      shell().clear(host);
      host.appendChild(renderTraceDetail(trace, state.events, context, state.performance));
      if (trace.status === 'running') startPolling();
    } catch (error) {
      if (await shell().handleAccessLost(error)) return;
      if (!shell().isCurrent(SURFACE, seq)) return;
      setMessage(error.message || 'Agent trace could not be loaded.', true);
    }
  }

  function stopPolling() {
    if (state.pollTimer !== null) {
      clearTimeout(state.pollTimer);
      state.pollTimer = null;
    }
  }

  async function loadPerformance(traceId, previous = null) {
    try { return await shell().request(`/api/admin/traces/${encodeURIComponent(traceId)}/performance`); }
    catch (_error) { return { ...(previous || {}), unavailable: true }; }
  }

  async function pollEvents() {
    if (!state.traceId || !state.trace || state.trace.status !== 'running') return;
    const traceId = state.traceId;
    const seq = shell().nextSeq(SURFACE);
    try {
      const fresh = await loadAllEvents(traceId, state.nextSequence);
      const performance = await loadPerformance(traceId, state.performance);
      const envelope = await refreshTraceEnvelope(traceId);
      if (!shell().isCurrent(SURFACE, seq) || state.traceId !== traceId) return;
      state.trace = envelope;
      state.performance = performance;
      {
        state.events = state.events.concat(fresh.filter((event) => event.sequence_no > state.nextSequence));
        state.nextSequence = state.events.length ? state.events[state.events.length - 1].sequence_no : state.nextSequence;

        const host = element('tracesView');
        if (host && state.trace) {
          state.openState = window.AdminTraceTimeline.capture(host);
          shell().clear(host);
          host.appendChild(renderTraceDetail(state.trace, state.events, state.context, state.performance));
        }
      }
      state.stale = false;
      if (state.trace.status === 'running') state.pollTimer = setTimeout(pollEvents, 3000);
    } catch (error) {
      if (await shell().handleAccessLost(error)) return;
      if (!shell().isCurrent(SURFACE, seq) || state.traceId !== traceId) return;
      state.stale = true;
      const host = element('tracesView');
      if (host) renderStaleNotice(host, 'Showing recorded events; refresh failed.');
      state.pollTimer = setTimeout(pollEvents, 5000);
    }
  }

  function startPolling() {
    stopPolling();
    state.pollTimer = setTimeout(pollEvents, 3000);
  }

  function onRoute(event) {
    if (event.detail.route !== SURFACE) {
      stopPolling();
      shell().nextSeq(SURFACE);
      state.traceId = null;
      return;
    }
    if (event.detail.id) loadDetail(event.detail.id, event.detail.query || {});
    else loadList();
  }

  document.addEventListener('admin:route', onRoute);
  window.AdminTraces = {
    renderTraceRows, renderEventTimeline, renderTraceList, renderTraceDetail,
    loadAllEvents, pollEvents, stopPolling,
  };
})();
