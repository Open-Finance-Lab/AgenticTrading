/** /admin Agent execution traces: a read-only list and ordered run timeline. */
(function () {
  'use strict';

  const SURFACE = 'traces';
  const TERMINAL_EVENTS = ['run_completed', 'run_failed'];
  const TRADING_LOG_LIMIT = 50;
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
    const items = Array.isArray(events) ? events : [];
    const columns = node('div', 'trace-three-column');
    const centerColumn = node('div', 'trace-analysis-column');
    if (window.AdminTracePerformance) centerColumn.appendChild(window.AdminTracePerformance.render(trace, items, performance));
    const title = node('h2', 'trace-section-title', 'Timeline');
    centerColumn.appendChild(title);
    centerColumn.appendChild(renderEventTimeline(items));
    columns.append(renderContextRail(trace, items), centerColumn, renderTradingLog(items));
    section.appendChild(columns);
    return section;
  }

  // A <details> whose open state survives the poll's re-render: capture()
  // records every [data-trace-expand] before the old tree is cleared.
  function disclosure(className, key, label) {
    const details = node('details', className);
    details.setAttribute('data-trace-expand', key);
    details.open = Boolean(state.openState?.[key]);
    details.appendChild(node('summary', '', label));
    return details;
  }

  function renderContextRail(trace, events) {
    const rail = node('aside', 'trace-context-rail');
    const config = trace?.initial_input || events.find((event) => event.event_type === 'run_started')?.payload?.config_summary || {};
    const overview = disclosure('trace-run-overview', 'run-configuration', 'Run configuration');
    const fields = [
      ['Agent', trace?.agent_id], ['Run', trace?.run_id], ['Model', config.model || config.model_name],
      ['Market / data source', config.market || config.data_source], ['Start date', config.start_date],
      ['End date', config.end_date], ['Initial capital', config.initial_capital ?? config.initial_equity],
    ];
    fields.forEach(([label, v]) => overview.appendChild(node('p', '', `${label}: ${v == null ? 'Not recorded' : String(v)}`)));
    const configRaw = disclosure('', 'run-configuration-json', 'Recorded configuration (JSON)');
    configRaw.appendChild(node('pre', 'trace-event-payload', payloadText(config)));
    overview.appendChild(configRaw);
    rail.appendChild(overview);
    if (!trace?.trace_id) return rail;
    rail.appendChild(renderBacktestReport(trace));
    const downloads = node('div', 'trace-downloads');
    [['json', 'Export JSON'], ['markdown', 'Export Markdown']].forEach(([format, label]) => {
      const link = node('a', 'auth-btn auth-btn-secondary', label);
      link.setAttribute('href', `/api/admin/traces/${encodeURIComponent(trace.trace_id)}/export?format=${format}`);
      link.setAttribute('download', '');
      downloads.appendChild(link);
    });
    downloads.appendChild(node('span', 'muted', 'All recorded events · running traces export a fixed snapshot'));
    rail.appendChild(downloads);
    return rail;
  }

  // One request per trace and status, shared by every re-render: the poll
  // rebuilds this panel, and restoring `open` fires `toggle` on the new
  // element. A status change re-fetches once, because the stored result is
  // written when the run finishes.
  function loadBacktestReport(trace) {
    const key = `${trace.trace_id}:${trace.status}`;
    if (state.report?.key !== key) {
      const promise = shell().request(`/api/admin/traces/${encodeURIComponent(trace.trace_id)}/backtest`);
      state.report = { key, promise };
      promise.catch(() => {
        if (state.report?.promise === promise) state.report = null;
      });
    }
    return state.report.promise;
  }

  function renderBacktestReport(trace) {
    const report = disclosure('trace-backtest-report', 'backtest-report', 'View this backtest · admin read-only');
    const body = node('div');
    report.appendChild(body);
    let shown = false;
    report.addEventListener('toggle', async () => {
      if (!report.open || shown) return;
      shown = true;
      body.textContent = 'Loading backtest…';
      try {
        const data = await loadBacktestReport(trace);
        shell().clear(body);
        if (!data?.available) {
          body.textContent = 'No stored backtest result for this exact run. Trace events remain available.';
          return;
        }
        Object.entries(data.run || {}).forEach(([key, v]) => body.appendChild(node('p', '', `${shell().humanize(key)}: ${v == null ? 'Not recorded' : String(v)}`)));
        const raw = disclosure('', 'backtest-configuration-json', 'Recorded backtest configuration');
        raw.appendChild(node('pre', 'trace-event-payload', payloadText(data.configuration)));
        body.appendChild(raw);
      } catch (error) {
        if (await shell().handleAccessLost(error)) return;
        shown = false;
        body.textContent = 'Backtest details unavailable. Close and reopen to retry.';
      }
    });
    return report;
  }

  // Fills carry no decision_id; their execution_result event does, and the
  // decision's reasoning is per action, so a fill takes the reason of the
  // action on its symbol (and side, when both name one).
  function fillReason(decision, fill) {
    const payload = decision?.payload || {};
    const actions = Array.isArray(payload.actions) ? payload.actions : [];
    const reasons = Array.isArray(payload.reasoning_summaries) ? payload.reasoning_summaries : [];
    const side = String(fill.side || '').toUpperCase();
    const onSymbol = (action) => action && action.symbol === fill.symbol;
    let index = actions.findIndex((action) => onSymbol(action) && String(action.action || action.side || '').toUpperCase() === side);
    if (index < 0) index = actions.findIndex(onSymbol);
    const matched = index < 0 ? null : (reasons[index] || actions[index].reason || actions[index].reasoning);
    if (typeof matched === 'string' && matched) return matched;
    const distinct = [...new Set(reasons.filter((reason) => typeof reason === 'string' && reason))];
    return distinct.length === 1 ? distinct[0] : null;
  }

  function renderTradingLog(events) {
    const rail = node('aside', 'trace-trading-log');
    const decisions = new Map(events
      .filter((event) => event.event_type === 'decision_recorded' && event.decision_id)
      .map((event) => [event.decision_id, event]));
    const fills = events.flatMap((event) => (
      event.event_type === 'execution_result' && Array.isArray(event.payload?.fills)
        ? event.payload.fills.map((fill) => ({ fill, decision: decisions.get(event.decision_id) }))
        : []
    ));
    // Newest first, so a running trace shows its latest fills without scrolling.
    const shown = fills.slice(-TRADING_LOG_LIMIT).reverse();
    const count = `${fills.length} fill${fills.length === 1 ? '' : 's'}`;
    const head = node('div', 'trace-log-head');
    head.append(node('h2', '', 'Trading log'), node('span', 'eyebrow', shown.length < fills.length ? `Latest ${shown.length} of ${count}` : count));
    rail.appendChild(head);
    if (!fills.length) rail.appendChild(node('p', 'panel-empty', 'No execution fills recorded.'));
    shown.forEach(({ fill, decision }) => {
      const item = node('article', 'trace-log-item');
      const side = String(fill.side || fill.action || 'ORDER').toUpperCase();
      item.append(
        node('div', 'trace-log-order', `${side} ${value(fill.symbol || 'Unknown symbol')}`),
        node('p', 'muted', `${fill.quantity ?? fill.shares ?? '—'} shares · ${fill.price == null ? 'Price not recorded' : fill.price}`),
      );
      const reason = fillReason(decision, fill);
      if (reason) item.appendChild(node('p', 'trace-reason', reason));
      rail.appendChild(item);
    });
    return rail;
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
      // Envelope first: the store appends a terminal event before it marks the
      // trace terminal, so events read after a terminal envelope are complete.
      let trace = await refreshTraceEnvelope(traceId);
      const [events, performance] = await Promise.all([loadAllEvents(traceId), loadPerformance(traceId)]);
      if (trace.status === 'running' && hasTerminalEvent(events)) trace = await refreshTraceEnvelope(traceId);
      if (!shell().isCurrent(SURFACE, seq)) return;
      if (state.traceId !== traceId) {
        state.openState = {};
        state.report = null;
      }
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

  function hasTerminalEvent(events) {
    return events.some((event) => TERMINAL_EVENTS.includes(event.event_type));
  }

  function performanceSignature(performance) {
    return JSON.stringify([
      performance?.point_count, performance?.unavailable, performance?.partial,
      performance?.metrics, performance?.points?.[performance.points.length - 1],
    ]);
  }

  async function pollEvents() {
    if (!state.traceId || !state.trace || state.trace.status !== 'running') return;
    const traceId = state.traceId;
    const seq = shell().nextSeq(SURFACE);
    try {
      // Envelope before events (see loadDetail): a trace that turns terminal
      // between the two reads is caught by the next poll, never by stopping
      // with its last flush and terminal event unread.
      const envelope = await refreshTraceEnvelope(traceId);
      const [fresh, performance] = await Promise.all([
        loadAllEvents(traceId, state.nextSequence),
        loadPerformance(traceId, state.performance),
      ]);
      if (!shell().isCurrent(SURFACE, seq) || state.traceId !== traceId) return;
      const added = fresh.filter((event) => event.sequence_no > state.nextSequence);
      const changed = added.length > 0
        || state.stale
        || envelope.status !== state.trace.status
        || envelope.ended_at !== state.trace.ended_at
        || performanceSignature(performance) !== performanceSignature(state.performance);
      state.trace = envelope;
      state.performance = performance;
      state.events = state.events.concat(added);
      state.nextSequence = state.events.length ? state.events[state.events.length - 1].sequence_no : state.nextSequence;
      const host = element('tracesView');
      // Re-render only on a change: a rebuild drops focus, selection and any
      // disclosure capture() cannot see, every three seconds otherwise.
      if (changed && host) {
        state.openState = window.AdminTraceTimeline.capture(host);
        const focusedId = document.activeElement?.id;
        shell().clear(host);
        host.appendChild(renderTraceDetail(state.trace, state.events, state.context, state.performance));
        if (focusedId) document.getElementById(focusedId)?.focus?.({ preventScroll: true });
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
