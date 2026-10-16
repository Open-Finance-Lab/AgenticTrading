/** /admin Agent execution traces: a read-only list and a Backtest-shaped run detail. */
(function () {
  'use strict';

  const SURFACE = 'traces';
  const TERMINAL_EVENTS = ['run_completed', 'run_failed'];
  const LOG_FILTERS = [['all', 'All orders'], ['buy', 'Buys only'], ['sell', 'Sells only'], ['rejected', 'Rejected']];
  const state = {
    traceId: null, trace: null, events: [], nextSequence: 0,
    pollTimer: null, stale: false, context: {}, listItems: [], listHasMore: false, listNextCursor: 0,
    timelineFilter: 'all', logFilter: 'all', chartView: { range: 'ALL', hidden: new Set() }, chartData: null,
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
    return window.AdminTraceTimeline.render(events, state.openState || {}, {
      filter: state.timelineFilter, onShowOrders: revealOrders,
    });
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

  // Where Back goes: the user's analytics page when the trace was opened from
  // it, otherwise the trace list.
  function backTarget(context) {
    return context?.user_id
      ? { href: `#users/${encodeURIComponent(String(context.user_id))}`, label: 'User analytics' }
      : { href: '#traces', label: 'Agent traces' };
  }

  function renderTopbar(trace, context) {
    const target = backTarget(context);
    const bar = node('div', 'trace-topbar');
    const back = node('a', 'trace-back', `← ${target.label}`);
    back.setAttribute('href', target.href);
    back.setAttribute('data-trace-back', '');
    const breadcrumb = node('nav', 'breadcrumb');
    breadcrumb.setAttribute('aria-label', 'Breadcrumb');
    const parent = node('a', '', target.label);
    parent.setAttribute('href', target.href);
    const current = node('span', '', value(trace?.trace_id));
    current.setAttribute('aria-current', 'page');
    breadcrumb.append(parent, node('span', '', '/'), current);
    const hint = node('span', 'trace-back-hint', 'Esc to go back');
    hint.setAttribute('aria-hidden', 'true');
    bar.append(back, breadcrumb, hint);
    return bar;
  }

  function renderTraceDetail(trace, events, context = {}, performance = null, chartData = null) {
    const section = node('section', 'trace-detail-panel');
    section.appendChild(renderTopbar(trace, context));
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
    // Backtest's three columns: run config | chart, comparison, timeline | trading log.
    const columns = node('div', 'trace-three-column');
    const centerColumn = node('div', 'trace-analysis-column');
    if (window.AdminTracePerformance) {
      centerColumn.appendChild(window.AdminTracePerformance.render(trace, items, performance, chartData, {
        view: state.chartView, onDecision: (decision) => revealDecision(decision.sequence_no),
      }));
    }
    centerColumn.appendChild(renderTimelinePanel(items));
    columns.append(renderContextRail(trace, items), centerColumn, renderTradingLog(items));
    section.appendChild(columns);
    return section;
  }

  function renderTimelinePanel(items) {
    const panel = node('section', 'trace-timeline-panel');
    panel.setAttribute('aria-labelledby', 'traceTimelineTitle');
    const head = node('div', 'trace-timeline-head');
    const titles = node('div');
    const title = node('h2', 'trace-section-title', 'Timeline');
    title.setAttribute('id', 'traceTimelineTitle');
    titles.append(title, node('p', 'trace-timeline-sub', 'Every recorded step: setup, holds, trades and failures'));
    const filter = node('div', 'trace-filter');
    filter.setAttribute('role', 'group');
    filter.setAttribute('aria-label', 'Filter timeline');
    const list = node('div', 'trace-timeline-host');
    const paint = () => {
      shell().clear(list);
      list.appendChild(renderEventTimeline(items));
    };
    [['all', 'All'], ['errors', 'Errors']].forEach(([key, label]) => {
      const button = node('button', '', label);
      button.setAttribute('type', 'button');
      button.setAttribute('data-trace-filter', key);
      button.setAttribute('aria-pressed', String(state.timelineFilter === key));
      button.addEventListener('click', () => {
        state.openState = { ...(state.openState || {}), ...window.AdminTraceTimeline.capture(list) };
        state.timelineFilter = key;
        Array.from(filter.children).forEach((other) => other.setAttribute('aria-pressed', String(other.getAttribute('data-trace-filter') === key)));
        paint();
      });
      filter.appendChild(button);
    });
    const rows = window.AdminTraceTimeline.rows(items);
    const failed = rows.filter((row) => row.failed).length;
    head.append(titles, filter, node('span', 'eyebrow', `${items.length} events${failed ? ` · ${failed} failed` : ''}`));
    panel.appendChild(head);
    paint();
    panel.appendChild(list);
    return panel;
  }

  // Timeline → Trading log: the decision's order group, scrolled into the
  // log's own scroller and marked. The filter resets first so it is there.
  function revealOrders(decisionId) {
    if (state.logFilter !== 'all') {
      state.logFilter = 'all';
      const log = document.querySelector('.trace-trading-log');
      if (log?.parentNode) log.parentNode.replaceChild(renderTradingLog(state.events), log);
    }
    const group = document.getElementById(`trace-orders-${decisionId}`);
    if (!group) return;
    group.scrollIntoView?.({ block: 'nearest', behavior: 'smooth' });
    group.querySelector?.('.trace-log-when')?.focus?.({ preventScroll: true });
    window.AdminTraceTimeline.flash(group);
  }

  // Trading log / chart → Timeline: switch off the Errors filter when it
  // hides the row, then open and mark it.
  function revealDecision(sequence) {
    if (window.AdminTraceTimeline.reveal(sequence)) return;
    if (state.timelineFilter === 'all') return;
    document.querySelector('[data-trace-filter="all"]')?.click();
    window.AdminTraceTimeline.reveal(sequence);
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

  function configRow(label, text, secondary) {
    const row = node('div', 'trace-config-row');
    const dd = node('dd', '', text == null || text === '' ? 'Not recorded' : String(text));
    if (secondary) dd.appendChild(node('span', 'trace-config-secondary', String(secondary)));
    row.append(node('dt', '', label), dd);
    return row;
  }

  function capitalText(v) {
    const numeric = Number(v);
    if (v == null || v === '' || !Number.isFinite(numeric)) return v;
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(numeric);
  }

  // Backtest's Run config list: the recorded fields shown, not folded away.
  function renderContextRail(trace, events) {
    const rail = node('aside', 'trace-context-rail');
    const config = trace?.initial_input || events.find((event) => event.event_type === 'run_started')?.payload?.config_summary || {};
    rail.appendChild(node('h3', 'trace-rail-title', 'Run config'));
    const list = node('dl', 'trace-config-list');
    const range = config.start_date || config.end_date ? `${config.start_date || '?'} → ${config.end_date || '?'}` : null;
    list.append(
      configRow('Agent', trace?.agent_id),
      configRow('Model', config.model || config.model_name),
      configRow('Market', config.market || config.data_source, config.universe),
      configRow('Date range', range),
      configRow('Initial capital', capitalText(config.initial_capital ?? config.initial_equity)),
    );
    rail.appendChild(list);
    const configRaw = disclosure('trace-config-details', 'run-configuration-json', 'Recorded configuration (JSON)');
    configRaw.appendChild(node('pre', 'trace-event-payload', payloadText(config)));
    rail.appendChild(configRaw);
    if (!trace?.trace_id) return rail;
    rail.appendChild(renderBacktestReport(trace));
    const meta = node('div', 'trace-config-meta');
    const started = node('span', '', 'Started');
    started.appendChild(node('strong', '', shell().formatTimestamp(trace.started_at)));
    const run = node('span', '', 'Run');
    run.appendChild(node('strong', '', value(trace.run_id)));
    meta.append(started, run);
    rail.appendChild(meta);
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

  // Orders carry no reasoning of their own: the decision records one summary
  // per action, so an order takes the reason of the action on its symbol (and
  // side, when both name one).
  function orderReason(decision, order) {
    const payload = decision?.payload || {};
    const actions = Array.isArray(payload.actions) ? payload.actions : [];
    const reasons = Array.isArray(payload.reasoning_summaries) ? payload.reasoning_summaries : [];
    const side = String(order.side || order.action || '').toUpperCase();
    const onSymbol = (action) => action && action.symbol === order.symbol;
    let index = actions.findIndex((action) => onSymbol(action) && String(action.action || action.side || '').toUpperCase() === side);
    if (index < 0) index = actions.findIndex(onSymbol);
    const matched = index < 0 ? null : (reasons[index] || actions[index].reason || actions[index].reasoning);
    if (typeof matched === 'string' && matched) return matched;
    const distinct = [...new Set(reasons.filter((reason) => typeof reason === 'string' && reason))];
    return distinct.length === 1 ? distinct[0] : null;
  }

  // One group per decision that produced orders, newest first. Fills carry no
  // decision_id; their execution_result event does.
  function orderGroups(events) {
    const decisions = new Map(events
      .filter((event) => event.event_type === 'decision_recorded' && event.decision_id)
      .map((event) => [event.decision_id, event]));
    return events
      .filter((event) => event.event_type === 'execution_result')
      .map((event) => ({
        decisionId: event.decision_id,
        decision: decisions.get(event.decision_id),
        orders: [
          ...(Array.isArray(event.payload?.fills) ? event.payload.fills : []).map((order) => ({ ...order, status: 'filled' })),
          ...(Array.isArray(event.payload?.rejected) ? event.payload.rejected : []).map((order) => ({ ...order, status: 'rejected' })),
        ],
      }))
      .filter((group) => group.orders.length)
      .reverse();
  }

  function keepOrder(order) {
    if (state.logFilter === 'all') return true;
    if (state.logFilter === 'rejected') return order.status === 'rejected';
    return order.status === 'filled' && String(order.side || order.action || '').toLowerCase() === state.logFilter;
  }

  function orderLine(order) {
    const side = String(order.side || order.action || 'ORDER').toUpperCase();
    const line = node('div', `trace-log-line trace-log-line--${order.status}`);
    const shares = order.quantity ?? order.shares ?? order.requested_shares;
    const amount = order.status === 'filled'
      ? `${shares ?? shell().DASH} × ${order.price == null ? 'price not recorded' : `$${Number(order.price).toFixed(2)}`}`
      : `${shares ?? shell().DASH} sh`;
    line.append(
      node('span', `trace-side-badge trace-side-badge--${side.toLowerCase()}`, side),
      node('span', 'trace-sym', value(order.symbol || 'Unknown symbol')),
      node('span', 'trace-log-amount', amount),
      node('span', `trace-status trace-status--${order.status}`, order.status === 'filled' ? 'Filled' : 'Rejected'),
    );
    return line;
  }

  function renderTradingLog(events) {
    const rail = node('aside', 'trace-trading-log');
    const groups = orderGroups(events);
    const all = groups.flatMap((group) => group.orders);
    const filled = all.filter((order) => order.status === 'filled').length;
    const head = node('div', 'trace-log-head');
    const titles = node('div');
    titles.append(node('h2', '', 'Trading log'), node('p', 'trace-log-summary', `${all.length} order${all.length === 1 ? '' : 's'} · ${filled} filled · ${all.length - filled} rejected`));
    const label = node('label', 'sr-only', 'Filter Trading log orders');
    label.setAttribute('for', 'traceLogFilter');
    const select = node('select', 'trace-log-filter');
    select.setAttribute('id', 'traceLogFilter');
    LOG_FILTERS.forEach(([key, text]) => {
      const option = node('option', '', text);
      option.setAttribute('value', key);
      if (key === state.logFilter) option.setAttribute('selected', '');
      select.appendChild(option);
    });
    select.value = state.logFilter;
    select.addEventListener('change', () => {
      state.logFilter = select.value;
      const fresh = renderTradingLog(state.events);
      rail.parentNode?.replaceChild(fresh, rail);
      document.getElementById('traceLogFilter')?.focus?.();
    });
    head.append(titles, label, select);
    rail.appendChild(head);
    const scroll = node('div', 'trace-log-scroll');
    scroll.setAttribute('tabindex', '0');
    scroll.setAttribute('role', 'list');
    scroll.setAttribute('aria-label', 'Trading log orders');
    let shownGroups = 0;
    groups.forEach(({ decisionId, decision, orders }) => {
      const shown = orders.filter(keepOrder);
      if (!shown.length) return;
      shownGroups += 1;
      const group = node('section', 'trace-log-group');
      group.setAttribute('role', 'listitem');
      if (decisionId) group.setAttribute('id', `trace-orders-${decisionId}`);
      const at = decision?.payload?.decision_at;
      const when = node('button', 'trace-log-when', at ? shell().formatTimestamp(at) : 'Decision time not recorded');
      when.setAttribute('type', 'button');
      when.setAttribute('aria-label', `Open the decision${at ? ` at ${shell().formatTimestamp(at)}` : ''} in the timeline`);
      if (decision) when.addEventListener('click', () => revealDecision(decision.sequence_no));
      else when.disabled = true;
      group.appendChild(when);
      // A reason shared by every order prints once, above them; otherwise
      // each order carries its own.
      const reasons = shown.map((order) => orderReason(decision, order));
      const shared = new Set(reasons).size === 1 ? reasons[0] : null;
      if (shared) group.appendChild(node('p', 'trace-log-reason', shared));
      shown.forEach((order, index) => {
        group.appendChild(orderLine(order));
        if (order.status === 'rejected' && order.reason) group.appendChild(node('p', 'trace-log-reject', `Rejected: ${shell().humanize(order.reason)}`));
        if (!shared && reasons[index]) group.appendChild(node('p', 'trace-log-reason trace-log-reason--order', reasons[index]));
      });
      scroll.appendChild(group);
    });
    if (!all.length) scroll.appendChild(node('p', 'panel-empty', 'No orders recorded.'));
    else if (!shownGroups) scroll.appendChild(node('p', 'panel-empty', 'No orders match this filter.'));
    rail.appendChild(scroll);
    return rail;
  }

  // Esc returns to the parent page, except from a form field or an open
  // dialog, where Esc already means something.
  function onKeydown(event) {
    if (event.key !== 'Escape' || event.defaultPrevented || !state.traceId) return;
    const target = event.target;
    if (target && typeof target.closest === 'function' && target.closest('input, textarea, select, [contenteditable], dialog[open]')) return;
    const back = document.querySelector('[data-trace-back]');
    if (!back) return;
    event.preventDefault();
    window.location.hash = back.getAttribute('href');
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
      const [events, performance, chartData] = await Promise.all([loadAllEvents(traceId), loadPerformance(traceId), loadChartData(traceId)]);
      if (trace.status === 'running' && hasTerminalEvent(events)) trace = await refreshTraceEnvelope(traceId);
      if (!shell().isCurrent(SURFACE, seq)) return;
      if (state.traceId !== traceId) {
        state.openState = {};
        state.report = null;
        state.timelineFilter = 'all';
        state.logFilter = 'all';
        state.chartView = { range: 'ALL', hidden: new Set() };
      }
      state.chartData = chartData;
      state.chartStatus = trace.status;
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
      host.appendChild(renderTraceDetail(trace, state.events, context, state.performance, state.chartData));
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

  // The benchmark series are stored with the run, so they are read once per
  // trace and again when it finishes, not on every poll. A failed read keeps
  // the chart on the trace's own equity.
  async function loadChartData(traceId) {
    try { return await shell().request(`/api/admin/traces/${encodeURIComponent(traceId)}/chart-data`); }
    catch (_error) { return null; }
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
      if (envelope.status !== state.chartStatus) {
        state.chartData = await loadChartData(traceId);
        state.chartStatus = envelope.status;
      }
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
        host.appendChild(renderTraceDetail(state.trace, state.events, state.context, state.performance, state.chartData));
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
  document.addEventListener('keydown', onKeydown);
  window.AdminTraces = {
    renderTraceRows, renderEventTimeline, renderTraceList, renderTraceDetail,
    renderTradingLog, loadAllEvents, pollEvents, stopPolling, onKeydown,
  };
})();
