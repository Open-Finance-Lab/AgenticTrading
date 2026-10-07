/** /admin Agent execution traces: a read-only list and ordered run timeline. */
(function () {
  'use strict';

  const SURFACE = 'traces';
  const state = {
    traceId: null, trace: null, events: [], nextSequence: 0,
    pollTimer: null, stale: false, listItems: [], listHasMore: false, listNextCursor: 0,
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
    const list = node('ol', 'trace-timeline');
    const items = Array.isArray(events) ? events : [];
    if (!items.length) {
      list.appendChild(node('li', 'panel-empty', 'No events recorded yet.'));
      return list;
    }
    items.forEach((event) => {
      const item = node('li', 'trace-event');
      const heading = node('div', 'trace-event-head');
      heading.appendChild(node('span', 'trace-sequence', `#${value(event.sequence_no)}`));
      heading.appendChild(node('strong', '', shell().humanize(event.event_type)));
      heading.appendChild(node('time', '', shell().formatTimestamp(event.occurred_at)));
      item.appendChild(heading);
      const meta = [event.actor_type, event.step_id, event.decision_id, event.artifact_id, event.parent_event_id ? `parent ${event.parent_event_id}` : null]
        .filter((part) => part !== null && part !== undefined && part !== '')
        .map((part) => String(part));
      if (meta.length) item.appendChild(node('p', 'trace-event-meta', meta.join(' · ')));
      const payload = node('pre', 'trace-event-payload', payloadText(event.payload));
      item.appendChild(payload);
      list.appendChild(item);
    });
    return list;
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

  function renderTraceDetail(trace, events) {
    const section = node('section', 'trace-detail-panel');
    const breadcrumb = node('nav', 'breadcrumb');
    breadcrumb.setAttribute('aria-label', 'Breadcrumb');
    const back = node('a', '', 'Agent traces');
    back.setAttribute('href', '#traces');
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

  async function loadDetail(traceId) {
    stopPolling();
    const seq = shell().nextSeq(SURFACE);
    try {
      let trace = await shell().request(`/api/admin/traces/${encodeURIComponent(traceId)}`);
      const events = await loadAllEvents(traceId);
      const terminal = [...events].reverse().find((event) => ['run_completed', 'run_failed'].includes(event.event_type));
      if (terminal) trace = await refreshTraceEnvelope(traceId);
      if (!shell().isCurrent(SURFACE, seq)) return;
      state.traceId = traceId;
      state.trace = trace;
      state.events = events;
      state.nextSequence = state.events.length ? state.events[state.events.length - 1].sequence_no : 0;
      state.stale = false;
      const host = element('tracesView');
      if (!host) return;
      shell().clear(host);
      host.appendChild(renderTraceDetail(trace, state.events));
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

  async function pollEvents() {
    if (!state.traceId || !state.trace || state.trace.status !== 'running') return;
    try {
      const fresh = await loadAllEvents(state.traceId, state.nextSequence);
      if (fresh.length) {
        state.events = state.events.concat(fresh.filter((event) => event.sequence_no > state.nextSequence));
        state.nextSequence = state.events.length ? state.events[state.events.length - 1].sequence_no : state.nextSequence;
        const terminal = [...state.events].reverse().find((event) => ['run_completed', 'run_failed'].includes(event.event_type));
        if (terminal) {
          state.trace = await refreshTraceEnvelope(state.traceId);
        }
        const host = element('tracesView');
        if (host && state.trace) {
          shell().clear(host);
          host.appendChild(renderTraceDetail(state.trace, state.events));
        }
      }
      state.stale = false;
      if (state.trace.status === 'running') state.pollTimer = setTimeout(pollEvents, 3000);
    } catch (error) {
      if (await shell().handleAccessLost(error)) return;
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
    if (event.detail.route !== SURFACE) return;
    if (event.detail.id) loadDetail(event.detail.id);
    else loadList();
  }

  document.addEventListener('admin:route', onRoute);
  window.AdminTraces = {
    renderTraceRows, renderEventTimeline, renderTraceList, renderTraceDetail,
    loadAllEvents, pollEvents, stopPolling,
  };
})();
