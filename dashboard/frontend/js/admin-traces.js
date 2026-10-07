/** /admin Agent execution traces: a read-only list and ordered run timeline. */
(function () {
  'use strict';

  const SURFACE = 'traces';
  const state = { traceId: null, trace: null, events: [], nextSequence: 0, pollTimer: null, stale: false };

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
      const meta = [event.actor_type, event.step_id, event.decision_id, event.artifact_id]
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

  async function loadList() {
    stopPolling();
    const seq = shell().nextSeq(SURFACE);
    try {
      const payload = await shell().request('/api/admin/traces');
      if (!shell().isCurrent(SURFACE, seq)) return;
      const host = element('tracesView');
      if (!host) return;
      shell().clear(host);
      host.appendChild(renderTraceList(payload));
    } catch (error) {
      if (await shell().handleAccessLost(error)) return;
      if (!shell().isCurrent(SURFACE, seq)) return;
      setMessage(error.message || 'Agent traces could not be loaded.', true);
    }
  }

  async function loadDetail(traceId) {
    stopPolling();
    const seq = shell().nextSeq(SURFACE);
    try {
      const [trace, eventPayload] = await Promise.all([
        shell().request(`/api/admin/traces/${encodeURIComponent(traceId)}`),
        shell().request(`/api/admin/traces/${encodeURIComponent(traceId)}/events?after_sequence=0&limit=100`),
      ]);
      if (!shell().isCurrent(SURFACE, seq)) return;
      state.traceId = traceId;
      state.trace = trace;
      state.events = Array.isArray(eventPayload?.items) ? eventPayload.items : [];
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
      const payload = await shell().request(
        `/api/admin/traces/${encodeURIComponent(state.traceId)}/events?after_sequence=${state.nextSequence}&limit=100`,
      );
      const fresh = Array.isArray(payload?.items) ? payload.items : [];
      if (fresh.length) {
        state.events = state.events.concat(fresh.filter((event) => event.sequence_no > state.nextSequence));
        state.nextSequence = state.events.length ? state.events[state.events.length - 1].sequence_no : state.nextSequence;
        const terminal = [...state.events].reverse().find((event) => ['run_completed', 'run_failed'].includes(event.event_type));
        if (terminal) state.trace.status = terminal.event_type === 'run_completed' ? 'completed' : 'failed';
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
  window.AdminTraces = { renderTraceRows, renderEventTimeline, renderTraceList, renderTraceDetail, pollEvents, stopPolling };
})();
