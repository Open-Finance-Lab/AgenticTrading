/** Process-only trace timeline: one row per step, quiet steps folded, raw detail on demand. */
(function () {
  'use strict';
  // Orders, symbols and reasoning belong to the Trading log beside this list;
  // the timeline says how each step ran (tools, timing, failures, raw events).
  const KIND_LABELS = { trade: 'Trade', hold: 'Hold', setup: 'Setup', error: 'Error', run: 'Run' };
  function isFailed(e) {
    return /failed|error/.test(e.event_type || '') || Boolean(e.payload?.error_code) || e.payload?.outcome === 'error';
  }
  function groupEvents(events) {
    const stepDecisions = new Map();
    events.forEach(e => { if (e.step_id && e.decision_id) {
      if (!stepDecisions.has(e.step_id)) stepDecisions.set(e.step_id, new Set());
      stepDecisions.get(e.step_id).add(e.decision_id);
    } });
    const groups = new Map();
    events.forEach(e => {
      const matches = stepDecisions.get(e.step_id);
      const decision = e.decision_id || (matches?.size === 1 ? [...matches][0] : null);
      const key = decision ? `decision:${decision}` : e.step_id ? `step:${e.step_id}` : `event:${e.sequence_no}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(e);
    });
    return [...groups].map(([key, events]) => ({ key, events }));
  }
  // A step trades when its execution has an order (filled or rejected), even
  // if a tool in it failed: the failure is flagged on the row, so neither the
  // trade nor the failure drops out of view under either filter.
  function classify(group) {
    const decision = group.events.find(e => e.event_type === 'decision_recorded');
    const execution = group.events.find(e => e.event_type === 'execution_result');
    const filled = execution?.payload?.fills?.length || 0;
    const rejected = execution?.payload?.rejected?.length || 0;
    const failed = group.events.some(isFailed);
    let kind = 'setup';
    if (decision && filled + rejected) kind = 'trade';
    else if (failed) kind = 'error';
    else if (decision) kind = 'hold';
    else if (/^run_(started|completed)$/.test(group.events[0].event_type)) kind = 'run';
    return { ...group, kind, decision, filled, rejected, failed };
  }
  // Consecutive holds or setup steps fold into one row, so an hourly week of
  // mostly-hold bars stays a short list. The first member's key names the row.
  function rows(events) {
    const out = [];
    groupEvents(Array.isArray(events) ? events : []).map(classify).forEach(group => {
      const prev = out.at(-1);
      if ((group.kind === 'hold' || group.kind === 'setup') && prev?.kind === group.kind) {
        prev.members.push(group); prev.failed = prev.failed || group.failed; return;
      }
      out.push({ key: group.key, kind: group.kind, failed: group.failed, members: [group] });
    });
    return out;
  }
  function span(events) {
    const ms = Date.parse(events.at(-1).occurred_at) - Date.parse(events[0].occurred_at);
    const tools = events.filter(e => e.event_type === 'tool_result').length;
    const parts = [];
    if (tools) parts.push(`${tools} tool call${tools === 1 ? '' : 's'}`);
    if (Number.isFinite(ms)) parts.push(`${(ms / 1000).toFixed(1)}s`);
    return parts.join(' · ');
  }
  // Rows are narrow, so a row shows a short UTC stamp: market time for a
  // decision, recorded time otherwise; the full stamps are in the event lines.
  const SHORT_TIME = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hourCycle: 'h23', timeZone: 'UTC' });
  const CLOCK = new Intl.DateTimeFormat('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23', timeZone: 'UTC' });
  function short(value, format) {
    const date = new Date(value);
    return Number.isFinite(date.getTime()) ? format.format(date).replace(',', '') : window.AdminShell.DASH;
  }
  function rowTime(row) {
    const at = row.members[0].decision?.payload?.decision_at;
    return at ? short(at, SHORT_TIME) : short(row.members[0].events[0].occurred_at, CLOCK);
  }
  function rowSummary(row) {
    const first = row.members[0];
    const shell = window.AdminShell;
    if (row.kind === 'trade') {
      const orders = first.filled + first.rejected;
      return [`${orders} order${orders === 1 ? '' : 's'} (${first.filled} filled${first.rejected ? `, ${first.rejected} rejected` : ''})`, span(first.events)].filter(Boolean).join(' · ');
    }
    if (row.kind === 'hold') {
      const last = row.members.at(-1).decision?.payload?.decision_at;
      return row.members.length > 1 ? `Held · ${row.members.length} bars, through ${short(last, SHORT_TIME)}` : 'Held positions';
    }
    if (row.kind === 'setup') return row.members.flatMap(m => m.events).map(e => e.payload?.tool_name || e.payload?.source || shell.humanize(e.event_type)).join(' · ');
    if (row.kind === 'error') {
      const e = first.events.find(isFailed);
      return `${e.payload?.tool_name ? `${e.payload.tool_name}: ` : ''}${e.payload?.error_code || shell.humanize(e.event_type)}`;
    }
    return shell.humanize(first.events[0].event_type);
  }
  function eventStatus(e) {
    const p = e.payload || {};
    if (isFailed(e)) return p.error_code || 'failed';
    if (p.accepted === false) return 'not accepted';
    if (typeof p.outcome === 'string') return p.outcome;
    return p.accepted === true ? 'accepted' : '';
  }
  function eventDetail(e, openState) {
    const el = window.AdminShell.el;
    const p = e.payload || {};
    const item = el('div', 'trace-event');
    item.setAttribute('id', `trace-event-${e.sequence_no}`); item.setAttribute('tabindex', '-1');
    const line = el('div', 'trace-event-line');
    const what = p.tool_name || p.source;
    line.appendChild(el('span', 'trace-event-seq', `#${e.sequence_no}`));
    line.appendChild(el('span', 'trace-event-type', `${window.AdminShell.humanize(e.event_type)}${what ? ` · ${what}` : ''}`));
    const status = eventStatus(e);
    if (status) line.appendChild(el('span', `trace-event-status${isFailed(e) ? ' is-failed' : ''}`, status));
    const meta = [p.duration_ms != null && Number.isFinite(Number(p.duration_ms)) ? `${Math.round(Number(p.duration_ms))} ms` : '', e.actor_type || 'Unknown actor', `Recorded ${window.AdminShell.formatTimestamp(e.occurred_at)}`];
    line.appendChild(el('span', 'muted', meta.filter(Boolean).join(' · ')));
    item.appendChild(line);
    const raw = el('details', 'trace-technical');
    const rawKey = `raw:${e.sequence_no}`;
    raw.setAttribute('data-trace-expand', rawKey); raw.open = Boolean(openState[rawKey]);
    raw.appendChild(el('summary', '', 'Technical details (JSON)'));
    const json = el('pre', 'trace-event-payload');
    json.textContent = JSON.stringify(e, null, 2);
    raw.appendChild(json);
    item.appendChild(raw);
    return item;
  }
  function render(events, openState = {}, options = {}) {
    const el = window.AdminShell.el;
    const list = el('ol', 'trace-timeline');
    const all = rows(events);
    const shown = options.filter === 'errors' ? all.filter(row => row.failed) : all;
    if (!shown.length) {
      list.appendChild(el('li', 'panel-empty', options.filter === 'errors' && all.length ? 'No failed events.' : 'No events recorded yet.'));
      return list;
    }
    shown.forEach(row => {
      const li = el('li', `trace-row trace-row--${row.kind}`);
      li.setAttribute('data-trace-row', row.key);
      const details = el('details', 'trace-group');
      details.setAttribute('data-trace-expand', row.key);
      details.open = Object.hasOwn(openState, row.key) ? openState[row.key] : row.kind === 'error';
      const heading = el('summary', 'trace-row-summary');
      const count = row.members.reduce((n, m) => n + m.events.length, 0);
      heading.appendChild(el('span', 'trace-row-time', rowTime(row)));
      heading.appendChild(el('span', `trace-kind trace-kind--${row.kind}`, KIND_LABELS[row.kind]));
      const text = el('span', 'trace-row-text');
      if (row.failed && row.kind !== 'error') {
        const failure = row.members.flatMap(m => m.events).find(isFailed);
        text.appendChild(el('span', 'trace-row-flag', `${failure.payload?.tool_name || window.AdminShell.humanize(failure.event_type)} failed`));
        text.appendChild(document.createTextNode(' '));
      }
      text.appendChild(document.createTextNode(rowSummary(row)));
      heading.appendChild(text);
      const countNode = el('span', 'trace-row-count', String(count));
      countNode.setAttribute('title', `${count} events`);
      heading.appendChild(countNode);
      details.appendChild(heading);
      const body = el('div', 'trace-row-body');
      if (row.kind === 'trade' && typeof options.onShowOrders === 'function') {
        const decisionId = row.members[0].decision.decision_id;
        const link = el('button', 'trace-row-orders', 'Show orders in Trading log →');
        link.setAttribute('type', 'button');
        link.addEventListener('click', () => options.onShowOrders(decisionId));
        body.appendChild(link);
      }
      row.members.forEach(m => m.events.forEach(e => body.appendChild(eventDetail(e, openState))));
      details.appendChild(body);
      li.appendChild(details); list.appendChild(li);
    });
    return list;
  }
  function capture(host) {
    const state = {};
    host.querySelectorAll('[data-trace-expand]').forEach(n => { state[n.getAttribute('data-trace-expand')] = n.open; });
    return state;
  }
  function flash(node) {
    if (!node?.classList) return;
    node.classList.remove('is-flash');
    void node.offsetWidth; // restart the animation on a repeat visit
    node.classList.add('is-flash');
  }
  // Opens every disclosure around the event, scrolls its row into view inside
  // the timeline's own scroller, and marks the row so the eye lands on it.
  function reveal(sequence) {
    const target = document.getElementById(`trace-event-${sequence}`);
    if (!target) return false;
    let parent = target.parentElement;
    let row = null;
    while (parent) {
      if (parent.tagName === 'DETAILS') parent.open = true;
      if (!row && parent.hasAttribute?.('data-trace-row')) row = parent;
      parent = parent.parentElement;
    }
    (row || target).scrollIntoView?.({ block: 'nearest', behavior: 'smooth' });
    target.focus?.({ preventScroll: true });
    flash(row);
    return true;
  }
  window.AdminTraceTimeline = { groupEvents, rows, render, capture, reveal, flash, isFailed };
})();
